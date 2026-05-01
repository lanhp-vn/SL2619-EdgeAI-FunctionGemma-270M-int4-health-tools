#!/usr/bin/env python3
"""FunctionGemma 270M-IT SFT — Unsloth + LoRA r=128 + train_on_responses_only.

Why this exists separately from `scripts/finetune.py`:
    The Gemma 3 health-QA path uses vanilla TRL + PEFT + bitsandbytes 4-bit. The
    FunctionGemma path is on the **vendor-blessed Unsloth recipe**
    (`FastLanguageModel` + `train_on_responses_only` + `save_pretrained_gguf`),
    which pins `transformers==4.56.2` / `trl==0.22.2` and replaces the loss
    masking. Mixing the two paths into one script would force every host CI
    install to pull `unsloth` (server-only, ~3 GB), so we keep them parallel.

Source-of-truth: `docs/plans/FunctionGemma/README.md` §10.2 (verbatim
hyperparameters from Unsloth notebook cells 6, 8, 29, 31). §10.3 documents the
RTX 5080 OOM ceiling and fallback ladder; §9.4.2 / §9.6 (rule 7) are the
double-BOS contract this script's dry-run gate enforces.

Deploy:
    scp scripts/finetune_functiongemma.py nouslogic-server:~/functiongemma-finetune/

T1 dry-run gate (host or server — host falls back to a vanilla tokenizer if
Unsloth isn't installed; the gate then validates dataset shape + render + length):
    uv run python scripts/finetune_functiongemma.py --dry-run --max-dry-run-rows 4

Full training run (server only — needs Unsloth + GPU):
    ssh -t nouslogic-server '
      cd ~/functiongemma-finetune &&
      source .venv/bin/activate &&
      python finetune_functiongemma.py
    '

After training: `save_pretrained_merged(merged_16bit)` →
`save_pretrained_gguf(F16, Q8_0)` → `llama-quantize Q4_K_M`. See §10.4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Default refusal-class members for --refusal-loss-weight (Block F1, 2026-05-01).
# Kept module-level so dry-run echo and train path read identical values; the CLI
# flag accepts a comma-separated override but defaults to this set.
DEFAULT_REFUSAL_CATEGORIES: tuple[str, ...] = (
    "off_topic_refusal",
    "medical_advice_refusal",
)

# Unsloth strips logits from forward outputs for memory (since 2024.11). TRL
# 0.22.2's `entropy_from_logits` reads them and crashes with NotImplementedError
# unless this env var is set BEFORE any `unsloth` import. Setting it at module
# load (not inside `_train`) guarantees both train and eval paths see it.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

# `gemma_tools` ships without a py.typed marker today (see pyproject — adding
# one is out of scope for this script). Suppress the missing-stubs warning
# locally so `mypy scripts/finetune_functiongemma.py` runs clean on the host.
from gemma_tools.functiongemma_dataset import (  # type: ignore[import-untyped]
    load_jsonl,
    render_training_text,
    validate_file,
)

# --------------------------------------------------------------------------
# Constants — verbatim from §10.2. Kept as module-level so the dry-run path
# and the train path read identical values; do NOT introduce dual-source
# config files for these (the plan is the contract).
# --------------------------------------------------------------------------

MODEL_ID = "unsloth/functiongemma-270m-it"

# Local fallbacks for dry-run on the host (no Unsloth, possibly no network).
# Order matters: prefer Unsloth's mirror, then Google's, then a local cache.
_HOST_TOKENIZER_FALLBACKS: tuple[str, ...] = (
    MODEL_ID,
    "google/functiongemma-270m-it",
    str(Path.home() / "hf-cache" / "functiongemma-270m-it"),
)

DEFAULT_MAX_SEQ_LENGTH = 4096

LORA_TARGET_MODULES: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "functiongemma" / "dataset_v1"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FunctionGemma 270M-IT SFT (Unsloth + LoRA r=128). See plan §10.2."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="T1 gate: validate splits, render a slice, check lengths, exit 0. "
             "Does NOT load the model (host has no GPU).",
    )
    p.add_argument(
        "--train-file",
        type=Path,
        default=_DEFAULT_DATA_DIR / "train.jsonl",
        help="Training split JSONL (default: data/functiongemma/dataset_v1/train.jsonl)",
    )
    p.add_argument(
        "--val-file",
        type=Path,
        default=_DEFAULT_DATA_DIR / "val.jsonl",
        help="Validation split JSONL",
    )
    p.add_argument(
        "--test-file",
        type=Path,
        default=_DEFAULT_DATA_DIR / "test.jsonl",
        help="Held-out test split JSONL (not consumed by SFTTrainer; kept for symmetry)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs_fg_v1",
        help="SFTTrainer output dir (checkpoints + final adapter)",
    )
    p.add_argument(
        "--logging-dir",
        default="runs",
        help="TensorBoard log dir",
    )
    p.add_argument(
        "--max-dry-run-rows",
        type=int,
        default=8,
        help="Number of rows to render + length-check during --dry-run",
    )
    p.add_argument(
        "--ctx-size",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Alias for max_seq_length (passed to FastLanguageModel.from_pretrained)",
    )
    # Block F1 (2026-05-01): refusal-class loss reweighting. Default 1.0 is a
    # no-op (compute_loss is bit-identical to vanilla SFTTrainer per the
    # equivalence test in tests/test_finetune_functiongemma_weighting.py).
    # Values > 1.0 amplify the per-row loss for refusal categories so the
    # gradient ratio leans back toward refusal behaviour, mitigating the
    # cp-111 → cp-333 medical_advice_refusal collapse documented in
    # docs/bench/2026-05-01_functiongemma-v2-finetune-eval.md §"Failure analysis".
    p.add_argument(
        "--refusal-loss-weight",
        type=float,
        default=1.0,
        help="Per-row loss multiplier for refusal categories (default 1.0 = no-op). "
             "Block F1: 2.0 is the primary candidate; 1.5/3.0 map the curve.",
    )
    p.add_argument(
        "--refusal-categories",
        default=",".join(DEFAULT_REFUSAL_CATEGORIES),
        help=f"Comma-separated category names treated as refusals for "
             f"--refusal-loss-weight. Default: {','.join(DEFAULT_REFUSAL_CATEGORIES)}.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------
# Tokenizer loading — distinct paths for dry-run (host) and train (server).
# --------------------------------------------------------------------------


def _load_tokenizer_for_dry_run() -> tuple[Any | None, str | None]:
    """Best-effort tokenizer load on the host.

    Returns (tokenizer, source_label). On failure returns (None, None) — the
    caller then degrades the dry-run to file-shape validation only.
    """
    # Why prefer Unsloth even on the host: if the user already pre-warmed the
    # cache (e.g. via `unsloth` on a workstation), `FastLanguageModel`'s
    # tokenizer matches what the server will see byte-for-byte.
    try:
        # Lazy: requires `unsloth` (server-only)
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        _, tok = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
            load_in_4bit=False,
            load_in_8bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        return tok, f"unsloth FastLanguageModel ({MODEL_ID})"
    except Exception as exc:
        # Both `unsloth` ImportError and HF weight-fetch errors funnel here —
        # broad catch is intentional; the dry-run must degrade gracefully.
        print(f"note: Unsloth tokenizer unavailable on host ({exc.__class__.__name__}); "
              f"falling back to transformers AutoTokenizer.", file=sys.stderr)

    # Lazy: only used as host-side fallback.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None, None

    last_err: Exception | None = None
    for src in _HOST_TOKENIZER_FALLBACKS:
        try:
            tok = AutoTokenizer.from_pretrained(src)
            return tok, src
        except Exception as exc:
            # Broad catch: network failures, missing local cache, HF gating,
            # tokenizer.json schema drift — all funnel here. The fallback loop
            # tries the next source; we surface the last error if all miss.
            last_err = exc
            continue
    print(f"note: no tokenizer source succeeded ({last_err!r}); skipping render gate.",
          file=sys.stderr)
    return None, None


# --------------------------------------------------------------------------
# Dataset assembly. Pre-renders `text` per §9.4.2; keeps `id` / `category` /
# `notes` out of the rendered string (the chat template ignores them, but
# leaving them on the row would wedge `Dataset.from_list` into a sparse-column
# layout; cleaner to drop them here).
# --------------------------------------------------------------------------


def _build_text_rows(path: Path, tokenizer: Any) -> list[dict[str, str]]:
    """Render each row to its training text and carry the `category` along.

    `category` is consumed by the Block-F1 weighted collator
    (`_WeightedCollator`) to inject `row_weight` into each batch. We keep it as
    an empty string when the JSONL row omits it so the collator falls back to
    weight=1.0 cleanly. Set `remove_unused_columns=False` on `SFTConfig` so the
    column survives `_prepare_dataset` and reaches the data collator.
    """
    rows: list[dict[str, str]] = []
    for raw in load_jsonl(path):
        text = render_training_text(raw, tokenizer)
        rows.append({"text": text, "category": str(raw.get("category") or "")})
    return rows


def _ensure_split_exists(path: Path, label: str) -> None:
    if not path.exists():
        print(
            f"ERROR: {label} split not found at {path}.\n"
            f"       Build it via the dataset-split script "
            f"(see docs/plans/FunctionGemma/README.md §9.7), then re-run.",
            file=sys.stderr,
        )
        sys.exit(2)


# --------------------------------------------------------------------------
# Dry-run path (T1 gate). Runs on host without Unsloth/GPU.
# --------------------------------------------------------------------------


def _dry_run(args: argparse.Namespace) -> None:
    for path, label in (
        (args.train_file, "train"),
        (args.val_file, "val"),
        (args.test_file, "test"),
    ):
        _ensure_split_exists(path, label)

    # File-shape validation runs unconditionally — it does not need a tokenizer.
    print("=== file-shape validation (validate_file, min_pass_rate=0.95) ===")
    for path, label in ((args.train_file, "train"), (args.val_file, "val")):
        report = validate_file(path)
        status = "OK" if report.meets_threshold else "FAIL"
        print(f"  {label:5s} {path.name}: {report.passed}/{report.total} pass "
              f"(rate={report.pass_rate:.4f}) [{status}]")
        if not report.meets_threshold:
            print(f"  failures (first 3): {[f.errors for f in report.failures[:3]]}",
                  file=sys.stderr)
            sys.exit(3)

    tokenizer, src = _load_tokenizer_for_dry_run()
    if tokenizer is None:
        print("\nT1 gate PARTIAL — file-shape OK; render gate skipped (no tokenizer).")
        print("Re-run on a machine with `transformers` + network or a local hf-cache.")
        sys.exit(0)
    print(f"\nloaded tokenizer from `{src}`")

    print(f"\n=== render + length gate (first {args.max_dry_run_rows} train rows) ===")
    for idx, raw in enumerate(load_jsonl(args.train_file)):
        if idx >= args.max_dry_run_rows:
            break
        text = render_training_text(raw, tokenizer)

        # G_DATASET_SHAPE rule 7 — double-BOS check. `render_training_text`
        # already strips the leading <bos>; this asserts no stray BOS appears
        # mid-string either (e.g. from a copy-pasted few-shot example).
        if "<bos>" in text:
            print(f"FAIL: row {idx} (id={raw.get('id')}) contains <bos> "
                  f"after render_training_text — G_DATASET_SHAPE rule 7 violated.",
                  file=sys.stderr)
            sys.exit(4)

        ids = tokenizer(text, add_special_tokens=False).input_ids
        n = len(ids)
        if n > args.ctx_size:
            print(f"FAIL: row {idx} (id={raw.get('id')}) tokenized to {n} "
                  f"tokens > max_seq_length={args.ctx_size}.",
                  file=sys.stderr)
            sys.exit(5)

        preview = text[:200].replace("\n", "\\n")
        print(f"  [{idx}] cat={raw.get('category', '?'):24s} "
              f"tokens={n:5d}  text[:200]={preview!r}")

    # Counts — total file lengths, not just the rendered slice.
    train_total = sum(1 for _ in load_jsonl(args.train_file))
    val_total = sum(1 for _ in load_jsonl(args.val_file))
    test_total = sum(1 for _ in load_jsonl(args.test_file))
    print(f"\nsplit counts: train={train_total}  val={val_total}  test={test_total}")
    print("\nT1 gate PASS — dataset shape + render + length OK")
    sys.exit(0)


# --------------------------------------------------------------------------
# Block F1 — refusal-class loss reweighting (2026-05-01).
#
# Why: cp-111 → cp-333 of v3 collapses `medical_advice_refusal` 100% → 62.5%
# despite the dataset itself being able to teach the contract (cp-111 proves
# it). Diagnosis: 679 tool-call rows vs 202 refusal rows → per-step gradient
# is ~3.4x stronger toward tool-call generation; later epochs erase the
# refusal abstraction. F1 upweights refusal-row token losses so the gradient
# ratio leans back toward the under-represented class without touching the
# validated dataset semantics.
#
# Aggregation choice (pinned, per advisor 2026-05-01):
#   loss = sum(per_token_CE * label_mask * row_weight_broadcast) / denom
# where `denom` is the unweighted unmasked-token count. With every row weight
# = 1.0 and `num_items_in_batch=None`, this is bit-identical to vanilla SFT
# loss (see equivalence test in test_finetune_functiongemma_weighting.py).
# When `num_items_in_batch` is provided (TRL 0.22.2 grad-accum path), `denom`
# is `num_items_in_batch` instead — matches HF Trainer's grad-accum scaling.
# Picking "weighted SUM / unweighted COUNT" instead of "per-row mean then
# weighted mean" is what makes weight=1.0 a true no-op.
# --------------------------------------------------------------------------


def weighted_masked_lm_loss(
    logits: Any,
    labels: Any,
    *,
    row_weight: Any | None = None,
    num_items_in_batch: int | None = None,
    chunk_tokens: int = 256,
) -> Any:
    """Causal-LM loss with optional per-row weighting and grad-accum scaling.

    **Per-row chunked CE** — pinned 2026-05-01 after the flat-CE OOM observed
    at step 6 of `outputs_fg_v4_f1_weight2`: Gemma 3 270M has V=262 144, so a
    flat `F.cross_entropy` over `[B*T, V]` materializes ~5 GiB on top of the
    11 GiB activation footprint and OOMs the 16 GiB RTX 5080 even though the
    *vanilla* SFT path runs fine (Gemma's internal forward uses a fused/chunked
    kernel when labels are passed). We unroll per row and only over the
    response (label != -100) positions, in `chunk_tokens`-sized slices, so the
    per-step CE intermediate is bounded by `chunk_tokens * V * 4 ≈ 256 MiB`.

    Aggregation (advisor 2026-05-01):
        loss = sum(per_token_CE * row_weight_broadcast) / denom

    where `denom` is the unweighted unmasked-token count when
    `num_items_in_batch` is None, else `num_items_in_batch`. With every
    `row_weight == 1.0` this is mathematically equivalent to vanilla SFT loss;
    fp32 associativity drift between flat-sum and per-row chunked-sum is
    bounded by ~1e-5 on test-scale tensors, which is what
    `test_weight_one_is_no_op` allows.

    Args:
        logits: `[B, T, V]` causal-LM head outputs (unshifted; this function
            does the shift).
        labels: `[B, T]` int64 labels with `-100` at masked positions
            (post-`train_on_responses_only`).
        row_weight: `[B]` per-row multiplier or `None` (treated as all-1.0).
        num_items_in_batch: HF Trainer 4.45+ grad-accum scaler. When set, the
            denominator is this value (matches the accumulator's expectation
            that micro-batches contribute SUMs, not means).
        chunk_tokens: max response tokens per CE call. Bound peak alloc; lower
            if VRAM is tight, higher for less Python-loop overhead. 256 is the
            production safe value on the 16 GiB RTX 5080 (V=262 144).
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    B = logits.size(0)
    device = logits.device

    if row_weight is None:
        weights = torch.ones(B, dtype=torch.float32, device=device)
    elif not isinstance(row_weight, torch.Tensor):
        weights = torch.tensor(row_weight, dtype=torch.float32, device=device)
    else:
        weights = row_weight.to(dtype=torch.float32, device=device)

    numerator = torch.zeros((), dtype=torch.float32, device=device)
    total_unmasked = 0  # Python int — divisor only, no grad needed.

    for i in range(B):
        # Standard causal shift: position t's logit predicts label at t+1.
        row_logits_all = logits[i, :-1]                       # [T-1, V] view
        row_labels_all = labels[i, 1:]                        # [T-1]
        valid_mask = row_labels_all != -100
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]      # [n_valid]
        n_valid = int(valid_idx.numel())
        if n_valid == 0:
            continue

        row_sum = torch.zeros((), dtype=torch.float32, device=device)
        for s in range(0, n_valid, chunk_tokens):
            e = min(s + chunk_tokens, n_valid)
            idx = valid_idx[s:e]
            chunk_logits = row_logits_all.index_select(0, idx)   # [chunk, V]
            chunk_targets = row_labels_all.index_select(0, idx)  # [chunk]
            ce_sum = F.cross_entropy(chunk_logits, chunk_targets, reduction="sum")
            row_sum = row_sum + ce_sum.float()

        numerator = numerator + row_sum * weights[i]
        total_unmasked += n_valid

    if num_items_in_batch is not None:
        denom_f = float(num_items_in_batch) if num_items_in_batch != 0 else 1.0
        return numerator / denom_f
    return numerator / float(max(total_unmasked, 1))


def _build_weighted_trainer_class() -> Any:
    """Lazy-import torch + TRL so the host (no GPU) can still import this module.

    Returns the `WeightedSFTTrainer` subclass closed over the imported symbols.
    Doing it inside a factory keeps `import scripts.finetune_functiongemma`
    cheap on the dry-run path; the train path is the only caller.
    """
    from trl import SFTTrainer  # type: ignore[import-not-found]

    class WeightedSFTTrainer(SFTTrainer):  # type: ignore[misc]
        """SFTTrainer with per-row loss weighting via `inputs['row_weight']`.

        The `_WeightedCollator` wrapper attaches a `[B]` float tensor to each
        batch; this method pops it, computes per-token CE without reduction,
        masks instruction tokens via `labels == -100`, multiplies by the
        per-row weight broadcast across the time axis, and reduces.

        TRL 0.22.2's `compute_loss` signature includes `num_items_in_batch`;
        we accept and respect it for grad-accum correctness. Dropping the
        kwarg silently scales the loss by `1 / GAS` which is wrong by a factor
        of `num_items_in_batch / unmasked_count` per micro-batch.
        """

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ) -> Any:
            # Pop the metadata BEFORE model forward — `model(**inputs)` will
            # raise TypeError on unknown kwargs.
            row_weight = inputs.pop("row_weight", None)

            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            # Skip the model's internal CE by stripping `labels`; we recompute
            # CE ourselves from the logits to apply per-row weighting.
            inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**inputs_no_labels)
            loss = weighted_masked_lm_loss(
                outputs.logits, labels,
                row_weight=row_weight,
                num_items_in_batch=num_items_in_batch,
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def _build_weighted_collator(
    base_collator: Any,
    refusal_categories: frozenset[str],
    refusal_weight: float,
) -> Any:
    """Wrap `base_collator` so the produced batch carries `row_weight: [B]`.

    Sequencing is critical: this wrapper MUST be applied AFTER
    `train_on_responses_only`, because that helper mutates `trainer.data_collator`
    in place. Applying it before would let unsloth's wrapping drop our
    `row_weight` field silently (the equivalence test would not catch this —
    the refusal grad would just never flow).

    Resolution order, per row:
      1. `row_weight` field on the row (numeric, populated by `_train` via
         `add_column` AFTER tokenization — this is the path that survives
         TRL 0.22.2's `_prepare_non_packed_dataloader`, which calls
         `dataset.map(remove_columns=dataset.column_names)` and silently
         strips `category` before the collator ever sees the row. Pre-fix,
         all three weighted runs (weight=1.5/2.0/3.0) produced bit-identical
         results because every batch came in with category=None → weight=1.0).
      2. `category` field (kept for the unit tests + as a defensive fallback
         when used outside the SFTTrainer pipeline).
    """
    import torch

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        weights: list[float] = []
        clean: list[dict[str, Any]] = []
        for f in features:
            if "row_weight" in f and f["row_weight"] is not None:
                w = float(f["row_weight"])
            else:
                cat = str(f.get("category") or "")
                w = refusal_weight if cat in refusal_categories else 1.0
            weights.append(w)
            # Strip metadata that downstream collators (DataCollatorForLanguageModeling)
            # don't expect. `text` is consumed by SFTTrainer's tokenize step
            # before the collator runs, but if remove_unused_columns=False
            # leaves it on the row dict it slips through and trips the collator.
            clean.append({k: v for k, v in f.items()
                          if k not in ("category", "text", "row_weight")})
        batch = base_collator(clean)
        batch["row_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch

    return _collate


# --------------------------------------------------------------------------
# Train path (server only — Unsloth + GPU).
# --------------------------------------------------------------------------


def _train(args: argparse.Namespace) -> None:
    for path, label in ((args.train_file, "train"), (args.val_file, "val")):
        _ensure_split_exists(path, label)

    # Lazy: requires `unsloth` / `trl` / `datasets` (server-only; version-pinned
    # per §10.1). Unsloth must be imported BEFORE trl/transformers/peft to apply
    # its monkey-patches; doing the reverse triggers the runtime warning we saw
    # on the first failed run, and on torch 2.10 (no cpp extensions) it is the
    # difference between gradient flow working vs grad_norm=0.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import (  # type: ignore[import-not-found]
        train_on_responses_only,
    )
    from datasets import Dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    refusal_categories = frozenset(
        c.strip() for c in args.refusal_categories.split(",") if c.strip()
    )
    use_weighting = args.refusal_loss_weight != 1.0
    print(json.dumps({
        "block_F1": {
            "refusal_loss_weight": args.refusal_loss_weight,
            "refusal_categories": sorted(refusal_categories),
            "active": use_weighting,
        }
    }, indent=2))

    # Switched §10.2's `load_in_16bit=True` → `load_in_4bit=True` because the
    # 16-bit LoRA path produced `Trainable parameters = 0` + `grad_norm = 0`
    # on Gemma3 (Unsloth 2026.4.8 + transformers 4.56.2). 4-bit is the proven
    # notebook path and uses adamw_8bit anyway. Force SDPA via the kwarg even
    # though Unsloth currently downgrades Gemma3 to eager — the request is
    # still recorded in case Unsloth adds Gemma3 SDPA support.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=args.ctx_size,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        attn_implementation="sdpa",
    )

    # Why use_gradient_checkpointing=True (standard PyTorch) vs "unsloth":
    # Unsloth's "smart offload" path requires the cpp extensions which are
    # gated on torch >= 2.11.0; on our torch 2.10.0+cu128 they're skipped
    # at startup, and the python-only fallback drops gradients (grad_norm=0
    # every step on the first attempt). Standard PyTorch GC is well-trodden
    # and adds modest VRAM cost (270M is small enough that this is fine).
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        lora_alpha=256,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=False,
        loftq_config=None,
        random_state=3407,
    )
    # Diagnostic — must be > 0 trainable. The previous run reported
    # "Trainable parameters = 0" with `load_in_16bit=True`; this surfaces the
    # mis-wiring at startup rather than after one wasted epoch.
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_rows = _build_text_rows(args.train_file, tokenizer)
    val_rows = _build_text_rows(args.val_file, tokenizer)
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)
    print(f"train rows: {len(train_dataset)}   val rows: {len(val_dataset)}")

    sft_cfg = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,        # effective batch 8
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        # Why adamw_torch instead of adamw_8bit (notebook default): bnb's
        # 8-bit optimizer interacts with the 4-bit base model + LoRA via the
        # cpp extensions Unsloth gates on torch 2.11+. On torch 2.10 the
        # 8-bit path can leave LoRA grads at zero (observed empirically on
        # this stack). adamw_torch is the safe default — memory cost is
        # negligible at 30M LoRA params.
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        logging_steps=1,
        seed=3407,
        output_dir=args.output_dir,
        # Why deviate from notebook's report_to="none": every other training
        # script in this repo (see scripts/finetune.py) ships TensorBoard
        # logs; keeping parity makes `tensorboard --logdir runs/` Just Work.
        report_to="tensorboard",
        logging_dir=args.logging_dir,
        # Why eval_strategy="epoch" (notebook has no in-loop eval): we have
        # a real held-out val split per §9.7; per-epoch eval surfaces the
        # convergence curve without 50x cost on a ~400-row dataset.
        per_device_eval_batch_size=1,         # §10.3 safety on the 16 GiB ceiling
        eval_strategy="epoch",
        save_strategy="epoch",
        # Block F1: keep `category` column on the dataset so the weighted
        # collator can read it. Without this the SFTTrainer column-pruner
        # drops every non-tokenized field after `_prepare_dataset` runs.
        remove_unused_columns=False,
    )

    # Block F1: subclass selection. Default 1.0 → identical to vanilla
    # `SFTTrainer` per the equivalence test; non-1.0 → the weighted compute_loss
    # branch fires.
    trainer_cls: Any = (
        _build_weighted_trainer_class() if use_weighting else SFTTrainer
    )

    # TRL 0.22.2 renamed `tokenizer=` to `processing_class=` (matches the
    # `scripts/finetune.py` Gemma 3 path on the same trl version). The §10.2
    # Unsloth notebook still shows `tokenizer=`, but on our pinned TRL it's
    # a TypeError.
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_cfg,
    )

    # FG-aware response-only masking — replaces TRL's `completion_only_loss`.
    # The instruction/response markers are taken verbatim from notebook cell 31.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Block F1: attach `row_weight` as a NUMERIC COLUMN after tokenization,
    # because TRL 0.22.2's `_prepare_non_packed_dataloader` calls
    # `dataset.map(remove_columns=dataset.column_names)` and silently drops the
    # `category` field. Pre-fix (2026-05-01 first F1 grid), all three weighted
    # runs (weight=1.5/2.0/3.0) produced bit-identical evals because every batch
    # arrived with category=None → the collator defaulted to weight=1.0. The
    # `row_weight` column is keyed by row index, in lock-step with `train_rows`.
    if use_weighting:
        # Compute weights from the source `train_rows` (still has `category`).
        # `len(trainer.train_dataset)` should equal `len(train_rows)` on the
        # FunctionGemma corpus — every row passes the `Filter` step in the
        # training log. Assert it; if a future supplement causes drops we want
        # to fail loud rather than apply weights to the wrong rows.
        weight_per_row = [
            args.refusal_loss_weight if r["category"] in refusal_categories else 1.0
            for r in train_rows
        ]
        assert len(trainer.train_dataset) == len(weight_per_row), (
            f"trainer.train_dataset has {len(trainer.train_dataset)} rows "
            f"but train_rows has {len(weight_per_row)} — Filter dropped rows "
            f"and the row_weight alignment would be off-by-N"
        )
        trainer.train_dataset = trainer.train_dataset.add_column(
            "row_weight", weight_per_row,
        )
        n_weighted = sum(1 for w in weight_per_row if w != 1.0)
        print(f"Block F1: row_weight column added — {n_weighted}/{len(weight_per_row)} "
              f"rows weighted at {args.refusal_loss_weight} "
              f"(refusal cats: {sorted(refusal_categories)})")

        trainer.data_collator = _build_weighted_collator(
            trainer.data_collator,
            refusal_categories=refusal_categories,
            refusal_weight=args.refusal_loss_weight,
        )
        print("Block F1 active: data_collator wrapped → reads row_weight column "
              "from the tokenized dataset")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nAdapters + tokenizer saved to: {args.output_dir}")
    print("Next steps (see plan §10.4):")
    print("  1. model.save_pretrained_merged('merged_fg_v1', tokenizer, save_method='merged_16bit')")
    print("  2. model.save_pretrained_gguf('merged_fg_v1', tokenizer, quantization_method='f16')")
    print("  3. model.save_pretrained_gguf('merged_fg_v1', tokenizer, quantization_method='q8_0')")
    print("  4. ~/llama.cpp/build/bin/llama-quantize merged_fg_v1.f16.gguf "
          "merged_fg_v1.q4_k_m.gguf Q4_K_M")


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def main() -> None:
    args = _parse()

    # Echo resolved paths up-front — useful when the script is invoked from
    # an unfamiliar cwd (e.g. via ssh -t).
    print(json.dumps({
        "cwd": os.getcwd(),
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "test_file": str(args.test_file),
        "output_dir": args.output_dir,
        "logging_dir": args.logging_dir,
        "ctx_size": args.ctx_size,
        "dry_run": args.dry_run,
        "refusal_loss_weight": args.refusal_loss_weight,
        "refusal_categories": args.refusal_categories,
    }, indent=2))

    if args.dry_run:
        _dry_run(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
