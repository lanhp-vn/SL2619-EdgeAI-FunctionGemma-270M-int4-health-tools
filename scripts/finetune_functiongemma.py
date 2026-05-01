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
    rows: list[dict[str, str]] = []
    for raw in load_jsonl(path):
        text = render_training_text(raw, tokenizer)
        rows.append({"text": text})
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
    )

    # TRL 0.22.2 renamed `tokenizer=` to `processing_class=` (matches the
    # `scripts/finetune.py` Gemma 3 path on the same trl version). The §10.2
    # Unsloth notebook still shows `tokenizer=`, but on our pinned TRL it's
    # a TypeError.
    trainer = SFTTrainer(
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
    }, indent=2))

    if args.dry_run:
        _dry_run(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
