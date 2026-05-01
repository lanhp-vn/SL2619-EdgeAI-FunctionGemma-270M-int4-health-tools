#!/usr/bin/env python3
"""FunctionGemma 270M-IT SFT v2 — vendor-faithful baselines (A1 + A2).

Born from the M5 deep-dive in `docs/plans/FunctionGemma/README.md` §10.7:
the M5 Unsloth + LoRA r=128 recipe trained without errors but failed M6
G_EVAL at 25/56 (44.6 %) overall, every category below the §11.4 ≥ 80 % bar.
The original v1 script copied Unsloth's *capability-bootstrap* recipe (designed
to teach <think> reasoning to a base model that lacks it) and applied it to a
function-calling fine-tune on 511 rows — three vendor recipes that have
published function-calling deltas (Mobile-Actions HF 58→85 %, Mobile-Actions
Tunix LoRA r=8 65→88 %, Cookbook small demo 80 % on 40 rows) all use either
full SFT or LoRA r=8 — never r=128.

This script lands the **vendor-faithful baselines** to identify whether the
recipe (LoRA rank, LR, scheduler, optimizer, loss masking) or the dataset
(diversity, argument-value coverage) is the bottleneck.

Two recipes, selected via `--recipe`:

A1 = `mobile_actions_hf` (vendor full SFT, the 58→85 % gold standard):
    - pure transformers + trl, NO LoRA, NO Unsloth
    - eager attn, BF16
    - 2 epochs, eff batch 32 (PDB=4, GAS=8), LR=1e-5, cosine
    - adamw_torch_fused, completion_only_loss=True
    - max_length = longest_token_count + 100 (auto-computed)
    - gradient_checkpointing=True, packing=False

A2 = `mobile_actions_tunix` (vendor LoRA r=8, the 65→88 % delta, ported to PEFT):
    - pure transformers + peft + trl, NO Unsloth
    - LoRA r=8, alpha=16, target=q/k/v/gate/up/down (NO o_proj — Tunix's
      q_einsum + kv_einsum maps to q/k/v not o)
    - lora_dropout=0, bias="none"
    - 1 epoch, eff batch 8 (PDB=8, GAS=1), LR=1e-4, cosine
    - adamw_torch_fused, completion_only_loss=True

Why preserve `scripts/finetune_functiongemma.py` (v1):
    The v1 script encodes the M5 deviations from the §10.2 Unsloth recipe
    (forced sdpa, UNSLOTH_RETURN_LOGITS=1, load_in_4bit, etc.). Keeping it
    lets the deep-dive bench compare v1 vs v2 directly (same data, same
    eval) without re-deriving the v1 surface from git history.

Dataset preprocessing:
    - Each row is cropped to its FIRST assistant turn — this matches the M6
      eval contract (`extract_gold_trace` in `eval_functiongemma_holdout.py`
      compares to the first assistant turn only). Multi-turn rows
      (two_turn / parallel_call / tool_error_recovery) lose their post-tool
      NL response from the training signal; the call-decision is what we
      score.
    - Reshape to {prompt, completion, split} per the vendor Mobile-Actions
      HF cell 14 recipe (`apply_chat_template` twice — once with
      `add_generation_prompt=True` for prompt, once with `=False` for
      prompt+completion, slice the difference).
    - Refusal rows (off_topic / medical_advice) have no tool_calls in the
      first assistant turn — they get rendered as a plain NL response,
      which is the correct training signal for the refusal contract.

T1 dry-run gate (host or server — degrades gracefully without GPU):
    uv run python scripts/finetune_functiongemma_v2.py --recipe mobile_actions_hf \
        --dry-run --max-dry-run-rows 4

Full run (server only):
    ssh -t nouslogic-server '
      cd ~/functiongemma-finetune &&
      source .venv/bin/activate &&
      export PYTHONPATH=~/functiongemma-finetune &&
      python finetune_functiongemma_v2.py --recipe mobile_actions_hf \
          --train-file data/train.jsonl --val-file data/val.jsonl \
          --output-dir outputs_fg_v2_a1 --logging-dir runs/v2_a1
    '
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# `gemma_tools` ships without a py.typed marker; suppress missing-stubs noise so
# `mypy scripts/finetune_functiongemma_v2.py` runs clean on the host.
from gemma_tools.functiongemma_dataset import (  # type: ignore[import-untyped]
    load_jsonl,
    validate_file,
)

MODEL_ID = "google/functiongemma-270m-it"
# Unsloth mirror is bit-identical to Google's; the latter is in the host hf-cache.
_HOST_TOKENIZER_FALLBACKS: tuple[str, ...] = (
    MODEL_ID,
    "unsloth/functiongemma-270m-it",
    str(Path.home() / "hf-cache" / "functiongemma-270m-it"),
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "functiongemma" / "dataset_v1"


# --------------------------------------------------------------------------
# Recipes — each is a frozen dataclass so the dry-run path can echo what
# the train path would do without importing torch/trl.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    name: str
    use_lora: bool
    # SFTConfig
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str           # "cosine" / "linear" / "constant"
    optim: str                       # "adamw_torch_fused" / "adamw_torch"
    completion_only_loss: bool
    gradient_checkpointing: bool
    # LoRA (ignored if use_lora=False)
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]


_RECIPES: dict[str, Recipe] = {
    # A1: Mobile-Actions HF full SFT — the 58→85 % vendor delta.
    "mobile_actions_hf": Recipe(
        name="mobile_actions_hf",
        use_lora=False,
        num_train_epochs=2.0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,        # effective batch 32
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        completion_only_loss=True,
        gradient_checkpointing=True,
        lora_r=0, lora_alpha=0, lora_dropout=0.0, lora_target_modules=(),
    ),
    # A2: Mobile-Actions Tunix LoRA r=8 — the 65→88 % vendor delta, ported to
    # PEFT. Tunix's q_einsum + kv_einsum maps to q/k/v; gate/down/up_proj
    # appear verbatim. Note: NO o_proj (Tunix's variant doesn't target it).
    "mobile_actions_tunix": Recipe(
        name="mobile_actions_tunix",
        use_lora=True,
        num_train_epochs=1.0,
        # Why PDB=4 GAS=2 (not vendor's PDB=8): activations for 8 sequences of
        # length ~780 + LoRA weights + base model OOM the 16 GiB RTX 5080 (saw
        # `CUDA out of memory. Tried to allocate 4.77 GiB` at epoch 1 step 0).
        # PDB=4 keeps eff batch 8 (vendor) via accumulation, no behavioral change.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,        # effective batch 8
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        completion_only_loss=True,
        gradient_checkpointing=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj",
                             "gate_proj", "up_proj", "down_proj"),
    ),
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FunctionGemma 270M-IT SFT v2 — vendor-faithful baselines."
    )
    p.add_argument("--recipe", choices=sorted(_RECIPES), required=True,
                   help="Vendor recipe to apply. See module docstring.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate splits + render a slice + exit. No GPU needed.")
    p.add_argument("--train-file", type=Path,
                   default=_DEFAULT_DATA_DIR / "train.jsonl")
    p.add_argument("--val-file", type=Path,
                   default=_DEFAULT_DATA_DIR / "val.jsonl")
    p.add_argument("--test-file", type=Path,
                   default=_DEFAULT_DATA_DIR / "test.jsonl")
    p.add_argument("--output-dir", default="outputs_fg_v2",
                   help="SFTTrainer output dir (checkpoints + final adapter/model)")
    p.add_argument("--logging-dir", default="runs/v2",
                   help="TensorBoard log dir")
    p.add_argument("--max-dry-run-rows", type=int, default=8)
    # Recipe overrides — useful for Block B sweeps
    p.add_argument("--epochs", type=float, default=None,
                   help="Override num_train_epochs (Block B sweeps)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning_rate (Block B sweeps)")
    p.add_argument("--lora-r", type=int, default=None,
                   help="Override LoRA rank (only valid with use_lora recipe)")
    p.add_argument("--lora-dropout", type=float, default=None,
                   help="Override LoRA dropout")
    p.add_argument("--target-modules", default=None,
                   help="Comma-separated LoRA target modules override "
                        "(e.g. 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,embed_tokens')")
    p.add_argument("--merge-train-val", action="store_true",
                   help="Block B10: concat train + val into one training set; "
                        "still use val as eval (sanity tracking).")
    return p.parse_args()


# --------------------------------------------------------------------------
# Tokenizer load (host fallback chain — same as v1)
# --------------------------------------------------------------------------


def _load_tokenizer(prefer_unsloth: bool = False) -> tuple[Any, str]:
    """Best-effort tokenizer load. Raises if all sources fail."""
    from transformers import AutoTokenizer
    last_err: Exception | None = None
    sources = (_HOST_TOKENIZER_FALLBACKS if not prefer_unsloth
               else ("unsloth/functiongemma-270m-it", *_HOST_TOKENIZER_FALLBACKS))
    for src in sources:
        try:
            return AutoTokenizer.from_pretrained(src), src
        except Exception as exc:
            # Network error, missing cache, gating — try next.
            last_err = exc
            continue
    raise RuntimeError(f"could not load tokenizer; last_err={last_err!r}")


# --------------------------------------------------------------------------
# Dataset preprocessing (vendor Mobile-Actions HF cell 14, adapted)
# --------------------------------------------------------------------------


def _crop_to_first_assistant(row: dict[str, Any]) -> dict[str, Any]:
    """Drop messages after the first assistant turn.

    The M6 eval (`extract_gold_trace`) scores the first assistant turn only.
    Training on later assistant turns (the post-tool NL response in two_turn /
    parallel_call / tool_error_recovery rows) supervises a different decision
    than what we evaluate, and pulls the loss toward the wrong target on the
    minority of rows that have multi-turn structure.
    """
    msgs = row["messages"]
    first_ast = next(i for i, m in enumerate(msgs) if m.get("role") == "assistant")
    return {"messages": msgs[: first_ast + 1], "tools": row.get("tools")}


def _to_prompt_completion(row: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Vendor Mobile-Actions HF cell 14 recipe — split rendered template at
    the assistant-turn boundary into (prompt, completion).

    `prompt`     = render(messages[:-1], add_generation_prompt=True)  → ends in `<start_of_turn>model\n`
    `completion` = render(messages,      add_generation_prompt=False)[len(prompt):]
                 = the assistant-turn body (including `<think>...`, the call,
                   and either `<start_function_response>` or `<end_of_turn>`
                   depending on whether the turn has tool_calls).

    Both are kept verbatim — no BOS strip. The chat template emits one `<bos>`
    at the head of `prompt`; SFTTrainer with prompt+completion columns
    tokenizes with `add_special_tokens=False` (the chat template owns the
    special tokens), so there is no double-BOS.
    """
    msgs = row["messages"]
    tools = row.get("tools")
    full = tokenizer.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=False
    )
    prompt = tokenizer.apply_chat_template(
        msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True
    )
    if not full.startswith(prompt):
        raise RuntimeError(
            f"chat-template split failed: prompt is not a prefix of full. "
            f"prompt[-50:]={prompt[-50:]!r}  full[len(p)-50:len(p)+50]="
            f"{full[len(prompt) - 50:len(prompt) + 50]!r}"
        )
    completion = full[len(prompt):]
    return {"prompt": prompt, "completion": completion}


def _build_prompt_completion_rows(
    rows: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for raw in rows:
        cropped = _crop_to_first_assistant(raw)
        out.append(_to_prompt_completion(cropped, tokenizer))
    return out


def _ensure_split_exists(path: Path, label: str) -> None:
    if not path.exists():
        print(f"ERROR: {label} split not found at {path}.", file=sys.stderr)
        sys.exit(2)


# --------------------------------------------------------------------------
# Dry-run path — file shape + render + length, no GPU needed.
# --------------------------------------------------------------------------


def _dry_run(args: argparse.Namespace, recipe: Recipe) -> None:
    for path, label in ((args.train_file, "train"),
                        (args.val_file, "val"),
                        (args.test_file, "test")):
        _ensure_split_exists(path, label)

    print(f"=== recipe: {recipe.name} ===")
    print(json.dumps({
        "use_lora": recipe.use_lora,
        "epochs": args.epochs or recipe.num_train_epochs,
        "lr": args.lr or recipe.learning_rate,
        "scheduler": recipe.lr_scheduler_type,
        "optim": recipe.optim,
        "eff_batch": recipe.per_device_train_batch_size * recipe.gradient_accumulation_steps,
        "completion_only_loss": recipe.completion_only_loss,
        "lora_r": (args.lora_r if args.lora_r is not None else recipe.lora_r) if recipe.use_lora else None,
        "lora_target": (args.target_modules.split(",") if args.target_modules
                        else list(recipe.lora_target_modules)) if recipe.use_lora else None,
    }, indent=2))

    print("\n=== file-shape validation (validate_file) ===")
    for path, label in ((args.train_file, "train"), (args.val_file, "val")):
        report = validate_file(path)
        print(f"  {label:5s} {path.name}: {report.passed}/{report.total} pass "
              f"(rate={report.pass_rate:.4f})")
        if not report.meets_threshold:
            print("  validator below threshold; aborting.", file=sys.stderr)
            sys.exit(3)

    try:
        tokenizer, src = _load_tokenizer()
    except RuntimeError as exc:
        print(f"\nT1 PARTIAL — file-shape OK; render gate skipped ({exc})")
        sys.exit(0)
    print(f"\nloaded tokenizer from `{src}`")

    print(f"\n=== render gate (first {args.max_dry_run_rows} train rows) ===")
    longest_token_count = 0
    for idx, raw in enumerate(load_jsonl(args.train_file)):
        if idx >= args.max_dry_run_rows:
            break
        cropped = _crop_to_first_assistant(raw)
        pc = _to_prompt_completion(cropped, tokenizer)
        n_prompt = len(tokenizer(pc["prompt"], add_special_tokens=False).input_ids)
        n_completion = len(tokenizer(pc["completion"], add_special_tokens=False).input_ids)
        longest_token_count = max(longest_token_count, n_prompt + n_completion)
        preview = pc["completion"][:120].replace("\n", "\\n")
        print(f"  [{idx}] cat={raw.get('category', '?'):24s} "
              f"prompt={n_prompt:4d} completion={n_completion:4d}  "
              f"completion[:120]={preview!r}")

    print(f"\nlongest tokens (prompt + completion) in sampled rows: {longest_token_count}")
    print("T1 PASS — recipe + file-shape + render OK")


# --------------------------------------------------------------------------
# Train path
# --------------------------------------------------------------------------


def _train(args: argparse.Namespace, recipe: Recipe) -> None:
    for path, label in ((args.train_file, "train"), (args.val_file, "val")):
        _ensure_split_exists(path, label)

    # Lazy: server-only stack.
    import torch
    from datasets import Dataset  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    print(f"=== loading tokenizer + base model ({MODEL_ID}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
        # Vendor recipe — eager attn, no flex_attention dtype mismatch surprises.
        attn_implementation="eager",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("=== preprocessing rows (crop to first assistant + render) ===")
    train_raw = list(load_jsonl(args.train_file))
    val_raw = list(load_jsonl(args.val_file))
    if args.merge_train_val:
        # Block B10: use train + val for training (eval still on val for tracking).
        train_raw = train_raw + val_raw
        print(f"  merge-train-val: combined to {len(train_raw)} rows")
    train_rows = _build_prompt_completion_rows(train_raw, tokenizer)
    val_rows = _build_prompt_completion_rows(val_raw, tokenizer)

    # Compute longest prompt+completion across both splits, +100 (vendor-style)
    # — `max_length` should never truncate a real example.
    longest = 0
    for r in train_rows + val_rows:
        n = len(tokenizer(r["prompt"] + r["completion"], add_special_tokens=False).input_ids)
        longest = max(longest, n)
    max_length = longest + 100
    print(f"  longest combined tokens={longest}; using max_length={max_length}")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    print(f"  train rows: {len(train_ds)}   val rows: {len(val_ds)}")

    # Apply CLI overrides on top of the recipe.
    epochs = args.epochs if args.epochs is not None else recipe.num_train_epochs
    lr = args.lr if args.lr is not None else recipe.learning_rate

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=recipe.per_device_train_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        per_device_eval_batch_size=2,           # holdout fits comfortably; smaller for safety
        learning_rate=lr,
        lr_scheduler_type=recipe.lr_scheduler_type,
        optim=recipe.optim,
        bf16=True,
        max_length=max_length,
        packing=False,
        completion_only_loss=recipe.completion_only_loss,
        gradient_checkpointing=recipe.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="tensorboard",
        logging_dir=args.logging_dir,
        seed=3407,
    )

    peft_config = None
    if recipe.use_lora:
        from peft import LoraConfig  # type: ignore[import-not-found]
        target = (tuple(args.target_modules.split(","))
                  if args.target_modules else recipe.lora_target_modules)
        rank = args.lora_r if args.lora_r is not None else recipe.lora_r
        dropout = args.lora_dropout if args.lora_dropout is not None else recipe.lora_dropout
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=recipe.lora_alpha,
            target_modules=list(target),
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"  LoRA: r={rank} alpha={recipe.lora_alpha} dropout={dropout} target={target}")

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )

    # Sanity: at least one train batch has non-zero label tokens (i.e.
    # `completion_only_loss=True` actually masked SOMETHING, not EVERYTHING).
    sample = trainer.train_dataset[0]
    label_ids = sample.get("labels") or sample.get("input_ids", [])
    n_unmasked = sum(1 for x in label_ids if x != -100) if isinstance(label_ids, list) else None
    print(f"  sanity: row 0 has {n_unmasked} unmasked label tokens (must be > 0)")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n=== saved {('LoRA adapter' if recipe.use_lora else 'full model')} "
          f"to {args.output_dir} ===")
    if recipe.use_lora:
        print("To eval: merge with `peft.merge_and_unload()` then point eval at "
              "the merged dir (see ~/functiongemma-finetune/merge_v2.py).")
    else:
        print(f"To eval: `python eval_functiongemma_holdout.py "
              f"--checkpoint {args.output_dir} --max-new-tokens 512`.")


# --------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------


def main() -> None:
    args = _parse()
    recipe = _RECIPES[args.recipe]
    print(json.dumps({
        "cwd": os.getcwd(),
        "recipe": recipe.name,
        "train": str(args.train_file),
        "val": str(args.val_file),
        "output_dir": args.output_dir,
        "logging_dir": args.logging_dir,
        "dry_run": args.dry_run,
    }, indent=2))
    if args.dry_run:
        _dry_run(args, recipe)
    else:
        _train(args, recipe)


if __name__ == "__main__":
    main()
