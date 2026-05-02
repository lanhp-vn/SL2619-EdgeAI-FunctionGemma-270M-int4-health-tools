#!/usr/bin/env python3
"""
QLoRA SFT for Gemma 3 270M-IT — closed-world health-YAML QA.

Hyperparameters mirror Google's emoji-translator notebook structure but track
the trl 1.3.0 API on the bootstrap'd server (`docs/tmp/nouslogic-server-status.md`):
- `DataCollatorForCompletionOnlyLM` was removed in trl 1.x → use the native
  prompt-completion dataset shape so `SFTConfig.completion_only_loss=True` masks
  user-turn loss without a custom collator.
- `max_seq_length` was renamed to `max_length`.
- Gemma 3's chat template has no `{% generation %}` markers, so the parallel
  `assistant_only_loss=True` path silently returns an all-zero assistant mask
  (verified on the server). `completion_only_loss` is the supported route for
  this template.

Single source of truth for the values below: `docs/plans/AI-models/a55-gemma-fine-tune.md` §6.

Deploy:
    scp scripts/finetune.py nouslogic-server:~/sl2619-finetune/

T1 dry-run gate (loads tokenizer + model + 1 dataset row, prints decoded
preview, exits 0):
    ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python finetune.py --dry-run'

Full training run:
    ssh -t nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python finetune.py'

After training, run merge.py to produce the merged BF16 HF checkpoint for Q0.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-3-270m-it"

_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_WORKSPACE, "data")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT for Gemma 3 270M-IT health-YAML QA")
    p.add_argument("--dry-run", action="store_true",
                   help="Load tokenizer + model + 1 dataset row, print preview, exit 0 (T1 gate)")
    p.add_argument("--output-dir", default=os.path.join(_WORKSPACE, "adapters_v1"))
    p.add_argument("--train-file", default=os.path.join(_DATA_DIR, "sft_v1.train.jsonl"))
    p.add_argument("--val-file",   default=os.path.join(_DATA_DIR, "sft_v1.val.jsonl"))
    return p.parse_args()


def _make_tokenizer() -> Any:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Gemma 3 has no dedicated pad token; EOS is the conventional choice and
    # matches what the chat template uses for the final turn boundary.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _make_model(bnb_cfg: BitsAndBytesConfig) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # use_cache is incompatible with gradient checkpointing; we disable GC per
    # §6 (270M is small enough — no need to trade wall-clock for memory).
    model.config.use_cache = False
    return model


def _to_prompt_completion(row: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Split one `messages` row into (prompt, completion).

    `prompt` = chat template applied to the user turn with `add_generation_prompt=True`,
    so it ends with `<start_of_turn>model\\n` — the open boundary the model
    learns to step over. `completion` = bare assistant text. trl's default
    pipeline tokenizes prompt+completion, masks prompt tokens to -100, and
    appends EOS to the completion side.

    Asserts user/assistant only — Gemma 3 has no system role; system content
    is folded into the user turn at dataset-build time per
    `docs/conventions/slm-system-prompt.md §2`.
    """
    msgs = row["messages"]
    roles = [m["role"] for m in msgs]
    if roles != ["user", "assistant"]:
        raise ValueError(f"expected ['user','assistant'] only, got {roles}")
    prompt: str = tokenizer.apply_chat_template(
        [msgs[0]], tokenize=False, add_generation_prompt=True,
    )
    return {"prompt": prompt, "completion": msgs[1]["content"]}


def _build_lora_cfg() -> LoraConfig:
    # Deviation from plan §6 — `modules_to_save=["lm_head","embed_tokens"]`
    # dropped for v1. Rationale (logged 2026-04-27 after dry-run measured
    # 55.86% trainable / 339M trainable params instead of the expected
    # ~6%):
    #   1. Gemma 3 has `tie_word_embeddings=True`. peft splits the tied
    #      pair into two independent full-precision modules-to-save copies
    #      (~167M each), so the trainable surface is ~334M FP-precision
    #      params on top of the 4-bit base. peft warns this corrupts the
    #      `merge_and_unload` → safetensors → GGUF chain (Q0 would emit a
    #      broken vocabulary projection); Phase 3 hard-blocked.
    #   2. 1023 train examples cannot safely retrain 167M+ embed params
    #      without catastrophic forgetting. The H6 failure mode is
    #      *behavioral* (definitional drift), not vocabulary — the IT
    #      model's English health-term embeddings are already correct.
    # Pure LoRA on `all-linear` (attention + MLP projections) is the
    # appropriate surface area: lands ~1-2% trainable, preserves tied
    # weights end-to-end into Q0. If T5 side-by-side smoke shows no
    # behavior change, the documented escalation is to reintroduce
    # `modules_to_save=["embed_tokens"]` (one only) + `ensure_weight_tying
    # =True` — not the original two-element list.
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def _build_sft_cfg(output_dir: str, logging_dir: str) -> SFTConfig:
    # Deviation from plan §6 — `per_device_train_batch_size: 4 → 1`,
    # `gradient_accumulation_steps: 4 → 16`, plus new
    # `per_device_eval_batch_size=1`. Effective train batch unchanged at 16.
    # Rationale (logged 2026-04-28 after first training attempt OOM'd at
    # step 0, log `logs/train-20260428-064234.log`):
    #   - Gemma 3 270M has vocab_size=262,144 (large vocab for a small
    #     model). Forward pass produces logits of shape
    #     (batch, seq, vocab); at PDB=4 / seq=1024 / BF16 that is
    #     4 * 1024 * 262144 * 2 = 2.0 GiB just for the logits tensor,
    #     and `logits[..., :-1, :].contiguous()` in the SFT loss path
    #     materializes another ~2 GiB peak — the OOM trace showed
    #     `Tried to allocate 3.66 GiB ... 11.72 GiB already in use`,
    #     leaving the run 0.4 GiB short on a 15.0 GiB-free GPU.
    #   - Plan §6 said "270M is small enough — no memory pressure" — true
    #     for params + grads + AdamW state; missed that Gemma 3's vocab
    #     dominates the activation footprint of the loss head.
    #   - PDB=1 drops the logits tensor 4x to 512 MiB and scales every
    #     other batch-dependent activation linearly. Effective batch
    #     stays 16 via GAS=16 → optimization trajectory is preserved
    #     (training loss curves should be ~identical to the §6 config).
    #   - `per_device_eval_batch_size` defaults to 8 in HF
    #     `TrainingArguments`; eval logits at PDB_eval=8 / seq=1024 /
    #     BF16 = 4 GiB for the logits tensor alone. Eval at epoch end
    #     would OOM after a full epoch of training. Pin it to 1.
    # Wall-clock cost: ~5-10% slower per epoch on this GPU (270M is small
    # enough that per-step launch overhead is the limiter, not throughput).
    # Documented escalation if PDB=1 still OOMs (it shouldn't): enable
    # `gradient_checkpointing=True` (25-35% wall-clock cost) before
    # touching `max_length` — the 1024 cap is load-bearing (sample-0 is
    # 930 tokens).
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,        # effective batch = 16 (preserved)
        per_device_eval_batch_size=1,          # eval default is 8 → 4 GiB logits OOM at vocab=262144
        learning_rate=5e-5,
        lr_scheduler_type="constant",
        max_length=1024,                       # ≥ 820 needed for Path-B user turns; Google notebook uses 512 (too small)
        gradient_checkpointing=False,          # PDB=1 leaves headroom — GC would add 25-35% wall-clock for no benefit
        packing=False,
        completion_only_loss=True,             # mask user-turn tokens (replaces removed DataCollatorForCompletionOnlyLM)
        optim="adamw_torch_fused",
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="tensorboard",
        logging_dir=logging_dir,
        seed=42,
    )


def _dry_run(tokenizer: Any, train_ds: Dataset, val_ds: Dataset,
             bnb_cfg: BitsAndBytesConfig, lora_cfg: LoraConfig,
             sft_cfg: SFTConfig) -> None:
    """T1 gate — loads everything, prints a decoded preview, exits 0."""
    print(f"train rows : {len(train_ds)}   val rows : {len(val_ds)}")

    sample = train_ds[0]
    full_text = sample["prompt"] + sample["completion"]
    n_prompt = len(tokenizer(sample["prompt"], add_special_tokens=False).input_ids)
    n_full = len(tokenizer(full_text, add_special_tokens=False).input_ids)

    print(f"sample-0 prompt tokens     : {n_prompt}")
    print(f"sample-0 prompt+completion : {n_full}  (max_length={sft_cfg.max_length})")

    if n_full > sft_cfg.max_length:
        print(f"FAIL: sample-0 length {n_full} exceeds max_length {sft_cfg.max_length}")
        sys.exit(1)

    # System-role check (folded-into-user contract)
    user_text = sample["prompt"]
    if "<start_of_turn>system" in user_text or "system\n" in user_text.split("user\n", 1)[0]:
        print(f"FAIL: prompt unexpectedly contains a system role:\n{user_text[:300]}")
        sys.exit(1)

    # BOS sanity — chat template emits <bos> once at the start; tokenizing
    # again with add_special_tokens=False must not double it.
    prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False).input_ids
    if prompt_ids.count(tokenizer.bos_token_id) != 1:
        print(f"FAIL: BOS token appears {prompt_ids.count(tokenizer.bos_token_id)} times "
              f"in prompt (expected exactly 1)")
        sys.exit(1)

    print("\n--- decoded prompt preview (first 800 chars) ---")
    print(sample["prompt"][:800])
    print("--- decoded completion ---")
    print(sample["completion"])
    print("--- end preview ---\n")

    # Loading the model surfaces 4-bit quantization path failures end-to-end.
    model = _make_model(bnb_cfg)
    dev = next(model.parameters()).device
    print(f"model dtype                : {model.dtype}   device: {dev}")

    # Build the full trainer stack to surface modules_to_save+tied-weights+
    # 4-bit failures and any SFTConfig param-name drift before training.
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )
    # Trainable-parameter ratio confirms LoRA wiring; expect single-digit %.
    trainer.model.print_trainable_parameters()

    print("\nT1 gate PASS — tokenizer + dataset + model + trainer stack OK")
    sys.exit(0)


def main() -> None:
    args = _parse()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = _make_tokenizer()

    raw = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})
    train_ds = raw["train"].map(
        lambda r: _to_prompt_completion(r, tokenizer), remove_columns=["messages"]
    )
    val_ds = raw["val"].map(
        lambda r: _to_prompt_completion(r, tokenizer), remove_columns=["messages"]
    )

    lora_cfg = _build_lora_cfg()
    sft_cfg = _build_sft_cfg(args.output_dir, os.path.join(_WORKSPACE, "runs"))

    if args.dry_run:
        _dry_run(tokenizer, train_ds, val_ds, bnb_cfg, lora_cfg, sft_cfg)

    # ── Full training run ────────────────────────────────────────────────────
    model = _make_model(bnb_cfg)
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapters + tokenizer saved to: {args.output_dir}")
    print("Next: python ~/sl2619-finetune/merge.py")


if __name__ == "__main__":
    main()
