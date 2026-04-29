# Hugging Face Training Stack — Upstream Sources

The fine-tune in this repo runs on `transformers + trl + peft + bitsandbytes`.
Each library moves fast; pin behavior to a specific docs revision when in
doubt and bump intentionally.

## Core libraries

| Library | URL | Authoritative for |
|---|---|---|
| Transformers (HF) | <https://huggingface.co/docs/transformers> | `AutoModelForCausalLM`, tokenizer, `apply_chat_template` |
| Transformers source | <https://github.com/huggingface/transformers> | actual implementation; ground-truth when docs lag |
| TRL — SFT trainer | <https://huggingface.co/docs/trl/sft_trainer> | `SFTTrainer`, `SFTConfig`, completion-only/assistant-only loss |
| TRL source | <https://github.com/huggingface/trl> | `trl/trainer/sft_trainer.py` for exact masking semantics |
| PEFT (LoRA / QLoRA) | <https://huggingface.co/docs/peft> | `LoraConfig`, adapter merge, `target_modules` choices |
| PEFT source | <https://github.com/huggingface/peft> | adapter-save format, merge details |
| bitsandbytes | <https://github.com/bitsandbytes-foundation/bitsandbytes> | NF4/4-bit quantization for QLoRA on consumer GPUs |
| datasets | <https://huggingface.co/docs/datasets> | JSONL ingestion, streaming, splits |
| Accelerate | <https://huggingface.co/docs/accelerate> | device placement, mixed precision, multi-GPU |

## Vendor-recommended Gemma 3 270M fine-tune walkthroughs

| Source | URL |
|---|---|
| Google AI Devs — full SFT (HF Transformers) | <https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune> |
| Google AI Devs — QLoRA SFT (HF Transformers) | <https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora> |
| Google DevBlog — own-your-AI 270M tutorial | <https://developers.googleblog.com/own-your-ai-fine-tune-gemma-3-270m-for-on-device/> |
| Unsloth — Gemma 3 fine-tune | <https://unsloth.ai/blog/gemma3> |
| Unsloth docs — Gemma 3 how-to | <https://docs.unsloth.ai/models/gemma-3-how-to-run-and-fine-tune> |

## TRL gotchas this repo has hit

- **`DataCollatorForCompletionOnlyLM` was removed in TRL 1.x.** Use the
  prompt-completion dataset shape and set `SFTConfig.completion_only_loss=True`
  to mask user-turn loss without a custom collator. See
  `scripts/finetune.py` header docstring.
- **`max_seq_length` was renamed to `max_length`** in `SFTConfig`.
- **`assistant_only_loss=True` silently no-ops on Gemma 3** because Gemma 3's
  chat template lacks `{% generation %}` markers — the parallel `assistant_only_loss`
  mask is all zeros. Use `completion_only_loss` for Gemma 3.
- **Gemma `token_type_ids` special case** — `SFTTrainer` has explicit handling
  for Gemma's tokenizer; don't strip it in custom collators.

## Recommended LoRA target modules for Gemma 3

| Setting | Value | Source |
|---|---|---|
| `r` | 8 (small data) → 16 (≥ 5k examples) | Google QLoRA tutorial |
| `lora_alpha` | `2 × r` | Standard heuristic; vendor tutorial uses 16 with r=8 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` (attention only) — add `["gate_proj", "up_proj", "down_proj"]` if quality plateaus | Vendor + community |
| `lora_dropout` | 0.05 | Standard for SFT; can drop to 0 with packing on |

## When to consult

| Question | File / URL |
|---|---|
| "Is `completion_only_loss` masking what I think?" | TRL source, `sft_trainer.py` |
| "How do I save just the adapter without merging?" | PEFT docs, `model.save_pretrained` |
| "Why is BF16 OOM on a 16 GB GPU?" | bitsandbytes 4-bit + QLoRA path |
| "What if I want full SFT (no LoRA) on the 270M?" | Google AI Devs full-finetune walkthrough |
