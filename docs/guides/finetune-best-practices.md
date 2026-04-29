# Fine-Tune Best Practices — Gemma 3 270M (and FunctionGemma)

Practical recommendations for SFT + LoRA on Gemma 3 270M class models, with a
clear separation between **vendor recommendations** (Google + Hugging Face)
and **local project decisions** (specific to this repo).

For deeper API references, see [`docs/references/`](../references/README.md).

## 1. Pick the right base

| Use case | Recommended base |
|---|---|
| Domain QA, instruction-following on closed-world data | `google/gemma-3-270m-it` |
| Function-calling / tool-use / on-device agent | `google/functiongemma-270m-it` |
| Plain language modeling, no chat | `google/gemma-3-270m` (non-IT) |

> **Vendor note:** the IT (instruction-tuned) variant already knows the
> `<start_of_turn>` chat template. SFT on top of IT preserves chat-format
> compliance with much smaller datasets than starting from the base PT model.

## 2. Choose SFT strategy by data size

| Dataset size | Strategy | Why |
|---|---|---|
| 10–100 examples (style transfer) | LoRA r=4, 1–3 epochs | Small adapter avoids overfitting; vendor-confirmed minimum |
| 500–5,000 examples (this repo's `sft_v1`) | QLoRA r=8 → 16, 1–2 epochs, completion-only loss | Memory-efficient; preserves general capability |
| 10k+ examples (full domain shift) | Full FT or QLoRA r=32, lower LR | Adapter alone may underfit |

> **Vendor recommendation:** Google's QLoRA tutorial uses `r=8`, `alpha=16`,
> `target_modules` = attention projections only. We follow that and add MLP
> projections only when validation plateaus.
>
> **Local decision:** we use `completion_only_loss=True` (per
> [`docs/references/transformers-trl-peft.md`](../references/transformers-trl-peft.md))
> because Gemma 3's chat template lacks `{% generation %}` markers — the
> parallel `assistant_only_loss=True` path silently masks nothing.

## 3. Hyperparameter starting points

| Param | Value | Source |
|---|---|---|
| `learning_rate` | 2e-4 (LoRA) / 5e-5 (full FT) | Vendor (Google QLoRA) |
| `num_train_epochs` | 1–3 | Local — we monitor val loss; >3 typically overfits 270M |
| `per_device_train_batch_size` | 4 (16 GB GPU) / 8 (24 GB+) | Vendor |
| `gradient_accumulation_steps` | 4 (effective bs 16–32) | Vendor |
| `lr_scheduler_type` | `"cosine"` | Vendor |
| `warmup_ratio` | 0.03 | Vendor |
| `weight_decay` | 0.01 | Vendor |
| `bf16` | `True` (RTX 30/40/50, A100, H100) | Vendor (CUDA-cap-permitting) |
| `max_length` | 1024 (this repo) / 2048 (default) | Local — closed-world health YAML fits in 1024 |
| `packing` | `False` (this repo) / `True` (large dataset) | Local — we want exact loss masks per example |

> **Local note:** the prompt template in `scripts/finetune.py:_to_prompt_completion`
> is the **single source of truth** for training-time prompt shape. `smoke_test.py`
> and `bench_prompt.py` must replicate it exactly; divergence creates a
> tokenization artifact, not a real quality delta.

## 4. Data hygiene

1. **Dedupe on (instruction, output) pairs** — exact case-folded; see
   `gemma_tools.sft_dataset.dedupe_pool`.
2. **Bench-leakage scan** — every held-out evaluation prompt must be removed
   from train+val. See `scan_bench_leakage` and the §4-D2 audit.
3. **Stratified split by class** — preserve class ratios (this repo: fact_lookup,
   fact_absence, domain_refusal, summarization).
4. **Drain guards** — never let a single class drop below 5 examples in any
   split; the splitter raises rather than silently skewing.

## 5. Evaluation order (vendor-shaped, local-anchored)

1. **Loss curves** — train + val loss per epoch. Val should track but stay above train; if val flatlines, stop.
2. **Regex pass-rate** on the held-out bench (`bench_prompt.py` + `bench_eval.py`).
3. **Smoke test** side-by-side vs the base model on 5 prompts (`smoke_test.py`).
4. **Logits-equivalence gate** before deploying a quantized GGUF — same_top_p delta ≤ 1.0 pp, max KL ratio target/ref ≤ 3.0× (`logits_equivalence.py`).
5. **Human rubric (0–3)** on the held-out bench — regex is necessary, not sufficient.

## 6. Quantization and deploy

| Format | When to use |
|---|---|
| HF BF16 | server-side validation, KL reference |
| GGUF BF16 | llama.cpp baseline, KL reference |
| GGUF Q8_0 | quality-priority deploy (slight footprint penalty) |
| GGUF Q4_0 | edge / on-device deploy (this repo's default for SL2619) |
| GGUF Q3_K, Q2_K | only if Q4_0 fails the logits gate too aggressively for your target |

> See [`docs/references/llama-cpp.md`](../references/llama-cpp.md) for the
> verified `convert_hf_to_gguf.py` → `llama-quantize` recipe.

## 7. What to push back on

These recommendations are commonly suggested but we explicitly do **not** use:

- **`assistant_only_loss=True`** for Gemma 3 — silently no-ops (no `{% generation %}` template markers). Use `completion_only_loss=True` instead.
- **Custom data collators** for masking — the prompt-completion dataset shape + `completion_only_loss` covers the case without one.
- **`max_seq_length` arg** in `SFTConfig` — renamed to `max_length` in TRL 1.x.
- **Packing on small datasets** — packing optimizes throughput, not quality; on a few thousand examples the wall-clock saving is dominated by checkpoint I/O.

## 8. References

- Vendor recommendation index: [`docs/references/README.md`](../references/README.md).
- Project plan with concrete chunks: [`docs/plans/a55-gemma-fine-tune.md`](../plans/a55-gemma-fine-tune.md).
- Prompt rule contract: [`docs/conventions/slm-system-prompt.md`](../conventions/slm-system-prompt.md).
