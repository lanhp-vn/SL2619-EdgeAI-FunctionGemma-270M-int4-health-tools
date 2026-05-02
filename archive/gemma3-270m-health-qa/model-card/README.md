# Gemma 3 270M-IT on SL2619 — Best-Practice Analysis

> Ground-truth analysis of Google's Gemma 3 270M-IT as deployed on SL2619 via the vendor-published Torq VMFB. Primary input to [`docs/conventions/16-slm-system-prompt.md`](../../docs/conventions/16-slm-system-prompt.md) and the Phase 1.5 bench strategy in [`docs/plans/models-testing-plan.md`](../../docs/plans/models-testing-plan.md).
>
> Lives next to the model weights by design: when the bundle at `/mnt/sdcard/models/gemma-3-270m-it/` is re-fetched or re-pinned, this doc is the "why and how" that travels with it.
>
> *Authored 2026-04-24 — Phase B post-mortem + Google / HF primary sources.*

---

## 1. Fingerprint

| Property | Value | Source |
|---|---|---|
| Parameters | **270M total** (170M embedding + 100M transformer) | [HF `google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) + [Google Developers blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/) |
| Vocabulary | **262,144 tokens** (Google Developers blog quotes "256k"; `config.json vocab_size: 262144` is authoritative) — oversized relative to params, designed for rare-token coverage + fine-tuning handoff | `references/HuggingFace/gemma-3-270m-it/config.json:53` |
| Context window | **32,768 tokens** (vs 128K for larger Gemma 3 variants) | HF model card |
| Knowledge cutoff | **August 2024** | HF model card |
| IFEval (0-shot) | **51.2** — "establishes a new level of performance for its size"; above SmolLM2-135M-IT and Qwen2.5-0.5B-IT | HF model card + Google blog |
| HellaSwag / PIQA / ARC-c / WinoGrande | 37.7 / 66.2 / 28.2 / 52.3 | HF model card |
| BIG-Bench Hard (few-shot) | 26.7 | HF model card |
| On-device footprint (INT4 QAT) | **~125 MB RAM**; "**0.75% battery for 25 conversations** on a Pixel 9 Pro" | Google Developers blog |
| Size reductions | 77% smaller at INT4, 55% smaller at Q8 (vs FP16) "with virtually no performance penalty" | QAT checkpoints on HF |
| On-SL2619 artifact | `model.vmfb` (bf16) **~516 MiB** (vendor-compiled via `torq-compile :v1.5`; not the INT4 QAT variant) | 2026-04-23 board snapshot §15.4 |
| On-SL2619 CMA peak | ~**520 MiB pinned** under active dispatch (VMFB weights lazy-pinned as ops dispatch) | `docs/conventions/15-model-compiler-runtime.md §2.5`; Phase B bench 2026-04-24 |

**Key takeaway**: the on-SL2619 bf16 VMFB is **4×** the INT4 QAT footprint Google cites. If Phase 2 production needs the INT4 path for CMA headroom, we'd have to compile the QAT checkpoint ourselves via `torq-tools` (no vendor VMFB for INT4 today; deferred until quality numbers justify).

---

## 2. When to use — Google's official positioning

From the Google Developers blog, Gemma 3 270M is designed for:

1. **"High-volume, well-defined task"** — sentiment analysis, entity extraction, query routing, **unstructured-to-structured text processing**, creative writing, compliance checks.
2. **"Every millisecond and micro-cent"** — edge inference where latency + power dominate.
3. **"Rapid fine-tuning experiments"** — find the right config in hours, not days. Small enough that LoRA fine-tunes run in under five minutes on a free Colab.
4. **"Ensure user privacy"** — model runs entirely on-device; no data leaves the box.
5. **"Fleet of specialized task models"** — many cheap task-specific variants beat one expensive generalist.

**Our use case (health-YAML QA on SL2619) is a textbook fit for (1) + (3) + (4).** We have a well-defined structured-text task, we're on edge hardware where latency matters, and the patient data is exactly the kind of sensitive info that belongs on-device.

---

## 3. When NOT to use — the hard limits

From the HF model card's "Ethics and Safety" + "Known Limitations" sections and our own Phase B evidence:

| Anti-pattern | Why it fails | Phase B evidence |
|---|---|---|
| **Multi-turn complex dialogue** | "Not designed for complex conversational use cases" (Google). | — |
| **Open-ended social conversation ("tell me a joke")** | The model either refuses or produces low-quality output that tokenizes poorly. In Phase B, P3 `make me laugh` scored 0/3 (refused outright). | `docs/tmp/bench/2026-04-24_gemma3-summary.md` |
| **Factual recall of world knowledge** | "Not knowledge bases. May generate incorrect or outdated factual statements." Training cutoff Aug 2024. | — |
| **Common-sense reasoning / sarcasm / nuance** | "Rely on statistical patterns… may struggle to grasp subtle nuances." | — |
| **Long-context tasks** | 32K context; long docs spanning > 20K tokens blow past attention budget. | — |
| **Hallucination-sensitive tasks without grounding** | Without retrieved context in the prompt, the model confabulates to fill turns. In Phase B, P5 `summarize my current health status` **hallucinated** vitals that weren't in the mocked table. | `docs/tmp/bench/2026-04-24_gemma3-summary.md` — P5 scored 0/3 for invented values |

**The load-bearing pivot for Phase 1.5**: give the model **no freedom to invent**. All factual content must come from the YAML in the prompt; the model's job is to *rephrase* and *route*, not *recall*.

---

## 4. Chat template — literal tokens (source-verified)

From [Gemma prompt-structure docs](https://ai.google.dev/gemma/docs/core/prompt-structure) + [`references/Synaptics/torq-examples/gemma3/src/runner.py:155-178`](../../references/Synaptics/torq-examples/gemma3/src/runner.py):

```
<start_of_turn>user
{content}<end_of_turn>
<start_of_turn>model
```

**Roles are `user` and `model` only** — there is **no `system` role**. Our `tools/src/sl2619_tools/prompt_composer.py:77-78` already emits this exact template; no change needed.

### 4.1 Where does the "system prompt" go?

Embedded **inside the first user turn**, before the actual user utterance, separated by a blank line:

```
<start_of_turn>user
{system_instructions}
{retrieved_context}

{user_query}<end_of_turn>
<start_of_turn>model
```

This is the pattern Google's docs explicitly recommend when the caller has system-level instructions to inject. Our composer already does this (`prompt_composer.py:76` — `full = sys + "\n" + utterance`).

### 4.2 Why the 262k vocab matters for YAML injection

Gemma 3's 262,144-token vocabulary (`config.json` authoritative; Google marketing rounds to "256k") was designed to reduce tokenization waste on rare tokens and domain-specific vocabulary (drug names, medical units, etc.). For health YAML, this means:

- Strings like `"Lisinopril 10mg"`, `"SpO2 98%"`, `"36.7°C"` usually tokenize in **1–3 tokens** each, not 6–10 as with a 32k-vocab model.
- Dates (`2026-04-24`) tokenize tightly.
- The YAML's own structure (colons, hyphens, indentation) is cheap to encode.

Net effect: we can pack **more facts per prompt** than with a same-size model using a smaller vocab. That's leverage for the closed-world QA strategy.

---

## 5. System-prompt strategy — directive/keyword, not prose

The 270M model follows **terse directive prompts better than polite prose**. Two orthogonal reasons:

1. **Per-query prompt-token cost.** Every token in the system prompt is re-processed on every query (unless KV-cached across turns, which our single-shot bench is not). A 200-token system prompt at ~100 tok/s prefill on SL2619 costs 2 s of TTFT before the user's token even lands.
2. **Instruction-following reliability.** IFEval at 51.2 is strong-for-size but still **fails ~49% of verifiable-instruction cases**. Directive format (`ROLE:`, `TASK:`, `RULES:`, explicit format example) consistently beats prose in published SLM prompt-engineering work because it's closer to the structured training signal.

### 5.1 Recommended system-prompt shape (for closed-world health YAML QA)

```
ROLE: health-records assistant for patient on SL2619 edge device.
INPUT: YAML block below is the single source of truth.
TASK: answer the user's question using ONLY facts in YAML. Quote values verbatim.
RULES:
- if YAML lacks the answer: reply "not in record".
- never invent values, dates, medications, or doses.
- never give medical advice; re-route: "consult your clinician".
- refuse off-topic / social chat: reply "I answer questions from your health record only".
FORMAT: 1-2 sentences. No lists unless YAML has them.
DATE: {date_iso}
YAML:
{yaml_block}
```

Placeholder tokens: ~80–120 depending on YAML size. Not a prose paragraph; every line is a machine-parseable directive.

**Why this shape works for 270M specifically**:

- `ROLE:` / `TASK:` / `RULES:` / `FORMAT:` labels align with Gemma's instruction-tuning data distribution.
- `ONLY facts in YAML` + `never invent` is an explicit hallucination guard — the exact failure mode Phase B hit.
- Closed-vocabulary refusal strings (`"not in record"`, `"consult your clinician"`) give the model predictable fall-back language instead of freelancing.
- `FORMAT: 1-2 sentences` sets a hard output-length cap that keeps TTFT + total-decode latency in a usable range.

### 5.2 What we deliberately do NOT put in the prompt

- **No few-shot examples.** Exemplars eat prompt budget; IFEval at 51.2 on 0-shot says the model follows direct directives without them for tasks this constrained.
- **No persona / style / "friendly" adjectives.** 270M wastes parameters parsing politeness; directive tone is cheaper.
- **No multi-language hedges.** Gemma 3 270M's safety testing was English-only (HF model card); we commit to English-only output.
- **No JSON schema output specification.** Natural-language answer is the target per P4/P5 of the bench; JSON emission is a future Phase-2 decision if we wire the model output into a downstream parser.

---

## 6. Context-budget discipline (32K, but practically far less)

Total 32,768-token window. Our budget per turn:

| Component | Expected tokens | Note |
|---|---|---|
| System prompt (§5.1 shape) | 80–120 | fixed per turn |
| YAML block (§3 expanded schema, typical patient) | 200–500 | grows with meds/appointments |
| Chat-template markers | 10 | `<start_of_turn>user/model`, newlines |
| User utterance | 20–50 | varies; health questions are short |
| **Prompt total** | **~310–680** | well below 32K |
| Generation cap | **128 max** | set by bench harness (`--max-gen-tokens 128`) |
| **Turn total** | **~440–810** | |

On SL2619 with ~1.7 tok/s observed sustained (Phase B), a 128-token generation is ~75 s of decode — dominated by HAL dispatch overhead per `docs/conventions/15-model-compiler-runtime.md §2.5`. TTFT is ~1–3 s of prompt prefill on a cold KV. **Generation cap matters more than prompt size** for wall-clock latency.

If a YAML grows past ~2000 tokens (patient with 20+ medications, multi-year appointment history), we'd need to either (a) trim the YAML to what's likely-relevant before composing the prompt, or (b) pre-tokenize the YAML once and reuse a warm KV cache. Not an issue for Phase 1.5; flagged for Phase 2 coordinator.

---

## 7. Quantization — the three paths, ground truth

SL2619 has two realistic LLM runtimes: the **Torq NPU** (our current on-device path, BF16-only in practice for Gemma 3 today) and **A55 CPU** via llama.cpp or onnxruntime (smaller-footprint path, different backend). Google's QAT release is written for the CPU-side ecosystem.

### 7.1 What the three pinned submodules actually are

| HF path (`references/HuggingFace/`) | On-disk dtype | QAT-trained? | Intended downstream |
|---|---|---|---|
| `gemma-3-270m-it` (Synaptics fork) | BF16 + vendor VMFB | No | Torq NPU path — the VMFB we run today |
| `gemma-3-270m-it-qat-q4_0-unquantized` (google/) | BF16 safetensors | Yes (instruct, ~5000 teacher-distilled steps) | User-side llama.cpp Q4_0 conversion |
| `gemma-3-270m-qat-q4_0-unquantized` (google/) | BF16 safetensors | Yes (base / pre-trained) | Domain SFT studies only |

All three are architecturally identical (`config.json` byte-match aside from `torch_dtype` / `transformers_version`). "QAT" is a **weight-distribution property**, not a storage format. Google's model card is explicit: *"The checkpoint in this repository is unquantized, please make sure to quantize with Q4_0 with your favorite tool."*

### 7.2 The INT4 runtime path — llama.cpp CPU, not Torq NPU

There is no Google-published INT4 GGUF for 270M (only 1B/4B/12B/27B). To get INT4 on-device for this model size, we convert ourselves:

```bash
# Host side — WSL-friendly (< 2 GiB peak RAM, unlike Torq iree-compile)
python llama.cpp/convert_hf_to_gguf.py \
  references/HuggingFace/gemma-3-270m-it-qat-q4_0-unquantized \
  --outfile gemma-3-270m-it-qat-bf16.gguf
llama.cpp/build/bin/llama-quantize \
  gemma-3-270m-it-qat-bf16.gguf gemma-3-270m-it-q4_0.gguf Q4_0
# Result: ~125 MB GGUF (file size, confirmed via ls -l); runtime RAM and
# tok/s on SL2619 A55 are not yet measured (Google's 125 MB figure was
# Pixel 9 Pro).
```

**This path changes the backend** — llama.cpp runs on A55 CPU, not the Torq NPU. Trade: CMA pressure drops (frees the ~516 MiB BF16 VMFB), on-process RAM rises to ~125 MiB (file-size-equivalent baseline; actual A55 runtime TBD). Loses the NPU offload. Latency/throughput impact must be measured if Phase 2 triggers the switch.

### 7.3 The INT4 Torq/IREE path — doesn't exist for Gemma 3 today

No vendor-published INT4 VMFB for 270M. Torq compiler's Q4_0 lowering for Gemma 3 is not in `release_notes.md` verified-models (verified by grep 2026-04-24). Compiling `*-qat-q4_0-unquantized` through `torq-compile` yields a BF16 VMFB identical in size to compiling plain `gemma-3-270m-it` — the compiler dequantizes weights during lowering; INT4 only saves on-board memory if the target runtime preserves the quantization. **Do not treat "compile the QAT checkpoint to a smaller VMFB" as a real option without a vendor lift.**

### 7.4 Decision rule (unchanged headline, corrected mechanism)

If bf16 + Phase 1.5 benchmarks pass G_QUALITY, don't chase INT4. If Phase 2 hits CMA headroom pressure and we do chase it, **the shortest path is llama.cpp Q4_0 on A55 CPU** (§7.2), not a Torq recompile. Related: `docs/conventions/15-model-compiler-runtime.md §2.9` verified-working models table.

---

## 8. Fine-tuning path — start from plain IT, not QAT

Google's pitch: LoRA fine-tune in "hours, not days" on a free Colab. Trigger for us: Phase 1.5 shows systematic failures the prompt (§5) can't fix.

### 8.1 Which checkpoint to start from

**Use plain `gemma-3-270m-it`, not the QAT variants.** Three reasons:

1. Google's own 270M fine-tune guide ([ai.google.dev/gemma/docs/core/huggingface_text_full_finetune](https://ai.google.dev/gemma/docs/core/huggingface_text_full_finetune)) uses `google/gemma-3-270m-it` as the base.
2. Google's tune overview states verbatim: *"Tools for tuning quantized models are limited … Typically, you must fine-tune a model like Gemma at full precision, then quantize the resulting model."*
3. No Google-documented workflow preserves QAT robustness through BF16 SFT. The QAT distribution is only ~5000 teacher-distilled steps; a few hundred domain SFT steps will overwrite it. "Continued QAT" is not a documented path.

The QAT submodules are kept as **Phase-2 comparison baselines**, not SFT starting points.

### 8.2 Canonical recipe (Google-endorsed)

| Step | Tool | Runtime |
|---|---|---|
| Authoring / SFT | HF Transformers + TRL `SFTTrainer`, or **Unsloth Colab** (Google-endorsed partner, 1.6× faster / 60% less VRAM) | Free Colab T4 (16 GiB VRAM sufficient for full BF16 fine-tune at 270M per Google's guide) |
| Official LoRA config | `r=16, lora_alpha=16, lora_dropout=0.05, target_modules='all-linear', modules_to_save=['lm_head','embed_tokens'], ensure_weight_tying=True` | |
| Data format | `{"messages":[{"role":"user","content":"<YAML+question>"},{"role":"assistant","content":"<answer>"}]}` — Gemma 3 has **no `system` role**, prepend YAML+directives into turn 1 of `user` (matches `16-slm-system-prompt.md`) | |
| Sample count | Google's demo works with **25 SFT pairs**; ~100-300 domain pairs is a realistic Phase-2 budget | |

### 8.3 Export back to SL2619

Two options after `trainer.save_model()` produces merged safetensors:

| Option | Target backend | Host cost | On-device footprint | Viability |
|---|---|---|---|---|
| **A — BF16 Torq VMFB** (current stack) | NPU | ~30 GiB peak `iree-compile` (needs ≥ 48 GiB Linux server — WSL will OOM per `15-model-compiler-runtime.md §2.6`) | ~516 MiB CMA (same as today) | Requires Linux server; vendor-compile pattern for a non-vendor checkpoint is unverified |
| **B — llama.cpp Q4_0 GGUF** | A55 CPU | < 2 GiB convert (WSL-friendly) | ~125 MiB RAM (file-size estimate; SL2619 runtime TBD) | Mainlined in llama.cpp; changes runtime stack |

**Default when the trigger fires**: Option B (llama.cpp). Faster iteration loop on WSL, lower risk of vendor-tool-chain breakage, smaller on-device footprint. Option A is the performance path for after Option B proves the fine-tune behaves correctly.

### 8.4 When not to fine-tune

Don't fine-tune until benchmarks show prompt-engineering alone (§5) is insufficient. Cost profile for Option B: ~2-8 hours Colab + ~30 min llama.cpp convert/quantize. Overkill if the prompt strategy gets us to G_QUALITY.

Frameworks (from Google's own 270M blog): **Hugging Face transformers**, **Unsloth** (partner), **JAX**. HF + Unsloth is the smallest-blast-radius path.

---

### 8.5 As-executed SFT recipe — Phase 2 (2026-04-28)

This section records the **actual recipe used**, including deviations from Google's canonical notebook that were required for this model on this stack. Full plan: [`docs/plans/AI-models/a55-gemma-fine-tune.md`](../../docs/plans/AI-models/a55-gemma-fine-tune.md).

#### 8.5.1 Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (15.47 GiB VRAM, sm_120 / Blackwell) |
| CUDA | 12.8 (torch cu128 wheels) |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.6.2 |
| trl | 1.3.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |
| datasets | 4.8.4 |

#### 8.5.2 Dataset

| Split | File | Rows | sha256 |
|---|---|---|---|
| train | `tools/data/sft_v1.train.jsonl` | 1023 | `6699ee41…` |
| val | `tools/data/sft_v1.val.jsonl` | 126 | `b6443d7d…` |
| test (held out) | `tools/data/sft_v1.test.jsonl` | 110 | — |

Data format per row: `{"messages":[{"role":"user","content":"<system-prompt+YAML+question>"},{"role":"assistant","content":"<answer>"}]}` (Path B — directive system prompt folded into user turn; no `system` role per Gemma 3 template). Generated via `uv run sft-build` from `tools/data/clean_sft_dataset.json` (1400 raw Alpaca-shape triples, chatbot-distilled).

At SFT time, each row is converted to `{"prompt": <user-turn with `add_generation_prompt=True`>, "completion": <assistant text>}` by `_to_prompt_completion()`. The prompt ends with `<start_of_turn>model\n`; trl's `completion_only_loss=True` masks prompt tokens so the loss is computed on the ~10-80 token assistant answer only. Longest sample: 930 tokens prompt+completion; max_length=1024 provides headroom.

#### 8.5.3 LoRA config (as-executed)

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",   # resolved: q/k/v/o/gate/up/down_proj (7 modules)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # modules_to_save NOT set — see deviation below
)
```

**Deviation from Google canonical notebook**: `modules_to_save=["lm_head","embed_tokens"]` was specified in the original plan but removed before training. Reason: Gemma 3 has `tie_word_embeddings=True`; peft splits the tied `lm_head/embed_tokens` pair into two independent full-precision copies (~167M params each), producing a corrupt vocabulary projection in the resulting adapter that blocks `merge_and_unload()` → GGUF conversion. Additionally, 1023 examples is insufficient to retrain 167M embedding params without catastrophic forgetting — the IT model's English health-term embeddings are already correct. Pure LoRA on attention + MLP projections is the right behavioral-fix surface for definitional drift.

Result: **trainable params = 3,796,992 / 271,895,168 total = 1.3965%** (peft tied-weight warning absent; merge path to GGUF intact).

#### 8.5.4 SFT config (as-executed)

```python
SFTConfig(
    output_dir="./adapters_v1",
    num_train_epochs=3,
    per_device_train_batch_size=1,    # was 4 in plan — OOM at step 0 (see below)
    gradient_accumulation_steps=16,   # was 4 — preserves effective batch 16
    per_device_eval_batch_size=1,     # not in plan — HF default 8 OOMs at epoch-end eval
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    max_length=1024,
    gradient_checkpointing=False,
    packing=False,
    completion_only_loss=True,        # trl 1.3.0 API — DataCollatorForCompletionOnlyLM removed
    optim="adamw_torch_fused",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="tensorboard",
    seed=42,
)
```

**OOM root cause and fix**: The plan estimated ~1-1.5 GiB peak VRAM for a 270M model, overlooking that `vocab_size=262,144` dominates the loss-head activations. At `per_device_train_batch_size=4`, `seq=1024`, BF16: the `outputs.logits` tensor = `4 * 1024 * 262144 * 2 = 2.0 GiB`, plus `logits[..., :-1, :].contiguous()` in the SFT loss path materializes another ~2 GiB peak. First training attempt OOM'd at step 0: `Tried to allocate 3.66 GiB ... 11.72 GiB already in use`. Fix: `per_device_train_batch_size=1` + `gradient_accumulation_steps=16` — drops logits 4× to 512 MiB, preserves effective batch=16, leaves ~5 GiB headroom.

**Rule for future fine-tunes on this model**: always estimate logits memory as `PDB * max_length * vocab_size * dtype_bytes` before choosing batch size. For Gemma 3 270M at seq=1024/BF16, each PDB unit costs ~512 MiB in the loss head alone. Set `per_device_eval_batch_size=1` explicitly — HF's default of 8 is independent of train PDB and will OOM.

**trl 1.3.0 API changes** (not in Google's 2025 notebook, which targets trl 0.x):
- `DataCollatorForCompletionOnlyLM` was removed → use prompt-completion dataset shape + `completion_only_loss=True`
- `max_seq_length` renamed to `max_length`
- Gemma 3 chat template has no `{% generation %}` markers → `assistant_only_loss=True` (trl 1.x parallel API) silently returns all-zero masks; `completion_only_loss` is the supported path

#### 8.5.5 Training results

Script: `tools/scripts/finetune.py` (deployed to server; sha256 `1b9e0a160…`). Training command:

```bash
cd ~/sl2619-finetune && source .venv/bin/activate && \
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
  LOG=~/sl2619-finetune/logs/train-$(date +%Y%m%d-%H%M%S).log && \
  python finetune.py 2>&1 | tee "$LOG" && echo "LOG: $LOG"
```

Log: `~/sl2619-finetune/logs/train-20260428-064801.log`

| Epoch | train_loss | eval_loss | eval_mean_token_accuracy | eval_entropy |
|---|---|---|---|---|
| 1 | 1.326 | 0.9697 | 0.7613 | 0.9664 |
| 2 | 0.7793 | 0.7983 | 0.7978 | 0.8305 |
| 3 | 0.6277 | **0.6936** | **0.8152** | 0.6528 |

- Aggregate `train_loss`: 0.911 / `train_runtime`: 326.4 s (5.4 min) / 192 steps / 9.402 samples/s
- T3 gate: eval_loss strictly monotone-decreasing 3/3 ✅; train < eval × 1.5 (0.6277 < 1.040) ✅; train < eval by ~10% (good generalization gap, no overfit) ✅; no OOM ✅
- `grad_norm` stable 4.2–6.2 (no gradient explosion); `entropy` decreases 1.352→0.800→0.615 (model gains confidence)

#### 8.5.6 Adapter artifacts (server)

| Path | Contents | Size |
|---|---|---|
| `~/sl2619-finetune/adapters_v1/` | Final adapter (= checkpoint-192): `adapter_model.safetensors` (7.6 MB), `adapter_config.json`, `chat_template.jinja`, `tokenizer.json` (33 MB), `tokenizer_config.json`, `README.md`, `training_args.bin` | 201 MB total |
| `adapters_v1/checkpoint-64/` | Epoch 1 adapter + optimizer state | ~54 MB |
| `adapters_v1/checkpoint-128/` | Epoch 2 adapter + optimizer state | ~54 MB |
| `adapters_v1/checkpoint-192/` | Epoch 3 adapter + optimizer state (= final) | ~54 MB |

Best-eval checkpoint: `checkpoint-192` (final epoch, lowest eval_loss 0.6936). `trainer.save_model()` writes to `adapters_v1/` (top-level), which byte-matches checkpoint-192. T4 merge consumes this directly.

#### 8.5.7 T4 → Q5 closure (2026-04-28)

Phase 2 T4-T5 + Phase 3 Q0-Q5 closed in two sessions. Headline numbers:

| Stage | Result | Source-of-truth |
|---|---|---|
| **T4** merge | `merged_v1/` 536 MB BF16 HF dir; smoke shows `'You are currently running at 72 beats per minute.'` for P1-style prompt (vs base YAML echo) | server `~/sl2619-finetune/logs/merge-20260428-071112.log` |
| **T5** server smoke (BF16, transformers, GPU) | base 0/5 → merged **4/5** real PASS on P3/P6/D1/S1; P1 caveat (`<eos>` first token on literal "current heart rate" — out-of-distribution training coverage) adjudicated as v1 dataset gap, NOT a regression | [`docs/tmp/bench/t5-smoke-20260428-072748.md`](../../docs/tmp/bench/t5-smoke-20260428-072748.md) |
| **Q0** convert + Q4_0 | `merged_v1.bf16.gguf` 518 MiB, `merged_v1.q4_0.gguf` 231 MiB / sha `587f1af6…`; **gotcha**: convert_hf_to_gguf.py:1238 asserts `max(tokenizer.vocab.values()) < vocab_size` and Gemma 3 has `len(vocab)=262145` vs `vocab_size=262144` → fails on the BPE-fallback path. Fix: pull `tokenizer.model` from HF Hub so convert takes the SentencePiece path (no such assertion). Tracked at [`docs/plans/backlogs.md §1.22`](../../docs/plans/backlogs.md). | server `~/sl2619-finetune/logs/q0-20260428-084616.log` |
| **Q1** logits equivalence | same-arch x86 Path B at n_ctx=2048 = 98.443% same_top_p (gate ≥ 95%, only 1.05 pp below the apples-to-apples base anchor 99.489% — fine-tune cost expected from SFT peakedness `entropy 1.352→0.615`); cross-arch H5R-shape Δ = 0.393 pp / ratio 0.996× (gates ≤ 1.0 pp / ≤ 3.0×). Both gates pass with substantial headroom; A55 NEON DOTPROD + REPACK is not silently corrupting the SFT delta. **Memory cliff documented**: n_ctx=2048 OOM-kills on 1.87 GiB / no-swap board (per-chunk reference-logits buffer = 2.15 GiB at vocab=262144) — Q1 cross-arch step uses n_ctx=256 (1.20 GiB fit). | [`docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md`](../../docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md) + [`2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](../../docs/tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md) |
| **Q2** transfer | `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf`, sha verified `587f1af6…` | `/board_probe --target=sl2619` |
| **Q3** smoke | board emits `'72 bpm.'` first content on the Path-B-shaped P1 prompt with `--jinja --no-display-prompt` (definitional drift fixed at deployment shape). Anecdotally **closes the T5 P1 OOD-`<eos>` caveat** at the Q4_0 envelope — quantization noise perturbs the BF16-greedy `<eos>` mass back into a recoverable state. | `.cache/q3/q3e-jinja-nodisplay-*.log` |
| **Q4** full bench | 15-prompt sweep, 7m 48s wall, **8/15 regex PASS** vs H6 base 2/15. Decode aggregate **17.29 tok/s** (1.82× faster than H6's 9.50 tok/s). | [`docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.{jsonl,log}`](../../docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.jsonl) |
| **Q5** score | manual rubric ≥ 2 (real grounded answer): **5/15** — P1, P7, P9, A1, S1 (vs H6's 0/15). Plan §9 ≥ 80% target NOT met; quality ceiling is dominated by training-pool gaps, NOT Q4_0 noise. | [`docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md`](../../docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md) |

**Three load-bearing findings beyond the headline numbers:**

1. **Deployment shape requires `--jinja --no-display-prompt`** — the existing `bench_prompt.py.LlamaCompletionBenchAdapter` text-wraps with literal `<start_of_turn>…` markers. llama.cpp without `--jinja` tokenizes these as plain bytes (~5-10 sub-tokens each) instead of the special control tokens (105/106) the model was trained on; the FT'd model never enters answer mode under that envelope and emits hallucinations (Q3b `'108<h4>You can also try…'`). With `--jinja` llama.cpp applies the model's chat-template metadata internally — special tokens land at the right ids and the SFT delta materializes. **For Q4 we built `tools/src/sl2619_tools/bench_remote.py`** (host-driven SSH-piped, R3-compliant) using the working envelope. This is also why `tools/scripts/chat_remote.sh` exists — it's the smallest reproducer of the right envelope.
2. **Repetitive degeneration is the new failure mode** — 14/15 Q4 prompts emit a correct (or refusal-shaped) first token, then loop the phrase `not in record.` until the n_predict cap. Greedy/top-k=1/temp=0 can't escape the loop without a learned `<eos>` from the model. The v1 SFT corpus's strong positive bias toward `not in record` as a fallback completion produced this attractor. Backlog item: train with terminator-rich completions (explicit `<end_of_turn>` after every answer), or ship at inference with `top-k 5` + small temp to break the loop.
3. **Q4_0 quantization survives the SFT delta** — Q1's 98.443% same-arch / 0.393 pp cross-arch numbers predicted this; Q4 confirmed it behaviorally. The 9-pp regex delta and 5-prompt rubric delta vs H6 are entirely the SFT contribution, not envelope-or-quant artifacts.

#### 8.5.8 Working deployment recipe (one-liner reproducer)

For ad-hoc questions against the FT'd Q4_0 on the live SL2619 (READ-ONLY SSH; no remote writes), use:

```bash
tools/scripts/chat_remote.sh "what is my heart rate?"
# → "72 bpm." then degeneration to "not in record." (greedy/top-k=1 attractor; ignore tail)
tools/scripts/chat_remote.sh "summarize my current medications"
# → ":\n- Lisinopril 10 mg 08:00 blood pressure control.\n- Metformin 500 mg…"
```

The script renders the §4 directive system + YAML record + question via `prompt_composer.compose_user_text`, pipes the body to the board's `llama-completion --jinja --no-display-prompt -p $BODY -no-cnv --single-turn` over SSH stdin. Output is the model reply only (`--no-display-prompt` suppresses the prompt echo). Override `N_PREDICT` / `SEED` / `TEMP` / `LLAMA_MODEL` etc. via env vars.

The `bench_remote` Python harness (`uv run bench-remote --ssh-host nouslogic-sl2619 --prompts data/prompts.yaml --health-table data/health_table_v1.yaml --output … --llama-binary /mnt/sdcard/llama-cpp/llama-completion --llama-model /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf …`) sweeps the full prompt suite using the same envelope; this is what produced the Q4 numbers above. Source at `tools/src/sl2619_tools/bench_remote.py` (290 LOC + 15 unit tests in `tools/tests/test_bench_remote.py`).

---

## 9. On-SL2619 deployment specifics

This section points at canonical conventions; do not duplicate:

| Topic | Canonical file |
|---|---|
| Compile-tag coupling (`:v1.5` ↔ ASTRA SDK 2.3) | `docs/conventions/15-model-compiler-runtime.md §2.2` |
| Torq runtime wheel numpy 1.x ABI | `docs/conventions/15-model-compiler-runtime.md §2.3` |
| NPU session lifecycle (reboot required between sessions) | `docs/conventions/15-model-compiler-runtime.md §2.5` |
| SD-backed model storage + tmpfs Python env symlinks | `docs/conventions/15-model-compiler-runtime.md §5` |
| On-board harness adaptation points (vendor runner quirks) | `docs/conventions/15-model-compiler-runtime.md §2.5` |
| Phase B post-mortem (what failed, what we learned) | `docs/plans/backlogs.md §1.19` |
| Vendor VMFB SHA-pin | `docs/tmp/sl2619-status.md §15.4` |
| FT'd Q4_0 deployment-shape envelope (`--jinja --no-display-prompt`) | `docs/get-started/gemma-on-a55-get-started.md §8` + this doc §8.5.8 + `tools/scripts/chat_remote.sh` |
| Host-driven R3-compliant on-board bench harness | `tools/src/sl2619_tools/bench_remote.py` (replaces text-wrap envelope from `bench_prompt.py.LlamaCompletionBenchAdapter` for FT'd model deployment) |

**Single-process rule** (`§11.2`): one Python process holds the `Gemma3Static` / `ManagedSelfAttnCacheRunner` for the full batch. Spawning a Python process per query triggers `failed to start network via IOCTL` within 2–3 cycles and requires reboot to clear. Any Phase 2 A55 service consuming Gemma3 MUST be a long-lived daemon, not a CGI-style per-request spawn.

---

## 10. Success criteria (what "Gemma 3 270M works for us" means)

Phase 1.5 Phase C (redefined for this pivot, see `docs/plans/models-testing-plan.md`):

1. **YAML-grounded factual retrieval** — every question answerable from the YAML gets the correct value, verbatim, with < 5% fabrication across ≥ 40 gold Q&A pairs.
2. **Off-topic rejection** — non-health questions (`"tell me a joke"`, `"what's the weather"`) hit the refusal string from §5.1 RULES ≥ 90% of the time.
3. **On-device latency** — TTFT ≤ 5 s; total decode for a 1-2 sentence answer ≤ 30 s on idle board.
4. **No CMA leak** — post-process-exit `CmaFree` returns within 50 MiB of pre-load baseline (partial retention acknowledged per `§11.1`).
5. **Repeatability** — 3-run variance on any single prompt ≤ 25% on TTFT + tok/s (no pathological warm-up outliers once the single-process rule is honored).

If all five hold: Gemma 3 270M is the Phase 1.5 winner and Phase 2's coordinator is cleared to integrate it. If factual retrieval fails: revisit prompt strategy § before chasing a larger model. If off-topic rejection fails: either tune the refusal directive or add a lightweight Python pre-filter (classify on-topic vs off-topic before invoking the SLM at all).

---

## 11. Sources

Primary:
- [Introducing Gemma 3 270M — Google Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [Gemma formatting and system instructions — ai.google.dev](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [google/gemma-3-270m-it on Hugging Face](https://huggingface.co/google/gemma-3-270m-it)
- [Synaptics/gemma-3-270m-it on Hugging Face](https://huggingface.co/Synaptics/gemma-3-270m-it) (vendor VMFB + sidecars)

Repo-local:
- [Phase B bench summary (2026-04-24)](../../docs/tmp/bench/2026-04-24_gemma3-summary.md) — empirical failure modes that drove this analysis.
- `docs/plans/backlogs.md §1.19` — Phase B post-mortem + W1–W7 working pathways.
- [Torq runtime convention](../../docs/conventions/15-model-compiler-runtime.md) — all on-board behavior gotchas.
- [Gemma 3 demo runner (vendor)](../../references/Synaptics/torq-examples/gemma3/src/runner.py) — chat template + KV-cache reference implementation.

---

## 12. HF submodule map (pinned 2026-04-24)

Three Gemma 3 270M variants are pinned under `references/HuggingFace/` (LFS-skipped — configs + tokenizer + chat template only; weights fetched on demand via `hf download`):

| Submodule | Upstream | Purpose | Gated? |
|---|---|---|---|
| `gemma-3-270m-it` | `Synaptics/gemma-3-270m-it` | **Production path.** Source of truth for the vendor-built BF16 VMFB running on-device today. | No |
| `gemma-3-270m-it-qat-q4_0-unquantized` | `google/gemma-3-270m-it-qat-q4_0-unquantized` | **Phase-2 INT4-CPU candidate.** Feed to llama.cpp `convert_hf_to_gguf.py` → Q4_0 GGUF (§7.2) if CMA pressure forces the NPU→CPU switch. | Yes (Gemma license) |
| `gemma-3-270m-qat-q4_0-unquantized` | `google/gemma-3-270m-qat-q4_0-unquantized` | **Reference only.** Base (non-instruct) QAT; we'd fine-tune from plain IT, not this (§8.1). Kept for completeness / QAT-quality baseline studies. | Yes (Gemma license) |

For fetching the actual weights on a host (needed for Phase-2 llama.cpp convert or SFT):

```bash
pipx run --spec huggingface-hub hf download \
  google/gemma-3-270m-it-qat-q4_0-unquantized \
  --local-dir ~/hf-cache/gemma-3-270m-it-qat-q4_0-unquantized
```

See `get-started/hugging-face-get-started.md` §9 for the full on-demand fetch pattern.
