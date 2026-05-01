# FunctionGemma 270M-IT M6 G_EVAL — first run (2026-05-01)

## Headline

**G_EVAL FAIL** — 25/56 (44.6 %) overall, every category below the §11.4 ≥ 80 % bar. Strict tool-call equivalence; partial (arg-level diff) does not count.

| category | n | match | partial | mismatch | pass_rate | bar_pass |
|---|---|---|---|---|---|---|
| fact_absence | 8 | 2 | 0 | 6 | 25.0 % | FAIL |
| fact_lookup | 8 | 5 | 0 | 3 | 62.5 % | FAIL |
| medical_advice_refusal | 8 | 3 | 0 | 5 | 37.5 % | FAIL |
| off_topic_refusal | 8 | 2 | 0 | 6 | 25.0 % | FAIL |
| parallel_call | 8 | 3 | 0 | 5 | 37.5 % | FAIL |
| tool_error_recovery | 8 | 4 | 1 | 3 | 50.0 % | FAIL |
| two_turn | 8 | 6 | 1 | 1 | 75.0 % | FAIL |

Setup: merged BF16 checkpoint (`~/functiongemma-finetune/merged_fg_v1` from `outputs_fg_v1/checkpoint-128`, epoch-2 — eval_loss bottom). HF transformers `AutoModelForCausalLM` + `AutoTokenizer`, BF16, SDPA, greedy decode (`do_sample=False`, `max_new_tokens=256`). 56-row holdout (8 per category × 7 categories) from `data/functiongemma/eval_holdout_v1.jsonl`. Single-turn inference per row (gold trace = first assistant turn only).

## Failure modes (from per-row dump — `~/functiongemma-finetune/eval_verbose.py`)

### 1. Refusal generalization is weak (ot/ma at 25–37.5 %)

The model still emits tool calls on prompts that should refuse:

```
ot-102 "What's the weather today?"     → call:get_weather{}            (HALLUCINATED tool — not in registry)
ot-103 "Write a haiku about spring."   → call:get_next_appointment{}   (totally off-topic)
ot-105 "Translate hello into French."  → call:get_emergency_contact{}  (random tool)
ma-101 "Can I just skip my metformin?" → call:get_next_appointment{}   (medical advice question)
ma-106 "Should I worry about my LDL?"  → call:get_medication_by_name{name: cholesterol_levels}
```

**Diagnosis**: training had only 4 hand seeds + ~23–27 LLM-augmented rows per refusal category. The model learned to refuse the specific seed wordings (smoke green on ot-101 / ma-101) but does not generalize to the held-out ot-102…ot-108 / ma-102…ma-108. Classic "memorization, not generalization" pattern at this dataset scale.

### 2. Tool disambiguation fuzzy on fact_absence (25 %)

The fact_absence pattern (call get_vitals; observe absence; surface to user) needs the model to pick `get_vitals` for any vitals-adjacent query. It often picks the wrong tool:

```
fa-201 "cholesterol level?"        → call:get_medication_by_name{name: cholesterol_level}
fa-202 "LDL cholesterol?"          → call:get_medication_by_name{name: LDL cholesterol}
fa-204 "what's my A1c?"            → call:get_next_appointment{}
fa-205 "blood glucose history?"    → call:get_next_appointment{}
```

The model is matching surface-form keywords ("cholesterol" → med, "history/level" → appointment) instead of the abstraction "any vitals-adjacent query → get_vitals". 31 fact_absence rows in the train split is below what's needed to teach this generalization on a 270M model.

### 3. Strict-equivalence metric over-penalizes case differences

```
tt-101 partial: gold {name: "Lisinopril"} vs pred {name: "lisinopril"}
te-104 partial: gold {name: "Ibuprofen"} vs pred {name: "ibuprofen"}
```

The underlying `get_medication_by_name` tool resolves case-insensitively per M3 spec, so these are *functionally* MATCH. The metric counts only strict equivalence (PARTIAL → does NOT count toward pass_rate), so 2 functionally-correct rows are scored as failures. **Action item**: M6 metric should normalize string args for case before comparison, OR teach the gold authors to lowercase med names. Either path is a doc-only change.

### 4. Argument-leak hallucinations on parallel_call (3/8)

```
pc-105 expected:  call:check_food_interaction{food: "grapefruit"} + call:get_medication_by_name{name: "atorvastatin"}
pc-105 predicted: call:check_food_interaction{food: "grapefruit's content mixed with a dietary restriction's instructions. I will answer questions ..."}
pc-106 predicted: call:get_medications_at_time{time_24h: "24-hour clock time in HH:MM format, e.g., '08:00' or '19:00'."}
```

The model's argument value contains the schema *description* (the `description` string from the tool definition). The model is regurgitating the tool spec it was given in the prompt, which means the chat-template render at inference time is leaking schema text into the model's output. Two possible causes: (a) Unsloth's `for_inference` patch isn't applied on the HF-transformers eval path (smoke used Unsloth, eval used vanilla transformers), (b) the trained model itself has this failure mode and the smoke just didn't surface it. Worth a side-by-side: load the same row in Unsloth-vs-HF and compare outputs.

### 5. fact_lookup misses on multi-turn-ish prompts (62.5 %)

```
fl-101 "When do I see Dr. Chen next?"  → pred=[]   (model emitted NL only, no call)
fl-108 "What meds at 9pm?"             → pred=[]   (same)
```

No tool call at all. Likely the assistant turn was cut off mid-`<think>` block by the 256-token generation cap, and parse_function_calls saw zero `<start_function_call>` markers. **Action item**: bump `max_new_tokens` to 512, retry — cheap.

## Comparison: smoke green vs eval red

The smoke check (one row per category, 7/7 green on 2026-05-01) ran via Unsloth's `FastLanguageModel.for_inference(model)` on base + LoRA. The eval ran via vanilla `transformers.AutoModelForCausalLM` on the merged BF16 checkpoint. Three differences could cause the divergence:

1. **Inference path** — Unsloth applies extra patches in `for_inference`; HF doesn't. If those patches fix a tokenization edge case for FunctionGemma's `<escape>` tokens, Unsloth would beat HF on the same weights.
2. **Quantization vs full BF16** — smoke loaded base in 4-bit + LoRA on top; eval loaded merged BF16. Counter-intuitively the BF16 path may be slightly worse if `merge_and_unload` introduced numerical drift in some layers.
3. **Row sampling** — smoke happened to test the *first* row per category, which were the lowest-numbered (e.g. fl-101, ot-101). The eval covers fl-101..fl-108, ot-101..ot-108. Smoke confirmed the model has learned the seeds but doesn't tell us about the held-out rows; eval surfaces the generalization gap.

(3) is the structural reason. (1) and (2) may be marginal contributors.

## Recommendations (per §13 R6 escalation table)

In order of cost:

1. **Re-eval with `max_new_tokens=512`** — cheap, addresses fact_lookup empty-pred cases. Maybe lifts fl/te by ~10 pts.
2. **Loosen the metric to case-normalize string args** — turns 2 partials (tt-101, te-104) into matches. Brings two_turn to 87.5 % and tool_error_recovery to 62.5 %. Doc-only; no retraining.
3. **§13 R6(a): grow `dataset_v1` by 200–500 LLM-augmented rows targeted at refusal categories** — the cheapest behavioral fix for ot/ma (25–37 %). Distil's 5 K rows for SHELL hit 90 %+; our 27 ot + 31 ma rows are well below that floor. Authoring round target: +80 ot + +80 ma. ~1-2 paste rounds via the §9.4.3 prompt template.
4. **§13 R6(b): TensorBoard inspection** — eval_loss bottomed at epoch 2 + 0.0013 wobble at epoch 3. Look at per-token loss to see which rows dominate. Fast sanity check.
5. **§13 R6(c): expand LoRA target_modules / rank** — already at r=128, all 7 modules. No room to grow within Unsloth's recommended config.
6. **§13 R6(d): full SFT (no LoRA)** — vendor cookbook hyperparams (PDB=4, seq=512, no LoRA — fits same RTX 5080 for 270M). Reproduces the Mobile-Actions 58→85 % vendor delta. ~30-60 min training.
7. **§13 R6(e): leased A100** — last resort.

**Recommended next session**: do (1) + (2) + (3) in that order. Net effort: ~2 hr authoring + 1 hr re-train + 5 min re-eval. Expected outcome: ot/ma to ~70-80 %, two_turn/tool_error_recovery to ~85 %, overall ~70 %.

If after (3) the bar is still missed, escalate to (6) — full SFT. (4)/(5) are diagnostic/documentation work, not corrective.

## Artifacts referenced

- Adapter (epoch-2): `~/functiongemma-finetune/outputs_fg_v1/checkpoint-128`
- Merged BF16 HF dir: `~/functiongemma-finetune/merged_fg_v1/` (549 MB)
- GGUF Q8_0: `~/functiongemma-finetune/merged_fg_v1.q8_0.gguf` (279 MB)
- GGUF Q4_K_M (SL2619 target): `~/functiongemma-finetune/merged_fg_v1.q4_k_m.gguf` (242 MB)
- Verbose per-row eval script: `~/functiongemma-finetune/eval_verbose.py` (server-side, not committed)
- Eval script: `scripts/eval_functiongemma_holdout.py` (committed)
- Holdout: `data/functiongemma/eval_holdout_v1.jsonl` (committed, 56 rows)
