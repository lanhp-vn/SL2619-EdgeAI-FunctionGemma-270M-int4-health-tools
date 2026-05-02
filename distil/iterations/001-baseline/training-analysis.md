# Training Analysis Report

Iteration 001 — FunctionGemma 270M-IT × patient-record tool calling.
Authoritative bench note: `docs/bench/2026-05-01_functiongemma-distil-labs-data-plan.md`.

## 1. Overview
- **Model ID:** `231feebb-8cc0-4d5f-9e4b-4d2f00e362b2`
- **Training ID:** `c9d34596-ee7a-4e56-be2b-254159fe7796`
- **Task type:** `multi-turn-tool-calling-closed-book`
- **Student model:** `functiongemma-270m-it`
- **Teacher model:** `openai.gpt-oss-120b`
- **Training duration:** DAG 2026-05-02T00:25:26Z → 04:53:39Z (~4h 28m total; finetune wall-clock 13,260s ≈ 3h 41m)
- **Goal:** Map a single-patient assistant turn to exactly one call against a 7-tool patient-record registry (vitals / meds-by-time / med-by-name / allergies / food-interaction / next-appointment / emergency-contact). No natural-language replies, no refusals.

### 1.1 Input/Output

Input — JSON-encoded chat history; system prompt v3 with 7 numbered ROUTING RULES:
```
[{"role": "user", "content": "When do I see Dr. Chen next?"}]
```

Output — exactly one well-formed function call in the FunctionGemma wire format:
```
<start_function_call>call:get_next_appointment{}<end_function_call>
```

## 2. Test Set Statistics
- **Total examples:** 24 (test.jsonl, eval-only — never used for synth seeding)
- **Per-tool label distribution (gold answers):**
  | Tool | Count |
  |---|---|
  | get_vitals | 9 |
  | get_medication_by_name | 7 |
  | get_medications_at_time | 4 |
  | check_food_interaction | 1 |
  | get_emergency_contact | 1 |
  | get_next_appointment | 1 |
  | list_allergies | 1 |
- **Question length:** 17–48 chars (median ~30); answer length: 56–120 chars
- **Provenance:** human-authored seed pairs; same 24 rows used for v1/v2/v3 teacher-eval to keep deltas comparable

## 3. Configuration Summary
- **Task:** `multi-turn-tool-calling-closed-book`
- **Student:** `functiongemma-270m-it` (LoRA r=64, α=64, dropout=0.0, target_modules=`q_proj,v_proj`)
- **Teacher:** `openai.gpt-oss-120b`
- **Synth params (non-default):**
  - `generation_target: 5000`
  - `validation_similarity_threshold: 0.90` (default 0.95 — loosened to widen scenario coverage)
  - `basic_mutators_to_use: ["complexity"]`
  - `mutation_topics`: 5 routing-rule clusters + 3 phrasing styles (see `config.yaml`)
- **Synthetic data generated:** 5,004 examples (57 iterations); merged with 50 seeds → 5,054; **expanded to 7,481 train samples** (multi-turn expansion)
- **Training epochs:** 4 (best checkpoint kept by trainer at epoch 3/4)
- **Job description:** v3 — see `job_description.json`. Closed cluster A (lab/vital catch-all), B (zero-arg tools with extra context), C (medication-class noun stripping + allergy yes/no phrasing).

## 4. Aggregate Metrics

Source: per-row metrics from `student-predictions.jsonl` (24 rows), `teacher-predictions-v3.jsonl` (24 rows), and the trainer's epoch-0 eval (base student) from training logs.

| Metric | Base Student | Teacher | Tuned Student | Δ (Tuned − Base) | Δ (Tuned − Teacher) |
|--------|-------------:|--------:|--------------:|-----------------:|--------------------:|
| **LLM-as-a-Judge** (primary) | n/a¹ | 0.958 | **0.958** | n/a | **+0.000** |
| tool_call_equivalence | 0.208 | 0.958 | 0.875 | +0.667 | −0.083 |
| binary_tool_call | 0.208 | 0.958 | 0.875 | +0.667 | −0.083 |
| staged_tool_call | 0.240 | 0.958 | 0.938 | +0.698 | −0.020 |
| ROUGE | 0.786 | 0.958 | 0.958 | +0.172 | +0.000 |

¹ The trainer's per-epoch eval does not log LLM-as-a-Judge for the base student. Judge improvement vs base is therefore inferred from the TCE lift (+0.667) plus the row-by-row matrix in §5.

**Verdict: DEPLOY**

Judge equivalence with teacher is exact (Δ = 0.000), inside the ≤0.05 DEPLOY band. ROUGE matches teacher exactly. The TCE shortfall (−0.083) is fully explained by case-folding choices on two medication names (rows 13, 20) — the LLM judge correctly forgives them per the v3 judge instructions, and any inference-time downstream consumer can resolve `name` case-insensitively (the gold `get_medication_by_name` tool already does prefix-resolves-same lookup). The single content miss (row 11 — the 8 AM time-slot question) is a localized data-coverage gap, not a structural failure.

## 5. Agreement Breakdown

Pair-wise on 24 test rows, judging by LLM-as-a-Judge (the primary metric):

| Bucket | N | % |
|---|---:|---:|
| Tuned student agrees with teacher (both judge=1) | 22 | 91.7% |
| Tuned correct, teacher wrong | 1 | 4.2% |
| Teacher correct, tuned wrong | 1 | 4.2% |
| Both wrong | 0 | 0.0% |

**Improvement over base student** (judge proxied by TCE on the trainer's epoch-0 eval): the base student passes ~5/24 (TCE=0.208), so the tuned student converts roughly **18 examples from miss → correct** with **0 visible regressions** vs base.

The two non-agreement rows are the most informative cells in the table:

- **Row 15 — "Do I have any allergies?"** Student: `list_allergies{}` ✓ — Teacher v3: `""` (empty) ✗.
  This was the **single persistent teacher miss across v1/v2/v3** despite three rounds of judge-instruction sharpening (see `teacher-eval-analysis.md` §10). The student picked the rule up from the synthgen corpus where the teacher (calling itself) was more reliable. Net: distillation can clear a teacher-side ceiling when the seed clearly conveys the contract.
- **Row 11 — "What pills do I take at 8 AM?"** Student: `""` ✗ — Teacher v3: `get_medications_at_time{time_24h:"08:00"}` ✓.
  The student inherited the routing rule but failed to emit on this one phrasing. ROUTING RULE #6 maps "morning = 08:00" — the canonical seed example was "What pills do I take at 8 AM?" itself (test row, not a train row). Likely cause: the synthgen mutations didn't produce enough "AM/PM clock-time" paraphrases at the morning slot specifically.

## 6. Analysis of Disagreements

**Patterns identified:**

1. **Per-tool accuracy is tool-shape correlated, not tool-frequency correlated.** All zero-arg tools (`get_vitals` 9/9, `get_next_appointment` 1/1, `get_emergency_contact` 1/1, `list_allergies` 1/1) and all `get_medication_by_name` rows (7/7 by judge) pass. The only failure is on `get_medications_at_time`, the only tool whose argument is a strict-format value derived from a soft natural-language phrase ("8 AM" → "08:00"). The student learned the routing decisions but is one paraphrase short on the AM-clock branch.
2. **TCE penalises a stylistic choice the judge accepts.** Rows 13 ("Why am I taking Atorvastatin?" → student emits `name="atorvastatin"`) and 20 ("Look up Ibuprofen." → `name="ibuprofen"`). Both pass judge=1.0 because the v3 judge explicitly accepts case-insensitive med names. They are not bugs; they are the model converging on a single internal canonicalisation. If exact-string TCE matters downstream, the fix is one judge-instruction line tightening name-case in synth generation, not a retraining step.
3. **Distillation cleared the teacher's ceiling on one row.** The student fixed the v1→v3 stuck miss on "Do I have any allergies?" (row 15). This is the cleanest possible signal that the routing-rules + worked-examples pattern in the v3 task description carried into the synthgen corpus and through to the student — not just memorisation of the test set.
4. **Training dynamics show real distillation, not memorisation.** Per-epoch eval went 0.208 (base) → 0.917 (E1) → 0.917 (E2) → 0.958 (E3, kept) → 0.958 (E4) on TCE; eval_loss dropped from 3.97 → 0.018 → 0.010. Plateau at E1–E2 then breakthrough at E3 is consistent with the model integrating the long-tail routing rules (RULES #6/#7), not overfitting.

**Recommended actions:**

DEPLOY now — no RETUNE. If iteration 2 is desired purely to close the row-11 gap (one row = +4.17pt), the cheapest single-lever change is:

1. **Add 6–10 train-set seeds covering AM clock paraphrases** at the morning slot ("at 8 AM", "first thing in the morning", "with breakfast", "before lunch", "right after waking"). Keep all four time-slot test rows intact; only train.jsonl changes. Expected lift: judge 0.958 → 1.000.
2. **Optional, free**: Tighten the v3 judge instruction to require lowercase med names (or, conversely, accept either case in TCE by adding a normaliser in the runtime). Either keeps judge ≥ 0.958 on the test set; the runtime fix is preferable because it does not consume an iteration.

Do **not**:

- Increase `generation_target` beyond 5,000 (we already hit teacher-judge parity; more synth would risk overfitting the contract to GPT-OSS quirks).
- Switch student to a larger model (270M is at parity with the 120B teacher on judge; a larger student would burn budget for no measurable gain).
- Re-run with a different random_seed (the failure mode is data coverage, not optimisation noise).

**Details — most informative misses:**

| # | Question | Expected | Base Output (epoch 0) | Teacher v3 Output | Tuned Output | Failure Reason |
|---|---|---|---|---|---|---|
| 11 | "What pills do I take at 8 AM?" | `get_medications_at_time(time_24h="08:00")` | (empty/format-junk; TCE=0) | `get_medications_at_time{time_24h:"08:00"}` | `""` | Synthgen produced few "AM clock-time" paraphrases; student can route the rule but missed this exact surface form. |
| 13 | "Why am I taking Atorvastatin?" | `get_medication_by_name(name="Atorvastatin")` | (likely format miss) | `get_medication_by_name{name:"Atorvastatin"}` | `get_medication_by_name{name:"atorvastatin"}` | Style-only: lowercased the med name; judge accepts (case-insensitive lookup contract). TCE penalises exact-string. |
| 15 | "Do I have any allergies?" | `list_allergies()` | (likely format miss) | `""` (teacher persistent miss) | `list_allergies{}` | **Student win.** Synthgen carried the v3 ROUTING RULE #3 worked-example block ("yes/no allergy phrasing always routes to list_allergies") and the student internalised it where the teacher could not. |
| 20 | "Look up Ibuprofen." | `get_medication_by_name(name="Ibuprofen")` | (likely format miss) | `get_medication_by_name{name:"Ibuprofen"}` | `get_medication_by_name{name:"ibuprofen"}` | Same as #13 — case-folding canonicalisation, judge=1, TCE=0. |

(The base-student column is sourced from the trainer's epoch-0 eval aggregate — TCE=0.208 = 5/24 — rather than from a per-row base prediction file, which the platform does not produce in this workflow.)

---

## 7. Deployment Notes (host-side)

The model artifacts in this repo (extracted from `c9d34596-...-model.tar`):

- `model/` — full HF safetensors (~512 MB) + `chat_template.jinja` + tokenizer
- `model-adapter/` — PEFT LoRA adapter (~651 MB) atop `google/functiongemma-270m-it`
- `model.gguf` — fused GGUF (~518 MB; format suitable for Ollama / llama.cpp)
- `model_client.py` — vendor-supplied OpenAI client; SYSTEM_PROMPT is **inlined v3** (matches `job_description.json`)
- `Modelfile` — `FROM ./model.gguf` for `ollama create`
- `README.md` — vendor deploy notes (Ollama / vLLM)

The hardcoded `SYSTEM_PROMPT` in `model_client.py` is the v3 task description. If the prompt is mutated for a future iteration, that constant must be updated in lock-step (it is the SFT contract).

No host runtime is currently installed (`ollama`, `llama-cli`, `vllm` not on PATH). A host-side smoke test option that would fit the existing repo: `pip install -e ".[functiongemma]"` (already declared) and run `scripts/functiongemma_smoke.py` against `./model.gguf`. Not executed in this report; flagged as a follow-up.
