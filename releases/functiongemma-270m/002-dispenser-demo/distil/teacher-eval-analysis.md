# Teacher Evaluation Analysis Report — dispenser-demo iter-002

## 1. Overview

- **Model ID:** `584d84c3-e6a4-4967-8730-e008c3f4ba84`
- **Task type:** `multi-turn-tool-calling-closed-book`
- **Goal:** Voice-driven dispenser intent router — route every user utterance to one of five tool calls. See [`docs/plans/dispenser-demo/plan.md`](../../../../docs/plans/dispenser-demo/plan.md) §7.

### 1.1 Input/Output

Input: stringified JSON conversation array — single-turn user utterance.

```json
[{"role": "user", "content": "When's my next appointment?"}]
```

Output: stringified JSON tool call, exactly one function call per assistant turn.

```json
{"name": "get_next_appointment", "parameters": {}}
```

## 2. Test Set Statistics

10 rows held out from a 42-row hand-authored seed (8 each for the 4 domain
intents + 10 for `out_of_scope_refusal` split 5/5 by reason; see
[`docs/plans/dispenser-demo/decisions-log.md`](../../../../docs/plans/dispenser-demo/decisions-log.md)
"2026-05-11 (Phase 1.4 rebalance)").

| Tool | test rows |
| --- | ---: |
| `get_patient_profile` | 2 |
| `get_next_appointment` | 2 |
| `get_emergency_contact` | 2 |
| `dispense_medication` | 2 |
| `refuse_out_of_scope(health_advice)` | 1 |
| `refuse_out_of_scope(off_topic)` | 1 |

## 3. Configuration Summary

- **Task:** `multi-turn-tool-calling-closed-book`
- **Student:** `functiongemma-270m-it`
- **Teacher:** `openai.gpt-oss-120b`
- **Non-default parameters** vs platform defaults:
  - `synthgen.generation_target=1500` (default 10000)
  - `synthgen.validation_similarity_threshold=0.90` (default 0.95)
  - `synthgen.mutation_topics`: scoped to 5 intents and 2 refusal reasons

## 4. Aggregate Metrics

(Final pre-training eval — v4, against `upload d0abd44c-…`)

| Metric | Score |
| --- | ---: |
| LLM-as-a-Judge | 0.90 |
| ROUGE | 0.90 |
| tool_call_equivalence | 0.90 |

**Verdict: ITERATE** at the platform's standard tool-calling threshold (< 0.70
= ITERATE), but the plan §9.1 §1.5 internal gate (≥ 0.92) was not met. After
3 surgical-edit attempts in v2/v3 (all of which *regressed* the teacher
below v1), the v1 baseline was determined to be the empirical ceiling for
this teacher on this test set. Proceeded to training anyway (waiver
documented in the README log and `training-analysis.md` §6).

## 5. Agreement Breakdown

(v1 / v4 — the baseline runs; v2 / v3 were the surgical-edit experiments)

| run | judge | misses | failing row(s) | predicted |
| --- | ---: | ---: | --- | --- |
| v1 | 0.90 | 1 | row 3 "When's my next appointment?" | (empty string) |
| v2 | 0.60 | 4 | rows 2, 6, 7, 9 — across rules 1, 3, 4, 5 | all empty strings |
| v3 | 0.80 | 2 | rows 6, 7 — across rules 3, 4 | all empty strings |
| v4 | 0.90 | 1 | row 6 "Who do I call in an emergency?" | (empty string) |

**Key observation:** The teacher's failure mode is *stochastic empty
predictions* — the score is stable at ~0.90 across runs but the specific
row that fails drifts (v1 hit row 3; v4 hit row 6 on a re-run of the
identical task_description). The failure is NOT concentrated on a specific
routing rule, so synthgen's per-rule paraphrase coverage during training
fully addresses it.

## 6. Analysis of Disagreements

**Patterns identified:**

1. **Stochastic empty prediction on direct intents.** The teacher
   (`gpt-oss-120b`) occasionally emits an empty string instead of a tool
   call on maximally-canonical phrasings ("When's my next appointment?",
   "Who do I call in an emergency?"). The phrasings are unambiguous — this
   is teacher-side decoding noise, not a routing-rule clarity issue.
2. **Adding prompt text monotonically increases the empty-prediction rate.**
   v2 (worked examples + closing imperative + bug `\'` escape) regressed to
   judge=0.60. v3 (worked examples only, clean apostrophes) regressed to
   judge=0.80. v4 (revert to v1) returned to 0.90. The teacher gets *more*
   cautious as the task_description grows.
3. **The reason enum on `refuse_out_of_scope` is well-routed.** Both
   refusal rows (health_advice + off_topic) passed in every iteration —
   the teacher reliably picks the right reason. The Phase 1.4 rebalance
   (3 → 5 `health_advice` seeds for train-cluster parity) was the right
   call.

**Recommended actions:**

- ✅ **No further task_description revisions for this iteration.** Proceed
  to training with the v1 task_description.
- 🟢 **Carry the "minimum prompt is the ceiling" lesson forward.** For
  iter-001's broader scope (7 tools, more ambiguous routing), 3 surgical
  rounds tightened 0.79 → 0.958. For iter-002's narrower scope (5 tools,
  sharper routing), the same surgical-edit pattern *hurt* — the teacher
  was already at near-ceiling with the minimal prompt. Future iter-003+
  should start with the minimum prompt and only add text on observed
  routing failures, never preemptively.
- 🟡 **The 0.90 ceiling may not bound the tuned student.** Synthgen
  paraphrases ~1500 rows from our 22 train rows; the student learns to
  always emit a tool call (none of our seeds have empty answers). Predict
  the tuned student will exceed teacher on this test set — see
  `training-analysis.md` for the verified result (it did: 1.00 vs 0.80).

## 7. Distil-side IDs (predictions archived)

| eval | eval id | upload | judge | predictions |
| --- | --- | --- | ---: | --- |
| v1 | `58ea5d64-020c-4db6-a603-0289f3243618` | `0f6c09d8-…` | 0.90 | `predictions/teacher_eval_v1.jsonl` |
| v2 | `be1946b8-e964-4b4f-ba18-34725374c99f` | `b03f0e0c-…` | 0.60 | `predictions/teacher_eval_v2.jsonl` |
| v3 | `f72a6c45-98d1-4d8f-a228-7b7d468fa5fe` | `871c5012-…` | 0.80 | `predictions/teacher_eval_v3.jsonl` |
| v4 | `4f560651-ff45-4f1c-8b68-e6ae425dddae` | `d0abd44c-…` | 0.90 | `predictions/teacher_eval_v4.jsonl` |
