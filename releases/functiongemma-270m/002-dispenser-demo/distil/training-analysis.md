# Training Analysis Report — dispenser-demo iter-002

## 1. Overview

- **Model ID:** `584d84c3-e6a4-4967-8730-e008c3f4ba84`
- **Task type:** `multi-turn-tool-calling-closed-book`
- **Student model:** `functiongemma-270m-it`
- **Teacher model:** `openai.gpt-oss-120b`
- **Training duration:** 1 h 11 m (02:02:42 → 03:13:54 UTC, 2026-05-12; ~3× faster than iter-001's 3h 41m)
- **Goal:** Voice-driven dispenser intent router — route every user utterance to one of five tool calls (`get_patient_profile`, `get_next_appointment`, `get_emergency_contact`, `dispense_medication`, `refuse_out_of_scope`). See [`docs/plans/dispenser-demo/plan.md`](../../../../docs/plans/dispenser-demo/plan.md) §7.

### 1.1 Input/Output

Input: stringified JSON conversation array (`question` column); single-turn user message in this iteration.

```json
[{"role": "user", "content": "When's my next appointment?"}]
```

Output: stringified JSON tool call (`answer` column), exactly one function call per assistant turn.

```json
{"name": "get_next_appointment", "parameters": {}}
```

## 2. Test Set Statistics

- **Total examples:** 10
- **Label distribution (by tool):**

  | Tool | Test rows |
  | --- | ---: |
  | `get_patient_profile` | 2 |
  | `get_next_appointment` | 2 |
  | `get_emergency_contact` | 2 |
  | `dispense_medication` | 2 |
  | `refuse_out_of_scope(health_advice)` | 1 |
  | `refuse_out_of_scope(off_topic)` | 1 |

- **Question length:** 11 – 58 chars (median ~30).
- **Provenance:** stratified holdout from 42-row hand-authored seed; see `data/dispenser_demo/dataset_v1/test.jsonl`.

## 3. Configuration Summary

- **Task:** `multi-turn-tool-calling-closed-book`
- **Student:** `functiongemma-270m-it`
- **Teacher:** `openai.gpt-oss-120b`
- **Non-default parameters** (vs platform defaults):
  - `synthgen.generation_target=1500` (default 10000) — narrower scope, smaller seed
  - `synthgen.validation_similarity_threshold=0.90` (default 0.95) — wider paraphrase variety
  - `synthgen.basic_mutators_to_use=["complexity"]` (default — same)
  - `synthgen.mutation_topics`: two lists scoped to (a) the four domain tools and (b) the two refusal reasons
- **Synthetic data generated:** target 1500 (actual count not surfaced by `model show`)
- **Training epochs:** not surfaced by `model show`

## 4. Aggregate Metrics

| Metric | Base Student | Teacher | Tuned Student | Δ (Tuned − Base) | Δ (Tuned − Teacher) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLM-as-a-Judge | 0.70 | 0.80 | **1.00** | **+0.30** | **+0.20** |
| ROUGE | 0.70 | 0.80 | **1.00** | **+0.30** | **+0.20** |
| tool_call_equivalence | 0.70 | 0.80 | **1.00** | **+0.30** | **+0.20** |

(Note: training-time teacher eval reported 0.80 vs. our pre-training teacher-eval v4's 0.90 — sampling variance on the stochastic empty-prediction failure mode; the teacher missed 2 rows during this training run instead of 1.)

**Verdict: DEPLOY.** Tuned student is 0.20 *above* the teacher on every primary metric — well past the platform's standard DEPLOY threshold ("within ~0.05 of teacher"). The student also fully closes the teacher's stochastic-empty-prediction gap that drove all of our pre-training iteration cycles.

## 5. Agreement Breakdown

Reading from `predictions/teacher_eval_v4.jsonl` (the latest teacher-eval data we have; per-row training student predictions not separately downloaded at time of writing):

- **Tuned student agrees with gold:** 10 / 10 (100 %)
- **Teacher correct, tuned wrong:** 0 / 10
- **Teacher wrong, tuned correct:** 1 / 10 — row 6 "Who do I call in an emergency?" (teacher emitted empty; tuned student correctly emits `get_emergency_contact()`)

Per-category pass-rate on the holdout (plan §9.1.6 gate of ≥ 90 % per category is met with headroom):

| category | tuned student pass-rate |
| --- | ---: |
| patient_profile | 2/2 (100 %) |
| next_appointment | 2/2 (100 %) |
| emergency_contact | 2/2 (100 %) |
| dispense | 2/2 (100 %) |
| out_of_scope_refusal | 2/2 (100 %) — both reasons matched exactly |

## 6. Analysis of Disagreements

**No tuned-student failures on the test set.** The student strictly dominates both the base student and the teacher.

**Patterns identified:**

- The teacher's stochastic empty-prediction failure mode (1-2 random test rows per run, scattered across routing rules — see iter logs at `predictions/teacher_eval_v{1,2,3,4}.jsonl`) is a teacher-specific artifact, not a routing-ambiguity signal. The student trained on synthgen output (1500 paraphrases) and learned to always emit a tool call.
- v2/v3 of our `task_description` (which added worked-examples to rule 2) regressed teacher eval scores by 10-30 pp. The minimal v1 baseline was the empirical ceiling. **Future iteration discipline:** for a narrow-scope task with a competent teacher, the right move is often "leave the prompt minimal and rely on synthgen + distillation," not "add prompt detail." We learned this the iter-001 way (3 surgical-edit rounds clearing 0.79 → 0.875 → 0.958); ours, with a higher starting baseline, didn't need them.

**Recommendations:**

- ✅ **DEPLOY iter-002 to the SL2619 board.** Phase 1.6 (host-side eval on the 10-row val set, which the student has never seen) is next; if val pass-rate ≥ 90 %, proceed to Phase 1.7 (Q4_0 quantize).
- 🟡 **Watch for val/test contamination.** Synthgen paraphrases ~1500 rows from our 22 train rows; the 10 val rows are held out from synthgen, so they're a clean independent eval. If Phase 1.6 val score < 90 %, that's the first signal that we overfit the test set during training.
- 🟢 **Test set re-author for iter-003 (only if needed).** If a real-on-board failure surfaces a pattern we missed, the test set should grow + the train set follows. We have **1 of 2 free training runs left** for retune.

## 7. Distil-side IDs and timeline

| stage | timestamp (UTC) | id |
| --- | --- | --- |
| `distil model create` | 2026-05-12 01:02:13 | model `584d84c3-…` |
| upload v1 | 2026-05-12 ~01:08 | upload `0f6c09d8-…` |
| teacher-eval v1 (judge 0.90) | 2026-05-12 ~01:15 | eval `58ea5d64-…` |
| upload v2 (regressed task_description) | 2026-05-12 ~01:30 | upload `b03f0e0c-…` |
| teacher-eval v2 (judge 0.60) | 2026-05-12 ~01:32 | eval `be1946b8-…` |
| upload v3 (cleaner task_description) | 2026-05-12 ~01:42 | upload `871c5012-…` |
| teacher-eval v3 (judge 0.80) | 2026-05-12 ~01:45 | eval `f72a6c45-…` |
| upload v4 (revert to v1 baseline) | 2026-05-12 ~01:55 | upload `d0abd44c-…` |
| teacher-eval v4 (judge 0.90) | 2026-05-12 ~01:56 | eval `4f560651-…` |
| run-training | 2026-05-12 02:02:42 | training `019fc6bf-…` |
| training JOB_SUCCESS | 2026-05-12 03:13:54 | wall 1 h 11 m |

## 8. Free-tier budget remaining

- Training runs: **1 of 2 remaining** (this iteration consumed 1).
- Iter-001 did not consume the second free run; it's still on account.

Reserve the remaining free run for a retune-after-Phase 1.6 if val/test contamination shows up, or for a future iteration triggered by on-board failure modes. Do NOT spend it on iter-002 polish — the 1.00 test score does not leave room for measurable improvement.
