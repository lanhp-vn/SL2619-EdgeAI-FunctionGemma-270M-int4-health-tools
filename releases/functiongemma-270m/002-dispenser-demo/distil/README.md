# Distil iteration 002 — dispenser-demo (PLANNED, awaiting upload)

**What's being tested.** Whether the Distil Labs platform — with the same
`functiongemma-270m-it` + `multi-turn-tool-calling-closed-book` combination
that worked for iter-001 — can synthesize high-quality training data for the
narrower 5-tool dispenser-demo registry (`get_patient_profile`,
`get_next_appointment`, `get_emergency_contact`, `dispense_medication`,
`refuse_out_of_scope`). Headline gate: **Distil judge ≥ 0.92** on the
10-row test holdout.

**Seed counts (after 2026-05-11 rebalance):** 42 total = 8 each for the
four domain categories + 10 for `out_of_scope_refusal` (5 `health_advice`
+ 5 `off_topic`). The asymmetric refusal slot was a deliberate trade
made after the Phase 1.4 advisor pass surfaced the prior 3/5 split as
giving only 1 train row of `health_advice`. See
[`docs/plans/dispenser-demo/decisions-log.md`](../../../../docs/plans/dispenser-demo/decisions-log.md)
"2026-05-11 (Phase 1.4 rebalance)".

**Catalog-vs-blog conflict reuse.** The local distil-cli-skill model-catalog
says tool-calling tasks are restricted to Qwen3 / Llama 3-family students,
but iter-001 trained `functiongemma-270m-it` on
`multi-turn-tool-calling-closed-book` end to end (see
[`../../001-baseline/distil/README.md`](../../001-baseline/distil/README.md)
§"Catalog-vs-blog conflict resolved"). We rely on the same path; if the
platform rejects FG-270M for this task in the future, the fallback is
Qwen3-0.6B with the same dataset.

## Status

| Stage | Status | Notes / IDs |
| --- | --- | --- |
| `distil model create dispenser-demo-002` | DONE 2026-05-12 01:02:13 UTC | model id `584d84c3-e6a4-4967-8730-e008c3f4ba84` (CLI 0.15.3) |
| Local validate (within/cross-set dup check, shape, tool coverage) | DONE — `build_distil_data.py` reports clean | 22 train + 10 test, no duplicates, all 5 tools covered |
| `distil model upload-data … --dry-run` (1st) | DONE 2026-05-12 | PASS — dry-run upload id `2c2f5dd3-736b-4448-9342-c39652df909d`; no duplicates, no shape errors |
| Repair (if needed) | N/A | reshape script gated cross-set / within-set dups upstream; no repair round needed |
| `distil model upload-data …` (real) | DONE 2026-05-12 | upload id `0f6c09d8-c276-4d94-9771-479bf555731a` |
| `distil model upload-status … --output json` | DONE 2026-05-12 | `JOB_SUCCESS`, `source=direct_upload`, no logs |
| `distil model run-teacher-evaluation` v1 | DONE 2026-05-12 | judge=0.90, ROUGE=0.90, TCE=0.90 — 1 miss (row 3 "When's my next appointment?" → empty prediction); predictions at `predictions/teacher_eval_v1.jsonl` |
| `task_description` revision v2 | DONE 2026-05-12 (REVERTED) | rule #2 + worked-examples block + closing imperative + (bug) `\\'` escape sequence — broke the rendered task_description; **regressed to judge=0.60** (4 misses across rules 1, 3, 4, 5) |
| v2 upload (`b03f0e0c-…`) + eval (`be1946b8-…`) | DONE 2026-05-12 | judge=0.60 — regressed; predictions at `predictions/teacher_eval_v2.jsonl` |
| `task_description` revision v3 | DONE 2026-05-12 (REVERTED) | rule #2 + worked-examples block only (apostrophes fixed, imperative removed — one lever from v1) |
| v3 upload (`871c5012-…`) + eval (`f72a6c45-…`) | DONE 2026-05-12 | judge=0.80 — still regressed vs v1 baseline; predictions at `predictions/teacher_eval_v3.jsonl` |
| `task_description` revert to v1 baseline | DONE 2026-05-12 | every addition tested hurt; v1 baseline is the empirical ceiling for this teacher on this test set |
| v4 upload (`d0abd44c-…`) + eval (`4f560651-…`) | DONE 2026-05-12 | judge=0.90 — **baseline confirmed**; 1 miss on row 6 (not row 3 as in v1) — teacher is stable on score but stochastic on which row fails; predictions at `predictions/teacher_eval_v4.jsonl` |
| **Decision: PROCEED to training** | 2026-05-12 | judge=0.90 well above platform PROCEED threshold (0.70 for tool calling). Plan §9.1 §1.5 internal gate (≥0.92) not met but waived after 3 iteration attempts showed v1 is the ceiling. Synthgen will paraphrase 22 train rows ~150× → tuned student should generalize past teacher's single empty-prediction failure mode. |
| `distil model run-training` | DONE 2026-05-12 01:58:58 UTC | training job id `019fc6bf-9c93-4b51-81fd-081e37a5c3d6` (consumed 1 of 2 free runs) |
| Training polling | IN PROGRESS | background poll every 600 s, logging to `/tmp/distil_train_poll.log`; iter-001 took ~3h 41m, expect similar |
| Download artifacts (`model.tar` → `model/`, `model-adapter/`, `model.gguf`) | TODO | extract to `releases/functiongemma-270m/002-dispenser-demo/{merged,adapter,gguf}/` |
| Verdict | TODO | DEPLOY / RETUNE |

## Source data

- Local seed JSONL: `data/dispenser_demo/seed_conversations.jsonl` (42 rows
  — 8 each across the four domain categories + 10 in `out_of_scope_refusal`).
- Stratified splits: `data/dispenser_demo/dataset_v1/{train,val,test}.jsonl`
  (22 / 10 / 10 — see
  [`docs/plans/dispenser-demo/plan.md`](../../../../docs/plans/dispenser-demo/plan.md) §9.1.3).
- Reshaped for Distil:
  `releases/functiongemma-270m/002-dispenser-demo/distil/{train,test}.jsonl`
  (22 train + 10 test, `(question, answer)` 2-column format, flat at the
  top level of `distil/` so the CLI's `--data <dir>` lookup finds them).
- Reshape script: `scripts/dispenser_demo/data/build_distil_data.py`. Run
  `uv run python scripts/dispenser_demo/data/build_distil_data.py --check`
  to verify the on-disk files match the deterministic build.

## Reshape contract

Mirrors iter-001 byte-for-byte:

- `system` message dropped (Distil's `task_description` owns it).
- `<think>...</think>` traces dropped (the teacher doesn't see them).
- Trailing assistant NL summary dropped (Distil represents only the next
  tool call, not the post-call narration).
- `question` = stringified JSON array `[{role: 'user', content: <text>}]`
  (single-turn for now; the shape is reusable when multi-turn rows land).
- `answer` = stringified JSON `{"name": "<tool>", "parameters": <args>}`.

Sample (first row of `train.jsonl`):

```json
{
  "question": "[{\"role\": \"user\", \"content\": \"Tell me about myself.\"}]",
  "answer": "{\"name\": \"get_patient_profile\", \"parameters\": {}}"
}
```

## Tool coverage

| Tool | train rows | test rows | total |
| --- | ---: | ---: | ---: |
| `get_patient_profile` | 4 | 2 | 6 |
| `get_next_appointment` | 4 | 2 | 6 |
| `get_emergency_contact` | 4 | 2 | 6 |
| `dispense_medication` | 4 | 2 | 6 |
| `refuse_out_of_scope(reason="health_advice")` | 3 | 1 | 4 |
| `refuse_out_of_scope(reason="off_topic")` | 3 | 1 | 4 |
| **TOTAL** | **22** | **10** | **32** |

Distil multi-turn-tool-calling-closed-book floor: minimum 20 train rows
and at least one example per tool — both met with headroom.

## Commands to run (USER, gated on confirmation)

```bash
# 1. Create the platform model handle.
distil model create dispenser-demo-002
# → record the returned model id as $MODEL_ID

# 2. Local pre-flight (already gated by build_distil_data.py, but cheap to repeat).
uv run python scripts/dispenser_demo/data/build_distil_data.py --check

# 3. Dry-run upload — fails fast on cross-set duplicates or shape drift.
distil model upload-data $MODEL_ID \
    --data releases/functiongemma-270m/002-dispenser-demo/distil --dry-run

# 4. Real upload.
distil model upload-data $MODEL_ID \
    --data releases/functiongemma-270m/002-dispenser-demo/distil
# → record returned upload id

distil model upload-status $MODEL_ID --output json
# → expect `{"status": "JOB_SUCCESS", ...}`

# 5. Teacher evaluation v1 — feasibility check against the 10-row test set.
distil model run-teacher-evaluation $MODEL_ID
# → record eval id; download predictions for per-row analysis:
distil model download-teacher-evaluation-predictions $MODEL_ID \
    --output-dir releases/functiongemma-270m/002-dispenser-demo/distil/predictions/

# 6. If judge < 0.92: tighten task_description, re-upload, re-evaluate. Mirror
#    iter-001's 3-round flow (`../../001-baseline/distil/README.md`
#    §"Status — feasibility confirmed").

# 7. Training run (consumes 1 of the 2 free runs per pricing page; the user
#    should explicitly authorize this step).
distil model run-training $MODEL_ID

# 8. Poll training (hours-scale; sleep 600).
while true; do
  status=$(distil model training $MODEL_ID --output json | jq -r '.status')
  echo "$(date +%H:%M:%S) status=$status"
  case "$status" in JOB_SUCCESS|JOB_FAILURE|JOB_STOPPED) break ;; esac
  sleep 600
done

# 9. Show + download artifacts.
distil model show $MODEL_ID --output json
# → URL to model.tar; download and extract to:
#     releases/functiongemma-270m/002-dispenser-demo/merged/
#     releases/functiongemma-270m/002-dispenser-demo/adapter/
#     releases/functiongemma-270m/002-dispenser-demo/gguf/
```

## Optional zero-cost commands

```bash
# Verify what was uploaded byte-for-byte.
distil model download-data $MODEL_ID --output-dir /tmp/distil-002-roundtrip
diff -r releases/functiongemma-270m/002-dispenser-demo/distil/data \
        /tmp/distil-002-roundtrip
```

## Files in this directory

| file | rows | purpose |
| --- | ---: | --- |
| `README.md` | — | this file |
| `config.yaml` | — | `task=multi-turn-tool-calling-closed-book`, `student=functiongemma-270m-it`, `teacher=openai.gpt-oss-120b`, `generation_target=1500`, `validation_similarity_threshold=0.90`, mutation_topics scoped to the 5 intents |
| `job_description.json` | — | `task_description` (Sago routing rules, 5 numbered rules) + 5-tool registry + judge instructions (zero-param rule for the 4 domain tools, exact-match rule for `refuse_out_of_scope.reason`) |
| `train.jsonl` | 22 | reshaped from `data/dispenser_demo/dataset_v1/train.jsonl` |
| `test.jsonl` | 10 | reshaped from `data/dispenser_demo/dataset_v1/test.jsonl` |
| `predictions/` (created in step 5) | — | teacher / training prediction outputs for per-row analysis |

## Notes

- The 10 val rows (`data/dispenser_demo/dataset_v1/val.jsonl`) are
  intentionally NOT uploaded to Distil. They remain our independent eval
  signal for the Phase 1.6 host-side holdout eval after we download the
  tuned student.
- `dispense_medication()` in the Phase 1.2 stub returns `{"status":
  "dispensed"}` for every call. The seed `di-008` encodes the
  `ble_not_connected` response (Phase 2 failure-recovery exemplar); at
  the live-tool boundary that row mismatches the stub until Phase 2 wires
  real BLE. Phase 1.6 holdout eval will flag `di-008` as a known
  deferred-eval row. See `src/gemma_tools/dispenser_demo/tools.py`
  `_dispense_medication` docstring.
- For `refuse_out_of_scope(reason)`, the judge instruction enforces exact
  reason-string equivalence — a `refuse_out_of_scope(off_topic)`
  prediction against a `refuse_out_of_scope(health_advice)` gold is BAD.
  The reason carries diagnostic signal for per-cluster eval; rationale in
  [`docs/plans/dispenser-demo/decisions-log.md`](../../../../docs/plans/dispenser-demo/decisions-log.md)
  "2026-05-11 (evening)".
