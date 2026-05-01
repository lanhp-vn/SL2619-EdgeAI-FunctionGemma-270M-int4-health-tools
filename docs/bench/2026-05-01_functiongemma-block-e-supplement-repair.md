# FunctionGemma Block E — supplement repair + ingest (2026-05-01)

Local-only dataset repair. No training, no upload, no remote SSH.

## Headline

`supplement_dataset.jsonl` (740 rows, 370 unique ids each duplicated, ~420
validator failures) replaced with a regenerated 370-row Block E supplement
that validates at 1.0, scans clean for PHI, and ingested into
`llm_expanded_v1.jsonl` without disturbing any G_EVAL artifact.

## Why regen, not patch

Reading the broken file row-by-row found three classes of damage that
co-occurred on most rows:

1. **Wire-format defects** (every row): `function.arguments` was a JSON
   *string* (`"{}"`), not a `dict`; tool messages were missing the required
   `name`; assistant final answers contained literal `<answer>` text.
2. **Tool-response shape drift**: ad-hoc payloads like `{"i": false}` or
   `{"medications": []}` that don't match the registry handler return
   shapes (`_med_to_dict`, `_get_vitals`). Training on these would teach the
   model nonsense to copy at inference time.
3. **Placeholder content** (~190 / 370 unique ids): user prompts like
   `"How does topic_501 work?"`, `"Combined request 538"`, `"Visit info
   501"`, `"Vitals too? 501"`. No semantic value to salvage.

The salvage ratio was too low to justify per-row triage. Deterministic
regeneration from a single script (`scripts/build_block_e_supplement.py`)
is shorter to write *and* shorter to audit than per-row repair.

## Generator design

`scripts/build_block_e_supplement.py`:

- Tools list comes from `as_function_declarations()` — no chance of registry
  drift.
- Vocabularies hand-curated to ≥ 30 / ≥ 30 / ≥ 35 entries (foods / times /
  meds), with all 36 / 31 / 42 actually used in tool-call arguments.
- Tool-response payloads built by helpers that mirror `_med_to_dict`,
  `_get_vitals`, etc. so the JSON the model reads as context matches what
  the registry actually returns at inference time.
- Per-category generators emit row dicts; the orchestrator runs
  `validate_conversation` + a custom Block E audit on the in-memory list
  before writing. Validator failures or audit failures abort with the
  failing rows printed; the file is only written when both gates pass.

In-process Block E audit checks (12 conditions; all hard fails):

1. exactly 370 rows
2. expected per-category counts (80/80/60/30/40/40/40)
3. exact id ranges (`ot-501..ot-580` etc.)
4. no duplicate ids
5. no duplicate user prompts (any role, any category)
6. no shared first-4-word user-prompt prefix within a category
7. no forbidden tokens (`<answer>`, `<bos>`, `<start_of_turn>`,
   `<end_of_turn>`, `<start_function_call>`, `<escape>`)
8. no placeholder patterns (`topic_\d+`, `Combined request \d+`,
   `Unique prompt fallback`)
9. all `function.arguments` are dicts
10. all tool messages carry `name`
11. `check_food_interaction.food` ≥ 25 unique values
12. `get_medications_at_time.time_24h` ≥ 25 unique values, `get_medication_by_name.name` ≥ 30

## Validator + PHI scan

```
$ uv run python -c 'from pathlib import Path; from gemma_tools.functiongemma_dataset import validate_file; r = validate_file(Path("data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl"), min_pass_rate=1.0); print(r.pass_rate, r.total, len(r.failures))'
1.0 370 0

$ uv run python scripts/pre-commit-functiongemma.py data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl
clean: scanned 1 path(s); no PHI patterns matched.
```

## Argument-vocabulary counts

Compared with the pre-existing v1 corpus (Block D audit, §D3):

| arg | v1 train (pre-Block E) | Block E supplement | combined uniqueness |
|---|---|---|---|
| `check_food_interaction.food` | 4 | **36** | 38 |
| `get_medications_at_time.time_24h` | 7 | **31** | 33 |
| `get_medication_by_name.name` | 11 | **42** | 47 |

The Block E supplement alone exceeds every minimum on its own, so once
ingest merges the two, the arg-vocabulary slot for each open-string
argument has ≥ 33 unique values across the training corpus. The M6
schema-description-regurgitation failure mode (`time_24h: "24-hour clock
time in HH:MM format..."`) was diagnosed as the predictable downstream of
N=4–7 unique slot values (Block D §D3); Block E pushes the slot count
into the regime where a 270M model can plausibly abstract slot-shape from
slot-values.

## Ingest result

```
$ uv run python scripts/functiongemma_ingest.py data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl
ingest: data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl
  batch: 370/370 passed (rate 1.0000)
  -> appended to data/functiongemma/llm_expanded_v1.jsonl
  -> quarantined  0 row(s) to data/functiongemma/quarantine.jsonl
  cumulative: 915/925 (rate 0.9892; M4.5 bar ≥ 0.80 — OK)
```

`llm_expanded_v1.jsonl` grew 545 → 915 rows.

## Split rebuild

```
$ uv run python scripts/build_functiongemma_splits.py
seed rows: 50, expanded rows: 915

category                   train    val   test
----------------------------------------------
fact_absence                  53      4      8
fact_lookup                  203      4      8
medical_advice_refusal       103      4      8
off_topic_refusal             99      4      8
parallel_call                135      4      8
tool_error_recovery          125      4      8
two_turn                     163      4      8
----------------------------------------------
TOTAL                        881     28     56
```

Pre-Block-E `off_topic_refusal` was 19 train rows (sub-`_TRAIN_THIN_FLOOR`
warning). Now 99 — clears the threshold by 5×.

## Holdout / val byte-stability check

| file | pre-ingest md5 | post-rebuild md5 | changed? |
|---|---|---|---|
| `data/functiongemma/eval_holdout_v1.jsonl` | `6722ab85…` | `6722ab85…` | NO |
| `data/functiongemma/eval_holdout_v2_clean.jsonl` | `4f5ab50d…` | `4f5ab50d…` | NO (out of split-builder's path) |
| `data/functiongemma/dataset_v1/test.jsonl` | `6722ab85…` | `6722ab85…` | NO |
| `data/functiongemma/dataset_v1/val.jsonl` | `f5759aea…` | `f5759aea…` | NO |
| `data/functiongemma/dataset_v1/train.jsonl` | `9b689304…` | `ac0e2617…` | YES (511 → 881 rows) |

Why: the splitter sorts each category by id and pins positions 1..8 as
holdout, 9..12 as val. Block E ids (`*-501`+) lex-sort after the existing
1xx/2xx ids, so the new rows fall into the train remainder only. No
G_EVAL artifact was overwritten.

## Files changed

| file | change |
|---|---|
| `supplement_dataset.jsonl` | **untouched** — preserved at repo root as audit artifact |
| `scripts/build_block_e_supplement.py` | NEW — deterministic generator |
| `data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl` | NEW — 370-row repaired candidate |
| `data/functiongemma/llm_expanded_v1.jsonl` | grew 545 → 915 rows |
| `data/functiongemma/dataset_v1/train.jsonl` | regenerated (511 → 881 rows) |
| `docs/plans/FunctionGemma/README.md` | added "Block E supplement landed" subsection |
| `docs/bench/2026-05-01_functiongemma-block-e-supplement-repair.md` | this file |

## Next recommended command (server-side, manual)

The user runs this on the GPU server, after rsync of the updated
`dataset_v1/` directory:

```bash
ssh nouslogic-server 'cd ~/functiongemma-finetune && source .venv/bin/activate && \
  python scripts/finetune_functiongemma.py \
    --train data/functiongemma/dataset_v1/train.jsonl \
    --val   data/functiongemma/dataset_v1/val.jsonl \
    --output outputs_fg_v2'
```

Then re-eval the resulting checkpoint via
`scripts/eval_functiongemma_holdout.py --eval data/functiongemma/eval_holdout_v2_clean.jsonl`.

Pre-Block-E M5 cp-192 baseline on the clean holdout = 26 / 45 = 57.8 % (cf.
`2026-05-01_functiongemma-eval-deepdive.md`). Block E hypothesis: with +160
refusal rows + broader open-string argument vocabulary, `off_topic_refusal`
moves off its 16.7 % floor and the schema-description-regurgitation failure
on `parallel_call` gets resolved, lifting overall pass-rate toward the
§11.4 ≥ 80 %-per-category bar.
