# FunctionGemma dataset_v1 quality audit (2026-05-01)

Source: `scripts/dataset_quality_audit.py` (MiniLM `sentence-transformers/all-MiniLM-L6-v2`, KMeans seed=3407).
Inputs: seed=50 | llm_expanded=545 | train=511 | val=28 | eval_holdout=56.

## Headline verdict

**Yes, the dataset is the bottleneck.** D3 shows the absolute argument-value space is too narrow: `get_medications_at_time.time_24h` (7 unique); `check_food_interaction.food` (4 unique). The schema-description regurgitation seen in M6 (model emitting `"24-hour clock time in HH:MM format..."` as a `time_24h` value) is the predictable failure mode of training on so few real values. D5 reveals **train/eval contamination**: max-cosine p80=0.99 and 5 of the top-5 closest pairs are byte-identical. G_EVAL on this holdout is not a generalization test -- it's measuring memorization. That the M6 model still scored 44.6% on a memorization-friendly eval implicates the recipe too. Block E (dataset expansion) is required: more unique argument values per tool, and the eval holdout must be re-stratified to remove verbatim duplicates of train-set prompts.

## D1 -- Phrasing diversity per category

| category | n_rows | n_unique | mean_cos | median_cos | p90_cos | seed_recycle_pct (>=0.85) |
|---|---|---|---|---|---|---|
| fact_absence | 31 | 31 | 0.199 | 0.180 | 0.356 | 3.2% |
| fact_lookup | 143 | 103 | 0.181 | 0.130 | 0.359 | 30.1% |
| medical_advice_refusal | 31 | 29 | 0.289 | 0.242 | 0.459 | 3.2% |
| off_topic_refusal | 27 | 27 | 0.085 | 0.067 | 0.210 | 11.1% |
| parallel_call | 101 | 99 | 0.305 | 0.266 | 0.560 | 12.9% |
| tool_error_recovery | 91 | 77 | 0.322 | 0.311 | 0.559 | 15.4% |
| two_turn | 121 | 76 | 0.200 | 0.145 | 0.397 | 28.1% |

**Flagged categories** (>70% of LLM-expanded rows have cos >= 0.85 to a hand seed): _none_

## D2 -- Tool-call distribution

### Tool counts (overall)

| tool | count |
|---|---|
| get_medication_by_name | 189 |
| get_vitals | 146 |
| get_medications_at_time | 109 |
| get_next_appointment | 73 |
| get_emergency_contact | 69 |
| list_allergies | 68 |
| check_food_interaction | 55 |

### Tool x category pivot

| category | check_food_interaction | get_emergency_contact | get_medication_by_name | get_medications_at_time | get_next_appointment | get_vitals | list_allergies | TOTAL |
|---|---|---|---|---|---|---|---|---|
| fact_absence | 0 | 4 | 0 | 0 | 3 | 24 | 0 | 31 |
| fact_lookup | 16 | 16 | 25 | 17 | 17 | 36 | 16 | 143 |
| medical_advice_refusal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| off_topic_refusal | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| parallel_call | 21 | 25 | 29 | 30 | 29 | 40 | 28 | 202 |
| tool_error_recovery | 0 | 0 | 63 | 28 | 0 | 0 | 0 | 91 |
| two_turn | 18 | 24 | 72 | 34 | 24 | 46 | 24 | 242 |

**Weak tools** (< 30 calls across the 595-row expanded set): _none_

## D3 -- Argument-value diversity (the headline schema-leak test)

| tool.arg | train_calls | train_uniq | eval_calls | eval_uniq | overlap | eval_only (gap) | eval coverage |
|---|---|---|---|---|---|---|---|
| `get_medication_by_name.name` | 185 | 11 | 13 | 8 | 6 | 2 | 75.0% |
| `get_medications_at_time.time_24h` | 109 | 7 | 8 | 4 | 4 | 0 | 100.0% |
| `check_food_interaction.food` | 53 | 4 | 4 | 1 | 1 | 0 | 100.0% |

### `get_medication_by_name.name`

- train: **185** calls / **11** unique values
- eval:  **13** calls / **8** unique values
- overlap (eval values seen in train): **6** / 8
- eval-only values (model never saw): **2**

Train values: `a`, `aspirin`, `atorvastatin`, `erythromycin`, `ibuprofen`, `lisinopril`, `metformin`, `tylenol`, `vitamin`, `vitamin d3`, `warfarin`

Eval values:  `a`, `as`, `at`, `atorvastatin`, `ibuprofen`, `lisinopril`, `metformin`, `tylenol`

**Eval-only (gap):** `as`, `at`

### `get_medications_at_time.time_24h`

- train: **109** calls / **7** unique values
- eval:  **8** calls / **4** unique values
- overlap (eval values seen in train): **4** / 4
- eval-only values (model never saw): **0**

Train values: `06:00`, `08:00`, `12:00`, `15:00`, `19:00`, `21:00`, `23:00`

Eval values:  `08:00`, `12:00`, `15:00`, `21:00`

### `check_food_interaction.food`

- train: **53** calls / **4** unique values
- eval:  **4** calls / **1** unique values
- overlap (eval values seen in train): **1** / 1
- eval-only values (model never saw): **0**

Train values: `alcohol`, `grapefruit`, `grapefruit juice`, `shellfish`

Eval values:  `grapefruit`


## D4 -- Refusal-prompt clustering

### `off_topic_refusal`  (27 rows, 5 clusters)

| cluster | size | intra_cos_mean | representative prompts |
|---|---|---|---|
| 3 | 8 | 0.235 | What's the weather today? // What's the weather like today? |
| 0 | 7 | 0.121 | Tell me a joke. // Tell me a funny joke about doctors. |
| 4 | 5 | 0.268 | Can you recommend a movie? // Can you recommend a good sci-fi movie to watch? |
| 1 | 4 | 0.363 | How do you say 'hello' in French? // Translate hello into French. |
| 2 | 3 | 0.181 | Set a timer for 10 minutes. // What's 12 times 9? |

### `medical_advice_refusal`  (31 rows, 5 clusters)

| cluster | size | intra_cos_mean | representative prompts |
|---|---|---|---|
| 2 | 9 | 0.489 | Do I really need the statin for my high cholesterol? // Will extra Vitamin D help my high cholesterol? |
| 0 | 6 | 0.554 | Can I take ibuprofen with my current medications? // Can I take ibuprofen with my current meds? |
| 3 | 6 | 0.528 | Should I try to see the cardiologist sooner than June? // Should I see a cardiologist sooner than my June appointment? |
| 4 | 6 | 0.749 | How long should I stay on Lisinopril? // How long should I stay on Lisinopril? |
| 1 | 4 | 0.777 | Is it OK to skip my evening Metformin? // Is it OK to skip my evening Metformin tonight? |


## D5 -- Train <-> eval-holdout overlap

### Distribution of max cosine (each eval row vs all train rows)

| mean | median | p20 | p80 | max |
|---|---|---|---|---|
| 0.824 | 0.844 | 0.727 | 0.994 | 1.000 |

- p80 > 0.95 -> eval is too easy; the model can memorize rather than learn.

### Top-5 closest eval<->train pairs

| cosine | eval_id | train_id | eval_prompt | train_prompt |
|---|---|---|---|---|
| 1.000 | `fl-103` | `fl-237` | What pills do I take at 8 AM? | What pills do I take at 8 AM? |
| 1.000 | `fl-107` | `fl-125` | Do I have any allergies? | Do I have any allergies? |
| 1.000 | `ot-101` | `ot-001` | Tell me a joke. | Tell me a joke. |
| 1.000 | `ot-102` | `ot-002` | What's the weather today? | What's the weather today? |
| 1.000 | `tt-103` | `fl-007` | When is my next appointment? | When is my next appointment? |

## Recommendations

- Broaden the absolute argument-value vocabulary for `get_medications_at_time.time_24h` -- currently only 7 unique training values across 109 calls. Target >= 20 unique values (real medication names from a public formulary, plausible HH:MM times across the day, real food items). The M6 schema-description regurgitation is a downstream symptom of this narrowness.
- Broaden the absolute argument-value vocabulary for `check_food_interaction.food` -- currently only 4 unique training values across 53 calls. Target >= 20 unique values (real medication names from a public formulary, plausible HH:MM times across the day, real food items). The M6 schema-description regurgitation is a downstream symptom of this narrowness.
- Re-stratify the eval holdout: 5 of the top-5 closest train<->eval pairs are byte-identical and p80 max-cosine is 0.99. Either move duplicate prompts out of train, or author novel eval prompts. Otherwise G_EVAL is measuring memorization, not generalization.

