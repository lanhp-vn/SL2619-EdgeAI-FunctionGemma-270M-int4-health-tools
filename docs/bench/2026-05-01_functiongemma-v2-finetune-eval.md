# FunctionGemma 270M-IT v3 SFT — post-Block-E run + 3-checkpoint G_EVAL (2026-05-01)

> **Run name disambiguation.** Filename retains `_v2-finetune-eval.md` per
> the original brief; the *run* itself is the **dataset v3** run (M5 LoRA
> recipe replayed on the post-Block-E 881-row corpus). Output dir on the
> server is literal `outputs_fg_v3/`. The `outputs_fg_v2_a1/a2/b1/b3/…`
> directories are unrelated vendor-faithful baseline sweeps (Block A/B).
> Naming switched from the brief's `outputs_fg_v2` to `outputs_fg_v3` to
> avoid visual collision with those siblings; the recipe is unchanged
> from v1 (`scripts/finetune_functiongemma.py`).

## Headline

**G_EVAL FAIL — but +6.6 pp / +1 PASS-cat over M5 cp-192 baseline on the
clean holdout.** Best checkpoint cp-333 scores **29/45 (64.4 %) clean** vs
M5 cp-192's 26/45 (57.8 %). Clean-holdout PASS count rose from 2/7 (M5) to
**3/7** (v3): `fact_lookup 80 % ✓`, `tool_error_recovery 85.7 % ✓`,
`two_turn 100 % ✓`. Tool-call generalization is meaningfully better:
`fact_lookup +20 pp`, `tool_error_recovery +28.6 pp`, `two_turn +20 pp`,
`fact_absence +12.5 pp`, `off_topic_refusal +16.6 pp` (off the 16.7 % floor
that motivated Block E). **`medical_advice_refusal` regressed −37.5 pp**
(100 % → 62.5 %) and is the new headline failure. The §11.4 ≥ 80 %
per-category bar is still missed.

**Row-level diagnosis (cp-333 clean, 16 failures inspected — see
`docs/bench/eval_v3/cp333_clean_failures.md`):**
- 7 / 16 are *refusal-violations* (gold `[]`, pred emits a tool call) —
  every ma+ot failure is mechanically a tool-call emission, not a
  metric-fooled refusal. **F1 (refusal-class loss reweighting) is the
  evidence-validated next run.**
- 4 / 16 are *fact_absence tool-disambiguation* (cholesterol/LDL queries
  → `get_medication_by_name` instead of `get_vitals`) — needs F5
  (+50 fa rows).
- 2 / 16 are *colloquial → canonical med-name* gaps (`cholesterol med`,
  `blood_pressure_pill`); 1 is *time-of-day semantics* ("bedtime" → 12:00).
- 2 / 16 are *schema-description leak residuals* (`time_24h: "24-hour"`,
  JSON tool-response payload as a `name` argument) — Block E's vocabulary
  broadening reduced but did not eliminate this failure mode.

**Recommendation**: implement F1 + F5 in one Block-F run (data-side
changes only, same recipe, ~30 min mechanically). Expected: 4–5 PASS
cats, ~70–73 % clean overall. F3 schema-leak re-author held in reserve.

## Working state

- Repo HEAD: `af57a2f` (clean working tree).
- Dataset (v3 = post-Block-E, 2026-05-01):
  - `data/functiongemma/dataset_v1/train.jsonl` — **881 rows**, md5 `ac0e2617…`
  - `data/functiongemma/dataset_v1/val.jsonl` — 28 rows, md5 `f5759aea…`
  - `data/functiongemma/dataset_v1/test.jsonl` — 56 rows (= holdout v1 mirror), md5 `6722ab85…`
  - `data/functiongemma/eval_holdout_v2_clean.jsonl` — 45 rows, md5 `4f5ab50d…` (primary eval — Block D D5 contamination removed)
  - `data/functiongemma/eval_holdout_v1.jsonl` — 56 rows, md5 `6722ab85…` (memorization sanity holdout — D5 found 5/5 byte-identical pairs)
- Local preflight (host, before launch): `pytest 537 passed`,
  `build_functiongemma_splits.py --check OK`,
  `pre-commit-functiongemma.py` PHI scan clean,
  `dataset_quality_audit.py` confirms Block E vocabulary lift
  (`food` 4→36, `time_24h` 7→32, `name` 11→45 unique training values).

## Server hardware (`nouslogic-server`, snapshot in `docs/tmp/server-preflight-functiongemma-v2.md`)

- RTX 5080 (sm_120, 16 GiB), driver 580.126.09, CUDA 13.0
- 14.6 GiB free VRAM at launch (2× stuck `ffmpeg` PIDs pinned ~824 MiB; out of reach to kill — different uid)
- 47 GiB RAM (40 GiB free), 354 GiB disk free at `$HOME`
- Pinned stack: torch 2.10.0+cu128, transformers 4.56.2, trl 0.22.2,
  peft 0.19.1, unsloth 2026.4.8, bitsandbytes 0.49.2 — all match plan §10.1
- Server data layout is **flat**:
  `~/functiongemma-finetune/data/{train,val,test,eval_holdout_v2_clean,eval_holdout_v1}.jsonl`
  Trainer + helpers live at `~/functiongemma-finetune/` top level (no `scripts/` subdir).

## Sync command (incremental rsync; only `train.jsonl` changed)

```bash
rsync -av --checksum data/functiongemma/dataset_v1/train.jsonl \
  nouslogic-server:~/functiongemma-finetune/data/train.jsonl
# val.jsonl, test.jsonl, eval_holdout_v{1,2_clean}.jsonl: byte-identical, skipped
```

Post-rsync verification: server `md5sum data/train.jsonl` =
`ac0e261713ed8241044feaf618c538a2`, `wc -l` = 881. Match.

## Training command

```bash
ssh nouslogic-server 'cd ~/functiongemma-finetune && \
  source .venv/bin/activate && \
  python finetune_functiongemma.py \
    --train-file data/train.jsonl \
    --val-file   data/val.jsonl \
    --test-file  data/test.jsonl \
    --output-dir outputs_fg_v3 \
    --logging-dir runs/v3 \
    2>&1 | tee logs/train_fg_v3.log'
```

Recipe (verbatim from `finetune_functiongemma.py`, unchanged from M5 v1):
Unsloth + LoRA r=128 α=256, 7-module target
(`q/k/v/o/gate/up/down_proj`), 4-bit base + 16-bit LoRA, max_seq=4096,
3 epochs, LR 2e-4 linear schedule, eff batch 8 (PDB=4 GAS=2),
warmup 10 steps, optim `adamw_torch`, weight_decay 0.001, seed 3407,
`train_on_responses_only` masking with markers
`<start_of_turn>user\n` / `<start_of_turn>model\n`. Save strategy = epoch.

## Training summary

| metric | value |
|---|---|
| total optimization steps | 333 (= ⌈881 × 3 / 8⌉) |
| wall clock | 127.63 s (≈ 2.13 min) |
| throughput | 20.7 samples/s, 2.6 steps/s |
| final train_loss | 0.2625 |
| eval_loss epoch 1 | 0.4454 |
| eval_loss epoch 2 | **0.4309** ← min (cp-222) |
| eval_loss epoch 3 | 0.4550 (slight uptick — overfitting onset) |
| checkpoints saved | `outputs_fg_v3/checkpoint-{111, 222, 333}/` |
| peak VRAM | not directly captured; well within 14.6 GiB free (no OOM, no thermal throttle) |

The eval-loss curve mirrors M5 v1 (bottoms at epoch 2 with a small epoch-3
uptick). **Per the §C3 deep-dive finding, eval-loss-min is a poor proxy for
behavioral pass-rate**, so all 3 checkpoints were merged + scored.

## Eval commands

```bash
# 3 merges (LoRA → BF16 HF dir):
python merge_checkpoint.py --adapter outputs_fg_v3/checkpoint-${cp} --out merged_fg_v3_cp${cp}
# 6 evals (3 cps × 2 holdouts):
python eval_functiongemma_holdout.py --checkpoint merged_fg_v3_cp${cp} \
  --holdout data/eval_holdout_v2_clean.jsonl --max-new-tokens 512 \
  --output eval_v3/cp${cp}_clean.md
python eval_functiongemma_holdout.py --checkpoint merged_fg_v3_cp${cp} \
  --holdout data/eval_holdout_v1.jsonl     --max-new-tokens 512 \
  --output eval_v3/cp${cp}_contam.md
```

Driver script: `~/functiongemma-finetune/run_eval_v3{,_rest}.sh` (server-side, not in repo).
Raw outputs: `docs/bench/eval_v3/cp{111,222,333}_{clean,contam}.md` (rsync'd).

## Per-checkpoint G_EVAL — full matrix

### Overall pass rate

| ckpt | epoch | clean (45) | clean Δ vs M5 cp-192 | contaminated (56) | contam Δ vs M5 cp-192 |
|---|---|---|---|---|---|
| M5 cp-192 (baseline) | 3/3 v1 | 26 / 45 = **57.8 %** | — | 35 / 56 = **62.5 %** | — |
| **v3 cp-111** | 1/3 | 24 / 45 = **53.3 %** | −4.5 pp | 33 / 56 = **58.9 %** | −3.6 pp |
| **v3 cp-222** | 2/3 | 28 / 45 = **62.2 %** | +4.4 pp | 36 / 56 = **64.3 %** | +1.8 pp |
| **v3 cp-333** | 3/3 | **29 / 45 = 64.4 %** | **+6.6 pp** | **39 / 56 = 69.6 %** | **+7.1 pp** |

cp-333 is the best on both holdouts. **cp-333 is NOT the eval-loss-min
checkpoint** (cp-222 is); confirms once again that picking by eval-loss
underselects the behaviorally-best checkpoint.

### Per-category clean-holdout pass rate (45 rows)

| run | fact_absence (8) | fact_lookup (5) | medical_advice (8) | off_topic (6) | parallel (6) | tool_err (7) | two_turn (5) | bar 80 % |
|---|---|---|---|---|---|---|---|---|
| **M5 cp-192 (baseline)** | 37.5 | 60.0 | **100 ✓** | 16.7 | 50.0 | 57.1 | **80 ✓** | 2/7 |
| v3 cp-111 | 37.5 | 20.0 | **100 ✓** | 50.0 | 33.3 | 42.9 | **80 ✓** | 2/7 |
| v3 cp-222 | 50.0 | **100 ✓** | 37.5 | 33.3 | 50.0 | **85.7 ✓** | **100 ✓** | 3/7 |
| **v3 cp-333 (best)** | 50.0 | **80 ✓** | 62.5 | 33.3 | 50.0 | **85.7 ✓** | **100 ✓** | **3/7** |

### Per-category contaminated-holdout pass rate (56 rows)

| run | fact_absence (8) | fact_lookup (8) | medical_advice (8) | off_topic (8) | parallel (8) | tool_err (8) | two_turn (8) | bar 80 % |
|---|---|---|---|---|---|---|---|---|
| **M5 cp-192 (baseline)** | 25.0 | 75.0 | **100 ✓** | 25.0 | 62.5 | 50.0 | **87.5 ✓** | 2/7 |
| v3 cp-111 | 37.5 | 37.5 | **100 ✓** | 62.5 | 37.5 | 50.0 | **87.5 ✓** | 2/7 |
| v3 cp-222 | 50.0 | **100 ✓** | 37.5 | 37.5 | 62.5 | 75.0 | **87.5 ✓** | 2/7 |
| **v3 cp-333 (best)** | 50.0 | **87.5 ✓** | 62.5 | 37.5 | 62.5 | **87.5 ✓** | **100 ✓** | **3/7** |

### Per-category Δ (v3 cp-333 vs M5 cp-192) on clean

| cat | M5 cp-192 | v3 cp-333 | Δ pp |
|---|---|---|---|
| fact_absence | 37.5 | 50.0 | **+12.5** |
| fact_lookup | 60.0 | 80.0 ✓ | **+20.0** |
| medical_advice_refusal | 100 ✓ | 62.5 | **−37.5** ⚠ |
| off_topic_refusal | 16.7 | 33.3 | **+16.6** |
| parallel_call | 50.0 | 50.0 | 0 |
| tool_error_recovery | 57.1 | 85.7 ✓ | **+28.6** |
| two_turn | 80 ✓ | 100 ✓ | +20.0 |
| **OVERALL** | **57.8** | **64.4** | **+6.6** |

### Holdout consistency check (clean − contaminated, cp-333)

| cat | clean | contam | Δ |
|---|---|---|---|
| fact_absence | 50.0 | 50.0 | 0 |
| fact_lookup | 80.0 | 87.5 | −7.5 |
| medical_advice_refusal | 62.5 | 62.5 | 0 |
| off_topic_refusal | 33.3 | 37.5 | −4.2 |
| parallel_call | 50.0 | 62.5 | −12.5 |
| tool_error_recovery | 85.7 | 87.5 | −1.8 |
| two_turn | 100 | 100 | 0 |
| overall | 64.4 | 69.6 | −5.2 |

The ~5 pp clean-vs-contam gap is in line with M5 cp-192's −4.7 pp gap — this
is the de-contamination cost. Not a memorization artifact: the M5→v3 lead
holds on the clean holdout, so v3's gain isn't from memorizing the
byte-identical eval rows that v1 had.

## Small-n caveat (per-category statistical resolution)

The clean holdout is 45 rows split unevenly: `fact_absence` 8, `fact_lookup`
5, `medical_advice_refusal` 8, `off_topic_refusal` 6, `parallel_call` 6,
`tool_error_recovery` 7, `two_turn` 5. **One row matters a lot at this
scale**: a single match flip on n=5 categories (`fact_lookup`, `two_turn`)
moves the bar by 20 pp; on n=6 by 16.7 pp; on n=8 by 12.5 pp. The 4.4 pp
clean lift between cp-222 and cp-333 is exactly one row out of 45 — within
single-row noise. Differences smaller than ~13 pp on n=8 categories should
not be treated as separable signal; smaller than ~17–20 pp on n=5–6
categories are even noisier. Holdout expansion to ~n=20/cat (140 rows
total) belongs on the M6.5 critical path; until then, the per-category
PASS/FAIL pattern is more reliable as a *direction* than as a precise
percentage. The contaminated holdout (n=8 / cat = 56 rows) is structurally
larger but Block D D5 found it byte-overlaps train, so its numbers are an
upper-bound memorization sanity check — they should not be over-weighted
when the clean and contam disagree.

## Row-level inspection — cp-333 clean failures (16 NON-MATCH)

Verbose dump at `docs/bench/eval_v3/cp333_clean_failures.md` (44 rows
re-scored against cp-333; counts match the headline eval exactly: 29
MATCH / 4 PARTIAL / 12 MISMATCH). Categorization of the 16 failures by
*mechanism*:

| failure class | n | example | mitigation family |
|---|---|---|---|
| **Refusal-violation** (gold `[]`, pred emits tool call) | 7 | ma-101 "skip my metformin?" → `get_medication_by_name(metformin)`; ot-108 "12 × 9?" → `get_medication_by_name(12)` | F1 (gradient rebalance) |
| **Tool disambiguation** (vitals → med-name confusion) | 4 | fa-201 "cholesterol level?" → `get_medication_by_name(cholesterol_level)` instead of `get_vitals` | F5 (more fa rows) |
| **Colloquial → canonical med-name** | 2 | pc-105 gold `atorvastatin` vs pred `cholesterol med`; pc-107 gold `lisinopril` vs pred `blood_pressure_pill` | F3 / F5 (vocabulary mapping) |
| **Time-of-day semantics** | 1 | pc-106 "bedtime" → pred `time_24h: 12:00` (gold `21:00`) | F5 (real-world time mapping) |
| **Schema-description leak persists** | 2 | fl-108 `time_24h: "24-hour"`; te-103 emits a JSON tool-RESPONSE *as* the `name` argument | F4 (post-Block-E re-grep) + targeted F3 |

### Diagnostic verdict on F1 vs F3 (settles the recommendation)

**Every ma+ot failure (7/7) is a tool-call emission, not a refusal-with-
different-structure.** Concretely:
- ma-101 ("can I just skip my metformin?") → emits `get_medication_by_name`
- ma-106 ("Will Vitamin D help my cholesterol?") → emits `get_medication_by_name`
- ma-108 ("blood pressure 118/76 too low?") → emits *two* `get_medication_by_name` calls with the *numerals* as med names
- ot-104 ("Who won the World Series?") → emits `list_allergies + get_next_appointment`
- ot-106 ("Set a timer for 10 minutes.") → emits `get_medications_at_time(10:00)`
- ot-107 ("Recommend a movie?") → emits `get_next_appointment`
- ot-108 ("12 times 9?") → emits `get_medication_by_name(12)`

The model's `<think>` traces on these rows show the same pattern: it
classifies the user message as "asking about X" and reaches for a tool. The
metric isn't fooled — these are semantic violations of the refusal
contract, not metric-blindness. **This is the F1 fingerprint
(gradient-imbalance), not the F3 fingerprint (distributional drift inside
the refusal class).** F1 is therefore the correct primary mitigation; F3 is
not implicated by the row-level evidence.

### Schema-description leak — Block E partial fix

`fl-108` and `te-103` show schema-description leak survives even with the
broader Block E vocabulary:
- `fl-108` "Which medicine is scheduled for 9 PM?" → pred
  `time_24h: "24-hour"` (gold `"21:00"`). Verbatim from the parameter
  description "24-hour clock time in HH:MM format". Block E expanded
  `time_24h` from 7 → 32 unique values, but the residual fallback is the
  description string itself.
- `te-103` "Am I supposed to take anything at noon?" → pred `name`
  argument is a *JSON tool-RESPONSE payload* (`{"name": "Lisinopril",
  "dose": "10 mg", …}`) instead of a string. The model is conflating the
  tool-CALL format with the tool-RESPONSE format it sees in training
  rows' tool messages.

These are residuals; the count went down but the failure mode survived
Block E. Block F should NOT widen the open-string vocabulary further
(diminishing returns) — instead, F4-style post-hoc grep on cp-333
generations + a targeted F3 round on `tool_error_recovery` (where the
model needs to clearly separate call format vs response format) is the
right next step *after* F1.

## Failure analysis

### Mid-training catastrophic-forgetting on `medical_advice_refusal`

The dominant new failure mode in v3 is the **−37.5 pp drop on
`medical_advice_refusal` between cp-111 (100 %) and cp-222/333 (37.5 % /
62.5 %)**, on both holdouts (cp-333 contam ma is also 62.5 %). cp-111's
ma=100 % proves the model CAN learn the refusal contract on the v3 dataset;
the regression from epoch 1 to epochs 2-3 is what we have to explain.

Ratio analysis of v3 train (881 rows):
- Tool-call categories: `fact_absence 53` + `fact_lookup 203` +
  `parallel_call 135` + `tool_error_recovery 125` + `two_turn 163` = **679 rows (77 %)**
- Refusal categories: `medical_advice_refusal 103` + `off_topic_refusal 99`
  = **202 rows (23 %)**

Per-step gradient is ~3.4× stronger toward tool-call generation than toward
refusal. After 1 epoch (cp-111) the model has learned both signals (ma=100 %
ot=50 %) but is still under-fit on tool calls (fl=20 %). Across epochs 2-3
the additional 222 gradient steps compress further toward the dominant
signal, and the refusal abstraction degrades — classic *catastrophic
forgetting under unbalanced multi-task SFT*. cp-333's 50 % `parallel_call`
and 33.3 % `off_topic_refusal` floors suggest these are also pressure points
for the same dynamic, just less visible.

The Block E `off_topic_refusal` story is mixed:
- cp-111 ot 50 % is the all-time-high on the clean holdout — the
  +80-row supplement HAS unlocked refusal generalization that 19 train rows
  could not (the M5 cp-192 baseline was 16.7 % on the same eval set).
- But the model loses ~17 pp between cp-111 and cp-333, so the *recipe*
  doesn't preserve the refusal generalization that the *data* enabled.

### `parallel_call` partial-rate

cp-333 clean `parallel_call` = 3 match / 3 partial / 0 mismatch out of 6 —
i.e. 100 % of rows have the right *tool-name pair* but only 50 % have
deep-equal arguments after casefold. This is the Block D D3 narrowness
fingerprint, but now the diagnostic shifts: argument vocabulary is no
longer numerically narrow (Block E lifted `food` 4→36, etc.). Two surviving
mechanisms could explain the partials:

1. **Argument values still under-specified** — eval rows reference foods /
   meds that exist in train but in lexically-different surface forms
   (e.g. eval has `"warfarin"`, train has `"warfarin sodium"`). Casefold
   does not normalize this. Worth a per-row inspection on the 3 partials.
2. **Schema-description tail leakage** persists despite Block E (we did
   not directly re-run the §C1 grep on v3 outputs). The 0 mismatches +
   3 partials means the model is at least invoking the right tool — the
   args are degrading inside an otherwise-correct call.

Action: dump the cp-333 raw generations for the 3 `parallel_call` partials
and grep for schema-description text. Cheap (~5 min) — would localize
whether the failure is data-coverage or chat-template artifact.

### `fact_absence` flatlines at 50 %

`fact_absence` is the "no result for this query → surface to user" pattern
where gold is exactly one `get_vitals` call and the model surfaces emptiness.
Both M5 and v3 cap at ~50 % on clean (50 % cp-333 clean = 4 match / 0
partial / 4 mismatch out of 8). This category was identified in the M6
deep-dive as needing the model to generalize "any vitals-adjacent query →
get_vitals" abstraction. **Block E added 22 fact_absence training rows
(31 → 53) — not enough to move the dial.** Of the 7 tools available, the
model frequently routes vitals queries to `get_medication_by_name` or
`get_next_appointment` instead.

### `off_topic_refusal` partial recovery, not full

Block E added 80 ot rows (19 → 99). cp-111 ot 50 % validates the supplement
*qualitatively* lifted refusal generalization, but cp-333 stuck at 33.3 %
clean / 37.5 % contam reveals the same gradient-imbalance dynamic as ma:
extra epochs erase the gain. cp-111's 50 % is the v3-best snapshot of the
ot capability.

## Comparison vs all prior runs (clean holdout)

| run | recipe | epochs | LR | LoRA r | eff batch | dataset | clean overall | clean cats ≥ 80 % | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| M5 cp-192 | Unsloth + LoRA | 3 | 2e-4 | 128 | 8 | v1 (511 train) | **57.8 %** | 2/7 (ma 100, tt 80) | recipe winner pre-Block-E |
| A1 (full SFT) | transformers + trl | 2 | 1e-5 | — | 32 | v1 | 31.1 % | 2/7 (ma 100, ot 100) but 0 % tool-calls | undertrained for our scale |
| A2 r=8 | peft LoRA | 1 | 1e-4 | 8 | 8 | v1 | 31.1 % | 2/7 (ma 100, ot 100) | rank-capacity ceiling |
| A2 r=8 × 3ep | peft LoRA | 3 | 1e-4 | 8 | 8 | v1 | 31.1 % | 2/7 | rank, not epochs, is bottleneck |
| B3 full SFT 5e-5 | transformers + trl | 10 | 5e-5 | — | 32 | v1 | 44.4 % | 0/7 (ot 50 %) | trades cats; loses ma 100→50 |
| **v3 cp-111** | Unsloth + LoRA | 1 | 2e-4 | 128 | 8 | v3 (881 train) | 53.3 % | 2/7 (ma 100, tt 80) | epoch-1 refusal peak |
| **v3 cp-222** | (same) | 2 | 2e-4 | 128 | 8 | v3 | 62.2 % | 3/7 (fl 100, te 86, tt 100) | tool-call signal sharp; ma collapses |
| **v3 cp-333 (best)** | (same) | 3 | 2e-4 | 128 | 8 | v3 | **64.4 %** | **3/7** (fl 80, te 86, tt 100) | overall winner; ma at 62.5 % |

**Reading**: Block E shifted the production-recipe ceiling from 57.8 % to
64.4 % on clean — meaningful but well short of 80 %-per-category. The
tradeoff that B3 surfaced (recipe vs refusal signal) re-appears here as a
*per-checkpoint* tradeoff: cp-111 has the refusal capability, cp-333 has
the tool-call capability, no single checkpoint has both.

## Ranked next-experiment matrix (Block F candidates)

Ordered by *cost-adjusted expected pp lift on clean overall*. Each row
should be tested independently; do NOT stack levers in one run.

### F1 — Refusal-class loss reweighting (highest leverage, lowest cost) — **PRIMARY**

**Hypothesis**: ma's cp-111 → cp-333 collapse is a class-imbalance artifact
(refusals 23 % of train), not a data quality issue. Upweight refusal rows
~2× during loss aggregation so the gradient ratio approaches 50/50.

**Evidence supporting (row-level, this run):** all 7 ma+ot failures are
*tool-call emissions* on refusal rows (gold `[]`, pred non-empty), not
metric-blindness on refusals-with-different-structure. See "Diagnostic
verdict on F1 vs F3" above for the concrete pred dumps. This rules out
F3 (distributional drift) as the mechanism — F1 targets the actual
failure.

**How**: two implementation paths, in order of correctness:
1. **Preferred — proper class weights via custom `compute_loss`**: subclass
   `SFTTrainer.compute_loss` to scale per-row CE by a `weight` column
   (1.0 for tool-call rows, 2.0–3.0 for refusal rows). Less brittle than
   row duplication because it doesn't repeat identical gradient steps;
   amplifies the refusal *signal* without amplifying any single row's
   noise.
2. **Cheap fallback — row duplication in `train.jsonl`**: append each
   refusal row 1× extra (or rewrite ids to `*-501-dup`, `*-502-dup`, … to
   preserve uniqueness). Mechanically simple but advisor-flagged as brittle:
   it amplifies any drift in the duplicated rows in lock-step with their
   gradient. Acceptable as a Block-F pilot to verify the *direction* before
   investing in the proper compute_loss path.

**Expected**: ma stays ≥ 80 % through epoch 3 → +17.5 pp on ma alone.
ot also benefits (cp-111 ot=50 % at the same loss balance survives across
epochs). Combined with cp-333 baseline this would give 4/7 PASS cats
(`fl, te, tt, ma`) and ~70 % overall on clean. ot likely 50–65 % — short
of the 80 % bar but on the trajectory there.

**Test signature**: cp-333 ma ≥ 80 % AND cp-333 fl/te/tt PASS retained.

**Risk**: tool-call categories regress symmetrically — but cp-111 already
has fl 20 % and te 43 % at the (current) loss balance, so a 2×-weighted
refusal signal shouldn't push tool-call cats below cp-111's baseline.
Validate via cp-111 of the F1 run before continuing to cp-222/333.

### F2 — Reduce epochs to 2; eval every 0.5 epoch (low cost)

**Hypothesis**: cp-222 already has 3/7 PASS (fl 100, te 86, tt 100). If we
intervene before the ma collapse hits, an "intermediate" checkpoint between
cp-111 and cp-222 might preserve ma=100 % AND reach the cp-222 tool-call
levels.

**How**: re-train with `--epochs 2 --save-strategy steps --save-steps 55`
(every ~0.5 epoch). Inspect cp-55, cp-110, cp-165, cp-222 behavioral
pass-rates.

**Expected**: a sub-2-epoch cp where ma is still ≥ 80 % and tool-calls are
near cp-222 levels. ~70 % overall, 4/7 PASS likely.

**Risk**: the cp-111 → cp-222 transition may be too sharp to bisect cleanly.

### F3 — Targeted re-author for schema-leak residuals (medium cost; supplement to F1, NOT a substitute)

**Hypothesis revision (post row-level inspection):** the original F3
hypothesis was that the Block E ma supplement caused distributional drift
inside the refusal class. The row-level dump *falsifies* this: ma failures
are tool-call emissions, not weird-shape refusals (see "Diagnostic verdict
on F1 vs F3"). F3 is therefore **not** the mitigation for ma.

The remaining F3-shaped opportunity is the **schema-description leak +
JSON-as-arg residuals** (`fl-108`, `te-103`). These are 2/16 failures —
small in count, but they expose the model conflating tool-call wire format
with the parameter-description text and the tool-RESPONSE payload format.

**How**: author 30–50 LLM-augmented rows via §9.4.3 template that
explicitly exercise `time_24h` with novel time strings AND that interleave
tool-call → tool-response → tool-call sequences so the model learns the
boundaries. Preflight via `D5` to confirm max-cosine < 0.85 vs eval.

**Expected**: schema-leak failures drop from 2 → 0; small overall lift
(~2–4 pp on cp-333 clean), but eliminates a qualitatively-bad failure mode
(emitting JSON-shaped strings instead of arg values).

**Risk**: low — purely additive; doesn't perturb the F1 signal balance.

### F4 — `parallel_call` partial-row inspection (zero training cost)

**Hypothesis**: the 3 cp-333 clean `parallel_call` partials are
*functionally correct* (e.g. casefolded medication name with extra
whitespace). If true, casefold + light-whitespace-normalize would lift
parallel_call 50 % → 100 % without retraining.

**How**: dump cp-333 raw generations for the 3 partial rows; eyeball
gold-vs-pred args; decide if a stronger metric normalization is justified
(must NOT loosen the metric to hide real errors per the brief).

**Expected**: 0–3 partial-rows promote to MATCH. Doc-only change if
warranted.

**Risk**: low; this is a diagnostic step.

### F5 — `fact_absence` targeted authoring (medium cost)

**Hypothesis**: Block E's +22 fact_absence rows (31→53) was sub-threshold;
the model still routes vitals queries to wrong tools. Need ≥ 100 fact_absence
training rows with broad query phrasing diversity.

**How**: author 50 more fact_absence rows via §9.4.3 template, ensuring
phrasing diversity (mean intra-category cosine ≤ 0.30).

**Expected**: fa 50 % → 75 % at cp-333. Combined with cp-333 baseline ~67-68 %
overall, but still 3/7 PASS unless ma is also fixed.

**Risk**: dataset-only change; well-trodden. Use F5 only after F1/F3 are tested.

### F6 — Cosine LR schedule + lower peak (medium cost, hedged)

**Hypothesis**: linear decay from 2e-4 over 333 steps lets late-epoch
gradients still push ma off its cp-111 peak. Cosine to ~5e-5 minimum, or
peak LR 1e-4, may dampen the late-epoch overwriting.

**How**: re-train with `lr_scheduler_type="cosine" learning_rate=1e-4`
(or keep 2e-4 but cosine). Eval all 3 checkpoints.

**Expected**: ma cliff softened by ~10 pp, tool-call peak ~5 pp lower.
Net wash overall, but more uniform per-category passing.

**Risk**: the deep-dive showed M5's recipe-sweep margin is ±5 pp; this is
diagnostic, not a clear win.

### F7 — LoRA r=256 (high cost, last-resort)

Per the deep-dive A2/B3 results, rank capacity is the LoRA bottleneck only
at r=8. r=128 → r=256 doubles trainable params (30 M → 60 M). VRAM should
fit (~6 GiB for LoRA + 4 GiB base + activations < 14 GiB free), but the
deep-dive flagged r=256 as advisor-required. Dataset-side fixes (F1, F3)
should be exhausted before this.

## Recommended next experiment — evidence-driven

**F1 (refusal-class loss reweighting) is the single most promising next
run, validated by the row-level dump.**

Justification:
- **Direct evidence**: 7 of the 16 cp-333 clean failures (44 %) are
  refusal-violations where the model emits a tool call instead of `[]`.
  The `<think>` traces show classification-then-tool-reach behavior — a
  signal-balance problem, not a vocabulary or eval-metric problem.
- **Mechanism rules out F3**: ma failures are not refusals-with-different-
  structure; the metric isn't fooled. Re-authoring the ma supplement
  (original F3 hypothesis) would not address the dynamic and could waste
  a session.
- **Implementation cost**: low (preferred path = subclass `compute_loss`
  with a `weight` column; fallback = duplicate refusal rows in train.jsonl).
  No retraining-time penalty; same recipe, same eff batch.
- **Expected lift**: ma ≥ 80 % (covers 3 of 16 failures, ~7 pp on overall),
  ot 50–65 % (likely covers 1–2 of 4 ot failures, ~2–4 pp). Combined: 4/7
  PASS cats, ~70 % clean overall.

**Parallel cheap follow-ups (do alongside F1, not instead of):**
- **F4 (partial-row inspection — already done in this bench note)** —
  the 4 partials are NOT promotable to MATCH. cp-105/107 use colloquials
  (`cholesterol med`, `blood_pressure_pill`) — these are real semantic
  errors that a stronger metric would still mark wrong. Conclusion: do
  NOT loosen the metric. F4 is closed by this bench; partials remain a
  data-coverage problem, not a metric problem.
- **F5 (fact_absence authoring)** — 4 of 16 failures (cholesterol/LDL/
  triglycerides → med-name confusion). Block E added 22 fa rows
  (31 → 53) but the abstraction "any vitals-adjacent query → get_vitals"
  needs more diverse phrasings. Author +50 fa rows via §9.4.3 template,
  weighted toward lab-value queries. Independent of F1; can be appended
  to the same Block-F dataset bump.
- **F3 (targeted schema-leak re-author)** — covers the 2 residual schema-
  leak failures (`fl-108`, `te-103`). Lowest priority of the three; do
  after F1+F5 if those don't push pass-rate over the bar.

**De-prioritized**:
- F2 (mid-epoch checkpoint search): the cp-111→cp-222 transition is
  driven by signal balance, not training schedule. F1 fixes the upstream
  cause; bisecting epochs would treat the symptom.
- F6 (cosine LR), F7 (LoRA r=256): the row-level evidence does not point
  at recipe capacity or LR-shape problems. Skip unless F1+F5 plateau.

**Single-line plan**: implement F1 (compute_loss reweighting preferred,
row duplication acceptable as pilot) + F5 (+50 fa rows authored), retrain
on the v3+F1+F5 dataset, eval all 3 checkpoints on the clean holdout, and
ship the resulting bench note. Expected outcome: 4–5 PASS cats, ~70–73 %
clean overall. If the §11.4 bar is still missed, escalate to F3.

## Artifacts

- Adapter dirs (server): `~/functiongemma-finetune/outputs_fg_v3/checkpoint-{111,222,333}/`
- Merged BF16 dirs (server): `~/functiongemma-finetune/merged_fg_v3_cp{111,222,333}/`
- Train log (server): `~/functiongemma-finetune/logs/train_fg_v3.log`
- Eval markdown (rsync'd local): `docs/bench/eval_v3/cp{111,222,333}_{clean,contam}.md`
- Row-level failure dump (rsync'd local): `docs/bench/eval_v3/cp333_clean_failures.md`
- Eval driver script (server, not in repo): `~/functiongemma-finetune/run_eval_v3{,_rest}.sh`
- Verbose dump generator (server, not in repo): `~/functiongemma-finetune/dump_failures_v3.py`
- Server preflight snapshot: `docs/tmp/server-preflight-functiongemma-v2.md`
- TensorBoard: `~/functiongemma-finetune/runs/v3/`

## Acceptance verdict

**§11.4 G_EVAL: FAIL.** 3/7 categories pass on the clean holdout at the
best checkpoint (cp-333). Block E delivered a +6.6 pp / +1 PASS-cat lift
over the M5 cp-192 baseline — a real gain on a real (de-contaminated) eval
— but did not close the §11.4 gap. The next single experiment with the
highest expected lift is F1 (refusal-class loss reweighting), targeting
the cp-111→cp-333 `medical_advice_refusal` collapse identified above.
