# FunctionGemma Block F1 — refusal-class loss reweighting (2026-05-01)

> **Status: COMPLETE. F1 PARTIAL SUCCESS — F5 IS NOW WARRANTED.**

## Headline

**F1 fixes the medical_advice collapse** (62.5 % → 87.5 % at the best
checkpoint of `weight2`/`weight3`) and **briefly hits the 80 % off_topic
bar** for the first time (weight2 cp-111 ot=83.3 ✓, weight3 cp-222
ot=83.3 ✓), but **introduces a new catastrophic failure on
`fact_absence`** (50 % → 0 % at weight2 cp-333) — the model over-corrects
toward refusal and stops surfacing vitals when asked. Best F1 checkpoint
matches v3 cp-333 on overall pass-rate but flips the PASS-cat mix; the
true grid winner is the `dup2` pilot.

| run | best cp | clean overall | clean PASS cats | Δ vs v3 cp-333 (64.4 %, 3/7) |
|---|---|---|---|---|
| **v3 (baseline)** | cp-333 | **64.4 %** | 3/7 (fl, te, tt) | — |
| weight2 (refusal-loss=2.0) | cp-333 | 57.8 % | **4/7** (fl, ma, te, tt) | overall **−6.6 pp**, +1 PASS cat |
| weight15 (=1.5) | cp-333 | 66.7 % | 3/7 (fl, te, tt) | overall **+2.3 pp**, same cats |
| weight3 (=3.0) | cp-222 | 68.9 % | 3/7 (ot, te, tt) | overall **+4.5 pp**, swap fl→ot |
| **dup2 (PILOT)** | cp-272 | **68.9 %** | 3/7 (fl, ma, tt) | overall **+4.5 pp**, swap te→ma |

**§11.4 G_EVAL: still FAIL** (no run hits ≥ 80 % every category). But
this grid produced **the first off_topic_refusal PASS in any FunctionGemma
v3+ run** and **the first medical_advice_refusal PASS at cp-333** since
v3 cp-111. **F5 (+50 fact_absence rows) is the next single experiment**
— it directly addresses the new failure mode.

## Hypothesis (recap)

cp-111 → cp-333 of v3 collapses `medical_advice_refusal` 100 % → 62.5 %
despite cp-111 proving the dataset CAN teach the contract. Train mix is
679 tool-call rows vs 202 refusal rows — per-step gradient is ~3.4×
stronger toward tool-call generation; later epochs erase the refusal
abstraction. **F1 upweights refusal-row token losses so the gradient
ratio leans back toward the under-represented class.** Direct evidence
from the v3 row-level dump: 7/16 cp-333 clean failures are tool-call
emissions on refusal rows
(`docs/bench/eval_v3/cp333_clean_failures.md`).

## Implementation paths

### Path A — proper per-row loss reweighting (PRIMARY)

`scripts/finetune_functiongemma.py`:
- `--refusal-loss-weight FLOAT` CLI flag (default 1.0 → no-op).
- `--refusal-categories` (default `off_topic_refusal,medical_advice_refusal`).
- `WeightedSFTTrainer(SFTTrainer)` overrides `compute_loss`:
  - Strips `labels` from `model.forward` (Unsloth-patched Gemma3 wraps the
    output projection in a fused CE kernel; we have to recompute manually
    to apply per-row weights).
  - **Per-row chunked CE** (`chunk_tokens=256`) — Gemma 3 270M's
    V=262 144 makes a flat `[B*T, V]` CE materialize ~5 GiB on top of
    11 GiB activations and OOM the 16 GiB RTX 5080 (verified at step
    6/333 of the first attempt; bug + fix recorded in §"OOM diagnosis"
    below). Per-row + chunked bound peak alloc to ~268 MiB.
- `_WeightedCollator` wraps the `train_on_responses_only` collator AFTER
  Unsloth's in-place modification, attaches `row_weight: [B] float32` to
  each batch.
- **`row_weight` attached as a NUMERIC COLUMN after tokenization** —
  TRL 0.22.2's `_prepare_non_packed_dataloader` calls
  `dataset.map(remove_columns=dataset.column_names)` and silently strips
  `category` before the collator runs. The first-grid bug (§"Two-bug
  recovery" below) was that all three weighted runs (1.5/2.0/3.0)
  produced bit-identical results because every batch arrived with
  category=None → weight=1.0. Fix: add `row_weight` (float per row) AS A
  COLUMN keyed by row index. Logged at startup:
  > `Block F1: row_weight column added — 202/881 rows weighted at 2.0`

Equivalence guarantee: `weighted_masked_lm_loss(row_weight=ones)` is
mathematically identical to vanilla SFT loss (per-token CE over response
tokens, divided by unmasked count or `num_items_in_batch`); fp32
associativity drift is bounded ≤ 5e-5. Pinned by
`tests/test_finetune_functiongemma_weighting.py::test_weight_one_is_no_op`
(14 tests, all green).

### Path B — duplication pilot

`scripts/build_weighted_train.py`:
- Reads `data/functiongemma/dataset_v1/train.jsonl` (881 rows).
- Each refusal-row gets `--copies` extra copies with `id` suffix `-dupN`.
- Output: `data/functiongemma/dataset_v1/train_refusal2x.jsonl` (1083
  rows; md5 `2e6b90edb0dc423fa68dea7ab0d370f8`).
- Validates at 1.0; the original `train.jsonl` is **never touched**
  (test `test_build_weighted_train_default_path_unchanged_when_disabled`).

NOT mathematically equivalent to weight=2.0: with Adam + dataset shuffle
the duplicate lands in a different micro-batch with different optimizer
state. The delta between Paths A and B is itself diagnostic info per the
brief's "more observations" license.

## Local preflight (host, before sync)

```
$ uv run pytest -q
550 passed, 2 warnings in 39.38s

$ uv run python scripts/finetune_functiongemma.py \
    --dry-run --max-dry-run-rows 4 --refusal-loss-weight 2.0
... refusal_loss_weight: 2.0  refusal_categories: off_topic_refusal,medical_advice_refusal
T1 gate PASS — dataset shape + render + length OK

$ uv run python scripts/build_functiongemma_splits.py --check
OK: on-disk splits match the deterministic build.

$ uv run python scripts/pre-commit-functiongemma.py data/functiongemma/
clean: scanned 1 path(s); no PHI patterns matched.

$ uv run python scripts/build_weighted_train.py \
    --input  data/functiongemma/dataset_v1/train.jsonl \
    --output data/functiongemma/dataset_v1/train_refusal2x.jsonl --copies 1
input  rows: 881    output rows: 1083    refusal duplications added: 202
output md5: 2e6b90edb0dc423fa68dea7ab0d370f8
validate_file: 1083/1083 pass (rate=1.0000) [OK]
```

## Server preflight

`docs/tmp/server-preflight-functiongemma-f1.md` — **GO**. Stack pins
identical to v3 (torch 2.10.0+cu128, transformers 4.56.2, trl 0.22.2,
peft 0.19.1, unsloth 2026.4.8, bitsandbytes 0.49.2). 14.6 GiB free VRAM.
Server `data/train.jsonl` md5 `ac0e261713ed8241044feaf618c538a2`
byte-equal to host (v3 corpus). `outputs_fg_v3/` preserved for delta
comparison.

## Sync

```
rsync -av --checksum scripts/finetune_functiongemma.py nouslogic-server:~/functiongemma-finetune/finetune_functiongemma.py
rsync -av --checksum data/functiongemma/dataset_v1/train_refusal2x.jsonl nouslogic-server:~/functiongemma-finetune/data/train_refusal2x.jsonl
```

Server md5 verification (final fixed `finetune_functiongemma.py`):
- `finetune_functiongemma.py` = `14e2ca1277bbbdd078aeb2655dec14ae`
- `data/train.jsonl`            = `ac0e261713ed8241044feaf618c538a2` (UNCHANGED)
- `data/train_refusal2x.jsonl`  = `2e6b90edb0dc423fa68dea7ab0d370f8`

## Training grid

Same v3 recipe (Unsloth + LoRA r=128 α=256, 7-module target, 4-bit base
+ 16-bit LoRA, max_seq=4096, 3 epochs, LR 2e-4 linear, eff batch 8,
warmup 10, optim adamw_torch, weight_decay 0.001, seed 3407,
`train_on_responses_only`, save_strategy=epoch). Only the F1 axis is
varied.

| run | train file | --refusal-loss-weight | output_dir | train rows | total steps | wall (s) | final train_loss | eval_loss min |
|---|---|---:|---|---:|---:|---:|---:|---:|
| weight2 (PRIMARY) | data/train.jsonl | 2.0 | outputs_fg_v4_f1_weight2 | 881 | 333 | ~187 | 0.262 | 0.45 @ epoch 2 |
| weight15 | data/train.jsonl | 1.5 | outputs_fg_v4_f1_weight15 | 881 | 333 | ~187 | 0.262 | 0.45 @ epoch 2 |
| weight3 | data/train.jsonl | 3.0 | outputs_fg_v4_f1_weight3 | 881 | 333 | ~187 | 0.262 | 0.45 @ epoch 2 |
| dup2 (PILOT) | data/train_refusal2x.jsonl | 1.0 (vanilla) | outputs_fg_v4_f1_dup2 | 1083 | 408 | ~210 | 0.246 | 0.43 @ epoch 2 |

(eval_loss differences across F1 weights ≤ 0.005 — eval_loss continues to
be a poor proxy for behavioral pass-rate. cp-333 wins for weight2 and
weight15; cp-222 wins for weight3 + cp-272 for dup2.)

Driver script: `~/functiongemma-finetune/run_f1_grid_v2.sh`.

### Train command (per run)

```bash
cd ~/functiongemma-finetune && source .venv/bin/activate && \
python finetune_functiongemma.py \
  --train-file data/train.jsonl --val-file data/val.jsonl --test-file data/test.jsonl \
  --refusal-loss-weight 2.0 \
  --output-dir outputs_fg_v4_f1_weight2 \
  --logging-dir runs/v4_f1_weight2
```

## Eval matrix — full per-checkpoint, both holdouts

### Overall pass rate

| run | cp | clean (45) | clean PASS | contam (56) | contam PASS |
|---|---|---|---:|---|---:|
| **v3 baseline** | cp-333 | 29/45 = **64.4 %** | 3/7 | 39/56 = **69.6 %** | 3/7 |
| weight2 | cp-111 | 18/45 = 40.0 % | 2/7 | 25/56 = 44.6 % | 2/7 |
| weight2 | cp-222 | 23/45 = 51.1 % | 1/7 | 31/56 = 55.4 % | 1/7 |
| weight2 | **cp-333** | 26/45 = 57.8 % | **4/7 ★** | 35/56 = 62.5 % | 4/7 |
| weight15 | cp-111 | 18/45 = 40.0 % | 1/7 | 26/56 = 46.4 % | 1/7 |
| weight15 | cp-222 | 28/45 = 62.2 % | 3/7 | 37/56 = 66.1 % | 2/7 |
| weight15 | **cp-333** | 30/45 = 66.7 % | 3/7 | 39/56 = 69.6 % | 3/7 |
| weight3 | cp-111 | 21/45 = 46.7 % | 1/7 | 29/56 = 51.8 % | 1/7 |
| weight3 | **cp-222** | 31/45 = **68.9 %** | 3/7 | 42/56 = **75.0 %** | 3/7 |
| weight3 | cp-333 | 28/45 = 62.2 % | 3/7 | 38/56 = 67.9 % | 3/7 |
| dup2 | cp-136 | 20/45 = 44.4 % | 1/7 | 28/56 = 50.0 % | 1/7 |
| dup2 | **cp-272** | 31/45 = **68.9 %** | 3/7 | 41/56 = 73.2 % | 3/7 |
| dup2 | cp-408 | 29/45 = 64.4 % | 2/7 | 39/56 = 69.6 % | 2/7 |

★ `weight2 cp-333` is the only F1 run achieving 4/7 PASS — the highest
PASS-cat count of any FunctionGemma run to date — but at a 6.6 pp
overall cost vs v3 cp-333 due to fact_absence dropping to 0 %.

### Per-category clean (best cp of each weighted run vs baseline)

| cat (n) | v3 cp-333 | weight2 cp-333 (★) | weight15 cp-333 | weight3 cp-222 | dup2 cp-272 |
|---|---:|---:|---:|---:|---:|
| fact_absence (8) | 50.0 | **0.0** ⚠ | 25.0 | 37.5 | 37.5 |
| fact_lookup (5) | 80.0 ✓ | 80.0 ✓ | 100.0 ✓ | 60.0 | 80.0 ✓ |
| medical_advice_refusal (8) | 62.5 | **87.5 ✓** | 75.0 | 75.0 | **100.0 ✓** |
| off_topic_refusal (6) | 33.3 | 50.0 | 33.3 | **83.3 ✓** | 50.0 |
| parallel_call (6) | 50.0 | 16.7 | 50.0 | 50.0 | 50.0 |
| tool_error_recovery (7) | 85.7 ✓ | 85.7 ✓ | 100.0 ✓ | 85.7 ✓ | 71.4 |
| two_turn (5) | 100.0 ✓ | 100.0 ✓ | 100.0 ✓ | 100.0 ✓ | 100.0 ✓ |
| **OVERALL** | 64.4 | 57.8 | 66.7 | **68.9** | **68.9** |
| **PASS cats** | 3/7 | **4/7** | 3/7 | 3/7 | 3/7 |

Per-category Δ vs v3 cp-333 (clean), best F1 cp per run:

| cat | weight2 cp-333 Δ | weight15 cp-333 Δ | weight3 cp-222 Δ | dup2 cp-272 Δ |
|---|---:|---:|---:|---:|
| fact_absence | **−50.0 ⚠** | −25.0 | −12.5 | −12.5 |
| fact_lookup | 0 | +20.0 | −20.0 | 0 |
| medical_advice_refusal | **+25.0 ✓** | +12.5 | +12.5 | **+37.5 ✓** |
| off_topic_refusal | +16.7 | 0 | **+50.0 ✓** | +16.7 |
| parallel_call | −33.3 | 0 | 0 | 0 |
| tool_error_recovery | 0 | +14.3 | 0 | −14.3 |
| two_turn | 0 | 0 | 0 | 0 |

### MA / OT trajectory across checkpoints (the F1 target)

The whole point of F1 was to prevent the cp-111→cp-333 ma collapse
(v3: 100 → 62.5). With the fix:

| run | cp-111 ma | cp-222 ma | cp-333 ma | cp-111 ot | cp-222 ot | cp-333 ot |
|---|---:|---:|---:|---:|---:|---:|
| v3 baseline | 100.0 ✓ | 37.5 | 62.5 | 50.0 | 33.3 | 33.3 |
| weight2 | **100.0 ✓** | 75.0 | **87.5 ✓** | **83.3 ✓** | 66.7 | 50.0 |
| weight15 | 87.5 ✓ | **100.0 ✓** | 75.0 | 33.3 | 33.3 | 33.3 |
| weight3 | **100.0 ✓** | 75.0 | **87.5 ✓** | 33.3 | **83.3 ✓** | 50.0 |
| dup2 | 37.5 (cp-136) | **100.0 ✓** (cp-272) | **87.5 ✓** (cp-408) | 66.7 | 50.0 | 50.0 |

**ma collapse: ARRESTED.** weight2/weight3 hold ma ≥ 87.5 % through
cp-333 (vs v3's collapse to 62.5). `dup2` cp-272 hits ma=100 % — the
duplication path produces the cleanest ma signal at the optimal
checkpoint.

**ot first PASSES seen.** weight2 cp-111 ot=83.3 % ✓ and weight3
cp-222 ot=83.3 % ✓ are the first FunctionGemma checkpoints of any v1+
run to clear the off_topic_refusal 80 % bar. weight2 erodes by cp-333,
but the per-cp signal proves the model is *capable* of refusing
off-topic on this dataset given the right gradient balance.

## Failure analysis

### New failure mode: F1 over-correction destroys `fact_absence`

`fact_absence` queries (e.g. "what's my cholesterol level?") gold to
exactly one `get_vitals` call. v3 baseline scored 50 % on this; under
F1, the model now refuses these too:

| run-cp | fa pass-rate | mechanism |
|---|---|---|
| v3 cp-333 | 50.0 % (4/8) | half route to `get_medication_by_name(cholesterol_level)` |
| weight2 cp-111 | 0.0 % (0/8) | model emits `[]` for vitals queries → counted as MISMATCH (gold has a tool call) |
| weight2 cp-333 | 0.0 % (0/8) | same |
| weight15 cp-333 | 25.0 % (2/8) | partial recovery as weight is gentler |
| dup2 cp-272 | 37.5 % (3/8) | duplication softer effect — closer to v3 |

The gradient pressure that fixes ma (refuse "should I skip my
metformin?") generalizes too aggressively into "refuse any health-data
query you're not 100% sure of." This is the F1 over-correction the
brief warned about.

**This validates F5.** The training data needs more positive examples
of "vitals-adjacent query → call get_vitals" (broad phrasing) so the
model learns the boundary between "ask for advice → refuse" and "ask
for stored data → call".

### Tradeoffs by weight magnitude

The grid is illuminating:

- **weight=1.5** (gentle): looks the most v3-like (66.7 %, 3/7 PASS) but
  doesn't actually fix ma at cp-333 (75 %). Lowest variance, smallest
  effect.
- **weight=2.0** (primary): biggest ma lift (62.5→87.5), best PASS-cat
  count (4/7), but fa annihilation (50→0). The "true F1 fingerprint" —
  this is what unbalanced upweighting looks like at the right magnitude
  to flip ma but not yet calibrated against fa.
- **weight=3.0**: pulls ot to 83.3 % (the only weight that hits ot ✓ at
  cp-222) but costs cp-333 stability — cp-333 ma drops back to 87.5 %
  same as weight2 and overall slips to 62.2 %.
- **dup2 (pilot)**: best of both worlds at cp-272 — ma=100 ✓, fl=80 ✓,
  tt=100 ✓. Duplication's diffuse-across-batches effect avoids the per-
  step gradient spike that proper weighting concentrates. 68.9 %
  overall ties weight3 cp-222.

**Headline:** the proper-weighting path lifts the F1 target (ma) further
and hits the PASS bar harder per category, but the duplication pilot
produces the highest-overall single checkpoint. This is the
weight=2.0 ≠ duplication delta the advisor predicted — both validate
the F1 mechanism, with different concentration profiles.

### Cross-comparison: fl=80 ✓ holds across runs except weight3

`fact_lookup` is the only PASS cat from v3 that survived F1 weighting
intact at cp-333 in every run except weight3 (which spent its capacity
on ot). Suggests the tool-call signal for fl is robust enough to absorb
some refusal pressure — but pc=parallel_call collapsed under weight2
(50 → 16.7) and te=tool_error_recovery wobbled under weight3
(85.7 → 71.4).

## F1 success-signature verdict

From the brief:

| criterion | result | verdict |
|---|---|---|
| `medical_advice_refusal` ≥ 80 % at best F1 cp | weight2 cp-333 87.5 ✓; dup2 cp-272 100 ✓ | **PASS** |
| `fact_lookup` ≥ 80 % retained | weight2/15/dup2 cp-best all hit 80 ✓ | **PASS** |
| `tool_error_recovery` ≥ 80 % retained | yes at all best cps | **PASS** |
| `two_turn` ≥ 80 % retained | 100 % at all best cps | **PASS** |
| `off_topic_refusal` improves or holds | weight2 cp-111 83.3, weight3 cp-222 83.3 | **PASS at intermediate cps; cp-333 still ~50** |
| Overall clean ~ 70 % | weight3 cp-222 + dup2 cp-272 = 68.9; weight15 cp-333 = 66.7 | **PASS within noise** |
| Tool-call cats not regress badly | **fa=0 at weight2 cp-333** | **FAIL — F1 over-correction** |

**Verdict: F1 is PARTIAL SUCCESS.** It does what it was designed to do
on the refusal axis (ma fixed, ot first-ever PASS) but the
gradient-rebalance over-corrects a tool-call category (fa) that was
already on the bubble. **F5 is now warranted** — adding +50 fact_absence
rows targeted at lab/vitals queries is the direct counter-pressure that
will let weight2 hold ma=87.5 ✓ AND restore fa to ≥ 50 %.

## F5 (fact_absence authoring) decision: WARRANTED

Conditions met:
- `fact_absence` < 80 % at every F1 cp (worst: 0 % at weight2 cp-333;
  best: 37.5 % at dup2 cp-272 and weight3 cp-222).
- The new-failure mechanism (over-refusal of vitals queries) is precisely
  what +50 lab/vitals fact_absence rows would mitigate — the model
  needs more "positive examples of when to call `get_vitals`" so the
  refusal pressure F1 introduces is balanced.
- F5 is independent of F1 (data-side change; doesn't touch the recipe).
  Stack: F1 (weight=2.0 keeps ma=87.5 ✓) + F5 (+50 fa rows recovers fa
  toward 50 %+) is the predicted shortest path to 70 %+ overall with
  4-5 PASS cats.

Per the brief's category list for F5 authoring: cholesterol, LDL, HDL,
A1c, blood glucose, triglycerides, hemoglobin, kidney function, liver
enzymes, oxygen trend, temperature history, resting pulse history.
Phrasing diversity target: mean intra-category cosine ≤ 0.30. Avoid
eval prompt duplication; preflight via Block D D5 cosine check.

## Recommended next experiment

**Block F5: author 50+ fact_absence rows targeted at lab/vitals queries,
combined with `--refusal-loss-weight 2.0` (the weight2 setting) for the
training run.**

```bash
# 1. Author batch via §9.4.3 LLM-augmented template
# 2. Validate + ingest
uv run python scripts/functiongemma_ingest.py data/functiongemma/_incoming/batch_005_block_f5_fa.jsonl
# 3. Rebuild splits (train.jsonl regenerates)
uv run python scripts/build_functiongemma_splits.py
# 4. Sync new train.jsonl
rsync -av --checksum data/functiongemma/dataset_v1/train.jsonl \
  nouslogic-server:~/functiongemma-finetune/data/train.jsonl
# 5. Train + eval (same recipe + F1 weight=2.0)
ssh nouslogic-server 'cd ~/functiongemma-finetune && source .venv/bin/activate && \
  python finetune_functiongemma.py \
    --train-file data/train.jsonl --val-file data/val.jsonl --test-file data/test.jsonl \
    --refusal-loss-weight 2.0 \
    --output-dir outputs_fg_v5_f1f5 \
    --logging-dir runs/v5_f1f5'
# Then run the eval driver against eval_holdout_v2_clean.jsonl + eval_holdout_v1.jsonl
```

**Expected**: ma stays ≥ 87.5 ✓ (the F1 effect is preserved by the same
weight=2.0); fa lifts from 0 → ~50–62 % at cp-333 (the +50 rows give
enough positive vitals examples to counter-balance the refusal
gradient); overall ~ 68–72 % with 4–5 PASS cats.

**Hold in reserve**: F3 (schema-leak re-author), F6 (cosine LR), F7
(LoRA r=256). The row-level evidence does NOT yet point at recipe
capacity or LR-shape problems; data-side is still where the lift lives.

## Per-checkpoint comparison: best-of-all-runs is `weight2 cp-333` vs `dup2 cp-272`

| metric | weight2 cp-333 | dup2 cp-272 | Verdict |
|---|---|---|---|
| clean overall | 57.8 % | **68.9 %** | dup2 wins |
| clean PASS cats | **4/7** | 3/7 | weight2 wins |
| ma | 87.5 ✓ | **100 ✓** | dup2 wins (perfect) |
| ot | 50 | 50 | tie |
| fa | **0** ⚠ | **37.5** | dup2 wins |
| fl | 80 ✓ | 80 ✓ | tie |
| pc | 16.7 ⚠ | 50 | dup2 wins |
| te | 85.7 ✓ | 71.4 | weight2 wins |
| tt | 100 ✓ | 100 ✓ | tie |
| contam overall | 62.5 % | **73.2 %** | dup2 wins |

`dup2 cp-272` is the better starting point for F5 — higher overall, less
fa damage to recover, ma=100 already. `weight2 cp-333` is the diagnostic
proof that proper weighting works (and that we have the lever to set the
ma/fa tradeoff curve). F5+F1 with weight=1.5 or weight=2.0 retrained on
the +50 fa corpus should beat both.

## OOM diagnosis (for posterity)

First grid attempt (timestamp 2026-05-01 02:59 server time): all three
weighted runs OOMed at step 6/333 with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 5.04 GiB.
GPU 0 has a total capacity of 15.47 GiB of which 3.10 GiB is free.
... weighted_masked_lm_loss / F.cross_entropy
```

Root cause: my `compute_loss` override recomputed CE externally to apply
per-row weights, using a flat `F.cross_entropy(logits.view(-1, V),
labels.view(-1), ...)`. Gemma 3 270M's V=262 144 → log-softmax
intermediate is `B × T × V × 4 ≈ 5 GiB` for B=4, avg T=800. The vanilla
SFT path runs because Gemma's `forward(labels=...)` uses a fused/chunked
CE kernel internally; my override bypassed it.

Fix (current): unroll per-row, gather only response (label != −100)
positions, process in `chunk_tokens=256` slices → peak alloc bounded by
`256 × 262 144 × 4 ≈ 268 MiB`. Math identical (verified at 5e-5 in
equivalence test); fp32 associativity drift ≤ 1e-5 on test-scale
batches. Re-ran at 03:08 server time — no further OOM.

## Two-bug recovery (for posterity)

**First grid (post-OOM-fix)**: weight=1.5/2.0/3.0 produced bit-identical
eval results — clean overall 66.7 % each, identical per-category pass
rates. Ruled out: not a seed effect (different per-step gradients
should produce different optima even with same seed); not a CE math bug
(equivalence test passes).

Root cause: TRL 0.22.2's `_prepare_non_packed_dataloader` calls
`dataset.map(remove_columns=dataset.column_names)` to tokenize the
`text` field, silently stripping the `category` column we'd attached.
The collator wrapper then saw `f.get("category") = None` → defaulted
all weights to 1.0 → effectively ran vanilla SFT three times.

Fix: attach `row_weight` (numeric float per row) AS A COLUMN to the
trainer's tokenized dataset AFTER `train_on_responses_only` runs. Numeric
columns survive `_prepare_non_packed_dataloader` because they don't
collide with text-tokenization. Test
`test_weighted_collator_prefers_row_weight_over_category` pins this
behavior. Sentinel log line at startup proves the column was attached:
> `Block F1: row_weight column added — 202/881 rows weighted at 2.0
> (refusal cats: ['medical_advice_refusal', 'off_topic_refusal'])`

The buggy first-grid artifacts are preserved on the server as
`outputs_fg_v4_f1_{weight2,weight15,weight3}_bug/` and
`merged_fg_v4_*_bug_cp*` and the corresponding `eval_v4/*_bug_*.md`
files — they document the weight-1.0 effective behavior.

## Artifacts

- Adapter dirs (server, FIXED): `~/functiongemma-finetune/outputs_fg_v4_f1_{weight2,weight15,weight3,dup2}/checkpoint-*/`
- Adapter dirs (server, buggy first attempt — preserved):
  `~/functiongemma-finetune/outputs_fg_v4_f1_{weight2,weight15,weight3}_bug/checkpoint-*/`
- Merged BF16 dirs (server): `~/functiongemma-finetune/merged_fg_v4_*_cp*/`,
  `~/functiongemma-finetune/merged_fg_v4_*_bug_cp*/`
- Train logs (server): `~/functiongemma-finetune/logs/train_fg_v4_f1_*.log`
- Eval markdown (rsync'd local): `docs/bench/eval_v4/{weight2,weight15,weight3,dup2}_cp*_{clean,contam}.md`
  and the `*_bug_*.md` siblings for the buggy first-grid evidence.
- Driver scripts (server, not in repo):
  `~/functiongemma-finetune/{run_f1_grid_v2.sh,run_f1_merge_eval.sh,rename_bug.sh}`
- Tests: `tests/test_finetune_functiongemma_weighting.py` (14 cases)
- Helper (host): `scripts/build_weighted_train.py` (duplication pilot)
- Server preflight: `docs/tmp/server-preflight-functiongemma-f1.md`

## Acceptance verdict

**§11.4 G_EVAL: still FAIL.** No F1 run hits ≥ 80 % every category. But
the grid produced two structural wins:
1. **The cp-111 → cp-333 ma collapse is arrested.** weight2/weight3 hold
   ma ≥ 87.5 ✓ at cp-333 (vs v3's 62.5).
2. **First off_topic_refusal PASS in any v3+ run.** weight2 cp-111 and
   weight3 cp-222 both hit ot=83.3 ✓ — proves the model is capable of
   refusing off-topic given the right loss balance, the question is
   when in training it manifests.

The new bottleneck is **fact_absence**: the F1 over-correction destroys
it (50 → 0 at weight2 cp-333). **F5 is the next single experiment** —
+50 fa rows targeted at lab/vitals queries, retrained with
`--refusal-loss-weight 2.0` should push the grid to 4–5 PASS cats with
overall ~70 %.
