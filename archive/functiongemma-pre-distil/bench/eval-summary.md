# Pre-distil FunctionGemma evaluation rollup (eval_v3 + eval_v4)

Consolidated overall pass rates from the 43 per-checkpoint Markdown files
that previously lived under `docs/bench/eval_v{3,4}/`. The full per-category
breakdowns are preserved in the original files (one level up:
`archive/functiongemma-pre-distil/bench/eval-v{3,4}/`) for anyone who needs
to dig in. This rollup is the at-a-glance index.

**Pass bar.** A run "passes" only if **every** prompt category clears 80%.
None of the 43 runs below cleared that bar — that result is the reason the
project switched from local Unsloth-based finetune to the Distil Labs platform.

## Test-set legend

- **clean** (n=45) — `eval_holdout_v2_clean.jsonl`, all-novel-phrasing holdout.
- **contam** (n=56) — `eval_holdout_v2_contaminated.jsonl`, 11 train-overlap items mixed in.
- **cpNNN** — checkpoint id (saved at epoch boundaries).
- **dup2 / weightN / weightN_bug** — refusal-class upweighting variants tried in eval_v4.

## eval_v3 — early checkpoints (cp111, cp222, cp333)

| run | clean pass | contam pass | all-categories ≥ 80% |
|---|---|---|---|
| cp111 | 24/45 (53.3%) | 33/56 (58.9%) | NO |
| cp222 | 28/45 (62.2%) | 36/56 (64.3%) | NO |
| cp333 | 29/45 (64.4%) | 39/56 (69.6%) | NO |

`cp333_clean_failures.md` (preserved in `eval-v3/`) captures the per-row
failure analysis that motivated the eval_v4 refusal-reweighting sweep.

## eval_v4 — refusal-reweight sweep (best of 36 runs)

The sweep multiplied refusal-class loss contribution by a constant factor
(`weight = 1.5, 2, 3`), with a `_bug` variant that exposed an off-by-one
in the loss-mask alignment. `dup2` is a parallel branch that duplicated
refusal rows in the dataset rather than reweighting in-loss.

| sweep family | best clean | best contam | bar passed |
|---|---|---|---|
| dup2 (cp136/272/408) | 31/45 (68.9%) at cp272 | 41/56 (73.2%) at cp272 | NO |
| weight=1.5 | 30/45 (66.7%) at cp222/cp333 | 39/56 (69.6%) at cp222/cp333 | NO |
| weight=1.5 + bug | 30/45 (66.7%) at cp222/cp333 | 39/56 (69.6%) at cp222/cp333 | NO |
| weight=2 | 26/45 (57.8%) at cp333 | 35/56 (62.5%) at cp333 | NO |
| weight=2 + bug | 30/45 (66.7%) at cp222/cp333 | 39/56 (69.6%) at cp222/cp333 | NO |
| weight=3 | 31/45 (68.9%) at cp222 | 42/56 (75.0%) at cp222 | NO |
| weight=3 + bug | 30/45 (66.7%) at cp222/cp333 | 39/56 (69.6%) at cp222/cp333 | NO |

The full 36-row matrix (one row per checkpoint × clean/contam combination)
is preserved in `eval-v4/` as the originals.

## Why this work was abandoned

1. **Headroom plateau.** Every variant peaked ~70% contam / ~65% clean and
   refused to push higher. Refusal classes got worse the more the loss was
   reshaped — suggesting the bottleneck was data quality, not loss weighting.
2. **One-shot refusal phrasings dominated failures.** The "off_topic_refusal"
   and "fact_absence" categories were brittle to paraphrase — the seed set
   simply didn't have enough variant phrasings.
3. **Distil Labs platform produced a cleared-bar model in one shot.**
   Iteration 001 hit 0.9583 on every metric (judge, ROUGE, TCE, binary,
   staged) — see `distil/iterations/001-baseline/training-analysis.md`.

The project switched to Distil for synthetic-data generation in early May 2026.
The pre-distil path is preserved here for reference and as a fallback;
the live local-finetune entry point is
`scripts/functiongemma/train/finetune_local.py`.
