# archive/functiongemma-pre-distil/

Frozen snapshot of the FunctionGemma SFT path before the switch to Distil
Labs. **Do not edit.** Read as historical record.

## What was tried

| Block | Goal | Outcome |
|---|---|---|
| Block A — host smoke | Round-trip `<start_function_call>...` on host CPU | GREEN — vendor weather example in ~5.7 s |
| Block B — tool registry | 7 read-only patient-YAML tools at 99% branch coverage | GREEN — survived the refactor; lives at `src/gemma_tools/functiongemma/tools.py` |
| Block C — hand seeds | 50 hand-authored multi-turn seed conversations | GREEN — `data/functiongemma/seed_conversations.jsonl` |
| Block C+ — LLM augmentation | Pro Perplexity / Claude / ChatGPT batches expanding to ~545 rows | GREEN — `data/functiongemma/llm_expanded_v1.jsonl` (still active) |
| Block D — Phase D Unsloth SFT (v1, with refusal weighting) | Train FG-270M with refusal-class loss reweighting | mixed — v1 produced workable LoRA but plateau'd at ~70% pass |
| Block D — Phase D Unsloth SFT (v2, cleaner) | Same recipe without refusal weighting | mixed — different failure mode, same plateau |
| Block E — supplement repair | 740-row supplement targeting refusal/parallel-call coverage gaps | GREEN as data; downstream training never cleared the bar |
| Block F1 — refusal-class loss reweight sweep | weight ∈ {1.5, 2, 3} × `_bug` variants | FAILED — none of 36 variants cleared the all-categories ≥ 80% bar |

The full pre-distil journey is in `plans/phase-d-readme-original.md`
(verbatim 2321-line plan).

## Why this work was abandoned

- **Headroom plateau.** Every variant peaked ~70% contam / ~65% clean and
  refused to push higher. Refusal classes got worse the more the loss was
  reshaped — the bottleneck was data quality, not loss weighting.
- **Distil Labs cleared the bar in one shot.** Iteration 001 hit 0.9583 on
  every metric (judge, ROUGE, TCE, binary, staged). See
  `distil/iterations/001-baseline/training-analysis.md`.

The local-finetune scripts are not removed — `finetune_functiongemma_v2.py`
became the live local fallback at
`scripts/functiongemma/train/finetune_local.py`. Only v1 (with the
refusal-weighting feature that's superseded by Distil's higher-quality
synthesis) is here.

## What's reachable from here

- `plans/phase-d-readme-original.md` — the full original 2321-line plan
- `bench/eval-summary.md` — consolidated rollup of 43 micro-files from eval_v3 + eval_v4
- `bench/2026-05-01_functiongemma-block-e-supplement-repair.md`
- `bench/2026-05-01_functiongemma-block-f1-refusal-reweight.md`
- `bench/2026-05-01_functiongemma-v2-finetune-eval.md`
- `scripts/finetune_functiongemma.py` — v1, with refusal-class loss reweighting
- `scripts/build_block_e_supplement.py` — Block E supplement generator
- `scripts/build_weighted_train.py` — refusal-weighted train-jsonl builder
- `data/supplement_dataset.jsonl` — Block E supplement output (740 rows)
- `data/_incoming/`, `data/_raw/` — raw teacher dumps (one-time use)
- `data/llm_expanded_v1.jsonl` — wait, this is **active** under
  `data/functiongemma/llm_expanded_v1.jsonl`, NOT archived. Same for the
  other live data files.
- `tests/test_finetune_functiongemma_weighting.py` — pytest for v1 weighting
  (NOT in default CI; run manually with `pytest archive/...`)
