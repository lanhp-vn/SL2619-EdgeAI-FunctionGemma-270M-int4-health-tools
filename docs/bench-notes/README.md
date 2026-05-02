# Bench notes

Active bench records for the FunctionGemma track. Each file is a one-shot
record of a specific bench run, authored at run time and not retroactively
edited.

## Layout

```
docs/bench-notes/
├── README.md                          # this file
└── functiongemma/                     # FunctionGemma current-state notes
    ├── 2026-05-01_functiongemma-distil-labs-data-plan.md
    ├── 2026-05-01_functiongemma-eval-deepdive.md
    ├── 2026-05-01_functiongemma-eval.md
    └── 2026-05-01_functiongemma-dataset-audit.md
```

## Naming

- `YYYY-MM-DD_<run-tag>.md` — human-readable summary.
- `YYYY-MM-DD_<run-tag>.jsonl` — machine-readable per-prompt records.

## Default output paths

- `scripts/functiongemma/eval/eval_holdout.py` writes its summary to
  `docs/bench-notes/functiongemma/<today>_functiongemma-eval.md`.
- `scripts/functiongemma/bench.py` writes per-run JSONL to
  `bench/functiongemma/runs/functiongemma_{local,remote}_<timestamp>.jsonl`
  (see top-level `bench/` tree, not here).

## Archived bench records

Pre-distil FunctionGemma iteration notes (block-e supplement, F1 refusal
reweight, v2 finetune eval, eval_v3/v4 sweeps) live under
`archive/functiongemma-pre-distil/bench/`. The 43 micro-files from the
weight-sweep are consolidated into `eval-summary.md` there.

Gemma 3 270M-IT health-QA bench records (2026-04-* + t5-smoke*) are at
`archive/gemma3-270m-health-qa/bench/`.
