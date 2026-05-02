# Bench notes

Active bench records for the FunctionGemma track. Each file is a one-shot
record of a specific bench run, authored at run time and not retroactively
edited.

## Layout

```
docs/bench-notes/
├── README.md                                  # this file
└── functiongemma/
    └── 2026-05-02_quantization-sweep.md       # canonical INT4/INT8 sweep report
```

The 2026-05-02 quantization sweep selected Q4_0 as the on-board variant.
Per-variant raw data lives at `bench/functiongemma/runs/2026-05-02-quant/`
(JSONL outputs from `scripts/functiongemma/bench.py --mode remote`,
gitignored under `bench/`).

## Naming

- `YYYY-MM-DD_<run-tag>.md` — human-readable summary committed to git.
- `bench/functiongemma/runs/<dir>/<variant>.jsonl` — machine-readable
  per-prompt records (gitignored).

## Default output paths

- `scripts/functiongemma/eval/eval_holdout.py` writes its summary to
  `docs/bench-notes/functiongemma/<today>_functiongemma-eval-<seam>.md`
  (default; pass `--output` to override).
- `scripts/functiongemma/bench.py` writes per-run JSONL to
  `bench/functiongemma/runs/functiongemma_{local,remote}_<timestamp>.jsonl`
  by default; pass `--out` to override.
- `scripts/functiongemma/bench/aggregate_quant.py` consumes the per-variant
  JSONL files and emits a Markdown table — see
  `2026-05-02_quantization-sweep.md` for the canonical example.

## Archived bench records

Pre-distil FunctionGemma iteration notes (block-e supplement, F1 refusal
reweight, v2 finetune eval, eval_v3/v4 sweeps) live under
`archive/functiongemma-pre-distil/bench/`. The 43 micro-files from the
weight-sweep are consolidated into `eval-summary.md` there.

Gemma 3 270M-IT health-QA bench records (2026-04-* + t5-smoke*) are at
`archive/gemma3-270m-health-qa/bench/`.
