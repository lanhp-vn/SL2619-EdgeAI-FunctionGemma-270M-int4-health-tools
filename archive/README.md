# archive/

Read-only snapshots of past work tracks. **Do not edit.** New work goes under
`docs/`, `src/`, `scripts/`, `tests/`, `data/`, `releases/`, `distil/`, or
`bench/` at the repo root — never inside this tree.

## What's here

```
archive/
├── gemma3-270m-health-qa/              # closed-world YAML-QA SFT track
│   ├── README.md                       # what was done, outcomes, where the live code lives
│   ├── bench/                          # 12 dated bench records + 3 paired JSONLs
│   ├── plans/gemma3-270M/              # 856-line Phase 0–4 plan + companions
│   ├── guides/                         # legacy fine-tune guide
│   ├── scripts/                        # finetune.py, merge.py, smoke_test.py, chat_remote.sh
│   ├── model-card/                     # per-model README (IFEval, quantization, prompt strategy)
│   └── (live code lives at src/gemma_tools/_legacy/, tests/_legacy/, data/_legacy/)
├── functiongemma-pre-distil/           # FunctionGemma path before switch to Distil Labs
│   ├── README.md                       # what was tried, why it was abandoned
│   ├── bench/
│   │   ├── eval-summary.md             # consolidated rollup of 43 micro-files
│   │   ├── 2026-05-01_functiongemma-block-e-supplement-repair.md
│   │   ├── 2026-05-01_functiongemma-block-f1-refusal-reweight.md
│   │   └── 2026-05-01_functiongemma-v2-finetune-eval.md
│   ├── data/                           # supplement_dataset.jsonl, raw teacher dumps, refusal2x
│   ├── plans/
│   │   └── phase-d-readme-original.md  # 2321-line original FunctionGemma plan, verbatim
│   ├── scripts/                        # finetune_functiongemma.py (v1, refusal-weighted),
│   │                                   # build_block_e_supplement.py, build_weighted_train.py
│   └── tests/                          # test_finetune_functiongemma_weighting.py (NOT in CI)
└── dispenser-demo-moonshine-streaming/ # Dispenser-demo STT, streaming variant — superseded same day
    ├── README.md                       # why streaming-tiny was provisionally pinned and then flipped
    └── working-recipe.md               # complete build/deploy/smoke recipe, captured before supersession
```

## Why three tracks

- **gemma3-270m-health-qa** — the original closed-world YAML-QA SFT path. The
  model retrieves and quotes facts from `data/health_table_v1.yaml`. Live
  code under `src/gemma_tools/_legacy/` and `tests/_legacy/` still passes
  CI as a working reference for anyone returning to this approach.
- **functiongemma-pre-distil** — the FunctionGemma SFT path before the switch
  to Distil Labs. Local Unsloth-based finetune (v1 with refusal weighting,
  v2 cleaner). v2 lives at `scripts/functiongemma/train/finetune_local.py`
  as the active local fallback; v1 + the Block-E/F1 experiments are here.
- **dispenser-demo-moonshine-streaming** — the streaming variant of Moonshine
  Tiny GGUF, provisionally pinned during dispenser-demo Phase 0 (2026-05-11 AM)
  and superseded by `moonshine-tiny` (non-streaming) the same afternoon after
  a head-to-head proof on the SL2619 showed -38 % wall, -29 % RSS. The
  recipe is preserved here in case Phase 3.5 redesigns voice capture for
  partial-hypothesis streaming, which is the streaming variant's actual win.

## Scope

- **Code in archive is runnable but not on the active critical path.** The
  legacy pytest tests under `tests/_legacy/` remain green and are collected
  by `uv run pytest`. Pre-distil tests under
  `archive/functiongemma-pre-distil/tests/` are NOT collected by default —
  run them manually with `pytest archive/...` if you need to verify.
- **No new bench records or new dated entries here.** New work creates new
  files under `docs/bench-notes/` or `bench/`.
- **Original paths in archived docs may be stale.** The Markdown narratives
  reference pre-refactor paths (e.g. `model.gguf` at repo root rather than
  `releases/.../gguf/model.gguf`). Read them as historical record, not
  current instruction.
