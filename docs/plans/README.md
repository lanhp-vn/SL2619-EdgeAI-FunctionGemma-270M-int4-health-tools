# Plans

> **Status: frozen historical narratives.** These plans were authored inside the [SynapticSL2619](https://github.com/nouslogic/SynapticSL2619) project and carried forward when this fine-tune workspace was extracted into a standalone repo. Internal paths inside the documents reference SynapticSL2619's `tools/` layout (e.g. `tools/data/`, `tools/src/sl2619_tools/`), not this repo's `src/gemma_tools/` and `data/` layout.

Read these as ground-truth records of what was planned and what was decided. Do not retroactively rewrite the SynapticSL2619 paths; the documents are useful only if they reflect the plan-of-record at the time it was executed. New plans go in new files alongside these.

## Files

| File | What it is | Phase |
|---|---|---|
| [`a55-gemma-fine-tune.md`](a55-gemma-fine-tune.md) | LoRA SFT recipe, hyperparameters, Q0–F5 phase plan, post-mortems on tokenizer / quantization issues | Phase 3 (executed) |
| [`a55-gemma-h5-logits-equivalence.md`](a55-gemma-h5-logits-equivalence.md) | KL-divergence cross-arch equivalence gate (H5R: `Δ_same_top_p ≤ 1.0 pp`, `max_delta_p ratio ≤ 3.0×`) | Phase 0 (executed, GREEN) |
| [`models-testing-plan.md`](models-testing-plan.md) | Bench protocol, prompt classes, gate definitions, P1–P5 path adjudications | Phase 1.5 (closed) |

## Authoring conventions

DRY-exempt per `docs/conventions/doc-update.md §8.2`. New plans should use the same style: numbered phases, status banner at top, dated update entries at bottom. Keep one plan per topic; do not split.
