# Bench Records

> **Status: frozen historical bench logs.** Each file is a one-shot record of a specific bench run, authored at run time. Many were authored inside the [SynapticSL2619](https://github.com/nouslogic/SynapticSL2619) project and carried forward when this fine-tune workspace was extracted; those files reference SynapticSL2619's `tools/` layout, not this repo's `src/gemma_tools/` and `data/` layout.

**Read-only.** A new bench run creates a new dated file; existing files are never edited. Do not retroactively rewrite paths in old records — the run log is useful only if it accurately reflects what was executed.

## Naming

`YYYY-MM-DD_<run-tag>.md` for human-readable summaries. `YYYY-MM-DD_<run-tag>.jsonl` for machine-readable per-prompt records (paired with the .md when both exist).

## Two flavors

| Pattern | Example |
|---|---|
| Dated bench summary | `2026-04-28_gemma3-finetuned-final.md` |
| Smoke / diagnostic record | `t5-smoke-20260428-072748.md` (smoke runs predate the rename to `smoke_test.py`) |
| Cross-discipline analysis | `2026-04-27_llama-onnx-plan-review.md`, `2026-04-24_gemma3-270m-practical-evaluation.md` (moved here from the now-removed `docs/analysis/` folder) |

## Authoring a new entry

Per `docs/conventions/doc-update.md §8.1`, the canonical home for the bench protocol is `docs/plans/models-testing-plan.md` (frozen). For new runs in this standalone repo, follow the same format: prompt class breakdown, regex pass-rate, manual rubric, environment fingerprint (model SHA, llama.cpp commit, host).
