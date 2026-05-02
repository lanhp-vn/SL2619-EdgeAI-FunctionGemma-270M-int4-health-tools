# archive/gemma3-270m-health-qa/

Frozen snapshot of the gemma3-270m health-QA SFT track. The model answered
patient-record questions by retrieving and quoting facts from
`data/health_table_v1.yaml` (closed-world YAML-QA).

**Do not edit.** Read as historical record. Live code lives under
`src/gemma_tools/_legacy/`, `tests/_legacy/`, `data/_legacy/` — those are
NOT in this archive directory because they need to remain importable and
their CI tests still need to pass.

## What was done

| Phase | Work | Outcome |
|---|---|---|
| Q0 — pool collection | 1400 Alpaca-shape triples authored via Pro Perplexity / Claude / ChatGPT / Gemini / DeepSeek; consolidated into `data/_legacy/clean_sft_dataset.json` | DONE |
| Q1 — schema + classifier | `gemma_tools._legacy.sft_dataset.classify_record` collapses 7 verbal classes to 4 splitter-relevant: `fact_lookup`, `fact_absence`, `domain_refusal`, `summarization` | DONE |
| Q2 — splitter | Stratified 80/10/10 with paraphrase-aware bench-leakage routing → `sft_v1.{train,val,test}.jsonl` (1023/126/110); audit JSONL alongside | DONE |
| Q3 — Path A ablation | Raw-pair `sft_v1_pathA.{train,val,test}.jsonl` (no composed prompt) | DONE — ablation only |
| Q4 — bench harness | `gemma_tools._legacy.bench_prompt`, `bench_remote`, `bench_eval` against `data/_legacy/prompts.yaml` 15-prompt suite | DONE — 53 host unit tests green |
| H5/H5R — logits-equivalence gate | KL divergence cross-arch (host vs board) — `Δ_same_top_p ≤ 1.0 pp`, `max_delta_p ratio ≤ 3.0×` | DONE — green at no-repack-fa-off |
| T5 — base BF16 vs merged BF16 smoke | `scripts/smoke_test.py` (now in this archive) on 5 prompts | DONE-WITH-NOTE — P1 phrasing failure adjudicated as v1 caveat |
| Phase 3 final bench | Q4_0 GGUF on SL2619 board, 15-prompt suite | DONE — see `bench/2026-04-28_gemma3-finetuned-final.md` |

## Outcomes

- v1 corpus is **proof-of-concept**, not product-quality. Phase 3 numbers
  freeze the v1 demo; future fine-tuning passes need to expand size +
  phrasing diversity per the gaps catalogued in the Phase 3 doc.
- Q4_0 quantization landed cleanly; ~9.5 tok/s decode on bench-sized
  prompts on A55 × 2.
- H5R cross-arch equivalence proved the host bench predicts board behavior
  within tolerance — board runs are not required for every iteration.

## Reachable live entry points

For anyone returning to this track:

| Action | Command |
|---|---|
| Build SFT artifacts | `uv run sft-build` (entry point → `gemma_tools._legacy.sft_build:main`) |
| Run host bench | `uv run bench-prompt` (→ `gemma_tools._legacy.bench_prompt:main`) |
| Run remote bench (board) | `uv run bench-remote --ssh-host nouslogic-sl2619 ...` |
| Score bench output | `uv run bench-eval` |
| Logits-equivalence | `uv run logits-equiv` |
| Interactive chat probe | `uv run chat-probe` |

The full CLI flag set is in each module's `argparse` block. Tests under
`tests/_legacy/` exercise every entry point.

## Pointers

- `plans/gemma3-270M/a55-gemma-fine-tune.md` — 856-line Phase 0–4 plan (verbatim)
- `plans/gemma3-270M/a55-gemma-h5-logits-equivalence.md` — KL gate spec
- `plans/gemma3-270M/models-testing-plan.md` — bench protocol + prompt classes
- `bench/` — 12 dated bench records spanning 2026-04-24 through 2026-04-28
- `model-card/README.md` — per-model README (IFEval, quantization, prompt strategy)
- `guides/fine-tuning-gemma3-270m-small-models.md` — narrative walk-through of the SFT recipe
- `scripts/finetune.py`, `merge.py`, `smoke_test.py`, `chat_remote.sh` — pre-FunctionGemma scripts
