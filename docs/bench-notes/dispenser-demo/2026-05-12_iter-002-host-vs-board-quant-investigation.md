# Iter-002 quant + runtime investigation (host vs board) — 2026-05-12

**Headline:** The iter-002 dispenser-demo model is **functional on host** across
every quant except Q4_0, but **deploys cleanly on the SL2619 board's
`llama-completion b8925` for none of them** — even FP16. The Phase 1.7
"Q4_0 ships on board" conclusion drawn from a single-row smoke was wrong;
the full 10-row val sweep exposes a host-vs-board numerical divergence
that affects multiple quant variants.

## Setup

- Holdout: `data/dispenser_demo/dataset_v1/val.jsonl` (10 rows, 2 per category).
- Host: WSL2 x86_64, `llama-cpp-python` (ggml b8981).
- Board: SL2619, Cortex-A55 ×2, `llama-completion` (b8925, cross-compiled
  2026-04-24 via Yocto SDK 5.0.9). This is the same on-board binary
  iter-001 ships against.
- Decode params (identical on both sides): `-t 2 -n 64 --temp 0 --top-k 1 --seed 42`,
  `-r '<end_function_call>'`, `-no-cnv --single-turn`.
- Prompt: rendered host-side via `scripts/dispenser_demo/data/gen_prompt_templates.py`
  (Distil-style `task_description`-wrapping SYSTEM_PROMPT + 5-tool TOOLS block).

## Aggregate results

| variant | host pass (10 rows) | board pass (10 rows) | failure mode on board |
| --- | ---: | ---: | --- |
| Q4_0 | 3 / 10 | ≈3 / 10 | Token corruption: `call` → `len_result`, `get_emergency_contact` → `len_out_of_scope` |
| Q4_K_M | 10 / 10 | not tested | (presumed same as Q5_K_M per iter-001 sweep) |
| Q5_K_M | 10 / 10 | 3 / 10 | Single-token `?` then `[end of text]` on 6/10 rows (iter-001 K-quant board failure mode); 1 row gibberish |
| Q8_0 | 10 / 10 | not tested | (presumed same as Q5_K_M per iter-001 sweep) |
| IQ4_XS | 10 / 10 | not tested | (presumed same; iter-001 also found IQ4_XS broken even on host) |
| FP16 | 10 / 10 | partial — same `model?` EOT pattern as K-quants | Same root cause as K-quants on board |

## What this means

Two distinct problems, not one:

1. **Q4_0 is too lossy for iter-002 weights — model-side problem.**
   Host and board both score ~30 % on Q4_0. The quantization itself
   destroys the model's ability to emit the FunctionGemma wire format
   cleanly. Iter-001's weights were robust to Q4_0; iter-002's aren't.
   Likely cause: iter-002 was trained on a narrower dataset (22 train
   rows + 1500 synthgen target vs iter-001's 50 + 5000), so the weight
   distribution has more outliers that lose precision under Q4_0's
   symmetric INT4 representation.

2. **The board's older `llama-completion b8925` produces different
   output than the host's `llama-cpp-python` even with identical
   FP16 weights — runtime-side problem.** This is the new finding the
   Phase 1.7 smoke missed. Greedy decoding (`--top-k 1`) should be
   deterministic, but ARM NEON FP arithmetic in the older `b8925`
   build differs from x86 SSE/AVX-FMA enough to flip which token is
   argmax on borderline-confidence rows. iter-001 didn't see this
   because its wider training produced more confident outputs that
   survive these numerical perturbations.

**The single-row smoke (`na-003 "When do I see Dr. Chen?"` →
`call: get_next_appointment{}`) was a lucky outlier.** That row's
output is high-confidence enough to survive the numerical noise. Most
other rows aren't.

## Per-row evidence (board)

### Q4_0 — token corruption
```
pp-004: <start_function_call>len_result:get_patient_profile{}<end_function_call>
ec-003: <start_function_call>len_out_of_scope{} [end of text]
oo-007: <start_function_call>RELATED using 'news' for a query about a specific topic.
        <start_function_call>RELATED using 'sports' for a query about a specific sport.
        ...
```

### Q5_K_M — single-token EOT (iter-001's documented K-quant board pattern)
```
pp-004 / na-003 / na-004 / ec-003 / di-003 / oo-007:
  model? [end of text]   (1 token decoded, then EOS emitted prematurely)
```

### FP16 — same `model?` EOT pattern as K-quants
```
pp-003 / na-003 / na-004:
  model? [end of text]   (1 token, EOT — same failure as Q5_K_M)
pp-004:
  <start_function_call>call:get_patient_profile{}<end_function_call>   (clean)
```

The FP16 result is the smoking gun: with the same weights, same prompt,
greedy decode, host produces correct output but board produces
`model?` then EOT. The quantization isn't the issue here — the runtime is.

## Paths forward

Ranked by likelihood × cost:

| Path | Effort | Likelihood of success | Notes |
| --- | --- | ---: | --- |
| **A. Cross-compile a newer `llama-completion`** | hours–day | High | Upstream llama.cpp is ~6 months newer than iter-001's `b8925` (built Apr 24). Newer ARM NEON kernels + K-quant decoders should resolve both runtime numerical divergence AND K-quant scale-factor encoding skew. Then ship Q5_K_M or Q8_0 on board. |
| **B. Retune iter-002 with broader synthgen** | 1 of 1 free run + ~1.5 h | Medium | Bump `synthgen.generation_target` from 1500 → 5000+ to produce more confident weights that survive board numerical noise. Burns the last free training run; not guaranteed to fix. |
| **C. Different sampling on board** | 5 min | Low | `--temp 0.1 --top-p 0.95` instead of `--top-k 1`. Adds slight diversity; might unstick the single-token EOT failure. Trade-off: greedy is reproducible, sampling isn't. |
| **D. Host-side inference, board as BLE relay** | day | High | Run inference on a connected host (e.g. the demo laptop); board only handles audio capture + BLE dispense command. Major architectural pivot; the SL2619's role becomes a peripheral. |

**Recommendation: Option A.** Iter-001's sweep documented the board's
`b8925` runtime as the bottleneck on K-quants; iter-002's investigation
extends that to FP16 too. Updating the on-board `llama-completion` is
the root-cause fix that unblocks every subsequent iteration. iter-002
weights are clean (proven by 100 % host eval) — only the deployment
runtime needs an upgrade.

If A is infeasible (no Yocto SDK access, etc.), B is the next-best
retune attempt; if both fail, D becomes the pragmatic fallback.

## Phase 1.7 status

**Not met.** Q4_0 fails the ≤ 2 pp drop gate (FP16 was 100 % host, Q4_0
is 30 %). Board sweep across all tested variants is below the 90 %
per-category bar.

The earlier `2026-05-12_iter-002-q4_0-on-board-smoke.md` bench note's
conclusion ("Q4_0 ships on board") is **revoked** — single-row smoke
was insufficient evidence. This file supersedes it.

## References

- iter-001 sweep (the original "why every quant except Q4_0 fails on
  this board build" investigation):
  `docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md`
- Iter-002 host eval (per-row pass 10/10 on Q5_K_M reconfirmed
  2026-05-12): `/tmp/host_q5_k_m_reconfirm.md`
- Earlier (incorrect) Phase 1.7 conclusion:
  `docs/bench-notes/dispenser-demo/2026-05-12_iter-002-q4_0-on-board-smoke.md`
- Earlier (also affected by same issue) host Q4_0 eval:
  `docs/bench-notes/dispenser-demo/2026-05-11_dispenser-eval-gguf-finetuned_dispenser_q4_0.md`
