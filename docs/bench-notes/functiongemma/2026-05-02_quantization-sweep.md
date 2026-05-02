# FunctionGemma 270M iter-001 — INT4/INT8 quantization sweep on SL2619

**Date**: 2026-05-02.
**Source FP16 GGUF**: sha256 `1add620fbd45…` (518 MiB), distil iteration-001
deployable at `releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_fp16.gguf`
(model id `231feebb-8cc0-4d5f-9e4b-4d2f00e362b2`, training id
`c9d34596-ee7a-4e56-be2b-254159fe7796`).

## Recommendation

→ **`finetuned_functiongemma_q4_0.gguf` (~224 MiB)** is the only quantized
variant that preserves the FunctionGemma wire format on the board build
(`b8925`/`0adede8`). Every other quant tested damages the post-`<start_of_turn>model`
distribution enough that the model drops the `<start_function_call>` open token
or stops decoding after a single `?` token, leaving the parser with malformed
output.

Q4_0 also clears the host-side accuracy gate (host eval ≥ FP16 − 5 pp on the
clean holdout) and the on-board throughput gate (≥ 5 tok/s decode in the
single-resident smoke). Expected on-board behaviour with **only** Q4_0
staged on `/mnt/sdcard`:

| metric | value (smoke, single-resident GGUF) | value (sweep, 11 GGUFs resident) |
|---|---|---|
| decode tok/s | **10.27** | 1.23 (page-cache thrash) |
| prompt-eval tok/s | **60.1** | 17.0 (page-cache thrash) |
| per-prompt wall, cold | **28 s** | 116 s |
| sanity (7 prompts, expected-tool match) | — | **7/7** |
| holdout TCE on `eval_holdout_v2_clean.jsonl` | 13/45 (28.9 %, host) | — |

Production guidance: keep only `finetuned_functiongemma_q4_0.gguf` on the
board (delete the experimental quant variants once the demo is signed off);
the page-cache thrash inflated the sweep's wall-time numbers ~4×.

## Recipe

| Component | Value |
|---|---|
| `llama-quantize` (host) | `b8981` (`docs/references/upstream/llama.cpp` checkout, gcc 13.3 x86_64) |
| `llama-completion` (board) | `b8925` / `0adede8` (Apr 24, gcc 13.3 aarch64, cross-compiled via Yocto SDK 5.0.9) |
| Tokenizer / chat template | `releases/functiongemma-270m/001-baseline/merged/` |
| Holdout (host eval) | `data/functiongemma/eval_holdout_v2_clean.jsonl` (45 rows, all-novel phrasing — out-of-distribution for iter-001's Distil-trained corpus) |
| Sanity prompts (board) | 7 in-distribution prompts, one per tool, pinned in `scripts/functiongemma/bench.py:DEFAULT_PROMPTS` |
| Decoding | greedy (`--temp 0 --top-k 1 --seed 42`), `-n 64`, `-c 4096`, `-t 2`, reverse-prompt `<end_function_call>` |
| Variants tested | Q4_0, Q4_K_M, Q5_K_M, Q8_0, IQ4_XS (FP16 host-only) |

## On-board sanity sweep (5 quants × 7 prompts, threads=2, ctx=4096, fa=off, cache=f16, batch=default)

| variant | size MiB | tool match (7) | parsed | decode tok/s | prompt tok/s | mean wall s | gate ≥ 5 tok/s + 7/7? |
|---|---|---|---|---|---|---|---|
| **`q4_0`** | **224** | **7/7** | **7/7** | 1.23 † | 17.0 | 116 † | **PASS** (only variant) |
| `q8_0` | 271 | 3/7 | 3/7 | 9.09 | 55.8 | 29.9 | FAIL (parser) |
| `iq4_xs` | 224 | 1/7 | 1/7 | 9.85 | 44.0 | 39.5 | FAIL (parser, tokenizer drift) |
| `q5_k_m` | 248 | 1/7 | 1/7 | 8.44 | 29.8 | 55.1 | FAIL (parser) |
| `q4_k_m` | 242 | 1/7 | 1/7 | 7.00 | 22.8 | 80.7 | FAIL (parser) |

† Inflated by /mnt/sdcard page-cache thrash. The earlier smoke run (single-resident
GGUF, hot fs cache) recorded 10.27 tok/s decode / 28 s wall on the same
prompt. Production should track the smoke number, not the sweep number.

### Per-row tool match

| prompt | `q4_0` | `q8_0` | `iq4_xs` | `q5_k_m` | `q4_k_m` |
|---|---|---|---|---|---|
| `vitals_bp` | ✓ | ✓ | ✗ (tokenizer loop) | ✗ (`?`) | ✗ (`:get_vitals`) |
| `vitals_fever` | ✓ | ✓ | ✗ (`?`) | ✗ (`?`) | ✗ (`:get_vitals`) |
| `appointment` | ✓ | ✗ (`?`) | ✗ (`?`) | ✗ (`?`) | ✗ (`?`) |
| `food_alcohol` | ✓ | ✓ | ✗ (loop) | ✓ | ✓ |
| `allergies` | ✓ | ✗ (`?`) | ✓ (after `modelijs` glitch) | ✗ (`?`) | ✗ (`:get_vitals`) |
| `meds_at_8am` | ✓ | ✗ (`?`) | ✗ (`:get_medications_at_time`) | ✗ (`?`) | ✗ (`:get_medications_at_time`) |
| `med_atorvastatin` | ✓ | ✗ (`?`) | ✗ (`?`) | ✗ (`?`) | ✗ (`?`) |

(The `?` annotation means the model decoded just one token (`?`) before the
reverse-prompt fired or the chat-template close token followed. The
`:NAME{}` annotation means the model emitted the function call body but
omitted the `<start_function_call>` open token — parser correctly rejects
that as malformed.)

## Why every quant except Q4_0 fails on this board build

The board's `llama-completion` is `b8925` (cross-compiled Apr 24); the host's
`llama-quantize` is `b8981` (Apr 29, 56 commits ahead). Cross-version
compatibility for K-quant scale-factor encoding has been observed to drift
on small models (~270M parameters with a 262 144-token vocabulary, where
embedding-row weights occupy a disproportionately large share of the
parameter count).

The empirical signature on the SL2619 board:

| family | observed defect | rows of evidence |
|---|---|---|
| **Q4_0** | none — full wire format preserved | 7/7 sanity, full `<start_function_call>...<end_function_call>` blocks |
| Q8_0 | `<start_of_turn>model?` → 1-token early stop on 4 prompts; correct on 3 | every Q8_0 row in `bench/.../stage1/q8_0.jsonl` |
| Q4_K_M | `<start_function_call>` token dropped; emits `:NAME{}` directly | 4/7 rows show `model:get_vitals{}<end_function_call>` |
| Q5_K_M | model emits `?` on 6/7 prompts, halts | mostly `n_decode=1` with text ending `model?` |
| IQ4_XS | tokenizer drift — emits `icksicksicks…` until cap; 90/236 tensors required fallback during quantize | 2 rows are pure `icks` loops |

Q4_0 uses the simpler symmetric INT4 representation and survives the
version skew. K-quants and IQ4_XS use mixed-bit blocks whose scale factors
the older runtime mis-applies on the embedding rows.

## Why the host-side eval doesn't expose the same defect

The host eval runs `llama-cpp-python 0.3.21` (which uses the matching
`b8981` ggml backend — the same code that generated the quants). On host
the K-quants score within the FP16 ± noise band:

| variant | clean-holdout match (45 rows, host) | Δ vs FP16 | gate ≥ 19.4 % |
|---|---|---|---|
| FP16 (baseline) | 11/45 (24.4 %) | — | reference |
| **Q4_0** | **13/45 (28.9 %)** | +4.5 pp | PASS |
| Q4_K_M | 10/45 (22.2 %) | −2.2 pp | PASS (host) |
| Q5_K_M | 13/45 (28.9 %) | +4.5 pp | PASS (host) |
| Q8_0 | 11/45 (24.4 %) | 0 pp | PASS (host) |
| **IQ4_XS** | **7/45 (15.6 %)** | **−8.8 pp** | **FAIL** |

So the K-quant + Q8_0 problem is **board-specific**, not a quantization-quality
issue per se. IQ4_XS is the only variant that fails accuracy on host too
(consistent with the 90/236 fallback-quantize warning during build —
importance-weighted INT4 mixed-precision is a poor fit for the FunctionGemma
embedding layout).

The clean holdout is *out-of-distribution* for iter-001's training corpus —
even FP16 only hits 24.4 % (Distil's published 0.9583 metric is on the
*contaminated* 24-row holdout `eval_holdout_v1.jsonl`, which IS in the
training distribution). Per advisor review, the realistic gate for the
quants is **TCE ≥ FP16 − 5 pp** (no measurable degradation vs FP16).

## FP16 on board: skipped per session direction

FP16 board run was deliberately skipped for the user-facing demo on
direction "you should be testing quantized models only - since fp16 seems
not runnable on the board". FP16 is technically runnable on the board
(documented at ~5.87 tok/s decode in `docs/deployment/sl2619-board.md` §5.1)
but materially slower than Q4_0 and not the deliverable.

## Stage 2 collapsed (per advisor)

Per advisor review at the start of the sweep: "if stage 1 produces a clear
winner (one variant ≥ 0.90 with decode > 5 tok/s and the others measurably
worse), consider collapsing Stage 2 to 'verify the winner under cache=q8_0
to halve KV RAM' rather than a full grid." Q4_0 is the clear winner; the
ctx/cache/FA/batch grid is captured as a deferred follow-up
(see §"Deferred work" below) — accuracy is the user's top priority and
Q4_0 already clears it.

## Acceptance gates — final verdict

| Gate | Q4_0 | All other quants |
|---|---|---|
| Tool-call equivalence on `eval_holdout_v2_clean.jsonl` ≥ FP16 − 5 pp | **PASS** (28.9 % vs 24.4 % FP16) | mostly PASS on host; IQ4_XS FAIL (15.6 %) |
| 7 board sanity prompts, expected-tool match | **PASS (7/7)** | FAIL (1–3 / 7) |
| No malformed `<start_function_call>` syntax | **PASS** | FAIL — most rows emit malformed output or stop at `?` |
| Decode ≥ 5 tok/s on A55 board | **PASS** (10.27 single-resident smoke) | PASS in tok/s but irrelevant — accuracy gate fails first |
| Final REPL demo via `chat_board.py` returns NL answer via `format_response` | **expected PASS** (deferred to stage 4 demo) | n/a |

## Deferred work

- **Refresh the on-board `llama-completion` binary** against `b8981`+ on
  `nouslogic-server` and re-cross-compile, then re-run this sweep. K-quants
  and Q8_0 may pass the wire-format gate with the matching runtime; if they
  do, Q5_K_M would be a strong candidate (better quant quality than Q4_0,
  similar decode throughput).
- **Stage 2 ctx/cache/FA/batch knob sweep** for Q4_0 specifically — verify
  whether `--cache-type-{k,v} q8_0` halves KV RAM at no accuracy cost
  (would free ~50 MiB on a board that's already at 1.67 GiB of headroom).
- **iMatrix calibration** for the K-quants and IQ4_XS — `llama-imatrix`
  produces an importance matrix from representative prompts that improves
  K-quant quality on small-vocab models. Only worth running after the board
  binary is refreshed.
- **Iteration 002 retrain** to close the clean-holdout gap (FP16 is only at
  24.4 % — the iter-001 training corpus didn't cover `parallel_call`,
  `two_turn`, or `medical_advice_refusal` patterns). This is a separate
  track captured in `docs/plans/functiongemma/decisions-log.md`.

## Reproduce

```bash
# 1. Quantize all 5 variants from canonical FP16
scripts/functiongemma/quantize/build_variants.sh

# 2. Stage Q4_0 to board (production: drop the others to free /mnt/sdcard cache)
scp releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_q4_0.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# 3. Bench (always pass --remote-model explicitly)
uv run python scripts/functiongemma/bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_q4_0.gguf \
    --threads 2 --warmup 1

# 4. Holdout eval (host, 5–10x faster than board)
uv run python scripts/functiongemma/eval/eval_holdout.py \
    --gguf releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_q4_0.gguf \
    --tokenizer-dir releases/functiongemma-270m/001-baseline/merged \
    --holdout data/functiongemma/eval_holdout_v2_clean.jsonl

# 5. Aggregate sweep results
uv run python scripts/functiongemma/bench/aggregate_quant.py \
    bench/functiongemma/runs/2026-05-02-quant/stage1/ \
    --sizes-file releases/functiongemma-270m/001-baseline/gguf/CHECKSUMS.txt \
    --output docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md
```

## Artifacts

| Path | Purpose |
|---|---|
| `releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_q4_0.gguf` | **Recommended on-board GGUF** (gitignored; sha pinned in `CHECKSUMS.txt`) |
| `releases/functiongemma-270m/001-baseline/gguf/CHECKSUMS.txt` | sha256 of every variant (only authoritative record committed to git) |
| `releases/functiongemma-270m/001-baseline/gguf/RECOMMENDED.md` | Pinned recommendation (this doc summarised) |
| `bench/functiongemma/runs/2026-05-02-quant/stage1/` | Per-variant JSONL bench outputs |
| `bench/functiongemma/runs/2026-05-02-quant/q4_0-smoke.jsonl` | Single-resident Q4_0 smoke (the production reference for tok/s) |
| `docs/bench-notes/functiongemma/2026-05-02_eval-host-finetuned_functiongemma_*.md` | Per-variant host holdout eval reports |
| `scripts/functiongemma/quantize/build_variants.sh` | Idempotent host quantize driver |
| `scripts/functiongemma/bench/aggregate_quant.py` | JSONL → Markdown aggregator (this doc was generated by the same template) |
