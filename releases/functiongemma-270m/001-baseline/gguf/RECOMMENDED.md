# FunctionGemma 270M iter-001 — recommended on-board variant

**`finetuned_functiongemma_q4_0.gguf`** (~224 MiB, sha256 `a484ad50…`).

This is the only INT4/INT8 quantization of the iter-001 student that
preserves the FunctionGemma wire format on the SL2619 board's
`llama-completion` (build `b8925` / `0adede8`, cross-compiled
2026-04-24). Every other variant tested (Q4_K_M, Q5_K_M, Q8_0, IQ4_XS)
fails on board because the older runtime mis-handles K-quant /
mixed-precision scale factors on a 270M model with a 262 144-token
embedding table; the model drops the `<start_function_call>` open token
or stops decoding after `?`.

| Property | Value |
|---|---|
| File | `finetuned_functiongemma_q4_0.gguf` |
| Size | ~224 MiB |
| sha256 | `a484ad50d4b66fdbd6ccb482389eec734b0de9fe988e8811b5e6683daf180e14` |
| BPW | 7.01 |
| Source | `finetuned_functiongemma_fp16.gguf` (sha `1add620fb…`, 518 MiB, BF16) — distil iter-001 deployable |
| Quantize tool | `docs/references/upstream/llama.cpp/build/bin/llama-quantize` (b8981) |
| Single-resident decode tok/s on A55 × 2 | **10.27** (smoke, vs FP16 ~5.87) |
| Single-resident prompt-eval tok/s | **60.1** |
| Single-resident per-prompt wall, cold | **28 s** |
| Sanity (7 board prompts, expected-tool match) | **7/7** |
| Holdout TCE (`eval_holdout_v2_clean.jsonl`, 45 rows, host eval) | 13/45 (28.9 %) — +4.5 pp over FP16 baseline of 24.4 % |

## Production deployment

Only this file should be staged on `/mnt/sdcard/models/functiongemma-270m/`
in production — the other quants and the FP16 baseline can be removed once
the demo is signed off (the page-cache thrash from cohabiting GGUFs
inflated the sweep's per-prompt wall by ~4×).

```bash
# Single-file deploy (drop the others to free /mnt/sdcard page cache)
scp releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_q4_0.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/
ssh nouslogic-sl2619 'sha256sum /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_q4_0.gguf'
# expected: a484ad50d4b66fdbd6ccb482389eec734b0de9fe988e8811b5e6683daf180e14
```

The on-board scripts (`run_prompt.sh`, `chat_board.py`, `ask_board.sh`)
default to this filename; override with the env var `FG_MODEL=<basename>`
or the `--model-name` flag if you need to compare against another variant.

## Why not the others — at-a-glance

| variant | board status | host status |
|---|---|---|
| Q4_K_M | drops `<start_function_call>` open token; emits bare `:NAME{}` | OK (host runtime matches quant tool) |
| Q5_K_M | same — emits `?` then halts on 6/7 prompts | OK |
| Q8_0 | mixed — 3/7 emit proper format, 4/7 stop at `?` | OK |
| IQ4_XS | tokenizer drift ("icksicksicks…" loops); 90/236 tensors fell back during quantize | FAIL on host too (−8.8 pp vs FP16) |

Full evidence + per-row breakdown:
[`docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md`](../../../docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md).

## When to revisit

If `nouslogic-server` is used to refresh `llama-completion` on the board
against `b8981` or newer, re-run `scripts/functiongemma/quantize/build_variants.sh`
+ `scripts/functiongemma/bench.py --mode remote` for each of Q4_K_M, Q5_K_M,
Q8_0 — the wire-format failure mode is version-skew specific and will
likely disappear with a matched runtime, opening the door to a higher-bit
recommendation (Q5_K_M is the most likely upgrade).
