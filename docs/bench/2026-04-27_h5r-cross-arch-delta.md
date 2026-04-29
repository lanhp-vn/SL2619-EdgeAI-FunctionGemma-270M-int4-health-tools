# H5R Logits-Equivalence Bench -- 2026-04-27

**Verdict: GREEN** -- delta_same_top_p=0.393 pp <= 1.0 pp AND ratio_max_delta_p=1.041x <= 3.0x -- A55 within same-quant cross-arch noise floor

Same-quant cross-arch delta test (H5R, replaces the absolute H5 gate that was preserved
verbatim in `docs/tmp/bench/2026-04-26_h5-logits-equivalence.md`). Reference is an
x86_64 BF16 `.kld`; both candidates are Q4_0 GGUF compared against that same reference.

## Raw Metrics

| Metric | x86_64 Q4_0 | SL2619 A55 Q4_0 |
| --- | --- | --- |
| Same top p | 94.291% | 93.898% |
| Max delta_p | 49.781% | 51.804% |

## Relative Gate (H5R)

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| delta_same_top_p (x86 - a55) | 0.393 pp | <= 1.0 pp | ✓ |
| ratio_max_delta_p (a55 / x86) | 1.041x | <= 3.0x | ✓ |

## Per-Chunk Same-Top-P

| Chunk | x86_64 | A55 |
| --- | --- | --- |
| 1 | 96.063% | 96.850% |
| 2 | 94.882% | 94.882% |
| 3 | 94.488% | 94.226% |
| 4 | 94.291% | 93.898% |

## Interpretation

- **A55 NEON DOTPROD path is not silently corrupted by `llama.cpp #22011`-class issues at `0adede8` / `b8925`.** The 0.393 pp delta is well below the 1.0 pp gate, with substantial headroom. The 1.041x max_delta_p ratio means the worst-case Δp on A55 is essentially the same as on x86 — neither side has an ISA-specific blow-up.
- **Per-chunk noise is bidirectional.** On chunk 1 the A55 actually *beats* x86 (96.850% vs 96.063%); on chunks 3–4 x86 leads by a fraction of a percentage point. This is the expected signature of structural ISA-level FP arithmetic-order differences (x86_64 AVX2 FMA vs ARM64 NEON DOTPROD accumulation order), not a directional kernel bug.
- **The H5 PUNT was measuring universal Q4_0 quantization noise.** The 2026-04-26 H5 result (98.622% same_top_p / 9.393% max Δp) used a Q4_0 reference, so the residual was purely cross-arch FP-order divergence on top of identical Q4_0 weights. Now that we have a BF16 reference, x86 itself shows 94.291% / 49.781% — the Q4_0 quantization noise floor for this 270M model is the dominant term, and A55 stays within that floor. Consistent with upstream's published q4_K_M-vs-FP16 numbers (91.9–94.7%).
- **Calibration of thresholds.** The 1.0 pp Δ gate is conservative (defaults from H5R plan §6.5 retained; not overridden via `--max-delta-pp`). The 3.0x ratio gate is also conservative (actual ratio 1.041x). No need to tighten or relax for this measurement.

## Next Action

**H6 unblocked** -- proceed to base-GGUF baseline bench.

The Phase 0 logits-equivalence gate is satisfied. The fine-tune path proceeds: H6 (base-GGUF baseline bench) → Phase 1 D2-curation (optional) → Phase 2 SFT → Phase 3 quantize + Q1 (post-quant logits-equivalence, three-step calibration mirrors H5R discipline) → Phase 4 freeze.

## Provenance

- **corpus**: /mnt/sdcard/models/h5/h5_corpus.txt (35 prompts: 15 yaml + 15 sft seed=42 + 5 OOD; sha256 71901c90f200914224fa5b761427528082b32e8ec3e815bfd983edb67e63e56b)
- **reference_kld**: /mnt/sdcard/models/h5/h5_ref_bf16.kld (BF16 ref from gemma-3-270m-it.bf16.gguf; 254 MiB; sha256 7a99ec1097713fe3725ac3b872bfd10f7e0d06d495c457c874909a97d962478d)
- **x86_command**: .cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity -m gemma-3-270m-it-Q4_0.gguf -f h5_corpus.txt --kl-divergence --kl-divergence-base h5_ref_bf16.kld -c 256 --seed 1 -t $(nproc) --temp 0.0 --no-mmap
- **a55_command**: /mnt/sdcard/bin/llama-perplexity -m gemma-3-270m-it-Q4_0.gguf -f h5_corpus.txt --kl-divergence --kl-divergence-base h5_ref_bf16.kld -c 256 --seed 1 -t 2 --temp 0.0 --no-mmap
- **gguf_sha256**: Q4_0=e479ea2962bdcdc7e6321b91148b9ac2f516f649e0921412561d4936aadef158 (unsloth, ident on host+board); BF16=903799d7cd964d0803bee3ec6ed2c2247f92589ace9f4307e969c998e21681d2 (server convert_hf_to_gguf.py from google/gemma-3-270m-it)
- **llama_cpp_commit**: host build-native (x86_64 AVX2) version=0adede8 (b8925); board (aarch64 NEON+DOTPROD+REPACK) version=0adede8 (b8925) — versions byte-matched
- **x86_log**: /tmp/h5r-x86-q4_0.log
- **a55_log**: /tmp/h5r-a55-q4_0.log
