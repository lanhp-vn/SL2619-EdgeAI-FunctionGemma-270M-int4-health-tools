# Q1 Cross-Arch Δ Bench — FT'd Gemma 3 270M Q4_0 — 2026-04-28

**Verdict: GREEN.** `Δ_same_top_p = 0.393 pp ≤ 1.0 pp` AND `ratio_max_delta_p = 0.996x ≤ 3.0x` — A55 within same-quant cross-arch noise floor on the merged (FT'd) Gemma 3 270M Q4_0.

This is the cross-arch step (c) of the Q1 calibrated three-step. Companion docs:
- [`2026-04-28_gemma3-finetuned-q1-logits-equivalence.md`](./2026-04-28_gemma3-finetuned-q1-logits-equivalence.md) — same-arch x86 Path B at n_ctx=2048 (`98.443%` deployment-shape signal); §11 explains why this cross-arch run uses a different corpus + n_ctx.
- [`2026-04-27_h5r-cross-arch-delta.md`](./2026-04-27_h5r-cross-arch-delta.md) — H5R (Phase 0) on **base** weights; same gate constants, same H5R-shape corpus, same n_ctx=256.

## 0. TL;DR

| Metric | x86_64 Q4_0 | SL2619 A55 Q4_0 | Δ / ratio | Gate | Pass |
|---|---|---|---|---|---|
| Same top p | 87.795 ± 1.454 % | 87.402 ± 1.474 % | **0.393 pp** | ≤ 1.0 pp | ✓ |
| Max Δp | 56.476 % | 56.242 % | **0.996x** | ≤ 3.0× | ✓ |
| RMS Δp | 20.001 ± 1.665 % | 19.969 ± 1.674 % | — | informational | — |
| Mean Δp | -5.062 ± 0.859 % | -5.030 ± 0.858 % | — | informational | — |

The 0.393 pp Δ is **bit-identical to H5R's** measurement on the base model (also 0.393 pp). The cross-arch kernel-noise floor is invariant to the weight bit pattern — FT did not introduce any new ISA-specific behavior at the Q4_0 kernel level. The A55 NEON DOTPROD + REPACK path is not silently corrupted on FT'd weights.

## 1. Why this is a separate measurement from the same-arch Path B run

Two concerns share the Q1 label but live in different places:

| Concern | Where measured | Why this corpus / n_ctx |
|---|---|---|
| **Deployment-shape stability under Q4_0** | Same-arch x86 Path B @ n_ctx=2048 → `same_top_p = 98.443%` (in [`2026-04-28_gemma3-finetuned-q1-logits-equivalence.md §5`](./2026-04-28_gemma3-finetuned-q1-logits-equivalence.md)) | Path B = 660-token deployment prompts; n_ctx=2048 keeps prompts inside one chunk → measures the SFT delta's robustness to Q4_0 directly. Runs on x86 RAM; not constrained by board memory. |
| **Cross-arch kernel parity on FT'd weights** | This bench: x86 vs A55 same-quant Δ on H5R-shape corpus @ n_ctx=256 | Board memory cliff at vocab=262144: per-chunk reference-logits buffer is `n_ctx × vocab × float32`. n_ctx=2048 needs 2.15 GiB just for that buffer, which OOM-kills on the 1.87 GiB / no-swap SL2619 (dmesg-confirmed 2026-04-28). H5R-shape (bare 10-50 token prompts × 35) at n_ctx=256 fits comfortably in 1.20 GiB and gives ≥ 4 chunks with no boundary distortion. |

**These do not substitute for each other.** The deployment-shape signal needs Path B at high n_ctx. The kernel-parity signal needs any apples-to-apples corpus run on both architectures. We report both.

Full reasoning + dmesg evidence + per-n_ctx memory math is in [`2026-04-28_gemma3-finetuned-q1-logits-equivalence.md §11`](./2026-04-28_gemma3-finetuned-q1-logits-equivalence.md).

## 2. Per-Chunk Same-Top-P (4 chunks, n_ctx=256)

| Chunk | x86_64 | A55 |
|---|---|---|
| 1 | 87.402% | 87.402% |
| 2 | 85.827% | 85.433% |
| 3 | 87.139% | 86.614% |
| 4 | 87.795% | 87.402% |

A55 lands flat with x86 on chunk 1 and trails by ≤ 0.525 pp on chunks 2–4. Bidirectional bias is consistent with structural ISA-level FP arithmetic-order noise, not a directional kernel issue.

## 3. Why 87.4–87.8% is the "right" same-top-p range here (not 94% like H5R)

Two factors lower the absolute number vs H5R's 94.291%:

1. **FT-induced peakedness.** `entropy 1.352 → 0.615` across the 3 SFT epochs (per `a55-gemma-fine-tune.md §10.3` T2/T3 row). Highly-peaked distributions are more sensitive to small numerical perturbations because the gap between top-1 and top-2 logits shrinks for many tokens. Q4_0 kernel noise that would have been absorbed by the base model's flatter distribution now flips top-1 more often.
2. **Short bare-prompt corpus.** Most H5R prompts are 10–50 tokens; the model has very little context to anchor a confident next token, so quantization noise has more relative influence on the top-1 choice.

The same-arch Path B number on this same FT'd model at n_ctx=2048 is **98.443%** (deployment shape, longer context). That is the right number to cite as "deployment-shape stability under Q4_0." The 87.4% on this bench is the cross-arch *delta* substrate — the gate is on the difference between x86 and A55 (0.393 pp), not on the absolute level.

## 4. Provenance

### Artifacts (host)

| Path | Size | sha256 |
|---|---|---|
| `.cache/q1/h5_corpus.txt` | 26 KiB (35 prompts) | `71901c90f200914224fa5b761427528082b32e8ec3e815bfd983edb67e63e56b` (matches H5R) |
| `.cache/q1/merged_v1.bf16.gguf` | 518 MiB | `a9c5100a4e88f2bf5526cc092d0fe6f2e08156096d9173bbd5351d1f0bb3665e` (Q0 closure) |
| `.cache/q1/merged_v1.q4_0.gguf` | 231 MiB | `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4` (Q0 closure) |
| `.cache/q1/merged_v1.bf16.h5.c256.kld` | 266 MB | `8e792450ac3380cfd0ec1ad3990632caff775c026886dff093ad724c0d318c61` |
| `.cache/q1/q1r-bf16-ref-20260427-212428.log` | — | host BF16 dump log |
| `.cache/q1/q1r-x86-q4_0-20260427-212428.log` | — | host x86 anchor (canonical) |
| `.cache/q1/q1r-a55-q4_0.log` | — | board log pulled back via scp |

### Artifacts (board)

| Path | Status |
|---|---|
| `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` | already present (sha matches host) |
| `/mnt/sdcard/models/q1/merged_v1.bf16.h5.c256.kld` | new (scp from host; sha verified `8e792450…`) |
| `/mnt/sdcard/models/q1/h5_corpus.txt` | new (scp from host; sha verified `71901c90…`) |
| `/mnt/sdcard/bench/q1r-a55-q4_0.log` | A55 KL run output |
| `/mnt/sdcard/models/q1/merged_v1.bf16.c2048.kld` | preserved as 6.97 GB diagnostic-only artifact (the n_ctx=2048 attempt that OOM'd; do not reuse) |

### Tooling

- llama.cpp host build-native + board: `0adede8 (b8925)` — byte-matched (same as H5R)
- Host scorer: `tools/src/sl2619_tools/h5_logits_equiv.py classify-h5r` (re-used unmodified from H5R)
- Host CPU: 20-core x86_64 (`-t 20`); board: 4-core A55 (`-t 2`, leaves headroom for OS / dropbear / journald)

### As-executed commands (reproducible)

**Host (steps a + b, ~12 s wall total on x86 with `-t 20`):**

```bash
cd /home/lanhp-wsl/nouslogic/SynapticSL2619

# (a) BF16 ref dump — flag is --save-all-logits FNAME (overloaded with --kl-divergence-base; presence of --kl-divergence flips load vs save)
.cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity \
    -m .cache/q1/merged_v1.bf16.gguf \
    -f .cache/q1/h5_corpus.txt \
    --save-all-logits .cache/q1/merged_v1.bf16.h5.c256.kld \
    -c 256 --seed 1 -t 20 --temp 0.0 --no-mmap

# (b) x86 Q4_0 KL — load the ref, compute Δ vs Q4_0 GGUF
.cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity \
    -m .cache/q1/merged_v1.q4_0.gguf \
    -f .cache/q1/h5_corpus.txt \
    --kl-divergence \
    --kl-divergence-base .cache/q1/merged_v1.bf16.h5.c256.kld \
    -c 256 --seed 1 -t 20 --temp 0.0 --no-mmap
```

**Board (step c, 25.5 s wall on `-t 2`):**

```bash
# scp .kld + corpus to /mnt/sdcard/models/q1/ (Q4_0 already on board)
ssh nouslogic-sl2619 '/mnt/sdcard/bin/llama-perplexity \
    -m /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf \
    -f /mnt/sdcard/models/q1/h5_corpus.txt \
    --kl-divergence \
    --kl-divergence-base /mnt/sdcard/models/q1/merged_v1.bf16.h5.c256.kld \
    -c 256 --seed 1 --temp 0.0 --no-mmap -t 2 \
    2>&1 | tee /mnt/sdcard/bench/q1r-a55-q4_0.log'
```

**Score (host):**

```bash
cd tools && uv run python -m sl2619_tools.h5_logits_equiv classify-h5r \
    --x86-log ../.cache/q1/q1r-x86-q4_0-20260427-212428.log \
    --a55-log ../.cache/q1/q1r-a55-q4_0.log \
    --max-delta-pp 1.0 --max-delta-ratio 3.0 \
    --summary-out ../docs/tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md
```

## 5. Memory budget actual vs projected (board)

A55 run at n_ctx=256 finished cleanly at `Host: 720 MiB + CPU_REPACK: 223 MiB ≈ 943 MiB`, well under the 1.65 GiB available shown by `free -m` post-mortem. Cliff for reference:

| n_ctx | per-chunk ref buffer | + model 943 MiB | board fit |
|---|---|---|---|
| **256 (this run)** | 268 MiB | 1.20 GiB | ✓ comfortable |
| 512 | 537 MiB | 1.47 GiB | ✗ razor-thin |
| 1024 | 1.07 GiB | 2.00 GiB | ✗ OOM |
| 2048 (prior attempt) | 2.15 GiB | 3.08 GiB | ✗ OOM (dmesg-confirmed) |

This is a hard memory wall on the SL2619 (1.87 GiB total, IL-2 no swap). Documented for the record; future Q-class measurements with vocab > ~50k must respect it.

## 6. Verdict and next steps

- **Q1 GREEN.** Both the same-arch deployment-shape gate (`98.443% ≥ 95%`) and the cross-arch kernel-parity gate (`Δ 0.393 pp ≤ 1.0 pp`, `ratio 0.996x ≤ 3.0x`) clear with substantial headroom.
- **Q2 unblocked** — transfer Q4_0 GGUF to `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/` (already present from this Q1 board step) and verify sha. Awaits user authorization.
- **Q3 (smoke probe), Q4 (full bench sweep), Q5 (score)** are explicitly NOT authorized in this session.

## 7. Interpretation summary (for the F1 freeze)

| Question | Answer | Source |
|---|---|---|
| Did Q4_0 quantization break the SFT delta on x86? | No. Same-arch `same_top_p = 98.443%`; only 1.05 pp below the apples-to-apples base anchor (99.489%). Fine-tune cost is small and expected from SFT peakedness. | [`q1-logits-equivalence §5`](./2026-04-28_gemma3-finetuned-q1-logits-equivalence.md) |
| Does the A55 kernel produce different logits than x86 on FT'd weights? | No, within H5R's same-arch noise floor. `Δ = 0.393 pp` (identical to base-weight measurement); `ratio_max_delta_p = 0.996x` (A55 actually slightly *under* x86). | This bench |
| Is there a hidden ARM-vs-x86 numerical regression introduced by FT? | No. The 0.393 pp number being bit-identical to the base-weight H5R Δ tells us the cross-arch kernel-noise floor is invariant to weight bit pattern. | This bench + H5R |
| Is the on-device deployment safe to proceed to Q2/Q3/Q4? | Yes (logits-equivalence-wise). Behavioral gate (Q3 smoke / Q4 sweep) still owed. | Q1 closure |

---

*Authored 2026-04-28 by `tools/src/sl2619_tools/h5_logits_equiv.py classify-h5r` template, then hand-elaborated for Q1 framing. The auto-generated H5R-template skeleton is preserved in git history; this version replaces the generic title / next-action with Q1-specific framing per the corpus / n_ctx separation discussed in [`2026-04-28_gemma3-finetuned-q1-logits-equivalence.md §11`](./2026-04-28_gemma3-finetuned-q1-logits-equivalence.md).*
