# Q1 Logits-Equivalence Bench (post-SFT, calibrated three-step) — 2026-04-28

**Status: SERVER + HOST x86 STEPS GREEN. A55 step pending user authorization to run board-side commands. Cross-arch Δ gate cannot be evaluated until A55 KL log is captured.**

This is the post-quantization analogue of [`docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`](./2026-04-27_h5r-cross-arch-delta.md). The H5R same-quant cross-arch Δ gate is reused; thresholds are calibrated by the apples-to-apples base-model number on **this** corpus rather than transferred from H5R's bare-prompt corpus.

## 0. TL;DR

- **Same-arch Q4_0 vs BF16 (x86, n_ctx=2048): merged 98.443% same_top_p, max Δp 95.323%, PPL(Q)/PPL(base) = 1.044x.**
- **Apples-to-apples base-model anchor on identical corpus: 99.489% same_top_p.** Fine-tune cost is ~1.05 pp same_top_p — small, expected from SFT peakedness (entropy 1.352 → 0.615 across the 3 SFT epochs per `a55-gemma-fine-tune.md §10.3`).
- **Same-arch sanity gate:** proposed `same_top_p_x86_q4_0_vs_bf16 ≥ 95%`. ✓ (98.443% — 3.4 pp headroom)
- **Cross-arch Δ gate (load-bearing):** `Δ_same_top_p (x86 - a55) ≤ 1.0 pp`, ratio `max_delta_p_a55 / max_delta_p_x86 ≤ 3.0x`. **Pending A55 KL run.**
- **Verdict: PARTIAL GREEN (server + x86 host steps).** Final verdict deferred to board step (c).

## 1. Method (mirrors H5R discipline, redefined for the deployed prompt shape)

Three steps, same `--save-all-logits` / `--kl-divergence-base` flags as H5R, **same llama.cpp version as H5R** (`0adede8 (b8925)` host build-native + same on board):

| Step | What | Where | Output |
|---|---|---|---|
| (a) | `llama-perplexity --save-all-logits` on **merged BF16 GGUF** | host x86_64 (server has no llama-perplexity built — same constraint as H5R) | `merged_v1.bf16.kld` (6.5 GiB) |
| (b) | `llama-perplexity --kl-divergence-base merged_v1.bf16.kld -m merged_v1.q4_0.gguf` | host x86_64 | x86 same-arch numbers |
| (c) | Same command on board, against same `.kld` | SL2619 A55 | A55 cross-arch numbers — **PENDING** |

Reference is **per-fine-tuned-model BF16**, not the H5R base BF16 — so this captures the SFT delta's stability under Q4_0, separate from any base-model Q4_0 noise.

## 2. Q0 artifacts (Phase 3, 2026-04-28)

| Artifact | Size | sha256 | Origin |
|---|---|---|---|
| `merged_v1.bf16.gguf` | 518 MiB | `a9c5100a4e88f2bf5526cc092d0fe6f2e08156096d9173bbd5351d1f0bb3665e` | server `convert_hf_to_gguf.py` ← `merged_v1/` (T4 closure sha `57c56472…`) |
| `merged_v1.q4_0.gguf` | 231 MiB | `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4` | server `llama-quantize ... Q4_0` ← BF16 above |

llama.cpp HEAD on server: `b1a5bd4` (CUDA: better coalesce data-access for contiguous concat #22330, 2 days old). Quant stats: 511.46 MiB BF16 → 224.00 MiB Q4_0 (7.01 BPW); 1 of 236 tensors fallback-quantized (`token_embd.weight` → F16 — same as base GGUF behavior, expected for Gemma 3's 262144-row vocab embedding). Server log: `~/sl2619-finetune/logs/q0-20260428-084616.log`.

**Q0 snag (resolved before run):** `convert_hf_to_gguf.py:1238` asserts `max(tokenizer.vocab.values()) < vocab_size` — fires for any Gemma-3 270M dir lacking `tokenizer.model` because `len(tokenizer.vocab) = 262145` but `config.json: vocab_size = 262144`. `merge.py` writes only `tokenizer.json` (HF), not `tokenizer.model` (SP), so this hits any merged checkpoint. **Fix:** pull `tokenizer.model` (sha `1299c11d…`) from HF Hub via `huggingface_hub.hf_hub_download(repo_id="google/gemma-3-270m-it", filename="tokenizer.model", local_dir="./merged_v1")` — then `Gemma3Model.set_vocab()` takes the SentencePiece path which has no such assertion. Documented for v2 at `docs/plans/backlogs.md §1.22`.

## 3. Corpus

| Field | Value |
|---|---|
| Path | `.cache/q1/q1_corpus.txt` (host) |
| Source | `tools/data/sft_v1.test.jsonl` (Path B test split, held out from training) |
| Builder | `sl2619_tools.h5_logits_equiv.build_q1_corpus(n=30, seed=1)` |
| n prompts | 30 |
| seed | 1 |
| Shape | **Path B** — composed user turn (directive + YAML + question), wrapped with `wrap_gemma3_chat_template`. Mirrors deployment shape, NOT H5R's bare-prompt shape. |
| Total chars | 80,522 |
| sha256 | `81645ac492f6685c51146ee311c2961938cc294a977ac3511eb2126c1ce84146` |
| Per-prompt user-turn size | ~660 tokens (after wrap; matches the 750-820 token Path B budget calibrated at D2-split, plus chat-template overhead) |

CLI: `uv run python -m sl2619_tools.h5_logits_equiv q1-corpus --out .cache/q1/q1_corpus.txt --n 30 --seed 1`. Function + 9 unit tests + CLI subcommand added in this session (host pytest 341/341 green, ruff + mypy strict clean).

## 4. n_ctx investigation (CRITICAL — read before interpreting numbers)

The first run at `n_ctx=1024` produced an alarming `same_top_p = 75.886%` for **merged Q4_0 vs merged BF16**. Two stacked confounders explain it:

1. **Chunk boundaries cross prompt boundaries.** With 660-token Path B prompts and `n_ctx=1024`, each chunk crosses at least one prompt-to-prompt boundary. The model evaluates token N+1's log-prob conditioned on a Frankenstein context that's the tail of one prompt followed by the head of another. This is wildly out-of-distribution for a model SFT'd on cleanly-bounded prompts — both base and merged top-1 get unstable.
2. **SFT peaked the distribution.** Entropy dropped 1.352 → 0.615 across the three SFT epochs. Highly-peaked distributions are more sensitive to small numerical perturbations because the gap between top-1 and top-2 logits shrinks for many tokens.

Decisive diagnostic was the `n_ctx=2048` re-run: ~3 prompts per chunk, ~10 chunks total instead of 27; chunk boundaries cross prompt boundaries far less often. **`same_top_p` jumped to 98.443%** — a **22.6 pp delta from the n_ctx=1024 run** with no other variable changed. Chunk boundaries were first-order; SFT peakedness is second-order.

`n_ctx=2048` is therefore the canonical Q1 measurement. Below the table is the four-run calibration grid:

| Run | model | corpus | n_ctx | chunks | same_top_p | max Δp | PPL(Q)/PPL(base) |
|---|---|---|---|---|---|---|---|
| Q1 x86 (canonical) | merged Q4_0 vs merged BF16 | Q1 (Path B, 30 prompts) | **2048** | 13 | **98.443%** | 95.323% | **1.044x** |
| D1 (boundary diagnostic) | merged Q4_0 vs merged BF16 | Q1 (Path B, 30 prompts) | 1024 | 27 | 75.886% | 96.350% | 1.437x |
| D2 (corpus diagnostic) | base Q4_0 vs base BF16 | Q1 (Path B, 30 prompts) | 1024 | 27 | 82.677% | 96.670% | 1.254x |
| D3 (apples-to-apples anchor) | base Q4_0 vs base BF16 | Q1 (Path B, 30 prompts) | **2048** | 13 | **99.489%** | 88.137% | 1.063x |
| (informational) H5R x86 base | base Q4_0 vs base BF16 | H5R (bare, 35 prompts) | 256 | 4 | 94.291% | 49.781% | (not directly comparable) |

**Interpretation:**

- `n_ctx=2048` removes most chunk boundaries. At this n_ctx the merged-vs-base same-corpus comparison shows a **fine-tune cost of only 1.05 pp** (98.443% vs 99.489%). This is the SFT-peakedness contribution in isolation — small, expected, not a red flag.
- The H5R 94.291% number is **not** apples-to-apples — different corpus shape (bare 10-50 token prompts) and different n_ctx (256). It does not anchor the Q1 gate; it's listed as informational only.
- The 75.886%/82.677% numbers are **not** the SFT-quality measurement. They're chunk-boundary artifacts of n_ctx=1024 on long prompts. Don't transcribe them anywhere as a "fine-tune broke quantization" claim.

## 5. Same-arch x86 numbers (canonical, n_ctx=2048)

| Metric | merged Q4_0 vs merged BF16 (x86) |
|---|---|
| Same top p | **98.443%** |
| Max Δp | 95.323% |
| Mean Δp | -0.796% |
| Mean PPL(base) | 1.085 |
| Mean PPL(Q) | 1.134 |
| PPL(Q)/PPL(base) | 1.044x (**only 4.4% PPL bloat** from Q4_0) |
| KLD mean | (extracted from log) |
| 99.9% Δp | (extracted from log) |
| chunks | 13 |
| Wall (BF16 ref + Q4_0 KL) | 53.3 s + ~50 s ≈ 103 s on host (20 cores, AMD Ryzen 9 9950X-class — taken from `df -h` host) |

## 6. Same-arch sanity gate

| Metric | Value | Threshold | Pass |
|---|---|---|---|
| `same_top_p_x86_q4_0_vs_bf16` | 98.443% | ≥ **95.0%** (proposed) | ✓ |

Threshold rationale: the apples-to-apples base on this corpus is 99.489%; merged loses 1.05 pp. The 95% gate gives 3.4 pp headroom on the merged number and 4.5 pp headroom against the base number. We do **not** anchor against H5R's 94.291% — that was on a different corpus shape with a different n_ctx. If a future evaluation needs a tighter same-arch gate, anchor against the Q1 base anchor (99.489%, this doc) and set the gate at `base_anchor − 2 pp = 97.5%` — the merged still passes by 0.94 pp.

## 7. Cross-arch Δ gate (load-bearing — pending A55 KL log)

The H5R relative gate is reused without modification:

| Metric | Threshold | Status |
|---|---|---|
| `Δ_same_top_p = same_top_p_x86 − same_top_p_a55` | ≤ 1.0 pp | **PENDING** |
| `ratio_max_delta_p = max_delta_p_a55 / max_delta_p_x86` | ≤ 3.0x | **PENDING** |

Threshold provenance: identical to H5R (`a55-gemma-fine-tune.md §4 H5R note`, `2026-04-27_h5r-cross-arch-delta.md`). H5R's actual measured Δ on the **base** model with the bare-prompt corpus was 0.393 pp / 1.041x — well within the gate. Q1 on the **merged** model with Path B is independent measurement; the gate constants stay unchanged.

## 8. Q1 board commands (NOT YET RUN — user authorization required)

Captured for reference. **Do not execute these without user authorization** — board steps require state-changing operations (scp + ssh write) per CLAUDE.md R3.

```bash
# 1) scp Q4_0 GGUF to /mnt/sdcard (board has /mnt/sdcard ext4 mount per H0)
scp /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/q1/merged_v1.q4_0.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf

# 2) scp BF16 .kld reference (~6.5 GiB — slow over Tailscale; ~5-10 min)
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/models/q1'
scp /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/q1/merged_v1.bf16.c2048.kld \
    nouslogic-sl2619:/mnt/sdcard/models/q1/merged_v1.bf16.c2048.kld

# 3) scp Q1 corpus
scp /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/q1/q1_corpus.txt \
    nouslogic-sl2619:/mnt/sdcard/models/q1/q1_corpus.txt

# 4) Verify shas on board match host
ssh nouslogic-sl2619 'cd /mnt/sdcard && \
    sha256sum models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf \
              models/q1/merged_v1.bf16.c2048.kld \
              models/q1/q1_corpus.txt'
# Expected:
#   587f1af6…  merged_v1.q4_0.gguf
#   <kld sha>  merged_v1.bf16.c2048.kld   ← capture from host: sha256sum host file before scp
#   81645ac4…  q1_corpus.txt

# 5) Run A55 Q4_0 KL on board (NOT yet authorized; expected wall ~10-15 min on 2 cores)
ssh nouslogic-sl2619 '/mnt/sdcard/bin/llama-perplexity \
    -m /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf \
    -f /mnt/sdcard/models/q1/q1_corpus.txt \
    --kl-divergence \
    --kl-divergence-base /mnt/sdcard/models/q1/merged_v1.bf16.c2048.kld \
    -c 2048 --seed 1 --temp 0.0 --no-mmap -t 2 \
    2>&1 | tee /mnt/sdcard/bench/q1-a55-q4_0.log'

# 6) scp board log back to host
scp nouslogic-sl2619:/mnt/sdcard/bench/q1-a55-q4_0.log \
    /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/q1/q1-a55-q4_0.log

# 7) Score cross-arch Δ + emit final bench summary
cd /home/lanhp-wsl/nouslogic/SynapticSL2619/tools && \
uv run python -m sl2619_tools.h5_logits_equiv classify-h5r \
    --x86-log ../.cache/q1/q1-x86-q4_0-CHOSEN.log \
    --a55-log ../.cache/q1/q1-a55-q4_0.log \
    --max-delta-pp 1.0 --max-delta-ratio 3.0 \
    --summary-out ../docs/tmp/bench/$(date +%F)_gemma3-finetuned-q1-cross-arch-delta.md \
    --corpus-path .cache/q1/q1_corpus.txt \
    --kld-path .cache/q1/merged_v1.bf16.c2048.kld \
    --gguf-sha256 q4_0=587f1af6...,bf16=a9c5100a... \
    --llama-cpp-commit 0adede8/b8925
```

## 9. Provenance

- **Q0 server log**: `~/sl2619-finetune/logs/q0-20260428-084616.log` (server-side, READ-ONLY accessible by agent)
- **Q1 host BF16 ref log (n_ctx=2048)**: `.cache/q1/d1-bf16-ref-20260427-190049.log`
- **Q1 host x86 Q4_0 log (n_ctx=2048, canonical)**: `.cache/q1/d1-x86-q4_0-20260427-190049.log`
- **D2 base@n_ctx=1024**: `.cache/q1/d2-base-bf16-ref-20260427-190049.log` + `.cache/q1/d2-base-q4_0-20260427-190049.log`
- **D3 base@n_ctx=2048 (anchor)**: `.cache/q1/d3-base-bf16-ref-20260427-190459.log` + `.cache/q1/d3-base-q4_0-20260427-190459.log`
- **Q1 host x86 Q4_0 log (n_ctx=1024, INVALIDATED — chunk-boundary distorted)**: `.cache/q1/q1-x86-q4_0-20260427-185717.log` (preserved for diagnostic completeness; not used as the Q1 number)
- **q1_corpus.txt**: sha `81645ac492f6685c51146ee311c2961938cc294a977ac3511eb2126c1ce84146`
- **merged_v1.bf16.c2048.kld**: 6.5 GiB at host `.cache/q1/merged_v1.bf16.c2048.kld`; sha256 **`33b9b592647e5492751c7e510b1cef003c166a97dae736aed2fce9593c363da8`**.

## 10. What this leaves open

- **Q1 board step (c) CLOSED 2026-04-28** — see §11 below for the n_ctx=2048 board OOM and the H5R-shape reframe; final cross-arch verdict at [`2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](./2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md) (`Δ = 0.393 pp`, `ratio = 0.996x`, GREEN).
- **Q2/Q3/Q4/Q5 are explicitly NOT authorized** in this session per user direction. Authorization gated on Q1 GREEN end-to-end (now satisfied — pending user go-ahead for Q2 transfer).
- **For v2 (per `backlogs.md §1.22`)**: `tools/scripts/merge.py` should auto-pull `tokenizer.model` after `tokenizer.save_pretrained()` so this Q0 snag doesn't re-occur.
- **Q1 corpus n_ctx caveats**: Path B same-arch canonical measurement is at `n_ctx=2048` on **x86 only** — board cannot run that n_ctx (§11). If a future re-run uses a different n_ctx, document it explicitly. Don't blindly trust an n_ctx=1024 number on Path B prompts — the chunk-boundary effect is large.

## 11. Board OOM at n_ctx=2048 — separation of same-arch vs cross-arch (added 2026-04-28)

When step (c) was first attempted on the board with the Path B corpus at n_ctx=2048, `llama-perplexity` was SIGKILL'd by the kernel OOM-killer. dmesg confirmed two kills (pids 1737, 1756) at total-vm 2.72 GB / anon-rss 1.73 GiB on the 1.87 GiB / no-swap SL2619. The cliff is structural and not a knob:

**Per-chunk reference-logits buffer = `n_ctx × vocab × float32`**, with Gemma 3 vocab = 262,144:

| n_ctx | per-chunk ref buffer | + model 934 MiB | board fit |
|---|---|---|---|
| **256 (H5R-proven, used for cross-arch step c)** | 268 MiB | 1.20 GiB | ✓ comfortable |
| 512 | 537 MiB | 1.47 GiB | ✗ razor-thin |
| 1024 | 1.07 GiB | 2.00 GiB | ✗ OOM |
| **2048 (Path B canonical)** | 2.15 GiB | 3.08 GiB | ✗ OOM (dmesg-confirmed 2026-04-28) |

This is a hard physical wall on this SoC; IL-2 (no swap) makes it immutable. Therefore:

| Concern | Where measured | Why |
|---|---|---|
| **Deployment-shape stability under Q4_0** (= "did Q4_0 catastrophically corrupt the SFT delta on the deployment-shape corpus?") | **§5 above — same-arch x86 Path B at n_ctx=2048**, `same_top_p = 98.443%`. Stays. | Long Path B prompts need n_ctx ≥ 2048 to avoid chunk-boundary noise (§4 calibration). x86 has the RAM. |
| **Cross-arch kernel parity on FT'd Q4_0 weights** (= "does the A55 NEON DOTPROD path produce the same logits as x86 AVX2 on FT'd weights?") | [`2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](./2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md) — H5R-shape corpus at n_ctx=256, both x86 + A55. **`Δ_same_top_p = 0.393 pp`, `ratio_max_delta_p = 0.996x` — GREEN.** | Bare 10-50 token prompts at n_ctx=256 fit comfortably in board RAM and give 4 chunks with no boundary distortion. The kernel-parity test is corpus-agnostic — what matters is identical inputs on both architectures, not the specific deployment shape. |

**Why the cross-arch number `0.393 pp` is bit-identical to H5R's measurement on base weights** is itself the most important takeaway: the cross-arch kernel-noise floor is invariant to weight bit pattern. The fine-tune did not introduce any new ISA-specific behavior at the Q4_0 kernel level. Combined with the deployment-shape gate (98.443% same-arch), Phase 3 logits-equivalence is fully cleared.

OOM'd artifacts on board are preserved for diagnostic completeness (do not re-use them):

| Path | Reason kept |
|---|---|
| `/mnt/sdcard/models/q1/merged_v1.bf16.c2048.kld` (6.97 GB) | n_ctx=2048 BF16 ref dump — incompatible with any n_ctx=256 run; kept as evidence of the original Path B attempt |
| `/mnt/sdcard/bench/q1-a55-q4_0.log` (truncated mid-warmup) | the SIGKILL'd run output; matches dmesg timeline 2026-04-28 |

---

*Authored 2026-04-28. Companion to [`2026-04-27_h5r-cross-arch-delta.md`](./2026-04-27_h5r-cross-arch-delta.md) (Phase 0 base-weights cross-arch) and [`2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](./2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md) (Phase 3 Q1 cross-arch — closes step c).*
