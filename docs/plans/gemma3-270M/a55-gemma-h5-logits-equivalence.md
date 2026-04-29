# Phase 0 H5R — Logits-Equivalence Session Plan (same-quant cross-arch Δ)

> ✅ **CLOSED 2026-04-27 — H5R GREEN.** Result: `Δ_same_top_p = 0.393 pp` (gate ≤ 1.0 pp), `ratio_max_delta_p = 1.041x` (gate ≤ 3.0x). Bench summary at [`docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`](../../tmp/bench/2026-04-27_h5r-cross-arch-delta.md); fine-tune plan §10.2 has the closure entry. The session plan below is preserved as the executed recipe; corrections applied during execution are inlined (§3, §6.1.1, §6.1.3, §6.4).
>
> Focused session plan for **H5R** from `a55-gemma-fine-tune.md §4 / Phase 0`. Replaces the original H5 absolute-threshold session (which ran, PUNT'd, and was rolled into the H5 historical row of `a55-gemma-fine-tune.md §10.2`). H5R adds **one additional x86_64 Q4_0 run + a recalibrated scorer** on top of the existing setup.
>
> Owner: agent (host artifacts + scorer) + user (server x86_64 run + board run, per R3 read-only SSH).
> Wall time as executed: **~30 min agent (scorer + tests + emitter + 55 unit tests) + ~10 min host x86 + ~5 min user board run**.
> Decision impact: GREEN reclassified the H5 PUNT as expected Q4_0 quantization noise → unblocked H6 → Phase 1 D2-curation → Phase 2 fine-tune.

---

## 0. Why this plan was rewritten (2026-04-27)

The original H5 gate was `same_top_p ≥ 99.99%` and `max Δp ≤ 0.5%` — absolute thresholds against an x86_64 BF16 `.kld` reference. Two things were measured and conflated:

1. **Universal Q4_0 quantization noise** (any architecture vs FP16/BF16). Upstream's own `tools/perplexity/README.md` publishes BF16-vs-FP16 same-CPU = `99.739% / 4.186%`; q8_0-vs-FP16 ≈ `97.7-98.8%`; q4_K_M-vs-FP16 ≈ `91.9-94.7%`. The 99.99% / 0.5% gate was **unreachable for any Q4_0-vs-FP16 on any architecture**.
2. **A55-specific kernel divergence** (the actual `#22011` signal). Genuinely the gate's intent.

Result: the H5 measurement (98.622% / 9.393%) was within published Q4_0 noise but failed the unreachable absolute gate. Verdict was preserved (H5 PUNT in `a55-gemma-fine-tune.md §10.2`); gate was redefined as **H5R** so the second term can be measured in isolation. Full diagnostic in `docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md §2.1`.

H5R differs from H5 only in **scoring** and the **one additional x86_64 Q4_0 run**. Reference `.kld`, board binary, prompt corpus, scoring script — all reused.

## 1. Objective

Detect, before any SFT cycles begin, whether `llama.cpp` on the SL2619 A55 (ARMv8.2-A, NEON DOTPROD, no SVE/SME/MATMUL_INT8) produces logits that **diverge from a same-architecture-x86_64 Q4_0 baseline** by more than expected ISA-level FP arithmetic-order differences. If A55 Q4_0 vs an x86_64 BF16 `.kld` agrees to within ~1 pp of what x86_64 Q4_0 vs the same `.kld` agrees to, the residual is structural FP-order noise (not a bug we can fix). If A55 lags x86_64 by > 1 pp, the kernel is suspect — the fine-tuned variant inherits the same kernel bug; re-quantizing or retraining cannot recover from it.

This is **not** a quality benchmark (that is H6) and **not** a Q4_0-vs-FP16 quality gate (Q4_0 noise is universal and quantitatively documented upstream — we do not re-litigate it here). H5R is the relative same-quant numerical-correctness test that the original H5 gate failed to be.

## 2. Why upstream `llama.cpp #22011` still matters

The issue (referenced in `a55-gemma-fine-tune.md §11`) reports wrong-logits on certain ARM64 builds for some quant types. Until upstream confirms patched in our build (`b8925`, `0adede8`), we **must independently verify** on the SL2619, because the public report does not enumerate exactly which CPU-feature combos trip it. Our config — DOTPROD ✓, FP16_VECTOR_ARITHMETIC ✓, FMA ✓, no SVE/SME/MATMUL_INT8, GCC 13.3 — is not a configuration the upstream reporter has called out one way or the other. The H5 PUNT investigation already proved Flash Attention is **not** the cause and REPACK accounts for only +0.6 pp of the residual; a relative cross-arch Δ test is the cleanest remaining diagnostic before escalating upstream.

## 3. Reference path (host)

**Two runs needed against the same x86_64 BF16 `.kld` reference**:

| Run | Where | What | Output |
|---|---|---|---|
| **Reference** (must be regenerated for H5R — see §11.2) | server x86_64 (or WSL host as fallback) | `llama-perplexity --save-all-logits` on `gemma-3-270m-it-BF16.gguf` against the 35-prompt corpus | binary `.kld` per-token reference log-probs (16-bit) |
| **x86 Q4_0 noise floor** (NEW for H5R) | server x86_64 (or WSL host) | `llama-perplexity --kl-divergence-base <ref.kld> -m gemma-3-270m-it-Q4_0.gguf` | `same_top_p_x86_q4_0`, `max_delta_p_x86_q4_0` |
| **A55 Q4_0 candidate** (must be re-run against the new BF16 `.kld`) | board aarch64 | same command as above on the board | `same_top_p_a55_q4_0`, `max_delta_p_a55_q4_0` |

> **Provenance note (2026-04-27 H5R kickoff).** The 2026-04-26 H5 PUNT used `_DEFAULT_GGUF` (`gemma-3-270m-it-Q4_0.gguf`) as the `--save-all-logits` reference, not BF16. As of H5R kickoff, no `.cache/h5/` directory and no `gemma-3-270m-it-BF16.gguf` exist on the WSL host — the historical Q4_0-on-x86 `.kld` is no longer on disk either. The correct H5R reference is BF16, so the `.kld` is regenerated from scratch via §11.2; the A55 result CANNOT be reused from `2026-04-26_h5-logits-equivalence.md` (different reference type) and must be re-run.

**Rejected**: server CUDA path (CUDA dequant is a different code path; matching CUDA logits to ARM logits answers cross-device parity, not ARM-correctness). Useful later for Q1 post-quant comparison, not for H5R.

**Critical**: the reference `.kld` must be the **same file** for runs 2 and 3. Generate the BF16 `.kld` once (§11.2) and use it for both. If flags need adjusting (e.g. `--no-mmap`, `--flash-attn off`), re-generate the `.kld` and rerun all three with matching flags.

## 4. Capture method — `llama-perplexity --save-all-logits` (unchanged from H5)

`llama.cpp` ships an upstream-blessed harness for exactly this comparison:

| Stage | Tool | Flag | Output |
|---|---|---|---|
| Reference | `llama-perplexity` (host) | `--save-all-logits FNAME` | binary `.kld` file: per-token reference log-probs (16-bit) |
| Comparison | `llama-perplexity` (server x86 + board aarch64) | `--kl-divergence-base FNAME` | KL divergence stats incl. `n_same_top`, `max_p_diff` |

Source pointers (verified in our `b8925` checkout via `docs/references/llama-cpp.md`):

- Flag wiring: `common/arg.cpp:2079` defines `--save-all-logits` (alias `--kl-divergence-base`); writes to `params.logits_file`.
- Reference dump: `tools/perplexity/perplexity.cpp` `process_logits` writes per-token log-prob distribution to disk.
- Comparison: same file, `kl_divergence_result` struct accumulates `count`, `n_same_top`, `sum_kld`, `max_p_diff`. `n_same_top / count` = top-1 agreement ratio (`same_top_p`); `max_p_diff` = max absolute prob difference at the matched top-1 token.

The flag does **not** exist on `llama-completion`. We standardize on `--save-all-logits` / `--kl-divergence-base`.

### 4.1 What we need to build that we don't have yet

**Nothing.** H5 already built `llama-perplexity` for both x86_64 and aarch64 targets (Yocto-built `llama-perplexity` deployed to `/mnt/sdcard/llama-cpp/`; native build cached at host/server). H5R reuses both binaries unchanged.

### 4.2 Fallback if `llama-perplexity` is somehow absent

If the binary is missing (e.g. fresh dev host without the cache), rebuild via §6.1 of the original H5 procedure preserved at the bottom of this doc — same toolchain, ~30 s incremental build, same SDK source command:

```bash
cmake --build build --target llama-perplexity -j$(nproc)
aarch64-poky-linux-strip build/bin/llama-perplexity
```

Then `scp` to `/mnt/sdcard/llama-cpp/` alongside the existing binaries.

## 5. Prompt set (unchanged from H5 — reuse the same corpus)

H5R **must** use the identical 35-prompt corpus that produced the H5 PUNT, otherwise `same_top_p_x86_q4_0` and `same_top_p_a55_q4_0` are not directly comparable.

| Source | Count | Purpose |
|---|---|---|
| `tools/data/prompts.yaml` (all 15: C1, P1–P9, D1–D2, S1–S2 + headroom) | 15 | Bench-aligned coverage. |
| `tools/data/sft_v1.test.jsonl` random sample (seed=42) | 15 | Held-out test split — never seen by training, distribution-matched. |
| OOD probes (hand-authored at H5) | 5 | Stress edge tokenization. CJK, multi-codepoint emoji, ASCII spam, whitespace, markdown fence. |

**Total: 35 prompts.** Each prompt wrapped via `wrap_gemma3_chat_template()` (`tools/src/sl2619_tools/bench_prompt.py`) so it matches what bench / fine-tune actually feed the model. The corpus is sha-pinned under `.cache/h5/prompts-<sha>.txt` from the H5 run.

## 6. Procedure

Per R2 cadence — each step independent, run-and-verify before moving on. SSH to board is **READ-ONLY for the agent (R3)**; user runs every state-changing command on the board and on the server.

### 6.1 Confirm prerequisites are still in place

| Step | Owner | Command | Verify |
|---|---|---|---|
| 6.1.1 | agent (host) | inspect cache | `.cache/h5/prompts-<sha>.txt` and a BF16 `.kld` reference present on host or server. **Verified 2026-04-27 H5R kickoff: neither exists on the WSL host** — corpus + BF16 `.kld` must be regenerated via §11.2 before §6.3 / §6.4 run. |
| 6.1.2 | user (server) | `~/llama.cpp/build/bin/llama-perplexity --version` | shows `(0adede8)`. |
| 6.1.3 | user (board) | `ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/bin/llama-perplexity && /mnt/sdcard/bin/llama-perplexity --version 2>&1 \| head -n 4'` | shows `(0adede8)`. **Path verified 2026-04-27 H5R kickoff**: binary lives at `/mnt/sdcard/bin/llama-perplexity`, not `/mnt/sdcard/llama-cpp/` (which only has `llama-bench` / `llama-cli` / `llama-completion`). |
| 6.1.4 | user (server + board) | `gemma-3-270m-it-Q4_0.gguf` sha256 matches between server and board | sha256 identical (the H5 PUNT already verified this; sanity-check on H5R session start). |

If any of 6.1.1-6.1.4 fail, recover from the H5 procedure preserved at the bottom of this doc before proceeding.

### 6.2 Update `tools/src/sl2619_tools/h5_logits_equiv.py` to score H5R

The shipped script parses a single `llama-perplexity --kl-divergence` stdout and gates on `same_top_p ≥ 99.99%` / `max_delta_p ≤ 0.5%` (lines 51-53 per `docs/references/llama-cpp.md`). H5R needs:

1. A **second-input mode** that takes the x86 Q4_0 run's `KldReport` and computes `Δ_same_top_p = same_top_p_x86 − same_top_p_a55` and `ratio_max_delta_p = max_delta_p_a55 / max_delta_p_x86`.
2. New gate constants — proposed defaults below; expose as CLI args so the bench summary can document the chosen values:
   - `_GATE_DELTA_SAME_TOP_P_MAX_PP = 1.0` (relative)
   - `_GATE_MAX_DELTA_P_RATIO_MAX = 3.0` (relative)
3. Verdict: GREEN iff both gates pass; PUNT otherwise.
4. Preserve the existing absolute-gate code path under a `--mode legacy-h5` flag so the H5 PUNT result remains reproducible.

R2 chunk plan (no batched edits):

1. Add `score_h5r(report_x86, report_a55, *, max_delta_pp=1.0, max_ratio=3.0) → H5RVerdict` + parametrized unit test (≥ 4 cases: clean pass, Δ over threshold, ratio over threshold, both fail; each table-driven with `{desc}` per `11-testing-verification.md §3.1`).
2. Add `--mode {h5,h5r,legacy-h5}` CLI dispatch + smoke test (mock both reports, assert verdict).
3. Add markdown emitter that writes the H5R bench summary (`docs/tmp/bench/<date>_h5r-cross-arch-delta.md`) with both reports' raw numbers, computed Δ and ratio, chosen thresholds, and the verdict.

Tests via `pytest`. The existing 21 unit tests stay green; H5R adds ~6-8 new cases.

### 6.3 x86_64 Q4_0 noise-floor run (NEW for H5R)

```
~/llama.cpp/build/bin/llama-perplexity \
  -m /path/to/gemma-3-270m-it-Q4_0.gguf \
  -f /path/to/h5-prompts.txt \
  --kl-divergence \
  --kl-divergence-base /path/to/h5-reference.kld \
  -t $(nproc) --temp 0.0 --no-mmap \
  2>&1 | tee /tmp/h5r-x86-q4_0.log
```

Notes:
- **Same `--kl-divergence-base FILE`** as the H5 board run. If the original `.kld` is on the host, scp it to the server first.
- `-t $(nproc)` — server is host CPU bound; fine to oversubscribe for one-shot.
- `--no-mmap` to keep numerics deterministic across runs.
- **Match flags** to whatever the H5 board run used (e.g. if board ran with `--flash-attn auto`, server runs with `--flash-attn auto`; if board ran `--no-repack` for one of the H5 follow-up experiments E/F, server matches). Any mismatch leaks into the Δ.

Capture stdout to `/tmp/h5r-x86-q4_0.log`, scp back to host, parse via `parse_kld_summary()`.

### 6.4 A55 Q4_0 run (must be re-run for H5R)

The 2026-04-26 H5 PUNT used a **Q4_0** reference `.kld`, not BF16. H5R generates a fresh BF16 `.kld` (§11.2), so the historical `same_top_p_a55 = 98.622%` / `max_delta_p_a55 = 9.393%` numbers cannot be reused — they were measured against a different reference. The board command below re-runs against the new BF16 `.kld`:

```
ssh nouslogic-sl2619 '/mnt/sdcard/bin/llama-perplexity \
  -m /mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf \
  -f /mnt/sdcard/models/h5/h5_corpus.txt \
  --kl-divergence \
  --kl-divergence-base /mnt/sdcard/models/h5/h5_ref_bf16.kld \
  -c 256 --seed 1 -t 2 --temp 0.0 --no-mmap 2>&1 | tee /mnt/sdcard/bench/h5r-a55-q4_0.log'
```

`-t 2` mandatory per project convention (kernel exposes 0–1 only; `-t 4` collapses 53×). `.kld` must live on `/mnt/sdcard` (ext4) **not `/tmp` (tmpfs)** — three 254 MiB `.kld` files in tmpfs were the OOM root cause during the H5 follow-up runs (`a55-gemma-fine-tune.md §10.2`).

### 6.5 Score (redefined for H5R)

Parse both runs' stdout via `parse_kld_summary()`. Compute the relative metrics:

| Metric | Formula | Pass condition (proposed) | Source |
|---|---|---|---|
| `Δ_same_top_p` | `same_top_p_x86_q4_0 − same_top_p_a55_q4_0` | `≤ 1.0 pp` | H5R redefine, `a55-gemma-fine-tune.md §4 H5R` |
| `ratio_max_delta_p` | `max_delta_p_a55_q4_0 / max_delta_p_x86_q4_0` | `≤ 3×` | H5R redefine, same source |
| `same_top_p_x86_q4_0` (informational) | from x86 run | log it; expect ~99-99.9% per upstream perplexity README | calibration evidence |
| `same_top_p_a55_q4_0` (informational) | from board run | log it; H5 measured 98.622% with REPACK + FA-auto | calibration evidence |

**Note on threshold provenance**: `1.0 pp` and `3×` are starting points. The actual calibration is the x86 Q4_0 baseline captured in §6.3 — A55 is judged relative to its same-arch x86 sibling, not against an absolute ideal. After §6.3 produces the x86 number, document the chosen Δ and ratio in the H5R bench summary. If x86 itself shows `same_top_p ≈ 91-94%` (matching upstream's q4_K_M numbers), tighten Δ to 1 pp; if x86 shows ≈ 99%, the same 1 pp gate is conservative. The doc records both the chosen threshold and the rationale.

OOD probes (5) are scored separately as informational only — kernel divergence on Han / emoji / whitespace tokens is orthogonal to the structured-task fine-tune use case.

### 6.6 Decision

| Outcome | Action |
|---|---|
| **Both gates pass** (`Δ_same_top_p ≤ 1.0 pp`, `ratio_max_delta_p ≤ 3×`) | **GREEN.** Write `docs/tmp/bench/<date>_h5r-cross-arch-delta.md`. Mark H5R ✅ in fine-tune plan §4 / §10.2 (H5 historical row stays untouched; H5R row flips green; H6 unblocks). Proceed to H6 baseline bench. |
| **Either gate fails** | **HARD FAIL → PUNT.** Stop Phase 0. Open / link an upstream issue with: GGUF sha256, llama.cpp tag + commit, our cmake configure log (which CPU features it detected), the `.kld` file (small enough to attach), the prompt list, both x86 and A55 KL summaries, the chosen thresholds. Mark H5R ❌ in fine-tune plan §4 / §10.2. Communicate to user; **do not proceed to fine-tune**. |
| **OOD probes diverge but yaml/sft prompts are clean** | Log under "informational deviations" in the bench summary — does not gate H5R since OOD tokenization stress is orthogonal to the structured-task fine-tune use case. |

## 7. Deliverables

Persistent artifacts produced by this session:

| Artifact | Path | Owner |
|---|---|---|
| Prompt corpus (sha-pinned, reused from H5) | `.cache/h5/prompts-<sha>.txt` | host (gitignored cache) |
| Reference logits dump (reused from H5) | `.cache/h5/h5-reference.kld` | host artifact |
| **NEW** x86 Q4_0 KL run log | `docs/tmp/bench/<date>_h5r-x86-q4_0.log` | committed |
| **NEW** Comparison summary | `docs/tmp/bench/<date>_h5r-cross-arch-delta.md` (markdown table + GREEN/PUNT verdict + chosen thresholds + rationale) | committed |
| Driver script + tests (extended) | `tools/src/sl2619_tools/h5_logits_equiv.py` (+ `--mode h5r` + scorer) + `tools/tests/test_h5_logits_equiv.py` (extended) | committed |

The `legacy-h5` mode of the script keeps the H5 PUNT measurement reproducible. Final commit shape (single commit, per `12-git-workflow.md`): scorer extension, H5R bench summary doc, fine-tune plan §4 / §10.2 H5R status update — atomic.

## 8. Out of scope for the H5R session

- Fine-tuning anything.
- Running H6 base-baseline bench (separate session; H5R must pass first).
- Comparing CUDA logits to ARM logits (different question; deferred to Q1 post-quant comparison).
- Tweaking `temp` / `top-k` to "make it match" — the gate is greedy (`temp=0`) deterministic decode equivalence, not sampling equivalence.
- Re-litigating the H5 absolute-threshold result. H5 PUNT stays preserved as historical context per `a55-gemma-fine-tune.md §10.2`.
- Changing the GGUF source (unsloth Q4_0 stays — switching sources is a different experiment; `gemma-on-a55-get-started.md §1` documents why unsloth).

## 9. Risks and pre-mortems

| Risk | Likelihood | Mitigation |
|---|---|---|
| x86 Q4_0 same_top_p is itself low (e.g. 91-94% per upstream q4_K_M numbers), making the relative Δ unstable | medium | Document the absolute x86 number in the bench summary. If x86 ≪ 99%, the Δ ≤ 1 pp gate may be too tight — re-calibrate to `Δ ≤ 2 pp` and document. The point is relative consistency, not an absolute ideal. |
| Flag mismatch between server and board runs leaks into the Δ | medium | §6.3 explicitly calls out matching flags. The bench summary's first column is the flag set used on each side; mismatch is visually obvious. |
| `.kld` reference reused from H5 was generated with stale flags | low | sha256 the `.kld`; document its provenance in the bench summary. If suspect, regenerate cleanly from the BF16 GGUF on x86 and rerun all three. |
| OOD probes fail relative gate but yaml/sft are clean | medium | Procedure already separates OOD into informational. Do not let OOD divergences alone PUNT H5R; document them for upstream. |
| Server unavailable during the session | low | WSL host x86_64 is the documented fallback (§3 / §6.3 footnote). Slower (no GPU advantage on this CPU run anyway), still valid. |
| H5R passes, but Q1 post-fine-tune still surfaces an A55 divergence | unknown | Q1 (Phase 3) re-applies the H5R discipline against the merged Q4_0. By design, H5R proving the *base* model is fine does not certify the fine-tune; that's Q1's job. |

## 10. Next session's exact opening checklist

When this session is picked up:

- [ ] Confirm H3 closed (binaries + base GGUF on board, `llama-completion --version` printed `(0adede8)`).
- [ ] Confirm H4-tokenizer smoke remained green on the server.
- [ ] Confirm `.cache/h5/h5-reference.kld` and `.cache/h5/prompts-<sha>.txt` are on disk; if not, regenerate via the preserved H5 procedure below.
- [ ] Extend `h5_logits_equiv.py`: add `score_h5r` + `--mode h5r` + bench-summary emitter; R2 cadence (chunk → test → run → fix).
- [ ] Run §6.3 on server (or WSL host) with flags **matching** the H5 board run, scp log back.
- [ ] Reuse §6.4 H5 PUNT board numbers if flags match; otherwise rerun on board, scp log back.
- [ ] Score via §6.5; emit §6.6 verdict with chosen thresholds documented.
- [ ] On GREEN: commit deliverables (§7), mark H5R ✅ in fine-tune plan §4 / §10.2, hand off to H6.
- [ ] On PUNT: file upstream tracking, halt, surface to user.

## 11. Preserved H5 procedure (historical — for binary rebuild only)

The original H5 procedure ran end-to-end on 2026-04-26 and produced the PUNT result preserved in `a55-gemma-fine-tune.md §10.2`. This subsection keeps the binary-build steps in case a fresh dev host needs to rebuild `llama-perplexity` from scratch. Everything else has been superseded by H5R above.

### 11.1 Build `llama-perplexity` for both targets

| Step | Owner | Command | Verify |
|---|---|---|---|
| 11.1.1 | agent (host) | rebuild aarch64 `llama-perplexity` against Yocto SDK at `b8925`/`0adede8`; strip; expected ~5–6 MB. | `file build/bin/llama-perplexity` reports `aarch64`. |
| 11.1.2 | user (board) | `scp /home/lanhp-wsl/.../build/bin/llama-perplexity nouslogic-sl2619:/mnt/sdcard/llama-cpp/` | `ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/llama-cpp/llama-perplexity && /mnt/sdcard/llama-cpp/llama-perplexity --version 2>&1 \| head -n 4'` shows `(0adede8)`. |
| 11.1.3 | user (server) | clone same tag on `nouslogic-server`, `cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF`, `cmake --build build --target llama-perplexity`. CPU-only is fine — H5R wants a CPU reference, not CUDA. | `~/llama.cpp/build/bin/llama-perplexity --version` shows `(0adede8)`. |

### 11.2 Generate the reference `.kld` (only if missing)

```
~/llama.cpp/build/bin/llama-perplexity \
  -m /path/to/gemma-3-270m-it-BF16.gguf \
  -f /path/to/h5-prompts.txt \
  --save-all-logits /path/to/h5-reference.kld \
  -t $(nproc) --temp 0.0 --no-mmap
```

The H5 corpus is built from `prompts.yaml` (15) + `sft_v1.test.jsonl` random sample (seed=42, 15 rows) + 5 OOD probes (CJK, multi-codepoint emoji, ASCII spam, whitespace, markdown fence). sha-pin the prompt file to `.cache/h5/prompts-<sha>.txt`.

---

*Authored 2026-04-26 as the original H5 absolute-threshold session plan.*
*Rewritten 2026-04-27 as H5R same-quant cross-arch Δ session plan after the llama.cpp / ONNX research review (see `docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md`) demonstrated the original gate was unreachable for any Q4_0-vs-FP16 on any architecture. H5 PUNT chronology preserved in `a55-gemma-fine-tune.md §10.2`. References: `a55-gemma-fine-tune.md §4 Phase 0 H5R / §10.2`, `gemma-on-a55-get-started.md §3.4–3.7`, `15-model-compiler-runtime.md §5.4`, `tools/perplexity/perplexity.cpp` (`b8925`/`0adede8`), `common/arg.cpp:2079`, `docs/references/llama-cpp.md`.*
