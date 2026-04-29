# Model Compiler & Runtime — llama.cpp / GGUF Working Knowledge

> Trimmed normative reference for the **llama.cpp + GGUF** stack used by this
> repo end-to-end: HF safetensors → BF16 GGUF → Q4_0 GGUF → x86 host validation
> → SL2619 A55 deployment. NPU/Torq and onnxruntime detail used in the larger
> SynapticSL2619 workspace lives there; this file is the llama-only excerpt.

---

## 0. Mental model

The full lifecycle for any GGUF model in this repo:

```
1. Export      — pull or convert HF checkpoint to BF16 GGUF
2. Quantize    — llama-quantize BF16 → Q4_0 (or Q8_0 for higher fidelity)
3. Host smoke  — load-test on host BEFORE scp to board;
                 catch format/shape/ABI mismatches early
4. Logits gate — KL-divergence vs BF16 reference, host AND target architecture
5. Deploy      — scp to /mnt/sdcard/models/<model>/ (user-performed)
6. Verify      — smoke-test on target (board SSH read-only for agent)
7. Bench       — capture perf + quality numbers in docs/bench/<date>_<model>-*.md
8. Freeze      — update per-model README.md with the outcome
```

Phase-specific recipes instantiate steps 1–8; this file captures the
cross-cutting rules.

## 1. Runtime inventory

| Runtime | Type | Format | On-target artifact dir | Host tooling | Doc pointer |
|---|---|---|---|---|---|
| `llama.cpp` (current pin: tag `b8925`, commit `0adede8`) | Cross-compiled C++ binary (aarch64 for SL2619; x86_64 native for host) | `.gguf` (BF16 / Q8_0 / Q4_0) | `/mnt/sdcard/llama-cpp/` (binaries), `/mnt/sdcard/models/<model>/*.gguf` | `convert_hf_to_gguf.py`, `llama-quantize` | §3 + [`docs/references/llama-cpp.md`](../references/llama-cpp.md) |

> Bump §1 and §3.6 together when the llama.cpp tag advances.

---

## 3. llama.cpp — A55 CPU inference (GGUF)

**Depth pointer**: [`docs/references/llama-cpp.md`](../references/llama-cpp.md) — full upstream source map and command recipes. This section is the conventions only.

### 3.1 Stack overview

- **Inference binary**: `llama-completion` (headless one-shot) and `llama-cli` (interactive) — cross-compiled against the Yocto SDK, deployed to `/mnt/sdcard/llama-cpp/`.
- **Model format**: Q4_0 GGUF — 4-bit quantized weights, mmap-loaded from SD card. Hot path: NEON DOTPROD kernels for GEMV/GEMM.
- **Conversion tooling** (host or server): `convert_hf_to_gguf.py` (HF safetensors → BF16 GGUF) and `llama-quantize` (BF16 → Q4_0).

Current pin: `b8925` / commit `0adede8`. Update §1 and §3.6 if the tag advances.

### 3.2 Cross-compile against Yocto SDK

**Why cross-compile is mandatory**: prebuilt GitHub releases (`llama-b8925-bin-ubuntu-arm64.tar.gz`) link against `CXXABI_1.3.15` (GCC 14+); the board's stock Yocto image ships `libstdc++.so.6.0.32` exporting only `CXXABI_1.3.14`. Symptom on prebuilt: `./llama-cli: /usr/lib/libstdc++.so.6: version 'CXXABI_1.3.15' not found`.

**The working configure**:

```bash
source /opt/poky/5.0.9/environment-setup-cortexa55-poky-linux
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \      # MUST be ON — see §3.6 pitfall
  -DBUILD_SHARED_LIBS=OFF        # single binary; no SO deps beyond system libs
cmake --build build --target llama-cli llama-bench llama-completion -j$(nproc)
```

Configure correctly detects: `aarch64`, `DOTPROD` ✓, `FP16_VECTOR_ARITHMETIC` ✓, `FMA` ✓, **no SVE / no SME / no MATMUL_INT8**. This is correct — A55 is ARMv8.2-A with DOTPROD but not I8MM.

Strip binaries before deploy (`aarch64-poky-linux-strip llama-cli llama-completion llama-bench`). Expected stripped sizes: `llama-cli` 8.3 MB, `llama-completion` 6.6 MB, `llama-bench` 4.8 MB.

Full runbook: [`docs/deployment/sl2619-board.md`](../deployment/sl2619-board.md) §3.

### 3.3 REPACK: CPU architecture gates (A55 path)

REPACK = on-load weight interleaving into a CPU-arch-specific layout to feed a vectorized matmul kernel without per-call gather. The selection is made at runtime in `ggml/src/ggml-cpu/repack.cpp:ggml_repack_get_optimal_repack_type`:

| CPU feature gate | Q4_0 layout | Architecture |
|---|---|---|
| AVX2 _or_ (SVE + I8MM) | `q4_0_8x8_q8_0` | x86_64 server/host |
| NEON + I8MM | `q4_0_4x8_q8_0` | Cortex-A76+, Neoverse |
| **NEON + DOTPROD** (no I8MM) | **`q4_0_4x4_q8_0`** | **A55 — our path** |

**Consequence**: A55 and x86_64 select different weight layouts → different dot-product accumulation order → ≤ ~10% logit-rank divergence even with arithmetically-identical weights. **This is not a bug.** It is expected ISA-level FP behaviour. `--no-repack` (`-nr`) forces scalar Q4_0 at the cost of much lower throughput.

### 3.4 Logits-equivalence discipline

Every new GGUF (base model or fine-tune) must pass a logits-equivalence gate **before** quality evaluation. This prevents blaming prompt-engineering for what is actually arithmetic corruption.

**Gate — same-quant cross-arch Δ** (calibrated 2026-04-27, supersedes obsolete absolute `same_top_p ≥ 99.99%` gate):

| Threshold | Value | Rationale |
|---|---|---|
| `Δ_same_top_p` (target Q4_0 vs host Q4_0, same `.kld` reference) | **≤ 1.0 pp** | Upstream's BF16-vs-FP16 Δ is ~0.25 pp; 1.0 pp allows 4× margin for quant + ISA differences. |
| `max_delta_p_target / max_delta_p_host` | **≤ 3.0×** | Guards against tail-logit explosion. |

Do NOT use `same_top_p ≥ 99.99%` as an absolute threshold — Q4_0-vs-FP16 for any model on any architecture achieves only ~91–98% (see `llama.cpp/tools/perplexity/README.md`). The Δ test subtracts out universal Q4_0 quantization noise.

**Three-step gate** (post-fine-tune logits-equivalence):

1. Host BF16 reference → generate `.kld` baseline.
2. Host Q4_0 vs same `.kld` → establish noise floor (`same_top_p_host`).
3. Target Q4_0 vs same `.kld` → check `|same_top_p_target − same_top_p_host| ≤ 1.0 pp`.

This isolates: (i) Q4_0 quantization noise (steps 1→2), (ii) fine-tune delta (step 2 vs fine-tuned step 2), and (iii) target-specific arithmetic divergence (steps 2→3). For the H5R result on the base Gemma 3 270M-IT model: `Δ = 0.393 pp` — well within gate. See `docs/bench/2026-04-27_h5r-cross-arch-delta.md`.

**Board OOM constraint**: `n_ctx ≥ 1024` OOM-kills `llama-perplexity` on the SL2619 board (per-chunk buffer = `n_ctx × vocab × f32`; at `n_ctx=2048` → 2.15 GiB; board has 1.87 GiB). **Cap at `n_ctx=256` for on-board perplexity runs.** Keep `.kld` files on `/mnt/sdcard`, not tmpfs.

### 3.5 Deployment conventions

| Artifact | Path | Notes |
|---|---|---|
| Binaries | `/mnt/sdcard/llama-cpp/llama-completion`, `llama-cli`, `llama-bench` | cross-compiled, stripped |
| Base GGUF | `/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf` | 231 MB, sha256 `e479ea29…` |
| Fine-tuned GGUF | `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` | v1 SFT, Q4_0 |

**Invocation conventions**:

- Always `-t 2` (board exposes 2 cores; `-t 4` = 53× decode regression — measured).
- For headless one-shot: `llama-completion -no-cnv`.
- For fine-tuned model: MUST use `--jinja` (routes special tokens to correct IDs; without it, text-wrapping as plain bytes → hallucinated tail generation). Also use `--no-display-prompt` to get clean stdout.
- For user-turn content: render body locally via `prompt_composer.compose_user_text()`, pipe over SSH stdin with `printf '%s' "$BODY" | ssh ...`.

**Performance baseline** (2026-04-28, fine-tuned Q4_0, `--jinja`, SL2619):

| Metric | Value |
|---|---|
| Aggregate decode | 17.29 tok/s |
| Cold load (mmap + REPACK) | 3273 ms |
| Prompt-eval rate (~930 tok) | ~62 tok/s |
| Memory (process RSS) | ~1071 MiB |

### 3.6 Known pitfalls

| Problem | Root cause | Fix |
|---|---|---|
| `CXXABI_1.3.15 not found` at first run | Prebuilt binary is GCC 14; board has GCC 13.3 | Cross-compile from source against Yocto SDK (§3.2) |
| `No rule to make target 'llama-cli'` | `LLAMA_BUILD_SERVER=OFF` removes `cli` subdirectory via CMakeLists coupling | Set `-DLLAMA_BUILD_SERVER=ON` |
| Decode drops from 5.87 to 0.11 tok/s | `-t 4` over-subscribes 2 available cores | Always `-t 2` |
| `--no-conversation is not supported` from `llama-cli` | `llama-cli` is interactive-only in `b8925` | Use `llama-completion` for headless runs |
| `-sysf sysprompt.txt --jinja` silently drops content | Gemma 3 chat template has no `system` role; content mis-routed | Compose user-turn body manually via `prompt_composer.compose_user_text()` |
| Fine-tuned model generates hallucinated tail HTML | Text-wrapped `<start_of_turn>` tokenized as plain bytes, not special token IDs | Use `--jinja` so the GGUF's embedded chat template handles wrapping |
| `requirements-convert_hf_to_gguf.txt` downgrades torch | File pins `torch~=2.6.0` — incompatible with cu128 torch 2.11 | Install `gguf` directly, skip the requirements file |

---

## 6. Model conversion and export conventions

### 6.1 HF safetensors → GGUF (llama.cpp path)

```
HF checkpoint (safetensors, BF16 or QAT) ─┐
  ↓  convert_hf_to_gguf.py --outtype bf16  │  host or server
  ↓  → model.bf16.gguf                     │
  ↓  llama-quantize model.bf16.gguf        │  host or server
     model.q4_0.gguf Q4_0                  │
  ↓  → model.q4_0.gguf                    ─┘
  ↓  Host smoke: llama-perplexity (logits-equivalence gate §3.4)
  ↓  scp to /mnt/sdcard/models/<model>/
  ↓  On-board verify: llama-completion smoke prompt
```

- **Do NOT pip-install `requirements-convert_hf_to_gguf.txt`** — it pins `torch~=2.6.0` which downgrades torch on cu128 stacks. Install `gguf` directly.
- **Fine-tune path (QLoRA)**: train on `google/gemma-3-270m-it` → merge adapter → `convert_hf_to_gguf.py` → `llama-quantize Q4_0` → logits gate → Q4 quality bench.
- **Don't fine-tune the `-qat-q4_0-unquantized` Gemma 3 checkpoint** — Google's documented workflow uses plain `gemma-3-270m-it`; no recipe preserves QAT robustness through domain SFT.

### 6.2 General vendor HF repo rules

1. **Always load-test on host** before scp — catches QDQ/shape mismatches before the board round-trip.
2. **Use the same runtime version** on host validation as on the board (llama.cpp tag pinned in §1).
3. **Pin SHAs before deploy** — `sha256sum` the artifact locally, record in the bench freeze.

---

## 7. Reproducibility and logging expectations

| Artifact / decision | What to pin | Where |
|---|---|---|
| llama.cpp binary | git tag (`b8925`) + commit SHA (`0adede8`) | [`docs/references/llama-cpp.md`](../references/llama-cpp.md), `docs/deployment/sl2619-board.md §1` |
| GGUF artifact | `sha256sum` of `.gguf` | `docs/deployment/sl2619-board.md §1` |
| Bench sweep results | JSONL output + Markdown summary | `docs/bench/<date>_<model>-*.md` (frozen; never re-opened) |
| As-executed recipes | Step-by-step runbooks | `docs/deployment/<topic>-deploy.md` |
| Phase-specific recipes | Exact flags, gate results | `docs/plans/*-plan.md` or `docs/plans/a55-gemma-fine-tune.md §10` |

**Bench summary naming convention**: `docs/bench/<YYYY-MM-DD>_<model-slug>[-<descriptor>].md`. Examples: `2026-04-24_gemma3-summary.md`, `2026-04-27_h5r-cross-arch-delta.md`, `2026-04-28_gemma3-finetuned-final.md`. Each file is a **frozen snapshot** — use the date prefix to track the sweep, and create a new file for each re-run.

**What is NOT required** for reproducibility: re-running the full bootstrap every time. The llama.cpp binaries and GGUFs persist on the SD card. Only when the SD card or board image changes does a full re-bootstrap apply.

---

## 8. Debugging and failure triage — llama.cpp

| Symptom | Cause | Fix |
|---|---|---|
| `CXXABI_1.3.15 not found` | Prebuilt binary; board has GCC 13.3 | Cross-compile from source against Yocto SDK (§3.2) |
| `No rule to make target 'llama-cli'` | `LLAMA_BUILD_SERVER=OFF` | Set `-DLLAMA_BUILD_SERVER=ON` |
| Decode 0.11 tok/s | `-t 4` with 2 cores exposed | Always `-t 2` |
| `--no-conversation is not supported` | `llama-cli` is interactive-only | Use `llama-completion -no-cnv` |
| `-sysf` content not seen by model | Gemma 3 has no `system` role in chat template | Compose user-turn body manually, skip `-sysf` |
| Fine-tuned model outputs `<h4>You can also try</h4>...` garbage | Text-wrapping tokenizes control markers as plain bytes | Use `--jinja` for fine-tuned GGUFs |
| `llama-perplexity` silently SIGKILL on board | `n_ctx ≥ 1024` → OOM; per-chunk buffer too large | Cap at `n_ctx=256` for on-board runs |
| `kl_divergence: failed to open FNAME` + no `.kld` written, exit 0 | `--save-all-logits` and `--kl-divergence-base` share the same FNAME slot; passing `--kl-divergence` alongside `--save-all-logits` flips the tool into LOAD mode and it tries to open the file you intended to write | SAVE: `--save-all-logits ref.kld` only (no `--kl-divergence`). LOAD: `--kl-divergence --kl-divergence-base ref.kld` only (no `--save-all-logits`) |

---

## 9. Checklist — adding a new GGUF model or upgrading the stack

**For any new model going to A55 CPU via llama.cpp (GGUF):**

- [ ] Cross-compiled against Yocto SDK (not prebuilt binary).
- [ ] Logits-equivalence gate run (§3.4) before quality evaluation.
- [ ] `-t 2` confirmed (not 4; check `cat /sys/devices/system/cpu/online`).
- [ ] `--jinja` flag in invocation if model is fine-tuned.
- [ ] Binaries deployed to `/mnt/sdcard/llama-cpp/`, GGUF to `/mnt/sdcard/models/<model>/`.

**Cross-cutting:**

- [ ] `sha256sum` of all model artifacts recorded before scp.
- [ ] On-target smoke test confirms runtime loads cleanly.
- [ ] Bench summary frozen to `docs/bench/<date>_<model>-*.md`.
- [ ] Per-model `models/<model>/README.md` updated with outcome.
- [ ] This file updated if a new runtime or pin is introduced.

---

## 10. When to update this document

Update this file when:

- A new model **runtime** is introduced to the workspace (e.g. ExecuTorch, a second llama.cpp variant, a new quantization format).
- The **llama.cpp pin** advances and changes any flags, REPACK kernel selection, or perf numbers.
- A new **model storage path or naming convention** is adopted.
- A new cross-cutting **pitfall or workaround** is discovered.
- A **gate threshold** (e.g. `Δ ≤ 1.0 pp`) is recalibrated.

Do NOT update this file for:

- Per-model analysis (goes in `models/<model>/README.md`).
- As-executed step-by-step runbooks (go in `docs/deployment/`).
- Phase-specific recipe details (those go in plan files under `docs/plans/`).

Canonical ownership rule: each fact lives in ONE file; others use pointers. See [`13-documentation-update-protocol.md`](13-documentation-update-protocol.md) §10.

---

## 11. What this file does NOT cover

- SLM prompt style and template rules — see [`slm-system-prompt.md`](slm-system-prompt.md).
- Fine-tune training loop, hyperparameters, dataset construction — see [`docs/plans/a55-gemma-fine-tune.md`](../plans/a55-gemma-fine-tune.md) and [`docs/guides/finetune-best-practices.md`](../guides/finetune-best-practices.md).
- Upstream llama.cpp source map — see [`docs/references/llama-cpp.md`](../references/llama-cpp.md).
- NPU (Torq) and onnxruntime detail — covered in the SynapticSL2619 workspace, not in this fine-tune repo.
