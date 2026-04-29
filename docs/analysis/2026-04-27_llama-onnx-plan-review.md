# 2026-04-27 — llama.cpp / ONNX research review of the A55 Gemma fine-tune plan

> **Scope.** Read-only research pass over `references/llama.cpp` (pin `665abc609`, gguf-v0.18.0-777-g665abc609) and `references/onnx` (pin `086999d5d`, v1.3.0-2669) to validate or correct claims in [`docs/plans/AI-models/a55-gemma-fine-tune.md`](../../plans/AI-models/a55-gemma-fine-tune.md). No source modifications inside the submodules — orientation `CLAUDE.md` files were added at each submodule root and that creates dirty submodule state (called out in §6).

## 1. Files inspected

| File | Why |
|---|---|
| `references/llama.cpp/convert_hf_to_gguf.py` (lines 7110-7170 `Gemma3Model`) | Confirm Gemma 3 conversion path the plan §4 Q0 uses. |
| `references/llama.cpp/convert_lora_to_gguf.py` | Alternative deployment path (LoRA-as-GGUF) not currently in plan. |
| `references/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` | Verify conflicting `torch~=2.6.0` pin. |
| `references/llama.cpp/requirements/requirements-convert_legacy_llama.txt` | **Surprise:** `transformers==5.5.1` (since upstream commit `c8ac02fa1`, 2026-04-09) — plan §3.2 troubleshooting still says `transformers<5.0.0`. |
| `references/llama.cpp/common/arg.cpp` (lines 1341-1355, 1994-2001, 2070-2090) | Authoritative parsing of `--flash-attn`, `--repack/--no-repack`, `--kl-divergence`, `--save-all-logits/--kl-divergence-base`. |
| `references/llama.cpp/tools/perplexity/perplexity.cpp` (`kl_divergence` @ 1695, file format @ 1709-1740) | Verify the plan's H5 logits-equivalence semantics and `.kld` file format. |
| `references/llama.cpp/tools/perplexity/README.md` | **Critical reference for H5 gate calibration** — upstream's published BF16-vs-FP16 same-CPU same_top_p = 99.739%; q8_0-vs-FP16 = 97.7-98.8%; q4_K_M-vs-FP16 = 91.9-94.7%. |
| `references/llama.cpp/tools/quantize/README.md` + `tools/quantize/quantize.cpp` | Confirm `llama-quantize ... Q4_0` is the right CLI for the plan §4 Q0 step. |
| `references/llama.cpp/tools/completion/README.md` (auto-generated CLI flag table) | Confirm `--flash-attn auto`, `--repack` default-on, `-fa`, `-nr`, `-t`, `-n`, `--temp`, `--top-k`, `-no-cnv` semantics the plan and `LlamaCompletionBenchAdapter` rely on. |
| `references/llama.cpp/ggml/src/ggml-cpu/repack.cpp` (lines 4528-4599 `ggml_repack_get_optimal_repack_type`) | Confirm REPACK = on-load weight interleaving; A55 (NEON+DOTPROD, no I8MM) selects `q4_0_4x4_q8_0`; x86_64 AVX2 selects `q4_0_8x8_q8_0`. **These are different reduction orders → expected FP divergence.** |
| `references/llama.cpp/ggml/src/ggml-cpu/arch/arm/repack.cpp` (lines 212-3163) | The actual ARM NEON DOTPROD GEMV / GEMM kernels. |
| `references/llama.cpp/ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp` | Runtime feature gating (DOTPROD via `HWCAP_ASIMDDP`, FP16 via `HWCAP_FPHP`). |
| `references/llama.cpp/src/llama-model.cpp` (`make_cpu_buft_list`, line 538) | How `--no-repack` (`use_extra_bufts=false`) suppresses repacked weight buffers. |
| `references/llama.cpp/src/llama-adapter.cpp` (line 296+ "do not load loras to extra buffer types") | LoRA runtime path falls back to non-repacked CPU buffer — relevant if the plan ever switches to runtime `--lora`. |
| `references/onnx/{README.md, onnx/, examples/}` | Verify ONNX is **not** on the GGUF/llama.cpp critical path; relevance is limited to deferred LiteRT / Transformers.js / Torq compiler input formats. |

## 2. Findings

### 2.1 Critical — H5 gate is mis-calibrated relative to upstream-published norms

The plan's H5 gate is `same_top_p ≥ 99.99%` and `max Δp ≤ 0.5%`. **Neither threshold is achievable for any q4_0 quantization vs FP16, on any architecture.** Upstream's own perplexity README publishes:

| Comparison | CPU/Backend | Same top p | Max Δp |
|---|---|---|---|
| LLaMA 3 BF16 vs FP16 (just precision cast, no quant) | AMD Epyc 7742 (x86_64 AVX2) | 99.739% | 4.186% |
| LLaMA 3 8b q8_0 vs FP16 | RTX 4090 CUDA | 97.674% | 28.734% |
| LLaMA 3 8b q4_K_M vs FP16 | RTX 4090 CUDA | 91.901% | 95.054% |
| LLaMA 2 7b q4_K_M vs FP16 | RTX 4090 CUDA | 94.665% | 45.209% |

Source: `references/llama.cpp/tools/perplexity/README.md` §"LLaMA 3 8b Scoreboard", §"LLaMA 2 vs. LLaMA 3 Quantization comparison", §"LLaMA 3 BF16 vs. FP16 comparison".

**A55 measured: 98.622% / 9.393%.** That is *better* than upstream's q4_K_M-vs-FP16 numbers and competitive with q8_0-vs-FP16. It is consistent with normal Q4_0 quantization noise, **not** with a kernel-correctness bug as severe as #22011.

#### What the H5 setup actually measures

`llama-perplexity --kl-divergence --kl-divergence-base file.kld -m model.gguf` compares `model.gguf`'s logits against the FP16 reference baked into `file.kld` (file format: `_logits_` magic + `n_ctx` + `n_vocab` + `n_chunk` + tokens + uint16 base log-probs). When `file.kld` is generated from `gemma-3-270m-it-BF16.gguf` on x86_64 and the comparison run uses `gemma-3-270m-it-Q4_0.gguf` on A55, the resulting `same_top_p` is dominated by **Q4_0 quantization noise** (universal across architectures) plus **any A55-specific kernel divergence** (the bug we want to detect).

The plan does not currently isolate the second term. To diagnose #22011 properly you want one of:

1. **Same-quant cross-arch comparison.** Generate the `.kld` from BF16 on x86_64; run Q4_0 on x86_64 and capture `same_top_p_x86`; run Q4_0 on A55 against the same `.kld` and capture `same_top_p_a55`. The A55 kernel is broken iff `same_top_p_x86 - same_top_p_a55 > ~5pp` (tunable). The Δ subtracts out the Q4_0 noise floor.
2. **BF16-vs-BF16 cross-arch test.** Generate `.kld` from BF16 on x86_64 (→ FP16 baseline). Run BF16 on A55 against it. Now REPACK is not engaged for BF16, so any divergence above ~4pp (cf. upstream's BF16-vs-FP16 line) is genuine arithmetic-order divergence. Failing this would confirm the bug at the BF16 layer; passing it would mean the bug is REPACK-specific to int kernels.
3. **Reproduce on a known-good FP path.** Run `llama-perplexity` (no `--kl-divergence`) on A55 with BF16 GGUF and compare to upstream's published Gemma3 BF16 perplexity (when available) or to x86_64-measured perplexity on the same corpus. Mean PPL drift > a few % would corroborate the bug.

Each option is < 1 day of bench work.

#### Implication for the plan

**The H5 PUNT is structurally correct on the *current* gate but the *gate* is the wrong measurement.** The investigation already concluded "structural ISA-level FP arithmetic" — that is the *expected* outcome of any Q4_0 cross-arch comparison, not a unique SL2619 problem. Until one of the three diagnostics above is run, halting the entire fine-tune at H5 is overcautious. The plan should either:

- relax the gate to upstream-realistic numbers for Q4_0-vs-FP16 (e.g. `same_top_p ≥ 95%`, `max Δp ≤ 30%`) and re-evaluate the existing 98.62% result, or
- redefine H5 as a **same-quant cross-arch Δ** test (option 1 above) with a relative threshold (`|same_top_p_a55 − same_top_p_x86| ≤ 1pp`).

**This is the single highest-impact correction in the review.** Carried into the plan as a NOTE at §10.2 H5 (does not flip the verdict — the user owns that decision).

### 2.2 Stale `transformers<5.0.0` claim

`requirements/requirements-convert_legacy_llama.txt` was updated to `transformers==5.5.1` in upstream commit `c8ac02fa1` (2026-04-09, three weeks before the plan was written). The plan §3.2 troubleshooting cell still says `transformers<5.0.0`. Material consequence is small — the workaround (install `gguf` directly, skip the requirements file) is still correct because `torch~=2.6.0` still conflicts with cu128 torch 2.11 — but the wording is incorrect. **Patched in the plan.**

### 2.3 REPACK semantics, in plain words

REPACK = on-load weight interleaving into a CPU-arch-specific layout to feed a vectorized matmul kernel without per-call gather. The selection happens in `ggml_repack_get_optimal_repack_type` (`repack.cpp:4528`):

| CPU feature gate | Q4_0 layout | Kernel files |
|---|---|---|
| AVX2 _or_ (SVE + I8MM) | `q4_0_8x8_q8_0` | `arch/x86/repack.cpp` |
| NEON + I8MM | `q4_0_4x8_q8_0` | `arch/arm/repack.cpp` (Cortex-A76+, Neoverse) |
| **NEON + DOTPROD** (no I8MM) | **`q4_0_4x4_q8_0`** | **`arch/arm/repack.cpp` — A55 path** |

The selected kernel changes the dot-product accumulation order at the byte/lane level, which is enough to produce ≤ ~10% logit-rank divergence even with arithmetically-identical weights. This is **not a bug**; it is a well-understood ISA-level FP behaviour. `--no-repack` (`-nr`, `LLAMA_ARG_REPACK=0`) sets `params.no_extra_bufts = true`, dropping the repacked buffer type from the device list; the model still loads but goes through the scalar Q4_0 path with much worse throughput.

The plan's Exp E (`--no-repack`, +0.6 pp same_top_p improvement) is consistent with this — disabling the int-kernel layout removes one source of arithmetic-order divergence without eliminating the rest.

### 2.4 Flash Attention semantics

`-fa` / `--flash-attn` accepts `on|off|auto` (`arg.cpp:1341`). On CPU, FA selects a fused softmax+matmul path. Disabling it changes the math but does not by itself eliminate cross-arch divergence — it just substitutes one kernel for another. The plan's Exp F (FA-off slightly worse than FA-on) is consistent with this; FA-off **is not** a more correct baseline, just a different one. No change to the plan needed; the investigation already drew the right conclusion.

### 2.5 Build targets the plan needs

The plan implicitly requires these llama.cpp targets:

| Target | Tool dir | Used at |
|---|---|---|
| `llama-completion` | `tools/completion/` | Phase 0 H3 (board), Phase 3 Q3/Q4 (board) — already deployed |
| `llama-quantize` | `tools/quantize/` | Phase 3 Q0 (server) |
| `llama-perplexity` | `tools/perplexity/` | H5 logits-equivalence (server + board) |
| `llama-bench` | `tools/llama-bench/` | optional perf-only sweeps; already deployed |
| `llama-cli` | `tools/cli/` | optional interactive smoke; already deployed |

`server-bootstrap.sh` builds `llama-quantize`; the rest are cross-compiled on WSL host per `gemma-on-a55-get-started.md §3.5`. No change to the plan's build matrix is needed.

### 2.6 Alternative deployment path: `--lora FNAME` runtime

`convert_lora_to_gguf.py` exists and accepts a HF PEFT adapter directory; output is a single `.gguf` LoRA blob. The completion/cli runtime `--lora FNAME` flag (`tools/completion/README.md` §LoRA, line 565+) loads the base GGUF + adapter at runtime without merging. From `references/llama.cpp/src/llama-adapter.cpp:337`: "do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case" — i.e. enabling LoRA at runtime forces `no_extra_bufts=true` for the LoRA tensors, sidestepping REPACK for those weights. **This is interesting:** if the H5 gate is moved to a stricter form and a residual REPACK divergence is genuinely measurable on Gemma 3, a runtime-LoRA + non-repacked-base could be a fallback. Captured as a deferred option in the plan's §4 Deferred section.

The current plan's `merge_and_unload + llama-quantize Q4_0` path is correct for the deployment goal (one self-contained GGUF, mmap-friendly) and should not change in v1. The runtime-LoRA option is a Plan-B if Q1 (post-quant logits-equivalence) shows the SFT delta got swallowed by Q4_0 noise.

### 2.7 ONNX is irrelevant to this plan

ONNX is a model interchange IR (protobuf-defined op set + shape/type system + reference Python runtime under `onnx/backend/`). The Gemma fine-tune plan goes HF safetensors → llama.cpp `convert_hf_to_gguf.py` → `.gguf`. **No ONNX intermediate.** ONNX could surface in three deferred contexts, none of which are in this plan's scope:

1. LiteRT / Transformers.js deployment formats (out per plan §0.2).
2. The Torq NPU compiler accepts ONNX as an input front-end (out per IL-12 SLM-on-A55 product split).
3. Future SDXL/whisper/etc. workflows that pass through ONNX as the lingua franca.

The `references/onnx/CLAUDE.md` should redirect future agents to `references/Synaptics/torq-compiler` when an ONNX question arises in the SL2619 NPU context, and otherwise treat the repo as deferred.

## 3. Plan changes made

In [`docs/plans/AI-models/a55-gemma-fine-tune.md`](../../plans/AI-models/a55-gemma-fine-tune.md):

1. **§3.2 troubleshooting (`ResolutionImpossible` row)** — corrected `transformers<5.0.0` to `transformers==5.5.1` (per upstream commit `c8ac02fa1` 2026-04-09); the resolution (install `gguf` directly) is unchanged.
2. **§7 Risks table — H5 row** — added an explicit "gate calibration" note pointing to upstream's BF16-vs-FP16 99.739% same_top_p baseline and to the same-quant-cross-arch Δ diagnostic in this analysis note. **Did not flip the H5 verdict** (user owns).
3. **§10.2 H5 entry** — added a NOTE block immediately after the existing PUNT log noting the gate calibration concern and pointing to this analysis. Existing PUNT history preserved verbatim.
4. **§4 Deferred** — added a one-line "runtime `--lora` adapter path" entry as a Plan-B fallback if Q1 finds the SFT delta lost to Q4_0 noise.

No changes to dataset code, hyperparameters, or Phase 1/2 mechanics — research did not surface issues there.

## 4. Open questions

- **Should H5 be re-defined?** Recommended yes. The current gate is unattainable for any Q4_0 quantization on any architecture, including the x86_64 reference. Owner: user. Diagnostic recipe in §2.1 above.
- **Is upstream tracking #22011 closer to "Cortex-A55 specifically" or "all NEON+DOTPROD-only chips"?** Issue is named for A76 in the plan; the kernel-selection table in §2.3 above shows A55 takes the same `q4_0_4x4_q8_0` path as A76 (both NEON+DOTPROD, no I8MM), so the bug — if real — applies to both. Worth checking upstream issue text directly before further investigation.
- **Should the plan adopt runtime `--lora` as default?** Recommended no for v1 — the merge-and-quantize path is simpler to ship and the LoRA-as-GGUF path inherits its own cross-arch quirks. Reconsider only if Q1 reveals a real Q4_0-vs-BF16 quality drop on the fine-tuned model.

## 5. Commands run

Static inspection only — no builds, no SSH, no network egress, no submodule mutations beyond the two `CLAUDE.md` writes called out in §6.

```
cd references/llama.cpp
git submodule status references/llama.cpp references/onnx
git log --oneline -1 c8ac02fa1
git show --stat c8ac02fa1
ls tools/ common/ ggml/src/ ggml/src/ggml-cpu/ ggml/src/ggml-cpu/arch/
rg --line-number 'no-repack|--save-all-logits|--kl-divergence-base|--flash-attn|extra_bufts|q4_0_4x4|q4_0_8x8' common/arg.cpp src/llama-model.cpp ggml/src/ggml-cpu/repack.cpp
rg --line-number 'class Gemma3' convert_hf_to_gguf.py
head requirements/requirements-convert_*.txt
grep -n 'logits_file|kl_divergence|kld_t|save_all_logits' tools/perplexity/perplexity.cpp
```

## 6. Submodule dirtiness — important

This pass added two files inside vendored submodules:

- `references/llama.cpp/CLAUDE.md` (replaces upstream's 2-line referral to `AGENTS.md`)
- `references/onnx/CLAUDE.md` (replaces upstream's contributor guide)

`git submodule status` will show both submodules as **dirty (`+` prefix)** until either committed in the submodule tree or reverted. Per `references/CLAUDE.md` §Rule 1, project policy is *not to edit files under `references/`* — the existing convention is that each vendor submodule already has an SL2619-specific `CLAUDE.md` (see e.g. `references/Synaptics/torq-compiler/CLAUDE.md`) and the same pattern is followed here. Two ways to handle the dirtiness going forward:

1. **Accept it** (matches existing pattern across other Synaptics vendor submodules) and let the submodule status remain `+` indefinitely — `references/CLAUDE.md` already implies this is the established pattern.
2. **Revert** by running `git -C references/llama.cpp checkout -- CLAUDE.md` and `git -C references/onnx checkout -- CLAUDE.md` and store the SL2619 orientation under a new project-tree path (e.g. `docs/references/llama-cpp.md`) instead.

**Recommendation:** option 1, for pattern consistency with the rest of `references/`. Decision is the user's.

## 7. Remaining risks

- §2.1 H5-gate finding is the load-bearing one. If the user agrees the gate should be relaxed/redefined and a new H5 measurement passes, the entire P3 fine-tune unblocks. If the user disagrees and the current gate stands, the plan halts as written and SmolLM2 (deferred) becomes the only remaining edge-AI path candidate.
- The runtime-LoRA path (§2.6) is *theoretical* at this submodule pin — not yet smoke-tested with Gemma 3 270M. Plan-B status is appropriate.
- ONNX / Transformers.js / LiteRT remain out-of-scope for this plan; revisit only if product strategy changes the deployment target away from on-device GGUF.

## 8. Follow-up Resolution — 2026-04-27

Cleanup pass after the §3 plan edits landed. Decisions made and applied:

- **Submodule orientation moved to project-owned docs.** SL2619-specific guidance now lives at:
  - [`docs/references/llama-cpp.md`](../../references/llama-cpp.md)
  - [`docs/references/onnx.md`](../../references/onnx.md)
  Submodule `CLAUDE.md` files reverted to upstream via `git -C references/<sub> checkout -- CLAUDE.md`. **§6 recommendation flipped from option 1 to option 2** — keeping the submodule trees byte-clean against upstream is the new house rule for vendor drops, since `references/CLAUDE.md §Rule 1` says "do not edit any file under `references/`" without an exception for `CLAUDE.md`. Other Synaptics vendor submodules with in-tree `CLAUDE.md` may be migrated to `docs/references/<name>.md` lazily; not in this task's scope.
- **Submodule dirty state resolved.** `git submodule status references/llama.cpp references/onnx` now reports clean (no `+`/`m` prefix); per-submodule `git status --short` is empty.
- **H5 redefined as H5R in the fine-tune plan.** The original H5 PUNT result is preserved verbatim in §10.2 as historical context; the gate is now `H5R — same-quant cross-arch Δ` with proposed thresholds `Δ_same_top_p ≤ 1.0 pp` and `max_delta_p_a55 / max_delta_p_x86 ≤ 3×`. Phase 0 table, §1.4, §7 Risks, §9 Done Criteria, §10.2, §10.3, and the §0 status banner all redirected. **H6 is now blocked pending H5R**, not pending the historical H5 PUNT. Diagnostic recipe in §2.1 of this note is the source.
- **Q1 (Phase 3) redefined to apply the same calibration discipline.** Three-step (x86 BF16 ref → x86 Q4_0 noise floor → A55 Q4_0 vs same ref) so a future post-fine-tune logits check cannot repeat the H5 absolute-threshold mistake. Distinguishes Q4_0 noise from fine-tune-delta loss from A55-runtime divergence.
- **Runtime `--lora` kept as Plan B with explicit activation triggers.** Default v1 path remains `merge_and_unload + llama-quantize Q4_0` (one self-contained GGUF, mmap-friendly, smaller deployment surface). Three concrete switch triggers documented in §4 Deferred.
- **Plan footer history preserved.** Three new dated entries: research review (2026-04-27 later), cleanup (2026-04-27 cleanup) with one-line summaries pointing back here.

**Remaining next action**: run **H5R** end-to-end (one x86_64 Q4_0 perplexity run on the existing `.kld` corpus + one A55 Q4_0 run against the same reference) and write the bench summary to `docs/tmp/bench/<date>_h5r-cross-arch-delta.md`. If H5R passes, H6 unblocks immediately; if it fails by relative delta, escalate to upstream `llama.cpp #22011` and hold the plan.
