# A55 Gemma 3 270M Fine-Tune Plan — Closed-World Health-YAML QA

> **Status (2026-04-28 IN PROGRESS — Phase 0 closed (H1-H6 ✅); Phase 1 D1-D3 closed; Phase 2 T0-T5 ✅ closed (T5 DONE-WITH-NOTE); Phase 3 Q0 + Q1 + Q2 + Q3 + Q4 + Q5 ✅ closed (DONE-WITH-NOTE — see below); next is Phase 4 F1-F5 freeze + handoff (NOT yet authorized in this session).** Q3-Q5 closure 2026-04-28: Q3 board smoke probe emits `'72 bpm.'` on the literal P1 prompt (definitional drift fixed at deployment shape — §6 of the bench note also closes the T5 P1 OOD-`<eos>` caveat at the Q4_0 envelope). Q4 full 15-prompt sweep on the FT'd Q4_0 GGUF via the new host-driven `bench_remote.py` (R3-compliant: SSH-piped llama-completion, no remote writes) cleared **8/15 regex PASS** (vs H6 base 2/15) with manual rubric ≥ 2 on **5/15 prompts** (P1, P7, P9, A1, S1; H6 baseline was 0/15). The plan §9 ≥ 80% target was **NOT met** — the 5/15 grounded number is the v1 demo floor against H6's 0/15. Quality ceiling is dominated by training-pool gaps (multi-field discrimination, refusal-canonical-string drift, repetitive degeneration after correct first-answer token), not by Q4_0 quantization noise (Q1 GREEN). Bench summary: [`docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md`](../../tmp/bench/2026-04-28_gemma3-finetuned-final.md). Decode improved 1.82× vs H6 (17.29 tok/s vs 9.50 tok/s) — `--jinja` envelope skips the H6 plain-text-wrap tokenization overhead. Phase 4 (F1 final summary, F2 model README update, F3 backlogs, F4 `/doc_update`, F5 tag commit) is the next ramp **once explicitly authorized**. Q1 closure 2026-04-28 (same-arch x86 Path B at n_ctx=2048 cleared the deployment-shape gate at `same_top_p = 98.443%`, gate ≥ 95%; cross-arch Δ on H5R-shape corpus at n_ctx=256 cleared the kernel-parity gate at `Δ = 0.393 pp ≤ 1.0 pp` and `ratio_max_delta_p = 0.996x ≤ 3.0×` — bit-identical to H5R's base-weight Δ): [`docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md`](../../tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md) (same-arch + §11 OOM/reframe) + [`docs/tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](../../tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md) (cross-arch). Phase 1.5 Phase D prompt-only experiments saturated below G_QUALITY (1.2/3 avg, two score-0 prompts). H6 base baseline (un-fine-tuned Q4_0 on A55 CPU) was frozen 2026-04-27 at 2/15 real regex pass — every prompt fell into the YAML-echo failure mode (`docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md`). Path forward: **QLoRA fine-tune on `google/gemma-3-270m-it`, deploy as Q4_0 GGUF on the A55 CPU via llama.cpp** — canonical recipe in Google's Gemma 3 270M fine-tune blog; only published pattern with structured-task wins (FunctionGemma 58% → 85%, financial sentiment F1 0.833).
>
> **H5R GREEN 2026-04-27.** Same-quant cross-arch Δ logits-equivalence gate cleared on the first run: `same_top_p_x86_q4_0 = 94.291%`, `same_top_p_a55_q4_0 = 93.898%` → `Δ = 0.393 pp` (gate ≤ 1.0 pp); `max_delta_p` ratio `1.041x` (gate ≤ 3.0x). The A55 NEON DOTPROD + REPACK path is **not silently corrupted by `llama.cpp #22011`-class issues** at `0adede8` / `b8925`. The original H5 PUNT (98.622% / 9.393% vs Q4_0 reference) is now correctly attributed to universal Q4_0 quantization noise, not an A55 kernel bug. H5 PUNT preserved verbatim as historical context. Bench: [`docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`](../../tmp/bench/2026-04-27_h5r-cross-arch-delta.md). H6 unblocks; fine-tune proceeds.
>
> **Dataset is BUILT** (Phase 1 D1-D3, 2026-04-25): 1259 unique pairs after dedupe → train 1023 / val 126 / test 110, paraphrase-aware splitter with bench-leakage routing, audit JSONL written. Path B (composed-prompt) and Path A (raw-pair ablation) JSONL artifacts live under `tools/data/sft_v1*.jsonl`. 53 host unit tests + ruff + mypy strict green. CLI: `uv run sft-build`. See §10 for the full inventory and §6 for the calibrated `max_length=1024` token budget.
>
> **Companion docs (ground-truth pointers; do NOT duplicate):**
> - [`docs/tmp/analysis/2026-04-24_gemma3-270m-practical-evaluation.md`](../../tmp/analysis/2026-04-24_gemma3-270m-practical-evaluation.md) — empirical justification for the prompt-only → fine-tune pivot. Eleven external sources synthesized.
> - [`docs/tmp/analysis/Fine‑Tuning Gemma 3 270M for Small On‑Device Task‑Specific Models.md`](../../tmp/analysis/Fine‑Tuning%20Gemma%203%20270M%20for%20Small%20On‑Device%20Task‑Specific%20Models.md) — best-practice synthesis: dataset size, hyperparameters, quantization, evaluation, failure modes.
> - [`docs/conventions/16-slm-system-prompt.md §4`](../../conventions/16-slm-system-prompt.md) — the directive-form system prompt the fine-tune target must learn to ground in.
> - [`docs/conventions/15-model-compiler-runtime.md §5`](../../conventions/15-model-compiler-runtime.md) — `/mnt/sdcard` storage layout and on-board Python env recipe (still load-bearing for the bench harness).
> - [`docs/get-started/gemma-on-a55-get-started.md`](../../get-started/gemma-on-a55-get-started.md) — proven A55 CPU llama.cpp deployment runbook (5.87 tok/s decode at `-t 2`).
> - [`docs/plans/AI-models/models-testing-plan.md`](./models-testing-plan.md) — the larger Phase 1.5 plan; this fine-tune doc operationalizes its OQ-3(c) escalation.
> - [`models/gemma-3-270m-it/README.md §8`](../../../models/gemma-3-270m-it/README.md) — per-model best-practice analysis; the fine-tune section there sketched what this plan executes.

---

## 0. Scope and Conventions

### 0.1 In scope

- QLoRA fine-tune of plain `google/gemma-3-270m-it` on a synthetic dataset distilled from the directive-form prompt + the canonical health YAML.
- Conversion of the merged checkpoint to **Q4_0 GGUF** via `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize`.
- On-board deployment via the **existing `llama-completion -t 2` runtime** (proven 5.87 tok/s decode, 37.2 tok/s prompt eval).
- Re-bench against the existing `tools/data/prompts.yaml` 15-prompt suite and the rubric in [`models-testing-plan.md §6.2`](./models-testing-plan.md).
- Logits-equivalence gate on every GGUF before any quality bench, per the ARM64 llama.cpp wrong-logits bug ([ggml-org/llama.cpp#22011](https://github.com/ggml-org/llama.cpp/issues/22011)).

### 0.2 Explicitly out of scope

- ❌ **NPU deployment of the SLM (P1/P2 paths).** NPU is reserved for YOLOv8n vision per current product split (user sign-off 2026-04-25). The SLM stays on A55 CPU.
- ❌ **Multi-patient / generalize-to-arbitrary-YAML.** Single-patient demo is the deliverable; the model can overfit to `health_table_v1.yaml` if the math points that way.
- ❌ **Real PHI / production medical advice.** Mocked patient only. No customer-shippable medical disclaimers in this phase.
- ❌ **Full fine-tune (all weights) or end-user QAT.** QLoRA only — Google's documented recipe + reference doc §1.2 endorsement.
- ❌ **LiteRT (MediaPipe) / ONNX (Transformers.js) deployment formats.** The product runs on Yocto Linux on aarch64; GGUF is the fit.
- ❌ **M52 firmware, motion control, vision pipeline integration.** Phase 2 territory.
- ❌ **Hardware verification gates.** Vision/servo paths are not exercised; no `/hardware_check` invocation here.

### 0.3 Conventions binding this plan

| Reference | What it pins |
|---|---|
| `CLAUDE.md §3 R1/R2/R3/R6` | Board-first pre-flight; write→test→fix; SSH read-only for the agent; ground-truth hierarchy. |
| `docs/conventions/11-testing-verification.md` | Pyramid: host unit first (≥ 2 cases per test, `{desc}` in every assert), `pytest -m 'not hardware'`. |
| `docs/conventions/16-slm-system-prompt.md` | Directive-form prompt rules R-1…R-10. **Same template at training time AND inference time** — IL of consistency. |
| `docs/conventions/15-model-compiler-runtime.md §5` | `/mnt/sdcard` is the durable artifact store; `/tmp/p15site` Python env restores via §5.4 symlinks. |
| `docs/conventions/10-code-style-shell.md §2.1` | One physical line per `ssh '…'` body; BusyBox long-form flags (`head -n N`, not `head -N`). |
| `CLAUDE.md §3 IL-2 / IL-11` | 1.87 GiB usable, 512 MiB CMA, no swap; cores 2-3 reserved for ATF/secure-world (`-t 2` is the perf cap). |
| `CLAUDE.md §3 IL-13` | DRY documentation: each fact in one file; this plan points to canonical files instead of duplicating. |

---

## 1. Decision Context — why fine-tune, why now

### 1.1 What prompt-engineering already proved

| Pass | Setup | Verdict | Source |
|---|---|---|---|
| Phase B Pass 1 | NPU vendor BF16 VMFB + system-prompt-embedded record + warm single-process | G_QUALITY 1.2/3 avg, P3 social-refusal score 0, P5 summarization-hallucination score 0; one win on multi-field fact_lookup | `2026-04-24_gemma3-270m-summary.md` |
| Phase B Pass 2 | Same model + 5 follow-up probes with record in user turn | Surfaced template-lock, key-blindness, definitional drift, YAML echo. Anchoring + shape-constraint prefaces did not fix multi-field discrimination. | `2026-04-24_gemma3-270m-practical-evaluation.md §2.2` |
| Phase D probe | A55 CPU Q4_0 GGUF via `llama-completion`, P1 prompt | Returned `"Okay, I understand. I will answer the question based solely on the information provided in the record."` — definitional drift identical to NPU path | `gemma-on-a55-get-started.md §3.7` |

The failure modes are **not engineering bugs.** They are documented in published 270M evaluations (reference doc §3 cites 11 sources). At 270M parameters the attention computation cannot reliably bind a natural-language concept to a YAML key when the binding has to be inferred at inference time. The model falls back to whichever heuristic fires first.

### 1.2 What fine-tune fixes (per published evidence)

- **FunctionGemma** (NL → API call): 58% → 85% with QLoRA on low-thousands of examples (`models/gemma-3-270m-it/README.md §8.2`).
- **Marketcalls financial sentiment**: F1 0.833 (vs Gemma 3 1B's 0.85) with 38k examples on Colab T4 (reference doc §3.2).
- **Google's own emoji translator**: shipped in Google Developers blog as the canonical demonstration of "fine-tune in 10 minutes on a free T4."
- **Common pattern**: every successful structured-task Gemma-3-270M deployment in the surveyed literature was fine-tuned. None deployed prompt-only.

### 1.3 Why P3 (A55 CPU GGUF) and not P1/P2 (NPU VMFB)

| Path | Decode tok/s | Cold load | Memory | Compile cost |
|---|---|---|---|---|
| P1 vendor BF16 VMFB on NPU | 1.7 (Phase B) | 99 s warmup | ~516 MiB CMA | 0 (pre-built) |
| P2 own BF16 VMFB on NPU | unknown for Gemma 3 (untested) | similar | similar | ~30 GiB peak `iree-compile` (server-class) |
| **P3 Q4_0 GGUF on A55 CPU** | **5.87** (validated 2026-04-24) | ~3.8 s mmap + ~2.2 s prefill | ~1.07 GiB host RSS (within IL-2) | < 2 GiB cross-compile (WSL-friendly) |

P3 is **3.5× faster** than P1 (vendor's own NPU pipeline), uses no CMA, and is the same backend the Google blog recommends (community llama.cpp / Ollama). NPU is freed for YOLOv8n. **P2 is deferred** — not on this plan's critical path.

### 1.4 Risk acknowledged up front

[`ggml-org/llama.cpp#22011`](https://github.com/ggml-org/llama.cpp/issues/22011) — Gemma 3 produces **silently-wrong logits** on Cortex-A76 due to interleaved sliding-window attention + fp16 accumulation in the ARM64 CPU kernel. SL2619 Cortex-A55 is the same ARM64 family, same kernel path. The original Phase 0 H5 absolute-threshold gate was mis-calibrated against this risk (see status banner above + §4 H5R + §10.2 H5 NOTE); the redefined gate **H5R** isolates the A55-specific kernel signal from universal Q4_0 quantization noise via a same-quant cross-arch Δ test. **H5R is the logits-equivalence gate that drives the whole plan.**

---

## 2. Architecture — training infra, deployment infra

```
┌────────────────────────────────────┐         ┌────────────────────────────────────┐
│  Server (RTX 5080, 47 GiB, U24.04) │         │  WSL host (dev workstation)        │
│  ─────────────────────────────────│         │  ─────────────────────────────────│
│  • venv .venv (Python 3.12)        │         │  • llama.cpp build (cross-compile) │
│  • PyTorch ≥2.7 + CUDA 12.8        │         │  • GGUF Q4_0 quantizer             │
│  • transformers + trl + peft +     │         │  • bench score harness             │
│    bitsandbytes + datasets         │         │  • host unit tests (97 already)    │
│  • llama.cpp (for convert→GGUF)    │         │                                    │
│                                    │         │                                    │
│  STAGE: SFT + merge + bf16 GGUF    │         │  STAGE: cross-compile + scp + bench│
└────────────────┬───────────────────┘         └────────────────┬───────────────────┘
                 │                                              │
                 │  scp merged HF checkpoint OR bf16 GGUF       │  scp Q4_0 GGUF + binaries
                 ▼                                              ▼
                 ┌──────────────────────────────────────────────┐
                 │  SL2619 board (Cortex-A55, 1.87 GiB, no swap)│
                 │  ─────────────────────────────────────────  │
                 │  /mnt/sdcard/llama-cpp/llama-completion -t 2 │
                 │  /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft/ │
                 │  Reads tools/data/health_table_v1.yaml at    │
                 │  prompt-compose time via prompt_composer     │
                 │                                              │
                 │  Bench: bench_prompt.py LlamaCompletionBench  │
                 │         Adapter (one process, all prompts)   │
                 └──────────────────────────────────────────────┘
```

**Three machines, three roles, no overlap.**
- **Server** owns SFT and the bf16 GGUF. Has the GPU.
- **WSL host** owns cross-compilation (llama.cpp against Yocto SDK), Q4_0 quantization, host unit tests, scoring. Stays the dev cockpit.
- **Board** owns inference + on-device bench. READ-ONLY from the agent (R3); user runs every state-changing command.

### 2.1 YAML-retrieval architecture (resolved)

The current `prompt_composer.render_system_prompt()` already injects the full health YAML into the per-turn system prompt (200-500 tokens, well under Gemma 3's 32 K context). This stays. The fine-tune teaches the model to **bind NL questions to YAML keys reliably** — same template at train and inference per reference doc §6.4 (prompt/response format mismatch is a documented failure mode).

No retrieval layer, no tool-calling, no RAG. The whole record fits the prompt.

---

## 3. Server Bootstrap

### 3.1 Server fingerprint (probed 2026-04-25)

| Property | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.17 |
| CPU | x86_64, multi-core (exact specs TBD on next probe) |
| RAM | 47 GiB + 8 GiB swap |
| GPU | NVIDIA GeForce RTX 5080, **16 GB VRAM**, **Blackwell sm_120** |
| Driver | 580.126.09, CUDA runtime 13.0 |
| Python | 3.12.3 (system), `/usr/bin/python3` |
| Compilers | gcc/g++ 13.3.0, cmake 3.28.3 |
| Docker | 29.4.0 (alternative path for vendor `torq-compile` if P2 ever revisits) |
| ML stack | **none installed** — start from clean venv |
| HF egress | reachable (HTTP/2 200) |
| sudo | available (password) |

**⚠ Critical setup gotcha — Blackwell sm_120.** Stable PyTorch ≤ 2.5 was built for sm_70/75/80/86/89/90 only. The RTX 5080 will fail with `CUDA error: no kernel image is available for execution on the device` if you install default `pip install torch`. **Required**: PyTorch ≥ 2.7 with CUDA 12.8 wheels (or 2.8+ nightlies). The bootstrap command below pins this.

### 3.2 H2 command sheet — bootstrap the server

**Source of truth: [`tools/scripts/server-bootstrap.sh`](../../../tools/scripts/server-bootstrap.sh).** That script is idempotent (re-running reuses the existing venv, fast-forwards `llama.cpp`, lets pip skip already-installed packages) and self-checks the Blackwell sm_120 path before declaring PASS. Do not paste pip commands by hand — drift between this doc and the script is the failure mode this section is designed to prevent.

**Step 1 — Get the script onto the server** (run from WSL host, cwd = repo root)

```bash
scp tools/scripts/server-bootstrap.sh <user>@<server>:~/server-bootstrap.sh
ssh <user>@<server> 'chmod +x ~/server-bootstrap.sh'
```

**Step 2 — Run the script** (interactive shell on the server, OR `ssh -t`)

The first invocation needs system packages (`python3.12-venv`, `python3-dev`, `build-essential`, `git`, `curl`). Pass `--with-system-deps` once; subsequent runs omit it. Output is tee'd to `~/sl2619-finetune/bootstrap-<timestamp>.log` automatically.

> ⚠ `--with-system-deps` invokes `sudo apt-get`, so the script needs a TTY. **Do not** run it via `ssh <host> '<cmd>'` (no `-t`) on the first run — `sudo` cannot prompt over a non-interactive ssh and the script aborts at Phase 2 with `sudo: a password is required`. Use one of the two patterns below.

```bash
# Pattern A — interactive shell (recommended; all prompts in one place):
ssh <user>@<server>
~/server-bootstrap.sh --with-system-deps

# Pattern B — one-liner with forwarded TTY:
ssh -t <user>@<server> '~/server-bootstrap.sh --with-system-deps'
# You will be prompted twice: SSH key passphrase, then sudo password.

# Subsequent re-runs (no sudo needed — non-interactive ssh works fine):
ssh <user>@<server> '~/server-bootstrap.sh 2>&1 | tail -n 60'

# Optional flags:
#   --use-nightly-pytorch   # only if stable cu128 lacks sm_120 kernels
#   --no-llama-cpp          # skip the llama.cpp build
#   --smoke-tokenizer       # adds Gemma 3 tokenizer load smoke (run AFTER `hf auth login`)
```

The script fail-fasts at Phase 2 if `sudo` would block on a password prompt with no TTY — it prints the three recovery options and exits before doing anything irreversible. Re-runs are idempotent (existing venv reused, llama.cpp fast-forwarded, pip skips installed wheels).

Expected runtime: **~3-6 min** on the RTX 5080 host (PyTorch download dominates at ~3 GB; subsequent runs are ~30 s).

**Step 3 — Hugging Face auth** (one-time, interactive — only needed before downloading Gemma 3 weights)

```bash
source ~/sl2619-finetune/.venv/bin/activate
hf auth login                    # huggingface_hub ≥ 0.25; paste a token with Read access to google/gemma-3-270m-it
# Older HF CLI (< 0.25): huggingface-cli login
~/server-bootstrap.sh --smoke-tokenizer    # confirms tokenizer pull works end-to-end
```

**Step 4 — Send the result back**

```bash
# On the server, copy the latest summary to a known name:
ls -1t ~/sl2619-finetune/bootstrap-*.log | head -n 1
# Then from WSL host:
scp <user>@<server>:~/sl2619-finetune/bootstrap-<timestamp>.log docs/tmp/h2-bootstrap-<date>.log
```

Paste the trailing `BOOTSTRAP SUMMARY` block back into chat; that's all the agent needs to confirm H2 GREEN.

#### Expected PASS lines (the script prints these literally)

Total PASS count is **18** on the first run with `--with-system-deps`, **17** on subsequent runs (the `System deps installed` row only appears when that flag is passed). The `venv ready` row's detail flips between `(created)` and `(reused)` depending on whether `$WORKSPACE/.venv` already exists — both are PASS.

```
================ BOOTSTRAP SUMMARY ================
  PASS  OS detected — Ubuntu 24.04.x LTS
  PASS  Python ≥ 3.11 — 3.12.x
  PASS  NVIDIA GPU + driver — NVIDIA GeForce RTX 5080 (driver 580.x)
  PASS  Disk ≥ 20 GB at $HOME — NN GB
  PASS  System deps installed                # only if --with-system-deps
  PASS  venv ready — /home/<user>/sl2619-finetune/.venv (created|reused)
  PASS  pip + wheel upgraded — pip 26.x
  PASS  PyTorch installed — 2.11.x+cu128
  PASS  SFT stack installed (with bitsandbytes)
  PASS  torch still on GPU wheel after Phase 5 — 2.11.x+cu128
  PASS  llama-quantize built — /home/<user>/llama.cpp/build/bin/llama-quantize
  PASS  convert_hf_to_gguf.py present
  PASS  gguf installed — 0.18.0
  PASS  Python imports clean
  PASS  torch.cuda.is_available()
  PASS  GPU capability — sm_120 (NVIDIA GeForce RTX 5080)
  PASS  bf16 matmul on GPU
  PASS  bitsandbytes 4-bit GPU smoke

  PASS: 17-18   FAIL: 0

RESULT: PASS
```

Anything other than `RESULT: PASS` halts H2. The summary lists the exact failed check; troubleshoot per the table below before re-running.

#### Troubleshooting

| Symptom in summary | Likely cause | Fix |
|---|---|---|
| `FAIL  NVIDIA driver — nvidia-smi not on PATH` | host has no NVIDIA driver, or you're on a non-GPU node by mistake | Verify with `lspci -nn \| grep -i nvidia`. Install: `sudo apt install nvidia-driver-580-open && sudo reboot`. The bootstrap is read-only of the driver. |
| `FAIL  CUDA not available — wheel built without CUDA, or driver not loaded` | PyTorch installed CPU wheel (PyPI default is CPU on Ubuntu when cu-index is unreachable) | Re-run with explicit network access. Confirm wheel: `python -c "import torch; print(torch.version.cuda)"` should print `12.8`. If empty, force-reinstall: `pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio`. |
| `FAIL  torch downgraded to CPU wheel during Phase 5` (also: `torchvision 0.26.0+cu128 requires torch==2.11.0, but you have torch 2.6.0+cpu`) | Pre-fix script: Phase 5's `pip install -U bitsandbytes` re-resolved `torch>=2.3,<3` against PyPI default (no cu128 visibility) and downgraded the cu128 wheel to a CPU wheel. Patched `server-bootstrap.sh` writes a `~/sl2619-finetune/.torch-pin.txt` constraints file in Phase 4 and passes `--extra-index-url cu128 -c <pin>` to every Phase 5/6 pip call to prevent recurrence. | Re-scp the patched script and re-run **after** wiping the broken torch:<br>`source ~/sl2619-finetune/.venv/bin/activate && pip uninstall -y torch torchvision torchaudio bitsandbytes && deactivate`<br>then `~/server-bootstrap.sh` (no `--with-system-deps`; sudo not needed for re-runs). |
| `ERROR: Cannot install torch~=2.6.0 because these package versions have conflicting dependencies. ResolutionImpossible` in Phase 6 | llama.cpp's `requirements/requirements-convert_hf_to_gguf.txt` declares `torch~=2.6.0` (with `transformers==5.5.1` from `requirements-convert_legacy_llama.txt` since upstream commit `c8ac02fa1`, 2026-04-09) — the legacy `torch` pin still collides with our cu128 `torch 2.11.x` (Phase 4) regardless of the transformers version. With our constraints pin in place, pip can't reconcile and aborts. | Patched `server-bootstrap.sh` no longer installs llama.cpp's full requirements file — it installs `gguf` directly (the only package convert_hf_to_gguf.py needs beyond what Phase 5 already provides). The convert script runs fine on the newer torch (2.11.x) + transformers (≥5.5.1) — both are forward-compatible with the script at the current `665abc609` submodule pin. Recovery: re-scp the patched script and re-run; Phase 6 will install just `gguf`. Or one-shot manual: `source ~/sl2619-finetune/.venv/bin/activate && pip install gguf && deactivate`. |
| `FAIL  bf16 matmul — PyTorch wheel may lack sm_120 kernels` | stable cu128 lags Blackwell by a release | Re-run with `--use-nightly-pytorch`. If that still fails, the RTX 5080 firmware needs a driver bump (≥ 580). |
| `FAIL  bitsandbytes 4-bit — QLoRA path will fail` | bnb wheel ABI mismatch (CUDA 12 vs 13, glibc < 2.35, or sm_120 pre-support) | Try `pip install --pre bitsandbytes`. If still red, fall back to BF16 LoRA (drop `BitsAndBytesConfig` from §6 and `load_in_4bit=True`). RTX 5080 has 16 GB so 270M BF16 LoRA fits trivially. |
| `FAIL  Gemma 3 tokenizer load — missing HF auth or network egress` | no token, or token lacks gated-repo read, or HF outage | Run `hf auth login` and accept the model license at https://huggingface.co/google/gemma-3-270m-it. Re-test with `--smoke-tokenizer`. |
| `FAIL  Python ≥ 3.11 — found 3.10.x` | system Python is too old | Install 3.12: `sudo apt install python3.12 python3.12-venv` and re-run. The script uses whichever `python3` is first on PATH. |
| `FAIL  llama-quantize binary missing — build target failed` | cmake or g++ not installed, or out of disk | Re-run with `--with-system-deps`; if disk-bound, `df -h ~` should show ≥ 5 GB free. |
| `sudo: a terminal is required to read the password; sudo: a password is required` | Phase 2 invoked sudo over a non-interactive ssh (`ssh <host> '<cmd>'` has no TTY) | Use Pattern A (interactive `ssh <host>` then run script) or Pattern B (`ssh -t <host> '~/server-bootstrap.sh --with-system-deps'`). The script aborts before doing anything irreversible — re-running is safe. |
| Re-run hangs at `pip install` | corporate proxy or upstream throttling | The pip transport is non-idempotent only on partial download. Re-running picks up where it left off; if persistently stuck, set `PIP_INDEX_URL` or run `pip config set global.timeout 60`. |

### 3.3 What stays on the WSL host

- `tools/` — host unit tests, bench score harness, prompt composer (already mature).
- `llama.cpp` cross-compiled against Yocto SDK (already at `.cache/llama-bench/llama.cpp/build/bin/`, runbook §3.4). Needed only if we replace the on-board llama-cpp binaries; the fine-tune doesn't change the binary layer.
- The `LlamaCompletionBenchAdapter` will live in `tools/src/sl2619_tools/bench_prompt.py` (host-tested first).

---

## 4. Phases

R2 cadence: **one chunk → unit test (or board smoke) → run → fix → next chunk.** No batched code-without-tests.

### Phase 0 — Feasibility de-risk (~1.5 days)

The point of Phase 0 is to fail fast. The logits-equivalence gate is now **H5R** (same-quant cross-arch Δ — H5 historical result is preserved as context but no longer the gate). If H5R fails by relative delta, the P3 path is gated until upstream llama.cpp #22011 (or a same-arch reproduction on x86_64) clarifies whether the divergence is structural or fixable.

| ID | Action | Gate | Owner |
|---|---|---|---|
| H0 | Mount + auto-mount SD card (DONE 2026-04-25 — fstab entry `LABEL=SL2619-models /mnt/sdcard ext4 defaults,nofail,noatime,x-systemd.device-timeout=10 0 2`) | persistent across reboot | user (done) |
| H1 | ✅ **DONE 2026-04-26**. Re-probe board + server READ-ONLY via `/board_probe` (extended with `--target=sl2619\|server` in the same session — board target preserves the SL2619 alias rename `nouslogic-wsl` → `nouslogic-sl2619`). Snapshots refreshed at `docs/tmp/sl2619-status.md` and `docs/tmp/nouslogic-server-status.md`. | snapshot under 10 min | agent + user (SSH) |
| H2 | ✅ **DONE 2026-04-26**. Server venv + PyTorch sm_120 + bnb 4-bit GREEN via `tools/scripts/server-bootstrap.sh` (paste-able command sheet at §3.2). Validated end-to-end on RTX 5080: `torch 2.11.0+cu128`, compute_cap (12, 0), bf16 matmul OK, `bnb.nn.Linear4bit` 4-bit forward OK, `BitsAndBytesConfig(load_in_4bit=True, ...)` builds. Three latent foot-guns surfaced and fixed during the run (kept here as install-chain ground truth — see `backlogs.md §1.20`): (1) `--with-system-deps` requires `ssh -t` or interactive shell (sudo TTY); (2) cu128 torch wheel must be pinned via `~/sl2619-finetune/.torch-pin.txt` constraints + `--extra-index-url` on every later pip call or `bitsandbytes` re-resolves it down to a CPU wheel; (3) `pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` declares `torch~=2.6.0` + `transformers<5.0.0` and ResolutionImpossibles against our cu128 + transformers 5 — install just `gguf` directly instead. | `RESULT: PASS` (17/0) — see `~/sl2619-finetune/bootstrap-20260426-161055.log` | user |
| H3 | On-board re-stage if needed: restore `/tmp/p15site` symlinks per `15-model-compiler-runtime.md §5.4`; redeploy llama-cpp + base Q4_0 GGUF if absent | `llama-completion --version` prints `(0adede8)` | user |
| H4 | ✅ **DONE 2026-04-26**. `LlamaCompletionBenchAdapter` shipped in `tools/src/sl2619_tools/bench_prompt.py` parallel to `Gemma3BenchAdapter`. Constructor takes `binary_path`, `model_path`, `n_threads=2`, `n_predict=128`, `temp=0.0`, `top_k=1`, `seed=42`, `subprocess_timeout_s=120.0`. `run(user_text) → AdapterRunResult` shells out via injectable `runner` callable (default = `subprocess.run`), wraps `user_text` with Gemma 3 chat-template markers identical to `compose_prompt(candidate="gemma3", …)`, writes to a per-call temp file fed via `-f`, parses stdout response after `<start_of_turn>model\n`, parses stderr `llama_perf_context_print:` block for load / prompt-eval / decode / total. Per-call `wall_ms_load` reported; main loop dispatcher on `--adapter {gemma3_vendor,llama_completion}` keeps both paths alive. JSONL now self-contained with `pass_pattern` / `pattern_flags` / `passed_regex`. | host `pytest -k llama_completion` 22/22 green; full 273-test suite green; ruff + mypy strict clean | agent (codes), user (deploys at H6) |
| H5 (historical) | ⚠ **PUNT 2026-04-26 — gate mis-calibrated, preserved as context only** (full chronology in §10.2 H5). Original gate `same_top_p ≥ 99.99%` / `max Δp ≤ 0.5%` is unreachable for Q4_0-vs-FP16 on any architecture (upstream BF16-vs-FP16 same-CPU baseline = 99.739% / 4.186%). Measured A55 = 98.622% / 9.393% — within the range upstream publishes for Q4_0 quantization noise. **No longer the blocking gate.** Investigation leads (REPACK, Flash Attention) ran to completion and ruled out FA as cause; full record in §10.2. | superseded by H5R | done |
| **H5R** | ✅ **GREEN 2026-04-27.** Same-quant cross-arch Δ test passed: `same_top_p_x86_q4_0 = 94.291%`, `same_top_p_a55_q4_0 = 93.898%` → `Δ_same_top_p = 0.393 pp` (gate ≤ 1.0 pp); `max_delta_p_x86_q4_0 = 49.781%`, `max_delta_p_a55_q4_0 = 51.804%` → ratio = `1.041x` (gate ≤ 3.0x). Both gates pass with substantial headroom. The A55 NEON DOTPROD + REPACK path is **not silently corrupted by `llama.cpp #22011`-class issues** at `0adede8` / `b8925`. Per-chunk data is bidirectional (A55 *beats* x86 on chunk 1 at 96.850% vs 96.063%) — confirms residual is structural ISA-level FP arithmetic-order, not a directional kernel bug. Bench: [`docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`](../../tmp/bench/2026-04-27_h5r-cross-arch-delta.md). | gates passed | done |
| H6 | ✅ **DONE 2026-04-27.** Baseline bench: un-fine-tuned base Q4_0 GGUF on A55 CPU via `LlamaCompletionBenchAdapter`, full 15-prompt `prompts.yaml` sweep. Frozen numbers at [`docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md`](../../tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md): **2/15 real regex pass** (manual rubric: 0/3 on every prompt — pure YAML-echo definitional drift). Aggregate decode 9.50 tok/s on 745-820 token bench prompts (slower than H3's 15.5 tok/s on 82-token probe1 due to KV-cache scaling). Two latent harness bugs surfaced + fixed in-flight: (1) perf-block prefix renamed to `common_perf_print:` at b8925; (2) chat-template special tokens detokenize to empty so the divider is bare `\nmodel\n` not `<start_of_turn>model\n` — full chronology in §10.2 H6. Variance check (3 runs) deferred to Q4 if perf regresses. | full sweep produces JSONL + markdown summary | done |

**Fail action for H5R**: if `Δ_same_top_p > 1.0 pp` (or `max_delta_p_a55` blows up against the x86 Q4_0 floor), stop here. Open / link to an upstream tracking issue (e.g. `llama.cpp #22011`), hold the plan, communicate to user. The fine-tuned model would inherit the same kernel divergence — chasing the SFT delta on top of an A55-only Q4_0 corruption is meaningless. **If H5R passes**, the H5 PUNT is reclassified as expected Q4_0 noise; H6 unblocks and the fine-tune proceeds.

**Note on H5R thresholds**: `1.0 pp` and `≤ 3× max_delta_p ratio` are starting points, not load-bearing. The actual calibration is the x86 Q4_0 baseline captured in step (2) — A55 is judged relative to its same-arch x86 sibling, not against an absolute ideal. Tighten or relax once the x86 baseline number is known. Document the chosen threshold + rationale in the H5R bench summary.

### Phase 1 — Synthetic dataset (D1-D3 ✅ DONE 2026-04-25; D2 hand-curation + D4 mixin still optional)

| ID | Action | Gate | Owner | Status |
|---|---|---|---|---|
| D0 | Generate raw examples by pasting §5 prompt into chatbots (Gemini / Claude / ChatGPT / Perplexity / DeepSeek). | ≥ 100 examples per source; combined into `tools/data/clean_sft_dataset.json` | user | ✅ DONE — 1400 raw rows |
| D1 | Author `tools/src/sl2619_tools/sft_dataset.py` — D1a loader + dataclass + schema, D1b dedupe, D1c class auto-tagger, D1d bench-leakage scanner (0.80 ratio threshold), D1e paraphrase-aware splitter with 5 routing reasons + drain guard, D1f Path B / Path A JSONL emitters, D1g `sft-build` CLI. | host pytest green; 53 cases pass; ruff + mypy strict; CLI emits 6 JSONL + 1 audit JSONL | agent | ✅ DONE — see §10.1 |
| D2 | **(2 sub-steps)** D2-split: stratified 80/10/10 split (paraphrase-aware: bench-prompt hits + same-instruction-conflicts + output clusters + instruction clusters all force-routed to test). D2-curation: optional human review pass — sample N rows of each class, eyeball-correct any output containing a number not in YAML. Per-row audit at `tools/data/sft_v1.audit.jsonl` lets the reviewer scan in O(linear). | split-DONE: `tools/data/sft_v1.{train,val,test}.jsonl` ✅; curation-PENDING (deferred — `clean_sft_dataset.json` was already chatbot-cleaned, so a stratified random sample of ~50 rows is the realistic next pass) | agent (split) + user (curate) | ⚠ split DONE; curation pending |
| D3 | Unit tests for `sft_dataset.py`: deterministic split, force-routing semantics for all 5 reasons, drain guard, leakage gate, JSONL round-trip. Each parametrized table-driven per `11-testing-verification.md §3.1`. | ≥ 10 cases pass; ruff + mypy strict | agent | ✅ DONE — 53 cases (35 unit + 5 canonical-pool + 11 emitter + 2 CLI smoke) |
| D4 | Optional: ~50 generic-instruction mix-ins (e.g. `databricks/databricks-dolly-15k` closed_qa subset) to prevent catastrophic forgetting per reference doc §6.2. Tag as class `mixin_general`. | mix at most 5% of total train set | user decides | ⏳ deferred per OQ-FT-5 (default off for v1) |

### Phase 2 — Fine-tune on server (~half day)

| ID | Action | Gate | Owner |
|---|---|---|---|
| T0 | ✅ **DONE 2026-04-27**. Copied `tools/data/sft_v1.{train,val}.jsonl` to server at `~/sl2619-finetune/data/`. Server-side line counts and sha256 match host byte-for-byte: train 1023 lines / sha256 `6699ee41…`, val 126 lines / sha256 `b6443d7d…`. (Initial attempt aborted with a malformed `scp` with no remote destination, which silently overwrote local `val.jsonl` with `train.jsonl`; recovered by deterministic regen via `uv run sft-build` — host hashes restored, then re-uploaded one file at a time.) | files present, line-count matches host | user |
| T1 | ✅ **DRY-RUN DONE 2026-04-27.** Authored `tools/scripts/finetune.py` mirroring Google notebook structure. **Key differences from emoji notebook:** dataset = our JSONL (`datasets.load_dataset("json", data_files=...)`); message format = `[{"role":"user","content":<§4 directive + YAML + question>},{"role":"assistant","content":<answer>}]` (no system role — Gemma 3 has none, fold into user turn per `16-slm-system-prompt.md §2`); train and inference templates **identical** (load directive system prompt from `prompt_composer.render_system_prompt()` at preprocess time). **trl 1.3.0 API drift surfaced 2026-04-27**: `DataCollatorForCompletionOnlyLM` was removed in trl 1.x and `max_seq_length` was renamed to `max_length`; Gemma 3's chat template lacks `{% generation %}` markers so `assistant_only_loss=True` no-ops silently. Workaround codified in script — convert messages → prompt-completion shape (user-turn rendered with `add_generation_prompt=True`, completion = bare assistant text), and set `SFTConfig(completion_only_loss=True)` for masking. **§6 deviation logged**: `modules_to_save=["lm_head","embed_tokens"]` removed for v1 (would split Gemma 3's tied weights into two FP-precision copies, ~167M each, blocking Q0's GGUF conversion per peft's `ensure_weight_tying` warning + risking catastrophic forgetting on 1023 examples). Pure LoRA on `target_modules="all-linear"` is the right surface for definitional-drift fix. Dry-run results: 930 tokens (≤ 1024), `<bos>` once, no system role, `model dtype: torch.bfloat16 device: cuda:0`, **`trainable%: 1.3965`** (3,796,992 / 271,895,168), peft tied-weight warning gone. Documented escalation if T5 smoke shows no behavior change: reintroduce `modules_to_save=["embed_tokens"]` (one only) + `ensure_weight_tying=True`. | script runs `--dry-run` that loads model + tokenizer + 1 dataset row, prints decoded preview + BOS-once + no-system-role assertions, exits | agent |
| T2 | ✅ **DONE 2026-04-28** (with §6 deviation). LoRA + SFT config — pinned values in §6 below (single source of truth). **Calibration update 2026-04-25**: `max_length=1024` (NOT 512 as the Google emoji notebook uses) — Path B user content is 2608-2689 chars (~750-820 tokens, measured against the actual `sft_v1.train.jsonl`); the assistant target adds ~80 tokens at most; 1024 leaves comfortable headroom for tokenizer drift. **§6 deviation logged 2026-04-28** (full rationale in `tools/scripts/finetune.py::_build_sft_cfg` and the footer entry below): `per_device_train_batch_size: 4 → 1`, `gradient_accumulation_steps: 4 → 16` (effective batch unchanged at 16), new `per_device_eval_batch_size=1`. Root cause: Gemma 3 270M has `vocab_size=262,144`; PDB=4 logits tensor at seq=1024 / BF16 = 2 GiB and the SFT loss path's `[..., :-1, :].contiguous()` materializes another ~2 GiB peak. First training attempt OOM'd at step 0 with `Tried to allocate 3.66 GiB ... 11.72 GiB already in use`. PDB=1 drops logits 4× and brings peak to ~10 GiB on a 15 GiB-free GPU. | first 100 steps complete without OOM (after PDB=1 deviation: 192/192 steps clean, peak ~10 GiB of 15 GiB free) | user (ran) |
| T3 | ✅ **DONE 2026-04-28**. Trained 3 epochs / 192 steps in **5.4 min wall-clock** (`train_runtime: 326.4s`). RTX 5080 / cu128. T3 gate **PASSES** with healthy margins: per-epoch `eval_loss = {0.9697, 0.7983, 0.6936}` — strictly monotone-decreasing for both transitions ✅; final `train_loss=0.6277 < eval_loss=0.6936 × 1.5 = 1.040` ✅ (and actually train < eval by 0.066, ~10% — generalizing, not overfitting); no OOM; all three epoch-end checkpoints landed at `~/sl2619-finetune/adapters_v1/checkpoint-{64,128,192}/` (7.6 MB adapter_model.safetensors each, plus optimizer.pt + scheduler.pt). **Bonus signals**: `mean_token_accuracy` (eval) climbs monotone 0.7613 → 0.7978 → 0.8152; `entropy` decreasing 1.352 → 0.800 → 0.615 (model becoming more confident); `grad_norm` stable in 4.2-6.2 range (no exploding/vanishing). **Masking-correctness gate**: initial `train_loss=1.326` (advisor flagged 1.5-3.0 as healthy; 1.326 slightly below lower bound but loss converges cleanly — `completion_only_loss=True` masking works as expected on the trl 1.3.0 prompt-completion path). Log: `~/sl2619-finetune/logs/train-20260428-064801.log`. Tensorboard events under `adapters_v1/runs/` (HF Trainer auto-relocated despite our explicit `logging_dir=runs/` — cosmetic). Adapter config confirms LoRA wiring: 7 target modules (q/k/v/o/gate/up/down_proj), `r=16`, `lora_alpha=32`, `modules_to_save=null` (deviation took effect), `ensure_weight_tying=false`, total adapter `7,594,496 BF16 params`. **Best-eval-loss checkpoint is `checkpoint-192` (eval_loss 0.6936 — final epoch).** Trainer ran with `load_best_model_at_end=False` (HF default), so `trainer.save_model()` wrote the FINAL adapters at top-level `adapters_v1/`, which equals checkpoint-192. T4 merge consumes that. | val loss decreases for ≥ 2 epochs ✅; train loss < val loss × 1.5 ✅ | user (ran), agent (read logs) |
| T4 | ✅ **DONE 2026-04-28**. Picked **`checkpoint-192`** (best eval_loss 0.6936; epoch 3). Ran `python merge.py --adapters ./adapters_v1/checkpoint-192` on server: load base `google/gemma-3-270m-it` BF16 + LoRA adapter → `merge_and_unload()` → `save_pretrained("./merged_v1/")` + `tokenizer.save_pretrained()`. **Wall: 6.20s** (cuda:0). Output `~/sl2619-finetune/merged_v1/` = `model.safetensors` 536,223,056 B (sha `57c56472…`), `config.json` 1495 B (sha `c544327e…`), `tokenizer.json` 33,384,567 B (sha `daab2354…`), `chat_template.jinja` 1532 B, `tokenizer_config.json` 731 B, `generation_config.json` 167 B. **Lightweight smoke (1 prompt)**: load merged_v1 as `Gemma3ForCausalLM` (vanilla HF, NOT PeftModel ✅); for prompt `"What is my heart rate? My YAML record says: heart_rate_bpm: 72"` (29 tokens) the merged model answered **`'You are currently running at 72 beats per minute.'`** (13 tokens, 0.32s, do_sample=False) — extracts the YAML value `72` instead of the base model's definitional drift / YAML-echo (per H6 baseline). **Bug found and fixed in `_resolve_adapter_path` (`tools/scripts/merge.py`)**: `sorted(glob(checkpoint-*))` is lexicographic, so `["checkpoint-128","checkpoint-192","checkpoint-64"][-1]` = `checkpoint-64` (worst eval_loss); patched to sort by integer step. First merge (auto-pick) consumed checkpoint-64 by accident; corrected merge with explicit `--adapters ./adapters_v1/checkpoint-192` and patched `merge.py` synced to server (sha `813334a1…`). Log: `~/sl2619-finetune/logs/merge-20260428-071112.log`. | merged dir contains `model.safetensors`, `config.json`, `tokenizer.json` ✅ | agent |
| T5 | ✅ **DONE-WITH-NOTE 2026-04-28**. Authored `tools/scripts/t5_smoke.py` (server-only, no `sl2619_tools` import; 96 max_new_tokens, `do_sample=False`, sequential model load with `del + gc + cuda.empty_cache()` between). Bundle of 5 prompts pre-rendered host-side via `prompt_composer.render_system_prompt(now=date(2026,4,25))` to match training-time prompt shape verbatim → `t5_smoke_bundle.json` (sha `ee93caa9…`). Both files scp'd to `~/sl2619-finetune/`; ran `python t5_smoke.py --bundle ./t5_smoke_bundle.json --base google/gemma-3-270m-it --merged ./merged_v1 --out-dir ./logs`. **Result**: regex pass `base 0/5` → `merged 4/5`, **delta +4**. Merged ≥ base on every prompt qualitatively (no regressions). Per-prompt merged completions: P3 `"Lisinopril, Atorvastatin, Aspirin, Vitamin D3."` (17 tok, ✓); P6 `"Penicillin."` (4 tok, ✓); D1 `"I answer questions from your health record only."` (10 tok, ✓ — exactly the §4 directive refusal); S1 `"Your current medications include: Lisinopril, Metformin, Atorvastatin, Aspirin, and Vitamin D3."` (26 tok, ✓ — grounded summarization, no hallucinated values). **P1 caveat**: both base and merged emit `<eos>` as the FIRST new token for the literal phrasing "what is my **current** heart rate?" → 1 new token decoded as `''` for both. Diagnostic (`docs/tmp/bench/t5-smoke-20260428-072748-p1-diagnostic.md`) shows the merged model knows `72` cleanly — three rephrasings (`"what is my heart rate?"`, `"what is my heart_rate_bpm value?"`, `"tell me my heart rate"`) all yield `"72.<eos>"`. The training pool has exactly 1 row mentioning any heart-rate question (`"tell me my blood pressure and heart rate"`); "current" before "heart rate" is **out-of-distribution greedy-decode trigger**, not a generation-config bug or SFT regression (base also emits `<eos>` first on the literal P1). Merged model passes plan §10.3 criterion (`merged ≥ base; no obvious regressions`); fails the brief's stricter `P1 must answer 72` criterion at literal phrasing only. **Phase 3 input** (NOT decided here): `prompts.yaml` P1 may need either acceptance-as-known-artifact, phrasing edit, or SFT pool augmentation before Q3-Q5 bench scoring. Artifacts under `docs/tmp/bench/`: `t5-smoke-20260428-072748.{jsonl,md}` (5-prompt run) + `t5-smoke-20260428-072748-p1-diagnostic.md` (variant probe). Server log: `~/sl2619-finetune/logs/t5-smoke-run-20260428-072747.log`. | merged ≥ base on each of the 5 prompts (qualitative); no obvious regressions ✅ (with P1 caveat above) | user reviews |

### Phase 3 — Quantize + on-board bench (~half day)

| ID | Action | Gate | Owner |
|---|---|---|---|
| Q0 | ✅ **DONE 2026-04-28**. On server (`~/sl2619-finetune`, llama.cpp HEAD `b1a5bd4`): `python ~/llama.cpp/convert_hf_to_gguf.py ./merged_v1/ --outfile ./merged_v1.bf16.gguf` then `~/llama.cpp/build/bin/llama-quantize ./merged_v1.bf16.gguf ./merged_v1.q4_0.gguf Q4_0`. **First attempt failed** at `convert_hf_to_gguf.py:1238` with `assert max(tokenizer.vocab.values()) < vocab_size` because Gemma 3 270M has `len(tokenizer.vocab) == 262145` IDs but `config.json: vocab_size = 262144` → `max == 262144` violates the strict-less-than gate. Diagnostic confirmed both base AND merged tokenizer have `max=262144` (so it's NOT a `tokenizer.save_pretrained()` regression — it's an upstream assertion that's wrong for Gemma 3 in the BPE-fallback code path). **Root cause + fix**: `merged_v1/` lacks `tokenizer.model` (only `tokenizer.json` from `tokenizer.save_pretrained()`), so `Gemma3Model.set_vocab()` falls into `_set_vocab_gpt2()` which trips the assertion. The SentencePiece path `_set_vocab_sentencepiece()` doesn't have this check. Fix: pull `tokenizer.model` from HF Hub via `huggingface_hub.hf_hub_download(repo_id="google/gemma-3-270m-it", filename="tokenizer.model", local_dir="./merged_v1")` (4.69 MB; sha256 `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`). After that, conversion + quantization both succeed in **5.5 s wall**. Backlog: this same fix is required for any future merge → GGUF; merge.py should auto-pull tokenizer.model. See `docs/plans/backlogs.md §1.22`. **Artifacts** (`~/sl2619-finetune/`): `merged_v1.bf16.gguf` 518 MiB / **sha256 `a9c5100a4e88f2bf5526cc092d0fe6f2e08156096d9173bbd5351d1f0bb3665e`**; `merged_v1.q4_0.gguf` 231 MiB / **sha256 `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4`**. **Quantize stats**: BF16 model size 511.46 MiB → Q4_0 quant size 224.00 MiB (7.01 BPW); 1 of 236 tensors required fallback quantization (`token_embd.weight` — F16 instead of Q4_0, expected for high-vocab small models — same behavior as the unsloth Q4_0 used in H6 baseline). Server log: `~/sl2619-finetune/logs/q0-20260428-084616.log`. | `merged_v1.q4_0.gguf` 231 MiB ✓ (within 130-230 MB target — 1 MiB over the upper edge due to F16 fallback on the 262144-row token_embd; BPW 7.01 vs nominal 4.5 same as base) | user (ran), agent (composed) |
| Q1 | ✅ **DONE 2026-04-28.** Calibrated three-step logits-equivalence cleared; both gates GREEN with substantial headroom. **(a) BF16 ref + (b) x86 Q4_0 same-arch, Path B at n_ctx=2048**: `same_top_p_x86_q4_0_vs_bf16 = 98.443%` (≥ 95% gate ✓, 3.4 pp headroom; apples-to-apples base anchor on same corpus = 99.489%, so SFT cost is only 1.05 pp). Bench: [`docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md`](../../tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md). **(c) A55 cross-arch, H5R-shape corpus at n_ctx=256**: `Δ_same_top_p = 0.393 pp` (≤ 1.0 pp gate ✓), `ratio_max_delta_p = 0.996x` (≤ 3.0× gate ✓ — A55 actually fractionally under x86). **Bit-identical to H5R's base-weight Δ** → cross-arch kernel-noise floor is invariant to weight bit pattern; FT introduced no new ISA-specific behavior. Bench: [`docs/tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`](../../tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md). **Cross-arch step c uses different corpus + n_ctx than same-arch step b — full reasoning in §11 of the same-arch bench**: Path B at n_ctx=2048 OOM-kills on the 1.87 GiB / no-swap SL2619 (per-chunk reference-logits buffer = `n_ctx × vocab × float32` = 2.15 GiB at n_ctx=2048 / vocab=262144; dmesg-confirmed two SIGKILLs 2026-04-28 at total-vm 2.72 GB). The H5R-shape reframe at n_ctx=256 fits in 1.20 GiB and is a pure kernel-parity test (corpus-agnostic) which is exactly what step c needs to measure. Same-arch step b (deployment-shape signal) remains the Path B / n_ctx=2048 number; step c (kernel-parity signal) is on the H5R corpus. Both numbers are needed; neither replaces the other. | (a) BF16 ref present ✓; (b) `same_top_p_x86_q4_0_vs_bf16 = 98.443% ≥ 95%` ✓; (c) `Δ_same_top_p = 0.393 pp ≤ 1.0 pp` ✓ + `ratio_max_delta_p = 0.996x ≤ 3.0×` ✓ | agent (host scoring), user (board scp + run) |
| Q2 | ✅ **DONE 2026-04-28**. GGUF transfer happened during Q1 cross-arch step c (sha verified at that time, `587f1af6…`). Re-confirmed this session via `/board_probe`: `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` 230 MiB, sha256 `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4` (matches Q0 closure). | file present on board, sha matches host ✅ | user (transfer), agent (re-verify) |
| Q3 | ✅ **DONE-WITH-NOTE 2026-04-28**. Smoke probe revealed a deployment-envelope mismatch first: `probe1_prompt.txt` uses an ad-hoc terse record schema that doesn't match the §4 directive + Path B envelope the model was fine-tuned on (legacy of the H3 base-model probe). On that probe the FT'd model emits prompt-echo / `model_user` hallucination (consistent with prompt-envelope drift, not a Q4_0 quality regression). Re-probed with the correct deployment shape — `compose_user_text()` body via `--jinja --no-display-prompt -p $BODY -no-cnv --single-turn` — and the FT'd Q4_0 model emits `'72 bpm.'` as the first content for the literal P1 prompt (`'what is my current heart rate?'`). **§4 Q3 gate met**: definitional drift fixed at the deployment shape. **Anecdote**: Q4_0 quantization perturbed the BF16-greedy `<eos>` mass on the literal "current" phrasing back into a recoverable state — a side effect that closes the T5 P1 OOD-greedy caveat at the deployment-shape envelope (full reasoning at `2026-04-28_gemma3-finetuned-final.md §6`). Smoke logs at `.cache/q3/q3*.log`. | model emits `72` (or close) — definitional drift fixed ✅ | agent (READ-ONLY SSH; piped via stdin to llama-completion) |
| Q4 | ✅ **DONE 2026-04-28**. Full 15-prompt board sweep via the **new** host-driven `tools/src/sl2619_tools/bench_remote.py` (R3-compliant: SSH-piped llama-completion, no remote writes). Built this session because the existing on-board `bench_prompt.py` `LlamaCompletionBenchAdapter` text-wraps with literal `<start_of_turn>…` markers, and llama.cpp without `--jinja` tokenizes those as plain bytes (~5-10 sub-tokens each) → FT'd model never enters answer mode. The new harness uses `--jinja --no-display-prompt -p $BODY -no-cnv --single-turn` so chat-template special tokens (105/106) land at the right ids. 15 unit tests on the new module (`tests/test_bench_remote.py`); ruff + mypy strict clean. JSONL + log: `docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.{jsonl,log}` (15 rows, 7m 48s wall). H6b 3-prompt sanity check (P1/P3/D1) on the **base** Q4_0 GGUF with the same `--jinja` envelope confirms the H6 base failure mode is intrinsic (0/3 with `--jinja` too — base hallucinates YAML-shaped output regardless of envelope). | JSONL + log written; 15 rows ✅ | agent (host-driven, R3-compliant) |
| Q5 | ✅ **DONE-WITH-NOTE 2026-04-28**. **8/15 regex PASS** (vs H6's 2/15 — every H6 pass was YAML-echo coincidence). **Manual rubric ≥ 2 on 5/15** (P1, P7, P9, A1, S1; H6 was 0/15). Real grounded answers: P1 `'72 bpm.'`, P7 `'"2026-05-06"'`, P9 `'not in record.'`, A1 `'Please consult your clinician.'`, S1 `'- Lisinopril 10 mg 08:00 blood pressure control. - Metformin 500 mg…'` (clean 5-medication list). Plan §9 target ≥ 80% (12+/15) **NOT met** — quality ceiling is dominated by training-pool coverage gaps (multi-field discrimination on P3/P4/P6, refusal-canonical-string drift on D1/D2, repetitive degeneration after correct first-answer token on most prompts), NOT Q4_0 quantization noise (Q1 GREEN at 98.443% same_top_p). Decode rate **17.29 tok/s** aggregate (1.82× faster than H6's 9.50 tok/s — `--jinja` envelope skips plain-text-wrap tokenization overhead). Freeze: [`docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md`](../../tmp/bench/2026-04-28_gemma3-finetuned-final.md). v2 dataset expansion (multi-field, refusal-anchoring, terminator-rich completions, "current"/"now"/"present" phrasing variants) is the path to ≥ 80%; recorded in `backlogs.md §1.21`. | regex pass count + manual rubric vs H6 baseline frozen ✅ (target deferred per OQ-FT-1; v1 demo numbers recorded) | both |

### Phase 4 — Freeze + handoff (~half day)

| ID | Action | Gate | Owner |
|---|---|---|---|
| F1 | Final bench summary at `docs/tmp/bench/<date>_gemma3-finetuned-final.md` per `13-documentation-update-protocol.md §10.1`. Include: server hyperparams used, dataset sha256, GGUF sha256, full prompt-by-prompt table (base vs fine-tuned), wall-clock and tok/s. | doc present + reviewed | agent |
| F2 | Update `models/gemma-3-270m-it/README.md` §8 with the as-executed SFT recipe (deviations from Google notebook). | sha-pinned dataset + recipe present | agent |
| F3 | `docs/plans/backlogs.md` post-mortem entry (numbered next free section). | entry present | agent |
| F4 | Run `/doc_update` to refresh `CLAUDE.md` + `README.md` if any architectural fact changed (likely: SD-card auto-mount fstab is now committed convention; A55 CPU is now the SLM serving path). | both files updated and reviewed | agent |
| F5 | Tag commit: `phase-d-finetune-v1` (see `12-git-workflow.md` for tag conventions). | tag pushed | user |

### Deferred (only if Phase 3 underperforms)

- **Increase dataset to 2000-5000** examples per reference doc §2.3 ("robust" range).
- **Try Unsloth instead of vanilla TRL** — 1.6× faster, 70-80% VRAM savings (reference doc §1.4); useful only if vanilla TRL hits memory ceilings, which is unlikely on RTX 5080 16 GB for a 270M model.
- **Runtime `--lora FNAME` deployment (Plan B only — keep merged-Q4_0 as the v1 path).** Instead of `merge_and_unload + llama-quantize Q4_0`, ship base Q4_0 GGUF + a separate `convert_lora_to_gguf.py` adapter blob and load it at runtime via `llama-completion --lora adapter.gguf`. Pros: hot-swap adapters; LoRA tensors bypass REPACK (`src/llama-adapter.cpp:296+` "do not load loras to extra buffer types") which sidesteps any residual ARM kernel divergence concern; mmap base only once. Cons: heavier runtime memory (base + adapter both resident on top of base, with no Q4_0 quantization on the adapter — typically BF16); two artifacts to ship and version; cold-load grows. **Why v1 stays merged + Q4_0:** simpler embedded story (one self-contained GGUF, mmap-friendly, smaller deployment surface), no extra resident memory, and the bug-bypass story is hypothetical until H5R / Q1 actually find an A55-specific REPACK signal. **Activation triggers (any one):** (a) Q1 step (b) shows Q4_0-vs-BF16 same-arch quality drop dragging the SFT delta into the noise; or (b) H5R confirms a residual REPACK-specific divergence we can't otherwise mitigate; or (c) a future product requirement adds adapter hot-swap.
- **P2 (own BF16 VMFB on NPU)** — only if the product story changes to "NPU-accelerated SLM is required." Needs server with ≥ 48 GiB RAM (47 GiB current is borderline; `iree-compile` peaks ~30 GiB).
- **Multi-patient generalization** — would require ≥ 5x dataset diversity and multi-patient YAML fixtures.

---

## 5. Dataset Generation Prompt (preserved verbatim — paste into chatbots)

This prompt was authored 2026-04-25, paste into Gemini / Claude / ChatGPT / Perplexity / DeepSeek one at a time. Each will return ~120 examples; combine, dedupe via `sft_dataset.py`, target 800-1200 final examples after curation.

````text
You are helping me build a supervised fine-tuning (SFT) dataset for Gemma 3 270M, an instruction-tuned small language model that will run on a single-board edge device. The model's job is to answer one patient's questions using ONLY the YAML health record below — not to give medical advice, not to chat off-topic.

# The patient record (single source of truth — every answer must be grounded in this YAML)

```yaml
patient:
  name: "Test Patient"
  age: 45
  sex: "F"
  blood_type: "O+"

vitals:
  heart_rate_bpm: 72
  blood_pressure_systolic: 118
  blood_pressure_diastolic: 76
  spo2_percent: 98
  body_temperature_c: 36.7
  respiratory_rate: 16
  last_measured: "2026-04-24 08:15"

conditions:
  - name: "Hypertension"
    diagnosed_at: "2019-03-12"
    severity: "moderate"
    controlled: true
  - name: "Type 2 Diabetes"
    diagnosed_at: "2021-06-08"
    severity: "mild"
    controlled: true
  - name: "High Cholesterol"
    diagnosed_at: "2022-01-20"
    severity: "mild"
    controlled: true

allergies:
  - substance: "Penicillin"
    severity: "severe"
    reaction: "anaphylaxis"
  - substance: "Shellfish"
    severity: "moderate"
    reaction: "hives"

medications:
  - name: "Lisinopril"
    dose: "10 mg"
    schedule: "08:00"
    with_food: false
    purpose: "blood pressure control"
    avoid_drugs: ["Potassium supplements", "NSAIDs"]
  - name: "Metformin"
    dose: "500 mg"
    schedule: "08:00, 19:00"
    with_food: true
    purpose: "blood sugar control"
    avoid_foods: ["Alcohol"]
  - name: "Atorvastatin"
    dose: "20 mg"
    schedule: "21:00"
    with_food: false
    purpose: "cholesterol control"
    avoid_foods: ["Grapefruit", "Grapefruit juice"]
    avoid_drugs: ["Erythromycin"]
  - name: "Aspirin"
    dose: "81 mg"
    schedule: "08:00"
    with_food: true
    purpose: "cardiovascular protection"
    avoid_drugs: ["Ibuprofen", "Warfarin"]
  - name: "Vitamin D3"
    dose: "1000 IU"
    schedule: "08:00"
    with_food: true
    purpose: "bone health"

dietary_restrictions:
  - rule: "low sodium (under 2000 mg per day)"
    reason: "hypertension management"
  - rule: "limit added sugar"
    reason: "diabetes management"
  - rule: "no grapefruit"
    reason: "atorvastatin interaction"

appointments:
  - date: "2026-05-06"
    time: "10:30"
    provider: "Dr. Evelyn Chen"
    purpose: "quarterly diabetes check-up"
    location: "Maple Clinic, Room 204"
  - date: "2026-06-15"
    time: "14:00"
    provider: "Dr. Marcus Patel"
    purpose: "annual cardiology review"
    location: "Cardiology Dept, 3rd floor"

emergency_contacts:
  - name: "Jane Doe"
    relation: "daughter"
    phone: "+1-555-0142"

notes:
  - "Vitals within nominal range"
  - "No medication interactions flagged"
```

# Behaviour rules the model must learn

- Quote YAML values verbatim — numbers, doses, times, names. No paraphrasing of facts.
- If the YAML doesn't contain the answer: reply exactly `not in record`.
- Never invent values, dates, medications, or food rules.
- Off-topic / social chat ("tell me a joke", "what's the capital of France"): reply exactly `I answer questions from your health record only`.
- Medical advice ("should I stop my BP med?"): reply with `consult your clinician` plus a 1-sentence factual reference to the relevant YAML row.
- Output: 1–2 sentences, no preamble, no lists unless the YAML field itself is a list.
- Today's date is 2026-04-25 (use this when answering "what day is it" / "when is my next appointment in days" style questions).

# Output format (strict — I will parse this with code)

Output a JSON array. Each element has exactly three keys: `instruction`, `input` (always the empty string ""), `output`.

[
  {"instruction": "what is my heart rate?", "input": "", "output": "72 bpm."},
  {"instruction": "do I have any allergies?", "input": "", "output": "Yes — penicillin (severe, anaphylaxis) and shellfish (moderate, hives)."},
  {"instruction": "tell me a joke", "input": "", "output": "I answer questions from your health record only"},
  {"instruction": "what is my cholesterol reading?", "input": "", "output": "not in record"}
]

# What I want you to generate

Produce **120 unique examples** stratified across these classes (target counts in parentheses):

1. **fact_lookup_simple (40)** — single-field retrieval. "what is my heart rate" / "what's my blood type" / "when do I take Atorvastatin" / "what dose of Metformin" / "what's my next appointment date". Vary phrasing (formal, casual, abbreviation, typo, partial info).
2. **fact_lookup_multi (20)** — combines 2+ YAML fields. "what meds at 8am AND why" / "what should I avoid eating because of my meds" / "list my allergies with severity".
3. **fact_absence (15)** — questions whose answers are NOT in the YAML. Cholesterol numeric reading, primary-care-physician name, last A1c, EKG date, weight, height, current pain level. Output must be exactly `not in record`.
4. **medical_advice_reroute (10)** — "should I stop my Lisinopril", "is my blood pressure dangerous", "can I exercise with diabetes". Output must contain `consult your clinician` plus one factual YAML reference.
5. **domain_refusal (15)** — off-topic / social. "tell me a joke", "what's the capital of France", "write a poem about the ocean", "what's the weather in Paris", "explain quantum entanglement". Output must be exactly `I answer questions from your health record only`.
6. **summarization (10)** — "summarize my meds", "summarize my conditions", "give me an overview of my health". 1-2 sentences, every fact in YAML, no invention.
7. **temporal (10)** — uses today's date (2026-04-25). "how many days until my diabetes appointment", "when is my next visit", "what should I take next" (next = closest schedule time after current time).

Style requirements:
- Every example must be answerable (or correctly-refused) from the YAML alone.
- Vary surface phrasing: questions, polite, terse, with typos, with partial information ("my BP med" instead of "Lisinopril").
- No two questions identical even if the answer is the same.
- The assistant tone is concise and clinical — no warmth-padding ("I'd be happy to help...").
- Numbers and units must match the YAML exactly: `"118/76 mmHg"` is OK, `"118/76"` alone is OK, `"around 118"` is NOT OK.

Output the JSON array directly with no preamble. If you cannot generate 120, output as many as you can in the same format.
````

### 5.7 Class auto-tagger heuristics (used by `sft_dataset.py`)

The §5 dataset-generation prompt asks chatbots for **7 fine-grained classes**, but the implemented auto-tagger in `sl2619_tools.sft_dataset.classify_record` collapses them to **4 splitter-relevant classes** because the splitter only needs enough granularity to stratify across the bench taxonomy in `tools/data/prompts.yaml` (calibration / fact_lookup / fact_absence / domain_refusal / summarization). Decision order, first match wins:

| Class | Heuristic | Coverage on canonical pool |
|---|---|---|
| `domain_refusal` | output contains substring `I answer questions from your health record only` (case-fold) | 119 |
| `fact_absence` | output contains `not in record` or `not in your record` (case-fold) | 151 |
| `summarization` | output is multi-fact: ≥ 2 commas, OR ≥ 80 chars, OR contains a newline; OR instruction matches `^(summarize\|summary\|sum up\|condense\|overview\|outline\|list (all\|my))\b` | 341 |
| `fact_lookup` | catch-all (single-fact retrieval) | 648 |

**Known gap (calibrated 2026-04-25)**: 116 rows that the §5 prompt requested as `medical_advice_reroute` (output containing `consult your clinician` plus a YAML factual reference) currently auto-tag as `summarization` (87) or `fact_lookup` (29) because the classifier has no `medical_advice` branch. The model still trains on these rows — they just stratify under whichever bucket their output shape lands in. Fix is a 5-line classifier extension if downstream bench shows medical-advice rerouting as a regression class. Not blocking Phase 1 closure.

**No `temporal`, `fact_lookup_multi`, `fact_lookup_simple`** — temporal questions land in `fact_lookup` (single-fact retrieval of dates/times); multi-fact retrievals land in `summarization` because they're multi-comma; the simple/multi distinction adds no value to the splitter.

---

## 6. Hyperparameters — single source of truth

The fine-tune script in T1/T2 must use exactly these values unless an explicit deviation is logged in F1's bench summary. **Two deviations were required during T1-T2 (2026-04-27/28) and are codified in `tools/scripts/finetune.py` with full rationale — see §10.3 deviation table and Phase 2 T1/T2 rows.** The as-executed values are reflected in the comments below.

```python
# bnb 4-bit quant config (unchanged — no deviation)
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA config
# DEVIATION (T1 2026-04-27): modules_to_save removed for v1.
#   Original §6 had ["lm_head","embed_tokens"] per Google emoji notebook.
#   Dropped because Gemma 3 tie_word_embeddings=True causes peft to split
#   the tied pair into two independent FP copies (~167M each), producing a
#   corrupt vocabulary projection after merge_and_unload → GGUF (Q0 Phase
#   3 hard-blocked). Documented escalation if T5 shows no behavior change:
#   reintroduce modules_to_save=["embed_tokens"] (one only) + ensure_weight_tying=True.
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # resolves to q/k/v/o/gate/up/down_proj (7 modules)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # modules_to_save=["lm_head", "embed_tokens"],  # REMOVED — see deviation note
)

# SFT config
# DEVIATION (T2 2026-04-28): per_device_train_batch_size 4→1, GAS 4→16, new eval_batch_size=1.
#   Root cause: vocab_size=262,144 → logits at PDB=4/seq=1024/BF16 = 2.0 GiB +
#   shift contiguous = ~2 GiB peak; first run OOM at step 0. PDB=1 drops 4x
#   to 512 MiB. Effective batch (16) and optimization trajectory unchanged.
#   eval_batch_size default is 8 (independent of train PDB) → must pin to 1
#   or epoch-end eval OOMs after wasting a full epoch of training.
SFTConfig(
    output_dir="./adapters_v1",
    num_train_epochs=3,
    per_device_train_batch_size=1,         # DEVIATION: was 4 — Gemma 3 vocab=262144 OOM
    gradient_accumulation_steps=16,        # DEVIATION: was 4 — preserves effective batch 16
    per_device_eval_batch_size=1,          # DEVIATION: was default 8 — eval OOM at vocab=262144
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    max_length=1024,                       # ≥ 820 tokens needed for Path B; Google notebook's 512 too small
    gradient_checkpointing=False,          # PDB=1 leaves plenty of headroom; GC unneeded
    packing=False,
    completion_only_loss=True,             # trl 1.3.0 API: DataCollatorForCompletionOnlyLM removed; this replaces it
    optim="adamw_torch_fused",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="tensorboard",
    seed=42,
)
```

Rationale lives in reference doc §3 — these are the published defaults for 270M structured-task QLoRA. The `max_length=1024` deviation from the Google notebook is documented in §4 T2 — Path B's composed user turn is ~770 tokens because the directive system + full YAML are folded in (Gemma 3 has no system role). Drop back to 512 only if the dataset moves to Path A (raw pairs).

Additional trl 1.3.0 API notes (not deviations from §6 intent, but required to make the original intent work on the actual installed version):
- `DataCollatorForCompletionOnlyLM` was removed in trl 1.x → `completion_only_loss=True` on `SFTConfig` is the replacement.
- `max_seq_length` was renamed to `max_length` in trl 1.x.
- Gemma 3's chat template has no `{% generation %}` markers → `assistant_only_loss=True` (trl 1.x) silently returns all-zero masks; `completion_only_loss=True` with the prompt-completion dataset shape is the supported path.
- Dataset shape: `messages` JSONL converted to `{"prompt": <user-turn-with-add_generation_prompt=True>, "completion": <assistant-text>}` via `_to_prompt_completion()`.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **ARM64 llama.cpp wrong-logits bug ([#22011](https://github.com/ggml-org/llama.cpp/issues/22011))** | HARD GATE | Phase 0 **H5R** same-quant cross-arch Δ test (redefined 2026-04-27 — the original H5 absolute gate was mis-calibrated; details in §10.2 H5 NOTE and [`docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md §2.1`](../../tmp/analysis/2026-04-27_llama-onnx-plan-review.md)). H5R isolates A55-specific kernel divergence from universal Q4_0 quantization noise by comparing `same_top_p_x86_q4_0` against `same_top_p_a55_q4_0` against the same x86_64 BF16 `.kld`. Proposed gate: `Δ_same_top_p ≤ 1.0 pp`, `max_delta_p_a55` within ~3× of `max_delta_p_x86`. If H5R fails by relative delta, halt the plan and escalate upstream. Q1 (Phase 3) re-applies the same discipline post-fine-tune so we never re-conflate noise with bug. |
| **RTX 5080 sm_120 unsupported by stable PyTorch 2.5** | HIGH (setup) | Phase 0 H2 pins `--index-url cu128`. Smoke checks `cuda.get_device_capability` before any training. |
| **Catastrophic forgetting** (270M model losing safety/refusal behaviors) | MEDIUM | D4 optional generic-instruction mix-in; LoRA-only training preserves base weights. Reference doc §6.2. |
| **Synthetic-data inconsistency** | MEDIUM | D2 hand-curation pass; regex-flag any output containing numbers not in YAML. Reference doc §6.3. |
| **Train/inference template drift** | MEDIUM | T1 explicit: same `prompt_composer.render_system_prompt()` at training-data formatting AND inference. Reference doc §6.4. |
| **Quantization quality loss (bf16 → Q4_0)** | LOW-MEDIUM | Q1 bf16 vs Q4_0 logits equivalence; Q5 manual rubric vs base. Drop to Q5_K or Q6_K if Q4_0 degrades. Reference doc §6.5. |
| **270M ceiling on multi-step reasoning** | KNOWN LIMIT | Out of scope per §0.2. If a downstream consumer needs reasoning, route to a different model. |
| **Single-process rule** (vendor NPU runner) | N/A | Not on this plan's path — A55 CPU has no equivalent constraint. `llama-completion` spawn-per-prompt is fine. |
| **Server RAM borderline for P2 VMFB** | N/A this round | P2 is deferred. If revisited, decide between server upgrade or cloud compile. |

---

## 8. Open Questions

| ID | Question | Default action |
|---|---|---|
| **OQ-FT-1** | Quality target for Phase 3 freeze | Per user 2026-04-25: deferred. Phase 3 is a "did the score-0 prompts get fixed" check; full G_QUALITY rubric applies in a later iteration if pursued. |
| **OQ-FT-2** | Distillation TOS posture | Phase 1 dataset distilled from Gemini / Claude / ChatGPT / Perplexity / DeepSeek for an internal demo on a mocked patient. User accepts; revisit if the model ever ships to a real customer. |
| **OQ-FT-3** | Should F4 update commit `LABEL=SL2619-models` fstab pattern as a normative convention? | Yes — `15-model-compiler-runtime.md §5.3` had marked auto-mount as a "deferred user decision"; that decision is now made (committed at H0). Update §5.3 in F4. |
| **OQ-FT-4** | Multi-patient generalization | Out of scope this round per user. If revisited, generate 5 synthetic YAMLs and 200 examples per YAML; restructure prompt_composer to accept per-turn YAML. |
| **OQ-FT-5** | Mixin generic-instruction examples (D4) — how many | Default off for v1. Add only if H6 baseline shows the un-fine-tuned model has usable refusal behavior we'd lose; otherwise the v1 SFT is a fresh start. |
| **OQ-FT-6** | Train on Unsloth or vanilla TRL | Vanilla TRL (Google notebook canonical). Unsloth only if vanilla hits memory ceilings, unlikely on 16 GB VRAM for 270M. |
| **OQ-FT-7** | Model registry — push to HF Hub or keep local | Local (server filesystem + WSL host) for v1. Pushing to HF Hub introduces TOS/redistribution questions for a mocked-PHI-derived model. |
| **OQ-FT-8** | Should `classify_record` add a 5th class `medical_advice_reroute` (output contains `consult your clinician`)? | Default: defer. The 116 such rows still train the model on the rerouting pattern; the only loss is precise stratification. Revisit if Phase 3 bench shows medical-advice rerouting as a regression class — at which point add the branch, re-run `sft-build`, re-train. ~5 lines + 2 tests. |

---

## 9. Done Criteria

Phase 4 is done when **all** of:

1. **F1** bench summary frozen with prompt-by-prompt comparison vs Phase 0 H6 baseline.
2. **Visible delta on the score-0 prompts**: P1 (heart rate) returns `72`, not "Okay, I understand…"; P5 (summarization) doesn't fabricate vitals; D1 (joke) hits the refusal string. Quantitative G_QUALITY threshold deferred per OQ-FT-1.
3. **Latency parity**: TTFT and decode tok/s within 25% of base GGUF (5.87 tok/s decode; total ≤ 30 s for 1-2 sentence answer).
4. **Logits-equivalence still green** (H5R + Q1, both calibrated relative to x86 Q4_0 baseline per the 2026-04-27 redefine) — no ARM64-specific kernel corruption introduced beyond what the x86 sibling already exhibits as Q4_0 noise.
5. **F2-F5** documentation refresh + tag committed.

If (2) fails (the score-0 prompts didn't move): the Deferred items kick in (more data, Unsloth, etc.). Do NOT silently accept "fine-tune was a wash" — rerun D1-D4 with diagnosed gaps.

---

## 10. Current State Snapshot (last updated 2026-04-28)

### 10.1 Phase 1 dataset pipeline — DONE

- ✅ **D0 raw dataset collected** by user from multiple chatbots. Saved at `tools/data/clean_sft_dataset.json` (1400 Alpaca-shape triples, 200 KB).
- ✅ **D1 `tools/src/sl2619_tools/sft_dataset.py` shipped** — 7 chunks (D1a-D1g), 53 unit tests, ruff + mypy strict clean. Public surface:
  - `load_sft_pool`, `dedupe_pool` (141 dups removed → 1259 unique pairs)
  - `classify_record` / `class_distribution` (4-class taxonomy — see §5.7)
  - `load_bench_prompts`, `scan_bench_leakage` (`NEAR_DUPLICATE_RATIO=0.80`, calibrated against canonical pool)
  - `split_pool` — paraphrase-aware stratified splitter, 5 routing reasons, drain guard at 50% per class
  - `write_split_jsonl` — Path B (composed prompt, primary) + Path A (raw pairs, ablation)
  - `SplitReport.write_audit_jsonl` — per-row provenance for human review
- ✅ **CLI `sft-build`** at `tools/src/sl2619_tools/sft_build.py` — runs the full pipeline end-to-end against the canonical fixtures, prints leakage + split summaries before writing. `[project.scripts] sft-build = "sl2619_tools.sft_build:main"`.
- ✅ **Artifacts on disk** (`tools/data/`, written by `uv run sft-build`):
  - `sft_v1.audit.jsonl` — 1259 rows, one JSON line per pool entry (9 fields: pool_index, instruction, output, class, split, routing_reason, matched_bench_id, matched_bench_text, similarity)
  - `sft_v1.{train,val,test}.jsonl` — Path B, 1023 / 126 / 110 rows, ~3.6 MB total
  - `sft_v1_pathA.{train,val,test}.jsonl` — Path A ablation, ~210 KB total
- ✅ **Final class-by-split histogram**:

  | Split | n | fact_lookup | fact_absence | domain_refusal | summarization |
  |---|---|---|---|---|---|
  | train | 1023 (81.3%) | 514 | 126 | 101 | 282 |
  | val | 126 (10.0%) | 63 | 15 | 13 | 35 |
  | test | 110 (8.7%) | 71 | 10 | 5 | 24 |

  Rare-class train shares: domain_refusal 84.9%, fact_absence 83.4% — both safely majority-trained.

- ✅ **Force-routing breakdown for the test split**: bench_exact 5 (P2/P7/P9/D1/D2), bench_near 38, same_instruction_conflict 13, cluster_output 47, cluster_instruction 7. Total 110 rows = 56 bench-leakage hits + 54 cluster expansions.
- ✅ **Path B token budget calibrated**: user content 2608-2689 chars (~750-820 tokens). Drives `max_length=1024` in §6 (deviates from Google notebook's 512).
- ⚠ **Known classifier gap**: `medical_advice_reroute` rows (116 in pool, output contains `consult your clinician`) currently auto-tag as `summarization` (87) or `fact_lookup` (29). Model still trains on them; only stratification precision is lost. See §5.7. Fix iff bench shows medical-advice as a regression class.

### 10.2 Phase 0 hardware/runtime — pending

- ✅ **Server identified** (RTX 5080 / 47 GiB / Ubuntu 24.04). Bootstrap recipe in §3.2.
- ✅ **SD card persistent mount** committed (fstab `LABEL=SL2619-models  /mnt/sdcard  ext4  defaults,nofail,noatime,x-systemd.device-timeout=10  0 2`). Card is 119 GB, ~109 GB free.
- ✅ **H1 board inventory** (SD-card scan 2026-04-25):
  - `/mnt/sdcard/p15site/` + `/mnt/sdcard/pipbase/` + `/mnt/sdcard/p15-env.sh` — **present** (Python env-on-SD survived; `/tmp/p15site` symlink restoration is the only post-reboot step needed per `15-model-compiler-runtime.md §5.4`).
  - `/mnt/sdcard/torq-examples/`, `/mnt/sdcard/models/moonshine-tiny/`, `/mnt/sdcard/{bench,bench-data,bench-src,fixtures,scripts}/` — **present**.
  - `/mnt/sdcard/llama-cpp/` — **ABSENT**. H3 must redeploy `llama-cli` / `llama-completion` / `llama-bench` per `gemma-on-a55-get-started.md §3.5`.
  - `/mnt/sdcard/models/gemma-3-270m-it-q4_0/` — **ABSENT**. H3 must redeploy `gemma-3-270m-it-Q4_0.gguf` + `probe1_prompt.txt` per `gemma-on-a55-get-started.md §3.5`. Source files still cached at host `/home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/` (assumed; verify before H3).
  - CMA Free 487 MiB / 524 MiB; uptime 2h48m; fstab entry confirmed.
- ⏳ **H2 server venv** not yet bootstrapped.
- ✅ **H3 board re-stage GREEN** (2026-04-27). Env-on-SD symlinks (`/tmp/p15site` → `/mnt/sdcard/p15site`, `/tmp/pipbase`, `/tmp/p15-env.sh`) restored after the fresh boot per `15-model-compiler-runtime.md §5.4`; `python3 -c "import torq.runtime, onnxruntime"` reports `env OK` (one benign onnxruntime warning about missing `/sys/class/drm/card0` — board has no DRM/GPU node, expected). `scp`'d `llama-cli` (8.3 MB), `llama-completion` (6.6 MB), `llama-bench` (4.8 MB) into `/mnt/sdcard/llama-cpp/` and `gemma-3-270m-it-Q4_0.gguf` (231 MB) + `probe1_prompt.txt` into `/mnt/sdcard/models/gemma-3-270m-it-q4_0/`. `./llama-completion --version` confirms `version: 1 (0adede8)` / `built with GNU 13.3.0 for Linux aarch64`. Deterministic probe (`-t 2 -n 64 --temp 0.0 --top-k 1 -no-cnv`) loaded clean, emitted the expected definitional-drift answer (`"Okay, I understand. I will answer the question based solely on the information provided in the record."` — same string `gemma-on-a55-get-started.md §3.7` warned about; the whole point of Phase 0 → fine-tune). **Perf surprise — measurably faster than the runbook**: load 3.24 s, prompt eval **96.01 tok/s** (runbook §5.1 ~37), decode **15.35 tok/s** (runbook §5.1 ~5.87, **2.6× faster**), total wall 2.28 s for the 103-token round-trip. Memory still ~849 MiB Host + 222 MiB CPU_REPACK = 1071 MiB total (within IL-2). **FA attribution confirmed NOT the cause** (2026-04-27, same session): isolated sweep with `--flash-attn off` vs `--flash-attn on` on identical prompt; both modes returned the same ~14.6–15.3 tok/s decode (~65–68 ms/token) — `off` yielded 14.65 tok/s / 15 decoded tokens, `on` yielded 15.33 tok/s / 21 decoded tokens. FA is not a significant decode-speed factor on this model+context size. Root cause of the runbook delta is most likely **CPU governor state at the 2026-04-24 measurement time** — the board had been running Phase A Moonshine smoke workloads the same day; today's measurement was fresh-boot with A55 cores at full frequency. Same binary (`0adede8`), same model (unsloth Q4_0, `general.quantized_by = "Unsloth"`, sha256 `e479ea29…`, downloaded 2026-04-24 — not compiled by us; our own Phase 3 Q0 quantization step covers the fine-tuned weights). **Implication for H6**: runbook §5.1 perf table is stale; re-freeze at H6 as the canonical performance floor. All H3 gates met.
- ✅ **H4 `LlamaCompletionBenchAdapter`** authored 2026-04-26 — `tools/src/sl2619_tools/bench_prompt.py` + 22 new host tests (273 total, ruff + mypy strict green). Public surface:
  - `LlamaCompletionBenchAdapter(binary_path, model_path, n_threads=2, n_predict=128, temp=0.0, top_k=1, seed=42, subprocess_timeout_s=120.0, runner=…, clock_ns=…)` — subprocess shim with injectable runner for host tests
  - `wrap_gemma3_chat_template(user_text)` — byte-identical to `compose_prompt(candidate="gemma3", …)` (test cross-checked) so train/inference prompts cannot drift
  - `parse_completion_response(stdout)` — slice generated text after `<start_of_turn>model\n`, strip `[end of text]` / `<end_of_turn>` / mid-stream marker hallucinations
  - `parse_llama_perf(stream)` — extract load / prompt-eval / decode / total from `llama_perf_context_print:` block (scans stderr + stdout for fork robustness); raises `ValueError` if `load time` is missing so we never silently report 0 ms cold-load
  - `LlamaPerfReport`, `LlamaCompletionError` — typed perf snapshot + dedicated exception for nonzero exit / timeout / unparseable footer
  - `BenchAdapter` Protocol + `default_adapter_factory(args)` — `--adapter {gemma3_vendor,llama_completion}` dispatcher, vendor remains default for back-compat
  - `BenchRow` + `AdapterRunResult` extended additively: `pass_pattern`, `pattern_flags`, `passed_regex` (JSONL is now self-contained — no longer need to re-join against `prompts.yaml` to read pass/fail), `wall_ms_load` per-call (vendor still inherits sweep-level)
  - Lifted `score_response` + `compile_pattern_flags` from `bench_eval.py` into `bench_prompt.py` (no cycle: bench_eval already imports from bench_prompt)
- ✅ **H4 server-side HF tokenizer/auth smoke GREEN** (2026-04-27 12:19 UTC). After installing a fresh read-only HF token via `hf auth login --token … --add-to-git-credential` (account `lanhp-vn`), `~/server-bootstrap.sh --smoke-tokenizer` reports `RESULT: PASS — 18/0` including the new `PASS Gemma 3 tokenizer load` row. Server log: `/home/hoanglan/sl2619-finetune/bootstrap-20260427-121952.log`. Confirms HF gated-license acceptance was already in place under the same account; the prior 17/0 PASS just predated the tokenizer-smoke flag.
- ❌ **H5 logits-equivalence — PUNT (2026-04-26, investigation closed 2026-04-27)**. `llama-perplexity --kl-divergence` x86_64 AVX2 (WSL2 `build-native`) vs A55 NEON DOTPROD (`b8925`/`0adede8`): original run `Same top p = 98.622%` (gate ≥ 99.99%), `Max Δp = 9.393%` (gate ≤ 0.5%). **Investigation (2026-04-27)** ran two controlled experiments with matched references (x86_64 reference regenerated with same flags as board run to eliminate measurement-mismatch variables):
  - **Exp E** — board `--no-repack`, FA auto (resolved to enabled), vs x86_64 no-repack FA-auto reference: `Same top p = 99.213%`, `Max Δp = 7.765%`. Still PUNT. Bench: `docs/tmp/bench/2026-04-27_h5-logits-equiv-no-repack.md`.
  - **Exp F** — board `--no-repack --flash-attn off`, vs x86_64 no-repack FA-off reference: `Same top p = 98.425%`, `Max Δp = 26.142%`. Worse than Exp E. Bench: `docs/tmp/bench/2026-04-27_h5-logits-equiv-no-repack-fa-off.md`.
  - **OOM root cause (found 2026-04-27)**: prior Exp E/F runs were killed by the kernel OOM killer (`dmesg` confirmed 3× `llama-perplexit` kills at ~969 MiB RSS). Cause: three 254 MiB `.kld` reference files in tmpfs (`/tmp`) pinned 762 MiB in RAM (no swap), leaving insufficient headroom for the 514 MiB compute buffer. Fixed by moving `.kld` files to `/mnt/sdcard` (ext4, file-cache-evictable) before the runs.
  - **Findings**: (1) REPACK accounts for +0.6 pp same_top_p improvement (98.62% → 99.21%) but does not close the gap to 99.99%. (2) Disabling Flash Attention *worsens* agreement (99.21% → 98.43%), ruling out FA as the cause — FA actually improves cross-arch consistency. (3) The Δp distribution is bimodal: 95th percentile is 0.364% (below gate), but the 99.9th percentile is 7.353% — a small fraction of tokens exhibit extreme ISA-level FP divergence. (4) The residual divergence is structural: x86_64 AVX2 FMA accumulation order vs ARM64 NEON DOTPROD accumulation order for Q4_0 dequant + matmul produce systematically different results for certain activation distributions. No software flag available in this llama.cpp build eliminates this.
  - Board `system_info`: `NEON=1 | ARM_FMA=1 | FP16_VA=1 | DOTPROD=1 | LLAMAFILE=1 | REPACK=1`. Corpus: 35 prompts (15 yaml + 15 sft-test seed-42 + 5 OOD), 4 chunks at n_ctx=256. CLI: `h5-logits-equiv` at `tools/src/sl2619_tools/h5_logits_equiv.py` (21 host unit tests, ruff + mypy strict). Original bench: `docs/tmp/bench/2026-04-26_h5-logits-equivalence.md`.
  - **Conclusion**: H5 PUNT stands. No available flag combination produces ≥ 99.99% same_top_p for x86_64 AVX2 vs ARM64 NEON DOTPROD at `0adede8`. Per §4 fail action: **P3 path halted; do not proceed to H6 or fine-tune** until upstream confirms a fix (see `llama.cpp #22011`).
  - ⚠ **Gate calibration NOTE (2026-04-27, llama.cpp / ONNX research review)**: the 99.99% / 0.5% thresholds are stricter than what upstream's own perplexity README publishes for Q4_0-vs-FP16 — even **same-CPU BF16-vs-FP16** (just precision cast, no quantization) yields `same_top_p = 99.739%`, `max Δp = 4.186%`; q8_0-vs-FP16 yields ~98%, q4_K_M-vs-FP16 yields ~92-94%. The A55 measured 98.622% / 9.393% is consistent with normal Q4_0 noise, not a kernel bug as severe as #22011. The current H5 setup compares A55-Q4_0 against an x86_64-BF16 `.kld` reference, which conflates Q4_0 quantization noise (universal) with any A55-specific kernel divergence (the actual #22011 signal). Full diagnostic background in [`docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md §2.1`](../../tmp/analysis/2026-04-27_llama-onnx-plan-review.md). The verdict above is preserved verbatim — this NOTE only records that the original gate was measuring the wrong thing.
  - ✅ **Decision (2026-04-27 follow-up)**: H5 PUNT is **preserved as historical context** but **no longer the blocking gate**. The gate is now **H5R — same-quant cross-arch Δ test** (defined in §4 Phase 0 and summarized in the §0 status banner). H5R isolates A55-specific kernel divergence from universal Q4_0 quantization noise.
- ✅ **H5R logits-equivalence — GREEN (2026-04-27)**. Same-quant cross-arch Δ test ran end-to-end on the first attempt, both gates passed with substantial headroom:
  - **Setup**: BF16 reference `.kld` from `gemma-3-270m-it.bf16.gguf` (sha256 `903799d7…`, server-converted from `google/gemma-3-270m-it` HF safetensors via `convert_hf_to_gguf.py`). Same 35-prompt corpus as H5 (sha256 `71901c90…`). x86 KL run on WSL host using `.cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity` (version `0adede8 (b8925)` — byte-matched to the board); A55 KL run on board using `/mnt/sdcard/bin/llama-perplexity` at the same version. Both runs flagged `-c 256 --seed 1 --temp 0.0 --no-mmap`, `t=2` on board / `t=$(nproc)` on host. `.kld` lives on `/mnt/sdcard` (ext4) — the H5 follow-up tmpfs OOM root cause is avoided.
  - **Results**: `same_top_p_x86_q4_0 = 94.291%`, `same_top_p_a55_q4_0 = 93.898%` → **Δ_same_top_p = 0.393 pp** (gate ≤ 1.0 pp ✓ with 0.6 pp headroom). `max_delta_p_x86_q4_0 = 49.781%`, `max_delta_p_a55_q4_0 = 51.804%` → **ratio = 1.041x** (gate ≤ 3.0x ✓ with 1.96x headroom).
  - **Why this matters**: the residual cross-arch divergence is structural ISA-level FP arithmetic-order (x86_64 AVX2 FMA vs ARM64 NEON DOTPROD accumulation), **not** a #22011-class kernel bug. Per-chunk data is bidirectional — A55 actually beats x86 on chunk 1 (96.850% vs 96.063%); x86 leads on chunks 3–4 by < 0.4 pp. The 98.622% H5 PUNT was measuring universal Q4_0 quantization noise (then reference was Q4_0 — x86 vs A55 cross-arch directly); H5R substitutes a BF16 reference so x86 has its own non-trivial noise floor (94.291%) which the A55 stays inside.
  - **Threshold calibration**: the 1.0 pp Δ gate is conservative (matches plan §6.5 note "if x86 ≈ 94%, tighten Δ to 1 pp"); the 3.0x ratio gate is also conservative (actual 1.041x). Defaults retained — no override needed.
  - **Bench summary**: [`docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`](../../tmp/bench/2026-04-27_h5r-cross-arch-delta.md).
  - **Implication for fine-tune**: Phase 0 logits-equivalence gate satisfied; the fine-tuned Q4_0 will inherit the same cross-arch FP-order behavior, which Q1 (Phase 3) will re-verify against the merged BF16 to make sure the SFT delta isn't lost in Q4_0 noise.
- ✅ **H6 base-GGUF baseline bench — DONE 2026-04-27.** Un-fine-tuned base Q4_0 GGUF run through `LlamaCompletionBenchAdapter` over the full 15-prompt `prompts.yaml` sweep on the SL2619 A55 CPU. Bench summary: [`docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md`](../../tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md); JSONL: `docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.jsonl` (15 rows, post-hoc re-scored with the fixed parser).
  - **Real regex pass: 2/15 (13.3%).** Both passes coincidental: C1 calibration matches `.`, P1 matches `72` because the YAML echo contains `heart_rate_bpm: 72`. **Manual rubric: 0/3 on every prompt** — every response was a ```yaml fence echoing the patient record back, with no question-grounded content. D1 prefixed the echo with "Okay, I'm ready to answer the user's questions using only YAML…". This is the *definitional drift* `gemma-on-a55-get-started.md §3.7` warned about, now confirmed across the full corpus, not just probe1.
  - **Aggregate timing**: load 3241 ms (mean) / prompt-eval 15563 ms / decode 13367 ms / total wall 32171 ms per prompt × 15 = ~ 8 min full sweep. Aggregate decode 9.50 tok/s (1905 tokens / 200.5 s). All 15 runs hit `n_predict=128` cap (`tok=127` because EOS not counted) — the model never enters answer mode early. Slower than H3's 15.5 tok/s probe1 because (a) prompts ~10× larger (KV cache scales prompt-eval) and (b) decode budget 2× larger. H6 is the right baseline for bench-sized fine-tune evaluation; H3's number is correct for short-context smoke probes.
  - **Memory**: CmaFree dropped 461 → 396 MiB during each subprocess (~64 MiB per call, released between calls per the per-call mmap pattern). Host buffers ~ 1071 MiB per llama-completion (224 model + 111 KV + 514 compute + 222 REPACK) — within IL-2's 1.87 GiB envelope. No OOM, no swap, no thermal throttling.
  - **Two harness defects surfaced and fixed in-flight** (R2 cadence, single-chunk fix each, host pytest 100/100 green after both):
    - **§6.1 perf-block prefix**: upstream renamed `llama_perf_context_print:` → `common_perf_print:` between our local `665abc609` checkout and the b8925 board build. The harness's `_PERF_FIELD_RE` matched only the older form, so the first H6 attempt failed every prompt with `LlamaCompletionError: could not parse llama_perf footer`. Fix: regex now accepts either prefix; new fixture `_PERF_FIXTURE_B8925` holds the real captured stderr from the on-board `0adede8` binary.
    - **§6.2 chat-template detokenization**: `parse_completion_response` looked for the literal `<start_of_turn>model\n` divider, but llama-completion (without `--special`) detokenizes special tokens to empty so the on-the-wire divider is the bare `\nmodel\n` role label. The buggy fallback path returned the entire stripped stdout — including the echoed prompt — so `score_response` matched its regex against the YAML in the user-turn echo, producing a false 14/15 PASS. Fix: try both dividers (explicit form first, bare second), add `\nuser\n` as a terminator. Three new test fixtures exercise the b8925 detokenized shape. The on-disk JSONL was re-scored *post-hoc* with the fixed parser before this summary was written; future runs (Q4 in Phase 3) write self-consistent JSONL out of the gate.
  - **Implication for Phase 3 Q5**: the fine-tuned merged-Q4_0 must clear 2/15 real pass with substantial headroom — target ≥ 80% (12+/15) on the same suite, with manual rubric ≥ 2/3 on the YAML-grounded classes. Anything less indicates the SFT didn't break the IT-tuned model out of definitional drift.

**Runtime mismatches surfaced at H4** (host-tested; board confirmation comes at H6):

1. **Per-call mmap is real** — `llama-completion` cold-loads ~3.8 s every invocation. JSONL `timing.wall_ms_load` is now per-call for the llama path (was sweep-level for vendor). Advisor flagged this before commit; fixed by adding `wall_ms_load: float = 0.0` to `AdapterRunResult` and letting the adapter override the sweep-level fallback.
2. **`compose_user_text` body lacks chat-template markers by design** — vendor `Gemma3Static.run()` wraps internally; llama.cpp owns tokenization so the adapter wraps explicitly. The wrap matches `compose_prompt(candidate="gemma3", …)` byte-for-byte, asserted by `test_wrap_gemma3_chat_template_round_trips_compose_prompt` so train/inference cannot drift silently (the failure mode `16-slm-system-prompt.md §3` warns about).
3. **`-sysf --jinja` is bypassed** — gemma-on-a55-get-started.md §4.6 already documented that `--jinja` silently drops the system turn for Gemma 3 (no `system` role in template). The adapter feeds the entire wrapped turn via `-f promptfile`, never `-sysf`. Codified in code now, not only docs.
4. **`subprocess_timeout_s=120.0` default** — at the revised ~15 tok/s decode and `n_predict=128`, worst case is ~8.5 s decode + ~3.2 s mmap + ~0.85 s prompt eval ≈ 12.6 s. 120 s gives ~10× headroom; a hung subprocess becomes an error row (`LlamaCompletionError`), not a wedged sweep. The original 4× estimate used the stale 5.87 tok/s from the runbook; actual headroom is larger. Default unchanged — there is no reason to tighten it.
5. **Perf footer location (open assumption)** — parser scans `stderr + "\n" + stdout` to be robust to llama.cpp forks that mirror perf to stdout. Stock `b8925` puts it on stderr; the first H6 board run will confirm. If absent on stderr (which would surprise me), the parser still finds it on stdout.
6. **JSONL is now self-contained** — `pass_pattern`, `pattern_flags`, `passed_regex` are written per row at bench time; `bench_eval.py` re-computes from `prompts.yaml` for the Markdown rollup so the two scorers double-check each other.

### 10.3 Phase 2 SFT — DONE (2026-04-28)

**Phase 0 is fully closed (H1-H6 ✅, H5 PUNT preserved as history)** and **Phase 2 is fully closed (T0-T5 ✅; T5 closed with a P1-phrasing note that is a Phase 3 input, not a T5 blocker).**

#### T2/T3 training results (log: `~/sl2619-finetune/logs/train-20260428-064801.log`)

| Epoch | Steps | train_loss | eval_loss | eval_mean_token_accuracy |
|---|---|---|---|---|
| 1 | 64 | 1.326 | 0.9697 | 0.7613 |
| 2 | 128 | 0.7793 | 0.7983 | 0.7978 |
| 3 | 192 | 0.6277 | **0.6936** | **0.8152** |
| (aggregate) | 192 | **0.911** | — | — |

- Wall-clock: **326.4 s (5.4 min)**, RTX 5080 cu128, step rate 1.54 s/step
- Peak VRAM: ~10 GiB of 15.0 GiB free (after PDB=1 fix; see below)
- Checkpoints: `adapters_v1/checkpoint-{64,128,192}/` + final `adapters_v1/` (= checkpoint-192)
- Total adapter dir: **201 MB** (7.6 MB adapter_model.safetensors; rest is tokenizer + optimizer state)
- Adapter config: 7 target modules (q/k/v/o/gate/up/down_proj), r=16, lora_alpha=32, modules_to_save=null, ensure_weight_tying=false, peft 0.19.1
- Best-eval checkpoint: **checkpoint-192** (final epoch; T4 merge consumes `adapters_v1/` which equals it)

#### T3 gate verdict: **PASS with margin**

| Criterion | Gate | Measured | Pass? |
|---|---|---|---|
| eval_loss strictly decreasing ≥ 2 epochs | yes | 0.9697 → 0.7983 → 0.6936 (3/3) | ✅ |
| train_loss < eval_loss × 1.5 | yes | 0.6277 < 1.040 | ✅ |
| No OOM | yes | clean (after PDB fix) | ✅ |
| Checkpoints exist | yes | 3/3 epochs | ✅ |
| Masking-correct initial loss | ~1.5–3.0 (advisor) | 1.326 (slightly below band; converges cleanly) | ✅ acceptable |

#### §6 hyperparameter deviations (as-executed — full rationale in `tools/scripts/finetune.py`)

Two deviations from §6 were required and logged before/during T1-T2:

| Setting | §6 value | Actual | Root cause | Logged |
|---|---|---|---|---|
| `modules_to_save` | `["lm_head","embed_tokens"]` | **removed** | Gemma 3 `tie_word_embeddings=True` → peft splits tied pair into two 167M FP copies; peft warns this corrupts `merge_and_unload` → Q0 GGUF; plus catastrophic forgetting risk on 1023 examples | T1 dry-run |
| `per_device_train_batch_size` | 4 | **1** | `vocab_size=262,144` → logits at PDB=4/seq=1024/BF16 = 2.0 GiB + shift `.contiguous()` = another ~2 GiB; first attempt OOM at step 0 | T2 first attempt |
| `gradient_accumulation_steps` | 4 | **16** | Preserves effective batch=16 after PDB 4→1 | T2 first attempt |
| `per_device_eval_batch_size` | (default 8) | **1** | HF default independent of train PDB; at 8×seq×vocab×BF16 = 4 GiB logits alone → would OOM at epoch-end eval | T2 first attempt |

Effective optimization trajectory (batch=16) is unchanged. Pure LoRA on `all-linear` is the correct behavioral-fix surface for definitional drift; embedding layer retraining is unnecessary and counterproductive on 1023 examples.

#### T5 adjudication decisions (2026-04-28 — user)

Three caveats surfaced at T5 closure were adjudicated by the user before Phase 3 entry. **All three are decided. T5 stays `DONE-WITH-NOTE`. Phase 3 is authorized to start.**

| # | Caveat | Decision | Rationale |
|---|---|---|---|
| 1 | P3 brief-vs-YAML mismatch (verbal brief said P3 was a refusal probe; `prompts.yaml` defines P3 as `fact_lookup` "which medications do I take at 8am?" with pass `lisinopril\|metformin\|aspirin\|vitamin`) | **`tools/data/prompts.yaml` is the source of truth.** P3 was correctly scored as fact_lookup ✓; D1 was the actual refusal probe and passed ✓. Brief was stale. | Bench scoring runs against `prompts.yaml` programmatically; aligning verbal briefs to that file is the only consistent rule. |
| 2 | P1 literal phrasing ("what is my **current** heart rate?") emits `<eos>` first token on both base and merged models; merged answers `72.` cleanly for three rephrasings | **Accept as known v1 phrasing-sensitivity caveat.** Do NOT edit `prompts.yaml`. Do NOT retrain now. Proceed to Phase 3 with the literal P1 phrasing in place. | The merged model knows the heart-rate value under nearby phrasings; the dataset has a coverage gap for "current heart rate" style wording (1 of 1023 train rows mentions any heart-rate question). This is a dataset / backlog item, not a Phase-3 blocker. |
| 3 | T5 closure mode | **Stays `DONE-WITH-NOTE`.** Not reopened. Treat as a dataset / backlog item for v2 fine-tune. | The §10.3 plan criterion (`merged ≥ base; no obvious regressions`) PASSES with substantial headroom (4-prompt delta out of 5). |

**Important clarifying direction from user (2026-04-28):** **The current `tools/data/sft_v1.{train,val,test}.jsonl` corpus is a v1 proof-of-concept / test dataset, not final product-quality coverage.** Future fine-tuning passes (post-v1) **must** address the dataset gaps listed in `docs/plans/backlogs.md §1.21` before any product-quality claims. Phase 3 + Phase 4 of this plan freeze the *v1 demo* numbers — they are not a sign-off on dataset coverage.

#### What's left (as of 2026-04-28 post-Q5 closure)

Phase 3 is **fully closed** (Q0-Q5 ✅). Remaining engineering effort is **~ 0.5 day** through Phase 4 (freeze + handoff):

1. **Phase 3 Q0 + Q1 ✅ DONE 2026-04-28.** Merged BF16 + Q4_0 GGUFs on server (Q0); same-arch x86 Path B at n_ctx=2048 = 98.443% (≥ 95% gate, 3.4 pp headroom); cross-arch H5R-shape Δ = 0.393 pp / 0.996× (gates ≤ 1.0 pp / ≤ 3.0×). See benches `2026-04-28_…q1-logits-equivalence.md` + `2026-04-27_…q1-cross-arch-delta.md`.
2. **Phase 3 Q2-Q5 ✅ DONE-WITH-NOTE 2026-04-28.** Q2 sha verified on board (`587f1af6…`); Q3 smoke confirmed `'72 bpm.'` on the Path-B-shaped P1 (definitional drift fixed); Q4 full sweep via new R3-compliant host-driven `bench_remote.py` (8/15 regex pass, 5/15 grounded rubric ≥ 2, 17.29 tok/s decode); Q5 freeze at `2026-04-28_gemma3-finetuned-final.md`. Plan §9 ≥ 80% target NOT met but the v1 demo improvement over H6 (0/15 → 5/15 grounded) is substantial; v2 corpus expansion (multi-field, refusal-string anchoring, terminator-rich completions) is the path forward — itemized in `backlogs.md §1.21` and §10 of the final bench summary.
3. **Phase 4 — Freeze + handoff (~ 0.5 day, F1-F5)** — NOT yet authorized in this session per user direction. F1 final bench summary already written this session (`2026-04-28_gemma3-finetuned-final.md`); F2 model README update, F3 backlogs post-mortem entry, F4 `/doc_update`, F5 tag commit are pending explicit user authorization.

Next user-runnable: **explicit authorization to start Phase 4 (F1-F5)**. The numbers and source-of-truth artifacts are already on disk; F2-F5 are mechanical doc / tag steps once the user signs off on the v1 demo numbers.

---

## 11. References

### Primary external

- [Google — Own your AI: Fine-tune Gemma 3 270M and run it on-device](https://developers.googleblog.com/own-your-ai-fine-tune-gemma-3-270m-for-on-device/)
- [Google emoji-translator Colab notebook](https://colab.research.google.com/gist/lanhp-vn/814859563538ad2b371dc79ebd5840fc/fine_tune_gemma_3_270m_for_emoji_generation.ipynb) — canonical TRL+QLoRA recipe
- [Gemma QLoRA text-to-SQL guide (ai.google.dev)](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)
- [HF model card: google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
- [HF unsloth/gemma-3-270m-it-GGUF](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) — Q4_0 base GGUF used in the on-board runbook
- [llama.cpp #22011 — ARM64 wrong-logits bug 🚨](https://github.com/ggml-org/llama.cpp/issues/22011)
- [Gemma prompt structure — ai.google.dev](https://ai.google.dev/gemma/docs/core/prompt-structure) — confirms no `system` role

### Repo-local

- `docs/tmp/analysis/2026-04-24_gemma3-270m-practical-evaluation.md` — Phase B post-mortem + literature survey (11 sources)
- `docs/tmp/analysis/Fine‑Tuning Gemma 3 270M for Small On‑Device Task‑Specific Models.md` — best-practice synthesis (29 footnoted sources)
- `docs/get-started/gemma-on-a55-get-started.md` — the proven A55 CPU llama.cpp deployment runbook
- `docs/conventions/16-slm-system-prompt.md` — prompt-style normative rules
- `docs/conventions/15-model-compiler-runtime.md` — on-board runtime + storage layout
- `docs/conventions/11-testing-verification.md` — testing pyramid + R2 cadence
- `docs/plans/AI-models/models-testing-plan.md` — parent Phase 1.5 plan
- `models/gemma-3-270m-it/README.md` — per-model best-practice analysis
- `tools/data/{health_table_v1,prompts}.yaml` — fixtures (unchanged)
- `tools/data/clean_sft_dataset.json` — D0 raw pool (1400 Alpaca-shape triples, chatbot-distilled)
- `tools/data/sft_v1.{train,val,test}.jsonl` — **D2 Path B artifacts** (composed prompt, primary training shape)
- `tools/data/sft_v1_pathA.{train,val,test}.jsonl` — D2 Path A artifacts (raw pairs, ablation only)
- `tools/data/sft_v1.audit.jsonl` — D1 per-row routing provenance (1259 rows, 9 fields)
- `tools/src/sl2619_tools/sft_dataset.py` — **D1 implementation** (loader / dedupe / classifier / leakage scanner / splitter / emitter)
- `tools/src/sl2619_tools/sft_build.py` — D1g CLI entrypoint (`uv run sft-build`)
- `tools/tests/test_sft_{dataset,build}.py` — D3 unit + smoke tests (53 cases, ruff + mypy strict)
- `tools/src/sl2619_tools/{prompt_composer,bench_prompt,health_table}.py` — bench scaffolding (extended at H4)
- `references/HuggingFace/gemma-3-270m-it/` — pinned vendor-fork submodule (not the SFT base; SFT base is `google/gemma-3-270m-it`, downloaded fresh on the server)

---

*Authored 2026-04-25 — replaces prior placeholder content (which was the prompt-engineering plan misfiled at this path). Update the SD-card-mount convention in `15-model-compiler-runtime.md §5.3` at F4. The prompt-engineering G0 chunk (LlamaCompletionBenchAdapter) is folded into this plan as H4 — single owner now, no parallel planning.*

*Updated 2026-04-25 (later same day) — Phase 1 D1-D3 closed: `sft_dataset.py` + `sft_build.py` shipped with 53 unit tests, paraphrase-aware splitter with 5-reason force-routing + drain guard, Path B/A JSONL artifacts on disk. `max_length` calibrated to 1024 from the actual Path B token count. New OQ-FT-8 tracks the `medical_advice_reroute` classifier gap. Token-budget finding will land in `docs/conventions/15-model-compiler-runtime.md` if Phase 3 confirms the deployed shape.*

*Updated 2026-04-26 — Phase 0 H4 closed: `LlamaCompletionBenchAdapter` shipped in `tools/src/sl2619_tools/bench_prompt.py` (273 host tests, ruff + mypy strict). `BenchAdapter` Protocol + `default_adapter_factory` dispatcher; `--adapter {gemma3_vendor,llama_completion}` CLI flag. JSONL self-contained: `pass_pattern` / `pattern_flags` / `passed_regex` written per row. Per-call `wall_ms_load` honestly attributes the ~3.8 s mmap cost on the llama path. Score helpers `score_response` + `compile_pattern_flags` lifted from `bench_eval.py` into `bench_prompt.py` for cycle-free reuse. Six runtime mismatches surfaced (subprocess timeout, chat-template re-wrap, perf footer stream, etc.) — full list in §10.2.*

*Updated 2026-04-26 (later same day) — H2 server-bootstrap script shipped at `tools/scripts/server-bootstrap.sh` (idempotent, shellcheck-clean, 8 phases: detect → opt-in apt → venv → PyTorch cu128 → SFT stack → llama.cpp build → smoke checks → PASS/FAIL summary). §3.2 rewritten as a paste-able command sheet (scp → run → capture log → send back the trailing summary). Inline pip recipe deleted to enforce single-source-of-truth (the script). Troubleshooting table covers PyTorch CUDA, bitsandbytes, sm_120, HF auth, missing driver. Smoke tests honestly check what they claim — Python imports / `torch.cuda.is_available()` / sm_120 capability / bf16 matmul / `bitsandbytes.nn.Linear4bit` 4-bit forward pass / (gated) Gemma 3 tokenizer load.*

*Updated 2026-04-26 (end of day) — **H5 PUNT**. Logits-equivalence gate failed: `Same top p = 98.622%` (gate ≥ 99.99%), `Max Δp = 9.393%` (gate ≤ 0.5%). A55 NEON DOTPROD + REPACK path for Q4_0 does not match x86_64 AVX2 reference numerically. P3 fine-tune path halted at Phase 0 per §4 fail action. Investigation leads: REPACK kernel accuracy + Flash Attention kernel divergence. H6 blocked. Parser bug fixed: ETA-interleaved chunk-1 line now handled in `h5_logits_equiv.py`. Bench summary at `docs/tmp/bench/2026-04-26_h5-logits-equivalence.md`.*

*Updated 2026-04-27 (later) — **H5 investigation closed — PUNT stands**. Ran two controlled experiments with matched x86_64 references (no-REPACK FA-auto, no-REPACK FA-off). Exp E (`--no-repack`, matched ref): same_top_p=99.213%, Max Δp=7.765% — PUNT. Exp F (`--no-repack --flash-attn off`, matched ref): same_top_p=98.425%, Max Δp=26.142% — PUNT and worse. REPACK accounts for +0.6 pp improvement; FA-off makes divergence worse (rules FA out as cause). Residual divergence is structural ISA-level FP arithmetic (x86_64 AVX2 FMA vs ARM64 NEON DOTPROD accumulation order). No flag combination achieves ≥ 99.99% same_top_p. Root cause of prior OOM kills also identified: three 254 MiB `.kld` files in tmpfs pinned 762 MiB RAM (no swap) — fixed by relocating to sdcard. Bench summaries at `docs/tmp/bench/2026-04-27_h5-logits-equiv-no-repack.md` and `*-fa-off.md`. H5 PUNT stands; H6 remains blocked.*

*Updated 2026-04-27 — **H3 + H4-tokenizer closed**. H3: env-on-SD symlinks restored, `llama-cli` / `llama-completion` / `llama-bench` + `gemma-3-270m-it-Q4_0.gguf` + `probe1_prompt.txt` deployed to board; `--version` confirmed `(0adede8)`; deterministic probe ran clean at expected definitional-drift answer. Discovered real decode speed is **15.35 tok/s** (2.6× above runbook §5.1); FA attribution sweep (`--flash-attn off`/`on`) ruled out Flash Attention as cause — root cause is CPU governor/board state at 2026-04-24 measurement time. Runbook §5.1 table flagged as stale, to be re-frozen at H6. GGUF origin confirmed: pre-quantized by unsloth (not compiled by us; our own Q4_0 quantization is Phase 3 Q0 on the fine-tuned weights). H4-tokenizer: server bootstrap `--smoke-tokenizer` GREEN (`RESULT: PASS — 18/0`) after HF token install (account `lanhp-vn`, log `bootstrap-20260427-121952.log`). Timeout headroom in runtime-mismatch #4 updated from 4× to ~10× to reflect real decode speed. H5 focused-session plan written at `docs/plans/AI-models/a55-gemma-h5-logits-equivalence.md`.*

*Updated 2026-04-27 (later) — **llama.cpp / ONNX research review** (read-only static inspection of `references/llama.cpp@665abc609` + `references/onnx@086999d5d`). Three concrete edits: (a) §3.2 troubleshooting cell: corrected stale `transformers<5.0.0` claim — current pin is `transformers==5.5.1` since upstream `c8ac02fa1` (2026-04-09); skip-the-requirements-file resolution still correct due to `torch~=2.6.0` conflict. (b) §7 Risks H5 row + §10.2 H5 entry: added gate-calibration NOTE — the 99.99%/0.5% thresholds are stricter than upstream's published BF16-vs-FP16 same-CPU baseline (99.739%/4.186%); Q4_0-vs-FP16 cannot achieve them on any architecture. Existing PUNT preserved verbatim; user owns the redefine-or-stand decision. Diagnostic recipe (same-quant cross-arch Δ, BF16-vs-BF16 cross-arch, single-arch Q4_0-vs-FP16 calibration) at `docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md §2.1`. (c) §4 Deferred: added runtime `--lora FNAME` deployment as a Plan-B (LoRA tensors bypass REPACK per `src/llama-adapter.cpp:296+`).*

*Updated 2026-04-27 (cleanup) — **plan revision applied + submodule dirtiness resolved**. (a) **H5 redefined as H5R**: §0 status banner, §1.4, §4 Phase 0 table (H5 historical row preserved + new H5R row + H6 blocked-pending-H5R), §4 fail-action paragraph, §7 Risks H5 row, §9 Done Criteria (4), §10.2 H5 follow-up, §10.3 estimate now all reference H5R as the load-bearing gate. H5 PUNT chronology preserved verbatim; H5R uses same-quant cross-arch Δ (proposed `Δ_same_top_p ≤ 1.0 pp`, `max_delta_p_a55 / max_delta_p_x86 ≤ 3×`). (b) **Q1 redefined** in Phase 3 to apply the same calibration discipline (three-step: x86 BF16 ref → x86 Q4_0 noise floor → A55 Q4_0 vs same ref) so we don't repeat the H5 mis-calibration on the fine-tuned artifact. (c) **Runtime `--lora` Deferred entry tightened** — explicit Plan-B-only labelling, three concrete activation triggers. (d) **Submodule dirtiness resolved**: SL2619-specific orientation moved to project-owned `docs/references/llama-cpp.md` and `docs/references/onnx.md`; `references/{llama.cpp,onnx}/CLAUDE.md` reverted to upstream via `git checkout --`; `git submodule status` now clean (no `+`/`m`). Follow-up summary appended to `docs/tmp/analysis/2026-04-27_llama-onnx-plan-review.md §8`.*

*Updated 2026-04-27 (evening) — **H5R GREEN. Phase 0 closed.** Same-quant cross-arch Δ test ran end-to-end on first attempt: BF16 reference `.kld` from `gemma-3-270m-it.bf16.gguf` (sha256 `903799d7…`, server-converted from `google/gemma-3-270m-it`), x86 KL on WSL host using `0adede8 (b8925)` build-native binary, A55 KL on board using `/mnt/sdcard/bin/llama-perplexity` at the same version. Results: `same_top_p_x86_q4_0 = 94.291%`, `same_top_p_a55_q4_0 = 93.898%` → **Δ = 0.393 pp** (gate ≤ 1.0 pp ✓, 0.6 pp headroom); `max_delta_p` ratio **1.041x** (gate ≤ 3.0x ✓, 1.96x headroom). Per-chunk data is bidirectional (A55 *beats* x86 on chunk 1 at 96.850% vs 96.063%) — confirms residual is structural ISA-level FP-order, not a `#22011`-class kernel bug. The historical H5 PUNT (98.622% / 9.393% vs Q4_0 reference) was correctly attributed to universal Q4_0 quantization noise, not an A55 kernel issue; preserved verbatim as historical context. **H6 unblocks; fine-tune proceeds.** Bench: `docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`. Scorer: `tools/src/sl2619_tools/h5_logits_equiv.py classify-h5r` (55 host tests, ruff + mypy strict). Two concrete plan corrections found while executing: §3 / §6.1.1 / §6.4 of `docs/plans/AI-models/a55-gemma-h5-logits-equivalence.md` corrected (BF16 reference must be regenerated, not "(already exists)"; A55 must be re-run, not reused from H5 PUNT — different reference type) and §6.1.3 board path corrected (`/mnt/sdcard/bin/llama-perplexity`, not `/mnt/sdcard/llama-cpp/`). Server `llama-perplexity` not built — flagged in `docs/tmp/nouslogic-server-status.md`; x86 work moved to host (version-byte-matched to board). New host-collaboration memory saved: `feedback_board_probe_before_remote_commands` — always run `/board_probe` before composing remote commands.*

*Updated 2026-04-27 (late) — **H6 DONE. Phase 0 fully closed.** Un-fine-tuned base Q4_0 GGUF baseline frozen at 2/15 real regex pass / 0-rubric on every prompt — pure YAML-echo definitional drift. Aggregate 9.50 tok/s decode on bench-sized prompts (~745-820 input tokens, 127 decode budget). Bench: [`docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md`](../../tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.md). Two latent bench-harness defects surfaced and fixed in-flight: perf-block prefix renamed to `common_perf_print:` at b8925; chat-template special tokens detokenize to empty so the divider is bare `\nmodel\n`. Both fixed in `tools/src/sl2619_tools/bench_prompt.py` with new test fixtures (host pytest 100/100 green, ruff + mypy strict clean). On-disk JSONL re-scored post-hoc with the corrected parser. Next user-runnable step is **T0** (`scp tools/data/sft_v1.{train,val}.jsonl nouslogic-server:~/sl2619-finetune/data/`).*

*Updated 2026-04-27 (very late) — **T0 DONE; T1 script authored, dry-run pending.** Dataset uploaded to `nouslogic-server:~/sl2619-finetune/data/` after recovering from a malformed `scp` (no remote destination on the multi-source form silently overwrote local `val.jsonl` with `train.jsonl`); deterministic regen via `uv run sft-build` restored host hashes (train `6699ee41…`, val `b6443d7d…`), then per-file scp succeeded. Server-side line counts and sha256 verified to match host byte-for-byte. **trl 1.3.0 API drift surfaced and codified** in `tools/scripts/finetune.py`: `DataCollatorForCompletionOnlyLM` was removed in trl 1.x (probed on server); `max_seq_length` renamed to `max_length`; Gemma 3's chat template lacks `{% generation %}` markers so `assistant_only_loss=True` returns an all-zero mask (probed on server). Workaround: convert `messages` JSONL → prompt-completion shape inside `_to_prompt_completion` (user turn rendered with `add_generation_prompt=True`, completion = bare assistant text), then `SFTConfig(completion_only_loss=True)` for masking. Dry-run gate now also asserts BOS appears exactly once and `<start_of_turn>system` is absent in the rendered prompt. Host smoke test (stub tokenizer) confirms transformation produces `<bos>...<start_of_turn>model\n` ending and rejects system-role / missing-assistant rows. `/advisor` consulted before authoring; flagged the `max_length` rename and "decoded preview missing" gaps explicitly. Next user-runnable: `scp tools/scripts/{finetune,merge}.py nouslogic-server:~/sl2619-finetune/` + `ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python finetune.py --dry-run'`.*

*Updated 2026-04-27 (very very late) — **T1 dry-run gate PASS on first attempt; one §6 deviation logged.** Server-side dry-run via `ssh -t nouslogic-server` printed: `train rows: 1023 / val rows: 126`, `sample-0 prompt+completion: 930 (max_length=1024)`, decoded preview shows `<bos><start_of_turn>user\nROLE: health-records assistant…<start_of_turn>model\n` followed by `--- decoded completion --- 45.`, no system role anywhere, BOS appears exactly once, `model dtype: torch.bfloat16 device: cuda:0`, `T1 gate PASS`. **One concerning measurement**: `trainable params: 339,341,312 || all params: 607,439,488 || trainable%: 55.8642` — far higher than the ~6% target. Root cause: Gemma 3 has `tie_word_embeddings=True`; peft splits `modules_to_save=["lm_head","embed_tokens"]` into two independent FP-precision copies (~167M each), and the peft warning `tie_word_embeddings=True and a tied layer is part of the adapter, but ensure_weight_tying is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors` is load-bearing for Phase 3 — Q0's `convert_hf_to_gguf.py` would silently emit a corrupt vocabulary projection. **Deviation from §6 hyperparams** (logged here per §6 "deviations logged in F1 bench summary" — pre-emptively at T1 because the §6 value would Phase-3-block): `modules_to_save` removed for v1. Pure LoRA on `target_modules="all-linear"` is the correct surface area for a behavioral fix (definitional drift) on 1023 examples; the IT model's English health-term embeddings are already correct. Documented escalation if T5 side-by-side smoke shows no behavior change: reintroduce `modules_to_save=["embed_tokens"]` (one only) + `ensure_weight_tying=True`, NOT the original two-element list. Also fixed: `torch_dtype=` → `dtype=` (transformers deprecation). `/advisor` consulted before the deviation; agreed and contributed the F1-style logging guidance. Re-running dry-run after the edit is the next user-runnable; expected `trainable%` ≈ 1-2 % (~3-6M params).*

*Updated 2026-04-27 (final) — **T1 dry-run RE-RUN GREEN. Phase 2 ready for T2/T3.** Patched `finetune.py` re-deployed; second dry-run (`ssh -t nouslogic-server`) produced: same dataset / token-budget / decoded-preview / BOS-once / no-system-role gates, plus the corrected `trainable params: 3,796,992 || all params: 271,895,168 || trainable%: 1.3965`. peft tied-weight warning gone; `torch_dtype` deprecation gone; `logging_dir` deprecation remains (cosmetic — fix in F1 if it stops being a warning). Tied weights now preserved end-to-end into the eventual GGUF, unblocking Q0. T2/T3 (actual training) is the next user-runnable; awaiting explicit go-ahead per the standing "do not start training unless asked" rule.*

*Updated 2026-04-28 — **T2/T3 CLOSED. Phase 2 GREEN.** Two-attempt run: first attempt (`logs/train-20260428-064234.log`) OOM'd at step 0 with `Tried to allocate 3.66 GiB ... 11.72 GiB already in use` — root-caused to Gemma 3 270M's `vocab_size=262,144` (large vocab for a small model: §6's `per_device_train_batch_size=4` produced a 2 GiB logits tensor at seq=1024 / BF16, plus another ~2 GiB peak from `[..., :-1, :].contiguous()` in the SFT loss path; original §6 estimate of 1-1.5 GiB peak missed the vocab dimension entirely). `/advisor` consulted; agreed PDB=1 + GAS=16 is the narrow fix (effective batch unchanged, optimization trajectory preserved) and surfaced the eval-PDB blind spot — HF `TrainingArguments.per_device_eval_batch_size` defaults to 8 independently of train PDB, so eval-end at vocab=262144 would have OOM'd at 4 GiB just for the logits tensor after a full epoch of training. Three §6 deviations now codified in `tools/scripts/finetune.py::_build_sft_cfg` with full rationale + OOM trace numbers: `per_device_train_batch_size: 4 → 1`, `gradient_accumulation_steps: 4 → 16`, new `per_device_eval_batch_size=1`. Belt-and-suspenders: re-ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` per the OOM trace's own suggestion. Second attempt (`logs/train-20260428-064801.log`) clean: 192/192 steps in **5.4 min** (`train_runtime: 326.4s`). T3 gate **PASSES with margin**: `eval_loss = {0.9697, 0.7983, 0.6936}` strictly monotone-decreasing over 3 epochs; final `train=0.6277 < eval=0.6936 × 1.5 = 1.040` ✅ (and train < eval by ~10% — generalizing, not overfitting); no OOM; all three epoch-end checkpoints present at `~/sl2619-finetune/adapters_v1/checkpoint-{64,128,192}/` (7.6 MB adapter_model.safetensors each). **Bonus signals**: eval `mean_token_accuracy` climbs monotone 0.7613 → 0.7978 → 0.8152; `entropy` decreases 1.352 → 0.800 → 0.615 (model becomes more confident); `grad_norm` stable 4.2-6.2 (no exploding/vanishing). **Masking-correctness check**: initial `train_loss=1.326` (advisor flagged 1.5-3.0 as healthy band; 1.326 slightly below floor but loss converges cleanly — `completion_only_loss=True` masking is working on trl 1.3.0's prompt-completion path; if masking had failed by diluting across the full ~770-token user turn, loss would have been ~0.2 or lower). Adapter config confirms LoRA wiring: 7 target modules (q/k/v/o/gate/up/down_proj), `r=16`, `lora_alpha=32`, `modules_to_save=null` (T1 deviation took effect end-to-end), `ensure_weight_tying=false` (tied weights preserved → Q0 GGUF conversion path unblocked). Best-eval checkpoint = `checkpoint-192` (final epoch); top-level `adapters_v1/` saved by `trainer.save_model()` equals checkpoint-192 byte-for-byte → T4 merge consumes that directly. New memory recorded: `feedback_gemma3_vocab_dominates_oom` (avoid repeating the §6 PDB=4 estimate on any vocab>~50k model). **STOP — T4 merge / Phase 3 quantize not started per user direction.** Next user-runnable when authorized: `scp tools/scripts/merge.py nouslogic-server:~/sl2619-finetune/` (already byte-identical — sha `8ac201c98…`; no scp needed) + `ssh -t nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python merge.py'` to produce the merged BF16 HF dir for Q0.*

*Updated 2026-04-28 (T4) — **T4 CLOSED. Merged BF16 HF checkpoint produced.** Two-attempt run: first `python merge.py` (defaults) auto-picked **`checkpoint-64`** instead of `checkpoint-192` because `_resolve_adapter_path` used lexicographic `sorted(glob("checkpoint-*"))[-1]` — and `["checkpoint-128","checkpoint-192","checkpoint-64"]` lex-sorts last to `checkpoint-64` (the WORST eval-loss epoch, 0.9697). Patched `tools/scripts/merge.py` to sort by integer step (`int(basename.split("-",1)[1])`); host `pytest -q` 332 passed, ruff + mypy strict clean; scp'd updated `merge.py` to server (sha `813334a1…`). Re-ran with explicit `--adapters ./adapters_v1/checkpoint-192` (belt-and-suspenders even though fixed auto-pick now resolves to 192): wall **6.20s** on cuda:0, single-shard `model.safetensors` 536 MB, full tokenizer + chat template + generation_config saved. **Lightweight 1-prompt smoke** (`/tmp/merge_smoke.py`): merged_v1 loads as `Gemma3ForCausalLM` (vanilla HF, NOT PeftModel — confirms `merge_and_unload()` worked); for `"What is my heart rate? My YAML record says: heart_rate_bpm: 72"` (29 tokens) the model answered **`'You are currently running at 72 beats per minute.'`** (13 tokens, do_sample=False, 0.32s) — extracts the YAML value `72` instead of the H6 base baseline's `\`\`\`yaml`-echo definitional drift. Strong positive preview that the §4 directive-system + Path B SFT actually changed model behavior on the target task. `merge_v1/` files / shas: `model.safetensors` (536,223,056 B, sha `57c56472…`), `config.json` (1495 B, sha `c544327e…`), `tokenizer.json` (33,384,567 B, sha `daab2354…`). Log: `~/sl2619-finetune/logs/merge-20260428-071112.log`. **STOP — T5 side-by-side smoke / Phase 3 quantize not started per user direction.** Next user-runnable when authorized: T5 5-prompt manual eval base bf16 vs merged bf16 on `prompts.yaml` P1, P3, P6, D1, S1.*

*Updated 2026-04-28 (T5) — **T5 DONE-WITH-NOTE. Phase 2 fully closed.** Authored `tools/scripts/t5_smoke.py` (server-only, 12.3 KB, no `sl2619_tools` import → standalone) + bundle generator inline that pre-renders 5 prompts (P1, P3, P6, D1, S1) via `prompt_composer.render_system_prompt(now=date(2026,4,25))` to match the training-time prompt envelope verbatim. Bundle = `t5_smoke_bundle.json` (14,864 B sha `ee93caa9…`); host dry-run (`--dry-run`) confirmed JSONL+MD plumbing before deploy. Both files scp'd to server (sha-identical). Server run: `ssh -t nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python t5_smoke.py --bundle ./t5_smoke_bundle.json --base google/gemma-3-270m-it --merged ./merged_v1 --out-dir ./logs'` — both models loaded BF16 sequentially with `del + gc + cuda.empty_cache()` between, `do_sample=False`, `max_new_tokens=96`, `pad_token_id=eos`. **Result: regex `base 0/5` → `merged 4/5`, delta +4.** Merged completions (verbatim): P3 `"Lisinopril, Atorvastatin, Aspirin, Vitamin D3."`; P6 `"Penicillin."`; D1 `"I answer questions from your health record only."` (exact §4 directive refusal); S1 `"Your current medications include: Lisinopril, Metformin, Atorvastatin, Aspirin, and Vitamin D3."` (grounded summarization). Base produced YAML-echo definitional drift on P3/P6/D1/S1 (96 tokens of literal `\`\`\`yaml\\npatient: …`) — exactly the H6 failure mode. **P1 caveat (the only ✗):** both base and merged emit `<eos>` (id=1) as the FIRST new token for the literal phrasing "what is my **current** heart rate?" — `clean=''`, 1 new token. Read-only diagnostic on server (`docs/tmp/bench/t5-smoke-20260428-072748-p1-diagnostic.md`) confirms the merged model **does** know the value: rephrasings `"what is my heart rate?"`, `"what is my heart_rate_bpm value?"`, `"tell me my heart rate"` all yield `"72.<eos>"`. Training pool has exactly 1 row mentioning any heart-rate question (`"tell me my blood pressure and heart rate"`); the word "current" preceding "heart rate" is out-of-distribution → greedy first-token mass collapses to `<eos>`. Not a generation-config / quantization / chat-template bug. **Plan §10.3 criterion (`merged ≥ base; no regressions`) PASSES** (merged tied with base on P1's empty output; merged dramatically better on P3/P6/D1/S1). The brief's stricter criterion (`P1 must answer 72`) fails at literal phrasing only — recorded as a Phase 3 input, not a T5 blocker. Brief-vs-YAML mismatch surfaced for user adjudication: brief described P3 as "domain/social refusal" but `prompts.yaml` P3 is `fact_lookup` ("which medications do I take at 8am?", pass `lisinopril|metformin|aspirin|vitamin`); D1 was the actual refusal probe and it passed cleanly. T5 was scored against the YAML `class` field, not the brief description. Artifacts: `docs/tmp/bench/t5-smoke-20260428-072748.{jsonl,md}` (5-prompt run + markdown table) + `docs/tmp/bench/t5-smoke-20260428-072748-p1-diagnostic.md` (variant probe). Server-side logs: `~/sl2619-finetune/logs/t5-smoke-20260428-072748.{jsonl,md}` + `~/sl2619-finetune/logs/t5-smoke-run-*.log`. **STOP — Phase 3 Q0 (GGUF convert + Q4_0 quantize) not started per user direction.** Next user-runnable when authorized: P1 phrasing decision (accept / edit / augment) → `python ~/llama.cpp/convert_hf_to_gguf.py ./merged_v1 --outfile ./merged_v1.bf16.gguf` then `~/llama.cpp/build/bin/llama-quantize ./merged_v1.bf16.gguf ./merged_v1.q4_0.gguf Q4_0`.*

*Updated 2026-04-28 (Q0) — **Phase 3 Q0 GREEN.** Merged BF16 + Q4_0 GGUFs landed on server: `merged_v1.bf16.gguf` 518 MiB sha `a9c5100a4e88f2bf5526cc092d0fe6f2e08156096d9173bbd5351d1f0bb3665e`; `merged_v1.q4_0.gguf` 231 MiB sha `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4`. Convert wall 4.10 s; quantize wall 0.71 s; total Q0 wall 5.5 s on AMD Ryzen 7 9800X3D (CPU-only). Quant stats: 511.46 MiB BF16 → 224.00 MiB Q4_0 (7.01 BPW), 1 of 236 tensors fallback-quantized (`token_embd.weight` → F16; expected for Gemma 3's 262144-row vocab embedding — same behavior as the unsloth Q4_0 used at H6 baseline). llama.cpp HEAD `b1a5bd4 (CUDA: better coalesce data-access for contiguous concat #22330, 2 days old)`. **First attempt failed** at `convert_hf_to_gguf.py:1238` (`AssertionError`); diagnostic showed Gemma 3 has `len(tokenizer.vocab) = 262145` IDs but `config.json: vocab_size = 262144` so `max == vocab_size` violates the strict-less-than assertion. Both merged AND base Gemma 3 270M-IT tokenizers fail this assertion — it's an upstream issue in the BPE-fallback `_set_vocab_gpt2` path, not a `tokenizer.save_pretrained()` regression. Fix: pulled `tokenizer.model` (4.69 MB, sha `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`) from HF Hub via `huggingface_hub.hf_hub_download(repo_id="google/gemma-3-270m-it", filename="tokenizer.model", local_dir="./merged_v1")` — `Gemma3Model.set_vocab()` then takes the SentencePiece path which has no such assertion. Backlog added at `docs/plans/backlogs.md §1.22` for the permanent fix (merge.py should auto-pull tokenizer.model). `/advisor` consulted twice: pre-Q0 (capture wrapper improved with `time` + `ls -lh`; flagged that x86 work is host-side per H5R precedent because server has no `llama-perplexity`) and post-failure (diagnostic-first branching saved a wrong-fix loop). Server log: `~/sl2619-finetune/logs/q0-20260428-084616.log`. Next: **Phase 3 Q1** — three-step calibrated logits-equivalence; corpus from `sft_v1.test.jsonl` Path B (deployment shape, NOT bare user text), BF16 ref `.kld` + x86 Q4_0 KL on host, A55 Q4_0 KL on board.*

*Updated 2026-04-28 (T5 adjudication) — **Three T5 caveats adjudicated by user; Phase 3 authorized to start.** (1) **P3 brief-vs-YAML mismatch**: `tools/data/prompts.yaml` is the source of truth — P3 is `fact_lookup`, was correctly scored ✓; D1 was the refusal probe, passed ✓; brief description was stale. (2) **P1 literal-phrasing failure**: accepted as known v1 phrasing-sensitivity caveat — `prompts.yaml` is NOT edited, T1-T4 are NOT re-run; the model knows `72` under nearby phrasings and the literal-"current" failure is a 1-row training-coverage artifact, not a generation-config / SFT regression. (3) **T5 closure**: stays `DONE-WITH-NOTE`, not reopened. **Crucial framing**: the user direction is to treat the current `sft_v1` corpus as a **v1 proof-of-concept / test dataset**, not final product-quality coverage. Future fine-tuning passes MUST expand size + phrasing diversity before any product claims; the gaps are catalogued in `docs/plans/backlogs.md §1.21` (current-heart-rate / current-pulse / heart_rate_bpm / pulse-rate phrasings; single-field-fact-lookup paraphrase diversity; explicit medical-advice-reroute class tagging if reroute weakness shows up; P3/D1 benchmark-intent clarification anchored in YAML). §10.3 What's-left list collapsed: P1-phrasing decision item is removed (decided); only Phase 3 (Q0-Q5) and Phase 4 (F1-F5) remain.*

*Updated 2026-04-28 (Q2-Q5) — **Phase 3 fully closed (Q2-Q5 DONE-WITH-NOTE).** Q2 sha re-verified on board (`587f1af6…`) via /board_probe — model is at `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf`, 230 MiB, sha matches Q0 closure. Q3 smoke probe **revealed a deployment-envelope mismatch first**: the legacy `probe1_prompt.txt` uses an ad-hoc terse record schema not matching the §4 directive + Path B envelope the model was fine-tuned on; on it the FT'd Q4_0 model emits `model_user` hallucination (envelope drift, not Q4_0 quality regression). Re-probed with the deployment shape (`compose_user_text()` body via `--jinja --no-display-prompt -p $BODY`); model emits `'72 bpm.'` cleanly — definitional drift fixed at deployment shape. **Anecdote**: Q4_0 quantization perturbed the BF16-greedy `<eos>` mass for the literal "current heart rate" phrasing back into a recoverable state — closes the T5 P1 OOD-`<eos>` caveat at the deployment envelope. Q4 ran via a **new** host-driven `tools/src/sl2619_tools/bench_remote.py` (R3-compliant: SSH-piped llama-completion, no remote writes; built this session because the existing `bench_prompt.py.LlamaCompletionBenchAdapter` text-wraps with literal markers and llama.cpp without `--jinja` tokenizes those as plain bytes, defeating the FT delta — confirmed by Q3 garbage output). 15 unit tests on the new module (`tests/test_bench_remote.py`) green; ruff + mypy strict clean; total host pytest 288/288. JSONL + log: `docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.{jsonl,log}` (15 rows, 7m 48s wall). H6b 3-prompt sanity (P1/P3/D1) on **base** Q4_0 with `--jinja` envelope: 0/3 — confirms the H6 base failure is intrinsic (hallucinated YAML-shaped output regardless of envelope). Q5 numbers: **8/15 regex PASS** (+6 vs H6's 2/15), **5/15 manual rubric ≥ 2 grounded** (P1, P7, P9, A1, S1; +5 vs H6's 0/15). Decode **17.29 tok/s** aggregate (1.82× faster than H6's 9.50 tok/s — `--jinja` skips plain-wrap tokenization overhead). Plan §9 ≥ 80% target **NOT met** — quality ceiling is dominated by training-pool gaps (multi-field discrimination on P3/P4/P6, refusal-canonical-string drift on D1/D2, repetitive degeneration after correct first-answer token), NOT Q4_0 quantization noise (Q1 GREEN). v1 demo numbers frozen at [`docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md`](../../tmp/bench/2026-04-28_gemma3-finetuned-final.md); v2 corpus expansion path itemized in §10 of that bench + `backlogs.md §1.21`. **STOP — Phase 4 F1-F5 (freeze + handoff) NOT authorized in this session per user direction**; the F1 final bench summary is already written, F2-F5 (model README update, backlogs post-mortem, `/doc_update`, tag commit) await explicit user sign-off.*

*Updated 2026-04-28 (Q1) — **Q1 GREEN. Phase 3 Q1 fully closed; Q2-Q5 unblocked.** Calibrated three-step logits-equivalence cleared end-to-end. **Same-arch x86 Path B at n_ctx=2048**: `same_top_p_x86_q4_0_vs_bf16 = 98.443%` (gate ≥ 95% ✓, 3.4 pp headroom; apples-to-apples base anchor on same corpus = 99.489%, so SFT cost is only 1.05 pp — small, expected from SFT peakedness `entropy 1.352→0.615`). **Cross-arch Δ on H5R-shape corpus at n_ctx=256**: `Δ_same_top_p = 0.393 pp` (gate ≤ 1.0 pp ✓), `ratio_max_delta_p = 0.996x` (gate ≤ 3.0× ✓ — A55 actually fractionally under x86). The 0.393 pp number is **bit-identical to H5R's base-weight Δ** → the cross-arch kernel-noise floor is invariant to weight bit pattern; FT introduced no new ISA-specific behavior at the Q4_0 kernel level. **Step c reframe and board OOM**: first attempt at step c was Path B at n_ctx=2048 on board — kernel SIGKILL'd `llama-perplexity` twice (pids 1737, 1756; dmesg-confirmed total-vm 2.72 GB / anon-rss 1.73 GiB on a 1.87 GiB / no-swap board) because the per-chunk reference-logits buffer = `n_ctx × vocab=262144 × float32` = 2.15 GiB at n_ctx=2048 alone, exceeding physical RAM. n_ctx=256 fits in 1.20 GiB (model 934 MiB + per-chunk 268 MiB) — H5R-proven. Resolved by separating the two Q1 concerns: deployment-shape stability (same-arch Path B / n_ctx=2048 / x86-only) and cross-arch kernel parity (H5R-shape / n_ctx=256 / both archs). Both are reported; neither replaces the other. Full per-n_ctx memory math + dmesg evidence in `docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md §11`. **Artifacts on board**: `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` (sha `587f1af6…` matches host); `/mnt/sdcard/models/q1/merged_v1.bf16.h5.c256.kld` (sha `8e792450…`); `/mnt/sdcard/models/q1/h5_corpus.txt` (sha `71901c90…` matches H5R provenance); `/mnt/sdcard/bench/q1r-a55-q4_0.log`. Old OOM'd artifacts preserved as diagnostic-only: `/mnt/sdcard/models/q1/merged_v1.bf16.c2048.kld` (6.97 GB, do not reuse), `/mnt/sdcard/bench/q1-a55-q4_0.log` (truncated mid-warmup). **Working flag-semantics gotcha**: in this llama.cpp version, `--save-all-logits FNAME` and `--kl-divergence-base FNAME` are the SAME flag overloaded — presence of `--kl-divergence` flips the mode from SAVE to LOAD. To save the BF16 ref `.kld`, drop `--kl-divergence` and pass only `--save-all-logits FNAME`. Documented in the cross-arch bench §4 reproducible commands. **Benches**: same-arch + §11 OOM/reframe → `docs/tmp/bench/2026-04-28_gemma3-finetuned-q1-logits-equivalence.md`; cross-arch → `docs/tmp/bench/2026-04-27_gemma3-finetuned-q1-cross-arch-delta.md`. Scorer reused unmodified from H5R: `tools/src/sl2619_tools/h5_logits_equiv.py classify-h5r`. **STOP — Q3/Q4/Q5 not authorized in this session per user direction.** Next user-runnable when authorized: **Q3 board smoke probe** — Q4_0 GGUF was scp'd to `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/` during this Q1 step (Q2 effectively complete), needs only sha re-verify if fresh transfer wanted; Q3 command in `§4 Phase 3 Q3 row`.*

*Updated 2026-04-26 (end of day) — **H1 + H2 closed**. SSH alias `nouslogic-wsl` → `nouslogic-sl2619` (board) + `nouslogic-server` added (RTX 5080 fine-tune host); `/board_probe` extended with `--target=sl2619\|server` (READ-ONLY R3 applies to both). Server bootstrap finalized at `RESULT: PASS (17/0)` on `bootstrap-20260426-161055.log` after surfacing four real foot-guns and patching all of them in the script: (1) `--with-system-deps` requires `ssh -t` (sudo TTY); (2) cu128 `torch` wheel must be pinned via `~/sl2619-finetune/.torch-pin.txt` + `--extra-index-url` on every later pip call or downstream installs silently downgrade it to a CPU PyPI wheel; (3) `pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt` ResolutionImpossibles against cu128 torch 2.11 + transformers 5 — install just `gguf` directly; (4) bnb 4-bit smoke must use bare `quant_type=` / `compute_dtype=` for `Linear4bit` (the prefixed `bnb_4bit_*` names are `BitsAndBytesConfig`'s) — the wrong kwargs raised TypeError before any GPU code ran and masked a working bnb 0.49.2 + sm_120 path. Validated end-to-end on the actual GPU: `Linear4bit` 4-bit forward + `BitsAndBytesConfig` build both succeed. Live snapshot: `docs/tmp/nouslogic-server-status.md`; closure record: `backlogs.md §1.20`. Memories saved: `feedback_ssh_t_for_sudo`, `feedback_pin_torch_cuda_wheel`, `feedback_dont_install_llamacpp_convert_reqs`, `feedback_bnb_kwargs_two_styles`.*
