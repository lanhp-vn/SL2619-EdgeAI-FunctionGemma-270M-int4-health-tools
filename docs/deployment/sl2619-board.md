# Gemma 3 270M-IT on the SL2619 A55 cores via llama.cpp

End-to-end runbook for getting the **unsloth Q4_0 GGUF** running on the SL2619 board's A55 cores via **llama.cpp** (cross-compiled against the Yocto SDK). End state: ~5.87 tok/s decode at INT4 / 270M params via `llama-completion`, plus interactive `llama-cli` for chat probing.

Verified working on **2026-04-24** against:

| Component | Pin |
|---|---|
| Board image | `scarthgap_6.12_v2.3.0`, kernel `6.12.62`, glibc 2.39, libstdc++.so.6.0.32 (GCC 13.3) |
| Yocto SDK | `/opt/poky/5.0.9/environment-setup-cortexa55-poky-linux` |
| `llama.cpp` tag | `b8925` (commit `0adede8`, released 2026-04-24) |
| GGUF | `unsloth/gemma-3-270m-it-GGUF` → `gemma-3-270m-it-Q4_0.gguf` (231 MB, sha256 `e479ea29…`) |
| HF auth | logged in via `hf auth login` |

> **Iron-Law fit.** Read-only SSH from the dev host (R3); no firmware touched (IL-8); under 1 GB RSS, 850 MiB host buffers + 222 MiB REPACK (within IL-2's 1.87 GiB envelope); no NPU traffic. Cores 2–3 are reserved for ATF / secure world (IL-11) — kernel only exposes cores 0–1 to Linux, which is the single most load-bearing fact in this whole runbook.

---

## 1. Prerequisites

On the **WSL2 host**:
- Yocto SDK installed at `/opt/poky/5.0.9/` (per `06-toolchain-build.md`).
- `git`, `cmake` ≥ 3.20, GNU `make`, `curl`, `tar`, `unzip`.
- `uv` available (`uvx --from huggingface_hub hf …`) **or** `pip install --user huggingface_hub`.
- HF account joined to `unsloth/gemma-3-270m-it-GGUF` — **ungated**, no license accept needed. (Google's `google/gemma-3-270m-it-qat-q4_0-gguf` repo *was* tested first; it returned `Repository not found` even after `hf auth login` succeeded — likely the exact name doesn't exist or is access-list-restricted. Use unsloth.)
- HF token written to `~/.cache/huggingface/token` (set up via `hf auth login`).

On the **board** (verify via `/board_probe`):
- `/mnt/sdcard` mounted at `/dev/mmcblk2p1` (manual mount, no fstab — see §3.1).
- `/usr/lib/libstdc++.so.6` present (it is on stock scarthgap; cap = `CXXABI_1.3.14`).
- 2 cores online: `cat /sys/devices/system/cpu/online` reports `0-1`.
- Board reachable as SSH alias `nouslogic-sl2619`.

---

## 2. What worked / what didn't

| Approach | Outcome |
|---|---|
| Prebuilt `llama-b8925-bin-ubuntu-arm64.tar.gz` from GitHub releases | **fail** — needs `CXXABI_1.3.15` (GCC 14+); board has GCC 13.3 |
| Cross-compile from source via Yocto SDK | **work** — links against `libstdc++.so.6.0.32` already on board |
| `cmake … -DLLAMA_BUILD_SERVER=OFF` | **fail** — `tools/CMakeLists.txt` gates `cli` subdir on `LLAMA_BUILD_SERVER=ON` (quirky coupling) |
| `cmake … -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF` | **work** — single binary, only system libs needed |
| `llama-cli … -no-cnv` for one-shot inference | **fail** — CLI is interactive-only in `b8925`; the binary prints `please use llama-completion instead` |
| `llama-completion … -t 4` (over-subscribed on 2 cores) | **fail** — 0.11 tok/s decode (scheduler thrash; 4 threads × 2 cores) |
| `llama-completion … -t 2` (matched to actual core count) | **work** — 5.87 tok/s decode, 12.9 s wall for 64-tok run |
| `-sysf sysprompt.txt --jinja` for system-grounded chat | **fail** — model says "I do not have the information" despite data in file. Gemma 3 has no `system` chat-template role; CLI prints `using custom system prompt` but content is silently dropped. **This is gate 0 of the prompt-engineering plan.** |
| Bare interactive (`-t 2 --temp 0.0 --top-k 1 --jinja`, no `-sysf`) | **work** — coherent general chat, 4.6–7.2 tok/s decode steady |

---

## 3. Step-by-step

### 3.1 Mount `/mnt/sdcard` (board, you run)

The Yocto image has no fstab entry for the SD card; every reboot drops the mount. From your WSL terminal:

```bash
ssh nouslogic-sl2619 'mountpoint -q /mnt/sdcard && echo MOUNTED || mount /dev/mmcblk2p1 /mnt/sdcard && df -h /mnt/sdcard && mkdir -p /mnt/sdcard/llama-cpp /mnt/sdcard/models/gemma-3-270m-it-q4_0 && ls -la /mnt/sdcard/'
```

Expected: `116.9G total, ≥ 100G free`, both new dirs present.

### 3.2 Download the GGUF (host)

```bash
cd /home/lanhp-wsl/nouslogic/SynapticSL2619 && mkdir -p .cache/llama-bench && cd .cache/llama-bench && uvx --from huggingface_hub hf download unsloth/gemma-3-270m-it-GGUF gemma-3-270m-it-Q4_0.gguf --local-dir . && ls -lh gemma-3-270m-it-Q4_0.gguf && sha256sum gemma-3-270m-it-Q4_0.gguf
```

Expected: `231M`, sha256 starts with `e479ea29…`. The download uses your HF token automatically.

> **Why unsloth and not Google's `qat-q4_0-gguf`.** Google publishes the *unquantized* QAT BF16 weights (which we have as a submodule under `references/HuggingFace/`), but the actual GGUF is community-converted. unsloth's conversion is the most cited and the Daily-Dose-of-DS Pi-5 tutorial uses it. If you need bit-deterministic reproducibility, switch the source and reset the SHA expectations in §1 — the rest of the runbook is unchanged.

### 3.3 Clone llama.cpp at the pinned tag (host)

```bash
cd /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench && git clone --depth 1 --branch b8925 https://github.com/ggml-org/llama.cpp.git && cd llama.cpp && git log -1 --format='%H %s'
```

Expected: HEAD = `0adede866ddb2e31992b3792eaea31d18ed89acf`.

### 3.4 Cross-compile against Yocto SDK (host)

```bash
cd /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/llama.cpp && bash -c 'source /opt/poky/5.0.9/environment-setup-cortexa55-poky-linux && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF && cmake --build build --target llama-cli llama-bench llama-completion -j$(nproc)'
```

Configure detects: `aarch64`, `DOTPROD` ✓, `FP16_VECTOR_ARITHMETIC` ✓, `FMA` ✓, **no SVE / no SME / no MATMUL_INT8** (correct — A55 baseline ARMv8.2-A). OpenMP not found, falls back to pthreads.

Build time on a modern WSL host: **~2 min** with `-j$(nproc)`.

Strip:

```bash
bash -c 'source /opt/poky/5.0.9/environment-setup-cortexa55-poky-linux && cd /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/llama.cpp/build/bin && aarch64-poky-linux-strip llama-cli llama-bench llama-completion && ls -lh llama-cli llama-bench llama-completion'
```

Expected stripped sizes:

| Binary | Size |
|---|---|
| `llama-cli` | 8.3 MB (interactive chat) |
| `llama-completion` | 6.6 MB (one-shot, takes `-no-cnv`) |
| `llama-bench` | 4.8 MB (synthetic micro-bench) |

`readelf -d llama-cli | grep NEEDED` confirms the only system deps are `libssl.so.3`, `libcrypto.so.3`, `libstdc++.so.6`, `libm`, `libgcc_s`, `libc`, `ld-linux-aarch64`. All present in `/usr/lib/` on the stock image.

### 3.5 scp binaries + GGUF + a probe prompt to the board (host)

A starter prompt file for §3.7 (one-shot fact retrieval, Gemma chat-template wrapped, deterministic):

```bash
cat > /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/probe1_prompt.txt <<'EOF'
<start_of_turn>user
You are a friendly assistant who answers questions using ONLY the user record below. If the answer is not in the record, say so. Do not invent values.

Record:
- name: Test Patient
- heart_rate_bpm: 72
- blood_pressure: 118/76

Question: What is my heart rate?<end_of_turn>
<start_of_turn>model
EOF
```

Then deploy as **two scp's** (each one a single physical line — multi-line `ssh '…'` bodies fragment in some terminals):

```bash
scp /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/llama.cpp/build/bin/llama-cli /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/llama.cpp/build/bin/llama-completion /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/llama.cpp/build/bin/llama-bench nouslogic-sl2619:/mnt/sdcard/llama-cpp/
```

```bash
scp /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/gemma-3-270m-it-Q4_0.gguf /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/probe1_prompt.txt nouslogic-sl2619:/mnt/sdcard/models/gemma-3-270m-it-q4_0/
```

### 3.6 Smoke (board, you run)

```bash
ssh nouslogic-sl2619 'cd /mnt/sdcard/llama-cpp && chmod +x llama-cli llama-completion llama-bench && ./llama-completion --version 2>&1 | head -n 6'
```

Expected:

```
version: 1 (0adede8)
built with GNU 13.3.0 for Linux aarch64
```

If you see `version `CXXABI_1.3.15' not found`, you ran the prebuilt; redo §3.4.

### 3.7 One-shot inference — Probe #1 (board, you run)

> **Use `llama-completion`, not `llama-cli`.** In `b8925`, `llama-cli` rejects `-no-cnv` and forces interactive mode. `llama-completion` is the headless one-shot.

> **Use `-t 2`, not `-t 4`.** The board's kernel exposes only 2 A55 cores. Asking for 4 threads on a 2-core machine collapses decode from 5.87 tok/s to 0.11 tok/s (a 53× regression — measured).

```bash
ssh nouslogic-sl2619 'cd /mnt/sdcard/llama-cpp && time ./llama-completion -m /mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf -f /mnt/sdcard/models/gemma-3-270m-it-q4_0/probe1_prompt.txt -t 2 -n 64 --temp 0.0 --top-k 1 -no-cnv'
```

Expected `common_perf_print` footer:

```
load time      = ~3800 ms
prompt eval    = ~37 tok/s   (82 tokens, ~2.2 s)
eval (decode)  = ~5.9 tok/s  (~21 tokens until [end of text])
total wall     = ~12-13 s
```

The model's actual answer in this run will be **`Okay, I understand. I will answer the question based solely on the information provided in the record.`** — *not* "72". This is **definitional drift**: the IT-tuned model treats the directive-style preamble as a request to acknowledge, not a question to answer. Solving this is **gate 0** of `docs/plans/a55-gemma-prompt-engineering.md`. Performance is fine; quality needs prompt-engineering iteration.

### 3.8 Interactive chat (board, you run)

```bash
ssh -t nouslogic-sl2619 'cd /mnt/sdcard/llama-cpp && ./llama-cli -m /mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf -t 2 --temp 0.0 --top-k 1 --jinja'
```

`ssh -t` allocates a pty so the interactive UI works. Inside, type `>` prompts. Per-turn perf prints under the answer (e.g. `[ Prompt: 22.4 t/s | Generation: 5.8 t/s ]`). Type `/exit` to leave.

Behaviorally the model is **coherent on social/general chat**, **wrong on basic math** (deterministically returned `23 + 47 = 60` twice in this session — INT4 + 270M ceiling), and **freely hallucinates on technical questions** ("a mailbox register is used to record the number of times a specific item is entered into a mailbox" — fiction). All matches the published 270M IFEval/factuality profile; nothing on this board is broken.

---

## 4. Pitfalls (the long version)

### 4.1 The prebuilt aarch64 release won't run

`llama-b8925-bin-ubuntu-arm64.tar.gz` is built on Ubuntu 24.04 (GCC 14+), needs `libstdc++.so.6.0.33` exporting `CXXABI_1.3.15`. The Yocto SDK ships `libstdc++.so.6.0.32` (`CXXABI_1.3.14`) and the board has the same. Symptom on first run:

```
./llama-cli: /usr/lib/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Yocto SDK has no newer libstdc++ to bundle; cross-compiling from source against the SDK is the *only* clean path on scarthgap.

### 4.2 `LLAMA_BUILD_SERVER=OFF` deletes `llama-cli`

`tools/CMakeLists.txt` couples both:

```cmake
if (LLAMA_BUILD_SERVER)
    add_subdirectory(cli)
    add_subdirectory(server)
endif()
```

If you turn the server off (reasonable — we don't need it), `llama-cli` goes with it and you'll see `gmake: *** No rule to make target 'llama-cli'`. Leave `LLAMA_BUILD_SERVER=ON`; the resulting `llama-server` binary is just a 9 MB extra file you don't have to ship.

### 4.3 `-t N > nproc` craters performance

The kernel exposes 2 cores; `-t 4` over-subscribes. Measured impact:

| Threads | Decode tok/s | Wall (64 tok) |
|---|---|---|
| `-t 4` | **0.11** | **3m 43s** |
| `-t 2` | **5.87** | **12.9 s** |

**53× regression** from oversubscription alone. Always `-t 2` on this board until the kernel/firmware exposes more cores. To verify, `cat /sys/devices/system/cpu/online` should print `0-1`.

### 4.4 Cores 2 and 3 are not yours

`/sys/devices/system/cpu/possible` and `…/present` both report `0-1`. The unused two A55 cores are reserved at the ATF / secure-world level for OP-TEE and Synaptics' own runtime (IL-11 territory; do not poke). If a future kernel/firmware variant exposes more, `-t` should be re-tuned then. For now, treat 2 cores as the immutable Linux-side ceiling.

### 4.5 `llama-cli --no-conversation` is gone

In `b8925`, `llama-cli` is interactive-only and rejects `-no-cnv`:

```
--no-conversation is not supported by llama-cli
please use llama-completion instead
```

Use `llama-completion` for headless / scripted runs. `llama-cli` for human chat. They share most flags.

### 4.6 `-sysf` does NOT inject system content for Gemma 3

`-sysf sysprompt.txt --jinja` prints `using custom system prompt`, but the model does not see the file's content. Confirmed by asking "what is my heart rate?" with `heart_rate_bpm: 72` in the file — model replied "I am sorry, I do not have the information." Gemma 3 has no `system` role in its chat template; the unsloth-published template (kv 37 in the GGUF) appears to drop or mis-route the system turn under llama.cpp's `--jinja` path. **This is unfixed and is gate 0 of the prompt-engineering plan.** Workarounds (any of):

1. Compose the entire turn as `-p "<system content>\n\n<user question>"` and skip `--jinja` (manual chat-template wrapping in your prompt file).
2. Compose user-turn content using the existing `prompt_composer.compose_user_text()` (already follows the directive-form rules in `docs/conventions/16-slm-system-prompt.md`).
3. Ship a `--chat-template-file` that explicitly maps `system` → user-prefix.

The plan exercises (1) first because it's zero new code. See `docs/plans/a55-gemma-prompt-engineering.md` G0.

### 4.7 BusyBox quirks on the board

| Won't work | Use instead |
|---|---|
| `head -40` | `head -n 40` |
| `ps -o pid,…` | `top -b -n 1` |
| `ip -br addr` | `ip addr show` + awk |
| `ldd llama-cli` | `readelf -d llama-cli \| grep NEEDED` (BusyBox has no `ldd`) |
| `grep -P` (PCRE) | `grep -E` (POSIX ERE) |

These are codified in `docs/conventions/10-code-style-shell.md` — when emitting commands for the board, default to long-form GNU flags and check the convention if unsure.

### 4.8 Single SSH sessions only for `ssh '…'`

A multi-line `ssh '…body…'` will fragment if any line in the heredoc exceeds the user's terminal soft-wrap (the wrap is rendered as a literal newline inside the single-quoted body, which BusyBox sh then splits at). Always one physical line. See `docs/conventions/10-code-style-shell.md §2.1`.

---

## 5. Performance baseline (reproducible)

### 5.1 One-shot via `llama-completion`, `-t 2`, `--temp 0 --top-k 1`, 64-tok cap

| Phase | Time | Rate |
|---|---|---|
| Cold model load (mmap from SD card + REPACK) | 3.76 s | — |
| Prompt eval (82 tokens) | 2.21 s | **37.2 tok/s** |
| Decode (21 tokens to `[end of text]`) | 3.57 s | **5.87 tok/s** |
| Total wall | **12.9 s** | — |

Run: `ssh nouslogic-sl2619 'cd /mnt/sdcard/llama-cpp && time ./llama-completion -m /mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf -f /mnt/sdcard/models/gemma-3-270m-it-q4_0/probe1_prompt.txt -t 2 -n 64 --temp 0.0 --top-k 1 -no-cnv'`

### 5.2 Interactive `llama-cli`, same flags + `--jinja`

12 free-form prompts; per-turn rates printed by the CLI:

| Prompt class | Decode tok/s |
|---|---|
| Greeting | 5.8 |
| Definition ("what is heart rate?") | 4.9 |
| Definition ("what is blood pressure?") | 7.2 |
| Identity ("what is your name?") | 4.7 |
| Math ("23 + 47?") | 5.3 |
| Hallucination-prone ("explain mailbox register") | 5.5 |
| Creative ("haiku about robots") | 5.8 |
| Knowledge ("what is Gemma 3?") | 5.9 |

Steady-state band: **4.6 – 7.2 tok/s decode**, **12 – 38 tok/s prompt eval**. No drift over the 12-turn session — no observable thermal throttling.

### 5.3 Memory footprint at run time (`llama_context: …` log lines)

| Allocation | MiB |
|---|---|
| `CPU_Mapped` (model mmap) | 224 |
| `CPU_REPACK` (NEON-repacked weights) | 222 |
| KV cache (non-SWA + SWA) | 111 |
| Compute buffer | 514 |
| **Total host buffers** | **~1071** |

Free during run: 164 MB. No swap (IL-2 green). Don't try to run a second model concurrently — the next allocation will OOM.

---

## 6. Files (board + host)

```
HOST  /home/lanhp-wsl/nouslogic/SynapticSL2619/.cache/llama-bench/
        ├── gemma-3-270m-it-Q4_0.gguf                 # 231 MB GGUF
        ├── probe1_prompt.txt                         # 315 B chat-template-wrapped Probe #1
        ├── llama.cpp/                                # cloned at b8925
        └── llama.cpp/build/bin/                      # cross-compiled, stripped
              ├── llama-cli                           # 8.3 MB
              ├── llama-completion                    # 6.6 MB
              └── llama-bench                         # 4.8 MB

BOARD /mnt/sdcard/llama-cpp/                          # ext4
        ├── llama-cli, llama-completion, llama-bench  # ours
        └── lib*.so* (residue from prebuilt try)      # safe to leave or rm
      /mnt/sdcard/models/gemma-3-270m-it-q4_0/
        ├── gemma-3-270m-it-Q4_0.gguf
        └── probe1_prompt.txt
```

---

## 7. Next step

Quality (Probe #1's "Okay, I understand" non-answer + interactive mode's 23+47=60 + fabricated definitions) is the unsolved half. Forward plan with gates G0–G6, R2 cadence, and 4-phase prompt iteration:

→ **`docs/plans/a55-gemma-prompt-engineering.md`**

Quality was ultimately solved by **fine-tuning, not prompt-only** — see §8 below. The original prompt-engineering plan saturated below G_QUALITY (1.2/3 avg) at Phase 1.5 Phase D. The fine-tuned Q4_0 GGUF clears the qualitative drift gate that the un-fine-tuned IT model could not.

---

## 8. Deployment shape for the fine-tuned Q4_0 — `--jinja --no-display-prompt` (2026-04-28)

After Phase 2 (QLoRA SFT on `google/gemma-3-270m-it`) and Phase 3 (Q4_0 GGUF + on-board bench), one practical gotcha overrode every other deployment knob: **the chat-template wrapping must come from llama.cpp's `--jinja` engine, not from text-level pre-wrap**.

### 8.1 The failure mode (what NOT to do)

The early bench harness (`tools/src/sl2619_tools/bench_prompt.py` `LlamaCompletionBenchAdapter`) text-wraps the user-turn body with literal `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n` markers and passes the wrapped string via `llama-completion -f promptfile`. This works for the un-fine-tuned base model (its definitional drift / YAML-echo failure mode dominates either way), but on the **fine-tuned** Q4_0 it produces hallucinated tail content:

```
$ # Path B body via plain wrap_gemma3_chat_template + -f (Q3b 2026-04-28):
$ ./llama-completion ... -f path-b-body-with-text-wrap.txt -no-cnv ...
... what is my heart rate?
<start_of_turn>model: 108<h4>You can also try</h4><h3>You can also try<h3>You can also try</h3>...
```

Root cause: llama.cpp without `--jinja` tokenizes `<start_of_turn>` etc. as plain bytes (~5–10 sub-tokens each) instead of the **special control tokens** (id 105 = `<start_of_turn>`, 106 = `<end_of_turn>`) the model was trained on. The fine-tuned model never sees the boundary it learned to enter answer mode at, so it falls back to whatever continuation is locally probable — typically nonsense.

### 8.2 The working envelope

```bash
# Render the Path B body locally (directive + YAML + user question), pipe to
# the board over SSH stdin, let llama.cpp wrap it via --jinja:
BODY="$(cd tools && uv run python -c "
from datetime import date
from pathlib import Path
import sys
from sl2619_tools.prompt_composer import compose_user_text
from sl2619_tools.health_table import load_health_table
ht = load_health_table(Path('data/health_table_v1.yaml'))
sys.stdout.write(compose_user_text(ht, date.today(), 'what is my heart rate?'))
")"
printf '%s' "$BODY" | ssh nouslogic-sl2619 'BODY=$(cat); /mnt/sdcard/llama-cpp/llama-completion \
    -m /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf \
    --jinja --no-display-prompt \
    -p "$BODY" \
    -t 2 -n 128 --temp 0.0 --top-k 1 --seed 42 \
    -no-cnv --single-turn 2>/dev/null'
# → "72 bpm.\nnot in record.\nnot in record. ..."
```

What each flag does:

| Flag | Effect |
|---|---|
| `--jinja` | Apply the model's `chat_template` metadata (embedded in the GGUF) → special tokens land at the right ids; the FT'd model enters answer mode reliably. |
| `--no-display-prompt` | Suppress the prompt echo on stdout → the parser doesn't have to find a divider; stdout is just the model reply. |
| `-p "$BODY"` | First user-turn content (Path B: directive + YAML + question, rendered via host `prompt_composer.compose_user_text`). |
| `-no-cnv --single-turn` | Headless one-shot — no interactive REPL; exit when generation completes. |
| `-t 2` | Use both A55 cores (cores 2-3 reserved for ATF / secure-world per `02-a55-application.md §4` IL-2). |
| `-n 128 --temp 0.0 --top-k 1 --seed 42` | Match the bench params (Q4 sweep) for reproducibility. |

Throughput on the board (measured 2026-04-28, FT'd Q4_0 over 15-prompt sweep):

| Metric | Value |
|---|---|
| Aggregate decode | **17.29 tok/s** (1.82× faster than the H6 base baseline's 9.50 tok/s — `--jinja` skips plain-byte tokenization overhead) |
| Per-prompt cold load (mmap + REPACK) | 3273 ms |
| Prompt-eval rate (~920-940 tok per Path B prompt) | ~62 tok/s |
| Per-prompt total wall (model-side) | 25.2 s (n_predict=128 cap) |
| Per-prompt total wall (incl SSH round-trip) | ~31 s |
| Memory | ~ 1071 MiB host RSS per `llama-completion` PID (within IL-2's 1.87 GiB) |

### 8.3 One-liner for ad-hoc testing — `tools/scripts/chat_remote.sh`

The above paste is wrapped as a committed script:

```bash
# From repo root, ssh-agent already authenticated to nouslogic-sl2619:
tools/scripts/chat_remote.sh "what is my heart rate?"
tools/scripts/chat_remote.sh "summarize my current medications"
tools/scripts/chat_remote.sh "what is my next appointment?"
echo "what am I allergic to?" | tools/scripts/chat_remote.sh    # stdin form

# Override generation params via env vars:
N_PREDICT=64 SEED=7 tools/scripts/chat_remote.sh "tell me a joke"
```

Source: `tools/scripts/chat_remote.sh`. Renders body locally via `prompt_composer.compose_user_text`, pipes to the board over SSH stdin, prints model reply only.

### 8.4 Full bench sweep — `bench_remote.py`

For a full 15-prompt suite run with JSONL output (the path that produced the Q4 numbers in `docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.jsonl`):

```bash
cd tools
uv run bench-remote \
    --ssh-host nouslogic-sl2619 \
    --prompts data/prompts.yaml \
    --health-table data/health_table_v1.yaml \
    --output ../docs/tmp/bench/$(date +%Y-%m-%d)_gemma3-finetuned-resweep.jsonl \
    --llama-binary /mnt/sdcard/llama-cpp/llama-completion \
    --llama-model /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf \
    --max-gen-tokens 128 --n-threads 2 --temp 0.0 --top-k 1 --seed 42 \
    --subprocess-timeout-s 180 --now $(date -I)
```

Source: `tools/src/sl2619_tools/bench_remote.py` (host-driven SSH-piped, R3-compliant — the agent never writes to the board). Score with `uv run bench-eval --jsonl <path> --prompts data/prompts.yaml`.

### 8.5 Known v1 limitations (deferred to v2 SFT pass)

The fine-tuned Q4_0 fixes definitional drift but the v1 corpus exposes three failure modes worth knowing before deploying:

1. **Repetitive degeneration**: under greedy / top-k=1 / temp=0, every prompt that doesn't terminate with a clear `<eos>` loops the phrase `not in record.` until the n_predict cap. Workaround at inference: relax to `--top-k 5 --temp 0.2`. Workaround at training (v2): include explicit `<end_of_turn>` after every completion in the SFT corpus.
2. **Multi-field discrimination weakness**: questions that need to filter the YAML by a sub-field (e.g. P3 "which medications do I take at 8am?", P6 "what am I allergic to?") fail. The 270M attention can't reliably bind a NL filter to a YAML key with the v1 dataset's coverage.
3. **Refusal canonical-string drift**: D1/D2 (off-topic) emit `not in record` or other refusal-shaped strings instead of the canonical `I answer questions from your health record only` from `16-slm-system-prompt.md §4` R-3. The 119 refusal-class rows in the v1 pool didn't establish a strong enough phrase prior.

Full v1 bench numbers + v2 backlog at [`docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md`](../tmp/bench/2026-04-28_gemma3-finetuned-final.md) and [`docs/plans/backlogs.md §1.21`](../plans/backlogs.md). The v1 demo numbers are 8/15 regex pass, 5/15 manual rubric ≥ 2 grounded; H6 base baseline was 2/15 regex (both spurious YAML-echo coincidences) and 0/15 grounded.

---

*Last verified: 2026-04-28. §1-§7 are unchanged from 2026-04-24 base-model bring-up; §8 adds the FT'd-Q4_0 deployment recipe. If `llama.cpp` tag, GGUF SHA, or board core count change, update §1 and §5 and re-run §3.7 to confirm the perf table still holds; for the FT'd model re-run the §8.4 bench sweep and update `docs/tmp/bench/<date>_gemma3-finetuned-final.md`.*
