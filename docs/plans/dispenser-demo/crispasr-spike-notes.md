# CrispASR runtime spike — Phase 0 notes

> **Status:** **CLOSED 2026-05-11.** Binding: CrispASR + `cstr/moonshine-tiny-GGUF`
> (non-streaming, `--backend moonshine`). Streaming variant provisionally pinned
> in the morning and superseded the same afternoon — frozen recipe at
> `archive/dispenser-demo-moonshine-streaming/`. See
> [`decisions-log.md`](decisions-log.md) for the supersession entry and §7 below
> for the original-vs-current decision blocks.
> **Plan ref:** [plan.md §9 Phase 0](plan.md#phase-0--crispasr-runtime-spike-closed-2026-05-11).
> **Owner:** Lan.
> **Result entries:** §6 is append-only; do not edit prior runs.

---

## 1. Objective

Decide whether to commit the dispenser-demo voice stack to the CrispASR runtime
(initially `cstr/moonshine-streaming-tiny-GGUF`, ultimately
`cstr/moonshine-tiny-GGUF` after the same-day comparison) or to fall back to
the proven Moonshine Tiny float ONNX path documented in
[`docs/references/sl2619-moonshine.md`](../../references/sl2619-moonshine.md).
The ONNX fallback remains the safety net for future regressions but was not
needed.

Pass criteria from the plan:

| Gate | Host (x86_64) | Board (SL2619 aarch64) |
| --- | --- | --- |
| Build | Compiles, links | aarch64 build present on board |
| Decode | Matches expected text on a known WAV | Same |
| Decode latency (3-s clip) | ≤ 1.0 s | ≤ 2.0 s |
| Peak RSS | (not gated) | ≤ 250 MB |
| Decision recorded | Yes, in this file | Same |

**Fallback rule (plan §9):** if 0.2 fails (build, OOM, latency > 5 s), drop
CrispASR and use Moonshine Tiny float ONNX. The fallback is fully documented
and was empirically validated 2026-04-23 (Phase A closure).

---

## 2. Upstream identification

| Item | Value | Source |
| --- | --- | --- |
| Runtime | CrispASR | https://github.com/CrispStrobe/CrispASR |
| Vendored submodule | `docs/references/upstream/CrispASR/` | this repo |
| Maintainer | CrispStrobe | GitHub |
| Style | whisper.cpp-style C++ runtime (24 ASR backends; moonshine-streaming is one) | upstream README |
| Runtime core | Vendored `ggml/` subdir; **independent of llama.cpp** — does not share the upstream `docs/references/upstream/llama.cpp/` submodule used by FunctionGemma. | `docs/references/upstream/CrispASR/CMakeLists.txt:186`, `ggml/` |
| Output binaries | `build/bin/crispasr`, `crispasr-quantize`, `crispasr-diff` | `docs/install.md` |
| Toolchain | C++17 compiler (GCC 10+ / Clang 12+) + CMake 3.14+; no Python at runtime | `docs/install.md` |
| Canonical invocation | `crispasr --backend moonshine-streaming -m MODEL.gguf -f AUDIO.wav` | `examples/cli/cli.cpp:364`, `crispasr_backend_moonshine_streaming.cpp:17` |
| Auto-detect | If `--backend` omitted, CrispASR detects from GGUF metadata; `-m auto` triggers HF download (needs curl/wget) | README quick-start; `crispasr_model_registry.cpp:103` |
| Threading flag | `-t N` / `--threads N`; default `min(4, hardware_concurrency())` — on the 2-core A55 SL2619 that resolves to 2 automatically | `examples/cli/whisper_params.h:20`, `cli.cpp:194` |
| Supported audio formats | `flac, mp3, ogg, wav` (the CLI decodes inside `read_audio_data()` before passing F32 PCM to the backend) | `examples/cli/cli.cpp:631`, `cli.cpp:1911` |
| Input sample rate | 16 kHz mono float32 (backend computes `n_samples / 16000.0`) | `examples/cli/crispasr_backend_moonshine_streaming.cpp:48` |
| Tokenizer | Separate `tokenizer.bin`, **auto-discovered from the model directory** via `dir_of(path_model) + "/tokenizer.bin"` | `src/moonshine_streaming.cpp:374` |
| Model artifact (HF, ACTIVE) | `cstr/moonshine-tiny-GGUF` — registry entry `moonshine-tiny-q4_k.gguf` + `tokenizer.bin`, ~20.2 MB on disk. Pinned 2026-05-11 (PM). | `src/crispasr_model_registry.cpp:97-99` |
| Model artifact (HF, archived) | `cstr/moonshine-streaming-tiny-GGUF` — `moonshine-streaming-tiny-q4_k.gguf` (~30.6 MB) + same `tokenizer.bin`. Superseded 2026-05-11 (PM); recipe at `archive/dispenser-demo-moonshine-streaming/working-recipe.md`. | `src/crispasr_model_registry.cpp:103-105` |
| Model size | 34 M params, 6-layer enc/dec | upstream README |
| Encoder architecture | Sliding-window transformer (80 ms lookahead), streaming | upstream README |
| Audio frontend | Raw waveform → 80-sample frames → CMVN → asinh → linear+SiLU → 2× causal Conv1d. No mel spectrogram. | `src/moonshine_streaming.cpp:387` (comment block) |
| GGML ops used | Uses `ggml_flash_attn_ext` ×3; capabilities flagged `CAP_FLASH_ATTN \| CAP_TIMESTAMPS_CTC \| CAP_AUTO_DOWNLOAD \| CAP_DIARIZE` | `crispasr_backend_moonshine_streaming.cpp:22-23` |
| CMake target for the CLI | **`crispasr-cli`** (`OUTPUT_NAME crispasr`). The `crispasr` target name produces only `libcrispasr.so` — passing `--target crispasr` will NOT yield a usable binary. | `examples/cli/CMakeLists.txt:12, set_target_properties(... OUTPUT_NAME crispasr)` |
| Auto-LID side effect | When `--language` is omitted or set to `auto`, CrispASR runs a whisper-tiny LID pass before the requested backend. **First run downloads `ggml-tiny.bin` (~77 MB) into `~/.cache/crispasr/`** (network required), and every run pays ~70 MB extra RSS for it. Disable with `-l en` (or any explicit language code). | empirical: see §6 host result; `cli.cpp` auto-detect path |
| Auto-punctuation side effect | For backends that emit unpunctuated text (moonshine-streaming, omniasr, kyutai-stt, etc.), CrispASR auto-enables a FireRedPunc post-pass. **First run downloads `fireredpunc-q4_k.gguf` (~80 MB) into `~/.cache/crispasr/`** (network required), and every run adds a second model load + decode pass (~3-4 s + ~60 MB RSS on Cortex-A55). Disable with `--no-punctuation`. The backend prints a cosmetic warning that the moonshine-streaming backend doesn't support a *native* `--no-punctuation` toggle, but the dispatch-layer auto-enable IS suppressed via `params.punctuation`. | `crispasr_punctuation_policy.h:11`, `crispasr_run.cpp:991`; empirical §6 board result |

> **Reminder:** the upstream README is the only authoritative source here.
> If the README is later moved or the binary CLI changes, refresh this section
> before the next spike run.
>
> **Critical for the board:** the SL2619 is offline by design. Always invoke
> crispasr with both `-l <code>` (e.g. `-l en`) AND `--no-punctuation` —
> neither auto-LID nor auto-punctuation should ever run on the board, both
> because the model downloads will fail without network and because together
> they consume ~150 MB / ~5 s of an envelope we cannot afford. The smoke
> scripts default both off; the production launcher (Phase 3.5) MUST do the
> same.

---

## 3. Bootstrap commands (host, one-time)

### 3.1 Build CrispASR for x86_64

CrispASR is also vendored at `docs/references/upstream/CrispASR/` in this repo
(submodule). Either clone fresh or build the submodule in-place:

```bash
# Option A — build the vendored submodule in-place (no extra clone):
cmake -S docs/references/upstream/CrispASR -B /tmp/crispasr-build \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/crispasr-build -j"$(nproc)" --target crispasr-cli
# Resulting binary: /tmp/crispasr-build/bin/crispasr

# Option B — fresh clone (matches upstream docs/install.md):
git clone https://github.com/CrispStrobe/CrispASR /tmp/crispasr-src
cd /tmp/crispasr-src
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target crispasr-cli
# Resulting binary: /tmp/crispasr-src/build/bin/crispasr
```

**Target name gotcha:** the user-facing binary is `crispasr` but the CMake
target is **`crispasr-cli`** (`OUTPUT_NAME crispasr` is set on it; see
`examples/cli/CMakeLists.txt:12`). The bare `crispasr` target builds only
`libcrispasr.so` — useful for downstream linkers, not for the spike. Phase 0
needs `--target crispasr-cli`.

If GPU acceleration on the host is desired (not needed for the gate):
`-DGGML_CUDA=ON` / `-DGGML_METAL=ON` / `-DGGML_VULKAN=ON`. The Phase 0 gate
is CPU only on host to mirror board conditions.

### 3.2 Download the model + tokenizer (human-executed)

> **Hand-off note:** the HF download is operator-driven, **not** agent-driven.
> The agent never runs `huggingface-cli` or network fetches as part of the
> spike; the operator stages the artifacts and hands back the local path.

```bash
# Operator runs (uses the new `hf` CLI; `huggingface-cli` is deprecated as of
# huggingface_hub >= 1.0):
mkdir -p /tmp/moonshine-tiny
source .venv/bin/activate    # `hf` ships with huggingface_hub already in [dev]
hf download cstr/moonshine-tiny-GGUF --local-dir /tmp/moonshine-tiny
# Then report back: the path to the chosen .gguf and confirm tokenizer.bin is co-located.
```

The repo contains multiple quant variants (q4_k, q8_0, plus the fp16 source)
and `tokenizer.bin`. For the Phase 0 gate use `moonshine-tiny-q4_k.gguf` —
the documented entry in `crispasr_model_registry.cpp:97-99` and the variant
proven on board (20.2 MB, 49.6 MB peak RSS).

The HF repo bundles both the model GGUF (e.g. `moonshine-tiny-q4_k.gguf`) and
the matching tokenizer artifact in a co-located layout — CrispASR auto-discovers
the tokenizer (`dir_of(path_model) + "/tokenizer.bin"`,
`moonshine_streaming.cpp:374` is the streaming-variant copy; the non-streaming
`moonshine.cpp` follows the same convention) when both sit in the same
directory. The `tokenizer.bin` is bit-identical between the two HF repos
(sha256 `0e90e02b...`).

> **Archived alternative:** for the now-superseded streaming variant, swap the
> repo name to `cstr/moonshine-streaming-tiny-GGUF` and the file to
> `moonshine-streaming-tiny-q4_k.gguf`. Full recipe at
> `archive/dispenser-demo-moonshine-streaming/working-recipe.md`.

### 3.3 Stage a test WAV

For the latency gate the clip must be ~3 s. A safe default:

```bash
ffmpeg -ss 0 -t 3 -i <some-speech.wav> -ar 16000 -ac 1 /tmp/spike-clip.wav
```

Or reuse `/mnt/sdcard/fixtures/say_hi.wav` if it's still on the board from the
Moonshine ONNX bootstrap (Phase A 2026-04-23).

### 3.4 Cross-compile / stage for the board

There is **no generic aarch64-Linux build script** in upstream, but
`docs/references/upstream/CrispASR/build-android.sh` cross-compiles the same
sources for `arm64-v8a` via the Android NDK toolchain file — a working proof
that the codebase supports the aarch64 ABI and the standard CMake
toolchain-file pattern. Options in priority order:

1. **Native build on an aarch64 host** (e.g. a Raspberry Pi 4/5 with the
   same toolchain) — simplest, no cross-compile setup.
2. **Cross-compile on the WSL2 host** with an Ubuntu 22.04 aarch64 sysroot
   and a standard CMake toolchain file. Use `build-android.sh` as a template
   for which CMake variables to set; ARM NEON SIMD paths in `ggml/` are the
   first place to look if the build fails.
3. **If both fail → Phase 0 fallback rule fires; switch to Moonshine ONNX.**

Stage the binary, GGUF + tokenizer to SD card:

```bash
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/bin /mnt/sdcard/models/moonshine-tiny /mnt/sdcard/fixtures'
scp crispasr.aarch64 nouslogic-sl2619:/mnt/sdcard/bin/crispasr
scp /tmp/moonshine-tiny/moonshine-tiny-q4_k.gguf nouslogic-sl2619:/mnt/sdcard/models/moonshine-tiny/
scp /tmp/moonshine-tiny/tokenizer.bin            nouslogic-sl2619:/mnt/sdcard/models/moonshine-tiny/
scp /tmp/spike-clip.wav nouslogic-sl2619:/mnt/sdcard/fixtures/spike-clip.wav
ssh nouslogic-sl2619 'chmod +x /mnt/sdcard/bin/crispasr; sync'
```

Everything lives on SD (ext4, persistent) — `/tmp/` is RAM-backed tmpfs and is
the wrong home for these artifacts.

---

## 4. Host smoke command (step 0.1)

```bash
uv run python scripts/dispenser_demo/spike/crispasr_host_smoke.py \
    --bin /tmp/crispasr-build/bin/crispasr \
    --model /tmp/moonshine-tiny/moonshine-tiny-q4_k.gguf \
    --wav /tmp/spike-clip.wav \
    --expected "<a few words you know are in the clip>" \
    --latency-budget-s 1.0
```

Default `--backend moonshine`. Pass `--backend moonshine-streaming` with a
matching streaming model GGUF if you need to re-test the archived variant.

`--threads N` is optional; omit to let CrispASR pick `min(4, nproc)`, or pass
`--threads $(nproc)` to pin explicitly for a known thread budget.

Exit 0 = PASS. Output reports model path, decode wall time, child peak RSS,
and the scraped transcript. If the transcript scrape is empty but exit code
is 0, re-run with `-v` to inspect raw stdout and update
`parse_transcript()` in the host script if upstream changed framing.

---

## 5. Board smoke command (step 0.2)

Pre-condition: `/board_probe` must have been run this session — the dispatcher
refuses to proceed otherwise. RAM safety: the dispatcher reports any large
file in `/tmp/` (RAM-backed tmpfs) and exits non-zero so the user can decide
whether to remove it (per Iron Law R3 the agent will not delete board state).

```bash
./scripts/dispenser_demo/spike/crispasr_board_smoke.sh \
    --bin /mnt/sdcard/bin/crispasr \
    --model /mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf \
    --wav /mnt/sdcard/fixtures/spike-clip.wav \
    --threads 2 \
    --latency-budget-s 2.0 \
    --rss-budget-mb 250
```

Default `--backend moonshine`, `--language en`, `--punctuation off`. Override
`--backend moonshine-streaming` + the streaming-tiny model path to re-test
the archived variant.

`--threads 2` matches the SL2619's two A55 cores. The CrispASR default
(`min(4, hardware_concurrency())`) resolves to 2 anyway, but pinning is
clearer in the spike record.

Exit codes:

| Exit | Meaning |
| --- | --- |
| 0 | PASS |
| 1 | Decode succeeded but missed a gate (latency / RSS) |
| 2 | Pre-flight refused (low RAM, polluted `/tmp/`, missing files) |
| 3 | `/board_probe` snapshot missing — refuse |
| 4 | Decode itself errored (non-zero rc, timeout) |

---

## 6. Result placeholders

When run, append a section like the one below. Do not edit prior runs — the
log is append-only so we can compare across attempts.

### Result — YYYY-MM-DD (host, step 0.1)

- Operator:
- CrispASR commit:
- Model: `moonshine-tiny-q4_k.gguf` (sha256: `<...>`) — active variant; for re-tests of the archived streaming variant substitute `moonshine-streaming-tiny-q4_k.gguf`.
- WAV: `<path>`, duration: `<s>`
- Decoded transcript: `<...>`
- Wall time: `<s>`
- Peak child RSS: `<kb>`
- Verdict: PASS / FAIL — `<reason>`

### Result — 2026-05-11 (host, step 0.1)

- Operator: Lan + agent (R3 override scoped to Phase 0)
- CrispASR commit: `docs/references/upstream/CrispASR/` submodule HEAD on 2026-05-11
  (libcrispasr.so.0.6.3 at link time)
- Build host: WSL2 Ubuntu, x86_64, GCC 11, `cmake --target crispasr-cli` Release
- Build dir: `/tmp/crispasr-build/`; binary: `/tmp/crispasr-build/bin/crispasr` (2.3 MB ELF)
- Model: `/tmp/moonshine-stream-tiny/moonshine-streaming-tiny-q4_k.gguf` (31 MB),
  `tokenizer.bin` (246 KB) co-located, auto-discovered
- WAV: `docs/references/upstream/CrispASR/samples/jfk.wav` (11.0 s, 16-bit PCM mono 16 kHz)
- Invocation (warm cache, `-l en`):
  `crispasr --backend moonshine-streaming -l en -m … -f …`
- Decoded transcript: `"ANd so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.."`
- Wall time: **1.100 s** (11.0 s audio → **10.0× realtime**; backend self-reports 1.00 s decode)
- Peak child RSS: **158628 KB ≈ 155 MB**
- Verdict: **PASS** — backend exits 0; expected substring `"ask not"` matches; far
  under the documented 1 s/3 s gate proportionally (1.10 s for 11 s = 0.30 RT-factor,
  i.e. for a 3 s clip ≈ 0.30 s wall, well under the 1 s gate).
- **Finding worth flagging (now codified in §2 and the smoke scripts):** the
  first invocation without `-l <code>` triggered CrispASR's auto-LID, which
  downloaded `ggml-tiny.bin` (~77 MB) into `~/.cache/crispasr/` and added
  ~70 MB to peak RSS (244 MB with auto-LID vs 155 MB with `-l en`). The host
  smoke script and the board dispatcher now default `--language en`. For the
  board this is non-negotiable: no network + 600 MB MemoryMax budget.

### Result — 2026-05-11 (board, moonshine non-streaming variant proof)

Parallel proof requested by the user: deploy and run `cstr/moonshine-tiny-GGUF`
(non-streaming) on board to verify the binary handles both backends. **Does
NOT change the §7 binding decision**; that remains KEEP CrispASR + moonshine-streaming-tiny.
The data below is captured for future-decision context.

- Operator: agent (R3 scoped override, same Phase 0 session)
- Model: `/tmp/moonshine-tiny/moonshine-tiny-q4_k.gguf` (~20.2 MB) +
  co-located `tokenizer.bin` (246 KB) — tokenizer sha256 IDENTICAL to the
  streaming variant (`0e90e02b...`)
- Model sha256: `333bb4a7df0c51da04fa2694fdc944936e75e79e57745c7ac3fd11f3176a8368`
- Board staging: `/mnt/sdcard/models/moonshine-tiny/{moonshine-tiny-q4_k.gguf,tokenizer.bin}`
- Same `/mnt/sdcard/bin/crispasr` binary (sha256 `5bfedc14...`) — no rebuild
- Invocation: `crispasr --backend moonshine -l en --no-punctuation -t 2 -m … -f /mnt/sdcard/fixtures/jfk.wav`
- Decoded transcript: `"and so my fellow americans ask not what your country can do for you ask what you can do for your country"`
  (lowercase — the non-streaming model normalizes case differently; convenient
  for downstream wordform normalization)
- Wall time: **4.66 s** (11.0 s audio → **2.4× realtime**; backend self-reports 4.55 s)
- Peak RSS (VmRSS, 50 ms polling): **50808 KB ≈ 49.6 MB**
- Verdict: **PASS** — exit 0, both gates met, transcript exact (semantically).

**Side-by-side with the streaming variant** (both on the same board, same WAV, same flags):

| Metric | moonshine-streaming-tiny (pinned in §7) | moonshine-tiny (this proof) | Delta |
|---|---|---|---|
| GGUF size (q4_k) | 30.6 MB | 20.2 MB | -34 % |
| Wall (11 s audio) | 7.48 s | **4.66 s** | **-38 %** |
| Realtime factor | 1.5× | **2.4×** | **+60 %** |
| Peak VmRSS | 69.5 MB | **49.6 MB** | -29 % |
| Transcript | mixed case | lowercase | normalization free for wordform layer |
| Exit | 0 | 0 | — |

**Interpretation (not a decision):** the non-streaming variant materially
outperforms the streaming variant for batch-decode of a complete utterance,
exactly as the architecture-level argument predicted (full-pass encoder vs
sliding-window bookkeeping; ~34 % fewer parameters; CrispASR's moonshine
backend has `CAP_PUNCTUATION_TOGGLE` so the punctuation auto-fetch policy is
gated natively — `--no-punctuation` is still honored). Extrapolated to a 3 s
command utterance: ~1.27 s wall, ~50 MB RSS — well inside plan §9 Phase 0
gates with comfortable headroom.

**Why this is a proof, not a decision flip:** the streaming variant was chosen
in part for its streaming-native architecture, which Phase 3.5 may need for
perceived latency (partial hypotheses while the user is still speaking). The
non-streaming variant cannot emit partial hypotheses. The latency win above
is for *batch* decode of a known-complete clip — Phase 3.5 may operate in
either mode depending on the wake-word vs VAD vs push-to-talk design.

**Action items for the user to consider (recorded here, not actioned):**

1. If Phase 3.5 settles on push-to-talk or VAD-cut-at-silence (batch decode
   of a complete utterance), open a fresh Phase-0-extension entry in
   `decisions-log.md` flipping the binding to `moonshine-tiny`.
2. If Phase 3.5 needs partial hypotheses (perceived-latency engineering),
   keep `moonshine-streaming-tiny` and accept the ~3 s extra wall.

### Result — YYYY-MM-DD (board, step 0.2)

- Operator:
- `/board_probe` snapshot: `docs/tmp/sl2619-status.md` at `<git sha>`
- MemAvailable at start: `<MB>`
- /tmp tmpfs pre-state: clean / had `<file>`s removed by operator
- crispasr aarch64 sha256: `<...>`
- Wall time: `<s>`
- Peak RSS (VmRSS sampled): `<MB>`
- Verdict: PASS / FAIL — `<reason>`

### Result — 2026-05-11 (board, step 0.2)

- Operator: Lan + agent (R3 scoped override for Phase 0)
- `/board_probe` snapshot: `docs/tmp/sl2619-status.md`, refreshed 2026-05-11
  (board libc 2.39, libstdc++ 6.0.32, Cortex-A55 ARMv8.2-A no I8MM/SVE, no
  on-board `libgomp`, 1.66 GiB MemAvailable, 109 GiB free on `/mnt/sdcard`)
- MemAvailable at start: 1705 MB
- /tmp tmpfs pre-state: clean after operator cleared 5 MB of P10S AEC probe WAVs
  pre-spike (`rm -f /tmp/*.wav`); only zero-byte systemd directories remained
- Cross-toolchain: Ubuntu 24.04 `gcc-aarch64-linux-gnu` 13.3.0,
  `libc6-dev-arm64-cross` 2.39 — exact glibc match with board
- Build flags (Release, static, no OpenMP):
  `-DCMAKE_TOOLCHAIN_FILE=… -DGGML_OPENMP=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE -DBUILD_SHARED_LIBS=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF`
  + toolchain file pinning `-mcpu=cortex-a55 -O3` and a flat `aarch64-linux-gnu` sysroot
- crispasr aarch64 sha256: `5bfedc148a665c56fe7a18fff857dfb4d9c8640695effaa30304e16bbb3304f8`
  (7.9 MB stripped, NEEDED = `libstdc++ libm libgcc_s libc ld-linux-aarch64`
  only; no `libgomp`, no RUNPATH, GLIBC ≤ 2.38, GLIBCXX ≤ 3.4.32)
- Model sha256: `46bf62ab1323da8ff3cf3936b62c08980590396a324bb822c91e38e821d972cc`;
  tokenizer sha256: `0e90e02b765a10f0fa35b7d67877df29dd22a1fd4890899c9b1b203a19bc8999`
- Invocation (warm cache, `-l en --no-punctuation`, 2 threads):
  `/mnt/sdcard/bin/crispasr --backend moonshine-streaming -l en --no-punctuation -t 2 -m … -f /mnt/sdcard/fixtures/jfk.wav`
- Decoded transcript: `"And so my fellow Americans ask not what your country can do for you ask what you can do for your country"`
- Wall time: **7.48 s** (11.0 s audio → **1.5× realtime**; backend self-reports 7.31 s decode)
- Peak RSS (VmRSS, 50 ms polling): **71132 KB ≈ 69.5 MB** — well under 250 MB
- Verdict: **PASS** — extrapolated to a 3 s utterance the gate (≤ 2.0 s wall,
  ≤ 250 MB RSS) is met: 1.5× RT × 3 s ≈ 2.0 s wall, ~70 MB RSS.
- **Findings worth flagging** (now codified in §2 and the smoke scripts):
  1. Bare `--target crispasr` builds only `libcrispasr.so`; the CLI target is
     **`crispasr-cli`** (`OUTPUT_NAME crispasr`).
  2. Auto-LID downloads `ggml-tiny.bin` (~77 MB) → suppress with `-l en`.
  3. **Auto-punctuation** downloads `fireredpunc-q4_k.gguf` (~80 MB) and adds
     a second decode pass → suppress with `--no-punctuation`. Without it, the
     first board run wall-clock was 11.20 s and RSS 132 MB (logged before
     re-spike with the flag) — punctuation is responsible for ~3.7 s and
     ~60 MB on this hardware.
  4. **BusyBox `date +%s%N`** prints the literal token `%N` instead of
     nanoseconds; the dispatcher now reads `/proc/uptime` (float seconds,
     0.01 s resolution, kernel-stable since 2.0). First run lost the wall
     measurement to this bug — substantive decode still completed.
  5. The board ran with `--no-punctuation`; backend emitted the cosmetic
     warning `"warning: backend 'moonshine-streaming' does not support
     --no-punctuation — ignoring"`. The flag IS effective at the dispatch
     layer (auto-enable check on `params.punctuation`); the warning refers to
     a backend-native toggle that doesn't exist. Behavior is correct.
  6. Build artifact ABI: GLIBC ≤ 2.38 (board 2.39), GLIBCXX ≤ 3.4.32 (board
     libstdc++ 6.0.32 = GCC 14-era). Forward-compatible with the board.

---

## 7. Decision (step 0.3)

> **Decision update — 2026-05-11 (PM).** The original decision below (KEEP
> CrispASR + `moonshine-streaming-tiny`) was superseded the same afternoon
> by the proof in §6 row "moonshine non-streaming variant proof". The
> current binding is **KEEP CrispASR + `cstr/moonshine-tiny-GGUF` (non-streaming
> `--backend moonshine`)**. -38 % wall, -29 % RSS, -34 % model size on the
> same board, same fixture, same flags. Streaming-variant recipe preserved
> in `archive/dispenser-demo-moonshine-streaming/`. See
> `docs/plans/dispenser-demo/decisions-log.md` for the supersession entry.
>
> The text below is preserved as the original 2026-05-11 (AM) reasoning.

### Original decision (2026-05-11 AM — superseded)

- **Outcome:** **KEEP CrispASR** (resolved 2026-05-11).
- **Why:** both gates met — host §6 row 1 (1.10 s wall, 155 MB RSS, exact
  transcript) and board §6 row 2 (7.48 s wall for 11 s audio = 1.5× RT,
  69.5 MB RSS, exact transcript). Extrapolated to the plan's reference
  3 s utterance the board hits ~2.0 s wall and ~70 MB RSS — at the latency
  gate but well under the RAM gate. Two runtime traps (`auto-LID`,
  `auto-punctuation`) discovered and pinned off in both smoke scripts and
  the spike-notes recipe; production launcher (Phase 3.5) must do the same.
- **Phase 3 STT runtime:** `cstr/moonshine-streaming-tiny-GGUF` via
  CrispASR (static aarch64 build, `-DGGML_OPENMP=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE -DBUILD_SHARED_LIBS=OFF`).
  Production invocation: `crispasr --backend moonshine-streaming -l en
  --no-punctuation -t 2 -m <model> -f <wav>`.
- **Mirror:** entry added to `docs/plans/dispenser-demo/decisions-log.md`.
- **Latency caveat for Phase 3.5 planning:** the board ASR sits right on
  the latency gate, so the end-to-end pipeline must avoid serial second
  passes (the punctuation pass we suppressed cost the same as the
  decode itself). Streaming partial hypotheses from moonshine-streaming
  may be needed to keep perceived latency reasonable.

---

## 8. Open questions

- **Q1 — aarch64 build path.** Upstream provides no cross-compile recipe.
  First attempt: native build on a Pi-class aarch64 box. If unavailable, try
  cross-compile from WSL2 with an Ubuntu 22.04 aarch64 sysroot. If both fail
  in <1 working day → invoke fallback per plan §9 ("Phase 0 fallback rule").
- **Q2 — tokenizer artifact name.** **RESOLVED 2026-05-11** during vendor
  source review. The file is literally `tokenizer.bin`, co-located with the
  model GGUF. Loader at `docs/references/upstream/CrispASR/src/moonshine_streaming.cpp:374`:
  `std::string tok_path = dir_of(path_model) + "/tokenizer.bin";`. The
  `cstr/moonshine-streaming-tiny-GGUF` HF repo ships both files; the registry
  entry at `src/crispasr_model_registry.cpp:103-105` confirms.
- **Q3 — transcript framing.** Partially resolved: CrispASR uses whisper.cpp
  segment-line framing (`[hh:mm:ss --> hh:mm:ss]  text`). The host script's
  `_LOG_PREFIX_RE` strips bracketed prefixes, so the longest surviving line
  should be the transcript text. The first real run will confirm and may
  prompt a regex refinement.
- **Q4 — Phase 0 still uses the legacy fixture (`say_hi.wav`)?** Reusing the
  Moonshine ONNX fixture lets us compare like-for-like across both runtimes;
  preferred over a new clip unless a streaming-specific test is needed.
- **Q5 — board-side RSS for moonshine-streaming-tiny under ggml CPU.** No
  vendor benchmark exists for an aarch64 Cortex-A55 target. The 250 MB plan
  gate is a best-guess; the spike itself will produce the measurement that
  validates or invalidates it.

---

## 9. Pointers

- [`plan.md`](plan.md) §9 Phase 0 — gate criteria, fallback rule.
- [`docs/references/sl2619-moonshine.md`](../../references/sl2619-moonshine.md) — the documented fallback path.
- [`docs/guides/usb-audio-testing-sl2619.md`](../../guides/usb-audio-testing-sl2619.md) — the WAV-capture recipe for fresh test clips.
- [`scripts/dispenser_demo/spike/crispasr_host_smoke.py`](../../../scripts/dispenser_demo/spike/crispasr_host_smoke.py) — host runner.
- [`scripts/dispenser_demo/spike/crispasr_board_smoke.sh`](../../../scripts/dispenser_demo/spike/crispasr_board_smoke.sh) — board dispatcher.
- `.claude/CLAUDE.local.md` — R3 (SSH read-only) and the rule against agent-driven board mutation.
