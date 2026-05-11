# CrispASR runtime spike — Phase 0 notes

> **Status:** NOT YET RUN (artifacts staged 2026-05-11).
> **Plan ref:** [plan.md §9 Phase 0](plan.md#phase-0--crispasr-runtime-spike-gate).
> **Owner:** Lan.
> **Result file (when run):** append `## Result — <date>` sections below;
> do not edit historical entries.

---

## 1. Objective

Decide whether to commit the dispenser-demo voice stack to the CrispASR runtime
(loading `cstr/moonshine-streaming-tiny-GGUF`) or to fall back to the proven
Moonshine Tiny float ONNX path documented in
[`docs/references/sl2619-moonshine.md`](../../references/sl2619-moonshine.md).
The fallback is the only safety net — there is no second alternative under
evaluation.

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
| Model artifact (HF) | `cstr/moonshine-streaming-tiny-GGUF` — registry entry `moonshine-streaming-tiny-q4_k.gguf` + `tokenizer.bin`, ~31 MB on disk | `src/crispasr_model_registry.cpp:103-105` |
| Model size | 34 M params, 6-layer enc/dec | upstream README |
| Encoder architecture | Sliding-window transformer (80 ms lookahead), streaming | upstream README |
| Audio frontend | Raw waveform → 80-sample frames → CMVN → asinh → linear+SiLU → 2× causal Conv1d. No mel spectrogram. | `src/moonshine_streaming.cpp:387` (comment block) |
| GGML ops used | Uses `ggml_flash_attn_ext` ×3; capabilities flagged `CAP_FLASH_ATTN \| CAP_TIMESTAMPS_CTC \| CAP_AUTO_DOWNLOAD \| CAP_DIARIZE` | `crispasr_backend_moonshine_streaming.cpp:22-23` |

> **Reminder:** the upstream README is the only authoritative source here.
> If the README is later moved or the binary CLI changes, refresh this section
> before the next spike run.

---

## 3. Bootstrap commands (host, one-time)

### 3.1 Build CrispASR for x86_64

CrispASR is also vendored at `docs/references/upstream/CrispASR/` in this repo
(submodule). Either clone fresh or build the submodule in-place:

```bash
# Option A — build the vendored submodule in-place (no extra clone):
cmake -S docs/references/upstream/CrispASR -B /tmp/crispasr-build \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/crispasr-build -j"$(nproc)" --target crispasr
# Resulting binary: /tmp/crispasr-build/bin/crispasr

# Option B — fresh clone (matches upstream docs/install.md):
git clone https://github.com/CrispStrobe/CrispASR /tmp/crispasr-src
cd /tmp/crispasr-src
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target crispasr
# Resulting binary: /tmp/crispasr-src/build/bin/crispasr
```

`--target crispasr` builds only the main CLI binary (skips
`crispasr-quantize` and `crispasr-diff`), per `docs/install.md`. Faster
build, no behavioral difference for the spike.

If GPU acceleration on the host is desired (not needed for the gate):
`-DGGML_CUDA=ON` / `-DGGML_METAL=ON` / `-DGGML_VULKAN=ON`. The Phase 0 gate
is CPU only on host to mirror board conditions.

### 3.2 Download the model + tokenizer

```bash
mkdir -p /tmp/moonshine-stream-tiny
huggingface-cli download cstr/moonshine-streaming-tiny-GGUF \
    --local-dir /tmp/moonshine-stream-tiny
```

The HF repo bundles both the model GGUF (e.g.
`moonshine-streaming-tiny-q4_k.gguf`) and the matching tokenizer artifact in a
co-located layout — CrispASR auto-discovers the tokenizer when both sit in the
same directory. If the download yields multiple quant variants, default to
`q4_k` (smallest viable; matches the board RAM budget).

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
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/bin /mnt/sdcard/models/moonshine-stream-tiny'
scp crispasr.aarch64 nouslogic-sl2619:/mnt/sdcard/bin/crispasr
scp /tmp/moonshine-stream-tiny/*.gguf nouslogic-sl2619:/mnt/sdcard/models/moonshine-stream-tiny/
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
    --model /tmp/moonshine-stream-tiny/moonshine-streaming-tiny-q4_k.gguf \
    --wav /tmp/spike-clip.wav \
    --expected "<a few words you know are in the clip>" \
    --latency-budget-s 1.0
```

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
    --model /mnt/sdcard/models/moonshine-stream-tiny/moonshine-streaming-tiny-q4_k.gguf \
    --wav /mnt/sdcard/fixtures/spike-clip.wav \
    --threads 2 \
    --latency-budget-s 2.0 \
    --rss-budget-mb 250
```

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
- Model: `moonshine-streaming-tiny-q4_k.gguf` (sha256: `<...>`)
- WAV: `<path>`, duration: `<s>`
- Decoded transcript: `<...>`
- Wall time: `<s>`
- Peak child RSS: `<kb>`
- Verdict: PASS / FAIL — `<reason>`

### Result — YYYY-MM-DD (board, step 0.2)

- Operator:
- `/board_probe` snapshot: `docs/tmp/sl2619-status.md` at `<git sha>`
- MemAvailable at start: `<MB>`
- /tmp tmpfs pre-state: clean / had `<file>`s removed by operator
- crispasr aarch64 sha256: `<...>`
- Wall time: `<s>`
- Peak RSS (VmRSS sampled): `<MB>`
- Verdict: PASS / FAIL — `<reason>`

---

## 7. Decision (step 0.3)

To be filled in after both 0.1 and 0.2 complete (or after one fails per the
fallback rule). Required fields:

- **Outcome:** KEEP CrispASR | FALLBACK to Moonshine ONNX | RE-SPIKE
- **Why:** one or two sentences pointing at the result rows above.
- **Phase 3 STT runtime:** `cstr/moonshine-streaming-tiny-GGUF via CrispASR`
  or `UsefulSensors/moonshine tiny float ONNX via onnxruntime`.
- **Mirror in `docs/plans/dispenser-demo/decisions-log.md`** under the
  Phase 0 entry once that file exists.

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
