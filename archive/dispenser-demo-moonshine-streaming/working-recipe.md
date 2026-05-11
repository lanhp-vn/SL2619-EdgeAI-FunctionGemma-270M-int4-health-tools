# Moonshine Streaming Tiny GGUF on SL2619 — frozen working recipe

Captured 2026-05-11 (PM) at the moment of supersession. This recipe was
verified end-to-end on real hardware before being superseded. Reproduce it
unmodified if you need the streaming variant back (e.g. for low-latency
partial-hypothesis emission in a future Phase 3.5 redesign).

Active path uses **moonshine-tiny** (non-streaming) — see
[`docs/plans/dispenser-demo/decisions-log.md`](../../docs/plans/dispenser-demo/decisions-log.md).

## Wire format identity

- HF repo: `cstr/moonshine-streaming-tiny-GGUF`
- Files used: `moonshine-streaming-tiny-q4_k.gguf` (~30.6 MB) + `tokenizer.bin` (~246 KB)
- CrispASR registry entry: `examples/cli/crispasr_model_registry.cpp:103-105`
- Backend dispatcher: `examples/cli/crispasr_backend_moonshine_streaming.cpp`
- Backend caps: `CAP_FLASH_ATTN | CAP_TIMESTAMPS_CTC | CAP_DIARIZE` (NO `CAP_PUNCTUATION_TOGGLE`)
- Sample rate: 16 kHz mono F32 (CLI decodes WAV/MP3/FLAC/OGG into F32 internally)
- Tokenizer auto-discovery: `dir_of(path_model) + "/tokenizer.bin"`
- Layers (q4_k): enc=6L×320, dec=6L×320, vocab=32768 (confirmed via on-board log)

## Critical runtime traps

Same as the active path — these apply to any CrispASR backend without
`CAP_PUNCTUATION_TOGGLE`:

1. **Auto-LID** — without `-l <code>`, first run fetches `ggml-tiny.bin`
   (~77 MB) from HF into `~/.cache/crispasr/`. Adds ~70 MB RSS / run.
   Suppress with `-l en`.
2. **Auto-punctuation** — without `--no-punctuation`, first run fetches
   `fireredpunc-q4_k.gguf` (~80 MB) and adds a second model pass (~3-4 s,
   ~60 MB RSS) post-decode. Suppress with `--no-punctuation` — the backend
   prints a cosmetic warning `"warning: backend 'moonshine-streaming' does
   not support --no-punctuation — ignoring"`; the flag IS effective at the
   dispatch layer via the `params.punctuation` policy check
   (`examples/cli/crispasr_punctuation_policy.h:11`). Behaviour is correct.

## Host build (x86_64) — same as active path

```bash
cmake -S docs/references/upstream/CrispASR -B /tmp/crispasr-build \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/crispasr-build -j"$(nproc)" --target crispasr-cli
# Binary: /tmp/crispasr-build/bin/crispasr
```

> **Target name gotcha**: `--target crispasr` (without `-cli`) builds only
> `libcrispasr.so`; `crispasr-cli` is the actual CLI target with `OUTPUT_NAME crispasr`.

## aarch64 cross-build (static, no OpenMP)

Use the toolchain file at `/tmp/crispasr-aarch64/toolchain-aarch64.cmake`
(pins `-mcpu=cortex-a55 -O3`; pure aarch64-linux-gnu, no Android NDK):

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(_a55_flags "-mcpu=cortex-a55 -O3")
set(CMAKE_C_FLAGS_INIT   "${_a55_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_a55_flags}")
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

Host packages required (Ubuntu 24.04, glibc 2.39 — matches SL2619):

```bash
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross
```

Configure + build:

```bash
cmake -S docs/references/upstream/CrispASR -B /tmp/crispasr-aarch64/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/tmp/crispasr-aarch64/toolchain-aarch64.cmake \
    -DGGML_OPENMP=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_EXAMPLES=OFF
cmake --build /tmp/crispasr-aarch64/build -j"$(nproc)" --target crispasr-cli
aarch64-linux-gnu-strip --strip-unneeded /tmp/crispasr-aarch64/build/bin/crispasr \
    -o /tmp/crispasr-aarch64/crispasr
```

> Both OpenMP knobs are required because CrispASR's top-level
> `find_package(OpenMP QUIET)` at `src/CMakeLists.txt:304` is *independent*
> of `-DGGML_OPENMP=OFF` (which only governs ggml's internal CPU backend).
> The SL2619 has no `libgomp.so.1` — without both knobs the binary will
> fail to start.

Expected ABI after strip:

- Size: ~7.9 MB
- NEEDED: `libstdc++.so.6 libm.so.6 libgcc_s.so.1 libc.so.6 ld-linux-aarch64.so.1` (no `libgomp`)
- GLIBC max: 2.38 (board has 2.39 — OK)
- GLIBCXX max: 3.4.32 (board libstdc++ 6.0.32 is GCC 14-era — OK)
- RUNPATH: empty

## Model download (operator-driven)

```bash
mkdir -p /tmp/moonshine-stream-tiny
source .venv/bin/activate  # `hf` CLI ships with huggingface_hub
hf download cstr/moonshine-streaming-tiny-GGUF \
    --local-dir /tmp/moonshine-stream-tiny
```

Expected files:

```
moonshine-streaming-tiny-q4_k.gguf   30.6 MB    sha256 46bf62ab1323da8ff3cf3936b62c08980590396a324bb822c91e38e821d972cc
moonshine-streaming-tiny.gguf       168.1 MB   (unquantized source — not needed for board)
tokenizer.bin                       246.0 KB    sha256 0e90e02b765a10f0fa35b7d67877df29dd22a1fd4890899c9b1b203a19bc8999
```

## Deploy to SL2619

```bash
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/bin /mnt/sdcard/models/moonshine-stream-tiny /mnt/sdcard/fixtures'
scp /tmp/crispasr-aarch64/crispasr                                  nouslogic-sl2619:/mnt/sdcard/bin/crispasr
scp /tmp/moonshine-stream-tiny/moonshine-streaming-tiny-q4_k.gguf   nouslogic-sl2619:/mnt/sdcard/models/moonshine-stream-tiny/
scp /tmp/moonshine-stream-tiny/tokenizer.bin                         nouslogic-sl2619:/mnt/sdcard/models/moonshine-stream-tiny/
scp docs/references/upstream/CrispASR/samples/jfk.wav                nouslogic-sl2619:/mnt/sdcard/fixtures/
ssh nouslogic-sl2619 'chmod +x /mnt/sdcard/bin/crispasr && sha256sum /mnt/sdcard/bin/crispasr /mnt/sdcard/models/moonshine-stream-tiny/* /mnt/sdcard/fixtures/jfk.wav'
```

## Smoke (host-side dispatcher)

The dispatcher script is at
`scripts/dispenser_demo/spike/crispasr_board_smoke.sh` (active codebase),
which defaults to `--backend moonshine` post-supersession. Override:

```bash
bash scripts/dispenser_demo/spike/crispasr_board_smoke.sh \
    --bin     /mnt/sdcard/bin/crispasr \
    --backend moonshine-streaming \
    --model   /mnt/sdcard/models/moonshine-stream-tiny/moonshine-streaming-tiny-q4_k.gguf \
    --wav     /mnt/sdcard/fixtures/jfk.wav \
    --latency-budget-s 10.0 \
    --rss-budget-mb    250 \
    --timeout-s        60
```

## Empirical result captured 2026-05-11 (board)

```
exit_code     : 0
elapsed_s     : 7.480  (11.0 s audio = 1.5× realtime)
peak_rss_mb   : 69.5
transcript    : "And so my fellow Americans ask not what your country can do for you ask what you can do for your country"
```

Extrapolated to plan §9 Phase 0 reference 3-s utterance: ~2.0 s wall, ~70 MB RSS.
At the latency gate, well under the RAM gate.

## Production launcher invocation (binding for this variant)

```
crispasr --backend moonshine-streaming -l en --no-punctuation -t 2 -m <model> -f <wav>
```

All four flags are required. `-t 2` matches the SL2619's two A55 cores.

## What this variant gives that moonshine-tiny doesn't

- **Sliding-window encoder** — produces intermediate encoder states usable
  for partial-hypothesis emission during ongoing speech.
- **Lower time-to-first-token** for long utterances.

If neither of those matter (dispenser-demo: push-to-talk or VAD-cut, complete
utterance handed to CrispASR in one go), prefer `moonshine-tiny` (active).
