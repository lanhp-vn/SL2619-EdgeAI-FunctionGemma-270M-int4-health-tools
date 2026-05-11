# SL2619 Moonshine Tiny — Working Recipe

> Empirically verified 2026-04-23 (Phase A closure, G_DMIC GREEN).
> Ground truth: `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §4`,
> `SynapticSL2619/docs/plans/backlogs.md §1.17`.

---

## 1. Model choice rationale

| Variant | ONNX weight size | Board RSS | Decision |
|---|---|---|---|
| Moonshine **Tiny** float | ~109 MB | ~180 MB | **USE THIS** |
| Moonshine **Base** float | ~400 MB | ~400 MB | **FORBIDDEN** — collides with 600 MB `MemoryMax` for the coordinator process (leaves no room for YOLO staging + SyNAP handles) |
| Moonshine Tiny **quantized** | ~28 MB | smaller | **BROKEN** — missing `MatMulNBits` scale tensor; onnxruntime rejects at session-create |

**Always use the `float/` variant.** The `quantized/` decoder has a QDQ metadata bug (missing
`model.decoder.embed_tokens.weight_merged_0_scale`) that causes onnxruntime to fail:

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
  qdq_actions.cc:136 TransposeDQWeightsForMatMulNBits
  Missing required scale: model.decoder.embed_tokens.weight_merged_0_scale
```

---

## 2. Model artifacts

**Source**: `UsefulSensors/moonshine` on HuggingFace.

**Files needed** (path inside the HF repo):

| File | HF path | Size |
|---|---|---|
| Encoder | `onnx/merged/tiny/float/encoder_model.onnx` | ~32 MB |
| Decoder | `onnx/merged/tiny/float/decoder_model_merged.onnx` | ~78 MB |
| Tokenizer | `tokenizer.json` (repo root) | ~1 MB |

**Deploy to** (board SD card): `/mnt/sdcard/models/moonshine-tiny/`

```
/mnt/sdcard/models/moonshine-tiny/
├── encoder_model.onnx
├── decoder_model_merged.onnx
└── tokenizer.json
```

**Download on host (WSL2)**:

```bash
# Option A — huggingface-hub CLI (fast, partial clone)
pip install -q huggingface_hub
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
for fname in [
    "onnx/merged/tiny/float/encoder_model.onnx",
    "onnx/merged/tiny/float/decoder_model_merged.onnx",
    "tokenizer.json",
]:
    hf_hub_download("UsefulSensors/moonshine", fname,
                    local_dir="/tmp/moonshine-tiny-dl")
EOF
mkdir -p /tmp/moonshine-tiny
cp /tmp/moonshine-tiny-dl/onnx/merged/tiny/float/encoder_model.onnx /tmp/moonshine-tiny/
cp /tmp/moonshine-tiny-dl/onnx/merged/tiny/float/decoder_model_merged.onnx /tmp/moonshine-tiny/
cp /tmp/moonshine-tiny-dl/tokenizer.json /tmp/moonshine-tiny/

# Option B — git clone sparse (slower; pulls LFS)
git clone --filter=blob:none --no-checkout https://huggingface.co/UsefulSensors/moonshine /tmp/moonshine-repo
cd /tmp/moonshine-repo
git sparse-checkout set onnx/merged/tiny/float tokenizer.json
git checkout
```

**Pin SHAs before scp**:

```bash
sha256sum /tmp/moonshine-tiny/*.onnx /tmp/moonshine-tiny/tokenizer.json
```

Record them. The float encoder starts with `cbbf580f…` and the float decoder with `4131cef0…` (per the Phase A bootstrap record; verify yours match).

---

## 3. Host recipe (WSL2)

Run this before scp'ing to the board — **same onnxruntime 1.25.0** the board uses.
Catches QDQ/shape mismatches cheaply before the board round-trip.

### 3.1 Environment setup

```bash
python3 -m venv /tmp/moonshine-host-venv
source /tmp/moonshine-host-venv/bin/activate

# Pin onnxruntime to the board version
pip install onnxruntime==1.25.0

# useful-moonshine-onnx with --no-deps to avoid 150 MB of librosa/numba/llvmlite
pip install useful-moonshine-onnx==20251121 --no-deps

# Remaining minimal deps
pip install tokenizers==0.22.2 soundfile==0.13.1 numpy
```

> `useful-moonshine-onnx` normally drags in `numba`, `llvmlite` (54 MiB), `librosa`,
> `scipy`, `scikit-learn` — 48 packages (~150 MB). `MoonshineOnnxModel.generate()` is
> pure numpy + onnxruntime; none of those are needed. `--no-deps` gives ~700 KB.

### 3.2 Host smoke test

```python
# host_moonshine_smoke.py
import numpy as np
import soundfile as sf
import moonshine_onnx as mo
from tokenizers import Tokenizer

MODEL_DIR = "/tmp/moonshine-tiny"

print("Loading model (one-time ~1-2 s on host)...")
model = mo.MoonshineOnnxModel(
    models_dir=MODEL_DIR,
    model_name="moonshine/tiny",
    model_precision="float",
)
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

# Use any 16kHz WAV; generate a quick test clip if needed:
#   python3 -c "import numpy as np; import soundfile as sf; sf.write('/tmp/test.wav', np.zeros(16000, np.float32), 16000)"
audio, sr = sf.read("/tmp/test.wav", dtype="float32")
assert sr == 16000, f"Expected 16kHz, got {sr}"
if audio.ndim == 2:
    audio = audio.mean(axis=1).astype(np.float32)

out = model.generate(audio[None])   # shape [1, N_samples]; pure numpy + onnxruntime
text = tok.decode(out[0], skip_special_tokens=True)
print(f"Transcript: {text!r}")
```

**Pass criterion**: no import error, model loads, `generate()` returns a list of token IDs,
decode produces a string (even silence → empty string is fine for this gate).

---

## 4. Board recipe (SL2619)

### 4.1 Pre-conditions

1. SD card (`/dev/mmcblk2p1`, ext4, label `SL2619-models`) formatted and mounted at `/mnt/sdcard`.
2. Python env bootstrapped from Phase A (pip + 19-file stdlib shim bundle; see §4.4 if starting fresh).
3. Model artifacts in `/mnt/sdcard/models/moonshine-tiny/` (scp from host per §4.2).

### 4.2 Deploy model artifacts (user — R3)

```bash
# From WSL2 host:
scp /tmp/moonshine-tiny/encoder_model.onnx \
    /tmp/moonshine-tiny/decoder_model_merged.onnx \
    /tmp/moonshine-tiny/tokenizer.json \
    nouslogic-sl2619:/mnt/sdcard/models/moonshine-tiny/
```

### 4.3 Per-reboot environment recovery (~10 s)

The Python env lives on SD (survives reboots); `/tmp/` symlinks must be recreated after every boot:

```bash
ssh nouslogic-sl2619 'mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard && ln -sfn /mnt/sdcard/p15site /tmp/p15site && ln -sfn /mnt/sdcard/pipbase /tmp/pipbase && ln -sfn /mnt/sdcard/p15-env.sh /tmp/p15-env.sh && . /tmp/p15-env.sh && python3 -c "import onnxruntime; print(onnxruntime.__version__)"'
# Expected: 1.25.0
```

If the symlinks are dangling (`ModuleNotFoundError`), the SD card isn't mounted. Run the mount line first.

### 4.4 First-time board bootstrap (only if Phase A env is missing)

The stock ASTRA SDK 2.3 Yocto image has Python 3.12.9 **without pip, venv, or 12 pure-Python stdlib modules**. The Phase A bootstrap installs them via a 19-file shim bundle.

Full procedure: `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §2.4`.
Abbreviated happy path (user — R3):

```bash
# Step 1: stage on host
mkdir -p /tmp/p15-stage/stdlib-shims
curl -sSL https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/p15-stage/get-pip.py

# Pull stdlib shims from CPython v3.12.9 (must match board Python exactly)
for mod in colorsys getpass mailbox statistics pty plistlib zipapp compileall py_compile tty filecmp; do
  curl -fsSL "https://raw.githubusercontent.com/python/cpython/v3.12.9/Lib/${mod}.py" \
    -o "/tmp/p15-stage/stdlib-shims/${mod}.py"
done
mkdir -p /tmp/p15-stage/stdlib-shims/tomllib
for f in __init__.py _re.py _parser.py _types.py; do
  curl -fsSL "https://raw.githubusercontent.com/python/cpython/v3.12.9/Lib/tomllib/${f}" \
    -o "/tmp/p15-stage/stdlib-shims/tomllib/${f}"
done
# Also create hand-written stubs for fcntl, compileall (hybrid), xmlrpc/client+server
# (See 15-model-compiler-runtime.md §2.4.4 for exact stub content)
tar czf /tmp/p15-stage/stdlib-shims.tar.gz -C /tmp/p15-stage/stdlib-shims .

# Create p15-env.sh
cat > /tmp/p15-stage/p15-env.sh <<'ENV'
export PYTHONUSERBASE=/tmp/pipbase
export PATH="/tmp/pipbase/bin:$PATH"
export PYTHONPATH="/tmp/p15site:${PYTHONPATH:-}"
ENV

# Step 2: ship to board
scp /tmp/p15-stage/get-pip.py /tmp/p15-stage/stdlib-shims.tar.gz /tmp/p15-stage/p15-env.sh nouslogic-sl2619:/tmp/

# Step 3: install (user — SSH)
ssh nouslogic-sl2619 'mkdir -p /tmp/p15site && tar xzf /tmp/stdlib-shims.tar.gz -C /tmp/p15site/ && PYTHONPATH=/tmp/p15site PYTHONUSERBASE=/tmp/pipbase python3 /tmp/get-pip.py --user --no-warn-script-location && . /tmp/p15-env.sh && pip --version'
```

Then install wheels:

```bash
# Install wheels onto the board (user — SSH)
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && pip install --target=/tmp/p15site onnxruntime==1.25.0'
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && pip install --target=/tmp/p15site tokenizers==0.22.2 soundfile==0.13.1'

# numpy: force 1.26.4 despite torq_runtime wheel lying about needing >2.0.0b1
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && rm -rf /tmp/p15site/numpy /tmp/p15site/numpy.libs && pip install --target=/tmp/p15site --no-deps numpy==1.26.4'

# useful-moonshine-onnx without heavy deps
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && pip install --target=/tmp/p15site --no-deps useful-moonshine-onnx==20251121'

# Migrate env to SD so it survives reboots
ssh nouslogic-sl2619 'cp -a /tmp/p15site /mnt/sdcard/ && cp -a /tmp/pipbase /mnt/sdcard/ && cp /tmp/p15-env.sh /mnt/sdcard/ && sync'
```

### 4.5 Board smoke test

Deploy the script once:

```bash
scp /tmp/moonshine_smoke.py nouslogic-sl2619:/mnt/sdcard/scripts/
```

`/mnt/sdcard/scripts/moonshine_smoke.py`:

```python
import numpy as np, soundfile as sf
import moonshine_onnx as mo
from tokenizers import Tokenizer
import time, sys

MODEL_DIR = "/mnt/sdcard/models/moonshine-tiny"

t0 = time.monotonic()
model = mo.MoonshineOnnxModel(
    models_dir=MODEL_DIR,
    model_name="moonshine/tiny",
    model_precision="float",
)
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
load_s = time.monotonic() - t0
print(f"Load: {load_s:.2f}s", flush=True)

audio_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdcard/fixtures/say_hi.wav"
audio, sr = sf.read(audio_path, dtype="float32")
assert sr == 16000, f"Need 16 kHz, got {sr}"
if audio.ndim == 2:
    audio = audio.mean(axis=1).astype(np.float32)

t1 = time.monotonic()
out = model.generate(audio[None])   # pure numpy + onnxruntime; no librosa
text = tok.decode(out[0], skip_special_tokens=True)
gen_s = time.monotonic() - t1

n_tok = len(out[0])
print(f"Generate: {gen_s:.2f}s | {n_tok} tokens | {n_tok/gen_s:.1f} tok/s")
print(f"Transcript: {text!r}")
```

Run on board (user — SSH):

```bash
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && python3 /mnt/sdcard/scripts/moonshine_smoke.py /mnt/sdcard/fixtures/say_hi.wav'
```

**Expected output** (11-second JFK clip baseline, 2026-04-23):

```
Load: 4.26s
Generate: 2.28s | 26 tokens | 11.4 tok/s
Transcript: 'Ask not what your country can do for you...'
```

---

## 5. Architecture deep-dive

```
Audio (16 kHz PCM float32, shape [1, N_samples])
    │
[Moonshine Encoder — ONNX CPU, onnxruntime]
    ·  Convolutional encoder
    ·  ~60–100 ms for 15 s audio
    ·  Output: hidden states [1, time_steps, 384]
    │
[Moonshine Decoder — ONNX CPU, autoregressive]
    ·  Cross-attention over encoder hidden states
    ·  Token-by-token generation; ~10–20 ms per token
    ·  50–200 tokens per utterance
    ·  Total generate latency: ~100–250 ms
    │
[BPE Tokenizer — tokenizers lib]
    ·  Token IDs → text, ~1 ms
    │
Transcribed text string
```

**Total measured latency** (SL2619, 2× A55 @ 2 GHz, 11 s JFK clip):
`load 4.26 s + generate 2.28 s = 6.54 s; 11.4 tok/s`

Load time is one-time per process. In a long-running systemd service the model is loaded once at startup and reused — steady-state latency is only the generate step (~2–3 s for a typical command utterance ≤ 5 s).

**Why CPU, not NPU?** The encoder could be NPU-compiled (ONNX→VMFB), but the autoregressive decoder stays on CPU regardless (attention KV-cache op shape changes each step — poor fit for fixed-function NPU tiling). The encoder is only 20–30 % of total latency; accelerating it yields minimal end-to-end gain for significant integration complexity. Phase 2+ NPU path is a research gate, not a planned delivery.

---

## 6. Full speech pipeline (Phase 5 integration)

```
arecord -D hw:0,3 -r 16000 -f S24_LE -c 2      (see docs/references/sl2619-dmic.md)
    │
    ▼  right-shift each sample >> 8              (S24_LE sample is in upper 24 bits of int32)
    ▼  downmix L+R → mono float32               (DMIC is stereo; Moonshine expects mono)
    ▼  normalize to [-1.0, 1.0]                 (divide by 2^23 after shift)
    │
[Silero VAD — onnxruntime CPU, ~2 ms / 30 ms frame]
    │  gate: speech_prob > 0.6 for ≥ 3 consecutive frames
    ▼  (only passes speech segments)
    │
[Moonshine Tiny encoder + decoder — onnxruntime CPU]
    │  ~250 ms per utterance
    ▼
Transcript string → command parser → Coordinator state machine
```

**ALSA → float32 conversion snippet** (for `a55/speech/alsa_capture.cpp`):

```cpp
// hw:0,3 DMIC: S24_LE, stereo, 16 kHz.
// Each int32 frame: sample in upper 24 bits; lower 8 bits are zero padding.
// Moonshine expects mono float32 in [-1.0, 1.0].
int32_t lr[2];
snd_pcm_readi(pcm, lr, 1);                            // one interleaved stereo frame
int32_t mono_raw = (lr[0] >> 8) + (lr[1] >> 8);      // sum then shift for averaging
float sample = static_cast<float>(mono_raw) / (2.0f * 8388608.0f);  // 2^23, stereo sum
```

**onnxruntime session config** (in `a55/speech/moonshine_inference.cpp`):

```cpp
Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "moonshine"};
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(2);    // A55 has 2 Linux-exposed cores; oversubscribing hurts
opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
opts.AppendExecutionProvider("CPU", {});
// Load encoder and decoder as separate sessions
Ort::Session enc_session{env, (model_dir / "encoder_model.onnx").c_str(), opts};
Ort::Session dec_session{env, (model_dir / "decoder_model_merged.onnx").c_str(), opts};
```

---

## 7. Gotchas

| Mistake | Symptom | Fix |
|---|---|---|
| Using `quantized/` decoder variant | `onnxruntime: Missing required scale: ...weight_merged_0_scale` at session-create | Use `onnx/merged/tiny/float/` variant |
| Installing `useful-moonshine-onnx` WITH deps | 150 MB of librosa/numba/llvmlite downloaded; board tmpfs may OOM during install | `pip install --no-deps useful-moonshine-onnx` |
| Using `transcribe()` instead of `generate()` | `ImportError: librosa` at runtime | Call `MoonshineOnnxModel.generate(audio[None])` directly |
| numpy 2.x in env alongside torq_runtime | `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | `rm -rf /tmp/p15site/numpy && pip install --target=/tmp/p15site --no-deps numpy==1.26.4` |
| `pip install --force-reinstall --target=` | Silently skips overwrite of existing dirs | `rm -rf /tmp/p15site/<pkg>` first, then reinstall |
| Audio at wrong sample rate | Silent inference / garbage transcript (no error) | `assert sr == 16000` before `generate()` |
| Feeding stereo numpy array to `generate()` | Shape mismatch in encoder | `audio.mean(axis=1).astype(np.float32)` to downmix |
| S24_LE DMIC samples fed raw to Moonshine | Silent transcript (48 dB attenuation from skipped shift) | Right-shift by 8 (see §6 + `sl2619-dmic.md §5`) |
| Missing `tokenizer.json` in MODEL_DIR | `FileNotFoundError` at `Tokenizer.from_file` | Ensure tokenizer.json is co-located with the ONNX files |
| Symlinks to SD dangling after reboot | `ModuleNotFoundError: No module named 'onnxruntime'` | Mount SD + recreate symlinks (§4.3) |
| Multi-line `python3 -c "..."` over SSH | Python `-c` rejects indented continuation lines | One-liner with `;`, or scp a `.py` file |

---

## 8. Version pins (board-proven)

| Package | Version | Note |
|---|---|---|
| `onnxruntime` | **1.25.0** | Matches board wheel; use same on host for pre-validation |
| `numpy` | **1.26.4** (forced `--no-deps`) | torq_runtime C-ext is numpy 1.x ABI despite wheel claiming >2.0 |
| `tokenizers` | 0.22.2 | BPE decode |
| `soundfile` | 0.13.1 | WAV read |
| `useful-moonshine-onnx` | 20251121 | High-level API; install `--no-deps` |
| Python (board) | 3.12.9 (Yocto ASTRA SDK 2.3) | shim bundle must match this exactly |

---

## 9. Source pointers

| Item | Location |
|---|---|
| onnxruntime section | `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §4` |
| Bootstrap recipe (stdlib shims) | `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §2.4` |
| Stub source (fcntl, compileall, xmlrpc) | `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §2.4.4` |
| Phase A closure post-mortem | `SynapticSL2619/docs/plans/backlogs.md §1.17` |
| Storage layout (SD card + tmpfs) | `SynapticSL2619/docs/conventions/15-model-compiler-runtime.md §5` |
| DMIC capture + S24_LE shift | `docs/references/sl2619-dmic.md §5` |
| Torq NPU path (deferred Phase 2+) | `SynapticSL2619/references/Synaptics/torq-tools/src/torq/models/moonshine/` |
