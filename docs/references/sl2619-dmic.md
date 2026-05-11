# SL2619 DMIC — Working Recipe

> Empirically verified 2026-04-22 (T9b, bin D-2). Ground truth: `SynapticSL2619/docs/plans/backlogs.md §1.16`.

---

## 1. Hardware identity

Two on-board PDM digital microphones share **one stereo PDM input**. No external USB or I²S mic needed. Biased from the board 3.3 V rail; negligible power draw.

ALSA mapping:
- **Card 0** — `klamathasoc` (SoC codename: klamath)
- **Device 3** — listed as `dummy-dmic` in `arecord -l` output

The `snd-soc-dummy-dai-3` DAI name is a kernel placeholder on the *codec side* only. The PDM controller (`berlin_capture.c` + CIC decimator) is fully wired and produces genuine PCM audio on the stock image.

---

## 2. Enumerate the device

```bash
ssh nouslogic-sl2619 'arecord -l'
```

Expected output includes:

```
card 0: klamathasoc [...], device 3: dummy-dmic-3 []
  Subdevice #0: subdevice #0
```

---

## 3. Format constraints

From `references/Synaptics/linux-drivers-synaptics/sound/soc/syna/dmic.c:22-25`:

```c
#define DMIC_RECORD_FORMATS  (SNDRV_PCM_FMTBIT_S24_LE \
                              | SNDRV_PCM_FMTBIT_S24_3LE \
                              | SNDRV_PCM_FMTBIT_S32_LE)
```

| Parameter | Accepted | Rejected |
|---|---|---|
| Format | `S24_LE`, `S24_3LE`, `S32_LE` | **`S16_LE` — exits nonzero immediately** |
| Channels | 2–8 | **mono (1) — rejected by ALSA** |
| Rate | 8000–96000 Hz | — |

Using `-f S16_LE -c 1` fails at `snd_pcm_hw_params: Sample format non available` before a single sample is captured.

---

## 4. Basic capture command

```bash
arecord -D hw:0,3 -r 16000 -f S24_LE -c 2 -d 1 /tmp/test_dmic.raw
```

| Flag | Value | Reason |
|---|---|---|
| `-D hw:0,3` | card 0, device 3 | PDM DMIC path |
| `-r 16000` | 16 kHz | Moonshine STT input requirement |
| `-f S24_LE` | 24-bit in 32-bit frame | only format accepted by the driver |
| `-c 2` | stereo | mono is rejected; downmix in software after |
| `-d 1` | 1 second | duration |

By-name alternative (preferred in production code — survives card reordering across Yocto updates):

```bash
arecord -D plughw:CARD=klamathasoc,DEV=3 -r 16000 -f S24_LE -c 2 -d 1 /tmp/test_dmic.raw
```

**WAV header note**: despite the `.raw` extension, `arecord` writes a **WAV file by default** (44 B RIFF/fmt/data header + PCM data). Add `--file-type raw` to suppress the header:

```bash
arecord -D hw:0,3 -r 16000 -f S24_LE -c 2 -d 1 --file-type raw /tmp/test_dmic.raw
```

---

## 5. Sample-packing gotcha — the 48 dB trap

**`S24_LE` on the SL2619 packs the 24 useful bits into the UPPER 24 bits of the 32-bit word. The lower 8 bits are zero padding.**

Empirical proof (T9b): every sample is exactly divisible by 256 (e.g., `2,095,104 = 8,184 << 8`). Full-scale positive = `(2²³ − 1) << 8 = 2,147,483,392`.

Feeding raw `int32` values directly into a model expecting S16-range input attenuates the signal by 48 dB — it reads as silence with no error.

### Python extraction

```python
import struct

raw = open('/tmp/test_dmic.raw', 'rb').read()
offset = 44  # skip WAV header; 0 if captured with --file-type raw
frames = struct.unpack('<' + 'i' * ((len(raw) - offset) // 4), raw[offset:])

left  = frames[0::2]
right = frames[1::2]

# Correct: right-shift to recover the 24-bit value
left_24  = [s >> 8 for s in left]
right_24 = [s >> 8 for s in right]

# Verification
nonzero_l = sum(1 for s in left if s != 0)
nonzero_r = sum(1 for s in right if s != 0)
print(f'L: {nonzero_l}/{len(left)}, R: {nonzero_r}/{len(right)} non-zero')
```

### C++ (a55/speech/alsa_capture.cpp)

```cpp
// hw:0,3 DMIC produces S24_LE with sample in upper 24 bits of 32-bit frame.
// Right-shift by 16 (>> 8 to drop zero padding, >> 8 again to reduce to S16 range)
// before feeding to Moonshine.
int32_t raw_sample;
snd_pcm_readi(pcm, &raw_sample, 1);
int16_t s16 = static_cast<int16_t>(raw_sample >> 16);
```

---

## 6. Observed performance (T9b empirical, 2026-04-22)

| Channel | Non-zero / 16,000 frames | Peak (raw int32) | Peak dBFS | RMS |
|---|---|---|---|---|
| L | **16,000 / 16,000** | 318,188,544 | −16.6 dBFS | 95,618,688 |
| R | **16,000 / 16,000** | 323,480,064 | −16.5 dBFS | 82,608,135 |

Both mics active. L/R symmetry within ~1 dB. Stimulation: hand clap at bench distance. Not clipping (headroom to −3 dBFS). Consider +6–12 dB soft gain or AGC for typical speech levels before Moonshine.

---

## 7. Application integration (Phase 5 / A55 C++)

### YAML config (bind by card name — not by index)

```yaml
speech:
  alsa_device: plughw:CARD=klamathasoc,DEV=3
  vad_threshold: 0.6
  moonshine_model: /usr/share/onnx/moonshine-tiny.onnx
```

### Speech pipeline

```
arecord (hw:0,3, S24_LE, 16 kHz, 2ch)
  → right-shift >> 8, downmix L+R → mono int16
  → ring buffer
  → Silero VAD (onnxruntime CPU, ~2 ms / 30 ms frame)
  → [gate: speech_prob > 0.6 for ≥ 3 consecutive frames]
  → Moonshine Tiny (onnxruntime CPU, ~250 ms/utterance, ~180 MB RSS)
  → transcript string → command parser → Coordinator
```

### onnxruntime session config

```cpp
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(2);    // A55 = 2 cores; oversubscribing hurts
opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
opts.AppendExecutionProvider("CPU", {});
```

---

## 8. Foot-guns

| Mistake | Symptom | Fix |
|---|---|---|
| `-f S16_LE` | `snd_pcm_hw_params: Sample format non available` | Use `-f S24_LE` |
| `-c 1` mono | Same ALSA error | Use `-c 2`; downmix in software |
| Feeding raw `int32` to Moonshine | Silent transcript (48 dB attenuation, no error) | Right-shift by 8 (or 16 for S16 output) before model ingestion |
| Forgetting WAV header | Parser misaligns on first 44 bytes | Use `--file-type raw` or skip header |
| Trusting `dummy-dmic` name as "no hardware" | Skipping DMIC, reaching for USB fallback | DAI label is CODEC-side placeholder only; PDM path is fully wired |
| Binding by index `hw:0,3` in production code | Breaks if card order changes on Yocto update | Bind by name `plughw:CARD=klamathasoc,DEV=3` |

---

## 9. Driver source pointers

| File | What it pins |
|---|---|
| `SynapticSL2619/references/Synaptics/linux-drivers-synaptics/sound/soc/syna/dmic.c:22-25` | Hardcoded `S24_LE\|S24_3LE\|S32_LE` format mask + 8000–96000 Hz rate range |
| `SynapticSL2619/references/Synaptics/linux-drivers-synaptics/sound/soc/syna/berlin_capture.c:60-61, 390-440` | PDM→PCM CIC decimator |
| `SynapticSL2619/docs/datasheets/sl2610-datasheets/NR-158526-AN-5_ASTRA_MACHINA_FOUNDATION_SERIES_I2S,_PDM_INTERFACES_APPLICATION_NOTES.pdf §1.1.1, §1.6` | PDM DMIC SoC electrical spec |
