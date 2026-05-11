# USB audio peripheral testing on SL2619

Stepwise recipe to verify a USB speaker + microphone on the SL2619 board image,
using only ALSA + Python 3 stdlib. The board image ships with no PulseAudio,
no PipeWire, no `opkg`, and no `sox`/`ffmpeg` — every step here is constrained
to what is already installed.

Verified working **2026-05-11** against the ROFALL P1U-4 "USB Audio 4.0"
speakerphone, which enumerates as **`MV-SILICON P10S` (USB ID `1234:5684`)** —
the chipset's identity, not the marketing name. The procedure applies to any
UAC1/UAC2 USB audio class device on this image.

## What you can rely on

| Resource | State on this board image |
|---|---|
| `aplay`, `arecord`, `speaker-test`, `amixer`, `alsamixer`, `lsusb` | Installed |
| `python3` (3.12.9) with `wave` + `audioop` | Installed (use for signal analysis) |
| `pactl`, `pw-cli`, `wpctl`, `pulseaudio`, `pipewire` | **Not installed** — there is no Pulse/PipeWire layer; everything goes via raw ALSA |
| `sox`, `ffmpeg`, `file`, `opkg` | **Not installed**, no package manager to add them |
| `/dev/snd/*` permissions | `root:audio 0660`; running as root has no permission gate |

## Critical gotchas (read these before the first test)

| Gotcha | Why it bites | Fix |
|---|---|---|
| PCM playback can sit at **0% / 0.00 dB / on** after device insertion | Speaker tests "succeed" silently — `speaker-test` returns 0, ALSA reports no underruns, you hear nothing. | Always `amixer -c <N>` to inspect before testing. If `PCM` Playback is 0%, raise it (`sset 'PCM' 50%`). |
| Mic capture defaults to **100% / +16 dB** (very hot) | Single-mic speakerphones easily peak at −1 dBFS — fine for "does it work" but borderline clipping for production capture. | For non-test use drop to ~70%: `amixer -c <N> sset 'Mic' 70%`. |
| The P10S advertises **48 kHz capture only** | `arecord -r 16000` resamples in kernel and hides device-side format issues. | Always probe `cat /proc/asound/card<N>/stream0` first; record at the device's native rate. |
| ALSA card number is **bus-order dependent** | Hard-coding `hw:1,0` breaks if another USB audio device is also plugged. | Re-read `cat /proc/asound/cards` every session. |
| `speaker-test -s` is **1-based** (1=FL, 2=FR) | `-s 0` is rejected as `Invalid parameter`. | Drop `-s` entirely to test all channels in sequence. |
| `audioop` is deprecated in Python 3.12 and slated for removal in 3.13 | Emits a `DeprecationWarning` at *import* time — `warnings.filterwarnings` set after the import does not suppress it. | Either accept the harmless warning, or invoke with `python3 -W ignore::DeprecationWarning`. After 3.13 the analysis step needs a rewrite (raw `struct`/`array` on the PCM bytes). |
| SSH identity key may require a passphrase | If `~/.ssh/sl2619_nouslogic_wsl` is encrypted and no askpass helper is available, SSH falls back to interactive password and fails. | Once per terminal session: `ssh-add ~/.ssh/sl2619_nouslogic_wsl` (interactive — agent must be running). |

## Recipe

Substitute `<N>` with the USB audio device's ALSA card number (from step 1).

### 1. Detect & identify (read-only)

```bash
ssh nouslogic-sl2619 'lsusb; echo; cat /proc/asound/cards'
```

The USB audio device appears in `lsusb` with its USB ID + chipset name, and in
`/proc/asound/cards` as a numbered ALSA card. Note the **card number** — this
is `<N>` for the rest of the recipe.

Probe the device's native PCM formats (non-negotiable — the device only accepts
what it advertises):

```bash
ssh nouslogic-sl2619 'cat /proc/asound/card<N>/stream0'
```

You will get a `Playback:` and a `Capture:` block. Record from each: supported
**Rates**, **Channels**, **Format**, and **Channel map**. The on-board P10S
example:

```
Playback:  S16_LE, 2 ch (FL/FR), 44100 or 48000 Hz
Capture:   S16_LE, 2 ch (FL/FR), 48000 Hz only
```

### 2. Inspect mixer; raise PCM if it's at floor

```bash
ssh nouslogic-sl2619 'amixer -c <N>'
```

For each Simple mixer control, check:
- `[on]` vs `[off]` — must be `on`
- Volume `[NN%]` — the trap is **PCM at `[0%]` and unmuted**, which is silent

If `PCM` is at 0%, raise to a safe-ish 50%:

```bash
ssh nouslogic-sl2619 "amixer -c <N> sset 'PCM' 50%"
```

Step up to 75% if needed; never start at 100% on an unfamiliar speaker.

### 3. Speaker test (synthesized sine, no asset staging)

```bash
ssh nouslogic-sl2619 'speaker-test -D plughw:<N>,0 -c 2 -r 48000 -t sine -f 440 -l 1'
```

Use the device's **native** channel count + rate (from step 1's `stream0`).
What to listen for:
- A 440 Hz tone (orchestra A / dial-tone pitch) on Front Left for ~2 s, then on Front Right.
- Clean tone — no buzz, no distortion, no clipping.
- No stutter at the end (a stutter usually means USB underrun on a full-speed bus).

### 4. Microphone capture (5 s, native format)

```bash
ssh nouslogic-sl2619 'arecord -D plughw:<N>,0 -f S16_LE -r 48000 -c 2 -d 5 /tmp/usb_mic_test.wav'
```

Speak immediately when `Recording WAVE …` prints. Expected file size at
48 kHz × 2 ch × 16-bit × 5 s = **~937.5 KiB**. Anything dramatically smaller
means truncation; anything dramatically larger means the wrong format.

### 5. Signal analysis with `wave` + `audioop` (stdlib only)

```bash
ssh nouslogic-sl2619 "python3 -W ignore::DeprecationWarning - <<'PY'
import wave, audioop, math
with wave.open('/tmp/usb_mic_test.wav','rb') as w:
    nch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
    frames = w.readframes(nf)
print(f'channels={nch} rate={sr} frames={nf} duration={nf/sr:.2f}s')
db = lambda v: float('-inf') if v == 0 else 20*math.log10(v/32768)
chans = [('L', audioop.tomono(frames, sw, 1.0, 0.0)),
         ('R', audioop.tomono(frames, sw, 0.0, 1.0))] if nch == 2 else [('M', frames)]
for name, ch in chans:
    peak, rms = audioop.max(ch, sw), audioop.rms(ch, sw)
    print(f'{name}: peak={peak:5d} ({db(peak):+6.1f} dBFS)   rms={rms:5d} ({db(rms):+6.1f} dBFS)')
PY
"
```

**Reading the numbers:**

| Peak (dBFS) | Verdict |
|---|---|
| 0 to −1 | Clipping — drop mic gain |
| −6 to −1 | Excellent for ASR / capture |
| −20 to −6 | Healthy voice |
| −40 to −20 | Quiet / distant — still usable |
| < −50 | Effectively silence — nothing reached the mic |

| Inter-channel balance | Meaning |
|---|---|
| L vs R within ~1 dB | Single capsule duplicated to stereo (typical for USB speakerphones) |
| L vs R differ by 1–6 dB | Two-capsule stereo or asymmetric routing |
| L vs R differ by > 6 dB | Likely one channel broken or muted |

### 6. End-to-end loopback

Play the recording back through the same device — proves the full
mic → storage → speaker chain works on a single command:

```bash
ssh nouslogic-sl2619 'aplay -D plughw:<N>,0 /tmp/usb_mic_test.wav'
```

You should hear yourself, intelligibly. Stutter on playback of an already-saved
WAV implicates USB / aplay scheduling, not the mic.

### 7. Cleanup (optional)

```bash
ssh nouslogic-sl2619 "amixer -c <N> sset 'PCM' 0%; rm -f /tmp/usb_mic_test.wav"
```

Only if you want to leave the board in its pre-test state. Mixer levels held by
USB Audio Class are in-kernel and reset when the device is unplugged or the
board reboots.

## Known-good baseline (2026-05-11)

Run logged against ROFALL P1U-4 (P10S, card 1) on `nouslogic-sl2619`:

```
channels=2 rate=48000 frames=240000 duration=5.00s
L: peak=29441 (  -0.9 dBFS)   rms=  663 ( -33.9 dBFS)
R: peak=27393 (  -1.6 dBFS)   rms=  662 ( -33.9 dBFS)
```

- Mic L/R within 0.7 dB → single capsule doubled to stereo (expected).
- Peak−RMS spread ~33 dB → typical voice dynamic range.
- Speaker 440 Hz sine: audible, clean, both channels.
- End-to-end loopback: audible and intelligible.

## Triage matrix

| Symptom | Likely cause | Action |
|---|---|---|
| `arecord` / `aplay`: `Invalid argument` opening `plughw:<N>,0` | Card number wrong or device disappeared from the bus | `cat /proc/asound/cards`; physically replug |
| Speaker test completes with `# 0 buffer underruns`, no sound | PCM volume at 0% (most common); device on a switched USB hub powering down between bursts | `amixer -c <N>` and raise PCM; check hub power |
| Mic peaks below −50 dBFS | Mic muted, gain at 0%, or another process holds the device | `amixer -c <N>`; `fuser -v /dev/snd/*` |
| Stutter on playback of a WAV already on local disk | USB underrun; full-speed bus contention with other devices | Unplug other USB devices; verify topology with `lsusb -t` |
| L vs R asymmetric > 6 dB | Hardware fault or unusual mic config | Investigate the device — not a board-side issue |
| SSH passphrase prompt loops forever | Identity key is encrypted and ssh-agent doesn't have it loaded | `ssh-add ~/.ssh/sl2619_nouslogic_wsl` in a terminal session before SSH-ing |
| `audioop` raises `DeprecationWarning` | Python 3.12 deprecation; will become an error on 3.13 | Run with `python3 -W ignore::DeprecationWarning`; replace with raw `struct`/`array` parsing before any 3.13 image |
