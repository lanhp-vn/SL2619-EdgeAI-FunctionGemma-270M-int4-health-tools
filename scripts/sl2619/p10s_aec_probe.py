#!/usr/bin/env python3
"""P10S firmware AEC capability probe.

Duplex tone test: play a 1234 Hz sine through the P10S speaker while capturing
from its mic, then compare 1234 Hz energy in the capture vs a silence baseline.

If firmware AEC is active, the tone is suppressed in the capture (small delta).
If not, the tone bleeds through the speakerphone enclosure to the mic (large
delta), and v2 will need software AEC (speexdsp / webrtc-audio-processing).

Stdlib only: no Pulse/PipeWire, no sox/ffmpeg, no numpy on the board image.
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import time
import warnings
import wave

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402  (must import after the warning filter)


TONE_HZ = 1234.0
RATE = 48000
CAP_DUR_S = 5
TONE_DUR_S = 8  # > CAP_DUR_S + warmup so capture is fully covered
WARMUP_S = 1.0
TONE_AMP = 0.5  # 0..1, scaled to S16 peak. 0.5 = -6 dBFS source.

CARD_MARKERS = ("P10S", "MV-SILICON", "MV_SILICON")  # /proc/asound/cards names
USB_ID = "1234:5684"

TONE_WAV = "/tmp/p10s_aec_tone.wav"
SILENCE_WAV = "/tmp/p10s_aec_silence.wav"
DUPLEX_WAV = "/tmp/p10s_aec_duplex.wav"


# region detection + mixer
def detect_card() -> int:
    with open("/proc/asound/cards", encoding="utf-8") as f:
        text = f.read()
    for line in text.splitlines():
        if any(m in line for m in CARD_MARKERS):
            return int(line.strip().split()[0])
    sys.stderr.write("P10S not found in /proc/asound/cards:\n" + text)
    sys.exit(2)


def set_mixer(card: int) -> None:
    # Best-effort: not every kernel exposes both controls; report what works.
    for ctrl, level in [("PCM", "50%"), ("Mic", "70%")]:
        r = subprocess.run(
            ["amixer", "-c", str(card), "sset", ctrl, level],
            capture_output=True,
            text=True,
        )
        status = "ok" if r.returncode == 0 else f"FAIL ({r.stderr.strip()})"
        print(f"  amixer -c {card} sset {ctrl} {level} -> {status}", flush=True)


# endregion


# region sine generation
def gen_sine_wav(path: str, freq_hz: float, dur_s: float, rate: int, amp: float) -> None:
    n = int(dur_s * rate)
    peak = int(amp * 32767)
    omega = 2 * math.pi * freq_hz / rate
    # Build interleaved L,R as Python ints; pack once.
    flat: list[int] = [0] * (n * 2)
    for i in range(n):
        v = int(peak * math.sin(omega * i))
        flat[2 * i] = v
        flat[2 * i + 1] = v
    pcm = struct.pack(f"<{n*2}h", *flat)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


# endregion


# region analysis
def rms_dbfs(pcm: bytes, sw: int) -> float:
    rms = audioop.rms(pcm, sw)
    return float("-inf") if rms == 0 else 20.0 * math.log10(rms / 32768.0)


def goertzel_dbfs(pcm: bytes, sw: int, rate: int, target_hz: float, channels: int) -> float:
    """Single-bin DFT magnitude at target_hz, expressed as dBFS of S16 peak."""
    mono = audioop.tomono(pcm, sw, 0.5, 0.5) if channels == 2 else pcm
    n = len(mono) // sw
    samples = struct.unpack(f"<{n}h", mono)
    omega = 2 * math.pi * target_hz / rate
    coeff = 2 * math.cos(omega)
    q1 = q2 = 0.0
    for s in samples:
        q0 = coeff * q1 - q2 + s
        q2, q1 = q1, q0
    re = q1 - q2 * math.cos(omega)
    im = q2 * math.sin(omega)
    mag = math.sqrt(re * re + im * im) * 2.0 / n  # peak amplitude estimator
    return float("-inf") if mag == 0 else 20.0 * math.log10(mag / 32768.0)


def analyze(path: str) -> tuple[float, float]:
    with wave.open(path, "rb") as w:
        nch, sw, sr, nf = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        frames = w.readframes(nf)
    total = rms_dbfs(frames, sw)
    tone = goertzel_dbfs(frames, sw, sr, TONE_HZ, nch)
    print(
        f"    nch={nch} sr={sr} dur={nf/sr:.2f}s "
        f"total={total:+6.1f} dBFS  {TONE_HZ:.0f}Hz={tone:+6.1f} dBFS",
        flush=True,
    )
    return total, tone


# endregion


def arecord_blocking(card: int, path: str, dur_s: int) -> None:
    """Single-shot blocking capture (mono — single capsule is duplicated anyway)."""
    subprocess.run(
        ["arecord", "-q", "-D", f"plughw:{card},0", "-f", "S16_LE",
         "-r", str(RATE), "-c", "1", "-d", str(dur_s), path],
        check=True,
    )


def arecord_background(card: int, path: str, dur_s: int) -> subprocess.Popen:
    """Start mono capture as a background process; wait() on the handle."""
    return subprocess.Popen(
        ["arecord", "-q", "-D", f"plughw:{card},0", "-f", "S16_LE",
         "-r", str(RATE), "-c", "1", "-d", str(dur_s), path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def aplay_background(card: int, path: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["aplay", "-q", "-D", f"plughw:{card},0", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def main() -> int:
    print("=== P10S AEC capability probe ===", flush=True)

    card = detect_card()
    print(f"\n[0] P10S on ALSA card {card}", flush=True)

    print(f"\n[1] Mixer setup (card {card})", flush=True)
    set_mixer(card)

    print(f"\n[2] Generating {TONE_DUR_S}s {TONE_HZ:.0f} Hz tone @ {RATE} Hz S16_LE stereo", flush=True)
    gen_sine_wav(TONE_WAV, TONE_HZ, TONE_DUR_S, RATE, TONE_AMP)
    print(f"    {TONE_WAV} = {os.path.getsize(TONE_WAV)} bytes", flush=True)

    print(f"\n[3] Capture A: silence baseline ({CAP_DUR_S}s mono, no playback)", flush=True)
    arecord_blocking(card, SILENCE_WAV, CAP_DUR_S)

    print(
        f"\n[4] Capture B: DUPLEX — arecord ({CAP_DUR_S}s mono) + aplay tone (bg)\n"
        f"    Start order: capture first, then playback, to give arecord the\n"
        f"    USB capture endpoint before the bus loads up with playback.\n"
        f"    >>> If no AEC, you should hear the {TONE_HZ:.0f} Hz tone clearly",
        flush=True,
    )
    rec = arecord_background(card, DUPLEX_WAV, CAP_DUR_S)
    time.sleep(WARMUP_S)
    play = aplay_background(card, TONE_WAV)
    rec.wait()
    if rec.returncode != 0:
        err = rec.stderr.read().decode("utf-8", "replace").strip() if rec.stderr else ""
        print(f"    ERROR: arecord exit {rec.returncode}: {err}", flush=True)
        play.terminate()
        return 1
    play.wait()
    if play.returncode != 0:
        err = play.stderr.read().decode("utf-8", "replace").strip() if play.stderr else ""
        print(f"    WARN: aplay exit {play.returncode}: {err}", flush=True)

    print("\n[5] Analysis", flush=True)
    print("  silence:", flush=True)
    s_total, s_tone = analyze(SILENCE_WAV)
    print("  duplex :", flush=True)
    d_total, d_tone = analyze(DUPLEX_WAV)

    delta_total = d_total - s_total
    delta_tone = d_tone - s_tone
    print(f"\n  Δ total RMS : {delta_total:+5.1f} dB", flush=True)
    print(f"  Δ {TONE_HZ:.0f} Hz   : {delta_tone:+5.1f} dB", flush=True)

    print("\n=== Verdict ===", flush=True)
    if delta_tone < 6:
        verdict = "FIRMWARE AEC LIKELY ACTIVE"
        detail = f"1234 Hz energy barely changed (Δ={delta_tone:+.1f} dB) — the P10S is suppressing its own playback in the captured stream."
    elif delta_tone > 20:
        verdict = "NO FIRMWARE AEC"
        detail = f"1234 Hz energy rose sharply (Δ={delta_tone:+.1f} dB) — playback bleeds straight into the capture. v2 needs software AEC."
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Partial suppression (Δ={delta_tone:+.1f} dB). Re-run with a speech source or different freq before deciding."
    print(verdict, flush=True)
    print(detail, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
