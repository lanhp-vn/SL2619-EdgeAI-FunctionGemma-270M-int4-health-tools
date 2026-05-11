#!/usr/bin/env python3
"""P10S AEC speech-survival probe.

Follow-up to p10s_aec_probe.py. The first probe proved the firmware AEC
suppresses the device's own playback echo. This one answers the next
question: when AEC is active, does the user's speech also get attenuated
(bad — half-duplex stays mandatory for v2) or does it pass through cleanly
(good — v2 can do continuous wake-word and barge-in during TTS)?

Method:
  Phase A — speech-only baseline (no playback, speak for 5 s).
  Phase B — speech + duplex (1234 Hz tone playing, speak for 5 s).

  Total RMS is a clean speech metric here because the prior probe showed
  the AEC drives the tone to the S16 quantization floor (-108 dBFS), so
  the tone contributes nothing to total RMS even during duplex.

Decision:
  |delta_speech_dB| <  6  → AEC is selective; v2 barge-in viable.
   delta_speech_dB < -15  → AEC also kills speech; v1 half-duplex stays
                            mandatory for v2.
  in between              → judgment call; re-run or test with louder voice.
"""
from __future__ import annotations

import math
import struct
import subprocess
import sys
import time
import warnings
import wave

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # noqa: E402


TONE_HZ = 1234.0
RATE = 48000
CAP_DUR_S = 5
PROMPT_LEAD_S = 3  # countdown before each capture
WARMUP_S = 1.0    # aplay → arecord stagger
TONE_DUR_S = CAP_DUR_S + WARMUP_S + 2  # outlast the capture

CARD_MARKERS = ("P10S", "MV-SILICON", "MV_SILICON")
TONE_WAV = "/tmp/p10s_aec_tone.wav"
SPEECH_ONLY_WAV = "/tmp/p10s_aec_speech_only.wav"
SPEECH_DUPLEX_WAV = "/tmp/p10s_aec_speech_duplex.wav"
SILENCE_WAV = "/tmp/p10s_aec_silence.wav"  # from the prior probe; optional


def detect_card() -> int:
    with open("/proc/asound/cards", encoding="utf-8") as f:
        text = f.read()
    for line in text.splitlines():
        if any(m in line for m in CARD_MARKERS):
            return int(line.strip().split()[0])
    sys.stderr.write("P10S not found:\n" + text)
    sys.exit(2)


def set_mixer(card: int) -> None:
    for ctrl, level in [("PCM", "50%"), ("Mic", "70%")]:
        subprocess.run(
            ["amixer", "-c", str(card), "sset", ctrl, level],
            capture_output=True,
            check=False,
        )


def gen_sine_wav(path: str, freq_hz: float, dur_s: float, rate: int, amp: float) -> None:
    n = int(dur_s * rate)
    peak = int(amp * 32767)
    omega = 2 * math.pi * freq_hz / rate
    flat = [0] * (n * 2)
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


def countdown(prefix: str) -> None:
    for n in range(PROMPT_LEAD_S, 0, -1):
        print(f"  {prefix} in {n}...", flush=True)
        time.sleep(1.0)
    print("  >>> === SPEAK NOW === <<<", flush=True)


def arecord_blocking(card: int, path: str, dur_s: int) -> None:
    subprocess.run(
        ["arecord", "-q", "-D", f"plughw:{card},0", "-f", "S16_LE",
         "-r", str(RATE), "-c", "1", "-d", str(dur_s), path],
        check=True,
    )


def arecord_background(card: int, path: str, dur_s: int) -> subprocess.Popen:
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


def rms_dbfs(pcm: bytes, sw: int) -> float:
    rms = audioop.rms(pcm, sw)
    return float("-inf") if rms == 0 else 20.0 * math.log10(rms / 32768.0)


def goertzel_dbfs(pcm: bytes, sw: int, rate: int, target_hz: float, channels: int) -> float:
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
    mag = math.sqrt(re * re + im * im) * 2.0 / n
    return float("-inf") if mag == 0 else 20.0 * math.log10(mag / 32768.0)


def analyze(path: str, label: str) -> tuple[float, float]:
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
        f"  {label:18} total={total:+6.1f} dBFS  {TONE_HZ:.0f}Hz={tone:+6.1f} dBFS",
        flush=True,
    )
    return total, tone


def main() -> int:
    print("=== P10S AEC speech-survival probe ===\n", flush=True)
    card = detect_card()
    set_mixer(card)
    gen_sine_wav(TONE_WAV, TONE_HZ, TONE_DUR_S, RATE, 0.5)

    print("[A] Speech-only baseline — speak normally for 5 s, NO playback.", flush=True)
    countdown("Recording")
    arecord_blocking(card, SPEECH_ONLY_WAV, CAP_DUR_S)
    print("  done.\n", flush=True)

    print("[B] Speech + duplex — speak normally for 5 s WHILE the tone plays.", flush=True)
    print("    (Same volume, same content, same distance as Phase A for a fair compare.)", flush=True)
    countdown("Recording + tone")
    rec = arecord_background(card, SPEECH_DUPLEX_WAV, CAP_DUR_S)
    time.sleep(WARMUP_S)
    play = aplay_background(card, TONE_WAV)
    rec.wait()
    if rec.returncode != 0:
        err = rec.stderr.read().decode("utf-8", "replace").strip() if rec.stderr else ""
        print(f"  ERROR: arecord exit {rec.returncode}: {err}", flush=True)
        play.terminate()
        return 1
    play.wait()
    print("  done.\n", flush=True)

    print("[Analysis]", flush=True)
    try:
        sil_total, sil_tone = analyze(SILENCE_WAV, "silence (prior)")
    except FileNotFoundError:
        sil_total = float("-inf")
        sil_tone = float("-inf")
        print("  silence (prior): n/a (no prior /tmp/p10s_aec_silence.wav)", flush=True)
    a_total, a_tone = analyze(SPEECH_ONLY_WAV, "speech only")
    b_total, b_tone = analyze(SPEECH_DUPLEX_WAV, "speech + duplex")

    delta_speech = b_total - a_total
    delta_tone_vs_silence = (b_tone - sil_tone) if sil_tone != float("-inf") else float("nan")
    print(f"\n  Δ speech total RMS (B − A) = {delta_speech:+5.1f} dB", flush=True)
    if sil_tone != float("-inf"):
        print(f"  Δ {TONE_HZ:.0f} Hz vs silence (B − sil) = {delta_tone_vs_silence:+5.1f} dB  "
              f"(sanity: should stay small if AEC still active)", flush=True)

    print("\n=== Verdict ===", flush=True)
    if abs(delta_speech) < 6:
        verdict = "AEC IS SELECTIVE — speech survives during playback."
        impl = (
            "v2 implication: barge-in / continuous wake-word during TTS is\n"
            "viable. The firmware AEC suppresses only its own playback echo,\n"
            "not human speech."
        )
    elif delta_speech < -15:
        verdict = "AEC ALSO ATTENUATES SPEECH — half-duplex stays mandatory."
        impl = (
            "v2 implication: even with firmware AEC, the mic stream is\n"
            "heavily attenuated during playback. v2 must keep the v1\n"
            "half-duplex rule (mute mic during TTS) or add software AEC\n"
            "with a richer ducking/comfort-noise model."
        )
    else:
        verdict = "PARTIAL — speech degraded but not killed."
        impl = (
            f"v2 implication: judgment call. Speech survives at "
            f"Δ={delta_speech:+.1f} dB. Retry the test with louder voice or\n"
            "different content (e.g., counting) before committing to v2\n"
            "barge-in. If retest stays in this band, treat as 'usable but\n"
            "degraded' and apply a noise-gate on the wake-word path."
        )
    print(verdict, flush=True)
    print(impl, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
