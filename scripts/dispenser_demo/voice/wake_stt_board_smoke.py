#!/usr/bin/env python3
"""Phase 3 Layer B smoke — SL2619 board wake (Hey Jarvis) → VAD → Moonshine STT.

Pure-stdlib + numpy + onnxruntime; runs the upstream openWakeWord
package vendored under `/mnt/sdcard/python-deps/site/openwakeword/`.
openWakeWord's `__init__.py` eagerly imports `train_custom_verifier`,
which transitively needs scipy + sklearn + tqdm + requests — none of
which are installable on the stripped Yocto board image. We stub those
four modules at process start; `Model.predict()` + `VAD.predict()` never
touch them at runtime, only at import time.

One-shot: arms the wake listener, captures the first `Hey Jarvis` →
post-wake utterance, decodes via CrispASR + Moonshine, prints the
transcript on stdout, exits. Loop variant deliberately omitted (see
2026-05-12 decisions-log "Phase 3 smoke topology" entry).

Expected on-board layout (deploy paths bound to defaults below):

  /mnt/sdcard/python-deps/site/onnxruntime/                      (12 MB ext'd wheel)
  /mnt/sdcard/python-deps/site/openwakeword/                     (vendored .py)
  /mnt/sdcard/python-deps/site/openwakeword/resources/models/    (4 ONNX, 5.3 MB)
  /mnt/sdcard/dispenser_demo/wake_stt_board_smoke.py             (this file)
  /mnt/sdcard/bin/crispasr                                       (existing aarch64)
  /mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf     (existing)

Run on board:

  PYTHONPATH=/mnt/sdcard/python-deps/site python3 \\
      /mnt/sdcard/dispenser_demo/wake_stt_board_smoke.py
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Module-stub gate: must run before `import openwakeword`. openWakeWord's
# top-level __init__.py imports `custom_verifier_model.train_custom_verifier`
# unconditionally; that module imports scipy + sklearn + tqdm at file-load
# time. None of those are on the board image, and none are touched by the
# Model / VAD predict paths we actually exercise. Wire fake modules so the
# import chain succeeds; runtime behaviour is unaffected.
# ----------------------------------------------------------------------------
import sys
import types


def _install_module_stubs() -> None:
    class _FakeTqdm:
        def __init__(self, it=None, *a, **kw): self.it = it
        def __iter__(self): return iter(self.it) if self.it is not None else iter([])
        def update(self, *a, **kw): pass
        def close(self): pass
        def set_description(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    for name in (
        "tqdm", "requests", "scipy",
        "sklearn", "sklearn.linear_model",
        "sklearn.pipeline", "sklearn.preprocessing",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    sys.modules["tqdm"].tqdm = _FakeTqdm  # type: ignore[attr-defined]
    sys.modules["sklearn.linear_model"].LogisticRegression = type(  # type: ignore[attr-defined]
        "LogisticRegression", (), {})
    sys.modules["sklearn.pipeline"].make_pipeline = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["sklearn.preprocessing"].FunctionTransformer = type(  # type: ignore[attr-defined]
        "FunctionTransformer", (), {})
    sys.modules["sklearn.preprocessing"].StandardScaler = type(  # type: ignore[attr-defined]
        "StandardScaler", (), {})


_install_module_stubs()

# ----------------------------------------------------------------------------

import argparse
import contextlib
import logging
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

# Importing openwakeword pulls onnxruntime transitively; do AFTER the stubs.
from openwakeword.model import Model  # noqa: E402
from openwakeword.vad import VAD  # noqa: E402

# ----------------------------------------------------------------------------

# Force UTF-8 stdio — Yocto images often default to C/POSIX locale. Mirrors
# the chat_board.py pattern; without it, llama.cpp / crispasr non-ASCII output
# crashes the surrounding Python script with a UnicodeDecodeError.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# region: defaults — bound to the 2026-05-12 board deployment

# arecord captures from the P10S USB mic enumerated as hw:1,0 (S16_LE @ 48 kHz
# stereo native — see /board_probe 2026-05-12). `plughw:` lets ALSA's plug
# layer resample to 16 kHz mono internally — same path used by
# `docs/guides/usb-audio-testing-sl2619.md`. No numpy resampling needed.
DEFAULT_ALSA_DEVICE = "plughw:1,0"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CRISPASR = Path("/mnt/sdcard/bin/crispasr")
DEFAULT_MOONSHINE = Path("/mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf")

# openWakeWord native frame: exactly 1280 int16 samples (80 ms @ 16 kHz).
# Anything else gets accumulated/truncated internally; matching is cheapest.
OWW_FRAME_SAMPLES = 1280
# Silero VAD native frame: 480 int16 samples (30 ms @ 16 kHz) — see
# openwakeword/vad.py:`predict(..., frame_size=480)`.
VAD_FRAME_SAMPLES = 480

# Wake threshold. openWakeWord docs land at 0.5 for hey_jarvis_v0.1 in
# quiet environments; bump toward 0.6–0.7 if FPR climbs in noise.
DEFAULT_WAKE_THRESHOLD = 0.5
# Patience: require 2 consecutive 80-ms frames above threshold before
# declaring WAKE. Cuts spurious single-frame triggers without adding
# meaningful latency (~160 ms cost).
DEFAULT_WAKE_PATIENCE = 2

# VAD threshold + hangover. Hangover = post-speech sub-threshold frames
# required to declare speech-end. 13 frames × 30 ms = 390 ms is enough to
# survive normal consonant tail-offs without making the user wait. Hard
# cap on LISTENING wall is the safety belt against VAD never converging
# (noisy room, trailing voice).
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_HANGOVER_FRAMES = 13
DEFAULT_LISTEN_MAX_S = 5.0
# Pre-wake roll-back: also include the audio captured slightly BEFORE the
# wake fired. The model usually triggers ~200 ms into "Hey Jarvis"; without
# this, "...Jarvis" gets included as the start of the utterance and skews
# Silero's speech-start detection. 240 ms = 4 wake-frames is empirically
# enough rollback to start cleanly after the wake-word tail.
DEFAULT_PREWAKE_ROLLBACK_FRAMES = 0
# Minimum LISTENING duration before VAD endpointing can fire. Even with
# a fast speaker, a 300 ms floor prevents the VAD from declaring
# speech-end on the silence between "Hey Jarvis" and the user's utterance.
DEFAULT_LISTEN_MIN_S = 0.3

WAKE_MODEL_NAME = "hey_jarvis"

# endregion

# region: logging

log = logging.getLogger("wake_stt_smoke")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

# endregion

# region: arecord wrapper


class ArecordMic:
    """Continuous mic capture as a generator of int16 1-D numpy arrays.

    Spawns `arecord -D <dev> -f S16_LE -r 16000 -c 1 -t raw` and reads its
    stdout in fixed-size byte chunks. ALSA's `plughw:` plug layer handles
    48 kHz → 16 kHz resampling and stereo → mono downmix internally, so the
    Python side stays a pure int16 PCM consumer.
    """

    def __init__(
        self,
        device: str,
        sample_rate: int,
        chunk_samples: int,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.chunk_bytes = chunk_samples * 2  # int16 = 2 bytes/sample
        self.verbose = verbose
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "ArecordMic":
        argv = [
            "arecord",
            "-D", self.device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self.sample_rate),
            "-t", "raw",
            # -q suppresses arecord's startup banner; keeps stderr clean
            # so we don't mistake it for an error in the smoke output.
            "-q",
        ]
        log.debug("starting arecord: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if not self.verbose else None,
            bufsize=0,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def chunks(self) -> "Iterable[np.ndarray]":  # noqa: F821
        """Yield int16 mono chunks of exactly `chunk_samples` per yield."""
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        target = self.chunk_bytes
        while True:
            need = target - len(buf)
            if need > 0:
                more = self._proc.stdout.read(need)
                if not more:
                    # arecord died — surface stderr for the operator.
                    err = b""
                    if self._proc.stderr is not None:
                        err = self._proc.stderr.read() or b""
                    raise RuntimeError(
                        f"arecord exited unexpectedly: {err.decode(errors='replace')[-300:]}"
                    )
                buf += more
                continue
            chunk = np.frombuffer(buf[:target], dtype=np.int16).copy()
            buf = buf[target:]
            yield chunk

# endregion

# region: STT subprocess


# CrispASR's stdout framing isn't pinned in upstream docs; this regex skips
# obvious log noise and takes the first non-empty text line as the transcript.
# Mirrors `scripts/dispenser_demo/spike/crispasr_host_smoke.py:_LOG_PREFIX_RE`.
_TRANSCRIPT_NOISE_RE = re.compile(
    r"^(?:\[[\w\d:.\- ]+\]|whisper_|crispasr_|ggml_|moonshine_)"
)


def write_wav_16k_mono(samples: np.ndarray, path: Path) -> None:
    """Write 1-D int16 array to a 16 kHz mono PCM WAV (Moonshine input format)."""
    if samples.dtype != np.int16:
        raise ValueError(f"expected int16, got {samples.dtype}")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(DEFAULT_SAMPLE_RATE)
        w.writeframes(samples.tobytes())


def crispasr_decode(
    bin_path: Path,
    model_path: Path,
    wav_path: Path,
    threads: int,
    verbose: bool = False,
) -> tuple[str, float]:
    """Run crispasr-cli on a 16 kHz mono WAV; return (transcript, wall_s).

    Pinned flags from `docs/plans/dispenser-demo/decisions-log.md` — the
    `-l en --no-punctuation -t 2` triplet prevents the auto-LID and
    auto-punctuation network fetches that are fatal on the offline SL2619.
    """
    argv = [
        str(bin_path),
        "--backend", "moonshine",
        "-l", "en",
        "--no-punctuation",
        "-t", str(threads),
        "-m", str(model_path),
        "-f", str(wav_path),
    ]
    log.debug("crispasr: %s", " ".join(argv))
    t0 = time.perf_counter()
    proc = subprocess.run(
        argv,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    wall = time.perf_counter() - t0
    if verbose:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"crispasr exit {proc.returncode}: {proc.stderr[-300:]!r}"
        )

    transcript = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or _TRANSCRIPT_NOISE_RE.match(line):
            continue
        transcript = line
        break
    return transcript, wall

# endregion

# region: main loop


def run_smoke(args: argparse.Namespace) -> int:
    log.info(
        "loading openWakeWord (model=%s, threshold=%.2f, patience=%d)",
        WAKE_MODEL_NAME, args.wake_threshold, args.wake_patience,
    )
    t0 = time.perf_counter()
    # vad_threshold=0 disables openwakeword's internal VAD filter on wake
    # predictions — we want raw scores; the SEPARATE VAD instance below
    # handles post-wake endpointing.
    wake_model = Model(
        wakeword_models=[WAKE_MODEL_NAME],
        inference_framework="onnx",
        vad_threshold=0.0,
    )
    vad = VAD()
    load_s = time.perf_counter() - t0
    log.info("loaded in %.2f s", load_s)

    # openWakeWord zeroes its first 5 predict() outputs during model init
    # (see openwakeword/model.py:332 — `if len(prediction_buffer[cls]) < 5:
    # predictions[cls] = 0.0`). That's a 5 × 80 ms = 400 ms window after
    # startup during which wake CANNOT fire. Prime with silence here so the
    # next "listening" message reflects when the model can actually trigger.
    silent = np.zeros(OWW_FRAME_SAMPLES, dtype=np.int16)
    for _ in range(5):
        wake_model.predict(silent)
    log.info("wake model primed (5 silent frames)")

    consecutive_above = 0
    wake_state = "IDLE"
    listen_buf: list[np.ndarray] = []
    prewake_ring: list[np.ndarray] = []
    listen_t0 = 0.0
    vad_speech_seen = False
    vad_silence_run = 0

    with ArecordMic(args.device, DEFAULT_SAMPLE_RATE, OWW_FRAME_SAMPLES,
                    verbose=args.verbose) as mic:
        log.info("listening on %s (say '%s' followed by the command)",
                 args.device, args.wake_phrase)
        for chunk in mic.chunks():
            if wake_state == "IDLE":
                scores = wake_model.predict(chunk)
                score = float(scores.get(WAKE_MODEL_NAME, 0.0))
                if score >= args.wake_threshold:
                    consecutive_above += 1
                    log.debug("wake score %.3f (run=%d)", score, consecutive_above)
                else:
                    consecutive_above = 0
                # Keep a rolling pre-wake buffer so the post-wake utterance
                # can include trailing wake-word audio if rollback > 0.
                prewake_ring.append(chunk)
                if len(prewake_ring) > max(1, args.prewake_rollback_frames):
                    prewake_ring.pop(0)

                if consecutive_above >= args.wake_patience:
                    log.info("[WAKE] hey_jarvis score=%.3f", score)
                    wake_state = "LISTENING"
                    # Reset wake model so its score state doesn't keep
                    # firing during the utterance window.
                    wake_model.reset()
                    vad.reset_states()
                    consecutive_above = 0
                    listen_buf = list(prewake_ring) if args.prewake_rollback_frames > 0 else []
                    prewake_ring = []
                    listen_t0 = time.perf_counter()
                    vad_speech_seen = False
                    vad_silence_run = 0
                continue

            # LISTENING: keep buffering; run VAD on consecutive 480-sample
            # sub-frames of the 1280-sample chunk (1280 / 480 = 2 full
            # sub-frames + 320-sample remainder we hold over to next round).
            listen_buf.append(chunk)
            elapsed = time.perf_counter() - listen_t0

            # Run VAD per 480-sample sub-frame using a small rolling
            # remainder buffer keyed off the cumulative listen_buf tail.
            # Simplest correct: concatenate, slice in 480-sample increments,
            # remember what's leftover.
            # NOTE: `_vad_consumed` is parked on `run_smoke` as a function
            # attribute — fine for one-shot, but if this function is ever
            # called twice in the same process the cursor will carry over
            # and corrupt the second run. Acceptable for the smoke; refactor
            # to a class on first non-smoke caller.
            audio_so_far = np.concatenate(listen_buf)
            consumed = getattr(run_smoke, "_vad_consumed", 0)
            new_audio = audio_so_far[consumed:]
            n_subframes = len(new_audio) // VAD_FRAME_SAMPLES
            for i in range(n_subframes):
                sub = new_audio[i * VAD_FRAME_SAMPLES:(i + 1) * VAD_FRAME_SAMPLES]
                vad_score = float(vad.predict(sub, frame_size=VAD_FRAME_SAMPLES))
                if vad_score >= args.vad_threshold:
                    vad_speech_seen = True
                    vad_silence_run = 0
                else:
                    vad_silence_run += 1
                log.debug("vad %.3f speech_seen=%s silence_run=%d",
                          vad_score, vad_speech_seen, vad_silence_run)
            run_smoke._vad_consumed = consumed + n_subframes * VAD_FRAME_SAMPLES  # type: ignore[attr-defined]

            # End conditions — whichever fires first wins.
            end_reason = None
            if elapsed >= args.listen_max_s:
                end_reason = f"max_listen ({args.listen_max_s:.1f} s)"
            elif (vad_speech_seen
                  and elapsed >= args.listen_min_s
                  and vad_silence_run >= args.vad_hangover_frames):
                end_reason = f"vad_end (silence_run={vad_silence_run})"

            if end_reason is None:
                continue

            captured = np.concatenate(listen_buf)
            captured_s = captured.size / DEFAULT_SAMPLE_RATE
            log.info("end of utterance — %s, captured %.2f s of audio", end_reason, captured_s)

            with tempfile.NamedTemporaryFile(
                suffix=".wav", dir="/tmp", delete=False,
            ) as tf:
                wav_path = Path(tf.name)
            try:
                write_wav_16k_mono(captured, wav_path)
                transcript, stt_wall = crispasr_decode(
                    args.crispasr_bin, args.moonshine_model, wav_path,
                    threads=args.threads, verbose=args.verbose,
                )
            finally:
                with contextlib.suppress(OSError):
                    wav_path.unlink()

            log.info("STT decode wall=%.2f s", stt_wall)
            print(f"\n[TRANSCRIPT] {transcript}\n", flush=True)
            return 0

    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", default=DEFAULT_ALSA_DEVICE,
                   help="ALSA capture device (default: %(default)s — plug layer "
                        "handles 48 kHz → 16 kHz mono).")
    p.add_argument("--wake-phrase", default="Hey Jarvis",
                   help="Cosmetic only — model is the pretrained hey_jarvis_v0.1.")
    p.add_argument("--wake-threshold", type=float, default=DEFAULT_WAKE_THRESHOLD,
                   help="Wake score threshold (default: %(default).2f).")
    p.add_argument("--wake-patience", type=int, default=DEFAULT_WAKE_PATIENCE,
                   help="Consecutive 80-ms frames above threshold required "
                        "before declaring WAKE (default: %(default)d).")
    p.add_argument("--prewake-rollback-frames", type=int,
                   default=DEFAULT_PREWAKE_ROLLBACK_FRAMES,
                   help="Pre-wake 80-ms frames included in the listening buffer "
                        "(default: %(default)d → no rollback). Bump to 4 to "
                        "include ~320 ms of pre-wake audio if Silero VAD trips "
                        "early on the wake-word tail.")
    p.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD,
                   help="Silero VAD speech probability threshold "
                        "(default: %(default).2f).")
    p.add_argument("--vad-hangover-frames", type=int, default=DEFAULT_VAD_HANGOVER_FRAMES,
                   help="Sub-threshold 30-ms frames after speech-start "
                        "before declaring speech-end (default: %(default)d → ~390 ms).")
    p.add_argument("--listen-min-s", type=float, default=DEFAULT_LISTEN_MIN_S,
                   help="Floor on LISTENING duration (default: %(default).2f s).")
    p.add_argument("--listen-max-s", type=float, default=DEFAULT_LISTEN_MAX_S,
                   help="Hard cap on LISTENING duration (default: %(default).1f s).")
    p.add_argument("--crispasr-bin", type=Path, default=DEFAULT_CRISPASR,
                   help="Path to crispasr-cli binary (default: %(default)s).")
    p.add_argument("--moonshine-model", type=Path, default=DEFAULT_MOONSHINE,
                   help="Path to moonshine GGUF (default: %(default)s).")
    p.add_argument("--threads", type=int, default=2,
                   help="STT decode threads (default: %(default)d — match A55 core count).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging + tee CrispASR stderr.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)

    for f in (args.crispasr_bin, args.moonshine_model):
        if not f.exists():
            log.error("missing: %s", f)
            return 2

    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
