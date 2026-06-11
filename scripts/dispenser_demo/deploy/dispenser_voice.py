#!/usr/bin/env python3
"""Phase 3 Layer C — SL2619 board long-running voice pipeline.

Single board process. On startup: prime openWakeWord + Silero VAD, set ALSA
mixer to safe levels, resolve (and prime if missing) the FunctionGemma KV
cache. Then loop forever:

    arecord plughw:1,0 → openWakeWord ("Hey Jarvis")
                       → on WAKE: Silero VAD endpoint
                                   → temp WAV (16 kHz mono)
                                   → CrispASR (Moonshine Tiny GGUF Q4_K)
                                              → transcript
                                              → chat_board._run_turn
                                                          → dispense
                                                            override:
                                                              [BLE→ESP32]
                                                              5A A5 01 00
                                                              + canned
                                                              response
                                                            OR real tool
                                                            dispatch
                       → reset, loop

The dispense override (`get_medications_at_time` /
`get_medication_by_name` → §6 dispense intent) is picked up by *importing*
`chat_board_dispense` — that module monkey-patches `chat_board.dispatch` and
`chat_board.format_response` at import time. No fork of chat_board needed.

Layer C folds the two Layer-B gaps recorded in
`docs/plans/dispenser-demo/decisions-log.md` (Phase 3 Layer B closed entry):

  1. **Speech-relative listen cap** — 5 s wall ticks from the first
     `speech_seen=True` VAD frame, not from the wake fire. Prevents the
     empty-transcript failure if the speaker pauses.
  2. **arecord SIGTERM stderr suppressed** — the benign
     "pcm_read: Interrupted system call" line on shutdown is captured + dropped.

R3: this script runs ON the board (not via agent SSH). Deploy via scp; the
agent never writes to the board.

Expected on-board layout (defaults bound below):

    /mnt/sdcard/python-deps/site/{onnxruntime,openwakeword}/   (Layer-B stage)
    /mnt/sdcard/models/functiongemma-270m/                     (FunctionGemma deploy)
        chat_board.py, chat_board_dispense.py,
        finetuned_functiongemma_q4_0.gguf,
        prompt-prefix.txt, prompt-suffix.txt,
        health_table.json, health_table_dispense.json
    /mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf (STT)
    /mnt/sdcard/bin/crispasr                                   (STT engine)
    /mnt/sdcard/dispenser_demo/dispenser_voice.py              (this file)

Run on board:

    PYTHONPATH=/mnt/sdcard/python-deps/site python3 \\
        /mnt/sdcard/dispenser_demo/dispenser_voice.py
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Stub gate — must run before openwakeword import. See wake_stt_board_smoke.py
# for the same rationale (openwakeword/__init__.py eagerly imports
# custom_verifier_model which transitively pulls scipy + sklearn + tqdm +
# requests; none on board image, none touched at runtime).
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
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import onnxruntime as _ort  # noqa: F401 — assert presence early
from openwakeword.model import Model  # noqa: E402
from openwakeword.vad import VAD  # noqa: E402

# ----------------------------------------------------------------------------

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# region: defaults

# Audio pipeline — same as wake_stt_board_smoke.py.
DEFAULT_ALSA_DEVICE = "plughw:1,0"
DEFAULT_SAMPLE_RATE = 16000
OWW_FRAME_SAMPLES = 1280
VAD_FRAME_SAMPLES = 480

# Wake-word + VAD tuning — same defaults that worked first-try on 2026-05-12.
# More sensitive than Layer B/C's first-run defaults (0.5 / 2) — those
# made the user repeat "Hey Jarvis" because patience=2 needed two
# consecutive 80-ms frames above threshold; quick phrasing only produces
# one peak frame. patience=1 + a slightly lower threshold trades a small
# FPR uptick for clean single-shot wake. Tunable per `--wake-threshold` /
# `--wake-patience` if a noisy room produces phantom wakes.
DEFAULT_WAKE_THRESHOLD = 0.45
DEFAULT_WAKE_PATIENCE = 1
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_HANGOVER_FRAMES = 13
# Speech-relative LISTENING cap — wall from first speech_seen=True frame,
# NOT from wake fire (Layer B used wake-relative which would empty-transcript
# if the speaker paused). 5 s is enough for ~2-3 sentence utterances.
DEFAULT_SPEECH_MAX_S = 5.0
# Absolute LISTENING cap — wall from wake fire. Higher than speech_max_s
# because we want to give the speaker time to start. If this fires without
# any speech, we abort the utterance, skip STT, and return to IDLE.
DEFAULT_LISTEN_ABS_MAX_S = 8.0
DEFAULT_PREWAKE_ROLLBACK_FRAMES = 0
WAKE_MODEL_NAME = "hey_jarvis"

# FunctionGemma — board paths from chat_board.py + chat_board_dispense.py.
DEFAULT_FG_MODEL_DIR = Path("/mnt/sdcard/models/functiongemma-270m")
DEFAULT_FG_MODEL_NAME = "finetuned_functiongemma_q4_0.gguf"
DEFAULT_FG_HEALTH_TABLE_NAME = "health_table_dispense.json"
DEFAULT_FG_LLAMA = Path("/mnt/sdcard/llama-cpp/llama-completion")
DEFAULT_FG_THREADS = 2
DEFAULT_FG_CTX_SIZE = 2048
DEFAULT_FG_MAX_NEW_TOKENS = 64
DEFAULT_FG_TEMP = 0.0
DEFAULT_FG_TOP_K = 1
DEFAULT_FG_SEED = 42

# STT — host paths confirmed on board.
DEFAULT_CRISPASR = Path("/mnt/sdcard/bin/crispasr")
DEFAULT_MOONSHINE = Path("/mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf")

# Mixer pre-set — Phase 3.1 mandate from plan §9.3.
DEFAULT_MIXER_CARD = 1
DEFAULT_MIC_PERCENT = 70
DEFAULT_PCM_PERCENT = 50

# Layer D — TTS settings.
#
# Default: **dynamic Piper render** of the LLM's formatted response text.
# Captures chat_board.format_response output via a side-effect wrapper
# (similar to parse_function_calls capture for the tool name), pipes
# through PiperVoice → WAV → aplay. Each turn renders fresh so the
# spoken answer matches what stdout shows.
#
# Fallback: canned WAV by tool name from <tts_dir>/<tool>.wav (Layer D
# v1 behavior) — used when --no-dynamic-tts is set OR Piper load /
# render fails on board.
DEFAULT_TTS_DIR = Path("/mnt/sdcard/dispenser_demo/tts")
DEFAULT_APLAY_DEVICE = "plughw:1,0"
DEFAULT_PIPER_MODEL = Path("/mnt/sdcard/dispenser_demo/piper-voices/en_US-lessac-medium.onnx")
# Wake-acknowledgement chime — short two-tone (~170 ms) played immediately
# after WAKE fires so the user knows the box transitioned to LISTENING.
# Generated host-side; not synthesized on board.
DEFAULT_WAKE_ACK_WAV = Path("/mnt/sdcard/dispenser_demo/wake_ack.wav")
# Command-received chime — distinct single-tone (~120 ms, lower pitch) played
# after STT decodes a transcript but BEFORE the LLM turn. Tells the user
# "I heard you" so the long (~6 s) LLM wall doesn't feel like a hang. Played
# AFTER `mic.pause()` would normally apply, but kept simple here: aplay is
# short enough that the arecord pipe survives without explicit half-duplex.
DEFAULT_COMMAND_ACK_WAV = Path("/mnt/sdcard/dispenser_demo/command_ack.wav")

# endregion

# region: logging

log = logging.getLogger("dispenser_voice")
# Per-frame trace logger — wake-score / VAD-score lines fire per 80 ms or 30 ms
# audio frame, so a couple seconds of LISTENING produces 50+ lines under -v.
# Sink them on a separate logger that defaults to WARNING (silent); flip ON
# with --trace when actually debugging frame-level behavior.
trace_log = logging.getLogger("dispenser_voice.trace")


def _setup_logging(verbose: bool, trace: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # The `.trace` child inherits the root configuration above; override its
    # level explicitly so per-frame messages are gated on --trace, not -v.
    trace_log.setLevel(logging.DEBUG if trace else logging.WARNING)
    trace_log.propagate = True

# endregion

# region: arecord wrapper — Layer C version (drops the benign SIGTERM stderr)


class ArecordMic:
    """Continuous 16 kHz mono int16 mic capture via arecord subprocess.

    Identical to the Layer B version except `__exit__` swallows arecord's
    stderr on shutdown — the SIGTERM-during-read produces a benign
    "pcm_read: Interrupted system call" line that's noise in a long-running
    pipeline.
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
        self.chunk_bytes = chunk_samples * 2
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
            "-q",
        ]
        log.debug("starting arecord: %s", " ".join(argv))
        # Capture stderr unconditionally so SIGTERM noise is contained.
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        return self

    def pause(self) -> None:
        """SIGSTOP arecord — half-duplex Layer D playback boundary.

        Per plan §8.3 E2: mute mic during any playback. SIGSTOP is the
        cheapest "mute" — ALSA buffer continues filling kernel-side, but
        no userspace bytes are produced. SIGCONT resumes; the resume may
        emit a buffer-overrun line (drained by the caller).
        """
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGSTOP)

    def resume(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGCONT)

    def drain(self, max_seconds: float = 6.0) -> int:
        """Read + discard up to `max_seconds` of pending pipe audio.

        Use after any blocking gap that kept the consumer side idle (LLM
        turn, TTS playback, SIGSTOP/SIGCONT) — those leave stale audio
        queued in arecord's stdout pipe + ALSA capture buffer that would
        otherwise be consumed as if it were live mic input. Without this:
        next wake listener processes seconds-old audio (phantom-wake
        risk, latency spike).

        Uses `os.set_blocking` (Py 3.5+ stdlib, no `fcntl` Python module
        required — Yocto's stripped image lacks `fcntl`). Restores
        blocking mode in `finally` so the subsequent `chunks()` reader
        keeps the same semantics it had before the drain.

        Returns bytes drained. The cap is generous (6 s ≈ longer than
        the LLM turn + TTS WAV combined) but capped to prevent a
        runaway drain if something is broken upstream.
        """
        if self._proc is None or self._proc.stdout is None:
            return 0
        max_bytes = int(self.sample_rate * 2 * max_seconds)  # int16 mono
        fd = self._proc.stdout.fileno()
        os.set_blocking(fd, False)
        drained = 0
        try:
            while drained < max_bytes:
                try:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    drained += len(chunk)
                except BlockingIOError:
                    break
        finally:
            os.set_blocking(fd, True)
        if drained:
            log.debug("arecord drained %d bytes (~%.2f s stale audio)",
                      drained, drained / 2 / self.sample_rate)
        return drained

    def __exit__(self, *_exc: object) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        # Drain stderr silently. Surface only if --verbose AND non-trivial
        # (i.e. ignoring the SIGTERM-interrupt line).
        if self._proc.stderr is not None:
            err = self._proc.stderr.read() or b""
            if self.verbose and err:
                text = err.decode(errors="replace").strip()
                # Strip the SIGTERM-induced line if it's the only thing there.
                meaningful = "\n".join(
                    line for line in text.splitlines()
                    if "Interrupted system call" not in line
                ).strip()
                if meaningful:
                    log.debug("arecord stderr: %s", meaningful)
        self._proc = None

    def chunks(self) -> Iterable[np.ndarray]:
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        target = self.chunk_bytes
        while True:
            need = target - len(buf)
            if need > 0:
                more = self._proc.stdout.read(need)
                if not more:
                    # arecord died mid-loop — surface real stderr.
                    err = b""
                    if self._proc.stderr is not None:
                        err = self._proc.stderr.read() or b""
                    raise RuntimeError(
                        "arecord exited unexpectedly: "
                        f"{err.decode(errors='replace')[-300:]}"
                    )
                buf += more
                continue
            chunk = np.frombuffer(buf[:target], dtype=np.int16).copy()
            buf = buf[target:]
            yield chunk

# endregion

# region: STT


_TRANSCRIPT_NOISE_RE = re.compile(
    r"^(?:\[[\w\d:.\- ]+\]|whisper_|crispasr_|ggml_|moonshine_)"
)


def write_wav_16k_mono(samples: np.ndarray, path: Path) -> None:
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
        argv, capture_output=True,
        encoding="utf-8", errors="replace", check=False,
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

# region: TTS playback — canned WAV per tool, half-duplex via SIGSTOP/CONT


def render_piper(voice: Any, text: str, sample_rate_attr: str = "sample_rate") -> Path:
    """Render `text` to a temp 22 kHz mono WAV via Piper. Caller unlinks.

    Uses the streaming `voice.synthesize(text)` iterator so longer texts
    don't OOM the wave buffer. The board's A55 cores render at roughly
    0.5–1 s per sentence (~3–5× slower than host's 0.19 s smoke).
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir="/tmp", delete=False,
    ) as tf:
        wav_path = Path(tf.name)
    sr = getattr(voice.config, sample_rate_attr, 22050)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for chunk in voice.synthesize(text):
            w.writeframes(chunk.audio_int16_bytes)
    return wav_path


def play_wav(
    wav_path: Path,
    device: str,
    mic: ArecordMic | None,
    verbose: bool = False,
) -> float:
    """aplay a WAV with arecord paused; return playback wall.

    `mic` is the live ArecordMic — paused via SIGSTOP before aplay starts
    and resumed via SIGCONT after aplay exits (plan §8.3 E2 half-duplex).
    On resume, ALSA will log an `overrun!!!` line to arecord stderr; that
    is the expected cost of the pause and is filtered in `ArecordMic.__exit__`.
    """
    argv = ["aplay", "-q", "-D", device, str(wav_path)]
    log.debug("aplay: %s", " ".join(argv))
    if mic is not None:
        mic.pause()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, capture_output=True,
            encoding="utf-8", errors="replace", check=False,
        )
    finally:
        if mic is not None:
            mic.resume()
    wall = time.perf_counter() - t0
    if verbose and proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        # Not fatal — TTS playback failure shouldn't kill the loop. Log + continue.
        log.warning("aplay exit %d: %s", proc.returncode,
                    proc.stderr[-200:].strip())
    return wall

# endregion

# region: mixer


def set_mixer_levels(card: int, mic_pct: int, pcm_pct: int) -> None:
    """Set ALSA Mic + PCM percent on the USB sound card.

    Per `docs/guides/usb-audio-testing-sl2619.md` (verified 2026-05-11), the
    P10S can come up with PCM at 0% (silent-success trap) and an unset Mic
    capture level. We force both on every startup; cost is two amixer
    invocations (~50 ms).
    """
    for control, pct in [("Mic", mic_pct), ("PCM", pcm_pct)]:
        try:
            subprocess.run(
                ["amixer", "-c", str(card), "sset", control, f"{pct}%"],
                check=False, capture_output=True, timeout=2.0,
            )
            log.info("mixer: card=%d %s=%d%%", card, control, pct)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("mixer pre-set failed for %s: %s", control, exc)

# endregion

# region: LLM glue — import chat_board_dispense from the FunctionGemma deploy dir


def _load_chat_board(fg_dir: Path) -> tuple[Any, Any]:
    """Add fg_dir to sys.path and import chat_board + chat_board_dispense.

    chat_board_dispense.py monkey-patches chat_board.dispatch and
    chat_board.format_response at import time — so importing it (for the
    side effect) is what wires up the dispense intent override. We do NOT
    call its main(); we use `chat_board._run_turn` directly with the
    overrides already installed.

    Two additional capture wrappers go on top of that:

    - `parse_function_calls` → captures `chat_board._last_tool` so Layer D
      can pick the right canned WAV without depending on `_run_turn`'s
      return value (which is `""` on some on-board revisions).
    - `format_response` → captures `chat_board._last_formatted` so Layer D
      dynamic-TTS can render exactly what stdout shows, rather than a
      tool-keyed canned WAV that may not match the user's question
      (e.g. "What's my BP?" → chat_board emits a BP-only sentence but
      the get_vitals canned WAV speaks the entire vitals dump).
    """
    sys.path.insert(0, str(fg_dir))
    # Side-effect import: patches chat_board's dispatch + format_response.
    import chat_board_dispense  # noqa: F401
    import chat_board  # noqa: E402

    chat_board._last_tool = None  # type: ignore[attr-defined]
    chat_board._last_formatted = None  # type: ignore[attr-defined]
    _orig_parse = chat_board.parse_function_calls
    _orig_format = chat_board.format_response

    def _capture_parse(text: str) -> list:
        calls = _orig_parse(text)
        if calls:
            chat_board._last_tool = calls[0]["tool"]  # type: ignore[attr-defined]
        return calls

    def _capture_format(user_text: str, tool: str, args_: dict, result: Any) -> str:
        out = _orig_format(user_text, tool, args_, result)
        chat_board._last_formatted = out  # type: ignore[attr-defined]
        return out

    chat_board.parse_function_calls = _capture_parse
    chat_board.format_response = _capture_format
    return chat_board, chat_board_dispense

# endregion

# region: utterance capture


def capture_utterance(
    mic: ArecordMic,
    wake_model: Any,
    vad: VAD,
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Listen for one wake → utterance cycle. Return captured int16 audio
    at 16 kHz mono, or None if the absolute LISTENING cap fires without
    speech (operator should retry).
    """
    consecutive_above = 0
    state = "IDLE"
    listen_buf: list[np.ndarray] = []
    prewake_ring: list[np.ndarray] = []
    wake_t0 = 0.0
    speech_t0: float | None = None
    vad_speech_seen = False
    vad_silence_run = 0
    vad_consumed = 0

    log.info("listening on %s (say '%s' followed by the command)",
             args.device, args.wake_phrase)

    for chunk in mic.chunks():
        if state == "IDLE":
            scores = wake_model.predict(chunk)
            score = float(scores.get(WAKE_MODEL_NAME, 0.0))
            if score >= args.wake_threshold:
                consecutive_above += 1
                trace_log.debug("wake score %.3f (run=%d)", score, consecutive_above)
            else:
                consecutive_above = 0
            prewake_ring.append(chunk)
            if len(prewake_ring) > max(1, args.prewake_rollback_frames):
                prewake_ring.pop(0)
            if consecutive_above >= args.wake_patience:
                log.info("[WAKE] hey_jarvis score=%.3f", score)

                # Layer D: wake-acknowledgement chime. Plays WITHOUT pausing
                # arecord — the chime is ~170 ms, far shorter than typical
                # user reaction time (~500 ms), so the user hasn't started
                # speaking yet. We drain the chime audio out of the mic
                # pipe AFTER the chime + before VAD starts, so VAD doesn't
                # trip on the tone tail.
                if not args.no_wake_ack and args.wake_ack_wav.exists():
                    try:
                        t_ack = time.perf_counter()
                        subprocess.run(
                            ["aplay", "-q", "-D", args.aplay_device,
                             str(args.wake_ack_wav)],
                            check=False, capture_output=True, timeout=2.0,
                        )
                        log.debug("wake-ack chime wall=%.2f s",
                                  time.perf_counter() - t_ack)
                    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                        log.warning("wake-ack chime failed: %s", exc)

                state = "LISTENING"
                wake_model.reset()
                vad.reset_states()
                consecutive_above = 0
                listen_buf = list(prewake_ring) if args.prewake_rollback_frames > 0 else []
                prewake_ring = []
                wake_t0 = time.perf_counter()
                speech_t0 = None
                vad_speech_seen = False
                vad_silence_run = 0
                vad_consumed = 0
                # Drain the chime tail (and any pipe latency from aplay)
                # so VAD starts clean on the user's actual command.
                mic.drain(max_seconds=1.0)
            continue

        # LISTENING
        listen_buf.append(chunk)
        audio_so_far = np.concatenate(listen_buf)
        new_audio = audio_so_far[vad_consumed:]
        n_sub = len(new_audio) // VAD_FRAME_SAMPLES
        for i in range(n_sub):
            sub = new_audio[i * VAD_FRAME_SAMPLES:(i + 1) * VAD_FRAME_SAMPLES]
            vad_score = float(vad.predict(sub, frame_size=VAD_FRAME_SAMPLES))
            if vad_score >= args.vad_threshold:
                if not vad_speech_seen:
                    speech_t0 = time.perf_counter()
                vad_speech_seen = True
                vad_silence_run = 0
            else:
                vad_silence_run += 1
            trace_log.debug("vad %.3f speech_seen=%s silence_run=%d",
                            vad_score, vad_speech_seen, vad_silence_run)
        vad_consumed += n_sub * VAD_FRAME_SAMPLES

        wake_elapsed = time.perf_counter() - wake_t0
        speech_elapsed = (time.perf_counter() - speech_t0) if speech_t0 else 0.0

        end_reason = None
        if vad_speech_seen and vad_silence_run >= args.vad_hangover_frames:
            end_reason = f"vad_end (silence_run={vad_silence_run})"
        elif vad_speech_seen and speech_elapsed >= args.speech_max_s:
            end_reason = f"speech_max ({args.speech_max_s:.1f} s)"
        elif not vad_speech_seen and wake_elapsed >= args.listen_abs_max_s:
            log.warning("LISTENING absolute cap with no speech — aborting utterance")
            return None

        if end_reason is None:
            continue

        captured = np.concatenate(listen_buf)
        log.info("end of utterance — %s, captured %.2f s of audio",
                 end_reason, captured.size / DEFAULT_SAMPLE_RATE)
        return captured

    return None  # mic stream ended (shouldn't happen)

# endregion

# region: main loop


def run_voice_loop(args: argparse.Namespace) -> int:
    # ----- load openWakeWord + Silero VAD -----
    log.info("loading openWakeWord (model=%s, threshold=%.2f, patience=%d)",
             WAKE_MODEL_NAME, args.wake_threshold, args.wake_patience)
    t0 = time.perf_counter()
    wake_model = Model(
        wakeword_models=[WAKE_MODEL_NAME],
        inference_framework="onnx",
        vad_threshold=0.0,
    )
    vad = VAD()
    log.info("oww + VAD loaded in %.2f s", time.perf_counter() - t0)

    # Prime with 5 silent frames — covers oww's documented score-zeroing
    # window (openwakeword/model.py:332).
    silent = np.zeros(OWW_FRAME_SAMPLES, dtype=np.int16)
    for _ in range(5):
        wake_model.predict(silent)
    log.debug("wake model primed (5 silent frames)")

    # ----- Layer D dynamic TTS: load Piper voice -----
    piper_voice = None
    if not args.no_dynamic_tts:
        try:
            t0 = time.perf_counter()
            from piper import PiperVoice  # type: ignore[import-not-found]
            piper_voice = PiperVoice.load(
                str(args.piper_model),
                config_path=str(args.piper_model) + ".json",
            )
            log.info("Piper voice loaded in %.2f s (sample_rate=%d)",
                     time.perf_counter() - t0, piper_voice.config.sample_rate)
        except Exception as exc:
            log.warning("Piper load failed (%s) — falling back to canned WAVs", exc)
            piper_voice = None

    # ----- mixer + FunctionGemma -----
    if args.no_mixer:
        log.info("mixer pre-set skipped (--no-mixer)")
    else:
        set_mixer_levels(args.mixer_card, args.mic_percent, args.pcm_percent)

    log.info("loading FunctionGemma context (dir=%s, model=%s, table=%s)",
             args.fg_dir, args.fg_model_name, args.fg_health_table_name)
    chat_board, chat_board_dispense = _load_chat_board(args.fg_dir)

    prefix_path = args.fg_dir / "prompt-prefix.txt"
    suffix_path = args.fg_dir / "prompt-suffix.txt"
    table_path = args.fg_dir / args.fg_health_table_name
    fg_model_path = args.fg_dir / args.fg_model_name
    for f in (prefix_path, suffix_path, table_path, fg_model_path, args.fg_llama,
              args.crispasr_bin, args.moonshine_model):
        if not f.exists():
            log.error("missing: %s", f)
            return 2

    prefix = prefix_path.read_text(encoding="utf-8")
    suffix = suffix_path.read_text(encoding="utf-8")
    table = json.loads(table_path.read_text(encoding="utf-8"))

    # Resolve / prime FG prompt cache. chat_board._resolve_prompt_cache
    # takes an args.Namespace with the right fields; build a minimal one.
    cache_args = SimpleNamespace(
        no_prompt_cache=args.no_prompt_cache,
        prompt_cache=args.fg_prompt_cache,
        model_name=args.fg_model_name,
        llama=args.fg_llama,
        threads=args.fg_threads,
        ctx_size=args.fg_ctx_size,
        verbose=args.verbose,
    )
    prompt_cache, cache_label = chat_board._resolve_prompt_cache(
        cache_args, prefix, fg_model_path,
    )
    log.info("FG prompt-cache: %s", cache_label)

    turn = 0
    with ArecordMic(args.device, DEFAULT_SAMPLE_RATE, OWW_FRAME_SAMPLES,
                    verbose=args.verbose) as mic:
        # ----- BLE peripheral (real pybleno notify on dispense turns) -----
        # Constructed *inside* the mic context (after arecord opens) so a busy
        # audio device fails before we claim hci0, and the inner `finally`
        # below always releases the radio. Advertise ONCE; the dispense
        # override fires notifies on the live subscription each turn. Any
        # failure (no staging, hci0 not reset) degrades to the [BLE→ESP32]
        # stdout mock — the demo must never block on a radio. The §1b reset
        # cycle (hciconfig hci0 up/down) must be run by the user BEFORE launch.
        ble_client: Any | None = None
        if args.no_ble:
            log.info("BLE disabled (--no-ble) — dispense turns use the stdout mock.")
        else:
            try:
                # Inject the staging dir now (heavy libs already loaded), NOT via
                # PYTHONPATH — the fcntl shim's import-time find_library() would
                # otherwise crash subprocess startup. pybleno itself loads even
                # later, at start_advertising.
                if str(args.ble_libs) not in sys.path:
                    sys.path.insert(0, str(args.ble_libs))
                from gemma_tools.dispenser_demo.ble_client import PyBlenoBleClient
                log.info("BLE: starting pybleno peripheral on hci%d (advertising "
                         "'NousVoice', notify 0xFFB2)…", args.ble_hci)
                ble_client = PyBlenoBleClient(hci_index=args.ble_hci)
                ble_client.start_advertising()  # blocks ≤15 s; raises on adapter failure
                chat_board_dispense.set_ble_client(ble_client)
                log.info("BLE: advertising — subscribe a central to 0xFFB2 to "
                         "receive 5A A5 01 00 on each dispense.")
            except Exception as exc:
                log.warning("BLE unavailable (%s) — falling back to stdout mock. "
                            "Did you run the hci0 up/down reset cycle and stage "
                            "pybleno on PYTHONPATH?", exc)
                if ble_client is not None:
                    with contextlib.suppress(Exception):
                        ble_client.stop()
                ble_client = None

        # ----- main loop -----
        log.info(
            "ready — press Ctrl-C to exit. Dispense-intent tools: "
            "get_medications_at_time, get_medication_by_name (%s).",
            "real BLE notify" if ble_client is not None else "stdout mock",
        )

        try:
            while True:
                turn += 1
                log.info("=== turn %d ===", turn)
                captured = capture_utterance(mic, wake_model, vad, args)
                if captured is None:
                    continue

                # STT
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", dir="/tmp", delete=False,
                ) as tf:
                    wav_path = Path(tf.name)
                try:
                    write_wav_16k_mono(captured, wav_path)
                    transcript, stt_wall = crispasr_decode(
                        args.crispasr_bin, args.moonshine_model, wav_path,
                        threads=args.stt_threads, verbose=args.verbose,
                    )
                finally:
                    with contextlib.suppress(OSError):
                        wav_path.unlink()

                log.info("STT %.2f s | transcript=%r", stt_wall, transcript)
                if not transcript:
                    log.warning("empty transcript — skipping LLM turn")
                    continue

                print(f"\nyou> {transcript}", flush=True)

                # Command-received chime. Tells the user "I heard you" before
                # the ~6 s LLM wall. arecord pause not strictly needed (the
                # chime is ~120 ms and we'll drain after the full turn anyway),
                # but matches the same path the wake-ack used so the audio
                # device stays consistent across both signals.
                if not args.no_command_ack and args.command_ack_wav.exists():
                    try:
                        t_ack = time.perf_counter()
                        subprocess.run(
                            ["aplay", "-q", "-D", args.aplay_device,
                             str(args.command_ack_wav)],
                            check=False, capture_output=True, timeout=2.0,
                        )
                        log.debug("command-ack chime wall=%.2f s",
                                  time.perf_counter() - t_ack)
                    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                        log.warning("command-ack chime failed: %s", exc)

                # LLM turn — dispense override already installed via
                # `import chat_board_dispense` inside _load_chat_board.
                # Reset captured state BEFORE the turn so a no-call turn
                # doesn't reuse previous tool / text.
                chat_board._last_tool = None
                chat_board._last_formatted = None
                t_llm = time.perf_counter()
                # `verbose=args.llama_verbose` (NOT args.verbose) — Layer B/C
                # behavior was to dump llama-completion's full model-loader
                # banner on every turn under -v, which buried the pipeline
                # logs. Pipeline -v controls dispenser_voice's own loggers;
                # llama subprocess noise stays silent unless --llama-verbose.
                chat_board._run_turn(
                    transcript, prefix, suffix, table,
                    args.fg_llama, fg_model_path,
                    args.fg_threads, args.fg_max_new_tokens,
                    args.fg_ctx_size, args.fg_temp, args.fg_top_k, args.fg_seed,
                    show_raw=args.raw,
                    verbose=args.llama_verbose,
                    prompt_cache=prompt_cache,
                )
                log.info("LLM turn wall=%.2f s", time.perf_counter() - t_llm)

                # Layer D: TTS playback.
                #
                # Path A (dynamic, default when piper_voice loaded):
                #   render `chat_board._last_formatted` via Piper → temp WAV
                #   → aplay. The spoken answer matches stdout exactly.
                #
                # Path B (canned fallback):
                #   look up <tts_dir>/<tool>.wav. Used when --no-dynamic-tts
                #   OR Piper load failed at startup OR render raises.
                if args.no_tts:
                    pass
                elif piper_voice is not None and chat_board._last_formatted:
                    text = chat_board._last_formatted
                    log.debug("piper text: %r", text)
                    wav_path: Path | None = None
                    try:
                        t_render = time.perf_counter()
                        wav_path = render_piper(piper_voice, text)
                        log.debug("piper render wall=%.2f s (%d KiB)",
                                  time.perf_counter() - t_render,
                                  wav_path.stat().st_size // 1024)
                        wall_tts = play_wav(
                            wav_path, args.aplay_device, mic,
                            verbose=args.verbose,
                        )
                        log.info("TTS dynamic wall=%.2f s text=%r",
                                 wall_tts, text[:60])
                    except Exception as exc:
                        log.warning("dynamic render failed (%s) — "
                                    "falling back to canned", exc)
                        tool = chat_board._last_tool
                        if tool:
                            fallback = args.tts_dir / f"{tool}.wav"
                            if fallback.exists():
                                play_wav(fallback, args.aplay_device, mic,
                                         verbose=args.verbose)
                    finally:
                        if wav_path is not None:
                            with contextlib.suppress(OSError):
                                wav_path.unlink()
                else:
                    # canned path (piper disabled or load failed)
                    tool = chat_board._last_tool
                    if tool:
                        wav_path = args.tts_dir / f"{tool}.wav"
                        if wav_path.exists():
                            wall_tts = play_wav(
                                wav_path, args.aplay_device, mic,
                                verbose=args.verbose,
                            )
                            log.info("TTS canned %s.wav wall=%.2f s",
                                     tool, wall_tts)
                        else:
                            log.warning("no canned WAV for tool=%s at %s",
                                        tool, wav_path)
                    else:
                        log.warning("no tool / formatted text captured — "
                                    "skipping TTS")

                # Drain the stdout pipe + ALSA buffer of stale audio that
                # accumulated while we were blocked in the LLM subprocess
                # (~6.5 s) + the TTS playback (~3 s WAV). Without this, the
                # next wake listener reads up to ~10 s of stale audio before
                # reaching live mic — phantom-wake risk + perceived ~10 s
                # wake-latency on turn N+1. Fold of the Layer C.1 carry-over.
                mic.drain()
        except KeyboardInterrupt:
            print("\n[exit]", flush=True)
            return 0
        finally:
            # Release hci0 so the next pybleno run (or BlueZ) can claim it.
            # Idempotent + best-effort; the stdout-mock path leaves this None.
            if ble_client is not None:
                with contextlib.suppress(Exception):
                    ble_client.stop()

# endregion

# region: argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Audio pipeline.
    p.add_argument("--device", default=DEFAULT_ALSA_DEVICE)
    p.add_argument("--wake-phrase", default="Hey Jarvis",
                   help="Cosmetic only; model is the pretrained hey_jarvis_v0.1.")
    p.add_argument("--wake-threshold", type=float, default=DEFAULT_WAKE_THRESHOLD)
    p.add_argument("--wake-patience", type=int, default=DEFAULT_WAKE_PATIENCE)
    p.add_argument("--prewake-rollback-frames", type=int,
                   default=DEFAULT_PREWAKE_ROLLBACK_FRAMES)
    p.add_argument("--vad-threshold", type=float, default=DEFAULT_VAD_THRESHOLD)
    p.add_argument("--vad-hangover-frames", type=int,
                   default=DEFAULT_VAD_HANGOVER_FRAMES)
    p.add_argument("--speech-max-s", type=float, default=DEFAULT_SPEECH_MAX_S,
                   help="Hard cap on speech_seen=True duration. Layer C "
                        "improvement over the Layer B wake-relative cap.")
    p.add_argument("--listen-abs-max-s", type=float, default=DEFAULT_LISTEN_ABS_MAX_S,
                   help="Absolute LISTENING cap from wake fire — aborts the "
                        "utterance (no STT) if no speech detected before this.")

    # Mixer.
    p.add_argument("--mixer-card", type=int, default=DEFAULT_MIXER_CARD)
    p.add_argument("--mic-percent", type=int, default=DEFAULT_MIC_PERCENT)
    p.add_argument("--pcm-percent", type=int, default=DEFAULT_PCM_PERCENT)
    p.add_argument("--no-mixer", action="store_true",
                   help="Skip mixer pre-set (use the existing levels).")

    # Layer D — TTS via aplay (default: dynamic Piper; canned WAVs as fallback).
    p.add_argument("--tts-dir", type=Path, default=DEFAULT_TTS_DIR,
                   help=f"Canned-WAV fallback dir (default: {DEFAULT_TTS_DIR}). "
                        "Used when --no-dynamic-tts or Piper render fails.")
    p.add_argument("--aplay-device", type=str, default=DEFAULT_APLAY_DEVICE,
                   help=f"ALSA playback device (default: {DEFAULT_APLAY_DEVICE}).")
    p.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL,
                   help=f"Piper voice .onnx (default: {DEFAULT_PIPER_MODEL}). "
                        "The .onnx.json config must sit next to it.")
    p.add_argument("--no-dynamic-tts", action="store_true",
                   help="Disable dynamic Piper render; use canned WAVs only "
                        "(reverts to Layer D v1 behavior).")
    p.add_argument("--no-tts", action="store_true",
                   help="Skip TTS playback entirely (stdout only).")
    p.add_argument("--wake-ack-wav", type=Path, default=DEFAULT_WAKE_ACK_WAV,
                   help=f"Short WAV (~170 ms chime) played after WAKE fires "
                        f"so the user knows to start speaking "
                        f"(default: {DEFAULT_WAKE_ACK_WAV}).")
    p.add_argument("--no-wake-ack", action="store_true",
                   help="Disable the post-WAKE acknowledgement chime.")
    p.add_argument("--command-ack-wav", type=Path, default=DEFAULT_COMMAND_ACK_WAV,
                   help=f"Short WAV (~120 ms chime, distinct from wake-ack) "
                        f"played after STT decodes the transcript and before "
                        f"the LLM turn (default: {DEFAULT_COMMAND_ACK_WAV}).")
    p.add_argument("--no-command-ack", action="store_true",
                   help="Disable the post-STT command-received chime.")

    # STT.
    p.add_argument("--crispasr-bin", type=Path, default=DEFAULT_CRISPASR)
    p.add_argument("--moonshine-model", type=Path, default=DEFAULT_MOONSHINE)
    p.add_argument("--stt-threads", type=int, default=2)

    # FunctionGemma.
    p.add_argument("--fg-dir", type=Path, default=DEFAULT_FG_MODEL_DIR)
    p.add_argument("--fg-model-name", type=str, default=DEFAULT_FG_MODEL_NAME)
    p.add_argument("--fg-health-table-name", type=str,
                   default=DEFAULT_FG_HEALTH_TABLE_NAME,
                   help="Default: health_table_dispense.json (David Smith).")
    p.add_argument("--fg-llama", type=Path, default=DEFAULT_FG_LLAMA)
    p.add_argument("--fg-threads", type=int, default=DEFAULT_FG_THREADS)
    p.add_argument("--fg-ctx-size", type=int, default=DEFAULT_FG_CTX_SIZE)
    p.add_argument("--fg-max-new-tokens", type=int, default=DEFAULT_FG_MAX_NEW_TOKENS)
    p.add_argument("--fg-temp", type=float, default=DEFAULT_FG_TEMP)
    p.add_argument("--fg-top-k", type=int, default=DEFAULT_FG_TOP_K)
    p.add_argument("--fg-seed", type=int, default=DEFAULT_FG_SEED)
    p.add_argument("--fg-prompt-cache", type=Path, default=None,
                   help="Override FG prompt cache path. Default: "
                        "/tmp/fg_pc_<model>.bin (per-model).")
    p.add_argument("--no-prompt-cache", action="store_true")

    # BLE — real pybleno notify on a dispense turn (plan §6.2 wire contract).
    p.add_argument("--no-ble", action="store_true",
                   help="Skip the pybleno peripheral; dispense turns keep the "
                        "[BLE→ESP32] stdout mock. Use when no phone/ESP32 is "
                        "present or pybleno staging is absent.")
    p.add_argument("--ble-hci", type=int, default=0,
                   help="hci adapter index for the BLE peripheral (default 0 "
                        "— SL2619 BT is hci0/UART). Requires the §1b reset "
                        "cycle (hciconfig hci0 up/down) run BEFORE launch.")
    p.add_argument("--ble-libs", type=Path, default=Path("/tmp/pylibs"),
                   help="Dir holding the staged pybleno pkg + fcntl shim + "
                        "gemma_tools (default: /tmp/pylibs). Injected into "
                        "sys.path at runtime — NOT via PYTHONPATH — because the "
                        "fcntl shim runs find_library() at import and collides "
                        "with subprocess startup if on the global path.")

    # Diagnostics.
    p.add_argument("--raw", action="store_true",
                   help="Show raw FG output before tool-call parsing.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Pipeline-level DEBUG logging (wake/VAD/STT/TTS "
                        "transitions, timings). Per-frame trace lines are "
                        "NOT enabled here — use --trace for that. Does NOT "
                        "pass through to llama-completion subprocess.")
    p.add_argument("--trace", action="store_true",
                   help="Per-frame trace logging (wake score every 80 ms "
                        "frame, VAD score every 30 ms frame). Independent "
                        "from -v; flip on only when debugging frame-level "
                        "behavior — produces ~50 lines per second.")
    p.add_argument("--llama-verbose", action="store_true",
                   help="Forward llama-completion stderr to console (model "
                        "loader banner, perf footer). Off by default — was on "
                        "implicitly under -v in Layer C; that produced ~150 "
                        "lines of dump every turn.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose, args.trace)
    return run_voice_loop(args)


if __name__ == "__main__":
    sys.exit(main())

# endregion
