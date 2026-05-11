#!/usr/bin/env python3
"""Phase 0 host smoke test for the CrispASR runtime + Moonshine Tiny GGUF.

Determines whether `cstr/moonshine-tiny-GGUF` decodes correctly via the
CrispASR binary (https://github.com/CrispStrobe/CrispASR) on the WSL2 host.
Plan gate: decoded text matches expected sentence; runtime <= 1 s for a 3 s clip.
See docs/plans/dispenser-demo/crispasr-spike-notes.md for the bootstrap recipe.

The script wraps the documented invocation:

    crispasr --backend moonshine -m MODEL.gguf -f AUDIO.wav

It measures wall-clock decode time and peak child RSS via
``resource.getrusage(RUSAGE_CHILDREN)`` (Linux: ``ru_maxrss`` in KB). No model
download, no build — the user runs the bootstrap commands first (see notes).

The streaming variant (`moonshine-streaming`, `cstr/moonshine-streaming-tiny-GGUF`)
was provisionally pinned and then superseded — see
archive/dispenser-demo-moonshine-streaming/ for the frozen recipe. Override
this script's defaults with --backend / --model to re-test the streaming path.
"""

from __future__ import annotations

import argparse
import logging
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

# CrispASR prints the decoded transcript on its own line(s); the exact framing
# isn't pinned in upstream docs, so we look for the first non-empty line that
# isn't obvious log noise. This is a best-effort scrape — the spike's primary
# job is to confirm the binary runs at all and meets the latency gate.
_LOG_PREFIX_RE = re.compile(r"^(?:\[[\w\d:.\- ]+\]|whisper_|crispasr_|ggml_|moonshine_)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bin",
        type=Path,
        required=True,
        help="Path to the prebuilt `crispasr` binary (e.g. /path/to/CrispASR/build/bin/crispasr).",
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the moonshine-tiny GGUF (e.g. moonshine-tiny-q4_k.gguf). "
        "Tokenizer must be co-located per CrispASR docs.",
    )
    p.add_argument(
        "--wav",
        type=Path,
        required=True,
        help="Input audio (16-bit PCM WAV). For the plan's 3-s gate use a ~3 s clip.",
    )
    p.add_argument(
        "--backend",
        default="moonshine",
        help="CrispASR backend flag (default: moonshine — the non-streaming "
        "variant pinned for dispenser-demo). Pass 'moonshine-streaming' to "
        "re-test the superseded streaming variant.",
    )
    p.add_argument(
        "--expected",
        default=None,
        help="Optional substring expected in the transcript. Match is case-insensitive.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Thread count passed as `-t N` to crispasr. Default: omit, "
        "letting CrispASR pick min(4, hardware_concurrency()).",
    )
    p.add_argument(
        "--language",
        default="en",
        help="Language code passed as `-l LANG`. Default 'en' skips the auto-LID "
        "step (which on first run downloads ggml-tiny.bin ~77 MB into "
        "~/.cache/crispasr/ and on every run costs ~70 MB RSS + extra latency). "
        "Pass 'auto' to re-enable auto-detect.",
    )
    p.add_argument(
        "--no-punctuation",
        dest="punctuation",
        action="store_false",
        help="Pass --no-punctuation to crispasr (default). For the active "
        "moonshine backend this turns off native punctuation (it has "
        "CAP_PUNCTUATION_TOGGLE, so no auto-fetch). For other backends "
        "without that capability, this also gates the dispatch-layer "
        "FireRedPunc auto-fetch (~80 MB + second model pass) — required "
        "on the offline SL2619.",
    )
    p.add_argument(
        "--punctuation",
        dest="punctuation",
        action="store_true",
        help="Re-enable upstream auto-punctuation behaviour (auto-downloads FireRedPunc).",
    )
    p.set_defaults(punctuation=False)
    p.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Subprocess timeout in seconds (default: 30).",
    )
    p.add_argument(
        "--latency-budget-s",
        type=float,
        default=1.0,
        help="Plan §9 Phase 0 host gate: <= 1.0 s for a 3-s clip (default: 1.0).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _check_inputs(args: argparse.Namespace) -> None:
    """Fail fast with clear messages — don't let CrispASR's own errors confuse the diagnosis."""
    if not args.bin.exists():
        raise FileNotFoundError(
            f"crispasr binary not found at {args.bin}. "
            "Build it first per docs/plans/dispenser-demo/crispasr-spike-notes.md §Bootstrap."
        )
    if not args.bin.is_file() or not shutil.which(str(args.bin)):
        # is_file() + exec bit via shutil.which on a full path is the simplest check
        # that survives symlinks and PATH oddities.
        raise PermissionError(f"crispasr binary at {args.bin} is not executable.")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model GGUF not found at {args.model}. "
            "Download with `hf download cstr/moonshine-tiny-GGUF` "
            "(see spike notes §Bootstrap)."
        )
    if not args.wav.exists():
        raise FileNotFoundError(f"Audio file not found at {args.wav}.")


def parse_transcript(stdout: str) -> str:
    """Best-effort scrape of the decoded transcript from crispasr stdout.

    CrispASR upstream doesn't pin a transcript-line marker, so we strip obvious
    log noise (bracketed timestamps, library log prefixes) and return the longest
    surviving line. The Phase 0 gate is "runs and decodes" — the exact transcript
    is human-verified against the input WAV.
    """
    candidates: list[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if _LOG_PREFIX_RE.match(s):
            continue
        candidates.append(s)
    if not candidates:
        return ""
    return max(candidates, key=len)


def run_smoke(args: argparse.Namespace) -> int:
    _check_inputs(args)

    cmd = [
        str(args.bin),
        "--backend",
        args.backend,
        "-l",
        args.language,
        "-m",
        str(args.model),
        "-f",
        str(args.wav),
    ]
    if not args.punctuation:
        cmd.append("--no-punctuation")
    if args.threads is not None:
        cmd.extend(["-t", str(args.threads)])
    log.info("invoking: %s", " ".join(cmd))

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"FAIL: timeout after {elapsed:.2f}s (budget {args.timeout}s)")
        return 2
    elapsed = time.monotonic() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss on Linux is the max RSS across all child processes since process
    # start, in KB. Subtracting before/after isolates this run.
    peak_rss_kb = rusage_after.ru_maxrss - rusage_before.ru_maxrss

    transcript = parse_transcript(proc.stdout)

    print(f"binary       : {args.bin}")
    print(f"model        : {args.model}")
    print(f"wav          : {args.wav}")
    print(f"backend      : {args.backend}")
    print(f"exit_code    : {proc.returncode}")
    print(f"elapsed_s    : {elapsed:.3f}")
    print(f"peak_rss_kb  : {peak_rss_kb}")
    print(f"transcript   : {transcript!r}")
    if args.verbose:
        print("---- stdout ----")
        print(proc.stdout)
        print("---- stderr ----")
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print(f"FAIL: crispasr exited {proc.returncode}")
        return 1
    if elapsed > args.latency_budget_s:
        print(f"FAIL: decode {elapsed:.3f}s exceeds host budget {args.latency_budget_s}s (plan §9 Phase 0)")
        return 1
    if args.expected and args.expected.lower() not in transcript.lower():
        print(f"FAIL: expected substring {args.expected!r} not in transcript")
        return 1
    print("PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
