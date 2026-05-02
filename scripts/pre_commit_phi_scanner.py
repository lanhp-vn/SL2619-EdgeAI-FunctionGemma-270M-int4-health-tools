#!/usr/bin/env python3
"""PHI scan for FunctionGemma seed/expanded JSONL + YAML data (Phase B).

Scans for likely real-PHI patterns:

- US Social Security Numbers — `\\d{3}-\\d{2}-\\d{4}`
- US phone numbers NOT in the synthetic `+1-555-` prefix range
- Email addresses

Excluded by design:

- Real-provider-name heuristics — false-positive arms race; the synthetic
  fixture name "Dr. Evelyn Chen" would trip every reasonable lexicon. We
  rely on the Iron Law (`docs/plans/FunctionGemma/README.md` §9.5) plus
  human review.

Usage:
    # Scan one or more files
    uv run python scripts/pre-commit-functiongemma.py data/functiongemma/seed_conversations.jsonl

    # Scan the whole dataset directory
    uv run python scripts/pre-commit-functiongemma.py data/functiongemma/

This script does NOT install itself as a git pre-commit hook automatically —
that is the user's call. To wire it up:

    ln -s ../../scripts/pre-commit-functiongemma.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Exit codes:
    0 — clean
    1 — PHI hits found (printed to stderr, one per line)
    2 — usage error / file not found
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Patterns. Tightened per advisor feedback 2026-04-30:
#
# - Phone regex anchors on `+1-` so an ISO date like `2026-04-24` does not
#   trip; the allowlist is exactly `+1-555-` (the synthetic-fixture range).
# - SSN regex requires a non-digit boundary on each side — otherwise a
#   long medical-record-number-looking sequence trips it.
# - Email regex is strict on the local-part to avoid `8:00@home` style
#   false positives in time strings.
# --------------------------------------------------------------------------

_SSN_RE = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")

# Match `+1-NNN-NNNN` or `+1-NNN-NNN-NNNN` patterns. The synthetic allowlist
# is `+1-555-...` only.
_PHONE_RE = re.compile(r"\+1-(\d{3})-(\d{3,4})(?:-(\d{4}))?")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# JSONL/YAML are the only formats that ship under data/functiongemma/.
_SCAN_SUFFIXES = frozenset({".jsonl", ".yaml", ".yml", ".json"})


@dataclass(frozen=True, slots=True)
class PHIHit:
    path: Path
    lineno: int
    pattern: str
    excerpt: str


def _scan_text(path: Path, text: str) -> list[PHIHit]:
    hits: list[PHIHit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _SSN_RE.finditer(line):
            hits.append(
                PHIHit(path=path, lineno=lineno, pattern="ssn", excerpt=m.group(0))
            )
        for m in _PHONE_RE.finditer(line):
            area = m.group(1)
            # Allowlist the synthetic range; everything else is suspect.
            if area != "555":
                hits.append(
                    PHIHit(
                        path=path,
                        lineno=lineno,
                        pattern="phone_non_555",
                        excerpt=m.group(0),
                    )
                )
        for m in _EMAIL_RE.finditer(line):
            hits.append(
                PHIHit(path=path, lineno=lineno, pattern="email", excerpt=m.group(0))
            )
    return hits


def scan_paths(paths: Iterable[Path]) -> list[PHIHit]:
    """Scan every file (or every scannable file under a directory)."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(q for q in p.rglob("*") if q.is_file() and q.suffix in _SCAN_SUFFIXES))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    hits: list[PHIHit] = []
    for f in files:
        hits.extend(_scan_text(f, f.read_text(encoding="utf-8")))
    return hits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to scan. Directories are recursed for .jsonl/.yaml/.yml/.json.",
    )
    args = p.parse_args(argv)

    try:
        hits = scan_paths(args.paths)
    except FileNotFoundError as exc:
        sys.stderr.write(f"path not found: {exc}\n")
        return 2

    if hits:
        for h in hits:
            sys.stderr.write(
                f"{h.path}:{h.lineno}: PHI:{h.pattern}: {h.excerpt}\n"
            )
        sys.stderr.write(f"\n{len(hits)} PHI hit(s) — fix before commit.\n")
        return 1

    print(f"clean: scanned {len(args.paths)} path(s); no PHI patterns matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
