#!/usr/bin/env python3
"""Quarantine-aware ingest for an LLM-augmented FunctionGemma batch (M4.5).

Pipeline:

1.  PHI scan the candidate file FIRST. On any hit, exit 1 and write nothing —
    the merged dataset never contains a row that would have tripped the guard.
    This is the §9.5 hard rule (synthetic-only) enforced at the ingest seam.
2.  Validate every row with `gemma_tools.functiongemma_dataset.split_by_validation`.
3.  Append passing rows to `data/functiongemma/llm_expanded_v1.jsonl`.
4.  Append failing rows to `data/functiongemma/quarantine.jsonl` wrapped as
    `{"row": <original>, "errors": [...], "row_id": ..., "category": ...}`
    so the user can fix-and-re-ingest rather than losing the row.
5.  Run a final PHI sanity scan on the merged expanded file. This should always
    pass given step 1, but a regression here surfaces a bug in the guard or the
    ingest, not a data problem.
6.  Print per-batch + cumulative pass-rate against the §14 ≥ 0.80 bar
    (cumulative is what gates M4.5; per-batch is what tells the user this
    batch was bad).

Authoring contract for the LLM teacher is in
`docs/plans/FunctionGemma/llm-augmentation-prompt.md` (paste-into-web-UI
artifact). Validator semantics are pinned in
`docs/plans/FunctionGemma/README.md` §9.6.

Iron rule (§9.6): the validator IS the gate. Do NOT loosen it to make rows
pass — quarantine and re-ingest are the right paths.

Usage:
    uv run python scripts/functiongemma_ingest.py data/functiongemma/_incoming/batch_001.jsonl

Exit codes:
    0 — clean ingest (rows possibly quarantined, but nothing blocked)
    1 — PHI hit on candidate file; nothing written
    2 — usage / file-not-found
    3 — sanity-scan PHI hit on merged expanded file (regression, write completed)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from gemma_tools.functiongemma_dataset import (
    ValidationOutcome,
    split_by_validation,
)

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_EXPANDED = _REPO / "data" / "functiongemma" / "llm_expanded_v1.jsonl"
_DEFAULT_QUARANTINE = _REPO / "data" / "functiongemma" / "quarantine.jsonl"
_PHI_SCANNER_PATH = _REPO / "scripts" / "pre-commit-functiongemma.py"

# §14 M4.5 acceptance bar.
_MIN_PASS_RATE = 0.80


@dataclass(frozen=True, slots=True)
class IngestSummary:
    candidate: Path
    expanded: Path
    quarantine: Path
    batch_total: int
    batch_passed: int
    batch_failed: int
    cumulative_total: int
    cumulative_passed: int
    cumulative_failed: int

    @property
    def batch_pass_rate(self) -> float:
        return self.batch_passed / self.batch_total if self.batch_total else 0.0

    @property
    def cumulative_pass_rate(self) -> float:
        return (
            self.cumulative_passed / self.cumulative_total
            if self.cumulative_total
            else 0.0
        )

    @property
    def meets_threshold(self) -> bool:
        return self.cumulative_pass_rate >= _MIN_PASS_RATE


def _load_phi_scanner() -> ModuleType:
    """Reuse `scripts/pre-commit-functiongemma.py` via the same importlib
    pattern as `tests/test_pre_commit_phi_scanner.py` — the dash in the
    filename blocks a normal import, and the dataclass `slots=True` decorator
    needs the module registered in `sys.modules` BEFORE `exec_module` runs
    (per the §9.7 authoring pitfall). Subprocess-shelling would lose Python
    tracebacks and double the cold-start cost on tests.
    """
    name = "pre_commit_phi_ingest"
    spec = importlib.util.spec_from_file_location(name, _PHI_SCANNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load PHI scanner at {_PHI_SCANNER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _format_phi_hits(hits: Iterable[Any]) -> str:
    return "\n".join(f"  {h.path}:{h.lineno}: PHI:{h.pattern}: {h.excerpt}" for h in hits)


def _quarantine_record(
    row: dict[str, Any], outcome: ValidationOutcome
) -> dict[str, Any]:
    """Wrap a failing row + its outcome.errors so the user can triage.

    Schema: `{"row": <original raw row>, "errors": [...], "row_id": ..., "category": ...}`.
    `row` is the verbatim parsed JSON from the candidate — re-running
    `validate_conversation(record["row"])` MUST reproduce `record["errors"]`
    (gated by `tests/test_functiongemma_ingest.py`).
    """
    return {
        "row": row,
        "errors": list(outcome.errors),
        "row_id": outcome.row_id,
        "category": outcome.category,
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append-only write. Does NOT create the parent directory; the data dir
    is committed and the script is run by humans, not by CI on a fresh image.
    """
    rows = list(rows)
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def ingest(
    candidate: Path,
    expanded: Path = _DEFAULT_EXPANDED,
    quarantine: Path = _DEFAULT_QUARANTINE,
    *,
    scanner: ModuleType | None = None,
) -> IngestSummary:
    """Run the full ingest pipeline. Raises on PHI hits; otherwise returns
    a summary the caller can print or assert on.

    `scanner` is injectable for tests so they don't reload the PHI module per
    case. Production callers leave it None.
    """
    if scanner is None:
        scanner = _load_phi_scanner()

    # Step 1: PHI gate on the candidate. No writes happen on a hit.
    candidate_hits = scanner.scan_paths([candidate])
    if candidate_hits:
        msg = (
            f"PHI hit(s) on candidate {candidate}; refusing to merge.\n"
            f"{_format_phi_hits(candidate_hits)}\n"
            f"\nFix the offending lines (or drop them from the batch) and re-run."
        )
        raise PHIRefusalError(msg)

    # Step 2: validate-split.
    passed, failed = split_by_validation(candidate)
    batch_total = len(passed) + len(failed)

    # Step 3 + 4: append both streams. Order: expanded first, then quarantine —
    # if the process is killed between, the user can re-run with a candidate
    # file containing only the rows that didn't make it to quarantine, since
    # the expanded write is the strict subset that is "safe."
    _append_jsonl(expanded, passed)
    _append_jsonl(quarantine, (_quarantine_record(row, oc) for row, oc in failed))

    # Step 5: sanity-scan the merged expanded file. This must pass given step 1.
    expanded_hits = scanner.scan_paths([expanded]) if expanded.exists() else []
    if expanded_hits:
        msg = (
            f"PHI hit(s) on merged {expanded} after a clean candidate scan — "
            f"regression in the PHI guard or the ingest pipeline.\n"
            f"{_format_phi_hits(expanded_hits)}"
        )
        raise PHIPostMergeRegressionError(msg)

    # Step 6: cumulative numbers — count both sides AFTER the append.
    cumulative_passed = _count_lines(expanded)
    cumulative_failed = _count_lines(quarantine)
    cumulative_total = cumulative_passed + cumulative_failed

    return IngestSummary(
        candidate=candidate,
        expanded=expanded,
        quarantine=quarantine,
        batch_total=batch_total,
        batch_passed=len(passed),
        batch_failed=len(failed),
        cumulative_total=cumulative_total,
        cumulative_passed=cumulative_passed,
        cumulative_failed=cumulative_failed,
    )


class PHIRefusalError(Exception):
    """Raised when the PHI scanner trips on the candidate file."""


class PHIPostMergeRegressionError(Exception):
    """Raised when the post-merge sanity scan trips. Bug in the guard or
    pipeline, not a data problem."""


def _format_summary(s: IngestSummary) -> str:
    bar = "OK" if s.meets_threshold else "BELOW"
    return (
        f"ingest: {s.candidate}\n"
        f"  batch: {s.batch_passed}/{s.batch_total} passed "
        f"(rate {s.batch_pass_rate:.4f})\n"
        f"  -> appended to {s.expanded}\n"
        f"  -> quarantined  {s.batch_failed} row(s) to {s.quarantine}\n"
        f"  cumulative: {s.cumulative_passed}/{s.cumulative_total} "
        f"(rate {s.cumulative_pass_rate:.4f}; "
        f"M4.5 bar ≥ {_MIN_PASS_RATE:.2f} — {bar})"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "candidate",
        type=Path,
        help="Path to the LLM-augmented JSONL batch to ingest.",
    )
    p.add_argument(
        "--expanded",
        type=Path,
        default=_DEFAULT_EXPANDED,
        help=(
            "Target merged JSONL (passing rows are appended). "
            f"Default: {_DEFAULT_EXPANDED.relative_to(_REPO)}"
        ),
    )
    p.add_argument(
        "--quarantine",
        type=Path,
        default=_DEFAULT_QUARANTINE,
        help=(
            "Quarantine JSONL (failing rows + errors). "
            f"Default: {_DEFAULT_QUARANTINE.relative_to(_REPO)}"
        ),
    )
    args = p.parse_args(argv)

    if not args.candidate.exists():
        sys.stderr.write(f"candidate not found: {args.candidate}\n")
        return 2

    try:
        summary = ingest(args.candidate, args.expanded, args.quarantine)
    except PHIRefusalError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    except PHIPostMergeRegressionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 3

    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
