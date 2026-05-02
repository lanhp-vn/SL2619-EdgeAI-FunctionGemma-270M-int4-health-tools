"""Tests for `scripts/functiongemma_ingest.py` (M4.5 quarantine-aware ingest).

Coverage targets the §14 M4.5 acceptance row + the §9.4.3 / §9.5 ingest seam:

- Round-trip on a mixed valid/invalid candidate JSONL.
- Quarantine entry MUST round-trip through `validate_conversation` and
  reproduce the same errors (so the user can fix-and-re-ingest).
- PHI hit on the candidate triggers `PHIRefusalError` and writes nothing.
- Empty candidate is a no-op.
- Cumulative pass-rate is computed across the merged expanded + quarantine
  files (§14 bar gates the whole expanded set, not per-batch).

The ingest script lives in `scripts/` (not the package), so the test loader
mirrors the importlib pattern from `tests/test_pre_commit_phi_scanner.py`
— necessary because `@dataclass(slots=True)` introspects
`sys.modules[cls.__module__]` (§9.7 authoring pitfall).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gemma_tools.functiongemma.dataset import (
    load_jsonl,
    split_by_validation,
    validate_conversation,
)

_REPO = Path(__file__).resolve().parents[2]
_INGEST_PATH = _REPO / "scripts" / "functiongemma" / "data" / "ingest.py"
_SEED_PATH = _REPO / "data" / "functiongemma" / "seed_conversations.jsonl"


def _load_ingest() -> ModuleType:
    """Dynamic import of the ingest CLI. Same `sys.modules` registration
    requirement as `test_pre_commit_phi_scanner._load_scanner`.
    """
    name = "functiongemma_ingest_under_test"
    spec = importlib.util.spec_from_file_location(name, _INGEST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load ingest module at {_INGEST_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ingest_mod() -> ModuleType:
    return _load_ingest()


@pytest.fixture(scope="module")
def passing_rows() -> list[dict[str, Any]]:
    """Real seed rows are guaranteed valid (M4 pass rate = 1.0). Use them as
    the passing fixtures so we are not depending on the validator's
    internal heuristics — only the round-trip through ingest.
    """
    rows = list(load_jsonl(_SEED_PATH))
    assert len(rows) >= 3, "seed file must have ≥ 3 rows for ingest tests"
    return rows[:3]


def _failing_unknown_tool_row(template: dict[str, Any]) -> dict[str, Any]:
    """Mutate a passing row so it references an unregistered tool name —
    a category the validator catches with a clear `unknown tool 'X'` error.
    Kept structural (not a typo / corrupted JSON) so the round-trip
    quarantine test exercises the realistic teacher-failure mode.
    """
    row: dict[str, Any] = json.loads(json.dumps(template))
    row["id"] = "broken-001"
    for m in row["messages"]:
        for tc in m.get("tool_calls", []) or []:
            tc["function"]["name"] = "no_such_tool_definitely_not_in_registry"
    return row


def _failing_bad_system_trigger_row(template: dict[str, Any]) -> dict[str, Any]:
    """Drift the system trigger string — silently disables FG function-calling
    mode at training time, so the validator gates it. Different failure
    surface than the unknown-tool case above.
    """
    row: dict[str, Any] = json.loads(json.dumps(template))
    row["id"] = "broken-002"
    row["messages"][0]["content"] = "You are a helpful assistant"
    return row


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# --------------------------------------------------------------------------
# split_by_validation — pure helper, no IO beyond load_jsonl.
# --------------------------------------------------------------------------


def test_split_by_validation_partitions_valid_and_invalid(
    tmp_path: Path, passing_rows: list[dict[str, Any]]
) -> None:
    """Two passing + one failing → returned tuple sizes match; the failed
    entry carries non-empty outcome.errors and the original row dict.
    """
    bad = _failing_unknown_tool_row(passing_rows[0])
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [passing_rows[0], passing_rows[1], bad])

    passed, failed = split_by_validation(candidate)

    assert len(passed) == 2
    assert len(failed) == 1
    failed_row, outcome = failed[0]
    assert failed_row["id"] == "broken-001"
    assert outcome.ok is False
    assert outcome.row_id == "broken-001"
    assert outcome.errors  # non-empty


def test_split_by_validation_empty_file_is_empty_tuple(tmp_path: Path) -> None:
    candidate = tmp_path / "empty.jsonl"
    candidate.write_text("")
    passed, failed = split_by_validation(candidate)
    assert passed == []
    assert failed == []


# --------------------------------------------------------------------------
# Ingest — round-trip + quarantine shape.
# --------------------------------------------------------------------------


def test_ingest_round_trip_mixed_batch(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """Passing rows append to expanded; failing rows append to quarantine
    with their full error list. Counts in the summary match the fixture.
    """
    bad = _failing_unknown_tool_row(passing_rows[0])
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [passing_rows[0], passing_rows[1], bad])

    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    summary = ingest_mod.ingest(candidate, expanded, quarantine)

    assert summary.batch_total == 3
    assert summary.batch_passed == 2
    assert summary.batch_failed == 1
    assert expanded.exists()
    assert quarantine.exists()

    expanded_lines = expanded.read_text(encoding="utf-8").strip().splitlines()
    quarantine_lines = quarantine.read_text(encoding="utf-8").strip().splitlines()
    assert len(expanded_lines) == 2
    assert len(quarantine_lines) == 1


def test_quarantine_entry_round_trips_through_validator(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """The quarantine entry MUST contain the original row dict, and re-running
    `validate_conversation(entry["row"])` MUST reproduce `entry["errors"]`.

    This is the contract that lets the user fix a quarantined row and
    re-ingest it. Without it, the quarantine is a dead letter.
    """
    bad = _failing_bad_system_trigger_row(passing_rows[0])
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [bad])

    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    ingest_mod.ingest(candidate, expanded, quarantine)

    [line] = quarantine.read_text(encoding="utf-8").strip().splitlines()
    entry = json.loads(line)

    assert "row" in entry
    assert "errors" in entry
    assert entry["row_id"] == "broken-002"
    assert entry["category"] == bad["category"]

    # Re-validate the verbatim row dict — same errors.
    re_outcome = validate_conversation(entry["row"])
    assert re_outcome.ok is False
    assert tuple(entry["errors"]) == re_outcome.errors


def test_ingest_appends_across_calls(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """Two successive ingests of disjoint single-row batches → expanded
    grows to 2 rows, cumulative_passed reflects both. Append-only is the
    contract; truncation would silently lose work between batches.
    """
    candidate_a = tmp_path / "batch_a.jsonl"
    candidate_b = tmp_path / "batch_b.jsonl"
    _write_jsonl(candidate_a, [passing_rows[0]])
    _write_jsonl(candidate_b, [passing_rows[1]])
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    s1 = ingest_mod.ingest(candidate_a, expanded, quarantine)
    s2 = ingest_mod.ingest(candidate_b, expanded, quarantine)

    assert s1.cumulative_passed == 1
    assert s2.cumulative_passed == 2
    assert s2.cumulative_failed == 0
    assert s2.batch_total == 1


# --------------------------------------------------------------------------
# PHI gate — refuse-to-merge before any writes.
# --------------------------------------------------------------------------


def test_phi_hit_refuses_to_merge(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """A row containing an SSN-shaped string trips the PHI scanner.
    The candidate must be rejected BEFORE any writes — neither the expanded
    nor the quarantine file should appear (rollback semantics, not
    write-then-revert).
    """
    poisoned = json.loads(json.dumps(passing_rows[0]))
    poisoned["id"] = "phi-001"
    poisoned["messages"][1]["content"] = "My SSN is 123-45-6789, look it up"
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [poisoned])

    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    with pytest.raises(ingest_mod.PHIRefusalError):
        ingest_mod.ingest(candidate, expanded, quarantine)

    assert not expanded.exists(), "PHI hit must not produce a partial expanded file"
    assert not quarantine.exists(), (
        "PHI hit must not produce a partial quarantine file either — "
        "the offending row could itself contain real PHI"
    )


def test_phi_hit_does_not_disturb_existing_expanded(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """Pre-existing expanded.jsonl must be untouched on a refusal.
    Idempotence under failure: re-running the user's clean batch later still
    starts from the prior good state.
    """
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    pre_existing_line = json.dumps(passing_rows[2], ensure_ascii=False)
    expanded.write_text(pre_existing_line + "\n", encoding="utf-8")

    poisoned = json.loads(json.dumps(passing_rows[0]))
    poisoned["messages"][1]["content"] = "Reach me at patient@example.com"
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [poisoned])

    with pytest.raises(ingest_mod.PHIRefusalError):
        ingest_mod.ingest(candidate, expanded, quarantine)

    assert expanded.read_text(encoding="utf-8") == pre_existing_line + "\n"


# --------------------------------------------------------------------------
# Cumulative pass-rate — gates §14 M4.5, not per-batch.
# --------------------------------------------------------------------------


def test_cumulative_pass_rate_above_bar(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """All-passing batch → cumulative pass-rate 1.0 → meets_threshold True."""
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, passing_rows)
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    summary = ingest_mod.ingest(candidate, expanded, quarantine)
    assert summary.cumulative_pass_rate == pytest.approx(1.0)
    assert summary.meets_threshold is True


def test_cumulative_pass_rate_below_bar_still_writes(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
) -> None:
    """Bad batch (1 pass / 4 fail = 0.20) still writes — the bar is a report
    threshold, not a gate. The user wants the quarantine entries to triage,
    not a hard stop. `meets_threshold` is False; the CLI prints "BELOW".
    """
    bads = [
        _failing_unknown_tool_row(passing_rows[0]),
        _failing_unknown_tool_row(passing_rows[0]),
        _failing_bad_system_trigger_row(passing_rows[0]),
        _failing_bad_system_trigger_row(passing_rows[0]),
    ]
    # Make ids unique so the quarantine rows are distinguishable.
    for i, r in enumerate(bads):
        r["id"] = f"broken-{i:03d}"
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [passing_rows[0], *bads])
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    summary = ingest_mod.ingest(candidate, expanded, quarantine)
    assert summary.batch_passed == 1
    assert summary.batch_failed == 4
    assert summary.cumulative_pass_rate == pytest.approx(0.2)
    assert summary.meets_threshold is False
    # Both files must still have been written — the quarantine is the
    # primary artifact when the bar is missed.
    assert expanded.exists()
    assert quarantine.exists()


def test_empty_candidate_is_no_op(
    tmp_path: Path, ingest_mod: ModuleType
) -> None:
    """Zero rows in → zero writes; counts are zero; no exception."""
    candidate = tmp_path / "empty.jsonl"
    candidate.write_text("")
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    summary = ingest_mod.ingest(candidate, expanded, quarantine)
    assert summary.batch_total == 0
    assert summary.batch_passed == 0
    assert summary.batch_failed == 0
    assert not expanded.exists()
    assert not quarantine.exists()


# --------------------------------------------------------------------------
# CLI surface — exit codes the user-facing wrapper actually returns.
# --------------------------------------------------------------------------


def test_main_returns_zero_on_clean_ingest(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, passing_rows[:1])
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    rc = ingest_mod.main(
        [str(candidate), "--expanded", str(expanded), "--quarantine", str(quarantine)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingest:" in out
    assert "cumulative" in out


def test_main_returns_one_on_phi_hit(
    tmp_path: Path,
    ingest_mod: ModuleType,
    passing_rows: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    poisoned = json.loads(json.dumps(passing_rows[0]))
    poisoned["messages"][1]["content"] = "Patient SSN 999-12-3456"
    candidate = tmp_path / "batch.jsonl"
    _write_jsonl(candidate, [poisoned])
    expanded = tmp_path / "llm_expanded_v1.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"

    rc = ingest_mod.main(
        [str(candidate), "--expanded", str(expanded), "--quarantine", str(quarantine)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "PHI hit" in err
    assert not expanded.exists()


def test_main_returns_two_on_missing_candidate(
    tmp_path: Path, ingest_mod: ModuleType
) -> None:
    rc = ingest_mod.main([str(tmp_path / "nope.jsonl")])
    assert rc == 2
