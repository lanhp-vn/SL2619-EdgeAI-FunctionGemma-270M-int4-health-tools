"""Tests for `scripts/pre-commit-functiongemma.py` (Phase B PHI guard).

The scanner is a separate script (not a package module) so it can also be
wired up as a git pre-commit hook directly. Tests import it via
`importlib.util.spec_from_file_location` to avoid renaming the file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCANNER_PATH = _REPO / "scripts" / "pre_commit_phi_scanner.py"
_SEED_PATH = _REPO / "data" / "functiongemma" / "seed_conversations.jsonl"


def _load_scanner() -> ModuleType:
    """Dynamic import. The scanner script uses `@dataclass`, which introspects
    `sys.modules[cls.__module__]` — so the module must be registered in
    `sys.modules` *before* `exec_module` runs the dataclass decorator.
    """
    name = "pre_commit_phi"
    spec = importlib.util.spec_from_file_location(name, _SCANNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load scanner module at {_SCANNER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scanner() -> ModuleType:
    return _load_scanner()


def test_scanner_clean_against_seed_file(scanner: ModuleType) -> None:
    """Regression gate — the seed file must always pass the scanner. If this
    fails, an authoring change introduced a real-PHI-shaped pattern."""
    hits = scanner.scan_paths([_SEED_PATH])
    assert hits == [], f"PHI hits in seed file: {hits}"


def test_scanner_flags_ssn(tmp_path: Path, scanner: ModuleType) -> None:
    p = tmp_path / "case.jsonl"
    p.write_text('{"x": "SSN 123-45-6789 here"}\n')
    hits = scanner.scan_paths([p])
    assert len(hits) == 1
    assert hits[0].pattern == "ssn"
    assert hits[0].excerpt == "123-45-6789"


def test_scanner_flags_non_555_phone(tmp_path: Path, scanner: ModuleType) -> None:
    p = tmp_path / "case.jsonl"
    p.write_text('{"phone": "+1-415-9876543"}\n')
    hits = scanner.scan_paths([p])
    assert len(hits) == 1
    assert hits[0].pattern == "phone_non_555"


def test_scanner_allows_555_phone(tmp_path: Path, scanner: ModuleType) -> None:
    """+1-555-0142 is the synthetic-fixture range."""
    p = tmp_path / "case.jsonl"
    p.write_text('{"phone": "+1-555-0142"}\n')
    assert scanner.scan_paths([p]) == []


def test_scanner_flags_email(tmp_path: Path, scanner: ModuleType) -> None:
    p = tmp_path / "case.jsonl"
    p.write_text('{"contact": "patient@example.com"}\n')
    hits = scanner.scan_paths([p])
    assert len(hits) == 1
    assert hits[0].pattern == "email"


def test_scanner_does_not_flag_iso_dates(tmp_path: Path, scanner: ModuleType) -> None:
    """Plan-relevant guard — `2026-04-24` and friends are everywhere in the
    seed; the SSN regex must not eat them."""
    p = tmp_path / "case.jsonl"
    p.write_text('{"date": "2026-04-24", "time": "08:15"}\n')
    assert scanner.scan_paths([p]) == []


def test_scanner_does_not_flag_dose_strings(tmp_path: Path, scanner: ModuleType) -> None:
    p = tmp_path / "case.jsonl"
    p.write_text('{"dose": "10 mg", "schedule": "08:00, 19:00"}\n')
    assert scanner.scan_paths([p]) == []


def test_scanner_recurses_directories(tmp_path: Path, scanner: ModuleType) -> None:
    """A directory arg should pick up every scannable file recursively."""
    sub = tmp_path / "nested" / "deep"
    sub.mkdir(parents=True)
    (sub / "a.jsonl").write_text('{"x": "1"}\n')
    (sub / "b.yaml").write_text("contact: patient@example.com\n")
    (sub / "c.txt").write_text("123-45-6789")  # excluded suffix
    hits = scanner.scan_paths([tmp_path])
    # b.yaml hits (email); c.txt is excluded so the SSN there is not picked up.
    assert len(hits) == 1
    assert hits[0].pattern == "email"


def test_scanner_raises_on_missing_path(tmp_path: Path, scanner: ModuleType) -> None:
    with pytest.raises(FileNotFoundError):
        scanner.scan_paths([tmp_path / "does_not_exist.jsonl"])
