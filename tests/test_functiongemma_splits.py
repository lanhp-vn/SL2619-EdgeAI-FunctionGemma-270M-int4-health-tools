"""Tests for `scripts/build_functiongemma_splits.py` (M5 stratified splits).

Coverage targets the §9.4.1 / §9.8 split-builder contract:

- All four output files exist and validate at 1.0 pass-rate.
- Holdout: 8 rows x 7 categories = 56, sorted by id within each category.
- Val: 4 rows x 7 categories = 28, disjoint from holdout.
- Train: 511 rows = 50 seed + 461 expanded remainder, disjoint from val + holdout,
  contains every seed id.
- `eval_holdout_v1.jsonl` and `dataset_v1/test.jsonl` are byte-identical
  (sha256 equality so a future drift surfaces with a unique error, not a diff).
- Determinism: re-running `build(write=False)` returns identical payload bytes.
- `--check` mode passes against freshly built files.

The build script lives in `scripts/` (outside the package), so the test loader
follows the same importlib pattern as `tests/test_functiongemma_ingest.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gemma_tools.functiongemma_dataset import load_jsonl, validate_file

_REPO = Path(__file__).resolve().parents[1]
_BUILD_PATH = _REPO / "scripts" / "build_functiongemma_splits.py"
_FG_DIR = _REPO / "data" / "functiongemma"
_SEED_PATH = _FG_DIR / "seed_conversations.jsonl"
_EXPANDED_PATH = _FG_DIR / "llm_expanded_v1.jsonl"
_HOLDOUT_PATH = _FG_DIR / "eval_holdout_v1.jsonl"
_DATASET_DIR = _FG_DIR / "dataset_v1"
_TEST_PATH = _DATASET_DIR / "test.jsonl"
_VAL_PATH = _DATASET_DIR / "val.jsonl"
_TRAIN_PATH = _DATASET_DIR / "train.jsonl"

_CATEGORIES = (
    "fact_absence",
    "fact_lookup",
    "medical_advice_refusal",
    "off_topic_refusal",
    "parallel_call",
    "tool_error_recovery",
    "two_turn",
)


def _load_build() -> ModuleType:
    """Dynamic import of the build script. Same `sys.modules` pre-registration
    requirement as `test_functiongemma_ingest._load_ingest`.
    """
    name = "functiongemma_splits_under_test"
    spec = importlib.util.spec_from_file_location(name, _BUILD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load build module at {_BUILD_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Module-scoped fixtures: build the splits exactly once for the whole module
# so re-validation across 10+ tests doesn't re-parse 600 JSONL rows each.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build_mod() -> ModuleType:
    return _load_build()


@pytest.fixture(scope="module", autouse=True)
def built_splits(build_mod: ModuleType) -> dict[str, Any]:
    """Run the builder once at module load. `autouse=True` because every test
    here either reads the on-disk files or asserts on this dict.
    """
    result: dict[str, Any] = build_mod.build(write=True)
    return result


@pytest.fixture(scope="module")
def seed_ids() -> set[str]:
    return {r["id"] for r in load_jsonl(_SEED_PATH)}


@pytest.fixture(scope="module")
def holdout_rows() -> list[dict[str, Any]]:
    return list(load_jsonl(_HOLDOUT_PATH))


@pytest.fixture(scope="module")
def val_rows() -> list[dict[str, Any]]:
    return list(load_jsonl(_VAL_PATH))


@pytest.fixture(scope="module")
def train_rows() -> list[dict[str, Any]]:
    return list(load_jsonl(_TRAIN_PATH))


# --------------------------------------------------------------------------
# Existence + validator pass-rate. The pass-rate post-condition is the
# contract that lets downstream code consume any of these four files
# without re-running the validator itself.
# --------------------------------------------------------------------------


# | path                  | description                                         |
@pytest.mark.parametrize(
    ("path", "desc"),
    [
        (_HOLDOUT_PATH, "eval_holdout_v1.jsonl exists"),
        (_TEST_PATH, "dataset_v1/test.jsonl exists"),
        (_VAL_PATH, "dataset_v1/val.jsonl exists"),
        (_TRAIN_PATH, "dataset_v1/train.jsonl exists"),
    ],
)
def test_split_files_exist(path: Path, desc: str) -> None:
    assert path.exists(), desc
    assert path.stat().st_size > 0, f"{desc}: file is empty"


@pytest.mark.parametrize(
    ("path", "desc"),
    [
        (_HOLDOUT_PATH, "holdout validates at 1.0"),
        (_TEST_PATH, "test validates at 1.0"),
        (_VAL_PATH, "val validates at 1.0"),
        (_TRAIN_PATH, "train validates at 1.0"),
    ],
)
def test_split_files_validate_at_one(path: Path, desc: str) -> None:
    report = validate_file(path, min_pass_rate=1.0)
    assert report.meets_threshold, (
        f"{desc}: pass_rate={report.pass_rate:.4f}, "
        f"failures={[f.row_id for f in report.failures[:3]]}"
    )


# --------------------------------------------------------------------------
# Holdout shape: 8 rows x 7 cats = 56, sorted by id within each category.
# --------------------------------------------------------------------------


def test_holdout_total_count(holdout_rows: list[dict[str, Any]]) -> None:
    assert len(holdout_rows) == 56, "holdout must be 8 x 7 = 56 rows"


def test_holdout_per_category_count(holdout_rows: list[dict[str, Any]]) -> None:
    per_cat: dict[str, int] = defaultdict(int)
    for r in holdout_rows:
        per_cat[r["category"]] += 1
    for cat in _CATEGORIES:
        assert per_cat[cat] == 8, f"holdout {cat} count = {per_cat[cat]}, want 8"


def test_holdout_sorted_by_id_within_category(
    holdout_rows: list[dict[str, Any]],
) -> None:
    """Determinism: within each category the holdout ids must equal the first
    8 ids from the expanded set (lex-sorted). Catches accidental shuffle.
    """
    expanded_by_cat: dict[str, list[str]] = defaultdict(list)
    for r in load_jsonl(_EXPANDED_PATH):
        expanded_by_cat[r["category"]].append(r["id"])
    held_by_cat: dict[str, list[str]] = defaultdict(list)
    for r in holdout_rows:
        held_by_cat[r["category"]].append(r["id"])

    for cat in _CATEGORIES:
        expected = sorted(expanded_by_cat[cat])[:8]
        assert held_by_cat[cat] == expected, (
            f"{cat}: holdout ids {held_by_cat[cat]} != first-8-by-id {expected}"
        )


# --------------------------------------------------------------------------
# Val shape: 4 x 7 = 28, disjoint from holdout, sourced from positions 9..12
# of the per-category expanded list.
# --------------------------------------------------------------------------


def test_val_total_count(val_rows: list[dict[str, Any]]) -> None:
    assert len(val_rows) == 28, "val must be 4 x 7 = 28 rows"


def test_val_per_category_count(val_rows: list[dict[str, Any]]) -> None:
    per_cat: dict[str, int] = defaultdict(int)
    for r in val_rows:
        per_cat[r["category"]] += 1
    for cat in _CATEGORIES:
        assert per_cat[cat] == 4, f"val {cat} count = {per_cat[cat]}, want 4"


def test_val_disjoint_from_holdout(
    val_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]
) -> None:
    val_ids = {r["id"] for r in val_rows}
    hold_ids = {r["id"] for r in holdout_rows}
    overlap = val_ids & hold_ids
    assert not overlap, f"val and holdout must be disjoint, got {sorted(overlap)}"


# --------------------------------------------------------------------------
# Train shape: 511 rows, includes all seed ids, disjoint from val + holdout.
# --------------------------------------------------------------------------


def test_train_total_count(train_rows: list[dict[str, Any]]) -> None:
    # 50 seed + (545 expanded - 56 holdout - 28 val) = 511.
    assert len(train_rows) == 511, f"train row count = {len(train_rows)}, want 511"


def test_train_disjoint_from_val_and_holdout(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> None:
    train_ids = {r["id"] for r in train_rows}
    val_ids = {r["id"] for r in val_rows}
    hold_ids = {r["id"] for r in holdout_rows}
    assert not (train_ids & val_ids), "train and val must be disjoint"
    assert not (train_ids & hold_ids), "train and holdout must be disjoint"


def test_train_contains_all_seed_ids(
    train_rows: list[dict[str, Any]], seed_ids: set[str]
) -> None:
    """§9.4.1: hand-authored seeds are part of the training corpus."""
    train_ids = {r["id"] for r in train_rows}
    missing = seed_ids - train_ids
    assert not missing, f"train missing seed ids: {sorted(missing)}"


# --------------------------------------------------------------------------
# Byte-identity: holdout file == test file. `shutil.copy` is the contract;
# this test fails the moment a future refactor switches to a second write.
# --------------------------------------------------------------------------


def test_holdout_and_test_are_byte_identical() -> None:
    a = _HOLDOUT_PATH.read_bytes()
    b = _TEST_PATH.read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest(), (
        "eval_holdout_v1.jsonl and dataset_v1/test.jsonl must be byte-identical "
        "(produced via shutil.copy); they have drifted."
    )
    assert a == b, "byte-equality fallback (in case of a hashlib regression)"


# --------------------------------------------------------------------------
# Determinism: building twice in-process must produce identical payloads.
# --------------------------------------------------------------------------


def test_determinism_two_in_process_builds_match(build_mod: ModuleType) -> None:
    """Why in-process: a subprocess reload would also exercise determinism but
    is 100x slower and can't introspect the payload strings. The build is a
    pure function of two input files; running it twice without writing is the
    cheapest determinism gate.
    """
    a = build_mod.build(write=False)
    b = build_mod.build(write=False)
    assert a["holdout_payload"] == b["holdout_payload"], "holdout payload drift"
    assert a["val_payload"] == b["val_payload"], "val payload drift"
    assert a["train_payload"] == b["train_payload"], "train payload drift"


# --------------------------------------------------------------------------
# `--check` mode: a freshly built tree must report no drift (rc 0).
# --------------------------------------------------------------------------


def test_check_mode_passes_on_freshly_built_files(build_mod: ModuleType) -> None:
    rc = build_mod.main(["--check"])
    assert rc == 0, "--check on freshly built files should report no drift"


def test_check_mode_detects_drift(
    build_mod: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutate one byte of `train.jsonl` and confirm --check reports rc=1.
    Restore the file after to keep the rest of the module's tests green.
    """
    original = _TRAIN_PATH.read_bytes()
    try:
        # Replace the trailing newline with ""; smallest possible drift that
        # is still observable byte-for-byte.
        _TRAIN_PATH.write_bytes(original.rstrip(b"\n"))
        rc = build_mod.main(["--check"])
        assert rc == 1, "--check must surface byte drift as exit 1"
    finally:
        _TRAIN_PATH.write_bytes(original)


# --------------------------------------------------------------------------
# Sanity: the rendered JSONL is one-object-per-line with no blank lines.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "desc"),
    [
        (_HOLDOUT_PATH, "holdout JSONL well-formed"),
        (_VAL_PATH, "val JSONL well-formed"),
        (_TRAIN_PATH, "train JSONL well-formed"),
    ],
)
def test_jsonl_one_object_per_line(path: Path, desc: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n"), f"{desc}: must end with single trailing newline"
    lines = text.split("\n")
    # Last element after split is "" because of the trailing newline.
    assert lines[-1] == "", desc
    for i, line in enumerate(lines[:-1], start=1):
        assert line.strip(), f"{desc}: line {i} is blank"
        json.loads(line)  # raises on malformed
