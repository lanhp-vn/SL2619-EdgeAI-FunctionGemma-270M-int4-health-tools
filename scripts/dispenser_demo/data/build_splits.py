#!/usr/bin/env python3
"""Deterministic stratified split builder for dispenser-demo Phase 1.3.

Source-of-truth contract: `docs/plans/dispenser-demo/plan.md` §9.1 step 1.3.

Inputs (read-only):

- `data/dispenser_demo/seed_conversations.jsonl` — 40 hand-authored rows
  (8 per category, 5 categories).

Outputs (rewritten atomically each run):

- `data/dispenser_demo/dataset_v1/train.jsonl`
- `data/dispenser_demo/dataset_v1/val.jsonl`
- `data/dispenser_demo/dataset_v1/test.jsonl`

Stratification: by `(category, refuse_out_of_scope.reason | None)`. The
`out_of_scope_refusal` category carries an additional `reason` sub-stratum
(3 `health_advice` + 5 `off_topic`); without sub-stratification a naive
positional split would put both `oo-001` and `oo-002` (the first two
health-advice rows) in test, leaving train with zero `health_advice`
examples — a coverage gap that synthgen alone cannot reliably backfill.

Per-sub-stratum split policy (60/20/20 target, with per-cat ≥ 1 floor in
val + test):

| N (rows in sub-stratum) | train | val | test |
| --- | --- | --- | --- |
| 3 (e.g. `health_advice`)            | 1 | 1 | 1 |
| 5 (e.g. `off_topic`)                | 3 | 1 | 1 |
| 8 (e.g. `patient_profile`)          | 4 | 2 | 2 |

Per-category totals (val + test ≥ 2 in every case):

| category                  | train | val | test |
| ---                       | ---   | --- | ---  |
| patient_profile           | 4     | 2   | 2    |
| next_appointment          | 4     | 2   | 2    |
| emergency_contact         | 4     | 2   | 2    |
| dispense                  | 4     | 2   | 2    |
| out_of_scope_refusal      | 4     | 2   | 2    |
| TOTAL                     | 20    | 10  | 10   |

Determinism: rows are sorted by `id` within each sub-stratum; ids are
uniform 6-char width (`pp-001`..`oo-008`) so lexicographic sort matches
natural sort. The split is the first `test_n` ids → test, next `val_n` →
val, remainder → train. The output files are byte-stable across runs.

Why test/val come from the LOW ids (not high): matches the FG iter-001
`build_splits.py` convention so a reader switching between the two scripts
sees the same pattern. The choice is arbitrary; the only requirement is
determinism.

Usage:

    uv run python scripts/dispenser_demo/data/build_splits.py
    uv run python scripts/dispenser_demo/data/build_splits.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gemma_tools.dispenser_demo.dataset import load_jsonl, validate_file

_REPO = Path(__file__).resolve().parents[3]
_DISP_DIR = _REPO / "data" / "dispenser_demo"
_SEED_PATH = _DISP_DIR / "seed_conversations.jsonl"

_DATASET_DIR = _DISP_DIR / "dataset_v1"
_TRAIN_PATH = _DATASET_DIR / "train.jsonl"
_VAL_PATH = _DATASET_DIR / "val.jsonl"
_TEST_PATH = _DATASET_DIR / "test.jsonl"

# Category visit order for the summary table — keeps runs visually diff-able.
# Matches the seed-authoring order in `scripts/dispenser_demo/data/build_seeds.py`.
_CATEGORY_ORDER: tuple[str, ...] = (
    "patient_profile",
    "next_appointment",
    "emergency_contact",
    "dispense",
    "out_of_scope_refusal",
)

# Per-cat floors in val + test. Plan §9.1 step 1.3 gate.
_MIN_VAL_PER_CAT = 1
_MIN_TEST_PER_CAT = 1


# --------------------------------------------------------------------------
# I/O helpers — atomic write + JSONL serialization match the seed generator
# so the four output files round-trip through the same loaders.
# --------------------------------------------------------------------------


def _serialize_rows(rows: Iterable[dict[str, Any]]) -> str:
    """One JSON object per line, compact, ensure_ascii=False, single trailing \\n."""
    parts = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    return "\n".join(parts) + ("\n" if parts else "")


def _atomic_write(path: Path, payload: str) -> None:
    """Write `payload` to `path` via tmp-file-and-rename. Idempotent and
    crash-safe — a SIGKILL between write and rename leaves the previous
    version intact rather than a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Sub-stratum key derivation. `refuse_out_of_scope` rows carry a `reason`
# argument on the first tool_call; we use it as a second-level stratum key
# so the split policy can balance health_advice vs off_topic in each split.
# --------------------------------------------------------------------------


def _stratum_key(row: dict[str, Any]) -> tuple[str, str | None]:
    """Return `(category, reason | None)` for stratification.

    `reason` is sourced from the first assistant turn's first tool_call
    arguments, if that tool is `refuse_out_of_scope`. Otherwise `None`.
    """
    category = row.get("category")
    if not isinstance(category, str):
        raise ValueError(f"row missing string `category`: id={row.get('id')!r}")

    for msg in row.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue
        tc0 = tool_calls[0]
        fn = tc0.get("function") or {}
        if fn.get("name") == "refuse_out_of_scope":
            args = fn.get("arguments") or {}
            reason = args.get("reason")
            return (category, reason if isinstance(reason, str) else None)
        # Found a non-refusal tool_call — no reason stratum.
        return (category, None)
    return (category, None)


# --------------------------------------------------------------------------
# Split derivation. Pure function — no I/O.
# --------------------------------------------------------------------------


def derive_splits(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train, val, test) deterministically via per-sub-stratum split.

    Policy: for each sub-stratum with N rows, take
        test_n = max(_MIN_TEST_PER_CAT, round(N * 0.2))
        val_n  = max(_MIN_VAL_PER_CAT,  round(N * 0.2))
        train_n = N - test_n - val_n
    and assert train_n ≥ 1. Sub-strata visit order is deterministic
    (`_CATEGORY_ORDER`, then reason alphabetically); rows within each are
    sorted by id.
    """
    by_stratum: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stratum[_stratum_key(r)].append(r)
    for sub_rows in by_stratum.values():
        sub_rows.sort(key=lambda r: r["id"])

    # Surface unexpected categories as hard errors so the operator updates
    # the constant rather than getting a silently asymmetric split.
    seen_cats = {key[0] for key in by_stratum}
    extra = seen_cats - set(_CATEGORY_ORDER)
    if extra:
        raise ValueError(
            f"seed carries unexpected categories {sorted(extra)!r}; "
            f"update _CATEGORY_ORDER and re-run"
        )

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for cat in _CATEGORY_ORDER:
        # Reasons sorted alphabetically; `None` first so the no-reason
        # sub-stratum (for the 4 domain categories) is the only one and
        # the order is unambiguous.
        sub_keys = sorted(
            (key for key in by_stratum if key[0] == cat),
            key=lambda k: (k[1] is not None, k[1] or ""),
        )
        for key in sub_keys:
            sub_rows = by_stratum[key]
            n = len(sub_rows)
            if n == 0:
                continue
            test_n = max(_MIN_TEST_PER_CAT, round(n * 0.2))
            val_n = max(_MIN_VAL_PER_CAT, round(n * 0.2))
            train_n = n - test_n - val_n
            if train_n < 1:
                raise ValueError(
                    f"stratum {key!r} (n={n}) has no train rows after "
                    f"test_n={test_n}+val_n={val_n}; rebalance the seed"
                )
            test.extend(sub_rows[:test_n])
            val.extend(sub_rows[test_n : test_n + val_n])
            train.extend(sub_rows[test_n + val_n :])

    return train, val, test


# --------------------------------------------------------------------------
# Build runner.
# --------------------------------------------------------------------------


def _load_and_validate(path: Path) -> list[dict[str, Any]]:
    report = validate_file(path, min_pass_rate=1.0)
    if not report.meets_threshold:
        msg = f"{path}: pass_rate {report.pass_rate:.4f} < 1.0; failures:"
        for f in report.failures[:5]:
            msg += f"\n  {f.row_id}: {'; '.join(f.errors)}"
        if len(report.failures) > 5:
            msg += f"\n  ... ({len(report.failures) - 5} more)"
        raise SystemExit(msg)
    return list(load_jsonl(path))


def build(*, write: bool = True) -> dict[str, Any]:
    """Run the full build. With `write=False`, returns the rendered payload
    strings without touching disk — used by `--check`."""
    seed_rows = _load_and_validate(_SEED_PATH)
    train, val, test = derive_splits(seed_rows)

    train_payload = _serialize_rows(train)
    val_payload = _serialize_rows(val)
    test_payload = _serialize_rows(test)

    if write:
        _DATASET_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write(_TRAIN_PATH, train_payload)
        _atomic_write(_VAL_PATH, val_payload)
        _atomic_write(_TEST_PATH, test_payload)

        # Defensive post-condition: every output validates at 1.0.
        for out in (_TRAIN_PATH, _VAL_PATH, _TEST_PATH):
            report = validate_file(out, min_pass_rate=1.0)
            if not report.meets_threshold:
                raise SystemExit(
                    f"post-condition failed on {out}: pass_rate {report.pass_rate:.4f}"
                )

    # Disjointness check.
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    test_ids = {r["id"] for r in test}
    overlaps = {
        "train_and_val": train_ids & val_ids,
        "train_and_test": train_ids & test_ids,
        "val_and_test": val_ids & test_ids,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise SystemExit(f"split disjointness violated: {bad}")

    # Per-category floor check. Plan §9.1 step 1.3 gate.
    val_per_cat = _per_cat_counts(val)
    test_per_cat = _per_cat_counts(test)
    floor_violations: list[str] = []
    for cat in _CATEGORY_ORDER:
        if val_per_cat.get(cat, 0) < _MIN_VAL_PER_CAT:
            floor_violations.append(
                f"val[{cat}] = {val_per_cat.get(cat, 0)} < {_MIN_VAL_PER_CAT}"
            )
        if test_per_cat.get(cat, 0) < _MIN_TEST_PER_CAT:
            floor_violations.append(
                f"test[{cat}] = {test_per_cat.get(cat, 0)} < {_MIN_TEST_PER_CAT}"
            )
    if floor_violations:
        raise SystemExit("per-cat floor violated: " + "; ".join(floor_violations))

    return {
        "seed_count": len(seed_rows),
        "train": train,
        "val": val,
        "test": test,
        "train_payload": train_payload,
        "val_payload": val_payload,
        "test_payload": test_payload,
    }


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------


def _per_cat_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["category"]] += 1
    return dict(counts)


def _format_summary(result: dict[str, Any]) -> str:
    train_cc = _per_cat_counts(result["train"])
    val_cc = _per_cat_counts(result["val"])
    test_cc = _per_cat_counts(result["test"])

    header = f"{'category':<24}  {'train':>6}  {'val':>5}  {'test':>5}"
    sep = "-" * len(header)
    lines = [
        f"seed rows: {result['seed_count']}",
        "",
        header,
        sep,
    ]
    for cat in _CATEGORY_ORDER:
        t = train_cc.get(cat, 0)
        v = val_cc.get(cat, 0)
        h = test_cc.get(cat, 0)
        lines.append(f"{cat:<24}  {t:>6}  {v:>5}  {h:>5}")
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<24}  {len(result['train']):>6}  "
        f"{len(result['val']):>5}  {len(result['test']):>5}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _check_on_disk(result: dict[str, Any]) -> int:
    """Compare expected payloads against the three on-disk files; return 0 on
    match, 1 on drift. Prints which file drifted to stderr.
    """
    expected: list[tuple[Path, str]] = [
        (_TRAIN_PATH, result["train_payload"]),
        (_VAL_PATH, result["val_payload"]),
        (_TEST_PATH, result["test_payload"]),
    ]
    drifted: list[Path] = []
    for path, payload in expected:
        if not path.exists():
            sys.stderr.write(f"missing: {path}\n")
            drifted.append(path)
            continue
        on_disk = path.read_text(encoding="utf-8")
        if on_disk != payload:
            sys.stderr.write(f"drift: {path}\n")
            drifted.append(path)
    if drifted:
        sys.stderr.write(
            "\nRegenerate via "
            f"`uv run python {Path(__file__).relative_to(_REPO)}`\n"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help=(
            "Read-only verify mode: re-derive splits and assert on-disk files "
            "match byte-for-byte; exit 1 on drift."
        ),
    )
    args = p.parse_args(argv)

    if args.check:
        result = build(write=False)
        rc = _check_on_disk(result)
        if rc == 0:
            print(_format_summary(result))
            print("\nOK: on-disk splits match the deterministic build.")
        return rc

    result = build(write=True)
    print(_format_summary(result))
    print(
        f"\nWrote:\n"
        f"  {_TRAIN_PATH.relative_to(_REPO)}\n"
        f"  {_VAL_PATH.relative_to(_REPO)}\n"
        f"  {_TEST_PATH.relative_to(_REPO)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
