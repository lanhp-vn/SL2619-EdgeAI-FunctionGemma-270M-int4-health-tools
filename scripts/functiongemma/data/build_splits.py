#!/usr/bin/env python3
"""Deterministic stratified split builder for FunctionGemma M5 (train/val/test).

Source-of-truth contract: `docs/plans/FunctionGemma/README.md` §9.4.1
(authoring pipeline ends with a stratified train/val/test split) + §9.8
(8 rows x 7 categories holdout, sorted deterministically by id) + §10.2
(Unsloth `SFTConfig` consumes a `train_dataset` and an `eval_dataset` with
`dataset_text_field="text"`).

Inputs (read-only):

- `data/functiongemma/seed_conversations.jsonl` -- 50 hand-authored rows
- `data/functiongemma/llm_expanded_v1.jsonl`    -- LLM-augmented rows

Outputs (rewritten atomically each run):

- `data/functiongemma/eval_holdout_v1.jsonl`    -- 8 rows x 7 cats = 56
- `data/functiongemma/dataset_v1/test.jsonl`    -- byte-identical copy of holdout
- `data/functiongemma/dataset_v1/val.jsonl`     -- 4 rows x 7 cats = 28
- `data/functiongemma/dataset_v1/train.jsonl`   -- seeds + remaining expanded

Why a separate `eval_holdout_v1.jsonl` AND `dataset_v1/test.jsonl`: the
holdout file is the M5 G_EVAL artifact named in the plan (§9.8); the
`dataset_v1/test.jsonl` mirror is what the Unsloth `SFTConfig` recipe in
§10.2 expects in a single dataset directory. We `shutil.copy` to keep them
byte-identical so they cannot drift.

Determinism: ids are uniform 6-char width (`fl-101`, `fa-001`, etc.) across
both seed and expanded files; lexicographic sort on `id` therefore matches
natural sort. If a future series breaks that invariant, switch to splitting
on the `-` separator and casting the trailing digits to int.

Usage:
    uv run python scripts/build_functiongemma_splits.py
    uv run python scripts/build_functiongemma_splits.py --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gemma_tools.functiongemma.dataset import load_jsonl, validate_file

_REPO = Path(__file__).resolve().parents[3]
_FG_DIR = _REPO / "data" / "functiongemma"
_SEED_PATH = _FG_DIR / "seed_conversations.jsonl"
_EXPANDED_PATH = _FG_DIR / "llm_expanded_v1.jsonl"

_DATASET_DIR = _FG_DIR / "dataset_v1"
_HOLDOUT_PATH = _FG_DIR / "eval_holdout_v1.jsonl"
_TEST_PATH = _DATASET_DIR / "test.jsonl"
_VAL_PATH = _DATASET_DIR / "val.jsonl"
_TRAIN_PATH = _DATASET_DIR / "train.jsonl"

# §9.8: 8 rows x 7 categories = 56 eval-holdout rows.
_HOLDOUT_PER_CAT = 8
# Convention chosen alongside holdout size: a 4-row val band per category
# gives 28 rows total — large enough to track per-category eval-loss trends
# without eating the train set further. Positions 9..12 within the per-cat
# sorted list, immediately after the holdout band.
_VAL_PER_CAT = 4

# Train-thinness warning floor. Below this count for any category, the per-cat
# coverage is too sparse for the loss curve to be a meaningful signal — we
# print a warning so the M6 reviewer notices, but do NOT fail the build:
# off_topic_refusal lands at ~19 rows today (50 seed + 545 expanded - holdout - val
# split unevenly across cats), and M6 may still be acceptable with that floor.
_TRAIN_THIN_FLOOR = 20

# Pretty-printed category order for the summary table — keeps runs visually
# diff-able. Alphabetical so a new category slots in unambiguously.
_CATEGORY_ORDER = (
    "fact_absence",
    "fact_lookup",
    "medical_advice_refusal",
    "off_topic_refusal",
    "parallel_call",
    "tool_error_recovery",
    "two_turn",
)


# --------------------------------------------------------------------------
# I/O helpers — atomic write + JSONL serialization match the seed / ingest
# scripts so the four output files round-trip through the same loaders.
# --------------------------------------------------------------------------


def _serialize_rows(rows: Iterable[dict[str, Any]]) -> str:
    """One JSON object per line, compact, ensure_ascii=False, single trailing \\n."""
    parts = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    return "\n".join(parts) + ("\n" if parts else "")


def _atomic_write(path: Path, payload: str) -> None:
    """Write `payload` to `path` via a tmp-file-and-rename. Idempotent and
    crash-safe — a SIGKILL between write and rename leaves the previous
    version intact rather than a half-written file.
    """
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
# Split derivation. `derive_splits` is the pure function — no IO; the runner
# below handles read/write. This shape lets the determinism test call the
# function twice on the same input and compare bytes.
# --------------------------------------------------------------------------


def _by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by `category`, sort each group by `id` (lex == natural for
    uniform-width ids — see module docstring)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cat = r.get("category")
        if not isinstance(cat, str):
            raise ValueError(f"row missing string `category`: id={r.get('id')!r}")
        groups[cat].append(r)
    for cat in groups:
        groups[cat].sort(key=lambda r: r["id"])
    return groups


def derive_splits(
    seed_rows: list[dict[str, Any]],
    expanded_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (train_rows, val_rows, holdout_rows) deterministically.

    Holdout: first 8 sorted rows per category from `expanded_rows`.
    Val: rows 9..12 sorted per category from `expanded_rows`.
    Train: all `seed_rows` (sorted by id) + remaining expanded rows
    (sorted by category then id).

    Why `seed_rows` is included in train: §9.4.1 lists hand-authored seeds
    AND the LLM-augmented expanded set as parallel authoring inputs feeding
    one training corpus. Excluding the seeds here would discard the
    hand-authored gold whose style the expanded set was bootstrapped from,
    AND today the validator passes both files at 1.0, so there is no shape
    drift to worry about.
    """
    expanded_by_cat = _by_category(expanded_rows)

    holdout: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    train_expanded_remainder: list[dict[str, Any]] = []

    # Iterate in fixed CATEGORY_ORDER (not dict insertion order) so the holdout
    # and val files are byte-stable across Python versions / dict orderings.
    for cat in _CATEGORY_ORDER:
        cat_rows = expanded_by_cat.get(cat, [])
        if len(cat_rows) < _HOLDOUT_PER_CAT + _VAL_PER_CAT:
            raise ValueError(
                f"category {cat!r} has only {len(cat_rows)} expanded rows; "
                f"need ≥ {_HOLDOUT_PER_CAT + _VAL_PER_CAT} for holdout+val"
            )
        holdout.extend(cat_rows[:_HOLDOUT_PER_CAT])
        val.extend(cat_rows[_HOLDOUT_PER_CAT : _HOLDOUT_PER_CAT + _VAL_PER_CAT])
        train_expanded_remainder.extend(cat_rows[_HOLDOUT_PER_CAT + _VAL_PER_CAT :])

    # Tolerate categories present in expanded but not in CATEGORY_ORDER —
    # surface as a hard error so the operator updates the constant.
    extra = set(expanded_by_cat) - set(_CATEGORY_ORDER)
    if extra:
        raise ValueError(
            f"expanded set carries unexpected categories {sorted(extra)!r}; "
            f"add them to _CATEGORY_ORDER and re-run"
        )

    # Train: seeds (sorted by id) FIRST, then expanded remainder grouped by
    # category. Why this order: the seed rows are the highest-quality fragments
    # of the corpus; placing them first makes the merged file's first 50 rows
    # a self-contained smoke set the operator can eyeball.
    seed_sorted = sorted(seed_rows, key=lambda r: r["id"])
    train = [*seed_sorted, *train_expanded_remainder]

    return train, val, holdout


# --------------------------------------------------------------------------
# Build runner — orchestrates load → validate → split → write → re-validate.
# Returns a summary dict the caller can pretty-print or diff against an
# on-disk snapshot in --check mode.
# --------------------------------------------------------------------------


def _load_and_validate(path: Path) -> list[dict[str, Any]]:
    """Validate the file at 1.0 pass-rate, then return its rows.

    Why the pre-condition: the post-condition on output files is also 1.0,
    which is only meaningful if the inputs were 1.0 to begin with. A
    validator regression on either input file should fail HERE with the
    clearer error message, not downstream after we've written split outputs.
    """
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
    strings without touching disk — used by --check.
    """
    seed_rows = _load_and_validate(_SEED_PATH)
    expanded_rows = _load_and_validate(_EXPANDED_PATH)

    train, val, holdout = derive_splits(seed_rows, expanded_rows)

    train_payload = _serialize_rows(train)
    val_payload = _serialize_rows(val)
    holdout_payload = _serialize_rows(holdout)

    if write:
        _atomic_write(_HOLDOUT_PATH, holdout_payload)
        # `shutil.copy` (not a second write) so the two files cannot drift
        # by serialization-order bugs.
        _DATASET_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(_HOLDOUT_PATH, _TEST_PATH)
        _atomic_write(_VAL_PATH, val_payload)
        _atomic_write(_TRAIN_PATH, train_payload)

        # Defensive post-condition: every output we just wrote validates at 1.0.
        # If this trips, the splitter mutated rows somehow — fail loud.
        for out in (_HOLDOUT_PATH, _TEST_PATH, _VAL_PATH, _TRAIN_PATH):
            report = validate_file(out, min_pass_rate=1.0)
            if not report.meets_threshold:
                raise SystemExit(
                    f"post-condition failed on {out}: pass_rate {report.pass_rate:.4f}"
                )

    # Disjointness: train & (val | test) == empty, val & test == empty.
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    holdout_ids = {r["id"] for r in holdout}
    overlaps = {
        "train_and_val": train_ids & val_ids,
        "train_and_holdout": train_ids & holdout_ids,
        "val_and_holdout": val_ids & holdout_ids,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad:
        raise SystemExit(f"split disjointness violated: {bad}")

    return {
        "seed_count": len(seed_rows),
        "expanded_count": len(expanded_rows),
        "train": train,
        "val": val,
        "holdout": holdout,
        "train_payload": train_payload,
        "val_payload": val_payload,
        "holdout_payload": holdout_payload,
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
    hold_cc = _per_cat_counts(result["holdout"])

    header = f"{'category':<24}  {'train':>6}  {'val':>5}  {'test':>5}"
    sep = "-" * len(header)
    lines = [
        f"seed rows: {result['seed_count']}, expanded rows: {result['expanded_count']}",
        "",
        header,
        sep,
    ]
    warnings: list[str] = []
    for cat in _CATEGORY_ORDER:
        t = train_cc.get(cat, 0)
        v = val_cc.get(cat, 0)
        h = hold_cc.get(cat, 0)
        lines.append(f"{cat:<24}  {t:>6}  {v:>5}  {h:>5}")
        if t < _TRAIN_THIN_FLOOR:
            warnings.append(f"WARN: train rows for {cat} = {t} (< {_TRAIN_THIN_FLOOR})")
    lines.append(sep)
    lines.append(
        f"{'TOTAL':<24}  {len(result['train']):>6}  "
        f"{len(result['val']):>5}  {len(result['holdout']):>5}"
    )
    if warnings:
        lines.append("")
        lines.extend(warnings)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI entry. `--check` re-derives splits and asserts the on-disk files match
# byte-for-byte; useful as a CI guard against accidental hand-edits.
# --------------------------------------------------------------------------


def _check_on_disk(result: dict[str, Any]) -> int:
    """Compare expected payloads against the four on-disk files; return 0 on
    match, 1 on drift. Prints which file drifted to stderr.
    """
    expected: list[tuple[Path, str]] = [
        (_HOLDOUT_PATH, result["holdout_payload"]),
        (_TEST_PATH, result["holdout_payload"]),
        (_VAL_PATH, result["val_payload"]),
        (_TRAIN_PATH, result["train_payload"]),
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
        # Derive without writing, then compare to disk.
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
        f"  {_HOLDOUT_PATH.relative_to(_REPO)}\n"
        f"  {_TEST_PATH.relative_to(_REPO)}\n"
        f"  {_VAL_PATH.relative_to(_REPO)}\n"
        f"  {_TRAIN_PATH.relative_to(_REPO)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
