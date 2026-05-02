#!/usr/bin/env python3
"""Block F1 — refusal-row duplication pilot (deterministic).

Why this exists alongside `--refusal-loss-weight` in `finetune_functiongemma.py`:
    The proper-weighting path subclasses SFTTrainer.compute_loss; the duplication
    path duplicates refusal rows in the JSONL itself. They are NOT
    mathematically equivalent: with Adam + dataset shuffle, the duplicated row
    lands in a different micro-batch with different optimizer state, so the
    update is similar but not identical to a 2x weighted gradient. We run both
    in Block F1 — the delta between them is itself diagnostic info (the brief
    in `docs/bench/2026-05-01_functiongemma-v2-finetune-eval.md` calls
    duplication the "cheap fallback").

Output contract:
    - Every row in `--input` survives intact.
    - Each row whose `category` is in the refusal set is emitted ONCE more,
      with its `id` suffixed `-dup1` so the seed-validator's no-duplicate-id
      and no-duplicate-prompt checks still pass.
    - The output JSONL validates at pass_rate == 1.0 (this script asserts).
    - A small summary printed to stdout: row counts, multipliers, md5.

Usage:
    uv run python scripts/build_weighted_train.py \\
        --input  data/functiongemma/dataset_v1/train.jsonl \\
        --output data/functiongemma/dataset_v1/train_refusal2x.jsonl \\
        --copies 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gemma_tools.functiongemma.dataset import (  # type: ignore[import-untyped]
    load_jsonl,
    validate_file,
)

DEFAULT_REFUSAL_CATEGORIES: tuple[str, ...] = (
    "off_topic_refusal",
    "medical_advice_refusal",
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Block F1 pilot — duplicate refusal rows in a train JSONL "
                    "without disturbing the original split."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Source train.jsonl (typically data/functiongemma/dataset_v1/train.jsonl)")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination JSONL. Will be overwritten.")
    p.add_argument(
        "--copies", type=int, default=1,
        help="Number of EXTRA copies per refusal row (default 1 → 2x effective). "
             "Pass 2 for 3x, etc.",
    )
    p.add_argument(
        "--refusal-categories",
        default=",".join(DEFAULT_REFUSAL_CATEGORIES),
        help=f"Comma-separated category names. Default: "
             f"{','.join(DEFAULT_REFUSAL_CATEGORIES)}.",
    )
    p.add_argument(
        "--no-validate", action="store_true",
        help="Skip the 1.0 validate_file gate. Default off — abort if the "
             "result wouldn't pass the seed validator.",
    )
    return p.parse_args()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = _parse()
    if args.copies < 1:
        print(f"ERROR --copies must be >= 1 (got {args.copies}); 0 would be a no-op.",
              file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"ERROR input not found: {args.input}", file=sys.stderr)
        return 2

    refusal_set = frozenset(
        c.strip() for c in args.refusal_categories.split(",") if c.strip()
    )

    rows = list(load_jsonl(args.input))
    n_in = len(rows)
    cat_counts_in: dict[str, int] = {}
    for r in rows:
        cat_counts_in[str(r.get("category", "<missing>"))] = (
            cat_counts_in.get(str(r.get("category", "<missing>")), 0) + 1
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    n_duped = 0
    seen_ids: set[str] = set()
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            row_id = r.get("id")
            if row_id in seen_ids:
                # Defensive: input already has duplicate ids — refuse to make it worse.
                print(f"ERROR duplicate id in input: {row_id!r}", file=sys.stderr)
                return 3
            if isinstance(row_id, str):
                seen_ids.add(row_id)

            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

            if str(r.get("category", "")) in refusal_set:
                if not isinstance(row_id, str) or not row_id:
                    print(f"ERROR refusal row missing string id: {r!r}", file=sys.stderr)
                    return 4
                for k in range(1, args.copies + 1):
                    dup = dict(r)
                    dup_id = f"{row_id}-dup{k}"
                    if dup_id in seen_ids:
                        print(f"ERROR generated dup id collision: {dup_id!r}",
                              file=sys.stderr)
                        return 5
                    seen_ids.add(dup_id)
                    dup["id"] = dup_id
                    f.write(json.dumps(dup, ensure_ascii=False) + "\n")
                    n_out += 1
                    n_duped += 1

    cat_counts_out: dict[str, int] = {}
    for r in load_jsonl(args.output):
        cat_counts_out[str(r.get("category", "<missing>"))] = (
            cat_counts_out.get(str(r.get("category", "<missing>")), 0) + 1
        )

    print(f"input  rows: {n_in}  ({args.input})")
    print(f"output rows: {n_out}  ({args.output})")
    print(f"refusal duplications added: {n_duped} (copies/row = {args.copies})")
    print(f"\nper-category counts (input → output):")
    for cat in sorted(set(cat_counts_in) | set(cat_counts_out)):
        ci = cat_counts_in.get(cat, 0)
        co = cat_counts_out.get(cat, 0)
        flag = "  [refusal]" if cat in refusal_set else ""
        print(f"  {cat:30s} {ci:4d} → {co:4d}{flag}")
    print(f"\noutput md5: {_md5(args.output)}")

    if not args.no_validate:
        report = validate_file(args.output, min_pass_rate=1.0)
        if not report.meets_threshold:
            print(f"\nERROR validate_file FAILED: pass_rate={report.pass_rate:.4f} "
                  f"(threshold 1.0); first 3 failures:", file=sys.stderr)
            for fail in report.failures[:3]:
                print(f"  {fail.row_id}: {fail.errors}", file=sys.stderr)
            return 6
        print(f"\nvalidate_file: {report.passed}/{report.total} pass (rate=1.0000) [OK]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
