#!/usr/bin/env python3
"""Build a "clean" eval holdout that excludes byte-identical train duplicates.

Why this exists: Block D's D5 audit found that 5 of the top-5 closest
train↔eval user-prompt pairs are at cosine = 1.000 (byte-identical), and the
p80 of max-cosine to train is 0.99. The current 56-row eval_holdout_v1.jsonl
measures memorization, not generalization.

This script flags every eval row whose user prompt (after lowercase + punct
strip + whitespace collapse) is byte-identical to any training row's user
prompt, and writes two outputs:

  - `data/functiongemma/eval_holdout_v2_clean.jsonl` — eval rows that have NO
    byte-identical train duplicate (the "real generalization" subset).
  - `data/functiongemma/eval_holdout_v2_contaminated.jsonl` — eval rows that
    DO have a byte-identical train duplicate (kept for diagnostic comparison).

Both files preserve the original `id` and `category` fields so existing
evaluator code continues to work. The split is deterministic — running twice
on the same inputs produces byte-identical outputs.

Note: this is the *minimum* cleanup. A proper fix per Block E recommendation
also broadens argument-value vocabulary and adds 60-80 LLM-augmented rows per
refusal class. That work is the §13 R6(a) authoring round and is not in this
script's scope (requires LLM-augmented authoring via the §9.4.3 paste flow).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from gemma_tools.functiongemma_dataset import load_jsonl  # type: ignore[import-untyped]

_REPO = Path(__file__).resolve().parents[1]
_FG_DIR = _REPO / "data" / "functiongemma"


def _normalize(text: str) -> str:
    """Lowercase, strip non-word chars, collapse whitespace.

    Mirrors the normalization used in the dataset_quality_audit's D1 probe so
    the "byte-identical" classification here matches the cosine = 1.000 pairs
    flagged in the audit report.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _user_prompt(row: dict[str, Any]) -> str:
    """First user-role message content — what the audit clusters on."""
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=_FG_DIR / "dataset_v1" / "train.jsonl")
    p.add_argument("--val", type=Path, default=_FG_DIR / "dataset_v1" / "val.jsonl")
    p.add_argument("--eval", type=Path, default=_FG_DIR / "eval_holdout_v1.jsonl")
    p.add_argument("--out-clean", type=Path,
                   default=_FG_DIR / "eval_holdout_v2_clean.jsonl")
    p.add_argument("--out-contaminated", type=Path,
                   default=_FG_DIR / "eval_holdout_v2_contaminated.jsonl")
    args = p.parse_args()

    # Build the train+val normalized-prompt index (we want eval to be disjoint
    # from BOTH; if a row leaks into val too, we'd still have a memorization
    # signal at eval time).
    train_norms: set[str] = set()
    for path in (args.train, args.val):
        for row in load_jsonl(path):
            train_norms.add(_normalize(_user_prompt(row)))

    clean: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    by_category_clean: dict[str, int] = {}
    by_category_total: dict[str, int] = {}
    for row in load_jsonl(args.eval):
        cat = row.get("category", "<missing>")
        by_category_total[cat] = by_category_total.get(cat, 0) + 1
        norm = _normalize(_user_prompt(row))
        if norm in train_norms:
            contaminated.append(row)
        else:
            clean.append(row)
            by_category_clean[cat] = by_category_clean.get(cat, 0) + 1

    args.out_clean.parent.mkdir(parents=True, exist_ok=True)
    with args.out_clean.open("w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.out_contaminated.open("w", encoding="utf-8") as f:
        for row in contaminated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input eval: {len(clean) + len(contaminated)} rows")
    print(f"  clean (no train-side duplicate): {len(clean)} -> {args.out_clean}")
    print(f"  contaminated (verbatim train dup): {len(contaminated)} -> "
          f"{args.out_contaminated}")
    print()
    print("category breakdown (clean / total):")
    for cat in sorted(by_category_total):
        clean_n = by_category_clean.get(cat, 0)
        total_n = by_category_total[cat]
        flag = "" if clean_n >= 5 else "  ← below 5-row floor for stable %"
        print(f"  {cat:24s}  {clean_n}/{total_n}{flag}")

    too_small = [c for c in by_category_total
                 if by_category_clean.get(c, 0) < 5]
    if too_small:
        print()
        print(f"WARNING: {len(too_small)} categories have < 5 clean rows: "
              f"{too_small}. The clean holdout is too small per category for "
              f"stable pass-rate reporting; an authoring round per Block E "
              f"is needed to backfill.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
