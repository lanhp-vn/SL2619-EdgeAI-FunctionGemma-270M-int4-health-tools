#!/usr/bin/env python3
"""Reshape dispenser-demo splits into the Distil multi-turn-tool-calling format.

Source-of-truth contract:
- `docs/plans/dispenser-demo/plan.md` §9.1 step 1.4.
- Distil task spec at
  `/home/lanhp-wsl/.claude/plugins/marketplaces/distil-cli-skill/references/tasks/prepare-data/multi-turn-tool-calling.md`
  (`question` = stringified JSON conversation array; `answer` = stringified
  JSON `{"name": ..., "parameters": ...}` object; exactly one tool call per
  assistant turn).

Inputs (read-only):
- `data/dispenser_demo/dataset_v1/{train,test}.jsonl` — produced by
  `scripts/dispenser_demo/data/build_splits.py` from the 40 hand-authored
  seed rows.

Outputs (rewritten atomically each run):
- `releases/functiongemma-270m/002-dispenser-demo/distil/train.jsonl`
- `releases/functiongemma-270m/002-dispenser-demo/distil/test.jsonl`

The Distil CLI's `upload-data --data <dir>` looks for `train.jsonl` and
`test.jsonl` at the top level of `<dir>` (alongside `config.yaml` and
`job_description.json`). Matches the flat layout iter-001 actually
uploaded (its checked-in `data/` subdir at `001-baseline/distil/data/`
is a post-upload reorganization for the repo, not what was on disk at
upload time).

Reshape rules (mirrors `releases/functiongemma-270m/001-baseline/distil/README.md`):

- `system` message dropped — Distil's `task_description` owns it.
- `<think>...</think>` traces dropped — the teacher doesn't see them.
- Trailing assistant NL summary dropped — Distil represents only the next
  tool call, not the post-call narration.
- `question` = stringified JSON array of `[{role: 'user', content: <text>}]`
  for our single-turn seeds. The shape is reusable when we add multi-turn
  rows later.
- `answer` = stringified JSON `{"name": "<tool>", "parameters": <args>}`.

Dry-run hygiene (FG iter-001 first-upload blocker): the script also reports
within-split + cross-split `(question, answer)` duplicates and exits 1 if
any are found. Iter-001 lost a free upload run to this; the gate is cheap.

Usage:

    uv run python scripts/dispenser_demo/data/build_distil_data.py
    uv run python scripts/dispenser_demo/data/build_distil_data.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from gemma_tools.dispenser_demo.dataset import load_jsonl

_REPO = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO / "data" / "dispenser_demo" / "dataset_v1"
# Flat layout: train.jsonl / test.jsonl live at the top of distil/, alongside
# config.yaml + job_description.json — the layout `distil model upload-data
# --data <dir>` expects.
_DST_DIR = _REPO / "releases" / "functiongemma-270m" / "002-dispenser-demo" / "distil"


# --------------------------------------------------------------------------
# Reshape — pure function from local row → Distil `(question, answer)` pair.
# --------------------------------------------------------------------------


def reshape_row(row: dict[str, Any]) -> dict[str, str]:
    """Return `{question, answer}` strings for one local seed row.

    Raises `ValueError` on shape drift (no user turn, no tool_call, etc.) so
    a malformed row fails loudly rather than producing a malformed Distil
    upload.
    """
    messages = row.get("messages") or []
    user_text: str | None = None
    tool_call: dict[str, Any] | None = None

    for msg in messages:
        role = msg.get("role")
        if role == "user" and user_text is None:
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError(f"row {row.get('id')!r}: user content not a string")
            user_text = content
        elif role == "assistant":
            tcs = msg.get("tool_calls") or []
            if tcs:
                # First assistant turn with a tool_call wins — matches
                # Distil's "one tool call per turn" rule (the seed validator
                # already enforces it, this is defensive).
                tc0 = tcs[0]
                fn = tc0.get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments")
                if not isinstance(name, str) or not isinstance(args, dict):
                    raise ValueError(
                        f"row {row.get('id')!r}: tool_call missing name/arguments"
                    )
                tool_call = {"name": name, "parameters": args}
                break

    if user_text is None:
        raise ValueError(f"row {row.get('id')!r}: no user turn found")
    if tool_call is None:
        raise ValueError(f"row {row.get('id')!r}: no assistant tool_call found")

    question = json.dumps(
        [{"role": "user", "content": user_text}], ensure_ascii=False
    )
    answer = json.dumps(tool_call, ensure_ascii=False)
    return {"question": question, "answer": answer}


# --------------------------------------------------------------------------
# I/O helpers.
# --------------------------------------------------------------------------


def _serialize(rows: list[dict[str, str]]) -> str:
    parts = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    return "\n".join(parts) + ("\n" if parts else "")


def _atomic_write(path: Path, payload: str) -> None:
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
# Build runner.
# --------------------------------------------------------------------------


def build(*, write: bool = True) -> dict[str, Any]:
    src_train = _SRC_DIR / "train.jsonl"
    src_test = _SRC_DIR / "test.jsonl"
    for p in (src_train, src_test):
        if not p.exists():
            raise SystemExit(
                f"missing input: {p.relative_to(_REPO)} — run build_splits.py first"
            )

    train_reshaped = [reshape_row(r) for r in load_jsonl(src_train)]
    test_reshaped = [reshape_row(r) for r in load_jsonl(src_test)]

    # Cross-set + within-set `(question, answer)` duplicate audit — the FG
    # iter-001 first-upload blocker. Cheap to gate.
    train_pairs = [(r["question"], r["answer"]) for r in train_reshaped]
    test_pairs = [(r["question"], r["answer"]) for r in test_reshaped]

    within_train_dups = [k for k, n in Counter(train_pairs).items() if n > 1]
    within_test_dups = [k for k, n in Counter(test_pairs).items() if n > 1]
    cross_dups = set(train_pairs) & set(test_pairs)

    if within_train_dups or within_test_dups or cross_dups:
        msg_lines = ["duplicate (question, answer) pairs detected:"]
        for pair in within_train_dups:
            msg_lines.append(f"  within-train: q={pair[0][:60]}... a={pair[1]}")
        for pair in within_test_dups:
            msg_lines.append(f"  within-test:  q={pair[0][:60]}... a={pair[1]}")
        for pair in cross_dups:
            msg_lines.append(f"  cross-split:  q={pair[0][:60]}... a={pair[1]}")
        raise SystemExit("\n".join(msg_lines))

    train_payload = _serialize(train_reshaped)
    test_payload = _serialize(test_reshaped)

    if write:
        _atomic_write(_DST_DIR / "train.jsonl", train_payload)
        _atomic_write(_DST_DIR / "test.jsonl", test_payload)

    return {
        "train_count": len(train_reshaped),
        "test_count": len(test_reshaped),
        "train_payload": train_payload,
        "test_payload": test_payload,
    }


def _format_summary(result: dict[str, Any]) -> str:
    return (
        f"train rows: {result['train_count']}\n"
        f"test rows:  {result['test_count']}\n"
        f"no within-split or cross-split (question, answer) duplicates."
    )


def _check_on_disk(result: dict[str, Any]) -> int:
    expected: list[tuple[Path, str]] = [
        (_DST_DIR / "train.jsonl", result["train_payload"]),
        (_DST_DIR / "test.jsonl", result["test_payload"]),
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
            "Read-only verify mode: re-reshape and assert on-disk files "
            "match byte-for-byte; exit 1 on drift."
        ),
    )
    args = p.parse_args(argv)

    if args.check:
        result = build(write=False)
        rc = _check_on_disk(result)
        if rc == 0:
            print(_format_summary(result))
            print("\nOK: on-disk distil data matches the deterministic build.")
        return rc

    result = build(write=True)
    print(_format_summary(result))
    print(
        f"\nWrote:\n"
        f"  {(_DST_DIR / 'train.jsonl').relative_to(_REPO)}\n"
        f"  {(_DST_DIR / 'test.jsonl').relative_to(_REPO)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
