"""Smoke tests for gemma_tools.sft_build (CLI entrypoint).

These run the full pipeline end-to-end against the canonical fixtures,
writing into a tmp_path so the in-tree `tools/data/sft_v1*.jsonl` files
are never clobbered by the test run.
"""

from __future__ import annotations

import json
from pathlib import Path

from gemma_tools.sft_build import main


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def test_sft_build_writes_all_artifacts(tmp_path: Path) -> None:
    repo = _repo()
    rc = main(
        [
            "--pool", str(repo / "data" / "clean_sft_dataset.json"),
            "--prompts", str(repo / "data" / "prompts.yaml"),
            "--health", str(repo / "data" / "health_table_v1.yaml"),
            "--out-dir", str(tmp_path),
            "--now", "2026-04-25",
            "--seed", "42",
        ]
    )
    assert rc == 0, "sft-build CLI returned non-zero"

    audit = tmp_path / "sft_v1.audit.jsonl"
    assert audit.exists(), "audit JSONL not written"
    audit_rows = audit.read_text(encoding="utf-8").splitlines()
    # Audit must contain one line per deduped pool row (1259 today). We
    # bound loosely to allow the dataset to grow without breaking the test
    # but tightly enough to flag a regression.
    assert len(audit_rows) >= 1200, f"audit row count too low: {len(audit_rows)}"

    # Path B artifacts.
    train_b = tmp_path / "sft_v1.train.jsonl"
    val_b = tmp_path / "sft_v1.val.jsonl"
    test_b = tmp_path / "sft_v1.test.jsonl"
    for f in (train_b, val_b, test_b):
        assert f.exists(), f"Path B file missing: {f.name}"

    # Path A artifacts (default emit, --skip-path-a not passed).
    train_a = tmp_path / "sft_v1_pathA.train.jsonl"
    val_a = tmp_path / "sft_v1_pathA.val.jsonl"
    test_a = tmp_path / "sft_v1_pathA.test.jsonl"
    for f in (train_a, val_a, test_a):
        assert f.exists(), f"Path A file missing: {f.name}"

    # Row counts match across Path A and Path B for the same split.
    for split in ("train", "val", "test"):
        a_rows = (tmp_path / f"sft_v1_pathA.{split}.jsonl").read_text().splitlines()
        b_rows = (tmp_path / f"sft_v1.{split}.jsonl").read_text().splitlines()
        assert len(a_rows) == len(b_rows), (
            f"{split} row counts diverge: pathA={len(a_rows)} pathB={len(b_rows)}"
        )

    # Spot-check the JSONL is parseable and shaped per TRL conversational.
    sample = json.loads(test_b.read_text(encoding="utf-8").splitlines()[0])
    assert "messages" in sample, sample
    assert {m["role"] for m in sample["messages"]} == {"user", "assistant"}, sample
    # Path B: user content must carry the YAML block.
    user_content = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    assert "YAML:" in user_content, "Path B user content missing YAML block"


def test_sft_build_skip_path_a(tmp_path: Path) -> None:
    repo = _repo()
    rc = main(
        [
            "--pool", str(repo / "data" / "clean_sft_dataset.json"),
            "--prompts", str(repo / "data" / "prompts.yaml"),
            "--health", str(repo / "data" / "health_table_v1.yaml"),
            "--out-dir", str(tmp_path),
            "--skip-path-a",
        ]
    )
    assert rc == 0
    # Path A files must NOT exist when --skip-path-a is set.
    for split in ("train", "val", "test"):
        f = tmp_path / f"sft_v1_pathA.{split}.jsonl"
        assert not f.exists(), f"--skip-path-a leaked file: {f}"
        # Path B counterparts should still exist.
        assert (tmp_path / f"sft_v1.{split}.jsonl").exists()
