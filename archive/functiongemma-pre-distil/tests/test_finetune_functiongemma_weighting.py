"""Tests for Block F1 — refusal-class loss reweighting.

Coverage:
- `weighted_masked_lm_loss` is bit-identical to vanilla causal-LM CE when every
  row weight is 1.0 (the equivalence guarantee — without this, weight=1.0 is NOT
  a no-op and the F1 experiment is contaminated).
- Per-row weighting actually scales the contribution of refusal-row tokens.
- `num_items_in_batch` divisor matches HF Trainer's grad-accum semantics.
- `_WeightedCollator` (wrapped via `_build_weighted_collator`) injects the
  expected `row_weight` tensor for refusal vs non-refusal rows.
- `build_weighted_train.py` produces a JSONL that validates at 1.0, with the
  expected refusal-row count multiplier and unique ids.

Why test the math via a pure helper instead of the full SFTTrainer subclass:
TRL 0.22.2 is an opt-in dep (`functiongemma` extras) and is NOT installed in
the host CI environment. The pure helper closes the math contract; the
subclass is a thin shim that calls it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402, N812

_REPO = Path(__file__).resolve().parents[1]
_FINETUNE_PATH = _REPO / "scripts" / "finetune_functiongemma.py"
_BUILD_WEIGHTED_PATH = _REPO / "scripts" / "build_weighted_train.py"
_TRAIN_PATH = _REPO / "data" / "functiongemma" / "dataset_v1" / "train.jsonl"


def _load_finetune_module() -> ModuleType:
    """Import `scripts/finetune_functiongemma.py` without executing main().

    Mirror of the importlib pattern in `tests/test_functiongemma_splits.py` —
    the file lives outside the package so we attach it under a sentinel name
    in `sys.modules` to keep relative-import semantics happy.
    """
    name = "finetune_functiongemma_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _FINETUNE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# weighted_masked_lm_loss math contract
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fnt() -> ModuleType:
    return _load_finetune_module()


def _vanilla_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor,
    *, num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """Reference impl mirroring HF GemmaForCausalLM.forward's loss math.

    Shifts logits, computes CE with ignore_index=-100, then either takes the
    mean over unmasked positions (no grad-accum) or divides the SUM by
    `num_items_in_batch` (grad-accum). This is the contract the equivalence
    test must hold against.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    per_token = flat.view(shift_labels.size())
    mask = (shift_labels != -100).to(per_token.dtype)
    numerator = (per_token * mask).sum()
    if num_items_in_batch is not None:
        return numerator / float(num_items_in_batch)
    return numerator / mask.sum().clamp(min=1.0)


@pytest.mark.parametrize(
    "desc,batch,seq_len,vocab,num_items",
    [
        ("small_batch_no_grad_accum", 2, 8, 16, None),
        ("small_batch_grad_accum",     2, 8, 16, 12),
        ("realistic_batch",            4, 32, 64, None),
        ("realistic_batch_accum",      4, 32, 64, 80),
    ],
)
def test_weight_one_is_no_op(
    fnt: ModuleType, desc: str, batch: int, seq_len: int, vocab: int,
    num_items: int | None,
) -> None:
    """The single equivalence guarantee — weight=1.0 reproduces vanilla CE
    to within float32 round-off (not float64 — model forward is fp32 here).
    A failure here means weight=1.0 is contaminated and every F1 experiment
    is moot."""
    torch.manual_seed(3407)
    logits = torch.randn(batch, seq_len, vocab)
    labels = torch.randint(0, vocab, (batch, seq_len))
    # Mask ~1/3 of positions to -100 to mimic train_on_responses_only behaviour.
    mask_positions = torch.rand(batch, seq_len) < 0.33
    labels[mask_positions] = -100

    weight_ones = torch.ones(batch)
    got = fnt.weighted_masked_lm_loss(
        logits, labels, row_weight=weight_ones, num_items_in_batch=num_items,
    )
    want = _vanilla_lm_loss(logits, labels, num_items_in_batch=num_items)
    # Tolerance 5e-5: per-row chunked-CE accumulates sums in a different order
    # than vanilla flat CE → fp32 associativity drift on order ~1e-5 even on
    # tiny test tensors. The math is identical; the bit pattern is not.
    # Tightening below ~1e-5 produces spurious failures.
    assert torch.isclose(got, want, atol=5e-5, rtol=5e-5), (
        f"{desc}: weight=1.0 not a no-op; got={got.item()} want={want.item()} "
        f"diff={abs(got.item() - want.item()):.3e}"
    )


def test_row_weight_none_matches_ones(fnt: ModuleType) -> None:
    """`row_weight=None` and an explicit ones-tensor must agree.
    Otherwise the dry-run path (no collator → no row_weight injected) and the
    weighted path diverge at the same input."""
    torch.manual_seed(0)
    logits = torch.randn(3, 10, 12)
    labels = torch.randint(0, 12, (3, 10))
    labels[torch.rand(3, 10) < 0.4] = -100
    a = fnt.weighted_masked_lm_loss(logits, labels, row_weight=None)
    b = fnt.weighted_masked_lm_loss(logits, labels, row_weight=torch.ones(3))
    assert torch.isclose(a, b, atol=5e-5, rtol=5e-5)


def test_doubling_refusal_row_doubles_its_contribution(fnt: ModuleType) -> None:
    """Hand-built case: 2 rows, row 0 a refusal at weight=2.0, row 1 normal.
    The numerator must equal 2 * sum(row0_CE_unmasked) + 1 * sum(row1_CE_unmasked).
    """
    torch.manual_seed(42)
    logits = torch.randn(2, 6, 8)
    labels = torch.randint(0, 8, (2, 6))
    # Make the masking explicit so the expected math is deterministic.
    labels[0, :2] = -100   # row 0: positions 0,1 masked (instruction tokens)
    labels[1, :3] = -100   # row 1: positions 0..2 masked

    weights = torch.tensor([2.0, 1.0])
    got = fnt.weighted_masked_lm_loss(
        logits, labels, row_weight=weights, num_items_in_batch=None,
    )

    # Hand-roll the expected: per-token CE, mask, then weighted sum / unmasked count.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    mask = (shift_labels != -100).to(per_token.dtype)
    expected_num = (per_token[0] * mask[0]).sum() * 2.0 + (per_token[1] * mask[1]).sum() * 1.0
    expected = expected_num / mask.sum().clamp(min=1.0)
    assert torch.isclose(got, expected, atol=1e-6, rtol=1e-6), (
        f"weighted sum diverges; got={got.item()} expected={expected.item()}"
    )


def test_grad_accum_divisor_matches_num_items(fnt: ModuleType) -> None:
    """`num_items_in_batch` overrides the per-batch denominator. Verify the
    output equals (weighted SUM / num_items) — this is what HF Trainer 4.45+
    does when scaling grad-accum. Off-by-this-factor is a silent ~2x training
    error at GAS=2."""
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 6)
    labels = torch.randint(0, 6, (2, 5))
    labels[torch.rand(2, 5) < 0.2] = -100
    weights = torch.tensor([1.5, 1.0])
    num_items = 7

    got = fnt.weighted_masked_lm_loss(
        logits, labels, row_weight=weights, num_items_in_batch=num_items,
    )

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())
    mask = (shift_labels != -100).to(per_token.dtype)
    weight_bt = weights.unsqueeze(1)
    expected = (per_token * mask * weight_bt).sum() / float(num_items)
    assert torch.isclose(got, expected, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------
# Collator wrapper
# --------------------------------------------------------------------------


def test_weighted_collator_injects_correct_weights(fnt: ModuleType) -> None:
    """The collator MUST tag each row with the right weight based on category.
    A bug here silently feeds vanilla weights into compute_loss — F1 would
    appear to do nothing and we'd misdiagnose the experiment as ineffective."""

    def fake_base(features: list[dict[str, Any]]) -> dict[str, Any]:
        # Stand in for DataCollatorForLanguageModeling — just bundle whatever
        # input_ids the test passes in; the real collator does padding etc.
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }

    wrapper = fnt._build_weighted_collator(
        fake_base,
        refusal_categories=frozenset({"off_topic_refusal", "medical_advice_refusal"}),
        refusal_weight=2.5,
    )
    feats = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "category": "off_topic_refusal"},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6], "category": "fact_lookup"},
        {"input_ids": [7, 8, 9], "labels": [7, 8, 9], "category": "medical_advice_refusal"},
        {"input_ids": [0, 1, 2], "labels": [0, 1, 2], "category": "two_turn"},
    ]
    batch = wrapper(feats)
    assert "row_weight" in batch
    assert torch.equal(batch["row_weight"], torch.tensor([2.5, 1.0, 2.5, 1.0]))


def test_weighted_collator_strips_metadata_before_base(fnt: ModuleType) -> None:
    """`category` (and `text` if present) must NOT reach the underlying
    DataCollatorForLanguageModeling — it doesn't know about them and would
    either drop them silently (best case) or trip on an unexpected key."""
    seen_keys: list[set[str]] = []

    def fake_base(features: list[dict[str, Any]]) -> dict[str, Any]:
        seen_keys.append(set(features[0].keys()))
        return {"input_ids": torch.tensor([[1]])}

    wrapper = fnt._build_weighted_collator(
        fake_base,
        refusal_categories=frozenset({"off_topic_refusal"}),
        refusal_weight=2.0,
    )
    wrapper([{"input_ids": [1], "labels": [1], "text": "x", "category": "off_topic_refusal"}])
    assert seen_keys == [{"input_ids", "labels"}], (
        f"metadata leaked into base collator: {seen_keys}"
    )


def test_weighted_collator_prefers_row_weight_over_category(fnt: ModuleType) -> None:
    """Production path: rows have a `row_weight` numeric column attached AFTER
    tokenization (because TRL strips the `category` field during
    `_prepare_non_packed_dataloader`). The collator MUST honor that column
    over re-deriving from `category` — failing to do so silently reverts to
    vanilla SFT (the bug that produced bit-identical weight=1.5/2.0/3.0
    results in the 2026-05-01 first F1 grid)."""

    def fake_base(features: list[dict[str, Any]]) -> dict[str, Any]:
        return {"input_ids": torch.tensor([f["input_ids"] for f in features])}

    wrapper = fnt._build_weighted_collator(
        fake_base,
        refusal_categories=frozenset({"off_topic_refusal"}),
        refusal_weight=2.0,  # category-derived would say 2.0 for the refusal row
    )
    # row 0 has row_weight=3.0 explicitly + category="off_topic_refusal" — the
    # explicit numeric column MUST win (3.0, not 2.0 from category lookup).
    feats = [
        {"input_ids": [1], "row_weight": 3.0, "category": "off_topic_refusal"},
        {"input_ids": [2], "row_weight": 1.0, "category": "off_topic_refusal"},
        {"input_ids": [3], "row_weight": 2.0},  # numeric column only
    ]
    batch = wrapper(feats)
    assert torch.equal(batch["row_weight"], torch.tensor([3.0, 1.0, 2.0]))


def test_weighted_collator_missing_category_falls_back_to_one(fnt: ModuleType) -> None:
    """If a row has no `category` (e.g. an LLM-augmented row that lost the
    field), it MUST get weight 1.0 — silently treating it as refusal would
    contaminate the gradient mix."""

    def fake_base(features: list[dict[str, Any]]) -> dict[str, Any]:
        return {"input_ids": torch.tensor([[1]] * len(features))}

    wrapper = fnt._build_weighted_collator(
        fake_base,
        refusal_categories=frozenset({"off_topic_refusal"}),
        refusal_weight=2.0,
    )
    out = wrapper([{"input_ids": [1]}, {"input_ids": [2], "category": ""}])
    assert torch.equal(out["row_weight"], torch.tensor([1.0, 1.0]))


# --------------------------------------------------------------------------
# build_weighted_train.py — duplication pilot
# --------------------------------------------------------------------------


def test_build_weighted_train_default_path_unchanged_when_disabled(tmp_path: Path) -> None:
    """Sanity check: the original `train.jsonl` is never touched by the build
    script — the Block F default training data path remains the unmodified
    file. This is the single guarantee that lets us run F1 alongside the
    v3 baseline without contamination concerns."""
    if not _TRAIN_PATH.exists():
        pytest.skip("dataset_v1/train.jsonl not present")
    pre = _TRAIN_PATH.read_bytes()
    out = tmp_path / "train_refusal2x.jsonl"
    rc = subprocess.run(
        [sys.executable, str(_BUILD_WEIGHTED_PATH),
         "--input", str(_TRAIN_PATH),
         "--output", str(out), "--copies", "1"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert rc.returncode == 0, f"build failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}"
    assert _TRAIN_PATH.read_bytes() == pre, "train.jsonl was modified — generator must be read-only on input"


def test_build_weighted_train_doubles_refusal_rows_only(tmp_path: Path) -> None:
    """Per-category counts: refusal categories double; everything else holds.
    Plus: every row in the output validates at 1.0, no duplicate ids."""
    if not _TRAIN_PATH.exists():
        pytest.skip("dataset_v1/train.jsonl not present")

    out = tmp_path / "train_refusal2x.jsonl"
    rc = subprocess.run(
        [sys.executable, str(_BUILD_WEIGHTED_PATH),
         "--input", str(_TRAIN_PATH),
         "--output", str(out), "--copies", "1"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert rc.returncode == 0, rc.stderr

    # Per-category counts.
    src_counts: dict[str, int] = {}
    for line in _TRAIN_PATH.read_text().splitlines():
        if not line.strip():
            continue
        src_counts[json.loads(line).get("category", "")] = (
            src_counts.get(json.loads(line).get("category", ""), 0) + 1
        )
    out_counts: dict[str, int] = {}
    out_ids: list[str] = []
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out_counts[row.get("category", "")] = out_counts.get(row.get("category", ""), 0) + 1
        out_ids.append(row.get("id", ""))

    refusal = {"off_topic_refusal", "medical_advice_refusal"}
    for cat, n in src_counts.items():
        if cat in refusal:
            assert out_counts.get(cat, 0) == 2 * n, (
                f"{cat}: expected {2*n} (2x) in output, got {out_counts.get(cat, 0)}"
            )
        else:
            assert out_counts.get(cat, 0) == n, (
                f"{cat}: expected {n} unchanged, got {out_counts.get(cat, 0)}"
            )

    # Unique ids
    assert len(out_ids) == len(set(out_ids)), "duplicate ids in output"
    # Duped ids are visibly suffixed
    assert any(_id.endswith("-dup1") for _id in out_ids), "no -dup1 suffix found"


def test_build_weighted_train_validates_at_one(tmp_path: Path) -> None:
    """The output JSONL must validate at pass_rate==1.0 against the seed
    validator — duplicating refusal rows shouldn't break any seed-shape rule."""
    if not _TRAIN_PATH.exists():
        pytest.skip("dataset_v1/train.jsonl not present")
    out = tmp_path / "train_refusal2x.jsonl"
    rc = subprocess.run(
        [sys.executable, str(_BUILD_WEIGHTED_PATH),
         "--input", str(_TRAIN_PATH),
         "--output", str(out), "--copies", "1"],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert rc.returncode == 0, rc.stderr
    # The script itself runs validate_file at threshold 1.0; surface its summary.
    assert "validate_file" in rc.stdout and "[OK]" in rc.stdout, rc.stdout
