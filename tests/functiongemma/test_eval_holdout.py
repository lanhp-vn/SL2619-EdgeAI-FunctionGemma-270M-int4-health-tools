"""Tests for scripts/eval_functiongemma_holdout.

Covers the runnable surface of the M6 spec: pure metric, gold-trace
extractor, and the two CLI subcommands that don't need a model
(`--list-categories`, `--dry-run`). The model-inference seam is intentionally
NOT covered here — it raises NotImplementedError until M6 wires it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# `scripts/` is not a package; load via spec to keep imports clean and avoid
# polluting the package layout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "functiongemma" / "eval" / "eval_holdout.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "eval_functiongemma_holdout", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_functiongemma_holdout"] = mod
    spec.loader.exec_module(mod)
    return mod


# Typed `Any` because mypy can't resolve attributes on a runtime-loaded module.
# The script lives under `scripts/` (not in `src/`), so it isn't importable
# normally. Test asserts give us the per-attribute coverage at runtime.
eh: Any = _load_module()


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


def _seed_row(
    *,
    category: str = "fact_lookup",
    tool_calls: list[dict[str, Any]] | None = None,
    final_text: str | None = "Your heart rate is 72 bpm.",
    row_id: str = "fl-001",
) -> dict[str, Any]:
    """Build a minimal but shape-correct holdout row.

    Mirrors `seed_conversations.jsonl` shape: system → user → assistant(call)
    → tool → assistant(answer). For refusal rows pass `tool_calls=[]` and
    `final_text="<refusal>"` and the call/tool pair is omitted automatically.
    """
    if tool_calls is None:
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_vitals", "arguments": {}},
            }
        ]
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a model that can do function calling with the following functions",
        },
        {"role": "user", "content": "What's my heart rate?"},
    ]
    if tool_calls:
        msgs.append(
            {
                "role": "assistant",
                "content": "<think>User wants vitals; call get_vitals.</think>",
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            msgs.append(
                {
                    "role": "tool",
                    "name": tc["function"]["name"],
                    "tool_call_id": tc["id"],
                    "content": json.dumps({"heart_rate_bpm": 72}),
                }
            )
        msgs.append(
            {
                "role": "assistant",
                "content": f"<think>HR is 72 bpm.</think>\n{final_text}",
                "tool_calls": [],
            }
        )
    else:
        msgs.append(
            {
                "role": "assistant",
                "content": f"<think>Out of scope; refuse.</think>\n{final_text}",
                "tool_calls": [],
            }
        )
    return {
        "id": row_id,
        "category": category,
        "messages": msgs,
        "tools": [{"type": "function", "function": {"name": "get_vitals"}}],
    }


@pytest.fixture
def tmp_holdout(tmp_path: Path) -> Path:
    """Holdout fixture: 2 fact_lookup + 1 refusal — exercises both branches."""
    rows = [
        _seed_row(row_id="fl-001"),
        _seed_row(row_id="fl-002"),
        _seed_row(
            row_id="ot-001",
            category="off_topic_refusal",
            tool_calls=[],
            final_text="I can only help with your health record.",
        ),
    ]
    p = tmp_path / "holdout.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")
    return p


# --------------------------------------------------------------------------
# `tool_call_equivalent` — full branch coverage.
# --------------------------------------------------------------------------


_PRED_A = [{"name": "get_vitals", "arguments": {}}]
_GOLD_A = [{"name": "get_vitals", "arguments": {}}]
_PRED_B_DIFF_ARGS = [{"name": "get_vitals", "arguments": {"window": "today"}}]
_PRED_C_DIFF_NAME = [{"name": "get_meds", "arguments": {}}]
_PARALLEL_GOLD = [
    {"name": "get_vitals", "arguments": {}},
    {"name": "get_meds", "arguments": {}},
]
_PARALLEL_PRED_REORDERED = [
    {"name": "get_meds", "arguments": {}},
    {"name": "get_vitals", "arguments": {}},
]


@pytest.mark.parametrize(
    ("pred", "gold", "category", "expected", "desc"),
    [
        (_PRED_A, _GOLD_A, "fact_lookup",
         eh.Equivalence.MATCH, "identical single call → MATCH"),
        (_PRED_B_DIFF_ARGS, _GOLD_A, "fact_lookup",
         eh.Equivalence.PARTIAL, "same name, different arguments → PARTIAL"),
        (_PRED_C_DIFF_NAME, _GOLD_A, "fact_lookup",
         eh.Equivalence.MISMATCH, "different name → MISMATCH"),
        ([], _GOLD_A, "fact_lookup",
         eh.Equivalence.MISMATCH, "predicted no-call when gold has calls → MISMATCH"),
        (_PRED_A, [], "fact_lookup",
         eh.Equivalence.MISMATCH, "predicted call when gold has none (non-refusal) → MISMATCH"),
        ([], [], "off_topic_refusal",
         eh.Equivalence.MATCH, "refusal: empty pred + empty gold → MATCH"),
        ([], [], "medical_advice_refusal",
         eh.Equivalence.MATCH, "medical refusal: empty pred + empty gold → MATCH"),
        (_PRED_A, [], "off_topic_refusal",
         eh.Equivalence.MISMATCH, "refusal violated: pred has call, gold empty → MISMATCH"),
        ([], _GOLD_A, "medical_advice_refusal",
         eh.Equivalence.MISMATCH, "refusal w/ gold call (authoring drift) → MISMATCH"),
        (_PARALLEL_GOLD, _PARALLEL_GOLD, "parallel_call",
         eh.Equivalence.MATCH, "parallel: same order → MATCH"),
        (_PARALLEL_PRED_REORDERED, _PARALLEL_GOLD, "parallel_call",
         eh.Equivalence.MISMATCH, "parallel: different order → MISMATCH (order matters)"),
        ([], [], "fact_lookup",
         eh.Equivalence.MATCH, "non-refusal: empty-vs-empty → MATCH"),
    ],
)
def test_tool_call_equivalent_branches(
    pred: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    category: str,
    expected: eh.Equivalence,
    desc: str,
) -> None:
    assert eh.tool_call_equivalent(pred, gold, category=category) is expected, desc


# --------------------------------------------------------------------------
# C5 regression: case-normalized arg comparison.
# --------------------------------------------------------------------------


def test_tool_call_equivalent_case_insensitive_args() -> None:
    """C5: tool resolvers are case-insensitive per M3 spec, so a case-only diff
    must MATCH (was scored PARTIAL pre-C5; flips tt-101 + te-104 in M6 holdout).
    """
    pred = [{"name": "get_med", "arguments": {"name": "Lisinopril"}}]
    gold = [{"name": "get_med", "arguments": {"name": "lisinopril"}}]
    assert eh.tool_call_equivalent(pred, gold, category="fact_lookup") is eh.Equivalence.MATCH


def test_tool_call_equivalent_non_str_args_passthrough() -> None:
    """C5: non-str values (ints, lists) pass through `_norm_args` unchanged.
    Verifies recursion is type-safe — a list of ints must not crash casefold.
    """
    pred = [{"name": "f", "arguments": {"x": 1, "y": [1, 2]}}]
    gold = [{"name": "f", "arguments": {"x": 1, "y": [1, 2]}}]
    assert eh.tool_call_equivalent(pred, gold, category="fact_lookup") is eh.Equivalence.MATCH


# --------------------------------------------------------------------------
# `extract_gold_trace`.
# --------------------------------------------------------------------------


def test_extract_gold_trace_call_plus_answer() -> None:
    row = _seed_row(row_id="fl-001")
    g = eh.extract_gold_trace(row)
    assert g.row_id == "fl-001"
    assert g.category == "fact_lookup"
    assert g.tool_calls == (
        {"name": "get_vitals", "arguments": {}},
    ), "single call extracted from the call-only assistant turn"
    assert g.assistant_text == "Your heart rate is 72 bpm.", (
        "assistant_text comes from the LAST assistant turn (post-</think>)"
    )


def test_extract_gold_trace_refusal_row_has_zero_calls() -> None:
    row = _seed_row(
        row_id="ot-001",
        category="off_topic_refusal",
        tool_calls=[],
        final_text="I can only help with your health record.",
    )
    g = eh.extract_gold_trace(row)
    assert g.tool_calls == (), "refusal → no flattened tool calls"
    assert g.assistant_text == "I can only help with your health record."


def test_extract_gold_trace_parallel_row_flattens_in_order() -> None:
    parallel = [
        {"id": "c0", "type": "function",
         "function": {"name": "get_vitals", "arguments": {}}},
        {"id": "c1", "type": "function",
         "function": {"name": "get_meds", "arguments": {}}},
    ]
    row = _seed_row(row_id="pc-001", category="parallel_call", tool_calls=parallel)
    g = eh.extract_gold_trace(row)
    assert [c["name"] for c in g.tool_calls] == ["get_vitals", "get_meds"], (
        "parallel calls (single assistant turn) preserved in conversation order"
    )


def test_extract_gold_trace_multi_turn_returns_first_turn_only() -> None:
    """Two-turn row (e.g. tt-* in holdout) has 2 assistant turns each with a
    tool_call. The eval inference path only generates the FIRST response
    from the user prompt, so gold must be the FIRST turn's calls — NOT
    flattened across all turns. Without this, every multi-turn row scores
    MISMATCH on len(pred=1) vs len(gold=2).
    """
    row = {
        "id": "tt-001",
        "category": "two_turn",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Look up Lisinopril"},
            {
                "role": "assistant",
                "content": "<think>...</think>",
                "tool_calls": [{"id": "c0", "type": "function",
                    "function": {"name": "get_medication_by_name",
                                 "arguments": {"name": "Lisinopril"}}}],
            },
            {"role": "tool", "name": "get_medication_by_name",
             "tool_call_id": "c0", "content": json.dumps({"dose": "10mg"})},
            {"role": "user", "content": "Now Atorvastatin"},
            {
                "role": "assistant",
                "content": "<think>...</think>",
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "get_medication_by_name",
                                 "arguments": {"name": "Atorvastatin"}}}],
            },
            {"role": "tool", "name": "get_medication_by_name",
             "tool_call_id": "c1", "content": json.dumps({"dose": "20mg"})},
            {"role": "assistant",
             "content": "<think>x</think>\nLisinopril 10mg, Atorvastatin 20mg.",
             "tool_calls": []},
        ],
        "tools": [{"type": "function", "function": {"name": "get_medication_by_name"}}],
    }
    g = eh.extract_gold_trace(row)
    assert len(g.tool_calls) == 1, (
        f"multi-turn row must extract first-turn only; got {len(g.tool_calls)} calls"
    )
    assert g.tool_calls[0]["arguments"] == {"name": "Lisinopril"}
    assert g.assistant_text == "Lisinopril 10mg, Atorvastatin 20mg.", (
        "assistant_text still comes from the LAST turn (final NL answer)"
    )


# --------------------------------------------------------------------------
# CLI: `--list-categories` and `--dry-run`.
# --------------------------------------------------------------------------


def test_list_categories_succeeds_when_holdout_exists(
    tmp_holdout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = eh.main(["--holdout", str(tmp_holdout), "--list-categories"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fact_lookup" in out
    assert "off_topic_refusal" in out
    assert "2" in out, "fact_lookup count appears"


def test_list_categories_exit_2_when_holdout_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.jsonl"
    rc = eh.main(["--holdout", str(missing), "--list-categories"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "holdout not found" in err


def test_dry_run_succeeds_when_holdout_exists(
    tmp_holdout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = eh.main(["--holdout", str(tmp_holdout), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "all gold-vs-gold" in out
    assert "100.0%" in out, "every category at 100 % when scoring gold against itself"
    assert "PASS" in out, "every category clears the 80 % bar at 100 %"


def test_dry_run_exit_2_when_holdout_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.jsonl"
    rc = eh.main(["--holdout", str(missing), "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "holdout not found" in err


def test_full_eval_requires_checkpoint(
    tmp_holdout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --list-categories or --dry-run, omitting --checkpoint exits 2."""
    rc = eh.main(["--holdout", str(tmp_holdout)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--checkpoint" in err


def test_full_eval_exits_2_when_checkpoint_missing(
    tmp_holdout: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_ckpt = tmp_path / "no_such_checkpoint"
    rc = eh.main(
        ["--holdout", str(tmp_holdout), "--checkpoint", str(missing_ckpt)]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "checkpoint not found" in err


def test_parse_function_calls_extracts_fg_wire_format() -> None:
    """The wire-format parser is the M6 host-testable part of run_inference.

    The full `run_inference()` pulls torch + transformers and loads a 500 MB
    BF16 checkpoint, so it lives behind an SSH boundary on the server. The
    pure-text parser is what we exercise here — it converts the FG wire
    format the model emits into the `{"name", "arguments"}` shape that
    `tool_call_equivalent` consumes.
    """
    text = (
        "<think>Look up Lisinopril.</think>"
        "<start_function_call>call:get_medication_by_name"
        "{name:<escape>Lisinopril<escape>}<end_function_call>"
    )
    calls = eh._parse_function_calls(text)
    assert calls == [{"name": "get_medication_by_name", "arguments": {"name": "Lisinopril"}}]

    # Empty-args call.
    calls = eh._parse_function_calls(
        "<start_function_call>call:get_vitals{}<end_function_call>"
    )
    assert calls == [{"name": "get_vitals", "arguments": {}}]

    # Two adjacent calls (parallel).
    text2 = (
        "<start_function_call>call:list_allergies{}<end_function_call>"
        "<start_function_call>call:get_vitals{}<end_function_call>"
    )
    calls = eh._parse_function_calls(text2)
    assert [c["name"] for c in calls] == ["list_allergies", "get_vitals"]

    # Refusal — no call markers.
    assert eh._parse_function_calls("I cannot give medical advice.") == []
