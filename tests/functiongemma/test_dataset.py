"""Tests for the FunctionGemma seed dataset (M4) and its validator.

Coverage targets per `docs/plans/FunctionGemma/README.md` §9.6 G_DATASET_SHAPE
and the M4 acceptance row in §14.

Conventions: table-driven via `@pytest.mark.parametrize` with a `desc` column
per `docs/conventions/code-style-python.md` §10. ≥ 2 cases per test function.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gemma_tools.functiongemma.dataset import (
    SYSTEM_TRIGGER,
    backfill_tool_message_names,
    load_jsonl,
    render_training_text,
    validate_conversation,
    validate_file,
)
from gemma_tools.functiongemma.tools import default_registry

_REPO = Path(__file__).resolve().parents[2]
_SEED_PATH = _REPO / "data" / "functiongemma" / "seed_conversations.jsonl"

# Per §9.3 of the plan. The M4 acceptance row in §14 ("≈ 50 hand seeds")
# permits drift; the test asserts the exact count + per-category split that
# the build script emits, so any future drift surfaces here loudly.
_TAXONOMY_TARGETS: dict[str, int] = {
    "fact_lookup": 12,
    "off_topic_refusal": 4,
    "fact_absence": 4,
    "parallel_call": 6,
    "two_turn": 14,
    "medical_advice_refusal": 4,
    "tool_error_recovery": 6,
}
_EXPECTED_TOTAL = sum(_TAXONOMY_TARGETS.values())  # 50


# --------------------------------------------------------------------------
# Module-scoped fixtures: load + validate once, reuse across test functions
# so we don't re-parse the JSONL 30 times.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_rows() -> list[dict[str, Any]]:
    return list(load_jsonl(_SEED_PATH))


@pytest.fixture(scope="module")
def report() -> Any:
    return validate_file(_SEED_PATH)


# --------------------------------------------------------------------------
# G_DATASET_SHAPE — file-level acceptance (M4 gate).
# --------------------------------------------------------------------------


def test_seed_file_exists() -> None:
    assert _SEED_PATH.exists(), f"missing seed file at {_SEED_PATH}"
    assert _SEED_PATH.stat().st_size > 0, "seed file is empty"


def test_row_count_matches_taxonomy(raw_rows: list[dict[str, Any]]) -> None:
    """Acceptance: exactly 50 rows. The §14 row says \"~50\"; we hold to the
    deterministic build-script output so drift surfaces in the diff."""
    assert len(raw_rows) == _EXPECTED_TOTAL, (
        f"expected {_EXPECTED_TOTAL} rows, got {len(raw_rows)}"
    )


def test_all_rows_parse_as_json(raw_rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(raw_rows):
        assert isinstance(row, dict), f"row {i}: not a dict — {type(row).__name__}"


def test_validator_pass_rate_meets_acceptance(report: Any) -> None:
    """M4 acceptance: ≥ 95%. Hand-authored seeds should hit 100%."""
    assert report.total == _EXPECTED_TOTAL
    assert report.pass_rate >= 0.95, (
        f"pass_rate={report.pass_rate:.4f} below 0.95; failures:\n"
        + "\n".join(f"  {f.row_id}: {f.errors}" for f in report.failures)
    )


def test_validator_meets_threshold_resolves() -> None:
    """`min_pass_rate` is part of the public contract — the report carries
    both the requested threshold and the resolved boolean so callers do not
    re-derive it."""
    high = validate_file(_SEED_PATH, min_pass_rate=0.95)
    assert high.min_pass_rate == 0.95
    assert high.meets_threshold is True

    impossible = validate_file(_SEED_PATH, min_pass_rate=1.5)
    assert impossible.min_pass_rate == 1.5
    assert impossible.meets_threshold is False, (
        "pass_rate cannot exceed 1.0, so a 1.5 threshold must always fail"
    )


def test_validator_pass_rate_is_total(report: Any) -> None:
    """Hand seeds should hit 100% — anything less is authoring drift."""
    assert report.pass_rate == 1.0, (
        f"pass_rate={report.pass_rate:.4f} not 1.0; failures:\n"
        + "\n".join(f"  {f.row_id}: {f.errors}" for f in report.failures)
    )


def test_taxonomy_counts_match_targets(report: Any) -> None:
    """Per-category split must match §9.3 to within 0 (we are deterministic)."""
    for category, want in _TAXONOMY_TARGETS.items():
        got = report.category_counts.get(category, 0)
        assert got == want, f"{category}: got {got}, want {want}"
    # No unexpected categories crept in.
    extras = set(report.category_counts) - set(_TAXONOMY_TARGETS)
    assert not extras, f"unexpected categories: {sorted(extras)}"


# --------------------------------------------------------------------------
# Per-row shape — roles, system trigger, JSON-encoded tool content.
# --------------------------------------------------------------------------


def _iter_messages(raw_rows: list[dict[str, Any]]) -> Iterator[tuple[int, int, dict[str, Any]]]:
    for ri, row in enumerate(raw_rows):
        for mi, m in enumerate(row["messages"]):
            yield ri, mi, m


def test_every_role_is_allowed(raw_rows: list[dict[str, Any]]) -> None:
    allowed = {"system", "user", "assistant", "tool"}
    for ri, mi, m in _iter_messages(raw_rows):
        assert m["role"] in allowed, f"row {ri} msg {mi}: bad role {m['role']!r}"


def test_first_message_is_system_trigger(raw_rows: list[dict[str, Any]]) -> None:
    """System trigger is the FG-mode activator; drift here would silently
    disable function-calling at training time. Verbatim equality is the gate."""
    for ri, row in enumerate(raw_rows):
        first = row["messages"][0]
        assert first["role"] == "system", f"row {ri}: first message not system"
        assert first["content"] == SYSTEM_TRIGGER, (
            f"row {ri}: system content drift — got {first['content']!r}"
        )


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def test_assistant_content_has_exactly_one_think_block(
    raw_rows: list[dict[str, Any]],
) -> None:
    """Per §9.4.2 — one <think>...</think> block per assistant turn. Both the
    \"with tool_calls\" and \"without tool_calls\" branches share this rule.
    """
    for ri, mi, m in _iter_messages(raw_rows):
        if m["role"] != "assistant":
            continue
        n = len(_THINK_RE.findall(m["content"]))
        assert n == 1, (
            f"row {ri} msg {mi}: expected 1 <think> block, got {n}; "
            f"content={m['content']!r}"
        )


def test_assistant_with_tool_calls_has_no_nl_tail(
    raw_rows: list[dict[str, Any]],
) -> None:
    """Per the assistant-content shape decision (validator module docstring):
    when an assistant turn carries `tool_calls`, content ends immediately
    after </think>. A trailing NL would emit a stray text turn during
    training."""
    for ri, mi, m in _iter_messages(raw_rows):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        assert m["content"].endswith("</think>"), (
            f"row {ri} msg {mi}: assistant w/ tool_calls has tail after </think>"
        )


def test_assistant_without_tool_calls_has_nl_answer(
    raw_rows: list[dict[str, Any]],
) -> None:
    """Final-answer / refusal turns must have `\\n<answer>` after </think>."""
    for ri, mi, m in _iter_messages(raw_rows):
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        content = m["content"]
        assert "</think>\n" in content, (
            f"row {ri} msg {mi}: missing newline after </think>"
        )
        # The bit after </think>\n must be a non-empty answer.
        tail = content.split("</think>", 1)[1]
        assert tail.startswith("\n"), f"row {ri} msg {mi}: missing leading \\n"
        assert tail.strip(), f"row {ri} msg {mi}: empty answer after <think> block"


def test_every_tool_call_references_a_known_tool(
    raw_rows: list[dict[str, Any]],
) -> None:
    """Per §9.6 rule 3 — tool_calls[*].function.name must exist in the M3 registry."""
    registry = default_registry()
    for ri, mi, m in _iter_messages(raw_rows):
        for tc in m.get("tool_calls", []) or []:
            name = tc["function"]["name"]
            assert name in registry, (
                f"row {ri} msg {mi}: tool_call {tc['id']} → unknown tool {name!r}"
            )


def test_every_tool_call_argument_validates(raw_rows: list[dict[str, Any]]) -> None:
    """Args must validate against the M3 Pydantic args_model."""
    registry = default_registry()
    for _ri, _mi, m in _iter_messages(raw_rows):
        for tc in m.get("tool_calls", []) or []:
            spec = registry[tc["function"]["name"]]
            # Will raise ValidationError on drift — pytest reports it as a
            # plain test failure with the Pydantic error chain attached.
            spec.args_model.model_validate(tc["function"]["arguments"])


def test_every_tool_message_matches_a_prior_tool_call(
    raw_rows: list[dict[str, Any]],
) -> None:
    """Per §9.6 rule 5 — tool messages must follow an assistant tool_call
    with a matching id. Authoring shortcut: same id may appear once on the
    assistant call and once on the tool response, never twice on either."""
    for ri, row in enumerate(raw_rows):
        pending: dict[str, str] = {}
        for mi, m in enumerate(row["messages"]):
            if m["role"] == "assistant":
                for tc in m.get("tool_calls", []) or []:
                    assert tc["id"] not in pending, (
                        f"row {ri} msg {mi}: duplicate tool_call id {tc['id']!r}"
                    )
                    pending[tc["id"]] = tc["function"]["name"]
            elif m["role"] == "tool":
                tcid = m["tool_call_id"]
                assert tcid in pending, (
                    f"row {ri} msg {mi}: tool message id {tcid!r} has no prior tool_call"
                )
                assert m["name"] == pending[tcid], (
                    f"row {ri} msg {mi}: tool name {m['name']!r} != tool_call name {pending[tcid]!r}"
                )


def test_every_tool_message_content_is_valid_json(raw_rows: list[dict[str, Any]]) -> None:
    for ri, mi, m in _iter_messages(raw_rows):
        if m["role"] != "tool":
            continue
        try:
            json.loads(m["content"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"row {ri} msg {mi}: tool content not JSON — {exc.msg}")


def test_refusal_rows_have_no_tool_calls(raw_rows: list[dict[str, Any]]) -> None:
    """Off-topic + medical-advice refusals must not call any tool — that is
    the *whole* training signal of those rows."""
    for ri, row in enumerate(raw_rows):
        if row.get("category") not in {"off_topic_refusal", "medical_advice_refusal"}:
            continue
        for mi, m in enumerate(row["messages"]):
            if m["role"] == "assistant":
                tc = m.get("tool_calls", []) or []
                assert tc == [], (
                    f"row {ri} ({row['category']}) msg {mi}: refusal must not "
                    f"emit tool_calls, got {[t['function']['name'] for t in tc]}"
                )


def test_per_row_tools_block_is_full_registry(raw_rows: list[dict[str, Any]]) -> None:
    """Convention pinned in the seed-authoring recipe: every row carries the
    full 7-tool registry. Drift surfaces here so any per-row subsetting
    requires explicit author intent + a test update."""
    expected = set(default_registry().keys())
    for ri, row in enumerate(raw_rows):
        names = {t["function"]["name"] for t in row["tools"]}
        assert names == expected, (
            f"row {ri}: tools block mismatch — got {sorted(names)}, want {sorted(expected)}"
        )


# --------------------------------------------------------------------------
# Privacy / PHI guard. Beyond the dedicated PHI scanner, the seed-content
# itself must contain no real-looking PHI patterns.
# --------------------------------------------------------------------------


_SSN_PAT = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE_PAT = re.compile(r"\+1-(\d{3})-\d{3,4}(?:-\d{4})?")
_EMAIL_PAT = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def test_seed_file_has_no_real_phi_patterns() -> None:
    text = _SEED_PATH.read_text(encoding="utf-8")
    assert not _SSN_PAT.search(text), "seed file contains an SSN-like pattern"
    assert not _EMAIL_PAT.search(text), "seed file contains an email-like pattern"
    for m in _PHONE_PAT.finditer(text):
        area = m.group(1)
        assert area == "555", (
            f"seed file contains a non-+1-555- US phone: {m.group(0)!r}"
        )


# --------------------------------------------------------------------------
# Validator unit tests — exercise validate_conversation directly with
# small, focused inputs.
# --------------------------------------------------------------------------


def _minimal_tools() -> list[dict[str, Any]]:
    """Full-registry tools list, harvested from the registry to dodge drift."""
    from gemma_tools.functiongemma.tools import as_function_declarations
    return as_function_declarations()


def _system_msg() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_TRIGGER}


def test_validate_conversation_accepts_minimal_refusal() -> None:
    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "tell me a joke"},
            {
                "role": "assistant",
                "content": "<think>Off-topic.</think>\nI answer questions from your health record only.",
            },
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert out.ok, out.errors


def test_validate_conversation_rejects_bad_system_content() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "you are a model"},   # wrong text
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "<think>x</think>\nhi",
            },
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok
    assert any("SYSTEM_TRIGGER" in e for e in out.errors)


# | mutation                                  | expect_substring                  | desc                                                |
@pytest.mark.parametrize(
    ("mutation", "expect_substring", "desc"),
    [
        ("missing_think",  "expected exactly 1 <think>",       "no <think> block"),
        ("two_think",      "expected exactly 1 <think>",       "duplicate <think> blocks"),
        ("tail_after_call","content must end immediately",     "tail after </think> on tool_call turn"),
        ("no_nl_after",    "must have '\\n<answer>'",          "no NL after </think> on answer turn"),
    ],
)
def test_validate_conversation_assistant_shape_drifts(
    mutation: str, expect_substring: str, desc: str,
) -> None:
    """Each mutation pokes a different branch of `_validate_assistant_content_shape`."""
    base_assistant_call = {
        "role": "assistant",
        "content": "<think>x</think>",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}}
        ],
    }
    base_assistant_answer = {
        "role": "assistant",
        "content": "<think>HR=72.</think>\nYour HR is 72.",
    }
    if mutation == "missing_think":
        base_assistant_answer = {"role": "assistant", "content": "Your HR is 72."}
    elif mutation == "two_think":
        base_assistant_answer = {
            "role": "assistant",
            "content": "<think>x</think>\n<think>y</think>\nanswer",
        }
    elif mutation == "tail_after_call":
        base_assistant_call = dict(base_assistant_call, content="<think>x</think>extra")
    elif mutation == "no_nl_after":
        base_assistant_answer = {
            "role": "assistant",
            "content": "<think>x</think>answer",   # no \n
        }

    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "hr?"},
            base_assistant_call,
            {"role": "tool", "name": "get_vitals", "tool_call_id": "call_1", "content": "{}"},
            base_assistant_answer,
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok, desc
    assert any(expect_substring in e for e in out.errors), f"{desc}: {out.errors}"


def test_validate_conversation_rejects_unknown_tool() -> None:
    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "<think>x</think>",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "schedule_appointment", "arguments": {}}}
                ],
            },
            {"role": "tool", "name": "schedule_appointment", "tool_call_id": "c1", "content": "{}"},
            {"role": "assistant", "content": "<think>x</think>\ndone"},
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok
    assert any("unknown tool" in e for e in out.errors), out.errors


def test_validate_conversation_rejects_invalid_arguments() -> None:
    """`get_medications_at_time` requires `time_24h` — missing it must fail."""
    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "morning meds"},
            {
                "role": "assistant",
                "content": "<think>x</think>",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_medications_at_time", "arguments": {}}}
                ],
            },
            {"role": "tool", "name": "get_medications_at_time", "tool_call_id": "c1", "content": "[]"},
            {"role": "assistant", "content": "<think>x</think>\ndone"},
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok
    assert any("invalid arguments" in e for e in out.errors), out.errors


def test_validate_conversation_rejects_unmatched_tool_call_id() -> None:
    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "<think>x</think>\nhi"},
            {"role": "tool", "name": "get_vitals", "tool_call_id": "ghost", "content": "{}"},
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok
    assert any("does not match any prior" in e for e in out.errors), out.errors


def test_validate_conversation_rejects_non_json_tool_content() -> None:
    row = {
        "messages": [
            _system_msg(),
            {"role": "user", "content": "x"},
            {
                "role": "assistant",
                "content": "<think>x</think>",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}}
                ],
            },
            {"role": "tool", "name": "get_vitals", "tool_call_id": "c1", "content": "not json{"},
            {"role": "assistant", "content": "<think>x</think>\nhi"},
        ],
        "tools": _minimal_tools(),
    }
    out = validate_conversation(row)
    assert not out.ok
    assert any("not valid JSON" in e for e in out.errors), out.errors


# --------------------------------------------------------------------------
# Helper — backfill_tool_message_names matches Unsloth notebook cell 23.
# --------------------------------------------------------------------------


def test_backfill_tool_message_names_fills_missing_name() -> None:
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "<think>x</think>",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    backfill_tool_message_names(msgs)
    assert msgs[1]["name"] == "get_vitals"


def test_backfill_tool_message_names_leaves_existing_name_alone() -> None:
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "<think>x</think>",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "manually_set", "content": "{}"},
    ]
    backfill_tool_message_names(msgs)
    assert msgs[1]["name"] == "manually_set"


# --------------------------------------------------------------------------
# Optional: tokenizer-render check. Skipped unless the local FG tokenizer
# is present at ~/hf-cache/functiongemma-270m-it.
# --------------------------------------------------------------------------


_TOKENIZER_DIR = Path(os.path.expanduser("~/hf-cache/functiongemma-270m-it"))
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(
    not _TOKENIZER_DIR.exists() or not _TRANSFORMERS_AVAILABLE,
    reason="local FG tokenizer or transformers not available",
)
def test_render_training_text_strips_double_bos() -> None:
    """The chat template emits a leading `<bos>`. `render_training_text` must
    strip it so SFTTrainer's tokenize-time `add_bos=True` does not double up.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR))
    rows = list(load_jsonl(_SEED_PATH))
    # Render a representative row from each non-trivial category.
    sample_ids = ["fl-001", "ot-001", "pc-001", "tt-001", "te-001"]
    samples = [r for r in rows if r["id"] in sample_ids]
    assert len(samples) == len(sample_ids), "missing seed ids for render check"
    for row in samples:
        text = render_training_text(row, tok)
        assert isinstance(text, str)
        assert not text.startswith("<bos>"), f"row {row['id']}: leading <bos> not stripped"
        # Sanity: the rendered prompt should at least carry the tool name and
        # the developer trigger fragment.
        assert "function calling with the following functions" in text
        # The <think> block from the assistant turn must round-trip through
        # the chat template — if it does not, the response-only loss mask
        # would never see the reasoning prelude at training time.
        assert "<think>" in text, f"row {row['id']}: <think> tag dropped by chat template"
        assert "</think>" in text, f"row {row['id']}: </think> tag dropped by chat template"
