"""Seed JSONL validator tests.

Asserts the committed `data/dispenser_demo/seed_conversations.jsonl` validates
at pass_rate=1.0 against the 5-tool dispenser registry, AND exercises the
validator against a handful of synthetic bad rows to confirm it actually
catches the violations the seed-authoring contract depends on.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gemma_tools.dispenser_demo.dataset import (
    SYSTEM_TRIGGER,
    validate_conversation,
    validate_file,
)

_REPO = Path(__file__).resolve().parents[2]
_SEEDS_PATH = _REPO / "data" / "dispenser_demo" / "seed_conversations.jsonl"


# --------------------------------------------------------------------------
# File-level: the committed seed file validates at 1.0 with the expected
# per-category counts.
# --------------------------------------------------------------------------


def test_seed_file_validates_at_full_pass_rate() -> None:
    report = validate_file(_SEEDS_PATH, min_pass_rate=1.0)
    if not report.meets_threshold:
        # Surface the first few failures inline so a regression is debuggable
        # from the pytest output alone.
        sample = "\n".join(
            f"  {f.row_id}: {'; '.join(f.errors)}" for f in report.failures[:5]
        )
        pytest.fail(
            f"seed validation failed: pass_rate={report.pass_rate:.4f}, "
            f"failures:\n{sample}"
        )
    assert report.total == 42, "expected 42 seed rows"
    assert report.category_counts == {
        "patient_profile": 8,
        "next_appointment": 8,
        "emergency_contact": 8,
        "dispense": 8,
        "out_of_scope_refusal": 10,
    }, "expected 8 rows per domain category and 10 in out_of_scope_refusal"


# --------------------------------------------------------------------------
# Round-trippable fixture row. Built fresh per test (no shared mutable state)
# so each negative case can mutate one field without polluting the next case.
# --------------------------------------------------------------------------


def _fixture_row() -> dict[str, Any]:
    """A minimal valid row: emergency_contact intent against the v2 fixture."""
    return {
        "id": "fixture-001",
        "category": "emergency_contact",
        "messages": [
            {"role": "system", "content": SYSTEM_TRIGGER},
            {"role": "user", "content": "Who's my emergency contact?"},
            {
                "role": "assistant",
                "content": "<think>Emergency contact lookup.</think>",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_emergency_contact",
                            "arguments": {},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_emergency_contact",
                "tool_call_id": "call_1",
                "content": json.dumps(
                    {
                        "name": "Jane Doe",
                        "relation": "daughter",
                        "phone": "+1-555-0142",
                        "phone_words": "plus one five five five zero one four two",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": "<think>Name plus relation.</think>\nYour emergency contact is Jane Doe.",
            },
        ],
        "tools": _five_tool_block(),
    }


def _five_tool_block() -> list[dict[str, Any]]:
    """Minimal `tools[]` block — names only; the validator checks names against
    the registry but does not enforce description text on each row."""
    names = [
        "get_patient_profile",
        "get_next_appointment",
        "get_emergency_contact",
        "dispense_medication",
        "refuse_out_of_scope",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": "stub",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
            },
        }
        for n in names
    ]


def test_fixture_row_validates_clean() -> None:
    outcome = validate_conversation(_fixture_row())
    assert outcome.ok, f"baseline row must validate: {outcome.errors}"


# --------------------------------------------------------------------------
# Negative cases — each builds the fixture, mutates one field, asserts the
# expected substring shows up in the validator's error list.
# --------------------------------------------------------------------------


def _mutate_tool_content(row: dict[str, Any], new_content: dict[str, Any]) -> dict[str, Any]:
    """Replace the tool message's JSON content with `new_content` (serialized)."""
    row["messages"][3]["content"] = json.dumps(new_content)
    return row


# | mutation_desc                                  | builder                                                              | expect_substring             |
@pytest.mark.parametrize(
    ("desc", "build", "expect_substring"),
    [
        (
            "missing *_words companion on phone",
            lambda r: _mutate_tool_content(
                r, {"name": "Jane Doe", "relation": "daughter", "phone": "+1-555-0142"}
            ),
            "phone_words",
        ),
        (
            "companion contains digits",
            lambda r: _mutate_tool_content(
                r,
                {
                    "name": "Jane Doe",
                    "relation": "daughter",
                    "phone": "+1-555-0142",
                    "phone_words": "plus 1 five five five zero one four two",
                },
            ),
            "digit-free",
        ),
        (
            "wrong system trigger",
            lambda r: (r["messages"][0].update(content="You are Sago.") or r),
            "SYSTEM_TRIGGER",
        ),
        (
            "unknown tool name",
            lambda r: (
                r["messages"][2]["tool_calls"][0]["function"].update(name="get_vitals")
                or r
            ),
            "unknown tool",
        ),
        (
            "assistant final NL turn missing newline after </think>",
            lambda r: (
                r["messages"][4].update(
                    content="<think>x</think>Your emergency contact is Jane Doe."
                )
                or r
            ),
            "after </think>",
        ),
    ],
)
def test_validator_rejects_known_violations(
    desc: str, build: Any, expect_substring: str
) -> None:
    row = copy.deepcopy(_fixture_row())
    row = build(row)
    outcome = validate_conversation(row)
    assert not outcome.ok, f"{desc}: expected rejection, got pass"
    flat = "\n".join(outcome.errors)
    assert expect_substring in flat, (
        f"{desc}: expected substring {expect_substring!r} in errors:\n{flat}"
    )


def test_validator_rejects_invalid_refusal_reason() -> None:
    """The `refuse_out_of_scope.reason` enum is locked to two values."""
    row = copy.deepcopy(_fixture_row())
    row["messages"][2]["tool_calls"][0]["function"] = {
        "name": "refuse_out_of_scope",
        "arguments": {"reason": "weather"},  # not in the enum
    }
    row["messages"][3]["name"] = "refuse_out_of_scope"
    row["messages"][3]["content"] = json.dumps(
        {"status": "refused", "reason": "weather"}
    )
    outcome = validate_conversation(row)
    assert not outcome.ok, "invalid enum value must be rejected"
    flat = "\n".join(outcome.errors)
    assert "reason" in flat, f"expected reason-related error, got: {flat}"
