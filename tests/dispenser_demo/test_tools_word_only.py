"""Tool-boundary `*_words`-companion invariant against the LIVE registry.

For every tool in `default_registry()`, invoke it with a valid input against
the canonical `data/health_table_v2.yaml` fixture and assert the response
satisfies the invariant: every digit-bearing key has a digit-free
`<key>_words` sibling.

NOTE: this test does NOT exercise the model's free narration — that lives in
the assistant content of the seed JSONL and is intentionally outside the
invariant (plan §1 / §10 / §11 R5). The companion check on tool messages
embedded in the seed file is in `test_dataset_validator.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_tools.dispenser_demo.dataset import find_word_only_violations
from gemma_tools.dispenser_demo.health_table_v2 import (
    HealthTableV2,
    load_health_table_v2,
)
from gemma_tools.dispenser_demo.tools import (
    default_registry,
    execute_tool,
)

_REPO = Path(__file__).resolve().parents[2]
_TABLE_PATH = _REPO / "data" / "health_table_v2.yaml"


@pytest.fixture(scope="module")
def table_v2() -> HealthTableV2:
    return load_health_table_v2(_TABLE_PATH)


# --------------------------------------------------------------------------
# Live-tool invariant. One row per tool; refuse_out_of_scope is exercised
# twice (one row per reason in the enum) to confirm the response is
# invariant-clean regardless of which reason is emitted.
# --------------------------------------------------------------------------


# | tool_name              | arguments                  | desc                                       |
@pytest.mark.parametrize(
    ("tool_name", "arguments", "desc"),
    [
        ("get_patient_profile", {}, "profile response satisfies invariant"),
        ("get_next_appointment", {}, "appointment response satisfies invariant"),
        ("get_emergency_contact", {}, "contact response satisfies invariant"),
        ("dispense_medication", {}, "dispense response satisfies invariant"),
        (
            "refuse_out_of_scope",
            {"reason": "health_advice"},
            "refusal w/ health_advice reason satisfies invariant",
        ),
        (
            "refuse_out_of_scope",
            {"reason": "off_topic"},
            "refusal w/ off_topic reason satisfies invariant",
        ),
    ],
)
def test_tool_response_invariant(
    table_v2: HealthTableV2,
    tool_name: str,
    arguments: dict[str, object],
    desc: str,
) -> None:
    response = execute_tool(tool_name, arguments, table_v2)
    violations = find_word_only_violations(response, path=f"<{tool_name}>")
    assert violations == [], f"{desc}: {violations}"


def test_registry_has_exactly_five_tools() -> None:
    """Lock the registry size against accidental drift (plan §7)."""
    registry = default_registry()
    assert set(registry.keys()) == {
        "get_patient_profile",
        "get_next_appointment",
        "get_emergency_contact",
        "dispense_medication",
        "refuse_out_of_scope",
    }, "registry must contain exactly the 5 plan §7 tools"


# --------------------------------------------------------------------------
# Live-vs-seed shape alignment. The seed JSONL encodes a specific tool
# response shape (used to train the model); the live registry must produce
# the SAME shape, otherwise the model trains on one contract and the runtime
# emits another. These tests lock the alignment.
# --------------------------------------------------------------------------


# | tool_name              | arguments                  | expected_response               | desc                              |
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_response", "desc"),
    [
        (
            "refuse_out_of_scope",
            {"reason": "health_advice"},
            {"status": "refused"},
            "refusal w/ health_advice — no reason echo (seed contract)",
        ),
        (
            "refuse_out_of_scope",
            {"reason": "off_topic"},
            {"status": "refused"},
            "refusal w/ off_topic — no reason echo (seed contract)",
        ),
        (
            "dispense_medication",
            {},
            {"status": "dispensed"},
            "Phase 1.2 dispense stub — happy path, matches 7 of 8 dispense seeds",
        ),
    ],
)
def test_live_tool_matches_seed_response_shape(
    table_v2: HealthTableV2,
    tool_name: str,
    arguments: dict[str, object],
    expected_response: dict[str, object],
    desc: str,
) -> None:
    response = execute_tool(tool_name, arguments, table_v2)
    assert response == expected_response, desc


# --------------------------------------------------------------------------
# Negative cases for `find_word_only_violations` — the helper must actually
# detect the violations the live tests rely on, otherwise the live tests
# could pass trivially against a no-op walker.
# --------------------------------------------------------------------------


# | obj                                                       | expect_violation | desc                                     |
@pytest.mark.parametrize(
    ("obj", "expect_violation", "desc"),
    [
        ({"age": 45, "age_words": "forty five"}, False, "well-formed pair"),
        ({"age": 45}, True, "missing companion"),
        ({"age": 45, "age_words": "45"}, True, "companion contains digits"),
        (
            {"phone": "+1-555-0142", "phone_words": "plus one five five five zero one four two"},
            False,
            "string field with digit-free companion",
        ),
        (
            {"diagnoses": ["Type 2 Diabetes"], "diagnoses_words": "Type Two Diabetes"},
            False,
            "list-of-strings field with companion",
        ),
        (
            {"diagnoses": ["Type 2 Diabetes"]},
            True,
            "list with digit-bearing element missing companion",
        ),
        ({"status": "dispensed"}, False, "no digits, no companion needed"),
        (
            {"status": "dispensed", "reason": "off_topic"},
            False,
            "refusal response shape",
        ),
        (
            {
                "outer_words": "forty five",
                "nested": {"age": 45, "age_words": "forty five"},
            },
            False,
            "nested dict walked recursively, well-formed",
        ),
        (
            {"nested": {"age": 45}},
            True,
            "nested dict walked recursively, missing companion",
        ),
    ],
)
def test_find_word_only_violations(
    obj: dict[str, object], expect_violation: bool, desc: str
) -> None:
    violations = find_word_only_violations(obj)
    assert bool(violations) == expect_violation, f"{desc}: {violations}"
