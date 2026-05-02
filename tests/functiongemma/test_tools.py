"""Tests for the FunctionGemma read-only tool registry.

Coverage objective per `docs/plans/FunctionGemma/README.md` §3.1 G_TOOLS_TESTS:
≥ 90 % branch coverage on `src/gemma_tools/functiongemma_tools.py`. Each test
function carries ≥ 2 cases (parametrize / multiple asserts with `desc`)
per `docs/conventions/code-style-python.md` §10.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gemma_tools.functiongemma.tools import (
    EmptyArgs,
    FoodArgs,
    NameArgs,
    TimeArgs,
    ToolSpec,
    as_function_declarations,
    default_registry,
    execute_tool,
)
from gemma_tools.health_table import (
    HealthTable,
    Patient,
    Vitals,
    load_health_table,
)

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO / "data" / "health_table_v1.yaml"
_TOOLS_YAML = _REPO / "data" / "functiongemma" / "tools_v1.yaml"


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table() -> HealthTable:
    return load_health_table(_FIXTURE)


@pytest.fixture
def empty_table() -> HealthTable:
    """Synthetic minimal table with all optional sections empty.

    Used for the negative-empty-section assertions: tools that depend on
    optional blocks must degrade to an explicit error dict, not raise.
    """
    return HealthTable(
        patient=Patient(name="Empty", age=30),
        vitals=Vitals(
            heart_rate_bpm=70,
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            spo2_percent=98,
            body_temperature_c=36.6,
            respiratory_rate=16,
            last_measured=None,
        ),
        notes=(),
    )


# --------------------------------------------------------------------------
# Registry shape.
# --------------------------------------------------------------------------


# | tool_name                   | desc                                                      |
@pytest.mark.parametrize(
    ("tool_name", "desc"),
    [
        ("get_vitals",                   "vitals tool registered"),
        ("get_medications_at_time",      "schedule-time lookup tool registered"),
        ("get_medication_by_name",       "name lookup tool registered"),
        ("list_allergies",               "allergies tool registered"),
        ("check_food_interaction",       "food interaction tool registered"),
        ("get_next_appointment",         "next-appointment tool registered"),
        ("get_emergency_contact",        "emergency-contact tool registered"),
    ],
)
def test_default_registry_lists_all_tools(tool_name: str, desc: str) -> None:
    reg = default_registry()
    assert tool_name in reg, desc
    assert isinstance(reg[tool_name], ToolSpec), desc


def test_default_registry_returns_fresh_copy_each_call() -> None:
    """The registry helper must not return the module-level dict — callers
    should be safe to mutate the returned mapping for ad-hoc filtering
    without poisoning the global state."""
    a = default_registry()
    b = default_registry()
    a["sentinel"] = ToolSpec(
        name="sentinel", description="x", args_model=EmptyArgs, handler=lambda *_: None,
    )
    assert "sentinel" not in b, "registry mutation leaked across calls"


def test_function_declarations_shape() -> None:
    """The HF chat-template `tools=` arg expects a list of {type=function,
    function={name, description, parameters}} dicts. Mirror the vendor
    weather schema shape from §6.4 of the plan."""
    decls = as_function_declarations()
    names = {d["function"]["name"] for d in decls}
    assert names == set(default_registry().keys())
    for d in decls:
        assert d["type"] == "function", d
        fn = d["function"]
        assert isinstance(fn["name"], str), d
        assert fn["name"], d
        assert isinstance(fn["description"], str), d
        assert fn["description"], d
        params = fn["parameters"]
        assert params["type"] == "object", d
        assert "properties" in params, d
        assert "required" in params, d
        assert isinstance(params["required"], list), d
        # `additionalProperties: false` is the strict-mode invariant set by
        # `_StrictBase.model_config`. Drift here means somebody removed
        # `extra=forbid` and is silently dropping unknown LLM-emitted args.
        assert params.get("additionalProperties") is False, d


def test_required_args_in_schema_match_pydantic_models() -> None:
    """Each tool's declared `required` list must match the Pydantic model's
    required fields exactly. This catches a class of drifts where someone
    relaxes a Field default but forgets to regenerate `tools_v1.yaml`."""
    decls = {d["function"]["name"]: d for d in as_function_declarations()}
    expected: dict[str, list[str]] = {
        "get_vitals":                  [],
        "get_medications_at_time":     ["time_24h"],
        "get_medication_by_name":      ["name"],
        "list_allergies":              [],
        "check_food_interaction":      ["food"],
        "get_next_appointment":        [],
        "get_emergency_contact":       [],
    }
    for name, want in expected.items():
        got = decls[name]["function"]["parameters"]["required"]
        assert got == want, f"{name}: required mismatch (got={got}, want={want})"


def test_tools_yaml_matches_registry() -> None:
    """`data/functiongemma/tools_v1.yaml` is a frozen mirror of the registry.

    Regen command on drift is documented at the top of the YAML file.
    Gating it here means a registry edit forces a YAML edit in the same PR
    and the diff surfaces in code review.
    """
    on_disk = yaml.safe_load(_TOOLS_YAML.read_text())
    in_code = as_function_declarations()
    assert on_disk == in_code, (
        "tools_v1.yaml drift — regenerate per the header comment in the YAML"
    )


# --------------------------------------------------------------------------
# Happy-path: every tool returns expected structure on the canonical fixture.
# --------------------------------------------------------------------------


def test_get_vitals_returns_full_set(table: HealthTable) -> None:
    out = execute_tool("get_vitals", {}, table)
    assert isinstance(out, dict)
    for key in (
        "heart_rate_bpm", "blood_pressure_systolic", "blood_pressure_diastolic",
        "spo2_percent", "body_temperature_c", "respiratory_rate", "last_measured",
    ):
        assert key in out, f"missing vital {key!r}"
    assert out["heart_rate_bpm"] == 72
    assert out["last_measured"] == "2026-04-24 08:15"


# | query    | expected_meds                       | desc                                      |
@pytest.mark.parametrize(
    ("query", "expected_meds", "desc"),
    [
        ("08:00",  {"Lisinopril", "Metformin", "Aspirin", "Vitamin D3"}, "morning slot — 4 meds"),
        ("19:00",  {"Metformin"},                                         "evening slot — Metformin only"),
        ("21:00",  {"Atorvastatin"},                                      "night slot — Atorvastatin only"),
        ("12:00",  set(),                                                 "no meds at noon"),
    ],
)
def test_get_medications_at_time(
    query: str, expected_meds: set[str], desc: str, table: HealthTable
) -> None:
    out = execute_tool("get_medications_at_time", {"time_24h": query}, table)
    assert isinstance(out, list), desc
    names = {m["name"] for m in out}
    assert names == expected_meds, desc


# | query           | expected_match | desc                                                  |
@pytest.mark.parametrize(
    ("query", "expected_match", "desc"),
    [
        ("Metformin",  "Metformin",     "exact case match"),
        ("metformin",  "Metformin",     "exact lowercase match"),
        ("METFORMIN",  "Metformin",     "exact uppercase match"),
        ("metf",       "Metformin",     "unique CI prefix"),
        ("Lisin",      "Lisinopril",    "Lisinopril prefix"),
    ],
)
def test_get_medication_by_name_resolves(
    query: str, expected_match: str, desc: str, table: HealthTable
) -> None:
    out = execute_tool("get_medication_by_name", {"name": query}, table)
    assert isinstance(out, dict), desc
    assert out.get("name") == expected_match, desc
    assert "error" not in out, desc


def test_get_medication_by_name_no_match(table: HealthTable) -> None:
    out = execute_tool("get_medication_by_name", {"name": "Ibuprofen"}, table)
    assert out == {"error": "no_match", "name": "Ibuprofen"}


def test_get_medication_by_name_ambiguous(table: HealthTable) -> None:
    """Two meds in the canonical fixture begin with "A" (Atorvastatin, Aspirin)
    — the ambiguous-prefix branch must surface both names."""
    out = execute_tool("get_medication_by_name", {"name": "A"}, table)
    assert out["error"] == "ambiguous"
    assert out["name"] == "A"
    assert set(out["matches"]) == {"Atorvastatin", "Aspirin"}


def test_get_medication_by_name_empty_string(table: HealthTable) -> None:
    """Pydantic accepts an empty string (no constraint), so the handler's own
    guard must surface the invalid_name error rather than degenerating to a
    silent multi-match."""
    out = execute_tool("get_medication_by_name", {"name": "   "}, table)
    assert out == {"error": "invalid_name", "name": "   "}


def test_list_allergies_returns_all(table: HealthTable) -> None:
    out = execute_tool("list_allergies", {}, table)
    assert isinstance(out, list)
    substances = {a["substance"] for a in out}
    assert substances == {"Penicillin", "Shellfish"}
    pen = next(a for a in out if a["substance"] == "Penicillin")
    assert pen["severity"] == "severe"
    assert pen["reaction"] == "anaphylaxis"


# | food          | expected_interacts | expected_meds       | desc                                |
@pytest.mark.parametrize(
    ("food", "expected_interacts", "expected_meds", "desc"),
    [
        ("Grapefruit",        True,  ["Atorvastatin"], "exact match against avoid_foods"),
        ("grapefruit",        True,  ["Atorvastatin"], "lowercase against capitalized fixture"),
        ("Grapefruit juice",  True,  ["Atorvastatin"], "second avoid_foods entry matches"),
        ("Alcohol",           True,  ["Metformin"],    "alcohol matches Metformin's avoid_foods"),
    ],
)
def test_check_food_interaction_positive(
    food: str, expected_interacts: bool, expected_meds: list[str], desc: str,
    table: HealthTable,
) -> None:
    out = execute_tool("check_food_interaction", {"food": food}, table)
    assert out["interacts"] is expected_interacts, desc
    assert out["with_meds"] == expected_meds, desc


def test_check_food_interaction_grapefruit_rule(table: HealthTable) -> None:
    """Per §9.2 of the plan, the dietary-restriction "no grapefruit" rule
    should surface alongside the medication match — both signals are
    load-bearing for the agent's NL answer."""
    out = execute_tool("check_food_interaction", {"food": "grapefruit"}, table)
    assert out["rule"] == "no grapefruit"


# | food          | desc                                              |
@pytest.mark.parametrize(
    ("food", "desc"),
    [
        ("pizza",       "unknown food has no interaction"),
        ("apple",       "apple not in any avoid_foods or rule"),
        ("strawberry",  "strawberry not in any avoid_foods or rule"),
    ],
)
def test_check_food_interaction_negative(
    food: str, desc: str, table: HealthTable
) -> None:
    out = execute_tool("check_food_interaction", {"food": food}, table)
    assert out["interacts"] is False, desc
    assert out["with_meds"] == [], desc
    assert out["rule"] is None, desc


def test_check_food_interaction_substring_avoid_foods_does_not_match(
    table: HealthTable,
) -> None:
    """The medication-side `avoid_foods` match is case-insensitive equality,
    not substring — querying "Grape" must NOT collect Atorvastatin via
    "Grapefruit". (The rule-side IS substring, hence the separate carve-out
    below.)
    """
    out = execute_tool("check_food_interaction", {"food": "Grape"}, table)
    assert out["with_meds"] == []


def test_check_food_interaction_invalid_food(table: HealthTable) -> None:
    """Whitespace-only food string is invalid — the handler's guard must
    surface the invalid_food error rather than matching everything."""
    out = execute_tool("check_food_interaction", {"food": "   "}, table)
    assert out == {"error": "invalid_food", "food": "   "}


def test_get_next_appointment_returns_earliest(table: HealthTable) -> None:
    out = execute_tool("get_next_appointment", {}, table)
    assert out["date"] == "2026-05-06", "earliest of the two fixture appointments"
    assert out["time"] == "10:30"
    assert out["provider"] == "Dr. Evelyn Chen"
    assert out["location"] == "Maple Clinic, Room 204"


def test_get_emergency_contact_returns_first(table: HealthTable) -> None:
    out = execute_tool("get_emergency_contact", {}, table)
    assert out == {
        "name": "Jane Doe",
        "relation": "daughter",
        "phone": "+1-555-0142",
    }


# --------------------------------------------------------------------------
# Empty-section behavior — exercises the "tool returns error dict instead
# of raising" branch on a synthetic minimal HealthTable.
# --------------------------------------------------------------------------


# | tool_name                  | args        | expected_error          | desc                              |
@pytest.mark.parametrize(
    ("tool_name", "args", "expected_error", "desc"),
    [
        ("get_next_appointment",   {},                    "no_appointments", "no appointments → error dict"),
        ("get_emergency_contact",  {},                    "no_contacts",     "no contacts → error dict"),
        ("get_medication_by_name", {"name": "Anything"},  "no_match",        "empty meds → no_match"),
    ],
)
def test_empty_table_returns_stable_error(
    tool_name: str, args: dict[str, str], expected_error: str, desc: str,
    empty_table: HealthTable,
) -> None:
    out = execute_tool(tool_name, args, empty_table)
    assert out.get("error") == expected_error, desc


# | tool_name                  | args                  | expected_value | desc                          |
@pytest.mark.parametrize(
    ("tool_name", "args", "expected_value", "desc"),
    [
        ("get_medications_at_time", {"time_24h": "08:00"}, [],            "empty meds → [] not error"),
        ("list_allergies",          {},                    [],            "empty allergies → [] not error"),
    ],
)
def test_empty_table_returns_empty_list(
    tool_name: str, args: dict[str, str], expected_value: list[object], desc: str,
    empty_table: HealthTable,
) -> None:
    """List-returning tools degrade to an empty list, not an error dict —
    the FG agent should be able to answer "you have no allergies" without
    a special-case branch on its end."""
    out = execute_tool(tool_name, args, empty_table)
    assert out == expected_value, desc


def test_empty_table_food_interaction_no_match(empty_table: HealthTable) -> None:
    out = execute_tool("check_food_interaction", {"food": "grapefruit"}, empty_table)
    assert out["interacts"] is False
    assert out["with_meds"] == []
    assert out["rule"] is None


# --------------------------------------------------------------------------
# Argument validation.
# --------------------------------------------------------------------------


# | bad_time | desc                              |
@pytest.mark.parametrize(
    ("bad_time", "desc"),
    [
        ("8:00",      "missing leading zero on hour"),
        ("08:00:00",  "seconds suffix is not allowed"),
        ("24:00",     "hour 24 is out of range"),
        ("23:60",     "minute 60 is out of range"),
        ("morning",   "non-numeric time string"),
        ("",          "empty time string"),
    ],
)
def test_get_medications_at_time_rejects_invalid_format(
    bad_time: str, desc: str, table: HealthTable,
) -> None:
    out = execute_tool("get_medications_at_time", {"time_24h": bad_time}, table)
    assert out.get("error") == "invalid_arguments", desc
    assert any("time_24h" in m for m in out["messages"]), desc


def test_invalid_arguments_unknown_field(table: HealthTable) -> None:
    """`extra=forbid` on the Pydantic models is the gate against the LLM
    quietly emitting unknown args; assert the validation surface holds."""
    out = execute_tool(
        "get_medication_by_name", {"name": "Metformin", "extra_field": True}, table
    )
    assert out["error"] == "invalid_arguments"
    assert any("extra_field" in m for m in out["messages"])


def test_invalid_arguments_wrong_type(table: HealthTable) -> None:
    out = execute_tool("get_medication_by_name", {"name": 42}, table)
    assert out["error"] == "invalid_arguments"


def test_invalid_arguments_missing_required(table: HealthTable) -> None:
    out = execute_tool("get_medication_by_name", {}, table)
    assert out["error"] == "invalid_arguments"
    assert any("name" in m for m in out["messages"])


# --------------------------------------------------------------------------
# Dispatch surface.
# --------------------------------------------------------------------------


def test_unknown_tool_raises(table: HealthTable) -> None:
    """Unknown tool name = developer/registration error → KeyError. Distinct
    from `invalid_arguments` (a model-side error returned as a dict)."""
    with pytest.raises(KeyError, match="schedule_appointment"):
        execute_tool("schedule_appointment", {}, table)


def test_argument_models_are_strict_base() -> None:
    """All argument models must inherit from `_StrictBase` (extra=forbid).
    Verified indirectly: instantiating with an unknown key must raise."""
    for model_cls in (TimeArgs, NameArgs, FoodArgs, EmptyArgs):
        with pytest.raises(ValueError, match="unknown_key"):
            model_cls.model_validate({"unknown_key": "x"})


# --------------------------------------------------------------------------
# JSON-serializability — the orchestrator round-trips tool returns through
# `json.dumps` to render the `<start_function_response>` turn.
# --------------------------------------------------------------------------


# | tool_name                  | args                       | desc                        |
@pytest.mark.parametrize(
    ("tool_name", "args", "desc"),
    [
        ("get_vitals",                {},                              "vitals → JSON"),
        ("get_medications_at_time",   {"time_24h": "08:00"},           "morning meds → JSON"),
        ("get_medication_by_name",    {"name": "Metformin"},           "named med → JSON"),
        ("list_allergies",            {},                              "allergies → JSON"),
        ("check_food_interaction",    {"food": "Grapefruit"},          "interaction → JSON"),
        ("get_next_appointment",      {},                              "appointment → JSON"),
        ("get_emergency_contact",     {},                              "contact → JSON"),
    ],
)
def test_outputs_are_json_serializable(
    tool_name: str, args: dict[str, str], desc: str, table: HealthTable,
) -> None:
    out = execute_tool(tool_name, args, table)
    rendered = json.dumps(out)
    assert isinstance(rendered, str), desc
    # Round-trip must preserve structure.
    assert json.loads(rendered) == out, desc
