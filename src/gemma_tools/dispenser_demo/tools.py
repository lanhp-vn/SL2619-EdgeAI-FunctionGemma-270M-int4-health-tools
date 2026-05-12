"""Dispenser-demo tool registry.

Five tools, one of them side-effecting (`dispense_medication`), one of them
the structural refusal tool (`refuse_out_of_scope`). Mirrors the
`gemma_tools.functiongemma.tools` pattern: `ToolSpec` dataclass, allowlisted
dispatch through a tuple (no `globals()` lookup), Pydantic arg validation.

Tool response shape contract (the §1 / §11 R5 invariant from plan):

- Every digit-bearing key in a tool response has a digit-free `*_words`
  companion. Derivation lives in `wordform.py`.
- The model is trained to QUOTE `*_words` verbatim; the runtime (Phase 3
  on-board TTS) reads `*_words` not the raw field.

Phase 1.2 status: `dispense_medication` is a stub that returns
`{"status": "dispensed"}`. Phase 2 will replace this with the real
`pybleno` BLE-aware implementation per plan §6.1 — at that point the
dispatch signature may grow a context arg.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gemma_tools.dispenser_demo import wordform
from gemma_tools.dispenser_demo.health_table_v2 import HealthTableV2

# --------------------------------------------------------------------------
# Argument models. `extra=forbid` rejects unknown keys; the FG tools.py
# rationale applies verbatim here.
# --------------------------------------------------------------------------


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(_StrictBase):
    pass


class RefusalArgs(_StrictBase):
    reason: Literal["health_advice", "off_topic"] = Field(
        description=(
            "Why the request is out of scope. Use 'health_advice' for "
            "medical-advice / symptom-diagnosis / treatment-plan questions. "
            "Use 'off_topic' for anything outside the health domain (weather, "
            "news, jokes, math, generic personal questions)."
        ),
    )


# --------------------------------------------------------------------------
# Tool implementations. All read-only against the `HealthTableV2` instance
# except `dispense_medication` (currently a stub — Phase 2 wires real BLE).
# Kept module-private; external callers go through `execute_tool`.
# --------------------------------------------------------------------------


def _get_patient_profile(_args: EmptyArgs, table: HealthTableV2) -> dict[str, Any]:
    p = table.patient
    return {
        "name": p.name,
        "age": p.age,
        "age_words": wordform.age_to_words(p.age),
        "sex": p.sex,
        "diagnoses": list(p.diagnoses),
        "diagnoses_words": wordform.diagnoses_to_words(p.diagnoses),
    }


def _get_next_appointment(_args: EmptyArgs, table: HealthTableV2) -> dict[str, Any]:
    # Chronological min on (date, time); both are zero-padded ISO strings so
    # lexicographic min == chronological min. v2.yaml ships with a single
    # appointment but the dispatcher does not assume that.
    nxt = min(table.appointments, key=lambda a: (a.date, a.time))
    return {
        "date": nxt.date,
        "date_words": wordform.date_to_words(nxt.date),
        "time": nxt.time,
        "time_words": wordform.time_to_words(nxt.time),
        "provider": nxt.provider,
        "purpose": nxt.purpose,
        "location": nxt.location,
        "location_words": wordform.freeform_to_words(nxt.location),
    }


def _get_emergency_contact(_args: EmptyArgs, table: HealthTableV2) -> dict[str, Any]:
    c = table.emergency_contacts[0]
    return {
        "name": c.name,
        "relation": c.relation,
        "phone": c.phone,
        "phone_words": wordform.phone_to_words(c.phone),
    }


def _dispense_medication(_args: EmptyArgs, _table: HealthTableV2) -> dict[str, Any]:
    # Phase 1.2 stub — returns the happy-path response that matches 7 of the
    # 8 `dispense` seed rows. Phase 1.6 holdout eval consequence: the one
    # seed encoding the failure path (`di-008` →
    # `{"status": "ble_not_connected"}`) will mismatch the live tool's
    # output until Phase 2 wires real BLE. Documented as a deferred-eval row
    # in `docs/plans/dispenser-demo/plan.md` §9.1.6 / §6.1.
    #
    # The Phase 2 implementation will:
    #   1. Check that a `BleClient` is subscribed (otherwise return
    #      `{"status": "ble_not_connected"}`).
    #   2. Send the `5A A5 01 00` notify (plan §6.2 wire contract).
    #   3. Return `{"status": "dispensed"}` on successful notify.
    return {"status": "dispensed"}


def _refuse_out_of_scope(_args: RefusalArgs, _table: HealthTableV2) -> dict[str, Any]:
    # Response shape matches the seed JSONL exactly (`{"status": "refused"}`).
    # The `reason` enum already lives in the preceding `tool_call.arguments`
    # in the conversation history; echoing it in the response would be
    # redundant AND would diverge from the seed-encoded contract, which the
    # model is being trained on. The runtime dispatcher prints the same
    # canned refusal NL regardless of reason.
    return {"status": "refused"}


# --------------------------------------------------------------------------
# Registry — the single allowlist. `execute_tool` dispatches through this;
# nothing else maps a string to a callable.
# --------------------------------------------------------------------------


# Same rationale as FG: the first arg is a Pydantic-validated model whose
# concrete type varies per tool; encoding it in the type would require
# generics with no real safety win because dispatch pairs (model, handler).
ToolHandler = Callable[[Any, HealthTableV2], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_patient_profile",
        description=(
            "Return the patient's profile (name, age, sex, diagnoses) with "
            "digit-free `*_words` companions for every digit-bearing field."
        ),
        args_model=EmptyArgs,
        handler=_get_patient_profile,
    ),
    ToolSpec(
        name="get_next_appointment",
        description=(
            "Return the earliest upcoming appointment (date, time, provider, "
            "purpose, location) with digit-free `*_words` companions for every "
            "digit-bearing field."
        ),
        args_model=EmptyArgs,
        handler=_get_next_appointment,
    ),
    ToolSpec(
        name="get_emergency_contact",
        description=(
            "Return the first listed emergency contact (name, relation, phone) "
            "with a digit-free `phone_words` companion for the phone number."
        ),
        args_model=EmptyArgs,
        handler=_get_emergency_contact,
    ),
    ToolSpec(
        name="dispense_medication",
        description=(
            "Dispense the patient's medication by sending a BLE notification "
            "to the dispenser. Returns the post-action status: 'dispensed' on "
            "success, 'ble_not_connected' if no BLE peer is subscribed."
        ),
        args_model=EmptyArgs,
        handler=_dispense_medication,
    ),
    ToolSpec(
        name="refuse_out_of_scope",
        description=(
            "Call when the user request cannot be answered with the four "
            "domain tools above. Returns an acknowledgement; the runtime "
            "responds with a canned refusal sentence. `reason` is a "
            "two-value enum: 'health_advice' for medical advice / symptom "
            "diagnosis / treatment-plan requests, 'off_topic' for anything "
            "outside the health domain."
        ),
        args_model=RefusalArgs,
        handler=_refuse_out_of_scope,
    ),
)


def default_registry() -> dict[str, ToolSpec]:
    """Fresh `{name: ToolSpec}` mapping in declaration order."""
    return {spec.name: spec for spec in _TOOL_SPECS}


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    table: HealthTableV2,
) -> Any:
    """Validate `arguments` against the tool's Pydantic model and dispatch.

    Raises:
        KeyError: when `name` is not in the registry — a developer error
            (the model emitted an unregistered tool name); crashes the
            orchestration loop rather than silently 404-ing.

    Returns:
        Whatever the tool handler returns; on `ValidationError`,
        ``{"error": "invalid_arguments", "tool": <name>, "messages": [...]}``
        so the orchestration loop can route the failure back to the model.
    """
    registry = default_registry()
    if name not in registry:
        raise KeyError(f"Unknown tool: {name!r}")
    spec = registry[name]
    try:
        validated = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        return {
            "error": "invalid_arguments",
            "tool": name,
            "messages": [
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()
            ],
        }
    return spec.handler(validated, table)


def _normalize_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic JSON schema in the shape FG's chat template expects.

    Same logic as `gemma_tools.functiongemma.tools._normalize_schema`: strip
    Pydantic-emitted `title` keys so the diff against the seed JSONL `tools[]`
    block stays clean.
    """
    schema: dict[str, Any] = model_cls.model_json_schema()
    schema.pop("title", None)
    schema.pop("description", None)
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    for prop in schema["properties"].values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def as_function_declarations() -> list[dict[str, Any]]:
    """Return the registry as HF chat-template tool declarations.

    Output shape mirrors `as_function_declarations` in
    `gemma_tools.functiongemma.tools` — `[{"type": "function", "function":
    {"name": ..., "description": ..., "parameters": ...}}, ...]`.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": _normalize_schema(spec.args_model),
            },
        }
        for spec in _TOOL_SPECS
    ]
