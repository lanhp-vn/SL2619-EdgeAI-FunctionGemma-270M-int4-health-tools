"""Read-only FunctionGemma tool registry over `data/health_table_v1.yaml`.

Surface (M3 — `docs/plans/FunctionGemma/README.md` §9.2):

- `default_registry()` — `dict[str, ToolSpec]` of allowlisted tools.
- `execute_tool(name, arguments, table)` — validate `arguments` with Pydantic,
  dispatch through the allowlist (no `globals()` — per the vendor security
  warning in `cookbook/docs/functiongemma/full-function-calling-sequence-with-functiongemma.ipynb`
  cell 20 reproduced at §6.5 of the plan), return JSON-serializable output.
- `as_function_declarations()` — HF `apply_chat_template(tools=...)` shape.

All tools are read-only. Adding a write/scheduling tool is OQ-4 territory and
requires its own safety story (audit log + consent prompts); this module
intentionally does NOT expose any.

Return contract: every tool returns plain Python (`dict`, `list`, `str`,
`int`, `float`, `bool`, `None`). No `HealthTable` dataclass instance escapes
this module — they are not directly JSON-serializable and the orchestration
loop needs to round-trip the result through `json.dumps` for the
`<start_function_response>` turn.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gemma_tools.health_table import HealthTable, Medication

# Strict 24-hour HH:MM regex. Hours 00..23, minutes 00..59. Tighter than a
# generic `\d\d:\d\d` because medication-time fixtures are hand-written and a
# typo like "08:60" should fail loudly at the tool boundary, not silently
# match nothing inside `_get_medications_at_time`.
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


# --------------------------------------------------------------------------
# Argument models. `extra=forbid` rejects unknown keys; this is the FG-tool
# equivalent of mypy strict — drift in the LLM-emitted JSON args surfaces
# here instead of getting silently dropped.
# --------------------------------------------------------------------------


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArgs(_StrictBase):
    pass


class TimeArgs(_StrictBase):
    # Schema stays a plain string so the chat-template render the model
    # sees matches the training shape (FG was trained on
    # `{"type": "string", "required": ["time_24h"]}`); the only relaxation
    # is `required: []` + empty-string default so `get_medications_at_time({})`
    # is a list-all request, not an error. The model already emits `({})`
    # for broad questions like "when should I take my pills?" — see
    # 2026-05-02 REPL session, fixed alongside `_format_meds_at_time`.
    time_24h: str = Field(
        default="",
        description=(
            "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'. "
            "Omit (or pass an empty string) to list every medication with "
            "its full schedule."
        ),
    )

    @field_validator("time_24h")
    @classmethod
    def _validate_hhmm(cls, value: str) -> str:
        if value == "":
            return ""
        if not _HHMM_RE.match(value):
            raise ValueError(
                f"time_24h must match HH:MM in [00:00..23:59], got {value!r}",
            )
        return value


class NameArgs(_StrictBase):
    name: str = Field(
        description=(
            "Medication name. Lookup is case-insensitive: exact match wins; "
            "otherwise a unique prefix match wins; ambiguous prefixes return "
            "an error dict."
        ),
    )


class FoodArgs(_StrictBase):
    food: str = Field(
        description=(
            "Food name to check for medication or dietary interactions, "
            "e.g. 'grapefruit'. Case-insensitive."
        ),
    )


# --------------------------------------------------------------------------
# Tool implementations. Signatures: `(args: <Model>, table: HealthTable) -> <JSON>`.
# Kept module-private (`_` prefix) — external callers go through `execute_tool`.
# --------------------------------------------------------------------------


def _med_to_dict(m: Medication) -> dict[str, Any]:
    return {
        "name": m.name,
        "dose": m.dose,
        "schedule": m.schedule,
        "with_food": m.with_food,
        "purpose": m.purpose,
        "avoid_foods": list(m.avoid_foods),
        "avoid_drugs": list(m.avoid_drugs),
    }


def _split_schedule(schedule: str) -> list[str]:
    # Schedule strings are comma-joined HH:MM times per the YAML convention
    # (e.g. "08:00" or "08:00, 19:00"). Strip whitespace per part to tolerate
    # the optional space after the comma.
    return [tok.strip() for tok in schedule.split(",") if tok.strip()]


def _get_vitals(_args: EmptyArgs, table: HealthTable) -> dict[str, Any]:
    v = table.vitals
    return {
        "heart_rate_bpm": v.heart_rate_bpm,
        "blood_pressure_systolic": v.blood_pressure_systolic,
        "blood_pressure_diastolic": v.blood_pressure_diastolic,
        "spo2_percent": v.spo2_percent,
        "body_temperature_c": v.body_temperature_c,
        "respiratory_rate": v.respiratory_rate,
        "last_measured": v.last_measured,
    }


def _get_medications_at_time(args: TimeArgs, table: HealthTable) -> list[dict[str, Any]]:
    target = args.time_24h
    meds = (
        table.medications
        if target == ""
        else [m for m in table.medications if target in _split_schedule(m.schedule)]
    )
    return [
        {
            "name": m.name,
            "dose": m.dose,
            "schedule": m.schedule,
            "with_food": m.with_food,
            "purpose": m.purpose,
        }
        for m in meds
    ]


def _get_medication_by_name(args: NameArgs, table: HealthTable) -> dict[str, Any]:
    query = args.name.strip().lower()
    if not query:
        return {"error": "invalid_name", "name": args.name}

    exact = [m for m in table.medications if m.name.lower() == query]
    if exact:
        return _med_to_dict(exact[0])

    prefix = [m for m in table.medications if m.name.lower().startswith(query)]
    if not prefix:
        return {"error": "no_match", "name": args.name}
    if len(prefix) > 1:
        return {
            "error": "ambiguous",
            "name": args.name,
            "matches": sorted(m.name for m in prefix),
        }
    return _med_to_dict(prefix[0])


def _list_allergies(_args: EmptyArgs, table: HealthTable) -> list[dict[str, Any]]:
    return [
        {"substance": a.substance, "severity": a.severity, "reaction": a.reaction}
        for a in table.allergies
    ]


def _check_food_interaction(args: FoodArgs, table: HealthTable) -> dict[str, Any]:
    food_raw = args.food.strip()
    food = food_raw.lower()
    if not food:
        return {"error": "invalid_food", "food": args.food}

    # Medication-side: case-insensitive equality against each `avoid_foods`
    # entry. Equality (not substring) avoids "grape" matching "Grapefruit"
    # — the fixture authors enumerate the foods explicitly.
    with_meds = [
        m.name
        for m in table.medications
        if any(af.lower() == food for af in m.avoid_foods)
    ]

    # Dietary-restriction-side: substring match against the rule text. Rules
    # are short free-form strings ("no grapefruit"); substring matching is
    # the closest semantic to the original prose.
    rules = [r.rule for r in table.dietary_restrictions if food in r.rule.lower()]

    return {
        "food": food_raw,
        "interacts": bool(with_meds) or bool(rules),
        "with_meds": with_meds,
        "rule": rules[0] if rules else None,
    }


def _get_next_appointment(_args: EmptyArgs, table: HealthTable) -> dict[str, Any]:
    if not table.appointments:
        return {"error": "no_appointments"}
    # Both fields are zero-padded ISO strings; lexicographic min == chronological min.
    nxt = min(table.appointments, key=lambda a: (a.date, a.time))
    return {
        "date": nxt.date,
        "time": nxt.time,
        "provider": nxt.provider,
        "purpose": nxt.purpose,
        "location": nxt.location,
    }


def _get_emergency_contact(_args: EmptyArgs, table: HealthTable) -> dict[str, Any]:
    if not table.emergency_contacts:
        return {"error": "no_contacts"}
    c = table.emergency_contacts[0]
    return {"name": c.name, "relation": c.relation, "phone": c.phone}


# --------------------------------------------------------------------------
# Registry — the single allowlist. `execute_tool` dispatches through this;
# nothing else maps a string to a callable. This is the dispatch table the
# vendor warning in §6.5 of the plan calls out as the safe alternative to
# `globals()`.
# --------------------------------------------------------------------------


# `Callable[[Any, HealthTable], Any]` here is intentional: the first arg is
# the validated Pydantic model instance whose concrete type varies per tool.
# Encoding it in the type would require a generic over each ToolSpec instance
# and offers no extra safety because the dispatch already enforces the
# (model, handler) pairing.
ToolHandler = Callable[[Any, HealthTable], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="get_vitals",
        description=(
            "Return the patient's most recent vital-sign measurements "
            "(heart rate, blood pressure, SpO2, body temperature, respiratory "
            "rate) along with the timestamp they were taken."
        ),
        args_model=EmptyArgs,
        handler=_get_vitals,
    ),
    ToolSpec(
        name="get_medications_at_time",
        description=(
            "List medications scheduled to be taken at a specific 24-hour "
            "clock time. Match is exact against the normalized HH:MM schedule."
        ),
        args_model=TimeArgs,
        handler=_get_medications_at_time,
    ),
    ToolSpec(
        name="get_medication_by_name",
        description=(
            "Look up a medication by name. Match is case-insensitive: exact "
            "match wins, otherwise a unique prefix match. An ambiguous prefix "
            "returns an error dict so the caller can re-prompt."
        ),
        args_model=NameArgs,
        handler=_get_medication_by_name,
    ),
    ToolSpec(
        name="list_allergies",
        description=(
            "List all known allergies for the patient with their severity "
            "and reaction."
        ),
        args_model=EmptyArgs,
        handler=_list_allergies,
    ),
    ToolSpec(
        name="check_food_interaction",
        description=(
            "Check whether a given food interacts with any of the patient's "
            "medications or dietary restrictions. Returns an `interacts` bool, "
            "the list of medication names that flag the food, and the matching "
            "dietary-restriction rule if any."
        ),
        args_model=FoodArgs,
        handler=_check_food_interaction,
    ),
    ToolSpec(
        name="get_next_appointment",
        description=(
            "Return the earliest upcoming appointment by date and time, "
            "with provider, purpose, and location."
        ),
        args_model=EmptyArgs,
        handler=_get_next_appointment,
    ),
    ToolSpec(
        name="get_emergency_contact",
        description=(
            "Return the first listed emergency contact (name, relation, phone)."
        ),
        args_model=EmptyArgs,
        handler=_get_emergency_contact,
    ),
)


def default_registry() -> dict[str, ToolSpec]:
    """Return a fresh `{name: ToolSpec}` mapping in declaration order."""
    return {spec.name: spec for spec in _TOOL_SPECS}


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    table: HealthTable,
) -> Any:
    """Validate `arguments` with the tool's Pydantic model and dispatch.

    Raises:
        KeyError: when `name` is not in the registry — this is a developer
            error (the model emitted an unregistered tool name) and should
            crash the orchestration loop rather than silently 404.

    Returns:
        Whatever the tool handler returns; on a Pydantic ValidationError, an
        ``{"error": "invalid_arguments", "tool": <name>, "messages": [...]}``
        dict is returned so the orchestration loop can route the failure
        back to the model rather than crashing.
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

    Pydantic emits a `title` on every model + property; HF's chat template
    happily renders them but the vendor weather example (§6.4) doesn't carry
    them, so we strip for visual parity. `required` is forced to be present
    (empty list for the no-arg case) so consumers don't have to special-case.
    """
    schema: dict[str, Any] = model_cls.model_json_schema()
    # Strip Pydantic-emitted metadata that the vendor weather example (§6.4)
    # does not carry: top-level `title` + `description` (the per-tool
    # description is already on the outer `function` dict) and per-property
    # `title`. Keeping them is harmless but it pollutes the diff against the
    # frozen `data/functiongemma/tools_v1.yaml`.
    schema.pop("title", None)
    schema.pop("description", None)
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    for prop in schema["properties"].values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def as_function_declarations() -> list[dict[str, Any]]:
    """Return the registry as a list of HF chat-template tool declarations.

    Output shape mirrors the vendor weather example at
    `docs/plans/FunctionGemma/README.md` §6.4 verbatim:

        [{"type": "function",
          "function": {"name": ..., "description": ..., "parameters": ...}}, ...]
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
