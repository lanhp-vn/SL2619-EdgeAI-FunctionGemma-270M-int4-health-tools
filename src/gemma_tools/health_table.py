"""Load and validate the health-record fixture.

Schema source of truth: data/health_table_v1.yaml (single canonical
fixture; no v2 spawned). The fixture is the closed-world knowledge base
for the Gemma 3 270M YAML-QA bench per
docs/conventions/slm-system-prompt.md §4 and
docs/plans/models-testing-plan.md §5.

Schema shape (all blocks under `conditions`, `allergies`, `medications`,
`dietary_restrictions`, `appointments`, `emergency_contacts` are
OPTIONAL — missing → empty tuple, same API as Phase A's minimal fixture):

    patient:               required (name, age; sex + blood_type optional)
    vitals:                required (6 closed-interval-validated fields,
                                     last_measured optional)
    notes:                 required (list of strings; empty list is fine)
    conditions:            optional (list of {name, diagnosed_at, severity,
                                               controlled})
    allergies:             optional (list of {substance, severity, reaction})
    medications:           optional (list of {name, dose, schedule, with_food,
                                              purpose, avoid_foods, avoid_drugs})
    dietary_restrictions:  optional (list of {rule, reason})
    appointments:          optional (list of {date, time, provider, purpose,
                                              location})
    emergency_contacts:    optional (list of {name, relation, phone})

Convention deviation (re-stated from Phase A): stdlib + PyYAML only — no
Pydantic — even though docs/conventions/09-code-style-python.md §6 mandates
Pydantic for runtime config. Rationale same as Phase A: this is a test-bench
fixture, and the on-board ephemeral venv has constrained wheel availability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import yaml

_T = TypeVar("_T")

# Inclusive [lo, hi] ranges per plan §5.2 vital-set + medical-sanity envelopes.
# Tuples are (lo, hi); the loader rejects values outside the closed interval.
_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate_bpm": (30, 200),
    "blood_pressure_systolic": (60, 200),
    "blood_pressure_diastolic": (40, 130),
    "spo2_percent": (70, 100),
    "body_temperature_c": (34.0, 42.0),
    "respiratory_rate": (8, 40),
}

# Top-level keys required in the YAML document. Missing any of these is a
# schema error, not a recoverable default.
_REQUIRED_TOP_KEYS: tuple[str, ...] = ("patient", "vitals", "notes")

# Severity labels accepted verbatim in conditions + allergies. The model quotes
# these back at the user (R-2 "quote verbatim"), so keep the set small and
# human-readable.
_ALLOWED_SEVERITIES: frozenset[str] = frozenset({"mild", "moderate", "severe"})


@dataclass(frozen=True, slots=True)
class Patient:
    name: str
    age: int
    sex: str | None = None
    blood_type: str | None = None


@dataclass(frozen=True, slots=True)
class Vitals:
    heart_rate_bpm: int
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    spo2_percent: int
    body_temperature_c: float
    respiratory_rate: int
    # ISO-8601 "YYYY-MM-DD HH:MM" per the fixture convention. Optional so the
    # Phase A minimal fixture (no timestamp) still loads.
    last_measured: str | None = None


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    diagnosed_at: str  # ISO-8601 date
    severity: str  # one of _ALLOWED_SEVERITIES
    controlled: bool


@dataclass(frozen=True, slots=True)
class Allergy:
    substance: str
    severity: str  # one of _ALLOWED_SEVERITIES
    reaction: str


@dataclass(frozen=True, slots=True)
class Medication:
    name: str
    dose: str  # free-form per YAML fixture convention (e.g. "10 mg", "1000 IU")
    schedule: str  # e.g. "08:00" or "08:00, 19:00" — comma-joined HH:MM times
    with_food: bool
    purpose: str
    # Empty tuples by default — a blank-interactions medication is common.
    avoid_foods: tuple[str, ...] = ()
    avoid_drugs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DietaryRestriction:
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class Appointment:
    date: str  # ISO-8601 date
    time: str  # HH:MM
    provider: str
    purpose: str
    location: str | None = None


@dataclass(frozen=True, slots=True)
class EmergencyContact:
    name: str
    relation: str
    phone: str


@dataclass(frozen=True, slots=True)
class HealthTable:
    patient: Patient
    vitals: Vitals
    notes: tuple[str, ...]
    # Optional blocks default to empty tuple so a minimal fixture (patient +
    # vitals + notes only — Phase A's shape) still constructs cleanly.
    conditions: tuple[Condition, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    medications: tuple[Medication, ...] = ()
    dietary_restrictions: tuple[DietaryRestriction, ...] = ()
    appointments: tuple[Appointment, ...] = ()
    emergency_contacts: tuple[EmergencyContact, ...] = ()


# --------------------------------------------------------------------------
# Internal type-guard helpers (kept private; no external consumers).
# --------------------------------------------------------------------------

def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a mapping, got {type(value).__name__}")
    out: dict[str, object] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(f"{context}: non-string key {k!r}")
        out[k] = v
    return out


def _require_int(value: object, context: str) -> int:
    # bool is a subclass of int; reject explicitly so True/False can't pass as
    # an age or vital reading.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: expected int, got {type(value).__name__}")
    return value


def _require_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected number, got {type(value).__name__}")
    return float(value)


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context}: expected str, got {type(value).__name__}")
    return value


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context}: expected bool, got {type(value).__name__}")
    return value


def _optional_str(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, context)


def _string_list(value: object, context: str) -> tuple[str, ...]:
    # Missing / None → empty. YAML authors often omit optional lists entirely.
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list, got {type(value).__name__}")
    return tuple(_require_str(item, f"{context}[{i}]") for i, item in enumerate(value))


def _check_range(name: str, value: float) -> None:
    lo, hi = _VITAL_RANGES[name]
    if value < lo or value > hi:
        raise ValueError(f"vital {name!r} = {value} outside [{lo}..{hi}]")


def _check_severity(value: str, context: str) -> str:
    if value not in _ALLOWED_SEVERITIES:
        raise ValueError(
            f"{context}: severity {value!r} not in {sorted(_ALLOWED_SEVERITIES)}"
        )
    return value


# --------------------------------------------------------------------------
# Per-block parsers. Each takes a raw dict + context string, returns a
# frozen dataclass, and raises ValueError with a self-locating message on
# any schema violation.
# --------------------------------------------------------------------------

def _parse_patient(doc: dict[str, object], ctx: str) -> Patient:
    if "name" not in doc:
        raise ValueError(f"{ctx}: missing 'name'")
    if "age" not in doc:
        raise ValueError(f"{ctx}: missing 'age'")
    return Patient(
        name=_require_str(doc["name"], f"{ctx}.name"),
        age=_require_int(doc["age"], f"{ctx}.age"),
        sex=_optional_str(doc.get("sex"), f"{ctx}.sex"),
        blood_type=_optional_str(doc.get("blood_type"), f"{ctx}.blood_type"),
    )


def _parse_vitals(doc: dict[str, object], ctx: str) -> Vitals:
    for key in _VITAL_RANGES:
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")

    hr = _require_int(doc["heart_rate_bpm"], f"{ctx}.heart_rate_bpm")
    bp_s = _require_int(doc["blood_pressure_systolic"], f"{ctx}.blood_pressure_systolic")
    bp_d = _require_int(doc["blood_pressure_diastolic"], f"{ctx}.blood_pressure_diastolic")
    spo2 = _require_int(doc["spo2_percent"], f"{ctx}.spo2_percent")
    temp = _require_number(doc["body_temperature_c"], f"{ctx}.body_temperature_c")
    rr = _require_int(doc["respiratory_rate"], f"{ctx}.respiratory_rate")

    _check_range("heart_rate_bpm", hr)
    _check_range("blood_pressure_systolic", bp_s)
    _check_range("blood_pressure_diastolic", bp_d)
    _check_range("spo2_percent", spo2)
    _check_range("body_temperature_c", temp)
    _check_range("respiratory_rate", rr)

    return Vitals(
        heart_rate_bpm=hr,
        blood_pressure_systolic=bp_s,
        blood_pressure_diastolic=bp_d,
        spo2_percent=spo2,
        body_temperature_c=temp,
        respiratory_rate=rr,
        last_measured=_optional_str(doc.get("last_measured"), f"{ctx}.last_measured"),
    )


def _parse_condition(doc: dict[str, object], ctx: str) -> Condition:
    for key in ("name", "diagnosed_at", "severity", "controlled"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return Condition(
        name=_require_str(doc["name"], f"{ctx}.name"),
        diagnosed_at=_require_str(doc["diagnosed_at"], f"{ctx}.diagnosed_at"),
        severity=_check_severity(
            _require_str(doc["severity"], f"{ctx}.severity"), f"{ctx}.severity"
        ),
        controlled=_require_bool(doc["controlled"], f"{ctx}.controlled"),
    )


def _parse_allergy(doc: dict[str, object], ctx: str) -> Allergy:
    for key in ("substance", "severity", "reaction"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return Allergy(
        substance=_require_str(doc["substance"], f"{ctx}.substance"),
        severity=_check_severity(
            _require_str(doc["severity"], f"{ctx}.severity"), f"{ctx}.severity"
        ),
        reaction=_require_str(doc["reaction"], f"{ctx}.reaction"),
    )


def _parse_medication(doc: dict[str, object], ctx: str) -> Medication:
    for key in ("name", "dose", "schedule", "with_food", "purpose"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return Medication(
        name=_require_str(doc["name"], f"{ctx}.name"),
        dose=_require_str(doc["dose"], f"{ctx}.dose"),
        schedule=_require_str(doc["schedule"], f"{ctx}.schedule"),
        with_food=_require_bool(doc["with_food"], f"{ctx}.with_food"),
        purpose=_require_str(doc["purpose"], f"{ctx}.purpose"),
        avoid_foods=_string_list(doc.get("avoid_foods"), f"{ctx}.avoid_foods"),
        avoid_drugs=_string_list(doc.get("avoid_drugs"), f"{ctx}.avoid_drugs"),
    )


def _parse_dietary_restriction(doc: dict[str, object], ctx: str) -> DietaryRestriction:
    for key in ("rule", "reason"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return DietaryRestriction(
        rule=_require_str(doc["rule"], f"{ctx}.rule"),
        reason=_require_str(doc["reason"], f"{ctx}.reason"),
    )


def _parse_appointment(doc: dict[str, object], ctx: str) -> Appointment:
    for key in ("date", "time", "provider", "purpose"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return Appointment(
        date=_require_str(doc["date"], f"{ctx}.date"),
        time=_require_str(doc["time"], f"{ctx}.time"),
        provider=_require_str(doc["provider"], f"{ctx}.provider"),
        purpose=_require_str(doc["purpose"], f"{ctx}.purpose"),
        location=_optional_str(doc.get("location"), f"{ctx}.location"),
    )


def _parse_emergency_contact(doc: dict[str, object], ctx: str) -> EmergencyContact:
    for key in ("name", "relation", "phone"):
        if key not in doc:
            raise ValueError(f"{ctx}: missing {key!r}")
    return EmergencyContact(
        name=_require_str(doc["name"], f"{ctx}.name"),
        relation=_require_str(doc["relation"], f"{ctx}.relation"),
        phone=_require_str(doc["phone"], f"{ctx}.phone"),
    )


def _parse_list(
    raw: object,
    ctx: str,
    parser: Callable[[dict[str, object], str], _T],
) -> tuple[_T, ...]:
    """Parse an optional list-of-mappings block. Missing → empty tuple."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{ctx}: expected list, got {type(raw).__name__}")
    out: list[_T] = []
    for i, item in enumerate(raw):
        item_ctx = f"{ctx}[{i}]"
        mapping = _require_mapping(item, item_ctx)
        out.append(parser(mapping, item_ctx))
    return tuple(out)


# --------------------------------------------------------------------------
# Public entrypoint.
# --------------------------------------------------------------------------

def load_health_table(path: Path) -> HealthTable:
    """Read YAML, validate every block, return a frozen HealthTable.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: on any schema violation; message names the offending key.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    raw = yaml.safe_load(path.read_text()) or {}
    doc = _require_mapping(raw, f"{path}")

    for key in _REQUIRED_TOP_KEYS:
        if key not in doc:
            raise ValueError(f"{path}: missing top-level key {key!r}")

    patient = _parse_patient(
        _require_mapping(doc["patient"], f"{path}: patient"), f"{path}: patient"
    )
    vitals = _parse_vitals(
        _require_mapping(doc["vitals"], f"{path}: vitals"), f"{path}: vitals"
    )

    notes_doc = doc["notes"]
    if not isinstance(notes_doc, list):
        raise ValueError(
            f"{path}: 'notes' must be a list, got {type(notes_doc).__name__}"
        )
    notes = tuple(
        _require_str(n, f"{path}: notes[{i}]") for i, n in enumerate(notes_doc)
    )

    conditions = _parse_list(doc.get("conditions"), f"{path}: conditions", _parse_condition)
    allergies = _parse_list(doc.get("allergies"), f"{path}: allergies", _parse_allergy)
    medications = _parse_list(
        doc.get("medications"), f"{path}: medications", _parse_medication
    )
    dietary_restrictions = _parse_list(
        doc.get("dietary_restrictions"),
        f"{path}: dietary_restrictions",
        _parse_dietary_restriction,
    )
    appointments = _parse_list(
        doc.get("appointments"), f"{path}: appointments", _parse_appointment
    )
    emergency_contacts = _parse_list(
        doc.get("emergency_contacts"),
        f"{path}: emergency_contacts",
        _parse_emergency_contact,
    )

    return HealthTable(
        patient=patient,
        vitals=vitals,
        notes=notes,
        conditions=conditions,
        allergies=allergies,
        medications=medications,
        dietary_restrictions=dietary_restrictions,
        appointments=appointments,
        emergency_contacts=emergency_contacts,
    )
