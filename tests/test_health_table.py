"""Tests for gemma_tools.health_table.

Schema source of truth: docs/plans/models-testing-plan.md §5.2.
Range envelopes: same plan, plus medical sanity per Phase A4 host
instructions Step 6.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from gemma_tools.health_table import (
    Allergy,
    Appointment,
    Condition,
    DietaryRestriction,
    EmergencyContact,
    HealthTable,
    Medication,
    Patient,
    Vitals,
    load_health_table,
)

_REPO = Path(__file__).resolve().parents[1]
CANONICAL_FIXTURE = _REPO / "data" / "health_table_v1.yaml"

# In-memory copy of the canonical fixture. Deep-copied per case so mutations
# in test_rejects_out_of_range_vital don't leak across parametrize rows.
_CANONICAL_DOC: dict[str, object] = {
    "patient": {"name": "Test Patient", "age": 45},
    "vitals": {
        "heart_rate_bpm": 72,
        "blood_pressure_systolic": 118,
        "blood_pressure_diastolic": 76,
        "spo2_percent": 98,
        "body_temperature_c": 36.7,
        "respiratory_rate": 16,
    },
    "notes": [
        "Vitals within nominal range",
        "No medication interactions flagged",
    ],
}


def _write_doc(path: Path, doc: object) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


# | source       | desc                                                       |
@pytest.mark.parametrize(
    ("source", "desc"),
    [
        ("on_disk_canonical", "loads the committed tools/data/health_table_v1.yaml"),
        ("tmp_path_copy",     "loads an identical doc written to tmp_path"),
    ],
)
def test_loads_canonical_fixture(source: str, desc: str, tmp_path: Path) -> None:
    if source == "on_disk_canonical":
        path = CANONICAL_FIXTURE
    else:
        path = _write_doc(tmp_path / "ht.yaml", copy.deepcopy(_CANONICAL_DOC))

    table = load_health_table(path)
    assert isinstance(table, HealthTable), desc
    assert isinstance(table.patient, Patient), desc
    assert isinstance(table.vitals, Vitals), desc
    assert table.patient.name == "Test Patient", desc
    assert table.patient.age == 45, desc
    assert table.vitals.heart_rate_bpm == 72, desc
    assert table.vitals.blood_pressure_systolic == 118, desc
    assert table.vitals.blood_pressure_diastolic == 76, desc
    assert table.vitals.spo2_percent == 98, desc
    assert table.vitals.body_temperature_c == pytest.approx(36.7), desc
    assert table.vitals.respiratory_rate == 16, desc
    assert table.notes == (
        "Vitals within nominal range",
        "No medication interactions flagged",
    ), desc


# | missing_key | desc                                                        |
@pytest.mark.parametrize(
    ("missing_key", "desc"),
    [
        ("patient", "ValueError names missing patient block"),
        ("vitals",  "ValueError names missing vitals block"),
        ("notes",   "ValueError names missing notes list"),
    ],
)
def test_rejects_missing_top_level_keys(
    missing_key: str,
    desc: str,
    tmp_path: Path,
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    del doc[missing_key]
    path = _write_doc(tmp_path / "broken.yaml", doc)

    with pytest.raises(ValueError, match=missing_key):
        load_health_table(path)
    assert desc  # keep desc referenced so -v output is self-identifying


# | vital_key                  | bad_value | direction | desc                          |
@pytest.mark.parametrize(
    ("vital_key", "bad_value", "direction", "desc"),
    [
        ("heart_rate_bpm",           29,    "low",  "HR 29 below [30..200]"),
        ("heart_rate_bpm",           201,   "high", "HR 201 above [30..200]"),
        ("blood_pressure_systolic",  59,    "low",  "BP-sys 59 below [60..200]"),
        ("blood_pressure_systolic",  201,   "high", "BP-sys 201 above [60..200]"),
        ("blood_pressure_diastolic", 39,    "low",  "BP-dia 39 below [40..130]"),
        ("blood_pressure_diastolic", 131,   "high", "BP-dia 131 above [40..130]"),
        ("spo2_percent",             69,    "low",  "SpO2 69 below [70..100]"),
        ("spo2_percent",             101,   "high", "SpO2 101 above [70..100]"),
        ("body_temperature_c",       33.9,  "low",  "T 33.9 below [34.0..42.0]"),
        ("body_temperature_c",       42.1,  "high", "T 42.1 above [34.0..42.0]"),
        ("respiratory_rate",         7,     "low",  "RR 7 below [8..40]"),
        ("respiratory_rate",         41,    "high", "RR 41 above [8..40]"),
    ],
)
def test_rejects_out_of_range_vital(
    vital_key: str,
    bad_value: float,
    direction: str,
    desc: str,
    tmp_path: Path,
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    vitals = doc["vitals"]
    assert isinstance(vitals, dict), desc
    vitals[vital_key] = bad_value
    path = _write_doc(tmp_path / f"oor_{vital_key}_{direction}.yaml", doc)

    with pytest.raises(ValueError, match=vital_key):
        load_health_table(path)


# | vital_key             | edge_value | desc                                          |
@pytest.mark.parametrize(
    ("vital_key", "edge_value", "desc"),
    [
        ("heart_rate_bpm",      30,    "HR 30 (lower bound) accepted"),
        ("heart_rate_bpm",      200,   "HR 200 (upper bound) accepted"),
        ("spo2_percent",        100,   "SpO2 100% accepted"),
        ("body_temperature_c",  34.0,  "T 34.0 (lower bound) accepted"),
        ("body_temperature_c",  42.0,  "T 42.0 (upper bound) accepted"),
        ("respiratory_rate",    8,     "RR 8 (lower bound) accepted"),
    ],
)
def test_accepts_inclusive_boundaries(
    vital_key: str,
    edge_value: float,
    desc: str,
    tmp_path: Path,
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    vitals = doc["vitals"]
    assert isinstance(vitals, dict), desc
    vitals[vital_key] = edge_value
    path = _write_doc(tmp_path / f"edge_{vital_key}.yaml", doc)

    table = load_health_table(path)
    actual = getattr(table.vitals, vital_key)
    assert actual == pytest.approx(edge_value), desc


# | notes_value                                       | desc                          |
@pytest.mark.parametrize(
    ("notes_value", "desc"),
    [
        (
            ["Vitals within nominal range", "No medication interactions flagged"],
            "canonical 2-element notes list parses as tuple of strings",
        ),
        (
            [],
            "empty notes list parses to empty tuple",
        ),
    ],
)
def test_notes_is_a_list_of_strings(
    notes_value: list[str],
    desc: str,
    tmp_path: Path,
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    doc["notes"] = notes_value
    path = _write_doc(tmp_path / "notes.yaml", doc)

    table = load_health_table(path)
    assert isinstance(table.notes, tuple), desc
    assert list(table.notes) == notes_value, desc
    for n in table.notes:
        assert isinstance(n, str), desc


# --------------------------------------------------------------------------
# Expanded schema (2026-04-24 pivot): optional blocks for conditions,
# allergies, medications, dietary_restrictions, appointments, emergency_contacts.
# Narrow `_CANONICAL_DOC` above continues to exercise the "all defaults empty"
# backward-compat path; cases below exercise the full schema.
# --------------------------------------------------------------------------

# | block                  | expected_substring                | desc                                         |
@pytest.mark.parametrize(
    ("block", "expected_substring", "desc"),
    [
        ("conditions",           "Hypertension",  "Hypertension appears in conditions block"),
        ("conditions",           "Type 2 Diabetes", "Diabetes condition present"),
        ("allergies",            "Penicillin",    "Penicillin appears in allergies"),
        ("medications",          "Lisinopril",    "Lisinopril appears in medications"),
        ("medications",          "Atorvastatin",  "Atorvastatin appears in medications"),
        ("dietary_restrictions", "low sodium",    "low sodium rule present"),
        ("appointments",         "2026-05-06",    "May 6 appointment date present"),
        ("emergency_contacts",   "Jane Doe",      "Jane Doe emergency contact present"),
    ],
)
def test_canonical_fixture_has_expanded_block_content(
    block: str, expected_substring: str, desc: str
) -> None:
    """The canonical on-disk fixture loads the full expanded schema. Asserts
    per-block content via substring check against the first item's string
    fields — any change to the YAML authoring that drops these entries will
    surface here rather than during bench."""
    table = load_health_table(CANONICAL_FIXTURE)
    items = getattr(table, block)
    assert len(items) >= 1, f"{desc}: block {block!r} empty on canonical fixture"
    haystack = " ".join(
        v for item in items for v in (
            str(getattr(item, f)) for f in item.__class__.__dataclass_fields__
        )
    )
    assert expected_substring in haystack, desc


# | block                  | item_type            | desc                          |
@pytest.mark.parametrize(
    ("block", "item_type", "desc"),
    [
        ("conditions",           Condition,          "conditions elements are Condition"),
        ("allergies",            Allergy,            "allergies elements are Allergy"),
        ("medications",          Medication,         "medications elements are Medication"),
        ("dietary_restrictions", DietaryRestriction, "dietary_restrictions are DietaryRestriction"),
        ("appointments",         Appointment,        "appointments elements are Appointment"),
        ("emergency_contacts",   EmergencyContact,   "emergency_contacts are EmergencyContact"),
    ],
)
def test_canonical_fixture_blocks_are_frozen_dataclasses(
    block: str, item_type: type, desc: str
) -> None:
    """Block items must be the declared frozen-dataclass types so that
    downstream consumers (prompt composer, bench scorer) can rely on fixed
    attribute names."""
    table = load_health_table(CANONICAL_FIXTURE)
    items = getattr(table, block)
    assert isinstance(items, tuple), desc
    for item in items:
        assert isinstance(item, item_type), desc


# | missing_block              | desc                                                   |
@pytest.mark.parametrize(
    ("missing_block", "desc"),
    [
        ("conditions",           "missing conditions block → empty tuple"),
        ("allergies",            "missing allergies block → empty tuple"),
        ("medications",          "missing medications block → empty tuple"),
        ("dietary_restrictions", "missing dietary_restrictions → empty tuple"),
        ("appointments",         "missing appointments → empty tuple"),
        ("emergency_contacts",   "missing emergency_contacts → empty tuple"),
    ],
)
def test_missing_optional_block_defaults_to_empty_tuple(
    missing_block: str, desc: str, tmp_path: Path
) -> None:
    """A minimal fixture (patient + vitals + notes only, no new blocks) must
    still load — this is the backward-compat guarantee for the Phase A
    fixture shape."""
    doc = copy.deepcopy(_CANONICAL_DOC)  # already has only the required keys
    path = _write_doc(tmp_path / "minimal.yaml", doc)
    table = load_health_table(path)
    assert getattr(table, missing_block) == (), desc


# | field_to_drop | desc                                                         |
@pytest.mark.parametrize(
    ("field_to_drop", "desc"),
    [
        ("name",     "medication missing 'name' raises with 'name' in message"),
        ("dose",     "medication missing 'dose' raises"),
        ("schedule", "medication missing 'schedule' raises"),
        ("with_food", "medication missing 'with_food' raises"),
        ("purpose",  "medication missing 'purpose' raises"),
    ],
)
def test_rejects_malformed_medication(
    field_to_drop: str, desc: str, tmp_path: Path
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    med = {
        "name": "Lisinopril",
        "dose": "10 mg",
        "schedule": "08:00",
        "with_food": False,
        "purpose": "blood pressure control",
    }
    del med[field_to_drop]
    doc["medications"] = [med]
    path = _write_doc(tmp_path / f"bad_med_{field_to_drop}.yaml", doc)
    with pytest.raises(ValueError, match=field_to_drop):
        load_health_table(path)


# | bad_severity | block        | desc                                            |
@pytest.mark.parametrize(
    ("bad_severity", "block", "desc"),
    [
        ("critical", "conditions", "conditions reject 'critical' severity"),
        ("lethal",   "allergies",  "allergies reject 'lethal' severity"),
        ("",         "conditions", "empty-string severity rejected"),
    ],
)
def test_rejects_out_of_set_severity(
    bad_severity: str, block: str, desc: str, tmp_path: Path
) -> None:
    doc = copy.deepcopy(_CANONICAL_DOC)
    if block == "conditions":
        doc["conditions"] = [{
            "name": "Hypertension", "diagnosed_at": "2020-01-01",
            "severity": bad_severity, "controlled": True,
        }]
    else:
        doc["allergies"] = [{
            "substance": "Penicillin", "severity": bad_severity,
            "reaction": "anaphylaxis",
        }]
    path = _write_doc(tmp_path / f"bad_severity_{block}.yaml", doc)
    with pytest.raises(ValueError, match="severity"):
        load_health_table(path)


def test_medication_with_empty_interactions() -> None:
    """A medication with no food/drug interactions (the common case) loads
    with empty-tuple defaults rather than requiring explicit empty lists."""
    table = load_health_table(CANONICAL_FIXTURE)
    # Vitamin D3 in the canonical fixture has no interactions.
    vit_d = next((m for m in table.medications if m.name == "Vitamin D3"), None)
    assert vit_d is not None, "Vitamin D3 missing from canonical fixture"
    assert vit_d.avoid_foods == (), "Vitamin D3 avoid_foods should be empty tuple"
    assert vit_d.avoid_drugs == (), "Vitamin D3 avoid_drugs should be empty tuple"
