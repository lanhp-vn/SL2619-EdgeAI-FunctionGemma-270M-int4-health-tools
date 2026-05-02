#!/usr/bin/env python3
"""Hand-authored FunctionGemma seed conversation generator (M4).

Reads `data/functiongemma/tools_v1.yaml` (the frozen mirror of the M3 read-only
tool registry) and writes `data/functiongemma/seed_conversations.jsonl`. Every
conversation literal in `_CONVERSATIONS` below is hand-authored against the
synthetic `Test Patient` fixture in `data/health_table_v1.yaml`; the script
just stamps the per-row `tools` block from the registry mirror so a registry
description tweak does not require touching all 50 rows.

Conversation shape and the assistant.content `<think>` discipline are pinned in
`docs/plans/FunctionGemma/README.md` §9.4.2 and enforced by
`src/gemma_tools/functiongemma_dataset.py`. See
`docs/plans/FunctionGemma/seed-authoring-recipe.md` for the recipe + worked
examples.

Usage:
    uv run python scripts/build_functiongemma_seeds.py
    uv run python scripts/build_functiongemma_seeds.py --check    # exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[3]
_TOOLS_YAML = _REPO / "data" / "functiongemma" / "tools_v1.yaml"
_OUT_PATH = _REPO / "data" / "functiongemma" / "seed_conversations.jsonl"

_SYSTEM_TRIGGER = "You are a model that can do function calling with the following functions"
_REFUSAL_OFF_TOPIC = "I answer questions from your health record only."
_REFUSAL_MEDICAL = (
    "I cannot give medical advice. Please consult your clinician about that."
)


def _system() -> dict[str, str]:
    return {"role": "system", "content": _SYSTEM_TRIGGER}


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant_call(think: str, calls: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    """Assistant turn that emits one or more tool_calls.

    `calls` is a list of (id, name, arguments) tuples. The content shape is the
    `<think>` block only (no NL tail) — the validator's
    `_validate_assistant_content_shape` enforces this for assistants with
    tool_calls.
    """
    return {
        "role": "assistant",
        "content": f"<think>{think}</think>",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in calls
        ],
    }


def _assistant_answer(think: str, answer: str) -> dict[str, str]:
    """Assistant turn that emits the final NL answer.

    Without `tool_calls`, the validator requires `<think>...</think>\\n<answer>`.
    """
    return {"role": "assistant", "content": f"<think>{think}</think>\n{answer}"}


def _tool(call_id: str, name: str, payload: Any) -> dict[str, str]:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


# --------------------------------------------------------------------------
# Canonical tool-result fixtures, derived 1:1 from data/health_table_v1.yaml.
# Keep these here (not in the YAML) so the seed remains a self-contained
# training artifact — the YAML is the *system-under-test* fixture and could
# evolve; the seed must freeze the snapshot it was authored against.
# --------------------------------------------------------------------------

_VITALS = {
    "heart_rate_bpm": 72,
    "blood_pressure_systolic": 118,
    "blood_pressure_diastolic": 76,
    "spo2_percent": 98,
    "body_temperature_c": 36.7,
    "respiratory_rate": 16,
    "last_measured": "2026-04-24 08:15",
}

_ALLERGIES = [
    {"substance": "Penicillin", "severity": "severe", "reaction": "anaphylaxis"},
    {"substance": "Shellfish", "severity": "moderate", "reaction": "hives"},
]

_NEXT_APPT = {
    "date": "2026-05-06",
    "time": "10:30",
    "provider": "Dr. Evelyn Chen",
    "purpose": "quarterly diabetes check-up",
    "location": "Maple Clinic, Room 204",
}

_EMERGENCY = {"name": "Jane Doe", "relation": "daughter", "phone": "+1-555-0142"}

_MED_LISINOPRIL = {
    "name": "Lisinopril",
    "dose": "10 mg",
    "schedule": "08:00",
    "with_food": False,
    "purpose": "blood pressure control",
    "avoid_foods": [],
    "avoid_drugs": ["Potassium supplements", "NSAIDs"],
}
_MED_METFORMIN = {
    "name": "Metformin",
    "dose": "500 mg",
    "schedule": "08:00, 19:00",
    "with_food": True,
    "purpose": "blood sugar control",
    "avoid_foods": ["Alcohol"],
    "avoid_drugs": [],
}
_MED_ATORVASTATIN = {
    "name": "Atorvastatin",
    "dose": "20 mg",
    "schedule": "21:00",
    "with_food": False,
    "purpose": "cholesterol control",
    "avoid_foods": ["Grapefruit", "Grapefruit juice"],
    "avoid_drugs": ["Erythromycin"],
}
_MED_ASPIRIN = {
    "name": "Aspirin",
    "dose": "81 mg",
    "schedule": "08:00",
    "with_food": True,
    "purpose": "cardiovascular protection",
    "avoid_foods": [],
    "avoid_drugs": ["Ibuprofen", "Warfarin"],
}
_MED_VITD = {
    "name": "Vitamin D3",
    "dose": "1000 IU",
    "schedule": "08:00",
    "with_food": True,
    "purpose": "bone health",
    "avoid_foods": [],
    "avoid_drugs": [],
}

# Light-shape variant returned by `get_medications_at_time` (no avoid_*).
def _at_time(med: dict[str, Any]) -> dict[str, Any]:
    return {k: med[k] for k in ("name", "dose", "schedule", "with_food", "purpose")}


_MORNING_MEDS = [
    _at_time(_MED_LISINOPRIL),
    _at_time(_MED_METFORMIN),
    _at_time(_MED_ASPIRIN),
    _at_time(_MED_VITD),
]
_EVENING_MEDS = [_at_time(_MED_METFORMIN)]
_NIGHT_MEDS = [_at_time(_MED_ATORVASTATIN)]


def _interaction(food: str, with_meds: list[str], rule: str | None) -> dict[str, Any]:
    return {"food": food, "interacts": bool(with_meds) or bool(rule), "with_meds": with_meds, "rule": rule}


# --------------------------------------------------------------------------
# Conversations. Each row is `{id, category, messages}`; the runner stamps the
# per-row `tools` block from the registry mirror.
# --------------------------------------------------------------------------


def _build_conversations() -> list[dict[str, Any]]:
    convos: list[dict[str, Any]] = []

    # ---- Single-turn fact lookup (12) -----------------------------------
    convos.extend([
        {
            "id": "fl-001",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What's my heart rate?"),
                _assistant_call("User wants vitals; call get_vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer("HR is 72 bpm.", "Your heart rate is 72 bpm."),
            ],
        },
        {
            "id": "fl-002",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What is my blood pressure?"),
                _assistant_call("User wants vitals; call get_vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer("BP is 118/76.", "Your blood pressure is 118/76 mmHg."),
            ],
        },
        {
            "id": "fl-003",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What's my body temperature?"),
                _assistant_call("Vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer("36.7 C.", "Your body temperature is 36.7°C."),
            ],
        },
        {
            "id": "fl-004",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What allergies do I have?"),
                _assistant_call("Allergy list.", [("call_1", "list_allergies", {})]),
                _tool("call_1", "list_allergies", _ALLERGIES),
                _assistant_answer(
                    "Two allergies on file.",
                    "You have two allergies: Penicillin (severe, anaphylaxis) and Shellfish (moderate, hives).",
                ),
            ],
        },
        {
            "id": "fl-005",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What medications do I take in the morning?"),
                _assistant_call(
                    "Morning is 08:00; call get_medications_at_time.",
                    [("call_1", "get_medications_at_time", {"time_24h": "08:00"})],
                ),
                _tool("call_1", "get_medications_at_time", _MORNING_MEDS),
                _assistant_answer(
                    "Four meds at 08:00.",
                    "At 08:00 you take Lisinopril 10 mg, Metformin 500 mg with food, Aspirin 81 mg with food, and Vitamin D3 1000 IU with food.",
                ),
            ],
        },
        {
            "id": "fl-006",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What dose of metformin do I take?"),
                _assistant_call(
                    "Look up Metformin.",
                    [("call_1", "get_medication_by_name", {"name": "metformin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_METFORMIN),
                _assistant_answer(
                    "Metformin 500 mg, twice daily.",
                    "Metformin 500 mg, taken twice daily at 08:00 and 19:00 with food.",
                ),
            ],
        },
        {
            "id": "fl-007",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("When is my next appointment?"),
                _assistant_call("Next appointment.", [("call_1", "get_next_appointment", {})]),
                _tool("call_1", "get_next_appointment", _NEXT_APPT),
                _assistant_answer(
                    "Diabetes check-up in early May.",
                    "Your next appointment is on 2026-05-06 at 10:30 with Dr. Evelyn Chen for a quarterly diabetes check-up.",
                ),
            ],
        },
        {
            "id": "fl-008",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("Who's my emergency contact?"),
                _assistant_call("Emergency contact.", [("call_1", "get_emergency_contact", {})]),
                _tool("call_1", "get_emergency_contact", _EMERGENCY),
                _assistant_answer(
                    "Jane Doe.",
                    "Your emergency contact is Jane Doe (daughter), phone +1-555-0142.",
                ),
            ],
        },
        {
            "id": "fl-009",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("Does grapefruit interact with my medications?"),
                _assistant_call(
                    "Check food interaction for grapefruit.",
                    [("call_1", "check_food_interaction", {"food": "grapefruit"})],
                ),
                _tool(
                    "call_1",
                    "check_food_interaction",
                    _interaction("grapefruit", ["Atorvastatin"], "no grapefruit"),
                ),
                _assistant_answer(
                    "Interacts with Atorvastatin.",
                    "Yes — grapefruit interacts with Atorvastatin, and you have a dietary restriction \"no grapefruit\" on file.",
                ),
            ],
        },
        {
            "id": "fl-010",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What is my SpO2?"),
                _assistant_call("Vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer("SpO2 98 percent.", "Your SpO2 is 98%."),
            ],
        },
        {
            "id": "fl-011",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("When do I take Atorvastatin?"),
                _assistant_call(
                    "Look up Atorvastatin schedule.",
                    [("call_1", "get_medication_by_name", {"name": "atorvastatin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_ATORVASTATIN),
                _assistant_answer(
                    "21:00.",
                    "Atorvastatin 20 mg is scheduled for 21:00, taken without food.",
                ),
            ],
        },
        {
            "id": "fl-012",
            "category": "fact_lookup",
            "messages": [
                _system(),
                _user("What should I take at 7 pm?"),
                _assistant_call(
                    "Evening slot is 19:00.",
                    [("call_1", "get_medications_at_time", {"time_24h": "19:00"})],
                ),
                _tool("call_1", "get_medications_at_time", _EVENING_MEDS),
                _assistant_answer(
                    "Metformin only.",
                    "At 19:00 you take Metformin 500 mg with food.",
                ),
            ],
        },
    ])

    # ---- Single-turn off-topic refusal (4) ------------------------------
    convos.extend([
        {
            "id": "ot-001",
            "category": "off_topic_refusal",
            "messages": [
                _system(),
                _user("Tell me a joke."),
                _assistant_answer("Off-topic; refuse without tool call.", _REFUSAL_OFF_TOPIC),
            ],
        },
        {
            "id": "ot-002",
            "category": "off_topic_refusal",
            "messages": [
                _system(),
                _user("What's the weather today?"),
                _assistant_answer(
                    "Weather is not in scope; refuse without tool call.",
                    _REFUSAL_OFF_TOPIC,
                ),
            ],
        },
        {
            "id": "ot-003",
            "category": "off_topic_refusal",
            "messages": [
                _system(),
                _user("Who won the world cup last year?"),
                _assistant_answer(
                    "Sports trivia is not in scope.",
                    _REFUSAL_OFF_TOPIC,
                ),
            ],
        },
        {
            "id": "ot-004",
            "category": "off_topic_refusal",
            "messages": [
                _system(),
                _user("Can you book me a flight to Paris?"),
                _assistant_answer(
                    "Booking flights is not in scope.",
                    _REFUSAL_OFF_TOPIC,
                ),
            ],
        },
    ])

    # ---- Single-turn fact-absence (4) -----------------------------------
    convos.extend([
        {
            "id": "fa-001",
            "category": "fact_absence",
            "messages": [
                _system(),
                _user("What's my cholesterol level?"),
                _assistant_call("Cholesterol may be in vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer(
                    "No cholesterol field in the response.",
                    "Cholesterol level is not in your record.",
                ),
            ],
        },
        {
            "id": "fa-002",
            "category": "fact_absence",
            "messages": [
                _system(),
                _user("What's my fasting blood glucose?"),
                _assistant_call("Check vitals for glucose.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer(
                    "No glucose field.",
                    "Blood glucose is not in your record.",
                ),
            ],
        },
        {
            "id": "fa-003",
            "category": "fact_absence",
            "messages": [
                _system(),
                _user("What was my weight at the last visit?"),
                _assistant_call("Vitals only — no weight field expected.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer(
                    "No weight field.",
                    "Weight is not in your record.",
                ),
            ],
        },
        {
            "id": "fa-004",
            "category": "fact_absence",
            "messages": [
                _system(),
                _user("What's my A1C?"),
                _assistant_call("Try vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer(
                    "No A1C field in vitals.",
                    "A1C is not in your record.",
                ),
            ],
        },
    ])

    # ---- Parallel calls (6) ---------------------------------------------
    convos.extend([
        {
            "id": "pc-001",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("What allergies do I have, and what's my BP?"),
                _assistant_call(
                    "Two independent lookups: list_allergies + get_vitals.",
                    [("call_1", "list_allergies", {}), ("call_2", "get_vitals", {})],
                ),
                _tool("call_1", "list_allergies", _ALLERGIES),
                _tool("call_2", "get_vitals", _VITALS),
                _assistant_answer(
                    "Combine both results.",
                    "Allergies: Penicillin (severe) and Shellfish (moderate). Blood pressure: 118/76 mmHg.",
                ),
            ],
        },
        {
            "id": "pc-002",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("What's my next appointment and who's my emergency contact?"),
                _assistant_call(
                    "Two independent lookups.",
                    [
                        ("call_1", "get_next_appointment", {}),
                        ("call_2", "get_emergency_contact", {}),
                    ],
                ),
                _tool("call_1", "get_next_appointment", _NEXT_APPT),
                _tool("call_2", "get_emergency_contact", _EMERGENCY),
                _assistant_answer(
                    "Combine both.",
                    "Next appointment: 2026-05-06 at 10:30 with Dr. Evelyn Chen. Emergency contact: Jane Doe (daughter), +1-555-0142.",
                ),
            ],
        },
        {
            "id": "pc-003",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("What meds at 8 am, and any allergies?"),
                _assistant_call(
                    "Morning meds + allergies.",
                    [
                        ("call_1", "get_medications_at_time", {"time_24h": "08:00"}),
                        ("call_2", "list_allergies", {}),
                    ],
                ),
                _tool("call_1", "get_medications_at_time", _MORNING_MEDS),
                _tool("call_2", "list_allergies", _ALLERGIES),
                _assistant_answer(
                    "Both results.",
                    "At 08:00: Lisinopril 10 mg, Metformin 500 mg with food, Aspirin 81 mg with food, Vitamin D3 1000 IU with food. Allergies: Penicillin (severe) and Shellfish (moderate).",
                ),
            ],
        },
        {
            "id": "pc-004",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("Vitals and morning meds, please."),
                _assistant_call(
                    "Vitals + morning slot.",
                    [
                        ("call_1", "get_vitals", {}),
                        ("call_2", "get_medications_at_time", {"time_24h": "08:00"}),
                    ],
                ),
                _tool("call_1", "get_vitals", _VITALS),
                _tool("call_2", "get_medications_at_time", _MORNING_MEDS),
                _assistant_answer(
                    "Combine both.",
                    "Vitals: HR 72, BP 118/76, SpO2 98%, temp 36.7°C, RR 16 (measured 2026-04-24 08:15). Morning meds: Lisinopril 10 mg, Metformin 500 mg with food, Aspirin 81 mg with food, Vitamin D3 1000 IU with food.",
                ),
            ],
        },
        {
            "id": "pc-005",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("What dose for metformin and lisinopril?"),
                _assistant_call(
                    "Two by-name lookups.",
                    [
                        ("call_1", "get_medication_by_name", {"name": "metformin"}),
                        ("call_2", "get_medication_by_name", {"name": "lisinopril"}),
                    ],
                ),
                _tool("call_1", "get_medication_by_name", _MED_METFORMIN),
                _tool("call_2", "get_medication_by_name", _MED_LISINOPRIL),
                _assistant_answer(
                    "Two doses.",
                    "Metformin 500 mg twice daily with food. Lisinopril 10 mg once daily at 08:00.",
                ),
            ],
        },
        {
            "id": "pc-006",
            "category": "parallel_call",
            "messages": [
                _system(),
                _user("Next appointment plus my BP?"),
                _assistant_call(
                    "Appointment + vitals.",
                    [
                        ("call_1", "get_next_appointment", {}),
                        ("call_2", "get_vitals", {}),
                    ],
                ),
                _tool("call_1", "get_next_appointment", _NEXT_APPT),
                _tool("call_2", "get_vitals", _VITALS),
                _assistant_answer(
                    "Combine.",
                    "Next appointment: 2026-05-06 at 10:30 with Dr. Evelyn Chen. Blood pressure: 118/76 mmHg.",
                ),
            ],
        },
    ])

    # ---- Two-turn slot-fill / follow-up (14) ----------------------------
    convos.extend([
        {
            "id": "tt-001",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("What dose of metformin?"),
                _assistant_call(
                    "Look up Metformin.",
                    [("call_1", "get_medication_by_name", {"name": "metformin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_METFORMIN),
                _assistant_answer("Dose answer.", "Metformin 500 mg, twice daily."),
                _user("With food?"),
                _assistant_call(
                    "Re-fetch with_food field.",
                    [("call_2", "get_medication_by_name", {"name": "metformin"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_METFORMIN),
                _assistant_answer("with_food=true.", "Yes, take Metformin with food."),
            ],
        },
        {
            "id": "tt-002",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("When is my next appointment?"),
                _assistant_call("Next appointment.", [("call_1", "get_next_appointment", {})]),
                _tool("call_1", "get_next_appointment", _NEXT_APPT),
                _assistant_answer(
                    "Date + provider.",
                    "Your next appointment is 2026-05-06 at 10:30 with Dr. Evelyn Chen.",
                ),
                _user("Where?"),
                _assistant_call("Re-fetch for location.", [("call_2", "get_next_appointment", {})]),
                _tool("call_2", "get_next_appointment", _NEXT_APPT),
                _assistant_answer(
                    "Location field.",
                    "It is at Maple Clinic, Room 204.",
                ),
            ],
        },
        {
            "id": "tt-003",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("What's my blood pressure?"),
                _assistant_call("Vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer("BP.", "Your blood pressure is 118/76 mmHg."),
                _user("And heart rate?"),
                _assistant_call("Re-fetch vitals for HR.", [("call_2", "get_vitals", {})]),
                _tool("call_2", "get_vitals", _VITALS),
                _assistant_answer("HR.", "Your heart rate is 72 bpm."),
            ],
        },
        {
            "id": "tt-004",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Tell me about lisinopril."),
                _assistant_call(
                    "Look up Lisinopril.",
                    [("call_1", "get_medication_by_name", {"name": "lisinopril"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_LISINOPRIL),
                _assistant_answer(
                    "Summary.",
                    "Lisinopril 10 mg once daily at 08:00, for blood pressure control.",
                ),
                _user("Any drugs to avoid?"),
                _assistant_call(
                    "Re-fetch avoid_drugs.",
                    [("call_2", "get_medication_by_name", {"name": "lisinopril"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_LISINOPRIL),
                _assistant_answer(
                    "avoid_drugs list.",
                    "Avoid Potassium supplements and NSAIDs while taking Lisinopril.",
                ),
            ],
        },
        {
            "id": "tt-005",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("What allergies?"),
                _assistant_call("Allergies.", [("call_1", "list_allergies", {})]),
                _tool("call_1", "list_allergies", _ALLERGIES),
                _assistant_answer(
                    "Two allergies.",
                    "You're allergic to Penicillin and Shellfish.",
                ),
                _user("Which is severe?"),
                _assistant_call("Re-fetch for severity.", [("call_2", "list_allergies", {})]),
                _tool("call_2", "list_allergies", _ALLERGIES),
                _assistant_answer(
                    "Penicillin severity.",
                    "Penicillin is severe (anaphylaxis); Shellfish is moderate (hives).",
                ),
            ],
        },
        {
            "id": "tt-006",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("What are my morning meds?"),
                _assistant_call(
                    "Morning slot.",
                    [("call_1", "get_medications_at_time", {"time_24h": "08:00"})],
                ),
                _tool("call_1", "get_medications_at_time", _MORNING_MEDS),
                _assistant_answer(
                    "Four meds.",
                    "At 08:00: Lisinopril, Metformin, Aspirin, Vitamin D3.",
                ),
                _user("And in the evening?"),
                _assistant_call(
                    "Evening slot.",
                    [("call_2", "get_medications_at_time", {"time_24h": "19:00"})],
                ),
                _tool("call_2", "get_medications_at_time", _EVENING_MEDS),
                _assistant_answer(
                    "Metformin only.",
                    "At 19:00: Metformin 500 mg with food.",
                ),
            ],
        },
        {
            "id": "tt-007",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Next appointment?"),
                _assistant_call("Next appointment.", [("call_1", "get_next_appointment", {})]),
                _tool("call_1", "get_next_appointment", _NEXT_APPT),
                _assistant_answer(
                    "Date + time.",
                    "It is on 2026-05-06 at 10:30.",
                ),
                _user("Who's the provider?"),
                _assistant_call("Re-fetch for provider.", [("call_2", "get_next_appointment", {})]),
                _tool("call_2", "get_next_appointment", _NEXT_APPT),
                _assistant_answer(
                    "Provider.",
                    "Dr. Evelyn Chen.",
                ),
            ],
        },
        {
            "id": "tt-008",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Vitals?"),
                _assistant_call("Vitals.", [("call_1", "get_vitals", {})]),
                _tool("call_1", "get_vitals", _VITALS),
                _assistant_answer(
                    "Summary.",
                    "HR 72, BP 118/76, SpO2 98%, temp 36.7°C, RR 16.",
                ),
                _user("When were these measured?"),
                _assistant_call("Re-fetch for last_measured.", [("call_2", "get_vitals", {})]),
                _tool("call_2", "get_vitals", _VITALS),
                _assistant_answer(
                    "Timestamp.",
                    "Last measured on 2026-04-24 at 08:15.",
                ),
            ],
        },
        {
            "id": "tt-009",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Tell me about Atorvastatin."),
                _assistant_call(
                    "Look up Atorvastatin.",
                    [("call_1", "get_medication_by_name", {"name": "atorvastatin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_ATORVASTATIN),
                _assistant_answer(
                    "Summary.",
                    "Atorvastatin 20 mg at 21:00 for cholesterol control.",
                ),
                _user("Should I take it with food?"),
                _assistant_call(
                    "Re-fetch with_food.",
                    [("call_2", "get_medication_by_name", {"name": "atorvastatin"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_ATORVASTATIN),
                _assistant_answer(
                    "with_food=false.",
                    "No, take Atorvastatin without food.",
                ),
            ],
        },
        {
            "id": "tt-010",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Does grapefruit interact with anything?"),
                _assistant_call(
                    "Check grapefruit.",
                    [("call_1", "check_food_interaction", {"food": "grapefruit"})],
                ),
                _tool(
                    "call_1",
                    "check_food_interaction",
                    _interaction("grapefruit", ["Atorvastatin"], "no grapefruit"),
                ),
                _assistant_answer(
                    "Atorvastatin.",
                    "Yes — grapefruit interacts with Atorvastatin.",
                ),
                _user("And alcohol?"),
                _assistant_call(
                    "Check alcohol.",
                    [("call_2", "check_food_interaction", {"food": "alcohol"})],
                ),
                _tool(
                    "call_2",
                    "check_food_interaction",
                    _interaction("alcohol", ["Metformin"], None),
                ),
                _assistant_answer(
                    "Metformin.",
                    "Alcohol is on Metformin's avoid-foods list.",
                ),
            ],
        },
        {
            "id": "tt-011",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Any meds at 9 pm?"),
                _assistant_call(
                    "21:00 slot.",
                    [("call_1", "get_medications_at_time", {"time_24h": "21:00"})],
                ),
                _tool("call_1", "get_medications_at_time", _NIGHT_MEDS),
                _assistant_answer(
                    "Atorvastatin.",
                    "At 21:00 you take Atorvastatin 20 mg.",
                ),
                _user("What's it for?"),
                _assistant_call(
                    "Look up Atorvastatin purpose.",
                    [("call_2", "get_medication_by_name", {"name": "atorvastatin"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_ATORVASTATIN),
                _assistant_answer(
                    "purpose=cholesterol.",
                    "Atorvastatin is for cholesterol control.",
                ),
            ],
        },
        {
            "id": "tt-012",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Who's my emergency contact?"),
                _assistant_call("Contact.", [("call_1", "get_emergency_contact", {})]),
                _tool("call_1", "get_emergency_contact", _EMERGENCY),
                _assistant_answer(
                    "Jane Doe.",
                    "Jane Doe, your daughter.",
                ),
                _user("What's her phone?"),
                _assistant_call("Re-fetch for phone.", [("call_2", "get_emergency_contact", {})]),
                _tool("call_2", "get_emergency_contact", _EMERGENCY),
                _assistant_answer(
                    "Phone.",
                    "+1-555-0142.",
                ),
            ],
        },
        {
            "id": "tt-013",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("Tell me about Vitamin D3."),
                _assistant_call(
                    "Look up Vitamin D3.",
                    [("call_1", "get_medication_by_name", {"name": "vitamin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_VITD),
                _assistant_answer(
                    "Summary.",
                    "Vitamin D3 1000 IU at 08:00 for bone health.",
                ),
                _user("With food?"),
                _assistant_call(
                    "Re-fetch with_food.",
                    [("call_2", "get_medication_by_name", {"name": "vitamin d3"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_VITD),
                _assistant_answer(
                    "with_food=true.",
                    "Yes, take Vitamin D3 with food.",
                ),
            ],
        },
        {
            "id": "tt-014",
            "category": "two_turn",
            "messages": [
                _system(),
                _user("What's my aspirin dose?"),
                _assistant_call(
                    "Look up Aspirin.",
                    [("call_1", "get_medication_by_name", {"name": "aspirin"})],
                ),
                _tool("call_1", "get_medication_by_name", _MED_ASPIRIN),
                _assistant_answer("Dose.", "Aspirin 81 mg, taken with food at 08:00."),
                _user("Any drug to avoid?"),
                _assistant_call(
                    "Re-fetch avoid_drugs.",
                    [("call_2", "get_medication_by_name", {"name": "aspirin"})],
                ),
                _tool("call_2", "get_medication_by_name", _MED_ASPIRIN),
                _assistant_answer(
                    "avoid_drugs.",
                    "Avoid Ibuprofen and Warfarin while on Aspirin.",
                ),
            ],
        },
    ])

    # ---- Medical-advice refusal (4) -------------------------------------
    convos.extend([
        {
            "id": "ma-001",
            "category": "medical_advice_refusal",
            "messages": [
                _system(),
                _user("Should I stop taking aspirin?"),
                _assistant_answer(
                    "Medical advice; refuse without tool call.",
                    _REFUSAL_MEDICAL,
                ),
            ],
        },
        {
            "id": "ma-002",
            "category": "medical_advice_refusal",
            "messages": [
                _system(),
                _user("Should I increase my metformin dose?"),
                _assistant_answer(
                    "Medical advice; refuse.",
                    _REFUSAL_MEDICAL,
                ),
            ],
        },
        {
            "id": "ma-003",
            "category": "medical_advice_refusal",
            "messages": [
                _system(),
                _user("Do I need to see a cardiologist?"),
                _assistant_answer(
                    "Clinical judgment; refuse.",
                    _REFUSAL_MEDICAL,
                ),
            ],
        },
        {
            "id": "ma-004",
            "category": "medical_advice_refusal",
            "messages": [
                _system(),
                _user("Can I skip my evening dose tonight?"),
                _assistant_answer(
                    "Dose-skipping is clinical judgment; refuse.",
                    _REFUSAL_MEDICAL,
                ),
            ],
        },
    ])

    # ---- Tool-error recovery (6) ----------------------------------------
    # Note: the §9.3 README example ("pulse at 3am" → vitals snapshot only) is
    # awkward against the M3 registry — `get_vitals` does not accept a time.
    # We exercise actual registry error paths instead: ambiguous prefix,
    # no-match, and empty time slots. Documented in §M4 progress notes.
    convos.extend([
        {
            "id": "te-001",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Tell me about that A-something pill."),
                _assistant_call(
                    "Single-letter prefix; expect ambiguity.",
                    [("call_1", "get_medication_by_name", {"name": "A"})],
                ),
                _tool(
                    "call_1",
                    "get_medication_by_name",
                    {"error": "ambiguous", "name": "A", "matches": ["Aspirin", "Atorvastatin"]},
                ),
                _assistant_answer(
                    "Ambiguous; ask user.",
                    "Two medications start with \"A\": Aspirin and Atorvastatin. Which one did you mean?",
                ),
            ],
        },
        {
            "id": "te-002",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Do I take ibuprofen?"),
                _assistant_call(
                    "Look up Ibuprofen; expect no_match.",
                    [("call_1", "get_medication_by_name", {"name": "ibuprofen"})],
                ),
                _tool(
                    "call_1",
                    "get_medication_by_name",
                    {"error": "no_match", "name": "ibuprofen"},
                ),
                _assistant_answer(
                    "Not on file.",
                    "Ibuprofen is not in your medication list.",
                ),
            ],
        },
        {
            "id": "te-003",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Am I on warfarin?"),
                _assistant_call(
                    "Look up Warfarin; expect no_match.",
                    [("call_1", "get_medication_by_name", {"name": "warfarin"})],
                ),
                _tool(
                    "call_1",
                    "get_medication_by_name",
                    {"error": "no_match", "name": "warfarin"},
                ),
                _assistant_answer(
                    "Not on file. (Warfarin appears as a drug-to-avoid for Aspirin, but not as a med I take.)",
                    "Warfarin is not in your medication list.",
                ),
            ],
        },
        {
            "id": "te-004",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Anything to take at noon?"),
                _assistant_call(
                    "12:00 slot; likely empty.",
                    [("call_1", "get_medications_at_time", {"time_24h": "12:00"})],
                ),
                _tool("call_1", "get_medications_at_time", []),
                _assistant_answer(
                    "Empty list — none.",
                    "Nothing is scheduled for 12:00.",
                ),
            ],
        },
        {
            "id": "te-005",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Anything at 3 pm?"),
                _assistant_call(
                    "15:00 slot; likely empty.",
                    [("call_1", "get_medications_at_time", {"time_24h": "15:00"})],
                ),
                _tool("call_1", "get_medications_at_time", []),
                _assistant_answer(
                    "Empty.",
                    "Nothing is scheduled for 15:00.",
                ),
            ],
        },
        {
            "id": "te-006",
            "category": "tool_error_recovery",
            "messages": [
                _system(),
                _user("Am I prescribed Tylenol?"),
                _assistant_call(
                    "Look up Tylenol; expect no_match.",
                    [("call_1", "get_medication_by_name", {"name": "Tylenol"})],
                ),
                _tool(
                    "call_1",
                    "get_medication_by_name",
                    {"error": "no_match", "name": "Tylenol"},
                ),
                _assistant_answer(
                    "Not on file.",
                    "Tylenol is not in your medication list.",
                ),
            ],
        },
    ])

    return convos


def _load_tools() -> list[dict[str, Any]]:
    raw = yaml.safe_load(_TOOLS_YAML.read_text())
    if not isinstance(raw, list):
        raise SystemExit(f"{_TOOLS_YAML}: expected a top-level list, got {type(raw).__name__}")
    return raw


def _emit(convos: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    """Serialize one conversation per line to a single string."""
    lines: list[str] = []
    for c in convos:
        row = {
            "id": c["id"],
            "category": c["category"],
            "messages": c["messages"],
            "tools": tools,
        }
        # Compact JSON keeps the seed file diff-readable. `ensure_ascii=False`
        # so the rendered Unicode (degree sign, etc.) survives round-trip.
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the on-disk seed JSONL differs from the generator's output.",
    )
    args = p.parse_args(argv)

    tools = _load_tools()
    convos = _build_conversations()
    payload = _emit(convos, tools)

    if args.check:
        existing = _OUT_PATH.read_text() if _OUT_PATH.exists() else ""
        if existing != payload:
            sys.stderr.write(
                f"{_OUT_PATH}: drift — regenerate via `uv run python {Path(__file__).relative_to(_REPO)}`\n"
            )
            return 1
        print(f"{_OUT_PATH}: in sync ({len(convos)} conversations)")
        return 0

    _OUT_PATH.write_text(payload)
    print(f"{_OUT_PATH}: wrote {len(convos)} conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
