#!/usr/bin/env python3
"""Hand-authored dispenser-demo seed conversation generator (Phase 1.1).

Writes `data/dispenser_demo/seed_conversations.jsonl` — 42 rows across the
5 categories: 8 each for `patient_profile`, `next_appointment`,
`emergency_contact`, `dispense`, and 10 for `out_of_scope_refusal` (5
`health_advice` + 5 `off_topic`). The asymmetric refusal count is the
2026-05-11 rebalance after advisor pass on Phase 1.4 — keeping refusals at
8/8 across reasons makes synthgen's per-cluster signal stronger; see
`docs/plans/dispenser-demo/decisions-log.md` for the rationale.

Design bindings (locked 2026-05-11):

- `messages[0]` is the vendor SYSTEM_TRIGGER string (FG iter-001 pattern); the
  Sago routing prompt lives in Distil's `task_description.json` and in the
  on-board `model_client.py` — not in the seed.
- No wake phrase ("Hey Sago") in user turns — plan §8.3 E1 says the FSM
  strips it before the STT output is routed to the LLM.
- Tool responses carry a digit-free `*_words` companion for every
  digit-bearing key (the tool-boundary invariant from plan §1 / §11 R5).
- Assistant NL after the tool response quotes `*_words` verbatim — the
  training signal that teaches the model to speak TTS-friendly. The model's
  free narration is NOT regex-gated (plan §1, §10, §11 R5).
- `dispense_medication()` is the only side-effecting tool. Seven seeds show
  the happy path (`status: dispensed`) and one (`di-008`) shows the
  BLE-fail fallback (`status: ble_not_connected` → canned NL "I cannot
  reach the dispenser right now.").
- A fifth tool `refuse_out_of_scope(reason)` carries all 8 refusal rows so
  every assistant turn emits exactly one tool call (the Distil
  `multi-turn-tool-calling-closed-book` contract). Pattern mirrors the
  public `distil-labs/distil-home-assistant-functiongemma` model card's
  `intent_unclear(reason)` tool; the reason enum is tuned to our two
  refusal clusters (`health_advice`, `off_topic`).
- The `tools[]` block lists all five tools on every row (FG convention).

Re-run idempotently:

    uv run python scripts/dispenser_demo/data/build_seeds.py
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_OUT_PATH = _REPO / "data" / "dispenser_demo" / "seed_conversations.jsonl"

SYSTEM_TRIGGER = (
    "You are a model that can do function calling with the following functions"
)

# Tool registry — mirrors the planned `src/gemma_tools/dispenser_demo/tools.py`
# shape (Phase 1.2). All four tools take zero parameters; `dispense_medication`
# is the side-effecting one (BLE notify happens at dispatch time, then the
# post-action status is returned).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_patient_profile",
            "description": (
                "Return the patient's profile (name, age, sex, diagnoses) with "
                "digit-free `*_words` companions for every digit-bearing field."
            ),
            "parameters": {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_appointment",
            "description": (
                "Return the earliest upcoming appointment (date, time, "
                "provider, purpose, location) with digit-free `*_words` "
                "companions for every digit-bearing field."
            ),
            "parameters": {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_emergency_contact",
            "description": (
                "Return the first listed emergency contact (name, relation, "
                "phone) with a digit-free `phone_words` companion for the "
                "phone number."
            ),
            "parameters": {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispense_medication",
            "description": (
                "Dispense the patient's medication by sending a BLE "
                "notification to the dispenser. Returns the post-action "
                "status: 'dispensed' on success, 'ble_not_connected' if no "
                "BLE peer is subscribed."
            ),
            "parameters": {
                "additionalProperties": False,
                "properties": {},
                "required": [],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refuse_out_of_scope",
            "description": (
                "Call when the user request cannot be answered with the four "
                "domain tools above. Returns an acknowledgement; the runtime "
                "responds with a canned refusal sentence. `reason` is a "
                "two-value enum: 'health_advice' for medical advice / "
                "symptom diagnosis / treatment-plan requests, 'off_topic' "
                "for anything outside the health domain."
            ),
            "parameters": {
                "additionalProperties": False,
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["health_advice", "off_topic"],
                        "description": (
                            "Why the request is out of scope. Use "
                            "'health_advice' for medical advice, symptom "
                            "diagnosis, or treatment-plan questions. Use "
                            "'off_topic' for anything outside the health "
                            "domain (weather, news, jokes, math, generic "
                            "personal questions)."
                        ),
                    }
                },
                "required": ["reason"],
                "type": "object",
            },
        },
    },
]

# Canonical tool responses for the v2 patient (data/health_table_v2.yaml).
# Wordforms per plan §5.2; derivation will be centralized in
# `src/gemma_tools/dispenser_demo/wordform.py` in Phase 1.2.
PROFILE_RESP = {
    "name": "David Smith",
    "age": 45,
    "age_words": "forty five",
    "sex": "Male",
    "diagnoses": ["Type 2 Diabetes", "Hypertension"],
    "diagnoses_words": "Type Two Diabetes and Hypertension",
}
APPOINTMENT_RESP = {
    "date": "2026-05-20",
    "date_words": "May twentieth, twenty twenty six",
    "time": "10:30",
    "time_words": "ten thirty",
    "provider": "Dr. Evelyn Chen",
    "purpose": "quarterly diabetes check-up",
    "location": "Maple Clinic, Room 204",
    "location_words": "Maple Clinic, Room two hundred four",
}
CONTACT_RESP = {
    "name": "Jane Doe",
    "relation": "daughter",
    "phone": "+1-555-0142",
    "phone_words": "plus one five five five zero one four two",
}
DISPENSE_OK_RESP = {"status": "dispensed"}
DISPENSE_FAIL_RESP = {"status": "ble_not_connected"}
# The refusal tool is a no-op acknowledgement; the canned NL is what reaches
# the user. We echo `status: "refused"` for symmetry with dispense_medication.
REFUSAL_TOOL_RESP = {"status": "refused"}

# Canned NL strings — fixed by plan §5.3 / §6.1.
CANNED_DISPENSE_OK = "Your medication is being dispensed. Please check the dispenser."
CANNED_DISPENSE_FAIL = "I cannot reach the dispenser right now."
REFUSAL_NL = (
    "I can only help with your patient profile, appointments, emergency "
    "contact, or dispensing medication."
)


def _tool_calling_row(
    row_id: str,
    category: str,
    user_text: str,
    tool_name: str,
    tool_arguments: dict,
    tool_response: dict,
    pre_think: str,
    post_think: str,
    final_nl: str,
) -> dict:
    """Five-message row: system, user, assistant(tool_call), tool, assistant(NL)."""
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM_TRIGGER},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": f"<think>{pre_think}</think>",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "name": tool_name,
                "tool_call_id": "call_1",
                "content": json.dumps(tool_response, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": f"<think>{post_think}</think>\n{final_nl}",
            },
        ],
        "tools": TOOLS,
    }


PROFILE_ROWS = [
    ("pp-001", "Who am I?",
     "Self-profile question; call get_patient_profile.",
     "Name and basic profile.",
     "Your name is David Smith. You are forty five, Male, with Type Two Diabetes and Hypertension."),
    ("pp-002", "What's on my chart?",
     "Profile summary; call get_patient_profile.",
     "Full profile readback.",
     "Your chart shows David Smith, forty five, Male, diagnosed with Type Two Diabetes and Hypertension."),
    ("pp-003", "How old am I?",
     "Age question; the profile tool returns age plus age_words.",
     "age_words is forty five.",
     "You are forty five."),
    ("pp-004", "What are my conditions?",
     "Diagnoses question; call get_patient_profile.",
     "Quote diagnoses_words verbatim.",
     "Your conditions are Type Two Diabetes and Hypertension."),
    ("pp-005", "Tell me about myself.",
     "Open self-summary; profile tool.",
     "Full readback.",
     "You are David Smith, forty five, Male, with Type Two Diabetes and Hypertension."),
    ("pp-006", "What's my diagnosis?",
     "Diagnoses lookup; call get_patient_profile.",
     "diagnoses_words gives the digit-free form.",
     "You are diagnosed with Type Two Diabetes and Hypertension."),
    ("pp-007", "Summarize my profile.",
     "Summarization over profile; call get_patient_profile.",
     "All four fields.",
     "David Smith, forty five, Male. Diagnoses: Type Two Diabetes and Hypertension."),
    ("pp-008", "What does my record say about me?",
     "Self-profile readback; call get_patient_profile.",
     "Quote the profile back.",
     "Your record lists David Smith, age forty five, Male, with Type Two Diabetes and Hypertension."),
]

APPOINTMENT_ROWS = [
    ("na-001", "When's my next appointment?",
     "Appointment lookup; call get_next_appointment.",
     "Quote date_words and time_words.",
     "Your next appointment is on May twentieth, twenty twenty six at ten thirty."),
    ("na-002", "What's coming up on my schedule?",
     "Upcoming visit; call get_next_appointment.",
     "Date plus provider plus purpose.",
     "You have a quarterly diabetes check-up with Dr. Evelyn Chen on May twentieth, twenty twenty six."),
    ("na-003", "When do I see Dr. Chen?",
     "Appointment with named provider; the tool returns the only upcoming appointment.",
     "Quote date_words and time_words.",
     "You see Dr. Evelyn Chen on May twentieth, twenty twenty six at ten thirty."),
    ("na-004", "Do I have a doctor's visit coming up?",
     "Confirmation plus details; call get_next_appointment.",
     "Confirm yes plus date_words.",
     "Yes, you have a check-up on May twentieth, twenty twenty six at ten thirty."),
    ("na-005", "What's my next visit?",
     "Appointment lookup; call get_next_appointment.",
     "Purpose plus provider plus date_words.",
     "Your next visit is a quarterly diabetes check-up with Dr. Evelyn Chen on May twentieth, twenty twenty six."),
    ("na-006", "What time is my appointment?",
     "Time-only question; call get_next_appointment.",
     "Quote time_words.",
     "Your appointment is at ten thirty."),
    ("na-007", "Where am I going for my check-up?",
     "Location question; call get_next_appointment.",
     "Quote location_words.",
     "You are going to Maple Clinic, Room two hundred four."),
    ("na-008", "When is my upcoming appointment and where?",
     "Date plus location; call get_next_appointment.",
     "Quote date_words, time_words, location_words.",
     "Your appointment is on May twentieth, twenty twenty six at ten thirty at Maple Clinic, Room two hundred four."),
]

CONTACT_ROWS = [
    ("ec-001", "Who's my emergency contact?",
     "Emergency contact lookup; call get_emergency_contact.",
     "Name plus relation.",
     "Your emergency contact is Jane Doe, your daughter."),
    ("ec-002", "Who do I call in an emergency?",
     "Emergency contact lookup; call get_emergency_contact.",
     "Name plus phone_words.",
     "Call your daughter, Jane Doe, at plus one five five five zero one four two."),
    ("ec-003", "What's my emergency number?",
     "Phone-number question; call get_emergency_contact.",
     "Quote phone_words.",
     "Your emergency number is plus one five five five zero one four two."),
    ("ec-004", "Who should I contact if something happens?",
     "Emergency contact lookup; call get_emergency_contact.",
     "Name plus relation.",
     "Contact your daughter, Jane Doe."),
    ("ec-005", "Who is listed as my emergency contact?",
     "Emergency contact lookup; call get_emergency_contact.",
     "Name plus relation.",
     "Jane Doe, your daughter, is listed as your emergency contact."),
    ("ec-006", "What's the phone number for my emergency contact?",
     "Phone-number question; call get_emergency_contact.",
     "Quote phone_words.",
     "Jane Doe's number is plus one five five five zero one four two."),
    ("ec-007", "Can you give me my daughter's number?",
     "Phone lookup via relation; call get_emergency_contact.",
     "Relation matches; quote phone_words.",
     "Your daughter Jane Doe's number is plus one five five five zero one four two."),
    ("ec-008", "If I have an emergency, who do I reach?",
     "Emergency contact lookup; call get_emergency_contact.",
     "Name plus relation plus phone_words.",
     "Reach your daughter, Jane Doe, at plus one five five five zero one four two."),
]

DISPENSE_ROWS = [
    ("di-001", "Dispense my pill.",
     "Direct dispense intent; call dispense_medication.",
     "Status is dispensed; quote canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-002", "Give me my medication.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-003", "Time for my pills.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-004", "Drop my meds.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-005", "Release my medication.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-006", "Activate the dispenser.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    ("di-007", "It's time for my medicine.",
     "Dispense intent; call dispense_medication.",
     "Dispensed; canned NL.",
     CANNED_DISPENSE_OK, DISPENSE_OK_RESP),
    # di-008 — BLE failure path. The ONE non-happy dispense row.
    ("di-008", "Dispense my evening pill.",
     "Dispense intent; call dispense_medication.",
     "Status is ble_not_connected; quote fallback NL.",
     CANNED_DISPENSE_FAIL, DISPENSE_FAIL_RESP),
]

# 5 health_advice + 5 off_topic. Symmetric per-reason coverage after the
# 2026-05-11 rebalance (advisor flagged the prior 3/5 split as thin training
# signal on health_advice). Reason maps to the `refuse_out_of_scope.reason`
# enum; the runtime acknowledges with `{"status": "refused"}` regardless of
# reason and the canned NL is the same constant — the enum exists for
# offline analytics / per-cluster eval, not for branching the user-facing reply.
REFUSAL_ROWS = [
    ("oo-001", "Should I take an aspirin for my headache?", "health_advice",
     "Medication-advice request; call refuse_out_of_scope with reason='health_advice'.",
     "Refused; quote canned refusal NL."),
    ("oo-002", "Why does my chest hurt?", "health_advice",
     "Symptom diagnosis; call refuse_out_of_scope with reason='health_advice'.",
     "Refused; quote canned refusal NL."),
    ("oo-003", "Should I increase my diabetes medication?", "health_advice",
     "Treatment-plan question; call refuse_out_of_scope with reason='health_advice'.",
     "Refused; quote canned refusal NL."),
    ("oo-004", "Is it safe to mix ibuprofen with my regular meds?", "health_advice",
     "Drug-interaction question; call refuse_out_of_scope with reason='health_advice'.",
     "Refused; quote canned refusal NL."),
    ("oo-005", "How much aspirin should I take for back pain?", "health_advice",
     "Dosage-advice question; call refuse_out_of_scope with reason='health_advice'.",
     "Refused; quote canned refusal NL."),
    ("oo-006", "What's the weather like today?", "off_topic",
     "Off-topic (weather); call refuse_out_of_scope with reason='off_topic'.",
     "Refused; quote canned refusal NL."),
    ("oo-007", "What's in the news?", "off_topic",
     "Off-topic (news); call refuse_out_of_scope with reason='off_topic'.",
     "Refused; quote canned refusal NL."),
    ("oo-008", "Tell me a joke.", "off_topic",
     "Off-topic (entertainment); call refuse_out_of_scope with reason='off_topic'.",
     "Refused; quote canned refusal NL."),
    ("oo-009", "What's two plus two?", "off_topic",
     "Off-topic (arithmetic); call refuse_out_of_scope with reason='off_topic'.",
     "Refused; quote canned refusal NL."),
    ("oo-010", "What's your favorite color?", "off_topic",
     "Off-topic (generic personal); call refuse_out_of_scope with reason='off_topic'.",
     "Refused; quote canned refusal NL."),
]


def build_all() -> list[dict]:
    rows: list[dict] = []
    for rid, q, pre, post, nl in PROFILE_ROWS:
        rows.append(_tool_calling_row(rid, "patient_profile", q,
                                      "get_patient_profile", {}, PROFILE_RESP,
                                      pre, post, nl))
    for rid, q, pre, post, nl in APPOINTMENT_ROWS:
        rows.append(_tool_calling_row(rid, "next_appointment", q,
                                      "get_next_appointment", {}, APPOINTMENT_RESP,
                                      pre, post, nl))
    for rid, q, pre, post, nl in CONTACT_ROWS:
        rows.append(_tool_calling_row(rid, "emergency_contact", q,
                                      "get_emergency_contact", {}, CONTACT_RESP,
                                      pre, post, nl))
    for rid, q, pre, post, nl, resp in DISPENSE_ROWS:
        rows.append(_tool_calling_row(rid, "dispense", q,
                                      "dispense_medication", {}, resp,
                                      pre, post, nl))
    for rid, q, reason, pre, post in REFUSAL_ROWS:
        rows.append(_tool_calling_row(rid, "out_of_scope_refusal", q,
                                      "refuse_out_of_scope",
                                      {"reason": reason},
                                      REFUSAL_TOOL_RESP,
                                      pre, post, REFUSAL_NL))
    return rows


def main() -> None:
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_all()
    assert len(rows) == 42, f"expected 42 rows, got {len(rows)}"
    # Compact JSONL: one object per line, UTF-8, single trailing newline. Matches
    # the FG iter-001 file convention so the same loader / validator works
    # without seam-specific tweaks.
    text = "\n".join(
        json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows
    ) + "\n"
    _OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {len(rows)} rows to {_OUT_PATH.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
