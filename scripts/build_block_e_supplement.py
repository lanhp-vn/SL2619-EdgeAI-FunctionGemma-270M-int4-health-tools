#!/usr/bin/env python3
"""Deterministic generator for the Block E supplement (370 rows).

Source: `docs/bench/2026-05-01_functiongemma-eval-deepdive.md` Block E §16
mandates +160 refusal rows + broader argument-value vocabulary
(≥25 foods, ≥25 times, ≥30 medication names) and a 370-row supplement
covering all seven categories with id ranges:

    ot-501..ot-580  off_topic_refusal       (80)
    ma-501..ma-580  medical_advice_refusal  (80)
    fl-501..fl-560  fact_lookup             (60)
    fa-501..fa-530  fact_absence            (30)
    te-501..te-540  tool_error_recovery     (40)
    tt-501..tt-540  two_turn                (40)
    pc-501..pc-540  parallel_call           (40)

Why a generator rather than a hand-edit of the prior `supplement_dataset.jsonl`:
the prior file had 740 rows = every target id duplicated, with pervasive
structural defects (`function.arguments` as JSON strings instead of dicts,
tool messages missing `name`, literal `<answer>` tags, ad-hoc tool-response
shapes inconsistent with the registry). The salvage ratio was too low to
justify per-row triage; regenerating from scratch with the validator gates
in-line is shorter and produces a corpus that round-trips through
`scripts/functiongemma_ingest.py`.

Output: `data/functiongemma/_incoming/batch_004_block_e_supplement_repaired.jsonl`.
The script enforces in-process: validator at 1.0, no duplicate ids, no
duplicate user prompts, no shared first-4-word prefix within a category,
no literal `<answer>`/`<bos>`/`<start_of_turn>` tokens, all tool-call
arguments are dicts, all tool messages carry `name`, and the registry
vocabulary minima for the three open-string slots.

Usage:
    uv run python scripts/build_block_e_supplement.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gemma_tools.functiongemma_dataset import (
    SYSTEM_TRIGGER,
    validate_conversation,
)
from gemma_tools.functiongemma_tools import as_function_declarations

_REPO = Path(__file__).resolve().parents[1]
_OUTPUT = (
    _REPO
    / "data"
    / "functiongemma"
    / "_incoming"
    / "batch_004_block_e_supplement_repaired.jsonl"
)

# Snapshot the canonical 7-tool registry once. Every row carries this exact
# list so the declared schema can never drift from the registry the validator
# checks against.
TOOLS = as_function_declarations()


# ==========================================================================
# region argument vocabularies — ≥25 / ≥25 / ≥30 (Block E mandate)
# ==========================================================================

# 35 medication names (real, common formulary entries). The validator only
# requires `name` be a non-empty string; the registry lookup is permissive,
# so values not present in `data/health_table_v1.yaml` are still legal — they
# exercise the no_match branch which the model needs to learn anyway.
MEDS_REAL = (
    "Atorvastatin", "Metoprolol", "Amlodipine", "Omeprazole", "Losartan",
    "Lisinopril", "Metformin", "Levothyroxine", "Simvastatin", "Sertraline",
    "Albuterol", "Gabapentin", "Hydrochlorothiazide", "Citalopram", "Pantoprazole",
    "Furosemide", "Escitalopram", "Tramadol", "Trazodone", "Fluoxetine",
    "Bupropion", "Carvedilol", "Clopidogrel", "Diltiazem", "Duloxetine",
    "Insulin Glargine", "Montelukast", "Pravastatin", "Rosuvastatin", "Tamsulosin",
    "Warfarin", "Aspirin", "Ibuprofen", "Acetaminophen", "Vitamin D3",
)

# 8 short prefixes used to deliberately trigger the ambiguous-match branch
# of `get_medication_by_name`. These are real prefixes of multiple real meds
# (e.g. "atro" prefixes Atorvastatin), not arbitrary letters.
MED_PREFIXES = ("a", "at", "lis", "me", "atr", "om", "ros", "tra")

# 30 24h times spanning the day, with non-quarter-hour values mixed in so the
# slot is understood as "a clock time" not "one of these 7 strings" (cf. the
# M6 schema-leak failure in `2026-05-01_functiongemma-dataset-audit.md` §D3).
TIMES = (
    "00:00", "01:30", "02:00", "03:15", "04:00", "05:30", "06:00", "06:45",
    "07:00", "07:30", "08:00", "08:30", "09:00", "09:45", "10:30", "11:00",
    "12:00", "12:30", "13:00", "13:45", "14:30", "15:00", "16:15", "17:00",
    "17:45", "18:00", "19:00", "19:30", "20:30", "21:00", "21:45", "22:00",
    "22:30", "23:00", "23:45",
)

# 30 foods with realistic drug-interaction relevance: warfarin/leafy greens,
# MAOI/tyramine, ACEI/potassium, statins/grapefruit, antibiotics/dairy.
FOODS = (
    "grapefruit", "grapefruit juice", "alcohol", "shellfish", "spinach",
    "kale", "broccoli", "bananas", "tofu", "soy milk",
    "yogurt", "cheese", "salmon", "tuna", "raw eggs",
    "licorice", "coffee", "green tea", "black tea", "kombucha",
    "aged cheddar", "miso paste", "smoked salmon", "cured ham", "almonds",
    "walnuts", "kiwi", "cranberry juice", "pomegranate", "ginger",
    "garlic", "St John's wort", "Brussels sprouts", "cabbage", "avocado",
)

# endregion


# ==========================================================================
# region fixture tool responses — match the registry handler return shapes
# ==========================================================================
# Why this matters: training will tokenize these strings as ground-truth
# context for the next assistant turn. Ad-hoc shapes ({"i": false}, etc.)
# teach the model nonsense to copy at inference time. Each shape below is
# isomorphic to the corresponding `_med_to_dict` / `_get_vitals` etc. in
# `src/gemma_tools/functiongemma_tools.py`.

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

_NEXT_APPOINTMENT = {
    "date": "2026-05-06",
    "time": "10:30",
    "provider": "Dr. Evelyn Chen",
    "purpose": "quarterly diabetes check-up",
    "location": "Maple Clinic, Room 204",
}

_EMERGENCY_CONTACT = {
    "name": "Sarah Park",
    "relation": "daughter",
    "phone": "555-0148",
}


def _med_record(name: str, dose: str, schedule: str, with_food: bool,
                purpose: str, avoid_foods: list[str], avoid_drugs: list[str]) -> dict:
    return {
        "name": name,
        "dose": dose,
        "schedule": schedule,
        "with_food": with_food,
        "purpose": purpose,
        "avoid_foods": avoid_foods,
        "avoid_drugs": avoid_drugs,
    }


# Realistic look-up records keyed by name. Anything not in this dict gets a
# generic record (when used in fact_lookup) or is treated as no_match (when
# used in tool_error_recovery).
_MED_RECORDS: dict[str, dict] = {
    "Lisinopril": _med_record("Lisinopril", "10 mg", "08:00", False,
                              "blood pressure", ["bananas"], []),
    "Metformin": _med_record("Metformin", "500 mg", "08:00, 19:00", True,
                             "type 2 diabetes", [], []),
    "Atorvastatin": _med_record("Atorvastatin", "20 mg", "21:00", False,
                                "cholesterol", ["grapefruit juice"], []),
    "Levothyroxine": _med_record("Levothyroxine", "50 mcg", "06:30", False,
                                 "hypothyroidism", ["soy milk"], []),
    "Amlodipine": _med_record("Amlodipine", "5 mg", "08:00", False,
                              "blood pressure", ["grapefruit"], []),
    "Omeprazole": _med_record("Omeprazole", "20 mg", "07:00", False,
                              "GERD", [], []),
    "Sertraline": _med_record("Sertraline", "50 mg", "20:00", True,
                              "depression", ["aged cheddar"], []),
    "Albuterol": _med_record("Albuterol", "90 mcg/spray", "as needed", False,
                             "asthma rescue", [], []),
    "Aspirin": _med_record("Aspirin", "81 mg", "08:00", True,
                           "cardioprotection", [], ["Warfarin"]),
    "Vitamin D3": _med_record("Vitamin D3", "2000 IU", "12:00", True,
                              "supplement", [], []),
}


def _generic_med(name: str, dose: str = "10 mg", time: str = "08:00") -> dict:
    return _med_record(name.title(), dose, time, False, "ongoing therapy", [], [])


def _meds_at(time: str) -> list[dict]:
    """Two-medication response with `schedule` containing `time`."""
    return [
        _med_record("Metformin", "500 mg", "08:00, 19:00", True,
                    "type 2 diabetes", [], []),
        _med_record("Lisinopril", "10 mg", time if time == "08:00" else "08:00", False,
                    "blood pressure", ["bananas"], []),
    ]


def _meds_at_empty() -> list[dict]:
    return []


def _food_response(food: str, interacts: bool, with_meds: list[str],
                   rule: str | None = None) -> dict:
    return {"food": food, "interacts": interacts, "with_meds": with_meds, "rule": rule}


# endregion


# ==========================================================================
# region prompt pools — paraphrastically novel; first-4-word prefixes unique
# ==========================================================================

# 80 off-topic prompts. Verified post-hoc by `_check_first4_unique` in main.
OT_PROMPTS = (
    "Recommend a thriller novel for tonight.",
    "Plan my Tokyo vacation itinerary please.",
    "Convert five miles into kilometers exactly.",
    "Summarize today's global news headlines please.",
    "Brief me on the recent election results.",
    "Translate good morning into Spanish.",
    "Spell the German phrase ich liebe dich.",
    "Pronounce 'bonjour' for me clearly.",
    "Calculate forty-seven times thirteen quickly.",
    "Solve the integral of x squared dx.",
    "Round 7.853 to two decimal places.",
    "Help me with long division steps.",
    "Define photosynthesis using simple language.",
    "Explain quantum entanglement very briefly.",
    "Give me a riddle to solve.",
    "Share a random fact about octopuses.",
    "Why is the sky blue?",
    "When did humans first land on the moon?",
    "Who painted the famous Mona Lisa?",
    "List ten Roman emperors briefly.",
    "Origins of the printing press?",
    "Briefly outline World War One causes.",
    "Best brownie recipe you know?",
    "Recipe for tomato bisque soup?",
    "Make pasta carbonara step by step.",
    "Quick breakfast smoothie ideas please.",
    "Sourdough starter steps for beginners.",
    "Debug this Python snippet for me.",
    "Explain quicksort algorithm step by step.",
    "Set up a React project quickly.",
    "Fix the regex pattern I'm using.",
    "Compare SQL versus NoSQL databases briefly.",
    "Should I adopt a kitten today?",
    "Tips for falling asleep faster please.",
    "Best workout routine for abs?",
    "How do I learn guitar quickly?",
    "Plan a morning meditation routine.",
    "Where can I buy sneakers cheaply?",
    "iPhone or Samsung — which is better?",
    "Suggest a gift for my sister.",
    "Coupons for grocery delivery this week?",
    "Best laptop for college students?",
    "Pick a movie for tonight please.",
    "Suggest a song for studying.",
    "Pick a sitcom to watch tonight.",
    "Find me a chill playlist online.",
    "What movie is trending right now?",
    "Tell me a knock-knock joke.",
    "Crack a doctor pun for me.",
    "Share a funny one-liner please.",
    "Make me laugh with humor.",
    "Recite a clever pun aloud.",
    "Joke about cats and dogs?",
    "Sing me happy birthday lyrics.",
    "Set a timer for ten minutes.",
    "Add a haircut to my calendar.",
    "Who won the football game last night?",
    "How is my favorite team doing?",
    "Give me current NBA scores.",
    "Update on the Olympics medal table?",
    "Forecast for tomorrow morning weather?",
    "Will it rain this weekend?",
    "Today's weather conditions outside, please.",
    "Is a storm coming this week?",
    "Ten-day weather outlook for here?",
    "What is the latest tech news?",
    "Headlines from sports today, please.",
    "Did Hemingway write Old Man and the Sea?",
    "Top fantasy book series of 2024?",
    "Suggest a romance audiobook to enjoy.",
    "Summary of Orwell's 1984 plot?",
    "How far is Paris by train?",
    "Best beaches in Thailand for swimming?",
    "Find cheap flights to Berlin tomorrow.",
    "Book a hotel in central Rome.",
    "Translate goodbye into Japanese for me.",
    "Recommend a hiking trail nearby.",
    "Best way to brew espresso?",
    "Random trivia about pandas, please.",
    "Who wrote the U.S. Constitution?",
)

OT_REFUSALS = (
    "I answer questions from your health record only.",
    "I can only help with information in your medical record.",
    "That request falls outside my health-record scope.",
    "I'm limited to questions about your medical record.",
    "Only health-record queries are within my scope.",
    "I can only assist with your health information.",
    "My role is to surface facts from your record; that's outside it.",
    "That topic isn't in my scope; I handle health-record lookups only.",
    "I can't help with that — my answers come from your medical record.",
    "Sorry, I'm scoped to your health record and nothing else.",
    "That's not something I can help with; I work from your record only.",
    "I'm here for your health record. That request is outside my role.",
)


# 80 medical-advice prompts.
MA_PROMPTS = (
    "Should I increase my Lisinopril dose tonight?",
    "Can I stop taking Metformin if I feel fine?",
    "Is it safe to skip my evening Atorvastatin?",
    "Will doubling Levothyroxine speed up weight loss?",
    "Do I need stronger medication for my blood pressure?",
    "Could I switch from Sertraline to St John's wort?",
    "Should my Albuterol inhaler be used more often?",
    "Would taking ibuprofen with Aspirin be okay?",
    "Is melatonin safe alongside my current meds?",
    "Can I cut my Amlodipine pill in half?",
    "How long must I stay on Omeprazole?",
    "Is it dangerous to mix alcohol with my prescriptions?",
    "Do I really need this statin therapy?",
    "Should I see a specialist sooner than scheduled?",
    "Would extra vitamin D fix my fatigue?",
    "Can I drink coffee while on my BP meds?",
    "What dosage adjustment do you recommend for me?",
    "Does my chest pain mean I should panic?",
    "Are these symptoms serious enough for the ER?",
    "Could I have an allergy I don't know about?",
    "Why do I feel dizzy after my morning dose?",
    "My headache won't quit — what should I take?",
    "Tell me whether to start a low-carb diet.",
    "Recommend an over-the-counter sleep aid for me.",
    "Prescribe something for my chronic back pain.",
    "Diagnose this rash on my forearm please.",
    "Is my fever too high to ignore?",
    "Could the pill I missed cause withdrawal?",
    "Need advice — should I taper off Sertraline?",
    "Help me decide on hormone replacement therapy.",
    "Am I a candidate for bariatric surgery?",
    "Will turmeric help my joint inflammation more?",
    "Replace my prescription with an herbal alternative?",
    "Walk me through stopping my Tramadol safely.",
    "How risky is it to skip statins for a week?",
    "Whether I can take ibuprofen with my meds?",
    "If I exercise more, can I drop my BP pill?",
    "Doctor cancelled — what should I do for pain?",
    "Suggest a safe dose for my insomnia tonight.",
    "Reduce my insulin if my sugar runs low?",
    "Recommend whether I should fast before lab work.",
    "Pregnant friend's question: which OTC is safe?",
    "Is my LDL number something to worry about?",
    "Are my A1c levels concerning to you?",
    "Cholesterol crept up — should I panic?",
    "Heart palpitations after coffee — dangerous?",
    "Lump on my neck — cancer worry?",
    "My BP reading was 160/100 — emergency?",
    "Bruising easily — should I drop the aspirin?",
    "Stomach pain after Metformin — bad sign?",
    "Tingling fingers — neuropathy starting?",
    "Persistent cough — antibiotics needed?",
    "Eye twitching — magnesium deficiency?",
    "Joint stiffness — early arthritis?",
    "Memory lapses — early dementia?",
    "Mood swings — bipolar?",
    "Constant thirst — diabetes worsening?",
    "Night sweats — menopause or infection?",
    "Hair loss — thyroid issue?",
    "Nausea every morning — serious?",
    "Frequent urination — should I be worried?",
    "Dry mouth — medication side effect?",
    "Weight gain on Sertraline — switch drugs?",
    "Cold hands and feet — circulation problem?",
    "Random bruises — clotting disorder?",
    "Stiff neck and fever — meningitis?",
    "Sudden vision blur — stroke warning?",
    "Numbness in my left arm — heart issue?",
    "Sharp chest twinges — anxiety or worse?",
    "Knee swelling — should I drain it?",
    "Acid reflux nightly — surgery option?",
    "Erectile dysfunction — testosterone tests?",
    "Restless legs at night — iron supplements?",
    "Frequent migraines — Botox indicated?",
    "Itchy scalp — anti-fungal needed?",
    "Foot pain — plantar fasciitis exercises?",
    "Tinnitus worsening — hearing-aid candidate?",
    "Depression returning — medication change?",
    "Flu shot today — safe with my meds?",
    "Travelling overseas — which vaccines first?",
)

MA_REFUSALS = (
    "I cannot give medical advice. Please consult your clinician about that.",
    "Medical advice is outside my scope; please ask your clinician.",
    "That decision needs your clinician's input — I can't advise on it.",
    "Please consult your prescribing doctor; I don't make medical decisions.",
    "I'm not able to recommend medication changes; ask your clinician.",
    "That requires a clinician's judgment — I can only surface record facts.",
    "Please contact your provider for that recommendation.",
    "Reach out to your medical team for advice on that.",
    "I'll defer that to your clinician; medical advice is outside my role.",
    "Talk with your doctor about that — I don't give medical advice.",
    "Your clinician should weigh in on that decision.",
    "I can't recommend changes to your treatment; please call your provider.",
)


# endregion


# ==========================================================================
# region row builders — one per category
# ==========================================================================


def _system_msg() -> dict[str, str]:
    return {"role": "system", "content": SYSTEM_TRIGGER}


def _row(rid: str, category: str, messages: list[dict]) -> dict:
    return {
        "id": rid,
        "category": category,
        "messages": messages,
        "tools": TOOLS,
    }


def _think_call(reason: str) -> str:
    return f"<think>{reason}</think>"


def _think_answer(reason: str, answer: str) -> str:
    return f"<think>{reason}</think>\n{answer}"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_result(call_id: str, name: str, payload: Any) -> dict:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def gen_off_topic() -> list[dict]:
    rows: list[dict] = []
    for i, prompt in enumerate(OT_PROMPTS):
        refusal = OT_REFUSALS[i % len(OT_REFUSALS)]
        # Vary the <think> reason slightly so the model can't pattern-match a
        # single literal string. The validator only requires exactly one
        # <think> block of any content.
        reasons = (
            "Off-topic; refuse without tool call.",
            "Outside scope; do not call any tool.",
            "Not a health-record question; refuse.",
            "Off-topic request; respond with refusal.",
        )
        reason = reasons[i % len(reasons)]
        rows.append(_row(
            f"ot-{501 + i}",
            "off_topic_refusal",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _think_answer(reason, refusal)},
            ],
        ))
    return rows


def gen_medical_advice() -> list[dict]:
    rows: list[dict] = []
    reasons = (
        "Medical advice request; refuse without calling any tool.",
        "Clinical decision request; defer to clinician.",
        "Treatment-recommendation question; refuse and refer.",
        "Diagnostic or prescriptive question; outside my role.",
    )
    for i, prompt in enumerate(MA_PROMPTS):
        refusal = MA_REFUSALS[i % len(MA_REFUSALS)]
        reason = reasons[i % len(reasons)]
        rows.append(_row(
            f"ma-{501 + i}",
            "medical_advice_refusal",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": _think_answer(reason, refusal)},
            ],
        ))
    return rows


def gen_fact_lookup() -> list[dict]:
    rows: list[dict] = []
    rid = 501

    # 15 rows: get_medication_by_name with 15 distinct meds and 15 distinct
    # prompt templates (cycling templates collides on first-4-word prefix when
    # {name} lands past position 4).
    fl_med_pairs = [
        ("Lisinopril",
         "What dose is {name} prescribed at?", "Look up {name} dose.",
         "Your {name} dose is {dose}, scheduled at {sched}."),
        ("Metformin",
         "Pull up the schedule for my {name}.", "Look up {name} schedule.",
         "{name} is scheduled at {sched}."),
        ("Atorvastatin",
         "Is {name} taken with food?", "Check whether {name} is with food.",
         "{name} is {food_str} food."),
        ("Levothyroxine",
         "Why was {name} prescribed?", "Look up {name} purpose.",
         "{name} treats {purpose}."),
        ("Amlodipine",
         "Read out my {name} prescription details.", "Look up {name}.",
         "Your {name} dose is {dose}."),
        ("Omeprazole",
         "Tell me about my {name} dose.", "Pull up {name} record.",
         "{name} is dosed at {dose}, scheduled at {sched}."),
        ("Sertraline",
         "Cite the dose of {name}, please.", "Get {name} from record.",
         "Per record, {name} is {dose} at {sched}."),
        ("Albuterol",
         "Verify the schedule of {name}.", "Surface {name} schedule.",
         "{name} schedule on file: {sched}."),
        ("Aspirin",
         "Confirm dosage on my {name}.", "Lookup {name} dose.",
         "Confirmed: {name} {dose}."),
        ("Vitamin D3",
         "Provide {name} schedule, please.", "Get {name} schedule.",
         "{name} is taken at {sched}."),
        ("Losartan",
         "Surface my {name} record now.", "Look up {name} record.",
         "{name} {dose} at {sched}, treats {purpose}."),
        ("Simvastatin",
         "Spell out {name} dosing for me.", "Get {name} dosing.",
         "{name} dosing: {dose} at {sched}."),
        ("Pantoprazole",
         "Open the {name} entry, please.", "Open {name} entry.",
         "{name}: {dose} at {sched}."),
        ("Furosemide",
         "Detail my {name} prescription, please.", "Pull {name} prescription.",
         "{name} prescription: {dose} at {sched}."),
        ("Citalopram",
         "Bring up {name} from my record.", "Surface {name} record.",
         "{name}: {dose} at {sched}, for {purpose}."),
    ]
    for name, prompt_t, reason_t, ans_t in fl_med_pairs:
        rec = _MED_RECORDS.get(name) or _generic_med(name)
        food_str = "taken with" if rec["with_food"] else "taken without"
        ans = ans_t.format(
            name=rec["name"],
            dose=rec["dose"],
            sched=rec["schedule"],
            purpose=rec["purpose"],
            food_str=food_str,
        )
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt_t.format(name=name)},
                {"role": "assistant",
                 "content": _think_call(reason_t.format(name=name)),
                 "tool_calls": [_tool_call("call_1", "get_medication_by_name",
                                           {"name": name})]},
                _tool_result("call_1", "get_medication_by_name", rec),
                {"role": "assistant",
                 "content": _think_answer("Surface dose/schedule from record.", ans)},
            ],
        ))
        rid += 1

    # 12 rows: get_medications_at_time with 12 distinct times and distinct
    # prompt templates per row.
    fl_time_pairs = [
        ("06:00", "Which pills come at {t}?",
         "Lookup meds scheduled at {t}."),
        ("07:30", "What is scheduled for {t}?",
         "Lookup {t} medications."),
        ("08:00", "Anything on the list at {t}?",
         "Check what's at {t}."),
        ("09:00", "Show me {t} medications, please.",
         "Look up {t}."),
        ("12:00", "Need the {t} medication list now.",
         "Pull up {t} list."),
        ("13:00", "Pull up meds for {t} please.",
         "Get {t} medications."),
        ("14:30", "Read out the {t} pill list.",
         "Surface {t} schedule."),
        ("16:15", "Cite my {t} schedule, please.",
         "Cite {t} schedule."),
        ("19:00", "Provide {t} dosing schedule.",
         "Provide {t} schedule."),
        ("20:30", "Open my {t} medication slot.",
         "Open {t} slot."),
        ("21:00", "Surface meds at {t}, please.",
         "Surface {t} meds."),
        ("22:00", "List medications at {t} now.",
         "List {t} meds."),
    ]
    fl_time_ans_t = "At {t}, the record lists {meds}."
    for i, (t, prompt_t, reason_t) in enumerate(fl_time_pairs):
        ans_t = fl_time_ans_t
        # Roughly half return empty, half non-empty. Vary realistically.
        if i % 3 == 0:
            payload = _meds_at_empty()
            meds_str = "no scheduled medications"
        else:
            # Build a 1-2 med list with this time in the schedule.
            payload = [_med_record(
                "Lisinopril", "10 mg", t, False,
                "blood pressure", ["bananas"], [])]
            meds_str = "Lisinopril 10 mg"
        ans = ans_t.format(t=t, meds=meds_str)
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt_t.format(t=t)},
                {"role": "assistant",
                 "content": _think_call(reason_t.format(t=t)),
                 "tool_calls": [_tool_call("call_1", "get_medications_at_time",
                                           {"time_24h": t})]},
                _tool_result("call_1", "get_medications_at_time", payload),
                {"role": "assistant",
                 "content": _think_answer("Report time-window meds.", ans)},
            ],
        ))
        rid += 1

    # 10 rows: check_food_interaction with 10 distinct foods.
    fl_food_specs = [
        ("Is {f} safe with my meds?", "Check food interaction for {f}.",
         "{f}: {summary}."),
        ("Can I have {f} today?", "Look up {f} interaction.",
         "{f} {summary}."),
        ("Will {f} interact with my prescriptions?", "Run interaction check on {f}.",
         "Result for {f}: {summary}."),
        ("Any issue eating {f}?", "Check {f} against meds.",
         "{f} interaction status: {summary}."),
    ]
    fl_foods_use = [
        "spinach", "salmon", "almonds", "kale", "yogurt",
        "cheese", "tofu", "ginger", "cabbage", "avocado",
    ]
    for i, food in enumerate(fl_foods_use):
        prompt_t, reason_t, ans_t = fl_food_specs[i % len(fl_food_specs)]
        # All these are benign for our patient. Validate that interacts=False
        # path is well-represented.
        payload = _food_response(food, False, [], None)
        summary = "no interaction with current medications"
        ans = ans_t.format(f=food, summary=summary)
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt_t.format(f=food)},
                {"role": "assistant",
                 "content": _think_call(reason_t.format(f=food)),
                 "tool_calls": [_tool_call("call_1", "check_food_interaction",
                                           {"food": food})]},
                _tool_result("call_1", "check_food_interaction", payload),
                {"role": "assistant",
                 "content": _think_answer("Summarize food interaction result.", ans)},
            ],
        ))
        rid += 1

    # 8 rows: get_vitals (no args).
    vitals_prompts = [
        ("What's my heart rate today?",
         "Check vitals for heart rate.",
         f"Your heart rate is {_VITALS['heart_rate_bpm']} bpm."),
        ("Tell me my latest blood pressure reading.",
         "Look up most recent BP.",
         f"Your blood pressure is {_VITALS['blood_pressure_systolic']}/"
         f"{_VITALS['blood_pressure_diastolic']} mmHg."),
        ("Pull up my SpO2 number.",
         "Get SpO2 from vitals.",
         f"Your SpO2 is {_VITALS['spo2_percent']}%."),
        ("How's my respiratory rate looking?",
         "Look up respiratory rate.",
         f"Your respiratory rate is {_VITALS['respiratory_rate']} breaths/min."),
        ("Body temperature reading on file?",
         "Pull body temp from vitals.",
         f"Your body temperature is {_VITALS['body_temperature_c']}°C."),
        ("Latest vitals snapshot, please.",
         "Read most recent vitals.",
         f"BP {_VITALS['blood_pressure_systolic']}/"
         f"{_VITALS['blood_pressure_diastolic']}, "
         f"HR {_VITALS['heart_rate_bpm']}, "
         f"SpO2 {_VITALS['spo2_percent']}%."),
        ("Heart-rate value from yesterday?",
         "Check vitals heart rate.",
         f"Latest heart rate on file: {_VITALS['heart_rate_bpm']} bpm "
         f"(measured {_VITALS['last_measured']})."),
        ("Was my BP high last reading?",
         "Look up most recent BP.",
         f"Last BP was {_VITALS['blood_pressure_systolic']}/"
         f"{_VITALS['blood_pressure_diastolic']} mmHg — within range."),
    ]
    for prompt, reason, ans in vitals_prompts:
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(reason),
                 "tool_calls": [_tool_call("call_1", "get_vitals", {})]},
                _tool_result("call_1", "get_vitals", _VITALS),
                {"role": "assistant",
                 "content": _think_answer("Surface vitals fact.", ans)},
            ],
        ))
        rid += 1

    # 5 rows: list_allergies (no args).
    allergy_prompts = [
        ("Do I have any allergies on file?",
         "List allergies.",
         "Yes: Penicillin (severe) and Shellfish (moderate)."),
        ("Run me through my allergies.",
         "List allergies for the user.",
         "Penicillin — severe (anaphylaxis); Shellfish — moderate (hives)."),
        ("Allergy list, please.",
         "Lookup allergies.",
         "Penicillin and Shellfish are listed."),
        ("Am I allergic to anything?",
         "Check allergy list.",
         "Yes — Penicillin (severe) and Shellfish (moderate) are on record."),
        ("Tell me my known allergens briefly.",
         "Surface allergy list.",
         "Two allergens: Penicillin (anaphylaxis) and Shellfish (hives)."),
    ]
    for prompt, reason, ans in allergy_prompts:
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(reason),
                 "tool_calls": [_tool_call("call_1", "list_allergies", {})]},
                _tool_result("call_1", "list_allergies", _ALLERGIES),
                {"role": "assistant",
                 "content": _think_answer("Read allergy list.", ans)},
            ],
        ))
        rid += 1

    # 6 rows: get_next_appointment (no args).
    appt_prompts = [
        ("When is my next appointment?",
         "Look up upcoming appointment.",
         "Next: 2026-05-06 10:30 with Dr. Evelyn Chen at Maple Clinic, Room 204."),
        ("Pull up my next visit, please.",
         "Get next appointment.",
         "Your next visit is on May 6, 2026 at 10:30 AM with Dr. Chen."),
        ("Remind me of the upcoming visit.",
         "Lookup next appointment.",
         "Upcoming visit: 2026-05-06 at 10:30, Maple Clinic, Room 204."),
        ("Doctor visit details on file?",
         "Get next appointment.",
         "Next appointment with Dr. Evelyn Chen on 2026-05-06 at 10:30."),
        ("Upcoming clinic time, please.",
         "Surface next appointment.",
         "May 6, 2026 at 10:30 AM, Maple Clinic Room 204."),
        ("Confirm my next checkup, please.",
         "Look up next appointment.",
         "Confirmed: 2026-05-06 10:30 with Dr. Evelyn Chen."),
    ]
    for prompt, reason, ans in appt_prompts:
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(reason),
                 "tool_calls": [_tool_call("call_1", "get_next_appointment", {})]},
                _tool_result("call_1", "get_next_appointment", _NEXT_APPOINTMENT),
                {"role": "assistant",
                 "content": _think_answer("Read appointment fact.", ans)},
            ],
        ))
        rid += 1

    # 4 rows: get_emergency_contact (no args).
    ec_prompts = [
        ("Who's my emergency contact?",
         "Get emergency contact.",
         "Sarah Park (daughter) — 555-0148."),
        ("Pull up my emergency contact info.",
         "Get emergency contact.",
         "Emergency contact: Sarah Park (daughter), phone 555-0148."),
        ("Emergency phone on file?",
         "Get emergency contact.",
         "On file: Sarah Park, your daughter, at 555-0148."),
        ("Family contact for emergencies, please.",
         "Look up emergency contact.",
         "Sarah Park, your daughter, can be reached at 555-0148."),
    ]
    for prompt, reason, ans in ec_prompts:
        rows.append(_row(
            f"fl-{rid}",
            "fact_lookup",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(reason),
                 "tool_calls": [_tool_call("call_1", "get_emergency_contact", {})]},
                _tool_result("call_1", "get_emergency_contact", _EMERGENCY_CONTACT),
                {"role": "assistant",
                 "content": _think_answer("Read emergency contact.", ans)},
            ],
        ))
        rid += 1

    assert len(rows) == 60, f"fact_lookup expected 60 rows, got {len(rows)}"
    return rows


def gen_fact_absence() -> list[dict]:
    rows: list[dict] = []
    rid = 501

    # 20 rows: get_vitals — user asks for a vital not in our 7 fields.
    vitals_absence = [
        ("What's my A1c level?", "A1c"),
        ("Cholesterol number on file?", "cholesterol"),
        ("LDL value, please.", "LDL"),
        ("HDL reading available?", "HDL"),
        ("Triglyceride number, please.", "triglyceride"),
        ("Blood glucose history?", "blood glucose"),
        ("Fasting sugar number?", "fasting glucose"),
        ("Hemoglobin number on record?", "hemoglobin"),
        ("Hematocrit on file?", "hematocrit"),
        ("White blood cell count?", "WBC count"),
        ("Tell me my blood type.", "blood type"),
        ("Vitamin D level?", "vitamin D level"),
        ("TSH value, please.", "TSH"),
        ("Free T4 reading?", "free T4"),
        ("Liver enzyme number?", "ALT/AST"),
        ("Creatinine on file?", "creatinine"),
        ("Potassium level recorded?", "potassium"),
        ("Sodium reading available?", "sodium"),
        ("Iron level, please.", "iron level"),
        ("Calcium number on file?", "calcium"),
    ]
    for prompt, missing_field in vitals_absence:
        rows.append(_row(
            f"fa-{rid}",
            "fact_absence",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(
                     f"{missing_field} might appear in vitals; check get_vitals."),
                 "tool_calls": [_tool_call("call_1", "get_vitals", {})]},
                _tool_result("call_1", "get_vitals", _VITALS),
                {"role": "assistant",
                 "content": _think_answer(
                     f"No {missing_field} field in vitals response.",
                     f"Your {missing_field} is not available in your "
                     "health record data.")},
            ],
        ))
        rid += 1

    # 10 rows: get_medication_by_name — record returned, but the asked-for
    # field (manufacturer, expiration, refills, prescriber, …) is not stored.
    med_absence = [
        ("Who's the manufacturer of my Metformin?",
         "Metformin", "manufacturer"),
        ("When does my Atorvastatin expire?",
         "Atorvastatin", "expiration date"),
        ("How many refills do I have on Lisinopril?",
         "Lisinopril", "refill count"),
        ("Which pharmacy filled my Levothyroxine?",
         "Levothyroxine", "pharmacy"),
        ("NDC number for my Omeprazole?",
         "Omeprazole", "NDC"),
        ("Generic substitute name for my Amlodipine?",
         "Amlodipine", "generic substitution name"),
        ("Lot number on my Sertraline?",
         "Sertraline", "lot number"),
        ("Color of my Aspirin tablets?",
         "Aspirin", "tablet color"),
        ("Coupon code for Vitamin D3?",
         "Vitamin D3", "coupon information"),
        ("Original prescriber for my Albuterol?",
         "Albuterol", "prescriber"),
    ]
    for prompt, name, missing_field in med_absence:
        rec = _MED_RECORDS.get(name) or _generic_med(name)
        rows.append(_row(
            f"fa-{rid}",
            "fact_absence",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(
                     f"{missing_field} might be on the medication record; "
                     f"call get_medication_by_name."),
                 "tool_calls": [_tool_call("call_1", "get_medication_by_name",
                                           {"name": name})]},
                _tool_result("call_1", "get_medication_by_name", rec),
                {"role": "assistant",
                 "content": _think_answer(
                     f"No {missing_field} field on medication record.",
                     f"The {missing_field} for {name} is not in your record.")},
            ],
        ))
        rid += 1

    assert len(rows) == 30, f"fact_absence expected 30 rows, got {len(rows)}"
    return rows


def gen_tool_error_recovery() -> list[dict]:
    rows: list[dict] = []
    rid = 501

    # 22 rows: no_match — patient asks about a med not in the record.
    # Use medication names not in _MED_RECORDS so the "you don't have a
    # prescription for X" answer is correct.
    no_match_meds = [
        "Hydrochlorothiazide", "Citalopram", "Pantoprazole", "Furosemide",
        "Escitalopram", "Tramadol", "Trazodone", "Fluoxetine",
        "Bupropion", "Carvedilol", "Clopidogrel", "Diltiazem",
        "Duloxetine", "Insulin Glargine", "Montelukast", "Pravastatin",
        "Rosuvastatin", "Tamsulosin", "Warfarin", "Ibuprofen",
        "Acetaminophen", "Gabapentin",
    ]
    no_match_prompts = [
        "Do I have a prescription for {name}?",
        "Is {name} on my medication list?",
        "Show me my {name} prescription.",
        "Pull up {name} from my record.",
        "What's the dose of my {name}?",
        "Refill schedule for {name}?",
        "When do I take {name}?",
        "Find {name} in my record.",
        "Look up {name} for me.",
        "Verify {name} is in my list.",
        "Schedule for {name} medication?",
        "Give me {name} details, please.",
        "Read out my {name} prescription.",
        "Confirm {name} is prescribed.",
        "Need information on {name}.",
        "Surface {name} from the record.",
        "Retrieve {name} for me.",
        "List {name} dosage, please.",
        "Fetch my {name} record.",
        "Open {name} entry, please.",
        "Bring up {name}, please.",
        "Cite {name} schedule, please.",
    ]
    for i, name in enumerate(no_match_meds):
        prompt = no_match_prompts[i].format(name=name)
        rows.append(_row(
            f"te-{rid}",
            "tool_error_recovery",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(f"Look up {name} in the medication record."),
                 "tool_calls": [_tool_call("call_1", "get_medication_by_name",
                                           {"name": name})]},
                _tool_result("call_1", "get_medication_by_name",
                             {"error": "no_match", "name": name}),
                {"role": "assistant",
                 "content": _think_answer(
                     "Tool returned no_match; surface absence to user.",
                     f"{name} was not found in your medication record. "
                     "You do not have a prescription for it.")},
            ],
        ))
        rid += 1

    # 8 rows: ambiguous prefix — the model issues a short prefix and the
    # registry returns an ambiguous error with multiple matches.
    ambiguous_prompts = [
        ("Pills starting with letter a in my list?",
         "a", ["Aspirin", "Albuterol", "Atorvastatin", "Amlodipine"]),
        ("Anything beginning at 'at' on my list?",
         "at", ["Atorvastatin"]),  # actually unique — change
        ("Search 'lis' in my prescriptions.",
         "lis", ["Lisinopril"]),
        ("Find meds starting 'me'.",
         "me", ["Metformin", "Metoprolol"]),
        ("Pull up the 'atr' med.",
         "atr", ["Atorvastatin"]),
        ("Surface meds with prefix 'om'.",
         "om", ["Omeprazole"]),
        ("Lookup 'ros' medication, please.",
         "ros", ["Rosuvastatin"]),
        ("Go for the 'tra' entry.",
         "tra", ["Tramadol", "Trazodone"]),
    ]
    # Force ambiguous payloads regardless of how many matches we listed —
    # this models the model issuing a too-short prefix and the tool returning
    # the ambiguous error. Replace 1-element matches with multi-element ones
    # for realism.
    realistic_ambiguous = [
        ("a", ["Aspirin", "Albuterol", "Atorvastatin", "Amlodipine"]),
        ("at", ["Atorvastatin", "Atenolol"]),
        ("lis", ["Lisinopril", "Lispro"]),
        ("me", ["Metformin", "Metoprolol"]),
        ("atr", ["Atorvastatin", "Atropine"]),
        ("om", ["Omeprazole", "Omega-3"]),
        ("ros", ["Rosuvastatin", "Rosiglitazone"]),
        ("tra", ["Tramadol", "Trazodone"]),
    ]
    for i, ((prompt, _q, _m), (prefix, matches)) in enumerate(
            zip(ambiguous_prompts, realistic_ambiguous)):
        rows.append(_row(
            f"te-{rid}",
            "tool_error_recovery",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(f"Look up prefix {prefix!r} in record."),
                 "tool_calls": [_tool_call("call_1", "get_medication_by_name",
                                           {"name": prefix})]},
                _tool_result("call_1", "get_medication_by_name",
                             {"error": "ambiguous", "name": prefix,
                              "matches": matches}),
                {"role": "assistant",
                 "content": _think_answer(
                     "Tool returned ambiguous; ask for clarification.",
                     f"The query '{prefix}' is ambiguous — possible matches: "
                     f"{', '.join(matches)}. Please specify the full name.")},
            ],
        ))
        rid += 1

    # 10 rows: empty results from no-arg tools (no appointment, no contacts).
    empty_specs = [
        ("Need my next visit, please.",
         "get_next_appointment",
         {"error": "no_appointments"},
         "No upcoming appointment is on file right now."),
        ("Schedule my next clinic visit?",
         "get_next_appointment",
         {"error": "no_appointments"},
         "There are no scheduled appointments in your record."),
        ("Outstanding appointments?",
         "get_next_appointment",
         {"error": "no_appointments"},
         "Your record currently has no upcoming appointments."),
        ("Visit on the books?",
         "get_next_appointment",
         {"error": "no_appointments"},
         "No appointments are currently scheduled."),
        ("Emergency contact details, please.",
         "get_emergency_contact",
         {"error": "no_contacts"},
         "No emergency contact is currently on file."),
        ("Family contact info?",
         "get_emergency_contact",
         {"error": "no_contacts"},
         "Your record has no emergency contact listed."),
        ("Reach-someone-in-emergency info?",
         "get_emergency_contact",
         {"error": "no_contacts"},
         "No emergency-contact phone is on record."),
        ("Coverage for emergencies on file?",
         "get_emergency_contact",
         {"error": "no_contacts"},
         "No emergency contact has been recorded."),
        ("Quick — emergency dial-out number?",
         "get_emergency_contact",
         {"error": "no_contacts"},
         "There is no emergency contact stored in your record."),
        ("Provide upcoming appointment details.",
         "get_next_appointment",
         {"error": "no_appointments"},
         "No upcoming appointment is available."),
    ]
    for prompt, tool_name, payload, ans in empty_specs:
        rows.append(_row(
            f"te-{rid}",
            "tool_error_recovery",
            [
                _system_msg(),
                {"role": "user", "content": prompt},
                {"role": "assistant",
                 "content": _think_call(f"Call {tool_name} for the user."),
                 "tool_calls": [_tool_call("call_1", tool_name, {})]},
                _tool_result("call_1", tool_name, payload),
                {"role": "assistant",
                 "content": _think_answer(
                     "Tool returned empty/error; surface that.", ans)},
            ],
        ))
        rid += 1

    assert len(rows) == 40, f"tool_error_recovery expected 40 rows, got {len(rows)}"
    return rows


def gen_two_turn() -> list[dict]:
    rows: list[dict] = []
    rid = 501

    # Each two-turn row has 2 user turns; each turn is one of several patterns.
    # We build 40 distinct (turn1, turn2) prompt pairs ensuring all 80 user
    # prompts have distinct first-4-word prefixes within the category.

    # 20 rows: med-then-time (lookup a med, then ask what's at HH:MM)
    med_then_time = [
        ("Lisinopril", "08:00"), ("Metformin", "19:00"),
        ("Atorvastatin", "21:00"), ("Levothyroxine", "06:30"),
        ("Amlodipine", "08:30"), ("Omeprazole", "07:00"),
        ("Sertraline", "20:00"), ("Albuterol", "as needed"),
        ("Aspirin", "08:00"), ("Vitamin D3", "12:00"),
        ("Losartan", "09:00"), ("Simvastatin", "22:00"),
        ("Pantoprazole", "07:30"), ("Furosemide", "10:30"),
        ("Citalopram", "21:30"), ("Hydrochlorothiazide", "06:45"),
        ("Escitalopram", "22:45"), ("Tramadol", "13:00"),
        ("Trazodone", "23:00"), ("Fluoxetine", "08:00"),
    ]
    turn1_med_templates = [
        "Tell me the dose of {name}.",
        "What dose is my {name}?",
        "Look up {name} dose, please.",
        "Pull up {name} dosage now.",
        "Read out my {name} dose.",
        "Surface {name} prescription details.",
        "Cite the {name} dose, please.",
        "Bring up dosage for {name}.",
        "Verify the dose of {name}.",
        "Open my {name} record briefly.",
        "Spell out {name} schedule.",
        "Display {name} info for me.",
        "Walk through {name} prescription.",
        "Recite {name} dose, please.",
        "Quote {name} from my record.",
        "Detail {name} prescription, please.",
        "Fetch the {name} entry.",
        "State the dose of {name}.",
        "Confirm {name} dose, please.",
        "Output {name} prescription, please.",
    ]
    turn2_time_templates = [
        "And which pills come at {t}?",
        "Now show meds at {t} please.",
        "Then what's scheduled around {t}?",
        "Also list meds for {t}.",
        "Anything at {t} on the list?",
        "Follow up — meds at {t}?",
        "Plus the {t} dosing, please.",
        "Continue with {t} meds.",
        "Likewise — pills at {t}?",
        "Could you also do {t}?",
        "Furthermore: meds at {t}?",
        "Subsequently — {t} list?",
        "Next, what's at {t}?",
        "Second part: {t} pills?",
        "Then — {t} schedule?",
        "Onward: meds at {t}?",
        "Additionally, the {t} dosing?",
        "Following that, {t} pills?",
        "Now, meds at {t}?",
        "After that — {t} schedule?",
    ]
    for i, (med, time) in enumerate(med_then_time):
        rec = _MED_RECORDS.get(med) or _generic_med(med)
        # Turn 2 uses a different time per row to broaden the time vocabulary.
        t2 = TIMES[(i * 3 + 5) % len(TIMES)]
        msgs = [
            _system_msg(),
            {"role": "user", "content": turn1_med_templates[i].format(name=med)},
            {"role": "assistant",
             "content": _think_call(f"Look up {med}."),
             "tool_calls": [_tool_call("call_1", "get_medication_by_name",
                                       {"name": med})]},
            _tool_result("call_1", "get_medication_by_name", rec),
            {"role": "assistant",
             "content": _think_answer(
                 "Surface dose/schedule.",
                 f"Your {rec['name']} dose is {rec['dose']} at {rec['schedule']}.")},
            {"role": "user",
             "content": turn2_time_templates[i].format(t=t2)},
            {"role": "assistant",
             "content": _think_call(f"Lookup meds at {t2}."),
             "tool_calls": [_tool_call("call_2", "get_medications_at_time",
                                       {"time_24h": t2})]},
            _tool_result("call_2", "get_medications_at_time", _meds_at_empty()),
            {"role": "assistant",
             "content": _think_answer(
                 "Empty list; surface that.",
                 f"At {t2} the record shows no scheduled medications.")},
        ]
        rows.append(_row(f"tt-{rid}", "two_turn", msgs))
        rid += 1

    # 10 rows: vitals-then-food (read vitals, then ask about a food)
    vit_then_food = [
        "broccoli", "raw eggs", "Brussels sprouts", "kiwi", "licorice",
        "soy milk", "aged cheddar", "smoked salmon", "cranberry juice", "walnuts",
    ]
    turn1_vit_templates = [
        "Read me my latest vitals briefly.",
        "Pull up my vitals snapshot, please.",
        "Latest BP and HR figures?",
        "Vitals overview, please.",
        "Quick vitals readout?",
        "Most recent vital signs?",
        "Show vitals from last reading.",
        "Surface my latest readings.",
        "Provide vitals on file.",
        "Cite my latest vital values.",
    ]
    turn2_food_templates = [
        "Plus — is {f} okay for me?",
        "Also: any issue eating {f}?",
        "Then check {f} interaction.",
        "Likewise on {f} — safe?",
        "Continue — does {f} interfere?",
        "Additionally test {f}, please.",
        "And {f} compatibility?",
        "Bonus: review {f}.",
        "Could you do {f} too?",
        "Add {f} to the check.",
    ]
    for i, food in enumerate(vit_then_food):
        msgs = [
            _system_msg(),
            {"role": "user", "content": turn1_vit_templates[i]},
            {"role": "assistant",
             "content": _think_call("Read latest vitals."),
             "tool_calls": [_tool_call("call_1", "get_vitals", {})]},
            _tool_result("call_1", "get_vitals", _VITALS),
            {"role": "assistant",
             "content": _think_answer(
                 "Surface vitals overview.",
                 f"BP {_VITALS['blood_pressure_systolic']}/"
                 f"{_VITALS['blood_pressure_diastolic']}, "
                 f"HR {_VITALS['heart_rate_bpm']}, "
                 f"SpO2 {_VITALS['spo2_percent']}%.")},
            {"role": "user",
             "content": turn2_food_templates[i].format(f=food)},
            {"role": "assistant",
             "content": _think_call(f"Check food interaction for {food}."),
             "tool_calls": [_tool_call("call_2", "check_food_interaction",
                                       {"food": food})]},
            _tool_result("call_2", "check_food_interaction",
                         _food_response(food, False, [], None)),
            {"role": "assistant",
             "content": _think_answer(
                 "No interaction; surface that.",
                 f"{food} has no interaction with your current medications.")},
        ]
        rows.append(_row(f"tt-{rid}", "two_turn", msgs))
        rid += 1

    # 10 rows: appointment-then-allergies / appointment-then-emergency_contact
    appt_then_other = [
        ("allergies", "Run through my known allergies briefly."),
        ("emergency", "Whose number do I call in emergency?"),
        ("allergies", "Allergens on file too?"),
        ("emergency", "Family emergency contact, please."),
        ("allergies", "Add my allergy list now."),
        ("emergency", "Emergency phone backup, please?"),
        ("allergies", "Allergen rundown, please."),
        ("emergency", "Backup contact for emergencies?"),
        ("allergies", "Allergy panel quickly?"),
        ("emergency", "Reach-out number for crisis?"),
    ]
    turn1_appt_templates = [
        "When is my next clinic visit?",
        "Confirm next appointment date.",
        "Upcoming visit details, please.",
        "Date of next checkup?",
        "Read my next appointment.",
        "Provide next visit info.",
        "Verify next appointment time.",
        "Cite my upcoming visit.",
        "Bring up next appointment.",
        "Surface next appointment, please.",
    ]
    for i, (kind, t2_prompt) in enumerate(appt_then_other):
        if kind == "allergies":
            tool2 = "list_allergies"
            payload2 = _ALLERGIES
            ans2 = "On file: Penicillin (severe) and Shellfish (moderate)."
            reason2 = "List allergies for follow-up."
        else:
            tool2 = "get_emergency_contact"
            payload2 = _EMERGENCY_CONTACT
            ans2 = "Sarah Park, your daughter, at 555-0148."
            reason2 = "Get emergency contact for follow-up."
        msgs = [
            _system_msg(),
            {"role": "user", "content": turn1_appt_templates[i]},
            {"role": "assistant",
             "content": _think_call("Look up next appointment."),
             "tool_calls": [_tool_call("call_1", "get_next_appointment", {})]},
            _tool_result("call_1", "get_next_appointment", _NEXT_APPOINTMENT),
            {"role": "assistant",
             "content": _think_answer(
                 "Surface upcoming appointment.",
                 "Next: 2026-05-06 at 10:30 with Dr. Evelyn Chen, Maple Clinic.")},
            {"role": "user", "content": t2_prompt},
            {"role": "assistant",
             "content": _think_call(reason2),
             "tool_calls": [_tool_call("call_2", tool2, {})]},
            _tool_result("call_2", tool2, payload2),
            {"role": "assistant",
             "content": _think_answer(
                 "Surface follow-up data.", ans2)},
        ]
        rows.append(_row(f"tt-{rid}", "two_turn", msgs))
        rid += 1

    assert len(rows) == 40, f"two_turn expected 40 rows, got {len(rows)}"
    return rows


def gen_parallel_call() -> list[dict]:
    rows: list[dict] = []
    rid = 501

    # 12 rows: med + time (look up med name + look up time slot in parallel)
    pc_med_time = [
        ("Lisinopril", "08:00"), ("Metformin", "19:00"),
        ("Atorvastatin", "21:00"), ("Levothyroxine", "06:30"),
        ("Amlodipine", "08:30"), ("Omeprazole", "07:00"),
        ("Sertraline", "20:00"), ("Aspirin", "08:00"),
        ("Vitamin D3", "12:00"), ("Albuterol", "13:00"),
        ("Losartan", "16:15"), ("Pravastatin", "22:30"),
    ]
    pc_med_time_templates = [
        "Dose of {name} and meds at {t}, please?",
        "Pull up {name} plus the {t} list.",
        "Show {name} dose and what's at {t}.",
        "Verify {name} schedule, plus {t} pills.",
        "Look up {name} and meds for {t}.",
        "Combine {name} info with {t} list.",
        "Provide {name} dose and {t} meds.",
        "Tell me about {name} and {t}.",
        "Bring up {name} and {t} pills.",
        "Run {name} lookup and {t} list.",
        "Get {name} details plus {t} meds.",
        "Surface {name} dose and {t} list.",
    ]
    for i, (name, t) in enumerate(pc_med_time):
        rec = _MED_RECORDS.get(name) or _generic_med(name)
        msgs = [
            _system_msg(),
            {"role": "user", "content": pc_med_time_templates[i].format(name=name, t=t)},
            {"role": "assistant",
             "content": _think_call(f"Two parallel lookups: {name} and time {t}."),
             "tool_calls": [
                 _tool_call("call_1", "get_medication_by_name", {"name": name}),
                 _tool_call("call_2", "get_medications_at_time", {"time_24h": t}),
             ]},
            _tool_result("call_1", "get_medication_by_name", rec),
            _tool_result("call_2", "get_medications_at_time", _meds_at_empty()),
            {"role": "assistant",
             "content": _think_answer(
                 "Combine both results.",
                 f"{rec['name']}: {rec['dose']} at {rec['schedule']}. "
                 f"At {t}, the record shows no scheduled medications.")},
        ]
        rows.append(_row(f"pc-{rid}", "parallel_call", msgs))
        rid += 1

    # 10 rows: food + med (food interaction in parallel with med lookup) —
    # uses foods not yet seen in fl/tt to broaden the vocabulary.
    pc_food_med = [
        ("grapefruit", "Atorvastatin"),
        ("alcohol", "Sertraline"),
        ("grapefruit juice", "Lisinopril"),
        ("shellfish", "Metformin"),
        ("bananas", "Aspirin"),
        ("black tea", "Levothyroxine"),
        ("cured ham", "Omeprazole"),
        ("garlic", "Amlodipine"),
        ("pomegranate", "Albuterol"),
        ("miso paste", "Vitamin D3"),
    ]
    pc_food_med_templates = [
        "Is {f} safe and what's my {name} dose?",
        "Check {f} interaction plus {name} info.",
        "Both: {f} okay? And {name} schedule?",
        "Combine {f} check with {name} lookup.",
        "Tell me about {f} and {name} together.",
        "Run {f} check plus {name} record.",
        "Verify {f} safety and {name} dose.",
        "Pull both — {f} and {name}.",
        "Status of {f} and {name}, please.",
        "Read {f} interaction plus {name}.",
    ]
    for i, (food, name) in enumerate(pc_food_med):
        rec = _MED_RECORDS.get(name) or _generic_med(name)
        # Make some interactions show as true (grapefruit/Atorvastatin) for realism.
        if food == "grapefruit" and name == "Atorvastatin":
            food_payload = _food_response(food, True, ["Atorvastatin"], None)
            food_summary = f"{food} interacts with Atorvastatin"
        else:
            food_payload = _food_response(food, False, [], None)
            food_summary = f"{food} has no interaction with current meds"
        msgs = [
            _system_msg(),
            {"role": "user",
             "content": pc_food_med_templates[i].format(f=food, name=name)},
            {"role": "assistant",
             "content": _think_call(
                 f"Two parallel lookups: food {food} and med {name}."),
             "tool_calls": [
                 _tool_call("call_1", "check_food_interaction", {"food": food}),
                 _tool_call("call_2", "get_medication_by_name", {"name": name}),
             ]},
            _tool_result("call_1", "check_food_interaction", food_payload),
            _tool_result("call_2", "get_medication_by_name", rec),
            {"role": "assistant",
             "content": _think_answer(
                 "Combine both results.",
                 f"{food_summary}. {rec['name']}: {rec['dose']} at {rec['schedule']}.")},
        ]
        rows.append(_row(f"pc-{rid}", "parallel_call", msgs))
        rid += 1

    # 10 rows: vitals + allergies / vitals + appointment / allergies + emergency / etc.
    pc_no_arg_pairs = [
        ("get_vitals", "list_allergies",
         "Vitals plus allergies overview, please.",
         "Look up vitals and allergies in parallel."),
        ("get_vitals", "get_next_appointment",
         "Vitals and next-visit summary, please.",
         "Vitals plus next appointment in parallel."),
        ("list_allergies", "get_emergency_contact",
         "Allergies plus emergency contact?",
         "Allergies and emergency contact in parallel."),
        ("get_next_appointment", "get_emergency_contact",
         "Next appointment plus emergency phone?",
         "Appointment and emergency contact in parallel."),
        ("get_vitals", "get_emergency_contact",
         "Quick vitals plus emergency phone, please.",
         "Vitals and emergency contact in parallel."),
        ("list_allergies", "get_next_appointment",
         "Allergies and the next appointment, please?",
         "Allergies plus next appointment in parallel."),
        ("get_vitals", "list_allergies",
         "Combine vitals and allergens, please.",
         "Vitals and allergies in parallel."),
        ("get_vitals", "get_next_appointment",
         "BP overview and next visit?",
         "Vitals plus appointment in parallel."),
        ("list_allergies", "get_emergency_contact",
         "Patient allergies plus emergency line?",
         "Allergies and emergency in parallel."),
        ("get_next_appointment", "get_emergency_contact",
         "Outstanding visit and emergency contact?",
         "Appointment and emergency in parallel."),
    ]
    payload_for = {
        "get_vitals": _VITALS,
        "list_allergies": _ALLERGIES,
        "get_next_appointment": _NEXT_APPOINTMENT,
        "get_emergency_contact": _EMERGENCY_CONTACT,
    }
    summary_for = {
        "get_vitals": (f"BP {_VITALS['blood_pressure_systolic']}/"
                       f"{_VITALS['blood_pressure_diastolic']}, "
                       f"HR {_VITALS['heart_rate_bpm']}, "
                       f"SpO2 {_VITALS['spo2_percent']}%"),
        "list_allergies": "Penicillin (severe), Shellfish (moderate)",
        "get_next_appointment": "Next: 2026-05-06 10:30 with Dr. Chen",
        "get_emergency_contact": "Sarah Park (daughter), 555-0148",
    }
    for i, (t1, t2, prompt, reason) in enumerate(pc_no_arg_pairs):
        msgs = [
            _system_msg(),
            {"role": "user", "content": prompt},
            {"role": "assistant",
             "content": _think_call(reason),
             "tool_calls": [
                 _tool_call("call_1", t1, {}),
                 _tool_call("call_2", t2, {}),
             ]},
            _tool_result("call_1", t1, payload_for[t1]),
            _tool_result("call_2", t2, payload_for[t2]),
            {"role": "assistant",
             "content": _think_answer(
                 "Combine both results.",
                 f"{summary_for[t1]}. {summary_for[t2]}.")},
        ]
        rows.append(_row(f"pc-{rid}", "parallel_call", msgs))
        rid += 1

    # 8 rows: time + food (parallel) — fresh foods not in fl/tt/pc_food_med.
    pc_time_food = [
        ("06:00", "coffee"),
        ("12:00", "tuna"),
        ("18:00", "green tea"),
        ("21:00", "kombucha"),
        ("09:45", "St John's wort"),
        ("13:45", "fermented sauerkraut"),
        ("17:45", "kiwi"),
        ("23:45", "ginger"),
    ]
    pc_time_food_templates = [
        "What's at {t} and is {f} okay?",
        "Pills at {t} plus {f} interaction?",
        "List {t} meds, also {f} status?",
        "Show {t} schedule and {f} check.",
        "Provide {t} meds and {f} status.",
        "Bring up {t} list with {f} check.",
        "Read {t} pills and {f} interaction.",
        "Pull {t} and {f} together.",
    ]
    for i, (t, food) in enumerate(pc_time_food):
        msgs = [
            _system_msg(),
            {"role": "user", "content": pc_time_food_templates[i].format(t=t, f=food)},
            {"role": "assistant",
             "content": _think_call(f"Two parallel lookups: time {t} and food {food}."),
             "tool_calls": [
                 _tool_call("call_1", "get_medications_at_time", {"time_24h": t}),
                 _tool_call("call_2", "check_food_interaction", {"food": food}),
             ]},
            _tool_result("call_1", "get_medications_at_time", _meds_at_empty()),
            _tool_result("call_2", "check_food_interaction",
                         _food_response(food, False, [], None)),
            {"role": "assistant",
             "content": _think_answer(
                 "Combine both results.",
                 f"At {t}, no scheduled medications. {food} has no interaction.")},
        ]
        rows.append(_row(f"pc-{rid}", "parallel_call", msgs))
        rid += 1

    assert len(rows) == 40, f"parallel_call expected 40 rows, got {len(rows)}"
    return rows


# endregion


# ==========================================================================
# region in-process gates — validator + Block E custom audit
# ==========================================================================

_PLACEHOLDER_RE = re.compile(
    r"topic_\d+|Combined request \d+|Unique prompt fallback",
    re.IGNORECASE,
)
_FORBIDDEN_TOKEN_RE = re.compile(
    r"<bos>|<start_of_turn>|<end_of_turn>|<start_function_call>|<escape>|"
    r"<answer>|</answer>",
    re.IGNORECASE,
)
_EXPECTED_COUNTS = {
    "off_topic_refusal": 80,
    "medical_advice_refusal": 80,
    "fact_lookup": 60,
    "fact_absence": 30,
    "tool_error_recovery": 40,
    "two_turn": 40,
    "parallel_call": 40,
}
_EXPECTED_ID_PREFIX = {
    "off_topic_refusal": ("ot-", 501, 580),
    "medical_advice_refusal": ("ma-", 501, 580),
    "fact_lookup": ("fl-", 501, 560),
    "fact_absence": ("fa-", 501, 530),
    "tool_error_recovery": ("te-", 501, 540),
    "two_turn": ("tt-", 501, 540),
    "parallel_call": ("pc-", 501, 540),
}


def _first4(s: str) -> tuple[str, ...]:
    """Lower-case, split on whitespace, take first 4 tokens. Punctuation stays
    on the token because the rule is 'first 4 words verbatim' — diff in
    punctuation is a meaningful diff."""
    return tuple(s.lower().split()[:4])


def _arg_vocab_counts(rows: list[dict]) -> dict[str, int]:
    """Count unique values seen for each open-string argument across rows."""
    food_set: set[str] = set()
    time_set: set[str] = set()
    name_set: set[str] = set()
    for r in rows:
        for m in r["messages"]:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls", []) or []:
                fn = tc["function"]
                args = fn["arguments"]
                if not isinstance(args, dict):
                    continue
                if fn["name"] == "check_food_interaction":
                    food_set.add(args.get("food", ""))
                elif fn["name"] == "get_medications_at_time":
                    time_set.add(args.get("time_24h", ""))
                elif fn["name"] == "get_medication_by_name":
                    name_set.add(args.get("name", ""))
    return {
        "check_food_interaction.food": len(food_set),
        "get_medications_at_time.time_24h": len(time_set),
        "get_medication_by_name.name": len(name_set),
    }


def _audit(rows: list[dict]) -> list[str]:
    """Run all Block E acceptance checks. Return list of failure strings."""
    fails: list[str] = []

    # 1. Total row count.
    if len(rows) != 370:
        fails.append(f"row_count: expected 370, got {len(rows)}")

    # 2. Per-category counts and exact id ranges.
    cat_count: Counter[str] = Counter(r["category"] for r in rows)
    for cat, expected in _EXPECTED_COUNTS.items():
        if cat_count[cat] != expected:
            fails.append(f"category {cat}: expected {expected}, got {cat_count[cat]}")
    for cat, (prefix, lo, hi) in _EXPECTED_ID_PREFIX.items():
        ids = sorted(r["id"] for r in rows if r["category"] == cat)
        expected_ids = [f"{prefix}{n}" for n in range(lo, hi + 1)]
        if ids != expected_ids:
            fails.append(
                f"id range for {cat}: expected {prefix}{lo}..{prefix}{hi} "
                f"({len(expected_ids)} ids), got {len(ids)} ids "
                f"(first={ids[0] if ids else 'none'}, "
                f"last={ids[-1] if ids else 'none'})"
            )

    # 3. No duplicate ids.
    id_counter: Counter[str] = Counter(r["id"] for r in rows)
    dups = [rid for rid, c in id_counter.items() if c > 1]
    if dups:
        fails.append(f"duplicate ids: {dups[:10]}")

    # 4. No duplicate user prompts.
    user_prompts: list[tuple[str, str]] = []
    for r in rows:
        for m in r["messages"]:
            if m.get("role") == "user":
                user_prompts.append((r["id"], m["content"]))
    seen_text: dict[str, str] = {}
    for rid, text in user_prompts:
        if text in seen_text:
            fails.append(
                f"duplicate user prompt: {rid} matches {seen_text[text]} — "
                f"{text!r}")
        else:
            seen_text[text] = rid

    # 5. No shared first-4-word prefix within a category.
    by_cat_prefix: dict[str, dict[tuple, str]] = defaultdict(dict)
    for r in rows:
        cat = r["category"]
        for m in r["messages"]:
            if m.get("role") == "user":
                pre = _first4(m["content"])
                if pre in by_cat_prefix[cat]:
                    fails.append(
                        f"first-4-word collision in {cat}: {r['id']} vs "
                        f"{by_cat_prefix[cat][pre]} — {pre!r}")
                else:
                    by_cat_prefix[cat][pre] = r["id"]

    # 6. No forbidden tokens (literal <answer>, special chat-template tokens).
    for r in rows:
        for i, m in enumerate(r["messages"]):
            content = m.get("content", "")
            if _FORBIDDEN_TOKEN_RE.search(content):
                fails.append(
                    f"forbidden token in {r['id']}.messages[{i}]: "
                    f"{content[:120]!r}")

    # 7. No placeholder patterns in user prompts.
    for r in rows:
        for m in r["messages"]:
            if m.get("role") == "user" and _PLACEHOLDER_RE.search(m.get("content", "")):
                fails.append(f"placeholder pattern in {r['id']}: "
                             f"{m['content']!r}")

    # 8. All tool-call arguments are dicts.
    for r in rows:
        for i, m in enumerate(r["messages"]):
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls", []) or []:
                args = tc.get("function", {}).get("arguments")
                if not isinstance(args, dict):
                    fails.append(
                        f"args not dict in {r['id']}.messages[{i}]."
                        f"tool_calls[{tc.get('id')}]: type="
                        f"{type(args).__name__}")

    # 9. All tool messages include `name`.
    for r in rows:
        for i, m in enumerate(r["messages"]):
            if m.get("role") == "tool" and not m.get("name"):
                fails.append(f"tool message missing name in "
                             f"{r['id']}.messages[{i}]")

    # 10. Argument-vocabulary minima.
    counts = _arg_vocab_counts(rows)
    minima = {
        "check_food_interaction.food": 25,
        "get_medications_at_time.time_24h": 25,
        "get_medication_by_name.name": 30,
    }
    for k, mn in minima.items():
        if counts[k] < mn:
            fails.append(f"vocab too narrow: {k} has {counts[k]} unique values, "
                         f"expected ≥ {mn}")

    return fails


# endregion


# ==========================================================================
# region main
# ==========================================================================


def build_rows() -> list[dict]:
    rows: list[dict] = []
    rows.extend(gen_off_topic())
    rows.extend(gen_medical_advice())
    rows.extend(gen_fact_lookup())
    rows.extend(gen_fact_absence())
    rows.extend(gen_tool_error_recovery())
    rows.extend(gen_two_turn())
    rows.extend(gen_parallel_call())
    return rows


def main() -> int:
    rows = build_rows()

    # Validator first — same gate as ingest. Any failure here means the
    # generated row is structurally broken.
    val_failures = []
    for r in rows:
        oc = validate_conversation(r)
        if not oc.ok:
            val_failures.append((r["id"], oc.errors))
    if val_failures:
        print(f"VALIDATOR FAILED on {len(val_failures)} rows:")
        for rid, errs in val_failures[:10]:
            print(f"  {rid}: {errs}")
        return 1

    # Custom Block E audit.
    audit_failures = _audit(rows)
    if audit_failures:
        print(f"BLOCK E AUDIT FAILED ({len(audit_failures)} issues):")
        for f in audit_failures[:30]:
            print(f"  {f}")
        return 2

    # Write the file.
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    counts = _arg_vocab_counts(rows)
    print(f"OK: wrote {len(rows)} rows -> "
          f"{_OUTPUT.relative_to(_REPO)}")
    print(f"  category counts: "
          f"{dict(Counter(r['category'] for r in rows))}")
    print(f"  arg vocab unique counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# endregion
