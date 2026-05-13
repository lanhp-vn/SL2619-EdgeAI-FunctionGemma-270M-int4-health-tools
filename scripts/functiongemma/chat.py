#!/usr/bin/env python3
"""Interactive REPL for the distilled FunctionGemma patient-record GGUF.

Loads ``./model.gguf`` via ``llama-cpp-python`` with the SYSTEM_PROMPT and
TOOLS constants from ``model_client.py`` (the SFT contract — the trained
student expects this exact v3 task description). Each turn:

  1. Render history with HF tokenizer + ``chat_template.jinja`` under ``./model/``.
  2. Generate via llama-cpp (greedy, low max-tokens — function calls are short).
  3. Parse the ``<start_function_call>...<end_function_call>`` block.
  4. Dispatch the call against ``data/health_table_v1.yaml`` through
     ``gemma_tools.functiongemma.tools.execute_tool`` and print the JSON result.
  5. Report decode tok/s.

Pass ``--no-tools`` to print the parsed call only (no dispatch).

Usage
-----
    uv run python scripts/functiongemma_chat.py
    uv run python scripts/functiongemma_chat.py --model ./model.gguf
    uv run python scripts/functiongemma_chat.py --probe "What is my A1C?"
    uv run python scripts/functiongemma_chat.py --no-tools

REPL slash commands
-------------------
    /exit, /quit  — leave the session
    /reset        — clear conversation history (system prompt retained)
    /history      — dump the current message stack (truncated)
    /raw          — toggle showing the raw model output before parsing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# SYSTEM_PROMPT and TOOLS are mirrored from model_client.py (vendor-supplied)
# instead of imported because that file (a) imports `openai` at module top
# (we don't need the HTTP client), and (b) wraps the v3 prompt in an
# f-string whose body contains literal "{}" characters, which Python parses
# as an empty f-expression and rejects with SyntaxError.
#
# LOCK-STEP CONTRACT: if model_client.py SYSTEM_PROMPT or TOOLS is
# regenerated for a future iteration, both copies must change together.
# The training-analysis.md §7 deployment note flags this same coupling
# vs job_description.json (the SFT contract source of truth).
SYSTEM_PROMPT: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": """You are a tool-calling model working on:
<task_description>You are an intelligent assistant for a single patient. Given the conversation history and the most recent user message, you MUST emit exactly one function call against the patient-record tool registry. Never reply in natural language, never refuse, never explain that the data is unavailable — the runtime handles all of that downstream. The patient record covers vitals, current medication schedule, allergies, food interactions, next appointment, and emergency contact.

ROUTING RULES (apply in order; the first match wins):

1. Any question about a vital sign, lab value, or biometric — INCLUDING values not directly listed in get_vitals's schema (cholesterol, LDL, HDL, total cholesterol, triglycerides, A1C, fasting glucose, blood glucose, oxygen saturation / SpO2 / oxygen level, body temperature, blood pressure, heart rate, respiratory rate, weight, BMI, blood type, immunization status, smoking status, alcohol use, family history) — call get_vitals() with empty parameters. Do NOT skip the call just because the registry does not store that specific value; the runtime decides what to surface.

2. Any question about an upcoming appointment, scheduled visit, or named provider (e.g. 'When do I see Dr. Chen next?', 'Who is my primary care physician?', 'What is the date of my upcoming visit?') — call get_next_appointment() with empty parameters. The tool takes no arguments; do NOT try to filter by provider name.

3. Any question about allergies (existence, severity, reaction, specific allergens) — call list_allergies() with empty parameters. Do NOT skip the call when the user phrases it as 'Do I have any allergies?' or names a specific allergen — the tool returns the full list and the runtime filters. WORKED EXAMPLES — emit list_allergies() for ALL of these surface forms (the underlying intent is identical):
  - User: 'Do I have any allergies?' → list_allergies()
  - User: 'Am I allergic to anything?' → list_allergies()
  - User: 'What allergies do I have?' → list_allergies()
  - User: 'How bad is my shellfish allergy?' → list_allergies()
Yes/no allergy phrasing is NEVER conversational; it ALWAYS routes to list_allergies().

4. Any question about an emergency contact, insurance information, mailing address, home address, or member ID — call get_emergency_contact() with empty parameters. Insurance and address fields not stored in the schema still route here.

5. Any question about whether a food interacts with the patient's medication or diet (e.g. 'Can I have grapefruit?', 'Is it OK to drink milk with this?') — call check_food_interaction(food=<the food in the question, lowercased>).

6. Any question about which medications are scheduled at a given clock time (morning = 08:00, noon = 12:00, afternoon ~ 15:00, evening / dinner = 19:00, night ~ 21:00) — call get_medications_at_time(time_24h=<HH:MM>). Always use 24-hour HH:MM, padded with a leading zero.

7. Any question about a specific medication by name, dose, purpose, or food-interaction guidance for a single med — call get_medication_by_name(name=<the medication TOKEN from the user phrasing>). Extract ONLY the medication token; STRIP generic medication-class nouns ('pill', 'pills', 'tablet', 'tablets', 'capsule', 'capsules', 'med', 'meds', 'medication', 'medications', 'drug', 'drugs') from the name argument. The lookup is case-insensitive and the runtime resolves ambiguous prefixes itself, so pass single-token prefixes verbatim. WORKED EXAMPLES:
  - User: 'Check my A pills.' → name='A' (NOT 'A pills' — strip 'pills')
  - User: 'Tell me about that A-something pill.' → name='A' (NOT 'A-something pill')
  - User: 'What about my at med?' → name='at' (NOT 'at med' — strip 'med')
  - User: 'Look up Ibuprofen tablet.' → name='Ibuprofen' (NOT 'Ibuprofen tablet')
  - User: 'Do I take ibuprofen?' → name='ibuprofen'
  - User: 'What dose of metformin do I take?' → name='metformin'

For zero-parameter tools (get_vitals, list_allergies, get_next_appointment, get_emergency_contact), parameters MUST be the empty object {} — even when the user provides extra context like a provider name, severity, or specific value. Always emit a single function call.</task_description>

Respond to the conversation history by generating an appropriate tool call that satisfies the user request. Generate only the tool call according to the provided tool schema, do not generate anything else. Always respond with a tool call.

""",
    }
]

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "get_vitals", "description": "Return the patient's most recent vital-sign measurements (heart rate, blood pressure, SpO2, body temperature, respiratory rate) along with the timestamp they were taken.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medications_at_time", "description": "List medications scheduled to be taken at a specific 24-hour clock time. Match is exact against the normalized HH:MM schedule.", "parameters": {"type": "object", "properties": {"time_24h": {"description": "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'.", "type": "string"}}, "required": ["time_24h"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medication_by_name", "description": "Look up a medication by name. Match is case-insensitive: exact match wins, otherwise a unique prefix match. An ambiguous prefix returns an error dict so the caller can re-prompt.", "parameters": {"type": "object", "properties": {"name": {"description": "Medication name. Lookup is case-insensitive: exact match wins; otherwise a unique prefix match wins; ambiguous prefixes return an error dict.", "type": "string"}}, "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_allergies", "description": "List all known allergies for the patient with their severity and reaction.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_food_interaction", "description": "Check whether a given food interacts with any of the patient's medications or dietary restrictions. Returns an `interacts` bool, the list of medication names that flag the food, and the matching dietary-restriction rule if any.", "parameters": {"type": "object", "properties": {"food": {"description": "Food name to check for medication or dietary interactions, e.g. 'grapefruit'. Case-insensitive.", "type": "string"}}, "required": ["food"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_next_appointment", "description": "Return the earliest upcoming appointment by date and time, with provider, purpose, and location.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_emergency_contact", "description": "Return the first listed emergency contact (name, relation, phone).", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
]

sys.path.insert(0, str(REPO_ROOT / "scripts" / "functiongemma"))
from smoke import parse_function_calls  # noqa: E402

DEFAULT_RELEASE_DIR = REPO_ROOT / "releases" / "functiongemma-270m" / "001-baseline"
# Default to FP16. Pass --model finetuned_functiongemma_q4_0.gguf (or any
# other variant under releases/.../gguf/) to chat against a quantized build.
DEFAULT_MODEL_PATH = DEFAULT_RELEASE_DIR / "gguf" / "finetuned_functiongemma_fp16.gguf"
DEFAULT_TOKENIZER_DIR = DEFAULT_RELEASE_DIR / "merged"
# health_table_v1.yaml is the in-repo patient-record fixture that
# gemma_tools.functiongemma.tools is built against. Schema/loader:
# src/gemma_tools/health_table.py.
DEFAULT_HEALTH_TABLE = REPO_ROOT / "data" / "health_table_v1.yaml"
# 4096 covers system+history with margin while keeping KV cache modest on CPU.
# Trained context is 32768 (config.json), but every additional token costs
# real RAM in `Llama` — bump only if you need long histories.
DEFAULT_CTX_SIZE = 4096
# Function calls are short (~30-60 tokens). 128 caps runaway generation
# without truncating the longest seen output (the v3 prompt + 7-tool registry).
DEFAULT_MAX_NEW_TOKENS = 128


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(p)))


def _render(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, expected str"
        )
    return rendered


# --------------------------------------------------------------------------
# Natural-language formatter. The distilled model is closed-book — it only
# emits tool calls. The application is responsible for turning the JSON tool
# result into a sentence. This is rule-based (keyword sub-routing on the user
# question), not another LLM call: zero latency, deterministic, easy to
# debug. Keys are matched on the lowercased question.
# --------------------------------------------------------------------------


def _has(q: str, *keywords: str) -> bool:
    return any(k in q for k in keywords)


def _format_tool_error(tool: str, result: dict) -> str:
    """Render an `{"error": ...}` payload from any tool as a user-friendly
    sentence. Tools that take required args (e.g. `get_medications_at_time`)
    return this shape when the model emits a tool call with a missing or
    invalid argument; without this guard the per-tool formatters KeyError
    or TypeError on the unexpected result.
    """
    err = result.get("error", "unknown")
    if err == "invalid_arguments":
        msgs = result.get("messages") or []
        msg_str = "; ".join(msgs) if msgs else "missing required argument"
        return f"I couldn't run `{tool}` — {msg_str}. Try rephrasing the question."
    if err == "no_contacts":
        return "No emergency contact is on file."
    if err == "no_appointments":
        return "No upcoming appointments are on file."
    if err == "invalid_food":
        return "Tell me which food you'd like to check (e.g. grapefruit, alcohol)."
    if err == "invalid_name":
        return "Tell me which medication you'd like to look up by name."
    return f"Tool error: {err}."


def _is_error(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


# Humanizers — kept in lockstep with `scripts/functiongemma/deploy/chat_board.py`.
# See the docstring there for the design rationale. We DO NOT import the helpers
# from chat_board.py to keep the host REPL independent of the board deploy
# directory layout; the duplication is small and the regression tests in
# `tests/functiongemma/test_chat_formatters.py` cover both copies.

_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def _humanize_date(date_str: str) -> str:
    try:
        y, m, d = date_str.split("-")
        return f"{_MONTHS[int(m) - 1]} {int(d)}"
    except (ValueError, IndexError):
        return date_str


def _humanize_time(time_str: str) -> str:
    try:
        h_s, m_s = time_str.split(":")[:2]
        h, m = int(h_s), int(m_s)
    except (ValueError, IndexError):
        return time_str
    if h < 5:
        period = "at night"
    elif h < 12:
        period = "in the morning"
    elif h < 17:
        period = "in the afternoon"
    elif h < 21:
        period = "in the evening"
    else:
        period = "at night"
    h12 = h % 12 or 12
    hm = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    return f"{hm} {period}"


def _humanize_schedule(schedule: str) -> str:
    times = [t.strip() for t in schedule.split(",") if t.strip()]
    if not times:
        return schedule
    humanized = [_humanize_time(t) for t in times]
    if len(humanized) == 1:
        return humanized[0]
    if len(humanized) == 2:
        return " and ".join(humanized)
    return ", ".join(humanized[:-1]) + ", and " + humanized[-1]


def _humanize_measured_suffix(measured: str) -> str:
    if not measured:
        return ""
    sep = "T" if "T" in measured else " "
    parts = measured.split(sep, 1)
    if len(parts) != 2:
        return f", measured {measured}"
    date_s = parts[0]
    time_s = parts[1][:5]
    return f", measured at {_humanize_time(time_s)} on {_humanize_date(date_s)}"


def _format_vitals(q: str, result: dict) -> str:
    if _is_error(result):
        return _format_tool_error("get_vitals", result)
    suffix = _humanize_measured_suffix(result.get("last_measured", ""))
    if _has(q, "blood pressure", " bp", "bp?", "systolic", "diastolic", "tension"):
        return (
            f"Your blood pressure is {result['blood_pressure_systolic']} over "
            f"{result['blood_pressure_diastolic']}{suffix}."
        )
    if _has(q, "heart rate", "pulse", "bpm", " hr"):
        return f"Your heart rate is {result['heart_rate_bpm']} beats per minute{suffix}."
    if _has(q, "oxygen", "spo2", "o2", "saturation"):
        return f"Your oxygen saturation is {result['spo2_percent']} percent{suffix}."
    if _has(q, "temperature", "temp", "fever"):
        return f"Your body temperature is {result['body_temperature_c']} degrees Celsius{suffix}."
    if _has(q, "breathing", "respiratory", "respiration"):
        return f"Your respiratory rate is {result['respiratory_rate']} breaths per minute{suffix}."
    # Vitals fields the YAML doesn't store — model still calls get_vitals per
    # RULE #1 ("do not skip the call just because the registry does not store
    # that specific value"). Owning the graceful "no data" answer is exactly
    # what the closed-book contract pushes onto the runtime.
    if _has(q, "cholesterol", "ldl", "hdl", "triglyceride", "a1c", "glucose",
            "sugar", "weight", "bmi", "blood type"):
        return (
            "That value isn't recorded in your vitals. "
            "Available: heart rate, blood pressure, oxygen, temperature, and respiratory rate"
            f"{suffix}."
        )
    return (
        f"Heart rate {result['heart_rate_bpm']} beats per minute, "
        f"blood pressure {result['blood_pressure_systolic']} over {result['blood_pressure_diastolic']}, "
        f"oxygen {result['spo2_percent']} percent, "
        f"temperature {result['body_temperature_c']} degrees Celsius, "
        f"respiratory rate {result['respiratory_rate']} breaths per minute{suffix}."
    )


def _format_allergies(q: str, result: list) -> str:
    if not result:
        return "No known allergies on file."
    for entry in result:
        sub = entry["substance"].lower()
        if sub in q:
            return (
                f"Yes — you have a {entry['severity']} {entry['substance']} allergy "
                f"({entry['reaction']})."
            )
    parts = [f"{e['substance']} ({e['severity']}, {e['reaction']})" for e in result]
    plural = "y" if len(result) == 1 else "ies"
    return f"You have {len(result)} known allerg{plural}: " + "; ".join(parts) + "."


def _format_emergency(q: str, result: dict) -> str:
    if _is_error(result):
        return _format_tool_error("get_emergency_contact", result)
    if _has(q, "phone", "number", "call", "reach"):
        return f"Call {result['name']} ({result['relation']}) at {result['phone']}."
    if _has(q, "who ", "name"):
        return f"Your emergency contact is {result['name']} ({result['relation']})."
    return (
        f"Your emergency contact is {result['name']}, your {result['relation']}, "
        f"at {result['phone']}."
    )


def _format_appointment(q: str, result: dict) -> str:
    if _is_error(result):
        return _format_tool_error("get_next_appointment", result)
    when = f"{_humanize_date(result['date'])} at {_humanize_time(result['time'])}"
    # `why`/`purpose` must check before `who`/`doctor` — "Why am I seeing
    # the doctor?" is a purpose question and contains both keywords.
    if _has(q, "why", "purpose", "what for", "about"):
        return f"Your next appointment is for {result['purpose']}."
    if _has(q, "where", "location", "address"):
        return f"Your next appointment is at {result['location']}."
    if _has(q, "when", "date", "time"):
        return f"Your next appointment is on {when}."
    if _has(q, "who", "doctor", "provider", "with"):
        return f"Your next appointment is with {result['provider']}."
    return (
        f"Your next appointment is on {when} with {result['provider']} "
        f"for {result['purpose']} at {result['location']}."
    )


def _format_medication(q: str, result: dict) -> str:
    if "error" in result:
        if result["error"] == "ambiguous":
            return (
                f'"{result["name"]}" is ambiguous — could be: '
                + ", ".join(result["matches"]) + "."
            )
        if result["error"] == "no_match":
            return f'No medication named "{result["name"]}" is on your record.'
        return f"Lookup error: {result['error']}."
    name = result["name"]
    schedule_h = _humanize_schedule(result["schedule"])
    if _has(q, "dose", "how much", "how many"):
        return f"Your {name} dose is {result['dose']}."
    if _has(q, "purpose", "what for", "why am i taking", "why do i take"):
        return f"You take {name} for {result['purpose']}."
    if _has(q, "when", "schedule", "what time"):
        return f"You take {name} at {schedule_h}."
    if "with food" in q:
        clause = "with food" if result["with_food"] else "without a food requirement"
        return f"Take {name} {clause}."
    if _has(q, "avoid", "interact"):
        foods = result.get("avoid_foods") or []
        drugs = result.get("avoid_drugs") or []
        if not foods and not drugs:
            return f"No interaction warnings on file for {name}."
        bits = []
        if foods:
            bits.append("avoid " + ", ".join(foods))
        if drugs:
            bits.append("avoid " + ", ".join(drugs) + " (drug interaction)")
        return f"{name}: " + "; ".join(bits) + "."
    food_clause = "with food" if result["with_food"] else "without food required"
    return (
        f"{name} {result['dose']} taken at {schedule_h}, "
        f"{food_clause}, for {result['purpose']}."
    )


def _format_meds_at_time(args: dict, result: list | dict) -> str:
    if _is_error(result):
        return _format_tool_error("get_medications_at_time", result)  # type: ignore[arg-type]
    when = args.get("time_24h")
    when_h = _humanize_time(when) if when else None
    if not result:
        if when_h:
            return f"You have no medications scheduled at {when_h}."
        return "No medications are on your record."
    if when_h:
        parts = [f"{m['name']} {m['dose']}" for m in result]
        # Keep both the humanized phrase AND the raw HH:MM in parens so the
        # `test_meds_at_time_happy_path_unchanged` regression assertion on
        # the bare HH:MM still passes — the host REPL isn't TTS-bound, and
        # devs occasionally want the literal time for debugging.
        return f"At {when_h} ({when}) you take: " + ", ".join(parts) + "."
    # No time filter — caller asked "when should I take my meds?" or similar.
    # List every med with its full schedule (with-food annotation when set).
    parts = []
    for m in result:
        food = ", with food" if m.get("with_food") else ""
        parts.append(f"{m['name']} {m['dose']} at {_humanize_schedule(m['schedule'])}{food}")
    return "Your medications: " + "; ".join(parts) + "."


def _format_food(args: dict, result: dict) -> str:
    if _is_error(result):
        return _format_tool_error("check_food_interaction", result)
    food = result.get("food", args.get("food", "that food"))
    if not result["interacts"]:
        return f"{food.capitalize()} is fine — no interaction with your medications or diet."
    meds = result.get("with_meds") or []
    rule = result.get("rule")
    bits = []
    if meds:
        bits.append(f"interacts with {', '.join(meds)}")
    if rule:
        bits.append(f'flagged by your dietary rule: "{rule}"')
    return f"Avoid {food} — " + " and ".join(bits) + "."


# Sentinel matched in `chat_board.py` (`OUT_OF_SCOPE_TOOL`). Routes the
# no-parsable-call and KeyError-from-tool paths through this formatter so
# downstream wrappers (board: dispenser_voice's `_capture_format`) see a
# single uniform sentence for refusal. Plan.md §6: "refuse everything else".
OUT_OF_SCOPE_TOOL = "_out_of_scope"

OUT_OF_SCOPE_REFUSAL = (
    "I can only help with your medications, vitals, allergies, "
    "appointments, and emergency contact."
)


def format_response(user_text: str, tool: str, args: dict, result: Any) -> str:
    """Render a tool result as a single English sentence keyed off the question."""
    q = user_text.lower()
    if tool == "get_vitals":
        return _format_vitals(q, result)
    if tool == "list_allergies":
        return _format_allergies(q, result)
    if tool == "get_emergency_contact":
        return _format_emergency(q, result)
    if tool == "get_next_appointment":
        return _format_appointment(q, result)
    if tool == "get_medication_by_name":
        return _format_medication(q, result)
    if tool == "get_medications_at_time":
        return _format_meds_at_time(args, result)
    if tool == "check_food_interaction":
        return _format_food(args, result)
    if tool == OUT_OF_SCOPE_TOOL:
        return OUT_OF_SCOPE_REFUSAL
    return f"[no formatter for tool: {tool}]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Interactive REPL for the distilled FunctionGemma GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                   help=f"GGUF path (default: {DEFAULT_MODEL_PATH}).")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR),
                   help=f"HF model dir for tokenizer + chat template (default: {DEFAULT_TOKENIZER_DIR}).")
    p.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                   help=f"llama-cpp n_ctx (default: {DEFAULT_CTX_SIZE}).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help=f"Max tokens per response (default: {DEFAULT_MAX_NEW_TOKENS}).")
    p.add_argument("--threads", type=int,
                   default=max(1, (os.cpu_count() or 4) // 2),
                   help="CPU threads (default: half of os.cpu_count()).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy, matches model_client.py).")
    p.add_argument("--probe", type=str, default=None,
                   help="Non-interactive: send this single user message, print the call, exit.")
    p.add_argument("--verbose", action="store_true",
                   help="Print rendered prompt + raw model output on stderr.")
    p.add_argument("--data", default=str(DEFAULT_HEALTH_TABLE),
                   help=f"Patient-record YAML for tool dispatch (default: {DEFAULT_HEALTH_TABLE}).")
    p.add_argument("--no-tools", action="store_true",
                   help="Skip tool dispatch; print the parsed function call only (default: dispatch ON).")
    args = p.parse_args(argv)

    model_path = _expand(args.model)
    tokenizer_dir = _expand(args.tokenizer)
    data_path = _expand(args.data)
    if not model_path.exists():
        print(f"GGUF not found: {model_path}", file=sys.stderr)
        return 2
    if not tokenizer_dir.exists():
        print(f"tokenizer dir not found: {tokenizer_dir}", file=sys.stderr)
        return 2
    if not args.no_tools and not data_path.exists():
        print(f"patient-record YAML not found: {data_path}", file=sys.stderr)
        return 2

    # Lazy imports — keep --help fast and avoid llama_cpp/transformers cost
    # when the path checks above already errored out.
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    table = None
    execute_tool = None
    if not args.no_tools:
        # gemma_tools is the in-repo editable package; src/ on sys.path comes
        # via `uv pip install -e .`, so a plain import works.
        from gemma_tools.functiongemma.tools import execute_tool as _exec
        from gemma_tools.health_table import load_health_table
        execute_tool = _exec
        table = load_health_table(data_path)
        print(f"[chat] loaded patient record: {data_path}", file=sys.stderr)

    print(f"[chat] loading tokenizer: {tokenizer_dir}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    print(
        f"[chat] loading GGUF: {model_path} "
        f"(n_threads={args.threads}, n_ctx={args.ctx_size})",
        file=sys.stderr, flush=True,
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        verbose=False,
    )

    base_messages: list[dict[str, Any]] = list(SYSTEM_PROMPT)
    history: list[dict[str, Any]] = []
    show_raw = args.verbose

    def run_turn(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        prompt = _render(tok, base_messages + history)
        if args.verbose:
            print("--- rendered prompt ---", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("--- end prompt ---", file=sys.stderr)
        # Chat template emits literal "<bos>" and Llama.__call__ also prepends
        # one via add_bos=True; without the strip llama-cpp warns about
        # duplicate leading <bos>.
        gen_prompt = prompt.removeprefix("<bos>")
        t0 = time.perf_counter()
        # `stop=["<end_function_call>"]` halts decoding the moment the first
        # function call closes. Without it, FunctionGemma greedy decoding
        # frequently loops the same call until n_predict (`<start_function_call>`
        # / `<end_function_call>` are single-token IDs and the model has high
        # bias to repeat them). Mirrors the bench harness — see bench.py:182.
        # Both KV cache and history must be reset so a stuck loop on one turn
        # doesn't leak into the next.
        llm.reset()
        out = llm(
            gen_prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=1.0,
            echo=False,
            stop=["<end_function_call>"],
        )
        elapsed = time.perf_counter() - t0
        assert isinstance(out, dict), f"unexpected llama-cpp response: {type(out)}"
        text = out["choices"][0]["text"]
        # `stop` strips the matched terminator; re-attach so the parser sees
        # a complete `<start_function_call>...<end_function_call>` block.
        if text and "<start_function_call>" in text and "<end_function_call>" not in text:
            text = text + "<end_function_call>"
        # llama-cpp's `usage` block reports prompt + decode token counts. We
        # measure wall-clock around the single `llm()` call, so `elapsed`
        # bundles prompt-eval and decode together — that's what the user
        # actually waits for. Report both counts so the interactive number
        # ("overall tok/s") is interpretable: with ~600 tokens of system
        # prompt + tool registry re-encoded every turn, decode-only rate
        # would be much higher than the bundled rate.
        usage = out.get("usage", {})
        n_in = int(usage.get("prompt_tokens", 0))
        n_out = int(usage.get("completion_tokens", 0))
        rate = (n_in + n_out) / elapsed if elapsed > 0 else 0.0
        if show_raw:
            print(f"\n[raw] {text!r}")
        calls = parse_function_calls(text)
        if not calls:
            # Out-of-scope / unparsable model output → route refusal through
            # `format_response` for parity with `chat_board.py`. Plan.md §6.
            print(f"\n[no parsable function call] {text!r}")
            print(f"  >> {format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)}")
        else:
            # Take ONLY the first call. Even with stop=["<end_function_call>"]
            # set, we re-attach the close tag above to make the parser happy,
            # which means a buggy build that emitted multiple calls before the
            # first close would still show up as len(calls) > 1. Note + take
            # the first — same contract as chat_board.py.
            if len(calls) > 1:
                print(f"\n[note: model emitted {len(calls)} calls; using the first]")
            c = calls[0]
            print(f"\n→ {json.dumps(c, ensure_ascii=False)}")
            if execute_tool is not None and table is not None:
                try:
                    result = execute_tool(c["tool"], c["args"], table)
                except KeyError as exc:
                    print(f"  [tool error] {exc}")
                    print(f"  >> {format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)}")
                else:
                    print(f"  ⤷ {json.dumps(result, ensure_ascii=False, default=str)}")
                    print(f"  >> {format_response(user_text, c['tool'], c['args'], result)}")
        print(
            f"  [prompt {n_in} + decode {n_out} tok in {elapsed:.2f}s "
            f"= {rate:.1f} tok/s overall]"
        )
        history.append({"role": "assistant", "content": text})

    if args.probe is not None:
        run_turn(args.probe)
        return 0

    print(
        "\nFunctionGemma chat — model.gguf loaded. "
        "Slash commands: /exit /quit /reset /history /raw\n",
        flush=True,
    )
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history = []
            print("[history cleared; system prompt retained]")
            continue
        if user == "/history":
            for m in base_messages + history:
                content = str(m["content"])
                print(f"  {m['role']:>9}: {content[:120]}{'…' if len(content) > 120 else ''}")
            continue
        if user == "/raw":
            show_raw = not show_raw
            print(f"[raw output: {show_raw}]")
            continue
        run_turn(user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
