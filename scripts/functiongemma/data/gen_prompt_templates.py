#!/usr/bin/env python3
"""Generate prompt-prefix.txt and prompt-suffix.txt for FunctionGemma on SL2619.

These pre-rendered templates allow on-board inference without requiring the
HF tokenizer. The board script (fg-chat-board.py) concatenates:
  prefix + user_question + suffix + [model output]

The templates embed the system prompt + tool registry in prefix, and the
assistant turn opening in suffix. User questions are inserted between them.

Usage (host):
  uv run python scripts/gen_prompt_templates.py \
    --tokenizer model/ \
    --output-dir /tmp/fg_board_files/
"""
import argparse
import sys
from pathlib import Path

SYSTEM_PROMPT = [
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

TOOLS = [
    {"type": "function", "function": {"name": "get_vitals", "description": "Return the patient's most recent vital-sign measurements (heart rate, blood pressure, SpO2, body temperature, respiratory rate) along with the timestamp they were taken.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medications_at_time", "description": "List medications scheduled to be taken at a specific 24-hour clock time. Match is exact against the normalized HH:MM schedule.", "parameters": {"type": "object", "properties": {"time_24h": {"description": "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'.", "type": "string"}}, "required": ["time_24h"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medication_by_name", "description": "Look up a medication by name. Match is case-insensitive: exact match wins, otherwise a unique prefix match. An ambiguous prefix returns an error dict so the caller can re-prompt.", "parameters": {"type": "object", "properties": {"name": {"description": "Medication name. Lookup is case-insensitive: exact match wins; otherwise a unique prefix match wins; ambiguous prefixes return an error dict.", "type": "string"}}, "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_allergies", "description": "List all known allergies for the patient with their severity and reaction.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_food_interaction", "description": "Check whether a given food interacts with any of the patient's medications or dietary restrictions. Returns an `interacts` bool, the list of medication names that flag the food, and the matching dietary-restriction rule if any.", "parameters": {"type": "object", "properties": {"food": {"description": "Food name to check for medication or dietary interactions, e.g. 'grapefruit'. Case-insensitive.", "type": "string"}}, "required": ["food"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_next_appointment", "description": "Return the earliest upcoming appointment by date and time, with provider, purpose, and location.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_emergency_contact", "description": "Return the first listed emergency contact (name, relation, phone).", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate prompt-prefix.txt and prompt-suffix.txt for SL2619."
    )
    p.add_argument("--tokenizer", required=True,
                   help="HF model directory with chat_template.jinja (e.g., model/)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for prompt-prefix.txt and prompt-suffix.txt")
    args = p.parse_args(argv)

    tokenizer_dir = Path(args.tokenizer)
    output_dir = Path(args.output_dir)

    if not tokenizer_dir.exists():
        print(f"tokenizer dir not found: {tokenizer_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    print(f"[gen] loading tokenizer from {tokenizer_dir}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    dummy_user = "<PLACEHOLDER_USER_MESSAGE>"
    messages = [*SYSTEM_PROMPT, {"role": "user", "content": dummy_user}]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(full_prompt, str):
        raise TypeError(f"apply_chat_template returned {type(full_prompt).__name__}")

    full_prompt = full_prompt.removeprefix("<bos>")

    idx = full_prompt.find(dummy_user)
    if idx < 0:
        print(f"ERROR: placeholder not found in rendered prompt", file=sys.stderr)
        print(f"Full prompt:\n{full_prompt}", file=sys.stderr)
        return 2

    prefix = full_prompt[:idx]
    suffix = full_prompt[idx + len(dummy_user):]

    prefix_path = output_dir / "prompt-prefix.txt"
    suffix_path = output_dir / "prompt-suffix.txt"

    prefix_path.write_text(prefix, encoding="utf-8")
    suffix_path.write_text(suffix, encoding="utf-8")

    print(f"[gen] prefix: {len(prefix)} bytes → {prefix_path}", file=sys.stderr)
    print(f"[gen] suffix: {len(suffix)} bytes → {suffix_path}", file=sys.stderr)
    print(f"[gen] done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
