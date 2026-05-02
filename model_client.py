import argparse
import json

from openai import OpenAI

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": f"""You are a tool-calling model working on:
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

DEFAULT_QUESTION = "[{\"role\": \"user\", \"content\": \"When do I see Dr. Chen next?\"}]"

TOOLS = [{'type': 'function', 'function': {'name': 'get_vitals', 'description': "Return the patient's most recent vital-sign measurements (heart rate, blood pressure, SpO2, body temperature, respiratory rate) along with the timestamp they were taken.", 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_medications_at_time', 'description': 'List medications scheduled to be taken at a specific 24-hour clock time. Match is exact against the normalized HH:MM schedule.', 'parameters': {'type': 'object', 'properties': {'time_24h': {'description': "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'.", 'type': 'string'}}, 'required': ['time_24h'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_medication_by_name', 'description': 'Look up a medication by name. Match is case-insensitive: exact match wins, otherwise a unique prefix match. An ambiguous prefix returns an error dict so the caller can re-prompt.', 'parameters': {'type': 'object', 'properties': {'name': {'description': 'Medication name. Lookup is case-insensitive: exact match wins; otherwise a unique prefix match wins; ambiguous prefixes return an error dict.', 'type': 'string'}}, 'required': ['name'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'list_allergies', 'description': 'List all known allergies for the patient with their severity and reaction.', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'check_food_interaction', 'description': "Check whether a given food interacts with any of the patient's medications or dietary restrictions. Returns an `interacts` bool, the list of medication names that flag the food, and the matching dietary-restriction rule if any.", 'parameters': {'type': 'object', 'properties': {'food': {'description': "Food name to check for medication or dietary interactions, e.g. 'grapefruit'. Case-insensitive.", 'type': 'string'}}, 'required': ['food'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_next_appointment', 'description': 'Return the earliest upcoming appointment by date and time, with provider, purpose, and location.', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_emergency_contact', 'description': 'Return the first listed emergency contact (name, relation, phone).', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}]


class DistilLabsLLM(object):
    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 11434):
        self.model_name = model_name
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)

    def invoke(self, conversation_history: list[dict]) -> dict | str:
        """Send *full* conversation history to the SLM and return a parsed
        function-call dict ``{"name": ..., "arguments": ...}`` or an error
        string if no valid tool call could be extracted.
        """
        messages = SYSTEM_PROMPT + conversation_history

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            tools=TOOLS,
            tool_choice="required",
            
        )
        response = chat_response.choices[0].message

        # --- Path A: proper tool_calls in the response ---
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return {"name": fn.name, "parameters": arguments}

        # --- Path B: model returned JSON in content (fallback) ---
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args)
                    return {"name": parsed["name"], "parameters": args}
            except (json.JSONDecodeError, KeyError):
                pass

        return f"No valid tool call in SLM response, model returned {response}"


def to_openai_arguments(tool_call, i):
    if not isinstance(tool_call["function"]["arguments"], str):
        tool_call["function"]["arguments"] = json.dumps(tool_call["function"]["arguments"])
        tool_call["id"] = str(i)
    return tool_call


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, required=False)
    parser.add_argument("--api-key", type=str, default="EMPTY", required=False)
    parser.add_argument("--model", type=str, default="model", required=False)
    parser.add_argument("--port", type=int, default=11434, required=False)
    args = parser.parse_args()

    conversation = json.loads(args.question)
    for msg in conversation:
        if "tool_calls" in msg:
            msg["tool_calls"] = [to_openai_arguments(t, i) for i, t in enumerate(msg["tool_calls"])]

    client = DistilLabsLLM(model_name=args.model, api_key=args.api_key, port=args.port)

    answer = client.invoke(conversation)
    print(json.dumps(answer) if not isinstance(answer, str) else answer)