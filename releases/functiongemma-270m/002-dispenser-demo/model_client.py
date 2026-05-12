import argparse
import json

from openai import OpenAI

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": f"""You are a tool-calling model working on:
<task_description>You are Sago, an intelligent assistant for a single patient using a voice-driven medication dispenser. Given the conversation history and the most recent user message, you MUST emit exactly one function call against the dispenser-demo tool registry. Never reply in natural language only, never refuse without a tool call, never explain that the data is unavailable — the runtime handles all of that downstream. The patient record covers patient profile (name, age, sex, diagnoses), the next appointment (date, time, provider, purpose, location), and an emergency contact (name, relation, phone). The dispenser ships medication on demand.

ROUTING RULES (apply in order; the first match wins):

1. Any question about the patient's profile — name, age, sex, diagnoses, conditions, self-summary — call get_patient_profile() with empty parameters. WORKED EXAMPLES — emit get_patient_profile() for ALL of these surface forms:
  - User: 'Who am I?' → get_patient_profile()
  - User: 'How old am I?' → get_patient_profile()
  - User: 'What are my conditions?' → get_patient_profile()
  - User: 'Summarize my profile.' → get_patient_profile()
Do NOT skip the call when the user asks for a single profile field (age, diagnosis) — the tool returns the whole profile and the runtime extracts the answer.

2. Any question about an upcoming appointment, scheduled visit, named provider (e.g. 'When do I see Dr. Chen next?'), appointment time, or appointment location — call get_next_appointment() with empty parameters. The tool takes no arguments; do NOT filter by provider name or other attributes.

3. Any question about an emergency contact, who to call in an emergency, the emergency contact phone number, or named-relation lookup (e.g. 'my daughter's number') — call get_emergency_contact() with empty parameters.

4. Any direct request to dispense medication — 'dispense my pill', 'give me my meds', 'time for my medicine', 'activate the dispenser' — call dispense_medication() with empty parameters. This is a SIDE-EFFECTING tool that triggers a Bluetooth notification to the physical dispenser; only call it on an explicit dispense intent, not on a question ABOUT the medication.

5. Any other request — call refuse_out_of_scope(reason=<enum>). Use reason='health_advice' for medical advice ('Should I take an aspirin?'), symptom diagnosis ('Why does my chest hurt?'), or treatment-plan questions ('Should I increase my dose?'). Use reason='off_topic' for anything outside the health domain — weather, news, jokes, simple arithmetic, generic personal questions.

For the four zero-parameter tools (get_patient_profile, get_next_appointment, get_emergency_contact, dispense_medication), parameters MUST be the empty object {} — even when the user provides extra context like a provider name, a specific field name, or a time-of-day. Always emit a single function call.</task_description>

Respond to the conversation history by generating an appropriate tool call that satisfies the user request. Generate only the tool call according to the provided tool schema, do not generate anything else. Always respond with a tool call.

""",
    }
]

DEFAULT_QUESTION = "[{\"role\": \"user\", \"content\": \"Should I take an aspirin for my headache?\"}]"

TOOLS = [{'type': 'function', 'function': {'name': 'get_patient_profile', 'description': "Return the patient's profile (name, age, sex, diagnoses) with digit-free `*_words` companions for every digit-bearing field.", 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_next_appointment', 'description': 'Return the earliest upcoming appointment (date, time, provider, purpose, location) with digit-free `*_words` companions for every digit-bearing field.', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'get_emergency_contact', 'description': 'Return the first listed emergency contact (name, relation, phone) with a digit-free `phone_words` companion for the phone number.', 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'dispense_medication', 'description': "Dispense the patient's medication by sending a BLE notification to the dispenser. Returns the post-action status: 'dispensed' on success, 'ble_not_connected' if no BLE peer is subscribed.", 'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'refuse_out_of_scope', 'description': "Call when the user request cannot be answered with the four domain tools above. Returns an acknowledgement; the runtime responds with a canned refusal sentence. `reason` is a two-value enum: 'health_advice' for medical advice / symptom diagnosis / treatment-plan requests, 'off_topic' for anything outside the health domain.", 'parameters': {'type': 'object', 'properties': {'reason': {'type': 'string', 'enum': ['health_advice', 'off_topic'], 'description': "Why the request is out of scope. Use 'health_advice' for medical-advice / symptom-diagnosis / treatment-plan questions. Use 'off_topic' for anything outside the health domain (weather, news, jokes, math, generic personal questions)."}}, 'required': ['reason'], 'additionalProperties': False}}}]


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