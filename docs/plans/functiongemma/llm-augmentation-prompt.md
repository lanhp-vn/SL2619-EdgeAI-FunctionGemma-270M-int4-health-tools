# FunctionGemma — LLM-augmentation prompt template (M4.5)

> **Audience:** **You** (the user), pasting this into Pro Perplexity / Claude.ai
> / ChatGPT to expand `data/functiongemma/seed_conversations.jsonl` (50 hand
> seeds) → `data/functiongemma/llm_expanded_v1.jsonl` (≥ 300 rows, validator
> pass-rate ≥ 0.80 per `docs/plans/FunctionGemma/README.md` §14 M4.5).
>
> **Not a Claude Code script.** This is a paste-into-web-UI artifact. Claude
> Code never opens this file in a browser nor pastes anywhere — see §9.4.3:
> "user runs interactively, not Claude Code."

## How to use this file

1. Open Claude.ai / Perplexity Pro / ChatGPT in your browser. Pick the
   highest-quality model available (e.g. Claude Sonnet 4.6 or Opus 4.7,
   GPT-5, Sonar Reasoning Pro).
2. Copy **everything below the `===PROMPT===` line** into the web UI as one
   message. The contract, registry, fixture, and exemplars are all needed —
   the teacher cannot infer them.
3. Iterate **~40 batches** across the seven categories so the cumulative
   distribution stays roughly proportional to §9.3 (see §"Per-batch category
   target" below). Budget rationale: at the §14 ≥ 0.80 validator pass-rate
   × 10 rows / batch, 300 *passing* rows requires ~38 batches, not 30. See
   `docs/plans/FunctionGemma/README.md` §9.4.5 for the vendor
   dataset-size reference table that motivates this target.
4. Save each reply as `data/functiongemma/_incoming/<NNN>.jsonl` (one
   conversation per line, no markdown fences, no preamble — the prompt asks
   for that explicitly).
5. Run the ingest:

   ```bash
   uv run python scripts/functiongemma_ingest.py data/functiongemma/_incoming/<NNN>.jsonl
   ```

   It splits into `data/functiongemma/llm_expanded_v1.jsonl` (passing rows)
   and `data/functiongemma/quarantine.jsonl` (failing rows + per-row error
   list). PHI hits refuse to merge **before** any writes.
6. Stop when `cumulative_passed ≥ 300` AND `cumulative_pass_rate ≥ 0.80`.

## Per-batch category target (the user, not the teacher, decides this)

The teacher tends to over-produce single-turn fact_lookup. Vary the target
category between batches so the cumulative distribution roughly mirrors §9.3:

| Category | Hand-seed count | M4.5 target (×6) |
|---|---|---|
| `fact_lookup` | 12 | ~72 |
| `off_topic_refusal` | 4 | ~24 |
| `fact_absence` | 4 | ~24 |
| `parallel_call` | 6 | ~36 |
| `two_turn` | 14 | ~84 |
| `medical_advice_refusal` | 4 | ~24 |
| `tool_error_recovery` | 6 | ~36 |
| **Total** | **50** | **~300** |

Edit the `**Category target for this batch:** <name>` line in the prompt
before each paste.

> **Why 300 rows is the floor, not the ceiling.** Vendor headline (FG model
> card): the `google/mobile-actions` benchmark fine-tunes on **9 650 rows**
> for the published 58 → 85 % uplift. The Distil reproduction (§6.9) used
> **5 000 synthetic conversations** for a 10–39 % → 90–96 % uplift across
> three domains. Our 300-row target is the cookbook-tutorial-class floor
> scaled for our narrower closed-world domain (single patient × 7 read-only
> tools); §13 R6's cheap-iteration path is **M4.6 → 1 000–1 500 rows** if
> the first M5 SFT run lands G_EVAL in the 70–85 % band. Full table at
> `docs/plans/FunctionGemma/README.md` §9.4.5.

## ID namespacing

Hand seeds use `fl-001..fl-012`, `ot-001..ot-004`, etc. The teacher MUST use
**`<prefix>-101+`** so quarantine rows can be re-ingested after manual fix
without colliding with the M4 namespace (`fl-101`, `ot-101`, …). The prompt
below pins this in the contract.

---

===PROMPT===

You are an expert author of synthetic training data for a 270 M-parameter
tool-calling small language model called **FunctionGemma**. I (the operator)
have already written 50 hand-authored seed conversations; I need you to
generate **10 more** that match the EXACT same shape so I can grow the SFT
dataset.

**Category target for this batch:** `<EDIT BEFORE PASTING — pick one of
fact_lookup / off_topic_refusal / fact_absence / parallel_call / two_turn /
medical_advice_refusal / tool_error_recovery>`

## 1. Hard constraints (any violation = the row is quarantined)

1. **Output one valid JSON object per line as JSONL.** No markdown fences,
   no preamble, no commentary, no trailing prose. Just 10 lines, each one a
   parseable JSON object. I will save your reply directly to disk and run a
   strict Pydantic validator on every row.
2. **Synthetic-only.** Use ONLY the values that appear in the **Test Patient
   fixture** (§4 below). Do NOT invent providers, phone numbers, dates,
   diagnoses, dose strings, medication names, or symptom values. The PHI
   guard rejects any US phone outside `+1-555-`, any SSN-shaped string, and
   any email address. The patient's name is always "Test Patient".
3. **Tools are frozen.** You may call ONLY the seven tools listed in §3. Do
   NOT invent new tool names, do NOT add fields to existing tool argument
   schemas (`additionalProperties: false` is enforced), do NOT change the
   schema. Every emitted row's `tools` field MUST equal the JSON in §3
   verbatim — copy-paste it, do not regenerate it.
4. **`<think>` discipline.** Every `assistant` turn carries **exactly one**
   `<think>...</think>` block, and:
   - If `tool_calls` is non-empty: `content` is `<think>...</think>` —
     **block only, NOTHING after.**
   - If `tool_calls` is empty/absent: `content` is
     `<think>...</think>\n<NL answer>` — block, then a single `\n`, then a
     non-empty natural-language answer.
5. **System trigger string is verbatim.** The first message of every row
   MUST be `{"role":"system","content":"You are a model that can do function
   calling with the following functions"}` — exact-string match. Drift here
   silently disables FunctionGemma's function-calling mode.
6. **`tool_call_id` pairing.** Every `tool` message's `tool_call_id` MUST
   match an `id` from the immediately preceding assistant turn's
   `tool_calls[*].id`. Use stable ids (`call_1`, `call_2`, …) within a
   conversation. Each tool call gets a fresh id even when re-fetching the
   same tool.
7. **`tool` content is a JSON-encoded string** — not an inline JSON value.
   Wrap the payload in `JSON.stringify(...)` (i.e. the value of `content`
   is a string whose contents parse as JSON). Use ONLY values from the Test
   Patient fixture; do not invent extra keys in the response payload.
8. **`id` and `category`.** Every row MUST carry an `id` in the form
   `<prefix>-<NNN>` where `NNN ≥ 101` to avoid colliding with the hand-seed
   namespace, and `category` is one of the seven valid values. Prefixes:
   `fl-` / `ot-` / `fa-` / `pc-` / `tt-` / `ma-` / `te-`. Ids must be
   unique within the batch.
9. **`messages` length ≥ 2.** A row with one message is rejected.

## 2. Row shape (verbatim contract)

Every emitted row is one JSONL line in this exact shape:

```json
{
  "id": "fl-101",
  "category": "fact_lookup",
  "messages": [
    {"role": "system", "content": "You are a model that can do function calling with the following functions"},
    {"role": "user",   "content": "<user phrasing>"},
    {"role": "assistant",
     "content": "<think>...reasoning...</think>",
     "tool_calls": [
       {"id": "call_1", "type": "function",
        "function": {"name": "<tool_name>", "arguments": {<args>}}}
     ]},
    {"role": "tool", "name": "<tool_name>", "tool_call_id": "call_1",
     "content": "<JSON-encoded string>"},
    {"role": "assistant",
     "content": "<think>...reasoning...</think>\n<final answer>"}
  ],
  "tools": [<paste the §3 array verbatim>]
}
```

Branches by category:

| Category | Shape | Notes |
|---|---|---|
| `fact_lookup` | system → user → assistant(call) → tool → assistant(answer) | 5 messages |
| `off_topic_refusal` | system → user → assistant(answer-only) | 3 messages, NO tool call. Refusal text variant of "I answer questions from your health record only." |
| `fact_absence` | system → user → assistant(call) → tool → assistant(answer that says the field is absent) | 5 messages — the assistant DOES call the tool, then surfaces the absence in NL |
| `parallel_call` | system → user → assistant(N calls) → tool×N → assistant(answer) | 4+N messages, all tool returns BEFORE the final assistant turn |
| `two_turn` | system → user → assistant(call) → tool → assistant(answer1) → user(follow-up) → assistant(call2) → tool2 → assistant(answer2) | 9 messages, fresh `call_id` per call |
| `medical_advice_refusal` | system → user → assistant(answer-only) | 3 messages, NO tool call. Refusal text variant of "I cannot give medical advice. Please consult your clinician about that." |
| `tool_error_recovery` | system → user → assistant(call) → tool(returns error dict) → assistant(answer that surfaces the error in NL) | 5 messages — the assistant must NOT paper over the `error` field |

## 3. Tool registry — paste this `tools` array into EVERY row verbatim

> **Key order matters.** This block is byte-identical to the `tools` field
> in `data/functiongemma/seed_conversations.jsonl` (function-first,
> alphabetical, sort_keys=True — produced by `yaml.safe_dump` of the M3
> registry mirror at generation time). M4 and M4.5 rows must render to the
> same training string for the same logical tool, otherwise the chat
> template renders two different texts and the SFT signal splits. Do NOT
> reorder keys.

```json
[
  {
    "function": {
      "description": "Return the patient's most recent vital-sign measurements (heart rate, blood pressure, SpO2, body temperature, respiratory rate) along with the timestamp they were taken.",
      "name": "get_vitals",
      "parameters": {
        "additionalProperties": false,
        "properties": {},
        "required": [],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "List medications scheduled to be taken at a specific 24-hour clock time. Match is exact against the normalized HH:MM schedule.",
      "name": "get_medications_at_time",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "time_24h": {
            "description": "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'.",
            "type": "string"
          }
        },
        "required": ["time_24h"],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Look up a medication by name. Match is case-insensitive: exact match wins, otherwise a unique prefix match. An ambiguous prefix returns an error dict so the caller can re-prompt.",
      "name": "get_medication_by_name",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "name": {
            "description": "Medication name. Lookup is case-insensitive: exact match wins; otherwise a unique prefix match wins; ambiguous prefixes return an error dict.",
            "type": "string"
          }
        },
        "required": ["name"],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "List all known allergies for the patient with their severity and reaction.",
      "name": "list_allergies",
      "parameters": {
        "additionalProperties": false,
        "properties": {},
        "required": [],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Check whether a given food interacts with any of the patient's medications or dietary restrictions. Returns an `interacts` bool, the list of medication names that flag the food, and the matching dietary-restriction rule if any.",
      "name": "check_food_interaction",
      "parameters": {
        "additionalProperties": false,
        "properties": {
          "food": {
            "description": "Food name to check for medication or dietary interactions, e.g. 'grapefruit'. Case-insensitive.",
            "type": "string"
          }
        },
        "required": ["food"],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Return the earliest upcoming appointment by date and time, with provider, purpose, and location.",
      "name": "get_next_appointment",
      "parameters": {
        "additionalProperties": false,
        "properties": {},
        "required": [],
        "type": "object"
      }
    },
    "type": "function"
  },
  {
    "function": {
      "description": "Return the first listed emergency contact (name, relation, phone).",
      "name": "get_emergency_contact",
      "parameters": {
        "additionalProperties": false,
        "properties": {},
        "required": [],
        "type": "object"
      }
    },
    "type": "function"
  }
]
```

## 4. Test Patient fixture — the ONLY allowed source of values

Use ONLY these values when constructing tool responses or referring to
patient state. Do not invent.

```yaml
patient:
  name: "Test Patient"
  age: 45
  sex: "F"
  blood_type: "O+"

vitals:
  heart_rate_bpm: 72
  blood_pressure_systolic: 118
  blood_pressure_diastolic: 76
  spo2_percent: 98
  body_temperature_c: 36.7
  respiratory_rate: 16
  last_measured: "2026-04-24 08:15"

conditions:
  - name: "Hypertension"
    diagnosed_at: "2019-03-12"
    severity: "moderate"
    controlled: true
  - name: "Type 2 Diabetes"
    diagnosed_at: "2021-06-08"
    severity: "mild"
    controlled: true
  - name: "High Cholesterol"
    diagnosed_at: "2022-01-20"
    severity: "mild"
    controlled: true

allergies:
  - substance: "Penicillin"
    severity: "severe"
    reaction: "anaphylaxis"
  - substance: "Shellfish"
    severity: "moderate"
    reaction: "hives"

medications:
  - name: "Lisinopril"
    dose: "10 mg"
    schedule: "08:00"
    with_food: false
    purpose: "blood pressure control"
    avoid_foods: []
    avoid_drugs: ["Potassium supplements", "NSAIDs"]
  - name: "Metformin"
    dose: "500 mg"
    schedule: "08:00, 19:00"
    with_food: true
    purpose: "blood sugar control"
    avoid_foods: ["Alcohol"]
    avoid_drugs: []
  - name: "Atorvastatin"
    dose: "20 mg"
    schedule: "21:00"
    with_food: false
    purpose: "cholesterol control"
    avoid_foods: ["Grapefruit", "Grapefruit juice"]
    avoid_drugs: ["Erythromycin"]
  - name: "Aspirin"
    dose: "81 mg"
    schedule: "08:00"
    with_food: true
    purpose: "cardiovascular protection"
    avoid_foods: []
    avoid_drugs: ["Ibuprofen", "Warfarin"]
  - name: "Vitamin D3"
    dose: "1000 IU"
    schedule: "08:00"
    with_food: true
    purpose: "bone health"
    avoid_foods: []
    avoid_drugs: []

dietary_restrictions:
  - rule: "low sodium (under 2000 mg per day)"
    reason: "hypertension management"
  - rule: "limit added sugar"
    reason: "diabetes management"
  - rule: "no grapefruit"
    reason: "atorvastatin interaction"

appointments:
  - date: "2026-05-06"
    time: "10:30"
    provider: "Dr. Evelyn Chen"
    purpose: "quarterly diabetes check-up"
    location: "Maple Clinic, Room 204"
  - date: "2026-06-15"
    time: "14:00"
    provider: "Dr. Marcus Patel"
    purpose: "annual cardiology review"
    location: "Cardiology Dept, 3rd floor"

emergency_contacts:
  - name: "Jane Doe"
    relation: "daughter"
    phone: "+1-555-0142"

notes:
  - "Vitals within nominal range"
  - "No medication interactions flagged"
```

Tool-response payloads are projections of this fixture (e.g. `get_vitals`
returns the `vitals` map keyed by the same field names). The 7 exemplars
in §5 show the exact response shape per tool — copy those response shapes.

## 5. Exemplars (one per category) — match these shapes

The exemplars below are real rows from `seed_conversations.jsonl`. They show
the wire format end-to-end. For brevity, each exemplar OMITS the per-row
`tools` array — when you emit your rows, you MUST include the §3 array
verbatim.

### 5.1 `fact_lookup` (id `fl-001`)

```jsonl
{"id": "fl-001", "category": "fact_lookup", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "What's my heart rate?"}, {"role": "assistant", "content": "<think>User wants vitals; call get_vitals.</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}}]}, {"role": "tool", "name": "get_vitals", "tool_call_id": "call_1", "content": "{\"heart_rate_bpm\": 72, \"blood_pressure_systolic\": 118, \"blood_pressure_diastolic\": 76, \"spo2_percent\": 98, \"body_temperature_c\": 36.7, \"respiratory_rate\": 16, \"last_measured\": \"2026-04-24 08:15\"}"}, {"role": "assistant", "content": "<think>HR is 72 bpm.</think>\nYour heart rate is 72 bpm."}]}
```

### 5.2 `off_topic_refusal` (id `ot-001`)

```jsonl
{"id": "ot-001", "category": "off_topic_refusal", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "Tell me a joke."}, {"role": "assistant", "content": "<think>Off-topic; refuse without tool call.</think>\nI answer questions from your health record only."}]}
```

### 5.3 `fact_absence` (id `fa-001`)

```jsonl
{"id": "fa-001", "category": "fact_absence", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "What's my cholesterol level?"}, {"role": "assistant", "content": "<think>Cholesterol may be in vitals.</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_vitals", "arguments": {}}}]}, {"role": "tool", "name": "get_vitals", "tool_call_id": "call_1", "content": "{\"heart_rate_bpm\": 72, \"blood_pressure_systolic\": 118, \"blood_pressure_diastolic\": 76, \"spo2_percent\": 98, \"body_temperature_c\": 36.7, \"respiratory_rate\": 16, \"last_measured\": \"2026-04-24 08:15\"}"}, {"role": "assistant", "content": "<think>No cholesterol field in the response.</think>\nCholesterol level is not in your record."}]}
```

### 5.4 `parallel_call` (id `pc-001`)

```jsonl
{"id": "pc-001", "category": "parallel_call", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "What allergies do I have, and what's my BP?"}, {"role": "assistant", "content": "<think>Two independent lookups: list_allergies + get_vitals.</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "list_allergies", "arguments": {}}}, {"id": "call_2", "type": "function", "function": {"name": "get_vitals", "arguments": {}}}]}, {"role": "tool", "name": "list_allergies", "tool_call_id": "call_1", "content": "[{\"substance\": \"Penicillin\", \"severity\": \"severe\", \"reaction\": \"anaphylaxis\"}, {\"substance\": \"Shellfish\", \"severity\": \"moderate\", \"reaction\": \"hives\"}]"}, {"role": "tool", "name": "get_vitals", "tool_call_id": "call_2", "content": "{\"heart_rate_bpm\": 72, \"blood_pressure_systolic\": 118, \"blood_pressure_diastolic\": 76, \"spo2_percent\": 98, \"body_temperature_c\": 36.7, \"respiratory_rate\": 16, \"last_measured\": \"2026-04-24 08:15\"}"}, {"role": "assistant", "content": "<think>Combine both results.</think>\nAllergies: Penicillin (severe) and Shellfish (moderate). Blood pressure: 118/76 mmHg."}]}
```

### 5.5 `two_turn` (id `tt-001`)

```jsonl
{"id": "tt-001", "category": "two_turn", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "What dose of metformin?"}, {"role": "assistant", "content": "<think>Look up Metformin.</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_medication_by_name", "arguments": {"name": "metformin"}}}]}, {"role": "tool", "name": "get_medication_by_name", "tool_call_id": "call_1", "content": "{\"name\": \"Metformin\", \"dose\": \"500 mg\", \"schedule\": \"08:00, 19:00\", \"with_food\": true, \"purpose\": \"blood sugar control\", \"avoid_foods\": [\"Alcohol\"], \"avoid_drugs\": []}"}, {"role": "assistant", "content": "<think>Dose answer.</think>\nMetformin 500 mg, twice daily."}, {"role": "user", "content": "With food?"}, {"role": "assistant", "content": "<think>Re-fetch with_food field.</think>", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "get_medication_by_name", "arguments": {"name": "metformin"}}}]}, {"role": "tool", "name": "get_medication_by_name", "tool_call_id": "call_2", "content": "{\"name\": \"Metformin\", \"dose\": \"500 mg\", \"schedule\": \"08:00, 19:00\", \"with_food\": true, \"purpose\": \"blood sugar control\", \"avoid_foods\": [\"Alcohol\"], \"avoid_drugs\": []}"}, {"role": "assistant", "content": "<think>with_food=true.</think>\nYes, take Metformin with food."}]}
```

### 5.6 `medical_advice_refusal` (id `ma-001`)

```jsonl
{"id": "ma-001", "category": "medical_advice_refusal", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "Should I stop taking aspirin?"}, {"role": "assistant", "content": "<think>Medical advice; refuse without tool call.</think>\nI cannot give medical advice. Please consult your clinician about that."}]}
```

### 5.7 `tool_error_recovery` (id `te-001`)

```jsonl
{"id": "te-001", "category": "tool_error_recovery", "messages": [{"role": "system", "content": "You are a model that can do function calling with the following functions"}, {"role": "user", "content": "Tell me about that A-something pill."}, {"role": "assistant", "content": "<think>Single-letter prefix; expect ambiguity.</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_medication_by_name", "arguments": {"name": "A"}}}]}, {"role": "tool", "name": "get_medication_by_name", "tool_call_id": "call_1", "content": "{\"error\": \"ambiguous\", \"name\": \"A\", \"matches\": [\"Aspirin\", \"Atorvastatin\"]}"}, {"role": "assistant", "content": "<think>Ambiguous; ask user.</think>\nTwo medications start with \"A\": Aspirin and Atorvastatin. Which one did you mean?"}]}
```

## 6. Authoring tips (vary the surface, not the contract)

- **User phrasing.** Vary register, length, slot-fill order, and intent. Use
  fragments ("BP?"), full sentences ("Could you tell me my blood pressure
  reading?"), informal ("what's my blood sugar like"), and typo'd
  ("hartrate"). The tool call should still resolve to the right tool.
- **`<think>` content.** One short sentence is enough for fact_lookup; keep
  parallel_call thoughts a touch longer (you're describing a plan).
- **Refusal variation.** Off-topic + medical-advice rows must NOT call any
  tool. Vary the refusal phrasing — but keep the spirit of the canonical
  templates ("I answer questions from your health record only." /
  "I cannot give medical advice. Please consult your clinician about
  that."). The rest of the assistant's NL surface is yours.
- **Tool errors.** For `tool_error_recovery`, exercise the registry's actual
  error paths: `get_medication_by_name` returns `{"error":"ambiguous", ...}`
  on multi-prefix or `{"error":"no_match", ...}` on miss; `get_medications_at_time`
  returns an empty list on time slots like `12:00` or `15:00`. Don't invent
  new error codes.
- **Argument shapes.** `get_vitals`, `list_allergies`, `get_next_appointment`,
  and `get_emergency_contact` take `{}`. `get_medications_at_time` requires
  `{"time_24h": "HH:MM"}` (00:00..23:59). `get_medication_by_name` requires
  `{"name": "<string>"}`. `check_food_interaction` requires
  `{"food": "<string>"}`. Anything else is rejected by the validator.

## 7. Output

Produce **exactly 10 JSONL lines** for the category specified above.
No fences. No preamble. No commentary. No trailing prose. Save-to-disk
ready.

===END PROMPT===

## After saving the reply

```bash
# Save the assistant's reply to:
data/functiongemma/_incoming/<NNN>.jsonl   # one conversation per line

# Then run:
uv run python scripts/functiongemma_ingest.py data/functiongemma/_incoming/<NNN>.jsonl
```

The ingest CLI prints a summary like:

```
ingest: data/functiongemma/_incoming/001.jsonl
  batch: 8/10 passed (rate 0.8000)
  -> appended to data/functiongemma/llm_expanded_v1.jsonl
  -> quarantined  2 row(s) to data/functiongemma/quarantine.jsonl
  cumulative: 8/10 (rate 0.8000; M4.5 bar ≥ 0.80 — OK)
```

`cumulative_pass_rate` is the gate (§14 M4.5). Per-batch rate is just a
per-batch quality signal — a single bad batch doesn't fail M4.5 if the
running rate stays ≥ 0.80.

Quarantine triage is manual: open `quarantine.jsonl`, find the row, fix the
listed errors, and re-ingest the fixed row through the same pipeline.

> **Iron rule (§9.6):** the validator IS the gate. Do NOT loosen
> `gemma_tools.functiongemma_dataset` to make rows pass — the right fix is
> always to reject the row.
