# FunctionGemma seed-authoring recipe

Companion to `docs/plans/FunctionGemma/README.md` §9.4.2 / §9.6. This file
captures the **operational recipe** for adding hand-authored seed
conversations to `data/functiongemma/seed_conversations.jsonl` — what shape
each row must have, how to validate it, and how to keep it in sync with the
M3 tool registry.

> Quick start: `uv run python scripts/build_functiongemma_seeds.py` to
> regenerate the JSONL after editing `_build_conversations()` in the same
> script. Then `uv run pytest tests/test_functiongemma_dataset.py` to validate.

## 1. Where the data lives

| Path | Role |
|---|---|
| `scripts/build_functiongemma_seeds.py` | **Source of truth** — every seed row is a Python literal in `_build_conversations()`. Edit here. |
| `data/functiongemma/seed_conversations.jsonl` | Generator output. Committed for diff visibility but never edited by hand. |
| `data/functiongemma/tools_v1.yaml` | M3 registry mirror — the per-row `tools` block is stamped from this file at generation time. |
| `src/gemma_tools/functiongemma_dataset.py` | Pydantic validator + `validate_file()` + the `<think>` shape gate. |
| `tests/test_functiongemma_dataset.py` | Acceptance gate (G_DATASET_SHAPE) — pass-rate, taxonomy counts, role/JSON checks. |
| `scripts/pre-commit-functiongemma.py` | Phase B PHI guard. Manual run — not auto-installed. |

The split (Python generator → JSONL artifact) is deliberate:

- The seed file is plain JSONL so consumers (`SFTTrainer`, manual review)
  do not need to import the repo.
- The generator is the authoring surface — fixing a typo in the system
  trigger string or a tool description propagates correctly across all 50
  rows from a single edit point.
- The `--check` mode keeps the JSONL artifact and the generator in sync; CI
  can wire it up later if needed.

## 2. Row shape (verbatim contract)

Every row is one JSON line in this shape:

```json
{
  "id": "fl-001",
  "category": "fact_lookup",
  "messages": [...],
  "tools": [...]
}
```

- `id` is a stable handle (`<category-prefix>-<NNN>`) so a failing test can
  point at one specific row.
- `category` MUST be one of the seven §9.3 categories
  (`fact_lookup`, `off_topic_refusal`, `fact_absence`, `parallel_call`,
  `two_turn`, `medical_advice_refusal`, `tool_error_recovery`).
- `messages` is the HF chat-template message list (rules §3 below).
- `tools` is the **full 7-tool registry** for every row (decision §4 below).

## 3. Message rules

### 3.1 Roles

`role ∈ {system, user, assistant, tool}` — the validator uses a
discriminated union on `role` so an unknown role surfaces with a clear
Pydantic error.

### 3.2 First message — system trigger

The first message MUST be:

```json
{"role": "system", "content": "You are a model that can do function calling with the following functions"}
```

Exact-string match. Drift here silently disables FunctionGemma's
function-calling mode at training time.

### 3.3 Assistant content shape

Every `assistant` turn carries exactly one `<think>...</think>` block. The
shape splits into two branches:

| Branch | When | Content shape |
|---|---|---|
| **With `tool_calls`** | The assistant is dispatching one or more tool calls (no NL surfaced yet) | `<think>...</think>` — block only, NOTHING after |
| **Without `tool_calls`** | The assistant is emitting the final NL answer (or a refusal) | `<think>...</think>\n<NL answer>` — block, then a `\n`, then a non-empty answer |

Why the split:

- The Unsloth `train_on_responses_only(...)` helper masks loss to the
  `<start_of_turn>model\n` … `<end_of_turn>` span. A stray NL after
  `</think>` on a tool-call turn would emit a stray text turn at training
  time — visible to loss but never to the model at inference.
- The validator (`_validate_assistant_content_shape`) enforces both
  branches; mutation tests in `test_functiongemma_dataset.py` cover the
  four drifts (`missing_think`, `two_think`, `tail_after_call`,
  `no_nl_after`).

### 3.4 Tool calls — wire shape

Per §9.4.2:

```json
{
  "role": "assistant",
  "content": "<think>…</think>",
  "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "...", "arguments": {...}}}
  ]
}
```

Rules:
- `id` MUST be unique within an assistant turn. The validator detects
  duplicates as authoring errors.
- `type` is always `"function"`.
- `function.name` MUST exist in `default_registry()`. Unknown names raise.
- `function.arguments` MUST validate against the matching Pydantic model
  from `gemma_tools.functiongemma_tools` (e.g. `TimeArgs`, `NameArgs`).
  Unknown keys are rejected (`extra=forbid`).

### 3.5 Tool messages

```json
{"role": "tool", "name": "<tool_name>", "tool_call_id": "call_1", "content": "<json string>"}
```

- `tool_call_id` MUST match an `id` from a prior assistant `tool_calls[*]`.
- `name` MUST match the corresponding `function.name`.
- `content` is a JSON-encoded **string** (not an inline JSON value). The
  validator runs `json.loads(content)` and rejects non-parseable payloads.
- The seed authoring helper `_tool(call_id, name, payload)` in the build
  script handles the JSON encoding for you.

## 4. Per-row `tools` convention

**Decision:** every row carries the **full 7-tool registry**.

Why (vs. a per-row used-subset):

- The training signal for `off_topic_refusal` and `medical_advice_refusal`
  rows is *"tools are available, but you should not call any of them"*.
  The full registry makes that lesson concrete.
- The cost is a fixed ~3 KB of duplication per row (the JSON-Schema'd
  declarations) — irrelevant at 50 rows. With the model's 32 768-token
  trained context (HF model card) and a representative parallel-call row
  rendering well under 4 KiB through `apply_chat_template`, full-registry
  per-row stays comfortably inside `max_seq_length=4096`.
- The build script stamps the registry from `tools_v1.yaml` at
  generation time, so a description tweak in `functiongemma_tools.py` →
  `tools_v1.yaml` regen → seed regen propagates cleanly.

If a future need warrants per-row subsetting (e.g. an M4.5 LLM-augmented
expansion that ships 3 000 rows and the byte cost matters), update the
`test_per_row_tools_block_is_full_registry` test alongside the change so
the convention shift is reviewed deliberately.

## 5. Adding a new seed row

1. Edit `_build_conversations()` in `scripts/build_functiongemma_seeds.py`.
   Use the `_system()`, `_user()`, `_assistant_call()`, `_assistant_answer()`,
   `_tool()` helpers — they enforce the row shape so you only write content.
2. Pick an `id` in the form `<prefix>-<NNN>` matching the category:
   `fl-` (fact_lookup), `ot-` (off_topic), `fa-` (fact_absence), `pc-`
   (parallel_call), `tt-` (two_turn), `ma-` (medical_advice), `te-`
   (tool_error_recovery).
3. Tool-result fixtures: re-use the constants at the top of the script
   (`_VITALS`, `_MORNING_MEDS`, `_MED_LISINOPRIL`, …). They are pinned to the
   `data/health_table_v1.yaml` snapshot the seed was authored against.
4. Run `uv run python scripts/build_functiongemma_seeds.py` to regenerate
   the JSONL.
5. Run `uv run pytest tests/test_functiongemma_dataset.py` to validate
   shape + taxonomy + tool-call-id matching.
6. If you added a category-target row, update `_TAXONOMY_TARGETS` in the
   test alongside the change.

## 6. Worked examples

### 6.1 Single-turn fact lookup

5 messages: system, user, assistant(call), tool, assistant(answer).

```python
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
}
```

### 6.2 Off-topic refusal (no tool call)

3 messages: system, user, assistant(answer-only).

```python
{
    "id": "ot-001",
    "category": "off_topic_refusal",
    "messages": [
        _system(),
        _user("Tell me a joke."),
        _assistant_answer(
            "Off-topic; refuse without tool call.",
            "I answer questions from your health record only.",
        ),
    ],
}
```

The assistant content here is `<think>...</think>\nI answer questions...`
— the `_assistant_answer` helper composes both parts.

### 6.3 Parallel call

6 messages: system, user, assistant(2 calls), tool×2, assistant(answer).
The two tool messages MUST appear before the final assistant turn —
interleaving them with a NL turn would break the wire format.

```python
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
}
```

### 6.4 Two-turn slot-fill

9 messages: system, user, assistant(call), tool, assistant(answer1),
user(follow-up), assistant(call), tool, assistant(answer2). Each tool
call gets a fresh `id` (`call_1`, `call_2`) — even when re-fetching the
same tool — so the `tool_call_id` → name map stays unambiguous.

```python
{
    "id": "tt-001",
    "category": "two_turn",
    "messages": [
        _system(),
        _user("What dose of metformin?"),
        _assistant_call("Look up Metformin.", [("call_1", "get_medication_by_name", {"name": "metformin"})]),
        _tool("call_1", "get_medication_by_name", _MED_METFORMIN),
        _assistant_answer("Dose answer.", "Metformin 500 mg, twice daily."),
        _user("With food?"),
        _assistant_call("Re-fetch with_food field.", [("call_2", "get_medication_by_name", {"name": "metformin"})]),
        _tool("call_2", "get_medication_by_name", _MED_METFORMIN),
        _assistant_answer("with_food=true.", "Yes, take Metformin with food."),
    ],
}
```

## 7. Keeping tool schemas in sync

When the M3 registry changes (`src/gemma_tools/functiongemma_tools.py`):

1. Regenerate `data/functiongemma/tools_v1.yaml` per the comment at the top
   of that file (`uv run python -c '...as_function_declarations()...'`).
2. Run `uv run pytest tests/test_functiongemma_tools.py` — the
   `test_tools_yaml_matches_registry` gate will fail if step 1 is missed.
3. Re-run `scripts/build_functiongemma_seeds.py` to refresh the per-row
   `tools` blocks in the JSONL.
4. Re-run `tests/test_functiongemma_dataset.py` to confirm every existing
   tool call still validates against the new arg-models.

If step 4 fails, an existing seed row's arguments became invalid —
either fix the row, or note the deviation in the M3 registry change and
update `_TAXONOMY_TARGETS` if the seed shape needs a new category.

## 8. Avoiding real PHI

**Hard rule (`docs/plans/FunctionGemma/README.md` §9.5):** training data
references the `Test Patient` synthetic fixture only.

The PHI guard at `scripts/pre-commit-functiongemma.py` flags:

- US Social Security Numbers (`\d{3}-\d{2}-\d{4}`).
- US phone numbers outside the synthetic `+1-555-` range.
- Email addresses.

It does NOT match real-provider-name heuristics — that's a false-positive
arms race ("Dr. Evelyn Chen" is also a real-looking name). Human review
is the second line of defense.

To run before committing:

```bash
uv run python scripts/pre-commit-functiongemma.py data/functiongemma/
```

The scanner is also exercised by `tests/test_pre_commit_phi_scanner.py`
against `data/functiongemma/seed_conversations.jsonl` — the regression
gate is part of normal `pytest`.

To wire up as a git pre-commit hook (optional — not auto-installed):

```bash
ln -s ../../scripts/pre-commit-functiongemma.py .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 9. The validator contract (G_DATASET_SHAPE)

`validate_file(path)` returns a `FileValidationReport`:

- `total` — rows seen.
- `passed` — rows that returned `ok=True`.
- `pass_rate` — `passed / total`.
- `failures` — tuple of `ValidationOutcome` for the bad rows, with
  per-row `errors`.
- `category_counts` — per-`category` row counts, for the taxonomy gate.

For M4 (hand-authored seed): pass rate is asserted **= 1.0**, not just
≥ 0.95 — anything less is authoring drift, and 50 rows is small enough
to fix all of them. M4.5 (LLM-augmented expansion) holds the looser
≥ 0.80 bar because the LLM teacher will produce some malformed rows by
construction.
