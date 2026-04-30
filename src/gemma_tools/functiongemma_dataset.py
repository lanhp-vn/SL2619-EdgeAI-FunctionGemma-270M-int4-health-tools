"""Validator + loader for the FunctionGemma seed-conversation JSONL.

Source-of-truth contract: `docs/plans/FunctionGemma/README.md` §9.4.2 / §9.6 +
the Unsloth `FunctionGemma_(270M).ipynb` `prepare_messages_and_tools` helper
(notebook cell 23, mirrored at `docs/references/upstream/unsloth-notebooks/`).

What this module does (and intentionally does NOT do):

- Validates the **shape** of a conversation row — roles, tool-call/tool-result
  pairing, tool name in the M3 registry, `<think>` block presence, and
  argument-shape against the registry's Pydantic models.
- Does NOT *execute* any tool — that needs a `HealthTable`, a runtime concern.
  Argument validation here calls `spec.args_model.model_validate(...)` so it
  catches drift in the LLM-emitted JSON without touching the patient YAML.
- Does NOT pre-render the chat template. `render_training_text(row, tokenizer)`
  is a thin convenience that wraps `apply_chat_template(..., tools=...)` and
  strips the leading `<bos>` per §9.4.2 step 2 — it is exercised only when a
  caller (typically the training script) hands in a real tokenizer.

Assistant content shape (decided 2026-04-30, advisor pass):

- Assistant turn **with** `tool_calls`     → `content == "<think>...</think>"` (block only, no trailing NL answer).
- Assistant turn **without** `tool_calls`  → `content == "<think>...</think>\\n<NL answer>"` (block + answer separated by a single newline).

Both branches must contain **exactly one** `<think>...</think>` block. The
shape pin is intentional — it is the rule the Unsloth notebook's
`train_on_responses_only` helper depends on, and it lets the validator catch
authoring drift before training rather than at a silent loss-mask boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gemma_tools.functiongemma_tools import ToolSpec, default_registry

# Vendor-mandated developer-trigger string. Source: `scripts/functiongemma_smoke.py`
# which lifts it verbatim from cookbook cell 14. Both the system role
# (Unsloth notebook recipe, §9.4.2) and the developer role (vendor cookbook)
# carry this exact content; the FG chat template normalizes both. We follow
# the Unsloth recipe and use `system`.
SYSTEM_TRIGGER = (
    "You are a model that can do function calling with the following functions"
)

# Single `<think>...</think>` regex; the assistant.content validator counts
# matches and asserts == 1. DOTALL so a multi-line reasoning block counts.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


# --------------------------------------------------------------------------
# Pydantic models. `extra=forbid` everywhere — the same drift-catching
# discipline as `_StrictBase` in functiongemma_tools.py.
# --------------------------------------------------------------------------


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionCall(_StrictBase):
    name: str
    arguments: dict[str, Any]


class ToolCall(_StrictBase):
    """OpenAI/HF wire shape.

    `id` MUST be unique within an assistant turn. The validator cross-references
    each tool message's `tool_call_id` against the most recent assistant turn's
    set of ids — see `_validate_pairing`.
    """

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class SystemMessage(_StrictBase):
    role: Literal["system"]
    content: str


class UserMessage(_StrictBase):
    role: Literal["user"]
    content: str


class AssistantMessage(_StrictBase):
    role: Literal["assistant"]
    content: str
    # The list is *required* shape-wise but may be empty for the
    # final-answer / refusal turns. Empty list and missing key are normalized
    # to `[]` — see `_normalize_messages`.
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(_StrictBase):
    role: Literal["tool"]
    # `name` is technically optional in the HF chat template (it is backfilled
    # from `tool_call_id`), but we require it in the seed so authors think
    # about which tool the response is for.
    name: str
    tool_call_id: str
    # Stored as a string per the chat-template's expectation; the validator
    # checks it is JSON-parseable.
    content: str


# Discriminated union on `role`. Pydantic dispatches to the matching submodel
# based on the literal — bad role → ValidationError with a clear message.
Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class ToolDeclaration(_StrictBase):
    """The HF chat-template `tools=` row entry — mirrors `as_function_declarations()`.

    We do not re-validate the parameter schema shape here (that is M3's
    responsibility, gated by `tests/test_functiongemma_tools.py`). What we DO
    enforce is that `function.name` exists and is a string — every other
    consistency check (e.g. `tool_calls[*].function.name in tools[*].function.name`)
    runs in `validate_conversation`.
    """

    type: Literal["function"] = "function"
    function: dict[str, Any]


class Conversation(_StrictBase):
    """One row of `seed_conversations.jsonl`.

    Optional metadata fields (`id`, `category`, `notes`) are kept out-of-band
    so the training pipeline can drop them with a `dict.pop` before handing
    the row to `apply_chat_template`. The chat template silently ignores
    extra keys, but `extra=forbid` here lets us catch typos in the seed.
    """

    messages: list[Message] = Field(min_length=2)
    tools: list[ToolDeclaration]
    id: str | None = None
    category: str | None = None
    notes: str | None = None


# --------------------------------------------------------------------------
# Validation outcome — a flat dict that round-trips through `json.dumps` for
# the validate_file summary.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    ok: bool
    errors: tuple[str, ...]
    row_id: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class FileValidationReport:
    total: int
    passed: int
    pass_rate: float
    failures: tuple[ValidationOutcome, ...]
    category_counts: dict[str, int]
    # The threshold the caller asked for, plus the resolved pass/fail. We do
    # NOT raise on a low pass_rate — the caller (typically a pytest test)
    # asserts on `meets_threshold` so the failure surfaces alongside the
    # per-row diagnostics in `failures`.
    min_pass_rate: float
    meets_threshold: bool


# --------------------------------------------------------------------------
# I/O helpers.
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-blank line in `path`.

    Raises:
        FileNotFoundError: when `path` does not exist.
        json.JSONDecodeError: with the line number folded into the message,
            so the author can jump straight to the offending row.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{path}:{lineno}: {exc.msg}", exc.doc, exc.pos
                ) from exc


# --------------------------------------------------------------------------
# Unsloth recipe: backfill `name` on tool messages from the prior assistant
# turn's `tool_call_id` → `function.name` map. Mirrors notebook cell 23 §3b/3c.
# Mutates the input list in place AND returns it for chaining.
# --------------------------------------------------------------------------


def backfill_tool_message_names(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if tc_id and name:
                id_to_name[tc_id] = name
    for m in messages:
        if m.get("role") == "tool" and not m.get("name"):
            tc_id = m.get("tool_call_id")
            if tc_id and tc_id in id_to_name:
                m["name"] = id_to_name[tc_id]
    return messages


# --------------------------------------------------------------------------
# Per-conversation validator. Returns ValidationOutcome — never raises on
# data shape. Reserves Python exceptions for programmer error (e.g. a bug
# in the validator itself).
# --------------------------------------------------------------------------


def _count_think_blocks(content: str) -> int:
    return len(_THINK_RE.findall(content))


def _validate_assistant_content_shape(
    msg: AssistantMessage, errors: list[str], idx: int
) -> None:
    """Encode the assistant.content shape decision (see module docstring).

    With tool_calls    → `<think>...</think>` (block only, nothing after).
    Without tool_calls → `<think>...</think>\\n<NL>` (block + newline + answer).
    """
    n_think = _count_think_blocks(msg.content)
    if n_think != 1:
        errors.append(
            f"messages[{idx}] (assistant): expected exactly 1 <think> block, got {n_think}"
        )
        return
    # The block must be at the start; whatever follows is either empty or
    # `\n<answer>`. Anything before `<think>` (e.g. a stray prelude) is
    # authoring drift.
    if not msg.content.startswith("<think>"):
        errors.append(
            f"messages[{idx}] (assistant): content must start with '<think>'"
        )
        return
    # Locate the closing tag — already known to exist because n_think == 1.
    close_idx = msg.content.find("</think>")
    tail = msg.content[close_idx + len("</think>"):]
    if msg.tool_calls:
        if tail != "":
            errors.append(
                f"messages[{idx}] (assistant w/ tool_calls): content must end "
                f"immediately after </think>, got tail {tail!r}"
            )
    else:
        # Final-answer or refusal: must have `\n<NL answer>` after the block.
        if not tail.startswith("\n") or len(tail.strip()) == 0:
            errors.append(
                f"messages[{idx}] (assistant w/o tool_calls): content must "
                f"have '\\n<answer>' after </think>, got tail {tail!r}"
            )


def _validate_pairing(
    messages: list[Message], tools_by_name: dict[str, ToolSpec], errors: list[str]
) -> None:
    """Cross-message invariants: tool_calls before tool messages; ids match."""
    pending_ids: dict[str, str] = {}
    for idx, m in enumerate(messages):
        if isinstance(m, AssistantMessage):
            seen_ids: set[str] = set()
            for tc in m.tool_calls:
                if tc.id in seen_ids:
                    errors.append(
                        f"messages[{idx}]: duplicate tool_call id {tc.id!r} within turn"
                    )
                seen_ids.add(tc.id)
                pending_ids[tc.id] = tc.function.name
                spec = tools_by_name.get(tc.function.name)
                if spec is None:
                    errors.append(
                        f"messages[{idx}].tool_calls.{tc.id}: unknown tool {tc.function.name!r}"
                    )
                    continue
                # Argument-shape check via the registry's Pydantic model. We
                # *do not* execute the tool; the table-side runtime is the
                # caller's concern.
                try:
                    spec.args_model.model_validate(tc.function.arguments)
                except ValidationError as exc:
                    msgs = "; ".join(
                        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                        for err in exc.errors()
                    )
                    errors.append(
                        f"messages[{idx}].tool_calls.{tc.id} ({tc.function.name}): "
                        f"invalid arguments — {msgs}"
                    )
        elif isinstance(m, ToolMessage):
            if m.tool_call_id not in pending_ids:
                errors.append(
                    f"messages[{idx}] (tool): tool_call_id {m.tool_call_id!r} "
                    "does not match any prior assistant tool_call"
                )
                continue
            expected_name = pending_ids[m.tool_call_id]
            if m.name != expected_name:
                errors.append(
                    f"messages[{idx}] (tool): name {m.name!r} does not match "
                    f"the tool_call name {expected_name!r}"
                )
            try:
                json.loads(m.content)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"messages[{idx}] (tool): content is not valid JSON — {exc.msg}"
                )


def _validate_tools_block(
    row: Conversation, tools_by_name: dict[str, ToolSpec], errors: list[str]
) -> set[str]:
    """Tools list contains every tool used by the row; surface unused-but-not-error
    declarations are tolerated (the seed convention is "include the full registry").
    Returns the set of tool names declared on the row, for downstream cross-checks.
    """
    declared: set[str] = set()
    for i, tool in enumerate(row.tools):
        name = tool.function.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"tools[{i}]: missing or non-string function.name")
            continue
        if name in declared:
            errors.append(f"tools[{i}]: duplicate tool name {name!r}")
        declared.add(name)
        if name not in tools_by_name:
            errors.append(f"tools[{i}]: tool {name!r} not in registry")
    used: set[str] = set()
    for m in row.messages:
        if isinstance(m, AssistantMessage):
            used.update(tc.function.name for tc in m.tool_calls)
    missing = used - declared
    if missing:
        errors.append(
            f"tools: row uses {sorted(missing)} but does not declare them"
        )
    return declared


def validate_conversation(
    raw: dict[str, Any],
    registry: dict[str, ToolSpec] | None = None,
) -> ValidationOutcome:
    """Validate one parsed JSONL row.

    `registry` defaults to `default_registry()`. Pass a custom subset only for
    tests that intentionally exercise the unknown-tool branch.
    """
    if registry is None:
        registry = default_registry()
    row_id = raw.get("id") if isinstance(raw, dict) else None
    category = raw.get("category") if isinstance(raw, dict) else None

    errors: list[str] = []
    try:
        row = Conversation.model_validate(raw)
    except ValidationError as exc:
        msgs = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        return ValidationOutcome(
            ok=False,
            errors=tuple(msgs),
            row_id=str(row_id) if row_id is not None else None,
            category=str(category) if category is not None else None,
        )

    # System turn must lead with the vendor trigger. Roles are already
    # validated by the discriminated union; this is the content gate.
    first = row.messages[0]
    if not isinstance(first, SystemMessage):
        errors.append("messages[0]: must be the system turn")
    elif first.content != SYSTEM_TRIGGER:
        errors.append(
            f"messages[0] (system): content must equal SYSTEM_TRIGGER, got {first.content!r}"
        )

    # Per-message shape checks.
    for idx, m in enumerate(row.messages):
        if isinstance(m, AssistantMessage):
            _validate_assistant_content_shape(m, errors, idx)

    _validate_pairing(row.messages, registry, errors)
    _validate_tools_block(row, registry, errors)

    return ValidationOutcome(
        ok=not errors,
        errors=tuple(errors),
        row_id=row.id,
        category=row.category,
    )


def split_by_validation(
    path: Path,
    registry: dict[str, ToolSpec] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], ValidationOutcome]]]:
    """Validate every row in `path`; return (passed_rows, failed_rows_with_outcomes).

    The quarantine-aware ingest path (M4.5 — `scripts/functiongemma_ingest.py`)
    uses this to triage an LLM-augmented batch: passing rows append to
    `data/functiongemma/llm_expanded_v1.jsonl`, failing rows append to
    `data/functiongemma/quarantine.jsonl` carrying their outcome.errors so the
    user can fix-and-re-ingest rather than losing the row.

    Why return `(row, outcome)` tuples instead of just outcomes: the caller
    needs both the original row dict (to write to quarantine) AND the
    per-row error list — `validate_conversation(raw)` does NOT round-trip
    `raw` (it returns only the outcome), so we keep both side-by-side.
    """
    if registry is None:
        registry = default_registry()
    passed: list[dict[str, Any]] = []
    failed: list[tuple[dict[str, Any], ValidationOutcome]] = []
    for raw in load_jsonl(path):
        outcome = validate_conversation(raw, registry)
        if outcome.ok:
            passed.append(raw)
        else:
            failed.append((raw, outcome))
    return passed, failed


def validate_file(
    path: Path,
    registry: dict[str, ToolSpec] | None = None,
    min_pass_rate: float = 0.95,
) -> FileValidationReport:
    """Run `validate_conversation` over every row, return the aggregate report.

    Does NOT raise on a low pass rate — the caller (typically a pytest test)
    asserts on `pass_rate` so the failure surfaces with the per-row diagnostics.
    """
    if registry is None:
        registry = default_registry()
    outcomes: list[ValidationOutcome] = []
    category_counts: dict[str, int] = {}
    for raw in load_jsonl(path):
        outcome = validate_conversation(raw, registry)
        outcomes.append(outcome)
        if outcome.category:
            category_counts[outcome.category] = category_counts.get(outcome.category, 0) + 1
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.ok)
    pass_rate = passed / total if total else 0.0
    failures = tuple(o for o in outcomes if not o.ok)
    return FileValidationReport(
        total=total,
        passed=passed,
        pass_rate=pass_rate,
        failures=failures,
        category_counts=category_counts,
        min_pass_rate=min_pass_rate,
        meets_threshold=pass_rate >= min_pass_rate,
    )


# --------------------------------------------------------------------------
# Optional render helper — kept thin. Imports `transformers` lazily so tests
# that only exercise the validator don't pull a 500 MB import graph.
# --------------------------------------------------------------------------


def render_training_text(row: dict[str, Any], tokenizer: Any) -> str:
    """Apply the chat template + strip the leading `<bos>` per §9.4.2.

    `tokenizer` is typed `Any` because the `transformers` import is lazy and
    we do not want this module to require `transformers` at import time. Pass
    `AutoTokenizer.from_pretrained(...)` from the caller.
    """
    messages = row["messages"]
    tools = row.get("tools")
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, expected str"
        )
    # Same fix the smoke script uses — without it `SFTTrainer` re-prepends a
    # second <bos> at tokenize time → silent training-data corruption.
    return rendered.removeprefix("<bos>")
