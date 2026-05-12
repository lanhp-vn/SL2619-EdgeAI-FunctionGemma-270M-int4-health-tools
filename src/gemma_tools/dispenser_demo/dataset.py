"""Dispenser-demo seed JSONL validator.

Mirrors `gemma_tools.functiongemma.dataset` for the conversation-shape checks
(roles, `<think>` discipline, tool_call/tool_message pairing, registry
membership) and ADDS the dispenser-specific tool-boundary invariant: every
digit-bearing key in a tool response (the `content` of a `tool` message) has
a digit-free `*_words` sibling.

Scope reminder (plan §10): this validator does NOT assert anything about the
LLM's free narration. Free-text assistant content can contain digits — the
invariant lives at the tool boundary so the TTS layer (which renders from
the `*_words` fields, not the model's prose) is unconditionally word-only.

Public API:

- `validate_conversation(raw)` → `ValidationOutcome` (errors, ok flag).
- `validate_file(path, min_pass_rate=1.0)` → `FileValidationReport`.
- `find_word_only_violations(obj, path)` → `list[str]`. Pure invariant
  walker; reused by `tests/dispenser_demo/test_tools_word_only.py` to
  assert against live tool outputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gemma_tools.dispenser_demo.tools import ToolSpec, default_registry

# Same vendor trigger as FG iter-001 — proven training pattern; the actual
# Sago routing prompt lives in Distil's `task_description.json` and in the
# on-board `model_client.py`, not in the seed system turn.
SYSTEM_TRIGGER = "You are a model that can do function calling with the following functions"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DIGIT_RE = re.compile(r"\d")

ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})

# Keys ending in `_words` are themselves the digit-free companions and are
# never required to carry a nested `*_words_words` companion.
_WORDS_SUFFIX = "_words"


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FunctionCall(_StrictBase):
    name: str
    arguments: dict[str, Any]


class ToolCall(_StrictBase):
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
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(_StrictBase):
    role: Literal["tool"]
    name: str
    tool_call_id: str
    content: str  # JSON-encoded string; validated downstream.


Message: TypeAlias = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class ToolDeclaration(_StrictBase):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class Conversation(_StrictBase):
    messages: list[Message] = Field(min_length=2)
    tools: list[ToolDeclaration]
    id: str | None = None
    category: str | None = None
    notes: str | None = None


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
    min_pass_rate: float
    meets_threshold: bool


# --------------------------------------------------------------------------
# I/O helpers.
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-blank line in `path`."""
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
# Tool-boundary `*_words`-companion invariant. Pure function — no I/O —
# reused by the live-tool tests in `test_tools_word_only.py`.
# --------------------------------------------------------------------------


def _value_has_digits(value: Any) -> bool:
    """`True` if `value` carries a digit in any leaf position.

    `int` / `float` (non-bool) are digit-bearing by definition. Strings are
    digit-bearing if any `\\d` matches. Lists are digit-bearing if any
    element is. Dicts and other types are NOT considered digit-bearing here
    — nested dicts are walked separately by `find_word_only_violations`.
    """
    if isinstance(value, bool):
        return False  # bool is a subclass of int; reject explicitly.
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        return bool(_DIGIT_RE.search(value))
    if isinstance(value, list):
        return any(_value_has_digits(v) for v in value)
    return False


def find_word_only_violations(obj: Any, path: str = "<root>") -> list[str]:
    """Walk `obj` and report any tool-boundary invariant violations.

    The invariant: in every `dict`, for every key `k` whose value carries a
    digit, there exists a sibling key `f"{k}{_WORDS_SUFFIX}"` whose value is
    a string with no digits.

    Returns an empty list when the invariant holds. Each entry in the
    non-empty list is a self-locating error message like
    `<root>.tool_response.age: digit-bearing but no 'age_words' sibling`.

    The walk is recursive: nested dicts and lists are descended; values
    inside `*_words` siblings are themselves checked for the absence of
    digits (so a buggy wordform.py that leaks a digit surfaces here).
    """
    errors: list[str] = []
    _walk(obj, path, errors)
    return errors


def _walk(obj: Any, path: str, errors: list[str]) -> None:
    if isinstance(obj, dict):
        # Step 1: every `*_words` value must be a digit-free string.
        for k, v in obj.items():
            if k.endswith(_WORDS_SUFFIX):
                if not isinstance(v, str):
                    errors.append(
                        f"{path}.{k}: must be a string, got {type(v).__name__}"
                    )
                elif _DIGIT_RE.search(v):
                    errors.append(
                        f"{path}.{k}: word-form must be digit-free, got {v!r}"
                    )
        # Step 2: every digit-bearing key must have a `<key>_words` sibling.
        for k, v in obj.items():
            if k.endswith(_WORDS_SUFFIX):
                continue
            companion = f"{k}{_WORDS_SUFFIX}"
            if _value_has_digits(v) and companion not in obj:
                errors.append(
                    f"{path}.{k}: digit-bearing but no {companion!r} sibling"
                )
        # Step 3: recurse into nested structures (skip `*_words` values; they
        # are leaf strings by contract).
        for k, v in obj.items():
            if k.endswith(_WORDS_SUFFIX):
                continue
            _walk(v, f"{path}.{k}", errors)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk(item, f"{path}[{i}]", errors)
    # Other types are leaves; nothing further to check.


# --------------------------------------------------------------------------
# Per-conversation validator.
# --------------------------------------------------------------------------


def _count_think_blocks(content: str) -> int:
    return len(_THINK_RE.findall(content))


def _validate_assistant_content_shape(
    msg: AssistantMessage, errors: list[str], idx: int
) -> None:
    """Encode the assistant-content shape: `<think>...</think>` block-only
    when tool_calls is non-empty; `<think>...</think>\\n<NL>` otherwise.

    Matches `gemma_tools.functiongemma.dataset._validate_assistant_content_shape`
    byte-for-byte — same training-data discipline.
    """
    n_think = _count_think_blocks(msg.content)
    if n_think != 1:
        errors.append(
            f"messages[{idx}] (assistant): expected exactly 1 <think> block, got {n_think}"
        )
        return
    if not msg.content.startswith("<think>"):
        errors.append(
            f"messages[{idx}] (assistant): content must start with '<think>'"
        )
        return
    close_idx = msg.content.find("</think>")
    tail = msg.content[close_idx + len("</think>"):]
    if msg.tool_calls:
        if tail != "":
            errors.append(
                f"messages[{idx}] (assistant w/ tool_calls): content must end "
                f"immediately after </think>, got tail {tail!r}"
            )
    else:
        if not tail.startswith("\n") or len(tail.strip()) == 0:
            errors.append(
                f"messages[{idx}] (assistant w/o tool_calls): content must "
                f"have '\\n<answer>' after </think>, got tail {tail!r}"
            )


def _validate_pairing(
    messages: list[Message], tools_by_name: dict[str, ToolSpec], errors: list[str]
) -> None:
    """Cross-message invariants:

    - Every `tool` message references a prior `assistant.tool_calls[*].id`.
    - Tool name on the tool message matches the call.
    - Tool message content parses as JSON.
    - The parsed JSON satisfies the `*_words`-companion invariant.
    - Each `tool_call` validates against the registry's Pydantic args model.
    """
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
                parsed = json.loads(m.content)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"messages[{idx}] (tool): content is not valid JSON — {exc.msg}"
                )
                continue
            for violation in find_word_only_violations(
                parsed, path=f"messages[{idx}].content"
            ):
                errors.append(violation)


def _validate_tools_block(
    row: Conversation, tools_by_name: dict[str, ToolSpec], errors: list[str]
) -> set[str]:
    """Tools list contains every tool used by the row.

    The seed convention is "include the full registry on every row", so an
    unused declaration is tolerated; an undeclared but used tool is an error.
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

    `registry` defaults to `default_registry()`; pass a custom subset only
    for tests that exercise the unknown-tool branch.
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

    first = row.messages[0]
    if not isinstance(first, SystemMessage):
        errors.append("messages[0]: must be the system turn")
    elif first.content != SYSTEM_TRIGGER:
        errors.append(
            f"messages[0] (system): content must equal SYSTEM_TRIGGER, got {first.content!r}"
        )

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


def validate_file(
    path: Path,
    registry: dict[str, ToolSpec] | None = None,
    min_pass_rate: float = 1.0,
) -> FileValidationReport:
    """Run `validate_conversation` over every row, return the aggregate report.

    Default `min_pass_rate=1.0` — the dispenser_demo seeds are hand-authored
    and small (40 rows), so any failure is a regression to fix at authoring
    time, not a soft signal to track.
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
