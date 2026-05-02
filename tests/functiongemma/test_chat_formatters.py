"""Regression tests for `scripts/functiongemma/chat.py:format_response`.

The model can emit a tool call with missing or invalid arguments — the
runtime resolves that into an `{"error": ...}` dict from the tool. Each
per-tool formatter must handle that dict shape gracefully or it crashes
the REPL (was: `TypeError: string indices must be integers, not 'str'`
when iterating `{"error": "invalid_arguments", ...}` as if it were a list
of medication dicts).

Mirrors the same fix in `scripts/functiongemma/deploy/chat_board.py` —
both formatter paths must stay in sync.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CHAT = _REPO / "scripts" / "functiongemma" / "chat.py"
_CHAT_BOARD = _REPO / "scripts" / "functiongemma" / "deploy" / "chat_board.py"


def _load(path: Path, modname: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def chat_mod() -> ModuleType:
    return _load(_CHAT, "fg_chat")


@pytest.fixture(scope="module")
def chat_board_mod() -> ModuleType:
    return _load(_CHAT_BOARD, "fg_chat_board")


# ---- error dicts the tool registry returns when args are missing/invalid ----


@pytest.mark.parametrize(
    ("tool", "args", "result"),
    [
        # The exact crash payload from the user's REPL session 2026-05-02.
        (
            "get_medications_at_time",
            {},
            {"error": "invalid_arguments", "tool": "get_medications_at_time",
             "messages": ["time_24h: Field required"]},
        ),
        ("get_emergency_contact", {}, {"error": "no_contacts"}),
        ("get_next_appointment", {}, {"error": "no_appointments"}),
        ("check_food_interaction", {}, {"error": "invalid_food", "food": ""}),
        ("get_medication_by_name", {}, {"error": "invalid_name", "name": ""}),
        # Unknown error key — must still produce a string, not crash.
        ("get_vitals", {}, {"error": "weird_unrecognized"}),
    ],
)
def test_format_response_handles_tool_error_dict(
    chat_mod: ModuleType, tool: str, args: dict[str, Any], result: dict[str, Any],
) -> None:
    """Any error-shaped tool result must produce a non-empty string answer
    rather than a TypeError / KeyError. Exact wording is allowed to drift,
    but the function MUST return without raising.
    """
    out = chat_mod.format_response("user question text", tool, args, result)
    assert isinstance(out, str)
    assert out  # non-empty


def test_meds_at_time_happy_path_unchanged(chat_mod: ModuleType) -> None:
    """The error-dict guard must not regress the success path."""
    result = [
        {"name": "Atorvastatin", "dose": "20 mg", "schedule": "08:00",
         "with_food": False, "purpose": "cholesterol"},
    ]
    out = chat_mod.format_response(
        "what at 8am", "get_medications_at_time", {"time_24h": "08:00"}, result,
    )
    assert "Atorvastatin" in out
    assert "20 mg" in out
    assert "08:00" in out


def test_chat_board_meds_at_time_error_handled(
    chat_board_mod: ModuleType,
) -> None:
    """The on-board copy of the formatter must apply the same guard.

    chat.py and chat_board.py duplicate the formatter logic (one reads YAML
    via Pydantic, the other uses pure stdlib JSON on the board). Keeping
    these two regression tests in lockstep prevents the bug from drifting
    into production via the on-board path.
    """
    err = {"error": "invalid_arguments", "tool": "get_medications_at_time",
           "messages": ["time_24h: Field required"]}
    out = chat_board_mod.format_response(
        "when do I take pills", "get_medications_at_time", {}, err,
    )
    assert isinstance(out, str)
    assert out
