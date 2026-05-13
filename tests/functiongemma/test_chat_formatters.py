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
    # The host formatter keeps the raw HH:MM in parens alongside the
    # humanized phrase (see chat.py:_format_meds_at_time). The board
    # formatter drops the raw form (TTS-only path).
    assert "08:00" in out or "8 in the morning" in out


# ----------------------------------------------------------------------------
# Humanizer regression coverage — chat.py and chat_board.py duplicate the
# helpers, so test both copies with the same inputs to guarantee parity.
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["chat", "chat_board"])
def fmt_mod(request: pytest.FixtureRequest) -> ModuleType:
    if request.param == "chat":
        return _load(_CHAT, "fg_chat")
    return _load(_CHAT_BOARD, "fg_chat_board")


@pytest.mark.parametrize(
    ("date_in", "want_substring"),
    [
        ("2026-05-20", "May 20"),
        ("2026-01-01", "January 1"),
        ("2026-12-31", "December 31"),
        # Malformed → return raw (don't crash).
        ("not-a-date", "not-a-date"),
    ],
)
def test_humanize_date(
    fmt_mod: ModuleType, date_in: str, want_substring: str,
) -> None:
    assert want_substring in fmt_mod._humanize_date(date_in)


@pytest.mark.parametrize(
    ("time_in", "want_substring"),
    [
        ("07:30", "7:30 in the morning"),
        ("08:00", "8 in the morning"),
        ("14:00", "2 in the afternoon"),
        ("20:00", "8 in the evening"),
        ("23:30", "11:30 at night"),
        ("00:00", "12 at night"),
        ("12:00", "12 in the afternoon"),
        # Malformed → return raw.
        ("not-a-time", "not-a-time"),
    ],
)
def test_humanize_time(
    fmt_mod: ModuleType, time_in: str, want_substring: str,
) -> None:
    assert want_substring in fmt_mod._humanize_time(time_in)


def test_humanize_schedule_conjunction(fmt_mod: ModuleType) -> None:
    out = fmt_mod._humanize_schedule("08:00, 20:00")
    assert "8 in the morning" in out
    assert "8 in the evening" in out
    assert " and " in out


def test_humanize_schedule_three_times(fmt_mod: ModuleType) -> None:
    out = fmt_mod._humanize_schedule("08:00, 14:00, 20:00")
    # Oxford-comma form: "X, Y, and Z"
    assert out.count(",") >= 2
    assert "and " in out


def test_humanize_measured_suffix_pretty(fmt_mod: ModuleType) -> None:
    out = fmt_mod._humanize_measured_suffix("2026-05-11 07:30")
    assert "May 11" in out
    assert "7:30 in the morning" in out
    # No raw "2026-05-11" or "07:30" leaks through.
    assert "2026" not in out
    assert "07:30" not in out


def test_humanize_measured_suffix_empty(fmt_mod: ModuleType) -> None:
    assert fmt_mod._humanize_measured_suffix("") == ""


def test_vitals_no_iso_leakage(fmt_mod: ModuleType) -> None:
    """The vitals sentence going to TTS must not contain ISO date/time
    fragments; Piper renders those digit-by-digit and breaks the cadence.
    """
    result = {
        "heart_rate_bpm": 82,
        "blood_pressure_systolic": 142,
        "blood_pressure_diastolic": 88,
        "spo2_percent": 97,
        "body_temperature_c": 36.6,
        "respiratory_rate": 16,
        "last_measured": "2026-05-11 07:30",
    }
    out = fmt_mod.format_response(
        "what is my heart rate?", "get_vitals", {}, result,
    )
    assert "82" in out
    assert "2026-05-11" not in out
    assert "07:30" not in out
    assert "May 11" in out
    assert "morning" in out


def test_appointment_no_iso_leakage(fmt_mod: ModuleType) -> None:
    result = {
        "date": "2026-05-20",
        "time": "14:30",
        "provider": "Dr. Sarah Kim",
        "purpose": "blood pressure follow-up",
        "location": "clinic room 3",
    }
    out = fmt_mod.format_response(
        "when is my next appointment?", "get_next_appointment", {}, result,
    )
    assert "2026-05-20" not in out
    assert "14:30" not in out
    assert "May 20" in out
    assert "2:30 in the afternoon" in out


def test_out_of_scope_refusal_via_format_response(
    fmt_mod: ModuleType,
) -> None:
    """The OUT_OF_SCOPE_TOOL sentinel must produce a refusal sentence
    (not the fallthrough `[no formatter for tool: ...]`).
    """
    out = fmt_mod.format_response(
        "tell me a joke", fmt_mod.OUT_OF_SCOPE_TOOL, {}, None,
    )
    assert isinstance(out, str)
    assert out
    assert "no formatter" not in out
    # The refusal mentions the allowed scope so the user knows what to
    # rephrase to — don't hard-code the exact wording but check for a
    # signature substring.
    assert "medication" in out.lower() or "help with" in out.lower()


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
