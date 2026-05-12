"""Table-driven oracle for the wordform helpers (plan §5.2)."""

from __future__ import annotations

import pytest

from gemma_tools.dispenser_demo import wordform

# --------------------------------------------------------------------------
# age_to_words — plan §5.2 row "age".
# --------------------------------------------------------------------------


# | n  | expected      | desc                                |
@pytest.mark.parametrize(
    ("n", "expected", "desc"),
    [
        (0, "zero", "zero edge"),
        (1, "one", "single-digit"),
        (7, "seven", "single-digit"),
        (10, "ten", "teen lower bound"),
        (19, "nineteen", "teen upper bound"),
        (20, "twenty", "round tens"),
        (45, "forty five", "plan §5.2 worked example"),
        (99, "ninety nine", "two-digit upper"),
    ],
)
def test_age_to_words(n: int, expected: str, desc: str) -> None:
    assert wordform.age_to_words(n) == expected, desc


def test_age_to_words_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        wordform.age_to_words(-1)


# --------------------------------------------------------------------------
# date_to_words — plan §5.2 row "date".
# --------------------------------------------------------------------------


# | iso          | expected                                | desc                          |
@pytest.mark.parametrize(
    ("iso", "expected", "desc"),
    [
        ("2026-05-20", "May twentieth, twenty twenty six", "plan §5.2 worked example"),
        ("2026-01-01", "January first, twenty twenty six", "year-start, ordinal=1st"),
        ("2026-12-31", "December thirty first, twenty twenty six", "year-end, two-digit ordinal"),
        ("1999-09-09", "September ninth, nineteen ninety nine", "20th-century year"),
        ("2007-03-21", "March twenty first, twenty oh seven", "single-digit-tail year"),
    ],
)
def test_date_to_words(iso: str, expected: str, desc: str) -> None:
    assert wordform.date_to_words(iso) == expected, desc


# | iso          | match_re             | desc                          |
@pytest.mark.parametrize(
    ("iso", "match_re", "desc"),
    [
        ("2026/05/20", "not ISO", "rejects slash separator"),
        ("26-05-20", "not ISO", "rejects 2-digit year"),
        ("2026-13-01", "month out of range", "rejects month=13"),
    ],
)
def test_date_to_words_rejects_invalid(iso: str, match_re: str, desc: str) -> None:
    with pytest.raises(ValueError, match=match_re):
        wordform.date_to_words(iso)


# --------------------------------------------------------------------------
# time_to_words — plan §5.2 row "time".
# --------------------------------------------------------------------------


# | hhmm    | expected           | desc                                     |
@pytest.mark.parametrize(
    ("hhmm", "expected", "desc"),
    [
        ("10:30", "ten thirty", "plan §5.2 worked example"),
        ("10:00", "ten", "minute=00, drop the minute word"),
        ("10:05", "ten oh five", "minute 1..9 takes 'oh' filler"),
        ("00:00", "zero", "midnight as bare zero"),
        ("23:59", "twenty three fifty nine", "upper bound"),
        ("08:00", "eight", "morning hour, minute=00"),
    ],
)
def test_time_to_words(hhmm: str, expected: str, desc: str) -> None:
    assert wordform.time_to_words(hhmm) == expected, desc


# | hhmm    | match_re               | desc                              |
@pytest.mark.parametrize(
    ("hhmm", "match_re", "desc"),
    [
        ("10-30", "not HH:MM", "rejects dash separator"),
        ("24:00", "hour out of range", "rejects hour=24"),
        ("10:60", "minute out of range", "rejects minute=60"),
    ],
)
def test_time_to_words_rejects_invalid(hhmm: str, match_re: str, desc: str) -> None:
    with pytest.raises(ValueError, match=match_re):
        wordform.time_to_words(hhmm)


# --------------------------------------------------------------------------
# phone_to_words — plan §5.2 row "phone".
# --------------------------------------------------------------------------


# | phone           | expected                                              | desc                       |
@pytest.mark.parametrize(
    ("phone", "expected", "desc"),
    [
        (
            "+1-555-0142",
            "plus one five five five zero one four two",
            "plan §5.2 worked example",
        ),
        (
            "+1-555-9999",
            "plus one five five five nine nine nine nine",
            "trailing repeats",
        ),
        (
            "555-0142",
            "five five five zero one four two",
            "no country-code prefix",
        ),
    ],
)
def test_phone_to_words(phone: str, expected: str, desc: str) -> None:
    assert wordform.phone_to_words(phone) == expected, desc


def test_phone_to_words_rejects_letters() -> None:
    with pytest.raises(ValueError, match="unexpected char"):
        wordform.phone_to_words("+1-CALL-NOW")


# --------------------------------------------------------------------------
# freeform_to_words — plan §5.2 row "room" (free-form contract).
# --------------------------------------------------------------------------


# | s                              | expected                                          | desc                                   |
@pytest.mark.parametrize(
    ("s", "expected", "desc"),
    [
        (
            "Maple Clinic, Room 204",
            "Maple Clinic, Room two hundred four",
            "plan §5.2 worked example",
        ),
        ("Room 1", "Room one", "single digit"),
        ("Building 3, Floor 2", "Building three, Floor two", "multiple digit runs"),
        ("No digits here", "No digits here", "passthrough when no digits"),
    ],
)
def test_freeform_to_words(s: str, expected: str, desc: str) -> None:
    assert wordform.freeform_to_words(s) == expected, desc


# --------------------------------------------------------------------------
# diagnosis_to_words / diagnoses_to_words — plan §5.2 row "diagnosis".
# --------------------------------------------------------------------------


# | name              | expected             | desc                                  |
@pytest.mark.parametrize(
    ("name", "expected", "desc"),
    [
        ("Type 2 Diabetes", "Type Two Diabetes", "plan §5.2 worked example"),
        ("Hypertension", "Hypertension", "passthrough when no digits"),
        ("COVID 19", "COVID Nineteen", "title-cased digit substitution"),
    ],
)
def test_diagnosis_to_words(name: str, expected: str, desc: str) -> None:
    assert wordform.diagnosis_to_words(name) == expected, desc


# | names                                       | expected                                  | desc                          |
@pytest.mark.parametrize(
    ("names", "expected", "desc"),
    [
        (
            ["Type 2 Diabetes", "Hypertension"],
            "Type Two Diabetes and Hypertension",
            "plan §5.1 worked example (2 items)",
        ),
        (["Hypertension"], "Hypertension", "single item"),
        ([], "", "empty list returns empty string"),
        (
            ["A", "B", "C"],
            "A, B, and C",
            "Oxford comma for 3+ items",
        ),
    ],
)
def test_diagnoses_to_words(names: list[str], expected: str, desc: str) -> None:
    assert wordform.diagnoses_to_words(names) == expected, desc
