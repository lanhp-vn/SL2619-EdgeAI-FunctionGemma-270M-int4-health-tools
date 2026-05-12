"""Pure word-form helpers for the dispenser-demo tool responses.

Source-of-truth contract: `docs/plans/dispenser-demo/plan.md` §5.2 table.

Every digit-bearing field in a tool response carries a digit-free `*_words`
companion derived here. The wordform module is the SINGLE site that converts
digits to words — `tools.py` only composes these helpers. Keeping derivation
centralized means a wordform tweak is one diff, and `test_wordform.py` is the
single oracle for the §5.2 contract.

Conventions:

- Cardinal numbers are space-separated, no hyphens: `45` → `"forty five"`,
  `204` → `"two hundred four"`. Plan §5.2 row "age" is explicit on this.
- Dates render as `"Month DayOrdinal, YearPair"` with the Oxford-style comma
  before the year: `"2026-05-20"` → `"May twentieth, twenty twenty six"`.
- Times render as `"hour [minute]"` with `"oh"` for minutes 1..9: `"10:30"`
  → `"ten thirty"`, `"10:00"` → `"ten"`, `"10:05"` → `"ten oh five"`. No
  AM/PM (plan §5.2 row "time").
- Phones are digit-by-digit with `"plus"` for `"+"`: `"+1-555-0142"` →
  `"plus one five five five zero one four two"`. Separators are stripped.
- Free-form strings (e.g. `location` = `"Maple Clinic, Room 204"`) get
  every contiguous digit run replaced with its cardinal: the location
  becomes `"Maple Clinic, Room two hundred four"`. Plan §5.2 row "room"
  documents the contract.
- Diagnoses are title-cased within the substituted digit run — `"Type 2
  Diabetes"` → `"Type Two Diabetes"` (the rest of the string is left
  verbatim). `diagnoses_to_words([...])` joins with `" and "` for 2 items
  and Oxford comma for 3+.

All helpers raise `ValueError` on malformed input. They never silently
return `""` or pass-through the raw digits — that would defeat the
`test_tools_word_only.py` invariant.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# --------------------------------------------------------------------------
# Cardinal / ordinal number tables. Range covered: 0..9999 (room numbers,
# ages, year pairs). Larger numbers raise; we don't anticipate them.
# --------------------------------------------------------------------------

_ONES: tuple[str, ...] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS: tuple[str, ...] = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
    "ninety",
)
_ORDINALS_ONES: tuple[str, ...] = (
    "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
)
_ORDINALS_TENS: tuple[str, ...] = (
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
    "seventieth", "eightieth", "ninetieth",
)
_MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


def _cardinal(n: int) -> str:
    """Cardinal number in words. Range: 0..9999."""
    if n < 0 or n > 9999:
        raise ValueError(f"cardinal out of range [0, 9999]: {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, units = divmod(n, 10)
        return _TENS[tens] if units == 0 else f"{_TENS[tens]} {_ONES[units]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return f"{_ONES[hundreds]} hundred"
        return f"{_ONES[hundreds]} hundred {_cardinal(rest)}"
    thousands, rest = divmod(n, 1000)
    if rest == 0:
        return f"{_ONES[thousands]} thousand"
    return f"{_ONES[thousands]} thousand {_cardinal(rest)}"


def _ordinal(n: int) -> str:
    """Day-of-month ordinal. Range: 1..31."""
    if not (1 <= n <= 31):
        raise ValueError(f"day ordinal out of range [1, 31]: {n}")
    if n < 20:
        return _ORDINALS_ONES[n]
    tens, units = divmod(n, 10)
    if units == 0:
        return _ORDINALS_TENS[tens]
    return f"{_TENS[tens]} {_ORDINALS_ONES[units]}"


def _year_pair(n: int) -> str:
    """Year-in-words using the two-pair convention.

    2026 → "twenty twenty six"; 1999 → "nineteen ninety nine".
    Years ending in 00 use "<high> thousand" if `high` is a multiple of 10,
    else "<high> hundred". Years ending in 01..09 use "<high> oh <low>".
    """
    if not (1000 <= n <= 9999):
        raise ValueError(f"year out of range [1000, 9999]: {n}")
    high, low = divmod(n, 100)
    if low == 0:
        # 2000 → "two thousand"; 1900 → "nineteen hundred".
        return f"{_cardinal(high)} thousand" if high % 10 == 0 else f"{_cardinal(high)} hundred"
    if low < 10:
        # 2007 → "twenty oh seven".
        return f"{_cardinal(high)} oh {_cardinal(low)}"
    return f"{_cardinal(high)} {_cardinal(low)}"


# --------------------------------------------------------------------------
# Public helpers. Each maps one §5.2 row.
# --------------------------------------------------------------------------


def age_to_words(age: int) -> str:
    """`age=45` → `"forty five"`. No hyphen (plan §5.2)."""
    if age < 0:
        raise ValueError(f"age cannot be negative: {age}")
    return _cardinal(age)


def date_to_words(iso_date: str) -> str:
    """`"2026-05-20"` → `"May twentieth, twenty twenty six"`.

    Strict ISO `YYYY-MM-DD` regex; malformed input raises `ValueError`.
    """
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso_date)
    if m is None:
        raise ValueError(f"not ISO YYYY-MM-DD: {iso_date!r}")
    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mm <= 12):
        raise ValueError(f"month out of range: {iso_date!r}")
    return f"{_MONTHS[mm - 1]} {_ordinal(dd)}, {_year_pair(yyyy)}"


def time_to_words(hhmm: str) -> str:
    """`"10:30"` → `"ten thirty"`.

    24-hour HH:MM; the convention drops the literal hour for minutes 00 and
    inserts `"oh"` for minutes 01..09 (plan §5.2 row "time" — no AM/PM).
    """
    m = re.fullmatch(r"(\d{2}):(\d{2})", hhmm)
    if m is None:
        raise ValueError(f"not HH:MM: {hhmm!r}")
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23):
        raise ValueError(f"hour out of range: {hhmm!r}")
    if not (0 <= mi <= 59):
        raise ValueError(f"minute out of range: {hhmm!r}")
    if mi == 0:
        return _cardinal(h)
    if mi < 10:
        return f"{_cardinal(h)} oh {_cardinal(mi)}"
    return f"{_cardinal(h)} {_cardinal(mi)}"


def phone_to_words(phone: str) -> str:
    """`"+1-555-0142"` → `"plus one five five five zero one four two"`.

    Digit-by-digit. `+` becomes `"plus"`; `-` and ASCII space are stripped.
    Any other character is a hard error so a typo in the YAML surfaces here
    rather than as a silently-mangled wordform.
    """
    out: list[str] = []
    for ch in phone:
        if ch == "+":
            out.append("plus")
        elif ch.isdigit():
            out.append(_cardinal(int(ch)))
        elif ch in {"-", " "}:
            continue
        else:
            raise ValueError(f"unexpected char {ch!r} in phone {phone!r}")
    return " ".join(out)


def freeform_to_words(s: str) -> str:
    """Convert digits in-place inside a free-form string.

    Each maximal digit run is replaced with its cardinal-in-words form.
    Non-digit characters are left verbatim.

    Examples:
        `"Maple Clinic, Room 204"` → `"Maple Clinic, Room two hundred four"`.
        `"Building 3, Floor 2"` → `"Building three, Floor two"`.

    Used for fields like `location` where the digit is embedded in prose.
    Diagnoses go through `diagnosis_to_words` instead (title-cased digit
    substitution) to match the §5.2 row.
    """
    return re.sub(r"\d+", lambda m: _cardinal(int(m.group(0))), s)


def diagnosis_to_words(name: str) -> str:
    """Title-case the digit substitution inside a diagnosis name.

    `"Type 2 Diabetes"` → `"Type Two Diabetes"` (plan §5.2 row "diagnosis").
    `"Hypertension"` (no digits) → `"Hypertension"` (verbatim).

    Differs from `freeform_to_words` only in the cap of the substituted
    word. Diagnoses are proper-noun-ish names where the digit-form would
    have been title-cased; the wordform should mirror that.
    """
    return re.sub(r"\d+", lambda m: _cardinal(int(m.group(0))).title(), name)


def diagnoses_to_words(names: Iterable[str]) -> str:
    """Join a list of diagnosis names with `" and "` (Oxford comma for 3+).

    `["Type 2 Diabetes", "Hypertension"]` → `"Type Two Diabetes and Hypertension"`.
    `["A", "B", "C"]` → `"A, B, and C"` (Oxford comma).
    `[]` → `""`. `["A"]` → `"A"`.
    """
    cleaned = [diagnosis_to_words(n) for n in names]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"
