"""Tests for gemma_tools.chat_probe.

Pure-helper tests run in isolation (no torq.runtime needed). The main()
smoke injects a stub `runner_factory` so we exercise the full argparse
+ streaming + metrics path without loading a VMFB.
"""

from __future__ import annotations

import io
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

from gemma_tools.chat_probe import (
    compose_probe_prompt,
    main,
    patched_sys_prompt,
    slice_health_yaml,
)
from gemma_tools.health_table import (
    HealthTable,
    Patient,
    Vitals,
    load_health_table,
)

_REPO = Path(__file__).resolve().parents[1]
CANONICAL_HEALTH = _REPO / "data" / "health_table_v1.yaml"

_MINIMAL = HealthTable(
    patient=Patient(name="Test Patient", age=45),
    vitals=Vitals(
        heart_rate_bpm=72,
        blood_pressure_systolic=118,
        blood_pressure_diastolic=76,
        spo2_percent=98,
        body_temperature_c=36.7,
        respiratory_rate=16,
    ),
    notes=("seed",),
)


# | field_path                      | required_substring            | forbidden_substring         | desc                                                  |
@pytest.mark.parametrize(
    ("field_path", "required_substring", "forbidden_substring", "desc"),
    [
        ("vitals.heart_rate_bpm",        "heart_rate_bpm: 72",             "name: Test Patient",    "deepest single-field leaf nests under its key"),
        ("vitals.blood_pressure_systolic", "blood_pressure_systolic: 118", "heart_rate_bpm",        "other vitals siblings are excluded"),
        ("patient.name",                  "name: Test Patient",             "age: 45",               "sibling dict fields are excluded"),
        ("vitals",                        "heart_rate_bpm: 72",             "patient:",              "whole block returns all its fields, siblings excluded"),
    ],
)
def test_slice_health_yaml_navigates_and_wraps(
    field_path: str, required_substring: str, forbidden_substring: str, desc: str,
) -> None:
    out = slice_health_yaml(_MINIMAL, field_path)
    assert required_substring in out, f"{desc}: '{required_substring}' missing from\n{out}"
    assert forbidden_substring not in out, f"{desc}: '{forbidden_substring}' leaked into\n{out}"


def test_slice_health_yaml_empty_path_returns_empty_string() -> None:
    assert slice_health_yaml(_MINIMAL, "") == "", (
        "empty path disables YAML injection (no-slice mode)"
    )


def test_slice_health_yaml_unknown_path_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="no_such_field"):
        slice_health_yaml(_MINIMAL, "vitals.no_such_field")


def test_slice_health_yaml_nested_miss_reports_offending_part() -> None:
    with pytest.raises(KeyError, match="nope"):
        slice_health_yaml(_MINIMAL, "nope.whatever")


# | preface  | field_path                | question                  | required_substrings                                                        | desc                                         |
@pytest.mark.parametrize(
    ("preface", "field_path", "question", "required_substrings", "desc"),
    [
        ("",               "vitals.heart_rate_bpm",  "what is my heart rate?", ("YAML:\n", "heart_rate_bpm: 72", "what is my heart rate?"),     "YAML + question, no preface"),
        ("",               "",                        "say hi",                  ("say hi",),                                                    "no preface, no YAML → bare question"),
        ("Answer briefly.", "vitals.heart_rate_bpm", "hr?",                     ("Answer briefly.", "YAML:\n", "heart_rate_bpm: 72", "hr?"), "preface leads, then YAML, then question"),
    ],
)
def test_compose_probe_prompt_builds_expected_blocks(
    preface: str,
    field_path: str,
    question: str,
    required_substrings: tuple[str, ...],
    desc: str,
) -> None:
    out = compose_probe_prompt(_MINIMAL, field_path, question, preface)
    for needle in required_substrings:
        assert needle in out, f"{desc}: '{needle}' missing from\n{out}"


def test_compose_probe_prompt_preface_and_yaml_are_separated_by_blank_line() -> None:
    """Blank-line separation lets the model see discrete blocks rather than
    one run-on paragraph — matters for attention on small-vocab tokens."""
    out = compose_probe_prompt(
        _MINIMAL, "vitals.heart_rate_bpm", "what is my heart rate?",
        preface="Answer briefly.",
    )
    # "Answer briefly." followed by "\n\nYAML:" — blank line between.
    assert "Answer briefly.\n\nYAML:" in out, (
        "preface and YAML separated by exactly one blank line"
    )
    # "heart_rate_bpm: 72" followed by "\n\nwhat is my heart rate?"
    assert "heart_rate_bpm: 72\n\nwhat is my heart rate?" in out, (
        "YAML block and question separated by exactly one blank line"
    )


def test_patched_sys_prompt_restores_on_exit() -> None:
    fake = types.ModuleType("fake_runner")
    setattr(fake, "DEFAULT_SYS_PROMPT", "ORIGINAL")  # noqa: B010
    with patched_sys_prompt(fake, "OVERRIDE"):
        assert getattr(fake, "DEFAULT_SYS_PROMPT") == "OVERRIDE", "override active"  # noqa: B009
    assert getattr(fake, "DEFAULT_SYS_PROMPT") == "ORIGINAL", "original restored"  # noqa: B009


def test_patched_sys_prompt_restores_even_on_exception() -> None:
    fake = types.ModuleType("fake_runner")
    setattr(fake, "DEFAULT_SYS_PROMPT", "ORIGINAL")  # noqa: B010
    with pytest.raises(RuntimeError, match="boom"), patched_sys_prompt(fake, "X"):
        raise RuntimeError("boom")
    assert getattr(fake, "DEFAULT_SYS_PROMPT") == "ORIGINAL", "original restored on raise"  # noqa: B009


# --------------------------------------------------------------------------
# main() smoke — stub `runner_factory` so we never hit torq.runtime on host
# but exercise every other path: argparse, streaming, metrics printing.
# --------------------------------------------------------------------------


class _StubGemma3:
    """Fake _Gemma3Like that yields scripted chunks."""

    def __init__(self, model_path: str, instruct_model: bool = False) -> None:
        self.model_path = model_path
        self.instruct_model = instruct_model
        self.last_input: str | None = None
        self.last_max_tokens: int | None = None

    def run_stream(
        self, user_input: str, max_tokens: int | None = None
    ) -> Iterator[str]:
        self.last_input = user_input
        self.last_max_tokens = max_tokens
        yield "72"
        yield " bpm."

    @property
    def time_to_first_token(self) -> float:
        return 1234.5

    @property
    def generated_tokens(self) -> int:
        return 2


def _stub_runner_factory() -> tuple[types.ModuleType, type]:
    """Returns a fake `runner` module + our stub class. The module-like
    object has `DEFAULT_SYS_PROMPT` so the monkey-patch path is exercisable."""
    mod = types.ModuleType("runner")
    setattr(mod, "DEFAULT_SYS_PROMPT", "VENDOR_DEFAULT_SENTINEL")  # noqa: B010
    setattr(mod, "Gemma3Static", _StubGemma3)  # noqa: B010
    return mod, _StubGemma3


def test_main_smoke_streams_chunks_and_reports_metrics(tmp_path: Path) -> None:
    # Use the canonical fixture so the health_table load path is real.
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    ticks = iter([0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000])

    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--health-table", str(CANONICAL_HEALTH),
            "--question", "what is my heart rate?",
            "--yaml-field", "vitals.heart_rate_bpm",
            "--max-gen-tokens", "64",
        ],
        runner_factory=_stub_runner_factory,
        out=out_buf,
        err=err_buf,
        clock_ns=lambda: next(ticks),
    )
    assert rc == 0, "clean exit code"
    stdout = out_buf.getvalue()
    stderr = err_buf.getvalue()
    assert ">>> 72 bpm." in stdout, f"stdout missing streamed chunks:\n{stdout}"
    assert "[prompt chars]" in stderr, "prompt-size diagnostic logged to stderr"
    assert "[load ms     ]" in stderr, "load-time diagnostic logged to stderr"
    assert "chunks=2" in stderr, "chunk count reported"
    assert "vendor_tokens=2" in stderr, "vendor token count reported"


def test_main_smoke_honors_patched_sys_prompt(tmp_path: Path) -> None:
    """--patched-sys-prompt must be applied DURING the Gemma3Static
    constructor call and restored afterwards."""
    observed: list[str] = []

    class _Capturing(_StubGemma3):
        def __init__(self, model_path: str, instruct_model: bool = False) -> None:
            super().__init__(model_path, instruct_model)
            # Peek at the module the factory returned.
            observed.append(getattr(_mod, "DEFAULT_SYS_PROMPT"))  # noqa: B009

    _mod = types.ModuleType("runner")
    setattr(_mod, "DEFAULT_SYS_PROMPT", "VENDOR_DEFAULT_SENTINEL")  # noqa: B010
    setattr(_mod, "Gemma3Static", _Capturing)  # noqa: B010

    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--health-table", str(CANONICAL_HEALTH),
            "--question", "hi",
            "--patched-sys-prompt", "",
        ],
        runner_factory=lambda: (_mod, _Capturing),
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert rc == 0, "clean exit"
    assert observed == [""], "empty-string override active during __init__"
    assert getattr(_mod, "DEFAULT_SYS_PROMPT") == "VENDOR_DEFAULT_SENTINEL", (  # noqa: B009
        "vendor default restored after main returns"
    )


def test_main_smoke_without_patched_sys_prompt_leaves_vendor_default(
    tmp_path: Path,
) -> None:
    """When --patched-sys-prompt is NOT supplied, no monkey-patch happens."""
    observed: list[str] = []

    class _Capturing(_StubGemma3):
        def __init__(self, model_path: str, instruct_model: bool = False) -> None:
            super().__init__(model_path, instruct_model)
            observed.append(getattr(_mod, "DEFAULT_SYS_PROMPT"))  # noqa: B009

    _mod = types.ModuleType("runner")
    setattr(_mod, "DEFAULT_SYS_PROMPT", "VENDOR_DEFAULT_SENTINEL")  # noqa: B010
    setattr(_mod, "Gemma3Static", _Capturing)  # noqa: B010

    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--health-table", str(CANONICAL_HEALTH),
            "--question", "hi",
        ],
        runner_factory=lambda: (_mod, _Capturing),
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert rc == 0, "clean exit"
    assert observed == ["VENDOR_DEFAULT_SENTINEL"], (
        "vendor default untouched at Gemma3Static.__init__ time"
    )


def test_main_smoke_with_canonical_health_fixture_composes_full_prompt(
    tmp_path: Path,
) -> None:
    """Integration: real health fixture + real slicer + stub runner → check
    the prompt the stub received contains exactly the shape we want."""
    stub_holder: dict[str, _StubGemma3] = {}

    class _Capturing(_StubGemma3):
        def __init__(self, model_path: str, instruct_model: bool = False) -> None:
            super().__init__(model_path, instruct_model)
            stub_holder["impl"] = self

    _mod = types.ModuleType("runner")
    setattr(_mod, "DEFAULT_SYS_PROMPT", "X")  # noqa: B010
    setattr(_mod, "Gemma3Static", _Capturing)  # noqa: B010

    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--health-table", str(CANONICAL_HEALTH),
            "--question", "what is my heart rate?",
            "--yaml-field", "vitals.heart_rate_bpm",
        ],
        runner_factory=lambda: (_mod, _Capturing),
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert rc == 0
    prompt = stub_holder["impl"].last_input
    assert prompt is not None
    assert "heart_rate_bpm: 72" in prompt, "canonical fixture has HR 72"
    assert "what is my heart rate?" in prompt, "question terminates prompt"
    # No chat-template markers — vendor runner adds those.
    assert "<start_of_turn>" not in prompt, "chat markers must not leak"


def test_main_smoke_real_canonical_fixture_loads() -> None:
    """Canary: the canonical health_table_v1.yaml used on-board is loadable
    here. If this breaks, the fixture drifted."""
    ht = load_health_table(CANONICAL_HEALTH)
    assert ht.vitals.heart_rate_bpm == 72, "expected HR=72 per canonical fixture"
