"""Tests for gemma_tools.bench_eval.

Synthetic JSONL fixtures drive the scorer independently of the board —
the happy-path round-trip (bench-prompt stub → bench-eval) is exercised
via a one-off integration test at the bottom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_tools._legacy.bench_eval import (
    ScoredRow,
    load_jsonl,
    main,
    render_markdown_summary,
    score_response,
    score_sweep,
)
from gemma_tools._legacy.bench_prompt import compile_pattern_flags

_REPO = Path(__file__).resolve().parents[2]
CANONICAL_PROMPTS = _REPO / "data" / "_legacy" / "prompts.yaml"


def _make_jsonl_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "prompt_id": "P1",
        "prompt_class": "fact_lookup",
        "prompt_text": "what is my current heart rate?",
        "response_text": "Your heart rate is 72 bpm.",
        "run_started_iso": "2026-04-24T12:00:00",
        "timing": {
            "wall_ms_load": 4260.0,
            "wall_ms_ttft_vendor": 1500.0,
            "wall_ms_ttft_external": 1520.0,
            "wall_ms_total": 7500.0,
            "tokens_generated": 50,
            "tokens_per_sec": 8.361,
        },
        "peak_rss_mb": 1134.5,
        "cma_free_kb_before": 425_000,
        "cma_free_kb_during": 4,
        "cma_free_kb_after": 328_000,
        "error": None,
    }
    base.update(overrides)
    return base


def _write_sweep(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "sweep.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")
    return path


# | flags_str | expected_bits_superset                            | desc                        |
@pytest.mark.parametrize(
    ("flags_str", "expects_icase", "expects_dotall", "desc"),
    [
        ("",    False, False, "empty flag string → no bits"),
        ("i",   True,  False, "i → IGNORECASE"),
        ("is",  True,  True,  "is → IGNORECASE + DOTALL"),
        ("I",   True,  False, "uppercase I accepted (lowercased internally)"),
        ("xi",  True,  False, "VERBOSE + IGNORECASE"),
    ],
)
def test_compile_pattern_flags_maps_known_chars(
    flags_str: str, expects_icase: bool, expects_dotall: bool, desc: str
) -> None:
    import re as _re
    bits = compile_pattern_flags(flags_str)
    assert bool(bits & _re.IGNORECASE) is expects_icase, f"{desc} — IGNORECASE"
    assert bool(bits & _re.DOTALL) is expects_dotall, f"{desc} — DOTALL"


def test_compile_pattern_flags_rejects_unknown_char() -> None:
    with pytest.raises(ValueError, match=r"unknown regex flag 'q'"):
        compile_pattern_flags("q")


# | pattern      | flags | response                  | expected | desc                          |
@pytest.mark.parametrize(
    ("pattern", "flags", "response", "expected", "desc"),
    [
        ("72",                   "",  "Your heart rate is 72 bpm.", True,  "integer literal matches"),
        ("lisinopril",           "i", "You take Lisinopril 10 mg.", True,  "case-insensitive word matches"),
        ("lisinopril",           "",  "You take Lisinopril 10 mg.", False, "case-sensitive word rejects"),
        (r"118.+76|118/76",      "",  "BP 118/76 mmHg",             True,  "alternation matches 118/76"),
        (r"not in record",       "i", "Not in record.",             True,  "case-insensitive refusal matches"),
        (r"health record",       "i", "I cannot help with jokes.",  False, "non-matching → False"),
    ],
)
def test_score_response_regex_contract(
    pattern: str, flags: str, response: str, expected: bool, desc: str
) -> None:
    assert score_response(pattern, flags, response) is expected, desc


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "sweep.jsonl"
    path.write_text(
        '{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n', encoding="utf-8",
    )
    assert [r["a"] for r in load_jsonl(path)] == [1, 2, 3], (
        "blank / whitespace-only lines are skipped"
    )


def test_load_jsonl_raises_on_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / "sweep.jsonl"
    path.write_text('{"a": 1}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"sweep.jsonl:2"):
        load_jsonl(path)


def test_score_sweep_joins_jsonl_with_prompt_suite(tmp_path: Path) -> None:
    """A canonical-fixture round-trip: single JSONL row for P1 → one
    ScoredRow with pass_pattern pulled from prompts.yaml."""
    jsonl_path = _write_sweep(
        tmp_path, [_make_jsonl_row(prompt_id="P1")]
    )
    scored = score_sweep(jsonl_path, CANONICAL_PROMPTS)
    assert len(scored) == 1, "one row in → one scored row out"
    assert scored[0].prompt_id == "P1", "id preserved"
    assert scored[0].passed_regex is True, (
        'response "Your heart rate is 72 bpm." matches P1 pattern "72"'
    )
    assert scored[0].tokens_generated == 50, "timing flattened from nested dict"


def test_score_sweep_fail_case_records_false(tmp_path: Path) -> None:
    jsonl_path = _write_sweep(
        tmp_path,
        [_make_jsonl_row(prompt_id="P1", response_text="I don't know.")],
    )
    scored = score_sweep(jsonl_path, CANONICAL_PROMPTS)
    assert scored[0].passed_regex is False, "response lacks '72' → regex FAIL"


def test_score_sweep_raises_on_unknown_id(tmp_path: Path) -> None:
    jsonl_path = _write_sweep(
        tmp_path, [_make_jsonl_row(prompt_id="NONEXISTENT")]
    )
    with pytest.raises(ValueError, match=r"NONEXISTENT.*not in"):
        score_sweep(jsonl_path, CANONICAL_PROMPTS)


def test_render_markdown_summary_empty_rows_safe() -> None:
    assert "_no rows scored" in render_markdown_summary([]), (
        "empty sweep produces a visible sentinel, not a broken table"
    )


def test_render_markdown_summary_header_lines_pass_fail_err_counts() -> None:
    rows = [
        ScoredRow(
            prompt_id="P1", prompt_class="fact_lookup",
            prompt_text="x", pass_pattern="y", pattern_flags="",
            response_text="y", passed_regex=True,
            tokens_generated=5, wall_ms_total=1000.0,
            wall_ms_ttft_external=500.0, tokens_per_sec=10.0,
            peak_rss_mb=100.0, cma_free_kb_during=1, error=None,
        ),
        ScoredRow(
            prompt_id="P2", prompt_class="fact_lookup",
            prompt_text="x", pass_pattern="y", pattern_flags="",
            response_text="no", passed_regex=False,
            tokens_generated=3, wall_ms_total=800.0,
            wall_ms_ttft_external=400.0, tokens_per_sec=10.0,
            peak_rss_mb=100.0, cma_free_kb_during=1, error=None,
        ),
        ScoredRow(
            prompt_id="P3", prompt_class="fact_lookup",
            prompt_text="x", pass_pattern="y", pattern_flags="",
            response_text="", passed_regex=False,
            tokens_generated=0, wall_ms_total=100.0,
            wall_ms_ttft_external=0.0, tokens_per_sec=0.0,
            peak_rss_mb=100.0, cma_free_kb_during=1, error="RuntimeError: boom",
        ),
    ]
    md = render_markdown_summary(rows)
    assert "**Regex pass rate**: 1/3 (33%)" in md, "header reports 1 PASS / 3 total"
    assert "**Errors**: 1" in md, "header reports 1 error row"
    assert "| P1 | fact_lookup | PASS |" in md, "P1 shown as PASS"
    assert "| P2 | fact_lookup | FAIL |" in md, "P2 shown as FAIL"
    assert "| P3 | fact_lookup | ERR |" in md, "P3 shown as ERR (error takes precedence)"
    assert "RuntimeError: boom" in md, "error message surfaced in note column"


def test_main_writes_markdown_file(tmp_path: Path) -> None:
    jsonl_path = _write_sweep(
        tmp_path, [_make_jsonl_row(prompt_id="P1")]
    )
    out_md = tmp_path / "summary.md"
    rc = main(
        ["--jsonl", str(jsonl_path),
         "--prompts", str(CANONICAL_PROMPTS),
         "--output", str(out_md)]
    )
    assert rc == 0, "clean exit"
    content = out_md.read_text()
    assert "| P1 |" in content, "summary file contains the P1 row"
    assert "PASS" in content, "P1 passed"


def test_main_stdout_when_no_output_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl_path = _write_sweep(tmp_path, [_make_jsonl_row(prompt_id="P1")])
    rc = main(["--jsonl", str(jsonl_path), "--prompts", str(CANONICAL_PROMPTS)])
    assert rc == 0, "clean exit"
    captured = capsys.readouterr()
    assert "| P1 |" in captured.out, "stdout got the markdown when no --output"
