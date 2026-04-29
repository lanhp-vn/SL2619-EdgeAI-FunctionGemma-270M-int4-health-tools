"""Tests for gemma_tools.bench_prompt.

Board-side bits (`torq.runtime`, `Gemma3Static`) are NOT imported here —
everything tested is host-pure logic behind the import guard in the module.
"""

from __future__ import annotations

import json
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gemma_tools.bench_prompt import (
    BenchAdapter,
    BenchRow,
    Gemma3BenchAdapter,
    LlamaCompletionBenchAdapter,
    LlamaCompletionError,
    MemorySampler,
    PromptSpec,
    Stopwatch,
    TimingRecord,
    VendorImportError,
    _MainArgs,
    _patched_default_sys_prompt,
    _safe_read_cma_free_kb,
    create_gemma3_bench_adapter,
    create_llama_completion_bench_adapter,
    default_adapter_factory,
    load_prompt_suite,
    main,
    parse_completion_response,
    parse_llama_perf,
    read_cma_free_kb,
    read_rss_mb,
    wrap_gemma3_chat_template,
    write_row,
    write_rows,
)
from gemma_tools.prompt_composer import compose_prompt

_REPO = Path(__file__).resolve().parents[1]
CANONICAL_PROMPTS = _REPO / "data" / "prompts.yaml"
CANONICAL_HEALTH = _REPO / "data" / "health_table_v1.yaml"

# Shared fixture values — a minimal well-formed timing+row pair used across tests.
_FROZEN_TIMING = TimingRecord(
    wall_ms_load=4260.0,
    wall_ms_ttft_vendor=1500.0,
    wall_ms_ttft_external=1520.0,
    wall_ms_total=7500.0,
    tokens_generated=50,
)


def _make_row(**overrides: object) -> BenchRow:
    base: dict[str, object] = {
        "prompt_id": "P1",
        "prompt_class": "fact_lookup",
        "prompt_text": "what is my current heart rate?",
        "response_text": "Your heart rate is 72 bpm.",
        "run_started_iso": "2026-04-24T12:00:00",
        "timing": _FROZEN_TIMING,
        "peak_rss_mb": 1134.5,
        "cma_free_kb_before": 425_000,
        "cma_free_kb_during": 4,
        "cma_free_kb_after": 328_000,
        "error": None,
    }
    base.update(overrides)
    return BenchRow(**base)  # type: ignore[arg-type]


# | clock_sequence_ns       | expected_elapsed_ms | desc                                          |
@pytest.mark.parametrize(
    ("clock_sequence_ns", "expected_elapsed_ms", "desc"),
    [
        ((0, 1_000_000),             1.0,       "1 ms elapsed"),
        ((0, 0),                     0.0,       "zero-duration window"),
        ((100, 1_000_000_100),       1_000.0,   "1 s elapsed with non-zero start"),
        ((42, 42 + 250_000_000),     250.0,     "250 ms elapsed across large offset"),
    ],
)
def test_stopwatch_elapsed_ms_matches_injected_clock(
    clock_sequence_ns: tuple[int, int], expected_elapsed_ms: float, desc: str
) -> None:
    calls = iter(clock_sequence_ns)
    with Stopwatch(clock_ns=lambda: next(calls)) as sw:
        pass
    assert sw.elapsed_ms == expected_elapsed_ms, desc


def test_stopwatch_raises_if_elapsed_ms_accessed_before_use() -> None:
    sw = Stopwatch(clock_ns=lambda: 0)
    with pytest.raises(RuntimeError, match="has not been used"):
        _ = sw.elapsed_ms


# | load | ttft_vendor | ttft_external | total   | tokens | expected_tps | desc                          |
@pytest.mark.parametrize(
    ("load", "ttft_vendor", "ttft_external", "total", "tokens", "expected_tps", "desc"),
    [
        (1000.0, 1500.0, 1500.0, 2500.0, 10,  10.0,  "10 tokens in 1000 ms decode → 10 tok/s"),
        (1000.0, 2000.0, 2000.0, 3000.0, 1,   1.0,   "single token in 1 s → 1 tok/s"),
        (1000.0, 1500.0, 1500.0, 1500.0, 0,   0.0,   "zero tokens → 0.0 tok/s (no decode window)"),
        (1000.0, 1500.0, 1500.0, 1500.0, 10,  0.0,   "zero decode window with >0 tokens → 0.0"),
        (1000.0, 1500.0, 1600.0, 1500.0, 5,   0.0,   "negative decode window → 0.0 (clamped)"),
    ],
)
def test_timing_record_tokens_per_sec(
    load: float,
    ttft_vendor: float,
    ttft_external: float,
    total: float,
    tokens: int,
    expected_tps: float,
    desc: str,
) -> None:
    rec = TimingRecord(
        wall_ms_load=load,
        wall_ms_ttft_vendor=ttft_vendor,
        wall_ms_ttft_external=ttft_external,
        wall_ms_total=total,
        tokens_generated=tokens,
    )
    assert rec.tokens_per_sec == pytest.approx(expected_tps), desc


# | field_path             | expected_value              | desc                                       |
@pytest.mark.parametrize(
    ("field_path", "expected_value", "desc"),
    [
        (("prompt_id",),                     "P1",                         "top-level prompt_id round-trips"),
        (("prompt_class",),                  "fact_lookup",                "prompt_class round-trips"),
        (("response_text",),                 "Your heart rate is 72 bpm.", "response_text round-trips"),
        (("timing", "wall_ms_total"),        7500.0,                        "nested timing field round-trips"),
        (("timing", "tokens_generated"),     50,                            "nested tokens_generated round-trips"),
        (("peak_rss_mb",),                   1134.5,                        "peak_rss_mb round-trips"),
        (("cma_free_kb_during",),            4,                             "cma_free_kb_during round-trips"),
        (("error",),                         None,                          "null error round-trips as JSON null"),
    ],
)
def test_bench_row_to_jsonl_dict_roundtrip(
    field_path: tuple[str, ...], expected_value: object, desc: str
) -> None:
    row = _make_row()
    d: object = row.to_jsonl_dict()
    for key in field_path:
        assert isinstance(d, dict), f"expected dict at {key}, got {type(d).__name__}"
        d = d[key]
    assert d == expected_value, desc


def test_bench_row_to_jsonl_dict_injects_tokens_per_sec() -> None:
    row = _make_row()
    d = row.to_jsonl_dict()
    timing = d["timing"]
    assert isinstance(timing, dict), "timing must round-trip as nested dict"
    # 50 tokens over (7500 - 1520) = 5980 ms decode window → ≈ 8.361 tok/s.
    assert timing["tokens_per_sec"] == pytest.approx(50 / 5980 * 1000), (
        "tokens_per_sec must be injected as a non-field JSONL column"
    )


def test_write_row_appends_valid_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    write_row(out, _make_row(prompt_id="P1"))
    write_row(out, _make_row(prompt_id="P2"))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "expected 2 JSONL lines after 2 write_row calls"
    parsed = [json.loads(line) for line in lines]
    assert [p["prompt_id"] for p in parsed] == ["P1", "P2"], (
        "JSONL preserves write order and each line is valid JSON"
    )


def test_write_rows_batched_equivalent_to_sequential(tmp_path: Path) -> None:
    seq = tmp_path / "seq.jsonl"
    batched = tmp_path / "batched.jsonl"
    rows = [_make_row(prompt_id=f"P{i}") for i in range(3)]
    for r in rows:
        write_row(seq, r)
    write_rows(batched, rows)
    assert seq.read_text(encoding="utf-8") == batched.read_text(encoding="utf-8"), (
        "write_rows must be byte-identical to a write_row loop"
    )


def test_write_row_preserves_unicode_without_escape(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    write_row(out, _make_row(response_text="Take with food — avoid grapefruit."))
    text = out.read_text(encoding="utf-8")
    assert "—" in text, "em-dash must survive as literal Unicode, not \\u escaped"


# Real on-board /proc/meminfo has a couple hundred lines; this fixture keeps
# only what our readers touch, in the actual column-aligned kernel format.
_MEMINFO_FIXTURE = """\
MemTotal:        1965824 kB
MemFree:          421568 kB
MemAvailable:    1437120 kB
CmaTotal:         524288 kB
CmaFree:          425192 kB
"""

_STATUS_FIXTURE = """\
Name:\tpython3
State:\tR (running)
VmSize:\t 1295436 kB
VmHWM:\t   76840 kB
VmRSS:\t   76840 kB
"""


def _write_proc(tmp_path: Path, meminfo: str = _MEMINFO_FIXTURE, status: str = _STATUS_FIXTURE) -> tuple[Path, Path]:
    meminfo_path = tmp_path / "meminfo"
    status_path = tmp_path / "status"
    meminfo_path.write_text(meminfo)
    status_path.write_text(status)
    return meminfo_path, status_path


# | meminfo_text                 | expected_cma_free_kb | desc                              |
@pytest.mark.parametrize(
    ("meminfo_text", "expected_cma_free_kb", "desc"),
    [
        (_MEMINFO_FIXTURE,                          425_192, "canonical on-board meminfo fixture"),
        ("CmaFree:               0 kB\n",           0,       "zero CMA free is valid (not sentinel)"),
        ("CmaTotal: 100 kB\nCmaFree: 1 kB\n",       1,       "last-line CmaFree is read past earlier fields"),
    ],
)
def test_read_cma_free_kb_parses_canonical_format(
    tmp_path: Path, meminfo_text: str, expected_cma_free_kb: int, desc: str
) -> None:
    meminfo_path, _ = _write_proc(tmp_path, meminfo=meminfo_text)
    assert read_cma_free_kb(meminfo_path) == expected_cma_free_kb, desc


def test_read_cma_free_kb_raises_if_field_absent(tmp_path: Path) -> None:
    meminfo_path, _ = _write_proc(tmp_path, meminfo="MemTotal: 100 kB\n")
    with pytest.raises(KeyError, match="CmaFree"):
        read_cma_free_kb(meminfo_path)


# | status_text                                        | expected_rss_mb       | desc                           |
@pytest.mark.parametrize(
    ("status_text", "expected_rss_mb", "desc"),
    [
        (_STATUS_FIXTURE,                                     76840 / 1024,     "canonical status fixture"),
        ("VmRSS:\t 1048576 kB\n",                             1024.0,            "exactly 1 GiB VmRSS"),
        ("Name:\tx\nVmRSS:\t    0 kB\nVmHWM:\t 100 kB\n",     0.0,               "zero RSS is valid"),
    ],
)
def test_read_rss_mb_parses_kib_to_mib(
    tmp_path: Path, status_text: str, expected_rss_mb: float, desc: str
) -> None:
    _, status_path = _write_proc(tmp_path, status=status_text)
    assert read_rss_mb(status_path) == pytest.approx(expected_rss_mb), desc


def test_memory_sampler_sample_once_tracks_peak_and_min(tmp_path: Path) -> None:
    meminfo_path, status_path = _write_proc(tmp_path)
    sampler = MemorySampler(
        interval_s=0.01, meminfo_path=meminfo_path, status_path=status_path
    )
    # Baseline sample.
    sampler.sample_once()
    assert sampler.peak_rss_mb == pytest.approx(76840 / 1024), "baseline RSS tracked"
    assert sampler.min_cma_free_kb == 425_192, "baseline CmaFree tracked"

    # Second /proc rewrite → higher RSS, lower CmaFree. Peak must climb;
    # min must fall.
    status_path.write_text("VmRSS:\t  200000 kB\n")
    meminfo_path.write_text("CmaFree:         4 kB\n")
    sampler.sample_once()
    assert sampler.peak_rss_mb == pytest.approx(200000 / 1024), (
        "peak follows the higher RSS reading"
    )
    assert sampler.min_cma_free_kb == 4, "min follows the lower CmaFree"

    # Third rewrite → RSS drops, CmaFree recovers. Peak/min must NOT move.
    status_path.write_text("VmRSS:\t    1000 kB\n")
    meminfo_path.write_text("CmaFree:    100000 kB\n")
    sampler.sample_once()
    assert sampler.peak_rss_mb == pytest.approx(200000 / 1024), (
        "peak is sticky when RSS drops"
    )
    assert sampler.min_cma_free_kb == 4, "min is sticky when CmaFree recovers"


def test_memory_sampler_tolerates_missing_cma_free(tmp_path: Path) -> None:
    """Host (WSL) kernels lack CmaFree. Sampler keeps going, just doesn't
    update the min-CMA tracker. This is how the same harness runs on both
    the host (for developer smoke tests) and the board."""
    meminfo_path = tmp_path / "meminfo"
    status_path = tmp_path / "status"
    meminfo_path.write_text("MemTotal: 100 kB\n")  # no CmaFree
    status_path.write_text("VmRSS:\t 50000 kB\n")
    sampler = MemorySampler(
        interval_s=0.01, meminfo_path=meminfo_path, status_path=status_path
    )
    sampler.sample_once()
    assert sampler.peak_rss_mb == pytest.approx(50000 / 1024), "RSS updates"
    # CmaFree absent → min stays at sentinel; caller can detect via samples_taken>0.
    assert sampler.samples_taken == 1, "sample counted even without CmaFree"


def test_memory_sampler_thread_lifecycle_collects_samples(tmp_path: Path) -> None:
    meminfo_path, status_path = _write_proc(tmp_path)
    with MemorySampler(
        interval_s=0.005, meminfo_path=meminfo_path, status_path=status_path
    ) as sampler:
        # Let the daemon tick a few times — a tight sleep is unavoidable for
        # lifecycle coverage; 50 ms at 5 ms interval → ~10 ticks.
        time.sleep(0.05)
    assert sampler.samples_taken > 0, "daemon thread must have sampled at least once"
    assert sampler.peak_rss_mb == pytest.approx(76840 / 1024), (
        "peak matches the static fixture value"
    )


def test_load_prompt_suite_round_trips_canonical_fixture() -> None:
    """The canonical prompts.yaml must load cleanly; any drift is a contract
    break that Phase D benchmarks would silently ingest."""
    suite = load_prompt_suite(CANONICAL_PROMPTS)
    assert len(suite) >= 6, "prompts.yaml must have at least the minimal 6-prompt suite"
    assert all(isinstance(p, PromptSpec) for p in suite), "all entries typed"
    # Expected ids from the 2026-04-24 fixture — failure here = spec drift.
    ids = {p.id for p in suite}
    required_ids = {"C1", "P1", "P2", "D1", "D2"}
    assert required_ids <= ids, f"missing prompt ids: {sorted(required_ids - ids)}"


def test_load_prompt_suite_preserves_order(tmp_path: Path) -> None:
    """Bench output rows must appear in the order prompts.yaml declares so
    warmup effects are consistently attributable."""
    fixture = tmp_path / "prompts.yaml"
    fixture.write_text(
        """prompts:
          - {id: Z1, class: calibration,     text: z, pass_pattern: ".", pattern_flags: ""}
          - {id: A1, class: fact_lookup,     text: a, pass_pattern: ".", pattern_flags: ""}
          - {id: M1, class: fact_absence,    text: m, pass_pattern: ".", pattern_flags: ""}
        """
    )
    suite = load_prompt_suite(fixture)
    assert [p.id for p in suite] == ["Z1", "A1", "M1"], "YAML order preserved"


# | yaml_text                                                              | match_substring              | desc                                              |
@pytest.mark.parametrize(
    ("yaml_text", "match_substring", "desc"),
    [
        ("prompts: []\n",                                                    "non-empty list",             "empty prompt list rejected"),
        ("not a mapping\n",                                                  "top-level",                   "non-mapping root rejected"),
        ("prompts:\n  - {id: P1, class: fact_lookup, text: x, pass_pattern: '.'}\n", "missing required keys", "missing pattern_flags rejected"),
        ("prompts:\n  - {id: P1, class: bogus, text: x, pass_pattern: '.', pattern_flags: ''}\n", "class 'bogus'",       "unknown class rejected"),
        ("prompts:\n  - {id: '',  class: fact_lookup, text: x, pass_pattern: '.', pattern_flags: ''}\n", "non-empty string",    "empty id rejected"),
        ("prompts:\n  - {id: P1, class: fact_lookup, text: x, pass_pattern: '.', pattern_flags: ''}\n"
         "  - {id: P1, class: fact_lookup, text: x, pass_pattern: '.', pattern_flags: ''}\n", "duplicate id",         "duplicate id rejected"),
    ],
)
def test_load_prompt_suite_rejects_schema_drift(
    tmp_path: Path, yaml_text: str, match_substring: str, desc: str
) -> None:
    fixture = tmp_path / "prompts.yaml"
    fixture.write_text(yaml_text)
    with pytest.raises(ValueError, match=match_substring):
        load_prompt_suite(fixture)


# --------------------------------------------------------------------------
# Vendor Gemma3Static shim — host tests exercise the wrapper logic with a
# structural stub; the actual NPU call is board-only and covered by D7.
# --------------------------------------------------------------------------


class _StubGemma3:
    """Satisfies `_Gemma3Like` without touching torq.runtime.

    `run_stream` yields the scripted chunks one at a time; the vendor-side
    TTFT and token count reported afterwards are scripted too so the
    adapter's tuple unpacking is fully testable.
    """

    def __init__(
        self,
        scripted_chunks: list[str],
        ttft_ms: float = 1234.5,
        tokens: int = 17,
    ) -> None:
        self._chunks = scripted_chunks
        self._ttft = ttft_ms
        self._tokens = tokens
        self.last_input: str | None = None
        self.last_max_tokens: int | None = None

    def run_stream(
        self, user_input: str, max_tokens: int | None = None
    ) -> Iterator[str]:
        self.last_input = user_input
        self.last_max_tokens = max_tokens
        yield from self._chunks

    @property
    def time_to_first_token(self) -> float:
        return self._ttft

    @property
    def generated_tokens(self) -> int:
        return self._tokens


def test_gemma3_bench_adapter_run_forwards_args_and_metrics() -> None:
    stub = _StubGemma3(scripted_chunks=["Your heart rate ", "is 72 bpm."], ttft_ms=950.0, tokens=8)
    # Injected clock: [start_before_run, first_yield, second_yield, ...].
    ticks = iter([0, 1_500_000_000, 2_000_000_000, 3_000_000_000])
    adapter = Gemma3BenchAdapter(stub, max_gen_tokens=64, clock_ns=lambda: next(ticks))
    result = adapter.run("what is my heart rate?")
    assert result.text == "Your heart rate is 72 bpm.", "chunks joined verbatim"
    assert result.wall_ms_ttft_vendor == pytest.approx(950.0), "vendor TTFT forwarded"
    assert result.wall_ms_ttft_external == pytest.approx(1500.0), (
        "external TTFT stamped at first yield (1.5s after start)"
    )
    assert result.tokens_generated == 8, "generated_tokens forwarded"
    assert stub.last_input == "what is my heart rate?", "user_text reaches run_stream"
    assert stub.last_max_tokens == 64, "max_gen_tokens forwarded to run_stream"


def test_gemma3_bench_adapter_zero_chunks_reports_zero_external_ttft() -> None:
    """Model refused before any token emerged — external TTFT is unknowable;
    report 0.0 and let the scorer detect via tokens_generated == 0."""
    stub = _StubGemma3(scripted_chunks=[], ttft_ms=0.0, tokens=0)
    adapter = Gemma3BenchAdapter(stub, clock_ns=lambda: 0)
    result = adapter.run("say nothing")
    assert result.text == "", "no chunks → empty text"
    assert result.wall_ms_ttft_external == 0.0, "zero-chunk run reports 0.0 external TTFT"
    assert result.tokens_generated == 0, "vendor reports zero tokens"


def test_patched_default_sys_prompt_restores_original() -> None:
    fake_mod = types.ModuleType("fake_runner")
    setattr(fake_mod, "DEFAULT_SYS_PROMPT", "ORIGINAL")  # noqa: B010
    with _patched_default_sys_prompt(fake_mod, override=""):
        assert getattr(fake_mod, "DEFAULT_SYS_PROMPT") == "", (  # noqa: B009
            "override active inside context"
        )
    assert getattr(fake_mod, "DEFAULT_SYS_PROMPT") == "ORIGINAL", (  # noqa: B009
        "original value restored on exit"
    )


def test_patched_default_sys_prompt_restores_even_on_exception() -> None:
    fake_mod = types.ModuleType("fake_runner")
    setattr(fake_mod, "DEFAULT_SYS_PROMPT", "ORIGINAL")  # noqa: B010
    with pytest.raises(RuntimeError, match="boom"), _patched_default_sys_prompt(
        fake_mod, override="X"
    ):
        raise RuntimeError("boom")
    assert getattr(fake_mod, "DEFAULT_SYS_PROMPT") == "ORIGINAL", (  # noqa: B009
        "original restored even after the body raised"
    )


def test_create_gemma3_bench_adapter_raises_vendor_import_error_on_host(tmp_path: Path) -> None:
    """Host has no torq.runtime / runner module on sys.path — the factory
    must raise `VendorImportError`, NOT a bare ImportError, so callers can
    handle the expected dev-host path distinctly from real bugs."""
    with pytest.raises(VendorImportError, match="torq-examples"):
        create_gemma3_bench_adapter(
            model_path=tmp_path / "model.vmfb",
            torq_examples_root=tmp_path,  # empty dir → import fails
        )


def test_create_gemma3_bench_adapter_applies_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: when the vendor module IS importable, the factory must
    (a) call the Gemma3Static constructor exactly once, and (b) have the
    patched DEFAULT_SYS_PROMPT visible at that call time."""
    import importlib

    fake_mod = types.ModuleType("runner")
    setattr(fake_mod, "DEFAULT_SYS_PROMPT", "VENDOR_DEFAULT")  # noqa: B010
    observed_default: list[str] = []

    class _CaptureGemma3:
        def __init__(self, model_path: str, instruct_model: bool = False) -> None:
            # Read the current module-level DEFAULT_SYS_PROMPT through
            # the fake module so we can verify the patch is active here.
            observed_default.append(getattr(fake_mod, "DEFAULT_SYS_PROMPT"))  # noqa: B009
            self.model_path = model_path
            self.instruct_model = instruct_model

        def run_stream(
            self, user_input: str, max_tokens: int | None = None
        ) -> Iterator[str]:
            yield "ok"

        @property
        def time_to_first_token(self) -> float:
            return 1.0

        @property
        def generated_tokens(self) -> int:
            return 1

    setattr(fake_mod, "Gemma3Static", _CaptureGemma3)  # noqa: B010

    def _fake_import(name: str) -> Any:
        assert name == "runner", f"expected runner import, got {name}"
        return fake_mod

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    adapter = create_gemma3_bench_adapter(
        model_path=Path("/dev/null/model.vmfb"),
        max_gen_tokens=64,
    )
    assert isinstance(adapter, Gemma3BenchAdapter), "factory returns adapter"
    assert observed_default == [""], (
        "DEFAULT_SYS_PROMPT must be '' at Gemma3Static.__init__ time"
    )
    assert getattr(fake_mod, "DEFAULT_SYS_PROMPT") == "VENDOR_DEFAULT", (  # noqa: B009
        "original DEFAULT_SYS_PROMPT must be restored after factory returns"
    )
    assert adapter.max_gen_tokens == 64, "max_gen_tokens plumbed into adapter"


# --------------------------------------------------------------------------
# main() — host smoke. The board NPU call is stubbed via `adapter_factory`
# injection; everything else (argparse, prompt iteration, JSONL emission,
# error capture) runs for real against the canonical fixtures.
# --------------------------------------------------------------------------


def _make_stub_adapter(
    responses_by_id: dict[str, list[str]] | None = None,
    raise_for_ids: tuple[str, ...] = (),
) -> Callable[[_MainArgs], BenchAdapter]:
    """Return a fake `adapter_factory` that yields an adapter wrapping a
    scripted `_StubGemma3`. `responses_by_id` maps a substring of the user
    input to scripted chunks; unmatched prompts get `["ok"]`."""
    default = ["Your ", "answer ", "here."]

    class _DispatcherStub:
        def __init__(self) -> None:
            self.last_input: str | None = None
            self.last_max_tokens: int | None = None
            self._ttft = 500.0
            self._tokens = 3

        def run_stream(
            self, user_input: str, max_tokens: int | None = None
        ) -> Iterator[str]:
            self.last_input = user_input
            self.last_max_tokens = max_tokens
            if responses_by_id is not None:
                for key, chunks in responses_by_id.items():
                    if key in user_input:
                        if key in raise_for_ids:
                            raise RuntimeError(f"scripted failure for key={key!r}")
                        self._tokens = len(chunks)
                        yield from chunks
                        return
            yield from default
            self._tokens = len(default)

        @property
        def time_to_first_token(self) -> float:
            return self._ttft

        @property
        def generated_tokens(self) -> int:
            return self._tokens

    def _factory(args: _MainArgs) -> BenchAdapter:
        return Gemma3BenchAdapter(_DispatcherStub(), max_gen_tokens=args.max_gen_tokens)

    return _factory


def test_main_smoke_emits_one_row_per_prompt(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--max-gen-tokens", "32",
            "--sampler-interval-s", "0.01",
            "--now", "2026-04-24",
        ],
        adapter_factory=_make_stub_adapter(),
    )
    assert rc == 0, "clean exit code"
    lines = out.read_text(encoding="utf-8").splitlines()
    suite = load_prompt_suite(CANONICAL_PROMPTS)
    assert len(lines) == len(suite), (
        f"one JSONL row per prompt (got {len(lines)}, expected {len(suite)})"
    )
    rows = [json.loads(line) for line in lines]
    assert [r["prompt_id"] for r in rows] == [p.id for p in suite], (
        "prompt order preserved in JSONL output"
    )
    assert all(r["error"] is None for r in rows), "no errors under happy-path stub"


def test_main_ids_filter_limits_rows(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--ids", "C1,P1,D2",
            "--now", "2026-04-24",
            "--sampler-interval-s", "0.01",
        ],
        adapter_factory=_make_stub_adapter(),
    )
    assert rc == 0, "clean exit code"
    ids = [json.loads(line)["prompt_id"] for line in out.read_text().splitlines()]
    assert ids == ["C1", "P1", "D2"], "only the listed ids run, in yaml order"


def test_main_ids_filter_no_matches_returns_error_code(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--ids", "NONEXISTENT",
            "--now", "2026-04-24",
        ],
        adapter_factory=_make_stub_adapter(),
    )
    assert rc == 2, "no-match filter exits nonzero"
    assert not out.exists() or out.read_text() == "", "no output emitted on empty filter"


def test_main_captures_runtime_error_as_row_error_field(tmp_path: Path) -> None:
    """A RuntimeError in adapter.run() must not abort the sweep — record
    the error on that row and keep going. Critical for 8-12 min board runs
    where losing the last 10 prompts over one bad one is unacceptable."""
    out = tmp_path / "sweep.jsonl"
    # Scripted failure: only the P5 utterance ("can I eat grapefruit?")
    # triggers the adapter to raise. The substring "eat grapefruit" is
    # unique to P5 — the YAML block contains "no grapefruit" elsewhere,
    # so a bare "grapefruit" match would fire on every prompt.
    rc = main(
        [
            "--model-dir", str(tmp_path),
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--now", "2026-04-24",
            "--sampler-interval-s", "0.01",
        ],
        adapter_factory=_make_stub_adapter(
            responses_by_id={"eat grapefruit": ["(doomed)"]},
            raise_for_ids=("eat grapefruit",),
        ),
    )
    assert rc == 0, "main returns 0 even with a per-prompt failure"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    error_rows = [r for r in rows if r["error"]]
    assert len(error_rows) == 1, "exactly one prompt failed"
    assert error_rows[0]["prompt_id"] == "P5", "P5 is the grapefruit prompt"
    assert "RuntimeError" in error_rows[0]["error"], "error type captured"
    assert error_rows[0]["timing"]["tokens_generated"] == 0, (
        "failed run reports zero tokens"
    )


def test_safe_read_cma_free_kb_returns_zero_on_host() -> None:
    """Host kernels lack CmaFree; the helper must not raise."""
    value = _safe_read_cma_free_kb()
    assert value == 0, "host fallback returns 0 sentinel (not a real 0-kB reading)"


# --------------------------------------------------------------------------
# LlamaCompletionBenchAdapter — A55 CPU GGUF subprocess path (H4)
# All host tests; the binary itself is not on PATH (cross-compiled aarch64).
# --------------------------------------------------------------------------

import subprocess  # noqa: E402 — local to this section to keep top imports clean
from datetime import date as _date  # noqa: E402

from gemma_tools.health_table import load_health_table  # noqa: E402

# Fixture: a complete `common_perf_print` block as it appears on stderr in
# upstream llama.cpp `b8925`. Numbers come from gemma-on-a55-get-started.md
# §5.1 (the proven A55 baseline). Treating this as truth: any drift in
# upstream output format breaks this test loudly, which is exactly the
# signal we want before the runtime mismatch reaches a board run.
_PERF_FIXTURE = """\
llama_perf_sampler_print:    sampling time =      18.42 ms /   103 runs   (    0.18 ms per token,  5594.13 tokens per second)
llama_perf_context_print:        load time =    3760.43 ms
llama_perf_context_print: prompt eval time =    2210.92 ms /    82 tokens (   26.96 ms per token,    37.09 tokens per second)
llama_perf_context_print:        eval time =    3572.31 ms /    21 runs   (  170.11 ms per token,     5.88 tokens per second)
llama_perf_context_print:       total time =   12944.02 ms /   103 tokens
"""


def _llama_argv_template() -> tuple[Path, Path]:
    """Stable (binary, model) paths for ctor tests — never opened, just
    rendered into argv strings."""
    return Path("/mnt/sdcard/llama-cpp/llama-completion"), Path(
        "/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf"
    )


def test_wrap_gemma3_chat_template_round_trips_compose_prompt() -> None:
    """The adapter's chat-template wrap MUST byte-match the user-facing
    `compose_prompt(candidate="gemma3", ...)` for the same composed body.
    If this test fails, train-time and inference-time prompts have drifted
    — exactly the failure mode `slm-system-prompt.md §3` warns about.
    """
    repo = _REPO
    health = load_health_table(repo / "data" / "health_table_v1.yaml")
    now = _date(2026, 4, 25)
    user_text = (
        "ROLE: health-records assistant on SL2619 edge device.\n"
        "(simulated body)\n"
        "what is my heart rate?"
    )
    direct = wrap_gemma3_chat_template(user_text)
    expected_prefix = "<start_of_turn>user\n"
    expected_suffix = "<end_of_turn>\n<start_of_turn>model\n"
    assert direct.startswith(expected_prefix), "wrap opens with Gemma user-turn marker"
    assert direct.endswith(expected_suffix), "wrap closes with model-turn marker"
    assert user_text in direct, "user_text passes through verbatim"
    # Cross-check against compose_prompt for a real composed body.
    from gemma_tools.prompt_composer import compose_user_text
    composed = compose_user_text(health, now, "what is my heart rate?")
    assert wrap_gemma3_chat_template(composed) == compose_prompt(
        candidate="gemma3", utterance="what is my heart rate?",
        health=health, now=now,
    ), "adapter wrap is byte-equivalent to compose_prompt(gemma3)"


# | n_threads | n_predict | temp | top_k | seed | desc                                |
@pytest.mark.parametrize(
    ("n_threads", "n_predict", "temp", "top_k", "seed", "desc"),
    [
        (2, 128, 0.0,  1,  42,  "canonical A55 baseline (-t 2, deterministic)"),
        (4, 64,  0.7,  40, 1,   "exploration sampling (top_k 40, temp 0.7)"),
        (1, 256, 0.0,  1,  0,   "single-thread, large budget, seed 0"),
    ],
)
def test_llama_completion_build_command_argv_shape(
    n_threads: int, n_predict: int, temp: float, top_k: int, seed: int, desc: str
) -> None:
    binary, model = _llama_argv_template()
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model,
        n_threads=n_threads, n_predict=n_predict,
        temp=temp, top_k=top_k, seed=seed,
    )
    prompt_file = Path("/tmp/probe.txt")
    argv = adapter.build_command(prompt_file)
    assert argv[0] == str(binary), f"{desc} — binary first"
    assert argv[-1] == "-no-cnv", f"{desc} — -no-cnv last (headless flag)"
    assert ["-m", str(model)] == argv[1:3], f"{desc} — -m model"
    assert ["-f", str(prompt_file)] == argv[3:5], f"{desc} — -f prompt"
    assert ["-t", str(n_threads)] == argv[5:7], f"{desc} — -t threads"
    assert ["-n", str(n_predict)] == argv[7:9], f"{desc} — -n predict"
    assert ["--temp", str(temp)] == argv[9:11], f"{desc} — --temp"
    assert ["--top-k", str(top_k)] == argv[11:13], f"{desc} — --top-k"
    assert ["--seed", str(seed)] == argv[13:15], f"{desc} — --seed"


# | stdout_text                                                                       | expected                       | desc                                          |
@pytest.mark.parametrize(
    ("stdout_text", "expected", "desc"),
    [
        ("<start_of_turn>user\nask<end_of_turn>\n<start_of_turn>model\n72 bpm.<end_of_turn>\n",
         "72 bpm.",
         "canonical wrap — slice between model-turn open and end-turn"),
        ("<start_of_turn>user\nask<end_of_turn>\n<start_of_turn>model\nOkay, I understand.[end of text]\n",
         "Okay, I understand.",
         "[end of text] terminator stripped"),
        ("<start_of_turn>user\nask<end_of_turn>\n<start_of_turn>model\n  spaced out  ",
         "spaced out",
         "trailing whitespace stripped"),
        ("no marker just plain output",
         "no marker just plain output",
         "missing model-turn marker → fallback to stripped stdout"),
        ("<start_of_turn>model\nfirst<end_of_turn>\n<start_of_turn>model\nsecond<end_of_turn>\n",
         "second",
         "rfind picks the last marker (handles hallucinated mid-stream marker)"),
        ("<start_of_turn>user\nx<end_of_turn>\n<start_of_turn>model\n",
         "",
         "empty body after marker → empty string"),
        # Real stdout shape from on-board `b8925/0adede8` llama-completion
        # (captured 2026-04-27). The chat-template special tokens are
        # detokenized to empty strings by default, so the wire sees the
        # bare `\nmodel\n` role divider — not `<start_of_turn>model\n`.
        # Without this case, H6's pass/fail signal is corrupt: the regex
        # matches the echoed YAML in the prompt's user turn instead of
        # the model's actual reply.
        ("user\nyou are an assistant\nQuestion: heart rate?\nmodel\n```yaml\nheart_rate_bpm: 72\n```[end of text]\n",
         "```yaml\nheart_rate_bpm: 72\n```",
         "b8925 detokenized markers — slice on bare `\\nmodel\\n` role divider"),
        ("user\nfirst question\nmodel\nfirst answer\nuser\nsecond question\nmodel\nsecond answer[end of text]",
         "second answer",
         "detokenized form — rfind picks the last `\\nmodel\\n`"),
        ("user\nask\nmodel\nplain reply, no terminator",
         "plain reply, no terminator",
         "detokenized form — no [end of text] terminator, return stripped body"),
    ],
)
def test_parse_completion_response_handles_real_stdout_shapes(
    stdout_text: str, expected: str, desc: str
) -> None:
    assert parse_completion_response(stdout_text) == expected, desc


def test_parse_llama_perf_canonical_block() -> None:
    perf = parse_llama_perf(_PERF_FIXTURE)
    assert perf.wall_ms_load == pytest.approx(3760.43), "load time parsed"
    assert perf.wall_ms_prompt_eval == pytest.approx(2210.92), "prompt eval ms"
    assert perf.n_prompt_tokens == 82, "prompt eval token count"
    assert perf.prompt_eval_tps == pytest.approx(37.09), "prompt eval tps"
    assert perf.wall_ms_decode == pytest.approx(3572.31), "decode ms"
    assert perf.n_decode_tokens == 21, "decode token count"
    assert perf.decode_tps == pytest.approx(5.88), "decode tps"
    assert perf.wall_ms_total == pytest.approx(12944.02), "total ms"


# Real stderr block captured 2026-04-27 from the on-board `b8925/0adede8`
# llama-completion (gemma-on-a55-get-started.md §3.7 deterministic probe).
# Upstream renamed the print site between this build and our local
# `665abc609` checkout: prefix is `common_perf_print:` here, not
# `llama_perf_context_print:`. Same fields, same column order.
# `unaccounted` and `graphs reused` lines are accepted-but-ignored.
_PERF_FIXTURE_B8925 = """\
common_perf_print:    sampling time =      57.09 ms
common_perf_print:    samplers time =      24.44 ms /   104 tokens
common_perf_print:        load time =    3252.55 ms
common_perf_print: prompt eval time =     859.74 ms /    82 tokens (   10.48 ms per token,    95.38 tokens per second)
common_perf_print:        eval time =    1355.04 ms /    21 runs   (   64.53 ms per token,    15.50 tokens per second)
common_perf_print:       total time =    2276.43 ms /   103 tokens
common_perf_print: unaccounted time =       4.57 ms /   0.2 %      (total - sampling - prompt eval - eval) / (total)
common_perf_print:    graphs reused =         20
common_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
"""


def test_parse_llama_perf_b8925_common_perf_print_prefix() -> None:
    """b8925 (`0adede8`) renamed the print site to `common_perf_print:`.
    Same fields and ordering as before — the parser must accept either
    prefix so the same JSONL schema survives an upstream rename. This is
    the format actually emitted by the on-board binary at H6 time."""
    perf = parse_llama_perf(_PERF_FIXTURE_B8925)
    assert perf.wall_ms_load == pytest.approx(3252.55), "load time parsed (b8925 prefix)"
    assert perf.wall_ms_prompt_eval == pytest.approx(859.74), "prompt eval ms"
    assert perf.n_prompt_tokens == 82, "prompt eval token count"
    assert perf.prompt_eval_tps == pytest.approx(95.38), "prompt eval tps"
    assert perf.wall_ms_decode == pytest.approx(1355.04), "decode ms"
    assert perf.n_decode_tokens == 21, "decode token count"
    assert perf.decode_tps == pytest.approx(15.50), "decode tps"
    assert perf.wall_ms_total == pytest.approx(2276.43), "total ms"


def test_parse_llama_perf_missing_load_raises() -> None:
    """`load time = ...` is the one mandatory field — without it we'd
    silently report 0 ms cold-load on every row, fooling the bench."""
    stream = "llama_perf_context_print: total time = 100.0 ms / 5 tokens\n"
    with pytest.raises(ValueError, match="load time"):
        parse_llama_perf(stream)


def test_parse_llama_perf_load_only_minimal_block() -> None:
    """Some forks emit only the load line in degenerate cases (e.g. -n 0).
    Parser must still succeed; missing optional fields default to 0."""
    stream = "llama_perf_context_print: load time =   100.5 ms\n"
    perf = parse_llama_perf(stream)
    assert perf.wall_ms_load == pytest.approx(100.5), "load extracted"
    assert perf.n_prompt_tokens == 0, "absent prompt-eval defaults to 0 tokens"
    assert perf.n_decode_tokens == 0, "absent decode defaults to 0 tokens"
    assert perf.wall_ms_total == 0.0, "absent total defaults to 0.0 ms"


def _make_completed_process(
    *, returncode: int, stdout: str, stderr: str
) -> subprocess.CompletedProcess[str]:
    """Synthesize the dataclass `subprocess.run` returns under text mode."""
    return subprocess.CompletedProcess(
        args=["/fake/llama-completion"], returncode=returncode,
        stdout=stdout, stderr=stderr,
    )


def test_llama_completion_adapter_run_happy_path() -> None:
    """Stubbed runner returns a canonical perf block on stderr and a
    template-wrapped response on stdout. Adapter must surface the
    response text and per-call wall_ms_load."""
    captured_argv: list[list[str]] = []

    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        captured_argv.append(argv)
        stdout = (
            "<start_of_turn>user\n"
            "what is my heart rate?<end_of_turn>\n"
            "<start_of_turn>model\n72 bpm.<end_of_turn>\n[end of text]\n"
        )
        return _make_completed_process(returncode=0, stdout=stdout, stderr=_PERF_FIXTURE)

    binary, model = _llama_argv_template()
    # Inject deterministic clock: start_ns → 0, end_ns → 13_000_000_000 (13s wall).
    ticks = iter([0, 13_000_000_000])
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model, runner=_runner,
        clock_ns=lambda: next(ticks),
    )
    result = adapter.run("what is my heart rate?")
    assert result.text == "72 bpm.", "response sliced from stdout"
    assert result.wall_ms_load == pytest.approx(3760.43), (
        "per-call load surfaced from perf footer"
    )
    assert result.wall_ms_ttft_vendor == pytest.approx(3760.43 + 2210.92), (
        "vendor TTFT = load + prompt_eval (first decoded token lands "
        "exactly when prompt eval finishes)"
    )
    assert result.wall_ms_ttft_external == pytest.approx(13000.0), (
        "external TTFT = subprocess wall in ms"
    )
    assert result.tokens_generated == 21, "decode-run count from perf footer"
    assert len(captured_argv) == 1, "runner called exactly once"
    argv = captured_argv[0]
    assert argv[0] == str(binary), "binary in argv[0]"
    # The temp prompt file path is generated; verify -f is followed by an
    # existing-shape path that ends in .txt.
    f_idx = argv.index("-f")
    assert argv[f_idx + 1].endswith(".txt"), "-f points at a .txt prompt file"


def test_llama_completion_adapter_run_writes_chat_template_to_prompt_file() -> None:
    """The adapter MUST write the Gemma-3-wrapped prompt to disk for
    `-f` to read; verify the file content during the runner's sync call."""
    seen_prompts: list[str] = []

    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        f_idx = argv.index("-f")
        prompt_path = Path(argv[f_idx + 1])
        seen_prompts.append(prompt_path.read_text(encoding="utf-8"))
        return _make_completed_process(returncode=0, stdout=seen_prompts[-1], stderr=_PERF_FIXTURE)

    binary, model = _llama_argv_template()
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model, runner=_runner,
    )
    adapter.run("ROLE: assistant.\nwhat is my BP?")
    assert seen_prompts, "runner saw at least one prompt"
    body = seen_prompts[0]
    assert body.startswith("<start_of_turn>user\n"), "Gemma user-turn open"
    assert body.endswith("<end_of_turn>\n<start_of_turn>model\n"), "model-turn open is the last marker"
    assert "ROLE: assistant." in body, "user_text passes through verbatim"


def test_llama_completion_adapter_run_nonzero_exit_raises() -> None:
    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        return _make_completed_process(
            returncode=1, stdout="", stderr="failed to mmap model: bad magic\n",
        )

    binary, model = _llama_argv_template()
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model, runner=_runner,
    )
    with pytest.raises(LlamaCompletionError, match=r"exited 1"):
        adapter.run("anything")


def test_llama_completion_adapter_run_unparseable_perf_raises() -> None:
    """Subprocess returned 0 but emitted no perf footer — must not
    silently report zeros; raise so the main loop captures the error."""
    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        return _make_completed_process(
            returncode=0, stdout="<start_of_turn>model\n42<end_of_turn>\n",
            stderr="some unrelated diagnostic\n",
        )

    binary, model = _llama_argv_template()
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model, runner=_runner,
    )
    with pytest.raises(LlamaCompletionError, match=r"could not parse llama_perf"):
        adapter.run("anything")


def test_llama_completion_adapter_run_timeout_raises() -> None:
    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout_s or 0)

    binary, model = _llama_argv_template()
    adapter = LlamaCompletionBenchAdapter(
        binary_path=binary, model_path=model, runner=_runner,
        subprocess_timeout_s=0.1,
    )
    with pytest.raises(LlamaCompletionError, match=r"timed out after 0.1s"):
        adapter.run("anything")


def test_create_llama_completion_bench_adapter_factory() -> None:
    """Factory returns an adapter with constructor args plumbed through.
    No environment dependencies — host-clean unlike the vendor factory."""
    binary, model = _llama_argv_template()
    adapter = create_llama_completion_bench_adapter(
        binary_path=binary, model_path=model, n_threads=2, n_predict=64,
        temp=0.0, top_k=1, seed=42,
    )
    assert adapter.binary_path == binary, "binary path plumbed"
    assert adapter.model_path == model, "model path plumbed"
    assert adapter.n_threads == 2, "n_threads plumbed"
    assert adapter.n_predict == 64, "n_predict plumbed"


# --------------------------------------------------------------------------
# main() smoke — adapter=llama_completion path, full prompts.yaml suite.
# --------------------------------------------------------------------------


def _make_llama_stub_factory(
    response_for_id: dict[str, str] | None = None,
) -> Callable[[_MainArgs], BenchAdapter]:
    """Wraps a fake LlamaCompletionBenchAdapter whose subprocess runner
    returns one canonical perf block + a per-id response."""
    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        f_idx = argv.index("-f")
        prompt = Path(argv[f_idx + 1]).read_text(encoding="utf-8")
        # Match against the user_text segment between the turn markers.
        body = prompt.split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
        response = "default response"
        if response_for_id is not None:
            for key, txt in response_for_id.items():
                if key in body:
                    response = txt
                    break
        stdout = (
            f"<start_of_turn>user\n{body}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>\n[end of text]\n"
        )
        return _make_completed_process(returncode=0, stdout=stdout, stderr=_PERF_FIXTURE)

    def _factory(args: _MainArgs) -> BenchAdapter:
        assert args.adapter == "llama_completion", "factory called for llama path"
        return LlamaCompletionBenchAdapter(
            binary_path=args.llama_binary or Path("/fake/llama-completion"),
            model_path=args.llama_model or Path("/fake/model.gguf"),
            n_threads=args.n_threads, n_predict=args.max_gen_tokens,
            temp=args.temp, top_k=args.top_k, seed=args.seed,
            runner=_runner,
        )

    return _factory


def test_main_smoke_llama_completion_full_suite(tmp_path: Path) -> None:
    """End-to-end through `main()` with adapter=llama_completion: every
    prompt in the canonical suite produces a JSONL row with the new
    self-contained scorer fields populated."""
    out = tmp_path / "sweep.jsonl"
    rc = main(
        [
            "--adapter", "llama_completion",
            "--llama-binary", "/fake/llama-completion",
            "--llama-model", "/fake/model.gguf",
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--max-gen-tokens", "32",
            "--sampler-interval-s", "0.01",
            "--now", "2026-04-25",
        ],
        # Match P1's pass_pattern "72" so we exercise both PASS and FAIL.
        adapter_factory=_make_llama_stub_factory(
            response_for_id={"heart rate": "72 bpm.", "joke": "no comment"}
        ),
    )
    assert rc == 0, "clean exit"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    suite = load_prompt_suite(CANONICAL_PROMPTS)
    assert len(rows) == len(suite), (
        f"one JSONL row per prompt (got {len(rows)}, expected {len(suite)})"
    )
    # Self-contained scorer fields populated.
    p1_row = next(r for r in rows if r["prompt_id"] == "P1")
    assert p1_row["pass_pattern"] == "72", "pass_pattern from prompts.yaml is on row"
    assert p1_row["pattern_flags"] == "", "pattern_flags from prompts.yaml is on row"
    assert p1_row["passed_regex"] is True, '"72 bpm." matches pattern "72" → PASS'
    assert p1_row["response_text"] == "72 bpm.", "response sliced from stub stdout"
    # Per-call wall_ms_load is populated from the canonical perf fixture.
    assert p1_row["timing"]["wall_ms_load"] == pytest.approx(3760.43), (
        "per-call mmap cost surfaced — vendor path's sweep-level value would be "
        "much smaller (factory built fast; stub has no real load)"
    )
    # D1 (joke) does NOT match "health record" — must FAIL.
    d1_row = next(r for r in rows if r["prompt_id"] == "D1")
    assert d1_row["passed_regex"] is False, "stub response 'no comment' fails D1 pattern"


def test_main_smoke_llama_completion_subprocess_failure_recorded(tmp_path: Path) -> None:
    """A LlamaCompletionError on one prompt must not abort the sweep — it
    becomes an error row, the rest of the suite continues."""
    def _runner(argv: list[str], timeout_s: float | None) -> subprocess.CompletedProcess[str]:
        f_idx = argv.index("-f")
        prompt = Path(argv[f_idx + 1]).read_text(encoding="utf-8")
        if "joke" in prompt:
            return _make_completed_process(returncode=1, stdout="", stderr="boom\n")
        stdout = (
            "<start_of_turn>user\nx<end_of_turn>\n"
            "<start_of_turn>model\nok<end_of_turn>\n[end of text]\n"
        )
        return _make_completed_process(returncode=0, stdout=stdout, stderr=_PERF_FIXTURE)

    def _factory(args: _MainArgs) -> BenchAdapter:
        return LlamaCompletionBenchAdapter(
            binary_path=Path("/fake/llama"), model_path=Path("/fake/model.gguf"),
            runner=_runner,
        )

    out = tmp_path / "sweep.jsonl"
    rc = main(
        [
            "--adapter", "llama_completion",
            "--llama-binary", "/fake/llama",
            "--llama-model", "/fake/model.gguf",
            "--prompts", str(CANONICAL_PROMPTS),
            "--health-table", str(CANONICAL_HEALTH),
            "--output", str(out),
            "--sampler-interval-s", "0.01",
            "--now", "2026-04-25",
        ],
        adapter_factory=_factory,
    )
    assert rc == 0, "main returns 0 even with a per-prompt subprocess failure"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    error_rows = [r for r in rows if r["error"]]
    assert len(error_rows) == 1, "exactly one prompt failed"
    assert error_rows[0]["prompt_id"] == "D1", "D1 'tell me a joke' is the failing prompt"
    assert "LlamaCompletionError" in error_rows[0]["error"], "error type captured"
    assert error_rows[0]["passed_regex"] is False, "errored row scored FAIL"


def test_main_rejects_llama_adapter_without_required_args(tmp_path: Path) -> None:
    """CLI must reject `--adapter llama_completion` without binary+model
    so a typo doesn't silently fall back to the vendor path."""
    out = tmp_path / "sweep.jsonl"
    with pytest.raises(SystemExit, match="--llama-binary"):
        main(
            [
                "--adapter", "llama_completion",
                "--prompts", str(CANONICAL_PROMPTS),
                "--health-table", str(CANONICAL_HEALTH),
                "--output", str(out),
            ],
        )


def test_main_rejects_vendor_adapter_without_model_dir(tmp_path: Path) -> None:
    out = tmp_path / "sweep.jsonl"
    with pytest.raises(SystemExit, match="--model-dir"):
        main(
            [
                "--adapter", "gemma3_vendor",
                "--prompts", str(CANONICAL_PROMPTS),
                "--health-table", str(CANONICAL_HEALTH),
                "--output", str(out),
            ],
        )


def test_default_adapter_factory_dispatches_on_adapter(tmp_path: Path) -> None:
    """Smoke-check that the default dispatcher routes both names. Vendor
    path will raise VendorImportError on the host (no torq.runtime); we
    catch it as the proof the dispatcher reached the vendor branch."""
    args = _MainArgs(
        adapter="gemma3_vendor", prompts=tmp_path / "p", health_table=tmp_path / "h",
        output=tmp_path / "o", max_gen_tokens=32, sampler_interval_s=0.01,
        now=None, only_ids=(),
        model_dir=tmp_path, torq_examples_root=tmp_path,
        llama_binary=None, llama_model=None,
        n_threads=2, temp=0.0, top_k=1, seed=42, subprocess_timeout_s=120.0,
    )
    with pytest.raises(VendorImportError):
        default_adapter_factory(args)
    # Llama branch reaches the constructor without raising — no env deps.
    args_llama = _MainArgs(
        adapter="llama_completion", prompts=tmp_path / "p",
        health_table=tmp_path / "h", output=tmp_path / "o",
        max_gen_tokens=32, sampler_interval_s=0.01, now=None, only_ids=(),
        model_dir=None, torq_examples_root=None,
        llama_binary=Path("/fake/llama"), llama_model=Path("/fake/model.gguf"),
        n_threads=2, temp=0.0, top_k=1, seed=42, subprocess_timeout_s=120.0,
    )
    adapter = default_adapter_factory(args_llama)
    assert isinstance(adapter, LlamaCompletionBenchAdapter), "llama branch returns the right class"
