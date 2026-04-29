"""Host unit tests for `gemma_tools.bench_remote` — the host-driven
on-board bench harness.

Each test uses a (description, …) tuple table per
`docs/conventions/testing.md §3.1`. The SSH runner is
stubbed via the injectable `runner` parameter — no live board is touched.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from gemma_tools.bench_prompt import PromptSpec
from gemma_tools.bench_remote import (
    RemoteBenchConfig,
    build_ssh_argv,
    parse_jinja_response,
    run_remote_prompt,
)
from gemma_tools.health_table import load_health_table

REPO_ROOT = Path(__file__).resolve().parent.parent
HEALTH_PATH = REPO_ROOT / "data" / "health_table_v1.yaml"


# region: parse_jinja_response

@pytest.mark.parametrize(
    ("desc", "stdout", "expected"),
    [
        ("plain answer with eos terminator",
         "Penicillin. [end of text]\n", "Penicillin."),
        ("answer terminated by <end_of_turn>",
         "72.<end_of_turn>", "72."),
        ("answer with leading bare divider then perf",
         "\nmodel\n72.\ncommon_perf_print: load = 1\n", "72."),
        ("answer with explicit divider then perf prefix older",
         "<start_of_turn>model\nLisinopril, Aspirin.\nllama_perf_context_print: x\n",
         "Lisinopril, Aspirin."),
        ("trailing whitespace stripped",
         "  72.  \n   ", "72."),
        ("empty stdout -> empty",
         "", ""),
        ("only perf footer -> empty",
         "common_perf_print: load = 1\n", ""),
    ],
)
def test_parse_jinja_response_extracts_model_only(desc: str, stdout: str, expected: str) -> None:
    got = parse_jinja_response(stdout)
    assert got == expected, f"{desc}: got={got!r} expected={expected!r}"

# endregion

# region: build_ssh_argv

@pytest.mark.parametrize(
    ("desc", "override", "must_contain"),
    [
        ("default seed/threads survive into argv",
         {}, ["ssh", "host-alias", "BODY=$(cat)", "--jinja",
              "--no-display-prompt", "-no-cnv", "--single-turn",
              "-t 2", "-n 128", "--seed 42", "--top-k 1", "--temp 0.0",
              "/mnt/sdcard/llama-cpp/llama-completion"]),
        ("custom n_predict reflected",
         {"n_predict": 64}, ["-n 64"]),
        ("custom seed reflected",
         {"seed": 99}, ["--seed 99"]),
    ],
)
def test_build_ssh_argv_carries_config(
    desc: str, override: dict[str, object], must_contain: list[str],
) -> None:
    cfg = RemoteBenchConfig(
        ssh_host="host-alias",
        binary_path=Path("/mnt/sdcard/llama-cpp/llama-completion"),
        model_path=Path("/mnt/sdcard/models/test/m.gguf"),
    )
    cfg = replace(cfg, **override)  # type: ignore[arg-type]
    argv = build_ssh_argv(cfg)
    assert argv[0] == "ssh", f"{desc}: argv[0]={argv[0]!r}"
    assert argv[1] == "host-alias", f"{desc}: argv[1]={argv[1]!r}"
    remote = " ".join(argv)
    for needle in must_contain:
        assert needle in remote, (
            f"{desc}: missing needle {needle!r} in remote command {remote!r}"
        )

# endregion

# region: run_remote_prompt

def _fake_runner(
    *,
    stdout: str = "",
    stderr: str = "",
    rc: int = 0,
) -> Callable[
    [list[str], str, float | None],
    subprocess.CompletedProcess[str],
]:
    """Build a minimal fake `subprocess.run`-shaped callable.

    Used in place of the real ssh runner so unit tests run on hosts that
    have neither network access to the board nor the cross-compiled
    binary on PATH.
    """

    def _runner(argv: list[str], stdin: str, timeout: float | None) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=argv, returncode=rc, stdout=stdout, stderr=stderr,
        )

    return _runner


_PERF_OK = (
    "common_perf_print:    sampling time =     162.06 ms\n"
    "common_perf_print:        load time =    3268.01 ms\n"
    "common_perf_print: prompt eval time =   14684.38 ms /   920 tokens "
    "(   15.96 ms per token,    62.65 tokens per second)\n"
    "common_perf_print:        eval time =     398.72 ms /     5 runs   "
    "(   79.74 ms per token,    12.54 tokens per second)\n"
    "common_perf_print:       total time =   15109.19 ms /   925 tokens\n"
)


@pytest.mark.parametrize(
    ("desc", "spec", "fake_stdout", "want_pass", "want_text"),
    [
        (
            "P1 fact_lookup PASS — model emits 72",
            PromptSpec(id="P1", prompt_class="fact_lookup",
                       text="what is my heart rate?",
                       pass_pattern="72", pattern_flags=""),
            " 72.<end_of_turn>\n",
            True, "72.",
        ),
        (
            "P3 fact_lookup FAIL — model rambles",
            PromptSpec(id="P3", prompt_class="fact_lookup",
                       text="which medications do I take at 8am?",
                       pass_pattern="lisinopril|metformin", pattern_flags="i"),
            "Please provide a list. [end of text]\n",
            False, "Please provide a list.",
        ),
        (
            "S1 summarization PASS — both meds named",
            PromptSpec(id="S1", prompt_class="summarization",
                       text="summarize my current medications",
                       pass_pattern="lisinopril.*metformin", pattern_flags="is"),
            ":\n- Lisinopril 10 mg.\n- Metformin 500 mg.[end of text]\n",
            True, ":\n- Lisinopril 10 mg.\n- Metformin 500 mg.",
        ),
    ],
)
def test_run_remote_prompt_round_trips(
    desc: str, spec: PromptSpec, fake_stdout: str, want_pass: bool, want_text: str,
) -> None:
    """Inject a stub runner so we exercise compose → ssh → parse → score
    end-to-end without touching the network."""
    health = load_health_table(HEALTH_PATH)
    cfg = RemoteBenchConfig(
        ssh_host="stub",
        binary_path=Path("/mnt/sdcard/llama-cpp/llama-completion"),
        model_path=Path("/mnt/sdcard/m.gguf"),
    )
    runner = _fake_runner(stdout=fake_stdout, stderr=_PERF_OK)
    counter = iter(range(0, 1_000_000_000, 1_000_000))  # advance 1ms per call
    row = run_remote_prompt(
        spec, health, date(2026, 4, 28), cfg,
        runner=runner, clock_ns=lambda: next(counter),
    )
    assert row.error is None, f"{desc}: unexpected error={row.error!r}"
    assert row.response_text == want_text, (
        f"{desc}: response_text={row.response_text!r} expected={want_text!r}"
    )
    assert row.passed_regex is want_pass, (
        f"{desc}: passed_regex={row.passed_regex} expected={want_pass}"
    )
    assert row.timing.wall_ms_load == 3268.01, (
        f"{desc}: load_ms parsed wrong: {row.timing.wall_ms_load!r}"
    )
    assert row.timing.tokens_generated == 5, (
        f"{desc}: decode tokens parsed wrong: {row.timing.tokens_generated!r}"
    )


@pytest.mark.parametrize(
    ("desc", "rc", "stderr_tail", "want_error_substring"),
    [
        ("nonzero exit captured as error row",
         1, "abort", "exited 1"),
        ("missing perf footer captured as error row",
         0, "no perf here", "could not parse llama_perf footer"),
    ],
)
def test_run_remote_prompt_error_paths(
    desc: str, rc: int, stderr_tail: str, want_error_substring: str,
) -> None:
    health = load_health_table(HEALTH_PATH)
    spec = PromptSpec(id="C1", prompt_class="calibration",
                      text="say hi", pass_pattern=".", pattern_flags="")
    cfg = RemoteBenchConfig(
        ssh_host="stub",
        binary_path=Path("/x"), model_path=Path("/y"),
    )
    runner = _fake_runner(stdout="ok\n", stderr=stderr_tail, rc=rc)
    counter = iter(range(0, 1_000_000_000, 1_000_000))
    row = run_remote_prompt(
        spec, health, date(2026, 4, 28), cfg,
        runner=runner, clock_ns=lambda: next(counter),
    )
    assert row.error is not None, f"{desc}: expected error row"
    assert want_error_substring in row.error, (
        f"{desc}: error={row.error!r} missing substring {want_error_substring!r}"
    )
    assert row.passed_regex is False, f"{desc}: error rows must not score PASS"


# endregion
