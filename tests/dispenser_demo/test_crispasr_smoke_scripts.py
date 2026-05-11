"""Unit tests for the Phase 0 CrispASR smoke artifacts.

Scope is intentionally narrow: parse-argument validation, transcript scraping,
and basic shell-syntax checks. No model, no binary, no network. The Phase 0
gate against a real runtime is human-executed and recorded in
docs/plans/dispenser-demo/crispasr-spike-notes.md.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_SCRIPT = REPO_ROOT / "scripts" / "dispenser_demo" / "spike" / "crispasr_host_smoke.py"
BOARD_SCRIPT = REPO_ROOT / "scripts" / "dispenser_demo" / "spike" / "crispasr_board_smoke.sh"


def _load_host_module() -> ModuleType:
    """Load the bare host smoke script as a module so we can unit-test its helpers.

    The script lives outside the importable package tree (scripts/ is not a
    package), so we use spec_from_file_location rather than adding sys.path
    hacks that would leak between tests.
    """
    spec = importlib.util.spec_from_file_location("crispasr_host_smoke", HOST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {HOST_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def host_mod() -> ModuleType:
    return _load_host_module()


def test_host_script_exists() -> None:
    assert HOST_SCRIPT.is_file(), "host smoke script missing"
    assert HOST_SCRIPT.stat().st_mode & 0o111, "host script not executable"


def test_board_script_exists() -> None:
    assert BOARD_SCRIPT.is_file(), "board smoke script missing"
    assert BOARD_SCRIPT.stat().st_mode & 0o111, "board script not executable"


# | argv                                          | want_field    | want_value                    | description                              |
@pytest.mark.parametrize(
    ("argv", "want_field", "want_value", "desc"),
    [
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c"],
            "backend",
            "moonshine-streaming",
            "default backend matches upstream documented value",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c", "--backend", "whisper"],
            "backend",
            "whisper",
            "backend override accepted",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c"],
            "latency_budget_s",
            1.0,
            "default latency budget matches plan §9 host gate (<=1s)",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c", "--timeout", "45"],
            "timeout",
            45.0,
            "timeout override propagated as float",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c"],
            "threads",
            None,
            "default threads is None (let CrispASR pick min(4, hwconc))",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c", "--threads", "2"],
            "threads",
            2,
            "threads override propagated as int (e.g. 2 for SL2619 A55)",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c"],
            "language",
            "en",
            "default language 'en' skips auto-LID (no ggml-tiny.bin download, no extra RSS)",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c", "--language", "auto"],
            "language",
            "auto",
            "language override 'auto' re-enables CrispASR's auto-LID",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c"],
            "punctuation",
            False,
            "default punctuation off (--no-punctuation propagated, suppresses fireredpunc auto-download)",
        ),
        (
            ["--bin", "/a", "--model", "/b", "--wav", "/c", "--punctuation"],
            "punctuation",
            True,
            "--punctuation re-enables upstream auto-fetch behaviour",
        ),
    ],
)
def test_parse_args_defaults_and_overrides(
    host_mod: ModuleType,
    argv: list[str],
    want_field: str,
    want_value: object,
    desc: str,
) -> None:
    ns = host_mod.parse_args(argv)
    assert getattr(ns, want_field) == want_value, desc


def test_parse_args_requires_bin_model_wav(host_mod: ModuleType) -> None:
    # argparse exits with code 2 on missing required args
    with pytest.raises(SystemExit) as exc:
        host_mod.parse_args([])
    assert exc.value.code == 2, "missing required args should exit 2"


# | stdout                                                   | want_substr | description                       |
@pytest.mark.parametrize(
    ("stdout", "want_substr", "desc"),
    [
        ("Hello world\n", "Hello world", "trivial single-line transcript"),
        ("[2026-01-01 00:00:00] starting\nHello world\n", "Hello world", "strips bracketed log prefix"),
        ("whisper_init: loaded\nggml_alloc: ok\nGood morning everyone\n", "Good morning", "strips library log prefixes"),
        ("", "", "empty stdout yields empty transcript (no crash)"),
        ("[log]\n[log]\n", "", "all-log stdout yields empty transcript"),
    ],
)
def test_parse_transcript(host_mod: ModuleType, stdout: str, want_substr: str, desc: str) -> None:
    got = host_mod.parse_transcript(stdout)
    assert want_substr in got, desc


def test_help_text_runs() -> None:
    """`--help` must not crash; this catches import-time errors in the script."""
    r = subprocess.run(
        [sys.executable, str(HOST_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 0, f"--help exited {r.returncode}; stderr={r.stderr}"
    assert "crispasr" in r.stdout.lower(), "help text should mention crispasr"


def test_host_script_fails_clearly_on_missing_binary(tmp_path: Path) -> None:
    """The host script must give a CrispASR-specific error, not a generic FileNotFound."""
    fake_wav = tmp_path / "x.wav"
    fake_wav.write_bytes(b"")
    fake_model = tmp_path / "x.gguf"
    fake_model.write_bytes(b"")
    r = subprocess.run(
        [
            sys.executable,
            str(HOST_SCRIPT),
            "--bin",
            "/no/such/crispasr",
            "--model",
            str(fake_model),
            "--wav",
            str(fake_wav),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode != 0, "missing binary must exit non-zero"
    combined = (r.stdout + r.stderr).lower()
    assert "crispasr" in combined, "error message should name the runtime"
    assert "spike" in combined, "error message should point at the spike bootstrap notes"


def test_board_script_bash_syntax() -> None:
    """`bash -n` parses the script without executing it — catches quoting bugs."""
    r = subprocess.run(
        ["bash", "-n", str(BOARD_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"


def test_board_script_help_runs() -> None:
    r = subprocess.run(
        ["bash", str(BOARD_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 0, f"--help exited {r.returncode}; stderr={r.stderr}"
    for needle in ("--bin", "--model", "--wav", "--threads", "--language", "MemAvailable", "/board_probe"):
        assert needle in r.stdout, f"help text missing {needle!r}"


def test_board_script_refuses_without_probe_snapshot(tmp_path: Path) -> None:
    """Without docs/tmp/sl2619-status.md the dispatcher exits 3 — sanity gate against ghost runs."""
    # We can't realistically delete the actual snapshot, so we copy the script
    # to a tmp dir whose parent has no docs/tmp/ — same exit-3 path.
    tmp_repo = tmp_path / "fakerepo"
    (tmp_repo / "scripts" / "dispenser_demo" / "spike").mkdir(parents=True)
    shutil.copy2(BOARD_SCRIPT, tmp_repo / "scripts" / "dispenser_demo" / "spike" / BOARD_SCRIPT.name)
    r = subprocess.run(
        [
            "bash",
            str(tmp_repo / "scripts" / "dispenser_demo" / "spike" / BOARD_SCRIPT.name),
            "--bin",
            "/x",
            "--model",
            "/y",
            "--wav",
            "/z",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 3, f"expected exit 3 (no probe snapshot), got {r.returncode}; stderr={r.stderr}"
    assert "board_probe" in r.stderr.lower(), "error should mention /board_probe"
