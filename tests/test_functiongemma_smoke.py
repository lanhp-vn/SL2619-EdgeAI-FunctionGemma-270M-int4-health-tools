"""M2 — FunctionGemma Phase A smoke (host CPU): dry-run + parser unit tests.

CI deliberately exercises only the `--dry-run` path (prompt rendering + the
parser). Loading the Q4_K_M GGUF through `llama-cpp-python` is the M2
acceptance gate but is too slow for CI; it's the one-shot
`uv run python scripts/functiongemma_smoke.py` invocation in
`docs/plans/FunctionGemma/README.md` §8.3.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "functiongemma_smoke.py"
_FG_TOKENIZER = Path(os.path.expanduser("~/hf-cache/functiongemma-270m-it"))


@pytest.fixture(scope="module")
def smoke_module() -> ModuleType:
    """Load `scripts/functiongemma_smoke.py` as a module for direct unit tests.

    The script is intentionally not in the package; `importlib.util` keeps it
    callable here without requiring a `setup.py scripts=` entry.
    """
    spec = importlib.util.spec_from_file_location("functiongemma_smoke", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Parser unit tests — no transformers, no llama_cpp, no GGUF.
# --------------------------------------------------------------------------


def test_parse_canonical_colon_form(smoke_module: ModuleType) -> None:
    """Vendor canonical: `<start_function_call>call:NAME{...}<end_function_call>`."""
    text = (
        "<start_function_call>call:get_current_temperature"
        "{location:<escape>London<escape>}<end_function_call>"
    )
    calls = smoke_module.parse_function_calls(text)
    assert calls == [{"tool": "get_current_temperature", "args": {"location": "London"}}]


def test_parse_space_form_observed_under_q4_k_m(smoke_module: ModuleType) -> None:
    """§15.4 Path B observed: the Q4_K_M GGUF emits `call NAME` (space, no colon).

    If we don't accept this, the M2 smoke fails on the very GGUF M1.5 produced
    — see the §15.6 Path B output capture from 2026-04-30.
    """
    text = (
        "<start_function_call>call get_current_temperature"
        "{location:<escape>London<escape>}<end_function_call>"
    )
    calls = smoke_module.parse_function_calls(text)
    assert calls == [{"tool": "get_current_temperature", "args": {"location": "London"}}]


def test_parse_escape_preserves_special_chars(smoke_module: ModuleType) -> None:
    """`<escape>` exists precisely so commas/braces inside strings don't
    terminate the block (§6.2). Verify a city with a comma round-trips."""
    text = (
        "<start_function_call>call:get_current_temperature"
        "{location:<escape>San Francisco, CA<escape>}<end_function_call>"
    )
    calls = smoke_module.parse_function_calls(text)
    assert calls == [
        {"tool": "get_current_temperature", "args": {"location": "San Francisco, CA"}},
    ]


def test_parse_multiple_calls(smoke_module: ModuleType) -> None:
    """Parallel calls (§6.6) — the parser must surface all of them so the
    M2 validator can fail loudly when there is more than one (we want
    exactly one for the single-turn smoke)."""
    text = (
        "<start_function_call>call:get_current_temperature"
        "{location:<escape>London<escape>}<end_function_call>"
        "<start_function_call>call:get_current_temperature"
        "{location:<escape>Paris<escape>}<end_function_call>"
    )
    calls = smoke_module.parse_function_calls(text)
    assert len(calls) == 2
    assert calls[0]["args"]["location"] == "London"
    assert calls[1]["args"]["location"] == "Paris"


def test_parse_no_call_returns_empty(smoke_module: ModuleType) -> None:
    assert smoke_module.parse_function_calls("the temperature is 42 degrees") == []


def test_validate_one_call_rejects_zero(smoke_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="zero"):
        smoke_module._validate_one_call([], expected_tool="get_current_temperature")


def test_validate_one_call_rejects_multiple(smoke_module: ModuleType) -> None:
    calls = [
        {"tool": "get_current_temperature", "args": {"location": "London"}},
        {"tool": "get_current_temperature", "args": {"location": "Paris"}},
    ]
    with pytest.raises(ValueError, match="expected exactly 1"):
        smoke_module._validate_one_call(calls, expected_tool="get_current_temperature")


def test_validate_one_call_rejects_wrong_tool(smoke_module: ModuleType) -> None:
    calls = [{"tool": "send_email", "args": {"location": "London"}}]
    with pytest.raises(ValueError, match=r"expected tool 'get_current_temperature'"):
        smoke_module._validate_one_call(calls, expected_tool="get_current_temperature")


def test_validate_one_call_rejects_missing_required_arg(smoke_module: ModuleType) -> None:
    calls = [{"tool": "get_current_temperature", "args": {}}]
    with pytest.raises(ValueError, match="missing required arg 'location'"):
        smoke_module._validate_one_call(calls, expected_tool="get_current_temperature")


# --------------------------------------------------------------------------
# Dry-run subprocess — proves the CLI plumbing + the prompt-render path
# without touching llama_cpp or the GGUF.
# --------------------------------------------------------------------------


def _need_tokenizer() -> None:
    if not _FG_TOKENIZER.exists():
        pytest.skip(
            f"FG tokenizer not at {_FG_TOKENIZER}; run M1.5 §15.3 first "
            "(`hf download google/functiongemma-270m-it ...`)."
        )
    pytest.importorskip(
        "transformers",
        reason="functiongemma extra not installed; "
               "run `uv sync --extra functiongemma`",
    )


def _run_smoke(*extra_args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra_args],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )


def test_dry_run_exits_zero_and_prints_pass(tmp_path: Path) -> None:
    """Dry-run must complete cleanly without loading llama_cpp.

    We poison `llama_cpp` on PYTHONPATH with a module that raises on import.
    If dry-run actually attempted the import, the script would crash; an
    exit-0 here is positive evidence the lazy-import gate works.
    """
    _need_tokenizer()
    shadow = tmp_path / "shadow"
    (shadow / "llama_cpp").mkdir(parents=True)
    (shadow / "llama_cpp" / "__init__.py").write_text(
        "raise ImportError('dry-run must NOT import llama_cpp')\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{shadow}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = _run_smoke(
        "--dry-run",
        "--query", "What is the temperature in London?",
        env=env,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "PASS-DRY-RUN" in proc.stdout
    # Negative-evidence check: the [smoke] loading line only fires past the
    # dry-run early-return, so it must not appear here.
    assert "[smoke] loading GGUF" not in proc.stderr


def test_dry_run_does_not_require_gguf(tmp_path: Path) -> None:
    """Acceptance: dry-run must succeed even if `--model` points at nothing.

    Confirms the GGUF-existence check is gated behind the non-dry-run path.
    """
    _need_tokenizer()
    proc = _run_smoke(
        "--dry-run",
        "--model", str(tmp_path / "definitely-does-not-exist.gguf"),
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_dry_run_verbose_emits_complete_prompt_with_required_fragments() -> None:
    """The rendered prompt MUST include every M2-acceptance fragment.

    Failure here is the load-bearing signal that the chat template (or the
    developer-trigger string) drifted away from §6.4. The four fragments
    map 1:1 to wire-format invariants that downstream parsing relies on.
    """
    _need_tokenizer()
    query = "What is the temperature in Reykjavík?"
    proc = _run_smoke("--dry-run", "--verbose", "--query", query)
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # Verbose dumps the rendered prompt to stderr between the markers.
    assert "--- rendered prompt ---" in proc.stderr
    rendered = proc.stderr.split("--- rendered prompt ---", 1)[1].split(
        "--- end prompt ---", 1
    )[0]
    # 1. Developer trigger string (vendor-mandated, prompt-based activator).
    assert (
        "You are a model that can do function calling with the following functions"
        in rendered
    )
    # 2. Tool name appears (chat template lowered the JSON-Schema into the
    #    bespoke wire format).
    assert "get_current_temperature" in rendered
    # 3. Wire-format declaration token bracket.
    assert "<start_function_declaration>" in rendered
    # 4. User query passes through verbatim.
    assert query in rendered
    # 5. Generation prompt marker — the model picks up here.
    assert "<start_of_turn>model" in rendered


def test_missing_tokenizer_dir_returns_two(tmp_path: Path) -> None:
    """A missing tokenizer dir is a hard, exit-2 misconfig (not exit 1)."""
    proc = _run_smoke(
        "--dry-run",
        "--tokenizer", str(tmp_path / "nope"),
    )
    assert proc.returncode == 2, proc.stderr
    assert "tokenizer dir not found" in proc.stderr
    # Remediation must point at the M1.5 prereq command.
    assert "M1.5" in proc.stderr


def test_default_ctx_size_avoids_n_ctx_train_warning(smoke_module: ModuleType) -> None:
    """Regression for the M2 follow-up.

    `llama-cpp` prints `n_ctx_seq (N) < n_ctx_train (32768)` whenever the
    requested context is below FunctionGemma's trained 32k. Our default must
    be ≥ 4096 — the Unsloth Phase D recipe's `max_seq_length` — so the warning
    only appears if a caller actively dials `--ctx-size` lower."""
    assert smoke_module.DEFAULT_CTX_SIZE >= 4096


def test_ctx_size_cli_option_is_recognized() -> None:
    """The `--ctx-size` flag must be wired through argparse so callers can
    override `Llama(n_ctx=...)` for multi-turn tests. An unknown flag would
    exit 2 with an `unrecognized arguments` message; absence of that message
    is positive evidence the flag is parsed and reaches the dry-run early
    return without triggering a usage error."""
    _need_tokenizer()
    proc = _run_smoke(
        "--dry-run", "--ctx-size", "8192",
        "--query", "What is the temperature in London?",
    )
    assert "unrecognized arguments" not in proc.stderr
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
