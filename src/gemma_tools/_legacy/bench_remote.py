"""Host-driven on-board bench harness.

Runs the closed-world health-YAML QA suite against an aarch64 `llama-completion`
binary on a remote SSH target without writing anything to the remote
filesystem (R3-compliant: SSH strictly read-only from the agent). Each prompt
spawns one local `ssh <host> 'llama-completion --jinja --no-display-prompt
-p "$BODY" -no-cnv --single-turn ...'` with the user-turn body piped via
stdin, then captures stdout (model reply only — `--no-display-prompt`
suppresses the prompt echo) plus stderr (perf footer) into a host JSONL row.

Why this exists separately from `gemma_tools.bench_prompt`:
- That module's `LlamaCompletionBenchAdapter` text-wraps the body with literal
  `<start_of_turn>user\\n…<end_of_turn>\\n<start_of_turn>model\\n` markers and
  passes via `-f`. llama.cpp without `--jinja` tokenizes those markers as
  plain bytes (~5-10 sub-tokens each) instead of the special control tokens
  (105 / 106) the FT'd model was trained on. Bench prompts at H6 surfaced
  the same wrap (the base model's YAML-echo failure mode swamped the issue);
  Q3 on the FT'd Q4_0 made it visible: model outputs hallucinated tail
  content because the chat-template boundary it learned to enter answer mode
  at is missing from the wire-level prompt.
- `--jinja` makes llama.cpp apply the model's chat_template metadata
  internally — the special tokens land at the right ids and the FT'd model
  enters answer mode reliably (Q3e: P1 emits "72" first, S1 emits a clean
  comma-separated medication list).
- `--no-display-prompt` is the divider-free way to isolate the model's reply
  out of stdout — no `<start_of_turn>model\\n` literal needed in the parser
  (which is exactly the H6 detokenized-divider gotcha that bit
  `bench_prompt.parse_completion_response`).

Argument naming follows the existing harness so the JSONL shape is
identical and `bench_eval.py` can score either source without conditionals.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from gemma_tools._legacy.bench_prompt import (
    BenchRow,
    LlamaCompletionError,
    PromptSpec,
    TimingRecord,
    load_prompt_suite,
    parse_llama_perf,
    score_response,
)
from gemma_tools.health_table import HealthTable, load_health_table
from gemma_tools._legacy.prompt_composer import compose_user_text

__all__ = [
    "RemoteBenchConfig",
    "build_ssh_argv",
    "main",
    "parse_jinja_response",
    "run_remote_prompt",
]

# region: parsers


_LLAMA_END_OF_TEXT = "[end of text]"
_GEMMA_TURN_CLOSE = "<end_of_turn>"
# Chat-template tokens detokenize to empty by default. Their bare role-label
# fallbacks appear when llama.cpp echoes the prompt without `--no-display-prompt`.
# We pass `--no-display-prompt`, so stdout is just the model's reply — but a
# defensive trim keeps the parser safe if a future caller drops the flag.
_GEMMA_USER_TURN_OPEN = "<start_of_turn>user\n"
_GEMMA_MODEL_TURN_OPEN = "<start_of_turn>model\n"
_USER_ROLE_DETOK = "\nuser\n"
_MODEL_ROLE_DETOK = "\nmodel\n"

# Lines that llama.cpp logs to stdout below the model output when stderr and
# stdout are merged (no `2>&1` separation upstream). The perf footer is one
# such block; we drop everything from its first line onward when isolating the
# response.
_PERF_LINE_PREFIXES = ("common_perf_print:", "llama_perf_context_print:")


def parse_jinja_response(stdout: str) -> str:
    """Slice the model's reply out of `--jinja --no-display-prompt` stdout.

    With `--no-display-prompt`, the prompt body is suppressed; the model
    output is the leading content. The perf block + memory breakdown follow
    after a blank line. Trim:

      - any leading divider that slipped through (`<start_of_turn>model\\n`
        or bare `\\nmodel\\n`)
      - terminators (`[end of text]`, `<end_of_turn>`)
      - everything from the perf block onward
    """
    # Strip leading dividers if they appear (defensive — `--no-display-prompt`
    # should suppress them, but some llama.cpp forks still emit a single bare
    # `\nmodel\n` divider before the model output).
    body = stdout
    for divider in (_GEMMA_MODEL_TURN_OPEN, _MODEL_ROLE_DETOK):
        idx = body.rfind(divider)
        if idx >= 0:
            body = body[idx + len(divider):]
            break

    # Cut at any terminator OR at the perf block.
    cut_indices: list[int] = []
    for terminator in (_LLAMA_END_OF_TEXT, _GEMMA_TURN_CLOSE,
                       _GEMMA_USER_TURN_OPEN, _USER_ROLE_DETOK):
        idx = body.find(terminator)
        if idx >= 0:
            cut_indices.append(idx)
    for prefix in _PERF_LINE_PREFIXES:
        idx = body.find(prefix)
        if idx >= 0:
            cut_indices.append(idx)
    if cut_indices:
        body = body[:min(cut_indices)]

    return body.strip()


# endregion

# region: SSH driver


@dataclass(frozen=True, slots=True)
class RemoteBenchConfig:
    """All the knobs for one remote bench run.

    `ssh_host` is an alias from `~/.ssh/config` (e.g. `nouslogic-sl2619`).
    Per `.claude/CLAUDE.local.md §2`, the agent loads the key into an
    ephemeral ssh-agent before the sweep — this script does NOT manage
    that lifecycle (callers do, so the same agent serves all 15 prompts).
    """

    ssh_host: str
    binary_path: Path
    model_path: Path
    n_threads: int = 2
    n_predict: int = 128
    temp: float = 0.0
    top_k: int = 1
    seed: int = 42
    subprocess_timeout_s: float = 120.0


def build_ssh_argv(cfg: RemoteBenchConfig) -> list[str]:
    """Argv for `ssh <host> 'llama-completion --jinja --no-display-prompt
    -p "$BODY" -no-cnv --single-turn -t N -n M --temp T --top-k K --seed S
    -m MODEL'`.

    The remote shell uses `BODY=$(cat)` to absorb stdin into a shell var,
    then expands it as the `-p` arg. This keeps the ssh argv constant
    across prompts (no shell-quoting nightmare for embedded YAML) and
    matches the Q3e/d pattern that produced clean output.
    """
    remote_cmd = (
        f'BODY=$(cat); {cfg.binary_path} '
        f'-m {cfg.model_path} '
        f'--jinja --no-display-prompt '
        f'-p "$BODY" '
        f'-t {cfg.n_threads} -n {cfg.n_predict} '
        f'--temp {cfg.temp} --top-k {cfg.top_k} --seed {cfg.seed} '
        f'-no-cnv --single-turn'
    )
    return ["ssh", cfg.ssh_host, remote_cmd]


SubprocessRunner = Callable[
    [list[str], str, float | None],
    "subprocess.CompletedProcess[str]",
]


def _default_subprocess_runner(
    argv: list[str], stdin_text: str, timeout_s: float | None,
) -> subprocess.CompletedProcess[str]:
    """Real `subprocess.run` wrapper, text mode, captured streams.

    Isolated so unit tests inject a fake — they don't need network
    connectivity to a live SL2619 board.
    """
    return subprocess.run(
        argv,
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )


def run_remote_prompt(
    spec: PromptSpec,
    health: HealthTable,
    now: date,
    cfg: RemoteBenchConfig,
    runner: SubprocessRunner = _default_subprocess_runner,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> BenchRow:
    """Compose, ssh, parse, build a JSONL row.

    Errors (non-zero exit, perf-footer-missing, timeout) are recorded as an
    `error` field on the row; the sweep continues. A killed remote process
    (OOM) returns -9 / -11 — captured as an error row with stderr tail.
    """
    user_text = compose_user_text(health, now, spec.text)
    argv = build_ssh_argv(cfg)
    run_started = datetime.now().isoformat(timespec="seconds")

    error: str | None
    response_text = ""
    perf_load_ms = 0.0
    perf_prompt_eval_ms = 0.0
    perf_decode_tokens = 0
    perf_total_ms = 0.0
    start_ns = clock_ns()
    try:
        proc = runner(argv, user_text, cfg.subprocess_timeout_s)
        if proc.returncode != 0:
            raise LlamaCompletionError(
                f"ssh+llama-completion exited {proc.returncode} "
                f"(stderr tail: {proc.stderr[-400:]!r})"
            )
        response_text = parse_jinja_response(proc.stdout)
        try:
            perf = parse_llama_perf(proc.stderr + "\n" + proc.stdout)
        except ValueError as e:
            raise LlamaCompletionError(
                f"could not parse llama_perf footer: {e} "
                f"(stderr tail: {proc.stderr[-400:]!r})"
            ) from e
        perf_load_ms = perf.wall_ms_load
        perf_prompt_eval_ms = perf.wall_ms_prompt_eval
        perf_decode_tokens = perf.n_decode_tokens
        perf_total_ms = perf.wall_ms_total
        error = None
    except subprocess.TimeoutExpired as e:
        error = (
            f"TimeoutExpired after {cfg.subprocess_timeout_s}s: {e}"
        )
    except LlamaCompletionError as e:
        error = f"LlamaCompletionError: {e}"
    end_ns = clock_ns()
    external_total_ms = (end_ns - start_ns) / 1e6

    passed = (
        score_response(spec.pass_pattern, spec.pattern_flags, response_text)
        if error is None and response_text
        else False
    )

    return BenchRow(
        prompt_id=spec.id,
        prompt_class=spec.prompt_class,
        prompt_text=spec.text,
        response_text=response_text,
        run_started_iso=run_started,
        timing=TimingRecord(
            wall_ms_load=perf_load_ms,
            wall_ms_ttft_vendor=perf_load_ms + perf_prompt_eval_ms,
            wall_ms_ttft_external=external_total_ms,
            wall_ms_total=perf_total_ms if perf_total_ms > 0 else external_total_ms,
            tokens_generated=perf_decode_tokens,
        ),
        # Host-driven; /proc memory probes are not meaningful here. Sentinel 0.
        peak_rss_mb=0.0,
        cma_free_kb_before=0,
        cma_free_kb_during=0,
        cma_free_kb_after=0,
        pass_pattern=spec.pass_pattern,
        pattern_flags=spec.pattern_flags,
        passed_regex=passed,
        error=error,
    )


# endregion

# region: CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench-remote",
        description=(
            "Host-driven SL2619 bench: SSH-pipes prompts to "
            "/mnt/sdcard/llama-cpp/llama-completion in --jinja "
            "--no-display-prompt mode. Writes JSONL locally only; never "
            "mutates the remote (R3 SSH read-only)."
        ),
    )
    p.add_argument("--ssh-host", required=True, help="SSH alias (e.g. nouslogic-sl2619).")
    p.add_argument("--prompts", type=Path, required=True, help="Path to prompts.yaml.")
    p.add_argument("--health-table", type=Path, required=True,
                   help="Path to health_table_v1.yaml.")
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL output path. Appends; parent dirs created.")
    p.add_argument("--llama-binary", type=Path, required=True,
                   help="Remote path to llama-completion binary on the board.")
    p.add_argument("--llama-model", type=Path, required=True,
                   help="Remote path to .gguf model on the board.")
    p.add_argument("--max-gen-tokens", type=int, default=128)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subprocess-timeout-s", type=float, default=180.0,
                   help="Wider default than bench_prompt (network round-trip + per-call mmap).")
    p.add_argument("--now", type=date.fromisoformat, default=None,
                   help="Override the date in the directive prompt; defaults to today.")
    p.add_argument("--ids", default="",
                   help="Comma-separated prompt ids; empty = all.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    health = load_health_table(args.health_table)
    suite = load_prompt_suite(args.prompts)
    if args.ids:
        keep = {s.strip() for s in args.ids.split(",") if s.strip()}
        suite = [p for p in suite if p.id in keep]
        if not suite:
            print(f"no prompts matched --ids {args.ids!r}", file=sys.stderr)
            return 2
    run_date = args.now if args.now is not None else date.today()
    cfg = RemoteBenchConfig(
        ssh_host=args.ssh_host,
        binary_path=args.llama_binary,
        model_path=args.llama_model,
        n_threads=args.n_threads,
        n_predict=args.max_gen_tokens,
        temp=args.temp,
        top_k=args.top_k,
        seed=args.seed,
        subprocess_timeout_s=args.subprocess_timeout_s,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"bench-remote ssh={args.ssh_host} model={args.llama_model.name} "
        f"prompts={len(suite)}",
        file=sys.stderr, flush=True,
    )
    for spec in suite:
        row = run_remote_prompt(spec, health, run_date, cfg)
        with args.output.open("a", encoding="utf-8") as f:
            obj = asdict(row)
            obj["timing"]["tokens_per_sec"] = row.timing.tokens_per_sec
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
        verdict = (
            "ERR" if row.error
            else ("PASS" if row.passed_regex else "FAIL")
        )
        snippet = re.sub(r"\s+", " ", row.response_text)[:80]
        print(
            f"  {spec.id:<4} {spec.prompt_class:<16} "
            f"tok={row.timing.tokens_generated:>3} "
            f"wall={row.timing.wall_ms_total:>6.0f} ms "
            f"{verdict} {snippet!r} "
            f"{'error=' + row.error if row.error else ''}",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# endregion
