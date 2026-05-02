#!/usr/bin/env python3
"""FunctionGemma 270M GGUF throughput bench.

Two execution modes share the same prompt-rendering path so host and board
numbers are directly comparable:

  --mode local   : load `./model.gguf` via llama-cpp-python on host CPU.
                   Default; verifies the bench logic and gives a host
                   baseline tok/s.
  --mode remote  : render the FunctionGemma prompt locally with the HF
                   tokenizer + tools=..., then SSH-pipe the rendered text
                   to the board's `llama-completion -p "$BODY"`. We do NOT
                   pass `--jinja` to the board because the SL2619 build
                   (b8925 / 0adede8) does not propagate `tools=` to the
                   chat template (FunctionGemma's function-calling block
                   would render with an empty tool registry); see
                   `docs/plans/FunctionGemma/README.md` §15.6.

Records per prompt (JSONL):
  prompt_id, prompt_text, mode, response_text, parsed_call,
  n_prompt_tokens, n_decode_tokens, wall_ms_{load,prompt_eval,decode,total},
  prompt_tps, decode_tps, overall_tps.

`-r "<end_function_call>"` is passed in remote mode to stop decoding the
moment the model closes the first function call — the FunctionGemma model
under greedy decoding occasionally repeats the same call until n_predict
otherwise.

Usage
-----
    # Host baseline (1 prompt, fast smoke):
    uv run python scripts/functiongemma_bench.py --mode local --limit 1

    # Full host bench (7 prompts):
    uv run python scripts/functiongemma_bench.py --mode local

    # On-board bench via SSH (model.gguf must already be at the board path):
    uv run python scripts/functiongemma_bench.py --mode remote \\
        --ssh-host nouslogic-sl2619 \\
        --remote-binary /mnt/sdcard/llama-cpp/llama-completion \\
        --remote-model  /mnt/sdcard/models/functiongemma-270m/model.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse the chat REPL's lock-step copy of SYSTEM_PROMPT + TOOLS + parse so the
# bench renders the SAME tokens the deployed model is contracted on. This
# makes the bench's tok/s directly comparable to whatever functiongemma_chat.py
# observes interactively.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from functiongemma_chat import SYSTEM_PROMPT, TOOLS  # noqa: E402
from functiongemma_smoke import parse_function_calls  # noqa: E402

# Reuse the bench_prompt perf-footer parser. It already accepts both
# `llama_perf_context_print:` and the newer `common_perf_print:` (b8925+),
# matching the on-board build (`0adede8`).
sys.path.insert(0, str(REPO_ROOT / "src"))
from gemma_tools._legacy.bench_prompt import parse_llama_perf  # noqa: E402

DEFAULT_MODEL_PATH = REPO_ROOT / "model.gguf"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "model"
DEFAULT_BENCH_DIR = REPO_ROOT / "bench_results"
DEFAULT_CTX_SIZE = 4096
DEFAULT_MAX_NEW_TOKENS = 64
# Wider than n_predict because cold mmap of the GGUF over the SD card can
# take 3-4 s, then 600+ tokens of prompt eval at ~30 tok/s on the A55.
DEFAULT_REMOTE_TIMEOUT_S = 180.0

# One prompt per tool. Surface forms chosen to match the §5 verdict in the
# distil training analysis (the AM-clock-time row deliberately included so
# board behavior on the known weak case is measured, not hidden).
DEFAULT_PROMPTS: list[tuple[str, str, str]] = [
    ("vitals_bp", "get_vitals", "What's my blood pressure?"),
    ("appointment", "get_next_appointment", "When is my next appointment?"),
    ("vitals_fever", "get_vitals", "Do I have a fever?"),
    ("food_alcohol", "check_food_interaction", "Can I drink alcohol?"),
    ("allergies", "list_allergies", "What am I allergic to?"),
    ("meds_at_8am", "get_medications_at_time", "What pills do I take at 8 AM?"),
    ("med_atorvastatin", "get_medication_by_name", "Why am I taking Atorvastatin?"),
]


@dataclass
class BenchResult:
    prompt_id: str
    expected_tool: str
    prompt_text: str
    mode: str
    response_text: str = ""
    parsed_call: dict[str, Any] | None = None
    parsed_call_count: int = 0
    n_prompt_tokens: int = 0
    n_decode_tokens: int = 0
    wall_ms_load: float = 0.0
    wall_ms_prompt_eval: float = 0.0
    wall_ms_decode: float = 0.0
    wall_ms_total: float = 0.0
    prompt_tps: float = 0.0
    decode_tps: float = 0.0
    overall_tps: float = 0.0
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(p)))


def _render_prompt(tokenizer: Any, user_text: str) -> str:
    """Apply the FunctionGemma chat template with tools= so the on-wire prompt
    contains the `<start_function_declaration>` block. Strip the leading
    `<bos>` because llama-cpp-python and llama-completion both prepend one
    by default (`add_bos=True`).
    """
    messages = [*SYSTEM_PROMPT, {"role": "user", "content": user_text}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, expected str"
        )
    return rendered.removeprefix("<bos>")


# -----------------------------------------------------------------------------
# Local mode (llama-cpp-python)
# -----------------------------------------------------------------------------


def _run_local(
    llm: Any,
    tokenizer: Any,
    prompt_id: str,
    expected_tool: str,
    user_text: str,
    max_new_tokens: int,
    temperature: float,
) -> BenchResult:
    """One prompt against an already-loaded llama_cpp.Llama instance.

    Streams token-by-token so we can split prompt-eval wall-clock from
    decode wall-clock the same way the on-board `llama_perf_context_print`
    footer does — without that split, "decode tok/s" on host would be
    `n_out / total_wall`, which under-reports by ~100x because the 1600+
    token system prompt dominates total wall.
    """
    res = BenchResult(
        prompt_id=prompt_id,
        expected_tool=expected_tool,
        prompt_text=user_text,
        mode="local",
    )
    try:
        gen_prompt = _render_prompt(tokenizer, user_text)
        t0 = time.perf_counter()
        first_token_t: float | None = None
        chunks: list[str] = []
        n_out = 0
        stream = llm(
            gen_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            echo=False,
            stop=["<end_function_call>"],
            stream=True,
        )
        for chunk in stream:
            text_piece = chunk["choices"][0]["text"]
            if text_piece == "":
                continue
            if first_token_t is None:
                first_token_t = time.perf_counter()
            chunks.append(text_piece)
            n_out += 1
        elapsed_s = time.perf_counter() - t0
        text = "".join(chunks)
        # `stream=True` doesn't apply `stop=` post-processing the same way
        # one-shot does; `<end_function_call>` may already be in `text`.
        if text and "<start_function_call>" in text and "<end_function_call>" not in text:
            text = text + "<end_function_call>"

        # llama-cpp-python doesn't expose prompt token count on stream chunks;
        # tokenize ourselves to get an honest n_prompt for the throughput math.
        prompt_ids = llm.tokenize(gen_prompt.encode("utf-8"), add_bos=False)
        n_in = len(prompt_ids)
        ttft_s = (first_token_t - t0) if first_token_t else elapsed_s
        decode_s = max(0.0, elapsed_s - ttft_s)

        res.response_text = text
        res.n_prompt_tokens = n_in
        res.n_decode_tokens = n_out
        res.wall_ms_load = 0.0  # Llama is loaded once per process; per-prompt is 0.
        res.wall_ms_prompt_eval = ttft_s * 1000.0
        res.wall_ms_decode = decode_s * 1000.0
        res.wall_ms_total = elapsed_s * 1000.0
        if ttft_s > 0:
            res.prompt_tps = n_in / ttft_s
        if decode_s > 0 and n_out > 0:
            res.decode_tps = n_out / decode_s
        if elapsed_s > 0:
            res.overall_tps = (n_in + n_out) / elapsed_s

        calls = parse_function_calls(text)
        res.parsed_call_count = len(calls)
        if calls:
            res.parsed_call = {"tool": calls[0]["tool"], "args": calls[0]["args"]}
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


# -----------------------------------------------------------------------------
# Remote mode (ssh-piped llama-completion on the board)
# -----------------------------------------------------------------------------


def _build_remote_argv(
    ssh_host: str,
    binary_path: Path,
    model_path: Path,
    n_threads: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> list[str]:
    """SSH argv: `ssh HOST 'BODY=$(cat); BINARY -m MODEL -p "$BODY" ...'`.

    `BODY=$(cat)` absorbs stdin into a remote shell var so we don't have to
    ferry hundreds of bytes of rendered chat-template text through ssh argv
    (single-quote escaping breaks on embedded newlines + `<escape>`).

    NOT passing `--jinja` is intentional: the SL2619 build (`0adede8`) does
    not forward `tools=` to the chat template, so the FunctionGemma function
    declarations would render empty. We pre-rendered them on host already.

    `-r "<end_function_call>"` (`--reverse-prompt`) is the early-stop guard
    against the user-reported repeated-call drift on greedy decoding.
    """
    remote_cmd = (
        f"BODY=$(cat); {binary_path} "
        f"-m {model_path} "
        f'-p "$BODY" '
        f"-t {n_threads} -n {max_new_tokens} "
        f"--temp {temperature} --top-k {top_k} --seed {seed} "
        f'-r "<end_function_call>" '
        f"-no-cnv --single-turn"
    )
    return ["ssh", ssh_host, remote_cmd]


def _slice_remote_response(stdout: str) -> str:
    """Drop the perf footer + any trailing terminators from the model output.

    The board build prints `common_perf_print:` lines after the model
    output; we cut everything from the first such line onward, then trim
    `[end of text]` / `<end_of_turn>` / our `-r` reverse prompt.
    """
    body = stdout
    cuts: list[int] = []
    for marker in ("common_perf_print:", "llama_perf_context_print:",
                   "[end of text]", "<end_of_turn>"):
        idx = body.find(marker)
        if idx >= 0:
            cuts.append(idx)
    if cuts:
        body = body[:min(cuts)]
    return body.strip()


def _run_remote(
    ssh_host: str,
    binary_path: Path,
    model_path: Path,
    tokenizer: Any,
    prompt_id: str,
    expected_tool: str,
    user_text: str,
    n_threads: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    timeout_s: float,
) -> BenchResult:
    """One prompt via SSH-piped llama-completion on the board.

    External wall-clock is captured around `subprocess.run` for sanity but
    the JSONL records the on-board perf footer's numbers (those exclude
    the SSH/network round-trip and are the real CPU-side throughput).
    """
    res = BenchResult(
        prompt_id=prompt_id,
        expected_tool=expected_tool,
        prompt_text=user_text,
        mode="remote",
    )
    try:
        gen_prompt = _render_prompt(tokenizer, user_text)
        argv = _build_remote_argv(
            ssh_host=ssh_host,
            binary_path=binary_path,
            model_path=model_path,
            n_threads=n_threads,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        t0 = time.perf_counter()
        proc = subprocess.run(
            argv,
            input=gen_prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        external_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            res.error = (
                f"ssh+llama-completion exit {proc.returncode}: "
                f"{proc.stderr[-300:]!r}"
            )
            return res

        body = _slice_remote_response(proc.stdout)
        # Re-attach so parse_function_calls finds a complete block — `-r`
        # consumes the close tag in stdout.
        if body and "<start_function_call>" in body and "<end_function_call>" not in body:
            body = body + "<end_function_call>"
        res.response_text = body
        res.extras["external_wall_ms"] = external_ms

        try:
            perf = parse_llama_perf(proc.stderr + "\n" + proc.stdout)
        except ValueError as e:
            res.error = (
                f"perf footer not found ({e}); stderr tail: {proc.stderr[-300:]!r}"
            )
            return res
        res.wall_ms_load = perf.wall_ms_load
        res.wall_ms_prompt_eval = perf.wall_ms_prompt_eval
        res.wall_ms_decode = perf.wall_ms_decode
        res.wall_ms_total = perf.wall_ms_total
        res.n_prompt_tokens = perf.n_prompt_tokens
        res.n_decode_tokens = perf.n_decode_tokens
        res.prompt_tps = perf.prompt_eval_tps
        res.decode_tps = perf.decode_tps
        if perf.wall_ms_total > 0:
            res.overall_tps = (
                (perf.n_prompt_tokens + perf.n_decode_tokens)
                / (perf.wall_ms_total / 1000.0)
            )

        calls = parse_function_calls(body)
        res.parsed_call_count = len(calls)
        if calls:
            res.parsed_call = {"tool": calls[0]["tool"], "args": calls[0]["args"]}
    except subprocess.TimeoutExpired:
        res.error = f"TimeoutExpired after {timeout_s}s"
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
    return res


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _print_row(res: BenchResult) -> None:
    if res.error:
        print(
            f"  {res.prompt_id:<18} ERR  {res.error}",
            file=sys.stderr, flush=True,
        )
        return
    call_str = (
        f"{res.parsed_call['tool']}({res.parsed_call['args']})"
        if res.parsed_call else "(no parsable call)"
    )
    verdict = "OK " if res.parsed_call and res.parsed_call["tool"] == res.expected_tool else "MISS"
    print(
        f"  {res.prompt_id:<18} {verdict} "
        f"prompt={res.n_prompt_tokens:>4} decode={res.n_decode_tokens:>3} "
        f"wall={res.wall_ms_total / 1000:>5.2f}s "
        f"P={res.prompt_tps:>5.1f} D={res.decode_tps:>5.1f} O={res.overall_tps:>5.1f} tok/s  "
        f"→ {call_str}",
        file=sys.stderr, flush=True,
    )


def _summarize(rows: list[BenchResult]) -> None:
    ok = [r for r in rows if not r.error]
    if not ok:
        print("\nNo successful runs.", file=sys.stderr)
        return
    total_decode = sum(r.n_decode_tokens for r in ok)
    total_decode_s = sum(r.wall_ms_total for r in ok) / 1000.0
    avg_decode_tps = (
        sum(r.decode_tps for r in ok) / len(ok) if ok else 0.0
    )
    avg_overall_tps = (
        sum(r.overall_tps for r in ok) / len(ok) if ok else 0.0
    )
    correct = sum(
        1 for r in ok
        if r.parsed_call and r.parsed_call["tool"] == r.expected_tool
    )
    print(
        f"\n=== summary ({len(ok)}/{len(rows)} ok, {correct}/{len(ok)} expected-tool match) ===\n"
        f"  total decode tokens: {total_decode}\n"
        f"  total wall (model-side): {total_decode_s:.2f}s\n"
        f"  mean decode tok/s: {avg_decode_tps:.1f}\n"
        f"  mean overall tok/s: {avg_overall_tps:.1f}",
        file=sys.stderr, flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="FunctionGemma 270M GGUF throughput bench (local + remote SL2619).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("local", "remote"), default="local",
                   help="local = llama-cpp-python on host; remote = ssh-piped llama-completion.")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR),
                   help=f"HF model dir for chat template (default: {DEFAULT_TOKENIZER_DIR}).")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                   help=f"Local GGUF path (mode=local only; default: {DEFAULT_MODEL_PATH}).")
    p.add_argument("--ssh-host", default=None,
                   help="SSH alias for mode=remote (e.g. nouslogic-sl2619).")
    p.add_argument("--remote-binary", default="/mnt/sdcard/llama-cpp/llama-completion",
                   help="Remote llama-completion path on the board.")
    p.add_argument("--remote-model", default="/mnt/sdcard/models/functiongemma-270m/model.gguf",
                   help="Remote GGUF path on the board.")
    p.add_argument("--threads", type=int,
                   default=max(1, (os.cpu_count() or 4) // 2),
                   help="CPU threads (default: half of host cpu_count; on board pass --threads 2).")
    p.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                   help=f"llama-cpp-python n_ctx for mode=local (default: {DEFAULT_CTX_SIZE}).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help=f"-n / max decode (default: {DEFAULT_MAX_NEW_TOKENS} — function calls are short).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy, matches model_client.py).")
    p.add_argument("--top-k", type=int, default=1,
                   help="Top-k (mode=remote only; default: 1 to match greedy).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed (mode=remote only; default: 42 for reproducibility).")
    p.add_argument("--remote-timeout-s", type=float, default=DEFAULT_REMOTE_TIMEOUT_S,
                   help=f"Per-prompt SSH timeout (default: {DEFAULT_REMOTE_TIMEOUT_S}s).")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N prompts (smoke).")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup prompts before recording (default: 1; use 0 to disable).")
    p.add_argument("--out", default=None,
                   help="Output JSONL path (default: bench_results/functiongemma_<mode>_<ts>.jsonl).")
    args = p.parse_args(argv)

    # Validate args by mode
    if args.mode == "remote" and not args.ssh_host:
        print("--ssh-host is required for --mode remote", file=sys.stderr)
        return 2
    tokenizer_dir = _expand(args.tokenizer)
    if not tokenizer_dir.exists():
        print(f"tokenizer dir not found: {tokenizer_dir}", file=sys.stderr)
        return 2

    # Output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        _expand(args.out) if args.out
        else DEFAULT_BENCH_DIR / f"functiongemma_{args.mode}_{ts}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[bench] mode={args.mode} out={out_path}", file=sys.stderr)
    print(f"[bench] loading tokenizer: {tokenizer_dir}", file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    llm = None
    if args.mode == "local":
        model_path = _expand(args.model)
        if not model_path.exists():
            print(f"GGUF not found: {model_path}", file=sys.stderr)
            return 2
        from llama_cpp import Llama
        print(
            f"[bench] loading GGUF: {model_path} "
            f"(n_threads={args.threads}, n_ctx={args.ctx_size})",
            file=sys.stderr, flush=True,
        )
        llm = Llama(
            model_path=str(model_path),
            n_ctx=args.ctx_size,
            n_threads=args.threads,
            verbose=False,
        )

    prompts = DEFAULT_PROMPTS if args.limit is None else DEFAULT_PROMPTS[: args.limit]

    # Warmup — model load is cold-cache; the first prompt-eval pays it.
    # In remote mode this also primes the SSH connection so the first
    # recorded row isn't penalised by ControlMaster setup.
    if args.warmup > 0:
        warmup_prompts = prompts[: args.warmup]
        print(f"[bench] warmup {len(warmup_prompts)} prompt(s)...", file=sys.stderr)
        for pid, expected, text in warmup_prompts:
            if args.mode == "local":
                _run_local(llm, tokenizer, pid, expected, text,
                           args.max_new_tokens, args.temperature)
            else:
                _run_remote(args.ssh_host, _expand(args.remote_binary),
                            _expand(args.remote_model), tokenizer, pid, expected, text,
                            args.threads, args.max_new_tokens, args.temperature,
                            args.top_k, args.seed, args.remote_timeout_s)

    print(f"[bench] running {len(prompts)} prompt(s)...", file=sys.stderr)
    rows: list[BenchResult] = []
    with out_path.open("w", encoding="utf-8") as f:
        for pid, expected, text in prompts:
            if args.mode == "local":
                # Reset KV cache before each prompt so local numbers reflect
                # cold per-call cost — matches remote mode (one process per
                # prompt). Without reset, the constant 1500-token system
                # prompt is re-used and prompt-eval tok/s reads as 30k+,
                # which is an artifact, not a real throughput.
                llm.reset()
                res = _run_local(llm, tokenizer, pid, expected, text,
                                 args.max_new_tokens, args.temperature)
            else:
                res = _run_remote(args.ssh_host, _expand(args.remote_binary),
                                  _expand(args.remote_model), tokenizer, pid, expected, text,
                                  args.threads, args.max_new_tokens, args.temperature,
                                  args.top_k, args.seed, args.remote_timeout_s)
            rows.append(res)
            f.write(json.dumps(asdict(res), ensure_ascii=False))
            f.write("\n")
            f.flush()
            _print_row(res)

    _summarize(rows)
    print(f"\n[bench] wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
