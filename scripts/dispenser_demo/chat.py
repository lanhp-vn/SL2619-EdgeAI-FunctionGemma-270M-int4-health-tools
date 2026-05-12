#!/usr/bin/env python3
"""Interactive REPL for the dispenser-demo iter-002 BF16 merged checkpoint.

Counterpart to `scripts/functiongemma/chat.py` — same UX (slash commands,
`--probe`, greedy decode, tool dispatch) but loads the HF safetensors
checkpoint directly via `transformers` in BF16, with the **dispenser-demo**
SYSTEM_PROMPT + TOOLS pulled from `releases/.../distil/job_description.json`.

The point of this REPL is to probe the model itself — the weights that came
out of the Distil platform, before any quantization or board deployment.
If this REPL works but board GGUF doesn't, the bug is downstream
(quantization / runtime), not in the trained model.

Usage
-----
    uv run python scripts/dispenser_demo/chat.py
    uv run python scripts/dispenser_demo/chat.py --device cuda
    uv run python scripts/dispenser_demo/chat.py --probe "When's my next appointment?"
    uv run python scripts/dispenser_demo/chat.py --no-tools

REPL slash commands
-------------------
    /exit, /quit  — leave the session
    /reset        — clear conversation history (system prompt retained)
    /history      — dump the current message stack
    /raw          — toggle showing the raw model output before parsing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RELEASE = REPO_ROOT / "releases/functiongemma-270m/002-dispenser-demo"
DEFAULT_CHECKPOINT = DEFAULT_RELEASE / "merged"
DEFAULT_JOB_DESCRIPTION = DEFAULT_RELEASE / "distil/job_description.json"
DEFAULT_HEALTH_TABLE = REPO_ROOT / "data/health_table_v2.yaml"
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_CTX_SIZE = 4096

# Mirrors `_DISTIL_SYSTEM_TEMPLATE` in eval_holdout.py — must stay byte-equal
# to the wrapping Distil's `model_client.py` uses at inference.
_DISTIL_SYSTEM_TEMPLATE = (
    "You are a tool-calling model working on:\n"
    "<task_description>{task_description}</task_description>\n\n"
    "Respond to the conversation history by generating an appropriate tool "
    "call that satisfies the user request. Generate only the tool call "
    "according to the provided tool schema, do not generate anything else. "
    "Always respond with a tool call.\n\n"
)

# `<start_function_call>call:<name>{<args-json>}<end_function_call>` — same
# wire format the iter-001 baseline parser handles. The `call:` prefix
# (with optional whitespace + colon) is FunctionGemma-base behaviour.
_FN_CALL_RE = re.compile(
    r"<start_function_call>\s*(?:call\s*:\s*)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<args>\{.*?\})\s*<end_function_call>",
    re.DOTALL,
)


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _load_prompt_setup(jd_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    system = {
        "role": "system",
        "content": _DISTIL_SYSTEM_TEMPLATE.format(task_description=jd["task_description"]),
    }
    tools = list(jd.get("tools", []))
    return system, tools


def _parse_call(raw: str) -> tuple[str, dict[str, Any]] | None:
    m = _FN_CALL_RE.search(raw)
    if not m:
        return None
    name = m.group("name")
    try:
        args = json.loads(m.group("args"))
    except json.JSONDecodeError:
        # Borderline JSON (e.g. trailing commas) — surface raw to the caller
        # rather than silently nullifying the call.
        return name, {"__parse_error__": m.group("args")}
    if not isinstance(args, dict):
        return name, {"__parse_error__": m.group("args")}
    return name, args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Interactive REPL for the dispenser-demo BF16 merged checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                   help=f"HF safetensors dir for BF16 seam (default: {DEFAULT_CHECKPOINT}).")
    p.add_argument("--gguf", default=None,
                   help="GGUF path — switches to llama-cpp-python (CPU). "
                        "Tokenizer still loads from --tokenizer-dir for the chat template.")
    p.add_argument("--tokenizer-dir", default=None,
                   help="HF tokenizer dir for GGUF seam (default: --checkpoint).")
    p.add_argument("--job-description", default=str(DEFAULT_JOB_DESCRIPTION),
                   help=f"Distil job_description.json (default: {DEFAULT_JOB_DESCRIPTION}).")
    p.add_argument("--data", default=str(DEFAULT_HEALTH_TABLE),
                   help=f"Patient-record YAML for tool dispatch (default: {DEFAULT_HEALTH_TABLE}).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="torch device for BF16 seam (default: auto). Ignored in --gguf mode.")
    p.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                   help=f"llama-cpp n_ctx for --gguf seam (default: {DEFAULT_CTX_SIZE}).")
    p.add_argument("--threads", type=int,
                   default=max(1, (os.cpu_count() or 4) // 2),
                   help="CPU threads for --gguf seam (default: half of os.cpu_count()).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help=f"Max tokens per response (default: {DEFAULT_MAX_NEW_TOKENS}).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy, matches Phase 1.6 eval).")
    p.add_argument("--probe", type=str, default=None,
                   help="Non-interactive: send this single user message, print the call, exit.")
    p.add_argument("--no-tools", action="store_true",
                   help="Skip tool dispatch; print the parsed function call only.")
    p.add_argument("--verbose", action="store_true",
                   help="Print rendered prompt + raw model output on stderr.")
    args = p.parse_args(argv)

    jd_path = _expand(args.job_description)
    data_path = _expand(args.data)
    if not jd_path.exists():
        print(f"job_description not found: {jd_path}", file=sys.stderr)
        return 2
    if not args.no_tools and not data_path.exists():
        print(f"patient-record YAML not found: {data_path}", file=sys.stderr)
        return 2

    use_gguf = args.gguf is not None
    gguf_path = _expand(args.gguf) if use_gguf else None
    if use_gguf and not gguf_path.exists():
        print(f"gguf not found: {gguf_path}", file=sys.stderr)
        return 2
    tokenizer_dir = _expand(args.tokenizer_dir) if args.tokenizer_dir else _expand(args.checkpoint)
    if not tokenizer_dir.exists():
        print(f"tokenizer dir not found: {tokenizer_dir}", file=sys.stderr)
        return 2
    checkpoint = _expand(args.checkpoint)
    if not use_gguf and not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    table = None
    execute_tool = None
    if not args.no_tools:
        from gemma_tools.dispenser_demo.health_table_v2 import load_health_table_v2
        from gemma_tools.dispenser_demo.tools import execute_tool as _exec
        execute_tool = _exec
        table = load_health_table_v2(data_path)
        print(f"[chat] loaded patient record: {data_path}", file=sys.stderr)

    # Lazy imports — keep --help fast, and only pay for the seam we're using.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    model = None
    llm = None
    device = "cpu"
    if use_gguf:
        from llama_cpp import Llama
        print(
            f"[chat] loading {gguf_path} via llama-cpp "
            f"(n_ctx={args.ctx_size}, threads={args.threads})",
            file=sys.stderr,
        )
        t0 = time.perf_counter()
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=args.ctx_size,
            n_threads=args.threads,
            verbose=False,
        )
        print(f"[chat] loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        print(f"[chat] loading {checkpoint} (bf16, {device})", file=sys.stderr)
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
        model.to(device).eval()
        print(f"[chat] loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    system, tools = _load_prompt_setup(jd_path)
    history: list[dict[str, Any]] = []
    show_raw = bool(args.verbose)

    def respond(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        messages = [system] + history
        prompt = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError(f"apply_chat_template returned {type(prompt).__name__}")
        # `<bos>` is auto-prepended by the tokenizer; strip it from the
        # template render so we don't end up with two leading BOS tokens.
        prompt = prompt.removeprefix("<bos>")
        if args.verbose:
            print("--- rendered prompt ---", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("--- end prompt ---", file=sys.stderr)

        t_gen = time.perf_counter()
        if use_gguf:
            llm.reset()
            out = llm(
                prompt,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=1.0 if args.temperature == 0 else 0.95,
                echo=False,
                stop=["<end_function_call>", "<end_of_turn>"],
            )
            raw = out["choices"][0]["text"]
            # llama-cpp strips the stop string — re-attach for the parser.
            if raw and "<start_function_call>" in raw and "<end_function_call>" not in raw:
                raw = raw + "<end_function_call>"
            n_tok = int(out.get("usage", {}).get("completion_tokens", 0)) or len(
                tok(raw, add_special_tokens=False)["input_ids"]
            )
        else:
            import torch
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-5),
                    top_k=0 if args.temperature == 0 else 50,
                    top_p=1.0 if args.temperature == 0 else 0.95,
                    pad_token_id=tok.eos_token_id,
                )
            gen_ids = gen[0][inputs.input_ids.shape[1]:]
            raw = tok.decode(gen_ids, skip_special_tokens=False)
            n_tok = int(gen_ids.shape[0])
        elapsed = time.perf_counter() - t_gen
        history.append({"role": "assistant", "content": raw})

        if show_raw or args.verbose:
            print(f"[raw] {raw!r}", file=sys.stderr)
        print(
            f"[chat] {n_tok} tok in {elapsed:.2f}s "
            f"({n_tok / max(elapsed, 1e-6):.1f} tok/s)",
            file=sys.stderr,
        )

        parsed = _parse_call(raw)
        if parsed is None:
            print(f"model> {raw.strip()}")
            return
        name, call_args = parsed
        print(f"call → {name}({json.dumps(call_args)})")

        if args.no_tools or execute_tool is None or table is None:
            return
        if "__parse_error__" in call_args:
            print(f"[warn] could not JSON-parse args: {call_args['__parse_error__']!r}")
            return
        try:
            result = execute_tool(name, call_args, table)
        except KeyError as exc:
            print(f"[error] {exc}")
            return
        print(f"tool → {json.dumps(result, default=str)}")

    if args.probe is not None:
        respond(args.probe)
        return 0

    print("ready. commands: /exit  /reset  /history  /raw", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/reset":
            history.clear()
            print("(history cleared)", file=sys.stderr)
            continue
        if line == "/history":
            print(json.dumps(history, indent=2, default=str))
            continue
        if line == "/raw":
            show_raw = not show_raw
            print(f"(raw output: {'on' if show_raw else 'off'})", file=sys.stderr)
            continue
        respond(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
