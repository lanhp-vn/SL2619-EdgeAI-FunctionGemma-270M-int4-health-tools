#!/usr/bin/env python3
"""FunctionGemma single-turn smoke (M2 / Phase A).

Renders the vendor weather prompt through the HF chat template, runs the
Q4_K_M GGUF via `llama-cpp-python`, and asserts the model emits exactly one
parsable `<start_function_call>...<end_function_call>` block.

This is the validated **Path A** pattern from §15.6 — host-side tokenizer
for chat-template rendering plus programmatic GGUF inference. We do NOT use
`llama-cli --jinja` because of the two upstream bugs documented at
`docs/plans/FunctionGemma/README.md` §15.6 (tools never reach the chat
template; `--no-conversation` falls through into the REPL loop).

Acceptance gates:
- G_FG_LOAD: GGUF loads on host CPU through `llama-cpp-python`.
- G_FG_SINGLE: output contains exactly one well-formed function call,
  parsed into `{"tool": "get_current_temperature", "args": {"location": ...}}`.

Usage:
    uv run python scripts/functiongemma_smoke.py
    uv run python scripts/functiongemma_smoke.py --query "What's the temp in Paris?"
    uv run python scripts/functiongemma_smoke.py --dry-run    # no GGUF, no llama_cpp
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# §6.4 verbatim — *prompt-based* trigger for FG's function-calling mode.
# Vendor doc + cookbook cell 14 comment: "This line activates the model's
# function calling logic." Do not paraphrase.
DEVELOPER_TRIGGER = (
    "You are a model that can do function calling with the following functions"
)

# §6.4 verbatim — vendor weather schema. M2 deliberately uses this instead of
# the patient-YAML tools so we isolate "model emits valid wire format" from
# "our tool schema is correct".
WEATHER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    },
}

# §6.2 canonical regex with one tolerated drift: the chat template renders
# `call:NAME` (colon), but the Q4_K_M GGUF was observed in §15.4 Path B to
# emit `call NAME` (space). Accept both — anything else is a wire-format
# regression worth failing on.
_CALL_RE = re.compile(
    r"<start_function_call>\s*call[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
# Inner-arg regex per §6.2 — `key:<escape>STRING<escape>` or `key:bareval`.
_ARG_RE = re.compile(
    r"(\w+)\s*:\s*(?:<escape>(.*?)<escape>|([^,}]*))",
    re.DOTALL,
)

DEFAULT_TOKENIZER = "~/hf-cache/functiongemma-270m-it"
DEFAULT_MODEL = "~/hf-cache/functiongemma-270m-it/fg-q4_k_m.gguf"

# FunctionGemma's HF model card lists `n_ctx_train = 32768`. Bumping `n_ctx`
# from the prior 2048 default (which surfaced
# `n_ctx_seq (2048) < n_ctx_train (32768)` from llama-cpp) to 4096 gives the
# single-turn smoke 2x headroom over its actual prompt length (~600 rendered
# tokens + < 100 generated). It also matches the Unsloth notebook's
# `max_seq_length` (Phase D, §10.2). The warning still prints at 4096 — the
# only n_ctx that fully silences it is 32768, which costs ~300 MB of extra
# KV cache on host CPU and is overkill for M2's one-shot weather query.
# Future multi-turn work passes `--ctx-size 32768` explicitly; the default
# stays moderate.
DEFAULT_CTX_SIZE = 4096


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(p)))


def _missing_path_msg(label: str, path: Path) -> str:
    return (
        f"{label} not found: {path}\n"
        "Run M1.5 first (docs/plans/FunctionGemma/README.md §15.3):\n"
        "  hf download google/functiongemma-270m-it "
        f"--local-dir {DEFAULT_TOKENIZER}\n"
        "  python docs/references/upstream/llama.cpp/convert_hf_to_gguf.py "
        f"{DEFAULT_TOKENIZER} --outfile "
        f"{DEFAULT_TOKENIZER}/fg-bf16.gguf --outtype bf16\n"
        "  docs/references/upstream/llama.cpp/build/bin/llama-quantize "
        f"{DEFAULT_TOKENIZER}/fg-bf16.gguf "
        f"{DEFAULT_TOKENIZER}/fg-q4_k_m.gguf Q4_K_M"
    )


def render_prompt(tokenizer_dir: Path, query: str) -> str:
    """Render the FunctionGemma single-turn prompt with the weather tool.

    Imports `transformers` lazily so unit tests of `parse_function_calls` do
    not pull a 500 MB import graph.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    messages = [
        {"role": "developer", "content": DEVELOPER_TRIGGER},
        {"role": "user", "content": query},
    ]
    rendered = tok.apply_chat_template(
        messages,
        tools=[WEATHER_TOOL_SCHEMA],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, expected str"
        )
    return rendered


def parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Extract `<start_function_call>...<end_function_call>` blocks.

    Returns a list of `{"tool": NAME, "args": {key: value, ...}}` dicts.
    `<escape>STRING<escape>` values are unescaped to the raw string; bare
    values (numbers, ids) are returned as stripped strings — numeric coercion
    is intentionally deferred until M3 plumbs typed schemas through.
    """
    calls: list[dict[str, Any]] = []
    for m in _CALL_RE.finditer(text):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for am in _ARG_RE.finditer(body):
            key = am.group(1)
            esc, plain = am.group(2), am.group(3)
            if esc is not None:
                args[key] = esc
            else:
                stripped = (plain or "").strip()
                # Inner regex matches the empty tail at end-of-body; skip those.
                if stripped == "":
                    continue
                args[key] = stripped
        calls.append({"tool": name, "args": args})
    return calls


def _validate_prompt(prompt: str, query: str) -> None:
    """Enforce every M2-acceptance fragment is present in the rendered prompt.

    Each missing fragment names the chat-template contract it covers, so a
    failure points at the drift instead of "something is wrong".
    """
    required = [
        ("developer trigger", DEVELOPER_TRIGGER),
        ("tool name", "get_current_temperature"),
        ("function declaration token", "<start_function_declaration>"),
        ("user query", query),
        ("model generation marker", "<start_of_turn>model"),
    ]
    for label, fragment in required:
        if fragment not in prompt:
            raise ValueError(
                f"rendered prompt missing {label}: {fragment!r}"
            )


def _validate_one_call(calls: list[dict[str, Any]], expected_tool: str) -> dict[str, Any]:
    if len(calls) == 0:
        raise ValueError("model emitted zero <start_function_call> blocks")
    if len(calls) > 1:
        raise ValueError(
            f"model emitted {len(calls)} function calls, expected exactly 1"
        )
    call = calls[0]
    if call["tool"] != expected_tool:
        raise ValueError(
            f"expected tool {expected_tool!r}, got {call['tool']!r}"
        )
    if "location" not in call["args"]:
        raise ValueError(
            f"call missing required arg 'location': {call['args']}"
        )
    return call


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="FunctionGemma single-turn smoke (M2 / Phase A).",
    )
    p.add_argument("--query", default="What is the temperature in London?")
    p.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HF model directory (tokenizer + chat template).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="GGUF path for llama-cpp-python.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="CPU threads for llama-cpp-python (default: ~half of os.cpu_count).",
    )
    p.add_argument(
        "--ctx-size",
        type=int,
        default=DEFAULT_CTX_SIZE,
        help=(
            "llama-cpp-python `n_ctx`. Default 4096 covers the single-turn smoke "
            "with margin and avoids the `n_ctx_seq < n_ctx_train` warning at the "
            "old 2048 default. Bump to 8192+ for multi-turn future work; the "
            "trained context is 32768 (HF model card) but the KV cache scales "
            "linearly so larger values cost real RAM on host CPU."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and validate prompt only; do not load llama_cpp or the GGUF.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print rendered prompt + raw model output on stderr.",
    )
    args = p.parse_args(argv)

    tokenizer_dir = _expand(args.tokenizer)
    if not tokenizer_dir.exists():
        print(_missing_path_msg("tokenizer dir", tokenizer_dir), file=sys.stderr)
        return 2

    prompt = render_prompt(tokenizer_dir, args.query)
    _validate_prompt(prompt, args.query)
    if args.verbose:
        print("--- rendered prompt ---", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("--- end prompt ---", file=sys.stderr)

    if args.dry_run:
        print(
            f"PASS-DRY-RUN prompt={len(prompt)} chars; "
            "developer-trigger present; tool='get_current_temperature'"
        )
        return 0

    model_path = _expand(args.model)
    if not model_path.exists():
        print(_missing_path_msg("GGUF", model_path), file=sys.stderr)
        return 2

    # Lazy import — keeps `--dry-run` clean of the 100 MB llama_cpp graph.
    from llama_cpp import Llama
    print(
        f"[smoke] loading GGUF: {model_path} (n_threads={args.threads})",
        file=sys.stderr, flush=True,
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        verbose=False,
    )
    print(
        f"[smoke] generating (max_tokens={args.max_new_tokens}, greedy)",
        file=sys.stderr, flush=True,
    )
    # The chat template emits a literal `<bos>` and `Llama.__call__` also
    # prepends one via `add_bos=True`; without the strip llama-cpp-python
    # warns "Detected duplicate leading <bos> ... reduce response quality".
    gen_prompt = prompt.removeprefix("<bos>")
    out = llm(
        gen_prompt,
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        echo=False,
    )
    # `Llama.__call__` returns `CreateCompletionResponse | Iterator[...]` —
    # the iterator only on `stream=True`. We never set that, so narrow.
    assert isinstance(out, dict), f"unexpected llama-cpp response type: {type(out)}"
    text = out["choices"][0]["text"]
    if args.verbose:
        print("--- raw model output ---", file=sys.stderr)
        print(text, file=sys.stderr)
        print("--- end output ---", file=sys.stderr)

    if "<unk>" in text:
        print(f"FAIL <unk> in output: {text!r}", file=sys.stderr)
        return 1

    calls = parse_function_calls(text)
    try:
        call = _validate_one_call(calls, expected_tool="get_current_temperature")
    except ValueError as e:
        print(f"FAIL {e}\nraw: {text!r}", file=sys.stderr)
        return 1

    print(f"PASS {json.dumps(call, ensure_ascii=False, separators=(', ', ': '))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
