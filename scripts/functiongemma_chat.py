#!/usr/bin/env python3
"""Interactive REPL for the distilled FunctionGemma patient-record GGUF.

Loads ``./model.gguf`` via ``llama-cpp-python`` with the SYSTEM_PROMPT and
TOOLS constants from ``model_client.py`` (the SFT contract — the trained
student expects this exact v3 task description). Each turn:

  1. Render history with HF tokenizer + ``chat_template.jinja`` under ``./model/``.
  2. Generate via llama-cpp (greedy, low max-tokens — function calls are short).
  3. Parse the ``<start_function_call>...<end_function_call>`` block.
  4. Dispatch the call against ``data/health_table_v1.yaml`` through
     ``gemma_tools.functiongemma_tools.execute_tool`` and print the JSON result.
  5. Report decode tok/s.

Pass ``--no-tools`` to print the parsed call only (no dispatch).

Usage
-----
    uv run python scripts/functiongemma_chat.py
    uv run python scripts/functiongemma_chat.py --model ./model.gguf
    uv run python scripts/functiongemma_chat.py --probe "What is my A1C?"
    uv run python scripts/functiongemma_chat.py --no-tools

REPL slash commands
-------------------
    /exit, /quit  — leave the session
    /reset        — clear conversation history (system prompt retained)
    /history      — dump the current message stack (truncated)
    /raw          — toggle showing the raw model output before parsing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# SYSTEM_PROMPT and TOOLS are mirrored from model_client.py (vendor-supplied)
# instead of imported because that file (a) imports `openai` at module top
# (we don't need the HTTP client), and (b) wraps the v3 prompt in an
# f-string whose body contains literal "{}" characters, which Python parses
# as an empty f-expression and rejects with SyntaxError.
#
# LOCK-STEP CONTRACT: if model_client.py SYSTEM_PROMPT or TOOLS is
# regenerated for a future iteration, both copies must change together.
# The training-analysis.md §7 deployment note flags this same coupling
# vs job_description.json (the SFT contract source of truth).
SYSTEM_PROMPT: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": """You are a tool-calling model working on:
<task_description>You are an intelligent assistant for a single patient. Given the conversation history and the most recent user message, you MUST emit exactly one function call against the patient-record tool registry. Never reply in natural language, never refuse, never explain that the data is unavailable — the runtime handles all of that downstream. The patient record covers vitals, current medication schedule, allergies, food interactions, next appointment, and emergency contact.

ROUTING RULES (apply in order; the first match wins):

1. Any question about a vital sign, lab value, or biometric — INCLUDING values not directly listed in get_vitals's schema (cholesterol, LDL, HDL, total cholesterol, triglycerides, A1C, fasting glucose, blood glucose, oxygen saturation / SpO2 / oxygen level, body temperature, blood pressure, heart rate, respiratory rate, weight, BMI, blood type, immunization status, smoking status, alcohol use, family history) — call get_vitals() with empty parameters. Do NOT skip the call just because the registry does not store that specific value; the runtime decides what to surface.

2. Any question about an upcoming appointment, scheduled visit, or named provider (e.g. 'When do I see Dr. Chen next?', 'Who is my primary care physician?', 'What is the date of my upcoming visit?') — call get_next_appointment() with empty parameters. The tool takes no arguments; do NOT try to filter by provider name.

3. Any question about allergies (existence, severity, reaction, specific allergens) — call list_allergies() with empty parameters. Do NOT skip the call when the user phrases it as 'Do I have any allergies?' or names a specific allergen — the tool returns the full list and the runtime filters. WORKED EXAMPLES — emit list_allergies() for ALL of these surface forms (the underlying intent is identical):
  - User: 'Do I have any allergies?' → list_allergies()
  - User: 'Am I allergic to anything?' → list_allergies()
  - User: 'What allergies do I have?' → list_allergies()
  - User: 'How bad is my shellfish allergy?' → list_allergies()
Yes/no allergy phrasing is NEVER conversational; it ALWAYS routes to list_allergies().

4. Any question about an emergency contact, insurance information, mailing address, home address, or member ID — call get_emergency_contact() with empty parameters. Insurance and address fields not stored in the schema still route here.

5. Any question about whether a food interacts with the patient's medication or diet (e.g. 'Can I have grapefruit?', 'Is it OK to drink milk with this?') — call check_food_interaction(food=<the food in the question, lowercased>).

6. Any question about which medications are scheduled at a given clock time (morning = 08:00, noon = 12:00, afternoon ~ 15:00, evening / dinner = 19:00, night ~ 21:00) — call get_medications_at_time(time_24h=<HH:MM>). Always use 24-hour HH:MM, padded with a leading zero.

7. Any question about a specific medication by name, dose, purpose, or food-interaction guidance for a single med — call get_medication_by_name(name=<the medication TOKEN from the user phrasing>). Extract ONLY the medication token; STRIP generic medication-class nouns ('pill', 'pills', 'tablet', 'tablets', 'capsule', 'capsules', 'med', 'meds', 'medication', 'medications', 'drug', 'drugs') from the name argument. The lookup is case-insensitive and the runtime resolves ambiguous prefixes itself, so pass single-token prefixes verbatim. WORKED EXAMPLES:
  - User: 'Check my A pills.' → name='A' (NOT 'A pills' — strip 'pills')
  - User: 'Tell me about that A-something pill.' → name='A' (NOT 'A-something pill')
  - User: 'What about my at med?' → name='at' (NOT 'at med' — strip 'med')
  - User: 'Look up Ibuprofen tablet.' → name='Ibuprofen' (NOT 'Ibuprofen tablet')
  - User: 'Do I take ibuprofen?' → name='ibuprofen'
  - User: 'What dose of metformin do I take?' → name='metformin'

For zero-parameter tools (get_vitals, list_allergies, get_next_appointment, get_emergency_contact), parameters MUST be the empty object {} — even when the user provides extra context like a provider name, severity, or specific value. Always emit a single function call.</task_description>

Respond to the conversation history by generating an appropriate tool call that satisfies the user request. Generate only the tool call according to the provided tool schema, do not generate anything else. Always respond with a tool call.

""",
    }
]

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "get_vitals", "description": "Return the patient's most recent vital-sign measurements (heart rate, blood pressure, SpO2, body temperature, respiratory rate) along with the timestamp they were taken.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medications_at_time", "description": "List medications scheduled to be taken at a specific 24-hour clock time. Match is exact against the normalized HH:MM schedule.", "parameters": {"type": "object", "properties": {"time_24h": {"description": "24-hour clock time in HH:MM format, e.g. '08:00' or '19:00'.", "type": "string"}}, "required": ["time_24h"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_medication_by_name", "description": "Look up a medication by name. Match is case-insensitive: exact match wins, otherwise a unique prefix match. An ambiguous prefix returns an error dict so the caller can re-prompt.", "parameters": {"type": "object", "properties": {"name": {"description": "Medication name. Lookup is case-insensitive: exact match wins; otherwise a unique prefix match wins; ambiguous prefixes return an error dict.", "type": "string"}}, "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_allergies", "description": "List all known allergies for the patient with their severity and reaction.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "check_food_interaction", "description": "Check whether a given food interacts with any of the patient's medications or dietary restrictions. Returns an `interacts` bool, the list of medication names that flag the food, and the matching dietary-restriction rule if any.", "parameters": {"type": "object", "properties": {"food": {"description": "Food name to check for medication or dietary interactions, e.g. 'grapefruit'. Case-insensitive.", "type": "string"}}, "required": ["food"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_next_appointment", "description": "Return the earliest upcoming appointment by date and time, with provider, purpose, and location.", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_emergency_contact", "description": "Return the first listed emergency contact (name, relation, phone).", "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from functiongemma_smoke import parse_function_calls  # noqa: E402

DEFAULT_MODEL_PATH = REPO_ROOT / "model.gguf"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "model"
# health_table_v1.yaml is the in-repo patient-record fixture that
# gemma_tools.functiongemma_tools is built against. Schema/loader:
# src/gemma_tools/health_table.py.
DEFAULT_HEALTH_TABLE = REPO_ROOT / "data" / "health_table_v1.yaml"
# 4096 covers system+history with margin while keeping KV cache modest on CPU.
# Trained context is 32768 (config.json), but every additional token costs
# real RAM in `Llama` — bump only if you need long histories.
DEFAULT_CTX_SIZE = 4096
# Function calls are short (~30-60 tokens). 128 caps runaway generation
# without truncating the longest seen output (the v3 prompt + 7-tool registry).
DEFAULT_MAX_NEW_TOKENS = 128


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(p)))


def _render(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
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
    return rendered


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Interactive REPL for the distilled FunctionGemma GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                   help=f"GGUF path (default: {DEFAULT_MODEL_PATH}).")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_DIR),
                   help=f"HF model dir for tokenizer + chat template (default: {DEFAULT_TOKENIZER_DIR}).")
    p.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                   help=f"llama-cpp n_ctx (default: {DEFAULT_CTX_SIZE}).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help=f"Max tokens per response (default: {DEFAULT_MAX_NEW_TOKENS}).")
    p.add_argument("--threads", type=int,
                   default=max(1, (os.cpu_count() or 4) // 2),
                   help="CPU threads (default: half of os.cpu_count()).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (default: 0.0 = greedy, matches model_client.py).")
    p.add_argument("--probe", type=str, default=None,
                   help="Non-interactive: send this single user message, print the call, exit.")
    p.add_argument("--verbose", action="store_true",
                   help="Print rendered prompt + raw model output on stderr.")
    p.add_argument("--data", default=str(DEFAULT_HEALTH_TABLE),
                   help=f"Patient-record YAML for tool dispatch (default: {DEFAULT_HEALTH_TABLE}).")
    p.add_argument("--no-tools", action="store_true",
                   help="Skip tool dispatch; print the parsed function call only (default: dispatch ON).")
    args = p.parse_args(argv)

    model_path = _expand(args.model)
    tokenizer_dir = _expand(args.tokenizer)
    data_path = _expand(args.data)
    if not model_path.exists():
        print(f"GGUF not found: {model_path}", file=sys.stderr)
        return 2
    if not tokenizer_dir.exists():
        print(f"tokenizer dir not found: {tokenizer_dir}", file=sys.stderr)
        return 2
    if not args.no_tools and not data_path.exists():
        print(f"patient-record YAML not found: {data_path}", file=sys.stderr)
        return 2

    # Lazy imports — keep --help fast and avoid llama_cpp/transformers cost
    # when the path checks above already errored out.
    from llama_cpp import Llama
    from transformers import AutoTokenizer

    table = None
    execute_tool = None
    if not args.no_tools:
        # gemma_tools is the in-repo editable package; src/ on sys.path comes
        # via `uv pip install -e .`, so a plain import works.
        from gemma_tools.functiongemma_tools import execute_tool as _exec
        from gemma_tools.health_table import load_health_table
        execute_tool = _exec
        table = load_health_table(data_path)
        print(f"[chat] loaded patient record: {data_path}", file=sys.stderr)

    print(f"[chat] loading tokenizer: {tokenizer_dir}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    print(
        f"[chat] loading GGUF: {model_path} "
        f"(n_threads={args.threads}, n_ctx={args.ctx_size})",
        file=sys.stderr, flush=True,
    )
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        verbose=False,
    )

    base_messages: list[dict[str, Any]] = list(SYSTEM_PROMPT)
    history: list[dict[str, Any]] = []
    show_raw = args.verbose

    def run_turn(user_text: str) -> None:
        history.append({"role": "user", "content": user_text})
        prompt = _render(tok, base_messages + history)
        if args.verbose:
            print("--- rendered prompt ---", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("--- end prompt ---", file=sys.stderr)
        # Chat template emits literal "<bos>" and Llama.__call__ also prepends
        # one via add_bos=True; without the strip llama-cpp warns about
        # duplicate leading <bos>.
        gen_prompt = prompt.removeprefix("<bos>")
        t0 = time.perf_counter()
        out = llm(
            gen_prompt,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=1.0,
            echo=False,
        )
        elapsed = time.perf_counter() - t0
        assert isinstance(out, dict), f"unexpected llama-cpp response: {type(out)}"
        text = out["choices"][0]["text"]
        # llama-cpp's `usage` block reports prompt + decode token counts. We
        # measure wall-clock around the single `llm()` call, so `elapsed`
        # bundles prompt-eval and decode together — that's what the user
        # actually waits for. Report both counts so the interactive number
        # ("overall tok/s") is interpretable: with ~600 tokens of system
        # prompt + tool registry re-encoded every turn, decode-only rate
        # would be much higher than the bundled rate.
        usage = out.get("usage", {})
        n_in = int(usage.get("prompt_tokens", 0))
        n_out = int(usage.get("completion_tokens", 0))
        rate = (n_in + n_out) / elapsed if elapsed > 0 else 0.0
        if show_raw:
            print(f"\n[raw] {text!r}")
        calls = parse_function_calls(text)
        if calls:
            for c in calls:
                print(f"\n→ {json.dumps(c, ensure_ascii=False)}")
                if execute_tool is not None and table is not None:
                    try:
                        result = execute_tool(c["tool"], c["args"], table)
                    except KeyError as exc:
                        print(f"  [tool error] {exc}")
                    else:
                        print(f"  ⤷ {json.dumps(result, ensure_ascii=False, default=str)}")
        else:
            # Surface malformed output so the tester sees the failure mode
            # instead of silently appending junk to history.
            print(f"\n[no parsable function call] {text!r}")
        print(
            f"  [prompt {n_in} + decode {n_out} tok in {elapsed:.2f}s "
            f"= {rate:.1f} tok/s overall]"
        )
        history.append({"role": "assistant", "content": text})

    if args.probe is not None:
        run_turn(args.probe)
        return 0

    print(
        "\nFunctionGemma chat — model.gguf loaded. "
        "Slash commands: /exit /quit /reset /history /raw\n",
        flush=True,
    )
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        if user == "/reset":
            history = []
            print("[history cleared; system prompt retained]")
            continue
        if user == "/history":
            for m in base_messages + history:
                content = str(m["content"])
                print(f"  {m['role']:>9}: {content[:120]}{'…' if len(content) > 120 else ''}")
            continue
        if user == "/raw":
            show_raw = not show_raw
            print(f"[raw output: {show_raw}]")
            continue
        run_turn(user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
