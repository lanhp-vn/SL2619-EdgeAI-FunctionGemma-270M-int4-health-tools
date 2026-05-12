#!/usr/bin/env python3
"""FunctionGemma chat — dispense-intent variant for SL2619.

A thin wrapper around ``chat_board.py`` (must live in the same directory)
that hijacks two tool calls — ``get_medications_at_time`` and
``get_medication_by_name`` — to route to the dispenser-demo dispense intent
from ``docs/plans/dispenser-demo/plan.md`` §6:

  1. Print a mock BLE notify payload (``5A A5 01 00``) to stdout, in place
     of a real ``pybleno`` notify. The byte string matches plan §6.2 so an
     ESP32 central with the firmware spec from §6.3 would see the same wire
     contract in Phase 2.
  2. Emit the canned response from plan §6.1, verbatim and unconditional
     (model args are ignored):

       "Your medication is being dispensed. Please check the dispenser."

Why this exists: lets us prototype the dispense flow end-to-end on the
SL2619 *today*, on the iter-001 FunctionGemma Q4_0 model, without
retraining for a new ``dispense_medication()`` tool. The model schema is
byte-identical, so the warm prompt cache at
``/tmp/fg_pc_finetuned_functiongemma_q4_0.gguf.bin`` reuses across
``chat_board.py`` and this wrapper — first turn stays fast.

The non-dispense tools (vitals / allergies / appointment / contact /
food-interaction) are inherited from ``chat_board.py`` unchanged. Patient
identity comes from a separate JSON (default ``health_table_dispense.json``
under ``--model-dir``), so the original ``health_table.json`` is left
intact for the non-dispense REPL.

Layout on the board:
  /mnt/sdcard/models/functiongemma-270m/chat_board.py              (existing)
  /mnt/sdcard/models/functiongemma-270m/chat_board_dispense.py     (this file)
  /mnt/sdcard/models/functiongemma-270m/health_table.json          (existing v1)
  /mnt/sdcard/models/functiongemma-270m/health_table_dispense.json (new patient)
  /mnt/sdcard/models/functiongemma-270m/prompt-{prefix,suffix}.txt (shared)
  /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_q4_0.gguf (shared)

Usage:
  python3 /mnt/sdcard/models/functiongemma-270m/chat_board_dispense.py
  python3 /mnt/sdcard/models/functiongemma-270m/chat_board_dispense.py \\
      --probe "give me my Lisinopril"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Load the unmodified chat_board.py as a module. Same-directory import keeps
# the wrapper self-contained; if you move this file, also move chat_board.py.
_spec = importlib.util.spec_from_file_location("chat_board", HERE / "chat_board.py")
assert _spec is not None and _spec.loader is not None
chat_board = importlib.util.module_from_spec(_spec)
sys.modules["chat_board"] = chat_board
_spec.loader.exec_module(chat_board)

# region: dispense-intent overrides

# plan.md §6.1 — verbatim canned response. Do NOT personalize.
DISPENSE_CANNED = "Your medication is being dispensed. Please check the dispenser."
# plan.md §6.2 — 4-byte notify payload: header 0x5A 0xA5 + opcode 0x01 + status 0x00.
BLE_PAYLOAD_HEX = "5A A5 01 00"
# Iter-001 tool names hijacked into the dispense intent. The model still
# emits these per its existing routing rules; we replace the dispatcher.
DISPENSE_TOOLS: frozenset[str] = frozenset({
    "get_medications_at_time",
    "get_medication_by_name",
})

_orig_dispatch = chat_board.dispatch
_orig_format_response = chat_board.format_response


def dispatch(name: str, args: dict[str, Any], table: dict[str, Any]) -> Any:
    if name in DISPENSE_TOOLS:
        # Mock the BLE peripheral. Phase 2 swaps this for a real pybleno
        # notify on characteristic 0xFFB2 (see plan.md §6.2).
        print(f"  [BLE→ESP32] {BLE_PAYLOAD_HEX}", flush=True)
        return {"status": "dispensed"}
    return _orig_dispatch(name, args, table)


def format_response(user_text: str, tool: str, args: dict[str, Any], result: Any) -> str:
    if tool in DISPENSE_TOOLS:
        # Args are intentionally ignored — dispense intent is unconditional.
        return DISPENSE_CANNED
    return _orig_format_response(user_text, tool, args, result)


# Reassign at module level. chat_board._run_turn resolves `dispatch` and
# `format_response` as bare names against the module's globals, so swapping
# here intercepts every subsequent call from inside the inherited REPL.
chat_board.dispatch = dispatch
chat_board.format_response = format_response

# endregion

# region: main — clone of chat_board.main with --health-table-name override

DEFAULT_TABLE_NAME = "health_table_dispense.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="FunctionGemma dispense-intent variant (on-board, pure stdlib).",
    )
    p.add_argument("--model-dir", type=Path, default=chat_board.DEFAULT_MODEL_DIR,
                   help=f"Dir holding prompt-prefix.txt, prompt-suffix.txt, the GGUF, "
                        f"and <health-table-name> (default: {chat_board.DEFAULT_MODEL_DIR}).")
    p.add_argument("--health-table-name", type=str, default=DEFAULT_TABLE_NAME,
                   help=f"Patient JSON basename inside --model-dir (default: {DEFAULT_TABLE_NAME}). "
                        "Kept separate from health_table.json so the original chat_board.py REPL "
                        "still works against the v1 fixture.")
    p.add_argument("--model-name", type=str,
                   default="finetuned_functiongemma_q4_0.gguf",
                   help="GGUF basename inside --model-dir.")
    p.add_argument("--llama", type=Path, default=chat_board.DEFAULT_LLAMA)
    p.add_argument("--threads", type=int, default=chat_board.DEFAULT_THREADS)
    p.add_argument("--ctx-size", type=int, default=chat_board.DEFAULT_CTX_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=chat_board.DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=chat_board.DEFAULT_TEMP)
    p.add_argument("--top-k", type=int, default=chat_board.DEFAULT_TOP_K)
    p.add_argument("--seed", type=int, default=chat_board.DEFAULT_SEED)
    p.add_argument("--probe", type=str, default=None,
                   help="Non-interactive: send one message, print the call, exit.")
    p.add_argument("--raw", action="store_true",
                   help="Show raw model output before parsing.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Live-tee llama-completion stderr.")
    p.add_argument("--prompt-cache", type=Path, default=None,
                   help="Path to llama.cpp KV-cache file. Default /tmp/fg_pc_<model>.bin "
                        "(shared with chat_board.py — same model + same prefix → same cache).")
    p.add_argument("--no-prompt-cache", action="store_true")
    args = p.parse_args(argv)

    prefix_path = args.model_dir / "prompt-prefix.txt"
    suffix_path = args.model_dir / "prompt-suffix.txt"
    table_path = args.model_dir / args.health_table_name
    model_path = args.model_dir / args.model_name

    for f in (prefix_path, suffix_path, table_path, model_path, args.llama):
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            return 2

    prefix = prefix_path.read_text(encoding="utf-8")
    suffix = suffix_path.read_text(encoding="utf-8")
    table = json.loads(table_path.read_text(encoding="utf-8"))

    patient_name = (table.get("patient") or {}).get("name", "<unknown>")
    print(
        f"[chat-dispense] model={model_path} table={table_path.name} "
        f"patient={patient_name!r} threads={args.threads} n_predict={args.max_new_tokens}",
        file=sys.stderr,
    )
    print(
        f"[chat-dispense] dispense tools: {sorted(DISPENSE_TOOLS)} → "
        f"mock BLE notify {BLE_PAYLOAD_HEX} + canned response",
        file=sys.stderr,
    )

    show_raw = args.raw

    prompt_cache, cache_label = chat_board._resolve_prompt_cache(
        args, prefix, model_path,
    )
    print(f"[chat-dispense] prompt-cache: {cache_label}", file=sys.stderr)

    history: list[dict[str, str]] = []

    if args.probe is not None:
        text = chat_board._run_turn(
            args.probe, prefix, suffix, table,
            args.llama, model_path, args.threads, args.max_new_tokens,
            args.ctx_size, args.temperature, args.top_k, args.seed, show_raw,
            verbose=args.verbose, prompt_cache=prompt_cache,
        )
        return 0 if text else 1

    print(
        "\nFunctionGemma chat (DISPENSE variant) — slash: /exit /quit /reset /history /raw\n",
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
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history = []
            print("[history cleared]")
            continue
        if user == "/history":
            for m in history:
                content = m["content"]
                print(f"  {m['role']:>9}: {content[:120]}{'…' if len(content) > 120 else ''}")
            continue
        if user == "/raw":
            show_raw = not show_raw
            print(f"[raw output: {show_raw}]")
            continue

        history.append({"role": "user", "content": user})
        text = chat_board._run_turn(
            user, prefix, suffix, table,
            args.llama, model_path, args.threads, args.max_new_tokens,
            args.ctx_size, args.temperature, args.top_k, args.seed, show_raw,
            verbose=args.verbose, prompt_cache=prompt_cache,
        )
        history.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    sys.exit(main())

# endregion
