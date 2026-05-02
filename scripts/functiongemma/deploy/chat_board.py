#!/usr/bin/env python3
"""On-board interactive FunctionGemma chat — pure stdlib, runs on SL2619.

Mirrors the host scripts/functiongemma_chat.py REPL contract:
  - reads questions interactively
  - subprocess llama-completion for each turn (each call cold-mmaps)
  - parses the FIRST <start_function_call>...<end_function_call> block
    (and only the first — the model occasionally loops greedy decoding;
    the host script also has this bug, see "fever" turn in chat history)
  - dispatches against health_table.json (YAML pre-converted on host)
  - formats a single-sentence natural-language response
  - prints timing parsed out of llama.cpp's common_perf_print footer
  - slash commands: /exit /quit /reset /history /raw

Pure stdlib because the SL2619 board has python3.12 but no
pip / transformers / pyyaml / pydantic / llama-cpp-python.

Layout assumed on the board:
  /mnt/sdcard/models/functiongemma-270m/prompt-prefix.txt   (system + tools)
  /mnt/sdcard/models/functiongemma-270m/prompt-suffix.txt   (turn-close + model-open)
  /mnt/sdcard/models/functiongemma-270m/health_table.json   (YAML→JSON of health record)
  /mnt/sdcard/models/functiongemma-270m/model.gguf
  /mnt/sdcard/llama-cpp/llama-completion

Usage on the board:
  python3 /mnt/sdcard/llama-cpp/fg-chat-board.py
  python3 /mnt/sdcard/llama-cpp/fg-chat-board.py --probe "What is my heart rate?"
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Force UTF-8 I/O regardless of the board's locale (Yocto often defaults to
# C/POSIX → ASCII). Without this, llama.cpp's stderr bytes get decoded with
# `surrogateescape`, then our echo to sys.stderr can't re-encode the U+DC80
# surrogates and crashes. Belt-and-braces: reconfigure stdio + set
# PYTHONIOENCODING (env var only matters for child Python procs we don't
# spawn, but harmless).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Defaults match the on-board layout from
# docs/deployment/sl2619-functiongemma.md §3a.
DEFAULT_MODEL_DIR = Path("/mnt/sdcard/models/functiongemma-270m")
DEFAULT_LLAMA = Path("/mnt/sdcard/llama-cpp/llama-completion")
DEFAULT_THREADS = 2
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TEMP = 0.0
DEFAULT_TOP_K = 1
DEFAULT_SEED = 42
# Cap the runtime context. FunctionGemma's `n_ctx_train` is 32768; allocating
# the KV cache for that on the board's 1.67 GiB headroom (alongside a 520 MB
# FP16 model + ~500 MB compute buffer) flirts with OOM and slows cold mmap.
# 2048 is comfortably bigger than our ~1700-token prompt envelope.
DEFAULT_CTX_SIZE = 2048

# region: parsing
# Same regex as scripts/functiongemma_smoke.py — keep in sync if either
# moves. Accepts both `call:NAME` and `call NAME` (trained chat template
# emits the colon; some Q4_K_M GGUFs were observed to drop it).
_CALL_RE = re.compile(
    r"<start_function_call>\s*call[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"(\w+)\s*:\s*(?:<escape>(.*?)<escape>|([^,}]*))",
    re.DOTALL,
)
# common_perf_print is the b8925+ format on the SL2619 board build.
# Older builds used llama_perf_context_print — accept both for portability.
# Run per-line (not on the merged stream) because `\s*` matches newlines and
# would let an early match's optional tps group consume the next line's tps.
_PERF_RE = re.compile(
    r"(?:common|llama)_perf(?:_context)?_print:\s+(?P<label>load time|prompt eval time|"
    r"eval time|total time)\s*=\s*(?P<ms>[\d.]+)\s*ms"
    r"(?:\s*/\s*(?P<count>\d+)\s*(?:tokens|runs))?"
    r"(?:[^,\n]*,\s*(?P<tps>[\d.]+)\s*tokens per second)?",
    re.IGNORECASE,
)


def parse_function_calls(text: str) -> list[dict[str, Any]]:
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
                if stripped == "":
                    continue
                args[key] = stripped
        calls.append({"tool": name, "args": args})
    return calls


def parse_perf(stream: str) -> dict[str, float]:
    """Pull n_prompt, n_decode, prompt_tps, decode_tps, total_ms out of the
    llama.cpp perf footer. Returns empty dict if nothing parsed.
    """
    fields: dict[str, dict[str, float]] = {}
    # Per-line scan so the optional tps group can't bleed across lines.
    for line in stream.splitlines():
        m = _PERF_RE.search(line)
        if not m:
            continue
        label = m.group("label").strip().replace(" ", "_")
        fields[label] = {
            "ms": float(m.group("ms")),
            "count": float(m.group("count") or 0),
            "tps": float(m.group("tps") or 0.0),
        }
    if not fields:
        return {}
    p = fields.get("prompt_eval_time", {})
    d = fields.get("eval_time", {})
    tot = fields.get("total_time", {})
    ld = fields.get("load_time", {})
    return {
        "n_prompt_tokens": int(p.get("count", 0)),
        "n_decode_tokens": int(d.get("count", 0)),
        "prompt_tps": p.get("tps", 0.0),
        "decode_tps": d.get("tps", 0.0),
        "load_ms": ld.get("ms", 0.0),
        "prompt_ms": p.get("ms", 0.0),
        "decode_ms": d.get("ms", 0.0),
        "total_ms": tot.get("ms", 0.0),
    }


# endregion

# region: tool dispatch (pure stdlib; mirrors gemma_tools.functiongemma.tools)


_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _split_schedule(schedule: str) -> list[str]:
    return [tok.strip() for tok in schedule.split(",") if tok.strip()]


def _med_to_dict(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": m["name"],
        "dose": m["dose"],
        "schedule": m["schedule"],
        "with_food": m["with_food"],
        "purpose": m["purpose"],
        "avoid_foods": list(m.get("avoid_foods") or []),
        "avoid_drugs": list(m.get("avoid_drugs") or []),
    }


def dispatch(name: str, args: dict[str, Any], table: dict[str, Any]) -> Any:
    if name == "get_vitals":
        return dict(table["vitals"])
    if name == "list_allergies":
        return [dict(a) for a in table.get("allergies", [])]
    if name == "get_emergency_contact":
        contacts = table.get("emergency_contacts", [])
        if not contacts:
            return {"error": "no_contacts"}
        return dict(contacts[0])
    if name == "get_next_appointment":
        appts = table.get("appointments", [])
        if not appts:
            return {"error": "no_appointments"}
        nxt = min(appts, key=lambda a: (a["date"], a["time"]))
        return dict(nxt)
    if name == "get_medications_at_time":
        t = str(args.get("time_24h", "")).strip()
        if not _HHMM_RE.match(t):
            return {"error": "invalid_arguments", "tool": name,
                    "messages": [f"time_24h must match HH:MM, got {t!r}"]}
        return [
            {"name": m["name"], "dose": m["dose"], "schedule": m["schedule"],
             "with_food": m["with_food"], "purpose": m["purpose"]}
            for m in table.get("medications", [])
            if t in _split_schedule(m["schedule"])
        ]
    if name == "get_medication_by_name":
        q = str(args.get("name", "")).strip()
        if not q:
            return {"error": "invalid_name", "name": q}
        ql = q.lower()
        meds = table.get("medications", [])
        exact = [m for m in meds if m["name"].lower() == ql]
        if exact:
            return _med_to_dict(exact[0])
        prefix = [m for m in meds if m["name"].lower().startswith(ql)]
        if not prefix:
            return {"error": "no_match", "name": q}
        if len(prefix) > 1:
            return {"error": "ambiguous", "name": q,
                    "matches": sorted(m["name"] for m in prefix)}
        return _med_to_dict(prefix[0])
    if name == "check_food_interaction":
        food_raw = str(args.get("food", "")).strip()
        if not food_raw:
            return {"error": "invalid_food", "food": food_raw}
        food = food_raw.lower()
        with_meds = [
            m["name"] for m in table.get("medications", [])
            if any(af.lower() == food for af in (m.get("avoid_foods") or []))
        ]
        rules = [
            r["rule"] for r in table.get("dietary_restrictions", [])
            if food in r["rule"].lower()
        ]
        return {"food": food_raw, "interacts": bool(with_meds) or bool(rules),
                "with_meds": with_meds, "rule": rules[0] if rules else None}
    raise KeyError(f"Unknown tool: {name!r}")


# endregion

# region: natural-language formatter — verbatim port from functiongemma_chat.py


def _has(q: str, *keywords: str) -> bool:
    return any(k in q for k in keywords)


def _fmt_vitals(q: str, r: dict) -> str:
    last = r.get("last_measured", "")
    suf = f" (measured {last})" if last else ""
    if _has(q, "blood pressure", " bp", "bp?", "systolic", "diastolic", "tension"):
        return f"Your blood pressure is {r['blood_pressure_systolic']}/{r['blood_pressure_diastolic']}{suf}."
    if _has(q, "heart rate", "pulse", "bpm", " hr"):
        return f"Your heart rate is {r['heart_rate_bpm']} bpm{suf}."
    if _has(q, "oxygen", "spo2", "o2", "saturation"):
        return f"Your oxygen saturation is {r['spo2_percent']}%{suf}."
    if _has(q, "temperature", "temp", "fever"):
        return f"Your body temperature is {r['body_temperature_c']}°C{suf}."
    if _has(q, "breathing", "respiratory", "respiration"):
        return f"Your respiratory rate is {r['respiratory_rate']} breaths/min{suf}."
    if _has(q, "cholesterol", "ldl", "hdl", "triglyceride", "a1c", "glucose",
            "sugar", "weight", "bmi", "blood type"):
        return ("That value isn't recorded in your vitals. "
                f"Available: heart rate, blood pressure, SpO2, temperature, respiratory rate{suf}.")
    return (f"HR {r['heart_rate_bpm']} bpm, "
            f"BP {r['blood_pressure_systolic']}/{r['blood_pressure_diastolic']}, "
            f"SpO2 {r['spo2_percent']}%, temp {r['body_temperature_c']}°C, "
            f"resp {r['respiratory_rate']}/min{suf}.")


def _fmt_allergies(q: str, r: list) -> str:
    if not r:
        return "No known allergies on file."
    for e in r:
        if e["substance"].lower() in q:
            return f"Yes — you have a {e['severity']} {e['substance']} allergy ({e['reaction']})."
    parts = [f"{e['substance']} ({e['severity']}, {e['reaction']})" for e in r]
    plural = "y" if len(r) == 1 else "ies"
    return f"You have {len(r)} known allerg{plural}: " + "; ".join(parts) + "."


def _fmt_emergency(q: str, r: dict) -> str:
    if _has(q, "phone", "number", "call", "reach"):
        return f"Call {r['name']} ({r['relation']}) at {r['phone']}."
    if _has(q, "who ", "name"):
        return f"Your emergency contact is {r['name']} ({r['relation']})."
    return f"Your emergency contact is {r['name']}, your {r['relation']}, at {r['phone']}."


def _fmt_appointment(q: str, r: dict) -> str:
    when = f"{r['date']} at {r['time']}"
    if _has(q, "why", "purpose", "what for", "about"):
        return f"Your next appointment is for {r['purpose']}."
    if _has(q, "where", "location", "address"):
        return f"Your next appointment is at {r['location']}."
    if _has(q, "when", "date", "time"):
        return f"Your next appointment is on {when}."
    if _has(q, "who", "doctor", "provider", "with"):
        return f"Your next appointment is with {r['provider']}."
    return f"Your next appointment is {when} with {r['provider']} for {r['purpose']} at {r['location']}."


def _fmt_medication(q: str, r: dict) -> str:
    if "error" in r:
        if r["error"] == "ambiguous":
            return f'"{r["name"]}" is ambiguous — could be: ' + ", ".join(r["matches"]) + "."
        if r["error"] == "no_match":
            return f'No medication named "{r["name"]}" is on your record.'
        return f"Lookup error: {r['error']}."
    name = r["name"]
    if _has(q, "dose", "how much", "how many"):
        return f"Your {name} dose is {r['dose']}."
    if _has(q, "purpose", "what for", "why am i taking", "why do i take"):
        return f"You take {name} for {r['purpose']}."
    if _has(q, "when", "schedule", "what time"):
        return f"You take {name} at {r['schedule']}."
    if "with food" in q:
        clause = "with food" if r["with_food"] else "without a food requirement"
        return f"Take {name} {clause}."
    if _has(q, "avoid", "interact"):
        foods = r.get("avoid_foods") or []
        drugs = r.get("avoid_drugs") or []
        if not foods and not drugs:
            return f"No interaction warnings on file for {name}."
        bits = []
        if foods:
            bits.append("avoid " + ", ".join(foods))
        if drugs:
            bits.append("avoid " + ", ".join(drugs) + " (drug interaction)")
        return f"{name}: " + "; ".join(bits) + "."
    food_clause = "with food" if r["with_food"] else "without food required"
    return f"{name} {r['dose']} taken at {r['schedule']} ({food_clause}) for {r['purpose']}."


def _fmt_meds_at_time(args: dict, r: list) -> str:
    when = args.get("time_24h", "(unknown)")
    if not r:
        return f"You have no medications scheduled at {when}."
    parts = [f"{m['name']} {m['dose']}" for m in r]
    return f"At {when} you take: " + ", ".join(parts) + "."


def _fmt_food(args: dict, r: dict) -> str:
    food = r.get("food", args.get("food", "that food"))
    if not r["interacts"]:
        return f"{food.capitalize()} is fine — no interaction with your medications or diet."
    bits = []
    if r.get("with_meds"):
        bits.append(f"interacts with {', '.join(r['with_meds'])}")
    if r.get("rule"):
        bits.append(f'flagged by your dietary rule: "{r["rule"]}"')
    return f"Avoid {food} — " + " and ".join(bits) + "."


def format_response(user_text: str, tool: str, args: dict, result: Any) -> str:
    q = user_text.lower()
    fns = {
        "get_vitals": lambda: _fmt_vitals(q, result),
        "list_allergies": lambda: _fmt_allergies(q, result),
        "get_emergency_contact": lambda: _fmt_emergency(q, result),
        "get_next_appointment": lambda: _fmt_appointment(q, result),
        "get_medication_by_name": lambda: _fmt_medication(q, result),
        "get_medications_at_time": lambda: _fmt_meds_at_time(args, result),
        "check_food_interaction": lambda: _fmt_food(args, result),
    }
    fn = fns.get(tool)
    return fn() if fn else f"[no formatter for tool: {tool}]"


# endregion

# region: inference subprocess


def _slice_response(stdout: str) -> str:
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


def _drain_stream(stream: Any, buf: list[str], echo: Any | None) -> None:
    """Read `stream` line-by-line into `buf`; if `echo`, also write to it.

    Runs in a thread so stdout and stderr can both be live-tee'd. Without
    this, llama.cpp's load-progress + KV-alloc + decode messages stay
    buffered until process exit — looks like a hang on a 50 s cold call.
    """
    try:
        for line in iter(stream.readline, ""):
            buf.append(line)
            if echo is not None:
                echo.write(line)
                echo.flush()
    finally:
        stream.close()


def run_inference(
    question: str, prefix: str, suffix: str,
    llama_bin: Path, model: Path,
    n_threads: int, n_predict: int, ctx_size: int,
    temp: float, top_k: int, seed: int,
) -> tuple[str, dict, float]:
    """Subprocess one llama-completion call. Returns (text, perf, wall_s).

    Tees stderr to the user's terminal in real time so cold-call progress
    (model load, KV alloc, prompt-eval messages) is visible. Stdout (the
    model output) is captured silently — we re-print the parsed call
    afterward.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     dir="/tmp", encoding="utf-8") as tf:
        tf.write(prefix)
        tf.write(question)
        tf.write(suffix)
        tmp_path = tf.name
    try:
        argv = [
            str(llama_bin),
            "-m", str(model),
            "-f", tmp_path,
            "-t", str(n_threads),
            "-c", str(ctx_size),
            "-n", str(n_predict),
            "--temp", str(temp),
            "--top-k", str(top_k),
            "--seed", str(seed),
            "-no-cnv", "--single-turn",
        ]
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        t_out = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_buf, None),
        )
        t_err = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_buf, sys.stderr),
        )
        t_out.start()
        t_err.start()
        rc = proc.wait()
        t_out.join()
        t_err.join()
        wall_s = time.perf_counter() - t0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
    stdout = "".join(stdout_buf)
    stderr = "".join(stderr_buf)
    if rc != 0:
        raise RuntimeError(
            f"llama-completion exit {rc}: {stderr[-300:]!r}"
        )
    text = _slice_response(stdout)
    perf = parse_perf(stderr + "\n" + stdout)
    return text, perf, wall_s


# endregion

# region: REPL


def _run_turn(
    user_text: str, prefix: str, suffix: str, table: dict,
    llama_bin: Path, model: Path, n_threads: int, n_predict: int,
    ctx_size: int, temp: float, top_k: int, seed: int, show_raw: bool,
) -> str:
    """One turn — run inference, parse, dispatch, format, print. Returns the
    raw model text (so the REPL can append it to history if it wants).
    """
    print("[generating... (cold call: ~45-60s for prompt eval at ~38 tok/s)]",
          file=sys.stderr, flush=True)
    try:
        text, perf, wall_s = run_inference(
            user_text, prefix, suffix, llama_bin, model,
            n_threads, n_predict, ctx_size, temp, top_k, seed,
        )
    except Exception as e:
        print(f"\n[inference error] {e}", flush=True)
        return ""

    if show_raw:
        print(f"\n[raw] {text!r}", flush=True)

    calls = parse_function_calls(text)
    if not calls:
        print(f"\n[no parsable function call] {text!r}", flush=True)
    else:
        # Take ONLY the first call. Greedy decoding occasionally loops the
        # same call until n_predict; the user-facing answer is the first
        # one regardless. Note repetition for visibility.
        if len(calls) > 1:
            print(f"\n[note: model emitted {len(calls)} calls; using the first]",
                  flush=True)
        c = calls[0]
        print(f"\n→ {json.dumps(c, ensure_ascii=False)}", flush=True)
        try:
            result = dispatch(c["tool"], c["args"], table)
        except KeyError as exc:
            print(f"  [tool error] {exc}", flush=True)
        else:
            print(f"  ⤷ {json.dumps(result, ensure_ascii=False, default=str)}", flush=True)
            print(f"  >> {format_response(user_text, c['tool'], c['args'], result)}",
                  flush=True)

    n_in = perf.get("n_prompt_tokens", 0)
    n_out = perf.get("n_decode_tokens", 0)
    p_tps = perf.get("prompt_tps", 0.0)
    d_tps = perf.get("decode_tps", 0.0)
    print(
        f"  [prompt {n_in} ({p_tps:.1f} tok/s) + "
        f"decode {n_out} ({d_tps:.1f} tok/s) "
        f"in {wall_s:.2f}s wall]",
        flush=True,
    )
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="On-board interactive FunctionGemma chat (pure stdlib).",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help=f"Dir holding prompt-prefix.txt, prompt-suffix.txt, "
                        f"health_table.json, model.gguf (default: {DEFAULT_MODEL_DIR}).")
    p.add_argument("--llama", type=Path, default=DEFAULT_LLAMA,
                   help=f"llama-completion binary (default: {DEFAULT_LLAMA}).")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help=f"-t N (default: {DEFAULT_THREADS} — match A55 core count).")
    p.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                   help=f"-c K (default: {DEFAULT_CTX_SIZE}; trained ctx is 32768, "
                        "but allocating that much KV blows past the board's RAM).")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help=f"-n M (default: {DEFAULT_MAX_NEW_TOKENS}; function calls are short).")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMP,
                   help=f"sampling temperature (default: {DEFAULT_TEMP} = greedy).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--probe", type=str, default=None,
                   help="Non-interactive: send one message, print the call, exit.")
    p.add_argument("--raw", action="store_true",
                   help="Show raw model output before parsing.")
    args = p.parse_args(argv)

    prefix_path = args.model_dir / "prompt-prefix.txt"
    suffix_path = args.model_dir / "prompt-suffix.txt"
    table_path = args.model_dir / "health_table.json"
    model_path = args.model_dir / "model.gguf"

    for f in (prefix_path, suffix_path, table_path, model_path, args.llama):
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            return 2

    prefix = prefix_path.read_text(encoding="utf-8")
    suffix = suffix_path.read_text(encoding="utf-8")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    print(
        f"[chat] model={model_path} threads={args.threads} n_predict={args.max_new_tokens}",
        file=sys.stderr,
    )
    print(
        f"[chat] prefix={len(prefix)}b suffix={len(suffix)}b "
        f"health-record loaded ({len(table.get('medications', []))} meds, "
        f"{len(table.get('allergies', []))} allergies)",
        file=sys.stderr,
    )

    show_raw = args.raw
    history: list[dict[str, str]] = []

    if args.probe is not None:
        text = _run_turn(
            args.probe, prefix, suffix, table,
            args.llama, model_path, args.threads, args.max_new_tokens,
            args.ctx_size, args.temperature, args.top_k, args.seed, show_raw,
        )
        return 0 if text else 1

    print(
        "\nFunctionGemma chat (on-board) — model.gguf staged. "
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
        text = _run_turn(
            user, prefix, suffix, table,
            args.llama, model_path, args.threads, args.max_new_tokens,
            args.ctx_size, args.temperature, args.top_k, args.seed, show_raw,
        )
        history.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    sys.exit(main())


# endregion
