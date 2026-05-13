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
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

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

# region: humanizers — render ISO dates/times as natural-language phrases
#
# The TTS layer (Piper) handles raw "2026-05-11 07:30" by reading every digit
# ("two zero two six dash zero five..."), which is robotic and clashes with the
# rest of the sentence. Humanize on the way in: "May 11 at 7:30 in the morning"
# reads naturally without Piper-side tweaks. Keep the helpers cheap (no
# datetime parsing) so they survive the BusyBox/Yocto image.

_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def _humanize_date(date_str: str) -> str:
    """'2026-05-20' → 'May 20'. Year is dropped — patient context rarely needs
    it, and 'twenty twenty six' adds noise for TTS. On parse failure return
    the raw string so we never crash the format path.
    """
    try:
        y, m, d = date_str.split("-")
        return f"{_MONTHS[int(m) - 1]} {int(d)}"
    except (ValueError, IndexError):
        return date_str


def _humanize_time(time_str: str) -> str:
    """'07:30' → '7:30 in the morning', '14:00' → '2 in the afternoon',
    '20:00' → '8 in the evening', '23:30' → '11:30 at night'. Drops the
    leading zero and uses a period-of-day clause so AM/PM doesn't get
    spelled out letter-by-letter by Piper.
    """
    try:
        h_s, m_s = time_str.split(":")[:2]
        h, m = int(h_s), int(m_s)
    except (ValueError, IndexError):
        return time_str
    if h < 5:
        period = "at night"
    elif h < 12:
        period = "in the morning"
    elif h < 17:
        period = "in the afternoon"
    elif h < 21:
        period = "in the evening"
    else:
        period = "at night"
    h12 = h % 12 or 12
    hm = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    return f"{hm} {period}"


def _humanize_schedule(schedule: str) -> str:
    """'08:00, 20:00' → '8 in the morning and 8 in the evening'. Comma-separated
    HH:MM list → English conjunction. Single time stays bare.
    """
    times = [t.strip() for t in schedule.split(",") if t.strip()]
    if not times:
        return schedule
    humanized = [_humanize_time(t) for t in times]
    if len(humanized) == 1:
        return humanized[0]
    if len(humanized) == 2:
        return " and ".join(humanized)
    return ", ".join(humanized[:-1]) + ", and " + humanized[-1]


def _humanize_measured_suffix(measured: str) -> str:
    """'2026-05-11 07:30' or '2026-05-11T07:30:00' → ', measured at 7:30 in
    the morning on May 11'. Empty input → empty string.
    """
    if not measured:
        return ""
    sep = "T" if "T" in measured else " "
    parts = measured.split(sep, 1)
    if len(parts) != 2:
        return f", measured {measured}"
    date_s = parts[0]
    time_s = parts[1][:5]  # take HH:MM, drop any seconds
    return f", measured at {_humanize_time(time_s)} on {_humanize_date(date_s)}"


# endregion


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
        meds = table.get("medications", [])
        # Empty / omitted time_24h ⇒ list every med with its full schedule
        # (matches `src/gemma_tools/functiongemma/tools.py:TimeArgs` after the
        # 2026-05-02 schema relaxation — model emits {} for broad questions).
        if not t:
            return [
                {"name": m["name"], "dose": m["dose"], "schedule": m["schedule"],
                 "with_food": m["with_food"], "purpose": m["purpose"]}
                for m in meds
            ]
        if not _HHMM_RE.match(t):
            return {"error": "invalid_arguments", "tool": name,
                    "messages": [f"time_24h must match HH:MM, got {t!r}"]}
        return [
            {"name": m["name"], "dose": m["dose"], "schedule": m["schedule"],
             "with_food": m["with_food"], "purpose": m["purpose"]}
            for m in meds
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


def _fmt_tool_error(tool: str, r: dict) -> str:
    """Render an `{"error": ...}` payload from any tool as a user-friendly
    sentence. Mirrors `chat.py:_format_tool_error` — keep in sync.
    """
    err = r.get("error", "unknown")
    if err == "invalid_arguments":
        msgs = r.get("messages") or []
        msg_str = "; ".join(msgs) if msgs else "missing required argument"
        return f"I couldn't run `{tool}` — {msg_str}. Try rephrasing the question."
    if err == "no_contacts":
        return "No emergency contact is on file."
    if err == "no_appointments":
        return "No upcoming appointments are on file."
    if err == "invalid_food":
        return "Tell me which food you'd like to check (e.g. grapefruit, alcohol)."
    if err == "invalid_name":
        return "Tell me which medication you'd like to look up by name."
    return f"Tool error: {err}."


def _is_error(r: Any) -> bool:
    return isinstance(r, dict) and "error" in r


def _fmt_vitals(q: str, r: dict) -> str:
    if _is_error(r):
        return _fmt_tool_error("get_vitals", r)
    suf = _humanize_measured_suffix(r.get("last_measured", ""))
    if _has(q, "blood pressure", " bp", "bp?", "systolic", "diastolic", "tension"):
        return f"Your blood pressure is {r['blood_pressure_systolic']} over {r['blood_pressure_diastolic']}{suf}."
    if _has(q, "heart rate", "pulse", "bpm", " hr"):
        return f"Your heart rate is {r['heart_rate_bpm']} beats per minute{suf}."
    if _has(q, "oxygen", "spo2", "o2", "saturation"):
        return f"Your oxygen saturation is {r['spo2_percent']} percent{suf}."
    if _has(q, "temperature", "temp", "fever"):
        return f"Your body temperature is {r['body_temperature_c']} degrees Celsius{suf}."
    if _has(q, "breathing", "respiratory", "respiration"):
        return f"Your respiratory rate is {r['respiratory_rate']} breaths per minute{suf}."
    if _has(q, "cholesterol", "ldl", "hdl", "triglyceride", "a1c", "glucose",
            "sugar", "weight", "bmi", "blood type"):
        return ("That value isn't recorded in your vitals. "
                f"Available: heart rate, blood pressure, oxygen, temperature, and respiratory rate{suf}.")
    return (f"Heart rate {r['heart_rate_bpm']} beats per minute, "
            f"blood pressure {r['blood_pressure_systolic']} over {r['blood_pressure_diastolic']}, "
            f"oxygen {r['spo2_percent']} percent, "
            f"temperature {r['body_temperature_c']} degrees Celsius, "
            f"respiratory rate {r['respiratory_rate']} breaths per minute{suf}.")


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
    if _is_error(r):
        return _fmt_tool_error("get_emergency_contact", r)
    if _has(q, "phone", "number", "call", "reach"):
        return f"Call {r['name']} ({r['relation']}) at {r['phone']}."
    if _has(q, "who ", "name"):
        return f"Your emergency contact is {r['name']} ({r['relation']})."
    return f"Your emergency contact is {r['name']}, your {r['relation']}, at {r['phone']}."


def _fmt_appointment(q: str, r: dict) -> str:
    if _is_error(r):
        return _fmt_tool_error("get_next_appointment", r)
    when = f"{_humanize_date(r['date'])} at {_humanize_time(r['time'])}"
    if _has(q, "why", "purpose", "what for", "about"):
        return f"Your next appointment is for {r['purpose']}."
    if _has(q, "where", "location", "address"):
        return f"Your next appointment is at {r['location']}."
    if _has(q, "when", "date", "time"):
        return f"Your next appointment is on {when}."
    if _has(q, "who", "doctor", "provider", "with"):
        return f"Your next appointment is with {r['provider']}."
    return f"Your next appointment is on {when} with {r['provider']} for {r['purpose']} at {r['location']}."


def _fmt_medication(q: str, r: dict) -> str:
    if "error" in r:
        if r["error"] == "ambiguous":
            return f'"{r["name"]}" is ambiguous — could be: ' + ", ".join(r["matches"]) + "."
        if r["error"] == "no_match":
            return f'No medication named "{r["name"]}" is on your record.'
        return f"Lookup error: {r['error']}."
    name = r["name"]
    schedule_h = _humanize_schedule(r["schedule"])
    if _has(q, "dose", "how much", "how many"):
        return f"Your {name} dose is {r['dose']}."
    if _has(q, "purpose", "what for", "why am i taking", "why do i take"):
        return f"You take {name} for {r['purpose']}."
    if _has(q, "when", "schedule", "what time"):
        return f"You take {name} at {schedule_h}."
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
    return f"{name} {r['dose']} taken at {schedule_h}, {food_clause}, for {r['purpose']}."


def _fmt_meds_at_time(args: dict, r: list | dict) -> str:
    if _is_error(r):
        return _fmt_tool_error("get_medications_at_time", r)  # type: ignore[arg-type]
    when = args.get("time_24h")
    when_h = _humanize_time(when) if when else None
    if not r:
        if when_h:
            return f"You have no medications scheduled at {when_h}."
        return "No medications are on your record."
    if when_h:
        parts = [f"{m['name']} {m['dose']}" for m in r]
        return f"At {when_h} you take: " + ", ".join(parts) + "."
    parts = []
    for m in r:
        food = ", with food" if m.get("with_food") else ""
        parts.append(f"{m['name']} {m['dose']} at {_humanize_schedule(m['schedule'])}{food}")
    return "Your medications: " + "; ".join(parts) + "."


def _fmt_food(args: dict, r: dict) -> str:
    if _is_error(r):
        return _fmt_tool_error("check_food_interaction", r)
    food = r.get("food", args.get("food", "that food"))
    if not r["interacts"]:
        return f"{food.capitalize()} is fine — no interaction with your medications or diet."
    bits = []
    if r.get("with_meds"):
        bits.append(f"interacts with {', '.join(r['with_meds'])}")
    if r.get("rule"):
        bits.append(f'flagged by your dietary rule: "{r["rule"]}"')
    return f"Avoid {food} — " + " and ".join(bits) + "."


# Sentinel "tool" name for the out-of-scope refusal path. `_run_turn` routes
# both the no-parsable-call case and the KeyError-from-dispatch case through
# `format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)` so the same
# `chat_board.format_response` capture wrapper in `dispenser_voice.py` picks
# up the refusal sentence for TTS. Without this, an out-of-scope question
# (per plan.md §6 — "refuse everything else") produced silent skip on the
# speaker while stdout still logged it as `[no parsable function call]`.
OUT_OF_SCOPE_TOOL = "_out_of_scope"

OUT_OF_SCOPE_REFUSAL = (
    "I can only help with your medications, vitals, allergies, "
    "appointments, and emergency contact."
)


def _fmt_out_of_scope(_q: str) -> str:
    # Single canned line — kept constant on purpose so the spoken refusal is
    # immediately recognizable across turns. Plan.md §6 also constrains the
    # iter-002 refusal seeds to a tight phrasing; mirror that intent here.
    return OUT_OF_SCOPE_REFUSAL


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
        OUT_OF_SCOPE_TOOL: lambda: _fmt_out_of_scope(q),
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


def prime_prompt_cache(
    prefix: str, llama_bin: Path, model: Path,
    n_threads: int, ctx_size: int, prompt_cache: Path,
    verbose: bool = False,
) -> float:
    """Build a prefix-only KV cache file. Called once when the cache is
    missing; subsequent `run_inference` calls open it read-only.

    Why prefix-only: `--prompt-cache` without `--prompt-cache-ro` saves
    the FULL post-decode state. When the next turn's prompt diverges at
    the question slot, llama.cpp's truncation leaves the KV state in a
    skewed position and the model samples gibberish ("ology", "menopause",
    "women's health" — observed 2026-05-02 on b8925). Priming the cache
    with ONLY the prefix bytes (no user question, no suffix, no decode)
    means every turn diverges at exactly the same token offset — clean
    truncation, stable output.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     dir="/tmp", encoding="utf-8") as tf:
        tf.write(prefix)
        tmp_path = tf.name
    try:
        argv = [
            str(llama_bin),
            "-m", str(model),
            "-f", tmp_path,
            "-t", str(n_threads),
            "-c", str(ctx_size),
            "-n", "1",  # decode 1 token so the cache file is definitely written
            "--temp", "0.0",
            "--top-k", "1",
            "--prompt-cache", str(prompt_cache),
            "-no-cnv", "--single-turn",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            argv, capture_output=True, encoding="utf-8", errors="replace",
            check=False,
        )
        wall_s = time.perf_counter() - t0
        if verbose:
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"prime_prompt_cache: llama-completion exit {proc.returncode}: "
                f"{proc.stderr[-300:]!r}"
            )
        if not prompt_cache.exists():
            raise RuntimeError(
                f"prime_prompt_cache: cache file not written at {prompt_cache}"
            )
        return wall_s
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def run_inference(
    question: str, prefix: str, suffix: str,
    llama_bin: Path, model: Path,
    n_threads: int, n_predict: int, ctx_size: int,
    temp: float, top_k: int, seed: int,
    verbose: bool = False,
    prompt_cache: Path | None = None,
) -> tuple[str, dict, float]:
    """Subprocess one llama-completion call. Returns (text, perf, wall_s).

    By default both stdout and stderr are captured silently — the REPL only
    prints the parsed call + the formatted answer + a one-line perf summary.
    Pass ``verbose=True`` (chat_board.py: ``--verbose``) to live-tee the
    llama-completion stderr (model loader dump, KV alloc, sampler config,
    perf footer) for cold-call debugging.

    ``prompt_cache``: path to a llama.cpp KV-cache file (`--prompt-cache`).
    The cache is opened READ-ONLY here (`--prompt-cache-ro`) so it stays
    at the prefix-only state primed by `prime_prompt_cache`; every turn
    diverges at the same offset (end of prefix) so llama.cpp's truncation
    is clean. Pass `None` to disable caching (every turn re-evaluates the
    full ~1627-token system prefix → ~33 s/turn).
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
        ]
        if prompt_cache is not None:
            # `--prompt-cache-ro` is critical: without it, llama.cpp saves
            # the full post-decode state to the cache file, which then
            # corrupts turn 2's truncation when the question diverges
            # (model emits gibberish like "ology", "menopause"). Read-only
            # keeps the cache file at the prefix-only state primed before
            # the first turn.
            argv += ["--prompt-cache", str(prompt_cache), "--prompt-cache-ro"]
        argv += ["-no-cnv", "--single-turn"]
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
        echo = sys.stderr if verbose else None
        t_out = threading.Thread(
            target=_drain_stream, args=(proc.stdout, stdout_buf, None),
        )
        t_err = threading.Thread(
            target=_drain_stream, args=(proc.stderr, stderr_buf, echo),
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
    verbose: bool = False, prompt_cache: Path | None = None,
) -> str:
    """One turn — run inference, parse, dispatch, format, print. Returns the
    raw model text (so the REPL can append it to history if it wants).
    """
    print("[thinking…]", file=sys.stderr, flush=True)
    try:
        text, perf, wall_s = run_inference(
            user_text, prefix, suffix, llama_bin, model,
            n_threads, n_predict, ctx_size, temp, top_k, seed,
            verbose=verbose, prompt_cache=prompt_cache,
        )
    except Exception as e:
        print(f"\n[inference error] {e}", flush=True)
        return ""

    if show_raw:
        print(f"\n[raw] {text!r}", flush=True)

    calls = parse_function_calls(text)
    if not calls:
        # Out-of-scope / unparsable model output. Route through `format_response`
        # with the OUT_OF_SCOPE_TOOL sentinel so the same capture wrappers used
        # by `dispenser_voice.py` (`_capture_format` → `_last_formatted`) pick
        # up the refusal sentence for TTS. Without this branch, the speaker
        # stays silent on out-of-scope questions even though stdout shows them.
        print(f"\n[no parsable function call] {text!r}", flush=True)
        print(f"  >> {format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)}",
              flush=True)
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
            # Unknown / hallucinated tool name — same TTS rationale as the
            # no-call branch above. Surface the error for stdout debugging
            # AND emit a spoken refusal so the dispenser doesn't go silent.
            print(f"  [tool error] {exc}", flush=True)
            print(f"  >> {format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)}",
                  flush=True)
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


def _resolve_prompt_cache(
    args: argparse.Namespace, prefix: str, model_path: Path,
) -> tuple[Path | None, str]:
    """Decide the prompt-cache path + prime it if missing. Returns
    (cache_path_or_None, label_for_status_line).

    Encapsulated so the type narrowing on `Path | None` stays local — the
    callers always get either a fully-resolved Path (cache file exists on
    return) or None (caching disabled / priming failed).
    """
    if args.no_prompt_cache:
        return None, "off"
    pc: Path = args.prompt_cache or Path(f"/tmp/fg_pc_{args.model_name}.bin")
    if pc.exists():
        return pc, f"{pc} (existing, {pc.stat().st_size // 1024} KiB)"
    print(
        f"[chat] priming prompt cache (~28 s, one-time): {pc}",
        file=sys.stderr, flush=True,
    )
    try:
        wall = prime_prompt_cache(
            prefix, args.llama, model_path, args.threads,
            args.ctx_size, pc, verbose=args.verbose,
        )
    except Exception as exc:
        print(
            f"[chat] prompt-cache priming failed: {exc} — falling back to "
            "uncached mode (every turn re-evaluates the full prefix)",
            file=sys.stderr,
        )
        return None, "off (priming failed)"
    return pc, f"{pc} (primed in {wall:.1f}s, {pc.stat().st_size // 1024} KiB)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="On-board interactive FunctionGemma chat (pure stdlib).",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help=f"Dir holding prompt-prefix.txt, prompt-suffix.txt, "
                        f"health_table.json, and the GGUF (default: {DEFAULT_MODEL_DIR}).")
    p.add_argument("--model-name", type=str,
                   default="finetuned_functiongemma_q4_0.gguf",
                   help="GGUF basename inside --model-dir. Default: "
                        "finetuned_functiongemma_q4_0.gguf — the recommended on-board variant "
                        "(see releases/.../gguf/RECOMMENDED.md). Pass the basename of any "
                        "finetuned_functiongemma_*.gguf to chat against another quant.")
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
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Live-tee llama-completion stderr (model loader + KV alloc + "
                        "sampler config + perf footer). Default: silent — the REPL "
                        "only prints the parsed call + answer + a one-line perf summary.")
    p.add_argument("--prompt-cache", type=Path, default=None,
                   help="Path to a llama.cpp KV-cache file (`--prompt-cache`). "
                        "Default: /tmp/fg_pc_<model>.bin (per-model). First turn "
                        "writes it; subsequent turns load it, dropping per-turn wall "
                        "from ~33 s to ~3-5 s by skipping prompt-eval of the 1627-token "
                        "system prefix. Pass `--no-prompt-cache` to disable.")
    p.add_argument("--no-prompt-cache", action="store_true",
                   help="Disable the prompt KV cache (always re-evaluate the full prompt).")
    args = p.parse_args(argv)

    prefix_path = args.model_dir / "prompt-prefix.txt"
    suffix_path = args.model_dir / "prompt-suffix.txt"
    table_path = args.model_dir / "health_table.json"
    model_path = args.model_dir / args.model_name

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

    # Resolve prompt-cache path. Default: /tmp/fg_pc_<model_basename>.bin so
    # different quants don't share a cache (their KV state is incompatible).
    prompt_cache, cache_label = _resolve_prompt_cache(
        args, prefix, model_path,
    )
    print(f"[chat] prompt-cache: {cache_label}", file=sys.stderr)

    history: list[dict[str, str]] = []

    if args.probe is not None:
        text = _run_turn(
            args.probe, prefix, suffix, table,
            args.llama, model_path, args.threads, args.max_new_tokens,
            args.ctx_size, args.temperature, args.top_k, args.seed, show_raw,
            verbose=args.verbose, prompt_cache=prompt_cache,
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
            verbose=args.verbose, prompt_cache=prompt_cache,
        )
        history.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    sys.exit(main())


# endregion
