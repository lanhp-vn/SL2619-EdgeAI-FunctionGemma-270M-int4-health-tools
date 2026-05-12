#!/usr/bin/env python3
"""Render canned TTS WAVs for the Layer D dispenser-demo voice pipeline.

Reads `data/health_table_v1_dispense_demo.yaml`, builds one canned response
sentence per tool (with the digits-to-words conversion from plan §5.2 via
`src/gemma_tools/dispenser_demo/wordform.py`), and renders each to a WAV
via host-side **Piper TTS** (https://github.com/rhasspy/piper). The board
has no TTS engine, so the WAVs are pre-rendered here and shipped via scp;
`dispenser_voice.py` picks the WAV by tool name after each LLM turn and
aplay-s it.

Why Piper over espeak-ng: Piper's neural voices (en_US-lessac-medium,
en_US-amy-medium, en_US-ryan-high, etc.) sound clearly human; espeak-ng's
formant synthesis is intelligible but unmistakably robotic. Both run
fully offline; Piper just adds a one-time ONNX model download (~30-60 MB
per voice) and an `onnxruntime` host install. Board side stays a plain
`aplay <wav>`.

This is the minimum-viable Layer D — one fixed framing per tool, with
the patient's actual data baked into the WAV at deploy time. Re-run this
script whenever the patient YAML changes.

Output:
  /tmp/dispenser_demo_board/tts/<tool>.wav      one WAV per tool
  /tmp/dispenser_demo_board/tts/manifest.json   tool → {text, wav} mapping
                                                  (debug aid; not used at runtime)

Tools covered (v1):
  get_vitals               — patient's vitals dump
  list_allergies           — full allergy list
  get_next_appointment     — date/time/provider/purpose/location
  get_emergency_contact    — name/relation/phone
  get_medications_at_time  — DISPENSE canned line (per plan §6.1)
  get_medication_by_name   — DISPENSE canned line (per plan §6.1)
  check_food_interaction   — generic fallback ("information available")

Run on host:
  # One-time setup (already done if `piper --help` resolves)
  uv tool install piper-tts
  mkdir -p ~/.local/share/piper-voices
  curl -fsSL -o ~/.local/share/piper-voices/en_US-lessac-medium.onnx \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
  curl -fsSL -o ~/.local/share/piper-voices/en_US-lessac-medium.onnx.json \\
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

  uv run python scripts/dispenser_demo/voice/build_tts_canned.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gemma_tools.dispenser_demo.wordform import (  # noqa: E402
    age_to_words,
    date_to_words,
    diagnoses_to_words,
    freeform_to_words,
    phone_to_words,
    time_to_words,
)

DEFAULT_YAML = REPO_ROOT / "data" / "health_table_v1_dispense_demo.yaml"
DEFAULT_OUT_DIR = Path("/tmp/dispenser_demo_board/tts")
# Piper voice model paths. Override with --piper-model. The medium-quality
# Lessac voice is the balance of natural prosody + small file size (~64 MB
# .onnx + 4.8 KB .json) that we want for a single dispenser deployment.
# Other good options: en_US-amy-medium (female, smoother), en_US-ryan-high
# (male, higher quality but ~110 MB).
DEFAULT_PIPER_MODEL = Path.home() / ".local" / "share" / "piper-voices" / "en_US-lessac-medium.onnx"
# `length-scale > 1.0` slows the voice; 1.05 is a touch slower than default
# without sounding sluggish. Dispenser context wants intelligibility over speed.
DEFAULT_LENGTH_SCALE = 1.05
# Inter-sentence silence (seconds). 0.4 keeps the WAV from sounding clipped
# at the end and adds a natural pause before the operator might respond.
DEFAULT_SENTENCE_SILENCE = 0.4


# --------------------------------------------------------------------------
# Sentence construction. Each function returns the digit-free string we want
# espeak-ng to render. The chat_board.py formatters do the same composition
# at runtime; here we pre-bake them with the patient YAML.
# --------------------------------------------------------------------------


def _vitals_sentence(table: dict) -> str:
    v = table["vitals"]
    # Temperature: 36.5 → "thirty six point five". The wordform helper
    # operates on integers; for the decimal we split + glue.
    temp_str = str(v["body_temperature_c"])
    if "." in temp_str:
        whole, frac = temp_str.split(".", 1)
        temp_words = f"{freeform_to_words(whole)} point {freeform_to_words(frac)}"
    else:
        temp_words = freeform_to_words(temp_str)
    return (
        f"Your vitals. "
        f"Heart rate {freeform_to_words(str(v['heart_rate_bpm']))} beats per minute. "
        f"Blood pressure {freeform_to_words(str(v['blood_pressure_systolic']))} "
        f"over {freeform_to_words(str(v['blood_pressure_diastolic']))}. "
        f"Oxygen {freeform_to_words(str(v['spo2_percent']))} percent. "
        f"Temperature {temp_words} Celsius. "
        f"Respiratory rate {freeform_to_words(str(v['respiratory_rate']))} "
        f"breaths per minute."
    )


def _allergies_sentence(table: dict) -> str:
    allergies = table.get("allergies", []) or []
    if not allergies:
        return "No known allergies on file."
    count_word = freeform_to_words(str(len(allergies)))
    plural = "y" if len(allergies) == 1 else "ies"
    parts = [
        f"{a['substance']}, {a['severity']}, causing {a['reaction']}"
        for a in allergies
    ]
    return f"You have {count_word} known allerg{plural}. " + ". ".join(parts) + "."


def _emergency_sentence(table: dict) -> str:
    contacts = table.get("emergency_contacts", []) or []
    if not contacts:
        return "No emergency contact is on file."
    c = contacts[0]
    return (
        f"Your emergency contact is {c['name']}, your {c['relation']}, "
        f"at {phone_to_words(c['phone'])}."
    )


def _appointment_sentence(table: dict) -> str:
    appts = table.get("appointments", []) or []
    if not appts:
        return "No upcoming appointments are on file."
    a = min(appts, key=lambda x: (x["date"], x["time"]))
    location = freeform_to_words(a["location"])
    return (
        f"Your next appointment is on {date_to_words(a['date'])} "
        f"at {time_to_words(a['time'])}, with {a['provider']} "
        f"for {a['purpose']} at {location}."
    )


def _dispense_sentence() -> str:
    # Plan §6.1 — verbatim.
    return "Your medication is being dispensed. Please check the dispenser."


def _food_fallback_sentence() -> str:
    # Generic fallback — per-food canned WAVs aren't tractable. The runtime
    # also prints the JSON detail on stdout, so the user sees the specifics
    # even if the speaker plays this generic line.
    return "I have food interaction information for you."


SENTENCE_BUILDERS = {
    "get_vitals": _vitals_sentence,
    "list_allergies": _allergies_sentence,
    "get_emergency_contact": _emergency_sentence,
    "get_next_appointment": _appointment_sentence,
    "get_medications_at_time": lambda _t: _dispense_sentence(),
    "get_medication_by_name": lambda _t: _dispense_sentence(),
    "check_food_interaction": lambda _t: _food_fallback_sentence(),
}


# --------------------------------------------------------------------------
# Piper TTS wrapper
# --------------------------------------------------------------------------


def render_wav(
    text: str,
    out_path: Path,
    piper_model: Path,
    length_scale: float,
    sentence_silence: float,
) -> None:
    """Render `text` to `out_path` via Piper. Text is fed on stdin, WAV
    written to disk via `-f`. Piper auto-loads the `<model>.onnx.json`
    config sibling of `--model`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "piper",
        "-m", str(piper_model),
        "-f", str(out_path),
        "--length-scale", str(length_scale),
        "--sentence-silence", str(sentence_silence),
    ]
    proc = subprocess.run(
        argv, input=text, capture_output=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"piper exit {proc.returncode}: {proc.stderr[-300:]!r}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", type=Path, default=DEFAULT_YAML,
                   help=f"Patient YAML (default: {DEFAULT_YAML}).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--piper-model", type=Path, default=DEFAULT_PIPER_MODEL,
                   help=f"Piper voice .onnx (default: {DEFAULT_PIPER_MODEL}). "
                        "The .onnx.json config must sit next to it.")
    p.add_argument("--length-scale", type=float, default=DEFAULT_LENGTH_SCALE,
                   help=f"Piper speed knob, >1 slower (default: {DEFAULT_LENGTH_SCALE}).")
    p.add_argument("--sentence-silence", type=float, default=DEFAULT_SENTENCE_SILENCE,
                   help=f"Pause between sentences in seconds (default: {DEFAULT_SENTENCE_SILENCE}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the rendered sentences only; skip piper.")
    args = p.parse_args(argv)

    if not args.dry_run:
        if shutil.which("piper") is None:
            print("FATAL: piper not on PATH. Install: uv tool install piper-tts",
                  file=sys.stderr)
            return 2
        if not args.piper_model.exists():
            print(f"FATAL: piper model not found: {args.piper_model}\n"
                  f"  download with:\n"
                  f"  curl -fsSL -o {args.piper_model} \\\n"
                  f"      https://huggingface.co/rhasspy/piper-voices/resolve/main/"
                  f"en/en_US/lessac/medium/{args.piper_model.name}",
                  file=sys.stderr)
            return 2

    if not args.yaml.exists():
        print(f"FATAL: YAML not found: {args.yaml}", file=sys.stderr)
        return 2

    with args.yaml.open(encoding="utf-8") as f:
        table = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, str]] = {}
    for tool, builder in SENTENCE_BUILDERS.items():
        text = builder(table)
        wav_path = args.out_dir / f"{tool}.wav"
        manifest[tool] = {"text": text, "wav": str(wav_path)}
        print(f"--- {tool} ---")
        print(f"  text: {text}")
        if args.dry_run:
            continue
        render_wav(text, wav_path, args.piper_model,
                   args.length_scale, args.sentence_silence)
        print(f"  wav:  {wav_path} ({wav_path.stat().st_size // 1024} KiB)")

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest → {manifest_path}")

    patient_name = (table.get("patient") or {}).get("name", "<unknown>")
    print(f"\nDone. Patient={patient_name}, {len(manifest)} WAVs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
