"""Compose per-candidate chat-formatted prompts for the SLM bench.

Template authority (literal token strings):
  Hugging Face `google/gemma-3-270m-it` chat_template.jinja
    Gemma 3 270M-IT: BOS=<bos>, turn markers <start_of_turn>/<end_of_turn>.
    The Jinja template silently concatenates a system-role message into
    the first user turn — our composer encodes that same effective shape.
  SmolLM2 (Hugging Face `HuggingFaceTB/SmolLM2-*-Instruct` chat_template):
    BOS=<|im_start|>, EOS=<|im_end|>; chat template resolves to
    bos+role+text+eos pattern at runtime.

System-prompt style: normative in docs/conventions/slm-system-prompt.md §4.
The directive-form template below is the single source of truth for SLM
prompts; changes land with test updates in the same commit.

Time injection: `compose_prompt` accepts `now: date` so date-injection tests
stay deterministic per docs/conventions/11-testing-verification.md §3.3
("inject time, never sleep"). The composer never calls `date.today()`.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Literal

import yaml

from gemma_tools.health_table import HealthTable

Candidate = Literal["gemma3", "smollm2"]

# Directive-form system template per docs/conventions/slm-system-prompt.md §4. Rules:
#   R-1 labeled directives, R-2 ground in YAML, R-3 refusal strings,
#   R-4 length cap, R-5 positive imperatives, R-6 fixed slots for date + YAML,
#   R-7 no persona, R-8 no few-shot.
# Total static-token budget ~90-130 (R-10).
_SYSTEM_TEMPLATE = """ROLE: health-records assistant on SL2619 edge device.
TASK: answer the user's question using ONLY facts in YAML.
RULES:
- quote YAML values verbatim (numbers, doses, times, names).
- if YAML lacks the answer: reply "not in record".
- never invent values, dates, medications, or food rules.
- refuse off-topic / social chat: reply "I answer questions from your health record only".
- never give medical advice; re-route: "consult your clinician".
FORMAT: 1-2 sentences. No preamble. No lists unless YAML has them.
DATE: {date}
YAML:
{yaml_block}"""


def render_vitals_table(health: HealthTable) -> str:
    """One-line, comma-separated rendering of the patient's vitals.

    Retained from Phase A for two reasons: (1) existing tests reference it,
    (2) downstream callers that want only the vitals line (e.g. a display
    widget) can still get the compact form. Prompts use the full YAML
    snapshot below — see `render_health_yaml`.
    """
    v = health.vitals
    return (
        f"HR {v.heart_rate_bpm} bpm, "
        f"BP {v.blood_pressure_systolic}/{v.blood_pressure_diastolic} mmHg, "
        f"SpO2 {v.spo2_percent}%, "
        f"T {v.body_temperature_c}C, "
        f"RR {v.respiratory_rate}"
    )


def _drop_none_and_empty(obj: object) -> object:
    """Recursively strip None values and empty tuples/lists/strings from a dict.

    The goal is to keep the YAML block in the prompt small and noise-free:
    optional fields that weren't set don't consume tokens. `False` and 0 are
    preserved (they are real, meaningful values).
    """
    if isinstance(obj, dict):
        result: dict[str, object] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            cleaned_v = _drop_none_and_empty(v)
            if cleaned_v is None or cleaned_v == () or cleaned_v == [] or cleaned_v == "":
                continue
            result[k] = cleaned_v
        return result
    if isinstance(obj, (list, tuple)):
        seq: list[object] = []
        for v in obj:
            cleaned_v = _drop_none_and_empty(v)
            if cleaned_v is None or cleaned_v == () or cleaned_v == [] or cleaned_v == "":
                continue
            seq.append(cleaned_v)
        return seq
    return obj


def render_health_yaml(health: HealthTable) -> str:
    """Dump the HealthTable as compact, readable YAML for the prompt body.

    The model sees this verbatim. Keys match the source fixture so it can
    cite "your metformin schedule is 08:00, 19:00" directly. Optional empty
    blocks are stripped to save prompt tokens (R-10 budget discipline).
    """
    cleaned = _drop_none_and_empty(asdict(health))
    # `default_flow_style=False` → block style (one key per line), easiest for
    # a 270M model to attend to. `sort_keys=False` → preserve insertion order
    # so the model sees vitals first, then the big-picture context.
    return yaml.safe_dump(
        cleaned, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).rstrip()


def render_system_prompt(health: HealthTable, now: date) -> str:
    """System-message body with date and YAML block slotted in."""
    return _SYSTEM_TEMPLATE.format(
        date=now.isoformat(),
        yaml_block=render_health_yaml(health),
    )


def compose_user_text(health: HealthTable, now: date, utterance: str) -> str:
    """Directive-form system prompt + blank line + user utterance, NO
    chat-template markers.

    This is the input shape the vendor `Gemma3Static.run()` runner expects:
    the runner adds its own `<start_of_turn>user/model` wrapping internally
    via its `_tokenize(role="user")` step. Passing a pre-templated string
    double-wraps the turn markers and corrupts tokenization. Use this variant
    whenever the downstream is the vendor runner (bench harness, on-board
    inference service); use `compose_prompt()` only when we own the
    tokenization.
    """
    return render_system_prompt(health, now) + "\n" + utterance


def compose_prompt(
    candidate: Candidate,
    utterance: str,
    health: HealthTable,
    now: date,
) -> str:
    """Return the full chat-formatted prompt for `candidate` (markers included).

    Both candidates wrap the system context + user utterance in a single
    user-role block followed by an empty assistant-role opener. The literal
    token strings differ (Gemma vs SmolLM2); hard-coding them here keeps the
    composer testable without loading a multi-hundred-MB tokenizer.

    Do NOT feed the output of this function to `Gemma3Static.run()` — see
    `compose_user_text()` for the runner-compatible variant.

    Raises:
        ValueError: if `candidate` is not one of the recognized literals.
    """
    full = compose_user_text(health, now, utterance)
    if candidate == "gemma3":
        return f"<start_of_turn>user\n{full}<end_of_turn>\n<start_of_turn>model\n"
    if candidate == "smollm2":
        return f"<|im_start|>user\n{full}<|im_end|>\n<|im_start|>assistant\n"
    raise ValueError(f"unknown candidate: {candidate!r}")
