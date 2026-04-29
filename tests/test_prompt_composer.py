"""Tests for gemma_tools.prompt_composer.

Template authority (literal token strings):
  references/Synaptics/torq-examples/gemma3/src/runner.py:155-178
  references/Synaptics/torq-tools/src/torq/models/smollm2/_inference.py:193-195
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from gemma_tools.health_table import (
    HealthTable,
    Patient,
    Vitals,
    load_health_table,
)
from gemma_tools.prompt_composer import (
    Candidate,
    compose_prompt,
    compose_user_text,
    render_health_yaml,
    render_system_prompt,
)

_REPO = Path(__file__).resolve().parents[1]
CANONICAL_FIXTURE = _REPO / "data" / "health_table_v1.yaml"
PROMPTS_FIXTURE = _REPO / "data" / "prompts.yaml"

# Single canonical HealthTable instance reused across cases — the composer is
# pure-functional, so sharing is safe and keeps test runtime under a tick.
_HEALTH = HealthTable(
    patient=Patient(name="Test Patient", age=45),
    vitals=Vitals(
        heart_rate_bpm=72,
        blood_pressure_systolic=118,
        blood_pressure_diastolic=76,
        spo2_percent=98,
        body_temperature_c=36.7,
        respiratory_rate=16,
    ),
    notes=("Vitals within nominal range", "No medication interactions flagged"),
)
_DEFAULT_NOW = date(2026, 4, 23)


# | candidate  | required_substring                                       | desc                                  |
@pytest.mark.parametrize(
    ("candidate", "required_substring", "desc"),
    [
        ("gemma3",  "<start_of_turn>user\n",                  "Gemma 3 user-role marker present"),
        ("gemma3",  "<end_of_turn>\n<start_of_turn>model\n",  "Gemma 3 end+model marker present"),
        ("smollm2", "<|im_start|>user\n",                     "SmolLM2 user-role marker present"),
        ("smollm2", "<|im_end|>\n<|im_start|>assistant\n",    "SmolLM2 end+assistant marker"),
    ],
)
def test_compose_prompt_emits_per_candidate_markers(
    candidate: Candidate,
    required_substring: str,
    desc: str,
) -> None:
    out = compose_prompt(candidate, "say hi", _HEALTH, _DEFAULT_NOW)
    assert required_substring in out, desc


# | candidate  | now              | required_substring | desc                                  |
@pytest.mark.parametrize(
    ("candidate", "now", "required_substring", "desc"),
    [
        ("gemma3",  date(2026, 4, 23), "2026-04-23", "Gemma 3 ISO date in system prompt"),
        ("smollm2", date(2026, 4, 23), "2026-04-23", "SmolLM2 ISO date in system prompt"),
        ("gemma3",  date(2024, 1, 1),  "2024-01-01", "Gemma 3 different date"),
        ("smollm2", date(2024, 1, 1),  "2024-01-01", "SmolLM2 different date"),
    ],
)
def test_compose_prompt_injects_date(
    candidate: Candidate,
    now: date,
    required_substring: str,
    desc: str,
) -> None:
    out = compose_prompt(candidate, "say hi", _HEALTH, now)
    assert required_substring in out, desc


# | candidate  | required_substring | desc                                          |
@pytest.mark.parametrize(
    ("candidate", "required_substring", "desc"),
    [
        ("gemma3",  "72",  "Gemma 3 prompt contains HR=72 from fixture"),
        ("gemma3",  "118", "Gemma 3 prompt contains BP-sys=118"),
        ("smollm2", "76",  "SmolLM2 prompt contains BP-dia=76"),
        ("smollm2", "98",  "SmolLM2 prompt contains SpO2=98"),
    ],
)
def test_compose_prompt_injects_vitals(
    candidate: Candidate,
    required_substring: str,
    desc: str,
) -> None:
    out = compose_prompt(candidate, "say hi", _HEALTH, _DEFAULT_NOW)
    assert required_substring in out, desc


# | candidate  | utterance              | desc                                          |
@pytest.mark.parametrize(
    ("candidate", "utterance", "desc"),
    [
        ("gemma3",  "what time is it?",    "Gemma 3 utterance follows system block"),
        ("smollm2", "what date is today?", "SmolLM2 utterance follows system block"),
    ],
)
def test_compose_prompt_user_slot_contains_utterance_after_system(
    candidate: Candidate,
    utterance: str,
    desc: str,
) -> None:
    out = compose_prompt(candidate, utterance, _HEALTH, _DEFAULT_NOW)
    # Directive-form system prompt per 16-slm-system-prompt.md §4 — ROLE
    # label is the stable marker across template revisions.
    sys_marker = "ROLE: health-records assistant"
    sys_idx = out.find(sys_marker)
    utt_idx = out.find(utterance)
    assert sys_idx >= 0, f"{desc}: system block missing"
    assert utt_idx >= 0, f"{desc}: utterance missing"
    assert sys_idx < utt_idx, f"{desc}: utterance must appear after system block"


# | bad_candidate | desc                                       |
@pytest.mark.parametrize(
    ("bad_candidate", "desc"),
    [
        ("foo", "rejects unknown candidate string"),
        ("",    "rejects empty candidate string"),
    ],
)
def test_compose_prompt_rejects_unknown_candidate(
    bad_candidate: str,
    desc: str,
) -> None:
    # `compose_prompt` is typed as Literal["gemma3","smollm2"], but at runtime
    # we accept arbitrary strings to validate the defensive ValueError branch.
    # The cast-via-Any pattern is intentional: production callers go through
    # mypy and Literal narrowing; this test exercises the runtime guard a
    # malformed YAML or operator typo would hit.
    with pytest.raises(ValueError, match="unknown candidate"):
        compose_prompt(bad_candidate, "say hi", _HEALTH, _DEFAULT_NOW)  # type: ignore[arg-type]
    assert desc


# | source_path        | desc                                                  |
@pytest.mark.parametrize(
    ("source_path", "desc"),
    [
        (PROMPTS_FIXTURE,  "committed tools/data/prompts.yaml entries are well-formed"),
        ("inline_minimal", "inline doc with the same schema parses identically"),
    ],
)
def test_prompts_yaml_parses(source_path: object, desc: str, tmp_path: Path) -> None:
    if source_path == "inline_minimal":
        doc = {
            "prompts": [
                {"id": f"X{i}", "class": "c", "text": "t",
                 "pass_pattern": ".", "pattern_flags": ""}
                for i in range(6)
            ]
        }
        path = tmp_path / "p.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
    else:
        assert isinstance(source_path, Path), desc
        path = source_path

    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict), desc
    prompts = raw.get("prompts")
    assert isinstance(prompts, list), desc
    # Don't pin exact count — prompts.yaml grows as we add bench coverage.
    # Schema is what matters: every entry has the 5 load-bearing keys.
    assert len(prompts) >= 6, desc
    required_keys = {"id", "class", "text", "pass_pattern", "pattern_flags"}
    for entry in prompts:
        assert isinstance(entry, dict), desc
        assert required_keys <= entry.keys(), f"{desc}: entry missing keys: {entry}"


def test_load_health_table_then_compose_smoke() -> None:
    """End-to-end smoke: canonical fixture loads + composer renders both
    candidate strings non-empty. Two independent assertions = 2 cases.
    """
    table = load_health_table(CANONICAL_FIXTURE)
    g = compose_prompt("gemma3", "say hi", table, _DEFAULT_NOW)
    s = compose_prompt("smollm2", "say hi", table, _DEFAULT_NOW)
    assert "say hi" in g, "Gemma 3 composer round-trip includes utterance"
    assert "say hi" in s, "SmolLM2 composer round-trip includes utterance"


# --------------------------------------------------------------------------
# Expanded schema coverage (2026-04-24 pivot): the composer now dumps the
# full HealthTable as YAML into the prompt body per 16-slm-system-prompt.md
# §4 R-6. Cases below verify the expanded blocks appear in the composed
# output so the SLM can attend to them.
# --------------------------------------------------------------------------

# | required_substring  | desc                                                  |
@pytest.mark.parametrize(
    ("required_substring", "desc"),
    [
        ("Lisinopril",  "Lisinopril medication name present in rendered YAML"),
        ("Penicillin",  "Penicillin allergy present"),
        ("Hypertension", "Hypertension condition present"),
        ("2026-05-06",   "upcoming appointment date present"),
        ("Jane Doe",     "emergency contact present"),
        ("low sodium",   "dietary restriction present"),
    ],
)
def test_render_health_yaml_contains_expanded_blocks(
    required_substring: str, desc: str
) -> None:
    table = load_health_table(CANONICAL_FIXTURE)
    rendered = render_health_yaml(table)
    assert required_substring in rendered, desc


# | required_label | desc                                                    |
@pytest.mark.parametrize(
    ("required_label", "desc"),
    [
        ("ROLE:",   "directive R-1 ROLE label present"),
        ("TASK:",   "directive R-1 TASK label present"),
        ("RULES:",  "directive R-1 RULES label present"),
        ("FORMAT:", "directive R-1 FORMAT label present"),
        ("DATE:",   "directive R-6 DATE slot present"),
        ("YAML:",   "directive R-6 YAML slot present"),
        ("not in record", "R-2 fallback refusal string present"),
        ("health record only", "R-3 off-topic refusal string present"),
    ],
)
def test_render_system_prompt_matches_convention_shape(
    required_label: str, desc: str
) -> None:
    table = load_health_table(CANONICAL_FIXTURE)
    sys = render_system_prompt(table, _DEFAULT_NOW)
    assert required_label in sys, desc


def test_render_health_yaml_strips_empty_optional_fields() -> None:
    """Minimal fixture (no conditions / meds / etc.) should render to a YAML
    block that does NOT include empty blocks — saves prompt tokens per §3 R-10.
    """
    minimal = HealthTable(
        patient=Patient(name="P", age=30),
        vitals=Vitals(
            heart_rate_bpm=70, blood_pressure_systolic=110,
            blood_pressure_diastolic=70, spo2_percent=99,
            body_temperature_c=36.5, respiratory_rate=14,
        ),
        notes=("baseline",),
    )
    rendered = render_health_yaml(minimal)
    assert "conditions" not in rendered, "empty conditions should be stripped"
    assert "allergies" not in rendered, "empty allergies should be stripped"
    assert "medications" not in rendered, "empty medications should be stripped"
    # Required blocks survive.
    assert "patient:" in rendered, "patient block still present"
    assert "vitals:" in rendered, "vitals block still present"


def test_render_health_yaml_preserves_medication_interactions() -> None:
    """Medications with non-empty avoid_foods / avoid_drugs must surface in
    the YAML block — the P5 grapefruit prompt depends on this."""
    table = load_health_table(CANONICAL_FIXTURE)
    rendered = render_health_yaml(table)
    assert "Grapefruit" in rendered, "atorvastatin-grapefruit interaction visible"
    assert "avoid_foods" in rendered, "avoid_foods key preserved in output"


# --------------------------------------------------------------------------
# compose_user_text — the vendor-runner-compatible variant (no chat markers).
# Consumed by the on-board bench harness (bench_prompt.py). A template-marker
# leak here produces double-wrapped tokens at Gemma3Static input and
# silently degrades retrieval.
# --------------------------------------------------------------------------


# | forbidden_marker                       | desc                                        |
@pytest.mark.parametrize(
    ("forbidden_marker", "desc"),
    [
        ("<start_of_turn>",              "no Gemma user/model markers"),
        ("<end_of_turn>",                "no Gemma end-turn marker"),
        ("<bos>",                        "no explicit BOS (tokenizer adds it)"),
    ],
)
def test_compose_user_text_has_no_chat_template_markers(
    forbidden_marker: str, desc: str
) -> None:
    out = compose_user_text(_HEALTH, _DEFAULT_NOW, "what is my heart rate?")
    assert forbidden_marker not in out, desc


# | required_substring | desc                                             |
@pytest.mark.parametrize(
    ("required_substring", "desc"),
    [
        ("ROLE:",                          "system prompt ROLE directive leads"),
        ("DATE: 2026-04-23",               "date-injected ISO slot present"),
        ("YAML:",                          "YAML slot marker present"),
        ("what is my heart rate?",         "utterance appears verbatim at end"),
    ],
)
def test_compose_user_text_includes_system_directives_and_utterance(
    required_substring: str, desc: str
) -> None:
    out = compose_user_text(_HEALTH, _DEFAULT_NOW, "what is my heart rate?")
    assert required_substring in out, desc


def test_compose_user_text_utterance_is_last() -> None:
    """The utterance sits AFTER the system directives + YAML; Gemma's user
    turn is "context first, then question" per `16-slm-system-prompt.md §4`."""
    utterance = "BOUNDARY_UTTERANCE_MARKER"
    out = compose_user_text(_HEALTH, _DEFAULT_NOW, utterance)
    sys = render_system_prompt(_HEALTH, _DEFAULT_NOW)
    assert out.endswith(utterance), "utterance must terminate the composed body"
    assert out.startswith(sys), "system directives must precede the utterance"


def test_compose_prompt_is_compose_user_text_plus_markers() -> None:
    """Wrapping contract: compose_prompt == markers(compose_user_text(...))."""
    utterance = "what is my blood pressure?"
    inner = compose_user_text(_HEALTH, _DEFAULT_NOW, utterance)
    gemma = compose_prompt("gemma3", utterance, _HEALTH, _DEFAULT_NOW)
    assert gemma == f"<start_of_turn>user\n{inner}<end_of_turn>\n<start_of_turn>model\n", (
        "Gemma compose_prompt = markers + compose_user_text + end markers"
    )
