"""Drift gate between the live tool registry and the Distil `job_description.json`.

The model trains on the schema in `releases/.../distil/job_description.json`
and runs at inference against `gemma_tools.dispenser_demo.tools.as_function_declarations()`.
A drift in name, description, or parameters between these two sources is
silent — the trained model and the live registry will quietly disagree.

This test locks the alignment. Update one source and re-run; if the test
fails, update the other source until they match before shipping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_tools.dispenser_demo.tools import as_function_declarations

_REPO = Path(__file__).resolve().parents[2]
_JD_PATH = (
    _REPO
    / "releases"
    / "functiongemma-270m"
    / "002-dispenser-demo"
    / "distil"
    / "job_description.json"
)


def _jd_tools_by_name() -> dict[str, dict[str, object]]:
    raw = json.loads(_JD_PATH.read_text(encoding="utf-8"))
    return {t["function"]["name"]: t["function"] for t in raw["tools"]}


def _registry_tools_by_name() -> dict[str, dict[str, object]]:
    return {t["function"]["name"]: t["function"] for t in as_function_declarations()}


def test_tool_name_sets_match() -> None:
    """The set of tool names must agree between job_description.json and the
    live registry — neither source may carry a tool the other doesn't.
    """
    jd_names = set(_jd_tools_by_name())
    reg_names = set(_registry_tools_by_name())
    assert jd_names == reg_names, (
        f"tool name drift: only in job_description.json = "
        f"{sorted(jd_names - reg_names)!r}; only in registry = "
        f"{sorted(reg_names - jd_names)!r}"
    )


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_patient_profile",
        "get_next_appointment",
        "get_emergency_contact",
        "dispense_medication",
        "refuse_out_of_scope",
    ],
)
def test_tool_description_matches(tool_name: str) -> None:
    """Per-tool description equality.

    The training prompt's tool description and the inference-time schema's
    tool description must match byte-for-byte; otherwise the model trains on
    one contract and runs against another. Iter-001 didn't have this gate;
    we add it now to lock the surface as `job_description.json` grows.
    """
    jd = _jd_tools_by_name()[tool_name]
    reg = _registry_tools_by_name()[tool_name]
    assert jd["description"] == reg["description"], (
        f"{tool_name}: description drift\n"
        f"  job_description.json: {jd['description']!r}\n"
        f"  registry            : {reg['description']!r}"
    )


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_patient_profile",
        "get_next_appointment",
        "get_emergency_contact",
        "dispense_medication",
        "refuse_out_of_scope",
    ],
)
def test_tool_parameters_match(tool_name: str) -> None:
    """Per-tool parameters-schema equality.

    For zero-arg tools the schema is trivial (`{}`); for `refuse_out_of_scope`
    the `reason` enum must be exactly aligned. A drift here would mean the
    judge marks predictions BAD because the model emits a parameter the
    registry rejects (or vice versa).
    """
    jd = _jd_tools_by_name()[tool_name]
    reg = _registry_tools_by_name()[tool_name]
    assert jd["parameters"] == reg["parameters"], (
        f"{tool_name}: parameters drift\n"
        f"  job_description.json: {jd['parameters']!r}\n"
        f"  registry            : {reg['parameters']!r}"
    )
