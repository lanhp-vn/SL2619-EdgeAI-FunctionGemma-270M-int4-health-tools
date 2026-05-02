"""Smoke-test for the `functiongemma` optional dependency extra.

Each third-party import is gated with `pytest.importorskip` so the suite
stays green for users who installed the base or `[dev]` extra only. When
`uv sync --extra functiongemma` has run, all parametrized rows should pass —
that's the M1 acceptance signal in `docs/plans/FunctionGemma/README.md`.
"""

from __future__ import annotations

import pytest

# Modules pulled in by [project.optional-dependencies].functiongemma.
# `gguf` and `sentencepiece` are needed by `convert_hf_to_gguf.py` (§15.2).
_FG_EXTRA_MODULES = [
    "transformers",
    "torch",
    "accelerate",
    "huggingface_hub",
    "pydantic",
    "jsonschema",
    "sentencepiece",
    "gguf",
    "llama_cpp",
]


@pytest.mark.parametrize("module_name", _FG_EXTRA_MODULES)
def test_functiongemma_extra_importable(module_name: str) -> None:
    pytest.importorskip(
        module_name,
        reason=f"functiongemma extra not installed; run `uv sync --extra functiongemma` to enable {module_name}",
    )


def test_existing_package_still_imports() -> None:
    # Guard against an extra-install accidentally shadowing the in-repo package.
    from gemma_tools import health_table

    assert hasattr(health_table, "load_health_table")
