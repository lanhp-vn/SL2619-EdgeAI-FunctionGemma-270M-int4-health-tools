"""Pydantic loader for the trimmed `data/health_table_v2.yaml`.

Schema: patient + appointments + emergency_contacts. NO vitals, NO notes,
NO medications — the dispenser demo's universe of intents is narrow (4
domain actions), so the legacy `src/gemma_tools/health_table.py` schema
(which requires vitals + notes) doesn't fit. New loader, no churn on
the legacy module.

API:

- `HealthTableV2`, `PatientV2`, `AppointmentV2`, `EmergencyContactV2`
  — frozen Pydantic models.
- `load_health_table_v2(path)` — read YAML, validate, return a frozen
  `HealthTableV2`. Raises `FileNotFoundError` or `pydantic.ValidationError`
  on schema violations.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _StrictBase(BaseModel):
    """Forbid unknown keys + freeze the instance.

    Same drift-catching discipline as `gemma_tools.functiongemma.tools._StrictBase`;
    a typo in the YAML surfaces at load time rather than silently dropping a
    key that downstream code reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class PatientV2(_StrictBase):
    name: str
    age: int = Field(ge=0, le=150)
    sex: str
    diagnoses: tuple[str, ...]


class AppointmentV2(_StrictBase):
    # ISO `YYYY-MM-DD` and 24-h `HH:MM` are validated downstream by
    # `wordform.date_to_words` / `time_to_words` at tool-dispatch time; the
    # schema just guarantees they are strings here.
    date: str
    time: str
    provider: str
    purpose: str
    location: str


class EmergencyContactV2(_StrictBase):
    name: str
    relation: str
    # Phone format (e.g. `"+1-555-0142"`) is validated by `wordform.phone_to_words`
    # at tool-dispatch time; here it must be a string.
    phone: str


class HealthTableV2(_StrictBase):
    """Top-level frozen container."""

    patient: PatientV2
    appointments: tuple[AppointmentV2, ...] = Field(min_length=1)
    emergency_contacts: tuple[EmergencyContactV2, ...] = Field(min_length=1)


def load_health_table_v2(path: Path) -> HealthTableV2:
    """Read YAML at `path`, validate, return a `HealthTableV2`.

    Raises:
        FileNotFoundError: if `path` does not exist.
        pydantic.ValidationError: on any schema violation; message names the
            offending field path.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return HealthTableV2.model_validate(raw)
