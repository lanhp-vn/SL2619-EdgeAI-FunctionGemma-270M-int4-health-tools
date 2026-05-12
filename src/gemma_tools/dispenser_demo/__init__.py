"""Dispenser-demo subpackage — 5-tool FunctionGemma intent model.

Public surface:

- `health_table_v2` — Pydantic loader for the trimmed `data/health_table_v2.yaml`
  (patient + appointments + emergency_contacts only; no vitals, no meds, no notes).
- `wordform` — pure functions for digit-to-word conversion (plan §5.2).
- `tools` — 5-tool registry (`get_patient_profile`, `get_next_appointment`,
  `get_emergency_contact`, `dispense_medication`, `refuse_out_of_scope`).
- `dataset` — JSONL validator + tool-boundary `*_words`-companion invariant.

Phase 1.2 status: `dispense_medication` returns `{"status": "dispensed"}` as a
stub. Phase 2 will plug in the real `pybleno` BleClient via an injected
context — see `docs/plans/dispenser-demo/plan.md` §6.1.
"""
