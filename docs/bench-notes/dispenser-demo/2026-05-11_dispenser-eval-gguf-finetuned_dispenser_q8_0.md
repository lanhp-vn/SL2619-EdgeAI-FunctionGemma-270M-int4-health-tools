# Dispenser-demo iter-002 holdout eval — 2026-05-11

- **Seam**: `gguf-finetuned_dispenser_q8_0`
- **Holdout**: `data/dispenser_demo/dataset_v1/val.jsonl` (10 rows)
- **Pass-rate gate**: ≥ 90% per category (plan §9.1 step 1.6).

## Aggregate

**Overall pass rate**: 10/10 (100.0%).  **All categories ≥ 90 %**: YES.

| category | n | match | partial | mismatch | pass_rate | bar_pass |
|---|---|---|---|---|---|---|
| dispense | 2 | 2 | 0 | 0 | 100.0% | PASS |
| emergency_contact | 2 | 2 | 0 | 0 | 100.0% | PASS |
| next_appointment | 2 | 2 | 0 | 0 | 100.0% | PASS |
| out_of_scope_refusal | 2 | 2 | 0 | 0 | 100.0% | PASS |
| patient_profile | 2 | 2 | 0 | 0 | 100.0% | PASS |

