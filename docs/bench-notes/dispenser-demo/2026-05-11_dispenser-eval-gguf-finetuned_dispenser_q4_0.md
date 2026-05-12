# Dispenser-demo iter-002 holdout eval — 2026-05-11

- **Seam**: `gguf-finetuned_dispenser_q4_0`
- **Holdout**: `data/dispenser_demo/dataset_v1/val.jsonl` (10 rows)
- **Pass-rate gate**: ≥ 90% per category (plan §9.1 step 1.6).

## Aggregate

**Overall pass rate**: 3/10 (30.0%).  **All categories ≥ 90 %**: NO.

| category | n | match | partial | mismatch | pass_rate | bar_pass |
|---|---|---|---|---|---|---|
| dispense | 2 | 2 | 0 | 0 | 100.0% | PASS |
| emergency_contact | 2 | 1 | 0 | 1 | 50.0% | FAIL |
| next_appointment | 2 | 0 | 0 | 2 | 0.0% | FAIL |
| out_of_scope_refusal | 2 | 0 | 0 | 2 | 0.0% | FAIL |
| patient_profile | 2 | 0 | 0 | 2 | 0.0% | FAIL |


## Per-row failures

| row_id | category | gold | predicted | raw output (truncated) |
|---|---|---|---|---|
| pp-003 | patient_profile | `[{"name": "get_patient_profile", "arguments": {}}]` | `[]` | `<start_function_call>len_of_age_digits: []

<start_function_call>people_of_age_d…` |
| pp-004 | patient_profile | `[{"name": "get_patient_profile", "arguments": {}}]` | `[]` | `<start_function_call>len_result['get_patient_profile']{}<end_function_call>` |
| na-003 | next_appointment | `[{"name": "get_next_appointment", "arguments": {}}]` | `[]` | `<start_function_call>len_result:refuse_out_of_scope{}<end_function_call>` |
| na-004 | next_appointment | `[{"name": "get_next_appointment", "arguments": {}}]` | `[]` | `<start_function_call>people
<start_function_call>people
<start_function_call>pri…` |
| ec-003 | emergency_contact | `[{"name": "get_emergency_contact", "arguments": {}}]` | `[]` | `<start_function_call>len_emergency_contact{}<end_function_call>` |
| oo-002 | out_of_scope_refusal | `[{"name": "refuse_out_of_scope", "arguments": {"reason": "health_advice"}}]` | `[]` | `<start_function_call>len_out_of_scope_reason_text{"health_advice"}=-90000000000"…` |
| oo-007 | out_of_scope_refusal | `[{"name": "refuse_out_of_scope", "arguments": {"reason": "off_topic"}}]` | `[]` | `<start_function_call>RELATED HEADINGS: "Off_topic" "discussions" "politics" "spo…` |
