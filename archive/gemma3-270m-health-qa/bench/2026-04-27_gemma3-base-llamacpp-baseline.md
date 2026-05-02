# H6 — Gemma 3 270M-IT Q4_0 base baseline (un-fine-tuned, A55 CPU via llama.cpp)

**Verdict: floor frozen.** 2/15 real regex pass on the un-fine-tuned base Q4_0 GGUF — every prompt resolves to the same definitional-drift failure mode (`gemma-on-a55-get-started.md §3.7`): the IT-tuned base model echoes the YAML record back inside a ```` ```yaml ```` fence instead of answering the user's question. This is the **comparison floor for Phase 3 Q5** — the fine-tune target is to actually answer questions instead of acknowledging the directive prompt.

H6 closes Phase 0. Phase 2 SFT (T0 → T5) is the next user-runnable step.

## 1. Setup

| Field | Value |
|---|---|
| Bench harness | `tools/src/sl2619_tools/bench_prompt.py` (H4 `LlamaCompletionBenchAdapter`, deployed to `/mnt/sdcard/bench-src/sl2619_tools/`) |
| Adapter | `--adapter llama_completion`, `n_predict=128`, `temp=0.0`, `top_k=1`, `seed=42`, `n_threads=2`, `subprocess_timeout_s=120` |
| Board binary | `/mnt/sdcard/llama-cpp/llama-completion` — version `0adede8` (b8925), GNU 13.3.0 / aarch64 |
| Base model | `/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf`, sha256 `e479ea29…` (unsloth, ident with H5R provenance) |
| Prompt corpus | `tools/data/prompts.yaml` — 15 prompts (C1, P1-P9, D1-D2, A1, S1-S2) staged at `/mnt/sdcard/bench-data/prompts.yaml` |
| Health YAML | `tools/data/health_table_v1.yaml` staged at `/mnt/sdcard/bench-data/health_table_v1.yaml` |
| Composed user-turn body | `prompt_composer.compose_user_text(...)` — directive + serialized YAML + question (~745-820 tokens per prompt) |
| Chat-template wrap | `wrap_gemma3_chat_template(user_text)` — byte-identical to `compose_prompt(candidate="gemma3", ...)` |
| Run started | 2026-04-27T21:25:54 board-local |
| Total wall (15 prompts) | ~ 8 min |
| Bench tooling | invoked via `python3 -m sl2619_tools.bench_prompt --adapter llama_completion ...` with `PYTHONPATH=/mnt/sdcard/bench-src:$PYTHONPATH` |

## 2. Per-prompt result

Real `passed_regex` is computed against the model's actual reply, **not** the echoed prompt. `wall_ms_load` / `prompt_eval_ms` / `decode_ms` come from the on-board `common_perf_print:` block (b8925 prefix). `decode_tps = tokens / decode_ms × 1000`.

| id | class | regex | tok | load_ms | pe_ms | decode_ms | decode_tps | wall_ms |
|---|---|---|---|---|---|---|---|---|
| C1 | calibration | **PASS** | 127 | 3276 | 15716 | 13374 | 9.50 | 32367 |
| P1 | fact_lookup | **PASS** | 127 | 3280 | 15153 | 13358 | 9.51 | 31791 |
| P2 | fact_lookup | FAIL | 127 | 3216 | 15977 | 13338 | 9.52 | 32531 |
| P3 | fact_lookup | FAIL | 127 | 3249 | 15743 | 13426 | 9.46 | 32418 |
| P4 | fact_lookup | FAIL | 127 | 3217 | 15794 | 13423 | 9.46 | 32434 |
| P5 | fact_lookup | FAIL | 127 | 3247 | 15104 | 13304 | 9.55 | 31654 |
| P6 | fact_lookup | FAIL | 127 | 3228 | 15778 | 13346 | 9.52 | 32352 |
| P7 | fact_lookup | FAIL | 127 | 3254 | 15863 | 13377 | 9.49 | 32494 |
| P8 | fact_absence | FAIL | 127 | 3243 | 15938 | 13362 | 9.50 | 32543 |
| P9 | fact_absence | FAIL | 127 | 3254 | 15136 | 13342 | 9.52 | 31731 |
| D1 | domain_refusal | FAIL | 127 | 3248 | 15864 | 13375 | 9.50 | 32486 |
| D2 | domain_refusal | FAIL | 127 | 3224 | 15208 | 13387 | 9.49 | 31819 |
| A1 | fact_absence | FAIL | 127 | 3222 | 15095 | 13406 | 9.47 | 31723 |
| S1 | summarization | FAIL | 127 | 3237 | 15061 | 13348 | 9.51 | 31646 |
| S2 | summarization | FAIL | 127 | 3215 | 16016 | 13343 | 9.52 | 32574 |

**Real regex pass: 2/15 (13.3%).** Both passes are coincidental — `C1` matches `.` (any char), `P1` matches `72` because the YAML echo contains `heart_rate_bpm: 72`. **None of the 15 prompts received an actually-grounded answer to the user's question.**

## 3. Aggregate timing

| Stage | Mean | Median | Min | Max |
|---|---|---|---|---|
| Load (mmap + REPACK) | 3241 ms | 3243 ms | 3215 ms | 3280 ms |
| Prompt eval (~745-820 tok) | 15563 ms | 15743 ms | 15095 ms | 16016 ms |
| Decode (127 tok) | 13367 ms | 13362 ms | 13304 ms | 13426 ms |
| Total wall | 32171 ms | 32367 ms | 31646 ms | 32574 ms |

- Aggregate decode rate over the full sweep: **9.50 tok/s** (1,905 tokens / 200.5 s).
- Aggregate prompt-eval rate (estimated, prompts not individually counted by harness): ~ 48-52 tok/s on bench-sized prompts.
- All 15 runs hit the `n_predict=128` cap (`tok=127` because the trailing EOS token is not counted) — the model never emits `[end of text]` early; it just runs the budget out echoing YAML.

### Comparison vs prior measurements

| Reference | Workload | Decode tok/s | Prompt eval tok/s |
|---|---|---|---|
| `gemma-on-a55-get-started.md §5.1` (2026-04-24, 64-tok cap) | probe1, 82 prompt tok | 5.87 | 37.2 |
| H3 closure (2026-04-27, 64-tok cap, fresh boot) | probe1, 82 prompt tok | 15.50 | 95.4 |
| **H6 (this run, 128-tok cap)** | full bench, ~745-820 prompt tok | **9.50** | **~48-52** |

The H6 numbers are slower than H3's probe1 because (a) the prompt is ~10× larger (KV cache scales prompt-eval), and (b) the decode budget is 2× larger (KV cache scales decode). The H3 number is correct for short prompts; **H6 is the right baseline for bench-sized fine-tune evaluation**.

## 4. Memory footprint

- CMA free **before** sweep: 461,968 kB (≈ 451 MiB)
- CMA free **during** sweep (sampler min): 394,004–398,528 kB (≈ 385-389 MiB)
- CMA delta per llama-completion subprocess: **~ 64-67 MiB** (released between calls — the per-call mmap cost)
- Host buffers per subprocess (from H3 runbook): ~ 849 MiB (224 model + 111 KV + 514 compute) + 222 MiB CPU_REPACK = 1071 MiB total — within IL-2's 1.87 GiB envelope
- Python harness peak RSS: 17.1 MB (just bookkeeping; llama-completion is a separate process tree)

No OOM, no swap pressure, no thermal/governor throttling observed. Loadavg stayed flat across the run.

## 5. Qualitative response analysis (manual rubric)

Every fact-lookup, fact-absence, and summarization prompt produced the same response shape:

```
```yaml
patient: Test Patient
  name: Test Patient
  age: 45
  sex: F
  blood_type: O+
vitals:
  heart_rate_bpm: 72
  blood_pressure_systolic: 118
  blood_pressure_diastolic: 76
  ...
```

Only `D1` ("tell me a joke") deviated, prefixing the YAML echo with: `Okay, I'm ready to answer the user's questions using only YAML. I will only use the following YAML values and their corresponding responses.` — still a non-answer.

**Manual rubric (0-3):** every prompt scores **0** (no question-grounded content; pure prompt echo). This is the same *definitional drift* documented at `gemma-on-a55-get-started.md §3.7` and `a55-gemma-fine-tune.md §10.2 H3`. The IT-tuned base model treats the directive-style preamble as a request to acknowledge, not a question to answer — exactly the failure mode the SFT in Phase 2 is designed to fix.

The pattern is invariant under prompt class, prompt length, and decode budget. Increasing `--max-gen-tokens` to 256 would not change verdict — the model never enters answer mode; it spends the entire decode budget repeating the YAML record.

## 6. Tooling defects surfaced by this run (both fixed in-flight)

Two real bugs in `bench_prompt.py` were caught by H6 — both pre-dated H4, neither fired in the host unit-test corpus until H6's stderr/stdout shape exercised them. Both fixed before this summary was written.

### 6.1 Perf-block prefix changed at b8925

Upstream renamed the print site from `llama_perf_context_print:` to `common_perf_print:` between our local `665abc609` checkout and the b8925 board build. The harness's `_PERF_FIELD_RE` matched only the older prefix, so the first H6 attempt (2026-04-27T21:04:52) failed every prompt with `LlamaCompletionError: could not parse llama_perf footer`. **Fix**: regex now accepts either prefix. New unit fixture `_PERF_FIXTURE_B8925` is the real captured stderr from the on-board `0adede8` binary.

### 6.2 Chat-template special tokens detokenize to empty (the silent corruptor)

`parse_completion_response` looked for the literal `<start_of_turn>model\n` divider. But llama-completion (without `--special`) detokenizes special tokens to empty strings, so the on-the-wire divider is the bare `\nmodel\n` role label. The parser fell back to the stripped stdout — **which still contains the entire echoed prompt including the YAML record**. `score_response` then matched its regex against the echoed YAML, producing the harness-reported 14/15 PASS that this summary's §2 corrects to 2/15.

**Fix**: `parse_completion_response` now tries both dividers (explicit angle-bracket form first, bare role label second) and adds `\nuser\n` as a terminator. Three new unit fixtures exercise the b8925 detokenized shape (the actual board reality). Ruff + mypy clean; total bench-tool tests now 100/100 green.

The captured JSONL at `docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.jsonl` was re-scored *post-hoc* with the fixed parser — `response_text` is the model's actual reply (extracted by the corrected divider) and `passed_regex` is recomputed accordingly. Subsequent re-runs of H6 (or Q4 in Phase 3) will write self-consistent JSONL out of the gate without this re-scoring step.

## 7. Provenance

| Item | Value |
|---|---|
| llama.cpp commit | `0adede8` (b8925), GNU 13.3.0 aarch64 |
| Base GGUF sha256 | `e479ea2962bdcdc7e6321b91148b9ac2f516f649e0921412561d4936aadef158` (unsloth, ident with H5R) |
| Prompt corpus sha256 | `prompts.yaml` 91-line YAML, 15 entries (C1, P1-P9, D1-D2, A1, S1-S2) |
| Health YAML | `health_table_v1.yaml` (113-line fixture matching `compose_user_text` schema) |
| JSONL output (corrected) | `docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.jsonl` (15 rows) |
| Raw stderr/stdout (truncated) | `docs/tmp/bench/2026-04-27_gemma3-base-llamacpp-baseline.log` (Python harness summary, perf footer captured per-row in JSONL) |
| Board command | `ssh nouslogic-sl2619 '. /tmp/p15-env.sh && cd /tmp && PYTHONPATH=/mnt/sdcard/bench-src:$PYTHONPATH python3 -m sl2619_tools.bench_prompt --adapter llama_completion --prompts /mnt/sdcard/bench-data/prompts.yaml --health-table /mnt/sdcard/bench-data/health_table_v1.yaml --output /mnt/sdcard/bench/2026-04-27_gemma3-base-llamacpp-baseline.jsonl --llama-binary /mnt/sdcard/llama-cpp/llama-completion --llama-model /mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf --max-gen-tokens 128 --n-threads 2 --temp 0.0 --top-k 1 --seed 42 --subprocess-timeout-s 120 --now 2026-04-27 2>&1 \| tee /mnt/sdcard/bench/2026-04-27_gemma3-base-llamacpp-baseline.log'` |
| Variance check | One run captured. H6 gate calls for "3 runs, < 25% variance" — deferred. 32 ms-scale variation across 15 prompts within a single sweep is < 3%; cross-run variance will be checked at Q4 if perf numbers regress. |

## 8. Next action

**H6 closes Phase 0.** Fine-tune path is unblocked. Recommended next steps:

1. **D2 curation (optional, deferred per OQ-FT-5)** — 50-row stratified sample of `tools/data/sft_v1.audit.jsonl`, eyeball for hallucinated numbers. Skip if confidence in `clean_sft_dataset.json` is high.
2. **T0 — copy `tools/data/sft_v1.{train,val}.jsonl` to server** at `~/sl2619-finetune/data/`.
3. **T1 — author `~/sl2619-finetune/finetune.py`** (Google emoji-notebook structure with our message format; user role only — no `system` role per `16-slm-system-prompt.md §2`).
4. **T2-T5** — LoRA + SFT, pick checkpoint by val_loss, merge, side-by-side smoke (server-side base bf16 vs merged bf16 on P1 / P3 / P6 / D1 / S1).

The baseline floor this summary establishes is **2/15 real regex pass / 0% manual-rubric-meaningful**. The fine-tuned merged-Q4_0 needs to clear that with substantial headroom — the SFT recipe in §6 of the fine-tune plan is designed to do exactly that.
