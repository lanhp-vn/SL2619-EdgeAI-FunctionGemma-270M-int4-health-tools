# Phase B Bench Summary — Gemma 3 270M-IT on SL2619 NPU

> **Date**: 2026-04-24
> **Model**: `Synaptics/gemma-3-270m-it` (vendor-compiled bf16 VMFB), `:v1.5` runtime tag
> **Artifacts on board**: `/mnt/sdcard/models/gemma-3-270m-it/{model.vmfb (516 MiB), token_embeddings.npy (320 MiB), tokenizer.json, config.json}`
> **Driver**: `torq.runtime` + `iree.runtime` torq HAL backend, single-process design (see §Working pathways below)
> **Verdict**: **G_QUALITY FAILS → Phase C (SmolLM2-360M) triggered per plan §9**
> **Owner**: record-of-truth for the Phase B phase-plan-§9-D1 deliverable; cited by `backlogs.md §1.19`.

---

## 1. Measurement methodology

- One process, one warmup, 18 decode runs (6 prompts × 3 runs) — required because **cross-process CMA fragmentation + `syna_npu` driver state retention** killed multi-process cycling (see Working pathways §W3 + W5 below).
- Greedy sampling (`temperature=0.0`) — deterministic outputs, confirmed by identical text across all 3 runs per prompt.
- System prompt: 341 chars, 130 tokens — includes current date/time + mocked `health_table_v1` vitals (HR/BP/SpO2/temp/RR + notes).
- User prompts from plan §5.1 verbatim.
- Bench harness: `/mnt/sdcard/models/gemma-3-270m-it/bench_prompt.py` (staged 2026-04-24; patched twice during this session — see §W1-W2).
- Metrics captured: TTFT (manually timed around `run_stream` — vendor's post-loop stats don't survive early break on `MAX_NEW_TOKENS` cap), total infer, tokens-generated, tok/s, CmaFree/MemAvailable deltas.

---

## 2. Raw results — all 18 runs

| Prompt | Run | Tag | TTFT (ms) | Total (ms) | Tokens | Tok/s | Output |
|---|---|---|---|---|---|---|---|
| C1 | 1 | cold | 5611 | 29218 | 39 | 1.65 | `"Hello! I am Gemma, your AI assistant. I am ready to help you with your medical information and provide you with information. Please provide your medical history and any relevant medical conditions."` |
| C1 | 2 | warm | 5566 | 29138 | 39 | 1.65 | (identical) |
| C1 | 3 | warm | 5572 | 29156 | 39 | 1.65 | (identical) |
| P1 | 1 | cold | 6808 | 17358 | 18 | 1.71 | `"The current time is 2026-04-24."` |
| P1 | 2 | warm | 6805 | 17356 | 18 | 1.71 | (identical) |
| P1 | 3 | warm | 6806 | 17356 | 18 | 1.71 | (identical) |
| P2 | 1 | cold | 6808 | 12401 | 10 | 1.79 | `"Today is April 24th."` |
| P2 | 2 | warm | 6805 | 12410 | 10 | 1.78 | (identical) |
| P2 | 3 | warm | 6806 | 12393 | 10 | 1.79 | (identical) |
| P3 | 1 | cold | 5578 | 14274 | 15 | 1.72 | `"Okay, I'm ready to help you with your test."` |
| P3 | 2 | warm | 5572 | 14265 | 15 | 1.73 | (identical) |
| P3 | 3 | warm | 5571 | 14262 | 15 | 1.73 | (identical) |
| P4 | 1 | cold | 9908 | 24194 | 24 | 1.68 | `"Your heart rate is 72 bpm and your blood pressure is 118/76 mmHg."` |
| P4 | 2 | warm | 9896 | 24177 | 24 | 1.68 | (identical) |
| P4 | 3 | warm | 9899 | 24184 | 24 | 1.68 | (identical) |
| P5 | 1 | cold | 7435 | 37284 | 49 | 1.64 | `"I am a helpful AI assistant. I am currently in good health and have a normal pulse and respiratory rate. I am also able to perform basic medical procedures and take medication. I am ready to help you with any questions or tasks."` |
| P5 | 2 | warm | 7428 | 37253 | 49 | 1.64 | (identical) |
| P5 | 3 | warm | 7430 | 37259 | 49 | 1.64 | (identical) |

Raw artifacts on board: `/mnt/sdcard/bench/2026-04-24_gemma3/{<PID>_run{1,2,3}.txt, <PID>_metrics.json, all_metrics.json}`.

---

## 3. Quality scoring per §6.2 rubric

0 = refused / off-topic / hallucinated · 1 = touches topic, misses fact · 2 = correct but verbose/disclaimed · 3 = concise + correct

| Prompt | Score | Rationale |
|---|---|---|
| C1 | 2 (calibration, not in gate) | Greets but rambles into medical-assistant framing; violates "1-2 sentences" system-prompt instruction. |
| P1 "what time is it?" | **1** | Returns *date* (`2026-04-24`) when asked for *time* (`04:58` was in context). Confuses semantic roles of two adjacent facts. |
| P2 "what date is today?" | **2** | "Today is April 24th" — correct day/month, drops the year. Partial success. |
| P3 "make me laugh" | **0** | "Okay, I'm ready to help you with your test." — full topic refusal, no humor attempt. |
| P4 "what is my current HR and BP?" | **3** ✓ | "Your heart rate is 72 bpm and your blood pressure is 118/76 mmHg." — **perfect**: concise, correct values from context, 1 sentence, no hallucination. |
| P5 "summarize my current health status" | **0** | "I am a helpful AI assistant. I am currently in good health…" — **hallucinated**: subject confusion (model claims it is in good health), invents "able to perform basic medical procedures and take medication." |

**Average P1-P5 = (1+2+0+3+0)/5 = 1.2** → below plan §7 gate (≥ 2.0).
**P5 score 0** → auto-fail per §6.2 ("no hallucinated diagnosis").

**G_QUALITY: FAIL** on both the average-below-2.0 rule and the P4/P5-score-0 rule.

---

## 4. Performance gate (§7 G_LATENCY — recalibrated)

| Metric | Value | Gate outcome |
|---|---|---|
| TTFT range | 5.57 s (C1) → 9.90 s (P4) | Varies with user-prompt length (more tokens to prefill → longer TTFT). P4 is outlier because "what is my current heart rate and blood pressure?" tokenizes to ~10 user tokens vs C1's ~3. |
| Decode rate | 1.64–1.79 tok/s (mean **1.70**) | Consistent across all 18 runs. |
| Cold-vs-warm variance (same process) | < 1 % (e.g., C1: 5611/5566/5572 ms) | Within any reasonable stability threshold; plan's 25 % limit satisfied comfortably. |
| Load + warmup (one-time per process) | 99.2 s | 130-token system prompt prefilled sequentially; dominant cost. |

**G_LATENCY revised gate: PASS** (stability + no-regression, not absolute throughput — per §7's calibration escape clause triggered 2026-04-24).

**Note on absolute speed**: 1.7 tok/s is slow for interactive UX but acceptable for this phase — Phase 1.5 selects *which SLM*, not *NPU throughput tuning*. Root cause: small decoder-only models (≤ 500M) on this HAL are dispatch-overhead-dominated (host↔NPU round-trip per token). Documented in §6.3 failure modes.

---

## 5. Memory behavior

| Sample point | CmaFree (kB) | MemAvailable (kB) |
|---|---|---|
| Pre-load (post-reboot idle) | 487,536 | 1,724,228 |
| Post-load+warmup | 6,580 | 630,096 |
| Post-18-run decode | (not re-sampled — in-process, stable) | (not re-sampled — in-process, stable) |

CmaFree dropped from 487 MiB → 6.4 MiB (−481 MiB) during warmup. **Gemma 3 alone saturates the 524 MiB CMA pool.** Confirms Scenario 2 from §4.1 (no Moonshine-NPU coexistence; see `backlogs.md §1.18 addendum-1`).

---

## 6. Decision + what's next

**Gemma 3 270M-IT is rejected as Phase 1.5 primary** — fails G_QUALITY on P1 (confusion), P3 (refusal), P5 (hallucination). Retrieves single facts correctly (P4) but fails at synthesis and disambiguation.

**Phase C triggered per plan §9 entry condition**: SmolLM2-360M-Instruct full compile + sweep. Entry gates:
- C1: install `torq-tools` from `references/Synaptics/torq-tools/`
- C2: `torq-export-model smollm2 -s 360M --instruct-model --convert-dtype bf16`
- C3: `torq-compile :v1.5 --iree-hw-target=SL2610 …` (compile risk — first HF→VMFB LLM compile in our repo)
- C4: `bundle-vmfb` to `.synap`
- C5–C6: scp + `bench_prompt.py --all` (reuse harness verbatim)

If SmolLM2-360M also fails G_QUALITY: fall back to SmolLM2-135M (C7) OR invoke plan §10 OQ-3 option (c) — retrieval-style template where Python renders structured health facts and the SLM only handles conversational prompts (P1-P3). Given P4 passed at 3/3 on 270M, option (c) is architecturally defensible regardless of Phase C outcome.

---

## 7. Working pathways that let us get this data

Captured for reproducibility + future-phase reference:

### W1 — `max_prompt_tokens=None` is mandatory
Vendor `runner.py:256-258` computes `_max_user_tokens = max(0, max_prompt_tokens - len(sys_tokens))` in `_warmup()`. If `max_prompt_tokens <= len(sys_tokens)`, the user query is truncated to 0 tokens on line 284-289 in `run_stream`. **Symptom**: model generates "Okay, I am ready to help. Please provide the information about the patient" template loops regardless of user prompt, because the user prompt literally isn't passed to the model. **Fix**: pass `max_prompt_tokens=None` to `Gemma3Static` — skips both truncation and padding entirely.

Also: `max_prompt_tokens` truncates the *system* prompt (same line 257). Setting it to a value below the full sys-prompt length silently discards the tail — e.g. a 80-char cap against a 130-token sys prompt chopped mid-sentence, which explains the first-run's garbage before we identified this.

### W2 — Manual TTFT/token counting beats vendor stats when capping early
Vendor's `Gemma3Static.run_stream` sets `self._n_tokens_gen` and `self._last_infer_ns` **only after** the inner `while` loop exits naturally (`runner.py:310-311`). If our caller breaks out on a `MAX_NEW_TOKENS` cap, those stats stay at their 0 initial values. **Fix**: track `t_stream_start`, `t_first_token`, and `chunk_count` in our for-loop — don't rely on `gemma3.generated_tokens` / `time_to_first_token` / `last_infer_time` when we may break early.

### W3 — Single-process multi-prompt design avoids CMA fragmentation
Cross-process cycling fails: each Python process leaves ~100 MiB of CMA residue + `syna_npu` kernel-driver state that accumulates. Even after `rm` of previous outputs and a fresh Python invocation, the second process can't allocate the initial NPU network (`failed to start network via IOCTL: Cannot allocate memory`). **Fix**: run all prompts in **one Python process** (`bench_prompt.py --all`). Shared warmup + shared pinned weight set — no new CMA allocations for subsequent runs. Paid ~99 s of warmup once vs the theoretical "one process per prompt" cost of 6 × 99 s = 600 s AND the failure mode above.

### W4 — Env-on-SD + `/tmp/` symlinks survive reboots
`/tmp` is tmpfs; reboot wipes `/tmp/p15site` (200 MiB Python env) + `/tmp/pipbase` (12 MiB pip) + `/tmp/p15-env.sh`. Rebuilding costs ~20 min of Phase-A bootstrap. **Fix**: migrate env to SD (`cp -a /tmp/p15site /tmp/pipbase /tmp/p15-env.sh /mnt/sdcard/`) then symlink back after every boot:

```
ln -sfn /mnt/sdcard/p15site /tmp/p15site
ln -sfn /mnt/sdcard/pipbase /tmp/pipbase
ln -sfn /mnt/sdcard/p15-env.sh /tmp/p15-env.sh
. /tmp/p15-env.sh
```

Phase A's env-source script expects `/tmp/*` paths; Python transparently follows symlinks. One-time migration cost; per-reboot recovery is now ~10 s.

### W5 — Board reboot required to reset NPU driver state
The `syna_npu` kernel driver retains per-network descriptor state across Python process cycles. Built-in driver (no loadable-module support on this kernel — `/lib/modules` empty), so no `modprobe -r` escape. After a handful of Torq-runtime sessions, even a fresh process with > 300 MiB CmaFree fails at `failed to acquire hardware`. **Fix**: `ssh nouslogic-sl2619 'reboot'` between NPU-heavy sessions. With env-on-SD (W4), recovery is ~1 min total.

Combined with W3, this means: **one bench sweep per boot, all prompts in one process**. Acceptable pattern for Phase 1.5; Phase 2 may need different ergonomics (e.g. a long-lived A55 service holding the NPU context).

### W6 — One-line commands only over SSH
All user-run shell blocks must be single physical lines. Newlines inside `ssh host '...'` are preserved verbatim and cause the remote shell to split commands at line breaks — `grep -E "pattern"<newline>/proc/meminfo` hangs because `grep` reads stdin with no file arg. Feedback memory at `~/.claude/projects/-home-lanhp-wsl-nouslogic-SynapticSL2619/memory/feedback_oneline_commands.md`.

### W7 — Script-editing discipline
The bench harness needed two fix cycles (W1, W2) to produce valid data. Both bugs were introduced by over-parameterization of the smoke script — `max_prompt_tokens=80` was my choice, not the vendor's pattern (vendor's `infer.py` never sets a value, leaving it `None`). **Lesson**: when adapting vendor scripts, default to vendor values unless there's a concrete reason to override. Over-constraining hyperparameters is a silent failure mode.

---

## 8. Pointer map

- Plan section: `docs/plans/models-testing-plan.md §9 Phase B → Phase D D1 (this file)`
- Post-mortem narrative: `docs/plans/backlogs.md §1.19`
- Working pathways (normative): `docs/conventions/15-model-compiler-runtime.md §5.4 + §11`
- Raw on-board artifacts: `/mnt/sdcard/bench/2026-04-24_gemma3/{*.txt,*.json}`
- Host-side log: `/tmp/p15-stage/gemma3_sweep.log` (~23 KB transcript of the 18-run sweep)

This file is **frozen** — future phase updates should not edit it. A new `docs/tmp/bench/2026-MM-DD_<candidate>-summary.md` should be written per candidate per run.
