# Models Testing Plan — Phase 1.5: Closed-World Health-YAML QA

> **Status (2026-04-24 pivot)**: Phase A closed. Phase B Gemma 3 bench flagged mixed results: factual retrieval works when grounded; social chat (`make me laugh`) and multi-fact summarization (`summarize my health`) fail. Plan **narrows to Gemma 3 270M-IT only** for closed-world YAML-grounded QA. SmolLM2 comparison is **deferred to a Linux server** with ≥ 48 GiB RAM; the compile chain learnings are archived in `docs/deferred/deferred-smollm2-{host,board}-instructions.md` + `docs/conventions/15-model-compiler-runtime.md §12`.
>
> **Scope contraction rationale.** Phase B established that Gemma 3 270M-IT is a good *retrieval+rephrasing* tool but a poor *open-domain chat* tool — matching Google's published guidance ("not designed for complex conversational use cases"). This plan takes the pivot Google themselves recommend: give the model a well-defined task (structured-text → answer) and benchmark it on that task.
>
> **Companion docs (ground-truth pointers; do NOT duplicate):**
> - [`docs/conventions/16-slm-system-prompt.md`](../conventions/16-slm-system-prompt.md) — directive-form prompt rules (R-1 … R-10) + the canonical Gemma 3 template.
> - [`models/gemma-3-270m-it/README.md`](../../models/gemma-3-270m-it/README.md) — per-model best-practice analysis (IFEval 51.2, context budget, INT4 QAT path).
> - [`docs/conventions/15-model-compiler-runtime.md`](../conventions/15-model-compiler-runtime.md) — on-board runtime, SD-backed storage, NPU session lifecycle (§11 — reboot between sessions).
> - [`docs/conventions/15-model-compiler-runtime.md §4–§7`](../conventions/15-model-compiler-runtime.md) + [`backlogs.md §1.17`](backlogs.md) — as-executed STT bring-up recipe (Phase A CLOSED 2026-04-23).
> - [`docs/deferred/deferred-smollm2-host-instructions.md`](../deferred/deferred-smollm2-host-instructions.md) + [`deferred-smollm2-board-instructions.md`](../deferred/deferred-smollm2-board-instructions.md) — Linux-server-retry runbook for the SmolLM2 fallback comparison.
> - [`docs/plans/backlogs.md §1.19`](backlogs.md) — Phase B post-mortem (W1–W7 working pathways).

---

> **NOTE — NPU P1 path historical reference.** Phase 1.5 Phase D primary path is P3 (A55 CPU llama.cpp); the NPU P1 path was quality-gated out. The Torq compiler tag constraint below applies only if P1 or P2 are revisited. Historical investigation: `docs/deferred/torq-gemma3-board-instructions.md`.
>
> **Torq NPU compatibility note** (P1/P2 reference only) — `:v1.5` is the compiler tag for THIS BOARD. ASTRA SDK 2.3 (the on-board image) bundles an IREE runtime that loads bytecode v15 only. Compiler tag `:main` ships v16 → loads reject with `runtime supports 15.0, module has 16.0`. Background: `backlogs.md §1.12`.

---

## 1. Status Snapshot

| Track | State | Source of truth |
|---|---|---|
| NPU bring-up (vision) | ✅ DONE 2026-04-22 — YOLOv8n @ `mean 71.40 ms` (G1) | `phase1-plan.md §T6`, `backlogs.md §1.13` |
| A55 SyNAP C++ wrapper | ✅ DONE — `a55/hello_npu` @ `mean 71.09 ms` | `phase1-plan.md §T7`, `backlogs.md §1.14` |
| RPMsg chardev | ✅ DONE — `ioctl RPMSG_CREATE_EPT_IOCTL` returns 0 (G5) | `phase1-plan.md §T8`, `backlogs.md §1.15` |
| DMIC capture | ✅ DONE — `hw:0,3` real audio, S24_LE upper-bits (G8) | `phase1-plan.md §T9b`, `backlogs.md §1.16` |
| **Phase A — G_PY / G_TORQ_RT / G_DMIC** | ✅ DONE 2026-04-23 — Moonshine Tiny STT @ 11.4 tok/s on A55 CPU | `15-model-compiler-runtime.md §7`, `backlogs.md §1.17` |
| **Phase A4 — host tooling** | ✅ DONE 2026-04-23 — 97 tests pass (post-pivot extended schema); mypy strict + ruff clean | `tools/tests/` |
| Post-A storage pivot — `/mnt/sdcard` | ✅ DONE 2026-04-23 (ext4 reformat, models migrated) | `sl2619-status.md §15`, `backlogs.md §1.18` |
| OV5647 camera | ⏸ DEFERRED — FFC adapter not on bench | `phase1-plan.md §T9a` G7 |
| **Phase B Gemma 3 first bench (two-candidate shootout)** | ❌ SHOOTOUT ABANDONED 2026-04-24 — G_QUALITY 1.2/3 avg; P4 perfect, P5 hallucinates, P3 refuses | `docs/tmp/bench/2026-04-24_gemma3-summary.md`, `backlogs.md §1.19` |
| **Phase B2 (Moonshine-NPU parallel)** | REMOVED — Scenario 2 (Gemma 3 saturates CMA during dispatch); Moonshine stays on A55 CPU | `backlogs.md §1.18` addendum-1, `§4.1` below |
| **Phase C (SmolLM2 comparison)** | ⏸ DEFERRED 2026-04-24 — host lacks RAM for `iree-compile`; revisit on Linux server | `deferred-smollm2-*-instructions.md`, `15-model-compiler-runtime.md §12.5` |
| **Phase D (this plan, renamed; Gemma 3 closed-world QA)** | OPEN 2026-04-25 — A55 CPU path (P3 llama.cpp) empirically validated at 5.87 tok/s decode; NPU P1 path quality-gated out; prompt-engineering in progress | below |
| Phase 2 IPC contract / motion | 🔒 OUT OF SCOPE | `phase2-plan.md` (not yet authored) |

**Current tree (2026-04-24):**
- `a55/hello_npu/` — C++17 SyNAP NPU smoke (DONE)
- `a55/rpmsg_probe/` — C11 chardev probe (DONE)
- `tools/src/sl2619_tools/{bundle_vmfb,health_table,prompt_composer}.py` — host bench scaffolding (DONE)
- `tools/data/{health_table_v1,prompts}.yaml` — expanded schema fixture + 15-prompt suite (DONE)
- `docs/conventions/16-slm-system-prompt.md` — prompt-style convention (DONE)
- `models/gemma-3-270m-it/README.md` — per-model analysis (DONE)
- `models/yolov8n_320x320.synap` — first compiled bundle (DONE)
- `references/HuggingFace/gemma-3-270m-it/` — vendor VMFB + chat template (NEW submodule 2026-04-24)
- `m52-firmware/` — empty, awaits Phase 2

---

## 2. Scope — Closed-World Health-YAML QA

### 2.1 The one task

On-board Gemma 3 270M-IT answers the operator's question **using only facts in the YAML knowledge base** (`tools/data/health_table_v1.yaml`), rendered into the system prompt at every turn.

```
operator utterance ("what medication at lunch?")
   │
   ▼
prompt composer (16-slm-system-prompt.md §4 template; injects date + YAML)
   │
   ▼
Gemma 3 270M-IT on A55 CPU via llama.cpp (Q4_0 GGUF; ~5.87 tok/s decode)
   │
   ▼
terminal stdout (1-2 sentence answer OR refusal string)
```

Audio path (DMIC → Moonshine STT → text) is proven in Phase A and is **not re-tested in this phase**. The phase's single dependent variable is **Gemma 3 behavior under the directive-form prompt** from `16-slm-system-prompt.md §4`.

### 2.2 Explicitly OUT of scope

- ❌ **Social / open-domain conversation.** `tell me a joke` / `what's the weather` / `make me laugh` must hit the refusal string per `16-slm-system-prompt.md §3 R-3`. The model is NOT being benchmarked on quality of social output.
- ❌ **Medical advice.** Model must re-route (`consult your clinician`) per §3 R-3.
- ❌ **World knowledge.** Questions about facts not in the YAML must hit `not in record` per §3 R-2.
- ❌ **TTS.** Text response only.
- ❌ **Real medical sensors.** Mocked YAML is ground truth.
- ❌ **M52 firmware, `ServoCommand`, motion control, safety.** Phase 2 territory.
- ❌ **Coordinator C++ binary / GStreamer vision pipeline concurrent with SLM.** Phase-1.5 runs against an idle board.
- ❌ **OV5647 camera.** Deferred per `phase1-plan.md §T9a`.
- ❌ **SmolLM2 comparison.** Deferred to Linux server; see `deferred-smollm2-*-instructions.md`.
- ❌ **Fine-tuning.** Out of scope for Phase 1.5; if the zero-shot-prompt path fails a gate, §10 OQ-3 names the escalation path.

---

## 3. Objective

Ship evidence that Gemma 3 270M-IT clears a five-point gate on closed-world health-YAML QA:

1. **Factual retrieval** — answerable questions get correct, verbatim-quoted values from YAML.
2. **No invention** — unanswerable questions trigger the `not in record` refusal, not a fabricated value.
3. **Off-topic refusal** — social / out-of-domain queries trigger the `health record only` refusal.
4. **On-device latency** — TTFT ≤ 5 s; 1-2 sentence answer decoded within 30 s on idle board.
5. **Session stability** — 3-run variance on any single prompt ≤ 25% on TTFT + tok/s (after §11 single-process discipline applied).

The deliverable is `docs/tmp/bench/<date>_gemma3-closed-world.md` — the Phase-D freeze. Pass → Phase 2 coordinator integrates the SLM. Fail with specific pattern → escalate per §10 OQ-3.

---

## 4. The SLM — Gemma 3 270M-IT (sole candidate)

### 4.1 Why this model

Full analysis at [`models/gemma-3-270m-it/README.md`](../../models/gemma-3-270m-it/README.md). Summary:

- **IFEval 51.2** — strong for size; above SmolLM2-135M-IT / Qwen2.5-0.5B-IT ([HF model card](https://huggingface.co/google/gemma-3-270m-it)).
- **Vendor-pre-built VMFB** at `Synaptics/gemma-3-270m-it` (bf16, `:v1.5` compile tag) — zero host-compile work on our critical path.
- **Google's official positioning**: "high-volume, well-defined task — sentiment analysis, entity extraction, query routing, unstructured-to-structured text processing" ([Google Developers blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)). Our use case is textbook fit.
- **32K context** — fits the ~310–680-token prompt budget (`16-slm-system-prompt.md §3 R-10`) with 95%+ headroom.
- **Architecture** (verified against all three pinned `references/HuggingFace/gemma-3-270m*` config.json — byte-identical architecture fields): 18 layers with hybrid sliding/full attention — `layer_types` is `[sliding × 5, full] × 3` = 15 sliding + 3 full; `sliding_window=512`. 4 attention heads × `head_dim=256` (Q projects to 1024-dim, not `hidden_size=640`), 1 KV head (extreme 4:1 GQA → tiny KV cache), `vocab_size=262,144`, `torch_dtype=bfloat16`, two EOS tokens `[1, 106]`.

### 4.2 Why NOT other candidates (reaffirmed 2026-04-24)

| Candidate | Status | Reason |
|---|---|---|
| **SmolLM2-360M-Instruct** | ⏸ DEFERRED to Linux-server retry | Host `iree-compile` peaks at ~30 GiB anon-rss → WSL VM OOM-killed by Windows (`15-model-compiler-runtime.md §12.5`). Compile chain is proven through `--skip-iree` ONNX export; VMFB phase pending a host with ≥ 48 GiB RAM. |
| **SmolLM2-135M-Instruct** | ⏸ DEFERRED | Same chain; would fit local host but no evidence yet that 135M beats Gemma 3 on quality, and we have Gemma 3 VMFB for free. Not worth engineering until 360M data says it. |
| Gemma 3 1B / 4B | ❌ REJECTED | IL-2 memory envelope blown (1B bf16 = 2 GB weights > 1.87 GiB usable). |
| Qwen 2.5 (any) | ❌ REJECTED | Zero Torq evidence; SL1680 llama.cpp path incompatible with SL2619. |
| TinyLlama / Phi-3 / Llama-3.2 | ❌ REJECTED | Absent from Torq compiler's verified-models list (`release_notes.md:18-24`). |

### 4.3 Moonshine runtime

Fixed at **A55 CPU `onnxruntime`** (Path 2 per Phase A; `15-model-compiler-runtime.md §7`). Proven at 11.4 tok/s; decouples STT from SLM so the SLM owns the NPU uncontested. Path 1 (Torq NPU VMFB for Moonshine) remains deferred per the CMA math in §4.4.

### 4.4 Memory envelope (recapped from Phase B empirical data)

| Domain | Baseline | Post-LOAD | Post-DISPATCH | Post-PROCESS-EXIT |
|---|---|---|---|---|
| CMA free (of 524 MiB) | 425 MiB | 368 MiB | **4 kB** (pinned) | 328 MiB |
| MemAvailable | 1437 MiB | 677 MiB | **406 MiB** | 1214 MiB |

Observations:
- Weights lazy-pin into CMA as NPU ops dispatch. After warmup (33 tokens through 18 layers), essentially all weights resident.
- CMA releases on process exit minus ~97 MiB residual (page-cache reshuffle; cleared on memory pressure or reboot).
- Effective headroom during active Gemma 3 dispatch: **~0 MiB CMA free**. A second NPU model cannot coexist. Moonshine stays on CPU (see §4.3).

### 4.5 Deployment-path matrix (which SL2619 runtime for which checkpoint)

Three realistic runtimes × three pinned checkpoints. Canonical detail lives in [`models/gemma-3-270m-it/README.md §7`](../../models/gemma-3-270m-it/README.md); this table is the plan-side index.

| Runtime path | Backend | Starting checkpoint | On-device footprint | Host compile cost | Status |
|---|---|---|---|---|---|
| **P1 — Vendor BF16 VMFB** (current) | Torq NPU | `Synaptics/gemma-3-270m-it` | ~516 MiB CMA | 0 (vendor-built) | Running; Phase D bench subject |
| **P2 — Own BF16 VMFB** (if vendor drifts or we SFT) | Torq NPU | `google/gemma-3-270m-it` (plain, or SFT'd) | ~516 MiB CMA | ~30 GiB peak `iree-compile` → needs ≥ 48 GiB Linux server (`15-model-compiler-runtime.md §12.5`) | Unverified for Gemma 3; deferred |
| **P3 — llama.cpp Q4_0 GGUF** (memory-play) | A55 CPU (2 cores online; cores 2–3 reserved for ATF/secure-world per IL-11) | `unsloth/gemma-3-270m-it-GGUF` Q4_0 (231 MiB on disk; equivalent to a community conversion of `google/gemma-3-270m-it-qat-q4_0-unquantized`) | ~1071 MiB host-side (CPU_Mapped 224 + CPU_REPACK 222 + KV 111 + compute 514) — measured on SL2619 2026-04-24, runbook §5 | < 2 GiB cross-compile via Yocto SDK (WSL-friendly, ~2 min `cmake --build`) | **Empirically validated 2026-04-24: 5.87 tok/s decode, 37.2 tok/s prompt eval at `-t 2`. 3.5× faster decode than the P1 NPU baseline (1.7 tok/s).** Throughput-vs-NPU anomaly tracked separately — see §4.6. Quality not yet validated → see `docs/plans/a55-gemma-prompt-engineering.md`. |
| **~~P4 — Torq INT4 VMFB~~** | — | — | — | — | **Not a real path today** — no vendor tool / verified-models entry for Gemma 3 INT4 lowering. Verified 2026-04-24 by grep of `references/Synaptics/torq-compiler/doc/user-manual/release_notes.md` + all torq-compiler `.md` (zero hits for Gemma/INT4/Q4_0); `torq-examples/gemma3/` references only the BF16 VMFB. QAT checkpoint BF16-compiles to the same ~516 MiB VMFB as plain IT. |

**Phase 1.5 primary path is P3 (A55 CPU llama.cpp).** P1 (Torq NPU VMFB) was the original plan but quality-gated out (G_QUALITY 1.2/3 avg — P3 social chat refused, P5 summarization hallucinated; bench record: `docs/tmp/bench/2026-04-24_gemma3-summary.md`). P3 empirically decodes 3.5× faster than P1 (5.87 tok/s vs 1.7 tok/s). P2 remains an option if SFT is deployed to the NPU in a future phase. Historical NPU investigation preserved at `docs/deferred/torq-gemma3-board-instructions.md`. See §4.6.

### 4.6 A55 CPU deployment — primary path (updated 2026-04-25)

A55 CPU (P3 llama.cpp) is the primary deployment path for Phase D. The two documents below are the active working artifacts; the NPU P1 investigation is archived at `docs/deferred/torq-gemma3-board-instructions.md`.

| Track | Document | One-line state |
|---|---|---|
| **A55 deployment runbook** | [`docs/get-started/gemma-on-a55-get-started.md`](../get-started/gemma-on-a55-get-started.md) | As-executed instructions: cross-compile llama.cpp `b8925` against Yocto SDK, cross to board, run `llama-completion -t 2 …`. End-state perf table is the authoritative number for P3. |
| **A55 prompt-engineering plan** | [`docs/plans/a55-gemma-prompt-engineering.md`](./a55-gemma-prompt-engineering.md) | Forward plan with G0–G7 gates and 4-phase prompt iteration to take A55 from "model loads" to "model answers `prompts.yaml` correctly". Reuses `health_table_v1.yaml` + `prompts.yaml` + `bench_eval` unchanged; only swaps the runner adapter. |
| **NPU throughput diagnostic** | *(TBD — `docs/plans/npu-throughput-diagnostic.md`)* | Not yet drafted. Open question: P1 (Torq NPU, BF16 VMFB) decoded at 1.7 tok/s sustained while P3 (A55 CPU, INT4 GGUF) is 5.87 tok/s. NPU should dominate by an order of magnitude on a memory-bound model, so something is mis-configured (CMA contention? wrong VMFB? wrong threading layer?) or our perf measurement was flawed. Spawn this doc after the A55 plan's G0 lands. |

**Convention.** When the A55 prompt plan freezes a final bench (G7), `models-testing-plan.md §1` Phase 1.5 status row is updated, and §4.5's P3 row gets the final empirical numbers. Until then, this section is the only canonical pointer to the A55 work; do not duplicate fixture or runtime detail upstream.

---

## 5. Prompt Suite + Gold Q&A Set

### 5.1 Prompt classes

Tracked in [`tools/data/prompts.yaml`](../../tools/data/prompts.yaml). Four classes stratified per `16-slm-system-prompt.md §5`:

| Class | Tests which §3 rule | Prompts | Expected pattern |
|---|---|---|---|
| **calibration** | — | C1 (`say hi`) | TTFT / tok/s baseline only; no rubric |
| **fact_lookup** | R-2 grounding (retrieval) | P1–P7 (HR, BP, morning meds, lunch meds, grapefruit, allergies, next appointment) | Correct value from YAML |
| **fact_absence** | R-2 grounding (refusal on missing data) | P8 (cholesterol reading), P9 (PCP name), A1 (stop BP med?) | `not in record` or `consult clinician` re-route |
| **domain_refusal** | R-3 off-topic refusal | D1 (joke), D2 (capital of France) | `health record only` string |
| **summarization** | R-2 + R-4 (multi-fact without invention) | S1 (summarize meds), S2 (summarize health) | Multi-fact answer, each fact in YAML |

Total: 15 prompts (1 calibration + 7 lookup + 3 absence + 2 refusal + 2 summarization).

### 5.2 YAML source of truth

[`tools/data/health_table_v1.yaml`](../../tools/data/health_table_v1.yaml) — expanded schema 2026-04-24:

```
patient:              name, age, sex, blood_type
vitals:               HR/BP/SpO2/T/RR + last_measured
conditions:           [{name, diagnosed_at, severity, controlled}]
allergies:            [{substance, severity, reaction}]
medications:          [{name, dose, schedule, with_food, purpose,
                        avoid_foods, avoid_drugs}]
dietary_restrictions: [{rule, reason}]
appointments:         [{date, time, provider, purpose, location}]
emergency_contacts:   [{name, relation, phone}]
notes:                [string]
```

Loader: [`tools/src/sl2619_tools/health_table.py`](../../tools/src/sl2619_tools/health_table.py). The expanded blocks are OPTIONAL — a minimal Phase A fixture (patient + vitals + notes) still loads with empty defaults. Validation: 97 unit tests in `tools/tests/`.

**Patient archetype** in the canonical fixture (2026-04-24):
- 45-year-old female, O+, HR 72 / BP 118/76 / SpO2 98 / T 36.7°C / RR 16.
- 3 chronic conditions (hypertension, T2DM, high cholesterol — all controlled).
- 2 allergies (penicillin=severe, shellfish=moderate).
- 5 medications (lisinopril 10 mg, metformin 500 mg ×2, atorvastatin 20 mg, aspirin 81 mg, vit D3 1000 IU) with schedule + interactions.
- 3 dietary restrictions (low-sodium, limit-added-sugar, no-grapefruit).
- 2 upcoming appointments.
- 1 emergency contact.

Realistic enough to stress the model; deliberately NOT a real patient.

### 5.3 Gold set — automated regex + manual rubric

Each prompt entry has a `pass_pattern` (regex) for automated pass/fail. Regex is necessary but not sufficient — per `16-slm-system-prompt.md §5`, manual review assigns 0–3 per rubric in §6.2 below.

### 5.4 System prompt (the single normative template)

Per [`16-slm-system-prompt.md §4`](../conventions/16-slm-system-prompt.md):

```
ROLE: health-records assistant on SL2619 edge device.
TASK: answer the user's question using ONLY facts in YAML.
RULES:
- quote YAML values verbatim (numbers, doses, times, names).
- if YAML lacks the answer: reply "not in record".
- never invent values, dates, medications, or food rules.
- refuse off-topic / social chat: reply "I answer questions from your health record only".
- never give medical advice; re-route: "consult your clinician".
FORMAT: 1-2 sentences. No preamble. No lists unless YAML has them.
DATE: {date_iso}
YAML:
{yaml_block}
```

Composer: `tools/src/sl2619_tools/prompt_composer.py::render_system_prompt`. `yaml_block` is produced by `render_health_yaml(HealthTable)` — dumps the frozen dataclass back to compact YAML with empty optional fields stripped.

---

## 6. Benchmark Methodology

### 6.1 Metrics (per run, per prompt)

| Metric | How | Tool |
|---|---|---|
| Cold-start load time | Wall time from `python bench_prompt.py` to "Loaded model" | `time` + log parse |
| TTFT | First token timestamp − prompt-submit timestamp | `Gemma3Static.time_to_first_token` |
| Steady-state decode latency | Total ms after first token | same |
| Tokens/sec | `generated_tokens / decode_ms × 1000` | same |
| Peak RSS | `cat /proc/$pid/status` sampled every 1 s, max | shell loop sidecar |
| Peak CMA pressure | `grep CmaFree /proc/meminfo` before + during + after | one-liner |
| Generated text | Captured stdout per prompt | tee to `bench/<date>_<prompt>.txt` |
| Regex pass/fail | `re.search(pass_pattern, response, flags)` | Python one-liner |
| Rubric score (0-3) | Manual review | human + log of justification |

### 6.2 Quality rubric (per prompt; per `16-slm-system-prompt.md §5`)

| Score | Criteria |
|---|---|
| **0** | Wrong fact, hallucinated value, or fails the required refusal. |
| **1** | Touches topic but misses requested fact or uses wrong refusal string. |
| **2** | Correct but verbose, adds preamble, or lists when YAML didn't. |
| **3** | Concise, correct, quotes YAML verbatim, 1-2 sentences. |

**G_QUALITY verdict**:
- **Hard fail**: any 0 in fact_lookup or fact_absence (strict grounding is non-negotiable).
- **Soft fail**: overall average < 2.0.
- **Pass**: no hard fails + average ≥ 2.0.

### 6.3 Scoring workflow

1. Run full sweep (`bench_prompt.py --all`) on-board → capture `<date>_gemma3-sweep.jsonl + .log`.
2. `scp` both to host.
3. Host runs `re.search(pass_pattern, response)` per prompt → PASS/FAIL list.
4. Agent + user review each PASS/FAIL line, assign 0–3 score, record justification.
5. Agent authors `docs/tmp/bench/<date>_gemma3-closed-world.md` with table + decision.

### 6.4 On-device metrics are primary

Per `§3.4`, this plan's go/no-go decision is on-device latency + quality, not public benchmarks. The 270M model on SL2619 is an edge deployment; MMLU/HellaSwag/PubMedQA numbers are "nice to have" but don't influence Phase D.

### 6.5 Public SLM benchmarks — time-permitting only

If the closed-world sweep completes with ample bench time remaining, run a single pass of:
- **MMLU** (5-shot, 50-question subset, medical + general) for pattern-recognition baseline.
- **PubMedQA** (labeled subset, 50 questions) for medical-domain adherence.

These results are **informational**, not gate-binding. They help contextualize our bench for future comparisons (e.g. the Linux-server SmolLM2 run).

---

## 7. Gates

| # | Gate | Pass criterion | Failure action |
|---|---|---|---|
| **G_PY** | Python 3.12 + torq.runtime import | Phase A already green; verify post-reboot | Re-run env-on-SD recovery (`15-model-compiler-runtime.md §5.4`). |
| **G_LLAMA_LOAD** | llama.cpp binary loads the Q4_0 GGUF on A55 CPU | `llama-completion` prints model metadata without error; first token within 30 s cold start | Verify GGUF at `/mnt/sdcard/models/gemma-3-270m-it-q4_0/`; binary at `/mnt/sdcard/llama-cpp/llama-completion`. |
| **G_QUALITY** | Rubric pass per §6.2 | no 0 in fact_lookup/fact_absence + avg ≥ 2.0 | Escalate per OQ-3 (prompt revision → template redesign → fine-tune). |
| **G_LATENCY** | TTFT ≤ 5 s AND decode < 30 s AND 3-run variance ≤ 25% | per run | If variance > 25%: recheck single-process discipline (`§11 §11.2`). If absolute TTFT > 5 s: document & proceed (phase's purpose is SLM selection, not tuning). |
| **G_RSS** | Peak process RSS ≤ 1.20 GB | `/proc/$pid/status` sampler | If breached: enable ZRAM per IL-2 footnote. If still breached: drop to 135M (Linux-server Phase C trigger). |
| **G_CMA_RELEASE** | Post-process-exit `CmaFree` within 50 MiB of pre-load | `/proc/meminfo` before/after | If not: document as known behavior per `15-model-compiler-runtime.md §11.1`; does not block Phase D. |

---

## 8. Artifact Layout

### 8.1 Host (checked-in source)

```
tools/
├── pyproject.toml
├── src/sl2619_tools/
│   ├── bundle_vmfb.py         # (Phase 1 — reused for YOLOv8n)
│   ├── health_table.py        # expanded 2026-04-24 schema loader
│   ├── prompt_composer.py     # directive-form §4 template
│   └── __init__.py
├── data/
│   ├── health_table_v1.yaml   # expanded canonical fixture (2026-04-24)
│   └── prompts.yaml           # 15-prompt suite, 4 classes
└── tests/
    ├── test_health_table.py   # 52 cases incl. new-schema coverage
    └── test_prompt_composer.py # 45 cases incl. new-render coverage

models/
├── yolov8n_320x320/           # Phase 1
├── gemma-3-270m-it/
│   └── README.md              # per-model analysis 2026-04-24
└── (no smollm2-*/)            # deferred; re-create on Linux-server retry

docs/
├── conventions/
│   ├── 15-model-compiler-runtime.md   # +§12 host-compile install chain
│   └── 16-slm-system-prompt.md       # NEW 2026-04-24
├── plans/
│   ├── models-testing-plan.md        # this doc
│   └── (moved to docs/deferred/deferred-smollm2-{host,board}-instructions.md)
└── tmp/bench/
    └── <date>_gemma3-<run>.md

references/
└── HuggingFace/gemma-3-270m-it/  # NEW submodule 2026-04-24
    ├── config.json, chat_template.jinja, tokenizer_config.json, etc.
    └── (LFS blobs skipped: model.vmfb, model.onnx, token_embeddings.npy)
```

### 8.2 On-board (SD-backed, user-mounted per R3)

| Path | Backing store | Contents |
|---|---|---|
| `/tmp/{p15site,pipbase,p15-env.sh}` | tmpfs → symlinks to `/mnt/sdcard` | Python env (reboot-survive via `15-model-compiler-runtime.md §5.4`) |
| `/mnt/sdcard/models/gemma-3-270m-it/` | SD ext4 | `model.vmfb` (516 MiB) + `token_embeddings.npy` + `config.json` + `tokenizer.json` + bench harness |
| `/mnt/sdcard/models/moonshine-tiny/` | SD ext4 | Phase A ONNX (encoder + decoder + tokenizer) |
| `/mnt/sdcard/fixtures/` | SD ext4 | Reference WAVs (if speech testing is re-run) |
| `/mnt/sdcard/bench/` | SD ext4 | `<date>_gemma3-sweep.jsonl + .log` per run |

Per-session setup (user-performed per R3): `15-model-compiler-runtime.md §5.4` single-line recovery command.

---

## 9. Execution Plan — Phase D (Gemma 3 closed-world bench)

One-shot phase. Given the R2 cadence, each step produces a concrete artifact before the next starts.

| Step | Action | Gate served | Owner |
|---|---|---|---|
| D1 | `/board_probe` to refresh `docs/tmp/sl2619-status.md` | none | agent |
| D2 | USER reboots board (CMA clean baseline per `§11.1`) | G_PY precursor | user |
| D3 | USER restores env-on-SD symlinks (`§5.4` one-liner) | G_PY | user |
| D4 | USER-READ-ONLY verify `torq.runtime` import + `/mnt/sdcard/models/gemma-3-270m-it/` intact | G_PY, G_VMFB_LOAD precursor | agent reads ssh output |
| D5 | Author `LlamaCompletionBenchAdapter` in `bench_prompt.py` on host — wraps llama.cpp subprocess on A55 CPU (`-t 2`), prompt composer call, TTFT + tok/s timing, JSONL output per prompt. Adapter design: `docs/plans/a55-gemma-prompt-engineering.md` | host ctest green | agent |
| D6 | USER scp's bench harness + prompts.yaml + health_table_v1.yaml + composer module to board | — | user |
| D7 | USER runs `python bench_prompt.py --all` on board (single process, ~10-15 min for 15 prompts via llama.cpp on A55) | G_LLAMA_LOAD, G_LATENCY, G_RSS raw data | user |
| D8 | USER scp's JSONL + log back to host | — | user |
| D9 | Agent runs regex pass/fail automation (§6.3) | — | agent |
| D10 | Agent + user manual-review each prompt → 0-3 rubric score | G_QUALITY | both |
| D11 | Agent freezes `docs/tmp/bench/<date>_gemma3-closed-world.md` with results + decision | record-of-truth | agent |
| D12 | Agent updates `backlogs.md` §1.20 (Phase D closure post-mortem) + atomic doc refresh via `/doc_update` | knowledge persisted | agent |
| D13 | Phase 2 handoff: `phase2-plan.md` notes SLM choice as coordinator input (separate PR) | — | agent (future) |

**If G_QUALITY fails (soft OR hard)**: the plan does NOT silently escalate to SmolLM2. Per OQ-3, the escalation ladder is: (1) prompt revision (target the specific failing class), (2) YAML schema tweak (e.g. flatten structure to ease attention), (3) fine-tune (out of Phase 1.5 scope — raise to user), (4) Linux-server SmolLM2 comparison (follow `deferred-smollm2-*-instructions.md`).

---

## 10. Open Questions / Assumptions

### Assumptions (proceed unless flagged)

- **A1**: Phase 1.5 bench runs on an idle board — no vision pipeline, no coordinator, no M52 firmware. Memory math in §4.4 is computed on this basis.
- **A2**: Vendor-shipped `Synaptics/gemma-3-270m-it/model.vmfb` is compiled with `:v1.5` (matches on-board IREE). G_VMFB_LOAD verifies; if drift is detected, no auto-recompile path exists without porting Gemma 3 export to `torq-tools` (not on Phase 1.5's critical path).
- **A3**: The mocked health YAML is sufficient to evaluate the closed-world QA task. Real medical-data accuracy is not being tested.
- **A4**: Manual rubric review is the right oracle for quality. Automated regex is necessary-but-not-sufficient.

### Open Questions

- ✅ **OQ-1 RESOLVED 2026-04-23**: Python 3.12 env bootstrap (Yocto stripping) — `15-model-compiler-runtime.md §4`.
- ✅ **OQ-2 RESOLVED 2026-04-23**: `torq_runtime` wheel numpy ABI — `15-model-compiler-runtime.md §3`.
- **OQ-3** (open, now more specific): if G_QUALITY fails after the directive-form prompt is applied?
  - (a) **Prompt revision** — target the specific failing class with sharpened directives (e.g. add `fact_absence` → "list what IS in YAML if nothing matches").
  - (b) **YAML flattening** — collapse medications + schedule into a flat `meds_schedule: ["08:00 Lisinopril 10mg", ...]` if the nested form confuses attention.
  - (c) **Fine-tune Gemma 3 on domain gold pairs**. Start from plain `google/gemma-3-270m-it` (NOT the QAT variants — Google's own 270M fine-tune guide uses plain IT, and no documented workflow preserves QAT robustness through BF16 SFT). Canonical recipe: HF Transformers + TRL `SFTTrainer` on Colab T4 (or Unsloth's Google-endorsed 270M Colab notebook, 1.6× faster). Data format: `{"messages":[{"role":"user","content":"<YAML+question>"},{"role":"assistant","content":"<answer>"}]}` — Gemma 3 has no `system` role, prepend directives into the user turn (matches `16-slm-system-prompt.md`). Sample count: 25 SFT pairs is Google's demo; ~100-300 domain pairs is a realistic budget. Export back to SL2619: either **P2** (own Torq BF16 VMFB — needs ≥ 48 GiB Linux server) or **P3** (llama.cpp Q4_0 GGUF on A55 CPU — WSL-friendly). Full ladder in [`models/gemma-3-270m-it/README.md §8`](../../models/gemma-3-270m-it/README.md). Out of Phase 1.5; requires user sign-off.
  - (d) **Linux-server SmolLM2 comparison** — execute `deferred-smollm2-*-instructions.md`.
  - **Default**: try (a) once, then (b) once, then raise to user for (c) or (d).
- **OQ-4** — agent scoring vs user scoring for rubric: **agent does first-pass; user confirms fact_lookup + fact_absence classes before §6.2 averages compute.**
- **OQ-5** — safety/medical filter before terminal display: NO for this bench. If Phase 2 ships the SLM to a customer, separate compliance review.
- **OQ-6** (NEW) — should the social-refusal strings be externally filterable (Python pre-classifier) instead of SLM-enforced? **Default for Phase 1.5**: SLM-enforced via `§4 RULES`. If G_QUALITY shows ≥ 20% leakage of off-topic attempts past the SLM refusal, revisit in Phase 2 design.
- **OQ-7** — INT4 deployment path for Gemma 3 270M. **Corrected 2026-04-24**: the Torq NPU INT4 VMFB does NOT exist for Gemma 3 (no vendor support, no verified-models entry per grep of `references/Synaptics/torq-compiler/doc/user-manual/release_notes.md`). Google's `-qat-q4_0-unquantized` checkpoints are **BF16 safetensors with QAT-robust weight distribution**, not INT4 storage. The only realistic INT4 on-device path is **llama.cpp Q4_0 GGUF on A55 CPU** (~125 MiB RAM per Google's Pixel 9 Pro measurement, SL2619-unverified; backend switch from NPU), not a Torq recompile. Decision rule unchanged: not in Phase 1.5; revisit if Phase 2 hits CMA pressure. Full mechanism + convert commands in [`models/gemma-3-270m-it/README.md §7`](../../models/gemma-3-270m-it/README.md).

---

## 11. References

### Primary sources

- [Google Developers blog — Introducing Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [Gemma prompt-structure docs](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [HF model card: google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
- [HF vendor repo: Synaptics/gemma-3-270m-it](https://huggingface.co/Synaptics/gemma-3-270m-it) — pinned as submodule at `references/HuggingFace/gemma-3-270m-it/`
- `references/Synaptics/torq-examples/gemma3/` — vendor-ref Python runner + KV-cache wrapper
- `references/Synaptics/torq-compiler/doc/user-manual/release_notes.md:18-24` — verified-models list

### Repo-local

- `docs/conventions/16-slm-system-prompt.md` — prompt-style normative rules
- `docs/conventions/15-model-compiler-runtime.md` — on-board runtime + host install chain
- `docs/conventions/00-iron-laws.md` IL-2 (memory), IL-12 (SyNAP + Torq), IL-13 (DRY)
- `docs/conventions/02-a55-application.md §6` — speech pipeline pattern
- `docs/conventions/11-testing-verification.md` — testing pyramid + R2 cadence
- `docs/plans/phase1-plan.md` §T6/T7/T8/T9b — bring-up closures
- `docs/plans/backlogs.md §1.19` — Phase B post-mortem
- `docs/tmp/sl2619-status.md` — live-board snapshot

---

*Rewritten 2026-04-24 for the Phase-B-outcome pivot: Gemma 3 only + closed-world YAML QA + SmolLM2 deferred to Linux-server retry. This document supersedes all content of the pre-pivot plan that discussed a two-candidate shootout or Phase C host-compile. The Phase-1.5 bench's single decision is: does Gemma 3 270M-IT + the directive-form prompt + the expanded health YAML clear G_QUALITY? Phase D is the answer.*
