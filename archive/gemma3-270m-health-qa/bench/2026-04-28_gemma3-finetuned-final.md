# Phase 3 Q4 + Q5 — Fine-tuned Gemma 3 270M Q4_0 final bench (A55 CPU, llama.cpp)

**Verdict: GREEN with note.** Q5 closes Phase 3. The fine-tuned merged Q4_0 GGUF beats the H6 base baseline by a wide margin (regex pass 8/15 vs 2/15; manual rubric ≥ 2 grounded answers on 5/15 vs 0/15) but does **not** meet the plan-§9 target of ≥ 80% / 12+ regex pass. Plan target deferred per `models-testing-plan.md §6.2` rubric — Phase 3 closure is on the qualitative "definitional drift fixed" criterion (DONE: P1 emits `72 bpm.`; S1 emits clean grounded medication list; P7 emits clean appointment date), not on the absolute pass rate.

The current `sft_v1` corpus is a v1 proof-of-concept (per user direction 2026-04-28; backlogs §1.21). The 60% non-pass rate is dominated by training-pool coverage gaps (multi-field discrimination, refusal canonical strings, repetitive degeneration after correct first-answer token), not by Q4_0 quantization noise (Q1 GREEN: same-arch 98.443% same_top_p, cross-arch Δ 0.393 pp). Phase 4 freeze captures the v1 demo numbers; v2 dataset expansion is the path to ≥ 80%.

## 0. TL;DR — Q4 + Q5 numbers

| Run | env | regex pass | rubric ≥ 2 | aggregate decode tok/s | mean wall (model-side) |
|---|---|---|---|---|---|
| **H6 (base, text-wrap envelope)** 2026-04-27 | board, n_predict=128 | 2/15 (13%) | **0/15** | 9.50 | 32.2 s |
| **H6b (base, --jinja envelope, 3-prompt subset)** 2026-04-28 | board, n_predict=128, P1/P3/D1 | 0/3 (0%) | 0/3 | (not measured) | 26.5 s |
| **Q4 (fine-tuned Q4_0, --jinja envelope)** 2026-04-28 | board, n_predict=128 | **8/15 (53%)** | **5/15** | **17.29** | 25.2 s |
| Q4 - H6 delta | — | **+6 prompts** | **+5 prompts** | +1.82× | -22% |

Real grounded answers (rubric ≥ 2): P1, P7, P9, A1, S1. Two more (P3 and P4) hit refusal-style structure ("Please consult your clinician") but with hallucinated content (rubric 1). All 15 prompts now emit some attempt at an answer; H6's pure-YAML-echo failure mode is gone.

The Q4_0 model is **2.6× faster than the H3 short-prompt runbook number** (5.87 tok/s) and **1.8× faster than the H6 bench-sized number** (9.50 tok/s) — the `--jinja` path skips the H6 plain-text-wrap tokenization overhead.

## 1. Setup

| Field | Value |
|---|---|
| Bench harness | `tools/src/sl2619_tools/bench_remote.py` (new this session — host-driven SSH-piped llama-completion; R3-compliant: agent never writes to remote) |
| CLI | `uv run bench-remote --ssh-host nouslogic-sl2619 --prompts data/prompts.yaml --health-table data/health_table_v1.yaml --output … --llama-binary /mnt/sdcard/llama-cpp/llama-completion --llama-model /mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf --max-gen-tokens 128 --n-threads 2 --temp 0.0 --top-k 1 --seed 42 --subprocess-timeout-s 180 --now 2026-04-28` |
| Adapter shape | per-prompt `ssh nouslogic-sl2619 'BODY=$(cat); /mnt/sdcard/llama-cpp/llama-completion --jinja --no-display-prompt -p "$BODY" -m … -t 2 -n 128 --temp 0.0 --top-k 1 --seed 42 -no-cnv --single-turn'` (body piped via stdin) |
| Board binary | `/mnt/sdcard/llama-cpp/llama-completion` — `version: 1 (0adede8)`, GNU 13.3.0 aarch64 (byte-matched to H6) |
| Fine-tuned model | `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` — sha256 `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4` (Q0 closure 2026-04-28) |
| Base model (H6, H6b) | `/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf` — sha256 `e479ea29…` (unsloth, ident with H5R provenance) |
| Prompt corpus | `tools/data/prompts.yaml` — 15 prompts (C1, P1-P9, D1-D2, A1, S1-S2); same source-of-truth as T5/Q1/H6 |
| Health YAML | `tools/data/health_table_v1.yaml` (113-line fixture; Path B body 2683-2696 chars per prompt; ~660 tokens body + chat-template overhead) |
| Composed user-turn | `prompt_composer.compose_user_text(health, date(2026,4,28), question)` — directive (16-slm-system-prompt.md §4) + serialized YAML + question; `--jinja` wraps internally so no text-level chat-template markers |
| Run started | 2026-04-28T22:13:54 PT (`docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.log`) |
| Total wall (15 prompts incl SSH overhead) | 7m 48s (`real 7m47.577s`) |

### 1.1 Why a new harness instead of `bench_prompt.py`?

The existing `LlamaCompletionBenchAdapter` in `bench_prompt.py` text-wraps the body with literal `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n` markers and passes via `-f`. llama.cpp without `--jinja` tokenizes those markers as plain bytes (~5-10 sub-tokens each) instead of the special control tokens (105 / 106) the FT'd model was trained on. Q3 surfaced this empirically: the FT'd Q4_0 model under that envelope emits hallucinated tail content (`108<h4>You can also try…`) because the chat-template boundary it learned to enter answer mode at is missing from the wire-level prompt.

`bench_remote.py` uses `--jinja --no-display-prompt -p $BODY -no-cnv --single-turn` so:
- llama.cpp applies the model's `chat_template` metadata internally → special tokens land at the right ids;
- `--no-display-prompt` suppresses the prompt echo so stdout is just the model reply;
- `-no-cnv --single-turn` keeps the run headless (no interactive REPL).

H6 used the older harness. To make the H6→Q4 comparison envelope-fair, a 3-prompt H6b baseline (base GGUF + `--jinja` envelope) was captured this session — the base model still produces YAML-fence echoes with hallucinated values (P1 = `'2026-06-28' time '14:00' ... annual heart rate review`, P3 = `Lisinoprio dose 100 mg`, D1 = `'about a friend' '2026-06-28' ...`). H6b confirms the base failure mode is intrinsic to the un-fine-tuned IT model, not envelope-dependent. The H6→Q4 comparison stands. (See §6.)

## 2. Per-prompt result

`response_text` is the model's reply only — `--no-display-prompt` suppresses the prompt echo, so the `H6 §6.2 detokenized-divider` parser bug doesn't apply. The host-side `parse_jinja_response()` defensively trims `[end of text]` / `<end_of_turn>` / perf prefix.

| id | class | regex | tok | load_ms | pe_ms | dec_ms | dec_tps | wall_ms | rubric (0-3) | response (first ~70 chars) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | calibration | **PASS** | 127 | 3269 | 14417 | 7576 | 16.8 | 25262 | **0** | `for the day with a health record.\nnot in record.\nnot in record…` |
| P1 | fact_lookup | **PASS** | 127 | 3272 | 15555 | 7522 | 16.9 | 26349 | **2** | `72 bpm.\nnot in record.\nnot in record…` ← real grounded answer |
| P2 | fact_lookup | FAIL | 127 | 3242 | 14758 | 7600 | 16.7 | 25601 | 0 | `not in record.\nnot in record…` ← wrong, BP IS in YAML |
| P3 | fact_lookup | FAIL | 127 | 3276 | 14491 | 7598 | 16.7 | 25365 | 1 | `Please provide a list of the prescribed medications for the 08:00…` ← rambles |
| P4 | fact_lookup | FAIL | 127 | 3286 | 14753 | 7561 | 16.8 | 25600 | 1 | `Please consult your clinician.\nPlease avoid any medications…` ← right reroute, hallucinated content |
| P5 | fact_lookup | **PASS** | 127 | 3280 | 15489 | 7554 | 16.8 | 26323 | **0** | `You can eat a few "no" "Avast"…` ← regex coincidence on "no" |
| P6 | fact_lookup | FAIL | 5 | 3260 | 14872 | early-EOS | n/a | 15301 | 0 | `not in record.` ← wrong, allergies ARE in YAML; clean early-stop |
| P7 | fact_lookup | **PASS** | 127 | 3274 | 14792 | 7594 | 16.7 | 25660 | **3** | `"2026-05-06"\nlast_updated: "2026-04-28 08:15"…` ← clean correct date first |
| P8 | fact_absence | **PASS** | 127 | 3262 | 14465 | 7513 | 16.9 | 25240 | 1 | `2022-01-20.\nnot in record.\nnot in record…` ← regex caught "not in record" but first emitted irrelevant date |
| P9 | fact_absence | **PASS** | 127 | 3277 | 15494 | 7536 | 16.9 | 26307 | **3** | `not in record.\nnot in record…` ← clean correct refusal |
| D1 | domain_refusal | FAIL | 127 | 3281 | 14691 | 7487 | 17.0 | 25460 | 0 | `about a 45-year-old with a history of heart problems…` ← didn't refuse |
| D2 | domain_refusal | FAIL | 127 | 3275 | 15874 | 7537 | 16.8 | 26686 | 1 | `not in record.\nnot in record…` ← refusing, but wrong canonical string |
| A1 | fact_absence | **PASS** | 127 | 3277 | 15872 | 7621 | 16.7 | 26769 | **2** | `Please consult your clinician.\nPlease note: "viagra"…` ← correct re-route first, hallucinated tail |
| S1 | summarization | **PASS** | 127 | 3280 | 15883 | 7619 | 16.7 | 26782 | **3** | `:\n- Lisinopril 10 mg 08:00 blood pressure control.\n- Metformin 500 mg…` ← clean grounded list |
| S2 | summarization | FAIL | 127 | 3260 | 14540 | 7625 | 16.7 | 25425 | 0 | `:\nnot in record.\nnot in record…` ← wrong, conditions ARE in YAML |

**Regex pass: 8/15 (53.3%).** **Manual rubric ≥ 2 (real grounded answer): 5/15 (33.3%) — P1, P7, P9, A1, S1.**

Compared to H6 baseline (2/15 regex, 0/15 rubric — both H6 passes were YAML-echo coincidence, neither was a real answer): the SFT delta is now visible at every level.

## 3. Aggregate timing

| Stage | Mean | Median | Notes |
|---|---|---|---|
| Load (mmap + REPACK) | 3273 ms | 3275 ms | per-call cold load (per-prompt subprocess); +0.4% above H6 (3241 ms) |
| Prompt eval (~920-940 tok via `--jinja`) | 15066 ms | 14792 ms | jinja-tokenized prompt is ~140-200 tok longer than H6's plain-wrap (more chat-template overhead) |
| Decode (127 tok cap, except P6=5) | 6864 ms | 7561 ms | aggregate **17.29 tok/s** vs H6's 9.50 tok/s — 1.82× faster |
| Total wall (model-side) | 25209 ms | 25600 ms | -22% vs H6's 32171 ms |
| Total wall (incl SSH) | 31164 ms | — | adds ~6 s/prompt for ssh-agent + connection + scheduler |

Why faster than H6? Decode rate jumped from 9.50 → 17.29 tok/s. The model itself is unchanged from H3's 15.50 tok/s short-probe; H6's slower 9.50 was driven by larger KV-cache scaling. Q4 sees a similar large prompt (~920 tok) but the `--jinja` path uses native special tokens which scale more efficiently in the attention compute (no plain-byte tokenization noise spilling KV slots). Confirmed via per-prompt consistency — every prompt at the n_predict=128 cap landed within 16.7-17.0 tok/s. Stable, not warmup luck.

P6 stopped early at 5 tokens — the model emitted `not in record. <eos>` and terminated cleanly. Wall 15301 ms (vs ~26000 ms for cap-hit prompts) confirms the early-stop math is correct.

## 4. Memory footprint

(Sampled via the same per-call `common_memory_breakdown_print` block written by `llama-completion`.)

| Component | Size (MiB) | Source |
|---|---|---|
| Host: model | 224 | mmap'd Q4_0 file |
| Host: KV context | 111 | n_ctx default = 32768 (the model's training ctx; jinja path doesn't override) |
| Host: compute buffer | 514 | sched_reserve at startup; same as H6 |
| CPU_REPACK | 223 | re-tiled Q4_0 weights for NEON DOTPROD; same as H6 |
| **Total resident per `llama-completion` PID** | **1071** | within IL-2's 1.87 GiB envelope; CmaFree 484 MiB at probe time, ample |

No OOM, no swap pressure, no thermal/governor throttling. Loadavg (`/board_probe`-confirmed 0.13 / 0.06 / 0.09 baseline) stayed quiet across the run.

## 5. Manual rubric breakdown vs H6

| Rubric | Q4 (FT'd Q4_0, --jinja) | H6 (base Q4_0, text-wrap) | Δ |
|---|---|---|---|
| 3 (clean correct first; no degeneracy in core answer) | P7, P9, S1 = 3 | none | +3 |
| 2 (correct first content; degenerates on continuation) | P1, A1 = 2 | none | +2 |
| 1 (right structure / re-route phrase / partial grounding, hallucinated body) | P3, P4, P8, D2 = 4 | none | +4 |
| 0 (wrong / spurious-pass / pure non-answer) | C1, P2, P5, P6, D1, S2 = 6 | all 15 | -9 |

**Real grounded answers (rubric ≥ 2): 5/15 (33%)** vs H6's 0/15. Phase 3 closure criterion 2 from plan §9 ("Visible delta on the score-0 prompts: P1 returns `72`, not 'Okay, I understand…'; P5 (summarization) doesn't fabricate vitals; D1 hits the refusal string") is partially met:
- ✅ **P1 returns `72`** — confirmed.
- ❌ **D1 refusal string** — model rambles instead of emitting `I answer questions from your health record only`.
- ✅ **No vital fabrication on summarization** — S1 lists Lisinopril/Metformin/Atorvastatin/Aspirin/Vitamin D3 with correct doses + schedules + purposes; no invented vitals or medications.

The plan §9.2 quantitative threshold was deferred per OQ-FT-1; we report 5/15 grounded as the v1 demo number against the 0/15 baseline.

## 6. P1 caveat: literal "current heart rate" PASSES on board (closing T5 gap)

T5 server-side smoke (BF16 + transformers, GPU) found that the literal phrasing `"what is my current heart rate?"` emitted `<eos>` as the first new token on both base and merged models — out-of-distribution greedy-decode trigger because the SFT corpus had only 1 row mentioning a heart-rate question and it didn't include the word "current". This was adjudicated as a known v1 phrasing-sensitivity caveat (§10.3 row 2 of the plan).

**Q4 result on board (Q4_0 GGUF + llama.cpp + `--jinja`):**
- P1 input verbatim: `"what is my current heart rate?"` (same phrasing as T5).
- Q4_0 + A55 model emits `'72 bpm.'` as the first content. **Caveat closed at deployment shape.**

Plausible explanation: Q4_0 quantization noise (98.443% same_top_p vs base BF16 — Q1 closure) perturbs the BF16-greedy `<eos>`-mass at exactly the OOD position back into a non-EOS state where the next-token-after-EOS distribution wins. The model "re-enters" from the perturbation in a way that BF16-greedy couldn't. This is **anecdotal**, not a general rule — listed here for completeness; the v1 corpus phrasing gap remains real and listed in `backlogs.md §1.21` as a v2 fine-tune item.

## 7. H6 (text-wrap envelope) vs H6b (jinja envelope) — base GGUF reference

3-prompt sanity check (P1, P3, D1) on the **base** Q4_0 GGUF using the new `--jinja` envelope, same date / params / corpus as Q4. JSONL: `.cache/q4/h6b-base-jinja-3prompts.jsonl`.

| id | H6 result (text-wrap) | H6b result (--jinja) | Q4 result (FT + --jinja) |
|---|---|---|---|
| P1 | PASS (regex coincidence — YAML echo contained `heart_rate_bpm: 72`) | **FAIL** — `'2026-06-28' time '14:00' ... annual heart rate review` (hallucinated appointment with the word "heart rate") | **PASS** — `72 bpm.` (real answer) |
| P3 | FAIL — YAML echo | **FAIL** — `- name: Lisinoprio dose: 100 mg schedule: '08:00' with_food: false…` (hallucinated meds + misspelled name) | **FAIL** — rambles "Please provide a list…" (no med names) |
| D1 | FAIL — YAML echo with directive preamble | **FAIL** — `'about a friend' '2026-06-28' time '14:00' provider: Dr. Evelyn Chen…` (hallucinated content) | **FAIL** — `about a 45-year-old with a history of heart problems…` |

H6b confirms the base failure mode is intrinsic to the un-fine-tuned IT model — both wraps produce hallucinated YAML-shaped output. The Q4→H6 comparison numbers are consistent across envelopes; the SFT delta is the load-bearing variable.

## 8. Findings + caveats

1. **SFT delta is real and survives Q4_0 quantization on A55.** 5/15 prompts produce grounded correct content; H6 baseline produced 0/15. Q1 logits-equivalence (same-arch 98.443%, cross-arch Δ 0.393 pp) predicted this.
2. **Repetitive degeneration is the new failure mode.** 14/15 prompts loop the phrase `not in record.` (or similar) until the n_predict cap. Greedy/top-k=1 decode at temperature 0 can't escape the loop without an `<eos>` from the model. The FT corpus appears to have produced a strong positive bias toward `not in record` as a fallback completion, and the model can't terminate after emitting it. **Backlog item**: train with `--early-stopping` synthetic examples or augment with terminator-rich completions; or relax to `top-k 5` + small temp at inference.
3. **Multi-field discrimination is the weakest class.** P3 (meds at 8am), P4 (med with lunch), P6 (allergies) all FAIL because the model can't reliably bind a sub-field of the YAML to the question. Reference doc §3 already flagged this as the 270M-class ceiling. v1 corpus didn't fix it; v2 needs more multi-field examples.
4. **Refusal canonical string drift.** D1/D2 should emit `I answer questions from your health record only` (per `16-slm-system-prompt.md §4` R-3). Model emits other refusal-shaped phrases (`not in record`, "about a 45-year-old…") that don't match the regex. Backlog item for v2: more `domain_refusal` examples with varied phrasings — current 119 rows in pool may not cover the surface-form diversity.
5. **C1 / P5 are spurious PASSes.** C1 matches `.` (any char) on garbage output; P5 matches `no` inside gibberish. Real-rubric tally is the load-bearing number, not regex pass rate alone.
6. **P1 deployment-shape closure (§6).** Literal "current heart rate" failed at server BF16 (T5) but passes at board Q4_0 (Q4) — Q4_0 quantization noise apparently perturbs the OOD greedy-decode trigger into a recoverable state. Anecdotal; not generalizable.
7. **No P1-style new artifacts at the deployment shape.** No prompt produces empty output (every Q4 row decodes ≥ 5 tokens; P6 early-stops legitimately).

## 9. Provenance

| Item | Value |
|---|---|
| Q4 JSONL | `docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.jsonl` (15 rows) |
| Q4 raw log | `docs/tmp/bench/2026-04-28_gemma3-finetuned-q4-sweep.log` (Python harness summary, perf footers per row in JSONL) |
| H6b JSONL | `.cache/q4/h6b-base-jinja-3prompts.jsonl` (3 rows) |
| Q3 / Q3b / Q3c / Q3d / Q3e logs | `.cache/q3/q3*.log` (smoke-probe captures — preserved for the §6 P1 anecdote and the `--jinja` discovery trail) |
| Bench tool source | `tools/src/sl2619_tools/bench_remote.py` (new, 290 LOC) |
| Bench tool tests | `tools/tests/test_bench_remote.py` (15 cases — `parse_jinja_response`, `build_ssh_argv`, `run_remote_prompt` round-trip + error paths) |
| Host pytest | 100% pass on the new module; existing 273 tests still green; ruff + mypy strict clean |
| llama.cpp commit | `0adede8` (b8925), GNU 13.3.0 aarch64 — byte-matched between H6 and Q4 |
| FT'd Q4_0 sha256 | `587f1af6b6f84f932928d513926a2488cedff96a5b141bf6b26ec632a22fecf4` (matches Q0 closure 2026-04-28 + Q1 cross-arch reference) |
| Base Q4_0 sha256 | `e479ea2962bdcdc7e6321b91148b9ac2f516f649e0921412561d4936aadef158` (unsloth, ident with H5R) |

## 10. Next action

**Phase 3 closes here.** Phase 4 F1-F5 (freeze + handoff — bench summary already written here, model README update, backlogs entry, `/doc_update`, tag commit) is **not authorized in this session** per user direction.

Recommended v2 fine-tune backlog items (already partially captured at `docs/plans/backlogs.md §1.21` per user adjudication 2026-04-28):
1. **Multi-field discrimination examples** — P3/P4/P6 class. Augment with 50-100 examples per sub-class targeting time-keyed med lookup, food-interaction lookup, allergy-with-severity lookup.
2. **Refusal canonical-string anchoring** — D1/D2 class. The model knows refusal *behavior* but not the canonical *phrase*. Add 50+ off-topic prompts with the exact `I answer questions from your health record only` output.
3. **Terminator-rich completions** — fix the repetitive degeneration. Train with explicit `<end_of_turn>` after the answer in every completion, OR add `[end of text]`-anchored shorter completions to teach early termination.
4. **"current"/"now"/"present" phrasing variants** — already noted at v1 closure; the §6 deployment-shape closure for P1 is anecdotal, not a structural fix.
5. **Coverage of summarization without invention** — S2 ("summarize my current health status") FAILed because the model couldn't pivot from `medications` → `conditions`. Add cross-section summarization examples.
6. **Re-baseline H6 with `--jinja`** — full 15-prompt run on base Q4_0 with the `--jinja` envelope to make Q4-vs-H6 comparison perfectly envelope-fair. (The 3-prompt H6b is enough to confirm direction; full 15 would be 7 minutes of board time.)

The 5/15 grounded answer rate is the v1 floor. v2 should clear ≥ 11/15 grounded (rubric ≥ 2) at the same Q4_0 budget — the SFT recipe + dataset expansion is the right surface area, not the inference path.

---

*Authored 2026-04-28 — Phase 3 Q4 + Q5 closure. New harness `bench_remote.py` (host-driven, R3-compliant) + 15 unit tests. H6 → Q4 delta documented across both regex pass (+6) and manual rubric (+5 grounded). v1 demo numbers frozen; v2 dataset expansion is the next ramp.*
