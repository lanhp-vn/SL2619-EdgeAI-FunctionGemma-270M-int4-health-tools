# Distil iteration recipe + lessons for our own SFT

> **Source**: 2026-05-01 FunctionGemma feasibility run (model ID
> `231feebb-8cc0-4d5f-9e4b-4d2f00e362b2`). Three teacher-eval iterations
> took the headline LLM-as-a-Judge score from 0.7917 → 0.8750 → 0.9583
> on a 24-row test. This document distills (pun intended) the working
> recipe and the transferable lessons for our local
> `scripts/finetune.py` / FunctionGemma SFT track.
>
> Companion artifacts:
> - `distil_functiongemma_iteration_001/teacher-eval-analysis.md` — per-iteration narrative + per-row diffs
> - Plugin: `~/.claude/plugins/marketplaces/distil-cli-skill/` (mirrors `docs/references/upstream/distil-cli-skill/`)

## 1 — The canonical pipeline (executable recipe)

```bash
# ── Phase 0: install + auth ────────────────────────────────────────────
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh   # first time
distil update                                                  # subsequently
distil login
distil whoami --output json   # confirm

# ── Phase 1: shape data ────────────────────────────────────────────────
mkdir -p iteration_001
# Drop in: README.md, job_description.json, config.yaml, train.jsonl, test.jsonl
# (Distil floor: ≥ 20 rows each in train + test; minimum 1 example per tool)

# ── Phase 2: feasibility (zero-credit) ─────────────────────────────────
distil model create my-feasibility-name
# Capture the model ID from output → save as $MID
distil model upload-data $MID --data ./iteration_001 --dry-run
# --dry-run validates schema + cross-set duplicates without consuming quota.
# This is THE keystone test — never skip it.

# ── Phase 3: real upload + teacher eval ────────────────────────────────
old_upload=$(distil model show $MID --output json | jq -r '.upload_ids[0] // "none"')
distil model upload-data $MID --data ./iteration_001
new_upload=$(distil model show $MID --output json | jq -r '.upload_ids[0]')
[ "$new_upload" = "$old_upload" ] && { echo "ERROR: upload didn't take"; exit 1; }
distil model upload-status $MID --output json | jq '.status'   # expect JOB_SUCCESS

distil model run-teacher-evaluation $MID
# Poll (canonical pattern from polling-jobs.md):
while true; do
  status=$(distil model teacher-evaluation $MID --output json | jq -r '.status')
  echo "$(date +%H:%M:%S) status=$status"
  case "$status" in JOB_SUCCESS|JOB_FAILURE|JOB_STOPPED) break ;; esac
  sleep 60   # teacher eval is minutes-scale
done
distil model teacher-evaluation $MID --output json | jq '.results'
distil model download-teacher-evaluation-predictions $MID \
  --file-name iteration_001/teacher-predictions.jsonl

# ── Phase 4: iterate (only if needed) ──────────────────────────────────
# Decision rule (from teacher-evaluation.md):
#   judge ≥ 0.80  → high-confidence PROCEED to training
#   judge ≥ 0.70  → PROCEED (acceptable)
#   judge < 0.70  → ITERATE (don't train yet)
# Repeat Phase 3 after each lever pull.

# ── Phase 5: train ─────────────────────────────────────────────────────
distil model run-training $MID
# Poll (sleep 600 — hours-scale):
while true; do
  status=$(distil model training $MID --output json | jq -r '.status')
  case "$status" in JOB_SUCCESS|JOB_FAILURE|JOB_STOPPED) break ;; esac
  sleep 600
done

# ── Phase 6: download + deploy ─────────────────────────────────────────
distil model download $MID
distil model deploy local $MID
distil model invoke $MID    # prints the curl command
```

Three pitfalls this exact sequence avoids: (1) consuming quota on a malformed
dataset (`--dry-run` catches it), (2) re-running teacher eval against the
*old* upload because the new one silently failed (the upload-ID check
catches it), (3) the polling loop picking the wrong status string because
human-readable text changes between CLI versions (`--output json | jq`
catches it).

## 2 — Iteration discipline (extracted from this run)

| rule | what we did | what we learned |
|---|---|---|
| **One lever per iteration** | v2 violated this (touched both task_description AND judge_instructions). v3 followed it (task_description only). | The §8 retro showed task_description carried *all* of v2's lift — judge edits were dead weight. Discipline matters because without it, attribution breaks. |
| **Capture old upload ID before re-upload** | Done in v3. | Distil's CLI silently accepts re-uploads but doesn't warn if the new upload was rejected. Confirming `new_upload != old_upload` is the only way to be sure your changes took. |
| **Token-burn awareness at iter #3+** | Surfaced explicitly to user before kicking off v3. | Each iteration costs uncertain credit + Claude tokens. Confirm the lever plan with concrete lift estimate (§9.2 table) before spending. |
| **Lever order: 1 (job_desc) → 2 (data) → 3 (synthgen) → 4 (teacher swap)** | Only Lever 1 was needed across v1–v3. | Most iterations end at Lever 1. Don't reach for teacher swaps before exhausting prompt clarity. |
| **Test rows are eval-only, not training data** | Recognized in §9.3. | Fixing a test miss raises the headline metric but does NOT directly improve the trained student unless the same fix lands in `train.jsonl`. Most "test failures" are diagnostic, not actionable. |

## 3 — Job-description writing patterns that actually moved the needle

### 3.1 Hard MUST/NEVER framing beats soft "should"

v1 said: *"Map the user request to whichever single tool best satisfies it."* — vague, teacher refused on edge cases.
v2 said: *"You MUST emit exactly one function call. Never reply in natural language, never refuse, never explain that the data is unavailable."* — drops EMPTY-prediction rate by half.

### 3.2 Numbered, ordered, "first match wins" rules

v1 had inline guidance scattered through the description. v2 rewrote as 7 explicit `ROUTING RULES (apply in order; the first match wins)`. The teacher follows numbered lists more reliably than narrative prose.

### 3.3 Worked examples beat verbose rules

v2 ROUTING RULE #3 said: *"Do NOT skip the call when the user phrases it as 'Do I have any allergies?'"* — verbatim quote of the failing row, still failed.
v3 ROUTING RULE #3 added a **worked-examples block**:
```
WORKED EXAMPLES — emit list_allergies() for ALL of these surface forms:
  - User: 'Do I have any allergies?' → list_allergies()
  - User: 'Am I allergic to anything?' → list_allergies()
  - User: 'What allergies do I have?' → list_allergies()
  - User: 'How bad is my shellfish allergy?' → list_allergies()
```
Worked examples didn't fix that *one* persistent row (gpt-oss-120b has a hard prior on yes/no allergy phrasing), BUT the same pattern in RULE #7 fixed the "A pills" row deterministically.

**Pattern**: when a rule isn't biting, escalate from "do X" to "User: 'literal failing phrasing' → tool_call(literal_args)" examples. Each worked example is worth ~5 lines of prose for LLM compliance.

### 3.4 Strip ambiguity in compound rules

v2 RULE #7 said: *"Use the user's literal medication phrasing, including ambiguous prefixes like 'A' or 'at'."* — self-contradictory: "literal" + "ambiguous prefixes" don't combine well. Teacher emitted `name="A pills"` (took it literally including the noun).

v3 RULE #7 split it:
```
Extract ONLY the medication token; STRIP generic medication-class nouns
('pill', 'pills', 'tablet', 'tablets', 'capsule', 'capsules', 'med',
'meds', 'medication', 'medications', 'drug', 'drugs') from the name argument.
```
Plus 6 worked examples. The teacher emitted `name="A"` on the next run.

**Pattern**: if your rule has two clauses that conflict in any case, the teacher will pick the wrong one ~half the time. Audit your rules for self-contradiction *before* blaming sampling noise.

### 3.5 Be explicit about empty parameters

```
For zero-parameter tools (get_vitals, list_allergies, get_next_appointment,
get_emergency_contact), parameters MUST be the empty object {} — even when
the user provides extra context like a provider name, severity, or specific value.
```

This single sentence fixed cluster B (Dr. Chen → `get_next_appointment()`) without further work.

## 4 — Judge-instruction patterns

### 4.1 Special-case rules with concrete examples

Generic "deeply equals after lowercasing" wasn't enough. v2 added 4 special cases:

```
(b) For get_medication_by_name(name=...): treat predicted name as equivalent
    to gold name if EITHER (i) they match exactly after lowercasing/trimming,
    OR (ii) one is a case-insensitive prefix of the other and would resolve
    to the same medication under a case-insensitive prefix lookup.
    Examples that should be marked GOOD: gold='at' / pred='Atorvastatin'
    (prefix resolves to a unique medication); gold='Atorvastatin' /
    pred='atorvastatin' (case folds match).
    Examples that should be marked BAD: gold='atorvastatin' / pred='metformin'.
```

The judge follows the **examples** more reliably than the prose. Always include both GOOD and BAD examples for each special case — otherwise the judge guesses.

### 4.2 Simulate the judge locally before chasing "judge bugs"

In §8 we labeled "Check my A pills." as a "judge bug". The §9.1 what-if simulator (`/tmp/distil_validate.py`-adjacent script) showed the judge was actually correct: under prefix-resolves-same lookup, "A pills" doesn't resolve to anything while "A" resolves to multiple meds — different result sets ⇒ rule (b) correctly says BAD.

**Pattern**: before adding "judge fix" iterations, write a local simulator that applies your judge rules to (gold, pred) tuples. If the simulator marks the case BAD, your model is wrong, not your judge. Saves credit-bearing iterations.

## 5 — Dataset / seed design lessons

| lesson | observation |
|---|---|
| **20-row floor per set is the hard minimum** | Our 24-row test cleared it. Below 20, Distil rejects upload. Plan for ≥ 30 if you can — single-row swing on 24 = 4.17 points. |
| **All tools must appear in both sets** | We had all 7 tools in both train (50) and test (24). Per `multi-turn-tool-calling.md` "Include examples for all tools." Otherwise the trained student won't see the underrepresented tool's pattern. |
| **Cross-set exact duplicates fail upload** | Our first dry-run was rejected on 1 cross-set dup. Distil's de-dup is **exact-string match** on (question, answer) tuples — not semantic. |
| **Within-set exact duplicates waste slots silently** | Our first audit found 3 within-train dups (lines 11/43, 12/44, 18/50). Distil doesn't reject them but they consume training slots. Run `/tmp/distil_validate.py` (or equivalent) on every dataset before upload. |
| **Per-tool seed phrasing variety drives synth diversity** | We had `list_allergies × 4` seeds with diverse phrasings ("What allergies do I have?", "Do I have any known allergies?", "How bad is my shellfish allergy?", "List the allergies on my chart."). The synth corpus inherits seed phrasing distribution; mono-phrased seeds make the trained student brittle. |
| **Refusals don't fit Distil's contract** | `medical_advice_refusal` and `off_topic_refusal` (no-tool replies) cannot be encoded in the multi-turn-tool-calling schema (one tool call per turn, no slot for `[]`). Either build a synthetic `report_outside_scope(reason)` tool (mirror distil-labs' `intent_unclear` pattern) or keep refusals on the local SFT path. |
| **Parallel calls don't fit either** | Distil multi-turn allows exactly one tool call per assistant turn. Reshape parallel calls as sequential turns OR keep them local. |

## 6 — Eval-metric discipline

| metric | when it's primary |
|---|---|
| **LLM-as-a-Judge** | Tools with free-text args (med name strings, food strings). Use this when args can have multiple valid surface forms. |
| **tool_call_equivalence (TCE)** | Tools with constrained args (enums, IDs, booleans, time strings). Strict exact-match — under-counts when pred is semantically right but lexically different. |
| **binary_tool_call** | "Did the model emit *any* tool call?" — useful as a refusal-rate proxy. EMPTY predictions drop both binary and judge to 0. |
| **staged_tool_call** | Multi-turn case: scores partial credit when call name is right but args are partially wrong. Worth 0.75 vs 0 in the "A pills" case. |
| **ROUGE** | Surface-token overlap on the stringified call. Useful for detecting formatting drift but rarely the decision metric. |

**Patterns**:

- All five metrics aligning at the same value (0.9583 in v3) = robust score, not a metric-policy artifact.
- Big judge ↔ TCE divergence = either free-text args or judge instructions are too lenient/strict.
- **EMPTY pred = teacher-side error.** No judge rule rescues it. Fix is either Lever 1 (job desc) or Lever 4 (teacher swap), not Lever 1 judge.
- 24-row test → 4.17pt per row. Below ≈ 4-row swing, treat as noise. Above, suspect a real pattern.

## 7 — What the trained student inherits (vs. what it doesn't)

```
                                                 ┌─ inherited ─┐
job_description.task_description ────────────────┤             │
job_description.tools (registry) ────────────────┤             │
config.yaml (synthgen knobs) ────────────────────┤             │
train.jsonl (50 rows) ──── synth corpus ─── 5K ──┘             │
                                                               ▼
                                                       Trained student
                                                               ▲
                                                 ┌─ NOT inherited ─┐
test.jsonl (24 rows) ──── teacher eval only ─────┤                 │
job_description.llm_as_a_judge_instructions ─────┘                 │
                          (used for scoring, not for training)
```

**Practical implication for any iteration decision**: if the change is in `train.jsonl` or `task_description`, it improves the student. If the change is in `test.jsonl` or `llm_as_a_judge_instructions`, it only changes the headline score. The cost of an iteration is the same either way.

## 8 — Take-homes for our local FunctionGemma SFT (`scripts/finetune.py`)

This is the bridge from "Distil platform" to our own SFT track. None of these require the platform; they apply directly to `data/functiongemma/dataset_v1/train.jsonl` and `scripts/finetune.py`.

### 8.1 Adopt the routing-rules prompt pattern in our system prompt

Our local SFT system prompt was generic. Distil's iteration showed that
**numbered ROUTING RULES with first-match-wins semantics** + worked
examples per rule beat narrative descriptions by +0.17 on the same
underlying model family. Our local `prompts.py` should mirror this
shape. The §3.1–3.5 patterns in this doc are the template.

### 8.2 Strip-noun rule for medication arguments

The "A pills" → "A" failure mode is real — gpt-oss-120b had it; FG-270M will
have it too. Add a strip-noun preprocessing rule to our prompt or a
postprocessing step in inference. Either works; preprocessing is cleaner.

### 8.3 Worked-examples block for cluster-B-style queries

Where our local F1+F5 evaluation shows fact_absence misroutes (lab values
to wrong tool), add WORKED EXAMPLES blocks rather than more prose. The
`task_description` gain from v2 → v3 (~ +8pt) came from worked examples,
not more rules.

### 8.4 Validate dataset with the same script before every SFT epoch

`/tmp/distil_validate.py` (or an `src/`-housed equivalent) catches:
- Cross-set duplicates (don't train on test rows)
- Within-set duplicates (waste capacity)
- Tool-name typos against the registry
- Required-arg violations
- Min-row floors

We have `scripts/pre-commit-functiongemma.py` for PHI; add a sibling for
shape/dedup. Cheap insurance against the kind of failure mode our v1
dry-run caught (and the M5 deep-dive that landed `dataset is the
bottleneck`).

### 8.5 24 rows is too small a test set for absolute-score decisions

Our local eval holdout is similarly small. Future evals should treat
single-row swings as noise; only 4-row+ pattern shifts indicate a real
regression. This is the antidote to chasing test misses that are just
sampling artifacts.

### 8.6 Test rows are diagnostic, not training fuel

Most of our local "test failures" surfaced in Block F1 / M5 are best
addressed by adding *more train rows* targeting the failure pattern,
not by editing the eval. Mirror Distil's Lever 2 discipline.

### 8.7 Refusals + parallel calls stay on the local SFT path forever

Distil's contract excludes both. If our local SFT has an architectural
advantage over Distil, it's exactly here: we can train on rows that
have `[]` tool calls (refusals) and parallel `tool_calls` arrays. The
F1 weight=2.0 + F5 fa-supplement strategy is doing real work that
Distil structurally can't match.

## 9 — What's still open (carry forward)

| OQ | description |
|---|---|
| OQ-D2 | Whether `run-teacher-evaluation` consumes a free training run. CLI doesn't expose quota meter. v1+v2+v3 all ran; if each costs a run, we'd be over budget. Confirm via dashboard. |
| OQ-D7 | Distil's chat template inside the trained checkpoint may not match our local prompt-template-as-contract. Inspect `chat_template.jinja` after `distil model download`. |
| OQ-D10 | "Do I have any allergies?" persistent EMPTY in `gpt-oss-120b`. Lever-resistant; would need teacher swap (Lever 4). Watch the trained student's allergy accuracy. |
| OQ-D11 | Iteration #3 was a one-shot; iteration #4 with a teacher swap is unbudgeted. |

## 10 — Run state at this checkpoint

- Model `231feebb-8cc0-4d5f-9e4b-4d2f00e362b2` is in `JOB_RUNNING` for training (started 2026-05-02 00:25 UTC, training ID `c9d34596-ee7a-4e56-be2b-254159fe7796`).
- Three uploads (`b4ff74fa…`, `fe8de9a2…`, `23532bf3…`).
- Three teacher evals (`c6a6ffd0…`, `635489b8…`, `14a00a0a…`).
- Best teacher score: judge=0.9583 across all 5 metrics on 24-row test.
- Predicted student floor: ~0.91 (Distil docs: student lands within ~0.05 of teacher).
- Next analysis after training: `iteration_001/training-analysis.md` per `references/tasks/analyze-predictions.md` Training Analysis Report template.
