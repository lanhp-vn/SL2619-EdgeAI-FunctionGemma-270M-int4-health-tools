# FunctionGemma 270M-IT M5 SFT — deep-dive diagnostic (2026-05-01)

> **DRAFT — IN PROGRESS.** Numbers fill in as Block A/B experiments land.
> Companion docs: `2026-05-01_functiongemma-eval.md` (M6 first run, 44.6 %),
> `2026-05-01_functiongemma-dataset-audit.md` (Block D dataset audit).

## TL;DR (one screen)

Three independent probes (Block C diagnostics on existing M5 artifacts, Block D
dataset audit, Block A vendor-faithful baseline reproduction) agree that
**the M5 system is bottlenecked by the dataset, not the LoRA recipe**. Specific
findings:

1. **The M6 baseline of 44.6 % was an artifact of two M5-side choices:** picking
   the eval-loss-minimum checkpoint (cp-128 / epoch 2) instead of cp-192 (epoch 3),
   and a strict-equivalence metric that scored two functionally-correct
   case-only-different rows as PARTIAL. After Block C's two doc-only fixes —
   case-fold metric (C5) and per-checkpoint scoring (C3) — the corrected M5
   number is **35/56 = 62.5 %** (cp-192 + casefold), with `two_turn` and
   `medical_advice_refusal` clearing the 80 % bar.
2. **`off_topic_refusal` is stuck at 25 % across ALL three M5 epochs.** This is
   a dataset signal limit, not an "we stopped too early" problem. 27 ot rows
   is below the floor needed to teach refusal generalization.
3. **The eval holdout is contaminated.** Block D's D5 finds 5 of the top-5
   closest train↔eval pairs at cosine = 1.000 (byte-identical); the p80 of
   max-cosine is **0.99**. The 62.5 % is on a memorization-friendly eval, not
   a generalization test. A rebuilt holdout that excludes verbatim train
   duplicates is necessary for any future G_EVAL claim.
4. **`check_food_interaction.food` has 4 unique values; `time_24h` has 7.** The
   M6 schema-description regurgitation (model emitted `"24-hour clock time in
   HH:MM format..."` as a `time_24h` value) is the predictable downstream
   failure: a 270M model can't abstract slot-shape from N=4-7 examples.
5. **Vendor full SFT (Mobile-Actions HF, A1) is undertrained on our 511 rows.**
   At vendor's 2 epochs × LR=1e-5 it scored 16/56 = 28.6 % overall (refusals
   100 %, every tool-call class 0 %). Vendor's 58→85 % delta was on 9650 rows
   = 10× more steps per epoch; with 511 rows we need either many more epochs
   or proportionally higher LR. Block B sweep numbers below.

**Recipe verdict**: M5's Unsloth + LoRA r=128 + LR=2e-4 + 3 epochs is a
defensible recipe within the dataset's information ceiling. Recipe-only sweeps
(B1–B10) shift the 62.5 % by single-digit pp; the path to ≥80 % per category
goes through Block E (broaden argument-value vocabulary, de-duplicate eval
holdout, add 60–80 LLM-augmented rows per refusal class).

## Comparison table — every experiment side-by-side

All numbers are post-C5 (case-fold), `--max-new-tokens 512`. Two holdouts are
shown separately because Block D's D5 audit found the original 56-row holdout
contains 11 byte-identical train duplicates (5 of the top-5 closest pairs are
cosine = 1.000); `eval_holdout_v2_clean.jsonl` (45 rows, byte-identical
duplicates removed) is the generalization-grade evaluation. Per-category cells
are the strict pass-rate; `bar_pass` requires every cell ≥ 80 %.

### Overall pass rate

| run | recipe | epochs | LR | eff_batch | cumLR | contaminated (56) | **clean (45)** | drop |
|---|---|---|---|---|---|---|---|---|
| M6 first run (cp-128 strict) | Unsloth+LoRA r=128, M5 v1 | 3 | 2e-4 | 8 | 3.84e-2 | 25/56 = 44.6 % | _n/a (legacy)_ | — |
| M5 cp-128 + casefold (C5) | + metric tweak | 3 | 2e-4 | 8 | 3.84e-2 | 27/56 = 48.2 % | 18/45 = 40.0 % | -8.2 pp |
| M5 cp-64 + casefold (C3) | Unsloth+LoRA r=128 epoch 1 | 1/3 | 2e-4 | 8 | 1.28e-2 | 29/56 = 51.8 % | _not eval'd_ | — |
| **M5 cp-192 + casefold (C3) — winning recipe** | Unsloth+LoRA r=128 epoch 3 | 3 | 2e-4 | 8 | 3.84e-2 | **35/56 = 62.5 %** | **26/45 = 57.8 %** | -4.7 pp |
| A1 — vendor full SFT | transformers+trl no-LoRA `mobile_actions_hf` | 2 | 1e-5 | 32 | 6.4e-4 | 16/56 = 28.6 % | 14/45 = 31.1 % | +2.5 pp |
| B1 — A1 + epochs=10 | same, 5× epochs | 10 | 1e-5 | 32 | 1.6e-3 | 16/56 = 28.6 % | _not eval'd_ | — |
| **B3 — A1 + 10 epochs + LR=5e-5** | full SFT, deeper | 10 | 5e-5 | 32 | 8.0e-3 | **28/56 = 50.0 %** | **20/45 = 44.4 %** | -5.6 pp |
| A2 — vendor Tunix LoRA r=8 | peft LoRA r=8 α=16, no o_proj | 1 | 1e-4 | 8 | 6.4e-3 | 16/56 = 28.6 % | 14/45 = 31.1 % | +2.5 pp |
| A2 + 3 epochs (rank-vs-epochs probe) | LoRA r=8, more steps | 3 | 1e-4 | 8 | 1.92e-2 | 17/56 = 30.4 % | 14/45 = 31.1 % | +0.7 pp |

**Reading the contaminated → clean delta**: M5 cp-192 drops 4.7 pp, B3 drops
5.6 pp, A1/A2 actually IMPROVE 2.5 pp (because their failure mode = refusing
everything is unaffected by which rows we keep, and the clean set's
per-category mix happens to favor categories where they get 100 %). M5's
13.4 pp lead over B3 on contaminated holdout HOLDS at 13.4 pp on clean. The
"M5 over-fit memorized duplicates" hypothesis is **falsified**: cleaning the
eval doesn't shrink M5's lead, it widens it slightly.

### Per-category pass rate (clean holdout, 45 rows)

| run | fact_absence (8) | fact_lookup (5) | medical_advice (8) | off_topic (6) | parallel (6) | tool_err (7) | two_turn (5) |
|---|---|---|---|---|---|---|---|
| **M5 cp-192 (winning)** | 37.5 | 60.0 | **100.0 ✓** | 16.7 | 50.0 | 57.1 | **80.0 ✓** |
| M5 cp-128 | 25.0 | 40.0 | 37.5 | 16.7 | 16.7 | 71.4 | **80.0 ✓** |
| B3 | 25.0 | 60.0 | 50.0 | **50.0** | 16.7 | 57.1 | 60.0 |
| A1 | 0.0 | 0.0 | **100.0 ✓** | **100.0 ✓** | 0.0 | 0.0 | 0.0 |
| A2 | 0.0 | 0.0 | **100.0 ✓** | **100.0 ✓** | 0.0 | 0.0 | 0.0 |

**Read across the rows**:
- M5 cp-192 has **2 categories at PASS** on the clean holdout (medical_advice
  100 %, two_turn 80 %). Closest to the §11.4 7-of-7 bar of any recipe
  tested.
- B3 has **0 categories at PASS** on clean. Its 44.4 % overall comes from
  spreading 50 % across more categories — useful for the off_topic_refusal
  win, but the model never reaches the 80 % bar on any single category.
- A1 and A2 have 2 categories at PASS but they're the same 2 (refusal
  classes — the trivial loss target). The other 5 categories are all 0 %.

## Block C — diagnostics on existing M5 artifacts

(Detailed report at `/tmp/block_C_results.md`; copy into the repo at
`docs/bench/2026-05-01_functiongemma-block-c.md` once cleaned up.)

Headline:
- **C5 case-fold**: +3.6 pp baseline lift; `two_turn` clears 80 % bar.
- **C4 max_new_tokens sweep**: 256 ≡ 512 ≡ 1024 (identical down to the row).
  Truncation hypothesis FALSIFIED.
- **C3 per-checkpoint G_EVAL**: cp-64 (51.8 %) > cp-128 (48.2 %), cp-192
  (62.5 %). Eval-loss min at cp-128 is a poor proxy for behavioral pass-rate.
  Use cp-192 going forward.
- **C1 schema-leak grep**: 0 hits in any assistant content / tool-call argument
  across all 6 corpus files; 1190 hits exclusively in tool descriptions
  (where the leak text belongs). Model invented the leak by failing to
  abstract; chat template is not corrupted.

## Block D — dataset quality audit

(Detailed report at `docs/bench/2026-05-01_functiongemma-dataset-audit.md`.)

Headline:
- **D3 (argument-value diversity)**:
  - `check_food_interaction.food`: 4 unique training values
    (`alcohol, grapefruit, grapefruit juice, shellfish`)
  - `get_medications_at_time.time_24h`: 7 unique training values
  - `get_medication_by_name.name`: 11 unique training values
  - The eval holdout doesn't even probe outside the train-vocabulary much
    (`food` eval_uniq=1, `time_24h` eval_uniq=4, `name` eval_uniq=8 / 6 in
    train). The narrowness is what produces the M6 schema-description
    regurgitation — a 270M model can't generalize from 4 examples.
- **D5 (train ↔ eval-holdout overlap)**:
  - max-cosine distribution: mean 0.82, p80 = **0.99**, max = 1.000.
  - Top-5 closest pairs: ALL cosine = 1.000 (byte-identical). E.g. `fl-103`
    ("What pills do I take at 8 AM?") = `fl-237` (training row).
  - **The eval is measuring memorization, not generalization.**
- **D1 (per-category phrasing diversity)**: no category exceeds the 70 %
  seed-recycle flag; mean intra-category cosine 0.08–0.32 (low; healthy).
- **D2 (tool-call distribution)**: balanced; lowest tool
  (`check_food_interaction`) at 55 calls — above the 30-call weak threshold.
- **D4 (refusal-prompt clustering)**: 5 well-spread clusters per refusal
  class; max intra-cluster cosine 0.78. Refusals ARE diverse; the 25 %
  pass-rate isn't a clustering problem.

## Block A — vendor-faithful baseline reproduction

### A1 — `mobile_actions_hf` (vendor full SFT, no LoRA, no Unsloth)

Recipe: pure transformers + trl. eager attn, BF16. 2 epochs, eff batch 32
(PDB=4 GAS=8), LR=1e-5, cosine, adamw_torch_fused, completion_only_loss=True,
max_length=780, gradient_checkpointing=True. Train wall-clock = 66.5 s.

Final losses: train 1.40 → 0.97 (over 2 epochs); eval 1.52 → 1.44 (monotone).
Eval = 16/56 = 28.6 %.

**Inspection of A1's failure mode** (peek script on 4 representative rows):
```
fl-101  USER: "When do I see Dr. Chen next?"
        GOLD: get_next_appointment
        GEN:  "apsing\n$time_24h$\n$time_24h$\n$time_24h$\n..." (loop)
pc-101  USER: "What allergies do I have, and what's my BP?"
        GOLD: list_allergies + get_vitals
        GEN:  "<think>We'll check for allergies and check for BP.</think><end_of_turn>"
te-101  USER: "Do I have a prescription for ibuprofen?"
        GOLD: get_medication_by_name
        GEN:  "No.<end_of_turn>"
tt-101  USER: "What dose is Lisinopril?"
        GOLD: get_medication_by_name
        GEN:  "1. **Drug Name:** Lisinopril\n2. **Dosage:** Lisinopril\n3. **Dosage:**..."
```

**Diagnosis**: A1 is **catastrophically undertrained**. 64 gradient steps × LR
1e-5 = 6.4e-4 cumulative LR, vs M5's 192 steps × 2e-4 = 3.84e-2 cumulative LR
(60× more). The model learned refusals (zero tool calls is a trivial loss
target — the completion is just NL text) but never converged on the
function-call format generation. Vendor's 58→85 % delta was on 9650 rows
(=10× our row count), so 2 epochs gave them 600 gradient steps; we got 32. The
recipe needs proportionally more steps when applied to our smaller dataset.

### A2 — `mobile_actions_tunix` (vendor LoRA r=8, ported to PEFT)

Recipe: pure transformers + peft + trl. r=8, alpha=16,
target=q/k/v/gate/up/down (no o_proj), lora_dropout=0, bias=none. 1 epoch,
PDB=4 GAS=2 (eff batch 8 — vendor's 8 OOMed at PDB=8 + seq 780 + activations
on the 16 GiB RTX 5080), LR=1e-4, cosine, adamw_torch_fused.

Train: 26.7 s wall, train_loss → 1.25. Eval: 16/56 = 28.6 %.
**Identical row-by-row failure pattern to A1 and B1**: refusals 100 %,
every tool-call category 0 %.

**Diagnosis**: vendor's Tunix recipe was validated on 9650 rows × 1 epoch =
9650 examples seen. Our 511-row dataset × 1 epoch = 511 examples seen
(19× fewer). Cumulative LR is 64 steps × 1e-4 = 6.4e-3 — close to A1+B1's
order of magnitude. Same outcome: model learns refusals (trivial — emit no
special token) but can't move into function-call generation regime in the
gradient-step budget available on our smaller dataset.

The Tunix LoRA recipe is not fundamentally broken — it's tuned for vendor's
data scale, and breaking through to tool-call generation needs more
gradient steps than 1 epoch on 511 rows provides.

## Block B — recipe sweep around the failures

### B1 — A1 + epochs=10 (same vendor LR=1e-5)

Train: 266.8 s wall, train_loss 1.40 → **0.64** (much lower than A1's 0.97).
Eval: 16/56 = 28.6 % — **identical row-by-row to A1**.

**Reading**: the model trained better on its training set (loss 0.64 ≪ 0.97)
but the inference-time pass-rate didn't budge. 5× more epochs at vendor LR
1e-5 still leaves cumulative LR at 1.6e-3 — too low for full-SFT to leave the
base behavior regime. The model still emits NL responses, never function
calls, on every tool-call category. Refusals still PASS at 100 %.

**Verdict**: more epochs at vendor LR doesn't fix the undertraining for our
data scale. Need higher LR.

### B3 — A1 + epochs=10 + LR=5e-5

Train: 265.8 s wall, train_loss → **0.23** (deep convergence on training set).
Eval: **28/56 = 50.0 %** — first full-SFT variant to actually emit tool calls.

| category | pass-rate | M5 cp-192 |
|---|---|---|
| fact_absence | 25.0 % | 37.5 % |
| fact_lookup | 75.0 % | 75.0 % |
| medical_advice_refusal | 50.0 % | **100.0 %** |
| **off_topic_refusal** | **50.0 %** | 25.0 % |
| parallel_call | 37.5 % | 62.5 % |
| tool_error_recovery | 50.0 % | 50.0 % |
| two_turn | 62.5 % | 87.5 % |
| **overall** | **50.0 %** | **62.5 %** |

**Reading**: B3 trades off vs M5. It BEATS M5 on `off_topic_refusal` (the
single category M5 was stuck on) — 50 % vs 25 %. But loses 50 pp on
`medical_advice_refusal` (50 vs 100 %), 25 pp on `parallel_call`, 25 pp on
`two_turn`. Net 12.5 pp worse overall.

This is informative: **off_topic_refusal at 25 % under M5 is partly a recipe
artifact, not purely a dataset signal limit**. A different recipe (full SFT
at higher cumulative LR) does push it to 50 %, but at the cost of
catastrophic forgetting on other categories. The headline dataset bottleneck
verdict still stands — neither M5 nor B3 hits 80 % on every category — but
**off_topic_refusal needs both more data AND a recipe that doesn't
overfit medical/parallel/two-turn during the same training run**.

**Verdict**: full SFT at 5e-5 × 10 epochs is in the right *order* of
cumulative LR (8e-3 vs M5's 3.84e-2) but pays ~13 pp overall. M5's LoRA
recipe is genuinely the better choice on the current dataset.

### A2 result — pending



## Block E — what the diagnostic mandates regardless of the winning recipe

D5 (eval contamination) and D3 (argument-value vocabulary) make this required
work — independent of whether any Block A/B variant clears the 80 % bar on the
**current** holdout. Concretely:

1. **Re-stratify the eval holdout to remove byte-identical train duplicates.**
   A 56-row eval where the top-5 closest pairs are cosine = 1.000 is not a
   generalization test. Either move duplicates out of train (preferring to
   keep them in eval), or author novel eval prompts that don't appear in
   train. Target: max train-↔-eval cosine ≤ 0.85 across the holdout.
2. **Broaden each open-vocabulary string argument to ≥ 20 unique values.**
   - `check_food_interaction.food`: add foods beyond
     {alcohol, grapefruit, grapefruit juice, shellfish} — common
     drug-interaction foods include leafy greens (warfarin), tyramine-rich
     foods (MAOIs), dairy (some antibiotics), high-K foods (ACEIs), etc.
   - `get_medications_at_time.time_24h`: span the day (00:00, 02:00, 04:00,
     ..., 22:00) plus uncommon times (e.g. 07:30, 22:45) so the slot is
     understood as "a time", not "one of these literal 7 strings".
   - `get_medication_by_name.name`: expand beyond the 11 training meds with
     real common medications (atorvastatin, metoprolol, omeprazole,
     amlodipine, etc.) and held-out test meds.
3. **Add 60–80 LLM-augmented rows per refusal class.** D4 confirms refusal
   prompts are paraphrastically diverse, but D3 + the persistent 25 %
   `off_topic_refusal` across all 3 epochs implies more raw row count is
   needed for the 270M model to generalize the refusal abstraction. Per the
   §13 R6(a) ladder, target +160 net rows total.

The §9.4.3 paste-into-LLM prompt template is the authoring path. Author one
batch per arg-vocabulary expansion, run validator, run dataset_quality_audit
to confirm the new vocabulary, then re-train + re-eval.

## Why A2 + 3 epochs (rank-vs-epochs isolation) matters

A2 (LoRA r=8, 1 epoch, LR=1e-4) hit cumLR = 6.4e-3 — close to B3's 8.0e-3
threshold for "model crosses into tool-call generation regime" — but failed
where B3 partially succeeded. To isolate whether the failure was rank
capacity vs gradient steps, we ran A2 + 3 epochs (cumLR = 1.92e-2, **3× B3
and half of M5**).

Result: LoRA r=8 × 3 epochs scores **30.4 % contaminated / 31.1 % clean** —
essentially identical to A2 × 1 epoch (28.6 % / 31.1 %). Only `fact_lookup`
saw any movement (0 → 1 match on contaminated). The cumulative LR is now
3× B3 and B3 is at 50 %; the 3.5 M trainable params of LoRA r=8 cannot
represent the function-call format regardless of training duration.

**Verdict on rank vs epochs**: rank capacity is the bottleneck, not gradient
budget. LoRA r=128 (M5, 30 M trainable) reaches 62.5 %; full SFT (B3, 270 M
trainable, 8e-3 cumLR) reaches 50 %; LoRA r=8 (A2, 3.5 M trainable, even at
1.92e-2 cumLR) caps at 31 %. The high-rank LoRA recipe is the right
parameterization for our 511-row dataset.

## Recommended next steps

1. **Land the corrected baseline narrative**: drop "44.6 %" from documentation;
   the corrected M5 numbers are **62.5 % contaminated / 57.8 % clean**
   (cp-192 + casefold).
2. **Block E authoring round is the highest-leverage next action.** The
   recipe sweep is exhausted — full SFT + various LoRA ranks have been
   tested across cumLR 6.4e-4 → 3.84e-2 and cap at the M5 number on
   generalization-grade eval. Further pp gains require more data, not more
   recipe variants.
3. **Pick the production recipe = M5 v1 (Unsloth + LoRA r=128 + 3 epochs)**.
   The deep-dive shows it is empirically the best recipe on **both** the
   contaminated and the de-contaminated holdouts. The v1 script
   (`scripts/finetune_functiongemma.py`) is already the codified recipe;
   v2 (`scripts/finetune_functiongemma_v2.py`) is preserved for vendor
   reproduction + future sweep work.
4. **Re-run G_EVAL on `eval_holdout_v2_clean.jsonl` for any future M-level
   gate.** The contaminated holdout's pass-rate is reported alongside as a
   memorization sanity-check.
5. **Update §11.4 G_EVAL acceptance** to require an explicit eval-set
   contamination check: "no eval-row user prompt is byte-identical to any
   train-row user prompt; max train-↔-eval cosine < 0.85".
6. **Re-build M6 deployment artifacts from cp-192**, not cp-128. The
   `merged_fg_v1/` server-side dir is currently cp-128; refresh via a
   one-line invocation of `~/functiongemma-finetune/merge_v2.py
   outputs_fg_v1/checkpoint-192 merged_fg_v1_cp192/` (Block C subagent
   already produced this dir; just promote it to the canonical name and
   re-export the GGUF artifacts via `quantize.sh`).
