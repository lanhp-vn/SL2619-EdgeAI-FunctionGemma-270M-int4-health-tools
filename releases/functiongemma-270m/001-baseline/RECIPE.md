# FunctionGemma 270M iteration 001 — recipe

How this iteration was produced and how to reproduce it (with or without
Distil Labs).

## Provenance

| Field | Value |
|---|---|
| Base student | `google/functiongemma-270m-it` (HF) |
| Teacher | `openai.gpt-oss-120b` (via Distil Labs platform) |
| Task type | `multi-turn-tool-calling-closed-book` |
| Distil model id | `231feebb-8cc0-4d5f-9e4b-4d2f00e362b2` |
| Training id | `c9d34596-ee7a-4e56-be2b-254159fe7796` |
| Training duration | DAG 2026-05-02 00:25 → 04:53 UTC (~4h 28m total; finetune wall-clock ~3h 41m) |
| Best checkpoint | epoch 3 of 4 |
| Final eval (24-row contaminated holdout) | judge 0.9583, ROUGE 0.9142, tool-call equivalence 0.9583, binary 0.9583, staged 0.9583 |

The full upload / re-upload / teacher-eval timeline is in
`distil/iterations/001-baseline/README.md` (3 prompt-engineering iterations
v1 → v2 → v3 lifted judge from 0.7917 → 0.8750 → 0.9583).

Per-row analysis: `distil/iterations/001-baseline/training-analysis.md`.

## Hyperparameters

| Parameter | Value |
|---|---|
| LoRA r | 64 |
| LoRA alpha | 64 |
| LoRA dropout | 0.0 |
| Target modules | `q_proj,v_proj` |
| Epochs | 4 (best at epoch 3) |
| Generation target (synthesis) | 5000 |
| Validation similarity threshold | 0.90 (default 0.95, loosened to widen scenario coverage) |
| Mutators | `["complexity"]` |
| Mutation topics | 5 routing-rule clusters + 3 phrasing styles |
| Synthetic data generated | 5004 examples (57 iterations) |
| Final train shape | 5054 (50 seeds + 5004 synth), expanded to 7481 multi-turn samples |

Full Distil config: `distil/iterations/001-baseline/config.yaml`.
Full job description (routing rules + judge instructions):
`distil/iterations/001-baseline/job_description.json`.

## Source data

| File | Rows | Origin |
|---|---|---|
| `distil/iterations/001-baseline/data/train.jsonl` | 50 | Hand-authored seeds covering 7 tools (fl×25, fa×15, te×10) |
| `distil/iterations/001-baseline/data/test.jsonl` | 24 | Held-out gold (fl×8, fa×8, te×8) — byte-equal to `data/functiongemma/eval_holdout_v1.jsonl` |

Refusal classes (`medical_advice_refusal`, `off_topic_refusal`) and
`parallel_call` are deliberately excluded from this iteration — Distil's
`multi-turn-tool-calling-closed-book` task enforces "exactly one tool call
per assistant turn", which those classes don't fit. The local F1+F5 path
in `archive/functiongemma-pre-distil/` was the prior attempt at refusal
training; it failed to clear the bar (see
`archive/functiongemma-pre-distil/bench/eval-summary.md`).

## Reproduce via Distil Labs (current production path)

1. **Login**: `distil login` (token from <https://app.distillabs.ai>).
2. **Create model**:

   ```bash
   distil model create fg-iter-002
   ```

3. **Stage training data** in the same shape as
   `distil/iterations/001-baseline/data/`:
   - `train.jsonl` — seed conversations.
   - `test.jsonl` — held-out gold; cross-set duplicates of `(question, answer)`
     are rejected by the platform.
4. **Edit `config.yaml`** to match this iteration's hyperparameters
   (LoRA r=64, alpha=64, mutators, generation target).
5. **Edit `job_description.json`** routing rules + judge instructions.
6. **Dry-run upload**:

   ```bash
   distil model upload-data fg-iter-002 --train-data train.jsonl \
       --test-data test.jsonl --dry-run
   ```

   No credit cost. Surfaces shape errors and cross-set duplicates without
   committing.
7. **Real upload**:

   ```bash
   distil model upload-data fg-iter-002 --train-data train.jsonl \
       --test-data test.jsonl
   distil model upload-status fg-iter-002 --output json    # await JOB_SUCCESS
   ```

8. **Run teacher evaluation** (free; verifies the teacher can synthesize
   the task before paying for SFT):

   ```bash
   distil model run-teacher-evaluation fg-iter-002
   ```

   Judge ≥ 0.80 is the proceed bar. If it fails, iterate on
   `task_description` + `llm_as_a_judge_instructions` (zero-cost edits) and
   re-upload. Iteration 001 took 3 such rounds.
9. **Run SFT**:

   ```bash
   distil model run-finetune fg-iter-002
   ```

10. **Pull the artifacts**:

    ```bash
    distil model download-artifact fg-iter-002 --output releases/functiongemma-270m/iter-002/
    ```

    Produces `merged/`, `adapter/`, `gguf/model.gguf`, `Modelfile`,
    `model_client.py` — the same shape as this iteration.

Skill pointer: `.claude/skills/distil-cli/distil-cli/SKILL.md`.

## Reproduce locally without Distil Labs (fallback path)

The fallback runs LoRA SFT on `nouslogic-server` (RTX 5080) using Unsloth
with the same hyperparameters. Live script:
`scripts/functiongemma/train/finetune_local.py`.

Trade-offs vs Distil:

- No teacher synthesis — input is the dataset as-is. Iteration 001 had
  5054 training rows after Distil synthesis; the local path uses ~545 LLM-augmented
  rows from `data/functiongemma/llm_expanded_v1.jsonl`. Expect lower
  performance.
- Full hyperparameter control. Direct PyTorch, no platform dependencies.
- Refusal classes and parallel-call workflows can be included (the
  pre-distil F1 sweep proved this is non-trivial — see
  `archive/functiongemma-pre-distil/bench/eval-summary.md`).

```bash
# 1. One-time bootstrap
scp scripts/setup/server-bootstrap.sh nouslogic-server:~/
ssh -t nouslogic-server 'bash ~/server-bootstrap.sh --with-system-deps'

# 2. Upload script + dependencies
scp scripts/functiongemma/train/finetune_local.py nouslogic-server:~/functiongemma-finetune/
scp -r src/gemma_tools/functiongemma/ src/gemma_tools/health_table.py \
    src/gemma_tools/__init__.py \
    nouslogic-server:~/functiongemma-finetune/gemma_tools/

# 3. Upload data
scp data/functiongemma/dataset_v1/{train,val,test}.jsonl \
    nouslogic-server:~/functiongemma-finetune/data/

# 4. Run SFT (~60 min on RTX 5080)
ssh nouslogic-server 'cd ~/functiongemma-finetune && source .venv/bin/activate && \
    python finetune_local.py \
        --train-data data/train.jsonl \
        --val-data data/val.jsonl \
        --output-dir outputs/iter-002 \
        --lora-r 64 --lora-alpha 64 --lora-dropout 0.0 \
        --target-modules q_proj v_proj \
        --num-epochs 4'

# 5. Merge LoRA → full BF16 (server-side)
ssh nouslogic-server 'cd ~/functiongemma-finetune && python merge_v2.py \
    --adapter outputs/iter-002 --output outputs/iter-002-merged'

# 6. Convert merged → GGUF (server-side, uses llama.cpp convert + quantize)
ssh nouslogic-server 'cd ~/functiongemma-finetune && bash quantize.sh \
    outputs/iter-002-merged outputs/iter-002.gguf'

# 7. Pull artifacts back
scp -r nouslogic-server:~/functiongemma-finetune/outputs/iter-002* \
    releases/functiongemma-270m/iter-002/
```

## Verify locally

```bash
uv run python scripts/functiongemma/chat.py \
    --model releases/functiongemma-270m/001-baseline/gguf/model.gguf \
    --tokenizer releases/functiongemma-270m/001-baseline/merged \
    --probe "What is my blood pressure?"
```

Expected output:

```
→ {"tool": "get_vitals", "args": {}}
  ⤷ {"heart_rate_bpm": 72, "blood_pressure_systolic": 118, ...}
  >> Your blood pressure is 118/76 (measured 2026-04-24 08:15).
```

## Files in this release

| File | Purpose |
|---|---|
| `merged/` | HF merged BF16 weights + tokenizer + chat template (for vLLM, Ollama, transformers) |
| `adapter/` | LoRA adapter only (r=64, alpha=64) — re-attach to base model with `peft` |
| `gguf/model.gguf` | FP16 GGUF for llama.cpp (518 MiB) |
| `gguf/Modelfile` | Ollama Modelfile pointing at `model.gguf` |
| `model_client.py` | Distil deploy client (Ollama/vLLM HTTP wrapper) |
| `RECIPE.md` | This file |

## Next steps

`docs/plans/functiongemma/quantization-plan.md` — INT4/INT8 sweep on the
SL2619 board to find the variant that minimizes board latency without
breaking tool-call accuracy.
