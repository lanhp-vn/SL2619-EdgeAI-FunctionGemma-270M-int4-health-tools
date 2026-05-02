# gemma3-270M-finetune

Fine-tuning, evaluation, and deployment workspace for **FunctionGemma 270M-IT**
on the SL2619 Synaptics Astra Machina board, with the original Gemma 3 270M-IT
health-QA SFT track preserved as a working reference.

## What this repo does

- Trains FunctionGemma 270M-IT for closed-world function calling against a
  synthetic 7-tool patient-record registry (`data/health_table_v1.yaml`).
- Ships a deployable bundle at
  `releases/functiongemma-270m/001-baseline/` (HF merged weights, LoRA
  adapter, FP16 GGUF, `Modelfile`, `model_client.py`).
- Provides a local interactive REPL (`scripts/functiongemma/chat.py`) and a
  two-mode bench harness (host + remote SL2619 board).
- Documents the path to INT4/INT8 quantization for the SL2619 board
  (`docs/plans/functiongemma/quantization-plan.md`, NOT YET EXECUTED).
- Preserves a runnable Gemma 3 270M health-QA SFT track under
  `src/gemma_tools/_legacy/` and `tests/_legacy/` for anyone returning to
  that approach (CI keeps it green).

## Status

| Track | State |
|---|---|
| FunctionGemma iteration 001 (Distil Labs) | DONE — every metric at 0.9583 on the 24-row contaminated holdout (`distil/iterations/001-baseline/`) |
| FunctionGemma INT4/INT8 board quantization sweep | PLANNED — see `docs/plans/functiongemma/quantization-plan.md` |
| Gemma 3 270M-IT health-QA SFT (legacy) | DONE / archived — runnable reference under `src/gemma_tools/_legacy/` and `archive/gemma3-270m-health-qa/` |

## Hardware

- **Host (this WSL machine).** Ubuntu 24.04 under WSL2; Python 3.12; CPU-only;
  used for dataset prep, host smoke, eval, and host-side bench.
- **Fine-tune server (`nouslogic-server`).** Ubuntu, RTX 5080 (16 GiB,
  cu128), 47 GiB RAM. Reachable over Tailscale/SSH. Used for the local
  Unsloth fallback finetune; provisioned by `scripts/setup/server-bootstrap.sh`.
- **SL2619 board (`nouslogic-sl2619`).** Synaptics SL2610 / Cortex-A55 × 2,
  Yocto Linux + BusyBox; ~1.7 GiB MemAvailable; cross-compiled
  `llama.cpp` aarch64 binaries staged at `/mnt/sdcard/llama-cpp/`.
  Read-only SSH access from the agent; user runs all writes.

## Install

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,functiongemma]"
```

The `functiongemma` extra pulls `transformers`, `accelerate`, `torch` (CPU
wheel for host smoke), `huggingface-hub`, `pydantic`, `jsonschema`,
`sentencepiece`, `gguf`, and `llama-cpp-python`. Combined size ~1.5 GiB.

## Repo layout

```
gemma3-270M-finetune/
|- CLAUDE.md                       Agent self-reference
|- README.md                       This file
|- pyproject.toml, uv.lock         Build + dependency manifests
|- src/gemma_tools/
|  |- __init__.py                  Package shim (version only)
|  |- health_table.py              Patient-record schema (dual-use: legacy + FG)
|  |- functiongemma/               Active sub-package: dataset, tools
|  |- _legacy/                     Frozen gemma3-270m health-QA modules
|- scripts/
|  |- functiongemma/
|  |  |- chat.py                   Interactive REPL (the local demo)
|  |  |- bench.py                  Two-mode bench harness
|  |  |- smoke.py                  Smoke runner
|  |  |- data/                     build_seeds, build_splits, ingest, quality_audit, gen_prompt_templates
|  |  |- train/finetune_local.py   Unsloth fallback SFT (server-side)
|  |  |- eval/eval_holdout.py      Holdout evaluation
|  |  |- deploy/                   chat_board.py, ask_board.sh, run_prompt.sh
|  |- setup/                       server-bootstrap.sh, add_synaptics_submodules.sh
|  |- pre_commit_phi_scanner.py    PHI gate for data ingest
|- tests/
|  |- functiongemma/               Active tests
|  |- _legacy/                     gemma3-270m health-QA tests (still in CI)
|  |- test_health_table.py         Shared schema test
|- data/
|  |- health_table_v1.yaml         Synthetic patient record (no real PHI)
|  |- functiongemma/               dataset_v1/, seed_conversations.jsonl, tools_v1.yaml, eval_holdouts
|  |- _legacy/                     Frozen gemma3-270m SFT corpora + prompts.yaml
|- releases/functiongemma-270m/001-baseline/
|  |- merged/                      HF merged weights
|  |- adapter/                     LoRA adapter (r=64, alpha=64)
|  |- gguf/                        FP16 GGUF + Modelfile
|  |- model_client.py              Distil deploy client
|- distil/iterations/001-baseline/
|  |- README.md                    Iteration journey, metrics, repair history
|  |- config.yaml                  Distil hyperparameters
|  |- job_description.json         Routing rules, task description, judge instructions
|  |- training-analysis.md         Aggregate + per-row metrics
|  |- teacher-eval-analysis.md     Teacher prediction analysis (v1, v2, v3)
|  |- data/{train,test}.jsonl      Dataset uploaded to platform
|  |- predictions/                 student.jsonl + teacher-v{1,2,3}.jsonl
|- bench/functiongemma/runs/       Active FunctionGemma bench JSONL outputs
|- docs/
|  |- conventions/                 Normative coding/repo/workflow rules (Python, shell, testing, git, doc-update, module-layering)
|  |- references/                  Upstream sources + opt-in submodules
|  |- plans/functiongemma/         Active plans (recipe, decisions-log, quantization-plan, seed-authoring, llm-augmentation, upstream-issue-drafts)
|  |- bench-notes/functiongemma/   Active bench analyses
|  |- guides/                      finetune-best-practices, distil-iteration-recipe-and-lessons
|  |- deployment/                  sl2619-board.md (cross-compile), functiongemma-board-deploy.md
|- archive/
|  |- README.md                    Archive index
|  |- gemma3-270m-health-qa/       Frozen gemma3-270m track (plans, bench, scripts, model-card, guides)
|  |- functiongemma-pre-distil/    Frozen pre-distil FunctionGemma path (plans, scripts, bench, data)
```

## Run the demo

The end-to-end acceptance test for this workspace:

```bash
uv run python scripts/functiongemma/chat.py
```

Loads `releases/functiongemma-270m/001-baseline/gguf/model.gguf` and
`releases/functiongemma-270m/001-baseline/merged/` as the tokenizer + chat
template; serves an interactive REPL that routes user prompts through the
7-tool patient-record registry against `data/health_table_v1.yaml`.

Single-prompt probe (non-interactive):

```bash
uv run python scripts/functiongemma/chat.py --probe "What is my blood pressure?"
```

## Train

### Current production path: Distil Labs platform

Iteration 001 was produced by uploading the seed dataset to Distil's
multi-turn-tool-calling platform. The hyperparameters, judge instructions,
and result analysis live alongside the artifacts:

```
distil/iterations/001-baseline/
|- README.md                       Upload/run/eval timeline (3 versions)
|- config.yaml                     LoRA r=64 alpha=64; teacher gpt-oss-120b; gen_target 5000
|- job_description.json            7 routing rules + 4 special-case judge rules
|- training-analysis.md            Final 0.9583 metrics, training duration ~4h28m
|- teacher-eval-analysis.md        Per-version teacher prediction analysis
```

Distil CLI invocations are documented in `.claude/skills/distil-cli/distil-cli/SKILL.md`.

### Local fallback path: Unsloth + LoRA on `nouslogic-server`

When the Distil platform is unavailable or full hyperparameter control is
needed, run the local SFT script on the GPU server:

```bash
# 1. One-time bootstrap (first run only)
scp scripts/setup/server-bootstrap.sh nouslogic-server:~/
ssh -t nouslogic-server 'bash ~/server-bootstrap.sh --with-system-deps'

# 2. Upload dataset
scp data/functiongemma/dataset_v1/{train,val,test}.jsonl \
    nouslogic-server:~/functiongemma-finetune/data/

# 3. Upload script + supporting modules
scp scripts/functiongemma/train/finetune_local.py nouslogic-server:~/functiongemma-finetune/
scp -r src/gemma_tools/functiongemma/ src/gemma_tools/health_table.py \
    nouslogic-server:~/functiongemma-finetune/gemma_tools/

# 4. Run SFT (server-side; takes ~60 min on RTX 5080)
ssh nouslogic-server 'cd ~/functiongemma-finetune && source .venv/bin/activate && \
    python finetune_local.py --output-dir outputs/iter-002'

# 5. Pull merged weights back to host
scp -r nouslogic-server:~/functiongemma-finetune/outputs/iter-002/ \
    releases/functiongemma-270m/iter-002/
```

The script mirrors the iteration 001 hyperparameter shape (LoRA r=64, alpha=64,
target `q_proj,v_proj`, 4 epochs). See
`docs/plans/functiongemma/recipe.md` for the full recipe.

## Evaluate

### Holdout eval

```bash
uv run python scripts/functiongemma/eval/eval_holdout.py \
    --model releases/functiongemma-270m/001-baseline/merged \
    --holdout data/functiongemma/eval_holdout_v2_clean.jsonl
```

Per-row JSONL output + summary Markdown table; default Markdown destination
is `docs/bench-notes/functiongemma/<today>_functiongemma-eval.md`. Same
metric set as `distil/iterations/001-baseline/teacher-eval-analysis.md`
(judge, ROUGE, tool-call equivalence, binary, staged).

### Bench (host + board)

```bash
uv run python scripts/functiongemma/bench.py --mode local --warmup 1
uv run python scripts/functiongemma/bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model  /mnt/sdcard/models/functiongemma-270m/model.gguf \
    --threads 2 --warmup 1
```

Both modes write per-prompt JSONL into `bench/functiongemma/runs/`. Compare
modes:

```bash
diff <(jq -c '{id:.prompt_id,call:.parsed_call}' bench/functiongemma/runs/*local*.jsonl | sort) \
     <(jq -c '{id:.prompt_id,call:.parsed_call}' bench/functiongemma/runs/*remote*.jsonl | sort)
```

Expected: parsed-call rows identical (same model, deterministic greedy decoding);
throughput differs by ~10–20×.

## Deploy to SL2619 board

See `docs/deployment/functiongemma-board-deploy.md` for the full recipe
(host-side prompt template generation, scp to `/mnt/sdcard/`, and the
on-board run / REPL paths). High-level:

```bash
# Generate prompt templates
uv run python scripts/functiongemma/data/gen_prompt_templates.py \
    --tokenizer releases/functiongemma-270m/001-baseline/merged/ \
    --output-dir /tmp/fg_deploy/

# scp to board (user runs)
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/models/functiongemma-270m'
scp releases/functiongemma-270m/001-baseline/gguf/model.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/
scp /tmp/fg_deploy/* scripts/functiongemma/deploy/* \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# Run on board
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh \
    "What is my blood pressure?"'
```

## Test

```bash
uv run pytest                            # full suite (537 tests)
uv run pytest tests/functiongemma/       # active FunctionGemma tests only
uv run pytest tests/_legacy/             # gemma3-270m health-QA tests
uv run ruff check src tests
uv run mypy src
```

The pre-distil archived weighting test under
`archive/functiongemma-pre-distil/tests/` is NOT collected by default; run
manually with `pytest archive/functiongemma-pre-distil/tests/` if you need
to verify it.

## Data layout

| Path | Content |
|---|---|
| `data/health_table_v1.yaml` | Synthetic patient record (vitals, conditions, medications, allergies, appointments, contacts) |
| `data/functiongemma/seed_conversations.jsonl` | 50 hand-authored multi-turn conversation seeds |
| `data/functiongemma/llm_expanded_v1.jsonl` | LLM-augmented expansion (~545 rows after PHI scan + ingest) |
| `data/functiongemma/dataset_v1/{train,val,test}.jsonl` | Distil-uploaded training splits |
| `data/functiongemma/eval_holdout_v1.jsonl` | Original 24-row holdout (mixed novel + train-overlap rows) |
| `data/functiongemma/eval_holdout_v2_clean.jsonl` | All-novel-phrasing holdout (45 rows) |
| `data/functiongemma/eval_holdout_v2_contaminated.jsonl` | 11 train-overlap items + clean (56 rows) |
| `data/functiongemma/quarantine.jsonl` | Per-row failures from past ingest runs (appended-to) |
| `data/functiongemma/tools_v1.yaml` | JSON-Schema mirror of the Python tool registry |
| `data/_legacy/` | gemma3-270m corpora: sft_v1.{train,val,test}.jsonl, sft_v1.audit.jsonl, prompts.yaml, clean_sft_dataset.json |

## Submodules

`docs/references/upstream/{gemma,llama.cpp}` are shallow git submodules with
`update = none`. They are NOT pulled on a fresh clone. Initialize on demand:

```bash
git submodule update --init docs/references/upstream/llama.cpp
git submodule update --init docs/references/upstream/gemma
```

`docs/references/upstream/unsloth-notebooks/` is a standalone shallow clone
(NOT a registered submodule) with sparse-checkout limited to
`nb/FunctionGemma_(270M).ipynb`. See `scripts/setup/add_synaptics_submodules.sh`.

## License

Model weights inherit the `gemma` license (open weights, license-gated download).
Inference and finetune scripts are first-party to this repository. See
`releases/functiongemma-270m/001-baseline/merged/{LICENSE,STUDENT_LICENSE,TEACHER_LICENSE}`
for the full terms applying to the iteration-001 deliverable.
