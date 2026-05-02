# CLAUDE.md — gemma3-270M-finetune

Claude Code instructions specific to this repository.

## Repository purpose

Active focus: **FunctionGemma 270M-IT** (function-calling on the SL2619
Synaptics Astra Machina board). The first deployable artifact at
`releases/functiongemma-270m/001-baseline/` was produced via Distil Labs;
the next track is INT4/INT8 quantization on the board.

The original Gemma 3 270M-IT health-QA SFT track is preserved as a working
reference under `archive/gemma3-270m-health-qa/` (with live code at
`src/gemma_tools/_legacy/`, `tests/_legacy/`, `data/_legacy/` so its tests
still pass in CI). Do NOT mix new work into that track.

## Key paths

| Path | Role |
|---|---|
| `src/gemma_tools/__init__.py` | Package shim — version only |
| `src/gemma_tools/health_table.py` | Patient-record schema + Pydantic loader (dual-use: legacy bench AND FunctionGemma tools) |
| `src/gemma_tools/functiongemma/` | Active sub-package — `dataset.py`, `tools.py` |
| `src/gemma_tools/_legacy/` | Frozen Gemma 3 270M health-QA modules (importable; tests under `tests/_legacy/` still run) |
| `scripts/functiongemma/chat.py` | Interactive REPL — the local FunctionGemma demo |
| `scripts/functiongemma/{bench,smoke}.py` | Bench harness (local + remote SL2619) and smoke runner |
| `scripts/functiongemma/data/` | Dataset prep — build_seeds, build_splits, ingest, quality_audit, gen_prompt_templates |
| `scripts/functiongemma/train/finetune_local.py` | Unsloth-based local SFT fallback (when Distil platform is not used) |
| `scripts/functiongemma/eval/eval_holdout.py` | Holdout evaluation; default output to `docs/bench-notes/functiongemma/` |
| `scripts/functiongemma/deploy/` | Board deploy: `chat_board.py`, `ask_board.sh`, `run_prompt.sh` |
| `scripts/setup/server-bootstrap.sh` | Idempotent Ubuntu-server SFT-stack bootstrap (RTX 5080) |
| `scripts/pre_commit_phi_scanner.py` | PHI scanner for FunctionGemma data ingest |
| `data/health_table_v1.yaml` | Synthetic patient record (no real PHI) |
| `data/functiongemma/dataset_v1/{train,val,test}.jsonl` | Active Distil iteration-001 training splits |
| `data/functiongemma/seed_conversations.jsonl` | 50-row hand-authored seeds |
| `data/functiongemma/eval_holdout_v{1,2_clean,2_contaminated}.jsonl` | Active eval holdouts |
| `data/_legacy/` | Frozen gemma3-270m SFT corpora and `prompts.yaml` |
| `releases/functiongemma-270m/001-baseline/` | Iteration-001 deployable: `merged/`, `adapter/`, `gguf/`, `Modelfile`, `model_client.py` |
| `distil/iterations/001-baseline/` | Distil training metadata: config, predictions, training-analysis, teacher-eval-analysis |
| `bench/functiongemma/runs/` | Active FunctionGemma bench JSONL outputs |
| `docs/plans/functiongemma/` | Active plan docs — `recipe.md`, `decisions-log.md`, `quantization-plan.md`, `seed-authoring-recipe.md`, `llm-augmentation-prompt.md`, `upstream-issue-drafts.md` |
| `docs/bench-notes/functiongemma/` | Active distil-related bench analyses |
| `docs/deployment/sl2619-board.md` | Board cross-compile + deploy runbook |
| `docs/deployment/functiongemma-board-deploy.md` | FunctionGemma-specific board deploy recipe |
| `docs/conventions/` | Normative coding/repo/workflow rules (out of scope for this CLAUDE.md) |
| `docs/references/` | Upstream source notes + opt-in submodules under `upstream/{gemma,llama.cpp}` |
| `archive/gemma3-270m-health-qa/` | Frozen gemma3-270m track (plans, bench, scripts, model-card) |
| `archive/functiongemma-pre-distil/` | Frozen pre-distil FunctionGemma path (plans, scripts, bench, data) |

## Workflows

### Install (host dev)

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,functiongemma]"
```

### Tests, lint, typecheck

```bash
uv run pytest                            # 537 tests (active + _legacy)
uv run ruff check src tests
uv run mypy src
```

### Run the local demo (Phase 5 acceptance test)

```bash
uv run python scripts/functiongemma/chat.py
```

Defaults to `releases/functiongemma-270m/001-baseline/{gguf/finetuned_functiongemma_fp16.gguf, merged/}`
and `data/health_table_v1.yaml` for tool dispatch.

### Bench

```bash
uv run python scripts/functiongemma/bench.py --mode local --warmup 1
uv run python scripts/functiongemma/bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model  /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_fp16.gguf
```

Output lands in `bench/functiongemma/runs/`.

### Holdout eval

```bash
uv run python scripts/functiongemma/eval/eval_holdout.py \
    --model releases/functiongemma-270m/001-baseline/merged \
    --holdout data/functiongemma/eval_holdout_v2_clean.jsonl
```

Output Markdown lands in `docs/bench-notes/functiongemma/<today>_functiongemma-eval.md`.

### Server-side fine-tune (local fallback)

```bash
# 1. One-time server bootstrap
scp scripts/setup/server-bootstrap.sh nouslogic-server:~/
ssh -t nouslogic-server 'bash ~/server-bootstrap.sh --with-system-deps'

# 2. Upload data and run SFT
scp data/functiongemma/dataset_v1/{train,val,test}.jsonl nouslogic-server:~/functiongemma-finetune/data/
ssh nouslogic-server 'cd ~/functiongemma-finetune && source .venv/bin/activate && \
    python finetune_local.py ...'
```

For Distil Labs platform invocations (current production path), see
`distil/iterations/001-baseline/README.md`.

## Discipline

- **No model weights in git** — `.gguf`, `.bin`, `.safetensors`, `.pt` are gitignored. Use explicit `git add -f` only for intentional small fixtures (tokenizer config, etc.).
- **SSH to board is read-only** — deployment commands are emitted for the user; the agent never mutates the board (R3 in `.claude/CLAUDE.local.md`).
- **PHI scanner gates data ingest** — `scripts/pre_commit_phi_scanner.py` scans every staged JSONL before merge into `llm_expanded_v1.jsonl`. Patient YAML stays synthetic; OQ-5 covers any switch to real PHI.
- **Tests before any data pipeline change** — `uv run pytest` must be green before editing `gemma_tools.functiongemma.dataset`, `gemma_tools.functiongemma.tools`, or `health_table.py`.
- **Submodules are opt-in** — `docs/references/upstream/{gemma,llama.cpp}` are shallow git submodules with `update = none`. Initialize on demand: `git submodule update --init docs/references/upstream/<name>`.
- **`unsloth-notebooks` is a nested git repo, not a submodule** — `docs/references/upstream/unsloth-notebooks/` is a standalone shallow clone with sparse-checkout (single notebook).
- **Archive is read-only** — `archive/` and `data/_legacy/` are frozen reference. New work goes under active dirs (`docs/`, `src/`, `scripts/`, `tests/`, `data/`, `releases/`, `distil/`, `bench/`).
- **`docs/conventions/` is normative** — agent edits go through normal PRs with review; this file's protocols (Python style, testing, git, doc-update, module layering) bind agent behavior. Path references inside `doc-update.md` §8.1 are pre-refactor and need a separate PR.

## Pointers

- `docs/conventions/doc-update.md` §8.1 — DRY canonical-ownership registry.
- `docs/conventions/code-style-python.md`, `code-style-shell.md`, `testing.md`, `git-workflow.md`, `module-layering.md` — normative rules.
- `docs/plans/functiongemma/recipe.md` — current FunctionGemma working recipe (model identity, wire format, train/eval paths).
- `docs/plans/functiongemma/decisions-log.md` — decisions table.
- `docs/plans/functiongemma/quantization-plan.md` — current focus (NOT YET EXECUTED).
- `archive/README.md` — archive index.
