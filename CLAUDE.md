# CLAUDE.md — gemma3-270M-finetune

Claude Code instructions specific to this repository.

## Repository purpose

Standalone fine-tuning, evaluation, and deployment tooling for **Gemma 3 270M-IT**
(forward-compatible with FunctionGemma when it ships). The reference deployment
target is the SL2619 Synaptics Astra Machina board, but the repo also stands
alone as a generic Gemma 3 270M fine-tune workspace and may optionally be
mounted as a git submodule at `SynapticSL2619/models/gemma-3-270m-it`.

## Key paths

| Path | Role |
|---|---|
| `src/gemma_tools/` | Python package — prompt, bench, SFT, logits-equivalence |
| `scripts/finetune.py` | LoRA/QLoRA SFT entry point (runs on GPU server) |
| `scripts/merge.py` | Merge LoRA adapter → full BF16 checkpoint |
| `scripts/smoke_test.py` | Side-by-side smoke: base vs merged |
| `scripts/server-bootstrap.sh` | Idempotent Ubuntu server setup (CUDA + SFT stack) |
| `scripts/chat_remote.sh` | Interactive chat via llama-server on board |
| `data/` | Health-QA YAML, SFT datasets (sft_v1*.jsonl), prompt templates |
| `docs/conventions/` | Normative coding/repo/workflow rules (Python, shell, testing, git, doc-update, module-layering, SLM-prompt) |
| `docs/references/` | Upstream sources — note files + opt-in submodules under `upstream/{gemma,llama.cpp}` |
| `docs/guides/` | Human-facing how-tos (e.g. fine-tune best practices) |
| `docs/plans/` | Frozen historical narratives (read-only; carried over from SynapticSL2619) |
| `docs/bench/` | Frozen bench run records (read-only) |
| `docs/deployment/sl2619-board.md` | SL2619 cross-compile + deploy runbook |
| `docs/conventions/slm-system-prompt.md` | Normative SLM prompt rules (R-1…R-10) |
| `docs/conventions/doc-update.md` | DRY canonical-ownership registry; CLAUDE.md/README.md refresh protocol |
| `models/gemma-3-270m-it/README.md` | Per-model analysis: IFEval, quantization, prompt strategy |

## Workflows

### Install (host dev)
```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run tests
```bash
uv run pytest
```

### Lint + typecheck
```bash
uv run ruff check src tests
uv run mypy src
```

### Fine-tune (GPU server)
```bash
# 1. Bootstrap server (once)
scp scripts/server-bootstrap.sh nouslogic-server:~/
ssh -t nouslogic-server 'bash ~/server-bootstrap.sh --with-system-deps'

# 2. Upload data and run SFT
scp data/sft_v1.train.jsonl data/sft_v1.val.jsonl nouslogic-server:~/sl2619-finetune/
ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python finetune.py ...'
```

## Discipline

- **No model weights in git** — `.gguf`, `.bin`, `.safetensors`, `.pt` are gitignored.
  Use explicit `git add -f` only for intentional small fixtures (e.g. tokenizer config).
- **Tests before any data pipeline change** — run `uv run pytest` and confirm green
  before editing `sft_dataset.py`, `health_table.py`, or the YAML schema.
- **SSH to board is read-only** — deployment commands are emitted for the user
  to run, never executed autonomously.
- **Prompt template is the SFT contract** — `scripts/finetune.py:_to_prompt_completion`
  is the single source of truth for training-time prompt shape. `smoke_test.py` and
  `bench_prompt.py` must replicate it exactly; divergence creates a tokenization artifact.
- **Submodules are opt-in** — `docs/references/upstream/{gemma,llama.cpp}` are
  shallow git submodules with `update = none`. A fresh clone does not pull them.
  Initialize on demand with
  `git submodule update --init docs/references/upstream/<name>`.
