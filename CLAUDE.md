# AGENTS.md — gemma3-270M-finetune

Claude Code instructions specific to this repository.

## Repository purpose

Fine-tuning, evaluation, and deployment tooling for **Gemma 3 270M-IT** on the
SL2619 Synaptics Astra Machina board. This repo is a companion to
`SynapticSL2619/` and will eventually be mounted as a git submodule at
`SynapticSL2619/models/gemma-3-270m-it`.

## Key paths

| Path | Role |
|---|---|
| `src/gemma_tools/` | Python package — prompt, bench, SFT, logits-equivalence |
| `scripts/finetune.py` | LoRA/QLoRA SFT entry point (runs on GPU server) |
| `scripts/merge.py` | Merge LoRA adapter → full BF16 checkpoint |
| `scripts/t5_smoke.py` | Side-by-side smoke: base vs merged |
| `scripts/server-bootstrap.sh` | Idempotent Ubuntu server setup (CUDA + SFT stack) |
| `scripts/chat_remote.sh` | Interactive chat via llama-server on board |
| `data/` | Health-QA YAML, SFT datasets (sft_v1*.jsonl), prompt templates |
| `docs/plans/` | Fine-tune and eval plans |
| `docs/bench/` | Frozen bench run records |
| `docs/conventions/slm-system-prompt.md` | Normative SLM prompt rules (R-1…R-10) |
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
- **SSH to board is read-only** — follow SynapticSL2619 R3; deployment commands
  are emitted for the user, never run autonomously.
- **Prompt template is the SFT contract** — `scripts/finetune.py:_to_prompt_completion`
  is the single source of truth for training-time prompt shape. `t5_smoke.py` and
  `bench_prompt.py` must replicate it exactly; divergence creates a tokenization artifact.
