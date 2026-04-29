# gemma3-270M-finetune

Fine-tuning, evaluation, and deployment tooling for **Gemma 3 270M-IT** (and
forward-compatible with **FunctionGemma** when its release lands).

The reference deployment target is the [Synaptics Astra Machina SL2619](https://github.com/nouslogic/SynapticSL2619)
Cortex-A55 edge platform via llama.cpp + GGUF — see [`docs/deployment/sl2619-board.md`](docs/deployment/sl2619-board.md).
The repo also stands alone as a generic Gemma 3 270M fine-tune workspace and may
optionally be mounted as a git submodule at `SynapticSL2619/models/gemma-3-270m-it`.

---

## Repository layout

```
gemma3-270M-finetune/
├── src/gemma_tools/          # Python package
│   ├── health_table.py       # Health-YAML schema + patient loader
│   ├── prompt_composer.py    # Directive-form system-prompt builder
│   ├── sft_dataset.py        # Chat-template → JSONL dataset builder
│   ├── sft_build.py          # CLI: build sft_v1.{train,val,test}.jsonl
│   ├── bench_prompt.py       # Local prompt bench runner
│   ├── bench_eval.py         # JSONL → Markdown scorer
│   ├── bench_remote.py       # Remote llama-server bench runner
│   ├── chat_probe.py         # Interactive chat probe (remote server)
│   └── logits_equivalence.py # Logits-equivalence gate (cross-arch KL)
├── scripts/
│   ├── finetune.py           # LoRA/QLoRA SFT entry point (GPU server)
│   ├── merge.py              # Merge LoRA adapter → full BF16
│   ├── smoke_test.py         # Side-by-side smoke: base vs merged
│   ├── server-bootstrap.sh   # Idempotent Ubuntu server setup
│   └── chat_remote.sh        # Interactive chat via llama-server on board
├── data/
│   ├── health_table_v1.yaml  # Health-QA fixture (patients, conditions, meds…)
│   ├── prompts.yaml          # Prompt suite for bench + logits-equivalence
│   ├── sft_v1.{train,val,test,audit}.jsonl   # Built SFT dataset (path B)
│   └── sft_v1_pathA.{train,val,test}.jsonl  # Built SFT dataset (path A)
├── docs/
│   ├── conventions/
│   │   └── slm-system-prompt.md  # Normative SLM prompt rules R-1…R-10
│   ├── plans/                # Fine-tune + eval + logits-equivalence plans
│   ├── bench/                # Frozen bench run records
│   ├── analysis/             # Model evaluation write-ups
│   ├── references/           # Curated upstream-source pointers (Gemma, HF, llama.cpp)
│   ├── deployment/           # Per-target deployment runbooks (SL2619, …)
│   └── deferred/             # Archived investigation notes
└── models/
    └── gemma-3-270m-it/
        └── README.md         # Per-model analysis: IFEval, quant, prompt strategy
```

---

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Workflows

### 1. Build SFT dataset

```bash
uv run sft-build --yaml data/health_table_v1.yaml \
    --out-train data/sft_v1.train.jsonl \
    --out-val   data/sft_v1.val.jsonl \
    --out-test  data/sft_v1.test.jsonl \
    --seed 42
```

### 2. Fine-tune on GPU server

Bootstrap the server once (Ubuntu 24.04 + RTX GPU):

```bash
scp scripts/server-bootstrap.sh nouslogic-server:~/
ssh -t nouslogic-server 'bash ~/server-bootstrap.sh --with-system-deps'
```

Upload data and run LoRA/QLoRA SFT:

```bash
scp data/sft_v1.train.jsonl data/sft_v1.val.jsonl nouslogic-server:~/sl2619-finetune/
ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && \
    python finetune.py --base google/gemma-3-270m-it \
        --train ./sft_v1.train.jsonl --val ./sft_v1.val.jsonl \
        --out ./checkpoints/v1'
```

### 3. Merge + quantize

```bash
# Merge LoRA adapter into full BF16
ssh nouslogic-server 'cd ~/sl2619-finetune && python merge.py \
    --base google/gemma-3-270m-it --adapter ./checkpoints/v1 --out ./merged_v1'

# Quantize to Q4_0 GGUF via llama.cpp (on server)
ssh nouslogic-server 'cd ~/llama.cpp && python convert_hf_to_gguf.py \
    ~/sl2619-finetune/merged_v1 --outfile ~/sl2619-finetune/merged_v1.bf16.gguf --outtype bf16 && \
    ./llama-quantize ~/sl2619-finetune/merged_v1.bf16.gguf \
        ~/sl2619-finetune/merged_v1.q4_0.gguf Q4_0'
```

### 4. Logits-equivalence gate

Validates that the fine-tuned Q4_0 GGUF preserves token-rank vs the BF16
reference before deploying to a target. See `docs/plans/a55-gemma-h5-logits-equivalence.md`.

```bash
# Build Q1 corpus (host)
uv run logits-equiv build-corpus --out .cache/q1/q1_corpus.txt

# Run on server (BF16 reference .kld)
# Run on x86 host (KL delta vs BF16)
# Run on A55 board (cross-arch KL delta)
# Gate: same_top_p delta ≤ 1.0 pp, max_delta ratio A55/x86 ≤ 3.0×
```

### 5. Smoke test (base vs merged)

```bash
# Build prompt bundle on host
python scripts/smoke_test.py --dry-run --bundle /tmp/smoke_bundle.json

# Run side-by-side on server
scp scripts/smoke_test.py /tmp/smoke_bundle.json nouslogic-server:~/sl2619-finetune/
ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && \
    python smoke_test.py --bundle ./smoke_bundle.json \
        --base google/gemma-3-270m-it --merged ./merged_v1 \
        --out-dir ./logs'
```

### 6. Deploy to board

See `docs/deployment/sl2619-board.md` for the cross-compile runbook and empirical
perf numbers (5.87 tok/s decode, 37.2 tok/s prompt-eval at `-t 2`).

```bash
# Deploy GGUF to board (user-performed)
scp merged_v1.q4_0.gguf nouslogic-sl2619:/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/
```

### 7. Run tests

```bash
uv run pytest               # all tests
uv run pytest tests/test_health_table.py   # specific module
```

---

## Data and model artifact policy

- **SFT datasets** (`data/sft_v1*.jsonl`) are git-tracked — they are small (< 4 MB)
  and are the canonical training artifacts for reproducibility.
- **Model weights** (`.gguf`, `.bin`, `.safetensors`, `.pt`) are gitignored.
  Store them on the GPU server or board SD card; reference them by path.
- **Fine-tune checkpoints** (`checkpoints/`) are gitignored. Use the merge + quantize
  pipeline to produce the deployable GGUF; that artifact lives on the server.
- **HuggingFace base model** (`google/gemma-3-270m-it`) is downloaded by the
  training stack at runtime — not stored in this repo. Set `HF_TOKEN` (or run
  `huggingface-cli login`) before the first SFT run; see
  [`docs/references/gemma.md`](docs/references/gemma.md) for the model card link.

---

## Relationship to SynapticSL2619

This repo is a peer of `SynapticSL2619/` on disk. It will be imported as:

```
SynapticSL2619/
└── models/
    └── gemma-3-270m-it/   ← git submodule pointing here
```

The robotic arm project's conventions (`docs/conventions/`) and board-control
code do not depend on this repo at build time — the only integration point is
the deployed GGUF on the SD card (`/mnt/sdcard/models/`).
