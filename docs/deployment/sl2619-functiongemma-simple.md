# FunctionGemma 270M Simple Board Workflow

Simplified SSH-based testing flow for the SL2619 board. No REPL complexity, no host-side Python dependencies, no parsing — just raw model output + llama.cpp perf footer.

## Overview

The workflow splits prompt rendering (requires HF tokenizer) from inference:
1. **Host** generates `prompt-prefix.txt` and `prompt-suffix.txt` once using the real tokenizer
2. **Board** receives pre-rendered templates + model GGUF + minimal Python/shell wrapper
3. **User** SSHes into board and runs a simple command: `./run-prompt.sh "question"`
4. **Output** is raw FunctionGemma completion + llama.cpp perf metrics

## Pre-flight

- Board status: run `/board_probe --target=sl2619` before deployment
- Model ready: local `model.gguf` exists (518 MiB)
- Board binary: `/mnt/sdcard/llama-cpp/llama-completion` present (cross-compiled build `0adede8`)
- Storage: `/mnt/sdcard` has >100 GiB free

## Host Setup (one-time)

Generate the prompt templates on your development machine:

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune

# Generate prompt-prefix.txt and prompt-suffix.txt
uv run python scripts/gen_prompt_templates.py \
    --tokenizer model/ \
    --output-dir /tmp/fg_board_files/
```

This produces:
- `prompt-prefix.txt` (7.2 KiB) — system prompt + tools + user msg opening
- `prompt-suffix.txt` (35 B) — assistant turn opening

Verify the generated files:
```bash
ls -lh /tmp/fg_board_files/
```

## Deploy to Board

Run these commands from the WSL host (R3: agent does NOT write to the board):

```bash
# 1. Create model directory on board
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/models/functiongemma-270m'

# 2. Copy model GGUF (518 MiB, ~15–20 s on Wi-Fi)
scp /home/lanhp-wsl/nouslogic/gemma3-270M-finetune/model.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/model.gguf

# 3. Copy prompt templates
scp /tmp/fg_board_files/{prompt-prefix.txt,prompt-suffix.txt} \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# 4. Copy health table (patient record)
scp /tmp/fg_board_files/health_table.json \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# 5. Copy board scripts
scp /home/lanhp-wsl/nouslogic/gemma3-270M-finetune/scripts/fg-chat-board.py \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/
scp /tmp/fg_board_files/run-prompt.sh \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# 6. Verify on board
ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/models/functiongemma-270m/ && echo && sha256sum /mnt/sdcard/models/functiongemma-270m/model.gguf'
```

## Usage

### Option A: Direct shell command (simplest)

SSH to the board and run a single prompt:

```bash
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "What is my blood pressure?"'
```

Expected output:
```
[fg] prompt: What is my blood pressure?
[fg] running inference...
[model output]
<start_function_call>call:get_vitals{}<end_function_call>

common_perf_print: prompt eval time =  1234.56 ms / 567 tokens ( 45.9 tokens per second)
common_perf_print: eval time =    78.90 ms /  12 tokens (  5.1 tokens per second)
...
```

### Option B: Interactive REPL (fg-chat-board.py)

For multi-turn conversation:

```bash
ssh nouslogic-sl2619 'python3 /mnt/sdcard/models/functiongemma-270m/fg-chat-board.py'
```

Then type prompts interactively (slash commands: `/exit`, `/reset`, `/history`, `/raw`).

Shorthand for a single probe:

```bash
ssh nouslogic-sl2619 'python3 /mnt/sdcard/models/functiongemma-270m/fg-chat-board.py --probe "When is my next appointment?"'
```

## Test Cases

Verify correct tool routing with these prompts:

```bash
# get_vitals
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "What is my blood pressure?"'

# get_next_appointment
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "When is my next appointment?"'

# check_food_interaction
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "Can I drink alcohol?"'

# list_allergies
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "What am I allergic to?"'

# get_medications_at_time
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "What pills do I take at 8 AM?"'
```

## Expected Performance

**Board (A55 2-core, CPU only):**
- Decode tok/s: ~5–7 (cold model load ~40–60 s total)
- Prompt tok/s: ~30–50
- Per-prompt wall: ~45–60 s (cold KV alloc dominated)

**Host (10-thread WSL2, for comparison):**
- Decode tok/s: ~50
- Overall tok/s: ~750

## Known Limitations

- **No quantization yet** — using full FP16 GGUF. Q4_0 variant (~130 MiB, 3× faster) is a future task.
- **No NPU path** — only CPU. A Torq/IREE build exists at `/mnt/sdcard/torq-examples/` but is separate from llama.cpp.
- **Prompt templates are pre-rendered** — the board cannot be used standalone for arbitrary finetuned chat templates (no tokenizer). For fully on-device deployment, future work includes shipping the tokenizer to the board.
- **No tool result formatting** — the shell wrapper shows raw function calls only. The Python script (fg-chat-board.py) formats results into English sentences.

## Files

| Path | Size | Purpose |
|---|---|---|
| `scripts/gen_prompt_templates.py` | 7 KiB | Host: generates prompt templates |
| `scripts/fg-chat-board.py` | 25 KiB | Board: interactive REPL (pure stdlib) |
| `/mnt/sdcard/models/functiongemma-270m/` | — | Board deployment directory |
| `prompt-prefix.txt` | 7.2 KiB | System prompt + tools + user msg opening |
| `prompt-suffix.txt` | 35 B | Assistant turn opening |
| `health_table.json` | 3.2 KiB | Patient record (YAML→JSON) |
| `model.gguf` | 518 MiB | FunctionGemma 270M FP16 GGUF |
| `run-prompt.sh` | 1.5 KiB | Board: minimal shell wrapper for direct commands |
| `fg-chat-board.py` | 25 KiB | Board: Python REPL alternative |

## Troubleshooting

| Issue | Fix |
|---|---|
| `llama-completion: file not found` | Board build missing or path wrong; re-run board bootstrap (see sl2619-board.md §2) |
| `model.gguf: file not found` | Run the scp command in Deploy §2 |
| `prompt-prefix.txt: file not found` | Run gen_prompt_templates.py on host (Deploy §1) |
| Load fails: `unknown model architecture` | Board llama.cpp too old; rebuild on nouslogic-server with latest tag |
| Output is garbage / no parseable call | Special-token-as-bytes regression; check chat_template.jinja alignment in the prefix |
| Decode rate <1 tok/s | Board under heavy load; monitor `top` — close other processes or reboot |

## See Also

- `docs/deployment/sl2619-functiongemma.md` — original full bench workflow (host-side JSONL logging, rich metrics)
- `docs/deployment/sl2619-board.md` — board cross-compile and baseline setup
- `docs/plans/FunctionGemma/README.md` — SFT recipe and training analysis
