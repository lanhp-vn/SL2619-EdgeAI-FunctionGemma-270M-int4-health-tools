# FunctionGemma SL2619 Simplified Deployment — Quick Start

Copy-paste workflow to deploy FunctionGemma to the board for simple SSH-based testing.

## Step 1: Generate Prompt Templates (Host)

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune

# Activate venv and generate templates
source .venv/bin/activate
python3 scripts/gen_prompt_templates.py \
    --tokenizer model/ \
    --output-dir /tmp/fg_deploy/

# Convert YAML to JSON
python3 << 'CONVERT'
import json
import yaml
with open("data/health_table_v1.yaml") as f:
    data = yaml.safe_load(f)
with open("/tmp/fg_deploy/health_table.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("[done] health_table.json created")
CONVERT

# Copy shell wrapper
cp scripts/fg-chat-board.py /tmp/fg_deploy/
cat > /tmp/fg_deploy/run-prompt.sh << 'WRAPPER'
#!/bin/bash
set -e
if [ $# -ne 1 ]; then
    echo "Usage: $0 <prompt>" >&2
    exit 2
fi
prompt="$1"
model_dir="/mnt/sdcard/models/functiongemma-270m"
llama_bin="/mnt/sdcard/llama-cpp/llama-completion"
if [ ! -f "$llama_bin" ] || [ ! -f "$model_dir/model.gguf" ]; then
    echo "ERROR: llama-completion or model.gguf not found" >&2
    exit 2
fi
tmpfile=$(mktemp /tmp/fg_prompt_XXXXXX.txt)
trap "rm -f $tmpfile" EXIT
cat "$model_dir/prompt-prefix.txt" > "$tmpfile"
echo -n "$prompt" >> "$tmpfile"
cat "$model_dir/prompt-suffix.txt" >> "$tmpfile"
echo "[fg] prompt: $prompt" >&2
echo "[fg] running inference..." >&2
"$llama_bin" -m "$model_dir/model.gguf" -f "$tmpfile" -t 2 -c 2048 -n 64 \
    --temp 0.0 --top-k 1 --seed 42 -no-cnv --single-turn
WRAPPER
chmod +x /tmp/fg_deploy/run-prompt.sh

# Verify
ls -lh /tmp/fg_deploy/
```

## Step 2: Copy Files to Board

```bash
# Create board directory
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/models/functiongemma-270m'

# Copy model GGUF (518 MiB)
scp /home/lanhp-wsl/nouslogic/gemma3-270M-finetune/model.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# Copy templates and scripts
scp /tmp/fg_deploy/{prompt-prefix.txt,prompt-suffix.txt,health_table.json,fg-chat-board.py,run-prompt.sh} \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/

# Verify
ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/models/functiongemma-270m/ && echo && sha256sum /mnt/sdcard/models/functiongemma-270m/model.gguf'
```

## Step 3: Test

### Direct command (simplest):
```bash
ssh nouslogic-sl2619 'bash /mnt/sdcard/models/functiongemma-270m/run-prompt.sh "What is my blood pressure?"'
```

### Interactive REPL:
```bash
ssh nouslogic-sl2619 'python3 /mnt/sdcard/models/functiongemma-270m/fg-chat-board.py'
```

### Single probe (no REPL):
```bash
ssh nouslogic-sl2619 'python3 /mnt/sdcard/models/functiongemma-270m/fg-chat-board.py --probe "When is my next appointment?"'
```

## Expected Output

Raw command:
```
[fg] prompt: What is my blood pressure?
[fg] running inference...
<start_function_call>call:get_vitals{}<end_function_call>
common_perf_print: prompt eval time = 1234.56 ms / 567 tokens (45.9 tokens per second)
common_perf_print: eval time =    78.90 ms /  12 tokens ( 5.1 tokens per second)
...
```

## See Also

- Full details: `docs/deployment/sl2619-functiongemma-simple.md`
- Architecture: `docs/deployment/sl2619-board.md`
- Training: `docs/plans/FunctionGemma/README.md`

## Notes

- Model load is cold; first prompt takes ~45–60 s
- No quantization yet (Q4_0 variant would be ~3× faster)
- No NPU acceleration (CPU only)
- All board commands shown above should be run by the user, not via agent (R3)
