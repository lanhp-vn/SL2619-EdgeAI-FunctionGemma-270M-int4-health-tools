#!/bin/bash
# Minimal FunctionGemma on-board prompt runner.
# Usage: ./run-prompt.sh "What is my blood pressure?"
# Shows raw model output + llama.cpp perf footer, no parsing.

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <prompt>" >&2
    exit 2
fi

prompt="$1"
model_dir="/mnt/sdcard/models/functiongemma-270m"
llama_bin="/mnt/sdcard/llama-cpp/llama-completion"
threads=2
n_predict=64
temp=0.0
top_k=1
seed=42

if [ ! -f "$llama_bin" ]; then
    echo "ERROR: llama-completion not found at $llama_bin" >&2
    exit 2
fi

if [ ! -f "$model_dir/model.gguf" ]; then
    echo "ERROR: model.gguf not found at $model_dir/model.gguf" >&2
    exit 2
fi

if [ ! -f "$model_dir/prompt-prefix.txt" ] || [ ! -f "$model_dir/prompt-suffix.txt" ]; then
    echo "ERROR: prompt templates missing at $model_dir/" >&2
    exit 2
fi

# Build prompt and run inference
# Use /tmp with a simple filename (busybox mktemp doesn't support -t flag)
tmpfile="/tmp/fg_prompt_$$.txt"
trap "rm -f $tmpfile" EXIT

cat "$model_dir/prompt-prefix.txt" > "$tmpfile"
echo -n "$prompt" >> "$tmpfile"
cat "$model_dir/prompt-suffix.txt" >> "$tmpfile"

echo "[fg] prompt: $prompt" >&2
echo "[fg] running inference..." >&2

"$llama_bin" \
    -m "$model_dir/model.gguf" \
    -f "$tmpfile" \
    -t $threads \
    -c 2048 \
    -n $n_predict \
    --temp $temp \
    --top-k $top_k \
    --seed $seed \
    -no-cnv --single-turn
