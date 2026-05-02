#!/bin/sh
# fg-ask-board.sh — Interactive FunctionGemma 270M tool-calling on SL2619.
#
# Runs ON the board (BusyBox sh). Reads a question from $1 or stdin,
# wraps it with the pre-rendered FunctionGemma prompt prefix/suffix
# (system prompt + tool declarations from the host), and pipes to
# llama-completion. Prints the model's function call plus the
# llama_perf_context_print footer (which contains tok/s).
#
# Layout assumed (set by the host-side scp commands in
# docs/deployment/functiongemma-board-deploy.md):
#   /mnt/sdcard/models/functiongemma-270m/prompt-prefix.txt
#   /mnt/sdcard/models/functiongemma-270m/prompt-suffix.txt
#   /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_q4_0.gguf
#   /mnt/sdcard/llama-cpp/llama-completion
#
# Usage:
#   ./fg-ask-board.sh "What's my blood pressure?"
#   echo "When is my next appointment?" | ./fg-ask-board.sh
#   N_PREDICT=128 ./fg-ask-board.sh "Why am I taking Atorvastatin?"
#   FG_MODEL=<other_quant>.gguf ./fg-ask-board.sh "..."   # override (rare)
#
# Loop form (paste into the board shell after the script is in place):
#   while IFS= read -p 'fg> ' Q; do [ -z "$Q" ] && continue; ./fg-ask-board.sh "$Q"; done

set -eu

MODEL_DIR=${MODEL_DIR:-/mnt/sdcard/models/functiongemma-270m}
LLAMA=${LLAMA:-/mnt/sdcard/llama-cpp/llama-completion}
N_THREADS=${N_THREADS:-2}
N_PREDICT=${N_PREDICT:-64}
TEMP=${TEMP:-0.0}
TOP_K=${TOP_K:-1}
SEED=${SEED:-42}

PREFIX="$MODEL_DIR/prompt-prefix.txt"
SUFFIX="$MODEL_DIR/prompt-suffix.txt"
# Default to the recommended on-board variant; override via FG_MODEL env.
MODEL="$MODEL_DIR/${FG_MODEL:-finetuned_functiongemma_q4_0.gguf}"

for f in "$PREFIX" "$SUFFIX" "$MODEL" "$LLAMA"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

if [ $# -ge 1 ]; then
    Q=$*
else
    Q=$(cat)
fi

if [ -z "$Q" ]; then
    echo "fg-ask-board.sh: empty question" >&2
    exit 1
fi

TMPF=$(mktemp /tmp/fg-ask.XXXXXX)
trap 'rm -f "$TMPF"' EXIT

# Prefix already ends at the close of the system+tools turn; we just append
# the user question and the saved suffix (`<end_of_turn>\n<start_of_turn>model\n`).
{
    cat "$PREFIX"
    printf '%s' "$Q"
    cat "$SUFFIX"
} > "$TMPF"

# `-r '<end_function_call>'` stops decode at the first complete tool call —
# the user-reported repeated-call drift on greedy decoding.
# Stderr (perf footer) is left on stderr so tok/s is visible per turn.
"$LLAMA" \
    -m "$MODEL" \
    -f "$TMPF" \
    -t "$N_THREADS" -n "$N_PREDICT" \
    --temp "$TEMP" --top-k "$TOP_K" --seed "$SEED" \
    -r '<end_function_call>' \
    -no-cnv --single-turn
