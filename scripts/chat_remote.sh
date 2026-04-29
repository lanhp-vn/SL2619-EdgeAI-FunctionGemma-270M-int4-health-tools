#!/usr/bin/env bash
# chat_remote.sh — One-shot interactive chat against the FT'd Gemma 3 270M Q4_0
# on the SL2619 via SSH-piped llama-completion.
#
# Usage:
#   scripts/chat_remote.sh "what is my heart rate?"
#   scripts/chat_remote.sh "summarize my current medications"
#   echo "what am I allergic to?" | scripts/chat_remote.sh
#
# Or, with custom params:
#   N_PREDICT=64 SEED=7 scripts/chat_remote.sh "what is my next appointment?"
#
# Renders the §4 directive system prompt + YAML record + user question locally
# via `prompt_composer.compose_user_text`, pipes the body to the board's
# `llama-completion --jinja --no-display-prompt -p "$BODY"` over SSH stdin,
# captures only the model reply (R3 SSH-read-only — nothing is written to the
# board). Matches the Q4 bench envelope so behavior here predicts bench
# behavior. See docs/tmp/bench/2026-04-28_gemma3-finetuned-final.md §1 for
# the working recipe rationale.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
HEALTH_TABLE="${HEALTH_TABLE:-${TOOLS_DIR}/data/health_table_v1.yaml}"

SSH_HOST="${SSH_HOST:-nouslogic-sl2619}"
LLAMA_BIN="${LLAMA_BIN:-/mnt/sdcard/llama-cpp/llama-completion}"
LLAMA_MODEL="${LLAMA_MODEL:-/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf}"
N_THREADS="${N_THREADS:-2}"
N_PREDICT="${N_PREDICT:-128}"
TEMP="${TEMP:-0.0}"
TOP_K="${TOP_K:-1}"
SEED="${SEED:-42}"

# Read question from $1 if given, otherwise stdin.
if [[ $# -ge 1 ]]; then
    QUESTION="$*"
else
    QUESTION="$(cat)"
fi
if [[ -z "${QUESTION// }" ]]; then
    echo "chat_remote.sh: empty question (pass as arg or via stdin)" >&2
    exit 1
fi

# Render the body locally — same prompt_composer the bench + SFT uses, so
# the on-device prompt shape is identical to the training distribution.
BODY="$(cd "$TOOLS_DIR" && uv run python -c "
import sys
from datetime import date
from pathlib import Path
from gemma_tools.prompt_composer import compose_user_text
from gemma_tools.health_table import load_health_table
ht = load_health_table(Path('${HEALTH_TABLE}'))
sys.stdout.write(compose_user_text(ht, date.today(), '''${QUESTION//\'/\'\\\'\'}'''))
")"

# Pipe via SSH stdin. The remote BODY=$(cat) absorbs stdin into a shell var
# so we don't have to ferry 2.6 KB of YAML through ssh argv (shell quoting
# explodes on the embedded single-quote dates / newlines).
# shellcheck disable=SC2029  # local expansion of LLAMA_BIN/LLAMA_MODEL/etc is intentional
printf '%s' "$BODY" | ssh "$SSH_HOST" "BODY=\$(cat); ${LLAMA_BIN} \
    -m ${LLAMA_MODEL} \
    --jinja --no-display-prompt \
    -p \"\$BODY\" \
    -t ${N_THREADS} -n ${N_PREDICT} \
    --temp ${TEMP} --top-k ${TOP_K} --seed ${SEED} \
    -no-cnv --single-turn 2>/dev/null"
