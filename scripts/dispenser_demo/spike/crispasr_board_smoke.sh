#!/bin/bash
# Phase 0 board smoke for CrispASR + Moonshine Streaming Tiny GGUF on SL2619.
#
# Host-side dispatcher. SSHes a single read-only pre-flight (memory, tmpfs,
# binary + model existence, SD-card space) and then runs the decode while
# polling /proc/PID/status for peak VmRSS. Honors Iron Law R3 (no persistent
# board state mutation): no writes outside what the user-supplied binary
# itself produces. The actual decode is invoked by the user running this
# script — the agent never invokes this against the board.
#
# Plan gate (docs/plans/dispenser-demo/plan.md §9 Phase 0 step 0.2):
#   RSS <= 250 MB; decode <= 2.0 s for a 3-s clip.
#
# Pre-condition: /board_probe has been run this session (the snapshot lives at
# docs/tmp/sl2619-status.md). This script will refuse to proceed without that.
#
# RAM safety: SL2619 /tmp is tmpfs (RAM-backed). Large stale files in /tmp/
# eat the same RAM the model load needs. This script lists any /tmp/* entries
# above --tmp-warn-mb and exits non-zero so the user can decide whether to
# remove them (per session preference: ask before removing).

set -euo pipefail

SSH_HOST="nouslogic-sl2619"
BIN_PATH=""
MODEL_PATH=""
WAV_PATH=""
BACKEND="moonshine-streaming"
THREADS="2"   # matches the SL2619's two A55 cores; CrispASR's default would resolve to this anyway
LATENCY_BUDGET_S="2.0"
RSS_BUDGET_MB="250"
MEM_FLOOR_MB="350"   # MemAvailable must be >= RSS_BUDGET_MB + headroom before we attempt
TMP_WARN_MB="50"
TIMEOUT_S="60"
SKIP_PROBE_CHECK=0

usage() {
    cat <<EOF
Usage: $(basename "$0") --bin PATH --model PATH --wav PATH [options]

Required (paths on the board):
  --bin PATH               aarch64 crispasr binary (e.g. /mnt/sdcard/bin/crispasr)
  --model PATH             moonshine-streaming-tiny GGUF (tokenizer must be co-located)
  --wav PATH               input WAV (16 kHz mono recommended)

Optional:
  --ssh-host HOST          SSH alias (default: ${SSH_HOST})
  --backend NAME           CrispASR backend (default: ${BACKEND})
  --threads N              '-t N' for crispasr (default: ${THREADS} for the 2-core A55)
  --latency-budget-s SEC   plan §9 gate (default: ${LATENCY_BUDGET_S})
  --rss-budget-mb MB       plan §9 gate (default: ${RSS_BUDGET_MB})
  --mem-floor-mb MB        abort if MemAvailable < this (default: ${MEM_FLOOR_MB})
  --tmp-warn-mb MB         abort if any /tmp entry exceeds this (default: ${TMP_WARN_MB})
  --timeout-s SEC          per-step SSH timeout (default: ${TIMEOUT_S})
  --skip-probe-check       proceed even if docs/tmp/sl2619-status.md is missing/stale
  -h, --help               this message

Bootstrap (run once on host before this script):
  Build crispasr for aarch64 (cross-compile or on a similar device) and scp it
  to the board. Stage the GGUF + tokenizer onto the SD card at /mnt/sdcard.
  See docs/plans/dispenser-demo/crispasr-spike-notes.md §Bootstrap.

Exit codes:
  0   PASS — decode within latency and RSS budget
  1   FAIL — decode succeeded but missed a gate
  2   FAIL — pre-flight refused (low RAM, /tmp tmpfs polluted, missing files)
  3   FAIL — /board_probe not run (override with --skip-probe-check)
  4   FAIL — decode error (non-zero exit from crispasr, timeout, etc.)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssh-host)          SSH_HOST="$2"; shift 2 ;;
        --bin)               BIN_PATH="$2"; shift 2 ;;
        --model)             MODEL_PATH="$2"; shift 2 ;;
        --wav)               WAV_PATH="$2"; shift 2 ;;
        --backend)           BACKEND="$2"; shift 2 ;;
        --threads)           THREADS="$2"; shift 2 ;;
        --latency-budget-s)  LATENCY_BUDGET_S="$2"; shift 2 ;;
        --rss-budget-mb)     RSS_BUDGET_MB="$2"; shift 2 ;;
        --mem-floor-mb)      MEM_FLOOR_MB="$2"; shift 2 ;;
        --tmp-warn-mb)       TMP_WARN_MB="$2"; shift 2 ;;
        --timeout-s)         TIMEOUT_S="$2"; shift 2 ;;
        --skip-probe-check)  SKIP_PROBE_CHECK=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${BIN_PATH}" || -z "${MODEL_PATH}" || -z "${WAV_PATH}" ]]; then
    echo "ERROR: --bin, --model, --wav are all required" >&2
    usage >&2
    exit 2
fi

# Step 0: confirm a board-probe snapshot exists. Phase 2 of the plan makes
# this mandatory, and Phase 0's board step inherits the same hygiene — we
# don't want to chase ghost failures because the board state drifted.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBE_SNAPSHOT="${REPO_ROOT}/docs/tmp/sl2619-status.md"
if [[ "${SKIP_PROBE_CHECK}" -eq 0 && ! -f "${PROBE_SNAPSHOT}" ]]; then
    echo "ERROR: ${PROBE_SNAPSHOT} not found." >&2
    echo "       Run /board_probe in Claude Code first, or pass --skip-probe-check." >&2
    exit 3
fi

echo "=== Phase 0 board smoke — CrispASR via ${SSH_HOST} ==="
echo "bin   : ${BIN_PATH}"
echo "model : ${MODEL_PATH}"
echo "wav   : ${WAV_PATH}"
echo

# Step 1: single batched read-only pre-flight. Per docs/conventions/code-style-shell.md
# §2 Rule 4 (one SSH call, multiple probes) and Rule 2 (single physical line).
echo "--- pre-flight (read-only) ---"
# shellcheck disable=SC2029  # path expansion of board-side $vars is intentional
PREFLIGHT=$(timeout "${TIMEOUT_S}" ssh "${SSH_HOST}" "echo '## meminfo'; awk '/^MemAvailable:/ {print int(\$2/1024)}' /proc/meminfo; echo '## free'; free -m; echo '## tmp_listing'; du -sm /tmp/* 2>/dev/null || true; echo '## tmp_total'; du -sm /tmp 2>/dev/null || true; echo '## df_sdcard'; df -m /mnt/sdcard 2>/dev/null || true; echo '## bin'; ls -l '${BIN_PATH}' 2>&1 || true; echo '## model'; ls -l '${MODEL_PATH}' 2>&1 || true; echo '## wav'; ls -l '${WAV_PATH}' 2>&1 || true; echo '## end'")

echo "${PREFLIGHT}"
echo

# /proc/meminfo MemAvailable is kernel-stable since 3.14 and not subject to
# BusyBox `free` column-layout variance (the "available" column may be absent
# on older BusyBox builds).
MEM_AVAIL_MB=$(echo "${PREFLIGHT}" | awk '/^## meminfo/{getline; print; exit}')
echo "MemAvailable (MB): ${MEM_AVAIL_MB}"
if [[ -z "${MEM_AVAIL_MB}" || "${MEM_AVAIL_MB}" -lt "${MEM_FLOOR_MB}" ]]; then
    echo "FAIL: insufficient RAM (need ${MEM_FLOOR_MB} MB, have ${MEM_AVAIL_MB:-unknown})" >&2
    exit 2
fi

# Pull out the /tmp listing and flag anything >= TMP_WARN_MB.
# /tmp is tmpfs (RAM-backed) on the Yocto ASTRA image — files there eat the
# same RAM crispasr needs.
TMP_BIG=$(echo "${PREFLIGHT}" | awk -v lim="${TMP_WARN_MB}" '
    /^## tmp_listing/ {section=1; next}
    /^##/ {section=0}
    section && $1 ~ /^[0-9]+$/ && $1+0 >= lim {print}
')
if [[ -n "${TMP_BIG}" ]]; then
    echo "FAIL: large files in /tmp (RAM-backed tmpfs):" >&2
    echo "${TMP_BIG}" >&2
    echo "Ask the user to remove them before re-running. (R3 keeps the agent" >&2
    echo "from doing it; the user can: ssh ${SSH_HOST} 'rm -i /tmp/<file>')." >&2
    exit 2
fi

# Confirm artifacts actually exist on the board.
if echo "${PREFLIGHT}" | grep -q "^## bin\$"; then
    if ! echo "${PREFLIGHT}" | awk '/^## bin/{getline; print}' | grep -q '^-..x'; then
        echo "FAIL: ${BIN_PATH} missing or not executable on board" >&2
        exit 2
    fi
fi
if echo "${PREFLIGHT}" | awk '/^## model/{getline; print}' | grep -q 'No such file'; then
    echo "FAIL: ${MODEL_PATH} not present on board" >&2
    exit 2
fi
if echo "${PREFLIGHT}" | awk '/^## wav/{getline; print}' | grep -q 'No such file'; then
    echo "FAIL: ${WAV_PATH} not present on board" >&2
    exit 2
fi
echo "pre-flight OK"
echo

# Step 2: decode + concurrent RSS polling. Single SSH session executes a
# small inline bash that:
#   - launches crispasr in the background, records PID + start ns
#   - samples VmRSS while alive
#   - waits and emits structured key=value lines
#
# Note: agent must NOT execute this block. The user runs this script.
echo "--- decode (live on board) ---"
REMOTE_CMD="$(cat <<REMOTE
set -eo pipefail
START_NS=\$(date +%s%N)
'${BIN_PATH}' --backend '${BACKEND}' -t '${THREADS}' -m '${MODEL_PATH}' -f '${WAV_PATH}' > /tmp/crispasr_smoke_out.\$\$ 2>&1 &
PID=\$!
MAX_KB=0
while kill -0 \$PID 2>/dev/null; do
    if [ -r /proc/\$PID/status ]; then
        cur=\$(awk '/^VmRSS:/ {print \$2}' /proc/\$PID/status 2>/dev/null || echo 0)
        if [ -n "\$cur" ] && [ "\$cur" -gt "\$MAX_KB" ]; then
            MAX_KB=\$cur
        fi
    fi
    sleep 0.05
done
wait \$PID
RC=\$?
END_NS=\$(date +%s%N)
ELAPSED_MS=\$(( (END_NS - START_NS) / 1000000 ))
echo "=== RESULT ==="
echo "exit_code=\$RC"
echo "elapsed_ms=\$ELAPSED_MS"
echo "peak_rss_kb=\$MAX_KB"
echo "=== STDOUT_BEGIN ==="
cat /tmp/crispasr_smoke_out.\$\$
echo "=== STDOUT_END ==="
rm -f /tmp/crispasr_smoke_out.\$\$
REMOTE
)"

RESULT=$(timeout "${TIMEOUT_S}" ssh "${SSH_HOST}" "bash -s" <<<"${REMOTE_CMD}")
echo "${RESULT}"
echo

# Parse + verdict
RC=$(echo "${RESULT}"   | awk -F= '/^exit_code=/    {print $2; exit}')
MS=$(echo "${RESULT}"   | awk -F= '/^elapsed_ms=/   {print $2; exit}')
RSS=$(echo "${RESULT}"  | awk -F= '/^peak_rss_kb=/  {print $2; exit}')

if [[ -z "${RC}" || -z "${MS}" || -z "${RSS}" ]]; then
    echo "FAIL: could not parse remote output" >&2
    exit 4
fi

ELAPSED_S=$(awk -v ms="${MS}" 'BEGIN {printf "%.3f", ms/1000}')
RSS_MB=$(awk -v kb="${RSS}" 'BEGIN {printf "%.1f", kb/1024}')

echo "=== verdict ==="
echo "exit_code     : ${RC}"
echo "elapsed_s     : ${ELAPSED_S} (budget ${LATENCY_BUDGET_S})"
echo "peak_rss_mb   : ${RSS_MB} (budget ${RSS_BUDGET_MB})"

if [[ "${RC}" -ne 0 ]]; then
    echo "FAIL: crispasr exited ${RC}"; exit 4
fi
if awk -v e="${ELAPSED_S}" -v b="${LATENCY_BUDGET_S}" 'BEGIN {exit !(e+0 > b+0)}'; then
    echo "FAIL: decode time over budget"; exit 1
fi
if awk -v r="${RSS_MB}" -v b="${RSS_BUDGET_MB}" 'BEGIN {exit !(r+0 > b+0)}'; then
    echo "FAIL: peak RSS over budget"; exit 1
fi
echo "PASS"
