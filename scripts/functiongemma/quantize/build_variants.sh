#!/bin/bash
# Generate the recommended Q4_0 (default) — or every variant from the
# 2026-05-02 quant sweep (`--all`) — from a canonical FP16 GGUF.
#
# Idempotent: if the target file already exists with a sha256 matching the
# committed CHECKSUMS.txt, the llama-quantize call is skipped. Pass
# `--force` to rebuild from scratch.
#
# Usage:
#   scripts/functiongemma/quantize/build_variants.sh                       # iter-001 Q4_0 (recommended on-board variant)
#   scripts/functiongemma/quantize/build_variants.sh --all                 # iter-001 Q4_0 + Q4_K_M + Q5_K_M + Q8_0 + IQ4_XS
#   scripts/functiongemma/quantize/build_variants.sh --force               # iter-001 rebuild Q4_0 from scratch
#   scripts/functiongemma/quantize/build_variants.sh --release-dir releases/functiongemma-270m/002-dispenser-demo
#                                                                          # iter-002 Q4_0; reuses iter-001 sweep notes
#   scripts/functiongemma/quantize/build_variants.sh --release-dir <path> --all --force
#                                                                          # full sweep, any release dir
#
# The 2026-05-02 sweep at
# docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md
# DISQUALIFIED Q4_K_M / Q5_K_M / Q8_0 / IQ4_XS on the SL2619 board
# (K-quant scale-factor encoding skew vs the on-board llama-completion
# build); only Q4_0 ships in production for iter-001 and (until a fresh
# sweep proves otherwise) iter-002.
#
# Relies on:
#   docs/references/upstream/llama.cpp/build/bin/llama-quantize       (host build)
#   <release-dir>/gguf/<prefix>fp16.gguf                              (FP16 source)
# where <prefix> is auto-derived from the unique `*fp16.gguf` filename in <release-dir>/gguf/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LLAMA_BIN_DIR="docs/references/upstream/llama.cpp/build/bin"
QUANTIZE="$LLAMA_BIN_DIR/llama-quantize"

# Default release dir = iter-001 baseline (back-compat). Override with
# --release-dir for iter-002+ deployables.
RELEASE_DIR="releases/functiongemma-270m/001-baseline"

# Map of variant -> llama-quantize ftype enum name
declare -A FTYPE=(
    [q4_0]=Q4_0
    [q4_k_m]=Q4_K_M
    [q5_k_m]=Q5_K_M
    [q8_0]=Q8_0
    [iq4_xs]=IQ4_XS
)

# Default to the recommended on-board variant; --all expands to the full
# sweep set.
VARIANTS=( q4_0 )
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)         VARIANTS=( q4_0 q4_k_m q5_k_m q8_0 iq4_xs ); shift ;;
        --force)       FORCE=1; shift ;;
        --release-dir) RELEASE_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

GGUF_DIR="$RELEASE_DIR/gguf"
CHECKSUMS="$GGUF_DIR/CHECKSUMS.txt"

# Auto-derive PREFIX from the FP16 source filename. Each release ships
# exactly one `*fp16.gguf` (iter-001 = finetuned_functiongemma_fp16.gguf,
# iter-002 = finetuned_dispenser_fp16.gguf, etc.). Fail loud if zero or
# multiple match — that's authoring drift, not a recoverable default.
mapfile -t FP16_MATCHES < <(find "$GGUF_DIR" -maxdepth 1 -name '*fp16.gguf' -print 2>/dev/null | sort)
if [[ "${#FP16_MATCHES[@]}" -eq 0 ]]; then
    echo "ERROR no *fp16.gguf in $GGUF_DIR — extract the Distil bundle first." >&2
    exit 2
fi
if [[ "${#FP16_MATCHES[@]}" -gt 1 ]]; then
    echo "ERROR multiple *fp16.gguf in $GGUF_DIR; expected exactly one:" >&2
    printf '  %s\n' "${FP16_MATCHES[@]}" >&2
    exit 2
fi
SOURCE="${FP16_MATCHES[0]}"
# Strip the leading dir + trailing "fp16.gguf" to recover the prefix.
PREFIX="$(basename "$SOURCE")"
PREFIX="${PREFIX%fp16.gguf}"
if [[ -z "$PREFIX" || "$PREFIX" == "fp16.gguf" ]]; then
    echo "ERROR could not derive PREFIX from $SOURCE (expected <prefix>fp16.gguf)" >&2
    exit 2
fi

if [[ ! -x "$QUANTIZE" ]]; then
    echo "ERROR llama-quantize not found at $QUANTIZE." >&2
    echo "  Build it from $REPO_ROOT/docs/references/upstream/llama.cpp/" >&2
    echo "  with cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-quantize" >&2
    exit 2
fi

if [[ ! -f "$SOURCE" ]]; then
    echo "ERROR FP16 source GGUF missing: $SOURCE" >&2
    exit 2
fi

# Read existing CHECKSUMS.txt (filename -> expected sha) so we can skip
# rebuilds when nothing changed.
declare -A EXPECTED_SHA=()
if [[ -f "$CHECKSUMS" ]]; then
    while read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        sha="$(awk '{print $1}' <<<"$line")"
        fn="$(awk '{print $2}' <<<"$line")"
        if [[ -n "$fn" ]]; then
            EXPECTED_SHA["$fn"]="$sha"
        fi
    done < "$CHECKSUMS"
fi

needs_rebuild() {
    local fn="$1" path="$GGUF_DIR/$1"
    [[ "$FORCE" == "1" ]] && return 0
    [[ ! -f "$path" ]] && return 0
    local expected="${EXPECTED_SHA[$fn]:-}"
    if [[ -n "$expected" ]]; then
        local actual
        actual="$(sha256sum "$path" | awk '{print $1}')"
        [[ "$actual" != "$expected" ]] && return 0
    fi
    return 1
}

for v in "${VARIANTS[@]}"; do
    out="${PREFIX}${v}.gguf"
    if needs_rebuild "$out"; then
        echo "[build] $out (${FTYPE[$v]}) ..."
        LD_LIBRARY_PATH="$LLAMA_BIN_DIR" "$QUANTIZE" \
            "$SOURCE" "$GGUF_DIR/$out" "${FTYPE[$v]}" 8
    else
        echo "[skip ] $out (sha256 matches CHECKSUMS.txt)"
    fi
done

echo ""
echo "[sha256 verification — only entries on disk]"
(cd "$GGUF_DIR" && sha256sum "${PREFIX}fp16.gguf" \
    $(for v in "${VARIANTS[@]}"; do printf '%s ' "${PREFIX}${v}.gguf"; done) 2>/dev/null)
