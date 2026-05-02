#!/bin/bash
# Generate the recommended Q4_0 (default) — or every variant from the
# 2026-05-02 quant sweep (`--all`) — from the canonical FunctionGemma 270M
# FP16 GGUF.
#
# Idempotent: if the target file already exists with a sha256 matching the
# committed CHECKSUMS.txt, the llama-quantize call is skipped. Pass
# `--force` to rebuild from scratch.
#
# Usage:
#   scripts/functiongemma/quantize/build_variants.sh                # Q4_0 only (recommended on-board variant)
#   scripts/functiongemma/quantize/build_variants.sh --all          # Q4_0 + Q4_K_M + Q5_K_M + Q8_0 + IQ4_XS (sweep reproduction)
#   scripts/functiongemma/quantize/build_variants.sh --force        # rebuild Q4_0 from scratch
#   scripts/functiongemma/quantize/build_variants.sh --all --force  # rebuild every variant
#
# Reproduces the sweep at
# docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md.
# That sweep DISQUALIFIED Q4_K_M / Q5_K_M / Q8_0 / IQ4_XS on the SL2619
# board (K-quant scale-factor encoding skew vs the on-board llama-completion
# build) — only Q4_0 ships in production.
#
# Relies on:
#   docs/references/upstream/llama.cpp/build/bin/llama-quantize          (host build)
#   releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_fp16.gguf  (FP16 source)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LLAMA_BIN_DIR="docs/references/upstream/llama.cpp/build/bin"
QUANTIZE="$LLAMA_BIN_DIR/llama-quantize"
GGUF_DIR="releases/functiongemma-270m/001-baseline/gguf"
SOURCE="$GGUF_DIR/finetuned_functiongemma_fp16.gguf"
CHECKSUMS="$GGUF_DIR/CHECKSUMS.txt"

# Map of variant -> llama-quantize ftype enum name
declare -A FTYPE=(
    [q4_0]=Q4_0
    [q4_k_m]=Q4_K_M
    [q5_k_m]=Q5_K_M
    [q8_0]=Q8_0
    [iq4_xs]=IQ4_XS
)

# All output files use this prefix so their lineage from the canonical
# Distil iteration-001 FP16 deployable is unambiguous.
PREFIX="finetuned_functiongemma_"

# Default to the recommended on-board variant; --all expands to the full
# sweep set.
VARIANTS=( q4_0 )
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --all)   VARIANTS=( q4_0 q4_k_m q5_k_m q8_0 iq4_xs ) ;;
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

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
