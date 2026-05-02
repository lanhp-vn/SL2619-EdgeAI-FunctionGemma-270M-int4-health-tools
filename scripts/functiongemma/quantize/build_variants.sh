#!/bin/bash
# Generate every quantized variant of the canonical FunctionGemma 270M FP16
# GGUF and append the new sha256 hashes to CHECKSUMS.txt.
#
# Idempotent: if a variant file already exists with a matching sha256, the
# llama-quantize call is skipped. Pass --force to rebuild from scratch.
#
# Usage:
#   scripts/functiongemma/quantize/build_variants.sh
#   scripts/functiongemma/quantize/build_variants.sh --force
#
# Reproduces the on-board sweep at
# docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md.
#
# Relies on:
#   docs/references/upstream/llama.cpp/build/bin/llama-quantize  (host build)
#   releases/functiongemma-270m/001-baseline/gguf/model.gguf      (FP16 source)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LLAMA_BIN_DIR="docs/references/upstream/llama.cpp/build/bin"
QUANTIZE="$LLAMA_BIN_DIR/llama-quantize"
GGUF_DIR="releases/functiongemma-270m/001-baseline/gguf"
SOURCE="$GGUF_DIR/finetuned_functiongemma_fp16.gguf"
CHECKSUMS="$GGUF_DIR/CHECKSUMS.txt"

VARIANTS=( q4_0 q4_k_m q5_k_m q8_0 iq4_xs )

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

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
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
        # lines look like "<sha>  <filename>"; ignore comment / blank lines
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
echo "[sha256] regenerating $CHECKSUMS"
{
    cat <<'EOF'
# FunctionGemma 270M iter-001 — GGUF checksums
#
# Source FP16 (`finetuned_functiongemma_fp16.gguf`) is the deployable from
# Distil iteration 001. Originally named `model.gguf`; renamed for
# unambiguous lineage 2026-05-02.
#
# All quantized variants below were produced on host via:
#   docs/references/upstream/llama.cpp/build/bin/llama-quantize \
#       releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_fp16.gguf \
#       releases/.../gguf/finetuned_functiongemma_<quant>.gguf <QUANT>
# (llama.cpp tag b8981, 2026-04-29 host checkout).
#
# .gguf files themselves are gitignored — this txt is the only authoritative
# record committed to git. Regenerate via:
#   scripts/functiongemma/quantize/build_variants.sh

EOF
    (cd "$GGUF_DIR" && sha256sum \
        "${PREFIX}fp16.gguf" \
        "${VARIANTS[@]/#/${PREFIX}}".gguf 2>/dev/null)
} > "$CHECKSUMS.tmp"
mv "$CHECKSUMS.tmp" "$CHECKSUMS"
cat "$CHECKSUMS"
