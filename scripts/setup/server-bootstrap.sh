#!/bin/bash
# server-bootstrap.sh — Idempotent Ubuntu-server bootstrap for SL2619 Gemma 3 270M SFT.
#
# Purpose:
#   Phase 0 H2 in docs/plans/AI-models/a55-gemma-fine-tune.md. Provisions a clean
#   x86_64 Ubuntu 24.04 host (RTX 5080 / Blackwell sm_120) with PyTorch + the
#   Google-canonical fine-tune stack (transformers/trl/peft/accelerate/bitsandbytes/
#   datasets/sentencepiece/tensorboard) and a host-side llama.cpp build for
#   convert + quantize. Smoke-checks CUDA + bf16 matmul + bitsandbytes + (optional)
#   Gemma 3 tokenizer load, then prints a concise PASS/FAIL summary.
#
# Idempotent: re-running reuses the existing venv, leaves an existing llama.cpp
# clone alone (does a fast-forward pull), and lets pip handle already-installed
# packages natively. No destructive actions are taken without explicit flags.
#
# Usage:
#   server-bootstrap.sh [options]
#
#   --with-system-deps    sudo apt install python3.12-venv python3-dev build-essential git curl
#   --use-nightly-pytorch use the PyTorch nightly cu128 index (only if stable cu128 lacks sm_120)
#   --no-llama-cpp        skip the llama.cpp clone+build (if you only need the SFT stack)
#   --smoke-tokenizer     extra smoke: load google/gemma-3-270m-it tokenizer (needs HF auth)
#   -h, --help            show this help
#
# Environment overrides:
#   SL2619_FT_HOME        workspace root (default: $HOME/sl2619-finetune)
#   LLAMA_CPP_DIR         llama.cpp checkout (default: $HOME/llama.cpp)
#
# Conventions:
#   docs/conventions/code-style-shell.md (bash 5+, set -euo pipefail, [[ ]], $(...), shellcheck-clean)

set -euo pipefail
IFS=$'\n\t'

#region Configuration
readonly WORKSPACE_DEFAULT="${HOME}/sl2619-finetune"
readonly LLAMA_CPP_DEFAULT="${HOME}/llama.cpp"
readonly PYTORCH_INDEX_STABLE="https://download.pytorch.org/whl/cu128"
readonly PYTORCH_INDEX_NIGHTLY="https://download.pytorch.org/whl/nightly/cu128"

WORKSPACE="${SL2619_FT_HOME:-$WORKSPACE_DEFAULT}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$LLAMA_CPP_DEFAULT}"

WITH_SYSTEM_DEPS=false
USE_NIGHTLY=false
INSTALL_LLAMA_CPP=true
SMOKE_TOKENIZER=false

# Set by Phase 4 and consumed by Phase 5/6 to keep cu128 wheels from being
# silently downgraded to CPU wheels by downstream pip resolutions
# (bitsandbytes / transformers / accelerate all declare torch deps).
PYTORCH_INDEX_USED=""
PIP_CONSTRAINTS=""
#endregion

#region Helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()   { printf '%s [%s] %s\n' "$(date '+%H:%M:%S')" "$1" "${*:2}"; }
info()  { log INFO  "$*"; }
warn()  { printf '%b%s [WARN] %s%b\n'  "$YELLOW" "$(date '+%H:%M:%S')" "$*" "$NC" >&2; }
err()   { printf '%b%s [ERROR] %s%b\n' "$RED"    "$(date '+%H:%M:%S')" "$*" "$NC" >&2; }
die()   { err "$*"; exit 1; }

declare -a CHECK_RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0

record_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    CHECK_RESULTS+=("PASS  $1${2:+ — $2}")
}
record_fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    CHECK_RESULTS+=("FAIL  $1${2:+ — $2}")
}

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}
#endregion

#region CLI
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --with-system-deps)    WITH_SYSTEM_DEPS=true; shift ;;
            --use-nightly-pytorch) USE_NIGHTLY=true; shift ;;
            --no-llama-cpp)        INSTALL_LLAMA_CPP=false; shift ;;
            --smoke-tokenizer)     SMOKE_TOKENIZER=true; shift ;;
            -h|--help)             usage; exit 0 ;;
            *)                     die "unknown option: $1 (use --help)" ;;
        esac
    done
}
#endregion

#region Phase 1 — environment detection
detect_environment() {
    info "Phase 1 — environment detection"

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        info "OS: ${PRETTY_NAME:-unknown}"
        record_pass "OS detected" "${PRETTY_NAME:-unknown}"
        if [[ "${ID:-}" != "ubuntu" ]]; then
            warn "Non-Ubuntu host (${ID:-?}) — script tested on Ubuntu 24.04 only"
        fi
    else
        record_fail "OS detection" "/etc/os-release missing"
    fi

    if ! command -v python3 >/dev/null; then
        die "python3 not on PATH — install with: sudo apt install python3"
    fi
    local pyver
    pyver="$(python3 --version 2>&1 | awk '{print $2}')"
    info "Python: $pyver"
    if [[ "$pyver" =~ ^3\.(11|12|13) ]]; then
        record_pass "Python ≥ 3.11" "$pyver"
    else
        record_fail "Python ≥ 3.11" "found $pyver"
    fi

    for tool in gcc make cmake git curl; do
        if command -v "$tool" >/dev/null; then
            info "$tool: $(command -v "$tool")"
        else
            warn "$tool missing — re-run with --with-system-deps or install manually"
        fi
    done

    if command -v nvidia-smi >/dev/null; then
        local gpu_name driver_ver
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
        driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
        if [[ -n "$gpu_name" ]]; then
            info "GPU: $gpu_name | Driver: $driver_ver"
            record_pass "NVIDIA GPU + driver" "$gpu_name (driver $driver_ver)"
        else
            record_fail "NVIDIA GPU + driver" "nvidia-smi present but no GPU reported"
        fi
    else
        record_fail "NVIDIA driver" "nvidia-smi not on PATH"
        warn "Install the NVIDIA driver before training: sudo apt install nvidia-driver-580-open"
    fi

    local free_kb total_ram_gi
    free_kb="$(df -k "${HOME}" | awk 'NR==2 {print $4}')"
    total_ram_gi="$(awk '/^MemTotal:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
    local free_gb=$(( free_kb / 1024 / 1024 ))
    info "Disk free at \$HOME: ${free_gb} GB | RAM total: ${total_ram_gi} GiB"
    if (( free_gb < 20 )); then
        warn "Less than 20 GB free at \$HOME — fine-tune + GGUF needs ≥ 15 GB headroom"
        record_fail "Disk ≥ 20 GB at \$HOME" "${free_gb} GB"
    else
        record_pass "Disk ≥ 20 GB at \$HOME" "${free_gb} GB"
    fi
}
#endregion

#region Phase 2 — system deps (opt-in)
install_system_deps() {
    if ! $WITH_SYSTEM_DEPS; then
        info "Phase 2 — system deps SKIPPED (use --with-system-deps to enable)"
        return 0
    fi
    info "Phase 2 — apt install python3.12-venv python3-dev build-essential git curl"
    if ! command -v sudo >/dev/null; then
        die "sudo not available; cannot install system deps"
    fi

    # Fail fast if sudo would block on a password prompt with no TTY (the classic
    # `ssh host '<cmd>'` foot-gun — non-interactive ssh has no pty for sudo to read on).
    # `sudo -n true` exits 0 if NOPASSWD is configured OR creds are still cached.
    if ! sudo -n true 2>/dev/null && [[ ! -t 0 ]]; then
        err "Phase 2 needs sudo but no TTY is attached and sudo credentials aren't cached."
        err "Re-run one of these instead (the script is idempotent — re-runs are safe):"
        err "  (a) ssh -t <host> '~/server-bootstrap.sh --with-system-deps'"
        err "  (b) ssh <host>              # interactive shell, then run the script there"
        err "  (c) install the packages manually once, then re-run without --with-system-deps:"
        err "      sudo apt-get update -y && sudo apt-get install -y python3.12-venv python3-dev build-essential git curl"
        die "aborting — fix the invocation and retry"
    fi

    sudo apt-get update -y
    sudo apt-get install -y python3.12-venv python3-dev build-essential git curl
    record_pass "System deps installed"
}
#endregion

#region Phase 3 — workspace + venv
setup_workspace() {
    info "Phase 3 — workspace + venv at $WORKSPACE"
    mkdir -p "$WORKSPACE/data" "$WORKSPACE/runs" "$WORKSPACE/checkpoints"

    local venv="$WORKSPACE/.venv"
    local venv_state
    if [[ -d "$venv" && -x "$venv/bin/python" ]]; then
        info "Reusing existing venv at $venv"
        venv_state="reused"
    else
        info "Creating venv at $venv"
        python3 -m venv "$venv"
        venv_state="created"
    fi
    record_pass "venv ready" "$venv ($venv_state)"
    # shellcheck disable=SC1091
    source "$venv/bin/activate"
    pip install --upgrade pip wheel setuptools >/dev/null
    record_pass "pip + wheel upgraded" "pip $(pip --version | awk '{print $2}')"
}
#endregion

#region Phase 4 — PyTorch (sm_120)
install_pytorch() {
    info "Phase 4 — PyTorch with CUDA 12.8 (Blackwell sm_120 support)"
    local index_url="$PYTORCH_INDEX_STABLE"
    local pre_flag=""
    if $USE_NIGHTLY; then
        info "Using nightly cu128 index"
        index_url="$PYTORCH_INDEX_NIGHTLY"
        pre_flag="--pre"
    fi
    # shellcheck disable=SC2086
    pip install $pre_flag --index-url "$index_url" torch torchvision torchaudio
    local torch_ver torchvision_ver torchaudio_ver
    torch_ver="$(python -c 'import torch; print(torch.__version__)')"
    torchvision_ver="$(python -c 'import torchvision; print(torchvision.__version__)')"
    torchaudio_ver="$(python -c 'import torchaudio; print(torchaudio.__version__)')"

    # Pin the cu128 wheels so downstream pip calls (bnb, transformers, etc.) cannot
    # re-resolve torch and silently downgrade to a CPU wheel from PyPI default index.
    PIP_CONSTRAINTS="$WORKSPACE/.torch-pin.txt"
    PYTORCH_INDEX_USED="$index_url"
    cat > "$PIP_CONSTRAINTS" <<EOF
torch==$torch_ver
torchvision==$torchvision_ver
torchaudio==$torchaudio_ver
EOF
    info "Wrote pip constraints to $PIP_CONSTRAINTS (locks torch=$torch_ver, torchvision=$torchvision_ver, torchaudio=$torchaudio_ver)"

    record_pass "PyTorch installed" "$torch_ver"
}
#endregion

#region Phase 5 — SFT stack
install_sft_stack() {
    info "Phase 5 — transformers / trl / datasets / peft / accelerate / bitsandbytes / sentencepiece / tensorboard"
    if [[ -z "$PYTORCH_INDEX_USED" || -z "$PIP_CONSTRAINTS" ]]; then
        die "internal: Phase 4 did not set PYTORCH_INDEX_USED / PIP_CONSTRAINTS — aborting before Phase 5 to avoid CPU-wheel downgrade"
    fi
    # --extra-index-url + -c pin keep pip's resolver from swapping the cu128 torch wheel
    # for a PyPI CPU wheel when bnb / transformers / accelerate's torch>=2.3 constraint is re-checked.
    pip install -U \
        --extra-index-url "$PYTORCH_INDEX_USED" \
        -c "$PIP_CONSTRAINTS" \
        transformers \
        trl \
        datasets \
        accelerate \
        evaluate \
        sentencepiece \
        peft \
        tensorboard \
        huggingface_hub
    # bitsandbytes is best-effort; on rare GPUs / older glibc it may need a custom build.
    if pip install -U --extra-index-url "$PYTORCH_INDEX_USED" -c "$PIP_CONSTRAINTS" bitsandbytes; then
        record_pass "SFT stack installed (with bitsandbytes)"
    else
        record_fail "bitsandbytes install" "fell back to CPU-only path; QLoRA needs bnb"
    fi

    # Sanity-check that torch is still cu128-tagged after Phase 5 — surfaces silent regressions early.
    local torch_post
    torch_post="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "import-failed")"
    if [[ "$torch_post" == *"+cu"* || "$torch_post" == *"+rocm"* ]]; then
        record_pass "torch still on GPU wheel after Phase 5" "$torch_post"
    else
        record_fail "torch downgraded to CPU wheel during Phase 5" "now $torch_post — pip resolver clobbered cu128 build despite constraint file"
    fi
}
#endregion

#region Phase 6 — llama.cpp (convert/quantize)
install_llama_cpp() {
    if ! $INSTALL_LLAMA_CPP; then
        info "Phase 6 — llama.cpp SKIPPED (--no-llama-cpp)"
        return 0
    fi
    info "Phase 6 — llama.cpp at $LLAMA_CPP_DIR"
    if [[ -d "$LLAMA_CPP_DIR/.git" ]]; then
        info "llama.cpp already cloned — fetching latest"
        git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin master >/dev/null 2>&1 || true
        git -C "$LLAMA_CPP_DIR" pull --ff-only origin master >/dev/null 2>&1 \
            || warn "git pull --ff-only failed; using existing tree as-is"
    else
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    fi
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
          -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
    cmake --build "$LLAMA_CPP_DIR/build" -j"$(nproc)" --target llama-quantize >/dev/null
    # llama.cpp's requirements-convert_hf_to_gguf.txt declares conservative legacy
    # pins (`torch~=2.6.0`, `transformers<5.0.0`) that collide with our cu128 torch
    # (2.11.x from Phase 4) and the transformers ≥5 from Phase 5. Installing the
    # full reqs file either fails as ResolutionImpossible (when our pin is in
    # place) or silently downgrades torch to a CPU wheel (when it isn't). The
    # convert_hf_to_gguf.py script itself runs fine on the newer versions — it
    # only needs `gguf` beyond what Phase 5 already provides (numpy, torch,
    # sentencepiece, transformers, protobuf, safetensors via transformers).
    if pip install --extra-index-url "$PYTORCH_INDEX_USED" -c "$PIP_CONSTRAINTS" gguf >/dev/null; then
        # gguf uses importlib.metadata for its version, not a __version__ attr.
        local gguf_ver
        gguf_ver="$(python -c 'from importlib.metadata import version; print(version("gguf"))' 2>/dev/null || echo "installed")"
        record_pass "gguf installed" "$gguf_ver"
    else
        record_fail "gguf install" "convert_hf_to_gguf.py will not be importable"
    fi
    if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]]; then
        record_pass "llama-quantize built" "$LLAMA_CPP_DIR/build/bin/llama-quantize"
    else
        record_fail "llama-quantize binary missing" "build target failed"
    fi
    if [[ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]]; then
        record_pass "convert_hf_to_gguf.py present"
    else
        record_fail "convert_hf_to_gguf.py missing"
    fi
}
#endregion

#region Phase 7 — smoke tests
smoke_tests() {
    info "Phase 7 — smoke tests"

    # 7.1 Python imports
    if python - <<'PY' >/dev/null 2>&1
import torch, transformers, trl, peft, datasets, accelerate, sentencepiece  # noqa: F401
PY
    then
        record_pass "Python imports clean"
    else
        record_fail "Python imports" "one of torch/transformers/trl/peft/datasets/accelerate/sentencepiece"
    fi

    # 7.2 CUDA available
    if python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        record_pass "torch.cuda.is_available()"
    else
        record_fail "CUDA not available" "wheel built without CUDA, or driver not loaded"
        return 0
    fi

    # 7.3 GPU name + capability
    local cap_pair gpu_name
    cap_pair="$(python -c 'import torch; print(*torch.cuda.get_device_capability(0))' 2>/dev/null || true)"
    gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || true)"
    if [[ -n "$cap_pair" && -n "$gpu_name" ]]; then
        local cap_compact="${cap_pair// /}"
        record_pass "GPU capability" "sm_${cap_compact} ($gpu_name)"
        if [[ "$cap_pair" == "12 0" ]]; then
            info "Confirmed Blackwell sm_120 — RTX 5080 path validated"
        fi
    else
        record_fail "GPU capability query"
    fi

    # 7.4 bf16 matmul
    if python - <<'PY' >/dev/null 2>&1
import torch
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.shape == (2048, 2048) and y.dtype == torch.bfloat16
PY
    then
        record_pass "bf16 matmul on GPU"
    else
        record_fail "bf16 matmul" "PyTorch wheel may lack sm_120 kernels — try --use-nightly-pytorch"
    fi

    # 7.5 bitsandbytes 4-bit forward pass via BOTH paths the trainer actually uses:
    # (a) bnb.nn.Linear4bit direct (kwargs: bare `quant_type` / `compute_dtype`),
    # (b) transformers.BitsAndBytesConfig (kwargs: `bnb_4bit_quant_type` / `bnb_4bit_compute_dtype`).
    # The two name styles are NOT interchangeable — getting them wrong fires a
    # Python TypeError before any GPU code runs and silently masks real bnb issues.
    # Capture stderr so the FAIL row carries the actual exception.
    local bnb_err_log="${WORKSPACE}/.bnb-smoke.err"
    if python - <<'PY' 1>/dev/null 2>"$bnb_err_log"
import bitsandbytes as bnb
import torch
from transformers import BitsAndBytesConfig

# (a) Direct Linear4bit — bare kwargs.
linear = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
_ = linear(x)
torch.cuda.synchronize()

# (b) BitsAndBytesConfig — bnb_4bit_-prefixed kwargs (this is what the trainer uses).
_ = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
PY
    then
        record_pass "bitsandbytes 4-bit GPU smoke"
        rm -f "$bnb_err_log"
    else
        local bnb_err_one
        bnb_err_one="$(tail -n 1 "$bnb_err_log" 2>/dev/null | tr -d '\n' | head -c 200)"
        record_fail "bitsandbytes 4-bit" "${bnb_err_one:-no stderr captured} — full traceback at $bnb_err_log"
        warn "bnb traceback ↓"
        cat "$bnb_err_log" >&2 || true
    fi

    # 7.6 Optional: tokenizer load (gated by flag, needs HF auth)
    if $SMOKE_TOKENIZER; then
        if python - <<'PY' >/dev/null 2>&1
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
ids = tok("hello world", return_tensors="pt").input_ids
assert ids.shape[1] > 0
PY
        then
            record_pass "Gemma 3 tokenizer load"
        else
            record_fail "Gemma 3 tokenizer load" "missing HF auth or network egress"
        fi
    else
        info "Tokenizer smoke skipped (use --smoke-tokenizer to enable; requires 'hf auth login')"
    fi
}
#endregion

#region Phase 8 — final summary
print_summary() {
    printf '\n%b================ BOOTSTRAP SUMMARY ================%b\n' "$CYAN" "$NC"
    local line
    for line in "${CHECK_RESULTS[@]}"; do
        if [[ "$line" == PASS* ]]; then
            printf '%b  %s%b\n' "$GREEN" "$line" "$NC"
        else
            printf '%b  %s%b\n' "$RED" "$line" "$NC"
        fi
    done
    printf '\n  PASS: %d   FAIL: %d\n\n' "$PASS_COUNT" "$FAIL_COUNT"
    if (( FAIL_COUNT == 0 )); then
        printf '%bRESULT: PASS%b\n' "$GREEN" "$NC"
        printf 'Workspace : %s\n' "$WORKSPACE"
        printf 'Activate  : source %s/.venv/bin/activate\n' "$WORKSPACE"
        printf 'llama.cpp : %s\n' "$LLAMA_CPP_DIR"
        printf 'Log       : %s\n' "$LOG_FILE"
    else
        printf '%bRESULT: FAIL — %d check(s) failed%b\n' "$RED" "$FAIL_COUNT" "$NC"
        printf 'Log       : %s\n' "$LOG_FILE"
        exit 1
    fi
}
#endregion

#region Main
main() {
    parse_args "$@"
    mkdir -p "$WORKSPACE"
    LOG_FILE="${WORKSPACE}/bootstrap-$(date '+%Y%m%d-%H%M%S').log"
    readonly LOG_FILE
    # Tee from this point onward so every step is captured (use FD-3 trick to keep stderr coloring).
    exec > >(tee -a "$LOG_FILE") 2>&1

    info "server-bootstrap.sh starting at $(date -Iseconds)"
    info "Workspace      : $WORKSPACE"
    info "llama.cpp dir  : $LLAMA_CPP_DIR"
    info "System deps    : $($WITH_SYSTEM_DEPS && echo on || echo off)"
    info "PyTorch nightly: $($USE_NIGHTLY && echo on || echo off)"
    info "llama.cpp build: $($INSTALL_LLAMA_CPP && echo on || echo off)"
    info "Tokenizer smoke: $($SMOKE_TOKENIZER && echo on || echo off)"

    detect_environment
    install_system_deps
    setup_workspace
    install_pytorch
    install_sft_stack
    install_llama_cpp
    smoke_tests
    print_summary
}

main "$@"
#endregion
