# 15 — Model Compiler & Runtime (SL2619 Working Knowledge)

> Consolidated working knowledge about the **model compiler and runtime stacks** used on and around the SL2619. Covers Torq (NPU VMFB), llama.cpp (A55 CPU GGUF), and onnxruntime (A55 CPU ONNX). Captured from Phase 1 through Phase 1.5 Phase D (Gemma fine-tune). Normative one-line rule lives in [IL-12](00-iron-laws.md#il-12-sl2619-ai--synap-framework--torq-npu-backend); this file is the depth pointer.
>
> **Canonical ownership (per `13-documentation-update-protocol.md §10.1`):**
>
> - Cross-phase *patterns* (this file) — compiler/runtime selection, version coupling, artifact storage, verified recipes, known pitfalls.
> - Phase-specific current recipes — `phase1-plan.md §P3/§T4/§T5/§T6`, `models-testing-plan.md §9`, `a55-gemma-fine-tune.md`, future `phase<N>-plan.md`.
> - Decision logs / post-mortems — `backlogs.md §1.8–§1.21`.
> - Per-submodule orientation — `docs/references/llama-cpp.md`, `docs/references/onnx.md`.

---

## 0. Shared mental model

### 0.1 Inference domains

SL2619 has two distinct inference domains. They are not interchangeable.

| Domain | Compute target | Runtime | Format | Current use |
|---|---|---|---|---|
| **NPU** | Synaptics Torq NPU (fixed-function) + Google Coral (RISC-V fallback) | `torq-runtime` (IREE HAL) | `.vmfb` packaged in `.synap` or bare | Vision — YOLOv8n-320×320-INT8 @ ~71 ms |
| **A55 CPU** | 2 × Cortex-A55 @ 2 GHz (Linux-exposed only; cores 2–3 reserved for ATF/OP-TEE) | `llama.cpp` or `onnxruntime` | `.gguf` (Q4_0) or `.onnx` | SLM — Gemma 3 270M-IT (GGUF); STT — Moonshine Tiny (ONNX) |

**Product split (current, user sign-off 2026-04-25)**: vision → NPU; SLM → A55 CPU. NPU is reserved for fixed-function vision models where CMA latency matters. SLM-on-NPU (P1/P2 paths) is quality-gated out — the A55 CPU Q4_0 GGUF path proved both faster (5.87 tok/s vs 1.7 tok/s on NPU) and simpler to maintain.

### 0.2 Per-runtime workflow template

Every model that lands on the SL2619 follows the same lifecycle shape:

```
1. Export      — pull or convert HF checkpoint to interchange format
                 (ONNX, TFLite, HF safetensors)
2. Compile /   — transform interchange format to on-board artifact
   Convert       Torq: torq-compile → .vmfb → .synap bundle
                 llama.cpp: convert_hf_to_gguf.py → BF16.gguf → llama-quantize → Q4_0.gguf
                 onnxruntime: vendor ONNX pulled from HF (no host compile step)
3. Host smoke  — load-test on host BEFORE scp to board;
                 catch format/shape/ABI mismatches early (cheaper than board round-trips)
4. Deploy      — scp to /mnt/sdcard/models/<model>/ (user-performed per R3)
5. Verify      — smoke-test on board (board SSH read-only for agent)
6. Bench       — capture perf + quality numbers in docs/tmp/bench/<date>_<model>-*.md
7. Freeze      — update this file + the per-model README.md with the outcome
```

Phase-specific recipes instantiate steps 1–7 in detail; this file captures the cross-cutting rules.

### 0.3 Runtime selection decision rule

```
Is the model a CNN / vision model (Conv, GEMM, Attention) with ≤ ~10 MB VMFB?
  → Torq NPU path (§2)

Is the model a text-generation LLM (autoregressive, ≥ 100M params)?
  → llama.cpp Q4_0 GGUF path (§3); SLM-on-NPU is ruled out until further notice

Is the model a sequence-to-sequence ONNX (e.g. encoder+decoder, Moonshine)?
  → onnxruntime direct path (§4)

Does the model use a GGUF → LoRA split? (Plan-B for fine-tune)
  → llama-completion --lora; runtime merging; deferred to §3.4
```

---

## 1. Runtime inventory

| Runtime | Type | Format | On-board artifact dir | Host tooling | Doc pointer |
|---|---|---|---|---|---|
| `torq-runtime` 1.5.0 | IREE HAL Python wheel (aarch64) | `.vmfb` (bare or in `.synap`) | `/mnt/sdcard/models/<model>/*.vmfb` | `torq-compiler` Docker `:v1.5`, `torq-tools` | §2 + `docs/datasheets/torq-architecture.md` |
| `llama.cpp` b8925 | Cross-compiled C++ binary (aarch64) | `.gguf` (Q4_0) | `/mnt/sdcard/llama-cpp/` (binaries), `/mnt/sdcard/models/<model>/*.gguf` | `convert_hf_to_gguf.py`, `llama-quantize` (host) | §3 + `docs/references/llama-cpp.md` |
| `onnxruntime` 1.25.0 | pip wheel (aarch64) | `.onnx` (encoder + decoder) | `/mnt/sdcard/models/<model>/` | vendor HF export (no compile) | §4 + `docs/references/onnx.md` |

---

## 2. Torq — NPU inference (VMFB / .synap)

### 2.1 Three-stack architecture

Torq is **three separate repositories**, each with a distinct role. Mixing them up is the source of most confusion.

| Repo | Purpose | Where used | Our pin |
|---|---|---|---|
| `torq-compiler` | Host-only. MLIR+IREE compiler that turns TFLite/ONNX/Torch → VMFB. Ships as a Docker image on GHCR. | Host (dev laptop/WSL) | `ghcr.io/synaptics-torq/torq-compiler/compiler:v1.5` |
| `torq-runtime` | On-board. Python wheel wrapping IREE's HAL torq driver. Loads VMFB + dispatches to the Coral NPU kernel. | SL2619 Python 3.12 | `torq_runtime-1.5.0-cp312-cp312-manylinux_2_28_aarch64.whl` |
| `torq-tools` | Host-only. Model-prep scaffolding — HuggingFace static-graph export (`torq-export-model`), bf16/int dtype conversion (`torq-convert-dtype`), inference runners for verification. | Host | `main` HEAD |

The **`synap-runtime`** on-board (from the SDK) is the **orchestration layer on top of** `torq-runtime`. `synap_cli`, `synapinfer` GStreamer element, and the SyNAP C++ `Network::load_model()` all dispatch via `torq-runtime` when the `.synap` bundle wraps a VMFB. This is built with `ENABLE_TORQRUNTIME=ON` per `references/Synaptics/sdk/meta-synaptics/recipes-devtools/synasdk/synasdk-synap-runtime_git.bb:46`.

Two dispatch paths exist on-board:

1. **`.synap` via SyNAP C++ API or `synap_cli`** — used by Phase 1's YOLOv8n bench, `a55/hello_npu`. C++-friendly, no Python needed. See `phase1-plan.md §T7`.
2. **`.vmfb` via Python `torq.runtime`** — used by the Gemma 3 demo pattern in `references/Synaptics/torq-examples/gemma3/`. Python-friendly, no `.synap` wrapper. Used by Phase 1.5 Phase B.

Both hit the same kernel driver (`syna_npu` / `f7600000.synpu`); the choice is about API convenience.

### 2.2 Compiler tag ↔ ASTRA SDK coupling (the single most load-bearing constraint)

**Rule**: the `torq-compiler` image tag MUST match the IREE bytecode version of the `torq-runtime` on the board, which MUST match the ASTRA SDK that built the Yocto image.

Current coupling (2026-04-23):

| Board image | On-board IREE runtime | Acceptable compiler tags | Forbidden tags |
|---|---|---|---|
| `scarthgap_6.12_v2.3.0` (our board) | IREE LLVM 19 / bytecode v15 | **`:v1.5`** (and digest pin `sha256:e2f0450777cfe11fc27860647bf2a49936d7fc930eb4b4a024ccb31022a10bc1`) | `:main` (LLVM 22 / bytecode v16) |

**Failure mode when tags drift**: `Network::load_model()` rejects the VMFB at load time with `runtime supports 15.0, module has 16.0`. Fully recoverable — just rebuild with `:v1.5`. Full trace in `backlogs.md §1.12`.

**When upgrading**: if we move to a newer ASTRA SDK, re-derive this coupling from `references/Synaptics/torq-compiler/doc/user-manual/release_notes.md` and the new SDK's IREE runtime version. Don't assume `:main` is safe.

### 2.3 `torq-runtime` wheel — numpy 1.x ABI (despite metadata saying otherwise)

**Known upstream bug** (to report): `torq_runtime-1.5.0` wheel metadata declares `numpy>2.0.0b1`, but the bundled `iree.runtime._binding._runtime` C extension was compiled against numpy **1.x**. Installing numpy 2.x and importing `torq.runtime` fails at import with:

```
ImportError: A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.4.4 as it may crash. ...
```

**Workaround**: force-install numpy 1.26.4 with `--no-deps` to bypass pip's check of the wheel's declared dep:

```
rm -rf /tmp/p15site/numpy /tmp/p15site/numpy.libs
pip install --target=/tmp/p15site --no-deps numpy==1.26.4
```

`--force-reinstall` alone is **insufficient** with `--target=` mode — pip skips the existing directory unless you delete it or pass `--upgrade`. Explicit `rm -rf` is the bulletproof path.

**onnxruntime 1.25.0 works with numpy 1.x** — it shipped dual-ABI wheels through the numpy 2.0 transition. Not guaranteed for every package; if a second package joins the stack with only numpy-2.x ABI, we'd need two Python sub-environments or a package-level split.

### 2.4 Stripped-Yocto-Python bootstrap — the 19-shim pattern

The stock `scarthgap_6.12_v2.3.0` image ships Python 3.12.9 **with `pip`, `venv`, `ensurepip`, and 12 pure-Python stdlib modules stripped** (Yocto omits `python3-misc`, `python3-fcntl`, `python3-pip`, `python3-venv`, `python3-xml`, etc. for flash-size reasons). No out-of-the-box way to install Python packages.

#### 2.4.1 Diagnostic: probe the exact stripping scope

```bash
ssh nouslogic-sl2619 'python3 - <<"PROBE"
candidates = ["colorsys","getpass","mailbox","statistics","pty","fcntl","plistlib","zipapp",
              "compileall","tomllib","xmlrpc","py_compile","filecmp","tty","xml.etree.ElementTree"]
for m in candidates:
    try: __import__(m); print(f"OK   {m}")
    except ImportError as e: print(f"MISSING {m}")
PROBE'
```

#### 2.4.2 Confirmed stripped modules (as of 2026-04-23)

| Module | Type | Resolution |
|---|---|---|
| `colorsys`, `getpass`, `mailbox`, `statistics`, `pty`, `plistlib`, `zipapp`, `py_compile`, `tty`, `filecmp` | pure Python | Ship unmodified from CPython **`v3.12.9`** (matches board exactly) into `/tmp/p15site/` |
| `tomllib/` | pure Python package (4 files) | Ship from CPython v3.12.9 |
| `xmlrpc/{client,server}.py` | pure Python package (3 files) | Ship **no-op stub** — upstream's `xmlrpc/client.py` itself imports `xml.parsers.expat` which is ALSO stripped; stubs bypass the transitive gap |
| `fcntl` | C extension | Ship **no-op stub** — real syscalls unavailable; file locking degenerates to no-ops (safe for single-process tmpfs install) |
| `compileall` | pure Python (but heavy deps) | Ship **hybrid stub** — `compile_file` delegates to `py_compile` (for real .pyc output), `compile_dir`/`compile_path` are no-ops |
| `xml.etree.ElementTree`, `curses`, `sqlite3` | C-ext-backed or package | Don't need for pip bootstrap; if a downstream package hits them, ship CPython v3.12.9 sources + note the `_elementtree` C-ext limitation |
| `distutils` | removed in Python 3.12 via PEP 632 | **Don't ship** — pip handles its absence with try/except |

**The asymmetric stub strategy**: when shimming a module, check whether the caller post-verifies the side-effect. If yes (pip's `assert os.path.exists(pyc_path)` after `compileall.compile_file`), make the stub do real work. If no (`compile_dir`, `xmlrpc.client.ServerProxy` — only subclassed at import), no-op is safe.

#### 2.4.3 Bootstrap recipe (the one we proved works)

One-time host staging:

```bash
mkdir -p /tmp/p15-stage && curl -sSL https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/p15-stage/get-pip.py
cd /tmp/p15-stage/stdlib-shims && mkdir -p tomllib xmlrpc
for mod in colorsys getpass mailbox statistics pty plistlib zipapp compileall py_compile tty filecmp; do
  curl -fsSL "https://raw.githubusercontent.com/python/cpython/v3.12.9/Lib/${mod}.py" -o "${mod}.py"
done
for f in __init__.py _re.py _parser.py _types.py; do
  curl -fsSL "https://raw.githubusercontent.com/python/cpython/v3.12.9/Lib/tomllib/${f}" -o "tomllib/${f}"
done
# xmlrpc/client.py + server.py → stubs (see §2.4.4)
# compileall.py → hybrid stub (see §2.4.4)
# fcntl.py → hand-written no-op stub
tar czf /tmp/p15-stage/stdlib-shims.tar.gz -C /tmp/p15-stage/stdlib-shims .
```

On-board bootstrap (**user action — R3**):

```bash
scp /tmp/p15-stage/{get-pip.py,p15-env.sh,stdlib-shims.tar.gz} nouslogic-sl2619:/tmp/
ssh nouslogic-sl2619 'mkdir -p /tmp/p15site && rm -rf /tmp/p15site/__pycache__ /tmp/p15site/xmlrpc/__pycache__ /tmp/p15site/tomllib/__pycache__ && tar xzf /tmp/stdlib-shims.tar.gz -C /tmp/p15site/'
ssh nouslogic-sl2619 'PYTHONPATH=/tmp/p15site PYTHONUSERBASE=/tmp/pipbase python3 /tmp/get-pip.py --user --no-warn-script-location'
ssh nouslogic-sl2619 '. /tmp/p15-env.sh && pip --version'
# Expected: pip 26.0.1 from /tmp/pipbase/lib/python3.12/site-packages/pip (python 3.12)
```

#### 2.4.4 Stub sources (hand-authored — not upstream CPython)

**`fcntl.py` (no-op stub)**:

```python
F_DUPFD=0; F_DUPFD_CLOEXEC=1030; F_GETFD=1; F_SETFD=2; F_GETFL=3; F_SETFL=4
LOCK_SH=1; LOCK_EX=2; LOCK_UN=8; LOCK_NB=4; FD_CLOEXEC=1; FASYNC=8192; O_NONBLOCK=2048
def fcntl(fd, op, arg=0):
    if op in (F_GETFL, F_GETFD): return 0
    return arg if isinstance(arg, int) else 0
def ioctl(fd, op, arg=0, mutate_flag=True):
    return arg if isinstance(arg, (bytes, bytearray)) else 0
def flock(fd, op): return None
def lockf(fd, op, length=0, start=0, whence=0): return None
```

**`compileall.py` (hybrid stub)** — `compile_file` does real work via `py_compile` (pip post-verifies .pyc exists), others are no-ops:

```python
def compile_file(fullname, ddir=None, force=False, rx=None, quiet=0, legacy=False,
                 optimize=-1, invalidation_mode=None, *args, **kwargs):
    try:
        import py_compile
        py_compile.compile(fullname, doraise=True, optimize=optimize)
        return True
    except Exception:
        if quiet < 2:
            import sys, traceback; traceback.print_exc(file=sys.stderr)
        return False
def compile_dir(*args, **kwargs): return True
def compile_path(*args, **kwargs): return True
PY_FILE_REGEX = r'\.py$'
```

**`xmlrpc/client.py` (no-op stub)** — distlib subclasses `ServerProxy`/`Transport` at import time; stubs satisfy that without dragging in the missing `xml` package:

```python
class _NotAvailable:
    def __init__(self, *a, **k):
        raise RuntimeError("xmlrpc.client stub: XML-RPC unavailable on this image")
ServerProxy = Transport = SafeTransport = MultiCall = Marshaller = Unmarshaller = GzipDecodedResponse = _NotAvailable
class Fault(Exception):
    def __init__(self, faultCode=0, faultString='', **k):
        super().__init__(faultString); self.faultCode=faultCode; self.faultString=faultString
class ProtocolError(Exception): ...
class ResponseError(Exception): ...
class Error(Exception): ...
MAXINT = 2**31 - 1; MININT = -(2**31)
def Boolean(v=False, *a, **k): return bool(v)
def DateTime(v=None, *a, **k): return v
def Binary(d=None, *a, **k): return d
def loads(*a, **k):   raise RuntimeError("xmlrpc.client.loads stub: not available")
def dumps(*a, **k):   raise RuntimeError("xmlrpc.client.dumps stub: not available")
def getparser(*a, **k): raise RuntimeError("xmlrpc.client.getparser stub: not available")
```

### 2.5 NPU session lifecycle (`syna_npu` driver state)

Empirical pattern discovered 2026-04-24 during Phase 1.5 Phase B (Gemma 3 270M-IT bench).

#### 2.5.1 Cross-process NPU cycling degrades, does not reset

Observed sequence:

| Process | CmaFree at start | Outcome |
|---|---|---|
| Process 1 (C1 sweep) | 246 MiB | Success — 3 runs completed cleanly |
| Process 2 (P1 sweep) | 240 MiB | **FAIL**: `failed to start network via IOCTL: Cannot allocate memory` |
| (After reboot) Process 1 (full sweep) | 487 MiB | Success — 18 runs completed cleanly |

The real constraint is driver-side: IREE Torq HAL's `TorqExecutable.cc` registers a per-execution-context descriptor with `syna_npu`; across process cycles these descriptors aren't fully released. **No recovery path exists short of reboot.** The driver is kernel-built (no loadable-module support — `/lib/modules` is empty), so `modprobe -r syna_npu` is unavailable.

#### 2.5.2 Normative mitigation — single-process multi-task design

For any NPU-heavy bench: **use ONE Python process that holds the runner instance for the full batch.** Warmup runs once; subsequent inferences reuse already-pinned CMA weights.

For Phase 2+ production design: **a long-lived A55 service** (systemd unit) that instantiates the model once at boot and serves requests over a socket / RPC. Spawning a Python process per request is NOT viable on this HAL.

#### 2.5.3 Clean-bench cycle (combined with env-on-SD, §5.4)

```bash
ssh nouslogic-sl2619 'reboot' 2>/dev/null ; sleep 45 ; ssh nouslogic-sl2619 'mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard && ln -sfn /mnt/sdcard/p15site /tmp/p15site && ln -sfn /mnt/sdcard/pipbase /tmp/pipbase && ln -sfn /mnt/sdcard/p15-env.sh /tmp/p15-env.sh && grep CmaFree /proc/meminfo'
```

Expected: `CmaFree: ~490000 kB` on fresh boot.

#### 2.5.4 Vendor-parameter-minimalism discipline

Two independent bench-harness bugs in Phase B were both caused by over-parameterizing an adapted vendor runner:

1. **`max_prompt_tokens=80`** silently truncated user queries to 0 tokens when the system prompt happened to be 80 tokens.
2. **Client-side `MAX_NEW_TOKENS` break** caused vendor-managed stats fields to stay at 0 initial values.

**Rule**: when adapting a vendor Python runner, default to vendor parameter values (check `infer.py` for baseline). Override only when there's a documented reason, and verify via a smoke run that the override doesn't collide with internal guards.

### 2.6 Host-side compile install chain

When compiling a new HF model → VMFB on the host (any model that isn't vendor-pre-built like Gemma 3), four installs must coexist in a single Python venv. Vendor docs don't spell this out.

#### 2.6.1 The four-part install chain (order matters)

| # | What | Where from | Why required | Fails if omitted |
|---|---|---|---|---|
| 1 | **`torq-tools`** (editable install) | `references/Synaptics/torq-tools/` submodule | Provides `torq.models.*` + `torq.tools.*` + the `torq-export-model` / `torq-convert-dtype` / `bundle-vmfb` CLIs. | CLI missing at shell. |
| 2 | **`torq-compiler` release tarball** | `github.com/synaptics-torq/torq-compiler/releases/download/<tag>/release.tar.gz` | Ships `torq.compile.*` + `torq.utils.*` + `torq-compile` / `iree-compile` / `iree-opt` binaries + `libTorqDialect.so`. `torq-tools` unconditionally imports `from torq.compile import add_iree_args` at CLI startup. | `ModuleNotFoundError: No module named 'torq.compile'` at CLI invocation. |
| 3 | **HF-hub + transformers dep-pin reconciliation** | `pip install 'transformers<4.58,>=4.36' 'huggingface_hub<1.0,>=0.34'` | Compiler's own `requirements.txt` pins `huggingface_hub==1.3.2`. That's incompatible with `transformers<4.58`, which requires `huggingface_hub<1.0`. Downgrading is safe. | `ImportError: huggingface-hub>=0.34.0,<1.0 is required, found 1.3.2`. |
| 4 | **`torq_runtime` wheel for host (x86_64, not aarch64)** | `github.com/synaptics-torq/torq-compiler/releases/download/<tag>/torq_runtime-<tag>-cp312-cp312-manylinux_2_28_x86_64.whl` | SmolLM2 static-cache runner imports `from torq.runtime import ...` during host-side export validation. Note: this is the *x86_64* wheel for host exports — different file than the on-board aarch64 wheel. | `ModuleNotFoundError: No module named 'torq.runtime'`. |

**The `torq.__path__` output should list TWO directories** — `torq-tools/src/torq` and `torq-compiler-release/release/python/torq`. That's the PEP 420 namespace-package merge working correctly.

#### 2.6.2 Known vendor gap — `torq-export-model` hardcodes `llvm-cpu` target

`torq-tools/src/torq/model_export/onnx.py:246-252` calls `export_iree(...)` **without `target=`**, so IREE compilation silently defaults to `--iree-hal-target-backends=llvm-cpu` (x86). The resulting VMFB is a host CPU binary and will not load via the on-board Torq HAL.

**Normative route** (two-CLI pivot, proven):

```bash
# 1. Get the bf16 ONNX (skip IREE compilation on export)
torq-export-model smollm2 -s 135M --instruct-model --convert-dtypes --skip-iree --models-dir /tmp/models

# 2. Compile the ONNX to NPU-targeted VMFB
torq-compile-model -t torq /tmp/models/.../model.bf16.onnx -o /tmp/models/model.vmfb
```

#### 2.6.3 Memory ceiling — SmolLM2-360M iree-compile needs ≥ 48 GiB

Empirical peak on WSL2: `anon-rss 29,910,988 kB` (≈29.9 GiB) in a single `iree-compile` process. Requirements:

- **WSL2** — `.wslconfig` `memory=24GB swap=24GB` on a 40 GiB Windows host.
- **Native Linux** — 32 GiB RAM + 16 GiB swap.
- **Cloud/office server** — recommended for any model ≥ 360M params.

The 135M variant peaks near 8–12 GiB and runs comfortably on a 16 GiB laptop.

#### 2.6.4 CLI flag truth (`torq-export-model smollm2`)

| Flag | Spelling (source-verified) | Meaning |
|---|---|---|
| `-s, --model-size` | `135M` / `360M` / `1.7B` | Size variant. |
| `--instruct-model` | store_true | Flips HF repo to `…-Instruct`. |
| `--convert-dtypes` | **plural**, store_true (not `--convert-dtype`) | Triggers the bf16 conversion pass. Singular form is rejected. |
| `--extract-embeddings` | store_true | Writes embedding LUT to `token_embeddings.npy` outside the VMFB. |
| `--skip-validation` | store_true | **Required workaround** for vendor bug in `_inference.py:462` (`self._token_embeddings` attribute never set when `--extract-embeddings` used). |
| `--skip-iree` | store_true | Stops after ONNX; use with §2.6.2 two-CLI pivot. |
| `--dynamic-models` | store_true | DO NOT use for NPU — dynamic shapes are CPU-LLVM path only. |

#### 2.6.5 Paste-artifact defense — long commands become script files

Bash-tool invocations of `torq-export-model` / `torq-compile-model` with 10+ flags routinely get corrupted by WSL terminal line-wrapping (observed: `SmolLM2` → `S molLM2` with an injected space at column 80).

**Rule**: any single-line command > 80 characters → write to a shell script file and invoke as `bash /path/to/script.sh`. Script files MUST use `set -euo pipefail` at the top.

### 2.7 Known upstream bugs

| Bug | Evidence | Where it bit us |
|---|---|---|
| `torq_runtime-1.5.0` wheel metadata declares `numpy>2.0.0b1` but C-ext is numpy 1.x | `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.4` | Phase A Step 3c, fixed in §2.3 |
| `torq-compile :v1.5` `--torq-hw=512:2:450:+m:nss_v1` (custom spec form) rejected with `Unable to find css config +m` | Phase 1 T3 | `backlogs.md §1.10` — use `--torq-hw=SL2610` preset |
| `:v1.5` CSS stack overflow on YOLOv8n detection-head `tosa.mul` (cryptic `<null>` error) | Phase 1 T4 | `backlogs.md §1.12` — workaround: `--torq-disable-css` |
| `torq-export-model` hardcodes `llvm-cpu` target, producing host-CPU VMFB silently | Phase C SmolLM2 | `backlogs.md §1.20` — use §2.6.2 two-CLI pivot |

### 2.8 File-path map

| Topic | File |
|---|---|
| Overall Torq compile user manual | `references/Synaptics/torq-compiler/doc/user-manual/index.md` |
| Getting started (host install) | `references/Synaptics/torq-compiler/doc/user-manual/getting_started.md` |
| TFLite/ONNX → MLIR/VMFB recipe | `references/Synaptics/torq-compiler/doc/user-manual/model_conversion.md` |
| **Op coverage (what the NPU supports)** | `references/Synaptics/torq-compiler/doc/user-manual/ops.md` |
| CSS host fallback rules | `references/Synaptics/torq-compiler/doc/user-manual/css_host_fallback.md` |
| Custom HW flags (`--torq-hw=...`) | `references/Synaptics/torq-compiler/doc/user-manual/custom_hw.md` |
| **SyNAP ↔ Torq integration (bundle format)** | `references/Synaptics/torq-compiler/doc/user-manual/torq-synap-integration.md` |
| Profiling flags + CSV columns | `references/Synaptics/torq-compiler/doc/user-manual/profiling.md` |
| Release notes (verified models, IREE bytecode version) | `references/Synaptics/torq-compiler/doc/user-manual/release_notes.md` |
| Runtime Python bindings | `references/Synaptics/torq-compiler/runtime/bindings/python/` |
| torq-compile CLI source | `references/Synaptics/torq-compiler/compiler/tools/torq-compile-main.cc` |
| **Gemma 3 270M demo (runner, setup)** | `references/Synaptics/torq-examples/gemma3/` |
| torq_runtime aarch64 wheel URL | `references/Synaptics/torq-examples/README.md:7-9` |
| KV-cache decoder runner (reusable) | `references/Synaptics/torq-examples/utils/cache_runner.py` |
| **SmolLM2 export pipeline** | `references/Synaptics/torq-tools/src/torq/models/smollm2/` |
| **Moonshine export pipeline (NPU path)** | `references/Synaptics/torq-tools/src/torq/models/moonshine/` |
| HF → ONNX static export helpers | `references/Synaptics/torq-tools/src/torq/model_export/` |
| fp32 → bf16/fp16/int dtype conversion | `references/Synaptics/torq-tools/src/torq/tools/convert_dtype/` |
| Torq architecture overview | `docs/datasheets/torq-architecture.md` |

### 2.9 Verified-working models (as of compiler `:v1.5`)

| Model | Size | Compile path | Runtime | Status |
|---|---|---|---|---|
| YOLOv8n-320×320 INT8 | ~4 MB VMFB | TFLite → `torq-compile :v1.5 --torq-hw=SL2610 --torq-disable-css` | `synap_cli` / SyNAP C++ API | **✅ Phase 1 closed** — `mean 71.40 ms` on NPU |
| YOLOv8s Body Pose / OD | TBD | Same pattern | Same | vendor-claimed; untested here |
| MobileNetV2 | TBD | Same pattern | Same | vendor-claimed |
| Moonshine Tiny (ONNX→VMFB) | TBD | `torq-export-model moonshine --convert-dtype bf16` | `torq.runtime` Python | **deferred** Phase 2+ (NPU path); Phase 1.5 uses onnxruntime instead |
| SmolLM2 135M/360M/1.7B | TBD | `torq-export-model smollm2 -s <size> --instruct-model` | `torq.runtime` Python | **deferred** to Linux server — needs ≥ 48 GiB for 360M |
| Gemma 3 270M-IT | ~540 MB VMFB (bf16) | Pre-built by Synaptics on HF (`Synaptics/gemma-3-270m-it`) | `torq.runtime` Python | Quality-gated out 2026-04-25; NPU throughput (1.7 tok/s) < A55 CPU (5.87 tok/s); archived at `docs/deferred/torq-gemma3-board-instructions.md` |

---

## 3. llama.cpp — A55 CPU inference (GGUF)

**Depth pointer**: `docs/references/llama-cpp.md` — the SL2619-specific orientation for the `references/llama.cpp/` submodule. Read it for the full file-path map, search command recipes, and submodule layout. This section captures only the conventions.

### 3.1 Stack overview

llama.cpp is the CPU-side LLM runtime on A55. Current use:

- **Inference binary**: `llama-completion` (headless one-shot) and `llama-cli` (interactive) — cross-compiled against the Yocto SDK, deployed to `/mnt/sdcard/llama-cpp/`.
- **Model format**: Q4_0 GGUF — 4-bit quantized weights, mmap-loaded from SD card. Hot path: NEON DOTPROD kernels for GEMV/GEMM.
- **Conversion tooling** (host or server): `convert_hf_to_gguf.py` (HF safetensors → BF16 GGUF) and `llama-quantize` (BF16 → Q4_0).

Current pin: `b8925` / commit `0adede8`. Update §1 and §3.6 if the tag advances.

### 3.2 Cross-compile against Yocto SDK

**Why cross-compile is mandatory**: prebuilt GitHub releases (`llama-b8925-bin-ubuntu-arm64.tar.gz`) link against `CXXABI_1.3.15` (GCC 14+); the board's stock Yocto image ships `libstdc++.so.6.0.32` exporting only `CXXABI_1.3.14`. Symptom on prebuilt: `./llama-cli: /usr/lib/libstdc++.so.6: version 'CXXABI_1.3.15' not found`.

**The working configure**:

```bash
source /opt/poky/5.0.9/environment-setup-cortexa55-poky-linux
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \      # MUST be ON — see §3.6 pitfall
  -DBUILD_SHARED_LIBS=OFF        # single binary; no SO deps beyond system libs
cmake --build build --target llama-cli llama-bench llama-completion -j$(nproc)
```

Configure correctly detects: `aarch64`, `DOTPROD` ✓, `FP16_VECTOR_ARITHMETIC` ✓, `FMA` ✓, **no SVE / no SME / no MATMUL_INT8**. This is correct — A55 is ARMv8.2-A with DOTPROD but not I8MM.

Strip binaries before deploy (`aarch64-poky-linux-strip llama-cli llama-completion llama-bench`). Expected stripped sizes: `llama-cli` 8.3 MB, `llama-completion` 6.6 MB, `llama-bench` 4.8 MB.

Full runbook: `docs/get-started/gemma-on-a55-get-started.md §3`.

### 3.3 REPACK: CPU architecture gates (A55 path)

REPACK = on-load weight interleaving into a CPU-arch-specific layout to feed a vectorized matmul kernel without per-call gather. The selection is made at runtime in `ggml/src/ggml-cpu/repack.cpp:ggml_repack_get_optimal_repack_type`:

| CPU feature gate | Q4_0 layout | Architecture |
|---|---|---|
| AVX2 _or_ (SVE + I8MM) | `q4_0_8x8_q8_0` | x86_64 server/host |
| NEON + I8MM | `q4_0_4x8_q8_0` | Cortex-A76+, Neoverse |
| **NEON + DOTPROD** (no I8MM) | **`q4_0_4x4_q8_0`** | **A55 — our path** |

**Consequence**: A55 and x86_64 select different weight layouts → different dot-product accumulation order → ≤ ~10% logit-rank divergence even with arithmetically-identical weights. **This is not a bug.** It is expected ISA-level FP behaviour. `--no-repack` (`-nr`) forces scalar Q4_0 at the cost of much lower throughput.

### 3.4 Logits-equivalence discipline

Every new GGUF (base model or fine-tune) must pass a logits-equivalence gate **before** quality evaluation. This prevents blaming prompt-engineering for what is actually arithmetic corruption.

**Gate H5R — same-quant cross-arch Δ** (calibrated 2026-04-27, supersedes obsolete absolute `same_top_p ≥ 99.99%` gate):

| Threshold | Value | Rationale |
|---|---|---|
| `Δ_same_top_p` (A55 Q4_0 vs x86 Q4_0, same `.kld` reference) | **≤ 1.0 pp** | Upstream's BF16-vs-FP16 Δ is ~0.25 pp; 1.0 pp allows 4× margin for quant + ISA differences. |
| `max_delta_p_a55 / max_delta_p_x86` | **≤ 3.0×** | Guards against tail-logit explosion. |

Do NOT use `same_top_p ≥ 99.99%` as an absolute threshold — Q4_0-vs-FP16 for any model on any architecture achieves only ~91–98% (see `references/llama.cpp/tools/perplexity/README.md` for upstream published norms). The Δ test subtracts out universal Q4_0 quantization noise.

**Three-step Q1 gate** (post-fine-tune logits-equivalence):

1. x86 BF16 ref → generate `.kld` baseline
2. x86 Q4_0 vs same `.kld` → establish noise floor (`same_top_p_x86`)
3. A55 Q4_0 vs same `.kld` → check `|same_top_p_a55 − same_top_p_x86| ≤ 1.0 pp`

This isolates: (i) Q4_0 quantization noise (steps 1→2), (ii) fine-tune delta (step 2 vs fine-tuned step 2), and (iii) A55-specific arithmetic divergence (steps 2→3). For the H5R result on the base model: `Δ = 0.393 pp` — well within gate. See `docs/tmp/bench/2026-04-27_h5r-cross-arch-delta.md`.

**Board OOM constraint**: `n_ctx ≥ 1024` OOM-kills `llama-perplexity` on the board (per-chunk buffer = `n_ctx × vocab × f32`; at `n_ctx=2048` → 2.15 GiB; board has 1.87 GiB). **Cap at `n_ctx=256` for on-board perplexity runs.** Keep `.kld` files on `/mnt/sdcard`, not tmpfs.

### 3.5 Deployment conventions

| Artifact | Path | Notes |
|---|---|---|
| Binaries | `/mnt/sdcard/llama-cpp/llama-completion`, `llama-cli`, `llama-bench` | cross-compiled, stripped |
| Base GGUF | `/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf` | 231 MB, sha256 `e479ea29…` |
| Fine-tuned GGUF | `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` | v1 SFT, Q4_0 |

**Invocation conventions**:

- Always `-t 2` (board exposes 2 cores; `-t 4` = 53× decode regression — measured).
- For headless one-shot: `llama-completion -no-cnv`.
- For fine-tuned model: MUST use `--jinja` (routes special tokens to correct IDs; without it, text-wrapping as plain bytes → hallucinated tail generation). Also use `--no-display-prompt` to get clean stdout.
- For user-turn content: render body locally via `prompt_composer.compose_user_text()`, pipe over SSH stdin with `printf '%s' "$BODY" | ssh ...`.

**Performance baseline (2026-04-28, fine-tuned Q4_0, `--jinja`)**:

| Metric | Value |
|---|---|
| Aggregate decode | 17.29 tok/s |
| Cold load (mmap + REPACK) | 3273 ms |
| Prompt-eval rate (~930 tok) | ~62 tok/s |
| Memory (process RSS) | ~1071 MiB (within IL-2) |

### 3.6 Known pitfalls

| Problem | Root cause | Fix |
|---|---|---|
| `CXXABI_1.3.15 not found` at first run | Prebuilt binary is GCC 14; board has GCC 13.3 | Cross-compile from source against Yocto SDK (§3.2) |
| `No rule to make target 'llama-cli'` | `LLAMA_BUILD_SERVER=OFF` removes `cli` subdirectory via CMakeLists coupling | Set `-DLLAMA_BUILD_SERVER=ON` |
| Decode drops from 5.87 to 0.11 tok/s | `-t 4` over-subscribes 2 available cores | Always `-t 2` |
| `--no-conversation is not supported` from `llama-cli` | `llama-cli` is interactive-only in `b8925` | Use `llama-completion` for headless runs |
| `-sysf sysprompt.txt --jinja` silently drops content | Gemma 3 chat template has no `system` role; content mis-routed | Compose user-turn body manually via `prompt_composer.compose_user_text()` |
| Fine-tuned model generates hallucinated tail HTML | Text-wrapped `<start_of_turn>` tokenized as plain bytes, not special token IDs | Use `--jinja` so the GGUF's embedded chat template handles wrapping |
| `requirements-convert_hf_to_gguf.txt` downgrades torch | File pins `torch~=2.6.0` — incompatible with cu128 torch 2.11 | Install `gguf` directly, skip the requirements file |

---

## 4. onnxruntime — A55 CPU inference (ONNX)

**Depth pointer**: `docs/references/onnx.md` — SL2619-specific orientation for the `references/onnx/` submodule. For NPU work, go to `references/Synaptics/torq-compiler/CLAUDE.md` first; Torq accepts ONNX as an input format but the ONNX submodule itself is rarely needed.

### 4.1 Role and current status

onnxruntime is used for **Moonshine Tiny STT on the A55 CPU** (Path 2, proven Phase A 2026-04-23). It is not on the critical path for the Gemma fine-tune (that uses llama.cpp GGUF, no ONNX intermediate).

| | Path 1 — Torq NPU VMFB | Path 2 — A55 CPU onnxruntime |
|---|---|---|
| Runtime | `torq.runtime` + IREE `torq` HAL backend | `onnxruntime` with `CPUExecutionProvider` |
| Model artifact | `.synap` bundle wrapping VMFB | `encoder_model.onnx` + `decoder_model_merged.onnx` |
| Compile step | Yes — host-side VMFB compile | No — use vendor-published ONNX |
| Expected throughput | TBD; likely 30-50 tok/s | **Measured 11.4 tok/s on 2× A55 @ 2 GHz** |
| Memory | Weights in CMA pool | Weights in process RSS (~108 MB for tiny-float) |
| Status | Deferred Phase 2+ | **CHOSEN + PROVEN 2026-04-23** |

### 4.2 Vendor HF ONNX variant selection rules

Some vendor-published HF repos ship ONNX variants with inconsistent quantization metadata. `UsefulSensors/moonshine` `onnx/merged/tiny/quantized/decoder_model_merged.onnx` has a `MatMulNBits` op referencing a missing scale tensor:

```
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
  qdq_actions.cc:136 TransposeDQWeightsForMatMulNBits
  Missing required scale: model.decoder.embed_tokens.weight_merged_0_scale
```

**General rules when picking an ONNX variant from a vendor HF repo**:

1. Prefer `float` (fp32) for first smoke — most likely to load cleanly.
2. Try `quantized_4bit` only after validating `MatMulNBits` metadata on host.
3. Always load-test on host with the **same onnxruntime version** the board will use (1.25.0 in our case) before scp. Don't discover QDQ mismatches from the board.

### 4.3 Moonshine Path 2 minimum deps

Install into `/tmp/p15site/` (see §5 for storage layout):

| Package | Version | Install flag | Why |
|---|---|---|---|
| torq_runtime | 1.5.0 (wheel) | direct URL install | Phase B needs this for Gemma 3; pre-staged during Phase A |
| onnxruntime | 1.25.0 | normal | Moonshine encoder + decoder |
| tokenizers | 0.22.2 | normal | Moonshine detokenization |
| soundfile | 0.13.1 | normal | WAV read; `wave` stdlib alternative works too |
| numpy | **1.26.4** (forced) | `--no-deps` + `rm -rf` then install | torq_runtime iree C-ext ABI (§2.3) |
| useful-moonshine-onnx | 20251121 | `--no-deps` | High-level API; wraps encoder+decoder loop |

The `useful-moonshine-onnx` `--no-deps` trick avoids installing `librosa`, `numba`, `llvmlite`, `scipy`, `scikit-learn` (~150 MB total). The `MoonshineOnnxModel(...).generate(audio_array)` call path is pure numpy + onnxruntime.

### 4.4 Smoke test recipe (reproducible)

```python
# moonshine_smoke.py — on /mnt/sdcard/scripts/
import numpy as np, soundfile as sf
import moonshine_onnx as mo
from tokenizers import Tokenizer

MODEL_DIR = "/mnt/sdcard/models/moonshine-tiny"
model = mo.MoonshineOnnxModel(models_dir=MODEL_DIR, model_name="moonshine/tiny", model_precision="float")
tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")

audio, sr = sf.read("/tmp/say_hi.wav", dtype="float32")
if audio.ndim == 2: audio = audio.mean(axis=1).astype(np.float32)
assert sr == 16000

out = model.generate(audio[None])   # pure numpy + onnxruntime; no librosa needed
text = tok.decode(out[0], skip_special_tokens=True)
print(text)
```

Measured on SL2619 (2026-04-23, 11-second JFK clip): **load 4.26s + generate 2.28s = 6.54s total; 11.4 tok/s; perfect transcript.**

---

## 5. On-board storage — tmpfs for Python env, SD card for models

**Design decision (split)**: use `PYTHONUSERBASE=/tmp/pipbase` + `pip install --target=/tmp/p15site` for the Python environment (tmpfs), and a dedicated **ext4-formatted microSD card mounted at `/mnt/sdcard`** for model artifacts, audio fixtures, and binaries.

| Path | Backing store | Typical contents | Persistence |
|---|---|---|---|
| `/tmp/pipbase/` | tmpfs | pip itself (bootstrapped by `get-pip.py`) | reboot-ephemeral |
| `/tmp/p15site/` | tmpfs → symlink to SD | pip-installed packages (torq_runtime, onnxruntime, numpy 1.26.4, tokenizers, soundfile, ml_dtypes, useful-moonshine-onnx) | reboot-ephemeral (symlink + SD survives) |
| `/mnt/sdcard/models/<model>/` | SD card (ext4) | VMFB + ONNX/GGUF weights + tokenizer + config per model | **survives reboot** |
| `/mnt/sdcard/llama-cpp/` | SD card (ext4) | llama-completion, llama-cli, llama-bench (cross-compiled) | **survives reboot** |
| `/mnt/sdcard/fixtures/` | SD card (ext4) | Reference WAVs, test audio, golden outputs | survives reboot |
| `/mnt/sdcard/scripts/` | SD card (ext4) | Reference smoke scripts | survives reboot |
| `/mnt/sdcard/p15site/`, `/mnt/sdcard/pipbase/`, `/mnt/sdcard/p15-env.sh` | SD card (ext4) | Python env mirrored from tmpfs (§5.4 symlink pattern) | **survives reboot** |

**Rule**: anything ≥ 10 MiB and binary belongs on `/mnt/sdcard`. Python packages symlink through to SD via §5.4. **Do not re-introduce `/tmp/p15models/`** — that path competes with process RSS for the same ~960 MiB tmpfs pool; a 516 MiB VMFB alone busts that ceiling.

### 5.1 Canonical env-source script

`/tmp/p15-env.sh` (scp'd once, then symlinked from SD via §5.4):

```bash
export PYTHONUSERBASE=/tmp/pipbase
export PATH="/tmp/pipbase/bin:$PATH"
export PYTHONPATH="/tmp/p15site:${PYTHONPATH:-}"
```

Every Python invocation on the board starts with `. /tmp/p15-env.sh && python3 ...`.

### 5.2 pip install flag pitfalls with `--target=`

| Goal | Correct flags | Common mistake |
|---|---|---|
| Install new package | `pip install --target=/tmp/p15site foo` | — |
| Replace existing package | `rm -rf /tmp/p15site/foo /tmp/p15site/foo.libs && pip install --target=/tmp/p15site foo==NEW` | `--force-reinstall` alone — silently skips writing because target dir exists |
| Avoid transitive deps (wheel-only) | `pip install --target=/tmp/p15site --no-deps foo` | — |
| Override a lying metadata dep | `pip install --target=/tmp/p15site --no-deps numpy==1.26.4` | Trying to satisfy wheel's declared constraint |

### 5.3 SD card mount + lifecycle (`/mnt/sdcard`)

The SD card is mounted by the user (R3), not by the agent. The mount is **not persistent across reboot** unless an `/etc/fstab` entry is added (eMMC write — deferred).

**Per-session recipe**:

```bash
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard && mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard && df -h /mnt/sdcard'
```

- Device path: `/dev/mmcblk2p1` on this board (`mmc2` is the MicroSD slot; `mmc1` is Wi-Fi SDIO).
- Filesystem MUST be ext4 — this kernel has no exfat support and `/lib/modules` is empty.
- Factory-formatted cards typically ship as exFAT; reformat on host (`mkfs.ext4 -L SL2619-models /dev/sdX1`) before first use.
- **Cold mmap latency**: SDR50 UHS-I caps at ~50 MB/s sequential. A 516 MiB VMFB cold-load takes ~10 s. One-time per session; subsequent reads stay in page cache.

For Phase 2+ systemd services: declare `RequiresMountsFor=/mnt/sdcard` in the `.service` unit.

### 5.4 Env-on-SD + `/tmp/` symlinks — reboot-survival pattern

After every reboot tmpfs is empty. Rebuilding the Python env from scratch costs ~20 min. The normative fix: migrate env to SD, symlink back into `/tmp/` after every boot.

**One-time migration** (paid once):

```bash
ssh nouslogic-sl2619 'cp -a /tmp/p15site /mnt/sdcard/ && cp -a /tmp/pipbase /mnt/sdcard/ && cp /tmp/p15-env.sh /mnt/sdcard/ && sync && du -sh /mnt/sdcard/p15site /mnt/sdcard/pipbase'
```

**Per-reboot recovery** (~10 s):

```bash
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard && mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard && ln -sfn /mnt/sdcard/p15site /tmp/p15site && ln -sfn /mnt/sdcard/pipbase /tmp/pipbase && ln -sfn /mnt/sdcard/p15-env.sh /tmp/p15-env.sh && . /tmp/p15-env.sh && python3 -c "import torq.runtime, onnxruntime; print(\"env OK\")"'
```

**Why symlinks**: `PYTHONPATH=/tmp/p15site` + `PYTHONUSERBASE=/tmp/pipbase` are referenced by downstream scripts. Python's import machinery follows symlinks invisibly. `pip install --target=/tmp/p15site foo` writes through the symlink to SD and persists.

**Caveat**: if `/mnt/sdcard` isn't mounted post-boot, symlinks become dangling → loud `ModuleNotFoundError`. Recovery: the one-liner above.

---

## 6. Model conversion and export conventions

### 6.1 HF safetensors → GGUF (llama.cpp path)

```
HF checkpoint (safetensors, BF16 or QAT) ─┐
  ↓  convert_hf_to_gguf.py --outtype bf16  │  host or server
  ↓  → model.bf16.gguf                     │
  ↓  llama-quantize model.bf16.gguf        │  host or server
     model.q4_0.gguf Q4_0                  │
  ↓  → model.q4_0.gguf                    ─┘
  ↓  Host smoke: llama-perplexity (logits-equivalence gate §3.4)
  ↓  scp to /mnt/sdcard/models/<model>/
  ↓  On-board verify: llama-completion smoke prompt
```

- **Do NOT pip-install `requirements-convert_hf_to_gguf.txt`** — it pins `torch~=2.6.0` which downgrades torch on cu128 stacks. Install `gguf` directly.
- **Fine-tune path (QLoRA)**: train on `google/gemma-3-270m-it` → merge adapter → `convert_hf_to_gguf.py` → `llama-quantize Q4_0` → H5R logits gate → Q4 quality bench.
- **Don't fine-tune the `-qat-q4_0-unquantized` Gemma 3 checkpoint** — Google's documented workflow uses plain `gemma-3-270m-it`; no recipe preserves QAT robustness through domain SFT.

### 6.2 HF/vendor ONNX → VMFB (Torq NPU path)

```
HF ONNX (or TFLite) ─┐
  ↓  torq-compile :v1.5 --torq-hw=SL2610 [--torq-disable-css]
  ↓  → model.vmfb
  ↓  bundle-vmfb → model.synap (if using SyNAP C++ API)
  ↓  Host smoke: torq.runtime Python (x86_64 wheel)
  ↓  scp to /mnt/sdcard/models/<model>/
  ↓  On-board verify: synap_cli OR torq.runtime smoke
```

- Compiler tag MUST be `:v1.5` for this board (IREE bytecode v15). See §2.2.
- `--torq-hw=SL2610` is the correct preset. Custom hw spec form rejected (see §2.7).
- `--torq-disable-css` if CSS stack overflow manifests (observed on YOLOv8n detection head).
- For HF→ONNX export needing Torq compile: use `torq-export-model ... --skip-iree` then manually invoke `torq-compile-model -t torq` (see §2.6.2 two-CLI pivot).

### 6.3 HF ONNX → onnxruntime (direct path)

For STT/encoder-decoder models where CPU path is sufficient (Moonshine Tiny):

```
HF ONNX export ─┐
  ↓  Prefer float/ variant over quantized/ (see §4.2 QDQ bug)
  ↓  Host smoke: import onnxruntime; load model; run dummy input
     (use SAME onnxruntime version as board: 1.25.0)
  ↓  scp to /mnt/sdcard/models/<model>/
  ↓  On-board: . /tmp/p15-env.sh && python3 /mnt/sdcard/scripts/smoke.py
```

### 6.4 General vendor HF repo rules

1. **Always load-test on host** before scp — catches QDQ/shape mismatches before the board round-trip.
2. **Use the same runtime version** on host validation as on the board (onnxruntime 1.25.0, torq_runtime 1.5.0).
3. **Pin SHAs before deploy** — `sha256sum` the artifact locally, record in the bench freeze.
4. **Check op coverage** before compiling for NPU — `torq-compiler/doc/user-manual/ops.md`.

---

## 7. Reproducibility and logging expectations

**What to pin and where to record it**:

| Artifact / decision | What to pin | Where |
|---|---|---|
| Compiler image | Docker digest (`sha256:...`) | Convention file (this doc, §2.2) or phase plan |
| On-board wheel | URL + version string | This doc §1 or phase plan |
| llama.cpp binary | git tag (`b8925`) + commit SHA (`0adede8`) | `docs/references/llama-cpp.md`, `gemma-on-a55-get-started.md §1` |
| GGUF artifact | `sha256sum` of `.gguf` | `gemma-on-a55-get-started.md §1` |
| Model artifacts (VMFB, ONNX) | `sha256sum` captured in `bundle_vmfb.py` or manually | per-model `models/<model>/README.md` |
| Bench sweep results | JSONL output + Markdown summary | `docs/tmp/bench/<date>_<model>-*.md` (frozen; never re-opened) |
| As-executed recipes | Step-by-step runbooks | `docs/get-started/<topic>-get-started.md` |
| Post-mortems / gotchas | Phase-specific log | `docs/plans/backlogs.md §1.x` |
| Phase-specific recipes | Docker tags, exact flags, gate results | `docs/plans/*-plan.md §P3/§T4/§T5/§T6` or `a55-gemma-fine-tune.md §10` |

**Bench summary naming convention**: `docs/tmp/bench/<YYYY-MM-DD>_<model-slug>[-<descriptor>].md`. Examples: `2026-04-24_gemma3-summary.md`, `2026-04-27_h5r-cross-arch-delta.md`, `2026-04-28_gemma3-finetuned-final.md`. Each file is a **frozen snapshot** — use the date prefix to track the sweep, and create a new file for each re-run.

**What is NOT required** for reproducibility: re-running the full bootstrap every time. The env-on-SD symlink pattern (§5.4) means `torq_runtime` + `onnxruntime` + numpy 1.26.4 persist across reboots. The llama.cpp binaries and GGUFs persist on the SD card. Only when the SD card or board image changes does a full re-bootstrap apply.

---

## 8. Debugging and failure triage

### 8.1 Torq failures

| Symptom | Cause | Fix |
|---|---|---|
| `runtime supports 15.0, module has 16.0` | Compiled with `:main` (bytecode v16); board has v15 | Rebuild with `:v1.5` |
| `Unable to find css config +m` | Custom hw spec form rejected | Use `--torq-hw=SL2610` preset |
| Cryptic `<null>` error during CSS tiling | CSS stack overflow on specific ops | Add `--torq-disable-css` |
| `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | `torq_runtime` C-ext is numpy 1.x ABI | `rm -rf /tmp/p15site/numpy && pip install --target=/tmp/p15site --no-deps numpy==1.26.4` |
| `failed to start network via IOCTL: Cannot allocate memory` (second process) | Driver-side descriptor exhaustion across processes | Reboot board; use single-process design (§2.5.2) |
| VMFB loads fine on host, crashes on board | aarch64 vs x86_64 runtime binary mismatch | Install aarch64 `torq_runtime` wheel on board, x86_64 on host |

### 8.2 llama.cpp failures

| Symptom | Cause | Fix |
|---|---|---|
| `CXXABI_1.3.15 not found` | Prebuilt binary; board has GCC 13.3 | Cross-compile from source against Yocto SDK (§3.2) |
| `No rule to make target 'llama-cli'` | `LLAMA_BUILD_SERVER=OFF` | Set `-DLLAMA_BUILD_SERVER=ON` |
| Decode 0.11 tok/s | `-t 4` with 2 cores exposed | Always `-t 2` |
| `--no-conversation is not supported` | `llama-cli` is interactive-only | Use `llama-completion -no-cnv` |
| `-sysf` content not seen by model | Gemma 3 has no `system` role in chat template | Compose user-turn body manually, skip `-sysf` |
| Fine-tuned model outputs `<h4>You can also try</h4>...` garbage | Text-wrapping tokenizes control markers as plain bytes | Use `--jinja` for fine-tuned GGUFs |
| `llama-perplexity` silently SIGKILL on board | `n_ctx ≥ 1024` → OOM; per-chunk buffer too large | Cap at `n_ctx=256` for on-board runs |
| `kl_divergence: failed to open FNAME` + no `.kld` written, exit 0 | `--save-all-logits` and `--kl-divergence-base` share the same FNAME slot; passing `--kl-divergence` alongside `--save-all-logits` flips the tool into LOAD mode and it tries to open the file you intended to write | SAVE: `--save-all-logits ref.kld` only (no `--kl-divergence`). LOAD: `--kl-divergence --kl-divergence-base ref.kld` only (no `--save-all-logits`) |

### 8.3 onnxruntime failures

| Symptom | Cause | Fix |
|---|---|---|
| `Missing required scale: ...weight_merged_0_scale` | `MatMulNBits` metadata gap in `quantized/` variant | Switch to `float/` variant (§4.2) |
| `ImportError: A module that was compiled using NumPy 1.x` | torq_runtime pulled numpy 2.x | Force `numpy==1.26.4` with `--no-deps` (§2.3) |
| `soundfile` missing or WAV read fails | `soundfile` not in `/tmp/p15site` | `pip install --target=/tmp/p15site soundfile==0.13.1` |

---

## 9. Checklist — adding a new model or upgrading the stack

**For any new model going to the NPU (Torq):**

- [ ] Compiler tag matches on-board IREE runtime (verify via `release_notes.md` + ASTRA SDK version)
- [ ] `torq_runtime` wheel arch + cp + manylinux tags match `uname -m` + Python ABI + glibc version on board
- [ ] Model op coverage validated against `torq-compiler/doc/user-manual/ops.md`
- [ ] For HF ONNX: load-test on host BEFORE scp (§6.4 rule 1)
- [ ] `--torq-disable-css` decision made per model (default: try without; add only if CSS stack-overflow)
- [ ] For autoregressive models: use `ManagedSelfAttnCacheRunner` from `torq-examples/utils/cache_runner.py`
- [ ] Single-process NPU discipline applied (§2.5.2)

**For any new model going to the A55 CPU via llama.cpp (GGUF):**

- [ ] Cross-compiled against Yocto SDK (not prebuilt binary)
- [ ] H5R logits-equivalence gate run (§3.4) before quality evaluation
- [ ] `-t 2` confirmed (not 4; check `cat /sys/devices/system/cpu/online`)
- [ ] `--jinja` flag in invocation if model is fine-tuned
- [ ] Binaries deployed to `/mnt/sdcard/llama-cpp/`, GGUF to `/mnt/sdcard/models/<model>/`

**For any new model going to the A55 CPU via onnxruntime (ONNX):**

- [ ] `float/` variant selected over `quantized/` for first smoke (§4.2)
- [ ] Load-tested with onnxruntime 1.25.0 on host before scp
- [ ] numpy pinned to 1.26.4 in `/tmp/p15site/` (§2.3)

**Cross-cutting:**

- [ ] `sha256sum` of all model artifacts recorded before scp
- [ ] On-board smoke test confirms runtime loads cleanly
- [ ] Bench summary frozen to `docs/tmp/bench/<date>_<model>-*.md`
- [ ] Per-model `models/<model>/README.md` updated with outcome
- [ ] Any new gotchas added to `docs/plans/backlogs.md §1.x`
- [ ] This file updated if a new runtime is introduced or an existing rule changes

---

## 10. When to update this document

Update this file when:

- A new model **compiler or runtime** is introduced to the workspace (e.g. TensorRT, OpenVINO, ExecuTorch, a second llama.cpp variant, a Torq major version bump).
- The **ASTRA SDK version** changes (triggers §2.2 compiler-tag coupling re-derivation).
- The **llama.cpp pin** advances and changes any flags, REPACK kernel selection, or perf numbers.
- The **onnxruntime version** on the board changes.
- A new **model storage path or naming convention** is adopted.
- A new cross-cutting **pitfall or workaround** is discovered that applies to more than one model.
- A **gate threshold** (e.g. H5R Δ ≤ 1.0 pp) is recalibrated.

Do NOT update this file for:
- Phase-specific recipe details (those go in `phase<N>-plan.md`).
- Per-model analysis (goes in `models/<model>/README.md`).
- Post-mortems (goes in `backlogs.md §1.x`).
- As-executed step-by-step runbooks (go in `docs/get-started/`).

Canonical ownership rule: each fact lives in ONE file; others use pointers. See `13-documentation-update-protocol.md §10`.

---

## 11. What this file does NOT cover

- Servo-domain code — see `05-servo-protocol.md`
- IPC wire format — see `04-ipc-rpmsg.md`
- M52 firmware — see `03-m52-baremetal.md`
- SLM prompt style and template rules — see `16-slm-system-prompt.md`
- Phase-specific recipes (exact Docker tag for a given phase) — see that phase's `phase<N>-plan.md` or `docs/get-started/<topic>-get-started.md`
- Fine-tune training loop, hyperparameters, dataset construction — see `docs/plans/AI-models/a55-gemma-fine-tune.md`
- Per-submodule layout and search recipes — see `docs/references/llama-cpp.md` and `docs/references/onnx.md`
