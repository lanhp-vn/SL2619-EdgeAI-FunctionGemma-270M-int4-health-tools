---
name: board_probe
description: READ-ONLY SSH probe of a trusted host (SL2619 board OR fine-tune server) and (re)generate the matching docs/tmp snapshot. Batches all diagnostic queries into a single SSH session for speed, parses the output into a structured Markdown snapshot, and flags any discrepancy vs Iron Laws or expected workspace state. Invoked as pre-flight before any non-trivial task per Requirement R1. Never writes to the target — SSH is strictly read-only per R3, on BOTH targets.
---

# board_probe

READ-ONLY SSH probe. Two targets, same R3 semantics:

| `--target` | Alias (in `~/.ssh/config`) | Output snapshot | Used by |
|---|---|---|---|
| `sl2619` (default) | `nouslogic-sl2619` | `docs/tmp/sl2619-status.md` | A55/M52/IPC dev, deploy verification |
| `server` | `nouslogic-server` | `docs/tmp/nouslogic-server-status.md` | Phase 0+ Gemma fine-tune planning (`docs/plans/AI-models/a55-gemma-fine-tune.md`) |

Every other skill in this repo calls this one first (or expects its output file to be fresh). If SSH is unreachable, writes a stub with `_live_verified: false` — never invents host state.

> **Inherits `CLAUDE.md` §3 Operational Discipline**: enforces R1 (pre-flight), R3 (SSH read-only) on **both targets**, R6 (ground-truth hierarchy). For the SL2619 target, never violates IL-6 (mailbox address), IL-7 (vring geometry), IL-8 (no flash), IL-11 (no secure-world poking).

## Invocation

```
/board_probe                        # default: --target=sl2619, write docs/tmp/sl2619-status.md
/board_probe --target=server        # probe fine-tune server, write docs/tmp/nouslogic-server-status.md
/board_probe --stale-max=1h         # skip re-probe if file is newer than N (works for either target)
/board_probe --inline               # return snapshot inline; do NOT write to disk
```

## Procedure

### 1. Resolve target and freshness

Parse `--target` (default `sl2619`). Map to:

| `--target` | Host alias | Key | Output file |
|---|---|---|---|
| `sl2619` | `nouslogic-sl2619` | `~/.ssh/sl2619_nouslogic_wsl` | `docs/tmp/sl2619-status.md` |
| `server` | `nouslogic-server` | `~/.ssh/nouslogic_server_ed25519` | `docs/tmp/nouslogic-server-status.md` |

If the output file exists and its `_generated_at:` front-matter is within the requested `--stale-max` window, print the timestamp and skip the SSH probe. Otherwise proceed.

### 2. Read SSH credentials from `.claude/CLAUDE.local.md`

```
Read(.claude/CLAUDE.local.md)  # extract host aliases (§1a, §1b) + passphrases
```

Do not hard-code credentials in any emitted tool call. If the file is missing, stop and ask the user to create it per the template in that file's git history.

### 3. Start ephemeral ssh-agent + load the key (one-shot)

Each Bash tool call is a fresh shell; persistent ssh-agent does not survive across calls. The skill starts an agent *inside* the same Bash call that runs the SSH probe.

**Use the canonical one-shot setup pattern in `.claude/CLAUDE.local.md` §2** — do NOT restate it here (IL-13). The pattern is parametric over `HOST_ALIAS` and `KEY`; substitute per the table above. The askpass tmp file is named `/tmp/askpass_${HOST_ALIAS}.sh` so concurrent probes against different targets don't collide.

### 4. Run ONE batched SSH call with delimited sections

Single session, all probes, separated by `echo "=== section ==="` so the parser can split deterministically. **All commands below are READ-ONLY** — no state change on the target.

#### 4a. SL2619 board probe (`--target=sl2619`)

```bash
ssh nouslogic-sl2619 '
    echo "=== ASTRA_VERSION ===";    cat /etc/astra_version
    echo "=== UNAME ===";             uname -a
    echo "=== HOSTNAME ===";          hostname
    echo "=== UPTIME ===";            uptime
    echo "=== MEMINFO ===";           head -n 25 /proc/meminfo
    echo "=== IOMEM ===";             grep -iE "reserved|vdev|vring|cma|synpu|ipc|f7e22|f7600000|mailbox" /proc/iomem
    echo "=== RESERVED_MEM ===";      ls /sys/firmware/devicetree/base/reserved-memory/ 2>&1
    echo "=== LSBLK ===";             lsblk
    echo "=== DF ===";                df -h / /home /tmp
    echo "=== MOUNTS ===";            mount | head -n 20
    echo "=== RPMSG_DEV ===";         ls /dev/rpmsg* 2>&1
    echo "=== RPMSG_CLASS ===";       ls /sys/class/rpmsg/ 2>&1
    echo "=== RPMSG_DEVICES ===";     ls /sys/bus/rpmsg/devices/ 2>&1
    echo "=== SYNPU ===";             ls /sys/devices/platform/soc/f7600000.synpu/ 2>&1
    echo "=== SERVICES_RUNNING ===";  systemctl list-units --type=service --state=running --no-pager
    echo "=== SERVICES_FAILED ===";   systemctl list-units --state=failed --no-pager
    echo "=== COORDINATOR ===";       systemctl status coordinator.service --no-pager 2>&1
    echo "=== CMDLINE ===";           cat /proc/cmdline
    echo "=== VIDEO ===";             ls /dev/video* 2>&1
    echo "=== AUDIO_CAPTURE ===";     arecord -l 2>&1 || echo "arecord unavailable"
    echo "=== AUDIO_PLAYBACK ===";    aplay   -l 2>&1 || echo "aplay   unavailable"
    echo "=== TORQ_LSMOD ===";        lsmod 2>&1 | grep -iE "torq|synap|syna_npu" || echo "no torq/synap modules"
    echo "=== TORQ_PACKAGES ===";     opkg list-installed 2>&1 | grep -iE "torq|synap|gstreamer1.0-plugins-syna" || echo "opkg unavailable or no matches"
    echo "=== GST_SYNAP ===";         gst-inspect-1.0 synapinfer 2>&1 | head -n 3 ; echo "---" ; gst-inspect-1.0 synavideoconvertscale 2>&1 | head -n 3 ; echo "---" ; gst-inspect-1.0 synapoverlay 2>&1 | head -n 3
    echo "=== SYNAP_MODELS ===";      ls /usr/share/synap/models/ 2>&1
    echo "=== SYNAP_CLI ===";         command -v synap_cli synap_cli_od torq_cli 2>&1 || echo "no synap_cli binaries in PATH"
    echo "=== WLAN ===";              ip addr show wlan0 2>&1 | head -n 10
    echo "=== ETH ===";               ip addr show eth0  2>&1 | head -n 5
    echo "=== ROUTE ===";             ip route
    echo "=== WIFI_LINK ===";         iw dev wlan0 link 2>&1 | head -n 10
    echo "=== CPUINFO ===";           head -n 45 /proc/cpuinfo
    echo "=== DMESG_HW ===";          dmesg 2>&1 | grep -iE "rpmsg|virtio|synpu|cma|reserved|mailbox|synaptics" | head -n 50
    echo "=== TEE ===";               systemctl status tee-supplicant@teepriv0.service --no-pager 2>&1 | head -n 10
'
```

#### 4b. Fine-tune server probe (`--target=server`)

Ubuntu, full GNU coreutils. No mailbox / RPMsg / NPU / CMA — those are SL2619 concepts. Instead probe what the Gemma fine-tune actually depends on (see `docs/plans/AI-models/a55-gemma-fine-tune.md` §3): GPU presence + compute capability, NVIDIA driver + CUDA, Python toolchain, project venv state, llama.cpp checkout state, HuggingFace credentials, disk + RAM headroom.

```bash
ssh nouslogic-server '
    echo "=== OS_RELEASE ===";        cat /etc/os-release
    echo "=== UNAME ===";              uname -a
    echo "=== HOSTNAME ===";           hostname
    echo "=== UPTIME ===";             uptime
    echo "=== MEMINFO ===";            head -n 5 /proc/meminfo
    echo "=== DISK_HOME ===";          df -h "$HOME" /tmp
    echo "=== CPUINFO ===";            grep -E "model name|cpu cores|siblings" /proc/cpuinfo | head -n 6
    echo "=== NVIDIA_SMI ===";         nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 || echo "nvidia-smi unavailable"
    echo "=== NVIDIA_SMI_FULL ===";    nvidia-smi 2>&1 | head -n 20 || true
    echo "=== NVCC ===";               command -v nvcc >/dev/null && nvcc --version 2>&1 || echo "nvcc not on PATH (CUDA toolkit may not be installed system-wide; PyTorch wheels bundle their own runtime)"
    echo "=== PYTHON ===";             command -v python3 && python3 --version 2>&1
    echo "=== PIP ===";                command -v pip3 && pip3 --version 2>&1 || echo "pip3 not on PATH"
    echo "=== SYSTEM_PY_DEPS ===";     dpkg -l 2>/dev/null | awk "/python3.*venv|python3-dev|build-essential|cmake|git|curl/ {print \$2,\$3}" | head -n 20
    echo "=== WORKSPACE ===";          ls -la "$HOME/sl2619-finetune" 2>&1 | head -n 20
    echo "=== VENV ===";               if [ -f "$HOME/sl2619-finetune/.venv/bin/python" ]; then "$HOME/sl2619-finetune/.venv/bin/python" --version 2>&1; "$HOME/sl2619-finetune/.venv/bin/pip" list 2>/dev/null | grep -iE "^(torch|transformers|trl|peft|bitsandbytes|accelerate|datasets|sentencepiece|huggingface)" || true; else echo "no venv yet"; fi
    echo "=== TORCH_CUDA ===";         if [ -f "$HOME/sl2619-finetune/.venv/bin/python" ]; then "$HOME/sl2619-finetune/.venv/bin/python" -c "import torch; print(\"torch\", torch.__version__); print(\"cuda_available\", torch.cuda.is_available()); print(\"cuda_version\", torch.version.cuda); print(\"device_count\", torch.cuda.device_count()); print(\"device_name\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"); print(\"capability\", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"\")" 2>&1 || echo "torch import failed"; else echo "no venv — skipping torch probe"; fi
    echo "=== LLAMA_CPP ===";          if [ -d "$HOME/llama.cpp/.git" ]; then git -C "$HOME/llama.cpp" rev-parse HEAD 2>&1; git -C "$HOME/llama.cpp" log -1 --format="%h %s (%ar)" 2>&1; ls "$HOME/llama.cpp/build/bin/llama-quantize" 2>&1; ls "$HOME/llama.cpp/convert_hf_to_gguf.py" 2>&1; else echo "no llama.cpp checkout"; fi
    echo "=== HF_CLI ===";             command -v hf >/dev/null && hf --version 2>&1 || echo "hf CLI not on PATH"
    echo "=== HF_AUTH ===";            command -v hf >/dev/null && hf auth whoami 2>&1 | head -n 5 || echo "hf CLI absent — skipping"
    echo "=== HF_LEGACY ===";          command -v huggingface-cli >/dev/null && huggingface-cli --version 2>&1 || echo "huggingface-cli not on PATH (legacy CLI)"
    echo "=== TENSORBOARD ===";        command -v tensorboard >/dev/null && tensorboard --version 2>&1 || echo "tensorboard not on PATH (may live inside venv)"
    echo "=== BOOTSTRAP_LOG ===";      ls -t "$HOME/sl2619-finetune"/bootstrap-*.log 2>/dev/null | head -n 3 || echo "no bootstrap log"
    echo "=== LAST_BOOTSTRAP_TAIL ===";LATEST=$(ls -t "$HOME/sl2619-finetune"/bootstrap-*.log 2>/dev/null | head -n 1); [ -n "$LATEST" ] && tail -n 25 "$LATEST" || echo "no bootstrap log to tail"
    echo "=== ROUTE ===";              ip route
    echo "=== HOMEDIR_TOP ===";        ls -la "$HOME" | head -n 25
'
```

### 5. Tear down the ephemeral agent

```bash
kill "$SSH_AGENT_PID" 2>/dev/null || true
rm -f "/tmp/askpass_${HOST_ALIAS}.sh"
```

### 6. Parse + write the snapshot

The output file has YAML front-matter + a numbered section per delimiter. Snapshot path depends on `--target`.

#### 6a. SL2619 snapshot — `docs/tmp/sl2619-status.md`

```markdown
---
_generated_at: 2026-04-26T16:32:00-07:00
_source: /board_probe --target=sl2619 (READ-ONLY SSH to nouslogic-sl2619)
_freshness_window: 24h
_live_verified: true
---

# SL2619 Live Board Snapshot

## 1. Identity
- Astra version: `scarthgap_6.12_v2.3.0`
- Kernel: `Linux 6.12.62 SMP PREEMPT …`
- Hostname: `nouslogic`
- Uptime: …

## 2. Memory (IL-2 verification)
- MemTotal: 1,962,988 kB (claimed: 1.87 GiB)
- SwapTotal: 0 kB (claimed: no swap)
- CMA: 512 MiB @ 0x5c000000–0x7bffffff

## 3. IPC (IL-6 / IL-7 verification)
- Mailbox: `f7e22000-f7e2205f : f7e22000.ipc`
- Vring0 (A55→M52): 0x0 (32 KiB)
- Vring1 (M52→A55): 0x8000 (32 KiB)
- vdevbuffer:       0x10000 (960 KiB)
- RPMsg: /dev/rpmsg_ctrl0, virtio0.rpmsg_ns.53.53

## 4. NPU / AI runtime (Torq + SyNAP — klamath)
…

## 5. Services / 6. Storage / 7. Network / 8. Video+audio / 9. BusyBox caveats / 10. dmesg / 11. Discrepancies
…
```

#### 6b. Server snapshot — `docs/tmp/nouslogic-server-status.md`

```markdown
---
_generated_at: 2026-04-26T16:32:00-07:00
_source: /board_probe --target=server (READ-ONLY SSH to nouslogic-server)
_freshness_window: 24h
_live_verified: true
---

# Fine-Tune Server Live Snapshot (nouslogic-server)

## 1. Identity
- OS: Ubuntu 24.04 LTS …
- Kernel: …
- Hostname: …
- Uptime: …

## 2. Hardware
- CPU: <model>, <cores> cores / <threads> threads
- RAM: MemTotal …, MemAvailable …
- Disk free at $HOME: …

## 3. GPU
- Name: NVIDIA GeForce RTX 5080
- Driver: 580.x
- VRAM: 16 GiB
- Compute capability: 12.0 (Blackwell sm_120)
- nvcc: <present | absent — PyTorch wheels bundle CUDA runtime>

## 4. Python toolchain
- python3: 3.12.x
- pip3: …
- System packages present: python3.12-venv, python3-dev, build-essential, cmake, git, curl

## 5. Project workspace (~/sl2619-finetune)
- Tree: data/, runs/, checkpoints/, .venv/
- venv Python: 3.12.x
- Key packages installed: torch X.Y.Z (cuda 12.8), transformers, trl, peft, bitsandbytes, accelerate, datasets, sentencepiece, huggingface_hub
- torch.cuda.is_available(): True
- device_name: NVIDIA GeForce RTX 5080
- compute capability: (12, 0)

## 6. llama.cpp (~/llama.cpp)
- HEAD: <sha> "<title>" (<date>)
- llama-quantize binary: present at build/bin/llama-quantize
- convert_hf_to_gguf.py: present

## 7. HuggingFace credentials
- hf CLI: <version | absent>
- hf auth whoami: <username | not logged in>

## 8. Last bootstrap run
- Log: bootstrap-YYYYMMDD-HHMMSS.log
- Tail (last 25 lines): RESULT: PASS (16/0) …

## 9. Discrepancies (⚠ if any)
(empty if all green — see §7 of this SKILL.md)
```

### 7. Flag discrepancies

#### 7a. SL2619 (Iron-Law violations)

| Symptom | Iron Law violated |
|---|---|
| Mailbox physical address ≠ `0xF7E22000` | IL-6 |
| CMA size ≠ 512 MiB | IL-2 |
| `SwapTotal > 0` | IL-2 |
| `/dev/rpmsg_ctrl0` missing | IL-7 (vring setup failed) |
| `/sys/devices/platform/soc/f7600000.synpu/` missing | NPU dead |
| `tee-supplicant@teepriv0` failed | IL-11 (TrustZone not alive) |

Do not proceed with downstream skills (`/a55_develop`, `/m52_develop`, `/ipc_develop`) until the user acknowledges the discrepancy. Serious discrepancies should trigger a bring-up PR, not a code PR.

#### 7b. Server (fine-tune readiness gates)

These are not Iron Laws — they are pre-conditions for the H2/H3/H6 gates in `docs/plans/AI-models/a55-gemma-fine-tune.md`. Flag a **⚠ NOT-READY** block at the top of the file if:

| Symptom | Affects |
|---|---|
| `nvidia-smi` absent or returns no GPU | All Phase 0+ training (H2 onward) |
| GPU compute capability ≠ `(12, 0)` for Blackwell | `--use-nightly-pytorch` may be required |
| NVIDIA driver < 555 (CUDA 12.5+) | sm_120 may not initialize on stable PyTorch |
| `~/sl2619-finetune/.venv` missing | H2 not yet run — point user at `tools/scripts/server-bootstrap.sh` |
| `torch.cuda.is_available()` False from venv Python | H2 smoke test would fail — investigate driver / wheel mismatch |
| `bitsandbytes` missing from venv | QLoRA blocked (BF16 LoRA fallback documented in fine-tune plan §3.2 troubleshooting) |
| `~/llama.cpp/build/bin/llama-quantize` missing | H3+ Q4_0 quantization blocked |
| `hf auth whoami` returns "not logged in" | Gated checkpoint pulls (Gemma 3) blocked — point user at `hf auth login` |
| Free disk at `$HOME` < 50 GiB | Checkpoint storage tight — fine-tune may evict mid-run |

Server discrepancies block Phase-0 gates only; they do not gate SL2619 board work.

### 8. SSH unreachable — graceful degrade

If the SSH call times out or returns auth failure (either target):

1. Write the appropriate snapshot file with `_live_verified: false` and a one-line reason.
2. Tell the user: "live verification skipped — reason: <SSH timeout | auth failure | …>. Downstream skills will run against the previous snapshot; flag that in any code review."
3. Do NOT retry indefinitely. One retry at most.

## What this skill does NOT do

- **Never writes to either target.** Every command in the batched SSH call is observation. Any `rm`/`mv`/`systemctl`/`astra-update`/`apt install`/`pip install`/`git push` via SSH is a bug — on **both** targets. R3 is host-agnostic.
- **Never flashes M52 firmware.** That is the user's job (IL-8).
- **Never edits `docs/conventions/`** based on probe output. Discrepancies go through normal PRs.
- **Never broadcasts passphrases** to chat, logs, or commit messages. They live only in `.claude/CLAUDE.local.md` (gitignored).
- **Never probes `/dev/mem` or secure-world memory** on the SL2619 — IL-11.
- **Never runs `sudo` over SSH on the server.** If a privileged step is needed, the agent emits the command for the user to run locally on the server (same shape as IL-8 for the board).

## Consistency

If `CLAUDE.md` §3 R1/R3/R6 change, this SKILL.md is reviewed in the same PR. If the SSH probe's delimiter format changes, `/doc_update` and the convention files that cite the snapshot files are reviewed together. If the fine-tune plan adds new ground-truth checks (e.g., a new dependency or directory layout), §4b and §7b are updated in the same PR.
