---
_generated_at: 2026-04-30T14:38:00-07:00
_source: /board_probe --target=server (READ-ONLY SSH to nouslogic-server)
_freshness_window: 24h
_live_verified: true
_purpose: FunctionGemma M5 server LoRA SFT pre-flight per docs/plans/FunctionGemma/README.md §10.1, §12.3
---

# Fine-Tune Server Live Snapshot (nouslogic-server)

## Snapshot summary

GPU is healthy and ready (RTX 5080 / 14.78 GiB free / sm_120 / driver 580.126.09 / CUDA 13.0
runtime, 12.8 wheel). Disk has 409 GiB free at `$HOME` — well above the 50 GiB §12.3 floor.
The shared venv at `~/sl2619-finetune/.venv` exists and `torch.cuda.is_available()` returns
True. `~/llama.cpp/build/bin/llama-quantize` is built and runnable. **However, M5 cannot
proceed yet:** the §10.1 step-1 pin file `.torch-pin-pre-fg-2026-04-29.txt` is **ABSENT**,
and the live venv pins (`transformers 5.6.2`, `trl 1.3.0`, `huggingface_hub 1.12.0`) are
**major versions ahead** of the §10.1 Unsloth pins (`transformers==4.56.2`, `trl==0.22.2`),
which means a naive `pip install unsloth && pip install transformers==4.56.2` will be a
much larger downgrade than §10.1 anticipates and is likely to break the existing
Gemma 3 SFT path. `unsloth`/`unsloth_zoo`/`xformers` are not installed. `~/functiongemma-finetune/`
does not exist. `hf` CLI is not installed (gated checkpoint pull will fail). User must run
the §10.1 step-1 capture command and then re-evaluate the pin diff before any install.

## Gate checklist

- [x] **G1 — `nvidia-smi` / RTX 5080 / ≥ 14 GiB free VRAM** — **PASS**
  - Name: NVIDIA GeForce RTX 5080, driver 580.126.09, sm_120, 16303 MiB total / 15405 MiB free / 436 MiB used.
  - 14.78 GiB reported free via `torch.cuda.mem_get_info(0)` (above 14 GiB floor).
- [x] **G2 — Disk free at `~hoanglan` ≥ 50 GiB** — **PASS**
  - `/dev/sda2` 786 GiB total, 409 GiB free at `$HOME` (46 % used).
- [x] **G3 — `~/sl2619-finetune/.venv` + `torch.cuda.is_available()`** — **PASS**
  - Python 3.12.3, torch 2.11.0+cu128, CUDA 12.8 wheel, `cuda_available True`, device 0 = RTX 5080, capability `(12, 0)`.
- [x] **G4 — `~/llama.cpp/build/bin/llama-quantize` runs** — **PASS**
  - HEAD `b1a5bd4e0c19ba8e82eea716a8362c30918b9560` "CUDA: better coalesce data-access for contiguous concat (#22330) (5 days ago)".
  - Binary 102168 bytes, executable, prints usage. `convert_hf_to_gguf.py` present.
  - Note: `--version` flag is not implemented in this build; usage banner is what `--help` returns. This is normal for current llama.cpp.
- [ ] **G5 — Pre-FG pin file `~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt`** — **FAIL (ABSENT)**
  - No `.torch-pin-*` files at all in `~/sl2619-finetune/`. Only `.torch-pin.txt` exists (older, single line, 71 bytes, dated 2026-04-27).
  - **§10.1 step 1 mandates capturing this BEFORE any `pip install`.** Without it the Gemma 3 path has no rollback baseline.
- [ ] **G6 — `unsloth` already installed?** — **NOT INSTALLED** (expected for fresh M5)
  - `pip show unsloth` → "Package(s) not found". `unsloth_zoo`, `xformers` also absent. `triton 3.6.0` is present.
- [x] **G7 — `~/functiongemma-finetune/` already exists?** — **ABSENT** (expected; §10.4 designates this as a NEW tree)
  - `ls /home/hoanglan/functiongemma-finetune` → "No such file or directory". Will be created by user during §12.4 data transfer.
- [x] **G8 — Architectural snapshot** — **CAPTURED**
  - `uname -a`: `Linux ubuntu-Standard-PC-Q35-ICH9-2009 6.17.0-22-generic #22~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Mar 26 15:25:54 UTC x86_64 GNU/Linux`
  - OS: Ubuntu 24.04.4 LTS (Noble Numbat), KVM/Q35 guest.
  - CPU: AMD Ryzen 7 9800X3D 8-Core / 8 threads. RAM: 49 GiB total, 44 GiB available.
  - `nvcc`: **NOT on PATH** (CUDA toolkit not installed system-wide; PyTorch wheels bundle their own runtime — fine for SFT, would matter only if the user wanted to compile custom CUDA kernels).
- [ ] **G9 — Versions of `transformers`, `trl`, `peft`, `bitsandbytes`, `accelerate`** — **CAPTURED, MAJOR DELTA vs §10.1**
  - See §Versions below. Live versions are major-version ahead of §10.1's Unsloth pins (`transformers==4.56.2`, `trl==0.22.2`) — that pin guidance was written against an older snapshot. **This is the largest M5 risk and must be reconciled before any install.**

## Versions

| Package | Live (`pip show`) | §10.1 Unsloth pin | Delta |
|---|---|---|---|
| torch | `2.11.0+cu128` | (no pin; current 2.11 is fine for sm_120) | OK |
| transformers | `5.6.2` | `==4.56.2` | **major downgrade** required by §10.1 |
| trl | `1.3.0` | `==0.22.2` (`--no-deps`) | **major downgrade** required by §10.1 |
| peft | `0.19.1` | (no pin) | OK |
| bitsandbytes | `0.49.2` | (no pin) | OK |
| accelerate | `1.13.0` | (no pin) | OK |
| datasets | `4.8.4` | (no pin) | OK |
| sentencepiece | `0.2.1` | (no pin) | OK |
| huggingface_hub | `1.12.0` | (no pin) | OK (but note `1.x` is post-`hf-cli` rename) |
| triton | `3.6.0` | (Unsloth dep) | OK |
| unsloth | (not installed) | install `unsloth` (latest) | **install pending** |
| unsloth_zoo | (not installed) | (Unsloth dep) | **install pending** |
| xformers | (not installed) | (Unsloth dep) | **install pending** |

## Disk + VRAM

```
$ nvidia-smi --query-gpu=memory.free,memory.total --format=csv
memory.free [MiB], memory.total [MiB]
15405 MiB,         16303 MiB

$ df -h ~
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda2       786G  337G  409G  46% /
```

VRAM headroom 14.78 GiB free (14.10 GiB free per `nvidia-smi`, 14.78 GiB free per
`torch.cuda.mem_get_info` — the latter is the more accurate post-init number).
RTX 5080 is currently servicing 4 MiB to Xorg and 412 MiB to a long-running ffmpeg
PID 472669; that ffmpeg should be reviewed before SFT to ensure it does not contend
for VRAM mid-training.

## Existing trees

```
$ ls ~/sl2619-finetune
adapters_v1/
bootstrap-20260426-153052.log .. bootstrap-20260427-121952.log   (8 logs)
checkpoints/
data/
finetune.py                                  (12852 bytes, 2026-04-28 06:47)
h5r/
logs/
merged_v1/
merged_v1.bf16.gguf                          (542 MiB)
merged_v1.q4_0.gguf                          (241 MiB)
merge.py                                     (3527 bytes)
runs/
t5_smoke_bundle.json
t5_smoke.py
.torch-pin.txt                               (71 B, 2026-04-27 — older, NOT the FG pin)
.venv/                                       (Python 3.12.3)

$ ls ~/functiongemma-finetune
absent
```

The Gemma 3 SFT path is healthy: `merged_v1.bf16.gguf` and `merged_v1.q4_0.gguf` exist
from the 2026-04-28 run, so the rollback target if Unsloth install breaks the venv is
well-defined.

## Pre-FG pin file

**Status: ABSENT.** The required §10.1 step-1 file is missing. The exact command the
user must run BEFORE any `pip install unsloth` is:

```bash
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip freeze > ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt &&
  ls -la ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt &&
  wc -l ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt
'
```

(The agent does NOT execute this — it is state-changing on the server and per R3 must
be user-invoked.)

## Blocking issues for M5

1. **Pre-FG pin file absent (G5).** Mitigations §10.1 require this file as the rollback
   baseline. Without it, an Unsloth install that breaks the Gemma 3 path is
   unrecoverable except by full venv rebuild.
2. **`transformers` major-version delta (G9).** Live `transformers==5.6.2` vs §10.1's
   Unsloth pin `==4.56.2`. **§10.1 explicitly anticipates this downgrade** ("this
   `transformers` pin matters, mitigations below cover the rollback path") and ships
   the §10.1 step-4 rollback for exactly this case. The open question is the
   *magnitude*: a ~1-major-version downgrade was likely envisioned, but the live
   stack has drifted further than that, raising the probability that Option A
   (shared venv) breaks the Gemma 3 `merged_v1.bf16.gguf` reload path and
   `scripts/finetune.py:_to_prompt_completion` tokenizer behavior. Recommendation:
   either (a) accept the downgrade and verify the Gemma 3 smoke still passes
   immediately after step (3), or (b) skip directly to §10.1's "Option B"
   isolated-venv path (`~/functiongemma-finetune/.venv` via a forked
   `scripts/server-bootstrap_functiongemma.sh`) — strictly safer given the delta size.
3. **`trl` major-version delta (G9).** Live `trl==1.3.0` vs §10.1 pin `==0.22.2`.
   Same risk profile as `transformers`. Note `trl 1.x` renamed several SFTConfig
   fields (e.g., `dataset_text_field` placement); rolling back to `0.22.2` may be
   required for Unsloth's `train_on_responses_only` helper to function.
4. **`huggingface_hub` 1.x + `hf` CLI absent (minor).** §12.4 data transfer is fine
   without `hf` CLI (uses `scp`), but if any §10.2 model loader call in the Unsloth
   notebook hits a gated repo (e.g. `unsloth/functiongemma-270m-it` if it requires
   click-through), the user has no `hf auth login` path on the server. Install with
   `pip install -U huggingface_hub[cli]` inside the venv — this is already implied
   if Unsloth pulls a current `huggingface_hub`.
5. **Hostname mismatch (informational).** Live hostname is
   `ubuntu-Standard-PC-Q35-ICH9-2009`, not the `nouslogic-server` alias —
   confirms the host is a Q35/KVM VM. No action required, but worth noting in any
   provenance metadata stamped into a checkpoint.
6. **Background ffmpeg holding 412 MiB VRAM (informational).** PID 472669 — review
   whether this is a known transcode job; if not, ask the user to verify before
   launching SFT.

## Suggested next user-approved commands

These are the §10.1 mitigations 1–4 lifted verbatim from the plan, with one extra
diagnostic step (1b) inserted to surface the major-version delta before committing
to the install. **The agent does NOT run any of these — they are state-changing on
the server and must be user-invoked.**

```bash
# (1) Capture exact pin file FIRST — never run after the install.
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip freeze > ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt &&
  ls -la ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt
'

# (1b) NEW — diff live versions vs §10.1 pins BEFORE installing. If transformers
#      and trl are major versions ahead (they are: 5.6.2 vs 4.56.2; 1.3.0 vs 0.22.2),
#      stop and reconvene with the user on Option A (shared venv + downgrade) vs
#      Option B (isolated ~/functiongemma-finetune/.venv via a forked
#      scripts/server-bootstrap_functiongemma.sh).
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  python -c "import transformers, trl, peft, accelerate, bitsandbytes; \
    print(\"transformers\", transformers.__version__); \
    print(\"trl         \", trl.__version__); \
    print(\"peft        \", peft.__version__); \
    print(\"accelerate  \", accelerate.__version__); \
    print(\"bnb         \", bitsandbytes.__version__)"
'

# (2) Compare BEFORE installing — what is the delta vs Unsloth's pins? (per §10.1)
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  python -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"
'

# (3) Install Unsloth (matches notebook cell 4 — local non-Colab branch).
#     Run ONLY after (1b) confirms the major-version delta is acceptable, or after
#     the user has decided to switch to Option B.
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip install unsloth &&
  pip install transformers==4.56.2 &&
  pip install --no-deps trl==0.22.2 &&
  python -c "from unsloth import FastLanguageModel; print(\"unsloth OK\")"
'

# (4) Rollback procedure (per §10.1) — only invoked if (3) breaks the Gemma 3 SFT smoke.
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip install --force-reinstall -r ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt
'
```

## Probe metadata

- Probe target: `nouslogic-server` (alias) → `100.116.133.62` (`hoanglan@`).
- Key: `~/.ssh/nouslogic_server_ed25519` (loaded into ephemeral `ssh-agent`, torn down post-probe).
- Probe mode: ONE batched SSH session, READ-ONLY (R3 / §12.5). No file writes, no `pip install`,
  no `mkdir`, no `apt`, no `rm` on the server.
- Probe rc: `0`.
- All §SKILL.md §4b sections plus M5-specific items 5–9 (pin file, unsloth, FG tree,
  arch snapshot, version pins) captured in a single SSH heredoc.
