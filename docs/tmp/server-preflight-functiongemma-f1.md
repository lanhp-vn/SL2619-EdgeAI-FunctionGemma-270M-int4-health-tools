---
_generated_at: 2026-05-02T02:52:00+07:00
_source: agent board_probe (READ-ONLY single-batched SSH to nouslogic-server) + one follow-up batch
_target: nouslogic-server (100.116.133.62, hoanglan@) — fine-tune host (RTX 5080, sm_120)
_freshness_window: 24h
_live_verified: true
_purpose: Pre-flight for FunctionGemma F1-reweighting block — re-run the v3 LoRA recipe (Unsloth + LoRA r=128, 3 epochs, LR 2e-4, eff-batch 8) on the same 881-row train / 28-row val with NEW CLI flags (loss reweighting). Merge then eval on 45-row clean holdout + 56-row contaminated holdout.
_reference_snapshot: docs/tmp/server-preflight-functiongemma-v2.md (generated 2026-05-01T18:42, immediately before v3 ran)
_constraint: SSH read-only (CLAUDE.md §3 R3). No mutating commands issued on either probe pass.
---

# Fine-Tune Server Pre-flight — FunctionGemma F1 (post-v3 reweighting block)

## Hardware

| Resource | Live value | Floor (caller) | Status |
|---|---|---|---|
| GPU | NVIDIA GeForce RTX 5080 (sm_120, capability (12, 0)) | RTX 5080 expected | OK |
| Driver | 580.126.09 | ≥570 for sm_120 | OK |
| CUDA runtime | 13.0 (driver) / 12.8 (torch wheel) | cu128 | OK |
| VRAM total | 16,303 MiB | — | OK |
| VRAM free | **14,987 MiB (14.63 GiB)** | ≥ 12 GiB | **PASS** |
| VRAM used | 854 MiB (Xorg 4 MiB + 2× ffmpeg 412 MiB) | — | non-blocking — see Discrepancies §4 |
| GPU temp / pwr / util | 39 °C / 51 W / 0 % | idle | OK |
| CPU | AMD Ryzen 7 9800X3D, 8c/8t (Zen 5, AVX-512+bf16+VNNI) | — | OK |
| RAM total / available | 47 GiB / **41 GiB available** | ≥ 16 GiB | OK |
| Swap | 7 GiB total / 0 used | — | OK |
| Disk free at $HOME | **351 GiB free / 786 GiB total** (53 % used) | ≥ 30 GiB | **PASS** |
| Server clock | 2026-05-02T02:51:39+07:00 (uptime 1d 18h) | — | OK |
| Hostname | `ubuntu-Standard-PC-Q35-ICH9-2009` (KVM Q35 guest) | — | OK |
| Kernel | Linux 6.17.0-22-generic #22~24.04.1-Ubuntu | — | OK |

**Drift vs v3 snapshot:** GPU/driver/CPU identical. RAM available **+1 GiB** (40→41). Disk free **−3 GiB** (354→351 — explained by the v3 run artefacts: outputs_fg_v3 1.3 GiB + 3× merged_fg_v3_cp{111,222,333} 549 MiB each ≈ 2.95 GiB; train.jsonl grew from 511→881 rows ≈ +1.2 MiB). Same two ffmpeg PIDs still pinning 824 MiB combined — different PIDs (1035739, 1040462) than yesterday's (1003061, 1007787); the encoder loop has restarted but the pattern persists.

## Stack versions vs v3

All v3 pins are still in place — no drift since the v3 snapshot.

| Package | Live | v3 pin (caller) | v3 snapshot value | Status |
|---|---|---|---|---|
| python | 3.12.3 | (3.12.x) | 3.12.3 | MATCH |
| torch | **2.10.0+cu128** | 2.10.0+cu128 | 2.10.0+cu128 | **MATCH** |
| transformers | **4.56.2** | 4.56.2 | 4.56.2 | **MATCH** |
| trl | **0.22.2** | 0.22.2 | 0.22.2 | **MATCH** |
| peft | **0.19.1** | 0.19.1 | 0.19.1 | **MATCH** |
| unsloth | **2026.4.8** | 2026.4.8 | 2026.4.8 | **MATCH** |
| bitsandbytes | **0.49.2** | 0.49.2 | 0.49.2 | **MATCH** |
| accelerate | 1.13.0 | — | 1.13.0 | OK |
| datasets | 4.3.0 | — | 4.3.0 | OK |
| unsloth_zoo | 2026.4.9 | — | 2026.4.9 | OK |
| nvidia-driver | 580.126.09 | ≥570 | 580.126.09 | OK |

`torch.cuda` self-test (live): `is_available=True`, `cuda_version=12.8`, `device=NVIDIA GeForce RTX 5080`, `capability=(12, 0)`. **Zero version drift** vs the v3 snapshot — re-running on this stack reproduces the v3 build exactly.

> Carry-over caveat from v3 snapshot (still applies; non-blocking): Unsloth prints `Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128).` Optional native fast-path is disabled; Triton/Python path runs normally. Do not bump torch — the cu128 + bnb 0.49.2 + unsloth 2026.4.8 quartet is consistent and is what produced cp-333.

## File layout

Top-level (`~/functiongemma-finetune/`, owner `hoanglan`, mtime 2026-05-02 02:29):

| File / dir | Size | mtime | Notes |
|---|---:|---|---|
| `finetune_functiongemma_v2.py` | 22,036 B | 2026-05-01 22:58 | **Trainer used for v3 and the upcoming F1 run.** No `finetune_functiongemma_v3.py` exists — the "v3 recipe" is this script invoked with the v3 flag set. F1 will be the same script with new reweighting flags. |
| `finetune_functiongemma.py` | 18,524 B | 2026-05-01 21:27 | v1 trainer (kept for reference) |
| `merge_checkpoint.py` | 1,969 B | 2026-05-01 22:42 | **Used by `run_eval_v3.sh` to merge cp-111/222/333.** Required for F1 merge step. |
| `merge_v2.py` | 1,939 B | 2026-05-01 21:47 | older v2 merge helper (still present) |
| `merge.py` | 2,974 B | 2026-05-01 21:45 | v1 merge helper |
| `eval_functiongemma_holdout.py` | 24,300 B | 2026-05-01 22:42 | **Eval driver used by `run_eval_v3.sh`. Required for F1.** |
| `eval_verbose.py` | 1,978 B | 2026-05-01 21:58 | smoke eval helper |
| `quantize.sh` | 967 B (exec) | 2026-05-01 21:47 | calls llama-quantize for BF16/Q8/Q4 |
| `run_eval_v3.sh` | 1,266 B (exec) | 2026-05-02 01:58 | reference v3 eval pipeline — model the F1 eval after this |
| `run_eval_v3_rest.sh` | 1,582 B (exec) | 2026-05-02 02:07 | rerun for late checkpoints |
| `dump_failures_v3.py` | 4,448 B | 2026-05-02 02:29 | new since v3 — produced `eval_v3/cp333_clean_failures.md` |
| `outputs_fg_v3/` | **1.3 GiB** | 2026-05-02 01:56 | **PRESERVED.** Contains `adapter_config.json`, `adapter_model.safetensors` (122 MB — LoRA r=128 footprint), `checkpoint-{111,222,333}/`, `tokenizer*`, `chat_template.jinja`, `training_args.bin`, README. cp-333 is the v3 baseline. |
| `merged_fg_v3_cp{111,222,333}/` | 549 MiB each | 2026-05-02 01:58–02:09 | merged BF16 of each v3 checkpoint |
| `eval_v3/` | 48 KiB | 2026-05-02 02:30 | `cp{111,222,333}_clean.md`, `cp{111,222,333}_contam.md`, `cp333_clean_failures.md`, `run.log` |
| `.venv/` | 7.9 GiB | 2026-05-01 17:18 | unchanged since v2/v3 |

Sibling output dirs preserved (none touched):
- `outputs_fg_v1/` 1.3 GiB, `outputs_fg_v2_a1/` 3.7 GiB, `outputs_fg_v2_a2/` 101 MiB, `outputs_fg_v2_a2_3ep/` 214 MiB, `outputs_fg_v2_b1/` 16 GiB, `outputs_fg_v2_b3/` 16 GiB.
- Plus `merged_fg_v1*` (5 dirs ~549 MiB each + 3 GGUFs: bf16 518 MiB, q8_0 279 MiB, q4_k_m 242 MiB) and `merged_fg_v2_a2*` (2 dirs).

`~/llama.cpp/build/bin/llama-quantize`: present, 102,168 B, executable, mtime 2026-04-27 12:20 — unchanged since v3 snapshot.

`~/functiongemma-finetune/.git`: **absent**. The server tree is not a git checkout — files arrive via rsync from this host's `/home/lanhp-wsl/nouslogic/gemma3-270M-finetune/` workspace.

## Data — MD5 + row-count matrix

`~/functiongemma-finetune/data/` (mtime 2026-05-02 01:53):

| File | Rows | Bytes | MD5 (live) | Host expectation | Status |
|---|---:|---:|---|---|---|
| `train.jsonl` | **881** | 3,061,531 | **`ac0e261713ed8241044feaf618c538a2`** | `ac0e261713ed8241044feaf618c538a2` | **MATCH** |
| `val.jsonl` | 28 | 95,761 | `f5759aea8a992631807cb5bb1b10a60e` | (not provided) | OK — verify locally if needed |
| `test.jsonl` | 56 | 192,560 | `6722ab854bffd4931712c1d20a14e536` | (not provided) | OK — note hash equals `eval_holdout_v1.jsonl` (test == v1 holdout) |
| `eval_holdout_v1.jsonl` | 56 | 192,560 | `6722ab854bffd4931712c1d20a14e536` | (not provided) | OK |
| `eval_holdout_v2_clean.jsonl` | **45** | 160,945 | `4f5ab50d381b0f81be8d2cb8cbfc23bc` | (not provided) | OK |
| `eval_holdout_v2_contaminated.jsonl` | 11 | 40,792 | (not requested) | (not provided) | OK |

**Critical gate satisfied:** server `train.jsonl` md5 **matches the host-cited `ac0e261713ed8241044feaf618c538a2` byte-for-byte**. The 881-row dataset that v3 trained on is intact and is the same dataset the F1 reweighting block will train on.

**Drift vs v3 snapshot:** v3 snapshot (taken 2026-05-01 18:42, before the upload) had `train.jsonl` at **511 rows / 1,850,181 B** with no md5 captured; the new 881-row dataset was uploaded between then and the v3 run start (file mtime now `2026-05-02 01:39`). All other data files unchanged.

## Discrepancies

1. **No `finetune_functiongemma_v3.py` script exists.** The "v3 recipe" is `finetune_functiongemma_v2.py` invoked with v3-flavor flags. The F1 run will use the same trainer with new CLI flags — confirm the trainer already accepts the F1 reweighting flags (they may need a code add-on host-side and rsync). This matches v3's launch pattern but is a brief-vs-reality discrepancy worth surfacing.
2. **Two `ffmpeg` processes pinning 824 MiB VRAM** (PIDs 1035739, 1040462; 412 MiB each). Different PIDs from yesterday but same pattern — the encoder loop respawned. Non-blocking (14.63 GiB still free, well above the 12 GiB floor). User may want to investigate / kill these outside the agent (R3 forbids the agent from doing so).
3. **Unsloth cpp-extensions warning persists** (torch 2.10.0+cu128 vs Unsloth's recommended ≥ 2.11). Optional native fast-path disabled; Triton path is the supported primary. Do not bump torch — the v3 stack is the validated baseline.
4. **`~/functiongemma-finetune/` is not a git checkout.** No version control on the server tree; rely on host-side git + rsync for provenance. (Same as v3 snapshot.)
5. **`test.jsonl` and `eval_holdout_v1.jsonl` are byte-identical** (same MD5). Not new — was true at v3 time too. Worth keeping in mind for any "fresh test set" claim.
6. **No collision risk for the F1 output dir.** `outputs_fg_f1/` does not exist; pick that literal name (or whatever the F1 spec says) and it will not shadow `outputs_fg_v3/` or any v2 sibling.

No blocking discrepancies. None of the v3-snapshot blockers (missing scripts/, missing data dir, 511-row mismatch, no v2 outputs dir) survive — they were all resolved before the v3 run.

## Verdict — **GO**

All four caller-defined go/no-go gates pass:

| Gate | Threshold | Live | Verdict |
|---|---|---|---|
| Stack version drift | none vs v3 pins | torch 2.10.0+cu128, transformers 4.56.2, trl 0.22.2, peft 0.19.1, unsloth 2026.4.8, bitsandbytes 0.49.2 — all match | PASS |
| Free VRAM | ≥ 12 GiB | 14.63 GiB free | PASS |
| Free disk | ≥ 30 GiB | 351 GiB free | PASS |
| `train.jsonl` md5 == host (`ac0e261713ed…`) | byte-equal | byte-equal | PASS |
| Required helpers present | `finetune_functiongemma*.py` + `merge_checkpoint.py` + `eval_functiongemma_holdout.py` | all present | PASS |
| `outputs_fg_v3/` preserved | dir exists with cp-111/222/333 | 1.3 GiB, all 3 checkpoints intact | PASS |

**Proceed with the F1 reweighting LoRA run.** Use `finetune_functiongemma_v2.py` as the trainer entrypoint (same script that produced cp-333). Mirror `run_eval_v3.sh`'s post-train pipeline: `merge_checkpoint.py --adapter outputs_fg_f1/checkpoint-N --out merged_fg_f1_cpN`, then `eval_functiongemma_holdout.py` against `data/eval_holdout_v2_clean.jsonl` (45 rows, MD5 `4f5ab50d…`) and `data/eval_holdout_v1.jsonl` (56 rows, MD5 `6722ab85…`). Optional cleanup: kill the two ffmpeg PIDs to reclaim 824 MiB headroom.
