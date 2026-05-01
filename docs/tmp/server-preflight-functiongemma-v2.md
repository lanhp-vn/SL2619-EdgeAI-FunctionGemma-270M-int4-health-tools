---
_generated_at: 2026-05-01T18:42:23Z
_source: agent board_probe (READ-ONLY single-batched SSH to nouslogic-server)
_target: nouslogic-server (100.116.133.62, hoanglan@) — fine-tune host (NOT the SL2619 board)
_freshness_window: 24h
_live_verified: true
_purpose: Pre-flight for FunctionGemma 270M-IT Unsloth + LoRA r=128 SFT (3 epochs, LR 2e-4, eff-batch 8) on 881-train / 28-val, then merge to BF16 + eval on 45-row clean holdout.
_constraint: SSH read-only (CLAUDE.md §3 R3). No mutating commands issued.
---

# Fine-Tune Server Pre-flight — FunctionGemma v2 (Unsloth LoRA r=128)

## Summary (one-paragraph)

Host is healthy. RTX 5080 / 14.63 GiB free / driver 580.126.09 / CUDA 13.0 runtime. 354 GiB free at `$HOME` — well above the 20 GiB headroom floor. Venv at `~/functiongemma-finetune/.venv` is intact and **all required pins match the plan exactly** (`transformers 4.56.2`, `trl 0.22.2`, `peft 0.19.1`, `unsloth 2026.4.8`, `unsloth_zoo 2026.4.9`). `merge_v2.py`, `eval_verbose.py`, and `~/llama.cpp/build/bin/llama-quantize` are all present and dated. **However, two layout deltas vs the caller's brief must be resolved before the run:** (1) data is at `~/functiongemma-finetune/data/{train,val}.jsonl` (FLAT layout) — there is **no `data/functiongemma/dataset_v1/` directory**; (2) the live `train.jsonl` has **511 rows, not 881** — the new 881-row dataset has not yet been rsync'd. There is **no `outputs_fg_v2/` directory**, so the planned output path is collision-free, but five sibling `outputs_fg_v2_*` runs already exist (a1, a2, a2_3ep, b1, b3) and `b1`+`b3` are 16 GiB each — flag if you intended `_v2` to start from a clean slate.

## Gate checklist

- [x] **G1 — SSH connectivity** — PASS
  - `whoami` = `hoanglan`, `hostname` = `ubuntu-Standard-PC-Q35-ICH9-2009`
  - `uname -a`: `Linux ... 6.17.0-22-generic #22~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Mar 26 15:25:54 UTC 2 x86_64 GNU/Linux`
  - Server clock: `2026-05-01T18:42:23Z`

- [x] **G2 — GPU + driver** — PASS (well above the ~14 GiB free floor)
  - GPU: NVIDIA GeForce RTX 5080, sm_120 (Blackwell), driver `580.126.09`, CUDA 13.0 runtime
  - VRAM: 16303 MiB total / **14987 MiB free** / 854 MiB used
  - Live GPU consumers: `Xorg` (4 MiB), two `ffmpeg` processes (412 MiB each — appears to be a stuck/duplicate ffmpeg pair using ~824 MiB combined). **Non-blocking** for SFT, but worth noting that ~14.6 GiB will be the actual free budget at training start unless ffmpeg is killed by the user.
  - Temp 39C / Pwr 51W / Util 0% — idle.

- [x] **G3 — CPU + RAM** — PASS
  - CPU: AMD Ryzen 7 9800X3D, 8-core / 8-thread (Zen 5, AVX-512 + bf16 + VNNI), 9400 BogoMIPS, KVM/Q35 guest
  - RAM: 47 GiB total / **40 GiB available** / 6.1 GiB used / 8 GiB swap (324 KiB used)
  - Comfortably above the 16 GiB ideal floor.

- [x] **G4 — Disk headroom** — PASS (well above 20 GiB floor)
  - `/dev/sda2`: 786 GiB total / 392 GiB used / **354 GiB free** at `$HOME`, `/tmp`, `/var` (single root partition)
  - Existing FG tree: `~/functiongemma-finetune` total = **49 GiB** (dominated by `outputs_fg_v2_b1` 16G + `outputs_fg_v2_b3` 16G + `.venv` 7.9G + `outputs_fg_v2_a1` 3.7G + 5×merged dirs ~549M each + 3 GGUFs totalling ~1.04G).
  - New v2 run footprint estimate: ~250–500 MiB adapter dir + ~549 MiB merged BF16 dir + ~520 MiB BF16 GGUF + ~280 MiB Q8_0 + ~242 MiB Q4_K_M ≈ **~2 GiB**. Trivial vs 354 GiB free.

- [x] **G5 — Project layout** — PASS with deltas (see below)
  - `~/functiongemma-finetune/` exists, owner `hoanglan`, mtime `2026-05-01 23:44`
  - `.venv/bin/python` -> python3 (symlink, OK)
  - **`scripts/` directory does NOT exist** — caller assumed scripts/. All scripts live at the **top level** of `~/functiongemma-finetune/`. This contradicts the brief but matches the actual layout.
  - Confirmed top-level files (sizes, mtimes):
    - `merge_v2.py` (1939 B, 2026-05-01 21:47) — present
    - `eval_verbose.py` (1978 B, 2026-05-01 21:58) — present
    - `merge.py` (2974 B, 2026-05-01 21:45) — v1 merge helper, also present
    - `finetune_functiongemma.py` (18524 B, 2026-05-01 21:27) — v1 trainer
    - `finetune_functiongemma_v2.py` (22036 B, 2026-05-01 22:58) — **v2 trainer (newer; this is the script the planned run will use)**
    - `eval_functiongemma_holdout.py` (24300 B, 2026-05-01 22:42)
    - `merge_checkpoint.py` (1969 B, 2026-05-01 22:42)
    - `quantize.sh` (executable, 2026-05-01 21:47)
    - `gguf_roundtrip.py`, `smoke.py`, `diag_template.py`, `gemma_tools/`, `runs/`, `logs/`, `unsloth_compiled_cache/`
  - `~/llama.cpp/build/bin/llama-quantize` — present, 102168 B, executable, mtime `2026-04-27 12:20`. `libllama*`, `libggml*` shared objects present alongside.

- [x] **G6 — Venv import sanity** — PASS, all pins match the plan
  - python `3.12.3` (built 2026-03-23)
  - `torch 2.10.0+cu128` — caller said "torch 2.10+cu128"; matches
  - `torch.cuda.is_available()` = True; device 0 = RTX 5080, capability `(12, 0)` (sm_120)
  - `transformers 4.56.2` — **matches pin** `==4.56.2`
  - `trl 0.22.2` — **matches pin** `==0.22.2`
  - `peft 0.19.1`
  - `datasets 4.3.0`
  - `accelerate 1.13.0`
  - `bitsandbytes 0.49.2`
  - `unsloth 2026.4.8` — **matches plan** `~2026.4.x`
  - `unsloth_zoo 2026.4.9`
  - `xformers 0.0.35`
  - `triton 3.6.0`
  - **Soft warning (non-blocking):** Unsloth printed `Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128).` This disables the optional native C++ kernel fast-path; training still runs (Unsloth's Python+Triton path is the primary one) but will not get the full advertised speedup until torch is bumped. **Do NOT** bump torch right before this run — the cu128 wheel + sm_120 + bnb 0.49.2 + unsloth 2026.4.8 quartet is currently consistent. Treat the warning as a "known acceptable cost" for v2 and revisit post-run.
  - **Soft warning (non-blocking):** Unsloth nagged that it should be imported before `trl/transformers/peft`. The probe imported torch first; in `finetune_functiongemma_v2.py` make sure `import unsloth` is the first ML import (this is the standard Unsloth recipe).

- [ ] **G7 — Existing checkpoints / collision risk** — REVIEW
  - **No `outputs_fg_v2/` directory** — the exact path implied by the caller's "outputs_fg_v2" reference is currently absent. If the new run writes to literal `outputs_fg_v2/`, no collision.
  - However, the following sibling dirs DO exist and would be touched by any glob `outputs_fg_v2*`:
    - `outputs_fg_v2_a1/` (3.7 GiB, mtime 2026-05-01 22:49)
    - `outputs_fg_v2_a2/` (101 MiB, mtime 2026-05-01 23:11)
    - `outputs_fg_v2_a2_3ep/` (214 MiB, mtime 2026-05-01 23:43) — appears to be a 3-epoch v2 run that finished tonight; likely the most recent prior attempt
    - `outputs_fg_v2_b1/` (16 GiB, mtime 2026-05-01 23:03)
    - `outputs_fg_v2_b3/` (16 GiB, mtime 2026-05-01 23:10)
    - `outputs_fg_v1/` (1.3 GiB, mtime 2026-05-01 21:30) — the M5 baseline run
  - `outputs_fg_v2_a2_3ep` already contains `adapter_config.json`, `adapter_model.safetensors` (6.66 MB — consistent with LoRA r=128), `checkpoint-{64,128,192}/`, `tokenizer*`, `training_args.bin`. **This looks like an already-completed 3-epoch run.** Confirm with the user whether the planned v2 run is a re-run (overwrite expected), a parallel new variant (pick a fresh suffix like `outputs_fg_v2_c1/`), or actually targets the literal name `outputs_fg_v2/` (no collision).
  - Five `merged_fg_v*` dirs (~549 MiB each = ~2.7 GiB) and three GGUFs (`bf16.gguf` 518M, `q8_0.gguf` 279M, `q4_k_m.gguf` 242M) all from the v1 / v2_a2 merge passes are present. New merge will produce a sixth.

- [ ] **G8 — Data on server** — DELTA vs caller's path + row count
  - **Layout delta:** caller said `data/functiongemma/dataset_v1/` and `data/functiongemma/eval_holdout_v2_clean.jsonl`. The actual server layout is **FLAT under `data/`**:
    ```
    /home/hoanglan/functiongemma-finetune/data/
      build_clean_eval_holdout.py          5224 B
      eval_holdout_v1.jsonl                56 lines (192560 B, 0600 perms)
      eval_holdout_v2_clean.jsonl          45 lines (160945 B, 0644 perms)   <-- target eval set, MATCHES
      eval_holdout_v2_contaminated.jsonl   11 lines (40792 B,  0644 perms)
      test.jsonl                           56 lines (192560 B, 0600 perms)
      train.jsonl                          511 lines (1850181 B, 0600 perms) <-- ROW COUNT DELTA: caller said 881
      val.jsonl                            28 lines (95761 B,  0600 perms)   <-- matches caller's 28
    ```
  - **`eval_holdout_v2_clean.jsonl` IS present at the expected sibling path** (`data/eval_holdout_v2_clean.jsonl`, not `data/functiongemma/...`) and has **exactly 45 lines** as the caller stated. Trainer + eval invocation paths must use this flat path.
  - **Row count delta on `train.jsonl`:** server has 511 rows, caller said 881. The new dataset (with the block_e supplement repair, per the unstaged `data/functiongemma/dataset_v1/train.jsonl` and `data/functiongemma/llm_expanded_v1.jsonl` in this repo's `git status`) **has not yet been rsync'd**. Local repo path = `/home/lanhp-wsl/nouslogic/gemma3-270M-finetune/data/functiongemma/dataset_v1/train.jsonl`; verify line count locally before push, then rsync to overwrite `~/functiongemma-finetune/data/train.jsonl` on the server. The 28-row val matches, so val may not need re-uploading (verify content hash all the same).
  - **Permissions oddity (non-blocking):** the v1-era files (`train.jsonl`, `val.jsonl`, `test.jsonl`, `eval_holdout_v1.jsonl`) are mode `0600`, while the v2 files are mode `0644`. New rsync upload will pick up the local file mode.

- [x] **G9 — GPU contention** — REVIEW
  - Two `ffmpeg` processes are pinning 412 MiB each (PIDs 1003061, 1007787). Total ~824 MiB. Likely a stuck encode loop or duplicated transcode. Non-blocking for the SFT memory budget (still 14.6 GiB free) but the user may want to investigate / kill them outside the agent (R3 forbids the agent from doing so). Once cleared, full ~16 GiB minus typical Xorg overhead would be available.

## Versions table (live)

| Package | Live | Plan pin / target | Status |
|---|---|---|---|
| python | 3.12.3 | (any 3.12) | OK |
| torch | 2.10.0+cu128 | 2.10+cu128 | **MATCH** |
| transformers | 4.56.2 | ==4.56.2 | **MATCH** |
| trl | 0.22.2 | ==0.22.2 | **MATCH** |
| peft | 0.19.1 | (Unsloth-compatible) | OK |
| datasets | 4.3.0 | — | OK |
| accelerate | 1.13.0 | — | OK |
| bitsandbytes | 0.49.2 | (cu128 build) | OK |
| unsloth | 2026.4.8 | ~2026.4.x | **MATCH** |
| unsloth_zoo | 2026.4.9 | (paired) | OK |
| xformers | 0.0.35 | — | OK |
| triton | 3.6.0 | — | OK |
| nvidia-driver | 580.126.09 | ≥570 for sm_120 | OK |

## Disk breakdown (existing FG tree — 49 GiB total)

| Path | Size | Mtime |
|---|---:|---|
| `outputs_fg_v2_b1/` | 16 GiB | 2026-05-01 23:03 |
| `outputs_fg_v2_b3/` | 16 GiB | 2026-05-01 23:10 |
| `.venv/` | 7.9 GiB | 2026-05-01 17:18 |
| `outputs_fg_v2_a1/` | 3.7 GiB | 2026-05-01 22:49 |
| `outputs_fg_v1/` | 1.3 GiB | 2026-05-01 21:30 |
| `merged_fg_v1.bf16.gguf` | 518 MiB | 2026-05-01 21:48 |
| `merged_fg_v1*` (5 merged dirs) | ~2.7 GiB total | 2026-05-01 21:48–23:44 |
| `merged_fg_v1.q8_0.gguf` | 279 MiB | 2026-05-01 21:48 |
| `merged_fg_v1.q4_k_m.gguf` | 242 MiB | 2026-05-01 21:48 |
| `outputs_fg_v2_a2_3ep/` | 214 MiB | 2026-05-01 23:43 |
| `outputs_fg_v2_a2/` | 101 MiB | 2026-05-01 23:11 |
| `runs/` | 264 KiB | 2026-05-01 23:42 |
| `unsloth_compiled_cache/` | 3.6 MiB | 2026-05-01 21:08 |

## GPU live snapshot

```
NVIDIA-SMI 580.126.09          Driver 580.126.09          CUDA 13.0
GPU 0: NVIDIA GeForce RTX 5080  16303 MiB total / 14987 MiB free / 854 MiB used  39C  51W/360W  0% util
Compute apps:
  PID 1003061  /usr/lib/ffmpeg/7.0/bin/ffmpeg   412 MiB
  PID 1007787  /usr/lib/ffmpeg/7.0/bin/ffmpeg   412 MiB
  PID 4432     /usr/lib/xorg/Xorg               4   MiB
```

## Discrepancies vs caller's brief (must reconcile before launching)

1. **`scripts/` does not exist.** All trainer / merge / eval scripts live at the top level of `~/functiongemma-finetune/`. Update launch commands accordingly. The likely target file is `finetune_functiongemma_v2.py` (22 KB, mtime tonight). Several `launch_v2_*.sh` wrappers exist that probably already encode the right flags.
2. **`data/functiongemma/dataset_v1/` does not exist on the server.** Server layout is flat: `data/{train,val,test,eval_holdout_v2_clean,...}.jsonl`. Either (a) rsync the new 881-row train into `data/train.jsonl` (overwriting the 511-row file — back it up first if desired) and the val into `data/val.jsonl`, or (b) push the `data/functiongemma/dataset_v1/` tree wholesale and update trainer args to point at the nested path. Option (a) matches existing convention.
3. **`train.jsonl` row count: 511 (server) vs 881 (caller).** New dataset has NOT been uploaded yet. Verify local row count (`wc -l data/functiongemma/dataset_v1/train.jsonl`) matches 881 before rsync.
4. **No `outputs_fg_v2/` directory.** Five `outputs_fg_v2_*` siblings already exist; `outputs_fg_v2_a2_3ep/` looks like the most recent successful 3-epoch run from earlier tonight. Confirm with the user whether this v2 run targets a fresh literal `outputs_fg_v2/` (no collision), overwrites one of the existing variants (data loss risk), or needs a new suffix.

## GO / NO-GO verdict

**CONDITIONAL GO.** Hardware (GPU/CPU/RAM/disk), driver, venv pins, `merge_v2.py`, `eval_verbose.py`, and `llama-quantize` are all confirmed ready. **Two prerequisites the agent cannot perform under R3** must be completed by the user before launch:

1. **Push the new 881-row `train.jsonl` (and re-verify `val.jsonl`) to `~/functiongemma-finetune/data/`** via rsync. Source = local `data/functiongemma/dataset_v1/{train,val}.jsonl` (per `git status` these are modified locally).
2. **Decide the output directory name** (literal `outputs_fg_v2/` vs a new suffix like `outputs_fg_v2_c1/`) so the new run does not silently shadow / share with `outputs_fg_v2_a2_3ep` (which is a completed run worth preserving).

Optional but recommended (non-blocking):
- Kill the two stuck `ffmpeg` processes (PIDs 1003061, 1007787; 412 MiB each) to reclaim ~824 MiB VRAM headroom before launch.
- Note the Unsloth-cpp-extensions warning in the run journal — train will work without the C++ fast-path; do not bump torch for this run.
- Ensure `import unsloth` is the very first ML import in `finetune_functiongemma_v2.py` (per the Unsloth nag).

No blocking infrastructure issues. Once items 1 and 2 above are resolved, training + merge + eval can proceed.
