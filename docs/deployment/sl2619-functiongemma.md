# FunctionGemma 270M GGUF on SL2619 — bench runbook

End-to-end recipe to copy `model.gguf` (the distil iteration-001 fused
GGUF) to the SL2619 board and measure tok/sec for the 7-tool patient-record
function-calling task. Assumes the Apr 24 Gemma 3 270M baseline runbook
(`docs/deployment/sl2619-board.md`) has already cross-compiled and staged
`/mnt/sdcard/llama-cpp/{llama-cli,llama-completion,llama-bench}` (build
`0adede8` / b8925, gcc 13.3 aarch64).

Verified host baseline (10-thread WSL2, llama-cpp-python, KV cache reset
per prompt, 7 prompts):

| Metric | Value |
|---|---|
| Decode tok/s (mean) | **52.8** |
| Prompt-eval tok/s (mean, post-warmup) | ~900 |
| Overall tok/s (mean) | 746.8 |
| Tool-routing accuracy | **7/7** correct (incl. `meds_at_8am` AM-clock case) |
| Per-prompt wall (cold KV, 7 prompts) | 2.0–3.6 s |

Expected board envelope based on the §5 Gemma 3 270M Q4_0 baseline:
**~5–7 tok/s decode**, **~30–60 tok/s prompt eval**, ~25–40 s per cold
prompt (1600 prompt tokens at ~37 tok/s ≈ 43 s prompt-eval; FP16 GGUF
will be slower than Q4_0). Quantization to Q4_0 is a separate task — out
of scope here.

---

## 1. Pre-flight (board-side, READ-ONLY)

The board snapshot at `docs/tmp/sl2619-status.md` is the truth source for
disk/RAM/toolchain availability. As of 2026-05-02 it reports:

- `/mnt/sdcard` mounted, **108 GiB free**
- `/mnt/sdcard/llama-cpp/llama-completion` present (6.82 MiB, build `0adede8`)
- 2 × Cortex-A55, NEON+dotprod, **no SVE**, **no `pip`/`uv`/transformers**
- 1.67 GiB MemAvailable, no swap

If `/mnt/sdcard/` is not mounted (post-reboot), re-run:

```bash
ssh nouslogic-sl2619 'mountpoint -q /mnt/sdcard && echo MOUNTED || mount /dev/mmcblk2p1 /mnt/sdcard'
```

## 2. Stage the GGUF on the board (user runs)

R3 forbids the agent from writing to the board. Run these manually from
the WSL host:

```bash
ssh nouslogic-sl2619 'mkdir -p /mnt/sdcard/models/functiongemma-270m'
scp /home/lanhp-wsl/nouslogic/gemma3-270M-finetune/model.gguf \
    nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m/model.gguf
ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/models/functiongemma-270m/model.gguf && sha256sum /mnt/sdcard/models/functiongemma-270m/model.gguf'
```

Expected: 518 MiB transferred in ~15–20 s on 5 GHz Wi-Fi at -48 dBm.

## 3. Smoke (board, single prompt, ~30 s)

Verify the existing `0adede8` build can load the FunctionGemma GGUF
metadata at all. If this fails with `unknown architecture` or
`missing key`, the on-board llama.cpp needs a refresh build on
`nouslogic-server` — flag and stop, do not work around.

From the WSL host (the agent **renders the prompt locally** with the HF
tokenizer because the board has no transformers; we do not pass `--jinja`
on the board because build `0adede8` does not propagate `tools=` to the
chat template — see `docs/plans/FunctionGemma/README.md` §15.6):

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv run python scripts/functiongemma_bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model  /mnt/sdcard/models/functiongemma-270m/model.gguf \
    --threads 2 \
    --limit 1 --warmup 0
```

Decision tree:

| Smoke result | Action |
|---|---|
| Output contains `<start_function_call>call:get_vitals{}<end_function_call>` and decode tok/s ≥ 1 | **Path A works.** Proceed to §4. |
| Loads but emits garbage tail / no parseable call | Special-token-as-bytes regression (sl2619-board.md §8.1 mode). Falls outside this task — flag and stop. |
| Load fails with `unknown model architecture` or similar | Board llama.cpp build is too old for FunctionGemma metadata. Cross-compile a newer `llama.cpp` tag on `nouslogic-server` against the Yocto SDK; see `docs/deployment/sl2619-board.md` §3.4 for the recipe. |

## 4. Full bench (board, 7 prompts, ~3–5 min)

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv run python scripts/functiongemma_bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model  /mnt/sdcard/models/functiongemma-270m/model.gguf \
    --threads 2 \
    --warmup 1
```

Output:
- `bench_results/functiongemma_remote_<timestamp>.jsonl` — one row per prompt
- console summary with mean decode/prompt/overall tok/s and tool-match count

## 5. Compare host vs board

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv run python scripts/functiongemma_bench.py --mode local --warmup 1
diff <(jq -c '{id:.prompt_id,call:.parsed_call}' bench_results/functiongemma_local_*.jsonl | sort) \
     <(jq -c '{id:.prompt_id,call:.parsed_call}' bench_results/functiongemma_remote_*.jsonl | sort)
```

Expected: parsed-call rows identical across modes (same model, deterministic
greedy decoding); throughput differs by ~10–20×.

## 6. Files

| File | Purpose |
|---|---|
| `model.gguf` (host) | Distil iteration-001 fused FunctionGemma 270M GGUF, ~518 MiB |
| `model/` (host) | HF tokenizer + `chat_template.jinja` for prompt rendering |
| `scripts/functiongemma_bench.py` | Two-mode bench driver (local + remote) |
| `bench_results/functiongemma_*.jsonl` | One JSONL row per bench prompt |
| `/mnt/sdcard/models/functiongemma-270m/model.gguf` (board) | Staged GGUF |
| `/mnt/sdcard/llama-cpp/llama-completion` (board) | Existing aarch64 llama.cpp binary |

## 7. Known gaps

- **No quantization step.** The host bench uses the FP16/BF16 GGUF
  (518 MiB). For deployment, the next iteration should quantize to Q4_0
  (~130 MiB) using the Yocto SDK's `llama-quantize` and re-run §4. The
  expected speedup follows the Gemma 3 baseline shape (Q4_0 is ~3× faster
  decode than F16 on A55 dotprod).
- **No NPU path.** The board snapshot shows `syna_npu` loaded but no
  llama.cpp Vulkan/SyNAP backend. CPU is the only path until a Torq-IREE
  build of FunctionGemma exists.
- **`--jinja` workaround is not free.** Pre-rendering on host means the
  board can't be used standalone (no tokenizer there). For an on-device
  deployment, fixing the FunctionGemma `--jinja` bug upstream — or
  shipping the HF tokenizer + a tiny Python renderer to the board via
  `p15-env.sh` — is the eventual fix.
