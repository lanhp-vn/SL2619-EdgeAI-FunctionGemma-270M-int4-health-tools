# SL2619 Storage Cleanup — FunctionGemma Deployment

After deploying the simplified FunctionGemma workflow, you may want to clean up stale files on the board to reclaim storage.

## Audit Current State

Run this on the board to see what's taking up space:

```bash
ssh nouslogic-sl2619 'bash -s' << 'EOF'
echo "=== /mnt/sdcard disk usage ==="
du -sh /mnt/sdcard/* 2>/dev/null | sort -rh

echo ""
echo "=== Gemma-related files ==="
find /mnt/sdcard -type f \( -name "*gemma*" -o -name "*model*.gguf" \) 2>/dev/null | xargs ls -lh

echo ""
echo "=== Old bench logs / transcripts ==="
find /mnt/sdcard -type f \( -name "*bench*" -o -name "*.jsonl" -o -name "*chat*.json" \) 2>/dev/null | xargs ls -lh 2>/dev/null || echo "(none found)"

echo ""
echo "=== llama-cpp binaries ==="
ls -lh /mnt/sdcard/llama-cpp/ 2>/dev/null || echo "(not found)"

echo ""
echo "=== Models directory ==="
ls -lh /mnt/sdcard/models/ 2>/dev/null || echo "(not found)"
EOF
```

## Safe Cleanup

These are known-safe to remove (do NOT delete without confirmation):

### Old quantized models (if present)
```bash
# Only if you're NOT using Q4_0 variants
ssh nouslogic-sl2619 'rm -fv /mnt/sdcard/models/gemma-3-270m-it-Q4_0.gguf'
```

### Old bench run logs
```bash
# Only if you're NOT actively analyzing bench results
ssh nouslogic-sl2619 'rm -fv /mnt/sdcard/*bench*.jsonl /mnt/sdcard/*bench*.json'
```

### Redundant fine-tune checkpoints
```bash
# Only if the merged_v1 model is the current production one
ssh nouslogic-sl2619 'rm -rfv /mnt/sdcard/models/merged_v1/'
```

### Temporary prompt files (safe, auto-cleaned)
The `/tmp/fg_prompt_*.txt` files created by `run-prompt.sh` are automatically cleaned up.

## What NOT to Delete

- `/mnt/sdcard/llama-cpp/` — llama.cpp binaries (required)
- `/mnt/sdcard/models/functiongemma-270m/` — new simplified deployment (required)
- `/mnt/sdcard/models/` if it contains your production GGUF (e.g., `merged_v1`)
- `/mnt/sdcard/torq-examples/` — NPU runtime examples (future work)
- Any `/sys` or `/proc` entries (those aren't real files)

## Expected Space After Cleanup

Rough breakdown on the SL2619:

| Item | Size | Purpose |
|---|---|---|
| llama-cpp binaries | 20 MiB | llama-completion + llama-cli + llama-bench |
| functiongemma-270m FP16 GGUF | 518 MiB | Production model |
| functiongemma-270m templates | 7.3 KiB | prompt-prefix/suffix (negligible) |
| torq-examples | ~500 MiB | NPU alternate path (optional) |
| **Total minimum** | **~545 MiB** | Just the simplified workflow |

With `/mnt/sdcard` at 119 GiB, you should have plenty of headroom even after deployment.

## Estimated Savings

- Removing one Q4_0 variant: ~130 MiB
- Removing bench logs: ~10–50 MiB (if they accumulate)
- Removing old checkpoints: ~200–500 MiB (context-dependent)

## Board Reboot (if needed)

If the board is behaving strangely after cleanup, reboot:

```bash
ssh nouslogic-sl2619 'sudo reboot'
# Wait ~30 s
ssh nouslogic-sl2619 'echo OK'
```
