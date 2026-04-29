#!/usr/bin/env python3
"""
Merge LoRA adapters into the base model and save a standard HF checkpoint.

Run after finetune.py (T3) to produce the merged BF16 checkpoint for Q0 quantization.

Deploy to server:
    scp tools/scripts/merge.py nouslogic-server:~/sl2619-finetune/

Usage:
    # Default: picks the last epoch checkpoint inside ./adapters_v1/
    python ~/sl2619-finetune/merge.py

    # Pick a specific checkpoint by val_loss (override --adapters):
    python ~/sl2619-finetune/merge.py --adapters ./adapters_v1/checkpoint-768

    # Custom output dir:
    python ~/sl2619-finetune/merge.py --output ./merged_v1

After this script:
    Q0 — convert BF16 → GGUF on the server:
        python ~/llama.cpp/convert_hf_to_gguf.py ./merged_v1 --outfile ./merged_v1.bf16.gguf
        ~/llama.cpp/build/bin/llama-quantize ./merged_v1.bf16.gguf ./merged_v1.q4_0.gguf Q4_0
"""
import argparse
import glob
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def _resolve_adapter_path(adapter_dir: str) -> str:
    """Return the highest-step checkpoint subdir, or the dir itself if no checkpoints."""
    ckpts = glob.glob(os.path.join(adapter_dir, "checkpoint-*"))
    if not ckpts:
        return adapter_dir
    # TRL saves checkpoint-N at each epoch (save_strategy="epoch"); highest N = last epoch.
    # Sort by NUMERIC step (not lexicographic — "64" sorts after "192" as strings).
    # Pass --adapters ./adapters_v1/checkpoint-N explicitly to pick a specific epoch.
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("-", 1)[1]))
    chosen = ckpts[-1]
    print(f"Auto-selected checkpoint: {chosen}")
    print("(To use a different epoch, re-run with --adapters ./adapters_v1/checkpoint-N)")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description="Merge LoRA adapters into base Gemma 3 checkpoint")
    p.add_argument("--adapters", default=os.path.join(_WORKSPACE, "adapters_v1"),
                   help="Adapter dir or specific checkpoint-N subdir")
    p.add_argument("--output",   default=os.path.join(_WORKSPACE, "merged_v1"),
                   help="Output dir for merged HF checkpoint")
    args = p.parse_args()

    adapter_path = _resolve_adapter_path(args.adapters)

    print(f"Base model : {MODEL_ID}")
    print(f"Adapters   : {adapter_path}")
    print(f"Output     : {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load base in BF16 (no 4-bit quant for merge — QLoRA merge requires full weights)
    print("Loading base model in BF16 …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading LoRA adapters …")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging and unloading adapters …")
    model = model.merge_and_unload()

    print(f"Saving merged checkpoint to {args.output} …")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    print("\nMerge complete.")
    print(f"  merged dir  : {args.output}")
    print(f"  next (Q0)   : python ~/llama.cpp/convert_hf_to_gguf.py {args.output} --outfile merged_v1.bf16.gguf")
    print("                ~/llama.cpp/build/bin/llama-quantize merged_v1.bf16.gguf merged_v1.q4_0.gguf Q4_0")


if __name__ == "__main__":
    main()
