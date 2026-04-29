# llama.cpp + GGUF — Upstream Sources

llama.cpp is the on-device inference path for both x86 host validation and
the SL2619 A55 deployment. GGUF is the on-disk format we ship.

## Core sources

| Source | URL | Authoritative for |
|---|---|---|
| llama.cpp repo | <https://github.com/ggml-org/llama.cpp> | C/C++ runtime, `llama-cli`, `llama-perplexity`, `llama-quantize` |
| `convert_hf_to_gguf.py` | <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py> | HF → GGUF converter; only authoritative source for supported architectures |
| `convert_hf_to_gguf_update.py` | <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf_update.py> | tokenizer-pre-tokenizer registration (run after pulling new architectures) |
| GGUF spec | <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> | binary layout, metadata keys (`tokenizer.chat_template`, etc.) |
| Conversion tutorial (community) | <https://github.com/ggml-org/llama.cpp/discussions/7927> | step-by-step HF → GGUF (updated thread) |
| Unsloth — Gemma 3 GGUFs | <https://huggingface.co/unsloth/gemma-3-270m-it-GGUF> | reference quantized variants for parity checking |

## Verified recipes (this repo)

### HF merged BF16 → GGUF BF16 → Q4_0

```bash
# On the GPU server, after merge.py produced merged_v1/
cd ~/llama.cpp
python convert_hf_to_gguf.py ~/sl2619-finetune/merged_v1 \
    --outfile ~/sl2619-finetune/merged_v1.bf16.gguf --outtype bf16

build/bin/llama-quantize \
    ~/sl2619-finetune/merged_v1.bf16.gguf \
    ~/sl2619-finetune/merged_v1.q4_0.gguf Q4_0
```

### Inference with the trained chat template

llama.cpp's `--jinja` flag tells the runtime to apply the model's embedded
`chat_template.jinja` instead of treating prompt text as a flat byte stream.
**Required for Gemma 3** — without `--jinja`, the chat-template literals
are tokenized as plain bytes (5–10 sub-tokens each) instead of the special
control tokens (105 / 106) the FT'd model expects. See `bench_remote.py`
header docstring for the failure mode.

```bash
llama-cli --jinja --no-display-prompt -m merged_v1.q4_0.gguf \
    -p "what is my heart rate?" -no-cnv --single-turn
```

### Logits-equivalence (KL divergence between architectures)

Used to certify that the Q4_0 GGUF preserves token-rank vs the BF16 reference
across architectures (x86_64 host vs A55 board). See
`docs/plans/a55-gemma-h5-logits-equivalence.md` and
`src/gemma_tools/logits_equivalence.py`.

```bash
# Reference (BF16, x86 native)
llama-perplexity -m merged_v1.bf16.gguf -f corpus.txt \
    --kl-divergence -c 256 --seed 1

# Test (Q4_0, A55)
llama-perplexity -m merged_v1.q4_0.gguf -f corpus.txt \
    --kl-divergence --kl-divergence-base merged_v1.bf16.kld -c 256 --seed 1 -t 2
```

## Known issues to watch

- **`Can not map tensor 'lm_head.weight._scale'`** has surfaced on prior Gemma
  conversions. If hit, regenerate from a fresh HF merge (don't reuse a
  bnb-quantized checkpoint as the converter input).
- **Tokenizer pre-tokenizer hash mismatch** after pulling a new converter:
  rerun `convert_hf_to_gguf_update.py` against your HF token, then retry.
- **`--no-display-prompt`** is the divider-free way to capture model-only
  output; do not parse out `<start_of_turn>model\n` literals (they tokenize
  as control tokens with `--jinja`).

## Vendor / mirror policy

- **Link-only.** Versions move fast and we pin recipes by command-line
  invocation, not by vendoring the converter.
- The build is reproduced from source on each target (x86 host + A55 cross).
  Do not check in prebuilt binaries.
