# Upstream Reference Sources

Curated pointers to authoritative upstream material for everything this repo touches. Two layers:

1. **Notes** (this folder, `*.md`) — short, opinionated summaries of what to read upstream and when. Authoritative for this repo's *interpretation* of upstream behavior.
2. **Submodules** (`upstream/`) — verbatim source for the highest-value upstream repos. Pinned to a commit, opt-in init.

## Note files

| Topic | File | Authoritative for |
|---|---|---|
| Gemma 3 270M (base + IT) and FunctionGemma | [`gemma.md`](gemma.md) | model architecture, chat template, license, model cards |
| Hugging Face training stack (Transformers, TRL, PEFT, datasets) | [`transformers-trl-peft.md`](transformers-trl-peft.md) | SFT API surface, `SFTConfig` flags, LoRA target modules |
| llama.cpp + GGUF | [`llama-cpp.md`](llama-cpp.md) | GGUF conversion, quantization recipes, `--jinja` chat template handling |
| Model compiler / runtime conventions | [`model-compiler-runtime.md`](model-compiler-runtime.md) | llama.cpp deploy paths, REPACK kernel selection, on-board placement |

## Submodules (`upstream/`)

| Path | Upstream | Pin | Authoritative for | Init |
|---|---|---|---|---|
| `upstream/gemma` | <https://github.com/google-deepmind/gemma> | tracking `main` (shallow) | Gemma JAX/Flax reference implementation, parameter counts, attention shape | `git submodule update --init docs/references/upstream/gemma` |
| `upstream/llama.cpp` | <https://github.com/ggml-org/llama.cpp> | tracking `master` (shallow) | C/C++ runtime, `convert_hf_to_gguf.py`, `tools/perplexity/`, `llama-quantize`, `llama-cli` | `git submodule update --init docs/references/upstream/llama.cpp` |

Both submodules are configured with `update = none` and `shallow = true` in `.gitmodules`, so a fresh `git clone --recurse-submodules` does NOT pull them. Initialize on demand only — see commands above.

To bump a pinned commit:

```bash
cd docs/references/upstream/llama.cpp
git fetch --depth 1 origin master
git checkout master
cd ../../../..
git add docs/references/upstream/llama.cpp
git commit -m "deps(llama.cpp): bump submodule to <new-sha>"
```

The HF Python stack (`transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`, `accelerate`) is **not** vendored as a submodule. We `pip install` those libraries; `transformers-trl-peft.md` is the only source-code-aware document we maintain.

## Vendor / mirror policy

- **Link-only** by default for note files. Each file records the URL and the section we rely on.
- **Inline snippet (≤ 50 lines)** is acceptable when the code block is the contract (e.g. an exact chat-template literal). Quote and cite.
- **Submodule** for source code we cite by file/line and want to be able to read locally without leaving the repo. Today: gemma + llama.cpp.

## When to consult these vs project docs

| Question | Look in |
|---|---|
| "What does `SFTConfig.completion_only_loss` actually do?" | `transformers-trl-peft.md` → upstream HF docs / source URL |
| "Is the chat template `<start_of_turn>` or something else?" | `gemma.md` → HF model card; or `upstream/gemma/` for the JAX template |
| "How do I quantize the merged BF16 to Q4_0?" | `llama-cpp.md` → quantize recipe; `upstream/llama.cpp/tools/quantize/` for source |
| "What did the H5R logits-equivalence gate verify?" | `docs/plans/a55-gemma-h5-logits-equivalence.md` (frozen narrative) |
| "What's the prompt rule for refusals?" | `docs/conventions/slm-system-prompt.md` (R-3) |
| "What hyperparameters did *we* settle on?" | `docs/guides/finetune-best-practices.md` |
