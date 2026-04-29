# Upstream Reference Sources

Curated pointers to authoritative upstream material for everything this repo
touches. Prefer **links** over vendoring; pin a commit SHA in source code only
when the implementation depends on a specific revision (none today).

| Topic | File | Authoritative for |
|---|---|---|
| Gemma 3 270M (base + IT) and FunctionGemma | [`gemma.md`](gemma.md) | model architecture, chat template, license, model cards |
| Hugging Face training stack (Transformers, TRL, PEFT, datasets) | [`transformers-trl-peft.md`](transformers-trl-peft.md) | SFT API surface, `SFTConfig` flags, LoRA target modules |
| llama.cpp + GGUF | [`llama-cpp.md`](llama-cpp.md) | GGUF conversion, quantization recipes, `--jinja` chat template handling |

## Vendor / mirror policy

- **Link-only** by default. The reference file records the URL and the section we rely on.
- **Inline snippet (≤ 50 lines)** is acceptable when the code block is the contract (e.g. an exact chat-template literal). Quote and cite.
- **Vendor (full file)** only with a clear reason and a pinned commit SHA. None today.

## When to consult these vs project docs

| Question | Look in |
|---|---|
| "What does `SFTConfig.completion_only_loss` actually do?" | `transformers-trl-peft.md` → TRL source |
| "Is the chat template `<start_of_turn>` or something else?" | `gemma.md` → HF model card |
| "How do I quantize the merged BF16 to Q4_0?" | `llama-cpp.md` → quantize recipe |
| "What hyperparameters did *we* settle on?" | `docs/plans/a55-gemma-fine-tune.md` |
| "What's the prompt rule for refusals?" | `docs/conventions/slm-system-prompt.md` (R-3) |
