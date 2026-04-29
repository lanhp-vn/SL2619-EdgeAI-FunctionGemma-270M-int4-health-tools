# Gemma 3 270M and FunctionGemma — Upstream Sources

## Gemma 3 270M-IT (base model)

| Source | URL | Authoritative for |
|---|---|---|
| Google AI for Developers — Gemma 3 docs | <https://ai.google.dev/gemma/docs/core> | model overview, prompt format, license, recommended sampling |
| Google Developers Blog — introducing 270M | <https://developers.googleblog.com/en/introducing-gemma-3-270m/> | sizing rationale, target use cases, on-device perf claims |
| HuggingFace model card — `google/gemma-3-270m-it` | <https://huggingface.co/google/gemma-3-270m-it> | weights, tokenizer, `chat_template.jinja`, license terms |
| Google DeepMind reference implementation | <https://github.com/google-deepmind/gemma> | architecture (JAX/Flax), parameter counts, attention shape |
| Google DevBlog — "Own your AI" 270M fine-tune walkthrough | <https://developers.googleblog.com/own-your-ai-fine-tune-gemma-3-270m-for-on-device/> | end-to-end SFT + on-device deploy story (vendor-recommended path) |

**Consult when:** changing tokenizer config, prompt format, chat-template
markers, or LoRA target modules. The HF model card's `chat_template.jinja`
is the contract for `prompt_composer.py` literals (`<start_of_turn>`,
`<end_of_turn>`).

**Vendor policy:** link-only. The chat-template token strings are inlined
in `src/gemma_tools/prompt_composer.py` as constants — replicate, don't import.

## FunctionGemma (270M function-calling variant)

| Source | URL | Authoritative for |
|---|---|---|
| Google blog — FunctionGemma announcement | <https://blog.google/technology/developers/functiongemma/> | release framing, target use cases (mobile actions, on-device agents) |
| Google AI for Developers — FunctionGemma overview | <https://ai.google.dev/gemma/docs/functiongemma> | tool-call schema, system-prompt shape, sampling defaults |
| HuggingFace — `google/functiongemma-270m-it` | <https://huggingface.co/google/functiongemma-270m-it> | weights + chat template (extended for tool turns) |
| Vertex AI Model Garden | <https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma> | hosted-inference parity reference |

**Status (as of 2026-04-29):** released April 2026. Same 270M backbone as
Gemma 3 270M-IT, fine-tuned for structured function-calling. Reported boost
in "Mobile Actions" eval from 58% (base) → 85% (FT). Future work in this repo
will add a `--base google/functiongemma-270m-it` SFT path.

**Consult when:** building agentic / tool-calling fine-tunes, or comparing
the 270M-IT base against FunctionGemma as the SFT starting point.

**Vendor policy:** link-only. If the chat template adds new turn markers
for tool calls, replicate them in `prompt_composer.py` with a new
candidate key (e.g. `"functiongemma"`).

## Gemma license

All Gemma weights ship under the Gemma Terms of Use: open weights, commercial
use permitted, redistribution conditional on the prohibited-use policy. See
the model card on HuggingFace for the canonical text.
