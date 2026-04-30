# FunctionGemma 270M-IT — Patient-Health-YAML Agent Plan

> **Status (2026-04-30):** **M1 + M1.5 DONE** — `functiongemma` extra (incl.
> `llama-cpp-python>=0.3`) lands; `fg-q4_k_m.gguf` produced; G_FG_GGUF_PREFLIGHT
> gate passes via two paths (see §15.6). Two upstream `llama-cli` bugs at
> submodule pin `d775992` were diagnosed and worked around — **do not** use
> `--no-conversation`/`-no-cnv` with this pin. Next runnable: **M2** (Phase A
> smoke script).
>
> **Status (2026-04-29):** OQ-1…OQ-9 RESOLVED — see §13. **Training is always
> on `nouslogic-server` (RTX 5080, cu128).** The host is reserved for what it
> does well — GGUF pre-flight, HF Transformers smoke, tool-registry unit
> tests, dataset authoring + LLM-augmented expansion. SFT itself runs on the
> server (Phase D); no CPU-only training path is planned. Use case unchanged:
> an on-device patient-health agent that answers questions by issuing function
> calls against a synthetic patient YAML knowledge base.
>
> **Two findings that drove the contraction (2026-04-29):**
> 1. **Distil cannot train FG-270M for tool-calling.** Distil's
>    [`model-catalog.md` Task Compatibility table](../../references/upstream/distil-cli-skill/references/model-catalog.md)
>    restricts `tool-calling-closed-book` and `multi-turn-tool-calling-closed-book`
>    to **Qwen3 and Llama 3-family students only**. FunctionGemma 270M is in the
>    student catalog but **excluded from both tool-calling task types**. Distil
>    also does not expose the synthesized training corpus (only test-set
>    predictions per `references/tasks/retrieve-predictions.md`), so "use Distil
>    as a synthetic-data generator only" is not a viable workaround.
> 2. **SyNAP toolkit is not for LLMs.** Vendor working-with-models doc lists
>    only TFLite/ONNX/TorchScript/TF/Caffe → vision toolkit. **llama.cpp is the
>    canonical CPU runtime** — Distil's own `deployment-integration.md` agrees.
>
> Result: the dataset workflow switches from Distil-CLI to **hand-authored +
> LLM-augmented seeds** (Pro Perplexity / Claude / ChatGPT) following the vendor
> cookbook's `finetuning-with-functiongemma.ipynb` HF chat-template format. See §9.
>
> **Training stack (decided 2026-04-29):** Phase D adopts **Unsloth** (`FastLanguageModel`
> + `train_on_responses_only` + `save_pretrained_merged` / `save_pretrained_gguf`)
> as the standard procedure, mirroring the
> [`unslothai/notebooks` FunctionGemma_(270M).ipynb](https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb)
> recipe. Rationale: ~2× speed, 30 % VRAM reduction (`use_gradient_checkpointing="unsloth"`),
> FG-aware response-only masking, native GGUF export. The vendor `transformers`/`trl`
> path is kept as a documented fallback only. See §10.
>
> This plan **does not authorize** training, large model downloads, or any write
> action on `nouslogic-server`. Every step that mutates the host or the server
> needs explicit user confirmation per the Discipline rules in the repo's
> [`CLAUDE.md`](../../../CLAUDE.md).
>
> **Companion docs (ground-truth pointers; do NOT inline-restate):**
> - [`docs/references/gemma.md` §FunctionGemma](../../references/gemma.md) — vendor URLs, license, "FunctionGemma future work" pointer this plan supersedes.
> - [`docs/references/transformers-trl-peft.md`](../../references/transformers-trl-peft.md) — TRL/PEFT API surface, `SFTConfig` flags, vendor LoRA recipe.
> - [`docs/references/llama-cpp.md`](../../references/llama-cpp.md) — HF→GGUF→Q4_0 + `--jinja` chat-template handling.
> - [`docs/conventions/doc-update.md` §8.1](../../conventions/doc-update.md) — DRY canonical-ownership registry; this plan is registered for the FunctionGemma topic.
> - [`docs/conventions/code-style-python.md`](../../conventions/code-style-python.md) — Python style baseline (mypy strict, ruff, Pydantic, type hints).
> - [`docs/plans/gemma3-270M/models-testing-plan.md`](../gemma3-270M/models-testing-plan.md) — frozen narrative of the closed-world QA pivot the FunctionGemma path inherits its YAML schema and prompt-class taxonomy from.
> - [`models/gemma-3-270m-it/README.md`](../../../models/gemma-3-270m-it/README.md) — base-model fingerprint (vocab, context, IFEval, sampling). Reused as background — FunctionGemma shares the architecture.
>
> **Local primary-source artifacts (in this repo after 2026-04-29 setup):**
> - [`docs/references/upstream/cookbook/docs/functiongemma/`](../../references/upstream/cookbook/docs/functiongemma/) — three vendor notebooks (`function-calling-with-hf.ipynb`, `full-function-calling-sequence-with-functiongemma.ipynb`, `finetuning-with-functiongemma.ipynb`).
> - [`docs/references/upstream/cookbook/.archive/FunctionGemma/`](../../references/upstream/cookbook/.archive/FunctionGemma) — Mobile-Actions fine-tune notebooks (HF + Tunix).
> - [`docs/references/upstream/distil-cli-skill/`](../../references/upstream/distil-cli-skill/) — Claude Code skill + `SKILL.md` + `workflows/` for the distil-cli teacher-student pipeline.
> - [`docs/references/upstream/unsloth-notebooks/nb/FunctionGemma_(270M).ipynb`](../../references/upstream/unsloth-notebooks/nb/FunctionGemma_(270M).ipynb) — **Phase D standard procedure**, the canonical Unsloth-blessed FG-270M LoRA recipe (LoRA r=128, `train_on_responses_only`, `save_pretrained_gguf`).

---

## 1. Status Snapshot

| Track | State | Source of truth |
|---|---|---|
| Local repo scaffolding (Gemma 3 SFT path) | ✅ Operational | `scripts/finetune.py`, `src/gemma_tools/`, `data/sft_v1.*.jsonl` |
| Vendor docs cloned to `docs/references/upstream/` | ✅ DONE 2026-04-29 — `cookbook/`, `distil-cli-skill/` (HEAD shallow, not pinned to submodules) | `git status --short docs/references/upstream/` |
| `google/functiongemma-270m-it` weights present | ✅ DONE 2026-04-30 — `~/hf-cache/functiongemma-270m-it/` (HF safetensors + tokenizer + chat-template) and `fg-q4_k_m.gguf` (253 MB Q4_K_M) | §15.6 |
| Distil CLI installed locally | ✅ Installed 2026-04-29 (`curl …/install.sh`) | §9.4 (deferred) |
| Distil account / API token | ✅ `distil login` as `lanhp@uci.edu` — but **blocked for tool-calling tasks** per vendor compatibility table (OQ-3) | OQ-3 |
| Local GPU (host) | ❌ None — WSL2 host has no `nvidia-smi`. CPU-only smoke is the only host capability | §7.1 |
| `nouslogic-server` SSH | ✅ Reachable; `ssh nouslogic-server` works for the user (passphrase-gated key). `ssh-agent` not loaded → agent probes fail until the user `ssh-add`s the key in this shell. | §12.2 |
| `nouslogic-server` GPU | ✅ RTX 5080 (16 GiB), driver 580.126.09, 15 GiB free; torch 2.11.0+cu128, CUDA available | server probe 2026-04-29 |
| `nouslogic-server` RAM / disk | ✅ 47 GiB RAM (40 GiB available), 411 GiB free under `/home/hoanglan` | server probe |
| `nouslogic-server` existing `~/sl2619-finetune/` | ✅ Already provisioned by `scripts/server-bootstrap.sh` — reusable venv + llama.cpp build | server probe |
| FunctionGemma plan / dataset / scripts in repo | ❌ None — this doc is the plan, no code yet | this file |

**Phase ladder (revised 2026-04-29):**
| Phase | Where | One-line outcome | Gate |
|---|---|---|---|
| **A0 — GGUF pre-flight** | host | Convert base FG-270M to GGUF, smoke `llama-cli --jinja`, verify control tokens survive (OQ-9) | G_FG_GGUF_PREFLIGHT |
| **A — Host smoke (HF / Transformers)** | host CPU | Round-trip a single-turn function call using the vendor HF example | G_FG_LOAD, G_FG_SINGLE |
| **B — Tool registry + seed dataset** | host | Patient-YAML tool surface (Python) + ~50 hand-authored multi-turn seeds | G_TOOLS_TESTS |
| **C — LLM-augmented seed expansion** | host | Use Pro Perplexity / Claude / ChatGPT to grow seeds → ~300–500 conversations in HF chat-template messages format (vendor cookbook recipe) | G_DATASET_SHAPE |
| **D — Server LoRA SFT (CRITICAL PATH)** | `nouslogic-server` (RTX 5080, cu128) | LoRA SFT of `google/functiongemma-270m-it`; merge; behavioral eval on held-out test set | G_TRAIN, G_EVAL |
| **D-post — GGUF round-trip on FT'd model** | host | Convert + Q4_0 quantize the FT'd merge; `llama-cli --jinja` smoke matches HF BF16 within tolerance | G_GGUF |
| **E — On-device packaging** | SL2619 | Push GGUF + on-device behavioral test on the SL2619 board | G_DEVICE — **out-of-scope for this plan; plan only stops at "GGUF available + smoke-tested on host"** |

**Critical path is A0 → A → B → C → D → D-post.** Host owns pre-flight,
smoke, tool registry, dataset; server owns training. Phase E is named for
navigation only and is **explicitly deferred**.

---

## 2. Scope

### 2.1 In scope (this plan)

- Make this repo capable of loading `google/functiongemma-270m-it` and emitting valid function-call wire format.
- Define a Python tool registry that operates on the existing `data/health_table_v1.yaml` patient fixture.
- Produce a teacher-distilled SFT dataset for multi-turn patient-health agent behavior using `distil-cli`.
- Run LoRA SFT on `nouslogic-server` and verify the fine-tuned model outperforms the base on a held-out patient-YAML eval.
- Convert the merged adapter to GGUF Q4_0 and behaviorally smoke-test it on the host.

### 2.2 Out of scope (explicit)

- ❌ **Real PHI of any kind.** All patient YAMLs in this plan are synthetic/anonymized; the existing `data/health_table_v1.yaml` is the only allowed seed. Per OQ-5, an external review must precede any switch to non-synthetic data.
- ❌ **On-device deploy to SL2619.** Phase E above is referenced but not executed in this plan. Reuse [`docs/deployment/sl2619-board.md`](../../deployment/sl2619-board.md) when the time comes.
- ❌ **Open-domain conversation** outside the agent task. FunctionGemma's vendor card is explicit: "not intended for use as a direct dialogue model."
- ❌ **Multi-step (chained) function calling beyond 2 turns.** Vendor: "Not Explicitly Trained" for chained Tool A → Tool B graphs. Distil's blog showed it can be SFT'd in, but Phase D's training budget caps at 2-turn slot-fill chains.
- ❌ **Fine-tuning the base Gemma 3 270M-IT** for a tool-calling task. The whole reason FunctionGemma exists is the new chat format; the base-model SFT path stays in [`docs/plans/gemma3-270M/`](../gemma3-270M/).
- ❌ **Production safety review / clinician sign-off** of generated answers. This is a research workspace.

---

## 3. Goals and Non-Goals

### 3.1 Goals (acceptance-testable)

1. **G_FG_LOAD** — `transformers.AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")` succeeds on this host with `dtype=torch.float32` (CPU); peak RSS ≤ 3 GiB; load time ≤ 90 s.
2. **G_FG_SINGLE** — A single-turn call against the HF cookbook's `get_current_weather` example emits exactly one well-formed `<start_function_call>...<end_function_call>` block, parsable by a Python regex.
3. **G_TOOLS_TESTS** — The Python tool registry covers ≥ 6 patient-YAML tools and has ≥ 90 % branch coverage in `pytest`. Every tool returns a stable JSON-serializable dict.
4. **G_DATASET_SHAPE** — Distil-generated dataset validates against a Pydantic schema: every row has ≥ 2 messages; every `assistant.tool_calls[*]` references a tool in the registry; every `tool` message has a matching call ID; ≥ 80 % of rows pass syntactic + semantic validation.
5. **G_TRAIN** — SFT on `nouslogic-server` completes in ≤ 60 min wall, eval-loss strictly monotone-decreasing across the 3 epochs the run is configured for, no OOM, trainable-parameter ratio ≤ 5 % under LoRA.
6. **G_EVAL** — On a 60-prompt held-out eval (mix of single, parallel, 2-turn slot-fill, refusal), the FT'd adapter achieves ≥ 80 % tool-call equivalence vs the gold trace; baseline FunctionGemma achieves < 30 % on the same set (per Distil's blog reproduction floor).

### 3.2 Non-goals

- Beating the vendor's BFCL numbers. Our gate is **patient-YAML behavioral fit**, not generic BFCL.
- A multi-LLM ensemble. One model per session.
- Latency optimization. We accept that on host CPU, a single turn takes 10-30 s; on the RTX 5080 it should be < 1 s.

---

## 4. Source / Reference Index

### 4.1 Vendor primary sources (URL-only, link policy per `docs/references/README.md`)

| Topic | URL |
|---|---|
| Model card | <https://huggingface.co/google/functiongemma-270m-it> |
| Google AI overview | <https://ai.google.dev/gemma/docs/functiongemma> |
| Formatting + best practices | <https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices> |
| Model card mirror (Google AI) | <https://ai.google.dev/gemma/docs/functiongemma/model_card> |
| Function calling with HF Transformers | <https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf> |
| Full multi-turn sequence | <https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma> |
| Fine-tuning guide | <https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma> |
| Vertex AI Model Garden listing | <https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/functiongemma> |
| Distil Labs — 270M multi-turn write-up | <https://www.distillabs.ai/blog/making-functiongemma-work-multi-turn-tool-calling-at-270m-parameters/> |
| Distil — distil-home-assistant-functiongemma | <https://huggingface.co/distil-labs/distil-home-assistant-functiongemma> |
| Unsloth FunctionGemma_(270M).ipynb (Phase D standard procedure) | <https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb> |
| Unsloth FG-270M model mirror (used by `FastLanguageModel.from_pretrained`) | <https://huggingface.co/unsloth/functiongemma-270m-it> |
| Unsloth docs — install + LoRA recipe | <https://unsloth.ai/docs/get-started/install> |
| Synaptics Astra Machina docs | <https://synaptics-synap.github.io/doc/v/latest/docs/manual/index.html> |

### 4.2 Local clones (under `docs/references/upstream/`)

| Path | Pin (HEAD captured 2026-04-29) | Purpose |
|---|---|---|
| `cookbook/` | shallow clone, HEAD `65dfbcf0d1f1f8af6824fc1601a7aef4473dbf1e` (not yet a submodule) | three FG notebooks under `docs/functiongemma/` + `.archive/FunctionGemma/` Mobile-Actions notebooks |
| `distil-cli-skill/` | shallow clone, HEAD `566eb9e588f5c9a244fff6e2ddb956b1e4d92e0d` (not yet a submodule) | Claude Code plugin + `SKILL.md` + `workflows/` for the distillation pipeline |
| `unsloth-notebooks/` | shallow + sparse clone, HEAD `fc876d51b973c1bf0058ed85e47602cfb4bac185` (not yet a submodule) | **Phase D standard procedure** — `nb/FunctionGemma_(270M).ipynb` only. Sparse-checkout via `git sparse-checkout set --no-cone '/nb/FunctionGemma_(270M).ipynb' '/README.md' '/LICENSE'` so we don't pull all 200+ unrelated notebooks. |
| `gemma/` | existing submodule | architecture reference (FG shares it) |
| `llama.cpp/` | existing submodule | GGUF convert + quantize |
| `Synaptics/*` | existing submodules | board / runtime — not load-bearing for Phases A–D |

> Pin recovery: `git -C docs/references/upstream/cookbook fetch --depth 50 origin && git -C docs/references/upstream/cookbook checkout <SHA>` if upstream renames the FG notebook paths after 2026-04-29.

> **Submodule promotion.** `cookbook` and `distil-cli-skill` were cloned with
> a plain `git clone --depth 1` so this plan can reference exact files. If the
> user wants them promoted to opt-in submodules (`update = none` like the
> existing entries), add them to `.gitmodules` per [`docs/references/README.md`](../../references/README.md)
> in a separate commit. Until then, treat them as developer-side scratch under
> `docs/references/upstream/` and **don't** `git add` their bodies.

### 4.3 Key in-repo files this plan extends

| File | Role in current repo | Role in this plan |
|---|---|---|
| `data/health_table_v1.yaml` | Patient fixture (closed-world QA) | Reused verbatim as the tool-registry's data source |
| `data/prompts.yaml` | 15-prompt suite for closed-world QA | Reused as the *baseline* eval (FG should still answer fact-lookup correctly via tools) |
| `src/gemma_tools/health_table.py` | Pydantic loader for the YAML | Reused; tool registry imports it |
| `src/gemma_tools/prompt_composer.py` | Renders the directive system + YAML for Gemma 3 IT | **Not used directly** — FG uses a different chat format. The composer's `render_health_yaml()` helper is reusable as a tool-output formatter, but the system-prompt template is replaced. |
| `scripts/finetune.py` | Gemma 3 270M-IT QLoRA SFT entry point | **Forked**, not edited. New `scripts/finetune_functiongemma.py` to keep both paths intact. |
| `scripts/server-bootstrap.sh` | Provisions `~/sl2619-finetune/` venv on the GPU server | Reusable as-is; the FG path will share the same SFT stack |
| `pyproject.toml` | Currently only `pyyaml` runtime + dev tooling | Add `pydantic`, `transformers`, `trl`, `peft`, `datasets`, `bitsandbytes` as **opt-in extras** so the host install stays lightweight (see §7.2) |

---

## 5. Repo Context (current state, 2026-04-29)

The repo currently ships:

- **Gemma 3 270M-IT QLoRA SFT path** (server-side): `scripts/finetune.py` → `scripts/merge.py` → llama.cpp convert → `scripts/smoke_test.py`. Operational; the as-executed run is documented in [`models/gemma-3-270m-it/README.md` §8.5](../../../models/gemma-3-270m-it/README.md).
- **Closed-world health-YAML QA** dataset (`data/sft_v1.*.jsonl`, 1023 train / 126 val / 110 test) generated from `data/clean_sft_dataset.json`. The training prompt shape folds the `ROLE/TASK/RULES/FORMAT/DATE/YAML` directive into the user turn (Gemma 3 has no `system` role).
- **Unit tests** (`tests/`) for the composer, health-table loader, SFT builder, bench harness, and KL-divergence equivalence gate.

What's missing for FunctionGemma:

1. **Wire-format support.** `prompt_composer.py` only knows the Gemma 3 IT chat shape. FunctionGemma adds `<start_function_declaration>`, `<start_function_call>`, `<start_function_response>`, `<escape>` tokens and a new `developer` role.
2. **Tool registry.** No Python module today maps a tool name + args → a typed result over `health_table_v1.yaml`.
3. **Multi-role dataset shape.** `_to_prompt_completion()` in `scripts/finetune.py` asserts the row has exactly `[user, assistant]`. FG rows look like `[developer, user, assistant_with_tool_calls, tool, assistant]`.
4. **Tool-call parser + validator.** No code today reads model output and extracts `<start_function_call>...<end_function_call>` blocks.

The plan adds these in Phases A–D below; nothing existing is mutated until the user approves Phase D.

---

## 6. FunctionGemma Overview

### 6.1 Identity (vendor-sourced)

| Property | Value | Source |
|---|---|---|
| Name | `google/functiongemma-270m-it` | HF model card |
| Backbone | Same architecture as Gemma 3 270M (4 attn heads, 1 KV head, vocab 262 144, sliding window 512, 18 layers, hidden 640) | HF model card; Gemma 3 270M README §1 |
| Chat format | **Different from Gemma 3** — adds `developer` role, function-call/decl/response control tokens, `<escape>` string delimiter | Formatting-and-best-practices doc |
| Knowledge cutoff | August 2024 | HF model card |
| Training tokens | 6T | HF model card |
| Headline BFCL Simple (0-shot) | 61.6 | HF model card |
| Headline BFCL Live Multiple | 25.7 | HF model card |
| Mobile Actions vendor fine-tune | base 58 % → SFT 85 % | HF model card |
| On-device perf (Samsung S25 Ultra, dynamic_int8, 512/32 prefill/decode) | TTFT 0.3 s; decode 125.9 tok/s; model 288 MB; peak RSS 551 MB | HF model card |
| License | `gemma` (open weights, license-gated download) | HF model card |
| Disclaimer | "Not intended for use as a direct dialogue model" | HF model card |

> **Design implication.** FunctionGemma is *not* a drop-in replacement for the
> closed-world QA path documented in [`docs/plans/gemma3-270M/models-testing-plan.md`](../gemma3-270M/models-testing-plan.md).
> Where Gemma 3 270M-IT *retrieves and quotes* YAML, FunctionGemma *issues a
> tool call* against a Python function that reads the YAML. The two paths are
> complementary — see OQ-1.

### 6.2 Wire format — special tokens

Verbatim from the formatting-and-best-practices doc:

| Token pair | Purpose |
|---|---|
| `<start_function_declaration>` / `<end_function_declaration>` | Defines a tool (placed inside the `developer` turn) |
| `<start_function_call>` / `<end_function_call>` | Model emits a tool invocation |
| `<start_function_response>` / `<end_function_response>` | Tool result (provided by the orchestration loop, role `tool`) |
| `<escape>` | Delimits all string values inside structured blocks (so commas/braces inside strings don't terminate the block) |

> The `<escape>` delimiter is **not** an XML/HTML escape; it is a literal
> sentinel surrounding string values inside the function call/response
> structure. Treat all parser code as needing to split on `<escape>...<escape>`
> rather than inferring quote boundaries.

**Critical caller-side distinction (cookbook-confirmed).** Application code
**never writes** the `<start_function_declaration>` string by hand. Callers
pass **standard JSON-Schema** tool definitions to
`processor.apply_chat_template(..., tools=[...])`; the chat template **renders**
the JSON-Schema into the bespoke wire-format string before the prompt hits the
model. Verified by reading `cookbook/docs/functiongemma/function-calling-with-hf.ipynb`
cell 14 (HEAD `65dfbcf0`). The same separation applies on output: model emits the
bespoke `<start_function_call>...` string, parser regex extracts back to JSON.

The canonical extract regex from `cookbook/docs/functiongemma/full-function-calling-sequence-with-functiongemma.ipynb`
cell 20 is:

```python
re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)
# inside each call's args:
re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
```

Reuse this verbatim in `src/gemma_tools/functiongemma_parser.py` (Phase A).

### 6.3 Conversation roles

| Role (in HF chat-template message dict) | On-the-wire token | Purpose |
|---|---|---|
| `developer` | `<start_of_turn>developer` | System / tool declarations / first turn only |
| `user` | `<start_of_turn>user` | User utterance |
| `assistant` | `<start_of_turn>model` | Model output (function call OR final NL answer) |
| `tool` | `<start_of_turn>tool` | Function-result payload, role-injected by orchestration |

> The HF processor's `apply_chat_template` translates the dict role
> `assistant` to the wire token `<start_of_turn>model`. **Do not hard-code
> the wire literal; always go through `processor.apply_chat_template(...)`.**

### 6.4 Single-turn flow

**Caller side (Python, what the application writes — verbatim from `function-calling-with-hf.ipynb` cell 14):**

```python
weather_function_schema = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }
}
message = [
    {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
    {"role": "user", "content": "What's the temperature in London?"},
]
inputs = processor.apply_chat_template(
    message, tools=[weather_function_schema],
    add_generation_prompt=True, return_dict=True, return_tensors="pt",
)
```

**On-the-wire (what `apply_chat_template` produces; readable via
`processor.decode(inputs["input_ids"][0], skip_special_tokens=False)`):**

```
<start_of_turn>developer
You are a model that can do function calling with the following functions
<start_function_declaration>
declaration:get_current_temperature{
  description:<escape>Gets the current temperature for a given location.<escape>,
  parameters:{
    properties:{location:{description:<escape>The city name, e.g. San Francisco<escape>,type:<escape>STRING<escape>}},
    required:[<escape>location<escape>],
    type:<escape>OBJECT<escape>
  }
}
<end_function_declaration>
<end_of_turn>
<start_of_turn>user
What's the temperature in London?
<end_of_turn>
<start_of_turn>model
<start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
<end_of_turn>
```

> The literal string **"You are a model that can do function calling with the
> following functions"** in the developer turn is a *prompt-based trigger* that
> activates the function-calling logic (vendor doc + `function-calling-with-hf.ipynb`
> cell 14 comment: *"This line activates the model's function calling logic."*).
> Do not paraphrase.

### 6.5 Multi-turn flow (with tool result)

After the model's first call lands, the orchestration loop appends two
messages and re-applies the chat template (verbatim shape from
`full-function-calling-sequence-with-functiongemma.ipynb` cells 17, 20, 23):

```python
# (1) Append the model's call as an assistant message with tool_calls:
message.append({
    "role": "assistant",
    "tool_calls": [{"type": "function", "function": call} for call in calls],
})

# (2) Execute and append a tool message:
results = [
    {"name": c["name"], "response": tool_registry[c["name"]](**c["arguments"])}
    for c in calls
]
message.append({"role": "tool", "content": results})

# (3) Re-apply chat template with tools= still set; model now generates the NL answer:
inputs = processor.apply_chat_template(
    message, tools=tools, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
out = model.generate(**inputs.to(model.device),
                     pad_token_id=processor.eos_token_id, max_new_tokens=128)
```

> Cell 20 carries an explicit security warning verbatim: *"Using `globals()`
> to call functions dynamically can be dangerous in production. In a real
> application, you should implement a secure way to map function names to
> actual function calls, such as a predefined dictionary of allowed tools and
> their implementations."* The plan's tool registry (§9.2) is exactly this
> dispatch dictionary — no `globals()` lookups in production code.

### 6.6 Parallel function calls

When the user asks two unrelated questions, the model can emit multiple
`<start_function_call>...<end_function_call>` blocks in a single assistant
turn. The orchestrator executes them in any order and concatenates the
responses into a single `tool` turn before re-prompting.

### 6.7 Sampling defaults

The HF model card and the GA docs **do not pin** temperature / top_p / top_k.
The example uses `model.generate(..., max_new_tokens=128)` only. **Plan default**:
`do_sample=False, max_new_tokens=128` for tests; flip to `do_sample=True,
temperature=0.2, top_p=0.95, top_k=64, min_p=0.0` for production agent runs.
Document the chosen pair next to every bench artifact.

### 6.8 Known limitations (vendor)

- "Not explicitly trained" for **multi-step (chained)** workflows where Tool A's output is Tool B's input.
- "Not explicitly trained" for **multi-turn** workflows that require state across turns to fill in tool args (slot-filling).
- English-only safety eval.
- "May generate incorrect or outdated factual statements" (Aug 2024 cutoff).
- Tokenizer note: "The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config." → **always set tokenizer/PAD before generation; do not assume Gemma 3 IT defaults.**

### 6.9 Distil's headline (the gap that motivates Phase C–D)

Distil reproduced FunctionGemma at 270M and measured baseline multi-turn
tool-call equivalence at **10 – 39 %**. After teacher-distilled SFT (GPT-oss-120B
teacher, 5 000 synthetic conversations), they took:

| Domain | Base | Fine-tuned |
|---|---|---|
| Smart-home control | 38.82 % | **96.71 %** (above the 92.11 % teacher) |
| Banking voice assistant | 23.35 % | **90.86 %** |
| Shell-command exec | 9.90 % | **96.04 %** |

Patient-health-YAML is the same shape: small fixed set of tools, multi-turn
slot-filling typical (e.g. *"Schedule X" → "with whom?" → tool call*). We
expect the same 4-10× behavioral lift. **G_EVAL** in §3.1 takes ≥ 80 % as the
floor.

---

## 7. Local x86 Host Setup (this WSL machine)

### 7.1 Hardware constraints (probed 2026-04-29)

| Capability | Value |
|---|---|
| OS | WSL2 Ubuntu 24.04.4 LTS (kernel 6.6.114.1-microsoft-standard) |
| CPU | 12th Gen Intel i7-12700H, 20 cores |
| RAM | 23 GiB total, 19 GiB available, 24 GiB swap |
| Disk free at `$HOME` | 834 GiB |
| GPU | **None visible** — `nvidia-smi` not present, no CUDA visible to WSL |
| Python | 3.12.3 (system), `uv` at `~/.local/bin/uv` |
| Network egress | 200 OK to `huggingface.co`, `github.com`, `ai.google.dev`, `distillabs.ai` |

> **Implication.** The host can do **CPU-only** smoke (Phase A), Distil dataset
> generation (Phase C — generation runs on Distil's cloud, the host just
> orchestrates), and unit tests. **Training (Phase D) MUST run on
> `nouslogic-server`.**

### 7.2 Python venv plan

The repo's existing `pyproject.toml` declares only `pyyaml` at runtime to keep
`uv sync` fast for users who only want the bench tooling. Two layered options:

- **Option 1 (recommended) — opt-in extra.** Add a `[project.optional-dependencies] functiongemma = [...]` group that pulls `transformers`, `accelerate`, `torch` (CPU wheel for host smoke), `huggingface-hub`, `pydantic`, and `jsonschema`. Install with `uv sync --extra functiongemma`. Server gets a separate `serve` extra that pulls the cu128 wheel set.
- **Option 2 — separate venv.** A `requirements-functiongemma.txt` and `python -m venv .venv-fg` keeps the existing `.venv` untouched. Heavier UX cost; chosen only if (1) breaks pin-sensitive tests.

**Plan default: Option 1.** Concrete diff to land in §10 with a unit test that
asserts `import transformers; import torch; from gemma_tools import health_table`
succeeds.

### 7.3 Disk / model-cache layout

```
~/.cache/huggingface/hub/                              # HF default; ~540 MB for FG-270M-IT BF16 safetensors
~/nouslogic/gemma3-270M-finetune/
├── data/
│   ├── functiongemma/
│   │   ├── tools_v1.yaml                               # tool registry schema (NEW — Phase B)
│   │   ├── seed_conversations.jsonl                    # 50 hand seeds (NEW — Phase B)
│   │   └── dataset_v1/                                  # hand+LLM-augmented training set (NEW — Phase C)
│   │       ├── train.jsonl
│   │       ├── val.jsonl
│   │       └── test.jsonl
│   └── (existing closed-world files untouched)
├── src/gemma_tools/
│   ├── functiongemma_composer.py                       # NEW — wire-format composer
│   ├── functiongemma_tools.py                          # NEW — tool registry (vitals, meds, allergies, ...)
│   ├── functiongemma_parser.py                         # NEW — extract <start_function_call>...
│   └── (existing modules untouched)
├── scripts/
│   ├── functiongemma_smoke.py                          # NEW — Phase A smoke
│   ├── functiongemma_finetune.py                       # NEW — Phase D SFT (cloned from finetune.py)
│   ├── functiongemma_merge.py                          # NEW — adapter merge (cloned from merge.py)
│   ├── functiongemma_bench.py                          # NEW — eval harness
│   └── (existing scripts untouched)
└── tests/
    ├── test_functiongemma_composer.py
    ├── test_functiongemma_tools.py
    ├── test_functiongemma_parser.py
    └── test_functiongemma_dataset.py
```

Total expected disk add: ≤ 1.5 GiB on host (HF cache + dataset + generated bench logs).
Total expected disk add on server: ≤ 5 GiB (full venv + adapters + GGUF outputs).

### 7.4 HuggingFace auth

`google/functiongemma-270m-it` is gated under the Gemma license. The user must
**accept the license once** at <https://huggingface.co/google/functiongemma-270m-it>
(logged in as their HF account), then `huggingface-cli login` (or `hf auth
login`) on both the host and the server. The model-card download is a no-op until
the click-through is done.

### 7.5 Concrete one-shot host install (after plan approval)

```bash
# After Option 1 lands in pyproject.toml:
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv sync --extra functiongemma
hf auth login   # interactive; user pastes token
hf download google/functiongemma-270m-it
```

---

## 8. Phase A — Single-Turn Smoke Test (host CPU)

### 8.1 Objective

Round-trip a single tool call against the vendor's `get_current_weather`
example to prove the FG wire format works end-to-end on this host with no
GPU. This deliberately uses the **vendor example** rather than the patient-YAML
tools so we isolate "does the model load & emit valid format" from "is our
tool schema correct".

### 8.2 Acceptance gates

- **G_FG_LOAD** — Model loads in `dtype=torch.float32` on CPU; peak RSS ≤ 3 GiB; load time ≤ 90 s on this i7-12700H.
- **G_FG_SINGLE** — Model output contains exactly one `<start_function_call>...<end_function_call>` block; the parser extracts `name="get_current_weather"` and `arguments={"location": "<some-string>"}`.

### 8.3 Concrete steps (run after plan approval)

```bash
uv run python scripts/functiongemma_smoke.py \
  --model google/functiongemma-270m-it \
  --device cpu \
  --max-new-tokens 64 \
  --query "What is the temperature in London?"
```

Expected output (parsed): `{"tool": "get_current_temperature", "args": {"location": "London"}}`.

### 8.4 What the smoke script does

1. Loads tokenizer + model at CPU/float32.
2. Builds the developer turn from `tools_v1.yaml` → JSON-schema → vendor `<start_function_declaration>` block via `apply_chat_template(..., tools=[...])`.
3. Runs `model.generate(..., do_sample=False, max_new_tokens=64)`.
4. Pipes the decode through `functiongemma_parser.extract_calls(text)`.
5. Asserts exactly one call and prints PASS / FAIL.

Total expected wall: 30 – 90 s for one query on this CPU.

---

## 9. Phase B/C — Patient-YAML Dataset Plan

### 9.1 Patient YAML (reused as-is)

`data/health_table_v1.yaml` is the **single source of patient state**. Schema
(see [`src/gemma_tools/health_table.py`](../../../src/gemma_tools/health_table.py)
for the Pydantic model):

```
patient:              name, age, sex, blood_type
vitals:               HR, BP_systolic, BP_diastolic, SpO2, T, RR, last_measured
conditions:           [{name, diagnosed_at, severity, controlled}]
allergies:            [{substance, severity, reaction}]
medications:          [{name, dose, schedule, with_food, purpose, avoid_foods, avoid_drugs}]
dietary_restrictions: [{rule, reason}]
appointments:         [{date, time, provider, purpose, location}]
emergency_contacts:   [{name, relation, phone}]
notes:                [string]
```

> All entries are **synthetic**. The fixture comment at the top of the file
> says so, and OQ-5 reaffirms: real PHI is forbidden.

### 9.2 Tool registry design (Python)

Each tool is a pure function `tool(args: dict, table: HealthTable) -> dict`
with a JSON-schema'd argument list. Initial six (proposed; sized to cover the
existing 15-prompt suite):

| Tool name | Args | Returns |
|---|---|---|
| `get_vitals` | `{}` | full vitals dict (all six measurements + `last_measured`) |
| `get_medications_at_time` | `{ "time_24h": "08:00" }` | list of `{name, dose, schedule, with_food, purpose}` matching the schedule |
| `get_medication_by_name` | `{ "name": "metformin" }` | single med dict (case-insensitive prefix match) |
| `list_allergies` | `{}` | list of `{substance, severity, reaction}` |
| `check_food_interaction` | `{ "food": "grapefruit" }` | `{ "interacts": bool, "with_meds": [str], "rule": str | null }` |
| `get_next_appointment` | `{}` | next appointment dict by date, or `null` if none upcoming |
| `get_emergency_contact` | `{}` | first emergency-contact dict |

Each tool has a Pydantic argument model (validated on call) and a docstring
that becomes the `description` in the FG declaration. The registry exposes
`as_function_declarations() -> list[dict]` returning the JSON-schema list HF's
`apply_chat_template(..., tools=...)` consumes.

> **Tool-set proposal — needs user sign-off (OQ-4).** Adding `schedule_appointment`,
> `set_reminder`, etc. would make the agent a *write* tool and demand a
> separate safety story. This plan stays read-only.

### 9.3 Conversation taxonomy

| Type | Count target (seed) | Example |
|---|---|---|
| Single-turn fact lookup | 12 | `user: "what's my heart rate?" → call:get_vitals{} → tool: {...} → assistant: "Your heart rate is 72 bpm."` |
| Single-turn refusal (off-topic) | 4 | `user: "tell me a joke" → assistant: "I answer questions from your health record only."` (no tool call) |
| Single-turn fact-absence | 4 | `user: "what's my cholesterol level?" → call:get_vitals{} → tool: (no cholesterol) → assistant: "Cholesterol is not in record."` |
| Parallel calls | 6 | `user: "what allergies do I have and what's my BP?" → assistant emits both calls in one turn` |
| 2-turn slot-filling | 14 | `user: "what dose of metformin?" → call:get_medication_by_name{name:metformin} → tool: {dose:"500 mg",...} → assistant: "Metformin 500 mg, twice daily."` then `user: "with food?" → call:get_medication_by_name{name:metformin} → tool: {with_food:true,...} → assistant: "Yes, take it with food."` |
| Domain refusal w/ medical-advice route | 4 | `user: "should I stop taking aspirin?" → assistant: "I cannot give medical advice; consult your clinician."` (no tool call) |
| Tool-error recovery | 6 | `user: "what's my pulse at 3am?" → call:get_vitals{} → tool: {error:"single snapshot only"} → assistant: "Only the last measurement is on record (08:15)."` |

Total seed: **50 conversations**. Distil-generated target: **1 000 – 3 000**
(per Distil's 5 000-shell-conversation benchmark; we cap lower because the
domain is narrower).

### 9.4 Dataset workflow — cookbook-style hand-authored + LLM-augmented seeds

> **Why not Distil?** Distil's
> [`model-catalog.md` Task Compatibility table](../../references/upstream/distil-cli-skill/references/model-catalog.md)
> restricts `tool-calling-closed-book` and `multi-turn-tool-calling-closed-book`
> to **Qwen3 and Llama 3-family students only**. Quoted verbatim from the
> Distil docs (line 110 of model-catalog.md): *"Tool-calling student shortlist:
> `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B`,
> `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`."*
> FunctionGemma 270M is **excluded**. Distil also does not surface the
> synthesized training corpus (only test-set predictions per
> `references/tasks/retrieve-predictions.md`), so the "use Distil only as a
> data generator" workaround is not viable. **Phase B/C uses the vendor
> cookbook recipe directly.**

#### 9.4.1 Authoring pipeline

| Stage | Source | Output | Validator |
|---|---|---|---|
| Hand-authored seeds | Author writes ~50 multi-turn conversations directly | `data/functiongemma/seed_conversations.jsonl` (HF chat-template messages) | `tests/test_functiongemma_dataset.py` (Pydantic shape) |
| LLM-augmented expansion | Pro Perplexity / Claude.ai / ChatGPT (user runs interactively, not Claude Code) | `data/functiongemma/llm_expanded_v1.jsonl` (~300–500 rows) | Same Pydantic + tool-call argument validator |
| Train/val/test split | Stratified by conversation type (§9.3) | `data/functiongemma/train_v1.jsonl`, `val_v1.jsonl`, `test_v1.jsonl` (held out) | Row counts logged |

#### 9.4.2 Row format (Unsloth notebook recipe; vendor-cookbook compatible)

Every row is a HF chat-template message list, matching cells 11–17 of the
[Unsloth `FunctionGemma_(270M).ipynb`](https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb)
notebook. The vendor cookbook (`finetuning-with-functiongemma.ipynb` HEAD
`65dfbcf0`) is identical except it uses `developer` instead of `system` — the
FG chat template normalizes both. **We follow Unsloth and use `system`** for
parity with the notebook examples.

```jsonl
{"messages":[
  {"role":"system","content":"You are a model that can do function calling with the following functions"},
  {"role":"user","content":"what's my heart rate?"},
  {"role":"assistant","content":"<think>User wants vitals; call get_vitals.</think>","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_vitals","arguments":{}}}]},
  {"role":"tool","name":"get_vitals","tool_call_id":"call_1","content":"{\"HR\":72,\"BP_systolic\":118}"},
  {"role":"assistant","content":"<think>HR is 72 bpm; report directly.</think>\nYour heart rate is 72 bpm."}
],"tools":[{"type":"function","function":{"name":"get_vitals", "...JSON-Schema..."}}]}
```

Three **must-do** transformations the training script applies before handing
the row to `SFTTrainer`:

1. **Pre-render** with `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)` and store as `text`. This is what `SFTTrainer(dataset_text_field="text")` consumes.
2. **`.removeprefix("<bos>")`** on every rendered string. Otherwise SFTTrainer prepends `<bos>` again at tokenize-time → double-BOS → silent training-data corruption. The Unsloth notebook does this in cell 25; we replicate it.
3. **Normalize `tool_calls`** to `{id, type:"function", function:{name, arguments:dict}}` and **backfill missing `name`** on `tool` turns via the `tool_call_id` → name map. Helper: port the Unsloth `prepare_messages_and_tools` function (notebook cell 23) verbatim into `src/gemma_tools/functiongemma_dataset.py`.

The `tools` field is **per-row** so different prompts can ship different
subsets of the registry. Train-on-completion uses Unsloth's
**`train_on_responses_only(trainer, instruction_part="<start_of_turn>user\n",
response_part="<start_of_turn>model\n")`** rather than TRL's
`completion_only_loss=True` flag — Unsloth's helper is FG-turn-boundary aware
and is what the vendor procedure recommends. See §10.2 for the full SFTConfig.

> **`<think>` block decision (2026-04-29):** Hand-authored seeds **include** a
> short `<think>...</think>` reasoning prelude inside `assistant.content`,
> mirroring Unsloth notebook cell 17. Base FG-270M has no native reasoning;
> the notebook explicitly *adds* it via the `LLM360/TxT360-3efforts` agent
> split. We reproduce that uplift on our own seeds. The G_DATASET_SHAPE
> validator (§9.6) gates: every assistant turn must contain exactly one
> `<think>...</think>` block.

#### 9.4.3 LLM-augmentation prompt template (host-side seed expansion)

User pastes the seed JSONL + the tool-registry JSON-Schema into Pro Perplexity
/ Claude / ChatGPT with the instruction:

```
You are expanding a synthetic training set for a 270M tool-calling SLM.
Patient YAML schema and tool registry are below. Hand-authored seed examples
follow. Produce 10 NEW conversations matching the EXACT same JSON shape:
- Vary the user phrasing, slot-fill order, and tool-error recovery cases.
- Use ONLY the tools listed in the registry — never invent tool names.
- Synthetic patient values only; no real PHI.
- Output one conversation per line as valid JSONL.
```

Iterate ~30–50 rounds → ~300–500 expanded rows. The `tests/test_functiongemma_dataset.py`
validator is the gate: rows that fail the Pydantic shape or reference unknown
tools are quarantined to `data/functiongemma/quarantine.jsonl` for manual
review. Target ≥ 80 % pass rate per G_DATASET_SHAPE.

> **Why not run a local teacher (`vllm serve gpt-oss-120b`)?** The user has Pro
> Perplexity / Claude / ChatGPT access; those teachers produce higher-quality
> synthetic conversations than a self-hosted 120B and require zero local
> compute. Local-teacher remains an OQ-3 fallback only if the LLM-augmentation
> output systematically fails G_DATASET_SHAPE.

#### 9.4.4 Distil — deferred but kept install-ready

Distil CLI is installed and logged in; if a Phase 2 follow-up adds a parallel
**Qwen3-1.7B or Llama-3.2-1B student** alongside FunctionGemma (e.g. for an
A/B comparison), the Distil workflow is the right tool for that variant. Until
then, no `distil model …` commands are run in this plan.

### 9.5 Privacy / synthetic constraints

- All training data references the existing `Test Patient` fixture or new fixtures of the same synthetic shape.
- Distil's teacher (GPT-oss-120B) only sees the synthetic YAML — never user data.
- Generated dataset goes into `data/functiongemma/dataset_v1/` and is **explicitly committable** (the `.gitignore` excludes binaries, not JSONL).
- A pre-commit hook (`pre-commit-functiongemma.sh`) runs a regex sweep for likely real-PHI patterns (US phone formats outside `+1-555-`, real-looking SSNs, real provider names). Any hit blocks commit. This script is part of Phase B.

### 9.6 G_DATASET_SHAPE — validator

A `tests/test_functiongemma_dataset.py` enforces, for every row in
`dataset_v1/{train,val,test}.jsonl`:

1. Row parses as JSON.
2. `messages` field is a list with `role ∈ {system, user, assistant, tool}`.
3. Every `assistant` message with `tool_calls` references a function name that exists in `tools_v1.yaml`.
4. Every `assistant` message contains exactly one `<think>...</think>` block in `content` (per Unsloth notebook cell 17 procedure; §9.4.2).
5. Every `tool` message follows an `assistant` message with `tool_calls` and carries a `tool_call_id` that matches one of the preceding `tool_calls[*].id`.
6. Tool call arguments validate against the tool's Pydantic model.
7. After `tokenizer.apply_chat_template(...).removeprefix("<bos>")`, the rendered string contains no `<bos>` token (double-BOS check).
8. ≥ 80 % of rows must pass; the remainder are quarantined to `dataset_v1/quarantine.jsonl` for manual review.

---

## 10. Phase D — Server SFT via Unsloth (CRITICAL PATH)

> **Status (2026-04-29 update):** Re-elevated to the critical path per user
> direction *"always training on server with high-performance GPU."* Host CPU
> SFT is **not** a planned path — the host owns pre-flight, smoke, tool
> registry, and dataset authoring; the server owns training. Phase D fires
> after Phase C (dataset green) lands.
>
> **Standard procedure:** Mirror the
> [Unsloth `FunctionGemma_(270M).ipynb` notebook](https://github.com/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M).ipynb)
> end-to-end. Rationale per user 2026-04-29: *"if using unsloth gives better
> performance — we should use it."* Unsloth is the vendor-blessed path for
> FG-270M and gives ~2× speed + 30 % VRAM reduction via
> `use_gradient_checkpointing="unsloth"`, FG-aware response-only masking, and
> built-in GGUF export.

### 10.1 Server provisioning

`nouslogic-server` is already provisioned for Gemma 3 SFT under
`~/sl2619-finetune/.venv` (probed 2026-04-29: `torch 2.11.0+cu128`, CUDA on RTX
5080, llama.cpp built at `~/llama.cpp`).

**Decided 2026-04-29: Option A (share the existing venv) + Unsloth install.**
Reuse `~/sl2619-finetune/.venv` because the base SFT stack (`transformers`,
`trl`, `peft`, `bitsandbytes`, `accelerate`) is identical to what Unsloth
requires. Unsloth itself adds ~3 GB of new deps (`unsloth`, `unsloth_zoo`,
`xformers`, `triton`) plus pinned `transformers==4.56.2` + `trl==0.22.2` —
**this `transformers` pin matters**, mitigations below cover the rollback
path. Trade documented in OQ-7 resolution.

**Mandatory mitigations before any pip install on the shared venv:**

```bash
# (1) Capture exact pin file FIRST — never run after the install
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip freeze > ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt &&
  ls -la ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt
'

# (2) Compare BEFORE installing — what's the delta vs Unsloth's pins?
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  python -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"
'

# (3) Install Unsloth (matches notebook cell 4 — local non-Colab branch)
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip install unsloth &&
  pip install transformers==4.56.2 &&
  pip install --no-deps trl==0.22.2 &&
  python -c "from unsloth import FastLanguageModel; print(\"unsloth OK\")"
'

# (4) Rollback procedure (in case Unsloth deps break the Gemma 3 path):
ssh nouslogic-server '
  source ~/sl2619-finetune/.venv/bin/activate &&
  pip install --force-reinstall -r ~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt
'
```

**Rule:** if step (3) breaks the existing Gemma 3 SFT smoke
(`uv run pytest -k gemma3` or `python scripts/smoke_test.py` on the server's
existing checkpoints), **immediately roll back via (4) and switch to Option B**
— isolated venv at `~/functiongemma-finetune/.venv` via a forked
`scripts/server-bootstrap_functiongemma.sh`. The 10-minute re-install + 5 GB
disk is cheaper than silently regressing the proven Gemma 3 path.

### 10.2 Hyperparameters (Unsloth notebook cells 6, 8, 29, 31)

**Model loader** (notebook cell 6):

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/functiongemma-270m-it",   # mirror, not google/...
    max_seq_length  = 4096,
    load_in_4bit    = False,
    load_in_8bit    = False,
    load_in_16bit   = True,    # 16-bit LoRA (new in Unsloth)
    full_finetuning = False,
)
```

**LoRA config** (notebook cell 8 — vendor-recommended for FG-270M):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r                = 128,     # vendor recommendation; was r=16 in our Gemma 3 path
    lora_alpha       = 256,
    target_modules   = ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    lora_dropout     = 0,
    bias             = "none",
    use_gradient_checkpointing = "unsloth",   # 30% VRAM win + 2x batch
    use_rslora       = False,
    loftq_config     = None,
    random_state     = 3407,
)
```

**SFTConfig** (notebook cell 29):

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_dataset,    # pre-rendered "text" field, see §9.4.2
    eval_dataset    = val_dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,    # effective batch 8
        warmup_steps                = 10,
        num_train_epochs            = 3,    # CHANGED from notebook's max_steps=500 — our dataset is ~300-500 rows
        learning_rate               = 2e-4,
        optim                       = "adamw_8bit",
        weight_decay                = 0.001,
        lr_scheduler_type           = "linear",
        logging_steps               = 1,
        seed                        = 3407,
        output_dir                  = "outputs_fg_v1",
        report_to                   = "none",
    ),
)
```

**Response-only masking** (notebook cell 31 — FG-aware, replaces TRL's
`completion_only_loss=True`):

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part    = "<start_of_turn>model\n",
)
```

**Why these settings (deltas from our Gemma 3 path):**

| Setting | Gemma 3 path | Unsloth FG path | Why the change |
|---|---|---|---|
| LoRA `r` | 16 | **128** | Unsloth + Google notebook recommendation for FG-270M; capacity for the new tool-calling skill |
| LoRA `target_modules` | `"all-linear"` | explicit `[q,k,v,o,gate,up,down]` | Identical in practice for Gemma; explicit list matches notebook |
| `max_seq_length` | 1024 | **4096** | Unsloth gradient checkpointing breaks the vocab-OOM constraint |
| `per_device_train_batch_size` | 1 | **4** | Same — Unsloth `use_gradient_checkpointing="unsloth"` lifts the 16 GiB ceiling |
| `gradient_accumulation_steps` | 16 | 2 | effective batch stays at 8 (vs 16) — keep an eye on convergence |
| `learning_rate` | 1e-4 | **2e-4** | Notebook default; safe for LoRA |
| `optim` | `adamw_torch` | **`adamw_8bit`** | bnb 8-bit optimizer, ~50 % memory saving |
| Loss masking | `completion_only_loss=True` | **`train_on_responses_only(...)`** | FG-aware turn-boundary detection |

### 10.3 OOM check on the RTX 5080 with Unsloth

The vocab × seq × bytes math from the Gemma 3 path no longer dominates because
`use_gradient_checkpointing="unsloth"` recomputes activations (incl. logits)
during backward. Empirical Unsloth ceiling for FG-270M on a 16 GiB card per
notebook README is **PDB=4, max_seq=4096** without OOM. Eval is run at
`per_device_eval_batch_size=1` for safety; if eval OOMs, drop to greedy decode
on a 50-row held-out test set instead of in-loop eval.

If the empirical ceiling falls short on our actual run, the documented
fallbacks in order of reach:

1. Drop `max_seq_length` to 2048 (covers all our patient-YAML conversations).
2. Drop PDB to 2, GAS to 4 (effective batch unchanged at 8).
3. Last resort: switch back to the Gemma 3 path's PDB=1, GAS=16 with `r=64`.

### 10.4 Output / artifacts

```
~/functiongemma-finetune/         (NEW on server, ≤ 5 GiB)
├── outputs_fg_v1/                # SFTTrainer checkpoints (interim)
├── functiongemma_lora/           # LoRA adapter via model.save_pretrained()
├── merged_fg_v1/                 # full BF16 HF dir via save_pretrained_merged(merged_16bit)
├── merged_fg_v1.f16.gguf         # via Unsloth save_pretrained_gguf(F16) OR vendor llama.cpp convert
├── merged_fg_v1.q8_0.gguf        # via Unsloth save_pretrained_gguf(Q8_0) — built-in
├── merged_fg_v1.q4_k_m.gguf      # via vendor llama-quantize on the F16 (Unsloth doesn't ship Q4_K_M yet)
├── data/                         # scp'd from host
└── logs/                         # train-*.log, merge-*.log, bench-*.log
```

> **Quantization choice (changed 2026-04-29):** The previous draft targeted
> `Q4_0`. For tool-calling — where small numerical errors in JSON arguments
> cause silent schema violations — `Q4_K_M` (mixed precision, K-quants) is the
> safer default and remains within SL2619 board memory. `Q8_0` is the
> conservative reference and is what Unsloth's `save_pretrained_gguf` ships
> natively. Q4_0 only if Q4_K_M proves too slow on the SL2619.

scp back to host:

```
data/functiongemma/dataset_v1/checkpoints/adapters_fg_v1/
data/functiongemma/dataset_v1/checkpoints/merged_fg_v1.q4_k_m.gguf
data/functiongemma/dataset_v1/checkpoints/merged_fg_v1.q8_0.gguf      # eval reference
docs/bench/2026-MM-DD_functiongemma-eval.md
```

> Per `.gitignore`, `*.gguf` and `*.safetensors` are excluded. Adapters and
> GGUFs are stored under `data/functiongemma/dataset_v1/checkpoints/` for
> ergonomic discovery but never committed.

### 10.5 Rollback / cleanup

- `~/functiongemma-finetune/` is a dedicated tree; `rm -rf` on it does **not** affect `~/sl2619-finetune/`.
- The HF cache `~/.cache/huggingface/hub/models--unsloth--functiongemma-270m-it/` is purgeable with `hf cache purge`.
- If the Unsloth install corrupts something in the shared venv (Option A), `~/sl2619-finetune/.torch-pin-pre-fg-2026-04-29.txt` documents the pre-FG pins; `pip install --force-reinstall -r .torch-pin-pre-fg-2026-04-29.txt` restores.
- If Unsloth itself misbehaves (kernel/triton issue on RTX 5080 + cu128), the documented fallback is the **vanilla TRL+PEFT recipe** preserved in §10.6 — same hyperparameters minus the Unsloth-specific knobs.

### 10.6 Fallback path: vanilla TRL + PEFT (kept for reference only)

If Unsloth proves unstable on the RTX 5080 / cu128 stack, fall back to:

- `transformers.AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it", torch_dtype=torch.bfloat16)`
- `peft.LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")` (smaller `r` because vanilla path has tighter VRAM ceiling)
- `SFTConfig(per_device_train_batch_size=1, gradient_accumulation_steps=16, max_length=1024, completion_only_loss=True, ...)`
- GGUF via `convert_hf_to_gguf.py` + `llama-quantize Q4_K_M` (no `save_pretrained_gguf`)

This is the original Phase D recipe, preserved here verbatim only as a
contingency. Do not run it unless §10.1 step (3) fails.

---

## 11. Testing Plan

### 11.1 Unit (host, `uv run pytest`)

| Test file | Coverage target |
|---|---|
| `tests/test_functiongemma_composer.py` | Builds developer/user/assistant/tool turns; literal token sanity (`<escape>` count even, no double `<start_of_turn>`) |
| `tests/test_functiongemma_tools.py` | Each of the 6+ tools: arg validation, expected return shape, edge cases (empty fields, missing keys, case-insensitive lookups) |
| `tests/test_functiongemma_parser.py` | Parses single, parallel, and tool-error responses; rejects malformed (unmatched braces, missing `<escape>`) |
| `tests/test_functiongemma_dataset.py` | Validates Distil-output JSONL against `Conversation` Pydantic model; counts pass-rate |

### 11.2 Inference regression (host CPU)

A `scripts/functiongemma_smoke.py --regression` mode runs 5 fixed prompts
(`get_current_weather`, `list_allergies`, `get_vitals`, off-topic refusal,
medical-advice refusal) and compares the parsed output against a frozen
`tests/fixtures/functiongemma_regression.jsonl`. Run on every PR that touches
`functiongemma_*`.

**Sampling defaults** (per Unsloth notebook cell 42 + vendor sampling
guidance): `top_p=0.95, top_k=64, temperature=1.0, add_generation_prompt=True`.
The smoke script also calls `.removeprefix('<bos>')` on the rendered prompt
before tokenizing, mirroring §9.4.2 step 2.

### 11.3 Tool-call format validation

A `tools/functiongemma_parser_strict.py` (called from the bench harness)
asserts:

- Exactly one `<start_function_call>...<end_function_call>` per assistant turn (or N parallel calls, no interleaving).
- Every string value enclosed in `<escape>...<escape>`.
- Function name in tool registry.
- Argument keys ⊆ tool's declared parameters.
- Required arguments present.

### 11.4 Patient-YAML QA acceptance set

Reuse the existing `data/prompts.yaml` 15-prompt suite as a **baseline parity
gate**: the FG'd model on the patient YAML must score ≥ 12/15 on the
fact_lookup + fact_absence + domain_refusal classes (matching the existing
fine-tuned Gemma 3 path's Q5 score of 5/15 → ≥ 12/15 is the FG behavioral
target). Plus a new `data/functiongemma/eval_v1.jsonl` of 60 multi-turn
conversations (held out of distil_v1 training).

### 11.5 Failure cases (must pass)

| Case | Expected behavior |
|---|---|
| User asks for a tool that doesn't exist | Model emits an off-topic refusal, NOT a hallucinated tool call |
| Tool returns an error field | Model surfaces the error in NL ("That value isn't in the record"), does not retry the call |
| User asks for medical advice | Model emits the "consult your clinician" refusal, NOT a tool call against any tool |
| YAML field missing | Tool returns `null`; model says "not in record" |
| Two-turn slot-fill where user changes their mind mid-fill | Model issues a fresh tool call with the new args, does not blend args |

---

## 12. Server Fallback Plan

### 12.1 When to use `nouslogic-server`

- **Always** for SFT (Phase D). The host has no GPU.
- **Optionally** for inference benchmarks larger than the 60-prompt eval (host CPU is the floor).
- **Never** for the host smoke (Phase A) — keep that path testable on the developer's laptop for fast iteration.

### 12.2 SSH access (already working interactively)

`~/.ssh/config` has:

```
Host nouslogic-server
    HostName 100.116.133.62
    User hoanglan
    IdentityFile ~/.ssh/nouslogic_server_ed25519
    IdentitiesOnly yes
    PreferredAuthentications publickey
```

The key is passphrase-protected. For automated probes (`board_probe`, agent
runs):

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/nouslogic_server_ed25519   # interactive prompt for passphrase
```

The agent then inherits the loaded identity for the session.

> **Do not** store the passphrase in the repo, the home dir, or any agent-readable file. The user pasted it into chat once for a one-off probe; the plan does not depend on it.

### 12.3 Pre-flight

Before any server-side mutation, run a `/board_probe` (re-purposed for the
GPU server target — the `board_probe` skill in this repo accepts the
fine-tune-server target per `docs/conventions/board_probe`-equivalent flow).
Confirm:

- nvidia-smi shows the RTX 5080 with ≥ 14 GiB free.
- Disk free at `~hoanglan` ≥ 50 GiB.
- `~/sl2619-finetune/.venv/bin/python -c 'import torch; ...'` reports CUDA available.
- `~/llama.cpp/build/bin/llama-quantize` is executable.

### 12.4 Data + model transfer

```
# Host → server (no `ssh`-side writes from agent; user runs):
scp data/functiongemma/dataset_v1/{train,val,test}.jsonl   nouslogic-server:~/functiongemma-finetune/data/
scp scripts/functiongemma_finetune.py                      nouslogic-server:~/functiongemma-finetune/
scp scripts/functiongemma_merge.py                         nouslogic-server:~/functiongemma-finetune/

# Server runs (interactive, user-invoked):
ssh -t nouslogic-server 'cd ~/functiongemma-finetune && source ~/sl2619-finetune/.venv/bin/activate && python functiongemma_finetune.py --dry-run'
ssh -t nouslogic-server 'cd ~/functiongemma-finetune && source ~/sl2619-finetune/.venv/bin/activate && python functiongemma_finetune.py'

# Server → host:
scp -r nouslogic-server:~/functiongemma-finetune/adapters_fg_v1   data/functiongemma/dataset_v1/checkpoints/
scp    nouslogic-server:~/functiongemma-finetune/merged_fg_v1.q4_0.gguf data/functiongemma/dataset_v1/checkpoints/
```

### 12.5 What the agent NEVER does on the server

- `apt install ...`
- `rm -rf ...`
- `pip install --force-reinstall ...` outside an explicit user-approved step
- `git push` / `git reset` against any server-side checkout
- start a long-running training process detached without user `nohup` consent

This mirrors the existing IL "SSH to board is read-only" rule, extended to the GPU server.

---

## 13. Risks and Open Questions

### Assumptions (proceed unless flagged)

- **A1** — The Gemma license click-through is acceptable to the user. (Same gate as existing Gemma 3 IT.)
- **A2** — Patient-YAML use case can be served with **read-only** tools. No "schedule an appointment", no "send to pharmacy".
- **A3** — Distil's teacher (GPT-oss-120B) sees only the synthetic YAML; no real PHI ever leaves the host.
- **A4** — Reusing `~/sl2619-finetune/.venv` on the server is acceptable; the LoRA run will not regress the Gemma 3 SFT path. Mitigation: pin file (§10.5).
- **A5** — The 50-seed → 1000–3000 distilled corpus is enough to land G_EVAL ≥ 80 %. Distil shows 5 000 was enough for SHELL; patient-health is a narrower domain.

### Open Questions — RESOLVED 2026-04-29

- ✅ **OQ-1 RESOLVED** — *Two paths or one?* **Keep both, but minimize coupling.**
  Gemma 3 270M-IT remains the closed-world YAML retrieval path
  ([`docs/plans/gemma3-270M/models-testing-plan.md`](../gemma3-270M/models-testing-plan.md));
  FunctionGemma is the agent / tool-calling path. They share the patient YAML
  fixture and the bench harness; they do **not** share a chat template, a
  system prompt, or a SFT script. Forked `scripts/finetune_functiongemma.py`
  (per §4.3) keeps the Gemma 3 SFT path untouched.

- ✅ **OQ-2 RESOLVED** — *Gemma license.* User confirms the Gemma license for
  `google/functiongemma-270m-it` has been accepted. **Failure handling:** if
  `hf download google/functiongemma-270m-it` returns 401/403, Phase A0/A
  stops and emits these exact commands for the user to run interactively:
  ```bash
  hf auth login                              # paste HF token (read scope OK)
  hf auth whoami                             # verify token recognized
  hf download google/functiongemma-270m-it   # retry
  ```

- ✅ **OQ-3 RESOLVED** — *Distil viability for FG-270M tool-calling.* **Blocked
  by vendor compatibility table.** Distil's
  [`model-catalog.md` Task Compatibility](../../references/upstream/distil-cli-skill/references/model-catalog.md)
  restricts both `tool-calling-closed-book` and `multi-turn-tool-calling-closed-book`
  to **Qwen3 and Llama 3-family students only** — FunctionGemma 270M is in the
  catalog but **excluded**. Distil also does not surface the synthesized
  training corpus (only test-set predictions per
  `references/tasks/retrieve-predictions.md`), ruling out the
  "use Distil only as a data generator" workaround. Distil CLI is installed
  + logged in (`lanhp@uci.edu`) and ready for any future Qwen3/Llama3 student
  variant; until then, no Distil commands are run. **Dataset path now is
  hand-authored + Pro Perplexity / Claude / ChatGPT augmentation** — see §9.4.

- ✅ **OQ-4 RESOLVED** — *Tool-set scope.* **Read-only tools only** for the
  initial implementation. The §9.2 registry stays read-only. Mutating tools
  would require a separate write-tool taxonomy + safety story (consent prompts,
  audit log) and is explicitly out of scope.

- ✅ **OQ-5 RESOLVED** — *Real-PHI escalation gate.* **Synthetic-only forever
  in this repo.** No real-PHI path is required. The `data/health_table_v1.yaml`
  fixture (and any expansion) stays synthetic. A pre-commit hook (proposed in
  §9.5) regex-blocks likely real-PHI patterns.

- ✅ **OQ-6 RESOLVED** — *Quantization target + SyNAP investigation.*
  - **SyNAP is not for LLMs.** Vendor working-with-models doc lists only
    TFLite / ONNX / TorchScript / TF / Caffe inputs (vision toolkit).
    Confirmed via WebFetch 2026-04-29.
  - **llama.cpp is the practical near-term path** — also matches Distil's own
    `references/tasks/deployment-integration.md` recommendation
    ("Option 1: Distil CLI with llama-cpp (Recommended)").
  - **Quantization target switched to Q4_K_M** (was Q4_0). For tool-calling
    where small numerical errors corrupt JSON arguments, K-quants are the
    safer default. Unsloth's built-in `save_pretrained_gguf` produces Q8_0 /
    F16 / BF16 directly (Q4_K_M support upcoming); we run vendor
    `llama-quantize` on the F16 to land Q4_K_M for SL2619. Q4_0 only as a
    fallback if SL2619 inference is too slow on Q4_K_M.

- ✅ **OQ-7 RESOLVED** — *LoRA vs full SFT vs Unsloth.* **Unsloth + LoRA is the
  default.** Per user 2026-04-29 ("if using unsloth gives better performance —
  we should use it"), Phase D adopts the Unsloth notebook recipe end-to-end:
  `FastLanguageModel.from_pretrained` + LoRA `r=128` + `train_on_responses_only`
  + `save_pretrained_merged` / `save_pretrained_gguf`. Unsloth's gradient
  checkpointing lifts the 16 GiB VRAM ceiling enough to run PDB=4, seq=4096
  (vs the vanilla path's PDB=1, seq=1024). Vanilla TRL+PEFT recipe preserved
  in §10.6 as fallback only. Full SFT on a leased A100 remains the terminal
  escalation if Unsloth+LoRA underperforms on G_EVAL.

- ✅ **OQ-8 RESOLVED** — *Submodule promotion for `cookbook/`,
  `distil-cli-skill/`, and `unsloth-notebooks/`.* **Yes, promote — but in a
  separate small PR**, not bundled with this plan revision. All three clones
  currently have their own `.git` dirs as untracked working-tree clones;
  promotion is metadata surgery (remove `.git`, add via
  `git submodule add -b main --depth 1`, pin to current SHA, update
  `.gitmodules` per
  [`docs/references/README.md` §Submodules](../../references/README.md)).
  **`unsloth-notebooks/` is a special case** — it's already a sparse-checkout
  (just `nb/FunctionGemma_(270M).ipynb` + `README.md` + `LICENSE`); the
  submodule version must preserve the same sparse rules to avoid pulling 100+
  MB of unrelated notebooks.
  **Exact promotion sequence (run in a fresh PR after Phase A0 lands):**

  ```bash
  # Pre-checks
  git status --short docs/references/upstream/cookbook docs/references/upstream/distil-cli-skill docs/references/upstream/unsloth-notebooks
  git -C docs/references/upstream/cookbook rev-parse HEAD
  git -C docs/references/upstream/distil-cli-skill rev-parse HEAD
  git -C docs/references/upstream/unsloth-notebooks rev-parse HEAD

  # Cookbook → submodule (re-pin to HEAD 65dfbcf0d1f1f8af6824fc1601a7aef4473dbf1e)
  rm -rf docs/references/upstream/cookbook
  git submodule add -b main --depth 1 \
    https://github.com/google-gemma/cookbook.git \
    docs/references/upstream/cookbook
  git -C docs/references/upstream/cookbook fetch --depth 50 origin 65dfbcf0d1f1f8af6824fc1601a7aef4473dbf1e
  git -C docs/references/upstream/cookbook checkout 65dfbcf0d1f1f8af6824fc1601a7aef4473dbf1e

  # Distil CLI skill → submodule (re-pin to HEAD 566eb9e588f5c9a244fff6e2ddb956b1e4d92e0d)
  rm -rf docs/references/upstream/distil-cli-skill
  git submodule add -b main --depth 1 \
    https://github.com/distil-labs/distil-cli-skill.git \
    docs/references/upstream/distil-cli-skill
  git -C docs/references/upstream/distil-cli-skill fetch --depth 50 origin 566eb9e588f5c9a244fff6e2ddb956b1e4d92e0d
  git -C docs/references/upstream/distil-cli-skill checkout 566eb9e588f5c9a244fff6e2ddb956b1e4d92e0d

  # Unsloth notebooks → submodule (sparse, re-pin to HEAD fc876d51b973c1bf0058ed85e47602cfb4bac185)
  rm -rf docs/references/upstream/unsloth-notebooks
  git submodule add -b main --depth 1 \
    https://github.com/unslothai/notebooks.git \
    docs/references/upstream/unsloth-notebooks
  git -C docs/references/upstream/unsloth-notebooks sparse-checkout init --no-cone
  git -C docs/references/upstream/unsloth-notebooks sparse-checkout set --no-cone \
    '/nb/FunctionGemma_(270M).ipynb' '/README.md' '/LICENSE'
  git -C docs/references/upstream/unsloth-notebooks fetch --depth 50 origin fc876d51b973c1bf0058ed85e47602cfb4bac185
  git -C docs/references/upstream/unsloth-notebooks checkout fc876d51b973c1bf0058ed85e47602cfb4bac185

  # Mirror existing pattern in .gitmodules: append `update = none` and `shallow = true`
  # (unsloth-notebooks also needs `sparseCheckout = /nb/FunctionGemma_(270M).ipynb /README.md /LICENSE`
  #  — git supports per-submodule sparse via `git config -f .gitmodules submodule.<name>.sparseCheckoutCone false`
  #  + a recursive init that reapplies the sparse rules.)
  $EDITOR .gitmodules

  # Commit
  git add .gitmodules docs/references/upstream/cookbook docs/references/upstream/distil-cli-skill docs/references/upstream/unsloth-notebooks
  git commit -m "deps: promote cookbook, distil-cli-skill, unsloth-notebooks to opt-in submodules"
  ```

  Until the promotion PR lands, all three clones remain as untracked
  working-tree clones with the SHAs pinned in §4.2.

- ✅ **OQ-9 RESOLVED — Authorized for Phase A0.** *GGUF convert + tokenizer
  round-trip risk.* Pre-flight is in scope. Spec only — see §16 for the
  exact command sequence. **Will NOT run** until: (a) host-probe checks pass
  (disk ≥ 5 GiB, RAM ≥ 8 GiB free, `convert_hf_to_gguf.py` and `llama-cli`
  available); (b) user approves the host-probe summary; (c) the FG model is
  fetched (Phase A precondition).

### Risk table

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | `convert_hf_to_gguf.py` doesn't recognize FG's new control tokens, GGUF emits `<unk>` for `<start_function_call>` etc. | Medium | M1.5/M7 blocked; on-device path stalls | M1.5 GGUF pre-flight (§15) is the gate — convert base FG-270M to GGUF on host before SFT, smoke with `llama-cli --jinja`. If fails: rerun `convert_hf_to_gguf_update.py` per `docs/references/llama-cpp.md` §"Tokenizer pre-tokenizer hash mismatch". |
| **R2** | Unsloth `r=128` LoRA underperforms vendor full SFT for tool-calling. | Low–Medium | G_EVAL fails; need leased A100. | Notebook recipe is vendor-validated; if it fails, escalate to full SFT (no LoRA, lr=5e-5) on the same RTX 5080 first, then leased A100. Document in §10.2. |
| **R2b** | Unsloth itself misbehaves on RTX 5080 / cu128 (kernel issue, triton incompat). | Low | Phase D blocked until rolled back. | §10.1 step (4) rollback restores pre-Unsloth pins; switch to §10.6 vanilla TRL+PEFT fallback. |
| **R3** | LLM-augmented seed expansion (Pro Perplexity / Claude / ChatGPT) yields conversations that violate the tool-call schema or invent tool names. | Low–Medium | Wasted training samples; G_DATASET_SHAPE drops. | §9.6 quarantine path; user manually reviews the first 50 augmented rows before scaling expansion. |
| **R4** | Shared `~/sl2619-finetune/.venv` upgrade for FG breaks the existing Gemma 3 SFT path. | Low–Medium | Two paths regress simultaneously. | §10.1 mandatory pin-file capture (`pip freeze > .torch-pin-pre-fg-2026-04-29.txt`) BEFORE any pip install + documented rollback. If FG requires a `transformers` minor bump: switch to Option B isolated venv. |
| **R5** | `hf download google/functiongemma-270m-it` returns 401/403 despite license click-through (token scope, expired token, etc.). | Low | M1/M1.5/M2 blocked. | OQ-2 explicit failure handling — emit `hf auth login` + `hf auth whoami` for the user to run. |
| **R6** | Server SFT (M5) eval-loss does not decrease, or G_EVAL underperforms after 3 epochs. | Medium | M6 blocked; need to expand dataset (M4.5 v2) or revisit hyperparameters / LoRA target modules. | (a) Cheap iteration first — return to M4.5 and grow `dataset_v1` by another 200–500 LLM-augmented rows; (b) inspect train/eval loss curves on TensorBoard for under- vs over-fit signal; (c) increase LoRA `r` 16→32 or expand `target_modules` if convergence is slow; (d) escalation: full SFT on the server with vendor cookbook hyperparameters (PDB=4/seq=512/no LoRA — same RTX 5080 fits this for 270M); (e) terminal escalation: leased A100 + full SFT to reproduce vendor Mobile-Actions 58→85% delta. |

---

## 14. Milestones with Concrete Acceptance (revised 2026-04-29)

**Critical path is M0 → M1 → M1.5 → M2 → M3 → M4 → M4.5 → M5 → M6 → M7.**
Host owns M1–M4.5 + M7; server owns M5/M6.

| # | Where | Milestone | Acceptance | Status |
|---|---|---|---|---|
| **M0** | — | Plan reviewed + OQs resolved | This file approved by user; OQ-1…OQ-9 marked RESOLVED in §13. | ✅ DONE 2026-04-29 |
| **M1** | host | Repo extras land | `pyproject.toml` has `[functiongemma]` extra (incl. `llama-cpp-python>=0.3` — see §15.6); `uv sync --extra functiongemma` succeeds on host; `tests/test_functiongemma_imports.py` 10/10 green. | ✅ DONE 2026-04-30 |
| **M1.5** | host | **GGUF pre-flight (OQ-9)** | Host probe passed; base FG-270M downloaded; `convert_hf_to_gguf.py` produced `fg-bf16.gguf`; `llama-quantize` produced `fg-q4_k_m.gguf` (Q4_K_M, vendor-aligned per OQ-6); G_FG_GGUF_PREFLIGHT green via `llama-cpp-python` (Path A) and `llama-cli -st --no-jinja` with pre-rendered prompt (Path B). See §15.6 for the two upstream `llama-cli` bugs that necessitated the workaround. | ✅ DONE 2026-04-30 |
| **M2** | host CPU | Phase A smoke green | `scripts/functiongemma_smoke.py --query "What's the temp in London?"` prints the parsed call within 90 s on host CPU. G_FG_LOAD + G_FG_SINGLE green. | OPEN |
| **M3** | host | Tool registry + tests | `src/gemma_tools/functiongemma_tools.py` with ≥ 6 tools; `uv run pytest tests/test_functiongemma_tools.py` green; ≥ 90 % branch coverage. G_TOOLS_TESTS green. | OPEN |
| **M4** | host | Seed dataset (cookbook recipe) | ~50 hand seeds in `data/functiongemma/seed_conversations.jsonl` (HF chat-template format); Pydantic validator passes ≥ 95 % on hand seeds. | OPEN |
| **M4.5** | host | LLM-augmented expansion | User-driven expansion (Pro Perplexity / Claude / ChatGPT) → `data/functiongemma/llm_expanded_v1.jsonl` with ≥ 300 rows; validator passes ≥ 80 %. **G_DATASET_SHAPE green.** | OPEN |
| **M5** | server (RTX 5080) | **Server LoRA SFT** | Pin file captured pre-install (§10.1); `scripts/finetune_functiongemma.py` runs end-to-end on `nouslogic-server`; eval-loss strictly monotone over ≥ 3 epochs; trainable params ≤ 5 %; wall ≤ 60 min; no OOM; merged adapter saved at `~/functiongemma-finetune/merged_fg_v1/`. **G_TRAIN green.** | OPEN — first server milestone |
| **M6** | server + host | Behavioral eval | 60-prompt held-out eval; FT'd model ≥ 80 % tool-call equivalence vs gold trace; baseline FG < 30 %; bench file at `docs/bench/<date>_functiongemma-eval.md`. **G_EVAL green.** | OPEN |
| **M7** | host | GGUF round-trip on FT'd model | `merged_fg_v1.q4_0.gguf` round-trips a tool call with `llama-cli --jinja` on host; behavior matches HF BF16 within tolerance. **G_GGUF green.** Reuses M1.5 conversion pattern on the FT'd merged model. | OPEN — final milestone of this plan |

Phase E (on-device deploy to SL2619) is intentionally not a milestone of this plan — it gets its own plan after M7 lands.

---

## 15. M1.5 — GGUF pre-flight spec (OQ-9 authorized; do NOT run without user approval)

### 15.1 Purpose

Convert base `google/functiongemma-270m-it` to GGUF on the host and verify
that FunctionGemma's four new control-token pairs (`<start_function_*>`,
`<escape>`) survive the conversion + `llama-cli --jinja` round-trip. This is
cheap (< 5 min wall, < 2 GiB RAM peak) and must pass before any FT'd model is
worth quantizing. Prior Gemma 3 270M conversion already needed a workaround
for the `len(vocab) > vocab_size` mismatch (see
[`models/gemma-3-270m-it/README.md` §8.5.7](../../../models/gemma-3-270m-it/README.md));
FG's tokenizer adds the four control-token pairs **plus new PAD/BOS/EOS** per
the HF model card, so the equivalent regression check is non-optional.

### 15.2 Host probe (READ-ONLY) — first

Run this to verify the host can do the conversion at all. **Stop and emit
remediation commands if any check fails.**

```bash
# (1) Disk + RAM headroom
df -h "$HOME" | tail -1                                  # need ≥ 5 GiB free
free -h | grep Mem                                       # need ≥ 8 GiB available

# (2) llama.cpp submodule + build artifacts
ls docs/references/upstream/llama.cpp/convert_hf_to_gguf.py
ls docs/references/upstream/llama.cpp/build/bin/llama-cli       2>/dev/null \
  || ls docs/references/upstream/llama.cpp/build/bin/llama-quantize 2>/dev/null \
  || echo "NEED BUILD: cd docs/references/upstream/llama.cpp && cmake -B build && cmake --build build -j"

# (3) HF auth (license already accepted per OQ-2; verify token recognized)
hf auth whoami                                           # must NOT 401

# (4) Python deps for convert_hf_to_gguf.py (sentencepiece, gguf, transformers)
python -c "import sentencepiece, gguf, transformers; print('deps OK')"
```

If any step fails, **stop**, surface the exact failing command, and present
remediation to the user. Do not proceed to §15.3.

### 15.3 Conversion (run only after §15.2 fully passes + user approves)

```bash
# (1) Fetch base FG-270M weights (~540 MB)
hf download google/functiongemma-270m-it \
  --local-dir ~/hf-cache/functiongemma-270m-it

# (2) Convert to BF16 GGUF
python docs/references/upstream/llama.cpp/convert_hf_to_gguf.py \
  ~/hf-cache/functiongemma-270m-it \
  --outfile ~/hf-cache/functiongemma-270m-it/fg-bf16.gguf \
  --outtype bf16

# (3) Quantize to Q4_K_M (vendor-aligned for tool-calling fidelity, per OQ-6)
docs/references/upstream/llama.cpp/build/bin/llama-quantize \
  ~/hf-cache/functiongemma-270m-it/fg-bf16.gguf \
  ~/hf-cache/functiongemma-270m-it/fg-q4_k_m.gguf \
  Q4_K_M

# (3b, optional reference) Q8_0 — what Unsloth's save_pretrained_gguf produces
# natively. Useful as the eval ceiling against the FT'd model later.
docs/references/upstream/llama.cpp/build/bin/llama-quantize \
  ~/hf-cache/functiongemma-270m-it/fg-bf16.gguf \
  ~/hf-cache/functiongemma-270m-it/fg-q8_0.gguf \
  Q8_0
```

### 15.4 Smoke test (the actual gate G_FG_GGUF_PREFLIGHT)

> **Status (2026-04-30): the originally specified `llama-cli --jinja -p '...'` invocation does NOT
> exercise tools** — see §15.6 for the root cause and the two validated working paths.

**Pass criterion (G_FG_GGUF_PREFLIGHT):** the output contains a
`<start_function_call>...<end_function_call>` block (parsable by the
canonical regex from §6.2). No `<unk>` tokens in the output.

#### Path A — `llama-cpp-python` (in-tree, default)

This is what the future `scripts/functiongemma_smoke.py` (M2) will use; it
renders the prompt host-side via `transformers.AutoTokenizer.apply_chat_template(tools=...)`
and feeds it programmatically. Reference smoke is at `/tmp/fg_smoke.py`
(captured in §15.6 working notes — to be promoted to `scripts/` in M2).

#### Path B — `llama-cli` with pre-rendered prompt (interactive debugging)

```bash
# 1) Pre-render the prompt host-side with tools= (skips llama-cli's empty-tools jinja path)
uv run python -c '
from pathlib import Path
from transformers import AutoTokenizer
TOOLS=[{"type":"function","function":{"name":"get_current_temperature",
  "description":"Get the current temperature for a given location.",
  "parameters":{"type":"object","properties":{"location":{"type":"string","description":"City name."}},"required":["location"]}}}]
tok=AutoTokenizer.from_pretrained(str(Path.home()/"hf-cache/functiongemma-270m-it"))
Path("/tmp/fg_prerendered.txt").write_text(
  tok.apply_chat_template([{"role":"user","content":"What is the temperature in London?"}],
                          tools=TOOLS, tokenize=False, add_generation_prompt=True))'

# 2) Single-turn run with --no-jinja (raw prompt, parse_special=true on tokenization)
docs/references/upstream/llama.cpp/build/bin/llama-cli \
  -m ~/hf-cache/functiongemma-270m-it/fg-q4_k_m.gguf \
  -f /tmp/fg_prerendered.txt \
  -st --no-jinja -n 96 -t 8 --temp 0.0 --top-p 1.0 --no-warmup </dev/null
```

Validated 2026-04-30: emits `<start_function_call>call get_current_temperature{location:<escape>London<escape>}<end_function_call>`,
exits 0, ~104 t/s gen, ~643 t/s prompt-eval. **Do not use `-no-cnv` / `--no-conversation` — see §15.6.**

### 15.6 Lessons learned (M1 + M1.5 — 2026-04-30)

#### Status

| Milestone | Gate | Result |
|---|---|---|
| **M1** | `functiongemma` extra resolves; `uv run pytest` green | **PASS** — 10/10 tests pass; extra adds transformers, accelerate, torch, huggingface-hub, pydantic, jsonschema, sentencepiece, gguf, **`llama-cpp-python>=0.3`** (added 2026-04-30 after §15.4 needed a programmatic path). |
| **M1.5 §15.2** | host probe (RAM, llama.cpp build, HF auth, deps) | **PASS** — only blocker was that llama.cpp binaries weren't built; resolved in §15.3. |
| **M1.5 §15.3** | `convert_hf_to_gguf.py` + `Q4_K_M` quantize | **PASS** — `~/hf-cache/functiongemma-270m-it/fg-q4_k_m.gguf` (253 MB). |
| **M1.5 §15.4** | `<start_function_call>...<end_function_call>` parsable, no `<unk>` | **PASS** via Path A (llama-cpp-python). Path B (llama-cli pre-rendered) also passes. |

#### Root cause: why the `llama-cli --jinja -p` recipe in the original §15.4 does not work

Two unrelated bugs in the llama.cpp submodule at pin `d775992` (build `b1-d775992`, tag `b8981`):

**Bug A — tools never reach the chat template.** `tools/cli/cli.cpp:210`:

```cpp
common_chat_templates_inputs inputs;
inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
inputs.tools                 = {}; // TODO          ← hardcoded empty
inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
```

`format_chat()` discards any tools — there is no CLI flag to pass them, and
`--chat-template-kwargs` writes to `params.default_template_kwargs` which is
also never propagated to `inputs.chat_template_kwargs` in this function. The
`tools=` jinja variable is therefore always undefined when `--jinja` runs
under `llama-cli`. `llama-server` does not have this gap (`tools` flows from
the `/v1/chat/completions` body through `tools/server/server-common.cpp:1056`).

**Bug B — `--no-conversation` is silently ignored, then leaks the REPL prompt.** `tools/cli/cli.cpp:357-360`:

```cpp
if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
    console::error("--no-conversation is not supported by llama-cli\n");
    console::error("please use llama-completion instead\n");
}
// no return; falls through into interactive while(true) loop at line 469
```

After printing the error, execution enters the interactive loop. With stdin
closed (e.g. `</dev/null`), `console::readline` returns false, `buffer` is
empty, `if (buffer.empty()) continue;` jumps back to the top, `console::log("\n> ")`
fires again — infinite tight loop emitting `> ` characters. This is what
produced the multi-GB log files in the prior session's attempts.

#### The proper exit flag is `-st` (single-turn)

`common/arg.cpp:1513-1521` and the `if (params.single_turn) break;` at
`tools/cli/cli.cpp:636-638` give a clean exit after one generation. Use `-st`,
**never** `-no-cnv` / `--no-conversation`, with this submodule pin.

#### Why we added `llama-cpp-python` to the `functiongemma` extra

- Path A (programmatic) is the only route that exercises the FunctionGemma
  wire format end-to-end without a server detour, and it's what the M2
  Phase A smoke (`scripts/functiongemma_smoke.py`) will import.
- Path B (pre-rendered + `-st --no-jinja`) is useful for interactive debugging
  but requires the user to render the prompt out-of-band; it's not
  CI-friendly.
- `llama-server` + OpenAI compat is a third option (real `tools=` flow) but
  needs an HTTP harness — out of M1.5 scope. Promoted to M-future.

#### Upstream contribution policy

The two cli.cpp bugs are real and worth filing upstream, but **we do not
patch the submodule** — the submodule is upstream-pinned, and local edits
disappear on `git submodule update`. Track upstream fix candidates as OQ-10
(future).

### 15.5 Failure handling

If §15.4 emits `<unk>` for the control tokens, follow the Gemma 3 §8.5.7
recipe: re-run `convert_hf_to_gguf_update.py` against the FG tokenizer to
register a new pre-tokenizer hash, then re-run §15.3. If that fails too,
upstream-issue territory; M1.5 is at risk and M7 (FT'd GGUF round-trip) is blocked.
Treat as risk R1.

---

## 16. References

### Primary

- HF model card: <https://huggingface.co/google/functiongemma-270m-it>
- Google AI overview: <https://ai.google.dev/gemma/docs/functiongemma>
- Formatting + best practices: <https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices>
- Multi-turn sequence: <https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma>
- Function calling with HF: <https://ai.google.dev/gemma/docs/functiongemma/function-calling-with-hf>
- Fine-tuning with FG: <https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma>
- Distil blog: <https://www.distillabs.ai/blog/making-functiongemma-work-multi-turn-tool-calling-at-270m-parameters/>
- Synaptics SyNAP doc (deferred E): <https://synaptics-synap.github.io/doc/v/latest/docs/manual/index.html>

### Local

- `docs/references/upstream/unsloth-notebooks/nb/FunctionGemma_(270M).ipynb` — **Phase D standard procedure** (Unsloth recipe, vendor-blessed)
- `docs/references/upstream/cookbook/docs/functiongemma/finetuning-with-functiongemma.ipynb` — vendor SFT recipe (vanilla TRL+PEFT, fallback per §10.6)
- `docs/references/upstream/cookbook/docs/functiongemma/full-function-calling-sequence-with-functiongemma.ipynb` — multi-turn loop
- `docs/references/upstream/cookbook/.archive/FunctionGemma/[FunctionGemma]Finetune_FunctionGemma_270M_for_Mobile_Actions_with_Hugging_Face.ipynb` — Mobile-Actions worked example
- `docs/references/upstream/distil-cli-skill/SKILL.md` — Distil CLI usage spec
- `docs/references/gemma.md` §FunctionGemma — vendor-source pointer
- `docs/references/transformers-trl-peft.md` — TRL + PEFT API surface (already used by `scripts/finetune.py`)
- `docs/references/llama-cpp.md` — GGUF + `--jinja` deploy notes
- `models/gemma-3-270m-it/README.md` — base-model fingerprint (FG shares the architecture)
- `docs/plans/gemma3-270M/models-testing-plan.md` — adjacent path (closed-world QA, retrieval-style)

---

*Authored 2026-04-29. This plan is the planning record only — every Phase A–D
step requires explicit user authorization before execution.*
