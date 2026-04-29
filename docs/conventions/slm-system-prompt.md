# SLM System-Prompt Convention (Edge Deployment)

> Normative rules for crafting system prompts when the target is a ≤ 500M-parameter SLM running on-device — specifically Gemma 3 270M-IT (and FunctionGemma when adopted). Where larger-model prompt-engineering practice differs from SLM-edge practice, this file picks the SLM-edge side with reasons.
>
> **Canonical ownership** (per `doc-update.md §8.1`): this file owns the *style* of on-device SLM prompts and the *per-model system-prompt template*. The health data schema it slots into lives in `data/health_table.yaml` and is loaded via `src/gemma_tools/health_table.py`. Bench evaluation of how well a given prompt performs is recorded in `docs/bench/<date>_*.md` (frozen historical records).
>
> *Authored 2026-04-24.*

---

## 1. Why SLMs need a different prompt style from LLMs

Advice written for GPT-4-class models is actively harmful on ≤500M-param SLMs. The three drivers:

1. **Every system-prompt token costs TTFT on every turn.** At SL2619's ~1.7 tok/s sustained Gemma 3 decode and ~50–100 tok/s prefill, a 500-token verbose system prompt is ~5–10 s of pre-user-token latency. A 100-token directive prompt is ~1–2 s. For interactive use this is the difference between usable and unusable.

2. **Weak instruction-following.** Gemma 3 270M-IT scores **IFEval 51.2** ([HF model card](https://huggingface.co/google/gemma-3-270m-it), [Google blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)) — strong for its size but still fails ~49% of verifiable-instruction cases. The model will ignore polite-prose requests it would have followed if labeled as hard directives. Labels compensate for what attention-at-270M can't enforce.

3. **Narrow context depth.** 270M–360M parameter models have fewer attention heads and less "working memory" than 7B+ models. Long prose dilutes salience; `ROLE: X` + `TASK: Y` + `RULES: …` keeps salience high.

## 2. Gemma 3's chat template (source-verified)

Gemma 3 has no `system` role. Confirmed from two primary sources:

1. [ai.google.dev/gemma/docs/core/prompt-structure](https://ai.google.dev/gemma/docs/core/prompt-structure) — "Gemma's instruction-tuned models are designed to work with only two roles: `user` and `model`. Therefore, the `system` role or a system turn is not supported."
2. The Jinja template embedded in the model's HF tokenizer silently **concatenates** a `system` message into the first user turn's prefix rather than emitting a separate system block. If you pass `messages[0]['role'] == 'system'`, its content is prepended (with `\n\n` separator) to the first user message, and `messages[1:]` becomes the loop. Verify against the model card's `chat_template.jinja` on HuggingFace (link in §8) or the local copy under `docs/references/upstream/` once the submodule is initialized.

### 2.1 Resulting on-wire format

```
<bos><start_of_turn>user
{system_instructions}

{user_query}<end_of_turn>
<start_of_turn>model
```

Literal tokens to memorize:

| Token | Usage |
|---|---|
| `<bos>` | Beginning-of-sequence. Jinja adds via `{{ bos_token }}` at template start. |
| `<start_of_turn>` | Opens a turn. ID 105. |
| `<end_of_turn>` | Closes a turn. ID 106. **Second EOS** — stop-condition must check `[1, 106]`, not just 1. (`generation_config.json:4-7`) |
| `<eos>` | Classic end-of-sequence. ID 1. |
| `user` / `model` | Role labels (assistant → model in output; template translates). |

### 2.2 Our runtime wrapper

`src/gemma_tools/prompt_composer.py` already emits this exact shape:

```python
return f"<start_of_turn>user\n{full}<end_of_turn>\n<start_of_turn>model\n"
```

Where `full = system_prompt + "\n" + utterance`. This matches the Jinja template's behavior for `messages = [{"role": "system", …}, {"role": "user", …}]`. The composer is correct; do not refactor to introduce a `system` role.

## 3. The SLM prompt style rules (normative)

The following rules apply to any system prompt we ship for an on-board SLM. Each is a one-sentence rule + a short rationale tied to the drivers in §1.

### R-1. Use directive labels, not prose

```
ROLE: <one-phrase role>
TASK: <one-sentence imperative>
RULES:
- <rule 1>
- <rule 2>
FORMAT: <one-sentence output shape>
INPUT: <name of the data block that follows>
```

*Rationale*: matches the instruction-tuning signal distribution; every line is one unit the attention head can bind to. Prose ("You are a helpful assistant who carefully…") has no anchor points, and small models dissolve it into a fuzzy prior rather than a set of rules.

### R-2. Ground all facts in retrieved context; forbid invention

The prompt MUST include an explicit "answer only from the data provided below" directive AND a fallback string for missing data:

```
RULES:
- answer ONLY from {INPUT_NAME}.
- if {INPUT_NAME} lacks the answer: reply "not in record".
- never invent values, dates, names, or numbers.
```

*Rationale*: Phase B 2026-04-24 bench showed Gemma 3 270M hallucinating vital values not in the YAML (`P5 summarize my current health status` → fabricated SpO2 reading). Without the explicit "only from X" + "never invent" phrasing, small-model autoregressive priors fill unknowns with plausible-looking numbers.

### R-3. Specify refusal strings for off-topic queries

```
RULES:
- refuse off-topic / social chat: reply "I answer questions from your {DOMAIN} only".
- never give medical/legal/financial advice; re-route: reply "consult a {PROFESSIONAL}".
```

*Rationale*: unconstrained, the model will attempt to answer anything — often poorly (P3 `make me laugh` → refused with a vague disclaimer that scored 0/3). Giving it pre-authored refusal strings means the refusal is deterministic and doesn't consume generation budget.

### R-4. Cap output length

```
FORMAT: 1-2 sentences. No lists unless {INPUT_NAME} has them. No preamble.
```

*Rationale*: at ~1.7 tok/s, an unconstrained 200-token answer is 2 minutes of decode. 1-2 sentences is ~30-50 tokens = ~20-30 s, usable. Also prevents the "I'll give you a helpful preamble first" pattern which wastes tokens.

### R-5. Prefer positive imperatives over negations

Good: `answer ONLY from YAML`. Bad: `do not answer from anything other than YAML`.

*Rationale*: published SLM prompt-engineering work ([Lakera 2026 guide](https://www.lakera.ai/blog/prompt-engineering-guide), [buildmvpfast](https://www.buildmvpfast.com/blog/system-prompt-design-best-practices-llm-instructions-engineering-2026)) consistently shows positive targets outperform negations in small models; the attention pattern binds to the target, not to the negation operator. Use negations only for the short "never invent" refrain where the verb is its own hook.

### R-6. Inject dynamic context (date, YAML) at a fixed slot, not interpolated through prose

```
DATE: {date_iso}
{INPUT_NAME}:
{yaml_block}
```

*Rationale*: deterministic position means the model learns (even zero-shot, via its instruction-tuning priors) exactly where to look. Interpolating date + YAML into natural sentences doubles prompt length with zero quality gain.

### R-7. No persona / "friendly tone" adjectives

Do NOT write: `You are a friendly, empathetic health assistant who cares deeply about the patient's well-being and always responds with warmth…`

Prefer: `ROLE: health records assistant.`

*Rationale*: the 270M model spends parameters on parsing the persona, not on the actual task. Polite tone emerges from the RULES and FORMAT; a persona adjective is pure overhead.

### R-8. No few-shot examples when the task fits in a directive

Good: `FORMAT: 1-2 sentences. Quote YAML values verbatim.` Bad: a 400-token block showing 3 examples of input/output.

*Rationale*: IFEval 51.2 0-shot is already strong; exemplars cost 400+ prompt tokens of TTFT every turn and provide diminishing returns for narrow tasks. Add few-shot only if bench numbers demand it.

### R-9. English-only, one language per prompt

Gemma 3 270M's safety evaluation was English-only (HF model card). Prompts that mix languages in the system section produce lower-quality output.

### R-10. Prompt length target: ≤ 150 tokens (excluding dynamic YAML)

Measured budget for SL2619's Gemma 3 270M path:

| Slot | Tokens |
|---|---|
| Directives (§3 rules applied) | 80–120 |
| `DATE: 2026-04-24` | ~6 |
| `INPUT: YAML block:` label + delimiters | ~4 |
| **Static system budget** | **~90–130** |
| YAML block | 200–500 (varies by patient) |
| User utterance | 20–50 |
| **Total prompt** | **~310–680** |

The static system part is checked into version control and budgeted; the YAML and user utterance are dynamic.

## 4. Canonical template — Gemma 3 270M-IT + health YAML QA

```
ROLE: health-records assistant on SL2619 edge device.
TASK: answer the user's question using ONLY facts in YAML.
RULES:
- quote YAML values verbatim (numbers, doses, times, names).
- if YAML lacks the answer: reply "not in record".
- never invent values, dates, medications, or food rules.
- refuse off-topic / social chat: reply "I answer questions from your health record only".
- never give medical advice; re-route: "consult your clinician".
FORMAT: 1-2 sentences. No preamble. No lists unless YAML has them.
DATE: {date_iso}
YAML:
{yaml_block}
```

Exact composition is owned by `src/gemma_tools/prompt_composer.py` (`render_system_prompt`). Unit tests in `tests/test_prompt_composer.py` freeze the template surface — changes to the template must land with test updates in the same commit.

### 4.1 Why this is the final shape (post-Phase-B iteration)

- **"health-records assistant"** (not "health assistant"): emphasizes retrieval over advice. Early Phase B prompt used "concise health assistant" — model sometimes slipped into advice mode (scored 0 on rubric). Swapping "records" grounds the role in the YAML.
- **"quote YAML values verbatim"**: directly addresses the P5 hallucination from Phase B. Paraphrasing a value (e.g., "your heart rate is around 70") opens the door to error; quoting forces the model to emit the exact YAML content.
- **Explicit refusal strings inline in RULES**: both "not in record" and "I answer questions from your health record only" are short, tokenize tightly, and give the model pre-approved outputs for the edge cases where §3 R-2 and R-3 apply.
- **`DATE:` on its own line**: the model treats it as a fact to reference (for `when should I take X` questions) rather than background context.
- **`YAML:` label before the block**: the model's instruction-tuning distribution sees labeled-block patterns during training; explicit label > implicit context.

### 4.2 Divergence from other SLM templates

ChatML-style models (e.g. SmolLM2, Qwen) use `<|im_start|>user` / `<|im_end|>` / `<|im_start|>assistant`. The body of the system prompt (`ROLE/TASK/RULES/FORMAT/INPUT`) is identical — the style rules in §3 are model-agnostic. Only the wrapper tokens differ; `prompt_composer.py` dispatches on `candidate` (e.g. `"gemma3"`, future `"functiongemma"`).

## 5. Quality-gate integration

Bench prompts (`data/prompts.yaml`) are stratified into three classes that exercise different parts of the template:

| Class | What it tests | Example ID |
|---|---|---|
| **fact_lookup** | R-2 grounding — does the model retrieve from YAML correctly? | P4, P6, P7 |
| **fact_absence** | R-2 refusal — when YAML doesn't have the answer, does "not in record" fire? | P8 |
| **domain_refusal** | R-3 refusal — when the question is off-topic, does the refusal string fire? | P9, P10 |
| **summarization** | R-2 + R-4 — multi-fact compression without invention | P5 |

G_QUALITY threshold (per `docs/plans/models-testing-plan.md §7`, frozen narrative): average rubric score ≥ 2.0 across all classes, with zero 0-scores in fact_lookup or fact_absence (strict grounding is non-negotiable).

## 6. What this file does NOT cover

- LLM (≥ 7B) prompt engineering — different set of trade-offs; refer to industry guides.
- Fine-tuning prompts / instruction-tuning — different artifact (training data) with different format rules; covered by `docs/guides/finetune-best-practices.md` and the SFT prompt-shape contract in `scripts/finetune.py:_to_prompt_completion`.
- Tool-use / function-calling prompts — not supported by Gemma 3 270M's instruction-tuning (model card is silent on tool-use; IFEval is closer to what the model does).
- Multi-turn dialog state management — explicitly out-of-scope; Gemma 3 270M is "not designed for complex conversational use cases" per Google.
- Safety / jailbreak resistance — a 270M model cannot defend a prompt; lean on pre-filters in Python (regex, classifier) for any safety-critical deployment.

## 7. Checklist (when adding a new bench prompt or template variant)

- [ ] Labels match §3 R-1 shape (`ROLE:`, `TASK:`, `RULES:`, `FORMAT:`, `INPUT:`)
- [ ] Every factual claim is grounded in a named data block (§3 R-2)
- [ ] At least one refusal string covers off-topic (§3 R-3)
- [ ] FORMAT caps output length (§3 R-4)
- [ ] Static system tokens ≤ 150 (§3 R-10)
- [ ] `src/gemma_tools/prompt_composer.py` template still round-trips after the edit; tests pass
- [ ] `data/prompts.yaml` gets coverage in all four classes (fact_lookup / fact_absence / domain_refusal / summarization)
- [ ] If changing on-wire format: confirm against `chat_template.jinja` from the model card on HuggingFace (or the local copy under `docs/references/upstream/` if the submodule is initialized) for whichever candidate

## 8. Sources

Primary:
- [Google Developers blog — Introducing Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
- [Gemma prompt structure — ai.google.dev](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [google/gemma-3-270m-it — HuggingFace model card](https://huggingface.co/google/gemma-3-270m-it)
- [Lakera — Prompt Engineering Guide 2026](https://www.lakera.ai/blog/prompt-engineering-guide)
- [buildmvpfast — System Prompt Design Best Practices 2026](https://www.buildmvpfast.com/blog/system-prompt-design-best-practices-llm-instructions-engineering-2026)

Repo-local:
- `src/gemma_tools/prompt_composer.py` — runtime wrapper (the composer this file governs)
- `models/gemma-3-270m-it/README.md` — per-model best-practice analysis (§5.1 prompt shape lineage)
- `docs/bench/2026-04-24_gemma3-summary.md` — Phase B failure modes that drove §3 R-2/R-3 (frozen)
- `docs/references/gemma.md` — upstream Gemma 3 / FunctionGemma source pointers
- `docs/references/upstream/gemma/` — Gemma reference implementation submodule (opt-in init)
