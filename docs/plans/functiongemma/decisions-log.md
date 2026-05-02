# FunctionGemma — decisions log

Major decisions made during the FunctionGemma SFT effort, with rationale and
current status. Distilled from the 2321-line phase-D plan
(`archive/functiongemma-pre-distil/plans/phase-d-readme-original.md`).

## Decisions

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-04-29 | Use `google/functiongemma-270m-it` rather than fine-tuning Gemma 3 270M-IT for tool calling | FunctionGemma ships with a tool-calling-aware chat template (`<start_function_call>`, `<escape>`, `developer` role). Adding tool calling to Gemma 3 means re-doing the chat template; FG already has it. | DONE |
| 2026-04-29 | Training MUST run on `nouslogic-server` (RTX 5080, cu128) | WSL host has no GPU. Phase D budget is one full SFT in ≤ 60 min — only feasible on the server. | DONE |
| 2026-04-29 | Patient-YAML registry uses 7 read-only tools | Smallest set that exercises each function-calling shape: zero-arg, one-arg, multi-arg, refusal-route. Mirrors a realistic on-device patient-assistant scope. | DONE |
| 2026-04-29 | Synthetic patient fixture only (`data/health_table_v1.yaml`) | OQ-5: no real PHI in repo without external review. Synthetic fixture is auditable, deterministic, dual-use with the gemma3-270m bench. | DONE |
| 2026-04-29 | Dataset: hand-authored seeds + LLM augmentation, NOT Distil-CLI synthesis | Initial reading of Distil's `model-catalog.md` showed FunctionGemma 270M excluded from `multi-turn-tool-calling-closed-book`. Switched to hand+LLM expansion via Pro Perplexity / Claude / ChatGPT following the vendor cookbook recipe. | SUPERSEDED 2026-05-01 |
| 2026-04-29 | Phase D training stack: Unsloth (`FastLanguageModel` + `train_on_responses_only` + `save_pretrained_gguf`) | ~2× speed, 30% VRAM reduction over vanilla TRL/PEFT, FG-aware response-only masking, native GGUF export. Mirrors the upstream `unslothai/notebooks` FunctionGemma_(270M).ipynb recipe. | KEPT AS FALLBACK |
| 2026-04-30 | M1.5 GGUF pre-flight: produce `fg-q4_k_m.gguf` + `--jinja` round-trip | Confirms the on-board chat template renders correctly before committing to a full SFT run. | DONE |
| 2026-04-30 | M3 tool registry covers 7 tools at 99% branch coverage with frozen JSON-Schema mirror at `data/functiongemma/tools_v1.yaml` | Registry is the durable contract — both the model (via tools= in chat template) and the dispatcher (`functiongemma/tools.py`) read the same schema. | DONE |
| 2026-04-30 | Block E supplement: 740-row stagedbatch to repair refusal/parallel-call coverage gaps | Initial 545-row pool had 14 unique IDs in refusal classes; below the floor for SFT generalization. Block E added 370 unique conversation IDs duplicated for class balance. | DONE — see `archive/functiongemma-pre-distil/bench/2026-05-01_functiongemma-block-e-supplement-repair.md` |
| 2026-05-01 | Refusal-class loss reweighting (Block F1) — sweep `weight ∈ {1.5, 2, 3}` × `bug` variants | Refusal precision was the primary failure mode in early iterations. Hypothesis: upweight refusal rows in the loss, push the model away from "always call something." | FAILED — none of 36 variants cleared the all-categories ≥ 80% bar. See `archive/functiongemma-pre-distil/bench/eval-summary.md` |
| 2026-05-01 | Switch primary training path from local Unsloth to Distil Labs | Pre-distil iteration sweep plateau'd at ~70% pass rate; Distil iter-001 hit 0.9583 on every metric in one pass. Distil also handles teacher-distilled synthesis automatically (5,054 conversations from 50 seeds + GPT-oss-120B). | DONE |
| 2026-05-01 | Iteration 001 hyperparameters: LoRA r=64, α=64, dropout 0, target `q_proj,v_proj`, 4 epochs, generation_target 5000, validation_similarity_threshold 0.90 | Inherited from the unslothai/notebooks reference recipe. `validation_similarity_threshold` loosened from default 0.95 to 0.90 to widen scenario coverage. Best checkpoint kept by trainer at epoch 3/4. | DONE |
| 2026-05-01 | Refusal classes (`medical_advice_refusal`, `off_topic_refusal`) and `parallel_call` excluded from Distil training | Distil's `multi-turn-tool-calling-closed-book` enforces "exactly one tool call per assistant turn" — the three excluded classes don't fit. They stay on the local F1+F5 path. | DONE |
| 2026-05-01 | Cross-set duplicate `(Do I have any allergies?, list_allergies())` rejected by Distil — repair seeds | Platform forbids any train/test pair sharing identical (q, a). 3 within-train wasted-slot duplicates also surfaced. Repaired by paraphrasing 4 train rows targeting the same tool. | DONE |
| 2026-05-01 | `task_description` v3 surgical edits — RULE #3 worked-examples + RULE #7 strip-generic-noun | Targeted the two highest-confidence misses: cluster B "yes/no allergy" phrasing + cluster C "A pills" prefix-noun stripping. | DONE — pushed all five eval metrics to 0.9583 |
| 2026-05-02 | Workspace refactor — separate Gemma 3 (archive) from FunctionGemma (active) | Pre-refactor workspace mixed two model tracks across every directory. Post-refactor: `releases/functiongemma-270m/`, `distil/`, `archive/gemma3-270m-health-qa/`. | DONE |
| 2026-05-02 | Quantization sweep is the next focus | Iteration 001 ships only FP16 GGUF (518 MiB). Board path needs INT4/INT8 to clear the latency bar. See `docs/plans/functiongemma/quantization-plan.md`. | IN PROGRESS |

## Resolved open questions (was §13)

| OQ | Question | Resolution |
|---|---|---|
| OQ-1 | Is FunctionGemma a drop-in for the closed-world QA path? | NO — wire format differs. Both paths now coexist: Gemma 3 in `archive/`, FG active. |
| OQ-3 | Does Distil support FG-270M for `multi-turn-tool-calling-closed-book`? | YES (verified empirically 2026-05-01). Catalog snapshot was stale. |
| OQ-5 | Is the `health_table_v1.yaml` fixture safe to use? | YES — synthetic, no real PHI. PHI scanner gates every dataset commit. |
| OQ-9 | Does `--jinja` chat-template rendering survive the GGUF round-trip? | YES (M1.5 — `fg-q4_k_m.gguf` round-trip green via two paths). |
| OQ-10 | Are upstream `llama-cli` `--no-conversation`/`-no-cnv` flags reliable at submodule pin `d775992`? | NO — two reproduction-confirmed bugs. Avoid those flags. Drafts in `docs/plans/functiongemma/upstream-issue-drafts.md`. |

## Open questions (still active)

| OQ | Question | Owner | Why it matters |
|---|---|---|---|
| OQ-Q1 | What quant level minimizes board latency without breaking tool-call accuracy? | quantization-plan | drives the on-device deployment story |
| OQ-Q2 | Is `--jinja` reliable on the on-board llama.cpp build at the latest tag? | quantization-plan | currently we pre-render host-side; on-board jinja would simplify deployment |
| OQ-Q3 | Does an Iteration 002 unlock parallel-call / refusal classes that Distil currently blocks? | post-quant | refusals + parallel calls are excluded from current training; would need a separate path or platform feature |
