# Plans

Active forward-looking plans for the FunctionGemma track. The repository's
focus is FunctionGemma 270M-IT (function-calling on the SL2619 board); the
gemma3-270m health-QA work is preserved under
`archive/gemma3-270m-health-qa/`.

## Active plan files

Inside `functiongemma/`:

| File | Purpose |
|---|---|
| `recipe.md` | Working recipe — model identity, wire format, current train/eval paths |
| `decisions-log.md` | Major decisions table with rationale and current status |
| `quantization-plan.md` | INT4/INT8 SL2619 testing plan (current focus, NOT YET EXECUTED) |
| `seed-authoring-recipe.md` | How to author a hand-seed conversation batch |
| `llm-augmentation-prompt.md` | Verbatim prompt for LLM augmentation of seeds |
| `upstream-issue-drafts.md` | Drafts for `--no-conversation`/`-no-cnv` and tools= upstream bugs |

## Archived plans

The Phase A–E historical narrative for FunctionGemma (2321 lines, 2026-04-29
through 2026-05-01) is preserved verbatim at
`archive/functiongemma-pre-distil/plans/phase-d-readme-original.md` for
context and audit.

The gemma3-270m health-QA plan documents are at
`archive/gemma3-270m-health-qa/plans/gemma3-270M/`:
- `a55-gemma-fine-tune.md` — LoRA SFT recipe, Q0–F5 phase plan
- `a55-gemma-h5-logits-equivalence.md` — KL-divergence cross-arch gate
- `models-testing-plan.md` — bench protocol, prompt classes

## Authoring conventions

DRY-exempt per `docs/conventions/doc-update.md §8.2`. Keep active plans concise
and use mermaid for workflows where prose would be wordy. When a plan is
superseded, move it to `archive/` and update this index.
