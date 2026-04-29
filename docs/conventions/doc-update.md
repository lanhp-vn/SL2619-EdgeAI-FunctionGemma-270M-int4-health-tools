# 13 — Documentation Update Protocol

> How `AGENTS.md` (repo root, agent self-reference) and `README.md` (repo root, human-facing) are refreshed after the codebase evolves. Also hosts the **DRY / single-source-of-truth registry** at §8. This file is the durable reference; it is invoked informally — there is no `/doc_update` skill in this repo.

> **Scope**:
>
> 1. **Routine refresh** of the two root-level docs (`AGENTS.md`, `README.md`) after architectural or workflow changes — §1–§7.
> 2. **DRY / canonical ownership** for all workspace Markdown files — §8–§10.

---

## 1. When This Protocol Fires

Run this protocol when any of the following is true:

1. **New tool or script added** to `src/gemma_tools/` or `scripts/`.
2. **New data path** introduced (new JSONL format, new YAML schema, new bench format).
3. **Workflow changed** (new GPU server setup step, new fine-tune argument, new bench pipeline).
4. **SFT/prompt contract changed** — `finetune.py:_to_prompt_completion` or `prompts.yaml` structure changed.
5. **Before every tagged release** — `AGENTS.md` and `README.md` must be current.

Do NOT run for:

- Routine bug fixes that don't change architecture or workflow.
- Edits inside `docs/conventions/` — those already reference each other.

## 2. Pre-Flight

Before proposing any changes, verify the current state on the host. Do not assume — read.

1. `git log --oneline -20` — surface recent commits that changed architecture.
2. `ls src/gemma_tools/ scripts/ data/` — what exists on disk right now?
3. `cat docs/plans/models-testing-plan.md | head -60` — current plan summary.
4. `uv run pytest --co -q 2>/dev/null | head -30` — which tests exist?

If the GPU server or board is involved, check their state via `/board_probe` in the SynapticSL2619 repo (which also probes the server) or read `docs/bench/` for the latest bench run.

## 3. Context Analysis

### 3.1 Scan codebase state

- **Source files**: `ls src/gemma_tools/` — which modules exist?
- **Scripts**: `ls scripts/` — which entrypoints exist?
- **Data**: `ls data/` — what YAML / JSONL files exist?
- **Tests**: `ls tests/` — which test modules exist?
- **Recent activity**: `git log --oneline -10` and `git status`.

### 3.2 Distinguish current vs. planned

Be careful to distinguish:

- **Current state** — what's on disk right now.
- **Planned state** — what `docs/plans/` calls for but isn't built yet.

Agents reading `AGENTS.md` will confuse the two unless flagged clearly.

## 4. Update `AGENTS.md` (Agent Self-Reference)

**Goal**: a compact document that an agent dropping into this repo for the first time can read to answer "what is this, how do I work here, what rules bind me?"

### 4.1 Required sections

1. **Repository purpose** — one paragraph.
2. **Key paths** — table of important files/dirs with their role.
3. **Workflows** — short command-driven sections: install, test, lint, fine-tune, bench.
4. **Discipline** — active constraints that govern agent behavior (R3, prompt-template atomicity, no weights in git, etc.).

### 4.2 Style

- Target length: **≤ 150 lines**.
- Short paragraphs. Bullets > prose.
- Code fences for commands.
- No emojis.

### 4.3 What to keep out

- Detailed rules — they live in `docs/conventions/`. `AGENTS.md` is an index, not a replacement.
- Changelogs — git history tracks that.

## 5. Update `README.md` (Human-Facing)

**Goal**: a developer who clones this repo can install, understand the data format, run fine-tuning on a GPU server, and evaluate a checkpoint.

### 5.1 Required sections

1. **Project title + tagline** — one paragraph.
2. **What it does** — 2–3 bullets.
3. **Prerequisites** — Python version, `uv`, GPU server requirements.
4. **Install** — `uv sync` steps.
5. **Data** — how `data/` is structured, what the YAML schema looks like.
6. **Fine-tuning** — `scripts/finetune.py` usage.
7. **Evaluation** — `scripts/bench_*.py` usage.
8. **Testing** — `uv run pytest` cheat sheet.
9. **Repo layout** — brief tree.
10. **License** — as applicable.

### 5.2 Style

- Audience: senior engineer new to this repo — not a beginner.
- No agent-discipline rules in README — those belong in `AGENTS.md` and `docs/conventions/`.
- Target length: **≤ 400 lines**.

## 6. Execution Steps

1. **Draft updates** — propose diffs for `AGENTS.md` and `README.md`. Prefer small, targeted edits over full rewrites unless architecture changed materially.
2. **Show the diff** — present changed hunks; summarize why each changed.
3. **Wait for approval** — do not write until the user confirms.
4. **Apply** — write the updated files.
5. **Verify** — `wc -l AGENTS.md README.md` to confirm size didn't balloon.
6. **Commit** — single commit, scope `docs`.

## 7. Failure Modes to Watch

- **Drift without detection**: `AGENTS.md` claims a module exists that doesn't. Fix: `grep -r 'bench_remote\|sft_build\|h5_logits' AGENTS.md README.md` — every path claimed should exist or be marked "planned".
- **Duplicate truth**: rules copied from `docs/conventions/` into `AGENTS.md` start to diverge. `AGENTS.md` holds **summaries only**; any normative text points to the convention file.
- **Overgrowth**: `AGENTS.md` past 150 lines loses its TL;DR value.

---

## 8. DRY / Canonical Ownership Registry

Every normative fact, procedure, spec, or reusable table lives in **one** Markdown file. Other files point to it with `[see X §Y](path)` — never inline-restate. Pointer forms (all acceptable):

- Cross-file link with anchor: `[prompt contract](docs/conventions/slm-system-prompt.md#r-1-closed-world)`
- Section reference: "see `model-compiler-runtime.md` §3 for the REPACK kernel selection"
- Path-only reference: "`docs/bench/` has the frozen bench logs"

### 8.1 Registry by domain

#### Prompt / SFT contract

| Topic | Canonical |
|---|---|
| SLM system prompt rules (R-1 … R-10) | `docs/conventions/slm-system-prompt.md` |
| Training-time prompt shape (`_to_prompt_completion`) | `scripts/finetune.py` (source of truth); all other scripts must replicate exactly |
| Health-QA YAML schema | `data/health_table.yaml` (source); `src/gemma_tools/health_table.py` (Pydantic model) |

#### Model compiler / runtime stacks

| Topic | Canonical |
|---|---|
| GGUF cross-compile + REPACK kernel selection (q4_0_4x4 vs q4_0_8x8) | `docs/conventions/model-compiler-runtime.md` §3 |
| H5R logits-equivalence gate (delta <= 1.0 pp, calibration corpus) | `docs/plans/a55-gemma-h5-logits-equivalence.md` |
| llama.cpp deployment conventions (`/mnt/sdcard/llama-cpp/`) | `docs/conventions/model-compiler-runtime.md` §3.5 |
| On-board tmpfs venv pattern (`/tmp/pipbase` + `/tmp/p15site`) | `docs/conventions/model-compiler-runtime.md` §5 (pointer to SynapticSL2619 `15-model-compiler-runtime.md`) |

#### Fine-tuning pipeline

| Topic | Canonical |
|---|---|
| LoRA / QLoRA SFT recipe (hyperparams, PEFT config, training loop) | `docs/plans/a55-gemma-fine-tune.md` |
| Server bootstrap (CUDA install, venv, finetune deps) | `scripts/server-bootstrap.sh` |
| SFT dataset format (JSONL fields, token budget) | `docs/plans/a55-gemma-fine-tune.md` §4 |

#### Evaluation / bench

| Topic | Canonical |
|---|---|
| Bench protocol (WER gate, F1 gate, run format) | `docs/plans/models-testing-plan.md` §5 |
| Frozen bench run logs | `docs/bench/` (one file per run; never rewrite) |
| Prompt variants used during bench | `docs/plans/models-testing-plan.md` §3 |

#### Code style, testing, git, docs

| Topic | Canonical |
|---|---|
| Python 3.11+ style | `docs/conventions/09-code-style-python.md` |
| Bash style | `docs/conventions/10-code-style-shell.md` |
| Testing pyramid + table-driven idioms | `docs/conventions/11-testing-verification.md` |
| Git workflow (branches, commits, PRs) | `docs/conventions/12-git-workflow.md` |
| Module layering discipline | `docs/conventions/14-module-layering.md` |
| Doc-update protocol + DRY registry (this file) | `docs/conventions/13-documentation-update-protocol.md` |

#### Live state and external repos

| Topic | Canonical |
|---|---|
| GPU server + board live snapshot | `docs/tmp/nouslogic-server-status.md` (regenerated by `/board_probe` in SynapticSL2619) |
| SynapticSL2619 integration (IPC, deploy, board conventions) | `SynapticSL2619/CLAUDE.md` (parent repo) |

### 8.2 Standalone / DRY-exempt documents

The following files carry their own narrative and **are not required to defer to pointers**. They may include summaries or inline restatements where readability demands it.

| Path | Status | Agent policy |
|---|---|---|
| `docs/plans/**/*.md` (fine-tune plan, testing plan, logits-equiv plan) | **Standalone planning docs.** Self-contained narratives: recipes, rationale, concrete run commands. | Agents may update these when the plan itself changes — but do not rewrite them to collapse content into pointers. |
| `docs/bench/**` | **Frozen bench run records.** Never modified after creation. | Read-only. A new run creates a new file; old files are never edited. |

### 8.3 When you add a new topic

1. Decide where the canonical file is. If no obvious home, create one.
2. Add the topic to §8.1 in this file in the **same PR**.
3. Every other file that mentions the topic uses a pointer.

## 9. Allowed Summaries

The only places where a controlled summary of normative content is permitted:

| Location | Content | Constraint |
|---|---|---|
| `AGENTS.md` §Discipline | Compact behavior rules | One line per rule; each implies a pointer to `docs/conventions/` |
| `AGENTS.md` §Workflows | Command-driven cheat sheet | Commands only; rationale in convention files |
| `README.md` capability bullets | Human orientation (tagline, "what it does") | Not normative; no specs |

### 9.1 Disallowed patterns (always)

- Copy-pasted table (same headers, same values) in two files.
- Multi-paragraph restatement of a rule, procedure, or spec.
- Divergent summaries of the same fact.
- The phrase "for convenience we restate…" — smell.

## 10. Checklist (for a doc-update PR)

- [ ] Pre-flight §2 completed.
- [ ] `AGENTS.md` <= 150 lines.
- [ ] `README.md` <= 400 lines.
- [ ] Every claimed file path exists or is marked "planned".
- [ ] New facts/specs added? Registered in §8.1 in this same PR.
- [ ] No inline restatement of content that has a canonical home.
- [ ] Commit scope `docs`.
