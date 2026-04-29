# 12 — Git Workflow

> Branch strategy, commit message format, PR protocol, and release process for this repo.

---

## 1. Branch Strategy

```
main          # Stable, release-ready. Tagged releases cut from here.
  └── develop # Integration branch. Feature branches merge here first.
        ├── feature/<short-name>    # New features
        ├── bugfix/<issue-ref>      # Bug fixes
        ├── hotfix/<issue-ref>      # Urgent fixes (branch from main)
        └── chore/<name>            # Deps, CI, tooling
```

**`main`** is always deployable: a release at any commit on `main` should produce a valid, tested checkpoint.
**`develop`** absorbs merges from feature branches; integration breakage gets caught here before `main`.

## 2. Branch Naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<short-description>` | `feature/h5r-logits-gate` |
| Bug fix | `bugfix/<issue-key-or-description>` | `bugfix/wer-off-by-one` |
| Hotfix | `hotfix/<issue-key-or-description>` | `hotfix/prompt-template-regression` |
| Chore | `chore/<description>` | `chore/bump-transformers-4.42` |
| Release | `release/v<major>.<minor>.<patch>` | `release/v0.2.0` |

Keep names short (< 40 chars) and all-lowercase with kebab separators.

## 3. Commit Messages

Conventional Commits format:

```
type(scope): imperative summary in lower case, no trailing period

Optional body — wrap at 72 cols. Explain **why** and the **context**, not
just **what** (the diff shows what). Note any prompt-template changes that
affect the SFT/inference contract.

Refs: ISSUE-123
```

### 3.1 Types

| Type | Use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructure, no behavior change |
| `perf` | Performance improvement |
| `test` | Test additions or fixes |
| `docs` | Documentation only |
| `chore` | Tooling, deps, CI, non-source |
| `style` | Formatting only |

### 3.2 Scopes

Subsystem-based, not file-based:

| Scope | Refers to |
|---|---|
| `finetune` | `scripts/finetune.py`, LoRA/QLoRA config, training loop |
| `eval` | Evaluation / bench scripts and metrics |
| `bench` | `bench_remote.py`, `bench_prompt.py`, bench data |
| `data` | YAML schema, JSONL builders, `health_table.py`, `sft_dataset.py` |
| `prompt` | `prompt_composer.py`, `prompts.yaml`, `slm-system-prompt.md` |
| `tools` | `src/gemma_tools/` library modules |
| `scripts` | `scripts/` shell and Python entry points |
| `h5r` | `h5_logits_equiv.py`, logits-equivalence gate |
| `deps` | Dependency bumps (`pyproject.toml`, `uv.lock`) |
| `docs` | Documentation under `docs/` |
| `ci` | CI pipelines |

### 3.3 Good vs bad

```
# Good
feat(prompt): add R-9 closed-world refusal to system prompt

Prevents the model from producing open-domain answers when no matching
entry exists in health_table.yaml. Without this, the model hallucinated
medical advice outside its training distribution. Calibration run shows
no regression on existing 47 Q-A pairs (bench/2026-04-28 baseline).

Refs: #12
```

```
# Bad — no scope, tense wrong, no "why"
Update some prompt stuff
Fixed the bug
wip
```

## 4. Prompt-Template Atomicity Rule

**`scripts/finetune.py:_to_prompt_completion` is the single source of truth for the training-time prompt shape.** Any PR that changes it must also update every call site that replicates the format:

- `src/gemma_tools/prompt_composer.py`
- `src/gemma_tools/bench_prompt.py`
- `src/gemma_tools/chat_probe.py`

Divergence between `finetune.py` and inference-time prompt construction creates a tokenization artifact that silently degrades model quality. A single commit must contain all affected files, with a commit body noting what changed in the template and why.

## 5. Pull Request Protocol

### 5.1 Opening a PR

- Base branch: `develop` (normal) or `main` (hotfix only).
- Title format matches commit format: `type(scope): summary`.
- Body uses the template below.

### 5.2 PR template

```markdown
## Summary
<1–3 bullets of what this does and why>

## Changes by domain
- [ ] `src/gemma_tools/` (Python library)
- [ ] `scripts/` (training / eval / deploy scripts)
- [ ] `data/` (YAML schema, JSONL datasets)
- [ ] Prompt template (`finetune.py:_to_prompt_completion`) — **all call sites updated**
- [ ] `docs/conventions/` (normative rules)
- [ ] `docs/plans/` (plan / bench docs)
- [ ] CI / tooling

## Prompt-template contract
If the prompt template changed: what was the old format, what is the new format, why?
(Skip if prompt template unchanged.)

## Testing
- [ ] `uv run pytest -m 'not server and not hardware'` passes locally
- [ ] `shellcheck scripts/*.sh` / `ruff check` / `mypy --strict` clean
- [ ] Server / board integration tests run (describe setup + result, or note deferred)

## Screenshots / logs
<bench output, eval metrics, or relevant log snippets>

Refs: <issue link>
```

### 5.3 Merge strategy

- **Squash merge** from feature branches into `develop`. Keeps `develop` history readable.
- **Merge commits** from `release/*` into `main` so the release branch-point is preserved.
- **Never rebase merged branches** after they've been merged to `develop`/`main`.

## 6. Code Review

### 6.1 Required reviewers

- **Any PR**: at least one reviewer who is not the author.
- **PRs touching the prompt template**: at least one reviewer who can verify the tokenization contract hasn't shifted.

### 6.2 What reviewers check

1. **Prompt-template atomicity** — if `finetune.py:_to_prompt_completion` changed, all call sites are updated in the same commit.
2. **Style** — `ruff format` idempotent; `ruff check` / `mypy --strict` clean.
3. **Tests** — new logic has unit tests; remote integration tests described.
4. **Data hygiene** — no model weights (`.gguf`, `.safetensors`, `.pt`) committed.

## 7. Release Process

### 7.1 Version scheme

Semantic versioning: `v<major>.<minor>.<patch>`.

- **Major**: breaking prompt-template change (old-format checkpoints no longer loadable).
- **Minor**: new feature, backwards-compatible.
- **Patch**: bug fix only.

### 7.2 Release procedure

1. Branch `release/v0.X.Y` from `develop`.
2. Update `VERSION` file (single line, e.g. `0.2.0`).
3. Update `CHANGELOG.md`.
4. Run full test matrix:
   - All host unit tests green.
   - Server integration: one full fine-tune run + eval pass.
   - Board integration: llama-server smoke + H5R gate green.
5. Merge `release/v0.X.Y` into `main` with a merge commit.
6. Tag `v0.X.Y` on `main`.
7. Merge `main` back into `develop` (so tags are visible downstream).

### 7.3 Hotfix procedure

1. Branch `hotfix/<description>` from `main` (not `develop`).
2. Minimal fix + minimal tests.
3. PR into `main`.
4. Tag `v0.X.(Y+1)`.
5. Merge `main` back into `develop`.

## 8. History Rewriting

- **Never `git push --force` to `main` or `develop`**. Hotfixes use revert commits.
- **Feature branches may `git rebase`** before opening a PR — once the PR is open, stop rebasing.
- **`git commit --amend`** is fine on your local branch before pushing. After pushing, prefer a new commit.

## 9. `.gitignore`

Repo-level `.gitignore` covers:

```
# Model weights and checkpoints (never in git — use explicit git add -f for tiny fixtures)
*.gguf
*.safetensors
*.pt
*.bin
*.pth
checkpoints/
lora_weights/

# Build and cache outputs
build/
__pycache__/
*.pyc
*.egg-info/
.venv/
venv/
.mypy_cache/
.ruff_cache/
.pytest_cache/

# IDE / editor
.vscode/
.idea/
*.swp
.DS_Store

# Local overrides
.env
*.local.yaml
```

## 10. Secrets

- **No credentials in the repo, ever** — not in code, not in configs, not in test fixtures.
- SSH keys and HuggingFace tokens live in `~/.ssh/` and `~/.cache/huggingface/token`, not in this repo.
- If a secret is accidentally committed: rotate the secret immediately, then `git filter-repo` to strip history, then force-push (the single documented exception to "never force-push").

---

## 11. Checklist (for PR authors)

- [ ] Branch name follows `type/short-description`
- [ ] Commits follow `type(scope): summary` with body when non-trivial
- [ ] Prompt template changes land atomically with all call sites
- [ ] PR body uses the template; prompt-contract section accurate
- [ ] Host unit tests pass locally
- [ ] Squash merge for feature → develop; merge commit for release → main
- [ ] No model weights, no binaries > 5 MB in git
- [ ] No credentials, no HF tokens
