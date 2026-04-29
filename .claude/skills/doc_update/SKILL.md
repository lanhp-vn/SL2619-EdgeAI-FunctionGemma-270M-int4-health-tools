---
name: doc_update
description: Refresh CLAUDE.md (agent self-reference) and README.md (human-facing) at the repo root after architectural or workflow changes. Use when subsystems are added/removed, the IPC wire contract changes, hardware setup changes, build/deploy workflow changes, an Iron Law is reworded, or before tagging a release. Does NOT update files inside docs/conventions/ — those go through normal PRs.
---

# doc_update

Keep `CLAUDE.md` and `README.md` at the repo root synchronized with the current state of the codebase. This skill is the invokable form of the **Documentation Update Protocol** in `docs/conventions/13-documentation-update-protocol.md` — that file holds the normative content. This skill is the thin hand-off.

## Invocation

User runs `/doc_update`. The agent follows the steps below.

## Procedure

### 1. Read the protocol

Before doing anything, read the normative protocol once so the behavior matches the written rules:

```
docs/conventions/13-documentation-update-protocol.md
```

If the procedure below diverges from that file, the file wins. Raise the inconsistency instead of proceeding.

### 2. Optional pre-flight

If the doc refresh touches deployment paths or runtime perf claims, ask the
user to run `/board_probe` first (or paste fresh output) so any board-state
claims in `README.md` / `CLAUDE.md` are grounded. The agent does **not** SSH
to the board itself.

If no deployment/runtime content is touched (e.g. fine-tune workflow,
training data, repo layout, references), skip pre-flight and proceed to §3.

### 3. Context analysis

Per §3 of the protocol:

1. Project root is `/home/lanhp-wsl/nouslogic/SynapticSL2619`.
2. Read these files in parallel (use batched `Read` calls):
   - `CLAUDE.md` (current content, if any)
   - `README.md` (current content, if any)
   - `docs/conventions/00-iron-laws.md`
   - `docs/conventions/01-architecture.md`
   - `docs/plans/plan.md`
   - `docs/tmp/sl2619-status.md` (the canonical live-board snapshot; run `/board_probe` first if stale)
3. Scan the codebase:
   - `ls a55/ m52-firmware/ tools/ scripts/` — which subsystems exist on disk? (Empty means "planned, not yet implemented".)
   - `git log --oneline -20` — recent activity.
   - `git status` — in-flight work.
4. Identify the gap between current docs and current state. List each discrepancy before drafting.

### 4. Draft CLAUDE.md updates

Per §4 of the protocol:

- Target length: **≤ 300 lines**.
- Required sections: project overview, Iron Laws TL;DR, directory tree, dual-domain map, common workflows, testing cheat sheet, tech-debt, pointers into `docs/conventions/`.
- **Summarize** Iron Laws; do not duplicate normative text. Point to `docs/conventions/00-iron-laws.md` for the full version.
- Distinguish **current** state from **planned** state explicitly.
- **Format for Notion compatibility** per §6 below — apply before presenting.

### 5. Draft README.md updates

Per §5 of the protocol:

- Target length: **≤ 500 lines**.
- Required sections: title + tagline, capability bullets, hardware BOM, host setup, build, flash/deploy, run, testing cheat sheet, repo layout, license.
- Audience: senior engineer new to this repo. Not a beginner.
- **No Iron Laws** in README.md — human-facing setup doc, not normative rules.
- **Format for Notion compatibility** per §6 below — apply before presenting.

### 6. Format for Notion compatibility (lint before diff)

Every `.md` file this skill produces must render cleanly in **both** GitHub-flavored Markdown and Notion's `File → Import → Markdown` importer. Notion is the stricter of the two — these rules keep drafts compatible with both readers, so the repo's docs drop straight into a Notion workspace without reflow.

**Required formatting:**

- **Headings**: H1–H3 for primary structure; H4 sparingly; avoid H5/H6. One blank line before and after every heading.
- **Lists**: use `-` for bullets (not `*` or `+`). Nest with 2-space indent. Task lists (`- [ ]` / `- [x]`) import as Notion to-dos.
- **Tables**: GitHub pipe syntax only. Header row + `---` separator. **Cells are single-line** — Notion does NOT render `<br/>`, `<br>`, or embedded bullets inside cells. Rewrite multi-line content as comma-separated clauses, or split into adjacent rows. Escape literal pipes as `\|`. Skip alignment markers (`:---:`) — they add noise and Notion ignores them.
- **Code fences**: triple-backtick with a language identifier (`c`, `bash`, `mermaid`, `text`, etc.). Indented code blocks import poorly.
- **No inline HTML outside code fences**: no `<br/>`, `<sub>`, `<sup>`, `<u>`, `<div>`, `<span>`, `<b>`, `<i>`. Use native Markdown or rephrase. Inside a ```` ```mermaid ```` block, `<br/>` is part of Mermaid's DSL — leave it.
- **Block quotes** (`>`): import as Notion callouts. Keep them short; `>` on every continuation line.
- **Links**: inline `[text](url)` only. Reference-style (`[text][1]` with footer `[1]: url`) survives GitHub but is unreliable through Notion's import.
- **Emoji**: **Do not use emojis** in any documentation file (project-wide rule — applies to `CLAUDE.md`, `README.md`, `docs/**`, `.claude/**`, and any `.md` this skill produces). If a cue is needed, use a plain-text label like `WARNING:`, `Hazard:`, `PASS` / `FAIL`, `OK`, `NOTE:`. Do not substitute GitHub shortcodes (`:warning:`, `:camera:`) either — those are emojis by another name.

**Mermaid diagrams:**

Notion natively renders Mermaid inside code blocks (feature shipped January 2022 — [reference](https://lukemerrett.com/using-mermaid-flowchart-syntax-in-notion/)) and its Markdown importer preserves triple-backtick fences with language tags, so a ```` ```mermaid ```` block survives import and renders correctly. **No ASCII fallback is required for Notion compatibility.** Keep the fence language tag literally `mermaid` (not `Mermaid`, not a comment before the block) — the importer keys off the exact tag. Only add a prose summary alongside the diagram when the target reader might be viewing the file in a plain-text editor without any Markdown renderer.

**Lint-before-diff checklist** — grep each draft and fix before presenting in §7:

- `<br/?>` outside ```` ```mermaid ```` blocks → rewrite in Markdown.
- `<sub|<sup|<u>|<div|<span|<b>|<i>` → rewrite.
- Lines starting with `*` followed by whitespace (outside code fences) → change to `- `.
- Any table row with mismatched pipe count → fix.
- Any Mermaid block with fence tag other than lowercase `mermaid` → fix (Notion importer is case-sensitive).
- Any emoji character (Unicode pictographs, dingbats, checkmarks like ✓, warning signs, colored symbols) anywhere in the file → delete or replace with a plain-text label (`OK`, `FAIL`, `WARNING:`). Structural box-drawing characters inside code fences (`├`, `│`, `└`, `─`, `→`, `▼`) are NOT emojis and may stay.

Fix silently; do not surface formatting churn in the user-facing diff summary.

### 7. Present diffs

Show the user:

1. A concise summary of what changed and why (1–3 bullets per file).
2. The full proposed content for each file if the change is material, or a focused diff if it's a minor edit.
3. Flag any uncertainty — places where the current repo state was ambiguous.

**Wait for explicit approval before writing.**

### 8. Apply

On approval:

1. Use the `Write` tool to replace `CLAUDE.md` and `README.md` at the repo root.
2. Do **not** commit automatically. Prompt the user: "Ready to commit as `docs: refresh CLAUDE.md and README.md for <change summary>`? (y/n)"
3. On yes, create the commit per `docs/conventions/12-git-workflow.md` conventions.

## What this skill does NOT do

- Does **not** update files inside `docs/conventions/`. Those go through normal PRs with review.
- Does **not** touch `docs/datasheets/tech-reference.md` — **frozen, human-curated compendium** per §10.2 of the protocol. Read it, never rewrite it.
- Does **not** restructure files under `docs/plans/**` (currently `plan.md` and `backlogs.md`; any new `.md` added there inherits the policy) — they are **standalone planning docs** per §10.2. Update their content when the plan/backlog itself changes; do not collapse their narrative into pointers.
- Does **not** invent content not present in the codebase or the board snapshot. If a subsystem isn't implemented, don't claim it is.
- Does **not** commit automatically without the user's explicit go.
- Does **not** run on-board commands itself — the user runs them and pastes results back.

## Consistency

If the normative protocol in `docs/conventions/13-documentation-update-protocol.md` is updated, this SKILL.md's hand-off instructions are reviewed in the same PR for consistency.
