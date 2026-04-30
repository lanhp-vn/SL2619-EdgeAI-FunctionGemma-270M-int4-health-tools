---
name: ruleset-update
description: Review and generalize the workspace permission ruleset in .claude/settings.local.json based on commands granted during the current session. Use when the user wants to tidy up per-approval rule entries into reusable glob patterns, consolidate duplicates, and reaffirm the deny list. Does NOT broaden scope beyond what the user has already approved in this or prior sessions.
---

# ruleset-update

Rewrite `.claude/settings.local.json` so the `allow` list reflects the user's approvals generalized into reusable patterns, and the `deny` list keeps the standard safety rails.

## Procedure

1. **Read the current ruleset**:
   ```bash
   cat .claude/settings.local.json
   ```
   The `allow` array accumulates specific command strings each time the user approves a Bash call. Over a session it fills with overly narrow entries like `Bash(curl -sL "https://example.com/very/specific/path")`.

2. **Scan this session's tool-use history** for every Bash call that was actually run and approved. Group by intent:
   - Python / pip invocations
   - gdown runs
   - curl / wget against specific domains
   - Read-only inspection (`ls`, `find`, `du`, `wc`, `head`, `tail`, `grep`, `cat`)
   - Git reads (`git status`, `git log`, `git diff`)
   - WebFetch domains approved during the session
   - Anything else the user approved

3. **Generalize each group into the narrowest glob that still covers the approved intent**. Examples:

   | Session-specific entry (before) | Generalized rule (after) |
 | - - - | - - - |
   | `Bash(python -m pip --version)` | `Bash(python:*)` |
   | `Bash(PYTHONIOENCODING=utf-8 python -m gdown --folder ABC -O ./drive_files)` | `Bash(PYTHONIOENCODING=utf-8 python:*)` |
   | `Bash(curl -sL "https://docs.google.com/spreadsheets/d/<ID>/export?format=xlsx" ...)` | `Bash(curl -sL https://docs.google.com/*)` |
   | `Bash(find ./drive_files -type f ...)` | `Bash(find ./drive_files:*)` |
   | `Bash(git -C /d/3-Nouslogic/robotic-arm-hand/SynapticSL2619 log --oneline)` | `Bash(git log:*)` |

   Rules:
   - Prefer `command:*` over fully-specified strings when the user clearly approved the whole command family.
   - Keep domain scope on network calls — do not widen `curl -sL https://docs.google.com/*` to `curl:*`.
   - Keep path scope on filesystem reads — e.g. `find ./drive_files:*` not `find:*`.
   - Drop exact duplicates and strings fully subsumed by a glob already in the list.
   - For `WebFetch`, keep the `domain:<host>` form — do not widen to all domains.

4. **Preserve the deny list** (standard safety rails — always include these, do not drop):
   - `Bash(rm -rf:*)`
   - `Bash(git push --force:*)`, `Bash(git push -f:*)`
   - `Bash(git reset --hard:*)`
   - `Bash(git clean -f:*)`
   - `Bash(git branch -D:*)`
   - `Bash(curl * | sh)`, `Bash(curl * | bash)`, `Bash(wget * | sh)` (pipe-to-shell installs)
   - `Bash(* --no-verify*)` (hook bypass)
   - `Bash(pip install --upgrade pip)`, `Bash(python -m pip install --upgrade pip)` (protects the system Python)

   If the user has added custom deny rules, keep those too.

5. **Write the new `.claude/settings.local.json`** with a single `permissions` object containing `allow` and `deny`. Preserve any non-permissions keys that were already in the file.

6. **Show the user a diff-style summary**: what consolidated into what, and anything new added to deny. Do not just rewrite silently.

## Guardrails

- **Never add a rule the user did not approve this session or in the existing file.** This skill generalizes; it does not invent new permissions.
- **Never remove a deny rule** unless the user explicitly asks. Broadening allow is reversible; dropping a deny is not.
- **Ask before widening scope** if a generalization would cover meaningfully more than what was approved — e.g. going from `Bash(curl -sL https://docs.google.com/spreadsheets/*)` to `Bash(curl:*)` is too broad without confirmation.
- **Settings do not hot-reload.** Tell the user to restart Claude Code in this workspace for the new rules to take effect.
