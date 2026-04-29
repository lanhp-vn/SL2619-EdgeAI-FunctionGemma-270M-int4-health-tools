# 10 — Code Style: Shell

> Governs all shell scripts in the gemma3-270M-finetune project — host-side data sync helpers (WSL2 Ubuntu), GPU server bootstrap scripts, and deploy/bench wrappers. Baseline: **Google Shell Style Guide** with overrides documented below. **Bash only** — do not write `/bin/sh` portable scripts.

> **Scope**: any file with a `.sh` extension or a `#!/bin/bash` shebang. Does **not** apply to Python scripts even if they're in `scripts/`.

> **Tooling baseline** (from industry research 2025–2026):
> - **Google Shell Style Guide** is still canonical (updated 2025).
> - **ShellCheck** is mandatory; all warnings resolved.
> - **`[[ ... ]]`** over `[ ... ]` (no word splitting, pattern matching).
> - **`$(...)`** over backticks (nestable, readable).
> - **`set -euo pipefail`** for production scripts (exception: interactive/TTY scripts may omit).

---

## 1. Core Principles

1. **Bash, not sh.** `#!/bin/bash` always. We target bash 5+; GNU extensions are fine.
2. **ShellCheck-clean.** Every script passes `shellcheck` with no warnings.
3. **Fail fast.** `set -euo pipefail` on every non-interactive script.
4. **Quote everything.** Every variable expansion is double-quoted. No exceptions.
5. **Minimize external processes.** Use shell builtins (`${var//pat/repl}`, `${#var}`, `$(< file)`) over forking `sed`, `wc`, `cat`.
6. **Wrap remote calls with `timeout`.** A hung `ssh` should not hang the script indefinitely.

## 2. SSH / Remote Command Rules

Scripts in this repo talk to two remotes: the GPU server (`nouslogic-server`) and the SL2619 board (`nouslogic-sl2619`). These rules prevent paste-fragility and quoting bugs.

**Rule 1 — Use the SSH alias, never raw `user@IP`.**

The alias carries the right key and user per `~/.ssh/config`. Using a raw IP silently picks the wrong identity file.

**Rule 2 — Single-line commands only. No embedded newlines.**

Shell + SSH quoting compose unpredictably across three layers. Embedded `\n` in a bash body inside `ssh host '...'` breaks when pasted into a terminal — the terminal auto-indents continuations and the remote shell receives them as separate commands.

```bash
# WRONG — multi-line body gets fragmented on paste:
ssh nouslogic-server 'cd ~/sl2619-finetune &&
  source .venv/bin/activate &&
  python finetune.py'

# RIGHT — semicolons, one physical line:
ssh nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && python finetune.py'
```

**Diagnostic tell**: if a paste failure shows `command not found` for what should be an argument, the remote shell got newline-fragmented input. Re-format as one physical line and re-send.

**Rule 3 — Long blocks go in a script file, not an inline string.**

If the remote command exceeds ~120 characters or needs more than 2–3 statements, write a real `.sh` file, `scp` it, then `ssh host 'bash /tmp/script.sh'`.

**Rule 4 — Batch diagnostic probes into one SSH call.**

A single `ssh host '...'` with multiple probes separated by `echo "=== SECTION ==="` beats many separate invocations (each pays the handshake cost).

**Rule 5 — Absolute paths only in `scp`/`ssh` bodies.**

Relative paths resolve against different cwd on host vs server. `scp foo nouslogic-server:/tmp/` works; `scp ./foo nouslogic-server:tmp/` does not.

### 2.1 BusyBox caveats (SL2619 board only)

The SL2619 board runs BusyBox coreutils, not GNU coreutils. Commands that target the board via SSH must respect these differences:

| GNU form (works on host/server) | BusyBox on board — use this instead |
|---|---|
| `head -5` | `head -n 5` (must have `-n`) |
| `ip -br addr` | `ip addr show` + `awk` |
| `grep -P` (PCRE) | `grep -E` (POSIX ERE) |
| `ls --color` | Avoid; don't parse colored output |
| `ps -eo pid,cmd` | `ps` or `top -b -n 1` |
| `nl`, `cat -n` | May be absent; use `awk '{print NR"\t"$0}'` |
| `timeout` (as builtin) | Available as external binary — still works, but not a shell builtin |

The GPU server (`nouslogic-server`) runs standard Ubuntu — GNU coreutils apply there.

---

## 3. File Conventions

### 3.1 Filename

- `kebab-case.sh`. Examples: `server-bootstrap.sh`, `sync-data.sh`, `run-bench.sh`.
- Must be executable (`chmod +x`).
- Rarely: `snake_case.sh` when the filename mirrors the function name it provides.

### 3.2 Header

```bash
#!/bin/bash
# Sync training data to the GPU server.
#
# Usage: sync-data.sh [--dry-run]
#
# Requires: ssh alias nouslogic-server defined in ~/.ssh/config

set -euo pipefail
```

### 3.3 Structure

Use `#region` / `#endregion` markers to fold logical blocks:

```bash
#region Configuration
readonly SERVER="nouslogic-server"
readonly REMOTE_DIR="~/sl2619-finetune"
readonly DATA_DIR="data"
#endregion

#region Functions
log()   { printf '%s [%s] %s\n' "$(date +'%H:%M:%S')" "$1" "${*:2}" >&2; }
info()  { log INFO  "$*"; }
error() { log ERROR "$*"; }
#endregion

#region Main
main() {
    # ...
}

main "$@"
#endregion
```

## 4. Naming

| Kind | Convention | Example |
|---|---|---|
| Filename | `kebab-case.sh` | `run-bench.sh` |
| Function | `snake_case()` | `sync_data()`, `verify_server()` |
| Global / readonly | `UPPER_SNAKE` | `SERVER`, `REMOTE_DIR` |
| Local | `lower_snake` | `local src path` |
| Loop index (simple) | single letter | `for i in ...` |
| Loop index (meaningful) | `snake_case` | `for checkpoint_name in ...` |

## 5. Shebang & Strict Mode

### 5.1 Standard preamble

```bash
#!/bin/bash
set -euo pipefail
IFS=$'\n\t'            # Prevents word-splitting on unexpected whitespace
```

| Flag | Meaning |
|---|---|
| `-e` | Exit on any command failure |
| `-u` | Exit on undefined variable reference |
| `-o pipefail` | Pipe fails if any stage fails |
| `IFS=$'\n\t'` | Safer default word splitting |

### 5.2 Exceptions to `-euo pipefail`

- **Scripts that intentionally check exit codes** with `cmd && ... || ...` need care — scope `set +e` / `set -e` pairs around the relevant block.

## 6. Quoting

### 6.1 Rule: quote every expansion

```bash
# Good
scp "$src" "$SERVER:/tmp/"
for file in "${files[@]}"; do ... done
if [[ -z "$var" ]]; then ... fi

# Bad
scp $src $SERVER:/tmp/        # word splitting on filenames with spaces
for file in ${files[@]}        # same
if [ -z $var ]                # unquoted var AND single-bracket test
```

### 6.2 `[[ ... ]]` over `[ ... ]`

```bash
# Good — no word splitting inside [[ ]], supports patterns
if [[ "$name" == "*.sh" ]]; then ...
if [[ -f "$path" && -r "$path" ]]; then ...

# Bad — [ ] is the /bin/sh test builtin with word-splitting landmines
if [ $name = "*.sh" ]; then ...
```

### 6.3 Fallback defaults

```bash
# ${VAR:-default} — use default if VAR unset or empty
SERVER="${SERVER:-nouslogic-server}"
TIMEOUT_S="${TIMEOUT_S:-30}"
```

### 6.4 `local` separated from assignment (ShellCheck SC2155)

```bash
# Good — exit code of command visible
local val
val="$(some_command)"

# Bad — local's exit code masks the command's, always 0
local val="$(some_command)"
```

## 7. Command Substitution & Builtins

### 7.1 `$(...)` always; never backticks

```bash
# Good
local sha
sha="$(sha256sum "$file" | awk '{print $1}')"

# Bad
sha=`sha256sum $file | awk '{print $1}'`
```

### 7.2 Prefer builtins to reduce fork/exec

| Prefer | Over | Saves |
|---|---|---|
| `${var//old/new}` | `echo "$var" \| sed 's/old/new/'` | 1 fork + 1 pipe |
| `${#string}` | `echo "$string" \| wc -c` | 1 fork + 1 pipe |
| `$(< "$file")` | `$(cat "$file")` | 1 fork |
| `printf '%s\n' "$x"` | `echo -e "$x"` | portability; avoids echo's `-e`/`-n` inconsistency |
| `[[ "$x" == prefix* ]]` | `echo "$x" \| grep -q '^prefix'` | 1 fork + 1 pipe |

### 7.3 Loop over arrays, not word splits

```bash
# Good
checkpoints=(v1 v2 v3)
for ckpt in "${checkpoints[@]}"; do
    ...
done

# Bad (works until a name has a space)
for ckpt in $(ls checkpoints/); do
    ...
done
```

## 8. External Probes

Every external command that talks to the server or the board wrapped in `timeout`:

```bash
if ! timeout 10 ssh nouslogic-server 'test -d ~/sl2619-finetune' >/dev/null 2>&1; then
    error "Server not reachable — is nouslogic-server accessible?"
    exit 1
fi
```

For retries:

```bash
retry() {
    local -i attempts=$1 delay_s=$2
    shift 2
    local -i i
    for ((i = 1; i <= attempts; i++)); do
        if "$@"; then return 0; fi
        sleep "$delay_s"
    done
    return 1
}

retry 3 5 timeout 15 ssh nouslogic-server 'true'
```

## 9. Data Sync Script Pattern

Canonical `scripts/sync-data.sh` skeleton:

```bash
#!/bin/bash
# Sync SFT training data to the GPU server.
set -euo pipefail

#region Configuration
readonly SERVER="nouslogic-server"
readonly REMOTE_DIR="~/sl2619-finetune"
readonly LOCAL_DATA="data"
#endregion

#region Helpers
log()   { printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$*" >&2; }
die()   { log "ERROR: $*"; exit 1; }
#endregion

#region Main
main() {
    local dry_run=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done

    log "verifying server reachable"
    timeout 10 ssh "$SERVER" 'true' || die "server not reachable"

    local rsync_flags=("-avz" "--exclude=__pycache__")
    if [[ "$dry_run" == true ]]; then
        rsync_flags+=("--dry-run")
        log "DRY RUN — no files will be transferred"
    fi

    log "syncing $LOCAL_DATA/ -> $SERVER:$REMOTE_DIR/"
    rsync "${rsync_flags[@]}" "$LOCAL_DATA/" "$SERVER:$REMOTE_DIR/data/"

    log "done"
}

main "$@"
#endregion
```

## 10. Argument Parsing

### 10.1 Simple — positional or `--flag` style

```bash
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --server)     SERVER="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        -*)           die "unknown option: $1" ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done
```

### 10.2 `usage()` is mandatory for anything > 20 lines of logic

```bash
usage() {
    cat <<EOF
Usage: $0 [--dry-run] [--server HOST]

  --dry-run       Show what would be synced without transferring
  --server HOST   Override SSH target (default: nouslogic-server)
  -h, --help      Show this help
EOF
}
```

## 11. Error Handling & Exit Codes

- **Exit 0**: success.
- **Exit 1**: generic failure.
- **Exit 2**: misuse (bad args, missing env).
- **Exit >= 10**: domain-specific (`10` = server not reachable, `11` = artifact missing).

Log with structured levels:

```bash
log()   { printf '%s [%s] %s\n' "$(date +'%H:%M:%S')" "$1" "${*:2}" >&2; }
info()  { log INFO  "$*"; }
warn()  { log WARN  "$*"; }
error() { log ERROR "$*"; }
die()   { error "$*"; exit 1; }
```

## 12. Forbidden

| Pattern | Why |
|---|---|
| Unquoted `$var` expansion | SC2086: word splitting, glob expansion |
| `local var=$(cmd)` (combined) | SC2155: exit code of cmd is masked |
| `` `backticks` `` | Not nestable, less readable |
| `[ ... ]` single-bracket test | Use `[[ ... ]]` |
| `echo -e` / `echo -n` | Not portable across bash/dash; use `printf` |
| `cat file \| grep pat` | `grep pat file` directly |
| `$(cat file)` | `$(< file)` |
| `cd dir` without `|| exit` | SC2164 |
| `for i in $(seq 1 N)` | Use `for ((i=1; i<=N; i++))` |
| `which cmd` | Not POSIX; use `command -v cmd` |
| Hardcoded paths to `/usr/bin/bash` etc | Use `#!/bin/bash` |

## 13. Forbidden (destructive)

**Never commit a script that does any of the following without a `--really` or equivalent opt-in flag**:

- `rm -rf "$path"` on a computed `$path` (always check `$path` is non-empty and not `/`).
- `sudo` in a non-interactive script without logging what it's about to do.

## 14. Running ShellCheck

```bash
# Run in the repo root; gate on zero warnings
shellcheck scripts/*.sh
# Or, with source-filter:
shellcheck --exclude=SC1091 scripts/*.sh   # SC1091 = "can't follow non-constant source"
```

CI runs this; pre-commit hooks are recommended but not required.

---

## 15. Checklist (paste into PR)

- [ ] Shebang is `#!/bin/bash`; file is `chmod +x`
- [ ] `set -euo pipefail` (or documented exception)
- [ ] `shellcheck` passes with no warnings
- [ ] Every `$var` is double-quoted
- [ ] `[[ ... ]]` not `[ ... ]`
- [ ] `$(...)` not backticks
- [ ] External probes wrapped in `timeout`
- [ ] `local` separated from assignment
- [ ] Arrays used in place of word splits
- [ ] Destructive actions gated behind explicit flag
