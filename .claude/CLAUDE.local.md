# CLAUDE.local.md — Developer-Local Configuration (gitignored)

> This file is **not committed** (covered by `.gitignore` rule `.claude/`). It stores machine-local secrets and setup that must never ship in git. Claude Code loads this alongside `CLAUDE.md`; skills also read it directly when they need credentials.
>
> If you share this workspace with a teammate, each teammate maintains their own `CLAUDE.local.md`. Do not paste its contents into chat, issues, or PRs.

---

## 1. Trusted SSH targets

The agent has READ-ONLY SSH access (per `CLAUDE.md` §3 R3) to two hosts. Both are reached via host aliases in `~/.ssh/config`; both keys are passphrase-protected. R3 forbids state-changing SSH on **either** host — the agent observes; the user mutates.

### 1a. SL2619 board

| Item | Value |
|---|---|
| Host alias | `nouslogic-sl2619` |
| IP | `192.168.12.240` (DHCP — may change; rerun `/board_probe` to discover) |
| User | `root` |
| Private key | `~/.ssh/sl2619_nouslogic_wsl` |
| Passphrase | `2411` |
| Hostname (on box) | `nouslogic` (per `get-started/sl2610-get-started.md` §8.2) |
| OS | Yocto Linux + BusyBox coreutils (mind caveats below) |

### 1b. Fine-tune server (RTX 5080, Blackwell sm_120)

| Item | Value |
|---|---|
| Host alias | `nouslogic-server` |
| IP | `100.116.133.62` |
| User | `hoanglan` |
| Private key | `~/.ssh/nouslogic_server_ed25519` |
| Passphrase | `2411` |
| Purpose | Phase 0+ Gemma 3 270M QLoRA fine-tune (see `docs/plans/AI-models/a55-gemma-fine-tune.md`) |
| OS | Ubuntu (full GNU coreutils — no BusyBox limits) |

### Security note
Passphrases live here rather than in `CLAUDE.md` because `CLAUDE.md` is committed to git — anything in `CLAUDE.md` stays in history forever. If a key ever leaves the developer laptop (screen-share, backup, lost device), rotate it:
- `ssh-keygen -p -f ~/.ssh/sl2619_nouslogic_wsl` and update this file + `authorized_keys` on the board.
- `ssh-keygen -p -f ~/.ssh/nouslogic_server_ed25519` and update this file + `authorized_keys` on the server.

---

## 2. Agent pattern for SSH pre-flight (READ-ONLY per R3)

Every skill that needs SSH constructs a one-shot `ssh-agent` + askpass helper, batches all probe commands in a **single SSH session** using `echo "===" ` delimiters, then tears down the agent. This avoids agent-across-shells fragility (each Bash tool call is a fresh shell) and minimizes authentication overhead.

The pattern is host-parametric — same shape, different alias / key / askpass tmp file. Use the relevant `${HOST_ALIAS}` and `${KEY}` for the target you're probing.

```bash
# Pick target — sl2619 OR server
HOST_ALIAS=nouslogic-sl2619                   # or nouslogic-server
KEY=~/.ssh/sl2619_nouslogic_wsl               # or ~/.ssh/nouslogic_server_ed25519
ASKPASS=/tmp/askpass_${HOST_ALIAS}.sh         # named per-host so concurrent probes don't collide

# 1. Start ephemeral ssh-agent and load the key non-interactively
eval "$(ssh-agent -s)" >/dev/null
cat > "$ASKPASS" <<'ASKPASS_EOF'
#!/bin/sh
echo "2411"
ASKPASS_EOF
chmod +x "$ASKPASS"
DISPLAY=dummy:0 SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force \
    ssh-add "$KEY" < /dev/null

# 2. Run ALL read-only probes in ONE SSH call, separated by echo delimiters
ssh "$HOST_ALIAS" '
    echo "=== UNAME ===";    uname -a
    echo "=== MEMINFO ==="; head -n 25 /proc/meminfo
    # ... more READ-ONLY probes; content differs per target — see /board_probe ...
'

# 3. Tear down the ephemeral agent and remove the askpass file
kill "$SSH_AGENT_PID" 2>/dev/null || true
rm -f "$ASKPASS"
```

### BusyBox caveats (SL2619 only — Yocto uses BusyBox coreutils)
- `head -20` fails — use `head -n 20`.
- `ip -br addr` unsupported — use `ip addr show` + awk.
- `grep -P` (PCRE) missing — use `grep -E` (POSIX ERE).
- `ls --color` may be missing; don't rely on colored output parsing.
- `cat -n` works but `nl` may not be present.

The fine-tune server runs Ubuntu with full GNU coreutils — none of the above caveats apply there.

---

## 3. Deploy paths (out-of-agent, user-performed per R3 + R5)

The agent **never** pushes binaries, **never** restarts services, **never** flashes firmware. When a deploy step is reached, the agent emits the exact commands below and stops. You run them in your own terminal.

### Path A — WSL, `scp` + `ssh` (primary for most A55 flows)

```bash
# A55 coordinator binary deploy
scp a55/build/coordinator/coordinator nouslogic-sl2619:/tmp/coordinator
ssh nouslogic-sl2619 '
    mv /tmp/coordinator /usr/bin/coordinator
    chmod +x /usr/bin/coordinator
    systemctl restart coordinator.service
    sleep 2
    journalctl -u coordinator.service -n 30 --no-pager
'
```

### Path B — Windows PowerShell / Terminal, `adb` (USB-attached)

```powershell
# ADB is unreliable from WSL; run from Windows terminal instead
adb push a55/build/coordinator/coordinator /tmp/coordinator
adb shell 'mv /tmp/coordinator /usr/bin/coordinator && chmod +x /usr/bin/coordinator'
adb shell 'systemctl restart coordinator.service'
adb shell 'journalctl -u coordinator.service -n 30 --no-pager'
```

### M52 firmware flash (always user-performed, ≥ 3 min cycle, IL-8)

The agent produces `build/SYNAIMG`; you flash it. This is a **hard** hardware gate per R5 and IL-8 — no automation attempts, ever.

```bash
# From Windows terminal with USB boot-mode cable attached:
astra-update --flash build/SYNAIMG --chip sl2619 --board rdk --slot b
adb reboot
# Wait ~30 s, then verify M52 handshake:
adb shell 'journalctl -u coordinator.service -n 50 --no-pager | grep -i handshake'
```

Slot A promotion happens only after a full soak — see `docs/conventions/06-toolchain-build.md` §6.3.

---

## 4. Post-deploy verification (agent does this via READ-ONLY SSH)

After you confirm a deploy is done, the calling skill verifies completion by **observation, not action**:

```bash
ssh nouslogic-sl2619 '
    echo "=== BINARY ==="; ls -la /usr/bin/coordinator 2>&1
    echo "=== SERVICE ==="; systemctl is-active coordinator.service
    echo "=== JOURNAL ==="; journalctl -u coordinator.service -n 40 --no-pager
    echo "=== RPMSG ==="; ls /dev/rpmsg* 2>&1; ls /sys/bus/rpmsg/devices/ 2>&1
    echo "=== MEM ==="; systemctl show coordinator.service --property=MemoryCurrent
'
```

If any of those checks fails, the skill reports the failure to you and stops — it does not attempt remediation via SSH (that would violate R3).

---

## 5. Notion integration (docs → Notion sync)

| Item | Value |
|---|---|
| Integration name | `robotic-arm-docs-sync` (internal, workspace-scoped) |
| Integration console | <https://www.notion.so/profile/integrations> |
| Root page title | Robotic Arm & Hand Project |
| Root page URL | <https://www.notion.so/Robotic-Arm-Hand-Project-342200b3998d8098a988e9256605c921> |
| Root page ID (dashed UUID) | `342200b3-998d-8098-a988-e9256605c921` |
| API version pinned | `2026-03-11` (native markdown endpoints) |
| Token (`ntn_…`) | `ntn_E86017048458nO8LNUHNSGwD7aUWjTCq7qK8GHipeJ9gTH` |

### Shell export snippet

Source this block in any shell that needs to push docs:

```bash
export NOTION_TOKEN="ntn_E86017048458nO8LNUHNSGwD7aUWjTCq7qK8GHipeJ9gTH"                               # replace with the ntn_ secret
export NOTION_ROOT_PAGE_ID="342200b3-998d-8098-a988-e9256605c921"
export NOTION_API_VERSION="2026-03-11"
```

### Security note
Token lives here (gitignored) for the same reason as the SSH passphrase in §1. Rotate if the laptop leaves your possession or if this file is ever shared: revoke in the integration console (<https://www.notion.so/profile/integrations>), mint a new `ntn_…`, replace both occurrences above. The integration → page share (granted in the Notion UI Connections menu) survives rotation; only the token changes.

### Sanity-check commands

```bash
# 1. Confirm the integration can see the page (expect "object": "page")
curl -sS -X GET "https://api.notion.com/v1/pages/$NOTION_ROOT_PAGE_ID" \
    -H "Authorization: Bearer $NOTION_TOKEN" \
    -H "Notion-Version: $NOTION_API_VERSION" | jq '{object, id, archived}'

# 2. Smoke-test page creation under the root
jq -n --arg pid "$NOTION_ROOT_PAGE_ID" --arg md "# Smoke test\n\nCreated $(date -Iseconds)" \
    '{parent: {page_id: $pid}, markdown: $md}' | \
    curl -sS -X POST https://api.notion.com/v1/pages \
        -H "Authorization: Bearer $NOTION_TOKEN" \
        -H "Notion-Version: $NOTION_API_VERSION" \
        -H "Content-Type: application/json" --data @- | jq '{object, url}'
```

### Directory sync (npm tool, for `docs/` subtrees)

```bash
# Pin the version; re-run is idempotent (only changed blocks update)
npx @vrerv/md-to-notion@1.1.1 -t "$NOTION_TOKEN" -p "$NOTION_ROOT_PAGE_ID" docs/conventions
```
