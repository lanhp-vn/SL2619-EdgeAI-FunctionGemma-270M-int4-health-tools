# SL2619 post-recovery bring-up — Wi-Fi, hostname, SSH, NTP, microSD

> ✅ **Executed and verified 2026-06-01** after the v2.4.0 recovery flash. End
> state: hostname `nouslogic`, Wi-Fi `Kylie` persistent at `192.168.12.240`, NTP
> synced, TZ correct, microSD mounted, key-auth SSH from both Windows and WSL.
> The exact values, the SSH gotchas hit, and the next step (BLE bring-up) are in
> §7. This is a proven runbook.

Run this **after** a fresh eMMC reflash (see
[`sl2619-windows-recovery.md`](sl2619-windows-recovery.md)) to return a bare
Scarthgap image to the project's working state. The reflashed image ships with
**passwordless root** and hostname **`sl2619`**; this guide sets a password,
renames it to **`nouslogic`**, brings up **persistent Wi-Fi**, configures
**SSH key auth**, fixes the **clock/timezone**, and **re-mounts the microSD**.

Steps that run **on the board** start from `adb shell` (USB-C still attached) or
a serial console. Steps that run **on the host** are marked `# host`. Do them in
this order — later steps (NTP, SSH-over-Wi-Fi) depend on earlier ones (network).

Substitute these placeholders: `YOUR_SSID`, `YOUR_PASSWORD`, `<BOARD_IP>` (the
DHCP address you read in §2.4), and the email/key names to taste.

> **Board-specific traps baked into this guide** (each has bitten before):
> - Root's home is **`/home/root`**, *not* `/root` — `authorized_keys` there.
> - Dropbear loads **only an RSA host key** by default → ed25519 client keys are
>   silently rejected until you add an ed25519 host key.
> - The image ships **no `tzdata`** → use a POSIX `TZ` string, not
>   `set-timezone`.
> - **No battery-backed RTC** → the clock is wrong every boot until NTP syncs.
> - BusyBox `ip` has **no `-br`** flag; `date` emits literal `%N`.

---

## 1. Re-mount the microSD (persistent)

The microSD (`mmcblk2p1`) is a **separate device** from the eMMC you reflashed —
your models/fixtures survived. Mount it and make it persistent.

On the board:
```sh
mkdir -p /mnt/sdcard
mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard        # one-shot, this boot
ls /mnt/sdcard                                   # expect your models/ fixtures/

# Persistent across reboot — nofail so boot survives if the card is ever absent
grep -q '/mnt/sdcard' /etc/fstab || \
  echo '/dev/mmcblk2p1  /mnt/sdcard  ext4  defaults,noatime,nofail  0  2' >> /etc/fstab
mount -a && mount | grep sdcard                  # verify fstab entry mounts cleanly
```
> If `mount` fails with a filesystem error, the card may be unformatted/exfat —
> this image's kernel supports only ext3/ext4/vfat. Reformat per get-started
> `sd-card-ext4-format-get-started.md`. Verify model integrity afterward against
> `releases/functiongemma-270m/001-baseline/gguf/CHECKSUMS.txt`.

---

## 2. Root password, hostname, Wi-Fi

### 2.1 Set the root password
```sh
passwd
# New password: nouslogic   (or your choice)
```

### 2.2 Set hostname to `nouslogic`
```sh
hostnamectl set-hostname nouslogic
printf '127.0.0.1   localhost nouslogic\n::1         localhost nouslogic\n' > /etc/hosts
hostname        # expect: nouslogic
```
The mDNS name becomes `nouslogic.local` once avahi picks it up.

### 2.3 Bring up Wi-Fi (one-shot first, to confirm the radio + credentials)
The onboard radio is a Broadcom FullMAC part (`bcmdhd`) exposing `wlan0`.
```sh
ip link set wlan0 up
iw dev wlan0 scan | grep -E 'SSID|signal' | head -n 40     # confirm your SSID is visible

# Hash the PSK (plaintext password is never written to disk)
wpa_passphrase "YOUR_SSID" "YOUR_PASSWORD" > /etc/wpa_supplicant.conf
chmod 600 /etc/wpa_supplicant.conf

wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf -Dnl80211
sleep 3
iw dev wlan0 link        # expect: Connected to <bssid>  SSID: YOUR_SSID
udhcpc -i wlan0
ip addr show wlan0       # note the inet address → this is <BOARD_IP>
ping -c 3 1.1.1.1
```
> Benign: `wpa_supplicant` logs `nl80211: Registration to specific type not
> supported` on start with bcmdhd — ignore, check `iw dev wlan0 link` for the
> real status. If it says **Not connected** after ~10 s, retry with the wext
> driver: `killall wpa_supplicant; wpa_supplicant -B -i wlan0 -c
> /etc/wpa_supplicant.conf -Dwext`.

### 2.4 Make Wi-Fi persistent (systemd-networkd + wpa_supplicant@wlan0)
The one-shot above dies on reboot. Hand the stack to systemd:
```sh
# 1. ctrl_interface so the systemd-managed supplicant is controllable
grep -q ctrl_interface /etc/wpa_supplicant.conf || \
  sed -i '1i ctrl_interface=/run/wpa_supplicant\nupdate_config=1\n' /etc/wpa_supplicant.conf

# 2. The template unit reads this exact path
mkdir -p /etc/wpa_supplicant
cp /etc/wpa_supplicant.conf /etc/wpa_supplicant/wpa_supplicant-wlan0.conf
chmod 600 /etc/wpa_supplicant/wpa_supplicant-wlan0.conf

# 3. DHCP on wlan0 (no leading whitespace — strict ini)
cat > /etc/systemd/network/25-wlan0.network <<'EOF'
[Match]
Name=wlan0

[Network]
DHCP=yes
IPv6AcceptRA=yes
EOF

# 4. Enable + start
systemctl enable wpa_supplicant@wlan0.service systemd-networkd.service
systemctl restart systemd-networkd

# 5. Hand the live association over to the systemd unit (proves the boot path)
killall wpa_supplicant 2>/dev/null; sleep 2
systemctl start wpa_supplicant@wlan0.service
sleep 3
networkctl status wlan0        # expect: State: routable (configured), Online
iw dev wlan0 link
ping -c 2 1.1.1.1
```
> **⚠️ The real `<BOARD_IP>` is the one in `networkctl status wlan0` here — NOT
> the one-shot `udhcpc` lease from §2.3.** They often differ: `udhcpc` and
> `systemd-networkd` each pull their own lease (seen 2026-06-01: udhcpc got
> `.239`, systemd-networkd got `.240`; `.239` then went dead → "No route to
> host"). Use the `networkctl` address everywhere below. Since it's a DHCP
> lease, add a **router DHCP reservation** for the wlan0 MAC if you want the
> address (and the `~/.ssh/config` alias) to survive reboots.

The board has no battery-backed RTC and the image ships no `tzdata`, so use NTP +
a POSIX `TZ` string. Do this **after** §2 (needs network).
```sh
# NTP
timedatectl set-ntp true
systemctl enable --now systemd-timesyncd
timedatectl status        # expect: System clock synchronized: yes

# Timezone via POSIX TZ (glibc built-in; no data files needed)
echo "export TZ='PST8PDT,M3.2.0,M11.1.0'" > /etc/profile.d/tz.sh   # US Pacific w/ DST
chmod 644 /etc/profile.d/tz.sh
export TZ='PST8PDT,M3.2.0,M11.1.0'
date        # expect local PST/PDT time
```
> `timedatectl` will still report `Time zone: n/a (UTC)` — that's fine, glibc
> honors `$TZ`. For other regions swap the string (e.g.
> `EST5EDT,M3.2.0,M11.1.0`, `CET-1CEST,M3.5.0,M10.5.0/3`). For a systemd service
> that needs the TZ, add `Environment=TZ=PST8PDT,M3.2.0,M11.1.0` to its
> `[Service]` section (services don't source `/etc/profile.d`).

---

## 4. SSH key authentication

Now the board is reachable at `<BOARD_IP>` (the **`networkctl` address**, §2.4)
over Wi-Fi. Set up key-based login.

> **WSL note:** WSL2 **cannot resolve mDNS / `.local` / bare hostnames** — `ssh
> root@sl2619` fails with "Could not resolve hostname". From WSL always use the
> **IP** (or a DHCP-reserved IP). Windows *can* resolve `nouslogic.local`. WSL2
> *can* reach the LAN IP (it NATs through the Windows host), so the IP works
> fine — just not the name.

### 4.1 (Board) Generate an ed25519 host key — REQUIRED
Dropbear only enables algorithms for host keys it has loaded; the image ships
only RSA, so ed25519 *client* keys are silently rejected until you do this:
```sh
dropbearkey -t ed25519 -f /etc/dropbear/dropbear_ed25519_host_key
cat > /etc/default/dropbear <<'EOF'
DROPBEAR_EXTRA_ARGS="-r /etc/dropbear/dropbear_ed25519_host_key"
EOF
# Note: this also drops the shipped " -B" flag (which allowed blank-password logins).
```

### 4.2 (Host) Generate a client key pair
**Windows PowerShell:**
```powershell
# host
mkdir $env:USERPROFILE\.ssh -Force
ssh-keygen -t ed25519 -C "you@example.com" -f $env:USERPROFILE\.ssh\sl2619_nouslogic
```
**WSL (separate key — Windows key isn't reachable from WSL paths):**
```bash
# host (WSL)
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -C "you@example.com" -f ~/.ssh/sl2619_nouslogic_wsl
```

### 4.3 (Host) Copy the public key to `/home/root/.ssh/` — NOT `/root`
**Windows PowerShell** (note `tr -d '\r'` fixes CRLF that Windows tools inject):
```powershell
# host
type $env:USERPROFILE\.ssh\sl2619_nouslogic.pub | ssh root@<BOARD_IP> "mkdir -p /home/root/.ssh && cat >> /home/root/.ssh/authorized_keys && tr -d '\r' < /home/root/.ssh/authorized_keys > /tmp/ak && mv /tmp/ak /home/root/.ssh/authorized_keys && chmod 700 /home/root/.ssh && chmod 600 /home/root/.ssh/authorized_keys"
# enter the root password from §2.1
```
**WSL** (preferred — `ssh-copy-id` handles perms + no CRLF risk):
```bash
# host (WSL)
ssh-copy-id -i ~/.ssh/sl2619_nouslogic_wsl.pub root@<BOARD_IP>
```
**Over adb** (no password, if USB-C still attached):
```powershell
# host
adb push $env:USERPROFILE\.ssh\sl2619_nouslogic.pub /tmp/
adb shell "mkdir -p /home/root/.ssh && tr -d '\r' < /tmp/sl2619_nouslogic.pub >> /home/root/.ssh/authorized_keys && chmod 700 /home/root/.ssh && chmod 600 /home/root/.ssh/authorized_keys && rm /tmp/sl2619_nouslogic.pub"
```

> **⚠️ Clear stale `known_hosts` first (you reflashed → new host key).** The
> board's IP/name still carry the *pre-reflash* host key, so the first connect
> screams `REMOTE HOST IDENTIFICATION HAS CHANGED`. This is expected, not a MITM
> — the offered fingerprint matches your §4.1 `dropbearkey` output. Purge **every**
> name the board answers to, on **both** hosts:
> ```bash
> # WSL
> ssh-keygen -f ~/.ssh/known_hosts -R <BOARD_IP>
> ```
> ```powershell
> # Windows — clear the IP AND every alias/.local/IPv6 it was cached under
> ssh-keygen -R <BOARD_IP>; ssh-keygen -R nouslogic; ssh-keygen -R nouslogic.local
> ```

### 4.4 (Host) Add an SSH config entry
> **SSH does NOT do partial/substring matching on `Host`.** `ssh nouslogic` will
> **ignore** a `Host nouslogic-sl2619` block entirely and fall back to defaults:
> user = your local login (e.g. `phl`), no key → it tries to password-auth a
> user that doesn't exist on the board → `Permission denied`. The board password
> was never the problem. **Put every name you'll type on the `Host` line.**

**Windows** `~/.ssh/config` (both aliases on one block):
```
Host nouslogic nouslogic-sl2619
    HostName <BOARD_IP>
    User root
    IdentityFile ~/.ssh/sl2619_nouslogic
    IdentitiesOnly yes
```
**WSL** `~/.ssh/config` (its own key file; both keys coexist in `authorized_keys`):
```
Host nouslogic nouslogic-sl2619
    HostName <BOARD_IP>
    User root
    IdentityFile ~/.ssh/sl2619_nouslogic_wsl
    IdentitiesOnly yes
```
> **`User root` is mandatory** — without it ssh uses your Windows/WSL username.
> The prompt `Enter passphrase for key …` is the **private-key passphrase**, NOT
> the board's root password — different secret. `root@…'s password:` is the
> board's; `Enter passphrase for key:` is your key's.

**Shortcut — install the Windows key from the working WSL session** (avoids the
Windows password-through-pipe dance; the Windows pubkey is visible under `/mnt/c`):
```bash
# WSL, once nouslogic-sl2619 key-auth works
cat /mnt/c/Users/<you>/.ssh/sl2619_nouslogic.pub | ssh nouslogic-sl2619 'cat >> /home/root/.ssh/authorized_keys && chmod 600 /home/root/.ssh/authorized_keys'
```

### 4.5 Restart Dropbear and test
```sh
# board
systemctl reboot      # picks up the ed25519 host key + hostname + Wi-Fi + TZ all at once
```
After ~30 s, from the host:
```bash
# host
ssh nouslogic hostname        # Windows   (WSL: ssh nouslogic-sl2619 hostname) → nouslogic
```
> If it falls back to a password prompt for **root**, the key landed in
> `/root/.ssh/` instead of `/home/root/.ssh/` — re-do §4.3. If it prompts for a
> **different user** (`phl@…`), your `~/.ssh/config` block didn't match — see the
> exact-alias note in §4.4.

### 4.6 (Optional) Stop the passphrase prompt — load the key into the agent
**WSL** (per session):
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/sl2619_nouslogic_wsl
```
**Windows PowerShell** (`ssh-agent` persists once started):
```powershell
Start-Service ssh-agent                              # run once, as admin
ssh-add $env:USERPROFILE\.ssh\sl2619_nouslogic       # enter passphrase once
```
After this, `ssh nouslogic` connects with no prompt at all. (Or strip the
passphrase entirely: `ssh-keygen -p -f <keyfile>`, leave empty.)

---

## 5. Final verification (after the §4.5 reboot)

```bash
# host
ssh nouslogic 'hostname; \
  iw dev wlan0 link | head -1; \
  ip addr show wlan0 | awk "/inet /{print \"IPv4:\",\$2}"; \
  timedatectl status | grep -E "synchronized|NTP"; \
  date; \
  mount | grep sdcard; \
  uname -a'
```
Expect: `nouslogic`, associated to `YOUR_SSID` with an IPv4 lease, NTP synced,
local time correct, `/mnt/sdcard` mounted, kernel 6.12.x. Bring-up complete.

Then re-baseline the board snapshot:
```
/board_probe          # regenerates docs/tmp/sl2619-status.md, checks Iron Laws
```

---

## 6. Troubleshooting (condensed)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` after reflash | Stale `known_hosts` entry from the pre-reflash host key (not a MITM — verify the fingerprint matches §4.1) | `ssh-keygen -R <BOARD_IP>` (+ `-R nouslogic`, `-R nouslogic.local`) on both hosts |
| `Permission denied`, prompt shows **`phl@…`** (your local user, not root) | `ssh nouslogic` didn't match `Host nouslogic-sl2619` — SSH does no partial matching | Put both names on one `Host` line and set `User root` (§4.4) |
| `No route to host` to the IP you "noted" | You used the one-shot `udhcpc` lease, not the live `systemd-networkd` one | Use the address from `networkctl status wlan0` (§2.4) |
| Windows asks for password but key was added; WSL works | Windows `~/.ssh/config` missing `User root`/`IdentityFile`, or key not installed for Windows | Fix config (§4.4); install the Windows key via the WSL pipe shortcut (§4.4) |
| `Enter passphrase for key …` every connect | Private-key passphrase (not the board password) | Load into agent (§4.6) or strip with `ssh-keygen -p` |
| SSH still asks for password, Dropbear log `Exit before auth … 0 fails` | `authorized_keys` at `/root/.ssh/` not `/home/root/.ssh/` | Move it: `mkdir -p /home/root/.ssh && cp /root/.ssh/authorized_keys /home/root/.ssh/ && chmod 700 /home/root/.ssh && chmod 600 /home/root/.ssh/authorized_keys` |
| SSH asks for password despite key present | Dropbear has only RSA host key | Complete §4.1 + reboot |
| `authorized_keys` rejected silently | CRLF line endings from a Windows tool | `tr -d '\r' < … > /tmp/ak && mv /tmp/ak …` |
| `iw dev wlan0 link` → Not connected | nl80211 path | `killall wpa_supplicant; wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf -Dwext` |
| After reboot no IP but associated | `systemd-networkd` off or `.network` malformed | `systemctl is-enabled systemd-networkd`; strip leading whitespace from `25-wlan0.network` |
| `timedatectl set-timezone …` → "Invalid or not installed time zone" | No `tzdata` in image | Use the `TZ` env-var approach (§3) |
| Clock resets to build date after reboot | NTP not enabled | `systemctl is-enabled systemd-timesyncd` must be `enabled` |
| `/mnt/sdcard` empty after reboot | fstab entry missing or card unformatted | §1 fstab line; reformat to ext4 if needed |
| `ping` by hostname fails, by IP works | DNS not handed out | check `/etc/resolv.conf` → `/run/systemd/resolve/resolv.conf`; `systemctl status systemd-resolved` |

---

## 7. As-executed record — 2026-06-01

The bring-up that proved this runbook, with real values. **Substitute your own.**

| Item | Value (this board) |
| --- | --- |
| Board IP (Wi-Fi, `systemd-networkd` lease) | **`192.168.12.240`** (one-shot `udhcpc` had transiently shown `.239` — ignore) |
| SSID | `Kylie` (5 GHz, `e8:9f:80:2f:fa:b0`) |
| wlan0 MAC (for DHCP reservation) | `9c:b8:b4:3c:27:7a` (AMPAK) |
| Hostname | `nouslogic` |
| Board ed25519 host-key fingerprint | `SHA256:7+YxbZ+v3WNCHA7dgIDMOnEvIAB/fVW6EuJrptOciQM` |
| Kernel / distro | `6.12.62` / Poky `5.0.9 (scarthgap)` |
| microSD contents (intact) | `models/`, `fixtures/`, `bench*`, `llama-cpp/`, `dispenser_demo/`, … |
| SSH keys | Windows `~/.ssh/sl2619_nouslogic`, WSL `~/.ssh/sl2619_nouslogic_wsl` (both passphrase-protected, both in board `/home/root/.ssh/authorized_keys`) |

What went smoothly: microSD mount + fstab, password, hostname, Wi-Fi (one-shot →
persistent `routable/online`), NTP sync, `TZ` → `15:17 PDT`, ed25519 host key.

What tripped us (now folded into §4 / §6): (1) used the stale `.239` lease at
first — real IP was the `networkctl` `.240`; (2) `REMOTE HOST IDENTIFICATION
CHANGED` on both hosts from the pre-reflash key — cleared with `ssh-keygen -R`;
(3) `ssh nouslogic` matched no `Host` block (SSH has no partial matching) →
logged in as `phl` and failed — fixed by `Host nouslogic nouslogic-sl2619` +
`User root`; (4) confused the **key passphrase** with the **board password**;
(5) installed the Windows key painlessly by piping `/mnt/c/.../*.pub` through the
already-working WSL session.

**Still TODO before this board is fully "done":** add a router DHCP reservation
for `9c:b8:b4:3c:27:7a` → `192.168.12.240` so the SSH aliases survive reboots.

---

## 8. Next: BLE / Bluetooth bring-up from the M.2 daughter card

The board is now recovered + networked + SSH-reachable — the platform for
resuming **BLE bring-up**, the original v1 dispenser-demo blocker (Synaptics bug
37861/37374, the **revB pin-mux** failure). **Good news as of the v2.4.0 flash:**
a read-only probe (2026-06-01) shows **`hci0` now enumerates** (UART, SYN43711
combo) — the revB pin-mux fix ships in v2.4 boot firmware. Full bring-up
(adapter UP + BLE notify) and the runbook now live in a dedicated doc:
**[`sl2619-ble-bringup.md`](sl2619-ble-bringup.md)** — start there.

> **This is exactly the work that bricked the board on 2026-06-01** (a blind
> `board_patched.dtb` write into the eMMC boot partitions). Do NOT repeat that.
> v2.4.0 already carries the fix, so **no device-tree surgery is needed.** If a
> DTS change ever *is* required, use a supported route for the UART / SYN43711
> (M.2 Broadcom BT-on-UART, Wi-Fi-on-SDIO) device tree:

- **Yocto rebuild** — patch the DTS in the kernel/device-tree recipe so the build
  repacks **and re-signs** the FIT boot image, then reflash via
  [`sl2619-windows-recovery.md`](sl2619-windows-recovery.md). Never `dd`/`seek`
  into `mmcblk0pN`.
- **U-Boot overlay** — `fdt apply` / `uEnv.txt` so a bad node can't corrupt the
  signed on-disk image.

Context: BLE status + bug refs in `docs/plans/dispenser-demo/decisions-log.md`;
root-cause of the brick in
[`sl2619-recovery-reflash.md`](sl2619-recovery-reflash.md) §root-cause; the M.2
BT-on-SDIO1 hardware note in the `synaptics-references` memory.

---

## References
- Verified source: `references/upstream/synaptic-sl2619/docs/get-started/sl2610-get-started.md` §7 (Wi-Fi/clock), §8 (hostname/SSH), §9 (diagnostics)
- Wi-Fi helper script (optional, for later SSID switches): get-started §7.6 (`wifi-connect`)
- microSD ext4 format (if reformatting needed): `references/upstream/synaptic-sl2619/docs/get-started/sd-card-ext4-format-get-started.md`
- Recovery (if you need to reflash): [`sl2619-windows-recovery.md`](sl2619-windows-recovery.md)
