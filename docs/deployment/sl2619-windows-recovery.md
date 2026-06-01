# SL2619 Windows recovery — step-by-step eMMC reflash (Scarthgap 6.12 v2.4.0)

Complete, self-contained procedure to recover a bricked **Synaptics Astra
Machina SL2619** (SL2600-series, Cortex-A55 ×2) from a **Windows** host, after a
bad direct write to `/dev/mmcblk0` left it unreachable over SSH and adb.

> ✅ **Executed end-to-end and verified on this board 2026-06-01** with the
> `sl2619_scarthgap_6.12_v2.4.0.zip` image → board recovered to kernel
> `6.12.62` / Poky `5.0.9 (scarthgap)`, `adb devices` → `sl2619 device`. The
> as-run transcript and deltas vs. the original (v2.3.0-derived) steps are in
> §11. This runbook reflects the **proven v2.4.0 path**.

This is the **hands-on companion** to the decision/overview doc
[`sl2619-recovery-reflash.md`](sl2619-recovery-reflash.md) (read that for the
Linux-vs-Windows tool choice and the root-cause analysis). **Follow top to bottom.**

> **Why Windows for this recovery?** The `usb_boot_tool.py` (`sl261x` branch)
> path is the one **verified working on this exact board** and needs **no build
> step** — lowest risk. `astra-update` *does* now support SL26xx (commit
> [`f7a3cdd`](https://github.com/synaptics-astra/astra-update/commit/f7a3cddaee22c481983b35ca4fc52fa1016bea39)
> adds `AstraDeviceSL26XXImpl` + multi-stage boot + `--chip sl2610`), so flashing
> from **native Linux** is also viable — but you must **build it from current
> source** (the vendored v1.0.6 binary predates that commit). If you'd rather not
> compile, Windows + `usb_boot_tool.py` is the path documented here. See
> [`sl2619-recovery-reflash.md`](sl2619-recovery-reflash.md) §3 for the Linux
> build path. **Do NOT flash from WSL** — the board re-enumerates several times
> mid-flash and `usbipd-win` drops the device on each reset.

---

## 0. Before you start — the board is recoverable, your data is safe

- **Not bricked permanently.** USB-boot mode lives in the SoC **mask ROM
  (eROM)**. Nothing written to `mmcblk0` (GPT, boot, rootfs) can touch it.
  `USB_BOOT` held across a `RESET` edge forces USB download mode regardless of
  eMMC state, and the flash rewrites the GPT first — even a trashed partition
  table recovers.
- **Your models/fixtures survive.** They live on the **microSD** (`mmcblk2p1`),
  a different device from the **eMMC** (`mmcblk0`) you're reflashing — untouched
  even by the user-partition wipe in this procedure. Re-mount after boot:
  `mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard`.

---

## 1. What you need

### Hardware
| Item | Notes |
| --- | --- |
| SL2619 dev kit (dock + core module) | — |
| DC supply shipped with the kit | Into the dock's **`PWR_IN`** port — **not** USB bus power. Leave it plugged in for the whole flash. |
| USB-C data cable | Host ↔ dock's **USB Boot / CDC (USB2.0)** port (not the UART/debug port). |
| Windows 10/11 host | Native Windows — **not WSL**. |

> **DDR type on this kit is `ddr4`** (2 GB, DDR4-3200). Do **not** use
> `ddr4-1x16`. This is load-bearing — wrong value fails the SM upload.

### Software on the host
1. **Python 3.12+** — `python` must resolve to it from a fresh terminal.
2. **pyserial** — `python -m pip install pyserial` (often pre-bundled).
3. **Android Platform Tools (`adb`)** — <https://developer.android.com/tools/adb>, on `PATH`.
4. **Git** — to clone the usb-tool repo.

### Downloads
1. **`usb_boot_tool.py`** — the **`sl261x` branch** of the Synaptics usb-tool repo.
2. **SL2619 v2.4.0 image bundle** — confirmed asset (GitHub API + `curl -IL`):
   - **`sl2619_scarthgap_6.12_v2.4.0.zip`** ← the one you want.
   - From: <https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0>
   - All assets are **`.zip`** (not `.tar.gz`). **Do NOT** grab `nand`,
     `spi_boot`, `usb_boot`, `coralboard`, or `oobe` variants.
   - **Note:** the doc site `v/latest/` still shows v2.3.0 — the **GitHub release
     tag is authoritative** for v2.4.0.

Pick a working dir. This guide uses `D:\sl2619-recovery\` (the prior setup used
`D:\3-Nouslogic\robotic-arm-hand\`); substitute your own and keep it consistent.

---

## 2. Host setup (one-time)

### 2.1 Verify Python + pyserial
```powershell
where python
python --version                                  # expect 3.12+
python -c "import serial; print(serial.__version__)"
```
**PATH gotcha:** System `PATH` is searched before User `PATH`. If `where python`
resolves to the wrong interpreter, fix PATH ordering and **open a new terminal**
(PATH is cached per-process).

### 2.2 Verify adb
```powershell
adb version                                       # expect 1.0.41+
```

### 2.3 Clone the usb-tool repo, sl261x branch
```powershell
cd D:\sl2619-recovery
git clone <usb-tool-repo-url> usb-tool
cd usb-tool
git checkout sl261x                               # contains usb_boot_tool.py
```
If you still have the prior checkout (`D:\3-Nouslogic\robotic-arm-hand\usb-tool\`
on branch `sl261x`), reuse it — just confirm the branch:
```powershell
git -C D:\3-Nouslogic\robotic-arm-hand\usb-tool branch --show-current   # expect: sl261x
```

### 2.4 Download and unpack the image
Download `sl2619_scarthgap_6.12_v2.4.0.zip` and unpack to `D:\sl2619-recovery\eMMCimg\`.
After unpacking you should see (the **Astra System Image** directory):
```
emmc_part_list              # GPT partition table
emmc_image_list             # incremental image-flash script
emmc_image_list_full        # full variant (wipes user partition)
preboot.subimg[.gz]
key / tzk / sysmgr / bl / boot / firmware  .subimg[.gz]
rootfs.subimg.gz   +  rootfs_s.subimg.0 / .1 / .2
home.subimg.gz / home_s.subimg
TAG--astra-media-sl2619...--TAG       # build provenance marker
```
`gpt.bin` is **generated at runtime** by `usb_boot_tool.py` from `emmc_part_list`
— if the bundle doesn't ship one, that's by design.

### 2.5 Activate the FULL image list (recovery = clean slate)
For a corruption recovery you want the clean-slate variant, which adds the
`format,sd1` step that wipes the user partition (harmless — your microSD is a
separate device, §0):
```powershell
cd D:\sl2619-recovery\eMMCimg
ren emmc_image_list       emmc_image_list.reflash
ren emmc_image_list_full  emmc_image_list
```

### 2.6 (Recommended) Diff the v2.4.0 file list
The verified sequence was captured against v2.3.0. Before flashing, eyeball the
unpacked v2.4.0 directory against the expected set above. If `run-sm` (§5.2)
later fails, the version-matched SM/BL bootstrap binaries live inside the
`sl2619_usb_boot_scarthgap_6.12_v2.4.0.zip` asset — pull them from there.

---

## 3. Cable and power
1. Plug the kit's DC supply into the dock's **`PWR_IN`**. Power LED on.
2. Plug USB-C from the Windows host into the dock's **USB Boot / CDC** port.
3. **Do not unplug `PWR_IN` during flashing.** USB-C re-enumeration across the
   flash is normal and expected; the board survives it because power stays on.

---

## 4. Enter USB Boot mode

Must be done **each time** you want the board in USB Boot mode (leaving the mode
ends the session). Two equivalent button orderings — use whichever enumerates
the device:

- **Get-started ordering:** press & hold `USB_BOOT` → tap `RESET` (press+release)
  → wait ~2 s → release `USB_BOOT`.
- **Official SL2600 guide (NR-160458) ordering:** hold `RESET` → press & hold
  `USB_BOOT` (still holding RESET) 1–2 s → release `RESET` (keep holding
  `USB_BOOT`) until console prints → release `USB_BOOT`.

Both just assert `USB_BOOT` across the reset edge.

**Verify (use `-PresentOnly` — critical):**
```powershell
Get-PnpDevice -Class Ports -PresentOnly | Where-Object { $_.InstanceId -like "*VID_06CB*PID_019E*" }
```
- Non-empty → board is in USB Boot mode (the eROM CDC `06CB:019E` is live).
- Empty → not in USB Boot mode; redo the sequence (first 1–2 tries often miss).

> **Ghost PnP caveat:** without `-PresentOnly`, Windows shows stale entries
> (`Present=False`, `Status=Unknown`) that look identical to a live port and
> will fool you into thinking the tool is broken. Always filter with
> `-PresentOnly`. Raw view:
> ```powershell
> Get-PnpDevice -Class Ports | ? { $_.InstanceId -like "*VID_06CB*" -or $_.InstanceId -like "*VID_CAFE*" } | Format-Table FriendlyName,Present,Status,InstanceId -AutoSize
> ```

---

## 5. Flash

Open a **new** terminal at the usb-tool dir. Do **not** close it or unplug USB-C
between the two commands.

```powershell
cd D:\sl2619-recovery\usb-tool
```

### 5.1 Confirm USB Boot mode
Confirm `06CB:019E` is **Present** (§4 verify command).

### 5.2 Run System Manager (SM) in RAM
```powershell
python usb_boot_tool.py --op run-sm --ddr-type ddr4
```
Expected (actual v2.4.0 output, abbreviated):
```
Auto-detecting VID:0x06CB, PID:0x019E serial port...
Syna USB CDC port detected: COM5
 ✔   key.bin UPLOADED      ✔   spk.bin UPLOADED      ✔   m52bl.bin UPLOADED
Auto-detecting VID:0x06CB, PID:0x019E serial port...
Syna USB CDC port detected: COM8           # SPK/BL stage hops to a new COM
[INFO] Uploading sysmgr.subimg.gz   ######## 100.0% Complete
[INFO] Sending RUN (0x0B)...
[INFO] SM started successfully.
```
The tool **auto-detects the COM port** at each stage — you never pass one. The
CDC then re-enumerates as `VID_CAFE&PID_4002` (SM CDC); `06CB:019E` goes away.
Confirm:
```powershell
Get-PnpDevice -Class Ports -PresentOnly | Where-Object { $_.InstanceId -like "*VID_CAFE*PID_4002*" }
```

### 5.3 Flash eMMC
Same terminal. **Pass the absolute path to your `eMMCimg`** (see the gotcha
below):
```powershell
python usb_boot_tool.py --op emmc --img-dir D:\0-Nouslogic\sl2619-recovery\eMMCimg
```

> **⚠️ Two benign-looking things that are actually fine (seen in the 2026-06-01 run):**
> 1. The command **starts** by printing
>    `[ERROR] SPK Port not found` and `[ERROR] Port not found for Run-SM`. This is
>    **expected** — it re-probes for the eROM `06CB:019E`, which is gone because
>    SM is already running from §5.2, then **falls back to the live SM CDC
>    (`CAFE:4002`)** and flashes correctly. Not a failure.
> 2. If you instead see `[ERROR] Partition list not found: ...\emmc_part_list`
>    and it stops, your `--img-dir` path is **wrong** — fix it to the real
>    absolute path of the unpacked `eMMCimg` dir and re-run. (This bit the
>    2026-06-01 run once: `D:\sl2619-recovery\...` vs the real
>    `D:\0-Nouslogic\sl2619-recovery\...`.)

What the tool does, in order (actual v2.4.0 partition map):
1. Builds `gpt.bin` in-memory from `emmc_part_list`, writes it into the img-dir.
2. `[INFO] Flashing GPT`.
3. `preboot.subimg.gz` → `preboot_a` (`b1`) + `preboot_b` (`b2`).
4. `key`/`tzk` → `key_a`/`tzk_a` (`sd2`/`sd3`), `key_b`/`tzk_b` (`sd4`/`sd5`).
5. `sysmgr`/`bl` → `sysmgr_a`/`bl_a` (`sd6`/`sd7`), `sysmgr_b`/`bl_b` (`sd8`/`sd9`).
6. `boot.subimg.gz` → `boot_a` (`sd10`) + `boot_b` (`sd11`).
7. Split rootfs `rootfs_s.subimg.0/1/2` → `rootfs_a` (`sd12`) + `rootfs_b` (`sd13`).
8. `fastlogo.subimg.gz` → `fastlogo_a` (`sd14`) + `fastlogo_b` (`sd15`).
9. Erases `devinfo` (`sd16`) + `misc` (`sd17`).
10. `home_s.subimg` → `home` (`sd18`).

> Note: this v2.4.0 layout differs from the older v2.3.0 get-started doc — rootfs
> is at `sd12/sd13` (not `sd14/sd15`) and there's a `fastlogo` partition at
> `sd14/sd15`. The `_full` image list's user-partition format is folded into this
> sequence; `firmware` is not a separate step in this bundle.

**Wall-clock ~5–15 min** (dominant cost: ~900 MB rootfs to both A and B slots).
**Do not interrupt or unplug.** Success ends with `=== ALL OPERATIONS COMPLETE ===`.

---

## 6. Verify recovery

> **Before you reset: `adb devices` being empty right after the flash is
> NORMAL.** The board never left flashing/SM mode — it does **not** auto-boot the
> new image. You must cold-boot it (§6.1). Don't panic at the empty list.

### 6.1 Cold boot from eMMC
Tap `RESET` once. Do **not** touch `USB_BOOT`. Leave the USB-C cable plugged in.
The board boots ROM → BL → SM → A-core → Linux from eMMC. **Wait ~30 s.**

### 6.2 Enumeration
Confirm the flashing-mode CDCs are gone:
```powershell
Get-PnpDevice -Class Ports -PresentOnly | ? { $_.InstanceId -like "*VID_CAFE*PID_4002*" -or $_.InstanceId -like "*VID_06CB*" }
```
After a good boot this is **empty** — the board now presents an **ADB composite**
device, which appears under *Android Device* / *Universal Serial Bus devices* in
Device Manager, **not** under Ports.

### 6.3 Confirm Linux userspace
**Bounce the adb server first** so it re-scans the freshly-enumerated device
(this was needed in the 2026-06-01 run — `adb devices` was empty until the
restart):
```powershell
adb kill-server
adb start-server
adb devices
```
Expected:
```
List of devices attached
sl2619    device
```
(Not `offline`, not empty. If `offline`, wait 20–30 s for init, then
`adb kill-server; adb start-server; adb devices`.)

```powershell
adb shell "uname -a; cat /etc/os-release; exit"
```
Expected (actual 2026-06-01 result on the v2.4.0 image):
```
Linux sl2619 6.12.62 #1 SMP PREEMPT Wed May 27 14:31:51 UTC 2026 aarch64 GNU/Linux
PRETTY_NAME="Poky (Yocto Project Reference Distro) 5.0.9 (scarthgap)"
```

### 6.4 Persistence
```powershell
adb shell "sync && echo OK"
adb reboot
# wait ~30 s
adb devices                    # still: sl2619    device  → durable, recovered
```

---

## 7. Flash troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `--op emmc` opens with `[ERROR] SPK Port not found` / `Port not found for Run-SM` then proceeds | **Benign** — SM already running, tool re-probes the gone eROM then falls back to `CAFE:4002` | None — let it run (confirmed harmless 2026-06-01) |
| `--op emmc` stops at `[ERROR] Partition list not found: …\emmc_part_list` | `--img-dir` path wrong | Pass the **correct absolute path** to the unpacked `eMMCimg` and re-run |
| `adb devices` empty **right after flash** (before reset) | Board still in SM mode — never cold-booted | **Normal.** Tap `RESET` (§6.1), wait 30 s, then `adb kill-server; adb start-server; adb devices` |
| `adb devices` empty **after** reset + 30 s | adb server hasn't re-scanned | `adb kill-server; adb start-server; adb devices` (this was needed in the 2026-06-01 run) |
| No new COM port after button sequence | Not actually in USB Boot mode | Redo §4; try 2–3 times |
| `06CB:019E` shows but `run-sm` says "Port not found" | Ghost PnP entry (`Present=False`) | Rerun with `-PresentOnly`; if empty, redo §4 |
| `run-sm` hangs at "Waiting for SM CDC" | Wrong DDR, SPK/BL mismatch, or wrong bundle | Confirm `--ddr-type ddr4` (not `ddr4-1x16`) + correct bundle; redo from §4 |
| `run-sm` fails right after SPK upload | Wrong DDR type | Confirm `ddr4`; retry |
| `emmc` errors "No such file or directory" | Ran from wrong directory | `cd` to the usb-tool dir first |
| `emmc` errors on first command | `CAFE:4002` not enumerated yet | Wait 5 s, retry `emmc`; if persistent, redo §5.2 |
| `emmc` aborts mid-flash with USB disconnect | Cable/host-port glitch (or you're on WSL — don't) | Swap USB-C cable or host port; flash from native Windows; redo from §4 |
| After cold boot, re-enumerates as `06CB:019E` | Flash did not persist | Redo full §5 |
| After cold boot, only `CAFE:4002` | SM runs but A-core didn't boot | Wrong DDR or bundle mismatch; redo with `ddr4` confirmed |
| After cold boot, nothing enumerates | Weird state | Hold `RESET` 5 s, release; retry §6.1 |
| `adb devices` empty after boot | ADB not up yet | Wait 30 s; `adb kill-server && adb start-server && adb devices` |
| `python` resolves to old version after PATH edit | Cached PATH / System-over-User PATH | Open a **new** terminal; recheck §2.1 |

---

## 8. Post-recovery — bring the board back into the project

Once Linux is up and `adb`/SSH reachable, restore the working state. The shipped
image is bare (passwordless root, hostname `sl2619`). **Full step-by-step
procedure:** [`sl2619-postrecovery-bringup.md`](sl2619-postrecovery-bringup.md) —
persistent Wi-Fi, hostname `nouslogic`, SSH keys, NTP/timezone, microSD re-mount,
in dependency order with the board-specific traps. Summary:

- **microSD models:** `mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard` (+ fstab for
  persistence). Intact — verify SHAs against
  `releases/functiongemma-270m/001-baseline/gguf/CHECKSUMS.txt`.
- **Wi-Fi (persistent):** `wpa_supplicant@wlan0` + `systemd-networkd`.
- **Hostname + SSH keys:** ed25519 host key required; `authorized_keys` at
  **`/home/root/.ssh/`** not `/root`.
- **Clock/timezone:** no RTC → NTP; no `tzdata` → `TZ` env var.
- **Re-baseline:** run `/board_probe` to regenerate `docs/tmp/sl2619-status.md`.

---

## 9. Do NOT re-brick — the lesson from this incident

The board was bricked by writing a recompiled `board_patched.dtb` directly into
`/dev/mmcblk0p10` + `p11` (the redundant A/B **boot partitions**) at a hard-coded
offset. Those partitions are **packaged, hash-verified, secure-boot-signed FIT
images** — not flat files. The in-place write invalidated the FIT hash, broke the
secure-boot signature, overran the fixed DTB slot (the patched DTB was larger),
and destroyed both A and B copies at once (killing the redundancy). Full
root-cause analysis: [`sl2619-recovery-reflash.md`](sl2619-recovery-reflash.md) §root-cause.

**Never `seek`+`write` a `.dtb` into `mmcblk0pN` again.** To change the device
tree, use a supported path:
- **Rebuild in Yocto** — patch the DTS in the kernel/device-tree recipe, let the
  build repack *and re-sign* the FIT, then reflash via this same procedure.
- **Runtime DT overlay** — `fdt apply` from U-Boot / `uEnv.txt`, so the signed
  on-disk boot image is never mutated and a bad node can't corrupt it.

BLE bring-up context (Synaptics bug 37861/37374):
`docs/plans/dispenser-demo/decisions-log.md`.

---

## 10. Condensed cheat-sheet

```powershell
# 0. Host: Python + pyserial + adb on PATH
where python; python --version; python -c "import serial; print(serial.__version__)"; adb version

# 1. usb-tool on sl261x branch; eMMCimg unpacked from sl2619_scarthgap_6.12_v2.4.0.zip
cd D:\sl2619-recovery\eMMCimg
ren emmc_image_list       emmc_image_list.reflash      # clean-slate recovery
ren emmc_image_list_full  emmc_image_list

# 2. Cable: dock PWR_IN to DC; host USB-C to dock USB-Boot port.

# 3. USB Boot mode: hold USB_BOOT, tap RESET, wait 2 s, release. Verify:
Get-PnpDevice -Class Ports -PresentOnly | ? { $_.InstanceId -like "*VID_06CB*PID_019E*" }

# 4. Flash:
cd D:\sl2619-recovery\usb-tool
python usb_boot_tool.py --op run-sm --ddr-type ddr4
python usb_boot_tool.py --op emmc   --img-dir <ABSOLUTE path to eMMCimg>   # benign SPK errors at start are OK

# 5. Cold boot: tap RESET only (NOT USB_BOOT). Wait ~30 s. Then:
adb kill-server; adb start-server; adb devices    # expect: sl2619    device
adb shell "uname -a; cat /etc/os-release; exit"
adb shell "sync && echo OK"; adb reboot           # persistence check
```

---

## 11. As-executed run log — 2026-06-01 (v2.4.0, recovery from the DTB-write brick)

Real transcript of the recovery that proved this runbook. **Host:** Windows
10.0.22635, Python 3.14.4, adb 1.0.41 (36.0.2). **Working dir:**
`D:\0-Nouslogic\sl2619-recovery\`. **Image:** `sl2619_scarthgap_6.12_v2.4.0.zip`.

What happened, in order:
1. `git clone https://github.com/synaptics-astra/usb-tool` → `git checkout sl261x`
   (the `tree/sl261x#…` URL form fails — clone the bare repo, then checkout).
2. Renamed `emmc_image_list_full` → `emmc_image_list` (clean-slate).
3. USB-boot button sequence → `06CB:019E` enumerated as **COM5** (verified with
   `Get-PnpDevice … -PresentOnly`).
4. `--op run-sm --ddr-type ddr4` → uploaded key/spk/m52bl, re-probed to **COM8**,
   uploaded `sysmgr.subimg.gz`, `SM started successfully`. SM CDC came up as
   `CAFE:4002` / **COM9**.
5. **First `--op emmc` failed** with `Partition list not found:
   D:\sl2619-recovery\eMMCimg\emmc_part_list` — **wrong path** (missing
   `0-Nouslogic`). Re-ran with the correct absolute path
   `D:\0-Nouslogic\sl2619-recovery\eMMCimg`.
6. Second `--op emmc` opened with the **benign** `[ERROR] SPK Port not found` /
   `Port not found for Run-SM`, fell back to `CAFE:4002`, and flashed GPT →
   preboot(b1/b2) → key/tzk/sysmgr/bl(A/B) → boot(A/B sd10/sd11) → rootfs
   (sd12/sd13) → fastlogo(sd14/sd15) → erase devinfo/misc → home(sd18) →
   `=== ALL OPERATIONS COMPLETE ===`.
7. `adb devices` **empty** at this point (board still in SM mode — expected).
8. **Tapped `RESET`** (not USB_BOOT). After boot, `CAFE:4002` gone;
   `adb devices` still empty until **`adb kill-server; adb start-server`**, then:
   ```
   sl2619    device
   Linux sl2619 6.12.62 #1 SMP PREEMPT Wed May 27 14:31:51 UTC 2026 aarch64 GNU/Linux
   PRETTY_NAME="Poky (Yocto Project Reference Distro) 5.0.9 (scarthgap)"
   ```
   **Recovered.**

Deltas folded into the runbook from this run: absolute-path requirement for
`--img-dir` (§5.3), benign opening errors on `--op emmc` (§5.3 / §7), the
`adb kill-server/start-server` step after cold boot (§6.3 / §7), and the actual
v2.4.0 partition map (§5.3).

---

## References
- Verified flash sequence (source of truth): `references/upstream/synaptic-sl2619/docs/get-started/sl2610-get-started.md` §2–6, §9
- Recovery overview + root cause + Linux path: [`sl2619-recovery-reflash.md`](sl2619-recovery-reflash.md)
- v2.4.0 release (authoritative for assets): <https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0>
- SL2600 HW guide (button sequence, straps): <https://cp.synaptics.com/cognidox/download/NR-160458-MS-APPROVED.pdf>
- Astra SDK docs: <https://synaptics-astra.github.io/doc/v/latest/>
