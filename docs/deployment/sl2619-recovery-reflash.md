# SL2619 eMMC recovery — fresh reflash (Scarthgap 6.12 v2.4.0)

Recovery runbook for an SL2619 board that no longer boots / is unreachable over
SSH **or** adb after a bad eMMC write. Triggered 2026-06-01 by writing a patched
`board_patched.dtb` directly into `/dev/mmcblk0p10` + `/dev/mmcblk0p11` at a
hard-coded offset (`0x1178258`) during the BLE DTS bring-up — this clobbered the
boot/dtb subimages in the A/B slots, so BL/SM/A-core can no longer hand off to
Linux.

> **This doc is the overview / decision layer:** what's recoverable, the
> Linux-vs-Windows tool choice, the WSL caveat, and the root-cause analysis.
> **For the full hands-on Windows procedure**, follow
> [`sl2619-windows-recovery.md`](sl2619-windows-recovery.md) (self-contained,
> copy-pasteable). The upstream verified sequence is §2–6/§9 of
> [`sl2610-get-started.md`](../references/upstream/synaptic-sl2619/docs/get-started/sl2610-get-started.md).

---

## 0. TL;DR — the board is NOT bricked

USB-boot mode lives in the SoC **mask ROM (eROM)**. Nothing you wrote to
`mmcblk0` — GPT, boot partitions, rootfs, anything — can touch it. Holding
`USB_BOOT` while tapping `RESET` forces the eROM into USB download mode
regardless of eMMC state, and the `emmc` flash **rewrites the GPT first**, so
even a trashed partition table recovers. A full reflash returns the board to a
known-good Scarthgap image.

**Your `/mnt/sdcard` data is safe.** Models and fixtures live on the **microSD**
(`mmcblk2p1`), a physically different device from the **eMMC** (`mmcblk0`) you
are reflashing. They survive untouched — even with the user-partition wipe in
§2 below. (Re-mount after boot: `mount -t ext4 /dev/mmcblk2p1 /mnt/sdcard`.)

---

## 1. Flash from a native host (Linux OR Windows) — NOT from WSL

`adb` is installed and works in this WSL, but **WSL is the wrong host for the
flash step.** During flashing the device re-enumerates multiple times —
`06CB:019E` (eROM) → SM CDC → ADB composite. `usbipd-win` auto-detaches the
device from WSL on every re-enumeration and does **not** reliably re-attach
(confirmed against Microsoft Learn + usbipd-win wiki + issues #703/#970): a
mid-flash reset is functionally an unplug that drops the device out of WSL,
breaking the multi-stage flash. So:

- **Flash → native Linux _or_ native Windows.** Both are officially supported
  (see §3). `astra-update` runs natively on Linux — no WinUSB driver, just
  `libudev` + the shipped `99-astra-update.rules` udev file — so a **bare-metal
  Linux box (or dual-boot), not WSL**, is the cleanest path for you. Windows is
  the fallback (and the path previously verified on this board with
  `usb_boot_tool.py`).
- **adb/ssh after boot → WSL is fine.** Once Linux is back up the ADB device is
  stable; `usbipd attach --wsl --busid <id>` then `adb devices` works, or just
  use SSH over Wi-Fi/Ethernet as the project normally does (`nouslogic-sl2619`).

---

## 2. What's different for a *corruption* recovery

The get-started guide is written for first-flash. For this recovery, two deltas:

1. **Use the `_full` image-list variant** (per get-started §2.5):
   `emmc_image_list_full` adds the `format,sd1` step that wipes the user
   partition — you want the clean slate after a blind eMMC write. The microSD is
   untouched (see §0), so this costs you nothing.
   ```
   ren emmc_image_list       emmc_image_list.reflash
   ren emmc_image_list_full  emmc_image_list
   ```
2. **DDR type is `ddr4`** on this kit — `--ddr-type ddr4` (the value the
   board-verified get-started run used). The dev kit is confirmed **DDR4**
   (2 GB, DDR4-3200), but the exact token (`ddr4` vs `ddr4-1x16`, which tracks
   1×16Gbit-2GB vs 2×x8-4GB population) is **not pinned to a primary source** —
   confirm via `astra-update --help` / the bundle's image config on your host,
   or sidestep it: the `update_emmc.sh` wrapper and the `sl2619_usb_boot` bundle
   encode the board default, so you can often omit the flag entirely.

---

## 3. Tooling — two viable paths

| Tool | Host | SL2619 support |
| --- | --- | --- |
| **`usb_boot_tool.py`** (`usb-tool` repo, **`sl261x` branch**) | Windows (Python + pyserial) | ✅ **Board-verified 2026-04-15.** The path the get-started guide uses end-to-end. Lowest risk; no build step. |
| **`astra-update`** (build from source ≥ commit `f7a3cdd`) | **Linux** / Mac / Windows | ✅ Officially "preferred" tool, runs natively on Linux (no WinUSB driver). **SL26xx support confirmed** — commit [`f7a3cdd`](https://github.com/synaptics-astra/astra-update/commit/f7a3cddaee22c481983b35ca4fc52fa1016bea39) adds `AstraDeviceSL26XXImpl`, the multi-stage boot (`run-sm` equivalent), eMMC/NAND flash, fastboot, and the `06cb:019e`/`cafe:4002`/`18d1:4ee0` VID/PIDs; chip id `sl2610`, board `rdk`. ⚠️ The **vendored binary is v1.0.6 and predates this commit** — it ships SL16xx boot images only. You must **build from current source** (or use a release that includes `f7a3cdd`). |

**Two genuine choices now** (the SL26xx commit makes native Linux first-class):

- **Native Linux via `astra-update` (build from source).** No Windows, no WinUSB:
  ```bash
  git clone https://github.com/synaptics-astra/astra-update.git
  cd astra-update
  sudo apt install libudev-dev cmake
  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config release
  sudo cp config/99-astra-update.rules /etc/udev/rules.d/   # README says /etc/systemd/system; udev rules belong in /etc/udev/rules.d — verify on your distro
  sudo udevadm control --reload-rules && sudo udevadm trigger
  # unzip sl2619_scarthgap_6.12_v2.4.0.zip → eMMCimg/, board in USB-boot mode, then:
  ./build/.../astra-update --chip sl2610 --board rdk --flash ./eMMCimg
  ```
  **Must be a bare-metal Linux box, not WSL** (re-enumeration, §1). Verify the
  built binary exposes SL26xx: `astra-update --help` should accept `--chip sl2610`.
- **Windows via `usb_boot_tool.py` (board-verified, no build).** Confirm the
  `sl261x` checkout (get-started setup had it at `D:\...\usb-tool\`), else
  `git clone <usb-tool-repo-url> usb-tool && cd usb-tool && git checkout sl261x`,
  then follow [`sl2619-windows-recovery.md`](sl2619-windows-recovery.md). **This
  is the lowest-risk path** (no compile, proven on this exact board) — use it if
  you don't want to build astra-update.

---

## 4. Image bundle — Scarthgap 6.12 v2.4.0

Release: <https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.4.0>
(published 2026-05-30. **Note:** the doc site's `v/latest/` still shows v2.3.0 —
the GitHub **release tag is authoritative** for v2.4.0.)

Confirmed asset filenames (verified via GitHub API + `curl -IL`, all **`.zip`**,
not `.tar.gz`):

- **`sl2619_scarthgap_6.12_v2.4.0.zip`** ← the standard eMMC image. Use this.
- `sl2619_usb_boot_scarthgap_6.12_v2.4.0.zip` ← USB-boot SBL stage (fallback if
  the boot/run-sm stage fails).
- *Not* these: `sl2619_nand_…`, `sl2619_spi_boot_…`, `sl2619_coralboard_…`,
  `sl2619_oobe_…`.

Unzipping yields the **Astra System Image** directory — point `astra-update
--flash` (or `usb_boot_tool.py --img-dir`) at it. Confirmed contents:

```
emmc_part_list          # GPT partition table
emmc_image_list         # incremental: which subimg → which partition
emmc_image_list_full    # full variant (use this for corruption recovery, §2)
*.subimg / *.subimg.gz  # preboot, key, tzk, sysmgr, bl, boot, firmware, rootfs(_s.0/1/2), home(_s)
```

---

## 5. Run the flash

**USB-boot button sequence.** The get-started doc holds `USB_BOOT` then taps
`RESET`; the official SL2600 Developer Kit User Guide (NR-160458) specifies the
RESET-first ordering — *hold RESET → press & hold USB_BOOT (still holding RESET)
1–2 s → release RESET while still holding USB_BOOT until console prints → release
USB_BOOT*. Both achieve the same thing (USB_BOOT asserted across the reset
edge); use whichever enumerates the device. Driver must be installed host-side
first.

For the **Windows / `usb_boot_tool.py`** path, follow
[`sl2610-get-started.md`](../references/upstream/synaptic-sl2619/docs/get-started/sl2610-get-started.md)
§3–6 verbatim. Summary of the two commands (run from the `usb-tool` dir, same
terminal, USB-C in the **USB Boot / CDC** port, `PWR_IN` powered):

```powershell
# After the USB-boot button sequence (hold USB_BOOT, tap RESET, release):
python usb_boot_tool.py --op run-sm --ddr-type ddr4
python usb_boot_tool.py --op emmc   --img-dir <path>\eMMCimg
```

Wall-clock ~5–15 min (rootfs flashed to both A and B slots). Ends on
`ALL OPERATIONS COMPLETE`. Then cold-boot (tap `RESET` only) and verify per
get-started §6 (`adb devices` → `sl2619 device`, `uname -a` shows 6.12.x
scarthgap).

---

## 6. After recovery — don't re-brick

The original DTB-into-eMMC approach is what bricked the board. Do **not** repeat
a blind `dd`/`seek`-write of a `.dtb` into `mmcblk0pN`. If the BLE bring-up still
needs the UART1/bcm43438 device-tree node, do it the supported way:

- Patch the DTS in the **Yocto build** (`linux-drivers-synaptics` /
  device-tree recipe) and reflash via this same `usb_boot_tool.py` path, **or**
- Use a U-Boot overlay (`fdt apply`) / `uEnv.txt` mechanism so a bad node can't
  corrupt the on-disk boot image.

See the BLE bring-up status (Synaptics bug 37861/37374) in
`docs/plans/dispenser-demo/decisions-log.md`.

---

## Root cause — why the DTB write corrupted the image {#root-cause}

The write was:
```python
for dev, offset in [('/dev/mmcblk0p10', 0x1178258), ('/dev/mmcblk0p11', 0x1178258)]:
    with open(dev, 'r+b') as f: f.seek(offset); f.write(new_dtb)
```
`mmcblk0p10`/`p11` are the **redundant A/B boot partitions** (`boot_a`/`boot_b`),
each a **packaged FIT image** (kernel + DTB + ramdisk, wrapped with a header,
per-node **hashes**, and — on this board, `secure_boot: genx` — a **signature**).
The DTB is a node *inside* that container, after the kernel blob (hence the
~17.47 MiB offset). Treating that container as a flat file you can `seek`+`write`
in place triggered **five** independent boot-killers, any one of which is fatal:

1. **FIT hash invalidated.** U-Boot's `bootm` verifies the FDT node hash before
   use; the in-place overwrite no longer matches → `bad hash` → abort.
2. **Secure-boot signature broken.** The boot image is signed; changing any byte
   fails signature verification in the ROM→BL→SM→A-core chain → hard stop.
3. **Size overrun.** The patched DTB is *larger* (you added `uart1_pmux` +
   `bluetooth` nodes), so a fixed-offset write spilled past the DTB's fixed slot
   into adjacent FIT data/ramdisk.
4. **Hard-coded offset, two partitions.** One constant `0x1178258` applied to
   both `p10` and `p11` assumes byte-identical layout; at least one write landed
   in the wrong place.
5. **A/B redundancy destroyed.** Writing the same bad blob to *both* slots
   removed the fallback that would otherwise have booted the good copy.

**Underlying mistake:** treating a signed, hash-verified, fixed-layout firmware
container as a flat patch target. **Recoverable** because only partition
*contents* were touched — the GPT and the mask-ROM eROM are intact, so USB-boot
recovery rewrites everything cleanly. Supported DTB-change paths are in §6.

---

## References

- Verified flash sequence: `references/upstream/synaptic-sl2619/docs/get-started/sl2610-get-started.md` §3–6
- `astra-update` usage + manifests: `references/upstream/synaptic-sl2619/references/Synaptics/astra-update/README.md`
- `usb-tool` README: `references/upstream/synaptic-sl2619/references/Synaptics/usb-tool/README.rst`
- Astra SDK docs: <https://synaptics-astra.github.io/doc/v/latest/>
- SDK releases: <https://github.com/synaptics-astra/sdk/releases>
- SL2600 HW guide (USB-boot button sequence): <https://synaptics-astra.github.io/doc/v/latest/hw/sl2600.html>
