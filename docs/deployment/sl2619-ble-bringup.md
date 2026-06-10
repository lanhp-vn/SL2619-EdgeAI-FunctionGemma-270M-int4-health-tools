# SL2619 BLE / Bluetooth bring-up (M.2 combo, Astra v2.4.0)

> **Status (2026-06-01): WORKING, proven end-to-end.** The revB pin-mux blocker
> (Synaptics bug 37861/37374) is **resolved by Astra v2.4.0**. The M.2 combo's
> BT (`hci0`, UART, SYN43711) enumerates + firmware-patches at boot; `bluetoothctl
> scan` discovers real LE devices under load with no power-down; and **pybleno
> advertised `NousVoice`, a phone connected, subscribed to char `0xFFB2`, and
> received `5A A5 01 00`** — the full [plan §6.2](../plans/dispenser-demo/plan.md)
> wire contract, on real hardware. §1 is the copy-paste recipe; §3 explains why
> each non-obvious step is needed.
>
> **Remaining (integration, not bring-up):** swap the `[BLE→ESP32]` stdout mock
> in `chat_board_dispense.py:dispatch` for a real `PyBlenoBleClient.notify`;
> verify the §3.5 boot auto-power-down on a clean `reboot`; persist the `/tmp`
> staging to `/mnt/sdcard`.

Bring-up companion to
[`sl2619-postrecovery-bringup.md`](sl2619-postrecovery-bringup.md) §8 (get the
board networked + SSH-reachable first). Targets the **M.2 daughter-card**
Wi-Fi/BT combo on the SL2619 RDK.

---

## 0. TL;DR (what works, and the four gotchas)

- **v2.4 fixed the revB pin-mux.** Pre-v2.4 there was *no* `hci*`; now `hci0`
  enumerates + patches at boot via `brcm_bt_start.service`.
- **BLE works in both roles:** central (`bluetoothctl scan`) and **peripheral**
  (pybleno advertise + GATT notify — the demo's role), confirmed end-to-end.
- The four traps that the recipe handles (each cost a debugging round):
  1. **Patch lives in chip RAM** — if `BT_REG_ON` drops, `hci0` goes DOWN and a
     bare `hciconfig up` can't recover (it re-powers but can't re-patch).
     Recovery = `systemctl restart brcm_bt_start.service` (§3.2).
  2. **Mandatory `hciconfig up`→`down` reset cycle** before pybleno, or it fails
     with `Command Disallowed` (§3.3).
  3. **Board Python has no `pip`** → pybleno is staged manually (§1a).
  4. **Board Python has no `fcntl`** → a ctypes shim is staged (§1a, §3.4).

---

## 1. The working recipe

### 1a. One-time host-side staging (pybleno + fcntl shim + runner)

The board can't `pip install` (no `pip`/`ensurepip`). Download + patch pybleno
on the host, then `scp` the package tree and the `fcntl` shim onto the board's
`PYTHONPATH`. Run from the repo root on the WSL host:

```bash
# host (WSL) — download pybleno (pure-Python 0.11, no deps) and unpack
python3 -m pip download pybleno --no-deps -d /tmp/dl
( cd /tmp/dl && tar xzf pybleno-*.tar.gz )

# apply the 5 Py3.12/kernel-6.x patches to BluetoothHCI.py
python3 scripts/dispenser_demo/deploy/patch_pybleno_bluetoothhci.py \
    /tmp/dl/pybleno-*/pybleno/hci_socket/BluetoothHCI/BluetoothHCI.py

# stage everything to the board: patched pybleno pkg, fcntl shim, runner + client
ssh nouslogic-sl2619 'mkdir -p /tmp/pylibs/gemma_tools/dispenser_demo && \
    : > /tmp/pylibs/gemma_tools/__init__.py && \
    : > /tmp/pylibs/gemma_tools/dispenser_demo/__init__.py'
scp -r /tmp/dl/pybleno-*/pybleno                       nouslogic-sl2619:/tmp/pylibs/
scp scripts/dispenser_demo/deploy/board_fcntl_shim.py  nouslogic-sl2619:/tmp/pylibs/fcntl.py
scp scripts/dispenser_demo/deploy/ble_test.py          nouslogic-sl2619:/tmp/
scp src/gemma_tools/dispenser_demo/ble_client.py       nouslogic-sl2619:/tmp/pylibs/gemma_tools/dispenser_demo/
```

> `ble_client.py` must land at the **package path**
> `gemma_tools/dispenser_demo/ble_client.py` (not flat) or `ble_test.py`'s import
> fails. `/tmp` is **volatile** — for a reboot-durable demo, stage to
> `/mnt/sdcard/pylibs/` instead and point `PYTHONPATH` there.

### 1b. Board-side bring-up (every boot / before each pybleno run)

```sh
# board (ssh nouslogic-sl2619)
systemctl stop bluetooth                    # pybleno needs exclusive control, not BlueZ
systemctl restart brcm_bt_start.service     # re-power + re-patch the chip; leaves hci0 DOWN
hciconfig hci0 up                           # REQUIRED kernel HCI Reset/init (else 'Command Disallowed' — §3.3)
hciconfig hci0 down                         # release for pybleno's HCI_CHANNEL_USER claim
```

### 1c. Run the peripheral + phone test

```sh
# board — advertise; -t / a TTY lets ENTER fire repeat notifies
PYTHONPATH=/tmp:/tmp/pylibs python3 /tmp/ble_test.py --hci hci0 --skip-patch-check --timeout-s 120
#   wire contract: name=NousVoice  adv-svc=0x00FB  primary=0xFFB0  notify=0xFFB2  payload=5A A5 01 00
```

On the phone (**nRF Connect**): scan → **NousVoice** → connect → expand service
**`0xFFB0`** → enable notifications on char **`0xFFB2`**. The board auto-sends
`5A A5 01 00` on subscribe; press **ENTER** for each subsequent notify
(`--send-once` auto-fires one then exits instead).

### 1d. Confirmed output (2026-06-01, this is what success looks like)

```
PyBlenoBleClient: BLE state = poweredOn
PyBlenoBleClient: advertising as 'NousVoice' on hci0, primary svc 0xFFB0
ble_test: advertising started, waiting up to 120.0 s for a subscriber
PyBlenoBleClient: central connected (5f:3f:d4:de:96:3a)
PyBlenoBleClient: central subscribed to FFB2
ble_test: subscriber attached to 0xFFB2
ble_test: dispense notify sent (5aa50100)        # auto on subscribe
ble_test: dispense notify sent (5aa50100)        # each ENTER
```
Phone receives `5A-A5-01-00` on FFB2. **End-to-end proven.**

---

## 2. As-probed findings — 2026-06-01 (read-only SSH, `nouslogic-sl2619`)

| Item | Value |
| --- | --- |
| Astra image | `scarthgap_6.12_v2.4.0` (kernel `6.12.62`) |
| DT model / compatible | `Synaptics SL2619 RDK` / `syna,sl2619-rdk syna,sl2619` |
| **BT transport** | **UART** — `hci0  Type: Primary  Bus: UART`, on **`/dev/ttyS1`** (`e5031000.uart`, `base_baud=6250000`) |
| BT BD address | `9C:B8:B4:3C:27:7B` (= wlan0 MAC `…:7A` **+ 1** → same combo module) |
| **Combo chip** | **SYN43711** — Wi-Fi fw `fw_sd_bcm43711.bin` (SDIO/mmc1, `bcmdhd`); BT patch `SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz…hcd` |
| BT patch firmware dir | **`/lib/firmware/bcm/`** (patchram arg is the *dir*; `/lib/firmware/brcm/` does not exist) |
| `BT_REG_ON` power | I²C GPIO expander at **`0-0044`** → `bluetooth-rfkill` platform driver (`/sys/bus/platform/devices/bt_reg_on`) |
| Attach service | **`brcm_bt_start.service`** (enabled, oneshot, `RemainAfterExit=yes`) → backgrounds `brcm_patchram_plus` at 3 Mbaud |
| `bluetoothd` / `bluealsa` | present + enabled; stop `bluetooth` before pybleno |
| Board Python | 3.12.9 — **no `pip`/`ensurepip`, no `fcntl` module** |
| LE caps (hci0) | `<LE support>`, `<LE and BR/EDR>`; advertising states supported (central + peripheral) |

### Corrections to prior project lore (now ground-truth)
The old `synaptics-sl2619-references` memory assumed BT-on-SDIO, `btbcm.ko`,
`/lib/firmware/brcm/*`, maybe `hci1`. The live board contradicts all four:
**BT is UART (`hci0`, ttyS1)**; only **Wi-Fi** is on SDIO1/mmc1; firmware is in
**`/lib/firmware/bcm/`**; the chip is **SYN43711**. Memory updated.

---

## 3. Why each step — root cause & gotchas

### 3.1 Boot bring-up succeeds (the pin-mux fix is real)
At boot, `brcm_bt_start.service` → `brcm_patchram_plus --enable_hci --patchram
/lib/firmware/bcm /dev/ttyS1` powers the chip on, downloads the `.hcd` at
3 Mbaud (`Done setting baudrate` / `Done setting line discpline`), and creates
`hci0`. BlueZ reads the full v5.3 cap set. v2.4's boot firmware carries the
revB pin-mux fix (CM52 `RELEASE_NOTE.txt`: *"Support RDK revA, revB and revC"*).

### 3.2 The patch lives in chip RAM → recovery is a service restart, not `up`
The `.hcd` is downloaded into the controller's RAM and is **lost whenever
`BT_REG_ON` drops** (the chip powers off). Once that happens, `hci0` is DOWN and
a bare `hciconfig hci0 up` **cannot** recover it: it re-powers the chip but does
not re-download the patch, so it drives an unpatched/dead controller →
`Frame reassembly failed (-84)` → driver powers off again → `HCI_Reset` (opcode
`0x1003`) tx-timeout → `Can't init device hci0: Connection timed out (110)`.
**Recovery = `systemctl restart brcm_bt_start.service`** (kills the stale
patchram holding `/dev/ttyS1`, re-powers, re-downloads `.hcd`, re-attaches).

### 3.3 The mandatory `up`→`down` reset cycle (`Command Disallowed`)
A freshly patchram-attached controller has **not** had a kernel HCI Reset.
pybleno's `HCI_CHANNEL_USER` advertising sequence then fails with
`RuntimeError: BLE advertising error: Exception('Command Disallowed')` (HCI
0x0C). `hciconfig hci0 up` runs the kernel reset+init; `hciconfig hci0 down`
hands the now-reset (still patched) controller to pybleno. Confirmed both ways:
without the cycle → `Command Disallowed`; with it → `advertising started`.

### 3.4 No `pip`, no `fcntl` on the board
The Yocto Python 3.12 ships without `pip`/`ensurepip` (→ stage pybleno manually,
§1a) and without the `fcntl` module. pybleno only calls `fcntl.ioctl`; its
`device_up` ioctl path is unused (commented out in `Hci.py`, since
HCI_CHANNEL_USER auto-ups the controller), so a small ctypes-backed `ioctl` shim
(`scripts/dispenser_demo/deploy/board_fcntl_shim.py`, staged as `fcntl.py`)
satisfies `import fcntl`.

### 3.5 OPEN — what cuts `BT_REG_ON` after boot?
`bluetooth_set_power: onoff = 0` deasserts `BT_REG_ON` at **boot +22 s** (no
preceding error) — which is why `hci0` was found DOWN before anyone touched it.
The §1b `brcm_bt_start` restart recovers it for a session, but the boot-time
trigger is unconfirmed. Candidates: a power-mgmt policy in the
`bluetooth-rfkill`/`bluesleep` LPM driver, or autosuspend on the shared `0-0044`
I²C-expander rail (also feeds `sdhci`/`mdio`). **To close:** on a clean
`reboot`, check `hciconfig hci0` *without* restarting anything — if DOWN, add a
`brcm_bt_start` restart (or an LPM/autosuspend disable) to the boot path so the
demo never needs manual intervention.

### 3.6 Demoted hypothesis (for the record)
*"3 Mbaud is a non-integer divisor of the 6.25 MHz `base_baud` → framing skew."*
**Unlikely:** the 50 KB `.hcd` transferred cleanly at 3 Mbaud at boot, and the
decisions-log records alternate baud rates were already tried historically. The
`-84` is a symptom of the unpatched chip (§3.2), not a baud problem.

---

## 4. Board revision (revA vs revB) — not software-discoverable

Synaptics support: BT failed on **revB** (pin-mux), worked on **revA**; **Astra
2.3** = revB patch, **2.4** = both revs. This board's rev **cannot be read in
software** — `/proc/device-tree/model`, `compatible`, and boot dmesg all return
the rev-less `Synaptics SL2619 RDK`; `fw_printenv` is absent and there is no
board-ID EEPROM. **Read the PCB silkscreen** if you need it. Do **not** infer
rev from "BT works" — v2.4 fixes revB too, so an enumerating radio is not
evidence of revA. For BT, rev is now academic.

---

## 5. Diagnostics reference (read-only)

```sh
# board
hciconfig -a                                              # transport, BD addr, state
ps w | grep -iE 'patchram|hciattach|btattach' | grep -v grep   # attach holding ttyS1?
systemctl status brcm_bt_start.service --no-pager
journalctl -u brcm_bt_start.service -b --no-pager | tail -n 40
journalctl -b --no-pager | grep -iE 'set_power|onoff|rfkill|bluesleep|-84|0x1003|codec' | tail -n 40
cat /var/lib/systemd/rfkill/platform-bt_reg_on:bluetooth  # 0 = unblocked (saved across reboot)
ls -la /lib/firmware/bcm/SYN43711*.hcd                    # the patch file the service points at
PYTHONPATH=/tmp:/tmp/pylibs python3 -c 'import pybleno, fcntl; print("imports OK")'  # staging sane?
```

The attach service (for reference):
```ini
# /usr/lib/systemd/system/brcm_bt_start.service  (oneshot, RemainAfterExit=yes)
ExecStartPre=/bin/sh -c 'rfkill unblock bluetooth'
ExecStart=/bin/sh -c '/usr/bin/brcm_patchram_plus -d --tosleep=300000 --baudrate 3000000 \
    --use_baudrate_for_download --no2bytes --enable_hci --patchram /lib/firmware/bcm /dev/ttyS1 &'
```

| Symptom | Cause | Fix |
| --- | --- | --- |
| `hciconfig up` → `Connection timed out (110)` | chip powered down, patch lost | `systemctl restart brcm_bt_start.service` (§3.2) |
| pybleno → `Command Disallowed` | no kernel HCI Reset since attach | `hciconfig hci0 up && hciconfig hci0 down` first (§3.3) |
| `No module named 'fcntl'` / `'pybleno'` | staging missing / wrong `PYTHONPATH` | re-do §1a; use `PYTHONPATH=/tmp:/tmp/pylibs` |
| pybleno bind fails / `Device or resource busy` | `hci0` UP or held by BlueZ | `systemctl stop bluetooth; hciconfig hci0 down` |
| `hci0` DOWN after a reboot | §3.5 boot trigger | §1b restart; consider a boot-path fix |

---

## 6. Hardware/driver reference (as-built)

```
M.2 daughter-card combo: SYN43711 (Broadcom-class Wi-Fi/BT)
├── Wi-Fi → SDIO1 / mmc1   bcmdhd (bcmsdh_sdmmc), SDIO 06CB:AABF
│            fw /lib/firmware/  fw_sd_bcm43711.bin + bcmdhd_sd_43711.cal + bcmdhd_clm_43711.blob
└── BT    → UART /dev/ttyS1 (e5031000, base_baud 6250000)   hci_uart (H4/BCSP)
             patch /lib/firmware/bcm/SYN43711A0_…_UART_37_4MHz…hcd  (loaded into chip RAM)
             power: BT_REG_ON via I²C expander 0-0044 → bluetooth-rfkill driver
             attach: brcm_bt_start.service → brcm_patchram_plus → hci0 (BD = wlan0 MAC + 1)
```
Combo caveat: Wi-Fi + BT share one module. The `0-0044` expander also feeds
`sdhci`/`mdio` consumers — relevant when chasing the §3.5 power-down trigger.

---

## 7. Remaining work (integration, not bring-up)

1. **Wire the real notify.** Replace the `[BLE→ESP32] 5A A5 01 00` stdout mock in
   `scripts/functiongemma/deploy/chat_board_dispense.py:dispatch` with a
   `PyBlenoBleClient.notify` call. Design note: pybleno needs `hci0` **DOWN +
   HCI_CHANNEL_USER (exclusive)** while the rest of the voice loop / `bluetoothctl`
   want it **UP/managed** — the notify client's bind/release lifecycle (incl. the
   §3.3 reset cycle) must slot into `dispenser_voice.py`'s turn flow.
2. **Close §3.5** — clean-`reboot` check; add a boot-path fix if `hci0` comes up DOWN.
3. **Persist staging** — copy `pybleno/` + `fcntl.py` (+ runner) to `/mnt/sdcard/`
   so the demo survives reboot (`/tmp` is volatile).

---

## References
- Probe + bring-up: read-only/authorized SSH to `nouslogic-sl2619`, 2026-06-01 (this session).
- Synaptics support (2026-06-01): revB pin-mux fixed in Astra 2.3 (patch) / 2.4 (both revs).
- Boot RELEASE_NOTE: `references/upstream/synaptic-sl2619/references/Synaptics/boot/mcu/cm52/image/chip/klamath/RELEASE_NOTE.txt`.
- Kernel config: `…/configs/product/sl2619_poky_aarch64_rdk/sl2619_poky_aarch64_rdk_defconfig` (`CONFIG_DRIVERS_ln_LPM=y`, `CONFIG_DRIVERS_ln_RFKILL=y`).
- pybleno patches (origin): `docs/references/old-dispenser-demo/pybleno-setup-guide.md`.
- Repo artifacts: `scripts/dispenser_demo/deploy/{patch_pybleno_bluetoothhci.py,board_fcntl_shim.py,ble_test.py}` + `src/gemma_tools/dispenser_demo/ble_client.py`.
- Wire contract + Phase-2: `docs/plans/dispenser-demo/plan.md` §6.2, §9; decisions-log 2026-06-01 entry.
- Prerequisite: [`sl2619-postrecovery-bringup.md`](sl2619-postrecovery-bringup.md) §8.
