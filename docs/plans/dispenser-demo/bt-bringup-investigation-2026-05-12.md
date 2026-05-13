# BT bring-up investigation — 2026-05-12 (PM)

This document closes out the M.2 Bluetooth bring-up question for the SL2619
dev kit. It is a hypothesis-by-hypothesis audit, deeper than the
2026-05-11 (late, addendum) entry in [`decisions-log.md`](decisions-log.md)
but reaching the same disposition: **board BT bring-up is blocked by
something outside the software stack and cannot be resolved without
Synaptics support or a hardware swap.**

The dispenser demo's BLE path therefore remains stdout-mocked (see
[`plan.md §6.2`](plan.md) and [`chat_board_dispense.py`](../../../scripts/functiongemma/deploy/chat_board_dispense.py))
until either (a) Synaptics ships a fixed DTS/firmware for the M.2 socket
(bug 37861/37374), or (b) the demo migrates to a different BLE-capable
host. See [§ "Practical disposition"](#practical-disposition).

> **Reading order.** Skim [§ Final hypothesis matrix](#final-hypothesis-matrix)
> first. Everything else is supporting evidence.

## What this session adds beyond the 2026-05-11 audit

The earlier audit (`decisions-log.md` 2026-05-11 late addendum) established
that `brcm_patchram_plus` loops `HCI_Reset` indefinitely on `/dev/ttyS1`
and the chip never replies. It speculated the UART driver might not honor
flow control. **This session disproves that and several other earlier
guesses**, narrowing the failure to one of three physical-layer issues
that cannot be tested from SSH.

| Aspect | 2026-05-11 audit | 2026-05-12 (this doc) |
| --- | --- | --- |
| UART1 pinmux for SM_GPIO7/8/14/15 | Untested | **Verified correct** via direct devmem read of pinctrl@`0xe5025b00`: `+0xc = 0x36246` (SM_GPIO14/8/7 = function 6 = `uart1`); `+0x08 = 0x30249249` (SM_GPIO15 = function 6 = `uart1`). The empty `pinctrl-0` on `uart@e5031000` in DT is a red herring — the bootloader sets pinmux at boot. |
| 32.768 kHz refclk to combo chip | Not considered | **Effectively ruled out** as the BT-side blocker: WiFi works, which requires the same slow clock on BCM4345-family silicon. Manually muxing `SM_GPIO30 → sm_clkout` (function 7) on top of the existing config produced no change in BT behavior. |
| BT_REG_ON wiring | Assumed correct | **DT-side verified**: `/proc/device-tree/bt_reg_on/bt-power-gpio = <phandle 0x0f, pin 5, ACTIVE_HIGH>`, and `phandle 0x0f` resolves to `gpio@44` (FXL6408 @ 0x44). Kernel claims `gpio-565` with consumer `bt_power`. **Whether pin 5 of the FXL physically reaches the M.2 socket's BT_REG_ON is the remaining open question** (see H13 below). |
| FXL6408 register map | NotebookLM guess (incorrect) | **Corrected** from mainline `gpio-fxl6408.c`: `0x03=Direction`, `0x05=Output`, `0x07=High-Z`, `0x0F=InputStatus`. The chip's Input Status register reads 0 for pins configured as output, which is why `/sys/kernel/debug/gpio` reports `out lo` for `bt_power` even when the kernel has it driven high. |
| UART RX at the chip side | Not measured | **Measured at three baud rates** (115200, 921600, 3 Mbps) with no patchram running, around BT_REG_ON cycle: **zero bytes** received in 5–9 seconds at each rate. |

## Final hypothesis matrix

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | M.2 card not present | **RULED OUT** | `bcmdhd` loads `fw_sd_bcm43711.bin` over SDIO; DHCP/EAPOL/ARP all complete; `wlan0` carries traffic. The combo chip is alive. |
| H2 | Wrong patchram `.hcd` file | **RULED OUT** | `/lib/firmware/bcm/SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz_wlbga_REF_sLNA_iLNA_ANT0.hcd` is present and matches the SYNA 43711S silicon. Synaptics SDK ships `BCM4345C5_003.006.006.1066.1126.hcd` (same silicon, older naming) as its canonical pick — also present. Both produce identical `HCI_Reset` retry loop. |
| H3 | `brcm_patchram_plus` missing/broken | **RULED OUT** | Installed at `/usr/bin/brcm_patchram_plus`, executes, writes the correct `01 03 0c 00` opcode at 4-second cadence per the journal. |
| H4 | UART1 controller missing or `status="disabled"` | **RULED OUT** | `/proc/device-tree/soc/sm/uart@e5031000/compatible = "snps,dw-apb-uart"`, `status = okay`. dmesg: `e5031000.uart: ttyS1 at MMIO 0xe5031000 (irq=15, base_baud=6250000)`. `dw-apb-uart` driver bound; `/dev/ttyS1` accepts writes (`rc=0`). |
| H5 | Wrong `/dev/ttyS*` (M.2 UART is elsewhere) | **RULED OUT** | Synaptics SDK's `klamath_brcm_bt_start.patch` pins `/dev/ttyS1`. `e5031000.uart` lives under `pinctrl@e5025b00` which controls SM_UART1 pins (SM_GPIO7/8/14/15). Dev kit user guide confirms M.2 socket BT UART = SM_UART1. |
| H6 | UART1 pinmux not applied | **RULED OUT** | `uart@e5031000`'s DT has empty `pinctrl-0` — but direct devmem read of `pinctrl@0xe5025b00` registers shows pins are **already** set to `uart1` (function 6) by the bootloader. See [§ Register evidence](#register-evidence). |
| H7 | 32.768 kHz refclk not muxed (`SM_GPIO30 → sm_clkout`) | **RULED OUT** | WiFi works, which requires the slow clock on BCM4345 silicon. Manually setting `SM_GPIO30` to `sm_clkout` (function 7) — overriding its `xspi` default — produced no change in BT behavior. The refclk must reach the chip via a different path (internal oscillator on the M.2 module, or a different pin we have not identified). |
| H8 | Wrong FXL6408 register addresses (NotebookLM's mistake) | **RULED OUT** | Mainline `drivers/gpio/gpio-fxl6408.c`: `0x03=IO_DIR`, `0x05=Output`, `0x07=Output_HighZ`, `0x0F=Input_Status`. NotebookLM had claimed `0x03=Direction` and a different layout — verifying against the kernel source corrected this. |
| H9 | FXL6408 cannot drive outputs high | **RULED OUT (debugfs artifact)** | The FXL6408's Input Status register reads 0 for pins configured as output. `gpio-fxl6408.c`'s `get()` reads Input Status, so `/sys/kernel/debug/gpio` displays `out lo` for pins the kernel has driven high. The state register `0x05` shows bit 5 = 1, meaning pin 5 is electrically driven high. (Confirmed by toggling direction: when pin 5 is briefly configured as input, Input Status bit 5 reads 1 — the line *is* high externally.) |
| H10 | `bt_reg_on` DT node points at wrong GPIO | **RULED OUT** | `/proc/device-tree/bt_reg_on/{compatible="syna,rfkill", bt-power-gpio=<phandle 0x0f, pin 5, ACTIVE_HIGH>}`. Phandle `0x0f` = `/proc/device-tree/soc/sm/i2c@e5035000/gpio@44`. Driver registers `gpio-565` with consumer `bt_power`. Wired as documented. |
| H11 | Kernel BT subsystem missing | **RULED OUT** | dmesg: `Bluetooth: Core ver 2.22`, `HCI UART driver ver 2.3`, `HCI UART protocol H4 registered`, `HCI UART protocol BCSP registered`, RFCOMM, BNEP, HIDP — all built-in. |
| H12 | `bluez5` userspace missing | **RULED OUT** | `bluetooth.service` runs `bluetoothd` at PID 704. `hciconfig`, `hciattach`, `btattach`, `rfkill`, `bluetoothctl` all installed. |
| **H13** | **BT_REG_ON wired to a different FXL pin** | **UNTESTED — physical** | FXL pin 4 of `gpiochip5` is configured as output, currently driven low, **with no consumer label** — the kernel set it up but no driver knows what it does. It's the only remaining FXL pin that could plausibly be the real BT_REG_ON. We started testing this but the board hung before the i2c write completed (likely a dependent reset elsewhere being held). |
| **H14** | **M.2 module's BT-side hardware fault** | **UNTESTED — physical** | The WiFi half of the combo could work while the BT half is dead (separate power domain, separate RF FE, separate bond wires). Requires swapping in a known-good M.2 card to confirm. |
| **H15** | **SL2619-RDK PCB trace from SoC to M.2 socket is broken/unpopulated** | **UNTESTED — physical** | Pinmux says SM_GPIO7/8 are `uart1`, but the actual PCB trace from SoC pads to the M.2 socket may be broken or never assembled (some dev kit revisions have unpopulated nets). Requires oscilloscope on the M.2 socket pins. |

Three of fifteen hypotheses survive. **All three need hardware action
the SSH agent cannot take.**

## Register evidence

Per `pinctrl-sl261x.c` (in the Synaptics SDK at
`docs/references/upstream/synaptic-sl2619/references/Synaptics/linux-drivers-synaptics/pinctrl/berlin/pinctrl-sl261x.c`):

```
BERLIN_PINCTRLCONF_GROUP("SM_GPIO15", 0x8, 0x3, 0x1b, ...,  BERLIN_PINCTRL_FUNCTION(0x6, "uart1"))  // RTSn
BERLIN_PINCTRLCONF_GROUP("SM_GPIO14", 0xc, 0x3, 0x00, ...,  BERLIN_PINCTRL_FUNCTION(0x6, "uart1"))  // CTSn
BERLIN_PINCTRLCONF_GROUP("SM_GPIO8",  0xc, 0x3, 0x0c, ...,  BERLIN_PINCTRL_FUNCTION(0x6, "uart1"))  // TXD
BERLIN_PINCTRLCONF_GROUP("SM_GPIO7",  0xc, 0x3, 0x0f, ...,  BERLIN_PINCTRL_FUNCTION(0x6, "uart1"))  // RXD
BERLIN_PINCTRLCONF_GROUP("SM_GPIO30", 0x4, 0x3, 0x0f, ...,  BERLIN_PINCTRL_FUNCTION(0x7, "sm_clkout"))
```

The pinmux register file is at MMIO base `0xe5025b00` (the 16-byte
region per `/proc/iomem`; the 164-byte region at `0xe5025c00` is the
pin **config** registers — drive strength, pulls — not mux). Reading
the live state with `devmem`:

```
0xe5025b00 = 0x12000264
0xe5025b04 = 0x09248000   # bits 15-17 = 0x1 = xspi → SM_GPIO30. Could be sm_clkout (0x7) but did not affect BT.
0xe5025b08 = 0x30249249   # bits 27-29 = 0x6 = uart1 → SM_GPIO15 (RTSn) ✓
0xe5025b0c = 0x00036246   # bits 0-2 = 6 (uart1 CTSn / SM_GPIO14)
                           # bits 12-14 = 6 (uart1 TXD / SM_GPIO8)
                           # bits 15-17 = 6 (uart1 RXD / SM_GPIO7) ✓
```

UART1 pinmux is correctly set. The bootloader (ATF or U-Boot) configures
this before the kernel boots, so the kernel-side DT pinctrl reference
on `uart@e5031000` being empty is not the bug.

## Loopback evidence

With no patchram running, `/dev/ttyS1` configured `raw $baud -echo`,
8 bytes written, then 3-second listen:

| Baud | Bytes RX |
|---|---|
| 115200 | 0 |
| 921600 | 0 |
| 3000000 | 0 |

Cycling rfkill (`rfkill block bluetooth; sleep 0.3; rfkill unblock bluetooth`)
between writes produces no boot-time chatter from the BT chip at any
baud. A BCM4345-family chip with a working UART will emit no traffic
on its own (no boot banner — it waits for HCI_Reset), so absence of
unsolicited bytes is not itself diagnostic. But the chip also never
responds to `brcm_patchram_plus`'s HCI_Reset over a 3+ hour observation
window, which is.

## Things ruled out as failure modes but worth keeping pinned

These came up in the investigation and are confirmed *not* the cause,
but the symptom-to-cause mapping is non-obvious — note them so the
next session does not re-derive them:

- **Pinmux register at `0xe5025c0X` is pinconf, not pinmux.** Writing
  to `0xe5025c0c` to set uart1 — as I did mid-session — only flips
  drive-strength bits. The true pinmux base is `0xe5025b00`. Easy
  to confuse since the same pinctrl device claims both regions per
  `/proc/iomem`.
- **`pinmux-select` debugfs is `<function> <group>`, not arbitrary
  apply.** The `sl261x-pinctrl` driver responds to writes with
  `invalid function SM_GPIO7 in map table` because it parses the
  first token as a function name. The mainline interface expects
  `<device-name> <state-name>`, not "apply function X to group Y" —
  the standard kernel does not provide a runtime "rebind a pin
  group" path on this driver.
- **`gpioset --mode=signal` is libgpiod v1 syntax.** The board ships
  libgpiod v2.1.2 which doesn't recognize it. Use `gpioset --daemonize
  --chip gpiochip5 -C <consumer> 5=1` on v2 — but note that the
  `bluetooth-rfkill` driver already holds line 5 with `EBUSY` for any
  external consumer.
- **FXL6408 pin 2 of `gpiochip5` (consumer = `reset`, ACTIVE_LOW)
  gates network connectivity.** Driving it low (via either `i2cset
  -f` on register `0x05` or `gpioset`) drops `wlan0` and SSH within
  ~1 second. Do not touch this pin in any "try toggling all the
  things" experiment. NotebookLM had speculated it was WL_REG_ON;
  that is wrong.
- **Synaptics `linux-firmware-syna_*.bb` installs `.hcd` to
  `/usr/lib/firmware/bcm/` (symlinked from `/lib/firmware/bcm/`),
  NOT `/lib/firmware/brcm/`.** Earlier audits checked `brcm/` and
  reported it missing; the SDK uses `bcm/` (no r). Both legacy
  `board_probe` snapshots and external docs have this wrong.

## Practical disposition

The dispenser demo's BLE notify is the only remaining v1 blocker.
There are three workable paths in order of cost:

1. **USB BT dongle on the SL2619 board.** Will NOT work on this image
   — the kernel has only `hci_uart` built-in; no `btusb` module or
   built-in support. Confirmed by inspection of
   `/lib/modules/$(uname -r)/kernel/drivers/bluetooth/` (does not
   exist). Rebuilding the kernel with `CONFIG_BT_HCIBTUSB=y` would
   work, but at that point we are doing a kernel rebuild and could
   equally well fix the M.2 path.

2. **Run BLE from a different host on the same network.** The dispenser
   demo's `dispense_medication` intent is just an outbound notify
   payload (`5A A5 01 00`) to the ESP32 dispenser. Move the BLE
   peripheral process to a Linux box with a working radio (WSL2 +
   usbipd-win + a CSR4.0 dongle, a Raspberry Pi, any laptop). Have
   `chat_board_dispense.py` send the dispatch payload to that host
   over the LAN instead of stdout-printing it. Wire contract
   (`docs/plans/dispenser-demo/plan.md §6.2`) is unchanged.

3. **Synaptics support.** File against bug 37861/37374 with the
   evidence below. The asks:
   - Confirm the FXL6408 @ 0x44 pin-to-M.2-signal map on the
     SL2619-RDK PCB. Specifically, is `BT_REG_ON` actually on pin 5,
     or is it on pin 4 (the unnamed output) — i.e., is the kernel
     `bt_reg_on` DT node's `<phandle 0x0f, pin 5>` correct for *this
     PCB*?
   - Is there an additional GPIO or regulator-enable sequence the
     dev kit firmware does that mainline kernel + the published
     `brcm_bt_start.service` does not?
   - Sample DTS overlay for the SL2619-RDK's M.2 BT socket, in the
     style of `astra-doc/subject/enable_sdio_wifi.rst` (which covers
     the SL1640/SL1680 WiFi case).
   - If none of the above: confirm whether the dev kit was shipped
     with the BT path validated end-to-end, or whether it is a
     known-deferred hardware-enablement item.

**Recommendation for v1 demo.** Take path 2. The host-side BLE work
in `src/gemma_tools/dispenser_demo/ble_client.py` and
`scripts/dispenser_demo/deploy/ble_test.py` is already
hardware-agnostic and tested with 43 cases green; pointing it at a
different machine costs less than another hour of board investigation
and avoids becoming hostage to a hardware-enablement bug we can't
debug remotely.

## Evidence to forward to Synaptics

Attach this document plus:

- `docs/tmp/sl2619-status.md` — board snapshot (any version after
  power-cycle; the §12 BT re-probe section captures the chip+firmware
  fingerprint).
- Image identity: kernel `6.12.62`, Poky 5.0.9 scarthgap, board fingerprint
  `Synaptics SL2619 RDK / compatible = "syna,sl2619-rdk syna,sl2619"`.
- This sentence: "WiFi over SDIO works (firmware `fw_sd_bcm43711.bin`
  loads, `wlan0` carries traffic). UART1 pinmux is set to `uart1`
  function on SM_GPIO7/8/14/15 per direct devmem read of
  `pinctrl@0xe5025b00`. `bluetooth-rfkill` claims `gpiochip5` line 5
  per `<phandle 0x0f, pin 5>` in the `bt_reg_on` DT node and toggles
  it on rfkill unblock. `brcm_patchram_plus -d --baudrate 3000000
  --use_baudrate_for_download --patchram /lib/firmware/bcm/SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz_wlbga_REF_sLNA_iLNA_ANT0.hcd
  /dev/ttyS1` loops `HCI_Reset` (`01 03 0c 00`) with zero response;
  `/dev/ttyS1` returns zero bytes on a 3-second listen at 115200,
  921600, and 3 Mbps after rfkill cycle."

## What NOT to retry next session

Five hours on 2026-05-11 plus several more hours on 2026-05-12 covered
the productive software variations. Do not retry without new evidence
from outside (Synaptics ticket, a known-good M.2 swap, an oscilloscope
trace, or a Synaptics-blessed DTS):

- Different `.hcd` files
- Different baud rates
- `hciattach` instead of `brcm_patchram_plus`
- More `rfkill block/unblock` cycles
- `pinmux-select` writes in any syntax
- `devmem` writes to pinmux or FXL registers
- Toggling FXL pin 4 (was started but board hung; if attempted again,
  start a long-running gpioset/i2cset that auto-reverts after 30s in
  case it brings down the board)

The next legitimate experiment is **only after one of**:
1. Synaptics returns a working DTS or invocation.
2. A different M.2 card is swapped in.
3. An oscilloscope confirms whether the SoC's SM_UART1 TX pad is
   physically connected to the M.2 socket's BT RX line.

---

*Last refresh: 2026-05-12 (PM).*
