# ASTRA™ Support ticket — SL2619-RDK M.2 BT bring-up

## Contact us about
Software & SDK

## Summary*
SL2619-RDK M.2 Bluetooth: brcm_patchram_plus loops HCI_Reset, /dev/ttyS1 RX silent

## Description*

**Board:** SL2619-RDK
**Image:** Poky 5.0.9 (scarthgap), kernel 6.12.62
**DT compatible:** `syna,sl2619-rdk syna,sl2619`
**Related internal bug refs:** 37861, 37374

### Symptom

`brcm_patchram_plus` cannot bring up the M.2 socket's Bluetooth radio.
It loops `HCI_Reset` (`01 03 0c 00`) on `/dev/ttyS1` at 4-second cadence
indefinitely; the chip never responds. `/dev/ttyS1` returns zero bytes
on a 3-second listen at 115200 / 921600 / 3 000 000 baud after an
`rfkill unblock bluetooth` cycle.

Invocation (verbatim):

```
brcm_patchram_plus -d \
  --baudrate 3000000 --use_baudrate_for_download \
  --patchram /lib/firmware/bcm/SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz_wlbga_REF_sLNA_iLNA_ANT0.hcd \
  /dev/ttyS1
```

### What works

- **WiFi** over SDIO on the same combo chip: firmware
  `fw_sd_bcm43711.bin` loads, `wlan0` associates, carries DHCP / EAPOL /
  ARP / TCP traffic.
- **Kernel BT stack** present: `Bluetooth: Core ver 2.22`, `HCI UART
  driver ver 2.3`, H4 + BCSP protocols registered, RFCOMM/BNEP/HIDP
  built in.
- **Userspace**: `bluez5`, `bluetoothd` PID 704, `hciconfig`,
  `hciattach`, `btattach`, `rfkill`, `bluetoothctl` all installed.
- **UART1 controller**: `e5031000.uart` (snps,dw-apb-uart) probes,
  `ttyS1 at MMIO 0xe5031000 (irq=15, base_baud=6250000)`. `/dev/ttyS1`
  accepts writes (rc=0).
- **Patchram .hcd**: both
  `SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz_wlbga_REF_sLNA_iLNA_ANT0.hcd`
  and `BCM4345C5_003.006.006.1066.1126.hcd` are present under
  `/lib/firmware/bcm/`; both produce the identical HCI_Reset retry loop.

### Diagnostics already completed (audit summary)

15 hypotheses checked; 12 ruled out. See attached investigation doc
for the full matrix.

- **UART1 pinmux is correctly applied by the bootloader.** Direct
  devmem read of `pinctrl@0xe5025b00`:
  - `+0x0c = 0x00036246` → SM_GPIO7 (RXD), SM_GPIO8 (TXD), SM_GPIO14
    (CTSn) all = function 6 = `uart1`.
  - `+0x08 = 0x30249249` → SM_GPIO15 (RTSn) = function 6 = `uart1`.
  - The empty `pinctrl-0` on `uart@e5031000` in DT is therefore a red
    herring — the bootloader sets pinmux before the kernel boots.
- **`bt_reg_on` DT node** points at `phandle 0x0f, pin 5, ACTIVE_HIGH`,
  resolving to FXL6408 @ I²C 0x44, line 5. Kernel registers `gpio-565`
  with consumer `bt_power`. `bluetooth-rfkill` toggles this line on
  `rfkill unblock`. FXL state register `0x05` bit 5 reads 1 after
  unblock (line is electrically driven high; `/sys/kernel/debug/gpio`
  showing `out lo` is a debugfs read-back artifact of the FXL6408
  Input Status register returning 0 for output-configured pins).
- **32.768 kHz slow clock to the combo chip is not the BT-side
  blocker.** WiFi works, which requires the slow clock on
  BCM4345-family silicon. Manually muxing SM_GPIO30 → `sm_clkout`
  (function 7) on top of the existing config produced no change in BT
  behavior.

## SDK Version*
2.3.0

## Embedded Processor*
SL2619
