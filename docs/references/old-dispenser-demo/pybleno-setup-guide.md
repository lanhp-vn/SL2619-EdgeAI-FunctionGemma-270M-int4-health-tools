# pybleno Setup Guide for SL2619 (Python 3.12 + Yocto Scarthgap)

## Overview
This guide covers installing pybleno and fixing all Python 3.12 compatibility
issues on the Synaptics SL2619 board (Yocto Scarthgap, kernel 6.12.11).

---

## Prerequisites
- BT driver loaded: `btrtl.ko`, `btbcm.ko`, `btintel.ko`, `btusb.ko`
- BT firmware in place: `/lib/firmware/rtl_bt/rtl8822b_fw.bin` and `rtl8822b_config.bin`
- `hci0` interface UP: `hciconfig -a` shows `UP RUNNING`

---

## Step 1: Install pybleno

```bash
pip3 install pybleno --break-system-packages
```

---

## Step 2: Find the BluetoothHCI.py file to patch

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
echo "File location: $FILE"
```

---

## Step 3: Patch 1 — Add missing Bluetooth socket constants

Python 3.12 on Yocto is compiled without Bluetooth socket support.
All required constants must be added manually.

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
python3 -c "
content = open('$FILE').read()
patch = '''import socket
if not hasattr(socket, 'AF_BLUETOOTH'):
    socket.AF_BLUETOOTH = 31
if not hasattr(socket, 'BTPROTO_HCI'):
    socket.BTPROTO_HCI = 1
if not hasattr(socket, 'BTPROTO_L2CAP'):
    socket.BTPROTO_L2CAP = 0
if not hasattr(socket, 'BTPROTO_RFCOMM'):
    socket.BTPROTO_RFCOMM = 3
if not hasattr(socket, 'BTPROTO_SCO'):
    socket.BTPROTO_SCO = 2
if not hasattr(socket, 'SOL_HCI'):
    socket.SOL_HCI = 0
if not hasattr(socket, 'HCI_FILTER'):
    socket.HCI_FILTER = 2
if not hasattr(socket, 'HCI_DATA_DIR'):
    socket.HCI_DATA_DIR = 1
if not hasattr(socket, 'HCI_TIME_STAMP'):
    socket.HCI_TIME_STAMP = 3
'''
content = content.replace('import socket\n', patch, 1)
open('$FILE', 'w').write(content)
print('Done')
"
```

---

## Step 4: Patch 2 — Fix removed 'rU' file mode (Python 3.11+)

The `'rU'` universal newlines mode was removed in Python 3.11.

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
python3 -c "
content = open('$FILE').read()
content = content.replace(\"'rU'\", \"'r'\")
open('$FILE', 'w').write(content)
print('Done')
"
```

---

## Step 5: Patch 3 — Fix HCI_FILTER setsockopt failure

`setsockopt(SOL_HCI, HCI_FILTER)` fails with `HCI_CHANNEL_USER` on kernel 6.x.
Skip the filter — `HCI_CHANNEL_USER` receives all HCI events anyway.

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
python3 -c "
content = open('$FILE').read()
old = '''    def set_filter(self, data):
        # flt = bluez.hci_filter_new()
        # bluez.hci_filter_all_events(flt)
        # bluez.hci_filter_set_ptype(flt, bluez.HCI_EVENT_PKT)
        self._socket.setsockopt( socket.SOL_HCI, socket.HCI_FILTER, data )   
        pass
        #self._socket.setsockopt(socket.SOL_HCI, socket.HCI_FILTER, data)'''
new = '''    def set_filter(self, data):
        # setsockopt HCI_FILTER not supported with HCI_CHANNEL_USER on kernel 6.x
        pass'''
content = content.replace(old, new)
open('$FILE', 'w').write(content)
print('Done')
"
```

---

## Step 6: Patch 4 — Switch to HCI_CHANNEL_USER

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
python3 -c "
content = open('$FILE').read()
content = content.replace(
    'self._socket.bind_hci(self.device_id, HCI_CHANNEL_RAW)',
    'self._socket.bind_hci(self.device_id, HCI_CHANNEL_USER)'
)
open('$FILE', 'w').write(content)
print('Done')
"
```

---

## Step 7: Patch 5 — Fix kernel_disconnect_workarounds ioctl failure

`ioctl HCIGETDEVINFO` fails on `HCI_CHANNEL_USER` socket.
Fall back to `hciconfig` to get the adapter address.

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")
python3 -c "
content = open('$FILE').read()
old = '''            # get device info
            dev_info = self.get_device_info()'''
new = '''            # get device info
            try:
                dev_info = self.get_device_info()
            except OSError:
                # ioctl not supported with HCI_CHANNEL_USER, get addr from hciconfig
                import subprocess
                out = subprocess.check_output([\"hciconfig\", \"hci0\"]).decode()
                for line in out.splitlines():
                    if \"BD Address\" in line:
                        addr = line.strip().split()[2]
                        break
                dev_info = {\"addr\": addr, \"type\": 0}'''
content = content.replace(old, new)
open('$FILE', 'w').write(content)
print('Done')
"
```

---

## Step 8: Verify all patches applied

```bash
FILE=$(find /usr/lib/python3.12 -name "BluetoothHCI.py")

echo "=== Patch 1: Bluetooth constants ==="
grep -c "AF_BLUETOOTH" $FILE

echo "=== Patch 2: rU mode removed ==="
grep "rU" $FILE || echo "OK - rU not found"

echo "=== Patch 3: set_filter skipped ==="
grep -A2 "def set_filter" $FILE

echo "=== Patch 4: HCI_CHANNEL_USER ==="
grep "bind_hci" $FILE

echo "=== Patch 5: hciconfig fallback ==="
grep "hciconfig" $FILE
```

---

## Step 9: Run the BLE peripheral application

```bash
# Bring hci0 down first (required for HCI_CHANNEL_USER exclusive access)
hciconfig hci0 down

# Run the application
BLENO_HCI_DEVICE_ID=0 python3 ble_peripheral.py
```

Expected output:
```
[INFO] Keyboard ready: press ENTER to notify, type 'quit' to exit.
[INFO] NousVoice BLE peripheral running...
[INFO] BLE state: poweredOn
[INFO] Advertising as 'NousVoice' | service 0x00FB | mfr data: ffffffdeadbeef
[INFO] Advertising started OK
[INFO]   Primary service : 0xFFB0
[INFO]   RW char         : 0xFFB1
[INFO]   Notify char     : 0xFFB2
```

---

## Step 10: Test with phone

1. Install **nRF Connect** app on your phone
2. Scan → find **NousVoice**
3. Check manufacturer data: `FF FF DE AD BE EF`
4. Connect → find service `FFB0`
5. Enable notifications on characteristic `FFB2`
6. Press **ENTER** on board → phone receives `5A A5 01 00`
7. Write to `FFB1` from phone → board logs the received value

---

## Keyboard commands while running

| Key | Action |
|-----|--------|
| `ENTER` | Send notification `5A A5 01 00` to connected central |
| `quit` + ENTER | Gracefully stop the peripheral |
| `Ctrl+C` | Force stop |

---

## Notification packet format

```
byte[0] = 0x5A   ← header
byte[1] = 0xA5   ← header
byte[2] = 0x01   ← cmd    (0x01 = face recognized)
byte[3] = 0x00   ← status (0x00 = success)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'dbus'` | Use pybleno instead of bluezero |
| `AF_BLUETOOTH not found` | Apply Patch 1 (Step 3) |
| `invalid mode 'rU'` | Apply Patch 2 (Step 4) |
| `Invalid argument` on setsockopt | Apply Patch 3 (Step 5) |
| `BTPROTO_L2CAP not found` | Apply Patch 1 again — all constants at once |
| `File descriptor in bad state` | Apply Patch 4 (Step 6) — use HCI_CHANNEL_USER |
| `OSError Errno 77` on ioctl | Apply Patch 5 (Step 7) — hciconfig fallback |
| Phone connects then immediately drops | Normal until GATT services are set — retry |
| `Not connected — notification skipped` | Enable notifications on FFB2 in nRF Connect first |
