#!/usr/bin/env python3
"""Idempotent patcher for pybleno's BluetoothHCI.py on the SL2619.

Applies the 5 Python-3.12 / kernel-6.x fixes from
docs/references/old-dispenser-demo/pybleno-setup-guide.md (socket constants,
'rU'->'r', skip HCI_FILTER setsockopt, HCI_CHANNEL_RAW->USER, hciconfig devinfo
fallback) so pybleno binds an HCI_CHANNEL_USER socket and advertises. Run on the
host against an unpacked pybleno tree, then scp the tree to the board; pair with
board_fcntl_shim.py (board Python lacks the fcntl module). See
docs/deployment/sl2619-ble-bringup.md §5.

Usage: python3 patch_pybleno_bluetoothhci.py <path/to/BluetoothHCI.py>
"""
import pathlib
import sys

if len(sys.argv) != 2:
    sys.exit("usage: patch_pybleno_bluetoothhci.py <path/to/BluetoothHCI.py>")
F = pathlib.Path(sys.argv[1])
s = F.read_text()
orig = s

def repl(old, new, label, required=True):
    global s
    if old not in s:
        if required and new not in s:
            sys.exit(f"PATCH FAIL [{label}]: anchor not found and target absent")
        print(f"  [{label}] anchor absent (already applied?) — skipped")
        return
    s = s.replace(old, new, 1)
    print(f"  [{label}] applied")

# Patch 1: socket constants (Py3.12/Yocto may lack AF_BLUETOOTH et al.). Marker-guarded.
if "_SL2619_BT_CONSTS" not in s:
    block = (
        "import socket\n"
        "# _SL2619_BT_CONSTS: Py3.12/Yocto build lacks Bluetooth socket constants\n"
        "if not hasattr(socket, 'AF_BLUETOOTH'): socket.AF_BLUETOOTH = 31\n"
        "if not hasattr(socket, 'BTPROTO_HCI'): socket.BTPROTO_HCI = 1\n"
        "if not hasattr(socket, 'BTPROTO_L2CAP'): socket.BTPROTO_L2CAP = 0\n"
        "if not hasattr(socket, 'BTPROTO_RFCOMM'): socket.BTPROTO_RFCOMM = 3\n"
        "if not hasattr(socket, 'BTPROTO_SCO'): socket.BTPROTO_SCO = 2\n"
        "if not hasattr(socket, 'SOL_HCI'): socket.SOL_HCI = 0\n"
        "if not hasattr(socket, 'HCI_FILTER'): socket.HCI_FILTER = 2\n"
        "if not hasattr(socket, 'HCI_DATA_DIR'): socket.HCI_DATA_DIR = 1\n"
        "if not hasattr(socket, 'HCI_TIME_STAMP'): socket.HCI_TIME_STAMP = 3\n"
    )
    assert s.count("import socket\n") >= 1, "no 'import socket' line"
    s = s.replace("import socket\n", block, 1)
    print("  [1 constants] applied")
else:
    print("  [1 constants] marker present — skipped")

# Patch 2: 'rU' mode removed in Py3.11+
repl("os.fdopen(self.__r, 'rU')", "os.fdopen(self.__r, 'r')", "2 rU-mode")

# Patch 3: skip HCI_FILTER setsockopt (unsupported on HCI_CHANNEL_USER, kernel 6.x)
repl(
    "self._socket.setsockopt( socket.SOL_HCI, socket.HCI_FILTER, data )",
    "pass  # SL2619: HCI_FILTER setsockopt unsupported on HCI_CHANNEL_USER (kernel 6.x)",
    "3 set_filter",
)

# Patch 4: HCI_CHANNEL_RAW -> HCI_CHANNEL_USER (exclusive claim of a DOWN adapter)
repl(
    "self._socket.bind_hci(self.device_id, HCI_CHANNEL_RAW)",
    "self._socket.bind_hci(self.device_id, HCI_CHANNEL_USER)",
    "4 channel-user",
)

# Patch 5: ioctl HCIGETDEVINFO fails on HCI_CHANNEL_USER -> hciconfig fallback
old5 = "            # get device info\n            dev_info = self.get_device_info()"
new5 = (
    "            # get device info\n"
    "            try:\n"
    "                dev_info = self.get_device_info()\n"
    "            except OSError:\n"
    "                # HCIGETDEVINFO unsupported on HCI_CHANNEL_USER; read addr from hciconfig\n"
    "                import subprocess\n"
    "                _addr = '00:00:00:00:00:00'\n"
    "                _out = subprocess.check_output(['hciconfig', 'hci%d' % self.device_id]).decode()\n"
    "                for _line in _out.splitlines():\n"
    "                    if 'BD Address' in _line:\n"
    "                        _addr = _line.strip().split()[2]\n"
    "                        break\n"
    "                dev_info = {'addr': _addr, 'type': 0}"
)
repl(old5, new5, "5 devinfo-fallback")

F.write_text(s)
print("OK — patched" if s != orig else "no change (all already applied)")
