#!/usr/bin/env python3
"""
BLE Peripheral - NousVoice WiFi Configurator (pybleno)
=======================================================
Board       : Synaptics SL2619 (hci0 = RTL8822BU USB adapter)
Advertising : Name "NousVoice", 16-bit Service UUID 0x00FB
              Manufacturer data: {0xFF, 0xFF, 0xDE, 0xAD, 0xBE, 0xEF}

GATT Layout:
  Primary Service 0xFFB0  — General control
    0xFFB1 : Read/Write   — General RW
    0xFFB2 : Notify       — General notifications

  Primary Service 0xFFB3  — WiFi Configuration
    0xFFB4 : Write        — SSID   (write plain text, e.g. "MyWiFi")
    0xFFB5 : Write        — Password (write plain text, e.g. "secret123")
    0xFFB6 : Write        — Command (write 0x01 to connect, 0x00 to disconnect)
    0xFFB7 : Notify       — WiFi status notifications

WiFi status notification format (sent on 0xFFB7):
  byte[0] = 0x5A  header
  byte[1] = 0xA5  header
  byte[2] = cmd   (0x02 = wifi connect result)
  byte[3] = status(0x00 = success, 0x01 = failed, 0x02 = connecting)

General notification format (sent on 0xFFB2 on ENTER key):
  byte[0] = 0x5A  header
  byte[1] = 0xA5  header
  byte[2] = cmd   (0x01 = face recognized)
  byte[3] = status(0x00 = success)

Usage:
  hciconfig hci0 down
  BLENO_HCI_DEVICE_ID=0 python3 ble_peripheral.py

LightBlue app instructions:
  1. Connect to NousVoice
  2. Write SSID  to 0xFFB4 as UTF-8 string
  3. Write Pass  to 0xFFB5 as UTF-8 string
  4. Write 0x01  to 0xFFB6 to trigger connection
  5. Subscribe to 0xFFB7 for connection result
"""

import sys
import array
import logging
import subprocess
import threading
import time
from pybleno import Bleno, BlenoPrimaryService, Characteristic

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── UUIDs ──────────────────────────────────────────────────────────────────────
ADV_SERVICE_UUID   = "00FB"
PRIMARY_SVC_UUID   = "FFB0"   # general service
CHAR_RW_UUID       = "FFB1"   # general read/write
CHAR_NOTIFY_UUID   = "FFB2"   # general notify

WIFI_SVC_UUID      = "FFB3"   # WiFi config service
CHAR_SSID_UUID     = "FFB4"   # write SSID
CHAR_PASS_UUID     = "FFB5"   # write password
CHAR_WIFI_CMD_UUID = "FFB6"   # write command (0x01=connect, 0x00=disconnect)
CHAR_WIFI_STA_UUID = "FFB7"   # notify WiFi status
CHAR_WIFI_IP_UUID  = "FFB8"   # read WiFi IP address

DEVICE_NAME        = "NousVoice"
MANUFACTURER_DATA  = bytes([0xFF, 0xFF, 0xDE, 0xAD, 0xBE, 0xEF])

# ── WiFi command values ────────────────────────────────────────────────────────
CMD_WIFI_CONNECT    = 0x01
CMD_WIFI_DISCONNECT = 0x00

# ── WiFi status notification values ───────────────────────────────────────────
WIFI_CMD_RESULT  = 0x02
WIFI_STATUS_OK   = 0x00
WIFI_STATUS_FAIL = 0x01
WIFI_STATUS_BUSY = 0x02

# ── Shared state ───────────────────────────────────────────────────────────────
connected = False
wifi_ssid = ""
wifi_pass = ""


# ── EIR helpers ───────────────────────────────────────────────────────────────

def _ad_struct(ad_type, data):
    b = bytes(data) if not isinstance(data, bytes) else data
    return bytes([len(b) + 1, ad_type]) + b

def _build_adv_data(service_uuid_str, name):
    """
    Advertising data (31 bytes max):
      0x01 - Flags (LE General Discoverable, no BR/EDR)
      0x03 - Complete list of 16-bit Service UUIDs
      0x09 - Complete Local Name
    """
    flags    = _ad_struct(0x01, bytes([0x06]))
    uuid_le  = bytes.fromhex(service_uuid_str)[::-1]   # little-endian
    svc_uuid = _ad_struct(0x03, uuid_le)
    dev_name = _ad_struct(0x09, name.encode("utf-8"))
    return flags + svc_uuid + dev_name

def _build_scan_data(manufacturer_data):
    """
    Scan response data (31 bytes max):
      0xFF - Manufacturer Specific Data
             byte[0..1] = company ID little-endian (0xFFFF)
             byte[2..]  = custom payload (0xDEADBEEF)
    """
    return _ad_struct(0xFF, manufacturer_data)


# ── WiFi helper ───────────────────────────────────────────────────────────────

def connect_wifi(ssid, password, status_callback):
    """
    Connect to WiFi using wpa_supplicant.
    Calls status_callback(success: bool) when done.
    Runs in a background thread to avoid blocking BLE.
    """
    def _run():
        log.info(f"[WiFi] Connecting to SSID: '{ssid}'")
        status_callback(WIFI_STATUS_BUSY)

        try:
            # Build wpa_supplicant config content
            config = f"""ctrl_interface=/var/run/wpa_supplicant
update_config=1

network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""
            # Save to temporary config (used for this session)
            with open("/tmp/wpa_ble.conf", "w") as f:
                f.write(config)

            # Save to permanent config (persists after reboot)
            import os
            os.makedirs("/etc/wpa_supplicant", exist_ok=True)
            perm_conf = "/etc/wpa_supplicant/wpa_supplicant-wlu1u4i2.conf"
            with open(perm_conf, "w") as f:
                f.write(config)
            os.chmod(perm_conf, 0o600)
            log.info(f"[WiFi] Credentials saved permanently to {perm_conf}")

            # Kill any existing wpa_supplicant
            subprocess.run(["killall", "wpa_supplicant"],
                           capture_output=True)
            time.sleep(1)

            # Find WiFi interface (wlu* on SL2619)
            result = subprocess.run(
                ["ip", "link", "show"],
                capture_output=True, text=True
            )
            wifi_iface = None
            for line in result.stdout.splitlines():
                if "wlu" in line or "wlan" in line:
                    wifi_iface = line.split(":")[1].strip()
                    break

            if not wifi_iface:
                log.error("[WiFi] No WiFi interface found")
                status_callback(WIFI_STATUS_FAIL)
                return

            log.info(f"[WiFi] Using interface: {wifi_iface}")

            # Start wpa_supplicant
            subprocess.run([
                "wpa_supplicant", "-B",
                "-i", wifi_iface,
                "-c", "/tmp/wpa_ble.conf"
            ], capture_output=True)
            time.sleep(3)

            # Wait for WPA association (up to 15 seconds)
            log.info("[WiFi] Waiting for WPA association...")
            associated = False
            for attempt in range(15):
                check = subprocess.run(
                    ["wpa_cli", "-i", wifi_iface, "status"],
                    capture_output=True, text=True
                )
                if "wpa_state=COMPLETED" in check.stdout:
                    associated = True
                    log.info(f"[WiFi] Associated after {attempt+1}s")
                    break
                time.sleep(1)

            if not associated:
                log.warning(f"[WiFi] Failed to associate to '{ssid}'")
                status_callback(WIFI_STATUS_FAIL)
                return

            # Request DHCP with retries
            log.info("[WiFi] Requesting DHCP...")
            dhcp_ok = False
            for attempt in range(3):
                log.info(f"[WiFi] DHCP attempt {attempt+1}/3 ...")
                result = subprocess.run(
                    ["udhcpc", "-i", wifi_iface,
                     "-t", "10",    # retry sending DISCOVER 10 times
                     "-T", "3",     # timeout 3s between retries
                     "-A", "3",     # wait 3s before restarting
                     "-q"],         # quit after obtaining lease
                    capture_output=True,
                    timeout=40
                )
                # Check if we got an IP
                ip_check = subprocess.run(
                    ["ip", "addr", "show", "dev", wifi_iface],
                    capture_output=True, text=True
                )
                if "inet " in ip_check.stdout:
                    dhcp_ok = True
                    log.info(f"[WiFi] DHCP OK on attempt {attempt+1}")
                    break
                log.warning(f"[WiFi] DHCP attempt {attempt+1} failed, retrying...")
                time.sleep(2)

            if not dhcp_ok:
                log.warning("[WiFi] DHCP failed after 3 attempts")
                status_callback(WIFI_STATUS_FAIL)
                return

            log.info(f"[WiFi] Connected to '{ssid}' and got IP OK")

            # Get gateway from routing table
            route = subprocess.run(
                ["ip", "route", "show", "dev", wifi_iface],
                capture_output=True, text=True
            )
            gateway = None
            for line in route.stdout.splitlines():
                if line.startswith("default"):
                    gateway = line.split()[2]
                    break

            # If udhcpc did not set default route, guess gateway from subnet
            if not gateway:
                for line in route.stdout.splitlines():
                    if "scope link" in line:
                        network = line.strip().split()[0]
                        base = ".".join(network.split("/")[0].split(".")[:3])
                        gateway = f"{base}.1"
                        break

            if gateway:
                # Remove old default route and add via WiFi gateway
                subprocess.run(["ip", "route", "del", "default"],
                               capture_output=True)
                subprocess.run(
                    ["ip", "route", "add", "default",
                     "via", gateway, "dev", wifi_iface, "metric", "20"],
                    capture_output=True
                )
                log.info(f"[WiFi] Default route set via {gateway} on {wifi_iface} metric 20 (eth0=10)")
            else:
                log.warning("[WiFi] Could not determine gateway")

            # Set DNS if resolv.conf is empty
            try:
                resolv = open("/etc/resolv.conf").read().strip()
            except Exception:
                resolv = ""
            if not resolv or "nameserver" not in resolv:
                with open("/etc/resolv.conf", "w") as f:
                    f.write("nameserver 8.8.8.8\nnameserver 8.8.4.4\n")
                log.info("[WiFi] DNS set to 8.8.8.8")

            # Verify internet connectivity
            ping = subprocess.run(
                ["ping", "-c", "1", "-W", "3", "-I", wifi_iface, "8.8.8.8"],
                capture_output=True
            )
            if ping.returncode == 0:
                log.info("[WiFi] Internet reachable OK")
            else:
                log.warning("[WiFi] Internet not reachable - check router/gateway")

            status_callback(WIFI_STATUS_OK)

        except Exception as e:
            log.error(f"[WiFi] Exception: {e}")
            status_callback(WIFI_STATUS_FAIL)

    threading.Thread(target=_run, daemon=True).start()


def disconnect_wifi(status_callback):
    """Disconnect WiFi."""
    try:
        subprocess.run(["killall", "wpa_supplicant"], capture_output=True)
        log.info("[WiFi] Disconnected")
        status_callback(WIFI_STATUS_OK)
    except Exception as e:
        log.error(f"[WiFi] Disconnect error: {e}")
        status_callback(WIFI_STATUS_FAIL)


# ── General Characteristics (Service 0xFFB0) ──────────────────────────────────

class RWCharacteristic(Characteristic):
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_RW_UUID,
            'properties': ['read', 'write', 'writeWithoutResponse'],
            'value': None
        })
        self._value = array.array('B', [0x00])

    def onReadRequest(self, offset, callback):
        log.info(f"[0xFFB1] Read → {bytes(self._value).hex()}")
        callback(Characteristic.RESULT_SUCCESS, self._value)

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        self._value = array.array('B', data)
        log.info(f"[0xFFB1] Write ← {bytes(self._value).hex()}")
        callback(Characteristic.RESULT_SUCCESS)


class NotifyCharacteristic(Characteristic):
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_NOTIFY_UUID,
            'properties': ['notify'],
            'value': None
        })
        self._update_value_callback = None

    def onSubscribe(self, maxValueSize, updateValueCallback):
        log.info("[0xFFB2] Notifications ENABLED")
        self._update_value_callback = updateValueCallback

    def onUnsubscribe(self):
        log.info("[0xFFB2] Notifications DISABLED")
        self._update_value_callback = None

    def send_notify(self, cmd=0x01, status=0x00):
        if self._update_value_callback is None:
            log.warning("[0xFFB2] No subscriber")
            return
        data = array.array('B', [0x5A, 0xA5, cmd, status])
        log.info(f"[0xFFB2] Notify → {bytes(data).hex()}")
        self._update_value_callback(data)


# ── WiFi Characteristics (Service 0xFFB3) ─────────────────────────────────────

class SSIDCharacteristic(Characteristic):
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_SSID_UUID,
            'properties': ['write', 'writeWithoutResponse'],
            'value': None
        })

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        global wifi_ssid
        wifi_ssid = bytes(data).decode('utf-8', errors='ignore').strip()
        log.info(f"[0xFFB4] SSID set: '{wifi_ssid}'")
        callback(Characteristic.RESULT_SUCCESS)


class PasswordCharacteristic(Characteristic):
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_PASS_UUID,
            'properties': ['write', 'writeWithoutResponse'],
            'value': None
        })

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        global wifi_pass
        wifi_pass = bytes(data).decode('utf-8', errors='ignore').strip()
        log.info(f"[0xFFB5] Password set: {'*' * len(wifi_pass)}")
        callback(Characteristic.RESULT_SUCCESS)


class WiFiStatusCharacteristic(Characteristic):
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_WIFI_STA_UUID,
            'properties': ['notify'],
            'value': None
        })
        self._update_value_callback = None

    def onSubscribe(self, maxValueSize, updateValueCallback):
        log.info("[0xFFB7] WiFi status notifications ENABLED")
        self._update_value_callback = updateValueCallback

    def onUnsubscribe(self):
        log.info("[0xFFB7] WiFi status notifications DISABLED")
        self._update_value_callback = None

    def send_status(self, status):
        """
        Send WiFi status notification:
          [0] = 0x5A
          [1] = 0xA5
          [2] = 0x02  (WiFi connect result command)
          [3] = status (0x00=success, 0x01=failed, 0x02=connecting)
        """
        if self._update_value_callback is None:
            log.warning("[0xFFB7] No subscriber for WiFi status")
            return
        status_str = {
            WIFI_STATUS_OK:   "SUCCESS",
            WIFI_STATUS_FAIL: "FAILED",
            WIFI_STATUS_BUSY: "CONNECTING..."
        }.get(status, "UNKNOWN")
        data = array.array('B', [0x5A, 0xA5, WIFI_CMD_RESULT, status])
        log.info(f"[0xFFB7] WiFi status → {bytes(data).hex()} ({status_str})")
        self._update_value_callback(data)



class WiFiIPCharacteristic(Characteristic):
    """
    0xFFB8 - Read the WiFi IP address after successful connection.
    Returns the IP as a UTF-8 string, e.g. "192.168.1.56"
    Returns "0.0.0.0" if not connected.
    """
    def __init__(self):
        Characteristic.__init__(self, {
            'uuid': CHAR_WIFI_IP_UUID,
            'properties': ['read'],
            'value': None
        })

    def onReadRequest(self, offset, callback):
        ip = self._get_wifi_ip()
        log.info(f"[0xFFB8] Read IP → {ip}")
        callback(Characteristic.RESULT_SUCCESS,
                 array.array('B', ip.encode('utf-8')))

    def _get_wifi_ip(self):
        try:
            import subprocess
            result = subprocess.run(
                ['ip', 'addr', 'show'],
                capture_output=True, text=True
            )
            iface = None
            ip = None
            for line in result.stdout.splitlines():
                # Find wlu* or wlan* interface line
                if ': wlu' in line or ': wlan' in line:
                    iface = line.split(':')[1].strip().split('@')[0]
                # Find inet address on that interface
                if iface and 'inet ' in line and 'scope global' in line:
                    ip = line.strip().split()[1].split('/')[0]
                    return ip
            return "0.0.0.0"
        except Exception as e:
            log.warning(f"[0xFFB8] IP lookup error: {e}")
            return "0.0.0.0"


class WiFiCmdCharacteristic(Characteristic):
    def __init__(self, wifi_status_char):
        Characteristic.__init__(self, {
            'uuid': CHAR_WIFI_CMD_UUID,
            'properties': ['write', 'writeWithoutResponse'],
            'value': None
        })
        self._wifi_status_char = wifi_status_char

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        cmd = data[0] if len(data) > 0 else 0xFF
        log.info(f"[0xFFB6] WiFi command: 0x{cmd:02X}")

        if cmd == CMD_WIFI_CONNECT:
            if not wifi_ssid:
                log.warning("[0xFFB6] SSID not set — write SSID to 0xFFB4 first")
                self._wifi_status_char.send_status(WIFI_STATUS_FAIL)
            else:
                connect_wifi(
                    wifi_ssid,
                    wifi_pass,
                    self._wifi_status_char.send_status
                )

        elif cmd == CMD_WIFI_DISCONNECT:
            disconnect_wifi(self._wifi_status_char.send_status)

        else:
            log.warning(f"[0xFFB6] Unknown command: 0x{cmd:02X}")

        callback(Characteristic.RESULT_SUCCESS)


# ── Bleno instance and characteristics ────────────────────────────────────────
bleno           = Bleno()
rw_char         = RWCharacteristic()
notify_char     = NotifyCharacteristic()
ssid_char       = SSIDCharacteristic()
pass_char       = PasswordCharacteristic()
wifi_status_char= WiFiStatusCharacteristic()
wifi_ip_char    = WiFiIPCharacteristic()
wifi_cmd_char   = WiFiCmdCharacteristic(wifi_status_char)


# ── Bleno event callbacks ──────────────────────────────────────────────────────

def on_state_change(state):
    log.info(f"BLE state: {state}")
    if state == "poweredOn":
        adv_data  = _build_adv_data(ADV_SERVICE_UUID, DEVICE_NAME)
        scan_data = _build_scan_data(MANUFACTURER_DATA)
        bleno.startAdvertisingWithEIRData(adv_data, scan_data)
        log.info(f"Advertising as '{DEVICE_NAME}' | service 0x{ADV_SERVICE_UUID}")
    else:
        bleno.stopAdvertising()


def on_advertising_start(error):
    if error:
        log.error(f"Advertising error: {error}")
        return
    log.info("Advertising started OK")
    log.info("  Service 0xFFB0  — General control")
    log.info("    0xFFB1 Read/Write")
    log.info("    0xFFB2 Notify (ENTER key → 5A A5 01 00)")
    log.info("  Service 0xFFB3  — WiFi configuration")
    log.info("    0xFFB4 Write SSID")
    log.info("    0xFFB5 Write Password")
    log.info("    0xFFB6 Write 0x01=connect / 0x00=disconnect")
    log.info("    0xFFB7 Notify WiFi status")

    bleno.setServices([
        BlenoPrimaryService({
            'uuid': PRIMARY_SVC_UUID,
            'characteristics': [rw_char, notify_char]
        }),
        BlenoPrimaryService({
            'uuid': WIFI_SVC_UUID,
            'characteristics': [
                ssid_char,
                pass_char,
                wifi_cmd_char,
                wifi_status_char,
                wifi_ip_char
            ]
        })
    ])


def on_accept(client_address):
    global connected
    connected = True
    log.info(f"★ Connected: {client_address}")


def on_disconnect(client_address):
    global connected
    connected = False
    log.info(f"✗ Disconnected: {client_address}")
    adv_data  = _build_adv_data(ADV_SERVICE_UUID, DEVICE_NAME)
    scan_data = _build_scan_data(MANUFACTURER_DATA)
    bleno.startAdvertisingWithEIRData(adv_data, scan_data)
    log.info("Re-advertising...")


# ── Keyboard listener ──────────────────────────────────────────────────────────

def process_command(cmd):
    """Central command processor — used by both pipe and keyboard."""
    cmd = cmd.strip().lower()
    if cmd == "quit":
        log.info("Quit command received — stopping.")
        bleno.stopAdvertising()
        bleno.disconnect()
        sys.exit(0)
    elif cmd == "wifi":
        log.info(f"WiFi config — SSID: '{wifi_ssid}' | "
                 f"Pass: {'*' * len(wifi_pass) if wifi_pass else '(not set)'}")
        log.info(f"WiFi IP: {wifi_ip_char._get_wifi_ip()}")
    elif cmd == "dispense":
        if connected:
            notify_char.send_notify(cmd=0x01, status=0x00)
        else:
            log.info("Not connected — dispense skipped.")
    elif cmd == "":
        pass   # ignore empty lines
    else:
        log.info(f"Unknown command: '{cmd}' | valid: dispense, wifi, quit")


def keyboard_listener():
    import os

    # ── Named pipe (FIFO) listener — works with or without TTY ────────────────
    FIFO_PATH = "/tmp/ble_cmd"

    # Create FIFO if not exists
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)
        log.info(f"Command pipe created: {FIFO_PATH}")

    log.info(f"Commands via pipe: echo dispense > {FIFO_PATH}")
    log.info(f"Commands via pipe: echo wifi     > {FIFO_PATH}")
    log.info(f"Commands via pipe: echo quit     > {FIFO_PATH}")

    if os.isatty(sys.stdin.fileno()):
        log.info("Interactive mode: type dispense / wifi / quit")

    while True:
        try:
            # Open FIFO — blocks until someone writes to it
            with open(FIFO_PATH, "r") as fifo:
                for line in fifo:
                    process_command(line)
        except Exception as e:
            log.warning(f"Pipe error: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bleno.on("stateChange",      on_state_change)
    bleno.on("advertisingStart", on_advertising_start)
    bleno.on("accept",           on_accept)
    bleno.on("disconnect",       on_disconnect)

    bleno.start()

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    log.info("NousVoice BLE peripheral running... (Ctrl+C to stop)")

    # Keep main thread alive using an Event instead of kb_thread.join()
    # This works both interactively and as a systemd service (no TTY)
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        log.info("Signal received — stopping.")
        bleno.stopAdvertising()
        bleno.disconnect()
        stop_event.set()

    import signal

    def sigusr1_handler(sig, frame):
        """SIGUSR1 = virtual ENTER key, send notification"""
        log.info("SIGUSR1 received — sending notification")
        if connected:
            notify_char.send_notify(cmd=0x01, status=0x00)
        else:
            log.info("Not connected — notification skipped.")

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)
    log.info("Send SIGUSR1 to dispense: kill -USR1 $(pgrep -f ble_peripheral.py)")

    stop_event.wait()   # blocks forever until SIGTERM or SIGINT
    sys.exit(0)
