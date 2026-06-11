"""BLE peripheral client for the dispenser demo (plan §6.2 wire contract).

Three pieces:

- `BleClient` — a `@runtime_checkable` Protocol that pins the four methods
  the demo needs (`start_advertising`, `wait_for_subscriber`,
  `send_dispense_notify`, `stop`). Production wires
  `PyBlenoBleClient`; unit tests inject `MockBleClient`.
- `MockBleClient` — in-process fake. Records every byte sequence pushed
  through `send_dispense_notify` and lets tests simulate a central
  subscribing / unsubscribing via `_simulate_subscribe()` /
  `_simulate_unsubscribe()`.
- `PyBlenoBleClient` — real pybleno-backed peripheral, lazily imported so
  the module loads on hosts that don't ship pybleno (e.g. CI). The hci
  device index is plumbed through `BLENO_HCI_DEVICE_ID` *before* the first
  pybleno import — bleno reads that env var at module-init time, so
  setting it after-the-fact is a silent footgun.

Scope: BLE notify path only (the dispense signal). The legacy
`docs/references/old-dispenser-demo/ble_peripheral.py` carries WiFi-config
characteristics (FFB3..FFB8); those are intentionally omitted here. Add a
second service later if needed — don't grow this one.

Wire contract (plan §6.2):

    Advertising device name      "NousVoice"
    Advertising service UUID     0x00FB
    Primary service UUID         0xFFB0
    Notify characteristic UUID   0xFFB2
    Dispense notification        bytes [0x5A, 0xA5, 0x01, 0x00]

The dispense payload is *not* configurable from the public API — it's the
fixed wire contract with the ESP32 firmware. If the contract ever changes,
update `DISPENSE_PAYLOAD` and the constant table in the plan in the same
PR.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from types import TracebackType

log = logging.getLogger(__name__)


# ── Wire contract constants (plan §6.2) ────────────────────────────────────

DEVICE_NAME: str = "NousVoice"
ADV_SERVICE_UUID: str = "00FB"
PRIMARY_SERVICE_UUID: str = "FFB0"
NOTIFY_CHAR_UUID: str = "FFB2"

# 4-byte fixed dispense payload: [header_hi, header_lo, cmd, status].
# Matches the legacy general-notify on 0xFFB2 in
# docs/references/old-dispenser-demo/ble_peripheral.py.
DISPENSE_PAYLOAD: bytes = bytes([0x5A, 0xA5, 0x01, 0x00])


# ── Protocol ───────────────────────────────────────────────────────────────


@runtime_checkable
class BleClient(Protocol):
    """Structural interface for the dispenser BLE peripheral.

    Implementations need not inherit from this class — Python's structural
    typing matches by method shape. Used by `dispense_medication` (Phase 3
    integration) and by `scripts/dispenser_demo/deploy/ble_test.py`.
    """

    def start_advertising(self) -> None:
        """Begin BLE advertising under `DEVICE_NAME` / `ADV_SERVICE_UUID`.

        Blocks until the radio reports `poweredOn` and advertising has
        actually started. Raises on adapter / stack failure.
        """

    def wait_for_subscriber(self, timeout_s: float | None = None) -> bool:
        """Block until a central subscribes to the notify characteristic.

        Returns `True` if a subscriber connected within the deadline,
        `False` on timeout. `timeout_s=None` waits forever. Must not
        poll-sleep — implementations gate on `threading.Event.wait`.
        """

    def send_dispense_notify(self) -> bool:
        """Push the fixed `DISPENSE_PAYLOAD` to any subscribed central.

        Returns `True` when the notify was emitted to a live subscriber,
        `False` when nothing is subscribed (so callers can degrade to the
        `ble_not_connected` branch without raising).
        """

    def stop(self) -> None:
        """Stop advertising and release adapter resources. Idempotent."""


# ── Mock for host unit tests ───────────────────────────────────────────────


class MockBleClient:
    """In-process `BleClient` used by host unit tests.

    Records every emitted notify in `sent_notifications` so assertions can
    check the exact wire bytes. `_simulate_subscribe()` flips the
    subscriber flag so `send_dispense_notify` succeeds; absent that, it
    returns `False` (mirroring real pybleno behavior with no central).
    """

    def __init__(self) -> None:
        self._advertising: bool = False
        self._subscribed: bool = False
        self._stopped: bool = False
        self._subscriber_event: threading.Event = threading.Event()
        self.sent_notifications: list[bytes] = []
        # Test knob: when True, a *subscribed* notify raises instead of
        # recording — stands in for pybleno's HCI write throwing on a
        # mid-notify disconnect, exercising the dispatch seam's degrade path.
        self.raise_on_notify: bool = False

    def start_advertising(self) -> None:
        if self._stopped:
            raise RuntimeError("BleClient: stop() already called; cannot restart")
        self._advertising = True

    def wait_for_subscriber(self, timeout_s: float | None = None) -> bool:
        return self._subscriber_event.wait(timeout=timeout_s)

    def send_dispense_notify(self) -> bool:
        if not self._subscribed:
            log.info("MockBleClient: send_dispense_notify with no subscriber — returning False")
            return False
        if self.raise_on_notify:
            raise RuntimeError("MockBleClient: simulated mid-notify radio failure")
        self.sent_notifications.append(bytes(DISPENSE_PAYLOAD))
        return True

    def stop(self) -> None:
        self._advertising = False
        self._subscribed = False
        self._stopped = True
        self._subscriber_event.clear()

    # ── Test helpers (not part of the BleClient Protocol) ─────────────────

    @property
    def advertising(self) -> bool:
        return self._advertising

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _simulate_subscribe(self) -> None:
        self._subscribed = True
        self._subscriber_event.set()

    def _simulate_unsubscribe(self) -> None:
        self._subscribed = False
        self._subscriber_event.clear()


# ── pybleno-backed implementation ──────────────────────────────────────────


class PyBlenoBleClient:
    """Real BLE peripheral backed by `pybleno` against BlueZ.

    pybleno is imported lazily on `start_advertising()` so this module
    stays importable on hosts that don't ship pybleno (CI, dev laptops).
    `BLENO_HCI_DEVICE_ID` is set *before* the import — bleno snapshots
    that env var at module-init time.

    Threading: bleno runs its libuv loop on a background thread. The two
    points where we observe its state — `start_advertising` (wait for
    `advertisingStart`) and `wait_for_subscriber` (wait for `onSubscribe`
    on the notify characteristic) — gate on `threading.Event`s; we never
    poll-sleep.
    """

    def __init__(self, hci_index: int = 0, device_name: str = DEVICE_NAME) -> None:
        if hci_index < 0:
            raise ValueError(f"hci_index must be >= 0, got {hci_index}")
        self._hci_index: int = hci_index
        self._device_name: str = device_name
        self._bleno: Any | None = None
        self._notify_char: Any | None = None
        self._adv_started: threading.Event = threading.Event()
        self._subscriber_event: threading.Event = threading.Event()
        self._adv_error: BaseException | None = None
        self._stopped: bool = False

    # The pybleno imports are inside the methods that need them so this
    # module loads cleanly on hosts without the package installed.

    def _build_adv_data(self) -> bytes:
        """Construct EIR advertising payload (flags + 16-bit svc UUID + name).

        Matches the legacy `_build_adv_data` in
        docs/references/old-dispenser-demo/ble_peripheral.py.
        """
        flags = bytes([0x02, 0x01, 0x06])
        uuid_le = bytes.fromhex(ADV_SERVICE_UUID)[::-1]
        svc_uuid = bytes([len(uuid_le) + 1, 0x03]) + uuid_le
        name = self._device_name.encode("utf-8")
        dev_name = bytes([len(name) + 1, 0x09]) + name
        return flags + svc_uuid + dev_name

    def _import_pybleno(self) -> Any:
        # Set BLENO_HCI_DEVICE_ID *before* the import. bleno reads the env
        # var at module-init; setting it later is a silent no-op.
        import os

        os.environ["BLENO_HCI_DEVICE_ID"] = str(self._hci_index)
        import pybleno  # type: ignore[import-not-found]

        return pybleno

    def start_advertising(self) -> None:
        if self._stopped:
            raise RuntimeError("PyBlenoBleClient: stop() already called; cannot restart")
        pybleno = self._import_pybleno()

        outer = self

        class _NotifyCharacteristic(pybleno.Characteristic):  # type: ignore[misc,name-defined]
            def __init__(self) -> None:
                super().__init__(
                    {
                        "uuid": NOTIFY_CHAR_UUID,
                        "properties": ["notify"],
                        "value": None,
                    }
                )
                self._update_value_callback: Any | None = None

            def onSubscribe(self, _max_value_size: int, update_value_callback: Any) -> None:  # noqa: N802 — pybleno API
                log.info("PyBlenoBleClient: central subscribed to %s", NOTIFY_CHAR_UUID)
                self._update_value_callback = update_value_callback
                outer._subscriber_event.set()

            def onUnsubscribe(self) -> None:  # noqa: N802 — pybleno API
                log.info("PyBlenoBleClient: central unsubscribed from %s", NOTIFY_CHAR_UUID)
                self._update_value_callback = None
                outer._subscriber_event.clear()

            def push(self, payload: bytes) -> bool:
                if self._update_value_callback is None:
                    return False
                import array

                self._update_value_callback(array.array("B", payload))
                return True

        notify_char = _NotifyCharacteristic()
        bleno = pybleno.Bleno()

        def _on_state_change(state: str) -> None:
            log.info("PyBlenoBleClient: BLE state = %s", state)
            if state == "poweredOn":
                bleno.startAdvertisingWithEIRData(self._build_adv_data(), b"")
            else:
                bleno.stopAdvertising()
                # If we get a non-poweredOn state before advertising
                # starts, surface that as a startup failure.
                if not self._adv_started.is_set():
                    self._adv_error = RuntimeError(
                        f"BLE adapter not poweredOn (state={state!r})"
                    )
                    self._adv_started.set()

        def _on_advertising_start(error: Any) -> None:
            if error:
                self._adv_error = RuntimeError(f"BLE advertising error: {error!r}")
                self._adv_started.set()
                return
            log.info(
                "PyBlenoBleClient: advertising as %r on hci%d, primary svc 0x%s",
                self._device_name,
                self._hci_index,
                PRIMARY_SERVICE_UUID,
            )
            bleno.setServices(
                [
                    pybleno.BlenoPrimaryService(
                        {
                            "uuid": PRIMARY_SERVICE_UUID,
                            "characteristics": [notify_char],
                        }
                    )
                ]
            )
            self._adv_started.set()

        def _on_accept(client_address: str) -> None:
            log.info("PyBlenoBleClient: central connected (%s)", client_address)

        def _on_disconnect(client_address: str) -> None:
            log.info("PyBlenoBleClient: central disconnected (%s)", client_address)
            self._subscriber_event.clear()
            bleno.startAdvertisingWithEIRData(self._build_adv_data(), b"")

        bleno.on("stateChange", _on_state_change)
        bleno.on("advertisingStart", _on_advertising_start)
        bleno.on("accept", _on_accept)
        bleno.on("disconnect", _on_disconnect)

        self._bleno = bleno
        self._notify_char = notify_char

        bleno.start()
        # Bleno's advertisingStart fires within a couple seconds; bail
        # loudly if it doesn't, so callers don't sit on a dead radio.
        if not self._adv_started.wait(timeout=15.0):
            raise TimeoutError("BLE advertising did not start within 15 s")
        if self._adv_error is not None:
            raise self._adv_error

    def wait_for_subscriber(self, timeout_s: float | None = None) -> bool:
        return self._subscriber_event.wait(timeout=timeout_s)

    def send_dispense_notify(self) -> bool:
        if self._notify_char is None:
            log.warning("PyBlenoBleClient: send_dispense_notify before start_advertising")
            return False
        if not self._subscriber_event.is_set():
            log.info("PyBlenoBleClient: send_dispense_notify with no subscriber")
            return False
        ok = bool(self._notify_char.push(DISPENSE_PAYLOAD))
        if ok:
            log.info("PyBlenoBleClient: notify sent (%s)", DISPENSE_PAYLOAD.hex())
        return ok

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._bleno is None:
            return
        try:
            self._bleno.stopAdvertising()
            self._bleno.disconnect()
        except Exception:
            # Best-effort teardown — pybleno can raise from its libuv thread
            # during shutdown; swallowing here is preferable to leaking a
            # half-stopped peripheral.
            log.exception("PyBlenoBleClient.stop: error during teardown")
        self._subscriber_event.clear()

    # Context-manager sugar so the smoke script can wrap the whole run.

    def __enter__(self) -> PyBlenoBleClient:
        self.start_advertising()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.stop()
