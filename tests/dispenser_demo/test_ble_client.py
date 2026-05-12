"""Host unit tests for `gemma_tools.dispenser_demo.ble_client`.

Scope: wire-contract constants, `MockBleClient` semantics, and the
`PyBlenoBleClient` surface area we can exercise without pybleno
installed (argument validation, lazy-import boundary). No board, no
network, no pybleno.

The real pybleno-backed path is verified by running
`scripts/dispenser_demo/deploy/ble_test.py` on the SL2619 — that's the
Phase 2 board gate, not a pytest test.
"""

from __future__ import annotations

import threading

import pytest

from gemma_tools.dispenser_demo.ble_client import (
    ADV_SERVICE_UUID,
    DEVICE_NAME,
    DISPENSE_PAYLOAD,
    NOTIFY_CHAR_UUID,
    PRIMARY_SERVICE_UUID,
    BleClient,
    MockBleClient,
    PyBlenoBleClient,
)

# ── Wire-contract constants (plan §6.2) ────────────────────────────────────


# | constant              | expected      | description                            |
@pytest.mark.parametrize(
    ("constant", "expected", "desc"),
    [
        (DEVICE_NAME, "NousVoice", "adv device name matches plan §6.2"),
        (ADV_SERVICE_UUID, "00FB", "16-bit adv service UUID matches plan §6.2"),
        (PRIMARY_SERVICE_UUID, "FFB0", "primary service UUID matches plan §6.2"),
        (NOTIFY_CHAR_UUID, "FFB2", "notify characteristic UUID matches plan §6.2"),
    ],
)
def test_wire_contract_constants(constant: str, expected: str, desc: str) -> None:
    assert constant == expected, desc


def test_dispense_payload_is_4_bytes_in_contract_order() -> None:
    # The ESP32 firmware decodes this as [header_hi, header_lo, cmd, status].
    # Any drift here silently breaks the integration with no error.
    assert isinstance(DISPENSE_PAYLOAD, bytes), "payload must be bytes for pybleno"
    assert len(DISPENSE_PAYLOAD) == 4, "dispense payload is fixed at 4 bytes"
    assert bytes([0x5A, 0xA5, 0x01, 0x00]) == DISPENSE_PAYLOAD, (
        "dispense bytes must match plan §6.2: 5A A5 01 00"
    )


# ── MockBleClient ──────────────────────────────────────────────────────────


def test_mock_implements_bleclient_protocol() -> None:
    # @runtime_checkable Protocol — structural match by method shape, so this
    # is a real test that we kept all four methods.
    assert isinstance(MockBleClient(), BleClient), (
        "MockBleClient must satisfy the BleClient Protocol"
    )


def test_mock_starts_with_no_advertising_and_no_subscriber() -> None:
    c = MockBleClient()
    assert c.advertising is False, "fresh mock must not be advertising"
    assert c.sent_notifications == [], "fresh mock has no recorded notifications"


def test_mock_send_returns_false_when_no_subscriber() -> None:
    c = MockBleClient()
    c.start_advertising()
    assert c.send_dispense_notify() is False, (
        "send must return False (not raise) when no central is subscribed — "
        "plan §6.1 ble_not_connected branch"
    )
    assert c.sent_notifications == [], "no payload recorded when send is dropped"


def test_mock_send_emits_exact_payload_when_subscriber_attached() -> None:
    c = MockBleClient()
    c.start_advertising()
    c._simulate_subscribe()
    assert c.send_dispense_notify() is True, "send must succeed once a subscriber attached"
    assert c.sent_notifications == [b"\x5a\xa5\x01\x00"], (
        "MockBleClient must record the exact wire bytes from plan §6.2"
    )


def test_mock_send_multiple_records_each_call() -> None:
    c = MockBleClient()
    c.start_advertising()
    c._simulate_subscribe()
    c.send_dispense_notify()
    c.send_dispense_notify()
    c.send_dispense_notify()
    assert len(c.sent_notifications) == 3, "every successful send must be recorded"
    assert all(b == DISPENSE_PAYLOAD for b in c.sent_notifications), (
        "every recorded byte sequence must match the wire contract"
    )


def test_mock_unsubscribe_blocks_subsequent_sends() -> None:
    c = MockBleClient()
    c.start_advertising()
    c._simulate_subscribe()
    assert c.send_dispense_notify() is True, "first send (subscribed) must succeed"
    c._simulate_unsubscribe()
    assert c.send_dispense_notify() is False, (
        "after unsubscribe send must return False — the BLE central went away"
    )


def test_mock_wait_for_subscriber_returns_false_on_timeout() -> None:
    c = MockBleClient()
    # 0.0 is a non-blocking poll; no subscriber, so must report False.
    assert c.wait_for_subscriber(timeout_s=0.0) is False, (
        "wait_for_subscriber returns False on timeout, must not raise"
    )


def test_mock_wait_for_subscriber_returns_true_after_subscribe() -> None:
    c = MockBleClient()
    c._simulate_subscribe()
    assert c.wait_for_subscriber(timeout_s=0.0) is True, (
        "wait_for_subscriber returns True once the event is set"
    )


def test_mock_wait_for_subscriber_unblocks_from_background_thread() -> None:
    # Mirrors the production threading model: pybleno fires onSubscribe from
    # its libuv thread; the main thread is parked in wait_for_subscriber.
    # No sleep — Event.set() is the synchronization point.
    c = MockBleClient()
    ready = threading.Event()

    def _subscribe_when_ready() -> None:
        ready.wait(timeout=1.0)
        c._simulate_subscribe()

    t = threading.Thread(target=_subscribe_when_ready)
    t.start()
    ready.set()
    try:
        assert c.wait_for_subscriber(timeout_s=1.0) is True, (
            "wait_for_subscriber must unblock when another thread subscribes"
        )
    finally:
        t.join(timeout=1.0)


def test_mock_stop_clears_state_and_blocks_restart() -> None:
    c = MockBleClient()
    c.start_advertising()
    c._simulate_subscribe()
    c.stop()
    assert c.stopped is True, "stop must flag the client as stopped"
    assert c.advertising is False, "stop must clear the advertising flag"
    with pytest.raises(RuntimeError, match="already called"):
        c.start_advertising()


# ── PyBlenoBleClient — argument validation & lazy import ──────────────────


def test_pybleno_client_rejects_negative_hci_index() -> None:
    with pytest.raises(ValueError, match="hci_index must be >= 0"):
        PyBlenoBleClient(hci_index=-1)


# | hci_index | expected_env | description                                        |
@pytest.mark.parametrize(
    ("hci_index", "expected_env", "desc"),
    [
        (0, "0", "hci0 -> BLENO_HCI_DEVICE_ID=0 (default)"),
        (1, "1", "hci1 -> BLENO_HCI_DEVICE_ID=1 (Broadcom may land here)"),
        (2, "2", "hci2 still maps cleanly"),
    ],
)
def test_pybleno_client_constructor_stores_hci_index(
    hci_index: int, expected_env: str, desc: str
) -> None:
    c = PyBlenoBleClient(hci_index=hci_index)
    assert str(c._hci_index) == expected_env, desc


def test_pybleno_client_does_not_import_pybleno_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    # If pybleno were imported at __init__, this test would explode on
    # hosts that don't ship the package. The lazy-import guarantee is the
    # only reason this module can live in the regular import graph.
    import builtins

    real_import = builtins.__import__

    def _refuse_pybleno(name: str, *a: object, **kw: object) -> object:
        if name == "pybleno" or name.startswith("pybleno."):
            raise ImportError(f"refusing to import {name} during construction")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _refuse_pybleno)
    # Must succeed despite the import guard.
    PyBlenoBleClient(hci_index=0)


def test_pybleno_client_send_without_start_returns_false() -> None:
    # Defensive: if a caller skips start_advertising, send must degrade
    # rather than dereference a None notify characteristic.
    c = PyBlenoBleClient(hci_index=0)
    assert c.send_dispense_notify() is False, (
        "send_dispense_notify before start_advertising must return False, not crash"
    )


def test_pybleno_client_stop_before_start_is_noop() -> None:
    # Idempotent stop is a property of the Protocol contract.
    c = PyBlenoBleClient(hci_index=0)
    c.stop()
    c.stop()  # second call must not raise


def test_pybleno_client_build_adv_data_contains_name_and_uuid() -> None:
    # The EIR payload is opaque bytes once wrapped, but we can spot-check
    # that the device name and 16-bit UUID actually got encoded.
    c = PyBlenoBleClient(hci_index=0)
    adv = c._build_adv_data()
    assert isinstance(adv, bytes), "adv data must be bytes for pybleno"
    assert DEVICE_NAME.encode("utf-8") in adv, (
        "advertising payload must carry the NousVoice local name"
    )
    # 00FB little-endian = FB 00
    assert b"\xfb\x00" in adv, (
        "advertising payload must carry the 16-bit svc UUID in little-endian order"
    )
