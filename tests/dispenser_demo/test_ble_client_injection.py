"""Injection-seam tests for the dispenser-demo BLE notify.

Verifies `scripts/functiongemma/deploy/chat_board_dispense.py` routes a
dispense tool call through an injected `BleClient` once `set_ble_client()`
has wired one, and falls back to the `[BLE→ESP32]` stdout mock otherwise.
No board, no pybleno — `MockBleClient` stands in for the radio.

The real pybleno-backed notify is proven by running
`scripts/dispenser_demo/deploy/ble_test.py` on the SL2619 (the Phase 2
board gate); this file only covers the host-side injection wiring.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from gemma_tools.dispenser_demo.ble_client import DISPENSE_PAYLOAD, MockBleClient

_REPO = Path(__file__).resolve().parents[2]
_DISPENSE = _REPO / "scripts" / "functiongemma" / "deploy" / "chat_board_dispense.py"

_MOCK_TOKEN = "[BLE→ESP32]"  # exclusively the no-radio stdout mock


@pytest.fixture
def dispense_mod() -> Iterator[ModuleType]:
    """Load the deploy script fresh; reset its module global after each test.

    `chat_board_dispense` holds the injected client in a module global that
    lives in `sys.modules` — without the teardown reset it would leak a
    subscribed `MockBleClient` into the next test.
    """
    spec = importlib.util.spec_from_file_location("fg_chat_board_dispense", _DISPENSE)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        mod.set_ble_client(None)


def _fire_dispense(mod: ModuleType) -> object:
    # `get_medications_at_time` is one of the hijacked dispense tools; an empty
    # table is fine because the dispense branch never reads it.
    return mod.dispatch("get_medications_at_time", {}, {})


def test_dispatch_uses_stdout_mock_when_no_client(
    dispense_mod: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _fire_dispense(dispense_mod)
    out = capsys.readouterr().out
    assert _MOCK_TOKEN in out, "no injected radio must keep the stdout mock"
    assert result == {"status": "dispensed"}


def test_dispatch_fires_notify_when_client_subscribed(
    dispense_mod: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockBleClient()
    client.start_advertising()
    client._simulate_subscribe()
    dispense_mod.set_ble_client(client)

    result = _fire_dispense(dispense_mod)
    out = capsys.readouterr().out

    assert client.sent_notifications == [DISPENSE_PAYLOAD], (
        "subscribed client must receive the exact wire payload"
    )
    assert _MOCK_TOKEN not in out, "real client path must not emit the stdout mock"
    assert result == {"status": "dispensed"}


def test_dispatch_degrades_when_client_not_subscribed(
    dispense_mod: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockBleClient()
    client.start_advertising()  # advertising, but no central subscribed
    dispense_mod.set_ble_client(client)

    result = _fire_dispense(dispense_mod)
    out = capsys.readouterr().out

    assert client.sent_notifications == [], "no notify recorded with no subscriber"
    assert _MOCK_TOKEN not in out, "client set → never the stdout mock"
    assert result == {"status": "dispensed"}, (
        "dispense must still succeed so the spoken reply plays (ble_not_connected degrade)"
    )


def test_dispatch_degrades_when_notify_raises(
    dispense_mod: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # A mid-notify disconnect raises from pybleno's HCI write. Because
    # chat_board._run_turn wraps dispatch in `except KeyError` only, an escape
    # here would crash the whole voice loop — so the dispatch seam must catch it
    # and degrade. This pins that contract.
    client = MockBleClient()
    client.start_advertising()
    client._simulate_subscribe()
    client.raise_on_notify = True
    dispense_mod.set_ble_client(client)

    result = _fire_dispense(dispense_mod)
    out = capsys.readouterr().out

    assert client.sent_notifications == [], "a raising notify records nothing"
    assert _MOCK_TOKEN not in out, "client set → never the stdout mock"
    assert "[BLE]" in out, "degrade path still logs a [BLE] line for the operator"
    assert result == {"status": "dispensed"}, (
        "a radio exception must degrade, not crash — dispense still succeeds"
    )


def test_non_dispense_tool_never_touches_the_radio(dispense_mod: ModuleType) -> None:
    client = MockBleClient()
    client.start_advertising()
    client._simulate_subscribe()
    dispense_mod.set_ble_client(client)

    # A non-dispense tool routes to the inherited dispatch and must not notify.
    result = dispense_mod.dispatch("get_vitals", {}, {"vitals": {"hr": 72}})

    assert result == {"hr": 72}, "non-dispense tool result must pass through unchanged"
    assert client.sent_notifications == [], "non-dispense tool must not fire a BLE notify"
