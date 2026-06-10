#!/usr/bin/env python3
"""Phase 2 BLE smoke runner for the dispenser demo (plan §9 step 2.4).

Stands up the SL2619 as a BLE peripheral under the plan §6.2 wire contract:

    name        NousVoice
    adv svc     0x00FB
    primary     0xFFB0
    notify char 0xFFB2
    payload     5A A5 01 00

Waits for a central (the ESP32 dispenser, or `nRF Connect` as stand-in)
to subscribe to the notify characteristic, then pushes the dispense
payload either once (`--send-once`) or on every ENTER keypress
(default — handy for iterating with nRF Connect).

This script is *board* code (Synaptics SL2619, BlueZ + pybleno against
the M.2 Broadcom adapter). It is meant to be invoked over SSH; the host
copy is mainly to keep ruff / mypy / pytest honest. Production board
deploy: `scp scripts/dispenser_demo/deploy/ble_test.py
nouslogic-sl2619:/tmp/`, then `ssh -t nouslogic-sl2619 'python3
/tmp/ble_test.py --hci hci<N>'`.

Preconditions (the script checks and reports them — does not fix them):

1. The Python 3.12 / kernel 6.x pybleno patches from
   docs/references/old-dispenser-demo/pybleno-setup-guide.md must be
   applied to `BluetoothHCI.py` (sentinel: `AF_BLUETOOTH = 31`).
2. The selected `hciX` must be brought down before launch so pybleno's
   HCI_CHANNEL_USER socket can claim the device exclusively:
   `hciconfig hci<N> down`. The script prints a hint if it suspects the
   adapter is still up; it does NOT execute this for you (out of scope
   per the BLE bring-up task; user runs it manually).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

# Make src/ importable when this script is run directly from a repo
# checkout. On the board the file lives outside the package tree (e.g.
# /tmp/ble_test.py), where parents[3] doesn't exist — guard the lookup so we
# fall through to the PYTHONPATH-staged gemma_tools package instead of crashing.
_parents = Path(__file__).resolve().parents
if len(_parents) > 3:
    _SRC = _parents[3] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

try:
    from gemma_tools.dispenser_demo.ble_client import (
        ADV_SERVICE_UUID,
        DEVICE_NAME,
        DISPENSE_PAYLOAD,
        NOTIFY_CHAR_UUID,
        PRIMARY_SERVICE_UUID,
        PyBlenoBleClient,
    )
except ImportError as exc:  # pragma: no cover — board may stage just this file
    raise SystemExit(
        f"could not import gemma_tools.dispenser_demo.ble_client: {exc}\n"
        f"deploy both ble_client.py and ble_test.py, or run from the repo root"
    ) from exc


log = logging.getLogger("ble_test")


_HCI_RE = re.compile(r"^hci(\d+)$")


def _resolve_hci_index(hci_name: str | None, hci_index: int | None) -> int:
    """Return the numeric hci device index from either form.

    Precedence: `--hci-index` wins if explicitly set; otherwise parse
    `--hci hciN`. Raises `ValueError` on a malformed name. Defaults to 0.
    """
    if hci_index is not None:
        if hci_index < 0:
            raise ValueError(f"--hci-index must be >= 0, got {hci_index}")
        return hci_index
    if hci_name is None:
        return 0
    match = _HCI_RE.match(hci_name)
    if match is None:
        raise ValueError(
            f"--hci must be of the form 'hciN' (e.g. hci0, hci1); got {hci_name!r}"
        )
    return int(match.group(1))


_PATCH_SENTINELS: tuple[tuple[str, str, bool], ...] = (
    # (sentinel, setup-guide step, must_be_present)
    ("AF_BLUETOOTH = 31", "step 3 (socket constants)", True),
    ("'rU'", "step 4 (rU mode removed in Py 3.11+)", False),
    ("HCI_CHANNEL_USER", "step 6 (use HCI_CHANNEL_USER)", True),
    ("hciconfig", "step 7 (hciconfig fallback for HCIGETDEVINFO)", True),
)


def _check_pybleno_patches() -> tuple[bool, str]:
    """Verify the four BluetoothHCI.py setup-guide patches are in place.

    Returns `(ok, message)`. `ok=False` does not abort — the script just
    logs the message and lets pybleno itself fail loudly. This sentinel
    just turns the resulting `setsockopt: Invalid argument` /
    `BTPROTO_HCI not found` / `mode 'rU'` errors into something a human
    can act on without having to repro the failure first.

    Covers 4 of the 5 setup-guide patches. Step 5 (the `set_filter` skip)
    is structural — there isn't a single grep-able sentinel that proves
    it landed without false positives, so we rely on the other four as a
    proxy for "the patch set was applied at all".
    """
    try:
        import pybleno  # type: ignore[import-not-found]
    except ImportError:
        return False, "pybleno is not installed (run: pip3 install pybleno --break-system-packages)"
    pkg_dir = Path(pybleno.__file__).resolve().parent
    candidates = list(pkg_dir.rglob("BluetoothHCI.py"))
    if not candidates:
        return False, f"BluetoothHCI.py not found under {pkg_dir}"
    path = candidates[0]
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for sentinel, label, must_be_present in _PATCH_SENTINELS:
        present = sentinel in text
        if present != must_be_present:
            verb = "missing" if must_be_present else "still has"
            missing.append(f"{verb} {sentinel!r} ({label})")
    if missing:
        return False, f"{path}: {'; '.join(missing)}. pybleno will crash on import."
    return True, str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    hci_group = p.add_mutually_exclusive_group()
    hci_group.add_argument(
        "--hci",
        type=str,
        default=None,
        metavar="hciN",
        help="BLE adapter name from `hciconfig -a` (e.g. hci0). "
        "Default: hci0. Mutually exclusive with --hci-index.",
    )
    hci_group.add_argument(
        "--hci-index",
        type=int,
        default=None,
        help="Numeric BLE adapter index (e.g. 0 for hci0). "
        "Mutually exclusive with --hci.",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help=(
            f"Seconds to wait for a central to subscribe to {NOTIFY_CHAR_UUID}. "
            "Default: 120. Pass 0 to wait indefinitely."
        ),
    )
    p.add_argument(
        "--send-once",
        action="store_true",
        help="Send exactly one dispense notification after the first subscriber "
        "appears, then exit. Default: send on every ENTER until SIGINT.",
    )
    p.add_argument(
        "--skip-patch-check",
        action="store_true",
        help="Don't verify the pybleno BluetoothHCI.py setup-guide patches. "
        "Use only when you're certain the patches are in.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logs.",
    )
    return p.parse_args(argv)


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        log.info("ble_test: signal %d received — stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _emit_one(client: PyBlenoBleClient) -> None:
    ok = client.send_dispense_notify()
    if ok:
        log.info("ble_test: dispense notify sent (%s)", DISPENSE_PAYLOAD.hex())
    else:
        log.warning("ble_test: no subscriber currently connected — notify dropped")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        hci_index = _resolve_hci_index(args.hci, args.hci_index)
    except ValueError as exc:
        log.error("ble_test: %s", exc)
        return 2

    log.info("ble_test: selected adapter = hci%d (env BLENO_HCI_DEVICE_ID will be set)", hci_index)
    log.info(
        "ble_test: precondition — run `hciconfig hci%d down` before this if pybleno fails to bind",
        hci_index,
    )

    if not args.skip_patch_check:
        ok, msg = _check_pybleno_patches()
        if ok:
            log.info("ble_test: pybleno patches OK (%s)", msg)
        else:
            log.warning("ble_test: pybleno precondition check: %s", msg)
            log.warning(
                "ble_test: see docs/references/old-dispenser-demo/pybleno-setup-guide.md "
                "for the 5 setup patches. Pass --skip-patch-check to silence."
            )

    log.info(
        "ble_test: wire contract — adv name=%s adv-svc=0x%s primary=0x%s notify=0x%s payload=%s",
        DEVICE_NAME,
        ADV_SERVICE_UUID,
        PRIMARY_SERVICE_UUID,
        NOTIFY_CHAR_UUID,
        DISPENSE_PAYLOAD.hex(),
    )

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    timeout_s: float | None = args.timeout_s if args.timeout_s > 0 else None

    client = PyBlenoBleClient(hci_index=hci_index)
    try:
        client.start_advertising()
        log.info("ble_test: advertising started, waiting up to %s s for a subscriber", timeout_s)

        if not client.wait_for_subscriber(timeout_s=timeout_s):
            log.error("ble_test: timed out after %.1f s — no central subscribed", args.timeout_s)
            return 1
        log.info("ble_test: subscriber attached to 0x%s", NOTIFY_CHAR_UUID)

        _emit_one(client)
        if args.send_once:
            log.info("ble_test: --send-once: exiting after one notification")
            return 0

        log.info("ble_test: press ENTER to send another notification, Ctrl+C to quit")
        # Loop until SIGINT/SIGTERM. We read stdin line-by-line so this
        # works under SSH with a PTY. If stdin is closed (systemd-managed
        # service, etc.), just wait on the stop_event.
        if os.isatty(sys.stdin.fileno() if sys.stdin else -1):
            while not stop_event.is_set():
                try:
                    line = sys.stdin.readline()
                except KeyboardInterrupt:
                    break
                if line == "":  # EOF
                    break
                if line.strip().lower() == "quit":
                    break
                _emit_one(client)
        else:
            log.info("ble_test: stdin is not a TTY — waiting on signal only")
            stop_event.wait()

        return 0

    finally:
        client.stop()
        log.info("ble_test: stopped")


if __name__ == "__main__":  # pragma: no cover — entry point
    sys.exit(main())
