"""Host unit tests for the Phase 2 BLE smoke script.

Mirrors `tests/dispenser_demo/test_crispasr_smoke_scripts.py`: load the
script as a module via `importlib.util.spec_from_file_location` (it
lives outside the importable package tree) and assert on its argparse
surface + the small pure helpers. No board, no pybleno required.

The real BLE test against the M.2 Broadcom adapter is the script
running on the SL2619 — that's the Phase 2 gate, not a pytest test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BLE_SCRIPT = REPO_ROOT / "scripts" / "dispenser_demo" / "deploy" / "ble_test.py"


def _load_ble_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ble_test", BLE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BLE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ble_mod() -> ModuleType:
    return _load_ble_module()


def test_ble_script_exists_and_is_executable() -> None:
    assert BLE_SCRIPT.is_file(), "BLE smoke script missing"
    assert BLE_SCRIPT.stat().st_mode & 0o111, "BLE smoke script must be executable"


# ── _resolve_hci_index ─────────────────────────────────────────────────────


# | hci_name | hci_index | expected | description                                     |
@pytest.mark.parametrize(
    ("hci_name", "hci_index", "expected", "desc"),
    [
        (None, None, 0, "no flags -> default to hci0"),
        ("hci0", None, 0, "--hci hci0 -> index 0"),
        ("hci1", None, 1, "--hci hci1 -> index 1 (Broadcom can land here per plan §2 note)"),
        (None, 2, 2, "--hci-index 2 wins when set explicitly"),
        ("hci0", None, 0, "--hci form alone is fine without --hci-index"),
    ],
)
def test_resolve_hci_index_happy(
    ble_mod: ModuleType,
    hci_name: str | None,
    hci_index: int | None,
    expected: int,
    desc: str,
) -> None:
    assert ble_mod._resolve_hci_index(hci_name, hci_index) == expected, desc


# | hci_name      | hci_index | description                                       |
@pytest.mark.parametrize(
    ("hci_name", "hci_index", "desc"),
    [
        ("hci0a", None, "malformed name with trailing chars must be rejected"),
        ("BT0", None, "non-hci prefix must be rejected"),
        ("", None, "empty string must be rejected"),
        (None, -1, "negative index must be rejected"),
    ],
)
def test_resolve_hci_index_rejects_bad_inputs(
    ble_mod: ModuleType,
    hci_name: str | None,
    hci_index: int | None,
    desc: str,
) -> None:
    with pytest.raises(ValueError, match=r"must be >= 0|must be of the form"):
        ble_mod._resolve_hci_index(hci_name, hci_index)


# ── parse_args ─────────────────────────────────────────────────────────────


def test_parse_args_defaults(ble_mod: ModuleType) -> None:
    args = ble_mod.parse_args([])
    assert args.hci is None, "no --hci by default; resolution defaults to hci0"
    assert args.hci_index is None, "no --hci-index by default"
    assert args.timeout_s == 120.0, "default timeout is 120 s — matches script docstring"
    assert args.send_once is False, "default behavior is interactive loop, not one-shot"
    assert args.skip_patch_check is False, (
        "by default we sanity-check the BluetoothHCI.py patches"
    )
    assert args.verbose is False, "DEBUG logs are opt-in"


# | argv                                  | want_field      | want_value | description                            |
@pytest.mark.parametrize(
    ("argv", "want_field", "want_value", "desc"),
    [
        (["--hci", "hci1"], "hci", "hci1", "--hci hci1 parsed verbatim"),
        (["--hci-index", "1"], "hci_index", 1, "--hci-index 1 parsed as int"),
        (["--timeout-s", "30"], "timeout_s", 30.0, "--timeout-s 30 parsed as float"),
        (["--timeout-s", "0"], "timeout_s", 0.0, "--timeout-s 0 sentinel for 'wait forever'"),
        (["--send-once"], "send_once", True, "--send-once sets the one-shot flag"),
        (["--skip-patch-check"], "skip_patch_check", True, "patch check can be muted"),
        (["-v"], "verbose", True, "short -v turns on DEBUG"),
        (["--verbose"], "verbose", True, "long --verbose also works"),
    ],
)
def test_parse_args_flag_table(
    ble_mod: ModuleType,
    argv: list[str],
    want_field: str,
    want_value: object,
    desc: str,
) -> None:
    args = ble_mod.parse_args(argv)
    assert getattr(args, want_field) == want_value, desc


def test_parse_args_rejects_both_hci_forms(
    ble_mod: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    # --hci and --hci-index are mutually exclusive — argparse exits 2.
    with pytest.raises(SystemExit) as excinfo:
        ble_mod.parse_args(["--hci", "hci0", "--hci-index", "0"])
    assert excinfo.value.code == 2, "argparse exits with code 2 on conflicting flags"
    err = capsys.readouterr().err
    assert "not allowed with" in err, "argparse must explain the mutex"
