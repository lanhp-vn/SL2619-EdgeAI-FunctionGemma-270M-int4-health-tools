# Task prompt — wire the real BLE notify into the dispenser demo

> Paste the block below into a fresh Claude Code session to start the BLE
> integration. It is the last open BLE item: replace the
> `[BLE→ESP32] 5A A5 01 00` **stdout mock** in `chat_board_dispense.py:dispatch`
> with a real `PyBlenoBleClient.send_dispense_notify()`, owned by the voice loop.
> Authored 2026-06-10 after the iter-001 voice demo passed end-to-end (mock BLE).

---

# Task: Replace the dispenser-demo BLE stdout mock with a real pybleno notify

Wire the **real** BLE GATT notify into the dispenser voice loop, replacing the
`[BLE→ESP32] 5A A5 01 00` stdout mock. The pybleno peripheral is **already
proven end-to-end** (2026-06-01: phone subscribed to `0xFFB2` and received
`5A A5 01 00`) via the standalone `ble_test.py` harness — `PyBlenoBleClient` in
`src/gemma_tools/dispenser_demo/ble_client.py` is the finished client. This task
is **lifecycle integration, not BLE bring-up**: give the voice loop ownership of
that client and fire `send_dispense_notify()` on each dispense turn.

**Success = one end-to-end dispense turn where a phone (nRF Connect) subscribed
to `0xFFB2` actually receives `5A A5 01 00`, AND the spoken Piper reply still
plays on the P10S — both in the same turn.**

## Pre-flight (mandatory, R1)
1. Run `/board_probe`. Confirm Astra `v2.4.0`, the board IP, P10S = ALSA card 1,
   and that `hci0` exists (`hciconfig -a` shows `Type: Primary  Bus: UART`).
2. Read, in order:
   - `docs/deployment/sl2619-ble-bringup.md` — §1 recipe, §3.2/§3.3 traps, §7
     remaining work. **This is the binding runbook.**
   - `src/gemma_tools/dispenser_demo/ble_client.py` — the `BleClient` Protocol +
     `PyBlenoBleClient` (finished) + `MockBleClient` (for host tests).
   - `scripts/functiongemma/deploy/chat_board_dispense.py` — the `dispatch`
     override holding the stdout mock (lines ~80–86).
   - `scripts/dispenser_demo/deploy/dispenser_voice.py` — `_load_chat_board`
     (~526) and the per-turn block (~829–854) that already tracks
     `chat_board._last_tool` / `_last_formatted`.
   - `scripts/dispenser_demo/deploy/ble_test.py` — the proven standalone harness;
     mirror its bind/reset assumptions.
   - `docs/plans/dispenser-demo/plan.md` §6.2 (wire contract) + §9.

## Environment facts (do not re-litigate)
- **The agent CANNOT run the BLE bring-up.** `systemctl stop bluetooth`,
  `systemctl restart brcm_bt_start.service`, `hciconfig hci0 up/down`, and
  `ssh … mkdir` are all **state-changing → denied per R3**. The agent **emits**
  them; the **user runs** them. The agent MAY `scp` to `/tmp` + `/mnt/sdcard`,
  and run `python3` / audio verbs over ssh (bounded-test exception, single-verb
  per ssh call).
- **Mandatory board bring-up before any pybleno run** (runbook §1b — emit for
  the user, every boot):
  ```sh
  systemctl stop bluetooth                 # pybleno needs exclusive HCI, not BlueZ
  systemctl restart brcm_bt_start.service  # re-power + re-patch chip; leaves hci0 DOWN
  hciconfig hci0 up                        # kernel HCI Reset (else 'Command Disallowed' — §3.3)
  hciconfig hci0 down                      # release for pybleno's HCI_CHANNEL_USER claim
  ```
  Skipping the `up`→`down` cycle → `Command Disallowed`. If `hci0` won't come up
  (`Connection timed out`), the patch was lost — only `systemctl restart
  brcm_bt_start.service` recovers it (§3.2), a bare `hciconfig up` cannot.
- **Staging is manual** (no `pip`/`fcntl` on board). The pybleno tree +
  `fcntl.py` shim must be on `PYTHONPATH`. Reuse the runbook §1a staging; the
  `mkdir` is emitted for the user, the `scp -r` the agent may run. The proven
  `PYTHONPATH` is `/tmp:/tmp/pylibs`. **For a reboot-durable demo, stage to
  `/mnt/sdcard/pylibs` instead** and add it to the voice loop's `PYTHONPATH`.
- **`PyBlenoBleClient` is done — do not rewrite it.** It exposes
  `start_advertising()` / `wait_for_subscriber()` / `send_dispense_notify()` /
  `stop()`, sets `BLENO_HCI_DEVICE_ID` before import, and gates on
  `threading.Event` (no poll-sleep). `send_dispense_notify()` returns `False`
  when no central is subscribed — use that for graceful degrade.
- **Wire contract (frozen, plan §6.2):** name `NousVoice`, adv-svc `0x00FB`,
  primary `0xFFB0`, notify char `0xFFB2`, payload `5A A5 01 00`. Do not change.

## Recommended design (confirm before building)
**Persistent advertise, per-turn notify, dependency-injected client.** Rationale:
pybleno claims `hci0` exclusively (DOWN + `HCI_CHANNEL_USER`) for its whole
lifetime; the voice loop needs `hci0` for nothing else, so a single peripheral
advertised once at startup has zero contention and keeps the phone's
subscription alive across turns. Per-turn bind/release would re-run the reset
cycle and drop the subscription every turn — reject it.

1. **`chat_board_dispense.py`** — add a module-level `_ble_client = None` and a
   `set_ble_client(client)` setter. In `dispatch`, when a dispense tool fires:
   call `_ble_client.send_dispense_notify()` if a client is set, **else** fall
   back to the existing `print("[BLE→ESP32] …")` mock (so the standalone REPL
   and host tests still work with no radio). Log the notify's `True/False`
   result. Keep the canned response unconditional in `format_response`.
2. **`dispenser_voice.py`** — construct `PyBlenoBleClient(hci_index=0)` at
   startup *after* the user has run the §1b reset cycle; call
   `start_advertising()` (blocks until `advertisingStart`), then
   `chat_board_dispense.set_ble_client(client)`. `stop()` in the shutdown
   `finally`. Add a `--no-ble` flag that skips all of this and leaves the stdout
   mock (so the loop still runs with no phone / no staging).
3. **Graceful degrade:** if `send_dispense_notify()` returns `False` (no
   subscriber), log `ble_not_connected` and **still** render the spoken reply —
   the demo must never hang waiting on a phone.

## Build + validate
- **Host first (TDD):** extend `tests/dispenser_demo/` (or wherever the
  `ble_client` tests live) to cover the injection seam with `MockBleClient` —
  assert `dispatch` calls `send_dispense_notify()` when a client is set and
  falls back to stdout when not. `uv run pytest` green before touching the board.
- **Board, standalone:** emit the §1b bring-up for the user, stage per §1a, then
  (agent may run) `PYTHONPATH=/tmp:/tmp/pylibs python3 /tmp/ble_test.py --hci
  hci0 --skip-patch-check --timeout-s 120` — confirm the phone still gets
  `5A A5 01 00`. This proves staging before the loop wiring is in play.
- **Board, integrated:** launch `dispenser_voice.py -v` (backgrounded; mirror
  the proven run procedure — wake "Hey Jarvis" → command → watch the log). With
  the phone subscribed to `0xFFB2`, a dispense turn must show the real notify
  (not the stdout mock) AND the phone must receive `5A A5 01 00` AND the Piper
  reply must play. Report first-turn wall (BLE notify adds negligible time vs the
  ~10.7 s pipeline).

## Out of scope (explicit follow-ons — do NOT bundle)
- **Close runbook §3.5** — on a clean `reboot`, check `hciconfig hci0` *without*
  restarting anything; if DOWN, propose a boot-path fix so the demo needs no
  manual bring-up. Separate task.
- **Persist staging to `/mnt/sdcard`** — once the `/tmp` path is proven, migrate
  staging + `PYTHONPATH` so it survives reboot. Fold into this task only if time
  allows; otherwise separate.
- **ESP32 firmware** (plan §6.3) — the central that consumes the notify. Not this
  task; the phone (nRF Connect) is the validation stand-in.

## Done criteria
- Host tests green (injection + fallback covered).
- One integrated dispense turn: phone receives `5A A5 01 00` on `0xFFB2` **and**
  the spoken reply plays — verified, not assumed.
- `--no-ble` still runs the loop with the stdout mock.
- Update `docs/deployment/sl2619-ble-bringup.md` §7 item 1 and the CLAUDE.md
  "Only remaining BLE item" line to reflect the notify is now wired
  (via `/doc_update`).

Full state: `docs/deployment/sl2619-ble-bringup.md`,
`docs/plans/dispenser-demo/{plan.md,decisions-log.md}`.
