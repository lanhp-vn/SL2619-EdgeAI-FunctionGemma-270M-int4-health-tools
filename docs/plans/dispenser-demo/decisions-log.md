# dispenser-demo decisions log

Append-only record of binding decisions for the dispenser-demo plan. Each
entry pins one resolved question. Update existing entries only to add follow-up
references; do not rewrite history.

The plan itself lives at [`plan.md`](plan.md). Phase-specific working notes
(e.g. [`crispasr-spike-notes.md`](crispasr-spike-notes.md),
[`bt-bringup-investigation-2026-05-12.md`](bt-bringup-investigation-2026-05-12.md))
are the authoritative record of the underlying analysis; this file is the index.

---

## 2026-05-12 (PM) — BT bring-up second-pass audit; disposition unchanged, hypotheses narrowed

Re-investigation of the M.2 BT failure mode under explicit user authorization
to run mutating SSH commands. The 2026-05-11 (late, addendum) entry below
established that `brcm_patchram_plus` loops `HCI_Reset` with no chip
response. This pass goes deeper.

**Net change: no new fix; hypothesis space narrowed from ~12 to 3.**

### What this audit added

- **UART1 pinmux verified correct.** Direct devmem read of pinctrl@`0xe5025b00`
  shows the bootloader has already set SM_GPIO7/8/14/15 to function 6
  (`uart1`). The empty `pinctrl-0` on `uart@e5031000` in DT is a red
  herring — kernel-side pinctrl is not applying mux; bootloader is. So
  the earlier hypothesis "the SL2619 DTS is missing a UART1 pinctrl
  reference" is wrong.
- **32.768 kHz refclk path effectively ruled out.** WiFi works → combo
  chip has the slow clock from *somewhere*. Forcing `SM_GPIO30 → sm_clkout`
  (function 7) on top of the existing config did nothing for BT.
- **FXL6408 register map corrected.** Mainline `gpio-fxl6408.c` uses
  `0x03=Dir, 0x05=Output, 0x07=High-Z, 0x0F=InputStatus` — distinct from
  the addresses an external chat tool (NotebookLM) had hallucinated.
  `bt_power`'s `out lo` in `/sys/kernel/debug/gpio` is a chip artifact
  (FXL6408 Input Status reads 0 for output pins), not a "chip can't
  drive high" failure mode.
- **`/dev/ttyS1` listen returns zero bytes** at 115200, 921600, and
  3 Mbps after rfkill cycle — confirming the chip is silent on UART
  regardless of baud, not just at patchram's chosen rate.
- **FXL pin 2 (`reset` ACTIVE_LOW, no documented consumer) gates network
  connectivity.** Driving it low — which NotebookLM had speculated was
  WL_REG_ON — drops `wlan0` and SSH within a second. Do not toggle it
  in any future experiment. Two board hangs this session traced to
  collateral effects of that pin or downstream registers; in both
  cases a clean power-cycle recovered.

### Three hypotheses survive — all physical-layer

| # | Hypothesis | Cannot test from SSH because |
|---|---|---|
| H13 | BT_REG_ON wired to FXL pin 4 (currently `output, no consumer, low`), not pin 5 | Toggling pin 4 hung the board mid-test; needs a self-reverting test wrapper, but better: schematic confirmation from Synaptics |
| H14 | M.2 module's BT-side hardware fault while WiFi works | Needs a known-good M.2 swap |
| H15 | SL2619-RDK PCB trace from SoC SM_UART1 TX (SM_GPIO8) to M.2 socket is broken/unpopulated | Needs an oscilloscope on the M.2 pin |

### Decision

**No further SSH-only investigation will productively move this.** Per
[`bt-bringup-investigation-2026-05-12.md`](bt-bringup-investigation-2026-05-12.md)
§ "Practical disposition", the v1 demo should take path 2 (run BLE
peripheral from a different Linux host on the LAN; wire contract
unchanged). The host-side BLE work is already hardware-agnostic and
tested. Board BT path stays gated on Synaptics bug 37861/37374.

### What NOT to retry next session

Different `.hcd` files, alternate baud rates, `hciattach` permutations,
`rfkill` cycle sweeps, `pinmux-select` syntaxes, `devmem` writes to
pinmux/FXL registers. All covered. The next legitimate experiment is
gated on one of: a Synaptics support response, a different M.2 card
swap, or a scope on the SM_UART1 TX pad.

### Evidence file

Full hypothesis matrix, register dumps, loopback measurements, and the
draft Synaptics-ticket payload are in
[`bt-bringup-investigation-2026-05-12.md`](bt-bringup-investigation-2026-05-12.md).

---

## 2026-05-12 (Phase 3 Layer D closed, afternoon) — Piper TTS landed; humanizer + refusal-as-TTS + command-ack chime + trace/-v split shipped

Four user-driven fixes closed Layer D and tightened the v1 voice loop end-to-end. With this entry, **BLE bring-up is the sole remaining v1 demo blocker** (board BT still gated on Synaptics bug 37861/37374 per the 2026-05-11 late entry; the `[BLE→ESP32] 5A A5 01 00` line in `chat_board_dispense.py` remains stdout-mock until pybleno binds a real radio — see the 2026-05-11 (late) ground-truth audit below).

### Decision 1 — Dynamic Piper TTS, with humanizer helpers for digit-bearing fields

- **Decision.** Layer D ships **Piper neural TTS** rendered per-turn from `chat_board.format_response`'s output, captured via the `_capture_format` side-effect wrapper in `dispenser_voice.py`. Canned per-tool WAVs at `<tts_dir>/<tool>.wav` remain as a fallback (`--no-dynamic-tts` or Piper load/render failure). Default voice: `en_US-lessac-medium` (host-rendered, ~30–60 MB ONNX, gitignored).
- **Decision.** All formatters in `scripts/functiongemma/deploy/chat_board.py` and the host `scripts/functiongemma/chat.py` route digit-bearing fields through new humanizer helpers — `_humanize_date`, `_humanize_time`, `_humanize_schedule`, `_humanize_measured_suffix` — before emitting the assistant line. Examples: `"2026-05-20"` → `"May 20"`; `"07:30"` → `"7:30 in the morning"`; `"23:30"` → `"11:30 at night"`; `"08:00, 20:00"` → `"8 in the morning and 8 in the evening"`. Vitals also spell units (`"82 beats per minute"`, `"142 over 88"`, `"97 percent"`, `"36.6 degrees Celsius"`).
- **Rationale.** Piper renders raw ISO strings digit-by-digit (`"two zero two six dash zero five dash twenty"`), which breaks cadence and listener comprehension. Cheap host-side normalization in the formatter sidesteps the problem without re-training. This is **not** the iter-002 `*_words`-companion contract — that remains the iter-002 target shape and stays unimplemented for v1.
- **Evidence.** `scripts/functiongemma/deploy/chat_board.py:168-233` (helpers) + `chat_board.py:362,408,428,457,468` (formatter call sites); identical mirror in `scripts/functiongemma/chat.py:193-243`. `tests/functiongemma/test_chat_formatters.py` parametrizes 35 new cases across both module copies via `@pytest.fixture(params=["chat", "chat_board"])` — **adding a new helper or formatter must update both module copies or the parity tests fail.** 729 tests green.

### Decision 2 — Out-of-scope refusal as TTS, routed through `format_response`

- **Decision.** Introduce sentinel `OUT_OF_SCOPE_TOOL = "_out_of_scope"` and constant `OUT_OF_SCOPE_REFUSAL = "I can only help with your medications, vitals, allergies, appointments, and emergency contact."` `_run_turn` (in both `chat_board.py` and `chat.py`) calls `format_response(user_text, OUT_OF_SCOPE_TOOL, {}, None)` on both the no-parse path and the dispatch-KeyError path so the assistant line reaches the same emission point as a normal answer.
- **Rationale.** Plan §6 + §7 wanted the iter-002 `refuse_out_of_scope(reason)` tool to own this surface. v1 runs on iter-001 which has no such tool, so refusals previously produced silent skip — both wrong UX and invisible to `dispenser_voice.py`'s `_capture_format` wrapper (no string captured → no Piper render → silent box). Routing through `format_response` keeps the TTS layer agnostic and reuses the existing capture wire. Per plan §6 ("refuse everything else") the user-facing line is the same for both refusal classes; the diagnostic `reason` enum from iter-002's design is dropped for v1 (the runtime layer can't infer it post-hoc anyway).
- **Evidence.** `chat_board.py:486-518,734-758` and `chat.py:411-441,571-589`. Plan §6 / §7 retain `refuse_out_of_scope(reason)` as the iter-002 target shape.

### Decision 3 — Post-STT "command received" chime before the LLM turn

- **Decision.** Play a short 160 ms two-blip 660 Hz WAV (`command_ack.wav`, distinct from the 170 ms rising `wake_ack.wav` already played on WAKE) after STT prints the transcript and before the LLM turn starts. New flags `--command-ack-wav` (default `/mnt/sdcard/dispenser_demo/command_ack.wav`) and `--no-command-ack`.
- **Rationale.** The LLM turn is ~6.5 s wall on first turn and ~6 s/turn warm. Without an audible "heard you, thinking" cue the user reflexively re-speaks during that gap (we're half-duplex per plan §8.3 E2 — mic is paused during playback, but no playback was happening, so the box looked dead). 160 ms is short enough that the arecord pipe survives without explicit half-duplex pause; the distinct two-blip pattern is differentiable from the rising-tone wake-ack at typical speaker output.
- **Evidence.** `dispenser_voice.py:197-204` (constant), `:815-823` (playback call site), `:976-983` (CLI flags). WAV staged at `/tmp/dispenser_demo_board/command_ack.wav` (gitignored).

### Decision 4 — Split `-v` (pipeline transitions) from `--trace` (per-frame)

- **Decision.** Introduce two loggers in `dispenser_voice.py`: `dispenser_voice` (INFO/DEBUG gated on `-v`) and `dispenser_voice.trace` (gated on the new `--trace` flag, default WARNING/silent). Per-frame `wake score %.3f` (80 ms cadence) and `vad %.3f speech_seen=...` (30 ms cadence) move to the trace logger. Pipeline transitions (`listening`, `[WAKE]`, `end of utterance`, `STT`, `LLM turn wall`, `TTS`) stay on the main logger.
- **Rationale.** Pre-fix, a few seconds of LISTENING under `-v` produced 50+ frame-level lines per second, drowning the pipeline-transition lines that matter for debugging. Two-logger split keeps the common case (`-v` for "what stage are we in?") readable while preserving access to frame-level data when needed (`--trace`).
- **Evidence.** `dispenser_voice.py:208-225` (logger setup), `:605` (wake-score trace), `:664` (VAD trace), `:1011-1016` (CLI flags). No call-site count change in the main logger.

### Binding (cumulative, supersedes the Layer C "Known carry-overs" list)

- **Long-running entry point unchanged**: `scripts/dispenser_demo/deploy/dispenser_voice.py`. Code size grew from ~676 to ~1036 lines (Piper render + render-cache plumbing + chime playback + humanizer-aware capture wrapper).
- **TTS rendering tool**: `scripts/dispenser_demo/voice/build_tts_canned.py` (host-side; bakes the fallback per-tool WAVs into `<tts_dir>/` using Piper). Run once per voice change.
- **On-board file layout adds**: `/mnt/sdcard/dispenser_demo/{wake_ack.wav, command_ack.wav, piper-voices/, tts/}`. Layer-B/C stages under `/mnt/sdcard/python-deps/site/` unchanged. WAVs are not committed (gitignored under `/tmp/dispenser_demo_board/`).
- **Test parity rule**: `tests/functiongemma/test_chat_formatters.py` parametrizes across `chat` and `chat_board`. Adding a helper or formatter to one module without the other is now a CI failure, not a runtime-only bug.
- **Layer C.1 arecord-overrun**: closed inside Layer D via `ArecordMic.drain(max_seconds=6)` — non-blocking pipe drain called after each LLM turn (and implicitly after TTS playback). See the Layer C entry below — the carry-over note there now reads "closed". No phantom-wake regression observed in this session's runs.

### What this leaves open

- **BLE bring-up is the sole remaining v1 demo blocker.** The board BT path remains gated on Synaptics bug 37861/37374 (see the 2026-05-11 (late) ground-truth audit). `chat_board_dispense.py:dispatch` will swap the stdout `[BLE→ESP32]` print for a `pybleno` notify call once the radio is reachable from any peripheral host.

### When to reconsider

- If Piper render wall exceeds ~500 ms per turn on board (current ballpark on lessac-medium), drop to `en_US-amy-low` or revert to canned WAVs by toolname.
- If real BLE bring-up reveals timing skew between the canned chime, Piper playback, and the BLE notify (e.g. dispense fires before the canned line finishes speaking), serialize at the dispatch level instead of relying on aplay returning.

---

## 2026-05-12 (Phase 3 Layer C closed) — Full pipeline (wake → STT → FunctionGemma → dispense override) works end-to-end on board

First-run, single-turn dispense intent on SL2619:

| Stage | Wall | Note |
| --- | --- | --- |
| oww + VAD load | 0.90 s | unchanged vs Layer B |
| Mixer pre-set (Mic 70%, PCM 50%) | <50 ms | per plan §9.3 step 3.1 — recommend keeping every boot |
| FG prompt-cache resolve | ~0 s | existing 20 MiB `/tmp/fg_pc_finetuned_functiongemma_q4_0.gguf.bin` reused (warm from Layer B) |
| Wake fire | ~1.55 s from speech | score 0.979 → 0.822, patience=2 |
| Pre-speech silence survived | 1.24 s | speech-relative cap (Layer C improvement over Layer B) kept LISTENING open until first `speech_seen=True` |
| Speech endpoint | 3.20 s captured | `silence_run >= 13` |
| CrispASR decode | 1.14 s (3.0× rt) | transcript: `"give me my medication"` |
| FunctionGemma turn | 6.48 s | 11 prompt @ 28.6 tok/s + 14 decode @ 10.4 tok/s — single-turn envelope identical to standalone `chat_board.py` |
| **End-to-end wake → done** | **~10.7 s** | one utterance |
| Loop reset (`=== turn 2 ===`) | <50 ms | `wake_model.reset()` + `vad.reset_states()` + buffer clears |
| Ctrl-C exit | clean | `[exit]` printed, no hung subprocess |

Dispense override confirmed wired: tool call was
`get_medication_by_name{}` (model emitted no args — the hijack treats
this as dispense intent unconditionally), dispatch printed
`[BLE→ESP32] 5A A5 01 00`, formatter returned the verbatim §6.1 canned
response. Same control flow that the standalone
`chat_board_dispense.py --probe` smoke validated.

### Binding

- **Long-running entry point: `scripts/dispenser_demo/deploy/dispenser_voice.py`.**
  Single board process: ArecordMic → openWakeWord → Silero VAD → CrispASR → `chat_board._run_turn` (with dispense override applied via side-effect `import chat_board_dispense`). 676 lines, pure stdlib + numpy + onnxruntime + vendored openwakeword + the chat_board library.
- **Board deploy layout (unchanged from Layer B + this new file)**:
  - `/mnt/sdcard/python-deps/site/{onnxruntime,openwakeword}/` (Layer B stage)
  - `/mnt/sdcard/dispenser_demo/{wake_stt_board_smoke,dispenser_voice}.py`
  - `/mnt/sdcard/models/functiongemma-270m/chat_board{,_dispense}.py`
  - FG model + STT model + binary paths unchanged.
- **Invocation**: `PYTHONPATH=/mnt/sdcard/python-deps/site python3 /mnt/sdcard/dispenser_demo/dispenser_voice.py`.

### Carried-over gap → Layer C.1 (closed 2026-05-12 inside Layer D)

**arecord stderr `overrun!!! (at least 5026.625 ms long)` during/after the
LLM turn.** Python is blocked in the llama-completion subprocess for
~6.5 s per turn; arecord's ALSA capture buffer (≈1 s) overflows during
that window. Effects on the next turn:

- Up to ~5 s of stale audio sitting in the buffer when LISTENING resumes.
- Risk of phantom wake from stale audio.
- Up to ~5 s wake-detection latency before real-time audio reaches the
  consumer.

**Closed 2026-05-12 inside Layer D**: `ArecordMic.drain(max_seconds=6)` —
non-blocking pipe drain (`fcntl.O_NONBLOCK` + `os.read` loop) called after
each LLM turn (and implicitly after TTS playback since the same code path
ran during SIGSTOP/SIGCONT). The 6 s cap is sized to cover the LLM
subprocess + TTS WAV combined; runs in <1 ms when there's nothing to
drain. Same logic prevents the post-playback stale-audio bug from Layer D
without a separate fix.

### Known carry-overs from Layer B (still applicable)

- **No espeak-ng / TTS yet.** Layer D promotes the §1 v2 non-goal to v1.

### When to reconsider

- If the overrun produces a phantom wake in real use, implement the
  drain-between-turns fix above.
- If LLM turn wall climbs past ~10 s (slower hardware, larger model),
  the same fix is mandatory; the overrun grows linearly with LLM wall.

---

## 2026-05-12 (Phase 3 Layer B closed) — Wake → VAD → STT pipeline works end-to-end on board

First-run measurement on SL2619 against the P10S USB mic, `plughw:1,0`,
fully cold (no warm caches):

| Phase | Wall | Note |
| --- | --- | --- |
| openWakeWord + Silero VAD model load | 0.97 s | ONNX runtime sessions × 4 (mel, embed, hey_jarvis, vad) |
| Wake-model prime (5 silent frames) | 0.13 s | covers the documented init-zeroing window from `openwakeword/model.py:332` |
| Wake-fire latency (from speech to `[WAKE]`) | ~0.45 s | 1 frame @ 0.967, 1 @ 0.901 — patience=2 satisfied immediately |
| VAD endpoint after silence_run=13 | — | speech captured cleanly; pre-speech silence (~1.5 s while user gathered the utterance) correctly *did not* end the listen state because `speech_seen` was still False |
| Audio captured | 3.68 s | 16-kHz mono PCM, 58 880 samples |
| CrispASR decode | 1.23 s | `--backend moonshine -l en --no-punctuation -t 2`, 3.2× realtime — matches Phase 0 envelope |
| **Total wake → `[TRANSCRIPT]`** | **~4.8 s** | first-utterance, no warm cache |

Transcript on the first try: `"give me my medication"` — correct.

### Binding

- Smoke entry point: `scripts/dispenser_demo/voice/wake_stt_board_smoke.py`
  (one-shot, `--probe`-equivalent for the wake→STT layer). Production
  long-running pipeline lands at `scripts/dispenser_demo/deploy/dispenser_voice.py`
  in Layer C.
- Board deploy layout — pinned at this run:
  - `/mnt/sdcard/python-deps/site/onnxruntime/` (manylinux_2_28 aarch64 wheel, extracted, 12 MB)
  - `/mnt/sdcard/python-deps/site/openwakeword/` (vendored .py + 4 ONNX models under `resources/models/`, 5.3 MB)
  - `/mnt/sdcard/dispenser_demo/wake_stt_board_smoke.py`
  - Invocation: `PYTHONPATH=/mnt/sdcard/python-deps/site python3 /mnt/sdcard/dispenser_demo/wake_stt_board_smoke.py`
- Stub gate at script top is binding — openWakeWord's `__init__.py` eagerly
  imports `train_custom_verifier` which transitively pulls scipy + sklearn +
  tqdm + requests; none are on the board image, none are touched at runtime,
  all four are stubbed via `sys.modules` before `import openwakeword`.

### Known gaps (carried into Layer C)

- **Listen-state hard cap is 5 s absolute, not 5 s post-speech-start.** If
  the user takes >5 s to start talking after wake, the cap fires and we
  capture mostly silence → empty transcript. Layer C should make the cap
  speech-relative (e.g., `max 5 s after first speech_seen=True frame`).
- **Trailing `arecord: pcm_read:2272: read error: Interrupted system call`
  on script exit is benign** (SIGTERM interrupting arecord's in-progress
  read syscall). Optional polish: capture arecord stderr silently in
  `ArecordMic.__exit__` if it's noise in the wider pipeline.
- **No mixer pre-set in the script.** Phase 3.1 in plan §9.3 calls for
  `PCM 50%, Mic 70%` on startup; deferred to the long-running entry point
  (smoke is one-shot, you set the mixer once per boot).

### When to reconsider

- If FPR climbs in a less quiet room, bump `--wake-threshold` to 0.6 or
  0.7. Current 0.5 + patience=2 fired clean here; the score was 0.967 so
  there's headroom.
- If the wake phrase tail leaks into the captured audio and hurts STT,
  bump `--prewake-rollback-frames` from 0 to a small negative offset
  (script doesn't currently support negative — would need an inverse
  "skip first N frames of LISTENING" knob).

---

## 2026-05-12 (Phase 3 pivot) — Demo uses iter-001 + dispatcher-hijack; iter-002 retained but not deployed

Despite iter-002 being trained (`releases/functiongemma-270m/002-dispenser-demo/`)
and routing correctly on board for `na-003`, two open quirks make iter-001
the safer demo substrate today:

1. **Iter-002 on-board wire format differs from iter-001.** It emits a bare
   `call: NAME{...}<end_function_call>` (no `<start_function_call>` opener)
   after a filler `information:` prefix (see the Phase 1.7 entry below).
   The iter-001 deploy path (`chat_board.py`) is fully proven against the
   `<start_function_call>...<end_function_call>` form.
2. **Iter-002 host eval Q4_0 collapsed to 30 %** (Phase 1.7 entry). Q4_0 is
   the on-board variant; the gap between host (broken) and board (correct)
   makes pre-deploy regression checks awkward.

The dispenser-demo functional surface is small enough that we can reuse
iter-001's existing tool registry instead of waiting on iter-002 to land
end-to-end.

### Binding

- **Active demo entry point: `scripts/functiongemma/deploy/chat_board_dispense.py`.**
  Thin wrapper around `chat_board.py` that monkey-patches `dispatch` and
  `format_response` so `get_medications_at_time` and `get_medication_by_name`
  short-circuit to the dispense intent: print `[BLE→ESP32] 5A A5 01 00`
  (mock BLE notify per plan §6.2) and return `{"status": "dispensed"}`; the
  formatter emits the verbatim §6.1 canned response. Tool schema unchanged
  → warm prompt cache `/tmp/fg_pc_finetuned_functiongemma_q4_0.gguf.bin`
  reuses across the original `chat_board.py` and the wrapper.
- **Patient fixture: `data/health_table_v1_dispense_demo.yaml`** (renders to
  `health_table_dispense.json` on the board). Same v1 schema, fresh patient
  (David Smith) so the non-dispense tools (vitals/allergies/appt/contact/
  food-interaction) still demo cleanly without exposing the iter-001 test
  patient identity.
- **The `dispense_medication()` named tool from plan §6.1 is NOT deployed
  for v1.** It remains the iter-002 target shape. Plan §6.1 and §7 retain
  it as the design intent; the wrapper is the bridge until iter-002 ships
  on board.
- **Mock BLE for v1.** Wrapper prints the payload to stdout. Real `pybleno`
  notify on characteristic `0xFFB2` lands in Phase 2 (currently deferred —
  not blocking the wake→STT smoke).

### When to reconsider

- When the iter-002 on-board wire-format quirk is reconciled (parser regex
  relaxation per the Phase 1.7 entry, validated across a 10-row board
  sweep), swap the wrapper out for the iter-002 release. The tool name in
  the registry becomes `dispense_medication()` per plan §6.1 / §7.
- When real BLE is wired (Phase 2), replace the stdout `[BLE→ESP32]` line
  in `chat_board_dispense.py:dispatch` with a `pybleno` notify call.

---

## 2026-05-12 (Phase 3 wake-word) — Pretrained openWakeWord `hey_jarvis_v0.1`; custom `Hey Sago` training deferred

Plan §3.2 left the choice open ("Train Hey Sago OR find pretrained"). For
the v1 demo we pick pretrained:

- Upstream `dscripka/openWakeWord` ships a `hey_jarvis_v0.1` model
  (TFLite + ONNX, ~3 MB) at
  <https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite>
  (replace `.tflite` with `.onnx` for the ONNX variant). The submodule
  ships only source; models are fetched on first use.
- Custom training "Hey Sago" requires a Colab run + synthetic data +
  threshold tuning (plan §11 R4 budgeted one day). Pretrained "Hey Jarvis"
  is available immediately and is a known-good baseline for shaking out
  the wake → STT plumbing.
- Inference framework: **ONNX** (not TFLite), because Silero VAD is
  ONNX-native; keeping a single runtime on the board avoids a duplicate
  wheel install. Both models load through `onnxruntime` ≥ 1.16.

### Binding

- Wake phrase for v1 demo: **"Hey Jarvis"** (not "Hey Sago"). Plan §1
  Goals + §4 architecture-diagram label updated in lock-step.
- Runtime: ONNX for both wake-word and VAD. Local cache lands under
  `models/wakeword/hey_jarvis_v0.1.onnx` and `models/vad/silero_vad.onnx`
  (gitignored; bootstrap script downloads on first run).
- Custom `Hey Sago` training is a **future iteration**, not a v1 gate.
  Plan §11 R4 (custom-model FPR/TPR risk) retires for v1 — pretrained
  performance is upstream-validated.

### When to reconsider

- When the demo's branding requires "Hey Sago" specifically (or another
  product-named wake phrase), schedule the openWakeWord training pass per
  the upstream `notebooks/training_models.ipynb` recipe.

---

## 2026-05-12 (Phase 3 smoke topology) — WSL host first, board second; speaker deferred until wake→STT proves out

> **Revised same day:** WSL2 (`PHL`) has no input device exposed through WSLg
> — `pactl list sources short` shows only `RDPSink.monitor` + `RDPSource`,
> and PortAudio cannot bind a usable mic (the host has no `libportaudio2`
> installed and the Windows side has not surfaced a mic to WSL). **Layer A
> is therefore SKIPPED for this user's environment**; Phase 3 starts at
> Layer B (board, wake → STT). Layer A remains the documented entry point
> for future sessions on a machine with a working mic.

Plan §3.1–§3.6 implies on-board work from day one. For the iter-001 +
dispatcher-hijack pivot we sequence smoke layers explicitly to limit blast
radius per integration:

1. **Layer A — host WSL, wake → STT only.** openWakeWord (Hey Jarvis) +
   Silero VAD + Moonshine via CrispASR `--backend moonshine`. No LLM, no
   speaker. Output: stdout transcript. Closes the audio-plumbing surface
   in isolation.
2. **Layer B — board, wake → STT.** Same code path on SL2619 with the P10S
   USB mic. Confirms the ONNX runtimes deploy cleanly and timing fits in
   the budget alongside FunctionGemma's KV cache (~600 MB RAM ceiling).
3. **Layer C — board, full pipeline.** Wire Layer B into
   `chat_board_dispense.py`. Output is still stdout (no espeak-ng yet).
4. **Layer D — board, with espeak-ng on speaker.** Add aplay-piped
   espeak-ng output for the answer. Promotes TTS from plan §1 v2 non-goal
   to v1 demo gate. Half-duplex rule (mute mic during playback) applies
   per §8.3 E2.

Each layer is independently testable and skippable in dev (hand audio
files into Moonshine to bypass mic; hand transcripts into the LLM to
bypass STT; etc.).

### Binding

- New code lands under `scripts/dispenser_demo/voice/` (host-side smoke +
  shared modules) and a thin board entry point at
  `scripts/dispenser_demo/deploy/dispenser_voice.py` when Layer B starts.
- **Done criterion for THIS sprint: Layer B (board).** Live mic on the
  SL2619 via the P10S USB capsule, say "Hey Jarvis" then a command, see
  the Moonshine transcript on stdout from the SSH session. (Layer A on
  WSL skipped — see callout above.)
- TTS via espeak-ng deferred to Layer D — keeps the speaker hardware path
  out of the wake/STT debugging loop.

### When to reconsider

- If a future session has a working WSL/host mic, Layer A is the faster
  iteration loop — re-enable it before touching board scripts. Today's
  pivot to Layer B was forced by missing mic plumbing, not chosen.

---

## 2026-05-12 (Phase 1.7) — Q4_0 ships on board; host eval invalid for Q4_0; parser regex needs relaxation for iter-002

Iter-001 documented Q4_0 as the only quant that decodes cleanly on the
SL2619's `llama-completion b8925` (K-quants drop `<start_function_call>`
on that older build). For iter-002 the asymmetry is preserved but with
two new wrinkles:

1. **Host eval of iter-002 Q4_0 collapsed to 30 %.** Output corrupted —
   `<start_function_call>len_of_age_digits...` style gibberish. Per
   iter-001's "Why every quant except Q4_0 fails on this board build"
   §, this is consistent with Q4_0's symmetric INT4 representation
   losing precision on weight distributions with extreme outliers; the
   newer host runtime is stricter than the older board runtime. K-quants
   (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `IQ4_XS`) all score 100 % on host.
2. **On-board Q4_0 routes correctly** (`docs/bench-notes/dispenser-demo/2026-05-12_iter-002-q4_0-on-board-smoke.md`)
   — tool routing on `na-003 "When do I see Dr. Chen?"` returns
   `get_next_appointment{}`, 10.39 tok/s decode, 849 MiB RSS. Matches
   iter-001's on-board envelope.
3. **The on-board output omits the `<start_function_call>` opener.**
   Iter-001's model emits `<start_function_call>call:NAME{...}<end_function_call>`;
   iter-002 emits a bare `call: NAME{...}<end_function_call>` after a
   filler `information:` prefix. Tool name + args still correct, but the
   wire format differs from iter-001.

### Binding

- **On-board production GGUF: `finetuned_dispenser_q4_0.gguf`.** Sha256
  `85893a795aec4b2adc2dbc7084f5b27e3ecd5a1ef885fd69d5af9678632368b9`,
  pinned in `releases/.../002-dispenser-demo/gguf/CHECKSUMS.txt`.
- **Host eval MUST use FP16 (or Q5_K_M / Q8_0).** Q4_0 host scores are
  meaningless. `scripts/dispenser_demo/eval/eval_holdout.py --gguf
  finetuned_dispenser_q4_0.gguf` is a known-broken combination; do not
  re-run as a gate. Use `--gguf finetuned_dispenser_fp16.gguf` for any
  re-validation pass.
- **Phase 3 parser regex needs relaxation.** The current iter-001
  parser at `scripts/functiongemma/eval/eval_holdout.py:_FG_CALL_RE`
  requires the `<start_function_call>` opener; iter-002 on-board output
  omits it. Update to make the opener optional:

  ```python
  _FG_CALL_RE = re.compile(
      r"(?:<start_function_call>)?\s*call\s*[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
      re.DOTALL,
  )
  ```

  Backward compatible with iter-001 (`(?:...)?` matches an empty prefix).
  Same update goes in `scripts/dispenser_demo/eval/eval_holdout.py` (host
  side, will be a no-op since host eval should not use Q4_0) and the
  forthcoming Phase 3 on-board dispatcher.

### When to reconsider

- If Phase 4 acceptance reveals on-board routing failures on rows other
  than `na-003`, a full 10-row on-board sweep is warranted before declaring
  iter-002 deployable end-to-end. `scripts/functiongemma/deploy/ask_board.sh`
  is the iter-001 template to mirror.
- If a future llama-completion update on board surfaces K-quant support,
  re-run the 2026-05-12 host sweep on board — Q5_K_M would be a better
  default than Q4_0 (higher precision at similar size; doesn't carry the
  symmetric-INT4 precision loss).
- If retune is needed (the remaining 1 of 2 free runs), one fix worth
  trying is tighter `tuning.num_train_epochs` to reduce the weight
  outliers that destabilize Q4_0 quantization on host.

---

## 2026-05-12 (Phase 1.6 eval) — Distil's SYSTEM_PROMPT is required for accurate host eval

The Distil-trained student is conditioned at training/inference on a
SPECIFIC system-prompt wrapping, NOT on the seed JSONL's SYSTEM_TRIGGER
string. This was non-obvious; seeds use the FG-style trigger because
that's the on-disk training-data format Distil consumes, but Distil's
synthgen + training pipeline SWAPS the system message for the
task_description-wrapping prompt before the model sees it. The deployed
inference path (`releases/.../model_client.py`) reproduces the wrap; any
eval that sends the seed's verbatim system message instead measures the
wrong input distribution and scores systematically lower.

### What the wrap looks like

```
You are a tool-calling model working on:
<task_description>{task_description from job_description.json}</task_description>

Respond to the conversation history by generating an appropriate tool call
that satisfies the user request. Generate only the tool call according to
the provided tool schema, do not generate anything else. Always respond
with a tool call.
```

`{task_description}` is the verbatim string from
`releases/functiongemma-270m/002-dispenser-demo/distil/job_description.json`.
The `<task_description>` tags + the trailing meta-instruction ("Respond
to the conversation history…", "Always respond with a tool call") are
the wrapping Distil applies; both pieces are part of what the student
expects to see.

### Bound impact

- Eval with seed-as-is SYSTEM_TRIGGER: 7/10 val (70 %) — 3 mismatches
  on phrasings the student saw indirectly via synthgen (Dr. Chen named
  provider, "Drop my meds" colloquial dispense).
- Eval with Distil SYSTEM_PROMPT: **10/10 val (100 %)** — every category
  PASS at the ≥ 90 % gate.

### Binding

- `scripts/dispenser_demo/eval/eval_holdout.py` defaults to the Distil
  wrap, derived at runtime from `job_description.json` (`--job-description
  <path>`). Single source of truth; drift-gated by the existing
  `tests/dispenser_demo/test_distil_alignment.py`.
- A `--seed-as-is` flag is preserved as a debug knob (reproduces the
  70 % failure mode) but should NEVER be the default for an
  acceptance-gate run.
- The on-board `chat_board.py` (Phase 3) must mirror Distil's wrap
  byte-for-byte. The reference is `releases/.../002-dispenser-demo/model_client.py`
  (note: that bundled file has an f-string syntax bug from Distil's
  generator — literal `{}` in an f-string body; do NOT `import` it
  directly. Read the SYSTEM_PROMPT_CONTENT verbatim into your own
  module.)

### When to reconsider

- If Distil rolls a CLI version that changes the wrapping template
  (e.g. removes the `<task_description>` tags), the bench-note score
  will drop. Re-run with `--seed-as-is` to confirm the model itself
  hasn't regressed, then update `_DISTIL_SYSTEM_TEMPLATE` in
  `eval_holdout.py` to the new wrap.
- If a future iteration trains on a non-Distil pipeline (e.g. local
  Unsloth fallback), the system-prompt wrapping changes again. Add a
  second flag like `--unsloth-prompt` rather than overloading the
  default.

---

## 2026-05-11 (Phase 1.4 rebalance) — `out_of_scope_refusal` seeds 8 → 10; `health_advice` ramped from 3 → 5

Phase 1.4 advisor pass flagged that the 8-row refusal slot (3 `health_advice`
+ 5 `off_topic`), after the 3/2/2-style sub-stratified split, gave Distil
only **1** train row for `health_advice` — significantly thinner than the
4 train rows the four domain categories each carried. Synthgen's
`generation_target=1500` would have paraphrased that single seed; the
teacher (gpt-oss-120b) is competent enough to expand, but starting from
1 example is materially riskier than from 3.

- **Binding:** the seed count grows from 40 → 42 rows. `out_of_scope_refusal`
  becomes 10 rows (5 `health_advice` + 5 `off_topic`). The four domain
  categories stay at 8 each. The "8 per category × 5 = 40" invariant
  documented in the prior decisions-log entry is **superseded** —
  `out_of_scope_refusal` is intentionally larger to give per-reason
  coverage parity.
- **New seeds (both `health_advice`):**
  - `oo-004`: "Is it safe to mix ibuprofen with my regular meds?"
    (drug-interaction question — a real failure mode the dispenser
    shouldn't answer).
  - `oo-005`: "How much aspirin should I take for back pain?"
    (dosage-advice question — same).
- **Original `oo-004..008` (off_topic) renumber to `oo-006..010`** so
  ID order preserves "health_advice first, off_topic second" within
  the refusal stratum. `build_splits.py`'s sub-stratification picks
  the right rows regardless of position, but ID order matches reason
  order for human readability.
- **Resulting split** (per `build_splits.py` 60/20/20 with sub-stratum
  3/1/1 on each refusal reason):

  | category | train | val | test |
  | --- | ---: | ---: | ---: |
  | patient_profile | 4 | 2 | 2 |
  | next_appointment | 4 | 2 | 2 |
  | emergency_contact | 4 | 2 | 2 |
  | dispense | 4 | 2 | 2 |
  | out_of_scope_refusal | 6 (3+3) | 2 (1+1) | 2 (1+1) |
  | **TOTAL** | **22** | **10** | **10** |

- **Distil iter-002 inputs (after rebalance):** train 22 / test 10 — both
  comfortably above the platform's 20-train-row floor.

### What this means for downstream phases

- **Phase 1.3** (`build_splits.py`): no logic change — the sub-stratification
  already handled uneven counts. Only the output sizes change.
- **Phase 1.4** (`distil/{config,job_description,README}.md`): row counts
  updated; mutation_topics already enumerate both reason classes.
- **Phase 1.6 holdout eval:** per-cat pass-rate gate (≥ 90 %) applies to
  `out_of_scope_refusal` overall, not split per reason. The judge instructions
  enforce exact `reason` equality so cluster-level signal is still legible.

### Test surface

`tests/dispenser_demo/test_dataset_validator.py::test_seed_file_validates_at_full_pass_rate`
was updated to expect `total == 42` and `out_of_scope_refusal: 10`. The
drift gate at `test_distil_alignment.py` is unaffected (it gates schema
shape, not row counts).

---

## 2026-05-11 (late) — Phase 2 BLE host implementation landed; board bring-up blocked by 3 user-side preconditions

Phase 2 BLE bring-up was split from P10S audio (already verified — see
`docs/guides/usb-audio-testing-sl2619.md`). The host-side BLE work is
complete; the board smoke run is blocked on three sequential
preconditions, all user-executed (R3 forbids the agent from mutating
the board).

### Host work landed

- `src/gemma_tools/dispenser_demo/ble_client.py`: `BleClient` Protocol
  (`@runtime_checkable`), `MockBleClient` for unit tests, and
  `PyBlenoBleClient` (lazy pybleno import; `BLENO_HCI_DEVICE_ID` set
  *before* the first pybleno import so bleno's module-init snapshot
  picks it up). Wire-contract constants pinned: `DEVICE_NAME="NousVoice"`,
  `ADV_SERVICE_UUID="00FB"`, `PRIMARY_SERVICE_UUID="FFB0"`,
  `NOTIFY_CHAR_UUID="FFB2"`, `DISPENSE_PAYLOAD=b"\x5a\xa5\x01\x00"`.
- `scripts/dispenser_demo/deploy/ble_test.py`: standalone board smoke
  runner. CLI: `--hci hciN | --hci-index N` (mutex), `--timeout-s`,
  `--send-once`, `--skip-patch-check`, `--verbose`. Prints precondition
  hints (does not execute `hciconfig hciX down`); checks 4 of 5
  setup-guide patch sentinels in `BluetoothHCI.py` (`AF_BLUETOOTH = 31`,
  no `'rU'`, `HCI_CHANNEL_USER`, `hciconfig` fallback — the 5th patch
  is structural and has no grep-able sentinel).
  WiFi-config characteristics from the old reference peripheral
  (`docs/references/old-dispenser-demo/ble_peripheral.py`, FFB3..FFB8)
  are intentionally omitted — dispenser demo only needs the notify path
  on 0xFFB2.
- `tests/dispenser_demo/test_ble_client.py` (23 cases) +
  `test_ble_smoke_script.py` (20 cases): table-driven, no
  pybleno/board/network required, no sleeps. Mirrors the
  `test_crispasr_smoke_scripts.py` pattern for the smoke-script
  argparse tests. Full suite: 680 passed.

### Board bring-up — three blockers (in order)

`/board_probe` 2026-05-11 (snapshot at `docs/tmp/sl2619-status.md`) found:

1. **No `hciX` interface** — `/sys/class/bluetooth/` is empty,
   `hciconfig` returns rc=0 with no output. The Broadcom BT chip is
   unregistered, not just down. **The old-doc RTL8822BU USB BT adapter
   is NOT present on this board today** — `lsusb` shows only the
   Genesys hub and the P10S USB mic. The path forward is the Broadcom
   M.2 combo plumbed over UART (HCI UART H4 + BCSP transports are
   registered in dmesg).
   - User action: identify the BT UART (likely `/dev/ttyS1..ttyS5`) and
     the correct `.hcd` from `/usr/lib/firmware/bcm/` (NOT
     `/lib/firmware/brcm/`; the board_probe skill currently checks the
     stale path), then run something like
     `brcm_patchram_plus --enable_hci --use_baudrate_for_download
     --baudrate 3000000 --patchram /usr/lib/firmware/bcm/BCMxxxxxx.hcd
     /dev/ttySN`. Verify `hciconfig` then lists `hci0`.
2. **`pybleno` not installed.** `import pybleno` →
   `ModuleNotFoundError`; no `BluetoothHCI.py` anywhere on the system.
   - User action: `pip3 install pybleno --break-system-packages` per
     `docs/references/old-dispenser-demo/pybleno-setup-guide.md` §1.
3. **Setup-guide patches not applied** (cannot apply until 2 lands).
   - User action: apply the 5 patches from
     `docs/references/old-dispenser-demo/pybleno-setup-guide.md` §3–§7.
     `ble_test.py` checks the `AF_BLUETOOTH = 31` sentinel and bails
     loudly if missing (use `--skip-patch-check` to override).

The user's anticipated "bring `hci0` down for `HCI_CHANNEL_USER`
exclusive access" concern is **moot until blocker 1 clears** — there
is no `hci0` to bring down. Once `hciconfig` lists `hci0`, the
standard pattern from the legacy peripheral applies: bring `hciX`
down immediately before launching `ble_test.py --hci hciX`.

### Fallback rule (carried forward from plan §9 step 2.3)

If, after the three blockers clear, pybleno still cannot bind the
Broadcom M.2 adapter (BlueZ socket errors, kernel mismatch beyond the
5 documented patches), the plan's branch rule applies: try
`bluez-peripheral` or a thin D-Bus shim. No D-Bus pivot was attempted
in this session — pybleno was not exercised against real hardware
yet, so there is no evidence justifying the switch. Stop and report
if pybleno actually fails; do not pre-empt.

### 2026-05-11 (late, addendum) — BT chip silent on UART; image's BT bring-up appears unvalidated

Attempted Blocker 1 directly on the board with user authorization.
Outcome: **`hci0` could not be brought up.** Pivot recommended.

Findings (all read-only diagnostics on `nouslogic-sl2619`):

1. **Vendor `brcm_bt_start.service` is broken from the factory.** Its
   `ExecStart` passes `--patchram /lib/firmware/bcm` (the *directory*),
   not a specific `.hcd` file. `brcm_patchram_plus` doesn't walk the
   directory, so no patch ever uploaded. The `.hcd` files are dated
   `Mar 9 2018` — never modified — which is consistent with "BT was
   never exercised end-to-end on this image."
2. **Chip silicon is SYN43711A0** (BCM43711 combo, BT half of the
   `fw_sd_bcm43711.bin` WiFi half). The matching patch is
   `/lib/firmware/bcm/SYN43711A0_001.001.005.0019.0000_Generic_UART_37_4MHz_wlbga_REF_sLNA_iLNA_ANT0.hcd`.
3. **Correct invocation fails identically.** With the fixed `.hcd`
   filename, after `rfkill block; rfkill unblock` power-cycle (driver
   logs `bluetooth_set_power: power up = 0` — chip is powered),
   `brcm_patchram_plus -d --tosleep=300000 --baudrate 3000000
   --use_baudrate_for_download --no2bytes --enable_hci --patchram <hcd>
   /dev/ttyS1` loops indefinitely writing the HCI Reset frame
   `01 03 0c 00` and never receives a response. The chip is silent on
   ttyS1 at the boot baud (115200) and on the operational baud
   (3 Mbps). Same symptom on `/dev/ttyS2`, which turns out to be a
   `serial8250` placeholder (`uartclk=0`) and not a real UART —
   `bt_vendor.conf`'s `UartPort = /dev/ttyS2` is stale.
4. **`hciattach` (lighter-weight alternative) is also silent.**
   `hciattach -t 30 -s 115200 /dev/ttyS1 bcm43xx 921600 flow` produces
   no output and no `hci0`.
5. **No USB / SDIO HCI transport.** `btusb`, `btsdio`, `btintel`,
   `btrtl` are NOT in the kernel (built-in modules list +
   `/lib/modules/.../*` both empty). The kernel only supports
   `hci_uart`. A USB BT dongle would not bind on this image.
6. **SDIO function 3 has CLASS=02** (the SDIO BT class) but is claimed
   by `bcmsdh_sdmmc` (Broadcom WiFi driver) and there's no `btsdio.ko`
   to bridge it to `hci_sdio`. The combo chip's BT half is reachable
   over SDIO inside the driver but never exposed as an HCI device.
7. **UART driver does not implement full termios.**
   `stty -F /dev/ttyS1 -a` returns `Inappropriate ioctl for device`.
   The Synaptics UART driver may not honor hardware flow control the
   way `brcm_patchram_plus` assumes — plausible root cause for the
   silent chip, but not user-fixable without kernel changes.

**Diagnosis (one line).** The image ships the kernel + firmware to
support BT but not a validated userspace bring-up. This is hardware
enablement territory, not a flag-tuning task.

**Disposition for Phase 2.** Board BT bring-up is **out of scope for
this session and likely for any further dispenser-demo work without
vendor support from Synaptics.** The host-side BLE implementation
(client + smoke script + tests, 43 cases green) is complete. To
actually exercise `ble_test.py` against a phone (nRF Connect) or the
ESP32, run it on a different Linux peripheral with a working radio —
WSL2 + `usbipd-win` + a CSR4.0 USB BT dongle, or any spare Linux box
with native BT. The wire contract is unchanged; the script is
hardware-agnostic by design.

**Do not retry** flag permutations, alternative `.hcd` files, or
alternative UARTs without new information (e.g., a Synaptics support
ticket result, a working vendor invocation from another image, or a
kernel rebuild with `btsdio` enabled). Five hours of diagnosis above
already covered the productive variations.

### Ground-truth confirmation (datasheet pass, 2026-05-11 late)

Per AMPAK `docs/references/AP12611_M2_datasheet.pdf` and Synaptics
`docs/references/upstream/synaptic-sl2619/docs/datasheets/sl2610-datasheets/astra-machina-sl2600-dev-kit-user-guide.pdf`:

- **Module is AMPAK AP12611_M2** with SYN43711 (BCM43711 family) — Wi-Fi
  over SDIO, **BT over UART (HCI UART up to 4 Mbps, default 115.2 Kbaud,
  CRTSCTS)** per AP12611 §1.2, §4.1, §8.4. The dev-kit guide §3.2 says
  "Wi-Fi/BT devices with SDIO" but that's shorthand for the Wi-Fi half
  only; BT physically goes UART.
- **BT UART is SM_URT1** → `/dev/ttyS1` (e5031000.uart,
  `snps,dw-apb-uart`) per dev-kit Table 10 (SM Pin-demuxing): SM_GPIO7
  (RXD), SM_GPIO8 (TXD), SM_GPIO14 (CTS_N), SM_GPIO15 (RTS_N), all on
  mode OPT7.
- **BT_REG_ON is on the I/O expander at I²C 0x44** as `gpiochip5 line 5
  consumer=bt_power` — the same physical M.2 pin 54 that the M.2 Key E
  spec calls `W_DISABLE2#`. AP12611 datasheet §5.2 reassigns pin 54 as
  the BT power-enable. Synaptics rfkill driver (`bluetooth-rfkill.c`
  line 58-72) pulses LOW for 10ms then HIGH for 150ms on rfkill unblock.

### Live ground-truth probe (read-only, BT_REG_ON asserted)

After `rfkill unblock bluetooth` (BT_REG_ON HIGH), opening `/dev/ttyS1`
at 115200/921600/1.5M/3M baud with CRTSCTS enabled (kernel default for
this port) and listening passively for 2 s each — **zero bytes from the
chip at any baud**. Active HCI Reset (`01 03 0c 00`) also drew no
response. The chip is dead silent.

### Unverifiable hardware-layer suspects (no schematic on hand)

1. **1.8V VDDIO rail may not be enabled.** AP12611 §2.2.2 requires both
   3.3V VBAT and 1.8V VDDIO. No live GPIO is labeled `vddio_en` /
   `m2_vio_en`; the WLAN side works because it powers off VBAT through
   SDIO, but BT_REG_ON is a 1.8V signal and the BT silicon needs VDDIO
   to operate. If VDDIO is off, BT_REG_ON HIGH does nothing.
2. **SM (M52) pinstrap may leave SM_GPIO7/8/14/15 in default mode**
   instead of OPT7. The Linux UART driver registers `ttyS1` regardless
   of physical routing — so opening `/dev/ttyS1` succeeds even if the
   pins aren't actually wired to the M.2 connector. Verifying this
   needs M52 firmware source access, not on this image.
3. **The vendor `brcm_bt_start.service` shipped with a directory-bug**
   (`--patchram /lib/firmware/bcm` is a directory not a file) and the
   `.hcd` files are dated `Mar 9 2018`, unmodified. Strong evidence
   the BT path was never validated on this Yocto image.

### Final disposition

Bringing up `hci0` on the SL2619 board is **a hardware-enablement task
that requires Synaptics support, board schematic access, or both.** It
is not a flag-tuning task. Five hours of read-only diagnosis confirms
the software layer is healthy (UART driver fine, CRTSCTS fine, rfkill
fine, kernel BT stack fine) — the failure is in the analog / mux /
power layer beneath the kernel.

The host-side BLE work is complete and unblocks any other peripheral:

- WSL2 + `usbipd-win` + a USB BT dongle (CSR4.0 / RTL8761 ~$5)
- Any spare Linux laptop / Pi with native Bluetooth
- The script + tests are hardware-agnostic by design

The board itself can rejoin Phase 2 once Synaptics confirms (a) which
GPIO controls M.2 VDDIO_EN and (b) whether SM_URT1 OPT7 muxing is
applied at boot.

### Vendor-source root-cause confirmation (2026-05-11 deep dive)

Initialized all Synaptics submodules at the
`scarthgap_6.12_v2.3.0` branch (matching the live board image) and
audited the BT bring-up path end-to-end. Four findings make the root
cause unambiguous:

1. **`klamath_brcm_bt_start.patch` exists, but only fixes UART number,
   not the `--patchram` directory bug.** Located at
   `references/Synaptics/sdk/meta-synaptics/recipes-devtools/synasdk/files/klamath_brcm_bt_start.patch`,
   authored by Rohit Tayal (Synaptics) on 2025-09-16. Changes
   `/dev/ttyS2` → `/dev/ttyS1` for klamath. Keeps
   `--patchram /lib/firmware/bcm` (the directory) — relying on the
   `0001-bt-auto-detect-chip-type-and-download-fw.patch` to make
   `brcm_patchram_plus` walk the directory and pick the right `.hcd`
   via `HCI_Read_Local_Name`. That auto-detect path requires the chip
   to respond to `proc_reset()` first — which it never does on our
   board.

2. **DT enables `&uart1` but applies NO `pinctrl-0`.** From the live
   board image's source
   (`arch/arm64/boot/dts/synaptics/sl261x-rdk-common.dtsi` line 393):

   ```dts
   &uart1 {
       status = "okay";
       /delete-property/ dmas;
       /delete-property/ dma-names;
   };
   ```

   No pinctrl reference, no pmux group. The `snps,dw-apb-uart` driver
   sees `uart1` as enabled, but the SoC pads for SM_URT1_RXD/TXD/CTS/RTS
   are not muxed to OPT7 via kernel pinctrl. Per dev-kit Table 10 those
   are System Manager pads (SM_GPIO7/8/14/15) and the kernel can't drive
   them — they're owned by the M52 / SM CM3 bootloader.

3. **`bluetooth-lpm.c` itself says the SM pinmux is bootloader-owned.**
   File header comment in
   `references/Synaptics/linux-drivers-synaptics/bluetooth/bluetooth-lpm.c`:

   ```c
   /*bt-host-wake-gpio is connected into SM_GPIO[6]
    *which is handled in bootloader                                       */
   ```

   So SM_GPIO routing for BT lives in the SM CM3 bootloader, not in
   Linux DT.

4. **Smoking gun — there is NO klamath SM bootloader customization.**
   `references/Synaptics/boot/bootloader/sm_cm3/syna/customization/`
   has subdirectories for `platypus` (SL1640) and `dolphin` (SL1680)
   only. There is no `klamath/` directory. `board_wifi_poweron()` /
   `board_wifi_poweroff()` are defined per-platform in those custom
   dirs (e.g. `platypus/platypus-rdk/platform_customization.c` writes
   to the FXL6408 expander GPIO to enable WiFi/BT). Without an
   analogous klamath file, the SM bootloader's WiFi/BT power-on hook
   is a no-op and the SM pin-mux to route SM_URT1 to the M.2 connector
   never happens.

   WiFi nonetheless works on the board because the kernel's
   `sdhci1_pwrseq` separately toggles `expander1 line 2` to power the
   SDIO Wi-Fi side. BT has no such kernel-side analogue for its
   UART-pin muxing — that's an SM-only operation.

### Why this matches the release-notes bug 37861/37374

The publicly tracked bug "Bluetoothctl is not working" on SL2611/2615/2619
in scarthgap_6.12_v2.3.0 reduces to: **the SL2619 (klamath) SM CM3
bootloader is missing the platform customization that routes the
M.2 BT UART signals.** Synaptics shipped a partial Linux-side fix
(`klamath_brcm_bt_start.patch`, UART number) but not the M52/SM
bootloader-side fix (pin-mux + GPIO expander toggle).

### What this closes

Investigation **closed for real this time.** No software-only fix
possible from a Linux user-space or kernel module patch on the
shipped image. The fix requires:

- A new `bootloader/sm_cm3/syna/customization/klamath/klamath-rdk/platform_customization.c`
  (Synaptics internal work).
- An updated SM CM3 firmware binary flashed via `astra-update`.
- Optionally a corrected `klamath_brcm_bt_start.patch` that pins a
  specific `.hcd` file as belt-and-braces.

Until Synaptics ships the SM bootloader update with klamath
customization, **board BT on SL2619 cannot come up.** The dispenser
demo's BLE peripheral test will run from a different Linux host
(WSL2 + USB BT dongle, or any laptop / Pi with native BT). The
host-side BLE code (`ble_client.py`, `ble_test.py`, 43 tests) is
hardware-agnostic and works unchanged on those targets.

### Reference paths (initialized submodules)

For future sessions that want to verify or extend this audit:

- `docs/references/upstream/synaptic-sl2619/references/Synaptics/sdk/meta-synaptics/recipes-devtools/synasdk/files/klamath_brcm_bt_start.patch`
- `docs/references/upstream/synaptic-sl2619/references/Synaptics/sdk/meta-synaptics/recipes-devtools/syna_connectivity/brcm-patchram-plus/0001-bt-auto-detect-chip-type-and-download-fw.patch`
- `docs/references/upstream/synaptic-sl2619/references/Synaptics/boot/bootloader/sm_cm3/syna/customization/`
  (note absence of `klamath/`)
- `docs/references/upstream/synaptic-sl2619/references/Synaptics/linux-drivers-synaptics/bluetooth/bluetooth-lpm.c:10-12`
- `docs/references/upstream/synaptic-sl2619/references/Synaptics/astra-doc/release_notes/scarthgap_6.12_v2.3.0.rst:827`
- DT source (fetched from GitHub raw, not in submodule):
  https://raw.githubusercontent.com/synaptics-astra/linux_6_12-main/scarthgap_6.12_v2.3.0/arch/arm64/boot/dts/synaptics/sl261x-rdk-common.dtsi

### Smoking gun — Synaptics-confirmed known bug (closes the investigation)

The board runs `scarthgap_6.12_v2.3.0`. The matching release notes at
`docs/references/upstream/synaptic-sl2619/references/Synaptics/astra-doc/release_notes/scarthgap_6.12_v2.3.0.rst`
list two open Bluetooth bugs against **SL2611, SL2615, AND SL2619**:

```
| SL1620 | SL1640 | SL1680 | SL2611 | SL2615 | SL2619 | Module    | ID    | Summary                       |
|  N/A   |  N/A   |  N/A   |   Y    |   Y    |   Y    | Bluetooth | 37861 | Bluetoothctl is not working.  |
|        |        |        |        |        |        |           | 37374 |                               |
```

The headline feature-matrix on line 378 marks Bluetooth as "Supported"
on SL2619, but the actual implementation is filed as broken in this
exact image. That matches every empirical symptom we observed
(chip silent on UART, vendor `brcm_bt_start.service`'s directory-bug
shipping unfixed, no working bring-up script anywhere in the BSP).

The intended transport on SL26xx **is UART** (not SDIO), per:

- AMPAK AP12611_M2 datasheet (BT over HCI UART, default 115.2 Kbaud, CRTSCTS)
- Dev-kit guide Table 10 (SM_URT1_{RXD,TXD,CTS_N,RTS_N} → WIFI/BT Module)
- `bluetooth-lpm.c` includes `hci_uart.h` and the comment
  "bt-host-wake-gpio is connected into SM_GPIO[6] which is handled in
  bootloader"
- Vendor `brcm_bt_start.service` calls `brcm_patchram_plus` (UART tool)

The `dhd_bt_interface.h` BT-over-SDIO exports in bcmdhd103 are
leftover from older Broadcom builds for other Synaptics SoCs (SL1640
runs an SDIO Wi-Fi+BT stack on `bcmdhd361`); no kernel consumer is
built/loaded on this image. Searching the entire
`linux-drivers-synaptics` tree shows no out-of-tree `bcmsdh_btsdio`
driver — the BT-over-SDIO path is unbuilt for SL26xx.

**Investigation closed. The fix is upstream from Synaptics. Hardware-
layer hypotheses listed above (missing VDDIO_EN, M52 pinstrap) remain
candidate root causes of bug 37861/37374, but identifying which one
is up to Synaptics, not us.**

The host-side BLE implementation (`ble_client.py` + `ble_test.py` +
43 host tests) ships unchanged. To actually demo NousVoice with
nRF Connect today: run `ble_test.py` from any Linux peripheral with
a working BT radio (USB dongle on WSL2 via `usbipd-win`, or a spare
laptop / Pi). Code is hardware-agnostic by design.

---

## 2026-05-11 (evening) — Phase 1.1 refusal shape: `refuse_out_of_scope(reason)` tool, not no-tool-call

Initial plan §7 modeled refusals as no-tool-call assistant turns
(`tool_calls: null`, canned NL). That contradicts the Distil
`multi-turn-tool-calling-closed-book` task contract, which enforces
"exactly one tool call per assistant turn." Pinned tool-call shape after
external evidence review:

- **Binding:** all 8 `out_of_scope_refusal` seed rows emit a tool call to
  `refuse_out_of_scope(reason)`, where `reason` is the two-value enum
  `["health_advice", "off_topic"]`. Tool response is `{"status": "refused"}`;
  the canned NL `"I can only help with your patient profile, appointments,
  emergency contact, or dispensing medication."` is the same for both
  reasons. The reason enum exists for offline analytics / per-cluster eval,
  not for branching the user-facing reply.
- **Reason mapping (seed authoring convention):** medication-advice,
  symptom-diagnosis, and treatment-plan requests → `health_advice` (3 rows:
  oo-001..003). Weather, news, joke, math, generic personal → `off_topic`
  (5 rows: oo-004..008).
- **System prompt revised** (plan §7, `model_client.py` will inline at
  inference): "call refuse_out_of_scope with reason='health_advice' for
  medical-advice / symptom-diagnosis / treatment-plan questions, or
  reason='off_topic' for anything outside the health domain."
- **Plan §7 tool-registry table updated** from 4 tools + "(refusal — no
  tool)" to 5 tools including `refuse_out_of_scope(reason)`.

### External evidence supporting refusal-as-tool

1. **Distil-published FunctionGemma model.**
   [`distil-labs/distil-home-assistant-functiongemma`](https://huggingface.co/distil-labs/distil-home-assistant-functiongemma)
   ships an `intent_unclear(reason)` tool with `reason ∈ {ambiguous,
   off_topic, incomplete, unsupported_device}` for exactly this case;
   model-card example: `User: "Can you order me a pizza?"` →
   `{"name": "intent_unclear", "arguments": {"reason": "off_topic"}}`.
2. **Distil CLI documentation.** `references/tasks/prepare-data/multi-turn-tool-calling.md`
   states each assistant turn "must contain **exactly one function call**".
   The platform-overview docs explicitly recommend: *"always respond with
   a tool call; if the request is invalid, call an `error` or `refuse`
   tool with a reason parameter."*
3. **FG iter-001 cost data.** Iter-001 chose no-tool-call refusals and was
   forced to exclude `medical_advice_refusal` + `off_topic_refusal` from
   Distil training (see `archive/functiongemma-pre-distil/plans/phase-d-readme-original.md`
   and `releases/functiongemma-270m/001-baseline/distil/README.md`). 36
   subsequent loss-reweighting variants failed to clear the ≥80% bar on
   the local Unsloth fallback path. Adopting refusal-as-tool here avoids
   that branch entirely.
4. **Google FG fine-tuning guide.** Best-practice recommendation: fine-tune
   on a dataset that includes both correct calls AND "ask for clarification"
   examples — also a tool-call shape.

### What this means for downstream phases

- **Phase 1.2** (tool registry): `src/gemma_tools/dispenser_demo/tools.py`
  exposes 5 tools. `refuse_out_of_scope` is the only non-domain tool; the
  dispatcher prints the canned NL on dispatch and the side-effect is null
  (no BLE write, no I/O).
- **Phase 1.4** (Distil `job_description.json`): routing rules include a
  rule mapping out-of-scope queries to `refuse_out_of_scope(reason)`. The
  judge scores tool-call equivalence; the reason enum carries the
  diagnostic signal.
- **Phase 1.5** (Distil dry-run): the no-tool-call rejection that blocked
  iter-001 from training refusals does NOT apply here — every row is a
  one-tool-call row.
- **Phase 4** (acceptance gate): per-intent accuracy includes the
  `out_of_scope_refusal` category; expected ≥ 90 % per category, same bar
  as the other four intents.

### When to reconsider

- If a Distil-side trace (1.5 or 1.6) shows the model emits
  `refuse_out_of_scope(reason)` with a wrong reason on >10 % of refusal
  rows, narrow the prompt — DO NOT collapse the enum, the diagnostic is
  the point of keeping `reason`.
- If a future intent (e.g. set-reminder, add-contact) wants to land in a
  follow-up dispenser version, add a new domain tool — do NOT overload
  `refuse_out_of_scope` reasons.

---

## 2026-05-11 (PM) — Phase 0 supersession: switch to Moonshine Tiny (non-streaming)

Supersedes the 2026-05-11 (AM) entry below. Empirical proof on the SL2619
showed the non-streaming `moonshine` backend with `cstr/moonshine-tiny-GGUF`
materially outperforms `moonshine-streaming` on the relevant axes for
batch-mode (push-to-talk / VAD-cut) voice command decoding.

- **Phase 3 STT runtime (binding, revised):** `cstr/moonshine-tiny-GGUF`
  via CrispASR's `--backend moonshine` (NOT `moonshine-streaming`).
- **Invocation flags (binding for production launcher, unchanged):**
  `-l en --no-punctuation -t 2`. For the moonshine backend
  `--no-punctuation` is honored natively via `CAP_PUNCTUATION_TOGGLE`; for
  defense-in-depth (so the same launcher works if the binding is ever
  re-flipped) the flag stays mandatory.
- **Model path on board (binding):** `/mnt/sdcard/models/moonshine-tiny/moonshine-tiny-q4_k.gguf`
  + co-located `tokenizer.bin` (sha `0e90e02b...`, identical to the streaming
  variant — both ship the same tokenizer).
- **Empirical numbers** (same `crispasr` binary, same JFK 11 s fixture, same
  flags — captured in `crispasr-spike-notes.md` §6 row "moonshine non-streaming
  variant proof"):

  | Metric | streaming-tiny (superseded) | tiny (active) |
  | --- | --- | --- |
  | GGUF q4_k size | 30.6 MB | **20.2 MB** |
  | Wall (11 s clip) | 7.48 s | **4.66 s** |
  | RT factor | 1.5× | **2.4×** |
  | Peak VmRSS | 69.5 MB | **49.6 MB** |

  Extrapolated to a 3 s command utterance: ~1.27 s wall, ~50 MB RSS. Comfortably
  inside plan §9 Phase 0 gates on both axes.

- **What changed in the code base:**
  - `scripts/dispenser_demo/spike/crispasr_host_smoke.py` — default `--backend`
    flipped to `moonshine`; help text updated.
  - `scripts/dispenser_demo/spike/crispasr_board_smoke.sh` — same.
  - `tests/dispenser_demo/test_crispasr_smoke_scripts.py` — parametrize
    expectation updated.
  - `docs/plans/dispenser-demo/crispasr-spike-notes.md` §7 — supersession
    block appended below the original decision.
  - `archive/dispenser-demo-moonshine-streaming/` — frozen recipe for the
    streaming variant preserved for the Phase-3.5 partial-hypothesis case
    (see "When to reconsider" below).
- **What did NOT change:**
  - Build profile for `crispasr-cli` (static aarch64, no OpenMP) — same.
  - Iron-Law R3, `/board_probe` pre-flight, BusyBox `/proc/uptime` timer
    convention, the auto-LID and auto-punctuation suppression flags — all
    still binding.
  - The same aarch64 `crispasr` binary handles both backends; no rebuild
    needed for the flip.
- **When to reconsider (would warrant a new dated entry, not a rewrite of
  this one):**
  - Phase 3.5 voice capture design moves to streaming-while-speaking with
    partial hypotheses sent to FunctionGemma → streaming variant's TTFT win
    becomes material; consult
    `archive/dispenser-demo-moonshine-streaming/working-recipe.md`.
  - The active moonshine path develops an accuracy regression on real
    command audio (the JFK fixture is general English, not imperative
    commands).

---

## 2026-05-11 (AM) — Phase 0: KEEP CrispASR + Moonshine Streaming Tiny GGUF (superseded the same day)

- **Phase 3 STT runtime (binding):** `cstr/moonshine-streaming-tiny-GGUF`
  via CrispASR (whisper.cpp-style C++ runtime, vendored
  `docs/references/upstream/CrispASR/`).
- **Build profile (binding):** static aarch64, no OpenMP. Configure with
  `-DCMAKE_TOOLCHAIN_FILE=<aarch64-linux-gnu> -DGGML_OPENMP=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE -DBUILD_SHARED_LIBS=OFF
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF`. Target
  `crispasr-cli` (the bare `crispasr` target produces only `libcrispasr.so`).
- **Invocation flags (binding for production launcher):** ALWAYS pass
  `-l <code>` (board is offline; auto-LID would fetch `ggml-tiny.bin`) AND
  `--no-punctuation` (board is offline; auto-punctuation would fetch
  `fireredpunc-q4_k.gguf` and add a ~3-4 s second pass).
- **Threads (binding):** `-t 2` on the SL2619 (two A55 cores; CrispASR's
  default would land here anyway, but pin it for reproducibility).
- **Measurements that justify the call** — full audit trail in
  [`crispasr-spike-notes.md`](crispasr-spike-notes.md) §6:
  - Host (WSL2 Ubuntu, x86_64): 1.10 s wall for 11 s audio = 10× RT,
    155 MB RSS, exact transcript.
  - Board (Synaptics SL2619, Cortex-A55 ×2): 7.48 s wall for 11 s audio
    = 1.5× RT, 69.5 MB RSS, exact transcript (bare ASCII, no punctuation —
    expected; downstream wordform layer in Phase 1 will normalize).
- **Gate status:** plan §9 Phase 0 gate (board: ≤2.0 s decode, ≤250 MB RSS
  for a 3 s clip) — proportional extrapolation = 2.0 s wall, 70 MB RSS.
  Latency at the line, RAM 3.5× under the line.
- **Followups carried into Phase 3.5:**
  - Production launcher MUST pass `-l en --no-punctuation -t 2`.
  - Stream partial hypotheses (moonshine-streaming is streaming-native) to
    keep perceived latency reasonable since the final decode is at the
    latency gate.
  - The unstripped ARM binary lives at `/tmp/crispasr-aarch64/build2/bin/crispasr`
    on the dev WSL host; the stripped 7.9 MB artifact is at
    `/tmp/crispasr-aarch64/crispasr` (sha256
    `5bfedc148a665c56fe7a18fff857dfb4d9c8640695effaa30304e16bbb3304f8`)
    and is staged on board at `/mnt/sdcard/bin/crispasr`. Future deploys
    should re-run the cross-build rather than checking the binary into git.
- **Negated alternative:** Moonshine Tiny float ONNX via onnxruntime
  (`docs/references/sl2619-moonshine.md`, Phase A 2026-04-23) — still
  documented as a fallback per plan §9, but not selected. CrispASR's
  smaller RAM footprint (70 MB vs 180 MB for ONNX) and streaming-native
  decoder tip the balance.

> **Note (added 2026-05-11 PM):** this entry is superseded by the 2026-05-11
> (PM) entry above. The streaming-tiny pin lasted only a few hours before
> the moonshine-tiny proof flipped the binding. The full streaming recipe
> survives in `archive/dispenser-demo-moonshine-streaming/`.
