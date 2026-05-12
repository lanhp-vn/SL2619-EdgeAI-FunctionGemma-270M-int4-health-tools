# CLAUDE.md — gemma3-270M-finetune

Claude Code instructions specific to this repository. The human-facing entry
point is [`README.md`](README.md); this file is the agent's self-reference.

## Repository purpose

Two active focuses sharing one board (Synaptics SL2619, Cortex-A55 ×2,
1.87 GiB RAM, ARMv8.2-A NEON+DOTPROD, no NPU/Vulkan):

1. **FunctionGemma 270M-IT** — closed-world function-calling over a synthetic
   patient registry. Iteration 001 (Distil Labs SFT) shipped at
   `releases/functiongemma-270m/001-baseline/`; the 2026-05-02 quantization
   sweep selected **Q4_0** as the on-board variant. No retrain planned.
2. **Dispenser demo** (active, Phase 0 closed 2026-05-11; Phase 3 Layer B + Layer C both closed 2026-05-12) — voice-driven medication dispenser stacking **openWakeWord (pretrained `hey_jarvis_v0.1` ONNX) + Silero VAD + CrispASR + Moonshine Tiny GGUF** (non-streaming, `--backend moonshine`) for the wake→VAD→STT front-end on top of the FunctionGemma tool-call brain, with BLE GATT to an ESP32-controlled dispenser. The v1 demo runs on **iter-001 + a dispatcher-hijack wrapper** (`scripts/functiongemma/deploy/chat_board_dispense.py`) that short-circuits `get_medications_at_time` / `get_medication_by_name` to the §6 dispense intent (mock BLE notify `5A A5 01 00` + canned response). Iter-002 (`releases/functiongemma-270m/002-dispenser-demo/`) is trained but **NOT deployed** for v1 (host Q4_0 host-eval collapsed to 30 %; on-board output omits the `<start_function_call>` opener); Phase 1 (Distil retrain to land iter-002) is **DEFERRED** until that wire-format quirk is reconciled. Phase 3 (voice stack integration) layering: **Layer A skipped** (WSL2 doesn't expose a mic via WSLg in this user's setup; documented for future sessions on a machine with a working mic). **Layer B closed 2026-05-12** — board wake→STT smoke. **Layer C closed 2026-05-12** — full pipeline (long-running `scripts/dispenser_demo/deploy/dispenser_voice.py` wires Layer-B wake/STT into `chat_board_dispense.py`); first-run wall ~10.7 s/turn end-to-end. Carry-over Layer C.1: arecord overrun during the LLM turn (non-blocking, ~10-line stdout-drain fix). **Layer D pending** — espeak-ng → aplay → P10S speaker (promotes plan §1 v2 non-goal to v1 demo gate; code does not exist yet). Custom "Hey Sago" training is deferred. Plan + decisions: [`docs/plans/dispenser-demo/{plan.md,decisions-log.md}`](docs/plans/dispenser-demo/).

The original Gemma 3 270M-IT health-QA SFT track is frozen as a reference
under `archive/gemma3-270m-health-qa/`; live code at `src/gemma_tools/_legacy/`,
`tests/_legacy/`, `data/_legacy/` keeps its tests green in CI. Do NOT mix new
work into that track.

## High-level flow

```mermaid
flowchart TB
    subgraph FG[FunctionGemma SFT and quantize]
        SEEDS[data/functiongemma/seed_conversations.jsonl]
        PHI[scripts/pre_commit_phi_scanner.py]
        SPLITS[data/functiongemma/dataset_v1/]
        DISTIL[Distil Labs SFT]
        REL[releases/functiongemma-270m/001-baseline/]
        GGUF[finetuned_functiongemma_q4_0.gguf]
        SEEDS --> PHI --> SPLITS --> DISTIL --> REL
        REL -- llama-quantize Q4_0 --> GGUF
    end
    subgraph DD[Dispenser voice pipeline - Phase 3 target]
        MIC[P10S USB mic]
        WAKE[openWakeWord hey_jarvis_v0.1 ONNX]
        VAD[Silero VAD ONNX]
        STT[crispasr-cli + moonshine-tiny GGUF q4_k]
        BRAIN[chat_board_dispense.py - iter-001 hijack]
        BLE[BLE GATT - pybleno - planned]
        ESP[ESP32 dispenser - planned]
        MIC --> WAKE --> VAD --> STT --> BRAIN --> BLE --> ESP
    end
    GGUF -- scp --> BRAIN
```

## Key paths

| Path | Role |
| --- | --- |
| `src/gemma_tools/health_table.py` | Patient-record Pydantic loader (dual-use: legacy bench + FunctionGemma tools) |
| `src/gemma_tools/functiongemma/` | Active sub-package — `dataset.py`, `tools.py` |
| `src/gemma_tools/_legacy/` | Frozen Gemma 3 270M health-QA modules (tests under `tests/_legacy/` still run) |
| `scripts/functiongemma/chat.py` | Host interactive REPL (the local FunctionGemma demo) |
| `scripts/functiongemma/{bench,smoke}.py` | Bench harness (local + remote SL2619) and smoke runner |
| `scripts/functiongemma/data/` | Dataset prep — build_seeds, build_splits, ingest, quality_audit, gen_prompt_templates |
| `scripts/functiongemma/quantize/build_variants.sh` | Idempotent host `llama-quantize` driver |
| `scripts/functiongemma/bench/aggregate_quant.py` | Sweep JSONL → Markdown aggregator |
| `scripts/functiongemma/train/finetune_local.py` | Unsloth-based local SFT fallback |
| `scripts/functiongemma/eval/eval_holdout.py` | Holdout evaluation; HF (`--checkpoint`) + GGUF (`--gguf`) seams |
| `scripts/functiongemma/deploy/` | Board deploy: `chat_board.py`, `chat_board_dispense.py` (v1 dispense-intent wrapper), `ask_board.sh`, `run_prompt.sh` |
| `scripts/dispenser_demo/spike/crispasr_host_smoke.py` | Phase 0 host-side CrispASR smoke (Python, uv-run) |
| `scripts/dispenser_demo/spike/crispasr_board_smoke.sh` | Phase 0 board-side dispatcher (SSH read-only pre-flight + decode + VmRSS poll) |
| `scripts/dispenser_demo/{data,eval,deploy}/` | Iter-002 substrate — dataset build, holdout eval, `ble_test.py`. NOT the v1 demo path (v1 = iter-001 hijack via `scripts/functiongemma/deploy/chat_board_dispense.py`) |
| `scripts/dispenser_demo/deploy/dispenser_voice.py` | Phase 3 Layer C long-running entry — voice→wake→VAD→STT→FunctionGemma→stdout, iter-001 hijack (imports `chat_board_dispense.py`, primes FG KV cache, sets ALSA mixer, speech-relative LISTENING cap, arecord SIGTERM stderr suppression) |
| `scripts/dispenser_demo/chat.py` | Iter-002 host REPL (not the v1 demo path) |
| `scripts/setup/server-bootstrap.sh` | Idempotent Ubuntu-server SFT-stack bootstrap (RTX 5080) |
| `scripts/sl2619/p10s_aec_probe.py` | P10S firmware AEC tone-suppression probe (duplex; verdict via Goertzel) |
| `scripts/sl2619/p10s_aec_speech_probe.py` | P10S speech-survival follow-up — operator speaks during duplex |
| `scripts/pre_commit_phi_scanner.py` | PHI scanner for FunctionGemma data ingest |
| `tests/dispenser_demo/test_crispasr_smoke_scripts.py` | 23 cases gating the Phase 0 smoke scripts |
| `data/health_table_v1.yaml` | Synthetic patient record (no real PHI) |
| `data/health_table_v1_dispense_demo.yaml` | Dispenser-demo patient (David Smith) — paired with `chat_board_dispense.py`; renders to `health_table_dispense.json` on board so `health_table.json` (v1 fixture) is preserved |
| `data/health_table_v2.yaml` | Iter-002 patient fixture (v2 schema) |
| `data/dispenser_demo/{dataset_v1,seed_conversations.jsonl}` | Iter-002 training substrate (NOT used by the v1 demo) |
| `data/functiongemma/dataset_v1/{train,val,test}.jsonl` | Active Distil iter-001 training splits |
| `data/functiongemma/seed_conversations.jsonl` | 50-row hand-authored seeds |
| `data/functiongemma/eval_holdout_v{1,2_clean,2_contaminated}.jsonl` | Active eval holdouts |
| `data/_legacy/` | Frozen gemma3-270m SFT corpora and `prompts.yaml` |
| `releases/functiongemma-270m/001-baseline/` | Iter-001 deployable: `merged/`, `adapter/`, `gguf/`, `distil/`, `Modelfile`, `model_client.py`, `RECIPE.md`. **Source of the v1 dispenser-demo runtime** (via `chat_board_dispense.py`) |
| `releases/functiongemma-270m/001-baseline/gguf/` | `CHECKSUMS.txt` (committed), `RECOMMENDED.md`, `Modelfile`; FP16 + Q4_0 GGUF gitignored |
| `releases/functiongemma-270m/002-dispenser-demo/` | Iter-002 deliverable: `merged/`, `adapter/`, `gguf/`, `distil/`, `Modelfile`, `model_client.py`, `DISTIL_README.md`. **Trained but NOT deployed for v1** — two open quirks (host Q4_0 collapse, on-board missing `<start_function_call>` opener) blocked promotion; see `docs/plans/dispenser-demo/decisions-log.md` 2026-05-12 entries |
| `bench/functiongemma/runs/2026-05-02-quant/` | Quant sweep JSONL outputs |
| `docs/plans/functiongemma/` | FunctionGemma plan docs — recipe, decisions-log, quantization-plan, seed-authoring-recipe, llm-augmentation-prompt, upstream-issue-drafts |
| `docs/plans/dispenser-demo/` | Dispenser demo — `plan.md`, `crispasr-spike-notes.md`, `decisions-log.md` (Phase 0 closed 2026-05-11; 2026-05-12 entries cover the iter-001 hijack pivot, pretrained Hey Jarvis wake word, WSL-first Layer A/B/C/D smoke topology, Layer B board wake/STT close, and Layer C full-pipeline close with per-stage wall-clock + arecord-overrun carry-over) |
| `docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md` | Single canonical quant sweep report |
| `docs/deployment/{sl2619-board,functiongemma-board-deploy}.md` | Board cross-compile + FunctionGemma deploy runbooks |
| `docs/guides/usb-audio-testing-sl2619.md` | USB speaker + mic + P10S firmware AEC probe recipe |
| `docs/conventions/` | Normative coding/repo/workflow rules (Python, shell, testing, doc-update) |
| `docs/references/upstream/` | Opt-in shallow submodules: `gemma`, `llama.cpp`, `CrispASR`, `openWakeWord`, `silero-vad`, `distil-cli-skill`, `synaptic-sl2619`, `unsloth-notebooks` (nested clone) |
| `docs/tmp/` | Local-only `/board_probe` snapshots (gitignored) |
| `archive/{gemma3-270m-health-qa,functiongemma-pre-distil,dispenser-demo-moonshine-streaming}/` | Frozen historical tracks (last entry = superseded streaming-STT recipe) |

## Workflows

### Install (host dev)

```bash
cd /home/lanhp-wsl/nouslogic/gemma3-270M-finetune
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,functiongemma]"
```

### Tests, lint, typecheck

```bash
uv run pytest                            # 568 tests (active + dispenser_demo + _legacy)
uv run ruff check src tests scripts/functiongemma
uv run mypy src
```

### Quantize from canonical FP16

```bash
# One-time host build of llama-quantize
cd docs/references/upstream/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-quantize -j$(nproc)
cd ../../../..

# Default Q4_0 only; --all for the sweep (Q4_0, Q4_K_M, Q5_K_M, Q8_0, IQ4_XS)
scripts/functiongemma/quantize/build_variants.sh
```

### Run the local FunctionGemma demo

```bash
uv run python scripts/functiongemma/chat.py
```

Defaults to `releases/functiongemma-270m/001-baseline/{gguf/finetuned_functiongemma_fp16.gguf, merged/}`
and `data/health_table_v1.yaml`. Override with `--model finetuned_functiongemma_q4_0.gguf` to
chat against the on-board quant from host.

### Bench

```bash
uv run python scripts/functiongemma/bench.py --mode local --warmup 1
uv run python scripts/functiongemma/bench.py --mode remote \
    --ssh-host nouslogic-sl2619 \
    --remote-binary /mnt/sdcard/llama-cpp/llama-completion \
    --remote-model  /mnt/sdcard/models/functiongemma-270m/finetuned_functiongemma_q4_0.gguf
```

Output lands in `bench/functiongemma/runs/`. Always pass `--remote-model` explicitly.

### Holdout eval (HF or GGUF seam)

```bash
# HF seam (server-side; torch + transformers)
uv run python scripts/functiongemma/eval/eval_holdout.py \
    --checkpoint releases/functiongemma-270m/001-baseline/merged \
    --holdout data/functiongemma/eval_holdout_v2_clean.jsonl

# GGUF seam (host CPU via llama-cpp-python; 5–10× faster than on-board eval)
uv run python scripts/functiongemma/eval/eval_holdout.py \
    --gguf releases/functiongemma-270m/001-baseline/gguf/finetuned_functiongemma_q4_0.gguf \
    --tokenizer-dir releases/functiongemma-270m/001-baseline/merged \
    --holdout data/functiongemma/eval_holdout_v2_clean.jsonl
```

### CrispASR Phase 0 smoke (host then board)

```bash
# Host-side build + decode
uv run python scripts/dispenser_demo/spike/crispasr_host_smoke.py

# Board-side dispatcher (read-only pre-flight, decode, VmRSS poll)
bash scripts/dispenser_demo/spike/crispasr_board_smoke.sh
```

Full audit trail (transcripts, RSS, gate verdict) is in
`docs/plans/dispenser-demo/crispasr-spike-notes.md`. Build profile + production
launcher flags are pinned in `docs/plans/dispenser-demo/decisions-log.md`.

### Deploy FunctionGemma to board

```mermaid
flowchart TB
    H[Host: gen_prompt_templates.py + YAML to JSON]
    S[Stage at /tmp/fg_deploy/]
    SCP[scp bundle - one-time]
    B[/mnt/sdcard/models/functiongemma-270m/]
    R[chat_board.py interactive REPL]
    H --> S --> SCP --> B --> R
```

Recipe: [`docs/deployment/functiongemma-board-deploy.md`](docs/deployment/functiongemma-board-deploy.md).
First turn primes `/tmp/fg_pc_<model>.bin` (~32 s, one-time). Subsequent turns:
~6 s wall, 10.3 tok/s decode. The cache is per-model; switching quants = priming
a new cache. Clear with `rm /tmp/fg_pc_*.bin` if the prefix changes.

## Discipline

- **No model weights in git** — `*.gguf`, `*.bin`, `*.safetensors`, `*.pt` are gitignored. `releases/.../gguf/CHECKSUMS.txt` is the authoritative SHA record.
- **SSH to board is read-only** — deployment commands are emitted for the user; the agent never mutates the board (R3 in `.claude/CLAUDE.local.md`).
- **CrispASR runtime traps** — any code invoking the board's crispasr binary MUST pass `-l en --no-punctuation -t 2`. Auto-LID (without `-l <code>`) silently fetches `ggml-tiny.bin` (~77 MB) from HF at runtime; auto-punctuation (without `--no-punctuation`) fetches `fireredpunc-q4_k.gguf` (~80 MB) and adds a ~3–4 s second pass on A55. Both are fatal on the offline SL2619. See `docs/plans/dispenser-demo/decisions-log.md`.
- **BusyBox quirks on SL2619** — board's `date` emits literal `%N` instead of nanoseconds; use `awk '{print $1; exit}' /proc/uptime` for sub-second timing. `head -20` fails (use `head -n 20`); no `grep -P` (use `-E`).
- **PHI scanner gates data ingest** — `scripts/pre_commit_phi_scanner.py` scans every staged JSONL. Patient YAML stays synthetic; OQ-5 covers any switch to real PHI.
- **Tests before any data pipeline change** — `uv run pytest` must be green before editing `gemma_tools.functiongemma.dataset`, `gemma_tools.functiongemma.tools`, or `health_table.py`.
- **Submodules are opt-in** — `docs/references/upstream/{gemma,llama.cpp,CrispASR,openWakeWord,silero-vad,distil-cli-skill,synaptic-sl2619}` are shallow submodules with `update = none`. Initialize on demand: `git submodule update --init docs/references/upstream/<name>`. `unsloth-notebooks/` is a standalone shallow clone (not a submodule), with sparse-checkout. openWakeWord + silero-vad models (TFLite/ONNX, ~3 MB each) are fetched from upstream GitHub releases on first use, NOT committed.
- **Archive is read-only** — `archive/` and `data/_legacy/` are frozen reference. New work goes under active dirs (`docs/`, `src/`, `scripts/`, `tests/`, `data/`, `releases/`, `bench/`).
- **`docs/conventions/` is normative** — agent edits go through normal PRs with review.
- **`docs/tmp/` is gitignored** — `/board_probe` snapshots contain board IPs / server usernames.

## Pinned runtime variants

- **FunctionGemma on board: Q4_0 only.** The 2026-05-02 sweep tested Q4_0, Q4_K_M, Q5_K_M, Q8_0, IQ4_XS. Only Q4_0 preserves the FunctionGemma wire format on the board's `b8925`/`0adede8` runtime (K-quant scale-factor encoding skew with the older runtime drops `<start_function_call>`). Repo ships only Q4_0 + FP16 source on disk. Rationale: [`releases/functiongemma-270m/001-baseline/gguf/RECOMMENDED.md`](releases/functiongemma-270m/001-baseline/gguf/RECOMMENDED.md).
- **Dispenser-demo v1 runtime: iter-001 Q4_0 + dispatcher-hijack.** Entry point is [`scripts/functiongemma/deploy/chat_board_dispense.py`](scripts/functiongemma/deploy/chat_board_dispense.py); it imports `chat_board.py` and monkey-patches `dispatch` + `format_response` so `get_medications_at_time` / `get_medication_by_name` route to the §6 dispense intent (mock BLE notify + verbatim canned response). Tool schema unchanged → the warm prompt cache `/tmp/fg_pc_finetuned_functiongemma_q4_0.gguf.bin` reuses across `chat_board.py` and the wrapper. Iter-002 (`releases/functiongemma-270m/002-dispenser-demo/`) is the eventual target shape (named `dispense_medication()` tool) but is **NOT deployed** for v1 — see `docs/plans/dispenser-demo/decisions-log.md` 2026-05-12 entries for the host Q4_0 collapse and on-board `<start_function_call>` omission that gate promotion.
- **Wake word: pretrained openWakeWord `hey_jarvis_v0.1` ONNX (from upstream v0.5.1 release).** Inference framework is ONNX (Silero VAD is ONNX-native — single runtime, no TFLite dependency). Custom "Hey Sago" training is deferred past v1; plan §11 R4 / O4 retired for the v1 demo. Models fetched at first use from openWakeWord GitHub releases, not committed.
- **STT on board: CrispASR + Moonshine Tiny GGUF (q4_k, non-streaming).** Pinned model: `cstr/moonshine-tiny-GGUF` → `moonshine-tiny-q4_k.gguf` (~20.2 MB) at `/mnt/sdcard/models/moonshine-tiny/`, invoked via CrispASR `--backend moonshine` (NOT `moonshine-streaming` — that variant was provisionally pinned 2026-05-11 (AM) and superseded the same afternoon after a head-to-head: −38 % wall, −29 % RSS, −34 % model size). Build profile: static aarch64, no OpenMP, `crispasr-cli` target (NOT bare `crispasr`, which builds only `libcrispasr.so`). Phase 0 (2026-05-11) measured: host (WSL2, x86_64) 1.10 s wall / 155 MB RSS on an 11 s clip; board (SL2619, 2× A55) **4.66 s wall (2.4× realtime) / 49.6 MB RSS** — extrapolated 3 s utterance ≈ 1.27 s wall, ≈ 50 MB RSS, comfortably meeting `plan.md` §9 Phase 0 gate. Production launcher flags (binding): `-l en --no-punctuation -t 2` — `--no-punctuation` remains mandatory for defense-in-depth even though the non-streaming backend honors it natively (`CAP_PUNCTUATION_TOGGLE`), so the same launcher survives a future re-flip to a backend without that cap. Full flag list, rationale, and reproduction in [`docs/plans/dispenser-demo/decisions-log.md`](docs/plans/dispenser-demo/decisions-log.md) and [`docs/plans/dispenser-demo/crispasr-spike-notes.md`](docs/plans/dispenser-demo/crispasr-spike-notes.md). Frozen streaming-variant recipe at [`archive/dispenser-demo-moonshine-streaming/working-recipe.md`](archive/dispenser-demo-moonshine-streaming/working-recipe.md).

## Pointers

- `docs/conventions/doc-update.md` — DRY canonical-ownership registry.
- `docs/conventions/{code-style-python,code-style-shell,testing}.md` — normative rules.
- `docs/plans/functiongemma/{recipe,decisions-log,quantization-plan}.md` — FunctionGemma working recipe + decisions.
- `docs/plans/dispenser-demo/plan.md` — full dispenser demo plan (phases, BLE wire contract, state machine, Phase 3 Layer A/B/C/D smoke topology).
- `docs/plans/dispenser-demo/crispasr-spike-notes.md` — Phase 0 STT spike audit trail.
- `docs/plans/dispenser-demo/decisions-log.md` — binding decisions (Phase 0 STT runtime/flags/build profile + 2026-05-12 entries: iter-001 hijack pivot, pretrained Hey Jarvis wake word, WSL-first smoke topology, Layer B board wake/STT close, Layer C full-pipeline close with per-stage wall-clock + arecord-overrun carry-over).
- `docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md` — single canonical sweep report.
- `docs/guides/usb-audio-testing-sl2619.md` — USB speaker + mic + P10S firmware AEC probe (verified 2026-05-11: firmware AEC handles echo cancellation, no software AEC needed for duplex voice pipeline).
- `releases/functiongemma-270m/001-baseline/{RECIPE.md,distil/README.md,gguf/RECOMMENDED.md}` — iter-001 reproduce + Distil timeline + Q4_0 rationale.
- `archive/README.md` — archive index.

Last refreshed: 2026-05-12
