# Run the Synaptics `torq-examples/gemma3` demo on the SL2619 — board instructions

> **DEFERRED — 2026-04-25.** Phase 1.5 Phase D uses A55 CPU (llama.cpp Q4_0 GGUF) as the primary path for Gemma 3 deployment. The Torq NPU path described below was quality-gated out (G_QUALITY 1.2/3 avg; bench record: `docs/tmp/bench/2026-04-24_gemma3-summary.md`). This document is preserved as a historical investigation record. For current instructions, see `docs/get-started/gemma-on-a55-get-started.md`.

> Goal: get the **official Synaptics Gemma 3 270M-IT chat REPL** (`torq-examples/gemma3/src/infer.py`) running on the SL2619 NPU, **strictly following the upstream `torq-examples/README.md`** — with three small, explicitly-flagged deviations forced by the SL2619's stripped Yocto Python.
>
> Source of truth: [`references/Synaptics/torq-examples/README.md`](../../references/Synaptics/torq-examples/README.md) (pinned at branch `torq-runtime-v1.5`).
>
> Pre-flight done: `/board_probe` 2026-04-25 02:55 UTC — see `docs/tmp/sl2619-status.md`. NPU `syna_npu` loaded, `torq.runtime` + `iree.runtime` importable, HF reachable, 109 G free on `/mnt/sdcard`.

## 0. Scope and ground rules

- **All work below is user-performed in your WSL terminal.** Per R3 the agent never `scp`/`mv`/`rm`/`pip install` over SSH; it just emits the commands.
- **Old work is ignored.** Anything previously placed at `/mnt/sdcard/models/gemma-3-270m-it/` (Phase B leftovers) is not used. Nothing here mutates it.
- **Purpose-named on-board paths.** All new artifacts live under `/mnt/sdcard/torq-examples/` (the upstream layout), not phase-prefixed.
- **Not host-testable.** `torq_runtime-1.5.0` is a `cp312-cp312-manylinux_2_28_aarch64` wheel — it does not import on x86_64 WSL. The substitute for "host-test before deploy" (per your standing rule) is the on-board smoke step §6.1, which prints clear pass/fail in seconds.

## 1. Required SL2619 deviations from upstream README (stated upfront)

| Upstream step | SL2619 reality | This guide does |
|---|---|---|
| `python3 -m venv .venv` | Stock Yocto image strips `python3-venv` (CLAUDE.md §7 G_PY note). `venv` will fail. | Reuse the **existing** PYTHONPATH-based env at `/tmp/p15site` + `/tmp/pipbase` (Phase A infrastructure). It already contains the torq_runtime wheel installed exactly as the README would. |
| `pip install <torq_runtime wheel URL>` | Already installed in `/tmp/p15site` (sha-stamped, version-locked, working `import torq.runtime`). | Skip. Verify with one-line import test (§4.1). |
| `pip install -r requirements.txt` | All deps importable except `requests`. `pip` itself is currently broken in `/tmp/pipbase` (post-tmpfs-eviction half-state). | (a) Repair pip from the SD-card mirror (one `cp`). (b) Then run the install — only `requests` will actually be fetched. |

Everything else (`setup_demos.py gemma3`, `cd gemma3`, `pip install -r requirements.txt`, `python src/infer.py …`, `python profile.py …`) is run **verbatim** from the upstream README.

## 2. Phase H1 — host pre-flight (WSL, this machine)

### 2.1 Verify the pinned vendor copy is intact

```bash
cd ~/nouslogic/SynapticSL2619
ls references/Synaptics/torq-examples/{README.md,setup_demos.py,profile.py,requirements.txt,gemma3/setup.py,gemma3/src/infer.py,gemma3/src/runner.py,gemma3/requirements.txt,utils/cache_runner.py,utils/download.py,utils/deps.py,utils/log.py,utils/errors.py}
```
Expected: every path resolves (no "No such file"). The pin is `torq-runtime-v1.5` per `references/Synaptics/torq-examples/CLAUDE.md`.

### 2.2 Stage a clean copy under `/tmp` on the host (so we don't ship `.git`)

```bash
rm -rf /tmp/torq-examples-stage
cp -r references/Synaptics/torq-examples /tmp/torq-examples-stage
find /tmp/torq-examples-stage -name '__pycache__' -exec rm -rf {} + 2>/dev/null
ls /tmp/torq-examples-stage
```
Expected output includes: `LICENSE  README.md  gemma3  profile.py  requirements.txt  setup_demos.py  utils`.

## 3. Phase H2 — push the demo tree onto the board

```bash
scp -r /tmp/torq-examples-stage nouslogic-sl2619:/mnt/sdcard/torq-examples
```
Expected: ~70 small files copied, < 5 s on Wi-Fi. Verify the landing:

```bash
ssh nouslogic-sl2619 'ls /mnt/sdcard/torq-examples'
```
Expected: `LICENSE  README.md  gemma3  profile.py  requirements.txt  setup_demos.py  utils`.

## 4. Phase B1 — env activation on board (READ-ONLY check first, then deviation §1 row 1)

### 4.1 Verify the existing env is healthy (no mutation)

```bash
ssh nouslogic-sl2619 '. /mnt/sdcard/p15-env.sh && python3 -c "import torq.runtime, iree.runtime, numpy, tokenizers, ml_dtypes, huggingface_hub, tqdm; print(\"deps ok:\", numpy.__version__, tokenizers.__version__)"'
```
Expected output: `deps ok: 1.26.4 0.22.2`.
If you see `ModuleNotFoundError`, the tmpfs env was wiped — re-run the Phase A bootstrap (`docs/plans/phase-a-board-instructions.md`) before continuing.

### 4.2 Confirm `requests` is the only gap

```bash
ssh nouslogic-sl2619 '. /mnt/sdcard/p15-env.sh && python3 -c "import requests" 2>&1 | tail -1'
```
Expected: `ModuleNotFoundError: No module named '\''requests'\''` (this is the gap §1 row 3 fixes next).

## 5. Phase B2 — repair pip + install missing deps (one-time)

### 5.1 Restore pip from the SD-card mirror (deviation §1 row 3a)

`/tmp/pipbase/bin/pip` exists but `pip._internal` is missing — a partial post-reboot rehydration. The persistent mirror at `/mnt/sdcard/pipbase/` is intact; copy the missing tree back into tmpfs.

**You run this in your shell:**
```bash
ssh nouslogic-sl2619 'cp -rn /mnt/sdcard/pipbase/lib /tmp/pipbase/ && /tmp/pipbase/bin/pip --version'
```
Expected: `pip 26.0.1 from /tmp/pipbase/lib/python3.12/site-packages/pip (python 3.12)` (or whatever pip version was bootstrapped — the point is it doesn't traceback).

> If this fails (e.g. pip mirror is also broken on the SD card), fall back to a manual `requests` wheel install: `ssh nouslogic-sl2619 'curl -L -o /tmp/requests.whl https://files.pythonhosted.org/packages/26/.../requests-2.32.3-py3-none-any.whl && unzip -o /tmp/requests.whl -d /tmp/p15site/'` (download URL changes — fetch from `https://pypi.org/project/requests/#files`). Skip step 5.2.

### 5.2 Install the upstream top-level requirements (deviation §1 row 3b)

This is the **literal upstream README step 2**, just routed to our existing `--target` instead of a venv site-packages:

```bash
ssh nouslogic-sl2619 '. /mnt/sdcard/p15-env.sh && /tmp/pipbase/bin/pip install --target=/tmp/p15site -r /mnt/sdcard/torq-examples/requirements.txt'
```
Expected: `huggingface_hub`, `ml_dtypes`, `numpy<2.0.0b`, `tqdm` reported as already satisfied (they are); only `requests` (and its tiny deps `idna`, `charset-normalizer`, `urllib3`, `certifi` — last two also already present) gets fetched. ~50 KB of actual download.

### 5.3 Install the gemma3-specific demo requirement

Upstream README step "cd gemma3 && pip install -r requirements.txt" — the only line in `gemma3/requirements.txt` is `tokenizers`, already present. Run anyway for parity:

```bash
ssh nouslogic-sl2619 '. /mnt/sdcard/p15-env.sh && /tmp/pipbase/bin/pip install --target=/tmp/p15site -r /mnt/sdcard/torq-examples/gemma3/requirements.txt'
```
Expected: `Requirement already satisfied: tokenizers`.

## 6. Phase B3 — download the Synaptics-published VMFB (the upstream `setup_demos.py` flow)

The `Synaptics/gemma-3-270m-it` HF repo is **public** (no `HF_TOKEN` needed). The VMFB is **540 MiB**; with `huggingface.co` reachable from the board (HTTP 200 confirmed in the snapshot), download takes ~30–60 s on the 5 GHz link.

```bash
ssh nouslogic-sl2619 'cd /mnt/sdcard/torq-examples && . /mnt/sdcard/p15-env.sh && python3 setup_demos.py gemma3'
```
Expected logged lines:
- `setup INFO: Added '/mnt/sdcard/torq-examples' to Python's import path. To undo, delete '/tmp/pipbase/lib/python3.12/site-packages/torq-examples.pth'`
- `Gemma3.setup INFO: Setting up gemma3 demo with models: [instruct]`
- (silent download of `model.vmfb` 540 MB, `token_embeddings.npy` 320 MB, `config.json`, `tokenizer.json`)
- `Gemma3.setup INFO: Downloaded gemma3 model files from Synaptics/gemma-3-270m-it`
- `Gemma3.setup INFO: gemma3 setup complete.`

Verify on disk:
```bash
ssh nouslogic-sl2619 'ls -lh /mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/'
```
Expected: `model.vmfb` (516M), `token_embeddings.npy` (320M), `tokenizer.json` (32M), `config.json` (≤ 2K).

### 6.1 Smoke check — proves the NPU actually dispatches before you sit at the REPL

```bash
ssh nouslogic-sl2619 'cd /mnt/sdcard/torq-examples/gemma3 && . /mnt/sdcard/p15-env.sh && echo "Say hi in one short sentence." | python3 src/infer.py -m ../models/Synaptics/gemma-3-270m-it/model.vmfb --instruct-model 2>&1 | tail -n 8'
```
Expected: a short greeting like `Agent: Hi! How can I help you today?` followed by a `(NNN ms, TTFT: NNN ms, X.X tok/s)` metrics line. If the line shows `0.0 tok/s` or hangs forever, jump to §10.

## 7. Phase B4 — run the official chat REPL (the actual demo)

This is the **verbatim upstream command from `torq-examples/README.md` line 58**:

```bash
ssh -t nouslogic-sl2619 'cd /mnt/sdcard/torq-examples/gemma3 && . /mnt/sdcard/p15-env.sh && python3 src/infer.py -m ../models/Synaptics/gemma-3-270m-it/model.vmfb --instruct-model'
```

(`-t` allocates a TTY so the `input(...)` prompt works.)

Type prompts at `You (type 'exit' or 'quit' to stop):`. Each turn prints `[thinking…]` while the NPU prefills, then streams tokens, then ends with the timing line `(NNN ms, TTFT: NNN ms, X.X tok/s)`. Quit with `exit`, `quit`, or `Ctrl-D`.

### Useful flags (per `python3 src/infer.py -h`)
- `--max-seq-len N` — caps prompt+generation total. Auto-detected from VMFB metadata (~256 for the published 270M build).
- `--max-inp-len N` — caps the user input portion.
- `-j N` — pin CPU threads. Board has 2× A55 enumerated → `-j 2` is the natural max.
- `--temperature 0.7 --top-p 0.95 --top-k 64` — sample instead of greedy.

## 8. Phase B5 — profile (upstream `profile.py`)

```bash
ssh nouslogic-sl2619 'cd /mnt/sdcard/torq-examples && . /mnt/sdcard/p15-env.sh && python3 profile.py models/Synaptics/gemma-3-270m-it/model.vmfb -r 5'
```
Expected: 5 NPU iterations, then `Avg infer time for ... (5 iters): NN.NNN ms`. This is per-step inference latency (one decode step, not full generation).

## 9. Quick reference — file/path map after the guide is run

```
HOST  ~/nouslogic/SynapticSL2619/references/Synaptics/torq-examples/   (read-only pin, untouched)
HOST  /tmp/torq-examples-stage/                                         (transient stage; can rm after step 3)
BOARD /mnt/sdcard/torq-examples/                                        (the demo tree, persistent)
BOARD /mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/       (VMFB + tokenizer, ~870 MiB, persistent)
BOARD /tmp/pipbase/lib/python3.12/site-packages/torq-examples.pth       (the .pth setup_demos installed; volatile)
BOARD /tmp/p15site/                                                     (existing PYTHONPATH env; volatile)
BOARD /mnt/sdcard/p15-env.sh                                            (env activator, persistent)
BOARD /mnt/sdcard/pipbase/                                              (pip mirror, persistent)
```

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `setup_demos.py` errors `Missing packages: requests` | Step 5.2 was skipped or failed silently. | Re-run §5.1 + §5.2; verify with `python3 -c "import requests; print(requests.__version__)"`. |
| `setup_demos.py` errors `Repository Not Found` or 401 | HF rate-limit hit, or transient network. | Retry. If repeated, check `curl -I https://huggingface.co`. The Synaptics repo is not gated, so 401 means wrong URL — verify `gemma3/setup.py` `_HF_REPO_MAP`. |
| `infer.py` ImportError on `runner` or `utils.cache_runner` | `setup_demos.py` was not run first → no `.pth` file. | Run §6 once. The `.pth` makes `/mnt/sdcard/torq-examples` importable so `from utils.cache_runner import …` resolves. |
| REPL prints `[thinking…]` forever (no tokens) | NPU not dispatching. | `ssh nouslogic-sl2619 'lsmod \| grep syna_npu; ls /sys/devices/platform/soc/f7600000.synpu/'` — both must succeed. If not, reboot the board and re-run `/board_probe`. |
| `MemoryError` / OOM kill mid-load | 540 MB VMFB + 320 MB embeddings + KV cache exceeds free RAM under contention. | `ssh nouslogic-sl2619 'free -h'` — need ~1 GiB MemAvailable. Stop other services or reboot. |
| `pip install` says `Operation not permitted` writing to `/tmp/p15site` | Tmpfs full or permission squash. | `ssh nouslogic-sl2619 'df -h /tmp; ls -ld /tmp/p15site'`. /tmp is 958 M; if near-full, reboot. |
| `python3 -m venv .venv` (if you tried strict-strict upstream) | `python3-venv` Yocto subpackage not installed (CLAUDE.md §7 G_PY). | Don't. Use the existing PYTHONPATH env per §1 row 1. |
| `ssh -t` complains `Pseudo-terminal will not be allocated` | Wrapping `ssh -t` in `bash -c` or scripting it. | Run §7 directly from your interactive terminal (not from inside another script). |

## 11. Cleanup (optional — only if you want to reclaim space)

The demo dir + model assets occupy ~870 MiB on the SD card. To remove:
```bash
ssh nouslogic-sl2619 'rm -rf /mnt/sdcard/torq-examples'
```
The `.pth` in tmpfs vanishes on reboot automatically; if you want to remove it sooner:
```bash
ssh nouslogic-sl2619 'rm -f /tmp/pipbase/lib/python3.12/site-packages/torq-examples.pth'
```

## 12. Provenance

- Upstream README: `references/Synaptics/torq-examples/README.md` lines 12–69.
- Upstream setup driver: `references/Synaptics/torq-examples/setup_demos.py` (`.pth` install + `gemma3.setup.setup_gemma3`).
- Upstream HF download helper: `references/Synaptics/torq-examples/utils/download.py` (`hf_hub_download` + `requests` import at module top — the only reason §5.2 is non-optional).
- Upstream model bundle: HF `Synaptics/gemma-3-270m-it`, sha `1220dfc9ad66ad900af54386fec093f4a4a030dd`, public, `apache-2.0`, files: `model.vmfb` (540,938,669 B, sha256 `ebb8c64bd707…d7d1`), `token_embeddings.npy`, `tokenizer.json`, `config.json`, plus extras (`model.onnx`, `chat_template.jinja`, `generation_config.json`, …) that this demo doesn't load.
- Board snapshot used for assumptions: `docs/tmp/sl2619-status.md` (`_generated_at: 2026-04-25T02:55:00+00:00`).
