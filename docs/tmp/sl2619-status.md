---
_generated_at: 2026-05-02T07:15:00-07:00
_source: /board_probe --target=sl2619 (READ-ONLY SSH to nouslogic-sl2619)
_freshness_window: 24h
_live_verified: true
_purpose_note: Pre-flight snapshot before planning a FunctionGemma 270M GGUF benchmark on the board (host-requested extras included in §6, §7, §8, §10, §11).
---

# SL2619 Live Board Snapshot

All Iron Laws (IL-2 memory, IL-6 mailbox, IL-7 RPMsg, IL-11 TEE) verified GREEN.
Prior Gemma 3 270M llama.cpp build + Q4_0 GGUF models already staged on the
SD card from the Apr 27 baseline run — a fresh FunctionGemma deploy can reuse
that infrastructure without rebuilding.

## 1. Identity
- Astra version: `scarthgap_6.12_v2.3.0`
- Kernel: `Linux nouslogic 6.12.62 #1 SMP PREEMPT Thu Apr  2 04:34:21 UTC 2026 aarch64`
- Hostname: `nouslogic`
- Uptime: 31 min, load 0.01 / 0.02 / 0.00
- Machine model (dmesg): `Synaptics SL2619 RDK`

## 2. Memory (IL-2 verification — GREEN)
- MemTotal: 1,962,988 kB (~1.87 GiB) — matches Iron Law claim
- MemAvailable: 1,749,848 kB (~1.67 GiB free for new workloads)
- SwapTotal: **0 kB** — no swap, IL-2 honored
- CMA: 512 MiB @ `0x5c000000–0x7bffffff` (524,288 KiB cma-reserved, dmesg L11) — IL-2 GREEN
- Other reserved: vdev0vring0 (32 KiB @ 0x0), vdev0vring1 (32 KiB @ 0x8000), vdevbuffer (960 KiB @ 0x10000)

## 3. IPC (IL-6 / IL-7 verification — GREEN)
- Mailbox: `f7e22000-f7e2205f : f7e22000.ipc ipc@f7e22000` — exactly the IL-6 address
- vring0 (A55→M52): `0x0` (32 KiB) — dmesg `vring0 0x0`
- vring1 (M52→A55): `0x8000` (32 KiB) — dmesg `vring1 0x8000`
- vdevbuffer:       `0x10000` (960 KiB)
- RPMsg control: `/dev/rpmsg_ctrl0` present
- RPMsg bus devices: `virtio0.rpmsg_ctrl.0.0`, `virtio0.rpmsg_ns.53.53` — IL-7 GREEN
- `virtio_rpmsg_bus virtio0: rpmsg host is online` (dmesg)

## 4. NPU / AI runtime (Torq + SyNAP — klamath)
- NPU platform: `/sys/devices/platform/soc/f7600000.synpu/` present (npu_region in /proc/iomem)
- Kernel module loaded: `syna_npu` (lsmod)
- iommu group: 1 (`torq f7600000.synpu: Adding to iommu group 1` — dmesg)
- `synap_cli`: `/usr/bin/synap_cli`, `/usr/bin/synap_cli_od` — present
- `/dev/synap*`: **absent** (this generation does not expose a userspace char dev — Torq runtime talks to NPU via libsynap → /dev/dma-heap, normal for klamath)
- `/sys/kernel/debug/synap*`: absent
- `/opt/synaptics`: absent (this build does not use /opt; Synaptics tooling lives under /usr)
- `/usr/share/synap/models/`: contains only `placeholder` (no preloaded SyNAP models)
- GStreamer plugins present: `synapinfer`, `synavideoconvertscale`, `synapoverlay`
- Vulkan: **absent** (no `vulkaninfo` binary; only `/dev/dri/card0` for DRM display)
- **Implication for FunctionGemma:** NPU acceleration of Gemma is *not* available via Vulkan/SyNAP today on this board image. CPU (NEON dotprod) is the only viable inference path for a GGUF run. A torq_examples/gemma3 staging dir exists at `/mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/` (model.vmfb, token_embeddings.npy, tokenizer.json) — that is the IREE-via-Torq path, separate from llama.cpp.

## 5. Services
- Running (key units): `tee-supplicant@teepriv0` (active 1y3mo — IL-11 TEE alive), `dropbear@…` (SSH), `weston`, `wpa_supplicant@wlan0`, `network-manager`, `dnsmasq`, `systemd-networkd`, `systemd-resolved`, `systemd-timesyncd`, `android-tools-adbd`
- Failed: `NetworkManager.service`, `swupdate.service`, `swupdate.socket` (NetworkManager replaced by network-manager LSB script + systemd-networkd; swupdate not in use here — non-blocking)
- `coordinator.service`: **not installed** (matches generic-finetune workspace state — coordinator is an SL2619 IPC-development binary, not relevant to a Gemma benchmark)

## 6. Storage (host-requested extras for ~520 MB GGUF + tokenizer fit)

| Mount | Device | Size | Used | Free | Notes |
|---|---|---|---|---|---|
| `/` | `/dev/mmcblk0p14` | 2.3 G | 838 M | **1.3 G** | Yocto rootfs, ext4 ro at boot then rw — fits a 520 MB GGUF but tight with tokenizer; not recommended |
| `/home` | `/dev/mmcblk0p18` | 22.7 G | 73 M | **21.4 G** | ext4, root-owned. Best general-purpose target if SD card unavailable |
| `/mnt/sdcard` | `/dev/mmcblk2p1` | 119.4 G | 2.6 G | **108.3 G** | ext2/ext3, removable — already hosts prior `gemma-3-270m-it-Q4_0.gguf` (230 MiB) and merged_v1 fine-tune (230 MiB). **Recommended target** for FunctionGemma GGUF + tokenizer |
| `/tmp`, `/dev/shm`, `/run`, `/var/volatile` | tmpfs | ~960 M each | small | ~370–960 M | RAM-backed; do NOT stage GGUF here — would consume the available ~1.67 GiB RAM |
| `/factory_setting` | `/dev/mmcblk0p1` | 11.7 M | 12 K | 11.4 M | Ignore |
| `/data`, `/opt` | — | — | — | — | **do not exist** as mountpoints on this image |

**Verdict for FunctionGemma 270M GGUF (~520 MB + tokenizer):** Stage on `/mnt/sdcard/models/<name>/` next to existing Gemma artifacts. 108 GiB free leaves room for many quantization variants and bench logs.

Block devices:
- `mmcblk0` (eMMC, 29.6 G) — boot/system, partitioned 1–18
- `mmcblk2` (SD card, 119.4 G) — the work-area mount
- `mtdblock0` (16 MiB) — boot/SPL flash, do not touch (IL-8 spirit)

## 7. CPU (host-requested extra)
- Cores: **2 × Cortex-A55** (CPU implementer 0x41 ARM, CPU part 0xd05 = Cortex-A55, variant 0x2, rev 0)
- Architecture: ARMv8-A AArch64
- BogoMIPS: 50.00 per core
- `nproc`: 2
- ARMv8 features (from `/proc/cpuinfo Features`):
  - **Required:** `fp asimd` (ASIMD = NEON, mandatory on ARMv8)
  - **Crypto:** `aes pmull sha1 sha2 crc32`
  - **Atomics + LSE:** `atomics`
  - **FP16 / half-precision:** `fphp asimdhp`
  - **Dot product (critical for Q4_0 / Q8_0 GEMM):** `asimddp` ✓
  - **Misc:** `cpuid asimdrdm lrcpc dcpop`
- **NO SVE / SVE2** (Cortex-A55 predates SVE; do not pass `-march=…+sve` to llama.cpp; the existing `/mnt/sdcard/llama-cpp/` build was compiled without SVE and works)
- **Implication for llama.cpp:** Build with `-mcpu=cortex-a55 -mfpu=neon-fp-armv8 -mfloat-abi=hard` flags. Q4_0/Q8_0 paths use the `asimddp` dotprod intrinsic — already confirmed working in the Apr 27 baseline run (see §11).

## 8. Toolchain on PATH (host-requested extra)

| Binary | On PATH? | Path / version |
|---|---|---|
| `llama-cli` | **No** (PATH = `/usr/sbin:/usr/bin:/sbin:/bin`) | But built and present at `/mnt/sdcard/llama-cpp/llama-cli` (8.6 MiB, llama.cpp version 1, build `0adede8`, gcc 13.3.0 aarch64, Apr 27) |
| `llama-server` | **No / not built** | Not present on disk — only `llama-cli`, `llama-bench`, `llama-completion`, `llama-perplexity` were cross-compiled |
| `llama-bench` | **No** | Present at `/mnt/sdcard/llama-cpp/llama-bench` (4.99 MiB, Apr 27) |
| `ollama` | **No** | Not installed |
| `python3` | **Yes** | `/usr/bin/python3` → Python **3.12.9** |
| `uv` | **No** | Not installed |
| `pip3` | **No** | Not installed; `python3 -m pip` also fails (`No module named pip`). Phase-1.5 doc shows pip is bootstrapped *into tmpfs* via `get-pip.py --user` per `/mnt/sdcard/p15-env.sh` |
| `cmake`, `make`, `gcc`, `g++`, `ninja` | **No** | Not on PATH — board has no native compiler. Cross-compile from server, scp binary across |

`PATH` (root): `/usr/sbin:/usr/bin:/sbin:/bin` (no /usr/local/bin even though `/usr/local/bin` exists and is empty)

## 9. Network (SSH connection method — host-requested extra)
- Active interface: `wlan0` only (`eth0` link-down NO-CARRIER)
- IPv4: `192.168.12.240/24`, gateway `192.168.12.1` (DHCP, lease ~7 d)
- IPv6: SLAAC + ULA + global (`2607:fb90:4880:1da:…`)
- Wi-Fi link: SSID `Kylie`, 5.765 GHz, signal -48 dBm (excellent), tx 292.5 / rx 325 MBit/s
- SSH client this session: `192.168.12.18 → 192.168.12.240:22` (per `$SSH_CONNECTION`)
- **scp path for FunctionGemma transfer:** `scp <file> nouslogic-sl2619:/mnt/sdcard/models/<dir>/`
  - Wi-Fi 5 GHz at -48 dBm should sustain ~30–40 MB/s real throughput → a 520 MB GGUF transfers in ~15–20 s
  - alternative: `adb push` over USB (android-tools-adbd is running) if Wi-Fi is congested

## 10. Existing Gemma / llama.cpp artifacts on board (host-requested extra)

`/mnt/sdcard/` is already the board-side staging area from the Apr 24–28 Gemma 3 270M baseline:

| Path | Size | Purpose |
|---|---|---|
| `/mnt/sdcard/llama-cpp/llama-cli` | 8.66 MiB | llama.cpp CLI — version 1, build `0adede8`, gcc 13.3.0 aarch64 |
| `/mnt/sdcard/llama-cpp/llama-bench` | 4.99 MiB | `llama-bench` for tg/pp throughput |
| `/mnt/sdcard/llama-cpp/llama-completion` | 6.82 MiB | low-level completion harness |
| `/mnt/sdcard/bin/llama-perplexity` | 6.89 MiB | perplexity eval tool |
| `/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf` | 230.4 MiB | base IT model, Q4_0 |
| `/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/merged_v1.q4_0.gguf` | 230.3 MiB | fine-tune merge v1, Q4_0 |
| `/mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/model.vmfb` | 540.9 MiB | IREE bytecode for the SyNAP/Torq path (separate runtime from llama.cpp) |
| `/mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/token_embeddings.npy` | 335.5 MiB | external embeddings for the Torq path |
| `/mnt/sdcard/torq-examples/models/Synaptics/gemma-3-270m-it/tokenizer.json` | 33.4 MiB | tokenizer |
| `/mnt/sdcard/bench/2026-04-27_gemma3-base-llamacpp-baseline.{log,jsonl}` | 71 KB / 870 B | prior baseline output |
| `/mnt/sdcard/bench/q1-a55-q4_0.log`, `q1r-a55-q4_0.log`, `h5r-a55-q4_0.log` | ~10–14 KB each | per-config bench captures |
| `/mnt/sdcard/p15-env.sh` | 800 B | Phase 1.5 env-activation script (PYTHONUSERBASE → /tmp tmpfs) |
| `/mnt/sdcard/p15site/` | (not measured) | Phase 1.5 wheels (onnxruntime, tokenizers, torq_runtime, …) |

**No FunctionGemma-specific GGUF on the board yet.** A FunctionGemma 270M GGUF is the new artifact to deploy.

`/home`, `/root`, `/usr/local` are essentially empty (root home has only `.ssh`; `/home/root` is 71 MiB and contains no GGUFs).

## 11. dmesg HW highlights (truncated, IL-relevant)

```
Machine model: Synaptics SL2619 RDK
OF: reserved mem: vdev0vring0@0  0x0..0x7fff   (32 KiB)
OF: reserved mem: vdev0vring1@8000 0x8000..0xffff (32 KiB)
OF: reserved mem: vdevbuffer 0x10000..0xfffff (960 KiB)
Reserved memory: created CMA memory pool at 0x5c000000, size 512 MiB
syna-rpmsg 0.rpmsg: rpdev vdev0: vring0 0x0, vring1 0x8000
virtio_rpmsg_bus virtio0: rpmsg host is online
torq f7600000.synpu: Adding to iommu group 1
[drm] Initialized synaptics 1.0.0 for soc:drm on minor 0
```

All values cross-check with §2 / §3 / §4 above — no drift.

## 12. BusyBox caveats encountered during this probe
- `head -N` (positional) is rejected — must use `head -n N`. One follow-up SSH call was needed.
- `gst-inspect-1.0 … | head -n 3` worked; `command | head -3` would have failed.
- Confirmed: `arecord -l`, `aplay -l`, `lsblk`, `lsmod`, `find -maxdepth N`, `du -sh`, `stat -fc` all behave normally.

## 13. Discrepancies vs Iron Laws

**None.** All four Iron-Law checks pass:

| IL | Check | Observed | Verdict |
|---|---|---|---|
| IL-2 | RAM ≈ 1.87 GiB, no swap | 1,962,988 kB total, SwapTotal=0 | GREEN |
| IL-2 | CMA = 512 MiB | 0x5c000000–0x7bffffff (524,288 KiB) | GREEN |
| IL-6 | Mailbox @ 0xF7E22000 | `f7e22000-f7e2205f : f7e22000.ipc` | GREEN |
| IL-7 | RPMsg nodes alive | `/dev/rpmsg_ctrl0` + `virtio0.rpmsg_*` | GREEN |
| IL-11 | TEE alive | `tee-supplicant@teepriv0` active | GREEN |

Failed services (`NetworkManager`, `swupdate`, `swupdate.socket`) are not Iron-Law concerns — NetworkManager is shadowed by `network-manager` (LSB) + `systemd-networkd`, and swupdate is unused on this image. No remediation required.

## 14. Recommendations for the FunctionGemma 270M GGUF benchmark

1. **Stage location:** `/mnt/sdcard/models/functiongemma-270m-q4_0/` next to the existing Gemma 3 dirs. 108 GiB free, ext2/ext3 (no fancy attrs needed).
2. **Reuse existing llama.cpp build:** `/mnt/sdcard/llama-cpp/llama-cli` (build `0adede8`, gcc 13.3 aarch64) and `/mnt/sdcard/llama-cpp/llama-bench` are ready. **Caveat:** if FunctionGemma's GGUF metadata uses a newer key/format than build `0adede8`, may need a refresh build from server. Smoke-test with `llama-cli --version` first; if model load fails with "unknown architecture" or "missing key", rebuild llama.cpp on `nouslogic-server` against the FunctionGemma GGUF spec.
3. **No NPU path for this benchmark.** Plan CPU-only (2 × Cortex-A55, NEON+dotprod, no SVE). Set `-t 2` in `llama-bench`. Expect throughput in the same ballpark as the Apr 27 Gemma 3 270M baseline.
4. **Transfer command:** `scp <file>.gguf nouslogic-sl2619:/mnt/sdcard/models/functiongemma-270m-q4_0/` from WSL host. Wi-Fi 5 GHz at -48 dBm → ~15–20 s for 520 MB.
5. **No `pip` / `uv` on board.** Any tokenizer-side Python harness must either (a) bootstrap pip into `/tmp/pipbase` via the existing `p15-env.sh` pattern, or (b) be a pure-`stdlib` Python 3.12 script (preferred for a one-shot bench).
6. **No `llama-server` binary on board.** If FunctionGemma needs an HTTP harness, build `llama-server` on `nouslogic-server` and add to the deploy bundle.
