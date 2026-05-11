# dispenser-demo decisions log

Append-only record of binding decisions for the dispenser-demo plan. Each
entry pins one resolved question. Update existing entries only to add follow-up
references; do not rewrite history.

The plan itself lives at [`plan.md`](plan.md). Phase-specific working notes
(e.g. [`crispasr-spike-notes.md`](crispasr-spike-notes.md)) are the authoritative
record of the underlying analysis; this file is the index.

---

## 2026-05-11 — Phase 0: KEEP CrispASR + Moonshine Streaming Tiny GGUF

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
