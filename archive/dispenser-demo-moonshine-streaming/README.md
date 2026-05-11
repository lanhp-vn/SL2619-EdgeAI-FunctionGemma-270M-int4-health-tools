# archive/dispenser-demo-moonshine-streaming/

Frozen reference for the **moonshine-streaming-tiny** GGUF path that was
provisionally pinned during dispenser-demo Phase 0 on 2026-05-11 (AM) and
then superseded that same afternoon after the **moonshine-tiny** (non-streaming)
proof clearly outperformed it on the SL2619.

This directory is read-only history. Do not run, edit, or extend it. Active
work uses `moonshine-tiny` via CrispASR — see
[`docs/plans/dispenser-demo/decisions-log.md`](../../docs/plans/dispenser-demo/decisions-log.md)
for the current binding.

## What's here

- [`working-recipe.md`](working-recipe.md) — complete build → deploy → smoke
  recipe for the streaming variant, including HF artifact names, scp paths,
  invocation flags, empirical numbers, and the cosmetic-warning explanation
  for `--no-punctuation`.

## Why it was superseded

Same `crispasr` aarch64 binary, same JFK fixture, same flags
(`-l en --no-punctuation -t 2`):

| Metric | moonshine-streaming-tiny (here) | moonshine-tiny (active) |
| --- | --- | --- |
| GGUF q4_k size | 30.6 MB | **20.2 MB** (-34 %) |
| Wall time (11 s clip) | 7.48 s | **4.66 s** (-38 %) |
| Realtime factor | 1.5× | **2.4×** |
| Peak VmRSS | 69.5 MB | **49.6 MB** (-29 %) |

The streaming variant's win is per-chunk incremental decoding (low time-to-first-token
during a still-in-progress utterance). The dispenser-demo voice flow operates
on complete utterances (push-to-talk or VAD-cut-at-silence), so that win
doesn't matter and the non-streaming variant's smaller-model latency advantage
wins on every axis.

## When to consult this

- If Phase 3.5 redesigns voice capture to stream partial transcripts to
  FunctionGemma during ongoing speech (low-latency interruption support), the
  streaming variant becomes relevant again — reopen `decisions-log.md` with a
  fresh dated entry rather than reusing the 2026-05-11 (AM) row.
- If the active `moonshine-tiny` path develops a regression, the streaming
  variant is a known-working fallback — recipe in `working-recipe.md` is
  self-contained.

## Pointers to historical detail

- Empirical result rows live in
  [`docs/plans/dispenser-demo/crispasr-spike-notes.md` §6](../../docs/plans/dispenser-demo/crispasr-spike-notes.md)
  (append-only log; both the streaming and non-streaming runs are preserved
  there verbatim).
- The original 2026-05-11 (AM) decision row is in
  [`docs/plans/dispenser-demo/decisions-log.md`](../../docs/plans/dispenser-demo/decisions-log.md);
  the supersession entry the same afternoon explains the flip.
