# Iter-002 Q4_0 on-board smoke — 2026-05-12

**Outcome:** Q4_0 GGUF deploys cleanly on the SL2619 board's
`llama-completion b8925`. Tool routing is correct on the canonical hard-case
row (`na-003 "When do I see Dr. Chen?"` → `get_next_appointment{}`); decode
throughput and memory match iter-001's Q4_0 numbers within noise. **Phase
1.7 gate met.**

## Setup

| Component | Value |
| --- | --- |
| Board | SL2619, Cortex-A55 ×2, 1.87 GiB RAM |
| `llama-completion` (board) | `b8925` / `0adede8` (same binary iter-001 uses) |
| GGUF | `finetuned_dispenser_q4_0.gguf` (224 MiB) |
| GGUF sha256 | `85893a795aec4b2adc2dbc7084f5b27e3ecd5a1ef885fd69d5af9678632368b9` |
| Prompt prefix (Distil wrap) | 5345 bytes, rendered by `scripts/dispenser_demo/data/gen_prompt_templates.py` |
| Prompt suffix (`<end_of_turn>` + `<start_of_turn>model`) | 35 bytes |
| Decode params | `-t 2 -n 64 --temp 0 --top-k 1 --seed 42 -r '<end_function_call>'` |

## Result (single prompt)

Input: `"When do I see Dr. Chen?"` (val row `na-003`, gold =
`get_next_appointment{}`).

Output:
```
model information:
call: get_next_appointment{}<end_function_call>
```

Tool name + args match gold. The `information:` prefix and missing
`<start_function_call>` opener are wire-format quirks (see §"Wire format
note" below), NOT a routing error.

## Throughput + memory

| metric | iter-002 Q4_0 (this run) | iter-001 Q4_0 baseline |
| --- | ---: | ---: |
| Prompt eval | 62.44 tok/s | 60.1 tok/s |
| Decode | 10.39 tok/s | 10.27 tok/s |
| Total wall (cold, 1160 tok) | 19.5 s | ~28 s |
| Host (RAM) | 849 MiB total = 224 model + 111 ctx + 514 compute | similar |

Within iter-001's measured envelope.

## Wire format note (Phase 3 parser implication)

The model emits a `call: NAME{...}<end_function_call>` block but
**without** the leading `<start_function_call>` opener iter-001 emits. The
iter-001 parser regex (`scripts/functiongemma/eval/eval_holdout.py:_FG_CALL_RE`)
requires the opener; it will fail to extract the call from iter-002's
on-board output unless updated.

Fix (Phase 3 — `scripts/dispenser_demo/eval/eval_holdout.py`,
`chat_board.py` analogue):

```python
# Make <start_function_call> optional and tolerate the `call ` (space) /
# `call:` (colon) prefix variants iter-002 produces on-board.
_FG_CALL_RE = re.compile(
    r"(?:<start_function_call>)?\s*call\s*[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
```

This is also tolerant of iter-001's pattern (the `(?:...)?` makes the opener
optional), so a single parser handles both iterations.

## Why host eval ≠ board eval for Q4_0 (the iter-001 lesson, reversed)

Iter-001's 2026-05-02 sweep documented an asymmetry: K-quants
(`Q4_K_M`/`Q5_K_M`/`Q8_0`/`IQ4_XS`) decode correctly on host
(`llama-cpp-python 0.3.21`/ggml `b8981`) but fail on the board's older
`llama-completion b8925` — scale-factor encoding skew drops
`<start_function_call>`. Q4_0 was the only variant that worked on board.

Iter-002 surfaces the OPPOSITE asymmetry for Q4_0:
- **Host eval of iter-002 Q4_0:** 30 % pass rate, output is gibberish
  (`<start_function_call>len_of_age_digits...`). The newer host runtime
  doesn't decode iter-002 Q4_0 cleanly — likely because iter-002's weight
  distribution has more outliers that lose precision under Q4_0's symmetric
  scaling, and the newer host runtime is stricter than the older board
  runtime.
- **Board eval of iter-002 Q4_0:** clean tool routing (this report). The
  older runtime tolerates the same precision loss.

This is consistent with iter-001's documentation — Q4_0 is the runtime-
compatibility quant, not the host-quality quant. For iter-002, ship Q4_0
on board; do NOT use Q4_0 for host eval. Host eval should use FP16 (or
Q5_K_M / Q8_0 if a quant is needed) — they're all clean on host (100 % on
val) per the 2026-05-12 quant sweep.

## What's left for full Phase 4 acceptance

This smoke validates 1 of 10 val rows. A complete on-board sweep is
deferred to Phase 4 acceptance:

```bash
# 10-row on-board eval (deferred; expected wall ~80 s with prompt cache primed):
ssh nouslogic-sl2619 'for row in <each val row>; do
  ./fg-ask-board-iter002.sh "$row"
done'
```

The Phase 4 gate is per-category pass-rate ≥ 90 % on board. Single-row
evidence is suggestive; final acceptance still needs the full sweep, and
the parser regex must be updated first (see §"Wire format note" above).

## References

- iter-001 sweep (host-vs-board Q4_0 asymmetry, K-quant disqualification):
  `docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md`
- Host quant sweep for iter-002 (Q4_0 30 %; K-quants 100 %):
  `docs/bench-notes/dispenser-demo/2026-05-11_dispenser-eval-gguf-*.md`
- Wire format issue + parser fix: this file §"Wire format note".
