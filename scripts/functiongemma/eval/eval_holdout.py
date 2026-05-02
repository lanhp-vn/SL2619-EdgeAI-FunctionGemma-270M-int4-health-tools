#!/usr/bin/env python3
"""FunctionGemma M5/M6 holdout evaluator — SPEC + skeleton.

SPEC ONLY — IMPLEMENTATION DEFERRED until M5 produces a merged checkpoint.

This file pins the metric definitions and CLI surface so M6 can fill the model
loop in without re-arguing the contract. The metric layer + holdout loader +
CLI flags are runnable today; the model-inference seam raises NotImplementedError.

Run-blocking conditions (asserted in main(), exit 2 with a clear stderr message):
1. ``--checkpoint`` arg points at a path that doesn't exist (only checked when
   we actually need to run inference — `--list-categories` / `--dry-run` skip).
2. ``--holdout`` defaults to ``data/functiongemma/eval_holdout_v1.jsonl``;
   abort if missing.
3. The actual ``run_inference()`` call is replaced with ``NotImplementedError``
   until M5/M6 lands.

Acceptance contract (`docs/plans/FunctionGemma/README.md` §11.4 / §14 M6):
- Per-category pass-rate ≥ 80 %, *every* category individually. An overall
  ≥ 80 % that hides one weak category fails the gate.
- ``partial`` (same tool names, different arguments) does NOT count toward
  pass_rate — only ``match`` does. The bar is strict equivalence.
  ``partial`` is reported for diagnostic use only (Phase D loss inspection).

Reference: ``scripts/functiongemma_smoke.py`` exercises Path A (HF tokenizer
chat-template render + ``llama-cpp-python`` GGUF inference). M6 should mirror
that pattern against the merged Q8_0 GGUF at
``~/functiongemma-finetune/merged_fg_v1/`` (path TBD by M5).
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from gemma_tools.functiongemma.dataset import load_jsonl  # type: ignore[import-untyped]

# Refusal categories: the gate is "predicted zero tool_calls". The post-tool
# NL answer is informational only; tool-call shape is the contract.
REFUSAL_CATEGORIES: frozenset[str] = frozenset(
    {"off_topic_refusal", "medical_advice_refusal"}
)

DEFAULT_HOLDOUT = Path("data/functiongemma/eval_holdout_v1.jsonl")


class Equivalence(StrEnum):
    """Per-row tool-call equivalence verdict.

    `StrEnum` so `parametrize` ids and stringified output stay readable
    without an explicit `.value` hop.
    """

    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class GoldTrace:
    """The reference behaviour for one holdout row.

    `tool_calls` is flattened in conversation order across *every* assistant
    turn — two-turn rows contribute one call from the first assistant turn and
    zero from the second, parallel rows contribute N from a single turn,
    refusals contribute none. The order matters: `tool_call_equivalent`
    compares by index, not as a multiset.

    `assistant_text` is the post-`</think>` body of the **last** assistant
    message. Per the seed-validator shape rule, a call-only assistant turn has
    an empty tail; the final-answer / refusal turn carries `\\n<NL answer>`.
    Reading the last assistant turn gives the user-facing reply for both
    `two_turn` and refusal rows uniformly. M6 may use this for a soft
    follow-on metric (e.g. response-similarity), but §11.4 acceptance is the
    tool-call gate alone.
    """

    tool_calls: tuple[dict[str, Any], ...]
    assistant_text: str
    category: str
    row_id: str | None


@dataclass(frozen=True, slots=True)
class CategoryStats:
    category: str
    n: int
    n_match: int
    n_partial: int
    n_mismatch: int

    @property
    def pass_rate(self) -> float:
        # Strict — only `match` counts; `partial` is diagnostic.
        return self.n_match / self.n if self.n else 0.0

    @property
    def bar_pass(self) -> bool:
        return self.pass_rate >= 0.80


# --------------------------------------------------------------------------
# Gold-trace extraction.
# --------------------------------------------------------------------------


def _extract_assistant_text(content: str) -> str:
    """Return the post-`</think>` tail (NL answer) or "" if there is none.

    Mirrors the shape rule pinned by `functiongemma_dataset` — assistant
    content always starts with one `<think>...</think>` block; what follows
    is either empty (call-only turn) or `\\n<NL>` (final-answer / refusal).
    We strip a single leading newline so callers see the answer verbatim.
    """
    close = content.find("</think>")
    if close == -1:
        # Defensive — validator should have rejected this, but if a row
        # bypasses validation we still return *something* useful.
        return content
    tail = content[close + len("</think>"):]
    return tail[1:] if tail.startswith("\n") else tail


def extract_gold_trace(row: dict[str, Any]) -> GoldTrace:
    """Pull the FIRST assistant turn's tool_calls + the final NL answer.

    The eval inference path generates exactly one assistant response from
    the user prompt (`run_inference` builds messages_prefix up to the first
    assistant turn). The gold contract therefore is "the FIRST turn's tool
    calls" — flattening across multiple turns (e.g. two_turn rows have 2
    assistant turns each with their own call) would produce gold=N vs
    pred=1 mismatches that score 0% on the multi-turn categories. M6
    multi-turn behavioral eval is a separate gate (deferred — would need
    a tool-execution loop) and is not what `tool_call_equivalent` measures.

    `assistant_text` is taken from the LAST assistant turn (the final NL
    answer in the gold conversation), kept for diagnostic reporting.
    Each entry is normalized to `{"name": str, "arguments": dict}` —
    the seed shape wraps these in `{"id", "type": "function", "function": {...}}`.
    """
    first_calls: list[dict[str, Any]] = []
    last_assistant_content: str = ""
    seen_first = False
    for m in row.get("messages", []):
        if m.get("role") != "assistant":
            continue
        last_assistant_content = m.get("content", "")
        if seen_first:
            continue
        seen_first = True
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            arguments = fn.get("arguments")
            if arguments is None:
                arguments = tc.get("arguments", {})
            if not isinstance(name, str):
                # Validator should have caught; bail loud rather than silently mis-score.
                raise ValueError(f"row {row.get('id')!r}: tool_call missing name: {tc!r}")
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"row {row.get('id')!r}: tool_call arguments not a dict: {arguments!r}"
                )
            first_calls.append({"name": name, "arguments": arguments})
    return GoldTrace(
        tool_calls=tuple(first_calls),
        assistant_text=_extract_assistant_text(last_assistant_content),
        category=str(row.get("category", "")),
        row_id=row.get("id") if isinstance(row.get("id"), str) else None,
    )


# --------------------------------------------------------------------------
# Pure metric.
# --------------------------------------------------------------------------


def _norm_args(args: Any) -> Any:
    """Deep-normalize string values in a tool-call ``arguments`` payload.

    String arguments are case-normalized via ``.casefold()`` because the
    underlying tools resolve case-insensitively per M3 spec; this prevents
    two functionally-correct rows from being scored as PARTIAL when the
    only difference is letter case (e.g. ``"Lisinopril"`` vs ``"lisinopril"``).
    Non-str values pass through unchanged. Recurses into nested dicts/lists.
    """
    if isinstance(args, str):
        return args.casefold()
    if isinstance(args, dict):
        return {k: _norm_args(v) for k, v in args.items()}
    if isinstance(args, list):
        return [_norm_args(v) for v in args]
    return args


def tool_call_equivalent(
    predicted_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    gold_calls: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    category: str,
) -> Equivalence:
    """Compare a predicted tool-call sequence to gold.

    Per-row verdict:
    - ``MATCH``    — same `name`s in same order AND every `arguments` dict
      deep-equal. For refusal categories (``off_topic_refusal``,
      ``medical_advice_refusal``) the contract is "predicted has zero
      tool_calls" regardless of NL output; gold has zero too.
    - ``PARTIAL``  — same set of names (in order) but at least one `arguments`
      dict differs. Reported for diagnostics; does NOT count toward pass_rate.
    - ``MISMATCH`` — names differ, OR predicted has calls when gold has none
      (refusal break), OR predicted has no calls when gold has calls.

    String arguments are case-normalized via ``.casefold()`` before deep
    equality (see ``_norm_args``) because the underlying tools resolve
    case-insensitively per M3 spec; this prevents two functionally-correct
    rows from being scored as PARTIAL.

    Pure function — no I/O. The branch order is deliberate: the
    `[] vs []` (refusal-clean) check fires before the names-differ check,
    so an empty-vs-empty comparison short-circuits to MATCH.
    """
    pred = list(predicted_calls)
    gold = list(gold_calls)

    if category in REFUSAL_CATEGORIES:
        # Refusal contract is binary on tool-call presence. Empty pred and empty gold → MATCH.
        if len(pred) == 0 and len(gold) == 0:
            return Equivalence.MATCH
        # Either side has calls → mismatch (refusal violated, or gold authoring drift).
        return Equivalence.MISMATCH

    # Empty-vs-empty for non-refusal: both parties agreed no tool was needed.
    # Treat as MATCH; mismatched empty/non-empty is MISMATCH.
    if len(pred) == 0 and len(gold) == 0:
        return Equivalence.MATCH
    if len(pred) == 0 or len(gold) == 0:
        return Equivalence.MISMATCH

    if len(pred) != len(gold):
        return Equivalence.MISMATCH

    pred_names = [c.get("name") for c in pred]
    gold_names = [c.get("name") for c in gold]
    if pred_names != gold_names:
        return Equivalence.MISMATCH

    # Names align; check args. Deep equality on parsed dicts, with str values
    # casefolded so case-only diffs (e.g. "Lisinopril" vs "lisinopril") MATCH.
    for p, g in zip(pred, gold, strict=True):
        if _norm_args(p.get("arguments")) != _norm_args(g.get("arguments")):
            return Equivalence.PARTIAL
    return Equivalence.MATCH


# --------------------------------------------------------------------------
# Inference seam — wired in M6 (2026-05-01).
# --------------------------------------------------------------------------


# Module-level cache for the loaded model+tokenizer. main() iterates 56 rows;
# without caching we'd reload the BF16 checkpoint per row (~10s * 56 = 9 min
# vs ~10s + 56 * 2s ~= 2 min). Keyed by resolved Path so re-runs against the
# same dir hit the cache.
_INFERENCE_CACHE: dict[Path, tuple[Any, Any]] = {}

# FunctionGemma wire-format parser. Mirrored from
# `scripts/functiongemma_smoke.py:_CALL_RE` / `_ARG_RE` — kept in sync there.
# Tolerated drift: chat template renders `call:NAME` (colon), but the Q4_K_M
# GGUF observed in §15.4 Path B emits `call NAME` (space). Accept both.
_FG_CALL_RE = re.compile(
    r"<start_function_call>\s*call[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
_FG_ARG_RE = re.compile(
    r"(\w+)\s*:\s*(?:<escape>(.*?)<escape>|([^,}]*))",
    re.DOTALL,
)


def _parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``<start_function_call>...<end_function_call>`` blocks.

    Returns ``[{"name": str, "arguments": dict}, ...]`` — the shape
    `tool_call_equivalent` expects (matches `extract_gold_trace`'s gold side).
    """
    calls: list[dict[str, Any]] = []
    for m in _FG_CALL_RE.finditer(text):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for am in _FG_ARG_RE.finditer(body):
            key = am.group(1)
            esc, plain = am.group(2), am.group(3)
            if esc is not None:
                args[key] = esc
            else:
                stripped = (plain or "").strip()
                if stripped == "":
                    continue
                args[key] = stripped
        calls.append({"name": name, "arguments": args})
    return calls


def _load_inference(checkpoint_path: Path) -> tuple[Any, Any]:
    """Load merged HF checkpoint + tokenizer once, cache for the eval loop.

    Lazy: requires `transformers` + `torch` (server-only). The host doesn't
    run this — it's the M6 server-side eval seam.
    """
    resolved = checkpoint_path.resolve()
    if resolved in _INFERENCE_CACHE:
        return _INFERENCE_CACHE[resolved]
    # Lazy imports — host CI does not pay this 500 MB import cost.
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(str(resolved))
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    _INFERENCE_CACHE[resolved] = (model, tokenizer)
    return model, tokenizer


def run_inference(
    checkpoint_path: Path,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Render the chat template, generate greedily, parse FG wire format.

    Returns:
        ``{"tool_calls": [{"name": str, "arguments": dict}, ...],
           "assistant_text": str}``  — the shape `extract_gold_trace` returns
        on the gold side. ``tool_calls`` is empty for refusal turns;
        ``assistant_text`` is whatever follows the last ``</think>`` tag (or
        the full output if no `<think>` block was emitted).

    Greedy decode is intentional — the M6 metric is tool-call equivalence,
    not lexical similarity, and sampling would inject noise into the
    pass-rate signal. ``temperature=0`` via ``do_sample=False``.
    """
    import torch  # type: ignore[import-not-found]
    model, tokenizer = _load_inference(checkpoint_path)

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template returned {type(prompt).__name__}")
    # Mirror the §9.4.2 step-2 BOS strip — the chat template emits `<bos>`
    # and the tokenizer would re-prepend it at tokenize time.
    prompt = prompt.removeprefix("<bos>")

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out_ids[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # Parse FG wire-format tool calls.
    tool_calls = _parse_function_calls(output_text)
    # Extract the NL answer after the last </think>. Mirrors
    # `_extract_assistant_text` on the gold side.
    assistant_text = _extract_assistant_text(output_text)

    return {"tool_calls": tool_calls, "assistant_text": assistant_text}


# --------------------------------------------------------------------------
# Aggregation + rendering.
# --------------------------------------------------------------------------


def aggregate_by_category(
    verdicts: list[tuple[str, Equivalence]],
) -> list[CategoryStats]:
    """Collapse `(category, verdict)` pairs into per-category stats.

    Sorted alphabetically by category for stable output ordering.
    """
    by_cat: dict[str, list[Equivalence]] = {}
    for cat, v in verdicts:
        by_cat.setdefault(cat, []).append(v)
    out: list[CategoryStats] = []
    for cat in sorted(by_cat):
        verd = by_cat[cat]
        out.append(
            CategoryStats(
                category=cat,
                n=len(verd),
                n_match=sum(1 for v in verd if v is Equivalence.MATCH),
                n_partial=sum(1 for v in verd if v is Equivalence.PARTIAL),
                n_mismatch=sum(1 for v in verd if v is Equivalence.MISMATCH),
            )
        )
    return out


def render_summary_table(stats: list[CategoryStats]) -> str:
    """Markdown table for the docs/bench output.

    Columns: category | n | match | partial | mismatch | pass_rate | bar_pass.
    `bar_pass` is the §11.4 ≥ 80 % gate per category — every row must show
    `PASS` for M6 to ship green.
    """
    if not stats:
        return "_no rows scored._\n"
    overall_match = sum(s.n_match for s in stats)
    overall_n = sum(s.n for s in stats)
    overall_rate = overall_match / overall_n if overall_n else 0.0
    all_bar_pass = all(s.bar_pass for s in stats)
    header = [
        f"**Overall pass rate**: {overall_match}/{overall_n} ({overall_rate * 100:.1f}%).  "
        f"**All categories ≥ 80 %**: {'YES' if all_bar_pass else 'NO'}.",
        "",
        "| category | n | match | partial | mismatch | pass_rate | bar_pass |",
        "|---|---|---|---|---|---|---|",
    ]
    body: list[str] = []
    for s in stats:
        body.append(
            f"| {s.category} | {s.n} | {s.n_match} | {s.n_partial} | "
            f"{s.n_mismatch} | {s.pass_rate * 100:.1f}% | "
            f"{'PASS' if s.bar_pass else 'FAIL'} |"
        )
    return "\n".join(header + body) + "\n"


# --------------------------------------------------------------------------
# CLI subcommands.
# --------------------------------------------------------------------------


def _print_category_counts(holdout_path: Path) -> int:
    """`--list-categories` body — pure I/O + count, no inference."""
    rows = list(load_jsonl(holdout_path))
    counts = Counter(r.get("category", "<missing>") for r in rows)
    print(f"holdout: {holdout_path} ({len(rows)} rows)")
    print(f"{'category':<30} count")
    print("-" * 38)
    for cat in sorted(counts):
        print(f"{cat:<30} {counts[cat]}")
    return 0


def _dry_run(holdout_path: Path) -> int:
    """`--dry-run` body — load holdout, extract gold traces, run gold-vs-gold.

    Sanity gate: every row's gold trace compared against itself MUST resolve
    to MATCH (including refusal rows where both sides have empty tool_calls).
    A non-MATCH here means either the extractor or the metric has a bug —
    fail loud so the M6 implementer doesn't ship a broken bar.
    """
    rows = list(load_jsonl(holdout_path))
    verdicts: list[tuple[str, Equivalence]] = []
    for raw in rows:
        gold = extract_gold_trace(raw)
        v = tool_call_equivalent(
            list(gold.tool_calls), list(gold.tool_calls), category=gold.category
        )
        verdicts.append((gold.category, v))
        if v is not Equivalence.MATCH:
            print(
                f"FAIL gold-vs-gold mismatch on row {gold.row_id!r} "
                f"(cat={gold.category}, verdict={v.value})",
                file=sys.stderr,
            )
            return 1
    stats = aggregate_by_category(verdicts)
    print(f"dry-run: {len(rows)} rows, all gold-vs-gold → MATCH")
    print(render_summary_table(stats))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval-functiongemma-holdout",
        description=(
            "Score the FunctionGemma merged checkpoint against the v1 holdout. "
            "M6 acceptance: per-category tool-call equivalence ≥ 80 % "
            "(see docs/plans/FunctionGemma/README.md §11.4)."
        ),
    )
    # `required=False` here; main() enforces "required iff actually running inference".
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to merged checkpoint (required for full eval; ignored by --list-categories / --dry-run).",
    )
    p.add_argument(
        "--holdout",
        type=Path,
        default=DEFAULT_HOLDOUT,
        help=f"Path to holdout JSONL. Default: {DEFAULT_HOLDOUT}.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown summary. Default: docs/bench-notes/functiongemma/<today>_functiongemma-eval.md.",
    )
    p.add_argument(
        "--list-categories",
        action="store_true",
        help="Load holdout, print per-category row counts, exit. No inference.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load holdout, extract gold traces, run gold-vs-gold sanity → 100 %% MATCH.",
    )
    # C4: per-row decode budget. Default mirrors the M6 first-run number; bump
    # to 512 / 1024 to test whether truncation suppresses category pass-rates
    # (empty-prediction symptom on fact_lookup / tool_error_recovery).
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Greedy-decode token budget per row. Default 256 (M6 first run).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    holdout: Path = args.holdout
    if not holdout.exists():
        print(
            f"ERROR holdout not found: {holdout}\n"
            "Sibling agent should produce this file (56 rows, 8 per category x 7 categories). "
            "See docs/plans/FunctionGemma/README.md §9.7.",
            file=sys.stderr,
        )
        return 2

    if args.list_categories:
        return _print_category_counts(holdout)

    if args.dry_run:
        return _dry_run(holdout)

    # Full-eval branch — checkpoint required.
    if args.checkpoint is None:
        print(
            "ERROR --checkpoint is required for full eval (omit only with "
            "--list-categories or --dry-run).",
            file=sys.stderr,
        )
        return 2
    if not args.checkpoint.exists():
        print(
            f"ERROR checkpoint not found: {args.checkpoint}\n"
            "M5 should produce this. The eval script is SPEC ONLY until then.",
            file=sys.stderr,
        )
        return 2

    output: Path = args.output or (
        Path("docs/bench-notes/functiongemma") / f"{datetime.date.today().isoformat()}_functiongemma-eval.md"
    )

    # Inference loop. Pass the messages-prefix (everything before the first
    # assistant turn) + the row's `tools` declaration. run_inference handles
    # chat-template render + BOS strip + greedy decode + FG parse.
    rows = list(load_jsonl(holdout))
    verdicts: list[tuple[str, Equivalence]] = []
    print(f"=== running inference on {len(rows)} rows ===", file=sys.stderr)
    for i, raw in enumerate(rows):
        gold = extract_gold_trace(raw)
        msgs = raw.get("messages", [])
        first_assistant = next(
            (j for j, m in enumerate(msgs) if m.get("role") == "assistant"),
            len(msgs),
        )
        prompt_msgs = msgs[:first_assistant]
        tools = raw.get("tools")
        predicted = run_inference(
            args.checkpoint, prompt_msgs, tools, max_new_tokens=args.max_new_tokens
        )
        v = tool_call_equivalent(
            predicted.get("tool_calls", []),
            list(gold.tool_calls),
            category=gold.category,
        )
        verdicts.append((gold.category, v))
        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            print(f"  {i+1}/{len(rows)} done", file=sys.stderr)

    stats = aggregate_by_category(verdicts)
    markdown = render_summary_table(stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0 if all(s.bar_pass for s in stats) else 1


if __name__ == "__main__":
    sys.exit(main())
