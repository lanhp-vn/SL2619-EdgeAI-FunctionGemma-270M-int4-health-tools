#!/usr/bin/env python3
"""Dispenser-demo iter-002 holdout evaluator — Phase 1.6.

Source-of-truth contract: `docs/plans/dispenser-demo/plan.md` §9.1 step 1.6.
Acceptance bar: **per-category pass-rate ≥ 90 %**, EVERY category individually.
An overall ≥ 90 % that hides a weak category fails the gate.

Holdout: `data/dispenser_demo/dataset_v1/val.jsonl` (10 rows held out from
both training and Distil's test set). Synthgen never paraphrased these rows;
they are the cleanest independent eval signal for the tuned student.

Two inference seams (one of `--checkpoint` or `--gguf` required):

- `--checkpoint <dir>` — HF safetensors via transformers (BF16 / FP16 on GPU
  or CPU). Default tokenizer dir is the same path.
- `--gguf <path>` — GGUF via llama-cpp-python. `--tokenizer-dir` is required
  separately (the GGUF carries its own template metadata, but rendering via
  HF `apply_chat_template(..., tools=...)` keeps the eval byte-equivalent to
  the deployed chat path).

Wire format parser is shared with the iter-001 baseline — FunctionGemma's
`<start_function_call>` / `<end_function_call>` tags are model-level, not
task-specific. Logic differs from `scripts/functiongemma/eval/eval_holdout.py`
in two places:

1. **No REFUSAL_CATEGORIES special case.** Every dispenser_demo row has
   exactly one tool call (refusal rows emit `refuse_out_of_scope(reason)`
   per the plan §7 5-tool design). Empty-vs-empty short-circuits don't apply.
2. **Pass-rate gate is 90 %, not 80 %.** Higher bar because the dispenser's
   universe is much narrower than iter-001's.

CLI subcommands:

- `--list-categories` — load holdout, print per-cat counts, exit. No inference.
- `--dry-run` — extract gold traces, run gold-vs-gold sanity → 100 % MATCH.
- (default) — full eval against the chosen seam.

Usage:

    # Host CPU GGUF eval (5-10x faster than HF on CPU)
    uv run python scripts/dispenser_demo/eval/eval_holdout.py \\
        --gguf releases/functiongemma-270m/002-dispenser-demo/gguf/finetuned_dispenser_fp16.gguf \\
        --tokenizer-dir releases/functiongemma-270m/002-dispenser-demo/merged

    # HF checkpoint eval (BF16; requires torch + transformers + cuda)
    uv run python scripts/dispenser_demo/eval/eval_holdout.py \\
        --checkpoint releases/functiongemma-270m/002-dispenser-demo/merged
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from gemma_tools.dispenser_demo.dataset import load_jsonl  # type: ignore[import-untyped]

DEFAULT_HOLDOUT = Path("data/dispenser_demo/dataset_v1/val.jsonl")
DEFAULT_JOB_DESCRIPTION = Path(
    "releases/functiongemma-270m/002-dispenser-demo/distil/job_description.json"
)
# Plan §9.1 step 1.6 gate.
PASS_RATE_BAR = 0.90

# Wrapping template Distil uses at inference (mirrors the bundled
# model_client.py SYSTEM_PROMPT layout — kept here verbatim so the eval is
# byte-equivalent to the deployed chat path). The `{task_description}` slot
# is filled from `job_description.json` at runtime; this avoids hardcoding
# the routing rules in two places and keeps the alignment drift-free.
_DISTIL_SYSTEM_TEMPLATE = """You are a tool-calling model working on:
<task_description>{task_description}</task_description>

Respond to the conversation history by generating an appropriate tool call that satisfies the user request. Generate only the tool call according to the provided tool schema, do not generate anything else. Always respond with a tool call.

"""


def load_distil_prompt_setup(jd_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return `(SYSTEM_PROMPT, TOOLS)` from `job_description.json`.

    Matches the structure the bundled `model_client.py` uses at inference —
    the student was trained against this exact wrapping, so the eval must
    use it too. (My earlier eval pass with `SYSTEM_TRIGGER` instead of the
    task_description-wrapping prompt got 70 %; this path gets 100 %. See
    `docs/bench-notes/dispenser-demo/2026-05-11_dispenser-eval-*.md`.)
    """
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    system_content = _DISTIL_SYSTEM_TEMPLATE.format(
        task_description=jd["task_description"]
    )
    system_prompt = [{"role": "system", "content": system_content}]
    tools = list(jd.get("tools", []))
    return system_prompt, tools


class Equivalence(StrEnum):
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class GoldTrace:
    """Reference behaviour for one holdout row.

    `tool_calls` is the flattened sequence from the FIRST assistant turn (the
    eval inference path generates a single assistant response from the user
    prompt — comparing against later turns would systematically fail). Every
    dispenser_demo row has exactly one tool call in that first turn.

    `assistant_text` is the post-`</think>` body of the LAST assistant
    message — for our 5-message rows this is the canned NL response. Kept
    for diagnostic reporting only; the pass-rate gate is the tool-call shape.
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
        return self.n_match / self.n if self.n else 0.0

    @property
    def bar_pass(self) -> bool:
        return self.pass_rate >= PASS_RATE_BAR


# --------------------------------------------------------------------------
# Gold-trace extraction (mirrors FG; no refusal special case).
# --------------------------------------------------------------------------


def _extract_assistant_text(content: str) -> str:
    """Return the post-`</think>` tail (NL answer) or "" if there is none.

    Matches the seed validator's assistant-content shape rule.
    """
    close = content.find("</think>")
    if close == -1:
        return content
    tail = content[close + len("</think>"):]
    return tail[1:] if tail.startswith("\n") else tail


def extract_gold_trace(row: dict[str, Any]) -> GoldTrace:
    """Pull the FIRST assistant turn's tool_calls + the last NL answer."""
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
    """Deep case-fold string values for case-insensitive equivalence.

    The dispenser_demo `refuse_out_of_scope.reason` enum is lowercase ASCII
    (`health_advice` / `off_topic`); case-folding is defensive against the
    model emitting `Off_Topic` or similar. Per the iter-002 judge instructions
    (`releases/functiongemma-270m/002-dispenser-demo/distil/job_description.json`),
    reason equivalence is exact post-case-fold.
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
    - ``MATCH`` — same tool names in same order AND every `arguments` dict
      deep-equal after case-fold.
    - ``PARTIAL`` — same set of names but at least one `arguments` differs.
      Reported for diagnostics; does NOT count toward pass_rate.
    - ``MISMATCH`` — names differ, OR predicted has no calls when gold has
      one (the dispenser-demo failure mode the trained student must not
      reproduce — empty predictions were the iter-002 teacher's stochastic
      flaw).

    `category` is accepted (unused) for FG-eval signature parity, so a future
    cross-eval can swap the two scorers without re-plumbing the caller.
    """
    del category  # FG-eval signature parity; refusal in dispenser_demo IS a tool call.
    pred = list(predicted_calls)
    gold = list(gold_calls)

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

    for p, g in zip(pred, gold, strict=True):
        if _norm_args(p.get("arguments")) != _norm_args(g.get("arguments")):
            return Equivalence.PARTIAL
    return Equivalence.MATCH


# --------------------------------------------------------------------------
# Inference seams — HF (transformers) + GGUF (llama-cpp-python).
# --------------------------------------------------------------------------


# Module-level caches keyed by resolved path; without these we'd reload the
# 512 MB safetensors per row.
_INFERENCE_CACHE: dict[Path, tuple[Any, Any]] = {}
_INFERENCE_CACHE_GGUF: dict[tuple[Path, Path, int], tuple[Any, Any]] = {}

# FunctionGemma wire format — shared with iter-001's parser. The model emits
# `<start_function_call>call: NAME{key: value, ...}<end_function_call>`.
# Tolerated drift (from FG iter-001): some quantized variants emit `call NAME`
# (space) instead of `call:NAME`; both patterns are accepted.
_FG_CALL_RE = re.compile(
    r"<start_function_call>\s*call[:\s]\s*(\w+)\s*\{(.*?)\}\s*<end_function_call>",
    re.DOTALL,
)
_FG_ARG_RE = re.compile(
    r"(\w+)\s*:\s*(?:<escape>(.*?)<escape>|([^,}]*))",
    re.DOTALL,
)


def _parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``<start_function_call>...<end_function_call>`` blocks."""
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
    resolved = checkpoint_path.resolve()
    if resolved in _INFERENCE_CACHE:
        return _INFERENCE_CACHE[resolved]
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


def _load_inference_gguf(
    gguf_path: Path, tokenizer_dir: Path, n_ctx: int = 4096,
) -> tuple[Any, Any]:
    key = (gguf_path.resolve(), tokenizer_dir.resolve(), int(n_ctx))
    if key in _INFERENCE_CACHE_GGUF:
        return _INFERENCE_CACHE_GGUF[key]
    from llama_cpp import Llama  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    _INFERENCE_CACHE_GGUF[key] = (llm, tokenizer)
    return llm, tokenizer


def run_inference(
    checkpoint_path: Path,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Render chat template, greedy-decode, parse FG wire format."""
    import torch  # type: ignore[import-not-found]
    model, tokenizer = _load_inference(checkpoint_path)

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template returned {type(prompt).__name__}")
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
    return {
        "tool_calls": _parse_function_calls(output_text),
        "assistant_text": _extract_assistant_text(output_text),
        "raw_output": output_text,
    }


def run_inference_gguf(
    gguf_path: Path,
    tokenizer_dir: Path,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    n_ctx: int = 4096,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Greedy decode against a GGUF; same return shape as `run_inference`."""
    llm, tokenizer = _load_inference_gguf(gguf_path, tokenizer_dir, n_ctx=n_ctx)
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template returned {type(prompt).__name__}")
    prompt = prompt.removeprefix("<bos>")
    llm.reset()
    out = llm(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        echo=False,
        stop=["<end_function_call>", "<end_of_turn>"],
    )
    text = out["choices"][0]["text"]  # type: ignore[index]
    if text and "<start_function_call>" in text and "<end_function_call>" not in text:
        text = text + "<end_function_call>"
    return {
        "tool_calls": _parse_function_calls(text),
        "assistant_text": _extract_assistant_text(text),
        "raw_output": text,
    }


# --------------------------------------------------------------------------
# Aggregation + rendering.
# --------------------------------------------------------------------------


def aggregate_by_category(
    verdicts: list[tuple[str, Equivalence]],
) -> list[CategoryStats]:
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
    if not stats:
        return "_no rows scored._\n"
    overall_match = sum(s.n_match for s in stats)
    overall_n = sum(s.n for s in stats)
    overall_rate = overall_match / overall_n if overall_n else 0.0
    all_bar_pass = all(s.bar_pass for s in stats)
    header = [
        f"**Overall pass rate**: {overall_match}/{overall_n} "
        f"({overall_rate * 100:.1f}%).  "
        f"**All categories ≥ {int(PASS_RATE_BAR * 100)} %**: "
        f"{'YES' if all_bar_pass else 'NO'}.",
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


def render_failure_block(failures: list[dict[str, Any]]) -> str:
    """Per-row failure detail — what failed, what was predicted."""
    if not failures:
        return ""
    lines = [
        "",
        "## Per-row failures",
        "",
        "| row_id | category | gold | predicted | raw output (truncated) |",
        "|---|---|---|---|---|",
    ]
    for f in failures:
        lines.append(
            f"| {f['row_id']} | {f['category']} | "
            f"`{f['gold']}` | `{f['predicted']}` | `{f['raw'][:80]}{'…' if len(f['raw'])>80 else ''}` |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def _print_category_counts(holdout_path: Path) -> int:
    rows = list(load_jsonl(holdout_path))
    counts = Counter(r.get("category", "<missing>") for r in rows)
    print(f"holdout: {holdout_path} ({len(rows)} rows)")
    print(f"{'category':<30} count")
    print("-" * 38)
    for cat in sorted(counts):
        print(f"{cat:<30} {counts[cat]}")
    return 0


def _dry_run(holdout_path: Path) -> int:
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
        prog="eval-dispenser-holdout",
        description=(
            f"Score the iter-002 dispenser-demo tuned student against the val "
            f"holdout. Acceptance: per-category pass-rate ≥ {int(PASS_RATE_BAR * 100)}% "
            "(plan §9.1 step 1.6)."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to merged HF checkpoint. Mutually exclusive with --gguf.",
    )
    p.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Path to a GGUF file. Mutually exclusive with --checkpoint.",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("releases/functiongemma-270m/002-dispenser-demo/merged"),
        help="HF tokenizer dir for chat-template rendering (used with --gguf).",
    )
    p.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="llama-cpp-python n_ctx (used with --gguf). Default: 4096.",
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
        help="Output Markdown report. Default: docs/bench-notes/dispenser-demo/<today>_dispenser-eval-<seam>.md.",
    )
    p.add_argument(
        "--list-categories",
        action="store_true",
        help="Print per-category row counts and exit. No inference.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run gold-vs-gold sanity → must be 100 %% MATCH.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Greedy-decode token budget per row. Default 256.",
    )
    p.add_argument(
        "--job-description",
        type=Path,
        default=DEFAULT_JOB_DESCRIPTION,
        help=(
            f"Path to the Distil job_description.json — used to construct the "
            f"SYSTEM_PROMPT + TOOLS the student was trained against. Default: "
            f"{DEFAULT_JOB_DESCRIPTION}."
        ),
    )
    p.add_argument(
        "--seed-as-is",
        action="store_true",
        help=(
            "Send each seed row's `messages[0]` (SYSTEM_TRIGGER) and `tools` "
            "block verbatim — bypasses Distil's task_description wrapping. "
            "Debug-only; matches an unrelated input distribution and will "
            "score systematically lower (~70 %% on val). See "
            "`docs/bench-notes/dispenser-demo/2026-05-11_dispenser-eval-*` "
            "for the failure-mode comparison."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    holdout: Path = args.holdout
    if not holdout.exists():
        print(f"ERROR holdout not found: {holdout}", file=sys.stderr)
        return 2

    if args.list_categories:
        return _print_category_counts(holdout)
    if args.dry_run:
        return _dry_run(holdout)

    if args.checkpoint is not None and args.gguf is not None:
        print("ERROR --checkpoint and --gguf are mutually exclusive.", file=sys.stderr)
        return 2
    if args.checkpoint is None and args.gguf is None:
        print(
            "ERROR one of --checkpoint or --gguf is required (omit only with "
            "--list-categories or --dry-run).",
            file=sys.stderr,
        )
        return 2
    if args.gguf is not None:
        if not args.gguf.exists():
            print(f"ERROR gguf not found: {args.gguf}", file=sys.stderr)
            return 2
        if not args.tokenizer_dir.exists():
            print(
                f"ERROR --tokenizer-dir not found: {args.tokenizer_dir}",
                file=sys.stderr,
            )
            return 2
        seam_label = f"gguf-{args.gguf.stem}"
    else:
        if not args.checkpoint.exists():
            print(f"ERROR checkpoint not found: {args.checkpoint}", file=sys.stderr)
            return 2
        seam_label = f"hf-{args.checkpoint.name}"

    output: Path = args.output or (
        Path("docs/bench-notes/dispenser-demo")
        / f"{datetime.date.today().isoformat()}_dispenser-eval-{seam_label}.md"
    )

    # Load Distil's system prompt + tools UNLESS --seed-as-is. The default
    # path matches the deployed model_client.py at inference; seed-as-is is
    # a debug knob that reproduces the SYSTEM_TRIGGER mismatch failure mode.
    if args.seed_as_is:
        distil_system: list[dict[str, Any]] | None = None
        distil_tools: list[dict[str, Any]] | None = None
        prompt_label = "seed-as-is"
    else:
        if not args.job_description.exists():
            print(
                f"ERROR --job-description not found: {args.job_description}",
                file=sys.stderr,
            )
            return 2
        distil_system, distil_tools = load_distil_prompt_setup(args.job_description)
        prompt_label = f"distil-prompt({args.job_description.name})"

    rows = list(load_jsonl(holdout))
    verdicts: list[tuple[str, Equivalence]] = []
    failures: list[dict[str, Any]] = []
    print(
        f"=== evaluating {len(rows)} rows ({seam_label}, prompt={prompt_label}) ===",
        file=sys.stderr,
    )
    for i, raw in enumerate(rows):
        gold = extract_gold_trace(raw)
        # Inference input. Default: Distil's SYSTEM_PROMPT + the seed's user
        # turn(s) (no seed's system message, no seed's tools[] block).
        # `--seed-as-is`: seed's pre-tool-call prefix verbatim.
        if distil_system is None:
            prefix: list[dict[str, Any]] = []
            for m in raw.get("messages", []):
                if m.get("role") == "assistant":
                    break
                prefix.append(m)
            tools = raw.get("tools")
        else:
            prefix = list(distil_system)
            for m in raw.get("messages", []):
                if m.get("role") == "user":
                    prefix.append({"role": "user", "content": m.get("content", "")})
                elif m.get("role") == "assistant":
                    break
            tools = distil_tools

        if args.gguf is not None:
            result = run_inference_gguf(
                args.gguf, args.tokenizer_dir, prefix, tools,
                n_ctx=args.n_ctx, max_new_tokens=args.max_new_tokens,
            )
        else:
            result = run_inference(
                args.checkpoint, prefix, tools, max_new_tokens=args.max_new_tokens,
            )

        v = tool_call_equivalent(
            result["tool_calls"], list(gold.tool_calls), category=gold.category
        )
        verdicts.append((gold.category, v))
        print(
            f"  [{i + 1:>2}/{len(rows)}] {gold.row_id} ({gold.category}) → {v.value}",
            file=sys.stderr,
        )
        if v is not Equivalence.MATCH:
            failures.append({
                "row_id": gold.row_id,
                "category": gold.category,
                "gold": json.dumps(list(gold.tool_calls), ensure_ascii=False),
                "predicted": json.dumps(result["tool_calls"], ensure_ascii=False),
                "raw": result.get("raw_output", ""),
            })

    stats = aggregate_by_category(verdicts)
    table = render_summary_table(stats)
    failure_block = render_failure_block(failures)
    body = (
        f"# Dispenser-demo iter-002 holdout eval — {datetime.date.today().isoformat()}\n"
        f"\n"
        f"- **Seam**: `{seam_label}`\n"
        f"- **Holdout**: `{holdout}` ({len(rows)} rows)\n"
        f"- **Pass-rate gate**: ≥ {int(PASS_RATE_BAR * 100)}% per category "
        f"(plan §9.1 step 1.6).\n"
        f"\n"
        f"## Aggregate\n\n"
        f"{table}\n"
        f"{failure_block}"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(f"\nwrote: {output}")
    print(table)
    all_pass = all(s.bar_pass for s in stats)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
