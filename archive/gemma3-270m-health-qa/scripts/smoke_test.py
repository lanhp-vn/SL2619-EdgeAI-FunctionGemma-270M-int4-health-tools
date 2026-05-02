#!/usr/bin/env python3
"""
T5 side-by-side smoke: base BF16 vs merged BF16 on 5 bench prompts.

Loads a pre-rendered prompt bundle (host-rendered via
`gemma_tools.prompt_composer.render_system_prompt(...)` so the §4 directive
template + YAML + `now=date(2026,4,25)` match the training-time prompt shape
verbatim), then runs base and merged models sequentially (free VRAM between
loads) on each prompt with deterministic generation. Emits a JSONL row
per prompt and a markdown summary table with regex pass/fail derived from
each prompt's `pass_pattern` (auto-signal floor; final verdict is qualitative
per plan §10.3).

Single source of truth for the prompt format is `scripts/finetune.py`
(`_to_prompt_completion` + `apply_chat_template(..., add_generation_prompt=True)`)
— the smoke replicates the exact same chat-template invocation so we measure
the SFT delta, not a tokenization artifact.

Deploy:
    scp scripts/t5_smoke.py nouslogic-server:~/sl2619-finetune/t5_smoke.py
    scp /tmp/t5_smoke_bundle.json nouslogic-server:~/sl2619-finetune/t5_smoke_bundle.json

Usage on server (real run):
    ssh -t nouslogic-server 'cd ~/sl2619-finetune && source .venv/bin/activate && \
        python t5_smoke.py --bundle ./t5_smoke_bundle.json \
            --base google/gemma-3-270m-it --merged ./merged_v1 \
            --out-dir ./logs'

Dry-run on host (no torch wheel needed):
    python scripts/t5_smoke.py --bundle /tmp/t5_smoke_bundle.json \
        --dry-run --out-dir /tmp
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import UTC, datetime
from typing import Any


def _lazy_torch_imports() -> tuple[Any, Any, Any]:
    """Import torch + transformers only when not in --dry-run.

    Lets the host-side dry-run smoke-test plumbing (bundle parse, regex,
    JSONL/MD shape) without a cu128 torch wheel installed in the host venv.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, AutoModelForCausalLM, AutoTokenizer


def _flags_to_int(flags: str) -> int:
    """Map prompts.yaml `pattern_flags` chars to re module flags."""
    f = 0
    if "i" in flags:
        f |= re.IGNORECASE
    if "s" in flags:
        f |= re.DOTALL
    if "m" in flags:
        f |= re.MULTILINE
    return f


def _passes(text: str, pattern: str, flags: str) -> bool:
    return re.search(pattern, text, _flags_to_int(flags)) is not None


def _run_model(
    label: str,
    model_path: str,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    print(f"\n=== {label} :: {model_path} ===", flush=True)

    if dry_run:
        # Stub: a string that matches the trivial pass patterns for both
        # models so the markdown/JSONL plumbing runs end-to-end. The merged
        # stub deliberately differs from base so the delta column on the
        # markdown table is non-trivial.
        stub_template = (
            "stub({label}) heart 72 bpm; lisinopril metformin "
            "penicillin shellfish; not in record; health record"
        )
        results: list[dict[str, Any]] = []
        for p in prompts:
            text = stub_template.format(label=label)
            results.append({
                "id": p["id"],
                "class": p["class"],
                "utterance": p["utterance"],
                "completion": text,
                "passed": _passes(text, p["pass_pattern"], p["pattern_flags"]),
                "gen_seconds": 0.0,
                "prompt_tokens": 0,
                "new_tokens": 0,
            })
        return results

    torch, AutoModelForCausalLM, AutoTokenizer = _lazy_torch_imports()  # noqa: N806

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    eos_id = tok.eos_token_id

    results = []
    try:
        for p in prompts:
            messages = [{"role": "user", "content": p["user_text"]}]
            # transformers 5.x apply_chat_template returns a BatchEncoding
            # when return_dict=True. Without return_dict the return type is
            # a bare tensor in some versions and a BatchEncoding in others —
            # T4 smoke hit that trap. Pin to dict explicitly.
            enc = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            prompt_len = int(input_ids.shape[1])

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=eos_id,
                )
            dt = time.time() - t0

            new_ids = out[0][prompt_len:]
            text = tok.decode(new_ids, skip_special_tokens=True).strip()
            new_tokens = int(new_ids.shape[0])

            print(f"  {p['id']:<4} ({p['class']:<14}) "
                  f"[{new_tokens:>3} tok / {dt:5.2f}s]", flush=True)
            preview = text.replace("\n", " ⏎ ")[:200]
            print(f"     -> {preview}", flush=True)

            results.append({
                "id": p["id"],
                "class": p["class"],
                "utterance": p["utterance"],
                "completion": text,
                "passed": _passes(text, p["pass_pattern"], p["pattern_flags"]),
                "gen_seconds": round(dt, 3),
                "prompt_tokens": prompt_len,
                "new_tokens": new_tokens,
            })
    finally:
        del model
        del tok
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _emit(
    out_dir: str,
    base: list[dict[str, Any]],
    merged: list[dict[str, Any]],
    bundle: dict[str, Any],
    dry_run: bool,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S")
    suffix = "-dryrun" if dry_run else ""
    jsonl_path = os.path.join(out_dir, f"t5-smoke-{ts}{suffix}.jsonl")
    md_path = os.path.join(out_dir, f"t5-smoke-{ts}{suffix}.md")

    rows = []
    for p, b, m in zip(bundle["prompts"], base, merged, strict=True):
        rows.append({
            "id": p["id"],
            "class": p["class"],
            "utterance": p["utterance"],
            "pass_pattern": p["pass_pattern"],
            "pattern_flags": p["pattern_flags"],
            "base": {
                "completion": b["completion"],
                "passed": b["passed"],
                "gen_seconds": b["gen_seconds"],
                "new_tokens": b["new_tokens"],
                "prompt_tokens": b["prompt_tokens"],
            },
            "merged": {
                "completion": m["completion"],
                "passed": m["passed"],
                "gen_seconds": m["gen_seconds"],
                "new_tokens": m["new_tokens"],
                "prompt_tokens": m["prompt_tokens"],
            },
        })

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    base_pass = sum(1 for r in rows if r["base"]["passed"])
    merged_pass = sum(1 for r in rows if r["merged"]["passed"])
    delta = merged_pass - base_pass

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# T5 side-by-side smoke — base bf16 vs merged bf16\n\n")
        f.write(f"- generated: `{ts}`\n")
        f.write(f"- bundle `now` (DATE: slot in directive): "
                f"`{bundle['now_iso']}` (matches training)\n")
        f.write(f"- base path: `{bundle.get('base_path_hint', 'google/gemma-3-270m-it')}`\n")
        f.write(f"- merged path hint: `{bundle.get('merged_path_hint', './merged_v1')}`\n")
        f.write(f"- generation: `do_sample=False`, "
                f"`max_new_tokens={bundle.get('max_new_tokens_hint', 96)}`\n")
        f.write(f"- base regex pass: **{base_pass}/{len(rows)}**\n")
        f.write(f"- merged regex pass: **{merged_pass}/{len(rows)}**\n")
        f.write(f"- delta (merged - base): **{delta:+d}**\n")
        if dry_run:
            f.write("- ⚠ **DRY-RUN** — completions are stubbed; not a behavioral signal.\n")
        f.write("\n## Per-prompt verdict (regex auto-signal)\n\n")
        f.write("| ID | class | base | merged | utterance |\n")
        f.write("|----|-------|:----:|:------:|-----------|\n")
        for r in rows:
            b_mark = "✓" if r["base"]["passed"] else "✗"
            m_mark = "✓" if r["merged"]["passed"] else "✗"
            f.write(f"| {r['id']} | {r['class']} | {b_mark} | {m_mark} | "
                    f"{r['utterance']} |\n")
        f.write("\n## Per-prompt outputs\n")
        for r in rows:
            f.write(f"\n### {r['id']} — {r['class']}\n")
            f.write(f"- utterance: `{r['utterance']}`\n")
            f.write(f"- pass pattern (flags=`{r['pattern_flags']}`): "
                    f"`{r['pass_pattern']}`\n")
            f.write(f"\n**base** ({r['base']['gen_seconds']}s, "
                    f"{r['base']['new_tokens']} new tokens) — "
                    f"{'✓' if r['base']['passed'] else '✗'} regex:\n\n")
            f.write(f"```\n{r['base']['completion']}\n```\n")
            f.write(f"\n**merged** ({r['merged']['gen_seconds']}s, "
                    f"{r['merged']['new_tokens']} new tokens) — "
                    f"{'✓' if r['merged']['passed'] else '✗'} regex:\n\n")
            f.write(f"```\n{r['merged']['completion']}\n```\n")

    return jsonl_path, md_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="T5 side-by-side smoke (base vs merged BF16)",
    )
    p.add_argument("--bundle", required=True,
                   help="Pre-rendered prompt bundle JSON")
    p.add_argument("--base", default="google/gemma-3-270m-it",
                   help="Base HF id or path")
    p.add_argument("--merged", default="./merged_v1",
                   help="Merged HF dir")
    p.add_argument("--out-dir", default="./logs",
                   help="Where to write t5-smoke-*.jsonl/.md")
    p.add_argument("--max-new-tokens", type=int, default=96,
                   help="Per advisor: 96 caps S1 summarization without chop; "
                        "others finish well under 30")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip torch — stub completions for plumbing test")
    args = p.parse_args()

    with open(args.bundle, encoding="utf-8") as f:
        bundle = json.load(f)

    # Stamp args into bundle so the markdown header records what actually ran.
    bundle["base_path_hint"] = args.base
    bundle["merged_path_hint"] = args.merged
    bundle["max_new_tokens_hint"] = args.max_new_tokens

    prompts = bundle["prompts"]
    print(f"Loaded {len(prompts)} prompts from bundle "
          f"(now={bundle['now_iso']}, dry_run={args.dry_run})", flush=True)

    base = _run_model("base", args.base, prompts, args.max_new_tokens, args.dry_run)
    merged = _run_model("merged", args.merged, prompts, args.max_new_tokens, args.dry_run)

    jsonl_path, md_path = _emit(args.out_dir, base, merged, bundle, args.dry_run)
    base_pass = sum(1 for r in base if r["passed"])
    merged_pass = sum(1 for r in merged if r["passed"])
    print("\nT5 smoke complete.")
    print(f"  JSONL : {jsonl_path}")
    print(f"  MD    : {md_path}")
    print(f"  base   pass: {base_pass}/{len(prompts)}")
    print(f"  merged pass: {merged_pass}/{len(prompts)}")
    print(f"  delta      : {merged_pass - base_pass:+d}")


if __name__ == "__main__":
    main()
