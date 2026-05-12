#!/usr/bin/env python3
"""Generate `prompt-prefix.txt` and `prompt-suffix.txt` for the dispenser-demo
iter-002 on-board deployment. Mirrors iter-001's
`scripts/functiongemma/data/gen_prompt_templates.py` pattern — the board's
`llama-completion` doesn't carry an HF tokenizer, so we pre-render the
chat-template prefix + suffix on host and ship them as flat text files.

Single source of truth for SYSTEM_PROMPT + TOOLS: this script reads
`releases/functiongemma-270m/002-dispenser-demo/distil/job_description.json`
at generate-time. The wrapping is the same Distil-published format used in
`scripts/dispenser_demo/eval/eval_holdout.py` and the bundled `model_client.py`
— see the "2026-05-12 (Phase 1.6 eval)" decisions-log entry for why this
matters.

Output layout (consumed by the board-side `ask_board.sh`):
    /mnt/sdcard/models/dispenser-demo-002/prompt-prefix.txt
    /mnt/sdcard/models/dispenser-demo-002/prompt-suffix.txt
    /mnt/sdcard/models/dispenser-demo-002/finetuned_dispenser_q4_0.gguf

Usage (host):
    uv run python scripts/dispenser_demo/data/gen_prompt_templates.py \\
        --tokenizer releases/functiongemma-270m/002-dispenser-demo/merged \\
        --output-dir /tmp/dispenser_board_files/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same wrapping template as `scripts/dispenser_demo/eval/eval_holdout.py`.
# Keep these in sync — `tests/dispenser_demo/test_distil_alignment.py` does
# NOT currently gate this (it only gates tool registry alignment, not the
# wrapping). If the wrap ever changes upstream, both files need editing.
_DISTIL_SYSTEM_TEMPLATE = """You are a tool-calling model working on:
<task_description>{task_description}</task_description>

Respond to the conversation history by generating an appropriate tool call that satisfies the user request. Generate only the tool call according to the provided tool schema, do not generate anything else. Always respond with a tool call.

"""


def load_distil_prompt_setup(jd_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    jd = json.loads(jd_path.read_text(encoding="utf-8"))
    system_content = _DISTIL_SYSTEM_TEMPLATE.format(
        task_description=jd["task_description"]
    )
    system_prompt = [{"role": "system", "content": system_content}]
    tools = list(jd.get("tools", []))
    return system_prompt, tools


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate prompt-prefix.txt + prompt-suffix.txt for dispenser-demo iter-002."
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help=(
            "HF model directory with chat_template.jinja "
            "(e.g. releases/functiongemma-270m/002-dispenser-demo/merged)."
        ),
    )
    p.add_argument(
        "--job-description",
        type=Path,
        default=Path(
            "releases/functiongemma-270m/002-dispenser-demo/distil/job_description.json"
        ),
        help="Path to the Distil job_description.json (single source of truth for SYSTEM_PROMPT + TOOLS).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write prompt-prefix.txt + prompt-suffix.txt.",
    )
    args = p.parse_args(argv)

    if not args.tokenizer.exists():
        print(f"tokenizer dir not found: {args.tokenizer}", file=sys.stderr)
        return 2
    if not args.job_description.exists():
        print(f"job_description.json not found: {args.job_description}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, tools = load_distil_prompt_setup(args.job_description)

    # Lazy import to keep ruff/mypy fast and not pay the 500 MB transformers
    # cost on host CI when this script is unused.
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    print(f"[gen] loading tokenizer from {args.tokenizer}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))

    dummy_user = "<PLACEHOLDER_USER_MESSAGE>"
    messages = [*system_prompt, {"role": "user", "content": dummy_user}]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(full_prompt, str):
        raise TypeError(f"apply_chat_template returned {type(full_prompt).__name__}")
    full_prompt = full_prompt.removeprefix("<bos>")

    idx = full_prompt.find(dummy_user)
    if idx < 0:
        print("ERROR: placeholder not found in rendered prompt", file=sys.stderr)
        print(f"Full prompt:\n{full_prompt}", file=sys.stderr)
        return 2

    prefix = full_prompt[:idx]
    suffix = full_prompt[idx + len(dummy_user):]

    prefix_path = args.output_dir / "prompt-prefix.txt"
    suffix_path = args.output_dir / "prompt-suffix.txt"
    prefix_path.write_text(prefix, encoding="utf-8")
    suffix_path.write_text(suffix, encoding="utf-8")

    print(f"[gen] prefix: {len(prefix)} bytes → {prefix_path}", file=sys.stderr)
    print(f"[gen] suffix: {len(suffix)} bytes → {suffix_path}", file=sys.stderr)
    print("[gen] done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
