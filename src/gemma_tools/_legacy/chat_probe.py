"""One-shot Gemma 3 probe — load model, answer one question, stream output.

Single-question alternative to `bench_prompt.py`'s 15-prompt sweep. Used to
iterate on prompt design under §1.4 of `11-testing-verification.md` ("host
first, board only when necessary; small wins"): prove the plumbing with one
question + a minimal YAML slice before expanding scope.

Prompt shape:
  [vendor warmup sys prompt (untouched by default)]
  <start_of_turn>user
  [optional --preface text]
  YAML:
  <field-sliced YAML block>

  <question>
  <end_of_turn>
  <start_of_turn>model
  ...

Streams each decoded chunk to stdout as it arrives — `bench_prompt.py`
accumulates internally, which was a diagnosability regression versus the
Phase B interactive demo.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from gemma_tools.health_table import HealthTable, load_health_table

# region: pure helpers

def slice_health_yaml(health: HealthTable, field_path: str) -> str:
    """Return a YAML-block rendering of the field named by a dotted path.

    `field_path` navigates `asdict(health)` dict-by-dict. An empty path
    returns an empty string (caller can skip injection). Raises `KeyError`
    with a path-locator if the field doesn't resolve — silent fallback
    would inject an empty slice and mask misconfiguration.
    """
    if not field_path:
        return ""
    cur: Any = asdict(health)
    for part in field_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(
                f"field path {field_path!r}: part {part!r} not in current level; "
                f"available keys={sorted(cur.keys()) if isinstance(cur, dict) else '(not a dict)'}"
            )
        cur = cur[part]
    nested: Any = cur
    for part in reversed(field_path.split(".")):
        nested = {part: nested}
    return yaml.safe_dump(
        nested, sort_keys=False, default_flow_style=False, allow_unicode=True
    ).rstrip()


def compose_probe_prompt(
    health: HealthTable, field_path: str, question: str, preface: str = ""
) -> str:
    """Build the user-turn body: [preface]? + [YAML slice]? + [question].

    Blocks separated by blank lines (`\\n\\n`) so the model sees a clear
    structural break. The vendor runner wraps this in the Gemma chat-
    template markers — do NOT prepend `<start_of_turn>` here.
    """
    slice_yaml = slice_health_yaml(health, field_path)
    blocks: list[str] = []
    if preface:
        blocks.append(preface)
    if slice_yaml:
        blocks.append(f"YAML:\n{slice_yaml}")
    blocks.append(question)
    return "\n\n".join(blocks)


@contextlib.contextmanager
def patched_sys_prompt(runner_mod: ModuleType, override: str) -> Iterator[None]:
    """Monkey-patch the vendor runner's `DEFAULT_SYS_PROMPT` for the duration.

    Duplicate of the context manager in `bench_prompt.py`; kept local so
    this probe script is self-contained and touches nothing else.
    """
    original = getattr(runner_mod, "DEFAULT_SYS_PROMPT")  # noqa: B009
    setattr(runner_mod, "DEFAULT_SYS_PROMPT", override)  # noqa: B010
    try:
        yield
    finally:
        setattr(runner_mod, "DEFAULT_SYS_PROMPT", original)  # noqa: B010

# endregion

# region: vendor runner adapter (minimal — this is NOT bench_prompt.py)


RunnerFactory = Callable[[], "tuple[ModuleType, type]"]


def _default_runner_factory() -> tuple[ModuleType, type]:
    """Import vendor `runner` module from sys.path; raise `ImportError`
    with guidance on host where torq.runtime is absent."""
    try:
        mod = importlib.import_module("runner")
    except ImportError as e:
        raise ImportError(
            f"vendor `runner` not importable: {e}. "
            f"On board, run from the Gemma model dir so sys.path[0] includes runner.py."
        ) from e
    return mod, mod.Gemma3Static

# endregion

# region: main entry


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chat-probe",
        description=(
            "Single-question Gemma 3 probe. Loads the VMFB, answers one "
            "question from --question, streams chunks to stdout, reports "
            "per-run timing on stderr."
        ),
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Directory with model.vmfb + tokenizer.json + config.json.")
    p.add_argument("--health-table", type=Path, required=True,
                   help="Path to health_table_v1.yaml.")
    p.add_argument("--question", type=str, required=True,
                   help="The user question to send to the model.")
    p.add_argument("--yaml-field", type=str, default="",
                   help="Dotted field path into the health table "
                        "(e.g. 'vitals.heart_rate_bpm'). Empty = no YAML.")
    p.add_argument("--preface", type=str, default="",
                   help="Text to prepend inside the user turn before YAML + question. "
                        "Empty = nothing added.")
    p.add_argument("--patched-sys-prompt", type=str, default=None,
                   help="If set, monkey-patches the vendor runner's "
                        "DEFAULT_SYS_PROMPT for the load. Empty string = disable "
                        "vendor warmup. If omitted, vendor default is used.")
    p.add_argument("--max-gen-tokens", type=int, default=64,
                   help="Generation-token cap. Default 64 keeps iteration fast.")
    return p


def main(
    argv: list[str] | None = None,
    *,
    runner_factory: RunnerFactory = _default_runner_factory,
    out: Any = sys.stdout,
    err: Any = sys.stderr,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> int:
    args = _build_arg_parser().parse_args(argv)
    health = load_health_table(args.health_table)
    user_text = compose_probe_prompt(
        health, args.yaml_field, args.question, args.preface,
    )
    print(f"[prompt chars] {len(user_text)}", file=err)
    print(f"[prompt head ] {user_text[:180]!r}", file=err)

    runner_mod, gemma_cls = runner_factory()
    load_start = clock_ns()
    if args.patched_sys_prompt is not None:
        with patched_sys_prompt(runner_mod, args.patched_sys_prompt):
            impl = gemma_cls(
                model_path=str(args.model_dir / "model.vmfb"),
                instruct_model=True,
            )
    else:
        impl = gemma_cls(
            model_path=str(args.model_dir / "model.vmfb"),
            instruct_model=True,
        )
    load_ms = (clock_ns() - load_start) / 1e6
    print(f"[load ms     ] {load_ms:.0f}", file=err)

    print(">>> ", end="", flush=True, file=out)
    run_start = clock_ns()
    first_chunk_ns: int | None = None
    chunk_count = 0
    for chunk in impl.run_stream(user_text, args.max_gen_tokens):
        if first_chunk_ns is None:
            first_chunk_ns = clock_ns() - run_start
        print(chunk, end="", flush=True, file=out)
        chunk_count += 1
    total_ms = (clock_ns() - run_start) / 1e6
    ttft_ms = (first_chunk_ns / 1e6) if first_chunk_ns is not None else 0.0
    print("", flush=True, file=out)
    vendor_ttft = float(impl.time_to_first_token)
    vendor_tokens = int(impl.generated_tokens)
    print(
        f"[done        ] chunks={chunk_count} ttft_external_ms={ttft_ms:.0f} "
        f"total_ms={total_ms:.0f} vendor_ttft_ms={vendor_ttft:.0f} "
        f"vendor_tokens={vendor_tokens}",
        file=err,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# endregion
