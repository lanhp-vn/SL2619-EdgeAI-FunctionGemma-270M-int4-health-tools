"""Post-bench scorer — host-only.

Reads the JSONL output of `bench-prompt`, applies each prompt's
`pass_pattern` (regex) per `pattern_flags` to `response_text`, and emits
a Markdown summary that drops into `docs/bench/<date>_gemma3-*.md`.

Regex pass/fail is NECESSARY-BUT-NOT-SUFFICIENT per plan §6.2. A human
rubric (0-3) sits atop these results — the scorer just surfaces every
prompt with its computed PASS/FAIL + latency + response so the human
doesn't have to re-parse JSONL by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from gemma_tools._legacy.bench_prompt import (
    PromptSpec,
    load_prompt_suite,
    score_response,
)

__all__ = ["ScoredRow", "load_jsonl", "main", "render_markdown_summary",
           "score_response", "score_sweep"]


@dataclass(frozen=True, slots=True)
class ScoredRow:
    prompt_id: str
    prompt_class: str
    prompt_text: str
    pass_pattern: str
    pattern_flags: str
    response_text: str
    passed_regex: bool
    tokens_generated: int
    wall_ms_total: float
    wall_ms_ttft_external: float
    tokens_per_sec: float
    peak_rss_mb: float
    cma_free_kb_during: int
    error: str | None


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSONL file; blank lines are skipped."""
    out: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                out.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON — {e}") from e
    return out


def score_sweep(jsonl_path: Path, prompts_path: Path) -> list[ScoredRow]:
    """Join the JSONL rows with the prompt suite by id; produce one
    `ScoredRow` per JSONL entry.

    Missing ids (row references a prompt that isn't in prompts.yaml) raise
    — that's a contract break between the bench harness and the scorer.
    """
    suite: dict[str, PromptSpec] = {p.id: p for p in load_prompt_suite(prompts_path)}
    rows = load_jsonl(jsonl_path)
    out: list[ScoredRow] = []
    for row in rows:
        pid = row["prompt_id"]
        assert isinstance(pid, str)
        spec = suite.get(pid)
        if spec is None:
            raise ValueError(f"prompt id {pid!r} present in JSONL but not in {prompts_path}")
        response_text = row["response_text"]
        assert isinstance(response_text, str)
        timing = row["timing"]
        assert isinstance(timing, dict)
        passed = score_response(spec.pass_pattern, spec.pattern_flags, response_text)
        error = row.get("error")
        assert error is None or isinstance(error, str)
        peak_rss = row["peak_rss_mb"]
        cma_during = row["cma_free_kb_during"]
        assert isinstance(peak_rss, (int, float))
        assert isinstance(cma_during, int)
        tokens_generated = timing["tokens_generated"]
        wall_ms_total = timing["wall_ms_total"]
        ttft_external = timing["wall_ms_ttft_external"]
        tokens_per_sec = timing["tokens_per_sec"]
        assert isinstance(tokens_generated, int)
        assert isinstance(wall_ms_total, (int, float))
        assert isinstance(ttft_external, (int, float))
        assert isinstance(tokens_per_sec, (int, float))
        out.append(
            ScoredRow(
                prompt_id=pid,
                prompt_class=spec.prompt_class,
                prompt_text=spec.text,
                pass_pattern=spec.pass_pattern,
                pattern_flags=spec.pattern_flags,
                response_text=response_text,
                passed_regex=passed,
                tokens_generated=tokens_generated,
                wall_ms_total=float(wall_ms_total),
                wall_ms_ttft_external=float(ttft_external),
                tokens_per_sec=float(tokens_per_sec),
                peak_rss_mb=float(peak_rss),
                cma_free_kb_during=cma_during,
                error=error,
            )
        )
    return out


def render_markdown_summary(rows: list[ScoredRow]) -> str:
    """Emit the Markdown table block for `<date>_gemma3-*.md`.

    Format is stable; the human rubric column is pre-populated with "—"
    so the reviewer just overwrites it in-place during §6.3 step 4."""
    if not rows:
        return "_no rows scored._\n"
    total = len(rows)
    passed = sum(1 for r in rows if r.passed_regex and r.error is None)
    errors = sum(1 for r in rows if r.error is not None)
    header = [
        f"**Regex pass rate**: {passed}/{total} ({passed * 100 / total:.0f}%).  "
        f"**Errors**: {errors}.",
        "",
        "| id | class | regex | tok | wall_ms | ttft_ms | tok/s | rubric 0-3 | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    body: list[str] = []
    for r in rows:
        status = "ERR" if r.error else ("PASS" if r.passed_regex else "FAIL")
        note = (r.error or "").replace("|", "\\|")
        body.append(
            f"| {r.prompt_id} | {r.prompt_class} | {status} | "
            f"{r.tokens_generated} | {r.wall_ms_total:.0f} | "
            f"{r.wall_ms_ttft_external:.0f} | {r.tokens_per_sec:.1f} | — | {note} |"
        )
    return "\n".join(header + body) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench-eval",
        description=(
            "Score a bench-prompt JSONL sweep against the canonical "
            "prompts.yaml. Emits a Markdown summary table to stdout."
        ),
    )
    p.add_argument("--jsonl", type=Path, required=True, help="Path to sweep JSONL output.")
    p.add_argument("--prompts", type=Path, required=True, help="Path to prompts.yaml.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional file sink for the Markdown table; default stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    rows = score_sweep(args.jsonl, args.prompts)
    markdown = render_markdown_summary(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
