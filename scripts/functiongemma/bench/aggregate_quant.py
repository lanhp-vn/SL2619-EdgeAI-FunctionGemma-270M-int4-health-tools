#!/usr/bin/env python3
"""Aggregate per-variant bench JSONLs into a Markdown summary table.

Reads one or more directories of bench output produced by `bench.py
--mode remote --out <dir>/<variant>.jsonl` (or `--mode local`), groups
rows by source filename, and emits a Markdown table sorted by tool-call
match rate then decode tok/s.

Usage:
    uv run python scripts/functiongemma/bench/aggregate_quant.py \
        bench/functiongemma/runs/2026-05-02-quant/stage1/ \
        --output docs/bench-notes/functiongemma/2026-05-02_quantization-sweep.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class VariantStats:
    name: str
    source: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    file_size_mib: float | None = None  # set externally if available

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if not r.get("error"))

    @property
    def n_match(self) -> int:
        # bench.py records `parsed_call.tool == expected_tool` as the OK criterion.
        return sum(
            1
            for r in self.rows
            if not r.get("error")
            and r.get("parsed_call")
            and r["parsed_call"].get("tool") == r.get("expected_tool")
        )

    @property
    def n_parsed(self) -> int:
        return sum(1 for r in self.rows if r.get("parsed_call"))

    @property
    def match_rate(self) -> float:
        return self.n_match / self.n if self.n else 0.0

    def _ok_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if not r.get("error")]

    @property
    def mean_decode_tps(self) -> float:
        ok = [r["decode_tps"] for r in self._ok_rows() if r.get("decode_tps")]
        return mean(ok) if ok else 0.0

    @property
    def mean_prompt_tps(self) -> float:
        ok = [r["prompt_tps"] for r in self._ok_rows() if r.get("prompt_tps")]
        return mean(ok) if ok else 0.0

    @property
    def mean_overall_tps(self) -> float:
        ok = [r["overall_tps"] for r in self._ok_rows() if r.get("overall_tps")]
        return mean(ok) if ok else 0.0

    @property
    def mean_wall_s(self) -> float:
        ok = [r["wall_ms_total"] / 1000.0 for r in self._ok_rows() if r.get("wall_ms_total")]
        return mean(ok) if ok else 0.0

    @property
    def mean_load_s(self) -> float:
        ok = [r["wall_ms_load"] / 1000.0 for r in self._ok_rows() if r.get("wall_ms_load")]
        return mean(ok) if ok else 0.0


def _variant_name_from_filename(p: Path) -> str:
    # Bench files are named e.g. q4_0.jsonl, q4_k_m-ctx2048.jsonl, fp16.jsonl
    return p.stem


def load_dir(d: Path, sizes: dict[str, float] | None = None) -> list[VariantStats]:
    out: list[VariantStats] = []
    for f in sorted(d.glob("*.jsonl")):
        rows: list[dict[str, Any]] = []
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        name = _variant_name_from_filename(f)
        size = (sizes or {}).get(name)
        out.append(VariantStats(name=name, source=f, rows=rows, file_size_mib=size))
    return out


def render(stats: list[VariantStats], *, title: str, sort_by_accuracy: bool = True) -> str:
    if not stats:
        return f"# {title}\n\n_no rows._\n"
    if sort_by_accuracy:
        # Primary: match rate desc, secondary: decode tok/s desc
        stats = sorted(
            stats,
            key=lambda s: (-s.match_rate, -s.mean_decode_tps),
        )
    lines = [
        f"# {title}",
        "",
        "| variant | size MiB | n | match | match% | parsed | decode tok/s | prompt tok/s | overall tok/s | mean wall s | load s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        size_str = f"{s.file_size_mib:.0f}" if s.file_size_mib is not None else "—"
        lines.append(
            f"| `{s.name}` | {size_str} | {s.n} | {s.n_match}/{s.n_ok} | "
            f"{s.match_rate * 100:.1f}% | {s.n_parsed}/{s.n} | "
            f"{s.mean_decode_tps:.2f} | {s.mean_prompt_tps:.1f} | "
            f"{s.mean_overall_tps:.1f} | {s.mean_wall_s:.2f} | {s.mean_load_s:.2f} |"
        )
    # Per-row details — useful for spotting which prompts cratered
    lines += ["", "## Per-row tool match by variant", ""]
    if stats:
        prompt_ids = sorted({r.get("prompt_id", "?") for s in stats for r in s.rows})
        header = "| prompt | " + " | ".join(f"`{s.name}`" for s in stats) + " |"
        sep = "|---|" + "---|" * len(stats)
        lines.append(header)
        lines.append(sep)
        for pid in prompt_ids:
            cells = [f"`{pid}`"]
            for s in stats:
                hit = next((r for r in s.rows if r.get("prompt_id") == pid), None)
                if hit is None:
                    cells.append("—")
                elif hit.get("error"):
                    cells.append("ERR")
                elif hit.get("parsed_call") and hit["parsed_call"].get("tool") == hit.get("expected_tool"):
                    cells.append("✓")
                else:
                    tool = (hit.get("parsed_call") or {}).get("tool", "—")
                    cells.append(f"✗ ({tool})")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="One or more directories containing per-variant *.jsonl files.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write Markdown to this path. Default: stdout.")
    ap.add_argument("--sizes-file", type=Path, default=None,
                    help="Optional `sha256  filename` file (e.g. CHECKSUMS.txt) — "
                         "size_mib for each variant is inferred from the filesystem if file is local.")
    ap.add_argument("--title", type=str, default="FunctionGemma 270M quantization sweep")
    args = ap.parse_args(argv)

    # Collect all variants across input dirs.
    sizes: dict[str, float] = {}
    if args.sizes_file and args.sizes_file.exists():
        for line in args.sizes_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].endswith(".gguf"):
                fn = parts[-1]
                p = args.sizes_file.parent / fn
                if p.exists():
                    name = fn.removeprefix("model-").removesuffix(".gguf")
                    sizes[name] = p.stat().st_size / (1024 * 1024)

    sections: list[str] = []
    for d in args.dirs:
        if not d.exists():
            print(f"WARN: dir not found: {d}")
            continue
        stats = load_dir(d, sizes=sizes)
        section_title = f"{args.title} — {d.name}"
        sections.append(render(stats, title=section_title))

    md = "\n\n---\n\n".join(sections) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
