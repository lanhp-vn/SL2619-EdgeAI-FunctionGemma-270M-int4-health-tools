"""CLI entrypoint that builds the Gemma 3 SFT artifacts from the chatbot pool.

Pipeline (matches `docs/plans/a55-gemma-fine-tune.md` §4 Phase 1):

    clean_sft_dataset.json          (chatbot-distilled raw pool)
        |  load_sft_pool          — schema validation
        v
    deduped pool                    (~1259 unique pairs)
        |  scan_bench_leakage     — bench-vs-pool overlap report
        v
    leakage report                  (5 exact + 51 near hits at NEAR_DUPLICATE_RATIO)
        |  split_pool             — paraphrase-aware stratified split
        v
    train / val / test assignments
        |  write_split_jsonl x 3 (Path B) + 3 (Path A)
        v
    sft_v1.{train,val,test}.jsonl                — TRL conversational, YAML-grounded
    sft_v1_pathA.{train,val,test}.jsonl          — TRL conversational, raw pairs (ablation)
    sft_v1.audit.jsonl                           — per-row routing provenance

The CLI prints the leakage summary, the split summary, and the per-class
counts so the user can audit before the JSONL files are consumed by the
server-side QLoRA job. No board access; runs entirely against the host
filesystem under `data/`.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from gemma_tools.health_table import load_health_table
from gemma_tools._legacy.sft_dataset import (
    SplitName,
    dedupe_pool,
    load_bench_prompts,
    load_sft_pool,
    scan_bench_leakage,
    split_pool,
    write_split_jsonl,
)

# Defaults match the canonical fixture paths under data/. Callers can
# override via CLI flags but the in-tree defaults exist so a `sft-build` with
# no args runs the canonical pipeline straight from a clean checkout.
_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_POOL = _REPO / "data" / "_legacy" / "clean_sft_dataset.json"
_DEFAULT_PROMPTS = _REPO / "data" / "_legacy" / "prompts.yaml"
_DEFAULT_HEALTH = _REPO / "data" / "health_table_v1.yaml"
_DEFAULT_OUT = _REPO / "data"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sft-build",
        description="Build SFT JSONL artifacts (Path B + Path A) from the chatbot pool.",
    )
    p.add_argument("--pool", type=Path, default=_DEFAULT_POOL,
                   help="Chatbot-distilled JSON pool (Alpaca shape).")
    p.add_argument("--prompts", type=Path, default=_DEFAULT_PROMPTS,
                   help="Bench prompts YAML used for leakage routing.")
    p.add_argument("--health", type=Path, default=_DEFAULT_HEALTH,
                   help="Patient YAML to inject into Path B user turns.")
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT,
                   help="Where to write sft_v1*.jsonl + audit.")
    p.add_argument("--now", type=date.fromisoformat, default=date(2026, 4, 25),
                   help="ISO date stamped into Path B prompts (deterministic).")
    p.add_argument("--seed", type=int, default=42, help="Stratified-split RNG seed.")
    p.add_argument("--skip-path-a", action="store_true",
                   help="Skip the ablation Path A emit (saves ~3 MiB).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # 1. Load + dedupe pool.
    raw = load_sft_pool(args.pool)
    pool, dedupe_report = dedupe_pool(raw)
    print(
        f"[load] pool={args.pool} raw={dedupe_report.input_count} "
        f"unique={dedupe_report.output_count} "
        f"dropped={dedupe_report.duplicates_removed}"
    )

    # 2. Scan bench leakage and surface the audit lines.
    prompts = load_bench_prompts(args.prompts)
    leakage = scan_bench_leakage(pool, prompts)
    print(f"[leakage] {len(leakage.all_hit_indices())} pool rows match bench prompts")
    for line in leakage.summary_lines():
        print(line)

    # 3. Stratified split.
    report = split_pool(pool, leakage, seed=args.seed)
    print(f"[split] seed={args.seed}  total={report.total}")
    for line in report.summary_lines():
        print(line)

    # 4. Emit artifacts.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "sft_v1.audit.jsonl"
    report.write_audit_jsonl(audit_path)
    print(f"[write] audit -> {audit_path}")

    splits: tuple[SplitName, ...] = ("train", "val", "test")
    health = load_health_table(args.health)
    for split_name in splits:
        out = args.out_dir / f"sft_v1.{split_name}.jsonl"
        n = write_split_jsonl(
            report, split_name, out, mode="path_b", health=health, now=args.now
        )
        print(f"[write] path_b {split_name:<5} n={n:<5} -> {out}")

    if not args.skip_path_a:
        for split_name in splits:
            out = args.out_dir / f"sft_v1_pathA.{split_name}.jsonl"
            n = write_split_jsonl(report, split_name, out, mode="path_a")
            print(f"[write] path_a {split_name:<5} n={n:<5} -> {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
