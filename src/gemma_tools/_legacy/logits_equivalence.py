"""H5 logits-equivalence: x86_64 vs SL2619 A55 for Gemma 3 270M-IT Q4_0.

Two-phase workflow:
  corpus   — build 35-prompt corpus, run native x86_64 llama-perplexity to produce .kld,
              print SCP + board commands for the user.
  classify — parse saved board llama-perplexity --kl-divergence output, emit GREEN/PUNT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from gemma_tools._legacy.bench_prompt import wrap_gemma3_chat_template

_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/gemma_tools/ → repo root
_PROMPTS_YAML = _REPO_ROOT / "data" / "_legacy" / "prompts.yaml"
_SFT_PATH_A = _REPO_ROOT / "data" / "_legacy" / "sft_v1_pathA.test.jsonl"
# Q1 corpus source: Path B test split (composed prompt = directive + YAML + question;
# matches the deployment shape and the SFT training shape, unlike _SFT_PATH_A).
_SFT_PATH_B_TEST = _REPO_ROOT / "data" / "_legacy" / "sft_v1.test.jsonl"
_NATIVE_BIN = (
    _REPO_ROOT
    / ".cache"
    / "llama-bench"
    / "llama.cpp"
    / "build-native"
    / "bin"
    / "llama-perplexity"
)
_DEFAULT_GGUF = (
    _REPO_ROOT / ".cache" / "llama-bench" / "gemma-3-270m-it-Q4_0.gguf"
)
_BOARD_GGUF = "/mnt/sdcard/models/gemma-3-270m-it-q4_0/gemma-3-270m-it-Q4_0.gguf"

_OOD_PROMPTS = [
    "What is the boiling point of water?",
    "Translate hello to French.",
    "List the planets in order from the Sun.",
    "What is 15 multiplied by 23?",
    "Who wrote the play Romeo and Juliet?",
]

# Gate thresholds (H5 plan §7 — preserved verbatim for legacy reproducibility)
_GATE_SAME_TOP_P_MIN: float = 99.99  # effective 100% with fp tolerance
_GATE_MAX_DELTA_P_MAX: float = 0.5   # percentage points

# Gate thresholds (H5R plan §6.5 — relative same-quant cross-arch Δ)
# Both are starting points; the bench summary records the chosen values + rationale.
_H5R_GATE_DELTA_SAME_TOP_P_MAX_PP: float = 1.0  # percentage points
_H5R_GATE_MAX_DELTA_P_RATIO_MAX: float = 3.0    # dimensionless


@dataclass
class KLStats:
    same_top_p: float
    max_delta_p: float
    chunks: list[dict[str, float | int]] = field(default_factory=list)


@dataclass
class Verdict:
    result: str   # "GREEN" or "PUNT"
    same_top_p: float
    max_delta_p: float
    reason: str


@dataclass(frozen=True)
class H5RVerdict:
    """Same-quant cross-arch Δ verdict (H5R plan §6.5)."""

    result: str  # "GREEN" or "PUNT"
    same_top_p_x86: float
    same_top_p_a55: float
    max_delta_p_x86: float
    max_delta_p_a55: float
    delta_same_top_p: float           # x86 - a55, percentage points
    ratio_max_delta_p: float          # a55 / x86; +inf if x86 == 0 and a55 > 0
    max_delta_pp_gate: float
    max_delta_ratio_gate: float
    reason: str


def build_corpus(seed: int = 42) -> str:
    """Return the 35-prompt corpus string: 15 yaml + 15 sft_pathA (seed) + 5 OOD."""
    texts: list[str] = []

    with _PROMPTS_YAML.open() as f:
        data = yaml.safe_load(f)
    for p in data["prompts"]:
        texts.append(wrap_gemma3_chat_template(p["text"]))

    with _SFT_PATH_A.open() as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), 15))
    for i in indices:
        user_text = rows[i]["messages"][0]["content"]
        texts.append(wrap_gemma3_chat_template(user_text))

    for t in _OOD_PROMPTS:
        texts.append(wrap_gemma3_chat_template(t))

    return "\n".join(texts)


def build_q1_corpus(
    n: int = 30,
    seed: int = 1,
    *,
    test_jsonl_path: Path = _SFT_PATH_B_TEST,
) -> str:
    """Return a Path B-shaped Q1 corpus string for post-quant logits-equivalence.

    Each prompt is the full composed user turn from `sft_v1.test.jsonl`
    (directive + YAML + question, ~600-700 tokens after wrapping) so the
    KL-divergence measurement runs on the deployment prompt shape rather than
    the bare-utterance H5R corpus. `random.Random(seed).sample` makes selection
    deterministic; default seed=1 differs from H5R's seed=42 to keep the two
    bench corpora visibly distinct in audit logs.

    Raises:
        FileNotFoundError: if `test_jsonl_path` is missing.
        ValueError: if the file has fewer than `n` rows.
    """
    with test_jsonl_path.open() as f:
        rows = [json.loads(line) for line in f]
    if len(rows) < n:
        raise ValueError(
            f"sft test pool has {len(rows)} rows, requested {n}: "
            f"{test_jsonl_path}"
        )
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    texts: list[str] = []
    for i in indices:
        user_text = rows[i]["messages"][0]["content"]
        texts.append(wrap_gemma3_chat_template(user_text))
    return "\n".join(texts)


def parse_kl_output(output: str) -> KLStats:
    """Parse llama-perplexity --kl-divergence stdout."""
    # Per-chunk rows: integer chunk number + 5 value±err pairs, last two have '%'.
    # The board may interleave an ETA log message on the same line as chunk 1, e.g.:
    #   "kl_divergence: 17.94 seconds per pass - ETA    1       1.9258 ± ..."
    # so we can't rely on parts[0].isdigit() — scan the first 15 tokens instead.
    chunks: list[dict[str, float | int]] = []
    for line in output.splitlines():
        parts = line.split()
        if parts[-1:] != ["%"] or len(parts) < 5:
            continue
        leading_digit = next((p for p in parts[:15] if p.isdigit()), None)
        if not leading_digit:
            continue
        try:
            same_top_p = float(parts[-4])
            chunks.append({"chunk": int(leading_digit), "same_top_p": same_top_p})
        except (ValueError, IndexError):
            pass

    # Summary stats
    m_same = re.search(r"^Same top p:\s*([\d.]+)", output, re.MULTILINE)
    if not m_same:
        raise ValueError("'Same top p' summary line not found in output")
    same_top_p = float(m_same.group(1))

    # "Maximum Δp:  0.002%" — Δ is U+0394, handle with \S+
    m_delta = re.search(r"Maximum\s+\S+\s*([\d.]+)%", output)
    if not m_delta:
        raise ValueError("'Maximum Δp' line not found in output")
    max_delta_p = float(m_delta.group(1))

    return KLStats(same_top_p=same_top_p, max_delta_p=max_delta_p, chunks=chunks)


def classify(stats: KLStats) -> Verdict:
    """Apply H5 gate."""
    reasons: list[str] = []
    if stats.same_top_p < _GATE_SAME_TOP_P_MIN:
        reasons.append(
            f"same_top_p={stats.same_top_p:.3f}% < {_GATE_SAME_TOP_P_MIN}%"
        )
    if stats.max_delta_p > _GATE_MAX_DELTA_P_MAX:
        reasons.append(
            f"max_delta_p={stats.max_delta_p:.3f}% > {_GATE_MAX_DELTA_P_MAX}%"
        )
    if reasons:
        return Verdict(
            result="PUNT",
            same_top_p=stats.same_top_p,
            max_delta_p=stats.max_delta_p,
            reason="; ".join(reasons),
        )
    return Verdict(
        result="GREEN",
        same_top_p=stats.same_top_p,
        max_delta_p=stats.max_delta_p,
        reason="same_top_p=100% AND max_delta_p≤0.5% — A55 NEON DOTPROD unaffected by #22011",
    )


def score_h5r(
    x86: KLStats,
    a55: KLStats,
    *,
    max_delta_pp: float = _H5R_GATE_DELTA_SAME_TOP_P_MAX_PP,
    max_delta_ratio: float = _H5R_GATE_MAX_DELTA_P_RATIO_MAX,
) -> H5RVerdict:
    """Apply H5R relative gate: same-quant cross-arch delta.

    GREEN iff (same_top_p_x86 - same_top_p_a55) <= max_delta_pp AND
    (max_delta_p_a55 / max_delta_p_x86) <= max_delta_ratio.

    Edge case: when max_delta_p_x86 == 0, the ratio is mathematically undefined.
    Treat as ratio = 0 if max_delta_p_a55 == 0 (perfect agreement on both sides),
    +inf otherwise (a55 has nonzero divergence with no x86 baseline to normalise against).
    """
    delta = x86.same_top_p - a55.same_top_p
    if x86.max_delta_p == 0.0:
        ratio = 0.0 if a55.max_delta_p == 0.0 else float("inf")
    else:
        ratio = a55.max_delta_p / x86.max_delta_p

    reasons: list[str] = []
    if delta > max_delta_pp:
        reasons.append(
            f"delta_same_top_p={delta:.3f} pp > {max_delta_pp} pp"
        )
    if ratio > max_delta_ratio:
        reasons.append(
            f"ratio_max_delta_p={ratio:.3f}x > {max_delta_ratio}x"
        )

    if reasons:
        return H5RVerdict(
            result="PUNT",
            same_top_p_x86=x86.same_top_p,
            same_top_p_a55=a55.same_top_p,
            max_delta_p_x86=x86.max_delta_p,
            max_delta_p_a55=a55.max_delta_p,
            delta_same_top_p=delta,
            ratio_max_delta_p=ratio,
            max_delta_pp_gate=max_delta_pp,
            max_delta_ratio_gate=max_delta_ratio,
            reason="; ".join(reasons),
        )

    return H5RVerdict(
        result="GREEN",
        same_top_p_x86=x86.same_top_p,
        same_top_p_a55=a55.same_top_p,
        max_delta_p_x86=x86.max_delta_p,
        max_delta_p_a55=a55.max_delta_p,
        delta_same_top_p=delta,
        ratio_max_delta_p=ratio,
        max_delta_pp_gate=max_delta_pp,
        max_delta_ratio_gate=max_delta_ratio,
        reason=(
            f"delta_same_top_p={delta:.3f} pp <= {max_delta_pp} pp AND "
            f"ratio_max_delta_p={ratio:.3f}x <= {max_delta_ratio}x -- "
            "A55 within same-quant cross-arch noise floor"
        ),
    )


def emit_h5r_summary(
    verdict: H5RVerdict,
    x86_stats: KLStats,
    a55_stats: KLStats,
    out_path: Path,
    *,
    provenance: dict[str, str] | None = None,
) -> None:
    """Write a H5R cross-arch-Δ bench summary.

    `provenance` may carry free-form keys (corpus_path, kld_path, x86_command, a55_command,
    a55_reused_from, gguf_sha256, llama_cpp_commit, ood_note, …) — values are dumped under
    a "Provenance" section. Order is preserved (insertion order).
    """
    today = date.today().isoformat()

    def _sym(ok: bool) -> str:
        return "✓" if ok else "✗"

    delta_pass = verdict.delta_same_top_p <= verdict.max_delta_pp_gate
    ratio_pass = verdict.ratio_max_delta_p <= verdict.max_delta_ratio_gate
    ratio_str = (
        "+inf" if verdict.ratio_max_delta_p == float("inf")
        else f"{verdict.ratio_max_delta_p:.3f}x"
    )

    next_action = (
        "**H6 unblocked** -- proceed to base-GGUF baseline bench."
        if verdict.result == "GREEN"
        else (
            "**P3 path halted.** Escalate to upstream `llama.cpp #22011` with: "
            "GGUF sha256, llama.cpp commit, prompt corpus, both KL summaries, "
            "chosen thresholds, and the reference `.kld`. Do NOT proceed to fine-tune."
        )
    )

    chunk_table = (
        "\n".join(
            f"| {x['chunk']} | {x['same_top_p']:.3f}% | "
            f"{a['same_top_p']:.3f}% |"
            for x, a in zip(x86_stats.chunks, a55_stats.chunks, strict=False)
        )
        if x86_stats.chunks and a55_stats.chunks
        else "| - | - | - |"
    )

    prov_block = ""
    if provenance:
        prov_lines = "\n".join(f"- **{k}**: {v}" for k, v in provenance.items())
        prov_block = f"\n## Provenance\n\n{prov_lines}\n"

    lines = f"""\
# H5R Logits-Equivalence Bench -- {today}

**Verdict: {verdict.result}** -- {verdict.reason}

Same-quant cross-arch delta test (H5R, replaces the absolute H5 gate that was preserved
verbatim in `docs/bench/2026-04-26_h5-logits-equivalence.md`). Reference is an
x86_64 BF16 `.kld`; both candidates are Q4_0 GGUF compared against that same reference.

## Raw Metrics

| Metric | x86_64 Q4_0 | SL2619 A55 Q4_0 |
| --- | --- | --- |
| Same top p | {verdict.same_top_p_x86:.3f}% | {verdict.same_top_p_a55:.3f}% |
| Max delta_p | {verdict.max_delta_p_x86:.3f}% | {verdict.max_delta_p_a55:.3f}% |

## Relative Gate (H5R)

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| delta_same_top_p (x86 - a55) | {verdict.delta_same_top_p:.3f} pp | <= {verdict.max_delta_pp_gate} pp | {_sym(delta_pass)} |
| ratio_max_delta_p (a55 / x86) | {ratio_str} | <= {verdict.max_delta_ratio_gate}x | {_sym(ratio_pass)} |

## Per-Chunk Same-Top-P

| Chunk | x86_64 | A55 |
| --- | --- | --- |
{chunk_table}

## Next Action

{next_action}
{prov_block}"""
    out_path.write_text(lines)


def emit_summary(verdict: Verdict, stats: KLStats, out_path: Path) -> None:
    today = date.today().isoformat()
    chunk_rows = "\n".join(
        f"| {c['chunk']} | {c['same_top_p']:.3f}% |" for c in stats.chunks
    )
    def _sym(ok: bool) -> str:
        return "✓" if ok else "✗"

    lines = f"""\
# H5 Logits-Equivalence Bench — {today}

**Verdict: {verdict.result}**  {verdict.reason}

## Gate Metrics

| Metric | Value | Threshold | Pass |
| --- | --- | --- | --- |
| Same top p | {verdict.same_top_p:.3f}% | ≥ {_GATE_SAME_TOP_P_MIN}% | {_sym(verdict.same_top_p >= _GATE_SAME_TOP_P_MIN)} |
| Max Δp | {verdict.max_delta_p:.3f}% | ≤ {_GATE_MAX_DELTA_P_MAX}% | {_sym(verdict.max_delta_p <= _GATE_MAX_DELTA_P_MAX)} |

## Per-Chunk Same-Top-P

| Chunk | Same top p |
| --- | --- |
{chunk_rows}
"""
    out_path.write_text(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cmd_q1_corpus(args: argparse.Namespace) -> int:
    """Write the Q1 Path B-shaped corpus to disk; do NOT run llama-perplexity.

    Q1 reference + x86 KL run is host-side (server has no llama-perplexity
    built — same constraint as H5R). The user runs llama-perplexity manually
    against the corpus + scp'd merged_v1 GGUFs after this command emits.
    """
    out = Path(args.out)
    src = Path(args.test_jsonl) if args.test_jsonl else _SFT_PATH_B_TEST
    corpus = build_q1_corpus(n=args.n, seed=args.seed, test_jsonl_path=src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(corpus)
    n_prompts = corpus.count("<start_of_turn>user")
    print(f"Q1 corpus: {out} ({n_prompts} prompts, {len(corpus)} chars)")
    print(f"  source : {src}")
    print(f"  seed   : {args.seed}")
    print("  shape  : Path B (composed user turn — directive + YAML + question)")
    print()
    print("Next steps (host-side, mirrors H5R discipline):")
    print("  1) scp {user}@nouslogic-server:~/sl2619-finetune/merged_v1.bf16.gguf .cache/q1/")
    print("  2) scp {user}@nouslogic-server:~/sl2619-finetune/merged_v1.q4_0.gguf .cache/q1/")
    print(
        "  3) ./.cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity "
        f"-m .cache/q1/merged_v1.bf16.gguf -f {out} "
        "--save-all-logits .cache/q1/merged_v1.bf16.kld -c 1024 --seed 1 --temp 0.0 --no-mmap -t $(nproc)"
    )
    print(
        "  4) ./.cache/llama-bench/llama.cpp/build-native/bin/llama-perplexity "
        f"-m .cache/q1/merged_v1.q4_0.gguf -f {out} "
        "--kl-divergence --kl-divergence-base .cache/q1/merged_v1.bf16.kld -c 1024 "
        "--seed 1 --temp 0.0 --no-mmap -t $(nproc) 2>&1 | tee /tmp/q1-x86-q4_0.log"
    )
    print("  5) Then board (user-runnable, NOT yet authorized):")
    print("     scp .cache/q1/merged_v1.q4_0.gguf nouslogic-sl2619:/mnt/sdcard/models/gemma-3-270m-it-q4_0-ft-v1/")
    print("     scp .cache/q1/merged_v1.bf16.kld   nouslogic-sl2619:/mnt/sdcard/models/q1/")
    print(f"     scp {out} nouslogic-sl2619:/mnt/sdcard/models/q1/q1_corpus.txt")
    return 0


def _cmd_corpus(args: argparse.Namespace) -> int:
    corpus_path = Path(args.out)
    kld_path = Path(args.ref_kld)
    gguf = str(args.gguf)

    corpus_text = build_corpus(seed=args.seed)
    corpus_path.write_text(corpus_text)
    n_prompts = corpus_text.count("<start_of_turn>user")
    print(f"Corpus: {corpus_path} ({n_prompts} prompts, {len(corpus_text)} chars)")

    if not _NATIVE_BIN.exists():
        print(f"ERROR: native binary not found: {_NATIVE_BIN}", file=sys.stderr)
        return 1

    print(f"Running native reference ({_NATIVE_BIN.name}) …")
    env = {**os.environ, "LD_LIBRARY_PATH": str(_NATIVE_BIN.parent)}
    cmd = [
        str(_NATIVE_BIN), "-m", gguf, "-f", str(corpus_path),
        "--save-all-logits", str(kld_path),
        "-c", "256", "--seed", "1", "-t", "10",
    ]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("ERROR: native reference run failed", file=sys.stderr)
        return 1

    kld_mb = kld_path.stat().st_size / (1024 * 1024)
    print(f"Reference KLD: {kld_path} ({kld_mb:.0f} MiB)")
    print()
    print("=== Board commands (run in YOUR terminal) ===")
    print(f"scp {corpus_path} nouslogic-sl2619:/tmp/h5_corpus.txt")
    print(f"scp {kld_path} nouslogic-sl2619:/tmp/h5_ref.kld")
    board_bin = "/mnt/sdcard/bin/llama-perplexity"
    print(
        f"ssh nouslogic-sl2619 '{board_bin} -m {_BOARD_GGUF}"
        f" -f /tmp/h5_corpus.txt"
        f" --kl-divergence --kl-divergence-base /tmp/h5_ref.kld"
        f" -c 256 --seed 1 -t 2' 2>&1 | tee /tmp/h5_board.log"
    )
    print()
    print("Then run:")
    print("  logits-equiv classify --kl-log /tmp/h5_board.log --summary-out docs/bench/$(date +%F)_logits-equivalence.md")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    kl_log = Path(args.kl_log)
    output = kl_log.read_text()
    try:
        stats = parse_kl_output(output)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    verdict = classify(stats)
    print(f"\nH5 Verdict: {verdict.result}")
    print(f"  Same top p : {verdict.same_top_p:.3f}%")
    print(f"  Max Δp     : {verdict.max_delta_p:.3f}%")
    print(f"  {verdict.reason}")

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        emit_summary(verdict, stats, out_path)
        print(f"\nSummary: {out_path}")

    return 0 if verdict.result == "GREEN" else 2


def _cmd_classify_h5r(args: argparse.Namespace) -> int:
    """Score same-quant cross-arch Δ from two `llama-perplexity --kl-divergence` logs."""
    x86_log = Path(args.x86_log)
    a55_log = Path(args.a55_log)
    try:
        x86_stats = parse_kl_output(x86_log.read_text())
        a55_stats = parse_kl_output(a55_log.read_text())
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    verdict = score_h5r(
        x86_stats,
        a55_stats,
        max_delta_pp=args.max_delta_pp,
        max_delta_ratio=args.max_delta_ratio,
    )

    print(f"\nH5R Verdict: {verdict.result}")
    print(f"  same_top_p_x86_q4_0 : {verdict.same_top_p_x86:.3f}%")
    print(f"  same_top_p_a55_q4_0 : {verdict.same_top_p_a55:.3f}%")
    print(f"  max_delta_p_x86_q4_0: {verdict.max_delta_p_x86:.3f}%")
    print(f"  max_delta_p_a55_q4_0: {verdict.max_delta_p_a55:.3f}%")
    print(f"  Δ_same_top_p        : {verdict.delta_same_top_p:.3f} pp "
          f"(gate ≤ {verdict.max_delta_pp_gate} pp)")
    ratio_disp = (
        "+inf" if verdict.ratio_max_delta_p == float("inf")
        else f"{verdict.ratio_max_delta_p:.3f}x"
    )
    print(f"  ratio_max_delta_p   : {ratio_disp} "
          f"(gate <= {verdict.max_delta_ratio_gate}x)")
    print(f"  {verdict.reason}")

    if args.summary_out:
        provenance: dict[str, str] = {}
        if args.corpus_path:
            provenance["corpus"] = args.corpus_path
        if args.kld_path:
            provenance["reference_kld"] = args.kld_path
        if args.x86_command:
            provenance["x86_command"] = args.x86_command
        if args.a55_command:
            provenance["a55_command"] = args.a55_command
        if args.a55_reused_from:
            provenance["a55_reused_from"] = args.a55_reused_from
        if args.gguf_sha256:
            provenance["gguf_sha256"] = args.gguf_sha256
        if args.llama_cpp_commit:
            provenance["llama_cpp_commit"] = args.llama_cpp_commit
        provenance["x86_log"] = str(x86_log)
        provenance["a55_log"] = str(a55_log)

        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        emit_h5r_summary(
            verdict, x86_stats, a55_stats, out_path,
            provenance=provenance or None,
        )
        print(f"\nSummary: {out_path}")

    return 0 if verdict.result == "GREEN" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="H5/H5R logits-equivalence: x86_64 vs A55 for Gemma 3 270M-IT Q4_0"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_corpus = sub.add_parser("corpus", help="Build corpus + run native reference")
    p_corpus.add_argument("--out", default="/tmp/h5_corpus.txt")
    p_corpus.add_argument("--ref-kld", default="/tmp/h5_ref.kld")
    p_corpus.add_argument("--gguf", default=str(_DEFAULT_GGUF))
    p_corpus.add_argument("--seed", type=int, default=42)

    p_q1 = sub.add_parser(
        "q1-corpus",
        help="Build Q1 Path B-shaped corpus for post-quant logits-equivalence",
    )
    p_q1.add_argument("--out", default=".cache/q1/q1_corpus.txt")
    p_q1.add_argument("--n", type=int, default=30)
    p_q1.add_argument("--seed", type=int, default=1)
    p_q1.add_argument(
        "--test-jsonl", default="",
        help="Override sft_v1.test.jsonl path (default: data/_legacy/sft_v1.test.jsonl)",
    )

    p_cls = sub.add_parser(
        "classify",
        help="Parse one KL log → legacy H5 absolute-gate GREEN/PUNT (preserves 2026-04-26 result)",
    )
    p_cls.add_argument("--kl-log", required=True)
    p_cls.add_argument("--summary-out", default="")

    p_h5r = sub.add_parser(
        "classify-h5r",
        help="Parse two KL logs (x86 + a55) → H5R relative-gate GREEN/PUNT",
    )
    p_h5r.add_argument("--x86-log", required=True, help="x86_64 Q4_0 vs BF16 .kld log")
    p_h5r.add_argument("--a55-log", required=True, help="A55 Q4_0 vs same BF16 .kld log")
    p_h5r.add_argument("--summary-out", default="")
    p_h5r.add_argument(
        "--max-delta-pp",
        type=float,
        default=_H5R_GATE_DELTA_SAME_TOP_P_MAX_PP,
        help="Max delta_same_top_p (x86 - a55) in percentage points (default 1.0)",
    )
    p_h5r.add_argument(
        "--max-delta-ratio",
        type=float,
        default=_H5R_GATE_MAX_DELTA_P_RATIO_MAX,
        help="Max ratio_max_delta_p (a55 / x86), dimensionless (default 3.0)",
    )
    # Provenance fields rendered into the summary's Provenance section.
    p_h5r.add_argument("--corpus-path", default="")
    p_h5r.add_argument("--kld-path", default="")
    p_h5r.add_argument("--x86-command", default="")
    p_h5r.add_argument("--a55-command", default="")
    p_h5r.add_argument("--a55-reused-from", default="")
    p_h5r.add_argument("--gguf-sha256", default="")
    p_h5r.add_argument("--llama-cpp-commit", default="")

    args = parser.parse_args(argv)
    if args.cmd == "corpus":
        return _cmd_corpus(args)
    if args.cmd == "q1-corpus":
        return _cmd_q1_corpus(args)
    if args.cmd == "classify-h5r":
        return _cmd_classify_h5r(args)
    return _cmd_classify(args)


if __name__ == "__main__":
    sys.exit(main())
