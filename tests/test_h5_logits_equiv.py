"""Tests for h5_logits_equiv: corpus builder, parser, classifier."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gemma_tools.bench_prompt import wrap_gemma3_chat_template
from gemma_tools.h5_logits_equiv import (
    _OOD_PROMPTS,
    H5RVerdict,
    KLStats,
    build_corpus,
    build_q1_corpus,
    classify,
    emit_h5r_summary,
    main,
    parse_kl_output,
    score_h5r,
)

# ── Corpus builder ────────────────────────────────────────────────────────────


def test_corpus_total_count() -> None:
    corpus = build_corpus(seed=42)
    n = corpus.count("<start_of_turn>user")
    assert n == 35, f"expected 35 prompts, got {n}"


def test_corpus_yaml_count() -> None:
    # Exact 15 prompts from prompts.yaml must be present
    corpus = build_corpus(seed=42)
    # The first yaml prompt is "say hi"
    assert "say hi" in corpus


def test_corpus_sft_count() -> None:
    # 15 SFT prompts added
    # Build twice with same seed → same indices sampled → same content
    c1 = build_corpus(seed=42)
    c2 = build_corpus(seed=42)
    assert c1 == c2, "corpus not deterministic for same seed"


def test_corpus_different_seeds_differ() -> None:
    c1 = build_corpus(seed=42)
    c2 = build_corpus(seed=99)
    # Different seeds should produce different SFT selections (overwhelmingly likely)
    assert c1 != c2


def test_corpus_contains_ood_prompts() -> None:
    corpus = build_corpus(seed=42)
    for ood in _OOD_PROMPTS:
        assert ood in corpus, f"OOD prompt missing: {ood!r}"


def test_corpus_uses_gemma_chat_template() -> None:
    # Every user turn must be wrapped in the Gemma3 chat template
    corpus = build_corpus(seed=42)
    wrapped_example = wrap_gemma3_chat_template("say hi")
    assert wrapped_example in corpus


# ── Q1 corpus builder ─────────────────────────────────────────────────────────

# Each row is `{"messages": [{"role":"user","content":<composed Path B>}, {"role":"assistant","content":<answer>}]}`.
_Q1_FIXTURE_ROWS = [
    '{"messages":[{"role":"user","content":"ROLE: directive...\\nQUESTION: q1?"},{"role":"assistant","content":"a1"}]}',
    '{"messages":[{"role":"user","content":"ROLE: directive...\\nQUESTION: q2?"},{"role":"assistant","content":"a2"}]}',
    '{"messages":[{"role":"user","content":"ROLE: directive...\\nQUESTION: q3?"},{"role":"assistant","content":"a3"}]}',
    '{"messages":[{"role":"user","content":"ROLE: directive...\\nQUESTION: q4?"},{"role":"assistant","content":"a4"}]}',
    '{"messages":[{"role":"user","content":"ROLE: directive...\\nQUESTION: q5?"},{"role":"assistant","content":"a5"}]}',
]


def _write_fixture_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "fake_test.jsonl"
    p.write_text("\n".join(_Q1_FIXTURE_ROWS) + "\n")
    return p


@pytest.mark.parametrize(
    ("desc", "n", "seed"),
    [
        ("n=2 seed=1", 2, 1),
        ("n=3 seed=42", 3, 42),
        ("n=full pool", 5, 1),
    ],
)
def test_q1_corpus_count(desc: str, n: int, seed: int, tmp_path: Path) -> None:
    src = _write_fixture_jsonl(tmp_path)
    corpus = build_q1_corpus(n=n, seed=seed, test_jsonl_path=src)
    got = corpus.count("<start_of_turn>user")
    assert got == n, f"{desc}: expected {n} prompts, got {got}"


@pytest.mark.parametrize(
    ("desc", "seed"),
    [
        ("seed=1 deterministic", 1),
        ("seed=42 deterministic", 42),
    ],
)
def test_q1_corpus_deterministic(desc: str, seed: int, tmp_path: Path) -> None:
    src = _write_fixture_jsonl(tmp_path)
    c1 = build_q1_corpus(n=3, seed=seed, test_jsonl_path=src)
    c2 = build_q1_corpus(n=3, seed=seed, test_jsonl_path=src)
    assert c1 == c2, f"{desc}: corpus not deterministic for seed={seed}"


def test_q1_corpus_different_seeds_differ(tmp_path: Path) -> None:
    src = _write_fixture_jsonl(tmp_path)
    c1 = build_q1_corpus(n=3, seed=1, test_jsonl_path=src)
    c2 = build_q1_corpus(n=3, seed=99, test_jsonl_path=src)
    # 5-row pool, choose 3 — different seeds overwhelmingly produce different sets
    assert c1 != c2, "different seeds produced identical corpora — sampler is broken"


def test_q1_corpus_uses_chat_template(tmp_path: Path) -> None:
    src = _write_fixture_jsonl(tmp_path)
    corpus = build_q1_corpus(n=2, seed=1, test_jsonl_path=src)
    # Each prompt must be wrapped with the Gemma3 chat template, not raw JSON
    assert "<start_of_turn>user" in corpus
    assert "<start_of_turn>model" in corpus
    assert "ROLE: directive" in corpus  # composed user content survives the wrap


def test_q1_corpus_rejects_too_large_n(tmp_path: Path) -> None:
    src = _write_fixture_jsonl(tmp_path)
    with pytest.raises(ValueError, match="5 rows, requested 6"):
        build_q1_corpus(n=6, seed=1, test_jsonl_path=src)


def test_q1_corpus_real_pool_default_30() -> None:
    """Smoke against the real sft_v1.test.jsonl Path B file (110 rows)."""
    corpus = build_q1_corpus(n=30, seed=1)
    n = corpus.count("<start_of_turn>user")
    assert n == 30, f"expected 30 prompts, got {n}"
    # Path B composed prompts include the directive
    assert "ROLE: health-records assistant" in corpus


# ── Parser ────────────────────────────────────────────────────────────────────

# Verbatim excerpt from native self-comparison run (same-arch, expected GREEN)
_SELF_COMPARE_OUTPUT = textwrap.dedent("""\
    kl_divergence: computing over 4 chunks, n_ctx=256, batch_size=2048, n_seq=8
    kl_divergence: 1.48 seconds per pass - ETA 0.00 minutes
    chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
       1       1.9283 ±    0.3726       0.00000 ±    0.00003      -0.00000 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %
       2       2.0429 ±    0.2789       0.00000 ±       -nan      -0.00000 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %
       3       2.0996 ±    0.2532       0.00000 ±       -nan      -0.00000 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %
       4       2.1076 ±    0.2257       0.00321 ±    0.00321      -0.00000 ±    0.00000     0.000 ±  0.000 %    100.000 ±  0.000 %
    ====== Perplexity statistics ======
    Mean PPL(Q)                   :   2.107592 ±   0.225714
    ====== KL divergence statistics ======
    Mean    KLD:  -0.000000 ±   0.000000
    Maximum KLD:   0.000031
    ====== Token probability statistics ======
    Mean    Δp: -0.000 ± 0.000 %
    Maximum Δp:  0.002%
    Same top p: 100.000 ± 0.000 %
""")

_FAILING_OUTPUT = textwrap.dedent("""\
    chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
       1       3.1200 ±    0.5000       0.10000 ±    0.01000       0.05000 ±    0.00100     0.300 ±  0.050 %     82.500 ±  1.200 %
       2       3.5000 ±    0.6000       0.15000 ±    0.02000       0.08000 ±    0.00200     0.450 ±  0.060 %     78.000 ±  1.500 %
    ====== Token probability statistics ======
    Mean    Δp:  0.050 ± 0.010 %
    Maximum Δp:  0.800%
    Same top p:  80.250 ± 1.350 %
""")


def test_parse_green_same_top_p() -> None:
    stats = parse_kl_output(_SELF_COMPARE_OUTPUT)
    assert stats.same_top_p == pytest.approx(100.0)


def test_parse_green_max_delta_p() -> None:
    stats = parse_kl_output(_SELF_COMPARE_OUTPUT)
    assert stats.max_delta_p == pytest.approx(0.002)


def test_parse_green_chunk_count() -> None:
    stats = parse_kl_output(_SELF_COMPARE_OUTPUT)
    assert len(stats.chunks) == 4


def test_parse_green_chunk_values() -> None:
    stats = parse_kl_output(_SELF_COMPARE_OUTPUT)
    for c in stats.chunks:
        assert c["same_top_p"] == pytest.approx(100.0), f"chunk {c['chunk']} not 100%"


def test_parse_punt_same_top_p() -> None:
    stats = parse_kl_output(_FAILING_OUTPUT)
    assert stats.same_top_p == pytest.approx(80.25)


def test_parse_punt_max_delta_p() -> None:
    stats = parse_kl_output(_FAILING_OUTPUT)
    assert stats.max_delta_p == pytest.approx(0.8)


def test_parse_missing_same_top_p_raises() -> None:
    with pytest.raises(ValueError, match="Same top p"):
        parse_kl_output("Maximum Δp:  0.002%\n")


def test_parse_missing_max_delta_p_raises() -> None:
    with pytest.raises(ValueError, match="Maximum"):
        parse_kl_output("Same top p: 100.000 ± 0.000 %\n")


# ── Classifier ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("same_top_p", "max_delta_p", "expected"),
    [
        (100.0, 0.002, "GREEN"),   # self-compare baseline
        (99.999, 0.499, "GREEN"),  # just inside both thresholds
        (99.98, 0.002, "PUNT"),    # same_top_p below threshold
        (100.0, 0.501, "PUNT"),    # max_delta_p above threshold
        (80.0, 1.0, "PUNT"),       # both fail
    ],
)
def test_classify_verdict(same_top_p: float, max_delta_p: float, expected: str) -> None:
    stats = KLStats(same_top_p=same_top_p, max_delta_p=max_delta_p)
    verdict = classify(stats)
    assert verdict.result == expected, (
        f"same_top_p={same_top_p}, max_delta_p={max_delta_p}: "
        f"expected {expected}, got {verdict.result}"
    )


def test_classify_green_reason_mentions_100() -> None:
    stats = KLStats(same_top_p=100.0, max_delta_p=0.0)
    verdict = classify(stats)
    assert "100%" in verdict.reason


def test_classify_punt_reason_mentions_metric() -> None:
    stats = KLStats(same_top_p=80.0, max_delta_p=0.8)
    verdict = classify(stats)
    assert "same_top_p" in verdict.reason
    assert "max_delta_p" in verdict.reason


# ── H5R scorer (relative same-quant cross-arch Δ) ────────────────────────────

# Table: (desc, x86_same_top_p, a55_same_top_p, x86_max_delta_p, a55_max_delta_p,
#         max_delta_pp, max_delta_ratio, expected_status)
@pytest.mark.parametrize(
    (
        "desc",
        "x86_top",
        "a55_top",
        "x86_dp",
        "a55_dp",
        "max_pp",
        "max_ratio",
        "expected",
    ),
    [
        # GREEN — A55 within both gates against the x86 noise floor
        ("clean_pass", 99.30, 98.62, 6.50, 9.39, 1.0, 3.0, "GREEN"),
        ("just_inside_both", 99.50, 98.51, 5.00, 14.99, 1.0, 3.0, "GREEN"),
        ("perfect_agreement", 100.0, 100.0, 0.0, 0.0, 1.0, 3.0, "GREEN"),
        ("x86_better_than_a55_under_gate", 99.99, 99.50, 4.20, 8.40, 1.0, 3.0, "GREEN"),
        # PUNT — Δ over threshold, ratio fine
        ("delta_just_over", 99.50, 98.49, 5.00, 5.00, 1.0, 3.0, "PUNT"),
        ("delta_far_over", 99.00, 80.00, 4.00, 4.00, 1.0, 3.0, "PUNT"),
        # PUNT — ratio over threshold, Δ fine
        ("ratio_just_over", 99.30, 99.00, 4.00, 12.50, 1.0, 3.0, "PUNT"),
        ("ratio_far_over", 99.30, 99.00, 1.00, 50.0, 1.0, 3.0, "PUNT"),
        # PUNT — both fail
        ("both_fail", 99.50, 80.0, 5.0, 95.0, 1.0, 3.0, "PUNT"),
        # Edge: x86 == a55 same_top_p but a55 max_delta blew up
        ("equal_top_p_blown_ratio", 99.50, 99.50, 1.0, 4.0, 1.0, 3.0, "PUNT"),
        # Custom threshold paths — relaxed gate flips PUNT → GREEN
        ("relaxed_pp_gate", 99.50, 98.0, 6.0, 9.0, 2.0, 3.0, "GREEN"),
        ("relaxed_ratio_gate", 99.30, 99.00, 4.0, 16.0, 1.0, 5.0, "GREEN"),
        # Edge: x86 max_delta == 0 and a55 max_delta == 0 → ratio collapses to 0
        ("both_zero_max_delta", 100.0, 99.50, 0.0, 0.0, 1.0, 3.0, "GREEN"),
        # Edge: x86 max_delta == 0 but a55 has divergence → ratio = +inf → PUNT
        ("zero_x86_nonzero_a55", 100.0, 99.50, 0.0, 0.10, 1.0, 3.0, "PUNT"),
    ],
)
def test_score_h5r_verdict(
    desc: str,
    x86_top: float,
    a55_top: float,
    x86_dp: float,
    a55_dp: float,
    max_pp: float,
    max_ratio: float,
    expected: str,
) -> None:
    x86 = KLStats(same_top_p=x86_top, max_delta_p=x86_dp)
    a55 = KLStats(same_top_p=a55_top, max_delta_p=a55_dp)
    verdict = score_h5r(x86, a55, max_delta_pp=max_pp, max_delta_ratio=max_ratio)
    assert verdict.result == expected, f"{desc}: expected {expected}, got {verdict.result}"


def test_score_h5r_green_reason_mentions_within_floor() -> None:
    x86 = KLStats(same_top_p=99.30, max_delta_p=6.50)
    a55 = KLStats(same_top_p=98.62, max_delta_p=9.39)
    verdict = score_h5r(x86, a55)
    assert verdict.result == "GREEN"
    assert "noise floor" in verdict.reason
    assert "delta_same_top_p=0.680 pp" in verdict.reason


def test_score_h5r_punt_reason_lists_failing_gate() -> None:
    x86 = KLStats(same_top_p=99.30, max_delta_p=4.0)
    a55 = KLStats(same_top_p=80.0, max_delta_p=15.0)
    verdict = score_h5r(x86, a55)
    assert verdict.result == "PUNT"
    assert "delta_same_top_p" in verdict.reason
    assert "ratio_max_delta_p" in verdict.reason


def test_score_h5r_punt_reason_only_failing_gate() -> None:
    # Only ratio fails -- delta is fine
    x86 = KLStats(same_top_p=99.30, max_delta_p=2.0)
    a55 = KLStats(same_top_p=99.00, max_delta_p=10.0)
    verdict = score_h5r(x86, a55)
    assert verdict.result == "PUNT"
    assert "delta_same_top_p" not in verdict.reason
    assert "ratio_max_delta_p" in verdict.reason


def test_score_h5r_records_thresholds() -> None:
    x86 = KLStats(same_top_p=99.30, max_delta_p=6.50)
    a55 = KLStats(same_top_p=98.62, max_delta_p=9.39)
    verdict = score_h5r(x86, a55, max_delta_pp=1.5, max_delta_ratio=4.0)
    assert verdict.max_delta_pp_gate == pytest.approx(1.5)
    assert verdict.max_delta_ratio_gate == pytest.approx(4.0)


def test_score_h5r_records_raw_metrics() -> None:
    x86 = KLStats(same_top_p=99.30, max_delta_p=6.50)
    a55 = KLStats(same_top_p=98.62, max_delta_p=9.39)
    verdict = score_h5r(x86, a55)
    assert verdict.same_top_p_x86 == pytest.approx(99.30)
    assert verdict.same_top_p_a55 == pytest.approx(98.62)
    assert verdict.max_delta_p_x86 == pytest.approx(6.50)
    assert verdict.max_delta_p_a55 == pytest.approx(9.39)
    assert verdict.delta_same_top_p == pytest.approx(0.68)
    assert verdict.ratio_max_delta_p == pytest.approx(9.39 / 6.50)


def test_score_h5r_handles_zero_x86_max_delta() -> None:
    x86 = KLStats(same_top_p=100.0, max_delta_p=0.0)
    a55 = KLStats(same_top_p=99.5, max_delta_p=0.10)
    verdict = score_h5r(x86, a55)
    assert verdict.result == "PUNT"
    assert verdict.ratio_max_delta_p == float("inf")


def test_score_h5r_handles_double_zero_max_delta() -> None:
    x86 = KLStats(same_top_p=100.0, max_delta_p=0.0)
    a55 = KLStats(same_top_p=99.99, max_delta_p=0.0)
    verdict = score_h5r(x86, a55)
    assert verdict.result == "GREEN"
    assert verdict.ratio_max_delta_p == 0.0


def test_score_h5r_returns_frozen_dataclass() -> None:
    x86 = KLStats(same_top_p=99.0, max_delta_p=4.0)
    a55 = KLStats(same_top_p=98.5, max_delta_p=8.0)
    verdict = score_h5r(x86, a55)
    assert isinstance(verdict, H5RVerdict)
    with pytest.raises((AttributeError, Exception)):
        verdict.result = "MUTATED"  # type: ignore[misc]


# ── H5R summary emitter ──────────────────────────────────────────────────────


def _make_h5r_stats(
    same_top_p: float, max_delta_p: float, chunks: list[float]
) -> KLStats:
    return KLStats(
        same_top_p=same_top_p,
        max_delta_p=max_delta_p,
        chunks=[
            {"chunk": i + 1, "same_top_p": v}
            for i, v in enumerate(chunks)
        ],
    )


def test_emit_h5r_summary_green(tmp_path: Path) -> None:
    x86 = _make_h5r_stats(99.30, 6.50, [99.50, 99.30, 99.20, 99.20])
    a55 = _make_h5r_stats(98.62, 9.39, [99.21, 98.42, 98.69, 98.62])
    verdict = score_h5r(x86, a55)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(verdict, x86, a55, out)
    text = out.read_text()
    assert "**Verdict: GREEN**" in text
    assert "delta_same_top_p (x86 - a55)" in text
    assert "0.680 pp" in text
    assert "H6 unblocked" in text
    # Both chunk columns rendered
    assert "| 1 | 99.500% | 99.210% |" in text


def test_emit_h5r_summary_punt(tmp_path: Path) -> None:
    x86 = _make_h5r_stats(99.50, 4.0, [99.5, 99.5, 99.5, 99.5])
    a55 = _make_h5r_stats(80.0, 50.0, [80.0, 80.0, 80.0, 80.0])
    verdict = score_h5r(x86, a55)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(verdict, x86, a55, out)
    text = out.read_text()
    assert "**Verdict: PUNT**" in text
    assert "P3 path halted" in text
    assert "Escalate to upstream `llama.cpp #22011`" in text
    assert "delta_same_top_p" in text
    assert "ratio_max_delta_p" in text


def test_emit_h5r_summary_renders_thresholds(tmp_path: Path) -> None:
    x86 = _make_h5r_stats(99.30, 6.50, [99.30, 99.30, 99.30, 99.30])
    a55 = _make_h5r_stats(98.62, 9.39, [98.62, 98.62, 98.62, 98.62])
    verdict = score_h5r(x86, a55, max_delta_pp=2.0, max_delta_ratio=4.0)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(verdict, x86, a55, out)
    text = out.read_text()
    assert "<= 2.0 pp" in text
    assert "<= 4.0x" in text


def test_emit_h5r_summary_renders_provenance(tmp_path: Path) -> None:
    x86 = _make_h5r_stats(99.30, 6.50, [99.30, 99.30, 99.30, 99.30])
    a55 = _make_h5r_stats(98.62, 9.39, [98.62, 98.62, 98.62, 98.62])
    verdict = score_h5r(x86, a55)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(
        verdict, x86, a55, out,
        provenance={
            "corpus": "/tmp/h5_corpus.txt (sha 1234abcd)",
            "reference_kld": "h5-reference.kld (BF16, sha deadbeef)",
            "x86_command": "llama-perplexity -m … --kl-divergence-base ref.kld",
            "a55_reused_from": "docs/tmp/bench/2026-04-26_h5-logits-equivalence.md (flags match)",
        },
    )
    text = out.read_text()
    assert "## Provenance" in text
    assert "**corpus**: /tmp/h5_corpus.txt" in text
    assert "**a55_reused_from**:" in text


def test_emit_h5r_summary_handles_zero_x86_delta(tmp_path: Path) -> None:
    # ratio = +inf rendering; should not crash
    x86 = _make_h5r_stats(100.0, 0.0, [100.0])
    a55 = _make_h5r_stats(99.50, 0.10, [99.50])
    verdict = score_h5r(x86, a55)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(verdict, x86, a55, out)
    text = out.read_text()
    assert "+inf" in text
    assert "**Verdict: PUNT**" in text


def test_emit_h5r_summary_handles_missing_chunks(tmp_path: Path) -> None:
    x86 = KLStats(same_top_p=99.30, max_delta_p=6.50, chunks=[])
    a55 = KLStats(same_top_p=98.62, max_delta_p=9.39, chunks=[])
    verdict = score_h5r(x86, a55)
    out = tmp_path / "h5r.md"
    emit_h5r_summary(verdict, x86, a55, out)
    text = out.read_text()
    assert "**Verdict: GREEN**" in text
    assert "| - | - | - |" in text


# ── classify-h5r CLI ─────────────────────────────────────────────────────────

# Synthetic KL outputs that match the parser's expected shape.
_X86_Q4_0_LOG = textwrap.dedent("""\
    chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
       1       2.0500 ±    0.3000       0.04500 ±    0.00500       0.02000 ±    0.00100     0.180 ±  0.020 %     99.500 ±  0.150 %
       2       2.1000 ±    0.2800       0.05000 ±    0.00600       0.02300 ±    0.00120     0.190 ±  0.025 %     99.300 ±  0.180 %
       3       2.1200 ±    0.2700       0.05200 ±    0.00650       0.02400 ±    0.00130     0.195 ±  0.027 %     99.200 ±  0.190 %
       4       2.1300 ±    0.2600       0.05300 ±    0.00700       0.02500 ±    0.00135     0.200 ±  0.028 %     99.200 ±  0.195 %
    ====== Token probability statistics ======
    Mean    Δp:  0.025 ± 0.005 %
    Maximum Δp:  6.500%
    Same top p: 99.300 ± 0.180 %
""")

_A55_Q4_0_LOG = textwrap.dedent("""\
    chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
       1       2.0700 ±    0.3100       0.04700 ±    0.00510       0.02100 ±    0.00102     0.190 ±  0.022 %     99.213 ±  0.155 %
       2       2.1100 ±    0.2810       0.05100 ±    0.00610       0.02400 ±    0.00121     0.200 ±  0.026 %     98.425 ±  0.182 %
       3       2.1300 ±    0.2705       0.05300 ±    0.00655       0.02500 ±    0.00131     0.205 ±  0.028 %     98.688 ±  0.192 %
       4       2.1400 ±    0.2604       0.05400 ±    0.00701       0.02600 ±    0.00136     0.210 ±  0.029 %     98.622 ±  0.197 %
    ====== Token probability statistics ======
    Mean    Δp:  0.030 ± 0.006 %
    Maximum Δp:  9.393%
    Same top p: 98.622 ± 0.180 %
""")

_A55_Q4_0_BLOWN_LOG = textwrap.dedent("""\
    chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
       1       2.0700 ±    0.3100       0.04700 ±    0.00510       0.02100 ±    0.00102     0.190 ±  0.022 %     80.000 ±  1.500 %
    ====== Token probability statistics ======
    Mean    Δp:  0.300 ± 0.080 %
    Maximum Δp:  85.000%
    Same top p: 80.000 ± 1.500 %
""")


def _write_logs(tmp_path: Path, x86_text: str, a55_text: str) -> tuple[Path, Path]:
    x86 = tmp_path / "x86.log"
    a55 = tmp_path / "a55.log"
    x86.write_text(x86_text)
    a55.write_text(a55_text)
    return x86, a55


def test_cli_classify_h5r_green_returns_zero(tmp_path: Path) -> None:
    x86, a55 = _write_logs(tmp_path, _X86_Q4_0_LOG, _A55_Q4_0_LOG)
    rc = main(["classify-h5r", "--x86-log", str(x86), "--a55-log", str(a55)])
    assert rc == 0


def test_cli_classify_h5r_punt_returns_two(tmp_path: Path) -> None:
    x86, a55 = _write_logs(tmp_path, _X86_Q4_0_LOG, _A55_Q4_0_BLOWN_LOG)
    rc = main(["classify-h5r", "--x86-log", str(x86), "--a55-log", str(a55)])
    assert rc == 2


def test_cli_classify_h5r_writes_summary(tmp_path: Path) -> None:
    x86, a55 = _write_logs(tmp_path, _X86_Q4_0_LOG, _A55_Q4_0_LOG)
    summary = tmp_path / "out" / "h5r.md"
    rc = main([
        "classify-h5r",
        "--x86-log", str(x86),
        "--a55-log", str(a55),
        "--summary-out", str(summary),
        "--corpus-path", "/tmp/h5_corpus.txt (35 prompts)",
        "--kld-path", "/tmp/h5_ref.kld (BF16, sha abc123)",
        "--a55-reused-from", "docs/tmp/bench/2026-04-26_h5-logits-equivalence.md",
    ])
    assert rc == 0
    assert summary.exists()
    text = summary.read_text()
    assert "**Verdict: GREEN**" in text
    assert "**corpus**: /tmp/h5_corpus.txt" in text
    assert "**a55_reused_from**: docs/tmp/bench/2026-04-26_h5-logits-equivalence.md" in text


def test_cli_classify_h5r_custom_thresholds_relax_to_green(tmp_path: Path) -> None:
    # Default thresholds: this would PUNT (a55 80% / 85% max_delta). Relax to absurd values → GREEN.
    x86, a55 = _write_logs(tmp_path, _X86_Q4_0_LOG, _A55_Q4_0_BLOWN_LOG)
    rc = main([
        "classify-h5r",
        "--x86-log", str(x86),
        "--a55-log", str(a55),
        "--max-delta-pp", "100.0",
        "--max-delta-ratio", "1000.0",
    ])
    assert rc == 0


def test_cli_classify_h5r_missing_log_returns_one(tmp_path: Path) -> None:
    a55 = tmp_path / "a55.log"
    a55.write_text(_A55_Q4_0_LOG)
    bad = tmp_path / "missing.log"
    bad.write_text("not a perplexity log\n")
    rc = main(["classify-h5r", "--x86-log", str(bad), "--a55-log", str(a55)])
    assert rc == 1


def test_cli_legacy_classify_still_works(tmp_path: Path) -> None:
    log = tmp_path / "board.log"
    log.write_text(_SELF_COMPARE_OUTPUT)
    rc = main(["classify", "--kl-log", str(log)])
    assert rc == 0  # 100% same_top_p, 0.002 max_delta_p → GREEN under legacy gate
