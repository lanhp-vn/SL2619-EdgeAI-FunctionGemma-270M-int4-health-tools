"""Tests for gemma_tools.sft_dataset.

Covers D1a only: loader + dataclass schema validation. Dedupe, classifier,
leakage scan, and split logic land in subsequent test files-or-blocks per
R2 (write -> test -> fix -> next chunk).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from gemma_tools.health_table import HealthTable, Patient, Vitals
from gemma_tools._legacy.sft_dataset import (
    NEAR_DUPLICATE_RATIO,
    SFT_CLASSES,
    TEST_CLASS_DRAIN_LIMIT,
    BenchPrompt,
    DedupeReport,
    SftClass,
    SftRecord,
    SplitName,
    SplitReport,
    class_distribution,
    classify_record,
    dedupe_pool,
    load_bench_prompts,
    load_sft_pool,
    scan_bench_leakage,
    split_pool,
    write_split_jsonl,
)

_REPO = Path(__file__).resolve().parents[2]
_CANONICAL_POOL = _REPO / "data" / "_legacy" / "clean_sft_dataset.json"
_CANONICAL_PROMPTS = _REPO / "data" / "_legacy" / "prompts.yaml"


def _write_pool(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    """Materialize a pool fixture without leaking state across tests."""
    out = tmp_path / "pool.json"
    out.write_text(json.dumps(records))
    return out


def test_load_sft_pool_canonical_count() -> None:
    # The chatbot-distilled pool currently has 1400 raw rows. This count
    # is allowed to drift forward as we extend the dataset; the test is a
    # tripwire that flags an unexpected shrink (e.g. accidental truncation).
    pool = load_sft_pool(_CANONICAL_POOL)
    assert len(pool) >= 1400, (
        f"canonical pool shrank to {len(pool)} (expected >= 1400)"
    )


def test_load_sft_pool_returns_frozen_records() -> None:
    pool = load_sft_pool(_CANONICAL_POOL)
    rec = pool[0]
    assert isinstance(rec, SftRecord), f"unexpected element type {type(rec).__name__}"
    with pytest.raises((AttributeError, TypeError)):
        rec.instruction = "mutated"  # type: ignore[misc]


def test_load_sft_pool_strips_whitespace(tmp_path: Path) -> None:
    pool_path = _write_pool(
        tmp_path,
        [{"instruction": "  what is my HR?  ", "input": "", "output": "  72 bpm.  "}],
    )
    rec = load_sft_pool(pool_path)[0]
    assert rec.instruction == "what is my HR?", f"instruction not stripped: {rec.instruction!r}"
    assert rec.output == "72 bpm.", f"output not stripped: {rec.output!r}"


def test_load_sft_pool_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_sft_pool(missing)


# | bad_doc                                 | match_substring        | desc                        |
@pytest.mark.parametrize(
    ("bad_doc", "match_substring", "desc"),
    [
        ({"not": "a list"},                  "must be a JSON array", "object root rejected"),
        ([{"instruction": "hi"}],            "missing keys",          "missing input + output"),
        ([{"instruction": "hi", "output": "y", "input": "", "extra": 1}],
                                             "unexpected keys",       "extra key rejected"),
        ([{"instruction": 1, "output": "y", "input": ""}],
                                             "expected str",          "non-string instruction"),
        ([{"instruction": "", "output": "y", "input": ""}],
                                             "empty after strip",     "empty instruction rejected"),
        ([{"instruction": "  ", "output": "y", "input": ""}],
                                             "empty after strip",     "whitespace-only instruction"),
        ([{"instruction": "x", "output": "", "input": ""}],
                                             "empty after strip",     "empty output rejected"),
        ([{"instruction": "x", "output": "y", "input": "ctx"}],
                                             "must be empty",         "non-empty input rejected"),
        (["plain string"],                   "expected mapping",      "non-dict element rejected"),
    ],
)
def test_load_sft_pool_schema_errors(
    tmp_path: Path,
    bad_doc: object,
    match_substring: str,
    desc: str,
) -> None:
    pool_path = tmp_path / "bad.json"
    pool_path.write_text(json.dumps(bad_doc))
    with pytest.raises(ValueError, match=match_substring):
        load_sft_pool(pool_path)
    # `desc` is asserted as part of the failure message via pytest's parametrize
    # id; we also include it in a no-op assert so a future flake report keeps
    # the human-readable description visible.
    assert desc, desc


# --------------------------------------------------------------------------
# D1b — dedupe.
# --------------------------------------------------------------------------


def _rec(inst: str, out: str) -> SftRecord:
    return SftRecord(instruction=inst, output=out, input="")


def test_dedupe_removes_exact_pair_duplicates() -> None:
    pool = [
        _rec("what is my HR?", "72 bpm."),
        _rec("HR?", "72 bpm."),                     # different inst → kept
        _rec("what is my HR?", "72 bpm."),          # exact dup of #1 → dropped
        _rec("blood type?", "O+."),
    ]
    kept, report = dedupe_pool(pool)
    assert len(kept) == 3, f"expected 3 unique pairs, got {len(kept)}"
    assert report == DedupeReport(input_count=4, output_count=3, duplicates_removed=1), report


def test_dedupe_is_case_insensitive_and_whitespace_collapsing() -> None:
    # Dedupe folds: case + leading/trailing whitespace + collapsed internal
    # whitespace runs. It does NOT modify punctuation-adjacent spacing
    # ("hr ?" stays distinct from "hr?") because that often signals
    # different intent in chatbot phrasings.
    pool = [
        _rec("HR?", "72 bpm."),
        _rec("hr?", "72 BPM."),       # case-fold only → same as #1
        _rec("HR?", "72  bpm."),      # double-space inside output collapses
    ]
    kept, report = dedupe_pool(pool)
    assert len(kept) == 1, f"case/whitespace dupes should fold to 1, got {len(kept)}"
    assert report.duplicates_removed == 2, report


def test_dedupe_keeps_same_instruction_with_different_output() -> None:
    # Chatbots disagreed on the answer — the instruction shape is identical
    # but the answer is genuinely different. We do NOT silently pick one;
    # the caller audits these manually.
    pool = [
        _rec("what is my HR?", "72 bpm."),
        _rec("what is my HR?", "Heart rate is 72 beats per minute."),
    ]
    kept, _ = dedupe_pool(pool)
    assert len(kept) == 2, f"different outputs should both survive, got {len(kept)}"


def test_dedupe_preserves_first_seen_order() -> None:
    pool = [
        _rec("a", "1"),
        _rec("a", "1"),     # dup of #1 — dropped
        _rec("b", "2"),
        _rec("c", "3"),
    ]
    kept, _ = dedupe_pool(pool)
    assert [r.instruction for r in kept] == ["a", "b", "c"], "order not preserved"


# --------------------------------------------------------------------------
# D1c — class auto-tagger.
# --------------------------------------------------------------------------


# | inst                                          | out                                                    | expected         | desc                                |
@pytest.mark.parametrize(
    ("inst", "out", "expected", "desc"),
    [
        ("tell me a joke",
         "I answer questions from your health record only",
         "domain_refusal", "canonical refusal recognized regardless of inst"),
        ("what is the capital of france?",
         "i answer questions from your health record only.",
         "domain_refusal", "case-insensitive refusal match"),
        ("what is my LDL?",
         "not in record",
         "fact_absence", "canonical absence marker"),
        ("who is my PCP?",
         "Not in your record.",
         "fact_absence", "alternate absence phrasing accepted"),
        ("what is my heart rate?",       "72 bpm.",                "fact_lookup",
         "single-fact short answer"),
        ("HR?",                          "72 bpm.",                "fact_lookup",
         "terse fact_lookup"),
        ("summarize my medications",
         "Lisinopril 10 mg at 08:00, Metformin 500 mg at 08:00 and 19:00, Aspirin 81 mg.",
         "summarization", "multi-comma multi-fact"),
        ("which conditions are controlled?",
         "Hypertension is controlled true, Type 2 Diabetes is controlled true, and High Cholesterol is controlled true.",
         "summarization", "long multi-fact answer"),
        ("summarize my BP",              "118/76 mmHg.",           "summarization",
         "imperative summary verb forces summarization even with terse answer"),
        ("list all my meds",             "Lisinopril.",            "summarization",
         "list-all imperative forces summarization"),
        ("what is my temperature?",      "36.7 C.",                "fact_lookup",
         "single-token answer fact_lookup"),
        ("what does aspirin do?",
         "Cardiovascular protection.",
         "fact_lookup", "no commas, short → fact_lookup"),
    ],
)
def test_classify_record_branches(
    inst: str,
    out: str,
    expected: SftClass,
    desc: str,
) -> None:
    rec = _rec(inst, out)
    actual = classify_record(rec)
    assert actual == expected, f"{desc}: got {actual!r}"


def test_classify_record_canonical_pool_distribution() -> None:
    # Profiled distribution against the chatbot-distilled pool. Bound is
    # exact and fails loudly on drift; intentional dataset extensions land
    # with the bound updated in the same commit.
    pool = load_sft_pool(_CANONICAL_POOL)
    kept, _ = dedupe_pool(pool)
    dist = class_distribution(kept)
    assert dist["domain_refusal"] >= 100, f"refusals shrank: {dist}"
    assert dist["fact_absence"] >= 100, f"absences shrank: {dist}"
    assert dist["fact_lookup"] >= 400, f"fact_lookup shrank: {dist}"
    assert dist["summarization"] >= 100, f"summarization shrank: {dist}"
    assert sum(dist.values()) == len(kept), f"counts don't sum: {dist} vs {len(kept)}"


# --------------------------------------------------------------------------
# D1d — bench leakage scanner.
# --------------------------------------------------------------------------


def test_load_bench_prompts_canonical_count() -> None:
    prompts = load_bench_prompts(_CANONICAL_PROMPTS)
    # 15 entries in data/_legacy/prompts.yaml today; bound is exact and fails
    # loudly on intentional drift.
    assert len(prompts) == 15, f"prompts.yaml entry count drifted: {len(prompts)}"


def test_load_bench_prompts_rejects_missing_keys(tmp_path: Path) -> None:
    bad = tmp_path / "prompts.yaml"
    bad.write_text("prompts:\n  - id: P1\n    class: fact_lookup\n")  # no text
    with pytest.raises(ValueError, match="missing 'text'"):
        load_bench_prompts(bad)


def _bp(pid: str, cls: str, text: str) -> BenchPrompt:
    return BenchPrompt(id=pid, cls=cls, text=text)


def test_scan_bench_leakage_exact_and_near() -> None:
    pool = (
        _rec("what is my blood pressure?", "118/76 mmHg."),                   # exact P2
        _rec("what is my current heart rate?", "72 bpm."),                    # near P1
        _rec("how many ribs do humans have?", "Off-topic."),                  # far miss
        _rec("BP reading", "118/76 mmHg."),                                   # too short → far
    )
    prompts = (
        _bp("P1", "fact_lookup", "what is my heart rate?"),
        _bp("P2", "fact_lookup", "what is my blood pressure?"),
    )
    report = scan_bench_leakage(pool, prompts)

    p1, p2 = report.per_prompt
    assert p1.exact == (), f"P1 should have no exact hit, got {p1.exact}"
    assert len(p1.near) == 1, f"P1 should match 'current heart rate' once, got {len(p1.near)}"
    assert p1.near[0].pool_index == 1, "wrong P1 near index"

    assert len(p2.exact) == 1, f"P2 should have 1 exact hit, got {len(p2.exact)}"
    assert p2.exact[0].pool_index == 0, "wrong P2 exact index"
    # "BP reading" vs "what is my blood pressure?" — too short to clear 0.80
    # ratio. The semantic-equivalence catch (paraphrase clustering by canonical
    # output) is the splitter's job, not the scanner's.
    assert all(h.pool_index != 3 for h in p2.near), (
        "scanner is over-eager — 'BP reading' should not match 'what is my blood pressure?'"
    )

    indices = report.all_hit_indices()
    assert 0 in indices, "exact P2 hit must be in must-route-to-test set"
    assert 1 in indices, "near P1 hit must be in must-route-to-test set"
    assert 2 not in indices, "unrelated row leaked into hit set"


def test_scan_bench_leakage_canonical_pool_routes_five_exact_hits() -> None:
    # Confirms the analysis-time finding: P2, P7, P9, D1, D2 each have
    # ≥ 1 exact match in the canonical pool. P1 has only near-paraphrases.
    pool = load_sft_pool(_CANONICAL_POOL)
    prompts = load_bench_prompts(_CANONICAL_PROMPTS)
    report = scan_bench_leakage(pool, prompts)
    by_id = {pl.prompt.id: pl for pl in report.per_prompt}
    for pid in ("P2", "P7", "P9", "D1", "D2"):
        assert by_id[pid].exact, f"{pid} should have at least one exact match in pool"
    assert by_id["P1"].exact == (), "P1 has no exact match per analysis"
    # And P1 should still be flagged as a near-paraphrase (current vs. no-current).
    assert by_id["P1"].near, "P1 should match 'what is my heart rate?' near-paraphrase"


def test_scan_bench_leakage_threshold_constant_unchanged() -> None:
    # Tripwire: the 0.80 ratio decides a lot of routing. A drift here changes
    # which pool rows must land in test, so the change must be intentional
    # and the test bound updated in the same commit.
    assert NEAR_DUPLICATE_RATIO == 0.80, "NEAR_DUPLICATE_RATIO drifted"


def test_dedupe_canonical_pool_shrinks_to_known_count() -> None:
    # Profiled count: 1400 raw rows -> 1259 unique (instruction, output) pairs.
    # If this drifts unexpectedly the test fails; intentional dataset
    # extensions land with the test bound updated in the same commit.
    pool = load_sft_pool(_CANONICAL_POOL)
    kept, report = dedupe_pool(pool)
    assert report.duplicates_removed == 141, (
        f"canonical dedupe drift: removed {report.duplicates_removed}, expected 141"
    )
    assert len(kept) == 1259, f"canonical unique count drifted: {len(kept)}"


# --------------------------------------------------------------------------
# D1e — paraphrase-aware stratified splitter.
# --------------------------------------------------------------------------


def _build_synthetic_pool() -> tuple[tuple[SftRecord, ...], tuple[BenchPrompt, ...]]:
    """A small but realistic pool exercising every routing reason.

    Index | reason fired       | how it fires
    ------+--------------------+------------------------------------------------
      0   | bench_exact        | matches P_FACT exactly
      1   | bench_near         | close paraphrase of P_FACT (ratio ~0.85)
      2   | same_inst_conflict | shares inst with #0 but different output
      3   | cluster_output     | shares output "72 bpm." with #0 (fact_lookup)
      4   | bench_exact        | matches P_REFUSE exactly
      5   | bench_near         | close paraphrase of P_SUM (ratio ~0.90)
      6   | cluster_instruction| sim(#6,#5) >= 0.80 but sim(#6,bench) < 0.80
                                  — fires cluster expansion in summarization class
    """
    refusal = "I answer questions from your health record only"
    summary_out = "Lisinopril, Metformin, Aspirin, Atorvastatin."
    pool: list[SftRecord] = [
        _rec("what is my heart rate?", "72 bpm."),
        _rec("what is my current heart rate?", "72 bpm."),
        _rec("what is my heart rate?", "Heart rate is 72 bpm."),
        _rec("hr please?", "72 bpm."),
        _rec("tell me a joke", refusal),
        # bench_near for P_SUM — adding "please" bumps length but shares all
        # 32 chars with the bench, ratio 32*2/(32+39)=0.901.
        _rec("summarize my current medications please", summary_out),
        # cluster_instruction trigger — three "please" suffixes drop sim with
        # P_SUM to 0.753 (below NEAR_DUPLICATE_RATIO) but keep sim with #5 at
        # 0.848 (above). Same summarization class as #5 → expands.
        _rec(
            "summarize my current medications please please please",
            summary_out,
        ),
    ]
    # 9 filler fact_lookup rows
    for i in range(9):
        pool.append(_rec(f"what is fact_{i}?", f"value_{i}."))
    # 5 filler refusals NOT instruction-similar to "tell me a joke"
    for phrase in (
        "what is the weather?", "translate hello", "set a timer",
        "explain photosynthesis", "what is 2+2?",
    ):
        pool.append(_rec(phrase, refusal))
    # 5 filler summarization rows that won't cluster against #5 / #6
    for i in range(5):
        pool.append(
            _rec(
                f"give me an overview of topic {i}",
                f"point a, point b, point c about topic {i}.",
            )
        )

    prompts = (
        _bp("P_FACT", "fact_lookup", "what is my heart rate?"),
        _bp("P_REFUSE", "domain_refusal", "tell me a joke"),
        _bp("P_SUM", "summarization", "summarize my current medications"),
    )
    return tuple(pool), prompts


def test_split_pool_routes_force_indices_to_test() -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    report = split_pool(pool, leakage, seed=42)
    by_idx = {a.pool_index: a for a in report.assignments}

    assert by_idx[0].split == "test", "exact bench hit must be in test"
    assert by_idx[0].routing_reason == "bench_exact", by_idx[0].routing_reason
    assert by_idx[0].matched_bench_id == "P_FACT"

    assert by_idx[1].split == "test", "near bench hit must be in test"
    assert by_idx[1].routing_reason == "bench_near"

    assert by_idx[2].split == "test", "same-inst-conflict must be in test"
    assert by_idx[2].routing_reason == "same_instruction_conflict"

    assert by_idx[3].split == "test", "cluster_output must be in test"
    assert by_idx[3].routing_reason == "cluster_output"

    assert by_idx[4].split == "test", "P_REFUSE exact match must be in test"
    assert by_idx[4].routing_reason == "bench_exact"

    assert by_idx[5].split == "test", "P_SUM near match must be in test"
    assert by_idx[5].routing_reason == "bench_near"

    assert by_idx[6].split == "test", "cluster_instruction must be in test"
    assert by_idx[6].routing_reason == "cluster_instruction"
    sim_6 = by_idx[6].similarity
    assert sim_6 is not None, "cluster_instruction must record a similarity score"
    assert sim_6 >= 0.80, f"cluster_instruction similarity below threshold: {sim_6}"


def test_split_pool_does_not_drain_refusals_via_cluster_output() -> None:
    # All 119 deduped refusals share a canonical output. If cluster_output
    # accidentally applied to refusals it would drain every refusal into
    # test — which is the very failure mode the user warned about.
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    report = split_pool(pool, leakage, seed=42)
    splits: tuple[SplitName, ...] = ("train", "val", "test")
    refusal_test = report.by_split_class["test"]["domain_refusal"]
    refusal_total = sum(report.by_split_class[s]["domain_refusal"] for s in splits)
    assert refusal_test < refusal_total, (
        f"all refusals routed to test: {refusal_test}/{refusal_total} — "
        "cluster_output is leaking into domain_refusal class"
    )


def test_split_pool_is_deterministic_with_seed() -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    a = split_pool(pool, leakage, seed=42)
    b = split_pool(pool, leakage, seed=42)
    assert [x.split for x in a.assignments] == [x.split for x in b.assignments]


def test_split_pool_seed_changes_assignments() -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    a = split_pool(pool, leakage, seed=42)
    b = split_pool(pool, leakage, seed=999)
    # Force-routed assignments must not change; only stratified-random ones.
    a_random = [(x.pool_index, x.split) for x in a.assignments
                if x.routing_reason == "stratified_random"]
    b_random = [(x.pool_index, x.split) for x in b.assignments
                if x.routing_reason == "stratified_random"]
    assert a_random != b_random, "seed has no effect on stratified split"


def test_split_pool_no_record_in_two_splits() -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    report = split_pool(pool, leakage, seed=42)
    seen: set[int] = set()
    for a in report.assignments:
        assert a.pool_index not in seen, f"row {a.pool_index} assigned twice"
        seen.add(a.pool_index)
    assert len(seen) == len(pool), "not every row was assigned"


def test_split_pool_ratio_validation() -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    with pytest.raises(ValueError, match="ratios must sum"):
        split_pool(pool, leakage, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_split_pool_writes_audit_jsonl(tmp_path: Path) -> None:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    report = split_pool(pool, leakage, seed=42)
    out = tmp_path / "audit.jsonl"
    report.write_audit_jsonl(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(pool), f"expected {len(pool)} audit lines, got {len(lines)}"
    parsed = [json.loads(line) for line in lines]
    required_keys = {
        "pool_index", "instruction", "output", "class", "split",
        "routing_reason", "matched_bench_id", "matched_bench_text", "similarity",
    }
    for row in parsed:
        assert set(row.keys()) == required_keys, f"audit row keys drifted: {set(row.keys())}"
    # Spot-check the exact-match row.
    exact_row = next(r for r in parsed if r["pool_index"] == 0)
    assert exact_row["routing_reason"] == "bench_exact"
    assert exact_row["matched_bench_id"] == "P_FACT"
    assert exact_row["similarity"] == 1.0


def test_split_pool_canonical_pool_class_proportions() -> None:
    # End-to-end: load pool, dedupe, scan leakage, split. Capture class counts
    # so the user can see whether stratification holds for rare classes.
    pool = load_sft_pool(_CANONICAL_POOL)
    kept, _ = dedupe_pool(pool)
    leakage = scan_bench_leakage(kept, load_bench_prompts(_CANONICAL_PROMPTS))
    report = split_pool(kept, leakage, seed=42)

    # Total preserved
    assert report.total == len(kept)
    n = report.total

    # Test should be roughly between 5% and 25% of pool (force-routing inflates
    # the lower bound from the nominal 10%).
    test_n = sum(report.by_split_class["test"].values())
    assert 0.05 * n <= test_n <= 0.25 * n, f"test size {test_n} out of band ({n})"

    # Train must hold the majority of every rare class.
    all_splits: tuple[SplitName, ...] = ("train", "val", "test")
    rare_classes: tuple[SftClass, ...] = ("domain_refusal", "fact_absence")
    for rare in rare_classes:
        train = report.by_split_class["train"][rare]
        total = sum(report.by_split_class[s][rare] for s in all_splits)
        assert train / total >= 0.6, (
            f"{rare} train share {train}/{total} too low — stratification leaked"
        )

    # No split is empty for any class
    splits: tuple[SplitName, ...] = ("train", "val", "test")
    for split_name in splits:
        for cls in SFT_CLASSES:
            assert report.by_split_class[split_name][cls] >= 1, (
                f"{split_name}/{cls} is empty — stratification failed"
            )


def test_split_pool_drain_guard_constants() -> None:
    # Tripwire on the public constants the splitter exposes for tuning.
    assert TEST_CLASS_DRAIN_LIMIT == 0.50, "drain limit drifted"
    assert SFT_CLASSES == (
        "fact_lookup", "fact_absence", "domain_refusal", "summarization",
    ), "class order drifted (affects audit output column order)"


def test_split_pool_ratio_constant_threshold_unchanged() -> None:
    # Make sure NEAR_DUPLICATE_RATIO didn't silently move on us.
    assert NEAR_DUPLICATE_RATIO == 0.80


# --------------------------------------------------------------------------
# D1f — Path B (composed prompt) + Path A (raw pairs) JSONL emitter.
# --------------------------------------------------------------------------

# Minimal HealthTable for emitter tests — keeps the fixture-load surface
# out of the JSONL test path.
_HEALTH = HealthTable(
    patient=Patient(name="Test Patient", age=45),
    vitals=Vitals(
        heart_rate_bpm=72,
        blood_pressure_systolic=118,
        blood_pressure_diastolic=76,
        spo2_percent=98,
        body_temperature_c=36.7,
        respiratory_rate=16,
    ),
    notes=("Vitals within nominal range",),
)
_NOW = date(2026, 4, 25)


def _make_split_report() -> tuple[SplitReport, tuple[SftRecord, ...]]:
    pool, prompts = _build_synthetic_pool()
    leakage = scan_bench_leakage(pool, prompts)
    return split_pool(pool, leakage, seed=42), pool


def test_write_split_jsonl_path_b_round_trip(tmp_path: Path) -> None:
    report, _ = _make_split_report()
    out = tmp_path / "test.jsonl"
    n = write_split_jsonl(report, "test", out, mode="path_b", health=_HEALTH, now=_NOW)
    assert n > 0, "test split unexpectedly empty"

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == n, f"file row count {len(rows)} != return value {n}"
    for ex in rows:
        assert set(ex.keys()) == {"messages"}, f"unexpected top-level keys: {ex.keys()}"
        msgs = ex["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], "role order drifted"
        # Path B contract: user content carries the YAML block + date stamp.
        user_content = msgs[0]["content"]
        assert "YAML:" in user_content, "Path B user content is missing YAML block"
        assert "2026-04-25" in user_content, "Path B user content is missing date"
        # And the assistant content is the literal training answer.
        assert msgs[1]["content"], "assistant content must not be empty"


def test_write_split_jsonl_path_a_excludes_yaml(tmp_path: Path) -> None:
    report, _ = _make_split_report()
    out = tmp_path / "ablation.jsonl"
    n = write_split_jsonl(report, "test", out, mode="path_a")
    assert n > 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for ex in rows:
        user_content = ex["messages"][0]["content"]
        assert "YAML:" not in user_content, (
            "Path A leaked the system block — emitter is mis-routed"
        )
        assert "ROLE:" not in user_content, "Path A leaked directive header"


def test_write_split_jsonl_path_b_requires_health_and_now(tmp_path: Path) -> None:
    report, _ = _make_split_report()
    out = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="Path B requires"):
        write_split_jsonl(report, "train", out, mode="path_b")


def test_write_split_jsonl_per_split_row_counts(tmp_path: Path) -> None:
    # Sum of train+val+test JSONL line counts must equal the total pool size.
    report, pool = _make_split_report()
    counts = {}
    for split in ("train", "val", "test"):
        out = tmp_path / f"{split}.jsonl"
        counts[split] = write_split_jsonl(
            report, split, out, mode="path_b", health=_HEALTH, now=_NOW
        )
    assert sum(counts.values()) == len(pool), f"row counts don't sum: {counts}"


def test_write_split_jsonl_assistant_content_matches_pool_output(tmp_path: Path) -> None:
    # Path B's assistant content must be the exact `output` string from the
    # source SftRecord — no rewriting, no truncation.
    report, _ = _make_split_report()
    expected = {a.output for a in report.assignments if a.split == "test"}
    out = tmp_path / "test.jsonl"
    write_split_jsonl(report, "test", out, mode="path_b", health=_HEALTH, now=_NOW)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    actual = {ex["messages"][1]["content"] for ex in rows}
    assert actual == expected, f"assistant strings drifted: missing {expected - actual}"
