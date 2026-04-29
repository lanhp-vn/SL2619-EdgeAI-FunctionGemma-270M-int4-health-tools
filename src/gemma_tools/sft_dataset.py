"""Load, dedupe, classify, leakage-scan, and split the Gemma 3 SFT dataset.

This module owns the offline pipeline that turns a chatbot-distilled
Alpaca-style JSON pool (`tools/data/clean_sft_dataset.json`) into the
training artifacts consumed by the server-side QLoRA SFT job:

    tools/data/sft_v1.{train,val,test}.jsonl  (Path B — composed prompt,
                                               TRL conversational format)
    tools/data/sft_v1_pathA.{train,val,test}.jsonl  (raw pairs, ablation)

Authority chain:
  - Plan:           docs/plans/AI-models/a55-fine-tune-gemma.md §4 D1-D4
  - Prompt rules:   docs/conventions/16-slm-system-prompt.md §4 (R-1..R-10)
  - Composer:       gemma_tools.prompt_composer.compose_user_text
  - Bench prompts:  tools/data/prompts.yaml (held-out test must include the
                    five exact-match bench hits — see §4 D2 leakage gate)

Convention deviation (re-stated from health_table.py): stdlib + PyYAML only —
no Pydantic — even though docs/conventions/09-code-style-python.md §6 mandates
Pydantic for runtime config. This is offline tooling that runs against a
fixture; matching health_table's schema-parser style keeps the test surface
flat and the dependency graph small.

D1a (this commit) ships only the loader + dataclass + schema validation. The
classifier, leakage scanner, splitter, and JSONL emitter land in subsequent
chunks per R2 (write -> test -> fix -> next chunk).
"""

from __future__ import annotations

import difflib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

from gemma_tools.health_table import HealthTable
from gemma_tools.prompt_composer import compose_user_text

SftClass = Literal["fact_lookup", "fact_absence", "domain_refusal", "summarization"]
SplitName = Literal["train", "val", "test"]
RoutingReason = Literal[
    "stratified_random",
    "bench_exact",
    "bench_near",
    "same_instruction_conflict",
    "cluster_output",
    "cluster_instruction",
]
SFT_CLASSES: tuple[SftClass, ...] = (
    "fact_lookup",
    "fact_absence",
    "domain_refusal",
    "summarization",
)


@dataclass(frozen=True, slots=True)
class SftRecord:
    """A single Alpaca-shape training pair from the chatbot-distilled pool.

    The pool was produced by the verbatim §6 dataset-generation prompt run
    against multiple frontier chatbots (Gemini, Claude, ChatGPT, ...). Every
    record is a single-turn `instruction -> output`; the `input` slot stays
    empty because the YAML lives in the system-side prompt at compose time,
    not in a per-example variable. Keeping `input` in the dataclass anyway
    preserves Alpaca-format round-trippability if we ever want to publish.
    """

    instruction: str
    output: str
    input: str = ""


# Allowed JSON keys in the pool. Anything else is a schema error — extra
# fields would silently absorb future drift, so reject loudly.
_ALLOWED_KEYS: frozenset[str] = frozenset({"instruction", "input", "output"})
_REQUIRED_KEYS: frozenset[str] = frozenset({"instruction", "input", "output"})


def _require_string(value: object, ctx: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{ctx}: expected str, got {type(value).__name__}")
    return value


def _parse_record(raw: object, ctx: str) -> SftRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: expected mapping, got {type(raw).__name__}")

    keys = {k for k in raw if isinstance(k, str)}
    extra = keys - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"{ctx}: unexpected keys {sorted(extra)}")
    missing = _REQUIRED_KEYS - keys
    if missing:
        raise ValueError(f"{ctx}: missing keys {sorted(missing)}")

    instruction = _require_string(raw["instruction"], f"{ctx}.instruction").strip()
    output = _require_string(raw["output"], f"{ctx}.output").strip()
    input_field = _require_string(raw["input"], f"{ctx}.input")

    if not instruction:
        raise ValueError(f"{ctx}.instruction: empty after strip")
    if not output:
        raise ValueError(f"{ctx}.output: empty after strip")
    # v1 contract: `input` is always blank because YAML grounding is injected
    # at compose time. If a future revision needs per-example context,
    # introduce a v2 schema rather than overloading this field.
    if input_field.strip():
        raise ValueError(f"{ctx}.input: must be empty in v1 schema, got {input_field!r}")

    return SftRecord(instruction=instruction, output=output, input="")


def load_sft_pool(path: Path) -> tuple[SftRecord, ...]:
    """Read the chatbot-distilled JSON pool, validate, return frozen records.

    Behaviour:
      - Whitespace at the edges of `instruction` and `output` is stripped.
      - Empty `instruction` or `output` is a schema error (would teach the
        model to emit blanks).
      - Duplicates are NOT removed here; the dedupe step is its own layer
        (see D1b in the next chunk) so callers can audit the raw pool count
        before we shrink it.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: on any schema violation, with a self-locating message.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top-level must be a JSON array, got {type(raw).__name__}")

    records: list[SftRecord] = []
    for i, item in enumerate(raw):
        records.append(_parse_record(item, f"{path}[{i}]"))
    return tuple(records)


# --------------------------------------------------------------------------
# D1b — exact (instruction, output) duplicate removal.
#
# The chatbot-distilled pool contains intentional paraphrase clusters: many
# rows share the same `output` (the canonical YAML answer) under different
# `instruction` phrasings — that's the whole point of distillation. We drop
# only ROWS that are exact case-folded duplicates on BOTH fields, because
# those teach the model nothing new and skew class proportions during the
# stratified split. Rows that share an instruction but differ in output
# (rare, usually a chatbot disagreement) are kept — caller can audit them
# separately if needed.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DedupeReport:
    """Summary returned by `dedupe_pool` for audit logging."""

    input_count: int
    output_count: int
    duplicates_removed: int


def _dedupe_key(rec: SftRecord) -> tuple[str, str]:
    # Case-fold + collapse internal whitespace so "HR?" and "hr ?" collide.
    inst = " ".join(rec.instruction.lower().split())
    out = " ".join(rec.output.lower().split())
    return (inst, out)


def dedupe_pool(records: Iterable[SftRecord]) -> tuple[tuple[SftRecord, ...], DedupeReport]:
    """Drop exact (case-folded instruction, case-folded output) duplicates.

    First-seen wins; insertion order is preserved so downstream stratified
    splits are deterministic regardless of dict ordering.
    """
    seen: set[tuple[str, str]] = set()
    kept: list[SftRecord] = []
    total = 0
    for rec in records:
        total += 1
        key = _dedupe_key(rec)
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    return tuple(kept), DedupeReport(
        input_count=total,
        output_count=len(kept),
        duplicates_removed=total - len(kept),
    )


# --------------------------------------------------------------------------
# D1c — class auto-tagger.
#
# We classify each record by content, not by source. The four classes mirror
# the bench taxonomy (tools/data/prompts.yaml header) and feed the stratified
# splitter so train/val/test keep proportional class coverage.
#
# Decision order (most specific first; first match wins):
#   1. domain_refusal — output contains the canonical refusal substring
#                       from prompt_composer._SYSTEM_TEMPLATE R-3 / R-7.
#   2. fact_absence   — output contains the canonical "not in record" string
#                       (also from the system template).
#   3. summarization  — output is multi-fact (>= 2 commas, OR >= 80 chars,
#                       OR contains a newline). Refusal/absence cases were
#                       already filtered above so they can't trip this.
#                       (We also accept imperative-summary instructions, but
#                       output shape is the more reliable signal — chatbots
#                       sometimes summarize without the user using "summarize".)
#   4. fact_lookup    — everything else (single-fact retrieval).
# --------------------------------------------------------------------------

_REFUSAL_MARKER = "i answer questions from your health record only"
_ABSENCE_MARKERS: tuple[str, ...] = (
    "not in record",
    "not in your record",
)
# Instructions that strongly imply a summary; checked AFTER output-shape rules
# so a chatbot that emitted a single-fact answer to "summarize my BP" still
# gets routed to fact_lookup (the data drives the label, not the verb).
_SUMMARY_INSTRUCTION_RE = re.compile(
    r"^\s*(summarize|summary|sum up|condense|condensed|overview|outline|list (all|my))\b",
    re.IGNORECASE,
)


def classify_record(rec: SftRecord) -> SftClass:
    """Return the four-class label for `rec` based on output content.

    See the comment block above for decision order. Pure-functional and
    cheap; the splitter calls this for every record without caching.
    """
    out_lower = rec.output.lower()
    if _REFUSAL_MARKER in out_lower:
        return "domain_refusal"
    for marker in _ABSENCE_MARKERS:
        if marker in out_lower:
            return "fact_absence"

    is_multi_comma = rec.output.count(",") >= 2
    is_long = len(rec.output) >= 80
    has_newline = "\n" in rec.output
    if is_multi_comma or is_long or has_newline:
        return "summarization"

    if _SUMMARY_INSTRUCTION_RE.match(rec.instruction):
        # Imperative summary verb but the answer is short enough that a
        # single-fact reply would fit. Still treat as summarization because
        # routing it to fact_lookup would let a "summarize my BP" example
        # poison the fact_lookup paraphrase clusters.
        return "summarization"

    return "fact_lookup"


def class_distribution(records: Iterable[SftRecord]) -> dict[SftClass, int]:
    """Count records per class — convenience for audit logs."""
    counts: dict[SftClass, int] = {
        "fact_lookup": 0,
        "fact_absence": 0,
        "domain_refusal": 0,
        "summarization": 0,
    }
    for rec in records:
        counts[classify_record(rec)] += 1
    return counts


# --------------------------------------------------------------------------
# D1d — bench leakage scanner.
#
# `tools/data/prompts.yaml` is the held-out evaluation harness. Any pool row
# that is identical to or a near-paraphrase of a bench prompt MUST be routed
# into the test split (and never into train/val) so we don't measure
# memorization. The scanner enumerates every bench-prompt match it can see
# and hands the verdict to the splitter; it also prints a human-readable
# audit summary because the user asked to review findings before splits are
# finalized.
# --------------------------------------------------------------------------

# Levenshtein-style similarity threshold via difflib.SequenceMatcher.ratio().
# Empirically calibrated against the deduped 1259-row pool x 15 bench prompts:
#   - 0.85 misses "what is my heart rate?" vs "what is my current heart rate?"
#     (ratio 0.846) — a clear semantic duplicate that must route to test.
#   - 0.80 catches the above and similar borderline pairs while still leaving
#     "what is my BP?" vs "what is my heart rate?" (ratio ~0.55) far below.
# Total bench-pool hits at 0.80: ~85 / 1259 (~6.7%); easily absorbable by
# the test split's ~10% capacity.
NEAR_DUPLICATE_RATIO = 0.80


@dataclass(frozen=True, slots=True)
class BenchPrompt:
    """One row of `tools/data/prompts.yaml` — id, class, text only."""

    id: str
    cls: str
    text: str


@dataclass(frozen=True, slots=True)
class LeakageHit:
    """A single SFT-pool row that matches a bench prompt."""

    pool_index: int
    instruction: str
    output: str
    ratio: float  # 1.0 for exact case-folded match


@dataclass(frozen=True, slots=True)
class PromptLeakage:
    """Per-bench-prompt findings."""

    prompt: BenchPrompt
    exact: tuple[LeakageHit, ...] = ()
    near: tuple[LeakageHit, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchLeakageReport:
    """Aggregated scan result for the whole pool x bench cross-product."""

    per_prompt: tuple[PromptLeakage, ...] = field(default_factory=tuple)

    def all_hit_indices(self) -> frozenset[int]:
        """Pool indices that must be routed to the test split."""
        out: set[int] = set()
        for entry in self.per_prompt:
            for h in entry.exact:
                out.add(h.pool_index)
            for h in entry.near:
                out.add(h.pool_index)
        return frozenset(out)

    def summary_lines(self) -> list[str]:
        """Human-readable audit lines, one per bench prompt."""
        lines: list[str] = []
        for entry in self.per_prompt:
            n_exact = len(entry.exact)
            n_near = len(entry.near)
            tag = "ok" if (n_exact + n_near) == 0 else "leak"
            lines.append(
                f"  [{tag}] {entry.prompt.id:<4} {entry.prompt.cls:<14} "
                f"exact={n_exact} near={n_near}  text={entry.prompt.text!r}"
            )
        return lines


def load_bench_prompts(path: Path) -> tuple[BenchPrompt, ...]:
    """Read tools/data/prompts.yaml and return only the fields we need."""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or "prompts" not in raw:
        raise ValueError(f"{path}: missing top-level 'prompts' key")
    items = raw["prompts"]
    if not isinstance(items, list):
        raise ValueError(f"{path}: 'prompts' must be a list")
    prompts: list[BenchPrompt] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}.prompts[{i}]: expected mapping")
        for key in ("id", "class", "text"):
            if key not in item:
                raise ValueError(f"{path}.prompts[{i}]: missing {key!r}")
        prompts.append(
            BenchPrompt(
                id=str(item["id"]),
                cls=str(item["class"]),
                text=str(item["text"]),
            )
        )
    return tuple(prompts)


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def scan_bench_leakage(
    records: Sequence[SftRecord],
    prompts: Sequence[BenchPrompt],
    *,
    near_ratio: float = NEAR_DUPLICATE_RATIO,
) -> BenchLeakageReport:
    """Cross-reference every bench prompt against every pool record.

    `records` order is significant — the indices stored in `LeakageHit`
    point back into this exact sequence so the splitter can route them
    deterministically.
    """
    pool_norm = [_normalize(r.instruction) for r in records]
    per_prompt: list[PromptLeakage] = []
    for prompt in prompts:
        target = _normalize(prompt.text)
        exact: list[LeakageHit] = []
        near: list[LeakageHit] = []
        for idx, (rec, norm) in enumerate(zip(records, pool_norm, strict=True)):
            if norm == target:
                exact.append(
                    LeakageHit(
                        pool_index=idx,
                        instruction=rec.instruction,
                        output=rec.output,
                        ratio=1.0,
                    )
                )
                continue
            ratio = difflib.SequenceMatcher(None, norm, target).ratio()
            if ratio >= near_ratio:
                near.append(
                    LeakageHit(
                        pool_index=idx,
                        instruction=rec.instruction,
                        output=rec.output,
                        ratio=ratio,
                    )
                )
        per_prompt.append(
            PromptLeakage(
                prompt=prompt,
                exact=tuple(exact),
                near=tuple(near),
            )
        )
    return BenchLeakageReport(per_prompt=tuple(per_prompt))


# --------------------------------------------------------------------------
# D1e — paraphrase-aware stratified splitter.
#
# Routing rules (deterministic; first match wins per row, in this order):
#   1. bench_exact                  — pool row's instruction == bench prompt
#                                     (case-folded). Forces TEST.
#   2. bench_near                   — pool row's instruction has SequenceMatcher
#                                     ratio >= NEAR_DUPLICATE_RATIO against any
#                                     bench prompt. Forces TEST.
#   3. same_instruction_conflict    — pool row shares case-folded instruction
#                                     with a bench-routed row but has a
#                                     different output (chatbot answer-style
#                                     disagreement). Forces TEST so the
#                                     conflicting answers never reach train.
#   4. cluster_output               — for fact_lookup / fact_absence: same
#                                     case-folded output as a bench-routed row.
#                                     Forces TEST so the test bench never asks
#                                     a fact whose answer string was already
#                                     paraphrase-trained.
#   5. cluster_instruction          — for domain_refusal / summarization:
#                                     instruction-similarity >=
#                                     CLUSTER_SIMILARITY against a bench-routed
#                                     row, same class. Forces TEST.
#
# Everything else is split by stratified random allocation per class. Train
# / val / test ratios sum to 1.0; with force-routing the actual test fraction
# usually overshoots `test_ratio` slightly and val/train absorb the slack
# proportionally to keep class distribution close to the pool average.
# --------------------------------------------------------------------------

CLUSTER_SIMILARITY = NEAR_DUPLICATE_RATIO  # alias — same threshold for now.

# Hard guard: if any class lands above this fraction in the test split,
# stratified balance has been compromised and the user must intervene.
TEST_CLASS_DRAIN_LIMIT = 0.50


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Per-row split decision plus its provenance for the audit artifact."""

    pool_index: int
    instruction: str
    output: str
    cls: SftClass
    split: SplitName
    routing_reason: RoutingReason
    matched_bench_id: str | None = None
    matched_bench_text: str | None = None
    similarity: float | None = None

    def to_audit_dict(self) -> dict[str, object]:
        return {
            "pool_index": self.pool_index,
            "instruction": self.instruction,
            "output": self.output,
            "class": self.cls,
            "split": self.split,
            "routing_reason": self.routing_reason,
            "matched_bench_id": self.matched_bench_id,
            "matched_bench_text": self.matched_bench_text,
            "similarity": self.similarity,
        }


@dataclass(frozen=True, slots=True)
class SplitReport:
    """Output of `split_pool` — assignments + per-split-class histogram."""

    assignments: tuple[SplitAssignment, ...]
    by_split_class: dict[SplitName, dict[SftClass, int]]
    total: int

    def write_audit_jsonl(self, path: Path) -> None:
        """Emit one JSON line per row with every routing field set."""
        with path.open("w", encoding="utf-8") as fh:
            for a in self.assignments:
                fh.write(json.dumps(a.to_audit_dict(), ensure_ascii=False))
                fh.write("\n")

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for split_name in ("train", "val", "test"):
            counts = self.by_split_class[split_name]
            total = sum(counts.values())
            pct = (total / self.total * 100.0) if self.total else 0.0
            details = "  ".join(
                f"{cls}={counts[cls]}" for cls in SFT_CLASSES
            )
            lines.append(
                f"  {split_name:<5} n={total:<5} ({pct:5.1f}%)  {details}"
            )
        return lines


@dataclass(frozen=True, slots=True)
class _RouteInfo:
    """Internal per-routed-row provenance carried until SplitAssignment lands."""

    reason: RoutingReason
    bench_id: str | None = None
    bench_text: str | None = None
    similarity: float | None = None


def split_pool(
    records: Sequence[SftRecord],
    leakage: BenchLeakageReport,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    cluster_similarity: float = CLUSTER_SIMILARITY,
) -> SplitReport:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio}"
        )
    if not 0.0 < cluster_similarity <= 1.0:
        raise ValueError(f"cluster_similarity must be in (0, 1], got {cluster_similarity}")

    n = len(records)
    classes: list[SftClass] = [classify_record(r) for r in records]
    norms_inst: list[str] = [_normalize(r.instruction) for r in records]
    norms_out: list[str] = [_normalize(r.output) for r in records]

    routing: dict[int, _RouteInfo] = {}

    def _set(idx: int, info: _RouteInfo) -> None:
        # First writer wins; routing-rule order is the priority.
        if idx not in routing:
            routing[idx] = info

    # Step 1 — bench-leak hits. Exact before near so a row matching both
    # records bench_exact, not bench_near.
    for entry in leakage.per_prompt:
        for hit in entry.exact:
            _set(
                hit.pool_index,
                _RouteInfo(
                    reason="bench_exact",
                    bench_id=entry.prompt.id,
                    bench_text=entry.prompt.text,
                    similarity=1.0,
                ),
            )
    for entry in leakage.per_prompt:
        for hit in entry.near:
            _set(
                hit.pool_index,
                _RouteInfo(
                    reason="bench_near",
                    bench_id=entry.prompt.id,
                    bench_text=entry.prompt.text,
                    similarity=hit.ratio,
                ),
            )

    # Step 2 — same-instruction-conflict labeling. The scanner already routes
    # every pool row whose normalized instruction matches a bench prompt, so
    # all sibling rows with the same instruction text are already in
    # `routing` after step 1. What step 2 does is RELABEL: when two-or-more
    # already-routed rows share a normalized instruction AND disagree on the
    # normalized output (chatbot answer-style disagreement that survived
    # dedupe), the first occurrence keeps its bench_exact/bench_near label
    # and the rest become same_instruction_conflict. This way the audit
    # artifact distinguishes "this row IS the answer the bench asks for"
    # from "this row was pulled because its sibling was".
    inst_groups: dict[str, list[int]] = {}
    for i in routing:
        inst_groups.setdefault(norms_inst[i], []).append(i)
    for idxs in inst_groups.values():
        if len(idxs) < 2:
            continue
        idxs.sort()  # lowest pool_index keeps original label
        if len({norms_out[i] for i in idxs}) <= 1:
            continue  # all rows agree on the answer — no conflict
        for i in idxs[1:]:
            prev = routing[i]
            routing[i] = _RouteInfo(
                reason="same_instruction_conflict",
                bench_id=prev.bench_id,
                bench_text=prev.bench_text,
                similarity=prev.similarity,
            )

    # Step 3 — cluster_output for fact_lookup ONLY. fact_absence rows all
    # share canonical output "not in record" by construction (same as
    # refusals share the canonical refusal string), so output-based
    # clustering would collapse the entire class into test. The semantic
    # uniqueness in fact_absence lives in the question, not the answer —
    # those are caught by cluster_instruction below.
    routed_out_keys: set[str] = {
        norms_out[i] for i in routing if classes[i] == "fact_lookup"
    }
    for i, key in enumerate(norms_out):
        if i in routing:
            continue
        if classes[i] != "fact_lookup":
            continue
        if key in routed_out_keys:
            _set(i, _RouteInfo(reason="cluster_output"))

    # Step 4 — cluster_instruction for fact_absence / domain_refusal /
    # summarization. Single-hop: only direct neighbors of a bench-routed
    # row in the same class. Transitive expansion would risk dragging the
    # whole class into test, which the user explicitly flagged as a drain
    # risk. (fact_lookup is already covered by cluster_output, which is
    # cheaper and more precise when the fact answer is canonical.)
    cluster_inst_classes: tuple[SftClass, ...] = (
        "fact_absence", "domain_refusal", "summarization",
    )
    routed_now: list[int] = list(routing.keys())
    for j, j_inst in enumerate(norms_inst):
        if j in routing:
            continue
        if classes[j] not in cluster_inst_classes:
            continue
        best_sim = 0.0
        for i in routed_now:
            if classes[i] != classes[j]:
                continue
            sim = difflib.SequenceMatcher(None, norms_inst[i], j_inst).ratio()
            if sim > best_sim:
                best_sim = sim
        if best_sim >= cluster_similarity:
            _set(j, _RouteInfo(reason="cluster_instruction", similarity=best_sim))

    # Step 5 — drain guard. If any class is over-routed to test the split is
    # invalid; surface immediately so the caller can lower thresholds or
    # extend the dataset.
    test_class_counts: dict[SftClass, int] = dict.fromkeys(SFT_CLASSES, 0)
    pool_class_counts: dict[SftClass, int] = dict.fromkeys(SFT_CLASSES, 0)
    for i in range(n):
        pool_class_counts[classes[i]] += 1
        if i in routing:
            test_class_counts[classes[i]] += 1
    for cls, denom in pool_class_counts.items():
        if denom == 0:
            continue
        share = test_class_counts[cls] / denom
        if share > TEST_CLASS_DRAIN_LIMIT:
            raise ValueError(
                f"class {cls!r} drained: {test_class_counts[cls]}/{denom} = "
                f"{share:.1%} routed to test (limit {TEST_CLASS_DRAIN_LIMIT:.0%})"
            )

    # Step 6 — stratified val allocation from the remainder. We want the
    # final val count to match `val_ratio` of the whole pool as closely as
    # the routing budget allows; we therefore spread val allocations
    # proportionally across remainder classes.
    remainder: list[int] = [i for i in range(n) if i not in routing]
    target_val_total = round(n * val_ratio)
    rng = random.Random(seed)

    val_indices: set[int] = set()
    if remainder:
        # Stable per-class shuffle: within each class we shuffle once with
        # the same RNG, so seed pins the assignment.
        by_class: dict[SftClass, list[int]] = {c: [] for c in SFT_CLASSES}
        for i in remainder:
            by_class[classes[i]].append(i)
        for cls in SFT_CLASSES:
            rng.shuffle(by_class[cls])

        rem_total = len(remainder)
        for cls in SFT_CLASSES:
            cls_remainder = by_class[cls]
            if not cls_remainder:
                continue
            # Proportional val share for this class. round() is fine; tiny
            # off-by-one drift between classes washes out at scale.
            val_count = round(len(cls_remainder) * (target_val_total / rem_total))
            val_count = min(val_count, len(cls_remainder))
            for i in cls_remainder[:val_count]:
                val_indices.add(i)

    # Step 7 — materialize SplitAssignment records.
    assignments: list[SplitAssignment] = []
    for i, rec in enumerate(records):
        cls = classes[i]
        if i in routing:
            info = routing[i]
            split: SplitName = "test"
            assignments.append(
                SplitAssignment(
                    pool_index=i,
                    instruction=rec.instruction,
                    output=rec.output,
                    cls=cls,
                    split=split,
                    routing_reason=info.reason,
                    matched_bench_id=info.bench_id,
                    matched_bench_text=info.bench_text,
                    similarity=info.similarity,
                )
            )
            continue
        split = "val" if i in val_indices else "train"
        assignments.append(
            SplitAssignment(
                pool_index=i,
                instruction=rec.instruction,
                output=rec.output,
                cls=cls,
                split=split,
                routing_reason="stratified_random",
            )
        )

    # Step 8 — by-split-class histogram (audit summary).
    by_split_class: dict[SplitName, dict[SftClass, int]] = {
        "train": dict.fromkeys(SFT_CLASSES, 0),
        "val": dict.fromkeys(SFT_CLASSES, 0),
        "test": dict.fromkeys(SFT_CLASSES, 0),
    }
    for a in assignments:
        by_split_class[a.split][a.cls] += 1

    return SplitReport(
        assignments=tuple(assignments),
        by_split_class=by_split_class,
        total=n,
    )


# --------------------------------------------------------------------------
# D1f — JSONL emitter for Path B (composed prompt, primary training shape)
# and Path A (raw instruction -> output, ablation only).
#
# Path B is the deployed-shape format: every example carries the full
# directive system + YAML + date so the model trains on the same prompt
# structure it will see at bench time. Source of truth for the user-turn
# composition is `prompt_composer.compose_user_text` — we route through
# it so a future tweak to the directive template propagates automatically.
#
# Path A drops the YAML and emits {"role":"user", "content": <instruction>}.
# It exists ONLY to support an ablation where we measure whether a
# memorization-only baseline matches Path B on bench accuracy. If they
# tie, the composer was wasted compute. If Path B wins, retrieval-style
# training is justified. We never SHIP a Path A model.
# --------------------------------------------------------------------------

# TRL conversational format spec: a JSONL where each line is
# {"messages": [{"role":"user","content":...},
#               {"role":"assistant","content":...}]}.
# Reference: HuggingFace `trl.SFTTrainer` formats — `messages` field
# triggers the chat-template path and the trainer applies the model's
# tokenizer.chat_template at collation. We therefore must NOT pre-template
# (no <start_of_turn> markers) — that's the trainer's job.
_TrlMessage = dict[str, str]
_TrlExample = dict[str, list[_TrlMessage]]


def _to_path_b_example(
    rec: SftRecord,
    health: HealthTable,
    now: date,
) -> _TrlExample:
    user_text = compose_user_text(health, now, rec.instruction)
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": rec.output},
        ]
    }


def _to_path_a_example(rec: SftRecord) -> _TrlExample:
    return {
        "messages": [
            {"role": "user", "content": rec.instruction},
            {"role": "assistant", "content": rec.output},
        ]
    }


def _records_for_split(
    report: SplitReport,
    split: SplitName,
) -> list[SftRecord]:
    return [
        SftRecord(instruction=a.instruction, output=a.output, input="")
        for a in report.assignments
        if a.split == split
    ]


def write_split_jsonl(
    report: SplitReport,
    split: SplitName,
    out_path: Path,
    *,
    mode: Literal["path_a", "path_b"],
    health: HealthTable | None = None,
    now: date | None = None,
) -> int:
    """Emit the split as a TRL conversational JSONL file. Returns row count.

    `mode="path_b"` requires `health` and `now` so the composer can stamp
    every example with the directive system + YAML + date. `mode="path_a"`
    ignores them.
    """
    records = _records_for_split(report, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if mode == "path_b":
                if health is None or now is None:
                    raise ValueError("Path B requires non-None `health` and `now`.")
                example = _to_path_b_example(rec, health, now)
            else:
                example = _to_path_a_example(rec)
            fh.write(json.dumps(example, ensure_ascii=False))
            fh.write("\n")
    return len(records)


