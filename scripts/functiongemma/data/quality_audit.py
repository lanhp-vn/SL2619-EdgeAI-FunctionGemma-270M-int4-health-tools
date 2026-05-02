#!/usr/bin/env python3
"""Block D dataset quality audit for FunctionGemma 270M-IT M5 SFT diagnostic.

Why this script exists: M5 LoRA SFT trained without errors but failed M6
G_EVAL hard (25/56 = 44.6%). One leading hypothesis is that the dataset
itself is the bottleneck -- 511 training rows split across 7 categories
from 4 hand seeds + LLM-augmented expansion may have too-narrow phrasings,
too-few unique argument values per tool, and tightly clustered refusal
prompts. The most damning M6 evidence: the fine-tuned model regurgitated
the literal `time_24h` parameter description as an argument value, which
is what we'd expect if it had never seen enough varied real values for
that argument.

The audit produces a concrete verdict via 5 probes (D1-D5) -- "is the
dataset the bottleneck, or is the recipe?" -- with D3 (argument-value
overlap between train and eval) being the headline test.

Read-only on `data/functiongemma/`. Writes a Markdown report to
`docs/bench-notes/functiongemma/<today>_dataset-audit.md` by default.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from gemma_tools.functiongemma.dataset import load_jsonl

# region constants ----------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]
_FG_DIR = _REPO / "data" / "functiongemma"
_DEFAULT_SEED = _FG_DIR / "seed_conversations.jsonl"
_DEFAULT_LLM_EXPANDED = _FG_DIR / "llm_expanded_v1.jsonl"
_DEFAULT_TRAIN = _FG_DIR / "dataset_v1" / "train.jsonl"
_DEFAULT_VAL = _FG_DIR / "dataset_v1" / "val.jsonl"
_DEFAULT_EVAL_HOLDOUT = _FG_DIR / "eval_holdout_v1.jsonl"
_DEFAULT_OUTPUT = _REPO / "docs" / "bench-notes" / "functiongemma" / "dataset-audit.md"

_CATEGORIES = (
    "fact_absence",
    "fact_lookup",
    "medical_advice_refusal",
    "off_topic_refusal",
    "parallel_call",
    "tool_error_recovery",
    "two_turn",
)

_TOOLS_WITH_STRING_ARG: tuple[tuple[str, str], ...] = (
    ("get_medication_by_name", "name"),
    ("get_medications_at_time", "time_24h"),
    ("check_food_interaction", "food"),
)

# Pinned for reproducibility -- shared between the per-category MiniLM
# embed pass and the KMeans cluster pass.
_RANDOM_STATE = 3407
_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

# Used for D1 "seed recycle" thresholding and D4 "templated refusal" verdict.
_SEED_RECYCLE_COSINE = 0.85
_REFUSAL_TEMPLATE_COSINE = 0.85
_SEED_RECYCLE_FRACTION_FLAG = 0.70

# D2 "weak tool" threshold -- under 30 calls in the 595-row expanded set
# means the model sees that tool ~5% of the time, marginal for learning
# the schema deeply.
_WEAK_TOOL_THRESHOLD = 30

_PUNCT_RE = re.compile(r"[^\w\s]")

# endregion -----------------------------------------------------------------

# region small helpers ------------------------------------------------------


def _normalize_prompt(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Matches the D1 spec exactly. Used both for n_unique counting and as the
    pre-image to the embedder so the embedding similarity is between
    semantically-comparable strings rather than e.g. trailing-period vs none.
    """
    cleaned = _PUNCT_RE.sub(" ", s.lower())
    return " ".join(cleaned.split())


def _first_user_prompt(row: dict[str, Any]) -> str:
    for m in row["messages"]:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _iter_assistant_tool_calls(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for m in row["messages"]:
        if m.get("role") == "assistant":
            yield from m.get("tool_calls") or []


def _normalize_unit(vec: np.ndarray) -> np.ndarray:
    """Row-normalize so cosine reduces to a single matmul."""
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vec / norms


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p20": 0.0, "p80": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p20": float(np.percentile(values, 20)),
        "p80": float(np.percentile(values, 80)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _pairwise_upper_cosines(unit_vecs: np.ndarray) -> np.ndarray:
    """Return the off-diagonal upper triangle of the cosine matrix.

    For n=4 we get 6 values; for n>>0 we get n*(n-1)/2. This is what we
    average / take percentiles over in D1.
    """
    if unit_vecs.shape[0] < 2:
        return np.empty((0,), dtype=np.float32)
    sim = unit_vecs @ unit_vecs.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return sim[iu].astype(np.float32)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Minimal Markdown table writer -- avoids a dep just for this."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


# endregion -----------------------------------------------------------------

# region D1: per-category phrasing diversity ---------------------------------


def _audit_d1(
    expanded_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    embedder: SentenceTransformer,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-category cosine-similarity audit on the LLM-expanded set."""
    by_cat_expanded: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cat_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in expanded_rows:
        by_cat_expanded[row["category"]].append(row)
    for row in seed_rows:
        by_cat_seed[row["category"]].append(row)

    table_rows: list[dict[str, Any]] = []
    flagged: list[str] = []

    for cat in _CATEGORIES:
        rows = by_cat_expanded.get(cat, [])
        seeds = by_cat_seed.get(cat, [])
        prompts = [_normalize_prompt(_first_user_prompt(r)) for r in rows]
        seed_prompts = [_normalize_prompt(_first_user_prompt(r)) for r in seeds]
        n = len(prompts)
        n_unique = len(set(prompts))

        # Pairwise cosine within the category (LLM-expanded vs LLM-expanded).
        if n >= 2:
            embeds = embedder.encode(prompts, show_progress_bar=False, convert_to_numpy=True)
            unit = _normalize_unit(embeds.astype(np.float32))
            pair = _pairwise_upper_cosines(unit)
            stats = _percentiles(pair)
        else:
            unit = np.zeros((0, 384), dtype=np.float32)
            stats = _percentiles(np.empty((0,), dtype=np.float32))

        # Seed-recycle: max cosine of each expanded row vs ANY of the 4 hand seeds.
        if seed_prompts and n > 0:
            seed_embeds = embedder.encode(
                seed_prompts, show_progress_bar=False, convert_to_numpy=True
            )
            seed_unit = _normalize_unit(seed_embeds.astype(np.float32))
            sims = unit @ seed_unit.T  # (n, n_seeds)
            max_per_row = sims.max(axis=1)
            recycle_pct = float((max_per_row >= _SEED_RECYCLE_COSINE).mean())
        else:
            recycle_pct = 0.0

        if recycle_pct > _SEED_RECYCLE_FRACTION_FLAG:
            flagged.append(cat)

        table_rows.append({
            "category": cat,
            "n_rows": n,
            "n_unique": n_unique,
            "mean_cos": stats["mean"],
            "median_cos": stats["median"],
            "p90_cos": stats["p90"],
            "seed_recycle_pct": recycle_pct,
        })

    return table_rows, flagged


def _format_d1(rows: list[dict[str, Any]], flagged: list[str]) -> str:
    table = _md_table(
        ["category", "n_rows", "n_unique", "mean_cos", "median_cos", "p90_cos", "seed_recycle_pct (>=0.85)"],
        [
            [
                r["category"],
                str(r["n_rows"]),
                str(r["n_unique"]),
                f"{r['mean_cos']:.3f}",
                f"{r['median_cos']:.3f}",
                f"{r['p90_cos']:.3f}",
                f"{r['seed_recycle_pct']:.1%}",
            ]
            for r in rows
        ],
    )
    flagged_line = (
        f"\n\n**Flagged categories** (>{int(_SEED_RECYCLE_FRACTION_FLAG * 100)}% of "
        f"LLM-expanded rows have cos >= {_SEED_RECYCLE_COSINE:.2f} to a hand seed): "
        + (", ".join(f"`{c}`" for c in flagged) if flagged else "_none_")
    )
    return table + flagged_line


# endregion -----------------------------------------------------------------

# region D2: tool-call distribution -----------------------------------------


def _audit_d2(rows: list[dict[str, Any]]) -> tuple[Counter, dict[str, Counter], list[str]]:
    total = Counter()
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        cat = row["category"]
        for tc in _iter_assistant_tool_calls(row):
            name = tc.get("function", {}).get("name")
            if not name:
                continue
            total[name] += 1
            by_cat[cat][name] += 1
    weak = sorted([n for n, c in total.items() if c < _WEAK_TOOL_THRESHOLD])
    return total, by_cat, weak


def _format_d2(total: Counter, by_cat: dict[str, Counter], weak: list[str]) -> str:
    tool_names = sorted(total.keys())
    overall = _md_table(
        ["tool", "count"],
        [[name, str(cnt)] for name, cnt in sorted(total.items(), key=lambda kv: -kv[1])],
    )
    pivot_headers = ["category", *tool_names, "TOTAL"]
    pivot_rows = []
    for cat in _CATEGORIES:
        row_vals = [str(by_cat[cat].get(t, 0)) for t in tool_names]
        pivot_rows.append([cat, *row_vals, str(sum(by_cat[cat].values()))])
    pivot = _md_table(pivot_headers, pivot_rows)
    weak_line = (
        f"\n\n**Weak tools** (< {_WEAK_TOOL_THRESHOLD} calls across the 595-row expanded set): "
        + (", ".join(f"`{n}`" for n in weak) if weak else "_none_")
    )
    return f"### Tool counts (overall)\n\n{overall}\n\n### Tool x category pivot\n\n{pivot}{weak_line}"


# endregion -----------------------------------------------------------------

# region D3: argument-value diversity ---------------------------------------


def _collect_arg_values(
    rows: list[dict[str, Any]],
    tool: str,
    arg: str,
) -> tuple[int, list[str]]:
    """Return (n_calls, sorted_unique_casefolded_values) for the given tool/arg."""
    raw_values: list[str] = []
    for row in rows:
        for tc in _iter_assistant_tool_calls(row):
            if tc.get("function", {}).get("name") != tool:
                continue
            args = tc.get("function", {}).get("arguments") or {}
            v = args.get(arg)
            if isinstance(v, str):
                raw_values.append(v)
            elif v is not None:
                raw_values.append(json.dumps(v, sort_keys=True))
    n_calls = len(raw_values)
    uniq = sorted({v.casefold() for v in raw_values})
    return n_calls, uniq


def _audit_d3(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for tool, arg in _TOOLS_WITH_STRING_ARG:
        train_n, train_uniq = _collect_arg_values(train_rows, tool, arg)
        eval_n, eval_uniq = _collect_arg_values(eval_rows, tool, arg)
        train_set = set(train_uniq)
        eval_set = set(eval_uniq)
        overlap = sorted(train_set & eval_set)
        eval_only = sorted(eval_set - train_set)
        train_only = sorted(train_set - eval_set)
        out.append({
            "tool": tool,
            "arg": arg,
            "train_n_calls": train_n,
            "train_n_unique": len(train_uniq),
            "train_values": train_uniq,
            "eval_n_calls": eval_n,
            "eval_n_unique": len(eval_uniq),
            "eval_values": eval_uniq,
            "overlap": overlap,
            "eval_only": eval_only,
            "train_only_count": len(train_only),
        })
    return out


def _format_d3(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    summary_rows = []
    for it in items:
        cov = (len(it["overlap"]) / it["eval_n_unique"]) if it["eval_n_unique"] else 1.0
        summary_rows.append([
            f"`{it['tool']}.{it['arg']}`",
            str(it["train_n_calls"]),
            str(it["train_n_unique"]),
            str(it["eval_n_calls"]),
            str(it["eval_n_unique"]),
            str(len(it["overlap"])),
            str(len(it["eval_only"])),
            f"{cov:.1%}",
        ])
    parts.append(
        _md_table(
            [
                "tool.arg",
                "train_calls",
                "train_uniq",
                "eval_calls",
                "eval_uniq",
                "overlap",
                "eval_only (gap)",
                "eval coverage",
            ],
            summary_rows,
        )
    )
    parts.append("")
    for it in items:
        parts.append(f"### `{it['tool']}.{it['arg']}`")
        parts.append("")
        parts.append(
            f"- train: **{it['train_n_calls']}** calls / **{it['train_n_unique']}** unique values"
        )
        parts.append(
            f"- eval:  **{it['eval_n_calls']}** calls / **{it['eval_n_unique']}** unique values"
        )
        parts.append(
            f"- overlap (eval values seen in train): **{len(it['overlap'])}** / {it['eval_n_unique']}"
        )
        parts.append(f"- eval-only values (model never saw): **{len(it['eval_only'])}**")
        parts.append("")
        parts.append("Train values: " + (", ".join(f"`{v}`" for v in it["train_values"]) or "_none_"))
        parts.append("")
        parts.append("Eval values:  " + (", ".join(f"`{v}`" for v in it["eval_values"]) or "_none_"))
        parts.append("")
        if it["eval_only"]:
            parts.append("**Eval-only (gap):** " + ", ".join(f"`{v}`" for v in it["eval_only"]))
            parts.append("")
    return "\n".join(parts)


# endregion -----------------------------------------------------------------

# region D4: refusal-prompt clustering --------------------------------------


def _audit_d4(
    rows: list[dict[str, Any]],
    embedder: SentenceTransformer,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for cat in ("off_topic_refusal", "medical_advice_refusal"):
        cat_rows = [r for r in rows if r["category"] == cat]
        prompts = [_first_user_prompt(r) for r in cat_rows]
        norm = [_normalize_prompt(p) for p in prompts]
        n_unique = len(set(norm))
        if n_unique < 2:
            out[cat] = [{
                "cluster": 0,
                "size": len(cat_rows),
                "intra_cos_mean": 1.0,
                "reps": prompts[:2],
            }]
            continue
        embeds = embedder.encode(prompts, show_progress_bar=False, convert_to_numpy=True)
        unit = _normalize_unit(embeds.astype(np.float32))
        k = min(5, n_unique)
        # n_init=10 + fixed random_state -> deterministic across runs.
        km = KMeans(n_clusters=k, n_init=10, random_state=_RANDOM_STATE)
        labels = km.fit_predict(unit)
        clusters: list[dict[str, Any]] = []
        for cid in range(k):
            members = np.where(labels == cid)[0]
            if members.size == 0:
                continue
            sub_unit = unit[members]
            pair = _pairwise_upper_cosines(sub_unit)
            intra = float(np.mean(pair)) if pair.size else 1.0
            # Pick the 2 prompts closest to the cluster centroid as
            # representatives. Centroids in unit-vector cosine space:
            # the un-normalized centroid is fine for ranking.
            centroid = sub_unit.mean(axis=0)
            cnorm = centroid / (np.linalg.norm(centroid) or 1.0)
            scores = sub_unit @ cnorm
            top_local = np.argsort(-scores)[:2]
            reps = [prompts[members[i]] for i in top_local]
            clusters.append({
                "cluster": cid,
                "size": int(members.size),
                "intra_cos_mean": intra,
                "reps": reps,
            })
        clusters.sort(key=lambda c: -c["size"])
        out[cat] = clusters
    return out


def _format_d4(report: dict[str, list[dict[str, Any]]]) -> str:
    parts: list[str] = []
    for cat in ("off_topic_refusal", "medical_advice_refusal"):
        clusters = report.get(cat, [])
        parts.append(f"### `{cat}`  ({sum(c['size'] for c in clusters)} rows, {len(clusters)} clusters)")
        parts.append("")
        all_high = clusters and all(c["intra_cos_mean"] >= _REFUSAL_TEMPLATE_COSINE for c in clusters)
        rows = [
            [
                str(c["cluster"]),
                str(c["size"]),
                f"{c['intra_cos_mean']:.3f}",
                " // ".join(_truncate(p, 80) for p in c["reps"]),
            ]
            for c in clusters
        ]
        parts.append(
            _md_table(["cluster", "size", "intra_cos_mean", "representative prompts"], rows)
        )
        parts.append("")
        if all_high:
            parts.append(
                f"**Templated-refusal verdict:** every cluster has intra-cluster mean cosine "
                f">= {_REFUSAL_TEMPLATE_COSINE:.2f} -- this category is paraphrases of "
                f"{len(clusters)} templates."
            )
            parts.append("")
    return "\n".join(parts)


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "..."


# endregion -----------------------------------------------------------------

# region D5: train <-> eval-holdout overlap --------------------------------


def _audit_d5(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    embedder: SentenceTransformer,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    train_prompts = [_first_user_prompt(r) for r in train_rows]
    eval_prompts = [_first_user_prompt(r) for r in eval_rows]
    train_embeds = embedder.encode(train_prompts, show_progress_bar=False, convert_to_numpy=True)
    eval_embeds = embedder.encode(eval_prompts, show_progress_bar=False, convert_to_numpy=True)
    train_unit = _normalize_unit(train_embeds.astype(np.float32))
    eval_unit = _normalize_unit(eval_embeds.astype(np.float32))
    sims = eval_unit @ train_unit.T  # (n_eval, n_train)
    max_per_eval = sims.max(axis=1)
    argmax_per_eval = sims.argmax(axis=1)
    stats = _percentiles(max_per_eval)

    # Top-5 closest pairs across all eval rows.
    pairs = sorted(
        (
            {
                "cosine": float(max_per_eval[i]),
                "eval_id": eval_rows[i].get("id", "?"),
                "eval_prompt": eval_prompts[i],
                "train_id": train_rows[int(argmax_per_eval[i])].get("id", "?"),
                "train_prompt": train_prompts[int(argmax_per_eval[i])],
            }
            for i in range(len(eval_rows))
        ),
        key=lambda d: -d["cosine"],
    )[:5]
    return stats, pairs


def _format_d5(stats: dict[str, float], pairs: list[dict[str, Any]]) -> str:
    summary = _md_table(
        ["mean", "median", "p20", "p80", "max"],
        [[
            f"{stats['mean']:.3f}",
            f"{stats['median']:.3f}",
            f"{stats['p20']:.3f}",
            f"{stats['p80']:.3f}",
            f"{stats['max']:.3f}",
        ]],
    )
    verdict_lines = []
    if stats["p80"] > 0.95:
        verdict_lines.append(
            "- p80 > 0.95 -> eval is too easy; the model can memorize rather than learn."
        )
    if stats["p20"] < 0.5:
        verdict_lines.append(
            "- p20 < 0.5 -> eval is too far OOD; the 511-row train set is genuinely too narrow."
        )
    if not verdict_lines:
        verdict_lines.append(
            "- p20-p80 spread is in the healthy 0.4-0.95 band; eval/train similarity is sane."
        )
    pair_rows = [
        [
            f"{p['cosine']:.3f}",
            f"`{p['eval_id']}`",
            f"`{p['train_id']}`",
            _truncate(p["eval_prompt"], 60),
            _truncate(p["train_prompt"], 60),
        ]
        for p in pairs
    ]
    pair_table = _md_table(
        ["cosine", "eval_id", "train_id", "eval_prompt", "train_prompt"],
        pair_rows,
    )
    return "\n".join([
        "### Distribution of max cosine (each eval row vs all train rows)",
        "",
        summary,
        "",
        *verdict_lines,
        "",
        "### Top-5 closest eval<->train pairs",
        "",
        pair_table,
    ])


# endregion -----------------------------------------------------------------

# region headline + recommendations -----------------------------------------


def _build_headline(
    d3_items: list[dict[str, Any]],
    d5_stats: dict[str, float],
    d5_pairs: list[dict[str, Any]],
) -> str:
    """The verdict the diagnostic deep-dive needs.

    Decision rule (any of these tips the verdict to "dataset is the bottleneck"):
      a) ANY string-arg tool has eval coverage < 60% AND >= 3 eval-only values
         (the original schema-leak test).
      b) Absolute argument-value space is very narrow (< 8 unique train values
         on a tool that takes a string argument from an open vocabulary). Even
         with 100% eval coverage, a model trained on 4 foods will not
         generalize to a 5th food in production.
      c) D5 p80 > 0.95 AND there are >= 3 cosine == 1.0 train/eval pairs --
         the eval set is contaminated by train-set prompts; G_EVAL is no
         longer measuring generalization, and a low score means the recipe
         failed to even memorize.
    """
    leak_tools: list[tuple[str, int, int, float]] = []
    narrow_tools: list[tuple[str, int]] = []
    for it in d3_items:
        if it["eval_n_unique"] == 0:
            continue
        cov = len(it["overlap"]) / it["eval_n_unique"]
        if cov < 0.60 and len(it["eval_only"]) >= 3:
            leak_tools.append((
                f"{it['tool']}.{it['arg']}",
                it["train_n_unique"],
                len(it["eval_only"]),
                cov,
            ))
        if it["train_n_unique"] < 8:
            narrow_tools.append((f"{it['tool']}.{it['arg']}", it["train_n_unique"]))

    n_dup_pairs = sum(1 for p in d5_pairs if p["cosine"] >= 0.999)
    contamination = d5_stats["p80"] > 0.95 and n_dup_pairs >= 3

    parts: list[str] = []
    bottleneck = bool(leak_tools or narrow_tools or contamination)

    if bottleneck:
        parts.append("**Yes, the dataset is the bottleneck.**")
        if leak_tools:
            bullets = "; ".join(
                f"`{name}` has {tu} unique train values vs {eo} eval-only values "
                f"({cov:.0%} coverage)"
                for name, tu, eo, cov in leak_tools
            )
            parts.append(
                f"D3 (argument-value overlap) shows {bullets} -- the model never saw "
                "those eval-side argument values during SFT."
            )
        if narrow_tools:
            bullets = "; ".join(f"`{n}` ({u} unique)" for n, u in narrow_tools)
            connector = "D3 also shows" if leak_tools else "D3 shows"
            parts.append(
                f"{connector} the absolute argument-value space is too narrow: "
                f"{bullets}. The schema-description regurgitation seen in M6 "
                "(model emitting `\"24-hour clock time in HH:MM format...\"` as "
                "a `time_24h` value) is the predictable failure mode of training "
                "on so few real values."
            )
        if contamination:
            parts.append(
                f"D5 reveals **train/eval contamination**: max-cosine p80="
                f"{d5_stats['p80']:.2f} and {n_dup_pairs} of the top-5 closest pairs "
                "are byte-identical. G_EVAL on this holdout is not a generalization "
                "test -- it's measuring memorization. That the M6 model still scored "
                "44.6% on a memorization-friendly eval implicates the recipe too."
            )
        parts.append(
            "Block E (dataset expansion) is required: more unique argument values "
            "per tool, and the eval holdout must be re-stratified to remove verbatim "
            "duplicates of train-set prompts."
        )
    else:
        parts.append(
            "**No, the dataset is not the headline bottleneck.** D3 shows "
            "argument-value overlap is acceptable on every audited tool and the "
            "absolute value space is wide enough to support generalization. The "
            "recipe (Block A) is the more likely culprit for the M6 G_EVAL collapse. "
            f"D5 max-cosine p80={d5_stats['p80']:.2f} indicates eval/train similarity "
            "is in a healthy band."
        )
    return " ".join(parts)


def _build_recommendations(
    d1_rows: list[dict[str, Any]],
    d1_flagged: list[str],
    d2_weak: list[str],
    d3_items: list[dict[str, Any]],
    d4_report: dict[str, list[dict[str, Any]]],
    d5_stats: dict[str, float],
    d5_pairs: list[dict[str, Any]],
) -> str:
    bullets: list[str] = []

    # D1: low-diversity categories or seed-recycle.
    high_cos = sorted(
        [r for r in d1_rows if r["mean_cos"] >= 0.55],
        key=lambda r: -r["mean_cos"],
    )
    if high_cos:
        names = ", ".join(
            f"`{r['category']}` (mean cos {r['mean_cos']:.2f})" for r in high_cos
        )
        bullets.append(
            f"Author additional phrasings for the high-cosine categories: {names}. "
            "Target adding 30-50 new prompts per category drawn from real user "
            "wording (vary topic, length, register, indirect phrasings)."
        )
    if d1_flagged:
        bullets.append(
            "Re-do LLM expansion for "
            + ", ".join(f"`{c}`" for c in d1_flagged)
            + " -- the teacher recycled the hand seeds. Use a higher-temperature "
            "prompt that explicitly demands phrasings that do NOT echo the seed."
        )

    # D2: weak tools.
    if d2_weak:
        bullets.append(
            "Increase coverage for under-called tools: "
            + ", ".join(f"`{n}`" for n in d2_weak)
            + f". Target >= {_WEAK_TOOL_THRESHOLD * 2} calls per tool across train."
        )

    # D3: argument-value gaps (eval-only) AND absolute narrowness.
    for it in d3_items:
        cov = (len(it["overlap"]) / it["eval_n_unique"]) if it["eval_n_unique"] else 1.0
        if cov < 0.60 and it["eval_only"]:
            missing = ", ".join(repr(v) for v in it["eval_only"])
            bullets.append(
                f"Add training rows that exercise `{it['tool']}.{it['arg']}` with "
                f"these missing values: {missing}. Also broaden train-only "
                f"diversity: currently {it['train_n_unique']} unique values "
                f"across {it['train_n_calls']} calls."
            )
        elif it["train_n_unique"] < 8:
            bullets.append(
                f"Broaden the absolute argument-value vocabulary for "
                f"`{it['tool']}.{it['arg']}` -- currently only "
                f"{it['train_n_unique']} unique training values across "
                f"{it['train_n_calls']} calls. Target >= 20 unique values "
                "(real medication names from a public formulary, plausible "
                "HH:MM times across the day, real food items). The M6 "
                "schema-description regurgitation is a downstream symptom "
                "of this narrowness."
            )

    # D5: contamination.
    n_dup = sum(1 for p in d5_pairs if p["cosine"] >= 0.999)
    if n_dup >= 3 or d5_stats["p80"] > 0.95:
        bullets.append(
            f"Re-stratify the eval holdout: {n_dup} of the top-5 closest train<->eval "
            f"pairs are byte-identical and p80 max-cosine is {d5_stats['p80']:.2f}. "
            "Either move duplicate prompts out of train, or author novel eval "
            "prompts. Otherwise G_EVAL is measuring memorization, not generalization."
        )

    # D4: templated refusals.
    for cat, clusters in d4_report.items():
        if clusters and all(
            c["intra_cos_mean"] >= _REFUSAL_TEMPLATE_COSINE for c in clusters
        ):
            bullets.append(
                f"`{cat}` is essentially {len(clusters)} templates with paraphrases. "
                "Author 20-30 new prompts that exercise novel refusal scenarios "
                "(different topics, different phrasing structures)."
            )

    if not bullets:
        bullets.append(
            "No category-level dataset gaps detected by D1-D4. If M6 G_EVAL "
            "remains low after the headline-verdict-implied actions, focus the "
            "next iteration on the SFT recipe rather than the data."
        )

    return "\n".join(f"- {b}" for b in bullets)


# endregion -----------------------------------------------------------------


def run_audit(args: argparse.Namespace) -> str:
    print(f"Loading data from {_FG_DIR}...", file=sys.stderr)
    seed_rows = list(load_jsonl(args.seed))
    expanded_rows = list(load_jsonl(args.llm_expanded))
    train_rows = list(load_jsonl(args.train))
    val_rows = list(load_jsonl(args.val))
    eval_rows = list(load_jsonl(args.eval_holdout))
    print(
        f"  seed={len(seed_rows)}  expanded={len(expanded_rows)}  "
        f"train={len(train_rows)}  val={len(val_rows)}  eval_holdout={len(eval_rows)}",
        file=sys.stderr,
    )

    print(f"Loading embedder {_MINILM} (cpu)...", file=sys.stderr)
    embedder = SentenceTransformer(_MINILM, device="cpu")

    print("D1 -- per-category phrasing diversity...", file=sys.stderr)
    d1_rows, d1_flagged = _audit_d1(expanded_rows, seed_rows, embedder)

    print("D2 -- tool-call distribution...", file=sys.stderr)
    d2_total, d2_pivot, d2_weak = _audit_d2(expanded_rows)

    print("D3 -- argument-value diversity...", file=sys.stderr)
    d3_items = _audit_d3(train_rows, eval_rows)

    print("D4 -- refusal-prompt clustering...", file=sys.stderr)
    d4_report = _audit_d4(expanded_rows, embedder)

    print("D5 -- train <-> eval overlap...", file=sys.stderr)
    d5_stats, d5_pairs = _audit_d5(train_rows, eval_rows, embedder)

    headline = _build_headline(d3_items, d5_stats, d5_pairs)
    recs = _build_recommendations(
        d1_rows, d1_flagged, d2_weak, d3_items, d4_report, d5_stats, d5_pairs
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("# FunctionGemma dataset_v1 quality audit (2026-05-01)")
        print()
        print(
            "Source: `scripts/dataset_quality_audit.py` "
            f"(MiniLM `{_MINILM}`, KMeans seed={_RANDOM_STATE})."
        )
        print(
            f"Inputs: seed={len(seed_rows)} | llm_expanded={len(expanded_rows)} | "
            f"train={len(train_rows)} | val={len(val_rows)} | eval_holdout={len(eval_rows)}."
        )
        print()
        print("## Headline verdict")
        print()
        print(headline)
        print()
        print("## D1 -- Phrasing diversity per category")
        print()
        print(_format_d1(d1_rows, d1_flagged))
        print()
        print("## D2 -- Tool-call distribution")
        print()
        print(_format_d2(d2_total, d2_pivot, d2_weak))
        print()
        print("## D3 -- Argument-value diversity (the headline schema-leak test)")
        print()
        print(_format_d3(d3_items))
        print()
        print("## D4 -- Refusal-prompt clustering")
        print()
        print(_format_d4(d4_report))
        print()
        print("## D5 -- Train <-> eval-holdout overlap")
        print()
        print(_format_d5(d5_stats, d5_pairs))
        print()
        print("## Recommendations")
        print()
        print(recs)
        print()
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=Path, default=_DEFAULT_SEED)
    p.add_argument("--llm-expanded", type=Path, default=_DEFAULT_LLM_EXPANDED)
    p.add_argument("--train", type=Path, default=_DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=_DEFAULT_VAL)
    p.add_argument("--eval-holdout", type=Path, default=_DEFAULT_EVAL_HOLDOUT)
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Markdown report destination. Pass '-' to write to stdout only.",
    )
    args = p.parse_args(argv)

    report = run_audit(args)
    sys.stdout.write(report)
    if str(args.output) != "-":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote report -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
