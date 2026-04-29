# 14 — Module Layering Discipline

> Every module in `src/gemma_tools/` belongs to exactly one abstraction layer. Dependencies only flow downward. This convention names what the directory layout already does — not a change to it.

---

## 1. The Layers

### Python tools (`src/gemma_tools/`)

| Layer | Role | Examples |
|---|---|---|
| `primitive` | Stateless helpers, pure math, stdlib-only I/O. No side effects, no network. | `health_table.py` (schema + validator), `bench_eval.py` (scoring functions: WER, F1, ROUGE) |
| `composite` | Groups primitives into a named mechanism. Owns one external format or protocol. | `prompt_composer.py` (assembles chat turns + system prompt), `sft_dataset.py` (reads YAML + builds JSONL) |
| `application` | Top-level script logic — reads config, calls composites, writes output. | `bench_remote.py`, `sft_build.py`, `chat_probe.py`, `h5_logits_equiv.py` |

Entry-point scripts in `scripts/` (`finetune.py`, `merge.py`, `t5_smoke.py`) are the `orchestrator` layer: they wire init order together and own `main()`. They call `application`-layer modules, not composites or primitives directly.

---

## 2. The Four Import Invariants

Every PR that touches `src/gemma_tools/` or `scripts/` must satisfy all four:

**1. No skip-layer imports.**
`application` imports `composite`, not `primitive` directly.
`composite` imports `primitive`.
`orchestrator` (`scripts/*.py`) imports `application`, not `composite` or `primitive` directly.

**2. No upward imports.**
Nothing below `orchestrator` imports a script's `main()` or any `scripts/*.py` module.
This makes every layer independently testable without running the full pipeline.

**3. No same-layer peer imports, except strictly orthogonal primitives.**
Two `composite` modules do not import each other. Two `application` modules do not import each other.
A `primitive` shared by several modules (e.g., a YAML loader, a scoring function) is the one allowed exception — and it must stay a true `primitive` (stateless, no side effects, no network).

**4. Config is data, not behavior.**
A file named `*_config.py` or `config.yaml` contains only declarations:
constants, enums, Pydantic model definitions, read-only lookup tables.
Zero function definitions that perform side effects. Zero I/O.
If a config file needs a computed derived value, that function belongs in the module it serves.

---

## 3. When to add a `*_config` sibling

A module gets a sibling config object only if it carries **tunable parameters** — numbers, thresholds, lookup tables that a calibration or integration step might legitimately change without touching logic. Examples:

- `bench_eval.py` → a `BenchConfig` Pydantic model (WER threshold, ROUGE variant, pass/fail gate)
- `sft_dataset.py` → a `DatasetConfig` Pydantic model (max token length, train/val split ratio)

Modules with no tunables (`health_table.py`, `prompt_composer.py`) do **not** get a config sibling. Absence is information.

---

## 4. No Network or GPU Contact at Import Time

Any module that owns an external resource (HTTP connection, model weights loaded via `transformers`, GPU session) must:

- Not open/start the resource at global scope, in a class constructor, or at module import time.
- Accept the resource via constructor injection or an explicit `init()` / `load()` call invoked from the `orchestrator`.

This ensures every layer below `orchestrator` is importable and testable with no GPU, no server, and no board attached — which is required by the R2 write→test→fix cadence. A test that imports `bench_remote.py` must not trigger an SSH connection on import.

```python
# Good — resource created on explicit init call
class RemoteBench:
    def __init__(self, ssh_client: SSHClient) -> None:
        self._client = ssh_client  # injected, not created here

# Bad — SSH connection opened at construction; untestable without a live server
class RemoteBench:
    def __init__(self, host: str) -> None:
        self._client = paramiko.SSHClient()
        self._client.connect(host)           # network call in __init__
```

---

## 5. Mapping to the Existing Directory Layout

```
src/gemma_tools/
├── __init__.py
├── health_table.py       # primitive — YAML schema loader + Pydantic model
├── bench_eval.py         # primitive — WER, F1, ROUGE scoring functions
├── prompt_composer.py    # composite — chat prompt assembly
├── sft_dataset.py        # composite — YAML → JSONL SFT builder
├── sft_build.py          # application — build pipeline entry point
├── bench_prompt.py       # application — prompt generation for bench runs
├── bench_remote.py       # application — bench driver (SSH + llama-server)
├── h5_logits_equiv.py    # application — H5R logits-equivalence gate
└── chat_probe.py         # application — interactive chat probe

scripts/
├── finetune.py           # orchestrator — LoRA SFT entry point
├── merge.py              # orchestrator — LoRA merge + export
└── t5_smoke.py           # orchestrator — base vs merged smoke comparison
```

If a new module does not fit into any existing layer, ask which layer it belongs to first, then decide where it lives. Do not create a new top-level directory unless it represents a genuinely new layer boundary.

---

## 6. Enforcement

- Reviewer-enforced at PR time. The import graph is visible in under a minute via:
  ```bash
  rg "^from gemma_tools|^import gemma_tools" src/gemma_tools/ --no-filename | sort | uniq
  ```
- No automated CI check yet. A future `tools/lint/check_layers.py` will automate the graph walk.
- Violations that are genuinely necessary must be documented inline with a comment naming the invariant being broken and why.

---

## 7. Rationale

The discipline keeps changes local: an edit inside one layer does not cascade upward into the orchestrator, and the mental model for reading any single file is "what does *this layer* know?" not "what does the whole pipeline do?"

It also makes the R2 cadence tractable: because GPU and network contact are forbidden outside `orchestrator` and `application`, every `primitive` and `composite` module can be exercised by a host unit test without a live server or board. This is not a nice-to-have — it is what makes the write→test→fix loop fast enough to catch mistakes before they compound.

---

*Added: 2026-04-29. Cross-references: `11-testing-verification.md` (testing pyramid), `AGENTS.md` §Discipline (R2 cadence).*
