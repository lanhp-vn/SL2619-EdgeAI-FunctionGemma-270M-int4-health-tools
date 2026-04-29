# 09 — Code Style: Python

> Governs Python 3 scripts, training helpers, evaluation harnesses, and library code in the gemma3-270M-finetune project. Baseline: **PEP 8** + **PEP 484** (type hints) + **PEP 621** (pyproject.toml packaging).

> **Scope**: host-side tools (WSL2 Ubuntu) and GPU server scripts (Ubuntu 22.04+). No embedded/Yocto target — all Python here runs on a standard Linux distribution.

> **Tooling baseline** (from industry research 2025–2026):
> - **`ruff`** replaces black + isort + flake8 in a single 10–100x faster Rust-based tool.
> - **`mypy --strict`** for static typing; PEP 484 type hints on every function signature.
> - **`pytest`** for tests; **`hypothesis`** for property-based tests where invariants exist.
> - **`uv`** for environment and dependency management (faster than pip + venv).
> - **`pydantic`** for parsed config data.

---

## 1. Core Principles

1. **Types from day one.** Every function annotated. `mypy --strict` is the gate, not a suggestion.
2. **Single Source of Truth for config defaults.** If the schema lives in `data/health_table.yaml` or `data/prompts.yaml`, the Pydantic model whose defaults must agree with the YAML — not the reverse.
3. **Scripts must degrade gracefully.** A script that talks to the GPU server or the SL2619 board must handle the remote being unreachable — log and exit nonzero, don't traceback.
4. **Procedural where possible; classes where state matters.** Don't over-architect a 50-line script.
5. **Fail fast on startup, degrade in the loop.** Missing file or unreachable remote → exit 1. Transient I/O glitch in loop → log a warning and retry.

## 2. Tooling

| Tool | Version | Role |
|---|---|---|
| `python` | 3.11+ (3.12 preferred) | Runtime |
| `uv` | latest | Env + dependency manager |
| `ruff` | latest | Lint + format (10–100x faster than black/isort/flake8) |
| `mypy` | latest | Static type checking in `--strict` mode |
| `pytest` | 8+ | Test runner |
| `hypothesis` | latest | Property-based testing |
| `pydantic` | 2+ | Config / schema validation |
| `pytest-ruff`, `pytest-mypy` | latest | Integrates lint + types into `pytest` runs |

### 2.1 Project layout

```
src/
├── gemma_tools/
│   ├── __init__.py
│   ├── prompt_composer.py    # prompt construction for SFT and inference
│   ├── health_table.py       # health-QA YAML schema and loader
│   ├── sft_dataset.py        # SFT JSONL builder
│   ├── sft_build.py          # SFT dataset build script
│   ├── bench_eval.py         # eval loop (WER, F1, ROUGE)
│   ├── bench_prompt.py       # bench prompt generator
│   ├── bench_remote.py       # bench driver targeting board llama-server
│   ├── h5_logits_equiv.py    # H5R logits-equivalence gate
│   └── chat_probe.py         # interactive chat probe
tests/
├── __init__.py
├── test_prompt_composer.py
├── test_health_table.py
└── test_sft_dataset.py
```

### 2.2 `pyproject.toml` canonical

```toml
[project]
name = "gemma3-tools"
version = "0.1.0"
description = "Fine-tuning, evaluation, and deployment tooling for Gemma 3 270M-IT"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "numpy>=1.26",
    "transformers>=4.40",
    "tokenizers>=0.19",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.4",
    "mypy>=1.10",
    "pytest>=8.0",
    "pytest-ruff>=0.3",
    "pytest-mypy>=0.10",
    "hypothesis>=6.100",
]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "PT", "SIM", "RUF"]
ignore = ["E501"]                  # ruff format handles line length

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
warn_return_any = true
no_implicit_optional = true
```

### 2.3 Running

```bash
uv sync --extra dev
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

## 3. Formatting (handled by `ruff format`)

- **Indentation**: 4 spaces.
- **Line length**: 100 characters (ruff default is 88; we widen for model-name strings).
- **Quotes**: double quotes everywhere (`ruff format` enforces).
- **Trailing commas**: required in multi-line collections (ruff adds).

## 4. Naming

PEP 8 with project-specific conventions:

| Kind | Convention | Example |
|---|---|---|
| Module | `snake_case` | `prompt_composer.py` |
| Package | `snake_case` | `gemma_tools/` |
| Function / method | `snake_case` | `build_chat_prompt()` |
| Class | `PascalCase` | `HealthEntry`, `FinetuneConfig` |
| Constant | `SCREAMING_SNAKE_CASE` | `MAX_NEW_TOKENS = 256` |
| Private (module-internal) | `_leading_underscore` | `_CACHED_TOKENIZER` |
| Dunder / magic | `__double_underscore__` | `__init__`, `__repr__` |
| Type variable | `T`, `T_co`, `T_contra`, `PascalCaseT` | `T = TypeVar("T")` |

## 5. Type Hints

### 5.1 Every function signature

```python
from collections.abc import Sequence
from pathlib import Path

def load_health_table(path: Path) -> dict[str, object]:
    ...

def build_chat_prompt(system: str, turns: Sequence[tuple[str, str]]) -> str:
    ...
```

### 5.2 Prefer modern syntax

- `list[int]`, not `List[int]` (Python 3.9+).
- `dict[str, int]`, not `Dict[str, int]`.
- `X | Y` for union types, not `Union[X, Y]` (Python 3.10+).
- `int | None` instead of `Optional[int]` for clarity.

### 5.3 Protocols over ABCs for structural typing

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class PromptFormatter(Protocol):
    def format(self, question: str, context: str) -> str: ...
    def decode(self, token_ids: list[int]) -> str: ...
```

## 6. Config Loading (Pydantic)

All runtime config uses Pydantic models. The model defaults must agree with the YAML shipped alongside it.

```python
from pydantic import BaseModel, Field

class FinetuneConfig(BaseModel):
    model_name: str = "google/gemma-3-270m-it"
    max_seq_length: int = Field(default=1024, ge=64, le=8192)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)

class BenchConfig(BaseModel):
    finetune: FinetuneConfig = FinetuneConfig()
```

Load with explicit failure:

```python
import yaml
from pathlib import Path
from pydantic import ValidationError

def load_config(path: Path) -> BenchConfig:
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}           # never yaml.load()
    try:
        return BenchConfig.model_validate(raw)
    except ValidationError as e:
        print(f"Config validation failed: {e}", file=sys.stderr)
        raise
```

**Rules**:

- Always `yaml.safe_load()`. `yaml.load()` can execute arbitrary Python.
- Defaults in the Pydantic model must match the reference YAML. Drift breaks developers who expect "works out of the box".
- **Never read config mid-run.** Load at startup, use the immutable Pydantic object thereafter.

## 7. Logging

Use `logging` module; one `logging.getLogger(__name__)` per module.

```python
import logging

log = logging.getLogger(__name__)

def do_something(x: int) -> None:
    log.info("doing something with %d", x)      # lazy formatting
    if x < 0:
        log.warning("negative x — using absolute value")
        x = abs(x)
```

**Rules**:

- `%`-style formatting with `log.info("... %d ...", x)` — not f-strings — so the formatting cost is skipped when the level is filtered.
- **No `print()` in library code.** Scripts that produce user output (e.g., `bench_eval.py` reporting results) may use `print()` at the top level; library modules don't.

## 8. Error Handling

- **Specific exceptions.** Never `except:` and never `except Exception:` unless you re-raise or log+exit.

  ```python
  # Good
  try:
      config = load_config(path)
  except (FileNotFoundError, yaml.YAMLError) as e:
      log.error("config load failed: %s", e)
      sys.exit(1)
  ```

- **Subprocess calls have a timeout**:

  ```python
  import subprocess
  res = subprocess.run(
      ["ssh", "nouslogic-server", "nvidia-smi"],
      capture_output=True,
      text=True,
      timeout=10,
      check=False,      # we handle the rc ourselves
  )
  if res.returncode != 0:
      log.warning("ssh probe failed rc=%d stderr=%s", res.returncode, res.stderr)
  ```

- **Resource cleanup via `with`**:

  ```python
  with path.open("rb") as fh:
      data = fh.read()
  ```

  Or `contextlib.ExitStack` for dynamic-count resources.

## 9. Performance & Memory

- **Precompile regex at module load**:

  ```python
  import re
  _SCORE_RE = re.compile(r"score\s*=\s*([0-9.]+)")

  def parse_score(line: str) -> float | None:
      m = _SCORE_RE.search(line)
      return float(m.group(1)) if m else None
  ```

- **`__slots__`** on hot-path dataclasses:

  ```python
  from dataclasses import dataclass

  @dataclass(slots=True, frozen=True)
  class BenchResult:
      question_id: str
      reference: str
      hypothesis: str
      wer: float
  ```

- **Generators over lists** for streaming transforms:

  ```python
  def passing(results: Iterable[BenchResult], threshold: float) -> Iterator[BenchResult]:
      for r in results:
          if r.wer <= threshold:
              yield r
  ```

- **NumPy-specific**:
  - Prefer contiguous arrays (`np.ascontiguousarray`) before passing to C extensions.
  - Avoid `np.append` in loops (O(n²)). Allocate up front.
  - Use `np.float32` (not default `float64`) for anything headed to onnxruntime.

- **`onnxruntime` CPU provider**: match thread count to core count:

  ```python
  import onnxruntime as ort
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = os.cpu_count() or 2
  opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
  ```

## 10. Testing (this section is an index — full rules in `11-testing-verification.md`)

- `pytest` for unit tests.
- **Table-driven tests** via `@pytest.mark.parametrize` with tuple + header comment:

  ```python
  # | input | expected | description                   |
  @pytest.mark.parametrize("inp,exp,desc", [
      ("valid",  True,  "accepts valid input"),
      ("",       False, "rejects empty string"),
      (None,     False, "rejects None"),
  ])
  def test_validate(inp: object, exp: bool, desc: str) -> None:
      assert validate(inp) == exp, desc
  ```

- **Every assertion includes a `desc` string** so test failures self-identify.
- **At least 2 cases per test function.** A single-case test is under-tested.
- **`hypothesis`** for invariant-driven tests (prompt round-trips, encode/decode symmetry).

## 11. Useful Patterns

### 11.1 Entry-point scripts

Every standalone tool follows this template:

```python
#!/usr/bin/env python3
"""Short description of what this tool does."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("data/bench_config.yaml"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # ... real work ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## 12. Forbidden

| Pattern | Why |
|---|---|
| `yaml.load()` | Arbitrary code execution. Use `safe_load`. |
| `eval()`, `exec()` with external input | Same. |
| Bare `except:` | Hides bugs; catches `KeyboardInterrupt`. |
| `pickle.load()` on untrusted data | Arbitrary code execution. |
| `subprocess.*` without `timeout=` | Hangs tools; breaks CI. |
| `os.system()` | Shell-injection risk; use `subprocess.run` with a list. |
| Mutable default arguments (`def f(x=[]):`) | Classic bug; use `x: list[int] | None = None`. |
| Deep relative imports (`from ....a import b`) | Fragile; use package-absolute. |
| Star imports (`from m import *`) outside interactive scratch | Pollutes namespace. |
| Logging in `%` + pre-formatted string (`log.info(f"x={x}")`) | Defeats level filtering. Use `log.info("x=%s", x)`. |

---

## 13. Checklist (paste into PR)

- [ ] `uv run ruff format .` idempotent
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src` passes in `--strict` mode
- [ ] `uv run pytest` passes; parametrize tuples have header comments
- [ ] Every function signature has type hints
- [ ] All YAML loaded via `yaml.safe_load`
- [ ] Every `subprocess` call has `timeout=`
- [ ] No `print()` in library modules; entry-point scripts use `logging.basicConfig()`
- [ ] Pydantic defaults match the reference YAML in `data/`
