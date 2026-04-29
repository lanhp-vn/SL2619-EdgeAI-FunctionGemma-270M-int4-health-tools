# 11 — Testing & Verification

> How we test this project. Pyramid order: **unit tests first, then server/board integration.** The unit-test discipline and idioms here are adapted from our in-house testing philosophy.

> **Scope**: applies to all code in the repo — Python library (`src/gemma_tools/`), scripts (`scripts/`), and shell helpers. Full integration tests require the GPU server or the SL2619 board and are gated behind a marker.

---

## 1. Core Principles

### 1.1 Testing is DRY — automate your manual tests

> *"You're testing it manually already. Just package your manual tests as automated tests so that you and team members don't have to do it again next time. It's time saving for YOU."*

Every time you run a script, inspect a log, or check model output manually, think: **can I encode this as an automated test?** If yes, do it now — while the mental model is fresh.

### 1.2 Tests are documentation

> *"Tests are internal documentation. They help reviewers understand the specs of the code, the usage of an API, and the expected outputs."*

A reviewer should be able to read your test file and learn **how your API is meant to be used**. Write tests for people to read, not to placate a coverage gate.

### 1.3 Tests must be meaningful

> *"Are you writing tests for people to read, or are you writing tests just because your code wouldn't pass code review without dummy tests?"*

A test that asserts `assert 1 == 1` is worse than no test — it signals coverage while providing none. If you can't articulate why a test case exists, delete it.

### 1.4 Unit tests run on the host — always

- **Python unit tests** run on WSL2 Ubuntu against the local venv. No GPU, no server, no board required.
- **Shell** uses `shellcheck` + `bats-core` on the host.

**Server/board access is the top of the pyramid, not the default.** A test that requires SSH is slow, fragile, and blocks CI. Invert the default: host first, remote only when necessary.

## 2. The Pyramid

```
                   +--------------------------------+
                   |  Server / board integration    |    <- slow, credential-gated
                   +--------------------------------+
              +------------------------------------------+
              |        Unit tests (on the host)           |    <- fast, deterministic, required
              +------------------------------------------+
```

| Layer | Runner | Required for merge? | Typical cycle time |
|---|---|---|---|
| Unit | `pytest` / `bats` | **Yes** | < 10 s |
| Server integration | `pytest -m server` driving SSH | On-demand | 1–5 min |
| Board integration | `pytest -m hardware` driving SSH | On-demand | 2–10 min |

## 3. Shared Test Idioms (all languages)

### 3.1 Table-driven tests with tuple + header comment

Keep the test-data structure flat. Never a dict per case — repeats keys.

**Python / pytest**:

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

### 3.2 Rules

- **At least 2 cases per test function.** One case is under-testing. If you can only think of one, you haven't thought hard enough.
- **Header comment** describes the columns once. Don't repeat.
- **Every assertion includes `desc`**. A failure that says `"rejects empty string"` is dramatically more useful than `"assertion failed on line 47"`.
- **> 20 cases**: package them as CSV and read at test start. Keeps test code compact.

### 3.3 Inject time — never sleep

State machines with timeouts must accept a `now()` function (or a mockable clock). Tests supply a fake clock that advances on demand — they **never** `sleep()`.

### 3.4 Mock external calls via interfaces

Any call that hits the network, a GPU, or the filesystem must be behind an injectable abstraction. Production passes the real implementation; tests pass a mock or fixture.

```python
from typing import Protocol

class ModelBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> str: ...

class FakeBackend:
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return "fake response"
```

## 4. Python Tests (pytest + hypothesis)

See `09-code-style-python.md` §10 for the style rules.

### 4.1 Setup

```bash
uv sync --extra dev
uv run pytest                              # all fast tests
uv run pytest -m 'not server and not hardware'  # explicit exclude
```

### 4.2 Property-based tests with Hypothesis

For invariants (encode/decode symmetry, ordering, idempotence):

```python
from hypothesis import given, strategies as st

@given(
    question=st.text(min_size=1, max_size=200),
    answer=st.text(min_size=1, max_size=500),
)
def test_prompt_roundtrip(question: str, answer: str) -> None:
    prompt = build_chat_prompt(question=question, answer=answer)
    assert len(prompt) > 0
    assert question in prompt or "Q:" in prompt
```

Hypothesis generates hundreds of cases and shrinks to a minimal failing case if it finds a bug.

### 4.3 Mark slow / remote tests

```python
@pytest.mark.server
def test_finetune_batch_on_server(ssh_server: SSHFixture) -> None:
    ...

@pytest.mark.hardware
def test_llama_server_responds_on_board(ssh_board: SSHFixture) -> None:
    ...

@pytest.mark.slow
def test_h5r_logits_equiv_full_corpus() -> None:
    ...
```

Run with:

```bash
uv run pytest -m 'not server and not hardware and not slow'   # default CI
uv run pytest -m 'server'                                      # on-server
uv run pytest -m 'hardware'                                    # on-board
```

### 4.4 What to test at the unit level

- **Prompt composition** — all template variants (nominal + edge: empty fields, None values).
- **YAML schema parsing** — valid inputs, invalid inputs, required field missing.
- **SFT dataset construction** — correct field mapping, correct JSONL encoding.
- **Scoring functions** — WER, F1, ROUGE math — each with a manual oracle case.
- **H5R logits-equivalence gate** — delta computation with a known-delta fixture.

### 4.5 What requires server/board markers

- Actual model forward pass (GPU required).
- SSH-based bench probe (`bench_remote.py`).
- On-board `llama-server` response test.

## 5. Shell Tests

### 5.1 Static analysis — mandatory

```bash
shellcheck scripts/*.sh
```

CI gate; zero warnings tolerated.

### 5.2 bats-core for testable logic

For shell scripts with non-trivial logic (parsing, argument handling), extract pure-shell functions into a sourceable library and test with `bats`:

```bash
# scripts/lib/parse-version.sh
parse_version() {
    local v="$1"
    echo "${v#v}" | awk -F. '{ printf "%d %d %d", $1, $2, $3 }'
}
```

```bash
# tests/test_parse_version.bats
load ../scripts/lib/parse-version.sh

@test "parses v1.2.3" {
    run parse_version "v1.2.3"
    [ "$output" = "1 2 3" ]
}

@test "parses without v prefix" {
    run parse_version "1.2.3"
    [ "$output" = "1 2 3" ]
}
```

Shell scripts that just wire `scp`/`ssh` calls don't need bats — `shellcheck` + manual smoke is enough.

## 6. Server / Board Integration Tests

These tests require a live SSH connection. Document the target environment in the test's top comment.

### 6.1 GPU server smoke (fine-tuning)

```bash
# From the host:
uv run pytest -m server tests/integration/test_finetune_smoke.py
# Pass: one training step completes, loss is finite, no CUDA OOM
```

### 6.2 On-board llama-server smoke

```bash
# From the host, with board reachable over SSH:
uv run pytest -m hardware tests/integration/test_board_inference.py
# Pass: /health returns 200, /completion returns a non-empty string in < 30 s
```

### 6.3 H5R logits-equivalence gate (on-board)

Run `h5_logits_equiv.py` with the base and fine-tuned logit files from the board bench run. Pass criterion: delta <= 1.0 pp on the calibration corpus. Full spec: `docs/plans/a55-gemma-h5-logits-equivalence.md`.

## 7. CI Matrix

| Stage | Environment | Runs | Required for merge |
|---|---|---|---|
| Lint | ubuntu-latest | `shellcheck`, `ruff check`, `mypy --strict` | Yes |
| Unit tests | ubuntu-latest + uv | `pytest -m 'not server and not hardware'` | Yes |
| Server integration | self-hosted with GPU | `pytest -m server` | Nightly |
| Board integration | self-hosted with board | `pytest -m hardware` | On-demand |

## 8. Test Procedure Template

For formal release-gate runs, use this template (place in `docs/tests/`):

```markdown
# Test Procedure: <Feature Name> — v<version>

## Overview
<what and why>

## Pre-test setup
- Environment: GPU server + SL2619 board connected
- Code: commit SHA <sha>

## Test Cases
### TC-01: <Name>
- **Procedure**:
  1. Step 1 with exact values
  2. Step 2
- **Acceptance**: <quantitative pass/fail, with units>
```

## 9. Anti-Patterns

| Anti-pattern | Why it's bad |
|---|---|
| Single-case test | Under-tested; delete or expand |
| `time.sleep(1)` in a unit test | Flaky, slow; inject time |
| `assert True, "TODO"` | Worse than no test |
| Testing private implementation | Brittle; couples test to internals |
| Mocking the system under test | Points to poor design — refactor |
| `setUp` that reaches the network or GPU | Not a unit test; gate with a marker |
| Silent test data (no `desc`) | Opaque failures |
| Identical tests copy-pasted per case | Use parametrize |

---

## 10. Checklist (paste into PR)

- [ ] Every new module has host-side unit tests
- [ ] Table-driven tests use tuple + header comment
- [ ] At least 2 cases per test function
- [ ] Every assertion has a `desc` string
- [ ] No `sleep()` in unit tests — time is injected
- [ ] Server/board-required tests tagged (`@pytest.mark.server` / `@pytest.mark.hardware`)
- [ ] `shellcheck` / `ruff` / `mypy --strict` all clean on changed files
- [ ] `pytest -m 'not server and not hardware'` green on host
