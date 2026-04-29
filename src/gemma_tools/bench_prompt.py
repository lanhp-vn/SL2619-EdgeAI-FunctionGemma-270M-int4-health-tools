"""On-board bench harness for Phase D closed-world health-YAML QA.

Drives Gemma 3 270M-IT against the prompt suite in `data/prompts.yaml`,
emits JSONL rows to `/mnt/sdcard/bench/<date>_gemma3-sweep.jsonl` plus a
companion `.log` of raw stdout.

Runs under the single-process rule (see `model-compiler-runtime.md §11.2`)
— one long-lived Python process iterates every prompt; spawning per-query
corrupts the NPU context within ~3 cycles.

Pure-Python helpers (timing, JSONL row, /proc sampler, prompt loader) are
host-testable. The vendor-`Gemma3Static` integration lives behind an import
guard so host unit tests don't need `torq.runtime`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, get_args

import yaml

from gemma_tools.health_table import HealthTable, load_health_table
from gemma_tools.prompt_composer import compose_user_text

PromptClass = Literal[
    "calibration",
    "fact_lookup",
    "fact_absence",
    "domain_refusal",
    "summarization",
]

# region: regex-flag scorer (shared with bench_eval.py)
#
# Lifted out of bench_eval so BenchRow can carry self-contained pass/fail
# at write time. bench_eval re-imports these — there's no cycle because the
# direction is bench_eval → bench_prompt and never the reverse.

_PATTERN_FLAG_BITS: dict[str, int] = {
    "i": re.IGNORECASE,
    "s": re.DOTALL,
    "m": re.MULTILINE,
    "x": re.VERBOSE,
}


def compile_pattern_flags(flags_str: str) -> int:
    """Translate the fixture's flags string ("", "i", "is", "m", …) into an
    `re` flag bitmask. Unknown characters raise — a silent typo in
    prompts.yaml must not quietly loosen a scorer."""
    bits = 0
    for c in flags_str.lower():
        bit = _PATTERN_FLAG_BITS.get(c)
        if bit is None:
            raise ValueError(f"unknown regex flag {c!r} in {flags_str!r}")
        bits |= bit
    return bits


def score_response(pass_pattern: str, pattern_flags: str, response_text: str) -> bool:
    """Return True iff `pass_pattern` appears in `response_text` under the
    given flags. Compiled fresh per call — patterns are few and cheap."""
    return bool(re.search(pass_pattern, response_text, compile_pattern_flags(pattern_flags)))

# endregion

# region: timing

class Stopwatch:
    """`time.perf_counter_ns` context manager with an injectable clock.

    Injecting `clock_ns` (not `sleep`) is the testability pattern from
    `docs/conventions/11-testing-verification.md §3.3`: we never sleep in
    tests — we advance a fake clock instead.
    """

    __slots__ = ("_clock_ns", "_end", "_start")

    def __init__(self, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self._clock_ns = clock_ns
        self._start: int | None = None
        self._end: int | None = None

    def __enter__(self) -> Stopwatch:
        self._start = self._clock_ns()
        return self

    def __exit__(self, *_args: object) -> None:
        self._end = self._clock_ns()

    @property
    def elapsed_ms(self) -> float:
        if self._start is None or self._end is None:
            raise RuntimeError("Stopwatch has not been used as a context manager yet.")
        return (self._end - self._start) / 1e6


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """Per-prompt timing snapshot.

    Records BOTH the external stopwatch (ours, around `run_stream()`) and
    the vendor-reported TTFT (`Gemma3Static.time_to_first_token`). They
    should agree within noise; storing both makes future discrepancies
    auditable without a re-run.
    """

    wall_ms_load: float
    wall_ms_ttft_vendor: float
    wall_ms_ttft_external: float
    wall_ms_total: float
    tokens_generated: int

    @property
    def tokens_per_sec(self) -> float:
        """Decode throughput over the post-TTFT window; 0.0 if no tokens.

        Uses the external TTFT as the decode-start anchor — vendor TTFT
        excludes the first sampled token's decode step, which double-counts
        if used here.
        """
        decode_ms = self.wall_ms_total - self.wall_ms_ttft_external
        if self.tokens_generated == 0 or decode_ms <= 0:
            return 0.0
        return self.tokens_generated / decode_ms * 1000

# endregion

# region: bench row + JSONL emission


@dataclass(frozen=True, slots=True)
class BenchRow:
    """One JSONL record per prompt run — the on-disk bench truth.

    `timing` is nested; `asdict` flattens it under the `timing` key. Memory
    counters come from `/proc/meminfo` (CMA, KiB) and `/proc/self/status`
    (process VmRSS, MiB) sampled by the sidecar thread during this prompt's
    decode window. `error` is set for infrastructural failures — the
    vendor runner raising mid-decode — so partial sweeps are still
    analyzable.

    `pass_pattern` / `pattern_flags` / `passed_regex` make the JSONL
    self-contained: a downstream consumer (UI, dashboard, SFT-bench
    diff) can read pass/fail without re-joining against `prompts.yaml`.
    `bench_eval.py` still computes its own from the suite for the
    Markdown rollup so the two scorers double-check each other.
    """

    prompt_id: str
    prompt_class: str
    prompt_text: str
    response_text: str
    run_started_iso: str
    timing: TimingRecord
    peak_rss_mb: float
    cma_free_kb_before: int
    cma_free_kb_during: int
    cma_free_kb_after: int
    pass_pattern: str = ""
    pattern_flags: str = ""
    passed_regex: bool = False
    error: str | None = None

    def to_jsonl_dict(self) -> dict[str, object]:
        """Include `timing.tokens_per_sec` (a property, not a field) for the
        downstream scorer that never reconstructs the dataclass."""
        row = asdict(self)
        row["timing"]["tokens_per_sec"] = self.timing.tokens_per_sec
        return row


def write_row(path: Path, row: BenchRow) -> None:
    """Append a single row as one JSON line. Flushes immediately so a killed
    process still leaves analyzable data on disk."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False))
        f.write("\n")


def write_rows(path: Path, rows: Iterable[BenchRow]) -> None:
    """Batched variant — one `open()` for the whole iterable."""
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False))
            f.write("\n")

# endregion

# region: /proc memory sampling

_KB_LINE_SEP = ":"


def _read_proc_kb(path: Path, field: str) -> int:
    """Read `<field>:\\s*<kb>\\s*kB` from a /proc-style file; raise if absent.

    The /proc text format is stable across kernels we care about — no csv
    / json involved. Returns kibibytes as an int.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(field + _KB_LINE_SEP):
                return int(line.split(":", 1)[1].strip().split()[0])
    raise KeyError(f"{field} not present in {path}")


def read_cma_free_kb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    return _read_proc_kb(meminfo_path, "CmaFree")


def read_rss_mb(status_path: Path = Path("/proc/self/status")) -> float:
    """VmRSS in MiB (the /proc unit is KiB → 1/1024)."""
    return _read_proc_kb(status_path, "VmRSS") / 1024


@dataclass(slots=True)
class _SamplerState:
    """Running max/min tracked by the background thread. Mutated under
    the sampler's lock so main-thread property reads are coherent."""
    peak_rss_mb: float = 0.0
    min_cma_free_kb: int = 2**31 - 1
    samples_taken: int = 0


class MemorySampler:
    """Daemon-thread /proc sampler. Tracks peak RSS and min CmaFree across
    its own lifetime.

    Usage:
        with MemorySampler(interval_s=0.5, meminfo_path=..., status_path=...) as s:
            ... do work ...
        print(s.peak_rss_mb, s.min_cma_free_kb)

    The thread is a daemon so a killed main process leaves no stragglers.
    `sample_once()` is public for deterministic unit tests that avoid the
    thread lifecycle entirely.
    """

    __slots__ = (
        "_interval_s",
        "_lock",
        "_meminfo_path",
        "_state",
        "_status_path",
        "_stop_event",
        "_thread",
    )

    def __init__(
        self,
        interval_s: float,
        meminfo_path: Path = Path("/proc/meminfo"),
        status_path: Path = Path("/proc/self/status"),
    ) -> None:
        self._interval_s = interval_s
        self._meminfo_path = meminfo_path
        self._status_path = status_path
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state = _SamplerState()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        """Read both /proc files once and update running peak/min.

        CmaFree absence is tolerated — on hosts without CMA the sampler
        still tracks RSS so host unit tests remain meaningful.
        """
        rss = read_rss_mb(self._status_path)
        try:
            cma = read_cma_free_kb(self._meminfo_path)
        except KeyError:
            cma = None
        with self._lock:
            self._state.samples_taken += 1
            if rss > self._state.peak_rss_mb:
                self._state.peak_rss_mb = rss
            if cma is not None and cma < self._state.min_cma_free_kb:
                self._state.min_cma_free_kb = cma

    def _run(self) -> None:
        # `/proc/self/status` vanishes at process exit, and a partial read
        # can yield a malformed int; drop those samples rather than crash
        # the daemon.
        while not self._stop_event.is_set():
            with contextlib.suppress(FileNotFoundError, ValueError):
                self.sample_once()
            self._stop_event.wait(self._interval_s)

    def __enter__(self) -> MemorySampler:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="MemorySampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_s * 3, 1.0))

    @property
    def peak_rss_mb(self) -> float:
        with self._lock:
            return self._state.peak_rss_mb

    @property
    def min_cma_free_kb(self) -> int:
        with self._lock:
            return self._state.min_cma_free_kb

    @property
    def samples_taken(self) -> int:
        with self._lock:
            return self._state.samples_taken

# endregion

# region: prompt-suite loader

_REQUIRED_PROMPT_KEYS: frozenset[str] = frozenset(
    {"id", "class", "text", "pass_pattern", "pattern_flags"}
)
_VALID_CLASSES: frozenset[str] = frozenset(get_args(PromptClass))


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One prompt from data/prompts.yaml, schema-validated.

    `prompt_class` is named with a trailing suffix because `class` is a
    Python keyword; the YAML-side name is preserved as `class`.
    `pattern_flags` is the free-form string from the fixture ("" / "i" /
    "is"); the scorer (bench_eval.py) converts it to `re.IGNORECASE` etc.
    """

    id: str
    prompt_class: PromptClass
    text: str
    pass_pattern: str
    pattern_flags: str


def load_prompt_suite(path: Path) -> list[PromptSpec]:
    """Load + validate prompts.yaml. Raises ValueError on any schema drift."""
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "prompts" not in raw:
        raise ValueError(f"{path}: top-level must be a mapping with key 'prompts'")
    entries = raw["prompts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'prompts' must be a non-empty list")

    out: list[PromptSpec] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{i}]: each prompt must be a mapping, got {type(entry).__name__}")
        missing = _REQUIRED_PROMPT_KEYS - entry.keys()
        if missing:
            raise ValueError(f"{path}[{i}]: missing required keys: {sorted(missing)}")
        pid = entry["id"]
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"{path}[{i}]: 'id' must be a non-empty string")
        if pid in seen_ids:
            raise ValueError(f"{path}[{i}]: duplicate id {pid!r}")
        seen_ids.add(pid)
        cls = entry["class"]
        if cls not in _VALID_CLASSES:
            raise ValueError(
                f"{path}[{i}] id={pid!r}: class {cls!r} not in {sorted(_VALID_CLASSES)}"
            )
        text = entry["text"]
        if not isinstance(text, str):
            raise ValueError(f"{path}[{i}] id={pid!r}: 'text' must be a string")
        pattern = entry["pass_pattern"]
        if not isinstance(pattern, str):
            raise ValueError(f"{path}[{i}] id={pid!r}: 'pass_pattern' must be a string")
        flags = entry["pattern_flags"]
        if not isinstance(flags, str):
            raise ValueError(f"{path}[{i}] id={pid!r}: 'pattern_flags' must be a string")
        out.append(
            PromptSpec(
                id=pid,
                prompt_class=cls,
                text=text,
                pass_pattern=pattern,
                pattern_flags=flags,
            )
        )
    return out

# endregion

# region: vendor Gemma3Static shim (board-only)


class VendorImportError(ImportError):
    """Raised when `Gemma3Static` (from torq-examples) cannot be imported.

    Vendor code is the Synaptics `torq-examples` package; on-board, the
    vendor `setup_demos.py` installs a `.pth` file that puts the runner
    dirs on `sys.path`. On a dev host without `torq.runtime`, import fails —
    host unit tests exercise everything except the live NPU call.
    """


class _Gemma3Like(Protocol):
    """Structural typing for Gemma3Static — matches the vendor API surface
    we use (board) and the fake we inject in unit tests (host).

    We go through `run_stream()` (not `run()`) so the adapter can stamp an
    external TTFT at the first token — vendor `time_to_first_token`
    excludes the initial sample's decode step and is noisy by ~50 ms
    (see backlogs.md §1.19 W1/W2)."""

    def run_stream(
        self, user_input: str, max_tokens: int | None = None
    ) -> Iterator[str]: ...
    @property
    def time_to_first_token(self) -> float: ...
    @property
    def generated_tokens(self) -> int: ...


@contextlib.contextmanager
def _patched_default_sys_prompt(
    runner_module: ModuleType, override: str = ""
) -> Iterator[None]:
    """Temporarily set `runner.DEFAULT_SYS_PROMPT` so `Gemma3Static.__init__`
    doesn't prefill the vendor's generic persona.

    Our directive prompt (`slm-system-prompt.md §4`) IS the system-level
    instruction. The vendor default ("You are a helpful AI assistant named
    Gemma…") would inject ~24 tokens of contradictory persona into the
    warmed-up KV snapshot that every bench `run()` call rewinds to — a
    silent quality tax on every prompt. Setting the override to `""`
    reduces warmup to the chat-template boilerplate (~6 tokens) with no
    persona content. Other consumers of the module in the same process
    see no permanent mutation.
    """
    # getattr/setattr keeps mypy-strict happy — ModuleType attributes are
    # untyped, and a direct attribute access would need a blanket cast.
    original = getattr(runner_module, "DEFAULT_SYS_PROMPT")  # noqa: B009
    setattr(runner_module, "DEFAULT_SYS_PROMPT", override)  # noqa: B010
    try:
        yield
    finally:
        setattr(runner_module, "DEFAULT_SYS_PROMPT", original)  # noqa: B010


class BenchAdapter(Protocol):
    """Structural protocol every bench adapter satisfies.

    The main loop holds an adapter through this protocol so the same
    `_run_one_prompt` works for the vendor `Gemma3BenchAdapter`
    (long-lived NPU runner) and the subprocess-based
    `LlamaCompletionBenchAdapter` (CPU GGUF, one process per prompt).
    Adding a future adapter (e.g. server-mode llama.cpp) means
    implementing `run` and registering it in `_ADAPTER_FACTORIES`.
    """

    def run(self, user_text: str) -> AdapterRunResult: ...


@dataclass(frozen=True, slots=True)
class AdapterRunResult:
    """Per-turn outputs of an adapter's `run()`.

    `wall_ms_ttft_external` is stamped at the first `run_stream` yield
    (ours); `wall_ms_ttft_vendor` comes from `time_to_first_token`
    post-decode (vendor). Both are kept for audit per plan §6.1.

    `wall_ms_load` is **per-call** load cost — meaningful only when the
    adapter cold-loads the model on every prompt (subprocess-based
    `LlamaCompletionBenchAdapter`). Long-lived adapters that load once
    at construction (vendor `Gemma3BenchAdapter`) leave it at 0.0 and
    the main loop falls back to the sweep-level value captured around
    `adapter_factory()` so the JSONL row still attributes the cost.
    """

    text: str
    wall_ms_ttft_vendor: float
    wall_ms_ttft_external: float
    tokens_generated: int
    wall_ms_load: float = 0.0


class Gemma3BenchAdapter:
    """Thin wrapper around a `_Gemma3Like` instance that exposes exactly
    the one method the bench needs.

    Holds `max_gen_tokens` (default 128 per plan §6.1) so the main loop
    doesn't thread that literal through every call site. Also owns the
    clock injection for deterministic unit tests — on the board, the
    default `time.perf_counter_ns` is correct.
    """

    __slots__ = ("_clock_ns", "_impl", "max_gen_tokens")

    def __init__(
        self,
        impl: _Gemma3Like,
        max_gen_tokens: int = 128,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._impl = impl
        self.max_gen_tokens = max_gen_tokens
        self._clock_ns = clock_ns

    def run(self, user_text: str) -> AdapterRunResult:
        """Iterate `run_stream()`, capturing external TTFT at the first
        yield. Vendor TTFT read after decode completes."""
        chunks: list[str] = []
        external_ttft_ns: int | None = None
        start_ns = self._clock_ns()
        for chunk in self._impl.run_stream(user_text, self.max_gen_tokens):
            if external_ttft_ns is None:
                external_ttft_ns = self._clock_ns() - start_ns
            chunks.append(chunk)
        # Zero-token runs (the model refused before emitting anything)
        # leave external TTFT unset; report 0.0 and let the scorer flag
        # it via tokens_generated==0.
        ttft_external_ms = (external_ttft_ns / 1e6) if external_ttft_ns is not None else 0.0
        return AdapterRunResult(
            text="".join(chunks),
            wall_ms_ttft_vendor=float(self._impl.time_to_first_token),
            wall_ms_ttft_external=ttft_external_ms,
            tokens_generated=int(self._impl.generated_tokens),
        )


def _import_vendor_runner(torq_examples_root: Path | None = None) -> ModuleType:
    """Resolve the vendor `runner` module (contains `Gemma3Static`).

    If `torq_examples_root` is given, prepends `<root>/gemma3/src` and
    `<root>/utils` to `sys.path` first — lets the user point at an
    unpacked tarball on `/mnt/sdcard` rather than requiring the vendor's
    `.pth` to be active.
    """
    if torq_examples_root is not None:
        for subdir in ("gemma3/src", "utils"):
            p = torq_examples_root / subdir
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))
    try:
        return importlib.import_module("runner")
    except ImportError as e:
        raise VendorImportError(
            f"Cannot import vendor `runner` module: {e}. "
            f"Ensure Synaptics torq-examples is installed (see vendor "
            f"README §Install) OR pass torq_examples_root=<path>. "
            f"Host unit tests do not exercise "
            f"this path — the board is the only environment with torq.runtime."
        ) from e


def create_gemma3_bench_adapter(
    model_path: Path,
    max_gen_tokens: int = 128,
    torq_examples_root: Path | None = None,
) -> Gemma3BenchAdapter:
    """Board-side factory: import vendor runner, patch sys prompt, construct
    `Gemma3Static`, return the thin adapter.

    Raises `VendorImportError` on any import failure (expected on dev hosts).
    """
    runner_module = _import_vendor_runner(torq_examples_root)
    gemma_cls = runner_module.Gemma3Static
    with _patched_default_sys_prompt(runner_module, ""):
        # `Gemma3Static.__init__` reads DEFAULT_SYS_PROMPT during `_warmup`;
        # the patch must wrap the constructor call itself, not just setup.
        impl = gemma_cls(model_path=str(model_path), instruct_model=True)
    return Gemma3BenchAdapter(impl, max_gen_tokens=max_gen_tokens)

# endregion

# region: llama-completion subprocess shim (A55 CPU GGUF path)
#
# `llama-completion -no-cnv -f promptfile` is the headless one-shot
# (the interactive `llama-cli` rejects `-no-cnv` since b8925 — see
# docs/get-started/gemma-on-a55-get-started.md §4.5). One process per
# prompt: ~3.8 s mmap per call, but it's the only path until we ship
# a long-lived llama-server fork.

_GEMMA_USER_TURN_OPEN = "<start_of_turn>user\n"
_GEMMA_MODEL_TURN_OPEN = "<start_of_turn>model\n"
_GEMMA_TURN_CLOSE = "<end_of_turn>"
_LLAMA_END_OF_TEXT = "[end of text]"

# Accept either the older `llama_perf_context_print:` prefix (upstream prior
# to b8925) or the newer `common_perf_print:` prefix (b8925 onward — what
# the on-board `0adede8` binary actually emits). Same fields, same column
# order; upstream just renamed the print site.
#   common_perf_print:        load time =    3252.55 ms
#   common_perf_print: prompt eval time =     859.74 ms /    82 tokens (   10.48 ms per token,    95.38 tokens per second)
#   common_perf_print:        eval time =    1355.04 ms /    21 runs   (   64.53 ms per token,    15.50 tokens per second)
#   common_perf_print:       total time =    2276.43 ms /   103 tokens
# `unaccounted` and `graphs reused` lines are accepted-but-ignored — they
# don't carry a `time = N ms` shape.
_PERF_FIELD_RE = re.compile(
    r"(?:llama_perf_context_print|common_perf_print):"
    r"\s+(?P<label>[a-z _]+?)\s*time\s*=\s*"
    r"(?P<ms>[0-9.]+)\s*ms"
    r"(?:\s*/\s*(?P<count>[0-9]+)\s*(?:tokens|runs))?"
    r"(?:[^(\n]*\(\s*[0-9.]+\s*ms per token,\s*(?P<tps>[0-9.]+)\s*tokens per second\s*\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LlamaPerfReport:
    """Parsed `common_perf_print` block from llama-completion output.

    `n_decode_tokens` corresponds to `eval time = ... / N runs` — that
    is the count of generated tokens (each "run" is one decode step
    after the prompt-eval phase). `n_prompt_tokens` is the input token
    count from `prompt eval time = ... / N tokens`.
    """

    wall_ms_load: float
    wall_ms_prompt_eval: float
    n_prompt_tokens: int
    prompt_eval_tps: float
    wall_ms_decode: float
    n_decode_tokens: int
    decode_tps: float
    wall_ms_total: float


class LlamaCompletionError(RuntimeError):
    """Raised when `llama-completion` exits non-zero, times out, or its
    perf footer is unparseable. The main loop catches it like any other
    `Exception` and records the prompt as an error row, so a single bad
    prompt never aborts an N-prompt sweep."""


SubprocessRunner = Callable[[list[str], float | None], "subprocess.CompletedProcess[str]"]


def _default_subprocess_runner(
    argv: list[str], timeout_s: float | None
) -> subprocess.CompletedProcess[str]:
    """Real `subprocess.run` wrapper, text mode, captured streams.

    Isolated so host smoke tests inject a fake — the cross-compiled
    aarch64 binary is not on the WSL host."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )


def wrap_gemma3_chat_template(user_text: str) -> str:
    """Wrap `user_text` in Gemma 3 turn markers for `llama-completion -f`.

    Equivalent to the user-turn portion of
    `prompt_composer.compose_prompt(candidate="gemma3", ...)` for the same
    inner body — re-derived here because the adapter only sees the
    already-composed `user_text` from the main loop's `compose_user_text()`
    call (HealthTable + date are not in adapter scope). A
    test cross-checks byte-equivalence so the two cannot drift silently.
    """
    return f"{_GEMMA_USER_TURN_OPEN}{user_text}{_GEMMA_TURN_CLOSE}\n{_GEMMA_MODEL_TURN_OPEN}"


_GEMMA_MODEL_ROLE_DETOK = "\nmodel\n"


def parse_completion_response(stdout: str) -> str:
    """Slice the model-generated text out of llama-completion stdout.

    `llama-completion -no-cnv -f promptfile` echoes the prompt then
    appends generated tokens then optionally `[end of text]`. The literal
    chat-template markers `<start_of_turn>` / `<end_of_turn>` are SPECIAL
    TOKENS — by default the binary detokenizes them to empty strings, so
    on the wire we see the bare role label (e.g. `\\nmodel\\n`) instead
    of `<start_of_turn>model\\n`. Try the explicit form first (so callers
    that pass `--special` keep working), then fall back to the bare role
    divider — `rfind` handles either form even if the model hallucinates
    a mid-stream role marker.

    If neither divider is found, return the stripped stdout — perf is on
    stderr so we never confuse the two streams.
    """
    body: str | None = None
    for divider in (_GEMMA_MODEL_TURN_OPEN, _GEMMA_MODEL_ROLE_DETOK):
        idx = stdout.rfind(divider)
        if idx >= 0:
            body = stdout[idx + len(divider):]
            break
    if body is None:
        body = stdout
    for terminator in (
        _LLAMA_END_OF_TEXT, _GEMMA_TURN_CLOSE,
        _GEMMA_USER_TURN_OPEN, "<start_of_turn>",
        "\nuser\n",  # detokenized counterpart of _GEMMA_USER_TURN_OPEN
    ):
        cut = body.find(terminator)
        if cut >= 0:
            body = body[:cut]
    return body.strip()


def parse_llama_perf(stream: str) -> LlamaPerfReport:
    """Extract load / prompt-eval / decode / total fields from the perf
    block. Accepts either `llama_perf_context_print:` (older upstream)
    or `common_perf_print:` (b8925 onward — current SL2619 board build).

    `stream` should be `stderr + "\\n" + stdout` (the parser scans both
    so we're robust to forks that mirror perf to stdout). Raises
    `ValueError` if the block is missing — caller wraps in
    `LlamaCompletionError` for the main loop's error-row path.
    """
    fields: dict[str, dict[str, float]] = {}
    for m in _PERF_FIELD_RE.finditer(stream):
        label = m.group("label").strip().replace(" ", "_")
        fields[label] = {
            "ms": float(m.group("ms")),
            "count": float(m.group("count") or 0),
            "tps": float(m.group("tps") or 0.0),
        }
    if "load" not in fields:
        raise ValueError(
            "no `(llama_perf_context_print|common_perf_print): load time = ...`"
            " line found in stream"
        )
    load = fields["load"]
    prompt = fields.get("prompt_eval", {"ms": 0.0, "count": 0.0, "tps": 0.0})
    decode = fields.get("eval", {"ms": 0.0, "count": 0.0, "tps": 0.0})
    total = fields.get("total", {"ms": 0.0, "count": 0.0, "tps": 0.0})
    return LlamaPerfReport(
        wall_ms_load=load["ms"],
        wall_ms_prompt_eval=prompt["ms"],
        n_prompt_tokens=int(prompt["count"]),
        prompt_eval_tps=prompt["tps"],
        wall_ms_decode=decode["ms"],
        n_decode_tokens=int(decode["count"]),
        decode_tps=decode["tps"],
        wall_ms_total=total["ms"],
    )


class LlamaCompletionBenchAdapter:
    """Subprocess-based bench adapter for the A55 CPU Q4_0 GGUF path.

    One `llama-completion -no-cnv -f promptfile` call per prompt:
    ~3.8 s mmap + ~2 s prompt eval + ~3-15 s decode at 5.87 tok/s
    (measured 2026-04-24 baseline). The adapter reports per-call
    `wall_ms_load` so the JSONL row attributes the cost honestly — vs
    vendor `Gemma3BenchAdapter` which loads once at construction and
    inherits sweep-level load.

    The `runner` callable is injectable so host smoke tests don't need
    the cross-compiled aarch64 binary on PATH; on the board the default
    `_default_subprocess_runner` shells out for real.
    """

    __slots__ = (
        "_clock_ns",
        "_runner",
        "binary_path",
        "model_path",
        "n_predict",
        "n_threads",
        "seed",
        "subprocess_timeout_s",
        "temp",
        "top_k",
    )

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        n_threads: int = 2,
        n_predict: int = 128,
        temp: float = 0.0,
        top_k: int = 1,
        seed: int = 42,
        subprocess_timeout_s: float = 120.0,
        runner: SubprocessRunner = _default_subprocess_runner,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.n_threads = n_threads
        self.n_predict = n_predict
        self.temp = temp
        self.top_k = top_k
        self.seed = seed
        self.subprocess_timeout_s = subprocess_timeout_s
        self._runner = runner
        self._clock_ns = clock_ns

    def build_command(self, prompt_file: Path) -> list[str]:
        """Argv for `llama-completion`. Argument order is pinned for
        stable test assertions; flag set matches the proven baseline in
        `gemma-on-a55-get-started.md §3.7` (-t 2, --temp 0.0, --top-k 1,
        -no-cnv) plus an explicit `--seed` for reproducibility."""
        return [
            str(self.binary_path),
            "-m", str(self.model_path),
            "-f", str(prompt_file),
            "-t", str(self.n_threads),
            "-n", str(self.n_predict),
            "--temp", str(self.temp),
            "--top-k", str(self.top_k),
            "--seed", str(self.seed),
            "-no-cnv",
        ]

    def run(self, user_text: str) -> AdapterRunResult:
        """Wrap → temp file → subprocess → parse → AdapterRunResult.

        `wall_ms_ttft_external` is the elapsed subprocess wall-clock.
        True TTFT (time to first decoded token) is not observable from a
        headless one-shot — we report `wall_ms_load + wall_ms_prompt_eval`
        from the perf footer as the "vendor" TTFT, which is the correct
        attribution since first-decoded-token lands the moment prompt
        eval finishes.
        """
        wrapped = wrap_gemma3_chat_template(user_text)
        # `delete=False` because the subprocess (a separate process group)
        # must be able to open the file by path — Python 3.11's tempfile
        # delete=True path holds an exclusive handle on Windows / WSL.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(wrapped)
            prompt_path = Path(f.name)
        try:
            argv = self.build_command(prompt_path)
            start_ns = self._clock_ns()
            try:
                proc = self._runner(argv, self.subprocess_timeout_s)
            except subprocess.TimeoutExpired as e:
                raise LlamaCompletionError(
                    f"llama-completion timed out after "
                    f"{self.subprocess_timeout_s}s "
                    f"(argv={shlex.join(argv)})"
                ) from e
            end_ns = self._clock_ns()
            if proc.returncode != 0:
                raise LlamaCompletionError(
                    f"llama-completion exited {proc.returncode} "
                    f"(stderr tail: {proc.stderr[-400:]!r})"
                )
            response = parse_completion_response(proc.stdout)
            try:
                # stderr is the canonical home of the perf block; we
                # concat stdout in case a fork mirrors it there.
                perf = parse_llama_perf(proc.stderr + "\n" + proc.stdout)
            except ValueError as e:
                raise LlamaCompletionError(
                    f"could not parse llama_perf footer: {e} "
                    f"(stderr tail: {proc.stderr[-400:]!r})"
                ) from e
            external_ms = (end_ns - start_ns) / 1e6
            return AdapterRunResult(
                text=response,
                wall_ms_ttft_vendor=perf.wall_ms_load + perf.wall_ms_prompt_eval,
                wall_ms_ttft_external=external_ms,
                tokens_generated=perf.n_decode_tokens,
                wall_ms_load=perf.wall_ms_load,
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                prompt_path.unlink()


def create_llama_completion_bench_adapter(
    binary_path: Path,
    model_path: Path,
    n_threads: int = 2,
    n_predict: int = 128,
    temp: float = 0.0,
    top_k: int = 1,
    seed: int = 42,
    subprocess_timeout_s: float = 120.0,
    runner: SubprocessRunner = _default_subprocess_runner,
) -> LlamaCompletionBenchAdapter:
    """Host-side factory: no environment dependencies. Returns an
    adapter ready to spawn `llama-completion` on first `.run()` call."""
    return LlamaCompletionBenchAdapter(
        binary_path=binary_path,
        model_path=model_path,
        n_threads=n_threads,
        n_predict=n_predict,
        temp=temp,
        top_k=top_k,
        seed=seed,
        subprocess_timeout_s=subprocess_timeout_s,
        runner=runner,
    )

# endregion

# region: main entry point

AdapterName = Literal["gemma3_vendor", "llama_completion"]


@dataclass(frozen=True, slots=True)
class _MainArgs:
    adapter: AdapterName
    prompts: Path
    health_table: Path
    output: Path
    max_gen_tokens: int
    sampler_interval_s: float
    now: date | None
    only_ids: tuple[str, ...]
    # Vendor-only:
    model_dir: Path | None
    torq_examples_root: Path | None
    # Llama-only:
    llama_binary: Path | None
    llama_model: Path | None
    n_threads: int
    temp: float
    top_k: int
    seed: int
    subprocess_timeout_s: float


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bench-prompt",
        description=(
            "SL2619 closed-world health-YAML QA bench harness. Drives "
            "either the vendor NPU runner (Gemma3Static) or the A55 "
            "CPU llama.cpp `llama-completion` headless one-shot, "
            "depending on `--adapter`. One process per sweep for "
            "vendor (single-process rule, "
            "model-compiler-runtime.md §11.2); one process per "
            "prompt for llama_completion (per-call mmap is the price "
            "of CPU GGUF on Yocto)."
        ),
    )
    p.add_argument("--adapter", choices=("gemma3_vendor", "llama_completion"),
                   default="gemma3_vendor",
                   help="Inference path. gemma3_vendor uses the on-board "
                        "torq runner (NPU VMFB); llama_completion shells "
                        "out per prompt to llama-completion (A55 CPU GGUF).")
    p.add_argument("--prompts", type=Path, required=True,
                   help="Path to prompts.yaml (see data/prompts.yaml).")
    p.add_argument("--health-table", type=Path, required=True,
                   help="Path to health_table_v1.yaml.")
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL output path. Appends; parent dirs are created.")
    p.add_argument("--max-gen-tokens", type=int, default=128,
                   help="Hard cap on decode-token count per prompt (default 128).")
    p.add_argument("--sampler-interval-s", type=float, default=1.0,
                   help="Interval between /proc memory samples in seconds (default 1.0).")
    p.add_argument("--now", type=date.fromisoformat, default=None,
                   help="Override the ISO date injected into the prompt (deterministic tests).")
    p.add_argument("--ids", default="",
                   help="Comma-separated prompt ids to run; empty = all.")
    # Vendor-only:
    p.add_argument("--model-dir", type=Path, default=None,
                   help="(adapter=gemma3_vendor) Directory with model.vmfb + tokenizer.")
    p.add_argument("--torq-examples-root", type=Path, default=None,
                   help="(adapter=gemma3_vendor) Path to torq-examples root if vendor .pth is not active.")
    # Llama-only:
    p.add_argument("--llama-binary", type=Path, default=None,
                   help="(adapter=llama_completion) Path to llama-completion binary.")
    p.add_argument("--llama-model", type=Path, default=None,
                   help="(adapter=llama_completion) Path to .gguf model.")
    p.add_argument("--n-threads", type=int, default=2,
                   help="(adapter=llama_completion) Threads for llama-completion (default 2 — CPU possible 0-1, IL-2).")
    p.add_argument("--temp", type=float, default=0.0,
                   help="(adapter=llama_completion) Sampling temperature (default 0.0 = deterministic).")
    p.add_argument("--top-k", type=int, default=1,
                   help="(adapter=llama_completion) Top-K (default 1 = greedy).")
    p.add_argument("--seed", type=int, default=42,
                   help="(adapter=llama_completion) RNG seed (default 42).")
    p.add_argument("--subprocess-timeout-s", type=float, default=120.0,
                   help="(adapter=llama_completion) Hard timeout per llama-completion call.")
    return p


def _parse_args(argv: list[str] | None) -> _MainArgs:
    ns = _build_arg_parser().parse_args(argv)
    ids = tuple(s.strip() for s in ns.ids.split(",") if s.strip())
    if ns.adapter == "gemma3_vendor" and ns.model_dir is None:
        raise SystemExit("--model-dir is required when --adapter=gemma3_vendor")
    if ns.adapter == "llama_completion" and (
        ns.llama_binary is None or ns.llama_model is None
    ):
        raise SystemExit(
            "--llama-binary and --llama-model are required when --adapter=llama_completion"
        )
    return _MainArgs(
        adapter=ns.adapter,
        prompts=ns.prompts,
        health_table=ns.health_table,
        output=ns.output,
        max_gen_tokens=ns.max_gen_tokens,
        sampler_interval_s=ns.sampler_interval_s,
        now=ns.now,
        only_ids=ids,
        model_dir=ns.model_dir,
        torq_examples_root=ns.torq_examples_root,
        llama_binary=ns.llama_binary,
        llama_model=ns.llama_model,
        n_threads=ns.n_threads,
        temp=ns.temp,
        top_k=ns.top_k,
        seed=ns.seed,
        subprocess_timeout_s=ns.subprocess_timeout_s,
    )


def default_adapter_factory(args: _MainArgs) -> BenchAdapter:
    """Pick the concrete adapter from `args.adapter` and call its
    factory with the right kwargs. Tests override this entirely (no
    inheritance from kwargs) so they can swap in a fully-stubbed
    BenchAdapter without faking subprocess or the vendor module."""
    if args.adapter == "gemma3_vendor":
        assert args.model_dir is not None  # CLI-validated above
        return create_gemma3_bench_adapter(
            model_path=args.model_dir / "model.vmfb",
            max_gen_tokens=args.max_gen_tokens,
            torq_examples_root=args.torq_examples_root,
        )
    if args.adapter == "llama_completion":
        assert args.llama_binary is not None
        assert args.llama_model is not None
        return create_llama_completion_bench_adapter(
            binary_path=args.llama_binary,
            model_path=args.llama_model,
            n_threads=args.n_threads,
            n_predict=args.max_gen_tokens,
            temp=args.temp,
            top_k=args.top_k,
            seed=args.seed,
            subprocess_timeout_s=args.subprocess_timeout_s,
        )
    raise ValueError(f"unknown adapter: {args.adapter!r}")


def _safe_read_cma_free_kb() -> int:
    """Best-effort CmaFree read — host has no CmaFree; return 0 as sentinel.
    The `bench_eval.py` scorer treats 0 as "not on a CMA-bearing kernel"
    rather than "zero free CMA"; context distinguishes."""
    try:
        return read_cma_free_kb()
    except (KeyError, FileNotFoundError, ValueError):
        return 0


def _run_one_prompt(
    spec: PromptSpec,
    health: HealthTable,
    now: date,
    adapter: BenchAdapter,
    wall_ms_load: float,
    sampler_interval_s: float,
) -> BenchRow:
    """Compose prompt, run through adapter, capture memory + timing.

    Isolated so the main loop stays flat and an injected stub-adapter in
    tests exercises the full path — including exception-to-error-row
    conversion."""
    user_text = compose_user_text(health, now, spec.text)
    cma_before = _safe_read_cma_free_kb()
    run_started = datetime.now().isoformat(timespec="seconds")

    error: str | None
    with (
        MemorySampler(interval_s=sampler_interval_s) as sampler,
        Stopwatch() as turn_sw,
    ):
        try:
            result = adapter.run(user_text)
            error = None
        except Exception as e:  # we capture any failure as a row — board runs must not abort
            result = AdapterRunResult(
                text="", wall_ms_ttft_vendor=0.0,
                wall_ms_ttft_external=0.0, tokens_generated=0,
            )
            error = f"{type(e).__name__}: {e}"

    cma_after = _safe_read_cma_free_kb()
    # When the sampler collected samples, use its min; otherwise fall back
    # to the pre-run snapshot so the column is never a stale sentinel.
    cma_during = (
        sampler.min_cma_free_kb if sampler.samples_taken > 0 else cma_before
    )

    # Adapter-reported per-call load wins; fall back to sweep-level for
    # long-lived (vendor) adapters that mmap once at construction.
    effective_load_ms = result.wall_ms_load if result.wall_ms_load > 0 else wall_ms_load
    passed = (
        score_response(spec.pass_pattern, spec.pattern_flags, result.text)
        if error is None and result.text
        else False
    )

    return BenchRow(
        prompt_id=spec.id,
        prompt_class=spec.prompt_class,
        prompt_text=spec.text,
        response_text=result.text,
        run_started_iso=run_started,
        timing=TimingRecord(
            wall_ms_load=effective_load_ms,
            wall_ms_ttft_vendor=result.wall_ms_ttft_vendor,
            wall_ms_ttft_external=result.wall_ms_ttft_external,
            wall_ms_total=turn_sw.elapsed_ms,
            tokens_generated=result.tokens_generated,
        ),
        peak_rss_mb=sampler.peak_rss_mb,
        cma_free_kb_before=cma_before,
        cma_free_kb_during=cma_during,
        cma_free_kb_after=cma_after,
        pass_pattern=spec.pass_pattern,
        pattern_flags=spec.pattern_flags,
        passed_regex=passed,
        error=error,
    )


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[_MainArgs], BenchAdapter] = default_adapter_factory,
) -> int:
    """Entry point. `adapter_factory` is injectable so host smoke tests
    can stub the runner-construction path without monkey-patching globals.

    Tests pass a factory of shape `Callable[[_MainArgs], BenchAdapter]`.
    The default `default_adapter_factory` dispatches on `args.adapter`."""
    args = _parse_args(argv)

    health = load_health_table(args.health_table)
    suite = load_prompt_suite(args.prompts)
    if args.only_ids:
        allow = set(args.only_ids)
        suite = [p for p in suite if p.id in allow]
        if not suite:
            print(f"no prompts matched --ids {args.only_ids}", file=sys.stderr)
            return 2
    run_date = args.now if args.now is not None else date.today()

    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] adapter={args.adapter} "
        f"loading...",
        file=sys.stderr, flush=True,
    )
    with Stopwatch() as load_sw:
        adapter = adapter_factory(args)
    wall_ms_load = load_sw.elapsed_ms
    print(f"  loaded in {wall_ms_load:.0f} ms", file=sys.stderr, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for spec in suite:
        row = _run_one_prompt(
            spec, health, run_date, adapter, wall_ms_load, args.sampler_interval_s,
        )
        write_row(args.output, row)
        verdict = "PASS" if row.passed_regex else "FAIL"
        print(
            f"  {spec.id:<4} {spec.prompt_class:<16} "
            f"tok={row.timing.tokens_generated:>3} "
            f"wall={row.timing.wall_ms_total:>6.0f} ms "
            f"{verdict} "
            f"{'error=' + row.error if row.error else ''}",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# endregion
