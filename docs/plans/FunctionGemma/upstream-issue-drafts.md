# Upstream issue drafts — `ggml-org/llama.cpp`

> **Status:** local drafts only. **Do NOT file these upstream from this repo
> or this account without explicit user instruction.** Pin them under the
> FunctionGemma plan as OQ-10; if/when the user wants to engage upstream,
> they can copy these verbatim into <https://github.com/ggml-org/llama.cpp/issues/new>.

Both bugs were diagnosed during M1.5 (`docs/plans/FunctionGemma/README.md`
§15.6) at submodule pin **`d775992`** (build label `b1-d775992`, release
tag `b8981`) on **2026-04-30**. They block any in-tree path that runs
`llama-cli --jinja` against a tool-calling chat template (FunctionGemma is
the load-bearing example here, but Llama 3.x function-calling and
Qwen-tool-calling templates are affected the same way).

The repo's workaround is to render the prompt host-side via
`transformers.AutoTokenizer.apply_chat_template(tools=…)` and feed the
result to `llama-cpp-python` (Path A) or `llama-cli -st --no-jinja -f` (Path
B). See `scripts/functiongemma_smoke.py` and the §15.4 / §15.6 notes.

---

## Issue 1 — `llama-cli --jinja` discards `tools` before chat-template rendering

**Title (suggested):** `llama-cli: --jinja chat template never receives \`tools\` (tools/cli/cli.cpp:210 hardcodes \`inputs.tools = {}\`)`

**Affected build:** `llama.cpp` HEAD `d77599234ea6e498775aeadbce665eece5bd98cd`
(release tag `b8981`, build label `b1-d775992`). Build configuration:
`cmake -B build && cmake --build build -j` on Ubuntu 24.04.4 LTS / WSL2,
gcc 13. No CUDA.

**Summary**

`llama-cli`'s `format_chat()` helper builds `common_chat_templates_inputs`
with `inputs.tools = {};` and `inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE`
hard-coded. It also drops `--chat-template-kwargs` on the floor: those are
parsed into `params.default_template_kwargs` but never propagated to
`inputs.chat_template_kwargs`. As a result, **any chat template that gates
tool-calling output behind a non-empty `tools` jinja variable produces a
plain assistant turn** when invoked via `llama-cli --jinja`. There is also no
CLI flag to pass tool declarations directly.

`llama-server`'s OpenAI-compat handler does not have this gap — `tools` flows
from the `/v1/chat/completions` request body through
`tools/server/server-common.cpp:1056` into the same
`common_chat_templates_apply` call.

**Code reference**

`tools/cli/cli.cpp` (line numbers at `d775992`):

```cpp
common_chat_templates_inputs inputs;
inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
inputs.tools                 = {}; // TODO          ← hardcoded empty (line 210)
inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
inputs.json_schema           = ""; // TODO
inputs.grammar               = ""; // TODO
inputs.use_jinja             = chat_params.use_jinja;
inputs.parallel_tool_calls   = caps["supports_parallel_tool_calls"];
```

**Minimal repro**

```bash
# 1. Fetch the FunctionGemma 270M GGUF (or any tool-aware GGUF).
hf download google/functiongemma-270m-it --local-dir ~/hf-cache/fg-270m
python convert_hf_to_gguf.py ~/hf-cache/fg-270m \
  --outfile ~/hf-cache/fg-270m/fg-bf16.gguf --outtype bf16
build/bin/llama-quantize ~/hf-cache/fg-270m/fg-bf16.gguf \
  ~/hf-cache/fg-270m/fg-q4_k_m.gguf Q4_K_M

# 2. Try to use the bundled jinja chat template with tools:
build/bin/llama-cli \
  -m ~/hf-cache/fg-270m/fg-q4_k_m.gguf \
  --jinja \
  -p "What is the temperature in London?" \
  -n 96 -t 8 --temp 0.0 --top-p 1.0 -st --no-warmup
```

There is no flag to pass tool definitions. Even if there were, `format_chat`
would discard them before the template runs.

**Expected behavior**

One of the following:

- A `--tools <path-to-json>` flag (or `--tools-json '...'`) that loads a list
  of OpenAI-compat tool dicts and forwards them via
  `inputs.tools = common_chat_tools_parse_oaicompat(...)`.
- At minimum, propagate `params.default_template_kwargs` into
  `inputs.chat_template_kwargs` so users can side-channel tools through
  `--chat-template-kwargs '{"tools": [...]}'` in templates that read
  arbitrary kwargs (Qwen-style).
- Update the `--jinja` documentation under `tools/cli/README.md` to call out
  that tool-calling templates require `llama-server` (or
  `llama-cpp-python` Python bindings) until the above lands.

**Actual behavior**

The chat template renders with `tools` undefined / empty, so a
FunctionGemma run that should emit
`<start_function_call>call:get_current_temperature{...}<end_function_call>`
emits a plain natural-language answer instead. Silent — no warning that
tools were dropped.

**Workaround used in this repo (`gemma3-270M-finetune`)**

Render the prompt host-side via the HF tokenizer's
`apply_chat_template(messages, tools=[...], add_generation_prompt=True,
tokenize=False)` and either (a) feed it programmatically to `llama-cpp-python`
or (b) write it to a file and run `llama-cli -f file.txt -st --no-jinja`.
Both paths are documented at
[`docs/plans/FunctionGemma/README.md` §15.4 / §15.6](README.md).

---

## Issue 2 — `llama-cli` prints `--no-conversation is not supported`, then enters the interactive REPL anyway

**Title (suggested):** `llama-cli: --no-conversation prints "not supported" but does not exit, falls through into REPL (tools/cli/cli.cpp:357-360)`

**Affected build:** same as Issue 1 (`d775992` / tag `b8981`).

**Summary**

When `--no-conversation` (alias `-no-cnv`) is passed, `llama-cli` prints two
error lines and **does not return**. Execution continues into the
interactive `while (true)` loop near `cli.cpp:469`. With stdin closed
(`</dev/null`), `console::readline()` returns false, `buffer` stays empty,
the loop's `if (buffer.empty()) continue;` jumps straight back to the top,
and `console::log("\n> ")` fires again — a tight CPU-bound loop that emits
`> ` characters until it's killed externally. In one of our M1.5 attempts
this filled disk with multi-GB log files before the wrapper noticed.

**Code reference**

`tools/cli/cli.cpp` (line numbers at `d775992`):

```cpp
// TODO: maybe support it later?
if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
    console::error("--no-conversation is not supported by llama-cli\n");
    console::error("please use llama-completion instead\n");
}                       // ← no return, no exit, no break

// struct that contains llama context and inference
cli_context ctx_cli(params);
```

**Minimal repro**

```bash
build/bin/llama-cli \
  -m any-gguf.gguf \
  --no-conversation \
  -p "hello" \
  -n 16 </dev/null
# Prints:
#   --no-conversation is not supported by llama-cli
#   please use llama-completion instead
# Then prints `\n> ` repeatedly forever (until SIGTERM).
```

**Expected behavior**

Either:

- Exit non-zero immediately after the two error lines (`return 1;` or
  `exit(EXIT_FAILURE);` after the second `console::error`), pointing the
  user at `llama-completion`. Matches the existing error message intent.
- OR honor the flag — i.e. run a single non-interactive completion and
  return — which is what the `-st` / `--single-turn` flag effectively does
  today. If the maintainers prefer the second route, deprecate `-no-cnv`
  with a one-line note in `--help`.

**Actual behavior**

Prints the error, then silently enters the interactive loop. With stdin
closed, this is a CPU-pinning infinite loop that produces unbounded output.

**Workaround used in this repo**

Use `-st` / `--single-turn` (parsed at `common/arg.cpp:1513-1521`, exit at
`tools/cli/cli.cpp:636-638`) for one-shot completions. Documented at
[`docs/plans/FunctionGemma/README.md` §15.6](README.md). Never use
`-no-cnv` / `--no-conversation` against this submodule pin.

---

*Local drafts authored 2026-04-30 alongside M2 of the FunctionGemma plan.
Owner: M2/M3 author. Submission policy: do not file upstream without user
approval; this file is the staging area, not the issue tracker.*
