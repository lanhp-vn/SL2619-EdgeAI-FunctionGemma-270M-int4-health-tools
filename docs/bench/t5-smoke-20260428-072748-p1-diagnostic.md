# T5 P1 phrasing diagnostic (sibling to `t5-smoke-20260428-072748.{jsonl,md}`)

Why this exists: the T5 smoke run showed P1 returning **1 new token** for both base and merged models, and `skip_special_tokens=True` decoded to `''`. Per advisor watch-out, empty-output could mean a generation-config bug rather than a behavioral signal — so we ran a READ-ONLY diagnostic on the server to check whether the merged model has actually learned the heart-rate fact.

## Training-data coverage of "heart rate" question phrasings

```
1 train rows mention heart-rate question phrasings:
   1  'tell me my blood pressure and heart rate'
```

The SFT corpus (1023 train rows) contains exactly **one** row that mentions a heart-rate question, and even that one combines it with blood-pressure rather than asking heart-rate alone. P1's literal phrasing — "what is my current heart rate?" — does not appear in training data.

## Variant probe on the merged model (greedy decode, max_new_tokens=64)

```
P1 original             :   1 tok | first_id=1      | raw='<eos>'    | clean=''
P1 without "current"    :   4 tok | first_id=236832 | raw='72.<eos>' | clean='72.'
P1 explicit YAML field  :   4 tok | first_id=236832 | raw='72.<eos>' | clean='72.'
P1 imperative           :   4 tok | first_id=236832 | raw='72.<eos>' | clean='72.'

eos_token_id = 1
bos_token_id = 2
id->token mapping (first IDs of interest): ['<eos>', '<end_of_turn>', '\n', '\n\n']
```

Conclusions:

1. **The merged model has learned the heart-rate fact.** Three of four phrasings extract `72.` cleanly from the YAML — `heart_rate_bpm: 72`.
2. **The literal P1 trigger word "current" pushes greedy first-token probability mass to `<eos>`.** This is a **sparse-coverage artifact**, not a generation-config bug, not a chat-template bug, not a quantization issue (no quantization — BF16 throughout).
3. **Base model emits `<eos>` first too** for the literal P1 phrasing — so this is not an SFT-introduced regression. The merged model is at least as good as base on this prompt; SFT did not make P1 worse.

## Implication for Phase 3

The Phase 3 bench rubric in `docs/plans/models-testing-plan.md` consumes `tools/data/prompts.yaml` verbatim. If P1's phrasing stays as-is, Phase 3 scoring will record P1 as a fail — not from Q4_0 quantization noise but from this same training-coverage gap. **Pre-Phase-3 decision** the user must adjudicate (out of T5 scope):

- (a) accept the artifact as a known phrasing-sensitivity in the rubric;
- (b) edit `tools/data/prompts.yaml` P1 to drop "current" or use a phrasing the merged model handles;
- (c) augment the SFT pool with heart-rate questions that include "current" / synonyms and re-run T1-T4.

(a) is least invasive but lowers the rubric ceiling. (b) cleanly aligns the bench prompt with what the model learned. (c) is the most expensive — re-runs T1-T4 — and only justified if other prompts also exhibit phrasing-sensitivity.

## Provenance

- Run timestamp: 2026-04-28 07:30 UTC+7
- Server: nouslogic-server (RTX 5080, sm_120, BF16, do_sample=False)
- Model: `~/sl2619-finetune/merged_v1/` (sha `57c56472…` — same as T4 closure record)
- Bundle: `~/sl2619-finetune/t5_smoke_bundle.json` (sha `ee93caa9…`, host-rendered with `prompt_composer.render_system_prompt(now=date(2026,4,25))` to match training-time prompt shape)
