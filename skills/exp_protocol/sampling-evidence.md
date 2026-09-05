# Sampling: preserve returned draws before parsing

Use this inside a matching already-locked sampling/evaluation command. CPU prompt
preparation may happen before lock; `record_vllm` performs model inference and may
not. The helper supports the pinned vLLM0.11.0 native offline `LLM.generate` path,
plain SamplingParams with explicit n/max_tokens and required single-token stops.
It does not implement or modify the official benchmark grader.

```python
from vllm import LLM, SamplingParams
from awm.exp_protocol.sampling import (
    prepare_prompts, resolve_stop_ids, record_vllm_from_factory,
    parse_recording, finite_float,
)

# These strings must already be rendered by your actual selected template.
prompts = prepare_prompts(rendered_strings, tokenizer, item_ids=training_row_ids,
                          bos_policy="single_at_start")  # only when your template requires this
stop_ids = resolve_stop_ids(tokenizer, [actual_template_stop_token])
params = SamplingParams(n=k, temperature=chosen_temperature, max_tokens=chosen_cap,
                        stop_token_ids=stop_ids, seed=chosen_seed)
# Inside the checked/locked sampling command, with a CPU-loaded tokenizer:
def engine_factory():
    return LLM(model=selected_checkpoint, **selected_engine_options)

capture = record_vllm_from_factory(
    engine_factory, prompts, params, new_recording_directory,
    tokenizer=tokenizer, card_path=locked_card_path,
    required_stop_tokens=[actual_template_stop_token],
)
# Only after raw output was flushed/fsynced and structurally checked:
parsed = parse_recording(new_recording_directory, your_parser)
```

`your_parser(text, input_metadata)` returns JSON-compatible developer evidence.
It must not run inference. A parser can use `finite_float` to reject NaN, infinity
or numeric overflow before conversions such as int. This is only a finite-number
conversion, not a claimed reproduction of a benchmark's answer grammar.

## Sampling output is a later training input

For a sample/filter/train route, use this stage pattern:

1. The sampling card names the sampler checkpoint and **existing input** rows.
   Its `setup.command` runs sampling and may parse the returned draws. The new
   recording and filtered data are outputs, not future `setup.data` inputs.
2. Read the actual exit/capture/parser evidence and persist the training data.
   A failed sampling attempt remains failed even if partial raw data survives.
3. Prepare the training card with the resulting existing data paths. Its lock
   pins their hashes; its parent checkpoint is the model you chose to train.
   The scientific choice to continue, restart or abandon remains yours.

`lock --override data_files_exist=...` cannot authorize missing live input
hashes. `sampling_ready(...)` exposes the factory's CPU readiness checks if a
separate diagnostic is useful: live card/lock/source/input integrity, native
adapter signature, actual prepared tokens/stops/parameters and fresh output path.
It does not mutate the card, load a model, prove future consumption or waive the
same checks at inference time. If inputs change after readiness, capture rejects
them again. Do not replace a rejected wrapper with direct inference to bypass it.

## Native call and bounded failure observation

The adapter uses vLLM0.11's native positional prompt input. Equivalent native
syntax, if your explicitly uninstrumented route needs it, is:

```python
from vllm import TokensPrompt
outputs = llm.generate(
    [TokensPrompt(prompt_token_ids=list(p.token_ids)) for p in prompts],
    params, use_tqdm=False,
)
```

`llm.generate(prompt_token_ids=...)` is not this API. The factory checks the
pinned source and binds the supported signature without constructing an engine;
it preserves your batch boundaries, temperature, seed and sample count.

Keep factory/capture in a dedicated command stage. For example, the **locked
script** can use an explicitly planned stage deadline with GNU timeout:

```bash
set -eu
# Set this from the card's stage budget, including startup and observation margin.
sample_stage_limit=1800s
timeout --signal=TERM --kill-after=10s "$sample_stage_limit" python sample_stage.py
# Reached only on successful exit; still inspect capture/parser status before use.
```

The value is an example, not a universal budget. `awm exp_protocol run` retains
that script's real exit result. A timeout/nonzero exit stops dependent training;
retain raw files and investigate the exact owned producer. Test the chosen
deadline/cleanup path in your environment before trusting it for long work.
This bounds the directly launched stage's wait, not descendants that deliberately
escape its process group. No tool here discovers or kills unrelated engines.
Factory errors before returning an engine cannot be cleaned up by this helper.
Optional `close_engine=your_cleanup` receives only the returned engine; cleanup
may itself fail or hang, so it is not a substitute for the outer stage bound.
Already-durable raw capture stays available if later validation/cleanup fails.

## What is checked and recorded

Preparation calls tokenizer.encode with `add_special_tokens=False`; it does not
wrap an already-rendered prompt a second time. `single_at_start` checks the actual
tokenizer BOS and rejects an immediate double BOS; `unconstrained` is available
when that is not the template contract. Required stops are resolved from the
actual vocabulary and encoding, not hard-coded Gemma IDs or an unknown-token
fallback. A multi-token string cannot be passed as a required token spelling;
additional SamplingParams.stop strings are passed unchanged and recorded, not
silently converted into token IDs or certified as a full stopping-policy audit.

Before native inference, the helper checks the live card/lock/source evidence,
reproduces each prompt's tokens with the engine tokenizer, and verifies explicit
stops in a deep copy of the actual SamplingParams. It preserves the caller's
chosen decoding; no implicit chunking, reseeding, recipe choice or extra probe.
The source/version and native entrypoint are checked. Best-of, custom logits
processors, streaming, non-native engines and model-specific extensions are not
silently certified by this initial adapter.

Each new recording directory contains `request.json` and `raw.jsonl`, followed
by `capture.json` only when capture validation succeeds. Failures use a separate
`capture-failure.json`; existing directories/files are never overwritten.
Missing parent directories are created under the explicitly supplied output
path; the recording directory itself must still be new.
Raw records preserve every returned draw's text/token IDs and native finish/stop
reason, plus returned prompt tokens and request/completion identity. Input source
IDs remain typed and are accompanied by ordinal; repeated source IDs are not
silently collapsed. Native offline vLLM returns input order; every returned token
sequence is checked against that ordered input, and actual request/n/index counts
must match. Identical prompts remain subject to that native order contract.
The record includes engine-call start/return observations, monotonic call duration
and returned token count. This is wall time for the native call, not measured
GPU compute time or proof of an isolated throughput effect.

All returned raw draws are written and fsynced **before** count/order/finish
validation or parser callbacks. A count mismatch, abort or parser error therefore
does not erase already returned evidence. If the native call dies before returning,
this wrapper cannot recover in-engine partial generations. An unsupported object
or disk failure can leave only a partial raw prefix; the failure record does not
call that prefix durable/complete. Plan chunk boundaries explicitly if partial
generation salvage is needed; the helper never changes batching for you.

Requested parameters and observed prompt/finish fields are not a full audit of
resolved engine configuration, weight identity, kernels or per-request RNG state.
An EOS stop can legitimately have `stop_reason: null`. Do not infer stopping from
a literal text suffix or treat every max-length finish as a proven stop-ID bug.

## Parsing and recovery

The parser verifies both request and raw hashes before use. Each pass gets a new
`parse-*.jsonl` plus a summary. One parser exception or non-finite result yields
`parser_error`, never an invented zero/correct value. Read parsed/error counts and
all_parsed; this utility computes no accuracy. Interrupted parsing leaves raw data
unchanged and a failed/incomplete pass; no completion summary is manufactured.
`all_parsed` means successful JSON-producing callbacks, not correct answers or
even non-null numeric values; inspect the callback's declared result semantics.
Parser name/module/source hash are recorded when available, with closure and
transitive dependencies explicitly unverified; no environment/closure secrets
are dumped. Raw and request identity are rechecked after the parser finishes.

Raw records from an invalid capture remain available for read-only diagnosis, but
`parse_recording` does not certify a comparison with broken identity/counts. Repair
the scientific interpretation explicitly rather than editing hashes or pretending
the requested limit was the completed count.

The helper owns no global engine/process cleanup. On failure inspect the engine
you actually created and its recorded parent/child relationship; use its supported
shutdown mechanism or exact owned handles. Never kill processes by a shared name,
Unix user or GPU-memory observation. Ending the scientist turn still ends its work;
use the execution-record guidance for actual producer/exit handling.

No helper grants permission to use benchmark test items as sampling/training input,
watch sets or failure examples. Follow rule7 and retain actual data provenance.
No captured/parsed output is an official score or PTB-validator-complete result.
