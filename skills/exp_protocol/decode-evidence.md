# Distinguish selected settings from the decoder that executes

A valid save and a byte-identical export do not establish the intended decoder.
In pinned vLLM0.11, `do_sample:false` is not a serving-mode switch: the native
model-default resolver selects fields such as `temperature`, `top_k`, `top_p`
and `min_p`. An actual API request may supply other settings. Do not infer
greedy solely from that boolean, nor infer the effective mode from a filename,
an agent's summary or a single repeated score. The scientist selects the mode;
this evidence tool neither chooses it nor edits any setting.

Use the CPU-only helper with the exact selected serving JSON and, when available,
the request object inside an existing evaluation log:

```python
from awm.exp_protocol.decode_evidence import (
    freeze_decode_evidence, verify_decode_evidence,
)

receipt = freeze_decode_evidence(
    selected_directory / "generation_config.json",
    new_evidence_directory,
    intent="greedy",  # your actual choice; also sampling, unspecified, or None
    request_path=existing_eval_log,
    request_pointer="/samples/0/events/3/call/request",  # locate the actual model event
)
verify_decode_evidence(new_evidence_directory)
```

The JSON pointer is an example; inspect the log structure instead of assuming
event3 is always a model call. If no request exists yet, omit both arguments:
that layer remains unknown. The helper retains exact selected JSON bytes and
hashes the request source; it copies only supported decode fields, not prompts
or headers. Keep the original log for later verification. Every new observation
uses a new directory. Source changes invalidate verification rather than silently
updating the frozen evidence. Link it alongside the card's artifact/eval evidence;
no required card schema is added.

Interpret the layers separately:

| Layer | What it establishes |
|---|---|
| Intent | Scientist's stated choice, not proof |
| Selected JSON | Exact frozen file fields; explicit temperature0 supports requested greedy, `do_sample` alone does not |
| Request | Supported fields actually present at the recorded request location; omitted values remain omitted |
| Native model defaults | Optional return from pinned `ModelConfig.get_diff_sampling_param`, including native overrides; not the merged API request |
| Native request object | Optional actual native `SamplingParams` fields; its connection to the captured request is not independently verified |
| Engine execution | Unknown from this CPU evidence; no loading, inference, determinism or score claim |

If you already have the native objects from the supported runtime, optional
`native_model_config=...` and `native_request_params=...` record them. The former
must be the native vLLM0.11 `ModelConfig` for the selected local directory in
`generation_config="auto"` mode; its installed resolver is called and fingerprinted.
The latter must be an actual vLLM0.11 `SamplingParams`, not a dict labelled as
resolved. Creating an arbitrary object does not prove an engine used it. This
helper never constructs either object. Its unknown execution boundary remains
explicit even when both observations are supplied.

Do not automatically rewrite temperature to resolve a mismatch. Inspect native
precedence and the actual selected route, decide whether to change settings,
then use a matching locked evaluation to measure that change. Such a test is a
scientific experiment, unlike this CPU evidence snapshot. A changed decoder is
also a changed evaluation protocol for comparisons. Preserve the prior artifact
and evidence rather than presenting the new score as a repeat of old settings.
