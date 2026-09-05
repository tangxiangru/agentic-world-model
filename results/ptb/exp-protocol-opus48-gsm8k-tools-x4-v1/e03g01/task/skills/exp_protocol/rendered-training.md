# Prepared training tokens and checked consumption

Use `RenderedTrainingBundle` for unpacked, completion-only causal-LM data. It
prepares the actual token artifact training reads; it is not a diagnostic JSON
claim about another pipeline. CPU-only preparation may precede lock. Do not load,
forward, train or evaluate a model in the renderer/preparation command; those
operations still need the matching prior checked/locked card.

## One materialization, then card/check/lock

Put the shared renderer in a source-backed module with no model-executing import
side effects. For data whose prompts already contain the chosen chat formatting:

```python
# preprocessing.py
from awm.exp_protocol.rendered_training import RenderedParts

def render(row, *, template, settings, rng):
    return RenderedParts(prefix=row["prompt"], target=row["completion"])
```

```python
# CPU-only prepare.py; tokenizer_dir and paths refer to local existing assets.
from pathlib import Path
from transformers import AutoTokenizer
from awm.exp_protocol.rendered_training import RenderedSettings, RenderedTrainingBundle
from preprocessing import render

task = Path.cwd()
tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, token=False)
settings = RenderedSettings(
    mode="separate_concat", prompt_mode="pre_rendered", max_seq_len=1280,
    stop_token=declared_stop_text, answer_marker=declared_answer_marker,
    tail_text="", pad_to_multiple_of=8, seed=0,
)
prepared = RenderedTrainingBundle.prepare(
    sources=[task / "data/train.jsonl"], render=render, tokenizer=tok,
    template_bytes=(task / "templates/chat.jinja").read_bytes(), settings=settings,
    source_files=[task / "train.py", task / "preprocessing.py", task / "prepare.py"],
    output=task / "data/exp-02-tokens",
)
print(prepared.declaration, prepared.data_entry, prepared.report)
```

Set `setup.data` to `[prepared.data_entry]`: this is the actual kept-token file
and retained count, not the raw builder input. Raw files/counts remain separately
bound inside the receipt. Set the optional v2 field below from
`prepared.declaration`, and make the card's stop token, answer marker and
max_seq_len match the prepared settings. No new required v2 field is introduced.

```yaml
setup:
  rendered_training:
    receipt: data/exp-02-tokens/receipt.json  # relative to session dir, or absolute
    sha256: <actual receipt byte SHA256>
```

Run normal `check` and `lock`. The existing data file is not the model output
directory: keep it separate from E5's optional fresh output namespace. Preparation
does not contain the later plan hash, avoiding a receipt/plan hash cycle.

## Preserve the actual formatting mode

- `separate_concat` separately tokenizes prefix and target, then concatenates.
  Already-rendered g01r03 prompts must not be wrapped again.
- `joint_prefix` tokenizes prefix and full text, then requires exact token-prefix
  alignment before masking; prefix drift is an explicit counted drop.
- `prompt_mode="pre_rendered"` needs no new messages. It records observed
  rendered-string hashes and actual arrays; template bytes are a bound reference,
  **not proof the template was applied or serving equivalence was established**.
- `prompt_mode="template_replay"` requires `RenderedParts.messages` for the
  supplied prefix, and, in joint mode, `full_messages` for `full`. The helper
  independently renders those messages with the exact supplied template snapshot
  and tokenizer; a stale cached formatter cannot merely relabel an old prefix.

For joint mode return `RenderedParts(prefix=prefix, full=full, messages=messages,
full_messages=messages_with_answer)` from the same pure formatter. For separate
mode return `target=target` instead. Do not substitute tokenization modes to pass
a check. The renderer receives a plain settings dictionary (use
`settings["stop_token"]`, not `settings.stop_token`) and a seeded `random.Random`;
include all local modules/config files affecting it in `source_files`. Hidden
closure state, custom tokenizer methods and unreported filtering are unsupported.
The file list is explicit provenance, not automatic proof of arbitrary imports.

## Inside the locked training command

```python
bundle = RenderedTrainingBundle.open_for_training(task / "memory/cards/exp-02.yaml")
training_args.remove_unused_columns = False  # preserve checked feature identities
trainer = SaveSafeTrainer(
    model=model, args=training_args, train_dataset=bundle.dataset,
    data_collator=bundle.collator(pad_to_multiple_of=8),
)
# Existing locked model execution goes here; E4 save checks remain separate.
bundle.flush_consumption()  # before closing the card; also flush after CPU inspections
```

Use the canonical `memory/cards/exp-NN.yaml` path matching `card_id`. At open,
the loader requires lock schema/timestamp, exact existing main-script and token
file hash bindings, and validates producer files, raw inputs, tokenizer/
template snapshots and every token record. It reads the same indexed JSONL file;
there is no second kept-token cache. The collator rejects changed/mixed/unidentified
features and pads with ignored labels and correct attention masks. Padding width
rounded to eight is not retained length or truncation. CPU tensor collation is
default; `return_tensors="python"` supports model-free inspection.

## Evidence and failures

`receipt.json` binds code/runtime, sources, settings, effective fast-tokenizer
backend (including added tokens, normalization/postprocessing and original active
padding/truncation), local tokenizer assets, template bytes and artifact hashes.
Tokenization explicitly disables automatic special-token addition, backend padding
and truncation. It does not mutate the supplied tokenizer's backend state.

`tokens.jsonl` contains kept input_ids/labels/target_start and source locators.
`decisions.jsonl` reconciles every raw occurrence, kept index, overlength/prefix
drop and declared global-prefix limit. Dropped rows retain their token evidence,
not just asserted lengths; kept arrays are not duplicated there. Malformed source
rows fail rather than disappearing. Pre/post lengths and rejected locators are
reported. High drop rates are visible; use an explicit `max_drop_fraction` cap
when the experiment needs one. Their scientific acceptability is not inferred.

Labels must be unshifted, prefix-masked and nonempty. Stop sequences come from
the effective tokenizer, not universal IDs. `tail_text` explicitly allows such
cases as supervised EOT plus newline. Only decoded supervised answer content is
checked for answer markers; prompt demonstrations and template tails are excluded.
Defaults retain the existing 95% stop and 2% bad-marker tolerances; only stricter
settings are accepted. Exact rates and policy are recorded. Permissive settings
cannot turn bad arrays into a superseding PASS; exceptions retain the recorded
raw-check override/unverified route. Structural corruption always fails.

Fresh complete valid evidence supersedes raw suffix/marker/chars-per-token
heuristics as SKIP, not a relabelled raw PASS. Stale/malformed claimed evidence
fails. No field means rendered coverage remains unverified and the old raw checks
and recorded-override route remain. Preflight never imports/runs the scientist's
renderer. Template replay is observed during preparation; preflight rechecks
bound token evidence, not independent template/serving semantic equivalence.
`RenderedTrainingBundle.verify(receipt_path)` rechecks all bindings/rows.
`prepared.report` describes its last verification, not continuous file monitoring.

Use `reuse=True` only to request verified reuse of the same output; changed
bindings refuse reuse. Invalid/partial preparations are retained for inspection;
choose a new path after a recipe change instead of overwriting old evidence.

Consumer records live under `memory/rendered-consumers/exp-NN/`, separately per
process. Binding checks/record writes happen at open, each observation kind's first
use, and explicit `flush_consumption()`, not continuously on every later batch.
Card/lock/receipt changes after those first uses are caught at the next explicit
flush. Per-row token/feature hashes are still checked on every read/collation.
Records distinguish verified loader binding, dataset access and collation;
counts at the last flush are lower bounds (workers can prefetch unused batches).
Preparation, access and collation all leave **actual model consumption unknown**.
Fork/spawn CPU workers are tested. Later arbitrary batch transformations, raw
script bypass, model/optimizer execution, contamination and serving equivalence
are not certified. Packed/multi-span/shifted labels, truncation and unknown/custom
tokenizer state need a future explicit adapter, not a fabricated PASS.

## CPU verification (construction, not a scientist model experiment)

Native support is tested with Transformers 4.57.3 and tokenizers 0.22.2. The
original repository environment has neither; native cases skip there explicitly.
Tests use synthetic real fast tokenizers and the existing local Gemma tokenizer,
without model weights, forwards, training, evaluation or downloads.

Run `pytest tests/test_exp_protocol_rendered_training.py` in that runtime. The
pinned no-network/no-GPU bwrap setup is documented in `save-safety.md`; select
this test file instead of the save tests and set `TOKENIZERS_PARALLELISM=false`.
For the volume measurement set `AWM_RENDERED_BENCHMARK_ROWS=10000` and select
`-k representative_cpu_cost`. It measures prepare (including full verification),
preflight, checked loader, first batch and actual token-file size. Recorded costs
are local CPU observations, not a promised full-corpus runtime or scientific gain.

Construction measurement (varied synthetic byte-level fast-tokenizer rows):
10,000 rows, 3,492,790 tokens, one 32,588,994-byte kept-token file. Prepare including
full verification: 15.96 s; preflight: 2.36 s; checked loader: 2.41 s; first 32-row
batch: 0.027 s. No second kept-token artifact was produced. These are local CPU
measurements, not performance extrapolations to the historical full corpus.
