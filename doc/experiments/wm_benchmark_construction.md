# World-model benchmark: construction log

This is the working record of how the benchmark defined in [wm_benchmark_spec.md](wm_benchmark_spec.md)
(worked example: [wm_benchmark_example.md](wm_benchmark_example.md)) is built from the private HF
dataset. Every step names its inputs, its code, its outputs and how to check them by hand. It is
appended to as construction proceeds; counts are updated in place and dated.

Status: **in construction** (started 2026-09-12).

## 1. Sources

All inputs come from one pinned revision of the private dataset
`JerrrrryL/awm-gsm8k-trajectories` (revision `07132f15e3c6cc6714ae84835b1896d734c5d54a`,
2026-09-12), mirrored to `data/hf-mirror/` (gitignored; `data/` is a symlink to
`/home/kalorona/awm-data` on the CPU box). Nothing is read from anywhere else.

| HF path | what it is | role |
|---|---|---|
| `cells/<cell>/solve_out_sanitized.txt` | the scientist's complete Claude Code session, stream-json, one event per line with a timestamp prefix; 160 cells | **X** (C, Δ) is read from here |
| `cells/<cell>/prompt.txt`, `output.log`, `cli_version.txt` | task prompt, PostTrainBench run log, harness version | task context; environment |
| `eval_matrix_1k/{results,locked_test/results}/<exp_id>.json` | 3,000 ten-run evaluations = 516 checkpoints × 5–8 generation policies, with server-resolved sampling parameters | **Y**, **S** (fully resolved) |
| `eval_matrix_1k/{trajectories,locked_test/trajectories}/<exp_id>.json.gz` | the Inspect logs behind those results | **Z** |
| `rescore10/results/<checkpoint>.json` | 1,182 ten-run evaluations under the checkpoint's own generation config | **Y**, **S** (derived, unverified) |
| `rescore10/trajectories/<checkpoint>.json.gz` | the Inspect logs behind those results | **Z** |
| `rescore10/eval/` | the evaluation kit (task scripts, chat templates, question sets) | protocol part of **S** |
| `checkpoints_meta/<checkpoint>/{config,generation_config}.json` | the two config files of every archived checkpoint | S derivation for native runs; weight identity |
| `eval_matrix_1k/policies.json` | the 13 generation policies | S |
| `checkpoint_index.json`, `manifest*.json` | checkpoint → cell → card bookkeeping | join keys only |
| `third_party/PostTrainBench/containers/standard.def` + `requirements-direct.txt` (repo submodule) | the container every scientist ran in | environment part of C |

Not used as X: `cells/<cell>/wm/cards/**` (the experiment cards). The cards are the scientist's
own description; the benchmark reads what was actually launched from the trace. The card
*submission events* inside the trace are used only as the join between a checkpoint id
(`<cell>-exp-NN`) and the workspace directory the recorder archived.

## 2. Definitions used (from the spec)

- **Example** = one (production chain, S) pair with ten complete runs. `example_id = <checkpoint_id>@<serving_id>`.
- **Checkpoint id** = `<cell>-exp-NN`, the recorder's archived copy of one workspace directory.
- **C** = the launch-time content of every file the launch depends on (training script,
  data builders, helpers, configs), the launch argv/cwd/env, the container (`standard.def`)
  plus any packages the scientist installed before the launch, chained back to the base model
  through every parent checkpoint produced in the same session.
- **Δ** = post-processing steps between a training output and the archived checkpoint:
  weight averaging ("soups"), choosing an intermediate `checkpoint-<step>` directory, editing
  the saved generation/tokenizer config.
- **S** = the resolved sampling parameters, stop set, output cap, chat template, dtype and
  context length, plus the fixed protocol (question set hash, prompt template, scorer, kit
  hashes, runtime versions).
- **Y** = mean of the ten run accuracies, recomputed from the per-question matrix, never copied.
- **Z** = per-(question, run) record normalized from the Inspect log, with the raw log kept by hash.
- **Task context** (shared with predictors): benchmark, base model, protocol, question set.
- **Splits** are by scientist session (cell). Sessions the evaluation matrix reserved as
  `locked_session_test` are the held-out test sessions; the rest are development.

## 3. Decisions taken during construction

| # | decision | reason |
|---|---|---|
| D1 | Two serving tracks. Matrix cells are `s_resolution = server_verified`; native rescore10 runs are `s_resolution = derived_unverified`. Both are examples; the flag is carried into the release. | Spec asks for the settings actually used; the matrix logged them (`server.resolved_sampling_params`), the native runs did not (see `experiments/eval_matrix_1k/runtime_preflight.md`). |
| D1a | Native S derivation rule, from vLLM v0.11.0 source (`vllm/config/model.py::get_diff_sampling_param`, `entrypoints/openai/protocol.py::ChatCompletionRequest.to_sampling_params`): the kit's request carries only `max_tokens` (4000 GSM8K / 16000 AIME) and `epochs`; with the default `--generation-config auto`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty` come from the checkpoint's `generation_config.json` when present, otherwise vLLM's neutral defaults (1.0, 0, 1.0, 0.0, 1.0); **`do_sample` is never read**, so `do_sample: false` without a `temperature` key is served at temperature 1.0; `stop_token_ids` default to the config's `eos_token_id`. Implemented in `build_labels.py::derive_native_sampling`. | Only a documented, source-cited rule can be audited; the resulting distribution (192 of 582 trace-bearing native runs at T=1.0/top-k 64/top-p 0.95, 143 at neutral T=1.0, 93 at T=0.6/20/0.95, 119 greedy, …) is consistent with the preflight's observation that nominally greedy runs varied. |
| D2 | A native run and a matrix cell of the same checkpoint are **not** merged as aliases even when their nominal settings agree. | Alias status requires equal *resolved* S; the native side cannot establish it. |
| D3 | Checkpoints whose session trace is not on HF (arms `opus5max-r0`, `opus48max-r0`, `opus47max-r0`, `glm52-r0`: 538 checkpoints with ten-run labels) are listed as *labels awaiting X* and excluded from the release until their traces are available. | X cannot be constructed without the trace. |
| D4 | Dojo-produced checkpoints (`abgsm8k-*`, 62, Qwen3 on GSM8K) are deferred to a second pass: their X lives in `dojo_ab_gsm8k/**/artifacts/step*/solution.py`, a different trajectory format. | Same spec, different extractor. |
| D5 | The 32 non-recorder cells (`c0`, `c123`) have no archived checkpoints and no ten-run labels; they contribute nothing. | No Y. |

## 4. Pipeline

Each step is a script under `tools/wm_benchmark/`; outputs live under `data/benchmark/`.

| step | script | input | output | how to check |
|---|---|---|---|---|
| 0 mirror | `data/mirror_download.py` | HF | `data/hf-mirror/` | file count and sizes vs the HF tree |
| 1 timeline | `trace_timeline.py` | `cells/*/solve_out_sanitized.txt` | `data/timeline/<cell>/{events.jsonl,fs.jsonl,summary.json,results/}` + `_files/<sha256>` | open `summary.json`; every `awm wm submit` and every launch is listed with its result |
| 2 labels | `build_labels.py` | matrix + rescore10 results, policies, checkpoints_meta | `data/benchmark/labels.jsonl`, `labels_summary.json` | Y recomputed from `per_problem`; mismatches are listed, not fixed. Result 2026-09-12: 4,182 rows; 3,000 matrix valid; 1,039 native valid, 143 native invalid (no archived `config.json`/`generation_config.json`: 97 `opus47max-r0`, 46 `opus5max-r0`); 573 native rows carry `sample_errors_unknown_until_log_checked` (older result files lack the field; step 5 settles it) |
| 3a targets | `prepare_targets.py` | labels + timeline | `data/benchmark/targets/<cell>.json` | per cell: the labeled checkpoints, their submit events, candidate launches, the yaml's `output_checkpoint` (hints only). 124 cells, 582 checkpoints |
| 3 launches | subagent extraction over `data/timeline/` | timeline | `data/benchmark/x/<checkpoint>/launch.json` (+ verified files) | see §5 |
| 4 assemble | `assemble.py` | steps 2–3 | `data/benchmark/examples.jsonl`, `x/`, `splits.json` | `verify.py` |
| 5 Z | `normalize_logs.py` | Inspect logs | `data/benchmark/z/<example>/samples.jsonl.gz` | agreement with `per_problem` |

## 5. X extraction from traces (step 3)

**Unit of work: one cell (scientist session).** For each of the 124 recorder cells that own at
least one labeled checkpoint, three agents run in sequence, all reading the same timeline
(`data/timeline/<cell>/`, produced deterministically by step 1) and never the cards:

1. **Extractor** (`tools/wm_benchmark/extract_prompt.md`) writes one *launch record* per target
   checkpoint to `data/benchmark/x_raw/<cell>/<checkpoint_id>.json`, following
   `tools/wm_benchmark/launch_record.schema.json`. A record is the production chain from the
   declared base model to the archived directory: every process the scientist ran that the
   checkpoint depends on (data builds, training launches, weight averages, checkpoint
   selection, config edits, copies), each with the Bash event number (`seq`), the full
   command, the split argv/cwd/env, the killed or failed attempts it superseded, the
   launch-time content of every code/config file it reads (by sha256 into the timeline's
   content store, or extracted from a heredoc into `x_raw/<cell>/files/`), its parent model(s)
   and data inputs, plus the environment (container, installs before the last launch, printed
   library versions) and open issues.
2. **Verifier** (`tools/wm_benchmark/verify_prompt.md`) independently re-derives the archive
   mapping, the producing launch, the launch-time file versions, the chain, Δ, the environment
   and checks for score leakage, and writes a verdict per checkpoint to
   `data/benchmark/x_verify/<cell>/<checkpoint_id>.json` (`confirmed` / `needs_fix` /
   `cannot_verify`, with findings and the seqs examined).
3. **Repair**, only when the verifier returned `needs_fix`: the extractor role re-opens the
   trace with the findings and corrects the record in place; the verifier then runs again.

Orchestration is a Workflow (`pipeline` over cells: extract → verify → repair → re-verify), 16
cells in flight at a time; every agent's transcript is kept under the session's workflow
directory.

**Rules the agents follow that a human checker should know.**
- *Archive-once.* `awm wm submit` copies `result.output_checkpoint` at the **first** submit
  whose yaml says `result.execution: completed`; later submits print the same `archived` path
  without copying (`awm/wm/record.py::submit_card`, `archive_checkpoint` raises on a second
  copy). The archived checkpoint is therefore the directory's state at that seq; later edits
  to the directory are not part of X. Step 3a records that seq as `archive_submit_seq`; 25 of
  582 targets have further submits after it.
- *Launch-time only.* A file edited after the launch keeps its pre-edit content in that step.
- *Killed and retried launches* are listed under `superseded_attempts`, never as steps.
- *Data files are not X.* Only the builder code and the build command are recorded; example
  counts the trace happens to print go to `observed_in_trace` as provenance.
- *No outcomes.* Scores, accuracies and conclusions never enter a record; card-yaml heredocs
  quoted inside a command are replaced by `<<card yaml redacted>>`.

**How to inspect one checkpoint by hand.** Open `x/<checkpoint_id>/launch_record.json`, take
any `seq`, and run
`grep -n '"seq": <seq>,' data/timeline/<cell>/events.jsonl` (the tool call) and
`cat data/timeline/<cell>/results/<seq+1>.txt` (its result); file contents are at
`data/timeline/_files/<sha256>` or `x/<checkpoint_id>/files/`. The verifier's verdict for the
same checkpoint is at `x_verify/<cell>/<checkpoint_id>.json`.

**Pilot (2026-09-12).** Cells `r0-29` (5 checkpoints, GSM8K, incl. two weight soups) and
`aime-r0-01` (5 checkpoints, AIME, incl. a two-parent merge) were extracted first and read by
hand; all ten records validate against the schema. The pilot is what produced the
archive-once rule (r0-29-exp-05 was closed three times with different `final_model`
contents), the `additional_parents` field, and the card-yaml redaction rule.

## 6. Counts

_Updated as steps complete._

| quantity | value | as of |
|---|---|---|
| cells with a trace | 160 | 2026-09-12 |
| recorder cells (r0, gsm2-r0, aime-r0, aime2-r0) | 128 | |
| checkpoints with ten-run labels, any source | 1,182 | |
| … with a trace on HF | 582 | |
| matrix cells (examples with verified S) | 3,000 over 516 checkpoints | |
| native runs on trace-bearing checkpoints | 582 | |
| held-out sessions (matrix `locked_session_test`) | 40 of 124 | |
