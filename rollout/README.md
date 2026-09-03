# PostTrainBench recorder study

A collection run. The ordinary PostTrainBench scientist explores with **no prior
information of any kind** and registers every experiment it runs, by command, so
that the run leaves behind a list of reproducible training recipes — one
experiment card per launch — each of whose checkpoints earns the official
test-set score after the run.

The point is coverage of recipe space, not convergence on known-good recipes.
The earlier information conditions (C1–C3: raw prior trajectories, and a peer
world-model agent over them) are retired from the launcher; their agent
directories remain in the tree but `wm_pack.sbatch` refuses their specs.

| condition | scientist information | registration | prompt |
|---|---|---|---|
| **r** (recorder) | none | `awm wm submit` before and after each experiment | PTB + "Experiment log" + completion note |
| c0 (baseline) | none | none | PTB + completion note |

## The matrix

`rollout/study_matrix.py --format specs` is the authoritative list. Every cell
is one H100 for ten hours, and the matrix is balanced by construction:

    2 scientists (claude-opus-4-8, claude-opus-5)
  × 2 tasks      (gpqamain, healthbench)
  × 2 base models (gemma3-4b = google/gemma-3-4b-pt, qwen3-4b = Qwen/Qwen3-4B-Base)
  × N repetitions (--reps, default 2)          → 16 cells; --reps 4 → 32

A spec names its cell completely: `r:<scientist>:<task>:<base>:<rep>`. The
task and base are in the spec rather than the environment so the same wrapper
launches the whole matrix. Opus 4.6 is not in the matrix.

## What a recorder cell does

`claude_recorder/solve.sh` is PTB's Claude baseline lifecycle plus
`awm wm init --mode record --arm null` and a copy of `exp-card.template.yaml`
into the task directory. No prior runs, no memory mount, no peer session — the
no-past-trajectories condition holds at the infrastructure level, not just in
the prompt (`awm wm init` forces `arm null` in record mode even if told
otherwise).

The prompt's "Experiment log" section (`input/wma_section.md`) tells the
scientist to fill sections 1–4 of a card and run `awm wm submit` before each
launch, and sections 5–6 after. `submit` validates the card against the
sufficiency checklist in `awm/wm/record.py`, prints `missing`, snapshots the
scripts the card names (scientists edit `train.py` in place), archives
`result.output_checkpoint` under `wm/checkpoints/<card>/`, and appends to
`wm/records.jsonl`. After the run, `apply_wm_checkpoint_eval.py` has PTB score
every archived checkpoint with its own evaluator into
`$EVAL_DIR/wm_metrics/<card>.json`, so a run of ten experiments yields ten
labelled recipes rather than one.

## Baseline-equivalent execution

PostTrainBench's `src/run_task.sh` owns sandbox setup, the scientist timeout,
trace parsing, evaluation, and the result status. The private checkout carries
only mechanical bridges:

- Only names listed in `agents/<agent>/env_passthrough.txt` are forwarded
  through PTB's `--cleanenv`; these are the Vertex routing names, not
  credential values.
- The packaged awm source is copied into `/home/ben/agent` for the recorder
  (it needs `awm wm submit` and the card template).
- The generated prompt is copied to `task/instruction.md`, because long
  multiline prompt values can be truncated by Apptainer.
- PTB's stable results root is mounted at the identical absolute path in its
  evaluation container.
- `wm/checkpoints` is copied out beside `final_model` and each archived
  checkpoint gets one official evaluation.

None of these interprets credentials or model artifacts.

## Site requirements the tasks add

- **HealthBench is LLM-graded.** Its `evaluate.py` calls `gpt-5-mini` and
  declares `OPENAI_API_KEY` in `required_api_keys`, so PTB provisions that key
  into the sandbox (rule 9 forbids the scientist from using it) and the
  post-run per-checkpoint evaluation grades every archived checkpoint the same
  way — budget for it. `evaluate_openrouter.py` is the `OPENROUTER_API_KEY`
  fallback. GPQA Main is objectively scored and needs no key.
- PTB's trace judges expect a ChatGPT-subscription `auth.json`; set
  `POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip` (or `apikey`) in the launch
  environment if that is not available.
- `Qwen/Qwen3-4B-Base` and `google/gemma-3-4b-pt` must be in the HF cache
  (`containers/download_hf_cache/`).

## Prepare a private checkout

Use a new destination for every harness revision so an older derived runner
cannot leak into a launch:

```bash
export PTB_SOURCE_DIR=/path/to/PostTrainBench
export HV_PTB_DIR=/path/to/new-private-ptb
export PTB_RESULTS_DIR=/path/to/new-private-results
export AWM_REPO_COMMIT=<full committed harness SHA>
bash rollout/setup.sh
```

## Launch

`wm_pack.sbatch` accepts one spec and calls PTB directly with `NUM_GPUS=1`.
The untracked Slurm wrapper should request one H100 per invocation.

```bash
HV_PTB_DIR=/path/to/new-private-ptb \
PTB_GPU_SLOTS="$CUDA_VISIBLE_DEVICES" \
bash rollout/wm_pack.sbatch r:claude-opus-4-8:gpqamain:gemma3-4b:1
```

For a smoke, set `AWM_STUDY_SMOKE=1` and `PTB_NUM_HOURS=1`; the recorder then
uses `prompt_record_smoke`, which asks for one registered minimal experiment
end to end. A smoke is valid when `wm/records.jsonl` holds a card submitted
before and after the launch and `wm/checkpoints/` holds its archive.
