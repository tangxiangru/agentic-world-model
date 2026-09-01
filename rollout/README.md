# PostTrainBench WMA study

The study runs the ordinary PostTrainBench scientist and evaluator under three
information conditions:

| condition | scientist information | peer WMA | prompt |
|---|---|---|---|
| C1 | raw prior trajectories | no | PTB + prior-runs section |
| C2 | raw prior trajectories | trajectory WMA | PTB + prior-runs + WMA sections |
| C3 | none of the raw trajectories | retrieval WMA over cards | PTB + WMA section |

The matrix is two scientist models (`claude-opus-4-8`, `claude-opus-5`) ×
three conditions × two information scopes (`train`, `train,test`) × two
repetitions: 24 one-H100 cells. `rollout/study_matrix.py --format specs` emits
the authoritative list. Opus 4.6 is not in the matrix. The peer model defaults
to `claude-opus-5` and can be selected with `AWM_WMA_MODEL`.

## Baseline-equivalent execution

PostTrainBench's `src/run_task.sh` owns sandbox setup, the scientist timeout,
trace parsing, evaluation, and the result status. C1 uses the same Claude CLI
arguments and update step as PTB's `claude_non_api` agent. C2/C3 add only the
peer WMA process. There is no study-specific credential scanner, stream
redactor, result-tree sanitizer, model/cache attester, final-model validator,
or release gate in the execution path.

The private PTB checkout has five small mechanical bridges for the two study
agents:

- `POST_TRAIN_BENCH_EXTRA_BINDS` mounts the selected prior corpus or card
  memory read-only.
- Only names listed in an agent's `env_passthrough.txt` are forwarded through
  PTB's `--cleanenv`; these are the Vertex routing names and the WMA model
  selection, not credential values.
- The packaged WMA source is copied into `/home/ben/agent` for C2/C3. C1 has no
  payload.
- The generated prompt is copied to `task/instruction.md` for the two study
  agents because long multiline prompt values can be truncated by Apptainer.
- PTB's stable results root is mounted at the identical absolute path in its
  evaluation container, so the evaluator can open `final_model` on relocatable
  unprivileged Apptainer installations.

None of these bridges interprets credentials or model artifacts. PTB's own trace
parsing and any sanitization it normally performs remain unchanged.

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

`setup.sh` copies the source PTB `.env` unchanged except for its results
directory, installs the two agents and prompt templates, and packages the WMA
code for C2/C3. Authentication, containers, caches, scheduler behavior, and
evaluation settings therefore come from the same PTB site configuration as a
baseline run. The optional offline corpus credential guard and unused strict
LLM-arm implementation are excluded from the sandbox payload.

## Launch

`wm_pack.sbatch` accepts one spec and calls PTB directly with `NUM_GPUS=1`.
The untracked Slurm wrapper should request one H100 per invocation and choose
the scope-matched `PRIOR_RUNS` or `WM_MEMORY` path.

```bash
HV_PTB_DIR=/path/to/new-private-ptb \
PRIOR_RUNS=/path/to/train-priors \
PTB_GPU_SLOTS="$CUDA_VISIBLE_DEVICES" \
bash rollout/wm_pack.sbatch c1:claude-opus-4-8:train:1
```

For a smoke, set `AWM_STUDY_SMOKE=1` and `PTB_NUM_HOURS=1`; C2/C3 then use the
smoke prompt. A smoke is considered WMA-valid only after its logs show the peer
started and the scientist consulted it. That monitoring decision is external
to PTB and cannot turn a completed baseline-compatible evaluation into a
custom sanitizer failure.
