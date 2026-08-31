# rollout/ — running the studies on PostTrainBench

Two studies live here. `hv_recipe` / `hv_noop` and `hv_pack.sbatch` are the
crossed rollout (a deterministic executor replays corpus recipes; see the
comments in those files). This README is about the second one.

## The scientist × information study

Three Claude Opus scientists (`claude-opus-4-6`, `claude-opus-4-8`,
`claude-opus-5`) post-train one base model for gsm8k under three information
conditions:

| | prior information | WMA | prompt | agent |
|---|---|---|---|---|
| **C1** | raw files of the prior runs, read-only at `/home/ben/prior_runs` | none | `prompt_fulltraj` | `claude_fulltraj_noawm` |
| **C2** | the same raw files, read by scientist **and** WMA | runtime + `traj` arm (autonomous, Claude Code read-only over `/home/ben/prior_runs`) | `prompt_wm_fulltraj` | `claude_wm:<model>:traj` |
| **C3** | none directly; WMA memory seeded from the reconstructed cards | runtime + `retrieval` (or `llm` for an autonomous reader of memory) | `prompt_wm` | `claude_wm:<model>:retrieval` |

Two versions of "prior": the split's 143 train-side runs, or all 193 including
the 50 gemma runs. Scores and agent identity are visible in both (decision
2026-08-31). Whatever set a cell sees, the C3 cell it is compared with must
see the same set through memory (`--memory-sides train` vs `train,test`).

### One-time setup (host)

```bash
# 1. private PTB checkout + agents + prompts + the extra-binds patch
bash rollout/setup.sh                       # HV_PTB_DIR / HV_PTB_SHA / AWM_REPO_REF override the defaults

# 2. the prior-runs directories (copies; ~0.9 GB each)
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test

# 3. a WMA memory seeded from the reconstructed cards (train side, optionally test)
awm wm --dir /tmp/seed init --memory-root /data/wm-memory
awm wm --dir /tmp/seed memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side train
awm wm --dir /tmp/seed memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side test   # only for the "gemma in" version
```

`setup.sh` also copies `agents/claude_non_api/oauth_token` from the shared
checkout into both new agent dirs — `run_task.sh` binds it into the sandbox
only when it sits in the agent's own directory.

### Launching

```bash
export PTB_MODEL=google/gemma-3-4b-pt PTB_NUM_HOURS=10
export PRIOR_RUNS=/data/prior_runs_143 WM_MEMORY=/data/wm-memory

# C1 + C2 pack (prior runs mounted for both agents)
sbatch rollout/wm_pack.sbatch \
  claude_fulltraj_noawm:claude-opus-4-6  claude_wm:claude-opus-4-8:traj  claude_wm:claude-opus-5:traj \
  claude_fulltraj_noawm:claude-opus-4-8  claude_wm:claude-opus-5:traj    claude_wm:claude-opus-4-6:traj \
  claude_fulltraj_noawm:claude-opus-5    claude_wm:claude-opus-4-6:traj
# C3 pack (no prior runs; memory seeded)
PRIOR_RUNS= sbatch rollout/wm_pack.sbatch \
  claude_wm:claude-opus-4-6:retrieval claude_wm:claude-opus-4-8:retrieval claude_wm:claude-opus-5:retrieval
```

`PRIOR_RUNS_FOR=claude_fulltraj_noawm` restricts the prior-runs bind to C1, so
`claude_wm` cells in the same pack run as C3. Held-out cells append `:ro` to
the config (`claude_wm:claude-opus-4-8:retrieval:train:ro`) and get memory
read-only.

`AGENT_CONFIG` for `claude_wm` is `<model>[:<arm>[:<memory sides>[:ro]]]`. Arms: `null`,
`retrieval` (deterministic, over memory), `traj` (autonomous Claude Code, read-only over the
raw prior runs — C2), `llm` (autonomous, over memory + prior runs). The autonomous arms run
on `WMA_MODEL` (default `claude-opus-4-8`, baked in by `setup.sh`), fixed across cells so the
scientist model is the only thing that varies along that axis; every call and its parsed
answer is logged under `task/wm/agent-calls/`.

**A cell's label must equal what ran.** Every ping carries `agent: {arm, sources, backend, model,
retrieval_k, produced_by, degraded}`. If an autonomous call fails or does not parse, the ping is
stamped `produced_by: deterministic` with the reason, the ledger gets `agent_degraded`, and
`awm wm status` counts `degraded_calls` per card; `solve.sh` prints the count at the end.
The autonomous arms run with `--wma-strict`, so a failure at the *brief* fails the proposal
loudly rather than letting the cell continue as a null-arm cell under a C2 label. Arms that read
memory (`retrieval`, `llm`) refuse to start without the memory bind; `traj` refuses without the
prior-runs bind. When analysing, treat any cell with `degraded_calls > 0` as its own category.

### What comes back

The usual PTB result directory per cell (`solve_out.txt` — the stream-json
trajectory — `metrics.json`, judgements) plus, for `claude_wm` cells,
`task/wm/`: `events.jsonl`, every card with its contract, observations, pings,
replies, and seal, and `awm_sha.txt` naming the runtime that ran. `task/` is
copied whole, so nothing there is size-capped.

### Pieces

| file | role |
|---|---|
| `patches/apply_extra_binds.py` | adds `POST_TRAIN_BENCH_EXTRA_BINDS` to `run_task.sh` (idempotent) |
| `build_prompts.py` | writes `prompt_fulltraj.txt`, `prompt_wm.txt`, `prompt_wm_fulltraj.txt` into the checkout; review copies under `prompts/` |
| `agents/claude_fulltraj_noawm/` | C1: PTB's Claude scaffold, unchanged |
| `agents/claude_wm/` | C2/C3: clones awm at `AWM_REPO_REF`, `awm wm init`, skill + Stop hook, then the same Claude invocation |
| `wm_pack.sbatch` | 8 cells/node, `<agent>:<config>` arms, per-cell prompt and binds |
| `../tools/build_prior_runs.py` | the prior-runs directory + `INDEX.md` |
