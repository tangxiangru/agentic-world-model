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
| **C1** | raw files of the prior runs, read-only at `/home/ben/prior_runs` and directly readable by the scientist | none | `prompt_fulltraj` | `claude_fulltraj_noawm` |
| **C2** | the same complete raw corpus, directly readable by the scientist and readable by the WMA | peer Claude Code WMA with the `traj` arm | `prompt_wm_fulltraj` | `claude_wm` |
| **C3** | full reconstructed-card corpus searchable by the WMA; no raw trajectories | peer Claude Code WMA with the `retrieval` arm | `prompt_wm` | `claude_wm` |

Crossing 3 scientist models × 3 information conditions × 2 prior scopes × 2
explicit repetitions gives **36 cells**. Every launcher spec ends in `:1` or
`:2`; the repetition is embedded in the PTB result identifier and duplicate
condition/model/scope/repetition keys in one pack are rejected.
`python rollout/study_matrix.py` emits the authoritative 36-cell design as
JSON; `--format specs` emits one launcher spec per line, and `--validate`
rejects a supplied matrix with any duplicate, missing, or unexpected cell.

The raw conditions have two scopes: the split's 143 train-side runs, or all
193 including the 50 Gemma runs. Their published scores and agent identities
remain visible (decision 2026-08-31). C3 uses the corresponding published card
source, not raw trajectories: train has 1,580 cards from all 143 runs;
train+test has 2,030 cards from all 193 runs (450 cards from all 50 Gemma
runs). The seeded side manifests record the expected, card-bearing, and missing
run inventories explicitly; both current scopes have zero missing runs.

For C2 and C3, a second, fixed-model Claude Code session serves the single
`consult` verb for the whole run. It drafts and updates the experiment card,
grounds its verdict and evaluation advice in the declared corpus, and logs each
consult under `task/wm/`. The scientist owns training, evaluation, the GPU, and
every decision. C2's WMA chooses what to inspect in the complete raw corpus;
C3 searches the complete reconstructed-card memory for the selected scope.

The WMA runs in its own cwd with its own `CLAUDE.md` and `consult` skill. Its
Claude invocation has a narrow read/communication/tool allowlist, while the
historical corpus mounts remain recursively read-only at the PTB boundary.
This is a protocol boundary inside one sandbox, not a separate Unix identity.

### One-time setup (host)

```bash
# 0. fetch the exact split first; this is idempotent and downloads only missing files
awm split fetch posttrainbench/gsm8k-gemma-holdout-v1

# 1. private PTB checkout + agents + prompts + the extra-binds patch
export PTB_SOURCE_DIR=/path/to/PostTrainBench
export HV_PTB_DIR=/data/ptb-wm-study
export PTB_RESULTS_DIR=/data/ptb-wm-study-results
export AWM_REPO_COMMIT=<full-40-hex-commit-present-in-this-checkout>
# PTB keeps this generated benchmark fixture outside Git. If it is absent:
python "$PTB_SOURCE_DIR/src/judges/test_data_download/download_test_data.py" --tasks gsm8k
bash rollout/setup.sh                       # HV_PTB_SHA can pin the PTB checkout

# 2. the prior-runs directories (copies; about 0.6 and 0.8 GB)
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test
# Record each generated corpus-manifest.json SHA-256 in the untracked submit
# environment; the launcher and every raw cell require the expected digest.
sha256sum /data/prior_runs_{143,193}/corpus-manifest.json

# 3. separate immutable inputs for the two memory scopes
awm wm --dir /tmp/seed-143 init --memory-root /data/wm-memory-143
awm wm --dir /tmp/seed-143 memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side train
awm wm --dir /tmp/seed-193 init --memory-root /data/wm-memory-193
awm wm --dir /tmp/seed-193 memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side train
awm wm --dir /tmp/seed-193 memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side test
# Record each validator's combined `manifest_sha256` in the untracked submit environment.
python rollout/validate_study_corpus.py cards /data/wm-memory-143 --sides train
python rollout/validate_study_corpus.py cards /data/wm-memory-193 --sides train,test
```

The split fetch may cache optional published artifacts, but
`tools/build_prior_runs.py` copies exactly three files per exposed run: the
complete raw Claude/Codex `solve_out.txt` stream, `metrics.json`, and
`time_taken.txt`. Every copied run file is hash/size-attested by
`corpus-manifest.json`; optional artifacts and upstream `task/` workspace
snapshots never enter the study bundle.

Pin `HV_PTB_SHA` to the PTB source revision you intend to audit. `setup.sh`
applies the idempotent portable-runner and extra-bind patches only to its
private clone, then probes payload installation, identifier-validated
environment passthrough, prompt selection, explicit GPU isolation, safe
evaluation cleanup, and honest solve-exit propagation. If the pinned runner's
shape has changed, patching fails instead of guessing at a rewrite. The GSM8K
test copy used by PTB's contamination checker is generated and Git-ignored;
setup requires a regular local copy, carries it through both private source
snapshots, and includes its content hash in the PTB surface attestation. Keep the
site values in the private PTB `.env`: this study requires
`POST_TRAIN_BENCH_ISOLATE_GPUS=1` and
`POST_TRAIN_BENCH_EVAL_GPU_REAP=own` (or `none`). The tracked launcher accepts
exactly one cell and one declared GPU per invocation. Environment masking is
not treated as device isolation: before CUDA initialization, the sandbox
temporarily removes `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` and
requires the driver to enumerate exactly one OS-accessible GPU. Therefore each
invocation needs a real one-GPU Slurm/device-cgroup allocation (or an
equivalent runtime-level device restriction); an ordinary multi-GPU
`apptainer --nv` shell fails closed. `none` signals no process. `own` considers only
PIDs on the cell's declared GPU list and signals a PID only when it has the
same Unix owner and inherited the cell's random launch token; the patched
runner has no cluster-wide `nvidia-smi | kill` path. The legacy
`POST_TRAIN_BENCH_KILL_GPU_PROCS` setting does not enable global cleanup.

`HF_HOME` must point at the exact model-only cache view accepted by
`validate_base_model_cache.py`. The launcher rejects extra files or directories,
nested credentials, token environment variables, implicit Apptainer/Singularity
environment injection, or an incomplete/changed base-model snapshot. It pins
`google/gemma-3-4b-pt` to revision
`cc012e0a6d0787b4adcc0fa2c4da74402494554d` and checks both weight shards,
their index, configuration, tokenizer, exact cache topology, and sizes before
consuming a GPU cell; smoke mode performs full content hashes of all 8.6 GB.
The exact model cache root is also mounted read-only at `/home/ben/pinned-base`,
and the first card must name its pinned snapshot as the parent. At a site with a
credential-bearing cache, construct an untracked allowlisted view containing
only this model's public artifacts (hard links are sufficient) and set
`HF_HOME` to that view in the submit wrapper. Slurm exports the submit
environment, so set it explicitly rather than relying on a private `.env` to
override an inherited cluster-wide value.

Every condition fails unless its submission is a local, non-symlink,
self-contained Hugging Face Gemma 3 4B safetensors artifact whose architecture,
tokenizer, declared identities, and exact tensor names/shapes/dtypes agree with
that pinned base. C1 writes `c1-final-model-attestation.json`; C2/C3 write
`wma-final-model-attestation.json` after the separate WMA lifecycle check. This
guards accidental model substitution; it is explicitly a
structural/declarative check, not proof of the causal process that trained the
weights.

The two study agents receive the canonical generated prompt as the read-only
file `/home/ben/task/instruction.md`, not through PTB's legacy multiline
`PROMPT` environment value. Apptainer 1.5.3 truncates that value at the first
newline when the prompt contains multiple `=` characters. The runner passes an
independent single-line SHA-256 and byte count instead; each solve verifies the
file before and after Claude runs, and the instruction plus checksum remain in
the task artifacts. Unrelated upstream PTB agents retain the legacy transport.
Each study solve also filters the retained Claude JSONL stream through the
commit-pinned `redact_claude_stream.py`: token-bearing environment assignments,
common access-key forms, authorization headers, and sensitive JSON fields are
replaced before `scientist-stream.jsonl` or PTB's `solve_out.txt` is written.
The redactor securely creates the private retained file and forwards identical
bytes itself, retrying pipe backpressure even if a shared descriptor is changed
to nonblocking mode; the study agents do not use an external `tee` process.
The scientist and WMA streams use the same redactor, and the harness scans every
text artifact under the task before accepting a cell. Any late match is atomically
redacted and the cell is quarantined rather than released. This protects
stored/published trajectories; it is not a substitute for the model-only cache
boundary.

`AWM_REPO_COMMIT` is also the exact harness commit, not merely the WMA runtime
revision. Setup refuses any live prompt builder, patcher, launcher, agent, or
instruction byte which differs from that commit. It writes a manifest of the
derived private PTB runner, prompts, agents, and payload; launch re-hashes that
surface before reading its `.env`, then records the manifest digest inside the
cell. Site paths, container locations, caches, and credentials remain local in
the private `.env` and are deliberately outside that Git/manifest surface.

Both Claude agents use Vertex. `env_passthrough.txt` contains only variable
names. The patched runner rejects anything that is not a shell identifier,
obtains each declared value from the submitting environment without `eval`,
and passes it as one argument through the clean sandbox without logging the
value. At minimum export
`CLAUDE_CODE_USE_VERTEX=1` and `ANTHROPIC_VERTEX_PROJECT_ID`, plus the regional
routing variables for the three scientist models. Also set `AWM_WMA_MODEL` to
one explicit, versioned Vertex Claude model and hold it fixed across every C2
and C3 cell; it is a study parameter, so the harness deliberately has no hidden
default. `AWM_WMA_MODEL` must be the exact canonical model ID which Claude
reports, not a generic `opus`, `sonnet`, `haiku`, or `default` alias. The WMA
uses the same configured Vertex provider and ambient ADC as
the scientist, although its model is selected separately. No credential or
OAuth token belongs in Git. On the configured H100 nodes, Claude obtains Vertex
ADC from the node's attached Google service account; both solve scripts verify
that metadata token endpoint before starting. Persistent file-based ADC is
rejected because the scientist has an arbitrary shell inside the sandbox.

Before launching, define the exact scientist alias-to-provider-ID mapping and
pin the Claude Code package in the untracked submit environment:

```bash
export AWM_SCIENTIST_MODEL_ID_4_6=<exact-ID-reported-for-claude-opus-4-6>
export AWM_SCIENTIST_MODEL_ID_4_8=<exact-ID-reported-for-claude-opus-4-8>
export AWM_SCIENTIST_MODEL_ID_5_0=<exact-ID-reported-for-claude-opus-5>
export AWM_CLAUDE_CLI_VERSION=<exact-npm-semver>
export AWM_EXPECTED_CLAUDE_CLI_VERSION_OUTPUT='<exact-one-line-claude---version-output>'
```

Do not assume an alias is the provider ID. With Vertex variables/ADC already
active, install the chosen CLI version under a temporary prefix, record its
exact `--version` line, and make one tiny stream-JSON probe per alias. Inspect
the init `model`, every assistant `message.model`, and the result
`modelUsage` keys; all reported IDs must agree, and that single value is the
mapping above (and, separately, the value used for `AWM_WMA_MODEL`). Keep the
probe output and mapping outside Git because routing is site/account metadata.
For example:

```bash
PROBE_PREFIX="$(mktemp -d)"
npm install -g --prefix "$PROBE_PREFIX" --no-fund --no-audit \
  "@anthropic-ai/claude-code@${AWM_CLAUDE_CLI_VERSION:?}"
"$PROBE_PREFIX/bin/claude" --version
printf 'Reply only OK.' | "$PROBE_PREFIX/bin/claude" --print --verbose \
  --model claude-opus-4-6 --output-format stream-json > /tmp/claude-model-probe.jsonl
```

Each cell installs that exact npm version inside its isolated home and rejects
any `--version` mismatch. The upstream `@latest` updater is forcibly disabled.
The scientist stream is preserved and the cell fails unless every actual model
ID exactly matches its configured provider ID. It also rejects any direct
Anthropic/OAuth secret present in the sandbox and requires Claude's init event
to report `apiKeySource: none` and every `modelUsage` entry to report
`provider: vertex`; Vertex routing is attested from actual runtime telemetry,
not inferred from requested environment variables. Subscription-only PTB judges
are disabled (`POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip`); GSM8K still runs its
deterministic evaluator and success requires a finite numeric `accuracy`.

### Launching

The tracked launcher contains no scheduler resources or host-specific paths
and deliberately accepts exactly one spec. Submit each spec through an
untracked site wrapper which requests a one-GPU device cgroup, then invokes
`wm_pack.sbatch` inside that allocation with exactly one identifier in
`PTB_GPU_SLOTS` (or an allocation-provided one-entry `CUDA_VISIBLE_DEVICES`).
Thirty-six simultaneous cells therefore mean 36 independent one-GPU jobs, not
36 child processes in one multi-GPU `apptainer --nv` allocation.

The wrapper must select the matching `PRIOR_RUNS`, `WM_MEMORY`, and manifest
digests from the spec's `train` versus `train,test` scope, and may set the local
`PTB_RUN_ID`/`PTB_LOG_DIR`. It must also set `POST_TRAIN_BENCH_TMP_ROOT` to a
real, owner-only mode-0700 directory on storage with at least 96 GiB and 100,000
free inodes per starting cell (or stricter local thresholds). The derived PTB
runner places the complete writable home below a unique `mktemp` directory and
removes only that exact owned directory on shell exit. `SIGKILL` or a node
failure can still leave a stale owned directory for the local launcher to reap.
Keep a machine-readable submit ledger outside Git
with at least spec, scheduler job ID, claimed device/allocation, input-manifest
digests, harness/PTB commits, and final PTB result path. Reconcile it against
the exact output of `study_matrix.py`; do not silently resubmit into an existing
result directory, because the launcher reserves the canonical result path and
refuses overwrite.

```bash
export HV_PTB_DIR=/data/ptb-wm-study
export PTB_MODEL=google/gemma-3-4b-pt PTB_NUM_HOURS=10
export POST_TRAIN_BENCH_TMP_ROOT=/data/private-ptb-scratch
# Use the same AWM_REPO_COMMIT packaged by setup.sh.
export AWM_WMA_MODEL=<explicit-versioned-Vertex-Claude-model>
export PRIOR_RUNS=/data/prior_runs_143 WM_MEMORY=/data/wm-memory-143
export AWM_PRIOR_CORPUS_MANIFEST_SHA256=<sha256-of-prior_runs_143/corpus-manifest.json>
export AWM_CARD_CORPUS_MANIFEST_SHA256=<combined-manifest-sha256-for-wm-memory-143>

# This command is run *inside one true one-GPU allocation*.
PTB_GPU_SLOTS="${ALLOCATED_GPU:?}" PTB_RUN_ID=wm-study \
  bash rollout/wm_pack.sbatch c1:claude-opus-4-6:train:1
```

Validate the complete design before submitting it. A site wrapper can consume
the one-spec-per-line form, map each scope to its local immutable inputs, submit
one one-GPU job, and append its scheduler ID to the ledger:

```bash
mapfile -t STUDY_SPECS < <(python rollout/study_matrix.py --format specs)
python rollout/study_matrix.py --validate "${STUDY_SPECS[@]}"
for spec in "${STUDY_SPECS[@]}"; do
  "${SITE_SUBMIT_ONE_GPU:?set an untracked one-GPU submit wrapper}" "$spec"
done
```

The requested release gate is exactly one train-scope C2 smoke, for example
`c2:claude-opus-4-6:traj:train:1`, submitted with the explicit untracked
`AWM_STUDY_SMOKE=1 PTB_NUM_HOURS=1` environment. It exercises the one-GPU
OS-visibility boundary, Vertex/CLI authentication, raw-corpus validation, the
peer WMA consult, requested-versus-reported model identity, harness provenance,
submission handling, and deterministic GSM8K evaluation in one cell. Smoke
mode accepts only that one-hour duration; normal production accepts only 10
hours. The result ID and `study-input.json` are labelled `smoke`, so it cannot
collide with or be mistaken for any of the 36 production cells. Require a
successful WMA-session postcondition, nonempty `final_model`, finite accuracy,
and honest zero `solve_exit_code.txt` before releasing the production matrix.
A C3 smoke is optional additional validation of the card-only protocol; it is
not part of this initial one-GPU release gate.
Smoke mode selects a separate attested prompt requiring a plan consult before
one-step training and a final shipped-outcome message to the WMA.

The condition prefix is mandatory. C1 fails if the raw mount is absent. C2
requires that same raw mount, exposes it directly to the scientist, and gives
the `traj` WMA the complete indexed raw root; it rejects historical-card
memory. C3 requires seeded card memory and rejects any raw-runs mount. The pack
mounts every historical input read-only and records the exact arm and side
scope in `wm/config.json`.

C3's separation is a study protocol, not an operating-system security
boundary: the card corpus is a read-only mount in the same Apptainer sandbox,
so the scientist process could technically address its path. The scientist
prompt forbids direct use, and the WMA is launched with its own restricted tool
policy and audited corpus roots. Treat direct scientist reads of the C3 memory
mount as protocol violations and verify the recorded trajectory/audit before
including a run. A separate container or Unix identity would be required for
hard isolation.

The tracked launcher intentionally has no node/GPU count, exclusivity, memory,
wall-time, partition, reservation, scheduler log path, or site-specific tool
path. Put those choices and any module setup in an untracked local submit
wrapper (or pass them on the scheduler command line).

### What comes back

The usual PTB result directory per cell (`solve_out.txt`,
`solve_exit_code.txt`, `metrics.json`) plus the preserved scientist stream and
CLI/model attestation files under `task/`. Optional subscription judgements are
intentionally absent. For `claude_wm` cells, the result also contains
`task/wm/`: `config.json`, `consults.jsonl`, every persisted card/consult
response, the redacted WMA session stream, and `awm_sha.txt` naming the harness
that ran. `task/` is copied whole, so nothing there is size-capped.

Every condition also writes `task/study-input.json`. For C1/C2 it records the
expected-and-observed raw `corpus-manifest.json` SHA-256, exact side scope,
split, dataset revision, run count, condition, and repetition after checking
every attested file and canonical root metadata inside the read-only sandbox
mount. For C3 it records a
combined digest of the selected side manifests and their exact card/run counts
after checking every card hash and rejecting an extra side/file. Every cell also
records the exact PTB and harness commits, derived PTB-surface manifest digest,
exact Claude CLI version, requested scientist alias, and all stream-reported
provider model IDs. C2/C3 add separate WMA model and session attestations;
production inclusion requires a normal WMA exit, a complete redacted stream,
at least one contract-valid logged consult, and a shipped outcome attached to
a consulted card. Missing/invalid finite `accuracy`, a timeout, or any
scientist/WMA/attestation failure leaves diagnostic artifacts but returns a
nonzero cell status.

### Pieces

| file | role |
|---|---|
| `patches/apply_study_runner.py` | adds safe env/payload forwarding, GPU isolation and scoped cleanup, and honest solve status to the private pinned runner (idempotent, fail-closed) |
| `patches/apply_extra_binds.py` | adds `POST_TRAIN_BENCH_EXTRA_BINDS` to `run_task.sh` (idempotent) |
| `patches/apply_scratch_root.py` | moves each writable PTB home to an explicit private scratch root, checks block/inode headroom, and cleans only its exact `mktemp` directory |
| `attest_ptb_surface.py` | setup-time manifest and launch-time verification of the derived private PTB runner/prompts/agents |
| `attest_claude_runtime.py` | exact npm CLI installation and requested-versus-reported scientist and WMA model attestations |
| `validate_study_corpus.py` | standalone in-sandbox verifier/attestor packaged into C1/C2/C3 |
| `validate_base_model_cache.py` | exact model-only cache allowlist and full-hash smoke attester |
| `validate_c1_final_model.py` | structural/declarative C1/C2/C3 final-model compatibility attester; does not claim causal training provenance |
| `validate_wma_session.py` | fail-closed C2/C3 peer-session, consult-ledger, card, and shipped-outcome postcondition |
| `redact_claude_stream.py` | credential scrubber and durable backpressure-safe dual sink for the retained/live Claude JSONL trajectory |
| `sanitize_result_tree.py` | final recursive text-artifact scrubber/attester; any redaction quarantines the cell |
| `study_matrix.py` | emits or validates the exact 36 unique launcher specs |
| `pin_ptb_source.sh` | creates the per-cell source snapshot and returns its root; cells execute relative dependencies only from that snapshot |
| `build_prompts.py` | writes the C1/C2/C3 production prompts plus separate C2/C3 lifecycle-smoke prompts; review copies under `prompts/` |
| `agents/claude_fulltraj_noawm/` | C1: Vertex Claude with a required read-only raw-runs mount |
| `agents/claude_wm/` | C2/C3: verifies the minimal payload, starts the fixed-model WMA peer, then runs and attests the scientist peer |
| `wm_pack.sbatch` | exactly one explicit `c1|c2|c3:<config>` cell per OS-isolated one-GPU invocation; fail-closed provenance, prompt, and mounts |
| `../tools/build_prior_runs.py` | the prior-runs directory + `INDEX.md` |
