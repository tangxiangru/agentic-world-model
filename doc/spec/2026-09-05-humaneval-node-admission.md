# HumanEval admission on the idle experiment node

User direction: skip GPQA, analyse the completed Opus4.8 comparison, run the
next justified experiments and use this subqueue's16 GPUs. Eight existing
knowledge/guard sessions run on ondem1. Admit ondem0 independently; do not wait
for those experiments to end or weaken isolation to fill capacity.

## Operation, not a scientist experiment

One `environment-acceptance` receipt, target `humaneval`, walltime15min,
one H100/16CPU/128G. The admitted GSM8K/Opus4.8 manifest is only the resource
and image reference profile. No scientist, provider call, real model, training,
benchmark-program execution, official score or clean scientific cell is produced.
The only programs executed are the fixed invented sandbox cases.

Use the ordinary committed launcher, hold-first receipt creation, registry
registration and separately checked release. Freeze probe/runner/helper hashes,
PTB/top source, image hashes, output root, runtime UID and positive node request.
The global site remains ondem0–1; `placement.requested_nodes` is ondem0 only.
Release, runtime, registry and results must all respect this narrower request.
Legacy receipts without placement retain their original full-site meaning.

Submit from a worker-readable shared checkout. Local `/tmp` development is
valid for construction/tests, never for the Slurm WorkDir/entrypoint or context
record paths. The earlier E-repair startup failures are retained as evidence
for the new submission guard; no failed attempt is erased or called a model run.

## Required actual evidence

Both fixed images must pass on the assigned node with `--nv`, non-root runtime
identity, their actual scientist/evaluator HOME/cwd shapes and public fixtures.
The frozen HumanEval parquet must match83,920 bytes/SHA256 and164 rows. The
generated code must not see the outer private environment/files/GPU devices.

Five invented programs run through native Inspect verify and the actual isolated
backend: correct, incorrect, import failure, wall timeout and private visibility.
Require C/I/I/I/C, one started/reaped execution per sample, actual timeout outcome
rather than an arbitrary I, and native log publication/revalidation. Preserve
request/runtime manifests, native logs and immutable Inspect snapshots.

For the separate outer interruption case, the real supervisor sends a one-time
admission event through an explicitly inherited pipe only after successful
bootstrap attestation and permission to execute. The parent validates the actual
supervisor PID and code hash; the untrusted process never inherits this channel.
Ordinary production callers use the unchanged default `on_admission=None`.

After this event, a three-second outer deadline signals the owned GNU timeout
handle. A five-second escalation grace deliberately ends before the independent
30-second program limit, so internal timeout cannot masquerade as outer cleanup.
Require the exact admitted supervisor and native sandbox to be live at that
deadline, all observed pidfds terminal afterward, and elapsed admission-to-end
below30s. Emergency cleanup, startup-only timeout, missing identity or incomplete
observation fails admission. No process is killed by shared name/user/GPU state.

Success and failure files are written to the receipt's durable result directory,
outside the job scratch cleaned at EXIT. The operator retains environment status
and raw evidence separately from scientific completion. A boolean `passed` is
insufficient: verify source/image/data/node/UID, the actual native case records,
runtime manifest identities, publication hashes, admission event and raw bytes.

## Scientific rollout after acceptance

Only a terminal receipt-backed accepted node may enter the HumanEval scientific
placement. Keep GPQA unadmitted. Then freeze the three approved method arms with
four independent sessions each, using one common new PTB pin and explicitly
replacing the unrun old-E draft with the reviewed E-repair tree. Interleave arms
where possible, retain every failure and keep metrics separate by benchmark.
AIME2025 remains reserved; this is discovery, not baseline promotion.

CPU regression/behavior checks and informed reviews are prerequisites, not
evidence the node passed. The actual acceptance receipt and its validated raw
results are the final admission gate. No HumanEval model result exists yet.

## Actual node entrypoint correction

Environment job93104 reached the frozen source on ondem0 but stopped before the probe: the worker has no `python` alias. Its exact stderr is retained in the failed environment bundle. A no-GPU, non-root read in owned job93070 confirmed `/usr/bin/python3`3.10.12, working pidfd APIs, and readable shared bwrap/Apptainer/config paths. V2 changes the host call to python3, retains the same probe logic and hashes, and freezes a new PTB commit. It is still an environment operation, not a scientist repeat or score.

## Actual outer lifecycle repair

Job93118 read all164 HumanEval rows and passed five invented native cases in
opus_5.sif on ondem0. The separate outer interruption left two live descendants
and failed; vllm image acceptance did not run. Keep its raw failed receipt.
V3 uses a production-shared child-subreaper wrapper with bounded pidfd cleanup,
revalidated ancestry and adopted-zombie reaping. Acceptance still requires the
real admission event, live sandbox at deadline and end before the inner30s
timeout. This is a source repair, never retrospective relabeling of V2.
Final focused CPU suite17 passed; related sandbox/evidence/provenance suite
passed with22 explicit native skips. Independent review found and fixed zombie
reaping and PID ownership races. Production retains the existing time budgets.
