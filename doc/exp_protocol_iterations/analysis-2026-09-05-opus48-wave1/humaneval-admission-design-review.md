# HumanEval node-scoped admission design review — 2026-09-05

Independent informed design review; no submission, login, queue change, training
or model evaluation. Reviewed AWM code in `/tmp/exp-protocol-nextwave-ckg3t0gc/repo`
and PTB `60df491` in `/tmp/ptb-opus48-onboarding-oGLZFUf4/repo`.

**Conclusion:** accepting ondem0 first and restricting the first HumanEval wave
to that node is a sound asynchronous path. Acceptance on ondem0 does not admit
ondem1. Keep the registry/site/reservation scope at ondem0–1; freeze the extra
exclusion separately and enforce it at submission, release, runtime and result
placement. The parent reports current capacity and authorization; this review
does not independently assert live occupancy or grant a queue-policy exception.

## Existing support and the smallest extension

`submit.sh` already supports held submission, `--runtime-smoke`, short walltime
and the existing Slurm entrypoint. `single_task.sbatch` drops the root allocation
to the configured non-root identity, materializes the frozen PTB commit, and
provides allocated GPU visibility. Reuse these mechanisms, not a new sbatch file.

There is no complete existing receipt-backed environment admission command:
`awm ptb context-smoke` returns job IDs without the formal receipt/registry/held
release chain. Its runtime branch checks the scientist image's torch GPU and
may call `context_probe.sh`, which can write the context record and contacts a
provider. It neither checks HumanEval nor exercises both images and outer
timeout. A one-hour pilot still runs a scientist/model and is not a substitute.

Recommended narrow additions:

1. Add an explicit manifest operation `environment-acceptance`, with target
   `humaneval`, frozen probe profile/path/hash, short walltime, output contract
   and placement. An admitted GSM8K contract can remain the compatibility
   profile used by existing common checks, but it must be named as a reference
   profile: measured task is absent, target environment is HumanEval, and no
   scientist/model is launched. Validate the target assets and probe separately;
   do not infer HumanEval admission from the carrier task's allowlist status.
2. In `awm/ptb_experiments.py`, reuse `submit()` source freeze, incremental
   receipt persistence, hold-first submission, registry registration and
   `release_held()`. Emit `kind` and `run_purpose` exactly
   `environment-acceptance`; never prefix it with `formal`. Freeze operation,
   exact target, image/helper/probe identity, exclusions and durable output path
   in the receipt. Disallow pilot/retry combinations for this operation.
3. Extend PTB `submit.sh` with a narrow
   `--environment-acceptance humaneval` option and explicit per-job `--exclude`.
   Reject combination with legacy runtime-smoke/preflight flags. Export the
   operation deliberately; absent operation/exclusion must not inherit an
   unrelated ambient setting. Preserve `--hold`, one-node/one-GPU resources,
   full site nodelist and short walltime. The new branch must exit before the
   scientific `run_task.sh` and before provider/context probing.
4. Commit the probe and runner in PTB before receipt creation. Use the frozen
   materialized source, not a mutable `/tmp` script. The existing five-case
   probe is reusable source material; its current hard-coded image/paths must
   be parameterized explicitly for the two actual layouts. Adding this runner
   produces a new common PTB pin beyond `60df491`, which both acceptance and
   subsequent HumanEval receipts must identify.

## Operator and result identity

`awm/ptb_ops.py:plan()` currently treats only receipt filenames whose kind
starts with `formal` as submitted production work. Merely adding a new receipt
kind will otherwise cause duplicate submit planning. Add an explicit
operation-to-receipt-kind selector: ordinary manifests accept formal kinds;
environment manifests accept their exact environment kind. Both use the same
held/release lifecycle. Existing terminal/cancel loops can remain shared.

Add an environment-specific terminal harvest branch in `apply()` (and manual
harvest if exposed). Resolve its output through the exact receipt/job/cell,
not scientific `result_for_job()` discovery. Preserve raw stdout/stderr,
interrupted attempts and the acceptance report even on nonzero/timeout exits.
Set `scientific_result: false`, `complete: false`, `eligible: false`, plus a
separate `acceptance_state`; do not fabricate metrics, final_model or judges to
pass the scientific validator. A successful operation is operationally done
without being a clean scientific cell, repeat or held-buffer cell.

`ptb_results._comparable_run_purpose()` already excludes a non-formal purpose;
retain that boundary and test that missing/forged purpose cannot turn probe
output into a scientific result. Registry registration accepts receipt kinds
generically, so a separate ownership mechanism is unnecessary.

## Exclusion and admission must agree

For the proposed first wave, freeze requested site nodes `{ondem0, ondem1}`,
excluded nodes `{ondem1}` and effective nodes `{ondem0}`. Use canonical expanded
names; reject an empty effective set, foreign excluded nodes and inconsistent
declarations. Slurm documents exclusions as preventing allocation on those
nodes, and allows a resource-sized subset of a supplied nodelist. The actual
controller must still accept this combination; confirm the held job's reported
ReqNodeList and ExcNodeList before release, rather than assuming normalization.
[Official sbatch semantics](https://slurm.schedmd.com/sbatch.html)

Keep `site_issues()` and native reservation comparison unchanged. Add the
exact per-job ExcNodeList check alongside the existing ReqNodeList check in
`release_held()`, and require the resulting effective nodes to be covered by
the frozen acceptance evidence for a HumanEval release. A post-submission
exclusion change must fail release. Old receipts without exclusions keep their
original full-site interpretation.

Reuse one receipt-placement helper across `receipt_status()`,
`ptb_results._expected_nodes_by_job()` and `ptb_ops._receipt_expected_nodes()`;
the latter needs the job identity if exclusions vary by cell. These presently
use full site nodes. They must quarantine a first-wave result on ondem1 even
though that node belongs to the same subqueue. Add a runtime assertion before
any scientist/benchmark work and include effective placement in provenance.
The registry monitor's subqueue ownership check remains an independent outer
bound; where it exposes per-receipt violations, include the narrowed placement.

## What acceptance must actually establish

One short ondem0 allocation can check both images sequentially. Each must use
its actual scientist/formal evaluation contain/pid/cleanenv/bind/tmp layout with
`--nv`, not just append --nv to a head-node CPU command. Verify allocated GPU
visibility in the outer image while generated code cannot access GPU devices,
private files or private environment. Check shared parquet and bwrap bytes and
permissions under the real execution identity without printing benchmark data.

Run the existing invented correct/incorrect/import/timeout/private programs
through native verify and the real backend; retain execution reports, native
JSON, snapshot publication and revalidation. Separately interrupt an active
outer container at a short timeout and verify owned-descendant cleanup and
failure evidence. An internal 30-second program timeout does not prove outer
timeout cleanup. There is no scientist invocation, provider call, training,
real-model inference or execution of benchmark examples.

The scratch EXIT trap removes the job scratch. Persist successful and failed
probe artifacts to the receipt's durable output path before that cleanup;
ephemeral files plus a terminal Slurm exit code are insufficient. Bind the
report to job/cell/node, runtime UID, frozen source/probe/helper/limits, both
image hashes, dataset hash and exact checks actually executed. Admission must
validate these bindings and require every mandatory check to pass, not trust
an unscoped `success: true`.

## Minimum tests and release sequence

- Manifest rejects unknown targets, operation/pilot combinations, untracked or
  changed probe, forged admission and inconsistent/exhaustive exclusions.
- Fake scheduler integration proves held-first receipt registration, partial
  submission accounting, no release on ownership failure, exact Req/Exc checks,
  and unchanged full-site reservation gate.
- Operator reconciliation does not duplicate environment jobs; terminal probe
  harvest retains failure evidence and never becomes scientific completion.
- Probe runner tests demonstrate both image layouts and outer cancellation;
  node acceptance is based on actual receipt-backed execution afterward.
- HumanEval admission/release fails for unaccepted nodes or changed source,
  image, helper, limits, data or probe bindings; placement on excluded ondem1 is
  quarantined in both discovery and harvest.

Freeze and validate the extension, submit the short environment job held,
register and release it through the normal gates, then validate its actual
ondem0 evidence. Only then admit and freeze the HumanEval wave constrained to
ondem0. Later ondem1 acceptance can independently enable a new unconstrained
receipt; it must not rewrite the first wave's frozen exclusions. Scientific
queue/buffer policy remains the main agent's separate authorized decision.
