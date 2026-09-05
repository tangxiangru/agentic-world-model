# Strict control eighth-repeat repair — 2026-09-04

Status: **registered held as job91965/c02s01**, no release, result or efficacy claim. This is not a new protocol candidate or a one-cell comparison. Its scientist remains protocol-free. Preparation/held registration is independent of Window04 synthesis; release still needs the native-isolation/explicit per-receipt authorization gate.

## Why one repeat, and why this attempt

The [original strict-control design](2026-09-02-exp-protocol-round00-null-control.md), fourth batch, froze8 protocol-free strict-site repeats. Its [receipt](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8-v1/formal-2026-09-02T210446.182614+0000.json) now has7 eligible clean completions plus c01s08/job90820, which is validator-complete but placement-quarantined. Runtime node `slurm2-a3nodeset-1` is outside the receipt's frozen ondem0–1 boundary. The placement violation was investigated before that job's official result, and the exclusion is unrelated to its score.

Keep all eight original attempts, including90820, in history. Its0.7717968157695224 remains sensitivity evidence; do not delete, relabel, or substitute a developer score. A fresh independent repeat restores the **intended strict cohort's8 eligible observations** for the protocol-free reference alongside strict guard/baseline cohorts. This is cohort completion, not a precision extension or evidence that n=1 establishes a variant. If a later scientific decision makes the whole remaining reference work unnecessary, withdraw the unstarted repair explicitly; no automatic repeat-until-good loop.

## Frozen replacement contract

Manifest: `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1.yaml`.

- New batch `exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1-v1`; new cell `c02s01`, manifest-local replicate1; `run_index: 5` is new for this control line.
- Cohort mapping: original strict eligible c01s01–07 plus c02s01 is the repaired8. c02s01 is an independent attempt assigned to the missing eighth-repeat role, not the same random trajectory as c01s08. It has no paired seed claim.
- Exactly the original c01s08 scientist/model/effort/context, task/base revision,10h/1GPU resource contract, containers, judge profiles, context validation and AWM source/setup. AWM remains `eaf50919ff5f79f15e33df7bb49f44ffebacfc64`, five code paths only, `--tool claude`, no protocol skill/tree. PTB must remain `dcf5da031435c54e3680b6ec3f63e7e317efc13e`.
- Only batch/description/spec/cell/run identities and manifest-local replication shape differ. No experiment, runtime protocol, prompt, evaluation or judge change is bundled with the repair. The full source comparison is checked before registration.
- The new receipt freezes its own branch head and site. Historical receipts/manifests remain immutable. All analyses retain per-attempt timestamps and cohort membership; a later repeat is not magically contemporaneous with the original eight.

## Gates and failure handling

Before held registration: local/full manifest checks, same-contract comparison, clean top/PTB worktrees, current registry-aware OWNERSHIP OK and exact new batch non-duplication. Use queue `want: held` and the committed `awm ptb reconcile` path, never hand-written sbatch. Verify the returned receipt/job and actual JobHeldUser/ReqNodeList against frozen ondem0–1. No ownership failure permits submission, even if a lower-level held code path would allow it.

Native reservation equality is a **release** gate. This new plan authorizes no release and adds no shared-reservation override; held-only registration consumes no GPU allocation. Do not extend the09:39 exception for90791–90798 to this job. The current user question about restoration/exception remains unanswered unless a real reply arrives.

If the repair terminates incorrectly, harvest and validate it exactly once through the normal workflow; failed/truncated/placement-only evidence remains separate. Decide any further attempt from the remaining strict/matched evidence, never automatically or by score. Upon eligible completion report original strict7, repaired8 and placement-sensitivity separately, and include the new trace in a future eligible analysis window without altering Window04's frozen14. A new immutable batch does not automatically join an old manifest's aggregate; derive the explicit mapped cohort.

Held accounting before this repair:29 actual JobHeldUser, including D4 under scope review and staleE4 awaiting decision;21 other independently specified held cells. A successful held-only repair adds one to30 actual /22 excluding D/E. No cancellation or release is needed. Recompute live counts rather than treating this arithmetic as evidence.

## Pre-registration validation

2026-09-04, before freezing this plan: `awm ptb check` returns0 issues in both local-only and full site modes. A parsed comparison asserts identical contracts except run_index/replication, identical c01s08/c02s01 cell settings except id/replicate, identical context validation and ownership branch. PTB remains clean at `dcf5da031435c54e3680b6ec3f63e7e317efc13e`. The dry operator plan contains exactly one held-only submission and no release/cancel/harvest. Current-checkout registry view05:23:45 is OWNERSHIP OK,0 running,29 held, no unknown/name/placement/capacity violations. Recheck ownership immediately before apply; this timestamp is not a permanent authorization.

## Held receipt and monitor update

Plan/manifest/queue frozen in `ab8a0b2`; fresh ownership05:25:00 was OK. Normal operator apply registered[receipt](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1-v1/formal-2026-09-04T052527.042436+0000.json) at05:25:27, job91965. The [verification](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1-v1/audit/held-verification-20260904T052736Z.json) asserts actual PENDING/JobHeldUser,zero runtime and ReqNodeList equal to frozen ondem0–1; `awm slurm show91965` resolves the registry to this exact receipt and cell. Source top/PTB worktrees were clean. At05:26:37 current ownership remains OK,30 actual held,0 running/allocated; no release/cancel occurred. The derived verification was subsequently moved under `audit/`: batch-root JSON files are reserved for receipts and placing it there made operator reconciliation reject it as an invalid receipt. The submission receipt was not changed.

The hourly detector's watched set was intentionally expanded from21 to22 receipt-backed IDs by adding91965 and retaining every prior ID. Before the change PID2579442 was verified as the owned detector; its last state is archived as `window04-local/held-monitor-before-expansion.json`. It was deliberately terminated and replaced by live PID2612586, not restarted on an observation timeout. First tick05:27:36:0/22 terminal; threshold6 plus2 buffered clean tail cells unchanged. No cumulative IDs were dropped. Log `/tmp/exp-protocol-held-monitor-expanded-wboe41dy/monitor.log`; next nominal tick06:27:36.
