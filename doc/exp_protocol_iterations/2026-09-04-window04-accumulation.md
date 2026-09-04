# Window04 accumulation — 2026-09-04

Status: **5 new receipt-backed validator-clean cells; waiting for at least8 before local Claude dispatch**. This is a collection checkpoint, not a completed analysis window or a protocol decision. Original bundles were committed and pushed in `7043fa4`.

| cell | job | arm | official accuracy |
|---|---:|---|---:|
| g01r04 |90650|old session-guard cohort|0.690674753601213|
| g01r05 |90651|old session-guard cohort|0.7611827141774071|
| g01r06 |90652|old session-guard cohort|0.7081122062168309|
| g01r08 |90654|old session-guard cohort|0.7028051554207733|
| c01s06 |90818|protocol-free strict control|0.7414708112206216|

All five are formal, PTB-complete, eligible, non-quarantined and judge-clean. Both relevant manifests were re-discovered with `awm ptb results --json`, not inferred from Slurm terminal state. The [accumulation roster](trace-reviews/window04-local/accumulation.json) preserves each receipt→cell→manifest→spec→bundle/raw-result path and frozen source identity. [Partial collect](trace-reviews/window04-local/collect.partial.csv) is raw bookkeeping only; its pitfall sums and matched counters have not received the required deep trace review.

The old guard manifest is now6/8 clean (g01r03/g01r07 still running), mean0.7251705837755876, max0.7778620166793025. Strict control is1/8 clean. The seven formal cohorts in the operator view now have45 eligible completions: **44 clean plus c00r02's separately flagged completion**. Highest clean official scores remain control c00r03=0.7968157695223654 and protocol g01r02=0.7778620166793025; no record was refreshed by this checkpoint. These partial scores do not establish a variant effect or promotion.

`awm_sha` alone is not proof a control used the protocol: c01s06's frozen bootstrap commit is `eaf50919`, but its five shipped paths exclude `skills/exp_protocol` and setup is only `--tool claude`; there are zero cards. The four guard cells ship `4ae3d87`, tree `189319d6`, and install the protocol plus Stop hook. Keep these conditions explicit in the future review.

No local Claude session is started yet. Do not count old Window03/strict-addendum/P5 cells again to manufacture eight NEW. At the next threshold, harvest every terminal attempt and freeze all eligible unreviewed cells available for the window, not an outcome-selected eight. Generate per-cell facts/timelines, then use local Opus5[1m] max read-only reviewers and an independent synthesis under the existing contract; the planner will adjudicate and write reusable lessons to meta.

## Operations and dependencies

Preparation follow-up: the five cells now have mechanical facts/timelines in
[prepared inputs](trace-reviews/window04-local/input-notes.md), with generator
and output hashes. This is not deep trace review or a scientific decision;
Window04 remains5/8 and undispatched. Future reviewers must use the named raw
result directories when git-bundle size caps omit relevant evidence.

03:02:42 UTC live queue: **11/16 owned GPUs allocated**,12 registered running jobs (one is the outside-node90820),29 actual `PENDING(JobHeldUser)`, no other pending reason. Five owned GPUs are idle in that snapshot. Native reservation still spans11 nodes. OWNERSHIP FAIL persists because90820 is outside `slurm2-a3nodesetondem-[0-1]`; no submit/release/cancel is authorized by this collection checkpoint. The earlier per-block shared-reservation exception applies only to the already-completed strict guard90791–90798.

The five harvest actions were explicitly isolated from all submit/release/cancel actions; no existing bundle was overwritten. An initial orchestration call used a nonexistent display method and exited before mutation; the subsequent call used the actual `Action.line()` API and completed normal operator harvest/validation. Raw harvested evidence is preserved as emitted, including its formatting.

The original hourly monitor PID2086813 remains live with all17 watched IDs. Its03:00:54 snapshot has5 terminal, below its8-attempt threshold. **Do not restart it or remove these five IDs:** the next three terminal attempts should trigger the existing cumulative threshold, after which scientific clean counts must still be checked. Restarting on only the remaining12 with threshold8 would unnecessarily wait for13 total terminals.

D/B/H plus drift A remain the first scientific wave when operational gates pass; A/E2 plus drift B follow subject to the already-recorded held-floor arithmetic. E2/J/K remain unregistered; old E stays held pending its replacement receipt. No new scientific evidence has yet been adjudicated to prune another block. Before future releases, both90820's ownership violation and native isolation need resolution; running jobs are not cancelled automatically.
