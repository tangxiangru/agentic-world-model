# Opus 4.8 cross-benchmark launch record

Status: production acceptance passed; all **60 scientific jobs submitted** in
15 four-cell manifests. At 17:31 UTC the WMA subqueue is ownership-clean,
16/16 GPUs allocated, with 45 safely routed pending jobs. Allocation does not
measure GPU utilization. The study has zero completed scientific results and
two incomplete terminal attempts; see the dated checks below.

Study contract: `doc/spec/2026-09-04-wma-opus48-crossbench.md`. The user first
requested the broader comparison and then explicitly waited for implementation
completion. The new implementation record states completion; this task then
reproduced the shared-UID isolation failure, fixed the resource policy without
removing the I/O boundary, and added explicit single/multi-self/multi-joint
study modes. The existing code and analysis work was preserved.

The scientist and WMA model route is Opus4.8/high/200k; the diagnostic partner
remains Opus5/max. The host route probe returned MODEL_ROUTE_OK. Shared tests
exercise real kernel file/network/inheritance canaries and the protocol flows;
the production SIF and real model/broker gate are still required.

The submitted scientific matrix is GSM8K/BFCL/HumanEval x R/P/S/M/J x four repeats.
GPQA-main is blocked by the dataset's access permission, after a legitimate
request using the existing Hugging Face login. BFCL and HumanEval contamination
assets were downloaded through the repository helper and their hashes recorded.

At the initial check the queue was ownership-clean but underfilled: nine GPUs
allocated and zero pending, 80 validator/judge-clean old results under the old
contracts. The current c10r08 tail and G/H wave remain running. They are not
dependencies for this new study's independently specified baselines.

Evidence and deployment audits: `evidence/2026-09-04-crossbench/`.

## Deployment compatibility and validation

At 2026-09-04T13:34:11.028776+00:00, the selected exp_protocol/WMA/sandbox/launcher/operator regression suite passed, including real Landlock/seccomp canaries in the non-root shared UID environment. The new high/200k scientist scaffold shell test and changed-source Ruff checks also passed.

The common deployment changes add explicit study modes, keep current review/action/version checks, supervise each probe group without imposing a UID-wide NPROC=128, and register/hold/route validation jobs through the formal receipt lifecycle. Validation jobs are excluded from scientific harvesting. Opus4.8 formal WMA cells additionally require a matching real SIF/model/broker acceptance artifact; the receipt freezes its hash. The completion hook discovers all receipt-backed wma-prefixed study manifests within the same subqueue and stratifies reports by task/model/mode.

Frozen implementation source: `31b854bbc5e1f7f66685a8ec0d43845a6c2472f1`. The final shared regression suite passed **430 tests**; the added multi-self test also proves that disabling joint comparison retains the selected-card review. PTB profiles and synthetic production acceptance are committed. Six R/P manifests (24 cells) are staged pending the real context gate; a separate validation-only manifest exercises the actual SIF and model/broker path.

Production validation job **92160** was registered and released after its held
route passed. On slurm2-a3nodesetondem-2 it verified one visible H100, exact
Opus4.8/high/200k, CLI2.1.219, the image digest, OS canaries and real isolated
joint+single-card calls (55.1s each). All retained model tool inventories contain
only the four wma_probe MCP tools. Its public/private source is `31b854bb`.
The eight GSM8K R/P cells passed full manifest checks and are activated first;
BFCL/HumanEval await committing their downloaded task assets, not any outcome.

The eight GSM8K R/P jobs were submitted first: **92163..92170**. Their PTB
source is 0bb448c. The next PTB revision adds only the BFCL/HumanEval task-data
copies and their attribution; a direct diff confirms all runtime, scientist,
judge, container and GSM8K files remain identical. Source/data assets are now
tracked instead of bypassing the existing frozen-asset gate. All 15 study
manifests retain identical public/private source `31b854bb`, image/model settings
and per-task inputs; WMA manifests pin the successful acceptance artifact.
The remaining 52 cells can be submitted after their full manifest checks.

All remaining 13 manifest checks returned **0 issues**. Together with the two already-submitted GSM8K R/P manifests, the complete 60-cell study is ready. A scoped reconcile preview is used so the remaining validated submissions happen before unrelated legacy harvesting.

## Completed launch and ownership audit

All 60 cells were released by 14:06:12 UTC after held-route checks. The 14:10
full audit and 14:14 live queue check agree: 16 RUNNING, 53 PENDING, ownership OK;
all 53 pending requests target exactly slurm2-a3nodesetondem-[2-3]. Seven new
GSM8K R/P cells are running, and 53 new cells are pending. The other nine
running cells are c10r08 and the four-cell G/H arms. No job was cancelled.

The pending cells have no dependency on an old straggler or another scientific
result: all context/runtime/data gates have already returned. They wait for
scheduler capacity; the 53-cell reserve exceeds the 32-cell target and >8 floor.
The final scheduler query confirms `(null)` dependencies for all 60 new jobs;
52 pending jobs report Priority and one reports Resources.
The two earlier GSM8K receipts use PTB 0bb448c; the remaining 13 use e62036f,
whose only additions are BFCL/HumanEval task assets and license notices.
AWM public/private implementation remains 31b854bb for every protocol/WMA cell.

| Task / arm | Cells (replicates 1–4) | Exact job IDs | Frozen receipt |
|---|---|---|---|
| bfcl-joint | w62r01, w62r02, w62r03, w62r04 | 92219, 92220, 92221, 92222 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-bfcl-joint-x4/formal-2026-09-04T140456.076627+0000.json) |
| bfcl-multi-self | w61r01, w61r02, w61r03, w61r04 | 92215, 92216, 92217, 92218 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-bfcl-multi-self-x4/formal-2026-09-04T140431.092631+0000.json) |
| bfcl-protocol | c54r01, c54r02, c54r03, c54r04 | 92185, 92186, 92187, 92188 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-bfcl-protocol-x4/formal-2026-09-04T140136.422517+0000.json) |
| bfcl-raw | c53r01, c53r02, c53r03, c53r04 | 92181, 92182, 92183, 92184 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-bfcl-raw-x4/formal-2026-09-04T140111.713356+0000.json) |
| bfcl-single | w60r01, w60r02, w60r03, w60r04 | 92211, 92212, 92213, 92214 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-bfcl-single-x4/formal-2026-09-04T140406.445216+0000.json) |
| gsm8k-joint | w59r01, w59r02, w59r03, w59r04 | 92206, 92207, 92208, 92209 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-gsm8k-joint-x4/formal-2026-09-04T140340.636348+0000.json) |
| gsm8k-multi-self | w58r01, w58r02, w58r03, w58r04 | 92202, 92203, 92204, 92205 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-gsm8k-multi-self-x4/formal-2026-09-04T140315.960001+0000.json) |
| gsm8k-protocol | c52r01, c52r02, c52r03, c52r04 | 92167, 92168, 92169, 92170 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-gsm8k-protocol-x4/formal-2026-09-04T134801.355200+0000.json) |
| gsm8k-raw | c51r01, c51r02, c51r03, c51r04 | 92163, 92164, 92165, 92166 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-gsm8k-raw-x4/formal-2026-09-04T134734.872385+0000.json) |
| gsm8k-single | w57r01, w57r02, w57r03, w57r04 | 92198, 92199, 92200, 92201 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-gsm8k-single-x4/formal-2026-09-04T140251.018528+0000.json) |
| humaneval-joint | w65r01, w65r02, w65r03, w65r04 | 92231, 92232, 92233, 92234 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-humaneval-joint-x4/formal-2026-09-04T140611.305192+0000.json) |
| humaneval-multi-self | w64r01, w64r02, w64r03, w64r04 | 92227, 92228, 92229, 92230 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-humaneval-multi-self-x4/formal-2026-09-04T140545.881036+0000.json) |
| humaneval-protocol | c56r01, c56r02, c56r03, c56r04 | 92193, 92194, 92195, 92196 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-humaneval-protocol-x4/formal-2026-09-04T140225.235582+0000.json) |
| humaneval-raw | c55r01, c55r02, c55r03, c55r04 | 92189, 92190, 92191, 92192 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-humaneval-raw-x4/formal-2026-09-04T140200.650211+0000.json) |
| humaneval-single | w63r01, w63r02, w63r03, w63r04 | 92223, 92224, 92225, 92226 | [receipt](../../results/ptb/wma-crossbench-opus48-r05-humaneval-single-x4/formal-2026-09-04T140521.082676+0000.json) |

Full receipt → cell → manifest → spec provenance and source/skill hashes are
in `evidence/2026-09-04-crossbench/study-provenance.json`; scheduler routes are
in `launch-audit.json`. Result bundles will appear under each receipt’s batch
in `results/ptb/`; authoritative final paths must still be discovered through
`awm ptb results MANIFEST --all --json` and pass validation/judges.

The three older ready Claude windows (00:34, 07:37, 09:38) have now been
reviewed against the integrated audit: all 30 new cells match provenance, score
and flags. Their proposals are superseded by the independently preregistered
redesign study, not automatically accepted or promoted. The detailed
`event-handoff.md` preserves rejected saving/compliance claims and four
nonblocking historical audit limitations. No original scorer or flag changed.

The existing hourly read-only Opus5/max hook remains alive (PID 3591763).
Validation-only job 92160 is excluded from scientific completion counts.

## Final result refresh and handoff

At 2026-09-04T14:15:08.357672+00:00, all 45 scientific manifests were refreshed through
`awm ptb results MANIFEST --all --json`: 80 complete results with zero PTB
validator issues and zero automatic judge flags; zero completed results belong
to this new Opus4.8 study. These status checks do not override separate WMA
access flags or the confirmed D/w09r03 semantic exposure.

The inspected global reconcile preview contained 32 harvests and 16 active
peeks, with no new submissions or cancellations. Applying it completed all
32 bundles and replaced their obsolete in-flight mirrors. Some jobs are Slurm
FAILED with complete valid PTB outputs; their scheduler failures are retained
in each status, not retried or reclassified. `legacy-harvest.json` records the
exact terminal jobs and scientific outcomes; `results-at-launch.json` freezes
all current completed provenance. Existing original cohorts, exclusions and
primary/sensitivity definitions continue to apply.

This check introduces no new outcome-based candidate or promotion. All 60
first-wave settings are fixed independently of unfinished G/H/c10r08 tails.
The nominal budget remains 600 GPU-hours plus preparation/grading; actual
scientist/WMA time, spend, wait overhead, within-arm variance and failed-cell
costs are pending and must be recorded in the readout.

## 15:00 UTC operator check

The queue remains ownership-clean with 16/16 GPUs allocated and 51 safe pending
jobs. GSM8K protocol c52r04 (92170) and BFCL raw c53r01 (92181) started as two
allocations ended. All remaining jobs keep their frozen treatments; no scheduler
dependencies or replenishment are needed. Utilization coverage remains partial.

New raw GSM8K c51r01 (92163) is **incomplete**, not a zero or valid accuracy:
the CLI ended after a final reply promising to wait for a background monitor,
then background tasks stopped before a final model was delivered. PTB reports
four missing-artifact/metrics issues and general_anomaly. Its scientist cost
was $3.50933425 and allocation 0.7244 GPU-h. The exact responsibility between
scientist waiting behavior and CLI lifecycle remains unconfirmed; no selective
retry or change to the frozen runtime is applied. The failure stays in the
four-replicate denominator, with three raw GSM8K cells still running.

Separately, old Opus5 G/w13r04 completed at 75.0569% (1/4 G replicates); it is
recorded in the Round 04 record and not pooled with this study. Overall PTB
validator/judge-clean completions are now 81, with zero in the new study.
Both terminal attempts and the 16 current snapshots have been harvested after
an inspected reconcile preview. See `evidence/2026-09-04-1500/operator-review.md`
for the direct failure evidence, unchanged G ledger, costs and disposition.

## 16:00 UTC operator check

BFCL raw c53r01 (92181) also ended without a model/metrics. A bounded delegated
review, verified against the trace and outer solve diagnostic, reproduces the
GSM8K failure sequence: final waiting reply, normal CLI end_turn, stopped
background waits and absent deliverable. Earlier training OOM was recovered;
it contradicts the general judge's blanket no-OOM statement but is not shown
to be the terminal cause. The exact near-30-minute stop mechanism remains open.
The common raw launcher and runtime are unchanged across the two failed cells.

The BFCL failure used 0.7375 allocated GPU-h, $4.987571 scientist and
$3.83153925 recorded judge cost. Keep both failed raw attempts in their
original four-repeat accounting; no selective retry or frozen-treatment edit
occurs. BFCL raw c53r02/03 (92182/92183) are now running and c53r04 is pending.
The new study still has no valid completed score. Lifecycle reproduction and
any resulting runtime fix would be independent of this frozen study.

Separately, c10r08 in the old Opus5 cohort completes at 73.0857%, bringing the
old control to 8/8 and total PTB/judge-clean completions to 82. Old matched
control/WMA means are 74.9716%/72.3180%; they are not pooled with this study and
do not constitute promotion evidence. Both new terminal attempts are harvested.
Evidence, unchanged baseline ledger, full provenance and the accepted limits
of the delegated diagnosis are in `evidence/2026-09-04-1600/`.

## 16:30 UTC operator check

The remaining three old G cells completed and were harvested after preview;
total PTB/automatic-judge-clean completions are 85. G averages 73.5406% (n4)
but fails four original scope flags and one separately confirmed semantic
held-out-input exposure; see the Round 04 record and `evidence/2026-09-04-1630/`.
No old flags/scorer are changed and G is not promoted.

This is evidence about the old G runtime. Current frozen Opus4.8 input selection
uses bounded explicit exports and does not automatically export session eval
JSONL trees; already contaminated selected text remains a separate limitation.
No current treatment is modified based on an inferred shared failure.

The new study is still zero complete, two known incomplete raw attempts,
12 RUNNING and 46 PENDING. BFCL jobs 92184/92185/92186 started by normal backfill.
All routes remain on nodes 2–3; no submission or cancellation was added. H's
four older cells continue. The existing hourly hook can use the upcoming
six-hour tail threshold; no duplicate analysis window or timer is created.

## 17:30 UTC operator check and independent policy reference

Old H/w14r04 completed at 68.4610%; total PTB/automatic-judge-clean completions
are 86, with WMA scope/manual flags kept separate. The initial Opus4.8 study
still has no valid completed score and two known incomplete raw attempts.
BFCL protocol c54r03 (92187) started through normal backfill. Queue reserve is
45 safe pending, with 16 allocated GPUs and no scheduler dependencies.

The seven-cell Opus5 report was reviewed with corrections: no G promotion,
no unverified perfect-compliance target, no3% gate-cost claim, and no deployed
client split that contradicts actual frozen bytes. Its useful old/new policy
comparison is separately preregistered as S0, four GSM8K repeats on the same
new runtime as S. This adds a reference arm, not a change to any original
receipt; it is not a replacement for failed raw attempts. Exact validation must
return before S0 scientific submission. See the policy-comparison spec/record
and `evidence/2026-09-04-1730/operator-review.md`.
