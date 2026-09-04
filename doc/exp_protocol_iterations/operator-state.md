# exp-protocol operator state and dependencies

Mutable handoff view; historical receipts/specs remain immutable. Updated2026-09-04 after the05:02 tail harvest and05:25 held control repair; live queue05:26:37, receipt node/hold verification05:27:36. Unchanged older cohorts retain the2026-09-03 validation baseline. Scope is only `gangda_exp-protocol-evolve` on `slurm2-a3nodesetondem-[0-1]`.

## Completed, running and held

Current receipt-aware `awm ptb results MANIFEST --json` revalidated the following formal cohorts. `clean` means validator-complete, eligible and no judge flags; Slurm completion alone is not sufficient. Pilots and administrative withdrawals are not in these counts.

| cohort / manifest | clean | other completed | running jobs | held jobs |
|---|---:|---|---|---|
| [v3 baseline16](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3.yaml) |14|0; p00r08/p00r16 incomplete failures|none|none|
| [null control8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-x8.yaml) |7|c00r02 complete with general_anomaly, separate from clean|none|none|
| [null control B8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-b-x8.yaml) |8|0|none|none|
| [old guard8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8.yaml) |8|0|none|none|
| [strict guard8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml) |8|0|none|none|
| [strict control8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml) |7 (c01s01–07)|c01s08/90820 complete but placement-quarantined|none|none|
| [strict baseline8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8.yaml) |3 (p00s01–03)|0|none|90826–90830 / p00s04–08|
| [strict control repair1](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-tail-x1.yaml) |0|no result; independent eighth-repeat role, not a new variant|none|91965 / c02s01|

The original seven cohort manifests have57 validator-complete results:56 eligible (**55 clean and1 flagged**) and1 placement-quarantined, plus2 failed/incomplete attempts and0 currently running. The new repair manifest contributes zero results; if eligible, c02s01 will complete the explicit c01s01–07+c02s01 strict8 cohort, without relabelling90820 or changing old manifest aggregates. [Window04](2026-09-04-trace-review-window04-local.md) remains frozen at14 NEW clean cells; all five local Opus max reviewers delivered, and synthesis/adjudication is still pending. The [tail harvest](2026-09-04-window04-tail-harvest.md) adds2 later clean cells, not a new eight-cell window. **Highest clean official score remains protocol g01r03=0.8278999241849886 (1092/1319,82.79%)**; control maximum remains c00r03=0.7968157695223654. Validator/judge clean does not establish universal card compliance or a variant effect; full trace review is ongoing. [p00r16 failure](2026-09-03-p00r16-scorer-failure.md) remains unscored; developer results do not fill the missing official metric.

Additional held Round02 blocks:

| block | exact receipt-backed job IDs | next scientific role |
|---|---|---|
| A v2 |91046–91049|second wave|
| B v2 |91050–91053|first wave|
| drift A v2 |91058–91059|first-wave comparator|
| D |91060–91063|do not release pending CPU audit of false blocks on non-saving cards|
| old E |91064–91067|remain held until E2 replacement receipt exists; then withdraw whole unstarted block|
| H |91068–91071|first wave|
| drift B |91072–91073|second-wave comparator|

With the five strict-baseline jobs and control repair91965, this is **30 actual PENDING(JobHeldUser)**, zero ordinary runnable pending. Excluding D4/staleE4 from scientifically usable work leaves22 held; those eight questionable cells must not inflate the post-release useful floor. C v2 jobs91054–91057 are cancelled and harvested, not counted. Resolve any individual job through `.venv/bin/awm slurm show JOB --json`, then receipt→cell→manifest→spec→result; ownership authority is `/rmeng_data/robtang/slurm-queue/registry.json`.

## Gates and next-wave graph

At05:26:37 UTC **0/16 owned GPUs were allocated** and no jobs were running.90820 ended naturally and OWNERSHIP is now OK. The reservation still covers11 nodes, so native two-node isolation and the independent release gate remain unsatisfied. Capacity is not being held idle for a scientific straggler or for local Claude analysis. Restoring isolation or an explicit per-receipt exception requires applicable operator/user authority. D4 is under scientific scope audit and oldE4 awaits replacement; even excluding both,22 held cells remain behind the operational gate. The independently planned control repair was safely registered held under ownership OK, not released.

```text
strict guard8 complete and fully reviewed
  └─ observed-no-harm gate PASSED (not promotion or universal compliance)

OWNERSHIP OK + per-job frozen ReqNodeList + native two-node isolation
  └─ NOT satisfied: OWNERSHIP OK, registered0/16, but reservation11 nodes
     └─ no releases; re-audit documented per-manifest held-registration gates

when operational gates pass, independently of unrelated stragglers:
  D4 scope audit must resolve before D is treated as releasable
  prior first-wave plan D4 + B4 + H4 + drift A2 is provisional, not an instruction to release
  new E2 held receipt4 → withdraw old E4, only if all old jobs remain unstarted
  replenish with≥4 scientifically valid, validated independent held cells
  A4 + E2 4 + drift B2 → later10 cells, after re-auditing scientific need and held floor
  winners → independent second4 cells → held-out confirmation before promotion
```

The buffer must be checked **after each proposed release/withdrawal**, not only before. Superseding the earlier29−14 arithmetic:30 actual held minus8 D/staleE under review leaves22 usable. Releasing14 independently valid cells would leave8 usable; adding the control repair to that same release would leave7, so replenish first or change the scientifically justified wave. Do not count a known problematic block merely because Slurm still says JobHeldUser. Frozen J/K are possible sources only after a fresh scientific/operational audit, not automatic releases or filler. If other jobs leave hold, recompute exact IDs. This is not release authorization.

Baseline-strict stragglers are not a fabricated dependency for independent screens. Genuine matched-arm/promotion requirements still need their designated evidence. The remaining5 baseline holds share a receipt with3 completed jobs: [CPU boundary audit](2026-09-04-mixed-receipt-release-boundary.md) confirms current `release_held` refuses mixed state even after mocked valid native gates. They require explicit receipt-backed state handling before selecting them for release; do not rewrite old job membership or release ad hoc. All-held independent receipts need not wait for that disposition. Source: [Round02 current decision](../spec/2026-09-02-exp-protocol-round02-independent-screens.md), sections10–11.

## Prepared but not registered

| candidate | frozen SHA / protocol tree | four-cell manifest |
|---|---|---|
| E2 process wait; non-saturation proof reopened |`c6f11d8` / `ceb68549`|[E2](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2.yaml)|
| J lock scope |`549e25a` / `7ae08ccf`|[J](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-j-lock-scope-x4.yaml)|
| K deferred comparator |`58a6992` / `ec7d5f2a`|[K](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-k-deferred-comparator-x4.yaml)|

All three have recorded CPU/independent-forward validation and local/full manifest checks; none has a scheduler receipt. They are separate baseline-relative variants; the host shipped tree is guard drift `2f64581`. P1 v1 must not be registered as written; G remains a future direction, not E's replacement. P5 is read-only investigation, not a candidate or queue entry.

## Monitoring and analysis

Original monitor2086813 completed normally at04:00:55 with14 terminal attempts; its ready event is preserved in Window04 `trigger.json`, and all14 are harvested/validator-clean. New hourly monitor **PID2446155** tracks only90817,90820,90825 with threshold3, the remaining original-cohort attempts. State remains `data/ptb/monitor/exp_protocol_goal.json`; log is `/tmp/exp-protocol-tail-monitor-l4_ww232/monitor.log`. This tail detector does not make three arrivals a new eight-clean window. Harvest all outcomes and track later clean evidence separately from the frozen14. Do not restart a live monitor on an observation timeout.

Tail detector2446155 completed3/3 normally at05:06:32; event archived and all outcomes harvested. The subsequent held detector2579442 was deliberately replaced at05:27:36 to add new receipt91965, preserving all21 original watched IDs and threshold6 plus2 buffered clean cells. **Current live monitor PID2612586** watches22 IDs, first tick0/22 terminal, next06:27:36; log `/tmp/exp-protocol-held-monitor-expanded-wboe41dy/monitor.log`. The prior state is archived in `held-monitor-before-expansion.json`; exact current and previous settings are in Window04 `launch.json`. Old PID absence and replacement liveness were verified. This was watch-set expansion, not a timeout/partial-harvest counter reset, and does not release jobs or equate terminals with clean cells.

All14 NEW reports and five focused audit reports were read by planner. Independent Opus max synthesis **ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70**, PID2593119, remains busy/working with actual report/source reads verified; do not duplicate it. E2's unconditional non-saturation proof is reopened; D v1 is not releasable as-is. The completed structural paired-count audit for c01s04/c01s07 corrects material regex-pairing errors, not merely one missing sample; all34 logs were independently replayed, exact inventory/pairs match. See Window04 `paired-counts/` and the later addendum in `planner-corrections.md`. Synthesis started before this audit, so verify/correct its paired tables after delivery before adjudicating candidates.

The exact strict guard cohort has been fully reviewed, including seven incremental cells outside Window03; do not double-count it as another eight-new window. Supplemental P5 local Claude session `145e42ee-b904-4829-9380-e4534ccbc7bf` has completed its read-only Opus5[1m] max review and was stopped after delivery. The [planner adjudication](trace-reviews/p5-serving-audit/planner-decision.md) retains the observation but makes no new protocol candidate or GPU repeat. Three original developer Inspect logs were recovered/read from the data volume, with actual1319 counts and logged settings; official per-item evidence remains unresolved across a scratch-persistence boundary. This consumes zero new clean cells. A [prospective retention design](../spec/2026-09-03-ptb-official-eval-evidence-retention.md) scopes the remaining tested timeout/cleanup and opt-in launcher/harvest integration. The [isolated CPU prototype](2026-09-03-official-evidence-prototype.md) now passes159 tests, including the real Inspect→archive success path with two synthetic MockLLM samples, plus three original developer-log format replays. The helper remains unwired; next are timeout/cleanup, actual Inspect sink and launcher/harvest integration. Source/standalone-container tests do not prove those callers. Keep inference logging local and archive afterward; do not change current frozen attempts, the PTB pin or the evaluation contract.
