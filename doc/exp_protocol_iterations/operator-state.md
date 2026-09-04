# exp-protocol operator state and dependencies

Mutable handoff view; historical receipts/specs remain immutable. Updated2026-09-04 after Window04 adjudication and D/E1 withdrawal/harvest; live queue06:54:54, reservation06:55:49. Unchanged older cohorts retain the2026-09-03 validation baseline. Scope is only `gangda_exp-protocol-evolve` on `slurm2-a3nodesetondem-[0-1]`.

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

The original seven cohort manifests have57 validator-complete results:56 eligible (55 clean,1 flagged) and1 placement-quarantined, plus2 failed/incomplete attempts and0 running. New control repair91965 contributes no result; if eligible it completes the explicitly mapped c01s01–07+c02s01 strict8, without relabelling90820. [Window04](2026-09-04-round-02-window04-decision.md) is CLOSED at14 NEW clean cells; two later clean tail cells remain buffered separately. Highest clean official score remains g01r03=0.8278999241849886(1092/1319,82.79%); no arm effect or promotion is established. [p00r16](2026-09-03-p00r16-scorer-failure.md) remains unscored. Administrative D/E1 withdrawals add zero clean results.

Additional held Round02 blocks:

| block | exact receipt-backed job IDs | next scientific role |
|---|---|---|
| A v2 |91046–91049|second wave|
| B v2 |91050–91053|first wave|
| drift A v2 |91058–91059|first-wave comparator|
| H |91068–91071|first wave|
| drift B |91072–91073|second-wave comparator|

With the five strict-baseline jobs and control repair91965, this is **22 actual PENDING(JobHeldUser)**, zero ordinary runnable pending. C v2 91054–91057, D v1 91060–91063 and E v1 91064–91067 are cancelled and harvested, not counted. All D/E cells were wholly unstarted; no scores or scientist failure trajectories were produced. Window04 explicitly superseded E1's replacement-first timing because22 other useful held cells suffice. E2 remains unregistered with retention unresolved. Resolve jobs with `.venv/bin/awm slurm show JOB --json`; ownership authority remains `/rmeng_data/robtang/slurm-queue/registry.json`.

## Gates and next-wave graph

At06:54:54 UTC **0/16 owned GPUs were allocated**,0 jobs running,and OWNERSHIP OK. Reservation recheck06:55:49 still shows11 nodes; no native-isolation restoration or new release authorization has been received. Capacity is not idle for a scientific straggler or Claude. D1/E1 are withdrawn;22 held remain behind the operational gate. Control repair91965 is held,not released. No running work was cancelled.

```text
strict guard8 complete and fully reviewed
  └─ observed-no-harm gate PASSED (not promotion or universal compliance)

OWNERSHIP OK + per-job frozen ReqNodeList + native two-node isolation
  └─ NOT satisfied: OWNERSHIP OK, registered0/16, but reservation11 nodes
     └─ no releases; re-audit documented per-manifest held-registration gates

when operational gates pass, independently of unrelated stragglers:
  old D1 block withdrawn; D2 requires a new accepted design/test/manifest before any queue entry
  select up to3 independently ready candidates per wave; prior D1/B/H order is obsolete
  old E1 withdrawn; E2 registration requires its separate unresolved scientific gate
  replenish with≥4 scientifically valid, validated independent held cells
  B/H/J are next scientific priority; A/K/P4 and revised D/E need their recorded remaining preparation
  winners → independent second4 cells → held-out confirmation before promotion
```

Check the useful held floor **after** every proposed release/withdrawal.22 actual held remain;D/E withdrawal did not reduce their already-excluded useful count. Releasing14 valid cells leaves8;adding control repair would leave7,so replenish first. Window04 advances frozenJ to fresh held-registration preparation; J4 would raise the pool to26 and leave11 after15 releases. K remains separate under its gates. Recompute exact IDs; a checked manifest is not a held receipt. This is not release authorization.

Baseline-strict stragglers are not a fabricated dependency for independent screens. Genuine matched-arm/promotion requirements still need their designated evidence. The remaining5 baseline holds share a receipt with3 completed jobs: [CPU boundary audit](2026-09-04-mixed-receipt-release-boundary.md) confirms current `release_held` refuses mixed state even after mocked valid native gates. They require explicit receipt-backed state handling before selecting them for release; do not rewrite old job membership or release ad hoc. All-held independent receipts need not wait for that disposition. Source: [Round02 current decision](../spec/2026-09-02-exp-protocol-round02-independent-screens.md), sections10–11.

## Prepared but not registered

| candidate | frozen SHA / protocol tree | four-cell manifest |
|---|---|---|
| E2 process wait; non-saturation proof reopened |`c6f11d8` / `ceb68549`|[E2](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2.yaml)|
| J lock scope |`549e25a` / `7ae08ccf`|[J](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-j-lock-scope-x4.yaml)|
| K deferred comparator |`58a6992` / `ec7d5f2a`|[K](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-k-deferred-comparator-x4.yaml)|

All three have recorded CPU/independent-forward validation and local/full manifest checks; none has a scheduler receipt. They are separate baseline-relative variants; the host shipped tree is guard drift `2f64581`. P1 v1 must not be registered as written; G remains a future direction, not E's replacement. P5 is read-only investigation, not a candidate or queue entry.

## Monitoring and analysis

Historical detectors:2086813 completed14/17 at04:00:55;2446155 completed3/3 at05:06:32. Both events were archived and outcomes harvested. Neither is a live detector now; do not restart a completed PID.

**Current live monitor PID2612586** watches the22 remaining receipt-backed held IDs, threshold6 plus2 buffered clean tail cells; latest tick06:27:36 is0/22 terminal,next07:27:36. Log `/tmp/exp-protocol-held-monitor-expanded-wboe41dy/monitor.log`; exact settings/history in Window04 `launch.json`. PID2579442 was deliberately replaced to add91965 while retaining all21 previous IDs; this was not a timeout/counter reset. D/E were already excluded,so their withdrawal changes no watched IDs. Terminals still require harvest/validator review.

**Window04 is CLOSED by [planner adjudication](2026-09-04-round-02-window04-decision.md)** after full reads of14 NEW reports,five focused audits,both syntheses(507/532lines) and the planner prefix audit. Both Claude sessions completed/stopped;do not duplicate them. Keep guard baseline,no promotion,no new helper proposal accepted as written. B/H/J are next scientific priority; J's frozen screen advances to fresh source/site checks for held registration. D2 predicate and E2 retention remain unresolved; SE-derived hard count gating is rejected; comparator binding/P4 need separate designs; A needs missing-endpoint/clock categories before release. Original reports remain unaltered.

Concurrent worktree note: `doc/exp_protocol_iterations/analysis-2026-09-04-user-review/` is not part of this operator's changes. Preserve it and exclude it from operator commits. Use a genuinely clean source-frozen execution checkout if needed for new registration; never delete,hide or commit unrelated files just to satisfy the clean-tree gate. Independent source/design preparation can continue without release authority.

The exact strict guard cohort has been fully reviewed, including seven incremental cells outside Window03; do not double-count it as another eight-new window. Supplemental P5 local Claude session `145e42ee-b904-4829-9380-e4534ccbc7bf` has completed its read-only Opus5[1m] max review and was stopped after delivery. The [planner adjudication](trace-reviews/p5-serving-audit/planner-decision.md) retains the observation but makes no new protocol candidate or GPU repeat. Three original developer Inspect logs were recovered/read from the data volume, with actual1319 counts and logged settings; official per-item evidence remains unresolved across a scratch-persistence boundary. This consumes zero new clean cells. A [prospective retention design](../spec/2026-09-03-ptb-official-eval-evidence-retention.md) scopes the remaining tested timeout/cleanup and opt-in launcher/harvest integration. The [isolated CPU prototype](2026-09-03-official-evidence-prototype.md) now passes159 tests, including the real Inspect→archive success path with two synthetic MockLLM samples, plus three original developer-log format replays. The helper remains unwired; next are timeout/cleanup, actual Inspect sink and launcher/harvest integration. Source/standalone-container tests do not prove those callers. Keep inference logging local and archive afterward; do not change current frozen attempts, the PTB pin or the evaluation contract.
