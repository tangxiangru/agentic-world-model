# exp-protocol operator state and dependencies

Mutable handoff view; historical receipts/specs remain immutable. Updated2026-09-04 after the03:05 harvest: affected manifests revalidated; live queue03:02:42; monitor03:00:54. Unchanged older cohorts retain the2026-09-03 validation baseline. Scope is only `gangda_exp-protocol-evolve` on `slurm2-a3nodesetondem-[0-1]`.

## Completed, running and held

Current receipt-aware `awm ptb results MANIFEST --json` revalidated the following formal cohorts. `clean` means validator-complete, eligible and no judge flags; Slurm completion alone is not sufficient. Pilots and administrative withdrawals are not in these counts.

| cohort / manifest | clean | other completed | running jobs | held jobs |
|---|---:|---|---|---|
| [v3 baseline16](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3.yaml) |14|0; p00r08/p00r16 incomplete failures|none|none|
| [null control8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-x8.yaml) |7|c00r02 complete with general_anomaly, separate from clean|none|none|
| [null control B8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-b-x8.yaml) |8|0|none|none|
| [old guard8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8.yaml) |6|0|90649,90653 / g01r03,g01r07|none|
| [strict guard8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml) |8|0|none|none|
| [strict control8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8.yaml) |1 (c01s06)|0|90813–90817,90819–90820 / c01s01–05,c01s07–08|none|
| [strict baseline8](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8.yaml) |0|0|90823–90825 / p00s01–03|90826–90830 / p00s04–08|

These seven manifests now have45 validator-complete eligible results, **44 clean and1 flagged**, plus2 failed/incomplete attempts and12 currently running. Five NEW clean cells are accumulating for [Window04](2026-09-04-window04-accumulation.md); no new analysis window has been dispatched yet. The largest clean official score remains c00r03=0.7968157695223654; protocol maximum g01r02=0.7778620166793025. [p00r16 failure](2026-09-03-p00r16-scorer-failure.md) remains unscored; developer results do not fill the missing official metric.

Additional held Round02 blocks:

| block | exact receipt-backed job IDs | next scientific role |
|---|---|---|
| A v2 |91046–91049|second wave|
| B v2 |91050–91053|first wave|
| drift A v2 |91058–91059|first-wave comparator|
| D |91060–91063|first wave|
| old E |91064–91067|remain held until E2 replacement receipt exists; then withdraw whole unstarted block|
| H |91068–91071|first wave|
| drift B |91072–91073|second-wave comparator|

With the five strict-baseline jobs, this is **29 actual PENDING(JobHeldUser)**, zero ordinary runnable pending. C v2 jobs91054–91057 are cancelled and harvested, not counted. Resolve any individual job through `.venv/bin/awm slurm show JOB --json`, then receipt→cell→manifest→spec→result; ownership authority is `/rmeng_data/robtang/slurm-queue/registry.json`.

## Gates and next-wave graph

At03:02:42 UTC only **11/16 owned GPUs were allocated**, leaving five idle;12 running jobs include the outside-node90820. The held buffer is sufficient, but the independent ownership/native-isolation gates still prohibit release. Capacity is not being held idle for a scientific straggler or for local Claude analysis. Resolving90820 and restoring native two-node isolation requires the applicable operator/user authority; running work is not cancelled automatically.

```text
strict guard8 complete and fully reviewed
  └─ observed-no-harm gate PASSED (not promotion or universal compliance)

OWNERSHIP OK + per-job frozen ReqNodeList + native two-node isolation
  └─ NOT satisfied:90820 outside assigned nodes; registered12/16; reservation11 nodes
     └─ no new submissions (including held) or releases now

when operational gates pass, independently of unrelated stragglers:
  D4 + B4 + H4 + drift A2 → first14 cells; replenish/useful held buffer remains≥8
  new E2 held receipt4 → withdraw old E4, only if all old jobs remain unstarted
  replenish with≥4 scientifically valid, validated independent held cells
  A4 + E2 4 + drift B2 → later10 cells, after re-auditing scientific need and held floor
  winners → independent second4 cells → held-out confirmation before promotion
```

The buffer must be checked **after each proposed release/withdrawal**, not only before. Under this snapshot and no other transitions:29−14=15; E's +4−4 replacement leaves15; releasing the later10 would leave5, below8. Registering one justified four-cell block first leaves19, then9. Frozen J/K are possible sources of such a block only after a fresh scientific/operational audit, not automatic releases or filler work. If other jobs leave hold, recompute with live exact IDs; do not use this arithmetic as a release authorization.

Baseline-strict stragglers are not a fabricated dependency for independent screens. Genuine matched-arm/promotion requirements still need their designated evidence. Source: [Round02 current decision](../spec/2026-09-02-exp-protocol-round02-independent-screens.md), section10.

## Prepared but not registered

| candidate | frozen SHA / protocol tree | four-cell manifest |
|---|---|---|
| E2 process wait |`c6f11d8` / `ceb68549`|[E2](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2.yaml)|
| J lock scope |`549e25a` / `7ae08ccf`|[J](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-j-lock-scope-x4.yaml)|
| K deferred comparator |`58a6992` / `ec7d5f2a`|[K](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-k-deferred-comparator-x4.yaml)|

All three have recorded CPU/independent-forward validation and local/full manifest checks; none has a scheduler receipt. They are separate baseline-relative variants; the host shipped tree is guard drift `2f64581`. P1 v1 must not be registered as written; G remains a future direction, not E's replacement. P5 is read-only investigation, not a candidate or queue entry.

## Monitoring and analysis

Hourly monitor PID2086813 is live, tracking the original17-job cohort (now five harvested terminals plus12 running), threshold8 terminal attempts, state `data/ptb/monitor/exp_protocol_goal.json`. Keep its original IDs while it is live so the next three terminal attempts reach the cumulative threshold; do not restart it to wait for eight additional attempts. Scientific NEW clean count is tracked separately in the Window04 roster. Harvest failed/cancelled/timeout spillover attempts as well, preserving quarantine. Do not restart a live monitor on an observation timeout.

Latest monitor tick2026-09-04 03:00:54 UTC:5/17 terminal, all five harvested and validator-clean; process verified live after that tick. The next nominal tick is04:00:54 UTC. This does not revalidate release gates or require a minute-scale Slurm check.

The exact strict guard cohort has been fully reviewed, including seven incremental cells outside Window03; do not double-count it as another eight-new window. Supplemental P5 local Claude session `145e42ee-b904-4829-9380-e4534ccbc7bf` has completed its read-only Opus5[1m] max review and was stopped after delivery. The [planner adjudication](trace-reviews/p5-serving-audit/planner-decision.md) retains the observation but makes no new protocol candidate or GPU repeat. Three original developer Inspect logs were recovered/read from the data volume, with actual1319 counts and logged settings; official per-item evidence remains unresolved across a scratch-persistence boundary. This consumes zero new clean cells. A [prospective retention design](../spec/2026-09-03-ptb-official-eval-evidence-retention.md) scopes the remaining tested timeout/cleanup and opt-in launcher/harvest integration. The [isolated CPU prototype](2026-09-03-official-evidence-prototype.md) now passes159 tests, including the real Inspect→archive success path with two synthetic MockLLM samples, plus three original developer-log format replays. The helper remains unwired; next are timeout/cleanup, actual Inspect sink and launcher/harvest integration. Source/standalone-container tests do not prove those callers. Keep inference logging local and archive afterward; do not change current frozen attempts, the PTB pin or the evaluation contract.
