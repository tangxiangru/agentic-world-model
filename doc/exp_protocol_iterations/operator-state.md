# exp-protocol operator state and dependencies

Mutable handoff view; historical receipts/specs remain immutable. Updated2026-09-04 after E8 combined CPU validation; live queue11:02:55, reservation10:15:12, monitor process11:12:00 / tick10:27:36. Unchanged older cohorts retain their previous validation baseline. Scope is only `gangda_exp-protocol-evolve` on `slurm2-a3nodesetondem-[0-1]`.

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
| A v2 |91046–91049|held; re-audit under bundle policy,not automatically funded|
| B v2 |91050–91053|component for E; standalone funding superseded|
| drift A v2 |91058–91059|no automatic drift top-up; replacement/withdrawal disposition needed|
| H |91068–91071|component for E; standalone funding superseded|
| drift B |91072–91073|no automatic drift top-up; replacement/withdrawal disposition needed|

With the five strict-baseline jobs and control repair91965, this is **22 actual PENDING(JobHeldUser)**, zero ordinary runnable pending. This physical count does **not** certify22 scientifically necessary cells under the new policy; controls/baseline top-ups and standalone micro-screens require renewed disposition before release. The useful/releasable floor is not proven merely by keeping them held. C v2 91054–91057, D v1 91060–91063 and E v1 91064–91067 are cancelled and harvested, not counted. All D/E cells were wholly unstarted; no scores or scientist failure trajectories were produced. E2 remains an unregistered component with its old retention claim unresolved. Resolve jobs with `.venv/bin/awm slurm show JOB --json`; ownership authority remains `/rmeng_data/robtang/slurm-queue/registry.json`.

## Gates and next-wave graph

At11:02:55 UTC **0/16 owned GPUs were allocated**,0 jobs running,and OWNERSHIP OK; all22 remaining jobs were still JobHeldUser. The10:15:12 reservation check still showed11 nodes; no native-isolation restoration or new release authorization has been received. A subsequent read-only inventory found no already-available exact two-node reservation for this line; registry scope still names the broad reservation. No infrastructure was changed. Capacity is not idle for a scientific straggler or Claude. D1/E1 are withdrawn;22 physical holds remain. Their scientific replacement audit and native release gate are separate. Control repair91965 is held,not automatically funded under the new policy. No running Slurm work was cancelled.

```text
strict guard8 complete and fully reviewed
  └─ observed-no-harm gate PASSED (not promotion or universal compliance)

OWNERSHIP OK + per-job frozen ReqNodeList + native two-node isolation
  └─ NOT satisfied: OWNERSHIP OK, registered0/16, but reservation11 nodes
     └─ no releases; re-audit documented per-manifest held-registration gates

new policy, independently of unrelated stragglers and native repair:
  construct/test integrated E and E+L/E+P packages; test L×P before funding E+L+P
  freeze complete six-path identities and new2-cell discovery manifests
  register genuinely useful held replacements; audit old whole blocks by exact receipt IDs
  release only after scientific,ownership,frozen-node,native-isolation and useful-floor gates
  review completed declared6/8-cell discovery; add≤2 per config only for a specific decision
  promotion requires predeclared quality tolerance and untouched held-out confirmation
```

Check the useful held floor **after** every proposed release/withdrawal. Old22-minus-release arithmetic is insufficient now that scientific priorities changed. Eight new discovery cells alone cannot supply16 running plus8 held,or even be released as a complete wave while preserving an8-cell-only useful buffer. Prepare genuinely independent downstream work and record the missing inventory; do not invent repetitions or count checked manifests as held receipts. The [bundle specification](../spec/2026-09-04-exp-protocol-bundle-discovery.md) fixes component scope and this transition gate. StandaloneJ4 is not a buffer top-up.

Baseline-strict stragglers are not a fabricated dependency for independent screens. Genuine matched-arm/promotion requirements still need their designated evidence. The remaining5 baseline holds share a receipt with3 completed jobs: [CPU boundary audit](2026-09-04-mixed-receipt-release-boundary.md) confirms current `release_held` refuses mixed state even after mocked valid native gates. They require explicit receipt-backed state handling before selecting them for release; do not rewrite old job membership or release ad hoc. All-held independent receipts need not wait for that disposition. Source: [Round02 current decision](../spec/2026-09-02-exp-protocol-round02-independent-screens.md), sections10–11.

## Prepared but not registered

**Policy reconciliation resolved:** the explicit2026-09-04 human direction in meta supersedes the saved objective's single-item wording. The earlier duplicate clarification request is not a planning blocker. J's clean clone passed source/site checks,but no J receipt existed and no J job was submitted; its unsubmitted queue entry was removed. Do not use the private763701c snapshot to execute that obsolete standalone plan. E/H/J/K and corrected B/D mechanisms feed complete bundles,with new identities and budgets. Existing22 physical holds and the hourly monitor are unchanged; review current scientific need before release.

| candidate | frozen SHA / protocol tree | four-cell manifest |
|---|---|---|
| E2 process wait; non-saturation proof reopened |`c6f11d8` / `ceb68549`|[E2](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-e-wait-on-process-x4-v2.yaml)|
| J lock scope |`549e25a` / `7ae08ccf`|[J](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-j-lock-scope-x4.yaml)|
| K deferred comparator |`58a6992` / `ec7d5f2a`|[K](../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r02-k-deferred-comparator-x4.yaml)|

All three have recorded CPU/independent-forward validation and local/full manifest checks; none has a scheduler receipt. They are historical separate baseline-relative variants,now components rather than automatic standalone funding; the host shipped tree is guard drift `2f64581`. P1 v1 must not be registered as written; G remains a future direction, not E's replacement. P5 is read-only investigation, not a candidate or queue entry.

**Bundle construction:** latest committed checkpoint is pushed5ec81dc on `codex/exp-protocol-bundles-20260904`,in `/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`; PTB remainsdcf5da0. H/J/K/B guidance,E4 saves,E5 execution records,E6 raw-first sampling and E7 prepared-token/checked-consumer paths are implemented. The worktree contains E8 builder-owned uncommitted files; do not stage or discard them blindly. E8's committed `doc/exp_protocol_iterations/2026-09-04-serving-export-design.md` has been implemented by `/root/bundle_save_contract`,owning only serving_artifacts.py and its dedicated test/guide; those files are frozen pending independent forward review. No complete E variant/manifests are frozen yet; L/P remain to construct. Keep the operator's runtime tree unchanged and preserve concurrent user meta/analysis drafts.

**Validation/evidence:** earlier original-env protocol/sandbox/PTB regression excluding E8 had415 passed/73 expected dependency skips. Main independently ran pinned E6(32)+E7(76)+schema/preflight(80):188 passed in38.81s. Earlier E4's51 pinned tests and E5's31 Python3.10 tests remain applicable. E6/E7 independent reviewers completed public-guide-selected CPU workflows; main read their full reports,verified E6's only-inert-call interception and independently reverified both E7 successful receipts. Builder's10k-row/3,492,790-token E7 measurement is15.96s prepare+verify,2.36s preflight,2.41s loader,.027s first batch—not a full-corpus or GPU prediction. Generic quick_validate still rejects the established underscore name; preserve it. Full histories/counts and proof limits are in the candidate's `2026-09-04-bundle-construction.md`.

**E8 current checkpoint:** the builder completed and froze `serving_artifacts.py`, its public guide and56 tests for independent review. Main read the full code/tests/guide and independently ran E8+E4/save_trainer+E5 in the pinned CPU runtime:138 passed in14.29s. Full original-env protocol/sandbox/PTB regression including E8 and its skill routing:464 passed,80 expected dependency skips in21.60s. These supersede the earlier exclusion-only aggregate above, not native coverage by themselves. Native tests include actual metadata loaders, tiny sampled/greedy single/indexed saves, Gemma3 processor assets and an E5-produced namespace. They perform no inference or scientific validation. `/root/window04_card_semantics` is conducting the public-guide forward pass in its own CPU fixtures; E8 is not yet independently accepted/frozen as E. Main's L observation design is drafted; `/root/window04_control_timing` is drafting P's real-Inspect/retention design. Neither L/P nor new manifests exist as implementations. Do not stage reviewer/builder files blindly.

**Persistence:** lightweight byte-identical E6/E7 review files and per-file archive indices are committed under `doc/exp_protocol_iterations/trace-reviews/bundle-e6-forward/` and `bundle-e7-forward/` in the candidate branch. Full byte-compared copies,including E7's123MB of tokenizer snapshots,are retained at `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/review-artifacts/e6-forward-sampling.ntRxXE` and `.../e7-forward-review.zLMH5o`; original temporary sessions were not deleted. The fixed CPU runtime remains `/tmp/exp-protocol-save-runtime.JEZlHo` (~14GB); exact reproduction is in save-safety.md. Reusable lessons are in committed meta/bundle_validation.md. Reviews add zero validator-clean cells; no new GPU jobs,receipts,releases or standalone micro-screen top-ups occurred.

## Monitoring and analysis

Historical detectors:2086813 completed14/17 at04:00:55;2446155 completed3/3 at05:06:32. Both events were archived and outcomes harvested. Neither is a live detector now; do not restart a completed PID.

**Current live monitor PID2612586**,verified by `ps` at11:12:00,watches the22 remaining receipt-backed held IDs, threshold6 plus2 buffered clean tail cells; latest tick10:27:36 is0/22 terminal,next11:27:36. Log `/tmp/exp-protocol-held-monitor-expanded-wboe41dy/monitor.log`; exact settings/history in Window04 `launch.json`. PID2579442 was deliberately replaced to add91965 while retaining all21 previous IDs; this was not a timeout/counter reset. D/E were already excluded,so their withdrawal changes no watched IDs. Terminals still require harvest/validator review. The SessionStart event helper now recognizes either8 NEW clean cells or a scientifically complete predeclared block under current meta; it never equates its terminal threshold with either condition. The live detector was not restarted.

**Window04 is CLOSED by [planner adjudication](2026-09-04-round-02-window04-decision.md)** after full reads of14 NEW reports,five focused audits,both syntheses(507/532lines) and the planner prefix audit. Both Claude sessions completed/stopped;do not duplicate them. Keep guard baseline,no promotion,no helper proposal accepted as written. Its old B/H/J standalone priority is superseded by bundle construction. D's replacement actual-save interface is implemented as candidate E4 and has native combined CPU tests; the complete E package still awaits export forward acceptance. E2's historical retention proof is unresolved; SE-derived hard count gating remains rejected. Original reports remain unaltered.

Concurrent worktree note: `doc/exp_protocol_iterations/analysis-2026-09-04-user-review/` is not part of this operator's changes. Preserve it and exclude it from operator commits. Use a genuinely clean source-frozen execution checkout if needed for new registration; never delete,hide or commit unrelated files just to satisfy the clean-tree gate. Independent source/design preparation can continue without release authority.

The exact strict guard cohort has been fully reviewed, including seven incremental cells outside Window03; do not double-count it as another eight-new window. Supplemental P5 local Claude session `145e42ee-b904-4829-9380-e4534ccbc7bf` has completed its read-only Opus5[1m] max review and was stopped after delivery. The [planner adjudication](trace-reviews/p5-serving-audit/planner-decision.md) retains the observation but makes no new protocol candidate or GPU repeat. Three original developer Inspect logs were recovered/read from the data volume, with actual1319 counts and logged settings; official per-item evidence remains unresolved across a scratch-persistence boundary. This consumes zero new clean cells. A [prospective retention design](../spec/2026-09-03-ptb-official-eval-evidence-retention.md) scopes the remaining tested timeout/cleanup and opt-in launcher/harvest integration. The [isolated CPU prototype](2026-09-03-official-evidence-prototype.md) now passes159 tests, including the real Inspect→archive success path with two synthetic MockLLM samples, plus three original developer-log format replays. The helper remains unwired; next are timeout/cleanup, actual Inspect sink and launcher/harvest integration. Source/standalone-container tests do not prove those callers. Keep inference logging local and archive afterward; do not change current frozen attempts, the PTB pin or the evaluation contract.
