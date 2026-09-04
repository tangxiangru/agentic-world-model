# Window04 local trace review — 2026-09-04

Status: **CLOSED by the [planner decision](2026-09-04-round-02-window04-decision.md)** after full reads of14 reports,both syntheses,five focused reviewer audits and the planner prefix audit. No promotion or new runtime variant is accepted as written; B/H/J are next scientific priority, with unresolved designs recorded explicitly. This supersedes the [five-cell accumulation checkpoint](2026-09-04-window04-accumulation.md), without counting its five twice. Nine additional clean completions were harvested and pushed in `6bcfd4e`; the prior five are in `7043fa4`. The [later tail harvest](2026-09-04-window04-tail-harvest.md) adds2 clean cells plus1 placement-quarantined result outside this frozen window. D1 withdrawal and planned oldE1 withdrawal add no clean cells.

## Frozen evidence and denominators

[Roster](trace-reviews/window04-local/roster.json) records every job, receipt, manifest, spec, bundle/raw-result path, frozen AWM/PTB identity and official score. [Inputs](trace-reviews/window04-local/inputs.json) hash the prepared facts/timelines. [Collect](trace-reviews/window04-local/collect.csv) is raw bookkeeping, not reviewed cost attribution.

| NEW variant | exact NEW cells | n | official mean |
|---|---|---:|---:|
| session guard |g01r03–g01r08|6|0.7354056103108415|
| protocol-free strict control |c01s01–04,c01s06–07|6|0.7514531210513015|
| protocol v3 strict baseline |p00s01–02|2|0.7736921910538286|

The high baseline mean has only two observations; none of these descriptive means establishes an arm effect. The old guard manifest as a whole is now8/8 clean, mean0.7375852918877938; its g01r01/02 belong to earlier review and are not NEW here. Historical p00r02 is one same-variant calibration input for the baseline reviewer, excluded from all NEW means/counts. The previous strict-guard addendum and P5 remain outside this window.

**New highest clean official score: g01r03/job90649,0.8278999241849886 =1092/1319 (82.79%)**. It is eligible and non-quarantined, with no validator issues or judge flags. This exceeds the earlier overall maximum c00r03=0.7968157695223654 and the earlier protocol maximum g01r02=0.7778620166793025. It is a single-run record, not evidence of average protocol improvement; its trace must be reviewed as critically as the lower-scoring cells.

Jobs90817,90820,90825 were still running at the fixed04:00 trigger and are explicitly excluded from this frozen window. Their outcomes are not needed to begin reviewing the valid completed evidence; later arrivals will be tracked separately. No outcome-based selection of an arbitrary eight was performed: all14 eligible NEW completions at this trigger are included.

## Local reviewer dispatch

[Launch record](trace-reviews/window04-local/launch.json) holds exact full session IDs, PIDs, prompts and source mapping. All use `claude-opus-5[1m] --effort max`, background, plan permission mode with no prompts, and only Read/Grep/Glob/Bash. Actual assigned-file/trace reads were verified in the same host context; this is not merely successful session creation.

| group | NEW cells | session prefix |
|---|---|---|
| guard-a |g01r03,g01r04,g01r05|a1cb8656|
| guard-b |g01r06,g01r07,g01r08|949ad74e|
| control-a |c01s01,c01s02,c01s03|3203e074|
| control-b |c01s04,c01s06,c01s07|56db4d25|
| baseline |p00s01,p00s02; p00r02 calibration only|4be21ecc|

All14 NEW reports and the separate p00r02 calibration note are now saved under `trace-reviews/window04-local/reports/`, with complete unaltered group outputs in `groups/`. Sessions were stopped only after complete report output and idle verification. The planner read every report and the four completed focused audits; [corrections](trace-reviews/window04-local/planner-corrections.md) preserve measurement/causal qualifications. Three additional reviewers supplied control timing, E2 retention timing, frozen-D CPU scope and card/runtime semantics. Frozen D is not releasable as-is; E2's unconditional non-saturation proof is reopened, not declared saturated. Runtime trees and scores are unchanged.

Independent local synthesis **`ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70`**, PID2593119, started05:16:40 UTC with Opus5[1m] max and the same read-only tools. Same-host agents reports busy/working, and actual reads of the brief, meta instructions, roster, corrections and audits were verified. [Synthesis brief](trace-reviews/window04-local/briefs/synthesis.md) requires every NEW report and separate treatment of uncertain evidence. No final candidate decision follows from merely launching synthesis.

**Delivery update:** its complete unaltered output is saved as `synthesis.initial.raw.md`, extracted report as `synthesis.initial.md` (507lines), all read by planner. Parent was idle and then stopped after delivery; it is no longer running. Local analystCLI is now verified2.1.260 (not the frozen scientistCLI2.1.219). A requested same-history correction with explicit flags caused the CLI to create revision copy **`a1e293bb-7b8c-4e8a-ae0b-d305c22d47e3`**, PID2690672; at dispatch only that revision was active, and actual reads of the follow-up brief and new audits were verified. `launch.json` records parent/copy lineage.

**Revision2 delivery:** the copy finished and was stopped after complete output/idle verification. Its full raw output and532-line report are preserved; both syntheses are now fully read and adjudicated in the linked planner decision. No Claude helper remains active for this window; do not restart completed PIDs.

The [follow-up brief](trace-reviews/window04-local/briefs/synthesis-followup.md) requires incorporation of exact control-b pairs, correction of13/14 executed max-n coverage and9 re-locked cards/12 events, and resolution of D2's known false-block/guardrail contradiction. The planner's [g01r03 prefix audit](trace-reviews/window04-local/g01r03-prefix-audit.md) additionally proves the scientist sliced stored sample order instead of declared dataset IDs:126/150 came from the wrong subset (122IDs outside the intended prefix); the aligned comparison is128→127, not126→127. The official82.79 score is unchanged. No first-version proposal is accepted as an experiment merely because it appears in the synthesis.

## Operations while analysis runs

**Current update06:55 UTC:** both D1(91060–91063) and oldE1(91064–91067) were withdrawn as wholly unstarted blocks and harvested; no score or running work was removed.22 actual held remain,0/16 allocated,OWNERSHIP OK. Native reservation still spans11 nodes and no new release authorization has arrived. Historical snapshots below are provenance,not current capacity claims.

At04:04:23 UTC the subqueue had **2/16 owned GPUs allocated**, three registered running jobs and29 actual `PENDING(JobHeldUser)`. The outside-node90820 still causes OWNERSHIP FAIL; the native reservation still covers11 nodes. Fourteen owned GPUs are idle because the operational release gates are not satisfied, not because the planner is waiting for a scientific straggler or for Claude. There were no unknown jobs/name mismatches or runnable pending jobs. No submit/release/cancel was performed.

Original monitor2086813 emitted the archived [ready trigger](trace-reviews/window04-local/trigger.json) at04:00:55 with14 terminal attempts and exited normally. After preserving that event and consuming it for collection, a new hourly monitor **PID2446155** tracks only90817,90820,90825 with threshold3 (the whole remaining set). Its first snapshot at04:06:32 has0/3 terminal; it was verified live. This new monitor is a tail-completion detector, not an assertion that three cells form a new eight-clean window. Log: `/tmp/exp-protocol-tail-monitor-l4_ww232/monitor.log`.

The original D/B/H dispatch ordering is superseded by the closed Window04 decision: D1/E1 are withdrawn and B/H/J are next scientific priority. E2/K remain separate prepared work under their gates. The [operator view](operator-state.md) carries current receipt-backed counts and post-release floor arithmetic.

## Required next steps

The analysis is closed; execute the [planner decision](2026-09-04-round-02-window04-decision.md)'s remaining preparations: J source/site validation and held registration, explicit A clock categories, separate D2/comparator/P4 designs, and baseline mixed-receipt disposition. No promotion or release from this record high. All required report/card reads are complete and reusable lessons are in meta.

The live hourly detector is now PID2612586,watching22 remaining held IDs,threshold6 plus2 buffered clean tail cells; latest06:27:36 tick0/22,next07:27:36. Both earlier ready events were archived. D/E were already excluded,so withdrawal did not reset or alter the watched set. Native restoration/scoped release approval remains a separate unanswered user question.
