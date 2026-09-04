# Window04 local trace review — 2026-09-04

Status: **14 NEW receipt-backed validator-clean cells frozen; five local Opus max reviewers running; synthesis and planner adjudication pending**. This supersedes the [five-cell accumulation checkpoint](2026-09-04-window04-accumulation.md), without counting its five twice. Nine additional clean completions were harvested and pushed in `6bcfd4e`; the prior five are in `7043fa4`.

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

Reviewers emit one report per NEW cell in final output, plus a separately marked calibration note where assigned. They must use original named raw-result directories when the git bundle omitted large logs; prepared regex/size-based markers are only locators. No reviewer may write files, change jobs/git, execute a model/evaluator or make a queue/promotion decision. The planner saves reports, then starts a separate independent Opus max synthesis over the whole window. No synthesis has been started yet, no reports are declared complete, and no hypotheses or candidate changes are accepted by this dispatch record.

## Operations while analysis runs

At04:04:23 UTC the subqueue had **2/16 owned GPUs allocated**, three registered running jobs and29 actual `PENDING(JobHeldUser)`. The outside-node90820 still causes OWNERSHIP FAIL; the native reservation still covers11 nodes. Fourteen owned GPUs are idle because the operational release gates are not satisfied, not because the planner is waiting for a scientific straggler or for Claude. There were no unknown jobs/name mismatches or runnable pending jobs. No submit/release/cancel was performed.

Original monitor2086813 emitted the archived [ready trigger](trace-reviews/window04-local/trigger.json) at04:00:55 with14 terminal attempts and exited normally. After preserving that event and consuming it for collection, a new hourly monitor **PID2446155** tracks only90817,90820,90825 with threshold3 (the whole remaining set). Its first snapshot at04:06:32 has0/3 terminal; it was verified live. This new monitor is a tail-completion detector, not an assertion that three cells form a new eight-clean window. Log: `/tmp/exp-protocol-tail-monitor-l4_ww232/monitor.log`.

D/B/H plus drift A remain independently specified first-wave work; E2/J/K remain unregistered and old E remains held under the prior replacement rule. Native isolation still requires operator/user coordination, and the running90820 is not cancelled automatically. On any ownership/isolation change, reassess which submissions/releases are actually safe rather than treating this timestamped state as permanent. Held-floor arithmetic remains in the [operator view](operator-state.md).

## Required next steps

Receive and read every NEW report; resolve cross-cell ambiguities with additional focused reviewers when needed. Then invoke a separate synthesis, personally inspect best/worst and three cards per variant, audit every held block, and record accepted/rejected proposals with reusable knowledge in meta. Do not promote from the record high or pool calibration into NEW counts. Keep all current frozen scientist trees, evaluation contracts and queue releases unchanged until their separate scientific/operational gates are satisfied.
