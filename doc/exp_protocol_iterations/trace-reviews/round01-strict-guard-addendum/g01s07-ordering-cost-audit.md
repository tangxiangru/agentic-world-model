# g01s07 ordering and duplicate-cost audit — 2026-09-03

Read-only additional reviewer: `/root/strict_guard_audit`. Planner checked the failed-lock/launch sequence and the first surfaced crash/diagnostic excerpts. Original Opus report remains unchanged. This audit does not decide full-cohort safety.

Provenance: job **90797**, cell g01s07, [immutable receipt](../../../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/formal-2026-09-02T204221.237369+0000.json), [manifest](../../../../experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml), [spec](../../../spec/2026-09-02-exp-protocol-round01-session-guard.md). All trace line numbers below are **uncompressed** lines of `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s07/solve_parsed.txt.gz`. Raw result path is recorded in that bundle's `status.json` and `runtime_provenance.json`.

## Two evaluations launched before lock

| card | evidence sequence | finding |
|---|---|---|
| exp-05 | card Write 16:21:48 (L6567–6569); missing comparator FAIL 16:21:52 (L6571–6579); `evaluate.py --model-path ckpts/exp-02/final --limit 500` launch 16:21:58 (L6589–6594); comparator output returned 16:36:59 (L6622–6627); lock 16:37:04 (L6633–6642); candidate eval 16:37:08 (L6644–6648) | lock occurred 15m06s after comparator launch and after seeing its output |
| exp-07 | comparator already returned 17:55:00 (L7608–7616); `evaluate.py --model-path ckpts/exp-02/final --limit -1` launch 17:55:14, followed by card-slot creation in the same call (L7625–7631); filled card Write 17:55:47 (L7644–7649/L7771–7773); lock 17:55:50 (L7775–7784) | deliberate 36-second pre-lock launch; no comparator blockage |

exp-05's command produces the same `eval/exp-02_dev500.json` path that its comparator prerequisite requires (L6517/L6538). That is a real dependency conflict for an unchanged clean-first lock. However, runtime instructions explicitly describe a reasoned `--override` escape hatch (L78–82), which the scientist did not use. Correct attribution: **check-induced ordering conflict resolved by a scientist-selected pre-lock launch**, not an unavoidable forced violation. Its persisted lock has `overrides: {}` and `preflight.fail: 0`; that snapshot does not preserve the earlier failed attempt.

exp-07's result notes explicitly admit the rule-1 violation and cite remaining-time pressure (`task/memory/cards/exp-07.yaml:111`). The report's training-only 3/3 lock-before-launch count is compatible with both evaluation-ordering violations; it does not prove global protocol compliance. Neither ordering violation alone establishes loss of live work at session end.

## One failed attempt recorded on two cards

`task/memory/cards/exp-03.yaml:24` and `exp-04.yaml:24` each attribute `cost_h: 2.0` to the **same exp-03 checkpoint-save failure**. exp-04 subsequently completes; its entry is retrospective, not a second crash.

| disjoint component | trace evidence |
|---|---|
| failed training loop | step2600/5144, elapsed 1:09:55 = **1.165 h** (L5623, L5648–5687); including initialization after 12:36:25 launch, approximately **1.194 h** failed-attempt wall time |
| post-exit discovery lag | approximately **0.66 h**; first failure output at **14:27:49** (L5617–5623), active investigation 14:28:05 (L5637–5645) |
| diagnosis/repair/smoke/re-card/relaunch | **14:27:49→14:33:20 = 0.092 h**: edits 14:28:50/56, smoke 14:29:01→14:31:39 (54.619 s actual training), relock/close 14:32:15/16, exp-04 launch14:33:20 (L5715–5813, L5920–5931, L6129–6134) |

The report's 13:48:04 crash time is an **estimate**, not an observed exit stamp. The source bar at 13:37:28 shows step2211/59:19 and step2212/59:21 (L5514–5515). Independent `task/system_monitor.log:3290–3308` bounds GPU release to 13:47:21–13:48:21, supporting an approximate 13:48 crash and **0.658–0.674 h** until the first surfaced failure. GPU release alone is not the death proof; it is corroborated here by the retained exception, stopped progress and subsequent zero trainer-process check.

Therefore **4.20→2.20 h** correctly deduplicates the *card-reported* total, but 2.20 h is not a precise trace-measured loss. Preserve the raw collect column and report unique-event attribution separately. Credit the same post-exit idle only once if comparing failure prevention with improved polling. A background download overlaps this interval; its useful-work status must be adjudicated separately when reporting cumulative E idle, as the original report already flags.
