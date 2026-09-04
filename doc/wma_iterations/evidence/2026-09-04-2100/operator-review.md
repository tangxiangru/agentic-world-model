# 21:00 UTC operator review — BFCL P n=3 and HumanEval contamination

Ownership is OK. At21:01 UTC the subqueue has16/16 GPUs allocated,16RUNNING
and35 safely routed PENDING jobs on nodes2–3, no bad routes or scheduler
dependencies. The pending count is34 original scientific jobs plus validation-
only92312; scientific reserve remains above32 and>8 without counting the smoke.
No replenishment/cancellation is needed. Allocation is not utilization; direct
utilization remains unavailable under the recorded access constraints.

The inspected reconcile preview contained three harvests and18 peeks, no
submit/cancel. Application archived all three terminal attempts. Total scientific
PTB-complete is95; automatic-judge-clean is92. The Opus4.8 study has6complete,
3clean. Two GSM8K single-WMA jobs w57r01/02 have started with frozen public/
private31b854bb, skill17be8a23046a, Opus4.8/high and blocking single mode.

## BFCL protocol-only P — three clean repeats

c54r02 /92186 is clean at **88/100**, scientist cost$20.64914275, agent time
04:06:47 and allocation04:18:23 (4.3064GPU-h); judge cost unavailable. Required
judges and final-model provenance are clean. Together P scores are91/88/94,
mean **91.0%**, sample SD **3.0pp**, n=3. Known scientist cost sums$49.99916425
and allocation09:00:46 (9.0128GPU-h), excluding judges; c54r04 remains running.

## BFCL raw R — two flagged sensitivities, zero primary

c53r03 /92183 has a valid final model and scores **91/100**, but general_anomaly
is confirmed. The preserved selected sft_v3 checkpoint is valid; a later v4 run
was around297/3082 when the scientist said it would continue after a watcher.
Text-only end_turn terminated the session and killed v4, so the intended larger
comparison was never run. Scientist cost$26.667334, agent time04:37:54,
allocation04:47:44 (4.7956GPU-h); judges otherwise clean.

With flagged c53r04=.92, raw sensitivity mean is91.5%, sample SD0.7071pp,n=2.
Raw primary-clean remains **0/4**; c53r01 is incomplete and c53r02 running.
Do not compare flagged raw mean against clean P or select their maxima. P–R
primary remains undefined despite similar displayed values.

## HumanEval protocol c56r03 — incomplete and contaminated

c56r03 /92195 has no checkpoint, final model, metrics or score. Scientist cost
is$6.60754875, judges$2.83253325, recorded total$9.440082 excluding GPU;
agent time00:29:16, allocation00:42:38 (0.7106GPU-h). Its
training reached roughly96/2250 before a text-only waiting end_turn killed
background work. This is incomplete/general_anomaly, never a numeric zero.

The more serious independent failure is `contamination`. The frozen exp-01 lock
binds `/home/ben/task/data/sft_final.jsonl`, 110,689,404bytes, SHA256
`6bf97a54d9eea153a54b975971136686b7d90b81c0ec567e3cf96456efc2012d`.
The scientist's n-gram guard reported zero, but the semantic judge's direct
inspection reports **279 documents across67 HumanEval-idiosyncratic function
families** in the selected Magicoder OSS/Evol lineage. That exceeds the allowed
tolerance and the training command actually used this frozen file. No protected
prompt/solution text is repeated in operator records.

The cell has no WMA (`wma=not_attached`, zero wait), so this is a protocol/data-
boundary failure, not a WMA-skill failure. Keep contamination/general flags and
the incomplete outcome. A prior c55r04 judge reported contamination=false for
Magicoder-derived data using the same n-gram style check; that does not establish
semantic cleanliness. Future Magicoder-derived HumanEval results require the
same semantic audit before entering a primary cohort. Do not retroactively
change judge files or infer dataset-wide contamination solely from lineage.

The delegated bounded audit independently verifies receipt/manifest/spec and
canonical/harvested hashes, the frozen training-file hash, 24,000 selected rows
(14k OSS/10k Evol), 23,986 retained by the trainer, zero-byte n-gram reports,
96/2,250 progress, zero checkpoint files and the original judge hashes. It
records the279/67 semantic finding without copying protected prompt/solution
text. Exact contaminated rows consumed before termination are unknowable because
the trainer did not log sample IDs. See `humaneval-contamination.md` and its
separate manual-flag JSON; original judge outputs remain authoritative artifacts.

## Ranked decision

1. HumanEval semantic decontamination must move into a trusted prelaunch
   protocol/harness surface; n-gram-only validation is insufficient. A design-
   only candidate is preregistered at
   `doc/spec/2026-09-04-humaneval-semantic-decontamination.md`, with positive/
   negative/evasion canaries, no protected-text export, <=10-minute CPU cost and
   fail-closed behavior. It does not modify current frozen cells or scorer.
2. The background-wait/end_turn lifecycle pattern now has seven observed
   terminal instances across raw/protocol and three tasks. Some preserve prior
   valid models; others produce none. It is common runtime behavior, not a WMA
   skill effect and not a fixed30-minute cutoff. Selective retries remain invalid.
3. BFCL P is promising at n=3 but has no clean raw comparator and one pending
   replicate. No method conclusion, promotion or new outcome-selected arm is
   authorized. Wait for frozen denominators.

No WMA skill, protocol implementation, scorer, guard, retry or scientific job
is changed/submitted in this check. The shared PostTrainBench worktree contains
a preserved external update and S0 validation92312 remains pending; those are
explicit dependencies before implementing or launching later harness candidates.

The hourly hook is alive and has seven new clean cells beyond the prior analyzed
event. One more clean result will trigger Opus5/max automatically; the six-hour
tail has not matured. The previous event already has a corrected handoff, so no
duplicate analysis starts here.

`new-results.json` preserves each receipt→cell→manifest→spec→result chain,
frozen treatments, scores, judges and costs. The delegated contamination audit
and its separate manual flag/cross-check preserve semantic evidence without
copying test content. Original results and flags remain unchanged.
