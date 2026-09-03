[WMA scheduled operator check]

Continue the user's ongoing gangda_wma_evolve experiment iteration in this task.
The user authorized recurring monitoring, scientifically justified next-wave
experiments, detailed experiment records, and commit/push of every validated
skill change. Work until this check's actionable work is handled; the timer
will revisit the task. Do not create another timer or archive this task.

Read AGENTS.md, skills/wma_meta/SKILL.md, the latest online round records and
doc/reference/wma_evolve_hook.md. Read the shared event's operator-review.md
alongside its Claude report; a ready report is advisory, not accepted evidence.
Use the existing user's instructions when older documentation conflicts.

1. Check the gangda ownership registry and the live gangda_wma_evolve subqueue
   with /rmeng_data/robtang/bin/awm-slurm-queue. Operate only receipt-backed PTB
   work on slurm2-a3nodesetondem-[2-3]. Investigate OWNERSHIP FAIL immediately.
   Distinguish GPU allocation from utilization. Identify RUNNING and PENDING
   cells, their frozen treatments and real dependencies. Below 24 safely routed
   pending jobs, prepare replenishment toward at least 32; preserve a >8 reserve.
   Submit independently specified, validated experiments asynchronously as soon
   as ready; do not wait for unrelated tails or invent scientifically redundant
   repetitions just to increase the queue count. Never cancel running jobs or
   anything outside the exact authorized receipts.
2. Use awm ptb results MANIFEST --all --json and the PTB validator to discover
   complete, judge-clean results. Inspect the reconcile preview before applying
   any needed harvest or submission. Keep failures, incomplete cells and distinct
   runtime cohorts separate. Preserve receipt -> cell -> manifest -> spec ->
   result paths and the primary/sensitivity exclusions. Investigate a score
   record within its matched cohort; a single high score is not promotion.
3. Check tools/wma-evolve-hook status. Keep its existing singleton hourly daemon
   running. Each eight new clean cells (or four after the documented tail wait)
   triggers local Claude Code claude-opus-5 / max / ultracode, read-only. Consume
   and verify new reports without waiting for PR23. Claude diagnoses trajectories
   and proposes experiments; this iteration task owns mutations. Avoid duplicate
   analysis of the same evidence window and record completed review handoffs.
4. Follow wma_meta: reproduce the authoritative ledger, delegate bounded trace
   questions in parallel when evidence exists, rank causes, then preregister
   single-edit candidates with metrics, falsification and leak/cost/PTB guards.
   Keep WMA skill changes separate from protocol/harness changes. Do not change
   the scorer or guards post hoc to make a candidate pass. Preserve the internal
   blocking WMA review: preparation may overlap the wait, a new experiment may
   not start before its verdict returns. Use four-cell manifests by default;
   respect formal comparison and held-out promotion gates. AIME is promotion-only.
5. Record every experiment's rationale, frozen SHAs/skill hashes, manifests,
   jobs, scores, variability, costs, exclusions and decision in the round record.
   Before repository mutations check git status and fetch/rebase safely; preserve
   existing work. Commit each validated skill edit together with its evidence
   record and push gangda_wma_evolve. Keep harvested results and operational
   records current. Do not force-push or merge.
6. Report material changes concisely in Chinese: queue capacity/reserve, newly
   valid scores, analysis/experiment decision, submitted jobs, commits/push and
   concrete blockers. If nothing changed, keep the check brief. Continue until
   the user stops monitoring. If access or ownership prevents action, record and
   report the limitation instead of claiming the loop is healthy.
