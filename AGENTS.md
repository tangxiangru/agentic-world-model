# Experiment operations memory

- Keep expensive accelerators fed asynchronously. Once a downstream batch is independently
  specified, validated, and safe to run, submit it to the scheduler immediately; do not wait for
  the final long-tail job merely to make the submission synchronous.
- Preserve real data dependencies. If a straggler is not needed for a sound decision, freeze the
  decision from the completed valid evidence, document why the straggler was excluded, and queue
  the next batch. If it is genuinely required, prepare and queue every independent job first.
- Treat queued jobs as useful work: they may start on currently free GPUs and naturally backfill
  as allocations become available.
- Never reclaim capacity by cancelling jobs outside the exact receipt/job IDs authorized for the
  current experiment.
- Use `/rmeng_data/robtang/slurm-queue/registry.json` as the cross-CLI ownership authority for the
  four H100 nodes. The shared queue name is `gangda`. A shared Unix user, `root`, reservation
  membership, or node placement is never
  sufficient ownership evidence. Query with `/rmeng_data/robtang/bin/awm-slurm-queue`; treat
  `OWNERSHIP FAIL` as an immediate investigation condition.
- Use the queue views according to their distinct purpose:
  - `gangda-slurm-queue` or `current --watch 5` shows only active and pending operations.
  - `gangda-slurm-queue failures` shows unresolved failures; `--include-resolved` is audit-only.
  - `gangda-slurm-queue history` shows terminal receipts/jobs and must not drive capacity claims.
  - `gangda-slurm-queue show JOB_ID` resolves a job through receipt, cell, manifest, and spec.
  The complete usage contract is `doc/reference/gangda_slurm_queue.md`.
- Do not equate Slurm terminal state with a scientifically complete PTB result. Use
  `uv run awm ptb results MANIFEST` to discover results by frozen provenance and require the PTB
  validator to pass. Use `--all` for incomplete attempts and `--json` for derived analysis. Resolve
  individual jobs with `gangda-slurm-queue show JOB_ID`, then follow receipt -> cell -> manifest ->
  spec -> result directory. Preserve these paths in reports. The full workflow is
  `doc/reference/ptb_result_analysis.md`.
- Reserve `slurm2-a3nodesetondem-[0-3]` exclusively for the receipt-backed PTB batches. AWM full
  studies must use partition `ptb-a3`, reservation `robtang-wm-a3-ondem`, requested nodes
  `slurm2-a3nodesetondem-[4-12]`, and an explicit exclusion for `[0-3]`. Let Slurm pack one-GPU
  jobs by GRES; never use the `a3` partition's whole-node `OverSubscribe=EXCLUSIVE` behavior for
  these studies. AWM full is an external queue and must not be registered in the `gangda`
  ownership registry; its own receipts remain under `wm-study-runtime`.
