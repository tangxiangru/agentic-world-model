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
