# Gangda Slurm queue interface

The `gangda` queue separates current operations, unresolved failures, terminal history, and
scientific task identity. The receipt-backed registry is
`/rmeng_data/robtang/slurm-queue/registry.json`; Unix users and node placement are not ownership
evidence.

## Current operations

```bash
gangda-slurm-queue
gangda-slurm-queue --summary
gangda-slurm-queue --watch 5
```

The default view contains only `RUNNING`, `PENDING`, `CONFIGURING`, `COMPLETING`, and `SUSPENDED`
jobs. `GPUS N/32 allocated` is a Slurm allocation count, not instantaneous GPU utilization.
`node=-` means a pending job has no allocation. `reason=(Resources)` means it is waiting for a
resource currently held by another registered job.

## Failures

```bash
gangda-slurm-queue failures
gangda-slurm-queue failures --include-resolved
```

The default failure view suppresses a failed cell when a later receipt for the same batch and
cell is `PENDING`, active, or `COMPLETED`. `--include-resolved` shows those audit records together
with the replacement job that resolved them. Ownership violations are always an immediate
failure condition.

## History

```bash
gangda-slurm-queue history --summary
gangda-slurm-queue history
```

History contains terminal receipt/job states such as `COMPLETED`, `FAILED`, and `CANCELLED`, plus
receipt, manifest, and spec paths. It does not describe current capacity.

## Explain one task

```bash
gangda-slurm-queue show 89589
gangda-slurm-queue show 89589 --json
```

`show` resolves one registered Slurm ID through its source receipt and cell. For PTB it prints the
task, base model, agent profile, effort, context window, replicate, receipt, manifest, spec, and
frozen commits. When a result directory exists, it also prints validation status, accuracy, and
judge flags. `scontrol show job JOB_ID -o` remains the lower-level scheduler view. Scientific
result analysis is documented in [`ptb_result_analysis.md`](ptb_result_analysis.md).

## Machine-readable state

The systemd monitor refreshes these every 15 seconds:

```text
/rmeng_data/robtang/slurm-queue/current.txt
/rmeng_data/robtang/slurm-queue/current.json
```

The JSON snapshot intentionally retains all registered sources for audit; the command-line views
decide whether to present current, failure, or historical records.
