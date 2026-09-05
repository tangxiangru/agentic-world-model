# Gangda Slurm queue interface

The `gangda` queue separates current operations, unresolved failures, terminal history, and
scientific task identity. The receipt-backed registry is
`/rmeng_data/robtang/slurm-queue/registry.json`; Unix users and node placement are not ownership
evidence.

## Subqueues and hard capacity

The `gangda` contract is a hard, non-borrowing node split; native Slurm isolation and
the frozen job constraints must enforce each line's 16-GPU ceiling:

| subqueue | branch | nodes | GPUs |
|---|---|---|---:|
| `gangda_exp-protocol-evolve` | `gangda_exp_protocol_evolve` | `slurm2-a3nodesetondem-[0-1]` | 16 |
| `gangda_wma_evolve` | `gangda_wma_evolve` | `slurm2-a3nodesetondem-[2-3]` | 16 |

The split is non-borrowing: an idle GPU in one subqueue is not available to the other. Each
line's PTB `.env` names both `POST_TRAIN_BENCH_SLURM_SUBQUEUE` and the matching two-node
`POST_TRAIN_BENCH_SLURM_NODELIST`; `awm ptb check` rejects a mismatch. The queue monitor also
checks every registered allocated job's actual node and the sum of its GPU requests. A job that
runs outside its subqueue nodes, or registered running GPU requests above `gpu_limit`, makes the
view say `OWNERSHIP FAIL` even when the owned nodes themselves still show `GPUS 16/16`.
Existing running jobs are never cancelled or moved automatically; an ownership failure stops new
submissions and requires an operator investigation.

## Current operations

```bash
gangda-slurm-queue
gangda-slurm-queue --summary
gangda-slurm-queue --subqueue gangda_exp-protocol-evolve --summary
gangda-slurm-queue --subqueue gangda_wma_evolve --summary
gangda-slurm-queue --watch 5
```

The default view contains only `RUNNING`, `PENDING`, `CONFIGURING`, `COMPLETING`, and `SUSPENDED`
jobs. `GPUS N/32 allocated` is a Slurm allocation count, not instantaneous GPU utilization.
`node=-` means a pending job has no allocation. `reason=(Resources)` means it is waiting for a
resource currently held by another registered job.

The default view shows the total `gangda` allocation and both subqueue capacities. A
`--subqueue NAME` view limits nodes and receipt sources to that line. Its GPU count is physical
allocation on the subqueue's nodes, so a pre-split legacy job is still visible in the capacity
number even when its historical receipt has no subqueue tag. `registered_running_gpus` is the
independent receipt-backed demand count; it is what detects spillover or oversubscription that a
physical-node-only total would hide.

### When a deployed wrapper disagrees with receipt evidence

Check which checkout the wrapper resolves to. The registry remains the same ownership authority,
but an older checker can omit registered-job placement and total-request checks. On 2026-09-03,
the shared wrapper resolved to the other `agentic-world-model` checkout at `6d9f866` and printed
`OWNERSHIP OK` for physical 16/16 while job 90820 ran outside its frozen nodes and registered
requests totaled 17. The current operator checkout's
`.venv/bin/awm slurm queue --subqueue gangda_exp-protocol-evolve --summary` correctly reported
`OWNERSHIP FAIL` from the same registry. See
[`90820` audit](../../results/ptb/placement-violation-90820-20260903T195038Z.json).

Do not accept a stale summary over direct receipt/job evidence or release work while the
disagreement is unresolved. Record the checker version and reconcile the shared deployment with
its owner; do not silently redirect the cross-CLI wrapper to a different worktree. Running jobs
are still never cancelled or moved automatically.

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
