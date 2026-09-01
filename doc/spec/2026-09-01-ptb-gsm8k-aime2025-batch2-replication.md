# PTB Batch 2: GSM8K + AIME 2025 independent replication

Batch 2 is an independent replication of the effective 32-cell Batch 1 matrix. Its purpose is to
measure run-to-run variation before interpreting context-window or effort differences. Batch 1
remains immutable; Batch 2 has its own batch identity, receipt, workspaces, checkpoints, and
result directories.

## Frozen matrix

- Tasks: `gsm8k`, `aime2025`.
- Agent setups: Opus 5 max/1M, xhigh/1M, high/1M, and max/200K.
- Starting models: Qwen3-1.7B-Base, Qwen3-4B-Base, SmolLM3-3B-Base, Gemma-3-4B-PT.
- Formal budget: 10 agent hours, one H100 80GB, 16 CPU, and 128G RAM per cell.
- Total: `2 tasks x 4 setups x 4 base models = 32 cells`, or 320 H100 agent-hours before
  evaluation overhead.

The committed manifest is
`experiments/posttrainbench/gsm8k-aime2025-opus5-4x4x2-batch2.yaml`. Its identity is
`gsm8k-aime2025-opus5-4x4x2-batch2-r4`, run index 4.

## Queue and isolation contract

All 32 jobs are submitted held, recorded in one receipt, and released together. Slurm may start
each job as soon as one GPU becomes available; Batch 2 therefore overlaps the tail of Batch 1
without sharing a GPU or workspace. The site launcher remains restricted to
`slurm2-a3nodesetondem-[0-3]`, partition `ptb-a3`, and reservation `robtang-a3`.

This batch does not add a task, evaluator, model, agent profile, container, or infrastructure
path. The already-running Batch 1 is the live qualification of those exact paths, so no new
one-hour pilot is placed ahead of the refill queue. The manifest retains the standard pilot
selection only so the batch remains testable with the common launcher contract.

## Interpretation

Pair Batch 2 cells with the same cell IDs from Batch 1. Report both values, their difference, and
the within-configuration range; do not select only the better replicate. Two runs still do not
support strong significance claims, but they expose conclusions that reverse under a second
agent trajectory.

The same contamination and holdout limitations as Batch 1 apply. In particular, AIME 2025 has
only 30 questions, so one item changes accuracy by 3.33 percentage points. This replication must
not be described as an untouched second holdout.

## Submission boundary

Formal submission requires a clean, pushed top-level repository and PTB submodule; all local and
site checks must report zero issues. The submission receipt is the only ownership authority for
monitoring or cancellation. Never act on jobs by shared Unix user, partition, or node alone.
