# PostTrainBench GSM8K + AIME Batch 1 runbook

This runbook operates `gsm8k-aime2025-opus5-4x4x2-batch1-v2` from branch `gangda_trial_0828`. The
only permitted nodes are `slurm2-a3nodesetondem-[0-3]`, via partition `ptb-a3` and reservation
`robtang-a3`.

## Local and infrastructure gates

```bash
uv run --extra dev pytest
uv run awm ptb dry-run
uv run awm ptb check --before-context-gate

cd third_party/PostTrainBench
bash src/commit_utils/slurm/run_gates.sh g1 slurm2-a3nodesetondem-0 \
  gsm8k-aime2025-opus5-4x4x2-batch1-v2
bash src/commit_utils/slurm/run_gates.sh g2 slurm2-a3nodesetondem-0 \
  gsm8k-aime2025-opus5-4x4x2-batch1-v2
bash src/commit_utils/slurm/run_gates.sh g3 slurm2-a3nodesetondem-0 \
  gsm8k-aime2025-opus5-4x4x2-batch1-v2
cd ../..

uv run awm ptb context-smoke \
  --cell g02 --cell g06 --cell g10 --cell g14
uv run awm ptb check
```

The permanent Slurm template must contain `AccountingStorageTRES=gres/gpu`; all four node
`CfgTRES` values must contain `gres/gpu=8`, and `ConstrainDevices=yes` must remain enabled. The
2026-09-01 pre-fix GCS backup is
`gs://slurm-slurm2834bd/slurm2-files/backups/config.yaml.pre-ptb-gres-20260901T015552Z`.

## Two pilots

```bash
uv run awm ptb submit --pilot
uv run awm ptb status \
  data/ptb/batches/gsm8k-aime2025-opus5-4x4x2-batch1-v2/pilot-*.json
uv run awm ptb audit-receipt \
  data/ptb/batches/gsm8k-aime2025-opus5-4x4x2-batch1-v2/pilot-*.json
```

The pilot receipt contains `g06` (GSM8K) and `a06` (AIME 2025), each with a one-hour agent budget
followed by all official judges and full final evaluation. Both must audit with zero issues.

## Formal confirmation boundary

Do not run the following command without fresh user confirmation:

```bash
uv run awm ptb submit
```

It submits all 32 jobs held, writes one receipt, and then releases all job IDs together. Each job
has a ten-hour agent budget and requests one H100, 16 CPU, and 128G RAM. Before confirmation,
verify the four nodes are otherwise idle, both repositories are clean and pushed, the two pilot
audits are clean, and `uv run awm ptb check` reports zero issues.
