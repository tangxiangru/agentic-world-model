# PTB official Claude Opus 5 high judge migration and recovery

## Decision

The canonical PostTrainBench judge runtime is Claude Code with
`claude-opus-5[1m]` at `high` effort for all four judge definitions, including
`general_judge`. The `official` profile retains the existing canonical output
ids (`gpt5_4`, `api`, `ptb_lookup`, and `general`) so validators and historical
aggregation continue to consume one stable artifact contract. New judge
metadata must record `profile=official`, `backend=claude`, the requested and
resolved model, `reasoning_effort=high`, Vertex auth, the container, and the
Claude CLI version.

The `claude` profile remains available only as a research-output namespace. It
uses the same model and effort but writes `judgement_claude_*`; it is not a
second model policy.

## Batch 2 recovery scope

The source receipt is
`data/ptb/batches/gsm8k-aime2025-opus5-selected16x2-batch2-v5/formal-retry1-2026-09-01T215322.199428+0000.json`.
Exactly the cells whose result validator reports only the four missing
canonical judge verdicts are eligible. A result missing a model, trace,
monitor, provenance, final evaluation, or metrics must be excluded rather than
silently treated as judge-only recovery.

Each eligible result receives one CPU-only Slurm recovery job. The job runs all
four judges from a frozen PostTrainBench commit, writes `_rerun` artifacts so
the failed originals remain auditable, and then runs the formal completion
validator. It must not rerun the agent or model evaluation.

## Ownership and launch contract

Recovery jobs are submitted held to the `gangda_wma_evolve` subqueue on
`slurm2-a3nodesetondem-[2-3]`, written to one receipt, registered in
`/rmeng_data/robtang/slurm-queue/registry.json`, and only then released. Unix
user identity is not ownership evidence. Every job records the source job id,
result directory, source receipt, top-level commit, PostTrainBench commit,
Opus 5 high context-validation digest, judge container digest, model, effort,
auth mode, and Slurm identity.

Success requires every recovery Slurm job to complete and every target result
to pass `validate_completed_run.py --judge-profile official`. Judge flags are
scientific review outcomes, not infrastructure failures, and remain visible in
the canonical verdict files.
