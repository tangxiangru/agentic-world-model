# PTB Batch 2: result-driven GSM8K + AIME 2025 replication

**Status:** selection frozen from 31 complete Batch 1 cells; Batch 2 manifest is submittable
asynchronously while the a13 long-tail cell finishes.

Batch 2 will measure run-to-run variation on a selected half of the Batch 1 matrix. Batch 1
remains immutable. The final Batch 2 will have its own batch identity, receipt, workspaces,
checkpoints, and result directories.

## Superseded submission

The first draft proposed one repeat of all 32 Batch 1 settings. Jobs `88926` through `88957` were
submitted from top-level commit `b4d874f` and PTB commit `0c11fa1`, then cancelled while all were
still pending. Every job consumed `00:00:00`; no agent, evaluator, or GPU process started. That
receipt is historical evidence only and must not be resumed or reused.

## Final Batch 2 shape

- Candidate pool: the 32 effective, audited Batch 1 settings.
- Selected settings: 16.
- Independent repeats per selected setting: 2.
- Total jobs: `16 settings x 2 repeats = 32 jobs`.
- Per-job budget: 10 agent hours, one H100 80GB, 16 CPU, and 128G RAM.
- Tasks remain `gsm8k` and `aime2025`; target eight selected settings per task unless invalid
  Batch 1 coverage makes that impossible.

The final manifest encodes each selected setting twice with distinct cell IDs, run identities,
workspaces, and output directories. Its contract declares
`replication: {settings: 16, repeats: 2, settings_per_task: 8}`; the launcher rejects any other
cardinality and requires each pair to carry explicit `replicate: 1` and `replicate: 2` labels.

## Frozen selection

The selection was frozen from the 31 cells with complete metrics and official judge coverage.
The remaining a13 cell was not required for a sound selection: its live clean-development result
was below the AIME cutoff while it continued a repair anneal. Per the queue-saturation policy,
that independent long tail does not block Batch 2 submission.

| Task | Batch 1 settings | Batch 1 score |
|---|---|---|
| GSM8K | g01, g02, g05, g06, g10, g11, g13, g15 | 0.8127, 0.9249, 0.8309, 0.9105, 0.9067, 0.8385, 0.8400, 0.8749 |
| AIME2025 | a02, a03, a06, a07, a10, a11, a14, a15 | 5, 4, 6, 4, 6, 3, 5, 5 correct out of 30 |

The GSM8K selection is the clean top eight after conservatively excluding the three
`general_anomaly=true` runs g03, g09, and g14. The AIME selection keeps the two strongest
base-model families across all four profiles, producing a balanced matched comparison; a11 wins
the three-question tie because it completes that contrast. No contamination, disallowed-model,
API-usage, or PTB-lookup flag is admitted.

## Selection rule

Selection uses only valid Batch 1 results; infrastructure failures and required-judge failures
are not treated as low-scoring scientific outcomes.

Within each task:

1. rank clean settings by official final score;
2. prefer stronger base models and the stronger agent profiles, with profile priority
   `max/1M`, then `xhigh/1M`, then `high/1M`, then `max/200K`;
3. select settings that are both high-performing and informative about an effort/context
   comparison, rather than taking 8 unrelated score maxima;
4. keep at least one matched contrast when it is needed to distinguish effort from context;
5. freeze the complete selection table and rationale before submitting either repeat;
6. do not wait for an irrelevant long-tail cell once the selection is already determined.

The default target is that most selected settings use `max/1M` or `xhigh/1M` and the stronger
Batch 1 base models. A weaker profile or model enters only when its Batch 1 result is competitive
or it supplies a necessary matched contrast. AIME differences must be reported in question
counts as well as percentages because one item changes accuracy by 3.33 percentage points.

## Queue and submission boundary

After selection, all 32 jobs are submitted held, written into one receipt, and released together.
The site remains restricted to `slurm2-a3nodesetondem-[0-3]`, partition `ptb-a3`, and reservation
`robtang-ptb-a3`. Jobs submit as root only to obtain a valid Slurm allocation, then immediately
drop to `robtang_google_com` before any task-side operation.

Submission is asynchronous: free GPUs begin Batch 2 immediately, while the remaining jobs stay
pending and backfill as existing allocations complete. A terminal straggler from the preceding
batch is not a synchronization barrier when it is not a dependency of the frozen selection.

Formal submission requires a clean, pushed top-level repository and PTB submodule, a zero-issue
local/site gate, and a committed manifest whose 16 settings exactly match the frozen selection
table. The receipt is the only ownership authority for monitoring or cancellation. Never act on
jobs by shared Unix user, partition, or node alone.
