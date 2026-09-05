# Experiment operations memory

- `skills/exp_protocol/` is runtime guidance for scientist agents that actually train or
  evaluate models inside PTB. It is not the workflow for planner, operator, reviewer, or ordinary
  repository code changes. Before a PTB scientist takes any exploratory or implementation action,
  the scaffold must make it explicitly invoke/read `exp_protocol`; no training or evaluation
  command may run before that scientist creates, checks, and locks its experiment card.
- The active main development and integration branch is `gangda-dev`. Treat
  `gangda_trial_0828` references in existing manifests, specs, receipts, and job names as frozen
  historical provenance; use `gangda-dev` for new development integration and PR targets unless
  the user explicitly names another branch.
- Keep expensive accelerators fed asynchronously. Once a downstream batch is independently
  specified, validated, and safe to run, submit it to the scheduler immediately; do not wait for
  the final long-tail job merely to make the submission synchronous.
- Preserve real data dependencies. If a straggler is not needed for a sound decision, freeze the
  decision from the completed valid evidence, document why the straggler was excluded, and queue
  the next batch. If it is genuinely required, prepare and queue every independent job first.
- Treat queued jobs as useful work: they may start on currently free GPUs and naturally backfill
  as allocations become available.
- For `gangda_exp-protocol-evolve`, maintain a floor of eight independently specified, validated
  cells as Slurm `PENDING(JobHeldUser)`, not ordinary runnable pending jobs. An ownership failure
  never releases them. Release requires `OWNERSHIP OK`, a per-job `ReqNodeList` match to the frozen
  receipt, and restored native two-node isolation; replenish the held buffer before it falls below
  eight whenever scientifically valid downstream work is available.
- Harvest every spillover job even when it is failed, cancelled, timed out, or otherwise terminated
  incorrectly. Placement-only validator-complete results are sensitivity evidence, never primary;
  incomplete attempts remain failed/truncated evidence. Requeue with a new immutable manifest and
  receipt when required for two valid repeats, a matched-arm comparison, or a strict-site promotion
  decision.
- Never reclaim capacity by cancelling jobs outside the exact receipt/job IDs authorized for the
  current experiment.
- For `exp_protocol` trace windows, do not wait for a Fable/GitHub reply. Once eight new
  receipt-backed validator-clean cells have accumulated, invoke local Claude Code in the
  background as `claude-opus-5[1m]` at `--effort max` (the installed CLI's mapping for the
  user's "Opus 5 ultracode" request), following
  `doc/reference/exp_protocol_local_claude_analysis.md`. Claude is a read-only analysis helper;
  the Codex planner reads its reports and owns every experiment, protocol, queue, and git decision.
- Keep the long-running Codex goal active and use `tools/exp_protocol_completion_monitor.py` as
  the slow external-event detector, normally at a one-hour polling cadence so terminal attempts
  accumulate into analysis batches. Its ready state is surfaced on resume/compaction by the
  project SessionStart hook; Slurm terminal state only wakes the loop and never substitutes for
  receipt-backed PTB validation.
- After every `exp_protocol` analysis window, perform the full trace review and use additional
  reviewer subagents when cross-cell ambiguity remains. Prune whole not-yet-started pending blocks
  that the evidence makes scientifically unnecessary, but only through their immutable receipt job
  IDs, never cancel running work, and preserve at least eight validated `PENDING(JobHeldUser)`
  cells. Persist reusable loop knowledge in `skills/exp_protocol_meta/`, not only in chat or Claude
  logs.
- Use `/rmeng_data/robtang/slurm-queue/registry.json` as the cross-CLI ownership authority for the
  four H100 nodes. The shared queue name is `gangda`. A shared Unix user, `root`, reservation
  membership, or node placement is never
  sufficient ownership evidence. Query with `/rmeng_data/robtang/bin/awm-slurm-queue`; treat
  `OWNERSHIP FAIL` as an immediate investigation condition.
- The `gangda` queue is hard-split into two non-borrowing 16-GPU subqueues:
  `gangda_exp-protocol-evolve` owns `slurm2-a3nodesetondem-[0-1]`, and
  `gangda_wma_evolve` owns `slurm2-a3nodesetondem-[2-3]`. New receipts inherit a subqueue from
  their branch. Do not route a line onto the other subqueue's nodes.
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
- PostTrainBench remote ownership is fixed: `fork` is
  `https://github.com/tangxiangru/PostTrainBench.git` for the AWM branch, and `upstream` is the
  official `https://github.com/aisa-group/PostTrainBench.git` for fetch/rebase. Do not configure
  `DeepCommit-ai/PostTrainBench` as upstream. `.gitmodules` intentionally clones the fork because
  the pinned AWM commits are not part of official upstream.
- Scope routine operations and reporting to the `gangda` registry, receipt-backed PTB batches, and
  `slurm2-a3nodesetondem-[0-3]`. AWM full is an external queue: do not monitor, modify, submit,
  cancel, analyze, or report it unless the user explicitly names it in a later request. Never
  register AWM full receipts in `gangda`.

# Skills

Skills for agents working in this repository live in `skills/<name>/SKILL.md`
(Agent Skills format; the same files serve Claude Code and Codex).

- `skills/exp_protocol/SKILL.md` — the experiment protocol a scientist follows
  when training or evaluating a model. Installed into a scientist's task
  directory with `awm exp_protocol install --target <dir> --tool <claude|codex|both>`.
- `skills/exp_protocol_meta/SKILL.md` — how the protocol itself is iterated on
  the GPU cluster. For the iteration agent only; never installed for a scientist.
  Both skills are visible in this checkout; the separation is enforced where it
  matters, by `awm exp_protocol install`, which refuses to copy the meta skill.
- `skills/wma/SKILL.md` — the world-model agent: given an experiment card's pre-launch
  sections, estimate what the run will do and write `exp-NN.verdict.json`. Invoked by
  `awm wma review`; iterated offline by `awm wma replay` over the historical card corpus.
- `skills/wma_meta/SKILL.md` — how the world-model agent itself is iterated: replay a round,
  read the ledger per level, change one thing in `skills/wma/`, record why. For the
  iteration agent only; the WMA never reads it while producing a verdict.

Codex: read the SKILL.md directly, or link a skill into `~/.codex/skills/`:
`ln -s "$(pwd)/skills/exp_protocol_meta" ~/.codex/skills/exp_protocol_meta`.
Claude Code discovers the same skills through the symlinks in `.claude/skills/`.

The boundary between the protocol and the world-model policy is defined in
`doc/reference/exp_protocol_and_wma_policy.md`.
