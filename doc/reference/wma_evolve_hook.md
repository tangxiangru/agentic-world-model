# WMA evolution completion hook

`tools/wma-evolve-hook` is the shared, branch-scoped background monitor for
`gangda_wma_evolve`. It replaces ad-hoc PR polling; GitHub PR #23 is a handoff
and audit surface, not a dependency for analysis.

The daemon reads `/rmeng_data/robtang/slurm-queue/registry.json` to discover
receipt-backed WMA manifests and `/rmeng_data/robtang/slurm-queue/current.json`
for queue health. Every 60 minutes it runs `awm ptb results --all --json` for
the in-scope manifests. Slurm terminal state is never treated as a completed
scientific result.

On first start it acknowledges all existing clean-complete cells. Thereafter
it freezes an event after eight new validator-clean cells. A four-cell tail is
also eligible after six hours so that the final partial window is not lost.
Claimed cell IDs make the trigger idempotent. Events and status are shared at:

```text
/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/
  daemon.json
  queue-health.json
  state.json
  status.json
  events/<timestamp>-<cell-hash>/
```

Each event launches Claude Code with explicit `claude-opus-5`, effort `max`,
and the `ultracode` workflow trigger. It runs in plan permission mode and is
instructed to perform read-only trajectory, ledger, uptake, harm, compliance,
score-lever and experiment-design analysis. It cannot submit/cancel Slurm
jobs, edit the checkout, commit, push, or comment on a PR. Codex consumes the
result, verifies it against the frozen evidence, and owns all mutations.

The queue health side checks all PENDING job routes against
`slurm2-a3nodesetondem-[2-3]`, emits a replenishment event below 24 safe pending
jobs, and marks `<=8` as the hard-floor breach. It never changes queue state.

## Operator commands

```bash
tools/wma-evolve-hook status
tools/wma-evolve-hook once --no-claude
nohup tools/wma-evolve-hook run --interval 3600 \
  > /rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/daemon.log 2>&1 &
```

Only one daemon can hold `daemon.lock`. To replace it, send `TERM` to the exact
PID in `daemon.json`, confirm it stopped, and then start the new version. This
process is not a Slurm job and does not own or cancel accelerator work.
