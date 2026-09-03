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

## Scheduled operator continuation

Since 2026-09-03 the operator also has a 30-minute server cron entry. It calls
`awm/wma_evolve_timer.py`, which uses the installed `codex queue` CLI to continue
the existing operator task with `doc/reference/wma_evolve_timer_prompt.md`.
The hourly hook still performs read-only Claude analysis; the operator task
verifies reports, replenishes experiments, records results and commits/pushes
validated changes. PR polling is not involved.

The dispatcher changes no model, approval or sandbox settings. It holds a local
lock and reads the local Codex queue database read-only to avoid adding a second
message when this task already has queued input. A missing queue database,
changed schema, wrong branch or unavailable app server prevents dispatch and is
recorded, rather than starting an unrelated task. This uses server cron, not the
desktop Scheduled interface; the server and the existing Codex app-server must
remain available.

Configuration, the last tick and cron logs live in
`/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/timer/`:

- `config.json`: enabled flag, exact task UUID, repo, CLI/socket/database paths,
  and operator prompt path; no credentials.
- `last_tick.json`: queued, skipped, disabled or error status.
- `cron.log`: dispatcher output.
- `crontab.before`: the user's crontab before installation.

Use `crontab -l` to inspect the marked `gangda-wma-evolve-operator` entry.
Set `enabled` to false in this timer's `config.json` to pause operator wakeups;
remove only that marked cron entry to uninstall. This does not stop the separate
read-only Claude completion hook. A paused or removed timer cancels no Slurm job.
