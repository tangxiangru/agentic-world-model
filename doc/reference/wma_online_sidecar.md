# Online WMA sidecar boundary

The scientist reads `skills/exp_protocol` and uses the public `awm wma`
client. The WMA implementation, skill, priors, historical corpus, and full
transcripts remain private inputs to the sidecar.

## Compare before selecting a training plan

```bash
awm wma propose --dir .
# edit memory/decisions/decision-01.proposal.json, including the initial preference
awm wma compare --dir . decision-01
awm wma choose --dir . decision-01 --candidate A --reason "why A is preferred" --card exp-02
```

The proposal set uses `awm-wma-proposal-set-v1`: shared situation and evidence,
the scientist's initial preference, and 1–3 candidate briefs. One candidate
requires a `singleton_reason`. Each brief contains its hypothesis, parent,
change, train/eval hours and cost basis, evidence, uncertainty, and decision
test. `skills/exp_protocol/proposal.example.json` is a synthetic example.

Prepare the same briefs in both WMA and control cells. Record the initial
preference before comparison and the final selection afterwards. The sidecar
compares the supplied candidates in one shared context; it does not separately
review each candidate and sort by verdict confidence. The client records a
control/unavailable or failed comparison too; the scientist can still choose.

A comparison does not approve a launch or replace the selected card's formal
review. Bind the choice to the formal card with `choose --card` once it exists.
Use measured training/evaluation costs where available and retain uncertainty
in estimates. Neither a higher candidate count nor more training runs is itself
evidence of improved efficiency.

`choose --card` binds the selected brief to the current formal plan and its
declared script/data/config bytes. Reconfirm that binding after an implementation
repair; compare again if the candidate brief changes. `choose --decline --reason
"..."` declines every candidate and supersedes an earlier selection. It does
not bind a runnable card.

## Review, record the action, and launch the exact version

```bash
awm exp_protocol lock --dir . exp-02
awm wma act --dir . exp-02 --action proceed --reason "review considered; why this version should run"
awm exp_protocol run --dir . exp-02
awm exp_protocol close --dir . exp-02
```

`lock` performs preflight, pins the proposal and inputs, then waits for an
attached WMA's verdict. The scientist must not launch before it returns.
An absent, failed, or timed-out review is recorded; it is not a negative verdict
and does not require a manual retry. The scientist may record a reasoned
`proceed` and launch. A `no` or `defer` remains advice that the scientist can
decline with a recorded reason.

`awm wma act` appends immutable action events with an action, reason, optional
suggestion ID such as `precondition-1`, and repeated `--evidence PATH` values.
Actions are `proceed`, `repair`, `probe`, `decline`, and `abandon`. Record the
decision before acting and add evidence as it becomes available. Record which
advice changed execution, including advice inside a `yes`. Unrun proposals
have unknown endpoints; cancelling them does not establish that the WMA
predicted failure correctly.

`awm exp_protocol run` validates the current lock hashes and current `proceed`
record, runs the locked argv, and records launch and exit. Changed inputs need
a new lock/review and a new proceed record. `close` retains its existing
completion checks. The launch wrapper checks this execution path; it is not a
claim that shell access or every possible filesystem operation is sandboxed.
Declare command configuration files in `setup.command.configs` so lock and
launch checks cover their contents too. Every lock revision has a unique ID;
a late reply for an earlier lock cannot overwrite the current lock.

For an additional review of already-locked cards:

```bash
awm wma review --dir . exp-01 [exp-02 ...] --background
awm wma status --dir .
```

This legacy batch entry point remains separate from joint candidate comparison
and never substitutes for the wait inside `lock`.

## Public and private records

The scientist-visible checkout contains `awm/wma_client.py` but not `awm/wma`,
`skills/wma`, `skills/wma_meta`, WMA history, or WMA transcripts. It does not
choose the backend, model, effort, budget, history, or skill version. Public
records include requests/status, card verdicts, decisions, comparisons, and
action events. Full WMA transcripts are harvested under `wma_private/` for
iteration-agent analysis.

Record locations:

- `memory/decisions/decision-NN.proposal.json`: editable briefs before submission.
- `memory/decisions/decision-NN/choices/*.json`: immutable choices and formal-plan bindings.
- `.wma/comparisons/REQUEST/`: frozen request, comparison and client completion state.
- `.wma/reviews/REQUEST/exp-NN/`: per-request card/lock copies, verdict and completion.
- `.wma/actions/exp-NN/`: scientist actions; one proceed event permits one launch.
- `.wma/launches/exp-NN/`: command start/exit records, not scientific completion.
- `wma_private/reviews/REQUEST/` and `wma_private/comparisons/`: private inputs,
  model transcripts and measured costs, including superseded requests.

Records are published atomically without overwriting an existing event. The
harvester retains nested private evidence and gzips transcripts; the old terminal
transcript surface remains for compatibility. Archive filenames do not match
the v1 ledger's verdict glob, so archived revisions are not scored twice.

`third_party/PostTrainBench/src/run_task.sh` starts the separate Apptainer
sidecar with the scientist session, public record destinations, private WMA
checkout, history, and private scratch/transcript directories. The launcher
controls these mounts. The runtime launches Claude from a separate empty
directory with `--bare`, empty setting sources, disabled slash commands and
disabled built-in tools. `--strict-mcp-config` exposes only the WMA probe
broker; the scientist's `CLAUDE.md`, project skills and hooks are not loaded.

The receipt records distinct checkout digests:

- `awm_checkouts`: public scientist protocol/client checkout.
- `wma_private_checkouts`: private WMA runtime/skill checkout.

`audit_receipt` checks frozen runtime identity, sidecar completion, verdict
presence, and exposure of WMA skills/transcripts in the scientist task. The
new decision/action records complement that audit; they do not retroactively
change how old verdicts or results are scored.

## Enforced probe I/O boundary

Built-in model backends require `awm/wma/isolation.py`. Claude's API process
retains the connection and credentials needed to call the model, while every
local command runs in a separate child with Linux Landlock filesystem rules,
seccomp syscall restrictions and a clean environment. The child cannot make
IPv4, IPv6 or Unix-domain sockets, read the host's process environment, or
recover dataset access by changing `HF_HOME` or calling a loader library.
Its children inherit the restrictions. New sessions/process groups, namespace
creation and `clone3` are blocked; normal child creation remains available
through a checked `clone` path. The broker kills the whole probe process group
on completion or timeout.

The file exports are explicit, bounded copies rather than full directory
grants:

- The private frozen current card, its matching lock and preflight report,
  and the WMA skill files. A newer live copy of the current card is not read.
- The locked script/config file, if it is a regular approved code/config type
  under the scientist session and matches its frozen SHA256. The known PTB
  `/home/ben/task` path is mapped to the private sidecar's session mount.
- Up to 128 earlier closed experiment cards. Open cards and later card numbers
  are excluded. `list_inputs` is their index; the live session `memory/index.md`
  is omitted because it may already summarize future results.
- Up to 512 card/index files from the operator-supplied historical run roots.
  Only the roots explicitly linked by the operator are resolved; inner
  symlinks, their parent directories and raw evaluation/log/data files are
  not exported. Any history truncation is recorded.
- Additional exact files explicitly exported by the trusted harness through
  `Brief.extra["probe_files"]`. Model-authored card paths and `allowed_roots`
  do not grant arbitrary access.

Each exported file is capped at 16 MiB and the complete input snapshot at
64 MiB. Copies record SHA256 and reject detectable concurrent mutation or a
locked-script hash mismatch. The broker offers `list_inputs`, `read_file`,
`run` and `write_result`; only the latter publishes a verdict/comparison.
The backend, rather than the model, stamps the isolation method and input
inventory on the result. Exported textual evidence is not semantically
decontaminated: an answer already copied into an authorized card is still in
that supplied text.

The current runtime supports CPU/static probes using Python's standard
library and a small OS command/library allowlist. It does not expose installed
ML packages, model weights, GPU devices, benchmark evaluator internals, raw
dataset caches or online downloads. Missing capability must be reported as
an untested claim, not as evidence that a training proposal is invalid. Dynamic
ML probes require a separately designed controlled interface; this change does
not claim to preserve that previous capability.

Probe limits include at most 60 seconds wall time, 60 seconds CPU time,
512 MiB address space, 128 open files and 1 MiB per output file. Inherited
`RLIMIT_NPROC` stays unchanged: imposing 128 on an entire shared UID prevented
even the self-check from forking, including when PID namespaces hide the other
jobs counted by Linux. The broker instead checks its own non-escaping process
group every 50ms and kills that group if it exceeds 128 threads/processes.
This supervised limit can transiently overshoot; it is not an atomic cgroup
pids limit. Inherited stricter limits remain in force, and production resource
isolation still depends on the scheduler/container quota. Filesystem and
network restrictions are unchanged and still inherited by all probe children.

Landlock ABI 3 or newer and seccomp support are required on Linux x86_64 or
aarch64. Every review runs a sandboxed Python self-check before the API call.
The built-in Codex backend has no corresponding isolated tool adapter yet and
fails closed. Unsupported kernels, missing CLI flags, failed self-checks and
invalid exports do not fall back to unrestricted execution. The surrounding
scientist protocol may still record WMA unavailability and proceed; this is
different from accepting an unisolated review.

Local tests execute real kernel canaries for indirect library/cache reads,
symlink escapes, network/process access, inherited child restrictions, resource
limits, immutable input selection and the MCP transport. These passed in the
development environment. No production-container CLI/API smoke or new Slurm
experiment was run for this change; deployed CLI flags, kernel support,
resource quotas and a real model/broker round trip remain an acceptance gate
before using the new runtime in an experiment.
