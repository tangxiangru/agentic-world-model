# Run the locked command and keep its actual exit evidence

After the normal card/check/lock steps, use:

```bash
awm exp_protocol run --dir /absolute/session exp-01
```

The wrapper runs exactly `setup.command.argv` in its declared absolute cwd,
with declared string-valued `setup.command.env` overrides. There is no shell
interpolation or interactive stdin. For a shell pipeline, put it in the pinned
script and declare that script in argv; do not smuggle shell text into an argv
element. The main script must be an absolute existing path hashed by the lock.
Every declared data file must have an absolute path and still exist with its
locked hash; v2 did not preserve the caller cwd needed to resolve relative data
paths safely. Imported scripts,
extra config files and the ambient environment are not fully pinned by v2.

No card is created, re-locked, closed or selected automatically. Each invocation
prints a private `memory/attempts/exp-NN/UNIQUE_ID` directory. It contains:

- `launch.json`: locked plan/card/source identities, exact argv/cwd, fresh
  preflight and declared environment names/hash (not an ambient credential dump).
- `process.json`: the actual direct-child PID, its new process group and, when
  readable, Linux boot/start-time identity.
- `stdout.txt` / `stderr.txt`: this invocation's streams. The wrapper does not
  overwrite the scientist's preexisting log files.
- `finish.json`: the observed wait return code, observer outcome, after-run
  integrity and separate artifact evidence. Missing/truncated finish is unknown,
  not success or proof the child stopped.

The wrapper stays in the foreground. Give the calling tool a sufficiently long
timeout and inspect these logs from other work when useful. Do not end the
scientist session while scientific work is alive. The child must wait for its
scientific descendants: a launcher exiting0 does not certify detached children.
SIGINT/SIGTERM are forwarded only to this new child group; a nonresponsive child
gets a bounded graceful wait before forceful termination. SIGKILL/observer death
cannot produce a guaranteed final record or cleanup.

An OS lock prevents concurrent guarded invocations of the same card. Its sidecar
file is not a liveness indicator. An unresolved prior attempt also blocks rerun,
even after the wrapper's OS lock was released; the error reports current birth-
identity evidence when available. Investigate the owned producer/descendants,
close the failed/unknown experiment honestly, and use a new card for a justified
retry. Do not erase attempt records or kill a process merely because a PID matches.

## Optional fresh output evidence

Before locking, declare:

```yaml
setup:
  output_dir: /absolute/session/checkpoints/exp-01-attempt-a
  execution:
    output_evidence: fresh-directory
```

The output parent must exist; the output itself must **not** exist. The wrapper
reserves it exclusively before launch and never deletes or empties an old path.
Your locked command must write its outputs there. After exit, the wrapper records
regular-file identities and SHA256 hashes in that fresh namespace, with bounded
file/byte inventory. Symlinks, hardlinks, replacement directories, changing files,
empty output and inventory errors cannot be reported as verified evidence.

This is an observed fresh namespace and file snapshot, **not** a model-format,
serving-config, contamination, numerical-correctness or PTB-validator verdict.
A failed command can leave a useful partial snapshot. Preserve both the snapshot
and its nonzero exit. Independently validate required artifacts before dependent
work. Copying an old model into a fresh directory does not make it a new training
result; lineage and the actual command still matter.

Without the optional mode (or with `output_evidence: unverified`), legitimate
existing-output/eval-only commands remain runnable, but artifact identity is
explicitly unverified. Do not relabel file existence or a zero exit as proof.

## Close and review

Read the exit record and actual logs. A changed locked plan, card identity,
lock or source after launch is reported separately from the real child exit
and makes the wrapper unsuccessful. Writing result/conclusion sections after
observing the result does not change the locked plan and is allowed; keep
re-lock/close calls outside the command until the wrapper finishes.
Failure, interruption or missing semantic validation must not be hidden behind
the command's scalar score. Fill result/conclusion honestly, then run normal
`close`; opted-in deferred comparators retain their own receipt requirements.
The execution record does not satisfy those requirements or release the Stop hook.
After interruption, an actual observed child exit is timestamped when available;
the partial output files remain in place, but no fresh snapshot is attempted
merely to delay interruption. Missing/partial inventory stays unverified.

The wrapper sets `AWM_EXP_ATTEMPT_ID` and `AWM_EXP_ATTEMPT_DIR` for optional
producer-side records. Do not override them in the card. Those environment values
are provenance hints, not proof that arbitrary code consumed the promised inputs.
Other processes and free-form commands can bypass these tools; report coverage
of instrumented commands rather than claiming universal execution enforcement.
