# Independent E5 forward behavioral review

Reviewed 2026-09-04, using published SKILL.md/execution-records.md and the CLI.
No E5 implementation/tests were read to choose these fixtures. No repository
files/index/commits, models, GPUs, Slurm, network, downloads or packages were changed/run.
All fixtures, generated records and this report are under:
`/tmp/e5-forward-review.35fG5B`.

## Outcome

No false success or unexpected launch block was observed in these six real
CPU-only attempts. The documented unresolved-attempt barrier worked, and useful
partial output remained distinguishable from completed work. One conservative
interruption evidence omission remains below; this is not scientific validation.

| Card | Independently chosen operation | Observed record/result |
|---|---|---|
| exp-01 | Sum three integers with declared FACTOR=7 and literal dollar/semicolon argv | Sum 42; label exactly `label $FACTOR ; not a shell`; child exit 0; fresh result hash observed |
| exp-02 | Write two-record partial sum then exit 23 | Child exit 23 retained alongside partial.json snapshot; never labeled scientific completion |
| exp-03 | Send SIGINT to owned foreground wrapper after child readiness | External wrapper rc 130; child acknowledged signal 2; wait rc 130; finish retained; child reaped |
| exp-04 | SIGKILL observer after readiness, then attempt same-card retry | External wrapper rc -9; no finish; retry rc 2 with matched live birth identity; no duplicate launch |
| exp-05 | Predeclared aggregation control | Not launched; closed `not_run` when the actual same-producer retry was chosen |
| exp-06 | Inspect an existing static reference without fresh-directory mode | Exit 0, artifacts explicitly unverified; no invented training-data requirement |
| exp-07 | Retry exp-04's unchanged producer under a new card/output | Completed its 20-second work, exit 0, ready/complete files inventoried; old unknown attempt untouched |

Both declared argv and environment were consumed as documented; no shell
interpolation occurred. Launch records contain override names/hash rather than
an ambient-environment dump. Card/source/data/plan identities are inspectable.
The retry kept the same interruptible.py source; only its new output path/card
changed. The separate aggregation control was not misrepresented as that retry.

## Concrete evidence omission on the interrupted path

`memory/attempts/exp-03/990913c1babd40738072859f19ab82ad/finish.json`
records observed_returncode 130, status interrupted, and finished_at, but lacks
child_exit_observed_at. Its artifacts are unverified and contain no file inventory,
although outputs/sigint/ready.json and interrupted.json remain available.

By contrast the normal nonzero exit 23 record retains both child_exit_observed_at
and the useful partial-file hash. This is a conservative omission, not a false
artifact or success claim. Another scientist cannot obtain the interrupted child's
observed-exit timestamp or a wrapper-produced post-interruption snapshot from that
finish file alone. The general documentation's after-exit inventory description
therefore does not describe this observed interruption path completely.

## Observer-death investigation and cleanup

exp-04 wrapper 3055687 was killed; direct child 3055689 remained in state S.
Same-card retry returned 2 and named unresolved attempt
786fb0d8d07e49d1bbd8af5be9995b57 with the same child birth identity still present.
Independent control_attempt.py verified process.json, Linux start_ticks, command
line and process group before sending SIGTERM only to that owned group 3055689.
The child disappeared. Its wait exit remains unknown: no finish was fabricated.
operator-observations.json preserves the observed driver output; the producer's
own interrupted.json preserves its signal acknowledgement, not a wait result.

Final exact-PID checks found all six launched direct children gone:
3051424, 3051427, 3051430, 3055611, 3055689, 3058810. No unrelated process was signaled.

## Consumers

Every card passed check/lock without overrides. run did not auto-close.
Normal close accepted completed, failed, killed/unknown and not_run conclusions
when recorded honestly. Final collect: n_cards 7, n_closed 7, n_locked 7,
n_locked_open 0, n_relocked 0, n_overrides 0, preflight_fail 0; no accuracy/adoption.
Index status `closed` is lifecycle status, not an assertion that exp-04 completed.
Index/collect do not surface E5 attempt IDs/exits themselves; the card failure text
and attempt directories remain necessary for interpreting these cases.

## Exact attempt anchors

All directories below are under this session's memory/attempts/:
- exp-01/faaf193d48114bd2890f6fb87cbdcda5
- exp-02/6c907c1ca31c423a9cf8d575fca27820
- exp-03/990913c1babd40738072859f19ab82ad
- exp-04/786fb0d8d07e49d1bbd8af5be9995b57
- exp-06/d52f954dc2ca41b28b9e9a701a07bfd6
- exp-07/f50bebb4aade4556a8e5465093f98c9b

## Invocation/reproduction

The following identify the actual CLI/runtime; imports were verified to resolve
to the candidate checkout, not the original environment's repository:
```bash
e5_python=/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/python
e5_cli=/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/bin/awm
e5_repo=/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$e5_repo" "$e5_python" "$e5_cli" exp_protocol run --dir /tmp/e5-forward-review.35fG5B exp-01
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$e5_repo" "$e5_python" /tmp/e5-forward-review.35fG5B/control_attempt.py exp-03 SIGINT
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$e5_repo" "$e5_python" /tmp/e5-forward-review.35fG5B/control_attempt.py exp-04 SIGKILL
```
The same CLI performed index/new/check/lock/close/collect. The published `awm`
entrypoint works; my initial unrelated `python -m awm` assumption did not (there is
no awm.__main__). This is not an E5 regression.
These exact cards are now closed; reproduce in a NEW temporary session with
paths/new output namespaces updated before lock, rather than erase these records.

Tested execution.py SHA256:
ded858f8147a7fba0eae21d1ad66ca1484ebb49e0c60d84bd04f7abb53542b2f
Reviewed execution-records.md SHA256:
db1e1c9ce5f3e553b466d9655cbc3792feedaf35522fc1b10fdf20fd68fc139b

Scope gaps: no detached descendants, PID reuse, ambient-environment reproducibility,
huge inventories, malformed receipts, semantic model artifacts or PTB completion
were tested. Absolute paths were used throughout; no cwd-relative binding claim.
