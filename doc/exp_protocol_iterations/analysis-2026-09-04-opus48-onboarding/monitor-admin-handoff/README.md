# One-shot administrative monitor handoff

This preserves the existing detector through its real process exit. It is not
a new Slurm scheduler or an early counter reset. Helper3676924 was armed at
16:01 UTC against detector3564003 and verified live waiting on its pidfd.

Before proceeding it requires the exact38 watched IDs, ready threshold8,
exactly the17 retired IDs in CANCELLED state, and all17 immutable receipt-backed
administrative harvests with no result/score. Extra terminals or changed state
leave the event for planner review. It never kills the old detector. On the
expected event it archives previous-ready-state.json and launches the same
provided monitor on21 remaining IDs with poll3600/threshold8, verifying the
new process and first written state. It does not touch Slurm or invoke Claude.

Original live directory: `/tmp/exp-protocol-admin-handoff.sTGNfhBs`.
waiting.json is an arming snapshot, not completion evidence. Read completed.json
and verify its new live PID before updating the handoff state. error.json means
inspect the same processes/logs; do not automatically start another copy.

Eight tests passed: predicate rejection of unknown events, actual harvested
receipts, and a real pidfd/process lifecycle using synthetic detector scripts
which never query Slurm or modify the real monitor state. Neither the test nor
arming proves that the later real handoff has completed.

## Real handoff verified

At16:16:40 UTC old detector3564003 naturally exited after recording the17 known
administrative terminals. Helper3676924 preserved previous-ready-state.json,
rechecked the harvested receipts, launched detector3684437 and verified its
first watching state on the21 remaining IDs, threshold8/poll3600. completed.json
records the actual transition. Main independently confirmed both old/helper
processes absent and the new process live with the exact arguments. No process
was killed early, no Slurm mutation occurred, and no clean result was created.
Next nominal detector tick17:16:40 UTC. The helper is finished, not another live
monitor. New terminal events still require normal harvest/validation/trace review.
