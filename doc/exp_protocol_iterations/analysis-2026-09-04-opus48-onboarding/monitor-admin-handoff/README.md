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
