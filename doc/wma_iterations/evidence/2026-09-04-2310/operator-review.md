# Operator review — first clean BFCL raw comparator

The inspected reconcile preview identified BFCL raw c53r02/job 92182 as the
only new harvest. The PTB result query returns a complete result with accuracy
0.94, empty validator issues, and empty automatic judge flags. The receipt
resolves to cell c53r02, manifest
`experiments/posttrainbench/wma-crossbench-opus48-r05-bfcl-raw-x4.yaml`, spec
`doc/spec/2026-09-04-wma-opus48-crossbench.md`, AWM source
`225bd584f35ecaf0ec3fac4c2fb02d946030180c`, PTB source
`e62036f0c244995a6f45496522d3310b239383c6`, and the canonical result path in
`bfcl-summary.json`. Reconcile then harvested this bundle and refreshed only
active in-flight mirrors.

c53r02 is the first primary-clean BFCL raw result. It used Opus4.8/high/200k,
Gemma3-4b, scientist cost $20.53324425, agent time 07:25:19, and one-H100
allocation 07:35:54. Required PTB lookup, general, data/model and API judges
are clean; judge costs are unavailable.

The matched protocol-only arm remains 91/88/94/90 = **90.75% ±2.50pp sample
SD (n4)**. Raw primary is **94% (n1)**, so P−R is descriptively −3.25pp and
cannot identify a protocol effect. A sensitivity read over all three scored raw
completions is 94/91/92 = 92.3333% ±1.5275pp, with c53r03/c53r04 still carrying
their lifecycle anomaly flags; its P−R is −1.5833pp and remains sensitivity
only. c53r01 remains incomplete rather than a zero.

Scientific totals are now 101 PTB-complete and 98 automatically clean; the new
Opus4.8 study contributes 12 complete and 9 clean. No WMA policy promotion is
possible from a raw/protocol contrast, and raw n1 is below any formal gate. No
skill, scorer, guard, runtime or frozen treatment changed.

At 23:10:44 UTC ownership and routes are clean, 16/16 H100s are allocated, 16
jobs are RUNNING and 29 are safely PENDING: 28 priority waits and one resource
wait, including validation-only job 92312. GSM8K multi-self w58r04 started by
normal backfill. Pending remains above the 24-job replenishment trigger and the
staged S0 arm will add four only after validation, so no redundant work is
submitted. S0 remains blocked on 92312; no job was cancelled or altered.
