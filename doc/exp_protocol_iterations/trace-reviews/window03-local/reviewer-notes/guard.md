I'll start by reading the guard brief to understand what's being asked.
I've finished the read-only trace review of all three assigned guard cells. All reports follow; nothing was modified.







---

## Cross-cell notes for the synthesis (not a cell report)

These are the three guard cells only. **n=3 is a partial block; nothing below is a saturation finding or a promotion result, and c01r03 is excluded throughout.**

| field | g01r01 | g01r02 | g01s04 |
|---|---|---|---|
| official accuracy | 0.7104 | 0.7779 | 0.7354 |
| hours used / unused | 7.59 / 2.42 | 8.88 / 1.12 | 7.62 / 2.38 |
| h to first **real** launch | 0.42 | 0.33 | 0.63 |
| protocol hours (corrected) | 0.12 | 0.11 | 0.15 |
| cards / closed / open | 9 / 9 / 0 | 7 / 7 / 0 | 5 / 5 / 0 |
| overrides / relocks | 0 / 0 | 0 / 2 | 0 / 1 |
| pitfalls_cost_h / hits | 0.72 / 4 | 1.60 / 4 | 1.60 / 6 |
| greedy shipped / gain | yes / +3.3 | yes / +10.7 | yes / +10.67 |
| post-SFT-eval → measured decode choice | 0.055 h | 0.13 h | 0.03 h (0.32 h to graded confirmation) |
| largest eval n | 1319 | 500 | 800 |
| paired statistic used | no | no | **yes (McNemar z=0.40)** |
| RL launched | no | no | no |
| **stop-hook blocks** | **0** | **0** | **0** |
| run lost to session end | no | no | no |
| **E idle (sum)** | **≈0.145 h** | **≈0.24 h** | **≈0.48 h** |

Interim only: guard n=3 mean 0.7412, sd 0.034, against the v3 baseline 14-cell clean mean 0.688563. Not a promotion result; the block is 3/8 and the sd is larger than the difference the design can resolve.

Points I would want the synthesis to carry:

1. **The guard's own criteria are clean 3/3.** Zero hook blocks, zero false blocking, zero open locked cards at stop, zero runs lost to session end, zero background scientific work killed at exit. All three launched every long run with `setsid nohup ... < /dev/null &` as the pitfall text prescribes. Separately, CLI 2.1.219 **backgrounds** a Bash call that exceeds its tool timeout rather than killing it (g01r01 06:58:30Z, g01s04 14:53:40Z), so the pilot-era "tool timeout kills the process group" mechanism did not reproduce in this block.
2. **E does not look saturated on these three cells** (0.145 / 0.24 / 0.48 h; 2/3 over the 0.15 h line), which differs from the window-02 baseline reading of 5/5 under 0.15 h. Two caveats before anyone acts on it: the target's aggregation convention (per-cell sum vs per-event max) changes g01r01's verdict, and the two failing instances have *different* mechanisms — a clock-sized `sleep 1700` overshooting a finished run (g01r02, 0.17 h) and a condition-wait on a log string from a crashed process (g01s04, 0.40 h). Only the first is what E's current text is aimed at. The remaining five guard cells should settle this.
3. **A v2's primary observable is met with large margin in 3/3** (0.03–0.13 h) with zero post-choice unmeasured-sampling decision cards. Consistent with the window-02 conclusion that this observable is saturating on the guard baseline.
4. **D is a parent-choice trap, not a greedy-adoption trap.** All three adopted greedy; only g01r02 made the greedy directory a training parent, and it lost a completed 1936 s run. The other two kept greedy in symlinked variant dirs and trained from unpatched parents — structurally immune. The D screen should read *whether the next stage's parent carries the greedy config*, not *whether greedy was adopted*.
5. **P1's screen observable may be unreadable.** g01r01 never printed a first-step loss (all monitoring is `tail -c 150`); g01r02 shows the flat signature twice, once before a gain and once before a loss; g01s04's mixed stage shows a real descent. Reading "hours after a flat first-20-step loss" out of a trace requires the scientist to have printed the head of the log, which one of three did.
6. **Measurement transfer is a live, uncovered gap.** g01r01's five in-session full-1319 reads of the shipped artifact span 0.7187–0.7202 and the official grader returned 0.7104 — 0.83 pp below the whole band. The visible difference is concurrency (`--max-connections 32` in-session vs `max_connections: 2` official). Prefix bias is separately confirmed against official numbers: +3.8 pp at n=500 (g01r02), +1.2 pp at n=800 (g01s04), but **not** uniformly at n=150 — g01r01 shows the first-150 subset favouring one checkpoint by 5.3 pp and disfavouring another, i.e. the subset effect is checkpoint-dependent, which is stronger than "the front is easier."
7. **H is confirmed 3/3 with zero overrides.** Twelve non-training cards, twelve `setup.data` entries that are not training data, and one outright fabrication: g01r01 generated a 150-row placeholder jsonl and set `mixture_weight: 1.0` to get past `data_files_exist` + `data_n_examples_match` (L1312/06:44:58Z). The cost is record corruption, not hours, and it does not appear in `n_overrides`.
8. **G is still untestable from the protocol arm.** Three cells, zero RL launches, zero zero-grad lines. The one explicit rejection (g01s04) is a time-and-setup argument, not a belief about TRL. A screen built on `trl_grpo_gemma_zero_gradient` would need a cell to attempt GRPO first.

Only one item here rises to a *new* single-surface candidate grounded in ≥2 cells: a `pitfalls.yaml` entry (check: null) on greedy non-reproducibility and grader concurrency (3/3 cells). Everything else is either confirmation of an already-frozen candidate (B v2, C v2, D, H, P2, P3), a wording correction inside an existing one (E), or an observation.

