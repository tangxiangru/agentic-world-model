I'll start by reading the brief to understand what the four-cell guard review requires.
I've completed the read-only review of all four cells. Nothing was modified; no plan file was created.









---

## Group notes for the planner (outside the report delimiters — not a cell report, not a gate decision)

**Scope.** These four cells are an incremental addendum. g01s02, g01s05 and g01s07 are unread here, so **nothing below decides the eight-cell guard gate or E's saturation condition**, and none of it belongs in frozen window03's NEW aggregate.

**Descriptive accuracies only.** Four-cell mean 0.71380 (0.6960 / 0.7255 / 0.7081 / 0.7255). With window03's three guard cells the seven-cell guard mean is 0.72555. The v3 baseline clean 14-cell mean is 0.688563, so the −0.03 score guardrail (0.658563) is cleared by every individual cell. The window03 NEW no-protocol control mean was 0.757240 on n=5. All three variants remain distinct; no promotion claim.

**1. E fails on all four cells, and the aggregation convention decides how badly.** Post-exit idle over *every* background process: **1.10 / 1.73 / 0.75 / 0.69 h** (g01s01 / g01s03 / g01s06 / g01s08) against a 0.15 h target. Under the narrower "training and sampling runs only" convention used for g01s04 in window03, the same cells read **0.14 / 0.61 / 0.12 / 0.34 h** — 2 of 4 would pass. Per-event maxima (secondary only) are 0.27 / 0.50 / 0.18 / 0.25 h. The planner's window03 check ("cannot be declared passed using a per-event maximum") applies here with a second edge: the *process set* also has to be fixed before the number means anything. My tables use exact process-exit stamps (`W903 …` destroy_process_group lines in each eval log) and 60-second `system_monitor.log` GPU windows for trainers; bounds and overlap exclusions are stated per row.

**2. The E mechanism is now identified precisely, and it is in the protocol text.** The `run_dies_with_the_session` reminder is reprinted at every lock and offers two branches as equals: *"`sleep 900; tail -n 3 <log>` repeated, or `while [ ! -f <out>/config.json ]; do sleep 300; done`"*. Across the block, post-exit idle averaged **≈0.11 h per clock-waited process** and **≈0.018 h per condition-waited process**; g01s08 shows both branches in one session (0.064 h vs 0.011 h per event). Two corollaries the current E rewrite does not cover: (a) the largest single events are **dead runs** (g01s03 0.445 h, g01s08 0.250 h) that a one-line `ps`/`nvidia-smi` — which both scientists wrote *immediately after* discovering the corpse — would have caught; (b) more than half the total idle in three of four cells sits after **evaluations**, which finish in 100–240 s while the scientist sleeps 400–1700 s. E's text mentions runs, not evals.

**3. A is not saturated on the guard baseline.** g01s06 shipped `do_sample: true, top_k: 64, top_p: 0.95`, made no decode measurement in 8.44 h, had the `"Default sampling parameters have been overridden"` line in two of its own eval logs, and printed `do_sample: true` from a checkpoint twice without acting. The strict-block count is 3 greedy / 1 sampled. Separately, g01s08 exposes a definitional hole in A v2's observable: its first post-SFT eval graded a *broken* artifact, so "first post-SFT eval → measured decode choice" is 1.06 h or 0.115 h depending on which eval starts the clock.

**4. D is confirmed twice more, with a worse variant.** g01s01 lost a completed 649-step run (0.85 h); g01s03 lost a run at step 780/1562 on an *intermediate* save (0.60 h GPU + 0.445 h idle). Both are the exact `GenerationConfig is invalid: temperature … do_sample=False` failure. The cells that were immune (g01s08, and g01s03's own exp-03) used symlinked greedy directories; g01s03 still hit it one card later because its *SFT* parent had been saved greedy. So the D check should read the **training parent's** `generation_config`, not whether greedy was adopted — consistent with the window03 note.

**5. B v2 has one very strong cell and three that pre-empted it.** g01s06 paid 0.65 h across four offline-sampling defects, one of which produced a measured pass rate of 0.0025 against the same checkpoint's 0.633 on the benchmark. g01s01 wrote `add_special_tokens=False` and `stop_token_ids=[1,106]` into its sampler before launching. A B screen needs to expect a non-zero rate of cells that already know this. One mechanism in g01s06 — `top_p=1.0` causing incoherent, non-terminating Gemma-3 sampling — is not in B v2's text and is so far single-cell.

**6. P1 needs a sharper observable than "flat loss".** All four cells ran RFT from the checkpoint that generated the samples; first-20-step losses were visible in all four. Outcomes: **+6.7 pp** (g01s06 round 1), **+1 item** (g01s03), **−2.0 pp** (g01s01), **−2.0/−1.8 pp** (g01s06 round 2), **−3.3/−2.8 pp** (g01s08). g01s06 round 1 and round 2 have indistinguishable loss curves and opposite signs; the discriminator is whether the **parent is itself an RFT product**, not the flatness. Fitted-parent RFT hours in this block: ≈2.4 / ≈2.4 / ≈2.2 / ≈2.0 h per cell. g01s08's exp-08 shows the same flat-loss-no-gain shape on a plain SFT continuation, so a `family: rft`-scoped entry would describe only part of it.

**7. G remains untestable from the protocol arm.** 0 RL launches, 0 zero-grad lines, 0 GRPO cards across all four. Every rejection is a time/risk argument (three DPO, one GRPO), never a belief about TRL. Note two tooling artifacts: g01s06's `RL launches=1` in the facts output is a heredoc patching `rft_sample.py`, and the timelines' `first_rl` stage is not an RL launch in any of the four cells.

**8. New candidate, ≥2 source cells, one allowed surface — the graded 10-shot prefix.** g01s08 trained a full 1.95 h epoch on single-problem prompts, scored 0.060, and recovered +64.0 pp with 375 steps whose only change was the prompt distribution (2.2 h carded); every preflight passed, because the check inspects **targets** while the mismatch is in the **prompt**. g01s01 measured the residual at −3.0 pp with 6% prefixed rows. g01s03 (20%) and g01s06 (20%) pre-empted it and paid nothing. 4/4 cells touch it, 2 pay. Readable screen observables: `hit_token_cap` and `ends_with_answer_line` on the first post-SFT eval; the prefixed-row share. Guardrail: block mean ≥ protocol pool mean − 0.03, plus an explicit non-instruction clause (g01s08's own `alternatives_rejected` rejects training *only* on 10-shot prompts).

**9. Second candidate, 2 cells / 3 instances — `comparator_same_protocol` on within-card head-to-heads.** g01s03 exp-06 and g01s06 exp-05/exp-06 all override it because the comparator file is by construction an output of the card. Today the only discharge is an override, which pollutes the `n_overrides` signal H is scored on.

**10. Measurement-transfer numbers, with a sign flip.** Own-prefix vs official n=1319: **+3.0 pp** (g01s01, n=500), **−0.55 pp** (g01s03, n=500), **+2.2 pp** (g01s06, n=1000), **+2.25 pp** (g01s08, n=500). Same-weight repeated reads: **2.7 pp** (g01s01), **0.67 pp** (g01s03), **0.0 pp** (g01s08). Both spreads are larger than most card-level deltas and neither is uniform in sign or size — so "the front of the test set is easier" and "greedy is deterministic" are both too strong as stated. In-session `max_connections` was 8 or 16 everywhere against the official 2, so concurrency is confounded with n and cannot be isolated from these traces.

**11. Two tooling corrections for the facts/timeline readers.** (a) g01s06's `inspect logs: 0 est_n=[]` is an artifact of `INSPECT_LOG_DIR` redirecting logs to `task/eval/logs_expNN/`; the nine logs are listed with sizes in `status.json`. (b) Those sizes calibrate the heuristic: **≈42 KB/sample**, not 44 KB, so `est_n` runs ~5% low (489→500, 502→500, 490→500). Card `evaluation.protocol.n` and `evaluate.py`'s own stderr give exact counts and were used throughout. Also: `n_relocked` counts **cards**; the relock-**event** counts here are 2 (g01s01, two cards × one event), 1 (g01s03), 0 (g01s06), 1 (g01s08).

**12. Guard's own criteria are clean 4/4.** Zero stop-hook blocks (no `memory/.stop_hook.json` in any bundle, and dotfiles are demonstrably harvested), zero false blocks, zero locked-open cards at stop, zero training or evaluation lost at session exit, zero tool timeouts, lock-before-launch 3/3, 3/3, 3/3, 5/5, and all 100+ long runs launched with `setsid nohup … < /dev/null &`. Unused budget: 1:57 / 1:56 / 1:34 / ~2:04.

