I'll start by reading the brief.
I have all the evidence I need. Let me record the plan file (the only write plan mode permits), then deliver the reports.
Read-only review complete. All three cells analysed from their own bundles; no files written, no jobs touched.







---

## Cross-cell notes for synthesis and pending-block triage

These are for the planner, not part of any single report. Only items grounded in ≥2 of my three cells are put forward as candidate evidence; the rest are marked observation. `c01r03` is calibration and is excluded from the NEW arithmetic below unless stated.

**1. Candidate E is not saturated. This is the load-bearing finding.** Measured as the brief defines it — process already exited until the next useful command, not total waiting time:

| cell | E total | largest event | vs 0.15 h target |
|---|---|---|---|
| c01r07 (NEW) | 0.482 h | 0.296 h | 3.2× over |
| c01r08 (NEW) | 2.045 h | 0.378 h | 13.6× over |
| c01r03 (calibration) | 0.398 h | 0.072 h | 2.7× over |

NEW-cell mean over my two new cells: **1.26 h/cell**, 0/2 meeting the target. I have only 2 NEW cells here, so this cannot settle the ≥7/8 guard condition on its own — but it is a direct counterexample to "the E target is already met at baseline," and the mechanism is identical in all three: a fixed `sleep` sized to a pessimistic ETA. Per decision 09-03 08:45 the E block (jobs 91064–91067) was to be withdrawn *only if* the guard shows ≥7/8 already under 0.15 h; on this evidence the withdrawal should not be pre-committed before the guard cells are read. Method note: my instrument (`task/system_monitor.log` 60 s GPU sampling + NCCL `W903` shutdown stamps) is more sensitive than reading sleep durations off the timeline, so earlier windows measured with the coarser method may have under-reported E.

**2. Candidate B: 3/3 cells, ~3.4 h, three distinct sub-mechanisms.** Orphaned engine surviving `pkill` (c01r07 L3867, c01r08 L3596, c01r03 L6211 — 66–68 GB held each time, all needing `kill -9`); over-scoped generation launched without a rate estimate (219,784 / 129,288 / 164,838 prompts, ETAs 4:56 / 1:39 / 2:23); and the weak sampled stop (c01r07 83.4% `length`, c01r03 pass@1 0.11→0.81). The third is the expensive one (0.84 h + 1.87 h) and c01r08 shows the cheap immunity: **filter kept samples on a parseable `ANSWER:` line, never on `finish_reason == "stop"`**. That is one concrete, testable sentence for the existing B v2 entry, grounded in all three cells.

**3. Candidate D: 2/3 cells, 2.01 h, identical stack trace.** c01r07 0.67 h (L4965), c01r03 1.34 h (L5675). Both fixed themselves the same way afterwards; neither had it beforehand. The check D proposes (`parent_generation_config_valid`) would have fired in both. Note the reconciliation item in §5 of the c01r03 report — my per-cell figure is larger than the window-02 apportionment.

**4. Candidate A: the observable is cheap; the payoff claim needs re-scoping.** Grader observable identified in 3/3 cells (c01r07 at +0.06 h, c01r03 at +0.47 h, c01r08 at +0.07 h). Acted on in 2/3. The cell that ignored it, **c01r08, tied for the highest official accuracy in the window at 0.778620**. So on this evidence the entry's value is not "greedy is worth 7–16 points" but "sampling makes your reads disagree with the grader's": own-vs-official full-test spread was 0.99 pp under sampling (c01r08) and 0.23 pp under greedy (c01r03). Separately, greedy does **not** buy read reproducibility across differing `--max-connections`/`--gpu-memory-utilization`: c01r07 measured 0.720 vs 0.660 on the same 100 items with 43/100 identical generations (L5721). Both facts fit A v2's "grader observable" framing better than its score framing.

**5. Candidate C: one clean inversion, one collapse, and a design conflict.** c01r08 inverted the ckpt-180/ckpt-120 ranking between n=300 (+6.0 pp) and n=700 (−0.4 pp). c01r07's shipping delta collapsed from +3.67 pp @300 to +0.8 pp @500 with a paired count of 38W/34L. Both cells produced the paired/repeated evidence C v2 asks for **spontaneously and after the decision**, not before it — the screen should read *when* the paired read happened, not just whether it exists.

**6. Design conflict to record before P3 is written.** c01r08's `grpo_train.py:145-150` and `train_sft.py:177-180` copy `preprocessor_config.json`, `processor_config.json` **and `generation_config.json`** from the base snapshot into every checkpoint — the P3 fix (checkpoint loadable by vLLM, cost 0 h in 3/3 cells) directly reinstating the A problem (stock sampling config shipped). A P3 pitfalls line must except `generation_config.json` or it will manufacture c01r08's outcome.

**7. Observations, not proposals (single-cell or not a protocol surface).**
- **G / TRL eos**: only c01r08 ran RL. The mechanism is real and named in the trace (L4930-4933), but the *signature* is wrong in the ledger — `grad_norm` was 0.25–0.37 before the fix, not zero (L4903-4906). A G entry keyed on zero gradient would not have matched.
- **P1 / RFT from a fitted parent**: diminishing, not flat. c01r07 round 1 gave exactly 0.0 pp @300, c01r08 called SFT round 2 "+2 pts — SFT had saturated", c01r03 got +0.61 pp on 1319. No first-step-loss-is-flat signature was visible in any of the three; c01r07's round-1 null had a different cause (the discarded 5× yield).
- **P2 / Gemma-3 logits memory**: 3/3, 0.02–0.13 h each, 0.30 h total. c01r08 adds a new sub-mechanism the entry does not currently cover: liger installed into the 64 MB root overlay **corrupted 121 already-installed files** and took down `from trl import GRPOTrainer` an hour later (L5059, L5238).
- **P4 / pricing the remaining time**: all three stopped early with 1:16, 1:34 and 1:38 left and no process running. In c01r08 the last 2.5 h contained 1.56 h of pure post-exit idle, so "no time left to do anything useful" is not what the trace shows.

**8. Provenance flag, not a claim.** `c01r03` and `c01r08` have byte-identical official accuracy `0.778620166793025` = **1027/1319**, from unrelated recipes (SFT+RFT, greedy, no RL vs SFT+RFT+GRPO checkpoint soup, sampling). This is the same situation as ledger direction #12 (closed 09-03 09:17): the retained bundles have no per-item official log, so item-level identity cannot be verified either way. Recording it as a third instance of scalar coincidence in this line; it should not be read as duplication, and it does not affect either cell's eligibility.

