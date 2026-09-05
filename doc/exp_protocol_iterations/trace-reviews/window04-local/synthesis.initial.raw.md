Write is disabled in this session (read-only tool set: Read/Grep/Glob/Bash), which matches the brief — the synthesis is emitted as final text only.

<!-- SYNTHESIS window=window04-local -->

# Window04 independent synthesis — 14 NEW clean cells

**Reviewer identity.** New independent local synthesis session, not one of the five cell reviewers and not the scientist/operator. `CLAUDE_CODE_SESSION_ID=ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70`, `CLAUDE_PID=2593119`, CLI **2.1.260** (the launch record for the five cell reviewers says 2.1.259 — a small provenance difference worth recording), model `claude-opus-5[1m]`, `--effort max`, `--permission-mode plan`, `--permission-prompts none`, tools `Read,Grep,Glob,Bash`. `Write` is disabled in this session, so nothing could be written even accidentally. No model, evaluator, training run, Slurm query/action, git operation, package install or network access was performed. Read-only research reporting; every candidate, queue, release and promotion decision is the planner's, and every authorization is the user's.

**Method.** I read the meta skill (`SKILL.md`, `trace_review.md`, `metrics.md`), `doc/reference/exp_protocol_local_claude_analysis.md`, `.claude/agents/trace-reviewer.md` under the stricter local overrides, this window's `roster.json` / `launch.json` / `collect.csv` / `input-notes.md` / `planner-corrections.md`, all four focused audits, **all 14 NEW cell reports in full**, and the p00r02 calibration note as explicitly historical calibration only. I then read the directions ledger, the Round02 independent-screens spec, the E2/J/K specs, and the **frozen candidate text** (not the labels) of A `f6cdccc`, B `9f294c3`, D `8332917`, H `b52e5f2`, E2 `c6f11d8`, J `549e25a`, K `58a6992`, P1 `a4c4954` by `git diff 2f64581 <sha>`. Where a report's claim carried weight I re-derived it from the original evidence with targeted greps and line windows — never a whole trace.

Throughout, four kinds of statement are kept apart and labelled:

- **[SAID]** — what a scientist wrote in a card, summary or assistant turn.
- **[SHOWS]** — what the trace, an artifact or a retained log actually records.
- **[INFER]** — statistical or logical inference by me, with its uncertainty.
- **[COUNTERFACTUAL]** — what a candidate would have done, derived from frozen code, never observed.

Claims I verified myself against the original sources are marked **[verified here]**; claims carried from a reviewer report or focused audit without independent re-derivation are marked **[reviewer-cited]**.

---

## 1. Per-cell and per-arm header table, with measurement definitions reconciled first

### 1.0 The denominator

Exactly **14 NEW cells**, frozen at the 04:00 trigger: guard `g01r03–g01r08` (6), protocol-free strict control `c01s01–c01s04, c01s06, c01s07` (6), protocol-v3 strict baseline `p00s01, p00s02` (2). All 14 are `validator_complete: true`, `eligible: true`, `judge_flags: []`, `quarantined: false` in `roster.json`. `p00r02` is same-variant historical calibration and enters **no** mean, count or denominator. The later clean `c01s05` (0.7255496588324488) and `p00s03` (0.7103866565579985) and the placement-quarantined `c01s08` (0.7717968157695224) are **outside** this window; they appear nowhere in this report's arithmetic and are mentioned only as operations context in §5. Jobs 90817/90820/90825 were running at the trigger and are excluded by construction, not by score.

| cell | arm | job | node | awm sha / protocol_tree | setup | accuracy | items | stderr |
|---|---|---|---|---|---|---:|---:|---:|
| g01r03 | guard | 90649 | ondem-1 | `4ae3d87` / `189319d6` | `--exp-protocol --tool claude --stop-hook` | 0.8278999241849886 | 1092/1319 | 0.010397 |
| g01r04 | guard | 90650 | ondem-1 | `4ae3d87` / `189319d6` | same | 0.6906747536012130 | 911/1319 | 0.012732 |
| g01r05 | guard | 90651 | ondem-1 | `4ae3d87` / `189319d6` | same | 0.7611827141774071 | 1004/1319 | 0.011744 |
| g01r06 | guard | 90652 | ondem-1 | `4ae3d87` / `189319d6` | same | 0.7081122062168309 | 934/1319 | 0.012523 |
| g01r07 | guard | 90653 | ondem-0 | `4ae3d87` / `189319d6` | same | 0.7217589082638363 | 952/1319 | 0.012344 |
| g01r08 | guard | 90654 | ondem-0 | `4ae3d87` / `189319d6` | same | 0.7028051554207733 | 927/1319 | 0.012589 |
| c01s01 | control | 90813 | ondem-0 | `eaf5091` / *(no `skills/exp_protocol`)* | `--tool claude` | 0.7892342683851402 | 1041/1319 | 0.011234 |
| c01s02 | control | 90814 | ondem-0 | `eaf5091` / — | `--tool claude` | 0.7816527672479151 | 1031/1319 | 0.011379 |
| c01s03 | control | 90815 | ondem-0 | `eaf5091` / — | `--tool claude` | 0.7202426080363912 | 950/1319 | 0.012364 |
| c01s04 | control | 90816 | ondem-0 | `eaf5091` / — | `--tool claude` | 0.7558756633813495 | 997/1319 | 0.011832 |
| c01s06 | control | 90818 | ondem-0 | `eaf5091` / — | `--tool claude` | 0.7414708112206216 | 978/1319 | 0.012060 |
| c01s07 | control | 90819 | ondem-1 | `eaf5091` / — | `--tool claude` | 0.7202426080363912 | 950/1319 | 0.012364 |
| p00s01 | baseline_v3 | 90823 | ondem-1 | `eaf5091` / `08674f2c` | `--exp-protocol --tool claude` | 0.7702805155420773 | 1016/1319 | 0.011587 |
| p00s02 | baseline_v3 | 90824 | ondem-1 | `eaf5091` / `08674f2c` | `--exp-protocol --tool claude` | 0.7771038665655799 | 1025/1319 | 0.011464 |

**[verified here]** Every accuracy is an exact integer/1319. Arm identity is taken from the frozen manifest `paths`/`setup`, not the AWM SHA: the controls share `eaf5091` with the baseline arm but ship no `skills/exp_protocol` and carry no `--exp-protocol` flag, exactly the distinction `metrics.md` records for c01s06.

Frozen window means (descriptive, as given): guard6 **0.7354056103108415**, control6 **0.7514531210513015**, baseline2 **0.7736921910538286**. **[verified here]** I recomputed all three to the last digit from the roster accuracies.

### 1.1 Measurement definitions reconciled before any averaging

Four definitional splits make the reviewers' headline numbers non-comparable as written. I resolve each explicitly and keep both readings.

1. **First training launch.** *(a)* first GPU training command of any kind (uncarded smoke/probe included) versus *(b)* first card-matched training launch after its lock. Every protocol cell has both; controls have only (a) plus "first production run". Reviewer headline `hours_to_first_train_launch` mixes the two. Both columns below.
2. **Protocol hours.** *(a)* `awm exp_protocol` tool execution time (0.00–0.05 h) versus *(b)* card-authoring wall time (`new`→first `lock` spans + closes + bootstrap read), which includes model generation (0.05–0.40 h). These differ by up to 40×. Both columns; (b) is corrected for overlap where the reviewer measured it (g01r08: 0.40 h raw, 0.27 h of it overlapping a running n=500 eval ⇒ ~0.13 h non-overlapping).
3. **Waiting.** *(a)* raw tool time classified as waiting (4.70–7.67 h; includes composite `launch; sleep` and foreground evaluation, and mostly overlaps a live producer) versus *(b)* strict post-exit idle. These are not the same quantity and (a) must never be read as (b).
4. **Largest n.** *(a)* maximum executed developer evaluation versus *(b)* n actually behind the shipping selection. They differ in 2 of 14 cells.

### 1.2 Header table A — time, launch, protocol cost

`hours_used` is `time_taken.txt`, **[verified here]** for all 14 (the reviewers' trace-span values differ by ≤0.02 h and are not used). "left at stop" is the last `timer.sh` reading, **[verified here]** in every trace.

| cell | hours_used | h to first GPU train (any) | h to first card-matched train | protocol tool-h | card-authoring wall-h | raw waiting-h | left at stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| g01r03 | 8.442 | 0.23 | 0.51 | 0.004 | ≤0.10 | 7.24 | 1:33 (1.550) |
| g01r04 | 7.758 | 0.18 | 0.37 | 0.002 | ≤0.05 | 6.56 | 2:15 (2.250) |
| g01r05 | 7.924 | 0.22 | 0.44 | 0.016 | ≤0.09 | 6.70 | 2:05 (2.083) |
| g01r06 | 7.741 | 0.19 | 0.38 | 0.03 | 0.125 | 5.84 | 2:16 (2.267) |
| g01r07 | 8.374 | 0.19 | 0.29 | 0.01 | 0.217 | 5.21 corrected (6.21 raw) | 1:38 (1.633) |
| g01r08 | 7.455 | 0.19 | 0.27 | 0.00 | 0.40 raw / ~0.13 non-overlap | 4.70 | 2:32 (2.533) |
| c01s01 | 8.759 | 0.11 | n/a (0.50 to production SFT) | 0 | 0 | 7.19 | 1:14 (1.233) |
| c01s02 | 8.626 | 0.16 | n/a (0.25) | 0 | 0 | 7.62 | 1:24 (1.400) |
| c01s03 | 8.408 | 0.11 | n/a (0.20) | 0 | 0 | 7.44 | 1:37 (1.617) |
| c01s04 | 8.625 | 0.15 | n/a (0.40) | 0 | 0 | 7.67 | 1:22 (1.367) |
| c01s06 | 7.890 | 0.185 | n/a (0.55) | 0 | 0 | 6.85 | 2:07 (2.117) |
| c01s07 | 8.190 | 0.10 | n/a (0.18) | 0 | 0 | 6.81 | 1:49 (1.817) |
| p00s01 | 7.891 | 0.19 | 0.44 | 0.01 | 0.16 | 6.72 | 2:07 (2.117) |
| p00s02 | 7.805 | 0.16 | 0.25 (0.35 to the launch that survived) | 0.05 | 0.18 | 6.78 | 2:12 (2.200) |

Arm aggregates **[verified here]**: mean `hours_used` guard 7.949, control 8.416, baseline 7.848, protocol-installed 8 = 7.923. Mean unused budget: guard 2.053 h, control 1.592 h, baseline 2.158 h, protocol-installed 2.079 h. Cells stopping with ≥1.5 h unused and no live process: **protocol-installed 8/8, control 3/6** (two-sided Fisher p ≈ 0.055).

Direct protocol overhead, reconciled: `awm exp_protocol` tool time is **0.00–0.05 h/cell**; card-authoring wall including generation is **0.05–0.40 h/cell**, mean **0.165 h** raw and **0.132 h** after the one measured overlap correction. Against 7.9 h that is **1.7–2.1 %**. This is the third window in a row to refute a large direct ceremony cost (ledger #11 recorded 0.136 h/cell in Window 02).

### 1.3 Header table B — decode, methods, evaluation

`greedy_shipped` is **[verified here]** by reading `final_model/generation_config.json` on each cell's receipt-backed `result_dir`. All 14 ship a config that resolves to greedy under vLLM. The **spellings differ in six ways**, which matters for §4.

| cell | shipped decode spelling | same-weight decode A/B measured? | A v2 clock (first post-SFT eval end → measured same-weight choice) | rl_used | rft_tried | executed max n | n behind the ship decision |
|---|---|---|---:|---|---|---:|---:|
| g01r03 | `do_sample:false, temperature:0.0` | yes, +7.7 pt @300 | 0.23 h | no | no (script written, never run) | 800 | 800 |
| g01r04 | `do_sample:true, temperature:0.0` | yes, +5.8 pt @500 | 0.37 h | no | yes (adopted, +3.0 @500) | 1000 | 1000 |
| g01r05 | `do_sample:false, temperature:0.0` | **no A/B — designed in** | **endpoint absent** | no | yes (2 rounds) | 1319 | 1319 |
| g01r06 | `do_sample:false, temperature:0.0, top_k:1, top_p:1.0` | **no A/B — designed in** | **endpoint absent** | no | yes (contradicted) | 1319 | 1319 |
| g01r07 | `do_sample:false, temperature:0.0` | yes, +5.8 pt @500, McNemar p=0.0076 | **6.22 h** | no | yes ×2 | 500 | 500 |
| g01r08 | `do_sample:false, temperature:0.0` | yes, +8.0 pt @150 | **2.29 h** | no | yes (rejected) | 500 | 500 |
| c01s01 | `do_sample:false, temperature:0.0, top_p:1.0, top_k:-1` | **no A/B on its own model** | **endpoint absent** | **yes (GRPO ×3)** | yes | 1319 | 1319 |
| c01s02 | `do_sample:true, temperature:1e-06, top_k:1, top_p:1.0` | **no A/B on its own model** | **endpoint absent** | **yes (LoRA GRPO)** | yes | 500 | 500 (paired) |
| c01s03 | `do_sample:false, temperature:0.0` | yes, +5.7 pt @300 | 0.06 h | no | yes | 1319 | 1319 |
| c01s04 | `do_sample:false, temperature:0.0` | yes, **+13.5 pt @200** | ≈0.03 h | **yes (full-param GRPO)** | no (built, never run) | 1319 | 1319 |
| c01s06 | `do_sample:false, temperature:0.0, top_p:1.0, top_k:0` | yes, **+15.4 pt @150** (pilot) | ≈0.08 h | no | yes | 400 | **300** |
| c01s07 | `do_sample:true, temperature:0.0, top_k:64, top_p:0.95` | yes, +9.0 pt @200 | ≈0.03 h | **yes (LoRA GRPO ×3)** | yes | 1000 | 500 |
| p00s01 | `do_sample:false, temperature:0.0` | yes, **≈0.0…+0.7 pt @150** | 0.16 h | no | yes (vote form) | 500 | 500 |
| p00s02 | `do_sample:false, temperature:0.0, top_k:0, top_p:1.0` | yes, **+11.3 pt @150** | 0.14 h | no | no (rejected 3×) | 500 | **150** (+ probe-300); n=500 was an artifact gate |

Reconciled aggregates:

- **Greedy shipped: 14/14.** The old binary A metric is saturated in both arms, as the 09-03 08:45 decision anticipated.
- **Measured same-weight decode A/B: 10/14** (guard 4/6, control 4/6, baseline 2/2). The four cells without one (g01r05, g01r06, c01s01, c01s02) designed greedy in from source inspection or a log line and never produced the frozen endpoint. Per the frozen A v2 definition and the planner's 09-03 clarification, **source inspection or an unmeasured greedy write does not fill that endpoint** — so these are **unknown/no measured choice**, not passes and not zeros.
- **A v2's revised metric is NOT saturated in the protocol arm.** On the protocol-installed 8: **4/8 measured within ≤0.5 h** (g01r03 0.23, g01r04 0.37, p00s01 0.16, p00s02 0.14), **2/8 clear failures** (g01r07 6.22 h, g01r08 2.29 h), **2/8 endpoint absent** (g01r05, g01r06). In the control arm 4/6 measured, all within ≈0.03–0.08 h, and 2/6 endpoint absent. c01s07's report calls the observable "saturated in the protocol-free control arm too"; **that overstates it — 4/6, not 6/6.**
- **Executed max n ≥ 500: 14/14. n behind the ship decision ≥ 500: 12/14** (c01s06 at 300, p00s02 at 150 + probe-300). The withdrawn C v2 threshold is saturated on the first reading and near-saturated on the second.
- **On-policy RL: protocol-installed 0/8, control 4/6** (two-sided Fisher p = 0.0150). **RFT attempted: protocol 6/8, control 5/6** (p = 1.00 — no arm difference).

### 1.4 Header table C — cost bookkeeping, kept in incompatible units on purpose

`pitfalls_cost_h` is a **sum of card entries**, so it is structurally **0.0 for all six controls** — the protocol's own KPI cannot be measured in the arm it is supposed to be compared against. That is a hard limitation of the metric, not a control-arm virtue.

| cell | raw `pitfalls_cost_h` (entries) | event-deduplicated card cost | reviewer trace-bounded ≥0.1 h losses | conditional post-exit idle (see caveat) | overrides / relocks / cards |
|---|---:|---:|---|---|---|
| g01r03 | 0.85 (6) | 0.85 **+ one unrecorded 0.09 h soup abort** (booked 0.0; visible only in the exp-07 relock reason) | ≈0.24 h mechanical + 0.50 h data-yield search | ≥0.85 h / 10 events, largest 0.18 h | 2 / 2 / 8 |
| g01r04 | 2.28 (7) | 2.28 (2.15 on one card) | 1.6 + 0.4 + 0.1 + 0.08 + 0.05 | ≥0.75 h (event-sum ≈0.96), largest **0.36 h** | 4 / 0 / 4 |
| g01r05 | 1.10 (4) | 1.10 | 0.85 + 0.15 + 0.10 | ≥0.46 h / 3 events, largest 0.30 h | 0 / 1 / 7 |
| g01r06 | 2.30 (7) | **1.25** (exp-04 carries exp-03's entries verbatim) | ≈1.0 h | ≤~0.02 h per event; not material | 0 / 4 / 7 |
| g01r07 | 0.10 (2) | 0.10 — **the cell's largest real cost, ≈0.8 h of re-work, is booked at 0.0** | 0.028 h OOM; 0.8 h re-work unbooked | ≈0.05 h total | 0 / 1 / 8 |
| g01r08 | 1.00 (6) | ≈1.00 | 0.50 + 0.202 + 0.13 + 0.10 | ≥0.202 h single event; one 0–6 min unresolved | 0 / 1 / 8 |
| c01s01 | 0.0 (n/a) | n/a | 0.17 + 0.17 + 0.13 (+0.27, 1.28 exploration) | audited ceiling **0–0.144 h** for the one selected RS transition; whole cell not adjudicated | — / — / 0 |
| c01s02 | 0.0 (n/a) | n/a | 0.50 + 0.30 + 2.07 + 0.12 (+0.87 exploration) | audited: one **exact 1.99917 h** uninterrupted wait after a logged OOM; strict post-exit component **[0, 1.99917 h]** | — / — / 0 |
| c01s03 | 0.0 (n/a) | n/a | 0.70 h recipe; ~0 mechanical | audited: five long-producer transitions sum ceiling **0.09889 h**; not a whole-cell total | — / — / 0 |
| c01s04 | 0.0 (n/a) | n/a | 0.175 + 0.113 (+0.65 selection evals) | ≈0.15 h total, largest 0.113 h | — / — / 0 |
| c01s06 | 0.0 (n/a) | n/a | **2.93 + 0.86 + 0.10 = 3.89 h** | ≈0.17 h total, largest ≈0.09 h | — / — / 0 |
| c01s07 | 0.0 (n/a) | n/a | 0.10 (+1.01 exploration) | ≈0.15 h total, largest ≈0.10 h | — / — / 0 |
| p00s01 | 0.96 (5) | failed compute 0.25 **+** idle 0.55 **+** 0.12–0.16 **+** 0.07–0.13 | as listed | 0.55 h (crash) + ≤0.12 h (exp-02) | 0 / 0 / 7 |
| p00s02 | 0.95 (7) | failed compute 0.49 **+** idle 0.53 **+** 0.095 + 0.083 + ~0.05 + 0.033 | as listed | 0.53 h (crash) + ≈0.13–0.15 h | 0 / 2 (index says re-locked 2×) / 7 |

**Post-exit idle caveat, applied uniformly.** Every post-exit figure above except the three audited control cells rests on a **GPU-release, artifact-mtime or `train_runtime` proxy for producer exit**. The E2 exit-evidence audit establishes that "GPU context gone ⇒ OS producer already exited" is an assumption, not an observation, and the control-timing audit establishes that retained `task/` file mtimes are copy-time values. Applying the same standard to the NEW cells:

> **No cell in this window has an unconditional positive lower bound on post-exit idle.** All the ≥0.1 h figures above are conditional on prompt teardown. The only unconditional quantities are: exact tool-wait durations (c01s02's 1.99917 h; g01r08's 20:25:56→20:38:04 = 0.202 h between a logged producer error and the first observation), and conservative *ceilings* (c01s01 ≤0.144 h on one transition, c01s03 ≤0.09889 h over five). `metrics.md` already forbids double-counting an invalid-save run's post-exit wait under both D and E; I keep failed compute and idle in separate columns and never sum them into one candidate's savings.

Card bookkeeping across the 8 protocol cells **[verified here]**: **56 cards, 56 closed, 0 `n_locked_open`, `fields_filled` 1.0 in 8/8**, 6 overrides (all in g01r03 and g01r04), 11 relocks. Family distribution: sft 17, other 12, decode-config 11, rft 10, merge 6.

---

## 2. Ranked explanations of the observed score differences

### R0 (first, because it bounds everything else) — this window establishes no between-arm score effect

**[INFER, verified here]** guard6 − control6 = **−0.01605** with Welch t = **−0.662**, df ≈ 8.0, SE 0.0242. The guard arm's spread (sd 0.0514, range 0.1372) is 1.7× the control's (sd 0.0297, range 0.0690). baseline2 has n=2 and is descriptive only. Pooling all protocol-installed cells (guard 6 + baseline 2, mean 0.744977) against control6 gives Welch t = −0.315. Nothing here is distinguishable from run-to-run variation, and `metrics.md` puts gsm8k noise at ~0.01–0.03.

g01r03's 0.8278999241849886 is the **maximum single clean official score in the line**, and nothing more: it is not protocol efficacy, not a mean effect, and not grounds for promotion. **No held-out AIME result exists for any variant, so promotion is impossible for every candidate discussed below.**

All 14 cells clear the fixed 24-cell J/K floor 0.6737212534748547 **[verified here]**; the lowest is g01r04 at 0.6907.

### R1 — the post-SFT starting point dominates every other term (14/14 cells)

Every cell's score is set mostly by one thing: how good its first format/termination SFT was. The jump from base is 55–67 points; everything afterwards is a 1–9 point layer.

- p00s01 exp-02: 0.06 → 0.7267 @150, format compliance 0.24 → 0.96, cap hits 67/150 → 6/150 **[reviewer-cited]**. [SAID, L9378, 02:54:58Z] *"Almost all of the gain is format, not math."*
- p00s02 exp-02: 0.04667 → 0.62667; truncated share 0.51 → 0.000/0.007 **[reviewer-cited]**.
- g01r08 exp-02 scored **0.060, below its 0.073 base**, and only exp-03's prompt-distribution fix took it to 0.6267 @150 **[reviewer-cited]**. That cell's whole trajectory turns on a diagnostic probe showing 0.76 bare vs 0.12 under the grader's 10-shot prefix.
- c01s06's round-1 SFT alone reached 0.7567 @300 — the highest post-SFT number in its group — and carried the cell despite 3.79 h of discarded compute **[reviewer-cited]**.

**Counterevidence to over-reading this:** the spread among cells that all cleared the format hurdle is still 0.69–0.83, so R1 explains the floor, not the ranking.

### R2 — decode configuration: the largest *measured* same-weight effect, but not a between-arm explanation (14/14 shipped, 10/14 measured)

Measured same-weight gains: **+15.4** (c01s06), **+13.5** (c01s04), **+11.3** (p00s02), **+9.0** (c01s07), **+8.0** (g01r08), **+7.7** (g01r03), **+5.8** (g01r04), **+5.8** (g01r07, McNemar p=0.0076), **+5.7** (c01s03), **≈0.0…+0.7** (p00s01).

Because all 14 shipped greedy, decode cannot explain the arm means. It explains two things:

1. **Magnitude is a property of the parent, not of the intervention.** p00s01's sampled exp-02 was already at 0.7267 with 6/150 cap hits, so greedy bought ≈0; the cells whose sampled parent sat at 0.61–0.63 gained 11–15 points. The calibration note makes the same point across the byte-identical p00r02/p00s01/p00s02 triple (+14.0 / ≈+0.7 / +11.3). `metrics.md` is right that Round 00's large gains are historical observations, not a universal benefit.
2. **Timing changed one cell's shipped artifact.** g01r07 discovered the inherited sampling config at 02:38:16Z, 8.1 h in, by accident: [SHOWS, L8469 02:38:00Z] the same soup weights scored 0.58 @150 against 0.664 @500, and grepping the vLLM log surfaced the override line **[reviewer-cited]**. The greedy re-run then **reversed the selection ordering** (exp-04 0.736 > soup 0.722 > exp-05 0.708, against the sampled 0.644 < 0.656 < 0.664), and `final_model` was re-staged from exp-04. Directly traceable re-work ≈0.8 h, **booked at `cost_h: 0.0`**.

**Counterevidence / caution.** g01r08 is the positive control on the other side: it deferred decode with an explicit, correct reason and then **kept the decode variable matched across every later comparison** — [SHOWS, L7766, 00:43:25Z] it copied the greedy config onto `ckpts/exp-05/final` before evaluating it **[reviewer-cited]** — and its n=150 and n=500 orderings agree. Per the planner correction, g01r07's pre-fix comparisons used a *common* sampled policy; the late fix changed a later ranking, which is not the same as proving an opposite-side decode mismatch for 8 h.

### R3 — post-SFT method stack: the largest between-cell differentiator, and not a protocol variable

**[verified here]** On-policy RL: **0/8 protocol-installed, 4/6 controls** (Fisher p = 0.0150). This extends ledger #8 (protocol v3 0/14, clean controls 8/15) to 0/22 and 12/21.

The two clean within-cell RL measurements are large:

- c01s01: same parent, same protocol, same n=1319 — `work/rft1_bf16` **0.7415** → `work/g350` **0.7998**, **+5.84 pt from GRPO alone** **[reviewer-cited]**.
- c01s04: SFT alone 0.640 @200; shipped GRPO stage-2 checkpoint 0.7528 @1319 **[reviewer-cited]**.

c01s03's report reads its own 6.9-point gap to c01s01 as exactly this: on every operational axis it matched or beat the top scorer, and its pipeline simply ends at SFT.

**Counterevidence, and why this is not a protocol conclusion:**

- **The highest cell in the window used no RL and no RFT.** g01r03 (0.8279) reached its score with a decode fix, solutions-per-problem data scaling, and a two-checkpoint soup — [SHOWS] `scripts/sample_rft.py` was written at 19:06:23Z and never invoked **[reviewer-cited]**.
- **RL is not sufficient.** c01s07 ran three LoRA GRPO rounds over ~4.04 h and tied for the *lowest* control score (0.7202); rounds 2 and 3 added nothing that survived a 500-item read.
- Adoption rate is a recipe difference measured on a non-randomised axis. Ledger #8 already forbids inferring protocol causality from it, and `SKILL.md` explicitly does not say what to run. **I do not propose any RL/RFT/soup/prefix rule.**

**Mixed soup outcomes (ledger #18 confirmed, not advanced):** attempted in ≥8 cells, **adopted in 2** (g01r03 +2.1 pt @800 — the shipped model; g01r06 +0.9 pt @1319), **rejected in 6** (g01r05 0.7536 vs 0.7619 @1319; g01r08 0.666 @500 below both parents; c01s01 0.7854 @1319 below g350; c01s03 twice; c01s06 by one item @300; g01r07's soup was staged then reversed). g01r03's adopted soup is verified as a real average, not inheritance: on the 91 items the parents split, the soup takes 65 **[reviewer-cited]**.

**Mixed prefix outcomes (ledger #28 confirmed, not advanced):** g01r08's 10-shot-matched continuation was decisive (0.060 → 0.6267); g01r07 *measured* the same lever as unnecessary for its own checkpoint (probe 0.730 with prefix vs 0.688 without) and kept its share at 6 % **[reviewer-cited]**. The same intervention is decisive in one cell and superfluous in another. No universal prefix percentage is supportable.

### R4 — selection quality at the decision n: four realised inversions, and two counterexamples

Realised rank inversions that changed or would have changed a shipped artifact:

- **g01r05**: exp-04 vs exp-02 read +1.0 pt at n=200 against a 3.2-pt SE; at n=1319 the gap was **+4.25 pt, McNemar z = 3.54** **[reviewer-cited]**. [SAID] *"Had I trusted n=200 I'd have shipped a materially worse model."*
- **c01s04**: at n=400 `soup1` led (0.750 vs `b70` 0.7425); at n=1319 that inverted (0.74375 vs **0.75284**) and the full-test pass changed the shipped artifact **[reviewer-cited]**.
- **g01r03**: exp-04 beat exp-05 by 3.0 pt at n=300; at n=800 the ordering collapsed to 0.8175 vs 0.81625 and the card's own McNemar (z = 0.1) called it **[reviewer-cited]**.
- **p00s01**: dev-150 ranked exp-06 > exp-04 > exp-05; the n=500 tie-break and an 800-problem held-out set both ranked exp-05 first, protecting ≈1.2 pt **[reviewer-cited]**.

**Counterexamples that stop this becoming a rule about n:** p00s02 selected at **n=150 + probe-300** and scored 0.7771, second-highest of the eight protocol-installed cells; c01s06 selected at **n=300 on a one-item margin** and scored 0.7415, above the guard mean. And c01s04's own recount shows its winner's margin over the runner-up is **8 items of 1318 with 430 discordant** (McNemar z ≈ 0.39) — a full-test pass that changed the artifact without establishing that the new artifact is better.

**Measurement caveats I am obliged to carry.** c01s04's and c01s07's paired counts are regex-derived and omit one record — treat them as approximate until an exact structured parse with aligned IDs recovers all items; c01s07's logged-ladder count needs a recount rather than copying its 15-log header. c01s07's n=500-vs-n=1000 contrast changes both `max_tokens` and n/batching, so it is a configuration contrast, not an isolated repeat-noise distribution. Official per-item identity remains unresolved wherever the official log was not retained (ledger #24/#29).

One repeat-variance result *is* item-level verified from retained developer logs: g01r05's exp-04 and exp-07 both score **994/1319 = 0.7536012130401819**, and re-derivation of the two retained 1319-item logs gives **994 correct each, 154 items decided differently, 77 each way** **[reviewer-cited]**. Equal scalars, disjoint error sets — the correct reading of ledger #12/#19, and a clean demonstration that equal accuracy is not equal correct-item sets.

### R5 — mechanical waste: large, well-evidenced, and *not* shown to move score here

The single most reproduced mechanism in the window is the **greedy-parent save-validation failure**. **[verified here]** by grepping `GenerationConfig is invalid` / `Fix these issues to save the configuration` across all 14 traces: it appears in **8 of 14 cells** — guard g01r03, g01r05, g01r06; control c01s01, c01s02, c01s06; baseline p00s01, p00s02 — on Trainer checkpoint saves, `soup.py` weight-averaging saves and a LoRA `merge_lora.py`. **It hits both arms**, so it is a mechanical waste the protocol could catch and the control cannot, not an arm-gap mechanism.

Event costs, kept in separate columns (never summed into one candidate's savings):

| cell | event | failed compute | conditional post-exit idle |
|---|---|---:|---:|
| g01r03 | exp-07 soup abort | ~0.09 h (booked 0.0) | — |
| g01r05 | exp-03 Trainer death at step 620/1238 | 0.85 h (card) | 0.10 h |
| g01r05 | exp-07 soup relock | ~0.01 h | — |
| g01r06 | exp-03 `trainer.save_model` after 48 min | 0.849 h (trace-bounded) | ~0 (tail-coupled) |
| g01r06 | `soup.py` | 0.05 h | — |
| c01s01 | `soup.py` refused `top_k:-1` | 0.02 h | — |
| c01s02 | `merge_lora.py` ×2 | ~0.02 h | — |
| c01s06 | round-2 SFT final save | 0.86 h | ~0.02 h |
| p00s01 | exp-06 Trainer at step 150/306 | 0.25 h | 0.55 h |
| p00s02 | exp-05 Trainer at step 150/313 | 0.49 h | 0.53 h |

**A dose-response worth naming [INFER]:** the two cells with `--save-steps 150` lost **0.25 h and 0.49 h**; the two that saved only at the end lost **0.849 h and 0.86 h**. An early first durable write bounds the loss of this failure mode — relevant to §4.

Other large mechanical losses: c01s06's **2.93 h fully discarded** rejection-sampling pass for missing `stop`/`stop_token_ids` (B v2's mechanism, in a control); g01r04's **1.6 h** of completed generation destroyed by an `OverflowError` in its post-generation grading loop; c01s02's **2.87 h** of `pgrep`-self-match idle-class time; g01r08's **0.50 h** tool-timeout kill; recurring Gemma-3 logits OOM and 64 MB-overlay install failures across ≥6 cells (P2/#21); missing `preprocessor_config.json`/`processor_config.json` in g01r03, g01r08 and c01s04 (P3/#22).

**Why this does not explain the scores.** c01s06 lost 3.79 h and still scored 0.7415, above the guard mean; g01r03 lost ~0.5 h and scored highest; g01r06's own report states the 0.85 h *"did not obviously cost score here (2.27 h went unused)"*. **[INFER]** Saved compute does not imply saved score. These are cost claims, not efficacy claims.

### R6 — the low guard cells, individually

- **g01r04 (0.6907, lowest NEW guard)** — one large early method bet with no reserve. Its 1.6 h sampling crash forced the pool from 28k+7k prompts down to 9k+7k, and it reached its only post-SFT training launch at 23:30 with 2.8 h left; g01r03 ran three post-SFT rounds and souped two. Per the planner correction: its fitted-parent P1 exposure is **not** established by flat loss — its mixed RFT corpus trains from base, not from its sampling parent — and the 1.6 h is real waste, not proof that it explains the cross-cell gap.
- **g01r08 (0.7028)** — ~0.9 h mechanical loss and **2.53 h unused**, having declined a ~60-minute full-test verification for "too little margin" with 2:49 on the clock.
- **g01r06 (0.7081)** — 0.849 h save crash, then 2.27 h unused.

**Do not equate free hours with score.** **[INFER, verified here]** Pearson r(hours_used, accuracy) = **0.519** over all 14 (t = 2.10, df 12, two-sided p ≈ 0.06); within protocol-installed 8, r = 0.583; within controls, r = 0.676. This is marginal, observational, n=14, and confounded in an obvious direction: a cell with a productive pipeline keeps running, and a cell that has nothing left to test stops. The clearest counterexample is g01r03 — mid-range hours (8.44), highest score.

---

## 3. What the protocol cost and what it helped

### 3.1 Corrected direct overhead

`awm exp_protocol` tool time **0.00–0.05 h/cell**; card-authoring wall including generation **0.05–0.40 h/cell** (mean 0.165 h raw, 0.132 h after the single measured overlap correction), i.e. **1.7–2.1 % of a ~7.9 h session**. Bootstrap first action (Skill load → first `awm exp_protocol index`) is ≈8–11 s in every protocol cell. Ledger #11's "direct ceremony cost basically refuted" holds for a third window.

**Time to first launch, both definitions.** Protocol-installed cells reach a *card-matched* training launch at 0.25–0.51 h; controls reach a *production* run at 0.18–0.55 h. Under the any-GPU-training definition the arms are 0.16–0.23 h (protocol) vs 0.10–0.19 h (control). **The "protocol launches later" effect is ≈0.05–0.2 h and largely disappears once the uncarded smokes are counted on both sides** — which is exactly why the lock-scope question (§3.3) is a compliance question, not a speed question.

### 3.2 Where the plan shape helped, with the cells

This is the strongest positive evidence in the window, and it is behavioural rather than statistical.

- **The comparator rule forced the decisive evaluation in ≥5 cells, and in 3 it changed or protected the shipped artifact.** g01r03's exp-06 exists only because the card demanded a same-protocol head-to-head, and it overturned the n=300 ordering. g01r05's exp-05 exists because [SAID] *"The protocol that produced both numbers cannot say which checkpoint to ship"* — it cost ~30 min and changed the shipped model. p00s01's exp-07 tie-break reversed dev-150 and protected ≈1.2 pt. g01r07's exp-08 and g01r06's exp-07 are the same pattern.
- **Rule 6 (keep the checkpoints you said you would) produced the window's highest-scoring artifact.** g01r03's exp-05 was `contradicted`/`reject`, and its checkpoint was still alive 3.5 h later for the soup that became `final_model`.
- **`falsified_if` closed directions rather than quietly abandoning them** in ≥5 cells: g01r06 exp-03 (17 fixed / 15 broken → *"Rejection sampling on top of an already-converged SFT model is not the lever"*), g01r07 exp-03 and exp-05, g01r08 exp-05 and exp-07, p00s02 exp-04 (whose two-part predicate turned a −10.7-pt result into the *diagnosis* that produced exp-05's +14.0-pt repair).
- **"One card, one intervention" prevented confounds explicitly**: g01r03 killed a temperature sweep with it; g01r06 kept its teacher corpus byte-identical so the union was the only difference; g01r07 refused to lower the learning rate while scaling data.
- **Bookkeeping completeness**: 56/56 cards closed, 0 open, `fields_filled` 1.0 in 8/8, no work lost at session end in 14/14. **[verified here]** `memory/.stop_hook.json` is absent in all six guard result dirs — **0 hook blocks**. Ledger #1's caveat stands: an untriggered hook does not prove the block works.

### 3.3 Lock/smoke/eval coverage — a successful main-launch ratio does not prove all commands were locked

`lock_before_launch` is 2/2–4/4 in every protocol cell. **[verified here]**, against `exp-02`'s `created_at` in each cell's cards:

| cell | first GPU training command | exp-02 card created | gap |
|---|---|---|---|
| g01r03 | 18:44:58Z | 18:59:49Z | 14.9 min |
| g01r04 | 18:41:40Z | 18:48:41Z | 7.0 min |
| g01r05 | 18:44:16Z | 18:54:53Z | 10.6 min |
| g01r06 | 18:42:12Z | 18:51:53Z | 9.7 min |
| g01r07 | 18:42:08Z | 18:46:27Z | 4.3 min |
| g01r08 | 18:42:19Z | 18:44:20Z | 2.0 min |
| p00s01 | 19:13:59Z | 19:26:58Z | 13.0 min |
| p00s02 | 19:23:00Z | 19:26:37Z | 3.6 min |

**8/8 protocol cells ran real GPU training before the covering card existed**, and all were recorded afterwards in `situation.smoke_runs` — a retrospective record, not a pre-launch lock. Two further ordering findings, both outside the matched denominator: **[verified here]** g01r08's exp-06 is an evaluation-only card whose first declared n=500 evaluation launched at **00:51:49Z**, while `exp-06.lock.json` records `locked_at 2026-09-04T01:06:55Z` — **15.1 minutes later**; and g01r08's exp-03 was re-locked at 21:36:13Z, 3.6 min *after* the launch that produced the shipped weights. This is frozen J's (#26) mechanism at **8/8 fresh exposure**, including its forward-review finding that graded generation is evaluation. J is frozen; this is evidence for it, not a new proposal.

Notably, g01r08's pre-lock evaluation was a *direct response to a preflight failure*: **[verified here, L8092, 00:51:41Z]**

> `FAIL comparator_same_protocol — … /home/ben/task/eval/exp-06_exp05_dev500.json does not exist`
> `not locked: preflight failed on comparator_same_protocol`

and eight seconds later the scientist launched the missing comparator evaluation instead of locking. That is candidate K's (#27) in-card future-comparator dependency, reproduced in a NEW cell, with the *third* response type (the other two being g01r03's recorded override at 02:07:22 and, in an earlier cohort, pre-lock launches). **Correction to a possible over-count:** g01r06's `comparator_same_protocol` FAIL at 01:44:17Z is **not** a future-comparator case — it is a filename typo (`exp03_full.json` vs the existing `exp-03_full.json`), fixed by `sed` in 7 seconds **[verified here]**. So the K mechanism is **2/8 cells** in this window, not 3.

### 3.4 Comparator honesty — a frozen check that cannot do its job

**[verified here]** by grepping all 14 traces and by reading `2f64581:awm/exp_protocol/preflight.py:265–286`:

- `comparator_same_protocol` returns **`pass` with the literal detail "exists; it does not record n"** in **18 events across 8/8 protocol cells** (g01r03 L4988/L6859; g01r04 L4359/L7123; g01r05 L4901/L5057/L6524; g01r06 L4062; g01r07 L4798/L6215; g01r08 L4152/L4262/L6212; p00s01 L4218/L6804; p00s02 L4436/L6137/L7171). The cause is structural: `evaluate.py --json-output-file` writes only `{"accuracy","stderr"}`, so the artifact this task produces can never record n.
- **[verified here]** Of the **46 cards that declare a comparator path**, exactly **2 declare a comparator whose n differs from the card's own `evaluation.protocol.n`**, and the check cannot see either: **g01r03 exp-08** (`protocol.n: 150`, comparator `exp-07_soup_dev800.json`, value 0.84 — the soup's n=800 read) and **p00s02 exp-07** (`protocol.n: 500`, comparator `exp05_dev150.json`, value 0.77333 — the n=150 exp-05/exp-06 tie). Frozen rule 2 says *"A number from a different `--limit` is not a comparator."* Both locked.

**[verified here]** K's frozen change (`58a6992:awm/exp_protocol/preflight.py:265–275`) intercepts this check **only** when `evaluation.comparator.defer_validation is True`; the ordinary path is byte-identical to the baseline. So this defect is genuinely uncovered by K, and — per the brief and the planner corrections — **must not be folded into K**.

### 3.5 Raw-render false alarms and the metrics blind spot

- **g01r05's `stop_token_consistent` failure was a raw-field/schema mismatch, not a missing rendered EOT.** The card-semantics audit establishes that the original `train_sft.py` (written 18:39:05, T05 L3392 onward) already did `full = f"{prefix}{completion.strip()}<end_of_turn>\n"` with the terminator inside the supervised labels, and that a pre-failure smoke at 18:44:54 reported EOT id 106 and *"all sampled targets end with `<end_of_turn>`"*. The 49-second repair changed the raw JSONL schema and its consumer symmetrically. **Rejected here, as in the planner corrections:** the g01r05 report's *"true positive"*, *"rescued a 2.3 h mis-terminated run"*, *"+65 pts"* and *"reverses ledger #9"* claims. The correct statement is: *preflight rejected un-terminated raw completion fields, prompting a 49-second schema/consumer adjustment; the existing renderer already supervised EOT; the matched dev200 improvement 0.055 → 0.710 is the bundled SFT/format/teacher-data/decode package, not isolated termination causality.* Note also that the cell's own diagnostic counts "not `max_tokens`", not EOT compliance.
- **Zero overrides does not mean zero nuisance failures.** g01r05 has `n_overrides = 0` and `preflight_fail = 0` and still paid a data rewrite.
- **[verified here] `collect` sees 6 of ~33 recorded friction events.** Across the 8 protocol traces there are **25 `check` ERROR lines** and **8 `not locked: preflight failed` events**; `collect` reports `preflight_fail` 2+4 = 6 and `n_overrides` 2+4 = 6, all from two cells. **Six of the eight cells show `0/0`** while every one of them hit at least one blocking `check` error. Of those errors, **`setup.data: required` on a card that trains nothing fires in 6/8 cells** and **`setup.data[0].n_examples: must be a positive integer` in 4/8** — together **8/8 cells** hit H's (#5) mechanism, against 0 overrides in 6 of them. That is fresh, broad evidence for frozen H and a precise measurement of the blind spot `metrics.md` warns about.
  - Correction to my own first reading: `problem.watch_set: needs path and n` (4/8 cells) is **not** a forced-fabrication error — `2f64581:awm/exp_protocol/schema.py:258–260` fires only when a partial `watch_set` is present. It is template-placeholder friction costing seconds, an observation only.
- **`pitfalls_cost_h` both over- and under-counts.** Over: g01r06's raw 2.30 h contains exp-04's verbatim carry-forward of exp-03's entries (dedup 1.25 h). Under: g01r03's 0.09 h soup abort is booked 0.0 and survives only in the relock reason; g01r07's ≈0.8 h of decode re-work is booked 0.0. And it is structurally 0.0 for all six controls.

### 3.6 Card policy boundaries — breach established, contamination not established

Per the card-semantics audit, and adopted here unchanged:

- **g01r03 breached rule 7 more than once.** `exp-02.yaml` L34–36 stores the Cody/Amir test question with gold `20` verbatim (source: a retained developer Inspect log; accepted card write T03 L4867, 19:01:21Z); `exp-03/04/05` each retain three to four test-item descriptions with golds. The g01r03 report's *"broken once"* understates the strict rule's scope.
- **g01r05 is not a clean compliant counterpart.** T05 L4674–4693 (18:55:50Z) first wrote all 189 incorrect dev200 samples as `{id, question, gold}`; L4702–4716 (18:56:00–01Z) rewrote them to `{id}` only, citing rule 7. Its cards still carry test-derived IDs and item-specific `model_output` detail. The shipped rule text has **no ID-only exception**, so this is content redaction under an unresolved literal clause, not full compliance.
- **No training contamination is established.** The inspected builder/trainer lineage in both cells reads only command-named JSONL `prompt`/`completion`; the harness expressly permits `evaluate.py` feedback. **Both** the automatic-invalidation claim and the stronger *"cards never affect score / no consequence"* claim are rejected. Official scores, validator status and judge status are unchanged: g01r03 remains 0.8278999241849886, g01r05 remains 0.7611827141774071.

### 3.7 Automatic status is not compliance

All 14 cells are validator-complete, judge-clean and unflagged. That establishes a scored, loadable, non-anomalous deliverable. It establishes **nothing** about lock-before-launch coverage (8/8 cells violated it outside the matched denominator), rule 7 (breached in one cell, unresolved in another), or comparator validity (2 cross-n comparators locked). Keep these separate.

---

## 4. Proposals — three, each one item, each traced to ≥2 cells

**Common score guardrail, declared before any of these runs and not to be updated after.** I propose the **same fixed 24-cell reference pool and floor already frozen for J and K**: mean **0.7037212534748547**, floor **0.6737212534748547**, membership exactly as in the J spec (v3 `p00r01–07`, `p00r09–15` = 14; old guard `g01r01/02` = 2; strict guard `g01s01–08` = 8). I checked the older candidates' own specs rather than assuming: **A/B/C/D/E/H use a different pool** — "baseline pool (v3 core 16 + guard) mean − 0.03" per Round02 §三, and E2's spec says "predeclared protocol-baseline pool mean −0.03". I am **not** recomputing any pool with this window's 14 cells. All 14 clear the J/K floor already, so it is a coarse failure guardrail, not a live constraint. Every proposal below also needs **exposure in ≥2 of its 4 cells to be judged at all** (J's rule: no exposure is uninformative, not success), a **second independent 4-cell block** if it wins, and **held-out confirmation** before any promotion — which no candidate can currently obtain, because no held-out AIME result exists.

---

### Proposal 1 — **D2**: scope the parent-generation-config guard to operations that actually save a model

**Surface (one item):** one preflight check with a test — a revision of `parent_generation_config_valid`. Leave D v1's `greedy_parent_generation_config` pitfalls entry **byte-identical**, so the diff against the guard baseline is the check plus its tests. (D v1's own construction bundled check + test + catalogue line as one item; keeping the catalogue line unchanged makes D2 strictly smaller.)

**Why v1 cannot ship as frozen.** `8332917:awm/exp_protocol/preflight.py:311–334` reads only `setup.parent_checkpoint.path` and applies to **every card** — no family, argv, framework or save-operation condition, and `preflight.py:368–383` runs every check for every card. The d-scope CPU replay confirms two false blocks on trace-verified evaluator-only cards (**g01r03 exp-08**, **p00s02 exp-07** — the latter is exp-07, not exp-06), one true Trainer-save hit (p00s01 exp-06), one true non-training `save_pretrained` hit (g01r03 exp-07 `merge`), and blindness to a verified in-code repair (p00s02 exp-06, repaired g01r03 soup).

**Scale of the false-block exposure, derived independently here.** Across the **56 cards** of the 8 protocol cells, **16 cards declare no model-saving command yet name a local parent checkpoint** — g01r03 3, g01r04 1, g01r05 1, g01r06 1, g01r07 2, g01r08 4, p00s01 2, p00s02 2, i.e. **8/8 cells have at least one**. Two of the 16 are CPU-replay-confirmed blocks; the other 14 are structurally exposed but their parent's config at lock time is not retained, so their outcome is **unknown, not confirmed**. For context: **11 of the 14 shipped `final_model/generation_config.json` files would fail frozen D's exact predicate if named as a local parent**, and the 3 that pass (g01r04, c01s02, c01s07) all spell greedy with `do_sample: true` — because `preflight.py:323` returns `pass` whenever `do_sample` is true or absent.

> **Correction to c01s07's report.** It argues that "a check that simply flags 'greedy' would fire on c01s07's safe artifact". **[verified here]** Frozen D does **not** fire on it: `do_sample: true` short-circuits at line 323. D already keys on the save-hazardous spelling, and it correctly separates c01s06's failing `do_sample:false, temperature:0.0, top_k:0` from c01s07's surviving `do_sample:true, temperature:0.0`. c01s07's "spelling check" is therefore **not a new candidate** — it is largely D's existing predicate. The residual gap is scope, which is what D2 fixes.

**Proposed predicate.** Keep the current test exactly, and gate its severity on the declared method family: **`fail` when `setup.method.family ∈ TRAINING_FAMILIES ∪ {merge}`**, otherwise **`warn`** with the same message plus one clause — *"this card declares no training/merge family; if its command does save a model, fix the parent or set `model.generation_config` before the first save."* A blanket exemption for all non-training families is explicitly rejected, because `merge` is where g01r03's real soup save failure lives.

**False-negative risk, measured.** **[verified here]** Across all 8 protocol cells, only `train_sft.py` (training families) and `soup.py` (merge) call `save_pretrained`/`save_model` — **no `other`, `decode-config` or packaging script does**. So the family gate has **zero observed false negatives in 56 cards**. One residual false *positive* remains: g01r08's exp-07 is `family: merge` whose declared argv is `evaluate.py`; it would still be blocked. That is 1 of 16, and it is a known, small, disclosed cost.

**Tests to add (the point of the item):** (1) `other` eval-only card with greedy parent → warn, not fail [g01r03 exp-08 fixture]; (2) `decode-config` with greedy parent → warn [g01r08 exp-06 fixture]; (3) `merge` with greedy parent → fail [g01r03 exp-07]; (4) `rft` with greedy parent → fail [p00s01 exp-06]; (5) `do_sample:true, temperature:0.0` parent → pass [c01s07's shipped spelling — documents the do_sample key]; (6) valid greedy (`do_sample:false`, no sampling keys) → pass; (7) stock sampling → pass; (8) missing `generation_config.json` → skip. D v1's existing tests at `tests/test_exp_protocol_preflight.py:203–241` are preserved.

**Predeclared 4-cell readout.**
- *Primary:* **hours attributable to a `GenerationConfig is invalid` save failure = 0 in all 4 cells**, with the check firing at `fail` severity at least once in **≥2 of 4** (exposure requirement).
- *False-positive guardrail:* **zero blocking failures on cards whose declared command does not save a model** (count them explicitly; separately count `warn`-severity firings, which are expected and are not failures), and **≤1 override on this check per cell**. A check overridden in many cells is a wrong check (`metrics.md`).
- *Score guardrail:* fixed 24-cell floor 0.6737212534748547.
- *Other guardrails:* required-field completeness must not fall; no rise in relocks caused by this check; do **not** count post-exit idle from the same crash as D2's savings (it belongs to E).

**Alternatives and what would change the decision.** (i) *Hold D entirely* and rely on the pitfalls text: forgoes the single most reproduced mechanical mechanism in the window (8/14 cells, both arms). (ii) *Withdraw D v1's block and do nothing*: same cost, plus it leaves the confirmed false blocks unaddressed if D is ever revived. (iii) *Inspect `setup.command.argv`/the named script for a save call instead of the family*: more precise, materially more fragile, and a second mechanism inside one item. Decision flips to (iii) if a future window shows `other`/`decode-config` cards that do save — none exist in these 56.

**Queue note, not a decision:** D v1's held jobs 91060–91063 carry the immutable old manifest/receipt. A corrected variant needs its own single-item commit off guard drift `2f64581`, its own tests, its own immutable manifest, and a **whole-block replacement** workflow. Whether and when that happens is the planner's call; nothing here authorizes a release, cancellation or registration.

---

### Proposal 2 — **comparator n must be verified, not assumed**

**Surface (one item):** one preflight check with a test — `comparator_same_protocol`. **No schema change and no template change are required**: **[verified here]** `2f64581:awm/exp_protocol/schema.py:328–330` validates only `evaluation.comparator.value`/`path` and rejects no unknown keys, so a scientist-declared `evaluation.comparator.n` is already schema-legal. This keeps the intervention to a single surface, which the planner corrections specifically demand ("optional metadata plus new enforcement is a coordinated intervention").

**Change.** Resolve the comparator's n from, in order: (a) the artifact's own recorded n (existing behaviour), else (b) a scientist-declared `evaluation.comparator.n`. If neither exists → **fail**, naming both remedies. If either exists and differs from `evaluation.protocol.n` → **fail** (this already happens for (a); it is new for (b)). `skip` on no comparator path is unchanged.

**Source cells (≥2, both conditions).**
- *Substantive:* **g01r03 exp-08** and **p00s02 exp-07** — the two cross-n comparators of §3.4, in two different cells, both locked, both invisible to the frozen check.
- *Mechanism frequency:* 18 unverifiable `pass` events in **8/8** protocol cells, with the exact line numbers listed in §3.4.

**Predeclared 4-cell readout.**
- *Primary:* **zero locks in which `comparator_same_protocol` returns `pass` without an n it actually compared**, and **zero completed decision cards whose comparator n differs from `evaluation.protocol.n` without a recorded reason**. Both targets 0 across all 4 cells; require exposure (a comparator-bearing card) in ≥2 of 4.
- *False-positive guardrail (the real risk):* **overrides on this check ≤1 per cell**. Because no artifact this harness produces records n, the check will fire on essentially every comparator card until scientists start declaring it; if they override instead of declaring, the check is wrong for this data and must become a `warn` plus a card question. Also: card-authoring wall time must not rise by more than 0.05 h above this window's 0.13–0.17 h band; `fields_filled` must not fall; **count any file written solely to carry an `n` key** as a fabrication signal (H's lesson).
- *Score guardrail:* fixed 24-cell floor.

**Explicit non-overlap.** This is **not** K (verified: K intercepts only `defer_validation: true`; the ordinary path is unchanged) and **must not be folded into K**. It is **not** C v2 (that was rule-2 *wording* with an n ≥ 500 threshold; this sets no threshold). It enforces the **existing frozen rule 2** clause that currently has no working enforcement. It is not stacked on anything.

**Alternatives and what would change the decision.** (i) *`warn` instead of `fail`*: cheaper and safer, but both substantive mismatches would still have locked. (ii) *Hold until K reads out*: K is opt-in by design and its spec says it does not solve the legacy comparator, so waiting buys nothing. (iii) *Make `evaluate.py` record n*: forbidden — the harness is immutable. Decision flips to (i) if a 4-cell run shows >1 override per cell.

**Honest limitation.** No cell in this window demonstrably shipped a worse model because of this defect; p00s01's tie-break rescued the one case that mattered most. This is a record-integrity and rule-enforcement item with a mechanism-count metric — the same shape as J and K — with score only as a guardrail.

---

### Proposal 3 — **P4 (#23)**: price the remaining time against a measured cost

**Surface (one item):** `SKILL.md` rule 8 wording. Ledger #23 already names this surface and this direction; it is queued, not frozen, so building it is a candidate decision rather than a re-proposal.

**Proposed clause (one sentence added to rule 8).** *When you reject an option because of time, quote the current `timer.sh` reading and the measured wall time of the most comparable operation you already ran in this session, and show the arithmetic.*

**Source cells (≥2).**
- **g01r08**, exp-08 `alternatives_rejected` [SAID, L8689, 01:41:06Z]: *"Verify at n=1319 — ~60 min; leaves too little margin, and n=500 already gives a +/-2 point interval"* — with 2:49 on the timer, and the cell ended at 01:58:01Z with **2:32** left **[verified here]**.
- **g01r07**, exp-08 `next_step` [SAID, L9299, 02:51:28Z]: *"Nothing further is launched; the remaining time is under an hour and any new run would be unverifiable"* — `bash timer.sh` printed **1:38** fifty seconds later (L9474–9475) **[verified here]**.
- Supporting: p00s01 (2.12 h unused, no time reason given; the untried lead is in exp-07's `next_step`), p00s02 (2.20 h unused, same pattern), g01r04 (2.25 h unused while naming *"a preference method would be the natural attack"*).
- Window context **[verified here]**: protocol-installed **8/8** stop with ≥1.55 h unused (mean 2.079 h); controls 3/6 (mean 1.592 h); Fisher p ≈ 0.055.

**Predeclared 4-cell readout.**
- *Primary (pricing accuracy, deliberately not hours used):* in all 4 cells, **every end-of-session `alternatives_rejected` entry that cites time contains both the timer reading and a measured comparable cost from the same session**, and **zero such entries under-price the remaining budget by more than 30 % against a timer reading taken within 15 minutes**. Target 0 violations.
- *Context only, reported and not targeted:* hours unused at stop, and any additional verified result obtained in the recovered time.
- *Failure guardrails, explicit:* **runs alive at session end = 0** and **`n_locked_open` = 0** (both are 14/14 today and must stay there); `fields_filled` must not fall; no rise in cards closed `failed`/`killed` for want of time.
- *Score guardrail:* fixed 24-cell floor.

**Why not the ledger's original target.** Ledger #23 proposes "cells with ≥1.5 h left and no process ≤ 1/4". I recommend against it as the primary: it is satisfiable by starting a low-value run, it collides with ledger #7's *"不诱导无意义跑"*, and this window's own data give no reliable hours→score relation (r ≈ 0.52, p ≈ 0.06, n = 14, confounded). A pricing-accuracy target cannot be gamed by burning GPU.

**Alternatives and what would change the decision.** (i) *Keep P4 as an observation*: costs nothing, but leaves the largest behavioural arm difference in the window unaddressed. (ii) *Fold into A or E*: refused — different surface, and it would stack. (iii) *Withdraw*: if a 4-cell block shows the clause is satisfied while neither hours unused nor verified results change, P4 is bookkeeping with no operational effect and should be withdrawn rather than iterated.

---

### Proposals I considered and am **not** making, with the reason

- **`test_item_text_absent` (g01r03's proposal) — rejected.** The card-semantics audit shows bare gold numbers (`3`, `20`, `150`) are non-specific and occur in legitimate train data, model outputs, metrics and reasoning; whole-token matching still rejects valid content; the scanner would miss paraphrases, aliases and ID-only selection; exact-text matching misses the Unicode/whitespace normalisation already present in the Cody example; and reading the checker-only test reference for an unrelated scanner needs an access-contract reconciliation. g01r05 is **not** a clean compliant negative control. The brief also forbids prescribing bare-number contamination scanning. The real blocker is upstream: **rule 7's intent regarding evaluator-derived item IDs is unresolved** (§5).
- **`long_run_defers_its_only_write` (g01r04's proposal) — rejected on coverage.** **[verified here]** frozen B v2 `9f294c3` already says: *"dump raw draws to disk before scoring and guard the parser for inf and nan."* g01r04's incident is exactly that — an `OverflowError: cannot convert float infinity to integer` in a post-generation grading loop, repaired by adding a raw dump. **Both halves are inside B v2's frozen guidance text**, contrary to the g01r04 report's coverage check. That removes the only non-D source cell; the remainder (g01r05 exp-03, g01r06 exp-03, c01s06 sft2) is D's mechanism. The residual lesson — an early first durable write bounds the loss (0.25/0.49 h with `--save-steps 150` versus 0.849/0.86 h with save-at-end) — belongs in **D2's design note**, not in a parallel candidate.
- **`evaluation.protocol.decode` optional field + comparator enforcement (g01r07's proposal) — deferred, with conditions.** Two objections stand: it sits in A v2's area (two simultaneous decode-area screens would make neither attributable), and a field plus new enforcement is a coordinated intervention. A decode card must also be allowed to differ in decode — the field must never prohibit its own treatment contrast. **What would make it a candidate:** A v2's 4-cell block reading out, plus a scope statement that the check compares decode only when both sides declare it and only for non-decode-family cards. Until then it is a **screen observable for A v2's readout**: *count cross-checkpoint comparisons whose two sides carried different effective `generation_config`* (g01r07: 8.1 h of them, ordering reversed once fixed; g01r08: enforced by hand twice; g01r06: sidestepped by writing greedy in the trainer).
- **Tool-timeout kills of model-executing foreground commands — not a candidate; a guardrail observable for E2.** Evidence is real: **[verified here]** g01r08 is the only cell in the window with a 30-minute tool timeout, and it killed a foreground vLLM diagnostic (`Exit code 143 / Command timed out after 30m 0s`, L5739) for an exact **0.500 h** loss; **[verified here]** p00s01's first 2-minute timeout killed a **foreground GPU training benchmark** at 19:17:17Z (`Exit code 143`, one config's results already printed) and was repaired by detaching to `/tmp/bench.sh`, carded at 0.10 h; g01r07 survived the same exposure three times only by setting `timeout 1800` inside a 1900 s tool budget; p00s02's second timeout killed a data build mid-run (≈0.05 h). Twelve of 14 cells hit at least one tool timeout, but almost all were harmless foreground observations of a detached producer. **Why not a candidate:** frozen E2 `c6f11d8` already edits the same `run_dies_with_the_session` entry *and* pushes the opposite way — *"Prefer launch/wait/exit-status handling and subsequent evaluation in one foreground script"* — so a parallel entry would confound interpretation, and a "zero tool-timeout kills" counter is gameable by moving the kill into an inner `timeout` while still losing the work. E2's spec already requires reporting **lost work** as a guardrail failure; this belongs there, phrased as *hours of model-executing work lost to a tool timeout, and whether the foreground script carried an appropriate lifetime/handle*. Recording it does not redefine E2's frozen primary metric.

### What the protocol should stop doing, with the actual cost

1. **Stop shipping the `awm wma review` step 4b and the whole "If a world-model agent is installed" section into this line's sandbox.** **[verified here]** Two cells executed it and got `awm: error: argument group: invalid choice: 'wma' (choose from 'slurm','traj','run','split','ptb','exp_protocol','sandbox')` — g01r07 L2552 at 18:36:06Z, p00s02 L1878 at 19:15:53Z. Direct cost ≈0 h each; the real cost is ~20 lines of dead instruction read at bootstrap by 8/8 cells. This is itself a protocol-tree edit, so it is candidate-shaped; given a measured benefit of essentially zero, I recommend **recording it and folding it into the next unrelated `SKILL.md` revision rather than spending a 4-cell screen on it**.
2. **Stop reading `collect`'s `preflight_fail`/`n_overrides` as the friction record.** 6 of ~33 events; 6/8 cells read `0/0` while every one hit a blocking `check` error. A reading rule, not a protocol change.
3. **Stop reporting raw `waiting_hours` or monitor low-memory spans as post-exit idle.** The monitor number is a sum of last-minus-first low-sample timestamps that discards singletons; no cell in this window has an unconditional positive post-exit lower bound.
4. **Do not stop:** the comparator rule, "one card one intervention", rule 6, and `falsified_if`. Each visibly changed a decision in ≥2 cells, and rule 6 produced the window's highest-scoring artifact. A rule may only be weakened by writing down why it was wrong, not merely inconvenient.

### Observations that strengthen existing directions (not proposals)

- **#26 / J** — 8/8 protocol cells ran GPU training before the covering card existed; plus one 15.1-minute pre-lock declared evaluation and one post-launch relock of the shipped configuration (both g01r08).
- **#5 / H** — 8/8 cells hit `setup.data: required` or `n_examples: must be a positive integer` on a card that trains nothing; 3 cells fabricated a file (g01r04's placeholder JSONL, p00s02's `NO_TRAINING_DATA.md` reused by 3 cards, g01r07's 180-byte `exp07_generation_config_before.json` reused by 2); ≥4 cells carry non-applicable `setup.data` entries; 6 overrides concentrated in 2 cells. Zero overrides ≠ no workaround.
- **#27 / K** — 2/8 cells (g01r03 override, g01r08 pre-lock evaluation launch). g01r06's FAIL is a filename typo, not this mechanism.
- **#3 / B v2** — c01s06's 2.93 h fully discarded pass for missing `stop_token_ids`; g01r04's parser-inf crash and 60,937/63,892 run-on samples; c01s06's six orphan-engine `pkill` round trips; c01s02's GRPO bring-up. Fresh, large, and in both arms.
- **#6 / G** — two fresh **control** exposures of the TRL EOS zero-gradient trap (c01s01 0.17 h, with `'loss': 0.0, 'grad_norm': 0.0 … 'clipped_ratio': 1.0`; c01s04 0.175 h). **Important design caution:** with **0/8 protocol cells running RL**, a G screen on protocol cells would very likely show zero exposure and be uninformative under J's own "no exposure is not success" rule. G's testability, not its evidence, is the blocker.
- **#21 / P2** and **#22 / P3** — ≥6 and 3 fresh cells respectively; both remain separate, un-stacked.
- **#17** — g01r05 adds a third manifestation of raw-field/script-owned termination, with the cost appearing as a data rewrite rather than an override.
- **#18, #28** — mixed as quantified in §2; both stay observations.
- **#19 / #24** — repeat variance quantified: g01r06's identical weights read 943/934/935 on 1319 across servers; c01s07's same greedy artifact moves 22 of 500 items between reads; c01s04's b105 and b70_rp share an accuracy to seven digits while differing on 162 items.

---

## 5. Open questions and the next independent wave

### Open questions this window could not settle

1. **Rule 7's intent.** Does *"No item from it in `failure_examples`, `watch_set`, or training data"* permit evaluator-derived **item IDs**? The shipped text has no ID-only exception; g01r05 redacted content and kept IDs while explicitly citing the rule; g01r03 stored verbatim text and golds. **This is a policy question for the user/planner, and it blocks any enforcement design** — including whether the checker-only test reference may be read by a new mechanism at all. Unknown is the correct state; it should not be resolved by inventing an exception in a screen.
2. **E2's retention proof remains unresolved.** The two cited events give strict selected-event intervals [0, 0.450556 h] and [0, 0.250833 h], so from those premises alone the number of *proven* failing cells is **0** with 2 strongly suspected. This is **not** saturation, **not** grounds to swap to G or P1, and **not** grounds to redefine the frozen `post_exit_idle_h` metric. **Nothing in this window can settle it**: the retention question is about the exact cohort g01s01–g01s08, and none of those cells is in this window. What would settle it: a timestamped actual-producer-absence observation before the deadline (g01s03 would need `e ≤ 15:38:39`, g01s08 `e ≤ 10:39:54`), a retained exit result, or an explicitly adjudicated bounded-teardown assumption recorded as *conditional*.
3. **A v2's definitional edge case must be resolved before A launches.** 2 of 8 protocol cells (g01r05, g01r06) and 2 of 6 controls never produce the frozen endpoint at all, because they designed greedy in from source inspection. Is that a fail, an exclusion, or a distinct category? The frozen clock cannot answer it, and the answer changes A's readout on a quarter of its cells. The rest of A's metric is *not* saturated (4/8 pass, 2/8 clear failures), so A remains discriminating once this is settled.
4. **Is the two-cell cross-n comparator finding representative?** My scan of 46 comparator-bearing cards used a filename heuristic for the comparator's n; I corroborated both hits against the cards' recorded values but did not open all 46 artifacts. An exact structured parse would settle the base rate.
5. **Per-item identity of official runs** remains unresolved wherever the official log was not retained (ledger #24/#29). g01r05's developer/official pair differ by exactly one item (1005 vs 1004 of 1319) on hash-identical weights under different serving configurations; that is repeat variability under a changed configuration, not evidence about either knob, and not per-item identity.
6. **Does the D-mechanism's dose-response hold?** Early first durable write bounded the loss to 0.25/0.49 h versus 0.849/0.86 h in four cells. Worth reading in D2's block; not worth its own screen.

### Suggested next independent wave — scientific readiness only

This is a **scientific-readiness ordering**, explicitly separate from ownership, native-reservation isolation and per-receipt release authorization. The user owns authorizations; the planner owns queue, git and experiment decisions. **I make no queue decision, release nothing, and treat no manifest, ledger "queued" word or checked file as a registered held cell.** Per the operator view, D4 (91060–91063) and the stale E4 (91064–91067) are excluded from the scientifically usable buffer, and 21 held cells remain behind the operational gate.

**Readiness order, by evidence strength in this window:**

1. **D2**, once designed, tested, specified and given its own immutable manifest — largest reproduced mechanical mechanism (8/14 cells, both arms), and its v1 defect is confirmed rather than suspected. Its old block is a whole-block replacement question, not a mutation.
2. **B v2 (`9f294c3`)** — already frozen, and this window adds the largest single discarded block in it (c01s06's 2.93 h) plus g01r04's parser-inf crash. No redesign needed.
3. **H (`b52e5f2`)** — already frozen, now with 8/8 fresh mechanism exposure and 3 fabricated-file cells. No redesign needed.
4. **Proposal 2 (comparator n)** and **Proposal 3 (P4 rule 8)** — new single items; both need spec, construction off guard drift `2f64581`, tests, forward review and an immutable manifest before they are anything at all.
5. **A v2** — after question 3 is answered; the metric is discriminating but its edge case is live on 2/8 protocol cells.
6. **J, K, E2** — frozen and unregistered; J and K need a fresh scientific/operational audit rather than automatic promotion into a wave, and E2 needs question 2 adjudicated first.

**Dependency notes.** Each candidate remains baseline-relative and single-item; none is stacked, and no frozen manifest or metric is silently changed. Winners need a second independent 4-cell block before any score claim, and held-out confirmation before promotion — which no candidate can obtain today, because **no held-out AIME result exists for any variant**. Two later clean cells (c01s05, p00s03) and one placement-quarantined result (c01s08) are outside this window's frozen denominator and must not be folded into these means; their trace review is separately due.

---

## Accepted, rejected and uncertain

**Accepted (well-evidenced in this window)**

- 14/14 cells ship a greedy-resolving decode config; 10/14 measured a same-weight gain of +5.7 to +15.4 pt; the magnitude tracks how damaged the sampled parent was, not the intervention.
- The format/termination SFT is the dominant score term in every cell (+55 to +67 pt).
- The greedy-parent save-validation failure occurs in 8/14 cells across both arms, on Trainer saves, weight-averaging saves and a LoRA merge.
- Frozen D's predicate is family-blind; two eval-only false blocks are CPU-replay-confirmed and 16 of 56 cards are structurally exposed; only `train_sft.py` and `soup.py` save in these cells.
- `comparator_same_protocol` passed 18 times without verifying n, in 8/8 protocol cells, and two locked cards declare a comparator at a different n than their own protocol.
- 8/8 protocol cells ran GPU training before the covering card existed; one launched a declared evaluation 15.1 min before its lock.
- 8/8 protocol cells hit H's `setup.data` schema pressure; `collect` records 6 of ~33 friction events.
- Direct protocol overhead is 1.7–2.1 % of the session.
- Protocol-installed cells stop earlier than controls (8/8 vs 3/6 with ≥1.5 h unused; Fisher p ≈ 0.055), and two cells' stated time reasons are contradicted by their own timers within a minute.
- On-policy RL adoption is 0/8 protocol vs 4/6 control (Fisher p = 0.0150) — an observation about recipes, not a protocol effect.
- 56/56 cards closed, `fields_filled` 1.0, zero stop-hook blocks, no work lost at session end in 14/14.

**Rejected (claims I decline to carry forward)**

- Any between-arm score effect in this window (guard − control = −0.016, Welch t = −0.66).
- g01r03's record as evidence of protocol efficacy or grounds for promotion.
- g01r05's `stop_token_consistent` failure as a rescued mis-terminated run, a +65-pt cause, or a reversal of ledger #9.
- The claim that card-rule breaches contaminated training or invalidate any official score; and the opposite claim that cards can never affect score.
- `test_item_text_absent` / any bare question-or-gold scanner as currently justified.
- `long_run_defers_its_only_write` as a new candidate — B v2's frozen guidance already covers its only non-D source cell.
- c01s07's "greedy spelling" check as new — frozen D already keys on `do_sample`, and passes c01s07's artifact.
- A blanket exemption of all non-training families from D (it would drop the real `merge` save hazard).
- "A v2 is saturated in the control arm" (4/6, not 6/6) and "A v2 is saturated in the protocol arm" (4/8 pass, 2/8 fail, 2/8 endpoint absent).
- Treating GPU release, an artifact mtime, `train_runtime`, or an empty GPU-process list as producer exit; treating monitor low-memory spans or raw `waiting_hours` as post-exit idle.
- Summing D's failed compute and E's post-exit idle into one candidate's savings.
- Equating more unused hours with a lower score.
- Folding an ordinary-path comparator check into frozen K, or re-registering C without a discriminating predeclared design.

**Uncertain (unknown is the answer, and it should block any pass/fail that depends on it)**

- Whether rule 7 permits evaluator-derived item IDs.
- E2's strict non-saturation for cohort g01s01–08.
- Whether any of the 14 structurally exposed non-saving cards beyond the two replayed ones would actually have been blocked by frozen D (their parents' configs at lock time are not retained).
- Whether the mechanical losses cost score at all — 3.79 h lost in c01s06 (0.7415) versus ~0.5 h in g01r03 (0.8279) argues no, at n=14.
- Official per-item identity wherever the official log was not retained.
- Whether the 2-of-46 cross-n comparator rate is the true base rate.
- Whether g01r08's exp-07 (`merge` family, evaluation argv) represents a class or a one-off, which is the only measured false-positive residual of D2.

---

## Reusable meta-knowledge (durable loop lessons)

1. **Check the frozen candidate text, not the label — in both directions.** This window produced two coverage errors in opposite directions: a proposal was made for something B v2 already says verbatim ("dump raw draws to disk before scoring and guard the parser for inf and nan"), and a proposal was made against a false positive D does not actually produce (D returns `pass` when `do_sample` is true). Read the predicate, then apply it to the actual artifacts.
2. **A metric that is structurally zero in one arm cannot compare arms.** `pitfalls_cost_h` is a sum of card entries and is 0.0 for every protocol-free control by construction. The protocol's own KPI is unmeasurable in its own comparator arm; reviewer-derived per-cell loss ledgers are the only cross-arm reading, and their definitions differ per reviewer.
3. **A filled field is not an exercised one.** `fields_filled = 1.0` in 8/8 cells while g01r04 declared a 500-item `watch_set` in three cards and left `watch_set_result: null` in all three. Complement `fields_filled` with an "exercised" count before treating it as quality.
4. **Distinguish executed maximum n from the n behind the selection.** They differ in 2 of 14 cells here, and a threshold saturated on the first reading (14/14 ≥500) is not saturated on the second (12/14).
5. **A decode observable needs an explicit rule for cells that never measure.** Four of 14 cells shipped greedy from source inspection and never produced a same-weight endpoint. Screens built on a latency between two events must predeclare what happens when the second event never occurs.
6. **Report tool timeouts by what they killed, not by their count.** Twelve of 14 cells hit a Bash tool timeout; only two killed a model-executing command, and only one lost ≥0.1 h. A raw timeout counter is noise; a "hours of model-executing work lost" counter is signal — and the latter cannot be gamed by moving the kill inside an inner `timeout`.
7. **Ship-path text that the sandbox cannot execute is dead weight.** `SKILL.md` advertises `awm wma review` in step 4b and a dedicated section; two cells ran it and got `invalid choice: 'wma'`. Audit the shipped skill against the shipped CLI when the six `EXP_PROTOCOL_SHIP` paths change.
8. **Quantify the metrics blind spot rather than restating it.** In this window, `collect` recorded 6 of ~33 trace-visible friction events, and 6 of 8 cells read `0/0` while every one hit a blocking `check` error. The number, not the warning, is what stops a synthesis from reading `preflight_fail: 0` as clean.
9. **When a scientist's own repair matches a frozen candidate's text, that is mechanism evidence, not an outcome.** c01s02 independently arrived at `echo $! > /tmp/x.pid; while kill -0 $PID` — E2's prescription — after paying ~2 h. It calibrates the mechanism; it does not resolve E2's retention proof, which concerns a different cohort.
10. **Early durable writes bound the cost of a save-path failure.** Two cells with `--save-steps 150` lost 0.25/0.49 h to the same fault that cost 0.849/0.86 h in two cells that saved only at the end. Record this as a design note on the candidate that prevents the fault, not as a separate candidate.
11. **A preflight FAIL can *cause* an ordering violation.** g01r08 responded to a future-comparator FAIL by launching the missing evaluation 15 minutes before locking the card that declared it. When designing a blocking check, ask what the cheapest way around it is and count that as a screen observable.
12. **Equal scalar accuracy is not equal correct items, and this is now verified twice.** g01r05's exp-04 and exp-07 both score 994/1319 while disagreeing on 154 items (77 each way, re-derived from retained developer logs); c01s04's b105 and b70_rp share 0.7414708 while differing on 162. Never infer artifact identity from a matching score.

<!-- END SYNTHESIS window=window04-local -->

---

The synthesis above is the deliverable, emitted as text only. Nothing was written to the repository or anywhere else — `Write` is disabled in this session, and no Slurm, git, model, evaluator or network operation was performed. Session identity for your record: **`ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70`**, PID 2593119, CLI 2.1.260, `claude-opus-5[1m]`, effort max, plan mode, tools `Read,Grep,Glob,Bash`.

Three things in it differ from the helper reports and may change your reading, so they are worth checking first: frozen **D already passes** c01s07's `do_sample:true` spelling (so that "new spelling check" is not new, and D's residual defect is scope only, now sized at 16 of 56 cards); **B v2's frozen text already covers** g01r04's 1.6 h deferred-write incident, which removes that proposal's only non-D source cell; and the **comparator-n defect is 18 events across 8/8 protocol cells with two actually-locked cross-n comparators** (g01r03 exp-08, p00s02 exp-07), which is larger than either baseline report alone showed and is genuinely outside frozen K's opt-in path.
