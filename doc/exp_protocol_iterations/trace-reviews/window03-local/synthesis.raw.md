# Raw local Opus synthesis (unaltered output)

Session: `09a3ddd9-6e5a-4c03-9398-c7d83245c65e`, Opus 5[1m], max effort. The planner read the full output. Known errors and rejected interpretations are intentionally retained here; use [planner-decision.md](planner-decision.md) for the adjudicated decision.

---

I'll start by reading the brief.
I have everything. Here is the synthesis.

---

# Trace review window 03 (local Claude) — synthesis

Written from the eight NEW cell reports and one calibration report in `doc/exp_protocol_iterations/trace-reviews/window03-local/cells/`, the three group appendices in `reviewer-notes/`, the frozen `launch.json` (harvest `9c7596a`), the interim window record, the directions ledger, the Round 02 independent-screens spec and the E-replacement spec. `L<n>` refers to the cell's `solve_parsed.txt.gz`; timestamps are UTC. **SAID** = the scientist's claim; **SHOWS** = what the trace or artifact establishes. **This is evidence. Every retention, withdrawal, release and promotion decision belongs to the Codex planner.**

## 0. Scope and exclusions

- **NEW = exactly 8 cells**: controls `c01r04 c01r05 c01r06 c01r07 c01r08`; session guard `g01r01 g01r02 g01s04`.
- `c01r03` is **calibration only** — reported for instrument comparison across windows, excluded from every NEW count, mean, and evidence threshold.
- `p00r16` is FAILED/incomplete (`complete=false`, `eligible=false`, no `metrics.json`). Its scorer-failure report is **mechanism evidence only**. Its n=500 developer read of 0.712 is not an accuracy observation and is not used anywhere below.
- `g01s01 g01s03 g01s06 g01s08` (commit `66ebd39`) are **outside this frozen window**. Not merged, not awaited.
- v3 protocol, session guard and no-protocol control are **three distinct variants**. Nothing here is a promotion claim; the guard block is 3/8.

---

## 1. Header metrics: per cell and per variant

### 1.1 Per cell (NEW), plus calibration

| cell | variant | official acc | items | hours used | h→1st real train | protocol h | E idle sum / max event | greedy shipped | RL | RFT | largest eval n | unused h at stop |
|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|---:|---:|
| c01r04 | control | 0.792267 | 1045/1319 | 8.71 | 0.40 | 0 | **1.27** / 0.83 | yes | no | yes | 600 | 1.30 |
| c01r05 | control | 0.724033 | 955/1319 | 9.07 | 0.25 | 0 | **0.25** / 0.15 | yes | no | yes | **200** | 0.93 |
| c01r06 | control | 0.756634 | 998/1319 | 8.51 | **0.87** | 0 | **2.45** / 0.36 | yes | no | yes | 1319 | 1.48 |
| c01r07 | control | 0.734647 | 969/1319 | 8.42 | 0.15 | 0 | **0.48** / 0.30 | yes | no | yes | 500 | 1.57 |
| c01r08 | control | 0.778620 | 1027/1319 | 8.36 | 0.14 | 0 | **2.05** / 0.38 | **no** | **yes** | yes | 1319 | 1.63 |
| g01r01 | guard | 0.710387 | 937/1319 | 7.59 | 0.42 | 0.12 | **0.145** / 0.072 | yes | no | yes | 1319 | 2.42 |
| g01r02 | guard | 0.777862 | 1026/1319 | 8.88 | 0.33 | 0.11 | **0.24** / 0.17 | yes | no | yes | 500 | 1.12 |
| g01s04 | guard | 0.735406 | 970/1319 | 7.62 | 0.63 | 0.15 | **0.48** / 0.40 | yes | no | yes | 800 | 2.38 |
| *c01r03* | *control (calibration)* | *0.778620* | *1027/1319* | *8.73* | *0.12* | *0* | *0.398 / 0.072* | *yes* | *no* | *yes* | *1319* | *1.27* |

All eight NEW accuracies are exact `k/1319`; the reviewers' stderr-implied-n method reproduces `SE = √(p(1−p)/(n−1))` exactly (verified on c01r04: 0.011175, and on c01r05's n=200 read: 0.031653). Denominator for every official score is **n = 1319**.

### 1.2 Per variant (NEW only)

| metric | control (n=5) | guard (n=3) | difference |
|---|---:|---:|---:|
| accuracy mean | **0.757240** | **0.741218** | −0.016022 |
| accuracy sd (sample) | 0.0287 | 0.0341 | — |
| accuracy range | 0.7240 – 0.7923 | 0.7104 – 0.7779 | fully overlapping |
| mean items correct | 998.8 / 1319 | 977.7 / 1319 | 21.1 items |
| hours used | 8.61 | 8.03 | −0.58 |
| h → first real train launch (mean / median) | 0.36 / 0.25 | 0.46 / 0.42 | +0.10 / +0.17 |
| protocol hours | 0.00 | **0.127** (1.4 % of session) | +0.127 |
| E idle (dead-process wait), mean | **1.30** | **0.29** | −1.01 |
| unused budget at stop, mean | 1.38 | **1.97** | +0.59 |
| greedy shipped | 4/5 | 3/3 | — |
| RL launched | 1/5 | 0/3 | — |
| RFT attempted | 5/5 | 3/3 | — |
| largest eval n ≥ 500 | 4/5 | 3/3 | — |
| full 1319 read in-session | 2/5 | 1/3 | — |

**The guard−control difference is not resolvable at this n.** Welch t = **−0.68**, df ≈ 3.7, **p ≈ 0.53**; the pooled SE of the difference (0.0235) is 1.5× the difference itself, and the guard sd (0.034) exceeds it. For context only, window 02 recorded the v3 protocol pool at 0.688563 (n=14) and the control pool at 0.755724 (n=10); both variants in this window sit above the v3 pool, but they are different variants and the pools are the planner's to update.

### 1.3 Arithmetic and unknown-value checks

Corrections the reviewers already applied (all verified):

| field | generated value | corrected | basis |
|---|---|---|---|
| `largest_eval_n` c01r05 | 353 | **200** | stderr 0.031653 = √(0.725·0.275/199) exactly |
| `largest_eval_n` c01r04 | 590 | 600 | `--limit 600` |
| `largest_eval_n` c01r06 | 1330 | 1319 | full test set |
| `final_model_written` c01r05 / c01r06 / g01r01 | 07:25 / 07:15 / 07:07 | false positives | string occurs in a `TaskCreate` subject / card text, not a write |
| `protocol_hours` g01r01 | 0.18 | **0.12** | the 649 s "protocol" call at L9471 is one lock + four full-1319 evals |
| `own-evaluator test-set cmds` g01s04 | 1 | **0** | the command is `build_rft.py … --out /tmp/rft_test.jsonl`, a data build |

Discrepancies I found that are **not yet reconciled** — the planner should decide the convention before any of these feed a screen threshold:

1. **`pitfalls_hit`, g01r01: 7 (`guard-collect.csv`) vs 4 (reviewer).** The reviewer enumerated losses at or above its reporting threshold (0.10 + 0.30 + 0.25 + 0.05 = 0.70 h) plus a 0.02 h `cd`-persistence trap = 0.72 h, which matches `collect`'s `pitfalls_cost_h` exactly. So the hours agree and the **counts** measure different things: card entries vs distinct ≥-threshold losses. g01r02 (4/1.60) and g01s04 (6/1.60) agree between the two instruments.
2. **`n_relocked`, g01r02: 1 (`collect`) vs 2 relock events (trace, L6867 11:41:23Z and L6977 11:41:45Z).** `collect` appears to count *cards* relocked, not relock *events*. The window record's "two relocks" total (1+1 from the CSV) is therefore a card count.
3. **c01r04 header inconsistency:** "three sequential SFT rounds on 173M fresh tokens … (64.9M/59.4M/28.8M)". Those three sum to **153.1 M**; 173.1 M requires the fourth run (20.0 M), which §2 states correctly. The ranking claim (c01r04 landed the most training volume) is unaffected; the "three rounds" label is wrong.
4. **c01r04 §7 parenthetical:** "+0.036 over the v3 baseline mean 0.6886" — 0.7923 − 0.6886 = **0.1037**. The +0.036 is the gap over **c01r06** (0.7566), which c01r06's own report states in the same direction. Mislabelled comparator, not a wrong number.
5. **c01r06 E total** stated ≈2.45 h; its own components sum to 2.47 h (1.755 + 0.49 + 0.13 + 0.10), and it says "eight" evals while listing nine durations. Immaterial to the verdict.
6. **`waiting_hours` is not comparable across cells.** g01s04 alone reports `sample_eval 2.01 h` as a first-class category (four n=800 reads at ~25 min); every other cell's evaluation cost is absorbed into `waiting_on_runs`. Do not average this column.

**Calibration read.** c01r03 re-measured in this window gives E = 0.398 h and D = 1.34 h. Window 02 recorded D as 4/8 cells totalling 3.3 h across p00r11, p00r14, c01r01, c01r03. If that figure counted marginal overhead rather than the re-run wall clock, the two reconcile; if it apportioned ≈0.8 h/cell, **window 02's D total is understated by ≈1.6×**. This changes D's expected saving and should be reconciled before D's threshold is fixed. The E instrument also differs across windows: measuring from `task/system_monitor.log` (60 s GPU sampling) plus NCCL `W903` shutdown stamps is strictly more sensitive than reading sleep durations off the timeline, so earlier windows measured the coarse way will under-report E.

---

## 2. Ranked explanations of the current score differences

Two things need explaining: the within-window spread (0.7104 – 0.7923, 82 items) and the guard−control difference (−0.016).

### Rank 1 — The first SFT stage sets the score; everything after it moves ≤4 pp. *(8/8 NEW)*

Every cell's initial supervised stage moves the number 55–66 pp and no later stage moves more than ~4 pp.

- c01r07: 6.7 % → **71.8 % @500**; *"Everything after this moved 0.8 pp"* (report §7, `baseline.json` 0.0667 / `sft_full_500.json` 0.718).
- g01r02 exp-02: **+65.3 pp**, renderer verified byte-for-byte against `templates/gemma3.jinja` before launch.
- g01s04 exp-02: 0.040 → **0.6267 @150**, format compliance 0.287 → 0.980, token-cap hits 75/150 → 3/150.
- g01r01 exp-02: 0.0267 → **0.6867 @150**, mean completion 5777 → 482 chars.

**This is a recipe observation, not a protocol intervention.** The arms overlap completely on the variable that matters (guard 88k–150k rows, controls 90k–166k rows) and training data is explicitly off the allowed surface. It is ranked first because any explanation that ignores it is explaining ~4 points of a ~66-point effect.

### Rank 2 — Landed training volume after the first stage, and what displaced it. *(5/8 NEW have an explicit attribution)*

This is the one place where lost hours convert into score, and the conversion is indirect: hours lost do not cost points, they cost a *training stage*.

- **c01r04** (highest, 0.7923): four stages, 64.9 + 59.4 + 28.8 + 20.0 = **173.1 M tokens** on disjoint fresh data, measured trajectory 78.5 → 81.3 → 81.0 @300.
- **c01r06** (0.7566): **96.6 M** over two rounds, and it names the cause itself — *"Roughly 1.5 hours went to those, which is why there was no third training round."* (footer).
- **c01r05** (lowest control, 0.7240): two rounds landed; the third (v3, 30,669 RFT rows) is measurably worth nothing — **145 vs 145 correct on the same 200 items** (L7869, 16:25:34Z).
- **g01r01** (lowest overall, 0.7104): the ceiling was set at exp-02 (88k rows × 2 epochs); its sampled SFT read 0.6867 and greedy bought only +3.3 pp, against +10.7/+10.67 in the other two guard cells.

**Counterevidence:** c01r08 landed only two SFT rounds and scored joint-highest (0.7786) — because GRPO plus a checkpoint soup substituted for a third SFT stage (0.670 @300 → 0.7533 @300 → soup 0.77714 @700, +3.8 pp ≈ 2.4 SE over its ingredients). Volume is one route to the ceiling, not the only one.

### Rank 3 — Decode configuration: large per-cell lever, but it does **not** order this window. *(7/8 shipped greedy; measured same-weight gains in 5 cells)*

Measured, same-weight gains: c01r03 **+6.0 pp** @200 (0.785/0.725, L5369/L5418 — the cleanest A/B in the line), g01r02 **+10.7 pp** @150 (0.700 → 0.8067, confirmed by an immediate identical rerun), g01s04 **+10.67 pp** @150 (predicted at +9.75 from a 400-item A/B before it ran — the tightest mechanism-to-result match in the window), c01r05 **+6.7 pp** @150, c01r04 **+4.0 pp** @200, g01r01 **+3.3 pp**, c01r06 **+2.75 pp** @400 on the 0.01-clamp axis only.

**Strongest counterevidence in the window:** `c01r08` looked straight at the observable twice (L1336, 07:18:30Z prints the base `generation_config.json`; `train_sft.py:177-180` and `grpo_train.py:145-150` deliberately copy it into every checkpoint) and **shipped stock sampling** — `{'do_sample': True, 'top_k': 64, 'top_p': 0.95}` on the delivered artifact (L6064, 14:34:11Z) — and tied for the highest official accuracy in the window at **0.778620**. Any strong form of "greedy is worth 7–16 points per cell" is falsified here.

What greedy *does* buy, measurably and across cells, is agreement between the scientist's read and the grader's: own-vs-official full-test spread **0.23 pp under greedy** (c01r03, 1030 vs 1027) against **0.99 pp under sampling** (c01r08, 1014 vs 1027). That fits A v2's "grader observable" framing, not a score framing.

### Rank 4 — The n behind the final selection. *(5/8 show a reversal or shrinkage; 2/8 changed or nearly changed the shipped artifact)*

- **c01r04** — a real reversal it caught itself: soup123 was the n=300 winner at **84.0 %** (L5629, 13:10:46Z, *"best result. Promoting it"*); at n=600 it read 80.5 % and soup23 read 81.8 %, and the ship moved (L5950, 13:21:24Z). Its own footer names the mechanism: *"that was subset noise — the larger paired eval reversed it."*
- **c01r08** — the cleanest inversion: ckpt-180 leads ckpt-120 by **6.0 pp at n=300** (0.7533 vs 0.6933) and **loses by 0.4 pp at n=700** (0.73429 vs 0.73857).
- **c01r07** — shrinkage, not reversal: the shipping delta collapsed from +3.67 pp @300 to **+0.8 pp @500**, with the cell's own paired count `final wins 38, loses 34` (L5894) — McNemar p ≈ 0.72.
- **g01r01** — two reversals, one of them stable: exp-02 leads the *first 150* by 5.3 pp while trailing on all 1319. It shipped the full-set winner.
- **c01r05** (lowest control) — the ship went the other way: v2 led the soup by 1.33 pp at n=150; the soup led by 0.50 pp at n=200; the soup shipped. Its own paired counts were `soup>v2: 9, v2>soup: 8` (L7869), McNemar p ≈ 1.0, and the footer states this correctly and then ships on *"edged"*.

**Counterevidence:** c01r06 had the best evaluation practice in the window — two full-1319 reads, an explicit prefix-bias correction, a discarded repetition-penalty A/B — and finished third of five controls. Good evaluation protects the shipped artifact; it does not raise the ceiling.

### Rank 5 — Operational hygiene does not order accuracy in this window. *(counterevidence that must be carried)*

Spearman rank correlation between per-cell E idle and official accuracy across the 8 NEW cells is **ρ = +0.60** (n = 8, critical ρ at α=0.05 is 0.738, so p ≈ 0.12) — **positive**, i.e. the cells that idled more scored slightly higher. Not significant, but decisively not the negative relation E's theory of change predicts.

- c01r04 lost the most (2.46 h attributed, 2.04 h GPU idle, 1.27 h dead-wait) and scored **highest**.
- c01r05 was the cleanest operator in the window (90 % GPU-allocated, 0.25 h dead-wait, the only cell that verified the grader observable *before* training, a real paired item-level statistic, zero cost to the D trap) and scored **lowest of the controls**.
- c01r08 idled 2.045 h and scored **joint-highest**.
- Self-attributed `pitfalls_cost_h` in the guard arm (0.72 / 1.60 / 1.60) is likewise unordered against 0.7104 / 0.7779 / 0.7354.

### Rank 6 — Guard vs control as an arm effect: not readable here.

−0.016 ± 0.024, p ≈ 0.53, guard n=3 of a prescribed 8. Recorded as an interim descriptive value only.

---

## 3. Direct protocol cost versus indirect plan-shaping

### 3.1 Direct cost is ~1.4 % of the session and is essentially reproved as negligible

| direct cost | g01r01 | g01r02 | g01s04 |
|---|---:|---:|---:|
| `awm exp_protocol` CLI wall time | 28 calls, **9 s** | 24 calls, **9 s** | 17 calls, **30 s** |
| card authoring (`new → lock`, no other work inside) | ≈0.09 h / 9 cards (~36 s/card) | ≈0.10 h / 7 cards | ≈0.11 h / 5 cards (once bundled `--dry-run` compute is removed) |
| total protocol wall | **0.12 h (1.6 %)** | **0.11 h (1.2 %)** | **0.15 h (2.0 %)** |

Two measurement traps the planner should keep out of the ledger: the naive sum reports 660 s of "protocol" for g01r01 because one call is `lock exp-09` **plus four full-1319 evals** (L9471); and g01s04's 231 s exp-02 authoring includes a 121 s `train_sft.py --dry-run`. Card cost also *falls within a session* — g01r01 went from 53 s (exp-01) to 5 s (exp-07) as the scientist learned to write the whole card in one heredoc.

For scale: the guard arm's paperwork (0.127 h) is **10 %** of its own self-attributed pitfall hours (1.31 h) and **6 %** of the budget it left unused (1.97 h).

### 3.2 First train launch: no protocol signature

Guard mean 0.46 h vs control 0.36 h (medians 0.42 vs 0.25) — but the ranges interleave and the extremes are on the *control* side. The slowest first launch in the window is **c01r06 at 0.87 h**, a pure control, and the delay is visible bring-up work (data build + decontamination over 403,140 docs, a liger install/uninstall, a smoke train, a custom sparse-LM-head trainer written after measuring, and two throughput probes). The slowest guard cell, **g01s04 at 0.63 h**, spends its extra time the same way: a liger install and OOM fix, a flash-attention throughput comparison (*"16.45 samples/s, +10 % over eager; adopted flash_attention_2"*), two data rebuilds and three `--dry-run` token-budget checks. **Hours-to-first-launch in this window measures how much bring-up measurement a cell chose to do, not whether it wrote cards.** This is consistent with window 02's finding (protocol ~4 min *earlier*) in the sense that both windows find the difference inside the noise, with the sign flipping.

### 3.3 Waiting after a process has already exited: the largest arm-associated operational difference — and it favours the guard

| | control (n=5) | guard (n=3) |
|---|---|---|
| E idle per cell | 1.27, 0.25, 2.45, 0.48, 2.05 | 0.145, 0.24, 0.48 |
| mean / median | **1.30 / 1.27** | **0.29 / 0.24** |
| meets < 0.15 h (per-cell sum) | 0/5 | 1/3 |

The discriminator is mechanical and visible, and it is the *wait idiom*, not the arm:

- **Fixed clock sleeps sized to a pessimistic ETA** — every failing control. c01r08's six late waits are `sleep 900`–`sleep 1500` on jobs that finished in 2–4 min (1.556 h). c01r06 backgrounded *every* eval and polled with `sleep 660–1500` (1.75 h). c01r07's `sleep 2700` fired after a progress bar whose remaining estimate was ~15 min too long because `group_by_length` shortens later batches (0.296 h).
- **Artifact/condition polls** — the guard cells: `while [ ! -f ckpts/exp-02/final/config.json ]; do sleep 60; done` cost g01s04 0.05 h and 0.03 h; g01r02's five condition-polled waits each cost <1 min.
- **Foreground execution** — c01r05, the only NEW cell near target, ran every evaluation in the foreground, so completion returns the tool call by construction.

Two alternative explanations, both examined:
1. *"Guard cells stopped earlier, so they never entered the late eval-polling phase."* Partly true (guard used 8.03 h vs 8.61 h), but **g01r01 refutes it**: it ran ~24 full-1319 reads, most of them in its last 1.7 h, and still accumulated only 0.145 h — because it bundled evals into single foreground calls instead of backgrounding and sleeping.
2. *"n=3 vs n=5."* Real and unresolved. The remaining strict guard cells settle it.

**Do not read this as guard safety or as an E verdict.** It is the same evidence pointing at the wait idiom that E targets, and it happens to correlate with the arm at this n.

### 3.4 Early endings

4/8 NEW cells ended with **≥1.5 h unused and zero live processes** (c01r07 1:34, c01r08 1:38, g01r01 2:25, g01s04 2:23); c01r06 is at the line (1:29). Only **c01r05** gave an explicit budget-based stop reason: *"The three top candidates are within noise of each other at n=200; a larger eval would be needed to separate them, and I ran out of budget for that."*

The sharp finding is that the guard cells **do** price time with measured numbers and stop early anyway: g01r01 exp-06 *"Evals turned out to cost ~4 minutes, so 3.5h is far more than the remaining plan needs"*; g01r02 exp-07 rejects k=8 because *"160k completions is ~55 min at this model's ~8k output tok/s"*; g01s04 exp-05 rejects the full split because *"roughly 45 min per arm, 1.5 h for both."* So P4's mechanism is not "they don't price the clock" — it is that they price **the next run they know how to launch**, not the smallest run that could change a decision. c01r08 is the extreme: its final 2.5 h contained 1.556 h of pure post-exit idle, so *"no time left"* is not what its trace shows.

### 3.5 Confusion between developer defaults and official scoring — uncovered, both arms

This appears in **≥5 NEW cells across both arms** and is not covered by any frozen candidate.

| cell | own read | official | gap | note |
|---|---:|---:|---:|---|
| g01r01 | 0.7187–0.7202 (five full-1319 reads) | **0.71039** | **−0.83 pp, below the entire band** | `--max-connections 32` in-session vs `max_connections: 2` official |
| c01r06 | 74.68 % (full 1319) | 75.66 % | +0.98 pp | identical items, identical greedy config, 13 items |
| c01r08 | 76.88 % (full 1319) | 77.86 % | +0.99 pp | sampling config |
| c01r03 *(cal)* | 1030/1319 | 1027/1319 | 0.23 pp | greedy |
| c01r07 | 0.720 vs 0.660 on the same 100 items, **43/100 identical generations** (L5721) | — | 6 pp | differs only in `--max-connections` / `--gpu-memory-utilization` |
| g01r02 | two identical greedy runs disagree on **7/150 items** | — | — | the cell then used ±1 pp as its noise floor |
| g01s04 | 0.7333 vs 0.72 on the same 150 items | — | — | *"about 2 items in 150"* |

Three consequences. (i) The gap is the same order as the deltas several cells used to pick a winner. (ii) **Greedy does not remove it** — c01r07's 6 pp on identical greedy weights is the proof. (iii) The sign is *not* consistent (g01r01 below its band, c01r06/c01r08 above), which a pure concurrency story must explain. g01r01's exp-09 card also optimized a first-150 criterion *because the scientist did not know what subset the grader uses* — the confusion has already shaped a card, not just a read.

Prefix bias, measured against official numbers this window: **3.32 pp** at n=400 (c01r06), **3.8 pp** at n=500 (g01r02), **2.6 pp** at n=600 (c01r04), **1.2 pp** at n=800 (g01s04), **3.2 pp** at n=150 (c01r03) — inside C v2's 2–7 pt band and shrinking with N, as C v2 assumes. **g01r01 is the exception worth keeping**: the first-150 subset ranks *checkpoints* differently (exp-02 +5.3 pp on the first 150, behind on all 1319), so the effect is checkpoint-dependent, which is a stronger statement than "the front is easier."

---

## 4. Candidate scorecard

### 4.1 Frozen candidates

| candidate | mechanism appears (NEW) | hours (NEW) | target metric readable? | saturation / exposure | counterevidence |
|---|---|---:|---|---|---|
| **A v2** decode | observable found in 7/8; decode choice measured in 7/8 | ~0.4 h + the clamp's 2.75 pp | **Yes** — primary clock met 7/8 at 0.006–0.13 h; post-choice unmeasured-sampling cards **0/8** | **Primary observable saturated** (same reading as window 02). The **correctness guardrail is not**: c01r08 shipped stock sampling; c01r06 shipped `1e-06` clamped to 0.01 for 4.3 h; c01r04 shipped the HF-invalid `do_sample:false + temperature:0.0` | c01r08 shipped no greedy and tied for top score. c01r06 never ran a greedy-vs-sampling A/B at all — its choice rests on source reading. The vLLM override line **is not proof of greedy**: it appears identically under sampling (c01r04's `eval_r1_sample.log`); the discriminating string is the clamp warning *"temperature … maxed it out to 0.01"* (400× per clamped eval in c01r06). c01r04 grepped `default_sampling_params` / `Using default` and **missed the line that was in the very file it grepped** |
| **B v2** vLLM offline sampling | **8/8 cells** show ≥1 named mechanism | **≈5.3 h** | **Yes** — per-cell: 0.02 / 0.59 / 0.49 / 1.29 / 0.18 / 0.30 / 0.90 / 1.15 h | **Widest exposure in the window.** Against the <0.3 h/cell target, only 2–3 of 8 pass | Two mechanisms fall outside the current wording: `limit_mm_per_prompt={"image": 0}` silently corrupts generation on a multimodal Gemma-3 checkpoint with **no error and 6.9 % apparent pass rate** (c01r05, 0.96 h across two passes, single cell → observation); and an own arithmetic-dedup filter that hangs on `**` expressions (c01r05, 0.23 h). c01r08 shows the cheap immunity: **keep on a parseable `ANSWER:` line, never on `finish_reason == "stop"`** |
| **C v2** eval n | ≥500 met **7/8**; reversal or shrinkage in **5/8** | — | **Partly.** The n-threshold is nearly saturated; the *decision-quality* half is not | Paired statistics: computed in **3/8** (c01r05, c01r07, g01s04), used **before** the decision in **1/8** (g01s04, McNemar z=0.40, verdict `inconclusive`, shipped with the reason written down) | **The sharpest counterevidence in the window**: c01r05 satisfied C v2 in form — correct paired discordant counts `9 vs 8`, McNemar p≈1.0 — and **shipped on "edged"** anyway. The rule produced the number, not the decision. What is missing is not the statistic but the instruction for what to do when it returns null. c01r07 computed its paired count *after* shipping |
| **D** parent generation config | **4/8** cells (c01r04 1.57, c01r06 1.20, c01r07 0.67, g01r02 0.55) | **3.99 h** | **Yes** — hours attributable to `GenerationConfig is invalid` | **Largest single-mechanism hour count in the window.** **Exposure caveat:** D is a *parent-choice* trap, not a greedy-adoption trap. 7/8 adopted greedy; 4/8 made the greedy directory a training parent; **3/8 (c01r05, g01r01, g01s04) are structurally immune** because greedy lives only in symlinked variant dirs and the next stage trains from the unpatched checkpoint. A 4-cell screen should expect ≈2/4 exposure, not 4/4 | None against the mechanism — the stack trace is identical in all four. The window-02 apportionment must be reconciled first (§1.3) |
| **E** wait on process | **8/8** cells show E-class idle; **7/8** exceed 0.15 h | **7.36 h**, mean **0.92 h/cell** | **Yes, but the aggregation convention is undecided** (see §6 Q1) | **Directly contradicts window 02's "5/5 baseline already under 0.15 h."** Guard 2/3 over; controls 5/5 over by sum. **E's ≥7/8 saturation rule is defined on the prescribed full strict-guard block and this window cannot satisfy it** | Two failing mechanisms are **not** what E's current text is aimed at: g01s04's 0.40 h was a `while ! grep -aq "wrote …"` condition-wait on a **log string a crashed process would never print** (a file/condition-poll rewrite would not have caught it), and c01r06's 0.49 h was invisible to any GPU-memory heuristic because a **zombie vLLM engine held 66 GB while the trainer was dead**. Only g01r02's 0.17 h is a plain clock-sized `sleep 1700` overshoot. Also: E hours do not predict score (ρ=+0.60, §2 Rank 5) |
| **H** eval-only `setup.data` | **3/3** guard cells; controls unexercised by design | 0 hours; cost is record corruption | **Not as currently written** — see Proposal 3 | **12 non-training cards carry 12 non-applicable `setup.data` entries, and `n_overrides = 0` in all three.** g01r01 generated a **150-row placeholder jsonl and set `mixture_weight: 1.0`** to get past `data_files_exist` + `data_n_examples_match` (L1312, 06:44:58Z, 19 s) — the record now claims 150 training examples at full mixture weight for a card that trains nothing | The override counter reads perfectly clean while the record is corrupted. `fields_filled = 1.0` in 3/3 |

### 4.2 Queued candidates

| candidate | appears (NEW) | hours | verdict from this window |
|---|---|---:|---|
| **G** `trl_grpo_gemma_zero_gradient` | **1/8** cells launched RL (c01r08, a control); guard **0/3** | ~0 | **Weakened.** The one NEW RL cell hit a *termination-accounting* bug, not a zero gradient: the 4-step smoke **before** the fix logged `grad_norm` 0.358 / 0.343 / 0.252 / 0.371 (L4903-4906); `zero-grad lines = 0`. A G entry keyed on zero gradient would not have matched. The real mechanism was `tok.eos_token = "<end_of_turn>"` (106, not 1) so TRL reinforces the real terminator. g01s04's GRPO rejection is a **priced time/setup argument** — *"neither is set up and a first GRPO launch inside 4.6 remaining hours risks the whole batch for an unmeasured gain"* — not a belief the trainer is broken, so the entry would not have changed that decision either |
| **P1** `rft_from_fitted_parent` | outcome claim supported broadly; **signature readable in 1/3 guard cells** | 0.85–2.7 h/cell of flat-loss RFT | **Outcome supported, screen observable broken.** Outcome: c01r05 v3 = 145 vs 145 on the same 200 items; c01r07 round 1 = exactly 0.0 pp @300; c01r06 "worth only ~1 point" (confounded with 60k fresh rows); c01r08 "+2 pts — SFT had saturated". Signature: **g01r01 never prints a first-step loss** (all monitoring is `tail -c 150/400`); **g01r02 shows flat loss twice — once before a +2.0/+4.0 gain and once before a −3.3 loss**, so it does not predict the sign; **g01s04's stage is a mixture** (39,540 unseen teacher rows + 8,387 self-verified) and shows a real descent 0.319 → 0.2633, so the signature is absent by construction. A rule that stopped both g01r02 rounds would have removed that cell's third contributor |
| **I** `stop_token_consistent` / `appended_by` | **0/8** | 0 | No manifestation. 3/3 guard cells: `n_overrides = 0`, `preflight_fail = 0`. Nothing to screen against in this window |
| **P2** Gemma-3 logits OOM + 64 MB overlay | **8/8** | **≈1.42 h** (0.55 / 0.02 / 0.02 / 0.10 / 0.15 / 0.35 / 0.10 / 0.15) | **Not saturated.** Against the <0.05 h target, **only 2/8 pass**. Three distinct outcomes from one wall: c01r04 installed liger into the full 64 MB overlay and lost 0.52 h to 160+ null-byte `.py` files; c01r08 lost 0.131 h when 121 corrupted liger files took down `from trl import GRPOTrainer` an hour later; **c01r06 ran `df -h` in 25 s, uninstalled and wrote its own sparse-LM-head trainer — the cheapest resolution in the window.** The one-line fix c01r04 eventually found is `uv pip install --target <task>/pylibs` + `PYTHONPATH` |
| **P3** checkpoint not vLLM-loadable | **2/8** paid (c01r04 0.012 h, g01s04 0.15 h); **3/8 pre-empted** | 0.16 h | Below threshold as an hours candidate. **But there is a design conflict that must gate its wording**: c01r08's `grpo_train.py:145-150` / `train_sft.py:177-180` copy `preprocessor_config.json`, `processor_config.json` **and `generation_config.json`** from the base snapshot into every checkpoint — the P3 fix is exactly what **reinstated the stock sampling config** on the shipped artifact. A P3 line that says "copy the base configs" without excepting `generation_config.json` manufactures c01r08's outcome |
| **P4** pricing the remaining time | **4/8** end ≥1.5 h early with no live process | ~6.9 h of unused budget | **Not saturated** — 50 % against a ≤25 % target. Secondary observable *is* met 3/3 in the guard arm (last `alternatives_rejected` cites a measured cost). See §3.4: the mechanism is mis-specified as "does not price the clock" |

### 4.3 Proposals

Four proposals, each exactly one item on the allowed surface, each grounded in ≥2 cells. Everything else in this window is either a confirmation of a frozen candidate or an observation.

---

**Proposal 1 — CHANGED: E's single item must name the producing PID, not the wait style.**
*Surface:* the same three places E already changes (`SKILL.md` rule 9, the Stop-hook text, `pitfalls.yaml: run_dies_with_the_session`) — still one item, only the wording changes.
*Source cells (4):* **g01s04** (0.40 h waiting on a log string from a process dead 24 min — a condition-poll rewrite would not catch it); **c01r06** (0.49 h invisible to a GPU-memory heuristic because a zombie engine held 66 GB while the trainer was dead); **c01r04** (`tail -1` of a tqdm bar returned the **identical line** at 11:40:17Z and 12:09:35Z and went unquestioned for 29 min); **c01r05** (positive control: foreground evals ⇒ 0.25 h total, the best in the window).
*Wording:* the liveness test is the producing **PID** plus a *changing* tail; a log string, a file, a progress bar and GPU memory are all insufficient, and the reason each is insufficient is one clause each.
*4-cell metric:* per-cell sum of (producer process exit → next useful command) **< 0.15 h**, reported **alongside the per-event max**; plus the count of waits that test only a log string, a file, a tqdm tail, or GPU memory rather than a PID (target 0).
*Score guardrail:* block mean ≥ the then-current protocol pool mean − 0.03.
*Falsifier:* if ≥2/4 cells still exceed 0.15 h with the PID named, the wait idiom is not E's mechanism and E should be withdrawn rather than re-worded.
*Standing:* this is a wording proposal **conditional on E being retained**. The retain/withdraw decision belongs to the full strict-guard block (§5).

---

**Proposal 2 — NEW: a `pitfalls.yaml` entry on read-instrument concurrency and greedy non-reproducibility.**
*Surface:* one `pitfalls.yaml` entry, `check: null`, printed by preflight as a reminder.
*Source cells (7, both arms):* g01r01 (five full-1319 reads span 0.7187–0.7202, official 0.7104, `--max-connections 32` vs `max_connections: 2`), c01r06 (74.68 vs 75.66 on identical items and config), c01r08 (76.88 vs 77.86), c01r07 (0.720 vs 0.660 on the same 100 items, 43/100 identical generations), g01r02 (7/150 disagreement between two identical greedy runs), g01s04 (~2 items in 150), c01r03 (0.23 pp).
*Content:* greedy decoding is **not** run-to-run reproducible under vLLM continuous batching; the official grader runs at `max_connections: 2`; therefore a checkpoint comparison must be a mean over ≥2 reads and must record the serving concurrency next to the number.
*4-cell metric:* number of shipping/selection decisions whose margin is **below the cell's own measured repeat-read spread** (target 0), and number of cards recording the read concurrency with the number (target ≥3/4 cells).
*Score guardrail:* block mean ≥ protocol pool mean − 0.03; secondary guardrail so the entry does not eat GPU — hours spent on repeated identical reads ≤ 0.5 h/cell.
*Falsifier:* if 4/4 cells measure a repeat spread < 0.3 pp, the entry describes noise that never reaches a decision and should be withdrawn.
*Overlap, stated honestly:* this is adjacent to C v2 and ledger #19, but the mechanism is **serving concurrency, not sample size**, and both the control-B and guard reviewer groups arrived at it independently. If the planner prefers it as a sentence inside C v2, that is a re-cut of a frozen tree, not a screen — hence the separate entry.

---

**Proposal 3 — CHANGED: H's screen observable, not H's tree.**
*Surface:* screen observable only (same class as the A v2 correction already accepted on 09-03 08:45). No protocol-tree change.
*Source cells (3/3 guard):* g01r01 (6 non-training cards, 6 entries, **1 fabricated 150-row jsonl at `mixture_weight: 1.0`**), g01r02 (3 non-training cards, 3 entries), g01s04 (3 non-training cards, 3 entries) — **and `n_overrides = 0` in all three.**
*Why:* H's current metric ("zero fabricated data entries, zero `data_files_exist` overrides, `fields_filled` does not fall") reads *clean* on this window while 12 cards assert training data that does not exist. The harm migrated out of the override counter.
*4-cell metric:* number of non-training cards carrying a `setup.data` entry the card does not read (target **0/4 cells**), and number of placeholder/synthetic files created solely to pass a data check (target **0**). Keep `fields_filled` as the guardrail.
*Score guardrail:* block mean ≥ protocol pool mean − 0.03; `fields_filled` must not fall below the guard baseline of 1.0.
*Falsifier:* if the tree change removes the entries but `fields_filled` drops or cards become materially vaguer, H trades record corruption for record thinness and should be re-scoped.

---

**Proposal 4 — CHANGED: P1's screen observable, before P1 is ever selected as E's replacement.**
*Surface:* screen observable only; P1's `pitfalls.yaml` text (`a4c4954`, tree `7294c236`) is unchanged.
*Source cells (5):* g01r01 (signature asserted in the card, **never shown in the trace** — all monitoring is `tail -c 150/400`), g01r02 (flat loss preceded a **+2.0/+4.0 gain** in round 1 and a **−3.3 loss** in round 2, within one cell), g01s04 (mixed stage, **real descent 0.319 → 0.2633**, signature absent by construction), plus c01r05 (145 vs 145) and c01r07 (0.0 pp @300, but the cause was the discarded 5× RFT yield, i.e. B, not P1).
*Change:* read **hours spent on a pure-self-sample RFT stage that did not move a ≥500-item read**, not "hours after a flat first-step loss." Add, as the screen's readability condition, that the card's `result.training_summary` must carry the first 20 logged losses — otherwise the observable is unreadable in the very common `tail -c` monitoring style.
*4-cell metric:* hours on pure-self-sample RFT stages with no ≥500-item movement, **< 0.5 h/cell**; cards whose `training_summary` reports the first 20 losses ≥3/4.
*Score guardrail:* block mean ≥ protocol pool mean − 0.03; **≥1/4 cells must still attempt RFT** (the entry must not be readable as "never RFT").
*Falsifier:* if 4/4 cells print flat first-20-step losses and the outcome sign is still split, P1 is describing a property of fitted-parent RFT that does not license a stop rule, and it should return to observation.

**Marked observation, not proposals** (single-cell, or not a protocol surface): the `limit_mm_per_prompt` silent-corruption mechanism (c01r05, 0.96 h); the `**`-expression dedup hang (c01r05, 0.23 h); the P3/`generation_config.json` copy conflict (c01r08 — a **design gate on P3's wording**, not a candidate); every recipe finding (staged fresh-data SFT, checkpoint souping, GRPO, fp32 master weights, thousands-separator targets, the `ANSWER:` keep filter). **None of the control recipes may become a required protocol training recipe.**

**Confirmation of an existing candidate, not a new item:** "size the sampling pass from a measured tok/s" — c01r04 (74,730 prompts, ETA visible at 08:03:20Z and not acted on for 10 min, 0.37 h), c01r05 (116,838 prompts, 0.12 h), c01r07 (219,784 prompts, ETA 4:56:42), c01r08 (129,288 prompts), g01r02 (0.90 h, the largest single instance). Five cells, ≈1.9 h — already B v2's last sentence.

---

## 5. Pending triage — recommendations, not commands

Held pool: 28 jobs, 91046–91073, all `PENDING(JobHeldUser)` after the 18:42:47 restore. **None of the following withdraws anything, so the ≥8 compliant `PENDING(JobHeldUser)` floor is maintained at 28.** Only whole unstarted blocks are considered; no individual cell is selected on any outcome. No release is recommended and no scientific or ownership/native-isolation gate is bypassed — the external release actor from 18:30:22 remains unresolved, the underlying reservation still spans eleven nodes, and the strict guard block is 5/8 complete.

| block | jobs | recommendation | basis |
|---|---|---|---|
| **B v2** | 91050–91053 | **Retain, first releasable wave** (as already ordered) | 8/8 NEW cells, ≈5.3 h, only 2–3/8 meet the <0.3 h/cell target. Widest exposure in the window |
| **D** | 91060–91063 | **Retain, first releasable wave** | 4/8 NEW, 3.99 h — the largest single-mechanism hour count. Size the screen for ≈2/4 exposure, not 4/4 (§4.1), and reconcile the window-02 apportionment first (§1.3) |
| **C v2** | 91054–91057 | **Retain, first releasable wave** | The n-threshold is nearly saturated (7/8 ≥500), but the decision half is not: 5/8 reversals, paired statistic used *before* the decision in only 1/8, and c01r05 satisfied the rule and shipped against it |
| **A v2** | 91046–91049 | **Defer (behind D/B/C, as already decided). Do not withdraw** | Primary clock saturated 7/8 at 0.006–0.13 h with 0 post-choice unmeasured-sampling cards. The *correctness* guardrail is live in 3/8 cells and should become the screen's primary reading if A runs |
| **E** | 91064–91067 | **Retain held. Do not withdraw and do not pre-commit withdrawal** | **E's ≥7/8 saturation rule belongs to the prescribed full strict-guard block; this window supplies 3 guard cells and cannot satisfy it.** On what is readable now, 7/8 NEW cells exceed 0.15 h and the NEW mean is 0.92 h/cell — the opposite of window 02's baseline reading — so a withdrawal pre-committed on window-02 evidence would now run against the current data. Settle the aggregation convention (§6 Q1) before the block is read |
| **H** | 91068–91071 | **Retain, second wave** | Confirmed 3/3 guard: 12 non-training entries, 1 fabricated file, 0 overrides. Apply Proposal 3 to the screen, not the tree |
| **drift A v2 / drift B** | 91058–91059 / 91072–91073 | **Retain** | Same-generation baseline anchors; nothing in this window touches them, and they are what makes the 4-cell blocks comparable |
| **G / P1** (E-replacement, built, unregistered) | none | **Defer the choice to the full strict block. No registration recommended now** | The spec's selection rule is not yet satisfiable: guard is 3/8, GRPO cards are **0/3**, and P1's signature is readable in only **1/3** (asserted-not-shown in g01r01, absent by construction in g01s04). This window also **weakens G** — the only NEW RL cell hit a termination-accounting bug with `grad_norm` 0.25–0.37, not a zero gradient. If the choice were forced today the hours argument favours P1, but with Proposal 4's observable, not the current one |
| **P2** (queued, entry-only) | none | **Keep queued as an entry-only rider on P3's commit**, as the ledger already plans | 8/8 NEW, ≈1.42 h, only 2/8 under the 0.05 h target — the broadest uncovered exposure in the window, but it does not need its own screen slot |
| **P3** (queued) | none | **Keep queued; gate its wording on the c01r08 conflict** (§4.2) | 2/8 paid, 0.16 h. Below an hours threshold, but the wording gate is load-bearing |
| **P4** (queued) | none | **Keep queued**, and re-specify the mechanism as "prices the next runnable job, not the smallest decision-changing run" | 4/8 end ≥1.5 h early with no process; the guard arm meets the secondary observable 3/3 and still fails the primary 2/3 |
| **I** (queued) | none | **Keep queued, no wave slot.** If it remains 0 through the full strict block, it becomes a withdrawal candidate on "the evidence has disappeared" (the ledger #9 disposition) | 0/8 manifestations; 0 overrides, 0 `preflight_fail` in 3/3 guard |

**Guard safety is not established by this window.** The guard's own criteria read clean 3/3 — zero Stop-hook blocks (no `memory/.stop_hook.json` in any bundle), zero false blocking, zero locked-open cards at stop, zero runs lost to session end, zero background scientific work killed at exit, and `setsid nohup … < /dev/null &` used for every long run exactly as the pitfall prescribes. CLI 2.1.219 also **backgrounds** a Bash call that exceeds its tool timeout rather than killing it (g01r01 06:58:30Z, g01s04 14:53:40Z), so the pilot-era "tool timeout kills the process group" mechanism did not reproduce. **All of that is 3 cells of a prescribed 8, and the frozen eight strict cells must be validated and reviewed before any safety claim is made.**

---

## 6. Open questions the next safe wave can resolve

1. **E's aggregation convention: per-cell sum or per-event max?** g01r01 flips between pass and fail on it (sum 0.145 h vs max 0.072 h) and c01r05 sits exactly on the line (sum 0.25 h, max 0.15 h). Under "sum" 1/8 NEW cells pass; under "max" 2/8 do. **The ≥7/8 rule must not be decided by an unfixed convention** — settle it before the strict block is read.
2. **A v2's clock definition: "observable identified" or "same-weight comparison closed"?** The two disagree by 2.3 h on c01r03 (0.09 h vs 2.31 h) and by 0.00 h on c01r07. The 2.2 h gap on c01r03 is "no checkpoint worth A/B-ing existed yet," not indecision.
3. **D's window-02 apportionment** (§1.3) — changes D's expected saving by ≈1.6×.
4. **Is the developer-vs-official read gap concurrency, or something else?** One cell reading the same artifact at `--max-connections 2` and 32 over the same 1319 items settles it. The current data has the gap in **both signs** (g01r01 −0.83 pp, c01r06/c01r08 ≈ +1 pp), which a pure concurrency story must explain.
5. **Does P1's signature exist when the card is required to print the first 20 losses?** Currently unreadable in 2/3 guard cells.
6. **Does H's tree change remove the fake entries without thinning the cards?**
7. **Is the guard's 4.5× lower dead-wait an arm effect or n=3?** The remaining strict cells decide it, and it bears directly on E.
8. **What does the protocol say when a paired statistic comes back null?** c01r05 is the existence proof that "compute the statistic" is not the binding instruction.

## 6b. Reusable meta knowledge (for `skills/exp_protocol_meta/`)

1. **Derive `largest_eval_n` from the reported stderr, not from inspect-log bytes ÷ 44 KB.** `n = p(1−p)/SE² + 1` is exact (inspect uses the n−1 denominator). The heuristic was wrong for c01r05 (353 vs 200), c01r04 (590 vs 600) and c01r06 (1330 vs 1319).
2. **`final_model_written` in the timeline tool is a false positive** whenever the string occurs in a `TaskCreate` subject or card text (c01r05, c01r06, g01r01).
3. **The timeline's category split is systematically wrong when launches bundle a `sleep` or evals run backgrounded.** `train_launch` inflates (real launch overhead is seconds), `sample_eval` deflates, and `waiting_on_runs` absorbs both. Correct by reading, and state the correction in the report.
4. **Measure E from `task/system_monitor.log` (60 s GPU sampling) plus NCCL `W903` shutdown stamps**, not from sleep durations. The coarse method under-reports, which is why window 02 and window 03 disagree about the baseline.
5. **Measure protocol cost as `new → lock` spans with no other work inside.** A naive CLI-time sum reports 660 s of "protocol" for a call that is one lock plus four full-1319 evals; a `--dry-run` bundled into `awm exp_protocol new` inflates it the same way. Card cost also falls ~10× within a session as the scientist learns the one-heredoc idiom.
6. **`n_relocked` is not a churn metric, and it counts cards not events.** 2/3 relocks this window were the protocol working: g01r02 relocked twice in 22 s because the first relock had pinned the *unfixed* script; g01s04 relocked to record that the shipped weights came from the locked version.
7. **`pitfalls_hit` from `collect` (card entries) and a reviewer's loss count (≥threshold losses) are different metrics** and will disagree — g01r01 reads 7 vs 4 while the hours agree exactly at 0.72.
8. **Thinking blocks are empty in every trace in this window** (70/76/70/66 blocks, 0 with content). Every "the scientist reasoned" claim must rest on visible prose turns, Bash `description` fields, code comments, `RESULTS.md`/`README.md`, or the harness footer — and reports must say so.
9. **Self-reported loss totals under-count by 1.5–3×.** c01r04 says *"Two incidents cost ~1h"* against 2.46 h; c01r05 says *"three failed generation passes cost roughly an hour"* against ≈2.49 h and five wasted launches; c01r06 says *"roughly 1.5 hours"* against ≈4.2 h once eval polling is included. Never take the footer's accounting as the measurement.
10. **A third instance of scalar-identical official accuracy from unrelated recipes**: c01r03 and c01r08 both read exactly 1027/1319 from completely different pipelines (SFT+RFT greedy, no RL vs SFT+RFT+GRPO soup, sampling). Retained bundles carry no per-item official log, so item-level identity cannot be verified either way. Same disposition as closed direction #12: record it, do not read it as duplication, and it does not affect either cell's eligibility.

---

## 7. Session attribution

| role | group | cells | session id |
|---|---|---|---|
| reviewer | control-a | c01r04, c01r05, c01r06 | `afa3cfee-8276-4292-96ab-bbe26d65704c` |
| reviewer | control-b | c01r07, c01r08 (+ c01r03 calibration) | `a165f583-9c88-420e-b83a-d2212dde2d35` |
| reviewer | guard | g01r01, g01r02, g01s04 | `9081a22d-3a4d-455e-89ab-d3164d577a76` |
| synthesis | all reports | — | *supplied by the planner after return* |

All four sessions: `claude-opus-5[1m]`, `--effort max`, read-only tool set (`Read,Grep,Glob,Bash`), plan permission mode. Model/effort match the local-analysis contract and `launch.json`. Frozen window: harvest commit `9c7596a`, `new_clean = 8`, `calibration_not_counted = [c01r03]`.

---

**Standing of this document.** Evidence only. It makes no promotion claim, and none is available from n=3 guard. It does not establish guard safety — that requires the frozen eight strict cells to be validated and reviewed. It does not convert any control recipe into a protocol training instruction. Every retention, deferral, withdrawal, release and candidate decision, and every update to the round record and the directions ledger, belongs to the Codex planner.

