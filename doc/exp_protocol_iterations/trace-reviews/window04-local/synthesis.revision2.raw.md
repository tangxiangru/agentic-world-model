I have the new audits and have re-derived the contested numbers. Emitting the revised synthesis.

<!-- SYNTHESIS window=window04-local revision=2 -->

# Window04 independent synthesis — revision 2

**Session identity.** This revision runs as `CLAUDE_CODE_SESSION_ID=a1e293bb-7b8c-4e8a-ae0b-d305c22d47e3`, PID 2690672, parented on the initial synthesis session `ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70` (PID 2593119), whose complete output is preserved unedited in `synthesis.initial.raw.md` / `synthesis.initial.md`. Model `claude-opus-5[1m]`, `--effort max`, plan mode, tools `Read,Grep,Glob,Bash`.

**What I actually did and did not do (revised for precision).** `Write` is not available in this session; `Bash` is, and Bash *can* write — so the accurate statement is about actions, not capability. Every command I ran was a read: `ls`, `cat`, `zcat | grep`, `zcat | sed -n`, `git show`, `git diff`, and Python scripts that read files and print to stdout. **I created no file and modified none.** I performed **read-only git operations** (`git show`, `git diff` against frozen candidate SHAs) — so: **no git mutation**, not "no git operation". No model, evaluator, training run, Slurm query or action, package install, or network access.

**CLI provenance — three separate versions, no silent image change.** The frozen scientist image is **2.1.219** for all 14 cells (**[verified here]** in `cli_version.txt` of the g01r03, c01s04 and p00s02 result directories, `update: skipped`, recorded at each session start). The five initial cell reviewers are recorded at **2.1.259**. This analysis session is **2.1.260** (**[verified here]**, `claude --version`). These are three distinct provenances of three distinct things; the scientist image did not change.

**Evidence labels** are unchanged from revision 1: **[SAID]** scientist text · **[SHOWS]** trace/artifact/log · **[INFER]** my inference with uncertainty · **[COUNTERFACTUAL]** derived from frozen code, never observed · **[verified here]** re-derived by me from the original source · **[reviewer-cited]** carried from a reviewer report or focused audit without independent re-derivation.

---

## 0. Substantive changes from revision 1 [REV2]

Read this list first; everything below incorporates it.

| # | Revision-1 statement | Revision-2 correction | Basis |
|---|---|---|---|
| 1 | c01s04/c01s07 paired counts "±1 approximate" (219/211, 74/60, 61/61 …) | **Materially wrong, not ±1.** Replaced with the exact structural tables. | `paired-counts/` audit + planner byte-identical rerun |
| 2 | "executed max n ≥ 500: 14/14" | **13/14** — c01s06's maximum executed evaluation is n=400 | **[verified here]** from c01s06's Inspect logs |
| 3 | "executed max vs selection n differ in 2 cells" | **3 cells**: c01s06 (400/300), c01s07 (1000/500), p00s02 (500/150) | **[verified here]** |
| 4 | "11 relocks" | Unit error. **9 cards re-locked** (`collect.csv`) and **12 relock events** (`.relocked_from` lengths over 56 lock files) | **[verified here]** |
| 5 | g01r03 exp-08 = "a comparator at n=800 declared against protocol n=150" | **Wrong mechanism.** The card claims a 150-item subset; the failure is a wrong-subset derivation plus an unreproducible recorded value | `g01r03-prefix-audit.md` + **[verified here]** |
| 6 | "2 of 46 cards declare a cross-n comparator" (filename heuristic) | **Withdrawn as a rate.** Replaced with exhaustive artifact-level checks (below) | **[verified here]** |
| 7 | Proposal 2 rested on a scientist-declared `comparator.n` | **Withdrawn.** A self-declaration is not verification and would have cured neither case. Replaced with an artifact-derived design | follow-up §3 + **[verified here]** |
| 8 | D2's guardrail "zero blocking failures on non-saving cards" while knowingly retaining one | **Self-contradictory, and the residual is larger than stated.** Predicate now explicitly deferred | follow-up §4 + **[verified here]** |
| 9 | "+5.84 pt from GRPO alone" (c01s01); c01s03's gap "almost exactly" explained by it | Causal and cross-cell framing removed | planner corrections §4 |
| 10 | c01s02 "2.87 h of idle" | Removed; replaced with the exact wait and its conditional bounds | control-timing audit |
| 11 | g01r07 "8.1 h of decode-mismatched comparisons" | Removed; pre-fix comparisons used a **common** sampled policy | planner corrections §4 |
| 12 | g01r03 and g01r05 listed as "realised inversions" | Neither is an inversion: g01r03's gap **collapsed** (same order), g01r05's **widened** (same order) | **[verified here]** |
| 13 | "format/termination SFT" causally credited with +55–67 pt | The base→post-SFT jump is a bundled package; g01r08's first SFT landed **below base** | **[verified here]** |
| 14 | "No cell has an unconditional positive post-exit lower bound" | Restated: **none was established by this review** | follow-up §5 |
| 15 | Pooled protocol-8 vs control-6 Welch test | Removed. Guard (`189319d6`) and baseline (`08674f2c`) are **different variants**; pooling them is not a variant effect | follow-up §5 |
| 16 | "Early durable write bounds the loss" / dose-response | Wrong: the `--save-steps 150` runs died **at** their first save and produced **no** usable checkpoint. Earlier *detection*, not durable output; 4-cell association, not dose-response | follow-up §6 + **[verified here]** |
| 17 | c01s02's PID+`kill -0` repair called "E2's prescription" | E2 additionally requires retained exit status and current-invocation artifact validation; the repair does not establish E2 compliance | planner corrections §1 |
| 18 | "Fold WMA removal into the next unrelated SKILL revision" | Withdrawn — that is a separate item, and the instructions serve more than this sandbox | follow-up §7 |
| 19 | Decode-field and tool-timeout ideas "confounded by construction" | Wrong framing. Parallel baseline-relative candidates sharing a topic are **not** stacked and not inherently confounded; the real issues are overlapping questions and redundant work | follow-up §7 |
| 20 | Operations snapshot (29 held, PID 2579442) | Updated: **30 held / 22 excluding D4+stale E4**, monitor **PID 2612586** over 22 IDs | `launch.json` **[verified here]** |

---

## 1. Per-cell and per-arm header table, with definitions reconciled

### 1.0 Denominator (unchanged)

Exactly **14 NEW cells**: guard `g01r03–g01r08`; protocol-free strict control `c01s01–c01s04, c01s06, c01s07`; protocol-v3 strict baseline `p00s01, p00s02`. All 14 are validator-complete, eligible, judge-clean, non-quarantined. `p00r02` is historical calibration and enters no denominator. `c01s05`, `p00s03` and the placement-quarantined `c01s08` are outside this window and appear nowhere in this arithmetic.

| cell | arm | job | awm sha / protocol_tree | accuracy | items | stderr |
|---|---|---|---|---:|---:|---:|
| g01r03 | guard | 90649 | `4ae3d87` / `189319d6` | 0.8278999241849886 | 1092/1319 | 0.010397 |
| g01r04 | guard | 90650 | `4ae3d87` / `189319d6` | 0.6906747536012130 | 911/1319 | 0.012732 |
| g01r05 | guard | 90651 | `4ae3d87` / `189319d6` | 0.7611827141774071 | 1004/1319 | 0.011744 |
| g01r06 | guard | 90652 | `4ae3d87` / `189319d6` | 0.7081122062168309 | 934/1319 | 0.012523 |
| g01r07 | guard | 90653 | `4ae3d87` / `189319d6` | 0.7217589082638363 | 952/1319 | 0.012344 |
| g01r08 | guard | 90654 | `4ae3d87` / `189319d6` | 0.7028051554207733 | 927/1319 | 0.012589 |
| c01s01 | control | 90813 | `eaf5091` / *(no protocol tree)* | 0.7892342683851402 | 1041/1319 | 0.011234 |
| c01s02 | control | 90814 | `eaf5091` / — | 0.7816527672479151 | 1031/1319 | 0.011379 |
| c01s03 | control | 90815 | `eaf5091` / — | 0.7202426080363912 | 950/1319 | 0.012364 |
| c01s04 | control | 90816 | `eaf5091` / — | 0.7558756633813495 | 997/1319 | 0.011832 |
| c01s06 | control | 90818 | `eaf5091` / — | 0.7414708112206216 | 978/1319 | 0.012060 |
| c01s07 | control | 90819 | `eaf5091` / — | 0.7202426080363912 | 950/1319 | 0.012364 |
| p00s01 | baseline_v3 | 90823 | `eaf5091` / `08674f2c` | 0.7702805155420773 | 1016/1319 | 0.011587 |
| p00s02 | baseline_v3 | 90824 | `eaf5091` / `08674f2c` | 0.7771038665655799 | 1025/1319 | 0.011464 |

Frozen means (descriptive): guard6 **0.7354056103108415** (sd 0.0514, range 0.1372), control6 **0.7514531210513015** (sd 0.0297, range 0.0690), baseline2 **0.7736921910538286** (n=2). **[verified here]** to the last digit.

**[REV2]** Guard and baseline are **two different protocol variants** (`189319d6` vs `08674f2c`), not one arm. I therefore report them separately and give **no pooled protocol-vs-control test**; a pooled t-statistic over two variants is not a single-variant effect.

### 1.1 Definitional splits (unchanged, with one addition)

1. **First training launch**: (a) first GPU training of any kind vs (b) first card-matched launch after its lock.
2. **Protocol hours**: (a) `awm exp_protocol` tool execution time vs (b) card-authoring wall (`new`→first `lock` + closes + bootstrap), which includes model generation. **[REV2]** These are heterogeneous card/tool intervals; the mean of (b) is a summary of a heterogeneous set, **not an exact common overhead estimate**.
3. **Waiting**: (a) raw tool time classified as waiting vs (b) strict post-exit idle. Never interchangeable.
4. **Largest n**: (a) maximum executed developer evaluation vs (b) n behind the shipping selection. **[REV2]** They differ in **3** cells.
5. **[REV2] Relocks**: `n_relocked` counts **cards locked more than once**; `.relocked_from` list lengths count **relock events**. Do not mix.

### 1.2 Table A — time, launch, protocol cost

`hours_used` from `time_taken.txt`; "left at stop" is the last `timer.sh` reading — both **[verified here]** for all 14.

| cell | hours_used | h→first GPU train (any) | h→first card-matched train | protocol tool-h | card-authoring wall-h | raw waiting-h | left at stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| g01r03 | 8.442 | 0.23 | 0.51 | 0.004 | ≤0.10 | 7.24 | 1:33 (1.550) |
| g01r04 | 7.758 | 0.18 | 0.37 | 0.002 | ≤0.05 | 6.56 | 2:15 (2.250) |
| g01r05 | 7.924 | 0.22 | 0.44 | 0.016 | ≤0.09 | 6.70 | 2:05 (2.083) |
| g01r06 | 7.741 | 0.19 | 0.38 | 0.03 | 0.125 | 5.84 | 2:16 (2.267) |
| g01r07 | 8.374 | 0.19 | 0.29 | 0.01 | 0.217 | 5.21 corr. (6.21 raw) | 1:38 (1.633) |
| g01r08 | 7.455 | 0.19 | 0.27 | 0.00 | 0.40 raw / ~0.13 non-overlap | 4.70 | 2:32 (2.533) |
| c01s01 | 8.759 | 0.11 | n/a (0.50 production) | 0 | 0 | 7.19 | 1:14 (1.233) |
| c01s02 | 8.626 | 0.16 | n/a (0.25) | 0 | 0 | 7.62 | 1:24 (1.400) |
| c01s03 | 8.408 | 0.11 | n/a (0.20) | 0 | 0 | 7.44 | 1:37 (1.617) |
| c01s04 | 8.625 | 0.15 | n/a (0.40) | 0 | 0 | 7.67 | 1:22 (1.367) |
| c01s06 | 7.890 | 0.185 | n/a (0.55) | 0 | 0 | 6.85 | 2:07 (2.117) |
| c01s07 | 8.190 | 0.10 | n/a (0.18) | 0 | 0 | 6.81 | 1:49 (1.817) |
| p00s01 | 7.891 | 0.19 | 0.44 | 0.01 | 0.16 | 6.72 | 2:07 (2.117) |
| p00s02 | 7.805 | 0.16 | 0.25 (0.35 to the launch that survived) | 0.05 | 0.18 | 6.78 | 2:12 (2.200) |

Aggregates **[verified here]**: mean `hours_used` guard 7.949, control 8.416, baseline 7.848. Mean unused: guard 2.053 h, control 1.592 h, baseline 2.158 h. Cells stopping with ≥1.5 h unused and no live process: **guard 6/6, baseline 2/2, control 3/6** (Fisher, protocol-installed 8/8 vs control 3/6, two-sided p ≈ 0.055).

**[REV2] Direct overhead, stated as a range over heterogeneous intervals, not a point estimate.** `awm exp_protocol` tool execution is **0.00–0.05 h/cell**. Card-authoring wall spans, which bundle model generation with tool time and in one cell overlap a running evaluation, are **0.05–0.40 h/cell**. Against ~7.9 h that is roughly **0.6 %–5 %** per cell depending on the definition and the cell. This remains the third window in a row with no evidence of a large direct ceremony cost (ledger #11), but the earlier "1.7–2.1 %" was a false precision over incomparable intervals.

### 1.3 Table B — decode, methods, evaluation

`greedy_shipped` **[verified here]** by reading `final_model/generation_config.json` on each receipt-backed `result_dir`. All 14 resolve to greedy under vLLM, in **six distinct spellings**.

| cell | shipped spelling | same-weight decode A/B? | A v2 clock | rl_used | rft_tried | executed max n | n behind ship decision |
|---|---|---|---:|---|---|---:|---:|
| g01r03 | `do_sample:false, temperature:0.0` | yes, +7.7 @300 | 0.23 h | no | no (script unrun) | 800 | 800 |
| g01r04 | `do_sample:true, temperature:0.0` | yes, +5.8 @500 | 0.37 h | no | yes | 1000 | 1000 |
| g01r05 | `do_sample:false, temperature:0.0` | **no — designed in** | **endpoint absent** | no | yes ×2 | 1319 | 1319 |
| g01r06 | `…false, 0.0, top_k:1, top_p:1.0` | **no — designed in** | **endpoint absent** | no | yes | 1319 | 1319 |
| g01r07 | `do_sample:false, temperature:0.0` | yes, +5.8 @500, p=0.0076 | **6.22 h** | no | yes ×2 | 500 | 500 |
| g01r08 | `do_sample:false, temperature:0.0` | yes, +8.0 @150 | **2.29 h** | no | yes | 500 | 500 |
| c01s01 | `…false, 0.0, top_p:1.0, top_k:-1` | **no on its own model** | **endpoint absent** | **yes** | yes | 1319 | 1319 |
| c01s02 | `do_sample:true, temperature:1e-06, top_k:1, top_p:1.0` | **no on its own model** | **endpoint absent** | **yes** | yes | 500 | 500 |
| c01s03 | `do_sample:false, temperature:0.0` | yes, +5.7 @300 | 0.06 h | no | yes | 1319 | 1319 |
| c01s04 | `do_sample:false, temperature:0.0` | yes, **+13.5 @200** | ≈0.03 h | **yes** | no (built, unrun) | 1319 | 1319 |
| c01s06 | `…false, 0.0, top_p:1.0, top_k:0` | yes, **+15.4 @150** (pilot) | ≈0.08 h | no | yes | **400** | **300** |
| c01s07 | `do_sample:true, temperature:0.0, top_k:64, top_p:0.95` | yes, +9.0 @200 | ≈0.03 h | **yes** | yes | 1000 | **500** |
| p00s01 | `do_sample:false, temperature:0.0` | yes, **≈0.0…+0.7 @150** | 0.16 h | no | yes (vote form) | 500 | 500 |
| p00s02 | `…false, 0.0, top_k:0, top_p:1.0` | yes, **+11.3 @150** | 0.14 h | no | no | 500 | **150** (+probe-300) |

Reconciled aggregates, **[REV2]** corrected:

- **Greedy shipped 14/14.** The old binary A metric is saturated in both arms.
- **Measured same-weight decode A/B: 10/14.** Four cells never produce the frozen endpoint.
- **A v2's revised metric is not saturated in the protocol arm**: 4/8 measured within ≤0.5 h; 2/8 clear failures (g01r07 6.22 h, g01r08 2.29 h); 2/8 endpoint absent. Control arm: 4/6 measured (0.03–0.08 h), 2/6 endpoint absent — **not** "saturated in the control arm".
- **[REV2] Executed max n ≥ 500: 13/14** (c01s06 tops out at 400 — **[verified here]** from its seven Inspect logs: 150, 150, 150, 300, 300, 300, 400). **Selection n ≥ 500: 12/14.** **Executed max ≠ selection n in 3 cells.**
- **On-policy RL: protocol-installed 0/8, control 4/6** (Fisher two-sided p = 0.0150). **RFT attempted: protocol 6/8, control 5/6** (p = 1.00).

### 1.4 Table C — cost bookkeeping in incompatible units

`pitfalls_cost_h` is a sum of card entries and is **structurally 0.0 for all six controls** — the protocol's own KPI cannot be measured in its comparator arm.

| cell | raw `pitfalls_cost_h` (entries) | event-deduplicated card cost | reviewer trace-bounded ≥0.1 h losses | post-exit idle (all conditional) | overrides / cards re-locked / relock events / cards |
|---|---:|---:|---|---|---|
| g01r03 | 0.85 (6) | 0.85 + one unrecorded ~0.09 h soup abort (booked 0.0) | ≈0.24 h mechanical + 0.50 h data-yield search | ≥0.85 h over 10 events, largest 0.18 h | 2 / 2 / 2 / 8 |
| g01r04 | 2.28 (7) | 2.28 (2.15 on one card) | 1.6 + 0.4 + 0.1 + 0.08 + 0.05 | ≥0.75 h (event-sum ≈0.96), largest 0.36 h | 4 / 0 / 0 / 4 |
| g01r05 | 1.10 (4) | 1.10 | 0.85 + 0.15 + 0.10 | ≥0.46 h over 3 events, largest 0.30 h | 0 / 1 / 1 / 7 |
| g01r06 | 2.30 (7) | **1.25** (exp-04 carries exp-03's entries verbatim) | ≈1.0 h | ≤~0.02 h/event; not material | 0 / 2 / **4** / 7 |
| g01r07 | 0.10 (2) | 0.10 — the cell's largest real cost (≈0.8 h re-work) is booked 0.0 | 0.028 h OOM | ≈0.05 h total | 0 / 1 / 1 / 8 |
| g01r08 | 1.00 (6) | ≈1.00 | 0.50 + 0.202 + 0.13 + 0.10 | ≥0.202 h single event; one 0–6 min unresolved | 0 / 1 / 1 / 8 |
| c01s01 | 0.0 (n/a) | n/a | 0.17 + 0.17 + 0.13 | audited ceiling **0–0.144 h** on one selected transition | — / — / — / 0 |
| c01s02 | 0.0 (n/a) | n/a | 0.50 + 0.30 + 2.07 + 0.12 (endpoint-span statistics) | **[REV2]** exact uninterrupted wait **1.99917 h**; two isolated waits **2.03028 h**; strict post-exit component **[0, 2.03028 h]**; coarse ceiling for the three selected gaps 2.88389 h *before* removing productive overlap. **The 2.87 h figure is withdrawn.** | — / — / — / 0 |
| c01s03 | 0.0 (n/a) | n/a | 0.70 h recipe; ~0 mechanical | audited: five long-producer transitions sum ceiling **0.09889 h** | — / — / — / 0 |
| c01s04 | 0.0 (n/a) | n/a | 0.175 + 0.113 (+0.65 selection evals) | ≈0.15 h total, largest 0.113 h | — / — / — / 0 |
| c01s06 | 0.0 (n/a) | n/a | 2.93 + 0.86 + 0.10 | ≈0.17 h total, largest ≈0.09 h | — / — / — / 0 |
| c01s07 | 0.0 (n/a) | n/a | 0.10 (+1.01 exploration) | ≈0.15 h total, largest ≈0.10 h | — / — / — / 0 |
| p00s01 | 0.96 (5) | failed compute 0.25 · idle 0.55 · 0.12–0.16 · 0.07–0.13 | as listed | 0.55 h (crash) + ≤0.12 h | 0 / 0 / 0 / 7 |
| p00s02 | 0.95 (7) | failed compute 0.49 · idle 0.53 · 0.095 · 0.083 · ~0.05 · 0.033 | as listed | 0.53 h (crash) + ≈0.13–0.15 h | 0 / 2 / **3** / 7 |

**[REV2] Relock totals, verified exhaustively over all 56 lock files:** **9 cards re-locked** (matching `collect.csv`'s `n_relocked` sum) and **12 relock events** — g01r03 2/2, g01r04 0/0, g01r05 1/1, g01r06 **2 cards / 4 events**, g01r07 1/1, g01r08 1/1, p00s01 0/0, p00s02 **2 cards / 3 events**.

**[REV2] Post-exit idle caveat, restated precisely.** Every ≥0.1 h post-exit figure above except the three audited control transitions rests on a GPU-release, artifact-mtime or `train_runtime` proxy for producer exit. **This review did not establish an unconditional positive lower bound on post-exit idle for any cell in the window.** That is a statement about what was established, not a proof that no such bound could be established from other evidence. Unconditional quantities available are: exact tool-wait durations (c01s02 1.99917 h; g01r08 20:25:56 → 20:38:04 = 0.202 h between a logged producer error and the first observation) and conservative ceilings (c01s01 ≤0.144 h; c01s03 ≤0.09889 h). Failed compute and post-exit idle stay in separate columns and are never summed into one candidate's savings.

Card bookkeeping **[verified here]**: 56 cards, 56 closed, 0 `n_locked_open`, `fields_filled` 1.0 in 8/8, 6 overrides (all in g01r03 and g01r04). Families: sft 17, other 12, decode-config 11, rft 10, merge 6.

### 1.5 [REV2] Corrected exact paired tables (replacing the regex tables)

From the structural audit of all 34 receipt-backed developer Inspect logs, reproduced byte-for-byte by the planner. Complete unique `[id, epoch]` coverage; **no missing samples**; 0 original-input, 0 target and 0 rendered-prompt mismatches on common keys.

**c01s04, all n=1319** (correct totals: b70 993, b160 986, soup1 981, soup2 979, b105 978, b70_rp 978):

| A vs B | Both | A-only / B-only | Neither | Net | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| b70 vs b160 | 935 | **58 / 51** | 275 | +7 | 0.5657 |
| b70 vs b105 | 934 | **59 / 44** | 282 | +15 | 0.1674 |
| b70 vs soup1 | 936 | **57 / 45** | 281 | +12 | 0.2760 |
| b70 vs b70_rp | 909 | **84 / 69** | 257 | +15 | 0.2576 |
| b105 vs b70_rp | 897 | **81 / 81** | 260 | 0 | 1.0000 |

The winner's margin over the runner-up is **7 items with 109 discordants**, not "8 items with 430 discordants".

**c01s07, selection pairs n=500** (correct totals: g2_200 372, rft_v1 366, g3_150 358, g3_100 358):

| A vs B | Both | A-only / B-only | Neither | Net | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| g2_200 vs rft_v1 | 322 | 50 / 44 | 84 | +6 | 0.6063 |
| g2_200 vs g3_150 | 315 | **57 / 43** | 85 | +14 | 0.1933 |
| g3_150 vs g3_100 | 317 | **41 / 41** | 101 | 0 | 1.0000 |
| g2_200@500 vs final@1000 shared-500 | 356 | 16 / 6 | 122 | +10 | 0.0525 |

c01s07's **15-log** total is correct; the corrected breakdown is **150×2 + 200×8 + 500×4 + 1000×1**, the omitted n=150 being the 01:46:15 default final-model artifact check (110/150 = 0.7333), distinct from the n=150 base evaluation (3/150).

**[REV2] Three interpretive corrections that follow:**
- The c01s07 g2_200@500 vs final@1000-shared-500 contrast **changes both the token cap (1024→2048) and the total evaluated population**. It is *"a trace-linked same-artifact, changed-serving/evaluation-size contrast that flipped 22 shared items (16 losses, 6 gains)"* — **not** repeat noise, a noise floor, or a repeatability estimate.
- b105 vs b70_rp and g3_150 vs g3_100 are **different artifacts** with equal scalar accuracy. Equal accuracy with disjoint error sets is a real and important observation; it is **not** a repeat-variance experiment.
- p > 0.05 on these pairs means *this exact two-sided test did not reject equal marginal success probabilities*. It does **not** establish equivalence, equal expected quality, or that a selection read was wasted.

---

## 2. Ranked explanations of the observed score differences

### R0 — no arm effect is established, and no equivalence is established either

**[verified here]** guard6 − control6 = **−0.01605**, Welch t = −0.662, df ≈ 8.0, SE 0.0242. baseline2 has n=2. **[REV2]** A null scalar test does not prove there is no arm effect, and it certainly does not prove equivalence; with n=6 per arm and per-cell stderr ≈0.012, this window has very little power. Guard and baseline are different variants and are not pooled.

g01r03's 0.8278999241849886 is the maximum single clean official score in the line and nothing more. **No held-out AIME result exists for any variant, so promotion is impossible for every candidate below.** All 14 cells clear the fixed 24-cell J/K floor 0.6737212534748547 (lowest: g01r04, 0.6907).

### R1 — the first supervised stage sets the floor, as a bundled package [REV2]

**[verified here]** Base → first-SFT deltas span **+41 to +69 pp** across the 13 cells whose first SFT improved on base, and **−1.3 pp in g01r08**, whose exp-02 scored 0.060 against a 0.073 base.

The first SFT card changes many things simultaneously — teacher data source and size, target rewriting to an `ANSWER:` line, terminator handling, grader-template rendering, a few-shot mixture share, and full-fine-tuning hyperparameters; in some cells the decode config too. The cells' own diagnostics show format and termination moving hard (p00s01: format compliance 0.24 → 0.96, cap hits 67/150 → 6/150; p00s02: truncated share 0.51 → 0.000/0.007) **[reviewer-cited]**, and [SAID, p00s01 L9378, 02:54:58Z] *"Almost all of the gain is format, not math."* **[REV2] That is the scientist's attribution, not an isolation.** g01r08 is the in-window counterexample: a format-and-termination SFT that left the model below base, recovered only by exp-03's prompt-distribution change (0.060 → 0.6267 @150). R1 explains the floor; it does not explain the ranking, and "format alone" is not supported.

### R2 — decode configuration: largest measured same-weight effect, not a between-arm explanation

Measured same-weight gains: +15.4 (c01s06), +13.5 (c01s04), +11.3 (p00s02), +9.0 (c01s07), +8.0 (g01r08), +7.7 (g01r03), +5.8 (g01r04), +5.8 (g01r07, McNemar p=0.0076), +5.7 (c01s03), ≈0.0…+0.7 (p00s01).

Since 14/14 shipped greedy, decode cannot explain the arm means. It explains two things:

1. **Magnitude is a property of the parent.** p00s01's sampled parent was already at 0.7267 with 6/150 cap hits and gained ≈0; cells whose sampled parent sat at 0.61–0.63 gained 11–15 pp. The byte-identical p00r02/p00s01/p00s02 triple spans +14.0 / ≈+0.7 / +11.3 — same guidance, three magnitudes.
2. **[REV2] Timing changed one cell's shipped artifact, without a decode-mismatch claim.** g01r07's comparisons before 02:38 were all made **under a common inherited sampling policy** — both sides decoded the same way, so they were not mutually mismatched. What happened is that the decode choice was made **6.22 h** after the first post-SFT eval, and the later greedy re-measurement **changed the ranking**: exp-04 0.736 > soup 0.722 > exp-05 0.708 under greedy, against 0.644 < 0.656 < 0.664 sampled. `final_model` was re-staged from exp-04. Traceable re-work ≈0.8 h, booked at `cost_h: 0.0`. The discovery itself was accidental — [SHOWS, L8469, 02:38:00Z] the same weights read 0.58 @150 against 0.664 @500, and grepping the vLLM log surfaced the override line at L8491, 02:38:16Z **[reviewer-cited]**.

Counterweight: g01r08 deferred decode with a correct reason and then **kept decode matched across every later comparison** — [SHOWS, L7766, 00:43:25Z] it copied the greedy config onto `ckpts/exp-05/final` before evaluating it **[reviewer-cited]** — and its n=150 and n=500 orderings agree.

### R3 — post-SFT method stack: the largest between-cell differentiator, and not a protocol variable

**[verified here]** On-policy RL: **0/8 protocol-installed, 4/6 controls**, Fisher two-sided p = 0.0150. This extends ledger #8 (v3 0/14; clean controls 8/15) to 0/22 and 12/21.

**[REV2] The within-cell RL observations, stated without causal or transfer claims.**
- **[verified here]** Inside c01s01, two checkpoints on the same lineage, evaluated under the same developer protocol at n=1319, differ by **77 items** (`rft1_full` 978/1319 = 0.7414708112206216 → `g350_full` 1055/1319 = 0.7998483699772555, +5.84 pp). The intervening change is a 350-step full-parameter GRPO run *together with* its data seed and prompt filter. This is a single within-cell before/after: no repeat, no paired test by the scientist, no isolation of the GRPO stage.
- c01s04's SFT read 0.640 @200 and its shipped GRPO stage-2 checkpoint read 993/1319 developer / 997/1319 official **[verified here]** — again a within-cell progression, not an isolated RL effect.
- **This does not transfer to c01s03.** c01s03's 6.9-pt gap to c01s01 involves different data, different checkpoints and a different pipeline; the planner correction is explicit that c01s01's within-cell improvement does not causally transfer, and that c01s03's 0.687@n300 vs 0.7081@n1319 is not a matched gain.
- **RL is neither necessary nor sufficient here.** The window's highest cell, g01r03 (0.8279), executed no RL and no RFT — [SHOWS] `scripts/sample_rft.py` written 19:06:23Z, never invoked **[reviewer-cited]**. c01s07 ran three LoRA GRPO rounds over ~4.04 h and tied for the lowest control score.
- Adoption rate is a recipe difference on a non-randomised axis. Ledger #8 forbids inferring protocol causality from it. **I propose no RL, RFT, soup, dataset or prefix rule.**

**Mixed soup outcomes (ledger #18 confirmed, not advanced):** attempted in ≥8 cells, **adopted in 2** (g01r03 +2.1 pt @800, the shipped model; g01r06 +0.9 pt @1319), **rejected in 6**. g01r03's soup is verified as a real average: on the 91 items its parents split, the soup takes 65 **[reviewer-cited]**.

**Mixed prefix outcomes (ledger #28 confirmed, not advanced):** decisive in g01r08 (0.060 → 0.6267), *measured as unnecessary* in g01r07 (probe 0.730 with prefix vs 0.688 without, prefix share kept at 6 %) **[reviewer-cited]**.

### R4 — selection quality: four genuine inversions, two mislabelled cases removed [REV2]

**Genuine ordering inversions between a smaller and a larger n, all [verified here] against retained artifacts:**

| cell | small n | large n | inverted? |
|---|---|---|---|
| c01s04 | n=400: soup1 300/400 (.750) > b70 297/400 (.7425) | n=1319: b70 993 (.75284) > soup1 981 (.74375) | **yes** — and the full-test pass changed the shipped artifact |
| g01r06 | n=200: exp-04 .710 > exp-03 .690 | n=1319: exp-03 .70584 > exp-04 .70281 | **yes** |
| p00s01 | dev-150: exp-06 .78667 > exp-04 .77333 > exp-05 .75333 | dev-500: exp-05 .780 > exp-06 .768 > exp-04 .760 | **yes** — complete three-way reversal; protected ≈1.2 pt |
| g01r07 | n=150: exp-04 .6733 > exp-05 .6467 | n=500: exp-04 .644 < exp-05 .656 | **yes** (both sides sampled) |

**Removed from the inversion list [REV2]:**
- **g01r03** — n=300: exp-04 .830 > exp-05 .800; n=800: exp-04 .8175 > exp-05 .81625. **Same ordering; the gap collapsed from 3.0 pt to 0.125 pt.** The card's own McNemar (46/45, z ≈ 0.1) called it a tie. A shrinking gap is not an inversion.
- **g01r05** — n=200: exp-04 .720 > exp-02 .710; n=1319: exp-04 .7536 > exp-02 .7111. **Same ordering; the gap widened** to +4.25 pt with paired z = 3.54. [SAID] *"Had I trusted n=200 I'd have shipped a materially worse model"* is the scientist's counterfactual about their own inclination to treat a 1.0-pt/±3.2-SE reading as noise — it does not describe an ordering flip.

**Counterexamples that stop this becoming a rule about n:** p00s02 selected at n=150 + probe-300 and scored 0.7771, second-highest of the eight protocol-installed cells; c01s06 selected at n=300 on a **one-item** margin (**[verified here]** sft2 233/300 vs soup 232/300) and scored 0.7415. And per §1.5, c01s04's full-test winner leads by 7 items with 109 discordants and p = 0.566 — the pass changed the artifact without establishing that the new artifact is better.

**Same-artifact repeat evidence, with each caveat [REV2]:** g01r06's exp-05/final read 0.71494 while the byte-identity-checked (one of two shards md5-matched) `final_model` read 0.70811 and 0.70887 — 943/934/935 of 1319 **[verified here]** from its retained scalar files; c01s02's two default-flag reads of weights with the same mtime gave 119/150 and 120/150; g01r08's shipped path read 0.706@500 against 0.714@500 via the symlinked directory. None is a designed repeat experiment, and official per-item identity remains unresolved wherever the official log was not retained (ledger #24/#29). Note that g01r06's `final_model_full.json` (0.7081122062168309) equals its official score digit-for-digit; equal scalars, per-item identity unverifiable.

### R5 — mechanical waste: large and well-evidenced, not shown to move score

The most reproduced mechanism is the **greedy-parent save-validation failure**. **[verified here]** by grepping `GenerationConfig is invalid` / `Fix these issues to save the configuration` across all 14 traces: present in **8 of 14 cells** — guard g01r03, g01r05, g01r06; control c01s01, c01s02, c01s06; baseline p00s01, p00s02 — on Trainer checkpoint saves, `soup.py` weight-averaging saves and a LoRA `merge_lora.py`. **It hits both arms.**

| cell | event | failed compute | conditional post-exit idle |
|---|---|---:|---:|
| g01r03 | exp-07 soup abort | ~0.09 h (booked 0.0) | — |
| g01r05 | exp-03 Trainer death at step 620/1238 | 0.85 h (card) | 0.10 h |
| g01r05 | exp-07 soup relock | ~0.01 h | — |
| g01r06 | exp-03 `save_model` after 48 min | 0.849 h | ~0 (tail-coupled) |
| g01r06 | `soup.py` | 0.05 h | — |
| c01s01 | `soup.py` refused `top_k:-1` | 0.02 h | — |
| c01s02 | `merge_lora.py` ×2 | ~0.02 h | — |
| c01s06 | round-2 SFT final save | 0.86 h | ~0.02 h |
| p00s01 | exp-06 Trainer at step 150/306 | 0.25 h | 0.55 h |
| p00s02 | exp-05 Trainer at step 150/313 | 0.49 h | 0.53 h |

**[REV2] Correction on the "save early" reading.** p00s01's exp-06 and p00s02's exp-05 both died **at their first checkpoint save** (steps 150/306 and 150/313). Neither produced a usable durable checkpoint; what `--save-steps 150` bought was **earlier detection of the save-path failure**, not durable output. The cross-cell contrast (0.25/0.49 h with an early first save attempt vs 0.849/0.86 h with save-at-end) is a 4-cell association confounded by different run lengths, save cadences and detection latencies. **It is not a dose-response proof.**

Other large mechanical losses: c01s06's **2.93 h** rejection-sampling pass whose samples were not used in the shipped pipeline (missing `stop`/`stop_token_ids` — B v2's mechanism, in a control); g01r04's **1.6 h** of completed generation destroyed by an `OverflowError` in its post-generation grading loop; c01s02's long post-OOM waits (see §1.4 for the exact and conditional figures); g01r08's **0.50 h** tool-timeout kill; recurring Gemma-3 logits OOM and 64 MB-overlay install failures in ≥6 cells (P2/#21); missing processor configs in g01r03, g01r08 and c01s04 (P3/#22).

**[REV2] Failed and rejected work was not automatically useless.** g01r03's *rejected* exp-05 checkpoint became half the shipped soup; p00s02's net-negative exp-04 produced the probe diagnostic that designed exp-05's +14.0-pt repair; c01s07's maj@6-vs-pass@1 measurement on already-dumped samples closed a direction for seconds of CPU, and its negative GRPO-3 closed another; c01s03's discarded polish pass redirected its last 1.5 h; c01s06's unused first sampling pass produced the stop-token diagnosis. The compute that produced neither an artifact nor a decision is a narrower set — chiefly the save-validation crashes and the tool-timeout kill.

**Why waste does not explain the scores.** c01s06 lost ~3.79 h and scored 0.7415, above the guard mean; g01r03 lost ~0.5 h and scored highest; [SAID, g01r06] the 0.85 h *"did not obviously cost score here (2.27 h went unused)"*. **[INFER]** Saved compute does not imply saved score.

### R6 — the low guard cells, and the budget correlation

g01r04 (0.6907, lowest NEW guard): one large early method bet; its 1.6 h crash cut the sampling pool from 28k+7k to 9k+7k prompts and it reached its only post-SFT training launch with 2.8 h left. Its fitted-parent P1 exposure is **not** established by flat loss — its mixed RFT corpus trains from base, not its sampling parent. g01r08 (0.7028): ~0.9 h mechanical loss and 2.53 h unused. g01r06 (0.7081): 0.849 h save crash and 2.27 h unused.

**[REV2] Do not equate free hours with score.** **[verified here]** Pearson r(hours_used, accuracy) = 0.519 over all 14 (t = 2.10, df 12, two-sided p ≈ 0.06); 0.583 within protocol-installed, 0.676 within controls. Marginal, observational, n=14, and confounded in an obvious direction. g01r03 is the counterexample: mid-range hours, highest score.

---

## 3. What the protocol cost and what it helped

### 3.1 Direct overhead
As §1.2: tool time 0.00–0.05 h/cell; card-authoring wall 0.05–0.40 h/cell across heterogeneous intervals. Bootstrap first action ≈8–11 s in every protocol cell. **[REV2]** Reported as a range, not an exact common estimate.

### 3.2 Time to first launch
Card-matched first training 0.25–0.51 h (protocol) vs first production run 0.18–0.55 h (controls); under the any-GPU-training definition, 0.16–0.23 h vs 0.10–0.19 h. The launch-latency difference is small and largely disappears once uncarded smokes are counted on both sides — which is why lock scope is a compliance question, not a speed question.

### 3.3 Where the plan shape helped
- **The comparator rule forced the decisive evaluation in ≥5 cells; in 3 it changed or protected the shipped artifact** (g01r03 exp-06's n=800 head-to-head; g01r05 exp-05, [SAID] *"The protocol that produced both numbers cannot say which checkpoint to ship"*; p00s01 exp-07's n=500 tie-break, which reversed the dev-150 three-way order and protected ≈1.2 pt).
- **Rule 6 produced the window's highest-scoring artifact**: g01r03's exp-05 was `contradicted`/`reject`, and its retained checkpoint became half the shipped soup 3.5 h later.
- **`falsified_if` closed directions** in ≥5 cells, including p00s02 exp-04, whose two-part predicate turned a −10.7-pt result into the diagnosis that designed exp-05.
- **"One card, one intervention"** prevented explicit confounds in g01r03, g01r06 and g01r07.
- **Bookkeeping**: 56/56 cards closed, 0 open, `fields_filled` 1.0 in 8/8, no work lost at session end in 14/14. **[verified here]** `memory/.stop_hook.json` is absent in all six guard result dirs — 0 hook blocks. An untriggered hook does not prove the block works (ledger #1).

### 3.4 A successful main-launch ratio does not prove all commands were locked
`lock_before_launch` is 2/2–4/4 everywhere. **[verified here]** against each cell's `exp-02` `created_at`: **8/8 protocol cells ran real GPU training before the covering card existed** (gaps 2.0–14.9 min), all recorded afterwards in `situation.smoke_runs` — a retrospective record, not a pre-launch lock. Two further findings outside the matched denominator: g01r08's exp-06 evaluation-only card launched its first declared n=500 evaluation at **00:51:49Z** against `exp-06.lock.json` `locked_at 01:06:55Z` (**15.1 min** later), and g01r08's exp-03 was re-locked 3.6 min *after* the launch that produced the shipped weights. This is frozen J's mechanism at 8/8 fresh exposure.

**[verified here]** g01r08's pre-lock evaluation was a direct response to a preflight failure (L8092, 00:51:41Z): `FAIL comparator_same_protocol — .../exp-06_exp05_dev500.json does not exist` / `not locked: preflight failed`, followed eight seconds later by the evaluation launch. That is candidate K's in-card future-comparator dependency with a third response type (g01r03 used a recorded override). **Correction retained:** g01r06's `comparator_same_protocol` FAIL at 01:44:17Z is a filename typo (`exp03_full.json` vs the existing `exp-03_full.json`), fixed by `sed` in 7 s — so the K mechanism is **2/8 cells**, not 3.

### 3.5 Comparator honesty — exhaustive artifact-level findings [REV2]

Replacing revision 1's filename heuristic, I checked **every** comparator-bearing card against its referenced artifact.

**[verified here], all 8 protocol cells, 56 cards, 46 with a comparator path:**

1. **46/46 referenced artifacts are present in the bundle, and 46/46 record no `n`.** `evaluate.py --json-output-file` writes only `{accuracy, stderr}`, so the frozen check's key search (`n`/`limit`/`num_samples`/`samples`) can never succeed in this task. Consequence: **18 printed `PASS … exists; it does not record n` events across 8/8 cells** (g01r03 L4988/L6859; g01r04 L4359/L7123; g01r05 L4901/L5057/L6524; g01r06 L4062; g01r07 L4798/L6215; g01r08 L4152/L4262/L6212; p00s01 L4218/L6804; p00s02 L4436/L6137/L7171). The true count of unverified passes is at least this, since several scientists filtered the lock output.
2. **45 cards declare a comparator value; 44 reproduce their referenced artifact's accuracy to 5e-5. Exactly one does not: g01r03 exp-08**, which records 0.84 against a file holding **0.83875**. One card (g01r03 exp-06) declares `value: null` — the in-card future comparator.
3. **11 of 46 cards reference a sibling card whose own declared `n` differs.** **Nine of these are legitimate re-measurements** — the scientist re-ran the referenced checkpoint at the new n and pointed at the new artifact (e.g. g01r04 exp-03 → `exp-02_final_dev500.json` = 307/500; g01r05 exp-06 → `exp-04_n1319.json`; g01r08 exp-06/07/08 → n=500 files). The frozen check cannot distinguish these nine from the two that are defective.
4. **The two defective cards need two different checks.** **[verified here]** using the sample-SE convention `n ≈ p(1−p)/SE² + 1` sanctioned by `metrics.md` for exactly these GSM8K binary-accuracy reports: the recovered n is an exact integer for **46/46** artifacts and equals the card's `evaluation.protocol.n` for **44/46**. The two exceptions are precisely:
   - **p00s02 exp-07** — card `protocol.n: 500`, artifact recovers **n = 150** (the exp-05 measurement, 0.7733333333333333). Its declared value *does* reproduce the artifact, so only an n-check catches it.
   - **g01r03 exp-08** — card `protocol.n: 150`, artifact recovers **n = 800**. Its declared value does *not* reproduce the artifact, so a value-check also catches it.

**[REV2] g01r03 exp-08 is a subset-identity error, not a "comparator at the wrong n".** Per the prefix audit and **[verified here]**: the card's *intent* is legitimate — compare the packaged `final_model` at the grader's default n=150 against the same 150 items of the soup's 800-item run. What failed is the derivation. At **02:45:11Z, L9026**, the trace runs `s=d["samples"][:150]` — slicing the stored sample array rather than the declared dataset order. **122 of those 150 IDs lie outside the actual dataset-first-150.** That slice gives 126/150 (= 0.84, the recorded value). Joining on `eval.dataset.sample_ids[:150]` gives **128/150 old vs 127/150 new**, 2 old-only and 1 new-only correct, 0 input/target mismatches, the new run's declared 150 IDs equal to the old dataset prefix in order and set. So the true matched-subset contrast is **−1/150, not the card's +1/150** — and even that is not a demonstrated regression, because n, `max_connections` (32→2) and GPU-memory fraction (0.8→0.3) all change. The official 1092/1319 is unchanged, validator-complete, eligible and judge-clean.

**[verified here]** K's frozen change (`58a6992:awm/exp_protocol/preflight.py:265–275`) intercepts `comparator_same_protocol` only when `evaluation.comparator.defer_validation is True`; the ordinary path is byte-identical to the baseline. This defect is uncovered by K and **must not be folded into K**.

### 3.6 Raw-render false alarms, and the metrics blind spot
- **g01r05's `stop_token_consistent` failure was a raw-field/schema mismatch, not a missing rendered EOT.** The original trainer already appended and supervised `<end_of_turn>`, confirmed by a pre-failure smoke; the 49-second repair changed producer and consumer symmetrically. **Rejected:** "true positive", "rescued a 2.3 h run", "+65 pts", "reverses ledger #9". The cell's own diagnostic counts "not `max_tokens`", not EOT compliance.
- **Zero overrides ≠ zero nuisance failures.** g01r05 has `n_overrides = 0` and `preflight_fail = 0` and still paid a data rewrite.
- **[verified here] `collect` records 6 of ~33 friction events.** The 8 protocol traces contain **25 `check` ERROR lines** and **8 `not locked: preflight failed` events**; `collect` reports `preflight_fail` 6 and `n_overrides` 6, all from two cells. **Six of eight cells read `0/0`** while every one hit a blocking `check` error. `setup.data: required` on a non-training card fires in 6/8 cells and `n_examples: must be a positive integer` in 4/8 — together **8/8 cells** hit H's mechanism.
  - Correction retained: `problem.watch_set: needs path and n` (4/8 cells) is **not** a forced fabrication — `2f64581:awm/exp_protocol/schema.py:258–260` fires only on a partially-written `watch_set`. Template-placeholder friction, seconds, observation only.
- **`pitfalls_cost_h` over- and under-counts**: over in g01r06 (2.30 raw vs 1.25 dedup); under in g01r03 (0.09 h soup abort booked 0.0) and g01r07 (≈0.8 h re-work booked 0.0); and structurally 0.0 for all six controls.

### 3.7 Card policy boundaries
g01r03 breached rule 7 in more than one card (verbatim test question and gold `20` in `exp-02.yaml`, plus test-item descriptions with golds in exp-03/04/05). g01r05 first wrote 189 `{id, question, gold}` rows (T05 L4674–4693, 18:55:50Z), then redacted to `{id}` only citing rule 7 (L4702–4716, 18:56:00–01Z); its cards still carry test-derived IDs and item-specific `model_output` detail, and the shipped rule has **no ID-only exception**. **No training contamination is established** in the inspected builder/trainer lineage, and the harness permits `evaluate.py` feedback. Both automatic score invalidation and the stronger "cards never affect score" claim are rejected. Official scores, validator and judge status are unchanged.

### 3.8 Automatic status is not compliance
All 14 are validator-complete, judge-clean, unflagged. That establishes a scored, loadable, non-anomalous deliverable — and nothing about lock-before-launch coverage (8/8 violated it outside the matched denominator), rule 7, or comparator validity.

---

## 4. Proposals — three, each one item, each traced to ≥2 cells

**Common score guardrail**, declared before any of these runs and **not** updated after: the **fixed 24-cell reference pool and floor already frozen for J and K** — mean 0.7037212534748547, floor **0.6737212534748547**, membership exactly as in the J spec (v3 p00r01–07 and p00r09–15; old guard g01r01/02; strict guard g01s01–08). I read the older specs rather than assuming: **A/B/C/D/E/H use a different pool** ("baseline pool (v3 core 16 + guard) mean − 0.03"). No pool is recomputed with this window's cells. Every proposal also requires **exposure in ≥2 of its 4 cells** to be judged at all, a **second independent 4-cell block** if it wins, and **held-out confirmation** before promotion — unobtainable today for every candidate.

---

### Proposal 1 — **D2: keep the objective and the readout; the predicate is explicitly deferred** [REV2 — substantially revised]

**Surface (one item):** one preflight check with a test, replacing `parent_generation_config_valid`. **No implementation is proposed in this pass**, per the follow-up.

**Why frozen D v1 cannot ship.** `8332917:awm/exp_protocol/preflight.py:311–334` reads only `setup.parent_checkpoint.path` and applies to **every card** — no family, argv, framework or save-operation condition. The d-scope CPU replay confirms two false blocks on trace-verified evaluator-only cards (g01r03 exp-08, p00s02 exp-07), one true Trainer-save hit (p00s01 exp-06), one true non-training `save_pretrained` hit (g01r03 exp-07, family `merge`), and blindness to verified in-code repairs.

**[REV2] The false-block population has (at least) two classes, and the larger one is not scope.**

- *Class A — non-saving cards with a local parent.* **[verified here]** 16 of the 56 cards declare no model-saving command yet name a local parent checkpoint; **8/8 cells have at least one**. Two are CPU-replay-confirmed; for the other 14 the parent's config at lock time is not retained, so their outcome is **unknown**.
- *Class B — saving cards whose code already repairs the config before saving.* **[verified here]** g01r05's `train_sft.py:188–195`, g01r06's `:116–122` and p00s02's `:128–138` assign a valid `model.generation_config` before Trainer construction, and g01r03's `soup.py` was repaired in-trace. Counting only *post-repair* saving cards that name a greedy local parent, this is on the order of **7–8 lock events across 4 cells** — i.e. **at least as large as Class A**, and untouched by any family or argv gate.

**[REV2] The family gate I proposed in revision 1 is withdrawn as a settled design.** It fixes Class A only; it leaves Class B entirely; and it knowingly retains a residual — **[verified here]** g01r08's exp-07 is `family: merge` whose declared argv is `evaluate.py`, so it would still be blocked — while my stated guardrail demanded zero such blocks. That was self-contradictory. Two further objections stand: family and argv are *claims about intended future work*, not verified save behaviour; and the absence of nonstandard saves in these 56 historical cards (**[verified here]**: only `train_sft.py` and `soup.py` call `save_pretrained`/`save_model`) does **not** prove zero future false negatives.

**What is proposable now.**

- *Objective (unchanged from D):* a run must not lose its compute to `GenerationConfig.validate` at its first save. Genuine Trainer **and** merge/`save_pretrained` protection must be preserved; no silent weakening to make tests pass.
- *Predeclared 4-cell readout:* **hours attributable to a `GenerationConfig is invalid` save failure = 0 in all 4 cells**, with the check firing at blocking severity at least once in **≥2 of 4**.
- *False-positive guardrail, stated consistently this time:* enumerate **in advance** every class of card the chosen predicate is expected to block that will not save, count them separately from unexpected blocks, and require **zero unexpected blocking failures** plus **≤1 override on this check per cell**. If the design retains a known residual class, it must appear in the pre-registered enumeration, not be discovered in the readout.
- *Score guardrail:* fixed 24-cell floor. *Other:* required-field completeness must not fall; no rise in relocks caused by the check; post-exit idle from the same crash belongs to E, never to D2's savings.

**Unresolved design question (the reason for deferral).** Which predicate? Three candidates, none free:
1. **Family/argv gate** — cheap; fixes Class A only; retains the `merge`-with-eval-argv residual; unverified as a predictor of future saves.
2. **Operation-aware inspection** of the declared script's save calls — closer to the actual hazard; harder; must not be dismissed merely for being harder; still a static claim about a command that has not run.
3. **A coordinated optional declaration/evidence path** — e.g. the card declares that the run repairs the config before saving, with a named provenance and a validation step. The follow-up allows this as *one hazard-focused item*, but it must define **where the evidence comes from, how it is validated, and what a false declaration costs**. An undefended declaration would convert a nuisance block into an unverified exemption, which is worse than the block.

**Alternatives and what would change the decision.** *Hold D entirely* forgoes the most reproduced mechanical mechanism in the window (8/14 cells, both arms). *Withdraw D v1's block and do nothing* has the same cost and leaves the confirmed false blocks unaddressed. The decision between predicates should be driven by evidence that does not exist yet: a count of `other`/`decode-config` cards that genuinely save, and a validated way to recognise an in-code repair. **Queue note, not a decision:** D v1's held jobs 91060–91063 carry an immutable manifest and receipt; any corrected variant needs its own single-item commit off guard drift `2f64581`, its own tests and manifest, and a whole-block replacement workflow. Nothing here authorises a release, cancellation or registration.

---

### Proposal 2 — **the comparator's n and value must be recovered from the artifact, not assumed** [REV2 — redesigned]

**Surface (one item):** one preflight check with a test — `comparator_same_protocol`. **No schema change, no template change, and — [REV2] — no card field.** The self-declared `comparator.n` of revision 1 is **withdrawn**: a declaration is not verification, and it would have cured neither failure in this window.

**Design.** Resolve the comparator artifact's n from the artifact itself, in order: (a) an explicit count key, if present; else (b) the sample-SE cross-check `n = p(1−p)/SE² + 1`, accepted only when it returns an integer within a tight tolerance. Then:
- **fail** if n cannot be resolved (unknown blocks; unknown must not pass);
- **fail** if the resolved n ≠ `evaluation.protocol.n`;
- **fail** if a non-null `evaluation.comparator.value` does not reproduce the artifact's metric within a stated tolerance (card values are rounded; artifacts carry full precision).

**Evidence [verified here].**
- Mechanism: **46/46** comparator artifacts record no n → 18 printed unverified passes across **8/8** cells.
- Feasibility with zero workflow change: the SE cross-check returns an **exact integer for 46/46** artifacts and matches the card's declared n for **44/46** — including all **nine** legitimate re-measurements whose sibling card declared a different n. In this window the design has **zero false positives on 44 legitimate comparators**.
- Both defects are caught, by different halves: **p00s02 exp-07** by the n check (recovered 150 vs declared 500); **g01r03 exp-08** by both the n check (recovered 800 vs declared 150) and the value check (0.84 vs 0.83875).

**What this explicitly does not do.** It does **not** verify subset identity. g01r03 exp-08's root cause is a wrong-ID slice; it is caught here only because the referenced whole file is an 800-item run whose scalar the card does not reproduce. A scientist who materialises a correctly-derived 150-item summary and points at it would pass — as they should — but so would one who materialises an *incorrectly*-derived one. Proper subset evidence requires declared dataset IDs bound to the source measurement, which this check does not attempt. It also does not verify dataset, seed, decode or model identity; rule 2's "same protocol" remains a manual obligation.

**Known behavioural cost, disclosed.** The check forbids "compare against a slice of a bigger run" unless the slice is materialised as its own artifact. That is arguably what rule 2 already requires, and materialising a slice summary costs seconds — but it is a real change in what scientists may do, and it is the most likely source of overrides.

**Predeclared 4-cell readout.** *Primary:* **zero locks where `comparator_same_protocol` returns `pass` without an n it actually resolved**, and **zero completed decision cards whose resolved comparator n differs from `evaluation.protocol.n` without a recorded reason**; exposure required in ≥2 of 4. *False-positive guardrail:* **overrides on this check ≤1 per cell**; card-authoring wall must not rise by more than 0.05 h above the observed band; `fields_filled` must not fall; count any artifact written solely to satisfy the check. *Score guardrail:* fixed 24-cell floor.

**Method caveats carried verbatim from `metrics.md`.** The SE inversion is sanctioned only for *these* GSM8K binary-accuracy reports with the verified sample-SE convention; it must **not** be generalised to other estimators, and rounded scores must not be treated as exact counts (which is why the check reads the artifact's full-precision fields, not the card's rounded value). It is a **cross-check**, not proof of n.

**Non-overlap and a real redundancy tension.** This is **not** K (verified: K's change is opt-in only) and must not be folded into K; it is **not** C v2 (which was rule-2 wording with an n ≥ 500 threshold — this sets no threshold and enforces the existing "same n" clause). **[REV2]** However, K's frozen `awm/exp_protocol/comparator.py` already implements comparator-report validation for the opt-in path. Two independent baseline-relative candidates are **not stacked and not inherently confounded** — but they would duplicate validation logic and answer overlapping questions. That is a prioritisation and redundant-work issue for the planner, and I flag it rather than resolving it.

**Alternatives and what would change the decision.** *`warn` instead of `fail`*: cheaper, but both defects would still have locked. *Require the Inspect log as the comparator path*: more direct evidence, but imposes a workflow change the SE cross-check avoids. *Hold*: defensible, since no cell demonstrably shipped a worse model from this defect — p00s01's tie-break rescued the case that mattered, and g01r03's exp-08 error did not change what shipped. Decision flips to `warn` if a 4-cell run shows >1 override per cell.

---

### Proposal 3 — **P4 (#23): price the remaining time against a measured cost**

**Surface (one item):** `SKILL.md` rule 8 wording. Ledger #23 already names this surface and direction; it is queued, not frozen.

**Proposed clause.** *When you reject an option because of time, quote the current `timer.sh` reading and the measured wall time of the most comparable operation you already ran in this session, and show the arithmetic.*

**Source cells [verified here].**
- **g01r08's exp-08 card prices two adjacent options, one correctly and one not.** Rejecting `--limit 20`: *"n=500 costs ~25 min of the 2.5 h left"* — cost and budget both quoted, and accurate (the timer read **2:50 at 01:39:54Z**). Rejecting n=1319 one line later: *"~60 min; leaves too little margin, and n=500 already gives a +/-2 point interval"* — an estimate with **no budget quoted**, against 2:50 remaining, in a cell whose own official full test took 12:22. The cell ended at 01:58:01Z with **2:32** unused. The remedy is demonstrably cheap: the scientist had already applied it one line above.
- **g01r07** wrote in exp-08's `next_step` (L9299, 02:51:28Z) *"the remaining time is under an hour and any new run would be unverifiable"* — while the **immediately preceding tool result at 02:51:07Z printed `Remaining time … 1:39`**. The correct number was on screen when the contradicting one was written.
- Supporting: p00s01 (2.12 h unused, no time reason), p00s02 (2.20 h unused), g01r04 (2.25 h unused while naming an untried lead).
- Window context: protocol-installed **8/8** stopped with ≥1.55 h unused (mean 2.079 h); controls 3/6 (mean 1.592 h); Fisher p ≈ 0.055.

**Predeclared 4-cell readout.** *Primary (pricing accuracy, deliberately not hours used):* in all 4 cells, **every end-of-session `alternatives_rejected` entry citing time contains both the timer reading and a measured comparable cost from the same session**, and **zero such entries under-price the remaining budget by more than 30 % against a timer reading taken within 15 minutes**. Target 0 violations. *Context only, reported not targeted:* hours unused at stop; any additional verified result obtained. *Failure guardrails:* runs alive at session end = 0 and `n_locked_open` = 0 (both 14/14 today and must stay there); `fields_filled` must not fall; no rise in cards closed `failed`/`killed` for want of time. *Score guardrail:* fixed 24-cell floor.

**Why not the ledger's original target.** "Cells with ≥1.5 h left and no process ≤ 1/4" is satisfiable by starting a low-value run, collides with ledger #7's *"不诱导无意义跑"*, and rests on an hours→score link this window does not support (r ≈ 0.52, p ≈ 0.06, n = 14, confounded). A pricing-accuracy target cannot be gamed by burning GPU.

**Alternatives.** Keep P4 as an observation (leaves the largest behavioural arm difference unaddressed); fold into A or E (refused — different surface, and that *would* be stacking within one variant); withdraw if a 4-cell block shows the clause satisfied while neither hours unused nor verified results change.

---

### Considered and not proposed [REV2 updated]

- **`test_item_text_absent` / any bare question-or-gold scanner — rejected.** Bare golds are non-specific; the scanner misses paraphrases, aliases and ID-only selection; exact matching misses normalisation differences already present in the Cody example; reading the checker-only test reference needs an access-contract reconciliation; g01r05 is not a clean compliant negative control. The upstream blocker is that **rule 7's intent regarding evaluator-derived item IDs is unresolved** (§5).
- **`long_run_defers_its_only_write` — rejected on coverage.** **[verified here]** frozen B v2 `9f294c3` already says *"dump raw draws to disk before scoring and guard the parser for inf and nan"* — both halves of g01r04's incident. **[REV2]** And the residual "save earlier" lesson is weaker than revision 1 claimed: the early-save cells died *at* their first save and produced no usable checkpoint (§R5).
- **c01s07's "greedy spelling" check — not new.** **[verified here]** frozen D returns `pass` when `do_sample` is true (`8332917:preflight.py:323`), so it already passes c01s07's `do_sample:true, temperature:0.0` artifact and already fails c01s06's `do_sample:false … top_k:0`. Applying the frozen predicate to the 14 shipped configs gives **11 fail / 3 pass**, the three passes being exactly the `do_sample:true` spellings. Separately, per the planner correction: c01s07's successful trajectory does not establish a universally save-safe spelling, `GenerationConfig.from_pretrained` loading is not strict save validation, and a form accepted by one installed version is not a cross-framework contract.
- **`evaluation.protocol.decode` optional field — deferred, with conditions.** **[REV2]** Not because parallel decode-topic candidates are confounded — they are not; independent baseline-relative variants sharing a topic are not stacked. The reasons are: it answers a question overlapping A v2's, it is optional metadata plus new enforcement (a coordinated intervention needing explicit scope), and it must never prohibit a decode card's own treatment contrast. It becomes proposable after A v2's block reads out, with that scope written down. Until then it is a **screen observable for A v2's readout**: count cross-checkpoint comparisons whose two sides carried different effective `generation_config` (g01r08 enforced this by hand twice; g01r06 sidestepped it by writing greedy in the trainer).
- **Tool-timeout kills of model-executing foreground commands — a guardrail observable for E2, not a fourth proposal.** Evidence **[verified here]**: g01r08 is the only cell with a 30-minute tool timeout, which killed a foreground vLLM diagnostic for an exact **0.500 h** loss (`Exit code 143 / Command timed out after 30m 0s`, L5739); p00s01's first 2-minute timeout killed a **foreground GPU training benchmark** at 19:17:17Z, repaired by detaching to `/tmp/bench.sh`, carded at 0.10 h; g01r07 survived the same exposure three times via an inner `timeout 1800` inside a 1900 s tool budget; p00s02's second timeout killed a data build (~0.05 h). Twelve of 14 cells hit a tool timeout, nearly all harmlessly. **[REV2]** A standalone `pitfalls.yaml` entry *is* constructible as an independent variant and would not be stacked with E2 — but it would edit the same entry E2 edits, duplicating work and answering an overlapping question, and a zero-timeout counter is gameable by moving the kill into an inner `timeout`. E2's spec already requires reporting **lost work** as a guardrail failure; this belongs there as *hours of model-executing work lost to a tool timeout, and whether the foreground script carried an appropriate lifetime and handle*. That does not redefine E2's frozen primary metric.

### What the protocol should stop doing, with actual cost

1. **Stop reading `collect`'s `preflight_fail`/`n_overrides` as the friction record.** 6 of ~33 events; 6/8 cells read `0/0` while every one hit a blocking `check` error. A reading rule, not a protocol change.
2. **Stop reporting raw `waiting_hours` or monitor low-memory spans as post-exit idle.** The monitor figure is a sum of last-minus-first low-sample timestamps that discards singletons.
3. **[REV2] `awm wma review`: recorded as scope/availability evidence, not folded into anything.** **[verified here]** two cells executed SKILL step 4b and received `awm: error: argument group: invalid choice: 'wma' (choose from 'slurm','traj','run','split','ptb','exp_protocol','sandbox')` — g01r07 L2552 at 18:36:06Z, p00s02 L1878 at 19:15:53Z; direct cost ≈0 h each. I withdraw revision 1's suggestion to fold a removal into an unrelated SKILL revision: that would be a second, unmeasured intervention inside another candidate's variant, and the instruction serves contexts beyond this sandbox. The correct disposition is a **separately scoped future cleanup** decided on its own evidence.
4. **Do not stop:** the comparator rule, "one card one intervention", rule 6, and `falsified_if`. Each visibly changed a decision in ≥2 cells; rule 6 produced the window's highest-scoring artifact.

### Observations strengthening existing directions (not proposals)

**#26/J** 8/8 cells with pre-card GPU training, plus one 15.1-min pre-lock declared evaluation and one post-launch relock of shipped weights. **#5/H** 8/8 cells hit the `setup.data` schema pressure; 3 cells fabricated a file; 6 overrides in 2 cells. **#27/K** 2/8 cells. **#3/B v2** c01s06's 2.93 h unused pass, g01r04's parser-inf crash and 60,937/63,892 run-on samples, c01s06's six orphan-engine `pkill` round trips. **#6/G** two fresh *control* exposures (c01s01 0.17 h with `'loss': 0.0, 'grad_norm': 0.0 … 'clipped_ratio': 1.0`; c01s04 0.175 h) — but with **0/8 protocol cells running RL**, a G screen on protocol cells would likely show zero exposure and be uninformative under J's own rule; G's testability, not its evidence, is the blocker. **#21/P2** ≥6 cells; **#22/P3** 3 cells. **#17** g01r05 adds a third manifestation. **#18/#28** mixed, as quantified. **#19/#24** repeat and same-scalar phenomena quantified in §1.5 and §R4.

---

## 5. Open questions and the next independent wave

### Open questions

1. **Rule 7's intent.** Does *"No item from it in `failure_examples`, `watch_set`, or training data"* permit evaluator-derived **item IDs**? The shipped text has no exception. This is a policy question for the user/planner and it blocks any enforcement design, including whether a new mechanism may read the checker-only test reference at all.
2. **E2's retention proof remains unresolved.** The two cited events give strict selected-event intervals [0, 0.450556 h] and [0, 0.250833 h]; from those premises alone the number of *proven* failing cells is 0 with 2 strongly suspected. This is not saturation, not authority to swap to G or P1, and not a reason to redefine the frozen `post_exit_idle_h` metric. **Nothing in this window can settle it** — the question concerns cohort g01s01–g01s08, none of which is here. What would settle it: a timestamped producer-absence observation before the deadline (g01s03 needs `e ≤ 15:38:39`; g01s08 `e ≤ 10:39:54`), a retained exit result, or an explicitly adjudicated bounded-teardown assumption recorded as conditional. **[REV2]** Note also that E2 is not merely "use `kill -0`": its retained exit-status and current-invocation artifact-validation conditions remain necessary, so c01s02's late PID repair addresses producer identity only and does not establish E2 compliance. No automatic retention or withdrawal decision follows.
3. **[REV2] A v2's missing endpoint needs a prospective category.** Four of 14 cells (2/8 protocol-installed) never produce the frozen endpoint because they designed greedy in from source inspection. A missing endpoint is **not** a zero-latency pass. The resolution must be declared **prospectively** as an explicit category and must not retroactively change A v2's frozen success requirements. Until it is declared, A's readout on a quarter of its cells is undefined.
4. **[REV2] D2's predicate** (§4, Proposal 1) — which of the three options, and on what evidence.
5. **How common is a mis-derived comparator subset?** One instance is verified (g01r03 exp-08). The base rate is unknown, and Proposal 2 does not measure it.
6. **Per-item identity of official runs** remains unresolved wherever the official log was not retained (ledger #24/#29).

### Suggested next independent wave — scientific readiness only

This is a **scientific-readiness ordering**, explicitly separate from ownership, native-reservation isolation and per-receipt release authorization. The user owns authorizations; the planner owns queue, git and experiment decisions. **I make no queue decision, release nothing, and treat no manifest, ledger word or checked file as a registered held cell.**

1. **Frozen B v2 (`9f294c3`)** and **frozen H (`b52e5f2`)** — both already built, both with large fresh in-window exposure (B v2: c01s06's 2.93 h and g01r04's parser crash; H: 8/8 cells), and neither needs redesign. **[REV2]** These now rank above D, whose predicate is unresolved.
2. **D2** — highest-value objective (8/14 cells, both arms), but not proposable as a design until its predicate question is answered.
3. **Proposal 2 (comparator binding)** and **Proposal 3 (P4 rule 8)** — new single items; each needs a spec, construction off guard drift `2f64581`, tests, independent forward review and an immutable manifest before it is anything at all.
4. **A v2** — after question 3 is answered prospectively.
5. **J, K, E2** — frozen and unregistered; J and K need a fresh scientific and operational audit, and E2 needs question 2 adjudicated.

Every candidate stays baseline-relative and single-item; none is stacked; no frozen manifest or metric is silently changed. Winners need a second independent 4-cell block before any score claim, and held-out confirmation before promotion — unobtainable today.

### Operations context only [REV2 — updated, no decisions]

Per the current records: the control repair **c02s01 / job 91965** is registered held under a new immutable receipt, giving **30 actual `PENDING(JobHeldUser)`**, **22 excluding D4 (91060–91063) and the stale E4 (91064–91067)**, and **0 running**. **[verified here]** the current hourly detector is **PID 2612586** over exactly those 22 IDs, threshold 6 plus 2 buffered clean tail cells, first tick 05:27:36 UTC, replacing PID 2579442 by intentional watched-set expansion with all 21 prior IDs retained and no reset. Native release authorization remains **unanswered**; no reply is assumed. The mixed strict-baseline receipt (3 complete / 5 held) is **not directly releasable by current code**; the planner documented and tested that refusal rather than changing gates. The two later clean cells (c01s05, p00s03) and the placement-quarantined c01s08 remain outside the frozen 14 and change none of these means. **None of this authorises or implies any queue action.**

---

## Accepted, rejected, uncertain

**Accepted (well-evidenced in this window)**
- 14/14 cells ship a greedy-resolving decode config in six spellings; 10/14 measured a same-weight gain of +5.7 to +15.4 pp, whose magnitude tracks the damaged-ness of the sampled parent.
- The greedy-parent save-validation failure occurs in 8/14 cells across both arms, on Trainer saves, weight-averaging saves and a LoRA merge.
- Frozen D's predicate is family-blind; two eval-only false blocks are CPU-replay-confirmed; 16 of 56 cards are structurally exposed to Class A and ~7–8 lock events to Class B (in-code repair).
- 46/46 comparator artifacts record no n; 18 printed unverified passes across 8/8 cells; exactly one declared value fails to reproduce its artifact; the SE cross-check recovers an exact integer n for 46/46 and flags exactly the two defective cards.
- g01r03 exp-08 is a wrong-subset derivation (122/150 IDs outside the intended prefix; true matched contrast −1/150), not a cross-n comparator.
- The corrected exact paired tables in §1.5, with complete ID/epoch coverage and zero input/target/prompt mismatches.
- 8/8 protocol cells ran GPU training before the covering card existed; one launched a declared evaluation 15.1 min before its lock.
- 8/8 protocol cells hit H's schema pressure; `collect` records 6 of ~33 friction events; 9 cards re-locked over 12 events.
- Protocol-installed cells stop earlier than controls (8/8 vs 3/6 with ≥1.5 h unused), and two cells' stated time reasons are contradicted by their own timers within a minute.
- On-policy RL adoption 0/8 vs 4/6 — an observation about recipes.
- 56/56 cards closed, `fields_filled` 1.0, zero stop-hook blocks, no work lost at session end in 14/14.

**Rejected**
- Any established between-arm score effect **and** any established equivalence; and any pooled protocol-vs-control test across two different variants.
- g01r03's record as protocol efficacy or grounds for promotion.
- The approximate c01s04/c01s07 pair tables (219/211, 74/60, 61/61, "±1", "430 discordants", "22 items of repeat noise", "noise floor", "selection inside the noise floor").
- "GRPO alone" causality in c01s01 and its transfer to c01s03.
- c01s02's "2.87 h measured idle".
- g01r07's "opposite-side decode mismatch for 8 h".
- g01r03 and g01r05 as rank inversions.
- Wholesale format-only attribution of the base→post-SFT jump.
- "Executed max n ≥ 500 in 14/14" (it is 13/14) and "selection differs in 2 cells" (3).
- "11 relocks" as a single unit.
- A self-declared `comparator.n` as verification.
- A family-gated D2 presented as a settled design with a zero-non-saving-block guardrail.
- "Early `--save-steps 150` produced an early usable durable checkpoint" and the dose-response framing.
- Treating GPU release, artifact mtimes, `train_runtime` or an empty GPU-process list as producer exit; summing D's failed compute with E's idle.
- Equating unused hours with lower score; treating all failed or rejected compute as useless.
- `test_item_text_absent` / bare-number scanning; `long_run_defers_its_only_write`; c01s07's spelling check as new.
- Folding an ordinary-path comparator check into frozen K, or re-registering C without a discriminating predeclared design.
- Piggybacking a WMA-instruction removal onto an unrelated SKILL revision.
- The claim that parallel baseline-relative candidates are confounded merely by sharing a topic.

**Uncertain (and should block any dependent pass/fail)**
- Rule 7's treatment of evaluator-derived IDs.
- E2's strict non-saturation for cohort g01s01–08; and, in this window, any unconditional positive post-exit idle bound — **none was established by this review**.
- Which D2 predicate is correct, and whether an in-code-repair declaration can be validated safely.
- Whether any of the 14 structurally exposed non-saving cards beyond the two replayed ones would actually have been blocked (parents' configs at lock time are not retained).
- Whether mechanical losses cost score at all — c01s06 lost ~3.79 h and scored 0.7415; g01r03 lost ~0.5 h and scored highest.
- The base rate of mis-derived comparator subsets.
- Official per-item identity wherever the official log was not retained.

---

## Reusable meta-knowledge

1. **Regex-streamed paired counts are not merely imprecise; they can be materially wrong.** Four of five c01s04 pairs and two of three c01s07 pairs changed substantially under a structural `[id, epoch]` join. Parse JSON structurally, key on typed sample IDs, and take an n-prefix from `eval.dataset.sample_ids`, never from the returned `samples` array order.
2. **Never infer sample order from a stored JSON array.** g01r03 exp-08's `d["samples"][:150]` put 122 of 150 IDs outside the intended dataset prefix and produced a recorded comparator value that was wrong in sign.
3. **Check the frozen candidate text in both directions.** This window produced a proposal for something B v2 already says verbatim, and a proposal against a false positive D does not produce.
4. **A self-declaration is not verification.** Prefer a quantity recoverable from the artifact. For these GSM8K binary-accuracy summaries, `n = p(1−p)/SE² + 1` recovers an exact integer from `{accuracy, stderr}` in 46/46 cases — a cross-check `metrics.md` sanctions for this estimator only, and one that must fail closed when it is non-integral.
5. **Keep card units and tool units apart.** `n_relocked` counts cards; `.relocked_from` lengths count events; card-authoring wall spans are heterogeneous intervals, not a common overhead. Averaging across incompatible definitions manufactures false precision.
6. **A metric that is structurally zero in one arm cannot compare arms.** `pitfalls_cost_h` is 0.0 for every protocol-free control by construction.
7. **A filled field is not an exercised one.** `fields_filled = 1.0` in 8/8 while g01r04 declared a 500-item watch set in three cards and left `watch_set_result: null` in all three.
8. **Distinguish a gap that collapses or widens from an ordering that inverts.** Only four of six candidate cases in this window were genuine inversions.
9. **Equal scalar accuracy recurs trivially and means nothing on its own.** Several different artifacts share 978/1319 here. Equal accuracy with disjoint error sets is a real phenomenon; it is not a repeat-variance experiment unless the artifact and the serving configuration are held fixed.
10. **A screen built on a latency between two events needs a prospective rule for cells where the second event never occurs** — 4 of 14 cells here.
11. **Report tool timeouts by what they killed, not by their count**, and design the guardrail around lost work so it cannot be satisfied by relocating the kill.
12. **A preflight FAIL can cause an ordering violation**: g01r08 answered a future-comparator FAIL by launching the missing evaluation 15 minutes before locking. When designing a blocking check, cost out the cheapest way around it and count that as a screen observable.
13. **Quantify a metrics blind spot rather than restating it.** Six of 33 friction events reached `collect`, and six of eight cells read `0/0` while every one hit a blocking `check` error.
14. **"Not established by this review" is different from "shown absent."** Say which one you mean, especially for exit bounds and null tests.
15. **Sharing a topic is not stacking.** Two baseline-relative single-item candidates that touch the same area are independent variants; the real costs are duplicated implementation and overlapping questions, which are prioritisation problems, not construction defects.

<!-- END SYNTHESIS window=window04-local revision=2 -->

---

Revision 2 is emitted as text only; no file was created or modified, no model/evaluator/Slurm/network action was taken, and the only git operations were read-only `git show`/`git diff`. Session `a1e293bb-7b8c-4e8a-ae0b-d305c22d47e3` (PID 2690672, CLI 2.1.260), parented on `ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70`.

Three things changed my conclusions rather than just my wording, so they are worth your attention first:

- **Proposal 2 got a better design out of the correction.** Dropping the self-declared `comparator.n` forced me to look for an artifact-derived route, and the sample-SE inversion recovers an exact integer n from the existing `{accuracy, stderr}` summaries in **46/46** cases, matching the declared n in 44 and flagging exactly the two defective cards — with no workflow change, no new field, and no false positives on the nine legitimate re-measurements.
- **D2 is now explicitly deferred, and for a bigger reason than the one you named.** Beyond the self-contradictory guardrail, the in-code-repair false-block class (~7–8 lock events across 4 cells) is at least as large as the eval-only class a family gate would fix, so scope was never the sole residual.
- **B v2 and H now rank above D in readiness**, since both are already built with large fresh in-window exposure while D's predicate question is open.
