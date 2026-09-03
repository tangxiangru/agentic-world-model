<!-- Fable review of window 02, run in parallel with the planner's (`3a02046`): same eight cells, three trace-reviewer subagents (effort high) + one synthesis, with every loss ≥0.1 h tagged by the Round 02 candidate that covers it. Reports: trace-reviews/round00-window02-fable/. Evidence, not a decision. -->
# Round 00 trace review — window 02 synthesis (the requeued wave: p00r11–p00r15, c01r01–c01r03)

Written 2026-09-03 from the eight per-cell reviewer reports of the requeued wave (protocol
p00r11–p00r15, jobs 90485–90489; control c01r01–c01r03, jobs 90491–90493), the window-01
synthesis (`2026-09-03-trace-review-round00.md`, 16 reports), the round record's rolling
addenda, the directions ledger (#1–#19), and the protocol these scientists saw
(`2f64581:skills/exp_protocol/{SKILL.md,pitfalls.yaml,card.template.yaml}` — the guard tree,
no Round 02 item). p00r12's report carries a Fable correction on stop ids; the corrected reading
is used throughout (its RFT runaway is a model-level weak-stop observation, not B's stop-id
mechanism).

`L<n>` is a line of that cell's `solve_parsed.txt.gz`; timestamps are UTC (all eight sessions
started 22:00–22:02Z on 2026-09-02). "SAID" is what the scientist wrote in a card or a text
turn; "SHOWS" is what the commands, logs and files establish. Counts "both windows" are over
the 14 clean protocol cells and 10 clean controls.

## 1. Per-arm table

| cell | acc | hours used | h to first train launch | protocol h | waiting h | greedy shipped (how verified, when) | RL | RFT tried (verdict) | largest eval n | time left — stated reason |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|
| p00r11 | 0.6952 | 8.11 | 0.36 | 0.40 | 6.1 | yes (inspect-log request config + vllm.py source, L8017–L8035 04:07Z, +6.1 h; +6.1 pp @1319) | no | yes (round 1 +1.5 full, adopted; round 2 −3.1) | 1319 | 1:53 — "a training round takes ~1 h plus evaluation and the last two training rounds both lost accuracy" (exp-09, 06:01Z) |
| p00r12 | 0.7195 | 6.91 | 0.26 | 0.50 | 5.0 | yes (vLLM server-log line L3037 22:08Z; acted 01:33Z at the first post-SFT eval; +9.3 @150, +9.4 @500) | no | yes (two sampling passes abandoned before any training) | 1319 | **3:05** — "Both remaining lines of attack are exhausted, so the budget goes to closing the record" (exp-09, 04:45Z) |
| p00r13 | 0.7074 | 8.07 | 0.28 | 0.35 | 7.4 | yes (server-log line L4244 22:19Z + `get_diff_sampling_param`; acted 01:48Z; +5.4 @500) | no | yes (completed, null: 0.7119 vs 0.7142 full, McNemar p=0.49, loss flat at 0.21) | 1319 | 1:55 — "2.3 h left and the shortest useful training run here was 2.2 h" (exp-06, 05:36Z) |
| p00r14 | 0.6702 | 9.01 | 0.43 | 0.35 | 7.1 | yes (request body L6017 + vLLM source L6022, 23:54Z, +1.9 h; +20 @150) | no | yes (+1.7 full, confounded with 30k MetaMath rows in a from-base retrain) | 1319 | 0:59 — abandon rule "if no checkpoint exists at T-35 min, exp-04 ships" (L8871, 06:03Z) |
| p00r15 | 0.7096 | 8.16 | 0.27 | 0.30 | 6.9 | yes (server-log line L3545 22:14Z; acted 22:59Z, +1.0 h; +6.0 @200) | no | yes (contradicted: 0.675 → 0.675 @200, loss flat from step 1) | 500 | 1:50 — "Do not spend the remaining time on more data of any kind" (L9518, 05:37Z) |
| **protocol wave mean** | **0.7004** (sd 0.019) | 8.05 | 0.32 | 0.38 | 6.5 | **5/5** | **0/5** | 5/5 tried; 1 clean adopt (p00r11 r1), 1 confounded adopt, 3 null | 4/5 full, 5/5 ≥500 | unused mean **1.94 h** |
| c01r01 | 0.7733 | 8.74 | 0.36 | 0 | 7.9 | yes (vllm.py source L4291 22:32Z; acted 00:37Z at SFT end; +10.0 @200) | yes (GRPO ~350 + 70 ext.) | no (`gen_rft.py` written L3149, never run) | 500 | 1:15 — extension tied at paired n=250, "Keeping checkpoint-300." (L16342, 06:43Z) |
| c01r02 | 0.7187 | 8.84 | 0.42 | 0 | 7.35 | yes (vLLM log grep L2560–L2613 22:43Z; acted 00:05Z; +8.7 @150) | yes (GRPO 450) | yes (68.69 vs 69.07 full, discarded) | 1319 | 1:09 — none stated; soup 72.0 < 72.86 |
| c01r03 | 0.7786 | 8.73 | 0.40 | 0 | 7.73 | yes (eval-log grep L3571–L3580 22:29Z; acted 00:39Z; +6.0 @200) | no | yes (+0.6 full, 78.09 vs 77.48, shipped) | 1319 | 1:16 — none stated |
| **control wave mean** | **0.7569** (sd 0.033) | 8.77 | 0.39 | 0 | 7.66 | **3/3** | **2/3** | 2/3 tried | 2/3 full, 3/3 ≥500 | unused mean **1.22 h** |

Wave gap −0.0565. Both windows: protocol n=14 **0.6886** (sd 0.048), control n=10 **0.7557**
(sd 0.032), gap −0.067. Greedy-shipping protocol cells both windows 9/14, mean **0.7123**;
the five sampling cells 0.6458; controls 10/10 greedy. RL: protocol 0/14, controls 7/10. Largest
eval ≥500: protocol 13/14 (8/14 full), controls 10/10 (5/10 full). Cells ending with ≥1.8 h
unused: protocol 6/14 (p00r05, p00r10, p00r11, p00r12, p00r13, p00r15), control 1/10 (c00r06).

Protocol execution this wave (from the reports; consistent with the rolling addenda): 37 cards,
all locked before launch and closed (16/16 launches after lock); overrides: p00r11 4, p00r13 4,
p00r14 5, p00r12 0, p00r15 0; relocks: p00r11 2, p00r14 1; card-recorded pitfall hours
1.55 / 1.95 / 0.48 / 2.15 / 0.20. Protocol tool time 0.01–0.09 h per cell; ≈0.3–0.5 h with
card authoring, 4–7 % of the session, almost all while the GPU was busy.

## 2. What changed versus window 01

Window 01's ranking put decode config first (5/9 protocol cells shipped sampling, ≈0.04 of
the 0.073). In this wave **every protocol cell shipped greedy, every one found it from an
observable rather than the config file (8/8 across both arms), and the wave still trails by
5.6 pp**. The greedy protocol subgroup did not hold its window-01 level either: 0.7272 (n=4)
in window 01, 0.7004 (n=5) here, 0.7123 over both. So the residual the window-01 synthesis
said screen A would reveal (§3.4 there: "if 4/4 greedy and the block still trails by ≥0.03, the
residual is framing") is already visible in the baseline: **greedy protocol −0.043 against the
controls over 9 vs 10 cells.** What follows ranks the explanations for this wave's residual.

Other changes against window 01: RL still 0/5 in the protocol arm (controls 2/3); the
protocol arm now evaluates on the full set *more* than the controls (4/5 vs 2/3), so
evaluation n is no longer an arm-gap mechanism; unused end-of-session hours widened
(protocol 1.54 → 1.94 h, control 1.39 → 1.22 h); the greedy-parent save crash (D) hit 4/8
cells of this wave in both arms; the blind-sleep loss (E) did not recur above 0.25 h in any
cell; and the largest single vLLM-sampling loss of the round is now in a control (c01r03, 1.9 h).

### 2.1 First-stage SFT: volume, mixture, prefix rendering — 5/5 protocol cells under 0.72 at the SFT stage, 2/3 controls over 0.76

**What the traces SHOW at the end of the first SFT, greedy, before any RFT/RL:**

| cell | first-stage greedy | rows (sources) | solutions / problem | prefix share (render) | notable |
|---|---:|---|---|---|---|
| c01r03 | **0.7748 full** (L7055) | 166,772: OMI-2 gsm8k ≤4 + augmented_gsm8k ≤2, 25k math/augmented_math, 7,473 GSM8K human CoT | ≤4 / ≤2 | 10 %, "exactly the way inspect_evals/gsm8k does" (L1706, 22:05Z), drawn from GSM8K train | integer-answer filter (`str(int(a))`, L1640 region); built in 5 min |
| c01r01 | **0.765 @200** (L8017) | 120,000 = `head -120000` of a 327k shuffled pool (L2899, 22:17Z): OMI-2 gsm8k+aug ≤4, 20k math, GSM8K train | ≤4 | 15 %, 2–5 shots from GSM8K train (L2127) | fp32 master, bs 12×10, lr 1.2e-5, max_len 1024 |
| c01r02 | 0.6907 full (L3587) | 118,897: 58k augmented_gsm8k, 12k augmented_math, **12k orca-math with "final answer = last number in the solution" (L1067)**, GSM8K train | 1 | 16 % (L1124) | `Gemma3ForCausalLM` text-only, hand-written chunked CE never validated (report §7) |
| p00r13 | 0.7142 full (L7077) | 80,821 OMI-2, "only 81,069 unique gsm8k-derived problems" (exp-04 alt-rej) | **1** | 15 % | 2 epochs, 65k tokens/step |
| p00r12 | 0.7104–0.714 full (L9588) | 67,473: 7,473 GSM8K + 60k OMI-2 | ≤2 | 25 %, 1–10 shots | 2 epochs; stage 2 on 32k unseen → 0.7210 |
| p00r11 | 0.687 full (exp-07) | 94,946: 80k OMI-2 (1/problem) + GSM8K ×2 | 1 | 10 %, 2–5 shots | first build kept 1.9k rows (regex), 0.15 h |
| p00r15 | 0.665 @200 (L6951) | 161,700 OMI-2 ≤2/problem (83.2M tokens); then +75k Orca → 0.675 @200 | ≤2 | 12–15 % | SAID L8329 (04:26Z): "+50.5, +3.0, +1.0 for 8.9M, 83.2M, 143M tokens" |
| p00r14 | 0.6422 full (L7819) | 92,435: GSM8K ×3 + OMI-2 1/problem; then 179k from-base retrain → 0.6596 full | 1 | **5 %** (L4157 region) | `mix_data --dedup` collapsed the ×3 repeat (L7110 vs L7145) |

Counts. First-stage SFT ≥0.75 greedy at n≥200: control 2/3 this wave, **5/10 both windows**
(c00r03 79.0 full, c00r04 ~77, c00r06 75.4@500, c01r01, c01r03); protocol 0/5 this wave,
**1/14 both windows** (p00r02, 75.3 at n=150 on the easy head). The within-control spread
(c01r02 0.691 vs c01r03 0.775, 8.4 pp, same node, same night) is larger than the arm gap,
and the weak control's post-SFT number sits inside the protocol range — c01r02 then reached
0.7187 only through GRPO (+3.3 to +3.8 full, L4869/L5206).

What separates the two strong controls from the five protocol cells is not rows alone
(p00r15 trained 161.7k rows and read 0.665 @200; p00r14's 179k retrain read 0.6596 full) and
not time-to-data (§2.3). The shared features of c01r01/c01r03 that no protocol cell of this
wave has all of: several solutions per problem (≤4) rather than one, a math/augmented_math
slice, harness-exact prefix rendering on 10–15 % of rows, and (c01r03) an integer-answer
filter; the shared feature of the weak cells is one solution per problem (p00r11, p00r13,
p00r14) or a 5 % prefix (p00r14). This is a recipe pattern in 2 strong + 1 weak control against
5 protocol cells, with p00r15 as a partial counter-example (≤2/problem, 12–15 %, 161k rows,
still 0.665). It is the largest explanation this wave offers for the residual, and it is
**knowledge, not protocol** (ledger #13, proposal F): see §4 for how far the protocol can carry it.

Strongest quotes:
- c01r03 L1706, 22:05:31Z (script docstring): "Render few-shot examples exactly the way inspect_evals/gsm8k does."; L1740–L1741: "Few-shot prefixes are drawn only from the human GSM8K *train* split, i.e. the same distribution the harness samples its 10-shot prefix from."
- c01r02 L1067 (prep_data.py, 22:06Z), orca-math rows: "final answer = last number in the solution" — the trace SHOWS no measurement between the 184k and the 119k mixtures (report §2); the scientist SAID at L1898 the proportions were "tuned".
- p00r13 exp-04 alt-rej (02:35Z), SAID: "the full 14M dump contains only 81,069 unique gsm8k-derived problems (logs/count_full.log) and exp-02 already used 80,821 of them" — one solution per problem was a ceiling the cell chose, not a pool limit; c01r01/c01r03 took up to four.
- p00r14 L4157 (22:24Z), SAID: the 10-shot prefix on every row rejected as "~5x compute for the same loss tokens"; 5 % of rows carried it; the trace SHOWS the prefix cost ~2.5 pts on the held-out probe for exp-02 and helped by 1.6 for exp-04 (L7956).

### 2.2 GRPO versus RFT from a parent that already fits its own samples — RFT null in 4/8 cells, GRPO the only post-SFT lever with a measured full-test gain above one point

**RFT this wave.** p00r13: 42k verified self-samples + 15k replay, 2 epochs, 3.1 h → 0.7119
vs 0.7142 full, McNemar p=0.49; the card's own warning: "loss flat at 0.21-0.23 throughout -
the parent already fits its own samples" (exp-04, 04:5xZ). p00r15: 14k self-samples + gold
for the unsolved + replay, 1.1 h GPU → 0.675 → 0.675 @200; SAID L9518 (05:37Z): "training
loss ... flat from step 1 (0.271 -> 0.274)". p00r12: two sampling passes, no training (1.35 h;
the second harvest 865/900 runaways at T=0.8 with stop ids effective — a weak sampled-stop
distribution of the model, per the Fable correction). p00r11: round 1 +1.5 full (adopted at
n=1319), round 2 from the sharpened parent −3.1 (1.2 h). p00r14: +1.7 full but folded into a
from-base retrain with 30k MetaMath rows (L7250 attributes it to both). c01r02: 46.4k on-policy
rows, 2.3 h → −0.4 full, discarded. c01r03: +0.6 full (8 items), shipped.

**GRPO this wave.** c01r02: SFT 69.07 full → 72.40/72.86 full after 450 steps (L4869, L5206) —
the one full-test-measured gain above one point in either arm this wave. c01r01: SFT greedy
76.5 @200 → ckpt-300 80.2 @500; the +3.7 claim (L16657) compares unequal n and the official
0.7733 lands 0.5 pp *below* c01r03's no-RL 0.7786.

**Does RL explain the control mean?** Both windows: 7 RL controls mean 0.7538, 3 non-RL
controls mean 0.7602 (c00r03 0.7968, c00r07 0.7051, c01r03 0.7786). No. What RL does is rescue
a weak SFT (c01r02) and it is the lever the protocol arm never has: the stated reasons this
wave are again time and an unbuilt pipeline, 4/5 cells (p00r11 never mentions GRPO).

Cell counts, "RFT/STaR from the checkpoint that generated the samples returned ≤0 with a flat
training loss": p00r06 (w01, loss flat 0.23, 0.670 vs 0.672), p00r13, p00r15 — 3 cells with
the loss signature in the trace; by outcome also p00r11 round 2, p00r03, p00r05, p00r07 (w01),
c01r02, c00r04 (w01). RFT rounds that paid were the first round from a fresh SFT parent
(p00r01 +4.8, p00r09 +3.4, p00r10 +4.0, c00r07 +7, p00r11 +1.5) or came with fresh problems.
Hours in this wave on RFT rounds that returned ≤0: p00r13 3.1, p00r11 1.2, p00r12 1.35,
p00r15 1.1, c01r02 2.3 — **9.05 h in five cells**, more than any pitfall class.

Strongest quotes:
- p00r13 exp-04 summary (04:5xZ), SAID: "the parent already assigns high likelihood to its own correct samples (RFT loss started at 0.21 and never moved), so there is almost no gradient signal in them."
- c01r02 L3698, 02:28:39Z, SAID: "Failures are reasoning errors, not formatting. Next lever: **GRPO reinforcement learning** directly on answer correctness." — the trace SHOWS the first RL launch 7 min later (L4122) after a 3-step smoke caught the zero gradient (L4022).
- p00r14 L7224, 01:46Z, SAID: "DPO or GRPO on the 6352 unsolved questions -> no time to build and validate a preference/reward pipeline inside the remaining 6 h, and RFT already covers the 84% the model can solve".
- p00r15 L8839, 04:5xZ, SAID: DPO is "not something to debug with 3 h left and a working SFT loop in hand."

### 2.3 Time to a committed dataset and to the first launch — not an explanation; the protocol arm launched earlier

Trace timings (session start → committed training file → first real launch): c01r01 +4 min
`prep_data.py` run (L1680), 120k committed +17 min (L2899), launch +0.36 h; c01r02 184k at
+6 min (L1194), retuned 119k at +11 min (L1939), launch +0.42 h; c01r03 166,772 at +5 min
(L1850), launch +0.40 h. Protocol: p00r11 build 22:04–22:20 (first build kept 1.9k rows),
launch +0.36 h (OOM) / +0.57 h; p00r12 ~+10 min, launch +0.26 h; p00r13 build at +7 min
(L2676), launch +0.28 h; p00r14 launch +0.43 h; p00r15 16k pilot at +0.27 h, the 161.7k main
run at +1.15 h. Means: protocol 0.32 h to first launch, control 0.39 h. The difference is the
*size and shape* of what was committed (§2.1), not the minutes.

### 2.4 Evaluation n, the dev-150/500 → full inversions, and run-to-run spread — 5/8 cells reversed a verdict at larger n; not an arm mechanism this wave, but C's evidence doubled

Reversals this wave (SHOWS): p00r11 dev-150 ranked the eventual winner third (0.660);
full-1319 sampled made it first (card exp-06: "the ranking changes again"). p00r12 exp-04
closed *contradicted* at n=150 (−1.3), then "n=500: exp-04 > soup > exp-02" (exp-06 diag)
and shipped. p00r14 exp-04 read −0.7 at dev-150 against its `falsified_if`, +1.7 at n=1319
(L8045 region, 04:42Z). c01r01 ckpt-200 > ckpt-300 at n=200 (L14446), reversed at n=500
(L14658). c01r02 SFT2 > SFT1 at n=500 (L3533), reversed on the full set (L3587/L3591). Both
windows: protocol 7/14 cells (p00r01, p00r04, p00r05, p00r09, p00r11, p00r12, p00r14),
controls 3/10 (c00r04, c01r01, c01r02).

The one protocol cell C would have changed here is p00r15: every card at n=200 (SE 0.033),
soup-C chosen best-of-four post hoc at 147 vs 144 of 200, 0.735 @200 → 0.734 @500 → 0.7096
official; items 501–1319 score 0.695 (report §4). It had 2:23 on the clock at the choice
(L9528) and spent the last 17 min on a redundant stock-defaults n=150 run (L9672).

Run-to-run spread of one greedy artefact on the full set: p00r13 four reads 0.7142 / 0.7051 /
0.7096 / 0.7134 (sd 0.0036; exp-06 SAID "treat any gap under ~1.5 points between checkpoints
as unresolved"); c01r02 72.86 vs 72.40 (L5206 vs L5400); p00r11 0.702 vs 0.7043. Ledger #19 —
now three cells; it belongs in C's uncertainty sentence (§3).

### 2.5 Hours unused at the end — protocol 1.94 h/cell vs control 1.22 h; the stated reasons price the last configuration, not the smallest useful one

| cell | left | SAID | SHOWS |
|---|---|---|---|
| p00r12 | 3:05 | "Both remaining lines of attack are exhausted" (exp-09, 04:45Z); "The runaway is a property of the model's post-ANSWER stop distribution, not a script bug" | the diagnosis holds (Fable correction), but stage-2 SFT had just paid +1.1 full and no smaller run was priced |
| p00r13 | 1:55 | "2.3 h left and the shortest useful training run here was 2.2 h" (exp-06, 05:36Z) | 2.2 h is exp-04's 2-epoch/57k-row pass; a 1-epoch or smaller pass was not costed |
| p00r11 | 1:53 | "a training round takes ~1 h plus evaluation" (exp-09, 06:01Z) | its own exp-08 relaunch trained 2 epochs on 21.9k rows in 22 min |
| p00r15 | 1:50 | "Do not spend the remaining time on more data of any kind" (L9518, 05:37Z) | the last 33 min were two confirmation evals of the promoted artefact; a paired ≥500 read of soup-B/soup-C was affordable |
| p00r14 | 0:59 | abandon rule at T-35 min (L8871) | followed to the minute; no idle |
| c01r01 | 1:15 | "With ~2h left, I'll extend GRPO and only swap if it clearly wins." (L14959, 05:42Z) | 1.0 h GPU on an extension that tied |
| c01r02 | 1:09 | "With time left, trying a weight average" (L5218, 06:43Z) | soup + re-verify; no statement on the last 69 min |
| c01r03 | 1:16 | none | nothing weighs the remaining time |

Size: the measured late-session marginal returns are small (p00r14's 55k-row pass +0.8;
c01r01's extension 0; c01r02's soup −0.9), so 0.7 h/cell is worth about a point at most, not
five. What the arm difference shows is the framing the window-01 synthesis named (ledger #16):
per-card `alternatives_rejected` arithmetic that prices "one more run" at the size of the last
run. Both windows 6/14 protocol cells vs 1/10 controls ended with ≥1.8 h unused. A wording
proposal is in §4 (P4); its expected effect is hours and a written costing, not points.

### 2.6 Sleep/waiting practice — E's mechanism did not recur; controls idle more than protocol cells

Protocol waits this wave were sized to the run and checked by tail: idle after a death or exit
p00r11 ≈0, p00r12 0.25 h (a diagnostic piped into `tail` killed by the 15-min tool timeout,
L5960 — E-adjacent), p00r13 0 (its 0.9 h "unplanned" wait was GPU-busy training), p00r14 0.03,
p00r15 ≈0.1 (three 2-min foreground timeouts). p00r14's save crash was seen 3 min after the
run ended (05:58 → 06:01:41, L8693). Controls: c01r01 ≈0.5 h idle despite PID-chained scripts
(223 × `sleep 118` polls); c01r02 ≤0.3 h; c01r03 ≈0.4 h (`sleep 2280` at 00:01 past a 00:33
save, L5276). E's target (<0.15 h idle per cell) is already met by 5/5 baseline protocol cells
of this wave; the p00r02/p00r07 pattern remains 2/14 both windows.

### 2.7 The greedy-config-then-save-crash trap (D) — 4/8 cells this wave, 3.3 h; 9 cells, ≈8.7 h both windows

| cell | run lost | lines | cost | how the cell then avoided it |
|---|---|---|---|---|
| p00r14 | exp-05, 85k-row pass, all steps trained, `trainer.save_model` raised "GenerationConfig is invalid" | L8693–L8733, 06:01:41Z; parent patched in place at 04:26Z L7624 | **1.30 h** + substitute pass shrunk to 55k rows | trainer writes greedy after the save (L8770–L8790) |
| c01r03 | SFT v2 at step 1270/1277, 75.8 min | L5675, 02:03:35Z; `--init ckpt/sft_v1` patched at 00:39 | **1.27 h GPU** + 1.35 h rerun | guard nulls the fields (L5709–L5752) |
| c01r01 | GRPO v1 at step 25 | L9061, 01:07:33Z | 0.40 h | rewrote greedy as `do_sample=True, top_k=1` (L9069–L9080); later saves passed |
| p00r11 | exp-08 at step 683 | L8838–L8870, 05:08Z | 0.35 h | relaunched from the unpatched `ckpts/exp-03/final` (L9044); shipped `do_sample true, temperature 0.0, top_k 1` (L9019) |
| p00r15 | — | L7379 (`--init ckpts/exp-04`, unpatched) | 0 | greedy lives in symlinked variant dirs; structurally impossible |
| p00r13 | — | greedy written into a copy `exp-03-greedy` (L6059) | 0 | same layout |

p00r12 has the mirror image (0.1 h: a candidate scored from a directory still carrying the
*sampling* config, L8140). All four losses came *after* the cell adopted greedy from the
observable A v2 names, in cells that had no A entry; whether A v2's sentence "write it into
final_model/ only, or strip it from any parent you train from" prevents this without D's check is
the first open question for the A block (§5).

### 2.8 The Gemma-3 262k-vocab logits OOM — 7/8 cells this wave, ≥20/24 both windows, 0.05–0.2 h each

p00r11 17.0 GiB alloc at bs 16×2 (L3499, 22:25:12Z) 0.2 h; p00r12 77.7 GB at per-device bs 8
(L3651–L3777) 0.1 h; p00r14 `logits.float()` batch 16 (L3280–L3300) 0.10 h; p00r15 80.5 GB at
token_budget 6144 0.05 h; c01r01 15.99 GiB (L2475, 22:12:37Z) 0.15 h; c01r02 32.01 GiB (L1637)
0.05 h; c01r03 22.5 / 8 GiB (L2547, L2995) 0.05 h; p00r13 pre-empted (token-budget + 8-bit
Adam after the liger install failed). Coupled trap: the reflex `pip install liger-kernel` fills
the 64 MB root overlay — p00r11 L3579 (0.1 h), p00r13 (0.08 h), c01r02 L1761 (0.05 h); window 01
p00r02 L3763, p00r03 L3941, c00r08 L4648 (0.2 h, zero-filled `.py` files); c00r02 and c00r08
knew `uv pip install --target ./pylibs`. Always caught in a smoke; never lost a real run; ≈0.9 h
this wave and ≈3 h across the round. Not an arm mechanism; the cheapest pitfalls entry on the
list (P2).

### 2.9 Gemma-3 processor files and tokenizer files missing from Trainer-saved checkpoints — 3 cells, 0.45 h

p00r14: vLLM exit 1 on the smoke checkpoint because `Trainer.save_model` omits
`preprocessor_config.json`/`processor_config.json` (L3628–L3700, 22:18:38Z; 0.15 h; fixed by
`save_loadable()` L3797–L3830). p00r11: checkpoint-702 had no tokenizer files, vLLM refused it
(02:20–02:24, 0.15 h). p00r01 (w01): the same smoke-checkpoint refusal (L3986, 0.15 h). p00r15
pre-empted it: `save()` copies both files from the snapshot (L3256, 22:10:28Z). `pitfalls.yaml`'s
`final_model_not_loadable` says "merge adapters, save the tokenizer" and does not name these
files (P3).

### 2.10 RL rollout prompt longer than vLLM `max_model_len` — 1 cell (observation)

c01r02 GRPO crashed at step 14: SAID L4252 (02:48:09Z) "a 10-shot prompt exceeded vLLM's max
length"; the script comment L4261 notes TRL does not truncate rollout prompts; fixed with
`--max-prompt 2300` and a length filter, 0.25 h and three launches. c00r08 (w01) carried the
2k-token 10-shot block on 15 % of RL prompts and was slowed, not crashed. One crash cell: an
observation, to be written into G's "related first-launch failures" when G is drafted, not a
proposal of its own. c01r01's flat reward at lr 1e-6 (0.6 h, L9999) is likewise one cell.

### Ranking for this wave's residual gap

1. **First-stage SFT recipe** (§2.1): 0/5 protocol vs 2/3 control at ≥0.75 after SFT; both
   windows 1/14 vs 5/10. Knowledge (F), not protocol; the only item that separates the strong
   controls from every protocol cell of this wave.
2. **What the middle hours bought** (§2.2): 9.05 h in five cells on RFT rounds that returned
   ≤0, three of them with the loss-flat signature in the log at step 1; GRPO 0/5 vs 2/3, the
   only post-SFT lever with a full-test gain above a point (c01r02), yet RL does not explain
   the control mean (0.754 vs 0.760). A pitfalls entry about the measurable signature (P1)
   and G as the belief test.
3. **Unused end budget and its costing** (§2.5): 1.94 vs 1.22 h; 6/14 vs 1/10 cells ≥1.8 h;
   worth ≈1 pt at the measured marginal returns; the wording is P4.
4. **Evaluation n** (§2.4): reversals in 5/8 cells of both arms; costs hours and one ship
   decision (p00r15), not the arm mean this wave; C v2's evidence base doubled.
5. **Mechanical traps, both arms** (§2.6–2.9): D 3.3 h in 4 cells (largest per cell), B ≈4.7 h
   in 5 cells (c01r03 1.9 h, p00r12 1.2 h, p00r14 0.95 h, p00r11 0.6 h), logits/overlay 0.9 h in
   8 cells, processor files 0.3 h in 2 cells. Hours, not points; not arm-specific.
6. **Decode config**: 5/5 shipped; not a mechanism in this wave. Its remaining cost is p00r11's
   3.3 h between the first post-SFT eval (00:52Z) and the greedy write (04:08Z), during which four
   cards were judged on single-sample dev-150 reads.

## 3. Candidate scorecard

Hours "saved" are the reports' own tags for losses the candidate covers; "would have covered"
means the loss or decision is inside the candidate's mechanism, not that the text is proven to
change behaviour.

| candidate (state) | cells in this wave it would have covered | hours (tags) | wording / metric / priority after this wave |
|---|---|---|---|
| **A v2** `decode_config_inherited` (held 91046–91049) | p00r11 (greedy at +6.1 h; four cards judged sampled, exp-06 measurement card ≈0.35 h); p00r12/p00r13 acted at the first post-SFT eval, p00r14 at +1.9 h, p00r15 at +1.0 h; controls 3/3 at SFT end. All 8 used the observable A v2 names (server log 5, request body/source 3). | ≈0.35 h direct; p00r11's decisions on noise | **Metric mis-targeted now:** "≥3/4 cells ship greedy" is met 5/5 by this wave's baseline, so the A block cannot move it. Read instead: hours between the first post-SFT eval and a measured decode choice (baseline this wave p00r11 3.3 h, the other four ≤12 min; controls ≤5 min) and the count of decision cards scored under sampling after that eval (p00r11 4, others ≤1). Wording: none needed. The §3.4 residual test A was to answer is already answered by the baseline (5/5 greedy, −5.6 pp); priority stays first only because 5/14 cells overall still shipped sampling — if a slot is scarce, A can run after D. |
| **B v2** `vllm_offline_prompt_and_stop` (held 91050–91053) | p00r11: stop ids 0.2 h (L4981), parser inf after 22 min 0.4 h (L5262), orphan engine (L5386); p00r12: orphan 0.1 (L5971–L6306), seeded n=4 collapse 0.1 (L5913), 1.0 h harvest without a finish_reason probe (L9006–L9218) — **not** a stop-id loss (correction); p00r14: n>1 stop-string default 0.9 h (L6478–L6872), orphan 0.05; p00r15: orphan 0.02; c01r03: stop ids 1.9 h over two generations (L6273–L6670), orphan 0.01; p00r13 pre-empted (`stop_token_ids=[1,106]` at L4730, 22:22Z); c01r01/c01r02 ran no offline sampling or lost nothing. | ≈4.7 h in 5 cells (protocol 2.8 h, control 1.9 h); both windows ≈5.9 h/8 protocol cells, ≈6.9 h/6 controls | Metric ("sampling-attributable hours in RFT cards < 0.3/cell") discriminates: baseline this wave 0.6 / 1.2 / 0 / 0.95 / 0.02. Wording: add to `symptom` "finish_reason 'length' on most samples while a greedy probe of the same checkpoint terminates — a weak sampled stop distribution of the model, not a sampler default (p00r12 865/900 at T=0.8, L9218)", so the 20-sample probe is read as a go/no-go for the harvest; add p00r11, p00r14, c01r03 to `source` (c01r03 is now the largest single B loss of the round). The sizing sentence ("Budget the pass on the probe throughput") covers p00r12 0.12 h, p00r13 0.25 h, p00r15 0.06 h, c01r03 0.2 h — read it in the screen. Priority unchanged. |
| **C v2** rule 2 + example card at n=500 (held 91054–91057) | p00r15 (all decisions at n=200; post-hoc best-of-4; 2:23 left at the pick); p00r11 (6 h on dev-150; self-escalated at 03:42Z); p00r12 (exp-04 contradicted@150 → shipped@500); p00r14 (falsified_if@150 → +1.7@1319 by the scientist's own escalation); controls c01r01 (200→500 inversion, never full), c01r02 (500→full inversion). | no hour tags; one ship decision (p00r15, ≈2.5 pp between its n=200 headline and official) | Metric confirmed by this wave: the spec's "every card's `evaluation.protocol.n` ≥ 500" discriminates (baseline 0/5 here: p00r11–p00r14 opened at 150, p00r15 at 200) where "largest n ≥ 500" does not (13/14). Wording: add one clause on identical-artefact spread — "the same greedy weights re-read on all 1319 items differ by up to a point (p00r13 four reads sd 0.0036; c01r02 72.86/72.40): a gap under ~1 point between checkpoints is unresolved on one read; repeat or pair it" (ledger #19). Priority unchanged; p00r15 shows the paired ≥500 read was affordable at 2:23 left. |
| **D** `parent_generation_config_valid` (held 91060–91063) | p00r14 exp-05 (1.30 h, would have failed the 04:43:47Z lock, L8239), p00r11 exp-08 (0.35 h, lock 04:48Z), c01r03 v2 (1.27 h + 1.35 h rerun), c01r01 GRPO v1 (0.40 h). p00r15/p00r13 structurally safe. | **3.3 h in 4 cells; ≈8.7 h in 9 cells both windows (+1 smoke catch)** | Highest hours per cell of any mechanical item this wave and both arms. Check logic verified against this wave's files: p00r14's `top_k: 0`/`top_p 1.0` (L6282) and c01r03's `top_k: 0` (L5498) FAIL correctly; c01r01's `do_sample=True, top_k=1` (L9069) and p00r11's shipped form (L9019) PASS and did survive later saves. Wording: name the two safe forms observed (`do_sample true, top_k 1` — c01r01; a symlinked variant directory — p00r15 L7379) and add p00r11, p00r14, c01r01, c01r03 to `source`. **Priority up: run D in the first free wave, not the second**; its screen also answers whether A v2's sentence alone suffices. |
| **E** rule 9 / hook / `run_dies_with_the_session` wait wording (held 91064–91067) | p00r12's 0.25 h tool-timeout kill (L5960) and three 2-min timeouts; p00r15's three 2-min timeouts; no blind wait past a dead run in any cell (p00r14 saw its crash within 3 min). Controls idle 0.3–0.5 h each on clock-sized sleeps. | ≈0.35 h protocol, ≈1.2 h control | **Metric cannot discriminate on this wave's baseline** (5/5 protocol cells already < 0.15 h idle); c01r01 shows PID-chaining still leaves 0.5 h idle from 2-min polls. The mechanism it targets is 2/14 protocol cells both windows (p00r02, p00r07), and the rule-9 text still literally suggests `sleep 900; tail`. Wording unchanged. **Priority last** among the held screens; if a slot is short, defer E behind D/H/G rather than run it as a 4-cell screen that can only show no regression. |
| **H** `setup.data` optional for non-training families (held 91068–91071) | p00r11 4 overrides (L1663, L7532, L8301, L9597), p00r13 4 (L1997, L6045, L7563, L8190), p00r14 1 (L6273), p00r12 3 placeholder measurement cards, p00r15 placeholder `exp-01_protocol_inputs.jsonl`. | minutes | Direct measurement-card overrides now 11 in 5 cells (p00r05, p00r09, p00r11 ×4, p00r13 ×4, p00r14) and placeholders in 7 more; 12/14 cells affected both windows. Metric ("zero fake entries / zero `data_files_exist` overrides, `fields_filled` not lower") discriminates (baseline 2/14 clean). Wording and priority unchanged; cheap, no score effect expected. |
| **G** `trl_grpo_gemma_zero_gradient` (queued, no commit) | c01r01 (zero grad L9157 0.05 h; GRPO v1 save crash is D; flat reward at lr 1e-6 0.6 h), c01r02 (zero grad caught by a 3-step smoke L3909–L4116, 0.05 h; rollout prompt > `max_model_len` 0.25 h L4252). Protocol 0/5 RL; stated reasons p00r14 L7224, p00r13 exp-04/exp-05 alt-rej, p00r15 L8839, p00r12 exp-09 ("unusable until it is fixed"); p00r11 silent. | ≈0.95 h in 2 controls | 7/7 RL controls both windows hit the eos mask; c01r02's 3-min catch is the smoke G prescribes. Wording when drafted: add "TRL does not truncate rollout prompts — filter to `max_prompt_length` ≤ `vllm_max_model_len` − completion (c01r02 L4252; a 10-shot prefix is ~2k tokens)" to the related-failures line, and c01r02's 3-step smoke as the check. Metric unchanged (≥1 `family: grpo` card, RL `pitfalls_hit` < 0.5 h, "no time to build a reward pipeline" absent from `alternatives_rejected`). Priority: up one place — with decode config gone from the residual, the recipe difference left between arms is SFT stage (F) and RL (G); G is the cheaper test and c01r02 is a within-arm demonstration of +3.5 full from a weak SFT. |
| **I** `stop_token_consistent` accepts a script-appended terminator, verifies a rendered row (queued) | p00r14: 5 overrides, the same sentence pasted (L4495 22:26Z, L6273, L7331, L8239, L8950) — the check carried no information after exp-02; p00r11 (L3340 → L3368–L3380) and p00r12 (L4262 → L4265–L4386) moved the terminator into the data to satisfy it; p00r12 then built RFT targets with the terminator *and* an `encode_rows` append (L4796 + L3126–L3134: latent double `<end_of_turn>`, never trained); p00r15's one FAIL was a real field-name problem (`target_text`, L3898–L3915, 36 s); p00r13 passed 500/500. | minutes; the latent p00r12 defect is the risk | Both windows 6/14 cells paid (3 rewrote data in w01, 2 here, 1 overrode ×5). Wording: the check should verify the *rendered* dry-run row ends with the declared terminator **exactly once** — that covers the false positive (p00r14) and the double append (p00r12) with one rule. Metric: overrides of `stop_token_consistent` = 0 and no data rewrite triggered by it; guardrail: 0 cells with a doubled terminator in the rendered row. Priority unchanged (queued after H); the double-terminator case is new evidence that the check's target is wrong, not only noisy. |

Plainly: **E now looks unnecessary as a 4-cell screen** (its metric is already at target in
the baseline); **A v2's metric is mis-targeted** (saturated at 5/5) and needs the hours-to-
measured-choice reading; **D's priority rises** above A's on hours per cell. No candidate is
contradicted by this wave.

## 4. Uncovered losses and proposals

### 4.1 Losses tagged "uncovered" in the eight reports, merged by mechanism

| mechanism | this wave: cell (h) | both-window cells | disposition |
|---|---|---|---|
| RFT/STaR from the parent that generated the samples: first-step loss ≈ parent's final loss, run completes, result ≤ 0 | p00r13 (3.1), p00r11 STaR-2 (1.2, −3.1 pp), p00r15 (1.1, flat 0.271→0.274; tagged under `rft_tried`), c01r02 (2.3, −0.4, loss not reported) | signature in the log: p00r06, p00r13, p00r15 (3); by outcome also p00r11, p00r03, p00r05, p00r07, c00r04, c01r02 | **P1** |
| Gemma-3 262k-vocab logits OOM in the first smoke, plus `pip install liger-kernel` filling the 64 MB overlay | p00r11 (0.2 + 0.1), p00r12 (0.1), p00r13 (0.08 overlay), p00r14 (0.10), p00r15 (0.05), c01r01 (0.15), c01r02 (0.05 + 0.05), c01r03 (0.05) | ≥20/24 (OOM in a smoke in every cell that reported one; overlay in p00r02, p00r03, c00r08 + 3 here) | **P2** |
| Trainer-saved Gemma-3 checkpoint unloadable by vLLM: `preprocessor_config.json`/`processor_config.json` or tokenizer files not copied | p00r14 (0.15), p00r11 (0.15, checkpoint-702 tokenizer files) | p00r01 (0.15); p00r15 pre-empted (L3256) | **P3** (one guidance line) |
| Budget unused at the end ≥1.8 h with a written "does not fit"/"exhausted" argument that prices the last run's configuration | p00r12 (3.05), p00r13 (1.9), p00r11 (1.9), p00r15 (1.8) | protocol 6/14, control 1/10 | **P4** (rule wording) |
| Sampling pass sized without a measured tok/s (ETA 65 min–4.8 h, killed) | p00r12 (0.12), p00r13 (0.25), p00r15 (0.06), c01r03 (0.2) | + p00r03 (0.15) = 5 | in B v2's last sentence already; read it in the B screen |
| RL rollout prompt > vLLM `max_model_len` (TRL does not truncate) | c01r02 (0.25) | 1 (c00r08 slowed, not crashed) | observation; a line for G's text |
| GRPO flat reward at lr 1e-6, restart at 2e-6 | c01r01 (0.6) | 1 | observation |
| planned_h underestimate on a 2-epoch pass (2.19 h vs 1.3) | p00r13 (0.9, GPU-busy) | 1 | observation |
| OMI-2 unique-problem pool discovered by building twice; boxed-regex build kept 1.9k rows; `--dedup` collapsed an intentional ×3; Orca token-length rebuild | p00r12 (0.2), p00r11 (0.15), p00r14 (0.05), p00r15 (0.10) | 1 each | observations (data-build planning) |
| stage-2 continuation on fresh OMI-2 + MetaMathQA regressed 78.6 → 75.4 @500 | c01r03 (1.35) | 1 (p00r06 Orca epoch −3.2 in w01 is a different corpus) | a falsified experiment, not a trap |

### 4.2 Proposals (each one item on the allowed surface; ≥2 source cells)

**P1. `pitfalls.yaml` entry `rft_from_fitted_parent` (check: null).** Sources: p00r13 exp-04
(loss 0.21 flat, 3.1 h, McNemar p=0.49), p00r15 exp-06 (L9518, 0.271 → 0.274, 1.1 h, 0.0),
p00r06 exp-05 (w01; loss flat at 0.23, 0.670 vs 0.672), p00r11 exp-08 (STaR round 2 from the
round-1 parent, −3.1 pp, 1.2 h). It names a failure the scientist can see in the first log
lines, not a method.

```yaml
- id: rft_from_fitted_parent
  symptom: A rejection-sampling / STaR round trains to completion, exits 0, and scores the same as its parent (or lower on a second round); the training loss is flat from step 1 at about the parent's final SFT loss (0.21-0.27 on gemma-3-4b here) instead of starting high and falling.
  cause: The samples were drawn from the checkpoint being trained and filtered for correctness, so the parent already assigns them high likelihood; there is almost no gradient in them. A second round from a parent that was itself RFT-sharpened makes it worse. The pass@k - pass@1 gap that motivated the round (0.90 vs 0.70 in two cells) is real but self-distillation does not close it.
  check: null
  guidance: Before committing the GPU hours, read the first 20 steps - if the loss starts within about 0.05 of the parent's final loss and does not move, the round will return the parent; stop it and record the round as contradicted rather than running the epochs out. Rounds that paid in Round 00 were the first round from a fresh SFT parent or carried unseen problems; the cells that had the flat-loss signature concluded the gap needs a signal the model does not already have (a verifier or preference/on-policy training), which is your call.
  source: p00r13 exp-04 (3.1 h, 0.7119 vs 0.7142 full), p00r15 exp-06 (L9518, 1.1 h), p00r06 exp-05, p00r11 exp-08 (round 2, -3.1); nullctl c01r02 (2.3 h, -0.4 full) (2026-09-03)
```
Target metric (4 cells): hours in cards of `family: rft` closed `contradicted` whose
`training_summary` reports a flat first-step loss < 0.5 h per cell (baseline this wave:
3.1 / 1.1 / 1.2 / 0 / 0); secondary, no second RFT round launched from an RFT parent.
Guardrail: block mean ≥ baseline pool − 0.03; RFT is still attempted in ≥1/4 cells (the entry
must not read as "never RFT").

**P2. `pitfalls.yaml` entry `gemma3_logits_memory` (check: null).** Sources: p00r11 L3499/L3579,
p00r12 L3651–L3777, p00r14 L3280–L3300, p00r15 (exp-02 pitfalls_hit), c01r01 L2475, c01r02
L1637/L1761, c01r03 L2547/L2995; window 01 p00r02 L3707/L3763, p00r03 L3941, p00r10, c00r08
L4648.

```yaml
- id: gemma3_logits_memory
  symptom: The first training smoke dies with "CUDA out of memory ... tried to allocate 8-32 GiB" at logits.float() or in cross_entropy at a modest batch; the reflex `pip install liger-kernel` then fails with "No space left on device" (the container root overlay is 64 MB), or installs half-written .py files that fail on import.
  cause: gemma-3 has a 262,144-token vocabulary; the stock loss materialises fp32 logits of batch x seq_len x 262k. Every Round 00 cell that used fixed-row batching hit this in its smoke run (20+ cells, 0.05-0.2 h each).
  check: null
  guidance: Compute the loss only over supervised positions in chunks (gather the labelled positions before lm_head, or a chunked cross-entropy), or use token-budget micro-batches with gradient accumulation; 8-bit AdamW keeps fp32 master weights inside 80 GB. If you want liger, install it with `uv pip install --target ./pylibs` and PYTHONPATH, never into the root overlay. Run a 12-step smoke before the real launch.
  source: p00r11, p00r12, p00r14, p00r15, nullctl c01r01, c01r02, c01r03 (2026-09-03); p00r02, p00r03, p00r10, c00r08 (2026-09-02)
```
Target metric (4 cells): `pitfalls_hit` hours attributable to logits OOM + package install
< 0.05 h per cell in ≥3/4 (baseline this wave 0.05–0.3). Guardrail: as above. Expected score
effect: none; this is the cheapest item and should not take a screen slot before D/H/G — it can
ride with whichever entry-only candidate is next drafted, as its own commit.

**P3. One guidance line in the existing `final_model_not_loadable` entry.** Sources: p00r14
L3628–L3700 (0.15 h), p00r11 checkpoint-702 (0.15 h), p00r01 L3986 (w01, 0.15 h); p00r15 L3256
pre-empted. Append to `guidance`:

> `Trainer.save_model` on gemma-3 writes weights and config only: copy `preprocessor_config.json`, `processor_config.json` and the tokenizer files from the base snapshot into every checkpoint you will serve with vLLM, intermediate ones included — vLLM reads the image-processor config even for text (p00r14 L3628, p00r11 checkpoint-702, p00r01 L3986).

Target metric: vLLM load failures of the cell's own checkpoints in `pitfalls_hit` = 0 in 4/4
(baseline 2/5 this wave). Guardrail: as above. No score effect expected; a one-line amendment,
not a screen of its own — bundle with P2.

**P4. SKILL.md rule 8 wording: price the smallest run that could change the decision.**
Sources: p00r11 exp-09 ("~1 h" vs its own 22-min round), p00r13 exp-06 ("2.2 h" = the 2-epoch
57k pass), p00r12 exp-09 (3:05 left, "exhausted"), p00r15 L9518; window 01 p00r05 (3.7 h on a
30-min eval estimate that was 4 min), p00r10 (2.2 h on "~35 min each"). Append to rule 8
("Write the situation honestly"):

> When you reject "one more run" in `situation.alternatives_rejected` for lack of time, price the *smallest* run that could change the decision — a one-epoch pass, a 20k-row continuation, a repeat or paired eval — at the throughput and eval time you measured in this session, and write that cost. "A training round takes ~1 h" is a description of your last round; six Round 00 cells ended with 1.8–3.7 h unused on such a sentence while their own logs showed a 4-minute full eval and a 22-minute training cycle.

Target metric (4 cells): cells ending with ≥1.5 h unused and no live process ≤ 1/4 (baseline
protocol 6/14; this wave 4/5), and the final stop's `alternatives_rejected` entry cites a
measured cost. Guardrail: Round 01's criterion (no cell ends with a run alive, no cell held by
the Stop hook without a run) plus block mean ≥ pool − 0.03. Expected effect: hours and a
written costing; at the measured late-session returns (§2.5) about a point, not five. This is
ledger #16's framing residual made concrete; it should run *after* D, with A's hours reading,
because it tests the same "decision framing" hypothesis from the other side.

### 4.3 The data-volume/style finding: what the protocol can and cannot carry

The finding (§2.1) is a recipe: several verified solutions per problem, a math slice, harness-
exact prefix rendering on 10–15 % of rows, an integer-answer filter. Two ways the protocol
could carry it without prescribing a method, and why neither is proposed now:

- *A pitfalls entry about a measurable failure* — "first-stage SFT below ~0.72 greedy on ≥500
  items on gemma-3-4b/GSM8K is a data-recipe ceiling, not a training bug" — would be a
  score-conditioned hint that names what to train; it is exactly the recipe leakage the ledger
  (#13) declined, and the evidence is n=2 strong controls against a counter-example (p00r15:
  161k rows, ≤2/problem, 12–15 % prefix, 0.665 @200). Written now it would also end the
  protocol-vs-null comparison's independence: every later protocol cell would inherit the
  controls' recipe.
- *An optional card field* (`setup.data[].solutions_per_problem`, `setup.data[].prefix_render:
  harness-exact | own | none`) carries no knowledge to the scientist; it only makes the choice
  countable for the meta loop. That is worth having for the *next* round's counting, and it is
  the one thing that could be added without risk — but it does not raise a cell.

So the finding stays `knowledge_to_transfer` (F remains queued as the terse-style entry; this
wave adds no terse-style cell). The observable that would justify an entry later is offline and
available now: first-stage greedy score at n≥500 against {rows, solutions/problem, prefix share
and render, math slice, integer filter, epochs, effective batch} over all 24 bundles; c01r02 vs
c01r03 (same night, same node, 119k mixed vs 167k filtered, 0.691 vs 0.775) is the cleanest
within-arm pair.

## 5. Open questions

| question | what settles it (observable) | where |
|---|---|---|
| Does the first-stage SFT gap hold (protocol ≤0.72, controls split 0.69/0.77) and what drives it — rows, solutions per problem, prefix render, math slice, filter? | Per cell: first post-SFT greedy read at n≥500 (inspect logs), the data-build summary line (rows, per-problem cap, prefix share/render, filters); tabulate over p00r16, c01r04–08 and the 24 existing bundles; c01r02 vs c01r03 as the anchor pair | p00r16, c01r04–08; offline any time |
| Do the remaining controls reach ≥0.77 without RL (c00r03, c01r03 pattern) — is RL absent from the ceiling or from the mean? | `rl_used` and first-stage score for c01r04–08; non-RL control count now 3/10 at 0.760 | c01r04–08 |
| Does the protocol arm's unused-budget split persist (6/14 vs 1/10 ≥1.8 h) and is the stated reason always the last run's size? | Timer at the last turn; the final `alternatives_rejected` wording; the cell's own shortest measured training cycle | p00r16; Round 01 guard cells (same protocol text) |
| Does A v2's "write it into final_model/ only, or strip it from any parent" sentence prevent the save crash without D's check? | In the A v2 block: any `result.failure`/`pitfalls_hit` matching "GenerationConfig is invalid"; count of runs dying at a save; compare with the D block's check firings | Round 02 A v2 and D |
| Does A v2 move the *hours to a measured decode choice* (the reading that still discriminates) rather than the saturated greedy count? | Per cell: first post-SFT eval timestamp → first greedy write; decision cards scored under sampling after that eval (baseline this wave p00r11 3.3 h / 4 cards; others ≤12 min / ≤1) | Round 02 A v2 |
| Does C v2 move every card's `n` or only the ship read, and does any cell repeat an identical-artefact read before ranking within a point? | `evaluation.protocol.n` of every card; `falsified_if` n; count of `contradicted` verdicts reversed at larger n; identical-weights repeat reads (p00r13 pattern) | Round 02 C v2 |
| Is E worth a screen? | Idle time from run death/exit to the next command in the Round 01 guard cells (same `sleep 900; tail` text): if ≤0.15 h in 7/8, E has no baseline to move and its slot goes to G or P1 | Round 01 guard |
| Does the Stop hook ever hold a cell without a live run, and does it change the unused-budget behaviour (a cell blocked 12 times vs a cell that closes cards and leaves 3 h)? | `memory/.stop_hook.json` block counts; timer at end; whether any guard cell ends with a card open | Round 01 guard |
| Does G's entry remove "no time to build and validate a reward pipeline" from `alternatives_rejected`, and does the first protocol GRPO card survive its first save and first 3 steps? | grep of the RL reasons in `alternatives_rejected`; a `family: grpo` card's `pitfalls_hit` (< 0.5 h) and `training_summary` (grad_norm ≠ 0 at step 1); prompt-length filter present | G screen when queued |
| Does the RFT flat-loss signature recur, and would P1 have stopped it early? | For every `family: rft` card: first-step loss vs the parent's final loss in the log, `wall_h`, verdict; count rounds launched from an RFT parent | p00r16, c01r04–08, Round 02 cells |
| Does I's rendered-row check catch both the pasted override (p00r14) and the double terminator (p00r12) without a false positive on data that carries the terminator on disk (p00r13)? | Fixture test: script-appended, on-disk, and doubled terminators; in cells, `stop_token_consistent` override count and a grep of the rendered dry-run row for `<end_of_turn><end_of_turn>` | I when drafted |
| Are the run-to-run spreads of one greedy artefact (p00r13 0.9 pt, c01r02 0.46 pt) typical, and how much of the head-of-test easiness (p00r15 items 501–1319 at 0.695 vs 0.734 on 1–500) is in every n=150/200 comparator? | One script over the full-set inspect logs in the bundles: per-log accuracy of items 1–150, 151–500, 501–1319; repeat-read pairs where they exist | offline |
