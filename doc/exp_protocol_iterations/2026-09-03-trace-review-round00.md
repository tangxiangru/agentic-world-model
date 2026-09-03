# Round 00 trace review — synthesis (protocol v3 vs null control)

Written 2026-09-03 from the 17 per-cell reviewer reports (`trace-reviews/round00/`), the
round record's "Analysis window 01", the directions ledger, and the protocol the scientists
saw (`4ae3d87:skills/exp_protocol/{SKILL.md,card.template.yaml,pitfalls.yaml}`). Clean
strict-site cells only: 9 protocol (p00r01–p00r10 minus the failed p00r08), 7 controls
(c00r01, c00r03–c00r08). c00r02 (truncated at 2.4 h) is shown but not counted. Arm means:
protocol **0.6820** (n=9, sd 0.056), control **0.7552** (n=7, sd 0.033), difference −0.073.

Every claim below is a reviewer's reading of one trace; `L<n>` is a line in that cell's
`solve_parsed.txt.gz`, timestamps are UTC. "SAID" is what the scientist wrote in a card or a
text turn; "SHOWS" is what the commands, logs and files in the trace establish.

## 1. Per-arm table

| cell | acc | hours used | h to first train launch | protocol h | greedy shipped | RL used | RFT tried (verdict) | largest eval n | stop (time left — stated reason) |
|---|---:|---:|---:|---:|---|---|---|---:|---|
| p00r01 | 0.5800 | 8.50 | 0.44 | 0.16 | no | no | yes (−1.0@200 "reject" → +4.8@600, shipped) | 1319 | 1:29 — "the last full run took 1.6 h and would leave no time to evaluate it" |
| p00r02 | 0.7763 | 9.14 | 0.40 | 0.15 | yes (+14.0@150) | no | yes (−0.7@150, reject) | 150 | 0:51 — "did not have budget to re-rank the top candidates at a larger n" |
| p00r03 | 0.6801 | 8.58 | 0.65 | 0.14 | no | no | yes (0.760→0.760 dev500, reject; its ckpt-260 shipped) | 500 | 1:25 — "a new training run plus evaluation does not fit with enough margin" |
| p00r04 | 0.6839 | 8.98 | 0.37 | 0.15 | no | no | yes (+3.0 full, confounded with fresh data) | 1319 | 1:01 — "Two hours are left, which buys one more short pass and one full-test measurement" |
| p00r05 | 0.7043 | 6.28 | 0.38 | 0.25 | yes (+16.0@150) | no | yes (−0.7@150 / −1.0@500, reject) | 500 | **3:43** — "stopped training rather than spend the remaining ~3.5 h on a change I could not have measured" |
| p00r06 | 0.7210 | 9.14 | 0.38 | 0.21 | yes (baked in, unmeasured) | no | yes (0.670 vs 0.672@500, reject; first attempt crashed) | 1319 | 0:51 — "not enough to train, evaluate and verify another candidate" |
| p00r07 | 0.7074 | 9.15 | 0.30 | 0.12 | yes (+7.0@500) | no | yes (−2.6@500, reject) | 500 | 0:51 — "three lever families exhausted"; one scoped RFT gamble |
| p00r09 | 0.6308 | 8.57 | 0.30 | 0.20 | no | no | yes (−1.5@200 "contradicted" → +3.4 full, adopted) | 1319 | 1:25 — "Any further single-run comparison at this scale would be reading noise" |
| p00r10 | 0.6543 | 7.77 | 0.25 | 0.17 | no | no | yes (+4.0@150, adopted) | 500 | **2:13** — "The data-scaling lever is exhausted … Remaining time goes to verifying" |
| **protocol mean / count** | **0.6820** | 8.46 | 0.39 | 0.17 | **4/9** | **0/9** | 9/9 tried; 2 adopted on a later larger-n read | 4/9 full, 8/9 ≥500 | unused 1.54 h mean |
| c00r01 | 0.7339 | 8.83 | 0.21 | 0 | yes (+10.5@200) | yes (GRPO 710 steps) | no (gen_rft.py written, never run) | 500 | 1:10 — "GRPO was still improving when the time budget ran out" |
| c00r03 | 0.7968 | 8.65 | 0.29 | 0 | yes (+4.5@200) | no | yes (+1.0 full, shipped) | 1319 | 1:20 — "All variants cluster within noise; v3 remains the best" |
| c00r04 | 0.7839 | 9.16 | 0.38 | 0 | yes (unmeasured) | yes (GRPO 350+75+50) | yes (75.0 vs 76.7@300 paired, dropped) | 1319 | 0:50 — none stated; last GRPO chunk did not win |
| c00r05 | 0.7415 | 8.52 | 0.41 | 0 | yes (+1.0@200) | yes (GRPO 540) | yes (+0.5 on non-comparable n, kept) | 500 | 1:29 — none stated; continuation rejected |
| c00r06 | 0.7832 | 8.15 | 0.30 | 0 | yes (baked in, unmeasured) | yes (GRPO 680) | no (build_rft.py written, never run) | 1319 | 1:50 — none; wrap-up after full-set pick |
| c00r07 | 0.7051 | 8.65 | 0.21 | 0 | yes (+7.0@200, applied 5.5 h after seen) | no | yes (two STaR rounds; +7 then regress; soup shipped) | 1000 | 1:21 — none; soup won at n=1000 |
| c00r08 | 0.7422 | 8.32 | 0.33 | 0 | yes (+8.4@300) | yes (GRPO 165+50) | yes (24k rows folded into SFT v2) | 800 | 1:43 — "Reward has plateaued"; head-to-head confirmed |
| **control mean / count** | **0.7552** | 8.61 | 0.30 | 0 | **7/7** | **5/7** | 5/7 tried | 3/7 full, 7/7 ≥500 | unused 1.39 h mean |
| c00r02 (truncated) | 0.6846 | 2.40 | 0.45 | 0 | yes | no | launched, killed at 6 % | 300 | 7:36 — text-only turn ended the session |

Protocol execution (from the reports, consistent with the window-01 record): 66 cards, all
locked and closed; every training launch after its lock (three cells chained `lock && launch`
in one command, p00r06 L7177, L7762, L8940); 2 overrides (both `data_files_exist` on
eval-only cards: p00r05 L1492, p00r09 L1824), 3 relocks, one self-reported breach (p00r06
exp-06 eval before its card, L8373).

## 2. Ranked explanations of the score difference

Ranking is by how much of the −0.073 each mechanism can account for, with the cell counts
that show it. Sizes are the reviewers' measurements, not fits.

### 2.1 Decode config — 5/9 protocol cells shipped Gemma's sampling config; 0/7 controls did

**Mechanism.** `evaluate.py` sends only `max_tokens`; vLLM fills temperature/top-k/top-p
from `<ckpt>/generation_config.json`, and gemma-3-4b-pt ships `do_sample: true, top_k 64,
top_p 0.95`, so a checkpoint that keeps that file is graded as a single T=1.0 draw.

**Counts.** Protocol: greedy shipped in p00r02, p00r05, p00r06, p00r07 (mean **0.7272**);
sampling shipped in p00r01, p00r03, p00r04, p00r09, p00r10 (mean **0.6458**); split 0.081.
Control: 7/7 greedy. Measured deltas on the same weights: p00r05 +16.0@150, p00r02
+14.0@150, c00r01 +10.5@200, c00r08 +8.4@300, p00r07 +7.0@500, c00r07 +7.0@200, c00r03
+4.5@200, c00r05 +1.0@200 — median +7.7.

**Size.** If the five sampling cells had taken the median delta the protocol mean would be
0.725 and the gap −0.030; at +10 it would be −0.018. Decode config is **roughly 0.04 of the
0.073** — the largest single item, and the greedy protocol subgroup (0.727) is already
within 0.03 of the control mean.

**What the trace SHOWS about how the two groups differed.** All 12 greedy cells (4 protocol,
7 controls, c00r02) confirmed the mechanism against an *observable* — the request body, the
vLLM log line, or vLLM source — before acting; all 5 sampling cells only `cat`ed the config
file in their first minutes and never looked further.

- p00r02 L4895–L4898, 12:55:43 — dumped the inspect request: `REQ keys: {… 'max_tokens': 4000}`, no temperature; greedy card at 14:49, +14.0.
- p00r05 L3088, 12:38 — grepped the eval log for "Default sampling parameters have been overridden by the model's Hugging Face generation config"; c00r04 L3448 and p00r06 L4707 found the same line; c00r06 L2855 tested `ModelConfig.get_diff_sampling_param()`.
- p00r01 L2317–L2340, 12:34:11 — printed `do_sample: true`; exp-02 card L4143–L4145 SAID: "evaluate.py and templates/ are immutable and the harness owns the decode config; nothing can be changed at inference time."
- p00r04 L1603–L1610, 12:28:45 — printed the file; exp-02 card L5160, 12:48:12 SAID: "Few-shot prompt engineering / decode-config only — cannot be submitted - final_model must be a fine-tune of the base checkpoint."
- p00r10 L3071, 15:30:49 — `train_sft.py` comment: "# vLLM reads generation_config for the stop ids; keep the base one verbatim".
- p00r09 L1697, 14:18:03 — every measurement card declares `temperature: 0.0` although nothing sets it; L7582, 22:46:28 SAID: "Greedy vLLM decoding is not bit-reproducible across runs" — the md5-identical weights scored 0.6406 and 0.6596 because the decode was sampled.
- p00r03 L1746 — its own evaluator defaults to `--temperature 0.0` while the official path sampled; L7731, 16:59 SAID the ~9-point dev/official gap is "a roughly fixed offset"; L9833 blamed "vLLM's batching". p00r04 L3073 has the same greedy own-evaluator and attributed a 4-point swing to `--max-connections` (L9997–L9999).

The belief that decoding is not the scientist's to change appears in explicit words in
p00r01 and p00r04, and implicitly (a `temperature: 0.0` card field with nothing behind it,
"greedy" used for the model's single pass) in p00r03, p00r09, p00r10. Nothing in
`pitfalls.yaml` at `4ae3d87` mentions decoding; candidate A (`3be3a29`) is the direct
response and the amendment in §4 is that A must point at the observable, because the
config file alone was seen by 5/5 non-adopters and moved none of them.

### 2.2 Decisions at n=150–200 anchored by the comparator rule and `falsified_if` — 7 wrong verdicts in 4 protocol cells

**Mechanism.** Rule 2 ("same `n`") was read as "fix `n` at the first card"; the `falsified_if`
written at that `n` then turned noise into a formal "contradicted"; the truth surfaced only
when the scientist re-scored at 500–1319, late.

**Counts.** Protocol cells with a verdict reversed at larger n: p00r01 (2), p00r04 (1),
p00r05 (2), p00r09 (2); p00r02 never went above 150; p00r07 and p00r10 decided everything
at 150–500 on the unshuffled head and never checked. Controls: 7/7 took their final read at
≥500 and 3/7 on the full set; their sweeps were at 300–500, and c00r04 found its 300-item
winner "placed 6th of 11 on the full set" (L8528).

- p00r02 exp-01 L1741, 12:30:42 SAID: "Measure at --limit 500 for a tighter CI — rejected: evaluate.py documents 150 as the intended limit; a 500-item baseline is not a comparator for the 150-item cards that follow".
- p00r09 exp-05 `pitfalls_hit` L6618, 19:40 SAID: "The n=200 dev protocol reversed the true ranking of exp-02 and exp-04 (-1.5 measured, +3.4 actual), which nearly caused a correct method to be abandoned".
- p00r01 L8778, 20:32:36: "n=200 was too noisy — at n=600 the order flips: exp-03 0.593, exp-04 0.563, exp-02 0.545"; final report L9612: "I initially rejected the model I ended up shipping".
- p00r05: @150 exp-03 > exp-04 > exp-05; @500 exp-05 0.718 > exp-03 0.706 > exp-04 0.696 (exp-08 card); exp-05 went from "reject" (18:10) to shipped (18:26).
- p00r04 L8171 → L8517: exp-03 read −8.0@150, +1.6@500, +3.0@1319 (paired z +2.17); the `falsified_if` fired and the scientist ignored it — correctly.

**The first N items are easier, so a small-n read is biased, not only noisy** (both arms):
p00r06 exp-03 card SHOWS "the n=150 slice reads ~7.5 pts easier"; c00r01 76.3@300 → 73.4
official (L5830, L6022); c00r05 76.0@300 → 74.6@500 → 74.1 official; p00r07 0.747@150 /
0.736@500 / 0.707 official; p00r10 0.707@150 / 0.682@500 / 0.654 official.

**The cost of a full read was mis-priced by an order of magnitude in the two cells that
stopped earliest.** p00r05 L12199, 18:17 SAID "1319 would take ~30 min per arm"; its own
n=500 runs SHOW 3–4 min (L12292→L12361, L12555→L12634). p00r10 L8366, 23:00 SAID "~35 min
each"; its n=500 took 4 min (23:00:57→23:05:01). Measured full-set costs: c00r03 ~4 min
(L14450), c00r04 ~2.5 min at `--max-connections 64`, p00r04 three in 8 min (L8517→L8713).

**Size.** Not directly points: the reversals were all caught in-session. The cost is the
hours: p00r05's 3.7 h unused rests on the "cannot be measured" argument (L13467); p00r10's
2.2 h on the 35-min estimate; p00r01's RFT promotion moved from 14:39 to 20:32; and the
selection bias of shipping an n=500-head winner (p00r07 −3, p00r10 −3 vs their last read).

### 2.3 Rejection sampling instead of on-policy RL — 0/9 protocol vs 5/7 controls; the stated reasons were time, "unproven trainer", "needs a reference model"

**Mechanism.** Controls fixed SFT→GRPO in their first 2–25 minutes as a prior (c00r01
`grpo_train.py` at 12:45:57 L3207; c00r05 in the plan at 12:27:08 L1046; c00r06 at 12:49:40;
c00r08 in the plan at 12:28:30 L1286; c00r04 during sft2 at 15:17 L5140). Protocol cells met
the question at card 3–4 with 4.7–7 h left, inside `situation.alternatives_rejected`, and
each time the shorter option won.

- p00r01 exp-03 L6495, 16:30 SAID: "GRPO / online RL — highest ceiling, but with 5.9 h left an unproven trainer risks the whole remaining budget"; exp-04 L7808: "An unproven trainer plus a sampling loop cannot be debugged and run inside that".
- p00r05 exp-05 L9115, 15:50 SAID: "with 6.6 h left, one 2 h supervised run plus a final large-n comparison is the schedule that fits; an RL stage does not".
- p00r06 exp-03 L6212, 16:43 SAID: "a full RL loop does not fit in the 6.1 h left alongside a final-model verification"; exp-09 next_step L9793, 20:45: "DPO/GRPO … was never tested".
- p00r04 exp-03 L7447, 16:35 SAID: "no time for a preference-pair build plus a policy-optimisation run inside the remaining 5.7 h".
- p00r02 L6741, p00r05 L8018, p00r09 L5612: DPO/GRPO "needs a reference-model pass" — every control GRPO ran with β=0 and no reference model (c00r01, c00r04, c00r05, c00r06, c00r08 configs).
- GRPO/PPO/reinforce never appear at all in p00r02, p00r03, p00r07, p00r09, p00r10 (grep counts in the reports); in p00r09 "on-policy" meant rejection sampling (L5139).

**What the trace SHOWS about the "unproven trainer" cost.** Control bring-up: c00r06 0.12 h
(L6125→L6142), c00r01 0.26 h, c00r05 0.26 h, c00r08 ~0.35 h, c00r04 0.81 h. The traps were
the same four in every cell: TRL masks every completion unless `tok.eos_token` is retagged to
`<end_of_turn>` (`grad_norm: 0.0` at the first step: c00r05 L5344, c00r06 L6125, c00r08
L3844, c00r04 L5779), missing `chat_template.jinja`, a greedy `generation_config` in the
`--model` dir crashing the first save, `expandable_segments` vs the colocated engine.

**Size.** Within the controls GRPO added +2 to +5 over each cell's own SFT (c00r05 71.0→76.0@300;
c00r06 75.4→80.4@500 / 76.50→77.94 full; c00r04 77.41→79.30 full; c00r08 +0.75@800; c00r01
~+4–6). But the two non-RL controls average 0.751 (c00r03 0.797 — the best cell of either
arm — and c00r07 0.705) against 0.757 for the five RL controls, and the protocol cells'
replacement levers paid similar amounts when they paid (p00r04 +3.0, p00r09 +3.4, p00r10
+4.0, p00r02 +4.0, p00r06 +3.2). **RL adoption is the largest recipe difference and
explains little of the arm mean.** It matters for the ceiling (three of the four controls at
≥0.78 ran GRPO; c00r03 got there with 193k×1.5 epochs instead) and it is the cleanest
marker of the decision-framing question in §3.

### 2.4 SFT data volume and target style

**Counts / measurements.** Post-SFT greedy scores where measured: c00r03 79.0 full (193k
rows × 1.5 epochs, bs 64, fp32 master, L14568), c00r04 ~77 (73.5k, bs 144), c00r06 75.4@500
(82k, bs 128, fp32), p00r02 75.3@150 (127k), p00r06 74.7@150 (90k × 2), p00r05 71.3@150
(50k × 1), c00r08 70.7@300 (120k, bf16), c00r05 70.5@200, c00r01 68.5@200, p00r07 69.2@500
(79k), c00r07 59@200 (45k GSM8K human CoT + MetaMath). p00r01 trained the smallest corpus
(63k × 1 epoch, exp-02 0.545@600) and never revisited volume.

**Style — the same finding in four cells, two per arm.** GSM8K-train human solutions and
their `<<48/2=24>>` calculator annotations hurt; OpenMathInstruct-2 long-form solutions
help.
- p00r10 L4739–L4771, 16:32: failure dump shows terse `<<a*b=c>>` chains; exp-03 dropped every target with `<<`, `####` or `ANSWER:` from human rows → +8.0@150 (0.533→0.613); its terse share went 6 % → 0/150 (L6051).
- p00r03 L6414, 15:35:56 SHOWS "fs10 acc 0.45 fs0 acc 0.752" on the same checkpoint; close L6638 SAID "380/500 completions imitate the demonstrations' <<48/2=24>> calculator annotations" — a 2.24 h zero-shot run at 0.367 official, repaired by a 0.9 h prefixed run (0.667 at 16:47).
- c00r07 sft1 59 % on GSM8K human CoT ×2 + MetaMath (L3474); c00r06's best-of-arm SFT used OMI-2 only ("no terse GSM8K human CoT in the SFT mix").
- c00r04 is the counter-example: human rationales with `<<…>>` stripped, 77 full.

**Few-shot prefix share**: controls 12–30 % (c00r01 30 %, c00r05 30 %, c00r03 12 %, c00r04
15 %+20 %, c00r06 15 %); protocol 6–12 % (p00r06 6 %, p00r07 8 %, p00r09 8 % with 1–4 shots
only, p00r10 8–9 %, p00r02/p00r04 10 %, p00r05 12 %, p00r01 30 %); p00r03 0 % for its main
run. Only p00r03 paid visibly; the measured prefix effects elsewhere are ±2–4 (p00r09 −1.8,
p00r10 +3.6, c00r05 +3.6).

**Size.** The within-arm spread of SFT quality (c00r03 vs c00r07 = 9 points inside the
control arm) is larger than the residual arm gap after decode config (~0.03). This is
recipe knowledge, not protocol; it is what §4 proposal F carries.

### 2.5 Where the middle hours went — sampler debugging and sequential small SFT steps versus GRPO

Protocol middle hours (SHOWS, from the corrected hour tables):
- p00r01: 14:39→17:52 = **3.2 h** on a self-built offline sampler broken twice (double `<bos>` L6261, no stop token L7118) before the first RFT step; the RFT then read +4.8 full.
- p00r03: RFT 2.05 h incl. 0.25 h of sampling pitfalls → dev500 0.760→0.760 (26 fixed / 26 broken); third slice 0.92 h → −3.4@500.
- p00r05: RFT 1.19 h (0.62 h lost to a parser crash after all 96k draws, L7037) → −0.7; 2.23 h retrain from base → +1.2@500 (p=0.56).
- p00r04: 6.0 h on three SFT passes (+3.0, +0.7); p00r09: two from-base retrains 2.3 h (exp-03 −1.5 noise; exp-04 ~0.9 h more than the continuation the card priced); p00r10: exp-06 1.07 h → −5.3.
- p00r06: 2.7 h on zero-or-negative bets (exp-03 crash 0.63 h, Orca-Math epoch −3.2 in 1.4 h, RFT −0.2); p00r07: four "more external data" cards, three null.
- p00r02 and p00r07: **1.8 h and 2.05 h** to the greedy-config save crash plus blind sleeps (§2.6).

Control middle hours: GRPO 2.6–6.3 h in 5/7 (c00r01 6.3 h, c00r06 5.3 h, c00r05 4.1 h,
c00r08 3.5 h, c00r04 3.5 h incl. bring-up); c00r03 1.9 h RFT + 1.45 h v3 + 1.0 h v4; c00r07
3.8 h of RFT generation of which 2.1 h produced nothing.

**The vLLM offline-sampling traps (candidate B) hit both arms about equally** — a correction
to the window-01 sentence "the controls' rejection sampling worked": c00r03 1.45 h (L4448
"6.3 %" that was 96 %, L10999), c00r07 2.1 h (L4448 "pass rate 0/117365", L5752), c00r06 0.6 h
(L5045), c00r08 0.35 h (parser crash + orphan, L4108, L4276), c00r04 0.3 h (L4561); protocol
p00r01 1.8 h, p00r05 0.65 h, p00r03 0.45 h, p00r10 0.2 h. ≈5.0 h over five controls, ≈3.1 h
over four protocol cells. B is worth doing for the hours; it is not an arm-gap explanation.

### 2.6 Hours lost to the greedy-config save crash and to blind waits — 5 cells, ~5.4 h

A greedy `generation_config.json` (`do_sample false` with `temperature 0.0`/`top_k -1`) in
a *parent* checkpoint makes `GenerationConfig.validate()` raise at the first Trainer save.
It is a direct consequence of the fix in §2.1 and it hit p00r02 exp-06 (died ~19:30, noticed
20:45:56 after `sleep 3300` + `sleep 3000`, L8228, L8274, L8323–L8337: **1.28 h idle GPU +
0.5 h lost**), p00r07 exp-05 (died ~18:03, noticed 19:15:21, L6926–L6951: **0.85 h GPU + 1.2 h
idle**), p00r06 exp-03 (all 508 steps then every save failed, L6787–L6792: 0.63 h), c00r03 v3
(L13202–L13206: 0.76 h), c00r04 GRPO #4 (L6293–L6297, 0.2 h) and its first soup; c00r05
caught it in a 4-step smoke (L5249–L5257). In p00r02 the tail at 19:55 already showed a
stale progress line ("250/659 [30:06<49:14]", L8261) and the scientist slept another 50
minutes. Rule 9's suggested wait (`sleep 900; tail -n 3 <log>`) is exactly the pattern that
missed both deaths. Size: ~2.5 h in two protocol cells that then shrank their last run
(p00r02 exp-07 48k rows instead of 95k, the n=500 re-ranking dropped, L9033) or turned RFT
into a 0.9 h gamble (p00r07 L8029).

### 2.7 Early stopping

Both arms stop with 0.8–1.9 h left (protocol mean unused 1.54 h, control 1.39 h). The two
outliers are protocol cells and both cite the eval-cost error of §2.2: p00r05 3.7 h (L13583:
"I stopped training rather than spend the remaining ~3.5 h on a change I could not have
measured"), p00r10 2.2 h (exp-07 L8366 "~35 min each and the ranking is not the decision
left to make"). p00r05 had also rejected a second epoch and RL at 15:50 "for schedule
reasons that no longer applied at 18:43 with 3.7 h left" (report §6). Not an arm mechanism;
a C-wording matter.

### 2.8 Smaller items seen in ≥2 cells

- Leaked or collapsed hold-outs: p00r04 probe-250 had 248/250 questions in training via OMI-2 (L8571, 19:48; every epoch/continuation decision before that rested on it); c00r05's `augmented_gsm8k` hold-out collapsed to 3 items (L4668) and it diagnosed on train problems.
- Orphaned `VLLM::EngineCore` after `pkill -f`: p00r01 L5586, p00r03 L7524, p00r05 L7100, p00r10 L5047, c00r03 L4337, c00r04 L4447, c00r06 L5664, c00r08 L4296 — in B's guidance already.
- Root overlay 64 MB: p00r02 L3763, p00r03 L3941 (liger install broke imports), c00r08 L4768 (zero-filled `.py`), c00r02 pre-empted it (L1712).

## 3. What the protocol cost and what it bought

### 3.1 Cost

| item | protocol arm | control arm |
|---|---|---|
| hours to first real training launch | 0.25–0.65, mean 0.39 | 0.21–0.41, mean 0.30 |
| protocol bookkeeping (cards + awm calls, composing + exec) | 0.12–0.25 h, mean 0.17 h; 40 s–2.7 min per card; 5–11 cards per cell (66 total) | 0 (TaskCreate ledgers ~1–2 min; RESULTS/README ~1–5 min) |
| bootstrap | 5–10 s of reads, ~490 lines of context re-sent every turn; first lock at +2.5–3.5 min, first GPU action right after | first GPU command at +1–2 min |
| exp-01 comparator card | 1.3–3.5 min in every cell; p00r05 launched it with `evaluate.py` defaults "because protocol rule 2 needs a comparator" (L1331), killed at 59/150, relocked — 0.25 h | baseline eval in background at +1–2 min, no ceremony |
| schema friction | `setup.data` required on eval-only cards: 2 overrides (p00r05, p00r09), 5 placeholders (p00r01 README, p00r02 README, p00r04 probe file at weight 0, p00r06 manifest+README, p00r10 pointed it at the **test parquet** and `data_files_exist` passed on the benchmark file — L1466); `stop_token_consistent` FAIL when the script appends the terminator: p00r01 L4559, p00r03 L5152 (116,971 completions rewritten; the new convention then broke a dedup, 0.05 h), p00r07 L4320–L4339 — no override, but 3 cells rewrote data to satisfy a check their dry-run had already verified; `Write` blocked "File has not been read yet" twice per cell in p00r01–p00r03 | — |

Direct cost ≈ 10 min of a 10 h budget, almost all while the GPU was busy. The ~5 min later
first launch is within the arms' spread.

### 3.2 Steering by the card format

- **Single-lever chains from one incumbent** (p00r07 §6: "+80k, +100k, soup, RFT" each a formal reject at n=500 without a larger-n check; p00r06 six SFT cards; p00r04 "one card = one experiment did not stop exp-03 from being a two-intervention card" — the rule was bent in p00r04 and p00r10 without harm).
- **`situation.alternatives_rejected` + `remaining_h` is where every RL/DPO/second-epoch/larger-n option was priced and dropped** (§2.3 quotes; p00r07 L7254, L7754, L8029; p00r09 "chose the shorter option each time"; p00r10 L8366). The field makes the arithmetic visible; whether it *causes* the shorter choice cannot be separated at n=9 (§3.4).
- **Comparator rule + `falsified_if` at the first card's n** (§2.2) — 7 wrong verdicts in 4 cells, all corrected by the scientist's own larger-n card; p00r02's exp-04 reject on −0.7 (< 1 stderr) and p00r07's rejects on −1.2/−0.2 were never re-checked.

### 3.3 What it bought

- **The protocol's own devices reversed the wrong verdicts.** The comparator rule produced the pure measurement cards that fixed things: p00r09 exp-05 (full-set re-score rescued RFT, L6208–L6579), p00r05 exp-07/08 (n=500, reversed exp-05's reject), p00r03 exp-05 ("deciding correctly between three checkpoints already in hand is worth more than a fourth rushed one"), p00r06's n=500 re-measure before exp-03 and full-set exp-09/exp-11 for the ship. `decision: iterate` forced p00r06's 9-min full eval that decided the ship (L10420).
- **Pre-registered `diagnostic` found the best single finding of the arm**: p00r03 exp-02's fs10/fs0 diagnostic surfaced the −30-point prefix penalty (L6437).
- **Pitfalls list cited and acted on**: `final_model_not_loadable` → CPU/fresh-process verification in p00r01 L8252, p00r04, p00r05 L13119, p00r06 L9330, p00r09, p00r10 L8364 (6/9 cite it by name); `template_unreachable` → byte/hash checks in p00r02 L2121, p00r03 L3254, p00r04 L2855, p00r06 L3410, p00r07; `double_answer_format` → MetaMathQA rejected in p00r01, p00r05, p00r06, p00r09; `comparator_protocol_mismatch` cited as the exp-01 trigger in p00r07, p00r09. Lock refusals that were real: p00r04 exp-01 (probe file fields, L2303).
- **Rule 5 produced honest records of failure**: p00r02 exp-06 `execution: failed` (L8414), p00r06 exp-03 failed/inconclusive with a relock to the archived script (L7097), p00r07 exp-05 relock (L7134); p00r06's self-reported breach (L8373); p00r01's final report "The cards keep the original wrong verdicts; exp-05 records the correction" (L9612).
- **A clean +14 / +16 / +7 attribution of the decode lever** as its own card in p00r02, p00r05, p00r07 — the controls' numbers are also A/Bs but c00r04 and c00r06 shipped greedy unmeasured.

Against this, the controls did most of the same hygiene unprompted: contamination checks 7/7,
template reproduction from the inspect log 7/7 (c00r01 L1320–L1362), `final_model` verified
with stock `evaluate.py` 7/7, safety-net copies from +1.0–3.9 h. What the controls did worse
is harness interaction, not method: c00r02's text-only turn (the Round 01 guard), c00r07's
`nohup … &` killed by the tool timeout (L2660), the 1,300 background waiters in c00r02.

### 3.4 Does the ceremony itself plausibly cost points?

**For:** (i) the protocol arm decided RL late and inside a budget field, and 0/9 tried it,
while 5/7 controls decided it at minute 2–25 as a prior; (ii) rule 2 was read as fixing n at
150–200 (p00r02 L1741 explicitly rejects n=500 on protocol grounds); (iii) `falsified_if` at
that n produced 7 formal wrong verdicts and p00r05's write-off of further training (L11921)
that became a 3.7 h stop; (iv) the exp-01 comparator card cost p00r05 0.25 h; (v) 7/9 cells
had to fake `setup.data`.

**Against:** (i) bookkeeping is 0.17 h and overlapped GPU time; (ii) the greedy protocol
subgroup (0.727, n=4) sits within 0.03 of the control mean and within the control spread
(0.705–0.797); p00r02 (7 cards) beat 4/7 controls; (iii) the two non-RL controls average
0.751 — the recipe the protocol arm converged on is not a losing recipe; (iv) the same rule
that anchored n also produced the measurement cards that repaired every reversal; (v) the
early stops rest on an eval-cost belief that a sentence of fact removes.

**Verdict:** the ceremony's *direct* cost is not points. Its plausible *indirect* cost is the
decision framing — small pre-registered steps judged at a fixed small n, and per-card time
arithmetic that always picks the shorter option — and that cannot be separated from decode
config at n=9. The test is already queued: if candidate A lifts ≥3/4 screen cells to greedy
and the A block still trails the guard-drift pair by ≥0.03, the residual is framing, and the
next variant should remove the fixed-n reading of rule 2 and the small-n `falsified_if`
(the "stop doing" list in §4) before anything is added.

## 4. Proposals

Each is one item on the allowed surface, with ≥2 source cells, the metric a 4-cell screen
reads, and a guardrail. All screens keep the standing guardrail: block mean accuracy not
below the concurrent baseline pool − 0.03. Candidates A/B/C (`3be3a29`, `92d5c79`, `7f117a0`)
stay first; D is A's mechanical half and should be screened after A or folded into A once A
wins.

### D. Preflight check `parent_generation_config_valid` (a check with a test)

**Source cells:** p00r02 exp-06 (1.8 h wall), p00r07 exp-05 (2.05 h wall), p00r06 exp-03
(0.63 h), c00r03 v3 (0.76 h), c00r04 GRPO #4 + soup (0.2 h); c00r05 caught it in a smoke.
Five cells, ~5.4 h, one root cause, and it is created by the fix A recommends.

**Exact text.** New check in the preflight, run for `family` in {sft, rft, dpo, grpo,
distill} when `setup.parent_checkpoint.path` is a local directory:

```
id: parent_generation_config_valid
reads: <setup.parent_checkpoint.path>/generation_config.json
FAIL if transformers.GenerationConfig.from_pretrained(path).validate() raises, or if
     do_sample is false and any of temperature/top_k/top_p is set to a non-default value
     (temperature 0.0, top_k -1 or 0, top_p 1.0).
message: "The parent's generation_config.json is greedy-patched; transformers validates it on
     every Trainer save and the run will die at its first checkpoint (p00r02 exp-06, p00r07
     exp-05). Train from a copy with a valid sampling config, or set model.generation_config
     to a valid one before the first save, and write greedy into final_model/ only."
```
Test: a fixture parent with `{"do_sample": false, "temperature": 0.0}` FAILs; the stock
gemma-3 config and `{"do_sample": false}` alone PASS; an absent file SKIPs. `pitfalls.yaml`
entry `decode_config_inherited` moves from `check: null` to this id.

**Target metric (4 cells):** hours in `pitfalls_hit` attributable to "GenerationConfig is
invalid" = 0, with the check having fired at least once in a cell that adopted greedy (so
the block exercised it). **Guardrail:** no false FAIL on a card whose parent carries the
stock config (count of overrides of this check = 0).

### E. Rule 9 wording: wait on the process, not the clock

**Source cells:** p00r02 (1.28 h idle, L8228/L8274 — the 19:55 tail already showed a stale
line), p00r07 (1.2 h idle, `sleep 3300`/`sleep 3000`, L6834/L6880; it then switched to
`while pgrep … sleep 30` at L7458), c00r04 (12-minute blind spots at 17:12 and 17:40).
Controls that never lost time this way waited on the PID (c00r01 `while ps -p PID` with
multi-hour timeouts, L3962; c00r08 40 foreground waits each tailing reward).

**Exact text** — replace the parenthetical in rule 9 "(`sleep 900; tail -n 3 <log>`, repeated,
with a long Bash timeout)" with:
> Wait on the process, not the clock: `while kill -0 <pid> 2>/dev/null; do sleep 300; tail -n 1 <log>; done` with a long Bash timeout, and compare each tail to the previous one — a progress line that has not changed across one wait means the run is dead (a save can fail after every step has trained). Chain the evaluation to the run's exit in the same script so the GPU never idles between them.

**Target metric:** GPU-idle time between a run's death/exit and the scientist noticing,
summed per cell, < 0.15 h (the timeline tool can read it from the last log timestamp vs the
next command). **Guardrail:** no cell held by the Stop hook without a live run (Round 01's
criterion), accuracy guardrail as above.

### F. `pitfalls.yaml` entry `terse_target_style` (check: null)

**Source cells:** p00r10 (+8.0 from dropping `<<…>>` human rows, L4739–L4771 → L6051),
p00r03 (fs10 0.450 vs fs0 0.752, L6414; completions imitating the demos' annotations, L6638),
c00r07 (59 % SFT on GSM8K human CoT + MetaMath, L3474), c00r06 (OMI-2-only SFT 75.4 %, the
best first-stage number of the round). Recipe knowledge; the entry describes a failure the
scientist sees, not what to train, and the ledger (#13) asked for a second occurrence, which
p00r10 and c00r07 supply.

**Exact text:**
```
- id: terse_target_style
  symptom: The tuned model writes one-line chains with calculator annotations (<<48/2=24>>) or
    keeps inventing a new problem after its own answer line; per-item failures are wrong
    arithmetic in short chains; accuracy with the grader's 10-shot prefix is far below
    accuracy without it on the same items.
  cause: The training targets were GSM8K-train human solutions (terse, annotated) or the
    model learned to continue the grader's demonstrations instead of answering in its own
    style. The demonstrations are in that same style, so a model trained on it imitates them.
  check: null
  guidance: Measure the same checkpoint with and without the grader's few-shot prefix on your
    dev set before the second card (p00r03 did this in one eval and found 30 points). Prefer
    long-form verified solutions (e.g. OpenMathInstruct-2) as targets and strip calculator
    annotations from any human rows; put a k-shot prefix on a share of rows so the model
    tolerates the demonstrations rather than copies them.
  source: p00r10 exp-02/03 (+8.0 at n=150), p00r03 exp-02/03 (−30.2 on dev500 with the
    prefix, repaired in 0.9 h), nullctl c00r07 (SFT 59 % on human CoT), c00r06 (OMI-2 only,
    75.4 %) (2026-09-02)
```
**Target metric (4 cells):** first-stage SFT accuracy under the card's own protocol ≥ 0.70
greedy in ≥3/4 (Round 00 protocol first-stage greedy: 0.692–0.753 where greedy was
measured; p00r10's was 0.533 sampled); no cell records a prefix penalty > 5 points after
its first card. **Guardrail:** as above.

### G. `pitfalls.yaml` entry `trl_grpo_gemma_zero_gradient` (check: null)

**Source cells:** c00r01 L4546–L4632, c00r04 L5779–L5801, c00r05 L5344–L5634, c00r06
L6125–L6142, c00r08 L3844–L3887 — 5/5 GRPO cells hit the identical first-step failure; none
in the protocol arm because none ran RL. The ledger (#6) queues this until a protocol-arm
RL card exists; the reviews suggest the reverse dependency: the "unproven trainer" belief
(p00r01 L6495, L7808) is what keeps the source cell from existing, and a listed pitfall with
its five-line fix is the cheapest test of whether the belief or the card framing blocks
adoption (§3.4). The entry names a failure and its fix, not a method to run.

**Exact text:**
```
- id: trl_grpo_gemma_zero_gradient
  symptom: TRL GRPOTrainer logs loss 0.0 and grad_norm 0.0 (or completions/clipped_ratio 1.0)
    from the first step; reward is non-zero; the run "trains" for hours and the checkpoint is
    the parent.
  cause: TRL marks a completion truncated when its last token is not tokenizer.eos_token_id;
    gemma-3 chat turns end with <end_of_turn> (106), not <eos> (1), so with
    mask_truncated_completions every rollout is masked. Related first-launch failures in the
    same cells: a greedy generation_config.json in the --model dir fails the first save;
    chat_template.jinja missing from the checkpoint; the colocated vLLM engine will not start
    under PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
  check: null
  guidance: Before the trainer is built set tok.eos_token = "<end_of_turn>" (retag
    tokenizer_config/special_tokens_map if you save that tokenizer), give the --model dir a
    valid sampling generation_config, copy the grader's template in as chat_template.jinja,
    unset PYTORCH_CUDA_ALLOC_CONF, and run a 3-step smoke that asserts grad_norm != 0 and
    survives one save. Five control cells did this in 0.12–0.81 h with beta 0 (no reference
    model) at 17–57 s per step of 128–192 rollouts.
  source: nullctl c00r01, c00r04, c00r05, c00r06, c00r08 (2026-09-02); no protocol cell ran RL
```
**Target metric (4 cells):** ≥1 cell locks a `family: grpo` card, and its RL `pitfalls_hit`
< 0.5 h; secondary, the phrase "unproven trainer"/"reference model" does not appear in
`alternatives_rejected` as the reason for rejecting RL. **Guardrail:** as above, plus no
cell loses > 1 h to an RL run it cannot evaluate (a `grpo` card closed `failed` with > 1 h
`wall_h` fails the screen).

### Amendments to A, B, C

**A (`decode_config_inherited`, `3be3a29`).** The reports change three things. (1) Numbers:
"Three of the seven" → five of nine shipped the stock config; greedy 0.727 vs sampling 0.646.
(2) The guidance must name the *observable*, because 5/5 non-adopters saw the config file and
0/5 acted, while 12/12 who checked the request or the server log acted: add "Confirm it in
one minute: the vLLM server log of any `evaluate.py` run prints 'Default sampling parameters
have been overridden by the model's Hugging Face generation config' (p00r05 L3088, c00r04
L3448), and the inspect log's request carries only max_tokens (p00r02 L4895)." (3) Two
beliefs to contradict in the text: "the harness owns the decode config" (p00r01 L4145) and
"decode-config only cannot be submitted" (p00r04 L5160) — add "The checkpoint owns decoding;
a fine-tuned checkpoint whose generation_config.json you rewrote is the deliverable." And
one sentence for the two cells with greedy own-evaluators (p00r03 L1746, p00r04 L3073): "If
you build your own evaluator, copy the grader's request body, not only its prompt; a
dev/official gap with a fixed sign is a decoding difference until shown otherwise."

**B (`vllm_offline_prompt_and_stop`, `92d5c79`).** (1) The `source` claim that c00r06 and
c00r07 "lost nothing" is wrong: c00r07 lost 2.1 h (L4448 → L5752) and c00r06 0.6 h (L5045)
to exactly these defaults; c00r03 1.45 h (L10999); c00r05 and c00r02 pre-empted them by
passing `stop_token_ids` / `stop=["<end_of_turn>"]` from the first script. (2) Add the n>1
mechanism c00r07 read from vLLM source (L5632): child requests of an `n>1` parent are fanned
out before `update_from_generation_config`, so the stop id is dropped only when `n>1` —
"symptom: 0 % pass at n>1 while n=1 works". (3) Add the third trap seen in three cells:
a parser that converts a model output like `1e999` crashes with `OverflowError: cannot
convert float infinity to integer` after the whole pass (p00r05 L7037, c00r01 L4724,
c00r08 L4108) — "dump raw draws to disk before scoring; guard the parser for inf/nan".
(4) Keep "probe 20 samples" and add "print finish_reason" (c00r03 L10808 found 96 % where
the pass read 6 %).

**C (rule 2 addendum, `7f117a0`).** (1) Say that `--limit N` scores the **first N** items
and that the head is easier: "the first 150–300 items read 2–7 points above the full set in
six cells of both arms (c00r01 76.3@300 → 73.4; c00r05 76.0@300 → 74.1; p00r07 0.747@150 →
0.707; p00r10 0.707@150 → 0.654; c00r04's 300-item winner placed 6th of 11 on the full set),
so a small-n number is biased upward as well as noisy." (2) State the cost, because the two
cells that stopped earliest mis-priced it 7–10×: "A full 1319-item `evaluate.py` run costs
3–10 minutes at `--max-connections 32–64` (c00r03 4 min, c00r04 2.5 min, p00r04 three in 8
min); p00r05 and p00r10 priced it at 30–35 min and stopped on that." (3) The current sentence
"Decide nothing that ships a checkpoint on fewer than 500 items" covers only the ship
decision; the 7 wrong verdicts were adopt/reject between cards. Add: "and write
`falsified_if` at an n whose standard error is below the delta you claim, or as a paired
statistic; a `contradicted` verdict on a delta inside one standard error is a description of
noise." (4) Say the comparator may be re-scored: "Raising `n` mid-session is allowed and
expected — re-score the incumbent under the new protocol (p00r06 did, 0.15 h) rather than
keeping the first card's `n` for every later card (p00r02 L1741)."

### What the protocol should stop doing

| stop | cells that show the cost | change |
|---|---|---|
| Reading rule 2 as "fix `n` at exp-01" — the comparator rule anchoring 150–200 for the whole session | p00r02 L1741 (explicit), p00r01 (n=200 ×3 cards), p00r10 (n=150 every decision), p00r05 (six n=150 decisions) | C amendment (4) above |
| `falsified_if` at n=150–200 producing formal `contradicted` verdicts on sub-stderr deltas | p00r01 exp-03/04, p00r05 exp-04/05, p00r09 exp-03/04, p00r04 exp-03 (7 reversed); p00r02 exp-04, p00r07 exp-06/08 (never re-checked) | C amendment (3); template comment on `falsified_if`: "at the declared n, only for a delta larger than its standard error" |
| Requiring `setup.data` (≥1 entry with a file path) on cards that train nothing | p00r05 override L1492, p00r09 override L1824, placeholders in p00r01 L1991, p00r02 L1876, p00r04, p00r06 L2076, p00r10 L1466 (the benchmark's test parquet, and `data_files_exist` passed) — 7/9 cells | schema: `setup.data` optional when `family` ∈ {decode-config, merge, other} and `command` trains nothing; `data_files_exist` SKIPs with a message. Ledger #5, queued Round 03; the trace count (7/9, one of them touching the test file) argues for Round 02's first free slot |
| `stop_token_consistent` reading the raw target field when the pipeline appends the terminator at render time | p00r01 L4559, p00r03 L5152 (116,971 completions rewritten; caused the 0.05 h dedup mismatch at L6790), p00r07 L4320–L4339 — 3/9 cells rewrote data instead of overriding, so the ledger's "0 overrides → evidence disappeared" (#9) undercounts | accept `stop_token: {value, appended_by: script}` and verify one rendered row from the dry-run instead of the file |
| The `sleep N; tail` wait pattern in rule 9 | p00r02 L8228/L8274, p00r07 L6834/L6880 (2.5 h idle) | proposal E |

## 5. Open questions for the next wave

| question | what settles it (observable) | where |
|---|---|---|
| Does decode config close the gap, or only most of it? | In the A block: greedy shipped in ≥3/4 (cell_read `do_sample=false` writes + `final_model/generation_config.json`); block mean vs the 2 guard-drift cells. Gap ≤ 0.03 → framing is not costing points; gap ≥ 0.03 with 4/4 greedy → §3.4 residual is real | Round 02 A, drift pair |
| Is on-policy RL absent from the protocol arm because of the "unproven trainer / reference model" belief or because of card framing (late decision inside `remaining_h` arithmetic)? | Any `family: grpo` card in A/B/C/drift cells; the reason strings in `alternatives_rejected` for RL (grep "unproven", "reference model", "does not fit"); with G, RL bring-up `pitfalls_hit` hours and whether ≥1 of 4 cells launches it | Round 02 cells; G screen |
| Does C move the *decision* n or only the final read? | `evaluation.protocol.n` of every card (not just the last), count of `contradicted` verdicts later reversed at larger n, and whether any cell still declares `falsified_if` at n ≤ 200 | Round 02 C block |
| How much easier is the head of the test set, exactly? | One script over the full-set inspect logs already in the bundles (p00r01, p00r04, p00r06, p00r09, c00r03, c00r04, c00r06): accuracy of items 1–150, 151–500, 501–1319 per log; if the head is +3–7 points, C's sentence stands and every n=150 comparator is biased, not only noisy | offline, any time |
| Does the greedy-save trap disappear under A's guidance alone, or does it need D's check? | In the A block: any `pitfalls_hit` or `result.failure` matching "GenerationConfig is invalid"; count of runs that died at a save | Round 02 A block |
| Is the c00r06 = batch-1 identity (1033/1319) reproduction or coincidence? | Operator diff of the two REPORTs and recipes (ledger #12) | operator |
| Are the late session starts (p00r06 +26 min, p00r07 +44 min with the timer at 9:59) job-level and recurring? | Job start vs session start over the requeued wave | operator |
| Does the Stop hook ever hold a cell without a live run, and how often does it block? | `memory/.stop_hook.json` counters in the 8 guard cells; any cell ending with hours unused and no card open | Round 01 guard |
| Would a `falsified_if` written as a paired statistic (McNemar on the same items) have produced fewer wrong verdicts than an n threshold? | p00r01 (p=0.043 exp-03 vs exp-02 at n=600), p00r04 (z=2.17 at 1319), p00r05 (all p>0.26 at 500), c00r04 (p≈0.54 at 300) already paired; the C block can count cards that report a paired statistic vs a threshold | Round 02 C block |
