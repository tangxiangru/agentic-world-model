# Round 00 trace review — analysis window 02

Written 2026-09-03 from all nine reports in
`doc/exp_protocol_iterations/trace-reviews/round00-window02/`, the Round 00 record, and the directions
ledger. This is evidence, not a candidate or release decision. `L<n>` refers to the
cell's `solve_parsed.txt.gz`; timestamps are UTC. **SAID** reports the scientist's
claim, while **SHOWS** reports what the trace or artifact establishes.

## Scope, exclusions, and uncertainty

The NEW window contains five protocol cells (p00r11–p00r15) and three controls
(c01r01–c01r03). Their official means are protocol **0.700379** (n=5, sd 0.01895)
and control **0.756887** (n=3, sd 0.03315), a protocol-minus-control gap of
**−0.05651**. The current clean pools are protocol **0.688563** (n=14, sd 0.04632)
and control **0.755724** (n=10, sd 0.03139), gap **−0.06716**.

p00r10 is **CALIBRATION only**. It is shown separately for continuity and is excluded
from every NEW count, mean, source threshold, and proposal. No failed, truncated,
non-strict-site, pilot, or unvalidated attempt is introduced here. Internal dev/full
scores are diagnostic evidence, never substitutes for official accuracy. With only
five versus three NEW cells, score-gap rankings are mechanistic explanations, not
causal estimates; many recipe changes are bundled within a cell.

## 1. Per-arm header tables

Every reviewer-header field is reproduced below. Semicolons compress list-valued fields
without dropping an item.

### Protocol arm — NEW cells

| cell | arm | official accuracy | hours used | h to first real train | protocol h | waiting h | greedy shipped | RL used | RFT tried (verdict) | largest eval n | stop reason | top contributors | one protocol change | knowledge to transfer |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|---|
| p00r11 | protocol | 0.695224 | 8.11 | 0.36 | 0.25 | 6.13 | yes | no | yes; small round supported at n=150, larger/second rounds contradicted at n=1319 | 1319 | completed after fresh-process verification; all processes exited with 1:53 left | format-correct full SFT; greedy generation config; first small RFT round | surface inherited sampling and require an explicit decode decision before first post-SFT comparison | — |
| p00r12 | protocol | 0.719484 | 6.91 | 0.27 | 0.20 | 5.06 | yes | no | yes; sampling attempts abandoned, no RFT train because rollouts were slow then non-terminating | 1319 | remaining research lines exhausted; artifact verified and all processes exited with about 3:05 left | format-correct full SFT; greedy generation config; continued SFT on unseen problems | surface inherited sampling and require an explicit decode decision before first post-SFT comparison | — |
| p00r13 | protocol | 0.707354 | 8.07 | 0.28 | 0.13 | 7.35 | yes | no | yes; contradicted by paired n=500 and full n=1319 | 1319 | post-SFT interventions were indistinguishable or worse; repeated full evals selected the simpler verified model and GPU was idle | format-correct full SFT; greedy generation config; repeated full-test selection under decode noise | surface inherited sampling and require an explicit decode decision before first post-SFT comparison | — |
| p00r14 | protocol | 0.670205 | 9.01 | 0.43 | 0.09 | 7.13 | yes | no | yes; supported | 1319 | completed, GPU clear, all six cards closed, with 0:59 left | termination-format SFT; greedy decode config; larger SFT plus RFT corpus | add a preflight decode audit requiring an explicit generation-config choice before first post-train eval | — |
| p00r15 | protocol | 0.709629 | 8.16 | 0.27 | 0.01 | 6.92 | yes | no | yes; contradicted | 500 | completed, GPU idle, final_model verified, with 1:50 left | termination-format SFT; greedy decode config; converged-checkpoint weight averaging | add a preflight decode audit requiring an explicit generation-config choice before first post-train eval | — |
| **mean / count** | protocol | **0.700379** | **8.05** | **0.32** | **0.14** | **6.52** | **5/5** | **0/5** | **5/5 considered/tried; 4/5 actually trained RFT** | **1155 mean; 4/5 full** | all deliberate; about 0:59–3:05 unused | format-correct SFT 5/5; greedy 5/5; robust post-SFT gain mixed | same decode-audit direction in 5/5 headers | none listed in 5/5 |

The counts and values above come directly from the five protocol report headers. The
distinction between “RFT tried” and “RFT trained” matters: p00r12 built/sampled twice but
abandoned before training [p00r12, 2026-09-03T02:18:13Z L6818-L6828;
2026-09-03T04:45:50Z L9477-L9482].

### Control arm — NEW cells

| cell | arm | official accuracy | hours used | h to first real train | protocol h | waiting h | greedy shipped | RL used | RFT tried (verdict) | largest eval n | stop reason | top contributors | one protocol change | knowledge to transfer |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|---|
| c01r01 | control | 0.773313 | 8.76 | 0.36 | 0 | 7.94 | yes | yes | no; pipeline prepared only | 500 | completed verification and cleanup with 1:15 still available | large format-matched SFT; greedy decode about +10 points; GRPO about +3.7 points | — | preflight generation-config saveability; map Gemma end-of-turn into RL termination; escalate checkpoint selection beyond n=150 with paired tests |
| c01r02 | control | 0.718726 | 8.86 | 0.42 | 0 | 7.35 | yes | yes | yes; no gain, discarded | 1319 | completed verification with 1:09 still available | large completion-only SFT; greedy decode about +8.7 points; GRPO about +3.3 points | — | preflight generation defaults; map end-of-turn for GRPO; require full-test or paired evidence for checkpoint selection |
| c01r03 | control | 0.778620 | 8.74 | 0.40 | 0 | 7.73 | yes | no | yes; +0.6 points on full test | 1319 | completed final verification with 1:16 still available | large format-matched SFT; greedy decode about +6.1 points; RFT about +0.6 points | — | save-reload generation config before long training; pass explicit stop ids to offline vLLM; compare continuations on the same larger slice |
| **mean / count** | control | **0.756887** | **8.79** | **0.39** | **0** | **7.67** | **3/3** | **2/3** | **2/3 actually tried RFT** | **1046 mean; 2/3 full** | all deliberate; about 1:09–1:16 unused | large SFT and greedy 3/3; post-SFT gain mixed | not applicable | artifact/termination preflight 3/3; larger/paired selection 3/3 |

### Calibration continuity — excluded from NEW statistics and proposals

| cell | arm | official accuracy | hours used | h to first real train | protocol h | waiting h | greedy shipped | RL used | RFT tried (verdict) | largest eval n | stop reason | top contributors | one protocol change | knowledge to transfer |
|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|---|---|
| p00r10 | protocol, **CALIBRATION** | 0.654284 | 7.77 | 0.25 | 0.07 | 6.87 | no | no | yes; supported | 500 | batch complete, all processes finished, GPU free, deliverable verified, with 2:13 left | termination-format SFT; removing terse raw GSM8K targets; fresh unseen-problem data | corroboration only: decode audit; this cell saw stock sampling but never tested or shipped greedy | — |

p00r10 inspected `do_sample:true` but copied the base generation configuration through to
the artifact [p00r10 calibration, 2026-09-02T15:24:42Z L1562-L1569;
2026-09-02T15:30:49Z L3071-L3072]. It corroborates continuity of the mechanism but supplies
zero NEW votes.

## 2. Ranked explanations of the NEW score difference

### 2.1 Initial SFT recipe strength is the largest remaining score separator — 3/3 controls, 5/5 protocol cells, but consistently higher control post-SFT reads

All eight NEW artifacts shipped greedy, so the old greedy-adoption imbalance cannot explain
this window's −0.0565 gap. What remains visible first is checkpoint quality after the main
format-matched SFT. Controls used 118k–167k broad, exact-format corpora and reported greedy
post-SFT reads of about 0.765, 0.691, and 0.786 before their final post-training stages
[c01r01, 2026-09-03T00:39:41Z L8005-L8010; c01r02,
2026-09-03T06:51:04Z L5527-L5533; c01r03, 2026-09-03T03:27:55Z L5964-L5971].
Protocol main/continued-SFT checkpoints clustered nearer 0.64–0.72 under larger reads; p00r11's
eventual full-set greedy leader was 0.7020, p00r12's final was 0.7210, p00r13's repeated
incumbent mean was 0.7106, and p00r14's pre-continuation full read was 0.6596
[p00r11, 2026-09-03T04:26:09Z L8504-L8523; p00r12,
2026-09-03T04:54:26Z L9837-L9845; p00r13, 2026-09-03T06:04:19Z L8390-L8400;
p00r14, 2026-09-03T04:42:32Z L8019-L8023]. p00r15's selected soup reached 0.734 at
n=500, but its official result was 0.7096 [p00r15, 2026-09-03T05:53:10Z L9627-L9644].

The strongest concise control summary is c01r01's final decomposition:

> “SFT … 3.3% -> 66.5%; deterministic decoding … 76.5%; GRPO … 80.2%.”  
> [c01r01, 2026-09-03T06:45:15Z L16565-L16582]

This is recipe evidence, not a protocol-surface proposal: data choice, optimizer, and method
are bundled, the internal reads use different slices, and c01r02 is a low control despite the
same broad pattern.

### 2.2 Plan shape and use of the remaining horizon — 5/5 protocol cells used serial SFT/RFT-style cards; 2/3 controls reached RL

Protocol cells created 6–9 cards (mean 7.4), normally isolating decode, another SFT slice,
RFT, soup, or confirmation as small comparator-gated steps. They launched no policy-gradient
RL, while all five considered or attempted RFT and four trained it. Controls used broad
narrative/TaskCreate stages at negligible bookkeeping cost and could switch from SFT to RFT
to GRPO, or preserve an incumbent while extending GRPO [c01r01,
2026-09-02T22:03:26Z L1299-L1357; 2026-09-03T05:42:17Z L14958-L14959; c01r02,
2026-09-03T02:27:57Z L3603-L3604; 2026-09-03T02:28:39Z L3697-L3698]. This difference
plausibly changes ceiling and exploration order.

The strongest protocol stop statement makes the framing visible:

> “Both remaining lines of attack are exhausted, so the budget goes to closing the record.”  
> [p00r12, 2026-09-03T04:45:50Z L9464]

It is not sufficient to attribute the arm gap to RL. In the NEW window, the two RL controls
average 0.7460 while the non-RL control is 0.7786. Across the current pool, protocol adoption
is **0/14** and control adoption **7/10**, yet the seven RL controls average about **0.7538**
versus about **0.7602** for the three non-RL controls (the Round 00 record's six-control mean
through c01r01 plus NEW c01r02). RL is therefore the sharpest plan-shape marker, not an
arm-mean explanation. It may still raise a cell: c01r01 and c01r02 report +3.7 and +3.3 points
respectively [c01r01, 2026-09-03T06:45:15Z L16578-L16582; c01r02,
2026-09-03T05:17:18Z L4868-L4869].

### 2.3 Small-n rankings and repeated-run variance distort selection — 5/8 NEW cells show a reversal or materially noisy ranking

Protocol: p00r11's n=150 leader changed at n=500/full; p00r12's n=150 ordering reversed at
n=500; p00r13's nominal n=500 RFT gain vanished under paired/full evidence and identical
full-set reads varied by 0.9 point; p00r14's one-item n=150 loss became a 1.7-point full-set
win [p00r11, 2026-09-03T04:03:02Z L7851-L7870; p00r12,
2026-09-03T03:30:32Z L8249-L8257; p00r13, 2026-09-03T05:16:04Z L7385-L7402;
2026-09-03T06:04:19Z L8390-L8398; p00r14, 2026-09-03T04:42:32Z L8019-L8023].
Control c01r01 adds a direct checkpoint reversal: n≈150 preferred step 200, while n=500
preferred step 300 by two points [c01r01, 2026-09-03T05:27:47Z L14445-L14446;
2026-09-03T05:36:00Z L14647-L14652]. c01r02 reran identical weights at 72.40% and 72.86%,
showing repeat variance without a clean rank inversion [c01r02, 2026-09-03T06:51:04Z
L5537-L5538].

The strongest scientist acknowledgement is:

> “n=150 was ±3.8 points and inverted at n=1319.”  
> [p00r11, 2026-09-03T06:07:08Z L9852]

This mechanism does not mechanically lower one arm—several protocol cells repaired their own
verdicts—but it consumes cards and makes sub-one-point post-SFT claims unreliable. It confirms
frozen direction C and adds a repeated-read amendment.

### 2.4 Preventable artifact and rollout failures consumed scientifically useful time — 6/8 NEW cells, both arms

Generation-config save validation hit p00r11 (~0.35 h), p00r14 (1.30 h), c01r01 (~0.45 h),
and c01r03 (76 minutes); c01r02 avoided a save crash but still hit RL prompt/termination bring-up
[p00r11, 2026-09-03T05:22:22Z L9025; p00r14, 2026-09-03T06:02:31Z L8789-L8808;
c01r01, 2026-09-03T01:07:33Z L9060-L9061; c01r03,
2026-09-03T02:03:35Z L5658-L5694; c01r02, 2026-09-03T02:32:55Z L4021-L4022].
The Round 00 rolling record called c01r01 the **seventh** cell with the greedy-parent save
failure; c01r03 was the sixth. p00r11 and p00r14 show the same signature in their reviewer
reports, but this evidence document does not retroactively renumber the ledger's frozen
seven-cell count. They are supplementary NEW support for the same root cause. This confirms
frozen D; it is not a new direction.

Offline vLLM traps were also bilateral. p00r11 lost time to missing explicit stop IDs;
p00r12 had orphan/seed/parser/non-termination failures; p00r14's n>1 stop-string behavior
projected 3.8 h until explicit token 106; c01r03 spent about 1.9 h on two bad termination
passes before explicit `[1,106]`; c01r02 hit a max-prompt-length crash after repairing EOS
[p00r11, 2026-09-03T01:30:05Z L5557-L5558; p00r12,
2026-09-03T02:18:13Z L6825-L6834; p00r14, 2026-09-03T01:47:14Z L7225-L7227;
c01r03, 2026-09-03T04:50:52Z L6470-L6478; 2026-09-03T05:36:37Z L6664-L6672;
c01r02, 2026-09-03T02:48:09Z L4251-L4252]. This confirms frozen B as an hours-saving
direction, not an explanation for the score gap.

### 2.5 Greedy decoding remains the strongest mechanism, but no longer separates arms in this window

All 8/8 NEW cells shipped greedy. Identical-weight gains were p00r11 +6.1, p00r12 +9.3,
p00r13 +5.4, p00r14 +20, p00r15 +6, c01r01 +10, c01r02 +8.7, and c01r03 +6.1 points
[p00r11, 2026-09-03T04:26:09Z L8504-L8523; p00r12,
2026-09-03T01:33:06Z L5628-L5648; p00r13, 2026-09-03T01:52:55Z L6155-L6183;
p00r14, 2026-09-02T23:58:16Z L6351-L6365; p00r15,
2026-09-02T23:07:45Z L5328-L5345; c01r01, 2026-09-03T00:39:41Z L8005-L8010;
c01r02, 2026-09-03T00:09:56Z L2850-L2851; c01r03,
2026-09-03T00:51:42Z L5560-L5566]. This is emphatic confirmation of frozen A and explains
continuity from calibration p00r10, but contributes approximately zero to the NEW arm
difference because adoption is balanced.

## 3. What the protocol cost versus control

| item | protocol NEW (n=5) | control NEW (n=3) | evidence-bearing interpretation |
|---|---|---|---|
| hours to first real train | mean **0.322** (0.27–0.43) | mean **0.393** (0.36–0.42) | Despite cards/bootstrap, protocol launched about 4.3 minutes earlier on average; direct ceremony did not delay first training in this window. Header timelines; p00r14's earlier 22:12 call was a smoke, real launch 22:26 [p00r14, 2026-09-02T22:26:32Z L4511]. |
| direct protocol work | mean **0.136 h** (0.01–0.25), about 8.2 min | 0; self-imposed TaskCreate/narrative plans took seconds | Small relative to ten hours and usually overlapped running work [p00r11, 2026-09-03T01:04:10Z L5110; 2026-09-03T01:54:30Z L5837; c01r03, 2026-09-02T22:04:06Z L1238-L1295]. |
| total / waiting hours | 8.05 / 6.52 mean | 8.79 / 7.67 mean | Controls used 0.74 h more session and waited 1.16 h more; this is mostly longer training/eval, not idle absence. It can contribute opportunity, but cell recipes differ. |
| plan shape | 6–9 locked cards, mean 7.4; usually one small intervention plus comparator/falsifier; 3.2 real training launches mean | broad 5-stage task list or narrative; incumbent-first branching; roughly SFT→RFT/GRPO→extension | The protocol made negative results and fallback rules explicit; controls changed methods within a broad bucket [p00r14, 2026-09-03T06:03:18Z L8865-L8874; c01r01, 2026-09-03T05:36:19Z L14661-L14675]. |
| RL / RFT | RL 0/5; RFT considered 5/5, trained 4/5; robustly positive only p00r14, mixed/negative elsewhere | RL 2/3; RFT actually tried 2/3; c01r03's +0.6 not decisive | Plan-shape difference is real; method benefit is mixed and does not explain pool means. |
| stop reasons | all deliberate; 0:59–3:05 left; lines declared exhausted, contradicted, or verified | all deliberate completion; tightly grouped 1:09–1:16 left | Protocol left more time on average, driven by p00r12. Yet p00r14 used 9.01 h and remained the lowest protocol cell, so unused time alone is not the score mechanism. |

What the format bought is also visible: p00r12 and p00r14 reversed misleading n=150
verdicts with explicit larger-n cards; p00r13 refused a nominal RFT gain after paired/full
evidence; p00r14 retained a fallback before a budget-capped rerun [p00r12,
2026-09-03T03:30:32Z L8249-L8257; p00r13, 2026-09-03T05:16:04Z L7385-L7402;
p00r14, 2026-09-03T06:03:18Z L8865-L8874]. Direct cost is minutes. The unresolved cost is
framing: serial small steps, small initial n, and method choice inside per-card remaining-time
arithmetic.

## 4. Rolling-direction reconciliation

- **A, greedy mechanism:** strongly confirmed by 8/8 NEW identical-weight gains, but balanced
  adoption means it no longer explains this window's arm gap. Calibration p00r10 is continuity
  only.
- **B, offline vLLM traps:** confirmed in at least p00r11, p00r12, p00r14, c01r02, and c01r03;
  bilateral hours loss, not an arm mechanism.
- **C, small-n/repeated-run uncertainty:** confirmed or refined by p00r11, p00r12, p00r13,
  p00r14, c01r01, and c01r02. Larger n alone is insufficient for deltas around one point when
  identical full reads move 0.4–0.9 point; paired or repeated evidence is the refinement.
- **D, greedy-parent save validation:** the rolling record's seven-cell finding remains valid;
  this window includes c01r01 as the seventh cited cell and adds direct detail from p00r11,
  p00r14, and c01r03. Frozen D already targets the root cause.
- **E, wait on process rather than clock:** no clean NEW repeated blind-wait evidence rises to
  a proposal source. Keep frozen E unchanged; do not infer absence from this small window.
- **H, eval-only overrides:** p00r11 and p00r13 each used four `data_files_exist` overrides on
  non-training cards, bringing the rolling direct count to ten across four cells
  [p00r11, 2026-09-03T04:08:39Z L8205; p00r13, 2026-09-03T05:37:24Z L8190]. This confirms
  frozen H.
- **G, TRL EOS zero-gradient:** the ledger froze the observation at **6/6** RL controls through
  c01r01; c01r02 independently hit the same end-of-turn/truncation zero-gradient defect before
  corrected GRPO [c01r02, 2026-09-03T02:32:55Z L4021-L4022;
  2026-09-03T02:35:43Z L4115-L4116]. Thus current evidence is **7/7 RL controls**, not a
  contradiction of the named six-of-six rolling checkpoint. A pitfall can remove bring-up loss;
  it must not prescribe RL.
- **Protocol RL adoption:** current clean pools are protocol **0/14** versus control **7/10**,
  while RL/non-RL control means are about **0.7538/0.7602**. Adoption is not the score-gap
  estimator.
- **Soup:** p00r15's pre-registered three-way soup reached 0.720 and adaptive four-way soup
  0.735@200/0.734@500, but the best-of-four choice was explicitly optimistic; p00r11's merge,
  p00r13's 0.5/0.5 soup, and c01r02's late average were flat or worse
  [p00r15, 2026-09-03T05:53:10Z L9625-L9644; p00r11,
  2026-09-03T04:03:02Z L7851-L7870; p00r13, 2026-09-03T05:26:06Z L7684-L7698;
  c01r02, 2026-09-03T06:46:59Z L5298-L5299]. Mixed recipe evidence; no protocol direction.
- **I, stop-token double append:** p00r12 stored `target = generation + STOP` and its encoder
  appended STOP again; RFT never trained, so harm is latent [Round 00 record addendum,
  p00r12 L4796 and L3120-L3134]. p00r14 repeatedly overrode a raw-target check after verifying
  500/500 rendered rows ended in token 106 [p00r14, 2026-09-02T22:26:26Z L4495-L4505], and
  p00r15's preflight initially read the wrong raw field before rendered verification
  [p00r15, 2026-09-02T22:15:58Z L3906-L3916; 2026-09-02T22:16:33Z L3989-L3997]. These are
  three NEW manifestations of ambiguous terminator ownership; only p00r12 demonstrates a
  potential double append, and none demonstrates a score loss.

## 5. Candidate proposals — evidence only, no decision

Each item changes exactly one allowed surface, uses at least two NEW cells, and retains the
standing score guardrail: four-cell screen mean not below the concurrent baseline pool by more
than 0.03. A/B/C/D/E/H are already frozen; confirmations do not reopen their wording unless an
amendment is explicitly named.

| status | candidate; exactly one allowed-surface item | NEW source cells | target metric for a 4-cell screen | guardrail |
|---|---|---|---|---|
| **Confirmation of frozen A; no wording decision** | `pitfalls.yaml`: retain A's grader-observable decode audit requiring an explicit measured generation-config choice before first post-training comparison | p00r14 (+20 identical weights), p00r15 (+6); independently p00r11–13 and all three controls | ≥3/4 ship greedy or another same-weight measured configuration, and ≥3/4 result records cite the effective grader observable | block accuracy ≥ baseline −0.03; no increase in run-on/missing-answer rate |
| **Amendment candidate to frozen C; decision deferred** | `SKILL.md` wording only: when the claimed delta is ≤1 percentage point or one marginal SE, require paired item counts or a repeated read, even if n≥500 | p00r12 (paired reversal/near tie), p00r13 (identical full-read sd 0.0036); also p00r11 and p00r14 | 3/4 close every sub-threshold selection with paired or repeated evidence; zero later reversal of a selected sub-threshold winner | block accuracy ≥ baseline −0.03; evaluation time ≤1.0 h/cell above drift and no forced full rerun for clearly large deltas |
| **Confirmation of frozen H; no wording decision** | preflight check with a test: `data_files_exist` applies only to training-family cards, so measurement-only cards do not require `setup.data` | p00r11 and p00r13, four direct overrides each | zero fabricated data entries and zero `data_files_exist` overrides on non-training cards in 4/4; `fields_filled=1.0` | block accuracy ≥ baseline −0.03; training-family data checks remain unchanged |
| **Existing queued direction I; not frozen and not genuinely new** | optional card field `stop_token.appended_by` with enum `data` or `script`, checked against one rendered/tokenized example rather than the raw source field | p00r12 (latent double append), p00r14 (raw-check overrides despite 500/500 rendered pass), p00r15 (wrong-raw-field preflight repaired by rendered verification) | 4/4 cards declare one owner; zero double terminators, raw-field false failures, or ownership overrides on exercised SFT/RFT paths | block accuracy ≥ baseline −0.03; ≥99% sampled targets end with exactly one intended terminator and answer-format rate does not regress |

Existing directions not proposed anew: B and D receive strong confirmation above and already
have frozen screens; E has no repeated NEW source; G is an existing queued direction rather
than a new proposal. **No genuinely new direction passes the two-NEW-cell threshold in this
window**: A/C/H are frozen confirmations or amendments, and I already exists in the ledger.
No candidate here tells the scientist to run RL, RFT, soup, or a particular data recipe.

### What the protocol should stop doing, conditional on the frozen screens

- Stop treating an n=150/200 marginal result as a terminal falsifier when a larger or paired
  read is affordable (p00r11, p00r12, p00r14).
- Stop training from a parent directory carrying a validation-invalid greedy config; put the
  grader config in `final_model` or neutralize it before the first save (p00r11, p00r14,
  c01r01, c01r03).
- Stop requiring a fake data path on measurement-only cards (p00r11, p00r13).
- Stop assigning stop-token ownership implicitly between raw data, builder, and encoder
  (p00r12, p00r14, p00r15).

These are evidence-backed descriptions of behavior to remove, not adoption decisions.

## 6. Open questions for the next wave

1. Once A makes greedy/measured decode universal, does a ≥0.03 residual remain against the
   drift pair? That is the clean test of decision framing; this NEW window cannot answer it
   because both arms self-adopted greedy.
2. Does C's frozen larger-n wording prevent wrong selections, and does the proposed paired/
   repeated amendment add value beyond n≥500 without consuming meaningful training time?
3. After B, D, and G preflights remove sampler/save/zero-gradient bring-up failures, do protocol
   scientists consider RL earlier without the protocol prescribing a method? Compare adoption,
   time-to-first-healthy-gradient, and score separately.
4. Is the protocol's 0.74 h lower session use caused by genuine exhaustion, over-conservative
   per-card budgeting, or simply faster recipes? Record runnable alternatives and actual eval
   durations at stop rather than rewarding budget consumption itself.
5. Does pre-registered soup reproduce p00r15's gain on ≥500 paired items? Until then the negative/
   flat p00r11, p00r13, and c01r02 evidence prevents a general soup claim.
6. Does direction I eliminate check friction and latent double terminators on an exercised RFT
   path? p00r12 never trained the affected corpus, so present evidence is structural, not outcome
   evidence.
7. Why do control main-SFT checkpoints tend to start higher in this window: data volume/style,
   optimization, broader up-front planning, or ordinary cell variance? This is recipe analysis,
   not a license to put a preferred recipe into the protocol.

The next decision should be made from the already-frozen screen results plus these trace metrics,
not from this document alone.
