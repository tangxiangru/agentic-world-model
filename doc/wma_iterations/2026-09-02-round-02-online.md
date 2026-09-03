# WMA online Round 02 — four single-edit candidates

Status: lock-gated v2 launched; all 24 replacement cells safely PENDING; no
candidate result read.

Amendment (2026-09-03 01:4x UTC): the first candidate receipts are withdrawn
before start because their public checkout used the obsolete non-blocking WMA
review. They are replaced by six lock-gated v2 manifests: the same four
candidate skills plus a same-wave v0.2 baseline and no-WMA control, all on
public `awm.sha=ae4672487cc16f1d75921dcaab85498d7adc938e`.

## Frozen basis

Round 01 completed 8/8 validator-clean cells per arm. WMA v0.2 scored 0.7565
versus control 0.7523; the paired delta was +0.0043 with sampling SE 0.0136
and 5/8 positive, so no reliable PTB benefit was established. The terminal
v0.2 ledger had 22 verdicts, 15 scored, 7 leak-suspected, L0/L1 0.933,
L2 coverage 0.636 at n=11, width/noise 6.1667, and L3 0.900.

Base WMA skill hash: `176f0a464986`. Public scientist checkout:
`e8a8599bce74df232e20e2fd3aa1226e7e04e999`. Runtime/measurement code is
identical across candidates; only the archived `skills/wma/` differs.

The D/E switch was resolved before candidate text: C5 was suggested on both
C5-labelled terminal verdicts, but neither result card evaluated the suggested
saved checkpoints (0/2 uptake), so D is tested and E is not.

## Candidate map

| candidate | WMA commit | skill hash | only policy difference |
|---|---|---|---|
| A | `f9012988ef0c1f1abd1888445cba04d62abf9eba` | `b6cc80891de4` | first intervention after a diagnosed format floor anchors L2 to measured floor and same-checkpoint capability |
| A+B | `04e71a9c6cbfc96fc22bd88f3aa9858c18969fed` | `46f66c240380` | A plus evidence required when L2 width exceeds 3× noise |
| C | `9d24834324af4e7f0219d9ecfa274d12490f566e` | `44d687f73c70` | an L0/L1 `no` needs a level-changing probe or confidence ≤0.5 and `unprobed` evidence |
| D | `8b812aba30b5f2fe265e20125ac11ce0a245348b` | `6e0d02c5afad` | C3/C4 with only a final checkpoint defers L3 pending a costed save/eval plan |

The branch head after this record returns to byte-identical v0.2. Candidate
commits remain reachable in the linear history and are immutable manifest
inputs.

## Manifests and cells

- A: `wma-gsm8k-gemma4b-high-r02-a-format-floor-x4`, `w06r01..04`.
- A+B: `wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4`, `w07r01..04`.
- C: `wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4`, `w08r01..04`.
- D: `wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4`, `w09r01..04`.

Each manifest has one setting and four repeats, no pilot, one H100 per cell,
the same public protocol/scientist/model/evaluation/judges/history as Round 01,
and the held-route gate for `slurm2-a3nodesetondem-[2-3]`. No new control arm
is launched; Round 01 core and w04/c04 provide the frozen v0.2/control views.

## Launch (2026-09-02 23:33–23:35 UTC)

Reconcile submitted all four manifests asynchronously after a clean-tree and
local-contract check:

- A: jobs `90861..90864`.
- A+B: jobs `90865..90868`.
- C: jobs `90870..90873`.
- D: jobs `90874..90877`.

The held-route gate passed before every release. A post-release `scontrol`
audit found all 16 jobs `PENDING` with
`ReqNodeList=slurm2-a3nodesetondem-[2-3]`. Owned allocation remained 16/16 and
safe PENDING rose from 25 to 41. No cancellation or cross-subqueue action was
part of the launch. Receipts are under the matching `results/ptb/<batch>/`
directories.

At 00:36 UTC the user-directed repeats-4 trim was applied to the Slurm-requeued
v2 inventory. To preserve the strict 24-job operational guard, the operator
kept the lowest-index extra pair `w02r05/c02r05` as sensitivity-only and
cancelled 16 exact still-PENDING receipt jobs: `w02/c02 r06..08`,
`w03 r01..03,r05..08`, and `c03r01..03`. Post-action owned allocation was
16/16 and safe PENDING was 25, including all 16 Round 02 candidates. No RUNNING
job was touched; r05 is excluded from the all-24 primary cohort.

## Readout and falsification

- A primary: first-format-fix L2 coverage and the fixed manual exp-01 absolute
  audit. Falsify on wrong floor diagnosis, redundancy, or worse non-floor
  coverage.
- A+B primary versus A: lower width/noise without coverage falling beyond A's
  own spread. Falsify if coverage falls.
- C primary: lower false-`no` rate with L0/L1 recall no worse by >0.05 and
  cost ≤1.5× v0.2. Falsify on recall or cost.
- D primary: checkpoint-precondition appearance and scientist uptake on C3/C4;
  PTB must not degrade beyond v0.2 spread. Falsify if it only adds deferral
  without checkpoint-selection behaviour.

Every candidate requires zero leak-suspected verdicts, accepted-verdict cost
≤1.5× v0.2, and PTB no worse than v0.2 beyond its observed spread. At most one
edit is promoted. A second passing edit is retested stacked on the promoted
baseline in the next round; candidates are never pooled post hoc.

## Verdict-in-lock amendment

Round 01 showed that 21/22 delivered verdicts arrived after the launch under
the old advisory background step. Commit `25a064c` therefore makes the verdict
part of `awm exp_protocol lock`; `ae46724` additionally prevents a re-lock from
mistaking an old verdict file for the new response. A WMA-attached scientist
may prepare while waiting, but cannot start the card's command before lock
returns. The control records `not_attached` and returns without a verdict.

This behavior changes the treatment and invalidates comparison of the original
Round 02 receipts against old w04/c04. Before any original candidate job
started, queue intent was changed to cancel those 16 exact cells and replace
them with:

- v0.2 baseline `w10r01..04`, private/public SHA `ae46724`;
- control `c10r01..04`, identical public checkout and no sidecar;
- A/A+B/C/D v2, reusing `w06..w09 r01..04` with the same immutable candidate
  private SHAs and public SHA `ae46724`.

All six v2 manifests have four repeats and otherwise retain the frozen Round 02
contract. The original candidate receipts never contribute to a result or
denominator.

At 01:47–01:48 UTC all 16 original jobs (`90861..90868`, `90870..90877`) were
rechecked PENDING and cancelled by exact tracked receipt. None had started and
no result was produced. Safe PENDING temporarily remained 9, above the hard
floor of 8, while the committed v2 replacements waited for a clean-tree submit.

## Protocol readout for the lock-gated wave (planner, 2026-09-03 01:5x UTC)

The `w10 − c10` pair reads the blocking lock itself, with skill v0.2 unchanged;
the candidates read against `w10`. Preregistered, from the Round 01 root-cause
analysis (`tools/wma-rca/`; PR #23 window comment):

1. Compliance: `wma.state` distribution in the lock files; verdict-before-launch
   rate from `solve_parsed` (Round 01: 1/22); wait per verdict; timeouts and
   `--no-wma-wait` reasons. Below 0.8 verdict-before-launch → the mechanism did
   not take; fix the harness before reading any effect.
2. Consumption: review coverage per cell (Round 01: 33/59 cards, endgame 0),
   uptake classes on the delivered verdicts (Round 01: 4/22 with an action by
   hand count), intermediate checkpoints scored after a C5 suggestion (Round 01:
   2/8 cells, the control arm's unprompted rate).
3. PTB: paired `w10 − c10` against Round 01's +0.004 ± 0.014; n = 4 gives a
   direction and a guard (wma not below ctl beyond ctl's spread), not a verdict.
4. Cost of the protocol: total wait per cell (GPU idle) and cards per cell;
   > 1.5 h waiting or > 30 % fewer cards → recorded as the price and fed into
   the next candidate (e.g. exempt cards with `planned_h < 0.5`).

Falsification: compliant but no rise in coverage, uptake or checkpoint scoring
over Round 01 → being in the loop is not enough; the next round moves to the L3
policy (`yes` 22/22 in Round 01) and to harness-enforced precondition answers.
Non-compliant → harness first. The skill candidates keep their own readouts
above; a candidate is compared only with `w10` cells that passed the compliance
check.

## Lock-gated v2 launch (2026-09-03 01:49–01:51 UTC)

At 01:49–01:51 UTC reconcile submitted the lock-gated v2 design:

- v0.2 baseline: `90982..90985`;
- no-WMA control: `90998..91001`;
- A: `91002..91005`;
- A+B: `91007..91010`;
- C: `91014,91021..91023`;
- D: `91024..91027`.

The held-route gate and post-release audit both passed for every cell: 24/24
were PENDING with `ReqNodeList=slurm2-a3nodesetondem-[2-3]`. Owned allocation
remained 16/16 and safe PENDING rose from 9 to 33. Only these v2 receipts enter
the Round 02 denominator.

## Baseline/control extension preregistration (2026-09-03 18:0x UTC)

At the hourly completion event, 16 new validator-clean cells became available:
five old w02 WMA, five matched c02 control, all four w10 lock-gated baselines,
and c10r01..03. Opus 5 UltraCode analysis was started asynchronously from the
frozen event `20260903T180444Z-79c8d29810`; no score or candidate readout was
used to select this refill. Meanwhile safe PENDING had fallen from 25 to 9,
one above the user's hard floor.

Two matched four-cell extensions are therefore preregistered immediately:

- `wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1`, `w10r05..08`;
- `wma-gsm8k-gemma4b-high-r02-ctl-ext-x4-v1`, `c10r05..08`.

They freeze the exact original w10/c10 contract: public protocol and v0.2 WMA
at `ae46724`, Opus 5 high scientist and official/research judges, Gemma 4B,
ten hours and no pilot. The extension changes no treatment; it supplies the
second four replicates needed for the `w10 − c10` gate comparison to reach the
online loop's eight-clean-cells-per-arm evidence window. Replicate IDs continue
at 5–8 so the combined cohort is explicit. The primary analysis remains
compliance first, then paired PTB, decision changes and transcript-derived
gate cost. A–F retain their frozen denominators and are not pooled with this
baseline/control contrast.

Cell IDs continue at r05–r08 to make the combined cohort unambiguous; the
manifest-local `replicate` ordinals reset to 1–4 as required by the frozen
replication schema.

This is the documented exception to the default preference for distinct
four-cell settings: without it, the lock-gated mechanism itself has only n=4
per arm and cannot satisfy the formal evidence gate. Submission is expected to
raise safe PENDING from 9 to 17; it is queue insurance as well as necessary
measurement, not a candidate promotion.

## In-flight gate read (planner, 2026-09-03 09:0x UTC)

Source: the operator's 08:44 peeks of the first seven lock-gated cells
(`w10r01..04`, `c10r01..03`), 200 transcript lines each, read with
`tools/wma-rca/inflight_gate.py`. A window, not the cells.

- **The gate holds.** Eight waits observed, 5.0–8.5 min each (median ≈ 6.3),
  all delivered; no `failed`, no `timeout`, no `--no-wma-wait`. Every training
  launch in the windows came after the verdict line: +6 s (w10r01), +11 s
  (w10r02), +42 s and +12 s (w10r03). Round 01 was 1/22 verdicts before launch.
- **Preconditions are answered before the launch, in 4/4 WMA cells.** w10r01's
  first exp-02 verdict said `L0_runs=no@0.68` (OOM at the real max-seq-len);
  the scientist edited the card, relocked, got `L0_runs=yes@0.86`, launched.
  w10r02 relocked with the tier-1 answers written into the card; w10r03 edited
  the card's `alternatives_rejected` and relocked; w10r04 ran the suggested
  `--limit 20 --max-connections 16` smoke (2 min) before the 150-item baseline.
- **The price: one relock per card so far, each a second full wait.** Card
  exp-02 cost 17 min of gate in w10r02 (6.5 + 5.0) and ≈ 25 min in w10r01
  (8.5 + 8.0). Scientists prepare data during the wait, but the training GPU is
  idle. At ~8 cards per cell this is 1–2 h of the 10 h budget — above the 1.5 h
  line preregistered as "the price goes into the next candidate".
- **Two scientists background the lock** (`nohup awm exp_protocol lock … > log &`,
  w10r04; a Claude Code background task, w10r03) and poll it. Both still waited
  for the verdict before launching; that is the intended shape ("prepare
  meanwhile, do not start"). But the backgrounded log stayed empty for the whole
  wait: block-buffered stdout (fixed in `7a75d84`, not in the pinned
  `ae46724`; rides the next public bump). w10r04 lost ~8 min of turns to it.
- **L3 is still `yes` 8/8**, including the verdict with `L0_runs=no@0.68` —
  the inconsistency Round 03 candidate E targets.
- Control cells (`c10`) lock and launch at once, as intended (`not_attached`).

Nothing here changes the preregistered readout; the compliance criterion
(verdict-before-launch ≥ 0.8) looks likely to pass. What the harvest must add:
the `lock.wma.state` distribution from the lock files, wait totals per cell,
relock counts per card, and whether the endgame cards get locked at all.
