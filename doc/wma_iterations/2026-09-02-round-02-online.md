# WMA online Round 02 — four single-edit candidates

Status: launched; all 16 candidate cells safely PENDING; no candidate result read.

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
