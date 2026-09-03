# Trace review window 03 — local Claude — 2026-09-03

Status: **reviewers running; synthesis and scientific decisions pending**. This is a frozen eight-new-clean window, not a claim that Round 01 or Round 02 is complete.

## Frozen evidence and sessions

Trigger: completion monitor reported 9 terminal jobs at 18:02:44 UTC. Harvest commit `9c7596a` contains eight eligible, PTB-validator-complete, judge-clean cells and the separate failed p00r16. Independent manifest-driven validation confirmed the eight. Window 02's counter had reset at 08:12 UTC.

| group | NEW cells | calibration only | local Claude session |
|---|---|---|---|
| control A | c01r04, c01r05, c01r06 | none | `afa3cfee-8276-4292-96ab-bbe26d65704c` |
| control B | c01r07, c01r08 | c01r03 | `a165f583-9c88-420e-b83a-d2212dde2d35` |
| guard | g01r01, g01r02, g01s04 | none | `9081a22d-3a4d-455e-89ab-d3164d577a76` |

All sessions use `claude-opus-5[1m]`, `--effort max`, `--background`, plan permissions and Read/Grep/Glob/Bash tools. At 18:35 UTC each session was independently visible as `working`, and logs showed file/trace reads. The calibration cell is excluded from every NEW count and aggregate. Facts and timelines were generated before dispatch with the two committed reader tools. Full job/result/bundle paths and trigger are in [launch.json](trace-reviews/window03-local/launch.json); scratch is `/tmp/exp-protocol-window03.XVcSr0`.

| cell | variant | job | official accuracy |
|---|---|---:|---:|
| c01r04 | no-protocol control | 90494 | 0.792267 |
| c01r05 | no-protocol control | 90495 | 0.724033 |
| c01r06 | no-protocol control | 90496 | 0.756634 |
| c01r07 | no-protocol control | 90497 | 0.734647 |
| c01r08 | no-protocol control | 90498 | 0.778620 |
| g01r01 | session guard | 90647 | 0.710387 |
| g01r02 | session guard | 90648 | 0.777862 |
| g01s04 | session guard | 90794 | 0.735406 |

NEW control mean (n=5) = 0.757240; NEW guard mean (n=3) = 0.741218. These are descriptive interim values, not a promotion claim. g01r01/02 are the valid restarted strict-site attempts; archived pre-requeue spillover traces are not merged with them. The v3 baseline, guard and no-protocol control remain distinct variants.

The four later strict guard completions g01s01/g01s03/g01s06/g01s08 were harvested in `66ebd39`. They are outside this frozen window, pending a guard-block addendum; they do not delay or silently expand the dispatched eight-cell window.

## Operational findings during dispatch

The first CLI invocation omitted the `--` separator after variadic `--tools`, so sessions a2e5dbbf/1ec7de0b/188f015b started idle with no dispatched prompt. The corrected command created the three working sessions above. Restricted-shell status/log queries then misreported access to the host daemon; checking the same IDs in the unrestricted context proved it was alive. The initial idle sessions were stopped without deleting their conversations. No active reviewer was restarted based solely on the sandbox observation.

## Queue boundary

At 18:38 UTC the external queue had returned to 16/16 GPU allocation, but all Round 02 user holds had also been externally removed around 18:30:22. Neither the committed queue (`want: held`) nor the eight frozen Round 02 receipts (`state: held`) contained a matching release record. Strict guard was still only 5/8 complete, and the reservation still spanned eleven nodes.

At 18:42:47 the operator restored user holds on exactly jobs 91046–91073 after verifying tracked receipt membership, full job names, PENDING state, zero runtime/restarts, and frozen ReqNodeList on nodes 0–1. All 28 then read `PENDING(JobHeldUser)`. No RUNNING job was modified or cancelled. Audit: [held restore](../../results/ptb/held-restore-20260903T184247Z.json). The external release actor/authority is unresolved and user clarification was requested. This is restoration of the existing gate, not a scientific rejection of the screens.

## Synthesis and planner decision

### Planner's three-card read (guard variant)

- [g01r01 exp-09](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r01/task/memory/cards/exp-09.yaml): a non-training soup/measurement card still carries a training-data entry. It compares four full-1319 repeats and also optimizes a first-150 criterion because the scientist says the grading subset is unknown. This is an H exposure and a question about developer defaults versus official scoring, not proof of a new causal score mechanism.
- [g01r02 exp-07](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r02/task/memory/cards/exp-07.yaml): the second fitted-parent RFT runs 0.32 h with loss 0.205→0.207, then loses 3.3 points at n=150 and 1.5 on the fixed 200-item probe. This is a new guard-arm P1 signature; one cell does not satisfy the E-replacement rule's two-cell/full-block requirement.
- [g01s04 exp-05](../../results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2/g01s04/task/memory/cards/exp-05.yaml): the n=800 comparison is 600 versus 595 correct, with 52 fixed and 47 broken (McNemar z=0.40). The card explicitly labels the evidence inconclusive even while shipping the nominally better checkpoint. This is counterevidence to a blanket claim that guard scientists ignore uncertainty; it also records a non-training data placeholder and repeat-read variance.

These reads were completed before synthesis. They are cross-checks for the reviewer findings, not decisions to rewrite or withdraw a candidate. The control arm has no experiment cards by design.

Card collection for the three NEW guard cells is saved in [guard-collect.csv](trace-reviews/window03-local/guard-collect.csv): 21 cards all locked and closed, zero locked-open cards, two relocks, zero overrides, fields_filled=1.0, and 3.92 self-attributed pitfall hours. Trace review still determines the mechanism and actual waiting losses.

The excluded baseline failure has a separate [p00r16 review](2026-09-03-p00r16-scorer-failure.md); its n=500 score is not an official result.

Pending the three reviewer outputs. Required before the window is closed: save one report per cell; launch a separate local Opus max synthesis; read its full output, best/worst reports per variant and three guard cards; adjudicate candidate observables and whole-block pending triage; update the ledger. E's saturation condition is evaluated on the prescribed full strict-guard block, not on the three guard cells in this window. No AIME2025 work or baseline promotion is authorized by this interim report.
