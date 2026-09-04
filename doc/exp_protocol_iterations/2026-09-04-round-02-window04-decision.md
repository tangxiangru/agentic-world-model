# Round02 planning — Window04 decision,2026-09-04

This is the planner's closed analysis window and prelaunch queue decision, **not a completed Round02 GPU screen or a promotion**. All14 NEW reports, five focused reviewer audits, two complete local Opus syntheses (507 and532 lines), and the planner's prefix audit have been read. No third synthesis is required merely to repeat unresolved questions.

## Variants

| label | frozen AWM / protocol tree | role in this window |
|---|---|---|
| session guard |4ae3d87 /189319d6|6 NEW old-cohort cells; current guard comparator lineage|
| protocol-free control |eaf5091 /no protocol path or setup flag|6 NEW strict controls|
| protocol v3 baseline |eaf5091 /08674f2c|2 NEW strict baseline cells|

Guard and v3 are different variants; do not pool them as a single-variant effect. Candidate construction/next screens stay baseline-relative to guard drift2f64581, whose six shipped paths are still the host baseline. Historical p00r02 calibration and the2 later clean tail cells are excluded from the14.

## Cells

GSM8K · google/gemma-3-4b-pt revisioncc012e0a6d0787b4adcc0fa2c4da74402494554d · scientist claude-opus-5[1m],high,1M ·10h · observed NEW repeats6/6/2. Held-out task this window:none. Actual job/receipt/manifest/spec/result paths are frozen in [roster](trace-reviews/window04-local/roster.json). Only official receipt-backed validator-complete eligible judge-clean results enter these means.

## Results

Raw bookkeeping (controls have no cards; their zero pitfall totals are not measured absence of waste):

```csv
session,accuracy,hours_used,n_cards,n_unreadable,n_closed,n_locked,n_locked_open,n_relocked,n_overrides,preflight_fail,pitfalls_hit,pitfalls_cost_h,adopted,fields_filled
g01r03/task,0.8278999241849886,8.44,8,0,8,8,0,2,2,2,6,0.8500000000000001,5,1.0
g01r04/task,0.690674753601213,7.76,4,0,4,4,0,0,4,4,7,2.28,3,1.0
g01r05/task,0.7611827141774071,7.92,7,0,7,7,0,1,0,0,4,1.1,3,1.0
g01r06/task,0.7081122062168309,7.74,7,0,7,7,0,2,0,0,7,2.3,3,1.0
g01r07/task,0.7217589082638363,8.37,8,0,8,8,0,1,0,0,2,0.1,5,1.0
g01r08/task,0.7028051554207733,7.45,8,0,8,8,0,1,0,0,6,1.0,5,1.0
c01s01/task,0.7892342683851402,8.76,0,0,0,0,0,0,0,0,0,0.0,0,
c01s02/task,0.7816527672479151,8.63,0,0,0,0,0,0,0,0,0,0.0,0,
c01s03/task,0.7202426080363912,8.41,0,0,0,0,0,0,0,0,0,0.0,0,
c01s04/task,0.7558756633813495,8.62,0,0,0,0,0,0,0,0,0,0.0,0,
c01s06/task,0.7414708112206216,7.89,0,0,0,0,0,0,0,0,0,0.0,0,
c01s07/task,0.7202426080363912,8.19,0,0,0,0,0,0,0,0,0,0.0,0,
p00s01/task,0.7702805155420773,7.89,7,0,7,7,0,0,0,0,5,0.96,6,1.0
p00s02/task,0.77710386656558,7.81,7,0,7,7,0,2,0,0,7,0.9500000000000001,4,1.0
```

| variant | accuracy mean(min–max) | raw pitfalls_cost_h sum | n_locked_open sum | fields_filled |
|---|---|---:|---:|---|
| guard6 |0.73540561(0.69067475–0.82789992)|7.63|0|1.0|
| control6 |0.75145312(0.72024261–0.78923427)|not comparable:card-derived0|0 cards|not applicable|
| baseline2 |0.77369219(0.77028052–0.77710387)|1.91|0|1.0|

No arm effect or equivalence is established. g01r03's1092/1319=82.79% is a single-run record, not a promotion claim.56 protocol cards closed;9 distinct re-locked cards and12 relock events. Scalar means, direct tool time, authoring intervals and post-exit idle are different measurements.

Cards independently read: guard g01r03 exp04(data scale,additional tokens/steps not isolated),exp07(soup and rejected-checkpoint reuse),exp08(packaging/comparator-subset defect),plus worst-cell g01r04 exp04(mixed RFT from base,not its sampling parent). Baseline p00s01 exp04(vote-format RFT),exp06(disagreeing held-out/official samples and save failure),p00s02 exp05(prompt-distribution repair). Controls have no cards; c01s01 RESULTS.md,c01s03 README.md and all control reports were read. Best/worst per variant were covered, including the tied low controls.

## Trace review

[Revised synthesis](trace-reviews/window04-local/synthesis.revision2.md), [initial unaltered synthesis](trace-reviews/window04-local/synthesis.initial.raw.md), and [planner corrections](trace-reviews/window04-local/planner-corrections.md). Local sessions ea5ac0e9-f5e4-4ae7-a9c6-cc328a80ef70 and its revision copy a1e293bb-7b8c-4e8a-ae0b-d305c22d47e3 both completed and were stopped after delivery/idle verification. AnalystCLI2.1.260, initial reviewerCLI2.1.259 and frozen scientistCLI2.1.219 are separate provenance.

Accepted evidence: all14 ship a greedy-resolving config;10 measured same-weight comparisons, including p00s01's near-null result;0/8 protocol cells executed RL versus4/6 controls, an observational recipe difference. Executed maxn>=500 is13/14; selectionn>=500 is12/14. The corrected header tables distinguish first GPU smoke from first main/card-matched training, tool time from authoring wall, and executed n from selection n. Direct ceremony is small relative to the session, but its heterogeneous intervals do not support a precise shared overhead percentage.

Mechanisms guiding action:8/8 protocol cells have pre-card model execution; H's data applicability pressure is widespread; B's stop/parser/raw-retention guidance covers repeated large losses; unsafe inherited-save config occurs in8/14 traces, but frozen D misclassifies safe cases. Method stacks,data scale,soups,prefix mixtures and saved compute do not identify protocol-caused score gains.

Additional adjudication beyond the revised helper:
- Do not call the quoted25 error lines plus8 lock-refusal messages33 distinct friction events, or claim collect captures6/33, without event deduplication and common units.
- Strong score/format/retention-rule causality remains qualified: retained checkpoints enabled a later soup; this is not proof that the rule itself caused its score. Large base-to-SFT changes bundle data,format,optimization and often decode.
- The revised helper's proposal to turn accuracy/SE inversion into an n-verification gate is **not accepted**. Even its caveat calls the inversion a cross-check, not proof. Integral output does not certify estimator/provenance; degenerate0/1 scores and other estimators need genuine count metadata.
- The revised D2 idea of enumerating expected false positives and requiring only zero unexpected ones does not solve the known scope/repair problem. Known safe non-saving and repaired-save paths remain required design cases, not exclusions used to make a screen pass.
- g01r03's subset intent does not make its comparator valid:122/150 stored-array IDs were wrong; true aligned128→127 differs from claimed126→127. Original file value/count binding and changed serving conditions remain distinct defects. Official score unchanged.

## Directions

| direction | current decision |
|---|---|
| D v1 |entire unstarted91060–91063 cancelled and harvested; no new clean cells|
| D2 |retain save-safety objective; defer implementation until operation/repair evidence and false-positive behavior are specified; do not adopt the family shortcut or a known-error whitelist|
| B v2,H |keep unchanged and first-wave scientific priority; no redesign justified by this window|
| J |fresh scientific audit passed:8/8 new protocol cells reproduce the specified scope defect; advance existing frozenJ to next held-registration preparation, with unchanged metric/score floor and fresh source/site checks|
| K |retain separate frozen lifecycle screen; g01r03/g01r08 add future-comparator evidence; not the ordinary-comparator fix; next-wave preparation under its operational boundary|
| A v2 |retain held; no measured endpoint is never a zero-latency pass. Declare missing/unknown categories and consistent first-post-SFT(smoke/pilot/main) clock before release; no retrospective success-threshold change|
| old E v1 |withdraw its whole unstarted91064–91067 block after exact-ID/hold checks; stale-tail=>dead assertion is false, so it is not useful held work|
| E2 |remain frozen/unregistered; strict non-saturation proof unresolved/conditional, not saturation and not an automatic G/P1 replacement|
| comparator binding(new direction30) |retain for separate design using actual-count/value evidence and explicit subset identity; reject SE-only hard gating and self-declared n as verification; no change to frozenK|
| P4 |retain for a separate single-rule design/forward review; timer/ comparable-cost honesty may be tested, not forced budget consumption. Define unknown costs,comparability and non-vacuous exposure first; exact proposed30%/15min cutoffs are not yet adopted|
| P1,G |no automatic registration; P1 observability/stop semantics and G exposure remain unresolved|
| P2,P3,I |independent backlog; no mixed riders. New raw-render evidence strengthensI; no universal recipe rule|
| rule7/WMA text |policy/scope questions remain separate; no bare-number scanner,implicit ID exception or unrelated cleanup piggyback|

## Decision

**Keep the guard baseline; no promotion and no new runtime variant from the helper proposals is accepted as written.** This is a valid no-runtime-change analysis window with substantive queue pruning and advancement of an already-frozen single-item screen(J). B/H/J are the next scientific priority, not release authorization. Retain independent next-wave work rather than waiting for unrelated stragglers.

The old E replacement-first timing rule is superseded **for E v1 withdrawal only**: after D removal,22 other usable held cells already remain, so keeping four known-invalid stale-tail jobs until an E2 receipt exists serves no buffer need. Removing them neither declares E2 saturated nor authorizesE2. Preserve its old manifest/tree/receipt and harvest all four; never cancel running work or select individual cells by outcomes.

## Change

Runtime exp_protocol: **none in this decision**; all six host shipped paths remain guard drift2f64581. Existing candidateJ remains549e25a/tree7ae08ccf,not stacked. Planner queue changes withdraw whole D1/E1 blocks; D cancellation is complete,E cancellation execution is recorded separately. Operator cancellation now requests controller-filtered PENDING-only targets and confirms CANCELLED before success recording;100 CPU regressions pass. Derived audit JSON was moved out of the receipt-only batch root, restoring reconciliation.

User-authorized reusable knowledge is in meta: cumulative monitor counters,source-aware arm/eligibility,raw-vs-rendered targets,actual exit bounds,structured ID/epoch joins,declared dataset prefixes,receipt-root layout and actual local-Claude session identity. Original helper output remains unaltered.

## Evidence

Five focused reviewer reports plus the planner prefix audit are linked by launch.json. The planner reran the exact frozenD CPU fixture and the34-log structured pair parser; inventory/pair tables match byte-for-byte. Official validator/judge status is unchanged.100 operator/launcher tests,8 focused cancellation cases and4 meta-file tests passed; tests do not establish GPU efficacy or native isolation. Mixed-state baseline release refusal is separately documented/tested; it has not been bypassed.

## Next round

1. Complete/harvest oldE's exact four unstarted cancellations; preserve22 useful held.
2. Prepare a clean source-frozen execution checkout for J registration if concurrent user files still occupy the operator tree. Preserve those files; never ignore them merely to pass the clean-tree gate. Revalidate J's existing SHA/tree,all-six-path baseline-relative diff,contract and site,then register held through the normal operator path. This authorizes held preparation under OWNERSHIP OK, **not release**.
3. Select at most3 single-item candidates per wave with same-generation guard drift comparisons. B4/H4/J4+driftA2 is14; control repair adds1. With J held added,the current useful pool would rise22→26,leaving11 after15 releases. Actual release needs live exact-ID math,OWNERSHIP OK,frozen ReqNodeList and native isolation or explicit per-receipt authorization. Do not count a checked manifest as a held receipt.
4. Handle the strict-baseline mixed receipt explicitly before selecting its5 remaining holds for release. A,K,P4 and any revisedD need their stated remaining scientific/operational preparation; no automatic releases or filler.
5. Maintain the current hourly detector and2 buffered clean tail cells. No new8-clean window exists; withdrawn jobs count as administrative terminals,not clean evidence. Winners still require a new second4-cell block and held-out confirmation before promotion.
