# 17:30 operator review — H first cell and Claude handoff

The seven-cell Claude event `20260904T164313Z-792dc7f482` completed at
17:09:50 UTC, returncode 0, actual modelUsage `claude-opus-5`, requested max /
ultracode, reported cost **$23.09177025** against its $25 cap. Its original plan
and output are archived here. The prior shared operator pre-review was read
alongside the report; it was an awaiting-report marker, not an accepted handoff.
This document completes the operator review **with corrections**, not blanket
acceptance of the report's claims or its purported authority to authorize edits.

## Queue, harvest and new result

At 17:31 UTC: ownership OK, 16/16 GPUs allocated, 16 RUNNING and 45 safely
routed PENDING on nodes 2–3, all null dependencies. Forty-four pending jobs
report Priority and one Resources. No capacity replenishment is required.
BFCL protocol c54r03 / 92187 has started by ordinary backfill. Utilization
remains unmeasured because of the previously recorded SSH trust / Slurm identity
limitations. The inspected reconcile preview contained one harvest and 16 peeks,
no submit/cancel; it was applied to archive H/w14r04 and update active snapshots.

H/w14r04 / 91448 was the first PTB validator/automatic-judge-complete cell at **68.4609552691%**
(standard error 1.27994 pp), but contains two original WMA scope-flagged verdicts.
Coverage at that first snapshot was H 1/4; no replicate SD or score effect was estimable. Its frozen H
skill is `a536a0af24d7`, private `7e69e5c`, public `ae46724`, old Opus5 cohort.
Slurm FAILED/2:0, allocation 08:14:07 and agent time 07:48:25 are retained.
Scientist reported spend is $55.70025725, excluding judges; retained-final WMA
cost is $11.9022, excluding three overwritten review costs.

The first-cell bounded review finds **two eligible soup cards**, exp-05 and exp-06,
although exp-06 contains two averages. Both locked plans/script hashes match.
Positive time-short-default endorsement is 0/2, explicit rejection also 0/2;
card-specific budget-fit reasoning is not the prohibited generic clock prior.
This is below the three-opportunity minimum. Soup3 was usefully evaluated and
adopted, but its merge/selection rule was already frozen; exp-06 ran both
variants despite B-only advice. No H-induced selection change is established.
Different probe/dev rulers and the incumbent recheck remain separate.

Nine request/response pairs completed and six final locks were delivered.
Final waits are 39.625 minutes; full request lifecycle 58.013 minutes, neither
measured GPU idle. Original scope flags in exp-02/03 cover three outside
metadata operations. They are preserved and fail the zero-flag guard; this is
not a new claim of outcome exposure. The unchanged H ledger is 6 final / 4
scored / 2 flagged, L2 coverage .333 (n3), width/noise5.55, L3 hit .333.
Full evidence and limitations are in `h-first-cell.md`; no H promotion or
unchanged extension is justified, and the other three H jobs continue.

Total PTB/automatic-judge-clean completions are 86. That count is not a statement
of semantic WMA validity. The initial Opus4.8 study remains zero complete and
two known incomplete raw attempts; no selective retries are made.

## Claude findings accepted, qualified or rejected

| Claim / proposal | Operator disposition |
|---|---|
| G fails scope/semantic guards; no promotion | Accepted; already independently established, original 25/21/4 and manual G exposure preserved. |
| Retained G25/25 and baseline58/58 L3 answers are yes; clean reject-truth counts 6+8 | Independently reproduced in `L3-final-denominator-check.json`. Saved/killed zero has a zero non-yes denominator; this does not prove all rejected experiments were knowably worthless before running. Expanded 42/76 review census is the report's reconstruction, not complete recovered costs/versions. |
| First baseline4 deployed a different flush-enabled client from extension4/G | **Contradicted.** All 20 configured cells share identical materialized public client bytes from ae46724; private ship lists omit that client. The operator checked its exact SHA and frozen-byte equality. Preserve 4+4 submission strata, but no flush-based change to the existing n8 cohort. |
| Seven Slurm failures have two error signatures | Confirmed signatures: five line-240 syntax messages, two stale-file-handle messages; all seven raw pipelines completed. A frozen-62203e4 line-240 epilogue cause is not proven: that script has only 212 lines and the error names a live entrypoint. Possible mutable-wrapper effects remain an operational limitation. |
| The old launch detector has false positives | Supported in bounded cases; lock commands, heredocs and unrelated work can match its regex. Preserve old instrument output and distinct actual launch evidence. |
| G25/25, baseline58/58, no version bypass, all earlier compliance conclusions wrong | **Not independently certified.** Five cases support reviewed-before-main-launch locally, not all83 versions. Old client matches verdict mtime/size rather than request/fingerprint identity. Do not erase earlier genuine asynchronous-review or version-alignment findings. |
| Response completion can lag visible delivery | Confirmed in the cited example; no universal clock correction or arbitrary timing tolerance is introduced. |
| Gate cost is an established ~3% of14h | **Rejected.** Budget is10h, earlier waits are missing and the corrected interval artifact was not supplied. No complete overlap/utilization estimate follows. |
| G's score difference is fully explained away by training hours | Unsupported causal language. A nonsignificant observational slope, self-selected recipes and possible post-treatment variables cannot prove what caused or removed a treatment effect. Keep the descriptive +1.2225 pp and uncertainty. |
| Fence was buying calibration; residual G operations were merely under-specified | Not established by dropped-hit counts. Cards/types/subjects differ; G explicitly restricted home/capacity probes. Successful metadata reads and manual semantic exposure are direct failures, not a reason to weaken the fence. |
| No previous candidate targeted width | Incorrect history: Round02 A+B explicitly included the width-basis edit. No new novelty claim is adopted. |
| N1 thresholds .60/.40 authorize N2; N2 promotes from reject-truth hits | Advisory thresholds are not accepted authority or a substitute for the unchanged scientific/held-out gates. Negative experiment results do not automatically establish ex-ante futility. No WORTH/scorer or no-rate target changes. |
| N3 detector must return perfect25/25 and58/58; N4 reporting edits | No rewrite is made to hit an unverified target. Denominator/row-exclusion caveats are recorded; any measurement repair needs a separate validated contract and retained old results. |
| Crossbench can attribute nothing to method/policy | Too broad. Its within-task arms estimate their stated practical method/component contrasts, but do not isolate the whole old/new policy bundle or individual skill lines. |
| N5 old-policy comparator under the same new runtime | Accepted as a useful independent four-cell reference, preregistered separately by the operator under the user's GSM8K/new-method comparison authorization. Its real dependency is exact production acceptance, not H or unrelated tails. |

The two delegated verification reports are committed here. The provenance
audit recomputed deployed package bytes/digests and exact wrapper signatures;
the gate audit checked five decisive cases and kept its limits explicit.
The operator cross-check independently confirms public client bytes, the
212-line frozen wrapper, and final L3 denominators. Report score means match
existing records; post-hoc power/OLS estimates and per-verdict significance
claims are not used as launch or promotion gates.

## Experiment decision and completed handoff

No current WMA skill, scientist runtime, scorer, guard, or existing receipt is
changed on this report's authority. A separate S0 legacy v0.2 policy reference
is being prepared per `doc/spec/2026-09-04-opus48-wma-policy-comparison.md`,
with four scientific repeats and a distinct validation-only smoke. Only policy
text changes against the new private runtime; current S stays frozen. Formal
cells must not start before exact runtime acceptance returns. The experiment
record will carry source, validation receipt and staged/submitted disposition.

The shared event receives a completed reviewed-with-corrections handoff and
must not be reanalysed or resubmitted as the same evidence window. The original
Claude report and its erroneous claims remain archived for audit; corrected
interpretations are here. Promotion remains None. Existing singleton monitoring
continues; no additional timer or PR dependency is introduced.

## H full-arm addendum

All four H cells subsequently completed and were harvested in this same operator
check: 73.6164%, 70.7354%, **80.0607%**, 68.4610%; mean **73.2183%**, sample
SD **5.0257pp**, range11.5997pp. The 80.0607% cell is the highest observed WMA
single score to date, but the arm's mean is only+0.9003pp over matched v0.2 and
is overwhelmed by dispersion. A maximum is not promotion evidence.

The full bounded mechanism readout finds six eligible soup cards across three
cells. Positive time-short-default endorsement is0/6, so the text-adherence
metric is met. Actual uptake is mixed and often already prescribed by the frozen
card; no clean H-caused soup, ingredient or incumbent decision is established.
The authoritative full H ledger is25final/14scored/**11 original scope flags**,
L2coverage.727(n11), width/noise4.9453,L3hit.600. The zero-flag guard fails,
so H is not promoted or repeated unchanged despite the text metric.

Scientist cost totals$224.60378525; retained-final WMA$49.117, with superseded
WMA costs and judges unavailable. Known combined spend is$273.72078525 and
allocated time35.0261GPU-h; allocation is not utilization. Complete evidence is
in `h-complete-*`, `h-costs.json`, and `h-complete-mechanism.md`. All Slurm
FAILED/2:0 and PTB-complete facts are kept.

Final queue audit at18:11UTC: ownership OK,16/16 allocated,16RUNNING and
**43 safely routed PENDING**, no bad routes. That pending count is42 original
scientific jobs plus validation-only92312; four S0 science cells are staged and
not counted. All live scheduler dependencies are null. No capacity fiction is
used: even excluding the technical job, scientific pending remains above32.
