# WMA Round 04 — scoped probes and candidate selection

Status: G/H approved for exploratory launch after local Opus 5/max UltraCode
review and operator verification; I/J deferred. No Round 04 outcome has been read.
The round record identifies the immutable commits and launch receipts.

## Frozen evidence and decision boundary

This design uses the completed w10r01..04 and c10r01..03 window from event
`20260903T180444Z-79c8d29810`, the verified reports and artifacts in
`doc/wma_iterations/evidence/2026-09-03-w10/`, and the already recorded Round 01
diagnosis. It reads no A–F outcome. c10r04 and the baseline/control extensions
are excluded from design, because their results are not required to identify
these mechanisms. They remain part of the preregistered comparison cohort
when complete. The four-cell diagnostic window is not promotion evidence.

The first Opus event report overstated gate costs and verdict loss. The
operator corrections and completed specialist reads supersede those claims:
43 reviews retain 43 recoverable L3 answers, all yes; two early L0/L1 no
answers led to repairs. Their first answers cannot be scored against the
modified proposals' outcomes. All 11 final access flags remain under the
original fence. Request lifecycle duration is not measured GPU-idle time.

## Contract shared by every candidate

- GSM8K; google/gemma-3-4b-pt at
  `cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- Scientist claude-opus-5[1m], high, 1M context, ten hours; one H100,
  16 CPU, 128 GiB, 400 GiB scratch per cell; no pilot.
- WMA claude-opus-5/high, online, cpu=10,gpu=0,wall=15,turns=40;
  the same read-only train history, containers, evaluator and judges as w10.
- Public protocol SHA `ae4672487cc16f1d75921dcaab85498d7adc938e`.
  Every private shipped path except `skills/wma` must be byte-identical to
  this same baseline. In particular, do not accidentally include the newer
  `awm/exp_protocol/lock.py` in a skill candidate's private archive. Restore
  the current operator runtime after freezing the candidate commits.
- Each candidate is exactly one skill edit against v0.2
  (`wma_skill=176f0a464986`), four repeats. No stacking or pooling candidates.
- The comparison is against lock-gated w10, including its identical-treatment
  four-cell extension when available. Old asynchronous cohorts are context
  only. The c10 extension diagnoses the WMA treatment, not a substitute for
  the w10 skill baseline.
- Node route strictly slurm2-a3nodesetondem-[2-3], checked while held and after
  release; registry-backed receipts only. Independently validated candidates
  may be submitted while the evidence extensions and other candidates run.

## Ranked causes and proposed single edits

1. **Scope discipline** is a concrete validity problem. Seven of eleven flagged
   verdicts queried unrelated mounts; four read outside package source;
   w10r01/exp-05 also changed an installation cache. In-scope preflight, config,
   prior results and logs often already support the relevant feasibility claim.
2. **Candidate-selection advice** is actionable but uneven. w10r02/exp-04
   followed a two-ingredient alternative (.710 versus four-way .694 at n500);
   w10r03/exp-06 mixed a weaker ingredient into a .7400 incumbent and scored
   .6933 at n150. The first improvement over the best ingredient is small and
   below the ruler's resolution; averaging is not universally harmful.
3. **Evaluation uncertainty is mischaracterized by the manual's categorical
   C18 rule.** w10r01/exp-08 and w10r03/exp-07 reversed smaller-prefix rankings
   on larger common item sets. Repeated runs still estimate runtime variance;
   greedy must not be equated with bit determinism. Fixed-item repetition
   alone cannot add coverage of items that were never evaluated.
4. **Probe scope and stopping** can waste current-review effort. w10r02/exp-01
   p2/p5 explicitly concern later training, and w10r04/exp-06 p5 repeats earlier
   corpus statistics without a stated discriminating outcome. This is not a
   claim that every changed:none probe is wasted; some confirming checks are
   useful and some recorded probes never ran.

| Candidate | Single edit | Mechanism and provisional primary readout |
|---|---|---|
| G probe-scope | One SKILL rule checking intentional probe inputs and side effects against the existing scope | Zero original-fence flags and zero outside mutations, with delivered verdicts and useful blocker detection retained |
| H soup-ingredients | Replace only the time-short-default clause in the C6 prior | Remove positive reliance on the clock as a default reason for averaging; read resulting L3, incumbent and merge decisions |
| I eval-uncertainty (deferred) | Proposed C18 row correction | Existing v0.2 already recommends larger common samples; no documented misallocated repeat establishes an incremental mechanism |
| J probe-stopping (deferred) | Proposed probe-value rule refinement | Few cited probes and an existing stopping rule do not support the proposed 20% latency target |

G regulates where/how probing operates; J regulates whether another probe can
inform the current card. H regulates averaging advice; I regulates the next
measurement. A–F retain their original questions and denominators. L0/L1-to-L3
consistency is deferred: the two negative cases already induced appropriate
repairs, so current evidence gives a weaker incremental behavior argument.

## Readout, falsification and guards

Read coverage and validity first. Report complete/intended cells and original
judge/access flags, then ledger by skill hash and change type, cost and score
mean/SD/range. Keep terminal-verdict metrics separate from per-request review
counts and latency, including relocks. Do not use the known background-tool
heuristic failures in gate.py/uptake.py as causal timing denominators.

For H (and I if later revived), eligibility is determined by the locked pre-launch proposal and
available pre-launch evidence, not by whether the result was favorable or the
WMA labelled it conveniently. Report numerator, denominator and cells with an
eligible opportunity; no opportunities means the mechanism was not tested.
Manual uptake requires a cited actual action after the returned verdict and
before the next relevant decision, not later matching text. Confirm sample
identities and effective evaluator settings when claiming a matched comparison.

- G is falsified if outside reads/writes persist, or an apparent validity gain
  comes from missing/invalid verdicts, lost justified blockers or indirect
  access that merely evades path detection. Do not clear any old flag.
- H eligibility: the locked proposal constructs or evaluates a named weight
  average of at least two checkpoints with identifiable pre-launch lineage.
  A merge-script command is not required: w10r03/exp-06 and exp-09 invoke an
  evaluator on a previously constructed soup. Exclude continued training with
  an incidental C6 label. This gives five baseline cards (w10r01/exp-07,
  w10r02/exp-04, w10r03/exp-06 and exp-09, w10r04/exp-05).
  Primary: positive endorsement of the time-short default in evidence/L3;
  baseline four evidence notes plus exp-06's L3 note (five distinct cards),
  target zero. Explicit rejection of that prior is not an endorsement.
  This is an adherence metric, not proof of utility: require a cited changed
  L3, merge, ingredient or incumbent decision for behavioral benefit. Fewer
  than three eligible soup cards means insufficient opportunity. Report all
  matched scores, costs and actual decisions, including beneficial soups.
  H fails to establish benefit if only wording changes or if useful merging
  is suppressed. No bound on soup quality from its weakest ingredient is valid.
- I, if revived in a later preregistration, is falsified if advice confuses item coverage and runtime variability,
  drives unmatched comparisons, is not acted upon, or spends extra evaluation
  budget without improving the quality of the selection evidence. It mandates
  neither full evaluation on every card nor zero-variance greedy assumptions.
- J's deferred draft target was a 20% reduction in the mean per-cell lifecycle
  minutes per distinct reviewed card, versus the matched w10 baseline. This
  threshold is a preregistered experimental choice, not an observed effect.
  Report total lifecycle minutes and request counts too. Falsify if irrelevant
  probing persists or latency does not improve, or if the gain drops required
  blocker checks, coverage or the blocking launch boundary.

All original guards remain: zero leak-suspected verdicts, accepted-verdict
cost at most 1.5x matched v0.2, PTB not below v0.2 beyond its observed spread,
and no unjustified killed work. Preserve verdict-before-launch and report all
timeouts/skips. Cost/latency of superseded reviews is missing from terminal
cost; it must be disclosed, not imputed as zero. G is not stacked into H/I/J;
if their unchanged scope discipline fails the guard, they cannot be promoted.

Four cells are exploratory screening, not a claimed score gain. Formal
comparison and promotion still require the wma_meta evidence/replication and
held-out gates, with at most one edit promoted. A second useful edit is retested
on the promoted baseline. No held-out failure details, scorer revision or
post-hoc guard adjustment is allowed inside this round.

## Final pre-launch review disposition

Opus completed the bounded review at 19:23 UTC, reported cost $3.08096925,
with a successful result. Its unchanged report is archived beside the operator
record. The operator accepts G and a narrower H, and defers I/J.

Corrections to that advice: a soup's gain is not bounded by its weakest
ingredient; that phrase is not shipped. H changes only the unsupported clock
prior, not type, tier, eligibility wording, ingredient-check instructions or
cost cells. The suggested merge-command-only classifier misses evaluation-only
soup cards and is replaced above. No hard latency ceiling or power conclusion
is inferred from self-reported probe CPU minutes. G blocker preservation is
judged on encountered opportunities, not a quota forcing negative verdicts;
insufficient opportunities leave that channel unconfirmed. Zero scanner flags
is not proof of zero hidden side effects or outcome exposure, so manual audit
remains required. Neither the old flags nor the scoring rules are changed.

I's possible manual inconsistency can be revisited when a cited bad repeat
recommendation leads to wasted evaluation instead of an available larger
common sample. J needs stronger measured unnecessary-probe evidence or a
separate relock mechanism; its four-cell 20% draft target is not adopted.

Only G/H enter this wave: eight jobs take the current safe pending reserve
from 17 toward 25. This exceeds the 24-job replenishment threshold but not the
32-job planning target. Do not launch deferred weak designs or redundant
repeats solely to close that difference; use later validated evidence to
prepare the next independent settings before the reserve drops again.
