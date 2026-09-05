# WMA online Round 04 — scoped probes and candidate selection

Status (2026-09-04 16:30 UTC): G is complete at 4/4 but fails its original
scope guard and a separate manual semantic audit; it is not promoted. At 17:31
UTC H has one completion, with two original scope flags and insufficient soup
opportunities; three H cells remain running. Frozen launch history is retained.

The frozen design and gates are in
`doc/spec/2026-09-03-wma-round04-probe-selection.md`. Completed evidence and
independent trace audits are in `evidence/2026-09-03-w10/`.

At the 18:52 UTC scheduled check, ownership was OK and the WMA nodes had
16/16 GPUs allocated, with 17 safely routed pending cells. There were 48 clean
completed cells across distinct historical cohorts, unchanged from the prior
check. A–F had no completed result. The reconcile preview required only running
cell peeks and no new harvest or submission.

The previous event analysis was incomplete. Three bounded parallel specialist
reads completed access, uptake/levers and decision/harm diagnosis. Their
corrected findings were recorded in commit `9fccdde`. A separate bounded local
Claude Code `claude-opus-5` / max / ultracode follow-up was launched for the new
G/H/I/J design question. Its artifacts live in the original shared event's
`followup-design/` directory; this is a new preregistration question, not another
blind replay of the same broad event analysis. It is read-only and does not own
any launch, edit, commit or promotion.

Baseline remains v0.2 (`176f0a464986`), public/private runtime reference
`ae46724`. Independently justified candidate manifests will use four repeats
and frozen matching runtime bytes. The c10 tail and baseline/control extensions
are not needed to identify these prospective mechanisms; they remain needed
for their original formal comparison. No candidate is promoted by this record.

## Candidate archive preparation

The four reversible drafts were reviewed before launching; the two accepted
G/H candidates are now being frozen. Freezing alone is not a launch. Each is
one skill intervention against v0.2, and all non-skill private shipped bytes
are checked against `ae46724`. A preparation commit restores the old private
`lock.py` for these archives; after freezing, the branch restores the current
operator `lock.py` and the baseline skill. No existing manifest is changed.
This keeps the new archive's runtime identical to the comparison baseline
without discarding the operator's newer relock-history fix.

## Verified local Opus review

The follow-up finished at 19:23 UTC (exit 0, is_error=false; reported cost
$3.08096925). It supports G/H and defers I/J. Its exact report is committed as
`evidence/2026-09-03-w10/opus-design-review.md`; operator qualifications are in
the final pre-launch spec. In particular H removes only the positive
"default when time is short" prior, never adds the false weakest-ingredient
bound. G preserves the original input fence and legitimate scratch probes.
The five-card H baseline and all 43 review summaries were checked against
actual artifacts; no output was regenerated or rescored.

I/J are not launched: existing larger-sample advice and a few irrelevant probes
provide weak incremental behavior evidence at this replication. Their draft
texts remain design history, not candidate sources. Eight justified G/H cells
replenish 17 toward 25 pending; the remaining gap to 32 is not filled with
redundant repetitions or unsupported edits.

## Candidate archives

- g-probe-scope: `e4402ffa6bca`; one new probe-scope rule; compared with v0.2 on byte-identical private runtime.
- h-soup-ingredients: `a536a0af24d7`; replace only the time-short default clause in the C6 prior; compared with v0.2 on byte-identical private runtime.

| Candidate | Immutable source | Skill hash | Cells |
|---|---|---|---|
| g-probe-scope | `125a434e6d73d067427911332663060fe2dce558` | `e4402ffa6bca` | w13r01..04 |
| h-soup-ingredients | `7e69e5c549447fe12fb863352235e3dc38676014` | `a536a0af24d7` | w14r01..04 |

The skill-file contract tests pass for each frozen candidate (6/6 each).
`git diff ae46724 CANDIDATE --` on every non-skill WMA_PRIVATE_SHIP path is
empty. The final operator head restores the current relock-history runtime
and byte-identical v0.2 skill; candidates remain immutable manifest inputs.
Full manifest/site checks and the reconcile preview precede submission.

Both full `awm ptb check` calls returned zero issues. The reconcile preview
contains exactly the two G/H submissions plus the existing 16 running-cell
peeks; it contains no cancellation or harvest. The non-skill manifest contract
matches the frozen baseline apart from experiment identifiers. The restored
operator source and v0.2 skill are byte-identical to pre-archive commit 9fccdde.

## Launch and post-release audit (2026-09-03 19:29–19:30 UTC)

Reconcile submitted both independently specified candidates before copying
receipts and updating the running-cell snapshots:

- G: `w13r01..04`, jobs **91441..91444**; receipt
  `results/ptb/wma-gsm8k-gemma4b-high-r04-g-probe-scope-x4/formal-2026-09-03T192916.640188+0000.json`.
- H: `w14r01..04`, jobs **91445..91448**; receipt
  `results/ptb/wma-gsm8k-gemma4b-high-r04-h-soup-ingredients-x4/formal-2026-09-03T192943.423015+0000.json`.

The immutable manifests are the matching batch-name YAML files under
`experiments/posttrainbench/`; both name the Round 04 spec. Their receipts
freeze source commit `4874838` and the explicit variant SHAs above. Scientific
result directories will be resolved through `awm ptb results` after the jobs
run; no pending job is counted as a completed result.

Both held-route gates passed. Direct post-release scontrol checks found all
eight PENDING with `ReqNodeList=slurm2-a3nodesetondem-[2-3]`. The complete
receipt/cell/manifest/spec and live-route evidence is
`evidence/2026-09-03-w10/round04-launch-audit.json`. Registry ownership was OK;
allocation stayed 16/16 and safe pending increased **17 → 25**. No job was
cancelled. Only the two intended submissions and 16 snapshot updates occurred.

No new valid PTB score arrived in this check: coverage remains 48 clean cells
across different historical cohorts. Overall best is control c01r05 0.815770;
best WMA is w01r05 0.797574. These are maxima, not matched treatment effects.
The hourly completion hook remains alive; the 30-minute operator timer already
queued the next continuation and suppressed a duplicate at 19:30. The shared
event receives an executed handoff marker so later checks do not repeat this
analysis or submit these immutable manifests again.

## First G completion — 2026-09-04 15:00 UTC

G/w13r04 (job 91444) is validator-complete and automatic-judge-clean at
**75.0569%**, standard error 1.1918 percentage points. Coverage is G 1/4 and
H 0/4; replicate variance and a matched effect are not yet estimable. This is
not promotion evidence and is separate from the new Opus4.8 crossbench cohort.

The original skill `e4402ffa6bca` ledger has five scored cards: L0/L1 hit 1.0,
L2 coverage 0.5 (four scorable), mean width 0.1276, width/noise 4.2534, L3 hit
0.5, scanner-suspected leaks 0, and saved/wrongly-killed GPU-h both 0. All five
retained locks record delivered, with 1796.9 seconds of waiting; this is not an
independent audit of every actual launch version or every semantic access.

WMA reported cost is $9.9815 and scientist reported cost $42.52521025, excluding
judges. Agent time is 06:22:09; allocated time 06:46:37. Slurm exit FAILED/2:0
is preserved alongside the successful PTB completion verdict; no retry occurs.
The full receipt/cell/manifest/spec/result chain, unchanged ledgers and bounded
review are in `evidence/2026-09-04-1500/`. Existing exclusions remain in force.

## G complete / guard failed — 2026-09-04 16:30 UTC

New jobs 91441/91442/91443 complete at 72.6308%, 70.2047%, 76.2699%; together
with w13r04, G is **73.5406% ± 2.6897 pp sample SD**, n=4. Matched w10 baseline
is 72.3180% ± 2.7382 pp, n=8. The +1.2225 pp descriptive difference is not a
formal effect. Public configuration matches; non-skill private shipped paths
are byte-identical to baseline. All four PTB results and original Slurm
FAILED/2:0 states are retained.

The authoritative ledger has 25 finals, 21 scored, **4 original access flags**;
L2 coverage .722 (n18), width/noise 5.5606, L3 hit .647. G therefore fails the
predeclared zero-original-flag criterion. A bounded parallel audit additionally
confirms **w13r01/exp-01 raw held-out-input exposure**, originally unflagged:
150 GSM8K test records were loaded from a session-resident input copy and two
question/gold records reached WMA context before this card ran. File/hash/source
and tool-call evidence were independently checked. Record it separately; keep
original ledger/scorer/flags unchanged. No score or training contamination
causality is inferred, and no question/answer text is copied into new notes.

42 request/response pairs completed and all 25 final locks delivered. Two early
negative reviews led to repairs, including a reproduced save-path failure;
another cell missed that same failure and lost an observed 69:03 training wall
before repair. This preserves evidence of useful blockers without establishing
matched recall or net benefit. Retained-final WMA cost is $47.6701, excluding
17 overwritten review costs; final waits are 142.395 minutes and full request
lifecycle 251.305 minutes, neither measured GPU idle.

**Decision: G falsified; no promotion or unchanged extension.** H continues
under its own original contract. The different Opus4.8 runtime is not judged
by G's older export path. Full scores, cost bounds, provenance, unchanged ledgers,
manual flag, two delegated audits and operator verification are preserved in
`evidence/2026-09-04-1630/`.

## H first completion and verified Claude handoff — 2026-09-04 17:30 UTC

H/w14r04 (91448) passes PTB/automatic judges at **68.4610%**, but retains two
original WMA scope flags. Coverage 1/4, no within-arm variance estimate. There
are two eligible soup cards, below the three-card mechanism threshold; neither
positively invokes the time-short default, but no H-caused selection change is
established. Nine request completions, six delivered final locks, terminal WMA
cost $11.9022 plus missing superseded costs; scientist cost $55.70025725. The
remaining three cells continue under their original frozen source.

Claude event `20260904T164313Z-792dc7f482` finished with actual Opus5, reported
$23.09177025. The operator accepted its no-promotion conclusion but corrected
its deployed-client split, universal compliance, 14h/3% gate-cost and causal
training-hours claims. All20 relevant cells use identical public ae46724 client
bytes; the n8 baseline/control comparisons remain unchanged with batch strata
preserved. No measurement rewrite is made to obtain a perfect compliance target.
Detailed H evidence, the original advisory, two independent bounded checks and
completed operator disposition are in `evidence/2026-09-04-1730/`.

A separate four-cell legacy-policy reference in the new Opus4.8 runtime is
preregistered in `doc/spec/2026-09-04-opus48-wma-policy-comparison.md`. It compares
old versus new policy without changing G/H or the original crossbench receipts;
its scientific launch remains gated on its exact production acceptance.

## H complete — 2026-09-04 18:06 UTC

All H cells pass PTB completion and automatic judges: 73.6164%, 70.7354%,
80.0607%, 68.4610%. Mean **73.2183%**, sample SD **5.0257 pp**, n=4; range
68.4610–80.0607%. Relative to matched w10 v0.2 (72.3180% ±2.7382,n8),
the descriptive difference is +0.9003pp, much smaller than H variability.
w14r03 is the highest observed WMA single score so far, but a maximum is not
a treatment effect or promotion.

The unchanged H ledger is 25 final /14 scored / **11 original access flags**;
L2coverage.727(n11), width/noise4.9453,L3hit.600 and saved/wrongly-killed
GPU-h both zero. Retained-final WMA cost is $49.117 for25 verdicts. Scientist
cost across four cells is $224.60378525; known combined scientist + retained
WMA is $273.72078525, excluding judges and superseded reviews. Allocated time
is 35.0261h and is not utilization. All Slurm states FAILED/2:0 are retained
alongside scientifically valid PTB results.

Eleven original scope flags alone fail the common zero-flag guard; no H promotion
is possible. The full soup-opportunity/advice readout and bounded audit of the
high-score cell are recorded under `evidence/2026-09-04-1730/`; original flags
and scorer remain unchanged. No additional H repetitions are submitted.
