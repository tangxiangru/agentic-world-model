# G four-cell readout and 16:30 operator disposition

**G fails its preregistered scope guard; do not promote or extend it unchanged.**
All four G cells pass PTB completion and automatic judges, but four original
access flags remain and a separate manual audit confirms held-out-input exposure
missed by the scanner. The new Opus4.8 study is a different frozen runtime.

## Queue and completion

Ownership is OK: 16/16 GPUs allocated, 16 RUNNING, 46 safely routed PENDING on
slurm2-a3nodesetondem-[2-3]. All 62 active jobs have null scheduler dependencies;
45 pending jobs report Priority, one Resources. No replenishment or cancellation
is needed. BFCL c53r04/c54r01/c54r02 (92184/92185/92186) started automatically.
The inspected reconcile preview contained three harvests and 16 peeks, with no
submit/cancel actions; all three new terminal G attempts were harvested.

Validator/automatic-judge-clean completions total 85. This does not clear WMA
access flags or manual semantic exclusions. H remains 0/4 complete; Opus4.8
remains zero complete, two previously recorded incomplete raw attempts, 12
running and 46 pending. GPQA access remains blocked.

Allocation is not utilization. Shared samples do not cover the current fleet.
Direct read-only GPU queries on the two owned nodes were blocked by strict SSH
host-key verification (no trusted ED25519 keys configured). Trust checks were
not disabled; continuous full utilization is not established.

## Frozen score and ledger comparison

G private source is `125a434`, skill `e4402ffa6bca`, public `ae46724`, PTB
`62203e4`. Comparing every private shipped non-skill path against `ae46724`
returns no diff. Public configuration matches w10. Other runtime cohorts and
Opus4.8 are kept separate; full provenance is in `provenance.json`.

| G cell | Job | Final accuracy | Allocated time | Scientist USD |
|---|---|---:|---|---:|
| w13r01 | 91441 | 72.6308% | 08:56:07 | 51.7140 |
| w13r02 | 91442 | 70.2047% | 08:24:34 | 54.0175 |
| w13r03 | 91443 | 76.2699% | 08:18:44 | 46.9158 |
| w13r04 | 91444 | 75.0569% | 06:46:37 | 42.5252 |

G mean is **73.5406%**, sample SD **2.6897 pp**, n=4; matched w10 v0.2 is
**72.3180% ± 2.7382 pp**, n=8. The descriptive difference **+1.2225 pp** is
smaller than within-arm variation and is not a formal effect or promotion.
Keep all four PTB scores visible rather than selecting unflagged-looking cells.
Slurm FAILED/2:0 is preserved alongside each valid scientific completion.

The unchanged G ledger is **25 final / 21 scored / 4 original flags**: L0/L1
hit 1.0, L2 coverage .722 over 18 scorable cards, mean width .1441, width/noise
5.5606, L3 hit .647, saved/wrongly-killed GPU-h both zero. Baseline width/noise
is 4.4546; these self-selected cards do not establish improved calibration.

Retained-final WMA cost is $47.6701, $1.9068/verdict, mean wall 5.946 minutes.
The same all-final cost ratio to baseline is .9643; restricting both sides to
unflagged finals gives .9899. These limited ratios do not establish whole-spend
efficiency: 42 requests completed, but 17 earlier review sessions/costs were
overwritten. Full review spend is at least the retained amount. Do not fill
missing cost with zero or scale final cost by 42/25. Final recorded waits total
142.395 minutes; all-request lifecycle is 251.305 minutes. Neither is measured
GPU idle. Full metrics and denominators are in `g-summary.json` and ledgers.

## Scope guard and independent manual flag

The scope audit identifies four successful outside metadata reads:
w13r02/exp-05 and w13r03/exp-01, exp-04, exp-05. These are directory/capacity
reads, not four confirmed label leaks; they still violate the original zero-flag
guard. An extra failed test-parquet metadata attempt in already flagged exp-04
exposed no file and does not become a fifth original flagged verdict.

The stronger finding is originally unflagged **w13r01/exp-01**. Scientist trace
lines 1843–1853 export GSM8K main test rows 0–149 into a session file. The
42,004-byte, 150-record file hash matches the frozen lock. Private tool lines
78–79 load all 150 records into the probe process and print two question/gold
records into WMA model-visible context before this card runs. Final evidence e4
uses that content. The operator independently verified export source, lock/file
hash, tool IDs, output fields and unchanged empty access.outside; see
`manual-flag-crosscheck.json`. New audit records do not copy question/answer text.

`g-manual-semantic-flag.json` records the confirmed violation independently.
Original ledger/scorer/flags stay **25/21/4**; no retroactive score change makes
G pass. No claim is made that exposure raised PTB score or entered training.
Two purpose-selected unflagged probes do not estimate prevalence; zero hidden
side effects is also unproven. Detailed evidence and limits are in
`g-scope-guard.md` and its provenance JSON.

Static inspection of current Opus4.8 source `31b854bb`,
`awm/wma/isolation.py:311`, confirms automatic selection of card/lock/code and
prior closed cards/history, not full session eval/data trees. The old G JSONL
path is not automatically exported there. Already contaminated selected text
remains a semantic limitation; do not transfer the old G failure to the new
cohort or change its frozen treatments on that inference.

## Blocker retention and decision

The other bounded audit finds 42/42 completed responses and 25/25 final delivered
locks; the operator cross-counted these artifacts. Two earlier negative reviews
in w13r04 led to repairs, including a reproduced GenerationConfig save failure.
Conversely w13r03/exp-03 initially received L0/L1 yes, trained for an observed
69:03, then failed on that save class before repair. The operator checked the
cited failure/recovery evidence. Final successful cards hide this miss; no old
verdict is rescored against a modified proposal. Three timelines do not establish
a global launch-version compliance rate. See `g-blocker-retention.md`.

Ranked conclusions: (1) pathname text alone failed both original-fence and
semantic boundary requirements, directly falsifying G; (2) useful blockers and
delivery remain possible, while matched recall and net benefit are unconfirmed;
(3) the small descriptive score increment and partial costs justify neither
promotion nor a new skill edit. Any export-policy change is a separate protocol
decision with preregistration/acceptance, not a scorer adjustment. No new skill,
runtime, scorer, guard, retry or experiment submission is made this check.

Both delegated reports now have this verified operator disposition. The hourly
singleton remains alive (PID 3591763). Seven new clean cells are beyond its
previous analysis watermark; tail eligibility begins at 16:39:06 UTC, so the
next hourly pass can freeze the window unless eight cells arrive sooner. Old
ready reports already have completed handoffs. No duplicate analysis or timer
is started here.

A receipt-scoped accounting fallback also yielded no utilization: `sstat`
accepted its supported TRESUsageInAve field but returned no rows and Invalid
user id errors from the compute nodes, despite process rc=0. This is an access/
identity limitation, not an ownership-registry failure. `telemetry-check.json`
preserves that result; no user identity or SSH trust check was bypassed.
