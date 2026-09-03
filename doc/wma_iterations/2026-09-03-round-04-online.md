# WMA online Round 04 — scoped probes and candidate selection

Status: launched. G/H each have four safely routed PENDING cells; no candidate result read.

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
