# Round02 K — 2026-09-03 prelaunch construction

Status: **0/4 run; no scheduler receipt, release or promotion**. This record's introducing commit freezes K. The resolved SHA/tree and manifest follow in the [K spec](../spec/2026-09-03-exp-protocol-round02-k-deferred-comparator.md).

## Variants

| label | commit / tree | difference |
|---|---|---|
| guard drift baseline | `2f64581` / `189319d6` | unchanged guard |
| K deferred comparator | this record's introducing commit | explicit optional prelaunch deferral, strict closure validation and shared portable completion receipts |

Operator parent `a94e1b98ac46f4e87a8dc9c00944f5d5cce5f08d` matches the drift baseline on all six shipped paths. K is one coordinated comparator-lifecycle feature, not stacked on J, H or E. No required field or schema version changes.

## Cells

Planned four independent `k02r01–04`, run_index1; GSM8K, pinned Gemma-3-4B, Opus5[1m] high/1M,10h each. Same Round02 PTB, images, evaluator, judges and six shipped paths. Held-out task: none.

## Results

No K scientist has run. The synthetic fixtures below are neither model results nor PTB-validator-clean cells.

| variant | accuracy mean (min–max) | pitfalls_cost_h sum | n_locked_open sum | fields_filled mean |
|---|---|---|---|---|
| K,0/4 | unmeasured | unmeasured | unmeasured | unmeasured |
| matched drift | not yet run | unmeasured | unmeasured | unmeasured |

Prior guard cards read by the planner are recorded in the [strict Round01 record](2026-09-03-round-01-strict-guard.md): g01s03 exp-05, g01s08 exp-02 and g01s04 exp-03. K uses the additional g01s07 ordering/lock audit and five-cell comparator evidence. Three K and matched-drift cards must be read after they exist; this requirement is not marked complete prelaunch.

## Trace review

The completed [strict-cohort decision](trace-reviews/round01-strict-guard-addendum/planner-decision.md), local Opus max synthesis and [g01s07 ordering/cost audit](trace-reviews/round01-strict-guard-addendum/g01s07-ordering-cost-audit.md) establish a future-output dependency in g01s02/03/06/07/08. Scientists used overrides or evaluated before locking. A legitimate recorded override already exists; the violation is not called unavoidable. The current close/index/collect/hook do not revalidate a comparator, so a missing-file WARN alone would not implement the requested lifecycle safely.

## Directions

Ledger #27 becomes implemented K; #26 J remains independent. D/B/H plus drift A retain first scientific priority, followed by A/E2 plus drift B; J/K are later independent screens. C's unstarted four-cell block was cancelled and harvested; the last live queue check retained29 JobHeldUser cells. No held or running job is modified by K construction.

## Decision

Prepare the independent four-cell screen, not promotion. Among exercised within-card comparisons, missing-future-output preflight failures/overrides and pre-lock evaluator starts must fall to0, with **zero invalid/unknown completed comparisons certified**. At least2/4 cells must exercise the feature; no exposure is uninformative. Inspect metadata fabrication, mode-changing relocks, abandoned diagnostics, failure closure and time/copy burden. Bypasses are intervention failures, not compliance.

Frozen24-cell protocol reference mean0.7037212534748547, block score floor **0.6737212534748547**, as specified in J and K. This historical guardrail is not a causal comparator. A winner needs another independently frozen four-cell block and held-out confirmation.

## Change

Optional `evaluation.comparator.defer_validation: true` declares a planned absolute comparator path with null prelaunch value, named metric and positive protocol n. Missing planned evidence warns at lock, but existing invalid evidence fails. Completed close validates actual evaluated/scored counts and finite metric, plus the recorded target measurement's metric/n/value/path; missing or partial evidence is not accepted. Failed/killed/unrun inconclusive non-adopted cards may close without certifying a comparison.

The lock retains the declaration and cannot drop it through relock. Close binds card/plan/lock/evidence hashes in a receipt sealed by the lock. One stdlib-only helper serves API and installed Stop hook. Index and starting points reject unverified adoptions; collect preserves old raw columns and adds deferred outcome columns. The ordinary manual-close behavior and twelve-block cap remain. Mode-specific instructions live in a linked reference, not a large universal rule.

This checks comparator-file counts/metric and recorded target consistency, **not** actual target-file contents or dev-set/seed/model/decoder identity. A historical receipt can survive omitted raw evidence after relocation; it is not a new verification and re-closing still needs declared source evidence. Hashes are reproducibility evidence, not a security boundary against rewriting every file.

## Evidence and validation

- Protocol schema/preflight/lock/CLI/lineage/collect/questions/hook/skill/install/K/meta-file suite: **164 passed**. No existing tests were changed;36 candidate tests cover the lifecycle, malformed/partial data, edits, failures, portability, no-mode-drop and all consumers.
- PTB experiment/results/ops/Slurm-gate/runtime suite: **111 passed**. These are CPU tests, not Slurm submissions or real scientist experiments.
- Six-shipped-path isolated-reader test loads only the shipped package/skill outside this checkout; installed hook runs under Python `-S`, with no YAML dependency. A retained g01s07 Inspect smoke report verifies actual8 rather than dataset1319. The original legacy behavior and twelve-block cap are tested.
- Independent reviewer `/root/k_forward_check` used real CLI-shaped synthetic workflows after reading the skill, template and linked guide, without specs/audits/intended answers. Preserve its [initial report](forward-reviews/round02-k/synthetic-forward.raw.md) and [focused target-count report](forward-reviews/round02-k/synthetic-target-count.raw.md) unaltered. Missing actual n, partial counts, error reports and later card/receipt/evidence edits never yielded a usable starting point; honest failed closure and retained historical receipts worked. Target n150 against locked/comparator n8 failed; correcting only the postrun record to8 succeeded.
- Forward review exposed unclear raw-versus-verified counters, historical inspection versus re-close, and ordinary manual-close text appearing on opted-in cards. Guidance now distinguishes them; the hook's deferred opening also covers filled but unverified conclusions. The focused report predates that final wording-only correction; the full164-test run includes it.
- `git diff --check` passes. Generic skill validation rejects the pre-existing underscore name `exp_protocol`; preserve installation/runtime compatibility, with repository skill/schema/install checks passing. This is a known naming mismatch, not a waived functional test.

## Next round

Construction follow-up: frozen candidate `58a6992b50378e39d59ec3ef72e988ca1988f855`, tree `ec7d5f2ab9aecd9f8a7a3278c9a9a54b9777eacf`. The K manifest passes local and full checks with0 issues; its fixed contract/context and per-cell settings match J except candidate identity, IDs and descriptive provenance. All six host shipped paths match drift `2f64581` after restoration;128 baseline tests pass. Candidate-specific files/tests remain in the immutable candidate commit, not active on the restored host.

Operational follow-up at22:54 UTC: live registry-backed queue shows29 `PENDING(JobHeldUser)`,17 running jobs and zero other pending reasons. Owned-node allocation is16/16 but registered demand is17/16;90820 remains outside the assigned nodes and OWNERSHIP FAIL persists. `robtang-ptb-a3` still spans11 nodes. Monitor PID2086813 is live with the unchanged hourly cadence and17 watched jobs; none of these17 is terminal in this check. This is a verified wait, not new clean evidence. No new receipt/submission/release/cancellation was performed.

Freeze SHA/tree, restore the guard runtime and remove only candidate-specific source/tests from the operator tree (recoverable in K's immutable commit). Construct/check the four-cell manifest; full/local checks do not grant ownership or native isolation. For future K bundle analysis, select this frozen candidate's six-path reader, not the restored baseline collector: raw `n_closed` is not K's verified primary metric. Run its isolated-reader test from the frozen source to confirm import origin.

The live hourly monitor PID2086813 was confirmed at22:47 UTC. Await eight new receipt-backed validator-clean cells before the next local Claude analysis window. OWNERSHIP FAIL and the11-node reservation still bar new submissions/releases; no running job is cancelled by this preparation.
