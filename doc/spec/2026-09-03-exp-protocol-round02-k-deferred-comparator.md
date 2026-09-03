# Round02 K — an explicitly deferred comparator, verified at close

Status: **implemented and CPU/independent-forward validated; awaiting immutable manifest; no receipt or launch**. Direction #27 from the strict-guard [planner decision](../exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/planner-decision.md). K is one coordinated optional comparator-lifecycle feature, baseline-relative and independent of J, H, C or E. Ownership/native-isolation gates remain closed.

## Evidence and the problem to solve

Five strict guard cells (g01s02/03/06/07/08) attempted within-card comparisons whose comparator file would be produced by that card. The current preflight requires the file before lock; scientists used recorded overrides or launched before locking. See the [source audit](../exp_protocol_iterations/trace-reviews/round01-strict-guard-addendum/g01s07-ordering-cost-audit.md) and synthesis Check4. This is a dependency conflict, not an unavoidable instruction to violate ordering: a reasoned override already exists.

Source inspection at `5fccd50` establishes why simply returning WARN for a missing file is insufficient:

- `preflight.comparator_same_protocol` checks existing output and an explicit n when present; missing output FAILs. Legacy JSON without n is accepted as unverifiable, not proof of matching protocol.
- CLI `close` runs result-schema and lock-integrity checks, **not comparator preflight again**.
- Index, starting points, collect and the standard-library-only Stop hook treat a populated conclusion as closed/adopted. A refused `close` does not erase that YAML conclusion.
- Harvested cards retain absolute sandbox paths; large evaluator logs can be omitted by the bundle cap. A portable, bounded verification receipt is needed for later readers rather than trying to reopen `/home/ben/task/...` on the operator host.

Hypothesis: explicit prelaunch deferral plus a verified completion receipt removes this false preflight dependency without accepting a missing, mismatched or unverified comparison as a successful result.

## Opt-in contract; legacy cards unchanged

Add optional boolean `evaluation.comparator.defer_validation`. Omitted/false retains existing behavior. A true value requires an absolute planned comparator output path, a nonempty comparator ref, `value: null`, a positive `evaluation.protocol.n`, and a named `hypothesis.expected_effect.metric`. No global required field or schema-version bump.

The flag means “validate this planned comparator at close,” not “the measurement already exists.” Keep these sections0–4 unchanged after lock. Read the eventual score from the output/verified receipt and put actual observations in sections5–6; do not backfill a guessed prelaunch value or flip the flag to mark completion.

### Preflight

- Missing planned file with valid opt-in: WARN/deferred, not a measurement PASS. Normal checks still run and must pass or have a legitimate recorded override before any model execution.
- If the file already exists, validate it immediately; do not hide a known mismatch behind deferral. It will still be validated again at close.
- Non-opt-in cards preserve the original check's behavior. K does not relax unrelated checks, change J's launch scope or solve H's data-field requirements.

### Successful paired experiment

At close, a completed opt-in card requires a readable, nonempty JSON comparator report with:

1. an **actual positive integer sample count** equal to the locked n;
2. a finite numeric value for the locked metric;
3. no reported failed/cancelled/incomplete evaluation state or contradictory count evidence.

Support a minimal actual-evaluation summary (`n` or `num_samples`, plus the named metric), and real Inspect reports (`results.total_samples`, completion/scoring counts and the named score metric; sample-list length when present). Every supplied actual count must agree; incomplete/scored-subset evidence must not be ignored. A requested `limit`, dataset population size, file-size heuristic or accuracy/SE inversion is not proof of actual n. In the retained g01s07 smoke report, dataset size1319 coexists with **8 evaluated/completed/scored samples**; validate8, not1319.

When a standard PTB accuracy/stderr summary omits n, preserve the successful raw evaluator report at the planned path or derive a minimal summary from its actual completed metadata, retaining the source log for audit. Do not edit the official grader or invent n to satisfy a check. Unknown/malformed/missing n or metric blocks successful closure. Dataset/seed/decode/model identity still require the existing manual same-protocol verification; K must state exactly what the machine checked and must not claim full protocol identity from n alone.

Completed cards must also record a finite target measurement for the locked metric/n with an evidence path; known recorded target-count mismatch must not be certified. Other-n probes do not substitute for that measurement. This is record consistency, not an independent audit of the target file or an assertion of full experiment validity.

### Failed, killed or unrun experiment

Such a card must still be closable without a nonexistent comparator: require an inconclusive verdict, a non-adopt decision, and an explicit failure reason for failed/killed execution (a reasoned summary for not_run). Preserve partial observations but do not certify a comparison or claim supported/contradicted/adopted success. This closes the failed experiment honestly; it is not a route to a “verified” comparison.

## Completion receipt and consumers

Record the opt-in declaration in the lock so removing a YAML flag cannot silently opt out. Once locked, the same card cannot drop this requirement via relock; use a new card for that contract change. Preserve declarations in relock history when applicable; changing a declared comparison after seeing its outcome remains an audited mode change, not a successful K exposure.

For opt-in cards only, successful CLI close writes a small `exp-NN.comparator.json` receipt **after both result validation and lock integrity succeed**, then seals its content hash in an additive lock field without altering the frozen plan. It binds the card/plan hashes, lock identity, planned comparator path/n/metric, completion classification, observed n/value and comparator content hash when verified. A refused/interrupted close must not leave a usable sealed receipt. A failed/inconclusive close gets a distinguishable failed/unverified-comparison receipt, never a verified metric certificate.

Use one standard-library-only receipt checker shared by the runtime API and hook, packaged with the hook so standalone installs work. It must reject missing, malformed, stale or mismatched receipts. A card changed after close invalidates its receipt; changing an available evidence file invalidates its hash. A matching receipt remains historical evidence when the original sandbox path is unavailable after harvest—report that distinction and do not pretend the raw file was re-read. Verify relocated retained artifacts when available. This is reproducibility evidence, not a security boundary against arbitrary filesystem tampering.

For opt-in cards:

- `close` validates and emits the receipt;
- index/starting-point selection must not treat an uncertified adoption as a valid starting point;
- collect must expose raw conclusion counts separately from verified/deferred outcomes, retaining the old raw columns for comparison;
- the Stop hook treats an uncertified opted-in card as unresolved, but preserves its existing12-block cap and does not change waiting guidance. A truthful failed close resolves it; manual YAML alone is not a verified comparison.

Legacy cards, their raw statistics, manual-close behavior and hook responses remain unchanged. Do not turn K into a general closure-state redesign. Operator analysis must use the candidate-capable receipt reader for K bundles; a legacy collector's raw conclusion count is not the new primary metric.

## Planned code surfaces and tests

One feature may require coordinated code in comparator validation, schema, preflight, lock/close, lineage/collect and the hook, plus the optional-field template and rule2 guidance. Keep the receipt checker free of YAML/third-party dependencies for the hook; avoid duplicated validation implementations. No change to training/evaluation harness, model, budget, waiting policy, data applicability or generic legacy comparator semantics.

Before freezing, test the complete lifecycle and its consumers—not just a permissive preflight:

- valid missing opt-in output locks as deferred; missing legacy output still FAILs;
- valid later summary/Inspect output closes and is recognized through index, starting points, collect and hook;
- wrong/unknown n, dataset-size-vs-evaluated-n confusion, partial counts, failed Inspect state, missing/nonfinite metric, unreadable/non-JSON output and stale/edited receipt all fail closed;
- changing the card, comparator path, mode or locked plan cannot reuse an old receipt;
- failed/killed/unrun inconclusive non-adopt cards close honestly; no false supported/adopt route;
- copied/harvested cards retain portable verification without silently reinterpreting missing raw artifacts as freshly checked evidence;
- every existing legacy test remains unchanged in behavior; direct/installed hook use both work and the12-block bound remains.

Run independent forward review after implementation with real CLI-shaped toy fixtures and no GPU/model execution. Preserve tested source identity with the candidate's iteration record; restore baseline runtime and candidate-specific tests after freezing.

## Four-cell screen and operational boundary

Planned batch `exp-protocol-gsm8k-gemma4b-high-r02-k-deferred-comparator-x4-v1`, cells `k02r01–04`, run_index1, same Round02 GSM8K/Gemma4B/Opus5-high/1M/10h/PTB/images contract and six shipped paths. Freeze SHA/tree only after implementation/tests. Construct from guard drift `2f64581`, not J or E.

Primary among exercised within-card comparisons: false missing-output prelaunch failures/overrides and pre-lock evaluator starts must fall to0, while **zero invalid or unknown completed comparisons are certified**. Require exposure in at least2/4 cells to judge the mechanism; no exposure is not success. Track adoption of deferral, failed-closure handling, abandoned diagnostics, time/copy overhead, fabricated metadata and every mode-changing relock. Treat bypasses as failures of the intervention, not compliance.

Use the same fixed24-cell reference pool and score floor **0.6737212534748547** documented in the [J spec](2026-09-03-exp-protocol-round02-j-lock-scope.md); it is a coarse guardrail, not a causal comparator. A winner needs a new second four-cell block and held-out confirmation before promotion. Do not add K to the scheduler while ownership/native isolation is unresolved; the existing held buffer is sufficient. No running job is cancelled.
