# P: budgeted checkpoint inventory and structurally paired evidence

2026-09-04. **Implementable design, not implemented P, an experiment, a manifest or release authorization.** Applies to complete E+P and E+L+P packages on the identical E core. The operator checkout's user-directed `skills/exp_protocol_meta/search_policy.md` is authoritative; no concurrent analysis draft is copied or modified. Two discovery cells and decision-priced continuation are outer-search policy, not an automatic scientist recipe or winner-to-eight rule.

## 1. Evidence checked for this design

The operator source root is `/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`. Its `doc/exp_protocol_iterations/analysis-2026-09-04-opportunity-review/` contains the source scripts and inventories below. They are analysis evidence, never scientist input assets.

| Source checked | Actual finding | Interface consequence |
|---|---|---|
| `high-case/audit_eval_pairs.py`, `developer-eval-inventory.json`, `paired-summary.json`; original g01r03 A/B/soup Inspect files rehashed and structurally joined in this review | Version2 Inspect JSON; dataset population1319, selected IDs800, completed/scored800; sample-array order differs from declared dataset order. Triple counts reproduced exactly:00086,00115,01012,01133,10014,10132,11017,111591; zero input/target differences. | Import genuine Inspect structure, not a bespoke test-only summary. Align typed IDs/epoch; expose eight-way parent/combination strata and explicit prefix order. |
| `counterexamples/audit.py`, `evals.json`, `pairs.json`; original g01r05 exp04/exp07 n1319 files rehashed and rejoined | Each994 correct, but917 both correct,77 each unique,248 both wrong; zero input/target differences. | Scalar ties are not identical capabilities; preserve rejected branches and exact fixes/breaks within a declared budget. No automatic keep-all or oracle deployment. |
| `counterexamples/cross-contract.json` | g03/g05 common800 input messages agree after removing volatile message UUIDs; cap4000 matches, concurrency32/16 and GPU fraction.80/.85 differ. Historical intermediate generation configs are absent. | Pairability, serving equality, artifact identity and causal interpretation are separate axes; unknown is not equal. |
| `high-case.md`, `high-case/data-summary.json` and L design |253045/350000 rows represent70315/69221 exact questions and93118 shared exact question/solution pairs; later corpus is not a superset. Only the final artifact survives in retained raw storage. | Attach actual L census/overlap to lineage; track retention promises and actual existence, not rows/filename as a proxy for diversity or availability. |
| Window04 `control-b-structural-pairs-report.md` and its CPU artifacts in the operator audit | c01s04 b70/b160 is58/51 of1319, not219/211; c01s07 g2/g3 is57/43 of500, not74/60. Shared500 final contrast changes token cap1024→2048 and n500→1000. | Structural joins, uniqueness and numerator reconciliation are acceptance tests. No regex pairing, dropped-record ±1 excuse, or changed-serving contrast called isolated noise. |

Raw paths for the real replay fixtures are resolved by `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/<cell>/status.json:result_dir`, then:

| Cell / model | Relative to raw result | SHA256 verified here |
|---|---|---|
| g01r03 / parent A | `task/logs/2026-09-04T02-08-28+00-00_gsm8k_ciJ3uHNVASqWiKqTmLazts.json` | `d7faffda79168331493f70f9c6eed07f0cf4f667af1d15f06de73da9f3f642f0` |
| g01r03 / parent B | `task/logs/2026-09-04T02-10-43+00-00_gsm8k_SXGDYKSPJyH3gr9Jf8KvKV.json` | `3238168ca23a463ae92ff46b32c100254466428f17182bcb18314f750b75560d` |
| g01r03 / measured soup | `task/logs/2026-09-04T02-35-39+00-00_gsm8k_fsTJmjcbhriNjK59F8PVF4.json` | `c488814af00c4352f934a77b1cfa2abaf2dcddf059dd777b5cc7b70982b8b032` |
| g01r05 / exp04 | `task/logs/2026-09-04T00-34-10+00-00_gsm8k_L7KLmYyDQ4sYfriGprNCaH.json` | `854bb8343d8281452a6579736b45fd619a0955ae96af3d11e98aa64cde08f5ef` |
| g01r05 / exp07 | `task/logs/2026-09-04T02-22-01+00-00_gsm8k_WYDiZMvjkQChNPoDGxcVYh.json` | `03b1f13327af85cd76b31fa133a24d5f20a7d859b084109dac065b0cb100bee2` |

No held-out task or new model execution was accessed. These developer evaluations use benchmark test items; inspecting them for operator audit does not permit exporting their rows into scientist training or watch sets. The known oracle counts and real-log expected outputs below remain operator-side CPU regressions only.

## 2. Two modules, explicit evidence levels

Proposed shipped modules: `awm.exp_protocol.branches` and `awm.exp_protocol.eval_evidence`, standard-library CPU paths by default. Names below are proposed public APIs, not existing commands. Neither module imports/constructs a model, runs an evaluator, selects a recipe, schedules work, or deletes/moves checkpoints.

Common `EvidenceRef` envelope: `{schema_version, kind, path, sha256, producer_version, scope, bindings}`. `scope` includes session/task, dataset purpose and visibility. Loaders are allowlisted by schema; a receipt must not cause arbitrary code imports. A hash proves byte identity, not semantic truth. Every result has separate statuses for `source_availability`, `structure`, `counts`, `item_alignment`, `prompt_identity`, `serving_identity`, `artifact_binding`, and `selection_context`; there is no universal `valid:true` shortcut.

Paths are locators, never artifact IDs. Resolve explicit session/root scope; reject traversal/special files and do not follow unexpected links. New evidence outputs use unique no-clobber directories, raw-source hashes, adapter/schema version and an after-read source recheck. Interrupted/oversized/changed sources produce incomplete evidence and preserved diagnostics. Use explicit finite resource ceilings (bytes/files/rows/CPU time); exceeded bounds return `unsupported_or_incomplete`, never a smaller unnoticed denominator. Optional missing evidence remains unknown, not a false block on ordinary E-only cards.

## 3. Checkpoint inventory: budget, promise, existence and lineage

```python
plan = BranchPlan.create(
    session_dir, task_id, reference_id,
    budget=BranchBudget(storage_bytes=..., retained_slots=...,
                        planned_model_seconds=..., observation_io_bytes=...),
    branches=[BranchIntent(branch_id=..., reason=..., future_use=...,
                           estimated_bytes=..., estimated_model_seconds=...)],
    output=new_plan_dir,
)
inventory = CheckpointInventory.open(session_dir, plan_ref=plan.ref)
node = inventory.register(
    branch_id, checkpoint_path,
    producer=ProducerRef(card_ref, lock_ref, attempt_ref, checkpoint_step),
    parents=[LineageEdge(parent_node, relation="trained_from", evidence=...)],
    serving_identity_ref=e8_snapshot_ref,  # optional; missing stays unknown
)
snapshot = inventory.observe(node_ids, output=new_snapshot_dir)  # read-only artifacts
inventory.record_intent(node_id, action="retain_for", reason=..., until=...,
                        estimated_next_cost=..., evidence_refs=[...])
```

`record_intent` appends an immutable scientist-authored decision; it does not execute it. `future_use` names a specific unresolved comparison, recovery dependency, promised handoff or export—not generic “diversity.” Explicit changes/reasons supersede earlier intents without editing history. An expired intent is a review warning, not deletion authority.

Required inventory projection:

- **Instance identity:** unique checkpoint-node ID, producing session/card/lock/attempt, declared output slot, observed step/time, all source refs; separate full-serving content identity and weight-file identity when available. Reused `final_model` paths produce new observations/instances. Equal weights with different decoding remain distinct serving variants. Identical bytes do not prove identical provenance.
- **Lineage:** directed typed edges `trained_from`, `merged_from`, `copied_from`, `serving_variant_of`. Combination edges bind all parents and declared method/coefficients/script hash. Reject cycles/conflicting explicit identities. Preserve `declared`, `observed_execution` and `content_verified` levels; observing a soup command is not verifying parameter arithmetic. Unknown parents stay unresolved rather than fabricated from filenames.
- **Promises:** read locked `setup.checkpoints.keep`, declared checkpoint policy and recorded kept outputs; map them to explicit nodes/slots. `all` protects every known produced checkpoint; unresolved `best`/`last` promises remain unresolved/protected until the recorded selection or step evidence identifies them. A score ranking must not silently resolve “best.” Missing promised artifacts are visible violations, not dropped inventory rows.
- **Existence:** `present`, `missing`, `changed_since_identity`, `unsupported_link_layout`, `unreadable`, plus timestamp. A historical receipt with absent weights is `historically_attested/unavailable`, not a currently reusable checkpoint. A fresh stat/snapshot cannot retroactively prove bytes measured earlier.
- **Cost:** logical file bytes, known unique-inode bytes, observed storage footprint with coverage, reservations and declared future model time. Do not infer physical CoW/reflink savings or content equality from same size. Charge retained incumbent/stage/backup evidence to storage accounting when in scope; protect E8 recovery backups. No optimizer-state copy merely to preserve a serving artifact.
- **Budget:** show planned/reserved, observed and unknown costs separately; no negative “remaining” hidden as zero. E5 process-wall intervals, E6 native-call wall, L observation overhead and training-runtime counters are different measurements. Do not sum overlapping intervals or rename them GPU compute. Budget overruns/missing estimates prompt an explicit scientist/planner decision within existing authority, not a wait for human input or automatic cancellation/deletion/ranking. Deletion or an expansion of external authority still requires separate authorization.

The initial inventory accepts explicit node paths only, with bounded scans of those directories. It does not crawl user homes, caches, queues or arbitrary shared runs. Unsupported symlinked historical serving copies can remain **listed** without obtaining E8 byte-valid status. P performs no eviction API, recursive delete, automatic export, or “keep top-k” policy. An operator/user may separately authorize artifact lifecycle changes; those are not implied by this tool.

## 4. Evaluator evidence: real Inspect import and exact joins

```python
evidence = import_inspect(
    inspect_json, scorer="match", metric="accuracy",
    dataset_scope=DatasetScope(...), measurement_ref=optional_invocation_ref,
    output=new_evidence_dir, limits=ReadLimits(...),
)
paired = pair_evaluations(
    a_ref, b_ref, cohort=DeclaredPrefix(reference=a_ref, n=500),
    comparison="descriptive_contrast", output=new_pair_dir,
)
strata = compare_combination(
    parent_refs=[a_ref, b_ref], combination_ref=c_ref,
    cohort=ExactSharedCohort(...), output=new_strata_dir,
)
```

### Initial supported import profile

Support real **Inspect JSON log version2**, specifically the observed `inspect_evals/gsm8k` text-only single-generate `match` binary-accuracy profile (`C`/`I`) and its explicit package/scorer identity. Profile dispatch inspects actual structure, not filenames. Other scorers/tasks/versions can be retained as unsupported evidence, but require an adapter before binary paired inference. Minimal `{n,accuracy}` summaries retain E3's existing limited role and **cannot** fabricate item rows. `.eval` ZIP/streaming formats, multimodal/tool-rich transcripts and arbitrary custom reductions are outside the first importer, not silently coerced into this profile.

Parse JSON structurally with duplicate-key and nonfinite-number rejection. Read:

| Real field | Separate normalized observation |
|---|---|
| `eval.config.limit`, `eval.config.epochs` | Requested selection/repetition; absent/unlimited remains distinct from an observed count. |
| `eval.dataset.samples` | Dataset population; never evaluated n. |
| `eval.dataset.sample_ids`, `shuffled` | Ordered declared selection and its provenance. May describe selected rows, not the full population. |
| `status`, `stats.started_at/completed_at`, sample errors/retries | Reported evaluation outcome/time, not OS producer-exit proof. |
| `results.total_samples/completed_samples`; selected `results.scores[].scored_samples/unscored_samples` | Actual aggregate count claims, independently reconciled with row records. |
| `samples[].id`, `epoch`, `scores[scorer].value` | Typed item/epoch key and actual scoring status/value; missing score is unknown, never wrong by default. |
| `samples[].input/target`; `events[event=model].input/config/output` | Original task inputs/targets, actual recorded request messages/configs, finish/token observations with coverage. |
| `eval.task/task_args/scorers/packages/model/model_args/model_generate_config`; `plan` | Requested/logged protocol and model locator, not automatically effective internal engine state. |

IDs are explicitly typed strings or integers; bool/null/composite IDs are unsupported for certified pairing initially, not coerced to strings. Epoch must be a positive integer explicitly present; do not copy historical scripts' unconditional `get(epoch,1)` fallback. Reject duplicate `(typed_id,epoch)` keys. Preserve every input occurrence/error in import diagnostics; never overwrite a dictionary entry silently. E6's repeated source IDs are legitimate **sampling occurrences**, not automatically unique evaluation items (section7).

Reconcile declared ID multiplicity/order, actual row count, unique-key count, completed/scored/unscored metadata, score-value vocabulary and aggregate accuracy. Resolve the requested scorer/metric unambiguously; conflicting same-n metrics fail count/metric certification. Partial/error logs remain importable with explicit status, missing/failed counts and no complete-comparison status. Do not derive n from bytes, standard error, requested limit or `dataset.samples`. A complete n800 report remains n800 even if a derived comparison uses500 of its rows.

Exact keys join rows regardless of stored sample-array order. Cohort modes are explicit: `exact_declared` (same selected IDs), `declared_prefix(reference,n)` (reference's declared order verified first), or `intersection` (report both populations, missing-on-each-side and the resulting n). No implicit intersection followed by a claimed full-test denominator. Record cohort digest/selection rule before using its results; a retrospectively chosen slice is labeled retrospective. Fail unknown prefix order rather than slicing `samples[:n]`.

Canonical hashes preserve input/target types, text, whitespace and message roles/content/source. Only profile-declared volatile **message UUID/ID fields** may be excluded, with both full and normalized hashes retained. Do not drop system messages, few-shot content, semantic IDs, source fields or earlier turns. Missing actual model-event inputs is missing prompt coverage, not proof of equality from a template filename. Unsupported referenced assets stay unverified.

Comparison result retains axes, not a lossy PASS:

- Input or target differences invalidate same-item correctness comparison; report mismatch counts/locators under the appropriate visibility policy. Excluding them requires a separately explicit common-valid cohort; never silently remove them from n.
- Different prompts/serving settings may be intentional diagnostic contrasts; preserve aligned outcome counts, differences and `changed_contract` status, not pure-weight or isolated-noise claims. Strict same-contract mode refuses unresolved/different required fields, but descriptive mode remains useful with unknown/config-change caveats.
- Separate **declared settings**, **logged per-model-event settings**, **observed native token/finish fields**, and **resolved engine state**. Per-item heterogeneous settings remain heterogeneity, not a single representative dict. Missing historical generation JSON is unknown even when the final exported file is greedy.
- E8 before/after artifact evidence, generation-file SHA, evaluator invocation and source hash can establish supported measurement binding. Snapshotting a current path after the evaluation is only post-hoc evidence. A model path string, equal score, same script or successful save is insufficient. Even before/after equality does not rule out unobserved external ABA mutation; exclusive/quiescent use remains a contract, not a security proof.

## 5. Parent and measured-combination strata

For A versus B retain both-correct/A-only/B-only/both-wrong, paired n, missing/unscored counts, metric numerators, delta orientation and all identity caveats. For parents A/B and an **actually evaluated** C, retain all eight `(A_correct,B_correct,C_correct)` counts and the four parent-status strata:

| Parent stratum | Required C observation |
|---|---|
| both correct | preserved correct versus **broken both-correct** |
| only A correct | kept A's solution versus lost it |
| only B correct | kept B's solution versus lost it |
| both wrong | **fixed both-wrong** versus remained wrong |

Also report C fixes/breaks against **each** parent, with common n and rates. Parent correctness agreement is not answer-text agreement; if normalized extracted answers are compared, name the extractor and retain ambiguous/missing coverage. The eight-way table itself supports arbitrary measured combinations, not only weight soup. Lineage/recipe identity determines whether “combination of these parents” is declared/observed/verified; a third unrelated model is labeled as such.

`oracle_union_correct = both_correct + A_only + B_only` is gold-conditioned analysis, with `deployable:false`, `official_score:false`, no artifact node and no selection/export method. C may fix cases outside the parent union and break cases inside it; do not equate measured C with the union or call the union an upper bound on all possible future models. It is only the oracle parent-output-selection ceiling on this cohort.

For g03's real800, required replay output is parent strata608/46/45/101, C-correct591/32/33/15; C breaks17 both-correct and repairs15 both-wrong. Parent union699 is not measured soup671 or official1092/1319. For g05 exp04/exp07, both994 does not erase77/77 discordance. Never infer the missing official g03 tail as `(1092−671)/519`: different invocations are not disjoint pieces of one evaluation.

Single-epoch binary paired comparisons may report exact two-sided McNemar/binomial and a clearly labeled item-conditional delta interval. A null test is not equivalence, a tie-break is not misconduct, and an adaptive selection-set interval is not confirmation/generalization evidence. Multiple epochs are retained and summarized per epoch and by unique item; **disable naive pooled significance by default**. A future clustered-by-item inferential adapter needs a predeclared reduction/estimand, not treating repeats as independent items. Selection multiplicity and cohort reuse remain visible.

## 6. Selection versus independent confirmation

Add immutable `EvaluationUse` events: `exploration`, `selection`, `confirmation`, `retrospective_diagnostic`, each binding report/cohort/artifact/decision refs and recorded time. A pre-evaluation `MeasurementPlan` can declare artifact/decoder, cohort, tolerances, intended use and prior selection exclusions. Later evidence references that plan; it cannot backdate it.

`confirmation` has separate `declared` and `verified_scope` states. Verify the planned artifact/contract/cohort, independence of invocation and any promised nonoverlap with prior selection cohorts; report unknown prior exposure rather than assuming it absent. Another invocation on the same items is an artifact-repeat check, not independent selection-set confirmation. Newly added suffix items that influenced selection are also not independent confirmation. No helper spends a repeat or allocates the reserved held-out task. Quality tolerance/promotion authority stay with predeclared outer policy.

P stores a selection record naming a **real inventory node**, exact evaluator evidence and selected generation SHA, the scientist's reason/constraints and unresolved caveats. It does not rank automatically or require the largest scalar. A lower-score branch can be retained for a named, budgeted question without calling that branch a winner or equal-quality model.

## 7. Real E/L/P consumers and safety boundary

| Producer / existing contract | P consumer and required behavior |
|---|---|
| **E3** `comparator.check_output/write_completion`, shared `hooks/comparator_receipt.py` | P importer may create a separate no-clobber minimal summary from a **complete whole evaluator report** with actual scored n/finite metric plus original path/hash. E3 retains its existing count-only close receipt; P's stronger item/artifact statuses are separate. A derived prefix500 of an actual n800 run is not silently relabeled an n500 E3 evaluator. Portable E3 receipts without source logs remain historical count attestation, never reconstructed pair rows. |
| **E5** locked `run` attempt/finish records | Inventory registration binds output/checkpoint instances to the producing attempt; optional fresh-directory hashes contribute current-invocation file observations. Missing finish keeps producer outcome unknown; no rerun or CPU→GPU transformation. `begin_measurement`/`finish_measurement` observational hooks surround the declared evaluator inside the existing covered command; they never launch it themselves. |
| **E6** `prepare_prompts`, `record_vllm`, `parse_recording`; `awm-sampling-record-v1` | Import raw native token/finish diagnostics by verified request/capture/raw/parse hashes. Key sampling draws by source occurrence ordinal + typed source ID + completion index, keeping parser status and requested n distinct. Do not reinterpret draw count as scored item n. A future developer-score adapter must explicitly bind gold source, scorer and draw-reduction policy before producing evaluator-shaped evidence; an E6 parser summary is neither Inspect accuracy nor official scoring. |
| **E8 implemented API** `snapshot_serving_artifact`, `verify_serving_artifact`, `publish_serving_artifact` | Reuse `awm-serving-artifact-v1` content identity/file sizes/profile and selected generation hash; never invent a weaker P model-valid flag. Inventory lists availability; selection refs feed E8's explicit expected identity/generation arguments. Publication journal feeds a new location/instance event and backup accounting. P never calls publication implicitly. E8 API is present; its public-guide/independent-forward acceptance is **not yet complete** at this design point. |
| **L section4** census and numerical references, bound to E7/source/card/attempt | `attach_observation(node_id, L_ref)` loads an allowlisted versioned L record, checks bytes and matching card/attempt/E7/data identities, and retains declared versus observed relation. Missing historical weights need not erase a valid data observation, but do not confer current checkpoint usability. Numerical parameter samples and optimizer-state dtypes retain partial coverage/step-or-interval labels. |

Implement the L→P consumer, not just a `misc_refs` list:

```python
inventory.attach_observation(parent_a, census_a_ref)
inventory.attach_observation(parent_b, census_b_ref)
inventory.attach_observation(parent_a, numerical_update_a_ref)
report = explain_branches(
    inventory_snapshot_ref, parent_nodes=[parent_a, parent_b], combination_node=c,
    paired_ref=measured_strata_ref,
    observations=[census_a_ref, census_b_ref, numerical_update_a_ref],
    view="scientist_aggregate", output=new_joint_report,
)
```

The joint report computes actual **question/solution overlap** from compatible L fingerprints/sets, distinguishing same questions/different solutions, raw/considered/kept occurrences, duplicates as training weight and rendered versus metadata prefixes. It displays the named parents' budget/retention/existence, paired fixes/breaks, measured combination and numerical observation coverage. Incompatible question/solution view versions yield overlap unknown; no rerendering a randomized/aliased historical row to invent L coverage. Updating the L receipt or node binding invalidates the joint report rather than silently preserving an old conclusion. Training-distribution/update differences do not establish why paired accuracy changed.

Avoid a receipt/plan hash cycle: prepare branch plan before card lock; the optional `setup.branch_plan:{path,sha256}` is checked and pinned by the real E+P preflight/lock consumer. Post-run inventory, observations, measurement/selection records bind back to the lock. Their refs may be recorded in existing result evidence/measurement paths; any new optional `result.research_evidence` field must have actual index/collect validation before shipping. Do not change required v2 fields or retroactively fill sections0–4. E-only behavior stays unchanged; mandatory future semantics require explicit migration.

**Benchmark-test data firewall:** import classifies observed Inspect benchmark-task test logs as `benchmark_test`; unknown provenance defaults to restricted, never scientist-dev. Operator-only normalized rows may retain typed IDs/input/target/output for audit, with explicit restricted paths. Scientist-facing summaries expose aggregate counts/strata/config/lineage only—**no test IDs, ID hashes usable as row selectors, questions, targets, answers or per-item failure examples**. P offers no test-derived training/watch-set/router-label export. Any E7/L training-source loader must reject refs carrying benchmark-test/restricted ancestry, and joint L/P reports cannot pass test rows into census or training adapters. Legal independent train/dev diagnostics require explicit source-backed purpose; caller labels alone cannot reclassify an observed benchmark log. This is an enforced supported-API dataflow boundary, not a promise to prevent arbitrary filesystem/Bash bypass.

## 8. CPU acceptance and implementation order

No new GPU run is needed to construct/test these interfaces. Use small synthetic **real-Inspect-shaped** JSON fixtures for everyday tests; separately run operator-side source-hash-checked golden replays on the original logs above. If originals are absent, report replay skipped/unavailable, never a passed synthetic replacement. Do not copy benchmark questions/answers into shipped skills/examples or scientist fixtures.

| Test group | Required negative and combined cases |
|---|---|
| Real importer | Actual version2 nested `results.scores[].metrics`; unordered `samples`; population1319/limit800/scored800; duplicate JSON keys; partial errors; absent/contradictory scored n; multiple scorer ambiguity; nonfinite metrics; extra/missing IDs; int7 versus string"7"; bool/null IDs; absent epoch; duplicate ID+epoch; repeated epochs with no pooled p. |
| Pair/strata | g03 triple counts and g05 77/77 golden replays; prefix from declared IDs, never array order; changed targets/inputs/prompts; message UUID-only changes versus semantic source changes; differing n/config/token caps; unknown decoder/weights; explicit intersection coverage; identical scalar/different sets; missing C does not manufacture combination score; oracle never becomes an artifact. |
| Inventory | Kept promises with missing files; unresolved best/last; expired intent; reused paths; content aliases/hardlinks; unsupported symlinks; lineage cycles/unknown parents; finite-budget overrun; immutable decisions; changed content; fresh namespace versus valid model; E8 retained backup charged/protected; no deletion or ranking side effects. |
| Measurement/selection | Before/after E8 binding and changed source; post-hoc snapshot remains unknown; ordinary historical comparator still closes under E3 rules while P says item identity unavailable; false confirmation on reused selection IDs refused; changed token cap not isolated repeat noise; subset summary not forged whole-run comparator n. |
| **E+L+P actual interaction** | Two independently materialized E7+L training views with equal question sets/different solution sets and duplicate weights, attached to two explicit retained artifacts, plus a recorded measured combination. Joint consumer recomputes compatible overlap and outputs parent/C strata, promises/budget and update coverage. Stale L/E7/card/attempt refs, conflicting lineage, missing artifacts, absent update coverage and observation from another attempt must remain visible. Test-derived IDs/content cannot enter either L training view. |
| Failure/cost | Hash/source mutation during read; no-clobber output; interrupted import and bounded I/O; oversized logs/unknown versions; missing portable sources; CPU latency/peak-memory/output-size measurements. Preserve errors/evidence; never launch inference, install a decoder, or rerun a failed evaluator. |

Construction sequence: (1) common refs/importer + golden shape/count tests; (2) inventory and E8/E5 bindings; (3) exact pair/strata and privacy views; (4) L-reference verifier/overlap consumer for E+L+P; (5) E3/index/collect and independent public-guide forward review. Implement P on a separate branch/package after the E core is frozen; keep the same E identity under E+P and E+L+P. If L is unavailable, steps1–3 plus the E+P scope of step5 can complete and validate E+P itself; E+P is not blocked on L implementation. This does not complete E+L+P: that package stays deferred until step4 and its joint step5 review are exercised with real E7/L outputs, not merely matching field names.

Acceptance reports whole outcomes: supported comparison/identity coverage, false blocks, retained promised artifacts, budget/cost, reused branches, measured combinations, resolved questions and resulting scientist actions. No weighted scalar composite, diversity-for-its-own-sake score, retrospective metric winner, automatic repeat or promotion follows. This document resolves implementation defaults; it requests no new scientific or scheduler authority.

### Design-source snapshots

- Operator search policy SHA256: `f462808e97aea0a382cedd3fb41e1f9688683533a308c3ef326b44d370ff6fa0`.
- Operator high-case pair script: `d7f07ff4641d18ced9078df6dd208d2e5c8ad8dfcbaf776001c1cae8e82c6bca`; counterexample audit: `9638dc45e7f42a02a041be5b5e4ef73ed02f418eca1e971abf57df1434ea9dd2`. These case-specific scripts motivate but are not imported as generic runtime adapters.
- Candidate discovery spec: `fbcfa38a71529a74bbba80f7d996b7e83ac49ec167454f1c2b24e26a871c66c3`.
- Candidate L design initially read: `fe2283c034bf688c71ad18a040aba9f90ec0ad6503203456daedccc8330416f7`; after main accepted the E7 implementer's raw-before-render/finalize-after-verify/overlap/step-observation clarifications: `5f02a91fb456ce2043e790a13ad3ceeddcad97ca4904e94c5386aaf5175440e4`. Section4 is consumed explicitly above.
- E8 source initially inspected: `awm/exp_protocol/serving_artifacts.py`, SHA256 `c4dd6050d3b868dfd64255a2d739b948e646d58301ba5e0f8da53bfd08a88f9a`. At E construction acceptance the final helper is `13e5befeb8177a3e0dd1e0289903677388392b4799651b72d38c014e3168174f`; main's328-test pinned run and independent E8 forward pass cover that source. This later acceptance supersedes the pending status at the design-time API table above; see `2026-09-04-bundle-construction.md`.
