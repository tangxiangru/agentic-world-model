# Recipe-to-checkpoint prediction experiments

This offline study asks whether registered experiment recipes and their ancestor
cards predict official GSM8K checkpoint accuracy. It does not change the deployed
`awm` runtime or implement its `PredictorAgent` stub.

Run from the repository root, using the downloaded private dataset:

```bash
uv pip install --python .venv/bin/python -r tools/outcome_prediction/requirements.txt
.venv/bin/python tools/outcome_prediction/build_examples.py
.venv/bin/python tools/outcome_prediction/benchmark.py \
  --examples data/analysis/outcome_prediction/examples.jsonl \
  --output-dir data/analysis/outcome_prediction/grouped --permutations 50
.venv/bin/python tools/outcome_prediction/sensitivity.py
.venv/bin/python tools/outcome_prediction/decision_tasks.py
.venv/bin/python -m pytest tests/test_outcome_prediction.py -o addopts='' -q
```

The report, audit, predictions, and model metrics live under the Git-ignored
`data/analysis/outcome_prediction/`. Keep generated examples and prompts there:
they derive from a private trajectory release. `build_examples.py` records
exclusions, manual lineage corrections, and the limits of recipe reconstruction.

## Input boundary

The primary cohort uses the first submitted plan, excluding later registrations,
changed canonical recipes, known mismatched targets, and pure re-evaluations.
Features contain whitelisted data sources/counts, scalar hyperparameters,
canonical package/version identifiers, merge coefficients when available, and
preceding recipe cards. Paths, scientist identity, results, conclusions, script
snapshots, and free-text observations are excluded from primary learned features.
Rich setup prose is a separate exploratory arm; it can mention earlier outcomes.

All train/test partitions hold out entire scientist runs. Vectorizers, imputers,
scalers, and regressors fit inside training partitions. Nested regression chooses
regularization using inner group splits. Classification and change-prediction
tasks distinguish known pre-plan scores from retrospective official labels.

“Complete lineage” means resolved producing-card dependencies, not a complete
executable checkpoint recipe: some intermediate snapshots, selection procedures,
and data-generation dependencies remain unrecorded. The experiment only evaluates
labeled surviving checkpoints; missing labels are never replaced with zero.

## Blinded in-context experiment

`prepare_icl.py --fold N --output-dir DIR` creates a scientist-stratified split
using a fixed hash order, independent of target scores. Each split holds out four
runs from each scientist and uses the other 24 runs as training context.

For each split, a **new context** prediction agent was allowed to read only
`DIR/test.jsonl` and then `DIR/train.jsonl`. Its task was:

1. Read test recipes alone and write `zero_shot_predictions.jsonl`, with one
   `{id, prediction, rationale}` object per target. Freeze this file.
2. Read training recipes and their official accuracies. Use those examples as
   in-context learning signal; write `few_shot_predictions.jsonl` for the same IDs.
3. Never inspect hidden labels, source data, reports, other folds, or other agents;
   never fit a statistical model or invoke another model/API.

Prediction agents inherited the session model; their API model identifier was
not independently pinned. Predictions are retained for exact rescoring, not a
claim of bit-identical LLM regeneration. Three separate sessions completed 107
test checkpoints over 24 disjoint runs. A fourth split was prepared but not run
because the session reached its agent-thread limit; it is excluded from results.

To score completed prediction files:

```bash
.venv/bin/python tools/outcome_prediction/benchmark.py \
  --examples data/analysis/outcome_prediction/examples.jsonl \
  --split-json data/analysis/outcome_prediction/icl/split.json \
  --output-dir data/analysis/outcome_prediction/icl/tabular_holdout
.venv/bin/python tools/outcome_prediction/evaluate_icl.py
# Repeat same-split baselines and scoring with --directory for icl_fold1 and icl_fold2.
.venv/bin/python tools/outcome_prediction/aggregate_icl.py
.venv/bin/python tools/outcome_prediction/plot_results.py
```

Evaluation protocol follows [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
and [training-only preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).
Bootstrap intervals resample runs; they are descriptive and do not account fully
for overlapping CV training sets, model comparison, or LLM generation variability.

## RPM-style same-parent preference benchmark

The follow-up adapts [AI Research Preference Models](https://arxiv.org/abs/2608.13940)
to **immediate checkpoint accuracy**, not future descendant-subtree performance.
There are 89 non-merge, complete-dependency, same-parent pairs with at least a
one-percentage-point score gap, drawn from 25 runs (77 pairs / 22 runs at two points).
These are retrospectively paired, adaptively generated recipes; no verified
simultaneous candidate batches exist in this release. Do not infer online compute
savings from selection accuracy or regret.

```bash
# Freeze canonical candidates, earlier-plan history, and eight whole-run folds.
.venv/bin/python tools/outcome_prediction/rpm_judge.py prepare
# CPU-only learned rankers; all preprocessing and tuning stay inside training folds.
.venv/bin/python tools/outcome_prediction/rpm_rankers.py \
  --folds-json data/analysis/rpm/judge/folds.json
# Optional paid inference, using the local authenticated Claude CLI.
# Inspect the frozen inputs first. Each job is a fresh tool-disabled session.
.venv/bin/python tools/outcome_prediction/rpm_judge.py run \
  --workers 4 --call-budget 1.5 --total-budget 60
.venv/bin/python tools/outcome_prediction/rpm_compare.py
```

Exploratory follow-up runs used the same frozen folds:

```bash
.venv/bin/python tools/outcome_prediction/rpm_rankers.py \
  --folds-json data/analysis/rpm/judge/folds.json --train-siblings-only \
  --methods fixed_recipe_numeric_logistic_C1 exploratory_recipe_numeric_forest exploratory_lineage_numeric_forest \
  --output-dir data/analysis/rpm/learned_siblings
.venv/bin/python tools/outcome_prediction/rpm_rankers.py \
  --folds-json data/analysis/rpm/judge/folds.json --train-siblings-only \
  --methods exploratory_recipe_numeric_contextual_forest exploratory_lineage_numeric_contextual_forest \
  --output-dir data/analysis/rpm/learned_contextual
# Optional order/repeat audit: 20 additional paid calls on ten fixed pairs.
.venv/bin/python tools/outcome_prediction/rpm_swap.py prepare
.venv/bin/python tools/outcome_prediction/rpm_judge.py run \
  --arms within_run_swap cross_run_swap --total-budget 65 --call-budget 1.5
.venv/bin/python tools/outcome_prediction/rpm_swap.py score
.venv/bin/python tools/outcome_prediction/rpm_compare.py
.venv/bin/python tools/outcome_prediction/rpm_verify.py
```

`within_run` receives only canonical candidate lineages and local measurements
recorded strictly before the earlier proposal. `cross_run` additionally receives
the entire matching outer-training bank and anonymous run-group membership.
Repeated historical recipes are losslessly interned into a catalog for inference;
no examples or recipe fields are dropped. The frozen judge has more local context
than the learned rankers, so this is a method comparison with the same labeled
bank, not an exact feature-matched experiment isolating weight training.

Current candidates' scores, full narrative plans, post-execution code snapshots,
actual run identifiers and scientist identities never enter judge prompts. One
process handles one pair, preventing another pair's history from disclosing its
candidate scores. Prompt inputs, hashes, CLI responses, costs and model metadata
are preserved under `data/analysis/rpm/`. Do not publish those private-data inputs.

Learned rankers use antisymmetric pair comparisons. The initial run learns from
all within-run pairs; `--train-siblings-only` restricts both training comparisons
and inner model selection to same-parent pairs. `--methods NAME ...` restricts the
model families. Sibling-only and contextual-forest follow-ups are exploratory,
motivated by initial results; all older outputs remain preserved. The original
fixed C1 classifier and its predeclared 50/50 probability blend with the cross-run
judge remain separate from those follow-ups.

Inference budget checks reserve per-call allowances before dispatch and count
all arms across resume. CLI budget limits are not a hard preflight token-price
guarantee; a single completed API request can exceed its allowance. Unknown-cost
failures retain their reservation. Existing outputs are never automatically
retried, and changing the model or frozen prompt on resume is rejected.

The discarded v1 pilot had a historical metric-field mismatch. Its artifacts are
retained separately under `data/analysis/rpm/pilot/`; they are not part of the v2
comparison. The regression suite includes the real recorder's `metric` schema.

```bash
.venv/bin/python -m pytest tests/test_outcome_prediction.py \
  tests/test_rpm_judge.py tests/test_rpm_history_regression.py \
  tests/test_rpm_rankers.py tests/test_rpm_compare.py -q
```

## RPM-style pairwise setting (arXiv 2608.13940 analog)

The AI Research Preference Model paper ranks unexecuted sibling candidates with a
frozen LLM judge that sees each candidate's plan and code plus scored history nodes
from the current search tree. `rpm_judge.py` is an early, limited adaptation of
that decision, not a reproduction: same-parent sibling pairs of
executed checkpoints (|Δ official accuracy| ≥ 1pt), candidate scores hidden,
history limited to local scores recorded before the earlier candidate's plan, whole
runs held out, and one fresh isolated CLI process per pair.

- `within_run` (v2): canonical recipes + scored current-run history — the inference-only judge.
- `cross_run` (v2): the same plus the whole labeled prior-run bank in context.
- `code_within_run` (legacy v3, `rpm_judge_code.py`): v2 within-run inputs plus each candidate's
  plan-stage structured `setup` and **mutable archived script snapshots, not verified
  launch-time code**. Sibling pairs are retrospective and the later plan usually names the
  earlier one, so free prose stays excluded and every mention of the other candidate
  (card id, output dirs) or accuracy-looking number is redacted; `tests/test_rpm_judge_code.py`
  guards the redactor. Subsequent auditing found that this redactor can also remove
  valid hyperparameters, while the inherited prompt incorrectly says code is
  unavailable. Preserve these outputs as a legacy exploratory arm; do not treat
  them as a faithful RPM baseline or leakage-cleared pre-execution code.

```bash
.venv/bin/python tools/outcome_prediction/rpm_judge_code.py prepare   # from data/analysis/rpm/judge
.venv/bin/python tools/outcome_prediction/rpm_judge_code.py run --workers 4 --total-budget 30
```

Learned and retrieval rankers trained on the other runs' outcomes (`rpm_rankers.py`,
`data/analysis/rpm/learned*/`) are scored on the identical pairs and folds so the
comparison is paired at the run level. Results: `data/analysis/rpm/judge_code/report.md`.

## Audited richer RPM baseline

`rpm_faithful.py` uses the paper-listed Claude Opus 4.8 at maximum reasoning effort,
with the Figure 7 rubric explicitly adapted from eventual subtree-best performance
to the user's immediate-checkpoint target. It keeps the paper's reasoned boxed A/B
output; these hard choices are not reported as calibrated probabilities.

Candidate code comes from successful, timestamped Write/Edit tool results before
first registration (`rpm_code_provenance.py`), never the mutable final snapshots.
Candidate plans retain safe earliest problem/hypothesis/setup/evaluation sections.
Known previous checkpoints carry scores, plans, and recovered code; weight ancestry
and data provenance are separate. Official predecessor scores are explicitly
assumed known retrospectively, not claimed to have existed at proposal time.

The original 89-pair cohort shrinks after checking actual data dependencies, code
availability, and explicit/implicit outcome leakage. Section-level removals and
manual semantic exclusions are recorded separately. This is still a retrospective
adaptive-pair study, not an exact simultaneous-branch RPM reproduction.

Two untuned learned comparators use the same rich per-test packet, train-only
TF-IDF, antisymmetric pair features, equal-run weights, and the original whole-run
outer folds. They additionally learn from other runs' labeled pairs; the frozen
judge does not receive that bank. Thus overall training information differs by
design, even though per-test evidence is shared. Earlier learned methods remain
exploratory references, not freshly confirmed model selections.

```bash
.venv/bin/python -m tools.outcome_prediction.rpm_code_provenance
.venv/bin/python -m tools.outcome_prediction.rpm_faithful prepare
.venv/bin/python -m tools.outcome_prediction.rpm_faithful fit
.venv/bin/python -m tools.outcome_prediction.rpm_faithful run --total-budget 25
.venv/bin/python -m tools.outcome_prediction.rpm_faithful redecode
.venv/bin/python -m tools.outcome_prediction.rpm_faithful score
```

Artifacts: `data/analysis/rpm/faithful/`. Preparation refuses to overwrite a cohort
with judgments. Prompts, model choice, input hashes, model settings, and eight
hash-selected A/B reversal checks are frozen before judging. The preserved
`code_provenance_manifest_used.json` records the exact reconstruction manifest
used even if later source formatting changes the helper's source hash. Invalid
calls are not silently retried; the report separates prepared, attempted, valid,
and missing judgments. Do not publish private trajectory/code inputs.

The `redecode` step fixes a presentation-only parsing issue: a response can quote
literal `\\boxed{}` from the supplied code while still giving exactly one valid
final boxed A/B answer. Quoted Markdown code is ignored for answer counting;
multiple actual answers still fail. Derived verdicts live in `decoded_outputs/`;
the original raw outputs are immutable and no model calls are repeated.
