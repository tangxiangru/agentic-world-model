# Full study: agent-fitted world models for RPM on PTB

Status: proposed roadmap, 2026-09-06. This extends the [GPU baseline specification](2026-09-06-rpm-ptb-baseline.md). It does not claim new models have been trained, new GPU experiments have run, or any proposed variant has beaten RPM.

## 1. The claim we want to test

Given the same historical evidence and research budget, can an agent that fits a lightweight, experiment-specific predictor help RPM choose training experiments that produce better final checkpoints than inference-only LLM reasoning?

There are three separate questions:

1. **Measurement:** are differences between checkpoints stable enough to learn and act on?
2. **Predictor construction:** do comparable historical examples plus numerical fitting give better decision-relevant forecasts than direct LLM inference on the same evidence?
3. **Research utility:** does access to that predictor improve the actual research trajectory under a fixed GPU/time budget?

A lower MAE answers none of these by itself. Copying the parent remains a diagnostic control, not a successful experiment-selection policy. Better retrieval alone can also be useful, but does not establish that a fitted predictor is necessary.

## 2. The whole system

The scientist proposes 15 frozen, unexecuted recipes from a known parent. The RPM judge selects one to train. After execution, the trusted evaluator produces a measured checkpoint score, and the search continues. The main intervention is **which prediction/evidence tool the judge receives**, not a change to the scientist, training budget, parent selection, or evaluator.

| Role | Baseline | Agent-WM treatment |
|---|---|---|
| Scientist | Frozen Qwen3.6-27B; writes plans and code | Identical |
| Selection judge | Recommend Qwen3.6-27B; identity still to be confirmed | Identical |
| Cross-run historical evidence | Complete approved historical trajectories and clean label index | Exactly the same corpus and retrieval interface |
| New-run history | Already-executed checkpoints and permitted measurements | Identical access policy; realized histories diverge after different choices |
| Additional tool | None | Agent-fitted lightweight predictor |
| Training executor and evaluator | Fixed PTB adaptation | Identical |

The WM construction agent is also a frozen model, initially the same Qwen backbone. **We are not initially fine-tuning the 27B scientist or judge.** The learned weights are those of a small regression model fitted on historical experiment outcomes. The agent's fitting policy is developed and frozen separately from those numerical weights.

The main task models remain Gemma-3-4B-PT for GSM8K and Qwen3-4B-Base for AIME 2025. Historical evidence is selector-side, not a task-model training-data source. PTB integrity judges remain separate and active. This is a declared PTB-with-historical-evidence adaptation, not an unmodified leaderboard submission.

## 3. Data that constrains the design

Source: [HF trajectories](https://huggingface.co/datasets/JerrrrryL/awm-gsm8k-trajectories), proposed pinned revision `01406da734fb9016530bcdfeee027e3760587c6e`. The [proposed membership allowlist](../../data/analysis/wm_gpu_protocol/v1_proposed/historical_membership.json) combines the previous offline training and holdout cohorts for a **new prospective study**.

| Historical cohort | Total eligible targets | Measured-parent continuations | Published-base-reference starts |
|---|---:|---:|---:|
| GSM8K, 23 scientist sessions | 80 | 19 | 61 |
| AIME 2025, 44 scientist sessions | 105 | 25 | 80 |
| Total | 185 | 44 | 141 |

Thus 185 rows do not mean 185 independent continuation treatments. We have only 44 measured-parent continuation examples, with further correlation within sessions. This is a reason to use a small model and partial pooling, not to discard clean but difficult test cases.

Adopting this union retires the old 62-example holdout as validation for the new model. The alternative is to keep the old 123-example bank for every arm. Freeze one choice before new runs. Missing target/parent scores never become fabricated delta labels. Published base references retain their source-policy limitations; they are not newly repeated official measurements.

The historical labels are single-pass. Repeated new evaluations can improve measurement but do not retroactively make old labels precise. Reevaluate historical weights only if exact artifacts and compatible protocols are available, and apply any updated historical label package to both baseline evidence and WM training.

The new Qwen scientist also changes the distribution of proposed code and recipes relative to the historical scientist runs. Historical cross-validation is development evidence, not proof of transfer to Qwen-generated experiments.

## 4. What the agent-fitted WM actually does

For a query consisting of a parent state plus proposed treatment:

1. **Read prospective information.** Known parent accuracy and its measurement metadata; previous recipes; proposed dataset construction, method, code and hyperparameters; remaining budget. No proposed-child or proposed-intermediate outcome is available.
2. **Specify comparable evidence.** The agent chooses eligible historical examples and relevance weights using input-side similarity: benchmark/base family, parent state, training history, data source, objective, dose, filtering, replay, and teacher/self-generated data. In the initial comparability ablation, final training outcomes are hidden until this selection is committed.
3. **Choose a bounded fitting specification.** Depending on the variant, the estimator/target/features are fixed controls or selected from a small allowed menu. The agent submits IDs, weights, feature names, target, model family and justification—not its own numerical guess.
4. **Fit through a trusted numerical tool.** The tool loads only approved historical labels, performs training-only preprocessing, enforces support/weight limits, validates candidate specifications using grouped training validation, and serializes the fitted model.
5. **Return a numerical forecast with support diagnostics.** Predicted final accuracy, predicted delta, number of examples and distinct sessions, distances/effective support, uncertainty diagnostics, selected model/target, and whether a global fallback was used.
6. **Let RPM decide.** The judge may use or disregard the forecast together with its usual evidence. Log that decision. A useful WM must improve chosen experiments, not merely make plausible-looking numbers.

Fitting can happen once per candidate, with all 15 candidate forecasts cached before the tournament. Do not refit the same candidate to obtain a favorable answer in later matches. Cache keys include candidate/parent hashes, corpus version, feature schema, fitting policy, and allowed history cutoff. Agent-model inference and CPU fitting time both count as selection overhead.

### Initial model menu

- **Regularized linear regression (Ridge):** primary numerical estimator; small feature set, at most 4–8 selected recipe/history features plus parent/reference state and required missingness indicators. Use a tiny predeclared regularization grid, for example three values, not an unconstrained search.
- **Global plus local correction:** primary proposed extension for sparse comparable data. A global model uses all eligible same-benchmark historical rows; an agent-selected neighborhood learns a regularized correction.
- **Weighted-median delta:** robust diagnostic/control, not the assumed best model and not an estimate of the mean in every distribution.

Ridge's coefficient penalty is a straightforward small-model regularization mechanism; it does not guarantee out-of-distribution accuracy. [Ridge documentation](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

Do not start with a trained trajectory transformer, large embedding encoder, unconstrained AutoML, or a fine-tuned Qwen WM. They are outside the first small-data study. A shallow tree can be a later nonlinear check if grouped validation supports spending effort on it.

### Two prediction targets

- **Delta:** predict child mean accuracy minus parent mean accuracy, then add the known parent reference back.
- **Final accuracy:** predict child mean accuracy directly; derive the corresponding delta by subtracting the same parent reference.

First compare these as fixed variants with identical data/features. In the full agent-fit variant, the agent may propose either target, but the choice must pass historical training-only validation. A target change must not silently change which examples are eligible. Current historical observations are noisy proxies for the means; record that distinction.

### Partial pooling rather than automatic zero-delta fallback

The candidate design is:

`predicted_delta(query) = global_prediction(query) + support_weight(query) * local_correction(query)`

The global predictor still depends on the proposed recipe, so it can distinguish alternatives even when no close neighborhood exists. With strong comparable evidence, the local correction has more influence; with weak evidence it shrinks toward the global prediction, not automatically toward the parent's score. This is an untested design choice, not a guaranteed improvement.

If residual correction is used, produce historical global residuals with cross-fitted predictions. Fit the entire process without the outer held-out session. Support thresholds and shrinkage settings are fixed or selected only inside training validation. Do not force a minimum sample count by relabeling distant examples as comparable; use the broad model and disclose weak support instead. Tiny local sets can produce a regularized intercept adjustment rather than a many-parameter regression.

### Feature groups to test, not indiscriminately concatenate

| Group | Prospective inputs |
|---|---|
| Parent state | Accuracy/reference source, evaluation count/uncertainty when available, base/continuation status |
| Immediate treatment | Training objective, learning rate, examples/tokens/steps, epochs, batch, sequence cap, adaptation method |
| Data construction | Dataset identity and overlap, self/teacher/gold source, correctness selection, wrong-answer retention, replay, mixture proportions when actually specified |
| Previous training | Immediate-parent recipe plus compact summaries of earlier datasets, objectives and doses; retain raw recipes for agent inspection |
| State–treatment interactions | Prior exposure versus new dataset, repeated versus changed objective, current dose relative to previous dose |

Use fixed-size summaries and agent retrieval to handle variable trajectory depth. More steps need not mean an ever-growing numerical feature vector. Missing declarations stay unknown, not false. Code gives evidence of intended behavior, not proof that it executed. Actual generated-token counts, realized mixture fractions and post-training loss are future measurements, not pre-execution features unless a separately budgeted pilot has genuinely produced them.

## 5. Component experiments before expensive rollouts

Start with the same feature representation and fixed delta-Ridge estimator for G0–G2. This separates data selection from model/target flexibility.

| ID | Predictor or control | Question it answers |
|---|---|---|
| G0 | Global Ridge, all eligible same-benchmark history | Is a simple trained model sufficient? |
| G1 | Rule-selected comparable examples, same Ridge | Does state/treatment matching help? |
| G2 | Agent-selected examples/weights, same Ridge | Does the agent select more useful evidence than fixed matching? |
| G3 | Exactly G2's selection, global + local correction | Does borrowing broad evidence fix sparse-neighborhood failures? |
| G4 | Agent chooses features, target and model from the bounded menu | Does additional agent fitting judgment improve over the fixed choices? |
| L0 | Direct LLM forecast from exactly G2's selected examples/labels/weights | Is numerical fitting useful beyond selecting good demonstrations? |
| R0 | Inference-only RPM with access to all historical evidence | Actual decision baseline to beat |

L0 gets the same prospective feature context and selected labels, but no fitted forecast, coefficients, or agent rationale. It is a controlled fixed-packet comparison, distinct from R0's all-history retrieval policy. Use identical outcome-blind G2 selections for G2/G3/L0; do not independently choose different demonstrations and call that a fit-only ablation.

G1 and G2 must use the **same support gate and fallback rule**, initially the frozen G0 prediction when the local fit lacks support. Retain every outer-fold query and report support and fallback separately. For the controlled G2-versus-L0 estimator comparison, use the same G2 support gate and G0 fallback in both arms; this makes L0 an explicitly hybrid policy on unsupported queries, not a pure inference-only result there. Report the supported-only comparison as secondary alongside the complete common-policy results. R0 remains the actual full-history inference-only baseline, without numerical-model access.

G4 is a combined-system variant, not a clean single-factor ablation. Add its flexibility only after G0–G3 have been measured. Freeze a small finite set of fitting specifications and an explicit tie-break favoring the simpler one. If the full agent procedure is worse, retain the simpler model rather than declaring agent freedom intrinsically better.

Additional bounded ablations, run mainly on historical data and saved decision packets:

- Current recipe alone versus parent + current recipe versus full prior-recipe summaries.
- Structured recipe versus audited code-derived data/treatment features.
- Fixed delta versus fixed accuracy versus training-validated target choice.
- Numerical prediction only versus prediction plus uncertainty/support supplied to RPM.
- Agent-selected evidence summary without fitted predictions: a reasoning-effort control if the main system appears to help.

Do not launch a separate 10-hour GPU research study for every row or feature combination. This stage primarily uses CPU fits and bounded Qwen inference.

## 6. Validation and criteria for promotion

Use outer folds holding out complete historical scientist sessions; also group cross-session duplicate trained artifacts or shared trained lineage if found. Every outer query must reconstruct its legal pre-execution input and remove its session from fitting and raw-history retrieval.

Inside each outer training fold, rerun the entire procedure: matching, agent feature/model/target proposals, scaling, fitting, support/fallback decisions and any calibration. Adaptive choices use inner grouped validation only. Evaluating one fitted model after an agent saw all outer labels would not validate the agent policy. This follows the separation between model selection and evaluation in [nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Historical records do not provide verified simultaneous 15-candidate batches. Use their forecast errors and legal retrospective ranking diagnostics for development, explicitly marked as such. Do not invent choice sets from arbitrary stages and claim online regret. MAE, accuracy/delta R², support, and sign/pairwise diagnostics remain secondary; near-ties and unsupported cases must not disappear from the primary prospective comparison.

For the first promotion gate, prefer a simple, stable model over the best of a large noisy search. Check base starts and continuations separately; verify whether recipes from the same parent receive meaningfully different forecasts. A model that emits the same value for every proposed action is not a useful child selector even if its MAE is good.

Uncertainty should distinguish noisy score measurement from sparse/out-of-distribution training evidence. Estimated intervals or `P(delta > epsilon)` may be reported only with their assumptions and validation; do not call them calibrated without evidence. Never give zero-accuracy/zero-stderr historical records infinite fitting weight. When comparing candidates, account for their shared parent and training data rather than assuming independent forecast errors.

## 7. Run sequence and GPU allocation

### E0 — Measurement and infrastructure smoke

Run fixed checkpoint repeats, first on the bases and then a few accessible trained parents/children. Measure repeat variance and winner/sign flips, verify restored artifact hashes, and check evaluator/decoding compatibility. This estimates evaluation noise; repeated training seeds would be a distinct, more expensive test of recipe execution variability.

Verify all-candidate logging, selector isolation, strict preselection freeze, checkpoint retention before PTB cleanup, and private audit separation. The baseline document proposes online repeats of 1/4 and private final repeats of 8/32 for GSM8K/AIME respectively; treat them as feasibility-test starting values and freeze them before measured comparisons.

### E1 — History-informed RPM baseline collection (can start before the WM is ready)

Run Qwen RPM+History on fresh PTB runs. Proposed pilot: 8 independent runs per benchmark, 10 hours each. Archive all 15 candidate artifacts and every starting/produced checkpoint. This costs 160 allocated task-GPU hours, plus inference and private scoring.

Assign this batch **development** by default so its problems can inform the WM design. It cannot later be relabeled as untouched confirmation. If instead keeping it as test data, seal outcomes until model/agent policy freezing. Do not feed any new-run outcome into the historical bank during the main fixed-corpus study.

### E2 — Historical WM construction and component comparisons (parallel with E1)

Run G0–G4 and L0 under grouped historical validation, plus legal decision diagnostics. Fit the selected global model and freeze the agent fitting policy, feature schema, model menu, regularization/selection rules, corpus and prompts. Per-query fits can vary with the proposal after this freeze, but the algorithm determining them and the available training data cannot change after observing test outcomes.

### E3 — Fresh paired decision audit

From identical frozen Qwen-generated batches and parent checkpoints, ask R0 and the candidate WM-assisted selector to choose before any candidate outcome is exposed. Execute and repeatedly evaluate the distinct selected recipes from the same parent, using the same training-seed schedule and resource caps. A shared choice requires only one execution for the paired comparison; it remains one correlated observation, not two samples.

Primary diagnostic: paired mean-accuracy difference of the selected children, also reported as delta from the common parent, cost and execution failures. Compare against L0-derived advice or the extra-reasoning control where feasible. Recipe-specific training randomness is not eliminated merely by setting the same seed; a subset of independent training repeats can quantify that separately.

Retain failed branch executions in the decision-utility comparison: apply the same predeclared parent/incumbent fallback policy to both selectors, using that valid artifact's measured score, and separately report raw child-success coverage and failure cost. This is the outcome of the selection policy, **not** a parent-valued or zero-valued supervised label for the failed child. Do not report only successful execution pairs as overall policy performance. Missing evaluator measurements remain missing and follow the fixed infrastructure policy.

If auditing saved E1 states, call it baseline-state replay and prevent access to outcomes already recorded in the trajectory. If results were already used in development, the audit is development too. This is not a full WM rollout, and exact best-of-15 regret requires executing all 15. No savings claim should subtract unexecuted candidates' imagined cost.

### E4 — Actual end-to-end research comparison

After E2/E3 model selection, launch fresh independent runs with a frozen policy. Required arms:

| Arm | Additional capability beyond the shared scientist and history |
|---|---|
| B: RPM + History | Inference-only selection |
| A: RPM + History + agent-fitted WM | The selected G2/G3/G4 procedure, frozen before these runs |

Strong optional third arm: **RPM + History + global WM (G0)**. This tests whether query-specific agent fitting is worth its extra overhead. If fitting-specific benefit remains unclear, preregister a compute-matched agent-evidence/direct-LLM arm before looking at confirmation outcomes; do not infer a numerical-fitting mechanism from a two-arm policy win alone.

With 8 seeds per benchmark and 10 hours per run, B+A require 320 task-GPU hours; adding G0 makes 480. These are pilot-scale examples, not a power calculation. Use development run-level variability and a prechosen practical effect size to set the final confirmation sample size. Never count the 15 candidate proposals or repeated generations as independent research-run replications.

Primary endpoint: private repeated mean pass@1 of the precommitted final checkpoint after the fixed research budget. Report benchmarks separately, absolute percentage-point differences, uncertainty, and costs. Supporting endpoints: independently evaluated incumbent curves at fixed milestones, time to a preregistered target, improvement over base, failed/invalid executions, and selection overhead. Retain base/incumbent fallback and missing-measurement accounting without fabricating child labels.

Use the same initial seed blocks and resource settings, but expect policies to visit different parents and generate different later proposals. Keep parent selection and final submission rules fixed. For this first study, change only child-selection assistance.

### E5 — Continually updated WM (later, only if the fixed-corpus system is useful)

Compare historical-only fitting against a WM that can incorporate **its own run's already-completed, valid experiments** as additional labeled data. Both selectors receive the same available observations; only the treatment numerically updates its predictor. Freeze each prediction before the next outcome is revealed, and never share live outcomes across supposedly independent runs or arms.

This is a separately named online-learning study. Further directions—using the WM to choose parents, learning runtime/failure risk, exploring for information gain, or training the fitting agent itself—are not bundled into the initial child-selection comparison. Failure/risk models require trustworthy status labels and censoring treatment; ungraded old records cannot simply become zero-accuracy examples.

## 8. Total resources and what to implement now

Do not interpret “320 GPU hours” as the cost of the entire project. If E1 is development, its 160 hours are additional to E4's 320-hour two-arm comparison: **480 allocated task-GPU hours before measurement, inference, and paired-audit costs**. Adding a global-WM confirmation arm brings those planned E1+E4 allocations to 640 hours. E3 costs depend on the number of audited states, distinct selected recipes and execution caps; budget them separately. A four-seed-per-task pilot halves the rollout allocations, not the auxiliary costs.

Implement now:

- The baseline GPU loop and an optional prediction-tool interface, disabled for B.
- Immutable candidate/parent snapshots, actual execution lineage, per-question seeded evaluation records, exact corpus and model identities, and time/cost accounting.
- A predictor result schema with candidate ID, mean accuracy/delta forecasts, support/uncertainty status, fit provenance and optional costs—not an interface that requires every tool to return a confident number.
- Independent private scoring and a clear development-versus-test run registry.

Then the WM can be added without changing the scientist or rebuilding the data collection. The scientific conclusion we seek is **“agent-fitted prediction helps choose better research actions at equal budget”**, with component evidence showing whether the benefit comes from data selection, fitting, uncertainty, or simply additional LLM reasoning.
