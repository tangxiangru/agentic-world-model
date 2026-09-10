# Statistical design for a 1,000-cell evaluation matrix

Status: proposal, not an approved spending plan or a claim of statistical power. No model API calls, GPU evaluation jobs, or training runs were launched for this design.

## Recommendation

Finish the clean launch-script/config inputs and run the serving-feature ablation on the existing 579 eligible labels first. If that supports further measurement, favor roughly 400 distinct trained weight sets with two or three shared serving policies over 200 weight sets with many policies. The objective is recipe prediction, so breadth of recipes and independent scientist sessions matters more than a large decoding sweep on a few checkpoints.

Ten repeated benchmark passes estimate a cell's mean accuracy; they do not turn that cell into ten independent training examples. Nor does evaluating an existing checkpoint under new policies prove that we can predict a genuinely untrained recipe.

## 1. State which prediction problem the matrix tests

| Claim | Required test | What this matrix can establish |
|---|---|---|
| Predict another serving policy for known weights | Hold out policies; explicitly declare whether other scores for those same weights are permitted inputs | A useful serving-response predictor, but not unseen-recipe prediction |
| Predict performance of a recipe represented by an unseen session | Hold out entire session/learned-weight groups, including every serving policy and all old scores | Retrospective cross-session prediction; prospectively frozen new-policy labels strengthen this, with the exposure caveat below |
| Decide whether to train a genuinely new recipe | Freeze its scripts, data specification and serving policy before training; evaluate newly trained weights afterward | Not established by re-serving old weights, even if all 1,000 cells are new evaluations |

Primary proposed input: the permitted training launch script(s), exact launch arguments and referenced configuration/data specification, plus the explicitly specified effective serving configuration. If chronological earlier training launches are part of the user's requested prefix, retain that training-only prefix consistently; do not silently substitute a different lineage definition. Keep outcomes, evaluation scripts/transcripts, final-card conclusions and broad unrelated code history out of the predictor input. Names that encode measured scores or later experimental order need sanitation too.

The parent-score/prior-best features in the older design document define a different, score-assisted task. They must not enter the primary no-score-at-inference comparison. A separately labeled score-assisted continuation experiment is possible, but cannot substitute for this test.

An archived generation configuration may have been chosen after the scientist observed performance. Attaching it to a historical training recipe is legitimate for predicting that archived **recipe plus serving policy**, but does not prove the policy could have been selected before training. New matrix policies should be fixed independently of the new outcomes.

## 2. What the existing corpus actually supports

Audit inputs: recorder manifest/results at `data/traj/raw/awm-gsm8k-trajectories-cc2ac9d884a7/`; label validator `tools/outcome_prediction/prefix_labels.py`; pinned checkpoint generation metadata at HF revision `446127629d7b271d537390e69bfb2d960a3aa515`. This is the eligible recorder cohort, not all 1,048 uploaded metadata directories.

There are **579 eligible labeled records in 124 scientist sessions**: GSM8K 326 records/63 sessions; AIME 253 records/61 sessions. The three existing checkpoint/recipe-binding quarantines remain excluded: `r0-25-exp-02`, `aime-r0-11-exp-02`, `aime2-r0-12-exp-05`. Numerical validity does not release those quarantines. Record counts are not verified unique-weight counts.

For run r, let accuracy be a_r = correct_r / N_r. The target is

`mean_pass_rate = (a_1 + ... + a_10) / 10`.

With the required identical question set, this equals total correct / (10 N). It is **not pass@10**, which asks whether at least one attempt succeeds per question. A missing or failed pass is not zero and is not silently dropped. Unequal question counts are invalid under this fixed-benchmark protocol, even though an unweighted average and a separately reported pooled rate can be defined mathematically.

The fresh audit validates ten runs and the per-problem binary matrices. The following SD is the population SD of the ten run accuracies. The nominal SE uses the sample SD divided by sqrt(10), assuming independent repeats. All quantities below are percentage points.

| Cohort | Records | Sessions | Median run SD | Median nominal SE of mean | p90 nominal SE |
|---|---:|---:|---:|---:|---:|
| GSM8K, all eligible | 326 | 63 | 0.708 | 0.236 | 0.389 |
| AIME, all eligible | 253 | 61 | 3.667 | 1.222 | 1.699 |
| GSM8K, raw JSON temperature = 0 | 111 | 24 | 0.356 | 0.119 | 0.158 |
| AIME, raw JSON temperature = 0 | 10 | 5 | 3.465 | 1.155 | 1.539 |

Every explicit-temperature-zero row in those last two groups has nonzero variation across its ten run accuracies. Therefore, do not assume that an archived temperature-zero JSON makes all ten evaluations deterministic. These are **raw JSON categories**, not a verified effective-greedy classification. Runtime overrides/defaults, evaluator interpretation, backend behavior and metadata binding must be checked before attributing that variation to a particular cause. There are seven AIME rows with identical run accuracies overall and none on GSM8K; identical accuracy would not by itself prove identical generated text.

Ten passes appear adequate for estimating coarse GSM8K differences under this protocol. They are not a guarantee that a 1 pp AIME configuration difference is resolvable: its typical single-cell nominal SE is already about 1.2 pp. For two independent cells with that SE, the difference has approximately 1.7 pp SE; use the actual paired covariance if the evaluation supports valid pairing. Do not infer a universal recipe noise floor or a causal sampling-versus-greedy benefit from the historical group means.

Repeated passes use the same 1,319 GSM8K questions or 30 AIME questions. They reduce generation randomness, not uncertainty about generalizing to new questions. AIME's ten-pass mean changes in steps of 1/300, or 0.333 pp. Preserve per-question/run results, seeds and process identifiers so run dependence, question uncertainty and config contrasts can be audited separately. Fixed-benchmark prediction and generalization to future questions are different claims.

Machine-readable audit: [label_noise_audit.json](label_noise_audit.json).

## 3. Historical results are development evidence, not a new success criterion

External analysis `data/analysis/wm_exp_designs/POC_ANSWER.md` contains sequential analyses; its RPM v2 session-disjoint results supersede optimistic earlier agent comparisons. On 93 pairs from 49 sessions, pairwise accuracy was 0.591 for recipes/scripts-only agents, 0.645 with raw corpus retrieval, 0.602 with prediction tools, and 0.677 for the GBM. The session-bootstrap interval for raw retrieval minus recipes-only was [-0.05, +0.17]; tools minus recipes-only was [-0.10, +0.13]. Those comparisons do not establish an agent improvement. Same-session batching and access to early scoring reports were concrete leakage problems. The later-sibling baseline reached 0.720, but exploits adaptive search and is not a pre-training recipe predictor.

The external earlier LOSO report (`data/analysis/wm_exp_designs/poc_cross_session_table_v2/report.md`) found from-base GBM MAE 9.73 pp GSM8K versus 14.70 for the mean, and 4.69 pp AIME versus 6.59. Continuation gains over parent-unchanged were not established: GSM8K 5.51 versus 5.44 pp; AIME 3.41 versus 4.65, with an interval crossing zero. These are useful reference scales, not directly comparable benchmarks: the cohort differed, serving was incompletely represented, and the feature protocol included parent score, prior-best-in-session and card index.

The old `first_exp/split.json` test set contains 31 sessions, overlapping 81 currently eligible GSM8K rows in 16 sessions and 71 AIME rows in 15 sessions. Its outcomes have already been inspected. The RPM tests and manually inspected twins are also development evidence. No old split should be relabeled “untouched” simply by reshuffling it.

## 4. Zero-GPU prerequisite and allocation

Before buying any new cells:

1. Finish the launch-only input selection, exact config binding and semantic leakage review. Existing broad-prefix outputs are not automatically model-ready. Record missing data contents explicitly; planned builder rules are not future realized dataset statistics.
2. Resolve effective generation settings from the actual evaluation path, including backend defaults/overrides, output budget, stop/EOS handling and tokenizer identity. Raw-file byte equality is neither necessary nor sufficient for equal serving behavior.
3. On identical eligible rows, compare benchmark mean, serving-only, training/data-only, and combined training/data+serving predictors. Use grouped nested cross-validation, with a small fixed model menu: regularized linear model, constrained trees/GBM, and nearest comparable regression. Historical outcome-bearing features stay out.
4. Run the matched-evidence inference-only LLM comparator described below. Do not launch 1,000 cells merely because the combined predictor explains benchmark-level or serving-policy-level means; it must add information beyond those baselines.

If the clean existing-data ablation has no promising incremental signal, pause the large allocation. New controlled cells can still be justified to answer a serving-mechanism question, but that is a narrower reason than demonstrated recipe-prediction feasibility.

Provisional broad allocation supplied during this design review:

| Block | Distinct trained weight sets | Shared/extra policies each | Cells |
|---|---:|---:|---:|
| GSM8K core | 240 | 2 | 480 |
| AIME core | 160 | 3 | 480 |
| Targeted diagnostic extensions on 20 of those AIME weights | No new weights | 2 extra | 40 |
| Total | 400 | — | 1,000 |

The intended recipe mix is all 313 eligible from-base records (190 GSM8K, 123 AIME) plus 87 diverse weight-changing continuations. This is conditional on verifying recoverable, genuinely distinct weight sets and recipe provenance. A duplicate decode/export record does not fill another distinct-weight slot. Prefer diversity across sessions, training methods and data regimes; do not select continuations only because their existing scores are extreme. Record any outcome-informed diagnostic selection separately.

This allocation is preferable to 200 checkpoints × five policies for the primary recipe question: it doubles candidate weight-set coverage while retaining within-checkpoint contrasts. A larger factorial design is preferable only if the primary question changes to detailed serving interactions. Both allocations are bounded by the much smaller number of independent scientist sessions. Select a small, predeclared set of effective policies; do not factorially cross every raw JSON key, including irrelevant/inactive fields.

Inventory overlap with the old evaluated weights × effective-policy cells before estimating incremental cost. Reusing a protocol-compatible old ten-pass label saves evaluation work; rerunning that cell gives a replication, not a previously unobserved configuration outcome. Neither can be presented as a fresh counterfactual test label. Report requested matrix cells, reused labels, genuinely new cells and extra replication jobs separately.

Use the 960 shared-policy cells for the primary cross-recipe metric. The 40 targeted diagnostics are secondary; oversampling unusual failures must not silently change the headline target population. Main-effect decoding contrasts should compare the same weights and compatible evaluation contracts. A policy that changes several settings yields a policy-bundle effect, not separate causal estimates for each field.

Staging: reserve whole-session groups first; run roughly 100 development pilot cells within the 1,000-cell budget; then the remaining development cells; open the locked final cells only after freezing the predictor. Around 200 locked cells is a planning target, not a reason to split a session. Pilot settings, independence checks, retries and artifact availability must pass before scaling. Exact cell counts should follow the group assignment rather than override it.

## 5. Splits and honest scope of the final test

The indivisible split unit is a scientist session joined with any verified duplicate/derived learned-weight components that cross sessions. All configs of a checkpoint stay together for the cross-recipe claim. Shared public base-model initialization does **not** connect every experiment into one giant group; shared learned checkpoint artifacts and descendants do. Unknown duplicate bindings are quarantined or conservatively grouped, not assumed independent.

Report the final connected-group count as well as the 124 original sessions; cross-session artifact reuse can reduce the effective number of groups.

Development: five outer session-group folds, with grouped inner folds for feature choices, hyperparameters, retrieval rules, calibration and LLM prompt/model selection. Fit imputers and any target-dependent feature selection on the inner training data only. Report out-of-fold predictions and session-cluster uncertainty; repeated CV seeds are sensitivity checks, not additional independent samples.

Final set: select whole session components before observing their new matrix outcomes; remove their old native scores, all same-weight policies, relevant learned-parent/descendant outcomes and their trajectories' outcome-bearing text from training and retrieval. Keep test labels inaccessible to predictor agents and feature extractors. Run final evaluation once after freezing code, prompts, splits and exclusion hashes. No shared agent context may carry a scored related packet into a test prediction.

Even with those controls, previously inspected recipe sessions are not genuinely untouched. A defensible label is **“prospectively frozen new-configuration outcomes on historically inspected recipes.”** This tests prediction of new counterfactual serving outcomes, with disclosed researcher exposure. Truly untouched recipe generalization requires exposure-audited new sessions and predictions fixed before those training outcomes. That requires additional authority and possibly new training; it is not silently included in this evaluation-only plan.

Use that label only for policies not already evaluated on those weights. Report any reserved native-policy replications separately. Additional newly uploaded result files are not automatically a fresh in-population test: missing launch artifacts or a different benchmark/base-model combination require a separate eligibility audit. In particular, the root's new-upload inventory found Qwen-based GSM8K trajectories; they should not be silently mixed into this Gemma/GSM8K and Qwen/AIME population to manufacture an untouched holdout. Fresh same-population PTB sessions remain an optional later study.

## 6. Metrics, comparisons and inference-only LLM

Primary metric, separately per benchmark: absolute error in pp, averaged first over the shared policies for a checkpoint, then over checkpoints within a session, then equally over sessions. This prevents sessions with many checkpoints or diagnostic policies from dominating. Also report ordinary cell-weighted MAE, median/p90 error, R² and performance on from-base versus continuation records. State the deployment population; equal-session weighting is a deliberate estimand, not a universal default.

Choose the strongest simple comparator on development folds, not on the final test. Necessary ablations are serving-only, training/data-only and combined. The question is whether combined evidence adds useful recipe-level information, not whether it beats a weak benchmark mean.

For decision utility, report regret when choosing among predefined candidate policies for the same unseen checkpoint, and separately among comparable training recipes. Report uncertainty/tie coverage; do not call every small AIME gap a reliable winner. “Best observed config” is a noisy maximum, so its apparent regret is upward-biased: use an independent assessment split of repeats for secondary selection analysis, or explicitly report the noisy-oracle limitation. Do not treat counterfactual policies as independent recipe choices.

Fair inference-only LLM comparison:

- Freeze model version, prompt, tool budget and any training-only calibration before final evaluation.
- Give it exactly the same permitted launch/config/data evidence. No target, parent or same-session benchmark score is supplied in the primary prediction input.
- Provide the same TRAIN-only labeled examples/retrieval corpus available for learning the statistical predictor. “Inference-only” means no parameter fitting to those examples, not deprivation of historical training evidence. Report a no-corpus LLM separately if desired.
- Exclude all reserved session/weight groups structurally, not by asking the model not to look. Do not provide raw traces containing measurements or let tools access the repository's test reports. Use isolated contexts and audit actual reads.
- Require numeric predictions and score the same rows, including missing-evidence and difficult cases. LLM abstention is reported with coverage and a declared fallback, not dropped from MAE.

Confidence intervals compare predictors **paired on the same heldout sessions**. Bootstrap complete session/weight components; preserve all their checkpoint/config observations together. A second question-level analysis is needed if claiming transfer to new benchmark questions. Cluster intervals do not remove prior researcher exposure or adaptive feature-selection bias.

## 7. How much holdout is enough, and when to stop?

Count heldout sessions, not 200 cells or 2,000 passes. As a planning approximation, the 95% CI half-width for a mean paired session error difference is about `1.96 × SD_session_difference / sqrt(S)`; use a small-sample t multiplier or cluster bootstrap in the actual report. The unknown SD must be estimated from development out-of-fold errors. Illustrative values below are **not empirical power estimates**.

| Heldout sessions per benchmark | Half-width if session-difference SD = 3 pp | If SD = 6 pp |
|---|---:|---:|
| 20 | 1.31 pp | 2.63 pp |
| 30 | 1.07 pp | 2.15 pp |
| 40 | 0.93 pp | 1.86 pp |
| 60 | 0.76 pp | 1.52 pp |

At SD = 3 pp, roughly 35 independent sessions give a normal-approximation half-width of 1 pp; at SD = 6 pp, roughly 139 are needed. This is a precision calculation, not a power guarantee. With only 63/61 eligible sessions total, reserving 20–30 per benchmark is already a substantial tradeoff against training breadth. A 200-cell reservation may not provide that many whole sessions once all their mandatory from-base checkpoints are included. Prefer a sufficiently broad grouped test over an exact round cell quota, and call a wide interval inconclusive rather than proof of no signal.

Proposed gates, requiring user agreement before spending or a final success claim:

1. **Provenance/label gate:** every scored cell has verified weight identity, frozen effective generation/evaluator/tokenizer settings, the fixed question set and ten complete validated passes. Failures remain explicit; no imputation. Resolve the observed temperature-zero variability before assuming greedy repetition saves measurement work.
2. **Development gate:** combined training/data+serving evidence improves on serving-only and on the training-only representation under grouped nested CV, without relying on measured-history leakage. If it does not, pause and diagnose before the full run.
3. **Practical final gain proposal:** at least 10% relative MAE reduction over the preselected strongest simple baseline, and at least 1 pp GSM8K / 0.5 pp AIME absolute reduction. These thresholds reflect the current historical error scale, not a user-agreed utility requirement. Require a favorable paired uncertainty interval for a strong claim; if making two benchmark success claims, predeclare multiplicity handling (for example Holm adjustment). Report the matched LLM comparison alongside it; beating the LLM alone is insufficient if a simple baseline is stronger.
4. **Outcome:** distinguish “useful on benchmark X,” “promising but imprecise,” and “no demonstrated incremental value.” Do not use one positive subgroup, a high R² alone, or selection among many tried thresholds to declare overall feasibility. No repeated peeking at the final set to decide whether to continue.

The stopping decision should precede expensive scale-up: a clean offline ablation, then a development-only operational pilot, then a frozen test. The matrix can resolve missing serving information and improve retrospective prediction. A later, separate prospective training-recipe test is still needed for the original “which experiments should we train?” claim.
