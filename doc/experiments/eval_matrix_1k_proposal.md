# Proposed 1,000-combination study: can recipe + serving predict performance?

Status: **concrete proposal, not launch-ready or authorized to execute**. The accompanying list contains 1,000 named `exp_id`s and seven explicit generation-policy files. No model evaluations or training runs were launched. Existing `wm_exp_designs.md` is unchanged.

The supporting files are now tracked in the [portable execution bundle](../../experiments/eval_matrix_1k/README.md). Start with its [execution-agent handoff](../../experiments/eval_matrix_1k/AGENT_HANDOFF.md): static validation and small-asset fetching are implemented; the GPU executor still needs implementation and preflight before launching the 100-combination pilot.

## Recommendation

Use broad checkpoint coverage and a small, corpus-supported set of serving policies. The primary question is: **given training-launch code, data/config specifications, and a prescribed serving policy, can we predict accuracy for a checkpoint from another scientist session without seeing that checkpoint's outcomes?**

| Allocation | Candidate weight sets | Policies per checkpoint | Combinations |
|---|---:|---:|---:|
| GSM8K / Gemma-3-4B core | 240 | 2 | 480 |
| AIME 2025 / Qwen3-4B core | 160 | 3 | 480 |
| AIME extensions on 20 of those checkpoints | No additional weights | 2 additional | 40 |
| Total | 400 | | **1,000** |

Every combination gets ten complete benchmark passes if evaluated afresh: 10,000 benchmark passes and 6,487,200 question completions. This is a maximum target list, not a requirement to spend the entire budget. Reuse demonstrably equivalent existing ten-pass labels, and do not automatically fill any savings with additional jobs.

The 960 shared-policy cells are the primary prediction dataset. The 40 extensions are secondary diagnostics, not extra independent training recipes.

## Why these checkpoints?

The current audited PTB cohort contains 579 eligible labeled records across 124 sessions. A record is not necessarily a distinct trained checkpoint. A conservative card/code classification finds 516 **candidate** weight-changing outputs and 63 decoding, evaluation or checkpoint-selection bundles. The latter are not counted as additional training recipes; some select intermediate weights, so this is not a claim that all 63 are byte-identical duplicates.

The selection retains **all 313 eligible from-base records**: 190 GSM8K and 123 AIME. This preserves the entire eligible population from the manual-comparables analysis instead of selecting only the apparent successes or treating the old eight-field signature as truth. All 19 grade-A query/match relationships have both members in the proposed cohort; their grades remain reader judgments, not verified weight or recipe equivalence.

The remaining 87 are 50 GSM8K and 37 AIME continuations/merges. Selection prioritizes sessions not represented by the base cohort, rare training methods, continuation-session coverage, recoverable code, and data/method diversity. It does not use accuracy, outcome gaps, or a winner/loser label. Metadata categories are sampling aids, not a definition of similarity.

The selected mix is 313 base-training outputs, 65 continued-training outputs and 22 parameter merges, spanning all 124 sessions. This retains about 76% of the GSM8K and 80% of the AIME candidate-weight pools. The checkpoints and selection reasons are in [checkpoint_inventory.json](../../experiments/eval_matrix_1k/checkpoint_inventory.json); the exact 400 IDs are in [selected_ids_400.json](../../experiments/eval_matrix_1k/selected_ids_400.json).

**Important readiness limit:** these are 400 candidate weight sets, not 400 hash-verified unique weights. The historical script audit says 371 reconstructed, 20 unavailable and nine blocked; the newer broad-prefix extraction flags three selected records with missing declared code and 177 needing content review. These audits have different scopes. Even a reconstructed plan-time script is not automatically the script actually launched. Resolve or replace unusable entries before spending on them; preserve benchmark/session/method coverage and refreeze the manifest if replacements are necessary.

## Why these generation policies?

The choice is based on the 579 eligible records, not arbitrary temperature increments. Two sampling/output-cap patterns cover 302/326 GSM8K records (92.6%). Three cover 189/253 AIME records (74.7%). AIME therefore needs an output-length contrast as well as a sampling-policy contrast.

| Config ID | Benchmark | Temperature | top_k | top_p | Repetition penalty | Output-token cap | Historical sampler/cap count |
|---|---|---:|---:|---:|---:|---:|---:|
| G01 | GSM8K | 1 | 64 | 0.95 | 1 | 4,000 | 192 |
| G02 | GSM8K | 0, greedy | 0, disabled | 1 | 1 | 4,000 | 110 |
| A01 | AIME | 1 | 0, disabled | 1 | 1 | 16,000 | 72 |
| A02 | AIME | 0.6 | 20 | 0.95 | 1 | 16,000 | 63 |
| A03 | AIME | 1 | 0, disabled | 1 | 1 | 2,048 | 54 |
| A04, extension | AIME | 0.6 | 20 | 0.95 | 1.05 | 16,000 | 22 |
| A05, extension | AIME | 0, greedy | 0, disabled | 1 | 1 | 16,000 | 8 |

`min_p=0` throughout. Counts describe observed sampling/cap patterns under the audited runtime interpretation, **not independently verified full effective-policy equivalence or proven performance effects**. In particular, the table does not preserve every historical EOS variant.

Use one standardized intended stop set within each benchmark: Gemma `{1,106}`; Qwen `{151643,151645}`. These union sets appear in 321/326 and 173/253 historical generation files, respectively. Validate each tokenizer's ID mapping and primary EOS, and pin the benchmark template/tokenization contract. Do not silently swap an incompatible tokenizer. Standardizing stops is a new intervention where the original policy differed.

This design identifies:

- GSM8K: common sampling versus greedy on the same weights.
- AIME: a common sampling-policy bundle contrast (A01 versus A02); an isolated output-cap contrast (A01 versus A03); and, on a smaller set, repetition penalty (A02 versus A04) and greedy decoding (A01 versus A05).
- Across recipes: prediction under the same intended serving policy, removing that serving-policy difference as an explanation for recipe-level gaps.

A01 versus A02 changes temperature and filtering jointly; it does not isolate each parameter's causal effect. Comparing independently trained checkpoints also does not isolate the causal effect of dataset versus training recipe. No claim is made about arbitrary unseen decoding policies, all EOS variants, or which configuration maximizes accuracy.

The seven explicit [generation policies](../../experiments/eval_matrix_1k/generation_policies.json) point to `configs/<config_id>/generation_config.json` and corresponding request templates. They are standardized study policies, not copies of the representative historical JSON files. Policy/source details are in [config_audit.md](../../experiments/eval_matrix_1k/config_audit.md).

For example, the disputed near-twins generate four concrete jobs:

| exp_id | Source checkpoint | Config | Benchmark passes |
|---|---|---|---:|
| `poc1k-r0-29-exp-02-G01` | `r0-29-exp-02` | Common sampling | 10 |
| `poc1k-r0-29-exp-02-G02` | `r0-29-exp-02` | Greedy | 10 |
| `poc1k-gsm2-r0-26-exp-03-G01` | `gsm2-r0-26-exp-03` | Common sampling | 10 |
| `poc1k-gsm2-r0-26-exp-03-G02` | `gsm2-r0-26-exp-03` | Greedy | 10 |

This measures whether their gap shrinks under a common policy. It does not assume the training recipes or weights are identical.

For the user's original **"best possible comparable"** question, predeclare a separate pair diagnostic: recheck the existing manual matches against the cleaned launch-only bundles, freeze the chosen partners and grades before collecting new outcomes, and compare their accuracy gaps under identical serving policies. Report every eligible pair, coverage, and gap distributions by grade; do not retain only pairs whose new results agree. The current selection preserves all 313 eligible queries and all 19 grade-A query/partner relationships. This diagnostic can show whether close recipes become substantially more consistent after serving is controlled. Remaining disagreement is not automatically irreducible noise—it may still reveal missing training/data information. Keep this diagnostic separate from the predictor's locked-session score.

## Stage the spend

First do the no-new-evaluation work: finish training-launch-only input bundles, bind scripts/configs/data specifications to the checkpoint, audit effective serving interpretation, and compare training-only, serving-only and combined predictors on the existing labels using session-grouped validation. Those historical results are development evidence. A weak result warrants diagnosis; it is not proof that controlled new measurements cannot help.

The proposal's execution phases are:

| Phase | Combinations | Purpose |
|---|---:|---|
| Operational pilot | 100 | 20 GSM8K checkpoints × two policies, plus 12 AIME checkpoints × all five policies. Verify every policy and measure real cost. |
| Remaining development core | 597 | Learn and validate cross-session prediction on development sessions. |
| Remaining diagnostic extensions | 16 | Complete the 40 extension cells; 24 were already included in the pilot. |
| Locked session test | 287 | Final measurement after freezing inputs, features, models, prompts and retrieval rules. |
| Total | **1,000** | |

The pilot is an operational/cost gate, not enough by itself to declare the predictor feasible. Pause if weight/input binding, resolved serving settings, stop behavior, labels or cost are not under control. After development, do not spend the locked-test budget merely to rescue a predictor with no credible incremental signal. If the test proceeds, do not repeatedly inspect it while tuning.

The locked split reserves **20 whole sessions per benchmark**: 70 GSM8K checkpoints/140 cells and 49 AIME checkpoints/147 cells. Every policy of a checkpoint stays in the same split. All old native labels and outcome-bearing material from these sessions must also be excluded from training and retrieval. Known learned-parent links are grouped; no cross-session links were declared in this cohort, but weight-hash preflight must still check for undeclared aliases. Public base initialization does not join all sessions into one group.

The four explicitly investigated mechanism-anchor sessions remain development-only. All old sessions have nevertheless been historically inspected. Call the final result a **frozen test on held-out sessions with newly collected and/or repeated serving-policy labels**, not an untouched future-recipe discovery test. A later prospective study on genuinely new training recipes is still required for the stronger deployment claim.

## What would count as a useful PoC result?

Primary input: the user's requested training-launch script prefix and referenced training/data configurations, plus the specified effective serving policy. Do not provide target accuracy, target evaluation outputs, final-card conclusions, or measured same-session history as primary prediction features. A score-assisted continuation task can be reported separately.

Primary target:

`mean_pass_rate = (pass_rate_1 + ... + pass_rate_10) / 10`.

With the same question set each time, this equals total correct divided by `10 × number_of_questions`. It is mean pass@1, **not pass@10**. Failed or missing passes are not zeros and are not silently discarded.

Evaluate separately on GSM8K and AIME, and on from-base versus continuation checkpoints. The combined predictor must improve on a strong development-selected simple baseline and on serving-only prediction; otherwise it might only be recognizing that short-cap AIME outputs score poorly. Compare it with training/data-only prediction to measure what serving information adds. Also compare an inference-only LLM with the same permitted inputs and TRAIN-only labeled reference corpus. Keep test labels inaccessible to predictor agents and feature extractors.

Report percentage-point MAE, median/p90 error, session-clustered paired uncertainty, and ranking/selection regret on predefined comparable candidate sets. The primary MAE averages policies within a checkpoint, checkpoints within a session, then sessions equally. Report ordinary cell-weighted MAE and per-policy results as well; an easily predicted short-cap condition must not conceal poor prediction under the long-budget policies. The targeted extensions do not alter the primary population weights.

Before launch, agree on a useful improvement threshold. A proposed starting point is at least 10% relative MAE reduction over the strongest simple baseline, with at least 1 pp absolute reduction on GSM8K or 0.5 pp on AIME, supported by paired uncertainty; these are **proposed decision thresholds, not agreed requirements or power guarantees**. Compare the matched LLM too. A wide interval is inconclusive, not evidence of feasibility or impossibility.

The existing ten-pass mean has median nominal SE about 0.24 pp on GSM8K and 1.22 pp on AIME. Ten passes reduce generation noise but do not create new questions or ten independent recipe examples. In particular, this study should not promise reliable ordering of every 1 pp AIME difference. Full statistical details are in [statistical_design.md](../../experiments/eval_matrix_1k/statistical_design.md).

## Release blockers and cost protection

1. **Inputs:** finish the launch-only extraction and content review. Do not pass the broad v6 code prefix directly to the predictor. Reconstruct or replace unresolved records before scheduling them. The current selected missing-declared-code flags are `gsm2-r0-19-exp-01`, `gsm2-r0-19-exp-03`, and `r0-17-exp-04`.
2. **Weights:** verify archive availability, shard hashes, loadability, and alias/lineage binding. Merge and training declarations are evidence, not tensor-hash proof.
3. **Serving:** record and assert actual server-resolved sampling parameters, loaded generation/tokenizer hashes, EOS/stopping and effective output caps. A saved JSON or request body alone is insufficient. The two inspected historical temperature-zero checkpoints had changing completions, while their saved requests omitted sampling parameters; their server-resolved settings were not logged. This does not establish the cause of that variation. See [runtime_preflight.md](../../experiments/eval_matrix_1k/runtime_preflight.md).
4. **No accidental inheritance:** the proposed vLLM path disables imported generation defaults and supplies the complete request policy, including stop IDs. A 16,000-token request alone does not defeat an inherited `max_new_tokens=2048` cap. Pin runtime, hardware/dtype, tokenizer/template, caching and concurrency. Never overwrite archival checkpoint files. See [vLLM's configuration implementation](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/config/model.py#L1214-L1287) and [request cap handling](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/utils.py#L177-L188).
5. **Reuse:** 368 proposed cells match a historical sampling/cap pattern, but that is not verified full-protocol equality. Audit weights, stops, tokenizer/template, scorer and runtime before reusing a label. Separate replays from genuinely new policy interventions in the final results.
6. **Cost:** cell counts are not GPU-hour estimates. In seven inspected examples, long-budget AIME cells produced more output tokens than GSM8K cells despite far fewer questions. Use the pilot to measure prefill/decode time, token lengths, checkpoint loading, cache behavior and retry overhead. Load each checkpoint once and evaluate its requested policies against that server where supported; do not download/reload weights ten times merely because there are ten passes. Details: [cost_preflight.md](../../experiments/eval_matrix_1k/cost_preflight.md).

## Files and scope

- [Complete 1,000-exp_id list](../../experiments/eval_matrix_1k/experiment_matrix.jsonl): checkpoint archive URI, config/request files and hashes, benchmark, ten-pass requirement, phase, split and explicit preflight-required status.
- [Selected checkpoint audit](../../experiments/eval_matrix_1k/selected_checkpoints.jsonl), [protocol](../../experiments/eval_matrix_1k/protocol.json), [splits](../../experiments/eval_matrix_1k/splits.json), and [machine-readable summary](../../experiments/eval_matrix_1k/matrix_summary.json).
- [100-cell pilot list](../../experiments/eval_matrix_1k/phases/1_operational_pilot.jsonl).
- [Manual-match coverage](../../experiments/eval_matrix_1k/manual_match_coverage.json): preserves historical reader grades without outcome gaps; only cross-split available comparables may be used for test predictions.
- [Checkpoint selector](../../tools/outcome_prediction/select_eval_matrix_checkpoints.py), [proposal generator](../../tools/outcome_prediction/propose_eval_matrix.py), and [tests](../../tests/test_propose_eval_matrix.py). The selector reproduces the frozen 400 IDs; tests also verify that changing outcome values does not change selection. The generator writes proposal artifacts only; it has no evaluation launcher.

The selection is pinned to the established PTB trajectory cohort at `cc2ac9d884a7d962a6024ab0d5cd8ed3370070de` and generation metadata at `446127629d7b271d537390e69bfb2d960a3aa515`. A read-only check of newer HF revision `60391697f6e6ce3d7024ae3c7b888190582062de` found additional results and Dojo trajectories. Those Dojo GSM8K checkpoints use Qwen3 rather than the Gemma GSM8K population here; they are not silently mixed into this matrix. Broader metadata-only series likewise need recipe/checkpoint binding before inclusion.

Bottom line: this list is designed to test whether complete recipe/data/serving evidence supports useful cross-session prediction, while preserving a genuine opportunity to stop early. It does not assume that additional evaluations will make the predictor work.
