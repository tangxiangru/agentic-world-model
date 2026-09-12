# World-model benchmark design

We are constructing a supervised benchmark for predicting a trained language model's evaluation outcomes from its experiment recipe. The input is `X = (C, Δ, S)`: everything that produced the checkpoint, i.e. the executed training/data-preparation programs with their resources, arguments and execution environment (C), post-processing applied to checkpoints such as weight averaging (Δ), and the effective serving settings plus a fixed benchmark/scoring protocol (S). The pretrained base model and version are declared task context; model weights are not predictor inputs. Each example links this recipe to one historically produced checkpoint evaluated in ten complete runs over the same N questions (1,319 for GSM8K; 30 for AIME), yielding Z: a question/run-indexed record of responses/trajectories, scorer outputs and available generation/error metadata. The initial target Y is the average pass rate across these ten runs. We preserve Z to support other target choices `Y_t = g_t(Z)` later; each task fixes the transformation and the metric used to evaluate predicted targets. Predictors may learn from training examples, but predict held-out targets from X and shared task context without training or evaluating the target model; its weights, trajectories and observed outcomes remain withheld. The ten repeats measure evaluation variability for one checkpoint, not variability from retraining.

**X = (C, Δ, S)**

| Part | Contents |
|---|---|
| **C: checkpoint production** | Exact training and data-preparation code, required helpers/resources, the arguments/configuration used to execute it, and the execution environment (library versions): the environment fixes every unspecified default and the meaning of every specified flag, so it is part of the recipe. Includes any changes to initialization, tokenizer or model configuration. If training starts from a checkpoint other than the declared base model, C also contains the C/Δ that produced that checkpoint, back to the base: X describes the whole production chain, not only its last step. C is the launch-time snapshot; text written after an evaluation (results, notes, later edits) is not part of X. |
| **Δ: patches** | Post-processing applied to produced checkpoints, e.g. weight averaging or checkpoint selection, with its inputs identified (which checkpoints, which weights). |
| **S: serving** | The effective generation settings used by the evaluator: sampler parameters (temperature, top-k/top-p/min-p, repetition penalty), the stop-token set, the output-token cap, and the fixed benchmark execution/scoring specification (question set, prompt template, scorer, evaluator version). S records the settings actually used after resolving checkpoint defaults (if loaded), server configuration and per-request overrides. Retain the resolved settings and the configuration/invocations that determine them. |

An example is identified by (production chain, S). Two X that share weights but differ in S are two examples; a record that re-evaluates an existing checkpoint without changing weights or S is an alias of the existing example, not a new one. Independent re-executions of the same C are separate examples and are the only measure of retraining variability. Standardizing X and Y comes first; train/test splits are defined afterwards, at the scientist-session level, so that a held-out example's production chain, session and outcomes are not seen during training.

**Initial Y: average pass rate across ten complete runs of one trained checkpoint.** Preserve the outcome record Z to support additional targets `Y_t = g_t(Z)` without collecting the evidence again.

**Z contains:** original result files and full response trajectories; a question-by-run index with question/run IDs, question/target references, responses, extracted answers and scores; available request/seed settings, finish reasons, token counts, timings and error/retry records. Keep immutable source references/hashes and checkpoint, serving, scorer and execution bindings. Preserve raw artifacts alongside normalized records; explicitly mark unavailable fields. Z is label-side data, never part of X.

For N fixed questions (1,319 GSM8K; 30 AIME), let `c[q,r]` be scored correctness, `a[r] = mean_q c[q,r]` the accuracy of run r, and `s[q] = sum_r c[q,r]`. The accuracy targets below use ten complete, matched runs.

| Possible Y | Definition |
|---|---|
| **Mean pass rate (initial Y)** | `Y = μ = mean(a[1], …, a[10])`. |
| **Raw run pass rates** | `[a[1], …, a[10]]`, retaining run IDs/seeds; also supports an empirical distribution over run accuracy. |
| **Observed range** | Interval `[min(a), max(a)]`; range width `max(a) − min(a)` is a separate scalar target. |
| **Run variability** | Population SD `sqrt(mean((a[r] − μ)^2))`; median/quantiles with a specified quantile convention. |
| **Question-level outcomes** | Pass fractions `[s[q]/10]` and the full `N × 10` correctness matrix `c[q,r]`. |
| **Any-correct pass@k** | For `1 ≤ k ≤ 10`: `mean_q(1 − choose(10 − s[q], k) / choose(10, k))`, with `choose(n,k)=0` for `n<k`; success over uniformly selected k-subsets of the recorded responses. |
| **Response/trajectory outcomes** | Raw response lists/trajectories, output-length distributions, truncation rates, extracted-answer agreement and majority-vote accuracy. Timing/error targets are possible where the required metadata exists. |

[Worked example: one outcome record, multiple Y targets](wm_benchmark_example.md).
