# Example: one recipe, several prediction targets

This is an invented three-question example to illustrate the [benchmark design](wm_benchmark_spec.md), not an HF result. All ten evaluation runs use the same trained checkpoint.

The X summary is: C runs data preparation then training with learning rate `1e-5` and two epochs, under pinned library versions, starting from the base model (so the production chain has one step); Δ applies no post-processing; S is temperature `0.6`, no top-k/top-p filtering, a 128-token output cap, the stop-token set, and the fixed questions/scorer below, as recorded from the evaluator's resolved settings rather than read off the training script. The base model is fixed task context. A real X would supply the actual code/resources/configuration, not just this summary. Had the checkpoint been trained from an earlier checkpoint, C would also contain that checkpoint's production scripts. A saved generation config contributes to S only if the serving runtime loads it, subject to server settings and request overrides.

The questions are Q1: `2 + 2` (answer 4), Q2: `3 × 3` (answer 9), and Q3: `10 − 4` (answer 6). Each response is `Answer: n`; the scorer extracts n and checks equality with the gold answer. These are the recorded response values in Z:

| Question | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Q2 | 9 | 9 | 9 | 8 | 9 | 8 | 9 | 8 | 9 | 8 |
| Q3 | 5 | 5 | 5 | 6 | 5 | 5 | 5 | 6 | 5 | 5 |

Scoring produces the following correctness matrix (1 = correct, 0 = incorrect):

| Question | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Pass fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1.0 |
| Q2 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0.6 |
| Q3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0.2 |
| Run accuracy | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 | 2/3 | 2/3 | 2/3 | 1/3 | |

The initial target is mean pass rate: `Y = 0.60`. The same X and preserved Z also support the other targets below:

| Target choice | Y for this example |
|---|---|
| Mean pass rate (initial Y) | `18/30 = 0.60` |
| Raw run pass rates | `[2/3, 2/3, 2/3, 2/3, 2/3, 1/3, 2/3, 2/3, 2/3, 1/3]` |
| Observed range | `[1/3, 2/3]`; width `1/3` (33.33 percentage points) |
| Population run SD | `2/15 ≈ 0.1333` (13.33 percentage points) |
| Question pass fractions | `[1.0, 0.6, 0.2]` |
| Full correctness target | The three-by-ten binary matrix above |
| Any-correct pass@2 | `(1 + 39/45 + 17/45) / 3 = 101/135 ≈ 0.7481`, averaging uniformly selected two-response subsets per question |
| Any-correct pass@10 | `1.0`: every question has at least one correct response |
| Majority-vote accuracy | `2/3`: modal answers are 4, 9 and 5; Q3's modal answer is incorrect. No ties occur here. |

For the mean target, a predictor returns one number; for the raw-run target, ten numbers aligned to run IDs; for the question target, three numbers aligned to question IDs. A benchmark task chooses its target and prediction metric before comparing methods. Z and the derived targets stay hidden for the examples being predicted.

There is one recipe/serving example here, containing 30 scored responses. The observed range is not a confidence interval, and the ten evaluations provide no measurement of retraining variability. The same checkpoint served at another temperature would be a second example with its own Z. No token/timing statistics are invented for this example; real records retain them where available.
