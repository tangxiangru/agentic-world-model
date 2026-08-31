# Reconstructed experiment cards

Base model: google/gemma-3-4b-pt. Benchmark: gsm8k. Budget: 10 h, one H100.
13 cards, one per launch that can be cited in the event stream.
Accuracies are the agent's own `evaluate.py` runs; `@150` is `--limit 150`,
`@1319` is `--limit -1` (the full benchmark).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 93 | 0.30 | sft | base_model | MetaMathQA GSM-derived, 100,000 (stratified 35/35/15/15) | 1e-5 / 1 | completed | 0.0400 @150 (half-epoch checkpoint; the endpoint was never scored) | inconclusive | adopt |
| exp-02 | 231 | 2.09 | sft | exp-01 | openai/gsm8k train, 7,473 in the evaluator's ten-shot format | 3e-6 / 1 | completed | 0.6262 @1319 (0.5933 @150) | supported | adopt |
| exp-03 | 290 | 2.69 | sft | exp-02 | same 7,473, seed 43 | 1.5e-6 / 1 | completed | 0.6232 @1319 (0.6200 @150) | inconclusive | adopt |
| exp-04 | 348 | 3.27 | sft | exp-03 | same 7,473, seed 44 | 7.5e-7 / 1 | completed | 0.6088 @1319 (0.6067 @150) | contradicted | reject |
| exp-05 | 385 | 3.86 | sft | exp-03 | MetaMathQA GSM-derived, the disjoint remaining 140,000 (offset 100k) | 5e-6 / 1 | completed | none (never evaluated without a recovery epoch) | inconclusive | adopt |
| exp-06 | 560 | 6.07 | sft | exp-05 | same 7,473, seed 45 | 2e-6 / 1 | completed | 0.6067 @150 | contradicted | reject |
| exp-07 | 612 | 6.65 | sft | exp-05 (checkpoint-4375) | same 7,473, seed 45 | 2e-6 / 1 | completed | 0.6073 @1319 (0.6267 @150) | contradicted | adopt |
| exp-08 | 667 | 7.23 | sft | exp-07 | same 7,473, seed 46 | 1e-6 / 1 | completed | 0.6027 @1319 (0.6333 @150) | inconclusive | reject |
| exp-09 | 774 | 8.05 | merge | exp-03 (with exp-04 at alpha 0.25) | none | n/a | killed | none (never evaluated) | inconclusive | abandon_line |
| exp-10 | 835 | 8.37 | rft | exp-02 | selftrain.jsonl, 6,248 self-generated verified-correct solutions | 1e-6 / 1 | completed | 0.6171 @1319 (checkpoint-98) | contradicted | reject |
| exp-11 | 868 | 8.89 | merge | exp-02 (with exp-10 checkpoint-98 at alpha 0.25) | none | n/a | completed | 0.6179 @1319 | contradicted | reject |
| exp-12 | 930 | 9.30 | merge | exp-02 (with exp-01 at alpha 0.1125) | none | n/a | completed | 0.6240 @1319 | inconclusive | reject |
| exp-13 | 951 | 9.43 | other (export to final_model) | none - the 12.5% aligned/stage-one soup, whose merge command is not in the stream | none | n/a | completed | 0.6338 @1319 under evaluate.py defaults (0.6293 @1319 at 16 connections) | inconclusive | adopt |

Submitted checkpoint: **exp-13** - `final_model`, a byte-identical copy of
`aligned_stage1_a0125_model` (0.875 x exp-02 + 0.125 x exp-01).
