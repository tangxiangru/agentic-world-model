# r-cf7852ab — gsm8k / google/gemma-3-4b-pt / 10 h / 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 40 | 0.03 | sft | base_model | openai/gsm8k train | 1e-4 / 2 | failed | — | inconclusive | iterate |
| exp-02 | 48 | 0.04 | sft | base_model | openai/gsm8k train (7099) | 1e-4 / 2 | failed | — | inconclusive | iterate |
| exp-03 | 52 | 0.05 | sft | base_model | openai/gsm8k train (7099) | 1e-4 / 2 | completed | — (never scored on its own) | inconclusive | adopt |
| exp-04 | 62 | 0.65 | merge | exp-03 | — | — / — | completed | accuracy 0.54, n=100 (eval_simple.py, gsm8k test, printed at [85]) | inconclusive | adopt |
| exp-05 | 90 | 9.94 | sft | base_model | openai/gsm8k train (7099) | 5e-5 / 3 | killed | — (230/2664 steps, no checkpoint) | inconclusive | abandon_line |

Submitted checkpoint: `final_model` from exp-04. No baseline or comparator was
ever measured (the pre-training eval at [28] crashed), so every verdict is
inconclusive.
