| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 44 | 0.17 | sft | base_model | openai/gsm8k main train (7,473) | 2e-4 / 3 | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-02 | 110 | 0.75 | merge | exp-01 | - | - / - | completed | accuracy 0.360, n=50 [121] | inconclusive | adopt |
| exp-03 | 126 | 0.84 | sft | base_model | openai/gsm8k (7,473) + meta-math/MetaMathQA GSM types (240,000) | 2e-4 / 2 | killed | none | inconclusive | abandon_line |
| exp-04 | 138 | 0.95 | sft | base_model | openai/gsm8k (7,473) + meta-math/MetaMathQA GSM types (50,000) | 2e-4 / 1 | completed | none (adapter not evaluated) | inconclusive | adopt |
| exp-05 | 164 | 2.05 | merge | exp-04 | - | - / - | completed | accuracy 0.320, n=100 [169] | inconclusive | adopt |
| exp-06 | 172 | 2.19 | other (packaging) | exp-05 | - | - / - | completed | none of its own (byte copy of exp-05) | inconclusive | adopt |
