# Index of reconstructed cards

| side | run_ref | card | base model | launch_i | family | parent | data sources | exec | best own eval | verdict | decision | hyp stated | official |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | r-016546b4 | exp-01 | Qwen/Qwen3-1.7B-Base | 23 | other | base_model |  | completed | 0.1 | inconclusive | reject | False |  |
| train | r-016546b4 | exp-02 | Qwen/Qwen3-1.7B-Base | 52 | sft | base_model | HF meta-math/MetaMathQA | failed |  | inconclusive | iterate | False |  |
| train | r-016546b4 | exp-03 | Qwen/Qwen3-1.7B-Base | 62 | sft | base_model | HF meta-math/MetaMathQA | completed |  | inconclusive | adopt | False |  |
| train | r-016546b4 | exp-04 | Qwen/Qwen3-1.7B-Base | 68 | other | exp-03 |  | completed | 0.14 | inconclusive | reject | False |  |
| train | r-016546b4 | exp-05 | Qwen/Qwen3-1.7B-Base | 84 | sft | base_model | HF meta-math/MetaMathQA | killed |  | inconclusive | adopt | True |  |
| train | r-016546b4 | exp-06 | Qwen/Qwen3-1.7B-Base | 119 | other | exp-05 |  | completed | 0.3 | supported | adopt | True | 0.40106141015921154 |
| train | r-016546b4 | exp-07 | Qwen/Qwen3-1.7B-Base | 201 | sft | base_model | HF meta-math/MetaMathQA,HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-01ed5927 | exp-01 | Qwen/Qwen3-4B-Base | 102 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.4266666666666667 | contradicted | adopt | False |  |
| train | r-01ed5927 | exp-02 | Qwen/Qwen3-4B-Base | 312 | decode-config | exp-01 |  | completed | 0.8133333333333334 | supported | adopt | True |  |
| train | r-01ed5927 | exp-03 | Qwen/Qwen3-4B-Base | 336 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-01ed5927 | exp-04 | Qwen/Qwen3-4B-Base | 401 | rft | base_model | derived:exp-02 + openai/gsm8k,synthetic:self | completed | 0.82 | inconclusive | reject | True |  |
| train | r-01ed5927 | exp-05 | Qwen/Qwen3-4B-Base | 529 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | failed |  | inconclusive | iterate | True |  |
| train | r-01ed5927 | exp-06 | Qwen/Qwen3-4B-Base | 577 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | completed | 0.8385140257771039 | supported | adopt | True |  |
| train | r-01ed5927 | exp-07 | Qwen/Qwen3-4B-Base | 663 | other | exp-06 |  | completed | 0.84 | inconclusive | adopt | False | 0.8339651250947687 |
| train | r-01ed5927 | exp-08 | Qwen/Qwen3-4B-Base | 685 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | completed | 0.8332069749810462 | contradicted | reject | True |  |
| train | r-0295e24f | exp-01 | Qwen/Qwen3-4B-Base | 129 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.04666666666666667 | inconclusive | adopt | True |  |
| train | r-0295e24f | exp-02 | Qwen/Qwen3-4B-Base | 187 | other | exp-01 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-0295e24f | exp-03 | Qwen/Qwen3-4B-Base | 206 | decode-config | exp-01 |  | completed | 0.14 | contradicted | reject | True |  |
| train | r-0295e24f | exp-04 | Qwen/Qwen3-4B-Base | 234 | sft | base_model | openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-0295e24f | exp-05 | Qwen/Qwen3-4B-Base | 249 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.04 | contradicted | reject | True |  |
| train | r-0295e24f | exp-06 | Qwen/Qwen3-4B-Base | 312 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.5 | supported | adopt | True |  |
| train | r-0295e24f | exp-07 | Qwen/Qwen3-4B-Base | 411 | rft | base_model | synthetic:self | completed | 0.42 | contradicted | reject | True |  |
| train | r-0295e24f | exp-08 | Qwen/Qwen3-4B-Base | 443 | rft | base_model | synthetic:self | completed | 0.36 | contradicted | reject | True |  |
| train | r-0295e24f | exp-09 | Qwen/Qwen3-4B-Base | 459 | other | exp-06 |  | completed | 0.5 | supported | adopt | False | 0.42987111448066717 |
| train | r-049e5f94 | exp-01 | Qwen/Qwen3-4B-Base | 252 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA | killed | 0.375 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-02 | Qwen/Qwen3-4B-Base | 414 | sft | exp-01 | derived: openai/gsm8k train + meta-math/MetaMathQA | killed | 0.54 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-03 | Qwen/Qwen3-4B-Base | 516 | sft | exp-02 | derived: openai/gsm8k train + meta-math/MetaMathQA | killed |  | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-04 | Qwen/Qwen3-4B-Base | 645 | decode-config | exp-03 |  | completed | 0.83 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-05 | Qwen/Qwen3-4B-Base | 711 | rft | base_model | derived:exp-04 (synthetic:self) + openai/gsm8k train + meta-math/MetaMathQA,synthetic:self (exp-04 checkpoint) | killed | 0.82 | inconclusive | reject | True |  |
| train | r-049e5f94 | exp-06 | Qwen/Qwen3-4B-Base | 892 | rft | exp-05 | derived:exp-04 (synthetic:self) + openai/gsm8k train + meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | True |  |
| train | r-049e5f94 | exp-07 | Qwen/Qwen3-4B-Base | 911 | sft | exp-03 | derived: openai/gsm8k train + meta-math/MetaMathQA | killed | 0.84 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-08 | Qwen/Qwen3-4B-Base | 1042 | grpo | exp-07 | HF: openai/gsm8k | killed | 0.84 | contradicted | reject | True |  |
| train | r-049e5f94 | exp-09 | Qwen/Qwen3-4B-Base | 1149 | grpo | exp-07 | HF: openai/gsm8k | killed | 0.847 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-10 | Qwen/Qwen3-4B-Base | 1219 | grpo | exp-09 | HF: openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| train | r-049e5f94 | exp-11 | Qwen/Qwen3-4B-Base | 1247 | sft | exp-07 | derived: openai/gsm8k train + meta-math/MetaMathQA | killed | 0.85 | inconclusive | adopt | True |  |
| train | r-049e5f94 | exp-12 | Qwen/Qwen3-4B-Base | 1351 | sft | exp-11 | derived: openai/gsm8k train + meta-math/MetaMathQA | killed | 0.87 | inconclusive | adopt | True | 0.8468536770280516 |
| train | r-049e5f94 | exp-13 | Qwen/Qwen3-4B-Base | 1464 | sft | exp-12 | derived: openai/gsm8k train + meta-math/MetaMathQA | completed | 0.85 | contradicted | reject | True |  |
| train | r-0632a2e4 | exp-01 | Qwen/Qwen3-1.7B-Base | 96 | sft | base_model | local | completed |  | inconclusive | adopt | True |  |
| train | r-0632a2e4 | exp-02 | Qwen/Qwen3-1.7B-Base | 178 | decode-config | exp-01 |  | completed | 0.6 | inconclusive | reject | True |  |
| train | r-0632a2e4 | exp-03 | Qwen/Qwen3-1.7B-Base | 277 | sft | base_model | local | completed | 0.673 | supported | reject | True |  |
| train | r-0632a2e4 | exp-04 | Qwen/Qwen3-1.7B-Base | 371 | sft | base_model | local | completed | 0.693 | inconclusive | adopt | False |  |
| train | r-0632a2e4 | exp-05 | Qwen/Qwen3-1.7B-Base | 481 | sft | base_model | local,synthetic:self | completed | 0.675 | contradicted | reject | True |  |
| train | r-0632a2e4 | exp-06 | Qwen/Qwen3-1.7B-Base | 548 | other | exp-04 |  | completed | 0.665 | inconclusive | adopt | False | 0.66565579984837 |
| train | r-0632a2e4 | exp-07 | Qwen/Qwen3-1.7B-Base | 583 | sft | base_model | local | failed |  | inconclusive | abandon_line | False |  |
| train | r-0632a2e4 | exp-08 | Qwen/Qwen3-1.7B-Base | 628 | sft | exp-04 | local | completed | 0.602 | inconclusive | reject | False |  |
| train | r-06a66e16 | exp-01 | Qwen/Qwen3-1.7B-Base | 238 | sft | base_model | mixture of HF sets: nvidia/OpenMathInstruct-2, microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA, openai/gsm8k | failed |  | inconclusive | abandon_line | False |  |
| train | r-06a66e16 | exp-02 | Qwen/Qwen3-1.7B-Base | 278 | sft | base_model | mixture of HF sets: nvidia/OpenMathInstruct-2, microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA, openai/gsm8k | killed |  | inconclusive | abandon_line | False |  |
| train | r-06a66e16 | exp-03 | Qwen/Qwen3-1.7B-Base | 337 | sft | base_model | mixture of HF sets: nvidia/OpenMathInstruct-2 (gsm8k- and math-sourced), microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA, openai/gsm8k train | completed | 0.8066666666666666 | inconclusive | adopt | True |  |
| train | r-06a66e16 | exp-04 | Qwen/Qwen3-1.7B-Base | 441 | grpo | exp-03 | HF openai/gsm8k (main), train split, saved to disk at [64] | completed | 0.88 | supported | adopt | False |  |
| train | r-06a66e16 | exp-05 | Qwen/Qwen3-1.7B-Base | 576 | grpo | exp-04 | HF openai/gsm8k (main), train split, saved to disk at [64] | completed | 0.878 | supported | adopt | False |  |
| train | r-06a66e16 | exp-06 | Qwen/Qwen3-1.7B-Base | 677 | decode-config | exp-05 |  | completed | 0.8733333333333333 | inconclusive | adopt | False |  |
| train | r-06a66e16 | exp-07 | Qwen/Qwen3-1.7B-Base | 699 | grpo | exp-05 | derived:exp-05 (GRPO-2 policy sampled over the openai/gsm8k train prompts) | completed | 0.884 | inconclusive | reject | True |  |
| train | r-06a66e16 | exp-08 | Qwen/Qwen3-1.7B-Base | 800 | decode-config | exp-07 |  | completed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-06a66e16 | exp-09 | Qwen/Qwen3-1.7B-Base | 810 | decode-config | exp-05 |  | completed | 0.8733333333333333 | supported | adopt | True | 0.8635329795299469 |
| train | r-0788f765 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 84 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 96 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | True |  |
| train | r-0788f765 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 104 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 112 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 122 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | completed | 0.1 | inconclusive | adopt | False |  |
| train | r-0788f765 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 148 | decode-config | exp-05 |  | completed | 0.5 | supported | adopt | True | 0.6504927975739196 |
| train | r-08a3dad0 | exp-01 | Qwen/Qwen3-1.7B-Base | 247 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.12666666666666668 | contradicted | adopt | True |  |
| train | r-08a3dad0 | exp-02 | Qwen/Qwen3-1.7B-Base | 576 | decode-config | exp-01 |  | completed | 0.6133333333333333 | supported | adopt | True |  |
| train | r-08a3dad0 | exp-03 | Qwen/Qwen3-1.7B-Base | 637 | other | exp-02 |  | completed | 0.6133333333333333 | supported | adopt | False | 0.5936315390447309 |
| train | r-08a3dad0 | exp-04 | Qwen/Qwen3-1.7B-Base | 712 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.47333333333333333 | contradicted | reject | True |  |
| train | r-08a3dad0 | exp-05 | Qwen/Qwen3-1.7B-Base | 806 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | failed |  | inconclusive | abandon_line | True |  |
| train | r-08a3dad0 | exp-06 | Qwen/Qwen3-1.7B-Base | 841 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.04 | contradicted | reject | True |  |
| train | r-09dc8376 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 33772 | sft | base_model | derived:local,openai/gsm8k train + meta-math/MetaMathQA + nvidia/OpenMathInstruct-2 | completed | 0.7717968157695224 | inconclusive | adopt | True |  |
| train | r-09dc8376 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 40548 | rft | exp-01 | derived:exp-01,synthetic:self,synthetic:self | failed |  | inconclusive | iterate | True |  |
| train | r-09dc8376 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 42608 | rft | exp-01 | derived:exp-01,synthetic:self,synthetic:self | completed | 0.8266666666666667 | supported | adopt | True |  |
| train | r-09dc8376 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 44396 | rft | exp-03 | derived:exp-03,synthetic:self,synthetic:self | completed | 0.82 | inconclusive | adopt | False |  |
| train | r-09dc8376 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 61939 | other | exp-04 |  | completed | 0.82 | inconclusive | adopt | True | 0.7945413191811979 |
| train | r-09dc8376 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 65711 | sft | exp-04 | derived:exp-04,nvidia/OpenMathInstruct-2 + synthetic:self | completed | 0.8036391205458681 | inconclusive | reject | False |  |
| train | r-0af4240f | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 76 | sft | base_model | HF:meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-0af4240f | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 166 | sft | base_model | HF:meta-math/MetaMathQA | completed | 0.4 | inconclusive | adopt | True |  |
| train | r-0af4240f | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 510 | decode-config | exp-02 |  | completed | 0.36 | inconclusive | adopt | True |  |
| train | r-0af4240f | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 531 | sft | exp-03 | HF:meta-math/MetaMathQA | completed | 0.48 | supported | adopt | True |  |
| train | r-0af4240f | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 689 | other | exp-04 |  | completed | 0.4 | inconclusive | adopt | False | 0.42153146322971946 |
| train | r-0d7c7a69 | exp-01 | Qwen/Qwen3-1.7B-Base | 368 | sft | base_model | HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k sources) + HF openai/gsm8k train | failed |  | inconclusive | abandon_line | False |  |
| train | r-0d7c7a69 | exp-02 | Qwen/Qwen3-1.7B-Base | 392 | sft | base_model | HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k sources) + HF openai/gsm8k train | completed | 0.7991 | inconclusive | adopt | False |  |
| train | r-0d7c7a69 | exp-03 | Qwen/Qwen3-1.7B-Base | 533 | decode-config | exp-02 |  | completed |  | inconclusive | adopt | True |  |
| train | r-0d7c7a69 | exp-04 | Qwen/Qwen3-1.7B-Base | 582 | grpo | exp-02 | HF openai/gsm8k, train split (prompts only; built in-process by train_grpo.py) | completed | 0.8196 | supported | adopt | False |  |
| train | r-0d7c7a69 | exp-05 | Qwen/Qwen3-1.7B-Base | 636 | grpo | exp-04 | HF openai/gsm8k, train split (prompts only; built in-process by train_grpo.py) | completed | 0.8347 | supported | adopt | False |  |
| train | r-0d7c7a69 | exp-06 | Qwen/Qwen3-1.7B-Base | 700 | grpo | exp-05 | HF openai/gsm8k, train split (prompts only; built in-process by train_grpo.py) | completed | 0.8332 | contradicted | reject | False |  |
| train | r-0d7c7a69 | exp-07 | Qwen/Qwen3-1.7B-Base | 753 | merge | exp-05 |  | completed | 0.8264 | contradicted | reject | False |  |
| train | r-0d7c7a69 | exp-08 | Qwen/Qwen3-1.7B-Base | 762 | decode-config | exp-05 |  | completed | 0.8533 | inconclusive | adopt | True |  |
| train | r-0d7c7a69 | exp-09 | Qwen/Qwen3-1.7B-Base | 772 | grpo | exp-05 | HF openai/gsm8k, train split (prompts only; built in-process by train_grpo.py) | completed | 0.8355 | inconclusive | adopt | False |  |
| train | r-0d7c7a69 | exp-10 | Qwen/Qwen3-1.7B-Base | 814 | decode-config | exp-09 |  | completed | 0.8533 | supported | adopt | True |  |
| train | r-0d7c7a69 | exp-11 | Qwen/Qwen3-1.7B-Base | 826 | decode-config | exp-09 |  | completed | 0.8533 | inconclusive | adopt | True | 0.8324488248673237 |
| train | r-114ff7d5 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 89 | sft | base_model | meta-math/MetaMathQA | failed |  | inconclusive | iterate | True |  |
| train | r-114ff7d5 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 117 | sft | base_model | meta-math/MetaMathQA | completed | 0.7066666666666667 | inconclusive | adopt | True |  |
| train | r-114ff7d5 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 187 | sft | exp-02 | openai/gsm8k (config main, split train) | completed | 0.49333333333333335 | contradicted | reject | True |  |
| train | r-114ff7d5 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 242 | sft | exp-02 | meta-math/MetaMathQA | completed | 0.72 | inconclusive | adopt | False |  |
| train | r-114ff7d5 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 289 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.7066666666666667 | inconclusive | reject | False |  |
| train | r-114ff7d5 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 486 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.6666666666666666 | contradicted | reject | True |  |
| train | r-114ff7d5 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 510 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.82 | supported | adopt | True |  |
| train | r-114ff7d5 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 594 | other | exp-07 |  | completed | 0.82 | supported | adopt | True | 0.7581501137225171 |
| train | r-11be89c8 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 106 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | killed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 192 | merge | exp-01 |  | completed | 0.041666666666666664 | inconclusive | reject | False |  |
| train | r-11be89c8 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 221 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | killed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 258 | merge | exp-03 |  | completed | 0.08333333333333333 | inconclusive | reject | True |  |
| train | r-11be89c8 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 276 | decode-config | exp-04 |  | completed | 0.075 | inconclusive | reject | True |  |
| train | r-11be89c8 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 297 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 382 | merge | exp-06 |  | completed | 0.05 | inconclusive | reject | False |  |
| train | r-11be89c8 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 387 | merge | exp-06 |  | completed | 0.025 | inconclusive | reject | False |  |
| train | r-11be89c8 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 427 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 484 | merge | exp-09 |  | completed | 0.1 | inconclusive | reject | False |  |
| train | r-11be89c8 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 485 | merge | exp-09 |  | completed | 0.075 | inconclusive | reject | False |  |
| train | r-11be89c8 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 507 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 512 | merge | exp-12 |  | completed | 0.125 | contradicted | reject | True |  |
| train | r-11be89c8 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 525 | sft | base_model | HF openai/gsm8k (config main, train split), loaded in-process by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-11be89c8 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 532 | merge | exp-14 |  | completed | 0.2375 | supported | adopt | True | 0.2092494313874147 |
| train | r-130da32f | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 52 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.225 | supported | adopt | True |  |
| train | r-130da32f | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 257 | merge | exp-01 |  | completed | 0.1 | contradicted | reject | True |  |
| train | r-130da32f | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 270 | other | exp-01 |  | completed | 0.1 | inconclusive | adopt | True | 0.10386656557998483 |
| train | r-139b3113 | exp-01 | Qwen/Qwen3-1.7B-Base | 88 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-139b3113 | exp-02 | Qwen/Qwen3-1.7B-Base | 119 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | False |  |
| train | r-139b3113 | exp-03 | Qwen/Qwen3-1.7B-Base | 138 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-139b3113 | exp-04 | Qwen/Qwen3-1.7B-Base | 293 | other | exp-03 |  | killed |  | inconclusive | abandon_line | False |  |
| train | r-139b3113 | exp-05 | Qwen/Qwen3-1.7B-Base | 311 | other | exp-03 |  | completed | 0.06 | inconclusive | adopt | False |  |
| train | r-139b3113 | exp-06 | Qwen/Qwen3-1.7B-Base | 334 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | completed |  | inconclusive | reject | True |  |
| train | r-139b3113 | exp-07 | Qwen/Qwen3-1.7B-Base | 364 | other | exp-06 |  | completed | 0.033 | inconclusive | reject | False |  |
| train | r-139b3113 | exp-08 | Qwen/Qwen3-1.7B-Base | 382 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | completed |  | inconclusive | reject | False |  |
| train | r-139b3113 | exp-09 | Qwen/Qwen3-1.7B-Base | 386 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-139b3113 | exp-10 | Qwen/Qwen3-1.7B-Base | 456 | other | exp-08 |  | completed | 0.02 | inconclusive | reject | False |  |
| train | r-139b3113 | exp-11 | Qwen/Qwen3-1.7B-Base | 468 | other | exp-03 |  | completed |  | inconclusive | adopt | False | 0.04245640636846096 |
| train | r-1441f3c6 | exp-01 | Qwen/Qwen3-1.7B-Base | 67 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.02 | inconclusive | reject | True |  |
| train | r-1441f3c6 | exp-02 | Qwen/Qwen3-1.7B-Base | 156 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | failed |  | inconclusive | iterate | True |  |
| train | r-1441f3c6 | exp-03 | Qwen/Qwen3-1.7B-Base | 167 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.05 | inconclusive | reject | True |  |
| train | r-1441f3c6 | exp-04 | Qwen/Qwen3-1.7B-Base | 222 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | inconclusive | abandon_line | True |  |
| train | r-1441f3c6 | exp-05 | Qwen/Qwen3-1.7B-Base | 234 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | inconclusive | abandon_line | True |  |
| train | r-1441f3c6 | exp-06 | Qwen/Qwen3-1.7B-Base | 249 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | killed |  | inconclusive | iterate | True |  |
| train | r-1441f3c6 | exp-07 | Qwen/Qwen3-1.7B-Base | 255 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.1417 | supported | adopt | True |  |
| train | r-1441f3c6 | exp-08 | Qwen/Qwen3-1.7B-Base | 267 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | contradicted | reject | True |  |
| train | r-1441f3c6 | exp-09 | Qwen/Qwen3-1.7B-Base | 283 | other | exp-07 |  | completed | 0.15 | inconclusive | adopt | False | 0.12357846853677028 |
| train | r-149cfe8a | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 208 | sft | base_model | openai/gsm8k[train] + meta-math/MetaMathQA[GSM_AnsAug,GSM_Rephrased],openai/gsm8k[train] | completed | 0.6667 | inconclusive | adopt | True |  |
| train | r-149cfe8a | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 374 | decode-config | exp-01 |  | completed | 0.8 | supported | adopt | True |  |
| train | r-149cfe8a | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 416 | other | exp-01 |  | completed | 0.8 | inconclusive | adopt | True |  |
| train | r-149cfe8a | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 443 | rft | base_model | derived:exp-01 (data/train.jsonl) + synthetic:self (data/rft.jsonl),synthetic:self (samples from runs/sft1/checkpoint-2344 on openai/gsm8k[train]) | completed | 0.827 | supported | adopt | True |  |
| train | r-149cfe8a | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 511 | other | exp-04 |  | completed | 0.801 | inconclusive | adopt | True |  |
| train | r-149cfe8a | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 568 | grpo | exp-04 | openai/gsm8k[train] | failed |  | inconclusive | abandon_line | True |  |
| train | r-149cfe8a | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 627 | grpo | exp-04 | openai/gsm8k[train] | completed | 0.818 | supported | adopt | True |  |
| train | r-149cfe8a | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 682 | other | exp-07 |  | completed | 0.818 | inconclusive | adopt | True |  |
| train | r-149cfe8a | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 682 | grpo | exp-07 | openai/gsm8k[train] | completed | 0.8203 | inconclusive | adopt | True |  |
| train | r-149cfe8a | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 740 | other | exp-09 |  | completed | 0.8733 | inconclusive | adopt | True | 0.819560272934041 |
| train | r-149cfe8a | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 830 | grpo | exp-04 | openai/gsm8k[train] | completed | 0.8533 | contradicted | reject | True |  |
| train | r-14faeaaa | exp-01 | Qwen/Qwen3-4B-Base | 419 | sft | base_model | synthetic:self | killed |  | inconclusive | abandon_line | False |  |
| train | r-14faeaaa | exp-02 | Qwen/Qwen3-4B-Base | 441 | sft | base_model | synthetic:self | killed |  | inconclusive | abandon_line | False |  |
| train | r-14faeaaa | exp-03 | Qwen/Qwen3-4B-Base | 454 | rft | base_model | synthetic:self | completed | 0.853 | supported | adopt | False |  |
| train | r-14faeaaa | exp-04 | Qwen/Qwen3-4B-Base | 559 | rft | base_model | derived:exp-03 | killed |  | inconclusive | abandon_line | False |  |
| train | r-14faeaaa | exp-05 | Qwen/Qwen3-4B-Base | 637 | rft | base_model | derived:exp-03 | killed |  | inconclusive | abandon_line | False |  |
| train | r-14faeaaa | exp-06 | Qwen/Qwen3-4B-Base | 685 | rft | base_model | derived:exp-03 | completed | 0.9266666666666666 | supported | adopt | True |  |
| train | r-14faeaaa | exp-07 | Qwen/Qwen3-4B-Base | 798 | grpo | exp-06 | HF openai/gsm8k | failed |  | inconclusive | abandon_line | False |  |
| train | r-14faeaaa | exp-08 | Qwen/Qwen3-4B-Base | 849 | grpo | exp-06 | HF openai/gsm8k | completed | 0.9133333333333333 | contradicted | reject | False |  |
| train | r-14faeaaa | exp-09 | Qwen/Qwen3-4B-Base | 911 | grpo | exp-06 | derived:exp-06 (GSM8K train indices selected by per-question success rate) | killed | 0.92 | inconclusive | adopt | False |  |
| train | r-14faeaaa | exp-10 | Qwen/Qwen3-4B-Base | 936 | rft | base_model | derived:exp-06 | completed | 0.9133333333333333 | contradicted | reject | False |  |
| train | r-14faeaaa | exp-11 | Qwen/Qwen3-4B-Base | 1005 | merge | exp-06 |  | completed | 0.92 | inconclusive | adopt | False |  |
| train | r-14faeaaa | exp-12 | Qwen/Qwen3-4B-Base | 1013 | decode-config | exp-06 |  | completed | 0.8466666666666667 | contradicted | reject | False |  |
| train | r-14faeaaa | exp-13 | Qwen/Qwen3-4B-Base | 1021 | grpo | exp-06 | derived:exp-06 (GSM8K train indices selected by per-question success rate) | completed | 0.9266666666666666 | inconclusive | reject | False |  |
| train | r-14faeaaa | exp-14 | Qwen/Qwen3-4B-Base | 1028 | other | exp-06 |  | completed | 0.9066666666666666 | inconclusive | reject | False |  |
| train | r-14faeaaa | exp-15 | Qwen/Qwen3-4B-Base | 1094 | merge | exp-06 |  | completed | 0.92 | inconclusive | reject | False |  |
| train | r-14faeaaa | exp-16 | Qwen/Qwen3-4B-Base | 1110 | other | exp-11 |  | completed | 0.9266666666666666 | supported | adopt | False | 0.8817285822592873 |
| train | r-14faeaaa | exp-17 | Qwen/Qwen3-4B-Base | 1133 | merge | exp-06 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-185ac8a3 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 92 | sft | base_model | meta-math/MetaMathQA | completed | 0.8 | inconclusive | adopt | False | 0.6307808946171342 |
| train | r-1fff43fc | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 173 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.533 | inconclusive | adopt | False |  |
| train | r-1fff43fc | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 320 | other | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| train | r-1fff43fc | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 346 | decode-config | exp-01 |  | completed | 0.733 | supported | adopt | True |  |
| train | r-1fff43fc | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 360 | decode-config | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-1fff43fc | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 415 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self | completed | 0.767 | supported | adopt | True |  |
| train | r-1fff43fc | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 450 | other | exp-05 |  | completed |  | inconclusive | adopt | False |  |
| train | r-1fff43fc | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 522 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-1fff43fc | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 548 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self + meta-math/MetaMathQA | completed | 0.756 | supported | adopt | True |  |
| train | r-1fff43fc | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 609 | other | exp-08 |  | completed | 0.793 | inconclusive | adopt | False |  |
| train | r-1fff43fc | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 649 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self + meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | True |  |
| train | r-1fff43fc | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 673 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self + meta-math/MetaMathQA | completed | 0.71 | contradicted | reject | True |  |
| train | r-1fff43fc | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 915 | sft | base_model | openai/gsm8k (main, train split) + synthetic:self + meta-math/MetaMathQA | completed | 0.793 | supported | adopt | True |  |
| train | r-1fff43fc | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 982 | other | exp-12 |  | completed | 0.78 | inconclusive | adopt | False | 0.7725549658832449 |
| train | r-2354e591 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 106 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | False |  |
| train | r-2354e591 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 130 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | False |  |
| train | r-2354e591 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 146 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-2354e591 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 160 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | completed |  | inconclusive | adopt | False |  |
| train | r-2354e591 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 195 | merge | exp-04 |  | completed | 0.5866666666666667 | inconclusive | reject | False |  |
| train | r-2354e591 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 217 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-2354e591 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 228 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | completed | 0.5866666666666667 | contradicted | reject | True |  |
| train | r-2354e591 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 265 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA,microsoft/orca-math-word-problems-200k | completed |  | inconclusive | adopt | True |  |
| train | r-2354e591 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 289 | merge | exp-08 |  | completed | 0.667 | supported | adopt | False |  |
| train | r-2354e591 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 299 | other | exp-09 |  | completed | 0.54 | inconclusive | adopt | False |  |
| train | r-2354e591 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 304 | sft | exp-09 | openai/gsm8k,meta-math/MetaMathQA | completed |  | inconclusive | adopt | True | 0.6171341925701289 |
| train | r-2354e591 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 328 | merge | exp-11 |  | completed | 0.6133333333333333 | contradicted | reject | False |  |
| train | r-23aab620 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 86 | sft | base_model | meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-23aab620 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 102 | sft | base_model | meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-23aab620 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 130 | sft | base_model | meta-math/MetaMathQA | completed |  | inconclusive | adopt | False | 0.3813495072024261 |
| train | r-2aeedf08 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 118 | sft | base_model | HF:openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | False |  |
| train | r-2aeedf08 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 157 | merge | exp-01 |  | completed | 0.62 | inconclusive | adopt | False |  |
| train | r-2aeedf08 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 173 | sft | base_model | HF:openai/gsm8k (configs main + socratic, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-2aeedf08 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 203 | other | exp-02 |  | completed | 0.54 | supported | adopt | False | 0.5913570887035633 |
| train | r-2c6de1f4 | exp-01 | Qwen/Qwen3-1.7B-Base | 65 | sft | base_model | meta-math/MetaMathQA | completed | 0.38 | inconclusive | reject | False |  |
| train | r-2c6de1f4 | exp-02 | Qwen/Qwen3-1.7B-Base | 85 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.42 | supported | reject | True |  |
| train | r-2c6de1f4 | exp-03 | Qwen/Qwen3-1.7B-Base | 107 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.22 | contradicted | reject | True |  |
| train | r-2c6de1f4 | exp-04 | Qwen/Qwen3-1.7B-Base | 126 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.5 | supported | adopt | True | 0.535253980288097 |
| train | r-2c6de1f4 | exp-05 | Qwen/Qwen3-1.7B-Base | 146 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-2cd75f43 | exp-01 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | HF gsm8k (main) train split | completed | 0.04 | inconclusive | abandon_line | False |  |
| train | r-2cd75f43 | exp-02 | Qwen/Qwen3-4B-Base | 151 | sft | base_model | HF gsm8k (main) train split | completed | 0.48 | inconclusive | adopt | True |  |
| train | r-2cd75f43 | exp-03 | Qwen/Qwen3-4B-Base | 213 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-2cd75f43 | exp-04 | Qwen/Qwen3-4B-Base | 229 | sft | base_model | HF gsm8k (main) train split + HF meta-math/MetaMathQA | completed | 0.48 | contradicted | reject | True |  |
| train | r-2cd75f43 | exp-05 | Qwen/Qwen3-4B-Base | 278 | sft | base_model | HF gsm8k (main) train split | completed | 0.713 | supported | adopt | False |  |
| train | r-2cd75f43 | exp-06 | Qwen/Qwen3-4B-Base | 298 | other | exp-05 |  | completed | 0.7 | inconclusive | adopt | False | 0.623199393479909 |
| train | r-2cd75f43 | exp-07 | Qwen/Qwen3-4B-Base | 308 | sft | base_model | HF gsm8k (main) train split + HF meta-math/MetaMathQA (GSM_Rephrased only) | completed | 0.627 | contradicted | reject | True |  |
| train | r-2cd75f43 | exp-08 | Qwen/Qwen3-4B-Base | 332 | sft | base_model | HF gsm8k (main) train split | completed | 0.607 | contradicted | reject | False |  |
| train | r-2cd75f43 | exp-09 | Qwen/Qwen3-4B-Base | 348 | sft | base_model | HF gsm8k (main) train split | completed | 0.613 | contradicted | reject | False |  |
| train | r-2cd75f43 | exp-10 | Qwen/Qwen3-4B-Base | 375 | sft | base_model | HF gsm8k (main) train split | completed | 0.68 | contradicted | reject | False |  |
| train | r-2cd75f43 | exp-11 | Qwen/Qwen3-4B-Base | 387 | sft | base_model | HF gsm8k (main) train split | completed | 0.547 | contradicted | reject | False |  |
| train | r-2f4530d4 | exp-01 | Qwen/Qwen3-4B-Base | 61 | sft | base_model | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | failed |  | inconclusive | iterate | True |  |
| train | r-2f4530d4 | exp-02 | Qwen/Qwen3-4B-Base | 71 | sft | base_model | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed |  | inconclusive | adopt | True |  |
| train | r-2f4530d4 | exp-03 | Qwen/Qwen3-4B-Base | 84 | merge | exp-02 |  | completed | 0.06 | inconclusive | adopt | False |  |
| train | r-2f4530d4 | exp-04 | Qwen/Qwen3-4B-Base | 104 | decode-config | exp-03 |  | completed | 0.08 | inconclusive | adopt | False |  |
| train | r-2f4530d4 | exp-05 | Qwen/Qwen3-4B-Base | 121 | decode-config | exp-04 |  | completed | 0.04 | inconclusive | abandon_line | False |  |
| train | r-2f4530d4 | exp-06 | Qwen/Qwen3-4B-Base | 139 | sft | base_model | openai/gsm8k (train),meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed | 0.38 | inconclusive | reject | True |  |
| train | r-2f4530d4 | exp-07 | Qwen/Qwen3-4B-Base | 159 | sft | base_model | openai/gsm8k (train),meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed | 0.747 | supported | adopt | True |  |
| train | r-2f4530d4 | exp-08 | Qwen/Qwen3-4B-Base | 192 | sft | base_model | openai/gsm8k (train),meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR),microsoft/orca-math-word-problems-200k | killed | 0.76 | inconclusive | adopt | True |  |
| train | r-2f4530d4 | exp-09 | Qwen/Qwen3-4B-Base | 238 | other | exp-08 |  | completed |  | inconclusive | adopt | False |  |
| train | r-2f4530d4 | exp-10 | Qwen/Qwen3-4B-Base | 245 | decode-config | exp-09 |  | completed | 0.733 | inconclusive | reject | False |  |
| train | r-2f4530d4 | exp-11 | Qwen/Qwen3-4B-Base | 255 | decode-config | exp-10 |  | completed | 0.747 | inconclusive | adopt | False | 0.7581501137225171 |
| train | r-2fa3e9e7 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 122 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.76 | inconclusive | adopt | True |  |
| train | r-2fa3e9e7 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 394 | sft | exp-01 | HF microsoft/orca-math-word-problems-200k | completed | 0.48 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 426 | sft | exp-01 | HF microsoft/orca-math-word-problems-200k | completed | 0.47333333333333333 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 444 | sft | exp-01 | HF openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-2fa3e9e7 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 452 | sft | exp-01 | HF openai/gsm8k (main, split=train) | completed | 0.38666666666666666 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 468 | sft | exp-01 | HF openai/gsm8k (main, split=train) | completed | 0.4666666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 509 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.72 | inconclusive | adopt | True |  |
| train | r-2fa3e9e7 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 557 | merge | exp-01 |  | completed | 0.5533333333333333 | inconclusive | reject | True |  |
| train | r-2fa3e9e7 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 564 | merge | exp-01 |  | completed | 0.7466666666666667 | supported | reject | True |  |
| train | r-2fa3e9e7 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 564 | merge | exp-01 |  | completed | 0.5533333333333333 | inconclusive | reject | True |  |
| train | r-2fa3e9e7 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 574 | merge | exp-01 |  | completed | 0.6066666666666667 | supported | reject | True |  |
| train | r-2fa3e9e7 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 574 | merge | exp-01 |  | completed | 0.6066666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 574 | merge | exp-01 |  | completed | 0.6333333333333333 | supported | adopt | True |  |
| train | r-2fa3e9e7 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 574 | merge | exp-01 |  | completed | 0.76 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 603 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.7 | supported | adopt | True |  |
| train | r-2fa3e9e7 | exp-16 | HuggingFaceTB/SmolLM3-3B-Base | 644 | merge | exp-07 |  | completed | 0.5733333333333334 | contradicted | adopt | True |  |
| train | r-2fa3e9e7 | exp-17 | HuggingFaceTB/SmolLM3-3B-Base | 644 | merge | exp-01 |  | completed | 0.74 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-18 | HuggingFaceTB/SmolLM3-3B-Base | 644 | merge | exp-01 |  | completed | 0.7466666666666667 | inconclusive | reject | True |  |
| train | r-2fa3e9e7 | exp-19 | HuggingFaceTB/SmolLM3-3B-Base | 644 | merge | exp-01 |  | completed | 0.7533333333333333 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-20 | HuggingFaceTB/SmolLM3-3B-Base | 691 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.7333333333333333 | contradicted | adopt | True |  |
| train | r-2fa3e9e7 | exp-21 | HuggingFaceTB/SmolLM3-3B-Base | 737 | merge | exp-01 |  | completed | 0.74 | contradicted | reject | False |  |
| train | r-2fa3e9e7 | exp-22 | HuggingFaceTB/SmolLM3-3B-Base | 962 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.06 | contradicted | adopt | True |  |
| train | r-2fa3e9e7 | exp-23 | HuggingFaceTB/SmolLM3-3B-Base | 1140 | sft | exp-22 | HF meta-math/MetaMathQA | completed | 0.37333333333333335 | inconclusive | adopt | False |  |
| train | r-2fa3e9e7 | exp-24 | HuggingFaceTB/SmolLM3-3B-Base | 1164 | sft | exp-23 | HF meta-math/MetaMathQA | completed | 0.4266666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-25 | HuggingFaceTB/SmolLM3-3B-Base | 1197 | merge | exp-01 |  | completed | 0.7533333333333333 | contradicted | reject | False |  |
| train | r-2fa3e9e7 | exp-26 | HuggingFaceTB/SmolLM3-3B-Base | 1208 | sft | exp-13 | HF meta-math/MetaMathQA | completed | 0.7266666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-27 | HuggingFaceTB/SmolLM3-3B-Base | 1229 | merge | exp-01 |  | completed | 0.76 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-28 | HuggingFaceTB/SmolLM3-3B-Base | 1230 | merge | exp-01 |  | completed | 0.7466666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-29 | HuggingFaceTB/SmolLM3-3B-Base | 1252 | merge | exp-01 |  | completed | 0.74 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-30 | HuggingFaceTB/SmolLM3-3B-Base | 1375 | merge | exp-01 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-2fa3e9e7 | exp-31 | HuggingFaceTB/SmolLM3-3B-Base | 1383 | merge | exp-01 |  | completed | 0.8 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-32 | HuggingFaceTB/SmolLM3-3B-Base | 1383 | merge | exp-01 |  | completed | 0.76 | inconclusive | reject | True |  |
| train | r-2fa3e9e7 | exp-33 | HuggingFaceTB/SmolLM3-3B-Base | 1383 | merge | exp-01 |  | completed | 0.7466666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-34 | HuggingFaceTB/SmolLM3-3B-Base | 1383 | merge | exp-01 |  | completed | 0.7266666666666667 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-35 | HuggingFaceTB/SmolLM3-3B-Base | 1383 | merge | exp-01 |  | completed | 0.74 | contradicted | reject | True |  |
| train | r-2fa3e9e7 | exp-36 | HuggingFaceTB/SmolLM3-3B-Base | 1397 | other | exp-01 |  | completed | 0.7452615617892343 | supported | adopt | True | 0.7414708112206216 |
| train | r-2fa3e9e7 | exp-37 | HuggingFaceTB/SmolLM3-3B-Base | 1464 | sft | exp-01 | HF meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | True |  |
| train | r-2fa3e9e7 | exp-38 | HuggingFaceTB/SmolLM3-3B-Base | 1472 | sft | exp-01 | HF openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-33444a20 | exp-01 | Qwen/Qwen3-1.7B-Base | 80 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | killed |  | inconclusive | abandon_line | False |  |
| train | r-33444a20 | exp-02 | Qwen/Qwen3-1.7B-Base | 84 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | completed | 0.127 | inconclusive | adopt | False |  |
| train | r-33444a20 | exp-03 | Qwen/Qwen3-1.7B-Base | 102 | decode-config | exp-02 |  | completed | 0.24 | supported | reject | False |  |
| train | r-33444a20 | exp-04 | Qwen/Qwen3-1.7B-Base | 120 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | completed | 0.6333333333333333 | supported | adopt | False |  |
| train | r-33444a20 | exp-05 | Qwen/Qwen3-1.7B-Base | 160 | other | exp-04 |  | completed | 0.6133333333333333 | inconclusive | adopt | False | 0.6004548900682335 |
| train | r-358cab54 | exp-01 | Qwen/Qwen3-4B-Base | 116 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.403 | inconclusive | adopt | True |  |
| train | r-358cab54 | exp-02 | Qwen/Qwen3-4B-Base | 213 | decode-config | exp-01 |  | completed | 0.8166666666666667 | supported | adopt | True |  |
| train | r-358cab54 | exp-03 | Qwen/Qwen3-4B-Base | 270 | rft | base_model | derived:exp-01 data + synthetic:self (sampled from sft_v1),synthetic:self | completed | 0.82 | inconclusive | adopt | False |  |
| train | r-358cab54 | exp-04 | Qwen/Qwen3-4B-Base | 380 | grpo | exp-03 | HF openai/gsm8k (main, train split), loaded in-process by grpo.py | killed |  | inconclusive | abandon_line | True |  |
| train | r-358cab54 | exp-05 | Qwen/Qwen3-4B-Base | 437 | grpo | exp-03 | HF openai/gsm8k (main, train split), loaded in-process by grpo.py | killed | 0.8266666666666667 | inconclusive | adopt | True |  |
| train | r-358cab54 | exp-06 | Qwen/Qwen3-4B-Base | 534 | sft | base_model | derived:exp-03 mix + HF meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | True |  |
| train | r-358cab54 | exp-07 | Qwen/Qwen3-4B-Base | 542 | sft | base_model | derived:exp-03 mix + HF meta-math/MetaMathQA,HF meta-math/MetaMathQA (train) | killed | 0.8466666666666667 | supported | adopt | True |  |
| train | r-358cab54 | exp-08 | Qwen/Qwen3-4B-Base | 647 | sft | base_model | derived:exp-03 mix + HF meta-math/MetaMathQA,HF meta-math/MetaMathQA (train) | completed | 0.8333333333333334 | contradicted | reject | True |  |
| train | r-358cab54 | exp-09 | Qwen/Qwen3-4B-Base | 782 | grpo | exp-07 | HF openai/gsm8k (main, train split), loaded in-process by grpo2.py | failed |  | inconclusive | abandon_line | True |  |
| train | r-358cab54 | exp-10 | Qwen/Qwen3-4B-Base | 791 | grpo | exp-07 | HF openai/gsm8k (main, train split), loaded in-process by grpo2.py | killed | 0.8633333333333333 | supported | adopt | True |  |
| train | r-358cab54 | exp-11 | Qwen/Qwen3-4B-Base | 869 | grpo | exp-10 | HF openai/gsm8k (main, train split), loaded in-process by grpo2.py | killed | 0.8766666666666667 | supported | adopt | True |  |
| train | r-358cab54 | exp-12 | Qwen/Qwen3-4B-Base | 926 | grpo | exp-11 | HF openai/gsm8k (main, train split), loaded in-process by grpo2.py | killed | 0.8866666666666667 | inconclusive | adopt | True |  |
| train | r-358cab54 | exp-13 | Qwen/Qwen3-4B-Base | 1022 | other | exp-12 |  | completed | 0.88 | supported | adopt | True | 0.8711144806671721 |
| train | r-3911d1bb | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 70 | sft | base_model | HF:openai/gsm8k (config main, train split) | failed |  | inconclusive | iterate | True |  |
| train | r-3911d1bb | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 79 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-3911d1bb | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 85 | merge | exp-02 |  | completed |  | inconclusive | reject | True |  |
| train | r-3911d1bb | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 106 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-3911d1bb | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 111 | merge | exp-04 |  | completed | 0.35 | inconclusive | reject | True |  |
| train | r-3911d1bb | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 120 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-3911d1bb | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 127 | merge | exp-06 |  | completed |  | inconclusive | reject | False |  |
| train | r-3911d1bb | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 138 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-3911d1bb | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 145 | merge | exp-08 |  | completed |  | inconclusive | reject | False |  |
| train | r-3911d1bb | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-3911d1bb | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 157 | merge | exp-10 |  | completed | 0.35 | inconclusive | reject | False |  |
| train | r-3911d1bb | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 170 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-3911d1bb | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 185 | merge | exp-12 |  | completed | 0.34 | inconclusive | adopt | False | 0.3502653525398029 |
| train | r-3fa66d9b | exp-01 | Qwen/Qwen3-4B-Base | 80 | sft | base_model | HF id: openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-02 | Qwen/Qwen3-4B-Base | 86 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-03 | Qwen/Qwen3-4B-Base | 90 | sft | base_model | HF id: openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-04 | Qwen/Qwen3-4B-Base | 137 | merge | exp-03 |  | completed | 0.13333333333333333 | contradicted | reject | False |  |
| train | r-3fa66d9b | exp-05 | Qwen/Qwen3-4B-Base | 170 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-06 | Qwen/Qwen3-4B-Base | 174 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed | 0.5833333333333334 | supported | adopt | True |  |
| train | r-3fa66d9b | exp-07 | Qwen/Qwen3-4B-Base | 199 | other | exp-06 |  | completed | 0.58 | inconclusive | adopt | False | 0.5837755875663382 |
| train | r-4254277e | exp-01 | Qwen/Qwen3-1.7B-Base | 118 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.053 | inconclusive | reject | True |  |
| train | r-4254277e | exp-02 | Qwen/Qwen3-1.7B-Base | 195 | sft | base_model | openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| train | r-4254277e | exp-03 | Qwen/Qwen3-1.7B-Base | 212 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.04 | contradicted | adopt | True |  |
| train | r-4254277e | exp-04 | Qwen/Qwen3-1.7B-Base | 382 | decode-config | exp-03 |  | completed | 0.64 | supported | adopt | True |  |
| train | r-4254277e | exp-05 | Qwen/Qwen3-1.7B-Base | 409 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA (GSM_AnsAug subset),meta-math/MetaMathQA (GSM_Rephrased subset) | failed |  | inconclusive | abandon_line | False |  |
| train | r-4254277e | exp-06 | Qwen/Qwen3-1.7B-Base | 435 | other | exp-04 |  | completed |  | inconclusive | adopt | False | 0.6762699014404853 |
| train | r-4254277e | exp-07 | Qwen/Qwen3-1.7B-Base | 649 | rft | base_model | synthetic:self (sampled from the exp-04 checkpoint, work/sft_gsm8k_fs_v2),openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-426bd770 | exp-01 | Qwen/Qwen3-1.7B-Base | 405 | sft | base_model | HF nvidia/OpenMathInstruct-2 (train_1M subset) + HF openai/gsm8k (train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-426bd770 | exp-02 | Qwen/Qwen3-1.7B-Base | 436 | sft | base_model | HF nvidia/OpenMathInstruct-2 (train_1M subset) + HF openai/gsm8k (train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-426bd770 | exp-03 | Qwen/Qwen3-1.7B-Base | 525 | sft | base_model | HF nvidia/OpenMathInstruct-2 (train_1M subset) + HF openai/gsm8k (train split) | completed | 0.82 | inconclusive | adopt | False |  |
| train | r-426bd770 | exp-04 | Qwen/Qwen3-1.7B-Base | 866 | grpo | exp-03 | HF openai/gsm8k (train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-426bd770 | exp-05 | Qwen/Qwen3-1.7B-Base | 918 | other | exp-03 |  | completed |  | inconclusive | reject | False |  |
| train | r-426bd770 | exp-06 | Qwen/Qwen3-1.7B-Base | 940 | grpo | exp-03 | HF openai/gsm8k (train split) | killed | 0.858 | supported | adopt | False |  |
| train | r-426bd770 | exp-07 | Qwen/Qwen3-1.7B-Base | 1042 | merge | exp-03 |  | completed | 0.8522 | supported | adopt | False |  |
| train | r-426bd770 | exp-08 | Qwen/Qwen3-1.7B-Base | 1094 | decode-config | exp-03 |  | completed | 0.858 | supported | adopt | False |  |
| train | r-426bd770 | exp-09 | Qwen/Qwen3-1.7B-Base | 1139 | merge | exp-03 |  | failed |  | inconclusive | abandon_line | False |  |
| train | r-426bd770 | exp-10 | Qwen/Qwen3-1.7B-Base | 1155 | merge | exp-03 |  | completed | 0.848 | contradicted | reject | False |  |
| train | r-426bd770 | exp-11 | Qwen/Qwen3-1.7B-Base | 1192 | merge | exp-03 |  | completed |  | inconclusive | reject | False |  |
| train | r-426bd770 | exp-12 | Qwen/Qwen3-1.7B-Base | 1208 | other | exp-07 |  | completed | 0.8522 | supported | adopt | False |  |
| train | r-426bd770 | exp-13 | Qwen/Qwen3-1.7B-Base | 1242 | grpo | exp-06 | HF openai/gsm8k (train split) | completed |  | inconclusive | adopt | False | 0.844579226686884 |
| train | r-426bd770 | exp-14 | Qwen/Qwen3-1.7B-Base | 1262 | merge | exp-03 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 532 | sft | base_model | derived: HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows of the first 4 parquet shards) + HF openai/gsm8k train + HF meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_FOBAR, GSM_SV) | killed |  | inconclusive | abandon_line | False |  |
| train | r-4463b5d3 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 632 | sft | base_model | derived: HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows of the first 4 parquet shards) + HF openai/gsm8k train + HF meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_FOBAR, GSM_SV),HF openai/gsm8k (main, train split) | completed | 0.695 | inconclusive | adopt | False |  |
| train | r-4463b5d3 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 896 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 1040 | grpo | exp-02 | derived:exp-02 data lineage - data/rft_pool.jsonl (gsm8k train questions + 22000 OpenMathInstruct-2 augmented_gsm8k problems, 29473 rows [683]) plus data/rl_pool.jsonl (up to 40000 further augmented_gsm8k problems from parquet shards 4-14, never trained on) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries unused elsewhere | failed |  | inconclusive | abandon_line | False |  |
| train | r-4463b5d3 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 1079 | grpo | exp-02 | derived:exp-02 data lineage - data/rft_pool.jsonl (gsm8k train questions + 22000 OpenMathInstruct-2 augmented_gsm8k problems, 29473 rows [683]) plus data/rl_pool.jsonl (up to 40000 further augmented_gsm8k problems from parquet shards 4-14, never trained on) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries unused elsewhere | killed | 0.866 | supported | adopt | False |  |
| train | r-4463b5d3 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 1245 | other | exp-05 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 1247 | grpo | exp-05 | derived:exp-02 data lineage - data/rft_pool.jsonl (29473 rows) plus data/rl_pool.jsonl (unused OpenMathInstruct-2 augmented_gsm8k problems) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries | killed | 0.89 | inconclusive | adopt | False |  |
| train | r-4463b5d3 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1325 | other | exp-07 |  | completed | 0.8696 | inconclusive | adopt | False | 0.8605003790750568 |
| train | r-46a821e3 | exp-01 | Qwen/Qwen3-1.7B-Base | 35 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.02 | contradicted | reject | True |  |
| train | r-46a821e3 | exp-02 | Qwen/Qwen3-1.7B-Base | 121 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | adopt | True |  |
| train | r-46a821e3 | exp-03 | Qwen/Qwen3-1.7B-Base | 142 | merge | exp-02 |  | completed | 0.1 | inconclusive | adopt | True |  |
| train | r-46a821e3 | exp-04 | Qwen/Qwen3-1.7B-Base | 153 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-46a821e3 | exp-05 | Qwen/Qwen3-1.7B-Base | 182 | other | exp-03 |  | completed |  | inconclusive | adopt | True | 0.10841546626231995 |
| train | r-46dad8f8 | exp-01 | Qwen/Qwen3-1.7B-Base | 111 | sft | base_model | local (built from HF microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA GSM_Rephrased/GSM_SV/GSM_FOBAR, nvidia/OpenMathInstruct-2 augmented_gsm8k) | killed |  | inconclusive | abandon_line | True |  |
| train | r-46dad8f8 | exp-02 | Qwen/Qwen3-1.7B-Base | 124 | sft | base_model | local (built from HF microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA GSM_Rephrased/GSM_SV/GSM_FOBAR, nvidia/OpenMathInstruct-2 augmented_gsm8k) | killed |  | inconclusive | abandon_line | False |  |
| train | r-46dad8f8 | exp-03 | Qwen/Qwen3-1.7B-Base | 146 | sft | base_model | local (built from HF microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA GSM_Rephrased/GSM_SV/GSM_FOBAR, nvidia/OpenMathInstruct-2 augmented_gsm8k) | killed |  | inconclusive | adopt | False |  |
| train | r-46dad8f8 | exp-04 | Qwen/Qwen3-1.7B-Base | 287 | decode-config | exp-03 |  | completed | 0.3933333333333333 | inconclusive | adopt | True |  |
| train | r-46dad8f8 | exp-05 | Qwen/Qwen3-1.7B-Base | 298 | sft | exp-03 | local (built from HF microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA GSM_Rephrased/GSM_SV/GSM_FOBAR, nvidia/OpenMathInstruct-2 augmented_gsm8k) | completed | 0.44 | supported | adopt | True |  |
| train | r-46dad8f8 | exp-06 | Qwen/Qwen3-1.7B-Base | 617 | sft | exp-05 | local (built from HF openai/gsm8k train gold, meta-math/MetaMathQA GSM_AnsAug, nvidia/OpenMathInstruct-2 gsm8k) | completed | 0.4066666666666667 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-07 | Qwen/Qwen3-1.7B-Base | 737 | dpo | exp-05 | synthetic:self (sampled from runs/broad) over HF openai/gsm8k train questions | completed | 0.38 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-08 | Qwen/Qwen3-1.7B-Base | 766 | sft | exp-05 | local (HF openai/gsm8k train split only) | completed | 0.44666666666666666 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-09 | Qwen/Qwen3-1.7B-Base | 800 | sft | exp-05 | local (HF openai/gsm8k train split only) | completed | 0.5533333333333333 | supported | reject | True |  |
| train | r-46dad8f8 | exp-10 | Qwen/Qwen3-1.7B-Base | 836 | sft | exp-05 | local (HF openai/gsm8k train split only) | completed | 0.62 | supported | adopt | True |  |
| train | r-46dad8f8 | exp-11 | Qwen/Qwen3-1.7B-Base | 871 | rft | exp-10 | synthetic:self (sampled from runs/eos_binary) over HF openai/gsm8k train questions | completed | 0.5733333333333334 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-12 | Qwen/Qwen3-1.7B-Base | 972 | dpo | exp-10 | synthetic:self (sampled from runs/eos_binary) over HF openai/gsm8k train questions | completed | 0.58 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-13 | Qwen/Qwen3-1.7B-Base | 1010 | sft | exp-10 | local (built from HF microsoft/orca-math-word-problems-200k, meta-math/MetaMathQA GSM_Rephrased/GSM_SV/GSM_FOBAR, nvidia/OpenMathInstruct-2 augmented_gsm8k) | completed | 0.5666666666666667 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-14 | Qwen/Qwen3-1.7B-Base | 1045 | merge | exp-10 |  | completed | 0.62 | inconclusive | reject | True |  |
| train | r-46dad8f8 | exp-15 | Qwen/Qwen3-1.7B-Base | 1057 | sft | exp-10 | local (HF openai/gsm8k train split only) | completed | 0.62 | supported | adopt | True |  |
| train | r-46dad8f8 | exp-16 | Qwen/Qwen3-1.7B-Base | 1089 | merge | exp-05 |  | completed | 0.42 | supported | adopt | True |  |
| train | r-46dad8f8 | exp-17 | Qwen/Qwen3-1.7B-Base | 1098 | sft | exp-16 | local (HF openai/gsm8k train split only) | completed | 0.6066666666666667 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-18 | Qwen/Qwen3-1.7B-Base | 1158 | merge | exp-10 |  | completed | 0.6057619408642911 | contradicted | reject | True |  |
| train | r-46dad8f8 | exp-19 | Qwen/Qwen3-1.7B-Base | 1174 | merge | exp-15 |  | completed | 0.6333333333333333 | inconclusive | adopt | False | 0.6239575435936315 |
| train | r-47a7873c | exp-01 | Qwen/Qwen3-4B-Base | 56 | sft | base_model | openai/gsm8k (config "main", train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-47a7873c | exp-02 | Qwen/Qwen3-4B-Base | 62 | sft | base_model | openai/gsm8k (config "main", train split) | completed | 0.6 | inconclusive | reject | False |  |
| train | r-47a7873c | exp-03 | Qwen/Qwen3-4B-Base | 110 | sft | base_model | openai/gsm8k (config "main", train split) | completed |  | inconclusive | adopt | True |  |
| train | r-47a7873c | exp-04 | Qwen/Qwen3-4B-Base | 131 | other | exp-03 |  | completed |  | inconclusive | adopt | True | 0.42608036391205456 |
| train | r-4a80d272 | exp-01 | Qwen/Qwen3-4B-Base | 74 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | killed |  | inconclusive | abandon_line | True |  |
| train | r-4a80d272 | exp-02 | Qwen/Qwen3-4B-Base | 86 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | failed |  | inconclusive | abandon_line | True |  |
| train | r-4a80d272 | exp-03 | Qwen/Qwen3-4B-Base | 91 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | completed | 0.04 | inconclusive | reject | True |  |
| train | r-4a80d272 | exp-04 | Qwen/Qwen3-4B-Base | 161 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | failed |  | inconclusive | abandon_line | True |  |
| train | r-4a80d272 | exp-05 | Qwen/Qwen3-4B-Base | 164 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | completed | 0.7267 | supported | reject | True |  |
| train | r-4a80d272 | exp-06 | Qwen/Qwen3-4B-Base | 299 | decode-config | exp-03 |  | completed | 0.06 | contradicted | reject | True |  |
| train | r-4a80d272 | exp-07 | Qwen/Qwen3-4B-Base | 327 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | completed | 0.7667 | supported | adopt | True |  |
| train | r-4a80d272 | exp-08 | Qwen/Qwen3-4B-Base | 430 | other | exp-07 |  | completed | 0.76 | inconclusive | adopt | False |  |
| train | r-4a80d272 | exp-09 | Qwen/Qwen3-4B-Base | 484 | decode-config | exp-08 |  | completed | 0.7933 | supported | adopt | True |  |
| train | r-4a80d272 | exp-10 | Qwen/Qwen3-4B-Base | 511 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | completed | 0.7 | contradicted | reject | True |  |
| train | r-4a80d272 | exp-11 | Qwen/Qwen3-4B-Base | 641 | merge | exp-10 |  | completed | 0.72 | inconclusive | reject | True |  |
| train | r-4a80d272 | exp-12 | Qwen/Qwen3-4B-Base | 680 | sft | base_model | HF: openai/gsm8k (main, train), plain prompt,HF: openai/gsm8k (main, train), exact harness prompt,HF: clarkkitchen22/SynthGSM8K-50K (train),HF: HuggingFaceH4/orca-math-word-problems-200k (train_sft) | completed | 0.8133 | contradicted | adopt | True |  |
| train | r-4a80d272 | exp-13 | Qwen/Qwen3-4B-Base | 757 | merge | exp-12 |  | completed | 0.8 | supported | adopt | True |  |
| train | r-4a80d272 | exp-14 | Qwen/Qwen3-4B-Base | 828 | other | exp-13 |  | completed |  | inconclusive | adopt | False |  |
| train | r-4a80d272 | exp-15 | Qwen/Qwen3-4B-Base | 831 | merge | exp-07 |  | completed | 0.7733 | contradicted | reject | True |  |
| train | r-4a80d272 | exp-16 | Qwen/Qwen3-4B-Base | 836 | decode-config | exp-14 |  | completed | 0.8 | supported | adopt | True |  |
| train | r-4a80d272 | exp-17 | Qwen/Qwen3-4B-Base | 860 | decode-config | exp-14 |  | completed | 0.7933 | contradicted | reject | True |  |
| train | r-4a80d272 | exp-18 | Qwen/Qwen3-4B-Base | 866 | decode-config | exp-12 |  | completed | 0.8 | contradicted | reject | True |  |
| train | r-4a80d272 | exp-19 | Qwen/Qwen3-4B-Base | 897 | decode-config | exp-17 |  | completed | 0.7915 | inconclusive | reject | True |  |
| train | r-4a80d272 | exp-20 | Qwen/Qwen3-4B-Base | 913 | other | exp-13 |  | completed | 0.7908 | inconclusive | adopt | True |  |
| train | r-4a80d272 | exp-21 | Qwen/Qwen3-4B-Base | 939 | decode-config | exp-20 |  | completed | 0.8067 | supported | adopt | True | 0.7915087187263078 |
| train | r-4aa3d061 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 164 | sft | base_model | HF id openai/gsm8k (config main, split train) | killed | 0.553 | inconclusive | adopt | True |  |
| train | r-4aa3d061 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 348 | other | exp-01 |  | completed | 0.35 | supported | adopt | True | 0.5284306292645944 |
| train | r-4d0f7a19 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 98 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split),openai/gsm8k (main, train split) | completed | 0.5666666666666667 | inconclusive | adopt | True |  |
| train | r-4d0f7a19 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 227 | sft | exp-01 | openai/gsm8k (main, train split) | completed | 0.54 | contradicted | reject | True |  |
| train | r-4d0f7a19 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 293 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split),openai/gsm8k (main, train split) | completed | 0.5266666666666666 | contradicted | reject | True |  |
| train | r-4d0f7a19 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 414 | other | exp-01 |  | completed | 0.6 | supported | adopt | True | 0.5231235784685367 |
| train | r-51c52152 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 146 | sft | base_model | HF:openai/gsm8k (main, train split, gold solutions) | completed | 0.5667 | inconclusive | adopt | True |  |
| train | r-51c52152 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 279 | rft | base_model | HF:openai/gsm8k gold x2 + derived:exp-01 (self-sampled, rft_round1.jsonl),synthetic:self (sampled from exp-01's checkpoint) | completed | 0.4 | contradicted | reject | True |  |
| train | r-51c52152 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 378 | sft | base_model | HF:openai/gsm8k gold x1 + HF:meta-math/MetaMathQA (GSM* types only) | completed | 0.42 | contradicted | reject | True |  |
| train | r-51c52152 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 453 | rft | base_model | HF:openai/gsm8k gold x2 + derived:exp-01 (self-sampled, rft_round1.jsonl) | completed | 0.4933 | contradicted | reject | True |  |
| train | r-51c52152 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 499 | sft | base_model | HF:openai/gsm8k (main, train split, gold solutions) | completed | 0.54 | inconclusive | reject | True |  |
| train | r-51c52152 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 533 | other | exp-01 |  | completed | 0.51 | inconclusive | reject | False |  |
| train | r-51c52152 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 537 | rft | base_model | HF:openai/gsm8k gold x2 + derived:exp-01 (self-sampled, rft_round1.jsonl) | completed | 0.3867 | contradicted | reject | False |  |
| train | r-51c52152 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 617 | sft | base_model | HF:openai/gsm8k (main, train split, gold solutions) | completed | 0.5467 | inconclusive | adopt | True |  |
| train | r-51c52152 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 648 | sft | base_model | HF:openai/gsm8k (main, train split, gold solutions) | completed | 0.5133 | contradicted | reject | True |  |
| train | r-51c52152 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 709 | other | exp-08 |  | completed | 0.554 | inconclusive | adopt | False | 0.5284306292645944 |
| train | r-51c52152 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 850 | grpo | exp-10 | HF:openai/gsm8k (main, train split - prompts only, built in-process) | killed | 0.5333 | contradicted | abandon_line | True |  |
| train | r-54aba2d1 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 64 | sft | base_model | HF:openai/gsm8k (config main, train split) | failed |  | inconclusive | iterate | False |  |
| train | r-54aba2d1 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 74 | sft | base_model | HF:openai/gsm8k (config main, train split) | failed |  | inconclusive | iterate | False |  |
| train | r-54aba2d1 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 80 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 94 | merge | exp-03 |  | completed | 0.42 | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 136 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 142 | merge | exp-05 |  | completed | 0.2866666666666667 | contradicted | reject | False |  |
| train | r-54aba2d1 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 148 | merge | exp-03 |  | completed | 0.4094010614101592 | supported | adopt | True | 0.39423805913570886 |
| train | r-5720f98a | exp-01 | Qwen/Qwen3-1.7B-Base | 99 | sft | base_model | HF:openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-02 | Qwen/Qwen3-1.7B-Base | 104 | sft | base_model | HF:openai/gsm8k (main, train split) | completed | 0.4866666666666667 | inconclusive | adopt | True |  |
| train | r-5720f98a | exp-03 | Qwen/Qwen3-1.7B-Base | 152 | sft | base_model | HF:nvidia/OpenMathInstruct-2 (train_1M) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-04 | Qwen/Qwen3-1.7B-Base | 157 | sft | base_model | HF:nvidia/OpenMathInstruct-2 (train_1M) | completed | 0.22 | contradicted | adopt | True |  |
| train | r-5720f98a | exp-05 | Qwen/Qwen3-1.7B-Base | 255 | sft | exp-04 | HF:openai/gsm8k (main, train split) | completed | 0.12 | contradicted | reject | True |  |
| train | r-5720f98a | exp-06 | Qwen/Qwen3-1.7B-Base | 287 | sft | exp-02 | HF:openai/gsm8k (main, train split) | completed | 0.553 | supported | adopt | True |  |
| train | r-5720f98a | exp-07 | Qwen/Qwen3-1.7B-Base | 367 | rft | exp-06 | HF:openai/gsm8k (main, train split),synthetic:self (exp-06 checkpoint) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-08 | Qwen/Qwen3-1.7B-Base | 371 | rft | exp-06 | HF:openai/gsm8k (main, train split),synthetic:self (exp-06 checkpoint) | completed | 0.6333333333333333 | supported | adopt | True |  |
| train | r-5720f98a | exp-09 | Qwen/Qwen3-1.7B-Base | 404 | rft | exp-08 | HF:openai/gsm8k (main, train split),synthetic:self (exp-06 checkpoint),synthetic:self (exp-08 checkpoint) | completed | 0.58 | contradicted | reject | True |  |
| train | r-5720f98a | exp-10 | Qwen/Qwen3-1.7B-Base | 422 | sft | exp-08 | HF:openai/gsm8k (main, train split),HF:nvidia/OpenMathInstruct-2 (train_1M) | completed | 0.5 | contradicted | reject | True |  |
| train | r-5720f98a | exp-11 | Qwen/Qwen3-1.7B-Base | 448 | sft | exp-08 | derived:exp-08 (verifier coverage of data/self_distilled_v2_nocontext.jsonl) | completed | 0.546 | contradicted | reject | True |  |
| train | r-5720f98a | exp-12 | Qwen/Qwen3-1.7B-Base | 475 | sft | exp-08 | HF:microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-13 | Qwen/Qwen3-1.7B-Base | 522 | sft | exp-08 | HF:microsoft/orca-math-word-problems-200k | completed | 0.5792 | contradicted | reject | True |  |
| train | r-5720f98a | exp-14 | Qwen/Qwen3-1.7B-Base | 610 | merge | exp-08 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-15 | Qwen/Qwen3-1.7B-Base | 769 | other | exp-08 | synthetic:self (exp-08 checkpoint) + GSM8K train gold | completed |  | inconclusive | adopt | True |  |
| train | r-5720f98a | exp-16 | Qwen/Qwen3-1.7B-Base | 833 | sft | exp-15 | HF:openai/gsm8k (main, train split) | completed | 0.534 | contradicted | reject | True |  |
| train | r-5720f98a | exp-17 | Qwen/Qwen3-1.7B-Base | 846 | dpo | exp-08 | synthetic:self (exp-08 checkpoint) + GSM8K train gold | completed | 0.5905989385898408 | supported | adopt | True |  |
| train | r-5720f98a | exp-18 | Qwen/Qwen3-1.7B-Base | 877 | dpo | exp-17 | synthetic:self (exp-17 checkpoint) + GSM8K train gold | completed |  | inconclusive | adopt | True |  |
| train | r-5720f98a | exp-19 | Qwen/Qwen3-1.7B-Base | 891 | sft | exp-18 | HF:openai/gsm8k (main, train split) | completed | 0.547 | contradicted | reject | True |  |
| train | r-5720f98a | exp-20 | Qwen/Qwen3-1.7B-Base | 906 | dpo | exp-17 | synthetic:self (exp-17 checkpoint) + GSM8K train gold | completed | 0.584 | contradicted | reject | True |  |
| train | r-5720f98a | exp-21 | Qwen/Qwen3-1.7B-Base | 968 | grpo | exp-17 | HF:openai/gsm8k (main, train split) | completed | 0.574677786201668 | contradicted | reject | True |  |
| train | r-5720f98a | exp-22 | Qwen/Qwen3-1.7B-Base | 1020 | dpo | exp-08 | derived:data/gsm8k_preferences.jsonl | failed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-23 | Qwen/Qwen3-1.7B-Base | 1025 | dpo | exp-08 | derived:data/gsm8k_preferences.jsonl | completed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-24 | Qwen/Qwen3-1.7B-Base | 1117 | dpo | exp-08 | synthetic:self (exp-08 checkpoint) + GSM8K train gold | completed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-25 | Qwen/Qwen3-1.7B-Base | 1131 | dpo | exp-08 | synthetic:self (exp-08 checkpoint) + GSM8K train gold | completed | 0.5875663381349507 | contradicted | reject | True |  |
| train | r-5720f98a | exp-26 | Qwen/Qwen3-1.7B-Base | 1163 | dpo | exp-08 | synthetic:self (exp-08 checkpoint) + GSM8K train gold | completed | 0.583 | contradicted | reject | True |  |
| train | r-5720f98a | exp-27 | Qwen/Qwen3-1.7B-Base | 1235 | dpo | exp-08 | synthetic:self (exp-08 checkpoint) | completed |  | inconclusive | abandon_line | True |  |
| train | r-5720f98a | exp-28 | Qwen/Qwen3-1.7B-Base | 1258 | sft | exp-17 | HF:openai/gsm8k (main, train split) | completed | 0.575 | contradicted | reject | False |  |
| train | r-5720f98a | exp-29 | Qwen/Qwen3-1.7B-Base | 1270 | dpo | exp-08 | derived:data/gsm8k_preferences.jsonl + data/gsm8k_preferences_local8.jsonl | completed | 0.579 | contradicted | reject | True |  |
| train | r-5720f98a | exp-30 | Qwen/Qwen3-1.7B-Base | 1290 | dpo | exp-08 | derived:data/gsm8k_preferences.jsonl + data/gsm8k_preferences_local8.jsonl | completed | 0.575 | contradicted | reject | True |  |
| train | r-5720f98a | exp-31 | Qwen/Qwen3-1.7B-Base | 1303 | other | exp-17 |  | completed | 0.5845337376800607 | inconclusive | adopt | True |  |
| train | r-5720f98a | exp-32 | Qwen/Qwen3-1.7B-Base | 1363 | other | exp-17 |  | completed | 0.5921152388172858 | supported | adopt | True | 0.5724033358605004 |
| train | r-59151c09 | exp-01 | Qwen/Qwen3-4B-Base | 63 | sft | base_model | HF: openai/gsm8k (main, train),HF: meta-math/MetaMathQA (train) | killed | 0.08 | inconclusive | adopt | True |  |
| train | r-59151c09 | exp-02 | Qwen/Qwen3-4B-Base | 97 | decode-config | exp-01 |  | completed | 0.74 | supported | adopt | True |  |
| train | r-59151c09 | exp-03 | Qwen/Qwen3-4B-Base | 107 | sft | exp-01 | HF: openai/gsm8k (main, train),HF: meta-math/MetaMathQA (train) | completed | 0.76 | inconclusive | adopt | True |  |
| train | r-59151c09 | exp-04 | Qwen/Qwen3-4B-Base | 132 | sft | exp-03 | HF: openai/gsm8k (main, train) | completed | 0.6666666666666666 | contradicted | reject | True |  |
| train | r-59151c09 | exp-05 | Qwen/Qwen3-4B-Base | 161 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-06 | Qwen/Qwen3-4B-Base | 183 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-07 | Qwen/Qwen3-4B-Base | 195 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-08 | Qwen/Qwen3-4B-Base | 219 | other | exp-03 |  | completed | 0.7 | inconclusive | adopt | False |  |
| train | r-59151c09 | exp-09 | Qwen/Qwen3-4B-Base | 230 | sft | base_model | HF: openai/gsm8k (main, train) | completed | 0.7666666666666667 | contradicted | reject | False |  |
| train | r-59151c09 | exp-10 | Qwen/Qwen3-4B-Base | 291 | decode-config | exp-08 |  | completed | 0.84 | supported | adopt | False | 0.8498862774829417 |
| train | r-5ca84abf | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 77 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | failed |  | inconclusive | iterate | True |  |
| train | r-5ca84abf | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 89 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | completed | 0.08 | inconclusive | reject | True |  |
| train | r-5ca84abf | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 178 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | completed | 0.6466666666666666 | supported | iterate | True |  |
| train | r-5ca84abf | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 221 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | completed | 0.5733333333333334 | contradicted | reject | True |  |
| train | r-5ca84abf | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 264 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-5ca84abf | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 277 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-5ca84abf | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 294 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | completed | 0.6266666666666667 | contradicted | reject | True |  |
| train | r-5ca84abf | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 336 | merge | exp-07 |  | completed | 0.5533333333333333 | inconclusive | reject | False |  |
| train | r-5ca84abf | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 351 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | completed | 0.5666666666666667 | contradicted | adopt | True | 0.5959059893858984 |
| train | r-5ca84abf | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 371 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-5ca84abf | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 384 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-01 | Qwen/Qwen3-1.7B-Base | 77 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (types containing GSM) | failed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-02 | Qwen/Qwen3-1.7B-Base | 80 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (types containing GSM) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-03 | Qwen/Qwen3-1.7B-Base | 92 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (types containing GSM) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-04 | Qwen/Qwen3-1.7B-Base | 111 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (types containing GSM) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-05 | Qwen/Qwen3-1.7B-Base | 141 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (types containing GSM) | completed | 0.4 | inconclusive | adopt | True |  |
| train | r-5cf89e69 | exp-06 | Qwen/Qwen3-1.7B-Base | 230 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5cf89e69 | exp-07 | Qwen/Qwen3-1.7B-Base | 248 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug) | completed | 0.6133333333333333 | inconclusive | adopt | True |  |
| train | r-5cf89e69 | exp-08 | Qwen/Qwen3-1.7B-Base | 259 | other | exp-05 |  | completed |  | inconclusive | reject | False |  |
| train | r-5cf89e69 | exp-09 | Qwen/Qwen3-1.7B-Base | 297 | sft | exp-05 | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug) | completed | 0.6066666666666667 | inconclusive | reject | True |  |
| train | r-5cf89e69 | exp-10 | Qwen/Qwen3-1.7B-Base | 302 | other | exp-07 |  | completed | 0.6133333333333333 | inconclusive | adopt | False | 0.5610310841546626 |
| train | r-5cf89e69 | exp-11 | Qwen/Qwen3-1.7B-Base | 371 | sft | base_model | openai/gsm8k (train) | completed | 0.5 | contradicted | reject | True |  |
| train | r-5cf89e69 | exp-12 | Qwen/Qwen3-1.7B-Base | 416 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5d3c708b | exp-01 | Qwen/Qwen3-4B-Base | 113 | sft | base_model | openai/gsm8k main:train | completed | 0.8 | inconclusive | reject | True |  |
| train | r-5d3c708b | exp-02 | Qwen/Qwen3-4B-Base | 287 | rft | base_model | openai/gsm8k main:train,synthetic:self (sampled from exp-01's checkpoint) | completed | 0.8066666666666666 | inconclusive | adopt | True |  |
| train | r-5d3c708b | exp-03 | Qwen/Qwen3-4B-Base | 382 | grpo | exp-02 | openai/gsm8k main:train | failed |  | inconclusive | abandon_line | True |  |
| train | r-5d3c708b | exp-04 | Qwen/Qwen3-4B-Base | 420 | grpo | exp-02 | openai/gsm8k main:train | completed | 0.92 | inconclusive | adopt | True |  |
| train | r-5d3c708b | exp-05 | Qwen/Qwen3-4B-Base | 474 | other | exp-04 |  | completed |  | inconclusive | reject | False |  |
| train | r-5d3c708b | exp-06 | Qwen/Qwen3-4B-Base | 497 | other | exp-04 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-5d3c708b | exp-07 | Qwen/Qwen3-4B-Base | 503 | other | exp-04 |  | completed | 0.892 | inconclusive | adopt | True |  |
| train | r-5d3c708b | exp-08 | Qwen/Qwen3-4B-Base | 546 | grpo | exp-07 | openai/gsm8k main:train | completed | 0.91 | supported | adopt | True |  |
| train | r-5d3c708b | exp-09 | Qwen/Qwen3-4B-Base | 614 | other | exp-08 |  | completed | 0.94 | inconclusive | reject | False |  |
| train | r-5d3c708b | exp-10 | Qwen/Qwen3-4B-Base | 624 | grpo | exp-08 | openai/gsm8k main:train | failed |  | inconclusive | abandon_line | True |  |
| train | r-5d3c708b | exp-11 | Qwen/Qwen3-4B-Base | 636 | grpo | exp-08 | openai/gsm8k main:train | completed | 0.914 | supported | adopt | True |  |
| train | r-5d3c708b | exp-12 | Qwen/Qwen3-4B-Base | 671 | other | exp-11 |  | completed | 0.94 | inconclusive | adopt | False |  |
| train | r-5d3c708b | exp-13 | Qwen/Qwen3-4B-Base | 681 | other | exp-12 |  | completed | 0.9467 | inconclusive | adopt | True | 0.9052312357846853 |
| train | r-5d426e36 | exp-01 | Qwen/Qwen3-4B-Base | 109 | sft | base_model | nvidia/OpenMathInstruct-2 | killed |  | inconclusive | abandon_line | False |  |
| train | r-5d426e36 | exp-02 | Qwen/Qwen3-4B-Base | 126 | sft | base_model | nvidia/OpenMathInstruct-2 | killed |  | inconclusive | adopt | False |  |
| train | r-5d426e36 | exp-03 | Qwen/Qwen3-4B-Base | 153 | merge | exp-02 |  | completed | 0.38 | inconclusive | adopt | False | 0.45716451857467777 |
| train | r-5d426e36 | exp-04 | Qwen/Qwen3-4B-Base | 184 | merge | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-5d426e36 | exp-05 | Qwen/Qwen3-4B-Base | 233 | merge | exp-02 |  | failed |  | inconclusive | abandon_line | False |  |
| train | r-5d426e36 | exp-06 | Qwen/Qwen3-4B-Base | 256 | merge | exp-02 |  | failed |  | inconclusive | abandon_line | False |  |
| train | r-5d426e36 | exp-07 | Qwen/Qwen3-4B-Base | 302 | other | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-5d426e36 | exp-08 | Qwen/Qwen3-4B-Base | 304 | other | exp-02 |  | failed |  | inconclusive | abandon_line | False |  |
| train | r-5dcadd31 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 206 | sft | base_model | local (tokenized from HF nvidia/OpenMathInstruct-2 train_1M + openai/gsm8k main train) | killed | 0.948 | supported | adopt | False |  |
| train | r-5dcadd31 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 472 | decode-config | exp-01 |  | completed | 0.873 | supported | adopt | False | 0.846095526914329 |
| train | r-5dcadd31 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 498 | grpo | exp-01 | openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-5dcadd31 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 518 | grpo | exp-01 | openai/gsm8k (main, train split) | completed | 0.944 | contradicted | reject | True |  |
| train | r-5dcadd31 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 642 | other | exp-01 |  | completed |  | inconclusive | reject | True |  |
| train | r-5dcadd31 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 702 | other | exp-01 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-5dcadd31 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 746 | sft | exp-01 | local (built from openai/gsm8k main train + HF nvidia/OpenMathInstruct-2 train_1M) | completed |  | inconclusive | reject | False |  |
| train | r-5dcadd31 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 782 | sft | exp-01 | derived:exp-07 | completed |  | inconclusive | reject | False |  |
| train | r-5f4b22de | exp-01 | Qwen/Qwen3-4B-Base | 132 | sft | base_model | openai/gsm8k (main, train) | completed | 0.7867 | inconclusive | iterate | True |  |
| train | r-5f4b22de | exp-02 | Qwen/Qwen3-4B-Base | 359 | rft | base_model | openai/gsm8k (main, train),synthetic:self (sampled from exp-01's checkpoint sft_v1) | completed | 0.84 | supported | adopt | True |  |
| train | r-5f4b22de | exp-03 | Qwen/Qwen3-4B-Base | 465 | grpo | exp-02 | openai/gsm8k (main, train) | completed | 0.8667 | supported | adopt | True |  |
| train | r-5f4b22de | exp-04 | Qwen/Qwen3-4B-Base | 528 | grpo | exp-03 | openai/gsm8k (main, train) | completed | 0.8533 | contradicted | reject | True |  |
| train | r-5f4b22de | exp-05 | Qwen/Qwen3-4B-Base | 582 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-5f4b22de | exp-06 | Qwen/Qwen3-4B-Base | 582 | sft | base_model | openai/gsm8k (main, train),synthetic:self (sampled from exp-01's checkpoint sft_v1),meta-math/MetaMathQA | completed | 0.8467 | inconclusive | adopt | True |  |
| train | r-5f4b22de | exp-07 | Qwen/Qwen3-4B-Base | 633 | grpo | exp-06 | openai/gsm8k (main, train) | completed | 0.9267 | supported | adopt | True |  |
| train | r-5f4b22de | exp-08 | Qwen/Qwen3-4B-Base | 688 | other | exp-07 |  | completed |  | inconclusive | adopt | False | 0.8923426838514026 |
| train | r-60904922 | exp-01 | Qwen/Qwen3-4B-Base | 176 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-02 | Qwen/Qwen3-4B-Base | 201 | merge | exp-01 |  | completed | 0.16 | inconclusive | reject | False |  |
| train | r-60904922 | exp-03 | Qwen/Qwen3-4B-Base | 226 | merge | exp-01 |  | completed | 0.8 | supported | reject | True |  |
| train | r-60904922 | exp-04 | Qwen/Qwen3-4B-Base | 253 | sft | base_model | HF openai/gsm8k (config main, split train), re-rendered in the benchmark's 10-shot prompt format | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-05 | Qwen/Qwen3-4B-Base | 323 | merge | exp-04 |  | completed | 0.35 | contradicted | reject | True |  |
| train | r-60904922 | exp-06 | Qwen/Qwen3-4B-Base | 393 | rft | base_model | synthetic:self | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-07 | Qwen/Qwen3-4B-Base | 414 | merge | exp-06 |  | completed | 0.51 | contradicted | adopt | True |  |
| train | r-60904922 | exp-08 | Qwen/Qwen3-4B-Base | 443 | other | exp-07 |  | completed |  | inconclusive | adopt | False | 0.44124336618650495 |
| train | r-628a7a20 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 20368 | sft | base_model | local | killed |  | inconclusive | abandon_line | False |  |
| train | r-628a7a20 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 25061 | sft | base_model | local | completed | 0.6733333333333333 | inconclusive | adopt | True |  |
| train | r-628a7a20 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 29588 | rft | exp-02 | synthetic:self,derived:exp-02 | failed |  | inconclusive | abandon_line | True |  |
| train | r-628a7a20 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 30851 | rft | exp-02 | synthetic:self,derived:exp-02 | completed | 0.74 | supported | adopt | True |  |
| train | r-628a7a20 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 31745 | other | exp-04 |  | completed | 0.7035633055344959 | inconclusive | adopt | False | 0.7020470053070508 |
| train | r-628a7a20 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 33171 | rft | exp-05 | synthetic:self,derived:exp-05 | completed | 0.7266666666666667 | inconclusive | reject | False |  |
| train | r-635b683e | exp-01 | Qwen/Qwen3-4B-Base | 447 | sft | base_model | HF nvidia/OpenMathInstruct-2 (local parquet snapshot) | killed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-02 | Qwen/Qwen3-4B-Base | 591 | sft | base_model | HF nvidia/OpenMathInstruct-2 (local parquet snapshot) | completed | 0.93 | supported | adopt | False |  |
| train | r-635b683e | exp-03 | Qwen/Qwen3-4B-Base | 747 | decode-config | exp-02 |  | completed | 0.855 | supported | reject | True |  |
| train | r-635b683e | exp-04 | Qwen/Qwen3-4B-Base | 810 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-635b683e | exp-05 | Qwen/Qwen3-4B-Base | 847 | grpo | exp-02 | derived:exp-04 | failed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-06 | Qwen/Qwen3-4B-Base | 863 | grpo | exp-02 | derived:exp-04 | failed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-07 | Qwen/Qwen3-4B-Base | 888 | grpo | exp-02 | derived:exp-04 | killed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-08 | Qwen/Qwen3-4B-Base | 894 | grpo | exp-02 | derived:exp-04 | failed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-09 | Qwen/Qwen3-4B-Base | 978 | grpo | exp-02 | derived:exp-04 | failed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-10 | Qwen/Qwen3-4B-Base | 1012 | grpo | exp-02 | derived:exp-04 | completed | 0.937 | inconclusive | adopt | False |  |
| train | r-635b683e | exp-11 | Qwen/Qwen3-4B-Base | 1168 | grpo | exp-10 | derived:exp-04 | failed |  | inconclusive | abandon_line | False |  |
| train | r-635b683e | exp-12 | Qwen/Qwen3-4B-Base | 1236 | grpo | exp-10 | derived:exp-04 | failed | 0.93 | inconclusive | adopt | False |  |
| train | r-635b683e | exp-13 | Qwen/Qwen3-4B-Base | 1308 | merge | exp-02 |  | completed | 0.93 | inconclusive | adopt | False |  |
| train | r-635b683e | exp-14 | Qwen/Qwen3-4B-Base | 1373 | other | exp-13 |  | completed | 0.927 | inconclusive | adopt | False | 0.9120545868081881 |
| train | r-655a20a6 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 125 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed | 0.46 | inconclusive | adopt | False | 0.4086429112964367 |
| train | r-655a20a6 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-65be88fb | exp-01 | Qwen/Qwen3-1.7B-Base | 68 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.08 | inconclusive | reject | True |  |
| train | r-65be88fb | exp-02 | Qwen/Qwen3-1.7B-Base | 107 | sft | base_model | HF openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | False |  |
| train | r-65be88fb | exp-03 | Qwen/Qwen3-1.7B-Base | 119 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.06 | inconclusive | reject | False |  |
| train | r-65be88fb | exp-04 | Qwen/Qwen3-1.7B-Base | 180 | decode-config | exp-03 |  | completed | 0.34 | supported | adopt | False |  |
| train | r-65be88fb | exp-05 | Qwen/Qwen3-1.7B-Base | 209 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.6 | supported | adopt | False |  |
| train | r-65be88fb | exp-06 | Qwen/Qwen3-1.7B-Base | 242 | sft | exp-05 | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (split train, rows whose type starts with GSM) | completed | 0.76 | supported | adopt | True |  |
| train | r-65be88fb | exp-07 | Qwen/Qwen3-1.7B-Base | 289 | grpo | exp-06 | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-65be88fb | exp-08 | Qwen/Qwen3-1.7B-Base | 317 | sft | exp-06 | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (split train, rows whose type starts with GSM) | completed | 0.72 | contradicted | reject | True |  |
| train | r-65be88fb | exp-09 | Qwen/Qwen3-1.7B-Base | 369 | rft | exp-06 | synthetic:self (rejection samples from the exp-06 checkpoint) + HF openai/gsm8k (config main, split train) | completed | 0.6333333333333333 | contradicted | reject | True |  |
| train | r-65be88fb | exp-10 | Qwen/Qwen3-1.7B-Base | 399 | grpo | exp-06 | HF openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-65be88fb | exp-11 | Qwen/Qwen3-1.7B-Base | 402 | grpo | exp-06 | HF openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-65be88fb | exp-12 | Qwen/Qwen3-1.7B-Base | 416 | grpo | exp-06 | HF openai/gsm8k (config main, split train) | completed | 0.7066666666666667 | contradicted | reject | True |  |
| train | r-65be88fb | exp-13 | Qwen/Qwen3-1.7B-Base | 441 | sft | exp-06 | HF openai/gsm8k (config main, split train) | completed | 0.64 | contradicted | reject | True |  |
| train | r-65be88fb | exp-14 | Qwen/Qwen3-1.7B-Base | 471 | grpo | exp-06 | HF openai/gsm8k (config main, split train) | completed | 0.72 | contradicted | reject | True |  |
| train | r-65be88fb | exp-15 | Qwen/Qwen3-1.7B-Base | 487 | sft | exp-06 | HF openai/gsm8k (config main, split train) | completed | 0.7133333333333334 | contradicted | reject | True |  |
| train | r-65be88fb | exp-16 | Qwen/Qwen3-1.7B-Base | 503 | other | exp-06 |  | completed | 0.7466666666666667 | inconclusive | adopt | True | 0.733131159969674 |
| train | r-6920c788 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 144 | sft | base_model | HF openai/gsm8k main, train split | completed | 0.54 | supported | adopt | True |  |
| train | r-6920c788 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 248 | sft | base_model | HF openai/gsm8k main train + HF meta-math/MetaMathQA train | completed | 0.47 | contradicted | reject | True |  |
| train | r-6920c788 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 636 | other | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| train | r-6920c788 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 679 | rft | base_model | HF openai/gsm8k main train + synthetic:self (samples from exp-01's checkpoint) | completed | 0.64 | supported | adopt | True |  |
| train | r-6920c788 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 814 | other | exp-04 |  | completed |  | inconclusive | adopt | False |  |
| train | r-6920c788 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 858 | rft | base_model | HF openai/gsm8k main train + synthetic:self (round-1 samples from exp-01, round-2 samples from exp-04) | completed | 0.74 | supported | adopt | True |  |
| train | r-6920c788 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 964 | other | exp-06 |  | completed | 0.647 | inconclusive | adopt | False |  |
| train | r-6920c788 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1006 | rft | base_model | HF openai/gsm8k main train + synthetic:self (rounds 1-3, sampled from exp-01, exp-04 and exp-06 checkpoints) | completed | 0.6933 | inconclusive | adopt | True |  |
| train | r-6920c788 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 1100 | other | exp-08 |  | completed | 0.6275 | contradicted | adopt | True |  |
| train | r-6920c788 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 1114 | other | exp-06 |  | completed | 0.6171 | contradicted | adopt | True |  |
| train | r-6920c788 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 1156 | other | exp-08 |  | completed | 0.6533 | inconclusive | adopt | True |  |
| train | r-6920c788 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 1169 | decode-config | exp-11 |  | completed | 0.7333 | supported | adopt | True |  |
| train | r-6920c788 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 1192 | decode-config | exp-06 |  | completed | 0.696 | contradicted | adopt | True |  |
| train | r-6920c788 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 1207 | other | exp-06 |  | completed | 0.7067 | inconclusive | adopt | False | 0.7005307050796058 |
| train | r-6920c788 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 1278 | rft | base_model | HF openai/gsm8k main train + synthetic:self (rounds 1-4, sampled from exp-01, exp-04, exp-06 and exp-06 again) | completed | 0.6914 | inconclusive | reject | True |  |
| train | r-6978b5cf | exp-01 | Qwen/Qwen3-4B-Base | 54 | sft | base_model | HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-6978b5cf | exp-02 | Qwen/Qwen3-4B-Base | 72 | sft | base_model | HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-03 | Qwen/Qwen3-4B-Base | 80 | merge | exp-02 |  | completed | 0.19 | contradicted | reject | False |  |
| train | r-6978b5cf | exp-04 | Qwen/Qwen3-4B-Base | 96 | sft | base_model | HF:openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-05 | Qwen/Qwen3-4B-Base | 98 | merge | exp-04 |  | completed | 0.17 | contradicted | reject | False |  |
| train | r-6978b5cf | exp-06 | Qwen/Qwen3-4B-Base | 120 | sft | base_model | HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-07 | Qwen/Qwen3-4B-Base | 128 | merge | exp-06 |  | completed | 0.2 | contradicted | reject | False |  |
| train | r-6978b5cf | exp-08 | Qwen/Qwen3-4B-Base | 140 | sft | base_model | HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-6978b5cf | exp-09 | Qwen/Qwen3-4B-Base | 158 | sft | base_model | HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-6978b5cf | exp-10 | Qwen/Qwen3-4B-Base | 170 | sft | base_model | HF:openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-11 | Qwen/Qwen3-4B-Base | 172 | merge | exp-10 |  | completed | 0.38666666666666666 | inconclusive | reject | False |  |
| train | r-6978b5cf | exp-12 | Qwen/Qwen3-4B-Base | 196 | sft | base_model | HF:openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-13 | Qwen/Qwen3-4B-Base | 198 | merge | exp-12 |  | completed | 0.14 | inconclusive | reject | False |  |
| train | r-6978b5cf | exp-14 | Qwen/Qwen3-4B-Base | 214 | sft | base_model | HF:openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | False |  |
| train | r-6978b5cf | exp-15 | Qwen/Qwen3-4B-Base | 216 | merge | exp-14 |  | completed | 0.42 | inconclusive | adopt | False | 0.4844579226686884 |
| train | r-6978b5cf | exp-16 | Qwen/Qwen3-4B-Base | 232 | sft | base_model | HF:meta-math/MetaMathQA (split=train),HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-6978b5cf | exp-17 | Qwen/Qwen3-4B-Base | 262 | sft | base_model | HF:meta-math/MetaMathQA (split=train),HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-6978b5cf | exp-18 | Qwen/Qwen3-4B-Base | 285 | sft | base_model | HF:meta-math/MetaMathQA (split=train),HF:openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-6c4c222b | exp-01 | Qwen/Qwen3-1.7B-Base | 189 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) + openai/gsm8k train | failed |  | inconclusive | abandon_line | True |  |
| train | r-6c4c222b | exp-02 | Qwen/Qwen3-1.7B-Base | 200 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) + openai/gsm8k train | killed |  | inconclusive | abandon_line | True |  |
| train | r-6c4c222b | exp-03 | Qwen/Qwen3-1.7B-Base | 270 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) + openai/gsm8k train | completed | 0.22 | inconclusive | reject | False |  |
| train | r-6c4c222b | exp-04 | Qwen/Qwen3-1.7B-Base | 476 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) targets + openai/gsm8k train demos | killed |  | inconclusive | abandon_line | True |  |
| train | r-6c4c222b | exp-05 | Qwen/Qwen3-1.7B-Base | 510 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) targets + openai/gsm8k train demos | completed | 0.695 | supported | adopt | True |  |
| train | r-6c4c222b | exp-06 | Qwen/Qwen3-1.7B-Base | 553 | sft | base_model | derived: meta-math/MetaMathQA (GSM subsets) targets + openai/gsm8k train demos | completed | 0.67 | inconclusive | adopt | True |  |
| train | r-6c4c222b | exp-07 | Qwen/Qwen3-1.7B-Base | 641 | grpo | exp-06 | openai/gsm8k (train split), built in-process by train_grpo.py | completed | 0.782 | supported | adopt | True |  |
| train | r-6c4c222b | exp-08 | Qwen/Qwen3-1.7B-Base | 675 | grpo | exp-07 | openai/gsm8k (train split), built in-process by train_grpo.py | completed | 0.775 | inconclusive | reject | True |  |
| train | r-6c4c222b | exp-09 | Qwen/Qwen3-1.7B-Base | 703 | grpo | exp-07 | openai/gsm8k (train split), built in-process by train_grpo.py | killed |  | inconclusive | abandon_line | True |  |
| train | r-6c4c222b | exp-10 | Qwen/Qwen3-1.7B-Base | 726 | grpo | exp-07 | openai/gsm8k (train split), built in-process by train_grpo.py | completed | 0.7933333333333333 | inconclusive | adopt | True |  |
| train | r-6c4c222b | exp-11 | Qwen/Qwen3-1.7B-Base | 783 | grpo | exp-10 | openai/gsm8k (train split) + microsoft/orca-math-word-problems-200k, built in-process by train_grpo.py | completed | 0.808 | supported | adopt | True |  |
| train | r-6c4c222b | exp-12 | Qwen/Qwen3-1.7B-Base | 841 | grpo | exp-11 | openai/gsm8k (train split) + microsoft/orca-math-word-problems-200k, built in-process by train_grpo.py | completed | 0.828 | supported | adopt | True |  |
| train | r-6c4c222b | exp-13 | Qwen/Qwen3-1.7B-Base | 865 | grpo | exp-12 | openai/gsm8k (train split) + microsoft/orca-math-word-problems-200k, built in-process by train_grpo.py | completed | 0.812 | contradicted | reject | True |  |
| train | r-6c4c222b | exp-14 | Qwen/Qwen3-1.7B-Base | 927 | decode-config | exp-12 |  | completed | 0.84 | inconclusive | adopt | True | 0.8180439727065959 |
| train | r-6dabe99e | exp-01 | Qwen/Qwen3-1.7B-Base | 70 | sft | base_model | HF: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_* rows) | completed | 0.08 | inconclusive | adopt | True |  |
| train | r-6dabe99e | exp-02 | Qwen/Qwen3-1.7B-Base | 231 | decode-config | exp-01 |  | completed | 0.0 | inconclusive | reject | True |  |
| train | r-6dabe99e | exp-03 | Qwen/Qwen3-1.7B-Base | 253 | decode-config | exp-01 |  | completed | 0.08 | supported | reject | True |  |
| train | r-6dabe99e | exp-04 | Qwen/Qwen3-1.7B-Base | 262 | sft | exp-01 | HF: openai/gsm8k (train) | completed | 0.46 | supported | adopt | True |  |
| train | r-6dabe99e | exp-05 | Qwen/Qwen3-1.7B-Base | 302 | decode-config | exp-04 |  | completed | 0.4 | supported | adopt | True |  |
| train | r-6dabe99e | exp-06 | Qwen/Qwen3-1.7B-Base | 314 | sft | base_model | HF: openai/gsm8k (train) | completed | 0.02 | contradicted | reject | True |  |
| train | r-6dabe99e | exp-07 | Qwen/Qwen3-1.7B-Base | 334 | decode-config | exp-06 |  | completed | 0.02 | contradicted | reject | True |  |
| train | r-6dabe99e | exp-08 | Qwen/Qwen3-1.7B-Base | 356 | sft | base_model | HF: openai/gsm8k (train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-6dabe99e | exp-09 | Qwen/Qwen3-1.7B-Base | 391 | decode-config | exp-04 |  | completed |  | inconclusive | adopt | True |  |
| train | r-6dabe99e | exp-10 | Qwen/Qwen3-1.7B-Base | 425 | sft | base_model | HF: openai/gsm8k (train) + meta-math/MetaMathQA (GSM-derived rows) | completed |  | inconclusive | adopt | True |  |
| train | r-6dabe99e | exp-11 | Qwen/Qwen3-1.7B-Base | 535 | sft | exp-10 | HF: openai/gsm8k (train) | completed | 0.42 | contradicted | reject | True |  |
| train | r-6dabe99e | exp-12 | Qwen/Qwen3-1.7B-Base | 561 | decode-config | exp-11 |  | completed | 0.42 | contradicted | reject | True |  |
| train | r-6dabe99e | exp-13 | Qwen/Qwen3-1.7B-Base | 579 | sft | exp-10 | HF: openai/gsm8k (train) | completed | 0.6 | inconclusive | adopt | True |  |
| train | r-6dabe99e | exp-14 | Qwen/Qwen3-1.7B-Base | 629 | decode-config | exp-13 |  | completed | 0.48 | inconclusive | adopt | True | 0.4927975739196361 |
| train | r-736cc5f9 | exp-01 | Qwen/Qwen3-1.7B-Base | 110 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-736cc5f9 | exp-02 | Qwen/Qwen3-1.7B-Base | 170 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6 | inconclusive | reject | True |  |
| train | r-736cc5f9 | exp-03 | Qwen/Qwen3-1.7B-Base | 335 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.66 | supported | adopt | True |  |
| train | r-736cc5f9 | exp-04 | Qwen/Qwen3-1.7B-Base | 401 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-736cc5f9 | exp-05 | Qwen/Qwen3-1.7B-Base | 427 | other | exp-03 |  | completed | 0.63 | inconclusive | adopt | False | 0.6277482941622441 |
| train | r-75836ae8 | exp-01 | Qwen/Qwen3-1.7B-Base | 124 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.01 | inconclusive | reject | True |  |
| train | r-75836ae8 | exp-02 | Qwen/Qwen3-1.7B-Base | 194 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.1 | contradicted | adopt | True |  |
| train | r-75836ae8 | exp-03 | Qwen/Qwen3-1.7B-Base | 348 | sft | exp-02 | openai/gsm8k (main, train split) | completed | 0.5 | supported | adopt | True |  |
| train | r-75836ae8 | exp-04 | Qwen/Qwen3-1.7B-Base | 463 | sft | exp-03 | openai/gsm8k (main, train split) | completed |  | inconclusive | abandon_line | True |  |
| train | r-75836ae8 | exp-05 | Qwen/Qwen3-1.7B-Base | 597 | other | exp-03 |  | completed | 0.53 | supported | adopt | True | 0.4579226686884003 |
| train | r-77b138a8 | exp-01 | Qwen/Qwen3-4B-Base | 82 | sft | base_model | openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-02 | Qwen/Qwen3-4B-Base | 102 | sft | base_model | openai/gsm8k (main, train) | killed |  | inconclusive | adopt | True |  |
| train | r-77b138a8 | exp-03 | Qwen/Qwen3-4B-Base | 157 | merge | exp-02 |  | completed | 0.006666666666666667 | inconclusive | reject | False |  |
| train | r-77b138a8 | exp-04 | Qwen/Qwen3-4B-Base | 185 | decode-config | exp-03 |  | completed | 0.02666666666666667 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-05 | Qwen/Qwen3-4B-Base | 190 | decode-config | exp-03 |  | completed | 0.0 | inconclusive | reject | False |  |
| train | r-77b138a8 | exp-06 | Qwen/Qwen3-4B-Base | 200 | sft | base_model | openai/gsm8k (main, train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-07 | Qwen/Qwen3-4B-Base | 206 | sft | base_model | openai/gsm8k (main, train) | completed | 0.8133333333333334 | supported | reject | True |  |
| train | r-77b138a8 | exp-08 | Qwen/Qwen3-4B-Base | 225 | sft | base_model | openai/gsm8k (main, train) | completed | 0.6866666666666666 | supported | adopt | True |  |
| train | r-77b138a8 | exp-09 | Qwen/Qwen3-4B-Base | 280 | sft | base_model | openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-10 | Qwen/Qwen3-4B-Base | 328 | sft | base_model | meta-math/MetaMathQA (GSM subset) | killed |  | inconclusive | adopt | True |  |
| train | r-77b138a8 | exp-11 | Qwen/Qwen3-4B-Base | 349 | merge | exp-10 |  | completed | 0.7 | contradicted | adopt | False |  |
| train | r-77b138a8 | exp-12 | Qwen/Qwen3-4B-Base | 357 | sft | exp-11 | openai/gsm8k (main, train) | completed | 0.6333333333333333 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-13 | Qwen/Qwen3-4B-Base | 372 | merge | exp-12 |  | completed | 0.7933333333333333 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-14 | Qwen/Qwen3-4B-Base | 377 | sft | exp-11 | openai/gsm8k (main, train) | completed | 0.8266666666666667 | contradicted | reject | False |  |
| train | r-77b138a8 | exp-15 | Qwen/Qwen3-4B-Base | 434 | sft | exp-11 | openai/gsm8k (main, train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-16 | Qwen/Qwen3-4B-Base | 483 | merge | exp-15 |  | completed | 0.3466666666666667 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-17 | Qwen/Qwen3-4B-Base | 484 | merge | exp-15 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-77b138a8 | exp-18 | Qwen/Qwen3-4B-Base | 503 | sft | exp-11 | openai/gsm8k (main, train) | completed | 0.82 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-19 | Qwen/Qwen3-4B-Base | 513 | merge | exp-18 |  | completed | 0.6466666666666666 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-20 | Qwen/Qwen3-4B-Base | 563 | sft | exp-08 | openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-21 | Qwen/Qwen3-4B-Base | 611 | sft | base_model | clarkkitchen22/SynthGSM8K-50K | failed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-22 | Qwen/Qwen3-4B-Base | 623 | sft | base_model | clarkkitchen22/SynthGSM8K-50K | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-23 | Qwen/Qwen3-4B-Base | 643 | sft | base_model | openai/gsm8k (main, train),clarkkitchen22/SynthGSM8K-50K | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-24 | Qwen/Qwen3-4B-Base | 660 | sft | exp-11 | openai/gsm8k (main, train) | completed | 0.36 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-25 | Qwen/Qwen3-4B-Base | 674 | sft | base_model | openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-77b138a8 | exp-26 | Qwen/Qwen3-4B-Base | 689 | other | exp-08 |  | completed | 0.64 | contradicted | reject | True |  |
| train | r-77b138a8 | exp-27 | Qwen/Qwen3-4B-Base | 696 | other | exp-08 |  | completed | 0.68 | inconclusive | adopt | True |  |
| train | r-77b138a8 | exp-28 | Qwen/Qwen3-4B-Base | 729 | decode-config | exp-27 |  | completed | 0.8333333333333334 | supported | adopt | True | 0.7968157695223654 |
| train | r-7842f260 | exp-01 | Qwen/Qwen3-1.7B-Base | 12086 | sft | base_model | HF openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-7842f260 | exp-02 | Qwen/Qwen3-1.7B-Base | 12539 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.66 | inconclusive | adopt | False |  |
| train | r-7842f260 | exp-03 | Qwen/Qwen3-1.7B-Base | 19507 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-02's checkpoint),HF openai/gsm8k (config main, split train) | completed | 0.843 | inconclusive | adopt | False |  |
| train | r-7842f260 | exp-04 | Qwen/Qwen3-1.7B-Base | 22143 | merge | exp-02 |  | killed |  | inconclusive | iterate | False |  |
| train | r-7842f260 | exp-05 | Qwen/Qwen3-1.7B-Base | 22789 | merge | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-7842f260 | exp-06 | Qwen/Qwen3-1.7B-Base | 23003 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-03's checkpoint) | completed | 0.793 | contradicted | reject | False |  |
| train | r-7842f260 | exp-07 | Qwen/Qwen3-1.7B-Base | 24174 | merge | exp-03 |  | completed | 0.833 | inconclusive | reject | False |  |
| train | r-7842f260 | exp-08 | Qwen/Qwen3-1.7B-Base | 24999 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-03's checkpoint),HF openai/gsm8k (config main, split train) | completed | 0.784 | inconclusive | reject | False |  |
| train | r-7842f260 | exp-09 | Qwen/Qwen3-1.7B-Base | 25224 | other | exp-03 |  | completed | 0.82 | inconclusive | adopt | False | 0.7816527672479151 |
| train | r-7842f260 | exp-10 | Qwen/Qwen3-1.7B-Base | 26690 | merge | exp-03 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-78f13a5c | exp-01 | Qwen/Qwen3-1.7B-Base | 94 | sft | base_model | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | killed |  | inconclusive | abandon_line | True |  |
| train | r-78f13a5c | exp-02 | Qwen/Qwen3-1.7B-Base | 119 | sft | base_model | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | completed | 0.06 | inconclusive | adopt | False |  |
| train | r-78f13a5c | exp-03 | Qwen/Qwen3-1.7B-Base | 177 | decode-config | exp-02 |  | completed | 0.06666666666666667 | inconclusive | reject | False |  |
| train | r-78f13a5c | exp-04 | Qwen/Qwen3-1.7B-Base | 198 | sft | base_model | HF openai/gsm8k (main, train) | completed | 0.25 | inconclusive | adopt | True |  |
| train | r-78f13a5c | exp-05 | Qwen/Qwen3-1.7B-Base | 260 | sft | exp-04 | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | killed |  | inconclusive | abandon_line | False |  |
| train | r-78f13a5c | exp-06 | Qwen/Qwen3-1.7B-Base | 295 | sft | exp-04 | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | killed |  | inconclusive | abandon_line | True |  |
| train | r-78f13a5c | exp-07 | Qwen/Qwen3-1.7B-Base | 321 | sft | exp-04 | HF openai/gsm8k (main, train) | completed | 0.54 | inconclusive | adopt | False |  |
| train | r-78f13a5c | exp-08 | Qwen/Qwen3-1.7B-Base | 355 | sft | exp-07 | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | completed | 0.67 | inconclusive | adopt | True |  |
| train | r-78f13a5c | exp-09 | Qwen/Qwen3-1.7B-Base | 411 | sft | exp-08 | HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM subset) | completed | 0.68 | inconclusive | adopt | False |  |
| train | r-78f13a5c | exp-10 | Qwen/Qwen3-1.7B-Base | 451 | decode-config | exp-09 |  | completed |  | inconclusive | adopt | False | 0.66868840030326 |
| train | r-792b20f6 | exp-01 | Qwen/Qwen3-4B-Base | 73 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-02 | Qwen/Qwen3-4B-Base | 93 | other | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-03 | Qwen/Qwen3-4B-Base | 103 | merge | exp-02 |  | failed | 0.2 | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-04 | Qwen/Qwen3-4B-Base | 171 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-792b20f6 | exp-05 | Qwen/Qwen3-4B-Base | 191 | other | exp-03 |  | completed |  | inconclusive | adopt | False | 0.19484457922668688 |
| train | r-792b20f6 | exp-06 | Qwen/Qwen3-4B-Base | 209 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-7a94150b | exp-01 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | failed |  | inconclusive | iterate | True |  |
| train | r-7a94150b | exp-02 | Qwen/Qwen3-4B-Base | 101 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.175 | inconclusive | adopt | True |  |
| train | r-7a94150b | exp-03 | Qwen/Qwen3-4B-Base | 131 | decode-config | exp-02 |  | completed | 0.15 | contradicted | reject | True |  |
| train | r-7a94150b | exp-04 | Qwen/Qwen3-4B-Base | 159 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.25 | contradicted | reject | True |  |
| train | r-7a94150b | exp-05 | Qwen/Qwen3-4B-Base | 177 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.45 | contradicted | reject | True |  |
| train | r-7a94150b | exp-06 | Qwen/Qwen3-4B-Base | 204 | other | base_model |  | completed | 0.5 | supported | adopt | True | 0.43214556482183475 |
| train | r-7f29490c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 97 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | completed | 0.3 | inconclusive | adopt | True |  |
| train | r-7f29490c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 182 | decode-config | exp-01 |  | completed | 0.56 | inconclusive | adopt | False | 0.577710386656558 |
| train | r-87e033c4 | exp-01 | Qwen/Qwen3-4B-Base | 82 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.04 | inconclusive | adopt | False |  |
| train | r-87e033c4 | exp-02 | Qwen/Qwen3-4B-Base | 146 | decode-config | exp-01 |  | completed | 0.08 | contradicted | adopt | True |  |
| train | r-87e033c4 | exp-03 | Qwen/Qwen3-4B-Base | 169 | decode-config | exp-02 |  | completed | 0.06 | contradicted | adopt | False |  |
| train | r-87e033c4 | exp-04 | Qwen/Qwen3-4B-Base | 201 | decode-config | exp-03 |  | completed | 0.013333333333333334 | contradicted | reject | False |  |
| train | r-87e033c4 | exp-05 | Qwen/Qwen3-4B-Base | 224 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.23333333333333334 | supported | adopt | True |  |
| train | r-87e033c4 | exp-06 | Qwen/Qwen3-4B-Base | 266 | decode-config | exp-05 |  | completed | 0.8225928733889311 | supported | adopt | False |  |
| train | r-87e033c4 | exp-07 | Qwen/Qwen3-4B-Base | 294 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.8133333333333334 | contradicted | reject | True |  |
| train | r-87e033c4 | exp-08 | Qwen/Qwen3-4B-Base | 361 | other | exp-06 |  | completed | 0.8133333333333334 | inconclusive | adopt | False | 0.8180439727065959 |
| train | r-87e033c4 | exp-09 | Qwen/Qwen3-4B-Base | 378 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.3933333333333333 | inconclusive | reject | False |  |
| train | r-87e033c4 | exp-10 | Qwen/Qwen3-4B-Base | 408 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.78 | inconclusive | reject | False |  |
| train | r-88141936 | exp-01 | Qwen/Qwen3-4B-Base | 102 | sft | base_model | HF meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | False |  |
| train | r-88141936 | exp-02 | Qwen/Qwen3-4B-Base | 116 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.6 | inconclusive | adopt | False | 0.3684609552691433 |
| train | r-89603e49 | exp-01 | Qwen/Qwen3-1.7B-Base | 17875 | sft | base_model | local | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-02 | Qwen/Qwen3-1.7B-Base | 18544 | sft | base_model | local | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-03 | Qwen/Qwen3-1.7B-Base | 19154 | sft | base_model | local | completed | 0.727 | supported | adopt | True |  |
| train | r-89603e49 | exp-04 | Qwen/Qwen3-1.7B-Base | 28893 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-05 | Qwen/Qwen3-1.7B-Base | 29447 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-06 | Qwen/Qwen3-1.7B-Base | 29614 | sft | base_model | local,synthetic:self | killed | 0.8266666666666667 | supported | adopt | True |  |
| train | r-89603e49 | exp-07 | Qwen/Qwen3-1.7B-Base | 33782 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-08 | Qwen/Qwen3-1.7B-Base | 34040 | sft | exp-06 | local,synthetic:self | completed | 0.8333333333333334 | inconclusive | adopt | True |  |
| train | r-89603e49 | exp-09 | Qwen/Qwen3-1.7B-Base | 34254 | other | exp-06 |  | completed |  | inconclusive | reject | False |  |
| train | r-89603e49 | exp-10 | Qwen/Qwen3-1.7B-Base | 38078 | other | exp-08 |  | completed | 0.8333333333333334 | supported | adopt | False | 0.7748294162244125 |
| train | r-89ab4cc5 | exp-01 | Qwen/Qwen3-4B-Base | 88 | sft | base_model | GSM8K train (agent's words, [43], [62]); the loader call in the script text is truncated in the stream | killed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-02 | Qwen/Qwen3-4B-Base | 186 | merge | exp-01 |  | completed | 0.05 | inconclusive | reject | True |  |
| train | r-89ab4cc5 | exp-03 | Qwen/Qwen3-4B-Base | 187 | merge | exp-01 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-89ab4cc5 | exp-04 | Qwen/Qwen3-4B-Base | 249 | sft | base_model | GSM8K train (same in-script pipeline as exp-01); the loader call in the script text is truncated in the stream | completed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-05 | Qwen/Qwen3-4B-Base | 267 | merge | exp-04 |  | completed | 0.1 | inconclusive | reject | True |  |
| train | r-89ab4cc5 | exp-06 | Qwen/Qwen3-4B-Base | 289 | sft | base_model | GSM8K train (same in-script pipeline as exp-01); one step consumes a single batch | completed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-07 | Qwen/Qwen3-4B-Base | 295 | merge | exp-06 |  | completed | 0.5 | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-08 | Qwen/Qwen3-4B-Base | 323 | other | exp-07 |  | completed |  | inconclusive | adopt | True | 0.4230477634571645 |
| train | r-8c4cb1bc | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 159 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.6206 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 202 | merge | exp-01 |  | completed | 0.4 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 213 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA | killed | 0.5005455613136292 | inconclusive | abandon_line | True |  |
| train | r-8c4cb1bc | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 281 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.4112212657928467 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 292 | merge | exp-04 |  | completed | 0.087 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 317 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.08 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 411 | merge | exp-01 |  | completed | 0.14 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 514 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.18 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 752 | merge | exp-01 |  | completed | 0.127 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 793 | other | exp-02 |  | completed | 0.38666666666666666 | supported | adopt | True | 0.3843821076573162 |
| train | r-8ec271ed | exp-01 | Qwen/Qwen3-4B-Base | 94 | sft | base_model | HF openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-8ec271ed | exp-02 | Qwen/Qwen3-4B-Base | 121 | merge | exp-01 |  | completed |  | inconclusive | reject | True |  |
| train | r-8ec271ed | exp-03 | Qwen/Qwen3-4B-Base | 122 | merge | exp-01 |  | completed |  | inconclusive | reject | True |  |
| train | r-8ec271ed | exp-04 | Qwen/Qwen3-4B-Base | 150 | sft | base_model | HF openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| train | r-8ec271ed | exp-05 | Qwen/Qwen3-4B-Base | 154 | sft | base_model | HF openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-8ec271ed | exp-06 | Qwen/Qwen3-4B-Base | 166 | merge | exp-05 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-8ec271ed | exp-07 | Qwen/Qwen3-4B-Base | 167 | merge | exp-05 |  | completed | 0.025 | contradicted | reject | True |  |
| train | r-8ec271ed | exp-08 | Qwen/Qwen3-4B-Base | 219 | sft | base_model | HF openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-8ec271ed | exp-09 | Qwen/Qwen3-4B-Base | 222 | merge | exp-08 |  | completed | 0.4 | contradicted | reject | True |  |
| train | r-8ec271ed | exp-10 | Qwen/Qwen3-4B-Base | 235 | sft | base_model | HF openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-8ec271ed | exp-11 | Qwen/Qwen3-4B-Base | 237 | merge | exp-10 |  | completed | 0.475 | supported | adopt | True |  |
| train | r-8ec271ed | exp-12 | Qwen/Qwen3-4B-Base | 246 | other | exp-11 |  | completed |  | inconclusive | adopt | True | 0.43896891584533737 |
| train | r-8fbb0755 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 155 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA GSM_* subset | completed | 0.56 | supported | reject | True |  |
| train | r-8fbb0755 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 211 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA GSM_* subset | killed | 0.48 | inconclusive | reject | False |  |
| train | r-8fbb0755 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 323 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA GSM_* subset | completed | 0.44666666666666666 | contradicted | reject | True |  |
| train | r-8fbb0755 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 385 | sft | base_model | derived:local - 80000 rows of train_data.jsonl (brief system prompt) + 40000 rows of train_data_v3.jsonl (eval's 10-shot system prompt),openai/gsm8k train + meta-math/MetaMathQA GSM_* subset,openai/gsm8k train + meta-math/MetaMathQA GSM_* subset | completed | 0.6866666666666666 | supported | adopt | True |  |
| train | r-8fbb0755 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 455 | other | exp-04 |  | completed |  | inconclusive | reject | False |  |
| train | r-8fbb0755 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 503 | rft | exp-04 | derived:local - rft_data.jsonl + 15000 rows of train_data_v3.jsonl + 15000 rows of train_data.jsonl,synthetic:self - solutions generated by sft_v3 on openai/gsm8k train questions | completed | 0.7133333333333334 | supported | adopt | True |  |
| train | r-8fbb0755 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 521 | other | exp-06 |  | completed | 0.7366666666666667 | inconclusive | adopt | False | 0.6929492039423806 |
| train | r-8fbb0755 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 521 | grpo | exp-06 | openai/gsm8k train (prompts only, built in-process by grpo.py) | failed |  | inconclusive | abandon_line | True |  |
| train | r-8fbb0755 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 532 | grpo | exp-06 | openai/gsm8k train (prompts only, built in-process by grpo.py) | failed |  | inconclusive | abandon_line | True |  |
| train | r-8fbb0755 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 545 | grpo | exp-06 | openai/gsm8k train (prompts only, built in-process by grpo.py) | failed |  | inconclusive | abandon_line | True |  |
| train | r-8fbb0755 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 570 | rft | exp-06 | derived:local - rft_data.jsonl + rft_data2.jsonl + 12000 rows of train_data_v3.jsonl + 12000 rows of train_data.jsonl,synthetic:self - solutions generated by sft_v4 on openai/gsm8k train questions,synthetic:self - solutions generated by sft_v3 on openai/gsm8k train questions | completed | 0.7066666666666667 | contradicted | reject | True |  |
| train | r-9142b2d3 | exp-01 | Qwen/Qwen3-4B-Base | 52 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | failed |  | inconclusive | iterate | False |  |
| train | r-9142b2d3 | exp-02 | Qwen/Qwen3-4B-Base | 59 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | killed |  | inconclusive | iterate | False |  |
| train | r-9142b2d3 | exp-03 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | completed | 0.133 | inconclusive | reject | False |  |
| train | r-9142b2d3 | exp-04 | Qwen/Qwen3-4B-Base | 119 | sft | base_model | openai/gsm8k | completed | 0.7266666666666667 | supported | adopt | True |  |
| train | r-9142b2d3 | exp-05 | Qwen/Qwen3-4B-Base | 150 | decode-config | exp-04 |  | completed | 0.62 | inconclusive | reject | True |  |
| train | r-9142b2d3 | exp-06 | Qwen/Qwen3-4B-Base | 156 | decode-config | exp-04 |  | completed | 0.76 | inconclusive | adopt | False |  |
| train | r-9142b2d3 | exp-07 | Qwen/Qwen3-4B-Base | 168 | other | exp-04 |  | completed | 0.68 | inconclusive | reject | True |  |
| train | r-9142b2d3 | exp-08 | Qwen/Qwen3-4B-Base | 182 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k,meta-math/MetaMathQA | completed | 0.7533333333333333 | inconclusive | adopt | True | 0.7543593631539045 |
| train | r-9142b2d3 | exp-09 | Qwen/Qwen3-4B-Base | 200 | sft | exp-08 | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| train | r-918be760 | exp-01 | Qwen/Qwen3-4B-Base | 453 | sft | base_model | meta-math/MetaMathQA (GSM_AnsAug rows only) | killed | 0.78125 | contradicted | adopt | True |  |
| train | r-918be760 | exp-02 | Qwen/Qwen3-4B-Base | 520 | sft | exp-01 | meta-math/MetaMathQA (GSM_AnsAug rows only) | killed | 0.8266666666666667 | contradicted | reject | True |  |
| train | r-918be760 | exp-03 | Qwen/Qwen3-4B-Base | 571 | sft | base_model | openai/gsm8k (main, train split) | killed | 0.82 | contradicted | reject | True |  |
| train | r-918be760 | exp-04 | Qwen/Qwen3-4B-Base | 637 | grpo | base_model | meta-math/MetaMathQA (GSM_Rephrased rows only) | completed |  | inconclusive | reject | True |  |
| train | r-918be760 | exp-05 | Qwen/Qwen3-4B-Base | 658 | merge | base_model |  | completed | 0.8266666666666667 | contradicted | reject | True |  |
| train | r-918be760 | exp-06 | Qwen/Qwen3-4B-Base | 693 | grpo | base_model | meta-math/MetaMathQA (GSM_Rephrased rows only) | killed |  | inconclusive | reject | True |  |
| train | r-918be760 | exp-07 | Qwen/Qwen3-4B-Base | 727 | merge | base_model |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-918be760 | exp-08 | Qwen/Qwen3-4B-Base | 731 | merge | base_model |  | completed | 0.8666666666666667 | supported | adopt | True |  |
| train | r-918be760 | exp-09 | Qwen/Qwen3-4B-Base | 785 | merge | base_model |  | completed | 0.796875 | contradicted | reject | False |  |
| train | r-918be760 | exp-10 | Qwen/Qwen3-4B-Base | 794 | merge | base_model |  | completed | 0.8466666666666667 | contradicted | reject | True |  |
| train | r-918be760 | exp-11 | Qwen/Qwen3-4B-Base | 838 | rft | exp-08 | synthetic:self (sampled from runs/interp_ansaug_a075, exp-08) | completed | 0.86 | supported | reject | True |  |
| train | r-918be760 | exp-12 | Qwen/Qwen3-4B-Base | 878 | merge | exp-08 |  | completed | 0.86 | contradicted | reject | True |  |
| train | r-918be760 | exp-13 | Qwen/Qwen3-4B-Base | 884 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | completed | 0.8666666666666667 | supported | adopt | True |  |
| train | r-918be760 | exp-14 | Qwen/Qwen3-4B-Base | 919 | merge | exp-08 |  | completed | 0.8466666666666667 | contradicted | reject | False |  |
| train | r-918be760 | exp-15 | Qwen/Qwen3-4B-Base | 930 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | completed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-918be760 | exp-16 | Qwen/Qwen3-4B-Base | 932 | merge | base_model |  | completed | 0.8533333333333334 | contradicted | reject | False |  |
| train | r-918be760 | exp-17 | Qwen/Qwen3-4B-Base | 950 | sft | base_model | meta-math/MetaMathQA (GSM_Rephrased rows only) | killed | 0.84 | supported | reject | True |  |
| train | r-918be760 | exp-18 | Qwen/Qwen3-4B-Base | 973 | merge | base_model |  | completed | 0.82 | contradicted | reject | False |  |
| train | r-918be760 | exp-19 | Qwen/Qwen3-4B-Base | 979 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | killed | 0.86 | contradicted | reject | False |  |
| train | r-918be760 | exp-20 | Qwen/Qwen3-4B-Base | 996 | merge | exp-13 |  | completed | 0.8466666666666667 | contradicted | reject | False |  |
| train | r-918be760 | exp-21 | Qwen/Qwen3-4B-Base | 1034 | sft | exp-13 | microsoft/orca-math-word-problems-200k | failed |  | inconclusive | abandon_line | True |  |
| train | r-918be760 | exp-22 | Qwen/Qwen3-4B-Base | 1047 | sft | exp-13 | microsoft/orca-math-word-problems-200k | killed | 0.84 | contradicted | reject | True |  |
| train | r-918be760 | exp-23 | Qwen/Qwen3-4B-Base | 1077 | merge | exp-13 |  | completed | 0.8733333333333333 | contradicted | reject | True |  |
| train | r-918be760 | exp-24 | Qwen/Qwen3-4B-Base | 1104 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | killed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-918be760 | exp-25 | Qwen/Qwen3-4B-Base | 1115 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | killed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-918be760 | exp-26 | Qwen/Qwen3-4B-Base | 1130 | rft | exp-08 | derived: openai/gsm8k train rationales + synthetic:self (round-2 samples from runs/sft_balanced_rft_v6/checkpoint-25, exp-13) | killed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-918be760 | exp-27 | Qwen/Qwen3-4B-Base | 1143 | decode-config | exp-13 |  | completed | 0.86 | contradicted | reject | False |  |
| train | r-918be760 | exp-28 | Qwen/Qwen3-4B-Base | 1175 | dpo | exp-13 | synthetic:self (sampled from runs/sft_balanced_rft_v6/checkpoint-25, exp-13) | killed |  | inconclusive | reject | True |  |
| train | r-918be760 | exp-29 | Qwen/Qwen3-4B-Base | 1185 | merge | exp-13 |  | completed | 0.8266666666666667 | contradicted | reject | True |  |
| train | r-918be760 | exp-30 | Qwen/Qwen3-4B-Base | 1190 | merge | exp-13 |  | completed | 0.8533333333333334 | contradicted | reject | True |  |
| train | r-918be760 | exp-31 | Qwen/Qwen3-4B-Base | 1222 | distill | exp-08 | derived: openai/gsm8k train rationales + synthetic:Qwen/Qwen2.5-Math-7B-Instruct | killed | 0.8466666666666667 | contradicted | reject | True |  |
| train | r-918be760 | exp-32 | Qwen/Qwen3-4B-Base | 1232 | merge | exp-13 |  | completed | 0.86 | contradicted | reject | False |  |
| train | r-918be760 | exp-33 | Qwen/Qwen3-4B-Base | 1236 | merge | exp-13 |  | completed | 0.8392721758908264 | supported | reject | False |  |
| train | r-918be760 | exp-34 | Qwen/Qwen3-4B-Base | 1245 | merge | exp-13 |  | completed | 0.8385140257771039 | contradicted | reject | True |  |
| train | r-918be760 | exp-35 | Qwen/Qwen3-4B-Base | 1253 | merge | exp-13 |  | completed | 0.8415466262319939 | supported | adopt | True |  |
| train | r-918be760 | exp-36 | Qwen/Qwen3-4B-Base | 1270 | merge | exp-13 |  | completed | 0.8377558756633814 | contradicted | reject | False |  |
| train | r-918be760 | exp-37 | Qwen/Qwen3-4B-Base | 1280 | merge | exp-13 |  | completed | 0.8362395754359363 | contradicted | reject | True |  |
| train | r-918be760 | exp-38 | Qwen/Qwen3-4B-Base | 1284 | merge | exp-13 |  | completed | 0.8354814253222138 | contradicted | reject | True |  |
| train | r-918be760 | exp-39 | Qwen/Qwen3-4B-Base | 1289 | merge | exp-13 |  | completed | 0.8377558756633814 | contradicted | reject | False |  |
| train | r-918be760 | exp-40 | Qwen/Qwen3-4B-Base | 1291 | merge | exp-13 |  | completed | 0.8369977255496588 | contradicted | reject | False |  |
| train | r-918be760 | exp-41 | Qwen/Qwen3-4B-Base | 1295 | merge | exp-13 |  | completed | 0.8385140257771039 | contradicted | reject | False |  |
| train | r-918be760 | exp-42 | Qwen/Qwen3-4B-Base | 1314 | merge | exp-35 |  | completed | 0.8324488248673237 | contradicted | reject | False |  |
| train | r-918be760 | exp-43 | Qwen/Qwen3-4B-Base | 1336 | sft | exp-35 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | completed | 0.8400303260045489 | inconclusive | adopt | False |  |
| train | r-918be760 | exp-44 | Qwen/Qwen3-4B-Base | 1363 | merge | exp-35 |  | completed | 0.8423047763457164 | supported | adopt | True |  |
| train | r-918be760 | exp-45 | Qwen/Qwen3-4B-Base | 1375 | merge | exp-35 |  | completed | 0.8362395754359363 | contradicted | reject | False |  |
| train | r-918be760 | exp-46 | Qwen/Qwen3-4B-Base | 1380 | merge | exp-35 |  | completed | 0.8423047763457164 | inconclusive | reject | False |  |
| train | r-918be760 | exp-47 | Qwen/Qwen3-4B-Base | 1392 | merge | exp-35 |  | completed | 0.8377558756633814 | contradicted | reject | False |  |
| train | r-918be760 | exp-48 | Qwen/Qwen3-4B-Base | 1396 | merge | exp-35 |  | completed | 0.8354814253222138 | contradicted | reject | False |  |
| train | r-918be760 | exp-49 | Qwen/Qwen3-4B-Base | 1402 | sft | exp-35 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | completed |  | inconclusive | reject | False |  |
| train | r-918be760 | exp-50 | Qwen/Qwen3-4B-Base | 1407 | merge | exp-35 |  | completed | 0.8362395754359363 | contradicted | reject | False |  |
| train | r-918be760 | exp-51 | Qwen/Qwen3-4B-Base | 1413 | sft | exp-35 | derived: openai/gsm8k train rationales + synthetic:self (exp-08 samples) | completed |  | inconclusive | reject | True |  |
| train | r-918be760 | exp-52 | Qwen/Qwen3-4B-Base | 1416 | merge | exp-35 |  | completed | 0.8347232752084913 | contradicted | reject | True |  |
| train | r-918be760 | exp-53 | Qwen/Qwen3-4B-Base | 1422 | sft | exp-44 | openai/gsm8k (main, train split) | completed | 0.8407884761182715 | contradicted | reject | True |  |
| train | r-918be760 | exp-54 | Qwen/Qwen3-4B-Base | 1430 | merge | exp-44 |  | completed | 0.8407884761182715 | contradicted | reject | False |  |
| train | r-918be760 | exp-55 | Qwen/Qwen3-4B-Base | 1459 | decode-config | exp-44 |  | completed | 0.8400303260045489 | contradicted | reject | False |  |
| train | r-918be760 | exp-56 | Qwen/Qwen3-4B-Base | 1489 | sft | exp-44 | open-r1/OpenR1-Math-220k (train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-918be760 | exp-57 | Qwen/Qwen3-4B-Base | 1496 | sft | exp-44 | open-r1/OpenR1-Math-220k (train split) | completed |  | inconclusive | reject | False |  |
| train | r-918be760 | exp-58 | Qwen/Qwen3-4B-Base | 1501 | merge | exp-44 |  | completed | 0.8400303260045489 | contradicted | reject | False |  |
| train | r-918be760 | exp-59 | Qwen/Qwen3-4B-Base | 1507 | merge | exp-44 |  | completed | 0.8385140257771039 | contradicted | reject | True |  |
| train | r-918be760 | exp-60 | Qwen/Qwen3-4B-Base | 1516 | other | exp-44 |  | completed | 0.8423047763457164 | supported | adopt | True | 0.8392721758908264 |
| train | r-918be760 | exp-61 | Qwen/Qwen3-4B-Base | 1527 | decode-config | exp-60 |  | completed | 0.8453373768006065 | contradicted | reject | True |  |
| train | r-918be760 | exp-62 | Qwen/Qwen3-4B-Base | 1545 | decode-config | exp-60 |  | completed | 0.8369977255496588 | contradicted | reject | False |  |
| train | r-918be760 | exp-63 | Qwen/Qwen3-4B-Base | 1569 | merge | exp-44 |  | completed | 0.8392721758908264 | contradicted | reject | True |  |
| train | r-918be760 | exp-64 | Qwen/Qwen3-4B-Base | 1574 | merge | exp-44 |  | completed | 0.8369977255496588 | contradicted | reject | False |  |
| train | r-918be760 | exp-65 | Qwen/Qwen3-4B-Base | 1581 | merge | exp-35 |  | completed | 0.8415466262319939 | contradicted | reject | True |  |
| train | r-918be760 | exp-66 | Qwen/Qwen3-4B-Base | 1585 | merge | exp-35 |  | completed | 0.8339651250947687 | contradicted | reject | True |  |
| train | r-928e1ff3 | exp-01 | Qwen/Qwen3-1.7B-Base | 299 | decode-config | base_model |  | completed | 0.287 | supported | adopt | True |  |
| train | r-928e1ff3 | exp-02 | Qwen/Qwen3-1.7B-Base | 501 | sft | base_model | nvidia/OpenMathInstruct-2 | completed | 0.822 | supported | adopt | False |  |
| train | r-928e1ff3 | exp-03 | Qwen/Qwen3-1.7B-Base | 648 | grpo | exp-02 | openai/gsm8k | failed |  | inconclusive | abandon_line | False |  |
| train | r-928e1ff3 | exp-04 | Qwen/Qwen3-1.7B-Base | 654 | grpo | exp-02 | openai/gsm8k | killed |  | inconclusive | abandon_line | False |  |
| train | r-928e1ff3 | exp-05 | Qwen/Qwen3-1.7B-Base | 732 | grpo | exp-02 | openai/gsm8k | killed |  | inconclusive | abandon_line | False |  |
| train | r-928e1ff3 | exp-06 | Qwen/Qwen3-1.7B-Base | 822 | grpo | exp-02 | openai/gsm8k | failed |  | inconclusive | abandon_line | True |  |
| train | r-928e1ff3 | exp-07 | Qwen/Qwen3-1.7B-Base | 894 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-928e1ff3 | exp-08 | Qwen/Qwen3-1.7B-Base | 896 | grpo | exp-02 | openai/gsm8k | killed | 0.8467 | inconclusive | adopt | True |  |
| train | r-928e1ff3 | exp-09 | Qwen/Qwen3-1.7B-Base | 1055 | other | exp-08 |  | completed |  | inconclusive | adopt | False |  |
| train | r-928e1ff3 | exp-10 | Qwen/Qwen3-1.7B-Base | 1057 | grpo | exp-08 | openai/gsm8k | failed | 0.84 | inconclusive | adopt | True |  |
| train | r-928e1ff3 | exp-11 | Qwen/Qwen3-1.7B-Base | 1155 | grpo | exp-10 | openai/gsm8k | completed | 0.8525 | inconclusive | adopt | False |  |
| train | r-928e1ff3 | exp-12 | Qwen/Qwen3-1.7B-Base | 1242 | other | exp-11 |  | completed | 0.84 | inconclusive | adopt | False | 0.824109173616376 |
| train | r-9395d5af | exp-01 | Qwen/Qwen3-4B-Base | 21187 | sft | base_model | synthetic:self | failed |  | inconclusive | abandon_line | True |  |
| train | r-9395d5af | exp-02 | Qwen/Qwen3-4B-Base | 21390 | sft | base_model | synthetic:self | failed |  | inconclusive | abandon_line | True |  |
| train | r-9395d5af | exp-03 | Qwen/Qwen3-4B-Base | 21759 | sft | base_model | synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-9395d5af | exp-04 | Qwen/Qwen3-4B-Base | 23390 | sft | base_model | synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-9395d5af | exp-05 | Qwen/Qwen3-4B-Base | 25839 | sft | base_model | synthetic:self | completed |  | inconclusive | adopt | False |  |
| train | r-9395d5af | exp-06 | Qwen/Qwen3-4B-Base | 30591 | decode-config | exp-05 |  | completed | 0.8666 | supported | adopt | True |  |
| train | r-9395d5af | exp-07 | Qwen/Qwen3-4B-Base | 30834 | decode-config | exp-05 |  | completed | 0.8467 | inconclusive | reject | False |  |
| train | r-9395d5af | exp-08 | Qwen/Qwen3-4B-Base | 33936 | sft | base_model | synthetic:self + synthetic:Qwen/Qwen2.5-Math-7B-Instruct | completed |  | inconclusive | reject | False |  |
| train | r-9395d5af | exp-09 | Qwen/Qwen3-4B-Base | 34789 | decode-config | exp-08 |  | completed | 0.873 | inconclusive | reject | False |  |
| train | r-9395d5af | exp-10 | Qwen/Qwen3-4B-Base | 34865 | decode-config | exp-08 |  | completed | 0.8533 | inconclusive | reject | False |  |
| train | r-9395d5af | exp-11 | Qwen/Qwen3-4B-Base | 35513 | decode-config | exp-08 |  | completed | 0.8552 | contradicted | reject | True |  |
| train | r-9395d5af | exp-12 | Qwen/Qwen3-4B-Base | 36735 | decode-config | exp-05 |  | completed | 0.86657 | supported | adopt | True | 0.8658074298711145 |
| train | r-946d7e92 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 417 | sft | base_model | derived: openai/gsm8k train (dev-500 held out) + nvidia/OpenMathInstruct-2 (gsm8k-sourced rows) | killed |  | inconclusive | abandon_line | False |  |
| train | r-946d7e92 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 460 | sft | base_model | derived: openai/gsm8k train (dev-500 held out) + nvidia/OpenMathInstruct-2 (gsm8k-sourced rows) | completed | 0.865 | supported | adopt | False |  |
| train | r-946d7e92 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 1128 | sft | base_model | derived:exp-02 (rejection-sampled self-solutions) + openai/gsm8k train + nvidia/OpenMathInstruct-2,synthetic:self | completed | 0.8333 | contradicted | reject | False |  |
| train | r-946d7e92 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 1329 | grpo | exp-02 | derived:exp-02 (questions filtered by the exp-02 model's own solve rate) | killed |  | inconclusive | abandon_line | False |  |
| train | r-946d7e92 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 1365 | grpo | exp-02 | derived:exp-02 (questions filtered by the exp-02 model's own solve rate) | failed |  | inconclusive | abandon_line | False |  |
| train | r-946d7e92 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 1458 | grpo | exp-02 | derived:exp-02 (questions filtered by the exp-02 model's own solve rate) | completed | 0.88 | supported | adopt | False |  |
| train | r-946d7e92 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 1763 | merge | exp-06 |  | completed | 0.878 | inconclusive | reject | False |  |
| train | r-946d7e92 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1763 | merge | exp-02 |  | completed | 0.84 | contradicted | reject | False |  |
| train | r-946d7e92 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 1845 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-946d7e92 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 1882 | other | exp-06 |  | completed | 0.874 | inconclusive | adopt | False |  |
| train | r-946d7e92 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 1929 | other | exp-10 |  | completed |  | inconclusive | adopt | False |  |
| train | r-946d7e92 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 1929 | grpo | exp-10 | derived:exp-02 (questions filtered by the exp-02 model's own solve rate) | completed | 0.88 | inconclusive | adopt | False |  |
| train | r-946d7e92 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 2009 | other | exp-12 |  | completed | 0.698 | contradicted | reject | False |  |
| train | r-946d7e92 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 2071 | other | exp-12 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-946d7e92 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 2089 | other | exp-12 |  | completed | 0.8533 | inconclusive | reject | False |  |
| train | r-946d7e92 | exp-16 | HuggingFaceTB/SmolLM3-3B-Base | 2175 | other | exp-02 |  | completed | 0.8467 | supported | adopt | False | 0.8021228203184231 |
| train | r-94f796fd | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 121 | sft | base_model | HF id: openai/gsm8k | completed | 0.1 | contradicted | reject | True |  |
| train | r-94f796fd | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 190 | sft | base_model | HF id: openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-94f796fd | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 213 | sft | exp-02 | HF id: openai/gsm8k | completed | 0.52 | supported | reject | True |  |
| train | r-94f796fd | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 228 | sft | exp-02 | HF id: openai/gsm8k | completed | 0.41 | inconclusive | adopt | True |  |
| train | r-94f796fd | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 290 | sft | exp-04 | HF id: openai/gsm8k | completed | 0.64 | supported | adopt | True |  |
| train | r-94f796fd | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 333 | sft | exp-05 | HF id: openai/gsm8k | completed |  | inconclusive | reject | True |  |
| train | r-94f796fd | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 355 | other | exp-05 |  | completed | 0.5666666666666667 | inconclusive | adopt | False | 0.5913570887035633 |
| train | r-960d1065 | exp-01 | Qwen/Qwen3-4B-Base | 91 | sft | base_model | derived: meta-math/MetaMathQA (GSM-typed rows) + openai/gsm8k train | killed |  | inconclusive | abandon_line | False |  |
| train | r-960d1065 | exp-02 | Qwen/Qwen3-4B-Base | 145 | sft | base_model | derived: meta-math/MetaMathQA (GSM-typed rows) + openai/gsm8k train | completed | 0.82 | inconclusive | adopt | True |  |
| train | r-960d1065 | exp-03 | Qwen/Qwen3-4B-Base | 301 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-960d1065 | exp-04 | Qwen/Qwen3-4B-Base | 301 | grpo | exp-02 | HF openai/gsm8k (config main), split train | completed | 0.78 | inconclusive | adopt | True |  |
| train | r-960d1065 | exp-05 | Qwen/Qwen3-4B-Base | 346 | sft | exp-04 | derived: meta-math/MetaMathQA (GSM-typed rows) + openai/gsm8k train | completed | 0.78 | inconclusive | reject | False |  |
| train | r-960d1065 | exp-06 | Qwen/Qwen3-4B-Base | 405 | other | exp-04 |  | completed |  | inconclusive | reject | False |  |
| train | r-960d1065 | exp-07 | Qwen/Qwen3-4B-Base | 421 | rft | exp-04 | synthetic:self (rejection samples from exp-04) + openai/gsm8k train | failed | 0.8067 | inconclusive | adopt | False |  |
| train | r-960d1065 | exp-08 | Qwen/Qwen3-4B-Base | 453 | other | exp-07 |  | completed | 0.7467 | inconclusive | reject | False |  |
| train | r-960d1065 | exp-09 | Qwen/Qwen3-4B-Base | 475 | decode-config | exp-08 |  | completed |  | inconclusive | reject | True |  |
| train | r-960d1065 | exp-10 | Qwen/Qwen3-4B-Base | 485 | decode-config | exp-09 |  | completed | 0.78 | inconclusive | reject | True |  |
| train | r-960d1065 | exp-11 | Qwen/Qwen3-4B-Base | 508 | other | exp-07 |  | completed |  | inconclusive | adopt | True | 0.7702805155420773 |
| train | r-96346341 | exp-01 | Qwen/Qwen3-4B-Base | 418 | sft | base_model | derived: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k, math + augmented_math) + microsoft/orca-math-word-problems-200k + openai/gsm8k train (few-shot pool) | killed |  | inconclusive | abandon_line | False |  |
| train | r-96346341 | exp-02 | Qwen/Qwen3-4B-Base | 713 | sft | base_model | derived: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k, math + augmented_math) + microsoft/orca-math-word-problems-200k + openai/gsm8k train (few-shot pool) | completed | 0.887 | supported | adopt | False |  |
| train | r-96346341 | exp-03 | Qwen/Qwen3-4B-Base | 950 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | failed |  | inconclusive | abandon_line | False |  |
| train | r-96346341 | exp-04 | Qwen/Qwen3-4B-Base | 1069 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | failed |  | inconclusive | abandon_line | True |  |
| train | r-96346341 | exp-05 | Qwen/Qwen3-4B-Base | 1086 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-96346341 | exp-06 | Qwen/Qwen3-4B-Base | 1171 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | killed | 0.912 | supported | adopt | True |  |
| train | r-96346341 | exp-07 | Qwen/Qwen3-4B-Base | 1270 | other | exp-06 |  | completed |  | inconclusive | adopt | False | 0.9037149355572404 |
| train | r-96bf32c3 | exp-01 | Qwen/Qwen3-4B-Base | 85 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (GSM_* subset) | completed | 0.36 | inconclusive | reject | False |  |
| train | r-96bf32c3 | exp-02 | Qwen/Qwen3-4B-Base | 212 | sft | base_model | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.64 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-03 | Qwen/Qwen3-4B-Base | 261 | sft | base_model | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.6666666666666666 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-04 | Qwen/Qwen3-4B-Base | 271 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-05 | Qwen/Qwen3-4B-Base | 326 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-06 | Qwen/Qwen3-4B-Base | 339 | sft | exp-03 | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.7 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-07 | Qwen/Qwen3-4B-Base | 381 | other | exp-06 |  | completed |  | inconclusive | adopt | False | 0.6762699014404853 |
| train | r-96bf32c3 | exp-08 | Qwen/Qwen3-4B-Base | 391 | sft | exp-06 | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_2M) | completed | 0.66 | contradicted | reject | True |  |
| train | r-98b1304c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 54 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | failed |  | inconclusive | iterate | False |  |
| train | r-98b1304c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 63 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | failed |  | inconclusive | iterate | False |  |
| train | r-98b1304c | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 74 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | adopt | False |  |
| train | r-98b1304c | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 119 | merge | exp-03 |  | completed | 0.125 | inconclusive | reject | False |  |
| train | r-98b1304c | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 167 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 190 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 209 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | completed | 0.2 | supported | adopt | True |  |
| train | r-98b1304c | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 233 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 261 | other | exp-07 |  | completed | 0.31 | inconclusive | adopt | False | 0.24715693707354056 |
| train | r-98c04d79 | exp-01 | Qwen/Qwen3-4B-Base | 25 | other | base_model |  | completed | 0.4 | inconclusive | reject | False |  |
| train | r-98c04d79 | exp-02 | Qwen/Qwen3-4B-Base | 75 | sft | base_model | HF:openai/gsm8k (main, train split) | completed | 0.16 | inconclusive | reject | True |  |
| train | r-98c04d79 | exp-03 | Qwen/Qwen3-4B-Base | 100 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script | completed | 0.14 | contradicted | reject | True |  |
| train | r-98c04d79 | exp-04 | Qwen/Qwen3-4B-Base | 121 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script | completed | 0.02 | contradicted | reject | True |  |
| train | r-98c04d79 | exp-05 | Qwen/Qwen3-4B-Base | 134 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script | completed | 0.5333 | supported | adopt | True |  |
| train | r-98c04d79 | exp-06 | Qwen/Qwen3-4B-Base | 148 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script,HF:meta-math/MetaMathQA (train), loaded in-process by the training script | killed |  | inconclusive | adopt | True |  |
| train | r-98c04d79 | exp-07 | Qwen/Qwen3-4B-Base | 171 | merge | exp-06 |  | completed | 0.58 | supported | adopt | False |  |
| train | r-98c04d79 | exp-08 | Qwen/Qwen3-4B-Base | 202 | other | exp-07 |  | completed | 0.573 | supported | adopt | False | 0.5693707354056103 |
| train | r-98c04d79 | exp-09 | Qwen/Qwen3-4B-Base | 218 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script,HF:meta-math/MetaMathQA (train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98c04d79 | exp-10 | Qwen/Qwen3-4B-Base | 249 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script | completed | 0.553 | contradicted | reject | True |  |
| train | r-98c04d79 | exp-11 | Qwen/Qwen3-4B-Base | 315 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script,HF:meta-math/MetaMathQA (train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-01 | Qwen/Qwen3-4B-Base | 50 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | failed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-02 | Qwen/Qwen3-4B-Base | 63 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | False |  |
| train | r-990a90a4 | exp-03 | Qwen/Qwen3-4B-Base | 82 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | completed | 0.34 | inconclusive | reject | True |  |
| train | r-990a90a4 | exp-04 | Qwen/Qwen3-4B-Base | 161 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | completed | 0.28 | contradicted | reject | True |  |
| train | r-990a90a4 | exp-05 | Qwen/Qwen3-4B-Base | 199 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | completed | 0.28 | contradicted | adopt | True |  |
| train | r-990a90a4 | exp-06 | Qwen/Qwen3-4B-Base | 261 | decode-config | exp-05 |  | completed | 0.5 | supported | adopt | True |  |
| train | r-990a90a4 | exp-07 | Qwen/Qwen3-4B-Base | 275 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-08 | Qwen/Qwen3-4B-Base | 344 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-09 | Qwen/Qwen3-4B-Base | 359 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-10 | Qwen/Qwen3-4B-Base | 375 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-11 | Qwen/Qwen3-4B-Base | 380 | sft | exp-05 | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | completed | 0.76 | supported | adopt | True |  |
| train | r-990a90a4 | exp-12 | Qwen/Qwen3-4B-Base | 508 | sft | exp-11 | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-13 | Qwen/Qwen3-4B-Base | 511 | other | exp-11 |  | completed |  | inconclusive | adopt | True |  |
| train | r-990a90a4 | exp-14 | Qwen/Qwen3-4B-Base | 530 | sft | exp-11 | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-15 | Qwen/Qwen3-4B-Base | 552 | sft | exp-11 | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | completed | 0.7 | contradicted | reject | True |  |
| train | r-990a90a4 | exp-16 | Qwen/Qwen3-4B-Base | 596 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-17 | Qwen/Qwen3-4B-Base | 611 | decode-config | exp-13 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-990a90a4 | exp-18 | Qwen/Qwen3-4B-Base | 654 | decode-config | exp-13 |  | completed | 0.76 | inconclusive | reject | True |  |
| train | r-990a90a4 | exp-19 | Qwen/Qwen3-4B-Base | 666 | decode-config | exp-13 |  | completed | 0.84 | supported | reject | True |  |
| train | r-990a90a4 | exp-20 | Qwen/Qwen3-4B-Base | 676 | decode-config | exp-13 |  | completed | 0.813 | supported | iterate | True |  |
| train | r-990a90a4 | exp-21 | Qwen/Qwen3-4B-Base | 681 | decode-config | exp-13 |  | completed | 0.787 | supported | reject | True |  |
| train | r-990a90a4 | exp-22 | Qwen/Qwen3-4B-Base | 686 | decode-config | exp-13 |  | completed | 0.8 | supported | reject | True |  |
| train | r-990a90a4 | exp-23 | Qwen/Qwen3-4B-Base | 691 | decode-config | exp-13 |  | completed | 0.787 | supported | reject | True |  |
| train | r-990a90a4 | exp-24 | Qwen/Qwen3-4B-Base | 696 | decode-config | exp-13 |  | completed | 0.793 | supported | reject | True |  |
| train | r-990a90a4 | exp-25 | Qwen/Qwen3-4B-Base | 701 | decode-config | exp-13 |  | completed | 0.8 | supported | adopt | True | 0.8347232752084913 |
| train | r-990a90a4 | exp-26 | Qwen/Qwen3-4B-Base | 708 | sft | base_model | HF: openai/gsm8k train + meta-math/MetaMathQA (GSM types) | killed |  | inconclusive | abandon_line | True |  |
| train | r-9e0ad3aa | exp-01 | Qwen/Qwen3-1.7B-Base | 82 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM') | failed |  | inconclusive | iterate | False |  |
| train | r-9e0ad3aa | exp-02 | Qwen/Qwen3-1.7B-Base | 88 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM') | completed | 0.03 | inconclusive | reject | False |  |
| train | r-9e0ad3aa | exp-03 | Qwen/Qwen3-1.7B-Base | 134 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.04 | inconclusive | adopt | False |  |
| train | r-9e0ad3aa | exp-04 | Qwen/Qwen3-1.7B-Base | 180 | decode-config | exp-03 |  | completed | 0.4 | supported | reject | False |  |
| train | r-9e0ad3aa | exp-05 | Qwen/Qwen3-1.7B-Base | 190 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM'); 8-shot prefix from HF openai/gsm8k (main, split=train) | completed | 0.02 | contradicted | adopt | True |  |
| train | r-9e0ad3aa | exp-06 | Qwen/Qwen3-1.7B-Base | 214 | decode-config | exp-05 |  | completed | 0.03 | inconclusive | adopt | False | 0.043214556482183475 |
| train | r-9f1c9470 | exp-01 | Qwen/Qwen3-1.7B-Base | 90 | sft | base_model | HF openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | True |  |
| train | r-9f1c9470 | exp-02 | Qwen/Qwen3-1.7B-Base | 127 | merge | exp-01 |  | completed | 0.175 | inconclusive | adopt | False |  |
| train | r-9f1c9470 | exp-03 | Qwen/Qwen3-1.7B-Base | 161 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.025 | contradicted | reject | True |  |
| train | r-9f1c9470 | exp-04 | Qwen/Qwen3-1.7B-Base | 235 | sft | base_model | HF openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | True |  |
| train | r-9f1c9470 | exp-05 | Qwen/Qwen3-1.7B-Base | 248 | merge | exp-04 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-9f1c9470 | exp-06 | Qwen/Qwen3-1.7B-Base | 291 | other | exp-02 |  | completed | 0.18666666666666668 | inconclusive | adopt | False | 0.1645185746777862 |
| train | r-a236488d | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 16794 | sft | base_model | openai/gsm8k (train) + nvidia/OpenMathInstruct-2 | failed |  | inconclusive | abandon_line | False |  |
| train | r-a236488d | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 23266 | sft | base_model | openai/gsm8k (train) + nvidia/OpenMathInstruct-2 | completed | 0.773 | supported | adopt | False |  |
| train | r-a236488d | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 27964 | rft | exp-02 | synthetic:self,derived:exp-03 (rft_r1.jsonl) + derived:exp-02 (slice of sft_v2.jsonl) | completed | 0.773 | supported | adopt | True |  |
| train | r-a236488d | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 36125 | dpo | exp-03 | synthetic:self,derived:exp-04 (labeled_r2.jsonl) | completed | 0.8 | supported | adopt | False |  |
| train | r-a236488d | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 37006 | dpo | exp-04 | synthetic:self,derived:exp-05 (labeled_r3.jsonl) | failed |  | inconclusive | abandon_line | False |  |
| train | r-a236488d | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 37131 | other | exp-04 |  | completed |  | inconclusive | reject | False |  |
| train | r-a236488d | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 38094 | dpo | exp-04 | derived:exp-05 (labeled_r3.jsonl) | completed | 0.7953 | supported | adopt | False |  |
| train | r-a236488d | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 39167 | other | exp-07 |  | failed |  | inconclusive | abandon_line | False |  |
| train | r-a236488d | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 39972 | other | exp-07 |  | completed | 0.8175 | inconclusive | adopt | False | 0.7952994692949203 |
| train | r-a2777ce1 | exp-01 | Qwen/Qwen3-1.7B-Base | 28 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | False |  |
| train | r-a2777ce1 | exp-02 | Qwen/Qwen3-1.7B-Base | 46 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-a2777ce1 | exp-03 | Qwen/Qwen3-1.7B-Base | 70 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-a2777ce1 | exp-04 | Qwen/Qwen3-1.7B-Base | 98 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-a2777ce1 | exp-05 | Qwen/Qwen3-1.7B-Base | 120 | sft | base_model | openai/gsm8k | completed | 0.34 | supported | adopt | False |  |
| train | r-a2777ce1 | exp-06 | Qwen/Qwen3-1.7B-Base | 140 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | completed | 0.327 | inconclusive | adopt | False |  |
| train | r-a2777ce1 | exp-07 | Qwen/Qwen3-1.7B-Base | 162 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-a2777ce1 | exp-08 | Qwen/Qwen3-1.7B-Base | 164 | sft | base_model | openai/gsm8k | completed | 0.507 | supported | adopt | False |  |
| train | r-a2777ce1 | exp-09 | Qwen/Qwen3-1.7B-Base | 176 | other | exp-08 |  | completed |  | inconclusive | adopt | False |  |
| train | r-a2777ce1 | exp-10 | Qwen/Qwen3-1.7B-Base | 178 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | completed | 0.453 | inconclusive | reject | False |  |
| train | r-a2777ce1 | exp-11 | Qwen/Qwen3-1.7B-Base | 210 | other | exp-08 |  | completed | 0.573 | inconclusive | adopt | False | 0.5435936315390447 |
| train | r-a2777ce1 | exp-12 | Qwen/Qwen3-1.7B-Base | 216 | sft | base_model | openai/gsm8k | completed | 0.333 | contradicted | reject | False |  |
| train | r-a2777ce1 | exp-13 | Qwen/Qwen3-1.7B-Base | 236 | sft | base_model | openai/gsm8k | completed | 0.1 | contradicted | reject | False |  |
| train | r-a2d62ed3 | exp-01 | Qwen/Qwen3-4B-Base | 307 | sft | base_model | local | failed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-02 | Qwen/Qwen3-4B-Base | 352 | sft | base_model | local | failed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-03 | Qwen/Qwen3-4B-Base | 377 | sft | base_model | local | failed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-04 | Qwen/Qwen3-4B-Base | 396 | sft | base_model | local | killed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-05 | Qwen/Qwen3-4B-Base | 418 | sft | base_model | local | killed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-06 | Qwen/Qwen3-4B-Base | 433 | sft | base_model | local | completed | 0.8733 | supported | adopt | True |  |
| train | r-a2d62ed3 | exp-07 | Qwen/Qwen3-4B-Base | 642 | rft | exp-06 | synthetic:self + local | failed |  | inconclusive | iterate | True |  |
| train | r-a2d62ed3 | exp-08 | Qwen/Qwen3-4B-Base | 697 | rft | exp-06 | synthetic:self + local | completed | 0.8933 | supported | adopt | True |  |
| train | r-a2d62ed3 | exp-09 | Qwen/Qwen3-4B-Base | 706 | other | exp-06 |  | completed |  | inconclusive | adopt | True |  |
| train | r-a2d62ed3 | exp-10 | Qwen/Qwen3-4B-Base | 735 | grpo | exp-08 | openai/gsm8k (main, train split) | failed |  | inconclusive | iterate | False |  |
| train | r-a2d62ed3 | exp-11 | Qwen/Qwen3-4B-Base | 759 | grpo | exp-08 | openai/gsm8k (main, train split) | completed | 0.9 | supported | adopt | False |  |
| train | r-a2d62ed3 | exp-12 | Qwen/Qwen3-4B-Base | 827 | grpo | exp-11 | openai/gsm8k (main, train split) | completed | 0.92 | supported | adopt | True |  |
| train | r-a2d62ed3 | exp-13 | Qwen/Qwen3-4B-Base | 838 | merge | exp-08 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-a2d62ed3 | exp-14 | Qwen/Qwen3-4B-Base | 904 | grpo | exp-12 | openai/gsm8k (main, train split) | failed | 0.92 | inconclusive | reject | True |  |
| train | r-a2d62ed3 | exp-15 | Qwen/Qwen3-4B-Base | 982 | other | exp-12 |  | completed | 0.92 | supported | adopt | True | 0.9067475360121304 |
| train | r-a3f23d29 | exp-01 | Qwen/Qwen3-4B-Base | 196 | sft | base_model | local (openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.6266666666666667 | inconclusive | adopt | True |  |
| train | r-a3f23d29 | exp-02 | Qwen/Qwen3-4B-Base | 401 | decode-config | exp-01 |  | completed | 0.84 | supported | adopt | True |  |
| train | r-a3f23d29 | exp-03 | Qwen/Qwen3-4B-Base | 501 | rft | base_model | local (openai/gsm8k train + synthetic:self + meta-math/MetaMathQA),synthetic:self (sampled from exp-02's checkpoint) | completed | 0.84 | contradicted | reject | True |  |
| train | r-a3f23d29 | exp-04 | Qwen/Qwen3-4B-Base | 650 | sft | base_model | local (openai/gsm8k train + synthetic:self + nvidia/OpenMathInstruct-2),nvidia/OpenMathInstruct-2 | completed | 0.846 | inconclusive | adopt | True |  |
| train | r-a3f23d29 | exp-05 | Qwen/Qwen3-4B-Base | 697 | sft | base_model | local (openai/gsm8k train + synthetic:self + nvidia/OpenMathInstruct-2 + meta-math/MetaMathQA) | completed | 0.86 | supported | adopt | True | 0.8673237300985596 |
| train | r-a436c040 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 39 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | failed |  | inconclusive | iterate | True |  |
| train | r-a436c040 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 45 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | failed |  | inconclusive | iterate | True |  |
| train | r-a436c040 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 51 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | False |  |
| train | r-a436c040 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 59 | merge | exp-03 |  | completed | 0.05 | contradicted | reject | True |  |
| train | r-a436c040 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 91 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-a436c040 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 97 | merge | exp-05 |  | completed | 0.35 | supported | reject | True |  |
| train | r-a436c040 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 109 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-a436c040 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 124 | merge | exp-07 |  | completed | 0.425 | supported | adopt | True |  |
| train | r-a436c040 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 133 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-a436c040 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 156 | other | exp-08 |  | completed | 0.4866666666666667 | inconclusive | adopt | True | 0.5322213798332069 |
| train | r-a4de3854 | exp-01 | Qwen/Qwen3-1.7B-Base | 62 | sft | base_model | local,local | killed |  | inconclusive | abandon_line | False |  |
| train | r-a4de3854 | exp-02 | Qwen/Qwen3-1.7B-Base | 66 | sft | base_model | local,local | completed | 0.15333333333333332 | inconclusive | reject | False |  |
| train | r-a4de3854 | exp-03 | Qwen/Qwen3-1.7B-Base | 158 | sft | base_model | local,local | completed | 0.22 | supported | reject | True |  |
| train | r-a4de3854 | exp-04 | Qwen/Qwen3-1.7B-Base | 194 | sft | base_model | local,local | completed | 0.13333333333333333 | contradicted | reject | True |  |
| train | r-a4de3854 | exp-05 | Qwen/Qwen3-1.7B-Base | 240 | sft | base_model | local,local | completed | 0.16666666666666666 | inconclusive | adopt | True |  |
| train | r-a4de3854 | exp-06 | Qwen/Qwen3-1.7B-Base | 278 | grpo | exp-05 | openai/gsm8k (train split, loaded inside train_grpo.py; no local file) | killed |  | inconclusive | abandon_line | False |  |
| train | r-a4de3854 | exp-07 | Qwen/Qwen3-1.7B-Base | 288 | grpo | exp-05 | openai/gsm8k (train split, loaded inside train_grpo.py; no local file) | killed |  | inconclusive | adopt | False |  |
| train | r-a4de3854 | exp-08 | Qwen/Qwen3-1.7B-Base | 322 | other | exp-07 |  | completed | 0.12 | contradicted | reject | False |  |
| train | r-a4de3854 | exp-09 | Qwen/Qwen3-1.7B-Base | 368 | sft | base_model | local,local | completed |  | inconclusive | adopt | True | 0.1281273692191054 |
| train | r-a5217b8e | exp-01 | Qwen/Qwen3-4B-Base | 300 | merge | base_model | derived:adapter checkpoint | completed |  | inconclusive | abandon_line | True |  |
| train | r-a5217b8e | exp-02 | Qwen/Qwen3-4B-Base | 301 | merge | base_model | derived:adapter checkpoint | completed | 0.1 | inconclusive | reject | True |  |
| train | r-a5217b8e | exp-03 | Qwen/Qwen3-4B-Base | 328 | decode-config | exp-02 | none - inference-time configuration change | completed | 0.15 | inconclusive | reject | True |  |
| train | r-a5217b8e | exp-04 | Qwen/Qwen3-4B-Base | 339 | merge | base_model | derived:adapter checkpoint | completed |  | inconclusive | abandon_line | True |  |
| train | r-a5217b8e | exp-05 | Qwen/Qwen3-4B-Base | 358 | sft | base_model | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.6 | inconclusive | adopt | True |  |
| train | r-a5217b8e | exp-06 | Qwen/Qwen3-4B-Base | 421 | sft | exp-05 | openai/gsm8k (main, train) loaded in-process by train_sft.py,math-ai/TemplateGSM (templategsm-1000-1k, streaming, shuffled seed 42, buffer 10000) | completed | 0.35 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-07 | Qwen/Qwen3-4B-Base | 503 | sft | base_model | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.44 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-08 | Qwen/Qwen3-4B-Base | 531 | merge | exp-05 | derived:adapter checkpoint | completed | 0.56 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-09 | Qwen/Qwen3-4B-Base | 545 | other | exp-05 | derived:exp-05 | completed | 0.68 | inconclusive | adopt | False |  |
| train | r-a5217b8e | exp-10 | Qwen/Qwen3-4B-Base | 614 | sft | exp-05 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.73 | supported | adopt | True |  |
| train | r-a5217b8e | exp-11 | Qwen/Qwen3-4B-Base | 641 | sft | exp-10 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.76 | supported | adopt | True |  |
| train | r-a5217b8e | exp-12 | Qwen/Qwen3-4B-Base | 660 | sft | exp-11 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.74 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-13 | Qwen/Qwen3-4B-Base | 674 | sft | exp-11 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.78 | supported | adopt | True |  |
| train | r-a5217b8e | exp-14 | Qwen/Qwen3-4B-Base | 687 | sft | exp-13 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.75 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-15 | Qwen/Qwen3-4B-Base | 701 | other | exp-13 | derived:exp-13 | completed | 0.78 | inconclusive | adopt | False |  |
| train | r-a5217b8e | exp-16 | Qwen/Qwen3-4B-Base | 823 | distill | exp-13 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train | completed | 0.853 | supported | adopt | True |  |
| train | r-a5217b8e | exp-17 | Qwen/Qwen3-4B-Base | 897 | other | exp-16 | derived:exp-16 | completed | 0.8533333333333334 | inconclusive | adopt | False |  |
| train | r-a5217b8e | exp-18 | Qwen/Qwen3-4B-Base | 901 | sft | exp-16 | openai/gsm8k (main, train) loaded in-process by train_sft.py | completed | 0.76 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-19 | Qwen/Qwen3-4B-Base | 922 | distill | exp-13 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train | completed | 0.76 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-20 | Qwen/Qwen3-4B-Base | 1026 | distill | exp-17 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train,derived:exp-17 (self-rationalised misses) mixed with the teacher corpus | completed | 0.87 | supported | adopt | True |  |
| train | r-a5217b8e | exp-21 | Qwen/Qwen3-4B-Base | 1064 | other | exp-20 | derived:exp-20 | completed | 0.87 | inconclusive | adopt | False |  |
| train | r-a5217b8e | exp-22 | Qwen/Qwen3-4B-Base | 1210 | distill | exp-21 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train | completed | 0.87 | inconclusive | adopt | True | 0.8347232752084913 |
| train | r-a5217b8e | exp-23 | Qwen/Qwen3-4B-Base | 1234 | distill | exp-22 | derived:exp-16 self-distillation mixed with the teacher corpus | completed | 0.81 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-24 | Qwen/Qwen3-4B-Base | 1248 | distill | exp-21 | derived:exp-16 self-distillation (new misses only) mixed with the teacher corpus | completed | 0.84 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-25 | Qwen/Qwen3-4B-Base | 1314 | decode-config | exp-21 | none - inference-time configuration change | completed |  | inconclusive | reject | True |  |
| train | r-a5217b8e | exp-26 | Qwen/Qwen3-4B-Base | 1341 | distill | exp-21 | derived:exp-21 self-rationalised misses mixed with the teacher corpus | completed | 0.79 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-27 | Qwen/Qwen3-4B-Base | 1488 | distill | exp-21 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train, with gold-conditioned repairs | completed | 0.65 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-28 | Qwen/Qwen3-4B-Base | 1505 | distill | exp-21 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train, with gold-conditioned repairs | completed | 0.81 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-29 | Qwen/Qwen3-4B-Base | 1519 | distill | exp-21 | synthetic:Qwen/Qwen2.5-Math-7B-Instruct distilled on openai/gsm8k main train,derived:exp-17 self-rationalised misses mixed with the teacher corpus | completed | 0.82 | contradicted | reject | True |  |
| train | r-a5217b8e | exp-30 | Qwen/Qwen3-4B-Base | 1616 | dpo | exp-21 | synthetic:self (rejected) + Qwen/Qwen2.5-Math-7B-Instruct (chosen), mined on openai/gsm8k main train | completed | 0.83 | contradicted | reject | True |  |
| train | r-a759897f | exp-01 | Qwen/Qwen3-1.7B-Base | 23924 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | failed |  | inconclusive | iterate | True |  |
| train | r-a759897f | exp-02 | Qwen/Qwen3-1.7B-Base | 26284 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | killed |  | inconclusive | abandon_line | False |  |
| train | r-a759897f | exp-03 | Qwen/Qwen3-1.7B-Base | 31594 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | completed | 0.766 | inconclusive | adopt | True |  |
| train | r-a759897f | exp-04 | Qwen/Qwen3-1.7B-Base | 41076 | sft | exp-03 | synthetic:self,derived:exp-03 (STaR) + local (openai/gsm8k train, nvidia/OpenMathInstruct-2) | completed | 0.768 | inconclusive | adopt | True |  |
| train | r-a759897f | exp-05 | Qwen/Qwen3-1.7B-Base | 42663 | sft | base_model | derived:exp-03 (STaR) + local (openai/gsm8k train, nvidia/OpenMathInstruct-2) | completed | 0.76 | contradicted | adopt | False |  |
| train | r-a759897f | exp-06 | Qwen/Qwen3-1.7B-Base | 46442 | merge | exp-03 + exp-04 |  | failed |  | inconclusive | iterate | False |  |
| train | r-a759897f | exp-07 | Qwen/Qwen3-1.7B-Base | 46915 | merge | exp-03 + exp-04 |  | completed | 0.78 | supported | adopt | False |  |
| train | r-a759897f | exp-08 | Qwen/Qwen3-1.7B-Base | 47871 | merge | exp-03 + exp-04 + exp-05 |  | completed | 0.766 | contradicted | reject | False |  |
| train | r-a759897f | exp-09 | Qwen/Qwen3-1.7B-Base | 48225 | other | exp-07 |  | completed | 0.7627 | supported | adopt | False | 0.7604245640636846 |
| train | r-a759897f | exp-10 | Qwen/Qwen3-1.7B-Base | 49858 | merge | exp-03 + exp-04 |  | completed | 0.782 | contradicted | reject | True |  |
| train | r-a9cac75f | exp-01 | Qwen/Qwen3-1.7B-Base | 54 | sft | base_model | local (built from HF openai/gsm8k, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k) | killed |  | inconclusive | abandon_line | True |  |
| train | r-a9cac75f | exp-02 | Qwen/Qwen3-1.7B-Base | 90 | sft | base_model | local (built from HF openai/gsm8k, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k) | completed | 0.06 | contradicted | reject | False |  |
| train | r-a9cac75f | exp-03 | Qwen/Qwen3-1.7B-Base | 182 | sft | base_model | local (built from HF openai/gsm8k, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k) | completed | 0.06 | contradicted | adopt | True |  |
| train | r-a9cac75f | exp-04 | Qwen/Qwen3-1.7B-Base | 220 | decode-config | exp-03 |  | completed | 0.4333333333333333 | supported | adopt | True |  |
| train | r-a9cac75f | exp-05 | Qwen/Qwen3-1.7B-Base | 236 | sft | exp-04 | local (built from HF openai/gsm8k, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k) | completed | 0.5733333333333334 | supported | adopt | True |  |
| train | r-a9cac75f | exp-06 | Qwen/Qwen3-1.7B-Base | 270 | other | exp-05 |  | completed | 0.46 | inconclusive | adopt | False |  |
| train | r-a9cac75f | exp-07 | Qwen/Qwen3-1.7B-Base | 277 | sft | exp-06 | openai/gsm8k (main, train split), loaded in-script | completed | 0.4466666666666667 | contradicted | reject | True |  |
| train | r-a9cac75f | exp-08 | Qwen/Qwen3-1.7B-Base | 312 | decode-config | exp-06 |  | completed | 0.76 | supported | adopt | True | 0.7293404094010614 |
| train | r-a9cac75f | exp-09 | Qwen/Qwen3-1.7B-Base | 326 | decode-config | exp-07 |  | completed | 0.74 | supported | reject | True |  |
| train | r-aaf3560a | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 18 | other | base_model |  | completed | 0.167 | inconclusive | reject | False |  |
| train | r-aaf3560a | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 64 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.1 | inconclusive | reject | False |  |
| train | r-aaf3560a | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 87 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.2 | supported | reject | True |  |
| train | r-aaf3560a | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 107 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.34 | inconclusive | adopt | True |  |
| train | r-aaf3560a | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 125 | sft | base_model | HF id: openai/gsm8k (config main, train split) | killed | 0.33 | supported | adopt | False |  |
| train | r-aaf3560a | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 144 | sft | exp-05 | HF id: openai/gsm8k (config main, train split) | completed | 0.25 | contradicted | reject | False |  |
| train | r-aaf3560a | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 153 | other | exp-05 |  | completed | 0.233 | inconclusive | reject | True |  |
| train | r-aaf3560a | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 187 | other | exp-04 |  | completed | 0.34 | inconclusive | adopt | False | 0.33131159969673996 |
| train | r-ac9606db | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 138 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) | killed |  | inconclusive | abandon_line | False |  |
| train | r-ac9606db | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 208 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) | completed | 0.593 | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 299 | decode-config | exp-02 |  | completed | 0.767 | supported | adopt | True |  |
| train | r-ac9606db | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 309 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 318 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | False |  |
| train | r-ac9606db | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 392 | rft | base_model | synthetic:self,derived:exp-06 (data/reject.jsonl x2) + data/sft2.jsonl subsample | completed | 0.793 | supported | adopt | True |  |
| train | r-ac9606db | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 454 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 523 | rft | base_model | synthetic:self,derived:exp-06 (data/reject.jsonl) + derived:exp-08 (data/reject2.jsonl) + data/sft2.jsonl subsample | completed | 0.793 | contradicted | adopt | True |  |
| train | r-ac9606db | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 571 | other | exp-08 |  | completed | 0.78 | inconclusive | adopt | False | 0.7611827141774071 |
| train | r-ad144811 | exp-01 | Qwen/Qwen3-4B-Base | 79 | sft | base_model | local: openai/gsm8k train + meta-math/MetaMathQA (rows with exact GSM8K-train original_question) | completed | 0.7533 | supported | adopt | False |  |
| train | r-ad144811 | exp-02 | Qwen/Qwen3-4B-Base | 252 | sft | exp-01 | openai/gsm8k train split | completed | 0.7333 | contradicted | reject | False |  |
| train | r-ad144811 | exp-03 | Qwen/Qwen3-4B-Base | 300 | sft | exp-01 | openai/gsm8k train split | completed | 0.7 | contradicted | reject | True |  |
| train | r-ad144811 | exp-04 | Qwen/Qwen3-4B-Base | 379 | sft | exp-01 | openai/gsm8k train + meta-math/MetaMathQA (GSM8K-train provenance, queries disjoint from train_data) | completed | 0.7267 | contradicted | reject | True |  |
| train | r-ad144811 | exp-05 | Qwen/Qwen3-4B-Base | 494 | sft | exp-01 | local: openai/gsm8k train + meta-math/MetaMathQA (exact GSM8K-train provenance) | killed |  | inconclusive | abandon_line | True |  |
| train | r-ad144811 | exp-06 | Qwen/Qwen3-4B-Base | 503 | sft | exp-01 | local: openai/gsm8k train + meta-math/MetaMathQA (exact GSM8K-train provenance) | completed | 0.7867 | contradicted | adopt | True |  |
| train | r-ad144811 | exp-07 | Qwen/Qwen3-4B-Base | 540 | other | exp-06 |  | completed | 0.82 | supported | adopt | True |  |
| train | r-ad144811 | exp-08 | Qwen/Qwen3-4B-Base | 586 | sft | base_model | local: openai/gsm8k train + meta-math/MetaMathQA (exact GSM8K-train provenance) | completed | 0.04 | contradicted | reject | True |  |
| train | r-ad144811 | exp-09 | Qwen/Qwen3-4B-Base | 622 | merge | exp-07 |  | completed | 0.72 | contradicted | reject | True |  |
| train | r-ad144811 | exp-10 | Qwen/Qwen3-4B-Base | 786 | grpo | exp-07 | HF id: openai/gsm8k | completed | 0.7267 | contradicted | reject | True |  |
| train | r-ad144811 | exp-11 | Qwen/Qwen3-4B-Base | 801 | sft | exp-07 | openai/gsm8k train + meta-math/MetaMathQA alternate rationales (exact GSM8K-train provenance) | completed | 0.7333 | contradicted | reject | True |  |
| train | r-ad144811 | exp-12 | Qwen/Qwen3-4B-Base | 841 | sft | exp-01 | local: openai/gsm8k train + meta-math/MetaMathQA (exact GSM8K-train provenance) | killed | 0.76 | contradicted | reject | True |  |
| train | r-ad144811 | exp-13 | Qwen/Qwen3-4B-Base | 900 | other | exp-07 |  | completed | 0.7733 | inconclusive | adopt | True | 0.7619408642911296 |
| train | r-ad144811 | exp-14 | Qwen/Qwen3-4B-Base | 1012 | sft | exp-07 | openai/gsm8k train split | completed | 0.7733 | contradicted | reject | True |  |
| train | r-ad144811 | exp-15 | Qwen/Qwen3-4B-Base | 1050 | rft | exp-07 | synthetic:self (rollouts from the exp-07 checkpoint on openai/gsm8k train prompts) | killed | 0.74 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 72 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.56 | inconclusive | adopt | True |  |
| train | r-af68ee40 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 133 | sft | exp-01 | HF openai/gsm8k (config main, split train),synthetic:self | completed | 0.5667 | supported | adopt | True |  |
| train | r-af68ee40 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 161 | sft | exp-02 | HF openai/gsm8k (config main, split train),synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-af68ee40 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 184 | sft | exp-02 | HF openai/gsm8k (config main, split train),synthetic:self | killed | 0.6267 | supported | adopt | True |  |
| train | r-af68ee40 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 221 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.7 | contradicted | adopt | True |  |
| train | r-af68ee40 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 236 | other | exp-04 |  | completed | 0.5333 | contradicted | adopt | True |  |
| train | r-af68ee40 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 270 | decode-config | exp-06 |  | completed | 0.6933 | supported | adopt | True |  |
| train | r-af68ee40 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 281 | decode-config | exp-07 |  | completed | 0.5867 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 297 | decode-config | exp-01,exp-02,exp-04,exp-05 |  | completed | 0.7 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 321 | sft | exp-05 | HF openai/gsm8k (config main, split train) | completed | 0.6933 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 334 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.7067 | supported | reject | True |  |
| train | r-af68ee40 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 343 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.7267 | inconclusive | adopt | False |  |
| train | r-af68ee40 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 350 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.7 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 356 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.7 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 363 | other | exp-12 |  | completed | 0.7267 | supported | adopt | True |  |
| train | r-af68ee40 | exp-16 | HuggingFaceTB/SmolLM3-3B-Base | 371 | sft | exp-04 | HF openai/gsm8k (config main, split train) | completed | 0.6933 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-17 | HuggingFaceTB/SmolLM3-3B-Base | 428 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF microsoft/orca-math-word-problems-200k (split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-af68ee40 | exp-18 | HuggingFaceTB/SmolLM3-3B-Base | 460 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF microsoft/orca-math-word-problems-200k (split train) | completed | 0.7133 | contradicted | adopt | True |  |
| train | r-af68ee40 | exp-19 | HuggingFaceTB/SmolLM3-3B-Base | 473 | sft | exp-18 | HF openai/gsm8k (config main, split train) | completed | 0.7067 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-20 | HuggingFaceTB/SmolLM3-3B-Base | 487 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.74 | supported | adopt | True |  |
| train | r-af68ee40 | exp-21 | HuggingFaceTB/SmolLM3-3B-Base | 496 | sft | exp-20 | HF openai/gsm8k (config main, split train) | completed | 0.72 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-22 | HuggingFaceTB/SmolLM3-3B-Base | 517 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.72 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-23 | HuggingFaceTB/SmolLM3-3B-Base | 531 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.6933 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-24 | HuggingFaceTB/SmolLM3-3B-Base | 545 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.7133 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-25 | HuggingFaceTB/SmolLM3-3B-Base | 558 | sft | exp-15 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.72 | inconclusive | reject | False |  |
| train | r-af68ee40 | exp-26 | HuggingFaceTB/SmolLM3-3B-Base | 585 | other | exp-20 |  | completed | 0.74 | inconclusive | adopt | True |  |
| train | r-af68ee40 | exp-27 | HuggingFaceTB/SmolLM3-3B-Base | 642 | decode-config | exp-26 |  | completed | 0.74 | supported | adopt | True | 0.7240333586050038 |
| train | r-af68ee40 | exp-28 | HuggingFaceTB/SmolLM3-3B-Base | 658 | sft | exp-26 | HF openai/gsm8k (config main, split train) | completed | 0.72 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-29 | HuggingFaceTB/SmolLM3-3B-Base | 683 | merge | exp-20 |  | completed | 0.7133 | contradicted | reject | True |  |
| train | r-af68ee40 | exp-30 | HuggingFaceTB/SmolLM3-3B-Base | 748 | sft | exp-12 | HF openai/gsm8k (config main, split train),HF meta-math/MetaMathQA (split train) | completed | 0.7067 | contradicted | reject | True |  |
| train | r-b351b70e | exp-01 | Qwen/Qwen3-4B-Base | 44 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.475 | inconclusive | reject | False |  |
| train | r-b351b70e | exp-02 | Qwen/Qwen3-4B-Base | 52 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.435 | contradicted | reject | False |  |
| train | r-b351b70e | exp-03 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.545 | supported | reject | False |  |
| train | r-b351b70e | exp-04 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.63 | supported | reject | False |  |
| train | r-b351b70e | exp-05 | Qwen/Qwen3-4B-Base | 106 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.745 | supported | adopt | False | 0.7172100075815011 |
| train | r-b8779e0c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 77 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.5266666666666666 | inconclusive | adopt | True |  |
| train | r-b8779e0c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 110 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types + open-r1/OpenR1-Math-220k) | completed | 0.32 | contradicted | reject | True |  |
| train | r-b8779e0c | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 134 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.58 | supported | adopt | True |  |
| train | r-b8779e0c | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 158 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.49333333333333335 | contradicted | reject | True |  |
| train | r-b8779e0c | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 170 | other | exp-03 |  | completed | 0.593 | inconclusive | adopt | False | 0.6194086429112965 |
| train | r-b8ac2f29 | exp-01 | Qwen/Qwen3-4B-Base | 14168 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.8467 | inconclusive | reject | True |  |
| train | r-b8ac2f29 | exp-02 | Qwen/Qwen3-4B-Base | 23150 | sft | base_model | derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),HF nvidia/OpenMathInstruct-2 (train_1M),HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-b8ac2f29 | exp-03 | Qwen/Qwen3-4B-Base | 26878 | sft | base_model | derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),HF nvidia/OpenMathInstruct-2 (train_1M),HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-b8ac2f29 | exp-04 | Qwen/Qwen3-4B-Base | 31855 | sft | base_model | derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),HF nvidia/OpenMathInstruct-2 (train_1M),HF openai/gsm8k (main, train split) | completed | 0.88 | supported | adopt | True |  |
| train | r-b8ac2f29 | exp-05 | Qwen/Qwen3-4B-Base | 34592 | rft | exp-04 | synthetic:self + derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),synthetic:self (sampled from the exp-04 checkpoint on GSM8K train questions),HF openai/gsm8k (main, train split),HF nvidia/OpenMathInstruct-2 (train_1M) | killed |  | inconclusive | abandon_line | False |  |
| train | r-b8ac2f29 | exp-06 | Qwen/Qwen3-4B-Base | 35547 | rft | exp-04 | synthetic:self + derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),synthetic:self (sampled from the exp-04 checkpoint on GSM8K train questions),HF openai/gsm8k (main, train split),HF nvidia/OpenMathInstruct-2 (train_1M) | failed |  | inconclusive | abandon_line | False |  |
| train | r-b8ac2f29 | exp-07 | Qwen/Qwen3-4B-Base | 38690 | rft | exp-04 | synthetic:self + derived:exp-01 gold file + local (OpenMathInstruct-2 derivative),synthetic:self (sampled from the exp-04 checkpoint on GSM8K train questions),HF openai/gsm8k (main, train split),HF nvidia/OpenMathInstruct-2 (train_1M) | completed | 0.84 | contradicted | reject | False |  |
| train | r-b8ac2f29 | exp-08 | Qwen/Qwen3-4B-Base | 39005 | other | exp-04 |  | completed | 0.8867 | supported | adopt | True | 0.8438210765731615 |
| train | r-bc465eaf | exp-01 | Qwen/Qwen3-1.7B-Base | 49 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.05 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-02 | Qwen/Qwen3-1.7B-Base | 74 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-bc465eaf | exp-03 | Qwen/Qwen3-1.7B-Base | 137 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.0 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-04 | Qwen/Qwen3-1.7B-Base | 187 | other | exp-03 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-bc465eaf | exp-05 | Qwen/Qwen3-1.7B-Base | 199 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-bc465eaf | exp-06 | Qwen/Qwen3-1.7B-Base | 211 | other | exp-05 |  | completed | 0.0 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-07 | Qwen/Qwen3-1.7B-Base | 229 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-bc465eaf | exp-08 | Qwen/Qwen3-1.7B-Base | 237 | other | exp-07 |  | completed |  | inconclusive | adopt | False | 0.12282031842304776 |
| train | r-bcc8974e | exp-01 | Qwen/Qwen3-1.7B-Base | 34 | sft | base_model | HF gsm8k (config main), train split only | failed |  | inconclusive | abandon_line | False |  |
| train | r-bcc8974e | exp-02 | Qwen/Qwen3-1.7B-Base | 44 | sft | base_model | HF gsm8k (config main), train split only | killed |  | inconclusive | abandon_line | False |  |
| train | r-bcc8974e | exp-03 | Qwen/Qwen3-1.7B-Base | 48 | sft | base_model | HF gsm8k (config main), train split only | completed | 0.14 | inconclusive | adopt | False |  |
| train | r-bcc8974e | exp-04 | Qwen/Qwen3-1.7B-Base | 56 | other | exp-03 |  | completed |  | inconclusive | adopt | False | 0.10917361637604246 |
| train | r-bda2d6c1 | exp-01 | Qwen/Qwen3-1.7B-Base | 65 | sft | base_model | gsm8k/main:train | completed | 0.66 | inconclusive | adopt | False |  |
| train | r-bda2d6c1 | exp-02 | Qwen/Qwen3-1.7B-Base | 194 | sft | base_model | gsm8k/main:train,meta-math/MetaMathQA:train | completed |  | inconclusive | reject | False |  |
| train | r-bda2d6c1 | exp-03 | Qwen/Qwen3-1.7B-Base | 220 | sft | base_model | gsm8k/main:train | completed | 0.053 | contradicted | reject | False |  |
| train | r-bda2d6c1 | exp-04 | Qwen/Qwen3-1.7B-Base | 252 | sft | base_model | gsm8k/main:train,derived:exp-01 | completed | 0.693 | supported | adopt | False |  |
| train | r-bda2d6c1 | exp-05 | Qwen/Qwen3-1.7B-Base | 269 | sft | base_model | gsm8k/main:train,derived:exp-04 | completed | 0.387 | contradicted | reject | False |  |
| train | r-bda2d6c1 | exp-06 | Qwen/Qwen3-1.7B-Base | 283 | sft | base_model | gsm8k/main:train,None | completed | 0.587 | contradicted | reject | False |  |
| train | r-bda2d6c1 | exp-07 | Qwen/Qwen3-1.7B-Base | 293 | sft | base_model | gsm8k/main:train,None | completed | 0.7066666666666667 | supported | adopt | False |  |
| train | r-bda2d6c1 | exp-08 | Qwen/Qwen3-1.7B-Base | 306 | sft | base_model | gsm8k/main:train,None | completed | 0.6133333333333333 | contradicted | reject | False |  |
| train | r-bda2d6c1 | exp-09 | Qwen/Qwen3-1.7B-Base | 373 | sft | base_model | gsm8k/main:train,None | completed | 0.54 | contradicted | reject | False |  |
| train | r-bda2d6c1 | exp-10 | Qwen/Qwen3-1.7B-Base | 415 | sft | base_model | gsm8k/main:train,None | completed |  | inconclusive | reject | False |  |
| train | r-bda2d6c1 | exp-11 | Qwen/Qwen3-1.7B-Base | 450 | sft | base_model | gsm8k/main:train,None | completed | 0.7267 | supported | adopt | False |  |
| train | r-bda2d6c1 | exp-12 | Qwen/Qwen3-1.7B-Base | 480 | sft | base_model | gsm8k/main:train,None | completed |  | inconclusive | reject | False |  |
| train | r-bda2d6c1 | exp-13 | Qwen/Qwen3-1.7B-Base | 499 | sft | base_model | gsm8k/main:train,None | completed | 0.7333 | inconclusive | adopt | False |  |
| train | r-bda2d6c1 | exp-14 | Qwen/Qwen3-1.7B-Base | 550 | sft | base_model | gsm8k/main:train,None | completed |  | inconclusive | reject | False |  |
| train | r-bda2d6c1 | exp-15 | Qwen/Qwen3-1.7B-Base | 596 | sft | base_model | gsm8k/main:train,derived:exp-07 | completed | 0.726 | supported | adopt | False | 0.7217589082638363 |
| train | r-bfd319db | exp-01 | Qwen/Qwen3-1.7B-Base | 113 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.02 | inconclusive | reject | True |  |
| train | r-bfd319db | exp-02 | Qwen/Qwen3-1.7B-Base | 131 | merge | exp-01 |  | completed | 0.02 | inconclusive | reject | False |  |
| train | r-bfd319db | exp-03 | Qwen/Qwen3-1.7B-Base | 148 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.075 | contradicted | adopt | False |  |
| train | r-bfd319db | exp-04 | Qwen/Qwen3-1.7B-Base | 154 | merge | exp-03 |  | completed | 0.075 | contradicted | adopt | False |  |
| train | r-bfd319db | exp-05 | Qwen/Qwen3-1.7B-Base | 165 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.05 | contradicted | reject | True |  |
| train | r-bfd319db | exp-06 | Qwen/Qwen3-1.7B-Base | 178 | merge | exp-05 |  | completed | 0.05 | contradicted | reject | False |  |
| train | r-bfd319db | exp-07 | Qwen/Qwen3-1.7B-Base | 187 | other | exp-04 |  | completed | 0.075 | supported | adopt | False | 0.08188021228203184 |
| train | r-c0173ea9 | exp-01 | Qwen/Qwen3-4B-Base | 59 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-02 | Qwen/Qwen3-4B-Base | 74 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-03 | Qwen/Qwen3-4B-Base | 83 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-04 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-05 | Qwen/Qwen3-4B-Base | 98 | sft | base_model | HF:openai/gsm8k (config main, split train) | killed |  | inconclusive | adopt | False |  |
| train | r-c0173ea9 | exp-06 | Qwen/Qwen3-4B-Base | 137 | sft | exp-05 | HF:openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-c0173ea9 | exp-07 | Qwen/Qwen3-4B-Base | 156 | merge | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-c0173ea9 | exp-08 | Qwen/Qwen3-4B-Base | 161 | other | exp-07 |  | completed |  | inconclusive | adopt | False | 0.0758150113722517 |
| train | r-c1110c15 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 1358 | distill |  | synthetic:Qwen/Qwen2.5-Math-7B-Instruct | completed | 0.78 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 1378 | distill |  | synthetic:Qwen/Qwen2.5-Math-7B-Instruct | completed | 0.79 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 1411 | distill |  | derived:teacher+self-train | completed | 0.81 | supported | adopt | True |  |
| train | r-c1110c15 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 1429 | other | exp-03 |  | completed | 0.7 | inconclusive | adopt | True | 0.733131159969674 |
| train | r-c1110c15 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 1511 | distill | exp-04 | derived:teacher+self-train | completed | 0.72 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 1529 | distill | exp-04 | derived:teacher+self-train | completed | 0.78 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 1545 | distill | exp-04 | derived:teacher+self-train | completed | 0.82 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1598 | sft | exp-04 | derived:openai/gsm8k gold + self-train | completed | 0.69 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 1614 | distill | exp-04 | derived:teacher+self-train | completed | 0.77 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 1646 | distill | exp-04 | derived:teacher+self-train | completed | 0.7 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 1669 | distill | exp-04 | derived:teacher+self-train | completed | 0.71 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 1686 | merge | exp-04 |  | completed | 0.7 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 1714 | decode-config | exp-04 |  | completed | 0.32 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 1757 | dpo | exp-04 | synthetic:self | failed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 1763 | dpo | exp-04 | synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-16 | HuggingFaceTB/SmolLM3-3B-Base | 1774 | dpo | exp-04 | synthetic:self | completed | 0.71 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-17 | HuggingFaceTB/SmolLM3-3B-Base | 1888 | distill |  | derived:teacher+self-train | completed | 0.68 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-18 | HuggingFaceTB/SmolLM3-3B-Base | 1899 | distill |  | derived:teacher+self-train | completed | 0.71 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-19 | HuggingFaceTB/SmolLM3-3B-Base | 1928 | distill |  | derived:teacher+self-train | completed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-20 | HuggingFaceTB/SmolLM3-3B-Base | 1940 | distill |  | derived:teacher+self-train | completed | 0.72 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-21 | HuggingFaceTB/SmolLM3-3B-Base | 1987 | decode-config | exp-04 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-22 | HuggingFaceTB/SmolLM3-3B-Base | 2009 | distill |  | derived:teacher+self-train | completed | 0.71 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-23 | HuggingFaceTB/SmolLM3-3B-Base | 2016 | distill |  | derived:teacher+self-train | killed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-24 | HuggingFaceTB/SmolLM3-3B-Base | 2038 | merge | exp-04 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-25 | HuggingFaceTB/SmolLM3-3B-Base | 2039 | merge | exp-04 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-26 | HuggingFaceTB/SmolLM3-3B-Base | 2062 | merge | exp-04 |  | completed | 0.74 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-27 | HuggingFaceTB/SmolLM3-3B-Base | 2068 | distill |  | derived:teacher+self-train | completed | 0.72 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-28 | HuggingFaceTB/SmolLM3-3B-Base | 2088 | merge | exp-04 |  | completed | 0.77 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-29 | HuggingFaceTB/SmolLM3-3B-Base | 2089 | merge | exp-04 |  | completed | 0.68 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-30 | HuggingFaceTB/SmolLM3-3B-Base | 2211 | distill |  | derived:teacher+self-train | completed |  | inconclusive | abandon_line | True |  |
| train | r-c1110c15 | exp-31 | HuggingFaceTB/SmolLM3-3B-Base | 2221 | distill |  | derived:teacher+self-train | completed | 0.77 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-32 | HuggingFaceTB/SmolLM3-3B-Base | 2305 | sft | exp-04 | derived:openai/gsm8k gold + synthetic:self | completed | 0.82 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-33 | HuggingFaceTB/SmolLM3-3B-Base | 2322 | sft | exp-04 | derived:openai/gsm8k gold + synthetic:self | completed | 0.77 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-34 | HuggingFaceTB/SmolLM3-3B-Base | 2336 | sft | exp-04 | derived:teacher+self-train+gold | completed | 0.75 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-35 | HuggingFaceTB/SmolLM3-3B-Base | 2355 | merge | exp-04 |  | completed | 0.7 | contradicted | reject | True |  |
| train | r-c1110c15 | exp-36 | HuggingFaceTB/SmolLM3-3B-Base | 2393 | distill |  | derived:teacher+self-train+gold | completed | 0.72 | contradicted | reject | True |  |
| train | r-c2a5a7bb | exp-01 | Qwen/Qwen3-4B-Base | 48 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | failed |  | inconclusive | iterate | True |  |
| train | r-c2a5a7bb | exp-02 | Qwen/Qwen3-4B-Base | 72 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | completed | 0.06 | inconclusive | reject | True |  |
| train | r-c2a5a7bb | exp-03 | Qwen/Qwen3-4B-Base | 139 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | killed |  | inconclusive | abandon_line | True |  |
| train | r-c2a5a7bb | exp-04 | Qwen/Qwen3-4B-Base | 165 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | completed | 0.04 | contradicted | reject | True |  |
| train | r-c2a5a7bb | exp-05 | Qwen/Qwen3-4B-Base | 197 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | failed |  | inconclusive | iterate | True |  |
| train | r-c2a5a7bb | exp-06 | Qwen/Qwen3-4B-Base | 202 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | completed | 0.6 | supported | adopt | True |  |
| train | r-c2a5a7bb | exp-07 | Qwen/Qwen3-4B-Base | 227 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | killed |  | inconclusive | iterate | True |  |
| train | r-c2a5a7bb | exp-08 | Qwen/Qwen3-4B-Base | 240 | sft | exp-06 | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | completed | 0.64 | inconclusive | adopt | True |  |
| train | r-c2a5a7bb | exp-09 | Qwen/Qwen3-4B-Base | 254 | other | exp-08 |  | completed | 0.66 | inconclusive | adopt | False |  |
| train | r-c2a5a7bb | exp-10 | Qwen/Qwen3-4B-Base | 284 | sft | exp-08 | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | completed | 0.7 | inconclusive | adopt | True |  |
| train | r-c2a5a7bb | exp-11 | Qwen/Qwen3-4B-Base | 303 | other | exp-10 |  | completed |  | inconclusive | adopt | False | 0.6611068991660348 |
| train | r-c2a5a7bb | exp-12 | Qwen/Qwen3-4B-Base | 341 | sft | exp-10 | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (type contains GSM) | killed |  | inconclusive | abandon_line | True |  |
| train | r-c3036179 | exp-01 | Qwen/Qwen3-1.7B-Base | 66 | sft | base_model | HF openai/gsm8k (config main, split train),synthetic:self (template generator inside the training script) | completed | 0.16 | inconclusive | reject | True |  |
| train | r-c3036179 | exp-02 | Qwen/Qwen3-1.7B-Base | 125 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.4266666666666667 | supported | adopt | True |  |
| train | r-c3036179 | exp-03 | Qwen/Qwen3-1.7B-Base | 164 | sft | exp-02 | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.4533333333333333 | supported | adopt | True |  |
| train | r-c3036179 | exp-04 | Qwen/Qwen3-1.7B-Base | 230 | sft | exp-03 | HF openai/gsm8k (config main, split train), loaded in-process by the training script,synthetic:self (gsmish template generator inside the training script) | completed | 0.44666666666666666 | contradicted | reject | True |  |
| train | r-c3036179 | exp-05 | Qwen/Qwen3-1.7B-Base | 274 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6853677028051555 | supported | adopt | False |  |
| train | r-c3036179 | exp-06 | Qwen/Qwen3-1.7B-Base | 317 | sft | exp-05 | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6133333333333333 | supported | adopt | False |  |
| train | r-c3036179 | exp-07 | Qwen/Qwen3-1.7B-Base | 362 | sft | exp-06 | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.54 | supported | adopt | True |  |
| train | r-c3036179 | exp-08 | Qwen/Qwen3-1.7B-Base | 411 | sft | exp-07 | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6266666666666667 | contradicted | reject | True |  |
| train | r-c3036179 | exp-09 | Qwen/Qwen3-1.7B-Base | 457 | sft | exp-07 | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.5 | contradicted | reject | True |  |
| train | r-c3036179 | exp-10 | Qwen/Qwen3-1.7B-Base | 490 | other | exp-07 |  | completed | 0.56 | inconclusive | adopt | True |  |
| train | r-c3036179 | exp-11 | Qwen/Qwen3-1.7B-Base | 500 | decode-config | exp-10 |  | completed | 0.6333333333333333 | supported | adopt | True |  |
| train | r-c3036179 | exp-12 | Qwen/Qwen3-1.7B-Base | 507 | decode-config | exp-11 |  | completed | 0.4866666666666667 | contradicted | reject | True |  |
| train | r-c3036179 | exp-13 | Qwen/Qwen3-1.7B-Base | 513 | decode-config | exp-12 |  | completed | 0.6466666666666666 | supported | adopt | False |  |
| train | r-c3036179 | exp-14 | Qwen/Qwen3-1.7B-Base | 554 | other | exp-05 |  | completed | 0.6466666666666666 | supported | adopt | True |  |
| train | r-c3036179 | exp-15 | Qwen/Qwen3-1.7B-Base | 599 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | False |  |
| train | r-c3036179 | exp-16 | Qwen/Qwen3-1.7B-Base | 612 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.7028051554207733 | supported | reject | True |  |
| train | r-c3036179 | exp-17 | Qwen/Qwen3-1.7B-Base | 661 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6733333333333333 | contradicted | reject | True |  |
| train | r-c3036179 | exp-18 | Qwen/Qwen3-1.7B-Base | 737 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6846095526914329 | contradicted | reject | True |  |
| train | r-c3036179 | exp-19 | Qwen/Qwen3-1.7B-Base | 754 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6868840030326004 | contradicted | reject | True |  |
| train | r-c3036179 | exp-20 | Qwen/Qwen3-1.7B-Base | 800 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6266666666666667 | contradicted | reject | True |  |
| train | r-c3036179 | exp-21 | Qwen/Qwen3-1.7B-Base | 832 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.7050796057619408 | supported | adopt | True |  |
| train | r-c3036179 | exp-22 | Qwen/Qwen3-1.7B-Base | 858 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.7005307050796058 | contradicted | reject | True |  |
| train | r-c3036179 | exp-23 | Qwen/Qwen3-1.7B-Base | 882 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6466666666666666 | contradicted | reject | True |  |
| train | r-c3036179 | exp-24 | Qwen/Qwen3-1.7B-Base | 924 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.68 | inconclusive | reject | True |  |
| train | r-c3036179 | exp-25 | Qwen/Qwen3-1.7B-Base | 947 | other | exp-21 |  | completed | 0.6959818043972706 | supported | adopt | True | 0.6990144048521607 |
| train | r-c3036179 | exp-26 | Qwen/Qwen3-1.7B-Base | 986 | decode-config | exp-25 |  | completed | 0.7005307050796058 | supported | reject | True |  |
| train | r-c51755a7 | exp-01 | Qwen/Qwen3-1.7B-Base | 92 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k | completed | 0.18 | inconclusive | reject | False |  |
| train | r-c51755a7 | exp-02 | Qwen/Qwen3-1.7B-Base | 138 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.46 | supported | reject | False |  |
| train | r-c51755a7 | exp-03 | Qwen/Qwen3-1.7B-Base | 160 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k,openai/gsm8k (config main, split train) | completed | 0.367 | contradicted | adopt | False |  |
| train | r-c51755a7 | exp-04 | Qwen/Qwen3-1.7B-Base | 186 | decode-config | exp-03 |  | completed | 0.447 | supported | reject | False |  |
| train | r-c51755a7 | exp-05 | Qwen/Qwen3-1.7B-Base | 204 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.48 | supported | reject | False |  |
| train | r-c51755a7 | exp-06 | Qwen/Qwen3-1.7B-Base | 220 | sft | base_model | AI-MO/NuminaMath-CoT,openai/gsm8k (config main, split train) | completed | 0.52 | supported | adopt | False |  |
| train | r-c51755a7 | exp-07 | Qwen/Qwen3-1.7B-Base | 230 | other | exp-06 |  | completed |  | inconclusive | adopt | False | 0.5435936315390447 |
| train | r-c7ff2a60 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 97 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | failed |  | inconclusive | iterate | True |  |
| train | r-c7ff2a60 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 105 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | killed |  | inconclusive | adopt | True |  |
| train | r-c7ff2a60 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 218 | other | exp-02 |  | completed | 0.167 | inconclusive | reject | False |  |
| train | r-c7ff2a60 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 241 | sft | exp-02 | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.22 | inconclusive | adopt | True |  |
| train | r-c7ff2a60 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 284 | decode-config | exp-04 |  | completed | 0.14 | contradicted | reject | False |  |
| train | r-c7ff2a60 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 310 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.28 | inconclusive | reject | True |  |
| train | r-c7ff2a60 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 330 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.413 | supported | adopt | True | 0.3813495072024261 |
| train | r-cb45aad0 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 75 | sft | base_model | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-cb45aad0 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 115 | sft | base_model | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | killed |  | inconclusive | adopt | True |  |
| train | r-cb45aad0 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 220 | sft | exp-02 | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | completed | 0.44 | inconclusive | adopt | True |  |
| train | r-cb45aad0 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 248 | other | exp-03 |  | completed | 0.44 | inconclusive | adopt | False |  |
| train | r-cb45aad0 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 281 | grpo | exp-03 | HF: openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| train | r-cb45aad0 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 308 | sft | base_model | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | killed | 0.44 | contradicted | reject | True |  |
| train | r-cb45aad0 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 475 | sft | base_model | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | killed | 0.44 | contradicted | reject | True |  |
| train | r-cb45aad0 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 519 | decode-config | exp-04 |  | completed | 0.56 | supported | adopt | True |  |
| train | r-cb45aad0 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 562 | sft | exp-08 | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | completed | 0.6 | supported | adopt | True |  |
| train | r-cb45aad0 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 594 | other | exp-09 |  | completed | 0.6 | inconclusive | adopt | False | 0.5830174374526156 |
| train | r-cb45aad0 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 602 | sft | exp-09 | derived: meta-math/MetaMathQA + openai/gsm8k (main, train) | completed | 0.587 | contradicted | reject | True |  |
| train | r-cb45aad0 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 637 | sft | exp-09 | derived: openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-cbcfc798 | exp-01 | Qwen/Qwen3-1.7B-Base | 60 | sft | base_model | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_SV, GSM_FOBAR, GSM_AnsAug),HF:meta-math/MetaMathQA (types MATH_AnsAug, MATH_Rephrased) | failed |  | inconclusive | iterate | False |  |
| train | r-cbcfc798 | exp-02 | Qwen/Qwen3-1.7B-Base | 69 | sft | base_model | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_SV, GSM_FOBAR, GSM_AnsAug),HF:meta-math/MetaMathQA (types MATH_AnsAug, MATH_Rephrased) | killed |  | inconclusive | adopt | False |  |
| train | r-cbcfc798 | exp-03 | Qwen/Qwen3-1.7B-Base | 119 | merge | exp-02 |  | completed | 0.14 | inconclusive | adopt | False |  |
| train | r-cbcfc798 | exp-04 | Qwen/Qwen3-1.7B-Base | 142 | decode-config | exp-03 |  | completed | 0.1 | inconclusive | reject | False |  |
| train | r-cbcfc798 | exp-05 | Qwen/Qwen3-1.7B-Base | 151 | decode-config | exp-03 |  | completed | 0.22 | inconclusive | adopt | False |  |
| train | r-cbcfc798 | exp-06 | Qwen/Qwen3-1.7B-Base | 160 | sft | base_model | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug only) | completed | 0.707 | supported | adopt | True |  |
| train | r-cbcfc798 | exp-07 | Qwen/Qwen3-1.7B-Base | 241 | sft | base_model | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed | 0.06 | contradicted | reject | True |  |
| train | r-cbcfc798 | exp-08 | Qwen/Qwen3-1.7B-Base | 282 | other | exp-06 |  | completed | 0.18 | inconclusive | adopt | False |  |
| train | r-cbcfc798 | exp-09 | Qwen/Qwen3-1.7B-Base | 293 | sft | exp-06 | HF:openai/gsm8k (main, split=train) | completed | 0.153 | contradicted | reject | True |  |
| train | r-cbcfc798 | exp-10 | Qwen/Qwen3-1.7B-Base | 310 | other | exp-06 |  | completed | 0.28 | inconclusive | adopt | False |  |
| train | r-cbcfc798 | exp-11 | Qwen/Qwen3-1.7B-Base | 326 | decode-config | exp-10 |  | completed | 0.727 | supported | adopt | True | 0.7119029567854435 |
| train | r-cbcfc798 | exp-12 | Qwen/Qwen3-1.7B-Base | 339 | sft | exp-06 | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-cbcfc798 | exp-13 | Qwen/Qwen3-1.7B-Base | 370 | sft | exp-06 | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-ce67516f | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 54 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process by the training script | completed | 0.1 | inconclusive | adopt | True |  |
| train | r-ce67516f | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 104 | decode-config | exp-01 |  | completed | 0.6 | supported | adopt | True |  |
| train | r-ce67516f | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 110 | sft | exp-02 | HF:openai/gsm8k (main, train split), loaded in-process,HF:meta-math/MetaMathQA (train, streamed), loaded in-process | completed | 0.5933333333333334 | inconclusive | adopt | True |  |
| train | r-ce67516f | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 153 | sft | exp-03 | HF:openai/gsm8k (main, train split), loaded in-process | completed | 0.6066666666666667 | inconclusive | adopt | True |  |
| train | r-ce67516f | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 196 | sft | exp-04 | HF:openai/gsm8k (main, train split), loaded in-process | completed | 0.5666666666666667 | contradicted | reject | True |  |
| train | r-ce67516f | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 226 | sft | base_model | HF:openai/gsm8k (main, train split), loaded in-process,HF:meta-math/MetaMathQA (train, streamed), loaded in-process | completed | 0.5933333333333334 | inconclusive | adopt | True |  |
| train | r-ce67516f | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 304 | sft | exp-06 | HF:openai/gsm8k (main, train split), loaded in-process | completed | 0.47333333333333333 | contradicted | reject | True |  |
| train | r-ce67516f | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 358 | merge | exp-04 |  | completed | 0.5933333333333334 | inconclusive | reject | True |  |
| train | r-ce67516f | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 367 | merge | exp-04 |  | completed | 0.5666666666666667 | contradicted | reject | True |  |
| train | r-ce67516f | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 405 | decode-config | exp-04 |  | completed | 0.7 | supported | reject | True |  |
| train | r-ce67516f | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 412 | decode-config | exp-02 |  | completed | 0.7133333333333334 | inconclusive | reject | True |  |
| train | r-ce67516f | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 418 | decode-config | exp-03 |  | completed | 0.7467778620166793 | supported | reject | True |  |
| train | r-ce67516f | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 448 | sft | exp-02 | HF:openai/gsm8k (main, train split), loaded in-process,HF:meta-math/MetaMathQA (train, streamed), loaded in-process | completed | 0.755117513267627 | supported | adopt | True |  |
| train | r-ce67516f | exp-14 | HuggingFaceTB/SmolLM3-3B-Base | 597 | other | exp-13 |  | completed | 0.7573919636087946 | supported | adopt | True | 0.7573919636087946 |
| train | r-ce67516f | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 623 | sft | exp-14 | HF:openai/gsm8k (main, train split), loaded in-process,HF:meta-math/MetaMathQA (train, streamed), loaded in-process | failed |  | inconclusive | abandon_line | True |  |
| train | r-cf5932d6 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 163 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus the openai/gsm8k train split | killed |  | inconclusive | abandon_line | False |  |
| train | r-cf5932d6 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 244 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus the openai/gsm8k train split | completed | 0.56 | inconclusive | adopt | False |  |
| train | r-cf5932d6 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 321 | sft | exp-02 | None | completed | 0.5533 | inconclusive | reject | False |  |
| train | r-cf5932d6 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 330 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-cf5932d6 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 379 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus meta-math/MetaMathQA (MetaMathQA-395K.json, GSM_* types only), plus the openai/gsm8k train split | completed | 0.5867 | inconclusive | adopt | False |  |
| train | r-cf5932d6 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 449 | other | exp-05 |  | completed | 0.5733 | inconclusive | adopt | False | 0.5822592873388931 |
| train | r-cfdd8de7 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 143 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA (GSM-typed rows) | completed | 0.707 | inconclusive | adopt | False |  |
| train | r-cfdd8de7 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 202 | decode-config | exp-01 |  | completed | 0.533 | supported | adopt | False |  |
| train | r-cfdd8de7 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 210 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-cfdd8de7 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 222 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA (GSM-typed rows) + microsoft/orca-math-word-problems-200k | completed | 0.747 | inconclusive | adopt | False |  |
| train | r-cfdd8de7 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 258 | other | exp-04 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-cfdd8de7 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 286 | rft | exp-04 | derived:exp-04 (synthetic:self) + openai/gsm8k train + meta-math/MetaMathQA (GSM-typed rows) | completed | 0.555 | inconclusive | adopt | False |  |
| train | r-cfdd8de7 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 326 | decode-config | exp-06 |  | completed | 0.767 | supported | adopt | False |  |
| train | r-cfdd8de7 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 345 | other | exp-07 |  | completed | 0.76 | inconclusive | reject | False |  |
| train | r-cfdd8de7 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 364 | rft | exp-07 | derived:exp-07 (synthetic:self) + openai/gsm8k train | killed |  | inconclusive | abandon_line | False |  |
| train | r-cfdd8de7 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 390 | rft | exp-07 | derived:exp-07 (synthetic:self) + openai/gsm8k train | completed | 0.748 | inconclusive | adopt | False |  |
| train | r-cfdd8de7 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 423 | other | exp-10 |  | completed | 0.743 | inconclusive | adopt | False | 0.7384382107657316 |
| train | r-d0634645 | exp-01 | Qwen/Qwen3-4B-Base | 68 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | True |  |
| train | r-d0634645 | exp-02 | Qwen/Qwen3-4B-Base | 113 | sft | base_model | openai/gsm8k (train) + meta-math/MetaMathQA | completed | 0.08 | inconclusive | adopt | True |  |
| train | r-d0634645 | exp-03 | Qwen/Qwen3-4B-Base | 217 | decode-config | exp-02 |  | completed | 0.62 | supported | adopt | True |  |
| train | r-d0634645 | exp-04 | Qwen/Qwen3-4B-Base | 230 | sft | exp-03 | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.82 | supported | adopt | True |  |
| train | r-d0634645 | exp-05 | Qwen/Qwen3-4B-Base | 289 | grpo | exp-04 | derived:exp-04 (prompts only, openai/gsm8k train questions) | failed |  | inconclusive | abandon_line | True |  |
| train | r-d0634645 | exp-06 | Qwen/Qwen3-4B-Base | 299 | grpo | exp-04 | derived:exp-04 (prompts only, openai/gsm8k train questions) | killed |  | inconclusive | abandon_line | True |  |
| train | r-d0634645 | exp-07 | Qwen/Qwen3-4B-Base | 308 | grpo | exp-04 | derived:exp-04 (prompts only, openai/gsm8k train questions) | killed |  | inconclusive | abandon_line | False |  |
| train | r-d0634645 | exp-08 | Qwen/Qwen3-4B-Base | 325 | grpo | exp-04 | derived:exp-04 (prompts only, openai/gsm8k train questions) | failed |  | inconclusive | abandon_line | True |  |
| train | r-d0634645 | exp-09 | Qwen/Qwen3-4B-Base | 351 | grpo | exp-04 | derived:exp-04 (prompts only, openai/gsm8k train questions) | completed | 0.6533333333333333 | contradicted | reject | True |  |
| train | r-d0634645 | exp-10 | Qwen/Qwen3-4B-Base | 399 | sft | exp-04 | openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.7066666666666667 | inconclusive | reject | True |  |
| train | r-d0634645 | exp-11 | Qwen/Qwen3-4B-Base | 466 | rft | exp-04 | synthetic:self (samples from exp-04's checkpoint) + openai/gsm8k gold | completed | 0.7275 | supported | adopt | True |  |
| train | r-d0634645 | exp-12 | Qwen/Qwen3-4B-Base | 501 | sft | exp-04 | openai/gsm8k (train) | failed | 0.6666666666666666 | contradicted | reject | False |  |
| train | r-d0634645 | exp-13 | Qwen/Qwen3-4B-Base | 536 | sft | exp-04 | openai/gsm8k (train) + EleutherAI/asdiv + ChilleD/SVAMP + synthetic:self (the exp-11 RFT samples) | failed |  | inconclusive | abandon_line | True |  |
| train | r-d0634645 | exp-14 | Qwen/Qwen3-4B-Base | 547 | sft | exp-04 | openai/gsm8k (train) + EleutherAI/asdiv + ChilleD/SVAMP + synthetic:self (the exp-11 RFT samples) | completed | 0.66 | inconclusive | reject | True |  |
| train | r-d0634645 | exp-15 | Qwen/Qwen3-4B-Base | 557 | other | exp-04 |  | completed |  | contradicted | reject | True |  |
| train | r-d0634645 | exp-16 | Qwen/Qwen3-4B-Base | 567 | other | exp-11 |  | completed | 0.708 | supported | reject | True |  |
| train | r-d0634645 | exp-17 | Qwen/Qwen3-4B-Base | 573 | grpo | exp-11 | derived:exp-11 (prompts only, openai/gsm8k train questions) | completed | 0.7525 | supported | adopt | True |  |
| train | r-d0634645 | exp-18 | Qwen/Qwen3-4B-Base | 596 | other | exp-17 |  | completed | 0.78 | supported | adopt | True | 0.7278241091736164 |
| train | r-d0defd39 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 73 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed | 0.0 | inconclusive | reject | False |  |
| train | r-d0defd39 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 105 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed | 0.04 | contradicted | reject | True |  |
| train | r-d0defd39 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 120 | sft | base_model | HF id: openai/gsm8k (config main, split train),HF id: nvidia/OpenMathInstruct-2 (split train) | completed | 0.16 | inconclusive | adopt | False | 0.10538286580742987 |
| train | r-d0defd39 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 138 | other | exp-03 |  | completed | 0.16 | inconclusive | reject | False |  |
| train | r-d0defd39 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | HF id: openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-d2611dd8 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 121 | sft | base_model | HF openai/gsm8k (train) + HF meta-math/MetaMathQA (GSM_* subsets) | completed | 0.54 | inconclusive | reject | False |  |
| train | r-d2611dd8 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 174 | sft | base_model | HF openai/gsm8k (train) + HF meta-math/MetaMathQA (GSM_* subsets) | completed | 0.58 | inconclusive | adopt | False |  |
| train | r-d2611dd8 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 224 | rft | exp-02 | synthetic:self (exp-02 samples, kept when the final answer matches gsm8k train gold) + HF meta-math/MetaMathQA + HF openai/gsm8k | completed | 0.716 | supported | adopt | False |  |
| train | r-d2611dd8 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 259 | rft | exp-03 | synthetic:self (exp-03 samples, answer-filtered) + HF meta-math/MetaMathQA + HF openai/gsm8k | completed |  | inconclusive | reject | False |  |
| train | r-d2611dd8 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 286 | rft | base_model | synthetic:self (exp-03 samples, answer-filtered) + HF openai/gsm8k (train) + HF meta-math/MetaMathQA | completed | 0.596 | contradicted | reject | False |  |
| train | r-d2611dd8 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 305 | other | exp-03 |  | completed | 0.692 | inconclusive | adopt | False | 0.6626231993934799 |
| train | r-d2611dd8 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 318 | rft | exp-03 | synthetic:self (exp-02 and exp-03 samples, answer-filtered) + HF openai/gsm8k (train) | completed | 0.68 | contradicted | reject | False |  |
| train | r-d42c24e0 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 99 | sft | base_model | local | completed | 0.06 | inconclusive | adopt | True |  |
| train | r-d42c24e0 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 198 | sft | exp-01 | local | completed | 0.5867 | supported | adopt | True |  |
| train | r-d42c24e0 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 287 | sft | base_model | local | completed | 0.42 | inconclusive | reject | False |  |
| train | r-d42c24e0 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 293 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-d42c24e0 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 350 | sft | exp-02 | local | completed | 0.6 | inconclusive | adopt | False |  |
| train | r-d42c24e0 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 409 | sft | exp-05 | local | completed | 0.5133 | contradicted | reject | True |  |
| train | r-d42c24e0 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 460 | sft | exp-05 | local | completed | 0.6067 | inconclusive | adopt | True |  |
| train | r-d42c24e0 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 475 | other | exp-05 |  | completed |  | inconclusive | adopt | False |  |
| train | r-d42c24e0 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 525 | rft | exp-07 | synthetic:self | completed | 0.6067 | inconclusive | adopt | True |  |
| train | r-d42c24e0 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 572 | merge |  |  | completed | 0.6333 | inconclusive | adopt | False | 0.6178923426838514 |
| train | r-d42c24e0 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 585 | rft |  | synthetic:self | failed |  | inconclusive | abandon_line | False |  |
| train | r-d42c24e0 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 610 | rft | exp-10 | synthetic:self | completed | 0.62 | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-01 | Qwen/Qwen3-1.7B-Base | 66 | sft | base_model | openai/gsm8k | failed |  | inconclusive | iterate | False |  |
| train | r-d4a6c52f | exp-02 | Qwen/Qwen3-1.7B-Base | 72 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | False |  |
| train | r-d4a6c52f | exp-03 | Qwen/Qwen3-1.7B-Base | 109 | merge | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-04 | Qwen/Qwen3-1.7B-Base | 112 | merge | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-d4a6c52f | exp-05 | Qwen/Qwen3-1.7B-Base | 173 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-06 | Qwen/Qwen3-1.7B-Base | 247 | merge | exp-05 |  | completed |  | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-07 | Qwen/Qwen3-1.7B-Base | 248 | merge | exp-05 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-d4a6c52f | exp-08 | Qwen/Qwen3-1.7B-Base | 313 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-09 | Qwen/Qwen3-1.7B-Base | 328 | merge | exp-08 |  | completed | 0.15 | supported | adopt | False |  |
| train | r-d4a6c52f | exp-10 | Qwen/Qwen3-1.7B-Base | 338 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-11 | Qwen/Qwen3-1.7B-Base | 352 | merge | exp-10 |  | completed |  | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-12 | Qwen/Qwen3-1.7B-Base | 374 | other | exp-09 |  | completed | 0.175 | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-13 | Qwen/Qwen3-1.7B-Base | 451 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-14 | Qwen/Qwen3-1.7B-Base | 461 | merge | exp-13 |  | completed | 0.087 | inconclusive | reject | False |  |
| train | r-d4a6c52f | exp-15 | Qwen/Qwen3-1.7B-Base | 469 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| train | r-d4a6c52f | exp-16 | Qwen/Qwen3-1.7B-Base | 528 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-17 | Qwen/Qwen3-1.7B-Base | 536 | merge | exp-16 |  | completed | 0.158 | supported | reject | False |  |
| train | r-d4a6c52f | exp-18 | Qwen/Qwen3-1.7B-Base | 551 | merge | exp-16 |  | completed | 0.1933 | supported | adopt | True |  |
| train | r-d4a6c52f | exp-19 | Qwen/Qwen3-1.7B-Base | 557 | sft | base_model | openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-d4a6c52f | exp-20 | Qwen/Qwen3-1.7B-Base | 562 | merge | exp-19 |  | completed | 0.125 | contradicted | reject | False |  |
| train | r-d4a6c52f | exp-21 | Qwen/Qwen3-1.7B-Base | 605 | other | exp-18 |  | completed | 0.1933 | supported | adopt | False | 0.1645185746777862 |
| train | r-d7327838 | exp-01 | Qwen/Qwen3-1.7B-Base | 65 | sft | base_model | HF:openai/gsm8k main split=train | completed | 0.0267 | inconclusive | adopt | True |  |
| train | r-d7327838 | exp-02 | Qwen/Qwen3-1.7B-Base | 136 | decode-config | exp-01 |  | completed | 0.0333 | contradicted | reject | True |  |
| train | r-d7327838 | exp-03 | Qwen/Qwen3-1.7B-Base | 169 | sft | base_model | HF:openai/gsm8k main split=train,local | completed | 0.3333 | supported | reject | True |  |
| train | r-d7327838 | exp-04 | Qwen/Qwen3-1.7B-Base | 224 | sft | base_model | HF:meta-math/MetaMathQA + HF:openai/gsm8k main split=train | completed | 0.5333 | supported | adopt | True |  |
| train | r-d7327838 | exp-05 | Qwen/Qwen3-1.7B-Base | 288 | sft | base_model | HF:meta-math/MetaMathQA + HF:openai/gsm8k main split=train | completed | 0.6267 | supported | adopt | True |  |
| train | r-d7327838 | exp-06 | Qwen/Qwen3-1.7B-Base | 301 | other | exp-04 |  | completed |  | inconclusive | adopt | False |  |
| train | r-d7327838 | exp-07 | Qwen/Qwen3-1.7B-Base | 324 | other | exp-05 |  | completed |  | inconclusive | adopt | False |  |
| train | r-d7327838 | exp-08 | Qwen/Qwen3-1.7B-Base | 351 | rft | base_model | derived:exp-05 (self-sampled) + HF:meta-math/MetaMathQA + HF:openai/gsm8k main split=train,synthetic:self (sampled from the exp-05 checkpoint) | completed | 0.6667 | supported | adopt | True |  |
| train | r-d7327838 | exp-09 | Qwen/Qwen3-1.7B-Base | 390 | other | exp-08 |  | completed |  | inconclusive | adopt | False |  |
| train | r-d7327838 | exp-10 | Qwen/Qwen3-1.7B-Base | 407 | rft | base_model | derived:exp-08 and derived:exp-05 (self-sampled) + HF:meta-math/MetaMathQA + HF:openai/gsm8k main split=train,synthetic:self (sampled from the exp-08 checkpoint),synthetic:self (sampled from the exp-05 checkpoint) | completed | 0.688 | supported | adopt | True |  |
| train | r-d7327838 | exp-11 | Qwen/Qwen3-1.7B-Base | 437 | other | exp-10 |  | completed | 0.6667 | inconclusive | adopt | False | 0.6413949962092494 |
| train | r-db1eeb72 | exp-01 | Qwen/Qwen3-1.7B-Base | 86 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.703 | inconclusive | adopt | True |  |
| train | r-db1eeb72 | exp-02 | Qwen/Qwen3-1.7B-Base | 181 | sft | exp-01 | HF openai/gsm8k (main, split=train) | completed | 0.35 | inconclusive | adopt | True |  |
| train | r-db1eeb72 | exp-03 | Qwen/Qwen3-1.7B-Base | 252 | sft | exp-02 | HF openai/gsm8k (main, split=train),HF EleutherAI/asdiv (split=validation),HF mwpt5/MAWPS (split=train),HF cq01/mawps-asdiv-a_svamp (splits train + validation) | completed | 0.25 | contradicted | reject | True |  |
| train | r-db1eeb72 | exp-04 | Qwen/Qwen3-1.7B-Base | 274 | other | exp-02 |  | completed | 0.35 | inconclusive | adopt | False | 0.4268385140257771 |
| train | r-dc28e30d | exp-01 | Qwen/Qwen3-1.7B-Base | 147 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.46 | inconclusive | reject | True |  |
| train | r-dc28e30d | exp-02 | Qwen/Qwen3-1.7B-Base | 309 | sft | base_model | HF openai/gsm8k (train) x3 + HF meta-math/MetaMathQA (GSM types) | failed |  | inconclusive | abandon_line | True |  |
| train | r-dc28e30d | exp-03 | Qwen/Qwen3-1.7B-Base | 358 | sft | base_model | HF openai/gsm8k (train) x3 + HF meta-math/MetaMathQA (GSM types) | completed | 0.63 | supported | adopt | True |  |
| train | r-dc28e30d | exp-04 | Qwen/Qwen3-1.7B-Base | 483 | other | exp-03 |  | completed | 0.63 | inconclusive | adopt | False | 0.6118271417740713 |
| train | r-dc28e30d | exp-05 | Qwen/Qwen3-1.7B-Base | 531 | grpo | exp-03 | HF openai/gsm8k (main, split=train) | completed | 0.8 | supported | iterate | True |  |
| train | r-dd7a76f5 | exp-01 | Qwen/Qwen3-1.7B-Base | 59 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.175 | inconclusive | adopt | False |  |
| train | r-dd7a76f5 | exp-02 | Qwen/Qwen3-1.7B-Base | 146 | decode-config | exp-01 |  | completed | 0.5133 | supported | adopt | True |  |
| train | r-dd7a76f5 | exp-03 | Qwen/Qwen3-1.7B-Base | 204 | sft | exp-02 | HF openai/gsm8k (main, split=train) | completed | 0.58 | supported | adopt | True |  |
| train | r-dd7a76f5 | exp-04 | Qwen/Qwen3-1.7B-Base | 253 | sft | exp-03 | HF openai/gsm8k (main, split=train) | completed | 0.5067 | contradicted | reject | True |  |
| train | r-dd7a76f5 | exp-05 | Qwen/Qwen3-1.7B-Base | 275 | other | exp-03 |  | completed | 0.48 | inconclusive | adopt | False |  |
| train | r-dd7a76f5 | exp-06 | Qwen/Qwen3-1.7B-Base | 340 | sft | exp-05 | HF openai/gsm8k (main, split=train),synthetic:self (templated generator inside train_gsm8k_sft.py) | completed | 0.54 | supported | adopt | True |  |
| train | r-dd7a76f5 | exp-07 | Qwen/Qwen3-1.7B-Base | 388 | other | exp-06 |  | completed | 0.5155 | inconclusive | adopt | False | 0.530705079605762 |
| train | r-dd7a76f5 | exp-08 | Qwen/Qwen3-1.7B-Base | 432 | sft | exp-07 | HF openai/gsm8k (main, split=train) | completed | 0.5 | inconclusive | reject | False |  |
| train | r-dd7a76f5 | exp-09 | Qwen/Qwen3-1.7B-Base | 474 | sft | exp-07 | HF openai/gsm8k (main, split=train),synthetic:self (templated generator inside train_gsm8k_sft.py),HF ChilleD/SVAMP + HF MU-NLPC/Calc-mawps (via --external-wordproblems) | completed | 0.4867 | inconclusive | reject | False |  |
| train | r-df6e4451 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 427 | sft | base_model | local (openai/gsm8k train + nvidia/OpenMathInstruct-2 train_5M) | killed |  | inconclusive | abandon_line | False |  |
| train | r-df6e4451 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 510 | sft | base_model | local (openai/gsm8k train + nvidia/OpenMathInstruct-2 train_5M) | killed |  | inconclusive | abandon_line | False |  |
| train | r-df6e4451 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 588 | sft | base_model | local (openai/gsm8k train + nvidia/OpenMathInstruct-2 train_5M) | failed |  | inconclusive | abandon_line | False |  |
| train | r-df6e4451 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 697 | sft | base_model | local (openai/gsm8k train + nvidia/OpenMathInstruct-2 train_5M) | killed |  | inconclusive | abandon_line | False |  |
| train | r-df6e4451 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 714 | sft | base_model | local (openai/gsm8k train + nvidia/OpenMathInstruct-2 train_5M) | completed | 0.844 | inconclusive | adopt | False |  |
| train | r-df6e4451 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 1106 | sft | base_model | local + synthetic:self (openai/gsm8k train, nvidia/OpenMathInstruct-2, and RFT traces sampled from the exp-05 checkpoint) | completed | 0.816 | contradicted | reject | False |  |
| train | r-df6e4451 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 1167 | merge | exp-05 |  | completed | 0.836 | inconclusive | reject | False |  |
| train | r-df6e4451 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1247 | grpo | exp-05 | HF openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| train | r-df6e4451 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 1271 | grpo | exp-05 | HF openai/gsm8k | killed | 0.896 | supported | adopt | True |  |
| train | r-df6e4451 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 1348 | decode-config | exp-09 |  | completed | 0.892 | supported | adopt | False | 0.8514025777103866 |
| train | r-e2ebe966 | exp-01 | Qwen/Qwen3-4B-Base | 67 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.0 | contradicted | reject | False |  |
| train | r-e2ebe966 | exp-02 | Qwen/Qwen3-4B-Base | 211 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.2 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-03 | Qwen/Qwen3-4B-Base | 239 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.1 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-04 | Qwen/Qwen3-4B-Base | 264 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.1 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-05 | Qwen/Qwen3-4B-Base | 292 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.1 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-06 | Qwen/Qwen3-4B-Base | 307 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.3 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-07 | Qwen/Qwen3-4B-Base | 371 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.4 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-08 | Qwen/Qwen3-4B-Base | 439 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.4 | contradicted | reject | True |  |
| train | r-e2ebe966 | exp-09 | Qwen/Qwen3-4B-Base | 463 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.4 | inconclusive | reject | True |  |
| train | r-e2ebe966 | exp-10 | Qwen/Qwen3-4B-Base | 479 | sft | base_model | gsm8k train split (loaded in the trainer script, no local file) | completed | 0.5667 | supported | adopt | True |  |
| train | r-e2ebe966 | exp-11 | Qwen/Qwen3-4B-Base | 490 | other | exp-10 |  | completed | 0.3 | inconclusive | adopt | False | 0.4184988627748294 |
| train | r-e57ca9db | exp-01 | Qwen/Qwen3-1.7B-Base | 1686 | decode-config | base_model |  | killed |  | inconclusive | abandon_line | False |  |
| train | r-e57ca9db | exp-02 | Qwen/Qwen3-1.7B-Base | 1719 | decode-config | base_model |  | completed | 0.74 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-03 | Qwen/Qwen3-1.7B-Base | 1800 | rft | base_model | synthetic:self | completed | 0.8125 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-04 | Qwen/Qwen3-1.7B-Base | 1859 | decode-config | exp-03 |  | completed | 0.76 | supported | reject | True |  |
| train | r-e57ca9db | exp-05 | Qwen/Qwen3-1.7B-Base | 1867 | merge | base_model |  | completed | 0.68 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-06 | Qwen/Qwen3-1.7B-Base | 1925 | decode-config | base_model |  | completed |  | inconclusive | reject | True |  |
| train | r-e57ca9db | exp-07 | Qwen/Qwen3-1.7B-Base | 1998 | dpo | base_model | synthetic:self | failed |  | inconclusive | abandon_line | True |  |
| train | r-e57ca9db | exp-08 | Qwen/Qwen3-1.7B-Base | 2006 | dpo | base_model | synthetic:self | completed | 0.859375 | supported | adopt | True |  |
| train | r-e57ca9db | exp-09 | Qwen/Qwen3-1.7B-Base | 2043 | dpo | base_model | synthetic:self | completed | 0.796875 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-10 | Qwen/Qwen3-1.7B-Base | 2061 | other | exp-08 |  | completed | 0.76 | inconclusive | adopt | False | 0.7391963608794542 |
| train | r-e57ca9db | exp-11 | Qwen/Qwen3-1.7B-Base | 2081 | merge | exp-10 |  | completed | 0.75 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-12 | Qwen/Qwen3-1.7B-Base | 2087 | decode-config | exp-10 |  | completed | 0.77 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-13 | Qwen/Qwen3-1.7B-Base | 2187 | dpo | exp-10 | synthetic:self | completed | 0.828125 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-14 | Qwen/Qwen3-1.7B-Base | 2229 | sft | exp-10 | local | completed | 0.828125 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-15 | Qwen/Qwen3-1.7B-Base | 2261 | sft | exp-10 | local | completed | 0.8125 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-16 | Qwen/Qwen3-1.7B-Base | 2301 | decode-config | exp-10 |  | completed | 0.22 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-17 | Qwen/Qwen3-1.7B-Base | 2320 | decode-config | exp-10 |  | completed | 0.79 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-18 | Qwen/Qwen3-1.7B-Base | 2351 | merge | exp-10 |  | completed | 0.76 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-19 | Qwen/Qwen3-1.7B-Base | 2407 | dpo | exp-10 | synthetic:self | completed | 0.859375 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-20 | Qwen/Qwen3-1.7B-Base | 2458 | decode-config | exp-10 |  | completed | 0.76 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-21 | Qwen/Qwen3-1.7B-Base | 2515 | dpo | exp-10 | synthetic:self | completed | 0.16 | contradicted | reject | True |  |
| train | r-e57ca9db | exp-22 | Qwen/Qwen3-1.7B-Base | 2605 | decode-config | exp-10 |  | completed | 0.78 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-01 | Qwen/Qwen3-1.7B-Base | 52 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | False |  |
| train | r-e5ed6bd2 | exp-02 | Qwen/Qwen3-1.7B-Base | 64 | sft | base_model | HF: gsm8k main, train split | completed | 0.02666666666666667 | inconclusive | reject | False |  |
| train | r-e5ed6bd2 | exp-03 | Qwen/Qwen3-1.7B-Base | 118 | sft | base_model | HF: gsm8k main, train split | completed | 0.02 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-04 | Qwen/Qwen3-1.7B-Base | 148 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | True |  |
| train | r-e5ed6bd2 | exp-05 | Qwen/Qwen3-1.7B-Base | 154 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | False |  |
| train | r-e5ed6bd2 | exp-06 | Qwen/Qwen3-1.7B-Base | 160 | sft | base_model | HF: gsm8k main, train split | completed | 0.17333333333333334 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-07 | Qwen/Qwen3-1.7B-Base | 191 | sft | base_model | HF: gsm8k main, train split | completed | 0.07808946171341925 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-08 | Qwen/Qwen3-1.7B-Base | 197 | other | base_model |  | completed | 0.08 | inconclusive | adopt | True | 0.10917361637604246 |
| train | r-e79c6a8d | exp-01 | Qwen/Qwen3-4B-Base | 17760 | sft | base_model | synthetic:self | failed |  | inconclusive | iterate | True |  |
| train | r-e79c6a8d | exp-02 | Qwen/Qwen3-4B-Base | 17977 | sft | base_model | synthetic:self | completed | 0.9066666666666666 | supported | reject | True |  |
| train | r-e79c6a8d | exp-03 | Qwen/Qwen3-4B-Base | 28270 | sft | base_model | derived:exp-02 | completed | 0.9 | inconclusive | adopt | False |  |
| train | r-e79c6a8d | exp-04 | Qwen/Qwen3-4B-Base | 32291 | decode-config | exp-03 |  | completed | 0.8620166793025019 | inconclusive | reject | False |  |
| train | r-e79c6a8d | exp-05 | Qwen/Qwen3-4B-Base | 32569 | decode-config | exp-04 |  | completed |  | inconclusive | adopt | False |  |
| train | r-e79c6a8d | exp-06 | Qwen/Qwen3-4B-Base | 35285 | decode-config | exp-05 |  | completed | 0.8794541319181198 | supported | adopt | True |  |
| train | r-e79c6a8d | exp-07 | Qwen/Qwen3-4B-Base | 36424 | decode-config | exp-05 |  | completed | 0.8933 | inconclusive | adopt | False | 0.8832448824867324 |
| train | r-e81868fd | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 134 | sft | base_model | local: openai/gsm8k + meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-e81868fd | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 163 | sft | base_model | local: openai/gsm8k + meta-math/MetaMathQA | completed | 0.54 | supported | adopt | True |  |
| train | r-e81868fd | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 205 | other | exp-02 |  | completed | 0.5 | inconclusive | adopt | False | 0.5625473843821076 |
| train | r-e81868fd | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 262 | sft | base_model | local: openai/gsm8k + meta-math/MetaMathQA + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | True |  |
| train | r-e81868fd | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 285 | sft | exp-03 | local: openai/gsm8k + microsoft/orca-math-word-problems-200k | completed | 0.48 | contradicted | reject | True |  |
| train | r-e81868fd | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 342 | sft | exp-03 | local: openai/gsm8k + meta-math/MetaMathQA | completed | 0.5133333333333333 | contradicted | reject | True |  |
| train | r-eb6370c9 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 55 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | failed |  | inconclusive | abandon_line | False |  |
| train | r-eb6370c9 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 65 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | killed |  | inconclusive | abandon_line | False |  |
| train | r-eb6370c9 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 67 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | adopt | False |  |
| train | r-eb6370c9 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 104 | merge | exp-03 |  | completed | 0.12 | contradicted | reject | False |  |
| train | r-eb6370c9 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 124 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | adopt | False |  |
| train | r-eb6370c9 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 136 | merge | exp-05 |  | completed | 0.247 | supported | adopt | False | 0.30932524639878695 |
| train | r-eb6370c9 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 148 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | abandon_line | False |  |
| train | r-ee1ca44a | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 62 | sft | base_model | meta-math/MetaMathQA | completed | 0.1 | contradicted | adopt | False |  |
| train | r-ee1ca44a | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 110 | decode-config | exp-01 |  | completed | 0.7 | supported | reject | False |  |
| train | r-ee1ca44a | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 124 | sft | base_model | AI-MO/NuminaMath-CoT | completed | 0.5133333333333333 | contradicted | reject | False |  |
| train | r-ee1ca44a | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | meta-math/MetaMathQA | failed |  | inconclusive | iterate | False |  |
| train | r-ee1ca44a | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 166 | sft | base_model | meta-math/MetaMathQA | completed | 0.6333333333333333 | supported | adopt | False |  |
| train | r-ee1ca44a | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 184 | other | exp-05 |  | completed | 0.6 | inconclusive | adopt | False | 0.6520090978013646 |
| train | r-ee271986 | exp-01 | Qwen/Qwen3-4B-Base | 129 | sft | base_model | local (built from HF openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.84 | inconclusive | adopt | True |  |
| train | r-ee271986 | exp-02 | Qwen/Qwen3-4B-Base | 275 | decode-config | exp-01 |  | completed | 0.84 | inconclusive | adopt | True |  |
| train | r-ee271986 | exp-03 | Qwen/Qwen3-4B-Base | 309 | other | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| train | r-ee271986 | exp-04 | Qwen/Qwen3-4B-Base | 342 | rft | base_model | derived:exp-01 (self-sampled solutions) + local (the exp-01 SFT file),synthetic:self | failed |  | inconclusive | abandon_line | True |  |
| train | r-ee271986 | exp-05 | Qwen/Qwen3-4B-Base | 422 | rft | base_model | derived:exp-01 (self-sampled solutions) + local (the exp-01 SFT file),synthetic:self | completed | 0.82 | contradicted | reject | True |  |
| train | r-ee271986 | exp-06 | Qwen/Qwen3-4B-Base | 500 | grpo | exp-01 | openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-ee271986 | exp-07 | Qwen/Qwen3-4B-Base | 528 | merge | exp-01 | derived:exp-06 | failed |  | inconclusive | abandon_line | False |  |
| train | r-ee271986 | exp-08 | Qwen/Qwen3-4B-Base | 542 | merge | exp-01 | derived:exp-06 | completed | 0.8635329795299469 | inconclusive | adopt | False |  |
| train | r-ee271986 | exp-09 | Qwen/Qwen3-4B-Base | 580 | other | exp-08 |  | completed | 0.84 | inconclusive | adopt | False |  |
| train | r-ee271986 | exp-10 | Qwen/Qwen3-4B-Base | 595 | grpo | exp-08 | openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True |  |
| train | r-ee271986 | exp-11 | Qwen/Qwen3-4B-Base | 621 | merge | exp-08 | derived:exp-10 | completed |  | inconclusive | abandon_line | False |  |
| train | r-ee271986 | exp-12 | Qwen/Qwen3-4B-Base | 801 | grpo | exp-01 | openai/gsm8k (main, train split) | completed |  | inconclusive | adopt | True | 0.8407884761182715 |
| train | r-ee271986 | exp-13 | Qwen/Qwen3-4B-Base | 829 | merge | exp-01 | derived:exp-12 | completed |  | inconclusive | abandon_line | False |  |
| train | r-ef3f45de | exp-01 | Qwen/Qwen3-4B-Base | 76 | sft | base_model | HF meta-math/MetaMathQA + HF openai/gsm8k (main, train) | killed |  | inconclusive | iterate | True |  |
| train | r-ef3f45de | exp-02 | Qwen/Qwen3-4B-Base | 82 | sft | base_model | HF meta-math/MetaMathQA + HF openai/gsm8k (main, train) | killed |  | inconclusive | iterate | False |  |
| train | r-ef3f45de | exp-03 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | HF meta-math/MetaMathQA + HF openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-ef3f45de | exp-04 | Qwen/Qwen3-4B-Base | 133 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-ef3f45de | exp-05 | Qwen/Qwen3-4B-Base | 157 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-ef3f45de | exp-06 | Qwen/Qwen3-4B-Base | 165 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| train | r-ef3f45de | exp-07 | Qwen/Qwen3-4B-Base | 217 | merge | exp-06 |  | completed | 0.08 | inconclusive | reject | False |  |
| train | r-ef3f45de | exp-08 | Qwen/Qwen3-4B-Base | 266 | sft | base_model | HF meta-math/MetaMathQA + HF openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| train | r-ef3f45de | exp-09 | Qwen/Qwen3-4B-Base | 276 | merge | exp-08 |  | completed | 0.04 | contradicted | reject | False |  |
| train | r-ef3f45de | exp-10 | Qwen/Qwen3-4B-Base | 331 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| train | r-ef3f45de | exp-11 | Qwen/Qwen3-4B-Base | 360 | merge | exp-10 |  | completed | 0.02 | contradicted | reject | False |  |
| train | r-ef3f45de | exp-12 | Qwen/Qwen3-4B-Base | 388 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| train | r-ef3f45de | exp-13 | Qwen/Qwen3-4B-Base | 403 | merge | exp-12 |  | completed | 0.04 | contradicted | reject | False |  |
| train | r-ef3f45de | exp-14 | Qwen/Qwen3-4B-Base | 425 | sft | base_model | HF meta-math/MetaMathQA (GSM_* only) + HF openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| train | r-ef3f45de | exp-15 | Qwen/Qwen3-4B-Base | 435 | merge | exp-14 |  | completed | 0.32 | supported | adopt | False | 0.312357846853677 |
| train | r-ef3f45de | exp-16 | Qwen/Qwen3-4B-Base | 519 | sft | base_model | HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (GSM_* only) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f099ff8e | exp-01 | Qwen/Qwen3-1.7B-Base | 188 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.1133 | inconclusive | reject | False |  |
| train | r-f099ff8e | exp-02 | Qwen/Qwen3-1.7B-Base | 341 | sft | base_model | HF openai/gsm8k (train) + HF meta-math/MetaMathQA | completed | 0.033 | inconclusive | adopt | True |  |
| train | r-f099ff8e | exp-03 | Qwen/Qwen3-1.7B-Base | 497 | sft | exp-02 | derived:exp-02 | killed |  | inconclusive | abandon_line | True |  |
| train | r-f099ff8e | exp-04 | Qwen/Qwen3-1.7B-Base | 551 | sft | exp-02 | derived:exp-02 | killed | 0.595 | inconclusive | adopt | True |  |
| train | r-f099ff8e | exp-05 | Qwen/Qwen3-1.7B-Base | 676 | rft | exp-04 | derived:exp-04 | completed | 0.67 | inconclusive | reject | True |  |
| train | r-f099ff8e | exp-06 | Qwen/Qwen3-1.7B-Base | 790 | rft | exp-04 | derived:exp-05 | completed | 0.632 | inconclusive | adopt | True |  |
| train | r-f099ff8e | exp-07 | Qwen/Qwen3-1.7B-Base | 918 | grpo | exp-06 | HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f099ff8e | exp-08 | Qwen/Qwen3-1.7B-Base | 934 | grpo | exp-06 | HF openai/gsm8k (main, train split) | completed | 0.756 | supported | adopt | True |  |
| train | r-f099ff8e | exp-09 | Qwen/Qwen3-1.7B-Base | 1010 | grpo | exp-08 | HF openai/gsm8k (main, train split) | completed | 0.801 | supported | adopt | True | 0.7831690674753601 |
| train | r-f099ff8e | exp-10 | Qwen/Qwen3-1.7B-Base | 1064 | grpo | exp-09 | HF openai/gsm8k (main, train split) | completed | 0.794 | inconclusive | reject | False |  |
| train | r-f099ff8e | exp-11 | Qwen/Qwen3-1.7B-Base | 1152 | merge | exp-09 |  | completed | 0.794 | inconclusive | reject | True |  |
| train | r-f238cc6e | exp-01 | Qwen/Qwen3-4B-Base | 143 | sft | base_model | HF id: openai/gsm8k (main, train) | killed | 0.828125 | inconclusive | reject | True |  |
| train | r-f238cc6e | exp-02 | Qwen/Qwen3-4B-Base | 283 | sft | base_model | HF id: math-ai/TemplateGSM (templategsm-1000-1k, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-03 | Qwen/Qwen3-4B-Base | 301 | sft | base_model | HF id: math-ai/TemplateGSM (templategsm-1000-1k, train) | completed |  | inconclusive | adopt | True |  |
| train | r-f238cc6e | exp-04 | Qwen/Qwen3-4B-Base | 328 | sft | exp-03 | HF id: openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-05 | Qwen/Qwen3-4B-Base | 333 | sft | exp-03 | HF id: openai/gsm8k (main, train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-06 | Qwen/Qwen3-4B-Base | 339 | sft | exp-03 | HF id: openai/gsm8k (main, train) | killed | 0.8125 | contradicted | reject | True |  |
| train | r-f238cc6e | exp-07 | Qwen/Qwen3-4B-Base | 394 | sft | base_model | HF id: openai/gsm8k (main, train) | completed | 0.0 | inconclusive | adopt | True |  |
| train | r-f238cc6e | exp-08 | Qwen/Qwen3-4B-Base | 509 | decode-config | exp-07 |  | completed | 0.4 | supported | adopt | True |  |
| train | r-f238cc6e | exp-09 | Qwen/Qwen3-4B-Base | 552 | decode-config | exp-08 |  | completed | 0.64 | supported | adopt | True | 0.6762699014404853 |
| train | r-f238cc6e | exp-10 | Qwen/Qwen3-4B-Base | 598 | sft | exp-09 | HF id: openai/gsm8k (main, train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-11 | Qwen/Qwen3-4B-Base | 609 | sft | exp-09 | HF id: openai/gsm8k (main, train) | completed | 0.6 | contradicted | reject | True |  |
| train | r-f4df2c91 | exp-01 | Qwen/Qwen3-1.7B-Base | 20 | other | base_model |  | completed | 0.09 | inconclusive | reject | False |  |
| train | r-f4df2c91 | exp-02 | Qwen/Qwen3-1.7B-Base | 51 | sft | base_model | HF openai/gsm8k (config 'main', train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f4df2c91 | exp-03 | Qwen/Qwen3-1.7B-Base | 69 | sft | base_model | HF openai/gsm8k (config 'main', train split) | completed |  | inconclusive | adopt | False |  |
| train | r-f4df2c91 | exp-04 | Qwen/Qwen3-1.7B-Base | 81 | merge | exp-03 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-f4df2c91 | exp-05 | Qwen/Qwen3-1.7B-Base | 102 | merge | exp-03 |  | completed | 0.04 | contradicted | reject | False |  |
| train | r-f4df2c91 | exp-06 | Qwen/Qwen3-1.7B-Base | 113 | sft | base_model | HF openai/gsm8k (config 'main', train split) | killed |  | inconclusive | abandon_line | False |  |
| train | r-f4df2c91 | exp-07 | Qwen/Qwen3-1.7B-Base | 124 | sft | base_model | HF openai/gsm8k (config 'main', train split) | completed |  | inconclusive | adopt | False |  |
| train | r-f4df2c91 | exp-08 | Qwen/Qwen3-1.7B-Base | 130 | merge | exp-07 |  | completed | 0.06 | contradicted | reject | False |  |
| train | r-f4df2c91 | exp-09 | Qwen/Qwen3-1.7B-Base | 144 | other | base_model |  | completed | 0.12 | supported | reject | True |  |
| train | r-f4df2c91 | exp-10 | Qwen/Qwen3-1.7B-Base | 151 | sft | base_model | HF openai/gsm8k (config 'main', train split) | completed |  | inconclusive | adopt | True |  |
| train | r-f4df2c91 | exp-11 | Qwen/Qwen3-1.7B-Base | 153 | merge | exp-10 |  | completed | 0.03 | contradicted | reject | False |  |
| train | r-f4df2c91 | exp-12 | Qwen/Qwen3-1.7B-Base | 160 | other | base_model |  | completed | 0.127 | inconclusive | adopt | True | 0.1152388172858226 |
| train | r-f5bfab57 | exp-01 | Qwen/Qwen3-4B-Base | 58 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-f5bfab57 | exp-02 | Qwen/Qwen3-4B-Base | 60 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.59 | inconclusive | reject | False |  |
| train | r-f5bfab57 | exp-03 | Qwen/Qwen3-4B-Base | 88 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.6 | inconclusive | reject | False |  |
| train | r-f5bfab57 | exp-04 | Qwen/Qwen3-4B-Base | 106 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.73 | supported | reject | True |  |
| train | r-f5bfab57 | exp-05 | Qwen/Qwen3-4B-Base | 120 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.6580742987111448 | supported | adopt | False | 0.6391205458680819 |
| train | r-ffa696e1 | exp-01 | Qwen/Qwen3-1.7B-Base | 139 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.0467 | inconclusive | adopt | False |  |
| train | r-ffa696e1 | exp-02 | Qwen/Qwen3-1.7B-Base | 261 | decode-config | exp-01 |  | completed | 0.66 | inconclusive | reject | True |  |
| train | r-ffa696e1 | exp-03 | Qwen/Qwen3-1.7B-Base | 282 | sft | base_model | HF openai/gsm8k (main, train split),HF meta-math/MetaMathQA (rows whose type starts with GSM) | killed |  | inconclusive | abandon_line | False |  |
| train | r-ffa696e1 | exp-04 | Qwen/Qwen3-1.7B-Base | 311 | sft | base_model | HF openai/gsm8k (main, train split),HF meta-math/MetaMathQA (rows whose type starts with GSM) | failed |  | inconclusive | abandon_line | False |  |
| train | r-ffa696e1 | exp-05 | Qwen/Qwen3-1.7B-Base | 373 | sft | base_model | HF openai/gsm8k (main, train split),HF meta-math/MetaMathQA (rows whose type starts with GSM) | killed | 0.7133 | inconclusive | reject | True |  |
| train | r-ffa696e1 | exp-06 | Qwen/Qwen3-1.7B-Base | 578 | rft | base_model | HF openai/gsm8k (main, train split),synthetic:self (sampled from exp-05's checkpoint on GSM8K train questions),HF meta-math/MetaMathQA (rows whose type starts with GSM) | killed | 0.7533 | supported | adopt | True |  |
| train | r-ffa696e1 | exp-07 | Qwen/Qwen3-1.7B-Base | 611 | other | exp-06 |  | completed | 0.7333 | inconclusive | adopt | False |  |
| train | r-ffa696e1 | exp-08 | Qwen/Qwen3-1.7B-Base | 650 | rft | base_model | HF openai/gsm8k (main, train split),synthetic:self (round 1 from exp-05's checkpoint, round 2 from exp-07's final_model),HF meta-math/MetaMathQA (rows whose type starts with GSM) | killed | 0.7533 | contradicted | reject | True |  |
| train | r-ffa696e1 | exp-09 | Qwen/Qwen3-1.7B-Base | 677 | rft | base_model | HF openai/gsm8k (main, train split),synthetic:self (derived:exp-05 and derived:exp-07 generations),HF meta-math/MetaMathQA (rows whose type starts with GSM) | killed | 0.7 | contradicted | reject | True |  |
| train | r-ffa696e1 | exp-10 | Qwen/Qwen3-1.7B-Base | 738 | grpo | exp-07 | HF openai/gsm8k (main, train split), prompts only | completed | 0.0 | contradicted | reject | True |  |
| train | r-ffa696e1 | exp-11 | Qwen/Qwen3-1.7B-Base | 859 | grpo | exp-07 | HF openai/gsm8k (main, train split), prompts only | completed | 0.8333 | supported | adopt | True |  |
| train | r-ffa696e1 | exp-12 | Qwen/Qwen3-1.7B-Base | 896 | other | exp-11 |  | completed | 0.8467 | supported | adopt | False | 0.7869598180439727 |
| train | r-ffa696e1 | exp-13 | Qwen/Qwen3-1.7B-Base | 922 | grpo | exp-12 | HF openai/gsm8k (main, train split), prompts only | failed |  | inconclusive | abandon_line | True |  |
| train | r-ffa696e1 | exp-14 | Qwen/Qwen3-1.7B-Base | 935 | grpo | exp-11 | HF openai/gsm8k (main, train split), prompts only | completed | 0.8 | contradicted | reject | True |  |
| train | r-ffa696e1 | exp-15 | Qwen/Qwen3-1.7B-Base | 999 | merge | exp-11 |  | failed |  | inconclusive | abandon_line | True |  |
| train | r-ffa696e1 | exp-16 | Qwen/Qwen3-1.7B-Base | 1003 | merge | exp-11 |  | completed | 0.8 | contradicted | reject | True |  |
| test | r-10e35756 | exp-01 | google/gemma-3-4b-pt | 78 | sft | base_model | openai/gsm8k (config: main, split: train),openai/gsm8k (config: socratic, split: train) | completed |  | inconclusive | adopt | False |  |
| test | r-10e35756 | exp-02 | google/gemma-3-4b-pt | 304 | merge | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| test | r-10e35756 | exp-03 | google/gemma-3-4b-pt | 314 | other | exp-02 |  | completed | 0.34 | inconclusive | reject | True |  |
| test | r-10e35756 | exp-04 | google/gemma-3-4b-pt | 327 | merge | exp-01 |  | completed | 0.4 | inconclusive | adopt | True | 0.3752843062926459 |
| test | r-10e35756 | exp-05 | google/gemma-3-4b-pt | 348 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-10e35756 | exp-06 | google/gemma-3-4b-pt | 399 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-11ba81b6 | exp-01 | google/gemma-3-4b-pt | 93 | sft | base_model | meta-math/MetaMathQA (GSM_* rows only), built in-process by the trainer | completed | 0.04 | inconclusive | adopt | True |  |
| test | r-11ba81b6 | exp-02 | google/gemma-3-4b-pt | 231 | sft | exp-01 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.6262319939347991 | supported | adopt | True |  |
| test | r-11ba81b6 | exp-03 | google/gemma-3-4b-pt | 290 | sft | exp-02 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.623199393479909 | inconclusive | adopt | True |  |
| test | r-11ba81b6 | exp-04 | google/gemma-3-4b-pt | 348 | sft | exp-03 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.6087945413191812 | contradicted | reject | True |  |
| test | r-11ba81b6 | exp-05 | google/gemma-3-4b-pt | 385 | sft | exp-03 | meta-math/MetaMathQA (GSM_* rows only), built in-process by the trainer | completed |  | inconclusive | adopt | True |  |
| test | r-11ba81b6 | exp-06 | google/gemma-3-4b-pt | 560 | sft | exp-05 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.6066666666666667 | contradicted | reject | True |  |
| test | r-11ba81b6 | exp-07 | google/gemma-3-4b-pt | 612 | sft | exp-05 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.6266666666666667 | contradicted | adopt | True |  |
| test | r-11ba81b6 | exp-08 | google/gemma-3-4b-pt | 667 | sft | exp-07 | openai/gsm8k main train split, built in-process by the trainer | completed | 0.6333333333333333 | inconclusive | reject | True |  |
| test | r-11ba81b6 | exp-09 | google/gemma-3-4b-pt | 774 | merge | exp-03 |  | killed |  | inconclusive | abandon_line | True |  |
| test | r-11ba81b6 | exp-10 | google/gemma-3-4b-pt | 835 | rft | exp-02 | synthetic:self (sampled from aligned_model, questions from openai/gsm8k main train) | completed | 0.6171341925701289 | contradicted | reject | True |  |
| test | r-11ba81b6 | exp-11 | google/gemma-3-4b-pt | 868 | merge | exp-02 |  | completed | 0.6178923426838514 | contradicted | reject | True |  |
| test | r-11ba81b6 | exp-12 | google/gemma-3-4b-pt | 930 | merge | exp-02 |  | completed | 0.6239575435936315 | inconclusive | reject | True |  |
| test | r-11ba81b6 | exp-13 | google/gemma-3-4b-pt | 951 | other |  |  | completed | 0.6338134950720242 | inconclusive | adopt | True | 0.6345716451857468 |
| test | r-157ef258 | exp-01 | google/gemma-3-4b-pt | 110 | sft | base_model | HF meta-math/MetaMathQA (split=train) | killed |  | inconclusive | abandon_line | False |  |
| test | r-157ef258 | exp-02 | google/gemma-3-4b-pt | 122 | sft | base_model | HF meta-math/MetaMathQA (split=train) | completed |  | inconclusive | adopt | False |  |
| test | r-157ef258 | exp-03 | google/gemma-3-4b-pt | 152 | other | exp-02 |  | completed | 0.6 | inconclusive | adopt | False | 0.5921152388172858 |
| test | r-1944f3f0 | exp-01 | google/gemma-3-4b-pt | 128 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-1944f3f0 | exp-02 | google/gemma-3-4b-pt | 254 | sft | base_model | openai/gsm8k | completed | 0.3466666666666667 | supported | adopt | True |  |
| test | r-1944f3f0 | exp-03 | google/gemma-3-4b-pt | 319 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-1944f3f0 | exp-04 | google/gemma-3-4b-pt | 357 | decode-config | exp-02 |  | completed | 0.275 | contradicted | reject | True |  |
| test | r-1944f3f0 | exp-05 | google/gemma-3-4b-pt | 361 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-1944f3f0 | exp-06 | google/gemma-3-4b-pt | 394 | other | exp-02 |  | completed | 0.37333333333333335 | supported | adopt | False | 0.33358605003790753 |
| test | r-241fe20e | exp-01 | google/gemma-3-4b-pt | 109 | sft | base_model | HF openai/gsm8k config=main split=train (loaded in-process, no file written) | failed |  | inconclusive | abandon_line | True |  |
| test | r-241fe20e | exp-02 | google/gemma-3-4b-pt | 118 | sft | base_model | HF openai/gsm8k config=main split=train (loaded in-process, no file written) | killed |  | inconclusive | abandon_line | True |  |
| test | r-241fe20e | exp-03 | google/gemma-3-4b-pt | 137 | sft | base_model | HF openai/gsm8k config=main split=train (loaded in-process, no file written) | completed | 0.15 | supported | reject | True |  |
| test | r-241fe20e | exp-04 | google/gemma-3-4b-pt | 222 | sft | base_model | HF openai/gsm8k config=main split=train (loaded in-process, no file written) | completed | 0.3667 | supported | adopt | True |  |
| test | r-241fe20e | exp-05 | google/gemma-3-4b-pt | 255 | sft | base_model | HF openai/gsm8k config=main split=train (loaded in-process, no file written) | killed |  | inconclusive | abandon_line | False |  |
| test | r-241fe20e | exp-06 | google/gemma-3-4b-pt | 286 | merge | exp-05 |  | completed | 0.3 | contradicted | reject | True |  |
| test | r-241fe20e | exp-07 | google/gemma-3-4b-pt | 301 | other | exp-04 |  | completed | 0.325 | supported | adopt | False | 0.38968915845337376 |
| test | r-2432c636 | exp-01 | google/gemma-3-4b-pt | 154 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.06 | inconclusive | abandon_line | True |  |
| test | r-2432c636 | exp-02 | google/gemma-3-4b-pt | 398 | sft | base_model | math-ai/TemplateGSM (default config, split train, streaming) | completed |  | inconclusive | adopt | True |  |
| test | r-2432c636 | exp-03 | google/gemma-3-4b-pt | 494 | sft | exp-02 | openai/gsm8k (config main, split train) | completed | 0.22 | supported | adopt | True |  |
| test | r-2432c636 | exp-04 | google/gemma-3-4b-pt | 642 | sft | exp-03 | openai/gsm8k (config main, split train) | completed | 0.28 | supported | adopt | True |  |
| test | r-2432c636 | exp-05 | google/gemma-3-4b-pt | 875 | sft | exp-04 | math-ai/TemplateGSM (config templategsm-7473-1k, split train, streaming) | killed |  | inconclusive | abandon_line | True |  |
| test | r-2432c636 | exp-06 | google/gemma-3-4b-pt | 950 | sft | exp-04 | math-ai/TemplateGSM (config templategsm-7473-1k, split train, streaming) | killed |  | inconclusive | abandon_line | True |  |
| test | r-2432c636 | exp-07 | google/gemma-3-4b-pt | 973 | sft | exp-04 | math-ai/TemplateGSM (config templategsm-7473-1k, split train, streaming) | completed | 0.25 | inconclusive | adopt | True |  |
| test | r-2432c636 | exp-08 | google/gemma-3-4b-pt | 1227 | sft | exp-07 | openai/gsm8k (config main, split train),meta-math/MetaMathQA (split train, streaming) | killed | 0.35 | supported | adopt | True |  |
| test | r-2432c636 | exp-09 | google/gemma-3-4b-pt | 1480 | sft | exp-08 | openai/gsm8k (config main, split train) | killed | 0.42 | supported | adopt | True |  |
| test | r-2432c636 | exp-10 | google/gemma-3-4b-pt | 1714 | sft | exp-09 | openai/gsm8k (config main, split train) | completed | 0.4933333333333333 | supported | adopt | True | 0.44200151630022744 |
| test | r-2432c636 | exp-11 | google/gemma-3-4b-pt | 1861 | sft | exp-10 | openai/gsm8k (config main, split train) | completed | 0.44 | contradicted | reject | True |  |
| test | r-2432c636 | exp-12 | google/gemma-3-4b-pt | 2072 | sft | exp-10 | openai/gsm8k (config main, split train) | completed | 0.42 | contradicted | reject | True |  |
| test | r-25b064a8 | exp-01 | google/gemma-3-4b-pt | 234 | sft | base_model | HF id: openai/gsm8k (main, train),HF id: meta-math/MetaMathQA (GSM* types only) | completed | 0.06 | inconclusive | adopt | True |  |
| test | r-25b064a8 | exp-02 | google/gemma-3-4b-pt | 420 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types | killed |  | inconclusive | abandon_line | True |  |
| test | r-25b064a8 | exp-03 | google/gemma-3-4b-pt | 432 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types | killed |  | inconclusive | abandon_line | True |  |
| test | r-25b064a8 | exp-04 | google/gemma-3-4b-pt | 443 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types | completed | 0.7133333333333334 | supported | reject | True |  |
| test | r-25b064a8 | exp-05 | google/gemma-3-4b-pt | 480 | other | exp-04 |  | completed |  | inconclusive | reject | True |  |
| test | r-25b064a8 | exp-06 | google/gemma-3-4b-pt | 594 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types | completed | 0.72 | contradicted | reject | True |  |
| test | r-25b064a8 | exp-07 | google/gemma-3-4b-pt | 649 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types,HF id: microsoft/orca-math-word-problems-200k | completed | 0.72 | contradicted | reject | True |  |
| test | r-25b064a8 | exp-08 | google/gemma-3-4b-pt | 682 | other | exp-07 |  | completed | 0.7067 | inconclusive | reject | True |  |
| test | r-25b064a8 | exp-09 | google/gemma-3-4b-pt | 682 | sft | exp-01 | derived:exp-01 inputs - openai/gsm8k (main, train) + meta-math/MetaMathQA GSM types,HF id: microsoft/orca-math-word-problems-200k | completed | 0.7266666666666667 | inconclusive | adopt | True |  |
| test | r-25b064a8 | exp-10 | google/gemma-3-4b-pt | 767 | other | exp-09 |  | completed | 0.72 | supported | adopt | True | 0.6755117513267627 |
| test | r-27c73665 | exp-01 | google/gemma-3-4b-pt | 24351 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA (GSM_* types) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | False |  |
| test | r-27c73665 | exp-02 | google/gemma-3-4b-pt | 29086 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA (GSM_* types) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | False |  |
| test | r-27c73665 | exp-03 | google/gemma-3-4b-pt | 32052 | sft | base_model | openai/gsm8k train + meta-math/MetaMathQA (GSM_* types) + microsoft/orca-math-word-problems-200k | completed | 0.6266666666666667 | inconclusive | adopt | False |  |
| test | r-27c73665 | exp-04 | google/gemma-3-4b-pt | 43357 | rft | exp-03 | synthetic:self,synthetic:self,openai/gsm8k train + meta-math/MetaMathQA (GSM_*) + microsoft/orca-math-word-problems-200k,derived:exp-03 | failed |  | inconclusive | abandon_line | False |  |
| test | r-27c73665 | exp-05 | google/gemma-3-4b-pt | 45393 | rft | exp-03 | synthetic:self,synthetic:self,openai/gsm8k train + meta-math/MetaMathQA (GSM_*) + microsoft/orca-math-word-problems-200k,derived:exp-03 | completed | 0.6867 | supported | adopt | False |  |
| test | r-27c73665 | exp-06 | google/gemma-3-4b-pt | 46893 | other | exp-05 |  | completed | 0.6596 | contradicted | reject | False |  |
| test | r-27c73665 | exp-07 | google/gemma-3-4b-pt | 47771 | other | exp-05 |  | completed | 0.6933333333333334 | supported | adopt | False | 0.6603487490523123 |
| test | r-28a5f423 | exp-01 | google/gemma-3-4b-pt | 42 | other | base_model |  | completed | 0.05 | inconclusive | reject | False |  |
| test | r-28a5f423 | exp-02 | google/gemma-3-4b-pt | 88 | sft | base_model | local (datasets save_to_disk) built from HF gsm8k/main train, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k | killed | 0.5 | inconclusive | adopt | False |  |
| test | r-28a5f423 | exp-03 | google/gemma-3-4b-pt | 117 | merge | exp-02 |  | completed | 0.5 | inconclusive | reject | False |  |
| test | r-28a5f423 | exp-04 | google/gemma-3-4b-pt | 135 | sft | exp-02 | local (datasets save_to_disk) built from HF gsm8k/main train, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k | completed | 0.647 | inconclusive | adopt | False |  |
| test | r-28a5f423 | exp-05 | google/gemma-3-4b-pt | 151 | merge | exp-04 |  | completed |  | inconclusive | abandon_line | False |  |
| test | r-28a5f423 | exp-06 | google/gemma-3-4b-pt | 179 | merge | exp-04 |  | completed | 0.647 | inconclusive | adopt | False | 0.5852918877937832 |
| test | r-28a5f423 | exp-07 | google/gemma-3-4b-pt | 194 | sft | exp-04 | local (datasets save_to_disk) built from HF gsm8k/main train, meta-math/MetaMathQA, microsoft/orca-math-word-problems-200k | completed | 0.6 | contradicted | reject | False |  |
| test | r-28a5f423 | exp-08 | google/gemma-3-4b-pt | 309 | merge | exp-07 |  | completed | 0.6 | contradicted | reject | False |  |
| test | r-28a5f423 | exp-09 | google/gemma-3-4b-pt | 322 | merge | exp-07 |  | failed |  | inconclusive | abandon_line | True |  |
| test | r-28a5f423 | exp-10 | google/gemma-3-4b-pt | 339 | merge | exp-07 |  | completed | 0.613 | contradicted | reject | False |  |
| test | r-28a5f423 | exp-11 | google/gemma-3-4b-pt | 345 | merge | exp-07 |  | completed | 0.64 | inconclusive | reject | False |  |
| test | r-28a5f423 | exp-12 | google/gemma-3-4b-pt | 360 | merge | exp-04 |  | completed | 0.58 | contradicted | reject | False |  |
| test | r-28a5f423 | exp-13 | google/gemma-3-4b-pt | 401 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-37da5218 | exp-01 | google/gemma-3-4b-pt | 394 | sft | base_model | derived: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k, math + augmented_math) + openai/gsm8k train | completed | 0.74 | inconclusive | adopt | False |  |
| test | r-37da5218 | exp-02 | google/gemma-3-4b-pt | 729 | sft | exp-01 | derived:exp-01 (rft) + nvidia/OpenMathInstruct-2 fresh slice + openai/gsm8k train | completed | 0.7266666666666667 | inconclusive | reject | False |  |
| test | r-37da5218 | exp-03 | google/gemma-3-4b-pt | 824 | merge | exp-01 |  | completed |  | inconclusive | abandon_line | False |  |
| test | r-37da5218 | exp-04 | google/gemma-3-4b-pt | 873 | rft | exp-01 | synthetic:self (sampled from the exp-01 checkpoint), problems from nvidia/OpenMathInstruct-2 and openai/gsm8k train | completed | 0.74 | inconclusive | adopt | False |  |
| test | r-37da5218 | exp-05 | google/gemma-3-4b-pt | 919 | merge | exp-01 |  | completed | 0.7466666666666667 | inconclusive | adopt | False |  |
| test | r-37da5218 | exp-06 | google/gemma-3-4b-pt | 964 | other | exp-05 |  | completed | 0.74 | inconclusive | adopt | False | 0.7725549658832449 |
| test | r-3c44629f | exp-01 | google/gemma-3-4b-pt | 208 | sft | base_model | HF openai/gsm8k (main, train split) + HF meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | failed |  | inconclusive | abandon_line | True |  |
| test | r-3c44629f | exp-02 | google/gemma-3-4b-pt | 334 | sft | base_model | HF openai/gsm8k (main, train split) + HF meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.682 | inconclusive | adopt | True |  |
| test | r-3c44629f | exp-03 | google/gemma-3-4b-pt | 673 | rft | exp-02 | synthetic:self (sampled from the exp-02 checkpoint) + HF openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| test | r-3c44629f | exp-04 | google/gemma-3-4b-pt | 729 | rft | exp-02 | synthetic:self (sampled from the exp-02 checkpoint) + HF openai/gsm8k (main, train split) | completed | 0.682 | contradicted | adopt | True |  |
| test | r-3c44629f | exp-05 | google/gemma-3-4b-pt | 800 | grpo | exp-04 | HF openai/gsm8k (main, train split), prompts only | completed | 0.68 | contradicted | reject | True |  |
| test | r-3c44629f | exp-06 | google/gemma-3-4b-pt | 836 | other | exp-02 |  | completed |  | inconclusive | adopt | True |  |
| test | r-3c44629f | exp-07 | google/gemma-3-4b-pt | 899 | grpo | exp-04 | HF openai/gsm8k (main, train split), prompts only | killed |  | inconclusive | abandon_line | True |  |
| test | r-3c44629f | exp-08 | google/gemma-3-4b-pt | 994 | grpo | exp-04 | HF openai/gsm8k (main, train split), prompts only | killed |  | inconclusive | abandon_line | True |  |
| test | r-3c44629f | exp-09 | google/gemma-3-4b-pt | 1063 | sft | base_model | HF meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) + HF openai/gsm8k (main, train split) + synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| test | r-3c44629f | exp-10 | google/gemma-3-4b-pt | 1078 | grpo | exp-04 | HF openai/gsm8k (main, train split), prompts only | completed | 0.728 | supported | adopt | True |  |
| test | r-3c44629f | exp-11 | google/gemma-3-4b-pt | 1249 | other | exp-10 |  | completed | 0.733 | inconclusive | adopt | True |  |
| test | r-3c44629f | exp-12 | google/gemma-3-4b-pt | 1253 | grpo | exp-10 | HF openai/gsm8k (main, train split), prompts only | completed | 0.716 | contradicted | reject | True |  |
| test | r-3c44629f | exp-13 | google/gemma-3-4b-pt | 1352 | merge | exp-10 |  | completed | 0.707 | inconclusive | reject | True |  |
| test | r-3c44629f | exp-14 | google/gemma-3-4b-pt | 1407 | grpo | exp-04 | HF openai/gsm8k (main, train split), prompts only | completed | 0.719 | supported | adopt | True |  |
| test | r-3c44629f | exp-15 | google/gemma-3-4b-pt | 1516 | other | exp-14 |  | completed | 0.7263078089461713 | supported | adopt | True | 0.7179681576952237 |
| test | r-3c44629f | exp-16 | google/gemma-3-4b-pt | 1566 | grpo | exp-14 | HF openai/gsm8k (main, train split), prompts only | completed | 0.7165 | contradicted | reject | True |  |
| test | r-45dd80cf | exp-01 | google/gemma-3-4b-pt | 95 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.447 | inconclusive | adopt | True |  |
| test | r-45dd80cf | exp-02 | google/gemma-3-4b-pt | 169 | sft | base_model | openai/gsm8k (config main, split train),MU-NLPC/Calc-gsm8k (split train),ChilleD/SVAMP (split train),MU-NLPC/Calc-mawps (splits train and validation) | completed | 0.333 | contradicted | reject | True |  |
| test | r-45dd80cf | exp-03 | google/gemma-3-4b-pt | 220 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.44 | inconclusive | reject | True |  |
| test | r-45dd80cf | exp-04 | google/gemma-3-4b-pt | 260 | other | exp-01 |  | completed | 0.48 | inconclusive | adopt | False | 0.45564821834723274 |
| test | r-49d8b87a | exp-01 | google/gemma-3-4b-pt | 127 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.4666666666666667 | inconclusive | adopt | True |  |
| test | r-49d8b87a | exp-02 | google/gemma-3-4b-pt | 256 | sft | base_model | HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (train) | completed | 0.5066666666666667 | inconclusive | adopt | True |  |
| test | r-49d8b87a | exp-03 | google/gemma-3-4b-pt | 412 | decode-config | exp-02 |  | completed | 0.6466666666666666 | supported | adopt | True |  |
| test | r-49d8b87a | exp-04 | google/gemma-3-4b-pt | 428 | sft | base_model | HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (train) | completed | 0.4533333333333333 | contradicted | reject | True |  |
| test | r-49d8b87a | exp-05 | google/gemma-3-4b-pt | 440 | other | exp-03 |  | completed | 0.64 | supported | adopt | True | 0.6391205458680819 |
| test | r-49d8b87a | exp-06 | google/gemma-3-4b-pt | 482 | sft | base_model | HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (train) | completed | 0.64 | contradicted | reject | True |  |
| test | r-49d8b87a | exp-07 | google/gemma-3-4b-pt | 707 | rft | base_model | synthetic:self (solutions sampled from the exp-06 checkpoint, verified against openai/gsm8k train gold) + HF meta-math/MetaMathQA via data/gsm8k_mm.jsonl | completed | 0.644 | inconclusive | reject | True |  |
| test | r-5633966f | exp-01 | google/gemma-3-4b-pt | 199 | sft | base_model | openai/gsm8k | completed | 0.16 | inconclusive | adopt | True |  |
| test | r-5633966f | exp-02 | google/gemma-3-4b-pt | 218 | merge | exp-01 |  | completed |  | inconclusive | abandon_line | False |  |
| test | r-5633966f | exp-03 | google/gemma-3-4b-pt | 237 | merge | exp-01 |  | completed | 0.16 | inconclusive | adopt | True |  |
| test | r-5633966f | exp-04 | google/gemma-3-4b-pt | 261 | sft | exp-03 | openai/gsm8k | completed | 0.04 | contradicted | reject | True |  |
| test | r-5633966f | exp-05 | google/gemma-3-4b-pt | 510 | merge | exp-04 |  | completed | 0.04 | contradicted | reject | False |  |
| test | r-5633966f | exp-06 | google/gemma-3-4b-pt | 521 | sft | exp-03 | openai/gsm8k + meta-math/MetaMathQA (local arrow cache) | completed | 0.06 | contradicted | reject | True |  |
| test | r-5633966f | exp-07 | google/gemma-3-4b-pt | 679 | merge | exp-06 |  | completed | 0.06 | contradicted | reject | False |  |
| test | r-5633966f | exp-08 | google/gemma-3-4b-pt | 691 | sft | exp-03 | openai/gsm8k | completed | 0.0 | contradicted | reject | True |  |
| test | r-5633966f | exp-09 | google/gemma-3-4b-pt | 743 | merge | exp-08 |  | completed | 0.0 | contradicted | reject | False |  |
| test | r-5633966f | exp-10 | google/gemma-3-4b-pt | 753 | merge | exp-01 |  | completed | 0.32666666666666666 | inconclusive | adopt | False | 0.2721758908263836 |
| test | r-5e526b67 | exp-01 | google/gemma-3-4b-pt | 81 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.3375 | inconclusive | reject | False |  |
| test | r-5e526b67 | exp-02 | google/gemma-3-4b-pt | 130 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.425 | supported | adopt | True |  |
| test | r-5e526b67 | exp-03 | google/gemma-3-4b-pt | 179 | decode-config | exp-02 |  | completed | 0.475 | supported | adopt | True |  |
| test | r-5e526b67 | exp-04 | google/gemma-3-4b-pt | 190 | sft | base_model | openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-5e526b67 | exp-05 | google/gemma-3-4b-pt | 225 | decode-config | exp-03 |  | completed | 0.447 | inconclusive | adopt | False |  |
| test | r-5e526b67 | exp-06 | google/gemma-3-4b-pt | 234 | other | exp-05 |  | completed | 0.35 | inconclusive | adopt | False | 0.35784685367702807 |
| test | r-6b7e26ac | exp-01 | google/gemma-3-4b-pt | 181 | sft | base_model | HF openai/gsm8k, main, train split,HF meta-math/MetaMathQA, train split, types GSM_Rephrased + GSM_AnsAug | completed | 0.593 | inconclusive | adopt | True |  |
| test | r-6b7e26ac | exp-02 | google/gemma-3-4b-pt | 414 | decode-config | exp-01 |  | completed | 0.6133333333333333 | supported | adopt | True |  |
| test | r-6b7e26ac | exp-03 | google/gemma-3-4b-pt | 441 | sft | base_model | HF openai/gsm8k, main, train split,HF meta-math/MetaMathQA, train split, types GSM_Rephrased + GSM_AnsAug,HF meta-math/MetaMathQA, train split, types GSM_SV + GSM_FOBAR,derived:exp-01 data (data/gsm8k_train_4k.jsonl, itself `head -4000 data/gsm8k_train.jsonl`) | completed | 0.707 | supported | adopt | True |  |
| test | r-6b7e26ac | exp-04 | google/gemma-3-4b-pt | 585 | sft | base_model | HF openai/gsm8k, main, train split,HF meta-math/MetaMathQA, train split, types GSM_Rephrased + GSM_AnsAug,HF meta-math/MetaMathQA, train split, types GSM_SV + GSM_FOBAR,synthetic:self (sampled from /home/ben/task/runs/v2, the exp-03 checkpoint) | completed | 0.2 | contradicted | reject | True |  |
| test | r-6b7e26ac | exp-05 | google/gemma-3-4b-pt | 645 | other | exp-03 |  | completed | 0.702 | supported | adopt | True |  |
| test | r-6b7e26ac | exp-06 | google/gemma-3-4b-pt | 685 | sft | base_model | HF openai/gsm8k, main, train split,HF meta-math/MetaMathQA, train split, types GSM_Rephrased + GSM_AnsAug,HF meta-math/MetaMathQA, train split, types GSM_SV + GSM_FOBAR | completed | 0.39 | contradicted | adopt | True | 0.6770280515542078 |
| test | r-6b7e26ac | exp-07 | google/gemma-3-4b-pt | 763 | sft | exp-06 | HF openai/gsm8k, main, train split | completed | 0.317 | contradicted | reject | True |  |
| test | r-71205e90 | exp-01 | google/gemma-3-4b-pt | 15027 | sft | base_model | HF openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| test | r-71205e90 | exp-02 | google/gemma-3-4b-pt | 15718 | sft | base_model | HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| test | r-71205e90 | exp-03 | google/gemma-3-4b-pt | 17355 | sft | base_model | HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | False |  |
| test | r-71205e90 | exp-04 | google/gemma-3-4b-pt | 17974 | sft | base_model | HF openai/gsm8k (main, train split) | completed | 0.5866666666666667 | supported | adopt | True |  |
| test | r-71205e90 | exp-05 | google/gemma-3-4b-pt | 25368 | sft | base_model | synthetic:self (sampled from the exp-04 checkpoint ckpts/serve_v1e3),HF meta-math/MetaMathQA (GSM_AnsAug and GSM_Rephrased types only),derived:exp-04 (concatenation of data/sft_v2_core.jsonl and data/metamath_sft.jsonl, shuffled) | completed | 0.684 | supported | adopt | True |  |
| test | r-71205e90 | exp-06 | google/gemma-3-4b-pt | 26333 | other | exp-04 |  | completed | 0.5866666666666667 | inconclusive | adopt | False |  |
| test | r-71205e90 | exp-07 | google/gemma-3-4b-pt | 28658 | other | exp-05 |  | completed | 0.684 | inconclusive | adopt | False |  |
| test | r-71205e90 | exp-08 | google/gemma-3-4b-pt | 29356 | sft | exp-05 | synthetic:self (sampled from the exp-05 checkpoint ckpts/serve_v2e2),derived:exp-05 (rs_v2 rows deduplicated against rs_v1, plus hard-question originals, plus a fresh MetaMath slice) | completed | 0.7066666666666667 | inconclusive | adopt | True |  |
| test | r-71205e90 | exp-09 | google/gemma-3-4b-pt | 30311 | merge | exp-08 |  | completed | 0.672 | inconclusive | reject | False |  |
| test | r-71205e90 | exp-10 | google/gemma-3-4b-pt | 31013 | other | exp-08 |  | completed | 0.684 | inconclusive | adopt | False |  |
| test | r-71205e90 | exp-11 | google/gemma-3-4b-pt | 31015 | sft | exp-08 | derived:exp-05 (the same file exp-08 trained on: deduplicated round-2 rejection samples + hard-question originals + fresh MetaMath slice) | completed | 0.684 | inconclusive | reject | False |  |
| test | r-71205e90 | exp-12 | google/gemma-3-4b-pt | 35794 | grpo | exp-08 | HF openai/gsm8k (main, train split), prompts only; difficulty weighting from data/rs_v2.jsonl | failed |  | inconclusive | abandon_line | True |  |
| test | r-71205e90 | exp-13 | google/gemma-3-4b-pt | 37208 | grpo | exp-08 | HF openai/gsm8k (main, train split), prompts only; difficulty weighting from data/rs_v2.jsonl | failed |  | inconclusive | abandon_line | False |  |
| test | r-71205e90 | exp-14 | google/gemma-3-4b-pt | 37704 | grpo | exp-08 | HF openai/gsm8k (main, train split), prompts only; difficulty weighting from data/rs_v2.jsonl | completed | 0.72 | inconclusive | adopt | True |  |
| test | r-71205e90 | exp-15 | google/gemma-3-4b-pt | 38872 | grpo | exp-14 | HF openai/gsm8k (main, train split), prompts only; difficulty weighting from data/rs_v2.jsonl | completed | 0.6876421531463229 | inconclusive | adopt | True |  |
| test | r-71205e90 | exp-16 | google/gemma-3-4b-pt | 40242 | merge | exp-15 |  | completed |  | inconclusive | adopt | True |  |
| test | r-71205e90 | exp-17 | google/gemma-3-4b-pt | 40247 | other | exp-16 |  | completed | 0.6868840030326004 | inconclusive | adopt | False | 0.6884003032600455 |
| test | r-71205e90 | exp-18 | google/gemma-3-4b-pt | 41204 | grpo | exp-15 | HF openai/gsm8k (main, train split), prompts only; difficulty weighting from data/rs_v2.jsonl | completed | 0.6823351023502654 | inconclusive | reject | False |  |
| test | r-71205e90 | exp-19 | google/gemma-3-4b-pt | 41754 | merge | exp-15 |  | completed | 0.6823351023502654 | inconclusive | reject | False |  |
| test | r-715efad9 | exp-01 | google/gemma-3-4b-pt | 64 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train) | failed |  | inconclusive | iterate | False |  |
| test | r-715efad9 | exp-02 | google/gemma-3-4b-pt | 79 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train) | completed | 0.52 | inconclusive | reject | False |  |
| test | r-715efad9 | exp-03 | google/gemma-3-4b-pt | 116 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train) | completed | 0.5 | contradicted | reject | True |  |
| test | r-715efad9 | exp-04 | google/gemma-3-4b-pt | 143 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM rows only) | completed | 0.52 | contradicted | adopt | True |  |
| test | r-715efad9 | exp-05 | google/gemma-3-4b-pt | 161 | decode-config | exp-04 |  | completed | 0.52 | contradicted | adopt | True | 0.288855193328279 |
| test | r-715efad9 | exp-06 | google/gemma-3-4b-pt | 173 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM rows only) | killed |  | inconclusive | abandon_line | True |  |
| test | r-715efad9 | exp-07 | google/gemma-3-4b-pt | 188 | sft | base_model | HF: openai/gsm8k (main, train) + microsoft/orca-math-word-problems-200k (train) + meta-math/MetaMathQA (train, GSM rows only) | killed |  | inconclusive | abandon_line | True |  |
| test | r-7778d260 | exp-01 | google/gemma-3-4b-pt | 48 | sft | base_model | meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| test | r-7778d260 | exp-02 | google/gemma-3-4b-pt | 86 | sft | base_model | meta-math/MetaMathQA | completed | 0.5 | inconclusive | adopt | False |  |
| test | r-7778d260 | exp-03 | google/gemma-3-4b-pt | 112 | other | exp-02 |  | completed | 0.5 | inconclusive | adopt | False | 0.5314632297194845 |
| test | r-80088b19 | exp-01 | google/gemma-3-4b-pt | 39 | sft | base_model | HF openai/gsm8k (config main, split train), loaded inside the script; no file written to disk | failed |  | inconclusive | iterate | False |  |
| test | r-80088b19 | exp-02 | google/gemma-3-4b-pt | 45 | sft | base_model | HF openai/gsm8k (config main, split train), loaded inside the script; no file written to disk | completed | 0.5066666666666667 | inconclusive | adopt | False | 0.46398786959818045 |
| test | r-80088b19 | exp-03 | google/gemma-3-4b-pt | 101 | sft | base_model | HF openai/gsm8k (config main, split train), loaded inside the script,HF microsoft/orca-math-word-problems-200k (split train, 200035 examples), loaded inside the script | killed |  | inconclusive | abandon_line | True |  |
| test | r-80088b19 | exp-04 | google/gemma-3-4b-pt | 134 | sft | base_model | HF openai/gsm8k (config main, split train), loaded inside the script,HF meta-math/MetaMathQA (split train, 395000 examples), loaded inside the script | killed |  | inconclusive | abandon_line | True |  |
| test | r-80088b19 | exp-05 | google/gemma-3-4b-pt | 155 | sft | base_model | HF openai/gsm8k (config main, split train), loaded inside the script | completed | 0.0 | contradicted | reject | False |  |
| test | r-80088b19 | exp-06 | google/gemma-3-4b-pt | 191 | sft | exp-02 | HF openai/gsm8k (config main, split train), loaded inside the script,HF meta-math/MetaMathQA (split train), loaded inside the script | completed | 0.38 | inconclusive | reject | True |  |
| test | r-80561db8 | exp-01 | google/gemma-3-4b-pt | 76 | sft | base_model | derived:local (openai/gsm8k train + meta-math/MetaMathQA GSM_* types) | completed | 0.38666666666666666 | inconclusive | adopt | True |  |
| test | r-80561db8 | exp-02 | google/gemma-3-4b-pt | 213 | sft | exp-01 | derived:local (openai/gsm8k train + meta-math/MetaMathQA GSM_AnsAug, GSM_Rephrased) | completed | 0.5733333333333334 | supported | adopt | True |  |
| test | r-80561db8 | exp-03 | google/gemma-3-4b-pt | 255 | grpo | exp-02 | openai/gsm8k train + meta-math/MetaMathQA GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR | completed | 0.5066666666666667 | contradicted | reject | True |  |
| test | r-80561db8 | exp-04 | google/gemma-3-4b-pt | 334 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + openai/gsm8k train gold) | killed |  | inconclusive | abandon_line | True |  |
| test | r-80561db8 | exp-05 | google/gemma-3-4b-pt | 347 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + openai/gsm8k train gold) | killed |  | inconclusive | abandon_line | False |  |
| test | r-80561db8 | exp-06 | google/gemma-3-4b-pt | 364 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + openai/gsm8k train gold) | killed |  | inconclusive | abandon_line | False |  |
| test | r-80561db8 | exp-07 | google/gemma-3-4b-pt | 372 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + openai/gsm8k train gold) | failed |  | inconclusive | abandon_line | False |  |
| test | r-80561db8 | exp-08 | google/gemma-3-4b-pt | 378 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + openai/gsm8k train gold) | killed |  | inconclusive | abandon_line | False |  |
| test | r-80561db8 | exp-09 | google/gemma-3-4b-pt | 390 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + data/sft_fewshot.jsonl) | failed |  | inconclusive | abandon_line | False |  |
| test | r-80561db8 | exp-10 | google/gemma-3-4b-pt | 399 | rft | exp-02 | synthetic:self (samples from the exp-02 checkpoint) + openai/gsm8k train gold,derived:local (data/rft_correct.jsonl + data/sft_fewshot.jsonl) | completed | 0.5066666666666667 | contradicted | reject | False |  |
| test | r-80561db8 | exp-11 | google/gemma-3-4b-pt | 450 | sft | exp-02 | derived:local (data/sft_fs10.jsonl, itself from openai/gsm8k train + meta-math/MetaMathQA GSM_AnsAug, GSM_Rephrased) | completed | 0.5266666666666666 | contradicted | reject | True |  |
| test | r-80561db8 | exp-12 | google/gemma-3-4b-pt | 493 | decode-config | exp-02 |  | completed | 0.6 | inconclusive | adopt | True | 0.5799848369977255 |
| test | r-87612f10 | exp-01 | google/gemma-3-4b-pt | 145 | sft | base_model | openai/gsm8k (config: main, split: train) | completed | 0.5117 | supported | reject | True |  |
| test | r-87612f10 | exp-02 | google/gemma-3-4b-pt | 230 | sft | base_model | openai/gsm8k (config: main, split: train) | completed | 0.5273 | inconclusive | adopt | True |  |
| test | r-87612f10 | exp-03 | google/gemma-3-4b-pt | 325 | sft | base_model | meta-math/MetaMathQA (split train), filtered to the GSM-derived types | completed |  | inconclusive | adopt | True |  |
| test | r-87612f10 | exp-04 | google/gemma-3-4b-pt | 454 | sft | exp-03 | openai/gsm8k (config: main, split: train) | completed | 0.4805 | contradicted | reject | True |  |
| test | r-87612f10 | exp-05 | google/gemma-3-4b-pt | 567 | sft | exp-02 | openai/gsm8k (config: main, split: train) | completed | 0.36 | supported | adopt | True |  |
| test | r-87612f10 | exp-06 | google/gemma-3-4b-pt | 713 | sft | exp-05 | openai/gsm8k (config: main, split: train) | completed | 0.4933333333333333 | supported | adopt | True | 0.5322213798332069 |
| test | r-8797fe5f | exp-01 | google/gemma-3-4b-pt | 268 | sft | base_model | HF openai/gsm8k (config main, split train) + HF nvidia/OpenMathInstruct-2 (splits train_5M and train_1M) | completed | 0.835 | inconclusive | adopt | False |  |
| test | r-8797fe5f | exp-02 | google/gemma-3-4b-pt | 368 | other | exp-01 |  | completed | 0.7533 | inconclusive | adopt | True | 0.7278241091736164 |
| test | r-8797fe5f | exp-03 | google/gemma-3-4b-pt | 536 | rft | exp-01 | synthetic:self (rejection-sampled from the exp-01 checkpoint) + derived:exp-01 data/sft_clean.jsonl | completed | 0.825 | inconclusive | reject | False |  |
| test | r-8797fe5f | exp-04 | google/gemma-3-4b-pt | 571 | merge | exp-03 |  | completed | 0.815 | contradicted | reject | False |  |
| test | r-8797fe5f | exp-05 | google/gemma-3-4b-pt | 657 | grpo | exp-01 | derived:exp-03 (the rejection-sampling output) + derived:exp-01 data/sft_clean.jsonl | completed | 0.815 | contradicted | reject | False |  |
| test | r-89659be7 | exp-01 | google/gemma-3-4b-pt | 90 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | failed |  | inconclusive | adopt | True |  |
| test | r-89659be7 | exp-02 | google/gemma-3-4b-pt | 169 | sft | exp-01 | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.46 | inconclusive | iterate | False |  |
| test | r-89659be7 | exp-03 | google/gemma-3-4b-pt | 235 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | killed |  | inconclusive | abandon_line | True |  |
| test | r-89659be7 | exp-04 | google/gemma-3-4b-pt | 260 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | killed |  | inconclusive | abandon_line | True |  |
| test | r-89659be7 | exp-05 | google/gemma-3-4b-pt | 281 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.06 | contradicted | reject | True |  |
| test | r-89659be7 | exp-06 | google/gemma-3-4b-pt | 395 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.22 | contradicted | reject | True |  |
| test | r-89659be7 | exp-07 | google/gemma-3-4b-pt | 490 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.26 | contradicted | adopt | True | 0.21152388172858225 |
| test | r-89659be7 | exp-08 | google/gemma-3-4b-pt | 532 | sft | base_model | local (derived from HF openai/gsm8k train + meta-math/MetaMathQA) | killed |  | inconclusive | abandon_line | True |  |
| test | r-8cbbb240 | exp-01 | google/gemma-3-4b-pt | 92 | sft | base_model | openai/gsm8k,openai/gsm8k | completed | 0.16666666666666666 | inconclusive | adopt | False |  |
| test | r-8cbbb240 | exp-02 | google/gemma-3-4b-pt | 149 | decode-config | exp-01 |  | completed | 0.3 | inconclusive | adopt | True |  |
| test | r-8cbbb240 | exp-03 | google/gemma-3-4b-pt | 181 | sft | exp-01 | openai/gsm8k + meta-math/MetaMathQA,openai/gsm8k | completed | 0.59 | inconclusive | adopt | True |  |
| test | r-8cbbb240 | exp-04 | google/gemma-3-4b-pt | 255 | sft | exp-03 | meta-math/MetaMathQA + openai/gsm8k,openai/gsm8k | completed | 0.53 | contradicted | reject | True |  |
| test | r-8cbbb240 | exp-05 | google/gemma-3-4b-pt | 296 | grpo | exp-03 | openai/gsm8k | completed |  | inconclusive | abandon_line | True |  |
| test | r-8cbbb240 | exp-06 | google/gemma-3-4b-pt | 339 | grpo | exp-03 | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-8cbbb240 | exp-07 | google/gemma-3-4b-pt | 342 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| test | r-8cbbb240 | exp-08 | google/gemma-3-4b-pt | 357 | grpo | exp-03 | openai/gsm8k | completed | 0.54 | contradicted | reject | True |  |
| test | r-8cbbb240 | exp-09 | google/gemma-3-4b-pt | 423 | rft | exp-03 | synthetic:self,derived:exp-09 rft_correct.jsonl + openai/gsm8k + meta-math/MetaMathQA,openai/gsm8k | completed | 0.5533333333333333 | supported | adopt | True |  |
| test | r-8cbbb240 | exp-10 | google/gemma-3-4b-pt | 479 | rft | exp-09 | derived:exp-09 rft_correct.jsonl + openai/gsm8k,openai/gsm8k | completed | 0.5333333333333333 | contradicted | reject | True |  |
| test | r-8cbbb240 | exp-11 | google/gemma-3-4b-pt | 510 | other | exp-09 |  | completed | 0.5533333333333333 | inconclusive | adopt | False |  |
| test | r-8cbbb240 | exp-12 | google/gemma-3-4b-pt | 529 | sft | exp-09 | derived:exp-09 rft_correct.jsonl + openai/gsm8k,openai/gsm8k | killed |  | inconclusive | abandon_line | False |  |
| test | r-8cbbb240 | exp-13 | google/gemma-3-4b-pt | 572 | merge | exp-09 |  | completed | 0.5533333333333333 | inconclusive | reject | False |  |
| test | r-8cbbb240 | exp-14 | google/gemma-3-4b-pt | 588 | other | exp-03 |  | completed | 0.52 | inconclusive | reject | False |  |
| test | r-8cbbb240 | exp-15 | google/gemma-3-4b-pt | 598 | other | exp-09 |  | completed | 0.5533333333333333 | inconclusive | adopt | False | 0.533737680060652 |
| test | r-8d0834c4 | exp-01 | google/gemma-3-4b-pt | 51 | sft | base_model | openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| test | r-8d0834c4 | exp-02 | google/gemma-3-4b-pt | 59 | merge | exp-01 |  | completed | 0.2 | supported | reject | False |  |
| test | r-8d0834c4 | exp-03 | google/gemma-3-4b-pt | 86 | sft | base_model | openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| test | r-8d0834c4 | exp-04 | google/gemma-3-4b-pt | 88 | merge | exp-03 |  | completed | 0.3 | supported | reject | False |  |
| test | r-8d0834c4 | exp-05 | google/gemma-3-4b-pt | 96 | sft | base_model | openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| test | r-8d0834c4 | exp-06 | google/gemma-3-4b-pt | 98 | merge | exp-05 |  | completed | 0.28 | contradicted | adopt | False | 0.3639120545868082 |
| test | r-8e27ac7b | exp-01 | google/gemma-3-4b-pt | 81 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | True |  |
| test | r-8e27ac7b | exp-02 | google/gemma-3-4b-pt | 94 | sft | base_model | HF:openai/gsm8k (config main, split train) | completed | 0.19 | supported | adopt | True |  |
| test | r-8e27ac7b | exp-03 | google/gemma-3-4b-pt | 136 | merge | exp-02 |  | completed |  | inconclusive | iterate | False |  |
| test | r-8e27ac7b | exp-04 | google/gemma-3-4b-pt | 139 | merge | exp-02 |  | completed |  | inconclusive | iterate | False |  |
| test | r-8e27ac7b | exp-05 | google/gemma-3-4b-pt | 175 | merge | exp-02 |  | completed | 0.19 | supported | reject | True |  |
| test | r-8e27ac7b | exp-06 | google/gemma-3-4b-pt | 196 | sft | base_model | HF:openai/gsm8k (config main, split train) | killed | 0.3 | supported | adopt | True |  |
| test | r-8e27ac7b | exp-07 | google/gemma-3-4b-pt | 229 | merge | exp-06 |  | completed | 0.3 | supported | adopt | False |  |
| test | r-8e27ac7b | exp-08 | google/gemma-3-4b-pt | 264 | other | exp-07 |  | completed | 0.28 | inconclusive | adopt | False | 0.3146322971948446 |
| test | r-9ed0b404 | exp-01 | google/gemma-3-4b-pt | 64 | sft | base_model | gsm8k (config: main, split: train) loaded in-process by the training script | killed |  | inconclusive | abandon_line | False |  |
| test | r-9ed0b404 | exp-02 | google/gemma-3-4b-pt | 82 | sft | base_model | gsm8k (config: main, split: train) loaded in-process by the training script | completed |  | inconclusive | adopt | False |  |
| test | r-9ed0b404 | exp-03 | google/gemma-3-4b-pt | 114 | merge | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| test | r-9ed0b404 | exp-04 | google/gemma-3-4b-pt | 162 | merge | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| test | r-9ed0b404 | exp-05 | google/gemma-3-4b-pt | 294 | rft | exp-04 | synthetic:self | completed |  | inconclusive | adopt | False |  |
| test | r-9ed0b404 | exp-06 | google/gemma-3-4b-pt | 350 | merge | exp-05 |  | completed | 0.4874905231235785 | inconclusive | adopt | False | 0.49583017437452614 |
| test | r-a870812c | exp-01 | google/gemma-3-4b-pt | 60 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | abandon_line | False |  |
| test | r-a870812c | exp-02 | google/gemma-3-4b-pt | 81 | sft | base_model | openai/gsm8k (config: main, split: train) | completed | 0.18 | inconclusive | adopt | False |  |
| test | r-a870812c | exp-03 | google/gemma-3-4b-pt | 108 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | adopt | True |  |
| test | r-a870812c | exp-04 | google/gemma-3-4b-pt | 133 | merge | exp-03 |  | completed | 0.267 | supported | adopt | False |  |
| test | r-a870812c | exp-05 | google/gemma-3-4b-pt | 142 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-a870812c | exp-06 | google/gemma-3-4b-pt | 153 | sft | base_model | openai/gsm8k (config: main, split: train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-a870812c | exp-07 | google/gemma-3-4b-pt | 193 | merge | exp-06 |  | failed |  | inconclusive | abandon_line | False |  |
| test | r-a870812c | exp-08 | google/gemma-3-4b-pt | 218 | merge | exp-03 |  | completed |  | inconclusive | adopt | False | 0.2137983320697498 |
| test | r-a88bb78b | exp-01 | google/gemma-3-4b-pt | 194 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA (GSM-derived subset) | killed |  | inconclusive | abandon_line | False |  |
| test | r-a88bb78b | exp-02 | google/gemma-3-4b-pt | 238 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA (GSM-derived subset) | killed |  | inconclusive | abandon_line | True |  |
| test | r-a88bb78b | exp-03 | google/gemma-3-4b-pt | 287 | sft | base_model | derived: openai/gsm8k train + meta-math/MetaMathQA (GSM-derived subset) | completed | 0.115 | inconclusive | adopt | False |  |
| test | r-a88bb78b | exp-04 | google/gemma-3-4b-pt | 463 | sft | exp-03 | derived:exp-03 data (data/sft.jsonl) + openai/gsm8k train few-shot pool | killed |  | inconclusive | abandon_line | True |  |
| test | r-a88bb78b | exp-05 | google/gemma-3-4b-pt | 479 | sft | exp-03 | derived:exp-03 data (data/sft.jsonl) + openai/gsm8k train few-shot pool | completed | 0.7 | supported | adopt | True |  |
| test | r-a88bb78b | exp-06 | google/gemma-3-4b-pt | 531 | other | exp-05 |  | completed | 0.7 | inconclusive | reject | False |  |
| test | r-a88bb78b | exp-07 | google/gemma-3-4b-pt | 570 | rft | exp-05 | synthetic:self (rejection-sampled from the exp-05 checkpoint) + derived:exp-03 data,synthetic:self (generated by the exp-05 checkpoint) | completed | 0.702 | supported | adopt | True |  |
| test | r-a88bb78b | exp-08 | google/gemma-3-4b-pt | 623 | other | exp-07 |  | completed | 0.702 | inconclusive | adopt | False | 0.7179681576952237 |
| test | r-a88bb78b | exp-09 | google/gemma-3-4b-pt | 664 | rft | exp-05 | synthetic:self (rejection-sampled from the exp-07 checkpoint) + derived:exp-03 data,synthetic:self (generated by the exp-07 checkpoint) | completed | 0.7 | contradicted | reject | True |  |
| test | r-aa43ab4f | exp-01 | google/gemma-3-4b-pt | 96 | sft | base_model | HF openai/gsm8k (config main, train split) | completed | 0.36 | inconclusive | adopt | False |  |
| test | r-aa43ab4f | exp-02 | google/gemma-3-4b-pt | 147 | sft | base_model | HF meta-math/MetaMathQA (GSM_* subsets) + HF openai/gsm8k (main, train split) | completed | 0.02666666666666667 | contradicted | adopt | False |  |
| test | r-aa43ab4f | exp-03 | google/gemma-3-4b-pt | 221 | sft | exp-02 | HF openai/gsm8k (main, train split) | completed | 0.49333333333333335 | supported | adopt | True |  |
| test | r-aa43ab4f | exp-04 | google/gemma-3-4b-pt | 245 | sft | exp-03 | HF meta-math/MetaMathQA (GSM_* subsets) + HF openai/gsm8k (main, train split) | completed | 0.66 | supported | adopt | False |  |
| test | r-aa43ab4f | exp-05 | google/gemma-3-4b-pt | 264 | sft | exp-04 | None | completed | 0.58 | contradicted | reject | False |  |
| test | r-aa43ab4f | exp-06 | google/gemma-3-4b-pt | 281 | other | exp-04 |  | completed | 0.6557998483699773 | inconclusive | adopt | False | 0.6535253980288097 |
| test | r-adcf5bfb | exp-01 | google/gemma-3-4b-pt | 797 | rft | base_model | synthetic:self | completed | 0.56 | supported | reject | False |  |
| test | r-adcf5bfb | exp-02 | google/gemma-3-4b-pt | 941 | rft | base_model | synthetic:self | completed | 0.52 | supported | iterate | True |  |
| test | r-adcf5bfb | exp-03 | google/gemma-3-4b-pt | 1146 | rft | base_model | synthetic:self | completed | 0.603 | inconclusive | adopt | False |  |
| test | r-adcf5bfb | exp-04 | google/gemma-3-4b-pt | 1173 | other | exp-03 |  | completed | 0.5466666666666666 | supported | adopt | False |  |
| test | r-adcf5bfb | exp-05 | google/gemma-3-4b-pt | 1199 | decode-config | exp-04 |  | completed | 0.58 | supported | adopt | False | 0.5784685367702805 |
| test | r-adcf5bfb | exp-06 | google/gemma-3-4b-pt | 1324 | grpo | exp-03 | HF id openai/gsm8k | killed | 0.599 | inconclusive | reject | False |  |
| test | r-ae9c94e6 | exp-01 | google/gemma-3-4b-pt | 79 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | completed | 0.23333333333333334 | inconclusive | adopt | False |  |
| test | r-ae9c94e6 | exp-02 | google/gemma-3-4b-pt | 127 | sft | base_model | openai/gsm8k,meta-math/MetaMathQA | killed | 0.0 | inconclusive | abandon_line | True |  |
| test | r-ae9c94e6 | exp-03 | google/gemma-3-4b-pt | 173 | merge | exp-01 |  | completed | 0.2 | inconclusive | reject | True |  |
| test | r-ae9c94e6 | exp-04 | google/gemma-3-4b-pt | 188 | merge | exp-01 |  | completed | 0.32666666666666666 | inconclusive | adopt | True | 0.33965125094768767 |
| test | r-ae9c94e6 | exp-05 | google/gemma-3-4b-pt | 193 | merge | exp-01 |  | completed | 0.2 | inconclusive | reject | False |  |
| test | r-b7c1e8a9 | exp-01 | google/gemma-3-4b-pt | 77 | sft | base_model | derived: HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (GSM_* types) | failed |  | inconclusive | abandon_line | False |  |
| test | r-b7c1e8a9 | exp-02 | google/gemma-3-4b-pt | 88 | sft | base_model | derived: HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (GSM_* types) | killed |  | inconclusive | abandon_line | False |  |
| test | r-b7c1e8a9 | exp-03 | google/gemma-3-4b-pt | 107 | sft | base_model | derived: HF openai/gsm8k (main, train) + HF meta-math/MetaMathQA (GSM_* types) | completed | 0.54 | inconclusive | adopt | False | 0.5193328278999242 |
| test | r-b7c1e8a9 | exp-04 | google/gemma-3-4b-pt | 223 | sft | base_model | derived: HF openai/gsm8k (main, train) x5 + HF meta-math/MetaMathQA (GSM_* types) | killed |  | inconclusive | abandon_line | True |  |
| test | r-b7c1e8a9 | exp-05 | google/gemma-3-4b-pt | 236 | sft | base_model | derived: HF openai/gsm8k (main, train) x5 + HF meta-math/MetaMathQA (GSM_* types) | completed | 0.113 | contradicted | reject | True |  |
| test | r-b7c1e8a9 | exp-06 | google/gemma-3-4b-pt | 292 | sft | exp-03 | HF openai/gsm8k (main, train) | completed | 0.153 | contradicted | reject | True |  |
| test | r-b8635438 | exp-01 | google/gemma-3-4b-pt | 49 | sft | base_model | HF:openai/gsm8k (config 'main', train split), loaded in-script | failed |  | inconclusive | abandon_line | True |  |
| test | r-b8635438 | exp-02 | google/gemma-3-4b-pt | 57 | sft | base_model | HF:openai/gsm8k (config 'main', train split), loaded in-script | killed |  | inconclusive | abandon_line | True |  |
| test | r-b8635438 | exp-03 | google/gemma-3-4b-pt | 92 | sft | base_model | HF:openai/gsm8k (config 'main', train split), loaded in-script | completed |  | inconclusive | adopt | False |  |
| test | r-b8635438 | exp-04 | google/gemma-3-4b-pt | 108 | merge | exp-03 |  | completed |  | inconclusive | abandon_line | False |  |
| test | r-b8635438 | exp-05 | google/gemma-3-4b-pt | 126 | merge | exp-03 |  | completed | 0.1875 | inconclusive | reject | False |  |
| test | r-b8635438 | exp-06 | google/gemma-3-4b-pt | 140 | sft | base_model | HF:openai/gsm8k (config 'main', train split), loaded in-script | completed |  | inconclusive | adopt | True |  |
| test | r-b8635438 | exp-07 | google/gemma-3-4b-pt | 163 | merge | exp-06 |  | completed | 0.3 | supported | adopt | True |  |
| test | r-b8635438 | exp-08 | google/gemma-3-4b-pt | 178 | other | exp-07 |  | completed | 0.36 | inconclusive | adopt | True | 0.35633055344958303 |
| test | r-bcdec442 | exp-01 | google/gemma-3-4b-pt | 79 | sft | base_model | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
| test | r-bcdec442 | exp-02 | google/gemma-3-4b-pt | 95 | sft | base_model | openai/gsm8k | completed | 0.6133333333333333 | inconclusive | adopt | True |  |
| test | r-bcdec442 | exp-03 | google/gemma-3-4b-pt | 146 | merge | exp-02 |  | completed | 0.38 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-04 | google/gemma-3-4b-pt | 151 | merge | exp-02 |  | completed | 0.4 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-05 | google/gemma-3-4b-pt | 169 | sft | base_model | openai/gsm8k | completed | 0.46 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-06 | google/gemma-3-4b-pt | 195 | other | exp-05 |  | completed | 0.48 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-07 | google/gemma-3-4b-pt | 201 | sft | base_model | openai/gsm8k | completed | 0.5466666666666666 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-08 | google/gemma-3-4b-pt | 246 | merge | exp-07 |  | completed | 0.38 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-09 | google/gemma-3-4b-pt | 255 | other | exp-02 |  | completed | 0.46 | inconclusive | reject | False |  |
| test | r-bcdec442 | exp-10 | google/gemma-3-4b-pt | 266 | sft | base_model | openai/gsm8k | completed | 0.52 | contradicted | reject | True |  |
| test | r-bcdec442 | exp-11 | google/gemma-3-4b-pt | 307 | other | exp-07 |  | completed | 0.5333333333333333 | inconclusive | reject | True |  |
| test | r-bcdec442 | exp-12 | google/gemma-3-4b-pt | 344 | decode-config | exp-02 |  | completed | 0.6133333333333333 | supported | adopt | True | 0.5610310841546626 |
| test | r-c3d185ea | exp-01 | google/gemma-3-4b-pt | 127 | sft | base_model | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | failed |  | inconclusive | abandon_line | True |  |
| test | r-c3d185ea | exp-02 | google/gemma-3-4b-pt | 132 | sft | base_model | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.23 | inconclusive | adopt | True |  |
| test | r-c3d185ea | exp-03 | google/gemma-3-4b-pt | 294 | sft | exp-02 | derived: openai/gsm8k (train) | completed | 0.61 | supported | adopt | True |  |
| test | r-c3d185ea | exp-04 | google/gemma-3-4b-pt | 329 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| test | r-c3d185ea | exp-05 | google/gemma-3-4b-pt | 338 | decode-config | exp-04 |  | completed | 0.664 | inconclusive | adopt | False |  |
| test | r-c3d185ea | exp-06 | google/gemma-3-4b-pt | 356 | sft | exp-03 | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | failed |  | inconclusive | abandon_line | False |  |
| test | r-c3d185ea | exp-07 | google/gemma-3-4b-pt | 371 | sft | exp-03 | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | completed | 0.68 | inconclusive | adopt | False |  |
| test | r-c3d185ea | exp-08 | google/gemma-3-4b-pt | 470 | other | exp-07 |  | completed | 0.685 | inconclusive | adopt | False |  |
| test | r-c3d185ea | exp-09 | google/gemma-3-4b-pt | 491 | sft | exp-07 | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | failed |  | inconclusive | abandon_line | False |  |
| test | r-c3d185ea | exp-10 | google/gemma-3-4b-pt | 539 | decode-config | exp-08 |  | completed | 0.602 | inconclusive | reject | False |  |
| test | r-c3d185ea | exp-11 | google/gemma-3-4b-pt | 541 | sft | exp-07 | derived: openai/gsm8k (train) + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | completed | 0.578 | contradicted | reject | False |  |
| test | r-c3d185ea | exp-12 | google/gemma-3-4b-pt | 593 | decode-config | exp-10 |  | completed | 0.67 | inconclusive | adopt | False | 0.6603487490523123 |
| test | r-c95a5e9a | exp-01 | google/gemma-3-4b-pt | 121 | sft | base_model | openai/gsm8k (config: main, split: train) | completed | 0.4866666666666667 | inconclusive | reject | False |  |
| test | r-c95a5e9a | exp-02 | google/gemma-3-4b-pt | 192 | sft | base_model | deepmind/aqua_rat (split: train) | failed |  | inconclusive | iterate | False |  |
| test | r-c95a5e9a | exp-03 | google/gemma-3-4b-pt | 196 | sft | base_model | deepmind/aqua_rat (split: train) | completed |  | inconclusive | adopt | False |  |
| test | r-c95a5e9a | exp-04 | google/gemma-3-4b-pt | 205 | sft | exp-03 | openai/gsm8k (config: main, split: train) | completed | 0.5133333333333333 | supported | adopt | False |  |
| test | r-c95a5e9a | exp-05 | google/gemma-3-4b-pt | 225 | sft | exp-03 | openai/gsm8k (config: main, split: train) | completed | 0.5 | contradicted | reject | False |  |
| test | r-c95a5e9a | exp-06 | google/gemma-3-4b-pt | 290 | sft | exp-03 | openai/gsm8k (config: main, split: train) | completed | 0.52 | supported | adopt | False |  |
| test | r-c95a5e9a | exp-07 | google/gemma-3-4b-pt | 379 | sft | exp-06 | openai/gsm8k (config: main, split: train) | completed | 0.52 | inconclusive | reject | False |  |
| test | r-c95a5e9a | exp-08 | google/gemma-3-4b-pt | 517 | other | exp-06 |  | completed | 0.56 | inconclusive | adopt | False |  |
| test | r-c95a5e9a | exp-09 | google/gemma-3-4b-pt | 647 | sft | base_model | meta-math/MetaMathQA (split: train), MATH_* rows only | completed |  | inconclusive | adopt | False |  |
| test | r-c95a5e9a | exp-10 | google/gemma-3-4b-pt | 739 | sft | exp-09 | openai/gsm8k (config: main, split: train) | completed | 0.3692191053828658 | contradicted | reject | False |  |
| test | r-c95a5e9a | exp-11 | google/gemma-3-4b-pt | 850 | other | exp-04 |  | completed | 0.5 | inconclusive | adopt | False | 0.4988627748294162 |
| test | r-cf7852ab | exp-01 | google/gemma-3-4b-pt | 40 | sft | base_model | HF id openai/gsm8k | failed |  | inconclusive | iterate | False |  |
| test | r-cf7852ab | exp-02 | google/gemma-3-4b-pt | 48 | sft | base_model | HF id openai/gsm8k | failed |  | inconclusive | iterate | False |  |
| test | r-cf7852ab | exp-03 | google/gemma-3-4b-pt | 52 | sft | base_model | HF id openai/gsm8k | completed |  | inconclusive | adopt | False |  |
| test | r-cf7852ab | exp-04 | google/gemma-3-4b-pt | 62 | merge | exp-03 |  | completed | 0.54 | inconclusive | adopt | False | 0.029567854435178165 |
| test | r-cf7852ab | exp-05 | google/gemma-3-4b-pt | 90 | sft | base_model | HF id openai/gsm8k | killed |  | inconclusive | abandon_line | False |  |
| test | r-d46f982f | exp-01 | google/gemma-3-4b-pt | 629 | sft | base_model | HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows) + HF openai/gsm8k (config main, split train) | completed | 0.665 | inconclusive | adopt | False |  |
| test | r-d46f982f | exp-02 | google/gemma-3-4b-pt | 1062 | decode-config | exp-01 |  | completed | 0.665 | supported | adopt | False |  |
| test | r-d46f982f | exp-03 | google/gemma-3-4b-pt | 1323 | grpo | exp-02 | local | failed |  | inconclusive | abandon_line | False |  |
| test | r-d46f982f | exp-04 | google/gemma-3-4b-pt | 1432 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| test | r-d46f982f | exp-05 | google/gemma-3-4b-pt | 1437 | grpo | exp-04 | local | killed | 0.735 | supported | adopt | False |  |
| test | r-d46f982f | exp-06 | google/gemma-3-4b-pt | 1526 | other | exp-05 |  | completed |  | inconclusive | adopt | False |  |
| test | r-d46f982f | exp-07 | google/gemma-3-4b-pt | 1564 | grpo | exp-05 | local | killed | 0.68 | contradicted | reject | False |  |
| test | r-d46f982f | exp-08 | google/gemma-3-4b-pt | 1568 | other | exp-06 |  | completed | 0.7125 | inconclusive | adopt | False |  |
| test | r-d46f982f | exp-09 | google/gemma-3-4b-pt | 1689 | merge | exp-05 |  | completed |  | inconclusive | reject | False |  |
| test | r-d46f982f | exp-10 | google/gemma-3-4b-pt | 1711 | merge | exp-05 |  | completed | 0.73 | supported | adopt | False |  |
| test | r-d46f982f | exp-11 | google/gemma-3-4b-pt | 1725 | merge | exp-05 |  | completed |  | inconclusive | reject | False |  |
| test | r-d46f982f | exp-12 | google/gemma-3-4b-pt | 1736 | other | exp-10 |  | completed | 0.7375 | supported | adopt | False | 0.7012888551933283 |
| test | r-d5b65920 | exp-01 | google/gemma-3-4b-pt | 44 | sft | base_model | openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | False |  |
| test | r-d5b65920 | exp-02 | google/gemma-3-4b-pt | 110 | merge | exp-01 |  | completed | 0.36 | inconclusive | adopt | False |  |
| test | r-d5b65920 | exp-03 | google/gemma-3-4b-pt | 126 | sft | base_model | openai/gsm8k (config main, split train),meta-math/MetaMathQA (split train) | killed |  | inconclusive | abandon_line | False |  |
| test | r-d5b65920 | exp-04 | google/gemma-3-4b-pt | 138 | sft | base_model | openai/gsm8k (config main, split train),meta-math/MetaMathQA (split train) | completed |  | inconclusive | adopt | False |  |
| test | r-d5b65920 | exp-05 | google/gemma-3-4b-pt | 164 | merge | exp-04 |  | completed | 0.32 | inconclusive | adopt | False |  |
| test | r-d5b65920 | exp-06 | google/gemma-3-4b-pt | 172 | other | exp-05 |  | completed |  | inconclusive | adopt | False | 0.27369219105382864 |
| test | r-e3d4334d | exp-01 | google/gemma-3-4b-pt | 16409 | sft | base_model | openai/gsm8k (config: main, split: train) | failed |  | inconclusive | iterate | False |  |
| test | r-e3d4334d | exp-02 | google/gemma-3-4b-pt | 16920 | sft | base_model | openai/gsm8k (config: main, split: train) | failed |  | inconclusive | iterate | True |  |
| test | r-e3d4334d | exp-03 | google/gemma-3-4b-pt | 20249 | sft | base_model | openai/gsm8k (config: main, split: train) | completed | 0.6 | inconclusive | adopt | False |  |
| test | r-e3d4334d | exp-04 | google/gemma-3-4b-pt | 24851 | sft | base_model | local mixture of meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) and openai/gsm8k (config: main, split: train) | killed |  | inconclusive | iterate | True |  |
| test | r-e3d4334d | exp-05 | google/gemma-3-4b-pt | 25405 | other | exp-03 |  | completed |  | inconclusive | iterate | True |  |
| test | r-e3d4334d | exp-06 | google/gemma-3-4b-pt | 26613 | sft | base_model | local mixture of meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) and openai/gsm8k (config: main, split: train) | killed |  | inconclusive | iterate | False |  |
| test | r-e3d4334d | exp-07 | google/gemma-3-4b-pt | 27473 | sft | base_model | local mixture of meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) and openai/gsm8k (config: main, split: train) | failed |  | inconclusive | iterate | True |  |
| test | r-e3d4334d | exp-08 | google/gemma-3-4b-pt | 27762 | sft | base_model | local mixture of meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) and openai/gsm8k (config: main, split: train) | completed | 0.6033333333333334 | inconclusive | adopt | True |  |
| test | r-e3d4334d | exp-09 | google/gemma-3-4b-pt | 32370 | rft | exp-08 | derived:exp-08 (self-generated) + openai/gsm8k (config: main, split: train) + meta-math/MetaMathQA (GSM_AnsAug),derived:exp-08 (synthetic:self, verified against openai/gsm8k train gold) | completed | 0.63 | supported | adopt | True |  |
| test | r-e3d4334d | exp-10 | google/gemma-3-4b-pt | 35592 | dpo | exp-09 | derived:exp-09 (synthetic:self, graded against openai/gsm8k train gold) | completed | 0.6266666666666667 | contradicted | reject | True |  |
| test | r-e3d4334d | exp-11 | google/gemma-3-4b-pt | 39270 | rft | exp-09 | derived:exp-09 (self-generated) + openai/gsm8k (config: main, split: train) + meta-math/MetaMathQA (GSM_AnsAug),derived:exp-09 (synthetic:self, verified against meta-math/MetaMathQA GSM_Rephrased answers) | completed | 0.5933333333333334 | contradicted | reject | True |  |
| test | r-e3d4334d | exp-12 | google/gemma-3-4b-pt | 39610 | other | exp-09 |  | completed | 0.6027293404094011 | inconclusive | adopt | True | 0.6072782410917361 |
| test | r-e78f871a | exp-01 | google/gemma-3-4b-pt | 121 | sft | base_model | openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| test | r-e78f871a | exp-02 | google/gemma-3-4b-pt | 186 | merge | exp-01 |  | completed | 0.44 | inconclusive | adopt | True |  |
| test | r-e78f871a | exp-03 | google/gemma-3-4b-pt | 211 | sft | exp-02 | openai/gsm8k (main, train),openai/gsm8k (socratic, train),microsoft/orca-math-word-problems-200k (train[:30000]) | completed |  | inconclusive | adopt | True |  |
| test | r-e78f871a | exp-04 | google/gemma-3-4b-pt | 319 | merge | exp-03 |  | failed |  | inconclusive | abandon_line | True |  |
| test | r-e78f871a | exp-05 | google/gemma-3-4b-pt | 325 | merge | exp-03 |  | completed | 0.41 | contradicted | reject | True |  |
| test | r-e78f871a | exp-06 | google/gemma-3-4b-pt | 333 | sft | exp-02 | openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-e78f871a | exp-07 | google/gemma-3-4b-pt | 338 | sft | exp-02 | openai/gsm8k (main, train) | completed |  | inconclusive | adopt | True |  |
| test | r-e78f871a | exp-08 | google/gemma-3-4b-pt | 361 | merge | exp-07 |  | completed | 0.42 | contradicted | reject | True |  |
| test | r-e78f871a | exp-09 | google/gemma-3-4b-pt | 367 | merge | exp-01 |  | completed | 0.4066666666666667 | inconclusive | adopt | True | 0.4131918119787718 |
| test | r-ec5ee8c8 | exp-01 | google/gemma-3-4b-pt | 123 | sft | base_model | HF id openai/gsm8k:main | completed |  | inconclusive | adopt | True |  |
| test | r-ec5ee8c8 | exp-02 | google/gemma-3-4b-pt | 153 | merge | exp-01 |  | completed | 0.48 | inconclusive | reject | True |  |
| test | r-ec5ee8c8 | exp-03 | google/gemma-3-4b-pt | 164 | sft | base_model | HF id openai/gsm8k:main,HF id openai/gsm8k:socratic | completed |  | inconclusive | adopt | True |  |
| test | r-ec5ee8c8 | exp-04 | google/gemma-3-4b-pt | 197 | merge | exp-03 |  | completed | 0.52 | supported | adopt | True |  |
| test | r-ec5ee8c8 | exp-05 | google/gemma-3-4b-pt | 223 | other | exp-04 |  | completed | 0.5 | inconclusive | adopt | True | 0.48218347232752085 |
| test | r-ed3e090c | exp-01 | google/gemma-3-4b-pt | 87 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM types) | completed |  | inconclusive | adopt | False |  |
| test | r-ed3e090c | exp-02 | google/gemma-3-4b-pt | 168 | merge | exp-01 |  | completed | 0.36666666666666664 | inconclusive | adopt | False |  |
| test | r-ed3e090c | exp-03 | google/gemma-3-4b-pt | 208 | decode-config | exp-02 |  | completed | 0.32666666666666666 | contradicted | reject | False |  |
| test | r-ed3e090c | exp-04 | google/gemma-3-4b-pt | 233 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM types) + nvidia/OpenMathInstruct-2 (problem_source=augmented_gsm8k) | completed |  | inconclusive | adopt | False |  |
| test | r-ed3e090c | exp-05 | google/gemma-3-4b-pt | 249 | merge | exp-04 |  | completed | 0.32666666666666666 | contradicted | adopt | False |  |
| test | r-ed3e090c | exp-06 | google/gemma-3-4b-pt | 254 | decode-config | exp-05 |  | completed | 0.29333333333333333 | contradicted | reject | False |  |
| test | r-ed3e090c | exp-07 | google/gemma-3-4b-pt | 268 | decode-config | exp-06 |  | completed | 0.32 | inconclusive | reject | False |  |
| test | r-ed3e090c | exp-08 | google/gemma-3-4b-pt | 276 | decode-config | exp-07 |  | completed | 0.26666666666666666 | contradicted | reject | False |  |
| test | r-ed3e090c | exp-09 | google/gemma-3-4b-pt | 281 | decode-config | exp-08 |  | completed | 0.24 | contradicted | reject | False |  |
| test | r-ed3e090c | exp-10 | google/gemma-3-4b-pt | 300 | merge | exp-01 |  | completed | 0.37333333333333335 | inconclusive | adopt | True |  |
| test | r-ed3e090c | exp-11 | google/gemma-3-4b-pt | 316 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM types) | completed |  | inconclusive | adopt | False |  |
| test | r-ed3e090c | exp-12 | google/gemma-3-4b-pt | 337 | other | exp-11 |  | completed | 0.04 | contradicted | reject | False |  |
| test | r-ed3e090c | exp-13 | google/gemma-3-4b-pt | 355 | merge | exp-01 |  | completed | 0.3333333333333333 | inconclusive | adopt | True | 0.3570887035633055 |
| test | r-ee7eb0ec | exp-01 | google/gemma-3-4b-pt | 63 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train) | failed |  | inconclusive | iterate | True |  |
| test | r-ee7eb0ec | exp-02 | google/gemma-3-4b-pt | 75 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train) | killed |  | inconclusive | abandon_line | True |  |
| test | r-ee7eb0ec | exp-03 | google/gemma-3-4b-pt | 109 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | completed | 0.16 | inconclusive | reject | True |  |
| test | r-ee7eb0ec | exp-04 | google/gemma-3-4b-pt | 228 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | failed |  | inconclusive | iterate | True |  |
| test | r-ee7eb0ec | exp-05 | google/gemma-3-4b-pt | 277 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | failed |  | inconclusive | adopt | True |  |
| test | r-ee7eb0ec | exp-06 | google/gemma-3-4b-pt | 345 | merge | exp-05 |  | completed | 0.26666666666666666 | inconclusive | reject | True |  |
| test | r-ee7eb0ec | exp-07 | google/gemma-3-4b-pt | 392 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | completed | 0.03333333333333333 | contradicted | reject | True |  |
| test | r-ee7eb0ec | exp-08 | google/gemma-3-4b-pt | 430 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | completed | 0.06666666666666667 | contradicted | adopt | True | 0.07278241091736164 |
| test | r-ee7eb0ec | exp-09 | google/gemma-3-4b-pt | 480 | sft | base_model | HF: openai/gsm8k (main, train) + meta-math/MetaMathQA (train, GSM subset) | killed |  | inconclusive | abandon_line | True |  |
| test | r-f79e3d68 | exp-01 | google/gemma-3-4b-pt | 287 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.427 | inconclusive | adopt | False |  |
| test | r-f79e3d68 | exp-02 | google/gemma-3-4b-pt | 350 | decode-config | exp-01 |  | completed | 0.327 | contradicted | reject | True |  |
| test | r-f79e3d68 | exp-03 | google/gemma-3-4b-pt | 401 | sft | base_model | local (openai/gsm8k train via prep_data.py + meta-math/MetaMathQA GSM_* via prep_meta.py) | completed | 0.287 | contradicted | reject | True |  |
| test | r-f79e3d68 | exp-04 | google/gemma-3-4b-pt | 720 | sft | base_model | openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | True |  |
| test | r-f79e3d68 | exp-05 | google/gemma-3-4b-pt | 753 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.48 | supported | adopt | True |  |
| test | r-f79e3d68 | exp-06 | google/gemma-3-4b-pt | 829 | other | exp-05 |  | completed | 0.487 | inconclusive | adopt | False | 0.46626231993934797 |
| test | r-f79e3d68 | exp-07 | google/gemma-3-4b-pt | 848 | sft | base_model | local (openai/gsm8k train + meta-math/MetaMathQA GSM_* via work/meta_gsm.jsonl) | completed | 0.407 | contradicted | reject | True |  |
| test | r-f9a8a01b | exp-01 | google/gemma-3-4b-pt | 131 | sft | base_model | local: derived from HF openai/gsm8k (train) + meta-math/MetaMathQA | completed | 0.22 | inconclusive | adopt | True |  |
| test | r-f9a8a01b | exp-02 | google/gemma-3-4b-pt | 317 | merge | exp-01 |  | completed | 0.22 | inconclusive | reject | False |  |
| test | r-f9a8a01b | exp-03 | google/gemma-3-4b-pt | 365 | sft | base_model | HF openai/gsm8k (train split) | completed | 0.32 | supported | adopt | True |  |
| test | r-f9a8a01b | exp-04 | google/gemma-3-4b-pt | 528 | merge | exp-03 |  | completed | 0.32 | supported | reject | True |  |
| test | r-f9a8a01b | exp-05 | google/gemma-3-4b-pt | 540 | sft | base_model | HF openai/gsm8k (train split) | completed | 0.5333333333333333 | supported | adopt | False |  |
| test | r-f9a8a01b | exp-06 | google/gemma-3-4b-pt | 979 | merge | exp-05 |  | completed | 0.5333333333333333 | supported | reject | False |  |
| test | r-f9a8a01b | exp-07 | google/gemma-3-4b-pt | 1005 | sft | exp-05 | HF openai/gsm8k (train split) | completed | 0.5666666666666667 | inconclusive | adopt | True |  |
| test | r-f9a8a01b | exp-08 | google/gemma-3-4b-pt | 1299 | merge | exp-07 |  | completed | 0.5666666666666667 | inconclusive | reject | False |  |
| test | r-f9a8a01b | exp-09 | google/gemma-3-4b-pt | 1337 | sft | exp-07 | HF openai/gsm8k (train split) | completed | 0.54 | inconclusive | adopt | True |  |
| test | r-f9a8a01b | exp-10 | google/gemma-3-4b-pt | 1462 | merge | exp-09 |  | completed | 0.54 | inconclusive | adopt | False |  |
| test | r-f9a8a01b | exp-11 | google/gemma-3-4b-pt | 1484 | merge | exp-07 |  | completed | 0.5231235784685367 | inconclusive | adopt | True |  |
| test | r-f9a8a01b | exp-12 | google/gemma-3-4b-pt | 1598 | other | exp-11 |  | completed | 0.5 | inconclusive | adopt | False |  |
| test | r-f9a8a01b | exp-13 | google/gemma-3-4b-pt | 1651 | merge | exp-05 |  | completed | 0.45716451857467777 | contradicted | reject | True |  |
| test | r-f9a8a01b | exp-14 | google/gemma-3-4b-pt | 1760 | merge |  |  | completed | 0.5367702805155421 | supported | adopt | True | 0.5344958301743745 |
| test | r-fac8a9cc | exp-01 | google/gemma-3-4b-pt | 118 | sft | base_model | HF openai/gsm8k (main, train split) + nvidia/OpenMathInstruct-2 + microsoft/orca-math-word-problems-200k, all read from the local hf_cache parquet shards | completed | 0.6533333333333333 | inconclusive | adopt | True |  |
| test | r-fac8a9cc | exp-02 | google/gemma-3-4b-pt | 438 | sft | exp-01 | HF nvidia/OpenMathInstruct-2 shards 8-15 (unique problems only) + openai/gsm8k (main, train) x2 | completed | 0.6266666666666667 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-03 | google/gemma-3-4b-pt | 686 | grpo | exp-01 | HF openai/gsm8k (main, train split), prompts and gold answers only | killed |  | inconclusive | abandon_line | True |  |
| test | r-fac8a9cc | exp-04 | google/gemma-3-4b-pt | 701 | grpo | exp-01 | HF openai/gsm8k (main, train split), prompts and gold answers only | completed | 0.6733333333333333 | supported | adopt | True |  |
| test | r-fac8a9cc | exp-05 | google/gemma-3-4b-pt | 786 | rft | exp-04 | synthetic:self (run_grpo1b/model-step-100) + HF openai/gsm8k (main, train) human solutions | completed | 0.6368460955269143 | supported | adopt | True |  |
| test | r-fac8a9cc | exp-06 | google/gemma-3-4b-pt | 829 | sft | base_model | HF meta-math/MetaMathQA (GSM_Rephrased / GSM_SV / GSM_FOBAR / GSM_AnsAug) + openai/gsm8k (main, train) x3 | completed | 0.6 | inconclusive | reject | True |  |
| test | r-fac8a9cc | exp-07 | google/gemma-3-4b-pt | 999 | grpo | exp-04 | derived:exp-05 (data_rft1 acceptance counts) over HF openai/gsm8k (main, train) prompts | completed | 0.6209249431387415 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-08 | google/gemma-3-4b-pt | 1031 | grpo | exp-01 | HF openai/gsm8k (main, train split), prompts and gold answers only | completed | 0.6269901440485216 | inconclusive | reject | True |  |
| test | r-fac8a9cc | exp-09 | google/gemma-3-4b-pt | 1063 | merge | exp-04 |  | completed | 0.6 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-10 | google/gemma-3-4b-pt | 1066 | merge | exp-04 |  | completed | 0.66 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-11 | google/gemma-3-4b-pt | 1088 | merge | exp-04 |  | completed | 0.58 | contradicted | reject | False |  |
| test | r-fac8a9cc | exp-12 | google/gemma-3-4b-pt | 1095 | decode-config | exp-04 |  | completed | 0.66 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-13 | google/gemma-3-4b-pt | 1105 | grpo | exp-01 | HF openai/gsm8k (main, train split), prompts and gold answers only | completed | 0.6 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-14 | google/gemma-3-4b-pt | 1149 | sft | exp-04 | HF openai/gsm8k (main, train split) x4, human solutions only | completed | 0.58 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-15 | google/gemma-3-4b-pt | 1205 | other | exp-04 |  | completed | 0.6666666666666666 | supported | adopt | True |  |
| test | r-fac8a9cc | exp-16 | google/gemma-3-4b-pt | 1314 | decode-config | exp-05 |  | completed | 0.6368460955269143 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-17 | google/gemma-3-4b-pt | 1337 | other | exp-05 |  | completed | 0.6353297952994693 | supported | adopt | True |  |
| test | r-fac8a9cc | exp-18 | google/gemma-3-4b-pt | 1373 | merge | exp-05 |  | completed | 0.6315390447308568 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-19 | google/gemma-3-4b-pt | 1404 | sft | exp-17 | synthetic:self (run_grpo1b/model-step-100) + HF openai/gsm8k (main, train) human solutions | failed |  | inconclusive | abandon_line | True |  |
| test | r-fac8a9cc | exp-20 | google/gemma-3-4b-pt | 1408 | sft | exp-17 | synthetic:self (run_grpo1b/model-step-100) + HF openai/gsm8k (main, train) human solutions | failed |  | inconclusive | abandon_line | True |  |
| test | r-fac8a9cc | exp-21 | google/gemma-3-4b-pt | 1429 | sft | exp-17 | synthetic:self (run_grpo1b/model-step-100) + HF openai/gsm8k (main, train) human solutions | completed | 0.6330553449583017 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-22 | google/gemma-3-4b-pt | 1458 | merge | exp-05 |  | completed | 0.645185746777862 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-23 | google/gemma-3-4b-pt | 1466 | other | exp-22 |  | completed | 0.6315390447308568 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-24 | google/gemma-3-4b-pt | 1477 | decode-config | exp-22 |  | completed | 0.6277482941622441 | contradicted | reject | True |  |
| test | r-fac8a9cc | exp-25 | google/gemma-3-4b-pt | 1488 | other | exp-17 |  | completed | 0.6368460955269143 | inconclusive | adopt | True | 0.6262319939347991 |
| test | r-fdeba838 | exp-01 | google/gemma-3-4b-pt | 197 | sft | base_model | HF id: openai/gsm8k (main, train split), loaded inside the training script | completed | 0.467 | inconclusive | adopt | True |  |
| test | r-fdeba838 | exp-02 | google/gemma-3-4b-pt | 414 | decode-config | exp-01 |  | completed | 0.46 | contradicted | adopt | True |  |
| test | r-fdeba838 | exp-03 | google/gemma-3-4b-pt | 451 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | completed | 0.553 | supported | adopt | True |  |
| test | r-fdeba838 | exp-04 | google/gemma-3-4b-pt | 463 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| test | r-fdeba838 | exp-05 | google/gemma-3-4b-pt | 563 | other | exp-03 |  | completed | 0.48 | inconclusive | adopt | False |  |
| test | r-fdeba838 | exp-06 | google/gemma-3-4b-pt | 592 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | failed |  | inconclusive | iterate | True |  |
| test | r-fdeba838 | exp-07 | google/gemma-3-4b-pt | 614 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | failed |  | inconclusive | iterate | True |  |
| test | r-fdeba838 | exp-08 | google/gemma-3-4b-pt | 626 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | failed |  | inconclusive | iterate | True |  |
| test | r-fdeba838 | exp-09 | google/gemma-3-4b-pt | 636 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | completed | 0.147 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-10 | google/gemma-3-4b-pt | 738 | decode-config | exp-09 |  | completed | 0.067 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-11 | google/gemma-3-4b-pt | 766 | sft | exp-09 | HF id: openai/gsm8k (main, train split) | completed | 0.113 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-12 | google/gemma-3-4b-pt | 843 | rft | exp-03 | synthetic:self,HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA (train split) | completed | 0.467 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-13 | google/gemma-3-4b-pt | 912 | decode-config | exp-05 |  | completed | 0.433 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-14 | google/gemma-3-4b-pt | 922 | decode-config | exp-05 |  | completed | 0.4 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-15 | google/gemma-3-4b-pt | 936 | decode-config | exp-05 |  | completed | 0.46 | contradicted | reject | True |  |
| test | r-fdeba838 | exp-16 | google/gemma-3-4b-pt | 948 | decode-config | exp-05 |  | completed | 0.46 | supported | adopt | True | 0.4609552691432904 |
