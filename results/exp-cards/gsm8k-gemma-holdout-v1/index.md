# Index of reconstructed cards

| side | run_ref | card | base model | launch_i | family | parent | data sources | exec | best own eval | verdict | decision | hyp stated | official |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | r-016546b4 | exp-01 | Qwen/Qwen3-1.7B-Base | 23 | other | base_model |  | completed | 0.1 | inconclusive | reject | False |  |
| train | r-016546b4 | exp-02 | Qwen/Qwen3-1.7B-Base | 52 | sft | base_model | HF meta-math/MetaMathQA | failed |  | inconclusive | iterate | False |  |
| train | r-016546b4 | exp-03 | Qwen/Qwen3-1.7B-Base | 62 | sft | base_model | HF meta-math/MetaMathQA | completed |  | inconclusive | adopt | False |  |
| train | r-016546b4 | exp-04 | Qwen/Qwen3-1.7B-Base | 68 | other | exp-03 |  | completed | 0.14 | inconclusive | reject | False |  |
| train | r-016546b4 | exp-05 | Qwen/Qwen3-1.7B-Base | 84 | sft | base_model | HF meta-math/MetaMathQA | killed |  | inconclusive | adopt | True |  |
| train | r-016546b4 | exp-06 | Qwen/Qwen3-1.7B-Base | 119 | other | exp-05 |  | completed | 0.3 | supported | adopt | True |  |
| train | r-016546b4 | exp-07 | Qwen/Qwen3-1.7B-Base | 201 | sft | base_model | HF meta-math/MetaMathQA,HF openai/gsm8k (main, train split) | killed |  | inconclusive | abandon_line | True |  |
| train | r-01ed5927 | exp-01 | Qwen/Qwen3-4B-Base | 102 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.4266666666666667 | contradicted | adopt | False |  |
| train | r-01ed5927 | exp-02 | Qwen/Qwen3-4B-Base | 312 | decode-config | exp-01 |  | completed | 0.8133333333333334 | supported | adopt | True |  |
| train | r-01ed5927 | exp-03 | Qwen/Qwen3-4B-Base | 336 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-01ed5927 | exp-04 | Qwen/Qwen3-4B-Base | 401 | rft | base_model | derived:exp-02 + openai/gsm8k,synthetic:self | completed | 0.82 | inconclusive | reject | True |  |
| train | r-01ed5927 | exp-05 | Qwen/Qwen3-4B-Base | 529 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | failed |  | inconclusive | iterate | True |  |
| train | r-01ed5927 | exp-06 | Qwen/Qwen3-4B-Base | 577 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | completed | 0.8385140257771039 | supported | adopt | True |  |
| train | r-01ed5927 | exp-07 | Qwen/Qwen3-4B-Base | 663 | other | exp-06 |  | completed | 0.84 | inconclusive | adopt | False |  |
| train | r-01ed5927 | exp-08 | Qwen/Qwen3-4B-Base | 685 | sft | base_model | derived:exp-04 + meta-math/MetaMathQA,meta-math/MetaMathQA | completed | 0.8332069749810462 | contradicted | reject | True |  |
| train | r-0295e24f | exp-01 | Qwen/Qwen3-4B-Base | 129 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.04666666666666667 | inconclusive | adopt | True |  |
| train | r-0295e24f | exp-02 | Qwen/Qwen3-4B-Base | 187 | other | exp-01 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-0295e24f | exp-03 | Qwen/Qwen3-4B-Base | 206 | decode-config | exp-01 |  | completed | 0.14 | contradicted | reject | True |  |
| train | r-0295e24f | exp-04 | Qwen/Qwen3-4B-Base | 234 | sft | base_model | openai/gsm8k (main, split=train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-0295e24f | exp-05 | Qwen/Qwen3-4B-Base | 249 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.04 | contradicted | reject | True |  |
| train | r-0295e24f | exp-06 | Qwen/Qwen3-4B-Base | 312 | sft | base_model | openai/gsm8k (main, split=train) | completed | 0.5 | supported | adopt | True |  |
| train | r-0295e24f | exp-07 | Qwen/Qwen3-4B-Base | 411 | rft | base_model | synthetic:self | completed | 0.42 | contradicted | reject | True |  |
| train | r-0295e24f | exp-08 | Qwen/Qwen3-4B-Base | 443 | rft | base_model | synthetic:self | completed | 0.36 | contradicted | reject | True |  |
| train | r-0295e24f | exp-09 | Qwen/Qwen3-4B-Base | 459 | other | exp-06 |  | completed | 0.5 | supported | adopt | False |  |
| train | r-0632a2e4 | exp-01 | Qwen/Qwen3-1.7B-Base | 96 | sft | base_model | local | completed |  | inconclusive | adopt | True |  |
| train | r-0632a2e4 | exp-02 | Qwen/Qwen3-1.7B-Base | 178 | decode-config | exp-01 |  | completed | 0.6 | inconclusive | reject | True |  |
| train | r-0632a2e4 | exp-03 | Qwen/Qwen3-1.7B-Base | 277 | sft | base_model | local | completed | 0.673 | supported | reject | True |  |
| train | r-0632a2e4 | exp-04 | Qwen/Qwen3-1.7B-Base | 371 | sft | base_model | local | completed | 0.693 | inconclusive | adopt | False |  |
| train | r-0632a2e4 | exp-05 | Qwen/Qwen3-1.7B-Base | 481 | sft | base_model | local,synthetic:self | completed | 0.675 | contradicted | reject | True |  |
| train | r-0632a2e4 | exp-06 | Qwen/Qwen3-1.7B-Base | 548 | other | exp-04 |  | completed | 0.665 | inconclusive | adopt | False |  |
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
| train | r-06a66e16 | exp-09 | Qwen/Qwen3-1.7B-Base | 810 | decode-config | exp-05 |  | completed | 0.8733333333333333 | supported | adopt | True |  |
| train | r-0788f765 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 84 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 96 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | True |  |
| train | r-0788f765 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 104 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 112 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | failed |  | inconclusive | iterate | False |  |
| train | r-0788f765 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 122 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded in-process by train.py) | completed | 0.1 | inconclusive | adopt | False |  |
| train | r-0788f765 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 148 | decode-config | exp-05 |  | completed | 0.5 | supported | adopt | True |  |
| train | r-08a3dad0 | exp-01 | Qwen/Qwen3-1.7B-Base | 247 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.12666666666666668 | contradicted | adopt | True |  |
| train | r-08a3dad0 | exp-02 | Qwen/Qwen3-1.7B-Base | 576 | decode-config | exp-01 |  | completed | 0.6133333333333333 | supported | adopt | True |  |
| train | r-08a3dad0 | exp-03 | Qwen/Qwen3-1.7B-Base | 637 | other | exp-02 |  | completed | 0.6133333333333333 | supported | adopt | False |  |
| train | r-08a3dad0 | exp-04 | Qwen/Qwen3-1.7B-Base | 712 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.47333333333333333 | contradicted | reject | True |  |
| train | r-08a3dad0 | exp-05 | Qwen/Qwen3-1.7B-Base | 806 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | failed |  | inconclusive | abandon_line | True |  |
| train | r-08a3dad0 | exp-06 | Qwen/Qwen3-1.7B-Base | 841 | sft | base_model | local: built from HF openai/gsm8k (main, train) + meta-math/MetaMathQA (GSM_* types) | completed | 0.04 | contradicted | reject | True |  |
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
| train | r-0d7c7a69 | exp-11 | Qwen/Qwen3-1.7B-Base | 826 | decode-config | exp-09 |  | completed | 0.8533 | inconclusive | adopt | True |  |
| train | r-114ff7d5 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 89 | sft | base_model | meta-math/MetaMathQA | failed |  | inconclusive | iterate | True |  |
| train | r-114ff7d5 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 117 | sft | base_model | meta-math/MetaMathQA | completed | 0.7066666666666667 | inconclusive | adopt | True |  |
| train | r-114ff7d5 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 187 | sft | exp-02 | openai/gsm8k (config main, split train) | completed | 0.49333333333333335 | contradicted | reject | True |  |
| train | r-114ff7d5 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 242 | sft | exp-02 | meta-math/MetaMathQA | completed | 0.72 | inconclusive | adopt | False |  |
| train | r-114ff7d5 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 289 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.7066666666666667 | inconclusive | reject | False |  |
| train | r-114ff7d5 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 486 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.6666666666666666 | contradicted | reject | True |  |
| train | r-114ff7d5 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 510 | sft | exp-04 | meta-math/MetaMathQA | completed | 0.82 | supported | adopt | True |  |
| train | r-114ff7d5 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 594 | other | exp-07 |  | completed | 0.82 | supported | adopt | True |  |
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
| train | r-11be89c8 | exp-15 | HuggingFaceTB/SmolLM3-3B-Base | 532 | merge | exp-14 |  | completed | 0.2375 | supported | adopt | True |  |
| train | r-130da32f | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 52 | sft | base_model | HF openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.225 | supported | adopt | True |  |
| train | r-130da32f | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 257 | merge | exp-01 |  | completed | 0.1 | contradicted | reject | True |  |
| train | r-130da32f | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 270 | other | exp-01 |  | completed | 0.1 | inconclusive | adopt | True |  |
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
| train | r-139b3113 | exp-11 | Qwen/Qwen3-1.7B-Base | 468 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-1441f3c6 | exp-01 | Qwen/Qwen3-1.7B-Base | 67 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.02 | inconclusive | reject | True |  |
| train | r-1441f3c6 | exp-02 | Qwen/Qwen3-1.7B-Base | 156 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | failed |  | inconclusive | iterate | True |  |
| train | r-1441f3c6 | exp-03 | Qwen/Qwen3-1.7B-Base | 167 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.05 | inconclusive | reject | True |  |
| train | r-1441f3c6 | exp-04 | Qwen/Qwen3-1.7B-Base | 222 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | inconclusive | abandon_line | True |  |
| train | r-1441f3c6 | exp-05 | Qwen/Qwen3-1.7B-Base | 234 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | inconclusive | abandon_line | True |  |
| train | r-1441f3c6 | exp-06 | Qwen/Qwen3-1.7B-Base | 249 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | killed |  | inconclusive | iterate | True |  |
| train | r-1441f3c6 | exp-07 | Qwen/Qwen3-1.7B-Base | 255 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed | 0.1417 | supported | adopt | True |  |
| train | r-1441f3c6 | exp-08 | Qwen/Qwen3-1.7B-Base | 267 | sft | base_model | HF: openai/gsm8k (config main, split=train, loaded in-process by the training script) | completed |  | contradicted | reject | True |  |
| train | r-1441f3c6 | exp-09 | Qwen/Qwen3-1.7B-Base | 283 | other | exp-07 |  | completed | 0.15 | inconclusive | adopt | False |  |
| train | r-185ac8a3 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 92 | sft | base_model | meta-math/MetaMathQA | completed | 0.8 | inconclusive | adopt | False |  |
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
| train | r-1fff43fc | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 982 | other | exp-12 |  | completed | 0.78 | inconclusive | adopt | False |  |
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
| train | r-2354e591 | exp-11 | HuggingFaceTB/SmolLM3-3B-Base | 304 | sft | exp-09 | openai/gsm8k,meta-math/MetaMathQA | completed |  | inconclusive | adopt | True |  |
| train | r-2354e591 | exp-12 | HuggingFaceTB/SmolLM3-3B-Base | 328 | merge | exp-11 |  | completed | 0.6133333333333333 | contradicted | reject | False |  |
| train | r-23aab620 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 86 | sft | base_model | meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-23aab620 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 102 | sft | base_model | meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | False |  |
| train | r-23aab620 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 130 | sft | base_model | meta-math/MetaMathQA | completed |  | inconclusive | adopt | False |  |
| train | r-2aeedf08 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 118 | sft | base_model | HF:openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | False |  |
| train | r-2aeedf08 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 157 | merge | exp-01 |  | completed | 0.62 | inconclusive | adopt | False |  |
| train | r-2aeedf08 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 173 | sft | base_model | HF:openai/gsm8k (configs main + socratic, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-2aeedf08 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 203 | other | exp-02 |  | completed | 0.54 | supported | adopt | False |  |
| train | r-2c6de1f4 | exp-01 | Qwen/Qwen3-1.7B-Base | 65 | sft | base_model | meta-math/MetaMathQA | completed | 0.38 | inconclusive | reject | False |  |
| train | r-2c6de1f4 | exp-02 | Qwen/Qwen3-1.7B-Base | 85 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.42 | supported | reject | True |  |
| train | r-2c6de1f4 | exp-03 | Qwen/Qwen3-1.7B-Base | 107 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.22 | contradicted | reject | True |  |
| train | r-2c6de1f4 | exp-04 | Qwen/Qwen3-1.7B-Base | 126 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | completed | 0.5 | supported | adopt | True |  |
| train | r-2c6de1f4 | exp-05 | Qwen/Qwen3-1.7B-Base | 146 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA | killed |  | inconclusive | abandon_line | True |  |
| train | r-2cd75f43 | exp-01 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | HF gsm8k (main) train split | completed | 0.04 | inconclusive | abandon_line | False |  |
| train | r-2cd75f43 | exp-02 | Qwen/Qwen3-4B-Base | 151 | sft | base_model | HF gsm8k (main) train split | completed | 0.48 | inconclusive | adopt | True |  |
| train | r-2cd75f43 | exp-03 | Qwen/Qwen3-4B-Base | 213 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-2cd75f43 | exp-04 | Qwen/Qwen3-4B-Base | 229 | sft | base_model | HF gsm8k (main) train split + HF meta-math/MetaMathQA | completed | 0.48 | contradicted | reject | True |  |
| train | r-2cd75f43 | exp-05 | Qwen/Qwen3-4B-Base | 278 | sft | base_model | HF gsm8k (main) train split | completed | 0.713 | supported | adopt | False |  |
| train | r-2cd75f43 | exp-06 | Qwen/Qwen3-4B-Base | 298 | other | exp-05 |  | completed | 0.7 | inconclusive | adopt | False |  |
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
| train | r-2f4530d4 | exp-11 | Qwen/Qwen3-4B-Base | 255 | decode-config | exp-10 |  | completed | 0.747 | inconclusive | adopt | False |  |
| train | r-33444a20 | exp-01 | Qwen/Qwen3-1.7B-Base | 80 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | killed |  | inconclusive | abandon_line | False |  |
| train | r-33444a20 | exp-02 | Qwen/Qwen3-1.7B-Base | 84 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | completed | 0.127 | inconclusive | adopt | False |  |
| train | r-33444a20 | exp-03 | Qwen/Qwen3-1.7B-Base | 102 | decode-config | exp-02 |  | completed | 0.24 | supported | reject | False |  |
| train | r-33444a20 | exp-04 | Qwen/Qwen3-1.7B-Base | 120 | sft | base_model | HF: meta-math/MetaMathQA (split=train, loaded directly by the script) | completed | 0.6333333333333333 | supported | adopt | False |  |
| train | r-33444a20 | exp-05 | Qwen/Qwen3-1.7B-Base | 160 | other | exp-04 |  | completed | 0.6133333333333333 | inconclusive | adopt | False |  |
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
| train | r-3911d1bb | exp-13 | HuggingFaceTB/SmolLM3-3B-Base | 185 | merge | exp-12 |  | completed | 0.34 | inconclusive | adopt | False |  |
| train | r-3fa66d9b | exp-01 | Qwen/Qwen3-4B-Base | 80 | sft | base_model | HF id: openai/gsm8k (config main, split train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-02 | Qwen/Qwen3-4B-Base | 86 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-03 | Qwen/Qwen3-4B-Base | 90 | sft | base_model | HF id: openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-04 | Qwen/Qwen3-4B-Base | 137 | merge | exp-03 |  | completed | 0.13333333333333333 | contradicted | reject | False |  |
| train | r-3fa66d9b | exp-05 | Qwen/Qwen3-4B-Base | 170 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-3fa66d9b | exp-06 | Qwen/Qwen3-4B-Base | 174 | sft | base_model | HF id: openai/gsm8k (config main, split train) | completed | 0.5833333333333334 | supported | adopt | True |  |
| train | r-3fa66d9b | exp-07 | Qwen/Qwen3-4B-Base | 199 | other | exp-06 |  | completed | 0.58 | inconclusive | adopt | False |  |
| train | r-4254277e | exp-01 | Qwen/Qwen3-1.7B-Base | 118 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.053 | inconclusive | reject | True |  |
| train | r-4254277e | exp-02 | Qwen/Qwen3-1.7B-Base | 195 | sft | base_model | openai/gsm8k (main, train split) | failed |  | inconclusive | abandon_line | True |  |
| train | r-4254277e | exp-03 | Qwen/Qwen3-1.7B-Base | 212 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.04 | contradicted | adopt | True |  |
| train | r-4254277e | exp-04 | Qwen/Qwen3-1.7B-Base | 382 | decode-config | exp-03 |  | completed | 0.64 | supported | adopt | True |  |
| train | r-4254277e | exp-05 | Qwen/Qwen3-1.7B-Base | 409 | sft | base_model | openai/gsm8k (main, train split),meta-math/MetaMathQA (GSM_AnsAug subset),meta-math/MetaMathQA (GSM_Rephrased subset) | failed |  | inconclusive | abandon_line | False |  |
| train | r-4254277e | exp-06 | Qwen/Qwen3-1.7B-Base | 435 | other | exp-04 |  | completed |  | inconclusive | adopt | False |  |
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
| train | r-426bd770 | exp-13 | Qwen/Qwen3-1.7B-Base | 1242 | grpo | exp-06 | HF openai/gsm8k (train split) | completed |  | inconclusive | adopt | False |  |
| train | r-426bd770 | exp-14 | Qwen/Qwen3-1.7B-Base | 1262 | merge | exp-03 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 532 | sft | base_model | derived: HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows of the first 4 parquet shards) + HF openai/gsm8k train + HF meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_FOBAR, GSM_SV) | killed |  | inconclusive | abandon_line | False |  |
| train | r-4463b5d3 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 632 | sft | base_model | derived: HF nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k rows of the first 4 parquet shards) + HF openai/gsm8k train + HF meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_FOBAR, GSM_SV),HF openai/gsm8k (main, train split) | completed | 0.695 | inconclusive | adopt | False |  |
| train | r-4463b5d3 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 896 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 1040 | grpo | exp-02 | derived:exp-02 data lineage - data/rft_pool.jsonl (gsm8k train questions + 22000 OpenMathInstruct-2 augmented_gsm8k problems, 29473 rows [683]) plus data/rl_pool.jsonl (up to 40000 further augmented_gsm8k problems from parquet shards 4-14, never trained on) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries unused elsewhere | failed |  | inconclusive | abandon_line | False |  |
| train | r-4463b5d3 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 1079 | grpo | exp-02 | derived:exp-02 data lineage - data/rft_pool.jsonl (gsm8k train questions + 22000 OpenMathInstruct-2 augmented_gsm8k problems, 29473 rows [683]) plus data/rl_pool.jsonl (up to 40000 further augmented_gsm8k problems from parquet shards 4-14, never trained on) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries unused elsewhere | killed | 0.866 | supported | adopt | False |  |
| train | r-4463b5d3 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 1245 | other | exp-05 |  | completed |  | inconclusive | reject | False |  |
| train | r-4463b5d3 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 1247 | grpo | exp-05 | derived:exp-02 data lineage - data/rft_pool.jsonl (29473 rows) plus data/rl_pool.jsonl (unused OpenMathInstruct-2 augmented_gsm8k problems) plus up to 20000 meta-math/MetaMathQA GSM_Rephrased / GSM_AnsAug queries | killed | 0.89 | inconclusive | adopt | False |  |
| train | r-4463b5d3 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 1325 | other | exp-07 |  | completed | 0.8696 | inconclusive | adopt | False |  |
| train | r-46a821e3 | exp-01 | Qwen/Qwen3-1.7B-Base | 35 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.02 | contradicted | reject | True |  |
| train | r-46a821e3 | exp-02 | Qwen/Qwen3-1.7B-Base | 121 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | adopt | True |  |
| train | r-46a821e3 | exp-03 | Qwen/Qwen3-1.7B-Base | 142 | merge | exp-02 |  | completed | 0.1 | inconclusive | adopt | True |  |
| train | r-46a821e3 | exp-04 | Qwen/Qwen3-1.7B-Base | 153 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-46a821e3 | exp-05 | Qwen/Qwen3-1.7B-Base | 182 | other | exp-03 |  | completed |  | inconclusive | adopt | True |  |
| train | r-47a7873c | exp-01 | Qwen/Qwen3-4B-Base | 56 | sft | base_model | openai/gsm8k (config "main", train split) | failed |  | inconclusive | abandon_line | False |  |
| train | r-47a7873c | exp-02 | Qwen/Qwen3-4B-Base | 62 | sft | base_model | openai/gsm8k (config "main", train split) | completed | 0.6 | inconclusive | reject | False |  |
| train | r-47a7873c | exp-03 | Qwen/Qwen3-4B-Base | 110 | sft | base_model | openai/gsm8k (config "main", train split) | completed |  | inconclusive | adopt | True |  |
| train | r-47a7873c | exp-04 | Qwen/Qwen3-4B-Base | 131 | other | exp-03 |  | completed |  | inconclusive | adopt | True |  |
| train | r-4aa3d061 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 164 | sft | base_model | HF id openai/gsm8k (config main, split train) | killed | 0.553 | inconclusive | adopt | True |  |
| train | r-4aa3d061 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 348 | other | exp-01 |  | completed | 0.35 | supported | adopt | True |  |
| train | r-4d0f7a19 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 98 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split),openai/gsm8k (main, train split) | completed | 0.5666666666666667 | inconclusive | adopt | True |  |
| train | r-4d0f7a19 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 227 | sft | exp-01 | openai/gsm8k (main, train split) | completed | 0.54 | contradicted | reject | True |  |
| train | r-4d0f7a19 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 293 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split),openai/gsm8k (main, train split) | completed | 0.5266666666666666 | contradicted | reject | True |  |
| train | r-4d0f7a19 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 414 | other | exp-01 |  | completed | 0.6 | supported | adopt | True |  |
| train | r-54aba2d1 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 64 | sft | base_model | HF:openai/gsm8k (config main, train split) | failed |  | inconclusive | iterate | False |  |
| train | r-54aba2d1 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 74 | sft | base_model | HF:openai/gsm8k (config main, train split) | failed |  | inconclusive | iterate | False |  |
| train | r-54aba2d1 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 80 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 94 | merge | exp-03 |  | completed | 0.42 | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 136 | sft | base_model | HF:openai/gsm8k (config main, train split) | completed |  | inconclusive | adopt | False |  |
| train | r-54aba2d1 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 142 | merge | exp-05 |  | completed | 0.2866666666666667 | contradicted | reject | False |  |
| train | r-54aba2d1 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 148 | merge | exp-03 |  | completed | 0.4094010614101592 | supported | adopt | True |  |
| train | r-59151c09 | exp-01 | Qwen/Qwen3-4B-Base | 63 | sft | base_model | HF: openai/gsm8k (main, train),HF: meta-math/MetaMathQA (train) | killed | 0.08 | inconclusive | adopt | True |  |
| train | r-59151c09 | exp-02 | Qwen/Qwen3-4B-Base | 97 | decode-config | exp-01 |  | completed | 0.74 | supported | adopt | True |  |
| train | r-59151c09 | exp-03 | Qwen/Qwen3-4B-Base | 107 | sft | exp-01 | HF: openai/gsm8k (main, train),HF: meta-math/MetaMathQA (train) | completed | 0.76 | inconclusive | adopt | True |  |
| train | r-59151c09 | exp-04 | Qwen/Qwen3-4B-Base | 132 | sft | exp-03 | HF: openai/gsm8k (main, train) | completed | 0.6666666666666666 | contradicted | reject | True |  |
| train | r-59151c09 | exp-05 | Qwen/Qwen3-4B-Base | 161 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-06 | Qwen/Qwen3-4B-Base | 183 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-07 | Qwen/Qwen3-4B-Base | 195 | grpo | exp-03 | HF: openai/gsm8k (main, train) - prompts only, gold answer used as the reward key | killed |  | inconclusive | abandon_line | True |  |
| train | r-59151c09 | exp-08 | Qwen/Qwen3-4B-Base | 219 | other | exp-03 |  | completed | 0.7 | inconclusive | adopt | False |  |
| train | r-59151c09 | exp-09 | Qwen/Qwen3-4B-Base | 230 | sft | base_model | HF: openai/gsm8k (main, train) | completed | 0.7666666666666667 | contradicted | reject | False |  |
| train | r-59151c09 | exp-10 | Qwen/Qwen3-4B-Base | 291 | decode-config | exp-08 |  | completed | 0.84 | supported | adopt | False |  |
| train | r-5dcadd31 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 206 | sft | base_model | local (tokenized from HF nvidia/OpenMathInstruct-2 train_1M + openai/gsm8k main train) | killed | 0.948 | supported | adopt | False |  |
| train | r-5dcadd31 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 472 | decode-config | exp-01 |  | completed | 0.873 | supported | adopt | False |  |
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
| train | r-5f4b22de | exp-08 | Qwen/Qwen3-4B-Base | 688 | other | exp-07 |  | completed |  | inconclusive | adopt | False |  |
| train | r-60904922 | exp-01 | Qwen/Qwen3-4B-Base | 176 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-02 | Qwen/Qwen3-4B-Base | 201 | merge | exp-01 |  | completed | 0.16 | inconclusive | reject | False |  |
| train | r-60904922 | exp-03 | Qwen/Qwen3-4B-Base | 226 | merge | exp-01 |  | completed | 0.8 | supported | reject | True |  |
| train | r-60904922 | exp-04 | Qwen/Qwen3-4B-Base | 253 | sft | base_model | HF openai/gsm8k (config main, split train), re-rendered in the benchmark's 10-shot prompt format | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-05 | Qwen/Qwen3-4B-Base | 323 | merge | exp-04 |  | completed | 0.35 | contradicted | reject | True |  |
| train | r-60904922 | exp-06 | Qwen/Qwen3-4B-Base | 393 | rft | base_model | synthetic:self | completed |  | inconclusive | adopt | True |  |
| train | r-60904922 | exp-07 | Qwen/Qwen3-4B-Base | 414 | merge | exp-06 |  | completed | 0.51 | contradicted | adopt | True |  |
| train | r-60904922 | exp-08 | Qwen/Qwen3-4B-Base | 443 | other | exp-07 |  | completed |  | inconclusive | adopt | False |  |
| train | r-628a7a20 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 20368 | sft | base_model | local | killed |  | inconclusive | abandon_line | False |  |
| train | r-628a7a20 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 25061 | sft | base_model | local | completed | 0.6733333333333333 | inconclusive | adopt | True |  |
| train | r-628a7a20 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 29588 | rft | exp-02 | synthetic:self,derived:exp-02 | failed |  | inconclusive | abandon_line | True |  |
| train | r-628a7a20 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 30851 | rft | exp-02 | synthetic:self,derived:exp-02 | completed | 0.74 | supported | adopt | True |  |
| train | r-628a7a20 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 31745 | other | exp-04 |  | completed | 0.7035633055344959 | inconclusive | adopt | False |  |
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
| train | r-635b683e | exp-14 | Qwen/Qwen3-4B-Base | 1373 | other | exp-13 |  | completed | 0.927 | inconclusive | adopt | False |  |
| train | r-655a20a6 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 125 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | completed | 0.46 | inconclusive | adopt | False |  |
| train | r-655a20a6 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | gsm8k (main/train) + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-736cc5f9 | exp-01 | Qwen/Qwen3-1.7B-Base | 110 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-736cc5f9 | exp-02 | Qwen/Qwen3-1.7B-Base | 170 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.6 | inconclusive | reject | True |  |
| train | r-736cc5f9 | exp-03 | Qwen/Qwen3-1.7B-Base | 335 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | completed | 0.66 | supported | adopt | True |  |
| train | r-736cc5f9 | exp-04 | Qwen/Qwen3-1.7B-Base | 401 | sft | base_model | HF: openai/gsm8k (config main, split train), loaded in-process by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-736cc5f9 | exp-05 | Qwen/Qwen3-1.7B-Base | 427 | other | exp-03 |  | completed | 0.63 | inconclusive | adopt | False |  |
| train | r-75836ae8 | exp-01 | Qwen/Qwen3-1.7B-Base | 124 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.01 | inconclusive | reject | True |  |
| train | r-75836ae8 | exp-02 | Qwen/Qwen3-1.7B-Base | 194 | sft | base_model | openai/gsm8k (main, train split) | completed | 0.1 | contradicted | adopt | True |  |
| train | r-75836ae8 | exp-03 | Qwen/Qwen3-1.7B-Base | 348 | sft | exp-02 | openai/gsm8k (main, train split) | completed | 0.5 | supported | adopt | True |  |
| train | r-75836ae8 | exp-04 | Qwen/Qwen3-1.7B-Base | 463 | sft | exp-03 | openai/gsm8k (main, train split) | completed |  | inconclusive | abandon_line | True |  |
| train | r-75836ae8 | exp-05 | Qwen/Qwen3-1.7B-Base | 597 | other | exp-03 |  | completed | 0.53 | supported | adopt | True |  |
| train | r-7842f260 | exp-01 | Qwen/Qwen3-1.7B-Base | 12086 | sft | base_model | HF openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-7842f260 | exp-02 | Qwen/Qwen3-1.7B-Base | 12539 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.66 | inconclusive | adopt | False |  |
| train | r-7842f260 | exp-03 | Qwen/Qwen3-1.7B-Base | 19507 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-02's checkpoint),HF openai/gsm8k (config main, split train) | completed | 0.843 | inconclusive | adopt | False |  |
| train | r-7842f260 | exp-04 | Qwen/Qwen3-1.7B-Base | 22143 | merge | exp-02 |  | killed |  | inconclusive | iterate | False |  |
| train | r-7842f260 | exp-05 | Qwen/Qwen3-1.7B-Base | 22789 | merge | exp-02 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-7842f260 | exp-06 | Qwen/Qwen3-1.7B-Base | 23003 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-03's checkpoint) | completed | 0.793 | contradicted | reject | False |  |
| train | r-7842f260 | exp-07 | Qwen/Qwen3-1.7B-Base | 24174 | merge | exp-03 |  | completed | 0.833 | inconclusive | reject | False |  |
| train | r-7842f260 | exp-08 | Qwen/Qwen3-1.7B-Base | 24999 | distill | base_model | local,synthetic:Qwen/Qwen3-14B,synthetic:self (sampled from exp-03's checkpoint),HF openai/gsm8k (config main, split train) | completed | 0.784 | inconclusive | reject | False |  |
| train | r-7842f260 | exp-09 | Qwen/Qwen3-1.7B-Base | 25224 | other | exp-03 |  | completed | 0.82 | inconclusive | adopt | False |  |
| train | r-7842f260 | exp-10 | Qwen/Qwen3-1.7B-Base | 26690 | merge | exp-03 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-792b20f6 | exp-01 | Qwen/Qwen3-4B-Base | 73 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-02 | Qwen/Qwen3-4B-Base | 93 | other | exp-01 |  | completed |  | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-03 | Qwen/Qwen3-4B-Base | 103 | merge | exp-02 |  | failed | 0.2 | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-04 | Qwen/Qwen3-4B-Base | 171 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-792b20f6 | exp-05 | Qwen/Qwen3-4B-Base | 191 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-792b20f6 | exp-06 | Qwen/Qwen3-4B-Base | 209 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-7a94150b | exp-01 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | failed |  | inconclusive | iterate | True |  |
| train | r-7a94150b | exp-02 | Qwen/Qwen3-4B-Base | 101 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.175 | inconclusive | adopt | True |  |
| train | r-7a94150b | exp-03 | Qwen/Qwen3-4B-Base | 131 | decode-config | exp-02 |  | completed | 0.15 | contradicted | reject | True |  |
| train | r-7a94150b | exp-04 | Qwen/Qwen3-4B-Base | 159 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.25 | contradicted | reject | True |  |
| train | r-7a94150b | exp-05 | Qwen/Qwen3-4B-Base | 177 | sft | base_model | openai/gsm8k (config main, split train), loaded in-script | completed | 0.45 | contradicted | reject | True |  |
| train | r-7a94150b | exp-06 | Qwen/Qwen3-4B-Base | 204 | other | base_model |  | completed | 0.5 | supported | adopt | True |  |
| train | r-7f29490c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 97 | sft | base_model | meta-math/MetaMathQA,openai/gsm8k (main, train split) | completed | 0.3 | inconclusive | adopt | True |  |
| train | r-7f29490c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 182 | decode-config | exp-01 |  | completed | 0.56 | inconclusive | adopt | False |  |
| train | r-87e033c4 | exp-01 | Qwen/Qwen3-4B-Base | 82 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.04 | inconclusive | adopt | False |  |
| train | r-87e033c4 | exp-02 | Qwen/Qwen3-4B-Base | 146 | decode-config | exp-01 |  | completed | 0.08 | contradicted | adopt | True |  |
| train | r-87e033c4 | exp-03 | Qwen/Qwen3-4B-Base | 169 | decode-config | exp-02 |  | completed | 0.06 | contradicted | adopt | False |  |
| train | r-87e033c4 | exp-04 | Qwen/Qwen3-4B-Base | 201 | decode-config | exp-03 |  | completed | 0.013333333333333334 | contradicted | reject | False |  |
| train | r-87e033c4 | exp-05 | Qwen/Qwen3-4B-Base | 224 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.23333333333333334 | supported | adopt | True |  |
| train | r-87e033c4 | exp-06 | Qwen/Qwen3-4B-Base | 266 | decode-config | exp-05 |  | completed | 0.8225928733889311 | supported | adopt | False |  |
| train | r-87e033c4 | exp-07 | Qwen/Qwen3-4B-Base | 294 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.8133333333333334 | contradicted | reject | True |  |
| train | r-87e033c4 | exp-08 | Qwen/Qwen3-4B-Base | 361 | other | exp-06 |  | completed | 0.8133333333333334 | inconclusive | adopt | False |  |
| train | r-87e033c4 | exp-09 | Qwen/Qwen3-4B-Base | 378 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.3933333333333333 | inconclusive | reject | False |  |
| train | r-87e033c4 | exp-10 | Qwen/Qwen3-4B-Base | 408 | sft | base_model | local: built from openai/gsm8k train + meta-math/MetaMathQA (GSM_AnsAug, GSM_Rephrased) | completed | 0.78 | inconclusive | reject | False |  |
| train | r-88141936 | exp-01 | Qwen/Qwen3-4B-Base | 102 | sft | base_model | HF meta-math/MetaMathQA | failed |  | inconclusive | abandon_line | False |  |
| train | r-88141936 | exp-02 | Qwen/Qwen3-4B-Base | 116 | sft | base_model | HF meta-math/MetaMathQA | completed | 0.6 | inconclusive | adopt | False |  |
| train | r-89603e49 | exp-01 | Qwen/Qwen3-1.7B-Base | 17875 | sft | base_model | local | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-02 | Qwen/Qwen3-1.7B-Base | 18544 | sft | base_model | local | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-03 | Qwen/Qwen3-1.7B-Base | 19154 | sft | base_model | local | completed | 0.727 | supported | adopt | True |  |
| train | r-89603e49 | exp-04 | Qwen/Qwen3-1.7B-Base | 28893 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-05 | Qwen/Qwen3-1.7B-Base | 29447 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-06 | Qwen/Qwen3-1.7B-Base | 29614 | sft | base_model | local,synthetic:self | killed | 0.8266666666666667 | supported | adopt | True |  |
| train | r-89603e49 | exp-07 | Qwen/Qwen3-1.7B-Base | 33782 | sft | base_model | local,synthetic:self | killed |  | inconclusive | abandon_line | True |  |
| train | r-89603e49 | exp-08 | Qwen/Qwen3-1.7B-Base | 34040 | sft | exp-06 | local,synthetic:self | completed | 0.8333333333333334 | inconclusive | adopt | True |  |
| train | r-89603e49 | exp-09 | Qwen/Qwen3-1.7B-Base | 34254 | other | exp-06 |  | completed |  | inconclusive | reject | False |  |
| train | r-89603e49 | exp-10 | Qwen/Qwen3-1.7B-Base | 38078 | other | exp-08 |  | completed | 0.8333333333333334 | supported | adopt | False |  |
| train | r-89ab4cc5 | exp-01 | Qwen/Qwen3-4B-Base | 88 | sft | base_model | GSM8K train (agent's words, [43], [62]); the loader call in the script text is truncated in the stream | killed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-02 | Qwen/Qwen3-4B-Base | 186 | merge | exp-01 |  | completed | 0.05 | inconclusive | reject | True |  |
| train | r-89ab4cc5 | exp-03 | Qwen/Qwen3-4B-Base | 187 | merge | exp-01 |  | completed |  | inconclusive | abandon_line | True |  |
| train | r-89ab4cc5 | exp-04 | Qwen/Qwen3-4B-Base | 249 | sft | base_model | GSM8K train (same in-script pipeline as exp-01); the loader call in the script text is truncated in the stream | completed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-05 | Qwen/Qwen3-4B-Base | 267 | merge | exp-04 |  | completed | 0.1 | inconclusive | reject | True |  |
| train | r-89ab4cc5 | exp-06 | Qwen/Qwen3-4B-Base | 289 | sft | base_model | GSM8K train (same in-script pipeline as exp-01); one step consumes a single batch | completed |  | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-07 | Qwen/Qwen3-4B-Base | 295 | merge | exp-06 |  | completed | 0.5 | inconclusive | adopt | True |  |
| train | r-89ab4cc5 | exp-08 | Qwen/Qwen3-4B-Base | 323 | other | exp-07 |  | completed |  | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 159 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.6206 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 202 | merge | exp-01 |  | completed | 0.4 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 213 | sft | base_model | HF id: openai/gsm8k (main, train split),HF id: meta-math/MetaMathQA | killed | 0.5005455613136292 | inconclusive | abandon_line | True |  |
| train | r-8c4cb1bc | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 281 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.4112212657928467 | inconclusive | adopt | True |  |
| train | r-8c4cb1bc | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 292 | merge | exp-04 |  | completed | 0.087 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 317 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.08 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 411 | merge | exp-01 |  | completed | 0.14 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 514 | sft | base_model | HF id: openai/gsm8k (main, train split) | completed | 0.18 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 752 | merge | exp-01 |  | completed | 0.127 | contradicted | reject | True |  |
| train | r-8c4cb1bc | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 793 | other | exp-02 |  | completed | 0.38666666666666666 | supported | adopt | True |  |
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
| train | r-8ec271ed | exp-12 | Qwen/Qwen3-4B-Base | 246 | other | exp-11 |  | completed |  | inconclusive | adopt | True |  |
| train | r-9142b2d3 | exp-01 | Qwen/Qwen3-4B-Base | 52 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | failed |  | inconclusive | iterate | False |  |
| train | r-9142b2d3 | exp-02 | Qwen/Qwen3-4B-Base | 59 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | killed |  | inconclusive | iterate | False |  |
| train | r-9142b2d3 | exp-03 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k | completed | 0.133 | inconclusive | reject | False |  |
| train | r-9142b2d3 | exp-04 | Qwen/Qwen3-4B-Base | 119 | sft | base_model | openai/gsm8k | completed | 0.7266666666666667 | supported | adopt | True |  |
| train | r-9142b2d3 | exp-05 | Qwen/Qwen3-4B-Base | 150 | decode-config | exp-04 |  | completed | 0.62 | inconclusive | reject | True |  |
| train | r-9142b2d3 | exp-06 | Qwen/Qwen3-4B-Base | 156 | decode-config | exp-04 |  | completed | 0.76 | inconclusive | adopt | False |  |
| train | r-9142b2d3 | exp-07 | Qwen/Qwen3-4B-Base | 168 | other | exp-04 |  | completed | 0.68 | inconclusive | reject | True |  |
| train | r-9142b2d3 | exp-08 | Qwen/Qwen3-4B-Base | 182 | sft | base_model | openai/gsm8k,microsoft/orca-math-word-problems-200k,meta-math/MetaMathQA | completed | 0.7533333333333333 | inconclusive | adopt | True |  |
| train | r-9142b2d3 | exp-09 | Qwen/Qwen3-4B-Base | 200 | sft | exp-08 | openai/gsm8k | killed |  | inconclusive | abandon_line | True |  |
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
| train | r-928e1ff3 | exp-12 | Qwen/Qwen3-1.7B-Base | 1242 | other | exp-11 |  | completed | 0.84 | inconclusive | adopt | False |  |
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
| train | r-9395d5af | exp-12 | Qwen/Qwen3-4B-Base | 36735 | decode-config | exp-05 |  | completed | 0.86657 | supported | adopt | True |  |
| train | r-94f796fd | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 121 | sft | base_model | HF id: openai/gsm8k | completed | 0.1 | contradicted | reject | True |  |
| train | r-94f796fd | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 190 | sft | base_model | HF id: openai/gsm8k | completed |  | inconclusive | adopt | True |  |
| train | r-94f796fd | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 213 | sft | exp-02 | HF id: openai/gsm8k | completed | 0.52 | supported | reject | True |  |
| train | r-94f796fd | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 228 | sft | exp-02 | HF id: openai/gsm8k | completed | 0.41 | inconclusive | adopt | True |  |
| train | r-94f796fd | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 290 | sft | exp-04 | HF id: openai/gsm8k | completed | 0.64 | supported | adopt | True |  |
| train | r-94f796fd | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 333 | sft | exp-05 | HF id: openai/gsm8k | completed |  | inconclusive | reject | True |  |
| train | r-94f796fd | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 355 | other | exp-05 |  | completed | 0.5666666666666667 | inconclusive | adopt | False |  |
| train | r-96346341 | exp-01 | Qwen/Qwen3-4B-Base | 418 | sft | base_model | derived: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k, math + augmented_math) + microsoft/orca-math-word-problems-200k + openai/gsm8k train (few-shot pool) | killed |  | inconclusive | abandon_line | False |  |
| train | r-96346341 | exp-02 | Qwen/Qwen3-4B-Base | 713 | sft | base_model | derived: nvidia/OpenMathInstruct-2 (gsm8k + augmented_gsm8k, math + augmented_math) + microsoft/orca-math-word-problems-200k + openai/gsm8k train (few-shot pool) | completed | 0.887 | supported | adopt | False |  |
| train | r-96346341 | exp-03 | Qwen/Qwen3-4B-Base | 950 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | failed |  | inconclusive | abandon_line | False |  |
| train | r-96346341 | exp-04 | Qwen/Qwen3-4B-Base | 1069 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | failed |  | inconclusive | abandon_line | True |  |
| train | r-96346341 | exp-05 | Qwen/Qwen3-4B-Base | 1086 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-96346341 | exp-06 | Qwen/Qwen3-4B-Base | 1171 | grpo | exp-02 | HF openai/gsm8k (train split), prompts only | killed | 0.912 | supported | adopt | True |  |
| train | r-96346341 | exp-07 | Qwen/Qwen3-4B-Base | 1270 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-01 | Qwen/Qwen3-4B-Base | 85 | sft | base_model | HF openai/gsm8k (config main, split train) + HF meta-math/MetaMathQA (GSM_* subset) | completed | 0.36 | inconclusive | reject | False |  |
| train | r-96bf32c3 | exp-02 | Qwen/Qwen3-4B-Base | 212 | sft | base_model | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.64 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-03 | Qwen/Qwen3-4B-Base | 261 | sft | base_model | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.6666666666666666 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-04 | Qwen/Qwen3-4B-Base | 271 | other | exp-02 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-05 | Qwen/Qwen3-4B-Base | 326 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-06 | Qwen/Qwen3-4B-Base | 339 | sft | exp-03 | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_1M) | completed | 0.7 | supported | adopt | True |  |
| train | r-96bf32c3 | exp-07 | Qwen/Qwen3-4B-Base | 381 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-96bf32c3 | exp-08 | Qwen/Qwen3-4B-Base | 391 | sft | exp-06 | HF openai/gsm8k (main, train) + HF nvidia/OpenMathInstruct-2 (split train_2M) | completed | 0.66 | contradicted | reject | True |  |
| train | r-98b1304c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 54 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | failed |  | inconclusive | iterate | False |  |
| train | r-98b1304c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 63 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | failed |  | inconclusive | iterate | False |  |
| train | r-98b1304c | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 74 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | adopt | False |  |
| train | r-98b1304c | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 119 | merge | exp-03 |  | completed | 0.125 | inconclusive | reject | False |  |
| train | r-98b1304c | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 167 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 190 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 209 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | completed | 0.2 | supported | adopt | True |  |
| train | r-98b1304c | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 233 | sft | base_model | openai/gsm8k (config main, split=train), loaded inside the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-98b1304c | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 261 | other | exp-07 |  | completed | 0.31 | inconclusive | adopt | False |  |
| train | r-9e0ad3aa | exp-01 | Qwen/Qwen3-1.7B-Base | 82 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM') | failed |  | inconclusive | iterate | False |  |
| train | r-9e0ad3aa | exp-02 | Qwen/Qwen3-1.7B-Base | 88 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM') | completed | 0.03 | inconclusive | reject | False |  |
| train | r-9e0ad3aa | exp-03 | Qwen/Qwen3-1.7B-Base | 134 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.04 | inconclusive | adopt | False |  |
| train | r-9e0ad3aa | exp-04 | Qwen/Qwen3-1.7B-Base | 180 | decode-config | exp-03 |  | completed | 0.4 | supported | reject | False |  |
| train | r-9e0ad3aa | exp-05 | Qwen/Qwen3-1.7B-Base | 190 | sft | base_model | HF meta-math/MetaMathQA (split=train), filtered type.startswith('GSM'); 8-shot prefix from HF openai/gsm8k (main, split=train) | completed | 0.02 | contradicted | adopt | True |  |
| train | r-9e0ad3aa | exp-06 | Qwen/Qwen3-1.7B-Base | 214 | decode-config | exp-05 |  | completed | 0.03 | inconclusive | adopt | False |  |
| train | r-9f1c9470 | exp-01 | Qwen/Qwen3-1.7B-Base | 90 | sft | base_model | HF openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | True |  |
| train | r-9f1c9470 | exp-02 | Qwen/Qwen3-1.7B-Base | 127 | merge | exp-01 |  | completed | 0.175 | inconclusive | adopt | False |  |
| train | r-9f1c9470 | exp-03 | Qwen/Qwen3-1.7B-Base | 161 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.025 | contradicted | reject | True |  |
| train | r-9f1c9470 | exp-04 | Qwen/Qwen3-1.7B-Base | 235 | sft | base_model | HF openai/gsm8k (main, split=train) | completed |  | inconclusive | adopt | True |  |
| train | r-9f1c9470 | exp-05 | Qwen/Qwen3-1.7B-Base | 248 | merge | exp-04 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-9f1c9470 | exp-06 | Qwen/Qwen3-1.7B-Base | 291 | other | exp-02 |  | completed | 0.18666666666666668 | inconclusive | adopt | False |  |
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
| train | r-a2777ce1 | exp-11 | Qwen/Qwen3-1.7B-Base | 210 | other | exp-08 |  | completed | 0.573 | inconclusive | adopt | False |  |
| train | r-a2777ce1 | exp-12 | Qwen/Qwen3-1.7B-Base | 216 | sft | base_model | openai/gsm8k | completed | 0.333 | contradicted | reject | False |  |
| train | r-a2777ce1 | exp-13 | Qwen/Qwen3-1.7B-Base | 236 | sft | base_model | openai/gsm8k | completed | 0.1 | contradicted | reject | False |  |
| train | r-a3f23d29 | exp-01 | Qwen/Qwen3-4B-Base | 196 | sft | base_model | local (openai/gsm8k train + meta-math/MetaMathQA) | completed | 0.6266666666666667 | inconclusive | adopt | True |  |
| train | r-a3f23d29 | exp-02 | Qwen/Qwen3-4B-Base | 401 | decode-config | exp-01 |  | completed | 0.84 | supported | adopt | True |  |
| train | r-a3f23d29 | exp-03 | Qwen/Qwen3-4B-Base | 501 | rft | base_model | local (openai/gsm8k train + synthetic:self + meta-math/MetaMathQA),synthetic:self (sampled from exp-02's checkpoint) | completed | 0.84 | contradicted | reject | True |  |
| train | r-a3f23d29 | exp-04 | Qwen/Qwen3-4B-Base | 650 | sft | base_model | local (openai/gsm8k train + synthetic:self + nvidia/OpenMathInstruct-2),nvidia/OpenMathInstruct-2 | completed | 0.846 | inconclusive | adopt | True |  |
| train | r-a3f23d29 | exp-05 | Qwen/Qwen3-4B-Base | 697 | sft | base_model | local (openai/gsm8k train + synthetic:self + nvidia/OpenMathInstruct-2 + meta-math/MetaMathQA) | completed | 0.86 | supported | adopt | True |  |
| train | r-a436c040 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 39 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | failed |  | inconclusive | iterate | True |  |
| train | r-a436c040 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 45 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | failed |  | inconclusive | iterate | True |  |
| train | r-a436c040 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 51 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | False |  |
| train | r-a436c040 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 59 | merge | exp-03 |  | completed | 0.05 | contradicted | reject | True |  |
| train | r-a436c040 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 91 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-a436c040 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 97 | merge | exp-05 |  | completed | 0.35 | supported | reject | True |  |
| train | r-a436c040 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 109 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | completed |  | inconclusive | adopt | True |  |
| train | r-a436c040 | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 124 | merge | exp-07 |  | completed | 0.425 | supported | adopt | True |  |
| train | r-a436c040 | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 133 | sft | base_model | openai/gsm8k (config main, train split), loaded by the training script | killed |  | inconclusive | abandon_line | True |  |
| train | r-a436c040 | exp-10 | HuggingFaceTB/SmolLM3-3B-Base | 156 | other | exp-08 |  | completed | 0.4866666666666667 | inconclusive | adopt | True |  |
| train | r-a4de3854 | exp-01 | Qwen/Qwen3-1.7B-Base | 62 | sft | base_model | local,local | killed |  | inconclusive | abandon_line | False |  |
| train | r-a4de3854 | exp-02 | Qwen/Qwen3-1.7B-Base | 66 | sft | base_model | local,local | completed | 0.15333333333333332 | inconclusive | reject | False |  |
| train | r-a4de3854 | exp-03 | Qwen/Qwen3-1.7B-Base | 158 | sft | base_model | local,local | completed | 0.22 | supported | reject | True |  |
| train | r-a4de3854 | exp-04 | Qwen/Qwen3-1.7B-Base | 194 | sft | base_model | local,local | completed | 0.13333333333333333 | contradicted | reject | True |  |
| train | r-a4de3854 | exp-05 | Qwen/Qwen3-1.7B-Base | 240 | sft | base_model | local,local | completed | 0.16666666666666666 | inconclusive | adopt | True |  |
| train | r-a4de3854 | exp-06 | Qwen/Qwen3-1.7B-Base | 278 | grpo | exp-05 | openai/gsm8k (train split, loaded inside train_grpo.py; no local file) | killed |  | inconclusive | abandon_line | False |  |
| train | r-a4de3854 | exp-07 | Qwen/Qwen3-1.7B-Base | 288 | grpo | exp-05 | openai/gsm8k (train split, loaded inside train_grpo.py; no local file) | killed |  | inconclusive | adopt | False |  |
| train | r-a4de3854 | exp-08 | Qwen/Qwen3-1.7B-Base | 322 | other | exp-07 |  | completed | 0.12 | contradicted | reject | False |  |
| train | r-a4de3854 | exp-09 | Qwen/Qwen3-1.7B-Base | 368 | sft | base_model | local,local | completed |  | inconclusive | adopt | True |  |
| train | r-a759897f | exp-01 | Qwen/Qwen3-1.7B-Base | 23924 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | failed |  | inconclusive | iterate | True |  |
| train | r-a759897f | exp-02 | Qwen/Qwen3-1.7B-Base | 26284 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | killed |  | inconclusive | abandon_line | False |  |
| train | r-a759897f | exp-03 | Qwen/Qwen3-1.7B-Base | 31594 | sft | base_model | local: openai/gsm8k train + nvidia/OpenMathInstruct-2 (gsm8k / augmented_gsm8k rows) | completed | 0.766 | inconclusive | adopt | True |  |
| train | r-a759897f | exp-04 | Qwen/Qwen3-1.7B-Base | 41076 | sft | exp-03 | synthetic:self,derived:exp-03 (STaR) + local (openai/gsm8k train, nvidia/OpenMathInstruct-2) | completed | 0.768 | inconclusive | adopt | True |  |
| train | r-a759897f | exp-05 | Qwen/Qwen3-1.7B-Base | 42663 | sft | base_model | derived:exp-03 (STaR) + local (openai/gsm8k train, nvidia/OpenMathInstruct-2) | completed | 0.76 | contradicted | adopt | False |  |
| train | r-a759897f | exp-06 | Qwen/Qwen3-1.7B-Base | 46442 | merge | exp-03 + exp-04 |  | failed |  | inconclusive | iterate | False |  |
| train | r-a759897f | exp-07 | Qwen/Qwen3-1.7B-Base | 46915 | merge | exp-03 + exp-04 |  | completed | 0.78 | supported | adopt | False |  |
| train | r-a759897f | exp-08 | Qwen/Qwen3-1.7B-Base | 47871 | merge | exp-03 + exp-04 + exp-05 |  | completed | 0.766 | contradicted | reject | False |  |
| train | r-a759897f | exp-09 | Qwen/Qwen3-1.7B-Base | 48225 | other | exp-07 |  | completed | 0.7627 | supported | adopt | False |  |
| train | r-a759897f | exp-10 | Qwen/Qwen3-1.7B-Base | 49858 | merge | exp-03 + exp-04 |  | completed | 0.782 | contradicted | reject | True |  |
| train | r-aaf3560a | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 18 | other | base_model |  | completed | 0.167 | inconclusive | reject | False |  |
| train | r-aaf3560a | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 64 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.1 | inconclusive | reject | False |  |
| train | r-aaf3560a | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 87 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.2 | supported | reject | True |  |
| train | r-aaf3560a | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 107 | sft | base_model | HF id: openai/gsm8k (config main, train split) | completed | 0.34 | inconclusive | adopt | True |  |
| train | r-aaf3560a | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 125 | sft | base_model | HF id: openai/gsm8k (config main, train split) | killed | 0.33 | supported | adopt | False |  |
| train | r-aaf3560a | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 144 | sft | exp-05 | HF id: openai/gsm8k (config main, train split) | completed | 0.25 | contradicted | reject | False |  |
| train | r-aaf3560a | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 153 | other | exp-05 |  | completed | 0.233 | inconclusive | reject | True |  |
| train | r-aaf3560a | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 187 | other | exp-04 |  | completed | 0.34 | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 138 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) | killed |  | inconclusive | abandon_line | False |  |
| train | r-ac9606db | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 208 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) | completed | 0.593 | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 299 | decode-config | exp-02 |  | completed | 0.767 | supported | adopt | True |  |
| train | r-ac9606db | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 309 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 318 | sft | base_model | HF: openai/gsm8k (main, train) x2 + meta-math/MetaMathQA (GSM_Rephrased, GSM_AnsAug) + microsoft/orca-math-word-problems-200k | killed |  | inconclusive | abandon_line | False |  |
| train | r-ac9606db | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 392 | rft | base_model | synthetic:self,derived:exp-06 (data/reject.jsonl x2) + data/sft2.jsonl subsample | completed | 0.793 | supported | adopt | True |  |
| train | r-ac9606db | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 454 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-ac9606db | exp-08 | HuggingFaceTB/SmolLM3-3B-Base | 523 | rft | base_model | synthetic:self,derived:exp-06 (data/reject.jsonl) + derived:exp-08 (data/reject2.jsonl) + data/sft2.jsonl subsample | completed | 0.793 | contradicted | adopt | True |  |
| train | r-ac9606db | exp-09 | HuggingFaceTB/SmolLM3-3B-Base | 571 | other | exp-08 |  | completed | 0.78 | inconclusive | adopt | False |  |
| train | r-b351b70e | exp-01 | Qwen/Qwen3-4B-Base | 44 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.475 | inconclusive | reject | False |  |
| train | r-b351b70e | exp-02 | Qwen/Qwen3-4B-Base | 52 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.435 | contradicted | reject | False |  |
| train | r-b351b70e | exp-03 | Qwen/Qwen3-4B-Base | 84 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.545 | supported | reject | False |  |
| train | r-b351b70e | exp-04 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.63 | supported | reject | False |  |
| train | r-b351b70e | exp-05 | Qwen/Qwen3-4B-Base | 106 | sft | base_model | openai/gsm8k (config main, split train), loaded from the hub inside the training script | completed | 0.745 | supported | adopt | False |  |
| train | r-b8779e0c | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 77 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.5266666666666666 | inconclusive | adopt | True |  |
| train | r-b8779e0c | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 110 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types + open-r1/OpenR1-Math-220k) | completed | 0.32 | contradicted | reject | True |  |
| train | r-b8779e0c | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 134 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.58 | supported | adopt | True |  |
| train | r-b8779e0c | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 158 | sft | base_model | local (built from HF gsm8k main/train + meta-math/MetaMathQA GSM_* types) | completed | 0.49333333333333335 | contradicted | reject | True |  |
| train | r-b8779e0c | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 170 | other | exp-03 |  | completed | 0.593 | inconclusive | adopt | False |  |
| train | r-bc465eaf | exp-01 | Qwen/Qwen3-1.7B-Base | 49 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.05 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-02 | Qwen/Qwen3-1.7B-Base | 74 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | abandon_line | True |  |
| train | r-bc465eaf | exp-03 | Qwen/Qwen3-1.7B-Base | 137 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.0 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-04 | Qwen/Qwen3-1.7B-Base | 187 | other | exp-03 |  | completed |  | inconclusive | abandon_line | False |  |
| train | r-bc465eaf | exp-05 | Qwen/Qwen3-1.7B-Base | 199 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-bc465eaf | exp-06 | Qwen/Qwen3-1.7B-Base | 211 | other | exp-05 |  | completed | 0.0 | inconclusive | reject | False |  |
| train | r-bc465eaf | exp-07 | Qwen/Qwen3-1.7B-Base | 229 | sft | base_model | HF openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-bc465eaf | exp-08 | Qwen/Qwen3-1.7B-Base | 237 | other | exp-07 |  | completed |  | inconclusive | adopt | False |  |
| train | r-bcc8974e | exp-01 | Qwen/Qwen3-1.7B-Base | 34 | sft | base_model | HF gsm8k (config main), train split only | failed |  | inconclusive | abandon_line | False |  |
| train | r-bcc8974e | exp-02 | Qwen/Qwen3-1.7B-Base | 44 | sft | base_model | HF gsm8k (config main), train split only | killed |  | inconclusive | abandon_line | False |  |
| train | r-bcc8974e | exp-03 | Qwen/Qwen3-1.7B-Base | 48 | sft | base_model | HF gsm8k (config main), train split only | completed | 0.14 | inconclusive | adopt | False |  |
| train | r-bcc8974e | exp-04 | Qwen/Qwen3-1.7B-Base | 56 | other | exp-03 |  | completed |  | inconclusive | adopt | False |  |
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
| train | r-bda2d6c1 | exp-15 | Qwen/Qwen3-1.7B-Base | 596 | sft | base_model | gsm8k/main:train,derived:exp-07 | completed | 0.726 | supported | adopt | False |  |
| train | r-bfd319db | exp-01 | Qwen/Qwen3-1.7B-Base | 113 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.02 | inconclusive | reject | True |  |
| train | r-bfd319db | exp-02 | Qwen/Qwen3-1.7B-Base | 131 | merge | exp-01 |  | completed | 0.02 | inconclusive | reject | False |  |
| train | r-bfd319db | exp-03 | Qwen/Qwen3-1.7B-Base | 148 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.075 | contradicted | adopt | False |  |
| train | r-bfd319db | exp-04 | Qwen/Qwen3-1.7B-Base | 154 | merge | exp-03 |  | completed | 0.075 | contradicted | adopt | False |  |
| train | r-bfd319db | exp-05 | Qwen/Qwen3-1.7B-Base | 165 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.05 | contradicted | reject | True |  |
| train | r-bfd319db | exp-06 | Qwen/Qwen3-1.7B-Base | 178 | merge | exp-05 |  | completed | 0.05 | contradicted | reject | False |  |
| train | r-bfd319db | exp-07 | Qwen/Qwen3-1.7B-Base | 187 | other | exp-04 |  | completed | 0.075 | supported | adopt | False |  |
| train | r-c0173ea9 | exp-01 | Qwen/Qwen3-4B-Base | 59 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-02 | Qwen/Qwen3-4B-Base | 74 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-03 | Qwen/Qwen3-4B-Base | 83 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-04 | Qwen/Qwen3-4B-Base | 92 | sft | base_model | HF:openai/gsm8k (config main, split train) | failed |  | inconclusive | iterate | False |  |
| train | r-c0173ea9 | exp-05 | Qwen/Qwen3-4B-Base | 98 | sft | base_model | HF:openai/gsm8k (config main, split train) | killed |  | inconclusive | adopt | False |  |
| train | r-c0173ea9 | exp-06 | Qwen/Qwen3-4B-Base | 137 | sft | exp-05 | HF:openai/gsm8k (config main, split train) | completed |  | inconclusive | adopt | True |  |
| train | r-c0173ea9 | exp-07 | Qwen/Qwen3-4B-Base | 156 | merge | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-c0173ea9 | exp-08 | Qwen/Qwen3-4B-Base | 161 | other | exp-07 |  | completed |  | inconclusive | adopt | False |  |
| train | r-c51755a7 | exp-01 | Qwen/Qwen3-1.7B-Base | 92 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k | completed | 0.18 | inconclusive | reject | False |  |
| train | r-c51755a7 | exp-02 | Qwen/Qwen3-1.7B-Base | 138 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.46 | supported | reject | False |  |
| train | r-c51755a7 | exp-03 | Qwen/Qwen3-1.7B-Base | 160 | sft | base_model | HuggingFaceH4/Bespoke-Stratos-17k,openai/gsm8k (config main, split train) | completed | 0.367 | contradicted | adopt | False |  |
| train | r-c51755a7 | exp-04 | Qwen/Qwen3-1.7B-Base | 186 | decode-config | exp-03 |  | completed | 0.447 | supported | reject | False |  |
| train | r-c51755a7 | exp-05 | Qwen/Qwen3-1.7B-Base | 204 | sft | base_model | openai/gsm8k (config main, split train) | completed | 0.48 | supported | reject | False |  |
| train | r-c51755a7 | exp-06 | Qwen/Qwen3-1.7B-Base | 220 | sft | base_model | AI-MO/NuminaMath-CoT,openai/gsm8k (config main, split train) | completed | 0.52 | supported | adopt | False |  |
| train | r-c51755a7 | exp-07 | Qwen/Qwen3-1.7B-Base | 230 | other | exp-06 |  | completed |  | inconclusive | adopt | False |  |
| train | r-c7ff2a60 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 97 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | failed |  | inconclusive | iterate | True |  |
| train | r-c7ff2a60 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 105 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | killed |  | inconclusive | adopt | True |  |
| train | r-c7ff2a60 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 218 | other | exp-02 |  | completed | 0.167 | inconclusive | reject | False |  |
| train | r-c7ff2a60 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 241 | sft | exp-02 | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.22 | inconclusive | adopt | True |  |
| train | r-c7ff2a60 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 284 | decode-config | exp-04 |  | completed | 0.14 | contradicted | reject | False |  |
| train | r-c7ff2a60 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 310 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.28 | inconclusive | reject | True |  |
| train | r-c7ff2a60 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 330 | sft | base_model | openai/gsm8k (main, train),meta-math/MetaMathQA (train) | completed | 0.413 | supported | adopt | True |  |
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
| train | r-cbcfc798 | exp-11 | Qwen/Qwen3-1.7B-Base | 326 | decode-config | exp-10 |  | completed | 0.727 | supported | adopt | True |  |
| train | r-cbcfc798 | exp-12 | Qwen/Qwen3-1.7B-Base | 339 | sft | exp-06 | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-cbcfc798 | exp-13 | Qwen/Qwen3-1.7B-Base | 370 | sft | exp-06 | HF:openai/gsm8k (main, split=train),HF:meta-math/MetaMathQA (types GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR) | killed |  | inconclusive | abandon_line | True |  |
| train | r-cf5932d6 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 163 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus the openai/gsm8k train split | killed |  | inconclusive | abandon_line | False |  |
| train | r-cf5932d6 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 244 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus the openai/gsm8k train split | completed | 0.56 | inconclusive | adopt | False |  |
| train | r-cf5932d6 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 321 | sft | exp-02 | None | completed | 0.5533 | inconclusive | reject | False |  |
| train | r-cf5932d6 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 330 | other | exp-02 |  | completed |  | inconclusive | reject | False |  |
| train | r-cf5932d6 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 379 | sft | base_model | OpenMathInstruct-2 subset held locally as /home/ben/task/work/data/omi2_gsm8k.parquet, plus meta-math/MetaMathQA (MetaMathQA-395K.json, GSM_* types only), plus the openai/gsm8k train split | completed | 0.5867 | inconclusive | adopt | False |  |
| train | r-cf5932d6 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 449 | other | exp-05 |  | completed | 0.5733 | inconclusive | adopt | False |  |
| train | r-d2611dd8 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 121 | sft | base_model | HF openai/gsm8k (train) + HF meta-math/MetaMathQA (GSM_* subsets) | completed | 0.54 | inconclusive | reject | False |  |
| train | r-d2611dd8 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 174 | sft | base_model | HF openai/gsm8k (train) + HF meta-math/MetaMathQA (GSM_* subsets) | completed | 0.58 | inconclusive | adopt | False |  |
| train | r-d2611dd8 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 224 | rft | exp-02 | synthetic:self (exp-02 samples, kept when the final answer matches gsm8k train gold) + HF meta-math/MetaMathQA + HF openai/gsm8k | completed | 0.716 | supported | adopt | False |  |
| train | r-d2611dd8 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 259 | rft | exp-03 | synthetic:self (exp-03 samples, answer-filtered) + HF meta-math/MetaMathQA + HF openai/gsm8k | completed |  | inconclusive | reject | False |  |
| train | r-d2611dd8 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 286 | rft | base_model | synthetic:self (exp-03 samples, answer-filtered) + HF openai/gsm8k (train) + HF meta-math/MetaMathQA | completed | 0.596 | contradicted | reject | False |  |
| train | r-d2611dd8 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 305 | other | exp-03 |  | completed | 0.692 | inconclusive | adopt | False |  |
| train | r-d2611dd8 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 318 | rft | exp-03 | synthetic:self (exp-02 and exp-03 samples, answer-filtered) + HF openai/gsm8k (train) | completed | 0.68 | contradicted | reject | False |  |
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
| train | r-d4a6c52f | exp-21 | Qwen/Qwen3-1.7B-Base | 605 | other | exp-18 |  | completed | 0.1933 | supported | adopt | False |  |
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
| train | r-d7327838 | exp-11 | Qwen/Qwen3-1.7B-Base | 437 | other | exp-10 |  | completed | 0.6667 | inconclusive | adopt | False |  |
| train | r-db1eeb72 | exp-01 | Qwen/Qwen3-1.7B-Base | 86 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.703 | inconclusive | adopt | True |  |
| train | r-db1eeb72 | exp-02 | Qwen/Qwen3-1.7B-Base | 181 | sft | exp-01 | HF openai/gsm8k (main, split=train) | completed | 0.35 | inconclusive | adopt | True |  |
| train | r-db1eeb72 | exp-03 | Qwen/Qwen3-1.7B-Base | 252 | sft | exp-02 | HF openai/gsm8k (main, split=train),HF EleutherAI/asdiv (split=validation),HF mwpt5/MAWPS (split=train),HF cq01/mawps-asdiv-a_svamp (splits train + validation) | completed | 0.25 | contradicted | reject | True |  |
| train | r-db1eeb72 | exp-04 | Qwen/Qwen3-1.7B-Base | 274 | other | exp-02 |  | completed | 0.35 | inconclusive | adopt | False |  |
| train | r-dc28e30d | exp-01 | Qwen/Qwen3-1.7B-Base | 147 | sft | base_model | HF openai/gsm8k (main, split=train) | completed | 0.46 | inconclusive | reject | True |  |
| train | r-dc28e30d | exp-02 | Qwen/Qwen3-1.7B-Base | 309 | sft | base_model | HF openai/gsm8k (train) x3 + HF meta-math/MetaMathQA (GSM types) | failed |  | inconclusive | abandon_line | True |  |
| train | r-dc28e30d | exp-03 | Qwen/Qwen3-1.7B-Base | 358 | sft | base_model | HF openai/gsm8k (train) x3 + HF meta-math/MetaMathQA (GSM types) | completed | 0.63 | supported | adopt | True |  |
| train | r-dc28e30d | exp-04 | Qwen/Qwen3-1.7B-Base | 483 | other | exp-03 |  | completed | 0.63 | inconclusive | adopt | False |  |
| train | r-dc28e30d | exp-05 | Qwen/Qwen3-1.7B-Base | 531 | grpo | exp-03 | HF openai/gsm8k (main, split=train) | completed | 0.8 | supported | iterate | True |  |
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
| train | r-e2ebe966 | exp-11 | Qwen/Qwen3-4B-Base | 490 | other | exp-10 |  | completed | 0.3 | inconclusive | adopt | False |  |
| train | r-e5ed6bd2 | exp-01 | Qwen/Qwen3-1.7B-Base | 52 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | False |  |
| train | r-e5ed6bd2 | exp-02 | Qwen/Qwen3-1.7B-Base | 64 | sft | base_model | HF: gsm8k main, train split | completed | 0.02666666666666667 | inconclusive | reject | False |  |
| train | r-e5ed6bd2 | exp-03 | Qwen/Qwen3-1.7B-Base | 118 | sft | base_model | HF: gsm8k main, train split | completed | 0.02 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-04 | Qwen/Qwen3-1.7B-Base | 148 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | True |  |
| train | r-e5ed6bd2 | exp-05 | Qwen/Qwen3-1.7B-Base | 154 | sft | base_model | HF: gsm8k main, train split | failed |  | inconclusive | abandon_line | False |  |
| train | r-e5ed6bd2 | exp-06 | Qwen/Qwen3-1.7B-Base | 160 | sft | base_model | HF: gsm8k main, train split | completed | 0.17333333333333334 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-07 | Qwen/Qwen3-1.7B-Base | 191 | sft | base_model | HF: gsm8k main, train split | completed | 0.07808946171341925 | contradicted | reject | True |  |
| train | r-e5ed6bd2 | exp-08 | Qwen/Qwen3-1.7B-Base | 197 | other | base_model |  | completed | 0.08 | inconclusive | adopt | True |  |
| train | r-e79c6a8d | exp-01 | Qwen/Qwen3-4B-Base | 17760 | sft | base_model | synthetic:self | failed |  | inconclusive | iterate | True |  |
| train | r-e79c6a8d | exp-02 | Qwen/Qwen3-4B-Base | 17977 | sft | base_model | synthetic:self | completed | 0.9066666666666666 | supported | reject | True |  |
| train | r-e79c6a8d | exp-03 | Qwen/Qwen3-4B-Base | 28270 | sft | base_model | derived:exp-02 | completed | 0.9 | inconclusive | adopt | False |  |
| train | r-e79c6a8d | exp-04 | Qwen/Qwen3-4B-Base | 32291 | decode-config | exp-03 |  | completed | 0.8620166793025019 | inconclusive | reject | False |  |
| train | r-e79c6a8d | exp-05 | Qwen/Qwen3-4B-Base | 32569 | decode-config | exp-04 |  | completed |  | inconclusive | adopt | False |  |
| train | r-e79c6a8d | exp-06 | Qwen/Qwen3-4B-Base | 35285 | decode-config | exp-05 |  | completed | 0.8794541319181198 | supported | adopt | True |  |
| train | r-e79c6a8d | exp-07 | Qwen/Qwen3-4B-Base | 36424 | decode-config | exp-05 |  | completed | 0.8933 | inconclusive | adopt | False |  |
| train | r-eb6370c9 | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 55 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | failed |  | inconclusive | abandon_line | False |  |
| train | r-eb6370c9 | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 65 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | killed |  | inconclusive | abandon_line | False |  |
| train | r-eb6370c9 | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 67 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | adopt | False |  |
| train | r-eb6370c9 | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 104 | merge | exp-03 |  | completed | 0.12 | contradicted | reject | False |  |
| train | r-eb6370c9 | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 124 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | adopt | False |  |
| train | r-eb6370c9 | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 136 | merge | exp-05 |  | completed | 0.247 | supported | adopt | False |  |
| train | r-eb6370c9 | exp-07 | HuggingFaceTB/SmolLM3-3B-Base | 148 | sft | base_model | HF id openai/gsm8k (config main, train split), loaded inside the training script | completed |  | inconclusive | abandon_line | False |  |
| train | r-ee1ca44a | exp-01 | HuggingFaceTB/SmolLM3-3B-Base | 62 | sft | base_model | meta-math/MetaMathQA | completed | 0.1 | contradicted | adopt | False |  |
| train | r-ee1ca44a | exp-02 | HuggingFaceTB/SmolLM3-3B-Base | 110 | decode-config | exp-01 |  | completed | 0.7 | supported | reject | False |  |
| train | r-ee1ca44a | exp-03 | HuggingFaceTB/SmolLM3-3B-Base | 124 | sft | base_model | AI-MO/NuminaMath-CoT | completed | 0.5133333333333333 | contradicted | reject | False |  |
| train | r-ee1ca44a | exp-04 | HuggingFaceTB/SmolLM3-3B-Base | 152 | sft | base_model | meta-math/MetaMathQA | failed |  | inconclusive | iterate | False |  |
| train | r-ee1ca44a | exp-05 | HuggingFaceTB/SmolLM3-3B-Base | 166 | sft | base_model | meta-math/MetaMathQA | completed | 0.6333333333333333 | supported | adopt | False |  |
| train | r-ee1ca44a | exp-06 | HuggingFaceTB/SmolLM3-3B-Base | 184 | other | exp-05 |  | completed | 0.6 | inconclusive | adopt | False |  |
| train | r-f238cc6e | exp-01 | Qwen/Qwen3-4B-Base | 143 | sft | base_model | HF id: openai/gsm8k (main, train) | killed | 0.828125 | inconclusive | reject | True |  |
| train | r-f238cc6e | exp-02 | Qwen/Qwen3-4B-Base | 283 | sft | base_model | HF id: math-ai/TemplateGSM (templategsm-1000-1k, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-03 | Qwen/Qwen3-4B-Base | 301 | sft | base_model | HF id: math-ai/TemplateGSM (templategsm-1000-1k, train) | completed |  | inconclusive | adopt | True |  |
| train | r-f238cc6e | exp-04 | Qwen/Qwen3-4B-Base | 328 | sft | exp-03 | HF id: openai/gsm8k (main, train) | killed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-05 | Qwen/Qwen3-4B-Base | 333 | sft | exp-03 | HF id: openai/gsm8k (main, train) | failed |  | inconclusive | abandon_line | True |  |
| train | r-f238cc6e | exp-06 | Qwen/Qwen3-4B-Base | 339 | sft | exp-03 | HF id: openai/gsm8k (main, train) | killed | 0.8125 | contradicted | reject | True |  |
| train | r-f238cc6e | exp-07 | Qwen/Qwen3-4B-Base | 394 | sft | base_model | HF id: openai/gsm8k (main, train) | completed | 0.0 | inconclusive | adopt | True |  |
| train | r-f238cc6e | exp-08 | Qwen/Qwen3-4B-Base | 509 | decode-config | exp-07 |  | completed | 0.4 | supported | adopt | True |  |
| train | r-f238cc6e | exp-09 | Qwen/Qwen3-4B-Base | 552 | decode-config | exp-08 |  | completed | 0.64 | supported | adopt | True |  |
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
| train | r-f4df2c91 | exp-12 | Qwen/Qwen3-1.7B-Base | 160 | other | base_model |  | completed | 0.127 | inconclusive | adopt | True |  |
| train | r-f5bfab57 | exp-01 | Qwen/Qwen3-4B-Base | 58 | sft | base_model | HF openai/gsm8k (config main, split train) | killed |  | inconclusive | abandon_line | False |  |
| train | r-f5bfab57 | exp-02 | Qwen/Qwen3-4B-Base | 60 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.59 | inconclusive | reject | False |  |
| train | r-f5bfab57 | exp-03 | Qwen/Qwen3-4B-Base | 88 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.6 | inconclusive | reject | False |  |
| train | r-f5bfab57 | exp-04 | Qwen/Qwen3-4B-Base | 106 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.73 | supported | reject | True |  |
| train | r-f5bfab57 | exp-05 | Qwen/Qwen3-4B-Base | 120 | sft | base_model | HF openai/gsm8k (config main, split train) | completed | 0.6580742987111448 | supported | adopt | False |  |
