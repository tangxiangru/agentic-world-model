# Final state

`final_model/` = weights of `ckpts/exp-08/final`, with `generation_config.json`
rewritten to greedy (`do_sample: false, temperature: 0.0`, `top_k`/`top_p` removed).
Verified loadable on CPU with transformers (4.30B params,
`Gemma3ForConditionalGeneration`) and scored from a fresh process by the
unmodified `evaluate.py`.

## Score on the official benchmark

| eval | n | accuracy | stderr | file |
|---|---:|---:|---:|---|
| base google/gemma-3-4b-pt | 150 | 0.040 | 0.016 | eval/exp-01_dev150.json |
| **final_model** | **500** | **0.712** | **0.020** | eval/final_model_dev500_exp08.json |
| final_model | 200 | 0.685 | 0.033 | eval/final_model_confirm200.json |
| final_model | 150 | 0.733 | 0.036 | eval/exp-08_dev150.json |
| exp-07 candidate (same 500 items) | 500 | 0.688 | 0.021 | eval/final_model_dev500.json |

The 500-item run is the number to quote; the 150-item slice of the test set is
noticeably easier than items 150-500, which is why the smaller runs read higher.

## How it was built

| card | intervention | official n=150 | greedy few-shot probe300 |
|---|---|---:|---:|
| exp-01 | base model, measured | 0.040 | - |
| exp-02 | SFT, 6973 gsm8k train rows, grader's template | 0.4667 | 0.440 |
| exp-03 | (same as exp-04, OOM at step 4) | - | - |
| exp-04 | SFT, 213885-row gsm8k/OpenMathInstruct-2/MetaMathQA mixture | 0.400 | 0.4733 |
| exp-05 | greedy `generation_config.json`, same weights | 0.4667 | - |
| exp-06 | RFT on exp-04's own correct terminated samples | 0.4733 | 0.4867 |
| exp-07 | + harness's 10-shot demo block in 65% of training prompts | 0.7200 | 0.7400 |
| exp-08 | + 24000 rows of fresh reasoning data, demos in 70% | 0.7333 | 0.7633 |

The two decisive steps were exp-05 and exp-07, and neither was about data volume:

- **exp-05** — `evaluate.py` never sends a temperature, so vLLM takes its sampling
  defaults from the checkpoint's own `generation_config.json`. The pretrained model
  ships `do_sample: true, top_k: 64, top_p: 0.95`. Setting greedy was +6.7 pts on
  identical weights.
- **exp-07** — the harness always prepends ~2044 tokens of 10 GSM8K-train
  demonstrations. A model trained only on zero-shot rows answers correctly and then
  *invents a new problem*, continuing the demo pattern; `match(numeric=True,
  location="end")` then grades the fabricated problem's last number. Same weights
  scored 0.4867 with the demo block and 0.7033 without it. Training with the demo
  block in context drove answer-then-continue from 53/300 to 0/300: +24.7 pts.

## Data provenance and contamination

Every training file was checked with `../contamination_check.py` against
`../test_data.json`; all returned 0 contaminated documents.

| file | rows | source | check |
|---|---:|---|---|
| data/sft_gsm8k.jsonl | 6973 | openai/gsm8k main/train[0:6973] | analysis/contam_gsm8k.txt |
| data/sft_mix.jsonl | 213885 | + OpenMathInstruct-2 (gsm8k, augmented_gsm8k), MetaMathQA (GSM*) | analysis/contam_mix.txt |
| data/sft_rft.jsonl | 29740 | exp-04's own samples on those same questions | inline, 0 matches |
| data/sft_stage3.jsonl | 24000 | unused RFT rows + short sft_mix rows | inline, 0 matches |

GSM8K's *test* split was used only by `evaluate.py` and as the reference input to the
contamination checker. The last 500 items of the *train* split
(`data/probe500.jsonl`, indices 6973-7472) were held out of every training file and
used as the private probe; the 10-shot demonstrations embedded in training prompts are
the same train-split items `inspect_evals` draws with `seed=42`.

## Where the remaining errors are

57 of 300 probe items still fail, and none of them is a formatting failure: the official
set now returns 150/150 clean `ANSWER: <number>` lines and 0 unterminated completions.
What is left is arithmetic and multi-step reasoning — e.g. `train-6983`, a unit
conversion where the model multiplies hours by a 20-minute slot instead of by 60. The
natural next step is a second RFT round seeded from the questions the model never solved
(2464 of 15973 were unsolved in 4 samples, per `analysis/rft_stats.json`), which the
current RFT corpus is blind to by construction.
