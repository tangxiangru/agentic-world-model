# GSM8K post-training of google/gemma-3-4b-pt — batch summary

Shipped model: `final_model/` — hardlinked copy of `ckpts/exp-04/final`
(md5 of shard 1 verified identical), plus a greedy `generation_config.json`.

## Score trail (all under `evaluate.py --limit 150 --max-connections 32 --gpu-memory-utilization 0.85`)

| stage | what changed | dev-150 |
|---|---|---|
| base snapshot | — | 0.033 |
| exp-01 | SFT, 59933 rendered chains, 2 epochs | 0.653 |
| exp-02 | same weights, greedy `generation_config.json` | **0.720** |
| exp-03 | + RFT on 10151 self-samples | 0.713 (rejected) |
| exp-04 | + 1 epoch on 96971 unseen rows | **0.773** |
| exp-05 | repeat of exp-06's plan | failed at save, unmeasured |
| exp-06 | + 1 epoch on 65560 more unseen rows | 0.720 (rejected) |

Confirmation of the shipped directory: `final_model` scored **0.740** at n=150
(`eval/final_model_dev150.json`) and **0.740 ± 0.022** at n=400
(`eval/final_model_dev400.json`).

## What the record actually supports

1. **Format was the whole first jump.** The pt base almost never emitted a
   scorer-readable answer line; SFT on targets ending in one `ANSWER: <int>`
   line then `<end_of_turn>` took format compliance from ~0 to 0.987 and the
   score from 0.033 to 0.653.
2. **The grader's decoding is the model's own file.** `evaluate.py` sends no
   temperature, inspect omits it from the request, and vLLM falls back to
   `generation_config.json` in the model directory. The pt snapshot's file says
   `do_sample: true, top_k: 64, top_p: 0.95`, so every answer was a
   temperature-1.0 sample. Shipping `temperature: 0.0` was worth +6.7 points on
   identical weights (exp-02).
3. **Unique teacher data was the only training-side lever that moved it.**
   +5.3 from 96971 unseen OpenMathInstruct-2 gsm8k rows (exp-04), and that run
   also drove malformed/runaway completions to exactly zero.
4. **Two things did not work.** Rejection-sampling fine-tuning on the model's
   own answer-verified chains: −0.7 (exp-03), despite an 89% pass@4 / 61% pass@1
   gap on train. A further epoch of the same augmented source: −5.3 (exp-06) —
   this source stops paying off somewhere past ~190k distinct problems.
5. **Greedy vLLM is not bit-reproducible.** The same directory scored 0.773 and
   0.740 on two runs of the identical protocol at n=150 (continuous batching
   changes reduction order). Treat n=150 deltas under ~5 points as noise; the
   n=400 number is the one to quote.

## Contamination

Every training file was checked against `../test_data.json` with
`../contamination_check.py`: 0 matches in each of 59933 / 107122 / 17624 / 65560
documents. All questions come from `openai/gsm8k` **train** or from
OpenMathInstruct-2's gsm8k-derived splits; no test item was read, paraphrased,
or used to seed generation.

## Cost of the two lost runs

- 0.45 h: RFT sampling crashed in post-processing with nothing written to disk,
  then its orphaned `VLLM::EngineCore` held 66 GB and blocked the retry.
- 1.55 h: exp-05 finished all 2049 steps and died in `trainer.save_model` —
  transformers refuses to re-save the greedy `generation_config` that vLLM
  needs. Fixed in `scripts/train_sft2.py`; the lesson is in exp-06's
  `situation.pitfalls_hit`.
