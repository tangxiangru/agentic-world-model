# GSM8K post-training of google/gemma-3-4b-pt

`final_model/` = the exp-02 SFT checkpoint with greedy decoding written into
`generation_config.json`. Weights are byte-identical to `ckpts/exp-03/greedy`
(md5 of both safetensors shards checked).

## Score

| | dev-150 | dev-500 |
|---|---|---|
| base `gemma-3-4b-pt` | 0.073 | – |
| `final_model/` | 0.740 / 0.713 (two runs) | **0.726 ± 0.020** |

The headline number is the n=500 one. Two runs of the identical greedy eval on
identical weights returned 0.740 and 0.713, so n=150 carries about ±3 points of
run variance on top of its ±3.7 binomial stderr; see exp-08.

## What was done

| card | intervention | result |
|---|---|---|
| exp-01 | measure the base model | 0.073. 72% of completions have no well-formed `ANSWER:` line, 44% run to the token cap - it never learned to stop |
| exp-02 | completion-only SFT, 162k rows (OpenMathInstruct-2 gsm8k/augmented_gsm8k + GSM8K-train gold), 1 epoch | **0.073 → 0.613**; format compliance 0.28 → 1.00, truncation 0.44 → 0.00 |
| exp-03 | `temperature: 0.0` in `generation_config.json` | **0.613 → 0.740**. vLLM was defaulting to the base config's `do_sample=true, top_k=64, top_p=0.95`, so every graded answer had been a temperature-1 sample |
| exp-04 | stage-2 SFT on 160k MetaMathQA GSM rows | 0.720, rejected |
| exp-05 | RFT: 53k of the model's own correct chains + 30k replay | 0.720, rejected |
| exp-06 | uniform weight soup of exp-02 / exp-04 / exp-05 | 0.727, rejected |
| exp-07 | re-score the two leaders at n=500 | 0.726 vs 0.722 - not separable; ship the single checkpoint |
| exp-08 | probe: does the harness's 10-shot prefix hurt? | yes, ~3 points (0.760 zero-shot vs 0.730 with prefix on 300 GSM8K-train questions). Not acted on - too small to verify in the time left |

Everything after exp-03 landed inside noise. The two interventions that
mattered were the first SFT stage and the decode config.

## Reproducing

```
python scripts/build_data.py  --out data/sft_v1.jsonl
python scripts/pack_data.py   --data data/sft_v1.jsonl --out data/packed_v1_4096.npz
python scripts/train_sft.py   --packed data/packed_v1_4096.npz --out ckpts/exp-02 \
                              --epochs 1.0 --lr 1e-5 --accum 12 --no-grad-ckpt --optim adamw_bnb_8bit
python scripts/make_final.py  --src ckpts/exp-02/final --dst final_model
```

`scripts/verify_template.py` asserts that the prompt the trainer sees is
byte-identical to what `templates/gemma3.jinja` renders for the grader.

## Decontamination

No GSM8K test item was read by any build script. Every training file was run
through `../contamination_check.py` against `../test_data.json`:

| file | docs | contaminated |
|---|---|---|
| `data/sft_v1.jsonl` | 162236 | 0 |
| `data/sft_v2.jsonl` | 160000 | 0 |
| `data/rft_mix_v1.jsonl` | 83121 | 0 |

Full record: `memory/index.md` and `memory/cards/exp-01..08.yaml`.
