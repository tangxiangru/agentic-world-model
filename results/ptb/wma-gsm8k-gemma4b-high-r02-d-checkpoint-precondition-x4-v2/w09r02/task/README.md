# Post-training `google/gemma-3-4b-pt` for GSM8K

`final_model/` holds the shipped checkpoint. Every experiment is carded under
`memory/cards/`; `memory/index.md` is the one-line-per-card index.

## Result

| card | intervention | n=150 accuracy |
|---|---|---:|
| exp-01 | base snapshot, measured | 0.0467 |
| exp-02 | full SFT, 67 472 rows | 0.6867 |
| exp-03 | same recipe, 141 914 rows | 0.7533 |
| exp-04 | + on-policy rejection-sampling round | **0.7667** |
| exp-05 | weight average of exp-03 and exp-04 | 0.762 (n=500) |
| exp-06 | second rejection-sampling round from exp-04 | 0.756 (n=500) |

`final_model/` = exp-04/final. Read at the grader's own default invocation
(`python evaluate.py --model-path final_model --limit 150`): **0.7667**.
Under the higher-powered protocol used to pick it (n=500, identical invocation
for every candidate): exp-04 **0.774**, soup 0.762, exp-03 0.756, exp-06 0.756.
Paired McNemar over the same 500 items puts every gap at p > 0.36, so exp-04 is
the best point estimate under both protocols rather than a proven winner; the
last two cards were rejected under a regression guard that only replaces
`final_model/` on a higher n=500 score.

## What mattered

1. **Termination and answer surface.** The pretrained checkpoint scored 0.0467
   not because it could not do the arithmetic but because it never stopped: 97
   of 150 completions ran on into invented new problems, and the grader
   (`inspect_ai.scorer.match(location="end", numeric=True)`) reads the *last*
   number in the completion. Training targets that end in `<end_of_turn>` after
   a single `ANSWER: <integer>` line took format failures from 89.5 % of
   failures to 6.4 % and were worth most of the +64 points in exp-02.
2. **Byte-identical rendering.** Training rows are rendered with
   `templates/gemma3.jinja` — the very file `evaluate.py` hands to vLLM — so the
   model is never trained on a string the grader cannot produce.
3. **Greedy decoding.** vLLM reads `generation_config.json` from the checkpoint.
   The base snapshot ships `do_sample: true, top_k: 64, top_p: 0.95`, so the
   graded decode is sampled. Every checkpoint written here sets
   `temperature: 0.0` and `eos_token_id: [1, 106]`.
4. **Data volume.** The retained-checkpoint curve in exp-02 (0.613 → 0.633 →
   0.653 → 0.687 as rows seen went 18k → 67k) said the mixture was not
   saturated; doubling it to 141 914 rows was worth +6.7.

## What did not work

- **A second rejection-sampling round** (exp-06) cost 1.8 points at n=500. Its
  training loss on its own samples was *lower* than round 1's (0.238 vs 0.251)
  while per-sample accuracy on the train questions had only moved 0.712 → 0.745
  and pass@8 was flat at ~0.96: the model was re-fitting text it already
  produces. One round helps, two do not.
- **Weight averaging** exp-03 with exp-04 (exp-05) landed between its
  ingredients, not above them.

## Data

All training data is derived from the GSM8K **train** split or from the model's
own samples on train questions. No test item is used, and every training file
passes `../contamination_check.py` with zero flagged documents
(`logs/decon_*.log`).

- `data/sft_v1.jsonl` (67 472) / `data/sft_v2.jsonl` (141 914) —
  `nvidia/OpenMathInstruct-2` rows with `problem_source` in
  `{gsm8k, augmented_gsm8k}` plus the 7 473 native GSM8K train solutions.
  Integer answers only, `\boxed{}` unwrapped, one `ANSWER:` marker per target,
  10 % carrying a 10-shot system prefix built from native train rows.
- `data/rft_v2.jsonl` (10 818) — exp-03 sampled 8× per train question at
  temperature 1.0, keeping only solutions whose final answer matches gold, with
  an adaptive per-problem cap (1 row if solved 8/8, up to 4 if solved rarely).
- `data/rft_mix_v1.jsonl` (21 818) — the above plus an 11 000-row replay slice
  of `sft_v2`.

## Scripts

| file | role |
|---|---|
| `scripts/build_data.py` | builds the SFT jsonl, pre-rendered with the grading template |
| `scripts/train_sft.py` | completion-only SFT; token-budget batching bounds the 262 144-wide logit tensor |
| `scripts/rft_sample.py` | rejection sampling from a checkpoint over GSM8K train |
| `scripts/soup.py` | weight-average several checkpoints |
| `scripts/finalize_ckpt.py` | cast a Trainer checkpoint to bf16 and give it a gradeable generation_config |
| `scripts/tag_eval.py` | tag an inspect-ai log: correct / runaway / malformed / wrong number |

## Reproducing the shipped model

```bash
python scripts/build_data.py --out data/sft_v2.jsonl --n-omi2 140000 \
    --max-per-question 4 --fewshot-frac 0.10 --seed 1
python scripts/train_sft.py --data data/sft_v2.jsonl --out ckpts/exp-03 \
    --lr 1e-5 --epochs 1 --grad-accum 8 --token-budget 6144 --max-seq-len 2560 \
    --attn flash_attention_2 --seed 0
python scripts/rft_sample.py --model ckpts/exp-03/final --out data/rft_v2.jsonl \
    --k 8 --temperature 1.0 --max-keep 4 --adaptive-keep
python scripts/train_sft.py --data data/rft_mix_v1.jsonl \
    --parent ckpts/exp-03/final --out ckpts/exp-04 --lr 7e-6 --epochs 1 \
    --grad-accum 8 --token-budget 6144 --attn flash_attention_2 --seed 0
```

## Traps hit along the way

- `transformers`' `GenerationConfig` validator rejects `do_sample: false` with
  `temperature: 0.0`, so any checkpoint written by this pipeline crashes
  `save_pretrained` when it is later used as a *parent*. `train_sft.py` and
  `soup.py` now neutralise the loaded generation config before saving.
- A 4B checkpoint saved in the training dtype is 17 GB; `evaluate.py` serves at
  `--gpu-memory-utilization 0.3` (~24 GB) and leaves no room for a KV cache.
  Everything is cast to bf16 before saving.
- The logit tensor is `batch_tokens × 262 144`; a 12 288-token micro-batch OOMs
  an 80 GB H100. 6 144 is the working budget.
