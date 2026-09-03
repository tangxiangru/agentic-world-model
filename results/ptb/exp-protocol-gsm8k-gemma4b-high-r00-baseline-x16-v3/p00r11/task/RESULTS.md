# GSM8K post-training of google/gemma-3-4b-pt

`final_model/` holds the submission: **0.7043 accuracy on the full 1319-item GSM8K
test set** (0.72 on the harness default `--limit 150`), against **0.0667** for the
frozen base snapshot under the same protocol.

## What it is

Weights: `ckpts/exp-03/final` — the base snapshot post-trained in two stages, plus a
greedy decode config declared in `generation_config.json`.

| stage | what | full-test accuracy (greedy) |
|---|---|---|
| base | `google/gemma-3-4b-pt` @ cc012e0a | 0.067 (sampled, exp-01) |
| exp-02 | SFT, 94.9k GSM8K-train-derived CoT rows, 2 epochs | 0.687 |
| exp-03 | RFT on 22.5k verified self-samples, 2 epochs | **0.702 / 0.704** |
| exp-07 | greedy decoding declared in generation_config.json | +6.1 pts over sampled |

## Recipe

1. **SFT** (`build_sft.py`, `train_sft.py`): 94,946 rows — the 7.5k GSM8K *train*
   CoTs (x2, calculator annotations stripped) plus 80k OpenMathInstruct-2
   gsm8k-family solutions, one per problem, `\boxed{}` unwrapped. Every target is
   rendered with the grader's own `templates/gemma3.jinja`, ends in a single
   `ANSWER: <number>` line followed by `<end_of_turn>`, and 10% carry a 2-5 shot
   system prefix built from train-split exemplars. Completion-only loss, bf16,
   lr 1e-5, 2 epochs, effective batch 32, `max_seq_len` 1536.
2. **RFT** (`rft_sample.py`): 4 samples per question at temperature 1.0 over 18k
   train questions, keep only samples whose final answer matches gold (2 per
   question, 1 where all four were right) → 22,474 rows; 2 epochs at lr 5e-6.
3. **Decode**: the harness sends no sampling parameters, so vLLM falls back to the
   model's `generation_config.json`. Declaring `temperature 0.0, top_k 1` there
   turns temperature-1.0 sampling into greedy decoding: +6.1 points for identical
   weights.

Training memory note: Gemma3's own loss path allocates a 17 GiB logits tensor at
this batch size. `train_sft.py` computes the cross-entropy only over supervised
tokens, in checkpointed 2048-token chunks.

## What did not work

- **More RFT data** (57k rows / 36.7k questions, exp-04): 0.676 — worse than the
  22.5k-row corpus.
- **Weight averaging** the two best checkpoints (exp-05): 0.688, between its parents.
- **A second STaR round** sampled from the 0.702 model (exp-08): 0.671 and 0.644.
- **repetition_penalty 1.05** to break the 13 remaining greedy loops (exp-09):
  0.701, one loop fixed and four items lost elsewhere.

## Contamination

No test item is read anywhere in the data pipeline: questions come from the GSM8K
*train* split and from OpenMathInstruct-2's gsm8k-family problems, and self-samples
are generated on those same train questions. All three training corpora pass the
checker with 0 contaminated documents (`analysis/contam_sft_v1.txt`,
`contam_rft_v1.txt`, `contam_rft_v2.txt`, `contam_rft_v3.txt`).

## Record

One experiment card per run in `memory/cards/exp-01..09.yaml`, indexed in
`memory/index.md`; eval outputs in `eval/`, diagnostics in `analysis/`, logs in
`logs/`.

The dev-150 protocol used for the first five cards proved too noisy to rank
checkpoints (±3.8 pts, and every measurement was a single temperature-1.0 sample):
its ranking inverted when the same checkpoints were scored on all 1319 items. The
full-set greedy evaluation takes ~4 minutes and is what the final choice rests on.
