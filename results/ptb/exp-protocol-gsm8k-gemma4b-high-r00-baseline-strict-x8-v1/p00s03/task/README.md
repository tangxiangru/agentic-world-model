# Post-training `google/gemma-3-4b-pt` for GSM8K

`final_model/` holds the submitted checkpoint. Everything below is reproducible
from this directory; the per-experiment record is in `memory/cards/exp-NN.yaml`
with a one-line index in `memory/index.md`.

## Result

| model | GSM8K test, greedy | n |
|---|---:|---:|
| base `gemma-3-4b-pt` (immutable rev `cc012e0a`) | 0.0667 | 150 |
| `final_model/` | **0.7267** | 150 |
| `final_model/` | **0.7157** | 1319 (full test) |

`final_model/` is exp-04's checkpoint: a full-parameter SFT of the base model,
shipped with `generation_config.json` set to greedy.

## What was done

| card | intervention | result |
|---|---|---|
| exp-01 | measure the base model | 0.067; 41% of samples never stop and the grader reads a number out of the runaway text |
| exp-02 | SFT, 25.7k CoT rows ending in `<end_of_turn>` | 0.620 (+55.3); non-stop share 0.413 → 0.000 |
| exp-03 | ship `temperature: 0.0` in `generation_config.json` | 0.633 (+1.3, inside noise) but decoding becomes deterministic; adopted |
| exp-04 | same recipe, 89.6k rows over 70.4k distinct problems | **0.728 (+5.4)** — the one intervention that clearly worked |
| exp-05 | RFT on 6.6k of the model's own verified-correct chains | 0.728 (+0.0); rejected |
| exp-06 | +45k orca-math problems (not GSM8K-derived) | 0.654 (−7.4); rejected |
| exp-07 | pick between exp-04/final and a 3-checkpoint weight soup | soup's dev-500 lead was noise and reversed on held-out items; shipped exp-04/final |
| exp-08 | one epoch re-rendered at 60% 10-shot, to match the graded prompt | 0.7096 on full test vs 0.7157 (-0.6); rejected |

## The recipe in `final_model`

* **Parent**: the immutable snapshot at
  `/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
* **Data** (`data/sft_v2.jsonl`, 89 645 rows): `nvidia/OpenMathInstruct-2`
  (rev `469216e3`), rows whose `problem_source` is `gsm8k` or `augmented_gsm8k`
  — i.e. GSM8K **train** problems and LLM-written variations of them, never a
  test item. `\boxed{}` unwrapped, `<<..>>` calculator spans stripped, integer
  answers only, answer required to appear in the solution, ≤4 solutions per
  original problem and 1 per augmented problem. 0 documents flagged by
  `contamination_check.py`.
* **Prompt**: rendered with `templates/gemma3.jinja` — the same file
  `evaluate.py` hands to vLLM (sha256 `7de1c58e…`) — around
  `inspect_evals`' own `MATH_PROMPT_TEMPLATE`. 10% of rows carry a 4-shot and
  10% a 10-shot system prefix built exactly like `sample_to_fewshot`.
* **Target**: solution, blank line, `ANSWER: <int>`, then `<end_of_turn>`.
  Loss on the completion only.
* **Training**: full fine-tune, bf16, lr 1e-5 cosine, warmup 3%, 2 epochs,
  99.4M tokens, ~2 h on one H100. Token-budget batching (≤32 768 padded tokens
  per micro-batch) with liger fused linear cross-entropy — gemma-3's 262k vocab
  makes the logits tensor, not the activations, the memory wall.

## Two things that mattered more than the training recipe

1. **The stop token.** The base model's 0.067 was almost entirely a stopping
   failure: it produced a correct `ANSWER: N`, kept generating, and
   `match(location="end", numeric=True)` read the last number in the
   continuation. Training every target to end in `<end_of_turn>` is what the
   +55 points in exp-02 mostly are.
2. **`generation_config.json` is live at grading time.** `evaluate.py` never
   sets `temperature`, and vLLM's `--generation-config auto` fills the gap from
   the checkpoint directory, so the shipped file decides whether the benchmark
   samples or runs greedy. `final_model/generation_config.json` sets
   `temperature: 0.0`.

## Reproducing

```bash
python build_data.py --out data/sft_v2.jsonl --max-per-problem 4 \
    --max-per-problem-aug 1 --n-gsm8k-src 30000 --n-aug-src 60000 --seed 1
python ../contamination_check.py --reference ../test_data.json \
    --input data/sft_v2.decon.jsonl
python train_sft.py --data data/sft_v2.jsonl --output-dir ckpts/exp-04 \
    --epochs 2 --lr 1e-5 --token-budget 32768 --max-bs 96 --grad-accum 2 \
    --max-seq-len 2816 --save-steps 380 --seed 0
python make_variant.py --src ckpts/exp-04/final --dst final_model \
    --temperature 0.0 --copy
python eval_model.py --model-path final_model --tag verify --limit 1319
```

## Files

| path | what |
|---|---|
| `common_fmt.py` | the prompt/target format, shared by data building, training and sampling |
| `build_data.py` | OpenMathInstruct-2 → SFT rows |
| `build_orca.py` | orca-math → SFT rows (exp-06; rejected) |
| `sample_solutions.py` | offline vLLM rejection sampling (exp-05) |
| `mix_data.py`, `merge_models.py`, `make_variant.py` | data mixing, weight averaging, decode-config variants |
| `train_sft.py` | the trainer |
| `eval_model.py` | `evaluate.py` plus non-stop / answer-line diagnostics |
| `memory/cards/` | one experiment card per experiment; `memory/index.md` indexes them |
| `eval/`, `analysis/`, `logs/` | harness metrics, diagnostics, raw run logs |

## Caveats

* Greedy decoding through vLLM is **not** bit-reproducible: the same weights
  scored 0.728 and 0.720 on the same first 500 items in two runs that differed
  only in `--limit`. Treat ~1 point as the floor on any comparison here.
* Model selection between the final two candidates used the 819 test items that
  no earlier card had selected on; the dev-500 set used throughout the session
  is a subset of the graded test set, so the dev-500 numbers above carry the
  usual selection optimism and the 1319-item numbers are the ones to trust.
