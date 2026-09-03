# Post-training `google/gemma-3-4b-pt` for GSM8K

`final_model/` is a full fine-tune of the pinned snapshot
`cc012e0a6d0787b4adcc0fa2c4da74402494554d`, shipped with a greedy
`generation_config.json`.

## Result

| eval | n | accuracy |
|---|---:|---|
| base model, dev-200 | 200 | 0.070 |
| **final_model, dev-200** | 200 | **0.735** |
| **final_model, dev-500** | 500 | **0.734 ± 0.020** |
| **final_model, `python evaluate.py` with stock defaults** | 150 | **0.733** |

## What moved the number

| card | intervention | dev-200 | delta |
|---|---|---:|---:|
| exp-01 | base model, measured | 0.070 | — |
| exp-02 | completion-only SFT, 16k rows, targets ending in `<end_of_turn>` | 0.575 | +50.5 |
| exp-03 | greedy `generation_config.json` instead of the inherited sampling one | 0.635 | +6.0 |
| exp-04 | SFT from base on all 161.7k rows / 80.9k distinct problems | 0.665 | +3.0 |
| exp-05 | continued SFT on 75k Orca-Math problems + 25k replay | 0.675 | +1.0 |
| exp-06 | rejection-sampling FT on the model's own verified-correct chains | 0.675 | 0.0 |
| exp-07 | uniform weight average of 4 checkpoints along the trajectory | **0.735** | +6.0 |

Two things dominated, and neither was more data.

**Termination.** `-pt` is a base checkpoint that has never emitted `<end_of_turn>`.
Under the grader's `gemma3.jinja` template it answers correctly and then keeps
writing invented follow-up problems; since the scorer is
`match(numeric=True, location="end")` it reads the *last* number in the
completion, so a correct answer gets overwritten by the run-on. 111 of 200 base
completions hit the token cap. Completion-only SFT with every target ending in
token 106 took termination from 0.445 to 0.995 and was worth 50 points.

**Decoding.** vLLM defaults to `--generation-config auto`, so the sampling
params in the checkpoint's `generation_config.json` become the server defaults,
and `inspect` passes no temperature. The stock gemma config asks for
`do_sample`/`top_k 64`/`top_p 0.95`, i.e. temperature-1 sampling. Shipping
`do_sample: false, temperature: 0.0` on identical weights was worth 6 points.

**Data scaling saturated fast.** 8.9M → 83.2M → 143M cumulative training tokens
returned +50.5, +3.0, +1.0. Rejection sampling returned exactly 0, with a
training loss flat from step 1 (0.271 → 0.274) — the filter kept the chains the
model was already going to produce.

**Weight averaging was the second-best lever and cost no GPU.** exp-04 → exp-05 →
exp-06 are points on one trajectory. Averaging exp-04, exp-05-step910, exp-05 and
exp-06 gave 0.735. Averaging only exp-04 and exp-05 gave 0.675, and adding
exp-04-step1218 — a mid-run point still at 4e-6 on the cosine — dropped it to
0.660. Averaging works over converged points on the trajectory, not over any two
checkpoints.

Caveat, recorded in exp-07: soup-C and soup-D were built *after* soup-B's score
was known, so picking soup-C is a best-of-four selection on dev-200 and its
0.735 is optimistically biased. The 500-item re-score (0.734 ± 0.020) and the
stock-defaults re-score (0.733) are what justify shipping it.

## Data

All training data is GSM8K-**train**-derived or independent; the GSM8K test set
was used only as the reference input to `contamination_check.py`.

| pool | source | rows | contamination check |
|---|---|---:|---|
| `data/pool_big.jsonl` | OpenMathInstruct-2 `train_1M`+`train_2M`, `problem_source ∈ {gsm8k, augmented_gsm8k}`, rev `469216e3` + `openai/gsm8k` main/train | 161,704 | 0 / 161,704 |
| `data/pool_orca.jsonl` | `microsoft/orca-math-word-problems-200k` | 190,258 | 0 / 190,258 |
| `data/pool_exp06_final.jsonl` | self-samples from exp-05 + gold for unsolved + replay | 26,748 | 0 / 26,748 |

Every target is rebuilt as `reasoning` + `\nANSWER: <number>` + `<end_of_turn>`,
with the row dropped unless `ANSWER:` appears exactly once and the completion's
last number equals the gold answer. 12–15% of rows carry a k-shot system message
built from the gsm8k **train** split, matching the shape of the grader's 10-shot
prompt.

## Layout

- `final_model/` — the shipped model (`provenance.json` names `ckpts/soup-C`)
- `memory/cards/exp-NN.yaml` — one experiment card each, sections 0–4 locked
  before launch; `memory/index.md` is the summary table
- `scripts/` — data build, trainer, sampler, soup, variant/promote, verifier
- `eval/`, `logs/`, `analysis/` — eval outputs, run logs, failure dumps
