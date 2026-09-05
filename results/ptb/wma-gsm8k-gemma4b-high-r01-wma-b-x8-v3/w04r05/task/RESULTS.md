# gemma-3-4b-pt → GSM8K

`final_model/` holds the submitted checkpoint. Full experiment records are in
`memory/cards/exp-0N.yaml` (one card per experiment, sections 0–4 written and
locked before each launch); `memory/index.md` is the one-line-per-card index.

## Result

| stage | what changed | official GSM8K |
|---|---|---|
| base `gemma-3-4b-pt` | — | **0.060** (n=150) |
| exp-02 SFT | format + eos on 146k GSM8K-train-derived CoT rows | 0.533 (n=150) |
| exp-03 decode | `temperature: 0.0` in `generation_config.json` | **0.720** (n=150) |
| exp-04 RFT | +48k verified self-generated solutions | 0.712 (n=500) |

`final_model` = exp-04's checkpoint. Measured three times on the official
harness at n=500 with byte-identical weights: **0.712, 0.688, and 0.714** (the
last at the harness's default `--max-connections 2` — see the caveat below).
The base model on the same n=150 protocol is 0.060, so the batch moved the model
roughly **+65 points**.

## What actually produced the gain

1. **Termination, not arithmetic (exp-01 → exp-02, +47).** The base model never
   emits `<end_of_turn>`; 45 % of its completions ran to the 4000-token cap and
   the grader — `match(numeric=True, location="end")`, which reads the *last
   number* in the completion — scored text the model invented after its answer.
   Every training target was made to end `\n\nANSWER: <int><end_of_turn>`.
   Token 106 is already in the base `generation_config`'s `eos_token_id`, so
   vLLM stops on it as soon as the model learns to emit it.

2. **The decode config (exp-03, +18.7 on identical weights).** `evaluate.py`
   passes no sampling parameters, so vLLM decodes with whatever the served
   `generation_config.json` says — and vLLM reads only
   `repetition_penalty / temperature / top_k / top_p / min_p / max_new_tokens`
   (`vllm/config/model.py:get_diff_sampling_param`). **`do_sample` is ignored.**
   A checkpoint carrying `do_sample: false` and no `temperature` is served at
   the OpenAI default temperature 1.0 with no truncation. Writing
   `"temperature": 0.0` took the same weights from 0.533 to 0.720.

3. **One rejection-sampling round (exp-04, +3.8 at n=500).** The SFT model had
   pass@1 0.649 but pass@4 0.864 over 40k training problems: the right chain was
   reachable, just not first. Training on 48k of its own verified-correct chains,
   weighted toward problems it solved only 1–2 times in 4, closed part of that gap
   (paired McNemar z = 2.11).

## What did not work, measured

- **A second epoch of SFT.** The held-out probe is flat across steps 1000 / 2000 /
  2009 (0.694 / 0.688 / 0.690). That corpus saturates in half an epoch.
- **Doubling the RFT corpus (exp-06).** 48k → 89k rows and 40k → 70k distinct
  problems, including 284 hard-tail problems only the stronger sampler could
  solve: official n=500 0.712 → 0.718 (z = 0.40), held-out probe 0.718 → 0.702.
  Nothing. RFT's return is in the on-policy correction itself, not in coverage.
- **Weight averaging (exp-05, exp-07).** exp-02+exp-04 soup probes 0.710, between
  its parents. exp-04+exp-06 soup probes 0.692, *below* both — 34 items broken to
  fix 21.

## The measurement caveat (exp-08)

Two runs of **byte-identical weights** under the identical command returned
0.712 and 0.688 at n=500: 18 of 500 graded items flip. vLLM's continuous
batching makes reduction order depend on batch composition, so a token near a
decision boundary flips and takes a whole chain with it. This run-to-run swing
(~2.4 points) is *larger* than the paired sampling SE (1.5), and larger than
three of the deltas this batch reasoned about. It does not change the ranking —
exp-05's +3.8 has an independent held-out probe agreeing at +2.8 and a paired
z of 2.11 — but every single-run number here should be read with it in mind.

## Data provenance and decontamination

All training data derives from the GSM8K **train** split or from the model's own
generations; the test split was never read except as reference input to the
contamination checker. Sources: `nvidia/OpenMathInstruct-2` rev `469216e3`
(`train_1M`, gsm8k- and math-augmented rows with integer answers),
`openai/gsm8k` rev `740312ad` (`main/train` gold solutions), and self-generated
verified solutions. 500 train problems were held out as `data/dev_train500.jsonl`
and excluded from every training file.

`python ../contamination_check.py --reference ../test_data.json` was run on every
training corpus before it was used:

| corpus | documents | contaminated |
|---|---:|---:|
| `data/sft_v1.jsonl` (exp-02) | 145,919 | 0 |
| `data/rft_v1.jsonl` (exp-04) | 68,186 | 0 |
| `data/rft_v2.jsonl` (exp-06) | 112,571 | 0 |

## Reproducing

```bash
python scripts/build_sft.py --out data/sft_v1.jsonl
python scripts/train_sft.py --data data/sft_v1.jsonl --out ckpts/exp-02 \
    --bs 24 --grad-accum 3 --lr 1e-5 --epochs 1 --max-seq-len 2048
python scripts/vllm_gen.py --model ckpts/exp-02/final --input data/rft_pool.jsonl \
    --out data/rft_samples.jsonl --n 4 --temperature 1.0 --top-p 0.95 --limit 40000
python scripts/build_rft.py --samples data/rft_samples.jsonl --out data/rft_v1.jsonl \
    --anchor data/sft_v1.jsonl --n-anchor 20000
python scripts/train_sft.py --data data/rft_v1.jsonl --parent ckpts/exp-02/final \
    --out ckpts/exp-04 --bs 24 --grad-accum 3 --lr 7e-6 --epochs 1 --max-seq-len 2048
```

`scripts/common.py` renders prompts with a byte-for-byte copy of the grader's
`MATH_PROMPT_TEMPLATE` and reproduces `templates/gemma3.jinja` exactly (checked
by diffing against a jinja render for the no-system, 10-shot-system, and
full-conversation cases). Training needs `liger-kernel` — gemma-3's 262k vocab
makes a 16k-token batch allocate a 17 GB fp32 logit tensor inside
`cross_entropy` and OOM an 80 GB H100; liger's fused linear CE never
materialises the logits. `final_model` itself needs nothing beyond the packages
the task shipped with.
