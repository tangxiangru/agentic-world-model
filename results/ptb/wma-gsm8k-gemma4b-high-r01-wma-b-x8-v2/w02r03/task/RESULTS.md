# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` = **exp-04**, a two-stage full fine-tune of the immutable snapshot
`cc012e0a6d0787b4adcc0fa2c4da74402494554d`, shipped with a greedy `generation_config.json`.

## Scoreboard (official `evaluate.py`, inspect_evals/gsm8k, 10-shot, `match(numeric=True)`)

| card | what | dev-150 | dev-500 |
|---|---|---:|---:|
| exp-01 | base `gemma-3-4b-pt` | 0.080 | – |
| exp-02 | + SFT, 135k GSM8K-train-derived rows (sampled decode) | 0.593 | – |
| exp-03 | + greedy decode (`temperature: 0.0`, no weight change) | 0.653 | – |
| **exp-04** | **+ RFT round 1 on 29.5k self-generated verified solutions** | **0.707** | **0.710** |
| exp-05 | + RFT round 2 | 0.700 | – |
| exp-06 | soup(exp-04, exp-05), alpha 0.5 | 0.700 | 0.696 |

Base → final: **0.080 → 0.710** (n=500, ±0.020).

## What each stage did

1. **exp-02 — SFT.** 135k rows: OpenMathInstruct-2 filtered to `problem_source ∈ {gsm8k, augmented_gsm8k}`
   (max 2 solutions/problem, `\boxed{}` unwrapped, one answer marker) plus the GSM8K **train** split twice
   in its native `<<a*b=c>>` style. Every row rendered through the grader's own `templates/gemma3.jinja`;
   targets end in one `ANSWER: N` line then `<end_of_turn>`. 12% of rows sit behind a k-shot prefix of
   random GSM8K *train* items, because the grader always sends a 10-shot system message.
   Full fine-tune of the 3.88B text tower (vision tower frozen), bf16, lr 1e-5, 1 epoch, 1h51m.
   This is what fixed the output contract: termination and single-answer formatting went 19% → 100%.
2. **exp-03 — decode.** vLLM takes its default sampling params from the model's own
   `generation_config.json`, and inspect never sets a temperature, so the base file's `top_k 64 / top_p 0.95`
   meant the model was being graded at T=1.0. Setting `temperature: 0.0` is a free +6 points.
3. **exp-04 — RFT.** k=4 samples at T=1.0 from exp-02 over 24k train-derived questions
   (sample acc 0.620, pass@4 0.858); keep only solutions whose first `ANSWER:` line matches gold;
   2 per hard problem, 1 per fully-solved problem, deduplicated by (numbers, operators) signature;
   25k stage-1 rows mixed back in. Continued training at lr 7e-6, 42 min. +5.3 points.
4. **exp-05 / exp-06 — both null.** A second RFT round and a weight soup each landed at 0.700.
   The pass@4 headroom that round 1 converted is not convertible by more of the same self-distillation.

## Contamination

The GSM8K test split was never used for training. Every training file passed
`../contamination_check.py --reference ../test_data.json`: `sft_v2` (135000 docs),
`rft_v1` (54477), `rft2_v1` (53706) — 0 contaminated, exit 0 each time.
500 GSM8K *train* problems were also held out to `data/devtrain500.jsonl` and excluded from every file.

## Two things worth knowing before the next run

- **The grader's decode comes from your model directory.** `vllm/config/model.py::get_diff_sampling_param`
  promotes `temperature`/`top_p`/`top_k` out of `generation_config.json` into the server defaults, and
  inspect_ai sends none of them. Shipping a greedy `generation_config.json` was worth more than the
  second RFT round.
- **The n=150 protocol has a nondeterminism floor of ~6 items.** Re-running `final_model/` — weights and
  `generation_config.json` md5-identical to `ckpts/exp-04` — scored 0.693 instead of 0.707, differing on
  6 of 150 items, from vLLM's batch-dependent floating-point reductions. Any dev-150 gap under ~4 points
  is unreadable; exp-04/05/06 differed by 1 item and needed n=500 (`exp-07`) to be ranked at all.

Full record: `memory/index.md` and `memory/cards/exp-0*.yaml` (situation → problem → hypothesis → setup →
evaluation written and locked before each launch; result and conclusion after).
