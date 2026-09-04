# Placeholder

exp-01 is an evaluation-only card: it trains on nothing. `setup.data` is a required
field, so it points here. The only data the run reads is the benchmark's own dev
slice, which inspect_evals/gsm8k loads directly from `openai/gsm8k` (main/test,
first 150 items), together with its 10-shot block drawn from the gsm8k TRAIN split
(fewshot_seed=42). No file in this repository is used as training input by exp-01.
