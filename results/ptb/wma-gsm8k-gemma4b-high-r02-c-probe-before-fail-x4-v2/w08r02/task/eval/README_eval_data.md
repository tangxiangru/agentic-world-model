# Evaluation data (not training data)

Every card in this session is scored by the unmodified `evaluate.py`, which runs
`inspect_evals/gsm8k` and loads the official `openai/gsm8k` **test** split itself.
`--limit N` takes the first N items of that split, in dataset order.

There is no local copy of the eval items in this directory and none is ever written:
`/home/ben/test_data.json` exists solely as the reference input to
`/home/ben/contamination_check.py`.

This file exists so experiment cards can point `setup.data[].path` at something real
for evaluation-only cards. `mixture_weight: 0.0` marks it as non-training.
