exp-01 is a measurement-only card: the untouched base checkpoint
google/gemma-3-4b-pt @ cc012e0a6d0787b4adcc0fa2c4da74402494554d is scored on the
harness protocol. No weights are updated and no training data is read.

This placeholder exists because the card schema requires at least one
setup.data entry. n_examples = 0, mixture_weight = 0.0.

The measurement input is the harness's own dataset (inspect_evals/gsm8k test
split, --limit 150). That is benchmark input, never training data.
