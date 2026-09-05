exp-01 is an evaluation-only card. It reads no training data.
The harness (evaluate.py -> inspect_evals/gsm8k) loads the official gsm8k test
split itself via hf_dataset and takes the first 150 items. This file exists only
so that setup.data can point at a real path, as the card schema requires.
