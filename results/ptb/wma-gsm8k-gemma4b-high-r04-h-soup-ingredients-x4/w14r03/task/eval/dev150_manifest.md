# exp-01 data manifest

exp-01 is a measurement-only card: **nothing is trained**, so there is no training file.
The card schema requires a non-empty setup.data, so this manifest stands in for it.

What is read: the official `inspect_evals/gsm8k` test split, loaded by `evaluate.py`
through `hf_dataset('openai/gsm8k', data_dir='main', split='test')`, truncated to the
first 150 items by `--limit 150`. n_examples=150 therefore counts *evaluated benchmark
items*, not training rows; mixture_weight is 0.0 because none of them are trained on.

No copy of the benchmark test items is stored in this repository.
