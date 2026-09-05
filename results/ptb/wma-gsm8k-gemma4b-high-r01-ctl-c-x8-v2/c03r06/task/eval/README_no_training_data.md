Placeholder for eval-only cards (exp-01). No training data is read by an
eval-only run; `setup.data` is a required field, so this file stands in for it.
The evaluated items are loaded inside evaluate.py from the harness dataset
(openai/gsm8k, split=test, --limit 150). Few-shot exemplars come from
openai/gsm8k split=train with seed 42, chosen by inspect_evals, not by me.
