# Dev evaluation set for this batch

Every card in this batch is scored with:

    python evaluate.py --model-path <ckpt> --limit 150 --max-connections 16 \
        --json-output-file eval/<card>_dev150.json

which is `inspect_evals/gsm8k` with its defaults: the **first 150 items of the
official `openai/gsm8k` test split**, a 10-shot system message built from the
*train* split (`fewshot=10`, `fewshot_seed=42`, shuffled), the
`MATH_PROMPT_TEMPLATE` instruction, and `match(numeric=True)` scoring the end of
the completion.

These 150 items are **evaluation input only**. They are never read into training
data, never paraphrased, and never used to seed generation. The local copy at
`/home/ben/test_data.json` is used solely as the reference input to
`/home/ben/contamination_check.py`.
