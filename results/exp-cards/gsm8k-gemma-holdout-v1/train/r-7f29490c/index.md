# r-7f29490c — reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 97 | 0.10 | sft | base_model | meta-math/MetaMathQA GSM subset (120000) + openai/gsm8k train (7473) = 127473 | 1e-5 / 1.0 | completed | accuracy 0.30, n=30 (eval_30.json) | inconclusive | adopt |
| exp-02 | 182 | 2.43 | decode-config | exp-01 | — | — / — | completed | accuracy 0.56, n=50 (eval_50.json) | inconclusive | adopt |

Notes

- exp-01 carries two smoke runs that are not cards: [82] crashed (`assistant_only_loss=True` with a chat template lacking `{% generation %}`), [91] passed at 2 steps.
- exp-02 is an in-place EOS/decode patch of exp-01's checkpoint (`<|im_end|>` written as eos into tokenizer, generation_config and config.json); no weights changed.
- No baseline eval of the base model exists, and the two measurements are at different `--limit` (30 vs 50), so neither card has a same-protocol comparator.
- `train_sft_r2.py` [121] and `train_sft_v2.py` [196] were written but never launched inside the stream; the stream ends at [197], t=+2.49 h of the 10 h budget.
