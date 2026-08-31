# r-c7ff2a60 — extracted experiment cards

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100
The digest records no event timestamps, so `elapsed_h` is null on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 97 | null | sft | base_model | gsm8k train (x3) + MetaMathQA (GSM all + MATH cap 30k) | 2e-5 / 2 ep | failed | none | inconclusive | iterate |
| exp-02 | 105 | null | sft | base_model | gsm8k train (x3) + MetaMathQA (GSM all + MATH cap 30k) | 2e-5 / 2 ep planned, ~0.7 done | killed | none | inconclusive | adopt |
| exp-03 | 218 | null | other (package checkpoint-1000 to final_model) | exp-02 | — | — | completed | 0.167 @ n=30 | inconclusive | reject |
| exp-04 | 241 | null | sft | exp-02 | gsm8k train 7473 (x3) + MetaMathQA 269,998 = 292,417 | 2e-5 / 3 ep | completed | 0.220 @ n=50 | inconclusive | adopt |
| exp-05 | 284 | null | decode-config (eos_token_id += 128012, use_cache) | exp-04 | — | — | completed | 0.140 @ n=50 | contradicted | reject |
| exp-06 | 310 | null | sft | base_model | gsm8k train 7473 (x5) + MetaMathQA GSM 80,000 = 117,365 | 5e-5 / 900 steps (0.25 ep) | completed | 0.280 @ n=50 (--max-tokens 512); 0.160 @ n=50 default | inconclusive | reject |
| exp-07 | 330 | null | sft | base_model | gsm8k train 7473 (x5) + MetaMathQA GSM 160,000 = 197,365 | 3e-5 / 3000 steps (0.49 ep) | completed | 0.413 @ n=150 | supported | adopt |

Submitted: **exp-07** — the checkpoint left in `/home/ben/task/final_model` at the end of the run.
`exp-02` and `exp-04` are marked `adopt` only in the chaining sense (their outputs are the
parent checkpoints of later cards); neither was the submission.

No smoke tests or dry runs appear in this run: exp-01 was launched as a real training run and
crashed on an SFTConfig keyword before the first step.
