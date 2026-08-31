# r-dd7a76f5 - reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base | benchmark: gsm8k | budget: 10 h, one H100.
9 launches carded. The stream ends at event [503], t=+4.40h, with the agent
signing off ("Done. The best selected model is packaged in final_model") while
the timer still showed 5:35 remaining, so the run stops well short of its budget
by the agent's own choice rather than by truncation.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 59 | 0.14 | sft | base_model | gsm8k train x3 prompt variants | 2e-5 / 2 | completed | accuracy 0.175, n=80 (ckpt-1200) | inconclusive | adopt |
| exp-02 | 146 | 1.78 | decode-config (EOS -> `<\|im_end\|>`) | exp-01 | - | - | completed | accuracy 0.4875, n=80 (base 0.175 same limit) | supported | adopt |
| exp-03 | 204 | 1.91 | sft | exp-02 | gsm8k train, eval_fixed prompt only, 7,217 rows | 5e-6 / 1 | completed | accuracy 0.580, n=150 (base 0.5133 same limit) | supported | adopt |
| exp-04 | 253 | 2.45 | sft | exp-03 | gsm8k train, eval_fixed prompt only | 2e-6 / 0.5 | completed | accuracy 0.5067, n=150 | contradicted | reject |
| exp-05 | 275 | 2.71 | other (package refine_exact -> final_model) | exp-03 | - | - | completed | accuracy 0.480, n=150 | inconclusive | adopt |
| exp-06 | 340 | 2.84 | sft | exp-05 | gsm8k train + 10k templated synthetic | 3e-6 / 1 | completed | accuracy 0.5315, n=1319 (ckpt-400; incumbent 0.4875 same limit) | supported | adopt |
| exp-07 | 388 | 3.64 | other (package synth ckpt-400 -> final_model) | exp-06 | - | - | completed | accuracy 0.5155, n=1319 | inconclusive | adopt |
| exp-08 | 432 | 3.74 | sft | exp-07 | gsm8k train, eval_fixed prompt only | 2e-6 / 0.5 | completed | accuracy 0.500, n=150 | inconclusive | reject |
| exp-09 | 474 | 3.99 | sft | exp-07 | gsm8k train + 5k synthetic + SVAMP/Calc-mawps | 1e-6 / 0.5 | completed | accuracy 0.4867, n=150 | inconclusive | reject |

Submission: exp-07 - final_model holding
runs/refine_synth10k_from_final_lr3e-6_ep1/checkpoint-400, verified byte-identical
to that checkpoint by sha256 at [501]. exp-01, exp-02, exp-03 and exp-05 are
marked adopt as the parents of the chain that leads to it, and exp-05 because its
output was the final_model for the 0.93 h between [275] and [388]; only exp-07's
output is the final_model the stream leaves behind.

The turning point is exp-02, which trained nothing: the SFT checkpoints had
inherited the base model's EOS id, so every generation ran to the 1024-token cap
and scored 0.10-0.175. Rewriting `eos_token_id` to `<|im_end|>` in the saved
config, generation config and tokenizer files took the same weights from 0.175 to
0.4875 on the same 80-sample slice, with output length falling from ~1024 tokens
per sample to ~171.

Smoke tests: four, all folded onto the next real launch - [49] (exp-01), [190]
(exp-03), [324] and [336] (exp-06), [468] (exp-09). [324] is the only one that
crashed (IndexError in the new synthetic-data generator).

Two cautions for any consumer of these numbers. The evaluator samples, so
identical weights scored 0.475 and 0.5625 on the 80-sample slice [246] and 0.58
then 0.48 on the 150-sample slice [252], [288]; only the four full-set runs
(n=1319) are tight enough to separate candidates. And exp-08 and exp-09 were both
launched with no stated problem or hypothesis, so their verdicts stay
inconclusive even though both were measured against the incumbent under the same
--limit.
