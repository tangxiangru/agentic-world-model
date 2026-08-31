# r-98c04d79 - reconstructed experiment cards

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, one H100.
11 launches carded. The stream ends at event [332], t=+7.91h, with about 2 h of
budget unspent and the incumbent already packaged and re-verified.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 25 | 0.02 | other (package base -> final_model) | base_model | - | - | completed | accuracy 0.400, n=20 | inconclusive | reject |
| exp-02 | 75 | 0.13 | sft | base_model | gsm8k train 7,473 (chat template, #### targets) | 2e-4 / 2 | completed | accuracy 0.160, n=50 | inconclusive | reject |
| exp-03 | 100 | 0.56 | sft | base_model | gsm8k train 7,473 (no system turn) | 5e-5 / 1 | completed | accuracy 0.140, n=50 | contradicted | reject |
| exp-04 | 121 | 0.80 | sft | base_model | gsm8k train 7,473 (eval wrapper in chat template) | 5e-5 / 1 | completed | accuracy 0.020, n=50 | contradicted | reject |
| exp-05 | 134 | 1.14 | sft | base_model | gsm8k train 7,473 (plain text, ANSWER: targets) | 5e-5 / 2 | completed | accuracy 0.533, n=150 | supported | adopt |
| exp-06 | 148 | 1.54 | sft | base_model | gsm8k 7,473 + MetaMathQA GSM 160,000 = 167,473 | 3e-5 / 2 (stopped at ~1.24) | killed | - (its checkpoint measured on exp-07) | inconclusive | adopt |
| exp-07 | 171 | 6.60 | merge (checkpoint-6500 adapter -> final_model_v5_checkpoint) | exp-06 | - | - | completed | accuracy 0.573, n=150 | supported | adopt |
| exp-08 | 202 | 6.73 | other (package v5 checkpoint -> final_model) | exp-07 | - | - | completed | accuracy 0.573, n=150 | supported | adopt |
| exp-09 | 218 | 6.74 | sft | base_model | gsm8k 7,473 + MetaMathQA GSM 160,000 | 5e-5 / 3 (killed at step 92) | killed | - | inconclusive | abandon_line |
| exp-10 | 249 | 6.84 | sft | base_model | gsm8k train 7,473 (plain text) | 8e-5 / 5 | completed | accuracy 0.553, n=150 | contradicted | reject |
| exp-11 | 315 | 7.83 | sft | base_model | gsm8k 7,473 + MetaMathQA GSM 160,000 | 5e-5 / 2 (killed within a minute) | killed | - | inconclusive | abandon_line |

Submission: exp-08 - final_model holding the merged LoRA adapter from
checkpoints_v5/checkpoint-6500, re-verified at 0.573 on 150 official samples at
[323] and recorded in results_final.json. exp-05, exp-06 and exp-07 are marked
adopt as the chain that produced it (exp-05's output was final_model until [202];
exp-06's checkpoint is exp-07's parent; exp-07's merged model is what exp-08
copies).

The run's one real finding is a format finding: with a chat template the base
model emits stray non-ASCII tokens and re-narrates the prompt (0.16, 0.14, 0.02
on 50 samples); trained as a plain completion of the harness's own prompt, the
same recipe scores 0.52 (exp-05). Everything after that is data volume - the
MetaMathQA-augmented mixture at 1.24 epochs is +4.0 pts over GSM8K alone - and
three launches that ran out of clock.

Smoke tests: two, both on exp-02 - [55] crashed on SFTConfig(max_seq_length=...)
under TRL 0.27, [62] passed at 20 steps on 100 samples after the rename to
max_length.

Scripts written but never run: train_v5_resume.py [215], written to resume
checkpoint-6500 with its optimizer state and dropped at [217] in favour of
exp-09; prepare_data.py built a 247,473-example GSM8K+MetaMathQA file at [47]
that no launch ever reads.
