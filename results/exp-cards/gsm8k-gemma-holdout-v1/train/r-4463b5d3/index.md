# r-4463b5d3 - reconstructed experiment cards (train side)

Base model: HuggingFaceTB/SmolLM3-3B-Base | benchmark: gsm8k | budget: 10 h, one H100.
8 cards, one per launch that produced or installed a candidate. The stream carries
timestamps, so every `elapsed_h` is the `t=` of the launch event. The digest runs
from [3] (t=+0.00h) to [1562] (t=+8.80h) and is complete: the agent signs off with
the GPU idle and 1:11 left on the timer.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 532 | 0.33 | sft | base_model | sft_v1.jsonl, 117,650 ex (OpenMathInstruct-2 gsm8k/augmented + gsm8k train + MetaMathQA GSM) | 1.4e-5 / 2 | killed | none (7 of 2460 steps) | inconclusive | abandon_line |
| exp-02 | 632 | 0.39 | sft | base_model | sft_v1.jsonl, 117,650 ex (same mix) | 1.4e-5 / 2 | completed | accuracy 0.695, n=200, test_sft_v1_ep2.json (base 0.207 at n=150) | inconclusive | adopt |
| exp-03 | 896 | 3.21 | other (package sft_v1 -> final_model) | exp-02 | - | - | completed | none (artifact never scored under that name) | inconclusive | reject |
| exp-04 | 1040 | 3.32 | grpo | exp-02 | rl_pool_full.jsonl, 20,000 prompts | 1.5e-6 / 1 | failed | none (CUDA OOM at step 1 of 416) | inconclusive | abandon_line |
| exp-05 | 1079 | 3.35 | grpo | exp-02 | rl_pool_full.jsonl, 20,000 prompts (~12,000 consumed) | 1.5e-6 / 1 | killed | accuracy 0.850, n=200, test200_ckpt_grpo_v1_checkpoint-125.json | supported | adopt |
| exp-06 | 1245 | 6.31 | other (package grpo_v1/checkpoint-125 -> final_model) | exp-05 | - | - | completed | none (artifact never scored under that name) | inconclusive | reject |
| exp-07 | 1247 | 6.31 | grpo | exp-05 | rl_pool_full.jsonl, 12,000 prompts (--skip 12000, disjoint slice) | 1.5e-6 / 1 | killed | accuracy 0.874, n=800, test800_ckpt_grpo_v2_checkpoint-50.json (0.890 at n=300) | inconclusive | adopt |
| exp-08 | 1325 | 8.25 | other (package grpo_v2/checkpoint-50 + bf16 re-save -> final_model) | exp-07 | - | - | completed | accuracy 0.8696, n=1319 (full test), final_full3.json | inconclusive | adopt |

Submission: exp-08 - `final_model` holding `ckpt/grpo_v2/checkpoint-50` re-saved in
bf16 and finalized with the eval chat template and `<|im_end|>` as eos. Nothing in
the stream overwrites `final_model` after [1347]. exp-02, exp-05 and exp-07 are
marked adopt as the parents of the packaging cards downstream of them; exp-03 and
exp-06 held final_model for a while and were overwritten.

Shape of the run: one SFT pass on 117,650 GSM8K-derived examples formatted to the
eval's own chat template and "ANSWER: <n>" ending (0.207 -> 0.695 at n=200), then
two GRPO rounds with an exact-match reward from that checkpoint (0.695 -> 0.850 at
n=200, 0.874 at n=800). RL supplied nearly all of the gain after SFT. Checkpoint
selection among the top four candidates was made on 800-sample evals spanning
0.859-0.874, which the agent itself called noise [1324].

Smoke tests (not carded): four train_sft.py throughput probes at [402], [451],
[470], [483] on exp-01, and two 96-prompt GRPO probes at [974] and [998] on
exp-04.

Not carded, though they were launches: the baseline eval of the untrained model
[128], the rejection-sampling generation `gen_rft.py` [817] whose output
`data/rft_v1.jsonl` no later launch ever trains on, and the eval wrappers
`after_sft.sh` [769], `eval_round.sh` [1186], `final_round.sh` [1272] and
`big_round.sh` [1300], whose numbers are recorded as measurements on the cards of
the checkpoints they scored.

Run-level caveats: the workspace snapshot holds only scripts, `RESULTS.md` and
`system_monitor.log` - no `logs/` and no `data/`, so every eval JSON a card cites
is named from the stream and cannot be re-read; and several numbers (the n=800
table, the epoch-1 SFT score, the full-test stderr) are read from the agent's own
`RESULTS.md` rather than from a printed eval output.
