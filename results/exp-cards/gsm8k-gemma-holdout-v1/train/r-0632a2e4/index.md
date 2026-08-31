# r-0632a2e4 - reconstructed experiment cards

Base model: Qwen/Qwen3-1.7B-Base | benchmark: gsm8k | budget: 10 h, one H100.
8 launches carded. The stream runs the full budget, ending at t=+9.64h with the
submission in place; the tail from [665] on is a long series of stale
background-task notifications and carries no further work.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 96 | 0.11 | sft | base_model | gsm8k train 7,473 + MetaMathQA GSM 160k, 80k of 167,473 | 1e-5 / 2 | completed | - (no accuracy printed for its eval) | inconclusive | adopt |
| exp-02 | 178 | 0.93 | decode-config (EOS fix on sft_v1) | exp-01 | - | - | completed | accuracy 0.140, n=150 (agent recount: 21/150 scored, 90/150 correct first ANSWER) | inconclusive | reject |
| exp-03 | 277 | 1.23 | sft | base_model | gsm8k train + MetaMathQA GSM, 60k, fewshot system messages | 1e-5 / 1 | completed | accuracy 0.673, n=150 (vs 0.140 same limit) | supported | reject |
| exp-04 | 371 | 2.57 | sft | base_model | same 60k file, both epochs | 1e-5 / 2 | completed | accuracy 0.693, n=150; 0.672, n=500 | inconclusive | adopt |
| exp-05 | 481 | 5.52 | sft | base_model | gsm8k 7,473 + MetaMath 30k + 25,136 self-rejection x2, 70k of 87,745 | 1e-5 / 1 | completed | accuracy 0.656, n=500 (exp-04 0.672 same limit) | contradicted | reject |
| exp-06 | 548 | 7.44 | other (package sft_v3 -> final_model) | exp-04 | - | - | completed | accuracy 0.665, n=200 | inconclusive | adopt |
| exp-07 | 583 | 7.74 | sft | base_model | gsm8k train x2 + 25,136 self-rejection, 40,082, no MetaMath | 1e-5 / 1 | failed | - (died before any eval; agent reports OOM) | inconclusive | abandon_line |
| exp-08 | 628 | 8.86 | sft | exp-04 | same 40,082 file, 20k subset | 3e-6 / 1 | completed | accuracy 0.602, n=500 (exp-04 0.672 same limit) | inconclusive | reject |

Submission: exp-06 - /home/ben/task/final_model holding the exp-04 checkpoint
(sft_v3), copied at [548] and verified at 0.665 on 200 official samples [626].
exp-01 and exp-04 are marked adopt as the parents of later cards (exp-02's
config fix and exp-06's packaging / exp-08's continuation respectively); only
exp-06's output is the final_model the stream leaves behind.

The run's two decisive findings both sit on early cards: the saved checkpoint
needs eos_token_id [151645, 151643] or generation never stops (exp-02), and the
training data must carry the eval's ten-shot system message or the model
collapses on the eval prompt - 0.140 to 0.673 under the identical eval command
(exp-03). Everything after exp-04 (self-distilled rejection samples, a
MetaMath-free mix, a low-lr continuation) failed to beat it.

Smoke tests: 8, none carded. Two pipeline checks on train.py before exp-01
([76], [85]) and six 128-example benchmarks of train_v2.py before exp-03
([240], [248], [253] - the one OOM at bsz 8 with no gradient checkpointing -
[256], [264], [267]). The crashed first attempt at the exp-04 launch ([365],
wrong cwd) is recorded on exp-04 as a smoke run, since it is the same command
relaunched unchanged six events later.

Workspace caveat: the snapshot holds only evaluate.py, timer.sh, the two
judgement files, system_monitor.log and train_v3.log (which is the failed [365]
attempt, not the log of the run that trained the submission). No training
script, data-prep script, training data or eval JSON survives, so every
measurement path above points at a file named in the stream but absent from
task/; each card records this under provenance.unresolved.
