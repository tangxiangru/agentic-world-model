# r-d2611dd8 - reconstructed experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base | benchmark: gsm8k | budget: 10 h, one H100.
7 launches carded. The stream runs to event [356], t=+8.93h, with the final
model in place and 1:04 of budget left, so the run is reconstructed end to end.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 121 | 0.37 | sft | base_model | gsm8k train 7,473 + MetaMathQA GSM_* 60,000 = 67,473 | 1e-5 / 1 | completed | accuracy 0.540, n=150 | inconclusive | reject |
| exp-02 | 174 | 1.22 | sft | base_model | gsm8k train 7,473 + all MetaMathQA GSM_* 240,000 = 247,473 | 1e-5 / 1 | completed | accuracy 0.580, n=150 (exp-01 0.540 same limit) | inconclusive | adopt |
| exp-03 | 224 | 4.23 | rft | exp-02 | self-samples of exp-02 52,318 + 50,000 of the exp-02 mixture = 102,318 | 5e-6 / 1 | completed | accuracy 0.716, n=500; 0.707 n=150 (exp-02 0.580 same limit) | supported | adopt |
| exp-04 | 259 | 5.58 | rft | exp-03 | self-samples of exp-03 53,381 + 20,000 of the exp-02 mixture = 73,381 | 2e-6 / 1 | completed | - (evaluated at n=500, result not in the stream) | inconclusive | reject |
| exp-05 | 286 | 6.46 | rft | base_model | self-samples of exp-03 53,381 + gsm8k canonical 7,473 x2 + 30,000 MetaMath = 98,327 | 1e-5 / 1 | completed | accuracy 0.596, n=500 (exp-03 0.716 same limit) | contradicted | reject |
| exp-06 | 305 | 7.58 | other (package exp-03 -> final_model) | exp-03 | - | - | completed | accuracy 0.692, n=500; 0.627 and 0.673 on n=150 | inconclusive | adopt |
| exp-07 | 318 | 7.70 | rft | exp-03 | self-samples of exp-02 52,318 + exp-03 53,381 + gsm8k canonical 7,473 = 113,172 | 1e-6 / 1 | completed | accuracy 0.680, n=500 (exp-03 0.716 same limit) | contradicted | reject |

Submission: exp-06 - final_model holding a verbatim copy of the exp-03
checkpoint. exp-02 and exp-03 are marked adopt as the parents of later cards;
only exp-06's output is the final_model the stream leaves behind, and nothing
after [305] overwrites it.

The run's one real gain is exp-03: warm-starting rejection-sampling fine-tuning
from the exp-02 SFT checkpoint, +12.7 pts on the 150-sample protocol. exp-04
through exp-07 all failed to beat it, and exp-05 shows why the warm start
mattered - the same data trained from base lost 12 pts.

Smoke tests: four, all before exp-01 - [96] and [99] and [111] passed on 100-200
sample slices, [108] crashed with a CUDA OOM at batch 16 with gradient
checkpointing off, which fixed the batch 8 x grad_accum 2 configuration every
later launch used.

Two launches in the stream are data generation, not candidates, so they are
carded as build steps rather than experiments: generate_rft.py at [208] (52,318
kept samples, on exp-03's card) and at [248] (53,381, on exp-04's card).

Run-level gaps: the eval of exp-02 and the eval result of exp-04 are both absent
from the digest, and no eval output file (`/tmp/eval_*.json`, `logs/*.json`)
survives in the workspace snapshot, so every accuracy here can only be re-read
from the stream.
