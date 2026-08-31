# r-6b7e26ac - reconstructed experiment cards

Base model: google/gemma-3-4b-pt | benchmark: gsm8k | budget: 10 h, one H100.
7 launches carded. The stream runs to event [831], t=+8.65h, and ends with the
submission packaged, re-verified and the GPU idle, so the run is complete.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 181 | 0.30 | sft | base_model | gsm8k train 7473 + MetaMathQA GSM rephrased/ansaug 30000 | 2e-5 / 2 | completed | accuracy 0.593, n=150 (base 0.100 at n=100) | inconclusive | adopt |
| exp-02 | 414 | 1.27 | decode-config | exp-01 | - | - | completed | accuracy 0.613, n=150 (+0.020 vs exp-01 same limit) | supported | adopt |
| exp-03 | 441 | 1.36 | sft | base_model | gsm8k train + MetaMathQA GSM 25k/12k per type + 4000 few-shot-augmented, ~85K | 2e-5 / 2 | completed | accuracy 0.707, n=150 and n=300 (+0.094 vs exp-02) | supported | adopt |
| exp-04 | 585 | 3.75 | sft | base_model | exp-03 mixture minus few-shot, plus self-generated rejection-sampled CoT, ~107K | 2e-5 / 2 | completed | accuracy 0.200, n=150 (-0.507 vs exp-03) | contradicted | reject |
| exp-05 | 645 | 6.37 | other (package runs/v2 + greedy config -> final_model) | exp-03 | - | - | completed | accuracy 0.702, n=500; 0.693 on the default invocation, n=150 | supported | adopt |
| exp-06 | 685 | 6.52 | sft | base_model | gsm8k train + MetaMathQA GSM 45k/18k per type, ~133K | 2e-5 / 1 | completed | accuracy 0.390, n=300 (-0.317 vs exp-03) | contradicted | adopt |
| exp-07 | 763 | 8.32 | sft | exp-06 | gsm8k train 7473 only (continued fine-tune) | 1e-5 / 2 | completed | accuracy 0.317, n=300 (-0.073 vs exp-06) | contradicted | reject |

Submission: exp-05 - final_model, a byte-identical copy of the exp-03 checkpoint
(runs/v2) with a greedy generation_config. Confirmed at 0.695 (n=200), 0.702
(n=500) and 0.693 on the exact default grader invocation (n=150). exp-01 and
exp-03 are marked adopt as the parents of cards downstream of them; exp-06 is
marked adopt only because runs/v4 is exp-07's parent checkpoint - it was
evaluated at 0.390 and never became the incumbent.

The shape of the run: format was fixed almost immediately (baseline 0.100 -> v1
0.593), greedy decoding added ~2 pts, one data scale-up to ~85K reached ~70%,
and then every further scale-up (v3 with rejection data at ~107K, v4 at ~133K,
and the v4b rescue) collapsed into models that answer and then re-emit the
prompt template instead of stopping. That failure is diagnosed but never
explained; the agent's closing conjecture is overfitting [741].

Smoke tests: one - [167], a 64-example, 1-epoch, max_len 768 dry run of
train_sft.py before the first real launch; it passed and is recorded on exp-01.

Not carded: the baseline eval of the untouched base model at [70] (no candidate
produced - it is exp-01's comparator); the data-building runs of data_prep.py,
make_fewshot_data.py and gen_rejection.py (they build files, not candidates -
they are recorded as `setup.data[].build_command`); and the make_greedy.py
invocations at [504], [604], [702] and [775], which reapply the decode setting
established by exp-02 as part of the evaluation protocol of exp-03, exp-04,
exp-06 and exp-07 and are recorded on those cards. The first one, at [414], is
carded, because there it was the intervention under test.

Run-level caveats: the workspace snapshot holds only the agent's scripts - no
`logs/`, no `data/` - so no measurement in these cards can be re-read from
`task/`; all values are the ones printed in the stream. `data_prep.py` in the
snapshot is its end-of-run version (MetaMath caps 45000/18000), which matches
only exp-06; `train_sft.py` likewise gained its processor-file copy at [392] and
its `--model` argument at [753], after exp-01 and before exp-07.
