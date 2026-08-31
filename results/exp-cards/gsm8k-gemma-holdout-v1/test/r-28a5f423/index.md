# r-28a5f423 — extracted experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100
13 cards. Submitted checkpoint: exp-06 (`final_model`, the merged end-of-epoch-1 LoRA adapter), accuracy 0.647 on 150 samples and 0.625 on 200.
Per-event timestamps are present, so `elapsed_h` is filled on every card.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 42 | 0.03 | other (package base to final_model) | base_model | — | — | completed | accuracy 0.050 @ n=20 | inconclusive | reject |
| exp-02 | 88 | 0.22 | sft | base_model | gsm8k train (7473) + MetaMathQA GSM (240000) + ORCA-Math (50000) = 297473 | 2e-4 / 1 | killed | accuracy 0.500 @ n=50 (merged, exp-03) | inconclusive | adopt |
| exp-03 | 117 | 2.24 | merge | exp-02 | — | — | completed | accuracy 0.500 @ n=50 | inconclusive | reject |
| exp-04 | 135 | 2.32 | sft (resume) | exp-02 | same 297473 mixture | 2e-4 / 1 | completed | accuracy 0.647 @ n=150 (merged, exp-06) | inconclusive | adopt |
| exp-05 | 151 | 2.77 | merge | exp-04 | — | — | completed | — (vLLM would not start; never scored) | inconclusive | abandon_line |
| exp-06 | 179 | 3.98 | merge | exp-04 | — | — | completed | accuracy 0.647 @ n=150; 0.625 @ n=200 | inconclusive | adopt |
| exp-07 | 194 | 4.07 | sft (2nd epoch) | exp-04 | same 297473 mixture | 5e-5 / 1 | completed | accuracy 0.600 @ n=150 (merged, exp-08; −0.047 vs exp-06) | contradicted | reject |
| exp-08 | 309 | 7.37 | merge | exp-07 | — | — | completed | accuracy 0.600 @ n=150 (−0.047 vs exp-06) | contradicted | reject |
| exp-09 | 322 | 7.44 | merge | exp-07 | — | — | failed | — (checkpoint-3000 already deleted) | inconclusive | abandon_line |
| exp-10 | 339 | 7.46 | merge | exp-07 | — | — | completed | accuracy 0.613 @ n=150 (−0.034 vs exp-06) | contradicted | reject |
| exp-11 | 345 | 7.53 | merge | exp-07 | — | — | completed | accuracy 0.640 @ n=150 (−0.007 vs exp-06) | inconclusive | reject |
| exp-12 | 360 | 7.61 | merge | exp-04 | — | — | completed | accuracy 0.580 @ n=150 (−0.067 vs exp-06) | contradicted | reject |
| exp-13 | 401 | 7.74 | sft | base_model | gsm8k train only (7473) | 2e-4 / 5 | killed | — (no checkpoint produced) | inconclusive | abandon_line |

The run is one SFT recipe plus a checkpoint sweep: a 297k mixture of GSM8K train,
MetaMathQA GSM-type items and ORCA-Math, LoRA r=64 for one epoch (exp-02 killed at
step 2500, exp-04 resumed and finished), merged at exp-06 for 0.647. A second epoch
at 5e-5 (exp-07) and four alternative checkpoints (exp-08, exp-10, exp-11, exp-12)
were all scored against that 0.647 at the same `--limit 150` and none beat it; the
closest, epoch-2 step 4500, came within 0.7 pts. The one recipe change the agent
proposed — GSM8K train alone, no synthetic mixture (exp-13) — never produced a
checkpoint.

Not cards: the smoke and API-fight launches, which are recorded as
`provenance.smoke_runs` on the card they preceded — [44] and [47] (vLLM could not
serve the packaged base model) on exp-01; [78] (`SFTConfig` has no `max_seq_length`)
on exp-02; and [372], [382], [391], [408] on exp-13. Also not cards: the `mv`/`rm`
restores of the incumbent into `final_model` at [317], [350] and [365], which carry
out the `reject` decision on exp-08, exp-11 and exp-12 rather than producing a
candidate; and the 200-sample re-score at [418], which re-measures exp-06's weights
and is recorded as a second measurement on that card.

Run-level: the digest ends at t=+7.86h with about 2.1 h of the 10 h budget
unaccounted for, so the final state of `final_model` rests on the listing at [414]
and the eval at [419], both of which still show exp-06's merge. The agent's own eval
logs (`logs/*.json`) are not in the workspace snapshot, so every accuracy is read
from the printed summary in the stream.
