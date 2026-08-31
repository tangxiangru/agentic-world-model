# r-b8ac2f29 - reconstructed experiment cards

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, one H100.
8 launches carded, from t=+0.17h to t=+5.15h. The stream runs to the end of the
budget (t=+9.08h); everything after the last card is evaluation, GRPO smoke
attempts and cleanup.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 14168 | 0.17 | sft | base_model | gsm8k train gold, 7,473, eval-format targets | 1e-5 / 2 | completed | accuracy 0.8467, n=150 (base 0.44 at n=100) | inconclusive | reject |
| exp-02 | 23150 | 1.27 | sft | base_model | gold + OpenMathInstruct-2 gsm8k mix ("60K externals", build args unknown) | 1e-5 / 1 | killed | - | inconclusive | abandon_line |
| exp-03 | 26878 | 1.56 | sft | base_model | gold x2 + 45K OMI2 gsm8k (pre-comma-fix data) | 1e-5 / 1 | killed | - | inconclusive | abandon_line |
| exp-04 | 31855 | 1.65 | sft | base_model | gold x2 + 45K OMI2 gsm8k, comma-convention answers | 1e-5 / 1 | completed | accuracy 0.8800, n=150 (+0.0333 vs exp-01); 0.8415, n=1319 | supported | adopt |
| exp-05 | 34592 | 3.69 | rft | exp-04 | RAFT self-solutions x2 (14,457) + gold + 15K OMI2 | 5e-6 / 1 | killed | - | inconclusive | abandon_line |
| exp-06 | 35547 | 3.76 | rft | exp-04 | RAFT self-solutions x2 + gold + 12K OMI2 | 5e-6 / 1 | failed | - (757 steps trained, lost at save) | inconclusive | abandon_line |
| exp-07 | 38690 | 5.15 | rft | exp-04 | RAFT self-solutions x2 + gold + 12K OMI2 | 5e-6 / 1 | completed | accuracy 0.8400, n=150 (-0.04 vs exp-04) | contradicted | reject |
| exp-08 | 39005 | 5.15 | other (package exp-04 checkpoint + greedy decode config -> final_model) | exp-04 | - | - | completed | accuracy 0.8867, n=150 at grader defaults (+0.0067 vs exp-04); 0.8750, n=400 | supported | adopt |

Submission: exp-08 - final_model/ holding the exp-04 (phase-2) weights,
sha256- and diff-verified byte-identical to the checkpoint that produced the one
completed 1319-sample run (0.8415). exp-04 is also marked adopt as the incumbent
and the parent of every later card.

Smoke tests (not carded): two packing dry runs before exp-03 ([25512] OOM at
bs12, [26543] passed at bs8/accum8), one save-path dry run before exp-07
([38670], --limit 64), and seven GRPO colocate smoke runs after exp-08
([40955]-[46502], all OOM or engine crash) which are recorded on exp-08 because
no real launch followed them.

Run-level notes: the RAFT generation step (gen_raft.py) produced data/raft.jsonl
but its invocation never appears in the digest, so it is carried as a data
provenance gap on exp-05/06/07 rather than as a card. Three of the four
1319-sample evaluations died on nondeterministic vLLM CUDA illegal-memory-access
crashes, so the full-test numbers for exp-01, exp-07 and final_model itself are
partial reads from live logs with no output file in the snapshot.
