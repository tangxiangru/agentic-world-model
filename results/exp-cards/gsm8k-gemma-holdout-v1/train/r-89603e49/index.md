# r-89603e49 — extracted experiment cards (gsm8k, Qwen/Qwen3-1.7B-Base, 10 h, 1x H100)

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 17875 | 0.33 | sft | base_model | data/sft_v1.jsonl (98205) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-02 | 18544 | 0.41 | sft | base_model | data/sft_v1.jsonl (98205) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-03 | 19154 | 0.47 | sft | base_model | data/sft_v1.jsonl (98205) | 1e-5 / 2 | completed | 0.727 @ n=150 (base 0.133) | supported | adopt |
| exp-04 | 28893 | 3.14 | sft | base_model | data/sft_v2.jsonl (+ rft_v1.jsonl) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-05 | 29447 | 3.27 | sft | base_model | data/sft_v2.jsonl (+ rft_v1.jsonl) | 1e-5 / 1.6 | killed | — | inconclusive | abandon_line |
| exp-06 | 29614 | 3.28 | sft | base_model | data/sft_v2.jsonl (+ rft_v1.jsonl) | 1e-5 / 1.6 (cut at 1.5) | killed | 0.8267 @ n=150 (vs exp-03 0.727) | supported | adopt |
| exp-07 | 33782 | 7.02 | sft | base_model (bug: intended exp-06) | data/sft_v3.jsonl (+ rft_v2.jsonl) | 5e-6 / 1 | killed | — | inconclusive | abandon_line |
| exp-08 | 34040 | 7.03 | sft | exp-06 | data/sft_v3.jsonl (+ rft_v2.jsonl) | 5e-6 / 1 | completed | 0.8333 @ n=150 (vs exp-06 0.8267) | inconclusive | adopt |
| exp-09 | 34254 | 7.09 | other (packaging) | exp-06 | — | — | completed | — | inconclusive | reject |
| exp-10 | 38078 | 9.19 | other (packaging) | exp-08 | — | — | completed | 0.8333 @ n=150 on final_model | supported | adopt |

Submitted artefact: exp-10 (`final_model/`, packaged from exp-08's checkpoint). Smoke runs [16290], [16887] are recorded on exp-01; the baseline eval at [8121] (0.1333 @ n=150) is the comparator for exp-01–exp-03.
