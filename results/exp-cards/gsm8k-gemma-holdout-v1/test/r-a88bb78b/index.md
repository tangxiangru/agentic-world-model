# Reconstructed experiment cards — run_ref r-a88bb78b

Base model post-trained: google/gemma-3-4b-pt · benchmark gsm8k · 10 h budget · one 80GB GPU.
9 cards, one per launch found in the digest. Smoke/dry runs are not cards (see `provenance.smoke_runs`).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 194 | 0.31 | sft | base_model | data/sft.jsonl (gsm8k train x2 + MetaMathQA GSM 120k) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-02 | 238 | 0.44 | sft | base_model | data/sft.jsonl | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-03 | 287 | 0.56 | sft | base_model | data/sft.jsonl | 1e-5 / 2 | completed | 0.115 @ n=200 (logs/eval_v1_final.json) | inconclusive | adopt |
| exp-04 | 463 | 3.95 | sft | exp-03 | data/sft_fs.jsonl (18k, K=0..8 few-shot) | 1e-5 / 2 | killed | — | inconclusive | abandon_line |
| exp-05 | 479 | 4.00 | sft | exp-03 | data/sft_fs.jsonl (18k, K=0..8 few-shot) | 1e-5 / 2 | completed | 0.70 @ n=200 (logs/eval_v2_final.json) | supported | adopt |
| exp-06 | 531 | 5.28 | other (packaging → final_model) | exp-05 | — | — / — | completed | 0.679 @ n=1319 (logs/eval_final_full.json) | inconclusive | reject |
| exp-07 | 570 | 5.62 | rft | exp-05 | data/sft_v3_fs.jsonl (rejection 19046 + 5000 sft, K=0..4) | 1e-5 / 2 | completed | 0.702 @ n=1319 (logs/eval_v3_full.json) | supported | adopt |
| exp-08 | 623 | 6.95 | other (packaging → final_model) | exp-07 | — | — / — | completed | 0.687 @ n=150 (logs/eval_FINAL_confirm.json); 0.702 @ n=1319 | inconclusive | adopt |
| exp-09 | 664 | 7.91 | rft | exp-05 | data/sft_v4_fs.jsonl (rejection2 28823 cleaned + 6000 sft, K=0..4) | 1e-5 / 2 | completed | 0.700 @ n=1319 (logs/eval_v4_full.json) | contradicted | reject |

Submission: **exp-08** — final_model rebuilt from the exp-07 checkpoint at [623] and confirmed under the grader's default command at [727]; this is the last write to final_model in the stream.

Run-level notes:
- The workspace snapshot holds only `contamination_judgement.txt`, `disallowed_model_judgement.txt`, `evaluate.py`, `system_monitor.log`, `timer.sh`. Every training/data script and every eval JSON the cards cite is absent, so all measured values are the agent's own reported numbers from the digest, and hyper-parameters not on a launch argv were read from the script text captured in the digest write events (noted per card in `hyperparams.other`).
- No baseline number for the base model exists: the baseline eval launched at [56] was killed at [167] before finishing.
