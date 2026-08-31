# r-4254277e — reconstructed experiment cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 118 | 0.15 | sft | base_model | gsm8k_train.jsonl (7473) | 1e-5 / 3 | completed | acc 0.053 @ limit 150 | inconclusive | reject |
| exp-02 | 195 | 0.35 | sft | base_model | gsm8k_train.jsonl (7473), few-shot wrapper | 1e-5 / 3 | failed | none (OOM at startup) | inconclusive | abandon_line |
| exp-03 | 212 | 0.38 | sft | base_model | gsm8k_train.jsonl (7473), few-shot wrapper | 1e-5 / 3 | completed | acc 0.04 @ limit 150 | contradicted | adopt |
| exp-04 | 382 | 0.88 | decode-config | exp-03 | none (config only) | n/a | completed | acc 0.64 @ limit 150 | supported | adopt |
| exp-05 | 409 | 0.96 | sft | base_model | gsm8k_train x2 + mm_ansaug 40k + mm_rephrased 40k | 1e-5 / 2 | failed | none (save_pretrained crashed; the reported 0.64 was a stale log) | inconclusive | abandon_line |
| exp-06 | 435 | 1.05 | other (package to final_model) | exp-04 | none (copy only) | n/a | completed | none of its own; inherits exp-04 acc 0.64 @ limit 150 | inconclusive | adopt |
| exp-07 | 649 | 3.20 | rft | base_model | rft1.jsonl (self-sampled, count unknown) + gsm8k_train x2 | 1e-5 / 2 | killed | none | inconclusive | abandon_line |

Run-level notes:
- The digest holds all 199 recipe-bearing events but ends at t=+3.27h of a 10h budget, with the rejection-sampling generation 11 percent done; NOTES.md in the workspace snapshot stops at the same point. Nothing about the last ~6.7h is recoverable.
- The submission in place at the end of the stream is exp-06, final_model, a copy of the exp-04 checkpoint (few-shot SFT on GSM8K plus a greedy generation_config patch) measured at 0.64 on limit 150.
- The workspace snapshot keeps only NOTES.md, evaluate.py, timer.sh and the two judgement files, so every cited script, data and eval-output path is a runtime path that cannot be re-read; all measurements come from the agent's prose in the stream.
