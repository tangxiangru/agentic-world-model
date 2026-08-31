# r-cbcfc798 — gsm8k, Qwen/Qwen3-1.7B-Base, 10 h, 1x H100

13 cards. The digest carries no timestamps, so every `elapsed_h` is null.
The submitted model is exp-11: the exp-06 epoch-3 weights decoded greedily.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 60 | null | sft (LoRA r=64) | base_model | gsm8k 7473 + MetaMathQA GSM 240000 + MATH 10000 | 2e-4 / 3 | failed | — | inconclusive | iterate |
| exp-02 | 69 | null | sft (LoRA r=64) | base_model | gsm8k 7473 + MetaMathQA GSM 240000 + MATH 10000 | 2e-4 / 3 (killed at 1.0) | killed | — (loss 0.197 @ step 5425/16212) | inconclusive | adopt |
| exp-03 | 119 | null | merge | exp-02 | — | — | completed | 0.140 @ 50 | inconclusive | adopt |
| exp-04 | 142 | null | decode-config | exp-03 | — | — | completed | 0.100 @ 50 (-0.040 vs exp-03) | inconclusive | reject |
| exp-05 | 151 | null | decode-config | exp-03 | — | — | completed | 0.220 @ 50 (+0.120 vs exp-04) | inconclusive | adopt |
| exp-06 | 160 | null | sft (full) | base_model | gsm8k 7473 x3 + MetaMathQA GSM_Rephrased/GSM_AnsAug | 2e-5 / 3 | completed | 0.260 @ 50 (+0.040 vs exp-05) | supported | adopt |
| exp-07 | 241 | null | sft (full) | base_model | gsm8k 7473 x3 + MetaMathQA 4 GSM types | 5e-5 / 2 | completed | 0.060 @ 50 (-0.200 vs exp-06) | contradicted | reject |
| exp-08 | 282 | null | other (packaging) | exp-06 | — | — | completed | 0.180 @ 50 (-0.080 vs exp-06) | inconclusive | adopt |
| exp-09 | 293 | null | sft (full) | exp-06 | gsm8k 7473 only | 1e-5 / 5 | completed | 0.140 @ 50 (-0.040 vs exp-08) | contradicted | reject |
| exp-10 | 310 | null | other (packaging) | exp-06 | — | — | completed | 0.280 @ 150 (no comparator) | inconclusive | adopt |
| exp-11 | 326 | null | decode-config | exp-10 | — | — | completed | 0.727 @ 150 (+0.447 vs exp-10) | supported | adopt |
| exp-12 | 339 | null | sft (full) | exp-06 | gsm8k 7473 x5 + MetaMathQA 4 GSM types (277365 total) | 3e-5 / 1 | killed | — (step 8/5697) | inconclusive | abandon_line |
| exp-13 | 370 | null | sft (LoRA r=128) | exp-06 | gsm8k 7473 x3 + MetaMathQA 4 GSM types (262419 total) | 1e-4 / 1 | killed | — (step 105/5454) | inconclusive | abandon_line |

Notes

- All accuracies are the agent's own `evaluate.py` runs against the official
  gsm8k set at `--limit 50` or `--limit 150`. Only exp-11's last run has a file
  in the workspace snapshot (`final_eval.json`, 0.7133); the other numbers cite
  the `logs/*.json` paths named in the stream, which the snapshot does not carry.
- No baseline evaluation of the base model was ever run, so no card has a
  `base_model` comparator.
- exp-04, exp-05 and exp-11 are file edits to `generation_config.json`, not
  shell launches; their `setup.command.argv` records the write block verbatim.
- Smoke runs: none. exp-01 is a card because it was the full run, not a
  truncated pipeline check, and it crashed.
- Checkpoint-level evaluations that produced no new artifact are folded into the
  card for the run that produced the checkpoint: exp-06 carries the epoch-1
  (0.200 @ 50) and epoch-2 (0.213 @ 150 sampled, 0.707 @ 150 greedy) numbers,
  exp-09 carries its greedy 0.153 @ 150.
