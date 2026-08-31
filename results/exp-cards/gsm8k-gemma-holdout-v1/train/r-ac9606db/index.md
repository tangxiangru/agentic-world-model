# r-ac9606db — extracted experiment cards

Base model post-trained: HuggingFaceTB/SmolLM3-3B-Base · benchmark gsm8k · 10 h budget · 1x NVIDIA H100 80GB HBM3.
9 launches, in launch order. `best measurement` is the agent's own eval; every path is the
run workspace path named in the stream, none of which survive in the workspace snapshot.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 138 | 0.25 | sft | base_model | sft.jsonl (gsm8k train x2 + MetaMathQA GSM_Rephrased/AnsAug), 135000 cap | 1e-5 / 2 | killed | none (killed at 4 min) | inconclusive | abandon_line |
| exp-02 | 208 | 0.44 | sft | base_model | sft.jsonl, 135000 cap | 1e-5 / 2 | completed | 0.593 @150 (runs/v1_eval.json) | inconclusive | adopt |
| exp-03 | 299 | 2.58 | decode-config | exp-02 | — | — / — | completed | 0.767 @150 (runs/v1_greedy_eval.json), vs 0.593 sampling | supported | adopt |
| exp-04 | 309 | 2.61 | other (package v1 → final_model) | exp-03 | — | — / — | completed | none (v1 copied to final_model) | inconclusive | adopt |
| exp-05 | 318 | 2.61 | sft | base_model | sft2.jsonl (adds microsoft/orca-math; 159946), 160000 cap | 1e-5 / 3 | killed | none (killed at 2 min) | inconclusive | abandon_line |
| exp-06 | 392 | 2.77 | rft | base_model | sft3.jsonl (reject.jsonl 20980 x2 + 110000 from sft2; 151960) | 1e-5 / 2 | completed | 0.793 @150 (runs/v3_ep2_eval.json); 0.773 @150 epoch-1; 0.770 @300 | supported | adopt |
| exp-07 | 454 | 5.97 | other (package v3 → final_model) | exp-06 | — | — / — | completed | none (v3 copied to final_model) | inconclusive | adopt |
| exp-08 | 523 | 6.12 | rft | base_model | sft4.jsonl (reject 20980 + reject2 28213 + 100000 from sft2; 149193) | 1e-5 / 2 | completed | 0.793 @150 (runs/v4_ep2_eval.json); 0.777 @300 (runs/v4_300.json) | contradicted | adopt |
| exp-09 | 571 | 9.25 | other (package v4 → final_model) | exp-08 | — | — / — | completed | 0.780 @150 (runs/final_check.json) | inconclusive | adopt |

Submission: exp-09 — final_model is a copy of the v4 checkpoint (exp-08), stripped of its
per-epoch checkpoint directories and carrying the greedy generation_config.json, verified
end to end at [577]–[588]. exp-02, exp-03, exp-04, exp-06 and exp-07 are marked adopt as
earlier final_model states or as the parent of a later card; only exp-09's output is the
artifact the stream leaves behind.

The run's own headline is a decoding change, not a training one: the eval sends no
temperature, so vLLM sampled at 1.0; writing `temperature: 0.0` into the checkpoint's
generation_config.json moved the same v1 weights from 0.593 to 0.767 at --limit 150
(exp-03). Every candidate after that carries the greedy config.

Smoke tests: one — the 12000-example, 1-epoch pipeline check at [164], recorded on
exp-02's card. Scripts written and never run: none; gen_reject.py, prepare_v3.py and
prepare_v4.py were all invoked.

Run-level gaps: the workspace snapshot holds only evaluate.py, timer.sh,
system_monitor.log and the two judgement files, so no training script, data file or
eval json can be re-read — every number here comes from the stream. The eval logs the
agent analysed (logs/*.json) are likewise absent.
