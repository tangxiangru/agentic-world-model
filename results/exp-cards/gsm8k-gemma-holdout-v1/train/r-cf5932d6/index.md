# r-cf5932d6 — extracted experiment cards

base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h, 1x H100
6 cards, launch order. All measurements are the agent's own evals; official accuracy is never written here.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 163 | 0.34 | sft | base_model | train.jsonl (OpenMathInstruct-2 + gsm8k train, 207473; --max_samples 150000) | 1e-5 / 1.0 | killed | none | inconclusive | abandon_line |
| exp-02 | 244 | 1.55 | sft | base_model | train.jsonl (OpenMathInstruct-2 + gsm8k train, 207473; --max_samples 100000) | 1e-5 / 1.0 | completed | accuracy 0.56 (n=150, eval_v1.json) | inconclusive | adopt |
| exp-03 | 321 | 2.96 | sft | exp-02 | train_v2.jsonl (build never shown; agent calls it "107k OMI2 continuation") | 3e-6 / 1.0 | completed | accuracy 0.5533 (n=150, eval_v2.json), -0.0067 vs exp-02 | inconclusive | reject |
| exp-04 | 330 | 3.05 | other (packaging) | exp-02 | — | — / — | completed | none | inconclusive | reject |
| exp-05 | 379 | 4.43 | sft | base_model | train_v3.jsonl (OpenMathInstruct-2 + MetaMathQA GSM_* + gsm8k train, 207237; --max_samples 200000) | 1e-5 / 1.0 | completed | accuracy 0.5867 (n=150, eval_v3.json), +0.0267 vs exp-02 | inconclusive | adopt |
| exp-06 | 449 | 7.13 | other (packaging) | exp-05 | — | — / — | completed | accuracy 0.5733 (n=150, eval_final.json), -0.0133 vs exp-05 | inconclusive | adopt |

Submitted candidate: **exp-06** — the exp-05 checkpoint copied into `final_model` at [449] and verified there at 0.5733 on the 150-item protocol.

Not cards (pipeline smoke tests, recorded on exp-01 as `provenance.smoke_runs`): [143], [145], [147] (500-sample / 0.005-epoch pipeline checks) and [154] (1024-sample throughput benchmark).
