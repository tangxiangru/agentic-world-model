# r-a3f23d29 — reconstructed experiment cards

Base model Qwen/Qwen3-4B-Base, gsm8k, 10 h budget, one H100. Five launches carry a
candidate; the digest's 110 recipe-bearing events end at t=+6.62h with ~3.5 h of
budget unspent. Accuracies are the agent's own evals; `@N` is the `--limit`.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 196 | 0.15 | sft | base_model | train.jsonl — gsm8k gold + MetaMathQA (43473) | 1e-5 / 3 | completed | 0.627 @150 (eval_v1.json) | inconclusive | adopt |
| exp-02 | 401 | 1.94 | decode-config | exp-01 | — (repackage only) | — / — | completed | 0.833 @150 (eval_v1_greedy.json); 0.840 @500 (eval_v1_500.json) | supported | adopt |
| exp-03 | 501 | 2.11 | rft | base_model | train_v2.jsonl — gold + rs.jsonl self-sampled + MetaMathQA (39916) | 1e-5 / 3 | completed | 0.840 @500 (eval_v2_500.json) | contradicted | reject |
| exp-04 | 650 | 3.64 | sft | base_model | train_v3.jsonl — gold + rs.jsonl + OpenMathInstruct-2 (29336) | 1e-5 / 3 | completed | 0.846 @500 (eval_v3_500.json) | inconclusive | adopt |
| exp-05 | 697 | 4.74 | sft | base_model | train_v4.jsonl — gold + rs.jsonl + OpenMathInstruct-2 + MetaMathQA (47336) | 1e-5 / 3 | completed | 0.860 @500 (eval_v4_500.json) | supported | adopt |

Notes

- exp-05 is the last checkpoint copied to `final_model` ([774]). Its full-test
  confirmation eval (1319 items) was launched but never returns inside the
  digest, and `eval_final_full.json` is absent from the workspace snapshot.
- `final_model` is a moving target: exp-02's checkpoint from [401], exp-04's from
  [693], exp-05's from [774]. Each card's `output_checkpoint` names the durable
  directory instead.
- The base-model probe (0.45 @100, `baseline.json`, launched at [45]) is not a
  card: it produced no candidate. It is exp-01's comparator, at a different
  `--limit`.
- No smoke tests or dry runs appear anywhere in the stream, so every card has
  `provenance.smoke_runs: []`.
- Every training card trains from base with the same hyper-parameters and differs
  only in `--data`; the run contains no hyper-parameter evidence.
