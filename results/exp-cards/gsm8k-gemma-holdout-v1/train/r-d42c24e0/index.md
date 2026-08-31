# r-d42c24e0 — extracted experiment cards

Base model: HuggingFaceTB/SmolLM3-3B-Base · benchmark: gsm8k · budget: 10 h on one H100.
All measurements are the agent's own `evaluate.py` runs (`official --limit N`); no official score is written here.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 99 | 0.11 | sft | base_model | train_messages.jsonl (191194) | 1e-5 / 1.0 | completed | 0.060 @50 (0.033 @30) | inconclusive | adopt |
| exp-02 | 198 | 1.46 | sft | exp-01 | train_messages_v2.jsonl (88787) | 1e-5 / 1.0 | completed | 0.587 @150 | supported | adopt |
| exp-03 | 287 | 2.79 | sft | base_model | train_messages_v2.jsonl (88787) | 2e-5 / 1.0 | completed | 0.420 @50 | inconclusive | reject |
| exp-04 | 293 | 2.80 | other (package to final_model) | exp-02 | — | — | completed | none | inconclusive | adopt |
| exp-05 | 350 | 4.14 | sft | exp-02 | train_messages_v4b.jsonl (100000) | 5e-6 / 1.0 | completed | 0.600 @150 | inconclusive | adopt |
| exp-06 | 409 | 5.58 | sft | exp-05 | train_messages_v5.jsonl (44946) | 5e-6 / 1.0 | completed | 0.513 @150 | contradicted | reject |
| exp-07 | 460 | 6.87 | sft | exp-05 | train_messages_v6.jsonl (120000) | 8e-6 / 1.0 | completed | 0.607 @150 | inconclusive | adopt |
| exp-08 | 475 | 6.91 | other (package to final_model) | exp-05 | — | — | completed | none | inconclusive | adopt |
| exp-09 | 525 | 8.83 | rft | exp-07 | train_messages_rft.jsonl (33074) | 5e-6 / 1.0 | completed | 0.607 @150 | inconclusive | adopt |
| exp-10 | 572 | 9.40 | merge (soup → final_model) | model_soup (merge absent from stream) | — | — | completed | 0.627 @150 (0.633 @30) | inconclusive | adopt |
| exp-11 | 585 | 9.73 | rft | model_soup (merge absent from stream) | train_messages_rft2b.jsonl (20000) | 3e-6 / 1.0 | failed (CUDA OOM at step 0) | none | inconclusive | abandon_line |
| exp-12 | 610 | 9.76 | rft | exp-10 | train_messages_rft2c.jsonl (8000) | 3e-6 / 1.0 | completed | 0.620 @150 | inconclusive | reject |

Notes

- The submission is exp-10: the weight-averaged `model_soup` promoted to `final_model` at [572] and re-packaged at [604] and [625]; 0.6267 on 150 samples, 0.6333 on the 30-sample confirmation.
- The command that built `model_soup` is not in the digest — it first appears already built as the eval target at [562]. Its ingredients are known only from the agent's closing summary ("v4/v6/rft") and the averaging weights are stated nowhere, so exp-10 and exp-11 have `parent_checkpoint.origin: null`.
- `adopt` on exp-01/02/05/07/09 follows the rule that an output which became `final_model` or the parent of a later card is adopted, not that the number was good.
- Packaging steps that only staged a checkpoint for evaluation (`package_model.py --dst model_vN`, and the dual-EOS repack at [248]) are folded into the corresponding training card; only the three writes to `final_model` are carded (exp-04, exp-08, exp-10 — the last covering [572], [604] and [625]).
