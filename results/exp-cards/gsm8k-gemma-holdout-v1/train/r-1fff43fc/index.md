# r-1fff43fc — gsm8k, HuggingFaceTB/SmolLM3-3B-Base, 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 173 | 0.12 | sft | base_model | work/train.jsonl (gold GSM8K-train CoT, ANSWER-format) | 1e-5 / 3 | completed | 0.533 @150 (same weights read 0.213 before the eos patch) | inconclusive | adopt |
| exp-02 | 320 | 0.73 | other (package to final_model) | exp-01 | — | — / — | completed | — (carries exp-01's 0.533 @150) | inconclusive | adopt |
| exp-03 | 346 | 0.74 | decode-config (greedy) | exp-01 | — | — / — | completed | 0.733 @150 | supported | adopt |
| exp-04 | 360 | 0.79 | decode-config (package to final_model) | exp-02 | — | — / — | completed | — (carries exp-03's 0.733 @150) | inconclusive | adopt |
| exp-05 | 415 | 0.92 | sft | base_model | work/train_v2.jsonl (gold + RFT round 1 from exp-01) | 1e-5 / 2 | completed | 0.767 @150; 0.742 @500 | supported | adopt |
| exp-06 | 450 | 1.98 | other (package to final_model) | exp-05 | — | — / — | completed | — (carries exp-05's 0.742 @500) | inconclusive | adopt |
| exp-07 | 522 | 2.04 | sft | base_model | work/train_v3.jsonl (gold + RFT1 + MetaMathQA 4-type 30K) | 1e-5 / 2 | killed (2.5 min in, to add per-epoch checkpointing) | — | inconclusive | abandon_line |
| exp-08 | 548 | 2.08 | sft | base_model | work/train_v3.jsonl (gold + RFT1 + MetaMathQA 4-type 30K) | 1e-5 / 2 | completed | 0.756 @500 (1-epoch checkpoint); 0.678 @500 (2-epoch) | supported | adopt |
| exp-09 | 609 | 4.67 | other (package to final_model) | exp-08 | — | — / — | completed | 0.758 @1319 at max-connections 4; 0.780 @150 at defaults | inconclusive | adopt |
| exp-10 | 649 | 4.82 | sft | base_model | work/train_v4.jsonl (gold + RFT round 2 + forward-only MetaMath 30K) | 1e-5 / 1 | failed (CUDA OOM) | — | inconclusive | abandon_line |
| exp-11 | 673 | 4.94 | sft | base_model | work/train_v4.jsonl (gold + RFT round 2 + forward-only MetaMath 30K) | 1e-5 / 1 | completed | 0.710 @500 | contradicted | reject |
| exp-12 | 915 | 6.70 | sft | base_model | work/train_v5.jsonl (gold + RFT1 + MetaMathQA 55K, forward-heavy) | 1e-5 / 1 | completed | 0.777 @1319 at max-connections 4; 0.768 @500 | supported | adopt |
| exp-13 | 982 | 8.79 | other (package to final_model) | exp-12 | — | — / — | completed | 0.780 @150 at evaluate.py defaults | inconclusive | adopt |

Submission: exp-13 — final_model rebuilt from the exp-12 checkpoint; nothing in the stream
changes final_model after event [982]. All figures are the run's own evals; none is an
official score.
