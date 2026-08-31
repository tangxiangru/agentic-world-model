# r-ad144811 - extracted experiment cards

Base model: Qwen/Qwen3-4B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
All accuracies are the run's own `evaluate.py` numbers. "150" is the evaluator's
fixed 150-example slice; "full" is all 1,319 test examples. Concurrency matters:
the run establishes late that BF16 vLLM greedy decoding varies with batch
composition, so 150-example scores taken at `--max-connections 16` (exp-01
through exp-12) are not strictly comparable with those at the evaluator's default
`--max-connections 2` (exp-13 onward).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 79 | 0.23 | sft | base_model | train_data (122,419: GSM8K-train x3 + MetaMath GSM-provenance) | 2e-5 / 1 | completed | 0.7533 (150, sft_v1_eos_150.json); 0.7331 (full) | supported | adopt |
| exp-02 | 252 | 1.56 | sft | exp-01 | official_data (7,473 GSM8K train) | 5e-6 / 3 | completed | 0.7333 (150, spec_v1_150.json) | contradicted | reject |
| exp-03 | 300 | 1.79 | sft | exp-01 | exact_context_data (3,000, evaluator's exact 10-shot system context) | 3e-6 / 1 | completed | 0.7000 (150, exact_v1_150.json) | contradicted | reject |
| exp-04 | 379 | 2.11 | sft | exp-01 | novel_train_data (53,473 prompt-disjoint augmentations + anchors) | 1e-5 / 1 | completed | 0.7267 (150, novel_v1_150.json) | contradicted | reject |
| exp-05 | 494 | 2.76 | sft | exp-01 | train_data (122,419) | 1e-5 / 1 | killed | none | inconclusive | abandon_line |
| exp-06 | 503 | 2.83 | sft | exp-01 | train_data (122,419), reshuffled (seed 2027) | 1e-5 / 1 | completed | 0.7867 (150, step-400, conn 16); endpoint 0.7400; 0.7642 (full, step-400) | contradicted | adopt |
| exp-07 | 540 | 3.25 | other (preserve step-200 + tokenizer/EOS normalization) | exp-06 | none | n/a | completed | 0.8200 (150, sft_v2_step200_150.json); 0.7612 (full) | supported | adopt |
| exp-08 | 586 | 4.12 | sft (LoRA r=64) | base_model | train_data (122,419) | 1e-4 / 1 | completed | 0.0400 (150, lora_sft_v1_150.json) | contradicted | reject |
| exp-09 | 622 | 4.26 | merge (alpha 0.2 toward step 400) | exp-07 | none | n/a | completed | 0.7200 (150, soup_post200_a20_150.json) | contradicted | reject |
| exp-10 | 786 | 5.74 | grpo (LoRA r=32) | exp-07 | openai/gsm8k train (7,473), 50 steps | 1e-6 / 50 steps | completed | 0.7267 (150, grpo_conservative50_150.json) | contradicted | reject |
| exp-11 | 801 | 5.99 | sft | exp-07 | alternate_train_data (41,698 alternate verified rationales) | 2e-6 / 1 | completed | 0.7333 (150, alt_rationale_v1_150.json) | contradicted | reject |
| exp-12 | 841 | 6.55 | sft (curve reconstruction, stopped at step 151) | exp-01 | train_data (122,419), seed 2027 | 1e-5 / 1 | killed | 0.7600 (150, curve_step50_eval.json); 0.7533 / 0.7200 at steps 100 / 150 | contradicted | reject |
| exp-13 | 900 | 6.99 | other (package final_model) | exp-07 | none | n/a | completed | 0.7733 (150, final_model_eval150.json, conn 2); 0.7612 (full) | inconclusive | adopt |
| exp-14 | 1012 | 7.47 | sft | exp-07 | official_data (7,473 GSM8K train) | 1e-6 / 1 | completed | 0.7733 (150, official_refine20_eval.json); 0.7333 at step 30 | contradicted | reject |
| exp-15 | 1050 | 7.76 | rft (self, stopped at step 27) | exp-07 | self_rft_data (14,510 rejection-sampled self rationales) | 1e-6 / 1 | killed | 0.7400 (150, self_rft20_eval.json) | contradicted | reject |

Submitted: **exp-13** (`final_model`), byte-identical to exp-07's step-200
checkpoint. Comparator throughout: base model 0.4733 on the 150-example slice
(baseline_150.json).

Not cards: the two SFT pipeline smoke runs ([70], [73], on exp-01) and the two
GRPO smoke runs ([761] crashed on FP32 + FlashAttention, [773] passed at 5 steps,
both on exp-10). `soup_pre200_a50` was built and scored (0.7533) but its
`average_models.py` invocation does not appear in the digest, so it has no card
(noted in exp-09).
