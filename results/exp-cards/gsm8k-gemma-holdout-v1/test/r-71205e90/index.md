# r-71205e90 — reconstructed experiment cards

- base model: google/gemma-3-4b-pt
- benchmark: gsm8k, 10 h budget, 1x H100 80GB
- cards: 19 (launches [15027] .. [41754]); the digest carries timestamps, so every `elapsed_h` is real
- no smoke runs: every crashed or killed launch in this run was a full-size run the agent meant as real

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 15027 | 0.20 | sft | base_model | data/sft_v1.jsonl (7473) | 2e-5 / 3 | failed | none (OOM in cross-entropy) | inconclusive | abandon_line |
| exp-02 | 15718 | 0.25 | sft | base_model | data/sft_v1.jsonl (7473) | 2e-5 / 3 | killed | none | inconclusive | abandon_line |
| exp-03 | 17355 | 0.29 | sft | base_model | data/sft_v1.jsonl (7473) | 2e-5 / 3 | killed | none | inconclusive | abandon_line |
| exp-04 | 17974 | 0.33 | sft | base_model | data/sft_v1.jsonl (7473) | 2e-5 / 3 | completed | 0.5867 @150 (results_v1e3.json, +0.5133 vs base) | supported | adopt |
| exp-05 | 25368 | 1.48 | sft | base_model | data/sft_v2.jsonl (58558) = rs_v1 (18435) + gsm8k originals + metamath_sft (30000) | 2e-5 / 2 | completed | 0.6733 @150 (results_v2e2.json, +0.0867 vs exp-04), 0.684 @500, 0.6755 @1319 | supported | adopt |
| exp-06 | 26333 | 1.58 | other (packaging to final_model) | exp-04 | none | n/a | completed | 0.5867 @150 (results_v1e3.json) | inconclusive | adopt |
| exp-07 | 28658 | 4.27 | other (packaging to final_model) | exp-05 | none | n/a | completed | 0.684 @500 (results_v2e2_500.json), 0.6755 @1319 | inconclusive | adopt |
| exp-08 | 29356 | 4.59 | sft | exp-05 | data/sft_v3.jsonl (rs_v2 21039 deduped + hard originals + fresh MetaMath) | 1e-5 / 1 | completed | 0.684 @500 (results_v3_500.json, +0.000 vs exp-05), 0.7067 @150, 0.6801 @1319 | inconclusive | adopt |
| exp-09 | 30311 | 5.32 | merge | exp-08 | none (50/50 soup with exp-05) | n/a | completed | 0.672 @500 (results_soup_500.json, -0.012) | inconclusive | reject |
| exp-10 | 31013 | 5.38 | other (packaging to final_model) | exp-08 | none | n/a | completed | 0.6801 @1319 (results_v3_full.json) | inconclusive | adopt |
| exp-11 | 31015 | 5.38 | sft | exp-08 | data/sft_v3.jsonl (second pass) | 7e-6 / 1 | completed | 0.684 @500 (results_v3b_500.json, +0.000) | inconclusive | reject |
| exp-12 | 35794 | 6.09 | grpo | exp-08 (via the text-only extraction ckpts/v3_text) | gsm8k train prompts, hard-weighted | 1e-6 / 48 steps | failed | none (liger-kernel import: null bytes) | inconclusive | abandon_line |
| exp-13 | 37208 | 6.25 | grpo | exp-08 (via ckpts/v3_text) | gsm8k train prompts, hard-weighted | 1e-6 / 48 steps | failed | none (crash on save: greedy generation_config) | inconclusive | abandon_line |
| exp-14 | 37704 | 6.34 | grpo | exp-08 (via ckpts/v3_text) | gsm8k train prompts, hard-weighted (2913) | 1e-6 / 96 steps | completed | 0.672 @500 (results_grpo96_500.json, -0.012 vs exp-08), 0.72 @150 | inconclusive | adopt |
| exp-15 | 38872 | 6.65 | grpo | exp-14 | gsm8k train prompts, hard-weighted (2913) | 2e-6 / 96 steps | completed | 0.6876 @1319 (results_grpo2_full.json, +0.0076 vs exp-08), 0.684 @500 | inconclusive | adopt |
| exp-16 | 40242 | 7.01 | merge | exp-15 | none (graft into the exp-08 multimodal shell) | n/a | completed | none (verified via exp-17) | inconclusive | adopt |
| exp-17 | 40247 | 7.02 | other (packaging to final_model) | exp-16 | none | n/a | completed | 0.6831 and 0.6869 @1319 (results_final_full.json, results_final_full2.json) | inconclusive | adopt |
| exp-18 | 41204 | 7.08 | grpo | exp-15 | gsm8k train prompts, hard-weighted (2913) | 2e-6 / 96 steps | completed | 0.6823 @1319 (results_grpo3_full.json, -0.0053) | inconclusive | reject |
| exp-19 | 41754 | 7.38 | merge | exp-15 | none (50/50 soup with exp-18) | n/a | completed | 0.6823 @1319 (results_soup_rl_full.json, -0.0053) | inconclusive | reject |

Notes

- **Submitted card: exp-17** — `final_model` holds the GRPO round-2 language
  weights grafted back into the base model's full multimodal architecture
  (exp-15 -> exp-16 -> exp-17), verified twice from that exact path at
  `--limit 1319`: 68.31% and 68.69%. No later launch overwrote it.
- exp-01 to exp-03 are three crashed/killed attempts at the same SFT v1
  training that exp-04 finally ran: an OOM in the 262K-vocabulary
  cross-entropy, a Liger-kernel retry that stops appearing, and a
  bf16 + fused-Adam configuration killed for FlashAttention-2. None was a
  deliberately truncated smoke test, so each is a card, and none produced a
  checkpoint.
- `adopt` on exp-04, exp-05, exp-08, exp-14 and exp-15 means the output became
  the incumbent `final_model` or the parent of a later card, not that it was the
  submission.
- Nearly every verdict is `inconclusive` because the run plateaued: from exp-05
  onwards every candidate sits at 68-69% on the full test, inside the +-1.28 pt
  standard error, and the agent said so itself ("Top candidates are
  statistically tied", [41753]). The two `supported` verdicts are the first two
  SFT rounds, which moved the metric by 51.3 and 8.7 points under an identical
  protocol.
- `prep_ckpt.py` invocations, which cast a checkpoint to bf16, copy the
  tokenizer/processor files and write the greedy `generation_config.json`
  (`do_sample=false`, `temperature 0.0`, `eos_token_id [1,106]`), are recorded
  on the training card that produced the weights rather than as separate
  decode-config cards: the same transformation is applied to every checkpoint as
  part of the eval path, and no non-greedy variant of a fine-tuned checkpoint
  was ever measured. The agent's own summary calls this decode config the
  single biggest lever in the run.
- Run-level gap: the text-only causal-LM extraction of the v3 checkpoint
  (`ckpts/v3_text`), which is the parent of exp-12 to exp-14 and the model
  behind the 0.7067 @150 comparator in `results_v3text_150.json`, has no card —
  the digest jumps from [31959] (turn 116) to [34194] (turn 117), so its launch
  command is not in the stream.
