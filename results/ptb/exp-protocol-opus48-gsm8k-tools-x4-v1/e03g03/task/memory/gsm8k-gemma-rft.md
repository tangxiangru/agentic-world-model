---
name: gsm8k-gemma-rft
description: What worked to post-train gemma-3-4b-pt on GSM8K (SFT + iterative RFT), and the traps
metadata:
  type: project
---

Task: maximize GSM8K accuracy of google/gemma-3-4b-pt (snapshot cc012e0…). Eval = evaluate.py (inspect_evals/gsm8k), 10-shot, template templates/gemma3.jinja, scorer match(numeric=True) which reads the **last number** in the output. Model must emit `ANSWER: N` then stop at `<end_of_turn>` (id 106; eos list [1,106]).

Results (n=150 unless noted): base 0.04 → SFT GSM8K-train 0.393 (exp-02) → RFT round1 0.44 (exp-06) → RFT round2 **0.493 (exp-08, SUBMITTED as final_model)** ; final_model n=300 = 0.443.

**What worked:** zero-shot SFT on GSM8K-train reformatted (strip `<<>>`, replace `#### N` with `ANSWER: N`, append `<end_of_turn>`), full FT (vision tower frozen) via awm SaveSafeTrainer, lr 1e-5, bf16, greedy generation_config. Then iterative RFT/STaR: sample 6/question from the current model (vLLM, temp 1.0), keep last-number==gold, retrain from base on gold + self-solutions. Coverage rose 0.857→0.894→0.916 across rounds; each RFT round added ~+5 pts.

**Traps that cost runs (see [[stopping-breakage-overtraining]], [[vllm-offline-gotchas]]):**
- Adding MetaMathQA did NOT improve reasoning (first-answer stayed 0.40) and broke stopping → 0.12 (exp-03).
- Over-training on larger merged data breaks EOT stopping: model keeps generating new problems after the first ANSWER, so the last-number scorer reads a wrong later number. exp-10 (3-round, 2 epochs, 4237 steps) had the BEST reasoning (first-answer 0.513) but stopping broke → 0.313. exp-12 (1 epoch) underfit → 0.373. The reasoning↔stopping trade-off was unsolved in-budget; the sweet spot was ~3500-4000 steps (exp-06/exp-08).

**Unfinished lever:** exp-10's 0.513 first-answer reasoning is real; a stopping fix (e.g. train on few-shot-format targets matching the eval, or interleaved EOT-robustness data, or a small EOT-upweight) could unlock ~0.51+.
