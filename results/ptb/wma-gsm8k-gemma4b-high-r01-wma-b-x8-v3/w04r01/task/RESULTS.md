# GSM8K post-training of `google/gemma-3-4b-pt` — results

Shipped artifact: **`final_model/`** (exp-04 = exp-03 weights + greedy `generation_config.json`).

| what | n=150 (protocol) | n=1319 (full test) |
|---|---:|---:|
| base `gemma-3-4b-pt` @ cc012e0a | 0.0867 ± 0.023 | — |
| exp-02 SFT (zero-shot prompts only) | 0.0200 | — |
| exp-03 + k-shot prompts, sampled decode | 0.6200 ± 0.040 | — |
| **exp-04 = exp-03 + greedy decode → `final_model`** | **0.6800 ± 0.038** | **0.6664 ± 0.0130** |
| exp-05 RFT on self-samples | 0.6733 ± 0.038 | 0.6543 ± 0.0131 |
| exp-07 soup(exp-03, exp-05) | — | 0.6664 ± 0.0130 |

## Recipe

1. **exp-02 — SFT from base**, 84 946 rows, 2 epochs, lr 1e-5, bf16, eff. batch 32,
   `max_seq_len` 1536, completion-only loss, 2.63 h.
   Data: `openai/gsm8k` train (7 473, ×2, `<<calc>>` and `#### N` stripped) plus 70 000
   `nvidia/OpenMathInstruct-2` `train_1M` rows with `problem_source ∈ {gsm8k,
   augmented_gsm8k}` (`\boxed{x}` unboxed in place, ≤2 solutions per problem).
   Every target is `CoT + "\n\nANSWER: <number>" + <end_of_turn>` rendered through the
   grader's own `templates/gemma3.jinja`.
2. **exp-03 — continue on k-shot prompts**, 15 000 rows, 1 epoch, lr 7e-6,
   `max_seq_len` 3456, 47 min. Each prompt carries k ∈ {0,3,6,10} shots
   (weights .30/.10/.15/.45) rendered byte-identically to
   `inspect_evals.gsm8k.sample_to_fewshot`. **This is the step that matters: 0.02 → 0.62.**
3. **exp-04 — greedy decode**: `temperature: 0.0` in `generation_config.json`,
   `top_k`/`top_p` removed, `eos_token_id [1, 106]` kept. 0.62 → 0.68.

## Why exp-02 scored below the base model

The grader always prepends a 10-shot system prefix. A model fine-tuned only on zero-shot
prompts answers correctly and then *continues the few-shot pattern* — inventing the next
question — instead of emitting `<end_of_turn>`; `match(numeric=True, location="end")` then
reads a number out of the invented continuation. Same weights, temperature 0, 200 gsm8k
train items: zero-shot 0.76 accuracy / 0.5 % bad format, ten-shot 0.39 / 79.5 % bad format
(`analysis/probe_exp02_final.json`). After exp-03 the ten-shot gap is 3 points and format
failures on the graded 150 fell from 127 to 1; after exp-04, to 0.

## Negative results

* **RFT** (exp-05): 22 034 answer-verified self-generated solutions + 3 000 replay rows,
  1 epoch from exp-03. −1.21 pp on the full test. pass@4 was 0.90 against 0.70 single-sample,
  so the headroom exists, but sharpening on the model's own correct modes did not capture it.
* **Weight soup** (exp-07) of exp-03 and exp-05: 0.6664 — exactly tied, 801 items correct in
  both, 78 only in each. No gain.

## Reproducing / files

* `scripts/fmt.py` — the single rendering path; `scripts/verify_template.py` asserts it equals
  `tokenizer.apply_chat_template(templates/gemma3.jinja)` and that `inspect_evals`'
  `MATH_PROMPT_TEMPLATE` has not drifted (template sha `7de1c58e208e`).
* `scripts/build_sft_data.py`, `scripts/build_fewshot_data.py`, `scripts/rft_sample.py`,
  `scripts/train_sft.py`, `scripts/package_model.py`, `scripts/soup.py`,
  `scripts/analyze_eval.py`, `scripts/probe_fewshot.py`.
* Experiment cards: `memory/cards/exp-01..07.yaml`, index in `memory/index.md`.
* Every training file passed `../contamination_check.py` against `../test_data.json` with
  0 contaminated documents (84 946 / 15 000 / 25 034 / 16 091 rows).

## Environment notes

* `liger-kernel==0.8.2` was installed for training only (gemma-3's 262k vocab OOMs the fp32
  cross-entropy at batch 16 × 1536). `final_model` is a plain bf16
  `Gemma3ForConditionalGeneration` checkpoint with tokenizer and processor files and needs
  nothing beyond the starting environment; it was loaded and generated from a fresh CPU
  process as a final check, and read twice by `evaluate.py` under vLLM.
