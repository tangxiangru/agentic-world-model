# google/gemma-3-4b-pt → GSM8K

`final_model/` = `ckpts/exp-05/final` (byte-identical weights; verified loadable
from a fresh CPU process and served by `evaluate.py`).

| what | dev-150 | full 1319 |
|---|---:|---:|
| base `gemma-3-4b-pt` (exp-01) | 0.033 | — |
| exp-02 SFT (2 ep, OpenMathInstruct-2) | 0.753 | 0.699 |
| exp-03 same weights, greedy decode | 0.727 | — |
| exp-04 + RFT round 1 (gsm8k train) | 0.747 | 0.729 |
| **exp-05 + RFT round 2 (unseen problems) — shipped** | **0.740** | **0.747** |

Protocol: `evaluate.py --limit 150 --max-tokens 4000 --max-connections 32
--gpu-memory-utilization 0.85` for dev-150; `--limit -1 --max-connections 16`
for the full split. Decoding is whatever the checkpoint's `generation_config.json`
says, because `evaluate.py` passes no decode parameters — that file is unchanged
from the base (top_k 64, top_p 0.95, temperature → 1.0), so every number is a
sampling draw with a ~3.5 pp noise floor at n=150 and ~1.2 pp at n=1319.

## What the run did

1. **exp-01 — baseline.** 0.033. The pt checkpoint's problem is not arithmetic:
   112/150 completions never produce a parseable `ANSWER: <n>` line and 81/150
   run to the 4000-token cap, degenerating into repetition or another language.
   The grader (`match(location="end", numeric=True)`) reads the *last number
   anywhere in the completion*, so a model that does not stop is scored on the
   wrong number.

2. **exp-02 — SFT.** 72k rows: 60k gsm8k/augmented_gsm8k + 12k math/augmented_math
   from OpenMathInstruct-2 `train_1M`. `\boxed{}` unwrapped, trailing answer
   restatements removed, exactly one `ANSWER: <n>` line appended, rows whose last
   number ≠ gold dropped. Prompts rendered with the grader's own
   `templates/gemma3.jinja`; 20% carry a random 1–10 shot prefix from the gsm8k
   **train** split in inspect's exact few-shot rendering. Full fine-tune of the
   3.88B text tower (vision tower frozen), 2 epochs, lr 2e-5, bf16, completion-only
   loss, liger fused CE. 0.033 → 0.753, no-ANSWER-line 0.747 → 0.020.

3. **exp-03 — decode config.** Greedy (temperature 0.0 written into a symlinked
   copy) scored 0.727 vs 0.753 sampled; paired, 15 greedy-only wins vs 19
   sampling-only out of 34 discordant items. Indistinguishable. Rejected; the
   shipped model keeps its inherited config.

4. **exp-04 — RFT round 1.** 6 samples per gsm8k **train** question from
   exp-02/final; keep only samples whose graded last number equals gold →
   20,013 rows over 7,078 questions, mixed 50/50 with sft_v1 replay, 2 epochs at
   lr 1e-5. dev-150 said −0.7 (15 vs 16 flips); the full split said
   **0.699 → 0.729, 165 wrong→right vs 125 right→wrong, McNemar z = 2.35**.

5. **exp-05 — RFT round 2 on unseen problems.** 24,000 gsm8k-style problems whose
   question hash does not appear in sft_v1, 4 samples each from exp-04/final,
   verified against `expected_answer` → 41,047 rows; 32k of them + 8k replay,
   1 epoch at lr 7e-6. **0.729 → 0.747, 143 wrong→right vs 120 right→wrong
   (z = 1.42)**; no-ANSWER-line 10/1319 → 5/1319.

## Contamination

Every training file was checked with `../contamination_check.py` against
`../test_data.json`: `sft_v1.jsonl` (72,000), `rft_v1.jsonl` (20,013),
`rft_mix_v1.jsonl` (40,013), `rft_mix_v2.jsonl` (40,000) — **0 contaminated
documents in all four**. No item from the test split was read for anything other
than that checker. Every question used for training came from the gsm8k *train*
split or from OpenMathInstruct-2, which is itself derived from the GSM8K/MATH
*train* splits.

## Things that cost time, recorded for the next run

* The 262k-token vocabulary makes the cross-entropy logits the memory bottleneck:
  bs=16 at seq 2048 wanted 22.45 GiB for logits alone. `use_liger_kernel=True`
  fixed it (peak 44.7 GB).
* `max_seq_len` must clear the *grader's* prompt, not the training rows: inspect's
  10-shot system message alone is 2043 tokens, so the real eval prompt is ~2215.
  At 2048 the trainer silently dropped 3.4% of rows — exactly the long few-shot
  ones. Raised to 3072.
* `Trainer` writes intermediate checkpoints without tokenizer files; vLLM then
  refuses them with `Either model_file or model_proto must be specified`
  (`fix_ckpt.py`).
* A corrupted `.pyc` under `gguf/__pycache__/` made every vLLM launch die with
  `ValueError: bad marshal data`. Deleting the directory fixed it.
* dev-150 could not see either RFT round (−0.7 and, for exp-05, not measured);
  both were only visible on the full 1319-item split. At n=150 the noise floor
  is larger than the effect.

Experiment cards: `memory/cards/exp-0{1..5}.yaml`, index in `memory/index.md`,
packaging rule fixed in advance in `SHIPPING_RULE.md`.
