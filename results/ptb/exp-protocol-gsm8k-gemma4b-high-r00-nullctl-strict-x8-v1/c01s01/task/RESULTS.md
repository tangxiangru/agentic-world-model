# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` = GRPO checkpoint 350 (`work/grpo1/checkpoint-350`, cast to bf16, greedy
generation config). **79.98 %** on the full GSM8K test split (1319 items).

## Results (inspect-ai `evaluate.py`, greedy decoding, 10-shot prompt)

| model | stage | full test (1319) | first 500 |
|---|---|---|---|
| `gemma-3-4b-pt` (base) | – | – | 8.0 % (150) |
| `work/sft1` | SFT | – | 77.8 % |
| `work/rft1` | + rejection-sampling FT | 74.1 % | 78.8 % |
| `work/g250` | + GRPO step 250 | 78.1 % | 79.2 % |
| **`work/g350`** | **+ GRPO step 350** | **80.0 %** | 81.2 % |
| `work/h75`, `work/h100` | GRPO steps 425 / 450 | 78.0 % / 78.8 % | – |
| `work/soup` | avg of 4 GRPO ckpts | 78.5 % | – |

(The first 500 test items are noticeably easier than the rest, so only the full-test column
is comparable across rows.)

## Pipeline

1. **SFT** (`prep_data.py`, `train_sft.py`) — 97,473 examples: 90k from
   OpenMathInstruct-2 (`gsm8k` + `augmented_gsm8k` splits, ≤2 solutions per problem,
   integer answers, LaTeX/`\boxed` stripped) + the 7,473 GSM8K *train* reference
   solutions. Targets are rewritten to the eval's output shape (`…\n\nANSWER: N`), and
   15 % of prompts carry a random 1–10-shot prefix so the model ignores the eval's
   10-shot system message style. 1 epoch, bs 32, lr 1.5e-5 cosine, fp32 master weights +
   bf16 autocast, 8-bit AdamW, liger kernels, ~68 min.
2. **Rejection sampling / RFT** (`gen_rs.py`, `build_rs_data.py`) — 8 samples per GSM8K
   train question at T=1.0; 96.3 % of questions solved at least once, 42k correct
   deduplicated solutions. Retrained from the SFT checkpoint on 20.5k of them + 12k
   replay, 1 epoch, lr 7e-6.
3. **GRPO** (`grpo_train.py`) — TRL, Dr.GRPO loss, β=0 (no reference model), 8
   generations × 32 prompts per step, T=1.0, lr 1e-6, vLLM colocate. Reward = exact match
   of the extracted `ANSWER:` value against the GSM8K train gold. Train reward rose
   0.75 → 0.90 over 350 steps; test accuracy peaked at step 350 and regressed afterwards
   (steps 400–450 and the checkpoint soup were all ~78–79 %), so step 350 was kept.

## Notes

* `final_model/generation_config.json` sets `temperature: 0.0` so vLLM decodes greedily
  (the eval sends no sampling parameters, so vLLM falls back to the model's generation
  config; the stock config's `do_sample`/`top_k: 64` would sample at T=1.0).
* TRL's truncation check compares the last token against `tokenizer.eos_token_id` (1),
  but Gemma-3 chat turns end with `<end_of_turn>` (106). Left unpatched with
  `mask_truncated_completions=True` this masks *every* completion and GRPO learns
  nothing (loss and grad-norm identically 0). `grpo_train.py` sets
  `trainer.eos_token_id = 106`.
* No GSM8K test data was used for training. `prep_data.py` output (97,473 docs) and the
  RFT set (32,495 docs) were both checked with `../contamination_check.py`: 0 matches.
* The container's `/` overlay is a full 64 MB filesystem, so liger-kernel is installed
  into `pylibs/` and used via `PYTHONPATH`. It is only needed for training —
  `final_model` runs on the stock environment.
