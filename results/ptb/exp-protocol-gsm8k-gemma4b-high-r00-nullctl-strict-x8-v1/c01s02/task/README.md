# Post-training `google/gemma-3-4b-pt` for GSM8K

Final model: `final_model/` (bf16, `Gemma3ForConditionalGeneration`, greedy decoding).

## Results (`evaluate.py`, inspect-ai `gsm8k`, 10-shot prompt, greedy)

| model | acc @150 (default args) | acc @500 |
|---|---|---|
| base `gemma-3-4b-pt` | 7.3% | – |
| SFT stage 1 | 76.5% (@200) | – |
| SFT stage 2 (+ RFT data) | 79.0% (@200) | 78.2% |
| **+ GRPO (final_model)** | **80.0%** | **79.8%** |
| + 2nd GRPO round (rejected) | – | 78.8% |

The base number is low mainly because the pretrained model's `generation_config.json`
enables sampling (T=1, top_p=0.95, top_k=64) and the un-tuned model degenerates into
repeating the prompt. vLLM picks up the model's generation config, so `final_model`
ships a greedy one (`temperature=1e-6, top_k=1`, valid for both HF and vLLM).

## Pipeline

1. **`prep_data.py`** — SFT set from the GSM8K-derived slice of
   `nvidia/OpenMathInstruct-2` (`gsm8k` + `augmented_gsm8k`, all derived from the
   GSM8K *train* split): 152,888 examples. `\boxed{}` stripped, reformatted to the
   exact eval prompt/response contract (`... ANSWER: <n>` as the last line, since the
   scorer is `match(numeric=True, location="end")`).
2. **`train_sft.py`** — full-parameter SFT (vision tower frozen), fp32 master weights
   + bf16 autocast, 8-bit Adam, gradient checkpointing, Liger fused linear CE (the
   262k vocab makes an unfused CE OOM), `group_by_length`, eff. batch 128, lr 1e-5
   cosine, 1 epoch. 12% of examples get a random k-shot prefix so the model stays
   robust to the eval's 10-shot system message.
3. **`gen_rft.py`** — rejection sampling from the stage-1 model on the 7,473 GSM8K
   *train* questions (16 samples @ T=1.0), keeping only solutions whose final number
   matches the gold answer → 28,772 on-policy verified traces (79.8% sample pass rate,
   98.3% of problems solved at least once).
4. **`prep_mix.py` + `train_sft.py`** — stage 2 SFT from the *base* checkpoint on
   155k OpenMathInstruct-2 + RFT data at 2× weight (3× for low-pass-rate problems),
   202,658 examples.
5. **`train_grpo.py`** — GRPO with a binary correctness reward on GSM8K train
   questions (LoRA r=32, β=0, 8 samples/prompt, 90 steps), merged via `merge_lora.py`.
   A second GRPO round did not help and was discarded.

## Decontamination

Both training sets were checked with `../contamination_check.py` against
`../test_data.json`: **0 contaminated documents out of 152,888 (OpenMathInstruct-2)
and 0 out of 28,772 (RFT)**. All training questions come from the GSM8K *train*
split or from OpenMathInstruct-2 augmentations of it; the test set was never used
for training or for seeding generation.

## Things that did not help

- `repetition_penalty=1.05` at decode time: 77.5% vs 79.0% @200. ~2.5% of samples
  still lose to greedy repetition loops; the penalty cost more than it recovered.
- A second GRPO round (150 steps, harder prompt filter): 78.8% vs 79.8% @500.

## Reproduce

```
python prep_data.py --out data/sft_gsm.jsonl
python train_sft.py --data data/sft_gsm.jsonl --out runs/sft1 --bs 64 --accum 2 --lr 1e-5
python make_final.py --src runs/sft1 --dst models/sft1
python gen_rft.py --model models/sft1 --out data/rft1.jsonl --stats-out data/rft1_stats.jsonl --n 16
python prep_mix.py
python train_sft.py --data data/mix2.jsonl --out runs/sft2 --bs 64 --accum 2 --lr 1e-5
python make_final.py --src runs/sft2 --dst models/sft2
python train_grpo.py --model models/sft2 --out runs/grpo1 --max-steps 90
python merge_lora.py --base models/sft2 --adapter runs/grpo1 --dst final_model
```

Note: `train_sft.py` needs `liger-kernel` (`uv pip install --system liger-kernel`);
`final_model` itself has no extra dependencies. Keep `--max-len 1024` at `--bs 64`
— 1280 OOMs on an 80 GB H100.
