# GSM8K post-training of `google/gemma-3-4b-pt`

Final artifact: `final_model/` (Gemma3ForCausalLM, bf16, text tower only — the vision tower
of the multimodal checkpoint is dropped since the task is text-only).

## Results (inspect-ai `inspect_evals/gsm8k`, 10-shot harness prompt)

| model | decode | samples | accuracy |
|---|---|---|---|
| `base_model` (gemma-3-4b-pt) | gen-config default (sampling) | 150 | 6.7% |
| SFT-1 (97k) | sampling | 300 | 63.0% |
| SFT-1 (97k) | greedy | 300 | 68.7% |
| SFT-2 (206k) | greedy | 1319 (full) | 70.8% |
| SFT-2 + narrow "polish" pass | greedy | 1319 (full) | 65.7% (rejected) |
| soup(SFT-1, SFT-2) | greedy | 1319 (full) | 68.5% (rejected) |
| **SFT-3 = SFT-2 + broad 2nd pass** | greedy | 1319 (full) | **71.4%** |
| SFT-3 (`final_model`) | greedy | 150 (harness default flags) | 72.0% |

## Approach

1. **Baseline** — the raw `-pt` checkpoint scores 6.7%: it copies the few-shot style badly and
   rambles past the answer.

2. **Data.** `nvidia/OpenMathInstruct-2` (12 of 32 shards, ~5.2M rows scanned) split by
   `problem_source` into GSM8K-derived (`gsm8k`, `augmented_gsm8k`, ~80k unique problems) and
   MATH-derived. Kept only integer/decimal answers, dropped LaTeX-display solutions, unwrapped
   `\boxed{...}`, and re-terminated every solution with `\n\nANSWER: <number>` — the exact format
   the harness scorer looks for (`match(location="end", numeric=True)` takes the *last* numeric
   token, so the response must stop at the answer). Added the 7,473 original GSM8K **train**
   reference solutions (calculator annotations stripped).
   All sources are GSM8K-*train*/MATH-train derived; **no test item was used**.
   `contamination_check.py` reports 0 contaminated documents for every training file
   (97,473 / 197,473 / 40,922 docs scanned).

3. **SFT-1** — full-parameter fine-tune, 97k examples, 1 epoch, lr 1e-5 cosine, effective batch 64,
   bf16 + 8-bit AdamW + Liger fused CE + gradient checkpointing, length-grouped batching.
   Prompts use the harness's own template; 12% of examples carry a 1–4-shot system prompt built
   from GSM8K *train* demos so the model keeps its own solution style instead of imitating the
   terse few-shot demos it sees at eval time. Loss is masked to completion tokens only.

4. **Decoding default (+5.7 pts).** `evaluate.py` never sets a temperature, so vLLM falls back to
   the model's `generation_config.json`. Gemma's shipped config implies sampling at T=1.0
   (top_k 64 / top_p 0.95). Setting `temperature: 0.0` / `do_sample: false` in the *model's own*
   generation config makes the eval greedy: 63.0% → 68.7%.

5. **Rejection sampling (STaR/RFT).** SFT-1 generated k=4 solutions at T=1.0 for 27,473 problems
   (all 7,473 GSM8K train + 20,000 augmented-GSM8K). 82% of problems were solved at least once;
   after answer-checking and near-duplicate removal this gave 40,922 on-policy correct solutions.

6. **SFT-2** — retrained *from the base checkpoint* on 206,395 examples
   (132k OMI-GSM + 26k OMI-MATH + 7.5k GSM8K-train references + 41k RFT), same recipe. 70.8%.

7. **SFT-3** — a further pass over a fresh, mixture-preserving 60k subset at lr 2e-6. 71.4%.
   A narrower RFT-heavy polish at 3e-6 was tried first and *hurt* badly (65.7%), so it was
   discarded; keeping the source mixture representative was what mattered.
   Weight soups (SFT-1+SFT-2 = 68.5%, SFT-2+SFT-3 = 71.5%) gave nothing over SFT-3.

## Files

- `prep_data.py` — build SFT pool from OpenMathInstruct-2 + GSM8K train
- `train_sft.py` — full-parameter SFT (`ATTN=eager|flash_attention_2`, `LIGER=0|1`)
- `gen_rft.py` — vLLM rejection sampling + answer verification
- `merge_data.py`, `soup.py`, `finalize.py` — dataset merge, weight averaging, artifact packaging
- `work/` — datasets (`sft1/sft_big2/rft/sft2/ep2.jsonl`) and checkpoints
- `logs/` — training logs, eval logs, contamination-check output
