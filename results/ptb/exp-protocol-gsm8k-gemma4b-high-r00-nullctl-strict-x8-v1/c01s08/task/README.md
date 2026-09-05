# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` is a full-parameter fine-tune of the frozen snapshot
`cc012e0a6d0787b4adcc0fa2c4da74402494554d` (vision tower + projector kept frozen,
language model trained).

## Results (`evaluate.py`)

| model | limit 150 | limit 500 |
|---|---|---|
| base `gemma-3-4b-pt` | 6.0% | – |
| round 1, 0.5 epoch | 70.0% | – |
| round 1, 1 epoch | 73.3% / 74.7% (greedy) | – |
| round 1, 2 epochs | 75.3% | – |
| round 2, 0.5 epoch | 76.7% | 75.6% |
| round 2, 1 epoch | 75.3% | 76.8% |
| round 3 (k-shot adaptation) | – | 77.8% |
| round 4 | – | 78.8% |
| **round 5 = `final_model`** | **77.3%** | **79.0%** |

The base model scores 6% only because it never emits `<end_of_turn>`: it keeps
generating past the answer until `max_tokens`, so the last number in the
completion (what `match(numeric=True)` scores) is garbage.

## Approach

**Format.** Training targets reproduce the exact string vLLM builds from
`templates/gemma3.jinja` for the `inspect_evals/gsm8k` task:

```
<bos><start_of_turn>user\n[k-shot prefix\n\n]<MATH_PROMPT_TEMPLATE><end_of_turn>\n<start_of_turn>model\n<solution>\n\nANSWER: <n><end_of_turn>
```

Loss is on the completion only. `<end_of_turn>` (id 106) is already in the base
`eos_token_id`, so the fine-tune fixes the truncation failure above.

**Data** (all from GSM8K *train* and OpenMathInstruct-2, never the test split;
every file passed `contamination_check.py` with 0 matches):

- GSM8K train ground-truth solutions, `<<...>>` calculator annotations stripped.
- `nvidia/OpenMathInstruct-2`, `gsm8k` + `augmented_gsm8k` slices (full 14M
  release), `\boxed{}` unwrapped, numeric answers only, capped per problem.
- Rejection-sampled on-policy solutions: 4 samples at T=1.0 for 7,473 GSM8K-train
  and 12,000 augmented problems with the round-1 model, keeping only completions
  whose final `ANSWER:` matches the reference (40,741 kept; pass@1 ≈ 51%),
  deduplicated by reasoning path.
- A fraction of examples carry a random k-shot prefix built from GSM8K train in
  inspect's own few-shot format, since the benchmark always prompts 10-shot.

**Schedule.**

1. Round 1 — 89k examples, 2 epochs, lr 2e-5, from base.
2. Round 2 — 158k examples (full OMI-2 GSM slice + GT×2 + 28k rejection-sampled),
   1 epoch, lr 1e-5, from round 1.
3. Rounds 3–5 — 26k examples each with **60% k-shot prefixes**, lr 5e-6 → 4e-6.
   The model gains ~22 points from the 10-shot prefix (51% 0-shot pass@1 vs 73%
   10-shot), so matching the eval's prompt distribution is worth optimising for;
   these three short rounds added ~2 points.

**Decoding.** `final_model/generation_config.json` sets `top_k: 1`, which vLLM
picks up as its default sampling params, so the benchmark decodes greedily
(inspect does not send a temperature). Worth ~1 point over sampling at T=1.0.

**Training details.** bf16, Liger fused linear cross-entropy (avoids
materialising the 262k-vocab logits), gradient checkpointing, `adamw_bnb_8bit`,
cosine schedule, and a token-budget batch sampler (45k padded tokens per
micro-batch, length-bucketed) that runs ~2x faster than fixed-size batching.

## Files

- `build_data.py` / `build_data2.py` — dataset construction.
- `gen_rft.py` — rejection sampling with vLLM.
- `train_sft.py` — training.
- `finalize.py` — copies processor configs into a checkpoint, sets greedy decoding.
- `runs/` — all logs and metric JSONs.
