# gemma-3-4b-pt → GSM8K

`final_model/` holds the shipped checkpoint. Full experiment record: `memory/index.md`
and `memory/cards/exp-01..08.yaml`.

## Result

| read | base `gemma-3-4b-pt` | `final_model/` |
|---|---:|---:|
| gsm8k test, all 1319 items | – | **0.7346 ± 0.0122** |
| gsm8k test, first 500 | – | 0.7400 ± 0.0196 |
| gsm8k test, first 150 (harness default) | 0.0533 ± 0.0184 | 0.7733 ± 0.0343 |

The full-split number is the honest one; the 150-item read that the harness defaults to
is optimistic by ~4 pts for this model, well inside its own 3.4 pp stderr.

## What the model is

One full fine-tune of `google/gemma-3-4b-pt` (snapshot `cc012e0a…`), one epoch, plus a
decode-config change. No adapters — `final_model/` is bf16 weights with the tokenizer and
processor configs beside them, loadable by vLLM from a fresh process.

**Corpus** (`data/sft_v2.jsonl`, 113 499 rows, contamination checker: 0 hits)

| slice | rows | source |
|---|---:|---|
| OMI2 gsm8k-derived | 70 747 | `nvidia/OpenMathInstruct-2`, `problem_source ∈ {gsm8k, augmented_gsm8k}`, 1 solution per problem — every distinct gsm8k-derived problem OMI2 has |
| OMI2 math | 8 000 | same dataset, `{math, augmented_math}` |
| gsm8k train | 7 473 | `openai/gsm8k` main/train, calculator spans stripped |
| rejection-sampled | 27 279 | self-generated: 8 samples/question at T=1 from the exp-03 checkpoint over the gsm8k **train** questions, kept only where the final number matches gold |

Every target is normalised to the shape the grader reads: chain of thought, then one
`ANSWER: <n>` line, then `<end_of_turn>`. 10% of rows carry a 2/4/8/10-shot prefix
rendered the way the grader renders its own shots.

**Training** — `train_sft.py`: completion-only loss, fp32 master weights with bf16
autocast, `adamw_bnb_8bit`, lr 1e-5 cosine, effective batch 32, `max_seq_len` 2688,
gradient checkpointing, liger fused linear cross-entropy (gemma3's 262 k vocab makes the
materialised logit tensor 18 GB and it OOMs without this). 1.9 h on one H100.

**Decode** — `final_model/generation_config.json` sets `temperature 0` and clears
`top_k`/`top_p`. `evaluate.py` passes no temperature, so vLLM takes these from the
checkpoint; the stock gemma defaults (`do_sample`, `top_k 64`, `top_p 0.95`) would make
the benchmark a sampled read. Greedy was worth +8.0 pts on identical weights (exp-03).

## What moved the number

| card | change | dev-150 greedy |
|---|---|---|
| exp-01 | base model | 0.053 |
| exp-02 | SFT, 57 k rows, targets ending in one `ANSWER:` line + `<end_of_turn>` | 0.640 *(sampled)* |
| exp-03 | greedy `generation_config` — same weights | 0.720 |
| exp-04 | wider corpus (all 70 747 distinct problems) + on-policy RFT rows, 1 epoch | **0.773** |
| exp-05 | second RFT round, warm start | 0.767 |
| exp-07 | weight average of exp-04 and exp-05 | 0.736 *(n=500)* |

The base model's problem was termination, not arithmetic: 49% of its completions ran to
the 4000-token cap and 97/150 emitted more than one `ANSWER:` marker, so the
last-number grader read a number out of an invented follow-up problem. After exp-02 that
is 0/150. Everything after exp-02 is ordinary accuracy.

Exhausted axes, for anyone picking this up: more epochs on a narrow corpus (exp-02,
+0.0), a second on-policy round (exp-05, −0.7), weight averaging (exp-07, −0.4). The one
axis never tested is more compute on the *wide* corpus — 2 epochs over `sft_v2` is 3.8 h
and never fit the budget.

## Files

| | |
|---|---|
| `build_data.py` | builds the SFT corpora from OMI2 + gsm8k train (+ `--extra` for RFT rows) |
| `gen_rft.py` | samples a checkpoint k× per gsm8k train question, keeps answer-verified chains |
| `train_sft.py` | completion-only SFT; asserts its rendering against `templates/gemma3.jinja` at startup |
| `package_ckpt.py` | Trainer checkpoint → bf16 + tokenizer/processor + chosen decode config |
| `merge_ckpts.py` | weight average of two checkpoints |
| `analyze_eval.py` | accuracy, termination shape and the high-concurrency garbage screen from an inspect log |
| `run_exp0{3,6,7,8}.sh` | the exact commands the corresponding cards ran |
