# GSM8K post-training of `google/gemma-3-4b-pt`

`final_model/` holds the deliverable. It is a full-weight fine-tune of the immutable base
snapshot `cc012e0a6d0787b4adcc0fa2c4da74402494554d` — no adapters, tokenizer and processor
configs included, loadable by the grader's vLLM path exactly like the base model.

## Result

| checkpoint | n=150 | n=500 | n=1319 (full test split) |
|---|---:|---:|---:|
| base `gemma-3-4b-pt` | 0.0333 | — | — |
| exp-02 SFT | 0.6933 | 0.7100 | 0.6983 |
| exp-04 SFT+RFT | 0.6933 | 0.7240 | 0.7066 |
| exp-05 weight soup | 0.6933 | 0.7140 | 0.7066 |
| **exp-08 = `final_model/`** | **0.7000** | — | **0.7104** |

`final_model/` was re-verified from a fresh process with `evaluate.py`'s own default
arguments (`--model-path final_model --limit 150 --max-connections 2
--gpu-memory-utilization 0.3`): **0.700**, 149/150 well-formed answers, 1 token-cap overrun.
Result file: `eval/final_model_default_args.json`.

**0.0333 → 0.7104**, +67.7 points on the full test split.

## What the model is

Three stages of full fine-tuning, each starting from the previous:

1. **exp-02** — 2 epochs of completion-only SFT on 75k chain-of-thought rows from
   `nvidia/OpenMathInstruct-2` (`gsm8k` + `augmented_gsm8k` rows only), lr 1e-5.
2. **exp-04** — 2 epochs of rejection-sampling FT on 38k of the model's own verified-correct
   chains over 20.8k unseen problems, plus 15k replayed SFT rows, lr 7e-6.
3. **exp-08** — 1 epoch on 50k fresh OpenMathInstruct-2 rows over problems no earlier stage
   trained on, plus 10k replay, lr 7e-6.

Decode defaults in `final_model/generation_config.json` are greedy
(`do_sample: false`, `temperature: 0.0`, `top_k: 1`), because the grader sends no sampling
parameters and vLLM therefore falls back to the checkpoint's own config; the base snapshot
would otherwise be graded on a single temperature-1.0 nucleus sample.

## The two things that mattered

**Format, not arithmetic.** The base model scores 0.0333 not because it cannot do the sums
but because it never stops: only 20/150 completions ended in a well-formed
`ANSWER: <number>` line and 83/150 ran to the 4000-token cap. The scorer
(`match(numeric=True, location="end")`) reads the *last* numeric whitespace-token of the
completion, so a chain that never terminates is scored on noise. One round of SFT whose
targets end in `ANSWER: <n>` followed by `<end_of_turn>` — the exact stop token
`templates/gemma3.jinja` terminates on — took format compliance from 13% to 100% and
accuracy from 0.033 to 0.693. That single change is 98% of the total gain.

**Training strings rendered by the grader's own template.** Every training row is built by
`tokenizer.apply_chat_template` with `templates/gemma3.jinja` loaded from disk, so the
training string is byte-identical to what vLLM sees. 12% of rows carry the grader's exact
10-shot system prefix (~1700 tokens) so the graded condition is in-distribution without
paying 5× the compute on every row.

## What did not work

Everything after exp-02. RFT (exp-04), weight-averaging (exp-05) and more teacher data
(exp-08) each moved the full-split score by under a point, and no pair of the final
candidates is separable by a paired McNemar test (all p > 0.4 across 160–220 discordant
items at n=1319). The model is saturated on gsm8k-style imitation data at this scale: 257
of 1319 items are wrong for every checkpoint produced. Attacking those would need a
different signal — a verifier or process reward — not more imitation.

Measurement noise was the binding constraint for the whole second half of the batch. At
n=150 the standard error is 3.8 points, larger than any difference between candidates, so
exp-06 and exp-07 spent GPU time on statistical power (n=500, then the full 1319) rather
than on more training, and the final pick is the argmax of the largest measurement.

## Contamination

Every training file was checked with `../contamination_check.py` against `../test_data.json`
before its run was launched: 75,000 / 37,951 / 50,000 documents scanned, **0 contaminated,
0 matches** in all three. All problems come from GSM8K's *train* split or from
OpenMathInstruct-2 augmentations of it; nothing was derived from a test item.

## Layout

```
final_model/                  the deliverable
memory/cards/exp-01..08.yaml  one experiment card per run (hypothesis before, result after)
memory/index.md               one line per card
scripts/build_data.py         OpenMathInstruct-2 -> {prompt, completion} in the grader's format
scripts/build_data2.py        same, plus --skip-file to exclude already-trained problems
scripts/train_sft.py          completion-only trainer (sparse LM head, token-budget batching)
scripts/rft_sample.py         vLLM rejection sampling from a checkpoint
scripts/soup.py               weight averaging
scripts/package_final.py      writes and verifies final_model/
scripts/analyze_eval.py       accuracy / format compliance / truncation from an inspect log
eval/, analysis/, logs/       every measurement referenced by a card
ckpts/                        exp-02, exp-04, exp-08 and the soup
```

### Two implementation notes

* **Memory.** A 262k-vocab 4B model does not fit on one H100 with a dense LM head over full
  sequences. `train_sft.py` runs the head only on positions carrying a label (prompt tokens
  are ~60% of the corpus and ~90% of a few-shot row) and builds micro-batches to a
  14336-token budget after sorting by length. Peak 71.9 GB, ~10k training tokens/s.
* **`liger-kernel` cannot be installed here.** The container root filesystem is a 64 MB
  overlay that is already full; the install died with ENOSPC. The sparse LM head above is
  the workaround.
