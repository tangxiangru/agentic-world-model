# GSM8K post-training of `google/gemma-3-4b-pt`

**Shipped: `final_model/` — 0.7286 on the full GSM8K test set (961/1319), up from 0.0333 for the base checkpoint.**
Verified end to end in exp-08: a fresh `python evaluate.py --model-path final_model --limit -1` loads the
directory, resolves `templates/gemma3.jinja`, and produces that number.

Full experimental record: `memory/index.md` and `memory/cards/exp-0*.yaml` (one card per experiment,
sections 0–4 written and locked before each launch).

## What ships

`final_model/` is `google/gemma-3-4b-pt` (snapshot `cc012e0a…`) with:

1. **Full-parameter SFT** (exp-02) on 80,505 GSM8K-style chains-of-thought derived from
   `nvidia/OpenMathInstruct-2` (the `gsm8k` + `augmented_gsm8k` portions, both seeded from the GSM8K
   **train** split). Every target is rendered through the grader's own chat template and ends with one
   `ANSWER: <n>` line followed by `<end_of_turn>`; loss on completion tokens only. 2 epochs, lr 1e-5,
   effective batch 64, bf16, 2.1 h on one H100.
2. **Greedy decoding** (exp-03) pinned in `generation_config.json`.

## Numbers

| card | what changed | n | accuracy |
|---|---|---:|---:|
| exp-01 | base checkpoint, as shipped | 150 | 0.0333 |
| exp-02 | + SFT on 80,505 chains | 150 | 0.6400 |
| exp-03 | + greedy decoding | 150 | **0.7267** |
| exp-04 | epoch-1 checkpoint instead of epoch-2 | 150 | 0.7133 |
| exp-05 | + 1 epoch of rejection-sampled self-training | 150 | 0.7067 |
| exp-06 | exp-03 vs exp-05, full test set | 1319 | 0.7293 / 0.7309 |
| exp-07 | weight-space average of the two | 1319 | 0.7248 |
| exp-08 | the shipped `final_model/` directory itself | 1319 | **0.7286** |

## The two things that actually moved the number

**Termination (exp-02, +60.7).** The base model already emitted an `ANSWER:` line on 81% of items — it
just never stopped. It would answer correctly, then fabricate three more questions and degenerate; the
grader (`match(numeric=True)`, location `end`) reads the *last* numeric token, so a correct answer got
overwritten by junk. 66/150 ran to the 4000-token cap. Graded on its *first* `ANSWER:` line the same base
completions score 0.153 rather than 0.033. After SFT: 0/150 run on, 3/150 hit the cap.

**Decoding (exp-03, +8.7 on identical weights).** `evaluate.py` sends no `temperature`, and inspect only
forwards sampling params that are explicitly set, so vLLM (`--generation-config auto`) takes its defaults
from the model directory's `generation_config.json`. Gemma-3's ships `do_sample: true, top_p: 0.95,
top_k: 64` and no temperature — i.e. T=1.0. Setting `temperature: 0.0` there is a property of the
checkpoint, costs nothing, and was the single largest per-minute gain in the batch.

## What did not work (all measured, all inside the noise floor)

- **A second epoch** (exp-04): 0.7133 vs 0.7267, a 1.3-point gap under a ~3-point floor at n=150. Chain
  length barely moved (492 vs 510 median chars), so the memorisation this was checking for is absent.
- **Rejection-sampling fine-tuning** (exp-05): sampling k=4 at T=1.0 over 27,473 training questions solved
  88.8% of them at least once against a 69.8% single-sample rate — a 19-point reliability gap that looked
  very much like free points. Training on 45,801 of those verified chains plus 12,039 teacher chains for
  the 3,039 questions it never solved moved nothing (0.7309 vs 0.7293 at n=1319).
- **A two-model weight average** (exp-07): 0.7248, below both parents.

Three structurally different interventions landing inside the floor is the finding: **0.73 is a plateau for
this base model on GSM8K-derived data**, not a limitation of any one recipe. The residual errors are
comprehension failures — one item reads *"10 of the remaining elves quit"* as *one tenth* of them; another
inverts *"each level has half the square footage as the level below"* — not arithmetic or formatting.

## Method notes worth reusing

- **`--limit -1` costs 4 minutes, not 20.** Two of this batch's decisions (exp-04, exp-05) were taken on
  n=150 reads whose ±3.7-point standard error was wider than the effect being argued about; exp-06 showed
  the n=150 ordering of the two finalists reversed on the full set. There was never a reason to decide
  anything on 150 items.
- **Sparse cross-entropy.** Gemma-3's vocabulary is 262k, so materialising `[B, T, V]` logits for a
  2.2k-token few-shot prompt — not the weights — is what OOMs a 4B full fine-tune on an 80 GB card. Taking
  CE only at loss-carrying positions (`scripts/train_sft.py`) drops the peak to 42 GiB with AdamW states,
  and returns a loss identical to the model's own `labels=` path to 6e-08.
- **Tied weights and in-place merges.** `embed_tokens.weight` and `lm_head.weight` are two `state_dict`
  entries over *one* storage: distinct Python objects with equal `data_ptr()`. An in-place per-key average
  applies twice and silently lands that tensor on 0.25/0.75. Dedupe on `data_ptr()`, not `id()`.
- **Few-shot mismatch is not the problem it looks like.** The harness puts a 2044-token 10-shot prefix on
  100% of eval prompts; training carried one on 10% of rows. Probed directly on 800 GSM8K *train*
  questions: 0.7825 with the prefix, 0.8000 without — 1.75 points, inside its own noise.

## Decontamination

All training data is derived from the GSM8K **train** split or generated by the model itself on train
questions. Every training file was run through `../contamination_check.py` against `../test_data.json`:
80,505 / 72,838 documents scanned, **0 contaminated, 0 matches**. The test copy was used only as the
checker's reference input.

## Layout

```
final_model/                  the deliverable (hardlinks into ckpts/exp-02/final + greedy generation_config)
ckpts/exp-02/                 SFT run: checkpoint-629/1258/1887/2516, final, final_greedy
ckpts/exp-05/                 RFT run: checkpoint-570/1139, final, final_greedy
ckpts/exp-07/soup             weight average of exp-02/final and exp-05/final
data/                         training files + their .decon.jsonl contamination-checker copies
eval/                         one metrics json per scored arm
analysis/                     per-card diagnostics (termination, length, error breakdowns)
scripts/                      build_data, build_hard, mix_data, train_sft, rft_sample, soup,
                              make_variant, finalize_ckpt, probe_fewshot, analyze_log
memory/                       experiment cards, lock files, WMA verdicts, index
```
