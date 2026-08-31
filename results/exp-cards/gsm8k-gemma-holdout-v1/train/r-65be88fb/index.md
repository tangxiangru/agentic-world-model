# Reconstructed experiment cards — r-65be88fb

Base model post-trained: `Qwen/Qwen3-1.7B-Base` · benchmark: gsm8k · budget: 10 h, 1x H100 80GB.
16 launches, in launch order. Adopted / submitted: **exp-16** (`final_model`, a copy of the exp-06
checkpoint `checkpoints/sft_v4_meta`).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 68 | 0.07 | sft | base_model | sft_gsm8k_only.jsonl (7,473 gsm8k train, duplicated `<think>` block) | 2e-5 / 3 | completed | 0.08 @ limit 50 | inconclusive | reject |
| exp-02 | 107 | 0.28 | sft | base_model | sft_gsm8k_nothink.jsonl (7,473 gsm8k train) | 2e-5 / 2 | failed | — (crashed on `assistant_only_loss=True`) | inconclusive | abandon_line |
| exp-03 | 119 | 0.29 | sft | base_model | sft_gsm8k_nothink.jsonl (7,473 gsm8k train) | 2e-5 / 2 | completed | 0.06 @ limit 50 | inconclusive | reject |
| exp-04 | 180 | 0.58 | decode-config | exp-03 | — | — | completed | 0.34 @ limit 50 (from 0.06) | supported | adopt |
| exp-05 | 209 | 0.63 | sft | base_model | sft_fewshot_gsm.jsonl (22,419 = 7,473 gsm8k train x3, 10-shot system message) | 2e-5 / 2 | completed | 0.60 @ limit 50 | supported | adopt |
| exp-06 | 242 | 2.20 | sft (continuation) | exp-05 | sft_fewshot_meta.jsonl (37,473 = 7,473 gold + 30k MetaMathQA GSM) | 5e-6 / 1 | completed | 0.74 @ limit 50 · 0.7467 / 0.76 @ limit 150 | supported | adopt |
| exp-07 | 289 | 3.62 | grpo | exp-06 | gsm8k train, in-process 5-shot prompts | 5e-7 / 250 steps | killed | — (killed at ~step 52) | inconclusive | abandon_line |
| exp-08 | 317 | 3.79 | sft (continuation) | exp-06 | sft_fewshot_meta2.jsonl (87,473 = 7,473 gold + 80k MetaMathQA GSM, seed 123) | 3e-6 / 1 | completed | 0.72 @ limit 150 | contradicted | reject |
| exp-09 | 369 | 7.28 | rft | exp-06 | sft_reject_mix.jsonl (14,694 = 7,221 self-sampled correct + 7,473 gold) | 5e-6 / 2 | completed | 0.6333 @ limit 150 | contradicted | reject |
| exp-10 | 399 | 8.26 | grpo | exp-06 | gsm8k train, in-process 5-shot prompts | 1e-6 / 120 steps | failed | — (`generation_batch_size (4) must be divisible by num_generations (8)`) | inconclusive | abandon_line |
| exp-11 | 402 | 8.27 | grpo | exp-06 | gsm8k train, in-process 5-shot prompts | 1e-6 / 120 steps | failed | — (rollout backend rejected `eos_token_id` in `generation_kwargs`) | inconclusive | abandon_line |
| exp-12 | 416 | 8.28 | grpo | exp-06 | gsm8k train, in-process 5-shot prompts | 1e-6 / 120 steps | completed | 0.7067 @ limit 150 | contradicted | reject |
| exp-13 | 441 | 8.39 | sft (continuation) | exp-06 | sft_fewshot_gsm.jsonl (22,419, reused from exp-05) | 1e-6 / 1 | completed | 0.64 @ limit 150 | contradicted | reject |
| exp-14 | 471 | 9.24 | grpo | exp-06 | gsm8k train, in-process 5-shot prompts | 5e-7 / 250 steps | completed | 0.72 @ limit 150 | contradicted | reject |
| exp-15 | 487 | 9.41 | sft (continuation) | exp-06 | sft_exact_fewshot.jsonl (7,463 gsm8k train under the evaluator's own 10 exemplars) | null / null (launch line truncated) | completed | 0.7133 @ limit 150 | contradicted | reject |
| exp-16 | 503 | 9.69 | other (packaging) | exp-06 | — | — | completed | 0.7467 @ limit 150 | inconclusive | adopt |

## Shape of the run

Two things moved the score, both in the first 2.2 hours. exp-04 is not a training run at all: the
checkpoint was answering correctly under a hand-run greedy call and scoring 0.06 under the harness,
because vLLM was sampling at its default temperature 1.0 and neither the harness nor the checkpoint
specified one. Writing `temperature: 0.0` into `generation_config.json` took the same weights from
0.06 to 0.34 at limit 50. exp-05 then rebuilt the training data in the eval's own prompt shape — a
10-shot system message, an empty `<think>` block, one `ANSWER:` line — which fixed the run-on failure
outright (50/50 generations stopped, 0 continued past the answer) and reached 0.60. exp-06 added 30k
MetaMathQA GSM rows on top and reached 0.7467 at limit 150.

Everything after exp-06 lost. Three more SFT variants (more MetaMath, rejection-sampled self-solutions,
a low-LR GSM8K-only pass, the evaluator's exact exemplars) came in at 0.72, 0.6333, 0.64 and 0.7133;
three GRPO passes produced one killed run, two crashes and two finished checkpoints at 0.7067 and
0.72. The incumbent was never beaten after t=+3.6h, and roughly six of the ten hours went into
launches that were all rejected.

## Comparators and protocol

There is **no base-model baseline anywhere in the run**, so exp-01, exp-03 and exp-05 have no
like-for-like comparator; exp-01 and exp-03 are `inconclusive` for that reason. exp-04 is measured
against exp-03 at limit 50 on the same weights. exp-06's comparator is exp-05 at limit 50.

From exp-08 onwards the agent printed its candidate and the incumbent side by side, so those verdicts
are `contradicted` on measured deltas: exp-08, exp-09, exp-12 and exp-13 against `eval_v4_150.json`
(0.7467, max-tokens 1024); exp-14 and exp-15 against `eval_final_v4_150.json` (0.76, max-tokens 4000).
The incumbent itself read 0.76 at [468] and 0.7467 at [513] under the identical command, so a 2-point
band of that size is harness noise, and every "regression" of 3-5 points here sits near one standard
error of a 150-item eval. Only exp-09 (-11.3) and exp-13 (-10.7) are clearly outside it.

## What is and is not a card

`final_model` was rebuilt sixteen times ([85], [131], [232], [263], [337], [347], [383], [393],
[431], [441], [451], [461], [477], [487], [493], [503]). Only the last is carded, as exp-16: every
other write was either the deploy step of a training launch that already has its own card — the
harness evaluates `--model-path final_model`, so a copy is how a checkpoint gets measured at all — or
a restoration of the exp-06 checkpoint between launches. Three crashed launches that the agent meant as
real runs (exp-02, exp-10, exp-11) are cards; two genuine dry runs are not, and appear as
`provenance.smoke_runs`: the in-process reward-function check at [281] (on exp-07) and the
transformers-based rejection sampler at [347] that was killed for running at ~10.4 s/item and
replaced with a vLLM sampler (on exp-09).

## Gaps in the record

- The workspace snapshot holds only `evaluate.py`, `system_monitor.log` and `timer.sh` — none of the
  training scripts, data files or `eval_*.json` outputs survive, so every measurement is cited to the
  agent's own path under `/home/ben/task/logs/` and read from the digest.
- No hyper-parameter is filled from an argparse default: the scripts exist only as `write` events in
  the digest, not in the workspace, so `warmup`, `weight_decay` and `seed` are null on every SFT card
  even though the script text shows them.
- exp-15's launch line is truncated in the digest immediately after `--epochs`, so that card carries
  no hyper-parameters and an incomplete argv.
- Several runs (exp-06, exp-09, exp-13, exp-14, exp-15) have no loss, step count or `train_runtime`
  line in the digest; their `wall_h` is the gap between the launch event and the packaging event that
  followed, noted as such on each card.
- The digest carries no official score, and the run's own closing summary reports the submission as
  "~75%".
