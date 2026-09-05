# Session notes (not experiment cards)

## New pitfall found at the very end: greedy vLLM evals are not reproducible

`final_model/` holds a byte-identical copy of `ckpts/exp-06_soup` (verified: same
safetensors sizes, same greedy `generation_config.json`). Scored twice under the
same protocol (`evaluate.py --limit 150 --max-connections 8
--gpu-memory-utilization 0.85`, `do_sample=false, temperature=0.0`) it returned:

| run | path | accuracy | log |
|---|---|---:|---|
| exp-06 | `ckpts/exp-06_soup` | 0.6867 | `eval/exp-06_soup_dev150.json` |
| final verification | `final_model` | 0.7067 | `eval/final_model_dev150.json` |

Same weights, same decode config, same 150 items: a 2.0-point spread. The cause
is vLLM's batching — continuous batching and chunked prefill change how the
reduction over each matmul is ordered, so a bf16 logit gap of ~1e-3 can flip an
argmax. Greedy decoding removes sampling noise, not execution noise.

**Consequence for this session's record.** Every `--limit 150` comparison
carries roughly this much run-to-run jitter *on top of* the 3.8-point binomial
standard error. exp-03 (0.6867), exp-04 (0.6933) and exp-06 (0.6867) were
already inside one standard error of each other; this makes them
indistinguishable twice over. exp-07's `--limit 500` paired comparison
(McNemar z = 0.61 over 67 discordant items) is the only measurement in the
session with enough resolution to be worth quoting, and even it did not
separate the two finalists.

**What to do next time.** Score every candidate under one protocol *and* budget
for repeat runs of the same checkpoint, so the jitter is measured rather than
assumed away. Suggested entry for `skills/exp_protocol/pitfalls.yaml`:

```yaml
- id: greedy_eval_nondeterminism
  symptom: The same checkpoint scored twice under the same greedy protocol returns
    different accuracies; a small A/B delta reverses sign between runs.
  cause: vLLM's continuous batching and chunked prefill reorder the reductions
    inside each matmul, so bf16 logit ties resolve differently depending on what
    else was in the batch. temperature=0 removes sampling noise, not this.
  check: null
  guidance: Re-score the incumbent (not just the candidate) in the same session and
    treat the spread as the floor on any delta you are willing to act on. Deltas
    under ~2 points at n=150 are unmeasurable regardless of binomial stderr.
  source: this session, exp-06 vs the final_model verification run (2026-09-03)
```

## Where things ended up

- `final_model/` = uniform weight average of `ckpts/exp-04/final` and
  `ckpts/exp-05/final`, served greedily. Best measurement: **0.670 at n=500**
  (`eval/exp-06_soup_dev500.json`); 0.687 and 0.707 on two n=150 runs.
- Base `google/gemma-3-4b-pt` under the same protocol: **0.033** (exp-01).
- The single largest gain was exp-02 (+56.7 pts), and most of that was teaching
  the model to emit `<end_of_turn>`: the grader reads the *last* number in the
  completion, and an unterminated pretrained model answers correctly and then
  invents further problems whose answers overwrite it.
- The second gain was free: exp-03, greedy decoding via `generation_config.json`
  (+8.7 pts), because the checkpoint had inherited `do_sample: true, top_k: 64,
  top_p: 0.95` from the base and vLLM was sampling at T=1.0.
- Everything after that (rejection sampling, 2x unique data, weight averaging)
  moved the score by less than the measurement noise.
