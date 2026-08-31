# r-c95a5e9a — reconstructed experiment cards

base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h on one H100 · 11 cards

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 121 | null | sft | base_model | openai/gsm8k main (train) | 2e-4 / 2 | completed | accuracy 0.487, n=150 (exp1_metrics_150.json) | inconclusive | reject |
| exp-02 | 192 | null | sft | base_model | deepmind/aqua_rat (train) | 1e-4 / 1 | failed | none (crashed in the run_config.json dump before training) | inconclusive | iterate |
| exp-03 | 196 | null | sft | base_model | deepmind/aqua_rat (train) | 1e-4 / 1 | completed | none (never scored on gsm8k) | inconclusive | adopt |
| exp-04 | 205 | null | sft | exp-03 | openai/gsm8k main (train), 2-shot prefix p=0.5 | 2e-4 / 2 | completed | accuracy 0.513, n=150 (stage2_metrics_150.json); 0.4996, n=1319 (stage2_aqua_fewshot2_full_metrics.json) | supported | adopt |
| exp-05 | 225 | null | sft | exp-03 | openai/gsm8k main (train), 4-shot prefix p=0.5 | 2e-4 / 2 | completed | accuracy 0.500, n=150 (stage2_fewshot4_metrics_150.json) | contradicted | reject |
| exp-06 | 290 | null | sft | exp-03 | openai/gsm8k main (train), 2-shot prefix p=1.0 | 2e-4 / 2 | completed | accuracy 0.520, n=150 (stage2_fewshot2_p1_metrics_150.json) | supported | adopt |
| exp-07 | 379 | null | sft | exp-06 | openai/gsm8k main (train), 10-shot prefix p=1.0 | 1e-4 / 1 | completed | accuracy 0.520, n=150 (stage3_fewshot10_from_p1_metrics_150.json) | inconclusive | reject |
| exp-08 | 517 | null | other (packaging: copy of the exp-06 merged model into final_model) | exp-06 | — | — / — | completed | accuracy 0.560, n=150 (final_model_metrics_150.json) | inconclusive | adopt |
| exp-09 | 647 | null | sft | base_model | meta-math/MetaMathQA, MATH_* rows only | 1e-4 / 1 | completed | none (never scored on gsm8k) | inconclusive | adopt |
| exp-10 | 739 | null | sft | exp-09 | openai/gsm8k main (train), 2-shot prefix p=1.0 | 2e-4 / 1.5 | completed | accuracy 0.369, n=1319 (new_stage2_metamath10k_full_metrics.json) | contradicted | reject |
| exp-11 | 850 | null | other (packaging: copy of the exp-04 merged model into final_model) | exp-04 | — | — / — | completed | accuracy 0.500, n=20 (final_model_metrics_20.json); 0.4996, n=1319 (final_model_metrics_full.json) | inconclusive | adopt |

Notes

- `elapsed_h` is null on every card: the digest header states this run carries no event
  timestamps. Only `timer.sh` readings bound the timeline (9:57 remaining at [21], 5:38 at
  [223], 3:15 at [383], 1:52 at [651], 0:30 at [846]).
- Nothing in this run states a problem or a hypothesis before a launch: the filtered
  stream is shell events only until the closing report at [862], so
  `provenance.stated_by_agent` is false/false on all 11 cards and every `hypothesis.claim`
  is null. Verdicts rest on whether the agent measured the output against a comparator
  under the same `--limit`.
- Smoke runs are not cards: the two `runs/exp_smoke*` launches at [77] (crashed: set not
  JSON serializable) and [85] (passed, 256 train samples, 0.300 on --limit 30 at [113])
  are recorded on exp-01 as `provenance.smoke_runs`.
- Two lines were tried. AQuA-RAT stage-1 (exp-03) → gsm8k stage-2 few-shot sweep
  (exp-04/05/06) → a third gsm8k pass (exp-07); then, with 2 h left, a MetaMathQA
  MATH-only stage-1 (exp-09) → gsm8k (exp-10), which came in 13 pts below on the full test
  and was dropped.
- `{dir}/final_model` is written twice: exp-08 packages the exp-06 checkpoint (best on
  --limit 150), exp-11 replaces it with the exp-04 checkpoint (the only candidate with a
  full-test number) and moves the earlier one to `final_model_prev`. exp-11 is the
  submitted artifact per the closing report at [862].
- The base-model eval at [22] (0.0 on --limit 30, `baseline_metrics.json`) is the run's
  only base number and was measured under a different protocol than every trained
  checkpoint (--limit 150), so no card carries a base-model delta.
- `final_model_metrics_full.json` holds exactly the value of
  `stage2_aqua_fewshot2_full_metrics.json` (0.49962092494313876). The full-test eval of
  the exp-06 packaging launched at [569] never reported, so that checkpoint has no
  full-test score in the snapshot.
