# r-e3d4334d — reconstructed experiment cards

Base model: google/gemma-3-4b-pt · benchmark: gsm8k · budget: 10 h, 1x H100 · 12 cards.
The digest carries event timestamps, so `elapsed_h` is the hours-since-start of each launch.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 16409 | 0.21 | sft | base_model | data/gsm8k_sft.jsonl (7473, gsm8k train in eval format) | 1e-5 / 2 | failed | none (OOM on the 262K-vocab CE before step 1) | inconclusive | iterate |
| exp-02 | 16920 | 0.23 | sft | base_model | data/gsm8k_sft.jsonl (7473) | 1e-5 / 2 | failed | none (liger FLCE patch inert: package not installed) | inconclusive | iterate |
| exp-03 | 20249 | 0.45 | sft | base_model | data/gsm8k_sft.jsonl (7473) | 1e-5 / 2 | completed | 0.60 @ n=150 | inconclusive | adopt |
| exp-04 | 24851 | 1.45 | sft | base_model | data/v2.jsonl (108946: MetaMathQA GSM subsets + gsm8k x2) | 2e-5 / 1 | killed | none (killed at 16 min to change batch geometry) | inconclusive | iterate |
| exp-05 | 25405 | 1.48 | other (packaging: exp-03 into final_model) | exp-03 | — | — / — | completed | none (never evaluated in place) | inconclusive | iterate |
| exp-06 | 26613 | 1.71 | sft | base_model | data/v2.jsonl (108946) | 2e-5 / 1 | killed | none (died with the shell call that launched it) | inconclusive | iterate |
| exp-07 | 27473 | 1.74 | sft | base_model | data/v2.jsonl (108946) | 2e-5 / 1 | failed | none (OOM at micro-batch 32) | inconclusive | iterate |
| exp-08 | 27762 | 1.81 | sft | base_model | data/v2.jsonl (108946) | 2e-5 / 1 | completed | 0.6033 @ n=300 | inconclusive | adopt |
| exp-09 | 32370 | 4.31 | rft | exp-08 | data/v3.jsonl (43578: 20633 self-sampled verified chains + gsm8k x2 + 8000 replay) | 5e-6 / 1 | completed | 0.63 @ n=300 | supported | adopt |
| exp-10 | 35592 | 5.67 | dpo | exp-09 | data/dpo_pairs.jsonl (2611 pairs, 2104 within max-len) | 5e-7 / 1 | completed | 0.6267 @ n=300; 0.6027 @ n=1319 | contradicted | reject |
| exp-11 | 39270 | 6.47 | rft | exp-09 | data/v5.jsonl (49183: 37710 round-2 chains on unused rephrased questions + gsm8k + 4000 replay) | 4e-6 / 1 | completed | 0.5933 @ n=300 | contradicted | reject |
| exp-12 | 39610 | 6.63 | other (packaging: exp-09 into final_model) | exp-09 | — | — / — | completed | 0.6027 @ n=1319 | inconclusive | adopt |

## Notes

- The submitted model is **exp-12**: `final_model`, package_final.py's copy of
  exp-09's checkpoint (`runs/sft_v3/final`) with a greedy generation_config,
  measured at 0.60273 (stderr 0.0135) over the whole 1319-item set through the
  official eval path, and verified on disk at [41051] with 1:56 left.
- The line that worked: eval-format SFT on gold gsm8k (exp-03, 0.60 @150) →
  teacher-data augmentation, flat (exp-08, 0.6033 @300) → on-policy rejection
  sampling, +2.7 pts (exp-09, 0.63 @300). DPO (exp-10) and a second RFT round on
  unused rephrased questions (exp-11) were both measured against exp-09 under the
  same `--limit 300` and dropped.
- Only exp-09, exp-10 and exp-11 share a protocol with their comparator
  (`--limit 300`), so they are the only supported/contradicted verdicts. The base
  measurement (0.07) was taken at `--limit 100` and exp-03 at `--limit 150`, so
  the 7% → 60% jump cannot be called a supported result under the extraction rule.
- Five of the twelve cards are launches that crashed or were killed (exp-01,
  exp-02, exp-04, exp-06, exp-07): two OOMs on gemma's 262K-vocab cross-entropy,
  one missing dependency, one process killed with its shell call, one deliberate
  restart at a different batch geometry. They are cards, not smoke runs — each was
  a full-data, full-epoch launch meant to produce a candidate.
- The one smoke run is the package_final.py dry run at [39455] into a temp dir; it
  is recorded on exp-12 as `provenance.smoke_runs`.
- The greedy `generation_config.json` rewrite (do_sample true, temperature 0.0,
  eos [1, 106]) is applied to every checkpoint before its eval — [23999], [29620],
  [35128], [36778], [39892] — and is recorded as part of each card's evaluation
  protocol, not as a decode-config card: it was applied uniformly and never
  measured against the same checkpoint decoded stochastically.
- `logs_train_v1.log` and `logs_train_v2.log` in the workspace snapshot hold only
  the attempts that finished; each relaunch overwrote the previous log, so the
  tracebacks of exp-01, exp-02, exp-06 and exp-07 survive only as the agent's
  readings in the digest.
- `final_model` is one path written twice: by exp-05 (exp-03's weights) and by
  exp-12 (exp-09's). Nothing in the stream touches it after [39610].
