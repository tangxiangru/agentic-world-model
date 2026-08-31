# Reconstructed experiment cards - r-049e5f94

Base model: Qwen/Qwen3-4B-Base | benchmark: gsm8k | budget: 10 h, 1x H100

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-01 | 252 | 0.61 | sft | base_model | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 0.30) | killed | 0.375 @40 | inconclusive | adopt |
| exp-02 | 414 | 1.11 | sft | exp-01 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 0.60) | killed | 0.54 @50 | inconclusive | adopt |
| exp-03 | 516 | 1.57 | sft | exp-02 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 0.90) | killed | none | inconclusive | adopt |
| exp-04 | 645 | 1.98 | decode-config | exp-03 | none | n/a | completed | 0.83 @100 | inconclusive | adopt |
| exp-05 | 711 | 2.22 | rft | base_model | data/sft_v2.jsonl (84946) + data/rs_gsm.jsonl (7473) | 1e-5 / 2.0 (ran to 0.90) | killed | 0.82 @100 | inconclusive | reject |
| exp-06 | 892 | 3.03 | rft | exp-05 | data/sft_v2.jsonl (84946) | 1e-5 / 2.0 (never started) | failed | none | inconclusive | abandon_line |
| exp-07 | 911 | 3.08 | sft | exp-03 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 1.21) | killed | 0.84 @150 | inconclusive | adopt |
| exp-08 | 1042 | 3.64 | grpo | exp-07 | openai/gsm8k train (7473) | 1e-6 / 320 steps (ran 112) | killed | 0.84 @150 | contradicted | reject |
| exp-09 | 1149 | 4.10 | grpo | exp-07 | openai/gsm8k train (7473) | 5e-6 / 340 steps (ran 100) | killed | 0.847 @150 | inconclusive | adopt |
| exp-10 | 1219 | 4.45 | grpo | exp-09 | openai/gsm8k train (7473) | 5e-6 / 400 steps (ran 152) | killed | none | inconclusive | abandon_line |
| exp-11 | 1247 | 4.61 | sft | exp-07 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 1.51) | killed | 0.85 @200 | inconclusive | adopt |
| exp-12 | 1351 | 5.08 | sft | exp-11 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (ran to 1.81) | killed | 0.8522 @1319 | inconclusive | adopt |
| exp-13 | 1464 | 5.58 | sft | exp-12 | data/sft_train.jsonl (127473) | 1e-5 / 2.0 (completed) | completed | 0.8446 @1319 | contradicted | reject |

## Notes

- **exp-12 is the submitted card**: its output, runs/sft_v1/checkpoint-3600 (epoch ~1.8),
  was packaged to `final_model` at [1464] and never overwritten afterwards; it was
  verified as delivered at [1579]-[1580] and [1604]-[1605].
- Comparator for the whole run: the unmodified base model at 0.38 (limit 50),
  /home/ben/task/logs/baseline_base.json, launched at [55].
- Five launches of the v1 training command that were killed or crashed while the
  agent was fighting stdout buffering, checkpoint policy and process detachment
  ([128], [168], [208], [233], [245]) are recorded as `provenance.smoke_runs` on
  exp-01 rather than as cards; two three-step GRPO feasibility runs ([1027],
  [1035]) are recorded the same way on exp-08.
- The workspace snapshot contains only the scripts; `logs/`, `data/`, `runs/` and
  `final_model/` are absent, so every measurement path cited in these cards is the
  in-run path named in the stream and cannot be re-read.
- `generate_solutions.py` (STaR sampling, [665]) produced training data rather than
  a candidate model, so it is recorded as `setup.data[].build_command` on exp-05
  rather than as its own card. LoRA merges and the `prep_checkpoint.py`
  eval-directory copies are treated as evaluation plumbing, except the one at [645]
  that first introduced greedy decoding, which is exp-04.
- `combine_data.py` and `fix_eos.py` were written but never invoked in the stream.
