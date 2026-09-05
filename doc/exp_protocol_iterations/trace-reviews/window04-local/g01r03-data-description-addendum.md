# g01r03 training-input description correction

2026-09-04, planner read-only verification after reading the concurrent opportunity review. This corrects an old helper description; it does not reopen the frozen14-cell denominator, change a score, adopt a recipe, or authorize a new experiment/package. No model, training, evaluation or Slurm mutation was performed.

## Confirmed correction: the first SFT did have few-shot-conditioned inputs

The old g01r03 cell report's “Few-shot prefix:none” statement is false. The actual exp02 card explicitly says8 percent of rows carry1–3-shot prefixes in `situation.alternatives_rejected`(line19) and `setup.data.source`(line54). The same plan SHA is retained for the original19:01:24 lock and the later20:21:08 relock.

The retained raw input is:

`/home/robtang_google_com/gangda_workspace/agentic-world-model-gangda-exp-protocol-evolve/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1_g01r03_formal_r1/gsm8k_google_gemma-3-4b-pt_90649/task/data/sft_v1.jsonl`

Its SHA256 is **e96166d08555444f3da9de192d4697622a10729d3325e13a12672bd7ba50fab0**,matching the retained exp02 lock's data entry(bytes171127369). Streaming structured JSON recount gives:

| nshot | raw rows |
|---|---:|
|0|110311|
|1|3246|
|2|3263|
|3|3180|
|total|120000|

Thus9689/120000=8.07417% of raw input rows carry few-shot metadata. This matches the independently produced opportunity-review inventory.

This is not merely a metadata claim. `task/scripts/build_data.py:35–41,96–105` constructs the demonstration text and passes it to `render.render_prompt(...,system=system)` when assigning nshot. `render.py:24–32` also puts one intrinsic `Reasoning:` and two `ANSWER:` references in the ordinary problem template. Across all120000 retained input strings, the planner confirmed **zero mismatches** between actual marker counts and `Reasoning: = nshot+1`, `ANSWER: = nshot+2`. The fixed intrinsic markers must be accounted for; raw marker counts alone are not a general shot-count parser.

`task/scripts/train_sft.py:33–58` consumes the stored prompt and completion,tokenizes them,uses prompt+completion as model input,and masks prompt labels. Therefore the demonstrations condition the model even though their tokens are not supervised labels. The retained `task/logs/exp-02_train.log:1` reports119945 kept,55 length-dropped,43.9M tokens. The8.07417% figure is explicitly **pre-length-filter**,not a silently substituted post-filter proportion. Even if all55 drops were prefixed,at least9634 prefixed rows remain.

Reproduction of the raw histogram uses streaming JSON,not regex over nested records:

```bash
jq -cn 'reduce inputs as $r ({rows:0,nshot:{}}; .rows+=1 | .nshot[($r.nshot|tostring)]+=1)' /absolute/path/to/sft_v1.jsonl
```

The retained lock contains the post-run relock's data hash; its old-lock history preserves time/plan hash/reason but not a separate old data hash. Do not claim that this alone independently proves byte immutability at the initial launch. The predeclared card,retained input,actual builder/consumer and training-log consistency together refute the categorical no-few-shot description.

## Boundaries

- This training-prompt prefix issue is separate from exp08's wrong **dataset-first150** slice,documented in `g01r03-prefix-audit.md`.
- The builder prints its `per_problem` dictionary size after capping/shuffling the row list without recomputing distinct questions; that printed number is not automatically the materialized corpus's unique-question count. The new opportunity report provides full data recounts; this focused verification independently recomputed only the first corpus's prefix counts and file hash.
- No contribution of this prefix mixture to82.79% is isolated. Do not turn8%,1–3shots,350k rows or FP32 into a universal recipe from this observation.
- The planner read the four new opportunity-review documents in full. Their dtype,corpus-overlap and complementarity analyses may inform later design,but this addendum does not silently adopt a bundle strategy while the task's strategy confirmation is outstanding.
- Original helper reports remain unedited. The official score remains1092/1319,validator-complete/eligible,with no judge flags. No clean-cell count was added.
