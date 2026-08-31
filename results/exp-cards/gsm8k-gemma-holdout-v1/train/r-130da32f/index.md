# Reconstructed experiment cards - run r-130da32f

Base model: HuggingFaceTB/SmolLM3-3B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
3 launches carry a card. Submitted directory: `final_model` = exp-03, a byte copy (`cp -a`) of exp-01's
epoch-2 merged model. The digest records no event timestamps, so every `elapsed_h` is null; the only
clock readings are timer.sh at [7] (9:59 left) and [280] (9:19 left), so the whole run used about 0.7 h
of the 10 h budget.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [52] | null | sft | base_model | gsm8k-train, evaluator prompt format (count not printed) | 2e-04 / 2ep | completed | 0.225 (n=40) | supported | adopt |
| exp-02 | [257] | null | merge | exp-01 | - | - | completed | 0.100 (n=40) | contradicted | reject |
| exp-03 | [270] | null | other | exp-01 | - | - | completed | 0.100 (n=10, smoke) | inconclusive | adopt |

Notes:

- All accuracy numbers except exp-03's smoke test were measured under the same quick protocol:
  `evaluate.py --limit 40 --max-connections 1 --max-tokens 128 --gpu-memory-utilization 0.3`.
  The base-model comparator under that protocol is 0.075 ([248], [256]).
- `baseline40.json` in the workspace holds a different base number, 0.175, measured at `--max-tokens 512`
  ([20], [34]); it is not comparable to the 0.075 comparator.
- No `eval_*.json` produced by these launches survives in the workspace snapshot; every measurement value
  comes from the digest.
- Four earlier attempts to score exp-01 (`--limit 150`, `--limit 100`, and two `--limit 40` runs at
  `--max-tokens 512`) never produced a number - two were killed for slowness, one died on a vLLM start-up
  failure from leftover GPU processes ([188], [204], [210], [235]). They changed no model and produced no
  candidate, so they carry no card.
