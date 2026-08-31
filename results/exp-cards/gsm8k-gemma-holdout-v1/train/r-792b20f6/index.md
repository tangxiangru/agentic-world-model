# Reconstructed experiment cards - run r-792b20f6

Base model: Qwen/Qwen3-4B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
6 launches carry a card: two trainings that produced weights, one merge, two packaging copies into
`final_model`, and one training killed as too slow. Submitted directory: `final_model` = exp-05, the
merged exp-01 adapter (exp-03's weights) copied into place at [191].

The digest ends at t=+1.60h with 8.4 h of the budget unspent, while the launch at [209] was still
training, so anything the run did after that point - further checkpoints, evaluations, or a replacement
of `final_model` - is outside the record.

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [73] | 0.10 | sft | base_model | gsm8k-train, all rows (7,473) | 2e-04 / 3ep, LoRA r32 | completed | - | inconclusive | adopt |
| exp-02 | [93] | 1.18 | other | exp-01 | - | - | completed | - | inconclusive | adopt |
| exp-03 | [103] | 1.18 | merge | exp-02 | - | - | failed | 0.200 (n=25); 0.167 (n=30) | inconclusive | adopt |
| exp-04 | [171] | 1.51 | sft | base_model | gsm8k-train, all rows (7,473) | 1e-04 / 5ep, LoRA r32 | killed | - | inconclusive | abandon_line |
| exp-05 | [191] | 1.57 | other | exp-03 | - | - | completed | - | inconclusive | adopt |
| exp-06 | [209] | 1.60 | sft | base_model | gsm8k-train, all rows (7,473) | 1e-04 / 3ep, LoRA r64 | killed | - | inconclusive | abandon_line |

Notes:
- No comparator was ever measured under a matching `--limit` (the base model was never evaluated), so
  every verdict is `inconclusive`.
- The only scores in the run are exp-03's, both on very small samples; the 100-item eval launched at
  [169] never scored a sample (vLLM server exited, see `eval_100.log`).
- Thirteen truncated pipeline runs (`--max_samples` 5-20, indices [21]-[69]) are recorded as
  `provenance.smoke_runs` on exp-01, not as cards.
