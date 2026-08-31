# Reconstructed experiment cards - run r-c3036179

Base model: Qwen/Qwen3-1.7B-Base. Benchmark: gsm8k. Budget: 10 h, one H100.
25 launches carry a card. Submitted directory: `final_model` = exp-07 (packaging of exp-21's weights),
with its generation cap last set by narration to max_new_tokens 1300 at [996] (no shell block for that write).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | [66] | null | sft | base_model | gsm8k-train x2 + 12k synthetic (26,946) | 2e-05 / 2ep | completed | 0.160 (n=150) | inconclusive | abandon_line |
| exp-02 | [125] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 1e-05 / 2ep | completed | 0.427 (n=150) | supported | adopt |
| exp-03 | [164] | null | sft | exp-02 | gsm8k-train fixed-prompt (7,473) | 5e-06 / 2ep | completed | 0.453 (n=150) | supported | reject |
| exp-04 | [178] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 1ep (stop@234) | completed | 0.685 (n=1319) | contradicted | reject |
| exp-05 | [199] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 1ep (stop@234) | completed | 0.687 (n=1319) | contradicted | reject |
| exp-06 | [274] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 2ep | completed | 0.685 (n=1319) | supported | adopt |
| exp-07 | [356] | 9.0 | other | exp-21 | - | - | completed | 0.696 (n=1319) | contradicted | adopt |
| exp-08 | [362] | null | sft | unknown | gsm8k-train fixed-prompt (7,473) | 2e-06 / 2ep | completed | 0.540 (n=150) | supported | adopt |
| exp-09 | [411] | null | sft | exp-08 | gsm8k-train fixed-prompt (7,473) | 1e-06 / 2ep | completed | 0.627 (n=150) | contradicted | reject |
| exp-10 | [457] | null | sft | exp-08 | gsm8k-train fixed-prompt (7,473) | 1e-06 / 1ep | completed | 0.500 (n=150) | contradicted | reject |
| exp-11 | [486] | null | other | exp-08 | - | - | killed | - | inconclusive | abandon_line |
| exp-12 | [490] | null | other | exp-08 | - | - | completed | 0.560 (n=150) | inconclusive | adopt |
| exp-13 | [500] | null | decode-config | exp-12 | - | - | completed | 0.633 (n=150) | supported | adopt |
| exp-14 | [507] | null | decode-config | exp-13 | - | - | completed | 0.487 (n=150) | contradicted | reject |
| exp-15 | [513] | null | decode-config | exp-14 | - | - | completed | 0.679 (n=1319) | supported | adopt |
| exp-16 | [554] | null | other | exp-06 | - | - | completed | 0.647 (n=150) | supported | adopt |
| exp-17 | [590] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 1ep | killed | - | inconclusive | abandon_line |
| exp-18 | [599] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 1ep | completed | 0.703 (n=1319) | supported | reject |
| exp-19 | [641] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 2ep | completed | 0.672 (n=1319) | contradicted | reject |
| exp-20 | [763] | 8.0 | sft | base_model | gsm8k-train fixed-prompt (7,473) | 2e-05 / 1ep (stop@234) | completed | 0.627 (n=150) | contradicted | reject |
| exp-21 | [794] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 1.5e-05 / 1ep (stop@234) | completed | 0.705 (n=1319) | supported | adopt |
| exp-22 | [820] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 1.25e-05 / 1ep (stop@234) | completed | 0.701 (n=1319) | contradicted | reject |
| exp-23 | [842] | null | sft | base_model | gsm8k-train fixed-prompt (7,473) | 1.5e-05 / 1ep (stop@260) | completed | 0.647 (n=150) | contradicted | reject |
| exp-24 | [883] | 9.0 | sft | base_model | gsm8k-train fixed-prompt (7,473) | 1.75e-05 / 1ep (stop@234) | completed | 0.680 (n=150) | inconclusive | reject |
| exp-25 | [941] | null | decode-config | exp-07 | - | - | completed | 0.701 (n=1319) | supported | reject |

## Run-level notes

- The digest's shell blocks are not in narration order: several eval, config-patch and packaging blocks sit
  hundreds of indices away from the say events that describe them (for example the submission copy at [356]
  is narrated at [903]-[908], and the evals of the [599] and [641] runs sit at [81], [142], [147], [162]).
  Cards are numbered by event index, which is therefore not wall-clock launch order for those cards.
- Two evaluated candidates have no launch event in the digest and so have no card: `run_fixed_lr2e5_more`
  (the parent of exp-08, eval_run_fixed_lr2e5_more_limit150.json = 0.493) and `run_fixed_synth6k`
  (eval_run_fixed_synth6k_limit150.json = 0.447, narrated at [229], [233], [262]).
- The final generation-config lock to max_new_tokens 1300 ([996]) and the stop-token patch of the
  short-prompt checkpoint ([98]-[102]) have no shell blocks, so neither is a card.
- elapsed_h is null except where the agent read the timer aloud immediately before a launch ([740], [864]);
  the digest carries no per-event timestamps.
