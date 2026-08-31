# Reconstructed experiment cards — r-8797fe5f

Base model post-trained: `google/gemma-3-4b-pt` · benchmark: gsm8k · budget: 10 h, 1x H100.
5 launches, in launch order. Adopted / submitted: **exp-02** (`final_model`, the exp-01 checkpoint
packaged with a greedy generation config).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 268 | 0.69 | sft | base_model | sft_clean.jsonl (257,872 rows loaded / 249,840 trained, 120.1M tok = GSM8K train + OpenMathInstruct-2 gsm8k/aug from train_5M + math/aug_math from train_1M) | 2e-5 / 1 | completed | 0.835 @ dev-400 greedy | inconclusive | adopt |
| exp-02 | 368 | 2.83 | other (packaging + decode config) | exp-01 | — | — | completed | 0.74 @ limit 150 · 0.7533 @ limit 150 (re-run) | inconclusive | adopt |
| exp-03 | 536 | 6.66 | rft | exp-01 | sft_round2.jsonl (72,264 rows loaded / 70,026 trained, 31.2M tok = 28,645 self-sampled correct solutions + gold for unsolved + ≤40k aug top-up) | 8e-6 / 1 | completed | 0.825 @ dev-400 · 0.7333 @ limit 150 greedy (0.6667 sampled) | inconclusive | reject |
| exp-04 | 571 | 7.25 | merge | exp-03 (0.5) + exp-01 (0.5) | — | — | completed | 0.815 @ dev-400 | contradicted | reject |
| exp-05 | 657 | 7.50 | grpo | exp-01 | 12,162 prompts (3,146 never solved + 9,016 solved ≤2 times out of 4 samples) | 2e-6 / 100 steps | completed | 0.815 @ dev-400 (step 40) · 0.7725 (step 100) | contradicted | reject |

Two cards carry `adopt`: exp-01 is the only checkpoint that ever became the incumbent, and exp-02 is the
single write to `final_model` that packages it (weights verified byte-identical at [677], directory
verified unchanged at [718] with 1 h 49 m left). Nothing was written to `final_model` after [368].

Comparators: exp-01 has none — the base model's only number is 0.05 at limit 100 under its own sampled
generation config, a different protocol, so the card is `inconclusive` despite the size of the jump.
exp-03, exp-04 and exp-05 are each measured against exp-01 on the same 400-item held-out dev set with
greedy decoding (0.835), and exp-03 additionally at limit 150 against exp-02's 0.74. All three came in
below exp-01: −1.0 pt (RFT), −2.0 pts (soup), −2.0/−6.3 pts (GRPO at step 40 / step 100). The RFT gap is
inside one standard error and the agent recorded it as a "statistical tie" [692], hence `inconclusive`
there and `contradicted` for the other two.

The run's one clearly measured mechanism is the decode config: the exp-03 checkpoint scored 0.6667 and
then 0.7333 at limit 150 under the same argv ([585] vs [606]), which the agent read as a 6.6-point
sampling-vs-greedy artifact — the reason exp-02 exists at all. The stream does not show the config edit
that produced the second number.

Smoke runs are recorded on the cards, not as cards of their own: eight precede exp-01 ([203], [209],
[215], [221], [227], [250], [256] — batch-size and collator sweeps on 2k–8k rows, none of whose result
blocks survive in the digest — and [262], a 50-item eval of the last smoke checkpoint), and four precede
exp-05 ([615] OOM, [621] vLLM KV-cache failure, [633] and [645] both crashing at save on an invalid
GenerationConfig, with [645] the first run to show the truncation-masking bug fixed).

The four rejection-sampling generation runs ([398] killed as too slow, [458] a vLLM startup OOM, [483]
crashed after 2 h 10 m in the agent's own `int(inf)` filter, [516] the successful k=4 rerun) are data
construction, not candidates; they are recorded in exp-03's `setup.data[0].selection`.
