# Reconstructed experiment cards — r-d46f982f

Base model post-trained: `google/gemma-3-4b-pt` · benchmark: gsm8k · budget: 10 h, 1x H100.
12 launches, in launch order. Adopted / submitted: **exp-12** (`final_model`, a copy of the exp-10 checkpoint soup).

| card | launch_i | elapsed_h | family | parent | datasets | lr/epochs | execution | best measurement | verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| exp-01 | 629 | 0.24 | sft | base_model | sft_v2.jsonl (115,000 = OpenMathInstruct-2 gsm8k/augmented ≤2 per problem + gsm8k train CoT x3) | 1e-5 / 1 | completed | 0.665 @ limit 200 greedy · 0.520 @ limit 200 sampling | inconclusive | adopt |
| exp-02 | 1062 | 2.22 | decode-config | exp-01 | — | — | completed | 0.665 @ limit 200 (+14.5 pts vs the same weights sampling) | supported | adopt |
| exp-03 | 1323 | 2.90 | grpo | exp-02 | problem_bank.jsonl (first 3,600 problems, x8 rollouts) | 2e-6 / 1 | failed | — (crashed at step 25/150) | inconclusive | abandon_line |
| exp-04 | 1432 | 5.80 | other (config fix + packaging) | exp-02 | — | — | completed | — (final_model never evaluated in this state) | inconclusive | adopt |
| exp-05 | 1437 | 5.89 | grpo | exp-04 | problem_bank.jsonl (first 2,400 problems, x8 rollouts) | 2e-6 / 1 | killed | 0.735 @ limit 200 · 0.715 @ limit 400 (ckpt-40) | supported | adopt |
| exp-06 | 1526 | 7.66 | other (packaging) | exp-05 | — | — | completed | — (fp32 copy never evaluated) | inconclusive | adopt |
| exp-07 | 1564 | 7.67 | grpo (continuation) | exp-05 | problem_bank2.jsonl (rows 2400:8000, first 960 used, x8 rollouts) | 1.5e-6 / 1 | killed | 0.680 @ limit 200 (ckpt-15) · 0.675 (ckpt-30) | contradicted | reject |
| exp-08 | 1568 | 7.68 | other (bf16 repackaging) | exp-06 | — | — | completed | 0.7125 @ limit 400 | inconclusive | adopt |
| exp-09 | 1689 | 8.75 | merge | exp-05 | — (3-way soup: grpo1-40, grpo1-60, grpo2-15) | — | completed | — (score never printed) | inconclusive | reject |
| exp-10 | 1711 | 8.80 | merge | exp-05 | — (2-way soup: grpo1-40, grpo1-60) | — | completed | 0.730 @ limit 400 | supported | adopt |
| exp-11 | 1725 | 8.86 | merge | exp-05 | — (soup of grpo1-20/40/60, arity unconfirmed) | — | completed | — (score never printed) | inconclusive | reject |
| exp-12 | 1736 | 8.91 | other (packaging) | exp-10 | — | — | completed | 0.7375 @ limit 400 · 0.720 @ limit 150 defaults | supported | adopt |

The line that ships runs exp-01 → exp-02 → exp-05 → exp-10 → exp-12: one epoch of full-parameter SFT
on 115k GSM8K-style CoT rows, a greedy `generation_config` shipped with the weights, 60 steps of GRPO on
exact-answer reward, a uniform average of two RL checkpoints, and a copy into `final_model`. The
measured chain, in the agent's own closing table [1759]: 5.3% base (n=150) → 52.0% SFT under default
sampling → 66.5% greedy (both n=200) → 71.3% after RL → 73.75% for the soup (both n=400).

Six cards carry `adopt`. Four of them (exp-04, exp-06, exp-08, exp-12) are the successive writes to
`final_model`; exp-01/exp-02 and exp-05 and exp-10 are the training, decode and merge cards those
copies packaged. Only exp-12 survives to the end of the digest — the copy was made at [1736] and the
directory was verified at 8.1G with an idle GPU at [1757], with 0:59 of the budget still on the timer,
so nothing rules out a later change.

Comparators are recorded per card and only three cards support a verdict. exp-02 (greedy vs sampling,
+14.5 pts), exp-05 (RL vs SFT, +7.0 pts) and exp-07 (continuation vs its own parent, −6.0 pts) are all
measured at limit 200; exp-10 (+1.5 pts over its best ingredient) and exp-12 (+2.5 pts over the
checkpoint it replaced) at limit 400. exp-01 is `inconclusive` despite the 47-point jump because the
base-model baseline was measured at limit 150 and the SFT model at limit 200. The two soup margins are
inside the ~2.2 pt standard error the agent quotes for a 400-item eval [1759], and neither was
re-measured.

Four smoke runs precede exp-01 ([502], [507], [527], [570] — the last killed at [608]) and three
precede exp-03 ([1169], [1190], and [1255], which passed at ~120 s/step with reward ~0.55). They are
recorded on those two cards, not as cards of their own.

Two prepared lines were never launched and so have no cards: `gen_rft.py`, the rejection-sampling
generator written at [650], was never run, and `data/sft_v3.jsonl`, a 200k mixture adding MetaMathQA-GSM
built at [927] and rebuilt at [962] after orca-math was dropped for unreliable answer labels, was never
trained on.

Run-level gaps: `data/problem_bank.jsonl`, the source of every RL prompt, is never built in the digest;
the `runs/` directory holding every eval JSON is not part of the workspace snapshot, so no measurement
path can be opened; and most eval result blocks are absent, so the accuracies come from the agent's own
quotes rather than from the JSON. Two evaluated candidates (exp-09, exp-11) have no score anywhere in
the stream. Roughly 2 h of the 10 h budget was lost to the exp-03 crash [1759].
