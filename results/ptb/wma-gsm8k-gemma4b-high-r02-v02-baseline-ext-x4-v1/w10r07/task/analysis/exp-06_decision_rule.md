# exp-06 decision rule — fixed before any n=500 number existed

Written at 2026-09-04T07:56Z, immediately after the two `--limit 500` evals were
launched and ~20 minutes before either result file was written. The WMA verdict
on exp-06 asked for this (precondition 1 and 3); it is kept outside
`memory/cards/exp-06.yaml` because that card is already locked and a rule added
to a locked plan after the fact is not a pre-commitment.

## Quantity

Not the two per-candidate accuracies (stderr ~0.018 each at n=500) but the
**paired gap** on the same 500 items. The WMA estimates the paired sd at 0.016
from a measured 12.7 % item-level disagreement between the two candidates.

## Rule

`final_model/` currently hash-matches `ckpts/exp-04_greedy`.

- Ship **`ckpts/soup2`** only if `acc(soup2) - acc(exp-04_greedy) >= +0.02` at
  n=500 (≈1.25 paired sd, and a reversal of the n=150 ordering by more than one
  paired sd).
- Otherwise ship **`ckpts/exp-04_greedy`**, including when the gap is positive
  but under 0.02, and including when it is negative.

The asymmetry is deliberate: exp-04_greedy already leads under the n=150
protocol (0.7933 vs 0.7733) and is a single trained checkpoint rather than a
merge, so it is the conservative default and the burden of evidence is on the
challenger.

## What this rule is not

It is not a prediction. exp-06's hypothesis says the n=500 gap should exceed
0.02 in *either* direction; if it does not, the card is `contradicted` /
`inconclusive` on its own terms and this rule still decides what ships.
