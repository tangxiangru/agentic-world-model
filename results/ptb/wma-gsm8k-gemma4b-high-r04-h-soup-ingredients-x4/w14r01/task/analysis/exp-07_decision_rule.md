# exp-07 candidate rule — written 2026-09-04T15:40Z, BEFORE the run started

The card's claim is about the ANNEALED end point of a one-epoch cosine schedule.
Therefore the single candidate of exp-07 is ckpts/exp-07/final (step 1484,
lr annealed to 0). checkpoint-1200 is written only as mid-run insurance and is
NOT a candidate; it will be read at --limit 150 only if final fails to load or
the full read is impossible in the remaining time.

Adoption rule, fixed in advance:
  Package ckpts/exp-07/final into final_model ONLY IF its n=1319 greedy accuracy
  is strictly greater than the best n=1319 greedy accuracy measured in exp-06.
  A tie or a loss means the exp-06 winner stays. scripts/paired_compare.py is
  run against the exp-06 winner's log and reported either way, but per the
  exp-06 rule it does not override the scalar comparison.
