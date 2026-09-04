# Pre-committed swap rule for exp-09 (weight soup)

Written 2026-09-03 ~23:37 UTC, after `lock exp-09` returned, before the soup is scored.
Recorded because the exp-09 verdict's first precondition asks for a paired rule rather
than "beats 0.7976" — the comparator (exp-08, 0.7976) is itself one noisy read, and
this session measured vLLM greedy read noise directly in exp-06.

Both arms are the full 1319-item test split, greedy, `--max-connections 16`.

`final_model/` is replaced by `ckpts/soup_a05` only if BOTH hold:

1. soup accuracy − 0.7976 ≥ **+0.010** (≥ 13 net items), and
2. the **paired** McNemar z of soup vs `analysis/exp-08_diag.json`, computed
   item-by-item as in exp-07, is **> 1.5** in the soup's favour.

Anything else — including a soup that wins by less than a point, or wins
unpaired but not paired — leaves `final_model/` exactly as it is
(the exp-05 weights, verified at 0.7976 in exp-08).

A soup that degrades termination below exp-08's 0.9955 is rejected regardless
of accuracy.
