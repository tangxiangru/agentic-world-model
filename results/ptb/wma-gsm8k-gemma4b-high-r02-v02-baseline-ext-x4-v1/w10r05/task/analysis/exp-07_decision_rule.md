# exp-07 decision rule — pre-committed BEFORE the soup arm was run

Written at 2026-09-04T07:45Z, after the comparator arm (exp-04_greedy, 0.7180 at
n=500) was in hand and before `evaluate.py --model-path ckpts/exp-06_soup_greedy
--limit 500` was launched. Recorded because exp-06 was faulted for letting the
noisiest measurement pick the arm after the fact, and because the WMA's first
precondition on this card is to fix the rule in advance.

## Facts the rule has to survive

- The disputed 150 items are a nested prefix of the 500, so the pooled number
  carries the disputed +6 items into itself by construction. The 350 items
  outside the prefix are the only genuinely new evidence.
- Identical exp-04 weights scored 106/150 at `--limit 150` and 110/150 on the
  same 150 items inside the `--limit 500` run — a 6-item swing from batching
  alone. So a gap of under ~2 pp at n=500 is not a ranking.

## The rule

Let `gap = acc(soup) - acc(exp-04)` on the pooled 500 items.

1. `gap >= +0.02` → ship the soup.
2. `gap <= -0.02` → ship exp-04 (final_model stays as it is).
3. `|gap| < 0.02` → not a ranking. Fall through to the 350 fresh items:
   - fresh-350 gap `>= +0.02` → ship the soup;
   - fresh-350 gap `<= -0.02` → ship exp-04;
   - otherwise → **ship the soup**, as the variance-reduced artefact: it is a
     uniform average of two independently trained fine-tunes of the same base,
     which is the lower-variance of the two candidates by construction, and no
     measurement separates them.

The falsifier in the card (soup ahead by more than 4.0 pts at n=500) is expected
to fire with ~5% probability and will carry almost no information either way;
the reported quantity is the paired difference and its interval, not the
falsifier.

## Guard

`final_model/` currently holds exp-04_greedy (0.7180 at n=500, 0.7067 at n=150).
It is not overwritten until the 500-item read is in hand, and the pre-overwrite
score is recorded above so the swap is reversible.
