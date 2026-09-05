# exp-08 swap rule — pre-committed before the run

Written 2026-09-04T08:12Z, before `scripts/train_sft.py` was launched.

`final_model/` currently holds the exp-06 soup with the greedy generation_config
(0.7140 at n=500, 0.7267 under the harness default invocation). The best n=500
number any candidate has produced is 0.7180 (exp-04_greedy).

exp-08's checkpoint replaces `final_model/` **only if** its accuracy at n=500
under the identical protocol is **>= 0.738** (0.7180 + 2.0 pp). Anything below
that is inside the noise this session has already measured twice — identical
weights moved 6 items on the same 150 between two runs — and the incumbent stays.

If training does not finish, or finishes too late to be evaluated at n=500, the
card closes as `killed`/`inconclusive` and `final_model/` is left untouched.
No candidate is ever shipped on a 150-item read.

## Amendment (08:20Z, still before launch) — WMA precondition 4

0.738 is exactly the n=500 noise floor above 0.718, so the bar is refined:

- `acc >= 0.745` → overwrite `final_model/`.
- `0.738 <= acc < 0.745` → overwrite ONLY if a `--limit 1319` confirmation read
  also clears 0.718; if there is not enough time left for that read, the
  incumbent stays. Ambiguity resolves in favour of the already-verified model.
- `acc < 0.738` → incumbent stays.
