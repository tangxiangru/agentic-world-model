# Packaging rule for final_model/ — fixed 2026-09-03T20:20Z, before exp-04 was scored

Written in advance (WMA precondition on exp-04) so the choice cannot be made
after seeing a favourable draw.

Candidates: `ckpts/exp-02/final` (incumbent, dev-150 0.7533) and whatever
exp-04 produces.

1. Both candidates are scored on the **full test split** (`--limit -1`, 1319
   items, `--max-connections 16`) under otherwise identical settings. The
   full-set read decides, because dev-150's noise floor (~0.035) is larger than
   the effect exp-04 is expected to have.
2. `final_model/` gets the exp-04 checkpoint **only if** its full-set accuracy
   is at least as high as exp-02/final's. Ties go to exp-02/final (the smaller
   claim). Otherwise exp-02/final ships.
3. If the full-set evals cannot both be completed in the time left, the
   decision falls back to the paired item-flip count on dev-150: exp-04 ships
   only if it has strictly more wrong->right than right->wrong flips against
   exp-02/final's log.
4. The shipped directory carries the checkpoint's own `generation_config.json`
   unmodified (exp-03 found greedy and the inherited sampler indistinguishable,
   so there is no measured reason to intervene), real files rather than
   symlinks, and the tokenizer/processor files. It is loaded once from a fresh
   process before the deadline.
