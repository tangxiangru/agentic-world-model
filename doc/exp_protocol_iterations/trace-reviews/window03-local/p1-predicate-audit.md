# Focused P1 predicate audit — window 03

Read-only reviewer: Codex subagent `p00r16_failure` (follow-up task). The planner independently checked the frozen entry, E wording, and selected raw loss logs. No training/evaluation or queue mutation was performed.

## Frozen predicate

`git show a4c4954:skills/exp_protocol/pitfalls.yaml`, lines 65–70, describes more than low RFT loss: the same sampled/trained parent, starting loss within **about 0.05** of the parent's terminal loss, little movement in the first 20 optimizer steps, and a prediction of no gain at n≥500 (or a second-round decline). The distinction between first and second self-distillation is explicit. The screening spec's shorthand “flat first loss” is not a substitute for this conjunction.

| stage | data/lineage | parent terminal loss | early logged loss | endpoint | adjudication |
|---|---|---:|---|---:|---|
| g01r01 exp-04 | self plus unseen teacher; sampling/training from exp02/final | 0.2729 | steps 5/10/15/20: 0.2581, 0.2315, 0.2301, 0.2268 | 0.2133 | mixed data and actual early descent; not a confirmed P1 hit |
| g01r02 exp-04 | self-only; exp-03 weights hardlinked to sampling parent exp-02, decode config differs | 0.3000 | steps 10/20: 0.2461, 0.2394 | 0.2238 | first difference 0.0539, borderline to “about 0.05”; not an unambiguous matched-score counterexample |
| g01r02 exp-07 | second self-only round, sampled/trained from exp-04/final | 0.2238 | steps 10/20: 0.2054, 0.2052 | 0.2065 | clearest lineage/loss-signature exposure; 0.32 h training, but no parent/child n≥500 comparison |

## Raw evidence and corrections

- **g01r01 data mix:** the exp-04 card's component counts are wrong. Trace L6209–6230 (11:10:18–20 UTC) merges 67,544 self and 40,000 teacher rows; L6353–6355 (11:11:27) retains 67,544 and 39,757, then removes five malformed rows. The raw final data source counts inspected by the reviewer are **67,539 self + 39,757 teacher = 107,296**. This is not self-only RFT.
- **Early losses are available:** `results/ptb/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8-v1/g01r01/task/logs/exp-04.log` lines 6–9 preserve the four five-step loss summaries, and `train_sft.py` uses logging_steps=5. The trace's short tails do not show these values, but the harvested log does. Do not promote “not printed in the trace” into “unobservable”; not every individual-step loss is retained.
- **g01r01 score/behavior:** exp-05 records n=500 child 0.724 versus parent 0.720 (two items), and fewer runaways. This neither establishes a score gain nor a no-learning result. The later exp-07 is reformatted-target SFT on mixed data, not a self-only RFT counterexample.
- **Terminal loss is not average train_loss:** g01r02 `task/logs/exp-02.log` line 153 ends at logged loss 0.3000; the following 0.323328 is the whole-run mean. Comparing P1 to that mean changes the predicate.
- **g01r02 exp-04 lineage:** trace L5273–5287 (09:55:31) shows hardlinked parent tensors and a decode-config-only variant. Its successful rerun logs are in `task/logs/exp-04.log`; the reviewer report also quoted the first attempt, which failed at save. That lost run belongs to D, not a separate P1 failure.
- **g01r02 exp-04 score:** +2.0 points at n=150 and +4.0 on a 200-item train probe are developer observations. The later 0.816 at n=500 compares exp-04 with exp-05, not with the RFT parent; it is not a matched n≥500 refutation of P1.
- **g01r02 exp-07:** `task/logs/exp-07.log` lines 8–9 show flat early loss with nonzero gradients. Trace L8704–8744: lock 15:12:41, launch 15:12:46, step-20 read 15:15:16; L8823–8828 at 15:36:51 shows n=150 0.79333 and probe 166/200 versus parent 0.82667 and 169/200. The decline is observed; “lucky chains caused it” remains the scientist's hypothesis.

## Planner disposition

This audit neither validates the stop rule nor proves it false. It rejects the synthesis's wholesale attribution of g01r01's 1.06 h and both g01r02 rounds as P1-avoidable loss, and rejects its claim that the early-loss observable cannot be recovered. The clearest new exposure is only g01r02 exp-07, itself 0.32 h (<0.5 h), and the sampling before its first 20 steps cannot be counted as savings from stopping at step 20.

Do not treat low loss as zero gradient: these logs show grad_norm around 1–2. Do not count mixed teacher stages, average train_loss, or an unmatched later n=500 read as satisfying the full predicate. Keep unknown/borderline cases separate. The prescribed “≥2 guard cells with P1 signature” replacement choice is not established by these two cells; await the full strict cohort. No P1 tree or manifest was changed.

