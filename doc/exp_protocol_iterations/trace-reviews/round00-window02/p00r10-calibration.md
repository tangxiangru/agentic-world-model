```yaml
cell: p00r10
arm: protocol
accuracy: 0.6542835481425322
hours_used: 7.77
hours_to_first_train_launch: 0.25
protocol_hours: 0.07
waiting_hours: 6.87
greedy_shipped: no
rl_used: no
rft_tried: yes (supported)
largest_eval_n: 500
stop_reason: "Batch complete, all processes finished, GPU free, deliverable verified, with 2:13 left"
top_contributors: ["termination-format SFT", "removing terse raw GSM8K targets", "fresh unseen-problem data"]
one_protocol_change: "CALIBRATION corroboration only: add the two-NEW-cell preflight decode audit; this cell noticed stock sampling but never tested or shipped greedy."
knowledge_to_transfer: []
```

**CALIBRATION — exclusion rule.** p00r10 is an already-reviewed calibration cell, not a new window-02 observation. It is excluded from new-cell counts and proposals below; it is used only to check whether a pattern independently proposed from NEW p00r14 and NEW p00r15 also appears here. Official accuracy 0.6542835481425322 is 0.03428 below the current protocol clean mean and 0.10144 below the null-control mean. The trace's internal final n=500 value was 0.682, which is not a substitute for the official accuracy (2026-09-02T23:07:19Z, L8587-L8598).

## 1. Timeline, hours, and stop

The session SHOWS a start at 2026-09-02 15:21:24Z and end at 23:07:41Z, 7.77 h later (2026-09-02T15:21:24Z, L6; 2026-09-02T23:07:41Z, L8660). The timeline classifier's 15:31 marker was a six-step smoke; the first real card training was exp-02 at 15:36:30Z, 0.25 h after start (2026-09-02T15:31:19Z, L3146; 2026-09-02T15:36:30Z, L3793).

Stage sequence SHOWS: exp-01 baseline at 15:24; exp-02 initial SFT at 15:36; exp-03 style/data retraining at 16:47; RFT sampling after exp-03; exp-04 RFT-mixture training at 19:28; exp-05 fresh-data continuation at 20:40; exp-06 repeat-problem continuation at 21:50; final n=500 validation at 23:00; completion at 23:07 (2026-09-02T15:24:04Z, L1512; 2026-09-02T15:36:30Z, L3793; 2026-09-02T16:47:58Z, L5696; 2026-09-02T18:51:06Z, L6124; 2026-09-02T19:28:41Z, L6656; 2026-09-02T20:40:05Z, L7222; 2026-09-02T21:50:41Z, L8100; 2026-09-02T23:00:57Z, L8474).

The timeline reader attributes 6.87 h to waiting on runs, 0.18 h to launch calls, 0.09 h to data work, 0.07 h to evaluation, 0.07 h to protocol tooling, about 0.01 h to other shell work, and 0.41 h to model-generation gaps. Long intervals were monitored training/sampling/eval rather than unexplained absence: exp-03 ran from 16:47 to its recorded decision at 18:51, after which RFT was launched (2026-09-02T16:47:58Z, L5696; 2026-09-02T18:51:06Z, L6123-L6124).

The last timer SHOWS 2:13 remaining (2026-09-02T23:07:26Z, L8656). The scientist SAID: “Done. All processes finished, GPU free, deliverable in place.” (2026-09-02T23:07:41Z, L8660). The stop was deliberate after n=500 artifact validation and all-card closure, not timer exhaustion.

## 2. Recipe decisions and RL/RFT/risk/budget reasoning

Exp-02 used 58,943 TRAIN-derived rows: OpenMathInstruct-2 integer-answer, single-box solutions plus repeated raw GSM8K train solutions; 8% carried the exact 10-shot prefix. It used full FT for one epoch at lr 1e-5, 8,192-token batches, accumulation 4, max length 3,072, cosine/3% warmup, bf16, and completion-only targets ending `ANSWER: <integer><end_of_turn>` (2026-09-02T15:34:50Z, L3670-L3716). This moved dev-150 0.0467 to 0.5333 and fixed termination, but its post-hoc style split exposed a new problem (2026-09-02T16:45:35Z, L5237-L5251).

Exp-03 removed raw human GSM8K targets and used 125,427 verbose OMI2 solutions, 9% with the exact 10-shot prefix, at lr 1e-5 for one epoch/max length 3,200 (2026-09-02T16:47:46Z, L5591-L5618). The reasoning was that real-GSM8K surface form, not the prefix, triggered terse calculator-annotation style; retraining on verbose answers for the same questions should invert that mapping (2026-09-02T16:47:46Z, L5591-L5592). It moved 0.5333 to 0.6133 and terse output from 90/150 to 0/150 (2026-09-02T18:51:06Z, L6104-L6118).

No policy-gradient RL was used. Exp-04 tried rejection-sampling FT: 38,006 verified self-samples from k=6/T=1 draws over GSM8K TRAIN and OMI2 questions, plus 32,006 fresh augmented rows; 9% carried the grader prefix; one epoch at lr 5e-6 (2026-09-02T19:28:30Z, L6525-L6574). The scientist explicitly rejected a second higher-temperature RFT sample because ~30 minutes of extra sampling would crowd out training and two evaluations (2026-09-02T19:28:30Z, L6525). RFT was marked supported on a +4-point n=150 result, while acknowledging it was about one standard error (2026-09-02T20:38:12Z, L6857-L6871).

Exp-05 continued on 64,832 previously unused augmented-GSM8K rows and gained +5.3 points (2026-09-02T21:47:29Z, L7576-L7590). Exp-06 tested 70k additional solution paths to already-seen problems; with 3.4 h left, the scientist rejected a second RFT round and reserved time for n=500 verification, while keeping exp-05 already installed as fallback (2026-09-02T21:49:49Z, L7980-L8000). Exp-06 regressed 0.7067 to 0.6533, so the falsification rule correctly rejected repeat-problem scaling (2026-09-02T23:00:09Z, L8305-L8319).

## 3. Decode configuration

The calibration cell did **not** ship greedy. It inspected the base `generation_config.json` and SHOWED `do_sample:true` at 15:24 (2026-09-02T15:24:42Z, L1562-L1569), but its save/finalize path intentionally copied that base generation config verbatim for EOS ids (2026-09-02T15:30:49Z, L3071-L3072). Its separate probe used temperature 0, but there was no controlled `evaluate.py` decode card and no write of `do_sample:false`; the cell reader found zero greedy-config writes. Final validation therefore tested the inherited sampled configuration, not greedy (2026-09-02T23:00:47Z, L8369-L8380).

This calibration observation corroborates, but is not counted toward, the NEW-cell proposal: both p00r14 and p00r15 independently showed that explicitly shipping greedy materially improves identical weights. p00r10's lack of a decode card plausibly left accuracy on the table, but the calibration trace itself does not measure the counterfactual.

## 4. Evaluation practice

Intermediate decisions were made at n=150, with standard error about .040; the scientist explicitly said exp-04's +4 points was about 1 sigma and not individually separable from noise (2026-09-02T23:07:41Z, L8683-L8687). The final artifact alone was checked at n=500 and scored 0.682 ± .021, versus exp-05's 0.7067 at n=150; the scientist treated those as the same model within sampling error (2026-09-02T23:07:19Z, L8587-L8598).

There was no larger-n candidate-ranking inversion because only the selected exp-05 artifact received n=500 evaluation. There were no reported paired McNemar statistics. The strongest item-level diagnostic was the n=300 held-out OMI2 few-shot/zero-shot probe that separated question-surface style from prefix effects; it was not a test-set substitute (2026-09-02T16:45:35Z, L5249-L5251). The cell reader found two direct `evaluate.py` calls (baseline n=150 and final n=500), seven inspect logs, and no home-grown test-set evaluator; the final deliverable was explicitly verified via shipped `evaluate.py` (2026-09-02T23:07:41Z, L8664).

## 5. Pitfalls and avoided traps

Every recorded loss of at least 0.1 h was:

- 0.15 h: fixed batch size 8 OOMed on Gemma's 262k-vocab fp32 logits; replaced with an 8,192-token-budget sampler (2026-09-02T15:34:50Z, L3670-L3671).
- 0.10 h: vLLM 0.11 rejected the old `prompt_token_ids=` API; probe and sampler were updated to structured prompt inputs (2026-09-02T16:47:46Z, L5571-L5572).
- 0.10 h: that crash left a 66 GiB orphaned `EngineCore`, so the scientist killed the compute PID before retrying (2026-09-02T16:47:46Z, L5573).
- 0.15 h: RFT sampling took 34 rather than 25 minutes because vLLM counted n=6 as six scheduled prompts; future sizing was changed to questions times k (2026-09-02T19:28:30Z, L6527-L6528).
- 0.25 h: a full 14-shard fresh-data scan produced only 81 rows under the three-solutions-per-problem cap; rejection counts were inspected and the cap raised to six (2026-09-02T21:49:49Z, L7999-L8000).

These total the card-reported 0.75 h. Avoided traps included filtering any target containing `####` or `ANSWER:` to preserve one answer marker (2026-09-02T15:34:50Z, L3705-L3708), embedding some exact grader-prefix rows rather than assuming zero-shot equivalence (2026-09-02T15:34:50Z, L3707-L3716), preserving token 106 in final generation config and CPU-loading the assembled model (2026-09-02T23:00:47Z, L8369-L8380), and using exp-05 as an already-written fallback before the risky final continuation (2026-09-02T21:47:29Z, L7588-L7590).

## 6. Protocol interaction, cost, and overrides

The timeline reader attributes 0.07 h (~4.2 minutes of command execution) to protocol tools. Observable create-to-lock spans, including card writing/check/preflight, were roughly: exp-01 1.1 min, exp-02 1.6 min, exp-03 1.0 min, exp-04 1.1 min, exp-05 1.0 min, exp-06 1.0 min, exp-07 0.7 min (2026-09-02T15:22:49Z, L1075; 2026-09-02T15:23:56Z, L1483; 2026-09-02T15:34:50Z, L3493; 2026-09-02T15:36:26Z, L3770; 2026-09-02T16:46:52Z, L5523; 2026-09-02T16:47:52Z, L5673; 2026-09-02T19:27:30Z, L6440; 2026-09-02T19:28:35Z, L6634; 2026-09-02T20:39:03Z, L7018; 2026-09-02T20:40:00Z, L7201; 2026-09-02T21:49:40Z, L7923; 2026-09-02T21:50:41Z, L8100; 2026-09-02T23:00:09Z, L8324; 2026-09-02T23:00:51Z, L8461). Close calls were generally seconds after result writing (2026-09-02T23:07:19Z, L8605-L8609).

Bootstrap cost about 1.4 minutes to the first card after reading the mandatory protocol/pitfalls (2026-09-02T15:21:24Z, L6; 2026-09-02T15:22:49Z, L1075). The format shaped a sequence of comparator-gated interventions: exp-03 abandoned terse raw GSM8K targets after a probe isolated their effect (2026-09-02T16:45:35Z, L5249-L5251), exp-06's explicit falsification threshold rejected repeat-problem scaling (2026-09-02T23:00:09Z, L8315-L8319), and exp-07 forced end-to-end artifact validation rather than assuming copied files would load (2026-09-02T23:00:47Z, L8371-L8380).

The main cost was serial card overhead plus repeated n=150 comparisons, but it was small relative to 6.87 h of run waiting. Unlike the null arm's typical self-imposed notes, this protocol cell locked all seven pre-launch plans and closed result/verdict sections; its principal weakness was not enforcing an explicit decode-choice card despite seeing stock `do_sample:true` (2026-09-02T15:24:42Z, L1562-L1569).

## 7. Verdict, contributors, and one protocol change

Verdict: a below-mean **CALIBRATION** cell with strong experimental documentation but an important untested decode lever. Its three largest positive contributors were:

1. Termination-format SFT: 0.0467 to 0.5333, with every completion ending in one `ANSWER:` and `<end_of_turn>` (2026-09-02T16:45:35Z, L5237-L5249).
2. Removing terse raw GSM8K human solutions: 0.5333 to 0.6133 and 90/150 terse outputs to 0/150 (2026-09-02T18:51:06Z, L6104-L6118).
3. Fresh unseen-problem data, including supported RFT/+fresh mixture then 64.8k new augmented rows: 0.6133 to 0.6533 to 0.7067 at n=150 (2026-09-02T20:38:12Z, L6857-L6871; 2026-09-02T21:47:29Z, L7576-L7590).

The largest negative contributors were the absence of a greedy decode test and the final repeat-problem continuation, which lost 5.3 points and about 1.1 h before being rejected (2026-09-02T23:00:09Z, L8305-L8319). Compared with the null arm, this cell used seven locked comparator cards, explicit data lineage, stop/format diagnostics, a fallback incumbent, and a final artifact-only verification (2026-09-02T23:07:41Z, L8664-L8689).

**One protocol change, calibration-only corroboration.** Do not count p00r10 as a proposal source. It corroborates the proposal already supported by both NEW cells: preflight should require an explicit effective decode choice before the first post-training eval. Here the scientist inspected `do_sample:true` but copied it through to the final model without a controlled greedy benchmark (2026-09-02T15:24:42Z, L1562-L1569; 2026-09-02T15:30:49Z, L3071-L3072). No p00r10-only change is proposed.
