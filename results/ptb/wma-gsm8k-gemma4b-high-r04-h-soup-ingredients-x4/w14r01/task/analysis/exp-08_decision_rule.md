# exp-08 adoption rule — written 2026-09-04T17:05Z, BEFORE the merge was built

Candidate: ckpts/exp-08_soup4 = uniform average of
  ckpts/exp-04/checkpoint-742, checkpoint-1484, checkpoint-2226, ckpts/exp-07/final

Adopt (overwrite final_model, which currently holds ckpts/exp-05_soup at 0.7354)
ONLY IF the candidate's greedy n=1319 accuracy is STRICTLY GREATER than 0.7407,
the best n=1319 number measured this session (ckpts/exp-04/checkpoint-1484).
Anything else leaves final_model untouched.

Recorded selection cost: this is the 5th distinct artefact read at n=1319. With
five reads the winner's curse on the maximum is roughly +0.5 pp, so the strict
inequality against the running maximum (not against the packaged artefact) is
deliberately the harder bar.
