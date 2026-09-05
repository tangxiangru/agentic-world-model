# Pre-committed decision rule for exp-05

Written 2026-09-03 ~20:22 UTC, after `awm exp_protocol lock exp-05` returned and the
training command was launched, before any exp-05 eval number exists.
Recorded because the exp-05 verdict's third precondition asks for it in writing.

Comparator: exp-04, `ckpts/exp-02/ep1_greedy`, **0.800** at n=150 greedy
(`eval/exp-04_ep1_dev150.json`), stderr 0.0327.

Every exp-05 checkpoint is exported through `scripts/export_ckpt.py --greedy`
before `evaluate.py`, so both arms are deterministic bf16 reads. An unexported
read would be a sampling read against a greedy comparator.

| dev-150 greedy result | reading | action |
|---|---|---|
| `>= 0.833` (+3.3 pts, ≥ 1 stderr) | real improvement | adopt exp-05 as final_model |
| `0.807 – 0.827` (+1 to +3 pts) | inside the n=150 floor | do **not** decide on the point estimate; run a `--limit 500` head-to-head of exp-05/final vs `ckpts/exp-02/ep1_greedy` and adopt the winner there |
| `0.780 – 0.800` | flat | keep the incumbent `ep1_greedy`; record that solution-diversity scaling bought nothing at this size |
| `< 0.780` | worse | keep the incumbent; hypothesis falsified as written |

Ties at `--limit 500` (within 1.5 pts) resolve to the incumbent `ep1_greedy`,
because it is the checkpoint with the older, independently reproduced number.

Checkpoint policy: exp-04's finding was that trained-to-the-end is not
automatically the best point, so `checkpoint-1500` is exported and held. Scoring
it is a separate card and only happens if the clock allows after the head-to-head.
