# Opus 4.8 WMA quota / sidecar health snapshot

Snapshot basis: repository `peek.json` files last refreshed at 2026-09-04 23:34:21–23:34:24 UTC, plus the already-harvested `w57r01`. This audit did not read any transcript or `solve_out` content. In-flight observations below are operational health signals only, not scientific results, validator-clean cells, or promotion evidence.

## Terminal evidence: harvested `w57r01`

- Slurm job `92198` is terminal `FAILED`; PTB harvest is incomplete (`complete=false`, no final model/config or metrics) and has `general_anomaly`. There is no valid scientific score from this cell.
- The WMA sidecar itself reached `state=completed`, but all three card reviews failed: `exp-01`, `exp-02`, and `exp-03` each have public `completion.json` state `failed`, no verdict, and an explicit HTTP 429 / `RESOURCE_EXHAUSTED` error for `online_prediction_input_tokens_per_minute_per_base_model` on `anthropic-claude-opus-4-8`.
- Review count: **0 successful, 3 failed, 3/3 explicit 429**.
- Each failed call reports one turn, USD 0.0, and 3.112–3.164 wall minutes; total review wall time **9.411 min**, reported cost **USD 0.0**.

Evidence: `results/ptb/wma-crossbench-opus48-r05-gsm8k-single-x4/w57r01/status.json`, `task/.wma/sidecar_status.json`, public `task/.wma/reviews/*/*/completion.json`, and `wma_private/reviews/*/*.measurement.json`.

## In-flight health signals

The peek snapshot marks nine WMA-bearing cells RUNNING: `w57r02`–`w57r04`, `w58r01`–`w58r04`, and `w59r01`–`w59r02` (jobs `92199`–`92207`).

- Ten measurement files exist across six cells: `w57r02` (`exp-01`, `exp-02`), `w57r03` (`exp-01`–`exp-03`), `w57r04` (`exp-01`, `exp-02`), and `w58r01`–`w58r03` (`exp-01`).
- These ten attempts have the same observable fingerprint as the three terminal 429 failures: model `claude-opus-4-8`, effort `high`, one turn, USD 0.0, wall time 2.964–3.214 min, and a 5,451-byte transcript listed by `peek.json`. Their aggregate wall time is **30.815 min**, reported cost **USD 0.0**.
- Therefore the snapshot provides **10 strong 429-like signals**, involving six cells and ten cell/card pairs. The in-flight peek bundle does not include public completion records, so the terminal review outcome cannot yet be verified. Count of in-flight reviews with verified success: **0**; count with terminal-verifiable failure: **0**. Do not promote the ten signals to confirmed failures until harvest exposes their completion state.
- Twelve request directories are visible. Ten have measurements. Two have frozen inputs but no measurement: recent `w58r04/exp-01` request `20260904T233139.745111Z-6431f5ac`, and older `w57r02/exp-01` request `20260904T213719.871658Z-8fef55fe`. The latter was followed by a later measured `exp-02` request, so its current execution state is indeterminate from the snapshot. `w59r01` and `w59r02` have no visible review request or measurement yet.
- All ten measured attempts' current card YAML, lock, and preflight files match the SHA-256 entries in the measurement input inventory. Each inventory contains 519–521 files. Measurements report no outside-file access and `leak_suspected=false`; because these calls show the quota-failure fingerprint, that is only an isolation-accounting observation, not evidence of a substantive WMA review.

## Operator conclusion

The only terminally proven condition is persistent Opus 4.8 input-token quota exhaustion in harvested `w57r01` (three failures spanning 20:40–23:11 UTC). Ten later in-flight measurements closely reproduce the same timing/cost/turn/file-size fingerprint, so quota exhaustion remains the leading operational diagnosis. No successful Opus 4.8 WMA review is verified in this scoped evidence, and none of the in-flight cells should be counted as a scientific result.
