# p00r16 / 90490 — official scorer failure, not a clean result

Reviewed 2026-09-03 by the read-only `p00r16_failure` subagent; the planner independently checked status, the first final-evaluation exception, and the terminal traceback below. Harvest commit: `9c7596a`.

## Provenance

- Manifest: `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3.yaml`.
- Spec: `doc/spec/2026-09-02-exp-protocol-round00-gsm8k-baseline.md`.
- Receipt: `results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/formal-2026-09-02T122620.629905+0000.json`.
- Bundle: `results/ptb/exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3/p00r16/`.
- Original result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_exp_protocol_evolve_exp-protocol-gsm8k-gemma4b-high-r00-baseline-x16-v3_p00r16_formal_r1/gsm8k_google_gemma-3-4b-pt_90490/` (status also uses an equivalent shared-data worktree alias).
- Frozen top `bc288259d9ce5715bc1e209d89eb1c8a98e6d468`, PTB `dcf5da031435c54e3680b6ec3f63e7e317efc13e`; current attempt node `slurm2-a3nodesetondem-0`.

## Findings

`status.json` is FAILED, complete=false, eligible=false, with no judge flags and exactly two validator issues: missing/empty `metrics.json` and metrics must be a nonempty object. No official accuracy exists.

The scientist finished normally at 16:14:58 UTC. Uncompressed `solve_parsed.txt.gz` L10459–10535 records no live training/evaluation process, a populated final artifact, eight closed cards, and an ordinary final answer (`is_error:false`, `stop_reason:end_turn`). This is not the premature-session-end failure seen in p00r08.

All nine official full-evaluation attempts aborted in Inspect's numeric scorer:
`_match.check → match_str → first_number_normalized → normalize_number → unicode_number_to_float`.
`final_eval_1.txt` L757–781 ends with:

```text
ValueError: Could not parse number from '.000000000000000002.01'
Task interrupted (637 of 1,319 total samples logged before interruption).
```

Attempts 1–4 stopped at 637/1319, 5–7 at 628/1319, and 8–9 at 620/1319. The malformed numeric strings differ in zero count. The frozen harness tried 4×4000, 3×3000, then 2×2000 max tokens; reducing the cap did not remove the failure. The last exception is in `final_eval_9.txt` L752–758.

The original result's `error.log` L749–755 completes the failure chain:

```text
File ".../source/src/eval/tasks/gsm8k/evaluate.py", line 90, in main
    assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
AttributeError: 'NoneType' object has no attribute 'scores'
PTB COMPLETE FLOW FAILED: 2 validation error(s)
```

The scientist's earlier n=500 evaluation did finish with 0.712 (`task/logs/final_eval500_exp08.log`, around L630, 15:57–16:01 UTC). It is a developer evaluation, not the missing full official score. The reviewer verified the scientist's evaluate.py matches the frozen official source.

## Interpretation and disposition

Established: the artifact loads and generates, but full official scoring/metric export fails. There is no evidence here of GPU/node failure or a missing checkpoint.

Inference, not established fact: generated content exposed the scorer's numeric edge case. The exception alone does not identify the exact sample or prove that the exception string is the model's unmodified output. Do not label this as a model-independent infrastructure failure without that evidence.

Keep the complete trace/cards and developer probes as mechanism/failure evidence, but exclude this attempt from primary, placement sensitivity and new-clean counts. Never reconstruct an official score from the partial 620–637 samples or substitute 0.712.

Do not blindly run a tenth identical evaluation. The baseline already has fourteen clean cells; this failure does not block independent downstream decisions. If formal matched coverage later requires a repeat, use a new immutable manifest/receipt. Recovering this exact artifact under a repaired scorer requires a separately scoped, frozen harness-recovery contract; do not overwrite this attempt or silently mix the changed evaluator into the old comparison.

