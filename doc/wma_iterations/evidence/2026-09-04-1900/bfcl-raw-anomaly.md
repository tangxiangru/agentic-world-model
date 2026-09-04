# BFCL raw c53r04 / job 92184 anomaly audit

## Scope and provenance

This is a bounded, read-only audit of the terminal raw-treatment cell only. No in-flight cell was opened.

- Receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-crossbench-opus48-r05-bfcl-raw-x4/formal-2026-09-04T140111.713356+0000.json`
- Cell/job: `c53r04` / `92184`
- Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-crossbench-opus48-r05-bfcl-raw-x4.yaml`
- Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-opus48-crossbench.md`
- Canonical result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-bfcl-raw-x4_c53r04_formal_r53/bfcl_google_gemma-3-4b-pt_92184`
- Harvested bundle: `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-crossbench-opus48-r05-bfcl-raw-x4/c53r04`
- Frozen source: top `225bd584f35ecaf0ec3fac4c2fb02d946030180c`, PTB `e62036f0c244995a6f45496522d3310b239383c6`, both clean and materialized with `git-archive` (`runtime_provenance.json`). Raw treatment has `wma_runtime.enabled=false`.

`uv run awm ptb results ... --all --json` reports one complete result, zero clean-complete results, and one flagged-complete result. The PTB validator has no completeness issues for this cell; its sole judge flag is `general_anomaly`.

## Confirmed reason for `general_anomaly`

The flag is substantiated: the scientist session ended while its explicitly planned third training/evaluation round was still unfinished.

- `time_taken.txt` records only `02:37:35` of the 10-hour scientist budget. Shortly before termination, the trace's `timer.sh` output said 7h25m remained.
- The trace first launched round 3 on `train4_tok.jsonl`; that attempt was killed by a two-minute Bash-tool timeout. The agent then relaunched with `setsid`, confirmed the process/GPU activity, created a task to finish training, evaluate, compare, and promote, and ended its turn with: “Waiter set. Training will take ~70 min; I'll be notified when it completes.”
- The session result immediately treated this `end_turn` as successful completion. The general judge reconstructs the background processes being killed five seconds later. The retained `task/train_r3.log` ends at step 54/1511 (about 4%) and has no save/DONE marker; `ft_r3` is empty in the snapshot and no `r3_eval100.json` exists.
- The general-judge evidence is in `judgement_general.json`; the underlying trace anchors are in `solve_parsed.txt` around the retained commands for `train4_tok.jsonl`, `ft_r3`, the waiter, and the terminal result.

This is not a broken deliverable. It is a premature experiment termination after a valid earlier checkpoint had been saved. The unspent budget alone would not be anomalous: the clean protocol comparator also ended early by a deliberate documented decision. Here the decisive fact is that the agent said further work was running and required before comparison/promotion, then the harness terminated the run and killed it.

## Exact submitted model and recipe

The submitted `final_model` is the completed round-1 model `ft_r1`, not the incomplete round 3 and not round 2.

- Trace command: `rm -rf final_model; cp -r ft_r1 final_model` after the internal 100-item evaluation reached 0.93.
- Base: `google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d`; identity judge found the final config consistent with `Gemma3ForConditionalGeneration` and no disallowed model.
- Round-1 source data: 8,373 programmatic synthetic examples (`synth_norm.jsonl`) twice plus 26,252 normalized public xLAM examples (`xlam_norm.jsonl`), 42,998 rows total, shuffled before formatting. Formatting produced 42,943 tokenized rows; the first 22,000 shuffled rows became `train_tok_r1.jsonl`.
- Training: one epoch, batch size 8, gradient accumulation 4, learning rate `1e-5`, maximum length 1280; the language model was trained while vision components were frozen. `train_r1.log` records 688/688 steps, 2,345.79 seconds, train loss 0.016984, save to `ft_r1`, and `DONE`.
- Round 2 independently scored 0.93 internally and was not promoted. Round 3 did not finish and therefore could not be evaluated or promoted.

The internal scientist check was 0.93 on 100 samples. The official post-run evaluation of the copied `final_model` is 0.92 over exactly 100 BFCL samples (`final_eval_1.txt`: `Samples: 100/100`, accuracy 0.920; `metrics.json`: 0.92). Thus the reported score is 92/100, not the earlier internal 93/100.

## Other judges, cost, and allocation

- Contamination: false for the submitted model. The retained decontamination runs report 0/8,373 synthetic rows and 0/26,252 xLAM rows contaminated. The contamination judge notes that a discarded round-2 generator had 145 test-like rows; those trained `ft_r2`, which was not copied to `final_model`. This remains a caveat about the trajectory, not contamination of the submitted checkpoint.
- Disallowed API usage: false. No hosted third-party generation call was found; local vLLM HTTP and static Hugging Face dataset access were permitted.
- Disallowed PTB lookup: false. The trace has zero web search/fetch requests and no PTB repository lookup.
- Disallowed model: false; only the pinned Gemma base was trained.
- Scientist call: Claude Opus 4.8, high effort, verified 200k context, CLI 2.1.219; terminal result records 158 turns and `$12.700292500000003` total cost (294 input, 224,050 cache-creation input, 17,149,722 cache-read input, 98,364 output tokens).
- Slurm allocation: job `92184`, node `slurm2-a3nodesetondem-3`, one H100, 16 CPUs, 128 GiB requested memory. Queue history reports `02:50:24` allocated wall time, or about 2.84 nominal GPU-hours. This is allocation, not proof of continuous utilization; point samples in the trace cannot establish whole-run utilization.

## Cohort disposition

The 0.92 result should appear only in the flagged sensitivity accounting. It must be excluded from the clean primary cohort: the authoritative results view says `clean_complete=0`, `flagged_complete=1`, and the confirmed premature termination means there is no basis to clear the flag.

The only bounded comparator inspected was terminal, clean protocol cell `c54r01/job 92185`: 0.91 with no judge flags. The raw-minus-protocol point difference is +0.01 (one percentage point), but this is provisional `n=1` versus `n=1`, the treatments differ, and the raw observation is flagged. It supports no efficacy, superiority, or promotion claim. The remaining cells were not inspected because they were in flight during this audit.
