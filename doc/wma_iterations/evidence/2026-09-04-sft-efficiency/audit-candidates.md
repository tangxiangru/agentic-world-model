# Candidate training episode audit

Count substantive full-parameter supervised training episodes that complete the planned scientific fit, including short deliberate endgame fits and RFT supervised fits. Count one training invocation per completed episode, not one per epoch/checkpoint/eval. Separate external-only targets (completed_sft) from any self-generated target mixture (completed_rft); their sum is completed_supervised. Do not count generation, smoke/probes, failed or intentionally aborted substantive attempts, no-run proposals, weight averaging, or selection-only cards.

Frozen cohort:15 A/AB/C/D cells; original result_dir and receipt-backed cell identifiers are preserved in JSON.

| Cell | External-only SFT | Self/RFT-containing fit | Completed supervised total | PTB accuracy |
|---|---:|---:|---:|---:|
| w06r01 | 2 | 1 | 3 | 0.692191 |
| w06r02 | 3 | 1 | 4 | 0.702047 |
| w06r03 | 1 | 1 | 2 | 0.717210 |
| w06r04 | 1 | 2 | 3 | 0.652009 |
| w07r01 | 1 | 2 | 3 | 0.727824 |
| w07r02 | 1 | 2 | 3 | 0.729340 |
| w07r03 | 3 | 0 | 3 | 0.714177 |
| w07r04 | 1 | 2 | 3 | 0.699773 |
| w08r01 | 2 | 0 | 2 | 0.732373 |
| w08r02 | 1 | 2 | 3 | 0.698256 |
| w08r03 | 2 | 1 | 3 | 0.647460 |
| w08r04 | 2 | 1 | 3 | 0.793783 |
| w09r01 | 2 | 1 | 3 | 0.765732 |
| w09r02 | 2 | 2 | 4 | 0.744503 |
| w09r03 | 1 | 2 | 3 | 0.688400 |

Total: {'completed_sft': 25, 'completed_rft': 20, 'othertraining': 0, 'completed_supervised': 45}

## Every counted card

- w06r01: exp-02 external_supervised_fit (493 steps; 1.0 planned epochs); exp-04 external_supervised_fit (1772 steps; 1.0 planned epochs); exp-05 rft_supervised_fit (791 steps; 1.0 planned epochs)
- w06r02: exp-03 external_supervised_fit (2520 steps; 1.0 planned epochs); exp-04 external_supervised_fit (397 steps; 1.0 planned epochs); exp-07 rft_supervised_fit (528 steps; 1.0 planned epochs); exp-10 external_supervised_fit (262 steps; 1.0 planned epochs)
- w06r03: exp-02 external_supervised_fit (2110 steps; 2 planned epochs); exp-04 rft_supervised_fit (963 steps; 1 planned epochs)
- w06r04: exp-02 external_supervised_fit (1112 steps; 1.0 planned epochs); exp-04 rft_supervised_fit (692 steps; 1.0 planned epochs); exp-05 rft_supervised_fit (504 steps; 1.0 planned epochs)
- w07r01: exp-02 external_supervised_fit (1736 steps; 2 planned epochs); exp-04 rft_supervised_fit (2122 steps; 2 planned epochs); exp-06 rft_supervised_fit (2236 steps; 2 planned epochs)
- w07r02: exp-02 external_supervised_fit (1126 steps; 2 planned epochs); exp-04 rft_supervised_fit (626 steps; 2 planned epochs); exp-05 rft_supervised_fit (313 steps; 1 planned epochs)
- w07r03: exp-02 external_supervised_fit (789 steps; 1.0 planned epochs); exp-03 external_supervised_fit (888 steps; 1.0 planned epochs); exp-05 external_supervised_fit (713 steps; 1.0 planned epochs)
- w07r04: exp-02 external_supervised_fit (1632 steps; 1.0 planned epochs); exp-03 rft_supervised_fit (2686 steps; 1.0 planned epochs); exp-04 rft_supervised_fit (3477 steps; 1.0 planned epochs)
- w08r01: exp-02 external_supervised_fit (3037 steps; 1 planned epochs); exp-04 external_supervised_fit (2188 steps; 1 planned epochs)
- w08r02: exp-02 external_supervised_fit (1110 steps; 1 planned epochs); exp-03 rft_supervised_fit (639 steps; 1 planned epochs); exp-04 rft_supervised_fit (1553 steps; 1 planned epochs)
- w08r03: exp-02 external_supervised_fit (1034 steps; 2 planned epochs); exp-04 external_supervised_fit (1019 steps; 1.5 planned epochs); exp-07 rft_supervised_fit (82 steps; 1 planned epochs)
- w08r04: exp-02 external_supervised_fit (1902 steps; 2 planned epochs); exp-05 external_supervised_fit (1607 steps; 1 planned epochs); exp-10 rft_supervised_fit (149 steps; 1 planned epochs)
- w09r01: exp-02 external_supervised_fit (1313 steps; 1.0 planned epochs); exp-04 external_supervised_fit (4469 steps; 1.0 planned epochs); exp-05 rft_supervised_fit (2145 steps; 1.0 planned epochs)
- w09r02: exp-02 external_supervised_fit (743 steps; 1.0 planned epochs); exp-03 external_supervised_fit (1580 steps; 1.0 planned epochs); exp-04 rft_supervised_fit (196 steps; 1.0 planned epochs); exp-06 rft_supervised_fit (188 steps; 1.0 planned epochs)
- w09r03: exp-02 external_supervised_fit (4686 steps; 2 planned epochs); exp-04 rft_supervised_fit (1484 steps; 1 planned epochs); exp-05 rft_supervised_fit (3826 steps; 1 planned epochs)

## Classification traps verified

- w06r02/exp-10: 8000 external OMI2 solutions selected using own sampling failures; selection is adaptive, but targets are external, not self-generated. Count as one external supervised fit. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003/task/memory/cards/exp-10.yaml`.
- w07r01/exp-06: Card source explicitly includes 11150 synthetic:self RFT rows at weight 2, despite family=sft. Card also performs soup and evaluation after training; only one gradient-training episode. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r01_formal_r7/gsm8k_google_gemma-3-4b-pt_91007/task/memory/cards/exp-06.yaml`.
- w07r04/exp-04: Card source explicitly concatenates the 33707 self/RFT rows used by exp-03, despite family=sft. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r04_formal_r7/gsm8k_google_gemma-3-4b-pt_91010/task/memory/cards/exp-04.yaml`.
- w08r01/exp-04: External-only fit on70000 fresh OMI2 teacher rows. Although card pitfalls mention RFT sampling, those generated rows are not the final training corpus; do not count generation as a fit. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r01_formal_r8/gsm8k_google_gemma-3-4b-pt_91014/task/memory/cards/exp-04.yaml`.
- w08r02/exp-04: sft_v5.jsonl src-field scan: rft:gsm8k_train=2178 and rft:augmented_gsm8k=5423, hence 7601 self/RFT rows of 98367; 15000-row replay anchor came from exp-03 mixture. Source code in solve_parsed.txt:10388-10389; raw substantive fit launch at :10642. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r02_formal_r8/gsm8k_google_gemma-3-4b-pt_91021/task/memory/cards/exp-04.yaml`.
- w08r03/exp-07: Completed non-probe full-parameter supervised fit; multi-epoch schedule counts one episode, not one per epoch. Endgame fit is short (82 or149steps) but completes its planned one-epoch scientific training; not a smoke. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r03_formal_r8/gsm8k_google_gemma-3-4b-pt_91022/task/memory/cards/exp-07.yaml`.
- w08r04/exp-04: family=other and training_summary.steps=1000 refers to an existing exp-02 checkpoint; explicit no-training card. Exclude. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023/task/memory/cards/exp-04.yaml`.
- w08r04/exp-10: Completed non-probe full-parameter supervised fit; multi-epoch schedule counts one episode, not one per epoch. Endgame fit is short (82 or149steps) but completes its planned one-epoch scientific training; not a smoke. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023/task/memory/cards/exp-10.yaml`.
- w09r03/exp-05: Card source concatenates 74946 external OMI/GSM8K rows and 47480 verified self samples, despite family=sft. Source: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r03_formal_r9/gsm8k_google_gemma-3-4b-pt_91026/task/memory/cards/exp-05.yaml`.

## Excluded substantive attempts

- w06r01/exp-05: First launch failed at checkpoint save step300/791; not a completed fit. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r01_formal_r6/gsm8k_google_gemma-3-4b-pt_91002`: task/memory/cards/exp-05.yaml:situation.pitfalls_hit, solve_parsed.txt:9118, solve_parsed.txt:9147, solve_parsed.txt:9164
- w06r01/exp-05: 20:43:44 restarted full command, 20:43:53 stopped it to perform WMA-requested save smoke; 9-second launch not a completed training and no optimizer-step claim made. Final full run relaunched20:46:05. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r01_formal_r6/gsm8k_google_gemma-3-4b-pt_91002`: solve_parsed.txt:9313, solve_parsed.txt:9331, solve_parsed.txt:9335, solve_parsed.txt:9413
- w06r02/exp-02: Explicit result.execution=failed; OOM at100steps. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003`: task/memory/cards/exp-02.yaml:result, solve_parsed.txt:4297
- w06r02/exp-06: Explicit result.execution=not_run; training command never run. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003`: task/memory/cards/exp-06.yaml:result
- w07r02/exp-02: First full launch max_seq_len2048 dropped2440/72000rows; killed then relaunched3072. Count only completed1126step run. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r02_formal_r7/gsm8k_google_gemma-3-4b-pt_91008`: solve_parsed.txt:5234, solve_parsed.txt:5265, solve_parsed.txt:5400, solve_parsed.txt:5635, task/memory/cards/exp-02.yaml:situation.pitfalls_hit
- w08r04/exp-05: First launch failed in Trainer initialization with corrupted liger metadata before actual training; restart completed1607steps. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023`: solve_parsed.txt:7562, solve_parsed.txt:7690, solve_parsed.txt:7695, solve_parsed.txt:8006, task/memory/cards/exp-06.yaml:situation.pitfalls_hit
- w09r02/exp-04: First attempt crashed at checkpoint save step60, followed by long failed-job polling; final196step fit counted once. Sources under `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r02_formal_r9/gsm8k_google_gemma-3-4b-pt_91025`: task/memory/cards/exp-04.yaml:result.training_summary, task/memory/cards/exp-04.yaml:situation.pitfalls_hit, task/memory/cards/exp-04.lock.json:relocked_from

Generation-only aborts (e.g. w07r01 exp04 first sampling on private-dev items, w08r01 exp04 shortened sampling, w09r03 exp04 pkill sampling wrapper) are not supervised fits. Per-card smoke records and raw smoke launch candidates are retained in JSON; they are not added to completed totals.

## Limits

- Card wall_h may include evaluation/tokenization/recovery; not pure optimization GPU hours.
- Smoke launch candidates are a bounded lexical aid, not an exhaustive independently timed process census. Recorded smoke records may repeat prior events and are not summed.
- No completed optimizer schedule followed solely by final-save failure was established in these15 cells. Known save failures stopped at intermediate steps and are excluded, not counted as completed fits.
- No extra completed full training episode hidden under a non-training family or duplicate completed episode inside a card was found after full card scan and targeted raw launch audit.

Canonical method labels are preserved in `audit-candidate-overrides.json` (`family` equals the original setup.method.family); `self_sample_mix` is a separate provenance observation. Main axis:45 completed full supervised episodes. Label-only sensitivity:29 family=sft +16 family=rft; data-provenance split:25 external-only +20 including self targets. w06r02/exp03 counts as completed:2520 steps and schedule completed despite its poor evaluation.
