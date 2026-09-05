# Baseline/control监督训练次数审计

A completed fit reaches the intended optimization schedule even if artifact save subsequently fails. Full-intent OOMs and fixed-short-step smokes are separate. Canonical card families retained; self-sampled mixture flag orthogonal.

|cell|SFT complete|RFT complete|全部完成fit|产物成功|完成后save失败|未完成正式尝试|训练smoke/bench调用|official final|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c10r01|3|1|4|3|1|0|8|0.815011|
|c10r02|2|1|3|3|0|0|3|0.767248|
|c10r03|1|1|2|2|0|0|5|0.702047|
|c10r04|2|2|4|3|1|0|5|0.687642|
|w10r01|2|1|3|3|0|0|1|0.694466|
|w10r02|2|1|3|3|0|0|4|0.717210|
|w10r03|3|0|3|3|0|1|5|0.719484|
|w10r04|3|1|4|4|0|0|3|0.722517|

按正式card拟合次数：WMA13完成fit（10SFT+3RFT），控制13（8SFT+5RFT）。WMA有1次额外OOM正式尝试；控制有2次完成后save失败。控制的这2次失败必须留在训练次数/成本分母里。
c10r02额外一个n4000×1epoch、无max_steps上限的启动smoke完成了自身schedule，但用途明确是smoke/load-test，没有正式实验假设或候选，因此仍排除主完整训练分母，单独记录。
smoke/benchmark次数按实际展开shell循环与os.execvp记载，包含失败调用；不含tokenize-only、dry-run、采样或eval smoke。此表不把所有smoke都归为完成fit。

## c10r01

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:24:43Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998/solve_parsed.txt:4439)|completed_artifact_saved|0.7333 n150 sampled inherited|
|exp-04/1|rft|True|exp-02|[2026-09-03T12:44:31Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998/solve_parsed.txt:8078)|completed_artifact_saved|0.81 n500 greedy|
|exp-05/1|sft|False|exp-04|[2026-09-03T14:17:40Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998/solve_parsed.txt:9163)|schedule_completed_save_failed|—|
|exp-06/1|sft|False|exp-04|[2026-09-03T15:25:26Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998/solve_parsed.txt:9790)|completed_artifact_saved|0.82 n500 greedy|

exp-05: FULL1245-step schedule completed; final GenerationConfig save failed and no usable weights. Count1 completed fit,0 artifact-success. Retried as exp-06.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## c10r02

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90999`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:40:15Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90999/solve_parsed.txt:4682)|completed_artifact_saved|0.72 n150 sampled inherited|
|exp-04/1|rft|True|exp-02|[2026-09-03T11:42:44Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90999/solve_parsed.txt:6664)|completed_artifact_saved|0.72 n150 greedy|
|exp-05/1|sft|True|exp-02|[2026-09-03T14:15:44Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90999/solve_parsed.txt:7679)|completed_artifact_saved|0.7133 n150 greedy|

exp-05: One full-intent supervised fit completed; intermediate checkpoints are not additional fits. Canonical family sft, but training targets are gold-majority-filtered self-sampled triple-attempt vote distillation; self_sample_mix=true.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## c10r03

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_91000`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:43:02Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_91000/solve_parsed.txt:4825)|completed_artifact_saved|0.6867 n150 greedy; 0.6067 n150 greedy|
|exp-03/1|rft|True|base_model|[2026-09-03T11:47:45Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_91000/solve_parsed.txt:6226)|completed_artifact_saved|0.6933 n150 greedy|

exp-03: One full-intent supervised fit completed; intermediate checkpoints are not additional fits. RFT-labeled mixed-data fit starts from BASE, not from the sampling parent.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## c10r04

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_91001`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T14:48:29Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_91001/solve_parsed.txt:4707)|completed_artifact_saved|0.7467 n150 greedy|
|exp-03/1|sft|False|base_model|[2026-09-03T17:28:08Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_91001/solve_parsed.txt:6215)|completed_artifact_saved|0.68 n150 greedy|
|exp-05/1|rft|True|exp-02|[2026-09-03T20:28:28Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_91001/solve_parsed.txt:7344)|schedule_completed_save_failed|—|
|exp-05/2|rft|True|exp-02|[2026-09-03T21:25:21Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_91001/solve_parsed.txt:7637)|completed_artifact_saved|0.7067 n150 greedy|

exp-05: TWO full-intent supervised fits under ONE card: first reached final save after training and lost weights; second repeats same2epoch schedule and saves. First failure misleadingly recorded under situation.smoke_runs. Claimed36min conflicts with actual timeline20:28launch,21:16epoch1.824,21:19finaldir; do not use36min as measured cost.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## w10r01

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:20:16Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:6991)|completed_artifact_saved|0.7067 n150 greedy|
|exp-03/1|sft|False|exp-02|[2026-09-03T10:55:10Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:10636)|completed_artifact_saved|0.72 n150 greedy|
|exp-04/1|rft|True|exp-03|[2026-09-03T12:51:14Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:13480)|completed_artifact_saved|0.74 n150 greedy|

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## w10r02

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:07:31Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:5460)|completed_artifact_saved|0.6733 n150 greedy; 0.68 n150 greedy|
|exp-03/1|rft|True|base_model|[2026-09-03T10:59:36Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:7924)|completed_artifact_saved|0.66 n150 greedy; 0.6867 n150 greedy|
|exp-06/1|sft|False|base_model|[2026-09-03T13:54:52Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10408)|completed_artifact_saved|0.738 n500 greedy|

exp-03: One full-intent supervised fit completed; intermediate checkpoints are not additional fits. RFT-labeled mixed-data fit starts from BASE, not from the sampling parent.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## w10r03

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_90984`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:10:53Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_90984/solve_parsed.txt:5844)|aborted_oom|—|
|exp-02/2|sft|False|base_model|[2026-09-03T08:29:27Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_90984/solve_parsed.txt:6688)|completed_artifact_saved|0.58 n150 sampled inherited; 0.5267 n150 sampled inherited|
|exp-04/1|sft|False|base_model|[2026-09-03T11:00:02Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_90984/solve_parsed.txt:8647)|completed_artifact_saved|0.74 n150 greedy|
|exp-05/1|sft|False|exp-04|[2026-09-03T12:52:50Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r03_formal_r10/gsm8k_google_gemma-3-4b-pt_90984/solve_parsed.txt:9554)|completed_artifact_saved|0.7267 n150 greedy|

exp-02: First formal launch bs64 OOM151/2636 (~.15h), then restarted bs32 and completed2636steps. Count1 completed fit +1 aborted full-intent; not2 completed fits.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。
## w10r04

Result: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985`

|card/episode|family|selfsample mix|parent|launch|outcome|score(s), n, decoder|
|---|---|---|---|---|---|---|
|exp-02/1|sft|False|base_model|[2026-09-03T08:50:19Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:6439)|completed_artifact_saved|0.7267 n150 greedy; 0.7133 n150 greedy|
|exp-03/1|rft|True|exp-02|[2026-09-03T11:07:19Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:8432)|completed_artifact_saved|0.699 n1319 greedy; 0.68 n150 greedy|
|exp-04/1|sft|True|base_model|[2026-09-03T11:54:42Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:9346)|completed_artifact_saved|0.7134 n1319 greedy; 0.707 n150 greedy; 0.71 n1319 greedy|
|exp-06/1|sft|False|exp-04|[2026-09-03T14:42:28Z](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:11048)|completed_artifact_saved|0.7119 n1319 greedy; 0.74 n150 greedy|

exp-04: One full-intent supervised fit completed; intermediate checkpoints are not additional fits. Canonical sft includes25,766 verified self-generated rows /110,912total; keep canonical family and flag mixture.

卡级全部accuracy测量、n、decoder、选中checkpoint与probe标注见JSON的scores_card_reported；不同尺不强行合并。

## 时长与规模边界

完整fit次数不等于计算量：2epochs与1epoch均算一次，8k行与150k行也均算一次。JSON为每个episode附训练数据声明、hyperparameters、训练steps和card原始training_summary。

各cell按唯一训练卡相加的reported wall_h：c10r01=5.31h, c10r02=5.03h, c10r03=4.61h, c10r04=6.4h, w10r01=3.26h, w10r02=4.43h, w10r03=4.83h, w10r04=4.5h。WMA合计17.02h，控制21.35h；这不是完整GPU计费：未含smoke、RFT数据采样、所有评估与审查等待；w10r03还需独立计入约.15h失败正式启动，c10r04原卡漏准了首次save-fail训练时长。
