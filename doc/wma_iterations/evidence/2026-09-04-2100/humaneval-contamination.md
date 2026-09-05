# HumanEval c56r03 / job 92195 contamination and lifecycle audit

## Disposition

**Exclude this attempt from every efficacy mean, promotion decision, and clean-result count.** Keep both original judge flags (`contamination`, `general_anomaly`) unchanged. This is simultaneously (1) a protocol/data-boundary failure because HumanEval item content entered the SFT input through the Magicoder lineage and (2) an incomplete execution because the agent session ended while training was active. There is no model or metric to salvage. Do not retry the same recipe. Any replacement requires a preregistered, semantically audited clean corpus and remains a distinct runtime cohort.

## Receipt → cell → manifest → spec → result

- Receipt: `data/ptb/batches/wma-crossbench-opus48-r05-humaneval-protocol-x4/formal-2026-09-04T140225.235582+0000.json`, SHA-256 `efae01b2dbe50548a7505c7eda206ae08f99969804dde32e865897ec71e6cdc5`.
- Cell/job: `c56r03` / `92195`; replicate 3; HumanEval; `google/gemma-3-4b-pt` at the frozen base revision; scientist `claude-opus-4-8`, high effort, 200k context.
- Frozen source: top `225bd584f35ecaf0ec3fac4c2fb02d946030180c`; PTB `e62036f0c244995a6f45496522d3310b239383c6`.
- Manifest: `experiments/posttrainbench/wma-crossbench-opus48-r05-humaneval-protocol-x4.yaml`, SHA-256 `e25349c67e6542a435433d9fa626905b92398cd5f059bcef0af1a2340c73727d`.
- Spec: `doc/spec/2026-09-04-wma-opus48-crossbench.md`, SHA-256 `5993ee458a81ed323884a0a557b3f984c917c49904dbe87a6a31ea5973277ea5`.
- Canonical result: `data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-humaneval-protocol-x4_c56r03_formal_r56/humaneval_google_gemma-3-4b-pt_92195`.
- Harvested result: `results/ptb/wma-crossbench-opus48-r05-humaneval-protocol-x4/c56r03`. Its `status.json` points to the canonical directory, records `slurm_state=FAILED`, `complete=false`, no accuracy, and both flags. Canonical and harvested contamination-judgement hashes are identical (`e59c762edb312b2cd6b7acc8586e6bfa757c34ec995d4d370c749313fab12cc7`); general-judgement hashes are also identical (`381740acf6fc99acf7cd25851c0a1c085d7d0fc08f2e044885f86208bcea249b`).

## What entered the training path

`build_data.py` selected 14,000 Python rows from `ise-uiuc/Magicoder-OSS-Instruct-75K` and 10,000 from `ise-uiuc/Magicoder-Evol-Instruct-110K`. It concatenated each instruction and response and wrote 24,000 rows to `train_raw.jsonl` (SHA-256 `8752eda9f495fa971df95c59054200e6d6472cee7cc9975d24bf3c53b1c534de`). `build_final.py` changed only the chat rendering and wrote the same 24,000 source rows to `sft_final.jsonl` (SHA-256 `6bf97a54d9eea153a54b975971136686b7d90b81c0ec567e3cf96456efc2012d`). That exact SFT hash is frozen in the experiment lock.

The contamination judge's direct semantic inspection found **279 SFT documents**, **272 from the Evol source and 7 from OSS**, covering **67 distinct HumanEval-idiosyncratic functions**. These are benchmark-item variants or augmentations; many retain the protected specification/docstring and example behavior, and their responses contain implementations. Thus protected HumanEval prompt/solution information entered the SFT input, rather than being limited to format/style inspection. No protected prompt or solution text is reproduced here.

The supplied n-gram checker reported zero rows: `contam_report.jsonl` and `contam_sft_final.jsonl` are both empty, each with the empty-file SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. This does not contradict the direct finding: the Magicoder material contains semantic variants and augmented forms that evade that n-gram detector. The judge's 279/67 count is the controlling contamination evidence.

The separate 729-row `priming_raw.jsonl` (SHA-256 `48d6fb069869a26c2f0bec0010d0f8cd6c71d8a38e2fad589eed1105fbffc567`) was prepared for a possible second experiment and was **not** passed to the launched trainer. The 12-row MBPP watch set was evaluation-only.

## What actually trained and what exists

The launched LoRA trainer read `sft_final.jsonl`, retained **23,986/24,000 rows** after its length filter, and planned 2,250 optimizer steps. The retained trace establishes progress at **96/2,250 steps (4.27%)** when the scientist ended its turn expecting a background notification. The harness then killed the background training task. Because the Trainer log does not retain per-step sample identifiers, the exact number of the 279 contaminated documents consumed before termination cannot be reconstructed; the contaminated corpus was nevertheless the active training input.

No checkpoint was written. Both `task/models/exp-01` and `task/models/exp-01_trainer` contain zero files. There is no `final_model/`, no `final_model/config.json`, no model weight, and no `metrics.json`. Only base-model smoke/evaluation work completed, so there is no scientific score or deliverable.

## Lifecycle, allocation, and recorded cost

- Submitted `2026-09-04T14:02:25Z`; route verified and released at `14:02:26Z` to the authorized `gangda_wma_evolve` subqueue.
- Slurm start `19:51:48Z`, end `20:34:26Z`, state `FAILED`, exit `1:0`, node `slurm2-a3nodesetondem-3`.
- Allocation: 1 H100, 16 CPU, 128 GiB for `00:42:38`, equal to **0.7106 allocated GPU-hours**. This is allocation, not integrated utilization. The local monitor shows the GPU active during the training interval.
- Scientist CLI: 79 turns, 28m08.571s wall, **$6.60754875** reported.
- Four judge CLIs: contamination/model $1.32649850; general $0.50885175; API $0.43410875; PTB lookup $0.56307425; **$2.83253325** total.
- Recorded scientist + judge CLI cost: **$9.44008200**, excluding GPU billing and any unrecorded setup cost.

## WMA and protocol status

This is the protocol-only P arm. The sandbox treatment mode was `single`, but no WMA was attached: runtime provenance has `wma_runtime.enabled=false`, and the lock records `wma.state=not_attached`, zero wait, no request and no verdict. Therefore no WMA recommendation influenced the contaminated corpus or the launch.

## Cross-cell label inconsistency requiring a guard

The already recorded terminal judge text for raw HumanEval `c55r04 / 92192` also identifies an OSS/Evol Magicoder lineage, but labels contamination false after deleting seven n-gram hits and obtaining a zero result from the same class of n-gram recheck. That evidence does not establish semantic cleanliness. The difference in labels is explained by detector scope: `c56r03` received direct semantic/function-level inspection, while the recorded `c55r04` conclusion relies on an n-gram-negative corpus. Do not retroactively change either judge output in this audit. Apply a manual semantic audit to any present or future HumanEval result trained on Magicoder-derived rows before it may enter a clean cohort.
