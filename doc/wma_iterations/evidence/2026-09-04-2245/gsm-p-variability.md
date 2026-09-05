# Opus 4.8 GSM8K protocol-only arm: variability readout

Date: 2026-09-04 UTC  
Scope: completed cells `c52r01`–`c52r04` / Slurm jobs `92167`–`92170` only. This is a read-only post hoc description of one four-replicate protocol-only arm. No WMA was attached or run (`wma_runtime.enabled=false`; the receipt has empty `wma_private_checkouts` and `wma_runtime_validation`). No in-flight cell, raw WMA material, or external model was read. The executed scientist recipes differ substantially, so this report does not assign causal effects to individual choices.

## Frozen provenance and coverage

- Manifest: `/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-crossbench-opus48-r05-gsm8k-protocol-x4.yaml`
- Formal receipt: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-crossbench-opus48-r05-gsm8k-protocol-x4/formal-2026-09-04T134801.355200+0000.json`
- Spec: `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-opus48-crossbench.md`
- Frozen source: AWM top commit `c914ef98ff84382193ec7509cd4fe78ce747e75e`; protocol checkout SHA `31b854bbc5e1f7f66685a8ec0d43845a6c2472f1`; PTB commit `0bb448cca7dbc5f54178507a66cfdfc15d682df3`.
- Common contract: `google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d`; scientist `claude-opus-4-8`, high effort, 200,000-token context; one H100, 16 CPU, 128 GiB memory, 400 GiB scratch, and a 10-hour scientist cap per cell; official GSM8K evaluation with 1,319 items.
- Validator discovery: `uv run awm ptb results <manifest> --all --json` returned 4/4 complete attempts, 3 clean, 1 flagged, no incomplete cells, and one attempt per cell. Slurm `COMPLETED` was not used as scientific completion evidence.

| Cell / job | PTB status | Official accuracy | Slurm allocation | Result directory |
|---|---|---:|---:|---|
| `c52r01` / `92167` | complete, clean | 0.6262319939 | 08:41:19 | `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-protocol-x4_c52r01_formal_r52/gsm8k_google_gemma-3-4b-pt_92167` |
| `c52r02` / `92168` | complete, **`general_anomaly`** | 0.4958301744 | 05:49:58 | `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-protocol-x4_c52r02_formal_r52/gsm8k_google_gemma-3-4b-pt_92168` |
| `c52r03` / `92169` | complete, clean | 0.5830174375 | 07:45:48 | `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-protocol-x4_c52r03_formal_r52/gsm8k_google_gemma-3-4b-pt_92169` |
| `c52r04` / `92170` | complete, clean | 0.4283548143 | 08:15:44 | `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-protocol-x4_c52r04_formal_r52/gsm8k_google_gemma-3-4b-pt_92170` |

The `c52r02` lifecycle flag is preserved. Its canonical `judgement_general.json` says the scientist ended at about 5.5 hours while an intended `exp-04` was training; that background experiment was killed and never evaluated. The submitted `exp-03/final` remained intact and validator-complete. `general_anomaly` is therefore a lifecycle comparability warning, not an instruction to rewrite its score or completion state.

## Primary clean estimate

Primary analysis excludes the lifecycle-flagged cell and uses the three validator-complete clean cells (`c52r01`, `c52r03`, `c52r04`). All dispersion values are across scientists; SD is the sample SD.

| n | Scores | Mean | Sample SD | Min–max | Range |
|---:|---|---:|---:|---:|---:|
| 3 | 0.6262319939, 0.5830174375, 0.4283548143 | **0.5458680819** | **0.1040379764** | 0.4283548143–0.6262319939 | **0.1978771797** |

This is a 10.40-point within-arm SD and a 19.79-point observed span. With only three clean, non-identically executed trajectories, it is an exploratory variability estimate rather than a stable noise-floor estimate.

## Sensitivity including every completed attempt

The all-attempt sensitivity retains `c52r02` at its observed score while retaining its `general_anomaly` label.

| n | Scores | Mean | Sample SD | Min–max | Range |
|---:|---|---:|---:|---:|---:|
| 4 | 0.6262319939, 0.4958301744 [flagged], 0.5830174375, 0.4283548143 | **0.5333586050** | **0.0885543998** | 0.4283548143–0.6262319939 | **0.1978771797** |

Including the flagged run lowers the mean by 0.0125094769 and the sample SD by 0.0154835765. It does not change the range because the flagged score lies between the clean extremes.

## Scientist cost and allocation

- Known aggregate scientist API cost: **$118.783249** for the four cells (mean **$29.69581225/cell**). This is scientist cost; official-judge calls are not folded into it.
- Receipt-backed Slurm allocation: **30:32:49 = 30.546944 H100-hours** in total because every job had one H100. Per cell: 8.688611, 5.832778, 7.763333, and 8.262222 hours for `r01`–`r04`.
- The nominal scientist allowance was 40 H100-hours (4 × 10 hours). Actual allocation was 76.37% of that cap. `c52r02` is the only run whose early ending is judge-flagged; shorter-than-cap duration alone is not treated as anomalous for the other cells.

## Executed recipe comparison

“Epoch-example presentations” below is the sum of `n_examples × epochs` over completed cards. It is a coarse volume measure: it does not normalize sequence length, dropped long rows, generation work, evaluation, or failed/retried launches.

| Cell | Completed training path from cards | Coarse completed volume | Selection evaluation | Checkpoint actually submitted |
|---|---|---:|---|---|
| `c52r01` | Four staged SFTs: (1) 7,473 GSM8K-gold rows from base, 3 epochs, LR 2e-5; (2) 55,473 GSM8K+48k MetaMath rows from base, 3 epochs, which failed the stop behavior; (3) 14,946 GSM8K few-shot-context rows continued from exp-02, 2 epochs, LR 1e-5, max length 2048; (4) 23,473 GSM8K+16k MetaMath rows in few-shot-context form continued from exp-03, 2 epochs, LR 7e-6. | 265,676 presentations; 8,286 optimizer steps recorded | exp-04 epoch 1: 0.6333 on n=150; epoch 2: 0.5933; copied-model repeat: 0.6200 on n=150 | `ckpts/exp-04/checkpoint-727` (intermediate epoch-1 checkpoint) |
| `c52r02` | Three completed SFTs: (1) 7,473 gold rows from base, 3 epochs, LR 1e-5; (2) 87,373 broad MetaMath/GSM rows from base, 2 epochs, rejected at 0.0267 on n=150; (3) fresh gold-only 7,473-row SFT from base, 3 epochs, retaining the calculator-annotation surface used by the evaluator. A fourth RFT card began but did not complete and is excluded from volume and selection. | 219,584 presentations over completed cards | selected exp-03: 0.4928 on the full n=1,319 test after 0.4800 on n=150 | `ckpts/exp03/final` |
| `c52r03` | Three runs: (1) 7,473 gold rows from base, 3 epochs, LR 2e-5; (2) 87,473 GSM8K+80k MetaMath rows from base, 2 epochs, LR 2e-5; (3) 101,215-row RFT/broad-corpus run from base, 2 epochs, rejected after 0.4223 on n=1,319. | 399,795 presentations; 12,496 optimizer steps recorded | exp-02: 0.5800 on n=150; later RFT was checked on n=1,319 and rejected | `ckpts/exp-02` (the pre-RFT broad-data SFT) |
| `c52r04` | Five completed attempts: (1) 7,473-row gold SFT from base, 3 epochs, LR 1e-5; (2) 23,284-row self-RFT from base, rejected; (3) 13,709-row gold-dominant continuation from exp-01 at LR 5e-6, rejected on full evaluation; (4) 42,473-row gold+MetaMath SFT from base, rejected after stop collapse; (5) 13,973-row gold-dominant MetaMath continuation from exp-01 at LR 5e-6, rejected on full evaluation. | 232,581 presentations; 7,274 optimizer steps recorded | the retained exp-01 checkpoint scored 0.447 on n=150 and 0.4337 on n=1,319; later candidates did not beat it | `ckpts/exp-01/checkpoint-468` (intermediate epoch-2 checkpoint) |

The final official results are consistent with the final local full-test checks where those exist: `r02` 0.4928 locally vs 0.4958 official, and `r04` 0.4337 locally vs 0.4284 official. `r01` and `r03` chose from n=150 development comparisons, but their official scores (0.6262 and 0.5830) are close to the chosen-card measurements (0.6333 and 0.5800). This agreement reduces concern about a gross selection artifact in these four outcomes, but it does not make the selection procedures identical.

## Ranked variance levers

### Directly observed differences

1. **The submitted recipes and parents differ.** `r01` submitted a fourth-stage, low-LR, few-shot-context MetaMath continuation; `r03` submitted a large MetaMath SFT from base and rejected a later RFT; `r02` submitted a fresh gold-only final model; `r04` retained an intermediate gold-only checkpoint after four later attempts failed to beat it. This is the largest concrete difference in what the official evaluator received.
2. **Data amount, composition, and prompt surface differ.** Completed coarse training volume spans 219,584–399,795 epoch-example presentations. Some trajectories used broad MetaMath data, some used self-generated RFT, and `r01` uniquely made later targets explicitly robust to few-shot context. Volume is not monotone with score: the largest-volume cell (`r03`) is second, while `r01` wins with less volume and `r04` is lowest with more volume than flagged `r02`.
3. **Checkpoint selection differs.** `r01` and `r04` submitted intermediate checkpoints after later training degraded; `r02` submitted a final gold-only checkpoint; `r03` retained the model before its later RFT regression. Checkpoint timing is therefore part of the executed treatment, not clerical packaging.
4. **Evaluation and decision depth differ.** `r02` and `r04` used full n=1,319 confirmation for the retained checkpoint. `r01` and `r03` selected primarily on n=150, although the later official evaluation landed close to those development estimates. The number of candidates also differs: 3 completed cards in `r02`/`r03`, 4 in `r01`, and 5 in `r04`.
5. **Available search time differs.** Receipt-backed allocation ranges from 5.833 to 8.689 hours. The flagged `r02` ended with a promised experiment in flight, unlike the other three. This is a direct opportunity-set difference and the reason to keep its lifecycle flag attached.

### Inferences, kept separate from observations

1. **Most plausible:** recipe trajectory and data-format handling account for a meaningful part of the observed span. The strongest within-cell evidence is `r01`: broad MetaMath initially broke stopping, few-shot-context continuation restored it, and a later low-LR mixed-data checkpoint improved its n=150 score. This supports the mechanism inside that trajectory; it does not isolate its causal contribution to the cross-cell 19.79-point range.
2. **Plausible:** retaining a well-timed intermediate checkpoint matters. Both `r01` and `r04` observed later-checkpoint or later-recipe degradation and shipped an earlier checkpoint. The direction and size of this effect cannot be generalized from two adaptive trajectories.
3. **Plausible:** `r02`'s truncated search reduced its chance to find an improvement. Its in-flight candidate has no outcome, so the counterfactual final score is unknown. The flag should not be used to assert that `r02` would have scored higher.
4. **Possible:** differences in n=150 versus n=1,319 selection added winner-selection noise. Official agreement with the chosen local scores argues against this being the sole explanation here.
5. **Not identified:** irreducible scientist stochasticity under an identical recipe. These are nominal arm replicates but not recipe replicates; each scientist adaptively chose different data, parents, schedules, evaluations, and checkpoints. The clean SD therefore mixes scientist/recipe variation, evaluation noise, and any execution noise. Four completed cells cannot partition those components.

The defensible result is that this protocol-only arm has high realized variability under adaptive scientist execution: 10.40 points sample SD across the three clean cells and 8.86 points when the lifecycle-flagged completion is included. The cards identify several concrete recipe and search-path differences, but the design does not support a causal ranking among them.
