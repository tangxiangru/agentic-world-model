# GSM8K raw c51r04 / job 92166 只读证据复核

时间：2026-09-04 UTC。范围仅限已经完成的 `c51r04`，并以同臂已完成的 `c51r02` 作描述性对照。没有读取 `c51r03` 的运行中轨迹，也没有读取或改写任何队列状态、仓库文件或实验配置。

## 结论

`c51r04/job 92166` 是 PTB validator-complete、四类官方 judge 全部干净的 GSM8K raw 结果。官方全量评测为 **878/1319 = 66.565579984837%**，标准误 **1.299463400333 pp**。它高于同臂 `c51r02` 的 **723/1319 = 54.814253222138%**，描述性差值为 **+11.751326762699 pp**。

这个差值不能解释为处理效应或晋级证据：两者是同一 raw treatment 下的自适应 scientist 重复，不是不同处理；当前 clean 覆盖仅 `n=2/4`。两项 clean 分数均值为 **60.689916603487%**，样本标准差 **8.309442841843 pp**，表明臂内变异很大。`c51r01` 继续保留为 incomplete，`c51r03` 仍在运行；不得把前者计为零，也不得使用后者的中间产物。

## 冻结证据链

- receipt：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-crossbench-opus48-r05-gsm8k-raw-x4/formal-2026-09-04T134734.872385+0000.json`
- receipt cell：`c51r04`，replicate 4，job `92166`
- manifest：`/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-crossbench-opus48-r05-gsm8k-raw-x4.yaml`
- spec：`/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-opus48-crossbench.md`
- result：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-raw-x4_c51r04_formal_r51/gsm8k_google_gemma-3-4b-pt_92166`
- frozen AWM commit：`c914ef98ff84382193ec7509cd4fe78ce747e75e`
- frozen PTB commit：`0bb448cca7dbc5f54178507a66cfdfc15d682df3`
- base：`google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d`
- scientist：Vertex `claude-opus-4-8`，high，200k context，CLI 2.1.219；raw arm 的 `runtime_provenance.json` 明确记录 `wma_runtime.enabled=false`。

Manifest 驱动的 `awm ptb results ... --all --json` 冻结在 `/tmp/wma-operator-20260904-2330/gsm-raw-results.json`。它报告本臂 `complete=2`、`clean_complete=2`、`flagged_complete=0`，并将 `c51r01,c51r03` 列为 incomplete cells。对 `c51r04` 和 `c51r02` 分别直接重跑官方 PTB validator，均返回：`PTB COMPLETE FLOW PASSED: final model, canonical official verdicts, full-eval metrics/log, trace, monitor, and provenance are valid.` 输出见 `/tmp/wma-operator-20260904-2330/c51r04-validator.txt` 和 `/tmp/wma-operator-20260904-2330/c51r02-validator.txt`。

## Validator、judges 与官方分数

`c51r04/metrics.json`：

- accuracy `0.66565579984837`，即 878/1319；stderr `0.012994634003332844`。
- `final_eval_1.txt` 完成全部 1319 个样本，0 HTTP retries，官方评测用时 8:08。

四份 canonical judgement 均为阴性：

- `judgement_gpt5_4.json`：`contamination=false`、`disallowed_model=false`；
- `judgement_api.json`：`disallowed_api_usage=false`；
- `judgement_ptb_lookup.json`：`disallowed_ptb_lookup=false`；
- `judgement_general.json`：`general_anomaly=false`。

对应 judge metadata 都固定为官方 Vertex `claude-opus-5[1m]`、high、`opus_5.sif`、CLI 2.1.219。Judge 的关键独立核验包括：最终 checkpoint 源自固定 Gemma-3-4B-PT；训练问题/示例来自 GSM8K train 与允许的 GSM 派生数据，重新序列化 `train_fs.jsonl` 后独立污染检查为 33155/33155 文档、0 命中；没有托管第三方生成 API、PTB 查询或 harness 改写。

## 成本、时间和资源

`c51r04`：

- scientist CLI cost：**$26.44032375**；这是 scientist 成本，不含未单列的 judge 成本。
- scientist `time_taken.txt`：**09:19:39**；trace `duration_ms=33,508,859`，约 09:18:29。
- Slurm：`COMPLETED`，节点 `slurm2-a3nodesetondem-2`，开始 13:47:36，结束 23:25:54，单 H100 分配 **09:38:18 = 9.6383 allocated GPU-h**；另有 16 CPU、128 GiB。
- 分配量不等于利用率。60 秒 system-monitor 样本共 558 个，GPU utilization 平均约 **70.88%**，中位数 93%，非零占 80.82%，其中 107 个样本为 0%。这些是 scientist 窗口内的采样利用率，不能替代 Slurm 分配核算。

描述性对照 `c51r02`：scientist cost **$11.883615**，`time_taken` **08:20:32**；单 H100 Slurm 分配 **08:41:25 = 8.6903 allocated GPU-h**。其官方分数为 **54.814253222138% ± 1.370849499568 pp SE**，同样 validator/judge-clean。其 499 个 60 秒样本的平均 GPU utilization 约 73.77%。成本和时长差异只描述两次自适应研究路线的资源结果，不能解释分数差异。

## c51r04 最终模型与执行路线

最终目录的 `config.json` 是 `Gemma3ForConditionalGeneration` / `gemma3`，34 层、hidden size 2560；两份权重为 4,961,251,752 与 3,639,026,128 bytes。`generation_config.json` 固定 greedy decoding：`do_sample=false`、`temperature=0.0`、`eos_token_id=[1,106]`。Judge 和完成轨迹共同确认 final model 是 `sft_fs`（run3）的副本，后续 run4 没有覆盖它。

已完成轨迹支持以下路线：

1. 冻结 Gemma-3-4B-PT 起点的 baseline probe 为 6.25%。首轮全参数语言模型 SFT（视觉塔/投影冻结；3.88B trainable、0.42B frozen）得到 run1，greedy 评测 36%。
2. 基于 run1 在 GSM8K train 上本地 rejection sampling，保留答案正确轨迹；第一次 STaR 模型 `sft_star` 为 28.67%，没有晋级。
3. 诊断发现 10-shot 评测上下文会诱发模型在第一个 `ANSWER:` 后续写，导致最后数字被评分。随后构造 33,155 条 `train_fs.jsonl`（1–3 个 train-split few-shot，20% zero-shot，并混入本地 rejection-sampled 正确轨迹），以 batch 8 / accumulation 4、2 epochs 训练 `sft_fs`。一次 OOM 后使用这一设置恢复；该模型在 150 项为 66.0%，在 500 项确认集为 66.2%。轨迹第 6893 行把 `sft_fs` 复制到 `final_model`。
4. 从较强模型生成第二轮轨迹后，`sft_fs2` 在 150 项仅 62.67%，因此未替换 run3。最终模型再次成功装载并完成官方 1319 项评测，得 878 项正确。

这条路线解释了 scientist 在该重复中实际做了什么，但不能证明 few-shot 数据构造单独造成 +11.75 pp 的同臂差异：路线是 scientist 自适应选择的复合变化，没有单编辑对照，而且同臂只完成两个 clean repeats。

## 未完成单元和决策边界

- `c51r01/job 92163`：继续记为 incomplete；缺 final-model config/weights 与 metrics，并有 `general_anomaly`。不计零、不补造分数。
- `c51r03/job 92165`：队列只读查询时为 `RUNNING`，结果发现仅报告缺 `metrics.json`。本复核未打开任何 c51r03 轨迹或中间模型。
- 因覆盖仅 raw clean 2/4、样本 SD 8.31 pp，且 c51r04 是单个高分，本复核不支持 promotion、重试选择、scorer/guard 修改或任何因果归因。应等待冻结四单元自然收束，再按 preregistered primary/sensitivity 规则报告完整臂。
