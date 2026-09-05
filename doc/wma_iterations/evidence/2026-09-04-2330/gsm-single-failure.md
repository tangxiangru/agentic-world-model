# GSM8K single WMA w57r01 / job 92198 终态失败核验

核验时间：2026-09-04 UTC。范围严格限定为 receipt 所指向的 `w57r01` / job `92198`，未读取或解释同 manifest 的其他运行中单元。

## 结论

`w57r01` **没有科学完成，不能计分，也没有可提交或可恢复的模型产物**。Slurm 终态是 `FAILED`（exit `1:0`），但决定科学状态的是 PTB validator：`uv run awm ptb results <manifest> --all --json` 将该单元列为 `complete=false`、`completed_attempt=null`、`accuracy=null`；直接运行官方 completion validator 同样 exit 1，报告四项缺失：

1. `final_model/config.json` 缺失或为空；
2. `metrics.json` 缺失或为空；
3. `final_model` 没有模型权重；
4. `metrics.json` 不是非空对象。

官方 judge 中 contamination、disallowed model、disallowed API、PTB lookup 均为 `false`，但 `general_anomaly=true`。该 anomaly 的实质是 agent 在仍有约 7 小时预算、exp-03 训练刚启动时以 `end_turn` 提前结束；后台任务随后被 harness 杀死，既未完成训练也未把已评过的 exp-01 候选包装到 `final_model/`。

## Receipt → cell → manifest → spec → result

- Receipt（调度权威路径）：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/batches/wma-crossbench-opus48-r05-gsm8k-single-x4/formal-2026-09-04T140251.018528+0000.json`
- Receipt 仓库镜像：`/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-crossbench-opus48-r05-gsm8k-single-x4/formal-2026-09-04T140251.018528+0000.json`
- Cell/job：`w57r01` / `92198`，formal，replicate 1，run index 57
- Manifest：`/home/robtang_google_com/gangda_workspace/agentic-world-model/experiments/posttrainbench/wma-crossbench-opus48-r05-gsm8k-single-x4.yaml`
- Spec：`/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-opus48-crossbench.md`
- Result：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_200k_awm_claude-opus-4-8_10h_gangda_wma_evolve_wma-crossbench-opus48-r05-gsm8k-single-x4_w57r01_formal_r57/gsm8k_google_gemma-3-4b-pt_92198`
- 路由：`gangda_wma_evolve`，`slurm2-a3nodesetondem-3`，receipt-backed，节点合法。
- 冻结来源：top `225bd584f35ecaf0ec3fac4c2fb02d946030180c`，PTB `e62036f0c244995a6f45496522d3310b239383c6`；AWM/WMA checkout SHA `31b854bbc5e1f7f66685a8ec0d43845a6c2472f1`；base model revision `cc012e0a6d0787b4adcc0fa2c4da74402494554d`。

## WMA 阻塞评审

三个实验卡的 preflight 都是 `9 pass / 0 warn / 0 fail`，但三次 WMA 评审均 **未产生 verdict**：

| 卡 | request | wait | 终态 | 后续动作 |
|---|---|---:|---|---|
| exp-01 | `20260904T204048.448568Z-11f1767b` | 190.2 s | Vertex 429 `RESOURCE_EXHAUSTED`，`verdict_path=null` | 显式记录 `proceed` fallback |
| exp-02 | `20260904T210910.097813Z-97acc62b` | 195.2 s | Vertex 429 `RESOURCE_EXHAUSTED`，`verdict_path=null` | 显式记录 `proceed` fallback |
| exp-03 | `20260904T230835.641467Z-c23d48cb` | 190.2 s | Vertex 429 `RESOURCE_EXHAUSTED`，`verdict_path=null` | 显式记录 `proceed` fallback |

429 原因一致：`online_prediction_input_tokens_per_minute_per_base_model` 对 `anthropic-claude-opus-4-8` 配额耗尽。三个 `.lock.json` 的 `wma.state` 都是 `failed`，三个 action 的 `review_state` 都是 `failed`。因此不存在可报告的 WMA verdict flags；`proceed` 是 agent 的失败回退动作，并非 WMA 接受结论。按当前 operator 的严格阻塞要求，这三次都不能表述为“WMA 阻塞评审完成”。

原始证据位于 result 下的 `task/memory/cards/exp-0{1,2,3}.lock.json`、`task/.wma/actions/`、`task/.wma/responses/` 和 `wma_private/reviews/`。

## 模型、分数和终止链

- `task/eval/` 留有开发集诊断：base `0.0467 ± 0.0173`、exp-01 `0.5000 ± 0.0410`、exp-02 `0.0667 ± 0.0204`（均 n=150）。这些是 agent 的中间开发集诊断，**不是 PTB 官方结果**，不得进入正式 ledger。
- 归档后的 `task/ckpts/` 只有空的 `exp-03/`；整个 result 中找不到 `config.json`、权重、adapter、trainer state，且不存在 `final_model/`。exp-01 虽曾得到 0.50 开发分，但其 checkpoint 没有被保留或包装，故当前没有可用候选模型。
- 主 Claude 会话在 23:11:59 启动 exp-03 后台训练；23:15:46 确认训练健康但仅约 step 25/1218，23:16:12 以 `stop_reason=end_turn`、`terminal_reason=completed` 结束。23:16:17 harness 杀死后台训练；日志止于 step 35/1218，`ckpts/exp-03/` 为空。
- 随后 official judges 完成；final evaluation 从 23:23 至 23:29 连续重试 9 次，每次都因缺少 `final_model/config.json` 而无法启动 vLLM。Slurm 最终于 23:29:14 以 exit 1 标记 FAILED。

## 已花时间与成本

- Slurm：20:28:46–23:29:14，elapsed `03:00:28`；分配 1×H100、16 CPU、128 GiB，因此为 **3.0078 H100 allocation-hours** 和 **48.124 CPU allocation-hours**。这是分配量，不等于持续 GPU 利用率。
- harness `time_taken.txt`：`02:47:39`；主 Claude trace `duration_ms=9,990,524`（约 2:46:31）。
- 三次失败 WMA 等待合计 `575.6 s`（约 9:36），API cost 均为 `$0`（请求最终 429、未产出 verdict）。
- 主 Claude 已记录费用 `$16.58684725`；四个官方 judge 已记录费用分别 `$1.232019`、`$0.94548575`、`$0.6072015`、`$0.6442175`，合计 `$3.42892375`。可见 API 费用总计 **`$20.015771`**，不含集群内部硬件折算价。

## 判定

将 job 92198 保持为独立的 **FAILED + PTB incomplete + general_anomaly** 尝试；不得收割为 complete，不得使用其中间分数，不得将 WMA 429 回退解释为评审通过。若未来重试，应是新 receipt/new attempt，并在开跑前取得真实 WMA verdict；本核验未对队列或仓库作任何修改。
