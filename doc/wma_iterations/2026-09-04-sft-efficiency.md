**按完整 SFT 次数比较 WMA 的实验效率**

已经把同一冻结快照中的 **64 条轨迹、472 张卡、655 条卡内 accuracy 记录和64个最终官方分数** 对齐到完整训练次数。当前对照数据没有稳定显示“有 WMA 能够用更少次完整训练获得更高分”：首轮 WMA 用的次数更多、最终分接近；阻塞审查组次数相同、WMA 最终均分较低。个别候选或轨迹有很好的早期分数，但不能概括为 WMA 的整体效率优势。

数据快照：2026-09-04 05:52:48 UTC，沿用上一轮逐 manifest PTB validator/judge 核验的64条完成轨迹，保留原 receipt/manifest/spec/result provenance。未将后来可能完成的作业加入本次比较，未重跑模型或修改实验。

**可直接查看的产物**

- [交互查看器：按批次、SFT口径和题量切换，点击每条traj看逐卡分数](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/viewer.html)
- [64条轨迹逐条表格及472张卡原始分数](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/all-trajectories.md)
- [全部轨迹曲线 PDF](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/all-trajectories.pdf)
- [每条traj汇总 CSV](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/trajectories.csv)、[逐次分数 CSV](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/measurements.csv)、[首次观察到分数门槛的训练次数 CSV](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/thresholds.csv)

![两组对照的训练次数效率](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/efficiency-comparison.png)

**计数与分数口径**

一次正式训练计划达到预定优化终点，计一次完整训练。两个epoch计一次，多个checkpoint或对同权重多次评测不会增加次数。重启后重新完整训练另计；优化已完成、仅最终保存失败，也计入消耗。未完成的OOM、人工中止，以及明确用于管线验证的smoke不计完整次数，另列。15k行pilot若有正式实验卡、达到其计划终点并产出正式比较结果，仍计一次，不能仅因训练短就删掉。

主横轴为 **SFT＋RFT 完整监督拟合总数**。这里RFT最终仍执行监督训练，而且部分标成SFT的卡实际混入了自生成样本。仅统计SFT会让换个标签的训练看似免费。查看器也提供原卡标签的SFT-only横轴；CSV同时给出SFT与RFT次数，数据混合标记保存在审计JSON。

卡片中成功完成的训练卡共207张；核验补回9次完整优化但保存失败/隐藏重训后，得到 **216次已确认完整训练（145 SFT标签＋71 RFT标签）**，另有至少27次未完成的正式训练尝试。R1的145次为卡片普查加重点原轨迹检查得到的已确认下界，不宣称穷尽全部历史launch；R2的基线/对照26次、候选45次单列核验。

分数按**观察时刻**对齐：完成第k次训练后，及下一次完整训练前已记录的评测，属于阶段k；后来才重评的旧checkpoint不倒填到早期。这里主要保留卡片级观察顺序，同卡多次完整重训后保存的结果归到这些训练都计费之后。阶段曲线是“当时已经测到的最高分”，不必等于scientist最后选中的模型。

不同题量分开画；150、500、1319题不接成同一条分数曲线。相同n下仍可能有不同decoder、并发或重复测量，图是记录层面的描述统计，不是固定配置的模型质量曲线。†表示没有新的同题量观测、仅保留之前的最高读数；—为缺测。正式最终1319题分数独立画黑点，不能用150题的80%替代最终官方分。

**整体比较：完整训练次数与最终官方分**

| 对照窗口 | 组 | traj数 | SFT标签总次数 | RFT标签总次数 | 完整训练均次/traj | 最终官方均分 |
|---|---|---:|---:|---:|---:|---:|
| R1原始core | WMA v0.2 | 8 | 25 | 6 | 3.875 | 75.65% |
| R1原始core | 无WMA | 8 | 19 | 8 | 3.375 | 75.23% |
| R2阻塞审查 | WMA v0.2 | 4 | 10 | 3 | 3.250 | 71.34% |
| R2阻塞审查 | 无WMA | 4 | 8 | 5 | 3.250 | 74.30% |

R1按当前核验次数，WMA平均多0.5次完整训练，最终只高0.43pp。R2双方都完成13次正式监督拟合，最终WMA低2.96pp。仅按原SFT标签看，R2也是WMA2.5次、对照2.0次，不能得出“更少SFT”结论。小样本不能证明WMA一定降低效率，但这里没有观察到所希望的整体优势。

同样次数的最终分数分层也没有消除选择偏差：R1恰好3次的WMA只有2条，对照5条；恰好4次为5对3。R2恰好3次为3对1。训练次数由轨迹自己选择，与难度、早期成绩和剩余时间有关，不能把这种事后分层当随机处理效果。

**同次数的早期曲线：有无被最终分掩盖的优势？**

| 窗口 | 完整训练次数k | WMA已测最高n150均分 | 无WMA已测最高n150均分 | WMA−对照 |
|---|---:|---:|---:|---:|
| R1 core | 1 | 69.58%（8条） | 72.76%（7条） | −3.18pp |
| R1 core | 2 | 75.08%（8条） | 74.95%（7条） | +0.13pp |
| R1 core | 3 | 76.25%（8条） | 75.33%（7条） | +0.92pp |
| R2阻塞 | 1 | 70.17%（4条） | 72.34%（4条） | −2.17pp |
| R2阻塞 | 2 | 71.84%（4条） | 72.50%（4条） | −0.67pp |

R1的c01r05使用n200/n600，不强行换算到n150，所以对照只有7条。R1第二次几乎持平，第三次有约0.92pp的描述性差异，但不是“更少完整训练”的明确证据。R2早期两点也没有同次数优势。

更高k需格外谨慎：R1第4次只剩WMA6条、对照3条，第5次为1对0；R2第3次为4对3、第4次1对2。图中的组均值可能因为剩下的轨迹变了而下降，不代表某个模型的历史最好分下降。

**R2基线与对照逐条展开**

下表按SFT＋RFT完整总数排列。n150序列是阶段末已知最高分；其他题量不能拿来填空。

| traj | SFT＋RFT | 完成第1次后n150 | 第2次后 | 第3次后 | 第4次后 | 最终官方1319 |
|---|---:|---:|---:|---:|---:|---:|
| w10r01 | 2＋1 | 70.67 | 72.00 | 74.00 | — | 69.45 |
| w10r02 | 2＋1 | 68.00 | 68.67 | 72.67 | — | 71.72 |
| w10r03 | 3＋0 | 69.33 | 74.00 | 74.00 | — | 71.95 |
| w10r04 | 3＋1 | 72.67 | 72.67 | 72.67 | 74.00 | 72.25 |
| c10r01 | 3＋1 | 73.33 | 73.33† | 73.33† | 73.33† | 81.50 |
| c10r02 | 2＋1 | 72.67 | 72.67 | 77.33 | — | 76.72 |
| c10r03 | 1＋1 | 68.67 | 69.33 | — | — | 70.20 |
| c10r04 | 2＋2 | 74.67 | 74.67 | 74.67† | 74.67 | 68.76 |

c10r01的n150横线不是模型一直只有73.33%；它后面改用n500，序列是 **78.0 → 81.0 → 81.0† → 83.2**，最终官方81.50%。这正是必须分尺展示的原因。c10r01第3次训练完整完成但保存失败，c10r04一张RFT卡里有两次完整训练；这些都已计入。

**候选与具体高效率轨迹**

| R2候选 | traj数 | 平均完整训练次数 | 第2次后最高n150均分 | 最终官方均分 |
|---|---:|---:|---:|---:|
| A | 4 | 3.00 | 72.00%（3条有n150） | 69.09% |
| A+B | 4 | 3.00 | 73.17%（4条） | 71.78% |
| C | 4 | 2.75 | 73.56%（3条有n150） | 71.80% |
| D | 3/4完成 | 3.33 | 75.11%（3条） | 73.29% |

这些点可以发现值得继续读的局部机制，但既不是完整候选确认，也不能合并成一个“大WMA组”：候选改动不同、覆盖不同，D还未完整且w09r03有独立数据访问审计标记。

`w08r04`的原始进展确实漂亮：第一次完整SFT后的模型n150为75.33%，同权重greedy仍75.33%，随后选已有epoch-1 checkpoint到80.00%；这三步都只消耗**一次完整训练**。第二次完整SFT的新模型先得77.33%，扩大比较后n500达81.2%，完整集合与后续soup观测最高79.98%。第三次RFT得77.33%（n1319），未超过已持有候选，最终官方79.38%。这说明按次数能看见“无需再训练的选择收益”，同时也看见额外训练未创造更优模型。

对照也有早期高分：`c10r01`第一次完整SFT＋解码选择后n500已达78.0%，第二次RFT后81.0%。因此不能从w08r04这一个例子直接归纳WMA才有这种效率。

**如何解释“效率”**

这次给出的两个直接观察是：同次数下的已记录分数曲线，以及最终分数对完整训练总次数的散点。它们比verdict采纳率更贴近用户关心的结果。首次观察到70/75/80%门槛的k也已输出，但缺测标为缺测、不当失败；不能拿后来重评的成绩回填到更早时刻，假装当时已经知道能达到门槛。

次数仍不是等量算力。一轮15k行训练与一轮150k行×2epoch都记一次；RFT还可能有生成成本。CSV保留训练卡reported wall、数据规模、步骤、未完成尝试与来源，供后续按真实GPU时长/token预算比较。当前reported wall混有生成、评估或等待，不能直接当纯训练GPU小时；本次没有用“分数÷次数”的比值冒充科学效率。

**核验与复现**

每条完成轨迹的卡片均进入统计；独立审阅核对R1、R2基线/对照、A–D三部分的正式训练次数，重点回读隐藏重训和保存失败。644条存在标量JSON的引用分数作了对照，唯一超过0.5pp的差异为c01r08/exp08的三次评测均值引用首跑文件，原轨迹已解释；保留卡片均值、单独标注aggregation，没有把首跑幸运高分替换进去。

审计来源：[R1历史](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/audit-historical.md)、[R2基线与对照](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/audit-baseline-control.md)、[候选](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/audit-candidates.md)。图表、审计override、输入快照和提取脚本都在 [数据目录](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency)。查看器JavaScript语法及全部批次/横轴/题量渲染逻辑组合通过检查；主图已人工查看。原实验、skill、评分器和队列没有改动。
