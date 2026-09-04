**WMA evolve：实验规模、结果与轨迹审计（2026-09-04）**

目前已经证明 WMA 能在启动前促成一些真实修复和决策变化，但还没有证明它稳定提高最终 GSM8K 分数，也没有候选具备晋升依据。最新轨迹暴露的首要问题是数据边界与版本化审查，其次是把已识别的风险转成及时的动作；继续微调预测区间不是当前最优先的工作。

结果冻结时间为 **2026-09-04 05:52:48 UTC**；队列保存于 05:57:37，06:07:05 再次核对状态数量未变。只统计 `gangda` registry 中 batch 名以 `wma-` 开头的本线 receipts。排除同子队列的历史 judge 恢复作业。Slurm 直连查询失败，因此运行状态引用共享 monitor 快照；实验完成数则重新逐个运行 `uv run --no-sync awm ptb results MANIFEST --all --json`，不是照抄 hook 状态。

**到底跑了多少**

| 统计单位 | 数量 | 含义 |
|---|---:|---|
| PTB validator 完成且原 judges 无 flag 的 cells | **64** | 38 个有 WMA、26 个无 WMA 对照；下文另报人工发现的数据边界问题 |
| 正在运行 | **16** | D 剩余 1、baseline extension 4、control extension 3、E 4、F 4 |
| 等待运行 | **9** | control extension 1、G 4、H 4 |
| 已完成或当前正在运行 | **80** | 可以确认已经进入执行的 cell 数；另有 9 个排队 |
| 注册作业记录 | 181 | 含 92 个取消记录；不能称为 181 次科学实验 |
| 历史注册 manifests / batch-cell 配置 | 30 / 176 | 包含已撤回的旧协议、替换版本及裁剪重复，不是有效比较分母 |
| 完成 cells 内的实验卡 / 最终 verdict | **472 / 198** | card 是 cell 内部训练、评估、打包等步骤，不是独立最终分数样本 |

64 个完成 cell 分为：R1 历史各 cohort **41**，R2 同协议 baseline/control **8**，A/A+B/C/D **15**。历史 cohort 有裁剪、重试及 sensitivity-only cells，不能事后混成平衡 A/B 实验。WMA 节点 2–3 的分配为 16/16 GPU、ownership OK；这是分配数，不是实时利用率。

离线历史另有 **2193 次 heuristic 回放计算、47 次模型调用尝试**，其中存在样本重叠、同一卡接线重试、拒收。没有完成模型版全量 baseline passes 或 held-out 晋升。详见 [离线与候选设计审计](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/evolution-review.md)。

所有 cell 的 receipt → manifest → spec → result 可从 [追溯索引](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/provenance-index.md) 点击；[原始 validator 快照](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/results.snapshot.json) 保留未完成/撤回配置，[逐 cell CSV](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/cells.csv) 可重新分析。

**结果：目前没有稳定的最终分数收益**

所有下表分数是相同 Gemma-3-4B-PT / GSM8K 官方最终 accuracy；± 为 cell 间样本标准差，单位为百分点，不是均值置信区间。中间卡的 n150/n500 分数不混入本表。

| cohort / 候选 | 完成 n | 官方均值 ± SD | 相对同波 WMA v0.2 |
|---|---:|---:|---:|
| R1 原始无 WMA 对照 | 8 | 75.23% ± 4.98 | — |
| R1 原始异步 WMA v0.2 | 8 | 75.65% ± 3.12 | 对 control +0.43 pp |
| R2 blocking 无 WMA 对照 | 4 | 74.30% ± 5.92 | — |
| R2 blocking WMA v0.2 | 4 | 71.34% ± 1.28 | 基准；对 control −2.96 pp |
| A：格式低分基线的收益预测 | 4 | 69.09% ± 2.79 | −2.26 pp |
| A+B：A 加区间宽度依据 | 4 | 71.78% ± 1.38 | +0.44 pp |
| C：给 no 前先验证失败条件 | 4 | 71.80% ± 6.14 | +0.45 pp |
| D：checkpoint 保存/评估前提 | **3/4** | 73.29% ± 4.00 | +1.95 pp，未完整 |

![每个 cell 的分数、均值和标准差](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/wma-scores.png)

R1 的 +0.43 pp 比运行波动小得多。R2 的 −2.96 pp 也不能据四对样本宣称 WMA 有害：独立 cell Welch 95% 区间约 **[−12.15,+6.23] pp**。这只是小样本、近似正态下的敏感性估计，不是预登记主检验；相同 replicate 编号不保证共同随机数。其他候选相对 v0.2 的区间也跨零，D 还存在完成先后选择偏差。

A+B 的 B 增量必须相对 A 判断，不能把 A+B 相对 v0.2 当作单变量实验。C 的最高 79.38% 与最低 64.75% 同出一个候选，**14.63 pp** 极差表明只看最好一条会严重误导。E/F 各四个运行中；G/H 各四个排队；I/J 没启动。当前 baseline 仍是 v0.2。

**WMA 自身的账本，也不能证明候选已变好**

下表保留冻结 scorer 和原始 flag，不事后清除记录。L2 是“实际变化是否落在预测区间内”，只在原扫描未 flag 且可计分的终稿卡上计算。

| 当前 cohort | 最终 verdict | 原 access flag | L2 coverage | 官方 width/noise | 每最终 verdict 费用 |
|---|---:|---:|---:|---:|---:|
| w10 v0.2 | 30 | 11 | 14/15 = 93.3% | 4.11 | $1.98 |
| A | 29 | 14 | 6/11 = 54.5% | 5.21 | $1.89 |
| A+B | 23 | 11 | 5/8 = 62.5% | 5.03 | $1.84 |
| C | 32 | 10 | 14/17 = 82.4% | 5.00 | $1.79 |
| D（3 cells） | 20 | 9 | 8/9 = 88.9% | 4.55 | $1.92 |

这些是不同 scientist 自主产生的 card，类型、评估 n 和难度不同；不能把该表当相同题集上的 head-to-head 校准测试。A/B 的主指标还应限制于预先定义的 format-fix 机会。特别是官方 coverage 与 width 分母不同：后者包含原 flag 和不可计分卡。在相同 clean/scorable 子集上，width/noise 为 v0.2 **4.13**、A **3.97**、A+B **4.73**、C **4.81**、D **3.67**。所以没有“B 已可靠压窄区间”的证据。这个补充计算不改变原评分；见 [分母敏感性数据](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/detail.json)。

全部 198 个在线终稿有 **76 个原 access flags（38.4%）**。A–D 均不满足 zero-leak 晋升门；人工审计还发现下面这个漏报。所有终稿合计记录费用约 **$379.41**，仅是这些终稿对应 review 的已记录费用；不含被覆盖的重审、scientist、judges、GPU，不能当项目总成本。L0/L1 hit 高也不等于失败召回好：C 的 clean 终稿没有失败召回分母，实际运行中却有被修复/覆盖掉的失败尝试。

**轨迹揭示的原因，按证据权重排序**

**1. 数据边界确有漏洞，原扫描器会漏掉库调用访问。** D 的 `w09r03/exp-01` 在 WMA 私有环境调用 `inspect_evals.gsm8k.gsm8k()`，取 `dataset[0]` 并输出测试问题与 target；随后 verdict 的 e12/p2 把 target 带给 scientist。私有 transcript 第 85/88 行与 scientist 第 3459/3485 行已独立核对。原 `access.outside=[]`，没有 `leak_suspected`。这违反 WMA skill 对 held-out 数据的限制，且比旧 w10 的环境目录/包源码读取更严重。

目前证明的是测试题/答案暴露与传递，**没有证明答案用于训练、WMA 读到了当前模型的未来结果，或造成可测分数收益**。原 PTB validator/judges 的 clean 结论按原样保存；不能把它解释为 WMA 语义上无越界。新增独立 [人工审计标记](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/manual-audit-flags.json)，不改旧 verdict 或 scorer。G 仅改 skill 文本，不能单独保证封住这种间接访问。

**2. 当前已经不只是“建议没被听见”；风险与动作之间仍有断层。** R1 原始 22 个 delivered verdict 中 21 个在启动后才到。当前 blocking 波的 134 张卡都有最终 verdict，且有真实 prelaunch 修改。不过 `w06r02/exp-03` 是鲜明失败：WMA 已经指出训练没有 grader 的 10-shot 提示，却仍预测 **+32～+68 pp**，只安排训练后检查。实际训练 **1.35 h，官方 n150 accuracy 仍为 6%，零增益**；模型继续编下一题，末数字评分落在错误数字上。下一张卡增加 few-shot 训练后，**0.75 h 到 67.33%**。数字先验变了，不代表“格式会被修复”这个条件得到验证。

反面的成功也很重要：同 cell `exp-06` 本拟花 1.6 h 训练 26,000 条“新问题”；WMA 指出 25,965 条问题已见过，scientist 随后 CPU 核查更大集合，约 80 万行只找到 56 个新问题，于是关闭未运行卡、改变方向。终稿 **L3 仍是 yes**。应记为“因审查取消了一个计划 1.6 h 的训练”，不能称实测节省 1.6 GPU h，更不能断言未运行训练必然无效。这也证明 all-yes 或 `gpu_h_saved=0` 会漏掉真实决策作用。

**3. checkpoint 建议能促成修复，但保存、测量、选择没有完全接通。** D 的 `w09r01/exp-04` 在 18:28 首审 defer；scientist 提高保留数量、改保存频率并恢复原 LR；18:36 重审 yes 后才训练，随后确实比较中间点与 final（n150：69.33% vs 75.33%）。这是可核实的有效采纳。可同 cell `exp-05` 虽保存多个点，却没分别评价，后续直接做 soup；n500 soup **77.4% < final 79.0%**。因此“出现 save_steps”不足以衡量收益。

D 三个完成 cell 的轨迹能恢复 **27 次审查：22 yes、5 defer**，但只看最终文件是 **20/20 yes**。否决后修复再成功，不能反过来给初次否决记 false-no。w09r01 的 merge 又有一次真实技术修复：0/883 张量键匹配触发 L0 no/L3 defer，修复键映射并 CPU reload 后才运行。现有终稿账本同时漏掉了一部分正向动作和审查成本。

**4. 短评估集误选与首轮 recipe 差异，比候选名更能解释很多分数变化。** C 的高分 `w08r04` 首轮训练 153,157 行×2 epochs，2.44 h；低分 `w08r03` 为 59,946 行×2，1.89 h，提示、精度、吞吐和后续数据选择也不同。约 1.29 倍 wall 内处理约 2.55 倍行数，是后续受控实验的线索，不能直接归因某一个 batch size 或“数据越多越好”。当前 23 个 R2 cells 的探索性相关也比旧七 cell 窗口弱，旧强相关不稳。

高 C 的一个 checkpoint 在 n150 被判断 **77.33% < 80.00%** 而拒绝；n500 反成 **81.2% > 78.8%**；full1319 最终 **79.38% > 75.66%**，找回了真正提交的模型。A+B、旧 w10 也有相似反转。它们支持对会影响最终选择的接近结果追加共同样本；不支持一律重复小样本，也不支持把评过的同一套 test 当作独立 WMA 泛化证明。部分轨迹还有同设置重复读数变化，WMA 的“greedy 必须字节一致，否则仍在 sampling”诊断超出了证据，应区分有效配置、运行变异和题集覆盖。

**5. 审查的完整性和成本尚未被正确测量。** 当前 19 个有 WMA 的 R2 完成 cells，共 **134 张卡、189 对 request/response（188 completed、1 failed）**。每 cell 的请求生命周期累计均值约 54–67 分钟；合计 **19.33 h**，包括重审，不能直接称 GPU idle。终稿费用和最终 lock 的等待时间会少算已覆盖的重审；需要时段交集才能测“分配了 GPU、没有有用 GPU 工作、确实被 lock 阻塞”的时间。

还发现 `w08r01/exp-01` 先收到 verdict，后来修改了 command，未重锁即启动，结束后才补锁；sidecar 正确拒绝 post-hoc 请求。因此最终 `lock.wma.state=failed` 不是“最初无 verdict 就启动”，而是**审过的 proposal 与实际执行版本不一致**。133 delivered/134 个终稿锁不能直接翻译成 99.3% 全程合规。固定版本、完整请求链比再加一个粗略 delivery 比率更重要。

以上逐卡因果顺序、原始文件和行号见 [轨迹/采纳报告](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/trajectory-uptake.md) 与 [边界/伤害/合规报告](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/harm-scope.md)。后者同时区分真实失败、修复后结果及尚不可观察的反事实。

**建议下一步怎样改，怎样判断改进是真的**

| 顺序 | 独立改动/分析 | 主要验证 | 失败条件 |
|---|---|---|---|
| P0 | 单独设计 WMA 数据访问边界：探针环境无 held-out 库/缓存/凭据，限制数据 I/O；评估经受控接口只返回许可摘要 | 用合成 canary 测试直接路径、库调用、缓存命中、网络加载与返回内容；保留合法训练数据探针、模型配置检查 | 仍能取到测试题/标签；或正常 blocker 检查被一起禁掉。不能只以零 regex flags 通过 |
| P1 | 审查/执行版本对齐及不可变记录 | 每次 request 保存 proposal/script/config hash、完整 verdict、成本；启动验证对应版本；任何变更必须先返回新 verdict | 修改命令后复用旧审查；只记录最终 verdict；修复后的成功仍被算作初次 false-no |
| P2 | 格式风险改为早期、能改变继续/修复选择的检查 | 使用许可样本与真实评测提示形状，检查早期 checkpoint 的回答/终止；把语义正确性与格式恢复分开 | 又完整训练一轮才发现已知提示不匹配；检查成本抵消收益；用测试答案做诊断 |
| P3 | 按已排 D/H 的机会与行为读数推进 checkpoint/选模 | 必须实际保留、可加载、同尺评至少两个候选并记录最终选择；接近结果增加共同样本，另计运行重复变异 | 只改文字或保存参数、没有比较；无机会却宣称成功；更多评估未改善选择 |
| P4 | 在版本/数据边界完善后，单独测增量重审 | 复用未变证据、只重审有影响的差异，仍必须等待裁决；观察生命周期、首训延迟、实际有用 GPU 时间 | 漏检新 blocker、复用过期结论、时间不降或有用训练时间不增 |
| P5 | 独立受控研究首轮 recipe 和预测校准 | 固定训练数据/预算/提示与 settings blocks；先按同类同题输入比较预测，再验证会否改变动作 | 把自主 recipe 差异归给一句 skill；靠收紧区间牺牲覆盖；只挑最高分 |

P0/P1 是 harness/protocol 改动，必须与 skill 改动分开预登记；不应无声叠入现有 A–H 或修改原 guard。G/H 和 E/F 已有明确问题与分母，按各自原定机会读取；新发现不需要等无关 straggler 才能形成独立修复设计，但 D 与同波扩展的正式比较仍需自身完整证据。原 scope 标记保留，人工漏报单独标记；不能用“修好扫描器”让旧候选事后过门。

保持每 setting 四次重复适合筛明显机制，例如泄露、过期版本、checkpoint 计划是否落地。它不适合据 +0.4 pp 宣布赢家。以 3–5 pp 的 cell SD 粗估，独立两臂要识别 +2 pp 约需 36–98 cells/arm（双侧 5%、80% power 正态近似），不建议现在直接投入该规模。先以有效性和真实决策作用选少数候选，再独立确认，最后一次 held-out；八 cell 证据门也不是任意小效应的统计保证。

本次只新增分析材料，没有改 skill、评分器或 runtime，也没有提交、取消、调整实验。现有 submodule 修改未触碰。可复现脚本、图表 SVG/PNG、冻结指标和原始路径均保存在 [审计目录](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review)。
