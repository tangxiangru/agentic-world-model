# 新完成F四个cell：是否真的因WMA放弃了好方案？

本轮读取 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/new-cells.json` 中F的w12r01–04，原始job91177–91180。所有原始result_dir、card/verdict路径以及检出的负面summary见 [F-negative-index.json](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/F-negative-index.json)。scientist原始全文逐字解压为本目录`w12r0N-solve.txt`。未重扫其他cohort。

## 结论先说

**F提供了两个明确“WMA建议→scientist放弃原训练计划”的新案例，即使两条L3最终都是yes。** 因而只统计L3 no/defer会漏掉实际对探索机会的压缩。但这两条原计划都没有完整执行，不能证明被放弃的方案本来会提高分数。

其中w12r03/exp04更值得怀疑：WMA让scientist用**旧数据、旧训练调度的尾段趋平**推断**新100k数据（30%新MATH领域）+新训练调度**的边际价值小，scientist将“削弱论据”升级为“已证伪”，直接放弃。这个判据没有测到新方案的效果。

另外一条显式L3 defer（w12r03/exp03）是保存配置问题；scientist验证、修复后实际执行并提高了分数，并非把训练永远压掉。

## 覆盖与分母

|cell|terminal cards/verdicts|response records|stdout中标准verdict行出现次数|terminal L3|未启动训练卡|
|---|---:|---:|---:|---|---|
|w12r01|7|12|12|7 yes|无|
|w12r02|6|9|7|6 yes|exp04|
|w12r03|6|9|10|6 yes|exp04|
|w12r04|6|8|8|6 yes|无|

共25份terminal verdict，L0/L1/L3最终全yes；38个response。37次stdout标准summary出现不能直接当unique-request分母：重复输出和关闭后的relock失败都会造成不同数量。明确检出的标准负面summary只有w12r03/exp03初审一条（8857行）；本分析额外逐条检查了所有card.execution/conclusion，才发现两条L3=yes的实际取消。

## 1. w12r03/exp04：新MATH混合训练被旧checkpoint平台判据取消

原result根：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179`。

### 原方案、WMA判据、实际取消

- **04:34:03**，原始输出确认`data/sft_v4.jsonl`100000行已准备好（[10635–10636](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:10635)）。
- **04:35:21**，scientist写原方案：从exp03/final继续1epoch，**70k进一步GSM8K-family教师解答+30k整数答案MATH-family**，lr6e-6、新调度。command是训练该100k文件，不是评旧checkpoint（[10794](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:10794)）。card.situation写剩余**5.6h**，planned_h **1.6h**；不是已经无法容纳的一次训练。
- **04:41:19–22**，WMA返回L3=yes@.55，但第一precondition要求先评exp03/checkpoint1600。它的判据是：若在final0.7635的±0.012内，则前轮曲线已平、再训100k边际价值小，削弱L2/L3（[10923–10935](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:10923)）。
- **04:41:42起**，scientist实际构建ck1600可评分目录并评full1319（[10971](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:10971)）。结果**0.7665 vs前轮final0.7635**，两个旧checkpoint只差391steps/约12.5k旧行。
- **04:49:51**，scientist明确说曲线已平、削弱exp04前提，转去hard-example-mining采样（[11069](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:11069)）。
- **04:52:22**，原card被写成`execution:not_run`、`decision:abandon_line`，并声称该probe“falsified this card's premise”（[11124–11144](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:11124)）。全量100k训练从未启动。

### 为什么这是更高风险的“可能误杀”，而非已证明省算力

- 评的是**前一训练的数据/学习率衰减尾段**。原待评估方案还增加了MATH域、新解答和重新开始的学习率调度。旧段不涨不能识别这些新因素会不会有效。
- 而且ck1600=0.7665高于final=0.7635；card写“最后12.5k gained+0.003”连符号都写反了，实际是约−0.003，不过“近乎持平”的结论不受这个笔误影响。
- 旧checkpoint比较是有用的候选选择信息；将它当成“新100k方案已证伪”则超出证据。WMA原文是“weakens L2 and L3”，scientist采用了更强的“falsified→abandon”。
- 取消时卡记还有5.6h，planned1.6h。实际probe/card耗时约0.15h。可以说**1.6h计划未执行并被其他工作替代**，不能说已经量到净节省1.6h或误杀损失多少分。

### 替代路线后来怎样，原方案是否被补测？

随后exp05做失败导向RFT，训练wall_h0.85，full1319 **0.7748**；相对旧final0.7635高1.13pp，相对该probe发现的ck1600=0.7665高0.83pp。card记录相对ck1600 paired p=.505，scientist凭另一local-probe也改善而选择它。exp06又做一轮RFT，wall_h0.75，full1319 **0.7544**，其epoch1=0.7574，均低于exp05，保留exp05。最终PTB官方分数0.7717968。

这些是替代路线的实际结果，不是被取消方案的反事实上界或下界。完整轨迹中`data/sft_v4.jsonl`只出现在构建、原card、WMA建议的未执行smoke里，取消后再未出现；后两轮是新RFT文件。**没有后来把原100k/MATH方案补跑的证据。**

判定：**实际WMA相关的计划抑制成立；“因此压掉本可获益的计划”尚未被观察。** 这是最具体的潜在false-kill机制，不应被终态L3=yes掩盖。

## 2. w12r02/exp04：Orca小训练probe失败后取消90k全量

原result根：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178`。

- 原计划：从exp02继续60kOrca+30k教师anchor、共90k行，lr5e-6。该全量command已写卡并lock；WMA L3最终yes@.55，L2 lower[-.06,.02]。
- **05:51:11**：首precondition提出先训练20k行/~150steps再评watch400，fail条件<.82，pass条件watch≥.828且dev150≥.78（[9032](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178/solve_parsed.txt:9032)）。WMA还要求修过滤不一致的Orca行；scientist确实重建90k文件，故probe用的是修后的mixture。
- **05:52:04**：实际head20000到scratch，以同脚本训练1epoch（[9125](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178/solve_parsed.txt:9125)）。
- **06:14:47**：工具输出watch400 **0.790**，低于incumbent **0.835**和.82停止阈值（[9567](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178/solve_parsed.txt:9567)）。probe训练159steps约19分钟，连评估/准备card记wall_h.45。
- **06:21:53**：写`not_run/abandon_line`，全量90k从未跑（[9669–9701](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178/solve_parsed.txt:9669)）；06:22:18的relock明确是为关闭修后数据hash，不是重启全量（[9738](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r02_formal_r12/gsm8k_google_gemma-3-4b-pt_91178/solve_parsed.txt:9738)）。后续转向独立from-base sibling及soup。

此处的停止证据比w12r03更直接：**确实在拟议新mixture上训练过并测到退化4.5pp**。但仍没有full90k结果；20k-head的独立1epoch调度也不等于90k完整训练的前22%（调度长度不同）。不能把short-probe负值当成完整计划必败的证明，或断言它确实误杀好计划。

判定：**实际由WMA判据触发的取消；短程实测支持谨慎停止；完整训练的机会成本和最终效应仍未知。**

## 3. w12r03/exp03：显式defer后修复并执行，不能计为永久否决

- **02:50:15**，初审L0=no@.85、L1=no@.75、L3=defer@.7，首条件检查继承parent的GenerationConfig会在保存时失败（[8857–8873](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:8857)）。
- scientist复现保存问题，同时修正probe250排除逻辑，做25-step save smoke；checkpoint20/25和final实际写出（[9239–9243](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:9239)）。
- **03:03:23**重锁理由点名WMA两处问题（[9375](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:9375)）；**03:09:17**变L0/L1/L3 yes（[9445](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-f-decisive-precondition-x4_w12r03_formal_r12/gsm8k_google_gemma-3-4b-pt_91179/solve_parsed.txt:9445)）。随后原训练目标被执行，最终full1319 **0.7635 vs0.7096**。

这是技术前置条件促成修复，而非取消有益训练。原负面预测对应未修的代码，不能拿修后成功判它原预测错误。

## 4. 对“WMA veto是否压低分数”的准确回答

- **会抑制实验机会：已有两条清楚实例。** 抑制可通过precondition的失败解释发生，L3仍可能是yes，所以L3负面计数不是充分测量。
- **已确认永久压掉好方案并降低终局分数：F这两条还没有反事实证据。** 原方案没跑，替代路线分数不能回答其本来会怎样。
- **最值得复核的错误机制：** 把旧checkpoint在旧数据上的尾段平台，外推为新mixture/新schedule无价值。w12r03/exp04就是有原方案、具体判据、实际probe、明确取消的完整链。
- 历史R1两例见 [historical-negatives.md](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/historical-negatives.md)：w01r07是错预测但原训练照常完成；w04r04是确认保存故障后中止修复并重跑。两者也不能单独当“已误杀好训练”的证据。
