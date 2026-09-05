# WMA 到底增加了什么：基于已核实轨迹的作用分解

本备忘只使用上一轮读过的 validator-clean 轨迹，不新增队列操作或大规模扫描。主问题是独立 WMA 相比同型号 scientist 的边际价值，而非建议是否听起来合理。

## 判断

当前证据足以证明 WMA 能充当有额外阅读和探针预算的独立审查者，并在某些场景补进 scientist 未检查的事实、促成修复或更好的比较。尚不足以证明它作为“世界模型”的反事实预测、价值排序、历史迁移，优于同等预算下 scientist 自审或普通 reviewer。

同一模型并不意味着零增量：scientist 已投入当前方案、有自己的上下文和工作优先级；WMA 独立看到锁定方案，另有历史与探针，可能搜到不同证据。其增量可能来自不同信息、不同注意力分配、额外计算或独立样本。现有 WMA vs no-WMA 对照把这四者与阻塞等待一并改变，分不清是哪一个在起作用。

## 六个作用案例

### 1. 补充未检查的事实：问题级 novelty，而非一味建议“多数据”

[w06r02/exp-06](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003/task/memory/cards/exp-06.yaml) 的 scientist 拟训练26,000条“新问题”数据，计划1.6h。WMA具体核对到25,965条问题已见，scientist随后做较大CPU探针，只找到56条未见问题，于是卡不启动、改做RFT。

- **支持的作用：** 新证据进入决策。原方案的机制前提被实测否定；不是 WMA 单凭悲观语言让 scientist 服从。
- **WMA 独特性：** 当前 scientist 没有做这次交叉核对，独立审查确实补上了。但交叉去重本身可以自动化；它不是只有预测模型才能完成的推理。
- **不支持的主张：** 被取消训练必然无收益、节省了实测1.6 GPUh、RFT一定优于继续SFT。未运行候选没有观测结果，第二答案多样性可能仍有价值。
- **判别实验：** 比较“普通reviewer+确定性问题去重结果”与完整WMA，固定审查预算。若同样促成合理的机制重写/取消，增量主要来自工具化事实而非 WMA 专属预测。

### 2. 可执行性批评：抓住 preflight 没覆盖的代码路径

[w09r01/exp-06](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/task/memory/cards/exp-06.yaml) 首次审查指出 soup.py 的 on-disk tensor 名与 live module 名0/883匹配，L0=no、L3=defer；scientist 改成按源键写权重并CPU reload，重锁后才执行。

- **支持的作用：** 审查者连接了代码、实际产物和加载语义，推动具体修复。原 preflight 只查输入、路径、stop token，不执行merge路径。
- **WMA 独特性：** 比当前固定 preflight 更广的代码审查能力；但可由普通coding reviewer或CPU round-trip smoke完成。它不要求历史收益预测。
- **不支持的主张：** 几小时训练GPU被挽救，或首次 no 是误杀。原计划主要是streaming/merge/eval，且终稿结果属于修复后方案，不能拿来反证原来的no。
- **判别实验：** “相同CPU smoke结果+普通reviewer”对“完整WMA”；记录新缺陷命中、误报、修复验证成本，以及需要LLM才能发现的剩余缺陷率。

无WMA对照 [c10r01/exp-05](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ctl-x4-v2_c10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90998/task/memory/cards/exp-05.yaml) 真的有.98h训练在save_pretrained失败后丢权重，说明产物完整路径检查有现实价值；但不构成“WMA肯定会救这一次”的反事实证明。

### 3. 反事实收益预测：是应有的独特作用，当前有明显断点

[w06r02/exp-03](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003/task/memory/cards/exp-03.yaml) 得到 L1=yes、L2 +.32..+.68，1.35h训练后正式分仍.060；前一轮WMA已发现训练0-shot与grader10-shot不同，却只要求训练后读续写诊断。[w06r02/exp-04](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-a-format-floor-x4-v2_w06r02_formal_r6/gsm8k_google_gemma-3-4b-pt_91003/task/memory/cards/exp-04.yaml) 用12k few-shot继续训练后正式分.6733。另一例 [w07r02/exp-02](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r02_formal_r7/gsm8k_google_gemma-3-4b-pt_91008/task/memory/cards/exp-02.yaml) 实际接受了WMA的提示格式修复并重建数据，但最终+72pp超过终稿L2上界+62pp。

- **支持的作用：** WMA能列出改变结果的条件，也能影响方案；它的绝对/增量预测不是这些行为效用的同义词。
- **当前缺口：** 从“如果输出契约真正修好，可有大收益”跳到“当前方案会修好”，未可靠建模干预生效的概率。因此进一步调窄/调高L2数值未必改进决策。
- **不是所有失败都是浪费：** exp-03揭示了真实prompt条件失效，并引导有效修复。问题是同样信息本可否通过更便宜的早期checkpoint、真实grader提示下诊断取得；应比较单位信息成本，不能只因delta=0就判浪费。
- **判别实验：** 同一冻结proposal比较完整WMA、只给风险条件/行动建议的reviewer、scientist自预测；先离线检验预测校准，再随机决定是否把数值预测展示给scientist。保持建议正文一致，才能测出“数值预测本身”是否改变更好的选择。

### 4. 成本与信息价值排序：checkpoint不是必须赢才有用

[w09r01/exp-04](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/task/memory/cards/exp-04.yaml) 首次defer促成save_limit与保存频率修复，后来真的评到中途checkpoint .6933 < final .7533。这个负结果回答“是否训练到终点反而更差”，是可解释的信息；不该称为一次无用evaluation。

但 [w09r01/exp-05](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/task/memory/cards/exp-05.yaml) 增加了540/1080/1620等保存点，后续没有单独评分，直接做soup而输给final，说明“多保存”与“完成有价值的模型选择”之间仍断开。

- **支持的作用：** WMA让一项成本有限、能够区分候选的观测变得可用；有明确实际动作。
- **尚未证明：** WMA能正确排序“继续训练、评checkpoint、做soup、停止”的期望效用。历史里C5可达±15pp，不自动意味着当前每个checkpoint值得评；控制组本来也会做checkpoint选择。
- **信息价值的正确判断：** 评估是否在事前能改变行动，改变哪个行动、概率多大、花费多少；结果为负仍可高价值。反之，无论结果如何都不会改变最终选择的重复读数，才更接近低价值。
- **判别实验：** 固定短名单与评估预算，比较完整L3排序与简单“最低成本未消除风险优先”的启发式；记录每次测试是否改变选择、避免何种不确定性、剩余预算，而不是用“正增益卡比例”代替效用。

### 5. 信息采集与测量设计：较强的现有功能，但并非WMA专属

[w07r02/exp-04](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r02_formal_r7/gsm8k_google_gemma-3-4b-pt_91008/task/memory/cards/exp-04.yaml) WMA要求full1319比较，scientist预先写shipping rule后执行：n150的-.66pp变成full1319的+3.03pp，候选得到保留。[w08r04/exp-05](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023/task/memory/cards/exp-05.yaml) 在n150被拒，[w08r04/exp-06](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023/task/memory/cards/exp-06.yaml)和 [w08r04/exp-07](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r04_formal_r8/gsm8k_google_gemma-3-4b-pt_91023/task/memory/cards/exp-07.yaml) 更大集合翻转排名并恢复最终赢家，属于重复出现的选模风险。

- **支持的作用：** 把不确定性变成具体测量计划；AB案例有明确建议→预写规则→实际执行链。
- **独特性限制：** scientist和control也会主动扩大样本，例如c10r03；“比较近分用更大共同N”可形成确定性决策支持。C high案例不能仅凭动作与建议一致就断定动作由WMA导致。
- **内容缺陷：** AB/D WMA出现“greedy两次必须字节一致，否则sampling开着”的不可靠诊断。需要分开有效参数读回、同items运行变异、扩items覆盖；不能把它们混称噪声。
- **判别实验：** 两个有reviewer的arm都拿到同样throughput、effective config、item IDs和配对计数，仅一边有WMA自由测量规划。比较是否能以同成本选到在预留集合更好的模型。用于选模的full test不是再独立一次的held-out证据。

### 6. 提出低成本反事实替代：具体two-way soup优于泛泛鼓励探索

[w10r02/exp-04](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/task/memory/cards/exp-04.yaml) 原方案四路soup；WMA额外提出两路subset，scientist确实执行：n500两路.710、四路.694、最好单个.704。建议增加了原方案没有的可比替代，改变暂定incumbent。后来两epoch SFT .738才成为提交，所以不能把最终PTB收益全归给soup。

- **支持的作用：** 把泛泛经验转成当前产物上的可运行、更便宜替代；比“这次可能有帮助”具体得多。
- **历史检索证据边界：** 可观察到C5/C6历史先验启发建议，但尚无证据表明私有历史不可替代。没有历史的同型号scientist也会提soup，control c10r01/c10r03已有此行为；必须区分“想起soup”与“挑对当前ingredients”。
- **失败也有信息：** 四路输、两路较好，说明原料质量/差异要被检验。D三路soup失败也否定特定组合，不能推成所有soup无效。每次只形成对该组合、该训练阶段的局部证据。
- **判别实验：** 普通reviewer、静态通用经验摘要、可检索完整历史三个arm，固定计算和建议长度。事前记录建议、引用的可迁移条件以及对未试替代的预测，再运行少量有判别力的替代比较；看历史是否提高替代质量而非只增加建议数量。

## 最能区分WMA价值的总体设计

先用已有固定proposal做成本很低的离线分解，不急着开新的训练波次：

1. scientist自审（等价新增tokens/时间）；
2. 独立同模型reviewer，仅当前proposal/代码/数据，无私有历史、无L0–L3模板；
3. 独立reviewer + 相同历史和探针；
4. 完整WMA的预测、建议和价值排序。

首先把输入信息与总预算对齐，否则“WMA赢”可能只意味着多读了一轮、多查了数据。随后只对离线显示有差异的组件做在线随机化：特别是“有无数值预测展示”“有无历史检索”“有无L3行动排序”。Scientist保持最终决定权，但记录建议前的计划、建议后的动作、为何改变、实际成本和预留评估集合结果。阻塞等待时间作为价格单独计入，避免与review内容作用混淆。

目前缺的关键证据：同预算自审/普通reviewer基线；未修改proposal上的稳定预测校准；counterfactual替代的可比结果；历史检索增量；区分预测措辞与行动建议的随机化；把负结果的信息价值纳入效用；独立最终选择验证。现有轨迹证明了局部机制效用，尚未证明世界模型模块本身的不可替代性。
