# 训练轮数、反馈深度与 WMA 的实际使用：R2 八个主体 cell 的核验

本文件只读既有结果，不监控队列、不提交实验。基础样本为 `doc/wma_iterations/evidence/2026-09-04-sft-efficiency/efficiency.json` 的 w10r01–04 / c10r01–04；候选例子另取 w09r01、w07r01。原始 scientist 轨迹逐字解压到本目录 `CELL-solve.txt`，行号可直接定位。下面区分“训练得到结果”“scientist用结果设计下一步”“WMA读结果改变建议”，避免把三者都叫学习。

## 1. 对“轮数太少”的最直接判断

**这个解释有一部分成立：绝大多数 w10 cell 只有首训后的两次新训练方案机会，无法充分研究很多级联训练改进。可是现有证据不支持把缺乏收益主要归因于‘WMA没有收到反馈’，也不支持‘多跑几次SFT会自动训练好WMA’。**

- w10四个cell已有 **30张实验卡、43次review**，其中 **13个不同训练方案、17张非训练卡**；实际完成fit为3/3/3/4。首训之后分别还有2/2/2/3个新训练方案机会，总计9个。
- c10四个cell同样完成 **13个fit**。其中两次完整fit是save失败后的重做：c10r01的exp05→exp06同方案，c10r04的exp05内部重做。它们增加算力消耗，却没有提供新的训练效果反馈。这不能当“control比WMA多学习两轮”。
- WMA不是连续带着上次隐藏状态聊天。每次review启动新Claude进程，模型和skill固定；但工作区里的先前卡片、结果、训练日志和verdict仍在，WMA实际反复读它们。
- 已查到的可归因作用包括**首训之前**的真实数据修复、**不增加fit**的候选补充、第二次训练前的checkpoint计划修正。因此“至少要很多次full SFT才有任何作用”已不符合这些轨迹。
- 仍未证明：这些局部作用可以抵消review/探测成本并带来同预算的稳定终局增益。增大horizon可能多给几次机会，但不会自动补上没有执行的评测、弱判别规则或不可靠的反馈尺。

## 2. 八个主体 cell 的实际反馈深度

| cell | cards | 完整fit（含save失败） | 训练card数 | 首训后新训练方案次数* | 非训练cards | WMA review cycles |
|---|---:|---:|---:|---:|---:|---:|
| w10r01 | 8 | 3 | 3 | 2 | 5 | 12 |
| w10r02 | 7 | 3 | 3 | 2 | 4 | 10 |
| w10r03 | 9 | 3 | 3 | 2 | 6 | 12 |
| w10r04 | 6 | 4 | 4 | 3 | 2 | 9 |
| c10r01 | 7 | 4 | 4 | 2 | 3 | 0 |
| c10r02 | 8 | 3 | 3 | 2 | 5 | 0 |
| c10r03 | 7 | 2 | 2 | 1 | 5 | 0 |
| c10r04 | 7 | 4 | 3 | 2 | 4 | 0 |

*去掉同方案save失败后的再跑；这是基于已记录proposal/trigger的计数，不保证每个方案都做了严格单变量对比。未完成的smoke和运行中断不算新完成fit。review次数从每cell `.wma/responses/*.json`逐个计数；它们包括13次relock额外review，所以43不是43个独立科学实验。w10r03首训exp02甚至有3次review，训练只发生1次。

各w10卡的review分布：

- w10r01：exp01/02/03/04各2次，exp05/06/07/08各1次。
- w10r02：exp01/02/03各2次，exp04/05/06/07各1次。
- w10r03：exp02 3次，exp04 2次，其余7张各1次。
- w10r04：exp01 3次，exp03 2次，其余4张各1次。

**所以正确的说法是：新训练结果的纵向深度短，但decision/review交互并不只有三次。**

## 3. WMA每次是不是“失忆”？它究竟看到什么？

代码证据：`awm/wma/backends.py:322–332`每次调用新的`subprocess.run`，`:383–385`使用`claude --print --no-session-persistence`，没有resume/continue；`review.py:89–98`每次为当前card建立新prompt；`review.py:73–85`提供当前card路径、相同scientist session目录和history路径。`skills/wma/SKILL.md:35–46`明确让它读同run的早期卡与其5–6节结果；`sidecar.py:75–93`每次将相同session/history交给review。

实物核验：w10的30份最终WMA transcript各有不同session_id，30份verdict全部`wma_skill=176f0a464986`。这些并非一个不断延长的WMA聊天session。另13份早期relock完整transcript被旧存储方式覆盖，因此不能把30个保留transcript说成43份都保留完整；每次重审确实发生，由43组request/response证明。

但**新进程不等于没有行为反馈**：以下都是最终verdict的实际evidence，不只是skill允许它读。

- w10r01/exp03 e7引用exp02：当前0.7067、37.5M tokens/6446s/72GB，以及同权重0-shot=0.740、10-shot=0.680或0.540的差别。
- w10r02/exp06 e5/e6/e7引用exp02、exp03、exp04：两个单epoch训练的中间/终点读数、遗失checkpoint的retention故障、n500的0.688/0.704/0.710候选梯度。
- w10r04/exp04 e5/e6引用exp02吞吐和exp03的n1319=0.6990、dev250退化15 fixes/19 regressions；exp06 e4/e5继续引用exp04实际吞吐、exp05同尺941/1319与44/44 exclusive wins。

这些文件均位于各cell原始`task/memory/cards/exp-NN.verdict.json`，原始result目录在efficiency.json对应cell中。它们说明：**已有文件介导的当场适应；没有模型权重或skill在cell内部被训练更新。** 被SFT更新的是Gemma checkpoint；出WMA verdict的是固定Opus，二者不是同一个被训练对象。

## 4. 具体链条：已有深度里，反馈怎样改变下一步？

### A. w10r02：三个fit都从base开始，仍然明显在用前两次结果更新研究方向

第一个fit exp02修终止格式，官方n150=0.6733，中间ck650=0.680。scientist据此诊断格式已修、剩余错误46/49为推理，转向增加29750条自己的正确采样解，exp03从base重训（原始卡`exp03.situation.trigger`和hypothesis；轨迹7477–7510含WMA对上一次结果的引用）。

exp03最终和中途读数0.660/0.6867，又做两个soups和一个decode试验后，scientist在exp06明确把“epochs未变过”识别为剩下的轴。**权重从base重启，但方案已经是前几步失败反馈的产物**。WMA不是无视这些反馈：它的exp06 evidence逐项引用前两个fit的末段平/降、当前n500候选表，预条件按实测22.237 samples/s估算双epoch耗时（[13:54:44，轨迹10384–10394](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10384)）。

scientist13:54:52实际launch2epochs（[10408](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10408)），结果n500=0.738，对同数据单epochck650=0.688（[10872–10885](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10872)）。这个+5pp是该cell内训练对比，不是WMA净贡献：双epoch方向在本次WMA返回前已由scientist提出，WMA贡献是预算/风险/已有候选方面的评审。

这例**支持继续探索的可能性**：最后明确留下“在更大现有corpus上跑2–3epochs”未完成。但它也说明当前三fit并非“三次独立盲抽”，而且不能把尚未执行的后续方向当已证明的horizon收益。

### B. w10r01：首训结果改变第二轮方案，WMA再改变可保存/可评估的点

首训的同权重probe显示0-shot与10-shot差异，scientist因此提出exp03：从exp02/ck1800继续、用8000条带前缀样本修prefix条件。WMA读取这个反馈后指出，8000 rows/bs2×accum16仅250steps，原`save_steps125`保留的选择点过少。

scientist实际把save interval改50并重锁，同时增加fs0诊断来防止只改善fs10而伤害0-shot（[轨迹10328](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:10328)）。这是“前轮轨迹特征→新方案→WMA具体修改”的完整链；不是必须先经历5–10个训练才存在。

第三个fit之前，WMA又发现新的RFT文件混入32/300本地dev问题，scientist声明查全部训练文件并发现总150/300污染，换用剩余150个clean-dev、修改判断阈值后重锁（[13173](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:13173)、[13293](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:13293)）。这里没有证明分数因WMA上涨，但反馈质量得到实际修正；如果继续原受污染的尺，多跑训练只会多收不可靠反馈。

### C. w10r04：第四个fit已有，不是“所有run都停在第三次”

exp03的RFT在full test只有约+1.14pp、dev250反而退化，scientist因此在exp04换成base重训、扩大不同问题覆盖；WMA实际引用这两个前轮结果。

exp04结束后，exp05 soup在full test和incumbent恰同941/1319，scientist又提出第四个fit exp06：57k新chains、低lr继续。WMAexp06引用exp04的87.9M tokens/7075s和exp05的44/44 exclusive wins估计成本/效果，而不是看不到历史。实际结果939/1319，dev250下降3.6pp；虽n150读数升至0.740，预先写好的full-test paired规则使其被reject（[11366–11383](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:11366)）。

这例是当前样本内对“再给一次训练机会自然见收益”的反例，不能推广成后续训练永远无用；它表明额外深度的收益取决于新方向和测量，而不仅是次数。

## 5. WMA的作用也出现在第一个fit之前或完全不增加fit时

### 首训前：w10r02/exp02，当前fit数=0

scientist已经想做completion-only SFT，先前的smoke却只是把自己的renderer与Jinja比，不是与grader产生的真实few-shot消息比。07:58:20 WMA要求比实际10-shot（[4983](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:4983)）；07:59:06工具结果明确“renderer比较True、per-shot字符串比较False”（[5134–5135](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:5134)）。

07:59:24 scientist修复把calculator annotations误删的`gold_fewshot`，07:59:30真正重建训练数据，随后检查per-shot identical=True并重锁（[5151–5184](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:5151)、[5224–5249](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:5224)）。修复对象是训练侧few-shot示例；这不是读取test标签的那个w09r03案例。

这是第一轮前的真实工作改变。但没有未修版本的训练对照，不能把首训全部大幅涨分归给这个修复。

### 第二fit之后、不增加fit：w10r02/exp04的具体两成分soup

前两个fit已有ck650/1293/1350/1757；scientist原计划四成分。WMA新增ck650+ck1350子集，13:09:29收到、13:10:09构建、13:24:32测量，n500=0.710，比原四成分0.694、已测最佳ingredient0.704高（[9233](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:9233)、[9300](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:9300)、[9639](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:9639)）。

局部候选发生变化，fit计数保持2；后来exp06替换它，所以没有证明此建议提高最终PTB。它仍说明WMA的机会数量不能用full-SFT数单独表示。

## 6. 两个候选例子：有反馈、有动作，并不自动形成实用收益

### D w09r01，exp04→exp05

exp04中，WMA使保存计划从2200/limit2改为1490/limit4，scientist实际测ck2980，得到0.6933<final0.7533（[21:11:17–21:13:35，轨迹7458–7510](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt:7458)）。

下一张exp05，WMA明确拿这个结果调整建议：原1080保存点不足，应540/1080/1620都留，以观察更晚位置。scientist22:18:17确实修改并重锁（[8037–8070](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt:8037)）。这是反馈被WMA吸收的直接证据。

但它另提议的checkpoint1080一落盘就eval/可能早停，没有执行：23:09:13已见1080，下一动作却`sleep1900`等训练结束（[8299–8323](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt:8299)）。因此这是“已读反馈、保存采纳、评测未闭环”；延长时间不能保证把未执行的规则变成执行。

### A+B w07r01，exp05→exp06

exp05两成分soup比ingredients好，scientist因此在exp06**本来就已计划**独立训练第三个ingredient C；WMA没有创造这个训练方向。WMA新增两个具体筛选条件：C单独dev300至少约0.73；与incumbent的共同失败不要超过约47（[10484–10486](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r01_formal_r7/gsm8k_google_gemma-3-4b-pt_91007/solve_parsed.txt:10484)）。

实际C=0.740、shared failures42，scientist说“两gate通过”再构建三成分（[10770–10785](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r01_formal_r7/gsm8k_google_gemma-3-4b-pt_91007/solve_parsed.txt:10770)）。最终n500只0.740 vs0.736，McNemar p≈0.90，仍选择三成分（[10921](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r01_formal_r7/gsm8k_google_gemma-3-4b-pt_91007/solve_parsed.txt:10921)、[11050–11068](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r01_formal_r7/gsm8k_google_gemma-3-4b-pt_91007/solve_parsed.txt:11050)）。

它已经到第三个fit、已经使用前轮误差结构、已经采纳WMA规则；缺口在于这些规则没有在本次产生可分辨的收益，不是缺少读结果的机会。

## 7. 对用户假设的有限支持与界限

可以支持：当前首训后2–3次训练方案机会偏少，某些“先改善配方、再在不同训练轮稳定复现”的作用可能还没充分表现；w10r02最后确实产生了未执行的新训练问题。数据不足以估计5次或10次时WMA的净效果。

不能据此推出：

1. **现在没有反馈学习。** 既有卡结果已明确进入下一方案和WMA证据，from-base重训仍是适应性决策。
2. **多fit会直接更新WMA能力。** 该cell里固定Opus+skill不训练，只能经文件读到新证据后重新推理；跨cell的skill改进是外层evolve的另一条链。
3. **horizon是目前唯一缺口。** 首训前和非训练card已有实际作用；后面的另一些建议已经被读到但未执行，或执行了但判别条件弱、噪声大。增加相同类型review或训练次数，并不自动修复这些已观察到的问题。

因此最符合原轨迹的表述是：**训练反馈的纵向深度确实短，但WMA并非没有机会行动；当前瓶颈同时包括反馈尺的可信度、建议到实际评测的闭环，以及是否提出了比scientist现有计划更有价值的修改。延长horizon是尚未验证的可能增益来源，不能用它解释掉现有的具体失效。**
