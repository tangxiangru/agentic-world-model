# WMA evolve 迭代史与实验设计审计

审计时间：2026-09-04；只读。在线当前数由主代理冻结的 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/summary.txt`（2026-09-04 05:57:37 UTC）提供；本文件独立核对设计、历史记录、候选 git diff 与实际评分代码。以下历史数字不替代当前 validator 结果，也不涉及其他队列。

## 1. 最重要的结论

WMA 已完成“能接入 scientist 工作流”和“通过阻塞 lock 在决策前送达”的工程迭代，但尚未建立稳定的最终分数或实际 GPU 利用率收益；尚无候选通过当前晋升门。当前 A、A+B、C 已各完成 4 cells，D 完成 3/4；它们的原始 access guard 分别失败于 14/29、11/23、10/32、9/20 verdicts。E/F 运行中，G/H 尚未开始。不能把后两轮“已 launch”写成“已有实验结果”。

最新同协议均值：w10 v0.2 71.342%（n=4，SD=1.282 pp），c10 74.299%（n=4，SD=5.920 pp）；A 69.086%，A+B 71.778%，C 71.797%，D 73.288%（n=3）。这说明现在的证据不支持“WMA 已有提升”，也不支持断言“WMA 确认有害”。D 尤其是未完 cohort；A+B 和 C 的 +0.44/+0.45 pp 对 w10 差距远小于当前方差，且 guard 已失败。

同样不能把数百张 card 或 verdict 当独立实验：正式最终质量的实验单位是一个 scientist cell。当前总量 64 validator-clean cells、472 cards、198 terminal verdicts，分别是三种不同分母；后两种只适合解释机制，不能把最终分数效应的 n 膨胀到 472 或 198。

## 2. 离线实际做了多少

| 阶段 | 完成内容 | 应如何计数 |
|---|---|---|
| round-00 heuristic | 抽样 300 与 train 全量 1580 | 1880 次离线启发式回放；300 是全量的子集，不是 1880 个独立样本 |
| round-00s heuristic | 强 scientist train 31 runs / 313 cards | 313 次回放，与全量集合重叠；新的可比基线，不是 GPU cells |
| round-01 Opus 接线 | 同一张卡运行 3 attempts | 1 schema rejected，1 合法但原 fence 误报，1 合法无越界；不是 3 个科学实验或完整 pass |
| round-01 Fable 计价 | 20 attempts，19 原始合法 verdict | 不同模型/effort，与后面的 Opus 版本不能当单变量比较 |
| round-01s v0.2 Opus 核验 | 24/24 合法，3 access 排除，21 进入账本 | 只验证样式、探针和少量预测；没有完整 313-card pass |

因此历史记录支持：**2193 次 heuristic 回放计算 + 47 次模型调用 attempts（其中接线重复、拒收等另计）**。不能说“2193 个离线独立实验”，也不能说已跑完两个模型版 baseline passes。

v0.2 小样本核验出现 L0 recall 1.0、L1 hit .882、L2 coverage .929（n=14）、L2 width/noise 4.99、L3 hit .75；这些是形状检查。v0.1 与 v0.2 同时换了 skill、模型/effort、样本集和部分评分语义，不能归因“v0.2 导致 L0/L1 改善”。没有 held-out promotion readout。离线上线门是用户当时明确豁免，原因是回放缺工作区，不能测试在线探针；这不是一次已通过的离线晋升。

来源：[round-00:10–46](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-00.md:10)、[round-01:26–70](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-01.md:26)、[round-01:75–136](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-01.md:75)、[上线豁免:296–308](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-02-wma-gsm8k-gemma4b-iteration-basis.md:296)。

## 3. 在线迭代链：每步究竟回答了什么

### 第一轮：v0.2 异步附加 vs 无 WMA

8+8 core 最终均值 75.654% vs 75.227%，差 +0.426 pp；按 replicate 编号配对 SE=1.36 pp，5/8 同向。没有可靠分数提升。收到的 22 条 verdict 中 21 条晚于 launch，59 cards 只请求 33 次；因此这轮主要检验了一个大多数建议不能影响原始 launch 的异步产品形态，不能充分检验“按建议改变实验后的 WMA”。

修复两个 runtime 缺陷后的 w04/c04 retained 4+4 也无明确收益（−2.03 pp，SE 4.35 pp）。它仍为异步，1/19 在 launch 前。历史文档的“uptake 改善而分数没变，因此 delivery/uptake 不是 binding constraint”措辞强于数据：两个小样本、不同 cohort 的无显著结果，不能排除 timing/uptake 是约束；较可靠的结论只是“修复交付问题本身尚未显示最终收益”。

来源：[R1 终局](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-01-online.md:164)、[w04 敏感性](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-01-online.md:303)、[有争议的 binding 推论](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-01-online.md:341)。

### 第二轮：blocking lock + A/A+B/C/D

旧候选 16 jobs 在 PENDING 时撤回，换成相同候选 skill + 公共 `ae46724` blocking runtime，同波新 w10/c10。因此旧 receipts 是撤回的计划/attempt，不是负结果；后续不能拿这些 skill 候选直接对旧异步 baseline。扩展 w10/c10 各 4 cells 是预先登记的同 treatment 扩展，用于达到每臂 8 clean cells。

已证明的局部机制包括：w10 全部 30 terminal reviewed cards 的 lock 为 delivered；手读有 launch 前修改/重审、checkpoint 保存和测量的实际动作。它不只改变“读到建议”的表面指标。与此同时 43 review cycles 对 30 cards、约 6 min 每次的全量重审显著占用工作流，且 43 L3 answers 全部 yes；这暴露的核心是“建议能进入行动，但还未形成可靠的资源分配判断”。

来源：[R2 替换与设计](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-02-online.md:96)、[扩展](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-02-online.md:167)、[已校正的轨迹诊断](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-02-online.md:222)。

### 第三轮：E/F

两者各 4 cells，由 R1 的 all-yes 与低 action rate 独立预注册，不是读 A–D 结果后选出的获胜者。E 为 C3/C4 的 L3 yes 加“效应可分辨、且无更便宜 C2/C5 判别器”；F 把首条 precondition 改为唯一最高价值的 pass/fail 动作。必须等自身完整结果，不得把运行中 trace 当最终效应。E 已事先登记关键伤害：若让首次 SFT 先反复做 decode probes，可能拖延最主要的得分杠杆。

来源：[R3:16–58](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-03-round-03-online.md:16)。

### 第四轮：G/H；I/J 未 launch

G 是输入/副作用边界约束；H 只撤掉 C6 的“时间短就默认 soup”正向先验。它们基于完成的 w10r01–04/c10r01–03 轨迹，与 A–F 未完成结果独立；G/H public 和所有非 skill private bytes 特别核验为与 `ae46724` 一致。I（repeat vs 扩样本）和 J（probe stopping）被审查后延后：v0.2 已存在很多对应好行为，缺少增量机制；J 的 latency 目标也未有可靠计时证据。不能把 I/J 计为已开展的候选实验。

来源：[R4 候选冻结](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-03-round-04-online.md:24)、[R4 主 spec 与审查修正](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md:47)。

## 4. 八个候选的科学问题与失败边界

| 候选 / skill hash | 独立机制 | 当前读法 / 关键风险 |
|---|---|---|
| A / b6cc80891de4 | first format-restoration 的 L2 不沿用普通 C3/C4 小效应先验 | 最新 n=4、均值 69.09%；单靠 overall L2 不能检验 first-SFT 主指标。需手读 floor 诊断是否成立、anchor 是绝对值还是 delta；不预断该歧义已导致错误 |
| A+B / 46f66c240380 | 在 A 基础上，>3× noise 的宽度必须有依据 | **B 的增量对照是 A，不是 v0.2**。相对 baseline 两处干预；没有 B-only 无法识别 A×B 交互。提高 coverage 或更窄必须在同类同分母上比较 |
| C / 44d687f73c70 | L0/L1 no 要证明失败路径；未证实 confidence≤.5 | n=4、71.80%，cell SD=6.14 pp。所有已评分 L0/L1 成功不能证明提高失败召回：没有失败机会时 recall 不可识别。且当前 hit scorer 不使用 confidence，所以“仍 no 但降 confidence”不会机械改善 false-no |
| D / 6e0d02c5afad | 只保留 final checkpoint 的 C3/C4 必须 defer，直到有 save/eval plan | n=3，73.29% 只是候选线索；需要看到实际保留且同尺测量，而非 card 多一行 save_steps。旧 w10 出现 save_total_limit 删除计划 checkpoint，是“部分采纳但执行缺陷” |
| E / 20c8837ab7a4 | L3 yes 要跨过效应/廉价判别门 | 检验是否减少真正无收益工作；不能用制造更多 defer 作为成功。首次 SFT latency 是已登记伤害通道 |
| F / 959fbdd5bbb2 | 第一条 suggestion 是能改变 verdict 的单个 pass/fail 行动 | 检验实际决策改变，不是 suggestion 格式。若动作相同、更多重审，则 salience 无净收益 |
| G / e4402ffa6bca | 探针仅用明确 in-scope 输入、scratch 副作用 | 当前 A–D 全都 fence guard 失败，G 有直接的实验有效性价值；零 flags 还须手审，并保留 review coverage / blocker detection，不能靠少产 verdict 过门 |
| H / a536a0af24d7 | 移除“时间短默认 soup”的先验 | 五张 baseline eligible soup cards 都出现该正向先验（四 evidence、一 L3）；机制明确。需 >=3 eligible opportunities；只有删掉措辞而没改变 merge/ingredient/incumbent 决策不算实用收益 |

候选差异已直接核对 `git diff ae46724 CANDIDATE -- skills/wma/`，不仅依据标题。A 与 A+B 关系见 [R2:28–35](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-02-online.md:28)；C/D 证伪见 [R2:78–94](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-02-round-02-online.md:78)；H 机会定义见 [R4 spec:90–115](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-03-wma-round04-probe-selection.md:90)。

## 5. 当前测量不能支持的推论

1. **“gpu_h_saved=0”不能直接等同实际一小时都没省；非零也不能当真实节约。** 代码只对 L3 no/defer 且最终 reject/abandon 的卡加 wall_h；不看 scientist 是否采纳、是否在 launch 前、资源是否真的回收。wrongly_killed 同理。它本质是标签对齐的潜在节约/误拒 proxy。真实资源效率应回到 command/step 时间轴和最终模型祖先链。见 [ledger.py:152–155](/home/robtang_google_com/gangda_workspace/agentic-world-model/awm/wma/ledger.py:152)。
2. **L3 hit 高不等于判断有价值。** scorer 将 adopt 当 worth，reject/abandon 当 not worth；但有信息价值的失败实验可能值得试，而控制 scientist 的 adopt 标签又受 treatment 影响。更强的终点是 final score、budget、成败决策的可审计作用。见 [schema.py:308](/home/robtang_google_com/gangda_workspace/agentic-world-model/awm/wma/schema.py:308)。
3. **confidence 校准尚未被现有 hit 率测量。** L0/L1/L3 score 只看 categorical answer，L2 只看区间包含；0.51 与 0.99 的概率并未受不同惩罚。若改 confidence，要单独登记 Brier/校准曲线等测量，不把现有高 hit 叫已校准。见 [schema.py:283–315](/home/robtang_google_com/gangda_workspace/agentic-world-model/awm/wma/schema.py:283)。
4. **L2 的现有 coverage/width 分母不同。** coverage 用 clean + scorable，width/noise 用所有 verdict（包括 fenced 或不可评分）。主报告应同时保留官方 ledger 和同 clean/scorable 子集上的敏感性统计，避免把缺失/排除造成的组成变化误作变窄。见 [ledger.py:143–166](/home/robtang_google_com/gangda_workspace/agentic-world-model/awm/wma/ledger.py:143)。
5. **noise floor 是经验分段尺，不是精确统计置信区间。** 代码只依 n 查表（并 max 1/n），没有题目配对相关、正确率、decode 和训练不确定性；不能机械要求“预测未知训练效果的区间必须只有若干倍测量噪声”。尤其 first format restoration 的真实大跳跃与 later-SFT 小改动不可混在一条 prior。见 [schema.py:90](/home/robtang_google_com/gangda_workspace/agentic-world-model/awm/wma/schema.py:90)、[manual:89](/home/robtang_google_com/gangda_workspace/agentic-world-model/skills/wma/change_types.md:89)。
6. **43 次 request lifecycle 总时长不等于 GPU idle。** 可以并行准备；还可能与前一个训练重叠。要计时段交集：GPU 已分配、无有用 GPU 作业、scientist 因该 lock 等待。当前 80.11/58.76/69.58/59.66 min 只可标 lifecycle proxy，不可写成已量得 >1.5 h idle。
7. **initial no 后修复成功，不是原预测 false-no。** 必须把 proposal version、脚本哈希和 verdict 对齐；30 个 final 文件会遗漏重审前的 13 个 decisions，早期保存机制也遗漏其完整 cost。任何首次 verdict 对修订后 outcome 的“预测准确率”无效。
8. **同 replicate 编号不是共同随机数实验。** 当前 manifest 指定编号，但不证明 seed、生成路径、起始 recipe 或同步时窗一一相同。配对表可保留为设计记录，不能假设自动减少方差；建议同时给 Welch/独立 cell 或分层随机化敏感性估计。
9. **相关性不是 recipe 因果排序。** first-SFT score、训练小时、训练行数和 final score 的关联为后续假设提供线索；first-SFT feature 甚至混入 sampled vs greedy，不能说已证明“多训练导致高分”。见 [uptake-levers:53–59](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-03-w10/uptake-levers.md:53)。

## 6. 样本量与设计上的下一步

默认四次重复适合探查大机制，例如近半 verdict 越界能否降到零；不适合从 +0.4 pp 找赢家。按独立 arms、双侧 5%、80% power 的普通正态近似，`n_each ≈ 2*(1.96+0.84)^2*SD^2/delta^2`：

- SD=3–5 pp，识别 +2 pp 需要约 36–98 cells/arm；+4 pp 约 9–25。
- +0.4 pp 则约 882–2450 cells/arm。这里只是效应量的数量级说明，不是建议现在提交这些规模；它假定可比独立重复、固定效应、无多重比较，实际设计还需重算。
- “每臂八个 clean”是工作流证据门，不是对任意微小效应的统计 power 保证。
- A–H 多候选筛选存在 winner's curse；先按机制选一个，再对新同 setting blocks 做确认，最后才一次 held-out。不要按最高单次分数选；不得为追求显著不断补同一 arm。
- 分析层级：cell 为主要统计单位；cards 嵌套在 cell，用 cell-cluster interval 或每-cell rates，不把 20–30 cards 当独立样本。
- 当机制机会稀少时，先报告“eligible opportunities / cells”，再决定是否换一个预登记 setting；不能把 0 opportunity 当 0 failure。

## 7. 改进优先级：以净决策价值为目标

**P0：完成本轮可识别结果，再决定晋升。** 先读完当前 A–F 和 w10/c10 扩展；G/H 按既定机制读取。A–D 当前 access guard 失败，保持 baseline，不准事后清 flag 或把 G 无声叠入别的 arm。G 若有效，按单独新波确认其 scope 收益与 blocker 保留，然后才能测试 G+D 等明确组合；原 D 结果仅是筛选依据。

**P1：重审只审改动，保留严格 launch gate。** 证据比泛泛削 probe 更直接：w10 43 cycles/30 cards，常因补一句 precondition 或调整保存计划再次等待完整约 6 min。建议作为独立 protocol/harness 候选：不可变记录 request ID / proposal hash / script hash；对确定已满足的上一轮 precondition 和未变证据复用，把新信息交给 WMA 作增量 review，仍要求 review 返回后才能 launch。主指标是每 distinct card lifecycle 与 p95、cycles/card；实用指标是 time-to-first-SFT、实际训练 GPU 时间与 final score。证伪：漏掉改变配置后引入的 blocker、过期 verdict 通过、latency 不降或实用工作不增。不能把减少完整 review 直接视作安全，应以 archive 确认对齐，另设 runtime baseline，不与 skill 改动同批合并。

**P2：从“建议被读到”转为“正确改变关键决策”。** 根据 D/E 结果只选一条：若 D 真增加保留/可比测量却未改善选择，进一步解决 `save_total_limit` 导致建议落空的确定性机制，归 protocol；若 E 会拖延首次 SFT，改成只在廉价 action 能回答当前同一决策时 defer，并区分未恢复格式的首训与已稳定 incumbent 后的边际训练。主指标应是避免的已证实劣方案与保留的有用方案，附首训 latency、训练量、final 质量；不能把 no/defer 比例上升当成功。

**P3：把预测校准与资源收益拆成两条可检验链。** A/B 主要改 L2，应先在训练侧、严格 pre-launch 冻结输入上检验相同 card 的预测，按 first-format-SFT、later-SFT、decode、soup 分层。L2 采用相同置信水平区间和包含宽度/漏出惩罚的评分，并与固定 prior 对比；这些是新测量，需独立 spec，不改变已冻结轮的主分数。只有能解释某次下游选择发生变化时才让它承担昂贵在线收益验证。修掉离线 history 父目录暴露，并确保不是用当前 card outcome 重建一个被污染的 session。

**P4：对 soup/selection 保持机会驱动。** H 先检验是否撤掉 clock-only prior 后改变选择；若只是措辞变化，停止该方向。若有动作，比较同尺、同 decode 的 incumbent/ingredients/soup 成对结果与时间成本；记录有益 soups，不能先验禁止弱 ingredient 或把弱者质量当 soup 上界。重复评估与扩大题集分别处理运行噪声和题集覆盖，见到真实错误决策后再启动 I，避免重复已有 v0.2 好行为。

所有下一步都是建议，本次审计未修改 repo、未 launch/cancel/submit，也未调用其他队列。

## 8. 关键补充：已证实的库调用测试标签暴露；G 不能单独修复

本节在前文之后完成，**覆盖前文 P0 中可能被理解为“G 过零 flags 就可恢复科学有效性”的读法**。主代理发现后，我独立检查了 w09r03/exp-01 的原始记录、G 的冻结 skill 与实际 runtime。

### 已独立核验的证据链

- cell：D `w09r03`，job `91026`；receipt-backed PTB result 位于
  `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r03_formal_r9/gsm8k_google_gemma-3-4b-pt_91026`。
- `wma_private/exp-01.transcript.jsonl:85`：WMA 在 scratch 中执行 `import inspect_evals.gsm8k as g; task = g.gsm8k(); ds = task.dataset; s = ds[0]`，打印 `s.input` 前缀和 `s.target`。`:88` 对应工具结果确实返回 test dataset、题面和 target；不是仅仅提出一个未执行的 probe。
- 同 trace `:90` 再次读取 `ds[0].input` 用于模板渲染，并通过 `inspect.getsource` 获取库源码。该源码访问也未以显式绝对路径出现。
- `task/memory/cards/exp-01.verdict.json:128–131` 的 e12 把“1319 test rows、first target …”写入给 scientist 的正式 evidence。`:212` 的 access 字段仍 `outside: []`，没有 leak_suspected 标志。
- 此处不复印实际题目或标签；审计不需要扩大其传播。库读取的是 GSM8K official evaluation 的 test 样本，不能与 WMA replay 的 Gemma run held-out、或 AIME promotion task 混为一谈。

因此，这是**实际测试标签已暴露且 scanner 漏报**，强于“普通 ambient filesystem 越界”。`validator issues=[]`、`judge_flags=[]` 只能表述原自动验证器通过，不能再直接称该 cell 科学上无泄漏。它的最终分数可保留为原始观测，但应标记为受污染审计对象，不能作为 clean treatment benefit。仅移除这一条 verdict 也不足以恢复该 cell 的后续独立性，因为信息已经进入 scientist 的上下文。不能从本例推断其他所有 cells 均污染，也不能从其他 cells 零 flags 推断均未污染。

### G 的文本与边界能力

独立读取 `git show 125a434e6d73d067427911332663060fe2dce558:skills/wma/SKILL.md`：

- `:120–132` 新规则要求 intentional inputs 留在 session、skill/history/scratch，禁止 inspect installed package source、为探索安装目录而 import、从 outside 搬数据等。
- `:154` 的“any held-out data”禁令已存在，是全局语义禁止。
- 所以此处行为违背 G 的总体输入与 held-out 契约；`:90` 的 `inspect.getsource` 还直接违反 G 新加的 package-source 禁令。
- 但 `:85` 通过合法 import 返回 `task.dataset` 不是“探索安装目录”，G 没有新增 API 调用层、数据集 split 层或数据返回层的强制限制。可以期待提示词减少这种行为，**不能保证防止它**。
- 已确认 `git diff ae46724 125a434 -- awm/wma awm/exp_protocol` 为空。G 不包含任何 runtime isolation 修复。

现行 `backends.py:117–141,204–240` 仅扫描 tool-call 文本的 path 字段与 Bash 命令里形似 `/...` 或 `../...` 的 token，且路径要在 scanner 所在环境实际存在。它不记录真正的文件 open、网络读取、模块 loader，也不扫描工具返回数据的 provenance；`_inside` 还是 lexical containment。上述命令显式路径只有被允许的 `/tmp/...` 与 `/session/...`，故“零 outside”是实现的自然漏报，而不是证据干净。

`backends.py:330` 直接启动 backend；`:383–385` 以 `--dangerously-skip-permissions` 运行 Claude。`sidecar.py:90–92` 的 allowed_roots 传递给事后 scanner，而非 syscall enforcement。

`third_party/PostTrainBench/src/run_task.sh:303–330` 用同一个 `POST_TRAIN_BENCH_CONTAINER_NAME.sif` 启动 WMA，绑定整个 scientist session，使用 writable-tmpfs。该启动段没有配置限制 probe 网络的边界。`containers/opus_5.def:70–76` 又明确把 inspect_evals 安装进通用 image。独立 container 隔离了 scientist 与 WMA 的私有代码/轨迹，**没有把评测 dataset loader 和标签能力从 WMA 进程中移走**。

### 最小可靠修复应是一个独立 harness 改动

不把加一句禁令、检测 `inspect_evals` 字符串或 Python import hook 视为完成修复。最小闭环要同时切断本地副本和重新下载两条数据路径，并在返回 scientist 之前验证：

1. **给 WMA 的 probe 进程一个专用数据边界。** 它的运行环境不包含评测数据/缓存，不挂载 benchmark 数据目录；不提供可加载 labels 的 inspect_evals task 对象。显式允许的输入是带 split/provenance 的 train 侧材料、当前 pre-launch 代码/config、允许的历史摘要以及独立 scratch。静态审查所需 evaluator 代码/模板可由可信 harness 提供无题目/标签的源码或 metadata；不能为了看默认参数顺带初始化 dataset。
2. **在 OS/tool 执行层禁止 probe 网络和越界读取。** WMA LLM 控制器仍可访问模型 API，但其任意 Bash/Python probes 应在单独的受限子进程/namespace 中运行；不得直接借模型客户端所在的网络能力下载数据。仅清 HF cache 或设 offline env var 不是可靠边界。文件 allowlist 必须按实际解析的目标执行，不能只按命令字符串或 lexical path 判断。
3. **评测能力留给可信 evaluator。** 如确需 dataset availability/模板兼容性检查，通过受信任的 preflight/broker 返回固定 schema 的 pass/fail、版本、split 标识、count 或已获准的 aggregate；不返回 dataset 对象、任意 sample、target 或包含它们的异常文本。普通允许的 train-data、模板渲染和已知 preflight 仍须可用。
4. **尝试越界与发现暴露时 fail closed。** 记录不可变 request/proposal hash、denied operation、输入 provenance，阻止该 verdict 作为 clean 结果送达。既有 `access` 原记录保持不动，新增独立审计状态/事件，而不是修改旧输出让它看起来从未污染。新 boundary/runtime 单独版本化、重新配 baseline，对本轮结果保留原始统计与明确的 contamination sensitivity。

这四点是修复库介导数据读取的最低功能要求；具体选 Apptainer 子沙箱、受限执行 broker 或其他已支持机制，是实现阶段的选择。本次没有实现、更改容器、重写旧 flags 或调度操作。

### 必要回归案例（建议用合成数据与秘密 sentinel，避免再次读取真实 test）

| Case | 模拟输入/动作 | 必须验证的结果 |
|---|---|---|
| 原案例 | mock `inspect_evals.gsm8k().dataset[0].target`，命令只有允许路径 | 不能拿到 hidden sentinel；有结构化拒绝/隔离证据；verdict 不向 scientist 泄露 |
| 间接 API | alias import、动态 import、任意辅助函数返回 target、`inspect.getsource` 暴露内嵌 fixture | 同样不可达，不依赖字符串匹配 |
| 本地 cache | 测试集 parquet/JSON/cache 经 loader 读，含 session/scratch symlink 指向禁读目标 | 实际目标拒绝；不因 lexical path 或复制到 scratch 获准 |
| 远端重建 | 从合成测试 HTTP endpoint 拉取 test 数据并写进 scratch | probe 网络被拒绝；模型 API 的正常请求仍可用 |
| 副作用加载 | import 当前允许脚本时，它自动加载 benchmark test 或包含 target 的异常 | 不泄露 sample/target；导入失败以无敏感 payload 的状态返回 |
| 可信 preflight | 需要知道 evaluator 可用、模板/停止符是否一致 | 通过 synthetic/train fixture 返回限定 metadata；不初始化真实 held-out dataset |
| 允许路径 | 读带 provenance 的 train 数据、config、此前允许的历史摘要，写自己 scratch，返回正常 verdict | 正常工作，避免边界修复靠禁掉所有 probes 获得表面通过 |
| 端到端回传 | 让模拟 tool 输出包含 hidden sentinel、让 agent 尝试把它写入 evidence / errors | broker/交付边界不回传敏感内容；response 标为不合格，保留审计记录 |

补充优先级：**先修复并验证数据边界，再讨论 G 是否达成行为约束或某候选晋升**。G/H 已冻结实验仍可作为原 runtime 下的探索记录，其零 lexical flags 将不足以证明无标签暴露。
