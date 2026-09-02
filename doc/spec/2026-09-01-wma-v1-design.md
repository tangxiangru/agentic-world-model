# WMA v1 设计:评估器、两条路径、离线回放与在线优化

**日期**:2026-09-01 **状态**:待 review **配套**:`doc/reference/exp_protocol_and_wma_policy.md`(定义与分界)、`doc/spec/2026-09-01-exp-protocol-card-v2.md`(card 格式,WMA 的输入)、`skills/exp_protocol_meta/`(迭代循环的同构参照)

---

## 一、定位与原则

**WMA = World-Model Agent,世界模型。** RL 里 world model 预测"如果做 X 会发生什么",policy 决定做什么。按这个词义:**WMA 是世界模型,scientist 是策略。** WMA 估计,不决定。

由此定下的原则:

1. **评估器为本体。** 输入是 scientist 的一个(或一组)待评估方案,输出是裁决:跑不跑得起来、会不会产出有效候选、相对现任的方向与量级、现在值不值得花这个预算。
2. **建议只是裁决的附属物,且必须基于输入的方案。** 只有两类:跑之前先验证什么(前置条件),以及这个方案的更便宜 / 更低风险的派生版本。**不提出新方向,不主动插话,不替 scientist 决定。**"提出新方向"是 v2 的事,解锁条件是账本证明它的裁决已经校准。
3. **context 不剥离。** scientist 能看到的它都能看到:工作区、脚本、数据、日志、评估输出、之前的卡片、原始历史轨迹。card 是它检索历史的索引,不是唯一输入。
4. **可以动手,但有三条绳子。** 它可以为出裁决而跑探测(单元测试、静态检查、数据对齐、小批量试跑),前提是:**只读**——探测跑在 scratch 副本里,不改 scientist 的工作区、不替它跑正式实验;**有预算**——每次裁决的 CPU / GPU / 墙钟上限固定;**由信息价值驱动**——一个探测只在"它的结果会改变裁决"时才跑。
5. **与实验规程的分界**(定义文档第二节):机械可判的检查归 preflight,永远不由 WMA 做;WMA 的探测是那些需要判断才知道该不该做的。
6. **每次裁决必须可对账。** 结构化写下,事后与实际结果对照——这就是账本,也是它学习的唯一反馈。

## 二、接口

### 输入

| 项 | 来源 | 说明 |
|---|---|---|
| 待评估方案 | card 第 0–4 节 | 用 exp_protocol 的格式,不另定义;一组候选 = 一组卡 |
| 当前处境 | session 目录全读 | 脚本、数据、日志、eval 输出、`memory/index.md`、之前的卡与 verdict |
| 历史 | 语料(卡片索引 + 原始轨迹,只读);经验库(P1 之后) | 语料目前在 jerry-dev 分支的 `results/exp-cards/`,v1 以只读路径引用;`test/` 侧是 answer key,受 splits 契约约束 |
| 预算 | `{cpu_min, gpu_min, wall_min}` | 由调用方给;默认在 lock 时刻——此时 GPU 空闲,scientist 尚未启动训练 |

### 输出:裁决记录 `exp-NN.verdict.json`

与 `exp-NN.lock.json` / `exp-NN.preflight.json` 同一模式,放在卡片旁边。schema `awm-wma-verdict-v1`:

```yaml
schema_version: awm-wma-verdict-v1
card_id: exp-07
wma_skill: 3f2a1c            # skills/wma 的 commit;账本按它分组
mode: offline | online
issued_at: <ISO-8601>
levels:                      # 四层裁决,置信度各自独立
  L0_runs:      {answer: yes|no|unknown, confidence: 0.9, basis: [<evidence ids>]}
  L1_valid:     {answer: yes|no|unknown, confidence: 0.8, basis: [...]}
  L2_effect:    {metric: accuracy, direction: higher, interval: [0.02, 0.06], confidence: 0.5, basis: [...]}
  L3_worth_now: {answer: yes|no|defer, confidence: 0.6, expected_cost_h: 1.5, basis: [...]}
evidence:                    # 每条都有出处,历史的与当前的都算
  - {id: e1, path: <abs path>, locator: <...>, note: <one line>}
probes:                      # 跑了什么,花了多少,改变了哪一层(没改变的也记)
  - {id: p1, kind: static_check|unit_test|data_probe|dry_run|sample_probe, cost: {cpu_min: 2}, result: <one line>, changed: L0|L1|L2|L3|none}
suggestions:
  preconditions: [<one line each>]        # 跑之前先验证
  cheaper_variants: [<one line each>]     # 派生的便宜版本
cost: {cpu_min: 6, gpu_min: 4, wall_min: 9}
```

四层的含义与来源:

| 层 | 问题 | 主要来源 | 预期可信度 |
|---|---|---|---|
| L0 | 跑得起来吗? | 探测(单测、静态检查、试跑) | 高 |
| L1 | 会产出**有效**候选吗?(能被 grader 加载、格式对) | 探测 + preflight 之外的判断 | 高 |
| L2 | 相对现任的方向与量级? | 历史经验 + 小批量信号 | 低→中,随账本改善 |
| L3 | **现在**值不值得?(预算、剩余时间、已排除的备选) | L0–L2 + `situation` | 决策层,scientist 真正看的 |

### 对账:`reconcile` 追加到同一文件

```yaml
reconciled_at: <ISO-8601>
actual: {execution: completed, measurements: [...], decision: adopt, wall_h: 1.4}
scored:
  L0: hit | miss
  L1: hit | miss
  L2: in_interval | above | below | unscorable      # unscorable = 无同协议测量
  L3: hit | miss | unscorable                        # 事后看:说值得的兑现了吗,说不值得的省下了吗
```

**账本 = 全部 verdict 文件的集合。** 按 `wma_skill` 分组、按层算校准,就是 P2 的反馈信号;攒够了就是 v2 训练权重的数据集。

### 调用方式

- **pull**:`awm wma review --dir DIR exp-NN [--budget cpu=,gpu=,wall=] [--mode offline|online]`。
- **自动**:exp_protocol `lock` 成功后触发一次,**非阻塞**:先返回"仅历史"的即时裁决(秒级),探测完成后更新同一文件(分钟级)。scientist 可以在任一时刻继续。这需要 exp_protocol 留一个钩子(见第七节)。
- **对账**:exp_protocol `close` 成功后调 `awm wma reconcile --dir DIR exp-NN`。

## 三、两条路径

| | **P1 · 知识路径** | **P2 · 技能路径** |
|---|---|---|
| 一句话 | 把离线轨迹提炼成经验库;在线检索 + 探测 | 给 WMA 一个初始 skill,rollout → 账本 → 改 skill → 再 rollout |
| 输入 | 193 个 run / 2030 张卡 / 原始轨迹 | 初始 `skills/wma/`(角色、裁决格式、探测手册、初始启发式) |
| 产物 | 经验库:带条件、效应量、出处的条目;检索索引 | 迭代过的 skill、账本、每轮记录 |
| 机制 | 离线 LLM 提炼 + 在线按需读原始证据 + 有预算的探测 | 与 `exp_protocol_meta` 同构:变体 = skill 的 commit;其余固定;每轮一个改动;先记录后改 |
| 评估 | 离线回放的校准 | 账本随轮次的校准趋势 + KPI |
| 何时 | P2 跑起来后,作为 skill 可读的一个知识源 | **v1 先走** |

两条路不冲突:P1 的经验库对 P2 只是 skill 多了一个可读资源;P2 的账本对 P1 是"经验库有没有用"的检验器。**v1 先走 P2**,理由:循环基础设施可以复用 exp_protocol_meta 的那套;没有经验库的 WMA 也能出裁决(先验 + 探测 + 直接翻语料);账本从第一天开始积累。

"训练"在 v1 里的含义是 **inference-time 的技能迭代**——改的是 `skills/wma/` 这份文字,不是模型权重。

## 四、P2 的两种 rollout:离线回放与在线优化

P2 的循环是同一个,差别在 rollout 从哪来:

```
skills/wma (vN)
   │
   ▼
rollout:WMA 对一批方案出裁决 ──► 方案的结果 ──► reconcile 对账 ──► 账本
   ▲                                                                  │
   │            改 skill(一轮一个改动,先写记录,held-out 不迭代)         │
   └──────────────────────────────────────────────────────────────────┘
```

### 4.1 离线回放(v1 第一阶段)

**一句话**:WMA 对**历史** run 的每张卡出裁决,用已知的结果对账。零 GPU,一轮几百个样本,缺的只是"在真实容器里在线执行"这一环。

**流程**

1. **数据**。三样东西都已存在:
   - 卡片语料:193 个 run 的 2030 张卡(v1 六节格式)。用 `awm.exp_protocol.schema.migrate_v1` 转成 v2 形态;v1 没有的 `situation.trigger` 与 `setup.checkpoints` 如实为空,**不补**。
   - 原始轨迹:HF release,`awm traj fetch`,pin 到 splits 契约的 revision(`39d3fcd`)。
   - 结果:每张卡自带的 `result` / `conclusion`;每个 run 的 official accuracy(catalogue,只在对账时读)。
2. **样本构造 = 回放处境**。对 run R 的第 k 张卡,WMA **可见**:
   - R 的卡 1..k 的第 0–4 节;R 的卡 1..k−1 的第 5–6 节(之前实验的结果它当然知道);
   - R 的事件流,**截断到卡 k 的 launch 之前**;
   - R 的工作区快照中卡 1..k 引用过的文件(HF 快照是**终态**,可能含卡 k 之后写的文件——所以只放卡片 cite 过、且能证明在卡 k 之前存在的;做不到证明的样本打上 `snapshot: final_state` 标记,分析时可剔除);
   - 语料里**其他** run 的全部内容(train 侧)——这是"历史经验"。
   
   **不可见**:R 的卡 k 的第 5–6 节、卡 k+1 及以后、R 的 official accuracy。
3. **裁决**。`mode: offline`。探测层**受限**:只有静态探测——读代码、读数据文件、读配置、对照历史;**不能**试跑训练或评估(历史 run 没有 checkpoint)。
4. **对账**。与卡 k 的 `result` / `conclusion` 对账;L2 的区间用同协议的测量;run 级的 official accuracy 只用于该 run 最终 adopt 那张卡。
5. **汇总**(每轮):
   - 每层校准:L0/L1 的命中率(**历史里 `result.execution` 100% 有值,213 张 failed/killed——这是现成的 L0/L1 标签**);L2 的区间覆盖率与 Brier;L3 对 `decision` 的预测。
   - 排序质量:同一 run 内多张卡的 L3 排序 vs 实际 adopt,以及 run 间的排序 Spearman。
   - **与基线比**:查表基线(main 上 Robert 的 lookup table,"表有信息时"是强基线)、常数预测、随机。
   - 省下的 GPU 小时(代理):WMA 说不值得、实际 reject/abandon 的卡的 `wall_h` 之和;误杀:WMA 说不值得、实际 adopt 的卡。
6. **迭代**。改 `skills/wma/`,一轮一个改动,记录到 `doc/wma_iterations/<date>-round-NN.md`(模板复用 `skills/exp_protocol_meta/iteration_record.template.md` 的结构)。

**泄漏规则**(硬约束):
- **时间截断**:上面第 2 步,机械执行,写成回放器的代码而不是靠自觉。
- **splits 契约**:`gsm8k-gemma-holdout-v1` 的 train 143 用于迭代,test 50(gemma)是 held-out——只在"是否晋升"时跑一次,永不据此改 skill。
- **不看分数**:official accuracy 只在 reconcile 读。

**能测 / 不能测**:

| 能 | 不能 |
|---|---|
| L2/L3 的校准与排序(判断层) | 探测层(动态探测无法执行) |
| L0/L1 在 failed 卡上的召回 | scientist 的采纳率(没有活的 scientist) |
| 证据使用是否合理(basis 指向的东西对不对) | 交互时机、预算分配 |
| 裁决格式够不够用 | 真实处境(历史卡的 `situation` 是空的) |

**标签噪声**要正视:语料 `verdict` 59% 是 inconclusive,L2 的标签很脏;L0/L1(execution)和 decision 干净得多。第一阶段把力气放在 L0/L1/L3。

### 4.2 在线优化(v1 第二阶段)

**一句话**:WMA 挂在 H100 上真实 scientist 的 lock 时刻,出裁决;close 后对账。真实,但一轮只有个位数样本。

**流程**

1. **环境**。H100 cell,scientist 遵守 exp_protocol;WMA 与 scientist 同沙箱或旁侧,由 lock 钩子调用;GPU 在此刻空闲。
2. **裁决**。`mode: online`。探测层**开启**:静态 + 动态(在 scratch 副本里 20 步试跑、用 parent model 在 watch set 上采样、跑 scientist 的单测),受预算约束。**两阶段输出**:即时裁决(仅历史,秒级)→ 探测后更新(分钟级);scientist 不必等。
3. **对账**。close 后 `reconcile`。
4. **汇总**(每轮):离线的全部指标,加上探测层专属的:
   - 探测的信息价值:`changed != none` 的探测占比;
   - 探测成本 vs 省下的 GPU 小时;
   - scientist 的采纳率:建议(前置条件 / 便宜版本)被采纳的比例,及采纳后的结果。
5. **迭代**。同上;变体 = `skills/wma` 的 commit;每变体 ≥2 seed;一个任务 held-out;规则与 `exp_protocol_meta` 完全一致。

### 4.3 两者的区别

| 维度 | 离线回放 | 在线优化 |
|---|---|---|
| 处境来源 | 历史卡 + 轨迹重建(`situation` 为空) | 真实 scientist 当下(`situation` 有内容) |
| 探测 | 只有静态 | 静态 + 动态 |
| 每轮样本 | 数百 | 个位数 |
| 成本 | CPU / API | GPU cell |
| 泄漏风险 | 高——靠时间截断 + 终态快照标记 + splits 契约 | 低 |
| 标签质量 | L0/L1/decision 干净;L2 噪声大 | 真实但少 |
| 能优化的 skill 部分 | 判断层、证据使用、裁决格式、初始启发式 | 探测手册、预算分配、两阶段时机、建议措辞 |
| 对账延迟 | 即时 | 小时级 |
| 主要指标 | 各层校准、排序 vs 查表基线、失败卡召回 | 省下的 GPU 小时、探测信息价值、采纳率 |
| 基础设施 | 回放器(时间截断的样本构造)+ `awm wma review/reconcile` | exp_protocol 的 lock/close 钩子 + H100 发射器 + `exp_protocol_meta` 的循环 |

### 4.4 从离线到在线的门槛

三条都满足再上 H100:
1. L3 与 L2 的校准在 train 侧**显著优于查表基线**;
2. L0/L1 在历史 failed 卡上的召回达到事先定的线(建议 ≥ 0.8);
3. 在 held-out(gemma 50)上跑一次,上述两条不塌。

上线后第一轮**只跑 baseline skill**(≥3 cell),先看探测层的信息价值和成本,再谈变体。

## 五、`skills/wma/` v0 的内容

- **角色**:世界模型,不是策略;估计不决定;建议依附于裁决。
- **输入清单**:card 第 0–4 节、session 目录、`memory/index.md`、语料(只读)、经验库(如有)。
- **裁决格式**:第二节的 schema,逐层写清什么算 basis。
- **探测手册**:每种探测的种类、成本、前提、能改变哪一层;只读、有预算;"不会改变裁决的探测不跑"。离线模式只允许 `static_check` / `data_probe`。
- **初始启发式**:一批粗先验(如"<2k 条 SFT 很少让 gsm8k 动 3 分以上"),**逐条标明是先验、待账本修正**。
- **禁区**:不改 scientist 工作区;不提新方向;不插话;不看 held-out。
- 与 `exp_protocol` 一样:真身在 `skills/wma/`,`.claude/skills/` 放 symlink;迭代者读 `skills/wma_meta/`(第一版可以直接复用 `exp_protocol_meta` 的结构,只换指标)。

## 六、账本与指标

账本 = `exp-NN.verdict.json` 的集合,按 `wma_skill` 分组。`awm wma ledger <dirs>` 汇总:

| 指标 | 定义 | 好的方向 |
|---|---|---|
| `L0_hit`, `L1_hit` | yes/no 与 execution / 有效候选的一致率 | 上 |
| `L2_coverage` | 实际 delta 落在区间内的比例;配 Brier | 上;区间不能靠放宽混过去——同时报区间宽度 |
| `L3_hit` | 说值得的兑现(adopt 或 delta>0)、说不值得的省下(reject/abandon) | 上 |
| `gpu_h_saved` / `gpu_h_wrongly_killed` | L3=no 且实际 reject/abandon 的 wall_h 之和 / L3=no 且实际 adopt 的 | 前者上,后者零 |
| `probe_voi` | `changed != none` 的探测占比(仅在线) | 上 |
| `adoption` | 建议被采纳的比例(仅在线) | 上下文,不是目标 |
| `cost_per_verdict` | 平均 CPU / GPU / 墙钟 | 下 |

**KPI**(定义文档第四节):scientist 的 GPU 小时利用率——在这里的可算形态就是 `gpu_h_saved − gpu_h_wrongly_killed`,除以 cell 的总 GPU 小时。

## 七、与 exp_protocol 及 evolve 分支的接口

WMA 线**只读** card、**只写** `exp-NN.verdict.json`。需要 exp_protocol 那边留两个钩子,都是一行调用、失败不阻塞:
- `lock` 成功后:若 `awm wma` 可用则 `awm wma review --dir DIR exp-NN --mode online`(后台,非阻塞);
- `close` 成功后:若存在 verdict 文件则 `awm wma reconcile --dir DIR exp-NN`。

离线回放阶段**不需要**这两个钩子——回放器自己调 review/reconcile。钩子等到 4.4 的门槛过了再和 evolve 分支协调加入。

## 八、研发顺序

0. 本文档 + 定义文档第四节修订(职责措辞、线名)。
1. `skills/wma/` v0 + verdict schema + `awm wma review | reconcile | ledger`(全部 CPU 可测;review 的推理部分是调用模型,测试用假模型)。
2. 回放器:时间截断的样本构造 + 泄漏规则的测试 + 一轮基线跑分(baseline skill vs 查表 / 常数 / 随机)。
3. 离线迭代 N 轮,每轮一个改动、一份记录。
4. 过门槛 → 与 evolve 分支协调钩子 → 在线第一轮(仅 baseline)。
5. P1 经验库:等在线跑起来、账本说明"历史证据用得不好"时再做——那时知道该提炼什么。

## 九、有意不做的与已知风险

- 不训练权重;不提新方向;不阻塞训练;不做机械检查(那是 preflight 的)。
- **风险 1 · 回放处境失真**:历史卡没有 `situation`,WMA 在离线时看不到"剩余预算、排除的备选"——L3 在离线阶段只能用卡内信息近似;在线阶段才是真的 L3。
- **风险 2 · 终态快照泄漏**:靠标记与剔除控制,不能根除;分析时报两组数(含 / 不含 `final_state` 样本)。
- **风险 3 · 标签噪声**:L2 的语料标签 59% inconclusive;第一阶段的结论主要建立在 L0/L1/L3 上。
- **风险 4 · 同族模型**:WMA 与 scientist 可能是同一族模型,它的"先验"和 scientist 的重合;价值必须来自历史与探测,账本会显示先验层贡献了多少。
