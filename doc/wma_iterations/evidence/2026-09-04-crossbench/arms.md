# 跨 benchmark 的最小可解释方法矩阵（只读建议，2026-09-04）

本建议只读当前未提交实现、2026-09-04 分析记录及其已完成结果；未调用模型、未提交Slurm、未修改repo。当前实现正在其他任务中变化，下面的实现观察是此次读取的状态，不是已冻结或已验证上线的承诺。

## 建议：五臂递进，优先回答“改进究竟来自哪里”

所有 setting 内保持相同 benchmark、base model、scientist Opus 4.8、effort、context限制、PTB版本、总时限、GPU、scorer/judges、数据与网络规则。每臂四条独立replicate。WMA存在的臂也统一使用Opus 4.8及相同review budget；明确记录scientist和WMA两个模型字段，不能只把scientist换4.8、侧车继续隐式继承Opus5。精确模型ID/上下文长度由当前环境可用映射验证，不从旧`claude-opus-5[1m]`字符串机械猜测4.8能力。

| 臂 | scientist规程 | 主要训练前候选 | WMA联合排序 | 被选正式card的WMA | 主要用途 |
|---|---|---|---|---|---|
| R raw_ptb | 标准PTB流程，完全不安装AWM exp_protocol/WMA skills、私有history | 自主，不强制brief | 无 | 无 | 用户要求的“什么都不加”基线 |
| P protocol_only | 同波冻结的单方案exp_protocol，含preflight、版本锁、动作和launch记录 | 单方案流程，不强制2–3份brief | 无 | 无 | 与R比较protocol整体作用 |
| S scoped_single | 与P同一公用规程 | 与P相同 | 无 | 新证据适用范围skill，阻塞返回 | 测单卡WMA整体净收益 |
| M multi_self_plus_single | 与S相同，增加通常2–3个真实brief、读侧车前首选与最终选择记录 | scientist自己选 | **关闭** | **与S完全相同的新规则单卡WMA** | 分离“让scientist多想几个方案”的作用 |
| J multi_joint_plus_single | 与M完全相同候选要求、预算、证据与记录 | scientist看联合意见后自己选 | **一次共同上下文比较** | **与S/M完全相同的新规则单卡WMA** | 测联合比较的增量价值 |

预先指定四个解释方向：

- **P − R**：protocol整套流程的净效应；R没有卡片要求，机制统计不能强迫与P同口径。
- **S − P**：有新证据规则的单卡WMA整体效应，含其调用成本与受限probe能力；**不**是“新规则比旧规则更好”的单编辑估计。
- **M − S**：显式多候选/自选流程的增量效应，selected-card review相同。
- **J − M**：联合WMA排序的增量效应，候选要求和正式review相同；这是比较机制最干净的主对照。

R/P/S/M/J是新方法体系的组件拆分；各臂共用新的版本/数据边界基础设施，不能拿S与旧ae46724在线结果之差当skill-only提升。

这里M故意保留单卡WMA：如果把M设成“多候选但完全没有WMA”，J−M会同时加入联合比较和正式单卡审查，不能单独归因联合比较。用户若特别需要完全无WMA的多候选臂，可增加第六臂M0；预算限定五臂时可用M0替换P，但会失去protocol-only净效应。不要把混合差异写成单因素提升。

如果压缩成四臂，建议 **R/S/M/J**，明确放弃protocol-only归因；仍保留最关键J−M与M−S。不建议R/P/S/J：J−S会同时多出候选生成和联合排序，正好重现当前最需要排除的归因问题。

## 为什么不是直接拿旧“最好A–F”跨benchmark

来源：`doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/complete-cells.json`，冻结79个原PTB validator/judges clean结果，时间2026-09-04 10:34:57 UTC。以下重新从其accuracy计算；±为cell间样本SD（pp），不是置信区间。

| 旧cohort | n | 官方最终accuracy均值 ± SD |
|---|---:|---:|
| w10 v0.2阻塞baseline | 8 | 72.318% ±2.738 |
| c10同波无WMAcontrol | 6 | 74.551% ±4.645 |
| A format-floor | 4 | 69.086% ±2.787 |
| A+B format-floor-width | 4 | 71.778% ±1.381 |
| C probe-before-fail | 4 | 71.797% ±6.142 |
| D checkpoint-precondition | 4 | 72.745% ±3.439 |
| E L3-evidence | 4 | 71.190% ±1.326 |
| F decisive-precondition | 4 | **75.834% ±3.589** |
| G scoped-probes | 0 | **该冻结样本无完成结果，不填分数** |
| H soup-ingredients | 0 | **该冻结样本无完成结果，不填分数** |

F可以称这份快照中A–F“观察均值最高的候选”，不能称已证明best、已晋升baseline或稳定优于control：

1. F只有4条，原独立确认/held-out门没有通过；baseline n8、control n6且尾部未齐。不能按最高均值再筛选后假装预先选定赢家。
2. 所有旧WMA臂保持原access flags。先前A/A+B/C/D原终稿flags分别14/29、11/23、10/32、9/20（D为当时3-cell口径）；D后补尾部不能误用这个旧分母作为完整4-cell数。本次直接读取79-cell JSON指向的原result_dir，E为 **10/29**、F为 **7/25** `leak_suspected`终稿；两者也没过原zero-leak门。不要事后清flag或换guard使它们晋升。
3. D/w09r03/exp01有原regex漏报的库调用读测试题与答案、再传给scientist；人工标记必须与原judge clean并存。它证明原自动clean不足以表示语义无越界，未证明该暴露用于训练或改变了分数。
4. F/w12r03/exp04由旧训练尾部平台筛掉新100k/30%MATH mixture；F/w12r02/exp04以独立20k短程试验筛掉90k完整训练。两次取消可复原，但原完整方案的终点未知。新证据规则正是为避免这种过度外推，不能一面修F机制、一面把F先命名为best。
5. A+B必须相对A解释B的增量；C最高79.378%与最低64.746%同臂，显示择单条冠军的风险。
6. G/H已有四replicate预登记，但未在上述快照完成；不从运行/排队状态猜任何分数。I/J旧Round04设计未启动。

因此建议跨benchmark测试的是**最有依据的几个方法变种**，不是包装成“过去证明最好的skills”。如果用户希望保留一个旧参考，F可在GSM8K额外四replicate作为“legacy-F参考”，使用相同新基础设施、明确skill版本；它是新runtime下的重测，不能与旧F直接池化。若要单独测新证据规则，则需要其准确父skill与“父skill+规则”相同输入/模型/runtime的额外对照；五臂主矩阵本身没有这个估计量。

## 当前实现对矩阵的约束：需要明确、冻结的处理开关

实际读取的代码：

- `skills/exp_protocol/SKILL.md` 已要求所有主要training（包括no-WMA control）先propose/compare/choose，通常2–3个真实brief。
- `awm/exp_protocol/run.py` 对training family强制已记录choice；所有protocol launches检查当前proceed、lock/fingerprint以及对应verdict版本。
- `awm/wma_client.py:compare_and_wait` 没侧车时记录`not_attached`并返回，有侧车就发联合比较。`record_choice`必须有已结束comparison请求。
- `awm/wma/compare.py` 是一次真正共享上下文的ranking调用，区别于旧多card逐个review按confidence排队；明确不是launch approval。
- `skills/wma/SKILL.md` 新Evidence scope要求将证据适用性对齐parent/data/objective/schedule/evaluator；manual/example也同步修正smoke不能证明完整训练终点的示范。
- `awm/wma/isolation.py` 新限制为CPU/static broker、白名单导出、Landlock/seccomp；不允许GPU设备、联网下载和benchmark evaluator。即使旧、新skill同SHA之外一致，其probe能力变化仍属于共同新runtime cohort。

**现有默认路径不能直接跑上述五臂而保持臂定义不串线。** 推荐在冻结实验合同中显式定义（名称可由实现决定）：

1. `proposal_mode=single|multi`：P/S单方案不强迫写多个brief；M/J同样2–3真实候选条件及singleton例外。单方案可以保留轻量choice绑定记录，但不能为满足schema编造“实际没有替代项”的singleton_reason。
2. `joint_compare=off|on`：P/S/M记录明确的预登记not_requested/disabled处理状态，J发一次比较；不能用假not_attached、故意timeout/失败、隐藏已attached侧车来模拟关闭。
3. `single_card_review=off|on`：P关、S/M/J开，所有打开者保持阻塞等待；任何改变执行版本必须取得匹配的新裁决。
4. 为上述设置生成匹配public skill指令/manifest冻结值。若只有控制代码、skill仍要求compare，会产生处理依从性错误。

这些是新增研究处理配置，不是通过绕开现有访问/启动检查来伪造对照。R应使用标准PTB agent/launcher，不安装AWM workflow；共用PTB评测、模型入口、Slurm ownership/routing、最终结果验证，但不能往R额外注入AWM流程后仍称raw。

当前git diff包含未提交新实现与文档；设计草案开头“尚未修改”已经滞后于工作树，需在实际启动前更新状态、测试适用合同、冻结SHAs与skill hashes。此报告没有声称这些新开关已经存在或任何安全测试已通过。

## 跨benchmark与GSM8K的分母

- 每个选定benchmark × base-model setting独立跑5臂×4replicate＝**20cells**；若4臂则16cells。不要把同一cell内多个training当额外replicate。
- **最新方法GSM8K**至少要与同波同Opus4.8的匹配对照一起跑，优先完整同一五臂（20cells）；只跑J四条然后对照旧Opus5分数无法归因。若预算必须更窄，GSM8K至少R/S/M/J四臂，明确失去P拆分。
- user此次明确跨benchmark可形成新比较合同；仍不把AIME悄悄当迭代筛选集，原AIME promotion-only边界保留，除非用户后续明确调整。
- 同一个benchmark保持base checkpoint不变。跨benchmark汇总用各benchmark的臂内相对差/预登记归一化，不能把GSM8K accuracy、其他评分器分数直接平均宣布winner。
- 四replicate是机制筛选，最终score均值/SD/每cell与成本都报，保留PTB validator/judge flags、独立manual access audit和失败类型；不把失败的cells静默补成功再减少失败率。
- 拟观察：同预算官方最终score；joint前后选择与实际执行是否改变；适用范围错误、未验证proxy导致的取消；正式blocker修复；版本一致性；有用GPU时间、review/comparison成本。候选数、no比例和采纳率只是解释变量。
- 不取消正式selected-card review来抵消J额外调用，新增联合调用成本属于方法结果；原来的独立确认与held-out晋升要求仍需满足。

## 证据索引

所有下面相对路径在 `/home/robtang_google_com/gangda_workspace/agentic-world-model/` 下：

- `doc/wma_iterations/2026-09-04-status-and-trajectory-review.md`：64-cell早期snapshot、旧flag/间接数据泄露、版本审查问题和小样本限制。
- `doc/wma_iterations/2026-09-04-usage-proposals-and-false-kill.md`：79-cell最新snapshot、304/304单card请求、F两例真实取消但未测反事实。
- `doc/wma_iterations/2026-09-04-why-no-net-gain-and-horizon.md`：review次数不等于新训练反馈，等待并非纯GPU idle，首训前选择混杂。
- `doc/wma_iterations/2026-09-04-sft-efficiency.md`：训练次数/题量/解码与首次观察时刻不能混算；没有稳定效率优势。
- `doc/spec/2026-09-04-wma-decision-review-redesign.md`：新规则与joint比较的分离设计及同candidate-pool control要求。
- `doc/wma_iterations/2026-09-03-round-04-online.md`：G/H冻结hash和submitted jobs；I/J未启动。
- `doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/complete-cells.json`：本报告A–F表与每cell result_dir/manifest/spec/job主索引；对应E/F原终稿路径为该result_dir下`task/memory/cards/exp-*.verdict.json`。
