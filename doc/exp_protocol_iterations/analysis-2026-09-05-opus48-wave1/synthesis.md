# Opus4.8 wave 1：独立综合、修复候选与下一波研究预算

2026-09-05。作者：独立 Codex synthesis reviewer。本文是提供给 planner 的证据综合与建议，不是队列释放决定。只读取资料并写本文；没有训练、评估、修改协议源码或操作队列。

## 证据范围与审阅身份

已逐一读取 [control-review](control-review/README.md) 下 n03g01—n03g04 和 [e-review](e-review/README.md) 下 e03g01—e03g04 全部八份报告、[validation-summary.json](validation-summary.json)、逐报告纠正、阶段锁最小复现、冻结 E 文档与历史固定 vLLM 源码证据。原始轨迹行号和 UTC 时间采用逐会话报告的约定；这些报告保留 receipt → cell → 原始任务目录链。

这是 **6 个 validator-complete/clean、1 个完整但 flagged、1 个 failed/truncated** 的已结束预声明比较窗口；不是八个 clean 新结果。按完整 discovery block 触发综合是合理的，无须添加凑够八个 clean 的重复。

请求的后台 Claude 分析没有完成：默认启动因创建 `~/.claude/jobs/` 返回 EROFS，随后两次提权尝试均在 automatic approval review 超时，未取得正在运行的 Claude 会话，也没有 Claude 读文件的证据。参见 [claude-blocker.md](e-review/claude-blocker.md)。当前八份报告与本综合都是独立 Codex 审阅，不冒充 Claude；Claude 交叉分析仍为未完成步骤。

## 1. 成绩、过程与分母

| arm / cell | 官方准确率 | 验证分类 | 会话 h | 首次真实训练 h | protocol h | waiting h | greedy 提交证据 | RL / RFT | 最大内部评估 n | 结束情况 |
|---|---:|---|---:|---:|---|---|---|---|---:|---|
| none / n03g01 |59.5148%|clean|8.91|0.170|0|7.78*|显式 temperature=0|无 / 有|1319|主动结束，余约1.10h|
| none / n03g02 |59.4390%|clean|8.48|0.200|0|5.63*|显式 temperature=0|无 / 有|800|主动结束，余约1.53h|
| none / n03g03 |缺失|failed/truncated|0.790|0.120|0|0.50*|未交付|无 / 未执行|200|等待后台任务时意外完成；没有 final_model|
| none / n03g04 |70.4321%|clean|6.52|0.165 smoke；0.195 full|0|5.87*|显式 temperature=0|无 / 有|1319|主动结束，余约3.48h|
| E / e03g01 |63.7604%|clean|7.05|0.3289|不能可靠分离|不能可靠分离|显式 temperature=0|无 / 有|1319|主动结束，余约2.94h|
| E / e03g02 |54.2835%|flagged complete|8.91|0.4786|不能可靠分离|不能可靠分离|仅 do_sample=false，不足|无 / 有|500|异常等待后主动结束，余约1.08h|
| E / e03g03 |42.7597%|clean|9.81|0.5061|不能可靠分离|不能可靠分离|仅 do_sample=false，不足|无 / 有|300|末轮补救失败，余约0.18h|
| E / e03g04 |58.3017%|clean|7.77|0.5175|不能可靠分离|不能可靠分离|显式 temperature=0|无 / 有|400|主动结束，余约2.22h|

`*` waiting 是分类器工具时间，含有产出的后台训练/采样，不能当成闲置 GPU 时间。内部最大 n 不包含外层官方评估，不替代官方分数。训练 smoke 也算模型执行，不能从锁覆盖审计中略去。

| cohort | n | 官方均值 / 范围 | 平均会话 h | 平均首次训练 h |
|---|---:|---|---:|---:|
| none clean |3|63.1286%；59.4390—70.4321%|7.970|0.178（n04含smoke）|
| E clean |3|54.9406%；42.7597—63.7604%|8.210|0.451|
| E all complete |4|54.7763%；42.7597—63.7604%|8.385|0.458|

clean E 比 clean none 低 **8.1880 个百分点**；包括 flagged 的 E 比 none clean 低8.3523个百分点。这是小样本、条件于完成的描述，不能认定稳定处理效应。none 的失败率1/4必须与均值同时报告；E 的1/4 flagged 不能从成本审计中消失。不能给 failed 的 n03g03 编造0%或把内部42%算正式成绩。validation-summary 的 `accuracy_primary.n=4` 包含 E flagged，不能直接重命名为 clean 均值。

外层的模型、10h预算、任务和资源契约匹配，但内层不是固定配方：E02用LoRA后续训练，其他会话多为从base重复SFT；数据、学习率、参数路径、attention、优化器、上下文与解码均不同。历史 Opus5 数据只能提供策略先例，不能把当前降分全部归因于 Opus4.8 或协议。

## 2. 解释按证据强弱排序

### A. 输出延续与实际解码是明确、反复出现的机制；尚不能解释全部组间差距

三个完整 none 都遇到few-shot任务输出继续生成下一道题/最后数字评分损失；E03、E04也有清晰同类轨迹，E02记录多答案33→0和首答正确但后续丢失19→0；E01则记录RFT后输出429→828 tokens，不足以单独诊断同一种错误。至少六个完整会话有直接续写/停止问题证据。

同权重改变有效解码已有三例直接观察：n01在n150上37.33→46.67%，n04在n150上61.33→70.67%，e04在n150上40.0→52.7%。这些是各自局部对比，不能把+9或+12.7个百分点代入别的会话。n02另有46.67→50.0%的小n读数。

最重要的纠正是 **e02/e03并未证明实际greedy**。两者最终generation JSON只有 `do_sample:false`，缺temperature；实际评估请求也没显式temperature。固定vLLM路径的 `get_diff_sampling_param` 接受temperature/top_k/top_p等字段而忽略do_sample。因此“中性sampling默认仍生效”是有源码依据的推断，而不是已记录每个request的完整resolved RNG证据。不能称其已确定greedy、也不能宣称补temperature必然提分。

证据：[E02](e-review/e03g02.md) L6641/07:19:00、L7156/07:30:09；[E03](e-review/e03g03.md) L8918/08:24:07；[固定运行时源码](e-review/historical-vllm-source-primary.txt) L3040—3059。e03的最终权重复制和字节核验正确，不能把低分重述为错误模型导出；问题是“保存了所写配置”不等于“实现了所声称的服务语义”。

### B. E有三次确定的采样执行损失；工具安装没有形成完整覆盖

- e01：旧 `prompt_token_ids` 调用不被支持，L5975/01:41:05→L6207/02:32:54，约0.9h含延迟发现/孤儿engine。
- e03：同API错误，L5213/01:42:01→L5611/03:03:09，约1.4h。修为TokensPrompt后继续。
- e02：将未来训练输出 `combined_rft.jsonl` 放进当前采样卡的输入列表，lock override虽然记录了理由，live launch仍正确要求已存在、已固定hash的真实输入。运行/record_vllm两次拒绝，约01:53→06:03共4.17h等待。数据hash拒绝是确定的；为何engine退出等待如此久，尚未完成代码层根因定位。

这些约6.5h是三次事件的粗略wall损失，不是精确GPU busy time，不能再把其卡片继承的时长重复相加。负向RFT训练耗时与真实故障分开：它们产生了科学反例，并非全部“浪费”。

E01的RenderedTrainingBundle、E01/E03的SaveSafeTrainer、E04的GenerationSaveContract确实使用；guarded-run使用覆盖各不相同；四个E会话没有成功的raw sampling adapter capture。e02遇到拒绝后绕过wrapper。实际采用率和成功记录率必须成为结果维度。

### C. e02/e04构成阶段锁scope的反例/正例，不支持放松输入冻结

E04 L5250/01:43:10 明确写：“RFT data must exist before I can lock a training card, so this is two cards: exp-05 (sample) → build combined data → exp-06 (train).” 随后成功完成采样。

[最小CPU复现](e-review/reproduce-future-data.txt)证明：缺失未来文件，即使override仍拒绝；有文件但无锁定hash仍拒绝；采样卡不声明未来训练输入则通过。这仅复现predicate（上层plan/lock部分被stub），不能声称完整运行时验证。冻结 E `sampling-evidence.md` 原本说明record_vllm在推理前检查，但示例已先有llm；`execution-records.md` 已说明真实hash必需。因此候选要修的是阶段表达、提前readiness及错误引导，不是重复添加已有规则或开“未hash训练输入”后门。

### D. none n03g03 是任务生命周期损失；主动早退另有六例，二者不能混合

n03g03 L3227/23:08:10取得后台waiter `bpksa8ghd`；L3911/23:23:12说“it will re-invoke me ... Standing by”，end_turn被报告completed，23:23:17同handle被杀停。没有最终模型；第二次训练仅有约0.29h截断日志，9.21h预算未使用。应修共同scaffold的生命周期/交付验证，不能靠新增科学家训练配方解决。

另六个完整会话主动提前结束：n01/n02/n04/e01/e02/e04，余约1.1—3.5h；e03几乎用完预算。n01/n04把几个配方后的平台称“capability ceiling”，证据不支持。可提供基于实际剩余时间的机会/风险记录，但不强制每次把预算烧光，不把负面枝数或方法数当成果。

### E. BF16解释目前未识别；策略变化和采样噪声同样未被控制

七个完整会话都通过BF16权重路径；E02另有LoRA参数/更新路径差异。n04较高分同时改变text-only模型类、flash、数据、LR、优化器等。因此既不能用BF16解释低分，也不能用n04反证任何数值问题。最小受控更新诊断仍值得做，必须记录实际trainable/master dtype、权重delta、零更新比例和固定diagnostic loss；不把“全用FP32”写成协议。

### F. 同题互补是真实研究线索，尚无可执行组合收益

| 比较 | 同题 n | A-only correct | B-only correct | 已观察含义 |
|---|---:|---:|---:|---|
| n01 v3 / v5 |250|18|18|同分154仍有不同错误|
| n02 exp4 / exp7 |500|47|89|总分提升伴随损失；多个配方因素同变|
| n04 iter2 / iter3 |1319|127|131|只净增4题但分歧258题|
| n04 iter3 / RFT |1319|148|119|RFT较低分仍有119个增量正确项|

来自 [paired-counts.json](control-review/paired-counts.json)，按typed真实sample ID对齐，不按数组位置。没有实际ensemble/selector/soup结果；gold oracle union只是上限。不能用这批测试ID或答案训练选择器，也不能把重复使用过的GSM8K测试重新叫独立确认集。E组尚无这些paired计算，不捏造互补收益。

## 3. 协议成本和计划形态：目前能说什么

E真实首次训练平均约27.5min，none clean约10.7min；差约16.8min是会话准备/基线/决策/协议的混合量，不是协议纯开销。E的分类器protocol 2.42、9.18、7.08h等吞进wrapped训练/采样，不能说“表格花了9h”。逐卡执行时间不足以恢复模型推理与混合shell的纯记账分钟。

E共有6、7、12、10张卡；卡数不是独立样本。e01最终自述7张而实际6张；e02 exp06申报7,473与实际19,765行不符；e03 stop hook确实拦到open exp07并促其关闭。这说明卡有实际作用，也仍需实际产物比对，而非仅看close率。已审阅E03的12次guarded attempts有先锁证据，E04直接decode检查亦已先锁；不能采用auto parser的假“session开始即eval/0覆盖”结论。

所有七个完整会话都尝试RFT，没有一个执行policy-gradient RL。因此现有证据反对“E普遍禁止RFT”，却不能得出RFT值得强制执行。大多数分支是从base重新SFT，不能称迭代policy训练。

## 4. 三项修复E包：建议命名 E-repair，完整冻结后才试验

参考树保持原 E `dcfa742dbc8813970192efe3fbf2bd30dfc38ea9` / protocol tree `b33422364c70f4ea3c08ff83c97009a41438caa6`。源码修复、示例和指引一起形成新包；不是三条孤立GPU消融。以下是候选，不声称已实现。

| 组件 | 证据与目的 | CPU验收 | 仍需GPU/真实运行证据 | 边界 |
|---|---|---|---|---|
| 1. 阶段化采样与前置readiness | e02失败、e04两卡成功。先锁采样输入→生成并持久化→再建训练卡/固定实际文件；把卡/路径/API静态拒绝放在engine创建前 | 缺文件/无hash/改source仍拒绝；单独采样卡通过；stage示例走真实lock/readiness；失败路径断言factory未调用 | 实际native sampler成功、生成结果可追溯且新训练锁覆盖真实产物 | 不允许尚不存在的训练输入；不重复声称已有hash规则是新功能 |
| 2. 原生API可用示例与有界失败观察 | e01/e03旧keyword；e02延迟退出。对固定版本用TokensPrompt示例；观察真实producer exit/owned descendants并保留错误与返回结果 | fake producer即刻失败/超时/中断/错误child身份；返回raw先落盘再parse；不把存活PID/锁文件当成功；不全局kill | 固定image native inference最小smoke，异常回收真实owned子进程与耗时，随后完整会话采用率 | 原helper已有raw-before-parse和局部ownership边界；改的是前置调用、示例与观察，不声称新发明记录能力。engine未返回的tokens无法凭空恢复 |
| 3. 实际服务参数证据 | e02/e03意图与生效可能不符；e01/e04正例 | 记录declared JSON、evaluator request、解析后支持字段及unknown；覆盖do_sample-only、显式temperature、request override、保存后修改；不自动改temperature | e02/e03同权重温度对照，固定题集/其他请求、原始采样重复；真实fixed-vLLM请求证据 | byte hash/CPU metadata≠loadable≠greedy≠quality；不强制greedy，不承诺提分 |

三者交互：readiness先拒绝错误stage/API，避免为必失败任务初始化engine；正确native调用加失败观察/原始结果保存，使采样可继续成下一阶段；服务证据把“按选定decoder评估”变得可检查。总包效应不能归因到单项。

另一个 **共同运行时修复** 是n03g03的background/end_turn交付契约。应先CPU复现并查实际scaffold，再冻结为所有相关臂共同环境。它不是E独享处理。若改变正式PTB/scaffold pin，旧结果保留旧语义，已有held任务须按新receipt冻结后才能共享新环境；不能把变更后的none/E与旧配置视为毫无差别。

支持保留的替代方向：轻量知识版用于判断process知识是否足以改变错误和机会选择；旧guard仅保留原已批准桥接问题，不自动续增。反对：再跑一整块旧E来平均掉故障、常规追加none、强制FP32/teacher/RFT/soup、以parser中的protocol小时排名、删除输入hash校验、自动重写decoder或全局杀vLLM。

## 5. 下一波矩阵、预算与16 GPU目标

GPQA明确跳过，不把授权/下载依赖引入本波。AIME2025保持独立晋升确认用途，本轮不得读取其内容或用作开发诊断。HumanEval是获准的新增discovery任务，须独立完成数据、真实compute-node执行隔离、官方评估/validator及证据留存准入；下载完成或CPU测试通过不等于已准入。

建议按决策价值组织可并行工作，不为16这个数字制造重复。下表是planner可转为不可变manifest的研究库存建议，不宣称目前已提交或占用。

| 方向 | 本波预算 | 具体决策 | 先决条件/停止条件 |
|---|---:|---|---|
| 已批准GSM8K knowledge |既有4会话，40 scientist GPU-h|知识版能否减少上述错误，是否值得保留|核验现有精确receipt；不新增重复；共同环境若变更需重冻结|
| 已批准GSM8K guard桥接 |既有4会话，40h|旧guard在Opus4.8下的历史桥接问题|若planner认为问题已无决策价值可整块撤回，不能仅为填GPU保留|
| GSM8K E-repair discovery |2新会话，20h|修复包的真实采用、故障时间、有效decode与同预算产出|三项CPU验证和native smoke通过；最多另2次只能在具体未决问题出现后申明|
| HumanEval none / knowledge / E-repair |各4会话，合计12、120h|任务迁移及三个完整包差异|延续用户已批准每任务4次的比较合同；E替换原草稿必须显式冻结新处理，不能静默沿用旧E标签|
| e02/e03同权重decode诊断 |2个固定artifact研究，初始每个≤1 GPU-h|估计该缺失字段在这些权重上的作用，决定是否还需训练解释|先验证权重仍存在且身份可冻结；card/check/lock先于任何模型执行；不改写原官方结果|

名义合计 **22个10h科学家会话 + 2个短diagnostic**，上限约222 scientist/diagnostic GPU-h，不含final eval、judges及排队开销；其中8个是已有批准待执行工作，12个来自已批准HumanEval任务矩阵，只有E-repair的2个完整会话为修复探索。不是把所有新变体默认扩到4或8。若已执行/取消，应扣除对应库存而非重复提交。

两份诊断的现有weights若无法恢复，不应重训整个旧会话来凑这两个位置。可改为明确单独研究的BF16/FP32更新健康配对：共同base/data/order/LR，两个臂各≤30min、50—100step，先看weight delta和固定loss，再决定是否需要新seed。该替代需要自己的完整科学规格和卡，不能只是“有两个空GPU”。配方、格式、teacher/RFT与组合研究留作有证据触发的后续，不在这轮全部扩成全矩阵。

队列算术与科学预算分开：若且仅若这24个独立cell都已验证且仍未开始，16运行+8个真实 `PENDING(JobHeldUser)` 可以成立；短诊断结束后需要有可论证的新库存或降低并发，不能提前释放held底线。这不是“24个文件等于24个任务”，也不是持久16忙碌的保证。HumanEval未准入、某桥接被撤回或artifact不可用时，上述算术不成立，planner须据实时receipt/ownership重新计算。

按用户目标应先异步释放已经独立、安全、有明确问题的批次，不等HumanEval或其他straggler；随后尽快补齐跨任务和修复包。严格只使用 `gangda_exp-protocol-evolve` 的 `slurm2-a3nodesetondem-[0-1]`：OWNERSHIP OK、每job ReqNodeList匹配receipt、恢复native两节点隔离、释放后仍≥8真实held。当前文档不是这些实时门槛的证明，也不授权触碰其他队列/运行任务。

## 6. 预声明结果向量与下一次决策

每次attempt同时报告：正式分数/分母或明确失败；validator、judge和placement；首个可用incumbent耗时；真实失败计算、修复、等待和主动unused预算；真实锁覆盖；工具采用/成功capture；有效服务字段与unknown；保留下来的负分支及其后续实际用途。任务间指标分开，不平均GSM8K和HumanEval百分比。

工程验收的硬阈值：CPU中不允许无hash训练输入通过、不允许readiness失败后engine factory被调用、不伪造成功capture/交付、不误标do_sample-only为已验证greedy、不扩大进程清理scope。真实会话中只要复现同类静态API/stage缺陷便先修，不靠更多重复稀释失败率。

本轮两次E-repair属于发现研究，没有足够精度设“均值不下降即证明安全”。当前约10个百分点的跨会话差异和条件完成偏差意味着，任何新包质量晋升容忍值必须由planner在独立确认前明确设定；不能用不显著宣称等效。允许先获得可靠性/有效时间改善而不宣称准确率增益，但不可忽略明显质量退化或证据完整性损害。

下一次完整block审阅需回答：

1. E-repair的静态错误是否在engine前被挡住；raw capture有无真正成功、是否转化为后续训练？
2. e02/e03缺temperature的同权重效应多大，原配置重复波动多大；是否仍需数值/数据配方解释？
3. knowledge/guard是否改变可用incumbent速度、主动早停与失败修复，而不是仅卡数？
4. HumanEval是否保留完整官方逐题与执行证据，跨任务错误是否相同；无数据或执行权限时不形成假的成绩？
5. 任何互补分支是否通过不看gold的实际方法带来确认集净收益，或明确解决了值得付费的不确定性？

路线知识分别沉淀：scope/实际decoder/返回结果持久化属于协议事实；few-shot、teacher、RFT、FP32和soup属于带条件的scientist/WMA策略先例；分母、混合计时、Claude身份、API失败后缓慢观察、真实held算术属于meta运行知识。本文不将任何不确定配方变成科学家强制规则。
