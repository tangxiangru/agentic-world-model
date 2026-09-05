# 旧 Opus 5 hook 窗口交接核验 — 2026-09-04

结论：三个事件可以标记为 **reviewed / superseded by the 2026-09-04 integrated audit and redesign**，不需要重新调用 Opus5 重做相同窗口，也不应按旧建议自动追加 F/K/L/M/N/J 实验。未发现这些旧报告带来、最新工作尚未处理且必须阻塞新实验的新证据。此结论不是给旧候选晋升，不是清除旧 access flags，也不是证明新 Opus4.8 科学结果已有效。

本次只读 payload、Claude 最终报告（含其指向的 plans 文件）、已提交综合审计及验收记录；没有重读全量 trace、读取新 Opus4.8 in-flight、启动模型/Slurm或修改仓库/共享事件。下述 disposition 由父任务按现有 hook 接口写入实际交接记录。

## Cell / provenance 覆盖

权威新覆盖索引：
`doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/complete-cells.json`，79个完成结果，2026-09-04 10:34:57 UTC综合审计。

以每事件 `payload.clean_complete_cells` 的 `batch/cell` 为键，逐条对比事件 `results.snapshot.json` 与该索引：**30/30 cells均覆盖，job ID、result_dir、accuracy、judge_flags全部一致，0项不匹配。**这里没有重做validator；核对的是两次已冻结validator结果的身份、分数与flag。

| 事件 | 新 cells | 最新审计覆盖 | 实际 Claude 报告 | disposition |
|---|---:|---:|---|---|
| `20260904T003453Z-2cb440f05f` |11：A4、AB4、C2、c10r04 |11/11| `/home/robtang_google_com/.claude/plans/ultracode-you-are-the-lexical-lamport.md` | 已由05:52完整R2审计和10:34使用/动作审计覆盖；旧“PASS”、eligibility=0和新改scorer建议不采纳 |
| `20260904T073748Z-b062a58bc6` |7：C尾2、D4、E首1 |7/7| `/home/robtang_google_com/.claude/plans/ultracode-you-are-the-golden-mango.md` | 已由R2审计＋最新E/D专项覆盖；fence/version风险纳入redesign；旧K/L/M/N计划不执行 |
| `20260904T093840Z-0f1f9495e3` |12：control扩展1、baseline扩展4、E尾3、F4 |12/12| `/home/robtang_google_com/.claude/plans/ultracode-you-are-the-valiant-gadget.md` | 已由79-cell使用/筛选审计覆盖；F“正确节省”判断被纠正；旧F扩展/P1–P4/J/M计划被独立新研究替代 |

共享原件均在：
`/rmeng_data/robtang/wma-evolve-hook/gangda_wma_evolve/events/EVENT_ID/` 的 `payload.json`、`results.snapshot.json`、`claude.output.json`、`status.json`。
三个 status 都是ready/returncode0；本次查看时未有operator-review.md。ready表示模型报告结束，不表示报告结论已获接受。

注意：clean_complete_cells是新增窗口，不是完整arm分母。09:38窗口不重复列E的w11r01，因此其新增12与batch完整数之和13并不矛盾。

## 逐事件核心处理

### 00:34 窗口

最新 `2026-09-04-status-and-trajectory-review.md` 已覆盖其A/AB/C/control全部结果，保留原ledger/flags并增加人工数据边界审计；后续 `2026-09-04-usage-proposals-and-false-kill.md` 恢复早期负面→修复→再运行序列。

不能采纳旧报告的几个强结论：

- 83/87、79/97、18 unmatched等计数内部不一致；没有把0.954当已确认的全程合规率。最新分析改用版本/事件链并明确delivery≠执行版本合规。
- C“零eligible”、所有yes→无决策作用不成立为总体断言：最新 `negative-followthrough.md`恢复C的no→decode修复，也恢复AB/D负面。保留“原终稿账本没有完整动作”的发现。
- P1建议把scope probe从scorer排除项拿掉、用新口径重算旧baseline，不执行。原scorer和原flag保留；新I/O隔离是前瞻共同runtime。
- K宽度硬阈值、L提高no比例和“等baseline到8才可新启动”均为旧提案，不是现有授权下的依赖。新版明确不以更多no或更窄区间本身作为成功。

0-byte sidecar日志与matcher漏配仍只能算旧运维/审计限制；已有请求、响应、私有transcript和最新综合审计支撑，不构成未完成科学结果或新部署阻塞。未借此宣布所有旧时序疑点均已逐条解决。

### 07:37 窗口

旧报告提出的主要风险已经处理：

- w09r03库调用读取测试题/target并传到scientist：最新status报告和`evidence/2026-09-04-review/manual-audit-flags.json`明确保留；没有认定所有flag良性或retro-clear。新Landlock/seccomp/MCP broker针对实际I/O路径，取代只改regex的K提案。
- w09r04“failed即第二个同类违规”的推论须拆开。`evidence/2026-09-04-usage-and-filtering/latest-e-d-followthrough.md`具体恢复exp07：L0=no→修parent→save smoke→全量运行，修复后未relock；这确有版本对齐问题，但不是永久否决好方案。新immutable request/fingerprint/action/run合同处理此类机制。
- E首cell的8/8终稿yes已被更新为完整4cell的35/35 review L3yes、29/29卡执行完成；无需再启动报告遗漏的专业子分析。
- “仅inflight导致ledger漏窗”的发现被权威completed_attempt.result_dir路径读取覆盖；不能把harvest镜像是否存在等同科学证据是否已分析。

旧L(as-written interval)、M(C5比较完成)、N(提交候选核验)可留为将来独立研究假设；不是此次必须实现的隐藏清单，更不应无声并入新skill或改scorer。新study已明确拆分共同规程、证据规则与联合候选选择。

### 09:38 窗口

后续综合审计纠正了该报告最关键的过度结论：

- F/w12r03/exp04旧checkpoint平台不足以证明新100k/MATH方案无效；F/w12r02/exp04的20k独立短训也不是90k完整方案的已知终点。两次取消成立，是否损失更好模型未知。不能按planned_h−wall_h自动记为正确节省，也不能据此优先扩F至8。
- 最新32条现代WMA轨迹有304次留存请求，全部单card；w10为76次L3yes，E为35次L3yes。终稿112、打印140等旧分母不再用作完整动作普查。
- 0.991是delivery，旧报告末段仍称verdict-before-launch；不采纳该误标。新版以对应proposal/script/config的已返回review和实际run事件为机制读数。
- P1保留旧verdict这一诉求已用不可变请求/回复和动作档案落实；“让旧ledger只评分首稿”、P2改L0 truth、P3重定义节省、P4低flag率过门等具体scorer改法未采纳。原历史口径保留，新counterfactual/action分析独立记录。
- 旧“相关性证明扩大评估集不值”“方差下降已经复现”等推断不作为晋升/实验删除依据；最新训练效率与horizon报告把recipe、评测题量、时间与样本组成分开。

## 新实现与真实部署验收的对应来源

- `doc/spec/2026-09-04-wma-decision-review-redesign.md`：证据适用性；不把旧平台/未验证短训代理当新方案失败；联合候选比较；不可变动作/版本；共同数据边界；保留未观察反事实。
- `doc/wma_iterations/2026-09-04-decision-review-implementation.md`：实现边界及旧scorer/schema保留，诚实说明`awm exp_protocol run`并非禁止scientist任意shell训练的完整容器边界；已导出文本中的污染也非I/O隔离能语义清除。
- `doc/wma_iterations/2026-09-04-opus48-crossbench.md`与`evidence/2026-09-04-crossbench/production-acceptance.json`：旧报告要求的生产边界不是只留在本机测试；job92160/node2记录真实OS canaries、实际joint+single模型调用、仅四个wma_probe工具、冻结public/private SHA31b854bb。此处仅核对验收记录，不读新科学in-flight。

## 保留的非阻塞审计尾项

本次没有把“窗口覆盖”误说成“每条旧建议/疑点全部被证实”。以下仍应随旧窗口保留，若今后用这些细粒度数据做新结论，须先限定/核验：

1. 09:38报告称w12r02/exp06的measurements.n=150与summary n500不一致、exp02的n500 incumbent未记为measurement。未在已读综合报告中找到这个具体修正；不得修改原卡。它是旧中间测量表的待核数据质量标记，不影响已经逐项匹配的官方最终accuracy或新study定义。
2. 09:38报告称w10r06/exp06曾请求test parquet而404、w11r02/exp01从/tmp导入task定义。新I/O隔离处理其访问类别；此有限交接未独立复核这两个历史例子，不能新增confirmed-exposure标签，也不能宣称原所有flag均只是良性读取。
3. D4有一条未完整恢复的早期summary，专项已明确不声称9/9 L3yes。这个缺口不妨碍“已有恢复序列不是永久误杀”的限定结论，也不构成新独立实验的真实数据依赖。
4. 两份F原全量计划仍无反事实补测结果；任何“减少误杀”“节省正确”的科学宣称需要对应补测。当前新实验用于检验新机制，不以旧F已经正确为启动前提。

因此本轮建议给三个事件写完成交接，注明accepted findings、rejected/superseded claims和上述historical limitations；无需重跑同一Opus5窗口、无需等待这些非依赖尾项再推进已通过独立真实验收的新实验。
