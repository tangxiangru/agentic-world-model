**WMA实际被用于什么：候选不足，还是误杀好方案？**

这次重新运行全部30份本线manifest的PTB结果验证，快照时间为 **2026-09-04 10:34:57 UTC**。共79个原PTB validator/judges通过的完成结果，比先前64条多15条：D尾部1、baseline扩展4、control扩展2、E四条、F四条。原人工数据访问标记保持不变；自动通过不等于已确认语义上无越界。本报告研究使用方式与实际行动，未提交/取消/修改实验。

**结论：基线主要被当作已选方案的检查与建议者，几乎没有真正使用多候选排序；但最新F确实出现了建议体导致完整训练取消的实例，其中一例的筛选依据不足以证明新方案无效。实际取消已确认，是否误杀了本来更好的模型仍未测到。**

**1. 实際使用量：不是调用太少，而是每次提交的候选范围窄。**

已完成的R2/R3中，有32条WMA轨迹、223张不同实验卡、304次留存请求。**304/304请求的card_ids长度都是1**；零次将多张完整待选方案放进同一个请求横向排序。不能由此推论WMA不比较任何备选文字，但其结构化ranking在这些调用里最多就是单元素。

| 卡片原family对应的工作 | 卡数 |
|---|---:|
| SFT/RFT训练方案 | 100（70 SFT、30 RFT） |
| 基线读数、评估、打包等other | 71 |
| 权重合并/模型平均 | 31 |
| 解码配置实验 | 21 |
| 总计 | 223 |

这些是工作用途计数，不是完成的训练次数；同卡可能重锁修改，RFT也包含监督拟合，部分SFT卡混入自生成数据。此前R1的原始请求存档覆盖不齐，因此现代304次是主要使用方式分母，未把R1混进来估计总体比例。

| 当前版本 | 完成traj | 不同card | 训练类card | 非训练card | 请求 |
|---|---:|---:|---:|---:|---:|
| v0.2阻塞基线 | 8 | 58 | 23 | 35 | 76 |
| E：L3 evidence | 4 | 29 | 10 | 19 | 35 |
| F：decisive precondition | 4 | 25 | 17 | 8 | 38 |

基线平均每条7.25张卡，其中只有2.875张训练类卡，却调用WMA9.5次。卡里平均约3.8条alternatives_rejected描述；它们有重复，不是额外3.8个完整独立方案。说明scientist有其他想法，但往往已先筛选掉，再把一个选定方案交给WMA。

**2. 原轨迹中，筛方案大多发生在WMA之前。**

- `w10r02/exp-06`：13:45:48 scientist已排除179k数据×1epoch、113k×2epochs，选择83k×2epochs；13:45:53只送exp-06一张卡。WMA给flat预测但L3=yes，随后训练成功。前两种更大方案不是WMA否掉的。
- `w10r04/exp-06`：14:33:42先排除168k从base重训、第二轮RFT和较高LR，选57k低LR继续；14:33:50才提交选定卡。也是scientist先收窄选择。
- `w10r01`末段：14:17:26先放弃第二RFT，14:17:31交WMA的是打包验证卡。没有独立的第二RFT待选卡被WMA拒绝。

WMA可以看到这些被排除方向的文字摘要，但这里没有它在同一决策点评估多份同样完整、同样可执行方案的证据。当前实际流程是：**scientist先选一个 → WMA检查这个方案并给条件/派生建议 → scientist改、跑或放弃。**

因此，用户关于“方案少”的担心，最有证据的表述是：**正式候选池和横向比较机会不足**；不是没有调用、没有idea或完全没有局部作用。[原始proposal筛选链](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/proposal-funnel.md)

**3. 基线低分没有‘大量L3否决好方案’这条证据链。**

旧4条baseline43次L3均yes；新4条33次也全部恢复为yes。合计 **76次yes、0次no/defer**；58张卡最终均完成，没有not_run/abandon_line。新扩展有两次L0=no：一次smoke证实OOM，修复token-budget batching后完整训练；一次补齐checkpoint包装参数后继续评估。它们不是取消好方案。

E四条35次review同样全L3=yes，29张正式卡全部完成，也没有发现建议导致正式卡取消。它原本被设计为增加L3区分度，但当前这些完成轨迹里该行为并未出现。

这不能排除yes裁决中的错误建议、等待、注意力分配或scientist主动取舍造成间接损失。但对这部分数据，不能把最终低分归因为大量明确的no/defer。[baseline扩展逐条恢复](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/baseline-extension-negatives.md)、[E/D尾部核验](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/latest-e-d-followthrough.md)

**4. 最新F改变了答案：终稿yes仍会经建议体筛掉训练。**

F有25份最终verdict，L3全yes，但两张完整训练卡是not_run+abandon_line。必须读实际动作，不能仅用L3枚举统计是否筛选了探索机会。

**F / w12r03 / exp-04：最值得验证的潜在误杀。**

原计划已准备100k行数据：70k GSM8K-family新教师解答＋30k MATH-family，使用新训练调度继续一epoch。卡记剩5.6h、计划1.6h。WMA04:41:22首条建议是先评上一轮checkpoint1600：若与上一轮final的0.7635相差不超过0.012，则认为曲线已平、新100k边际价值小，削弱L2/L3。

Scientist实际评到旧checkpoint0.7665、旧final0.7635，04:49:51说据此转hard-example mining，04:52:22把原100k训练写成未运行、放弃。L3原文仍是yes@.55。

这里存在两个推断跨越：旧数据与旧学习率调度的尾段趋平，不足以说明加入MATH与新调度也无效；WMA的“weakens”又被scientist升级为“falsified this card's premise”。这是明确的筛选判据风险，而不是只是缺时间。原100k方案后来也没有补跑。

后续替代路线RFT为full1319 .7748，再一轮RFT .7544被拒，最终官方 .7718。这些结果不构成原方案的反事实上下界，不能据此认定原100k更好或更差。

**F / w12r02 / exp-04：有短程实测支持，但全量仍未知。**

原计划90k行（60k Orca＋30k anchor）、约1.45h。WMA要求先20k行/约150steps，再看watch400，低于.82则停止。Scientist实际跑159steps、watch400由incumbent .835降至.790，于是取消全量90k。

这比前一例有更直接的新方案证据：拟议mixture的短程训练确实退化。但20k独立一epoch的调度不等于90k完整训练的前22%，短程负值仍不能证明全量终点必然差。

**因此：已确认两次WMA建议相关的完整训练取消；尚未确认其中哪次压掉了本可更好的模型，也未量出由此损失的分数。** “可能误杀”已经有具体可检验案例；“已经误杀并造成整体低分”仍缺反事实结果。[F完整前后链与原始行号](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/F-negative-actions.md)

**5. 显式负面裁决不应与永久否决混在一起。**

旧R2已恢复的8次L3负面（7defer、1no，涉及7张卡），后续均有修订后执行。AB/w07r03一次把“110k未见GSM问题”纠正为含新MATH的问题组合；原数据方向没有照原样跑，但修后卡完成，旧计划反事实未知。C的greedy no修正temperature后继续，D的defer主要修保存/加载后继续。

新F/w12r03/exp-03的显式defer也因真实GenerationConfig保存故障；修复并smoke后继续完整训练，full1319 .7096→.7635。不能拿修后成功反过来说初次裁决误杀。

历史w01r07的确有错误L0/L1预判，但训练已经启动并正常完成；w04r04的no则促成确认保存问题、中止、修复、重新运行。两者也不是已证明永久压掉好方案。[旧负面裁决后续](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/negative-followthrough.md)、[历史两例](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/historical-negatives.md)

**对两个担心的最终判断**

| 问题 | 当前轨迹证据 |
|---|---|
| Scientist正式候选范围是否太窄？ | 有直接证据：现代304次请求全部单卡；不少备选在WMA之前已被排除，排序能力几乎未以多候选方式使用 |
| WMA是否没有被使用？ | 不符合记录：基线8条有76次review，大量用于运行、产物、评测、合并和现有方案的局部修正 |
| 基线是否因大量L3否决而低分？ | 不支持：76次L3均yes、58张卡均完成 |
| WMA是否会实际压缩探索机会？ | 会：新F两份主训练因precondition检查结果被取消；A也有yes之后因数据重合事实而由scientist放弃的案例 |
| 是否已经证明错杀正确方案、降低分数？ | 尚未：被取消原方案没有完整结果。F/旧平台→新mixture这条推断风险最具体 |

下一步最有区分力的证据不是继续数no：一方面记录同一时刻真实待选的完整方案集合及谁排除了哪个；另一方面对这两份被筛掉的F原方案做冻结parent、数据、脚本与schedule的独立反事实核验。若补测，比较应对齐当时的incumbent和预算，不能拿不同时间的最终traj分数代替原方案的价值。本次仅完成分析，没有执行这些训练。
