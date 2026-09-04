**WMA修改建议：可靠的筛选依据与真正的候选比较**

状态：下述证据规则、联合候选比较、不可变动作记录和探针 I/O 隔离已在当前工作区实现，通过本地回归检查；尚未部署、晋升或提交新实验。历史 verdict schema、scorer 和既有冻结 manifest 保留原口径。预算优化、增量重审、动态 ML 探针和更长训练窗口仍是后续工作。实现与验收范围见[实施记录](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-04-decision-review-implementation.md)。

依据是2026-09-04 10:34:57 UTC冻结的79个完成结果及逐事件分析，主要报告：[实际使用与潜在误杀](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-04-usage-proposals-and-false-kill.md)、[训练深度与预算](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/2026-09-04-why-no-net-gain-and-horizon.md)。不读取held-out失败细节来拟合政策，不以不完整或运行中的结果晋升。

**建议优先级**

1. 修正WMA筛选建议的证据适用范围，连同互相矛盾的示例。这是最小的skill政策改动。
2. 将WMA接入scientist最终选择之前，联合比较2–3个真实候选简案。这是单独的接口/规程改动。
3. 保存每条建议引起的真实动作、版本与成本，使停止收益和误杀风险可审计。这是独立测量基础设施。
4. 在上述机制能够被测量后，再独立优化预算使用、重复审查和确定性preflight，最后验证更长训练反馈窗口。

更多no、更高采纳率、更多SFT次数均不单独作为成功标准。主要看同预算的最终结果、真实候选选择和拒绝方案的反事实质量。

**改动一：证据只用于它实际检验过的命题。**

直接触发案例：F/w12r03/exp04用旧checkpoint尾段持平筛掉新100k、30%MATH混合训练；F/w12r02/exp04用独立20k短训练的退化筛掉90k完整训练。实际取消已确认，完整方案本来会怎样尚未观测。

修改前的文件还有明确矛盾：`skills/wma/SKILL.md`指出C3/C4的终点效果不能由smoke推断；`skills/wma/verdict.example.json`却示范“150steps、600rows，dev150不涨3pp就stop”。此次已同步修正示例，避免只在规则中写原则、在示范中继续教相反行为。不能由此认定示例已经被证明造成F行为，但一致性问题本身明确。

建议作为同一项政策改动加入`skills/wma/SKILL.md`，并同步调整manual与example中对应措辞：

> Before a recommendation would stop, replace, or deprioritize the proposed run, state the exact claim tested by the evidence and its applicability to this proposal's parent, data, objective, schedule, and evaluator. Separate a directly observed implementation defect from an uncertain forecast of final model quality.
>
> A plateau among checkpoints from an earlier training run does not falsify a proposal that changes the data distribution, objective, or training schedule. A short run with an independently shortened schedule is screening evidence, not the observed outcome of the proposed full run, unless a relevant surrogate relationship has been validated.
>
> You may prefer another proposal under uncertainty. State the competing proposal and the opportunity-cost reason; leave the unexecuted proposal's outcome unknown. Do not turn a weak proxy into an arbitrary pass/fail claim that the scientific hypothesis has been disproved.

示例中的短训建议可改为：

> [tier 2] Use a scratch smoke to verify finite loss, memory and save/load behavior. Failure may require a repair before the formal run; passing the smoke does not establish the recipe's final score. A separate short-run score is preliminary evidence, and an unvalidated small gain threshold is not a full-run failure criterion.

同时限定手册中的历史先验：C3的单个近零消融不能代表所有新数据训练收益近零；C5历史上出现过大收益，不等于当前未评checkpoint一定比新数据更值得。这里先修证据被怎样使用，不凭这批结果随意重填一组更乐观的数字。

**期望变化**：在w12r03式情境中，WMA可以建议把更好的旧checkpoint作为incumbent，也可以根据预算偏好RFT；但须说新MATH方案尚未验证，不能把旧训练平台当成新方案已失败。对于真实OOM、缺权重、save/load错误，保留修复后再运行的能力。不是要求所有不确定方案都必须跑。

**验证**：相同冻结输入、模型、预算和运行机制下比较有/无此规则；可用具有F式筛选机会的固定参考版本做修正对照，但不自动把F晋升为baseline。逐条检查跨数据/调度外推、探针触发的取消理由、真正修复的blocker是否保留。使用启动前可证明的快照，不能把当前终态工作区当无泄漏回放。未运行方案没有终点真值；若要证明误杀减少，需要事先选定样本补测原方案，不能只因取消变少就宣布成功。

**改动二：从单卡检查扩展到选择前的联合比较。**

直接依据：现代32条WMA轨迹304次留存请求全部单卡。Scientist卡里虽有约4条被排除备选摘要，但大多在WMA之前已经选定训练方向。WMA很少获得比较同等完整待选方案的机会。

在下一笔主要训练预算的决策点，scientist先准备通常2–3个真实可行简案。已有明确可行的一个方案时允许只有一个并说明原因；不要为了凑数写同义改写或虚构方案。只有在它是真实选项时才加入“维持当前模型/结束”；不要求每次打包、修配置都生成多份候选。

候选共用一份现状：当前模型、前轮测量、资源、剩余预算与证据索引。每份简案至少包含：

- parent checkpoint及来源；
- 数据/目标/训练调度的具体变化；
- 它在检验什么，依据是什么、未知什么；
- 训练和评估成本，区分实测估计与猜测；
- 什么观测会改变后续选择；
- 已有代码、数据版本或可执行命令依据；尚未准备好的依赖如实标注。

这些简案不要求先准备三套完整数据、启动三次训练或填三张长卡。Scientist在读WMA前记录自己的首选，再让WMA一次共同上下文比较。WMA返回优先顺序、每个相邻选择的理由、不确定性、关键验证和预算取舍；scientist保留最后选择权。

现有多card接口不足以实现这一点：`sidecar.py`逐卡独立调用review，再按yes/defer/no和confidence排序。confidence是判断的把握，不是预期效用。只把card_ids从1个改成3个，可能增加成本，却没有真正联合比较。

需要独立的候选比较入口与记录合同；这是新的harness/规程处理，不伪装成原冻结prompt里的一句小改动。当前单卡review与其历史scorer保持可追溯；新选择阶段建立自己的基线。Scientist生成候选，WMA不因此获得自由提出无关方向或执行实验的权限。

**成本与启动边界**：初版每个重要决策只加一次有界联合比较，共享证据只读一次；不对每候选分别调用一整次WMA。被选中方案仍走正式card、preflight及现有阻塞lock审查，返回前不启动。先把这一次额外调用的成本完整计入；不要同时取消正式审查或引入未经验证的旧裁决复用。将来能否合并调用/增量重审，另做实验，必须验证proposal与证据hash仍匹配。

**验证**：无WMA与有联合比较的两组都要求同样候选简案、相同信息和总预算。对照由scientist自己选，实验组看WMA后选；保存读WMA之前的选择。否则仅仅“强迫scientist多想几个方案”的收益会被算给WMA。可保留当前单卡流程作为第三组评估整体流程的净收益。不能把候选数量增加、本次选中项变化或更高confidence直接当最终效用。

**改动三：以实际动作记录作用，保存不可变版本。**

直接依据：F的两次取消都藏在L3=yes的建议体里；D终稿全yes覆盖了早期defer；还有改完command却未在执行前重新取得对应裁决的案例。当前ledger只看最终文件及adopt/reject，无法可靠区分修复、取消和误杀。

每个decision保存：候选集合版本、scientist先前选择、WMA输出、scientist最后选择与理由。每个request保存完整verdict、proposal/script/config/evaluation指纹、成本及时间。每条会改变执行的建议记录：采纳/拒绝/修改后采纳，实际做了哪个probe，是否真的重训或关闭计划，发生时间及产物。

记录义务放在决定/启动前，结果完成后补实际观察；不能等close时才凭记忆改写理由。记录拒绝WMA的理由，不要求服从WMA。真正执行的版本必须对应已返回的review；变化需要重审，旧记录保留。

测量上把“未运行”保留为未观察终点，不能因abandon_line自动认为WMA正确预测了坏结果。新增动作/反事实分析与旧冻结ledger并列，单独登记测量改动，不把旧分数静默洗成新口径。

**基础修复与后续顺序**

数据边界是所有新行为实验的共同前提：已证实有库调用读取测试答案且原扫描未报警；不能只补正则或只靠G的文字规则。WMA的探针需使用限制实际I/O的环境与许可数据；评估走可信接口返回许可摘要。用合成canary验证直接路径、库调用、缓存及网络入口。原始flag不清除，共同修复不当作skill能力提升。

预算使用另立改动：用已测吞吐/评估耗时更新估计；在提前结束时记录可容纳的真实备选与为什么不做，保留已验证incumbent。不要把“至少跑6次SFT”直接写成目标，也不把新增时限当已证明有效。

成熟的确定性检查逐步交给preflight：有效解码参数、完整权重文件、save/load round-trip和实际checkpoint保留点。Probe smoke不保证正式训练不会OOM或最终保存一定成功，覆盖范围应明确。这样能把WMA注意力留给有不确定性的方案比较，减少每次重新阅读同样的库代码。

**建议实施与评估顺序**

1. 先修证据适用范围及矛盾示例，保留旧接口，验证停止判据改变及真实blocker检查是否仍有效。
2. 把数据边界与不可变动作记录做好，作为两边一致的基础设施；不重新解释旧flag来宣布通过。
3. 多候选比较作为独立接口实验，用同样候选简案的scientist自选作对照。一个比较调用，保留正式阻塞review，费用与等待都报告。
4. 只在真实选择收益有迹象后，再独立测增量重审、预算使用和更长反馈窗口。每项单独记录、单独比较；最多晋升一项，不事后合并候选成绩。

首要终点是同预算的最终官方成绩；机制读数是是否选中更好的方案、已取消方案补测是否显示机会损失、建议是否形成真实动作。训练次数、no比例、采纳率、review次数只用于解释。费用、等待和同尺评价质量一起报告；每setting四次重复用于机制筛查，晋升仍需原定独立确认与held-out门槛。
