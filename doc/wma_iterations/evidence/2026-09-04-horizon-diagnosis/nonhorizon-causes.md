# 对“WMA没有整体收益只是因为完整训练次数少”的轨迹审查

范围：同协议 w10r01..04 的已完成轨迹为主要证据，辅助使用已核实的 C/D 修复案例和同轨迹反例。这里的“没有整体收益”只表示当前比较尚未建立稳定的总体收益，不表示WMA因果效应已被证明为零。数量与分数有关，也不自动说明增加训练次数就能提高WMA的相对收益。

结果根目录通过 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-sft-efficiency/input-complete-cells.json` 的 result_dir 解析。便于短引用，B指：
`/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2`

B下`solve_parsed.txt.gz:L`为解压后的原始行号；相同内容的未压缩trace在frozen complete-cells.json中对应result_dir的`solve_parsed.txt`。所有card位于`B/CELL/task/memory/cards/`。

## 先固定与解释直接相关的数量

- w10共30张card、43次review requests/responses。回收42条打印level summary，另一个single-review card有最终verdict；43个L3均yes。不能只看30个最终文件而遗漏relock。
- 完整监督训练次数是**3、3、3、4，总13、均3.25**，不是3 each。对应卡：w10r01 exp02/03/04；w10r02 exp02/03/06；w10r03 exp02/04/05；w10r04 exp02/03/04/06。RFT的监督fit计入，sampling不计入；多epoch一次训练只计一次。
- 43个request的生命周期合计每cell80.11、58.76、69.58、59.66分钟，均67.03分钟。它包含relock与等待/执行开销，不是纯GPU idle；也不能把它全部视作可兑换成更多训练的预算。来源：`doc/wma_iterations/evidence/2026-09-03-w10/review-cycles.json`。

## 1. 最直接的瓶颈：有可执行、有余时的建议，也没有进入行动

**观察。**w10r03/exp05：12:52:33 WMA要求先用约10分钟评估已经在盘上的exp04 checkpoint1040/2080，并提出把同一130k行的fewshot比例3%改到30%（trace:9533、9539）。同一输出显示尚余**4小时32分**（:9541–9542）。17秒后scientist直接启动原来的sft_v6、原parent final的1.44h continuation（:9554），没有先完成该checkpoint比较、也没有重建3%比例的数据。

w10r02/exp06同样：13:54:44 scientist读到“先花6分钟评价尚未在n500上评分的旧final checkpoints”（:10392），13:54:52直接启动2.07h训练（:10408）。卡记录remaining_h3.5（:10122）。这不是最后一分钟来不及读建议。

**边界。**这证明了建议到行动的断点，不证明被跳过的checkpoint或30%fewshot一定更好。w10r02随后新训练还得到.738@n500，高于当时incumbent soup .710；不能把未听从便宜建议一律判成坏决定。

**对更多fits的含义（推断）。**当已有建议在4.5h余时下仍被跳过，额外延长时间或增加“可以训练的轮数”不会自动让建议被采纳。更多fits可能增加候选，但不能替代“读到建议后究竟改了什么”的问题。当前总体收益缺失不宜只归结为观察窗口短。

## 2. WMA最确定的作用集中在纠正实施错误，收益并不必然变成更高最终分数

**观察。**三个已逐条核实的案例详见 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/empirical-repairs.md`：

- w08r03/exp03：scientist已决定测试greedy，WMA才发现do_sample=false在当前vLLM路径中没有使其greedy。scientist明确承认、补temperature0、relock再评估；同权重sampled .5533→正确greedy .6533@n150。WMA贡献是纠正实验实现，不是从零提出greedy方向。
- w08r02/exp06：包装日志被grep误读为成功，实际checkpoint900已删除。WMA读盘指出缺weights，scientist换1200、看到2shards/8.6GB，三arm评估顺利完成；替换的checkpoint只有.6801，未赢incumbent。
- w09r01/exp06：WMA发现磁盘key与模型load_state_dict的0/883匹配错误。scientist改写平均模型、实际CPU reload OK；soup n500=.774，仍低于incumbent .790。

**推断。**WMA确实有局部作用，但局部作用的目标常是“真正执行计划并取得可信结果”。如果修好的候选本身没有更强能力，这项作用会表现为少走一次故障恢复，而非最终accuracy增加。更多fits提供更多遇到这类错误的机会，也可能稀释单次修复相对10h总成绩的影响；不能从次数本身预知净效应方向。

## 3. 初次训练的recipe风险被看见，但review的大部分落点是可运行/可评估与checkpoint，而不是挑选更优recipe

**对4张首次训练exp02最终verdict逐张复核。**

| cell | 首要precondition | 对recipe/训练分布的实际内容 | 首次L2 delta区间 |
|---|---|---|---|
| w10r01 | 训练后补processor文件并CPU load | 明确指出0-shot训练/10-shot评分；要求step600做with/without-prefix探测；建议1epoch先读 | [.35,.70] |
| w10r02 | 前90秒错误监测 | prior review已促使fewshot calculator annotations修正；当前重点是decode配置、checkpoint配置、诊断与复合delta说明 | [.40,.72] |
| w10r03 | bs风险须看至step160 | 指出93.6%训练0-shot而评分100%10-shot；要求render检查，建议同权重sampled/greedy对比 | [.32,.70] |
| w10r04 | 用正确parent重新做save smoke | 指出只有3% exact10shot；明确建议提高fewshot-frac或固定k10，但放在第三条cheaper_variant | [.30,.68] |

来源：四张`exp-02.verdict.json`的suggestions及levels。四张首要precondition都是具体运行/产物问题，**不是说所有建议都泛泛而谈**。其中确有recipe分布诊断；不能声称WMA完全不看recipe质量。

**最清楚的recipe轨迹。**w10r01/exp02最终checkpoint sweep：0-shot下step1800和3546都.740；10shot下分别.680和.540。第一轮两epoch中的后半段加剧了被评分prompt下的问题，而非增加通用算术能力。WMA事前已建议step600 probe；实际card说明GPU占72GB，未能边训练边跑该vLLM探测，改在训练后做。随后exp03针对prefix继续训练0.6h，step200的with/without-prefix都.7433，原6pp差消失；最终n150由.7067到.720。来源：`w10r01/exp-02.yaml/result.diagnostic_result`和result.notes；`exp-03.yaml/result.diagnostic_result`。

**推断。**这条结果支持“做对一个目标明确的fit有用”，不支持“多做同一种fit总会有用”。若新增fits继续重复不匹配的prompt分布，更多训练也可能重现甚至放大该退化。反之，更多预算允许修复后的独立full retrain，这是尚未被这些轨迹直接观测到的可能收益。

## 4. 更多训练不是单调收益：现有同协议w10里，后续fit已出现增益、无增益与小样本误判三种情况

**观察A：额外训练有真实正向读数。**w10r02/exp06将原小数据recipe从1epoch扩成2epoch，2.07h后.738@n500，比incumbent .710高2.8pp。卡写delta+.050是对旧checkpoint650的.688，不能把+.050当作超过incumbent的幅度。该例确实支持训练预算还有可能买到改善，但没有说明WMA是这次选择的唯一原因。

**观察B：第四个fit没有提高成绩。**w10r04/exp06在已有3个fit后又做57k fresh-chain continuation，.72h，full1319为939正确，对incumbent941正确，-0.15pp；dev250 .744→.708。WMA事前给flat[-.012,.018]并yes，保留incumbent。已观察的第四次fit不能作为“次数越多就越好”的例子。

**观察C：选错看法可以抹掉已训练出的候选。**w10r03/exp05初读n150：.7267<.7400，被记录为reject；exp07扩大为同一n400后：.735>.705，最终改为提交exp05。这一改选需要的是辨清候选，而不是再多训练一个。原始卡仍保存最初结论，并用superseded_by保留反转，不能只读第一次reject。

**推断。**平均次数少限制了探索广度，用户这个假说合理；但这里看不到一个足以替代recipe、执行与选择问题的单一“次数不足”解释。实际边际fit质量差异很大，只有计数无法区分在解新问题还是重复低收益continuation。

## 5. 已有候选的保留、选取和组合还有未兑现的收益，新增fit不会自动修好它们

**观察。**四个w10 cell都有一些真实checkpoint保存/评估行动，不是完全没有uptake。但w10r02/exp03的save_total_limit2删掉450/900，只留下1350/1757；想看的中期点没有保住（card training_summary；trace:8622）。w10r04/exp06最终仅保留final；WMA的cheaper_variants本来提出midpoint和一个约5分钟full read，但scientist只打印了preconditions（:11034），直接启动不带step-save的命令（:11048），当时仍余2:59（:11021–11022）。这比笼统“没时间”更具体：有建议没有进入阅读/执行。

组合也不是一律有用。w10r02/exp04确实执行WMA建议的two-way subset：four-way soup .694，two-way .710，best ingredient .704，n500；trace:9233、9299–9300、9392、9626–9639。改善被观察到，但+.006超过best ingredient仍很小。w10r03/exp06把.6933与.7400的两个训练endpoint平均，得到.6933@n150；WMA的L2[-.06,.02]已经预示可能无收益，但继续yes。

**推断。**新增fullfit可能提供更强或互补的ingredient，却不能保证现有选择/组合方式会保留它。先前已有6pp级checkpoint差和prefix排名反转，说明最终提交模型取决于如何读取已有产物。用更多次数解释最终分数时，要留出这一条独立机制，而不是假设每个已训练候选都被正确选择。

## 6. L2 与 L3 的输出常没有形成有区分度的资源分配

**观察。**w10四张first-fit最终L2宽32–38pp，对n150的0.03账本floor约10.7–12.7倍；这与全终态ledger width/floor均值4.1145不是同一个子集统计。它们能表达“firstfit大概率大涨”，但没有展示在若干具体recipe之间选择的能力。w10全部43次review L3=yes，包括两次先前L0/L1否定后被修好的链。w10r04/exp06明说flat、w10r03/exp06 soup也偏flat，仍都yes，因为incumbent保护且剩余预算允许。

D的27次历史review则有22yes、5defer，修后最终20个verdict又全yes；C的greedy no也因修配置变yes。因此不能用终态all-yes说WMA从未阻止原计划，不能用修后成功反过来把原no算成误杀。

**推断。**当前WMA更多是在回答“这个可修复、fallback安全的尝试可以做吗”，而不是稳定改变“未来几小时优先做哪个训练”。增加fullfit数量有可能让这种局部校验积累收益，但不会从逻辑上自动把all-yes变成有用的调度区分。宽interval是否应缩窄也不能仅凭这点决定；暂无轨迹证明较窄区间会让scientist换一个更好recipe。

## 7. 有些建议本身不够可靠，scientist不采纳不总是瓶颈

w08r02/exp04是必要反例：WMA据启发式建议删除4887条“答案数字也出现在题面”的orca行。Scientist做随机14条手检，见到25km/5km/h=5h等合法重现数字的样本，拒绝过滤。实际未执行删除，所以没有已发生的过滤致害；14条也不能证明全部4887都正确。此例说明采纳率本身不能代表价值，“更多轮次让agent更听WMA”也不是由轨迹支持的解释。精确前后证据见 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/empirical-repairs.md` 最后一节，C2 trace:10550–10551、10563–10594、10642、10950。

## 与“多给几次完整训练”假说的关系

**确有时间边界证据：**w10r04/exp06的alternatives_rejected说168k rows从base完整重训约2.5h，而只剩3.1h，担心评估/打包来不及，选择了短continuation；w10r03/exp05放弃更大combined-corpus从base重训以及RFT sampling+fit，也明确引用剩余时长。这说明更长窗口可能改变recipe集合，不能排除。

**尚未证明的部分：**没有同一scientist、同一已冻结策略在相同状态下得到更多预算的实际对照，因此不知道它会把新增预算花在这些被拒绝的recipe上，还是更多相似continuation、重复读数或整理记录。已有4.5h余时下跳过10分钟checkpoint read，说明至少一部分瓶颈不依赖总时间。

证据排序是：①具体建议到行动的断点与真实产物/配置问题，证据直接；②recipe分布及候选选择的影响，多条实际before/after；③WMA区间/yes未体现资源优先级，输出分布明确但对最终分数的因果作用仍属解释；④短窗口限制WMA长期累积收益，合理、被部分时间拒绝理由支持，但还未被直接比较证实。不能把第④项作为前③项的替代，也不能因当前没建立整体收益就断言延长窗口无效。
