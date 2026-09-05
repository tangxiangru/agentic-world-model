# WMA evolve：轨迹、负面建议、输入边界与协议审计

冻结时间：2026-09-04 05:52:48 UTC；输入是 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/results.snapshot.json` 中 validator-complete 的 receipt-backed WMA 批次。不读运行中候选结果；没有重新评分、清除 flag、改 runtime、提交或取消任务。PTB validator/judge clean 不等于 WMA 输入语义 clean。本报告的终态计数由父分析复核；轨迹判断来自原始 card、lock、processed/response、scientist trace 和 WMA private transcript。

## 最重要的新发现：已有的零 flag 不能证明没有测试标签暴露

**w09r03 / exp-01 存在明确的 GSM8K test 问题和 gold target 读取，原始 `access.outside=[]`，`leak_suspected` 缺省。**这不同于旧 w10 审计发现的磁盘元数据或已安装包源码读取。

证据目录（下称 D3）：
`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r03_formal_r9/gsm8k_google_gemma-3-4b-pt_91026`

- `D3/wma_private/exp-01.transcript.jsonl:85`：Bash 在 scratch 导入 `inspect_evals.gsm8k`，调用 `task = g.gsm8k(); ds = task.dataset; s = ds[0]`，随后打印 `str(s.input)[:200]` 和 `s.target`。这是主动读取真实样本及答案，超出了导入检查和 schema 检查。
- 同一 transcript `:88` 的实际工具返回：`Loading dataset openai/gsm8k from Hugging Face...`，train 7473、test 1319；`dataset_ok n=1319`；Janet 鸭蛋问题的前 200 字符和 `target: '18'`。因此不是仅由 WMA 自述推断出来的访问。日志显示数据加载/缓存保存，但未保留足够信息确定实际缓存文件位置，不据此声称特定路径的 outside write。
- `D3/task/memory/cards/exp-01.verdict.json` 的 `evidence[e12].note`、`probes[p2].result` 都写入 `1319 test rows, first target '18'`；p2 标记 `changed:L0`。
- `D3/solve_parsed.txt:3459` 和 `:3485` 原样包含这段已交付 verdict；**测试 target 确实通过 WMA verdict 进入 scientist 可见上下文**。没有证据显示完整问题片段通过 verdict 交付，也没有证据显示该标签被加入训练数据、用于定向训练或提高了最后得分。
- `D3/runtime_provenance.json` 明确绑定批次、w09r03、job 91026、Gemma revision；manifest 为 `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2.yaml`，receipt 由 snapshot sources 中该批次条目解析。

边界：WMA skill 的 `What you read` 限制 session + history；`Out of scope` 禁止 “Reading the card's own result, or any held-out data”。`awm/wma/review.py:84` 同样限制输入并禁止找当前 card 结果。此处确定的是**benchmark test 内容/答案暴露**，不是读取该模型当前卡的未来结果；也不是 WMA 离线 Gemma held-out 卡集或 AIME promotion 集被读取。这些概念不能混用。

当前 `awm/wma/backends.py:124` / `:138` / `:238` 的检查依赖命令中的路径 token、运行时存在性与词法路径包含关系；库调用隐式下载/读取数据没有显式 test 文件路径，因此可能完全绕过 flag。scratch 只约束执行位置，不证明输入来源合规。一个有针对性的关键词检索在本次 104 个新候选 terminal private transcripts 中只命中这一张的两条 `gsm8k()` 命令；这不是完整无遗漏访问审计。

处理建议：**保留 frozen ledger 原样，同时新增独立语义访问审计状态**，在 promotion 前处理此问题。优先为 sidecar 提供冻结的无标签 evaluator schema/template 信息、使用合成样本做模板检查，并禁止通过 library/network 取 benchmark test payload。若引入 OS/network isolation 或事件级输入记录，是独立的 harness/measurement 变更，需要另立 spec 和 baseline；不能把已有 flag 清零伪装成 skill 进步。G probe-scope 已经排队，但本次证据并不能证明 G 会防住库调用路径。

## 第二个关键事实：终态全 yes 掩盖了有用的 defer/no 与失败重试

原始汇总见 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/harm-review-summary.json`。

D 三个完成 cell（w09r01..03）有 **27 requests / 27 responses / 27 printed level summaries**：22 个 L3 yes、5 个 defer。最终只保留 **20 个 terminal verdict，全 yes**。5 次 defer 位于：w09r01 exp-04、exp-06；w09r02 exp-02、exp-04；w09r03 exp-04。D 的行为机制已经发生，不能根据 terminal all-yes 宣布 D 没有起效；但它主要使 scientist 修补计划再 relock，没有证明节约了 GPU hours，也没有证明总体 PTB 提升。

C 中，w08r02 exp-06 初判 L0=no/L1=no，w08r03 exp-03 初判 L3=no，随后修复并得到 yes。C 也有终态 card 内记录的早期失败：w08r01 exp-01 的 vLLM 显存残留启动失败、exp-05 的缺失 Gemma processor 文件与 vLLM 进程释放竞争；w08r04 exp-04 的 pyc 损坏、exp-05 的 Liger METADATA 损坏。**终态 L0/L1 100% 且 recall denominator 为空，不等于不存在失败，也不能证明 C 降低了 false no。**之前的 verdict 不能直接与后来修好的 proposal 结果配对评分。

## 八条可复核轨迹

除上面的 D3，以下根目录来自 snapshot 的 `completed_attempt.result_dir`（简称只是本报告里的路径缩写）：

- C2 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r02_formal_r8/gsm8k_google_gemma-3-4b-pt_91021`
- C3 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r03_formal_r8/gsm8k_google_gemma-3-4b-pt_91022`
- D1 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024`
- D2 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r02_formal_r9/gsm8k_google_gemma-3-4b-pt_91025`
- B = `/home/robtang_google_com/gangda_workspace/agentic-world-model/results/ptb/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2`

1. **C2 / exp-06：真实 prelaunch blocker 修复，但 L3 条件不一致。**`solve_parsed.txt:12214` 初判 L0=no@.75、L1=no@.70、L3=yes@.60：checkpoint-900 已被 rotation 删除，包装目录只有 tokenizer，没有 config/weights。`task/memory/cards/exp-06.lock.json` 的 relocked_from 理由确认 scientist 漏读了包装 assert，按 WMA 改用尚存的 checkpoint-1200，并让一个评估 arm 失败不会中断另两个。修复后完成三候选评估，soup234=.7058、soup37=.7013、checkpoint1200=.6801（n1319）；paired p=.520 未满足预先标准，未立刻覆盖 incumbent。结论：帮助修复是真的，“避免了多少小时”不可测；L3 yes 应写清是 as-written 还是 conditional-on-repair。
2. **C3 / exp-03：no 不是错误否决。**`solve_parsed.txt:6257` 初判 L3=no@.70、L2 lower [-.12,.02]，因为所谓 greedy artifact 只改 do_sample，vLLM 实際仍 sampling、甚至移除了 top-k/p 限制。lock relock 理由确认加入 temperature=0 并修补构建脚本。新方案最终 .5533→.6533，n150，0.08h，27 fixes/12 regressions。不能用修好后的 +10pp 把最初 no 计为 wrongly-killed；原始 no 针对另一份 artifact。
3. **D1 / exp-04：defer 驱动 checkpoint 保留；没有实际杀掉有价值训练。**`solve_parsed.txt:6575` defer；`:6630` scientist 说明 relock；`:6698` 修改 lr 回原值、save1490/limit4，明确记录不同意经济上的 defer。终态训练2.45h，143k rows，final=.7533 vs comparator=.7267；middle2980=.6933。checkpoint action 有发生但没选中更好的中期点；主训练的小增益 n150 仍不确定。不能推断“D 已避免浪费”，也不能把 scientist 仍做训练解释为 WMA 完全没被采纳。
4. **D1 / exp-06：merge 路径真实代码 bug，被修复；soup 收益依然没有保证。**`solve_parsed.txt:9011` L0=no@.85/L3=defer；`:9166` relock 理由：模型 live keys 与磁盘 keys 0/883 匹配，原 load_state_dict 会失败。改为原磁盘 key 写平均 tensor，再 CPU reload。训练成本0；card wall .35h；soup n500=.774 vs incumbent=.790，18 fixes/26 regressions；拒绝并保持 incumbent。WMA 在可运行性上的帮助与 soup 的价值必须分别评价。
5. **D2 / exp-04：强推 checkpoint 不足以保证可保存，更没保证真的评估。**`:7457` defer；`:7469` 首要建议加保存 cadence；`:7509` scientist relock 加 save60/limit4。实际第一次跑到step60，parent greedy GenerationConfig 导致保存失败。第二次 relock 清理 config，最终完成 .35h，.7533→.7667@n150；card training_summary 记录失败加无界轮询合计约1.6h的预算损失，最终仅评分 final，未评估保存的中间点。不是证明 D 造成全部1.6h：不加中间保存时 final save 也有同一 bug；实际揭示缺少 save/load smoke 与失败监测，机械满足 checkpoint 条款不够。
6. **D3 / exp-04：保留 checkpoint 是诊断工具，不是性能保证。**`:6753` defer；lock relock 改 save1484→370。但 save_total_limit3 最终删掉370/740，仅留1110/1480/1484（1480与1484只差4步）。final和1110都是.6933 vs incumbent .7467（-5.34pp），.49h；incumbent保存。card 将RFT选“最短正确链”对应的输出缩短133→120 tokens作为机制怀疑。可支持下一轮比较随机正确链 vs shortest，而不能从单轨迹宣布 RFT 整体无效。该 cell 同时存在上述 exp01 测试标签暴露，机制阅读可保留但不能当无污染promotion证据。
7. **B/w10r01 / exp-02：OOM 警告引发真实修改；避免 OOM 是反事实。**gzip解压后的 `solve_parsed.txt.gz:6485` 初判L0=no；lock记录bs16→8、accum2→4，保留有效batch32，并加入checkpoint cadence；修后得到yes再启动。最终1.79h、.0600→.7067@n150。卡作者断言bs16会超过80GB，但最终 reviewer也承认旧smoke可能已覆盖长序列。因此只能说降低了风险并改变了行为，不把未发生的 OOM 当成已观察节省。
8. **B/w10r03 / exp-05：小样本误拒绝是比“WMA太乐观”更有力的解释。**首次n150=.7267 vs .7400，花1.44h，看似继续训练无收益；`exp-05.yaml/conclusion.superseded_by` 明确：exp07 的n400变成.735 vs .705，撤销原先选择，exp05最后被提交。WMA interval[-.02,.05]覆盖首次结果。该例不能列为“已观察的浪费训练”，更不能以此给泛化的L3 stop规则背书。

## 另一个实质协议问题：w08r01 的 failed 最终 lock 是事后补锁

根目录：
`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r01_formal_r8/gsm8k_google_gemma-3-4b-pt_91014`

- 第一 request `20260903T155413.317894Z-ee060062` 的 response 在16:01:09 completed；终态 verdict issued_at相同。
- `solve_parsed.txt:3410` 开始读 verdict，`:3418` 时间16:02:44；`:3684`–`:3715` 在16:06更改concurrency16→2、日志env、seed描述，未重新lock；`:4097` 在16:11:44启动变更后的 command；`:4540` 在16:15:18清空遗留GPU进程后重跑。
- `:5650` 在16:37:17开始写result；request `20260903T163745.601795Z-2cfb31d7` 才补relock；`:5835` sidecar因已有结果而拒绝posthoc review。

所以不能把该failed简单归为“没有收到任何verdict就启动”；真实问题是**收到verdict后改动了proposal，却没有在launch前绑定新hash并重新取得verdict**。终态 lock 的“delivered比率”与实际fence机制不是同一个指标。改进应记录每个immutable proposal hash的request/verdict/launch/result事件链，并在启动时验hash，禁止事后补锁伪装成prelaunch provenance。此处未重新运行或清洗评分。

## 时长、flags与改进优先次序

每request生命周期=`response.completed_at - processed.created_at`，包含relock。w10每cell合计80.11/58.76/69.58/59.66min；D三cell55.39/53.98/52.83min；C为49.18/91.04/68.27/60.00min。**这些不是测量到的GPU idle时间**：scientist会并行准备或运行独立工作；terminal-verdict cost又漏掉已覆盖的旧review。不能把WMA自报cpu_min、request时间总和与纯GPU浪费互换；C2的91min也不自动构成“GPU空转超过1.5h”的证据。

新A/AB/C/D终态flag分别14/29、11/23、10/32、9/20，原始flag均保留。`harm-outside-candidate-commands.json`保存64条原始flagged command；粗检包括磁盘/memory元数据、真实package源码、/dev/null重定向和父级skills查找。类别不能混成已验证答案泄露；但D3的真实test target读取又完全不在其中。这正说明检测有误报语义和漏报语义两端，不能用“看着大多无害”清除冻结flag。

建议按证据权重排序：

1. **输入边界与per-attempt provenance先于晋升判断**：独立语义访问审计，阻断benchmark target自动加载；保存每次request对应的verdict正文与proposal hashes。根因属于harness/measurement，独立设baseline。
2. **可靠的save/load预检与失败watch**：Gemma GenerationConfig、processor文件、state_dict映射、GPU进程释放具有多条实证；静态checkpoint cadence条款不能替代它。预检应绑定将要运行的保存/加载路径。
3. **把 checkpoint 条件写成可执行保留与评估计划**：计算训练steps、轮转、final额外保存后实际存活的时间点，至少两个充分分离的候选，再给出真实评估预算。仅设save_steps经常被rotation吞掉；避免把“不一定有价值的所有中间点评估”机械化。
4. **L3明确针对原始还是修后计划，修复后再决定**：有代码证据的precondition用defer，不把条件成功写成as-written yes。不能自动把低置信L0 no转成永久kill；C3+10pp与w10小样本逆转均是防止误杀的反证。
5. **当前卡有决策价值才继续probe，并做cost-accounting**：旧w10的未来卡格式/污染检查占用了当前阻塞review；但changed:none也可能是必要确认，不能按该字段一刀切。先记录总生命周期、实际锁等待与独立GPU利用率，再讨论速度改善。

目前确认了一批有用的局部修复、一次严重未被flag的输入边界违例、以及终态聚合抹平中途决策的测量问题；没有在这些手工案例中确认“一个本来会成功的计划因WMA永久no而被实际放弃”的因果损害，也没有确认可归因WMA的实际GPU小时节省。
