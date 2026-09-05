# 2026-09-05 最新分析与本轮执行记录

截至18:31 UTC，实验子队列已核实16/16张H100分配给16个RUNNING科学会话，另6个HumanEval会话已释放为可运行PENDING。OWNERSHIP OK；仅使用ondem0–1。

| 任务 / 处理 | 运行 | 排队 | 正式job ID |
|---|---:|---:|---|
| GSM8K process-knowledge |4|0|92129–92132|
| GSM8K guard桥接 |4|0|92137–92140|
| GSM8K E-repair |2|0|93070–93071|
| HumanEval 无协议 |2|2|93139–93140；93145–93146|
| HumanEval process-knowledge |2|2|93141–93142；93147–93148|
| HumanEval E-repair |2|2|93143–93144；93149–93150|

## 数据与真实运行边界

HumanEval数据已经下载：164题、83,920字节，SHA256 `2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6`。这份数据用于评估/污染参考，不是新增训练语料。GPQA Main上次官方源访问403，尚未完成真实数据准入；按用户要求本轮跳过，无GPQA正式会话。

HumanEval环境job93133在ondem0的两种固定镜像均通过五项人工程序检查、原生评分、内部超时、隔离以及外层中断清理。完整原始证据和归档均通过validator。此前93104的host python入口失败、93118的真实外层清理失败，以及93133的两次收集器诊断均保留。修复了正式运行的自有进程回收，收集器另修复无扩展名JSON遗漏和原生运行时哈希空字段约定，原冻结哈希未改变。

正式HumanEval实验现在已启动，尚无HumanEval模型成绩；环境人工程序通过不计为模型结果。12个会话保持每臂4次、相同Gemma-3-4B-PT/Opus4.8-high/10h和PTB `167415012dd27e03a627a5443679e5dd73654e12`。所有HumanEval仅请求已验收ondem0，不自动借用未验收节点。

## 最新GSM8K结论与本轮问题

已结束的首轮：无协议3个clean均值63.1286%，另1次失败；旧E3个clean均值54.9406%，另1个完整但异常结果54.2835%。clean均值相差8.188个百分点，只是小样本、条件于完成的描述，不能认定稳定因果效应。

E-repair把明确复现的阶段输入/前置readiness、固定版本采样API与有界失败观察、实际生效解码参数证据一起修复。先看错误复发、有效工作时间、交付完整性与质量分布；不强制训练配方，也不预先宣称提分。知识版与guard沿用已批准比较；HumanEval研究这些过程改进能否迁移到代码任务。AIME2025继续保留用于独立确认。

每小时完成监测现覆盖全部22个receipt-backed科学job，阈值8个terminal只用于唤醒；结果仍须receipt/PTB validator/judge/placement判定。原监测job集合完整保留，首轮tick已确认：16 RUNNING、6 PENDING、0 terminal。后续审阅不把排队、Slurm完成或人工环境测试算成clean科学结果。

证据索引：同目录 `synthesis.md`、`validation-summary.json`、`wave3-launch-state.json`、`wave3-monitor.json`；验收包 `results/ptb/exp-protocol-humaneval-environment-ondem0-v3/h04env01/`；科学配置为六份HumanEval A/B manifests及两份既有GSM8K方向和E-repair v2 receipt。
