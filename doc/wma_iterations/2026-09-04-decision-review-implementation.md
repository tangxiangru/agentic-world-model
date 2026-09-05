# WMA 决策审查改动：实施记录

日期：2026-09-04。状态：已在工作区实现并通过本地验证，未部署、未晋升、未提交新的 GPU 实验。

依据与设计见[修改建议](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/spec/2026-09-04-wma-decision-review-redesign.md)。本次包含政策、规程、测量和运行隔离四类改动；它们不能在后续实验中合并归因成某一条 skill 的收益。

## 已实现的行为

1. **限制否决依据的外推。** WMA 在建议停止、替换或降低方案优先级时，要说明证据检验的命题，以及对当前 parent、数据、目标、调度和评估的适用性。旧 checkpoint 平台期和独立短训练结果不再被示范为新完整方案的失败判据。修正了短训“不涨 3pp 就停止”的示例及过度泛化的历史先验；真实实现错误仍可要求修复。
2. **在最终选择前共同比较候选。** `awm wma propose/compare/choose` 支持通常 2–3 个、最多 3 个真实候选；单候选须说明原因。Scientist 先记录自己的偏好，WMA 在一次调用中比较相同背景下的成本、证据和不确定性。对照组也能记录相同简案后自行选择。Scientist 可以不同意 WMA，或明确放弃本次决策。所选方案仍须走正式卡和阻塞审查。
3. **保留每次选择、审查和执行版本。** 候选、审查请求及回复、动作和命令启动记录按事件不可变保存。`awm wma act` 记录实际行动及依据；`awm exp_protocol run` 核对正式方案、脚本/数据/配置指纹、当前锁、对应回复和 proceed 动作。修改输入后要重新审查并绑定选择；同一 proceed 不能重复启动。后来的放弃或新选择会使旧选择失效。记录命令结束不代表训练或科学评估完成。
4. **约束实际探针访问。** 内置 Claude 后端只暴露受限工具代理，探针进程通过 Linux Landlock 和 seccomp 限制文件、网络及进程访问。输入使用有大小限制、带哈希的明确快照；不导出原始数据缓存、测试答案文件或未来卡片。返回的成本和隔离元数据由运行时记录。完整嵌套审查与比较证据可被收集，且不会重复计入旧 ledger。

## 实现位置

- 政策和 scientist 流程：`skills/wma/`、`skills/exp_protocol/`。
- 公共记录合同和启动校验：`awm/exp_protocol/decisions.py`、`awm/exp_protocol/run.py`、`awm/wma_client.py`；旧导入路径保留兼容转发。
- 联合比较及私有审查：`awm/wma/compare.py`、`awm/wma/sidecar.py`、`awm/wma/backends.py`。
- 探针隔离：`awm/wma/isolation.py`。审查记录收集：`awm/ptb_ops.py`。
- [使用说明与记录路径](/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/reference/wma_online_sidecar.md)。

既有单卡 verdict schema、scorer、显式 ship 列表和冻结 manifest SHA 未改。既有 PostTrainBench 子模块工作区变化未由本次修改产生。

## 验证

针对本次行为及兼容边界的 22 个测试文件共 **354 项测试全部通过**；修改文件的 Ruff 检查通过，`git diff --check` 通过。

覆盖范围包括：一次联合比较调用、无 WMA/超时/失败分支、过期候选与迟到回复、方案和文件变化、同一秒重锁、旧回复不能覆盖新锁、scientist 与 sidecar 挂载路径映射、选择绑定及放弃语义、重复启动防护、原子记录发布、私有证据收集，以及旧 schema/ledger/replay/安装接口回归。

隔离测试使用本机真实内核验证间接文件/缓存读取、符号链接、网络和进程访问、子进程继承、资源限制及 MCP 传输；模型进程由测试替身提供。测试通过不表示真实模型已正确使用新接口，也不表示最终 GSM8K 分数或训练效率已提升。

## 当前边界与后续验收

- 当前隔离探针只支持 Claude 的 CPU 静态检查、Python 标准库和少量系统工具，不提供 ML 包、GPU、模型权重或在线下载。不能执行的检查须报告未验证，不能据此否决训练方案。动态 ML 检查需另建受控接口。
- 内置 Codex 尚无对应隔离适配器，会拒绝不受保护的执行。内核或 CLI 能力不满足时同样不会回退到无限制探针；scientist 流程仍按原策略记录 WMA 不可用。
- 启动校验约束 `awm exp_protocol run` 入口，不是禁止 scientist 任意 shell 启动训练的完整容器边界。授权文本若本身含有泄漏答案，不会因文件隔离自动得到语义清洗。
- 尚未进行生产容器中的真实 Claude/API 验收。新实验前需验证容器内核、CLI 参数、配额和真实模型到工具代理的完整往返。
- 科学验证须分别比较证据规则和联合选择机制；候选规程、隔离及测量口径在对照两侧保持一致。未执行候选的终点仍是未知。预算优化、减少重复审查和延长训练反馈窗口未包含在本次实现中。
