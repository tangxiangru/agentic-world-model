# exp_protocol Round 00 对照组：同一 scaffold、无规程

**日期**：2026-09-02
**状态**：提议；与 Round 00 baseline calibration 配对
**line**：`gangda_exp_protocol_evolve`
**subqueue**：`gangda_exp-protocol-evolve`
**基础合同**：`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md`
**配对批次**：`doc/spec/2026-09-02-exp-protocol-round00-gsm8k-baseline.md`

## 为什么需要它

Round 00 的 baseline calibration 测的是"装了规程的 scientist"的分布。它回答不了这条线
最先要回答的问题：**规程本身让分数、卡片和坑的代价相对于什么变了**。以后每一轮
candidate 对 baseline 的比较都在"装了规程"这个前提内进行；如果规程的存在本身就
让 accuracy 掉了 0.05，那所有后续改进都是在一个被拖累的区间里比。只有一个没有规程的
对照能把这个前提量出来。

历史批次里的 `claude_vertex_high` cells 不能充当这个对照：它们用 `--setting-sources ""`、
没有 `/home/ben/awm` 挂载、PTB commit 与 judge 也不同。对照必须与 baseline 只差规程。

## 设计

| 项 | 值 |
|---|---|
| scaffold | `claude_vertex_high_awm`（与 baseline 相同） |
| 挂载 commit `awm.sha` | `eaf50919ff5f79f15e33df7bb49f44ffebacfc64`（与 baseline 相同） |
| 挂载路径 | `awm/__init__.py, awm/cli.py, awm/paths.py, awm/sandbox.py, awm/exp_protocol` —— **不含 `skills/exp_protocol`** |
| setup | `--tool claude`（不带 `--exp-protocol`；`awm sandbox setup` 只写 `awm_sandbox.json`） |
| 其余 | task、base model、scientist、effort、context、时长、judge、PTB commit 逐字段与 baseline 相同 |
| repeats | 8 个预先声明的 formal cells，`replicate=1..8` |
| pilot | `c00r01` 先跑 1 h 验证接线；不计入 formal |

对照 cell 里 Claude Code 仍以 `--setting-sources project` 启动，任务目录里没有
`.claude/skills`、没有 `CLAUDE.md` 指针、没有 `skills/exp_protocol`。scientist 看到的
是原生 PTB 任务加一个空的 setup 记录。receipt 中该 checkout 的 `protocol_tree` 为
`null`，这就是"无规程"变体的标识。

## 分析

对照与 baseline 用同一套指标：accuracy 的 mean / range / stderr。规程指标
（`pitfalls_cost_h`、`fields_filled` 等）在对照里没有卡片可读，`collect` 会给出零张卡；
这本身就是要记录的结果。比较只在两批的 receipt 都指向同一 PTB commit、同一 judge
容器时成立。

对照不参与 candidate 选择，也不进入 held-out 候选池；它只给这条线一个零点。

## 停止条件

与 baseline batch 相同：`OWNERSHIP FAIL` 停止；pilot 不完整不放 formal；不取消 RUNNING。
