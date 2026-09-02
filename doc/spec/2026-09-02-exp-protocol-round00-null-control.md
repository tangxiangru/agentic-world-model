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

## 第二批（2026-09-02 12:40 UTC 追加）

`exp-protocol-gsm8k-gemma4b-high-r00-nullctl-b-x8`：执行合同相同的第二个 immutable batch，
`run_index: 2`，cells `c01r01..c01r08` 在本 manifest 内仍用 schema 要求的
`replicate: 1..8`，汇总时对应零点的全局第 9–16 个观测。它排在 v3 baseline 之后作为待排缓冲：第一波
（8 对照 + 8 baseline）结束后，剩余 8 个 baseline 与这 8 个对照一起填满第二波，对照与 baseline
各达 16 个。它不依赖任何结果；分析时两批对照合并，前提是 receipt 指向同一 PTB commit。

## 第三批：配对 precision extension

在所有 Round 00 formal jobs 尚无 terminal result、尚未观察任何 formal 分数时，冻结
`exp-protocol-gsm8k-gemma4b-high-r00-nullctl-c-x8-v1`：`run_index: 3`、manifest 内
`replicate: 1..8`，汇总时是零点的全局第 17–24 个观测。它与 baseline-b-x8 成对，执行
合同保持一致且仍不安装 protocol。前两批 16 个 control 是不可替换的 core set；第三批
不能替换其中失败或低分的 cell。报告同时展示 core-16 与 all-24。

## 第四批：strict-site 补跑（held，2026-09-02 21:10 UTC 追加）

nullctl-b（`c01r01..c01r08`，jobs 90491–90498）在 Slurm 中丢失了 `ReqNodeList`，全部跑在冻结的
`slurm2-a3nodesetondem-[0-1]` 之外；它们照常收割，但按 placement quarantine 只进 sensitivity，
不进 primary。`exp-protocol-gsm8k-gemma4b-high-r00-nullctl-strict-x8`（batch id `…-v1`）逐字段
复用 nullctl-b 的合同，只换标识：cells `c01s01..c01s08`、`run_index: 4`（1 = nullctl，2 = nullctl-b，
3 = 已取消的 nullctl-c；提交过的 run_index 不复用）。它以 `want: held` 提交并登记，保持
`PENDING(JobHeldUser)`，只有通过 `doc/spec/2026-09-02-exp-protocol-round01-session-guard.md` 的
放行门（`OWNERSHIP OK`、逐 job `ReqNodeList` 与 receipt 一致、原生两节点隔离恢复）才可 release。
它的作用是让零点在 primary 口径下至少保有 8 个有效 cell，即使首波对照再有 truncated；
不替换 core set 里任何已终结的观测，报告继续同时给出 core 与 all。

## 分析

对照与 baseline 用同一套指标：accuracy 的 mean / range / stderr。规程指标
（`pitfalls_cost_h`、`fields_filled` 等）在对照里没有卡片可读，`collect` 会给出零张卡；
这本身就是要记录的结果。比较只在两批的 receipt 都指向同一 PTB commit、同一 judge
容器时成立。

对照不参与 candidate 选择，也不进入 held-out 候选池；它只给这条线一个零点。

## 停止条件

与 baseline batch 相同：`OWNERSHIP FAIL` 停止；pilot 不完整不放 formal；不取消 RUNNING。
