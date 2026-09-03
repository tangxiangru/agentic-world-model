# exp_protocol Round 01：session guard candidate

**日期**：2026-09-02
**状态**：已排队；与 Round 00 的 baseline（core-16 + baseline-b）比较
**line**：`gangda_exp_protocol_evolve`，subqueue `gangda_exp-protocol-evolve`
**基础合同**：`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md`
**round record**：`doc/exp_protocol_iterations/2026-09-02-round-00.md`（interim）

## 证据与改动

Round 00 第一个 formal 终态 p00r08（job 90482）：scientist 锁了 exp-02、启动 4154 步全参 SFT，
在第 207 步写下 "I'll report back when the run finishes" 并结束本轮。PTB 里 scientist 是单轮
`claude --print`：结束即结束会话，Claude Code 杀掉全部后台进程，checkpoint 尚未写出，9.4 小时作废。
pilot 90463/90464 也以更小的规模丢过后台进程（Bash 工具超时连带杀进程组）。这是 harness 的确定性
性质，规程必须把它教给 scientist，并在它要犯错的那一刻拦住。

候选改动（commit `4ae3d87`，一个改动、三处协同，全部在 `skills/exp_protocol/` 内）：

1. `hooks/stop_open_cards.py`：有 locked 且无 conclusion 的卡片时阻止结束本轮，说明"结束即结束会话"、
   如何前台等待、如何记录已死的 run；最多阻止 12 次（`memory/.stop_hook.json` 计数），CLI 坏掉时手工
   填 result/conclusion 即视为关闭。
2. `pitfalls.yaml` 新增 `run_dies_with_the_session`（check: null）。
3. `SKILL.md` 新增规则 9 "Your turn is the session"；hook 一节改为由 `awm sandbox setup --stop-hook` 安装。

已在本机以真实 `claude --print` 验证（`tools/ptb-sandbox-e2e --scenario stop-hook`）：model 回复 WAITING
试图结束，hook 阻止两次，model 据提示把已崩溃的 run 记为 `execution: failed` 并关闭卡片后才结束。

## 冻结设置

| 项 | 值 |
|---|---|
| manifest | `experiments/posttrainbench/exp-protocol-gsm8k-gemma4b-high-r01-guard-x8.yaml` |
| 变体 | `awm.sha` `4ae3d87c446bbda9732537a72b2f0fb3f96ac35a`，`protocol_tree` `189319d63d301d64d96f8f41d051795404679f37`，setup `--exp-protocol --tool claude --stop-hook` |
| 其余 | task、base、scientist、effort、context、时长、judge、PTB commit 逐字段与 Round 00 baseline v3 相同 |
| repeats | 8 formal cells，`replicate=1..8`，不走 pilot |
| 比较对象 | baseline v3 core-16 + baseline-b（同一 PTB commit `dcf5da0`，baseline-b 与本批同一波运行） |

## 队列决定

按用户指令（GPU 用满、队列常备、发现不行再撤），把尚未开始的 `nullctl-c-x8`（零点第 17–24 个观测）撤下，
让本批在下一波与 baseline-b 同时运行。对照组保留 16 个（nullctl + nullctl-b）。撤下的是 PENDING cell；
任何 RUNNING cell 不动。

## 判据

- 主要：因结束会话而丢失的 cell 数（baseline 臂 vs 本批）；`n_locked_open`；每个 cell 的 hook 阻止次数。
- 分数：accuracy mean/stderr 相对 baseline；不得低于 baseline mean 0.03 以上。
- 风险：hook 把没有 run 的 scientist 困住（上限 12 次）；若出现，记录并降低上限或改提示。
- 与 Round 00 相同的 validator/judge 门；排除按 receipt 顺序列出。

## Spillover 后的 strict-site replacement buffer

首波 8 个 guard jobs `90647–90654` 在 Slurm 中丢失 `ReqNodeList`，全部运行到冻结的
`slurm2-a3nodesetondem-[0-1]` 之外。它们正常收割但标为 placement quarantine，只进入
sensitivity，不能成为 primary 或晋升证据。

用户把持续队列下限更新为 **8 个 held pending cells**。Manifest
`exp-protocol-gsm8k-gemma4b-high-r01-guard-strict-x8-v2.yaml` 逐字段复用本候选，仅使用新
batch/cell identity 与 `run_index: 2`，作为 strict-site replacements。队列状态先设为
`want: held`：提交、登记 receipt，但保持 `PENDING(JobHeldUser)`。只有同时满足以下条件才可
把队列项改为 `want: submitted` 并 release：

1. `gangda_exp-protocol-evolve` 为 `OWNERSHIP OK`；
2. 每个 held job 仍为 `PENDING(JobHeldUser)`；
3. Slurm 中的 `ReqNodeList` 展开后仍逐节点等于 receipt 冻结节点；
4. 原生两节点 reservation/partition 或等价的 16-GPU 隔离已经恢复，不能只相信一次性的
   `ReqNodeList` 检查。

### 2026-09-03 09:39 UTC 用户放行决定

ondem-1 的 `SlurmdSpoolDir is full` drain 已由集群侧清除，节点重新可调度。底层
`robtang-ptb-a3` reservation 仍覆盖 11 个节点，但用户明确要求立即占用 ondem-1；对本 strict
replacement block，以下四项合在一起由用户接受为第 4 项的“等价隔离”：registry 为
`OWNERSHIP OK`、8/8 jobs 仍为 `PENDING(JobHeldUser)`、每个 job 与 receipt 都冻结到
`slurm2-a3nodesetondem-[0-1]`、节点 0–1 正是本 subqueue 的 registry 边界。operator 因此只释放
jobs 90791–90798；原 wave 的 6 个 `launch_failed_requeued_held` jobs 90649–90654 不动，Round 02
也不随之放行。若任何 job 的节点约束或 ownership 在 apply 前变化，整次 release 停止。

任何失败、取消、超时或错误终结的首波/补跑 cell 都要收割。validator-complete 的
placement-only 结果保留在 sensitivity；不完整结果作为 truncated/failed evidence。是否再补
取决于 primary 每个 variant 至少两个有效重复和 matched-arm 平衡，补跑永远使用新的 immutable
manifest/receipt。
