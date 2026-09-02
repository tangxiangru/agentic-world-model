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
