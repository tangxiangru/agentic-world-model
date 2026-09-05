# exp_protocol Round 00：GSM8K baseline calibration

**日期**：2026-09-02
**状态**：批准发射前检查
**line**：`gangda_exp_protocol_evolve`
**subqueue**：`gangda_exp-protocol-evolve`，16 H100
**基础合同**：`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md`

## 目的

在没有 candidate 的情况下测量当前 exp_protocol baseline 的真实 run-to-run 分布，并
验证 high-effort AWM scaffold、card/lock/close 路径与 canonical Claude judges 能在完整
PTB 流程中工作。Round 00 不产生晋升结论，不运行 AIME2025。

## 冻结设置

| 项 | 值 |
|---|---|
| task | `gsm8k` |
| base model | `google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d` |
| scientist | `claude_vertex_high_awm` / `claude-opus-5[1m]` / `high` / 1M |
| protocol SHA | `eaf50919ff5f79f15e33df7bb49f44ffebacfc64` |
| PTB SHA | `dcf5da031435c54e3680b6ec3f63e7e317efc13e` |
| agent budget | 10 h per formal cell |
| repeats | 16 formal cells，`replicate=1..16` |
| canonical judges | official = Claude Opus 5 high / Vertex |
| judge container | `opus_5.sif@35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759` |

16 repeats 是一次预先声明的 baseline calibration，不是看结果后追加的选择性重复。它们
共享完全相同的设置，只用独立 agent stochasticity 测量方差，并可填满本线 16-GPU
硬分区。

## Pilot 与异步发射

> 2026-09-02 12:20 UTC 用户指令：不走 pilot，formal cells 直接发射；见基线 spec §七的
> 「用户指令」小节。本节其余内容保留为历史记录。

`p00r01` 先以 1 h pilot 验证完整 wiring。pilot 不计入 16 个 10 h formal repeats。
operator 通过 `pilot: first` 提交；只有 pilot 被 PTB validator 接受后才提交 formal
batch。planner 不运行 `sbatch`。

原始 v1 pilot（job `90462`，PTB `f4ae55a`）虽然发现并读取了 skill，却在
`check` / `lock` 前启动训练，因此只保留为 protocol-adherence 失败证据，绝不放行其
formal cells。PTB PR #4 在 `claude_vertex_high_awm` 中增加条件式首步 bootstrap：只有
安装了 exp_protocol 的 cell 才要求第一动作 invoke/read skill，并在训练或评估前成功
lock；未安装 protocol 的 null control prompt 保持原样。v2 pilot（job `90464`）用于
留下 bootstrap 行为证据，但其 receipt 暴露了缺失的 `awm.protocol_tree`；它同样不放
formal。job `90464` 的 live trace 已证明修复生效：Claude 初始化后的第一个 tool call
就是 `Skill(exp_protocol)`，随后完整载入 skill。最终 Round 00 baseline 使用 v3 batch：
manifest 显式声明并由 `awm ptb check` 核验 protocol tree，checkout 的旧 marker 若缺少
tree 也会被重建。根据保持 16 GPU 有独立工作排队的指令，这份已验证 v3 直接提交 formal，
不再重复 wiring pilot。

### 预声明 precision extension

在所有 Round 00 formal jobs 尚无 terminal result、尚未观察任何 formal 分数时，额外冻结
`exp-protocol-gsm8k-gemma4b-high-r00-baseline-b-x8-v1`：`run_index: 2`、manifest 内
`replicate: 1..8`，汇总时是 baseline 的全局第 17–24 个观测。它与 null-control 的第三个
x8 batch 成对，只用于提高方差/规程执行率估计精度并维持至少 16 个 pending backfill。
原 v3 的前 16 个是不可替换的 core set；extension 不能因 core cell 失败或分数不利而替换
它。报告同时给出 core-16 与 all-24，逐项列出 validator 排除，不做分数选择。

formal 16 cells 一旦 gate 打开便作为同一 immutable manifest 异步提交。后续分析不等待
无关队列；但不得在本批结果出现前预造依赖 Round 00 结论的 candidate。

### Strict-site 补跑（held，2026-09-02 20:56 UTC 追加）

p00r08（job 90482）因会话结束杀掉训练而 FAILED；p00r11–p00r16 与整个 baseline-b 波在 Slurm 中
丢失 `ReqNodeList`，跑在冻结节点之外，只进 sensitivity。
`exp-protocol-gsm8k-gemma4b-high-r00-baseline-strict-x8`（batch id `…-v1`）逐字段复用 baseline-b
的合同（protocol tree `08674f2c`、awm `eaf5091`、setup `--exp-protocol --tool claude`），只换标识：
cells `p00s01..p00s08`、`run_index: 3`（1 = core x16，2 = baseline-b）。以 `want: held` 提交并登记，
保持 `PENDING(JobHeldUser)`；放行门见 `doc/spec/2026-09-02-exp-protocol-round01-session-guard.md`。
它为 Round 01 的 strict guard 补跑提供同样大小的 strict comparator；不替换 core-16 中任何已终结的观测。

## 收割与分析门

所有 cells 都照常收割，失败也保留。用于统计的 cell 必须同时：

1. 能由 receipt 解析到本 manifest、spec、top/PTB commits；
2. `awm ptb results MANIFEST` 判定 validator-complete；
3. 四个 canonical judgement 存在且 judge-clean；
4. `task/memory/cards/` 可读。

当 8 个有效 formal cells 到达时，可向 Fable 发一个 interim analysis window，但不得据此
改 baseline。Round 00 的主要分布报告等待全部 16 cells，或至少 12 个有效 cells 且其余
仅为尚未完成/明确基础设施失败。任何排除都按 receipt 顺序列出，不能按分数选择。
precision extension 不推迟 core-16 的 primary analysis；其结果到达后追加 all-24 sensitivity
analysis。

最终记录写到 `doc/exp_protocol_iterations/2026-09-02-round-00.md`，至少包含：

- accuracy mean、range、stderr 分布；
- `pitfalls_cost_h`、`pitfalls_hit`；
- `n_locked_open`、`n_closed/n_cards`、`fields_filled`；
- `preflight_fail`、`n_relocked`、`n_overrides`、`n_unreadable`；
- 每个 variant 至少三张人工 card 阅读（本轮只有 baseline）；
- 一个可追溯的下一 candidate，或明确的 `no change`。

## 停止条件

- `OWNERSHIP FAIL`：停止新提交并报告；
- pilot 不完整：不放 formal，收割证据后修基础设施；
- manifest/check/source 不干净：不提交；
- canonical judge 重新出现 auth failure：不把缺 judge 的 cell 计为科学完成；
- 不取消任何 RUNNING job。
