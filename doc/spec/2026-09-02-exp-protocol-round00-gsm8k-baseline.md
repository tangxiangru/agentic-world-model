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
| PTB SHA | `f4ae55a8f2bbfb8809839b87248c8f9998015518` |
| agent budget | 10 h per formal cell |
| repeats | 16 formal cells，`replicate=1..16` |
| canonical judges | official = Claude Opus 5 high / Vertex |
| judge container | `opus_5.sif@35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759` |

16 repeats 是一次预先声明的 baseline calibration，不是看结果后追加的选择性重复。它们
共享完全相同的设置，只用独立 agent stochasticity 测量方差，并可填满本线 16-GPU
硬分区。

## Pilot 与异步发射

`p00r01` 先以 1 h pilot 验证完整 wiring。pilot 不计入 16 个 10 h formal repeats。
operator 通过 `pilot: first` 提交；只有 pilot 被 PTB validator 接受后才提交 formal
batch。planner 不运行 `sbatch`。

formal 16 cells 一旦 gate 打开便作为同一 immutable manifest 异步提交。后续分析不等待
无关队列；但不得在本批结果出现前预造依赖 Round 00 结论的 candidate。

## 收割与分析门

所有 cells 都照常收割，失败也保留。用于统计的 cell 必须同时：

1. 能由 receipt 解析到本 manifest、spec、top/PTB commits；
2. `awm ptb results MANIFEST` 判定 validator-complete；
3. 四个 canonical judgement 存在且 judge-clean；
4. `task/memory/cards/` 可读。

当 8 个有效 formal cells 到达时，可向 Fable 发一个 interim analysis window，但不得据此
改 baseline。Round 00 的主要分布报告等待全部 16 cells，或至少 12 个有效 cells 且其余
仅为尚未完成/明确基础设施失败。任何排除都按 receipt 顺序列出，不能按分数选择。

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
