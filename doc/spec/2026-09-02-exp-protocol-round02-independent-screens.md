# exp_protocol Round 02：三个独立的 4-cell 筛选 + guard 漂移对

**日期**：2026-09-02 23:07 UTC **状态**：待登记为 held **前置**：`doc/exp_protocol_iterations/2026-09-02-round-00.md` 的 Analysis window 01、`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md` §四（两段式）、planner 决定（PR #20 评论 5517615610）

## 一、依据

Round 00 首波 7 对 7：规程臂 0.6933（sd 0.059）对照 0.7552（sd 0.033），差 −0.062（Welch t −2.4）。
规程被完整执行（51 张卡全部锁定结案，29/29 训练启动在 lock 之后），分数差距来自规程留给
scientist 的三项选择，每项都能被一条 pitfalls 记录或一条规则点名：

1. 解码配置：对照 7/7 交付 greedy `generation_config`，规程臂 4/7；规程臂内 greedy 的四个 cell 平均
   0.727，交付采样配置的三个 0.648。两个 cell 又因把 greedy 配置写进父 checkpoint 而丢掉训练
   （`GenerationConfig.validate()`，0.55 h 与 1.2 h）。
2. vLLM 离线采样：规程臂 5 张 RFT 卡有 4 张 contradicted，p00r01 1.8 h、p00r03 0.45 h、p00r05 一个
   孤儿引擎，全是同三个默认值（第二个 `<bos>`、stop ids 未生效、被杀引擎不释放显存）。
3. 评估样本数：p00r01（200 对 600）、p00r03（同一 checkpoint 在 n=150 上 0.660–0.727）、
   c00r04（300 题第一名在全集排第 6）三次排序反转。

## 二、构造：每个候选只比 guard 基线多一项

基线 = Round 01 guard 规程树 `189319d63d301d64d96f8f41d051795404679f37`（commit `4ae3d87`）。分支是线性的，所以按
planner 的要求先回退再添加：每个候选 commit 的规程树与 guard 树恰好差一项，互不包含。

| 候选 | 改动（一项） | commit | protocol_tree | manifest / cells |
|---|---|---|---|---|
| A | `pitfalls.yaml` 新增 `decode_config_inherited`（check: null，preflight 每次打印） | `3be3a29` | `d300656e` | `…-r02-a-decode-x4`，`a02r01–04` |
| B | `pitfalls.yaml` 新增 `vllm_offline_prompt_and_stop`（A 不在树内） | `92d5c79` | `f319e5ae` | `…-r02-b-vllm-sampling-x4`，`b02r01–04` |
| C | `SKILL.md` 规则 2 加一段：n 必须承载所声称的差距，交付决定不得少于 500 题（A、B 不在树内） | `7f117a0` | `beef82de` | `…-r02-c-eval-n-x4`，`n02r01–04` |
| 漂移对 | 无改动，guard 本身 | `4ae3d87` | `189319d6` | `…-r02-guard-drift-x2`，`g02r01–02`，`run_index: 3` |

commit `6853a14` 把分支 head 的规程树还原为 guard 树；三个候选只存在于各自的 commit 中，
manifest 以 `awm.sha` + `awm.protocol_tree` 冻结，operator 从指定 sha materialize。
`exp_protocol_meta` 未改动。

## 三、每个筛选读什么

- **A**：目标指标 = 交付 greedy 或有测量依据的 `generation_config` 的 cell 数（Round 00 规程臂 4/7 →
  期望 ≥ 3/4），从 trace 的 `do_sample: false` 写入与 `result` 记录读；附带看是否再有父 checkpoint 的
  validator 失败。
- **B**：目标指标 = RFT 卡 `pitfalls_hit` 中可归因于采样的小时数（Round 00 约 1 h/尝试 RFT 的 cell →
  期望 < 0.3），以及 RFT 卡的 verdict。
- **C**：目标指标 = 每 cell 最大评估 n（inspect log 大小 ÷ 44 KB）≥ 500 的 cell 数 ≥ 3/4，且 trace
  中无"小样本排序反转"的报告。
- **护栏**：三者的 accuracy mean 不低于 baseline 池（v3 core 16 + guard）均值 − 0.03；n=4 只能分辨
  0.06，护栏只挡大跌。
- **赢家**：目标指标动了且护栏通过 → 新 immutable manifest 再补 4 个 cell（`run_index` +1）到 8 个，
  才谈分数效应或晋升；输家撤下未开始的 cell，不补跑。

## 四、队列与放行

四个 manifest 以 `want: held` 登记（`PENDING(JobHeldUser)`），在两件事之前不放行：
（1）Round 01 的 8 个 strict guard cell 显示 guard 无害（无 cell 因会话结束丢失训练、accuracy 不低于
baseline − 0.03）；（2）operator 的原生两节点隔离门通过。一波 16 卡 = 3 × 4 + 2 漂移 + 2 机动。
held receipt 登记后，planner 批准把 `baseline-b`（8 个 requeue 后仍 PENDING 的精度扩展 cell）
改为 `want: cancelled`，整块撤回不构成挑选。

## 五、记录

三个候选的证据表与判定写入 `doc/exp_protocol_iterations/2026-09-02-round-00.md` 的
Analysis window 01；Round 02 的结果记录另起 `doc/exp_protocol_iterations/<date>-round-02.md`。
