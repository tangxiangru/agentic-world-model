# exp_protocol Round 02：三个独立的 4-cell 筛选 + guard 漂移对

**日期**：2026-09-02 23:07 UTC **状态**：已登记 held，放行门关闭 **前置**：`doc/exp_protocol_iterations/2026-09-02-round-00.md` 的 Analysis window 01、`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md` §四（两段式）、planner 决定（PR #20 评论 5517615610）

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

- **A**：最终交付 greedy / measured config 已在后续 guard baseline 5/5 饱和，不再作为主指标。主指标改为
  首次 post-SFT 评估结束到基于 grader observable、同权重对照而确定 decode 配置的时间：4/4 ≤ 0.5 h；
  次指标为这段时间之后仍在未测量 sampling 配置下判定的 decision card 数（每 cell ≤ 1）。仍记录最终
  `generation_config`、grader observable 与父 checkpoint validator 失败，防止 candidate 只是更快写错配置。
- **B**：目标指标 = RFT 卡 `pitfalls_hit` 中可归因于采样的小时数（Round 00 约 1 h/尝试 RFT 的 cell →
  期望 < 0.3），以及 RFT 卡的 verdict。
- **C**：目标指标 = 每 cell 最大评估 n（inspect log 大小 ÷ 44 KB）≥ 500 的 cell 数 ≥ 3/4，且 trace
  中无"小样本排序反转"的报告。Window 02 额外记录：若 claimed delta ≤ 1 percentage point 或
  不大于一个 marginal standard error，card 是否使用 paired item counts 或 repeated read；这是 C v2
  现有"SE 小于 delta，或 paired statistic"规则的 screen observable，不改写已冻结的 protocol tree。
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

## 六、trace review 之后的修订（2026-09-03 02:27 UTC）

`doc/exp_protocol_iterations/2026-09-03-trace-review-round00.md`（16 份 cell 报告的合成，报告存于
`doc/exp_protocol_iterations/trace-reviews/round00/`）改写了 A、B、C 的措辞并给出了新候选。A/B/C 的 v1
held job（90845–90856）从未启动，整块撤回，不构成挑选；v2 manifest 以新 commit 登记。第二波三个新筛选
D、E、H 与漂移对 B 同样登记为 held，排在第一波之后。head 在 `2f64581` 还原为 guard 树；六个候选各在
自己的 commit 里、互不包含（先回退再添加）。

| 筛选 | 改动（一项） | commit | protocol_tree | manifest / cells | 4 cell 里读什么 |
|---|---|---|---|---|---|
| A v2 | `decode_config_inherited` 改写：点名可核实的观察（vLLM 日志行 / 请求体只有 max_tokens）、反驳 trace 里记录的两个错误信念、自建评估器要复制请求体；父 checkpoint 陷阱扩到 5 个 cell | `f6cdccc` | `73083443` | `…-r02-a-decode-x4-v2`，`a02s01–04`，run_index 2 | 首次 post-SFT eval → measured decode choice ≤0.5 h（4/4）；之后未测量 sampling decision card ≤1/cell；最终 config 与 grader observable 仅作正确性护栏 |
| B v2 | `vllm_offline_prompt_and_stop` 改写：n>1 才丢 stop id 的机制、解析器 inf 崩溃、probe 打印 finish_reason；来源改为两臂都付出过（对照 5.0 h） | `9f294c3` | `2ca65d4d` | `…-r02-b-vllm-sampling-x4-v2`，`b02s01–04`，run_index 2 | RFT 卡里可归因于采样的小时数 < 0.3/cell |
| C v2 | 规则 2 段落改写：`--limit N` 取前 N 题且前段偏易 2–7 点、全集评估 3–10 分钟、`falsified_if` 要在标准误小于所称差距的 n 上或用配对统计、允许中途提高 n 并重测 incumbent；示例卡改为 n=500；模板 `falsified_if` 行加注 | `57511f9` | `f528e150` | `…-r02-c-eval-n-x4-v2`，`n02s01–04`，run_index 2 | 每张卡 `evaluation.protocol.n` ≥ 500 的 cell ≥ 3/4；无被更大 n 推翻的 contradicted；无 n ≤ 200 的 falsified_if |
| 漂移对 A v2 | guard tree；使用六候选构造后恢复的同代 AWM paths | `2f64581` | `189319d6` | `…-r02-guard-drift-a-x2-v2`，`g02v2r01–02`，run_index 5 | 并入 baseline 池；与 A/B/C 的 shipped paths 只差候选项 |
| D | preflight 检查 `parent_generation_config_valid`（父 checkpoint 的 greedy 配置会让 Trainer 首次保存失败）+ 测试 + pitfalls 目录行 `greedy_parent_generation_config` | `8332917` | `7160d360` | `…-r02-d-parent-config-x4`，`d02r01–04` | 归因于 GenerationConfig-is-invalid 的小时数 = 0 且检查至少触发一次；对 stock 配置零 override |
| E | 规则 9、Stop hook 文案、`run_dies_with_the_session` 三处的"怎么等"：等 PID 与变化的 tail，把评估链到 run 退出 | `7832cb9` | `58af0780` | `…-r02-e-wait-on-process-x4`，`e02r01–04` | 条件候选：先读 Round 01 guard；若 ≥7/8 已 <0.15 h，则 target 饱和、释放前整块撤回 E；否则最后运行，读 run 死亡/退出到下一命令的空转 <0.15 h/cell |
| H | `setup.data` 只在 family 训练目标文本时必填（schema、questions、模板注释） | `b52e5f2` | `88133acb` | `…-r02-h-eval-only-data-x4`，`h02r01–04` | 非训练卡上零伪造数据条目、零 `data_files_exist` override；`fields_filled` 不降 |
| 漂移对 B | guard tree；使用六候选构造后恢复的同代 AWM paths，run_index 4 | `2f64581` | `189319d6` | `…-r02-guard-drift-b-x2`，`g03r01–02` | 并入 baseline 池；与 D/E/H 的 shipped paths 只差候选项 |

原冻结波次为 A/B/C 与 D/E/H。Window 02 后的**放行顺序**调整为：第一可释放波 = D、B v2、C v2 +
漂移对 A（14）+ 2 机动；第二波 = A v2、H、以及由当时证据在 G/P1 中只选一个单项候选 + 漂移对 B。
E 保持 held 但不进入释放波，直到 Round 01 guard 的条件判定完成；若未饱和才作为最后一个 screen。
D 是 A 推荐的修法所制造的陷阱的机械守卫，synthesis 建议 A 胜出后并入 A；作为独立筛选它回答的是
检查是否误报、是否真的省下小时。

旧漂移对 A jobs 90857–90858 虽然 protocol tree 也是 `189319d6`，却冻结在旧 SHA `4ae3d87`；六个
候选 commit 共同包含后来加入的非干预 `awm/exp_protocol/collect.py` 基础设施。单个 `awm ptb check`
只能证明 manifest 自洽，不能证明变体间 shipped paths 相同。因此旧 pair 保持未启动并整块撤回，
用 `2f64581` 的 drift A v2 替代；尚未登记的 drift B 也在登记前改为 `2f64581`。
修正后的第一、二波共 8 份 manifest 均通过完整 site `awm ptb check`（0 issues）。

operator 于 2026-09-03 02:48 UTC 登记 held receipts（commit `3fd73fc`）：A v2 jobs
91046–91049，B v2 91050–91053，C v2 91054–91057，漂移对 A v2 91058–91059；D
91060–91063，E 91064–91067，H 91068–91071，漂移对 B 91072–91073。28/28 均为
`PENDING(JobHeldUser)`，`ReqNodeList=slurm2-a3nodesetondem-[0-1]`，ownership OK。旧 A/B/C v1
与旧 drift A jobs 90845–90858 均从 PENDING 整块取消；无 RUNNING job 被触碰。

排队未做（台账）：F `terse_target_style`（数据风格，规程不该规定训练数据，先观察）、G
`trl_grpo_gemma_zero_gradient`（5/5 对照 GRPO cell 踩到；规程臂无 RL 卡，条目会测未验证的训练器这个
信念是否挡住了 RL）、I `stop_token_consistent` 接受脚本追加的声明（3/9 cell 为它重写数据）。

## 七、Window 02 的 Fable 平行审阅裁决（2026-09-03 08:45 UTC）

Fable 的八份独立 cell 报告与 synthesis 在
`doc/exp_protocol_iterations/2026-09-03-trace-review-round00-window02-fable.md`。planner 完整读取 synthesis
后接受三个实验设计结论，但不改任何已冻结的 protocol tree：

1. A 的原主指标已饱和：本窗口 baseline 5/5 最终都交付 greedy；真正有区分度的是首次 post-SFT eval
   到 measured decode choice 的时间（p00r11 3.3 h，其余四个 ≤12 min）及期间错误判定的卡片数。因此只改
   screen observable，不造 A v3。
2. D 在本窗口覆盖 p00r11、p00r14、c01r01、c01r03 共 3.3 h；两窗口约 8.7 h，且 check 对安全/不安全
   配置的方向由 trace 验证。D 提到第一可释放波，A 延后。
3. E 的 `<0.15 h` 指标在本窗口 baseline 5/5 已达标。Round 01 guard 使用同一旧等待措辞，是预注册的
   判定样本：若 ≥7/8 仍达标，E 没有可移动的 baseline，jobs 91064–91067 在启动前整块撤回，并由 G 或
   P1（届时只选一个、先写 spec、单项树）替代；若不达标，E 保留但最后放行。当前不取消、不释放。

B 的 weak sampled-stop 更正、C 的 repeated-read 证据、H 的 fake-data 证据和排队方向 P1–P4 均进入分析
口径与方向台账；它们不重写本轮冻结候选。所有放行仍同时要求 clean Round 01 guard 与原生两节点隔离；
截至本裁决节点 1 仍因 `SlurmdSpoolDir is full` drain，所以门保持关闭。AIME2025 仍只在晋升前运行。
