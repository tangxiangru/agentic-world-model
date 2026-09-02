# exp_protocol 自迭代基线：GSM8K / Gemma 4B / scientist high

**日期**：2026-09-02
**状态**：生效；第一轮 manifest 的设计基线
**分支**：`gangda_exp_protocol_evolve`
**集成目标**：`gangda-dev`，绝不直接合并到 `main`
**控制规程**：`skills/exp_protocol_meta/SKILL.md`

## 一、目标和边界

这条线改进的是 `skills/exp_protocol/`：让 scientist 更少浪费 GPU 时间、更完整地
记录实验、避开能机械检查的坑，同时不牺牲正式 PTB 分数。iteration agent 不是
scientist；它并行运行受不同 protocol commit 约束的 scientist cells，从 cards、preflight
报告和正式分数中决定下一次**单一改动**。

- 唯一迭代任务是 `gsm8k`。
- `aime2025` 是抽样式 held-out promotion gate，只验证少量已经由 GSM8K 选出的候选。
- 不用 AIME2025 选择改动、调阈值、写 pitfall 或设计下一候选。
- 一轮内除 protocol commit 外，task、base model、scientist、effort、context、时长和
  PTB 资源完全相同。
- `exp_protocol_meta` 永不装进 scientist sandbox，也不在本循环内部修改。

## 二、冻结的实验合同

| 项 | 固定值 |
|---|---|
| iteration task | `gsm8k` |
| held-out task | `aime2025` |
| base model | `google/gemma-3-4b-pt` |
| base revision | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` |
| scientist model | `claude-opus-5[1m]` |
| scientist effort | `high` |
| context | 1,000,000 tokens |
| agent time | `PTB_NUM_HOURS=10` |
| cell resources | 1 H100、16 CPU、128 GiB RAM、400 GiB scratch |
| canonical judges | `official` profile = Claude Opus 5 high through Vertex |
| judge container | `opus_5.sif` @ `35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759` |
| minimum repeats | 每个 task × protocol variant 至少 2 个正式、独立、有效 cell |
| initial protocol-content commit | `9680574019311c98b0171c3df8d81f2294a21244`（短 SHA `9680574`） |
| initial shippable baseline | `eaf50919ff5f79f15e33df7bb49f44ffebacfc64`（短 SHA `eaf5091`） |

`replicate` 是预先写入 manifest 的独立重复编号，不得在看到结果后重命名或挑选。
pilot 若使用缩短预算，只验证接线，不计入正式重复。

`9680574` 是最后一次直接修改 scientist skill 的 commit，但它早于
`awm/sandbox.py` 和只读 source 的 install 修复，不能满足六路径 shipping contract。
`eaf5091` 保留当前 protocol 内容并包含可执行的 CLI、sandbox、`awm/exp_protocol`
与 skill，因此是 Round 00 实际写入 `awm.sha` 的 baseline。

### High-effort AWM scaffold 门

本 spec 编写时 PTB 只有 `claude_vertex_max_awm`，其 `profile.env` 和 `solve.sh`
都真实请求 `max`。本实验不得把它在 manifest 中伪装成 `high`。第一轮发射前必须：

1. 在 PTB fork 增加并测试 `claude_vertex_high_awm`，保留 AWM 只读挂载与
   `awm sandbox setup`，但 profile 与 Claude CLI 均使用 `high`；
2. 通过 fork feature branch → PR → merge 流程；
3. 把顶层 submodule 指针固定到合并 commit；
4. 将
   `("claude_vertex_high_awm", "claude-opus-5[1m]", "high", 1_000_000)`
   加入 launcher 白名单并补测试；
5. PTB scaffold test、顶层 launcher/sandbox tests 与 `awm ptb check` 全部通过。

**完成状态（2026-09-02）**：PTB PR #2 建立 high scaffold；首个 pilot 随后证明仅靠
skill discovery 不能保证顺序。PTB PR #4 已合并为
`dcf5da031435c54e3680b6ec3f63e7e317efc13e`，当 exp_protocol 被安装时会把
invoke/read protocol 设为 scientist 第一动作，并要求训练/评估前成功 lock；同一
scaffold 的 null-control 路径保持 PTB 原 prompt 不变。scaffold test、canonical judge
profile test、顶层 allowlist test 与完整 CPU suite 均须通过。后续 manifest 必须使用
该 commit 或包含它的后继 commit；任何未来替换 scaffold 仍受相同门约束。

## 三、Round 00：只测 baseline 分布

第一批只运行 baseline，默认 2 个正式 GSM8K repeats；如果两次 accuracy 相差超过
0.03，或 protocol 指标给出相反方向，再补第 3 个预注册 repeat。Round 00 的目的不是
宣布改进，而是得到：

- Gemma 4B / Opus 5 high / 10 h 下的 accuracy mean、range 和 stderr；
- `pitfalls_cost_h`、`pitfalls_hit`；
- `n_locked_open`、`n_closed / n_cards`、`fields_filled`；
- `preflight_fail`、`n_relocked`、`n_overrides`、`n_unreadable`；
- scientist 是否确实执行 card → check → lock → launch → close。

iteration agent 对 baseline 至少人工读 3 张 cards；若所有 cells 合计不足 3 张，则读
全部并把 card 数不足本身记录为结果。分析写入
`doc/exp_protocol_iterations/2026-09-02-round-00.md`。根据明确的 card、pitfall 或
缺失字段，先写证据，再产生一个单变量 candidate commit。

## 四、常规 GSM8K 迭代轮

每轮比较 2 或 3 个 protocol variants，其中一个总是当前 baseline。默认设计是
baseline 对一个 candidate，各 2 个正式 repeats；证据不清时追加第 3 个，而不是用
单次最高分做决定。

每轮执行：

1. 冻结 variant SHA；candidate 相对 baseline 只改一个 protocol 事项。
2. 写 spec、immutable manifest 和 queue entry；所有非 protocol 条件逐字段相同。
3. `awm ptb check MANIFEST` 必须为零问题。
4. operator 通过 queue/reconcile 异步发射和收割，planner 不运行 `sbatch`。
5. `awm ptb results MANIFEST` 只接受 validator 完整且 judge-clean 的 cells。
6. `awm exp_protocol collect results/ptb/<batch>/*/task --csv` 汇总指标。
7. 每 variant 人工读至少 3 张 cards。
8. 先写 round record，再做下一次单一改动；record 与改动放在同一 commit。

可改范围只有：

- `skills/exp_protocol/SKILL.md` 的一条规则或表述；
- `skills/exp_protocol/pitfalls.yaml` 的一条有 source 的记录；
- `awm/exp_protocol/preflight.py` 的一个 check，并在
  `tests/test_exp_protocol_preflight.py` 中先加测试；
- card 的 optional 字段；required 字段变化必须升级 schema 并另写 spec。

无改动是合法结论，也必须记录。

## 五、GSM8K 高分候选池

“高分”按 variant 的预注册重复判断，不按单个 lucky cell 判断。candidate 进入
held-out 候选池必须同时满足：

1. 至少 2 个 validator-complete、judge-clean 的 GSM8K repeats；
2. candidate accuracy mean 高于当前 baseline mean；
3. 任一 repeat 不得比 baseline mean 低超过 0.03；
4. `n_locked_open`、`n_unreadable` 不增加，`fields_filled` 不下降；
5. 至少一项 protocol KPI 有可追溯改善，例如 `pitfalls_cost_h` 下降、某个重复 pitfall
   消失、或 preflight/lock/close 执行率提高；
6. 改动及其理由已经写进对应 round record。

如果两组 accuracy range 重叠且结论依赖分数，先给 baseline 和 candidate 各追加一个
repeat。追加前写入 manifest；不得看完第三次结果后只保留有利的那一格。

候选池按 GSM8K mean accuracy 排序，protocol KPI 作为并列判据。每个 promotion
window 默认积累 2 个合格 candidate variants；只把排名最高的最多 2 个送入 AIME2025。
这样避免每轮查看 held-out，减少对 AIME 的隐式迭代。

## 六、AIME2025 抽样式 held-out gate

触发后创建一个独立 immutable manifest：

- current baseline 与最多 2 个 GSM8K 入围 candidates；
- 每 variant 至少 2 个正式 AIME2025 repeats；
- 除 task 从 `gsm8k` 改为 `aime2025` 外，第二节的实验合同不变；
- 报告 accuracy 的同时报告 `correct/30`，因为一题等于 3.33 个百分点。

AIME2025 只回答“GSM8K 上看到的 protocol 改善是否泛化”。candidate 晋升需：

1. 所有用于决定的 AIME cells validator-complete 且 judge-clean；
2. accuracy 不出现超过一道题（1/30）的平均退化；
3. GSM8K 上改善的 protocol KPI 在 AIME cards 中方向一致，或至少没有反转；
4. card 完整性和 lock/close 执行率不退化。

若 held-out 不通过，保留旧 baseline。不得读取 AIME 的具体失败模式来设计下一候选；
下一候选仍只能来自 GSM8K cards。一次 held-out 结果无论通过与否都要记录，不能重跑到
满意为止；只有明确的基础设施失败才可用新 receipt 做等配置 retry。

## 七、异步发布、审核与改进

规划者与 operator 使用同一分支、不同 checkout，路径所有权不重叠：

| 角色 | 唯一写入范围 |
|---|---|
| iteration agent / planner | `experiments/posttrainbench/`、`doc/`、`skills/`、`awm/` |
| operator | `results/ptb/**` |

异步循环是硬要求：

1. exp-protocol subqueue 的利用率目标是有可运行工作时 `RUNNING=16`。调度库存
   `RUNNING + PENDING` 不低于 16；当独立、已验证的工作足够时，额外保持最多一整波
   `PENDING≈16`。不得为了字面 pending 数量提前编造依赖未完成结果的实验。
2. planner 对独立、已验证、可安全运行的 manifest 立即 commit/push，并更新
   `experiments/posttrainbench/queue.yaml`；不等待不相关的长尾 job。
3. operator 每 10–15 分钟运行
   `pull --rebase → reconcile → reconcile --apply → commit results → push`。
4. `pilot: first` 只阻挡依赖该 pilot 的正式 cells；其他已独立验证的 batch 可以先排队
   backfill 空闲 GPU。
5. receipt 是 job 所有权的唯一依据。operator 只取消 receipt 中列出的 PENDING job，
   永不自动取消 RUNNING job。
6. 失败 cell 照常收割并记录，不由 operator 重试；planner 以新 queue entry 明确决定
   是否做等配置 retry。
7. iteration agent 在结果陆续回来时立即验证和阅读，但只有每 variant 至少 2 个有效
   repeats 才能做比较结论。
8. 若第 3 个或更晚的 straggler 对一个已有两个有效 repeats 的稳健决定不再必要，可冻结
   决定并在 round record 写明排除理由；不得选择性排除不利结果。

Fable 5.1 通过长期 PR 批量参与 manifest 设计与结果审核，契约见
`doc/reference/exp_protocol_fable_collaboration_prompt.md`。一个 analysis window 在形成
完整 comparison block，或自上次分析后累计至少 8 个新有效 cells 时触发；无新证据时
不产生仅用于报平安的 PR comment。

`exp_protocol_meta` 不在单轮中修改。每完成 3 个 GSM8K rounds，iteration agent 与
Fable 单独写一次跨轮 meta retrospective；只有至少两个 rounds 重复出现的循环问题才
能形成独立的 meta 修改，不能与 protocol candidate 混在同一 commit。

任何 `OWNERSHIP FAIL` 立即停止新提交并报告。Slurm `COMPLETED` 不等于科学完成。

## 八、结果与 provenance

四层记录必须同时保留：

1. `experiments/posttrainbench/<batch>.yaml`：冻结实验设计；
2. `results/ptb/<batch>/*.json`：receipt，连接 job、cell、manifest 与冻结 commit；
3. `results/ptb/<batch>/<cell>/`：operator 提交的 status、metrics、provenance、judges、
   日志尾部与 `task/` cards；
4. `doc/exp_protocol_iterations/<date>-round-NN.md`：跨 cells 的指标、人工 card 阅读、
   决策、改动与下一轮。

完整模型和原始 trace 留在共享 `data/ptb/results/`；Git 结果包不复制权重、二进制或
超过 2 MiB 的文件。所有分析必须保留 receipt、manifest、spec 和 result 路径。

## 九、manifest 中的 protocol 注入

每个正式 cell 使用 high-effort AWM scaffold，并声明：

```yaml
agent: claude_vertex_high_awm
agent_model: "claude-opus-5[1m]"
effort: high
context_tokens: 1000000
base_model: google/gemma-3-4b-pt
awm:
  sha: <protocol-variant-top-level-commit>
  paths:
    - awm/__init__.py
    - awm/cli.py
    - awm/paths.py
    - awm/sandbox.py
    - awm/exp_protocol
    - skills/exp_protocol
  setup: "--exp-protocol --tool claude"
```

launcher 必须把这些路径从指定 SHA materialize 到共享
`data/ptb/awm-checkouts/`，只读挂载到 `/home/ben/awm`。scientist 只得到
`exp_protocol`；meta skill、docs 和 WMA trees 必须被 launcher 拒绝。

## 十、预算与启动门

本线 GPU 额度固定为 `G=16`，由 `gangda_exp-protocol-evolve` subqueue 的
`slurm2-a3nodesetondem-[0-1]` 硬隔离。planner 保持足够的独立、安全 cells 已提交或
排队以使用该额度，但不跨到 WMA 节点，也不通过取消其他实验抢占容量。

开始 Round 00 前必须全部满足：

- high-effort AWM scaffold 门通过；
- exp_protocol CPU tests、PTB scaffold tests 和 launcher tests 全绿；
- `awm ptb check` 为零问题；
- queue operator 已在独立 checkout 中以 10–15 分钟周期运行；
- `gangda-slurm-queue --summary` 为 `OWNERSHIP OK`；
- manifest、spec、queue 与当前 branch 已 commit/push；
- registry 与本线 `.env` 均确认 `gangda_exp-protocol-evolve` 的 `G=16` 节点边界。

完成这些门只表示可以发射，不表示 candidate 可以晋升。
