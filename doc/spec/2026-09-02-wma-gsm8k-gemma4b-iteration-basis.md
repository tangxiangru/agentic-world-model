# WMA 自迭代基线:GSM8K 语料离线回放 / Gemma 4B held-out / 同合同上线

**日期**:2026-09-02
**状态**:生效;round-01 起的设计基线
**分支**:`gangda_wma_evolve`(叠在 `gangda-dev` 上)
**集成目标**:`gangda-dev`,绝不直接合并到 `main`
**控制规程**:`skills/wma_meta/SKILL.md`
**上游设计**:`doc/spec/2026-09-01-wma-v1-design.md` 定义了循环本身;本文把当前阶段的
合同、门槛和晋升规则冻结成可执行条款
**对照线**:`gangda_exp_protocol_evolve` 分支的
`doc/spec/2026-09-02-exp-protocol-gsm8k-gemma4b-iteration-basis.md`——同一 setup,
改进对象不同:那条线改 scientist 的规程,这条线改 WMA 的裁决

## 一、目标和边界

这条线改进的是 `skills/wma/`:让 WMA 对 scientist 提案的四层裁决更准、更校准——
在会 fail 的卡上提前说 L0/L1 no,在会被 reject / abandon 的卡上说 L3 no,同时不误杀
会被 adopt 的卡——最终换成 scientist 的 GPU 小时利用率。iteration agent 不是 WMA;
它让受不同 `skills/wma` commit 约束的 WMA 对同一批**结果已知**的提案出裁决,从账本和
手读的 verdict 中决定下一次**单一改动**。

- 当前阶段是**离线回放**:样本是历史卡,没有活的 scientist、没有 GPU,探测只有
  `static_check` / `data_probe`。上线(在线优化)是第二阶段,合同见第二节 B 表,
  门槛见第十节。
- 唯一迭代样本集是 `gsm8k-gemma-holdout-v1` 的 **train 侧**(post-train Qwen3-4B /
  Qwen3-1.7B / SmolLM3-3B 的 143 个 run,1580 张卡)的固定 300 抽样。
- **test 侧(post-train gemma-3-4b 的 50 个 run,450 张卡)是 held-out promotion gate**,
  只验证少量已由 train 侧选出的候选;不用它选择改动、换先验、写规则。它的 base
  model 与在线阶段相同,是离线能得到的最接近上线的检验。
- 一轮内除 `skills/wma` commit 外,样本集、后端、模型、effort、预算、mode 完全相同。
- `skills/wma_meta` 永不进入 WMA 的会话;WMA 出裁决时不改 `skills/wma`。
- 本线不改进 scientist 的规程:`skills/exp_protocol/` 只允许 WMA-aware 的改动
  (第 4b 步一类),纯规程改进归消融线;两条线暂不互相同步。
- 测量——verdict schema、打分规则、回放器的泄漏规则、样本集、`review.py` 里的
  prompt——不在轮内改;要改先在第十一节登记,或另写 spec。

## 二、冻结的实验合同

### A. 离线回放(当前阶段)

| 项 | 固定值 |
|---|---|
| 语料 | `results/exp-cards/gsm8k-gemma-holdout-v1`,jerry-dev 分支 `a8b12af`,只读;卡片抽自 HF `aisa-group/PostTrainBench-Trajectories@39d3fcd` |
| split 契约 | `splits/posttrainbench/gsm8k-gemma-holdout-v1.yaml`,train 143 / test 50 |
| 迭代集 | train 侧 `--sample 300 --seed 0`;(run, card) 集合指纹 `509ba0772ffc4fab`(排序后每行 `run_ref card_id` 的 sha256 前 16 位);与 round-00 逐对相同 |
| held-out 集 | test 侧 `--sample 200 --seed 0`,第一次晋升前构建一次(`heldout-00`),之后不变;若 round-01 计价后全量 450 张装得进一轮预算,构建时改为全量并在此处改数 |
| mode | `offline`;探测只有 `static_check` / `data_probe` |
| 地板 | `heuristic` 后端(`wma_skill: heuristic-priors`),round-00 已测 |
| 后端 / 模型 | `claude` / `claude-opus-5`;离线与在线同一个模型 |
| effort | `high`,由后端显式传 `--effort`(CLI 默认值就是 `high`),盖进 verdict;不继承本机 settings 的默认值(现为 `xhigh`) |
| 预算 | `cpu=5,gpu=0,wall=8,turns=30` |
| 每变体 pass 数 | baseline 2 个完整 pass;candidate 1 个;目标层的差距落在 baseline 两 pass 之差以内时补第 2 个 |
| initial baseline | `skills/wma` v0 修掉过时措辞、放平语气后的 commit `971ab3b`(`SKILL.md` sha `116b98b07d1a`);round-01 记录里再抄一遍 |

- 一个 out 目录 = 一个 (variant, backend, model, effort, pass),命名
  `round-NN-<label>` 与 `round-NN-<label>-b`。账本按 (wma_skill, backend, model, effort,
  mode) 分组;同一变体的两个 pass 目录放进同一次 `awm wma ledger` 调用得到合并估计,
  分开调用得到 `spread_L`。
- 一个 pass 的 300 条 verdict 必须出自同一个 `skills/wma` commit。pilot(`--limit 1`、
  `--limit 20`)的 verdict 可以续跑成正式 pass(回放器跳过已写的 verdict);pilot 之后
  改了 skill,该目录作废重建。
- `verdict` 文件写一次不改;重跑进新目录。

### B. 在线优化(第二阶段;与消融线的合同逐字相同,只加 WMA 行)

| 项 | 固定值 |
|---|---|
| iteration task | `gsm8k` |
| held-out task | `aime2025` |
| base model | `google/gemma-3-4b-pt` @ `cc012e0a6d0787b4adcc0fa2c4da74402494554d` |
| scientist model / effort / context / 时长 | `claude-opus-5[1m]` / `high` / 1,000,000 / `PTB_NUM_HOURS=10` |
| cell resources | 1 H100、16 CPU、128 GiB RAM、400 GiB scratch |
| minimum repeats | 每个 task × variant 至少 2 个正式、独立、有效 cell |
| scaffold | `claude_vertex_high_awm`——消融线的 high-effort 门也是本线的门 |
| scientist 规程 | `skills/exp_protocol` 固定在本线的一个 commit(含第 4b 步),整个在线阶段不变 |
| WMA backend / model / effort | `claude` / `claude-opus-5`(与离线相同)/ `high` |
| WMA mode / 预算 | `online`;`cpu=10,gpu=0,wall=15,turns=40` 【待定:第一轮是否开放 `gpu_min`】 |
| WMA history | train 侧语料只读挂载进 cell 【需要发射器支持,见第九节】 |
| 变体 | `skills/wma` 的 commit;scientist、规程、任务、模型、时长、资源全部相同 |

在线阶段的 variant 是 WMA 的 skill,不是 scientist 的规程;"有 WMA 对无 WMA"这个
问题由本线的第一轮在线 baseline 与一个无 WMA 对照臂回答(对照臂的构造见第九节),
不借用消融线的 cell——两条线的 `skills/exp_protocol` 已经岔开,拿来对照不干净。

## 三、Round 00(已完成)与 Round 01:模型驱动 baseline 的分布

**Round 00**(`doc/wma_iterations/2026-09-02-round-00.md`)只跑 heuristic 后端,
得到的是"常数回答"的基率,任何模型驱动的 skill 必须打赢的线:

| 集合 | n | L0_hit | L1_hit | L2_coverage(宽) | L3_hit | gpu_h_saved | gpu_h_wrongly_killed |
|---|---|---|---|---|---|---|---|
| 标准 300 | 300 | 0.907 | 0.623 | 0.489(0.083),可打分 184 | 0.418 | 1.17 | 1.20 |
| train 全量 | 1580 | 0.911 | 0.632 | 0.500(0.082) | 0.435 | 6.05 | 4.73 |

标准 300 的真值分布:execution completed 229 / killed 43 / failed 28;L1 为 no 的 113 张;
decision adopt 121 / reject 116 / abandon_line 50 / iterate 13。抽样与全量各层相差
< 0.02,标准集无偏。

**Round 01** 是第一个模型驱动的 baseline:v0 skill、`claude` 后端、第二节 A 表的全部
固定值。目的不是宣布改进,而是得到:

- 每层的 hit 对 round-00 基率;**`L0_recall_failed`**(28 张 failed 卡上说 no 的比例)
  与 **`L1_recall_invalid`**(113 张上说 no 的比例)——一个只会答 yes 的 WMA 已经拿到
  0.907 / 0.623,召回才是 skill 的贡献;
- L2 的 coverage、宽度和可打分数(184);
- L3 的 hit、`gpu_h_saved` 对 `gpu_h_wrongly_killed`;
- `basis` 的填充率:每层至少一条 basis、且指向的 evidence 有真实路径的比例;
- `n_leak_suspected`(必须为 0;非 0 逐条手查);
- `cost_usd_mean`、`cost_wall_min_mean`、平均 turns;
- baseline 两个 pass 之间每层的差 `spread_L`——以后所有比较的噪声底。

发射阶梯,每级有停止条件:

1. `--limit 1`:verdict 合法、`cost.usd` 与 `turns` 已盖、`access.outside` 为空、
   能引用 `history/` 里的路径。任一不满足,修接线再来,不计价。
2. `--limit 20`:`cost_usd_mean × 300 × 2` 落在用户批准的本轮上限 `$B` 内,
   `cost_wall_min_mean` 使一个 pass 能在一天内跑完;否则停下报数。
3. 全 300(pass a),随后 pass b。

iteration agent 对 round-01 至少手读 10 份 verdict:5 个 L3 命中、5 个 L3 失误,外加
28 张 failed 卡上 L0 失误的全部(若少于 10 张则全读)。分析写入
`doc/wma_iterations/2026-09-02-round-01.md`。根据明确的 verdict、basis 或缺失的
evidence,先写证据,再产生一个单变量 candidate commit。

## 四、常规离线迭代轮

每轮比较 baseline 与 1 或 2 个 candidate;两个 candidate 必须改不同的地方,各自相对
baseline 只改一件事。API 回放没有 GPU 排队,两个 candidate 可以同时跑。

每轮执行:

1. 冻结 variant:提交 `skills/wma`,记录 commit 与 `SKILL.md` 的 sha256(账本分组键)。
2. **先预注册**再跑:在 round record 写明改动、它针对的层、预期方向,以及什么结果
   算失败。
3. `awm wma replay --corpus <corpus> --out <root>/round-NN-<label> --side train --sample 300 --seed 0 --backend claude --model <model> --budget cpu=5,gpu=0,wall=8,turns=30`;
   构建后核对集合指纹;`awm wma ledger <root>/round-NN-<label>`。
4. 按层读账本,对 baseline 两个 pass 的均值,并附 `spread_L`;L2 报 coverage 时并排报
   宽度与可打分数;L3 报 hit 时并排报 `gpu_h_saved` / `gpu_h_wrongly_killed`。
5. 每个 variant 手读 10 份 verdict(5 中 5 错),外加所有 `leak_suspected` 的。
6. 先写 round record,再做下一次单一改动;record 与改动放在同一 commit。

可改范围只有:

- `skills/wma/SKILL.md` 的一条层定义、basis 规则、探测手册条目或先验;被账本反驳的
  先验是替换,不是软化;
- `skills/wma/verdict.example.json`,保持 `tests/test_wma_skill_files.py` 通过;
- `awm/wma/backends.py` 的 heuristic 后端,只为保基线诚实,必须仍是固定明示的先验。

无改动是合法结论,也必须记录。skill 学的是"类",不是"例":任何指名道姓的 run 或卡片
不得写进 skill。

## 五、held-out 候选池

"更好"按 pass 判断,不按单张 verdict 判断。candidate 进入候选池必须同时满足:

1. 完整的 300 pass(`--limit` 截短的轮次只能定方向,不能入池);
2. 预注册的目标层上,candidate 高于 baseline 两 pass 的均值,幅度超过
   `max(spread_L, 0.03)`;只有 1 个 pass 且差距在这个线以内时,先补第 2 个 pass 再判;
3. 其他任一层不低于 baseline 均值超过 `max(spread_L, 0.03)`;
4. L2 的 coverage 提升只有在宽度不增时才算;
5. `gpu_h_wrongly_killed` 不增加;candidate 的 pass 里 `n_leak_suspected = 0`;
6. `cost_usd_mean` 不超过 baseline 的 1.5 倍,否则 record 里必须写明这笔交换;
7. 改动及其理由已写进对应 round record。

候选池按**目标层增益 ÷ `max(spread_L, 0.03)`** 排序(跨层可比的效应量),同分时按
`L3_hit`。每个 promotion window 默认积累 2 个合格 candidate 或经过 3 轮,先到者触发;
只把排名最高的最多 2 个送入 held-out。这样避免每轮查看 test 侧。

## 六、test 侧 held-out gate

触发后:

- 第一次触发时构建 `heldout-00`(第二节 A 表的 held-out 集),之后每次复用;
- 对 current baseline 与最多 2 个入围 candidate 各跑 1 个 pass,后端、模型、effort、
  预算与 train 侧完全相同;
- 报每层的 hit / recall 与可打分数,不做任何 skill 改动。

held-out 只回答"train 侧看到的改善是否泛化到另一个 base model"。candidate 晋升需:

1. 目标层上 candidate 仍高于 baseline(方向不反转);
2. 其他层不低于 baseline 超过 train 侧的 `max(spread_L, 0.03)`;
3. `gpu_h_wrongly_killed` 不增加,`n_leak_suspected = 0`。

不通过则保留旧 baseline。**不得读 test 侧 verdict 的 `basis` / `evidence` / 探测记录来
设计下一候选**——比消融线"不读 AIME 失败模式"更严,因为 verdict 的正文就是失败模式;
只允许看账本的汇总行。一次 held-out 无论通过与否都要记录,不重跑到满意为止;只有
`errors.jsonl` 里的基础设施失败可以对失败样本续跑。晋升的 candidate 成为 baseline 时
补齐它在 train 侧的第 2 个 pass。

## 七、角色、目录与异步

离线阶段没有 operator:iteration agent 自己发射回放、自己收割。

| 角色 | 唯一写入范围 |
|---|---|
| iteration agent(本线) | `skills/wma/`、`skills/wma_meta/`(轮外)、`awm/wma/`(仅第十一节登记的测量改动)、`doc/wma_iterations/`、`doc/spec/` |
| 回放输出 | `/data2/gangda/hv/wma-replay/`,机器本地,不入 git |
| 语料 | jerry-dev worktree,只读,不动 |

- 回放在后台跑;iteration agent 在 verdict 陆续落地时可以先读,但只在 pass 完整后做
  比较结论。
- 回放可续跑:被超时或基础设施打断的 pass 用同一命令续跑,已写的 verdict 跳过;
  `errors.jsonl` 里的样本续跑一次,仍失败的记为该 pass 的缺口并写进 record 的 `n`。
- 不删除、不改写任何 verdict;要"重来"就新建目录。

## 八、结果与 provenance

四层记录必须同时保留:

1. `skills/wma` 的 commit:变体的身份,commit 与 `SKILL.md` sha256 都写进 record;
2. out 目录:`samples.jsonl`(集合指纹)、`_truth/`、`_history/`、全部
   `exp-NN.verdict.json`(自带 `wma_skill`、`backend`、`mode`、`cost`、`access`、
   `leak_suspected`)、`errors.jsonl`;
3. `doc/wma_iterations/<date>-round-NN.md`:账本表(完整贴入,因为 out 目录不在 git)、
   指纹、手读记录、决策、改动与下一轮;
4. 语料与 split 契约:jerry-dev 的 commit 与 `splits/` 文件,已在 git 里 pin 住。

所有分析必须保留 out 目录路径、指纹、variant commit 和 record 路径。

## 九、在线阶段的 WMA 注入(第二阶段;先登记,不执行)

每个正式 cell 使用 `claude_vertex_high_awm`,manifest 在消融线的 `awm` 块之上加:

```yaml
awm:
  sha: <wma-variant-top-level-commit>
  paths:
    - awm/__init__.py
    - awm/cli.py
    - awm/paths.py
    - awm/sandbox.py
    - awm/exp_protocol
    - awm/wma
    - skills/exp_protocol
    - skills/wma
  setup: "--exp-protocol --tool claude"
wma:
  backend: claude
  model: <与离线相同>
  effort: high
  budget: "cpu=10,gpu=0,wall=15,turns=40"
  history: <train 侧语料的只读挂载路径>
```

上线前必须解决、且每一项都要写进本文的修订或单独 spec:

1. 发射器的 `AWM_FORBIDDEN_TREES` 现在按名拒绝 `skills/wma` 与 `awm/wma`。需要一种
   "WMA-aware study"模式允许这两棵树,同时继续拒绝 `skills/wma_meta`、
   `skills/exp_protocol_meta`、`doc`。
2. `history` 挂载:WMA 在 cell 里读 train 侧语料的方式(只读、路径固定、在 `--add-dir`
   栅栏之内)。
3. 无 WMA 对照臂:同一 `skills/exp_protocol` commit 去掉第 4b 步。现成的做法是把 4b 段
   搬到 `skills/wma/scientist_section.md`,由 `awm exp_protocol install --with-wma`
   装船时追加——用户 2026-09-02 决定暂不做;上线前重新决定。
4. WMA 子进程的 effort 与 model 必须显式传入并盖进 verdict(第十一节第 1 条)。
5. scientist 能读到挂载里的 `skills/wma/`(含先验)。是否接受,上线前定。

scientist 只得到 `exp_protocol`(含 4b)与调用 `awm wma review` 的权限;meta skill 和
docs 仍由发射器拒绝。

## 十、预算与启动门

**离线**:本 spec 不自行花钱。每轮 API 上限 `$B` 由用户在 round record 里批准;
`--limit 20` 计价后超出 `$B` 的 pass 不跑。

开始 round-01 前必须全部满足:

- 过时措辞已修、语气已放平并提交(`skills/wma/SKILL.md` 的 reconcile 与 Forbidden 一节,
  `skills/wma_meta/SKILL.md` 的成本列名、`--budget` 示例与 Never 条目),baseline commit
  由此冻结;
- 第十一节第 1–3 条已落地并有测试;
- wma 与 exp_protocol 的 CPU 测试、ruff 全绿;
- round-01 目录已构建,集合指纹 = `509ba0772ffc4fab`;
- 用户已确认模型与 `$B`。

**上线**:除消融线第十节的全部门(scaffold、tests、`awm ptb check`、operator、
`OWNERSHIP OK`、GPU 额度 `G`)外,本线另需:

1. train 侧:baseline skill 的 `L3_hit` 与 `L0_recall_failed` / `L1_recall_invalid`
   打赢 heuristic 地板超过 `max(spread_L, 0.03)`;
2. `L0_recall_failed` 与 `L1_recall_invalid` 达到 0.8(设计文档 §4.4 的建议线);
3. held-out(test 侧)上前两条不塌;
4. 第九节五项全部解决。

设计文档 §4.4 原写"显著优于查表基线";查表基线没有实现,本阶段不补,§4.4 已改为
对 heuristic 地板。上线后第一轮**只跑 baseline skill**(≥3 cell)加对照臂,先看探测层
的信息价值和成本,再谈变体。

## 十一、本 spec 登记的测量侧改动

这些改动碰的是测量,不是 skill,所以在这里登记而不是在轮内做;每条都要先加测试。

1. **已做**:`CommandBackend` 显式传 `--effort <level>`(claude:`--effort`;codex:
   `-c model_reasoning_effort=`),并把 `model`、`effort` 盖进 verdict;账本分组键改为
   (wma_skill, backend, model, effort, mode)。
2. **已做**:账本新增 `L0_recall_failed`、`L1_recall_invalid`、`n_L2_scorable` 三列;
   `n_reconciled` 改名 `n_scored`。
3. **已做**:`replay` 写出并打印集合指纹(`samples.sha`,对排序后的 `run_ref card_id`
   行做 sha256,与目录路径无关);加 `--jobs N` 并行 review。round-00 与 round-01 的
   指纹都是 `509ba0772ffc4fab`。
4. `review.py::build_prompt` 在本阶段冻结;改它等于换测量。
5. 不做:查表基线不补进回放器;设计文档 §4.4 的门槛改为对 heuristic 地板。
6. **已做**(round-01 第一份真裁决暴露的三件事):`L2_effect.direction` 允许 `flat`
   (打包 / baseline 卡预期无变化;打分只用区间,不受影响);无效的 verdict 文件移到
   `exp-NN.verdict.json.rejected`(带测得的成本与栅栏结果),样本回到待审,账本不读它
   但 `awm wma ledger` 汇报其数量与花费;`wma_skill`、`backend`、`mode`、`model`、
   `effort`、`issued_at`、`card_id` 一律由 harness 按实际写入,agent 自填的值不算数
   (第一份裁决把示例文件里的占位符原样抄了进去)。
7. **已做**(同一份裁决的第二次):栅栏只把**真实存在**的栅外路径算作越界——sed/awk 的
   正则字面量(`/^result:/,/^conclusion:/p`、`/1.7B/`)和打错的路径读不到任何东西,
   而 glob 会展开后逐个判断;stream-json transcript 完整保存为
   `exp-NN.transcript[.tag].jsonl`,供手读和栅栏规则改动后重扫。

完成这些门只表示可以发射 round-01,不表示 candidate 可以晋升。
