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
  Qwen3-1.7B / SmolLM3-3B)里 **scientist 为 claude-opus-5 / claude-fable-5 /
  claude-opus-4-8 / claude-opus-4-7 的 31 个 run 的全部 313 张卡**(用户 2026-09-02 定:
  只回放强模型的 trial;不抽样)。
- **test 侧同一批 agent 的 11 个 run、100 张卡(post-train gemma-3-4b)是 held-out
  promotion gate**,只验证少量已由 train 侧选出的候选;不用它选择改动、换先验、写规则。
  它的 base model 与在线阶段相同,是离线能得到的最接近上线的检验。
- **WMA 不得凭 scientist 的身份预测。** 过滤按 split 文件里的 run id 做
  (`run_ref = "r-" + sha256(run_id)[:8]`),session、history、truth 里只有不透明的
  `run_ref`;已核对语料卡片里的 `claude-opus-5` 只出现在 `provenance.extractor`(抽卡的
  模型),`openai` 只出现在数据集名 `openai/gsm8k`。skill 明文禁止从身份推断。
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
| 迭代集 | train 侧 `--agents 'claude-(opus-5\|fable-5\|opus-4-8\|opus-4-7)'`,不抽样:31 run / 313 卡(completed 241 / killed 34 / failed 38;adopt 165 / reject 79 / abandon 59 / iterate 10);集合指纹见 round-00s 记录。旧的 300 抽样(指纹 `509ba0772ffc4fab`,全体 agent)只用于 round-00 与两次计价 |
| held-out 集 | test 侧同一 `--agents` 过滤,不抽样:11 run / 100 卡;第一次晋升前构建一次(`heldout-00`),之后不变 |
| mode | `offline`;探测只有 `static_check` / `data_probe` |
| 地板 | `heuristic` 后端(`wma_skill: heuristic-priors`),round-00 已测 |
| 后端 / 模型 | `claude` / `claude-opus-5`;离线与在线同一个模型(2026-09-02 先改成 fable-5-1 计了 20 份的价,又由用户定回 opus-5) |
| effort | `high`,由后端显式传 `--effort`(也是 CLI 默认值),盖进 verdict;不继承本机 settings 的默认值(现为 `xhigh`) |
| 预算 | `cpu=5,gpu=0,wall=8,turns=30` |
| 每变体 pass 数 | baseline 2 个完整 pass;candidate 1 个;目标层的差距落在 baseline 两 pass 之差以内时补第 2 个 |
| initial baseline | **skill v0.2** `47ec0b6`(sha `176f0a464986`,对整个 `skills/wma/` 目录):类型 × 档位手册版。v0.1(`971ab3b`,sha `116b98b07d1a`,类型无关)保留为对照臂 |

- 一个 out 目录 = 一个 (variant, backend, model, effort, pass),命名
  `round-NN-a` / `round-NN-b`(baseline 的两个 pass)与 `round-NN-<label>`(candidate)。
  `round-01/` 只装 opus-5/high 的接线验证,不是一个 pass。账本按 (wma_skill, backend, model, effort,
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

**Round 01** 的早期 Fable 5.1 / medium 计价结果只保留为历史接线证据，不是当前合同的
baseline pass。当前模型驱动 baseline 使用 v0 skill、`claude` 后端、第二节 A 表的全部
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
4. L2 的 coverage 提升只有在 `L2_width_over_noise`(区间宽度 ÷ 该卡评测 n 的噪声底)不增时才算——
   宽度本身随评测 n 变,不能直接比;
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

## 九、在线阶段的 WMA 注入(第二阶段)

每个正式 cell 使用 `claude_vertex_high_awm`。Scientist 的公共 checkout 只包含
`exp_protocol`、薄 `awm wma` 请求客户端和必要 CLI；WMA runtime、skill、历史由 PTB host
启动的独立 sidecar 私有挂载，绝不进入 scientist container。manifest 在 `awm` 块之外加:

```yaml
awm:
  sha: <wma-variant-top-level-commit>
  paths:
    - awm/__init__.py
    - awm/cli.py
    - awm/paths.py
    - awm/sandbox.py
    - awm/exp_protocol
    - skills/exp_protocol
  setup: "--exp-protocol --tool claude"
wma:
  sha: <wma-variant-top-level-commit>
  backend: claude
  model: claude-opus-5
  effort: high
  mode: online
  budget: "cpu=10,gpu=0,wall=15,turns=40"
  history: <train 侧语料的只读挂载路径>
```

在线装船必须满足:

1. 公共 checkout 继续拒绝 `skills/wma`、`awm/wma`、两个 meta skill 与 `doc`;只增加
   `awm/wma_client.py`。私有 checkout 只挂进 sidecar container。
2. `history` 只读挂载到 sidecar 的 `/history`,不挂进 scientist container。
3. **无 WMA 对照臂(已实现,2026-09-02)**:对照 cell 装**同一个**公共 checkout、同一份
   含 4b 步的 `skills/exp_protocol`,只是 manifest 不写 `wma` 块,于是 host 不起 sidecar。
   sidecar 启动时会先建 `<task>/.wma/requests`;客户端 `awm wma review` / `status` 发现
   这个目录不存在就回答 "no world-model agent is attached to this cell; no verdict will
   come",返回 0,不入队、不建目录;规程告诉 scientist 这句就是全部答案,继续第 5 步。
   两臂 scientist 看到的字节逐一相同。**已硬化(fork 侧):**`run_task.sh` 在启动
   scientist 之前同步 `mkdir -p task/.wma/requests`,消除 sidecar 进程起慢时被薄客户端
   错认成无 WMA 对照的竞态；adapter 测试锁住这个顺序。
   另一条路——把 WMA 装进 scientist 沙箱、用 `awm_wma.json` 开关——已实现过但被 sidecar
   设计取代,留在 tag `wip-online-in-sandbox` 供查。
4. WMA 子进程的 effort 与 model 必须显式传入并盖进 verdict(第十一节第 1 条)。
5. Scientist 只能写 review request、读 status/verdict;不能读取 WMA skill、先验、历史或
   sidecar checkout。完整 WMA transcript 写到 result-private 目录,不落在 scientist 的
   `memory/cards`;WMA 只接受已 lock 的 card。

Scientist 只得到 `exp_protocol`(含 4b)与调用 `awm wma review --background` 的权限;
sidecar 异步处理批量 request,meta skill 和 docs 仍由发射器拒绝。

## 十、预算与启动门

**离线**:本 spec 不自行花钱。本机的 `claude` CLI 走 claude.ai OAuth(Max 订阅),
verdict 里的 `cost.usd` 是 CLI 报的**影子价**,不是账单——它用来比变体之间的相对成本,
真正的约束是订阅的用量窗口。每轮的用量上限 `$B`(影子价计)由用户在 round record 里
批准;`--limit 20` 计价后超出 `$B` 的 pass 不跑。

开始 round-01 前必须全部满足:

- 过时措辞已修、语气已放平并提交(`skills/wma/SKILL.md` 的 reconcile 与 Forbidden 一节,
  `skills/wma_meta/SKILL.md` 的成本列名、`--budget` 示例与 Never 条目),baseline commit
  由此冻结;
- 第十一节第 1–3 条已落地并有测试;
- wma 与 exp_protocol 的 CPU 测试、ruff 全绿;
- round-01 目录已构建,集合指纹 = `509ba0772ffc4fab`;
- 用户已确认模型与 `$B`。

**用户 2026-09-02 的决定:离线阶段快速收尾,更新完 skill 就转在线;下面三条离线门槛
不再作为上线前提。** 理由:回放 session 没有工作区(脚本、数据、config),第一档探针在
离线做不了,L1 只能靠猜——而 skill v0.2 的价值正在于探针;这一层只有在线才测得到。
离线分析的不足记在 GitHub issue(见 round-01 记录)。

**上线**:除消融线第十节的全部门(scaffold、tests、`awm ptb check`、operator、
`OWNERSHIP OK`、GPU 额度 `G`)外,本线另需(前三条已由上述决定豁免,保留作记录):

1. train 侧:baseline skill 的 `L3_hit` 与 `L0_recall_failed` / `L1_recall_invalid`
   打赢 heuristic 地板超过 `max(spread_L, 0.03)`;
2. `L0_recall_failed` 与 `L1_recall_invalid` 达到 0.8(设计文档 §4.4 的建议线);
3. held-out(test 侧)上前两条不塌;
4. 第九节五项全部通过装船和 sidecar 隔离测试。

设计文档 §4.4 原写"显著优于查表基线";查表基线没有实现,本阶段不补,§4.4 已改为
对 heuristic 地板。上线后第一轮**只跑 baseline skill**加对照臂,先看探测层的信息价值
和成本,再谈变体。

**用户指令(2026-09-02,覆盖本节与 §九 第 4 项里"先冒烟、先跑 ≥3 cell"的措辞)**:
不走 pilot;16 卡必须始终用满,queue 里永远保留待排的 cells;全部启动、边跑边分析,
发现不成立就撤下尚未开始的 cells。第一轮在线批次据此为两臂各 16 cell、四个 immutable
manifest 同时排队,见 `doc/spec/2026-09-02-wma-round01-online-gsm8k-gemma4b.md`;
sidecar 的冒烟由第一波的 8 个 wma cell 承担。Fable 以 commit 落实 manifest 与
`queue.yaml`,仍不跑 `sbatch` / `scancel` / `reconcile --apply`,不写 `results/ptb`。

**用户指令(2026-09-02 21:0x UTC,复制设定)**:每个 manifest `replication:
{settings: 1, repeats: 2}`,不再是 8。理由是用户的判断:同一 setting 8 次重复太多,
容量应给不同的 setting。已在跑的 8 重复批次(首波与 v2)按原样跑完并按原 cohort 报告;
尚未启动的八 cell v3 扩展已撤回、以 2+2 重排(round-01 在线 spec 同日 21:0x 一节)。
此后任何新 manifest(含 Round 02 的候选与 held-out)都按 2+2 配对写;要多于 2 个
观测,写成多个不同 setting 的 manifest,而不是加 repeats。

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
8. **已做**(20 份计价 pass 暴露的):栅栏再排除两类假阳性——`sed 's/x//'` 抽出的 `//`
   (是根目录)和 CLI 自己的溢出目录 `~/.claude/projects/<cwd 改写>/…/tool-results/`
   (agent 回读自己过长的工具输出);`awm wma rescan <dirs>` 用保存的 transcript 重算
   `access` / `leak_suspected`,verdict 其余字段不动(栅栏是 harness 规则,不是 agent 的
   判断,规则改了就重算而不是重买);`direction` 的同义写法(`none`、`n/a`、`unchanged`…)
   在校验前折成 `flat`。
9. **已做**(2026-09-02,用户批准的第二批):verdict 带 `change_types`(C1–C18,C1 分
   a/b),`awm wma ledger --by type|family` 按 WMA 的分类或卡的 family 切片;**L1 只在
   completed 卡上打分**(killed 是 scientist 的时间决策,failed 是 L0 的失误);
   `noise_floor(n)` 与新列 `L2_width_over_noise`;把 base model 原样打包的 baseline 卡
   在 L3 上 unscorable;`skill_sha` 改为覆盖 `skills/wma/` 全部文件(手册是 skill 的一部分);
   回放器 `--agents REGEX`(经 split 文件的 run id 过滤,session 不带身份)与 `filter.json`。
10. **已做**(在线隔离):scientist-side `awm wma` 是只写原子 request、只读 status 的薄
   client;真正的 `awm/wma`、`skills/wma` 与 history 在独立 sidecar container。WMA
   prompt 使用私有 skill 的绝对路径,Claude 以空 setting sources 启动;review 不在
   scientist session 建立 `skills/wma` symlink,完整 transcript 写入 scientist container
   不可见的 `wma_private/`。该改动不改变 verdict schema 或 scorer。
11. **已做**(2026-09-02,用户"边跑边分析"指令):operator 的 `awm ptb reconcile` 对每个
   RUNNING / COMPLETING 的 job 增加 `peek` 动作——把共享结果卷上已经存在的
   `wma_sidecar.log`、`wma_private/*.jsonl`(gzip)与 `solve_out.txt` 尾部快照到
   `results/ptb/<batch>/<cell>.inflight/`(附 `peek.json`:sidecar 日志最后一行、
   transcript 清单、stdout 行数),每轮原地覆盖,cell 结束后由收割删除。收割本身也开始
   带走 `wma_sidecar.log` 与 gzip 的 transcripts,`status.json` 记 `sidecar_log` 与
   `transcripts`。scientist 的 task 树在 job 结束前是节点本地的,快照里没有它;cards 与
   verdict 仍只在收割后可读。不改 verdict schema、scorer 或样本契约。
12. **已做**(首个在线 inflight window):`review.py` 的 post-hoc guard 把
   `result.execution: not_run` 视为合法的 pre-launch sentinel，而不是已经观察到的结果；
   `completed` / `failed` / `killed` 或已有 `conclusion.decision` 仍拒绝。首波 jobs
   `90558`、`90560`、`90561` 的 scientist 都在 lock 前后预填 `not_run`，旧 runtime
   因而错误回答 `already has a result`，另外五张留空的同类卡正常产出 verdict。该修复
   只改 measurement/runtime，不改 prompt、schema、scorer、skill 或历史语料，并有
   `test_review_accepts_the_not_run_prelaunch_sentinel` 回归测试。

13. **已做**(2026-09-02,首波 26 份在线 verdict 的读时审计):**自测量卡**——
   `evaluation.comparator.ref == base_model` 且 comparator 没有 `value` / `path`(这张卡就是
   comparator 的第一次读数)、parent 是 base_model、family 为 `other` / `decode-config`——
   L1 只看有没有读数(不训练就没有 checkpoint 可要求),L2 与 L3 unscorable(对自己的
   delta 恒为 0,决定说的是下一步)。此前 scorer 给这类卡记了两次假的 L1 miss
   (r02/r08 exp-01,无 checkpoint)和两次假的 L2 miss(绝对值写进 delta 槽);首波按
   committed transcript 重算:L1 12/15 → 14/15,L2 9/15(含 4 张假分)→ 7/12(4 above,
   1 below),与 operator 手工排除后的表一致。带 comparator path 的 decode-config 卡
   (r07 exp-01,greedy vs stock)仍照常打分。只改读时 scorer,不改 verdict、skill、样本。

完成这些门只表示可以发射 round-01,不表示 candidate 可以晋升。
