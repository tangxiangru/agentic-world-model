# WMA Round 01(在线):baseline skill v0.2 对无 WMA 对照,GSM8K / gemma-3-4b-pt

**日期**:2026-09-02
**状态**:生效;第一轮在线批次,六个 immutable manifest 异步排队
**line**:`gangda_wma_evolve`
**subqueue**:`gangda_wma_evolve`(`slurm2-a3nodesetondem-[2-3]`,16 GPU)
**基础合同**:`doc/spec/2026-09-02-wma-gsm8k-gemma4b-iteration-basis.md`(§二 B、§九)
**沙箱边界**:`doc/reference/wma_online_sidecar.md`

## 要回答的问题

在线阶段的第一轮不测 skill 变体,只测 **WMA 存在本身**:把 skill v0.2(hash
`176f0a464986`,opus-5 / high,sidecar 私有挂载)接到 scientist 规程的第 4b 步,
相对于一字不差的同一规程、同一 checkout、只是**没有 sidecar** 的对照,

1. scientist 的 GSM8K 最终 accuracy 是否变化(方向、幅度、相对两臂各自的 spread);
2. 在线 verdict 的账本(L0–L3 对 scientist 自己卡片的结果打分)长什么样——这是离线
   回放做不到的一层:探针(第一档 verifier)只有在线才有工作区可探(issue #22)。

离线 baseline 的数字(`doc/wma_iterations/2026-09-02-round-01.md`)是这一轮的先验,
不是它的对照。

## 两臂

| 项 | `wma` 臂 | `ctl` 臂 |
|---|---|---|
| scaffold | `claude_vertex_high_awm` | 同 |
| `awm.sha` | `e8a8599bce74df232e20e2fd3aa1226e7e04e999` | 同 |
| `awm.paths` | `awm/__init__.py, awm/cli.py, awm/paths.py, awm/sandbox.py, awm/wma_client.py, awm/exp_protocol, skills/exp_protocol`(= `EXP_PROTOCOL_SHIP`) | 同 |
| `awm.setup` | `--exp-protocol --tool claude` | 同 |
| `wma` 块 | `sha` 同上;backend `claude`、model `claude-opus-5`、effort `high`、mode `online`、budget `cpu=10,gpu=0,wall=15,turns=40`;history 见下 | **无**(host 不起 sidecar) |
| scientist 看到的字节 | 同一公共 checkout、同一 `skills/exp_protocol`(含 4b) | 逐一相同 |
| 4b 的回答 | verdict 文件 | "no world-model agent is attached to this cell; no verdict will come"(`b9ca64c`) |
| 其余 | task、base model 与 revision、scientist 模型/effort/context、10 h、judge、容器、PTB commit 逐字段相同 | 同 |

`e8a8599` 是提交本 spec 时分支的顶端:它含 sidecar(`a18485a`)、客户端的对照臂语义
(`b9ca64c`)、skill v0.2(`47ec0b6`)。`git archive` 该 commit 的 `EXP_PROTOCOL_SHIP`
是公共 checkout,`WMA_PRIVATE_SHIP`(`awm/wma`、`skills/wma` 等)是只挂进 sidecar 的
私有 checkout;receipt 分别记 `awm_checkouts` 与 `wma_private_checkouts`。

### history:只挂语料的 train 侧

在线 base model 是 gemma-3-4b-pt,它就是语料 `gsm8k-gemma-holdout-v1` 的 **test 侧**
(50 runs / 450 cards,held-out)。所以 `wma.history` 只能指向 train 侧(143 runs /
1580 cards),永不指向语料根或 `test/`。manifest 写死:

```
/rmeng_data/robtang/wma-history/gsm8k-gemma-holdout-v1-a8b12af/train
```

它需要 operator 在集群上一次性准备(语料在 `origin/jerry-dev` 的 `a8b12af`,只有 14 MB):

```bash
git fetch origin jerry-dev
dest=/rmeng_data/robtang/wma-history/gsm8k-gemma-holdout-v1-a8b12af
mkdir -p "$dest"
git archive a8b12af results/exp-cards/gsm8k-gemma-holdout-v1/train \
  | tar -x --strip-components=3 -C "$dest"
chmod -R a-w "$dest"
ls "$dest/train" | wc -l      # 143 个 r-xxxxxxxx 目录,没有 test
```

若想放在别处,在声明路径上做符号链接,manifest 不改。目录不存在时 `awm ptb check`
会拒收 `wma` 臂的两个 manifest,operator 写 `blocked.md` 后每轮 reconcile 都会重试,
目录一到就自动提交;`ctl` 臂不受影响。receipt 的 `environment` 里会留下
`POST_TRAIN_BENCH_WMA_HISTORY`,分析时核对它就是这条路径。

## 六个 immutable batch,异步排队

| manifest | 臂 | cells | `run_index` | 全局观测 |
|---|---|---|---|---|
| `wma-gsm8k-gemma4b-high-r01-wma-x8` | wma | `w01r01..w01r08` | 1 | wma 1–8 |
| `wma-gsm8k-gemma4b-high-r01-ctl-x8` | ctl | `c01r01..c01r08` | 1 | ctl 1–8 |
| `wma-gsm8k-gemma4b-high-r01-wma-b-x8` | wma | `w02r01..w02r08` | 2 | wma 9–16 |
| `wma-gsm8k-gemma4b-high-r01-ctl-b-x8` | ctl | `c02r01..c02r08` | 2 | ctl 9–16 |
| `wma-gsm8k-gemma4b-high-r01-wma-c-x8` | wma | `w03r01..w03r08` | 3 | wma 17–24 |
| `wma-gsm8k-gemma4b-high-r01-ctl-c-x8` | ctl | `c03r01..c03r08` | 3 | ctl 17–24 |

每个 manifest 内 `replicate: 1..8`(schema 要求),`run_index` 区分三个 immutable batch。
队列顺序就是上表:第一波 8 wma + 8 ctl 占满 16 卡,后四项是待排缓冲,任何 cell 结束
就有下一个顶上。六批都不依赖任何结果。前四批提交时没有 formal 分数被观察过；
`-c` 两批于 2026-09-02 14:07 UTC 冻结，当时第一波仍全部 RUNNING、四个 manifest 的
`complete=0`，所以也不是结果驱动的追加。前两批是不可替换的 core-16(两臂各 8)；
后四批是两组配对 precision extension，不能替换 core 里失败或低分的 cell。报告依次给
core-16、all-32 与 all-48 sensitivity，任何结论都不按分数选择 cohort。

`-b` / `-c` 两批的 `replicate` 在汇总时分别对应各臂的全局 9–16 / 17–24；合并的前提是
receipt 指向同一 PTB commit(`62203e4`)与同一 judge 容器。

## 2026-09-02 14:2x UTC:撤回 v1 缓冲,以 v2 重排

首波 8 个 wma cell 里有 3 个(`w01r03`、`w01r05`、`w01r06`,jobs 90558/90560/90561)的
scientist 在 lock 前后把 `result.execution: not_run` 预填进卡片——这是 schema 合法的
pre-launch sentinel——而 `e8a8599` 冻结的私有 runtime 把任何非空 execution 当作已观察到
的结果,拒答 "already has a result"。这三个 cell 的 exp-01 因此没有 verdict;operator 在
`bf87dfb` 修了 guard(基础合同 §十一 第 12 条,只改 runtime,不改 prompt / schema /
scorer / skill / 语料),skill hash 仍是 `176f0a464986`。

按用户指令第 3 条,四个尚未开始的缓冲批次(`-b`、`-c`,32 个 PENDING job)全部撤回,
以 v2 重排:

| v1(撤回,全部 PENDING) | v2 | 改动 |
|---|---|---|
| `wma-b-x8-v1` jobs 90572–90579 | `wma-b-x8-v2` | `wma.sha` → `bf87dfb`;其余逐字节相同 |
| `ctl-b-x8-v1` jobs 90580–90587 | `ctl-b-x8-v2` | 无改动;只为与 wma-b-v2 同时入队 |
| `wma-c-x8-v1` jobs 90588–90595 | `wma-c-x8-v2` | `wma.sha` → `bf87dfb` |
| `ctl-c-x8-v1` jobs 90596–90603 | `ctl-c-x8-v2` | 无改动;同上 |

ctl 也重排是为了保住配对:Slurm 按提交先后出队,只撤 wma 会让第二波变成 16 个对照、
第三波 16 个 WMA。v2 的公共 digest 与 v1、与首波完全相同(`63aef0f1a9da…`);wma 臂私有
digest 由 `8c3d40a84716…` 变为 `f79c75a6c3a8…`,差异只有 `review.py` 的 guard。观测槽位
(各臂 9–16、17–24)与 `run_index` 不变。RUNNING 的首波不动。

**分析规则(现在声明,不等分数)**:wma 臂报告两种口径——**as attached**(装了 sidecar
的全部 cell,主口径,对应"WMA 存在本身"的问题)与 **as answered**(至少一张卡拿到
verdict 的 cell)。首波三个被拒答的 cell 属于前者、不属于后者;它们的后续卡片若不再预填
`not_run` 仍会得到 verdict,以收割后的 `.wma/responses` 为准。账本按 `wma_skill`
(hash 不变)合并 v1 首波与 v2 缓冲。

## 2026-09-02 19:08 UTC:probe-basis 修复与独立 v3 扩展

首波产生 26 个 review transcript,旧 validator 接受 21 个、把 5 个移到
`.rejected`。五者都至少有一个 `levels.*.basis` 引用了该 verdict 已记录的
`probes[].id`;旧 schema 只允许 `evidence[].id`,尽管 probe 的设计就是记录它改变了哪一层。
`34535c7` 把合法集合放宽为 `evidence[].id ∪ probes[].id`;预测、scorer、prompt、
`skills/wma/` 与 skill hash `176f0a464986` 均不变。26 个 transcript payload 在新
validator 下 26/26 通过;另有 leak flag 的 verdict 在 clean ledger 中仍照常排除。

修复完成前,v2 已有 26 个 job 启动、6 个仍 PENDING,所以它们不撤回,也不把 v3 伪装成
替代结果。新增两个各 8+8 的独立配对扩展:

| manifest pair | cells | `run_index` | 全局观测 | 私有 runtime |
|---|---|---:|---|---|
| `wma-b-x8-v3` / `ctl-b-x8-v3` | `w04r01..08` / `c04r01..08` | 4 | 25–32 | WMA=`34535c7`;control 无 sidecar |
| `wma-c-x8-v3` / `ctl-c-x8-v3` | `w05r01..08` / `c05r01..08` | 5 | 33–40 | 同上 |

公共 scientist checkout 仍为 `e8a8599`;task、model revision、Opus 5 high、1M context、
10 h、容器与 judge 逐字段不变。v3 的 ledger 单列 runtime 接受率,再与 v2 合并
scientific prediction;PTB as-attached 仍按 WMA/control 配对报告。

同时发现 receipt 虽记录正确的
`POST_TRAIN_BENCH_SLURM_NODELIST=slurm2-a3nodesetondem-[2-3]`,部分已启动 v2 job 的
Slurm `ReqNodeList` 却变成 `(null)` 并落到 scope 外节点。RUNNING job 按用户边界不取消。
从本 checkpoint 起,formal job 全部先以 HOLD 提交;ownership registration 后、atomic
release 前逐 job 反查 Slurm `ReqNodeList`,展开后的节点集合必须与 subqueue 完全相同。
为空、查询失败或集合不等时整批保持 HOLD 并写
`routing_verification_failed`,不得运行。这是 receipt 意图之外的 scheduler-state gate。

## 2026-09-02 20:3x UTC:ctl-c v2 的五个 PENDING cell 经路由门重试

operator 发现 v2 里尚未启动的五个 ctl-c job(`c03r04..08`,90633–90637)在 Slurm 里
已丢失 `ReqNodeList`,一旦调度就会落到子队列之外,于是在 PENDING 状态取消(`2b6bfeb`),
并用发射器的 **retry receipt**(`formal-retry1-…`,jobs 90786–90790)在同一 manifest 内
重试这五个 cell,全部通过路由门、PENDING 且 `ReqNodeList=[2-3]`(`5f2189e`)。其余 v2
job 已在 RUNNING,不动。我在同一时间写的替代 manifest `ctl-c-x5-v2b` 与之重复,在提交
前撤回、未曾提交到 Slurm;重试与原 job 永不重复计数。这五个 cell 的 wma-c-x8-v2 配对已
在跑,时间上晚于配对——分析时按 `wma_skill` 合并,并在 core-16 / all-32 / all-48 之外
单列"配对同波"的敏感性口径。

## 2026-09-02 21:1x UTC:用户把 repeats 定为 4,撤回八 cell 的 v3 扩展

用户看过五个批次 × 两臂共 80 cell 的清单后决定:**`repeats: 8` 太多;先说 2,随即
因为同设置的方差大改为 4——从现在起每个 manifest 都是
`replication: {settings: 1, repeats: 4}`**(每个候选 4 个 cell);GPU 容量用来跑不同的
setting,而不是同一 setting 的更多重复。落实:

- 四个八 cell 的 v3 manifest(`wma/ctl-b-x8-v3`、`wma/ctl-c-x8-v3`,jobs 90676–90707)
  原计划在全部 PENDING、一个都没启动时撤回。21:35 UTC 的 operator 监控确认这 32 个
  job 仍在安全 PENDING,取消计划尚未执行。为满足“任何时候至少 8 个 PENDING”的硬约束,
  它们暂时保持 `want: submitted`;只在 Round 02 候选全部冻结且可提交时做一次原子切换。
  切换 preview 必须证明操作后安全 PENDING ≥ 24,这样 30 分钟检查间隔内即使一整波
  16 GPU 开始,仍保留 ≥ 8。若其中某个先开始,不取消 RUNNING,按实际 cohort 分析。
- 同一 setting 只保留一对 4+4:`wma-x4-v3` / `ctl-x4-v3`(`w04r01..04` / `c04r01..04`,
  `run_index 4`,全局观测 25–28,私有 runtime 仍为 `34535c7`,skill hash 不变)。它承担
  runtime 修复的在线接受率 cohort;v2 的 26 份 transcript 在新 validator 下已 26/26
  重放通过。中间版本的 2+2 pair(`wma/ctl-x2-v3`,commit `afcb86a`)在 operator 提交前
  就被本节替换,从未进过 Slurm。因为 `x4-v3` 的 r01..04 是暂存旧 x8 inventory 的
  子集,替代 pair 在切换前保持 `want: cancelled` 且无 receipt,不能与旧 batch 同时提交。
  若旧 r01..04 先启动,它们直接成为该 cohort,不再提交同 cell 的 x4 duplicate。
- 43 个 RUNNING cell(首波 16 + v2 27)不动;5 个 ctl-c 重试 cell(90786–90790)不动。
  不能先落到“撤回后 5 + 8 = 13”的中间状态:13 虽在检查瞬间高于 8,但一波启动后会
  变成 0。旧 32 个 pending 是替代候选 submit-ready 前的安全库存。

**对"16 卡永远用满"的影响(明说)**:要在 repeats 4 下填满 16 卡,
一波 = 4 个 setting;"一轮一处 skill 改动"的协作规则只给一个候选,其余容量的去向
(并行多个单改动候选、各 4+4;或换任务/模型的 breadth)由用户决定,Fable 不自行
编造 setting。现有 v3 保证过渡期间不会空卡;上文"目标 `PENDING ≈ 16`"从此按
repeats 4 解释:硬下限 8、操作 guard 24、补货目标至少 32,补货单位是 4+4 的配对
manifest。

## 2026-09-02 21:3x UTC:用户选定容量去向——并行多个单改动候选(Round 02 的形状,预注册)

被问到 repeats 4 之下 16 卡的剩余容量往哪放,用户选了**并行多个单改动候选**
(而不是换任务的 breadth,也不是空着等单候选)。这改变了"一轮一处 skill 改动"的
落实方式,但不改变归因原则:**每个候选 = v0.2 + 恰好一处 `skills/wma` 编辑**,各自
独立与 v0.2 比;一轮只晋升一个编辑,落选的编辑若也通过,下一轮叠在新基线上再测。

### 候选池(4 个上场,每个 4 cell = 一整波 16 卡)

| 候选 | 唯一编辑 | 针对的证据 | 主指标(对 v0.2) | 预注册证伪 |
|---|---|---|---|---|
| **A `format-floor`** | 手册 §4 加一行"format floor, not capability"(收窄到诊断出地板后的**第一次**干预;上界锚定该 checkpoint 的已知能力与 C2 行的 headroom 份额;**不写 gemma 的具体分数**)+ SKILL.md L2 步一句交叉引用 | WINDOW 04 的 4/5 L2 落空都在区间上方,机制相同 | exp-01 预测落在地板带的比例;first-format-fix 卡的 L2 coverage | (i) ctl 基线不在地板 → 行错;(ii) v0.2 exp-01 已在带内 → 行冗余;(iii) 非地板 comparator 的卡 coverage 低于 v0.2 → 过度泛化,收窄 |
| **A+B `format-floor + width-cap`** | A 之上再加:L2 区间宽于 §5 地板 3× 时须在 `basis` 引 precedent 或 probe,否则收到该类型先验的宽度 | WINDOW 03:exp-02 区间 0.30–0.40 宽 ≈ 10× 地板;WINDOW 04 判定宽度问题被位置误差支配,须在 A 之后再测 | 对 **A**(不是对 v0.2):`L2_width_over_noise` 下降且 L2 coverage 不低于 A 超过 A 自身 spread | coverage 掉 → B 在 A 之上仍不成立,撤 |
| **C `probe-before-fail`** | L0/L1 答 `no` 必须有一条 `changed: L0|L1` 的 probe(static_check 证明失败路径会被走到,或 unit_test/dry_run 复现);没有则 confidence ≤ 0.5 且 note 标 `unprobed` | `w01r07/exp-02` 真 miss:Liger 不可 import → 误推 Trainer 会崩,没探 | L0/L1 的 false-`no` 率下降;recall 不劣化超过 0.05;每 verdict 的 probe 数上升 | recall 掉、或成本 > 1.5× v0.2 → 撤 |
| **D `checkpoint-precondition`** | C3/C4 卡若计划只存 1 个 checkpoint(放弃 C5),`preconditions` 必须含 `save_steps` 计划(tier 3,标分钟),L3 由 `yes` 改 `defer` 并点名该前提 | 手册 C5 = 唯一严格单变量对照(±15 pp);WINDOW 05 要先看 transcript 里 C5 被建议/被采纳的比例 | C3/C4 verdict 含 checkpoint 前提的比例;scientist 端 `save_steps`/C5 卡的出现率对 ctl 与 v0.2;PTB 最终分不劣化 | 若 WINDOW 05 显示 C5 已被建议且采纳 ≥ 50 %,D 换成 **E `history-cite`**(C3/C4 的 L2 必须引 ≥1 条同类型同 n 的 history precedent 或写明"无") |

选择规则(WINDOW 05 之前定死):A 与 A+B 固定上场;C、D/E 按 terminal 账本里各自
针对的落空数排序,取两个。每个候选一个 commit,线性链
`A → A+B → (撤 A+B,上 C) → (撤 C,上 D/E) → (head 回到 v0.2 + manifest + queue + 记录)`;
每个 `wma.sha` 的 `git archive` 恰好是 v0.2 + 该候选,skill hash 各不相同,ledger 按
`wma_skill` 分组即得四组;head 始终是当前基线。runtime 代码(`awm/wma`)四个候选
一致(`34535c7` 之后无 runtime 改动),不允许候选夹带 runtime 变更。

### 参照与 cohort

- 同波 v0.2 参照:`w04r01..04`(runtime `34535c7`);合并参照:首波 8 + v2 16 的 v0.2
  cell(as-attached)。PTB 护栏用 core-16 + `w04/c04`。
- 不加新的 ctl cell:ctl 已有 8 + 16 + 4。
- 每个候选 4 cell,每 cell 3–7 个 verdict,账本每组 ≈ 12–28 张卡;PTB 每组 n=4,只做
  护栏,不做主结论。

### 晋升规则(预注册)

一个候选晋升为 v0.3 当且仅当:主指标对参照的改善超过参照自身 spread;L0/L1 recall
不劣化超过 0.05;`n_leak_suspected` = 0;每 verdict 成本 ≤ 1.5× v0.2;PTB as-attached
均值不低于 v0.2 超过 v0.2 的 spread。多个通过时取主指标增益最大者;其余通过者下一轮
叠在 v0.3 上重测(归因仍是每次晋升一个编辑)。都不通过 → v0.2 不动,按各自证伪条款
收窄或撤。

### 时间与队列

候选的 skill 文本在 **WINDOW 05(首波 terminal validator 证据)之后**才 commit——这是
与 operator 的约定,不因并行而提前;patch、manifest 草稿先在本地备好,证据一到即
提交。旧 v3 的 32 个安全 PENDING 在此之前不撤;候选 16 cell 与 `w04/c04` 8 cell
submit-ready 后,同一次 reconcile preview 才允许把旧 v3 标成取消并加入新任务。切换
前还要按 receipt/job state 去重:旧 r01..04 若仍 PENDING,取消后激活 x4;若已启动则直接
复用且 x4 保持不提交。无重复时切换后 PENDING 约 29(候选 16 + `w04/c04` 8 + ctl-c
重试 5),足以满足 guard,不会出现 13→0 的空窗。cell id:`w06..w09`
(`run_index 6..9`),不复用 `w05`。

## 用户指令(2026-09-02,覆盖基础合同 §十"上线后第一轮只跑 ≥3 cell"的条款)

1. **不走 pilot。** manifest 不带 `pilot` 块,queue entry 一律 `pilot: null`,直接以
   formal cells 发射。sidecar 的"在真实 cell 里冒烟"(基础合同 §九 第 4 项)由第一波
   的 8 个 wma cell 承担:`wma_sidecar.log` 与 `.wma/sidecar_status.json` 会说明它起没起来。
2. **16 卡必须始终用满。** 只要有独立、已验证的 manifest 就立即排队;queue 里永远保留
   待排的 cells(`RUNNING + PENDING ≥ 16`,目标 `PENDING ≈ 16`)。不得为了凑数编造依赖
   未完成结果的实验。
3. **全部启动,边跑边分析。** 分析异步进行,不等批次跑完。发现某个设计不成立
   (例如 sidecar 在所有 wma cell 里都没起来),把它尚未开始的 cells 从 queue 撤下
   (`want: cancelled` 只取消 PENDING;RUNNING 由 operator 保留到自然结束),修好后以
   新 manifest 重排。
4. 本指令由分支负责 agent(Fable)以 commit 落实;`queue.yaml` 直接提交到
   `gangda_wma_evolve`,不另开 PR。Fable 仍然不跑 `sbatch` / `scancel` /
   `reconcile --apply`,不写 `results/ptb`。

## 分析

**PTB 层**:两臂各自的 accuracy mean / range / stderr;matched delta = wma − ctl,
与两臂 spread 及 Batch 2 观察到的同设置重复间距(0.15–7 pp)一起报。比较只在
receipt 的 PTB commit、judge 容器、`awm_checkouts.digest` 两臂一致时成立。任何 cell
先经 validator(四个 canonical judge verdict + `audit_receipt`),不合格的列出排除理由;
Slurm `COMPLETED` 不是科学完成。

**账本层**(只有 wma 臂有):`awm wma ledger results/ptb/<batch>/*/task [--by type]`,
按 `wma_skill / backend / model / effort / mode` 分组;读 `L0_recall_failed`、
`L1_recall_invalid`、`L2_width_over_noise`、`L3_hit`、`gpu_h_saved` /
`gpu_h_wrongly_killed`、`n_leak_suspected`、成本列。truth 来自 scientist 自己的卡片,
不来自 PTB 分数。在线的第一档探针是否被用、`probes[].changed` 是否与 L1 的命中相关,
是这一轮最想看的一列。

**规程层**:`awm exp_protocol collect results/ptb/<batch>/*/task --csv` 两臂都有卡片;
比较卡片数、`pitfalls_cost_h`、adopt/reject 的分布——WMA 若有用,应先在这里看到
(更少的无效卡、更早的 abandon),再在 accuracy 上看到。

**人工样本**:每个分析窗口读固定数量的命中/落空 verdict 与其私有 transcript(只作
迭代协作者,内容不给 scientist,也不用 held-out 细节设计候选)。

**比较结论的门槛**:每臂 ≥ 8 个有效 cell 才做 supported / contradicted / inconclusive
的判断;之前只报告分布与 sidecar 健康。

**边跑边分析**:operator 每轮 reconcile 顺带把 RUNNING cell 在共享卷上已有的
`wma_sidecar.log`、`solve_out.txt` 尾部与 `wma_private/` 目录清单快照进
`results/ptb/<batch>/<cell>.inflight/`(基础合同 §十一 第 10 条);Fable 据此在
cell 结束前判断 sidecar 是否工作、scientist 是否调用了 4b。

## 停止条件

- `OWNERSHIP FAIL` → 停止一切提交。
- 第一波任一 wma cell 的 `wma_sidecar.log` 显示 sidecar 未启动或持续失败 → 撤下两个
  wma manifest 里仍 PENDING 的 cells,修复后以新 manifest(`-v2`)重排;ctl 臂照跑。
- 任一 verdict 被 `audit_receipt` 判为暴露私有 skill / transcript 给 scientist → 同上,
  且该 cell 排除。
- 不取消 RUNNING。

## 下一轮的入口

Round 02 的第一个候选已在离线记录里预告:L2 区间宽度封顶在 k× 噪声底,除非引用证据。
它只有在本轮的在线账本给出 `L2_width_over_noise` 与 `L2_coverage` 之后才能落地;
候选 manifest 会以新的 `wma.sha`(skill 改动的 commit)与同一 `awm.sha` 排在本轮后面,
对照臂共用本轮的 ctl。
