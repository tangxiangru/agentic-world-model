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
