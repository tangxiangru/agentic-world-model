# WMA Round 01(在线):baseline skill v0.2 对无 WMA 对照,GSM8K / gemma-3-4b-pt

**日期**:2026-09-02
**状态**:生效;第一轮在线批次,四个 immutable manifest 同时排队
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

## 四个 immutable batch,一次排队

| manifest | 臂 | cells | `run_index` | 全局观测 |
|---|---|---|---|---|
| `wma-gsm8k-gemma4b-high-r01-wma-x8` | wma | `w01r01..w01r08` | 1 | wma 1–8 |
| `wma-gsm8k-gemma4b-high-r01-ctl-x8` | ctl | `c01r01..c01r08` | 1 | ctl 1–8 |
| `wma-gsm8k-gemma4b-high-r01-wma-b-x8` | wma | `w02r01..w02r08` | 2 | wma 9–16 |
| `wma-gsm8k-gemma4b-high-r01-ctl-b-x8` | ctl | `c02r01..c02r08` | 2 | ctl 9–16 |

每个 manifest 内 `replicate: 1..8`(schema 要求),`run_index` 区分两个 immutable batch。
队列顺序就是上表:第一波 8 wma + 8 ctl 占满 16 卡,后两项是待排缓冲,任何 cell 结束
就有下一个顶上。四批都不依赖任何结果,提交时没有任何 formal 分数被观察过。前两批
是不可替换的 core-16(两臂各 8);后两批是配对的 precision extension,不能替换 core
里失败或低分的 cell。报告同时给 core-16 与 all-32。

`-b` 两批的 `replicate` 在汇总时对应各臂的全局 9–16;两批合并的前提是 receipt 指向
同一 PTB commit(`2af3ccd`)与同一 judge 容器。

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
