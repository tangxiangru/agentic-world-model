# PTB 操作员 runbook:两个 agent 用 git 协作

**用途**:集群上的操作员 agent 照此执行;本机的规划者 agent 照此理解结果何时、以何种形式回来。适用于任何用 `awm ptb` 发射器的线:实验规程迭代、WMA。**日期**:2026-09-02 **状态**:生效

## 一、角色与路径所有权

| 角色 | 在哪 | 只写这些路径 |
|---|---|---|
| 规划者 | 本机 worktree | `experiments/posttrainbench/*.yaml`、`experiments/posttrainbench/queue.yaml`、`doc/`、`skills/`、`awm/` |
| 操作员 | 集群登录机的 worktree,同一分支 | `results/ptb/**` |

两边每次写之前 `git pull --rebase`,写完立刻 push。路径不相交,分支不需要合并。规划者永远不跑 `sbatch`;操作员永远不改 manifest、队列、规程或代码。

## 二、四种文件就是协议

1. **manifest** `experiments/posttrainbench/<batch>.yaml`:跑什么。提交后不可变;发射器把顶层 commit、PTB commit 与 manifest 冻结进 receipt。每格若用 `_awm` scaffold,`awm` 块写明要挂进沙箱的 commit、路径与 `awm sandbox setup` 的参数。
2. **队列** `experiments/posttrainbench/queue.yaml`:期望状态。有序列表,每项 `manifest`、`want: submitted | held | cancelled`、可选 `pilot: first`、一句 `why`。`held` 只提交并登记 ownership，保留 `PENDING(JobHeldUser)`；改成 `submitted` 前必须通过 ownership 与冻结节点 release gate；`cancelled` 撤回还没开跑的格。这是规划者唯一的操作杆。只有用户明确授权时，单个 `want: submitted` 项可带 `release_override: {allow_shared_reservation: true, authorized_by, authorized_at, reason}`：它只豁免 reservation 节点集合相等检查，ownership、receipt frozen nodes、逐 job `PENDING(JobHeldUser)` 与 `ReqNodeList` 仍强制，授权与实际 reservation/frozen nodes 写回 receipt；绝不能设成全局环境开关。
3. **receipt** `results/ptb/<batch>/<kind>-<ts>.json`:发射器写在 `data/ptb/batches/` 的原件由操作员复制进来,取消记录追加在 `cancellations`。它是唯一的归属凭证:**只有它列出的 job ID 可以被取消**,这是 `AGENTS.md` 的硬规则。
4. **结果包** `results/ptb/<batch>/<cell>/`,布局见 `results/ptb/README.md`;`results/ptb/ops-log.md` 每个动作一行。

## 三、操作员的一轮

在集群登录机上,分支的 worktree 里,每 10 到 15 分钟一轮(Claude Code 会话用 `/loop 15m`):

```bash
git pull --rebase                      # 1. 失败就停下报告,不要 --force
awm ptb reconcile                      # 2. 看它打算做什么
awm ptb reconcile --apply              # 3. 做:先 submit,再复制 receipt、取消、收割
git add results/ptb && git commit -m "ops: <粘贴 reconcile 打印的动作行>"
git push                               # 4. 被拒就 pull --rebase 再 push
```

`--apply` 的顺序是硬的:submit 在前,因为发射器要求工作树干净,而 submit 只写 gitignored 的 `data/`;收割在后,它会弄脏工作树,留给这一轮的 commit。若一轮里既有收割又有新的 submit,收割完提交后**再跑一次** `reconcile --apply`,submit 才会发生。

什么都不判断:
- `check` 或 `submit` 被发射器拒绝 → 写 `results/ptb/<batch>/blocked.md`,提交,继续下一项。修 manifest 是规划者的事。
- job 失败 → 照常收割,`status.json` 记 `slurm_state` 与校验问题;**不重试**。重试是规划者往队列里加一项。
- `gangda-slurm-queue` 报 `OWNERSHIP FAIL` → 停止一切 submit,把输出贴进 `results/ptb/ops-log.md`,通知用户。
- 队列里标 `cancelled` 但 job 已在 RUNNING → 不取消,`reconcile` 打印 `wait`;规划者想杀正在跑的格必须找用户。

## 四、规划者的一轮

```bash
git pull
awm exp_protocol collect results/ptb/<batch>/*/task --csv     # 读数
# 读三张卡;先写 doc/exp_protocol_iterations/<date>-round-NN.md,再改规程或队列
git add experiments doc skills awm && git commit && git push
```

替换实验:把旧项改成 `want: cancelled`(写 `why`),新 manifest 加成新项。只有 PENDING 的格会被撤回。

## 五、超额与额度

设这条线的 GPU 额度为 G，用户指定 held pending floor 为 8。规划者保证至少 8 个独立、已冻结且
科学上需要的格处于 `PENDING(JobHeldUser)`；低了先补 held receipt。held 不因 `OWNERSHIP FAIL` 自动
release。Release 前重新验证 ownership、hold reason 与 receipt 的冻结 `ReqNodeList`；原生子队列
隔离未恢复时保持 held。发现错误终结照常收割，再按有效重复数和 matched-arm 平衡决定是否用新
immutable manifest 补跑。

## 六、集群上的一次性准备

```bash
git worktree add /data/<user>/worktrees/agentic-world-model_<line> <branch>   # 与主线 checkout 分开
cd 那个目录 && git submodule update --init third_party/PostTrainBench
ln -s <数据卷> data                                                             # 与主线相同的 data 卷
cp <主线 checkout>/third_party/PostTrainBench/.env third_party/PostTrainBench/.env
uv run awm ptb check experiments/posttrainbench/<manifest>.yaml                # 0 issue 才算就绪
gh auth status                                                                  # 操作员要能 push
```

操作员会话的 `CLAUDE.local.md` 模板见 `doc/reference/ptb_operator_claude_local.md`。
