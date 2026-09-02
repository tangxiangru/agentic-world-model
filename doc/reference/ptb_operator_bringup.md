# H100 操作员的一次性准备清单

**给谁**:在 H100 登录机上为分支 `gangda_exp_protocol_evolve` 建操作员 worktree 的 agent。**日期**:2026-09-02。角色与日常循环见 `ptb_operator_runbook.md`;这里只是第一次把环境搭起来,并把规划者写 manifest 需要的事实报回来。**这一步不提交任何 Slurm 作业**,队列文件现在是空的。

## 一、先决条件(不是操作员做的)

- fork PR `tangxiangru/PostTrainBench#1`(`claude_vertex_max_awm` scaffold 与只读挂载开关)合并后,规划者会把顶层的 submodule 指针更新到合并 commit。指针更新前,下面第 2 步检出的 submodule 里还没有这个 scaffold,第 8 步会报 `missing agent asset`,属预期。

## 二、按顺序做

0. **先把主线 checkout 的分支名改过来**。主线 2026-09-02 由 `gangda_trial_0828` 改名为 `gangda-dev`,GitHub 上旧名已不存在;集群上的主线 checkout 还叫旧名,先改,否则它下次 push 会把旧名重新创建到远端:
   ```bash
   cd <主线 checkout>
   git fetch --prune origin
   git branch -m gangda_trial_0828 gangda-dev && git branch -u origin/gangda-dev gangda-dev
   ```
   batch1、batch2 的 manifest、receipt 与 job 名里保留旧名,不改;要重试那些旧批次时临时开一个叫 `gangda_trial_0828` 的本地分支即可。
1. **建 worktree**,与主线 checkout 分开,放在数据盘上:
   ```bash
   cd <主线 checkout>            # 主线 `gangda-dev` 的那个目录,历史批次是以它的旧名 `gangda_trial_0828` 提交的
   git fetch origin
   git worktree add /data/<user>/worktrees/agentic-world-model_exp-protocol-evolve gangda_exp_protocol_evolve
   cd /data/<user>/worktrees/agentic-world-model_exp-protocol-evolve
   ```
2. **检出 PTB submodule**:
   ```bash
   git submodule update --init third_party/PostTrainBench
   ```
3. **接上同一个 data 卷**(与主线相同的目标,这样 HF cache、容器、receipt、`awm-checkouts` 都共享):
   ```bash
   ln -s "$(readlink -f <主线 checkout>/data)" data
   df -h data/ | tail -1                      # 必须是计算节点也能看到的共享文件系统
   ```
4. **复制 PTB 的 `.env`**(gitignored,含站点配置与凭据路径):
   ```bash
   cp <主线 checkout>/third_party/PostTrainBench/.env third_party/PostTrainBench/.env
   chmod 600 third_party/PostTrainBench/.env
   ```
5. **确认 GitHub 推送能力**,操作员每轮都要 push 到 `tangxiangru/agentic-world-model`:
   ```bash
   gh auth status
   git push --dry-run origin gangda_exp_protocol_evolve
   ```
6. **跑测试**。`.env` 的 `HF_HOME` 等于 `<repo>/data/ptb/hf` 时应全绿:
   ```bash
   uv run python -m pytest -q 2>&1 | tail -3
   ```
7. **操作员工具冒烟**(空队列应答 `nothing to do`;`sacct`、`scancel`、`gangda-slurm-queue` 必须在 PATH 上):
   ```bash
   uv run awm ptb reconcile
   command -v sacct scancel gangda-slurm-queue
   gangda-slurm-queue --summary
   ```
8. **发射器站点门**,用主线已有的 manifest 检查容器 digest、HF snapshot、分区与节点(指针更新前 `_awm` 的 manifest 还没有,先用 batch2 的):
   ```bash
   uv run awm ptb check experiments/posttrainbench/gsm8k-aime2025-opus5-selected16x2-batch2.yaml
   ```
9. **写操作员会话的 `CLAUDE.local.md`**(gitignored):把 `doc/reference/ptb_operator_claude_local.md` 里的模板复制到 worktree 根目录,`<branch>` 填 `gangda_exp_protocol_evolve`。

## 三、报回来的事实

规划者写第一轮 manifest 需要这些,写在回复里,不要提交到仓库:

| 项 | 怎么得到 |
|---|---|
| worktree 绝对路径、`data` 软链指向、`df` 显示的文件系统 | 第 1、3 步 |
| `.env` 里的非密值:`HF_HOME`、`POST_TRAIN_BENCH_RESULTS_DIR`、`POST_TRAIN_BENCH_CONTAINERS_DIR`、`POST_TRAIN_BENCH_SLURM_PARTITION`、`POST_TRAIN_BENCH_SLURM_NODELIST`、`POST_TRAIN_BENCH_SLURM_RESERVATION`、`POST_TRAIN_BENCH_SLURM_OWNERSHIP_REGISTRY`、`POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT` | `grep -E '^(HF_HOME\|POST_TRAIN_BENCH_(RESULTS_DIR\|CONTAINERS_DIR\|SLURM_))' third_party/PostTrainBench/.env` |
| `data/ptb/context-validation/claude-opus-5-1m-max.json` 是否存在 | `ls -la data/ptb/context-validation/` |
| 第 6、7、8 步的完整输出 | 原样粘贴 |
| 当前队列负载 | `gangda-slurm-queue --summary` 的输出 |
| 推送用的 GitHub 账号 | `gh auth status` |

**不要做的**:不要 `awm ptb submit`;不要改 `experiments/`、`doc/`、`skills/`、`awm/`;不要在主线 checkout 里操作;不要把 `.env` 或任何凭据提交进 git。

## 四、之后

规划者收到报告后:更新 submodule 指针,写第一轮的 spec 与 manifest,往 `experiments/posttrainbench/queue.yaml` 加第一项(`pilot: first`)。从那时起操作员按 runbook 第三节每 10 到 15 分钟跑一轮。
