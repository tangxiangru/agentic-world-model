# PostTrainBench Slurm 运行手册

本文是 `gsm8k-claude5-1m-batch1` 的站点运行手册。实验语义和验收标准见
[`../spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md`](../spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md)，配置归属见
[`posttrainbench_configuration_ownership.md`](posttrainbench_configuration_ownership.md)。

## 1. 固定站点边界

正式 PTB 作业只允许使用：

```text
slurm2-a3nodesetondem-0
slurm2-a3nodesetondem-1
slurm2-a3nodesetondem-2
slurm2-a3nodesetondem-3
```

集群永久配置源是 `gs://slurm-slurm2834bd/slurm2-files`：

- `config.yaml` 的 `slurm_conf_tpl` 包含 `AccountingStorageTRES=gres/gpu`；
- `partition_configs/ptb-a3.yaml` 定义四节点、`OverSubscribe=NO` 的 `ptb-a3` partition；
- `cgroup_conf_tpl` 使用 `ConstrainDevices=yes`；
- 修改前的配置备份为 `backups/config.yaml.pre-ptb-20260830`。

站点路径、节点、reservation、secret 和认证只放在 gitignored
`third_party/PostTrainBench/.env`。当前 reservation 是 `robtang-a3`；提交器显式传入
它，从而可以消费已为该用户保留的节点。

控制面验收：

```bash
scontrol show config | grep -E 'SelectType|AccountingStorageTRES|GresTypes|TaskPlugin'
scontrol show partition ptb-a3 -o
scontrol show node slurm2-a3nodesetondem-0 -o
```

四节点均必须出现 `CfgTRES=...gres/gpu=8`，partition 必须合计
`gres/gpu=32`，且 `OverSubscribe=NO`。

## 2. 固定软件与数据资产

- Agent SIF：`/rmeng_data/robtang/ptb-containers/opus_5.sif`
  - SHA-256：`35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759`
  - Claude Code：2.1.219；正式 batch 关闭 auto-update。
- Official judge SIF：`/rmeng_data/robtang/ptb-containers/gpt_5_5.sif`
  - SHA-256：`765cae4e7893171e935c89fba27fa9bff93bb884b139a7f319a9fbfbbbded117`
- Final-eval SIF：`/rmeng_data/robtang/ptb-containers/vllm_debug.sif`
- HF cache：顶层 `data/ptb/hf`。
- 结果：顶层 `data/ptb/results`。
- 每 job local scratch：`/mnt/localssd/posttrainbench/$USER/$SLURM_JOB_ID`。

所有大文件和 provider 证据在 `/data` 下，已被顶层 `.gitignore` 排除。历史 trace
不挂载进 agent sandbox。

## 3. 一次性验证阶梯

在仓库根目录：

```bash
# G0：单测、schema 和六格 dry-run
uv run --extra dev pytest -q
uv run awm ptb dry-run
uv run awm ptb check --before-context-gate

# G1–G3：选择一台没有旧式 exclusive job 的目标节点，严格顺序运行
cd third_party/PostTrainBench
bash src/commit_utils/slurm/run_gates.sh g1 slurm2-a3nodesetondem-0
bash src/commit_utils/slurm/run_gates.sh g2 slurm2-a3nodesetondem-0
bash src/commit_utils/slurm/run_gates.sh g3 slurm2-a3nodesetondem-0
cd ../..

# G4：真实 GPU/container/provider smoke；b1 代表 Fable，b6 代表 Opus
uv run awm ptb context-smoke --cell b1 --cell b6
squeue -u "$USER"
uv run awm ptb check
```

`context-smoke` 会同时验证一张 H100 的容器可见性、task/judge/auth assets、真实 Claude
provider 调用，以及响应中的 `modelUsage.contextWindow`；它写入 gitignored
`data/ptb/context-validation/*.json`。裸 `claude-opus-5` 在 2026-08-30 实测只有
200k，正式配置必须是 `claude-opus-5[1m]`。

当前已知 G4 blocker：`sercan-v1` 的继承组织策略拒绝
`publishers/anthropic/models/claude-fable-5:predict`。管理员解除 deny 后必须重新运行
b1 smoke；不得手写成功记录或把 Fable 静默换成其他 provider/model。

## 4. Pilot 与正式提交

两次提交都要求顶层和 PTB submodule worktree 干净，并在 receipt 中冻结两个 commit。

```bash
# G5：B6 形状，1h agent budget，仍跑全部 official judges + full GSM8K eval
uv run awm ptb submit --pilot

# 查看命令打印出的 receipt
uv run awm ptb status data/ptb/batches/gsm8k-claude5-1m-batch1/pilot-*.json
uv run awm ptb audit-receipt data/ptb/batches/gsm8k-claude5-1m-batch1/pilot-*.json

# G6：只有 pilot audit 为 0 issue 才提交六个 10h cell
uv run awm ptb submit
uv run awm ptb status data/ptb/batches/gsm8k-claude5-1m-batch1/formal-*.json
uv run awm ptb audit-receipt data/ptb/batches/gsm8k-claude5-1m-batch1/formal-*.json
```

正式提交是一次冻结、连续提交六个独立 Slurm job。launcher 拒绝同类 receipt 的第二次
提交；如果中途失败，receipt 会保留已经提交的 job 和失败 cell，不能直接重跑造成重复。

每个 job 请求 `1 H100 + 16 CPU + 128G RAM`，agent budget 10h，Slurm walltime
另留 12h 给排队的 official judge lock、四个 judge 和 full evaluation。官方 ChatGPT
judge 共享认证文件时，完整 judge phase 通过
`/rmeng_data/robtang/ptb-locks/official-judges.lock` 串行化 refresh-token 写入；agent
本身使用无可写 token 文件的 Vertex/GCE metadata ADC。

## 5. Research judges

只有 formal receipt 的每个 cell 都通过 official audit 后，才运行研究 profile：

```bash
uv run awm ptb research-judges \
  data/ptb/batches/gsm8k-claude5-1m-batch1/formal-*.json
```

它为每个不可变 result 提交一个 CPU-only Slurm job，使用
`claude-opus-5[1m]`、xhigh、Vertex 和完全相同的四份 judge prompt。输出为独立的
`judgement_claude_*_rerun.json`，不会覆盖 official canonical verdict。

## 6. 取消、故障与恢复

查看 receipt 后，只取消其中明确列出的 job：

```bash
python3 -c 'import json,sys; print(" ".join(j["job_id"] for j in json.load(open(sys.argv[1]))["jobs"]))' RECEIPT
scancel JOB_ID [JOB_ID ...]
```

不要用用户名或 partition 作为宽泛 `scancel` 目标。取消后结果保持原位，scratch 默认由
job trap 清理。一次失败不从 pilot workspace 或半成品 checkpoint resume；先审计
`logs/slurm/`、result 的 `output.log/error.log/runtime_provenance.json`，修复基础设施后用新
batch/run index 明确提交。Gemma 三格一旦揭盲就算消费 holdout，不能把修复后针对结果的
迭代继续标成同一个 untouched holdout。
