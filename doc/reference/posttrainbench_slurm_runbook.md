# PostTrainBench Slurm 运行手册

本文是 `gsm8k-opus5-4x4-batch1` 的站点运行手册。实验语义和验收标准见
[`../spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md`](../spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md)，配置归属见
[`posttrainbench_configuration_ownership.md`](posttrainbench_configuration_ownership.md)。

## 0. 共享账号归属与命名

当前集群账号由多人共用。任何时候都不能根据 `squeue -u "$USER"` 或 `UserId` 把 job
认作本项目；这两个字段只能说明它使用了共享账号。

本批唯一允许的 ownership 是：

```text
branch = gangda_trial_0828
spec   = doc/spec/2026-08-30-ptb-gpu-slicing-and-gsm8k-batch1.md
batch  = gsm8k-opus5-4x4-batch1
```

所有新 PTB Slurm job，包括 G1–G4、pilot、十六个 formal cell 和 research judges，名称必须
以 `gangda_trial_0828.` 开头。正式名称示例：

```text
gangda_trial_0828.ptb.gsm8k-opus5-4x4-batch1.b01.formal.r1
gangda_trial_0828.ptb.gsm8k-opus5-4x4-batch1.b16.formal.r1
```

顶层 launcher 会验证当前分支，生成带 branch/batch/cell/purpose/run 的 job name 和结果
suffix；pilot/formal/research job 还会在 receipt 中保存 ownership、job name 和 job ID，
gate/smoke 则写入带相同 identity 的独立验证目录。分支不一致、detached HEAD、名称没有
分支前缀或正式 receipt 无法落盘时必须 fail closed，不提交或不 release。

判断与取消流程：

1. 先从 `data/ptb/batches/gsm8k-opus5-4x4-batch1/*.json` 找 receipt；
2. 再核对 `scontrol show job JOB_ID -o` 中的 `JobName`、`Command`、`WorkDir`、`StdOut`；
3. 五项 identity（branch/spec/batch/cell/commit）和脚本路径都吻合才视为本批 job；
4. 只取消 receipt 明确列出的 job ID。通用 `ptb-*`、同一 Unix 用户或同一 reservation
   均不构成归属证据。

2026-08-30 的现有队列审计记录：

- `82830/82831` 名为 `airsgpu`，命令是 `/home/robtang_google_com/airs-ops/airs_gpu.sbatch`，
  输出位于 `/rmeng_data/robtang/airs-runs/exp6-gpu-s2`/`s3`；它们不属于本 PTB spec。
- `83836/83837` 从当前仓库 `run_gates.sh` 提交，输出位于本项目 `_slurm_gates/g1-*`，是
  本项目的旧 G1 canary；但旧名 `ptb-g1-canary` 不符合新规则，只能作为 legacy 记录。
- `ptb-pack`、`ptb-rescore`、`ptb-greedy-board` 等通用名称没有本批 receipt，归属未知，
  不得操作。

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
  - SHA-256：`72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8`
  - 内部 Inspect fork commit：`64db0afdd3796732b232954ef440c66ed22923a7`；该实现会在
    30000–39999 随机选择并占用可用端口，不复用固定 vLLM port。
- HF cache：顶层 `data/ptb/hf`。
  - `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1` 必须完整缓存；
  - `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` 已完整缓存；
  - `HuggingFaceTB/SmolLM3-3B-Base@d78a42f79198603e614095753484a04c10c2b940` 必须完整缓存；
  - `google/gemma-3-4b-pt@cc012e0a6d0787b4adcc0fa2c4da74402494554d` 必须以已接受
    Gemma 使用条款的 Hugging Face 身份下载；snapshot 不完整时 launcher/preflight 失败关闭。
- 结果：顶层 `data/ptb/results`。
- 每 job local scratch：`/mnt/localssd/posttrainbench/$USER/$SLURM_JOB_ID`。

固定 revision 的缓存命令（在顶层仓库执行）为：

```bash
HUGGINGFACE_HUB_CACHE="$PWD/data/ptb/hf/hub" hf download Qwen/Qwen3-1.7B-Base \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1
HUGGINGFACE_HUB_CACHE="$PWD/data/ptb/hf/hub" hf download Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539
HUGGINGFACE_HUB_CACHE="$PWD/data/ptb/hf/hub" hf download HuggingFaceTB/SmolLM3-3B-Base \
  --revision d78a42f79198603e614095753484a04c10c2b940
HUGGINGFACE_HUB_CACHE="$PWD/data/ptb/hf/hub" hf download google/gemma-3-4b-pt \
  --revision cc012e0a6d0787b4adcc0fa2c4da74402494554d
```

所有大文件和 provider 证据在 `/data` 下，已被顶层 `.gitignore` 排除。历史 trace
不挂载进 agent sandbox。

## 3. 一次性验证阶梯

在仓库根目录：

```bash
# G0：单测、schema 和十六格 dry-run
uv run --extra dev pytest -q
uv run awm ptb dry-run
uv run awm ptb check --before-context-gate

# G1–G3：选择一台没有旧式 exclusive job 的目标节点，严格顺序运行
cd third_party/PostTrainBench
bash src/commit_utils/slurm/run_gates.sh g1 slurm2-a3nodesetondem-0 gsm8k-opus5-4x4-batch1
bash src/commit_utils/slurm/run_gates.sh g2 slurm2-a3nodesetondem-0 gsm8k-opus5-4x4-batch1
bash src/commit_utils/slurm/run_gates.sh g3 slurm2-a3nodesetondem-0 gsm8k-opus5-4x4-batch1
cd ../..

# G4：真实 GPU/container/provider smoke；覆盖 Opus max/xhigh/high 1M 和 max 200K
# 选择 Qwen arms，避免 provider smoke 被 Gemma gated-download 前置条件混淆。
uv run awm ptb context-smoke --cell b02 --cell b06 --cell b10 --cell b14
squeue -u "$USER"
uv run awm ptb check
```

`context-smoke` 会同时验证一张 H100 的容器可见性、task/judge/auth assets、真实 Claude
provider 调用，以及响应中的 `modelUsage.contextWindow`；它写入 gitignored
`data/ptb/context-validation/*.json`。launcher 会把成功记录的实际 SHA-256 写入 receipt，
job 启动时重新校验，防止排队期间被改写。裸 `claude-opus-5` 在 2026-08-30 实测为
200K，显式 `claude-opus-5[1m]` 为 1M；二者都是正式 setup，验证记录必须精确匹配，
不得在两种 context 之间静默回退。

另一个 formal-only blocker 是当前 Hugging Face 身份尚未获准下载 gated 的
`google/gemma-3-4b-pt`。先在其官方模型页接受条款，再执行 manifest 中固定 revision 的
`hf download`；不得用微调版、镜像模型或别的 revision 替代。该 blocker 不影响 B06/Qwen pilot。

## 4. Pilot 与正式提交

两次提交都要求顶层和 PTB submodule worktree 干净，并在 receipt 中冻结两个 commit。
job 启动时会把冻结的 PTB commit 以 `git archive` 物化到 job-local scratch；后续共享开发
工作区变更不会影响正在运行的 prompt、evaluator 或 judge。

```bash
# G5：B06 形状，1h agent budget，仍跑全部 official judges + full GSM8K eval
uv run awm ptb submit --pilot

# 查看命令打印出的 receipt
uv run awm ptb status data/ptb/batches/gsm8k-opus5-4x4-batch1/pilot-*.json
uv run awm ptb audit-receipt data/ptb/batches/gsm8k-opus5-4x4-batch1/pilot-*.json

# G6：只有 pilot audit 为 0 issue 才提交十六个 10h cell
uv run awm ptb submit
uv run awm ptb status data/ptb/batches/gsm8k-opus5-4x4-batch1/formal-*.json
uv run awm ptb audit-receipt data/ptb/batches/gsm8k-opus5-4x4-batch1/formal-*.json
```

正式提交先把十六个独立 Slurm job 全部置于 hold，receipt 写齐后再通过一次
`scontrol release` 共同放行。launcher 拒绝同类 receipt 的第二次提交；如果提交中途失败，
已创建的 job 保持 hold，receipt 会保留 job 和失败 cell，不能直接重跑造成重复。

每个 job 请求 `1 H100 + 16 CPU + 128G RAM`，agent budget 10h，Slurm walltime
另留 12h 给排队的 official judge lock、四个 judge 和 full evaluation。官方 ChatGPT
judge 共享认证文件时，完整 judge phase 通过
`/rmeng_data/robtang/ptb-locks/official-judges.lock` 串行化 refresh-token 写入；agent
本身使用无可写 token 文件的 Vertex/GCE metadata ADC。

## 5. Research judges

只有 formal receipt 的每个 cell 都通过 official audit 后，才运行研究 profile：

```bash
uv run awm ptb research-judges \
  data/ptb/batches/gsm8k-opus5-4x4-batch1/formal-*.json
```

它为每个不可变 result 提交一个 CPU-only Slurm job，使用
`claude-opus-5[1m]`、xhigh、Vertex 和完全相同的四份 judge prompt。输出为独立的
`judgement_claude_*_rerun.json`，不会覆盖 official canonical verdict。
research job 也从 official receipt 冻结的 PTB commit 物化 source，避免 judge prompt 随后漂移。
它同时复用并校验 receipt 中冻结的 Opus 5 xhigh provider 证据；模型字符串保持
`claude-opus-5[1m]`，因此不会退化到裸 alias 的 200k context。

## 6. 取消、故障与恢复

查看 receipt 后，只取消其中明确列出的 job：

```bash
python3 -c 'import json,sys; print(" ".join(j["job_id"] for j in json.load(open(sys.argv[1]))["jobs"]))' RECEIPT
scancel JOB_ID [JOB_ID ...]
```

不要用用户名或 partition 作为宽泛 `scancel` 目标。取消后结果保持原位，scratch 默认由
job trap 清理。一次失败不从 pilot workspace 或半成品 checkpoint resume；先审计
`logs/slurm/`、result 的 `output.log/error.log/runtime_provenance.json`，修复基础设施后用新
batch/run index 明确提交。Gemma 四格一旦揭盲就算消费 holdout，不能把修复后针对结果的
迭代继续标成同一个 untouched holdout。
