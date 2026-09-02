# PostTrainBench 配置与修改归属规范

**用途**：统一 PostTrainBench 相关配置、代码修改、实验定义和运行资产的归属，避免本地秘密进入 Git、submodule 修改丢失，或项目实验逻辑污染官方 benchmark 语义。

## 一、三层划分

### 第一层：机器运行环境、秘密和大文件

这些内容属于具体机器或集群，不进入 Git：

- `third_party/PostTrainBench/.env`
- `third_party/PostTrainBench/agents/<agent>/auth.json`、`oauth_token` 等认证文件
- Hugging Face token 与 provider API keys
- Apptainer `.sif` 镜像、HF cache、下载的数据、临时目录和实验结果
- Slurm 的 account、partition、QoS、节点名等站点配置

大文件统一放在顶层 `data/ptb/`，例如：

```text
data/ptb/
├── containers/
├── hf/
├── results/
└── scratch/
```

顶层 `/data` 已被 `.gitignore` 忽略；`.env` 和认证文件也必须保持 gitignored。`.env` 只保存这些目录的绝对路径和本机配置，不能作为实验定义文件。

### 第二层：benchmark harness 的通用行为

会改变 PostTrainBench 执行语义或可复用运行能力的修改，进入我们的 PostTrainBench fork：

- `src/run_task.sh`、prompt、judge、evaluator
- task 实现和容器定义
- agent `solve.sh` 与 trace parser
- 通用调度器 backend，例如 Slurm submission adapter
- 与特定集群无关的安全性、可复现性和资源隔离修复

`third_party/PostTrainBench` 是 Git submodule。不能把长期修改留在 detached HEAD 或未提交状态中；标准流程是：

```bash
cd third_party/PostTrainBench
git switch -c awm/<feature-name>
# 修改、验证、提交并 push 到 PostTrainBench fork

cd ../..
git add third_party/PostTrainBench
# 在 agentic-world-model 中提交新的 submodule commit 指针
```

通用 Slurm 支持属于这一层；具体站点的 partition、account、节点列表和秘密仍属于第一层。

### 第三层：项目实验定义与分析

描述“我们要跑什么、为什么跑、怎样比较”的内容进入顶层 `agentic-world-model`：

- task、base model、agent、预算、重复次数和随机种子
- 使用的 PostTrainBench commit、容器版本和 judge 模式
- trace 下载、转换、索引和深度分析代码
- holdout 与 train/test split
- 实验 manifest、结果元数据和可复现性记录

例如 `splits/posttrainbench/gsm8k-gemma-holdout-v1.yaml` 只定义公开 trace 的分析划分，不是 PostTrainBench 运行配置。建议将可提交的运行定义单独放在 `experiments/posttrainbench/*.yaml`，由薄 launcher 转换成调度器参数，而不是反复修改上游 `commit.sh` 的 sweep。

## 二、快速归属判断

| 修改内容 | 保存位置 |
|---|---|
| API key、OAuth、HF token | 本机 secret / PTB gitignored 文件 |
| SIF、模型 cache、task data、trace、结果 | 顶层 `data/ptb/` |
| Slurm account、partition、节点和挂载路径 | 本机或集群配置，不提交 |
| 通用 `sbatch` adapter、GPU 隔离修复 | PostTrainBench fork |
| evaluator、judge、prompt、agent scaffold | PostTrainBench fork |
| GSM8K × Gemma、10h、重复次数等实验矩阵 | 顶层 experiment manifest |
| holdout 和分析方法 | 顶层 `splits/`、`awm/`、`doc/` |
| submodule 采用哪个 PTB 版本 | 顶层仓库的 gitlink commit |

## 三、Slurm 集群运行边界

“一台开发/提交机写代码，多台计算机只接收实验作业”的结构是标准 HPC 架构：开发机承担 Git、实验定义和 `sbatch`；计算节点由 Slurm 分配 GPU，在 Apptainer 中运行；容器、HF cache 和结果通过共享存储访问，临时工作目录使用节点本地 scratch。

官方 PostTrainBench 提交层只实现了 `htcondor` 和 `htcondor_mpi-is`；我们的 fork 已增加 non-exclusive GRES Slurm adapter，入口为 `src/commit_utils/slurm/submit.sh`。adapter 负责资源、环境、日志、preflight、scratch 和单 GPU 设备隔离，核心 worker 仍是 `src/run_task.sh`，不会重写 benchmark 语义。

当前通用实现是：

1. **正式 GRES 模式**：每个 PTB cell 请求 `16 CPU + 128G + gres/gpu:1`，不携带 `--exclusive`；Slurm 的 device cgroup 和 Apptainer 共同限制可见设备。
2. **allocation-scoped cleanup**：evaluation 前只查询当前 `SLURM_JOB_GPUS`，只清理当前 Unix 用户的进程；不再执行全节点 CUDA PID kill。
3. **legacy manual 模式**：只作为无 GRES 站点的显式兜底，并强制 `--exclusive`，不能用于同节点多任务。
4. **完整流程合同**：正式/pilot 需要完整权重、trace、monitor、provenance、四个 schema
   正确的 official canonical verdict、full-eval log 和数值 `metrics.json`，否则 job 失败。
5. **冻结 source 与批量放行**：pilot/formal job 从 receipt 的 PTB commit 在 local scratch
   物化 source；正式十六格先全部 held submit，再共同 release，避免共享 worktree 和先后揭盲。
6. **base-model revision**：顶层 manifest 固定 Hugging Face commit；共享 cache 中对应 snapshot
   和全部权重 shard 必须完整，agent prompt 使用该 local snapshot，provenance 记录 revision/config digest。
7. **setup-specific provider evidence**：真实 provider smoke 记录 model/effort/context/CLI/Vertex
   route 和实际 SIF digest；max/xhigh/high 1M 必须精确解析为 1M，max 200K 必须精确解析为
   200K。提交 receipt 冻结记录 SHA-256，正式 cell 与 research judge 启动时再次比对。
8. **共享账号 ownership**：Unix `UserId` 不代表项目归属。每个 Slurm job name 必须带当前
   顶层 Git 分支；顶层 manifest/receipt 另外冻结 spec、batch、cell、两个 commit 和 job ID。
   无匹配 receipt 的通用任务名一律视为他人或归属未知，不能取消。具体命名与历史队列审计
   见 `posttrainbench_slurm_runbook.md` 第 0 节。

详细配置、gate 和命令见 `third_party/PostTrainBench/src/commit_utils/slurm/README.md`。

当前项目的本机 `.env` 将作业限制在以下四个节点（该文件 gitignored）：

```text
slurm2-a3nodesetondem-0
slurm2-a3nodesetondem-1
slurm2-a3nodesetondem-2
slurm2-a3nodesetondem-3
```

站点的永久配置源位于集群配置 bucket；`slurm.conf` 增加 `AccountingStorageTRES=gres/gpu`，专用 `ptb-a3` partition 仅覆盖上述四节点且 `OverSubscribe=NO`。2026-08-30 live controller 已显示每节点 `CfgTRES=...gres/gpu=8`，partition 合计 `gres/gpu=32`。G1–G3 共享节点冲突 gate 必须在旧式 exclusive job 释放节点后通过，才允许启动 pilot。

## 四、保持官方可比性的底线

- 调度器从 HTCondor 换成 Slurm，不应改变 prompt、10 小时预算、可见 GPU 数、evaluation、judge 或评分规则。
- smoke test、缩短预算、跳过 judge 必须使用独立 experiment 名称，不能当作正式结果。
- 每次正式结果至少记录顶层 commit、PTB submodule commit、experiment manifest、容器 digest、agent CLI 版本和 Slurm job ID。
- 站点适配只负责资源分配和启动 worker；benchmark 语义修改必须单独审查和标记。

## 五、reward-hacking judge profiles

PTB fork 提供两套运行/输出 profile，但复用完全相同的 `prompt.md`、
`get_judge_prompt.py`、judge tools 和 `task/judgement.json` schema：

- `official`（默认）：Claude Code CLI + 显式 `claude-opus-5[1m]` + `high`，四个 judge
  （包括 general）统一生成聚合和验证器读取的 canonical
  `judgement_gpt5_4/api/ptb_lookup/general*.json`。
- `claude`（研究用途）：相同的 Claude Opus 5 + `high` runtime，但
  effort；生成隔离的 `judgement_claude_*.json`，不会冒充或覆盖官方 verdict。

通用 backend、认证隔离、trace parser 和输出 contract 属于第二层，保存在 PTB
fork；某次实验选用哪个 profile、是否 pin 精确 Opus 5 model id 属于第三层，保存
在顶层 experiment manifest。Claude judge 的 file-backed ADC 或 OAuth token 路径属于
第一层，只能放在 gitignored `.env`，且必须使用 judge 专用凭据，不能复用被测 agent
的配置目录或 token。GCP 集群优先使用计算节点的 Vertex/GCE metadata ADC；非 GCE 环境可以只读
挂载独立 ADC 文件。Claude judge 使用 safe mode，禁止加载 agent 留下的 `CLAUDE.md`、skills、
plugins、hooks、MCP 和 custom agents；其唯一任务规则来自共享 judge prompt。

因此，`claude` profile 可用于 judge agreement 研究；正式 PTB 结果使用同一 Claude
runtime 的 `official` profile 来写 canonical verdict。每次 Claude verdict
还应保留 metadata 中的 requested model、resolved model、effort、container 和 CLI
version，以免 provider 路由随时间变化后无法复现。

2026-08-30 已完成 Claude judge 的真实四节点认证 smoke。四台机器都通过各自的
GCE metadata ADC 使用 Claude Code 的 Vertex provider，`opus` 均解析为
`claude-opus-5`，effort 为 `xhigh`，容器内 CLI 均为 2.1.219：

| 节点 | 执行方式 | 结果 |
|---|---|---|
| `slurm2-a3nodesetondem-0` | Slurm job `82098` | PASS |
| `slurm2-a3nodesetondem-1` | Slurm job `82099` | PASS |
| `slurm2-a3nodesetondem-2` | 无 GPU SSH smoke（节点已有 PTB job） | PASS |
| `slurm2-a3nodesetondem-3` | 无 GPU SSH smoke（节点已有 score job） | PASS |

节点 2/3 对应的排队 Slurm smoke `82100`/`82101` 在 SSH 验证成功后已取消，避免
现有独占作业结束后产生重复模型调用。可复用的真实认证检查入口是 PTB fork 的
`src/judges/smoke_claude_vertex.sh`；它走与正式 Claude judge 相同的 isolated home、
safe mode、clean environment、Vertex auth、模型解析和 metadata 路径。

该四节点 smoke 只证明认证和模型路由可用，不证明指定 context。后续真实 provider
最小请求显示裸 `claude-opus-5` 的 `contextWindow=200000`；显式
`claude-opus-5[1m]` 返回 `contextWindow=1000000`。本批把两者都作为有意设置：前三个
setup 使用 `[1m]`，max/200K setup 使用裸模型名；G4 和正式提交对 resolved context
执行精确匹配，禁止静默回退。研究 judge 仍固定使用 `claude-opus-5[1m]` xhigh。
