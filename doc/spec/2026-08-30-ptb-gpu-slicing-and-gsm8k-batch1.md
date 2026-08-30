# PTB 单卡切分集群与 GSM8K 首批六实验 Spec

- 状态：Accepted / implementation in progress
- 日期：2026-08-30
- 范围：PostTrainBench（PTB）Slurm 运行层，以及 GSM8K 首批实验
- 数据分析契约：`splits/posttrainbench/gsm8k-gemma-holdout-v1.yaml`

## 1. 目标

这项工作交付两个相互依赖的结果：

1. 把四台 A3 计算节点配置成 GPU 可消费的 Slurm 资源池。一个 PTB job 只申请、锁定并看见一张 H100；同一节点可以同时运行八个互不干扰的 job。
2. 在隔离验证通过后，同时启动 GSM8K 的首批六个正式 cell：三个 Claude Code agent 配置分别训练两个 4B base model。所有 Claude 5 agent cell 都使用 1M context；必须在提交前通过真实 provider 响应验证，并在结果里记录实际解析值。

正式 cell 保留 PTB 的 benchmark 语义：同一官方任务 prompt、每个 agent 最多 10 小时、单张 H100、原始 evaluator 和 judge 输入。一次有效运行必须走完 PTB 官方完整流程：agent solve、trace parsing、workspace/final-model 收集、全部官方 reward-hacking judges、最终完整 evaluation 和 metrics 落盘。Slurm、Apptainer 和站点 scratch 只负责提供等价资源，不向 agent 添加解题策略。

本 spec 不把 Docker 作为 PTB 的新运行时。Docker 已安装，但官方资产是 Apptainer `.sif`；切换运行时会增加容器差异，且不能解决 Slurm 控制面的整节点锁定。

## 2. “一个实验”的准确含义

一个实验是一个独立 PTB cell：

```text
(task, agent model + effort, base model, budget, run index)
    -> 一个隔离 sandbox
    -> agent 自主研究和训练
    -> final_model/
    -> reward-hacking judges
    -> GSM8K final evaluation
    -> trace、metrics 和环境元数据
```

这里的 judge 不是可选的后处理。正式 cell 只有在 PTB 当前启用的全部官方 judge 都完成、canonical verdict 文件齐全，并且最终 evaluation 成功生成 `metrics.json` 后才算完成。Claude Opus 5 xhigh research judge 可以追加，但不能替代 official canonical judges。

Agent 不只是“设计一种数据提取方法”。它拿到 PTB 官方 prompt 和 `evaluate.py` 后，可以自主：

- 搜索、下载、生成、过滤或清洗允许使用的训练数据；
- 选择 SFT、LoRA、全参数训练、课程学习或其他合法 post-training 方法；
- 做多轮小规模评测和超参数迭代；
- 把最优且能在起始环境加载的 checkpoint 写入 `final_model/`。

它不得使用 GSM8K test questions/answers 训练，不得从具体测试题派生训练样本，不得修改 evaluator/templates，不得微调指定 base model 之外的模型，也不得利用环境中的 provider key 直接调用外部 LLM 生成训练数据。

## 3. 不可改变的正式资源与语义

每个正式 cell 的资源请求为：

| 资源 | 每个 cell | 八个 cell/节点 | 单节点容量 |
|---|---:|---:|---:|
| GPU | 1 × H100 80GB | 8 | 8 × H100 80GB |
| CPU | 16 logical CPUs | 128 | 208 |
| RAM | 128 GiB | 1 TiB | 约 1.8 TiB |
| scratch reservation | 400 GB | 3.2 TB | 5.9 TB local SSD |
| agent budget | 10 小时 | 各自独立计时 | 不共享 |

四节点理论上可并行 32 个正式 cell。首批只运行 6 个，因此应能落在一台节点上，但调度器可以为了容量、故障域或公平性把它们分散到多台节点。

400 GB 与上游 HTCondor 的 `request_disk=400G` 同义：调度前保证容量并为每个 cell 提供独立目录。当前 ext4 local SSD 没有 project quota，因此第一版不声称这是硬写入上限；若必须硬限制，需要由站点管理员启用 filesystem quota。GPU、CPU 和 RAM 必须由 Slurm/cgroup 硬隔离。

正式运行不得通过缩短 prompt 中的时间、减少最终测试集、修改打分器或给 agent 额外训练资产来节约资源。所有 smoke/pilot 使用独立 experiment name，不能进入正式聚合。

## 4. Slurm 目标状态

### 4.1 当前问题

当前集群已经使用 `select/cons_tres`、`task/cgroup` 和 `jobacct_gather/cgroup`，具备按资源共享节点的基础，但尚未达到可用状态：

- `a3` 分区是 `OverSubscribe=EXCLUSIVE`；
- PTB Slurm submitter 固定传入 `--exclusive`；
- 节点显示 `Gres=gpu:8`，但 `CfgTRES` 和 partition `TRES` 没有 `gres/gpu`；
- 一个请求 `16 CPU + 128G + 1 GPU` 的 job 实际被分配 208 CPU 和整个节点。

只删除提交命令里的 `--exclusive` 不安全。必须先让 GPU 成为真实 consumable TRES，并验证 Slurm 能分配不同的 GPU device cgroup。

### 4.2 站点配置

站点管理员应为用户指定的四台 A3 节点提供一个非整节点独占的 PTB partition，或等价地修改专用 partition。具体 partition 和节点名保存在 gitignored `.env`，不进入本 spec 的可提交配置。

期望属性：

```text
SelectType=select/cons_tres
SelectTypeParameters=CR_CORE_MEMORY
GresTypes=gpu
Partition OverSubscribe=NO
TaskPlugin=task/cgroup,task/affinity
Cgroup ConstrainDevices=yes
每节点 GRES=gpu:h100:8（由 NVML autodetect 或逐设备 File 配置）
```

验收时，`scontrol show node` 必须出现等价于：

```text
CfgTRES=cpu=208,mem=...,gres/gpu=8
```

运行一个正式形状的单卡 smoke 后应出现等价于：

```text
AllocTRES=cpu=16,mem=128G,gres/gpu=1
CPUAlloc=16
```

而不是 `CPUAlloc=208`。

### 4.3 每个 cell 的提交形状

目标提交参数为：

```text
--nodes=1
--ntasks=1
--cpus-per-task=16
--mem=128G
--gres=gpu:h100:1       # 若站点不使用 typed GRES，则 gpu:1
--time=<10h agent budget + documented harness overhead>
```

不得携带 `--exclusive`，不得使用 PTB `manual` GPU mode，也不得手工假设物理 GPU 一定是 `0`。Slurm 分配设备并设置可见性；Apptainer 继承 Slurm device cgroup。

## 5. 单容器严格隔离合同

一个 Slurm job 对应一个 PTB Apptainer sandbox。每个 sandbox 必须满足：

1. `torch.cuda.device_count() == 1`，且唯一设备是 H100 80GB。
2. 容器不能打开未分配 GPU 的 device node；只设置 `CUDA_VISIBLE_DEVICES` 不算通过。
3. 使用 `-c --cleanenv --pid --no-init`，只显式注入 allowlist 中的环境变量和认证。
4. 独享 `${scratch_base}/${USER}/${SLURM_JOB_ID}`，以及独立的 `/tmp`、home/task、writable overlay、HF overlay 和日志路径。
5. base-model/HF blob cache 可以只读共享；所有 lock、overlay upperdir 和下载中的临时文件必须 cell-local。
6. vLLM 使用动态空闲端口或 cell-local 显式端口，不得固定复用同一端口。
7. 训练结束或 final evaluation 前的 CUDA 清理只能作用于当前 Slurm allocation 中、当前用户的进程。禁止全节点 `nvidia-smi ... | xargs kill -9`。
8. Judge 使用新的 isolated home，不加载 agent 留下的 `CLAUDE.md`、skills、plugins、hooks、MCP 或配置目录。
9. 每个结果记录 Slurm job ID、node、GPU UUID、CPU/RAM 请求、顶层 commit、PTB commit、SIF SHA-256、CLI version、requested/resolved agent model、requested/resolved context window、effort、auth provider 和 judge profile。

若使用 Vertex/GCE metadata ADC，各 cell 不共享可写 token 文件，适合并发。若改用 Claude OAuth，必须为每个 cell 复制独立的可写认证状态，或提供经过验证的并发锁；六个进程同时改写同一个 OAuth 文件不属于严格隔离。

## 6. 代码与配置归属

### 6.1 集群站点层（不进 Git）

- Slurm partition、`slurm.conf`、`gres.conf`、`cgroup.conf`；
- 具体四节点 nodelist、account、QoS；
- local SSD mount、secret、ADC/OAuth 和 HF token；
- `.env` 中的站点路径。

### 6.2 PTB fork（通用能力）

- Slurm submitter 的 non-exclusive GRES 模式；
- per-GPU/own-user CUDA process reap；
- single-cell scratch、overlay、port 和 log 隔离；
- `preflight-only`、`runtime-smoke` 和冲突 canary；
- 明确的 Claude `high/xhigh/max` agent scaffold 与 trace metadata；
- 不改变 prompt/evaluator/scoring 的通用安全修复。

### 6.3 agentic-world-model 顶层仓库

- 本 spec；
- `experiments/posttrainbench/*.yaml` 中的六-cell manifest；
- launcher、结果索引、holdout 纪律和分析代码；
- PTB submodule commit 指针。

`splits/posttrainbench/gsm8k-gemma-holdout-v1.yaml` 是历史公开 trace 的分析划分，不是运行配置，不能直接拿来提交 job。

## 7. 快速、局部、无冲突的验证阶梯

所有 gate 按顺序执行；任一 gate 失败都停止，不提交后面的长任务。

### G0：零 GPU 静态验证（约 1 分钟）

- shell/unit tests；
- dry-run 检查每个 cell 是 `gpu:1 / cpu:16 / mem:128G`；
- 断言命令中没有 `--exclusive` 或 `manual` GPU mode；
- 断言六个 result/scratch ID 唯一。

### G1：两个 Slurm GPU canary（约 2–3 分钟）

在同一节点同时启动两个轻量 job。每个 job 打印 allocation、GPU UUID 和 cgroup device 可见性，并保持 60 秒。

通过条件：

- 两个 job 同时处于 RUNNING，节点没有整机分配给任一 job；
- 每个 job 只看见一张 H100；
- 两个 GPU UUID 不同；
- 每个 job 的 `AllocTRES` 是 16 CPU、128G、1 GPU。

### G2：八路容器 canary（约 5 分钟）

在同一节点启动八个 Apptainer smoke，每个容器只做 CUDA tensor allocation、记录 UUID、保持 90 秒后退出；再提交第九个。

通过条件：八个容器的 UUID 完整覆盖八张卡且无重复；第九个在前八个占满时等待，释放一张卡后才启动；任何容器都不能访问邻居设备。

### G3：清理冲突测试（约 5–10 分钟）

两个不同 GPU 的 cell 同时运行。A 启动持续 CUDA canary；B 执行与正式 evaluation 相同的 GPU reap，然后进入一次轻量 eval smoke。

通过条件：B 只清理自己的 GPU；A 的 PID、CUDA allocation 和 heartbeat 全程存活。该 gate 专门防止一个 cell 结束时杀掉其他七个实验。

### G4：单容器 PTB runtime smoke（约 5–10 分钟）

只验证 SIF、GPU、cache、task assets、agent CLI 和认证；最多做一次最小模型请求，不训练、不跑全量 GSM8K、不生成可报告 score。该请求必须同时验证所选 Claude 5 model、effort 和 1M context 是否被当前 CLI/provider 接受；只看到本地 CLI `--help` 或假设 alias 默认值不算通过。

### G5：单 cell 1 小时 pilot

默认候选是 `Opus 5 xhigh 1M × Qwen3-4B-Base`。使用 `_pilot_1h` 结果命名，跑完整的官方 agent solve → trace parsing → workspace/final-model collection → 全部 official judges → full final evaluation → metrics 路径，确认 trace、`final_model`、所有 canonical judge files、metrics、清理和结果复制完整。pilot 不得设置 `POST_TRAIN_BENCH_SKIP_JUDGES`，也不得用 research-only Claude judge 代替 official judges；它不是正式十小时结果。

### G6：六个正式 cell 同时提交

只有 G0–G5 全部通过才提交第 8 节矩阵。正式 cell 固定 10 小时 agent budget，不从 pilot resume，不复用 pilot workspace。

## 8. 首批六实验矩阵

共同参数：

```text
task = gsm8k
agent_cli = Claude Code
context_window = 1M
agent_budget = 10h
gpu = 1 × H100 80GB
cpu = 16
ram = 128 GiB
scratch reservation = 400 GB
run_index = 1
```

矩阵是三个 agent 配置乘两个被训练的 base model：

| Cell | Agent model | Context | Effort | 被训练模型 | 建议 agent scaffold |
|---|---|---:|---|---|---|
| B1 | `claude-fable-5[1m]` | 1M | max | `google/gemma-3-4b-pt` | `claude_non_api_max` 或等价 Vertex scaffold |
| B2 | `claude-fable-5[1m]` | 1M | max | `Qwen/Qwen3-4B-Base` | 同上 |
| B3 | `claude-opus-5[1m]` | 1M | max | `google/gemma-3-4b-pt` | `claude_vertex_max` |
| B4 | `claude-opus-5[1m]` | 1M | max | `Qwen/Qwen3-4B-Base` | 同上 |
| B5 | `claude-opus-5[1m]` | 1M | xhigh | `google/gemma-3-4b-pt` | `claude_vertex_xhigh` |
| B6 | `claude-opus-5[1m]` | 1M | xhigh | `Qwen/Qwen3-4B-Base` | 同上 |

Claude Code 2.1.219 的本地 `--help` 明确接受 `low, medium, high, xhigh, max`。正式 launcher 应显式传 `--effort` 并记录值，不能只依赖用户级默认配置。2026-08-30 的 Vertex 实测表明：裸 `claude-opus-5` 返回 `contextWindow=200000`，而 `claude-opus-5[1m]` 返回 `contextWindow=1000000`，所以正式 Opus cell 必须使用显式 `[1m]` 路由。不得静默回退到裸 alias 或非 1M context。

这六个 cell 是 `n=1` 的首批描述性比较，不足以估计方差或作统计显著性结论。基础设施稳定后，复现实验至少增加到每格 3 次独立 run。

## 9. 公平性与复现口径

六个正式 cell 必须在提交前一次性冻结：

- 顶层 commit 和 PTB submodule commit；
- agent prompt、task assets 和 evaluator；
- agent CLI version；
- requested/resolved model 与 1M context；
- 容器 digest；
- base model revision 与公共只读 cache snapshot；
- agent auth provider/Vertex project 与 region；
- judge profiles；
- 所有资源和时间预算。

正式批次默认关闭 CLI auto-update，避免先启动和后启动的 cell 得到不同 CLI。可以共享已经存在的 base-model blobs，但不得为某个 agent 预置专属训练数据、历史解法或公开 PTB trace。

Judge 是官方完整流程的一部分，不是第 8 节的自变量。六个 cell 应使用完全相同的 judge 组合：每个 cell 在 `run_task.sh` 主流程中运行当前 PTB 启用的全部 official judges 并生成 canonical verdict，然后执行 full final evaluation；可再对同一不可变结果运行 Claude Opus 5 xhigh research profile，输出独立文件，不覆盖官方 verdict。缺少任一 required official verdict、只有 research verdict、跳过 judge 或没有 `metrics.json` 的 cell 都判为流程失败，不进入六格比较。

## 10. Holdout 与“先理解、再复现”

`gsm8k-gemma-holdout-v1` 把历史公开 trace 按被训练的 base model 划分：Qwen/SmolLM 是 train，Gemma 是 test。为了保留这个实验含义：

1. 理解阶段只分析 split 的 train traces，不查看 Gemma test trace 的策略或结果细节。
2. 六个 agent sandbox 都不得看到任何历史 PTB trace，也不得被提示历史 agent 的方法。
3. 六个 cell 同时冻结和提交，保证先看到 Qwen 新结果的人无法据此修改本批 Gemma 配置。
4. Gemma 的三个新结果一旦揭盲，就算消费了一次 holdout。之后根据这些结果改方法再跑 Gemma，必须标成开发迭代，不能继续声称是同一个 untouched holdout test。
5. Qwen cells 可用于复现/开发诊断；Gemma cells用于检查 agent 方法对 held-out base-model family 的迁移。

## 11. 产物与第一批分析

每个 cell 至少必须完整产出：

- 原始和解析后的 agent trace；
- `final_model/` 与其 config/model identity 检查；
- GSM8K `metrics.json` 和完整 final-eval log；
- PTB 当前要求的全部 official judge verdicts，包括 canonical contamination、API-usage、PTB-lookup 和 general judge 产物；
- 可选但推荐的 Claude Opus 5 xhigh research verdicts；
- `system_monitor.log`；
- 第 5 节要求的 runtime provenance。

第一批报告包括：

- 六格 GSM8K accuracy；
- 同 agent 配置下 Gemma 与 Qwen 的成对差异；
- 同 base model 下 Fable max、Opus max、Opus xhigh 的描述性差异；
- 成功率、实际用时、CLI tokens、GPU 利用率和训练/评测阶段占比；
- 数据来源、过滤方法、训练方法、迭代次数的 trace-derived 摘要；
- judge flags、基础设施异常和任何不满足官方口径的偏差。

## 12. 已确认的实现决策

### D1：Agent 认证来源

已确认：六个 agent 都使用 Claude Code + Vertex/GCE metadata ADC。四节点已验证 Opus 5/xhigh 认证路径，而且它避免六个 job 竞争写同一个 OAuth token 文件。

备选：严格沿用公开 trace 名称对应的 `claude_non_api*` OAuth scaffold。若选择它，必须先解决每-cell OAuth 状态隔离，并单独确认 Fable 5 与 Opus 5 的并发额度。

### D2：容器口径

已确认：六个 cell 使用同一个经过验证、钉 digest 的 `opus_5.sif`，从而把 agent model/effort 作为主要自变量。

备选：复现历史公开配置——Fable 使用其历史 `standard.sif` 路径，Opus 5 使用上游 `single_task_opus5.sub` 指定的 `opus_5.sif`。这更接近各自历史 run，但 agent 比较会同时混入 container 差异；当前还需要构建缺失的 `standard.sif`。

### D3：Judge 组合

要求：每个 pilot/正式 cell 都通过 `run_task.sh` 跑完当前 PTB 官方完整流程，包括全部 official canonical judges 和 full final evaluation。建议在此之后再对同一不可变结果运行 Claude Opus 5 xhigh research judges。只跑 Claude profile、跳过任何 required official judge 或只完成 evaluation 都不能进入正式聚合。

### D4：正式预算

本 spec 假设六个正式 cell 都是官方 10 小时、1M context、各 1 次；1 小时只用于 B6 形状的 pilot。10 小时只限制 agent solve 阶段；judges 和 final evaluation 按官方顺序在其后完成，Slurm walltime 需要另留明确的 harness overhead。如果目标不是这个预算，应在任何 pilot 前修改 manifest，而不是提交后临时覆盖。

## 13. 完成定义

以下条件全部满足才算交付：

- Slurm 对四节点实现真实 GPU TRES 和 device-cgroup 隔离；
- 单卡 job 不再锁整节点；
- 八路 canary 和跨 cell cleanup conflict test 通过；
- PTB fork 的 non-exclusive GRES adapter、per-GPU cleanup 和测试已提交；
- 顶层六-cell manifest 可 dry-run、可逐 cell/整批提交；
- 单 cell pilot 端到端产物完整；
- pilot 和六个正式 cell 均确认 resolved context 为 1M，并跑齐全部 official judges 与 full final evaluation；
- 六个正式 job 在冻结的同一口径下并行启动并可追溯；
- 文档记录最终站点要求、运行命令、恢复/取消流程和已知限制。
