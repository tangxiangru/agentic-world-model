# 仓库架构设计:三家 benchmark 的接入、轨迹统一与 scope 集合

**日期**:2026-08-24 **状态**:待 review **配套**:《评估原则》《benchmark 案例与数据清单》、`doc/reference/harness_facts/`(六份 harness 调研原文)

## 一、目的与范围

把《案例与数据清单》里的筛选结果落成可运行、可分析的仓库。首批只接三家 Track-Optimize 基准:**PostTrainBench、AIRS-Bench、NanoGPT Speedrun(Prime Intellect 设定)**。RE-Bench / MALT、AutoLab、Track-Replica 全部延后。本机(4× RTX 6000 Ada)只做轨迹分析;实验在外部 GPU 机器上跑,产物同步回本机。

## 二、核心决策

**D1 混合 runtime,不强求统一**。判据只有一条:上游是否发布了完整"运行器"(建沙箱、挂数据、把 prompt 交给 agent、限时、收产出物、评分)。

| 基准 | 上游给了什么 | 运行方式 |
|---|---|---|
| PostTrainBench | 题目 + 容器定义 + 运行器 `run_task.sh` + agent 接口 `agents/<name>/solve.sh` | **原生**,submodule 指向我们带补丁的 fork 分支 |
| AIRS-Bench | 只有题目(prompt、prepare、evaluate、metadata);公开版 aira-dojo 跑不了 AIRS | **Harbor**,`awm/adapters/airs.py` 从 20 个 `metadata.yaml` 生成 task 目录 |
| Speedrun(PI) | 题目(program.md + baseline 脚本)+ 41 条轨迹;run.sh / verify.py 可从轨迹恢复;launcher 与沙箱未开源 | **Harbor**,`tasks/speedrun_pi/track3_optimizer/` 手写(只有一题) |

原生跑 PostTrainBench 的理由是可比性:与 HF 上 1,842 个公开 run(其中 1,745 个可转成事件,见第四节)同 prompt、同容器、同 `timer.sh`、同 `system_monitor.log`、同原生 CLI JSONL,自跑数据与公开数据可以直接合并分析。什么时候改用 Harbor:官方合并其 Harbor 适配器 PR #8 并以此出数据,或我们需要在同一批机器上混跑三家统一调度。

**D2 adapter 的定义**:给没有运行器的基准补运行器,借 Harbor 的 task 目录格式(task.toml + instruction.md + environment/ + tests/test.sh)来写,不重写评分逻辑——`tests/test.sh` 调用上游自己的 `evaluate.py`。只有任务成"族"时才写生成脚本(AIRS 20 题);单题手写(PI)。

**D3 统一发生在两层**:`scope/`(哪些任务算数,机器可读)与 `/data/hv/traj/events/`(所有轨迹转成同一事件流)。启动命令允许两条路(`run_task.sh` / `harbor run`),由薄壳 `awm run <task_id>` 分发。

**D4 我们的 agent 做成 CLI 可执行文件**(形如 claude-code),使 PostTrainBench 的 `solve.sh` 与 Harbor 的 `BaseInstalledAgent` 包装都是一行。Phase 2 才建。

**D5 事件 schema 以 PI 的四类事件为基底**(text / thinking / tool_use / tool_result),它已在 9 种 harness 上验证过,41 条 run 零转换加载。Harbor 的 ATIF 只作自跑产物的中间格式。

**D6 上游代码一律 submodule 钉 SHA,不复制**:溯源、许可(AutoLab 与 PI 无 LICENSE;AIRS / aira-dojo CC BY-NC)、体积。例外:PI 只有两个代码文件(21 KB),直接拷入 task 目录并记录来源 SHA;其轨迹属于数据,走 `/data`。

**D6a 只有需要改上游代码时才 fork**。判据:我们是否必须往上游的目录树里写东西。

| 上游 | 是否 fork | 理由 |
|---|---|---|
| PostTrainBench | **已 fork** → `tangxiangru/PostTrainBench`,当前 AWM 分支 `awm/slurm-gres-claude-profiles`;官方 upstream 为 `aisa-group/PostTrainBench` | 原生运行,必须在其树内维护 Slurm GRES、H100/local-SSD runtime、agent profiles、AIME scorer 与完整 provenance/validator 合同 |
| AIRS-Bench | 否 | 只读:adapter 读 `metadata.yaml` / `project_description.md` / `evaluate.py`,生成的 task 目录落在**我们**仓库;从不改其树。将来若要打评分器补丁再 fork |
| Speedrun(PI) | 否(也不做 submodule) | 只拷两个文件;缺失的 runner 由我们自己写在 `tasks/speedrun_pi/` |
| Harbor | 否 | pip 依赖。除非 Phase 1 验出 0.22.0 的 GPU 直通有问题需要打补丁 |

submodule 的 `fork` 指向 `tangxiangru/PostTrainBench` 并承载私有 AWM 分支，`upstream` 指向官方
`aisa-group/PostTrainBench`。`.gitmodules` 必须克隆 fork，确保固定的 AWM commit 可检出；同步官方用
`git -C third_party/PostTrainBench fetch upstream`，不能把 `DeepCommit-ai/PostTrainBench` 当作 upstream。

**D7 数据全放 `/data/hv/`**(本机根分区已 95% 满),仓库里不放任何轨迹;测试样本例外(截断到几十个事件)。

## 三、仓库布局

```
agentic-world-model/
├── doc/                      # 现有文档 + spec/ + reference/harness_facts/
├── third_party/              # submodule:PostTrainBench(fork 补丁分支)、airs-bench
├── tasks/                    # Harbor task 目录
│   ├── airs/<task>/          # adapter 生成,scope 内 8 题(GPU-heavy,2026-08-25 收窄)
│   └── speedrun_pi/track3_optimizer/   # 手写
├── scope/                    # posttrainbench.yaml、airs.yaml、speedrun_pi.yaml + schema.json
├── awm/                       # 唯一 Python 包:核心方法全部在此
│   ├── agent/                # Phase 2:自研 agent CLI(实验侧方法)+ shims/(PostTrainBench 的 solve.sh、Harbor 的 InstalledAgent 包装)
│   ├── adapters/airs.py
│   ├── traj/                 # schema.py、fetch.py、convert_{claude_code,codex,pi,atif}.py、index.py
│   ├── analysis/             # Phase 3:层级恢复、双层拆分、verifier(分析侧方法)
│   └── cli.py                # awm traj fetch|convert|index、awm scope list、awm run
├── tests/                    # pytest;tests/data/ 每源一条截断样本
└── pyproject.toml            # uv;harbor 作 pip 依赖
```

不设:顶层 `adapters/`、`agents/`、`fixtures/`、`data/manifests/`、`derived/`、`assets/`;`awm/agent/` 到 Phase 2 再建。

## 四、数据布局(`data/` 软链)

仓库里的 `data/` 是一个 **gitignore 掉的软链**,指向本机实际存放位置(当前 `/data2/gangda/hv`,14 TB 盘)。代码和文档只认这一个路径,换机器只改软链;`AWM_DATA_ROOT` 可覆盖(测试用它指向空目录跑)。新克隆需要 `ln -s <volume> data` 一次。

```
data/
├── traj/raw/posttrainbench/<agent_config>/<run>/  # HF 目录原样
├── traj/raw/pi_speedrun/traces/                   # 41 × {events,subagents,scratch}.json.gz + manifest
├── traj/runs/<job>/                               # 自跑产物,rsync 回来
├── traj/events/<source>/<run_id>.jsonl.gz         # 统一事件流;分析代码只读这一层
├── traj/index.parquet                             # run_id、source、model、harness、分数、token、时长
└── assets/                                        # 任务只读挂载的数据集与模型权重
```

公开轨迹与自跑轨迹只在 raw / runs 层不同,转换后同构。

**可下载轨迹清单**(2026-08-25 按上游文件列表实测):

| 源 | run 数 | 大小 | 说明 |
|---|---|---|---|
| PI speedrun | 41 | 50 MB | 全量。9 harness / 18 模型,172,694 主 + 118,980 子 agent 事件,附 scratchpad 决策日志 |
| PostTrainBench(trace) | 1,842 | 7.3 GB | 62 配置 × 7 基准;1,842 是 **run 目录数**,可分析语料是 **1,745**,见下 |
| — 默认首批 | 82 | 0.6 GB | 4 配置 × 5 个客观评分基准 |
| — 全配置 × 5 核心基准 | 1,260 | 4.9 GB | 去掉两个 LLM 评审基准 |

> **2026-08-28 更新**:1,842 这个数底下藏着三个不同的量,拉全量后逐目录实测:
> **1,842 个 run 目录 → 1,786 个有 `solve_out.txt` → 1,745 个能转成事件**。
> 完全没有轨迹的 56 个全部来自两个 opencode 配置
> (`opencode_opencode_kimi-k2.5_10h_run1`、`opencode_zai_glm-5_10h_run1`,各 28 个,
> 每个基准正好 8 个),目录里只有 `metrics.json`,其中 7 个的内容还是
> `No metrics.json produced.` 这句话本身而不是 JSON;剩下 41 个有轨迹但里面没有
> 任何 agent 事件,转换器按 `NoAgentOutput` 跳过。损耗集中在一个 scaffold 上,
> 所以引用时要说清楚引的是哪一层:
>
> | scaffold | 目录 | 有轨迹 | 已转换 | 损耗 |
> |---|---|---|---|---|
> | claude-code | 849 | 849 | 841 | 0.9% |
> | codex | 522 | 522 | 519 | 0.6% |
> | cursor-cli | 56 | 56 | 56 | 0.0% |
> | **opencode** | 415 | 359 | **329** | **20.7%** |
> | 合计 | 1,842 | 1,786 | 1,745 | 5.3% |
>
> 7.3 GB 是 run 目录下所有文件;只算 `solve_out.txt` 是 **4.13 GB**,其余是
> `metrics.json`、judge 判定和计时。另:`viewer_data/index.json` 里 1,509 条
> 全部在磁盘上有轨迹,所以从目录派生的 split 不会 pin 到读不出来的 run;
> 目录外还有 277 个有轨迹但未进目录的 run。

逐基准(trace 字节数 / 目录 → 已转换):bfcl 1.31 GB / 288 → 276、arenahardwriting 1.23 / 287 → 269、healthbench 1.17 / 295 → 276、gpqamain 1.01 / 241 → 230、humaneval 0.89 / 243 → 231、gsm8k 0.85 / 243 → 230、aime2025 0.84 / 245 → 233。

PostTrainBench 全量 28.9 GB 里其余部分暂不拉:工作区快照 10.6 GB(agent 写的代码,后续分析会用)、上游预解析的 viewer JSON 5.3 GB(与 trace 重复)、error log 1.5 GB。

## 五、事件 schema(v0)

每条 run 一个 `events/<source>/<run_id>.jsonl.gz`(主 agent 与子 agent 事件都在里面,按 `agent_id` 区分)+ 一个 `meta.json`。

事件字段:`run_id`、`agent_id`(`main` 或 `sub-<id>`)、`i`(序号)、`turn`(assistant 调用计数)、`ts`(ISO-8601,可空)、`type`(text / thinking / tool_use / tool_result)、`role`(user / assistant)、`text`、`redacted`(thinking 被闭源时为真)、`tool`、`args`(原始工具入参,可空)、`summary`(一行渲染)、`is_error`、`usage`({in, out, cache_read, cache_write},同 turn 只记一次)、`parent_tool_use`(tool_result 指向其 tool_use;子 agent 首事件指向父 agent 的派生调用)、`origin`(agent / harness / human——区分 launcher 注入的 `continue`、task-notification、goal 重注入)、`source_ref`({file, line},指回原生日志)。

`meta.json`:model、harness、task_id、budget(小时 / GPU)、t_start、t_end、final_score(含原始指标与归一化)、tokens、cost、n_events、subagents 索引(id、label、t_start、t_end、n_events)、来源特有字段整体塞进 `extra`(PI 的 `progression`、PostTrainBench 的 judge 判定与 `system_monitor` 路径)。

转换器四个:Claude Code stream-json(PostTrainBench 自带时间戳前缀)、Codex exec-json、PI(直读,补 `run_id` / `agent_id` / `origin`)、Harbor ATIF(Phase 1 有自跑产物后写)。已知需处理:Anthropic 模型 thinking 全为空(`redacted`);Claude Code 家族工具结果被 PI 截断(约 1.5–8 KB)而 codex / prime-agent 不截断;PI 13 条 "live" run 的 `progression` 结构与其余不同;PostTrainBench `solve_out.txt` 里夹有非 JSON 行。

## 六、scope 集合

> **2026-08-27 更新**:本节描述的 `scope/` 机制已整体移除,由 `splits/` 取代——每个
> YAML 是一份自包含的数据契约(pinned 上游 revision + 生成规则 + 物化清单),
> `awm split list|check|fetch` 是新入口;AIRS 的指标锚点不再拷贝,由 adapter 直接读
> 上游 `metadata.yaml`。以下保留作历史记录。

**分工:判断留在文档,清单留在 YAML。** `scope/<bench>.yaml` 只回答"我们跑哪些任务、它们的指标和成本是什么";为什么入选、G1–G4 各自的依据、什么被剔除,全部在《案例与数据清单》里,因为那是判断,判断读起来就该是散文。

第一版把判断也结构化了(`gates` 逐条 evidence、`scores` S1–S5、`verdict`、`track`、`domain_level`、`trajectories`),结果是 **3,128 行 YAML 记录 53 个任务**,其中同一段 G2 理由被抄了 32 遍(因为门槛依据多数是 benchmark 级的),`track` 53 条取值完全相同,S1 和 S5 各只填了 1 条。2026-08-25 精简到 **108 行 / 21 条**,判断回归文档。

文件形状:`tasks` 之上的键是 benchmark 级默认值,每条任务继承、可覆盖。

```yaml
resources: {gpus: 1, gpu_type: H200, cpus: 24, mem_gb: 200}
budget: {official_h: 24, poc_h: 8}
tasks:
  - id: TextualClassificationSickAccuracy
    family: Text Classification
    metric: {name: Accuracy, direction: higher_is_better, reference: 0.905, s_min: 0.145, s_opt: 1.0}
```

字段共七个:`id`、`family`、`metric`(名称、方向,以及归一化要的 baseline / reference / s_min / s_opt 锚点)、`resources`、`budget`、`variants`(该任务按此列表逐个跑一遍,PostTrainBench 的 4 个 base 模型即是)、`self_run`(默认真;为假表示只分析其公开轨迹、不花 GPU,两个 LLM 评审基准如此)。文件里只列在 scope 内的任务,所以没有 `verdict`——待定的(AIRS 的 DuoRC / FinQA,G1 未落实)留在文档里,定了再进来。

`awm scope list [--bench X] [--self-run]` 列清单;`awm scope check` 做三项对账:AIRS 的指标锚点 vs 上游 `metadata.yaml`(它们是拷贝,必须防漂移)、任务数 vs 文档 3.2/3.4 的声明、GPU·h vs 文档 5.1 的预算表。原先的 154 行 JSON Schema 一并删除——7 个字段不值得一个 schema 解释器,校验直接写在 `awm/scope.py` 里。

## 七、三家接入要点

### PostTrainBench(原生)

- 现成:prompt 模板与渲染、7 个评分器、chat 模板、`containers/standard.def`、去污染与模型身份两个确定性检查脚本、trace 解析器、28.9 GB 公开轨迹(1,842 run,Apache-2.0,不 gated)。
- 补丁(fork 分支):裁判的 ChatGPT-Pro 登录预检改为 API-key 或可关闭;`check_cuda.py` 的 "H100" 字符串检查按实验机放宽;AIME `endswith` 评分 bug(#44)与评测镜像 transformers 版本不一致(#65)。
- 我们写:`awm/agent/shims/posttrainbench/{solve.sh,api_keys.json}`(接入时拷入 submodule 的 `agents/hv/`);绕过 HTCondor 直接调 `run_task.sh` 的多机队列脚本;`test_data.json` 下载(GPQA 与 gemma gated,需 HF token)。
- 口径:官方 HF 缓存(14 模型 + 约 150 数据集)第一版不预热,agent 自行下载(规则允许无限制上网);单次解码评分噪声大,自跑至少 3 次取均值;首版只跑确定性检查,LLM 裁判后置。

### AIRS-Bench(Harbor + adapter)

- 现成:20 个任务目录、`metadata.yaml`(SOTA / s_min / s_opt / pip 依赖)、原始数据下载脚本、`test_task_folder.py` 自检。
- adapter 生成:`instruction.md` = `project_description.md`;Dockerfile 按 metadata 钉 pip 版本;数据不进镜像——`prepare.py` 在实验机跑一次,只读卷挂到容器 `./data`;`tests/test.sh` 跑 `evaluate_prepare.py` + `evaluate.py`,解析 `--- EVALUATION RESULT ---` 后的 JSON,按论文公式算归一化分(上游无此代码)。
- 口径:只评最终 `submission.csv`(MLGym 口径),不像 aira-dojo 每节点评测试集;WSC / WinoGrande / SQuAD 的隐藏测试集是公开的 validation split,允许上网则记为观察项;submodule 钉在 4–5 月评分器修复之后的 SHA;许可 CC BY-NC。

### Speedrun(Harbor + 手写)

- 来源:`program.md`、`train_gpt_simple.py` 拷自 PI(记 SHA);`run.sh`、`verify.py` 从轨迹里的工具结果恢复;`requirements.txt`(torch==2.11、huggingface_hub、wandb)与数据脚本(上游 modded-nanogpt `cached_fineweb10B.py`,40 训练分片 + 1 验证分片,8 GB)同样从轨迹恢复。
- 我们写:Dockerfile、docker-compose(8 GPU、只读挂数据)、`task.toml`(默认 24h 等预算切片)、`tests/test.sh`:扫描 `logs/*.txt`,按文件头部的脚本源码哈希分组,找齐 seed 0xC0FFEE+0..7 的 8 条终点 val loss,均值 < 3.27859 为有效记录,取最小 `train_steps`,连同 baseline 3290 与人类 2600 写入 `reward.json`。
- agent 包装层(三家共用):预算未用完则 `--continue` 并重注入目标(PI launcher 的行为,PostTrainBench 的 `*_reprompt` 变体同理)。
- 网络:track3-noweb 靠 bwrap + 只放行模型 API 的代理;Harbor 里 CLI agent 跑在容器内,`allow_internet=false` 会连带切断其 API 调用。冒烟阶段允许上网并审计轨迹,正式跑加代理 sidecar 白名单。
- 标定:baseline 3,290 步与 σ≈0.0013 在 8×H200 上标定;换机器先重跑一次 8-trial baseline 确认。

## 八、阶段与验收

**Phase 0(本机,零 GPU)**:骨架 + submodule;`scope/` 三个 yaml 与文档表格对账通过;`awm traj fetch` 拉下 PI 41 条与 PostTrainBench 首批(20 核心配置 × {claude-code, codex},约 2–4 GB);三个转换器产出 `events/`,PI 每条 run 的事件数与 manifest `n_events` 一致,PostTrainBench 每条 run 的 tool_use 数与 `solve_parsed.txt` 一致;`tests/data/` 样本 + pytest 通过。对应待办 4、13。

**Phase 1(实验机)**:验证 Harbor 0.22.0 的 GPU 直通、容器内多卡 torchrun、compose 卷挂载、`allow_internet=false` 下容器内 agent 出网;AIRS adapter 生成 scope 内 8 题;冒烟仍用 SICK 目录(已生成,出 scope 后保留作样例)用 claude-code 端到端跑出 `reward.json`;PostTrainBench 原生 mock prompt 端到端跑通;PI task 目录 1-trial baseline 复现 `step:3290/3290 val_loss≈3.277`。对应待办 14。

**Phase 2**:`awm` agent CLI + 两种包装;S1 最小自跑。

**Phase 3**:层级恢复指标(待办 5)、双层拆分(待办 12),先在 Tier A 轨迹上做。

## 九、待验证假设

1. Harbor 0.22.0 的 GPU 支持(AutoLab 在 0.3.0 上是打补丁才有的)。
2. Harbor 在容器内跑 CLI agent 时是否保留原生 stream-json(否则自跑轨迹要自己 tee)。
3. PostTrainBench `run_task.sh` 脱离 HTCondor 直接调用是否顺畅。
4. PostTrainBench Harbor 适配器 PR #8 是否还能 rebase 到 main(仅在将来切 Harbor 时需要)。

## 十、Phase 0 实施记录(2026-08-24 完成)

**交付**:`awm/` 2,700 行 + `tests/` 1,450 行;134 项测试通过(挂载数据卷),无数据卷时 128 通过 + 6 跳过;`ruff` 干净。

**验收结果**(数字均为独立重算,未采信实现者自述):

| 项 | 结果 |
|---|---|
| PI:主 agent 事件数 = manifest `n_events` | 41/41 精确相等,合计 172,694;另 118,980 条子 agent 事件 |
| PI:`validate_stream` | 41/41 通过,`write_run` → `read_events` 往返后仍通过 |
| PI:子 agent 数 = manifest `n_subagents` | 41/41(含 id 复用那条:66 条索引 → 40 个唯一 id,26 条被覆盖的记入 `extra`) |
| PI:token 与 manifest `economics` | 修复后 codex 7/7、qwen-code 2/2、prime-agent 6/6 **逐 token 相等** |
| PTB:转换成功率 | 82/82(第 83 个目录是 HF 的 `.cache`,正确跳过),73,407 事件 |
| PTB:claude 的 `tool_use` 数 = 上游 `solve_parsed.txt` 的 "Tool call" 行数 | 38/38 |
| PTB:带时间戳的 run 数 | 62 = 38 claude + 24 有前缀的 codex,与逐 run 实测吻合 |
| scope:与文档表格及 AIRS 上游 `metadata.yaml` 对账 | 53 条,120 项字段比对 0 处不一致;21 项 schema 故障注入、8 项 registry 故障注入全部捕获 |
| 端到端 | `index.parquet` 123 run × 28 列;PI 最佳记录 2726 / 2920 / 2930 / 2974 / 3018 与公开排行榜一致 |

**与本文档原方案的偏差**:

1. 数据根改为 `/data/gangda/hv`(`/data` 是多用户共享盘),由 `AWM_DATA_ROOT` 覆盖。
2. `snapshot_download(allow_patterns=...)` 在 218k 文件的仓库上十分钟不落盘,改为自己分页列表(缓存)+ 精确并行 `hf_hub_download`:463 文件 / 0.59 GB 约一分钟。
3. 事件 schema 相对第五节增加两个字段:`tool_use_id`(结果与子 agent 的溯源锚点)、`truncated`(上游截断标记);并明确 `USAGE_KEYS` 的口径——`in` 是否含 `cache_read` 由各 harness 决定、我们不调和,跨 harness 求和无意义。
4. `RunMeta.flags` 收紧为"只放判定":溯源进 `source_paths`、描述性元数据进 `extra`。否则 `judgement_file`、`fidelity=full` 这类值会把 123 条里的 91 条误标为有问题(修复后为 6 条:PI 3 条 `validity=flagged`、PTB 3 条污染)。

**上游事实的三处更正**(原调研报告有误):

1. **时间戳前缀逐 run 变化,而非逐 harness**:同一个 agent 配置内部都不一致(`codex_non_api_high_gpt-5.4` 里 20 条无、4 条有)。必须逐行判断。
2. **PI 的 `usage` 不是简单"每 turn 重复"**:实测 564 条流里 11,854 个 turn 的 usage 记录并不全等,且 codex 每个 session 重置 turn 编号(单条 run 就有 6,583 次重置)。只保留 turn 首条会丢掉整次 API 调用——codex 的 output token 只剩 0.40–0.92 倍、qwen-code 只剩 0.00–0.08 倍。正确做法是把一个 turn 内**相邻且不同**的 usage 记录相加。
3. **WinoGrande 的 SOTA 是 T5-3B 全量微调,不是 LoRA**(依据其 `metadata.yaml` 的 `sota_notes`);《评估原则》G1 一节已更正。

**已知遗留**:codex 跨 session 复用 turn 号,token 总量正确但 run 内归属偏粗(`session_id` 保留在 `Event.extra`,分析层可重组);263 条 codex 与 212 条 grok-cli 子 agent 无法链接父调用(上游未记录 spawn 或无时间戳),`parent_id` 已保留;`tool_result` → `tool_use` 未配对(codex 主流里 13,654/33,707 条结果前面没有对应调用,配对只能靠猜)。

## 十一、对现有文档的影响

- 《案例与数据清单》3.4:"轨迹自跑 AIRA-dojo(搜索树自带 parent/child 层级)"须改为"经 Harbor 用 CLI agent / 我们的 agent 自跑,搜索结构来自 agent 遥测";公开版 aira-dojo 跑不了 AIRS。
- 3.2:补一句"单一 prompt 模板 × (4 模型 × 7 基准)实例化,差异只在评分器"。
- 3.5 / 四:PI 的 run.sh、verify.py 虽未随仓库公开,但可从轨迹逐字恢复。
- 四:PostTrainBench-Trajectories 实测 28.9 GB / 218,361 文件 / 1,842 run;MALT 为点击式 gated。
