# exp_protocol 自迭代线:任务简报

**日期**:2026-09-01 **状态**:生效 **给谁**:负责分支 `gangda_exp_protocol_evolve` 的 agent **配套**:`doc/reference/exp_protocol_and_wma_policy.md`(定义)、`doc/spec/2026-09-01-exp-protocol-card-v2.md`(卡片)、`skills/exp_protocol_meta/`(循环)

---

## 一、你是谁,负责什么

- **分支** `gangda_exp_protocol_evolve`,worktree `/data/gangda/worktrees/agentic-world-model_exp-protocol-evolve`。PR 目标是 **`gangda_trial_0828`**;**永远不要主动 PR 或 merge 到 `main`**。
- **负责两件事**:(1) 让 `exp_protocol_meta` 描述的自迭代循环真正在 H100 上跑起来,并据此改进 `skills/exp_protocol/`;(2) 下面第三节的两个接线决策。
- **不负责**:WMA policy 的设计与实现——那是主线 `gangda_trial_0828` 上另一个 agent 的事。两条线通过 experiment card 交换信息,各自独立迭代(定义文档第五节)。
- **用户会直接和你对话**。每个需要拍板的决策,先给方案、利弊、推荐,再动手。

## 二、上手必读(按顺序,约 40 分钟)

1. `doc/reference/exp_protocol_and_wma_policy.md` —— 定义与判定规则:能写成 checklist/lint/schema/脚本的归规程;需要权衡的归策略。
2. `doc/spec/2026-09-01-exp-protocol-card-v2.md` —— 卡片 v2 的增删理由,锁的覆盖范围与威胁模型(**痕迹,不是屏障**)。
3. `skills/exp_protocol/SKILL.md`、`card.template.yaml`、`example-card.yaml`、`pitfalls.yaml` —— scientist 看到的全部。
4. `skills/exp_protocol_meta/SKILL.md`、`metrics.md`、`iteration_record.template.md` —— 你自己的循环。
5. `awm/exp_protocol/*.py` 与 `tests/test_exp_protocol_*.py`(123 个测试,全部 CPU)——先跑一遍:`python3 -m pytest tests/test_exp_protocol_*.py -q`。
6. **`AGENTS.md` 前半段**:H100 四节点的 Slurm 队列所有权规则(`gangda` 队列、registry、`OWNERSHIP FAIL`、节点保留)。这是硬约束,违反会影响别人的实验。配套 `doc/reference/gangda_slurm_queue.md`、`tools/awm-slurm-queue`。
7. `doc/reference/harness_facts/posttrainbench.md` §3–4 —— `run_task.sh` 的流程、agent `solve.sh` 的接口(`$PROMPT`、`$AGENT_CONFIG`、cwd `/home/ben/task`、写 `final_model/`)、结果目录布局。
8. `awm/ptb_experiments.py` + `experiments/posttrainbench/*.yaml` —— 主线的批次发射器(`awm ptb check|dry-run|submit`)和 `APPROVED_AGENT_SETUPS` 白名单;它从 `third_party/PostTrainBench/agents/<agent>/{solve.sh,api_keys.json,profile.env}` 读 agent。
9. **只读参考**:jerry-dev 的 worktree `/data/gangda/worktrees/agentic-world-model_jerry-dev`,看 `rollout/agents/claude_wm/solve.sh`(如何把仓库 clone 进沙箱 `/home/ben/awm`、加 `PYTHONPATH`、把 `.claude/` 拷进 `/home/ben/task`)、`rollout/patches/apply_extra_binds.py`(给 `run_task.sh` 加只读 bind 的幂等补丁)、`rollout/setup.sh`(私有 checkout 模式)。**不要改 jerry-dev。**

## 三、第一步:两个接线决策(C 项)

### C1 · `install` 怎么进 PTB 沙箱

目标:每个 scientist cell 在拿到 prompt **之前**执行
`awm exp_protocol install --target /home/ben/task --tool <claude|codex>`,
其中 `AWM_EXP_PROTOCOL_DIR` 指向**该变体**的 `skills/exp_protocol` 检出;沙箱内 `awm` 可 import(Jerry 的做法:clone 仓库到 `/home/ben/awm`,`PYTHONPATH` 指过去,`printf` 写一个 `awm` 入口脚本)。scientist **绝不能**看到 `skills/exp_protocol_meta`(`install` 已拒绝拷贝它,但沙箱里如果整仓 clone,meta 就在 `/home/ben/awm/skills/` 里可读——要么只 clone 需要的、要么 bind 只读的单个目录、要么装完就删)。

两条路,先给用户比较后拍板:
- **A. 在 PTB fork 上新加一个 agent 目录**(例如 `agents/claude_vertex_max_exp/solve.sh`,复制 `claude_vertex_max/solve.sh` 加装 install 步骤),走 fork 的 PR 流程(feature 分支 → PR 到 fork 的补丁分支 → 合并 → 更新主仓库 submodule 指针;命令见记忆 `fork-pr-workflow`),再把新 agent 加进 `awm/ptb_experiments.py` 的白名单。优点:与主线发射器完全兼容、可审计;缺点:改 fork 要走流程、变体切换靠环境变量。
- **B. 私有 checkout 模式**(Jerry 的 `rollout/setup.sh`):在 H100 上维护一份私有 PTB checkout,setup 脚本往里装 agent 与补丁,不动 fork。优点:快、变体随便切;缺点:与主线发射器的 receipt/registry 体系脱节,审计弱。

验收(本地、CPU):用一个模拟的 `/home/ben/task` 目录跑通 `install → new → check → lock → close`,并证明 `exp_protocol_meta` 不可见。**不跑 GPU。**

### C2 · meta 循环第一轮的参数

- **baseline** = 合并后 `gangda_trial_0828` 上 `skills/exp_protocol/` 的 commit(`git log -1 --format=%h -- skills/exp_protocol`)。
- **第一轮建议只跑 baseline**,≥3 个 seed,先拿到基线分布(`accuracy` 的方差、`pitfalls_cost_h`、`fields_filled`),再谈候选变体;或者 baseline vs 一个最小变体。给用户选。
- **任务与 held-out**:提方案——例如 gsm8k 迭代、aime2025 held-out(两者都在 `APPROVED_TASKS` 里);base model 从 `APPROVED_BASE_MODELS` 选一个固定;scientist 用 `APPROVED_AGENT_SETUPS` 里的一个固定。
- `PTB_NUM_HOURS`、cell 数、预算——按 `AGENTS.md` 的队列规则和用户给的额度。
- **全部在 H100 上跑,用 `awm ptb` 发射器**;本地只做 smoke。
- 每轮记录到 `doc/exp_protocol_iterations/<date>-round-NN.md`(模板在 `skills/exp_protocol_meta/`),**先写记录再改规程**。

## 四、第二步:让循环真跑

- `awm exp_protocol collect` 在**真实结果目录**上验证一次(`session` 标签应为 `<cell>/task`,`metrics.json` 在 `task/` 的父目录)。
- 读三张真实卡片,对照 `metrics.md` 的"Reading them together"。
- 循环规则(来自 `exp_protocol_meta/SKILL.md`):变体 = `skills/exp_protocol` 的 commit;其余全部固定;每变体 ≥2 seed;held-out 任务不迭代;每轮一个改动、可追溯;`n_overrides` 高的检查是该修的检查,不是该怪的 scientist。
- 可选:把 `awm traj spans` 接进 `collect`,补训练/空闲时长指标。

## 五、工作规则

- **先对齐再动手**:每个决策点先给用户方案与推荐;用户批准后再改代码。
- **TDD**;规程线的一切机制必须能在 CPU 上端到端测试;不引入状态机、不引入等待训练的命令、不引入需要 LLM 才能执行的规则。
- 提交风格:英文陈述句标题 + 说明为什么的正文;trailer `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- PR base **`gangda_trial_0828`**;绝不 `main`。
- worktree 一律放 `/data`(根盘 93% 满)。
- 推送用 HTTPS + `gh`(账号 Hydrapse);本机默认 SSH key 是另一个账号,无写权限。
- 若要改 PTB fork:feature 分支 → PR 到 fork 的补丁分支 → 合并 → 更新 submodule 指针。

## 六、已知状态与坑

- 主工作区 `tests/test_ptb_experiments.py::test_manifest_is_exact_approved_matrix` 在本机因 `data` 软链解析失败,H100 上应过;不要为它改代码。
- 在 worktree 里 `git submodule update --init` 直连 GitHub 可能失败(沙箱网络);用 `--reference /home/gangda/workspace/DeepCommit-ai/agentic-world-model/third_party/PostTrainBench`。
- Codex:本机 0.152.0,用户级 skill 路径 `~/.codex/skills/`;沙箱内版本见 `harness_facts/posttrainbench.md` §3 的容器清单。
- 审核遗留的两个 Minor 未处理(有意):`lineage.load_cards` 把 `_path` 塞进卡片字典;`exp-100` 会排在 `exp-99` 前。
- 上一轮审核修复的完整清单在 PR #9 的评论里,可作为"什么容易出错"的参考。
