# exp_protocol 本地 Claude trace 分析合同

**生效**：2026-09-03 09:39 UTC，用户决定。**取代的协作阻塞**：planner 不再等待 Fable 在 PR
回复或推送 commit；PR 只保留审计与非阻塞信息。本地 Claude Code CLI 是分析助手，Codex planner
仍拥有收割、证据判定、候选选择、代码改动、队列与 git 的最终责任。

## 触发

`tools/exp_protocol_completion_monitor.py` 每 5 分钟只读一次指定 Slurm job 集合；达到 8 个 terminal
jobs 时写 `data/ptb/monitor/exp_protocol_goal.json` 并退出。terminal 只是唤醒信号，不是科学完成：
planner 随后运行 `awm ptb reconcile --apply`、提交收割物，并以 PTB validator + receipt provenance
数 clean cells。只有自上个 trace window 以来达到 8 个新 clean cells 才启动分析；不足则重启 monitor。

项目 `.codex/hooks.json` 的 `SessionStart` hook 只在 goal resume/compact 时把 ready event 注入上下文。
它不声称外部事件能主动创建一轮；持续 `/goal` 才是后台续跑机制。

## “Opus 5 ultracode”的本机映射

已安装 Claude Code 2.1.259 的 effort 枚举是 `low|medium|high|xhigh|max`，没有名为
`ultracode` 的 CLI 值。本线把用户所说的 “Opus 5 ultracode” 固定为：

```bash
claude --model 'claude-opus-5[1m]' --effort max --background \
  --permission-mode plan --permission-prompts none \
  --tools 'Read,Grep,Glob,Bash' '<review prompt>'
```

后台命令返回的 session id 必须写进本窗口 synthesis。planner 用 `claude logs <id>` 读取结果；
Claude 只读 evidence 并输出报告建议，不 release/cancel job、不 push、不决定晋升。

## 每个分析窗口

1. planner 先生成每个 clean cell 的 `tools/exp_protocol_cell_read.py` 与
   `tools/exp_protocol_trace_timeline.py` 事实输出。
2. 按 `skills/exp_protocol_meta/trace_review.md`，每个 Claude session 读 3–4 个 cell；覆盖本窗口全部
   clean cells，一 cell 一报告，明确每个 ≥0.1 h 损失、已有候选覆盖或 `uncovered`。
3. 另起一个 Opus 5[1m] max session 读所有报告做 synthesis：分数差解释排名、两臂计数、候选来源、
   单项 surface、4-cell 指标、score guardrail、备选与反证。
4. planner 自己完整读 synthesis、两臂最好/最差报告、每 variant 三张卡，再决定。Claude 建议不是
   evidence gate，也不能把 control recipe 泄漏成 protocol 指令。
5. 结果、session ids、planner 的接受/拒绝理由进入 `doc/exp_protocol_iterations/` 并提交；无需等待 PR
   上的任何 agent。
