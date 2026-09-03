# Round 02 · E 的条件替代候选:G 与 P1(单项、已构建、未排队)

**日期**:2026-09-03 09:05 UTC **状态**:候选 commit 已在分支上,manifest 未建 **前置**:`doc/spec/2026-09-02-exp-protocol-round02-independent-screens.md` §七(planner 裁决 `5f3a69a`:若 Round 01 guard ≥7/8 的 cell 已满足 E 的 <0.15 h 目标,E 的 held 块 91064–91067 在启动前整块撤回,并由 G 或 P1 中**一个**独立单项候选替代)、`doc/exp_protocol_iterations/2026-09-03-trace-review-round00-window02-fable.md` §3–§4。

## 一、为什么现在就构建

替代块只有在 guard 判定之后才登记,但构建 commit 不占任何队列资源,而先建好可以让撤回 E 与登记替代块在同一次操作里完成,可运行队列不出现空档。两个候选都建好,planner 届时按证据只选一个;未选的那个保留为排队方向。

## 二、构造

两者都从 guard 树 `189319d6` 出发,各自只多一条 `pitfalls.yaml` 条目(`check: null`,preflight 每次作为提醒打印),用 revert-then-add 建在线性分支上;head 在 `a4c4954` 之后回到 guard 树。

| 候选 | 唯一一项 | commit | `skills/exp_protocol` 树 | 相对 guard 的 diff |
|---|---|---|---|---|
| G | `pitfalls.yaml` 新增 `trl_grpo_gemma_zero_gradient`:TRL `mask_truncated_completions` 以 tokenizer eos(1)判定截断而 gemma-3 轮次止于 `<end_of_turn>`(106),loss/grad_norm 全零、clipped_ratio 1.0;rollout 提示超 `max_model_len` 直接中止;3 步 smoke 读 grad_norm | `f28dd88` | `4ea16d13` | `pitfalls.yaml` +6 |
| P1 | `pitfalls.yaml` 新增 `rft_from_fitted_parent`:从生成样本的同一 checkpoint 出发的 RFT/STaR 轮次首步 loss 即父模型末 loss、跑完等于父模型;前 20 步可判 | `a4c4954` | `7294c236` | `pitfalls.yaml` +6 |

两条目都只描述失败签名、原因与几分钟内可读到的判据,不规定是否要跑 GRPO 或 RFT。

## 三、来源 cell

- **G**:对照臂跑过 GRPO 的 7/7 个 cell 都命中同一根因——c00r01、c00r04、c00r05、c00r06、c00r08(window 01),c01r01(L9099→L9157→L9330,0.05 h 损失但 25 min 才发现),c01r02(L3909→L4022→L4116,3 步 smoke 3 min 发现;L4252 rollout 提示超长在第 14 步崩溃,0.25 h)。规程臂 0/14 跑过 GRPO,所以这条目在规程臂**没有直接的 baseline 损失**;它测的是"知道这个陷阱之后是否有 cell 敢于/能够跑通 on-policy RL",这正是台账 #8 的问题。
- **P1**:p00r13 exp-04(3.1 h,0.7119 对 0.7142 全集,McNemar p=0.49)、p00r15 exp-06(L9518,1.1 h,loss 0.271→0.274)、p00r06 exp-05(window 01)、p00r11 exp-08(从第一轮 RFT 的父模型再来一轮,−3.1 pp,1.2 h)、c01r02(2.3 h,−0.4)。两窗口 5 个 cell,约 9 h,两臂都有。

## 四、每个筛选读什么(4 cell)

| 候选 | 主指标 | 次指标 | guardrail |
|---|---|---|---|
| G | 出现 `family: grpo` 卡的 cell 数,以及该卡从首次启动到日志出现非零 grad_norm 的时间 ≤0.5 h(对照 baseline:c01r01 0.42 h,c01r02 0.05 h) | 归因于 zero-gradient 或 rollout 超长的 `pitfalls_hit` 小时数 <0.5 h/cell;失败的 GRPO run <1 h | 块均值 ≥ 当时规程池均值 −0.03;**不把"跑了 RL"当作分数 KPI**,条目若把 cell 推进一次 >1 h 的无效 RL,即为负面证据 |
| P1 | `family: rft` 且 `training_summary` 报首步 loss 平坦、仍跑完的卡的小时数 <0.5 h/cell(本波 baseline:3.1 / 1.1 / 1.2 / 0 / 0) | 无第二轮从 RFT 父模型出发的 RFT;≥1/4 cell 仍尝试过 RFT(条目不得被读成"永不 RFT") | 块均值 ≥ 规程池均值 −0.03 |

## 五、选择依据(留给 planner,届时按证据)

- 若 Round 01 guard 的 8 个 cell 里有 ≥2 个 cell 在 RFT 上出现 P1 的签名,而 GRPO 仍是 0/8:P1 有 baseline 可移动,G 没有 → 选 P1。
- 若 guard cell 里出现了 GRPO 卡(哪怕失败):G 有了规程臂的第一份 baseline → 选 G。
- 两者都不满足时,按小时数选 P1(两窗口约 9 h 对 G 在对照臂的约 0.6 h)。

## 六、manifest(未建)

登记时各 4 cell:`exp-protocol-gsm8k-gemma4b-high-r02-g-grpo-zero-grad-x4`(cell `z02r01–04`)或 `…-r02-p1-rft-fitted-parent-x4`(cell `f02r01–04`),`want: held`,shipped 路径与漂移对同代(`awm.sha` 取当时 head、`protocol_tree` 取上表的树);其余字段照 `…-r02-h-eval-only-data-x4`。
