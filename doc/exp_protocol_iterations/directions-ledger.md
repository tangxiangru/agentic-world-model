# exp_protocol 改进方向台账与决策日志

**用途**：一处看全这条线上出现过的每个改进方向——它从哪些 cell 来、现在处于什么状态、为什么、什么证据会改变它的状态；以及每个影响实验设计的决策、当时的备选与理由。逐轮的证据表在 `doc/exp_protocol_iterations/<date>-round-NN.md`，逐候选的冻结设置在 `doc/spec/`。本台账随每个分析窗口更新，状态只增不删。

状态含义：**已采纳**（进了基线树）· **筛选中**（Round 02 起的 4-cell screen）· **排队**（有证据、等 slot）· **观察**（不是规程能改的事，只记数）· **搁置**（证据消失或被更强的候选压后）· **已否决**（评估后放弃，附理由）。

## 一、方向台账（截至 2026-09-03 00:10 UTC，Round 00 首波 9 对 7）

| # | 方向 | 来源 | 证据现状 | 状态 | 理由 / 改变状态的条件 |
|---|---|---|---|---|---|
| 1 | **会话结束杀掉训练**（Stop hook + 规则 9 + pitfall `run_dies_with_the_session`） | p00r08（9.4 h 作废）、c00r02（7.6 h 作废）、pilot 90463/90464 | harness 的确定性性质；两臂各丢一个 cell | **已采纳为 Round 01 候选**，commit `4ae3d87`，8 个 cell 在 ondem-0 重跑中 | 判据在 `doc/spec/2026-09-02-exp-protocol-round01-session-guard.md`：无 cell 再因会话结束丢训练、hook 阻止次数、accuracy 不低于基线 −0.03。通过则成为新基线树 |
| 2 | **解码配置**：评分器继承 `generation_config`，Gemma 默认采样；关键是检查 grader observable | protocol: +5.4 到 +16；**c01r03 control 同权重 +6.1**；controls 8/8 greedy | protocol greedy 8/13，均值 0.7176；sampling 5/13，均值 0.6458，差约 7.2 pp | **筛选中：Round 02 A v2**（`f6cdccc`；held 91046–91049） | 目标 ≥3/4 交付 greedy/有测量依据配置；所有新 adopter 都按 server-log observable 行动 |
| 3 | **vLLM 离线采样默认值**：重复 `<bos>`、stop ids 不生效、parser inf、孤儿引擎 | protocol 多 cell；controls 约 5 h，**c01r03 首轮也因无 explicit stop 得到 bogus pass@1** | 两臂都中；不是 arm-gap 机制，但浪费可机械避免 | **筛选中：Round 02 B v2**（`9f294c3`；held 91050–91053） | B mechanisms 的 pitfall h <0.3/cell；model weak-stop 单列观察 |
| 4 | **评估样本数**：n 必须承载所声称的差距 | synthesis 4 cells/7 reversals；p00r12 是第 5 cell/第 8 次；**p00r11 dev-150 第三名在 full 变第一** | first-N 既 noisy 又偏易；两个新 cells 都靠 500/full 重评修复选择 | **筛选中：Round 02 C v2**（`57511f9`；held 91054–91057） | 每张卡 n≥500 的 cells ≥3/4；无 later reversal；`falsified_if` 不落在 sub-SE delta |
| 5 | **eval-only 卡片被迫编造 `setup.data`**（`data_files_exist` override） | 首波 7/9 有 override/placeholder；p00r11 4 次，**p00r13 再 4 次** | measurement-card 直接 override 累计 10 次；rule 7 正确禁止本地 benchmark test 文件 | **筛选中：Round 02 第二波 H**（`b52e5f2`；held 91068–91071） | schema/questions 只在训练 family 要 data；目标零伪造 entry/override 且 fields_filled 不降 |
| 6 | **TRL 截断掩码的 eos 变体**：`mask_truncated_completions` 用 tokenizer eos（1）而 Gemma 停在 `<end_of_turn>`（106），loss 全零 | c00r06（一分钟内自修）、c00r05、c00r01 | 只在对照的 GRPO 里出现，因为规程臂没人跑 RL | **排队** | 现有 `eos_mismatch` 条目只写了 SFT 目标那种变体；等规程臂出现 RL 卡再补，否则条目没有本臂的 source |
| 7 | **预算使用**：两臂都提前收工；新 protocol 余 1.8–3.1 h，c01r03 control 余 1:16 | clean cells | 未证明为臂间差异 | **观察** | 不诱导无意义跑；Round 02 后仍系统性早停再设计 |
| 8 | **on-policy RL 的采用**：protocol 0/13，对照 5/8 | c01r03 0.7786、无 RL；RL controls 均值 0.7569，non-RL 0.7602 | adoption 不解释 arm mean；只可能影响 ceiling/framing | **观察** | A/C 后仍有 gap 才测试 framing/G |
| 9 | **`stop_token_consistent` 误报**（见 #17：代价以数据重写而非 override 出现）：脚本追加终止符时 check 读文件报 FAIL | pilot 90462/90464 各 override 一次 | v3 正式 cell 里 0 次 override（scientist 改为把终止符写进数据） | **搁置** | 证据消失；若某个 v3/guard cell 再 override，恢复为候选（修法：接受"由脚本追加"的声明并校验一条渲染样例） |
| 10 | **bootstrap 文本进规程树**（现在由 PTB fork 的 solve.sh 前缀注入） | Round 00 设计时 | 与分数无关，是 meta 循环的可迭代性问题 | **排队（meta 回顾）** | 三轮后与 `exp_protocol_meta` 一起改；改法：`skills/exp_protocol/bootstrap.md` 由 `awm sandbox setup` 写入任务目录 |
| 11 | **仪式本身的代价**：写卡、锁、结案是否在花分数 | 首波：规程执行完美（66 卡全闭、39/39 先锁后跑）但 −0.073 | 现在无法与 #2–#4 分离 | **假设，待 Round 02** | 若 A/B/C 把知识装进 pitfalls 后规程臂仍追不上对照，下一步测"更轻的规程变体"（例如只保留 pitfalls 提醒 + 锁）而不是继续加条目 |
| 12 | **对照 c00r06 与 batch 1 的 0.7832 完全相同** | c00r06 REPORT vs batch 1 结果包（在集群上） | 答对题数相同（1033/1319），配方是否相同未核 | **待核实** | operator 比对两份 REPORT 后写入记录 |
| 13 | **GSM8K-train 人写解法的简洁风格拖分 ~22 点** | p00r10 exp-02/03、p00r03 exp-03（10-shot 前缀拷贝演示风格） | 数据配方发现，两个 cell 各自独立发现 | **观察（知识，非规程）** | 若再出现，作为 pitfalls 的"数据风格"条目候选；现在不加是因为规程不该规定训练数据 |
| 14 | **父 checkpoint 的 greedy 配置让 Trainer 首次保存失败** | 原 5 cells 约 5.4 h；**c01r03 再丢 76 min** | 一个根因，六个 cells、两臂都有；是 A 修法制造的陷阱 | **筛选中：Round 02 第二波 D**（`8332917`；held 91060–91063） | 该错误小时数 0 且 check 触发；stock config 零误报 |
| 15 | **按时钟等而不是按进程等**：固定 `sleep 3000+` 错过已死的 run | p00r02（1.28 h 空转）、p00r07（1.2 h）；对照 c00r01/c00r08 按 PID 等则没有损失 | 规则 9 与 hook 文案里 `sleep 900; tail` 的写法本身就是诱因 | **筛选中：Round 02 第二波 E**（`7832cb9`；held 91064–91067） | 目标：run 死亡到下一命令的空转 < 0.15 h/cell |
| 16 | **仪式本身的代价（承接 #11）** | synthesis §3.4：直接成本 0.17 h/cell 不是分数；间接成本是决策框架（小步、固定小 n 的 `falsified_if`、逐卡时间算术总选更短的选项） | 与解码配置在 n=9 时无法分离 | **假设，待 A v2 的 4 个 cell** | 若 ≥3/4 交付 greedy 而 A 块仍落后漂移对 ≥ 0.03，残差就是框架，下一步先执行『停止做的事』清单 |
| 17 | **`stop_token_consistent` 读原始字段迫使 3/9 个 cell 重写数据** | p00r01 L4559、p00r03 L5152、p00r07 L4320；**p00r12 L4796+L3126 数据里追加一次、`encode_rows` 再追加一次（潜在双终止符，RFT 未跑到）** | 台账 #9 的『0 次 override』低估了代价：scientist 改数据而不是 override；歧义的另一种形态是重复追加 | **排队（I）** | 接受 `stop_token: {{value, appended_by: script}}` 并校验 dry-run 的一条渲染行 |
| 18 | **trajectory weight averaging / soup** | p00r15 +4.5@200 且 0.734@500；p00r11/p00r13/p00r07 contradicted | 效果混合，p00r15 4-way 又是 post-hoc best-of-four | **观察（配方，非 protocol）** | 再现时预注册 checkpoint set/权重并用 ≥500；不把“跑 soup”写进 protocol |
| 19 | **同一 greedy artifact 的 vLLM full-eval run-to-run variance** | p00r13 identical weights/config full 0.7051–0.7142；4-run mean 0.7106、sd 0.0036；p00r09 也见同权重波动 | 单次约 1 pp spread，与很多卡片声称的 delta 同量级 | **观察（归入 C 的 uncertainty）** | 小于约 1 pp 的差距需 paired/repeated read；先看 C 是否解决 selection，不单开候选 |

## 二、决策日志

| 时间（UTC） | 决策 | 备选 | 理由 | 记录处 |
|---|---|---|---|---|
| 09-02 12:20 | 不走 pilot；16 卡常满；边跑边分析；发现不行撤 PENDING | pilot-first 门 | 用户指令 | iteration-basis §七 第 1–4 条 |
| 09-02 12:40 | Round 00 = 规程 v3 对无规程对照，各 16 | 只跑 baseline ≥3 seed | 没有零点就无法说"规程买到了什么" | round00-null-control spec、round-00 记录 |
| 09-02 14:40 | Round 01 候选 = session guard，8 个 cell，与 baseline-b 同波 | `stop_token_consistent` 修 check | p00r08 的失败是确定性的、两臂都中、可在本机真实 `--print` 下验证 | round01 spec、round-00 记录 Decision/Change/Evidence |
| 09-02 19:30 | 溢出 cell 记为 quarantined 且保留分数；主口径只算严格站点 | 判 incomplete 丢弃 | placement 是 provenance 不是变量，硬件相同 | PR 评论、`a9e69a1`/`00b39aa` |
| 09-02 20:40 | held 缓冲 + 放行门第 4 条（原生两节点隔离）保留 | B 方案：去掉第 4 条，靠启动后审计 | 用户向 planner 明确"不能再溢出"，宁可空转 | PR 评论 5516176003/5516241300、iteration-basis §七 |
| 09-02 ~21:15 | 每候选 4 cell 筛选、赢家补到 8、一轮并行多候选、每波 2 个 baseline | 固定 8/候选 | 用户："8 个太多不利于发现"；n=4 能分辨 0.06，目标指标是机制性的 | iteration-basis §四/§五/§七 第 5 条（`d89fe2f`/`5756c9a`） |
| 09-02 22:05 | 提议取消 baseline-b、阶梯式 A→B→C | 独立变体 | 线性分支上"每候选只改一项"难以构造 | PR 评论 5516992281 |
| 09-02 23:05 | **planner 否决阶梯，采用独立变体**（先回退再添加） | 阶梯 | B=A+B 相对基线是两项改动；A 失败会污染 B/C 的护栏 | PR 评论 5517615610、round02 spec §二 |
| 09-02 23:20 | 取消 baseline-b（8 个 PENDING），Round 02 held 登记后 | 保留精度扩展 | 整块撤回不构成挑选；让 Round 02 早一波 | `696dc6b`/`3bb4a07` |
| 09-02 23:45 | 候选 C 措辞由 planner 修正 | 我的原文 | "抛硬币"表述不可辩护 | `7f117a0`、`ca90b31` |
| 09-03 00:20 | **meta 循环加入 subagent trace review**：每个分析窗口由 reviewer subagent 分组读全部 cell 的 trace、synthesis subagent 排名解释并提候选；`exp_protocol_meta` 据此修订（用户授权在轮中修改） | 继续由我手工逐 cell 读 | 用户要求 meta skill 像本次一样启动 subagent 批量分析 trace 再提出新一轮修改，全部自主执行；手工读 16 个 cell 不可持续，且 Round 00 证明数字说不出原因、trace 才说得出 | iteration-basis §七 第 6 条、`skills/exp_protocol_meta/trace_review.md`（`fe7895e`） |
| 09-03 02:27 | **按 trace review 改写 A/B/C（v2，旧 held job 撤回），新增第二波 D、E、H + 漂移对 B，全部 held** | 保持 A/B/C v1 原样；把 D 并入 A；把 F/G 也排进第二波 | synthesis 的证据：A 的关键是怎么核实；B 的来源写错了；C 的示例卡本身在教 n=150；D 五个 cell 5.4 h；E 两个 cell 2.5 h 空转；H 7/9 伪造。F/G 是配方知识或针对信念的测试，证据面窄，先排队 | round02 spec §六、`2f64581` 前的六个 commit |
| 09-03 02:38 | **两个 drift pair 改 ship `2f64581`；已登记的旧 drift A 整块撤回并以 v2 替代** | 继续用 `4ae3d87`，只因 protocol_tree 相同 | 候选与 `2f64581` 的六个 shipped paths 仅差单项；与 `4ae3d87` 还差共同的 `awm/exp_protocol/collect.py` 基础设施。单 manifest check 不证明跨变体同代 | round02 spec §六、round-00 trace-review provenance addendum |

## 三、这份台账之外还没写下来的

- Round 01 与 Round 02 的结果记录（`<date>-round-01.md`、`-round-02.md`）在各自的 cell 落地后另起。
- `exp_protocol_meta/iteration_record.template.md` 目前没有"方向台账"一节；三轮后的 meta 回顾时加上，并把 `metrics.md` 补上 `hours_used`、"greedy 是否交付"、"最大评估 n"三项。
