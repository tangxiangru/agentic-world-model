# exp_protocol 改进方向台账与决策日志

**用途**：一处看全这条线上出现过的每个改进方向——它从哪些 cell 来、现在处于什么状态、为什么、什么证据会改变它的状态；以及每个影响实验设计的决策、当时的备选与理由。逐轮的证据表在 `doc/exp_protocol_iterations/<date>-round-NN.md`，逐候选的冻结设置在 `doc/spec/`。本台账随每个分析窗口更新，状态只增不删。

状态含义：**已采纳**（进了基线树）· **筛选中**（Round 02 起的 4-cell screen）· **排队**（有证据、等 slot）· **观察**（不是规程能改的事，只记数）· **搁置**（证据消失或被更强的候选压后）· **已否决**（评估后放弃，附理由）。

## 一、方向台账（更新至 2026-09-03 Window 03；原始来源保留）

| # | 方向 | 来源 | 证据现状 | 状态 | 理由 / 改变状态的条件 |
|---|---|---|---|---|---|
| 1 | **会话结束杀掉训练**（Stop hook + 规则9 + pitfall） | p00r08、c00r02、pilots 90463/90464 | strict8/8 clean，mean0.720148；57卡关闭、0会话结束丢失工作、0hook触发 | **预注册observed-no-harm门通过；非晋升**（`4ae3d87`） | 未触发hook不能证明阻断有效；历史对照不是同期因果；全部命令合规另见#26，ownership/native门仍关闭 |
| 2 | **解码配置**：评分器继承 `generation_config`；关键是 grader observable | Window 02 8/8 NEW 都 self-adopt greedy，same-weight +5.4 到 +20 | 最终 greedy 在 baseline 5/5 饱和；p00r11 首个 post-SFT eval 后仍等 3.3 h、用 4 张卡按 sampling 判定，其余 ≤12 min | **筛选中：Round 02 A v2**（`f6cdccc`；held 91046–91049，优先级在 D 后） | 4/4 在首次 post-SFT eval 后 ≤0.5 h 完成 measured decode choice；之后未测量 sampling decision card ≤1/cell；记录 observable 与 run-on rate |
| 3 | **vLLM 离线采样默认值**：重复 `<bos>`、stop ids 不生效、parser inf、孤儿引擎 | protocol 多 cell；controls 约 5 h，**c01r03 首轮也因无 explicit stop 得到 bogus pass@1** | 两臂都中；不是 arm-gap 机制，但浪费可机械避免 | **筛选中：Round 02 B v2**（`9f294c3`；held 91050–91053） | B mechanisms 的 pitfall h <0.3/cell；model weak-stop 单列观察 |
| 4 | **评估 n / repeated-read uncertainty** | W02/W03 noisy ranks；strict最终选择n≥500已有7/8 | 当前C v2的≥3/4通过条件已被baseline满足，不能单靠通过宣称增量；不否定uncertainty方向 | **撤回C v2整块91054–91057；待鉴别力更强的重设计** | 全部未启动，无按结果挑选；新screen须预先定义可移动的判据，不能事后把次指标当赢家标准；原tree/manifest保留 |
| 5 | **eval-only 卡片被迫编造 `setup.data`** | W01 7/9；W02 direct overrides 10；W03 三个 guard 共12个非训练 data entries、1个 placeholder 文件 | 0 overrides 仍有 workaround；原 zero-fake-entry 判据并非通过 | **筛选中：Round 02 H**（`b52e5f2`；held 91068–91071） | fake files、non-applicable entries、overrides 分列；真实适用可选引用不禁止，其余 required fields 完整性不降 |
| 6 | **TRL EOS zero gradient** | 原7/7 RL controls与c01r02命中；本批已读strict无GRPO暴露 | 原机制保留；无暴露不算修复或反证 | **排队（G）；`f28dd88` / `4ea16d13` 已构建未登记** | E已由指定cohort非饱和界限决定保留，当前不走G/P1替代分支；未来独立方向仍需可移动baseline |
| 7 | **预算使用**：两臂都提前收工；新 protocol 余 1.8–3.1 h，c01r03 control 余 1:16 | clean cells | 未证明为臂间差异 | **观察** | 不诱导无意义跑；Round 02 后仍系统性早停再设计 |
| 8 | **on-policy RL adoption** | v3 protocol 0/14；clean controls 8/15；Window03 NEW guard 0/3 | 方法使用率不是随机干预或 score KPI；不能用 recipe 差推断 protocol 因果 | **观察** | A/C 后若仍有可重复残差再测 framing/G，不规定 RL 方法 |
| 9 | **`stop_token_consistent` 误报**（见 #17：代价以数据重写而非 override 出现）：脚本追加终止符时 check 读文件报 FAIL | pilot 90462/90464 各 override 一次 | v3 正式 cell 里 0 次 override（scientist 改为把终止符写进数据） | **搁置** | 证据消失；若某个 v3/guard cell 再 override，恢复为候选（修法：接受"由脚本追加"的声明并校验一条渲染样例） |
| 10 | **bootstrap 文本进规程树**（现在由 PTB fork 的 solve.sh 前缀注入） | Round 00 设计时 | 与分数无关，是 meta 循环的可迭代性问题 | **排队（meta 回顾）** | 三轮后与 `exp_protocol_meta` 一起改；改法：`skills/exp_protocol/bootstrap.md` 由 `awm sandbox setup` 写入任务目录 |
| 11 | **仪式直接代价** | Window 02 protocol work 0.136 h/cell，first real train 比 controls 早约 4 min | 直接时间不是 gap；cards 修复多次错误 selection | **观察，直接成本基本反证** | 不因分数删 ceremony；只在 A/C 后测间接 framing |
| 12 | **对照 c00r06 与 batch 1 的 0.7832 完全相同** | c00r06 job 90472 REPORT vs batch 1 g12 job 87250 RESULTS/结果包 | 两者同为 1033/1319，但配方完全不同：g12 是 125k SFT + 28k few-shot pass + RFT/fresh-data 后三 checkpoint soup；c00r06 是 82k SFT + 三阶段 GRPO 620 steps | **已核实：不是配方复现；标量同分/独立收敛** | retained official bundle 无 1319-item per-item log，不能声称答对集合相同；不作为 protocol 候选，只保留为高方差/多路径同分证据 |
| 13 | **GSM8K-train 人写解法的简洁风格拖分 ~22 点** | p00r10 exp-02/03、p00r03 exp-03（10-shot 前缀拷贝演示风格） | 数据配方发现，两个 cell 各自独立发现 | **观察（知识，非规程）** | 若再出现，作为 pitfalls 的"数据风格"条目候选；现在不加是因为规程不该规定训练数据 |
| 14 | **greedy parent 使 Trainer save 失败** | W01/W02 两臂多 cell；W03 c01r04、c01r06、c01r07、g01r02 | 同根因再次出现；failed compute 与 post-exit wait 拆开，不能与 E 重复相加 | **筛选中：Round 02 D，第一可释放波**（`8332917`；held 91060–91063） | 错误小时数0、check exercised、stock零误报；未遇 unsafe parent 不算有效性证据 |
| 15 | **按时钟/非生产进程信号等待** | 原p00r02/p00r07；strict g01s03、g01s08的单次死亡后空等下界0.433/0.234h | 两个明确fail使≥7/8饱和不可能；E v2修正stale-tail⇒dead，并明确跨shell退出状态及current-run artifacts | **保留E；原jobs 91064–91067继续held；E v2 `c6f11d8` / `ceb68549` 已构建，manifest v2，未登记** | 三处等待指导、同代shipped paths、34项CPU测试通过；ownership FAIL期间不提交，替代receipt到位后才整块撤旧E |
| 16 | **间接 decision framing** | Window 02 both arms 8/8 greedy but gap −0.0565；protocol serial cards/RFT 5/5，controls broader plans | 与 initial SFT recipe 混杂，不能由 observational window 决定 | **假设，待 A/C screens** | A balanced 且 residual ≥0.03 才测 lighter framing；先不改 |
| 17 | **stop-token ownership / raw-field false check** | old 3 cells；NEW p00r12 latent double append、p00r14 5 overrides、p00r15 wrong-raw-field repair | 3 NEW manifestations，structural evidence；score harm未证明 | **排队（I）** | optional `appended_by` + rendered-row check；exercised RFT 评估后再排 wave |
| 18 | **trajectory weight averaging / soup** | p00r15 +4.5@200 且 0.734@500；p00r11/p00r13/p00r07 contradicted | 效果混合，p00r15 4-way 又是 post-hoc best-of-four | **观察（配方，非 protocol）** | 再现时预注册 checkpoint set/权重并用 ≥500；不把“跑 soup”写进 protocol |
| 19 | **同 artifact vLLM repeat variance** | p00r13/c01r02/p00r09；W03 g01r01/g01r02/g01s04/c01r07等 | 近阈值差异与重复波动同量级；serving config 差别另列 #24，不直接归因为并发 | **观察，归入 C screen** | ≤1 pp/≤1 SE 的 effect claim 读 paired/repeated；显式 tie-break 与证明提升分开；C 失败才考虑改字 |
| 20 | **RFT/STaR fitted-parent 风险** | strict g01s03 exp05、g01s08 exp06；g01s06 exp03盈利边界 | 两个低loss同parent暴露，但step20日志当时不可见；2.54h仅理想上界；冻结文本没有pure-self排除或精确0.05门 | **P1 v1 `a4c4954` / `7294c236`不登记；先校准可观测性与stop语义** | first20记录不等于step20可见；+1/500不是等效、−14/500不是已证明显著退化；不能以合成报告新增条件宣布旧规则安全 |
| 21 | **Gemma-3 logits OOM 与64 MB overlay安装故障** | 原≥20/24 cells；W03 两臂继续广泛暴露 | 机械性 bring-up 损失仍存在；不同修复路线成本差异大 | **排队（P2），未来独立单项冻结，不作为 P3 的混合 rider** | 目标仍为 OOM+装包小时下降及安全安装；不把两个独立机制叠进同一候选树 |
| 22 | **Trainer checkpoint 缺 processor/tokenizer 文件** | W01/W02 多cell；W03 c01r04、g01s04付费，其他cell预防 | 复制所有 base 配置会把 generation_config 也覆盖回 sampling；必须区分文件用途 | **排队（P3，独立于 P2）** | 单项修订须保留 measured decode config，不能笼统“复制所有 base configs” |
| 23 | **结束时按『最后一次运行的配置』而非『最小可改变决策的运行』定价剩余时间** | p00r11 exp-09（说 ~1 h，自身 RFT 周期 22 min）、p00r13 exp-06（2.2 h = 两 epoch）、p00r12 exp-09（余 3:05）、p00r15；w01 p00r05（余 3.7 h）、p00r10 | protocol ≥1.8 h 未用 6/14，control 1/10；按晚期边际收益约值 1 分 | **排队（P4，规则 8 措辞；在 D 与 A 的小时读数之后）** | 目标：余 ≥1.5 h 且无进程的 cell ≤1/4，且最后一次 `alternatives_rejected` 引用实测成本；是 #16 framing 的具体化 |
| 24 | **Serving/evaluation contract 与重复读数（P5 调查）** | W03 g01r01 full n=1319 的32-vs-2 mismatch；g01r02重复150分歧；c01r07同100题但并发/显存一起变 | 重复波动成立；并发唯一因果未证实，且与 C 重叠 | **排队调查，未构建或登记 screen** | 先核对 matched n/weights/decode/memory，再判断独立单项 pitfall 是否值得4-cell screen；不强制所有卡两遍全量 |
| 25 | **开发评测通过但官方 full scorer 失败** | p00r16 /90490九次数字 scorer 异常，无 metrics.json | scientist正常完成、judge-clean不等于validator-complete；n=500的0.712不是官方分数 | **观察 / harness failure evidence，非 protocol 候选** | 不盲目第十次重跑；recovery需新冻结合同，不能静默改评分器或填分 |
| 26 | **GPU smoke 与锁卡覆盖范围冲突（J）** | strict g01s01/90791、g01s07/90797 的原始smoke训练早于exp-02创建/锁定 | card-matched计数漏掉probes；模板“非实验”标签与训练/评估前锁卡要求冲突 | **J单项spec已写，未构建/登记** | 仅rule1与smoke_runs注释，明确匹配锁/CPU准备/已声明评估边界；4-cell完整launch审计，固定score floor0.673721；不叠#27 |
| 27 | **卡内head-to-head的future comparator依赖** | strict g01s02/03/06/07/08；override与pre-lock eval两类应对 | missing comparator是本卡将产生的输出；check造成冲突，但已有reasoned override，不能说违规不可避免 | **独立设计优先；未构建/登记** | prelaunch计划与postrun实测验证分离；不放过真实n/protocol mismatch，不和#26或H叠成一个候选 |
| 28 | **训练prompt与grader few-shot分布不匹配** | g01s08灾难性首轮与g01s01小幅残差；g01s04渲染差异 | 值得调查，但一个灾难例不足以支持统一prefix比例；已存在template相关指导 | **观察，暂不建screen** | 先区分渲染可达性、训练分布与配方效果；不以≥20%prefix作为规程正确性判据 |

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
| 09-03 08:12 | **Window 02 不改 protocol；A/B/C/D/E/H 保持冻结，C 加 paired/repeated screen observable，G/I 排队** | 立即改写 C；把 G/I 加第三波；按 recipe 差直接写数据/RL 规则 | 8-cell synthesis 无全新方向；现有 C 已覆盖 sub-SE paired 条件，先用 screen 检验。Recipe 差不是 protocol surface；held pool 52，无需扩波 | Window 02 synthesis、Round 00 Analysis window 02 decision |
| 09-03 08:17 | **Fable 平行 window 02 review 落盘**（`2026-09-03-trace-review-round00-window02-fable.md` + 8 份 report）；同意本窗口不改 protocol；新增排队方向 #20–#23；向 planner 提出 A v2 指标饱和（baseline 5/5 greedy，改读『首个 post-SFT eval 到有测量依据的解码选择』的小时数）、E 在 baseline 已达标（5/5 idle <0.15 h）无法区分、D 按小时/cell 优先级应高于 A | 直接改 A/E 的 manifest；把 P1–P4 立即建 screen | 冻结的 screen 由 planner 定；held 池 52 已满，先把证据与指标问题写清 | 本行；PR #20 评论 |
| 09-03 08:45 | **planner 接受 Fable 的 screen-design 更正，不改冻结 protocol tree**：A 改读首次 post-SFT eval→measured decode choice ≤0.5 h；第一可释放波改为 D/B/C；E 等 guard，≥7/8 已达标则启动前整块撤回并由 G/P1 中一个单项候选替代 | 保持 A 的饱和 greedy 指标；立即取消 E；重写 A/B/C/D/E/H protocol | A 的原指标和 E 的 target 在新 baseline 饱和；D 有 4/8、3.3 h 的可避免损失。guard 是 E 的同文本预注册样本，先读完比现在按窗口挑选更稳健；held/release gate 均不受影响 | round02 spec §七；Fable window 02 synthesis |
| 09-03 09:17 | **关闭方向 #12：c00r06 与 batch1 g12 的 0.7832 不是配方复现** | 把同分当作同一 recipe/trajectory 的复现 | operator 比较两个集群结果包：一个是多阶段 SFT/RFT soup，一个是 SFT+620-step GRPO；只有 scalar 1033/1319 相同。official per-item log 未保留，逐题同一性不可验证 | c00r06 `REPORT.md`；batch1 g12 job 87250 `RESULTS.md` |
| 09-03 11:30 | **meta loop 改为小时级批次监控；每窗口深入 trace review；允许整块撤回失去科学价值的 pending；可复用知识写回 meta skill** | 继续 5 分钟 planner 轮询；保留所有 held 以维持表面 backlog；只把结论留在 Claude/对话 | 用户明确要求监控不必过密、可一次攒够一批；分析后可删减不必要实验；必要时派 subagent 仔细读 trace；知识必须沉淀。receipt-only、PENDING-only、整块撤回与 held floor=8 保留安全边界 | iteration-basis §七 7–9；`skills/exp_protocol_meta/`；local Claude contract |
| 09-03 18:50 | **记录运行中发现的 meta 操作知识：Claude prompt 用 `--` 分隔、在同一权限环境核验 session；held floor 读 Slurm 实际 reason；judge-clean 与开发评测通过不替代官方 validator-complete** | 根据受限 shell 的 socket/PID 观察重启 helper；相信 receipt 的 held 字段；用 p00r16 的 n=500 分数填补官方缺失 | 同一 session 在 host 查询为 working；外部 release 使 28 个 receipt-held jobs 变成 Priority；p00r16 九次全量 scorer 异常。均为可复现的操作/证据边界，依据用户 11:30 的知识沉淀指令写入 meta，不修改冻结 protocol tree | window03 local launch record；held-restore-20260903T184247Z.json；p00r16 scorer-failure review |
| 09-03 Window03 | **关闭本地8-new-clean窗口，不改冻结protocol tree、不cancel、不promote；D/B/C第一波，A/H后续，E条件held，P5仅调查** | 照抄synthesis改E/P1；因零overrides判H饱和；P2/P3叠成rider；用混合3个guard判strict8 | 主审读完整synthesis、两臂最好/最差、三张卡及P1专项日志；E已有PID，P1原始日志收窄反例；C/H/E指标在未开跑前澄清。原始helper输出保留，接受/拒绝理由单列 | window03-local/planner-decision.md；P1 audit；round02 spec §八；meta metrics |
| 09-03 20:00 | **strict cohort 8/8 validator-clean；用g01s03/g01s08明确失败界限保留E，准备单项E v2，不走G/P1替代** | 等最后3份报告才判断已无法反转的条件；按原“可能饱和”撤E | 两个单次保守下界已>0.15h，最多6/8可pass；这不替代完整strict safety review。90820越界导致真实OWNERSHIP FAIL，新Slurm提交/放行停止；33 held保留 | E2 process-wait spec；strict addendum launch；placement-violation-90820 audit |
| 09-03 E v2 construction | **冻结 `c6f11d8` / `ceb68549` 及新4-cell manifest，恢复guard基线；不登记、不release、不cancel** | 继续运行旧E的stale-tail判死文本；把生命周期修订叠到其他候选 | 独立forward review发现跨shell退出结果与旧artifact归属缺口，三处同步澄清；34测试与AST/YAML/六路径差异验证通过。代码准备与Slurm所有权门分离 | round-02-e2-prelaunch；E2 spec；e-wait-on-process-x4-v2 manifest |
| 09-03 strict launch-scope audit | **保留matched training指标但撤销其“所有命令合规”的外推；登记#26与meta统计盲区** | 忽略smoke；把card后补记录当预先锁定；直接改已冻结runtime | 两个raw trace确认真实GPU训练早于训练卡；helper按script/output匹配首条主训练，未计smoke/eval。既不豁免用户硬约束，也不把新合规问题混成session-end损失或改PTB成绩 | strict addendum/launch-scope-audit.md；meta metrics |
| 09-03 strict cohort close | **guard observed-no-harm门通过但不晋升；撤C v2未启动整块；D/B/H首波；P1 v1先校准；#26/#27独立设计** | 原样保留C，只因有pending；照收P1“两次确认/2.60h”；把所有规程违规混成guard伤害 | C baseline7/8已超过3/4门；P1 literal predicate/日志可见性不支持强结论；33held撤4后仍≥8。所有权17/16及原生11节点仍禁止新提交/放行 | 2026-09-03-round-01-strict-guard；strict planner-decision及P1 audit |

## 三、这份台账之外还没写下来的

- Round 01 与 Round 02 的结果记录（`<date>-round-01.md`、`-round-02.md`）在各自的 cell 落地后另起。
- `exp_protocol_meta/iteration_record.template.md` 目前没有"方向台账"一节；三轮后的 meta 回顾时加上，并把 `metrics.md` 补上 `hours_used`、"greedy 是否交付"、"最大评估 n"三项。
