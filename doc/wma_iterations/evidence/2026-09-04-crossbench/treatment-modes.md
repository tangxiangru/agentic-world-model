# R/P/S/M/J treatment modes：最小实现接口审查

2026-09-04，只读架构建议。已读 `awm/sandbox.py`、`awm/wma_client.py`、`awm/exp_protocol/{decisions,run,lock,cli,install}.py`、现有相关测试和最新implementation记录。记录所述完成的是联合比较/新证据规则/版本审查的默认流程；还没有表达本次五臂消融的显式treatment mode。本文不改代码、repo或Slurm。

## 一、只增加一个模式维度，不增加跳过单卡review的实验开关

建议 `awm sandbox setup --exp-protocol --decision-mode MODE`，MODE为：

- `single`：不要求formal candidate pool；scientist按单方案写card。训练仍必须preflight、lock、当前proceed、run；attached侧车仍照常单卡阻塞review。
- `multi-self`：通常2–3个真实brief，scientist先记preference，再自行选；不调用联合WMA。所选card照常单卡阻塞review。
- `multi-joint`：同样brief/preference，一次共同上下文WMA比较，再由scientist选择；所选card照常单卡阻塞review。

| 臂 | sandbox/规程 | decision_mode | 实际侧车 |
|---|---|---|---|
| R | 标准PTB，不装AWM protocol | 不适用 | 不附加 |
| P | exp_protocol | single | 不附加 |
| S | exp_protocol | single | 附加新证据规则单卡WMA |
| M | exp_protocol | multi-self | 与S相同 |
| J | exp_protocol | multi-joint | 与S相同 |

**不需要 `single_card_review=off` 这样的新运行时开关。** P的无WMA由真实manifest不附加sidecar表达；S/M/J的formal lock行为全相同。`--decision-mode`只控制选择阶段，不控制lock是否等裁决。

建议默认保留当前`multi-joint`行为以免破坏已实现接口及测试。新实验manifest必须显式写模式，不靠默认值做处理定义。无protocol时提供mode应拒绝；R继续标准PTB入口而非安装空AWM skill来冒充raw。

## 二、唯一配置来源与冻结

在既有`awm_sandbox.json`增加`decision_mode`和确定性的`treatment_sha256`，例如digest只覆盖 `{protocol_sha, exp_protocol, decision_mode}`（若stop_hook是处理的一部分也纳入），**不包含时间戳或written顺序**。复用现有`AWM_SANDBOX_SETUP`参数传递，不需要新环境变量或每命令可覆写mode。

最好新建小型公共模块 `awm/exp_protocol/treatment.py`，提供：

- `load(session) -> Treatment`：读取并校验枚举、配置结构和digest；不接受来自card/brief的mode覆盖。
- `decision_mode(session)` / `treatment_fingerprint(session)`：client与run共用，避免客户端允许而run拒绝。
- 未setup的开发目录或旧schema无mode：明确返回`multi-joint`且标`legacy_default`；**坏JSON/未知mode不能回退single**。新receipt门要求显式mode，不能把legacy fallback当新实验有效配置。

写入策略：同一target重复setup相同有效treatment可幂等；已出现lock/requests/actions/launch后不同treatment应拒绝重设。新cell通过新目录启动。不要以重新setup覆盖配置的方式把M切成S继续跑。

`awm_sandbox.json`在scientist可写工作区，因此“chmod只读”不是完整安全边界：本轮最低要求是manifest/receipt冻结配置及启动时核对digest、发现变化即失败并记为处理违规。若需要预防scientist在首条记录前改mode，运行器还应通过既有只读public挂载下的冻结配置或只读绑定文件提供authority；不要声称仅自报digest防止恶意改配置。该问题属于treatment归属，不应通过关闭lock保护解决。

## 三、三种流程及准确状态

### single

Scientist无需调用propose/compare/choose，直接card→lock→act proceed→run。`run.execute`只在**明确single模式**省去`_choice`校验；所有原本在`_choice`之前的plan、files、review、current-action检查原样执行，training和非training都不减少这些要求。choice字段写null，同时launch记录写`decision_mode=single`及`treatment_sha256`。

建议single模式下`propose/compare/choose`给清楚的“本cell为single流程”错误，不创建临时fake pool。Scientist可以在card的alternatives_rejected中如实说明思考过程；single的含义是不要求结构化多候选实验处理，不声称世界上只有一个可行方案。

### multi-self

为最小改动，**保留当前命令序列**：propose→compare→choose，但将compare解释为“冻结本次选择上下文”；multi-self时它只生成本地不可变request/completion，不enqueue给sidecar，不轮询、不调用backend。

建议completion：

```json
{"state":"not_requested","reason":"treatment_multi_self","decision_mode":"multi-self","treatment_sha256":"...","request_id":"...","waited_s":0.0}
```

- 输出“候选已冻结；本处理不请求联合比较，请自行选择”。可以另记实际`sidecar_attached`布尔用于审计。
- **即使侧车确实attached也返回not_requested**；这不影响稍后正式card的单卡WMA。
- 不写`comparison.json`或伪ranking；保留输入brief、preference、最终choice及理由。
- `_compare_command`把not_requested当成功的本地冻结状态（exit 0）。它不是模型review成功。

### multi-joint

沿用当前比较路径。只有确实未附加侧车时才返回`not_attached`；真实失败/timeout保留其原状态。J运行合同预期attached，因此实际not_attached是依从性问题，不能当J的正常“无辅助自选”结果悄悄混入机制成功分母；按已定义不可用策略继续与否、最终评分和失败记录照常保留。

`not_requested`只对multi-self合法。不能在joint模式写一个not_requested completion来跳过已定义处理；`record_choice`和run必须核验模式及配置digest而非只检查“文件存在”。

## 四、需要改的函数（优先最小触点）

1. **`sandbox.setup` / `_setup` / `register`**：接受mode；在任何install变更前验证枚举/exp_protocol依赖/已有treatment一致性；记录有效mode/digest。现有setup sha、stop hook与written记录保留。
2. **`exp_protocol/install.py`及`skills/exp_protocol/SKILL.md`**：让安装后scientist明确读当前mode。最小做法是公用skill含清晰三模式分支、安装生成一份当前mode小说明并在指针块引用；single分支不要求brief，multi-self不声称请求模型，multi-joint保留现有流程。无需复制三套完整skill，但生成文件内容/hash必须进provenance。仅改client而保留现在“所有training必须compare”的无条件skill会让处理串线。
3. **`decisions.create_proposal/load_proposal`**：single拒绝formal pool；multi两个模式使用原1–3真实candidate schema和singleton_reason规则，不改变成本/ID/结果字段验证。模式不能由singleton_reason伪造；multi模式只有确实一个真实候选才用原singleton例外。
4. **`wma_client.compare_and_wait`**：先加载固定mode；single拒绝；multi-self在写冻结request后立即写not_requested completion并返回；multi-joint保留现有调用。request加mode/digest。**不得修改 `sidecar_attached()`、`enqueue()` 或 `review_and_wait()` 来实现禁用joint。**
5. **`wma_client.record_choice`**：保持要求“有matching冻结request和已终止completion”。检查request/completion的request_id、proposal_sha、mode/digest与当前一致；允许multi-self+not_requested，以及multi-joint现有真实状态。保持preference、真实candidate、decline-all不能绑card、bound_plan_sha/bound_inputs与不可变文件。
6. **`run.execute` / `_choice`**：显式single省candidate-choice步骤；multi模式保留current_choice、superseded/declined、proposal_sha、selected、bound_plan/bound_inputs检查，另外核验模式/digest和合法comparison_state。不得把“没找到choice”自动解释为single。
7. **`decisions.card_fingerprint` / `append_action`与lock pinning**：把有效treatment digest纳入审查/动作/launch链。只给launch加mode不够，因为已有M choice/proceed后编辑配置到single会跳过choice。建议lock顶层保存treatment digest（`lock._write_lock`或`exp_protocol.cli._lock`在任何review前统一标注），run比较锁定与当前digest；fingerprint也包含此digest。**P的review_and_wait not_attached分支目前没有review fingerprint，所以不能只把digest放进delivered verdict而漏掉P。** 保留已存在的lock_id防并发、版本哈希及新proceed要求。
8. **传递/验收**：新manifest的`awm.setup`沿既有`AWM_SANDBOX_SETUP`带`--decision-mode`。public ship目录已有`awm/exp_protocol`，新helper随目录自动包含；需要核对manifest实际ship合同，不新增private能力给scientist。receipt/harvest沿用awm_sandbox.json收集，并在operational分析标出mode缺失或实际不匹配。

无需为此次mode改动修改WMA verdict schema、scorer、隔离broker、model prompt中的证据规则或单卡L0–L3语义。Opus4.8 profile是主线程独立输入配置，三个WMA臂必须同model/effort/budget。

## 五、必要的行为测试（有实际风险的最小集合）

### `tests/test_sandbox.py` / 新treatment模块测试

- 参数枚举及必须exp_protocol；三mode记录可复原、CLI传递；相同setup幂等。
- R/no protocol不写实验skill/说明，不给它隐式mode。
- 明确未知mode/坏JSON拒绝；旧无字段仅走已说明legacy默认；不能回退single。
- setup同target改mode（已有记录）拒绝；模式digest不随时间戳变化。
- installed instruction明确当前mode；single没有无条件多候选要求，M不会被命令说明误导发joint。

### `tests/test_wma_comparison_flow.py`

- M **真实attached侧车**，相同真实2–3 brief：compare返回not_requested，backend调用0、sidecar requests enqueue0、等待0、无ranking；仍有冻结request/completion和可选择的真实candidate。
- M后续正式card lock确实调用单卡backend并等待，证明只关闭joint。
- J同输入调用joint backend恰好1次；J无侧车才是not_attached；M无侧车仍是not_requested并可另外如实记录attachment状态。
- single调用propose/compare不产生fake singleton/失败回退工件。
- 修改proposal/preference仍使旧冻结request不能choose；伪造not_requested给J、跨mode重用completion/choice都拒绝。
- J真实timeout/failure/迟到响应原测试保留；mode功能不改这些状态语义。

### `tests/test_exp_protocol_run.py` / `test_wma_record_integrity.py`

- P single/no-sidecar与S single/attached训练在完整lock+current-proceed后可以run，**不需fake choice**。
- 三模式参数化复用关键保护：pending review、过期/异request verdict、修改plan/script/config、same-second relock、无proceed/旧proceed、同proceed二次启动都失败。
- M/J没有choice、choice被新选项/decline取代、formal-card binding变化时不能launch。
- M choice+lock+action之后把配置改single，或删除配置触发legacy默认，均不能复用旧记录启动；P no-sidecar路径同样检测配置变化。
- launch记录mode/digest与其lock/action/choice一致；scientific_completion仍not_assessed。

### 小型matrix集成测试

用stub模型、真实client/sidecar/launch wrapper证明各臂调用计数：R=0/0、P=0/0、S=0/1、M=0/1、J=1/1（joint/formal，在一个主要training决策下）。M/J同样保存候选与preference，S/P不强制pool。该测试比仅测enum或复制实现条件更能防处理混杂。

生产验收继续按implementation记录的限制：静态/本机测试不证明真实Claude容器完整往返。这个mode接口不引入绕过隔离或review的运行策略；真实生产canary由主线程另做。
