# Experiment card v2:相对六节版的增删与理由

**日期**:2026-09-01 **状态**:生效 **配套**:`doc/reference/exp_protocol_and_wma_policy.md`(定义)、`skills/exp_protocol/card.template.yaml`(模板)、`skills/exp_protocol/example-card.yaml`(示例)

## 一、基底

v2 以 2030 张历史重建卡所用的六节格式(problem / hypothesis / setup / evaluation / result / conclusion)为基底,目录布局不变:`{dir}/memory/cards/exp-NN.yaml`、`{dir}/memory/index.md`。历史语料的 v1 卡经 `awm.exp_protocol.schema.migrate_v1` 处理后(把顶层 `elapsed_h` 挪进新建的 `situation` 节、改 `schema_version`)可以被 `load`、`index`、`chain` 读取;但 `check` 仍会报出 v2 新增而 v1 没有的必填字段——`situation.trigger` 与 `setup.checkpoints.keep`,训练类 family 还有 `setup.method.stop_token` 与 `hyperparams.max_seq_len`——它们只能由人补。`tests/test_exp_protocol_schema.py::TestV1Compat` 把这句话钉成测试。

## 二、新增

| 位置 | 字段 | 为什么 |
|---|---|---|
| 第 0 节 `situation` | `elapsed_h`(自 v1 顶层挪入)`remaining_h` `incumbent` `trigger` `trigger_evidence` `alternatives_rejected` `pitfalls_hit` `smoke_runs` | 决策发生时的处境。历史语料里"状态类"字段覆盖率 2%–20%,"动作和结论类"100%:轨迹能考古出做了什么,考古不出当时处于什么处境。v2 给处境一个必须先写的位置。 |
| `setup.method` | `stop_token` `answer_marker` | 让 pre-flight 能对训练数据机械检查 eos 一致性与答案格式唯一性——这两个坑各自让一整个 run 干净地跑出错误答案且 exit 0。**训练类 family(sft/rft/dpo/grpo/distill)必须声明 `stop_token` 与 `hyperparams.max_seq_len`,否则 `check` 报错、`lock` 不过**;`answer_marker` 保持建议性,因为不是每个任务都有答案标记。 |
| `setup` | `checkpoints: {every_steps, keep}` | 对后续卡的承诺:保存哪些 checkpoint。没有它,"从历史起始点续跑"永远积累不起来。 |
| `result` | `checkpoints_kept` | 承诺的兑现记录;谱系工具用它判断一个起始点是 checkpoint 级还是配方级。 |

## 三、删除

| 字段 | 为什么 |
|---|---|
| `setup.resume_argv` `setup.progress` | 只服务于"训练中让渡 GPU 评估"的机制;规程线不含挂起训练的协议。 |
| `evaluation.diagnostic.items` | 同上,是给运行时的评估器用的;`diagnostic` 保留为对机制的描述。 |
| `result.pings_acted_on` | ping 协议不存在了。 |
| `evaluation.comparator.value` 的"由运行时在冻结时测量" | 改为:scientist 自己在同一协议下测量,给了 `value` 就必须给 `path`。 |

## 四、随卡片一起的两个文件

- `exp-NN.lock.json`:`lock` 时写,内含第 0–4 节的哈希、`setup.command.script` 的文件哈希、以及每个 `setup.data[].path` 的文件哈希。只锁 argv 会留下"锁定后改脚本/改数据"的缝,锁文件把两者的内容都钉住。`close` 重验,不符只报告不修复——实质性改动是新的一张卡。第二次 `lock` 必须带 `--relock <理由>`,上一次的哈希、时间和理由保留在锁文件的 `relocked_from` 里;误判的检查可用 `--override <检查>=<理由>` 放行,同样记录在案。

  **锁的范围与威胁模型。** 锁文件和卡片在 scientist 自己可写的目录里,所以它是**痕迹,不是屏障**:防的是无痕修改,不是对抗。事后可检测的是:0–4 节改了、脚本改了、数据改了、锁被删了(`close` 直接失败)、锁被重写(必须有理由且旧哈希保留)。**不覆盖**的:argv 里除 `setup.command.script` 之外的文件(例如 `--config configs/exp03.yaml` 里的超参)、`card_id` / `created_at`。真正对抗性的冻结(推到 scientist 不可写的外部)属于 rollout 接线,不在规程线内。
- `exp-NN.preflight.json`:pre-flight 的完整报告。`lock` 只在零 FAIL 时写锁。

## 五、有意不做的

不做状态机(没有 draft/frozen/running 之类的状态字段),不做会等待训练的命令,不做需要 LLM 才能执行的规则。规程的全部机制在 CPU 上端到端测试(`tests/test_exp_protocol_*.py`)。
