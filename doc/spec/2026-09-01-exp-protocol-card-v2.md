# Experiment card v2:相对六节版的增删与理由

**日期**:2026-09-01 **状态**:生效 **配套**:`doc/reference/exp_protocol_and_wma_policy.md`(定义)、`skills/exp_protocol/card.template.yaml`(模板)、`skills/exp_protocol/example-card.yaml`(示例)

## 一、基底

v2 以 2030 张历史重建卡所用的六节格式(problem / hypothesis / setup / evaluation / result / conclusion)为基底,目录布局不变:`{dir}/memory/cards/exp-NN.yaml`、`{dir}/memory/index.md`。历史语料的 v1 卡加一个空的 `situation` 节、把 `schema_version` 改为 `awm-experiment-card-v2`,即可被 v2 工具读取。

## 二、新增

| 位置 | 字段 | 为什么 |
|---|---|---|
| 第 0 节 `situation` | `elapsed_h` `remaining_h` `incumbent` `trigger` `trigger_evidence` `alternatives_rejected` `pitfalls_hit` `smoke_runs` | 决策发生时的处境。历史语料里"状态类"字段覆盖率 2%–20%,"动作和结论类"100%:轨迹能考古出做了什么,考古不出当时处于什么处境。v2 给处境一个必须先写的位置。 |
| `setup.method` | `stop_token` `answer_marker` | 让 pre-flight 能对训练数据机械检查 eos 一致性与答案格式唯一性——这两个坑各自让一整个 run 干净地跑出错误答案且 exit 0。 |
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

- `exp-NN.lock.json`:`lock` 时写,内含第 0–4 节的哈希与 `setup.command.script` 的文件哈希。只锁 argv 会留下"锁定后改脚本"的缝,锁文件把脚本内容也钉住。`close` 重验,不符只报告不修复——实质性改动是新的一张卡。
- `exp-NN.preflight.json`:pre-flight 的完整报告。`lock` 只在零 FAIL 时写锁。

## 五、有意不做的

不做状态机(没有 draft/frozen/running 之类的状态字段),不做会等待训练的命令,不做需要 LLM 才能执行的规则。规程的全部机制在 CPU 上端到端测试(`tests/test_exp_protocol_*.py`)。
