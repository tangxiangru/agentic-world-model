# 新过程知识 skill：GPQA 风格与代码任务独立 forward 检查

结论：在受测的两个非 GSM 合成任务上，当前 root 指引和既有 CLI 可以完成数据准备、卡/check/lock、CPU 输入检查与诚实关闭。
未发现强迫数学格式、伪造模型成绩或必须先评估才能 lock 的新问题。旧 raw-stop 限制真实出现；新指引给出的证据化 override 路径可用。

## 独立性与范围

- 原始目录：`/tmp/process-cross-task-forward.qLgkKz/`，下称 ROOT。
- 按 skill-creator 的独立 forward-testing 要求，从公开 scientist SKILL/process_checks、卡模板、pitfalls 与 CLI help 使用。
  自选 fixture；没有阅读旧测试或实现来挑选预期答案，也未读真实 GPQA/HumanEval/GSM8K test 项。
- 无 repo 源码、用户草稿、git/index、Slurm、网络改动；全部材料在新的自有 ROOT。
- 使用 root `.venv/bin/awm`。help 只有 new/check/preflight/lock/close/index/chain/collect/install，没有 E 的 run 或新 API。
  未导入或调用任何 E helper。实际命令在 lock 后手工按卡执行。
- 仅用既有离线固定 runtime 和本地 Gemma tokenizer 做 tokenization；无模型构造、forward、训练、评估或生成。
  代码目标只做 AST parsing，未执行函数或任何隐藏/真实测试。
- 这两张卡明确定义为 `family: other` 的 CPU 输入检查，不冒充 SFT 训练或基准评估。
  已知 tokenizer 来源作为真实 parent 元数据路径；实验数据为真实自编 JSONL，未创建虚假输出或数据占位符。

## 版本身份

受测 root：`/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator`。
SKILL SHA256：`86af8cc6d9c35858d01d551d368a4def6bbfb37400d3b11618f8e7862d9cccc3`。
process_checks SHA256：`67a570378d170a6448830f38f40751af7dcd94298dbe48a1dd28229b7b37e81a`。
受测前后哈希相同；两份逐字副本在 ROOT/guidance/，模板/pitfalls/CLI/schema/preflight 哈希在 source-hashes.txt。

## 实际案例与可复核结果

| 项目 | GPQA 风格合成选择题 | 代码任务 |
|---|---|---|
| 数据 | fictional assay/circuit，单字母 A/C 目标 | 3 个自编 Python 函数题，code-only 目标 |
| raw 行 / unique prompt / unique pair | 3 / 2 / 2，重复明确保留 | 3 / 3 / 3 |
| 实际全行 token 长度 | 81、78、81 | 54、53、66 |
| 非 mask 监督 token 数 | 3、3、3 | 25、18、31 |
| BOS 与监督 EOT | 每行各一次 | 每行各一次 |
| raw completion EOT | 3/3 有 | 0/3 有，由 renderer 正确追加 |
| marker | null；仅 WARN，不妨碍 lock | null；仅 WARN，不妨碍 reasoned lock |
| comparator | value/path 均 null，SKIP | value/path 均 null，SKIP |
| 结果 | 直接 check/lock，CPU 检查，close | 先真实 FAIL，证据 override 后 lock，CPU 检查，close |

1. 数据、脚本由 apply_patch 创建。初始 CPU tokenization 属于指引允许的准备，不是未锁定模型实验。
   全行 arrays、labels、保留结果、source/script SHA 在 gpqa-input-audit.json 与 code-input-audit.json。
2. GPQA 卡 exp-01 于 11:59:14 UTC lock；preflight 是 7 PASS、1 WARN、1 SKIP。
   没有补数值 `ANSWER:`，没有给 unknown comparator 填假路径或假得分。
3. 代码中一个合法函数返回字面字符串 `ANSWER: `。由于 task 不需要答案标记，answer_marker 保持 null，未被迫删掉合法代码。
4. exp-02 首次 lock 返回 1：`stop_token_consistent` 的 raw 检查为 0/3。没有启动后续命令绕过失败。
   原失败保留在 code-first-preflight.json 和 transcript.jsonl。
5. 依指引先确认有效序列：脚本用 actual local tokenizer 检查全部三条，labels 中 EOT 一次，且监督数非零。
   使用包含文件/hash/覆盖范围的具体理由 override；12:03:13 UTC 成功 lock。
   lock 文件保留原 `fail: 1` 和 override 原因，没有把 raw FAIL 改称 PASS；raw 数据 hash 始终未变。
6. 锁定后执行各卡声明的 CPU 命令，捕获真实 returncode=0。没有调用模型或测试函数。
7. 结果用了真实非 accuracy 指标 `inspected_source_rows=3`、`syntax_parsed_source_rows=3`；close 均成功。
   `supported` 仅支持卡中明确的 CPU 输入不变量；summary 明示不是 GPQA 分数、代码 pass@1 或 serving 验证。
   没有 checkpoint 产物，输出 checkpoint 保持 null；没有虚构 loss、模型更新或比较值。

## 具体观察及无需修改的真实边界

- 非数学任务可执行：答案标记是可选字段，指标名没有被 schema 强制成 accuracy，代码体没有被数学格式侵入。
  marker 的 advisory WARN 仍会出现，但该运行中没有造成阻塞或误导性改写；不据此建议弱化检查。
- raw-stop False Alarm 没有被“skill-only”自动修好。它仍需科学家提供实际 renderer/tokenizer 证据并记录 override。
  本次新指引确实避免了为迎合 raw 检查而重复追加终止符；没有使用 E 的 rendered bundle 代替 legacy CLI。
- null comparator 可以锁定，不强迫预先跑 evaluator 或伪造结果。SKIP 不是有效比较证明；本次未作任何模型成绩比较。
  相同 n、请求 limit、文件存在、CPU 输入检查都不能证明后续 evaluator 的同题/同设置。
- 当前 schema 的 nonempty setup.data/positive n 限制仍在，guide 已明示。本次两项操作都有真正输入行，因此不需要假数据。
  这不证明所有没有可表示输入的 metadata-only 操作都能建卡；该边界不应通过伪造行数“解决”。
- 输入审计是科研人员自己执行的静态脚本，不是 CLI 自动检查的 whole-runtime proof；wrapper/import/实际模型消费仍不被旧 lock 全覆盖。
- 自编科学题不声称达到 GPQA Main 的难度或分布；自编代码不声称为 HumanEval，AST parse 不证明功能正确。
  自选格式不证明冻结 benchmark 的真实模板、答案提取器、stop behavior、实际 scored IDs 或 token cap。
- 模型 dtype/optimizer/effective decoder、失效/停止进程追踪、checkpoint export 与跨模型 paired performance 未运行，均不填推测值。
  它们是未来合法锁定模型实验需要产生的证据，而不是本次 forward 检查漏填的强制数值。

## 原始入口与复现

ROOT 包含 data/、inspect_inputs.py、两份 native-tokenizer audit、两张已关闭卡及 lock、原失败 preflight、执行 transcript 和 bootstrap 说明。
CLI 本身不含 E 的 run；capture.py 是本 reviewer 的 subprocess stdout/exit 记录器，不是新增 runtime API。
离线重跑 CPU 输入检查（不构造模型，覆盖更新自有 audit 文件）：

```bash
bash /tmp/process-cross-task-forward.qLgkKz/offline_tokenizer.sh /tmp/process-cross-task-forward.qLgkKz/inspect_inputs.py gpqa
bash /tmp/process-cross-task-forward.qLgkKz/offline_tokenizer.sh /tmp/process-cross-task-forward.qLgkKz/inspect_inputs.py code
```

本次未修改 skill。受测 CPU 路径未出现新增的跨任务格式或虚假字段障碍；矩阵决策仍由主审负责，不能从此检查预判模型收益。
