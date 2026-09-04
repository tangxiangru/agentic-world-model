# E8 独立 forward-test 报告

结论：完成自有 native HF artifact 的保真发布及一次可恢复 incumbent 更新。
当前 final 是选定 B；旧 A 的独立 backup 与原冻结身份一致。没有发现受测支持路径的功能性误拦截。
首次 guide-only 替换因入参类型理解错误失败，原尝试未覆盖；完整 manifest 修正尝试单独保留。

## 范围、输入和冻结版本

- 候选仓库：`/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo`。
- 全部原始材料、脚本、输出、失败及报告根目录：`/tmp/e8-serving-forward.lQggAi/`，下称 ROOT。
- 从公开 serving-artifacts、scientist SKILL、save-safety、execution-records 指引执行。
  未阅读 helper 实现或 builder tests；只额外检查公开函数 signature（无类型注解），未据此抄取答案。
- 无候选源码/guide/tests/git/index 改动，无已有模型目录改写，无网络/下载/安装/GPU/Slurm。
  仅在锁定卡后构造两份微型随机 GPT2 并执行 native CPU save；无 forward、训练、评估或推理。
- `offline.sh` 复现固定 bwrap Python：宿主只读，仅 ROOT 可写，隔离网络、全新 /dev，HF offline。
  Torch/Transformers 使用已有 `/tmp/exp-protocol-save-runtime.JEZlHo`；未载入任何真实基准权重。
- 完整身份在 `source-hashes.txt`，guide 原样副本在 `serving-artifacts-as-reviewed.md`。
  helper SHA256：`13e5befeb8177a3e0dd1e0289903677388392b4799651b72d38c014e3168174f`。
  guide SHA256：`4bebb5b0673449a34363f9aa800b622cfc9ca67750a01610293a5b148933dd35`。
  二者审阅前后未变。scientist SKILL 路由新增由主审告知；最终版本已完整重读，未提供预期测试答案。

## 实际工作流和输出

1. 用 `awm exp_protocol new/check/lock` 建立 exp-01，声明 CPU serialization 命令。
   原始配置和已选 generation JSON 在 setup.data 中绑定；没有伪造模型评分或 comparator。
   11:13:06 UTC lock 成功，随后 E5 run 执行 build.py；真实 child returncode 为 0。
2. `build.py` 自建 WordLevel fast tokenizer，词表 16；GPT2 为 1 层、2 heads、embedding 16。
   两份随机初始化 seed 31/47，各产生三个 native safetensors shard 和完整 index。
   所选配置 A 为 greedy，B 为 sampled；均在保存前写入独立 selected-*.json，并计算冻结字节 hash。
3. 使用 E4 `check_before_compute` 和实际 native saving scope，导出选定 serving JSON。
   A 的 selected_serving_hash 与 serializer-only hash 不同，验证没有误用后者。
   `build-record.json` 和 artifacts/.exp-protocol-save-events 保留完整 save 证据。
4. `freeze.py` snapshot 两个 source，将 manifest 写到 memory/selections，执行默认 native metadata verify。
   两者都是 native-hf-text-v1 / GPT2LMHeadModel / indexed safetensors，均加载本地 config/tokenizer/generation。
   每份 manifest 含九个 serving 文件；`trainer_state.json` 明确列为 ignored，不复制到 final。
5. 11:14:38–11:14:47 UTC：`publish.py` 将 A 发布到原本不存在的 ROOT/final_model。
   journal：`memory/serving-publications/35bde47403a24106b628d1a181365dcc.json`，status=published。
   新发布使用单次 no-replace rename，选定 A generation 字节与 source 原样一致。
6. 同一脚本进行已有目标/替换前置条件案例；错误类型见下。尝试替换的 old_identity 被传成 SHA 字符串，失败。
   第一轮脚本最终退出 1；`publication-observations.json` 和 transcript 保留原始调用及异常。
7. `replace_retry.py` 使用此前已冻结的完整 incumbent manifest，而非重新接受当前任意字节。
   合法但错误的 B incumbent manifest 被拒绝；确认 final 仍为 A 后才正式更新。
   quiescence-evidence.json 记录自有随机目录、未启动任何模型/evaluator consumer、metadata 调用均已同步返回。
   条件是调用者在此自有 fixture 范围建立的，不是 helper 独立推导的全系统进程证明。
8. 11:19:26 附近，选定 B 替换成功，journal：
   `memory/serving-publications/711799eee2064a67bed086cc9beb949c.json`。
   最终：`/tmp/e8-serving-forward.lQggAi/final_model/`。
   旧 A：`/tmp/e8-serving-forward.lQggAi/.final_model.backup-711799eee2064a67bed086cc9beb949c/`。
   返回记录明确是 recoverable_two_rename_sequence_not_atomic_exchange；rollback=not_needed。
9. 对 final B 与 backup A 重新执行 native metadata/完整冻结身份校验；结果见 final-verification.json。
   card exp-01 已诚实关闭，科学 verdict=inconclusive；未把 artifact 发布当作模型质量试验。

## 最终身份

| 对象 | 冻结 manifest content identity |
|---|---|
| 原 A、曾发布 A、保留 backup A | `ee329c5d99d64b26b4a44ea36bf1b6b75d983eaf0e1ee1f67e5b984811c6940d` |
| 原 B、当前 final B | `4ef942cabde3515c68eda6dcba79cec4f5b1714644a48523160e20c3a0bd1740` |

A selected generation SHA256：`f821d68b03fa34c8f764e99eb34bca3be4d0122dfc72121f8aadae9dbdbfb286`。
B selected generation SHA256：`1533ab9763390b4fdfea521a98657139dcb82a6a561e7791891bb4f7cf4e9c4d`。
Shell SHA256 与 helper 的 final/backup 校验都匹配。source A/B、final、backup 和失败材料均未清理。

## 独立失败及不支持案例

- 已存在 final，未指定 replace：ServingPublicationError，有 journal，旧 A 未动。
- 未建立/未声明 quiescence：ServingArtifactError；没有 record_path，调用者 transcript 保留失败。
- expected_old_identity 传 SHA 字符串：ServingArtifactError，提示 frozen serving identity malformed；无 journal。
- 正确类型但选错 incumbent manifest（B，而实际 A）：ServingPublicationError，有 journal，old_incumbent_untouched。
- 用 B 的 selected-generation hash 验证 A：拒绝，不把当前文件 hash 自动当作所选 hash。
- 拷贝 A 并将 temperature 改成 1.0 后仍声明原 A：发布拒绝，生成失败 journal；未产生目标目录。
- 缺少 index 所列第二个 shard：snapshot 拒绝；该 shard 移到 boundary-fixtures/held-missing-shard.safetensors 保留，未删除。
- 加入 chat_templates/custom.jinja 子目录：UnsupportedServingArtifact，明确 unknown nested serving layout。
- 任意 opaque .bin 字节、未 opt-in：UnsupportedServingArtifact，要求 allow_opaque_weights。

完整结果分别在 publication-observations.json、replacement-observations.json、boundary-observations.json。
各原始 fixture 均保留于 boundary-fixtures；只改变了新建的 reviewer-owned 副本。

## 工具实际证明的层级及已实证限制

- 它证明当前选定文件身份、所选 generation 字节、native shard/header/index 结构、支持的本地 metadata 加载，
  以及此次发布的 stage/source/目标身份与保留 backup；未证明任何此前 evaluator 读到的身份。
- 特意将一份副本 config.n_embd 改为 32，而 native 保存权重仍为 16；新冻结后 metadata verify 成功。
  这准确实证 guide 所说“不证明 config 与参数形状对齐”，不是假称该模型可加载。
- 显式 allow_opaque_weights=True 后，故意不是 pickle 的普通字节也能通过 byte-only/metadata 验证，
  weight_structure 明确为 opaque_bytes_only_no_pickle_load。没有执行 pickle，也没有证明它是有效权重。
- A 的 inactive sampling 字段会产生 Transformers warning，但不被 helper 重写；保真不是引擎解码语义保证。
- 所有记录保持 model_load/inference_engine_validation/scientific_validation=not_performed。
  不证明 vLLM loadability、tensor 值正确、contamination、模型质量、选择优越性或训练完备性。
- 未独立触发中途 copy/rename 故障、并发 namespace 冲突、SIGKILL、unsupported filesystem、Gemma processor 路径。
  因此只观察到正常 recoverable replacement 与 precondition 拒绝，不声称已验证全部 rollback/崩溃恢复路径。

## 剩余指引精度意见（未修改候选）

1. 明确 expected_old_identity 接收完整 snapshot manifest，并给出赋值例，而不是只命名 frozen_incumbent_identity。
   第一轮将 identity_sha256 字符串传入是可理解但错误的用户尝试；纠正后成功，不属于 helper 功能误拦截。
2. 区分 ServingPublicationError 的 durable journal 与 ServingArtifactError 的早期参数/身份失败。
   “Each operation records”及只提 publication exception 的文字容易让调用者误以为每次失败都有 record_path；
   实测部分前置失败只有 report。应提示调用者另外保存这些异常，不能依据缺少 journal 推断没有尝试。

## 复现及原始证据入口

只读重验当前 final 和 backup：

```bash
bash /tmp/e8-serving-forward.lQggAi/offline.sh /tmp/e8-serving-forward.lQggAi/verify_current.py
```

`transcript.jsonl` 保留真实 argv、UTC 时间、returncode、耗时和完整输出；不要盲目重跑 build/publish 去覆盖旧实验。
成功/失败顺序分别由 build.py、freeze.py、原 publish.py、修正后的 replace_retry.py、boundaries.py 直接复现。
单次新发布和替换含 native metadata 冷启动约 9 秒，微型 fixture 的局部 CPU 成本不外推到真实模型规模。
