# R1历史负面裁决：错误预测还是实际压掉好方案？

范围：仅复核冻结64-cell样本中的两个已知R1负面案例及现有R1审计索引，不把本节结论推广到后来完成的E/F或全部78cells。使用原始result_dir、scientist完整轨迹、card、verdict和lock；未重新扫描所有轨迹。

## w04r04/exp-05：实际停止过一次训练，但随后修复并重跑；不是永久否决RFT

来源：job90679；结果目录
`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r01-wma-b-x8-v3_w04r04_formal_r4/gsm8k_google_gemma-3-4b-pt_90679`。
Receipt：`data/ptb/batches/wma-gsm8k-gemma4b-high-r01-wma-b-x8-v3/formal-2026-09-02T191938.312190+0000.json`。
以下行号在 [原始scientist轨迹](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r01-wma-b-x8-v3_w04r04_formal_r4/gsm8k_google_gemma-3-4b-pt_90679/solve_parsed.txt)。

- **01:42:07，7338–7344：** scientist已launch RFT，随后异步请求review。
- WMA `exp-05.verdict.json` issued_at **01:49:39**：L0=no、L1=no、L3=no。L3原文先说“no as locked”，理由是约step800保存时会因GenerationConfig失败；并明确“fix is2min”“precondition, not reason to abandon line”“sequencing verdict, not veto”。另建议先评existingcheckpoint/soup。
- **02:16:02，7520–7522：** scientist直到695/1598steps时才看到verdict ready；这是晚到/晚读的咨询，没阻止首次launch。
- **02:16:08，7535；02:16:26–29，7553–7579：** scientist读具体保存警告，并亲自执行`GenerationConfig.from_pretrained(parent).save_pretrained('/tmp/gctest')`，实际得到`SAVE FAILS: ValueError`。
- **02:16:51–52，7590–7595：** 明确说“Confirmed a latent crash…Killing and fixing now”，随后停止训练。
- **02:17:06–26，7611–7668：** 改脚本给model设置合法generation config；save_steps800→400；记录relock，立即从同parent、同数据、同lr/epoch重新launch。
- **03:37:28，7965；03:41:29，8025–8029：** 重新训练完成，n150=0.7267，原incumbent0.7067。
- 后续n500比较exp05=0.744、exp04=0.730、soup=0.728；scientist保留exp05并继续以它为parent做exp07，最终exp07=0.758@n500（8555–8557、9030、9341）。

判定：**WMA警告实际引发一次中止、修复和重跑，但没有让这个训练方向被放弃。** 保存失败机制被scientist独立复现；不能把695steps已花的约0.62h全算“WMA错杀”：原command的保存路径确有真实错误，也不能把后续修复版本的成功当作对原始command的简单false-no。可称“迟到的正确技术警告，改变了运行”；没有可识别的“原本可交付好模型被永久压掉”。

其L3附带的“修后也应先选checkpoint/soup”经济判断并未被严格执行：scientist直接重跑RFT，后续RFT优于旧incumbent。它因此是过强no措辞/排序建议值得审查的例子，但不是已兑现的压低分数证据。该verdict因package-source越界有原始access flag，不能当clean校准样本。

## w01r07/exp-02：错误的L0/L1警报；原训练照常完成

来源：job90562；结果目录
`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r01-wma-x8-v1_w01r07_formal_r1/gsm8k_google_gemma-3-4b-pt_90562`。
Receipt：`data/ptb/batches/wma-gsm8k-gemma4b-high-r01-wma-x8-v1/formal-2026-09-02T135535.899465+0000.json`。
以下行号在 [原始scientist轨迹](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r01-wma-x8-v1_w01r07_formal_r1/gsm8k_google_gemma-3-4b-pt_90562/solve_parsed.txt)。

- **14:20:07**原lock；**14:20:10，4429–4434**先launch`run_exp02.sh`，再异步请求review。
- WMA issued_at **14:28:59**：L0=no@0.65、L1=no@0.65，但**L3=yes@0.8**。它在自己的环境里判断Liger缺失会使Trainer启动失败；这不是一个L3“不值得训练”的裁决。
- **14:32:31，4929–5017：** scientist确实打开并打印该verdict。
- 此后没有停止/替换正在进行的exp02。**14:32:49，5028–5094**写的是用于未来中间checkpoint评估的`fix_ckpt.py`；**14:42:31，5174–5178**原训练已运行152/365并产checkpoint150。
- **15:13:22，5594–5598：** accuracy=0.653333；卡记录全部365steps完成、0.94h、最终adopt。lock无relocked_from。
- 后续以其结果继续checkpoint sweep，再扩数据；没有因该负面L0/L1而取消这条路线（5720–5747、6132–6142）。

判定：**这是实际错误预测，但不是实际否决/压掉好实验。** WMA关于训练环境的推断没有预测对原run；它到达时训练已开始且继续进行。不能把这个案例写成“WMA导致丢失+61pp收益”，因为这份收益已经拿到了。也不要将later exp09的安装损坏与此处exp02混为一件事。

## 现有R1审计还显示了什么

`/tmp/wma-deep-analysis/detail.json`的R1终态分布：core22份全L3yes；B-v2 19份全yes；C-v2 4份全yes；B-v3 18yes+1no，唯一no就是w04r04/exp05。core的唯一L0/L1no是w01r07/exp02。

现有`harm-review-summary.json`主要覆盖w06–w10候选/阻塞波，不能冒充旧R1的全量初始review审计。本次没有从现有R1索引再识别到其它原始L3no/defer；**终态文件可能遗漏早期版本/未归档review，因此此处不宣称所有历史初始verdict绝无其它负面回答。**

这两例都不支持“已确认WMA永久否决了本来会成功的方案而降低分数”。它们分别展示：预测错误但没有阻断；以及正确技术警报引发中止修复、同计划随后继续。`failed/killed`执行字段单独不足以判定WMA veto。
