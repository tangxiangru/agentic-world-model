# WMA 什么时候实际有用：两条原始轨迹的归因核验

只读重读原始 scientist solve_parsed、card、lock 和 verdict；没有依赖原先诊断备忘录作事实来源。时间均 UTC。下面的增益只指该次观测，不是随机对照因果效应。

## 来源

- w10r02 / job 90983：receipt `data/ptb/batches/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2/formal-2026-09-03T014918.962581+0000.json`；manifest `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2.yaml`。
- 原始结果目录：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983`。
- w09r01 / job 91024：receipt `data/ptb/batches/wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2/formal-2026-09-03T015112.001284+0000.json`；manifest `experiments/posttrainbench/wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2.yaml`。
- 原始结果目录：`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024`。
- 两个 cell 均由 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-review/complete-cells.json` 的 result_dir 定位。以下行号对应逐字解压的 [w10r02 原始 scientist 轨迹](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt) 与 [w09r01 原始 scientist 轨迹](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt)，未重新生成叙述。

## 案例一：w10r02/exp-04，新增一个更好的局部候选；不是发明 soup，也不是最终涨分

### 先挑战归因：scientist 之前已经会做什么？

1. 10:49:43，上一张 exp-03 的 WMA 已经建议过一个两成分 soup：exp-02 的 ck650+ck1293；这是另一个组合（轨迹 7484–7504）。所以不能称 exp-04 WMA 第一次教会 scientist “两成分 soup”这个想法。
2. 12:40:23，scientist 自己在 exp-03 close 前写下一步是做两个训练 run 的 uniform soup，并用 n=500 仲裁；12:40:30 又公开宣布即将做 soup card（8636–8667）。
3. exp-04 的原始、WMA 审查前卡片已经明确计划 **四成分** ck650、ck1293、ck1350、ck1757，且 n=500、失败保留单 checkpoint 的策略已经存在（8741–8748、8773、8785–8787）。
4. 本次 WMA 前，已实际测量 ck1350 的 n=500 comparator=0.704；这也不能归功于收到 exp-04 verdict 后的提醒。

可以归因的新东西因此很窄：**把“原计划四成分 soup”补充为一个特定的、只取两个现有优胜点的跨-run soup：ck650+ck1350，并实际加入候选比较。** 在审查前轨迹中未找到这个特定两成分组合的计划或执行。

### 建议到执行到结果的完整链条

| 时间 | 原始证据 | 发生了什么 |
|---|---|---|
| 13:09:24 | 9210–9214 | scientist 看到 exp-04 verdict 已返回 |
| 13:09:29 | 9216–9237，尤其 9233 | scientist 主动打印 suggestions；WMA 明确说“also make the 2-ingredient soup of ck650 + ck1350”；原因是另两成分在此前 n=150 读数更低 |
| 13:09:42 | 9246–9294 | 先按原计划生成四成分 soup，补 tokenizer/config，并实际 diff 验证其 generation_config 与 comparator 一致 |
| 13:10:09 | 9296–9334 | 启动四成分评估，同时执行 `python soup.py --inputs ...checkpoint-650 ...checkpoint-1350 --out ...exp-04-soup2` 并 prep；这是建议对应的真实新增动作 |
| 13:24:32 | 9389–9431 | 执行 `bash run_eval.sh ...exp-04-soup2 ... 500` |
| 13:27:27 | 9433–9442 | 读取两成分 accuracy=0.710、stderr≈0.0203 |
| 13:27:33–13:29:46 | 9452–9496 | 补测 ingredient ck650 at n=500，结果 0.688 |
| 13:30:24–13:30:26 | 9506–9515 | scientist 明确宣布把两成分 winner 打包，并实际复制到 final_model |
| close 记录 | 9624–9641 | 四成分=0.694，两成分=0.710，ck650=0.688，ck1350=0.704；scientist 明确把该两成分候选归为 WMA 建议 |

没有发现一句在执行前单独说“我接受两成分建议”的自然语言；但 13:09:29 读建议、40 秒后构建精确对应组合、稍后测量/打包，再在 close 明确注明 WMA 来源，足以支持可观察的建议采纳链。

### 到底贡献了多少？

- 它确实让候选集合增加了一个原计划之外的具体模型，并改变了**当时的** final_model 选择。
- 同一 n=500 下，两成分比原四成分高 **1.6 pp（8/500 items）**，比已测最强 ingredient 高 **0.6 pp（3/500）**。后者很小，不能声称已证明质量优于 ingredient。
- 不能把 +1.6 pp 写成“避免了最终退化”：原卡本来已经规定 soup 不如 ingredient 就退回 ingredient；没有 WMA 时原计划的 fallback 是 0.704，而不是强制提交 0.694。
- 也不能把它写成 cell 最终 PTB 提升。16:08 左右，scientist 用后来独立的 exp-06 SFT（n=500=0.738）替换 soup；16:09:22 对两 shard 做 hash 相等核验，明确 final_model 是 exp-06（10718–10739、10797–10855）。最后表格也明确 shipped=exp-06，soup 只是中途 incumbent（11363–11368、11403–11408）。
- WMA 的建议“所有四个 ingredients 都补测 n=500”只部分执行：这里看到 ck650 加上先前已有的 ck1350，没有 ck1293/ck1757 的匹配 n=500 补测。因此不能说完成了全四成分的最佳 ingredient 证明。

**本例支持的“什么时候有用”：已经有模型和局部对比结果、scientist 正准备合并它们时，WMA 能提出具体低成本子集，补上一个 scientist 当时未列出的候选，并推动真正评测。已观察到局部候选改进；未证明最终任务分数提升。**

## 案例二：D w09r01/exp-04，把表面的“有保存”修成可实际比较的保存与评估

### 先挑战归因：并不是从零增加保存意识

原脚本早在 2600 行就写着 `save_total_limit=2`。原 exp-04 card 已有 `--save-steps 2200`、`every_steps:2200`、`keep:last`（6376–6385），因此不能称“scientist 本来完全不保存 checkpoint”。其预先评估段只指定常规 n=150 和“decode 保持一致”的断言，没有要评哪些中间点的明确执行计划（6387–6401）。

WMA 指出的增量问题是：预计 4469 steps，而保存间隔 2200 与 retention=2 联合作用，会让较早的 2200 被最后 4400/4469 旋掉；看似定期保存，训练结束后实际只剩几乎相邻的尾部点。它又指出直接评分 final/ 会继承 sampled config，以及 card 宣称只变 data、实际 argv 却同时变 lr。

### 行为链

| 时间 | 原始证据 | 发生了什么 |
|---|---|---|
| 18:20:56 | exp-04.lock 的 relocked_from:27；原卡 6376–6398 | 原始方案锁定：lr=1e-5、save_steps=2200、limit=2 |
| 18:28:12–18:28:17 | 6571–6604 | 返回 L3=defer；precondition 1 指出 retention 问题并要求改保存间隔/上限；cheaper_variant 明确要求中间点与 final 同 n=150 比较 |
| 18:28:53 | 6629–6630 | scientist 明确回应“three preconditions that change the run”，决定带修复重锁 |
| 18:28:54–18:28:59 | 6632–6662 | 真正改脚本，新增 `--save-limit`，用 `args.save_limit` 代替 hard-coded 2 |
| 18:29:24 | 6664–6685 | 真正改 card：save_steps=1490、save_limit=4；lr 改回1.5e-5；加入 greedy 产物生成步骤和中间 checkpoint 的可加载准备 |
| 18:29:31 | lock:4、29；轨迹 6698 | 重锁理由明确点名 WMA preconditions 1/3；scientist 同时记录自己不赞同 WMA 放弃这次数据扩量、也不接受半量 pilot |
| 18:36:36–18:36:43 | 6802–6823 | verdict 变 yes；scientist 真正用 `--save-steps 1490 --save-limit 4` launch |
| 21:11:15–21:11:17 | 7452–7460 | final=0.753 后，scientist 说“Now the checkpoint-selection lever”，执行 checkpoint-2980 的 greedy build 和同协议 n=150 eval |
| 21:13:23 | 7489–7500 | checkpoint-2980=0.6933 |
| 21:13:35 | 7509–7510 | scientist 明确选择 end-of-run，随后推进 RFT；card 记录保留1490/2980/4469/final（exp-04.yaml:114–126） |

### 贡献的准确表述

- **强归因：** WMA 让 scientist 修掉保存间隔×retention 的相互作用，落实到源代码、lock、真实 launch 参数；保存出了实际有区分度的较早点，并真的测了其中一个。
- **这次额外比较没有发现更优 checkpoint。** final 比被测中间点高 6.0 pp（9/150），scientist 继续用 final。不能把这6点算成 WMA 提升，因为原方案本来就很可能使用 final；没有发生从坏的中间点改选好 final 的已证实反事实。
- 这次测量从 21:11:17 到 21:13:23 约 **126 秒**（包含 build/load/eval），买到的是一个具体否定结果：此中间点不值得替换 final。没有证据可把它报告为节约若干训练小时。
- 同次还修了 lr 和 greedy config，因而不能把训练后 +2.7 pp 全归给 checkpoint 建议；更不能把 scientist 明确不采纳的“放弃扩量/半量试验”算作成功的 L3 资源指导。
- 该 card 的 terminal verdict 仍有原始 access flag（`df ... /home`）；这里仅报告可审计行为链，不将其提升为 clean predictor 效果证据。

**本例支持的“什么时候有用”：scientist 已有正确意图，但两个配置之间的相互作用让意图落不了地时，WMA 通过读具体脚本把遗漏变成可执行修复。价值是保住可比较的实验选项，并完成一次真实排除；本次未带来观测到的分数提升。**

## 最近的反例：同一个 D cell 的 exp-05 保存采纳，但中途评估/早停没有闭环

1. exp-05 原计划已经 `save_steps=1080, save_limit=4`；22:18:01 的 WMA 提醒预计2154步下只保留一个约半程的中间点，建议改为540，以保留540/1080/1620+final（8033–8040）。它另建议 checkpoint-1080 一落盘就 greedy eval，若明显低于0.7533，则节省剩余约35分钟（8043）。
2. 22:18:17 scientist 真正改 `1080→540` 并重锁；理由明确归于 WMA（8056–8070，exp-05.lock:24–29）。22:27:11 使用540/limit4正式 launch（8219–8224）。
3. **23:09:13 已看到 checkpoint-1080/540 落盘，而且还有约30分钟训练。** 紧接着23:09:18的真实动作是 `sleep 1900` 等训练结束（8299–8323），不是评 checkpoint-1080。23:40:58全部训练完成，确认540/1080/1620/2145均在（8339–8347）。
4. 从本卡 launch 到 close 的完整可见轨迹，只有 final 的n150/n500和incumbent n500评估；未找到任何单独 checkpoint-540/1080/1620 的 eval 命令或结果。于是保存建议采纳了，**C5比较/早停没有执行；不能计为GPU节省或checkpoint selection成功。**
5. 这些 checkpoints 并非永远未使用：下一张exp-06把1080/1620/final做成tail soup。它在n150与final同为0.7733，n500为0.774 < final0.790，scientist保留final（9430、9461–9463）。所以观察到后续使用，但未观察到候选收益。
6. 与之相对，同张 exp-05 的“较大 n 再对比”建议是真的执行了：22:27:08 scientist 明确预留预算（8220），随后实际跑两个n500，并在00:01:16读到0.790 vs0.768、45 fixes/34 regressions、paired SE1.78pp（8404–8408、8521–8545）。**不过扩样本并非全新意图**：上一张exp-04 close已经写“Confirm eventual winner at a larger n”（exp-04.yaml:136）；所以准确归因是本次 WMA 对已有打算的强化与具体化，不是独自创造了该评测策略。

## 这三段原始事实允许说什么

- WMA 在“已有候选上的具体补充选择”和“计划×实现遗漏的纠正”上，出现了明确的新动作链。
- 是否有用不能只看一句建议被抄入 card：同一 scientist 下一张卡已经说明，保留 checkpoint、真正评测、据评测改变选择是三件不同的事。
- 这两条被指定的正面例子里，最强证据是**改变局部候选集合/修复配置并补做比较**。没有一条足以独立证明 cell 的最终 official score 因 WMA 提升，更没有已量出的净 GPU 小时节约。
