# 原始轨迹中的proposal漏斗：谁先排除方案，WMA实际审什么

本文件复核w10r02/exp06、w10r04/exp06、w10r01末段exp05的**原scientist轨迹**，并逐一核对真实request/response。结论仅覆盖这三个例子：scientist先挑出一个完整方案，把其他方向写进`alternatives_rejected`，随后提交所选单张card；WMA在此基础上预测、排查风险、建议便宜变体。未观察到这三个请求对多张独立冻结的待选方案做排名。

完整request/response复制在 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/proposal-funnel.requests.json`，每条保留原文件路径。

## 1. w10r02 / exp06：扩大corpus的两个方向在WMA之前就被scientist排除

**13:45:48，scientist写card：** 它已经选定从base在原83,473行数据上训2epochs，计划2.1h，并写好命令。理由是此前每次都只训1epoch，想单独测试epoch轴。备选已被排除：

- 更大178,907行corpus训1epoch：scientist估2.25h，认为无法留n500评估余量；
- exp03的113k corpus训2epochs：scientist估2.8h，认为不够；
- 停止训练重复测incumbent：scientist认为只能提高precision不能提高score。

这些原始文本见 [轨迹L10127](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10127)、[轨迹L10133](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10133)、[轨迹L10190](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10190)。

**13:45:53才产生WMA request** `20260903T134553.456048Z-574fc325`，`card_ids=["exp-06"]`。**13:51:52 response** 为 `ranking=["exp-06"]`；没有另两份大corpus方案的独立card进入ranking。

**WMA的实际输出：** 对选中的2epoch方案给L2 flat[-.03,.035]、L3 yes@.55、expected_cost2.3h。建议核算退出时刻、先评已存在checkpoint，以及预算变紧才改1.4epochs；没有否决选中的2epochs，也没有用独立预测重新排序那两个已排除的大corpus方向。[轨迹L10370](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10370)、[轨迹L10386](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10386)、[轨迹L10392](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10392)、[轨迹L10394](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10394)。

**13:54:52实际按2epochs启动**（未缩成1.4）：[轨迹L10408](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r02_formal_r10/gsm8k_google_gemma-3-4b-pt_90983/solve_parsed.txt:10408)。最后n500 .738，高于同corpus已有.688及当时soup .710。此例分类为：**scientist预先筛掉大corpus；WMA审查并允许所选训练；所选训练后来有效。** 不能说那两个未运行方案是WMA误杀，更不能知道它们若运行会更好还是更差。

## 2. w10r04 / exp06：训练形式也是scientist先选，WMA审的是低LR continuation

**14:33:42，scientist先写出完整方案：** 从exp04模型继续一epoch，57k fresh chains，lr5e-6，计划.9h。它已经排除：

- 第二轮RFT：采样约20min，前一轮只+1.1pp且未显著，scientist更偏好现成teacher chains；
- 从base重训168k行：估88Mtokens/epoch=2.5h，对剩3.1h认为没有评估/打包余量；
- 更高LR：science判断此前continuation的regression风险不值得；
- 当场停止：科学家认为incumbent已保护，仍有条件试一次。

原记录 [轨迹L10790](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:10790)、[轨迹L10808](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:10808)、[轨迹L10841](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:10841)、[轨迹L10876](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:10876)。

**14:33:50 request** `20260903T143350.089722Z-e6277a20`仅有`["exp-06"]`；**14:39:33 response** ranking同样只有它。

WMA给所选方案L2 flat[-.012,.018]、L3 yes@.72、expected_cost1.4h；实际读到的preconditions集中在generation_config、同尺full-test比较、incumbent保护和启动日志检查。其其他便宜变体包括保留/评mid-epoch checkpoint，时间紧才减为28k行；没有再次提出从base168k的独立完整方案比较。[轨迹L11019](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:11019)、[轨迹L11039](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:11039)。

**14:42:28按原57k一epochcontinuation启动** [轨迹L11048](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r04_formal_r10/gsm8k_google_gemma-3-4b-pt_90985/solve_parsed.txt:11048)，full1319 .7119 vs incumbent .7134，后被scientist按预写规则拒绝覆盖最终模型。这是**scientist先选择低成本continuation，WMA给予谨慎预测但允许执行，结果不支持替换**；不是WMA把可能更好的from-base/RFT否决了。后两者根本没有各自作为完整待选card送审。

## 3. w10r01末段：决定不再做第二RFT发生在提交“打包验证”card之前

**14:16:43**，scientist先说“Final step: verifying … final_model … plus a wider read”。**14:17:26**，它在exp05（eval/verification）card中把第二RFT排除：剩2.8h，认为采样+训练+选择+eval约1.8h却“no slack”，而上一轮只+2pp；同时也把exp03对照n500排除，称多花约40min、不会改提交方向。[轨迹L15171](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:15171)、[轨迹L15227](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:15227)、[轨迹L15228](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:15228)。

**14:17:31 request** `20260903T141731.540253Z-496cb10e`只审`["exp-05"]`，这是打包eval卡，不是第二RFT卡；**14:22:44 response** ranking=`["exp-05"]`。WMA对这次eval给L3 yes@.88、成本.3h，建议n150先做、再决定n500，确认serving环境等；没有给第二RFT一个L3=no verdict。[轨迹L15468](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:15468)。

**14:27:06实际启动final_model eval** [轨迹L15487](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-x4-v2_w10r01_formal_r10/gsm8k_google_gemma-3-4b-pt_90982/solve_parsed.txt:15487)。以后继续checkpoint选择、soup、full1319验证，16:09结束仍有约1:02。在这个明确例子里，**不继续训练是scientist的预筛/收尾决定**；WMA接受了送到它面前的验证方案，没有主动重开第二RFT，但也没有否决一个完整第二RFT proposal。

## 4. 已核实的反向边界：L3=yes仍可能触发scientist自行取消

A/w06r02 exp06是不同路径：scientist先选了26k“新问题”SFT；WMA L3=yes但指出25,965/26,000问题已见。scientist再查更大pool仅56条新问题，主动关闭未运行卡。证据链已逐条复核于 `/tmp/wma-deep-analysis/empirical-uptake-a.md`（原轨迹L7135→7246→7265→7387→7398）。

这里取消确实发生在WMA的新事实之后，但仍不是`L3=no`强制否决；也没有被取消训练的结果可证明误杀好想法。它显示WMA能影响选中的proposal是否继续，影响来自事实核查而非负向标签本身。

## 这几条轨迹支持的归因

| 被排除/改变的方向 | 实际决定方和时点 | 可否称WMA否决好想法 |
|---|---|---|
| w10r02更大corpus1epoch、113k2epochs | scientist在13:45:48、request之前排除 | 不可；没有对应候选结果，也非独立送审方案 |
| w10r04从base168k、第二RFT、更高LR | scientist在14:33:42、request之前排除 | 不可 |
| w10r01第二RFT | scientist在14:17:26、验证card request之前排除 | 不可；WMA实际审验证卡 |
| A/w06r02原26k训练 | WMA给新重合事实后，scientist实测并取消；L3仍yes | 可归于WMA引发事实纠错，不能证明是误杀 |

本次三个单例的response虽然带`ranking`字段，内容全是单元素列表：它本身没有展示“在多份独立可执行pending proposals中挑最好”的动作。备选方向会以文字出现在已选card的`alternatives_rejected`，WMA并非完全看不到它们；但“看过被拒备选的摘要”与“对每份方案有独立预测并实际进行比较”是不同证据强度。
