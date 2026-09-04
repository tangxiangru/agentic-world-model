# v0.2 baseline扩展w10r05–08：负向建议与实际动作复核

范围仅新validator-clean的4个baseline extension cells，来自 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/new-cells.json`；批次 `wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1`。没有把E/F新候选混入baseline。扫描原始scientist transcript、每一张终稿card/verdict以及所有processed/request/response文件。

## L3与请求口径

|cell|requests/responses|request单card|终稿cards|raw完整verdict摘要行|重建L3 yes/no/defer|not_run或abandon_line|
|---|---:|---:|---:|---:|---|---:|
|w10r05|9/9|9|8|9|9/0/0|0|
|w10r06|7/7|7|7|1|7/0/0|0|
|w10r07|9/9|9|7|9|9/0/0|0|
|w10r08|8/8|8|6|8|8/0/0|0|
|合计|33/33|33|28|27|33/0/0|0|

w10r06只有exp01的完整摘要行被打印，其余6次没有这种摘要行；但是7请求一一对应7个card、每card恰好1次review，没有重锁替换早期verdict，所以终稿verdict可确定其余6次L3均yes。其余3cells按摘要后的verdict文件路径映射card，摘要出现次数与每card请求次数逐一吻合（w05 exp02两次；w07 exp01/02各两次；w08 exp01/03各两次）。不是以终稿yes推断被覆盖的早期判定。

33个response的ranking也均为对应单元素card；没有多张独立pending proposals排序。28cards均`result.execution=completed`。有若干`conclusion.decision=reject`，但发生在已运行且有测量的候选上，不能当成WMA启动前否决。

## 真正出现的两次L0=no：都在w10r05，均未取消方向

**exp02首次审查：实际验证了OOM风险，并修复后训练。**

- 00:59:24 WMA为L0=no@.55、L3=yes@.8，要求full-file25step smoke。[原轨迹L4480](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4480)。
- 00:59:41科学家明确说“Running its precondition”；00:59:44实际运行25step完整文件smoke。[原轨迹L4561](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4561)、[原轨迹L4567](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4567)。
- 01:00:29发生OOM，进度仍0/25，cross_entropy请求15.67GiB；01:00:57科学家确认并改token-budget batching。[原轨迹L4611](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4611)、[原轨迹L4626](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4626)。
- 01:05:08修复后的25step smoke完成；01:05:51重锁，明确写“WMA L0=no…confirmed”。01:15:46新L0=yes，01:16:05启动原SFT方向。[原轨迹L4865](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4865)、[原轨迹L4916](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:4916)、[原轨迹L5112](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:5112)、[原轨迹L5236](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:5236)。
- 最终exp02训练完成1.35h，n150 .6267，对原base .0333。它没有杀掉这个有效SFT想法；负向可行性判断触发了提前验证与batching修复。不能拿修复后完成结果说第一次L0=no误判。

**exp05（不是exp06）：checkpoint500包装缺参，补参数后照常评估。**

- 06:17:52 WMA给L0=no@.6、L3=yes@.88，指出`--fill-from`默认None、checkpoint无tokenizer/processor，包装会失败，并要求`--verify-load`。[原轨迹L9098](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:9098)、[原轨迹L9124](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:9124)。
- 06:17:59科学家实际运行带`--fill-from SNAPSHOT --verify-load`的make_final；06:18:30返回CPU load4.30B、generation_config IDENTICAL；06:18:35启动eval。[原轨迹L9142](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:9142)、[原轨迹L9181](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:9181)、[原轨迹L9188](/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-v02-baseline-ext-x4-v1_w10r05_formal_r10/gsm8k_google_gemma-3-4b-pt_91425/solve_parsed.txt:9188)。
- 该候选n150 .7067与终点相同，held-out250差2.8pp，科学家事后reject。终稿L0仍no，因为补的是执行命令而没有产生另一轮review；它不是未运行/被扼杀，更不能把事后reject归为L3否决。

## 可支撑的结论边界

这4个baseline扩展cell增加了**33次L3=yes、0次no/defer**；没有not_run/abandon_line卡，没有发现由负向L3终止的候选。两次L0负向有完整“建议→实际修复/验证→继续执行”证据。

与已独立核实的旧w10r01–04的43次L3=yes合起来，当前w10 baseline8cells为**76次L3=yes、0次no/defer**。这支持“baseline当前几乎不靠L3否决来省训练”；它**不等于**WMA没有通过事实、悲观L2、建议或延迟影响scientist自主选择，也不证明所有未提出来的idea都安全。

机器可读明细：`/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/baseline-extension-scan.json`。上述否决计数是运行过程/建议的审计口径，独立于预测准确度ledger的scorable/leak过滤口径。
