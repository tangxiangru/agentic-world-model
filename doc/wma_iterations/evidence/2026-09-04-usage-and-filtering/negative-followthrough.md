# 冻结R2候选的negative verdict后续：是修复还是误杀？

范围：15个完成的A/AB/C/D cells；已恢复的8次negative L3，落在7张不同card。另查3次L0/L1负面但L3=yes的episode。这里只陈述实际后续。

## 结论

8次negative L3中7个defer、1个no；7张相关card最后全部执行完成，没有一张被永久放弃。**这不等于所有原方案都原样获准：w07r03/exp03的第一次defer确实促使scientist不用原GSM-only语料，改成加入34k新MATH问题的方案。**原语料版本没有运行，其潜在收益未知。没有识别出“已证明会成功的提案被WMA永久否决”的实际损害；不能由此断言不存在误伤。

| episode | 原否定针对什么 | 实际后续 | 实际结果/限定 |
|---|---|---|---|
| w07r03/exp-03:7108 defer | 0 of110000 claimed unseen augmented rows were on new problems;28339 exact repeats. | 18:24:36 scientist explicitly accepts diagnosis; original GSM-only sft_pool_v3 retained but not run. Rebuilds around34000 genuinely new MATH rows,52000 GSM solutions,28339 anchors. | 0.6867@n150 (delta-0.0333); 0.716@n500 (delta0.008); 0.7367@n300 (delta0.0267); final decision=adopt. The originally proposed dataset is not run. This is a real direction/data change, not merely a typo fix. Its score is unobserved, so the change cannot be classified as a beneficial or harmful rejection from the final revised result. |
| w07r03/exp-03:7719 defer | GenerationConfig save-time failure plus stale setup.data pointing at old file. | 18:36:52 reads blocker; CPU reproduces it; script fixed, actual2-step save passes; data path corrected; checkpoint sweep400/600/789 done(.690/.7033/.710).18:51:02 new verdict yes;18:51:09 launches full revised fit. | 0.6867@n150 (delta-0.0333); 0.716@n500 (delta0.008); 0.7367@n300 (delta0.0267); final decision=adopt. Same card as prior episode; do not count this as a second full training or independent outcome. The proposed scientific line continued. |
| w08r03/exp-03:6257 no | do_sample=false did not produce greedy decoding; absent temperature made sampling broader than parent. | 19:00:48 scientist says caught real defect; writes temperature0 and fixes builder;19:01:13 relocks;19:06:39 new verdict yes;19:06:44 eval launched. | 0.6533@n150 (delta0.1); final decision=adopt. The positive result belongs to corrected greedy artifact, not the original no-op artifact. No permanent veto of greedy experiment. |
| w09r01/exp-04:6575 defer | Final-only/rotated checkpoint plan forfeited C5; effect near noise; unaligned lr confounded data-quantity claim. | Scientist fixeslr back1.5e-5 and save1490/limit4, explicitly records disagreement with economic defer; relocks and receives yes; launches full training. | 0.7533@n150 (delta0.0266); 0.6933@n150 (delta-0.0334); final decision=adopt. Scientist did not obey a stop-the-line instruction: it retained training ambition while satisfying concrete conditions. |
| w09r01/exp-06:9011 defer | Raw on-disk tensor keys do not match live model load_state_dict keys(0/883). | 00:12:58 scientist calls correct catch; writes tensors under disk keys and reloads via from_pretrained;00:20:30 new verdict yes;00:21:13 actual reload OK4.300B. | 0.7733@n150 (delta0.0); 0.774@n500 (delta-0.016); final decision=reject. Merge was executed after repair. Soup was later rejected from measured performance, not prevented by WMA. |
| w09r02/exp-02:4881 defer | fp32 checkpoint serving risk and no useful mid-epoch checkpoints. | Re-lock adoptsbf16 final save andsave200/limit3, names final eval target, validates end-to-end smoke; full743step fit completes and checkpoints are scored. | 0.6866666666666666@n150 (delta0.64); 0.6533333333333333@n150 (delta0.6067); 0.6333333333333333@n150 (delta0.5867); final decision=adopt. First SFT not killed; final.6867 from.0467, intermediates.6133/.6333/.6533. |
| w09r02/exp-04:7457 defer | Defaultsave500 never fires in~196step fit, leaving no selection checkpoints. | Changes tosave60/limit4 and relocks; first actual run fails atstep60 because parent GenerationConfig rejects saving; second relock fixes config; full196step fit completes. | 0.7666666666666667@n150 (delta0.0133); final decision=iterate. There is real failure/recovery cost after accepting checkpoint advice, but finalsave had the same underlying bug; cannot attribute the entire cost to a harmful veto. No permanent abandonment. |
| w09r03/exp-04:6753 defer | Final-only checkpoint plan plus ensuring greedy packaging and two-sided diagnostic. | Relock changes save1484 to370 and records greedy repackaging; launches full1484step RFT fit. Surviving1110 andfinal both score.6933 vs incumbent.7467; rejects after measurement. | 0.6933@n150 (delta-0.0534); 0.6933@n150 (delta-0.0534); final decision=reject. No deferred experiment was permanently withheld; later rejection is scientist outcome-based, not WMA veto. |

## AB两次defer的关键原文与时间

- 9月3日18:23:44，w07r03/exp03第一次defer（trace:7108）。18:23:52 probe p2给出110000/110000 augmented rows均为已见问题（:7181）。18:24:36 scientist明确说“verdict caught a real error”（:7212）。18:30:47记录原v3“kept on disk but not used”（:7592）；18:31:28 relock改为114339行：34000新MATH、52000新GSM解法、28339anchors（:7673）。这是实际改变了数据方向，不能只称修配置。

- 18:36:34第二次defer（:7719）：save-time blocker，并指出setup.data还指着旧文件。scientist18:37:16说“Reproduced-and-confirmed launch blocker”（:7769）；18:37:32承认setup.data修改没生效（:7830）。18:43:30确认实际checkpoint存出weights（:8199）；还执行此前建议的checkpoint400/600/789 sweep，分数.690/.7033/.710（:8207）。18:51:02新review为yes（:8270），18:51:09全量训练启动（:8286）。最终.6867@n150、.716@n500、.7367@local300：读数方向不一致，采用但结论inconclusive。

## L0/L1负面但L3仍yes：3个额外episode

- w07r03/exp-04:9423 — L0=no/L1=no; L3=yes。Reset GenerationConfig before save and deduplicate tied embedding/lm_head storage in averaging. Relock then merge runs; n500=.702 vs bestingredient.716, subsequently rejected.
- w07r04/exp-02:3972 — L0=no; L3=yes。Fix save_model ordering: save_pretrained first, patch greedy config on disk after.64-rowend-to-end save smoke passes; relock then full1632step SFT completes.660@n150 from.0467.
- w08r02/exp-06:12214 — L0=no/L1=no; L3=yes。Checkpoint900 already deleted; scientist misread filtered package log as success. Changes arm to1200, confirms2shards8.6GB and reruns review; all3 eval arms complete.

L0/L1负面还与上述8个L3negative重叠两次（AB w07r03 exp03第二review、D w09r01 exp06），不重复计数。negative union为11次episode。

## 可复核路径

- w07r03/exp-03 episode@7108: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r03_formal_r7/gsm8k_google_gemma-3-4b-pt_91009/solve_parsed.txt:7108`；其余精确path/line和card result均保存在JSON evidence数组。
- w07r03/exp-03 episode@7719: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-ab-format-floor-width-x4-v2_w07r03_formal_r7/gsm8k_google_gemma-3-4b-pt_91009/solve_parsed.txt:7719`；其余精确path/line和card result均保存在JSON evidence数组。
- w08r03/exp-03 episode@6257: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r03_formal_r8/gsm8k_google_gemma-3-4b-pt_91022/solve_parsed.txt:6257`；其余精确path/line和card result均保存在JSON evidence数组。
- w09r01/exp-04 episode@6575: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt:6575`；其余精确path/line和card result均保存在JSON evidence数组。
- w09r01/exp-06 episode@9011: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024/solve_parsed.txt:9011`；其余精确path/line和card result均保存在JSON evidence数组。
- w09r02/exp-02 episode@4881: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r02_formal_r9/gsm8k_google_gemma-3-4b-pt_91025/solve_parsed.txt:4881`；其余精确path/line和card result均保存在JSON evidence数组。
- w09r02/exp-04 episode@7457: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r02_formal_r9/gsm8k_google_gemma-3-4b-pt_91025/solve_parsed.txt:7457`；其余精确path/line和card result均保存在JSON evidence数组。
- w09r03/exp-04 episode@6753: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r03_formal_r9/gsm8k_google_gemma-3-4b-pt_91026/solve_parsed.txt:6753`；其余精确path/line和card result均保存在JSON evidence数组。

## 证据覆盖限制

146个requests有138条标准打印level summary。缺失8条集中在w06r01五条、w08r01两条（其中一次是已知posthoc失败）、w08r02一条。对全部15条scientist trace额外搜索JSON结构的L3=no/defer，未发现这8次之外的negative episode。由于superseded review正文并非全保留，准确说法是“对全部已恢复negative episode穷尽核对后续”，不是声称每个不可见review都绝无negative。

因此当前证据更支持：WMA多用于审查一个已经写好的方案、指出具体前提错误、促使修后再做；本批negative很少表现为永久砍掉训练。它有没有让scientist最初提出太少候选，需要看提交给它的proposal集合，本项审计不能把不存在的候选归因于WMA否决。
