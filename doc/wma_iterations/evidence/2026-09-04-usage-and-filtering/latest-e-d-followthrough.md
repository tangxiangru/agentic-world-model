# 最新 E 与新增 D4 的否定判决后续

以 `/home/robtang_google_com/gangda_workspace/agentic-world-model/doc/wma_iterations/evidence/2026-09-04-usage-and-filtering/complete-cells.json` 的最新 validator 结果为准。F 的两个“L3=yes但precondition让全量方案不再运行”由另一专项报告负责，不能把本报告的E结论推广到F。

## E：当前没有实际 L3 veto，已写成正式卡的方案也全部执行

| Cell | review requests | 标准summary | 补回的初次JSON verdict | terminal cards | final execution=completed |
|---|---:|---:|---:|---:|---:|
| w11r01 | 11 | 11 | 0 | 8 | 8 |
| w11r02 | 6 | 5 | 1 | 5 | 5 |
| w11r03 | 9 | 9 | 0 | 8 | 8 |
| w11r04 | 9 | 9 | 0 | 8 | 8 |

**35/35次review的L3都是yes，0no、0defer。29/29终态card也是yes，29/29 execution=completed。**已确认实际WMA skill hash为E的`20c8837ab7a4`。w11r02缺失的标准summary不是遗漏未知判决：exp01第一次review正文在 `solve_parsed.txt:3325–3326` 明确L3=yes，时间9月3日23:08:21；随后因concurrency等配置修改relock，不是veto。

所有已保留正式card都有实际完成结果，没有not_run/abandon_line的正式card。因此在这个最新E批次，既没有枚举上的L3否决，也没有像F那样通过precondition取消全量card的可见案例。某些card最后reject是看到实际结果后的scientist选择，与prelaunch否决不同。这个范围不能回答未提交给WMA、被删除的草稿想法。

## 新增 w09r04：L0=no后修复再训练；没有永久否决

根目录：
`/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r04_formal_r9/gsm8k_google_gemma-3-4b-pt_91027`

- 9月4日05:44:34，exp07第一次（也是最终保留的）verdict：L0=no@.88、L1=yes@.60、L3=yes@.55（trace:11244）。05:44:49完整首要建议指出parent checkpoint900的greedy GenerationConfig会令step460保存失败；建议修parent副本或模型内存配置，并做save smoke（:11265–11266）。
- 05:44:56 scientist实际复制parent、还原base config，跑8step/save6的smoke（:11274）；05:47:41工具输出checkpoint/final保存成功（:11306–11310）。
- 05:47:51 scientist把原parent的generation_config修成base配置，直接运行全量exp07（:11322）。没有记录这次修复后再relock；因此最终文件仍保留L0=no，不应把它与已改过parent的成功结果直接配成“误判原方案”。
- full训练完成，checkpoint460=.7081@n1319，final=.6990，均低于incumbent .7165；scientist在实测后reject并保留旧模型。没有训练线因这个no被永久放弃。

D4共9requests，8条标准summary均L3=yes，额外JSON-negative扫描无新增negativeL3。缺失的一条summary没有在本次被完全补出，因此只说“恢复到的L3没有negative”，不制造9/9结论。七张终态卡全部completed。

## 对当前问题的直接约束

E批次不能支持“WMA把scientist的好实验普遍否了”；这里它没有发出任何L3no/defer。D4也主要表现为找具体bug、促成修复。**但不能据此说当前WMA从不筛掉计划：最新F确有两张full训练卡被precondition探测后关闭not_run，尽管L3最后是yes。**所以必须把“verdict枚举”“建议体触发的试探/停止”“实际full run是否执行”分开。F被筛掉的全量方案没有反事实结果，不能已证为合理节省，也不能已证为有害误杀。

## E精确根目录

- w11r01: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-e-l3-evidence-x4_w11r01_formal_r11/gsm8k_google_gemma-3-4b-pt_91173`。card与review字段详见配套JSON。
- w11r02: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-e-l3-evidence-x4_w11r02_formal_r11/gsm8k_google_gemma-3-4b-pt_91174`。card与review字段详见配套JSON。
- w11r03: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-e-l3-evidence-x4_w11r03_formal_r11/gsm8k_google_gemma-3-4b-pt_91175`。card与review字段详见配套JSON。
- w11r04: `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r03-e-l3-evidence-x4_w11r04_formal_r11/gsm8k_google_gemma-3-4b-pt_91176`。card与review字段详见配套JSON。
