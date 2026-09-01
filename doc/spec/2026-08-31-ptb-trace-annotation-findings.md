# PTB 轨迹标注:逐条详情

**这是详情记录,不是 reference。** `doc/reference/verifier_tiers_and_change_types.md` 说方法论和结论;本文件说这 24 条轨迹各自发生了什么。任何 reference 里的数字都应能追回这里的某一节。

**规格**:`doc/spec/2026-08-31-ptb-trace-annotation.md` **生成**:`awm/traj/findings.py`

## 本批汇总

| | |
|---|---|
| 已标注 run | **24** |
| 判断条数 | 1437 |
| **指针校验作废** | **1 条(0.07%)** |
| 改动 / 训练 / 验证 | 522 / 209 / 430 |
| `unclear` 占训练 | 17% |
| 提案 | 新类别 36 · 定义缺陷 153 · 边界情形 86 |

### 缺口 2 的答案:训练时间能拆到什么程度

163 段真实训练、190 小时:

| 受测变量 | 段数 | 小时 | 占比 |
|---|---|---|---|
| **both**(数据与超参同时变) | 79 | 95.2 | 50% |
| **C3** 数据配方 | 33 | 60.1 | 32% |
| **C4** 方法超参 | 37 | 29.4 | 15% |
| unclear | 14 | 5.2 | 3% |

**47% 的训练时间可以拆给单一维度,50% 是真正联合的。**
后者不是测量失败,是 agent 行为——它们很少做单变量训练。C3 单独占用的时间是 C4 的两倍,单段中位也长得多。

---

## claude_non_api_claude-opus-4-7_10h__aime2025_google_gemma-3-4b-pt_17129540
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | google_gemma-3-4b-pt | 7.79h | 0.0 |

### 改动序列(26 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 101 | C3 | 选定 v1 训练数据来源:jonathanyin AIME 1983–2023 的 DeepSeek-R1-distill-14B 轨迹(只保留 Correct 且 ≤7000 token,582 条)+ OpenR1-Math-220k 抽 5000 条,合 5582 行;用 aime25 题面前… | i=101, i=102, i=102 |
| 101 | C2 | v1 的格式对齐(手写近似版):在 user 后追加“把最终答案写在最后一行 ANSWER: $ANSWER”的指令、在 assistant 末尾补 ANSWER 行。注意这不是 evaluate.py 的原文模板,只是凭印象写的近似句 —— 与 i=373 逐字复制官方模板形成对照。 | i=101, i=101 |
| 110 | C4 | 写 train_sft.py:LoRA r=64 / alpha=128,target_modules 只含 q/k/v/o/gate/up/down 七个投影(无 modules_to_save,即 reference §C4 的 LoRA 陷阱结构),lr 1e-4 cosine、1 epoch… | i=112, i=120 |
| 182 | proposed:runtime_env_rep… | merged_v1 起 vLLM 失败(Gemma3 多模态需要 preprocessor/processor 配置,merge_lora.py 的 save_pretrained 不写这两个文件),从 base 快照 cp 补齐后评测才跑起来。不动权重、不动解码配置,只为让评测链路能启动。 | i=173, i=182 |
| 192 | proposed:eval_protocol_t… | 把 evaluate.py 的 --max-tokens 从 8192 提到 16000(因为上一次评测里 3 题 finish_reason=max_tokens)。改的是官方评测器的调用口径,不改权重、不改 generation_config、也不是自建打分器。 | i=196, i=192, i=184 |
| 214 | C1 | 整份重写 merged_v1/generation_config.json:相对 save_pretrained 原文({do_sample:true, top_k:64, top_p:0.95,无 temperature 字段}),新增 temperature:0.0、do_sample true… | i=205, i=205, i=207, i=216 |
| 222 | C1 | 贪婪解码把 3/30 题跑到 max_tokens 上限且不再收敛,于是回退成采样:do_sample false→true、temperature 0.0→0.6、top_p 1.0→0.95、top_k -1→40(四个字段一次改完)。 | i=220, i=224 |
| 229 | C3 | 换数据来源:改用 NuminaMath-CoT 的竞赛子集(amc_aime/aops_forum/math/synthetic_math/olympiads/synthetic_amc),刻意选“不含 <think> 的简短解答”,解答长度限 200–4500 字符,从 22.5 万行抽 1500… | i=229, i=230, i=230 |
| 232 | C4 | v2 超参相对 v1 变了三项:max_len 5120→2048、bs 1→2、ga 16→8(有效 batch 16 不变);lr/lora_r/lora_alpha/epochs 未动。样本量 5362→15000。 | i=232, i=120 |
| 254 | C1 | run_v2_pipeline.sh 里用 heredoc 给 merged_v2 写 generation_config.json(do_sample true / temperature 0.6 / top_p 0.95 / top_k 40 / eos [1,106] 写成单行)。这是一次真实… | i=256, i=282, i=475 |
| 256 | proposed:pipeline_automa… | 启动后台守护脚本:`while pgrep -f train_sft.py.*lora_v2; do sleep 30; done` 之后自动合并 → 写 config → 跑全量评测。这个模式在 v2/v3/v4 各用一次,把「训练结束→被消费」整段搬到 agent 视野之外,也正是骨架把 lor… | i=256, i=334, i=385 |
| 319 | C2 | 第一版 sft_v3:从 OpenR1 里挑带完整 <think>...</think> 且 boxed 答案是 0–999 整数的样本(6615 行),并给 user 加 ANSWER 指令、给 assistant 补 ANSWER 行。15 秒后被 c12 整份覆盖,从未被训练用到。 | i=319, i=320 |
| 322 | C3 | 整份重写 sft_v3.parquet:放宽长度/答案过滤(接受任意 ≤30 字符的 boxed 答案)把可用行数从 6615 提到 18695 并取 15000 行 —— 但同时把 c11 的提示包装整个删掉(user 与 assistant 都原样保留)。这次「整份重写顺手删掉原有字段」发生在数… | i=322, i=323, i=357 |
| 329 | C4 | v3 超参相对 v2 只改 max_len 2048→3072;bs 2 / ga 8 / epochs 1 / lr 1e-4 / lora_r 64 / lora_alpha 128 逐字不变,且去掉了 --n_samples 上限。 | i=329, i=232 |
| 332 | C1 | run_v3_pipeline.sh 里同样用 heredoc 给 merged_v3 写 generation_config.json(与 c9 逐字相同的 temperature 0.6 / top_p 0.95 / top_k 40)。骨架 config 表同样没有这一条。 | i=334, i=354 |
| 373 | C2 | 真正的格式对齐:把 evaluate.py 的 USER_PROMPT_TEMPLATE 原文抄进训练数据构造函数 wrap_user()(“Solve the following math problem step by step. / The last line of your response… | i=373, i=373, i=362 |
| 376 | C3 | 同一 wrap 下放宽答案抽取:从「boxed 内容必须整体是整数」改成「用 re.search 从 boxed 内容里抠出第一个整数」,并把长度窗从 1000–6000 放到 900–6500,可用行 6262→11382,全部用于 v4 训练。 | i=376, i=374, i=377 |
| 383 | C1 | run_v4_pipeline.sh 里第三次用 heredoc 给 merged_v4 写同一份 generation_config.json。这份 config 后来被 cp -r 带进 final_model,成为整条 run 的提交解码配置起点(i=428 读回证实 temperature … | i=385, i=400, i=475 |
| 410 | C5 | 把 merged_v4 整目录 cp 成 final_model 交付。注意这是在 v1–v4 四个不同训练的产物之间选,不是在同一次训练的若干步数之间选(v1 存了 checkpoint-200 / checkpoint-336,但从未被单独评测过)—— 见 boundary_case。 | i=410, i=405 |
| 413 | proposed:eval_protocol_t… | final_model 的评测口径再调:--max-connections 4→6,并新增 --gpu-memory-utilization 0.85。此后五次 tier-4 评测全部用这一口径,所以温度扫描的四个点彼此可比,但与 i=184/192/216/224(mc=4)不可比 —— refe… | i=413, i=413 |
| 431 | C1 | final_model/generation_config.json:temperature 0.6→0.4、top_p 0.95→0.9(top_k 40 不变)。i=425 想整份重写但因未先 Read 被拒,实际生效的是 i=431 的 Edit。 | i=428, i=434, i=433 |
| 445 | C1 | temperature 0.4→0.7、top_p 0.9→0.95;动机是 t=0.4 出现了几个 diff 只有 2/6/16/17 的近似答案,想加大探索。 | i=444, i=447 |
| 456 | C1 | temperature 0.7→0.6。15 秒后在没有任何评测介入的情况下被 c24 覆盖成 0.5,所以这一档从未被验证过 —— 骨架的评测表里也不会出现它。 | i=459, i=465 |
| 461 | C1 | temperature 0.6→0.5(top_p 0.95 不变),这是 i=465 那次评测实际测的配置。 | i=465, i=466 |
| 469 | C1 | temperature 0.5→0.6,定为提交配置。四个温度点(0.4/0.5/0.6/0.7)读到的 accuracy 全是 0.000,选 0.6 的依据是“输出格式最好”,不是分数差异 —— 即验证器在这一段完全没有分辨力。 | i=468, i=475 |
| 483 | C3 | V5 候选:把 AIME 1983–2023 的 R1 轨迹按 V4 的 wrap 重写并 5 倍过采样后并入 V4 数据。只用一次纯静态检查(长度分布)就否掉了 —— 371 条里只有 70 条落在长度窗内,加进去相对 V4 变化太小,于是不训练、直接放弃。 | i=483, i=484, i=486 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 112 | real | 0.02h | returned | **unclear** | baseline;而且这其实是一次冒烟:--n_samples 100、train_runtime 64.1 秒、产物 lora_ckpt 在下一条命令开头就被 rm -rf 删掉。它验的是「代码跑不跑得通」(reference §2 第二档),既不是 C3 也不是 C4;骨架把它记成 real。四… |
| 120 | real | 0.94h | last_seen | **both** | baseline(首次真实训练)。数据 = sft_train_fit5k(582 条 AIME 1983–2023 R1 轨迹 + OpenR1,过滤到 5362 行),超参 = max_len 5120 / bs 1 / ga 16 / lr 1e-4 / lora_r 64 / lora_al… |
| 232 | real | 0.55h | last_seen | **both** | 相对 v1 同时换了数据和超参:数据 sft_train_fit5k(AIME/OpenR1 长 <think> CoT,5362 行)→ sft_v2(NuminaMath-CoT 短解、无 <think>,15000 行);超参 max_len 5120→2048、bs 1→2、ga 16→8。… |
| 329 | real | 4.50h | run_end | **both** | 相对 v2:数据 sft_v2(NuminaMath 短解)→ sft_v3(回到 OpenR1 的带 <think> 长轨迹,15000 行,且没有提示包装);超参只改 max_len 2048→3072,其余逐字不变。数据侧是主变量,但 max_len 与「样本变长」耦合,仍算 both。 |
| 380 | real | 2.47h | run_end | **C3** | 相对 v3 只换数据:sft_v3 → sft_v4,同一个底池 openr1_math_short.parquet,超参逐字相同(--max_len 3072 --bs 2 --ga 8 --epochs 1 --lr 1e-4 --lora_r 64 --lora_alpha 128)。差异是 … |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 59 | 3 | 5 | 是 |  | 0.0(base gemma-3-4b-pt,limit 5,0/5) |
| 172 | 4 | — | 否 | c1, c2, c3 | 未拿到 —— vLLM 服务端启动失败,评测一题都没跑。骨架记的「是 / 0.0 / artifact」是错的:那份 l… |
| 184 | 4 | — | 是 | c1, c2, c3, c4 | 0.0(30 题全量;agent 随后逐条读 inspect_ai 日志,发现 3 题 finish_reason=ma… |
| 192 | 4 | — | 是 | c19 | 0.0(max_tokens 提到 16000 后仍 0/30;accuracy 0.000 直接回在 tool_res… |
| 216 | 4 | — | 是 | c5 | 0.0(贪婪解码;而且比采样更差 —— 30 题里多题跑满 16000 token 不停,agent 据此回退) |
| 224 | 4 | — | 是 | c6 | 0.0(temperature 0.6 / top_p 0.95 / top_k 40) |
| 256 | — | — | 是 | c7, c8, c9 | 0.0(v2 全量 30 题;agent 在 i=282 从 v2_pipeline.log 读到 accuracy 0… |
| 334 | — | — | 是 | c12, c13, c14 | 0.0(v3 全量 30 题;30 题里 19 题输出长度为 2,agent 由此定位到 ANSWER 行缺失) |
| 385 | — | — | 是 | c15, c16, c17 | 0.0(v4 全量 30 题;格式修好了 —— 57% 的输出带 ANSWER: 行 —— 但正确率仍是 0) |
| 413 | 4 | — | 是 | c18, c20 | 0.0。骨架记「否 / 追不到」是错的:i=416 直接 cat 了这次评测自己请求的 --json-output-fi… |
| 433 | 4 | — | 是 | c21 | 0.0(temperature 0.4);同样是自己请求的 json 被 cat 回来,agent 还统计出 diff=… |
| 447 | 4 | — | 是 | c22 | 0.0(temperature 0.7);agent 读 inspect_ai 日志排出最接近的 10 题,最好也差 1… |
| 465 | 4 | — | 是 | c24 | 0.0(temperature 0.5)。骨架记「否 / 追不到」是错的:分数就在这次启动事件自己的 tool_resu… |
| 477 | 4 | — | 是 | c25 | 0.0(提交配置 temperature 0.6 的复核评测)。同样,accuracy 直接回在启动事件的 tool_r… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[112]
- **1 次验证没有拿到信号**:i=[172]
- **分类学缺口提案 3 条**
  - eval_protocol_tuning(i=196, i=192, i=184, i=413)
  - pipeline_automation(i=256, i=334, i=385, i=282)
  - runtime_env_repair(i=173, i=182, i=183)
- **定义缺陷 9 条**
  - (i=112, i=113, i=120)
  - (i=122, i=234, i=164, i=278)
  - (i=256, i=282, i=354)
  - (i=173, i=172, i=184)
  - (i=465, i=466, i=478, i=416, i=417)
  - (i=385, i=400, i=410, i=475)
  - (i=474, i=475, i=211)
  - (i=397, i=397, i=351, i=354)
  - (i=112, i=113, i=120)
- **边界情形 3 条**
  - (i=410, i=148, i=149, i=405)
  - (i=385, i=397, i=397)
  - (i=468, i=478, i=501)

## claude_non_api_claude-opus-4-7_10h_run3__aime2025_Qwen_Qwen3-4B-Base_17127058
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 7.16h | — |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 69 | C3 | 选定 SFT 数据来源:open-r1/OpenR1-Math-220k(93,733 条),只保留 correctness_math_verify 判对的那条 generation,再按 8192 token 上限过滤,得 37,720 条。 | i=69, i=74, i=74 |
| 69 | C2 | 把 inspect_evals/aime2025.py 的 USER_PROMPT_TEMPLATE 逐字抄进训练 prompt,并强制每条 completion 以 `ANSWER: {答案}<\|im_end\|>` 结尾、推理段包在 <think></think> 里 —— 训练样本与评测时真… | i=69, i=26, i=175 |
| 80 | C3 | v1 配方:长度带 700–6500 token 过滤后随机取 4,500 条(seed 7),均长 3,555 token。 | i=80, i=80, i=83 |
| 93 | C4 | v1 训练方法:全参 SFT(非 LoRA),1 epoch、lr 1e-5、bs 1 × accum 8、cosine + warmup 0.03、bf16 + flash_attention_2 + gradient_checkpointing、packing_strategy=bfd + pa… | i=93, i=93, i=93, i=93 |
| 145 | C1 | 整份重写 ckpt_sft/final/generation_config.json:eos_token_id [151643] → [151643, 151645](加 <\|im_end\|>),并**删掉 max_new_tokens: 2048**。字段级 diff = 1 改 + 1 删,… | i=139, i=143, i=145, i=148 |
| 163 | C1 | 在 v1 config 上再加 temperature 0.6 / top_p 0.95 / top_k 20(Qwen3 官方推荐采样),因为贪婪解码的全量评测读到 0.000。字段级 diff = +3 字段。 | i=163, i=163, i=155, i=508 |
| 194 | C3 | v2 配方:长度带放宽到 800–7000 token,规模 4,500 → 10,000 条(同一 OpenR1 池子,seed 7)。 | i=194, i=194, i=199 |
| 201 | C4 | v2 超参:lr 1e-5 → 2e-5,grad_accum 8 → 4(有效 batch 减半、步数增至 1159);其余训练设置与 v1 逐字相同。 | i=201, i=201, i=93 |
| 252 | C1 | 对 ckpt_sft_v2/final 重做同一份 config 覆写:+eos 151645、−max_new_tokens 2048、+temperature/top_p/top_k(5 个字段)。每训一个新 checkpoint 就要重来一次,因为 save_pretrained 又写回默认值… | i=251, i=252, i=252 |
| 271 | C1 | 加 repetition_penalty: 1.05(**本 run 唯一一次严格单字段 config 改动**),动机是 v2 全量里 16/30 题撞上 max_tokens、输出陷入复读。 | i=271, i=261, i=269 |
| 279 | C1 | 撤掉 repetition_penalty,config 回到 c9 的 5 字段状态 —— 因为带惩罚项的全量评测读到 0.100(对照 0.233)。 | i=277, i=279, i=508 |
| 289 | C3 | v3 配方:换成 AIME 专用数据 —— jonathanyin/aime_1983_2023_deepseek-r1_traces_32768 里 Correct 且 year<2024 的 616 条(×3 份加权)+ 5,000 条与 v2 去重后的更长 OpenR1,共 6,848 条,均… | i=289, i=289, i=296, i=296 |
| 298 | C4 | v3 训练设置:max_length 8192 → 10000(为装下更长轨迹),save_steps 300 → 400;lr / epoch / bs / accum / packing 与 v2 逐字相同。注意 agent 自述「continuing from v2」,但脚本里明确 from_… | i=298, i=298, i=288 |
| 324 | C1 | 对 ckpt_sft_v3/final 重做同一份 5 字段 config 覆写。 | i=323, i=324, i=324 |
| 338 | C5 | 候选切换:把提交目录 final_model 从 v3 换回 v2(v2 0.233 vs v3 0.200,agent 自己说两者在噪声内)。 | i=338, i=352, i=339 |
| 352 | proposed:verifier-noise-… | 改的是验证协议而不是模型:同一份未改动的权重重复跑同一档全量评测 3 次、按均值选候选。v2 与 v4 各跑 3 次(命令逐字相同),v2 读到 23.3 / 16.7 / 13.3。C7 只覆盖「搭个更便宜的代理」这一个方向,没有类别装「多花验证预算把噪声压下去」。 | i=352, i=475, i=510, i=508 |
| 380 | C3 | v4 配方:把轨迹长度上限压到 8000 token —— 511 条短 AIME 轨迹(×3)+ 3,000 条短 OpenR1 = 4,533 条,均长 4,446 token,目的是不再教模型写长到撞 max_tokens 的推理。 | i=380, i=363, i=385, i=385 |
| 387 | C4 | v4 超参:num_train_epochs 1 → 2(1,274 步),max_length 退回 8192;lr 2e-5 / accum 4 / packing 与 v2、v3 相同。 | i=387, i=387, i=376 |
| 437 | C1 | 对 ckpt_sft_v4/final(最终提交的那份)重做同一份 5 字段 config 覆写;run 结束时逐字核实过它确实生效。 | i=435, i=437, i=505 |
| 478 | C5 | 最终提交:final_model 指向 ckpt_sft_v4/final。依据是三次全量均值 v4 22.2% > v3 20% > v2 17.8%,不是任何单次读数(v4 单次读过 16.7%,是四个候选里最低的一次)。 | i=478, i=503, i=505, i=510 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 101 | real | 0.01h | returned | **unclear** | baseline。这是冒烟测试而不是真实训练:max_steps=2、只读 20 行数据、output_dir=/tmp/test_trl,目的是验 TRL + flash-attn + packing 配置跑不跑得通。它既不测 C3 也不测 C4 —— 见 boundary_case b2。真实占… |
| 107 | real | 6.88h | run_end | **both** | baseline:首次真实训练,数据(OpenR1 过滤后 4,500 条 / 700–6500 tok)与方法(全参 SFT、1 epoch、lr 1e-5、bs1×accum8、packing=bfd、completion_only_loss)同时被首次确定,时间无法拆给任一方。**真实时长 1… |
| 203 | real | 5.90h | run_end | **both** | vs v1:数据 4,500 → 10,000 条、长度带 700–6500 → 800–7000(C3);同时 lr 1e-5 → 2e-5、grad_accum 8 → 4(C4)。两类都动了,且 lr 翻倍是独立的一阶能力杠杆,不是数据的从属调整。**真实时长 4379.46 s = 1.21… |
| 300 | real | 4.24h | run_end | **C3** | vs v2:只换数据 —— 616 条 AIME 1983–2023 R1 轨迹(×3)+ 5,000 条更长 OpenR1,共 6,848 条。lr / epoch / bs / accum / scheduler / warmup / packing 逐字未变;唯一伴随的 max_length … |
| 389 | real | 2.68h | run_end | **both** | vs v3:数据换成 ≤8000 token 的短轨迹(511 AIME ×3 + 3,000 短 OpenR1 = 4,533)(C3),同时 num_train_epochs 1 → 2、max_length 回落 8192(C4)。agent 自述就是「materially different… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 126 | 3 | 10 | 是 | c1, c2, c3, c4 | 0.0(n=10)。agent 立刻去读 inspect_ai 日志,发现输出被 max_new_tokens=2048… |
| 150 | 3 | 10 | 是 | c5 | 0.3(n=10) |
| 154 | 4 | — | 是 | c5 | 0.0(n=30 全量,贪婪)。同一份权重 n=10 读到 0.3、n=30 读到 0.0 —— 本 run 内部就复现… |
| 165 | 4 | — | 是 | c6 | 0.033(n=30 全量,temperature 0.6) |
| 256 | 4 | — | 是 | c7, c8, c9 | 0.233(7/30);诊断出 16/30 撞 max_tokens |
| 273 | 4 | — | 是 | c10 | 0.100(--max-tokens 16000,对照 i=256 的 15000) |
| 326 | 4 | — | 是 | c12, c13, c14 | 0.200(6/30);仍有 16/30 撞 max_tokens |
| 355 | 4 | — | 是 | c7, c8, c15, c20 | 0.167 —— v2 第二次全量。分数由 i=373 直接 Read eval_v2_confirm.json 取回(… |
| 441 | 4 | — | 是 | c16, c17, c18 | 0.167(5/30)—— v4 第一次全量,分数经后台任务 b40d9b2fr 的 TaskOutput 在 i=45… |
| 454 | 4 | — | 否 |  | 未拿到。这次启动本身作废:同一条命令里 `rm -rf final_model` 把 shell 的 cwd 删掉了,后… |
| 463 | 4 | — | 是 | c7, c8, c20 | 0.133(4/30)—— v2 第三次全量,分数经后台任务 borb3iw22 在 i=473 连同 `cat eva… |
| 476 | 4 | — | 否 |  | 未拿到,且确实不存在。与 i=454 同一形态(cwd 被自己删掉 → ln -s 全失败 → 对空目录起 evalua… |
| 480 | 4 | — | 否 | c16, c17, c20 | 0.267(8/30)—— **机械层没关联到,但 agent 确实拿到了**。取回通道:i=482 起了一条 `sle… |
| 493 | 4 | — | 否 | c16, c17, c19, c20 | 0.233(7/30)—— 同样机械层没关联到而 agent 拿到了。通道:i=495 的后台任务 bvh6b4ngq … |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[101]
- **4 次验证没有拿到信号**:i=[454, 476, 480, 493]
- **分类学缺口提案 1 条**
  - verifier-noise-budgeting(i=510, i=508, i=475, i=352)
- **定义缺陷 5 条**
  - `_out_dir()` 只从**启动命令行**上抓 `--output-dir`,而本 run 的 agent 把输出目录写死在自己写的 train_sft*.py 里,命令行只有 `nohup python train_sft.py > train.log 2>&1 &`。于是 5 个 span 的 out_dir 全是 None,supersede / kill / discard / co…(i=107, i=120, i=243, i=429)
  - §4.1 写:「长训练用 nohup … & 启动、输出重定向进 logs/*.log,摘要行只有在 agent 事后 cat 整个文件时才进入轨迹,而 agent 基本不这么做」。本 run 是反例:**4/4 次后台训练的 train_runtime 摘要行都进了轨迹**,因为 agent 用 `tail -3 train_v2.log \| tr ... \| tail -10` 之类的轮询…(i=203, i=243, i=319, i=312)
  - 冒烟的判据是产物目录名含 smoke/sanity/debug/dryrun/tiny,或 `--max-samples ≤ 5000`,且明确排除 `--max-steps`。本 run 的冒烟叫 `test_train.py`、输出到 `/tmp/test_trl`、用 `max_steps=2` 和 20 行数据,三条判据一条都不沾,于是被记成 **real 训练**。后果:这条 run 的…(i=98, i=99, i=99, i=102)
  - i=454 的评测在启动 **7 秒后被 agent 亲手 kill**:同一条命令里 `rm -rf /home/ben/task/final_model` 把 shell 自己的 cwd 删掉了,后续 13 条 `ln -s` 全部 `No such file or directory`,evaluate.py 是对着空目录起的。它不可能产出分数,而骨架给它记了一个(0.167;当前抽取器则记…(i=455, i=457, i=457, i=464, i=478)
  - 22 行里有三类问题。(一)**3 行是幻影**:i=73 / 295 / 384 分别是 `python prepare_data.py` / `prepare_v3.py` / `prepare_v4.py`,被 `_FINALIZER` 的 `prepare` 分支认成 finalizer,但这三个脚本只写 sft_data*.jsonl,全程不碰 generation_config.jso…(i=73, i=69, i=141, i=137)
- **边界情形 4 条**
  - cross-run-candidate-selection-vs-C5(i=338, i=478, i=365, i=510)
  - smoke-has-no-tested-variable(i=98, i=99, i=102)
  - data-change-forces-hyperparam-change(i=289, i=298, i=288, i=298)
  - single-field-C1-ablation-drowned-in-noise(i=256, i=273, i=271, i=508)

## claude_non_api_max_claude-opus-4-8_10h_run1__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17315919
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 9.16h | 0.1666666666… |

### 改动序列(17 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 147 | C2 | 训练格式与评测逐字对齐:复用 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE,用 templates/smollm.jinja 渲染,靠 return_assistant_tokens_mask 只在 assistant token 上算 loss;并在训… | i=147, i=150, i=146 |
| 152 | C1 | 把 eos_token_id 设成 [128012 (<\|im_end\|>), 128001],并给 generation_config.json 补 pad_token_id=128004,让 vLLM 在 chat 模板的回合终止符上停;这是 prep_ckpt.py 对每个被评测/被提交的… | i=152, i=836, i=1143 |
| 169 | C3 | 建 SFT 数据集:open-r1/OpenR1-Math-220k,只留答案为 [0,999] 整数、且有至少一条 correctness_math_verify 通过且 reasoning_complete 的 R1 轨迹,每题取最短的那条正确轨迹;对 AIME2025 做全文+前60字符去重(… | i=169, i=178, i=184 |
| 199 | C4 | 训练方法:全参 bf16 SFT + FlashAttention-2 padding-free —— 自写 FFD 装箱把样本打包进 16384 token 的块(实测 99.5% 填充率),用 DataCollatorWithFlattening 给每个文档单独的 position_ids/cu… | i=199, i=208, i=214 |
| 228 | C4 | Run A 超参:全参 SFT、2 epoch、lr 1e-5、cosine、grad_accum 8(约 131k token/优化步)、save_only_model 每 200 步存一次。 | i=228, i=231 |
| 308 | C3 | 数据来源升级候选:下载 nvidia/OpenMathReasoning(cot)6 个 shard,建 data_omr_tokenized(39,406 例)与 data_combined(25,159 OpenR1 + 32,457 OMR = 57,616 例)。最终从未用于任何一次训练。 | i=308, i=374 |
| 422 | proposed:eval_protocol_n… | 写 eval_avg.sh:对同一份权重把**官方评测重复跑 N 次并按均值选型**,以对抗 temperature=1.0 下 30 题 pass@1 的采样噪声。这不改模型/数据/超参,只改「怎么取得判据」。 | i=422, i=911 |
| 431 | C5 | 在第 ~200/1828 步主动 pkill Run A,改为评测中途的 checkpoint-200,把「端到端推理管线对不对」这个风险提前 3.7 小时暴露(而不是等训练跑完)。 | i=426, i=431 |
| 476 | C3 | Run M 数据侧改动:把 R1 池子的长度上限从 12288 收到 10240,并抽样到 20,000 例(25,159 → 20,000),意图是用更短的训练分布压住 rambling。 | i=475, i=476, i=479 |
| 558 | C7 | 写 analyze_eval.py:从官方 inspect_ai 日志里额外算出**截断率**(是否闭合 </think> 并产出 text 部分)与输出长度分布,与 accuracy 并排报告。此后每一次训练决策都以截断率而非分数为主判据。 | i=558, i=467 |
| 669 | C5 | checkpoint 扫描:把 sft_runM 的 checkpoint-250 与 checkpoint-500 各评一次,与训练终点对比,找「跑满是不是更差」。 | i=669, i=723 |
| 696 | C3 | 建 data_concise:把 R1 长 CoT 整个换成 OpenR1 自带的简短人写参考解(中位 677 token),照样包进 <think>…</think> + ANSWER,用来检验「rambling 来自 R1 的 verify/reconsider 文体而不是长度」这个假设。32,… | i=696, i=712, i=704 |
| 770 | C3 | 建 data_capped(最终交付所依赖的改动):reasoning ≤5000 token 的轨迹原样保留(18,124 条);超过的**在 5000 token 处截断 reasoning,追加一句强制收尾语再接正确答案**(8,649 条 = 32%)。目的是教模型在预算内 commit,而… | i=766, i=796 |
| 835 | C5 | 把 sft_runM/checkpoint-500(训练中途的 checkpoint,不是训练终点)复制成 final_model 作为保底交付;选它是因为 i=669 的扫描显示 ckpt-500(0.1333/43% 截断)优于 ckpt-250(0.1000/50%)和训练终点 sft_run… | i=834, i=835 |
| 876 | C5 | 最终选型:用 sft_capped(训练终点)覆盖 final_model,放弃 R1-500。依据是多次评测的均值相当(capped 11.7% n=6 vs R1-500 13.3% n=1)但 capped 的截断率 3–17% vs R1-500 的 43%,且 R1-500 评测本身慢到对… | i=876, i=1091, i=1198 |
| 980 | C4 | 训练方法换成 RL:在 sft_capped 之上跑 GRPO,奖励是自写的、与官方 aime_scorer 同口径的规则奖励(取最后一个数值 token 比对),vLLM colocate 生成 + paged_adamw_8bit。三次冒烟全部失败,从未产出权重。 | i=980, i=1012 |
| 986 | C1 | 把整份 templates/smollm.jinja 作为 chat_template 写进 sft_capped/tokenizer_config.json **和 final_model/tokenizer_config.json**(为了让 GRPO 里 vLLM 自带的 tokenizer … | i=986, i=986 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 199 | smoke | 0.07h | superseded | **C4** | baseline —— 第一次启动 train_sft.py。20 步冒烟(--subset 3000 --max_steps 20 --grad_accum 4),目的是量 FFD+FA2 padding-free 管线的吞吐(实测 ~14k token/s、4.68 s/step)并确认不 OO… |
| 211 | smoke | 0.01h | returned | **C4** | vs i=199:--subset 3000→2500、--max_steps 20→8、--grad_accum 4→1,并加 --no_ckpt。唯一被测的变量是**关掉 gradient checkpointing 能否换来吞吐**。结局:第 0 步就 OOM(峰值 80,855 MiB / … |
| 228 | real | 0.59h | killed | **both** | baseline(第一次真实训练)。同一次启动里**同时**固定了数据侧(OpenR1 shortest-correct、≤12288 token、25,159 例、119M token)和方法/超参侧(全参 SFT、2 epoch、lr 1e-5、grad_accum 8、cosine),没有任何… |
| 476 | real | 7.82h | run_end | **both** | vs runA:--max_len 12288→10240、新增 --subset 20000(25,159→20,000 例)、--save_steps 200→250;lr / epochs / grad_accum / 数据来源逐字相同。agent 明写的意图有两条,分属两类:(a) 用更短的… |
| 713 | real | 0.03h | superseded | **C3** | vs runM:--data data_sft_tokenized→data_concise(R1 长 CoT → OpenR1 简短参考解,中位长度 4222→677 token)、--max_len 10240→4096、--epochs 2→3、去掉 --subset。agent 声明的假设只… |
| 726 | real | 4.18h | run_end | **C3** | 与 i=713 的训练命令逐字相同(--data data_concise --max_len 4096 --epochs 3 --lr 1e-5 --grad_accum 8 --save_steps 200),只是在 i=723 确认 GPU 已空(procs train=0 eval=0)后重… |
| 805 | real | 2.74h | run_end | **C3** | vs sft_concise:--data data_concise→data_capped(同一批 OpenR1 shortest-correct 轨迹,但 reasoning >5000 token 的被截断并追加强制收尾句 + 正确答案,占 32%)、--max_len 4096→8192、-… |
| 980 | smoke | 0.03h | discarded | **C4** | 第一次 RL 冒烟:训练方法从 SFT 换成 GRPO(num_gen 8 / prompt_bs 8 / grad_accum 2 / max_completion 8192 / lr 1e-6 / beta 0 / max_steps 3 / subset 200),起点是 sft_capped… |
| 988 | smoke | 0.05h | superseded | **C4** | 与 i=980 的 GRPO 参数逐字相同;唯一变化在模型目录 —— i=986 把 smollm.jinja 写进了 sft_capped/tokenizer_config.json。结局:过了模板这关,但在把 [batch×seq×128256] logits 转 fp32 时 OOM(单次要 … |
| 997 | smoke | 0.06h | discarded | **C4** | vs i=988:--prompt_bs 8→2、--grad_accum 2→4、--max_completion 8192→7168、--max_steps 3→2,纯粹为压 logits 显存。结局:跑完 1 步(51.5 s)后再次 OOM;而且那一步的诊断显示 completions/cl… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 153 | 4 | 30 | 是 |  | 0.03333333333333333 (1/30) —— base 模型原样的基线,不判定任何改动 |
| 440 | 4 | 30 | 是 | c1, c2, c3, c4, c5, c7 | 0.13333333333333333 (4/30);同一份日志经 i=466 的自写脚本读出 18/30 闭合、12/… |
| 655 | 4 | 30 | 是 | c10, c5 | 0.0333 (1/30);16/30 闭合、14/30(47%)截断、中位 15860 token —— 比只训 0.… |
| 669 | 4 | 30 | 是 | c16, c10 | checkpoint-250 = 0.1000 (3/30, 50% 截断);checkpoint-500 = 0.13… |
| 732 | 4 | 30 | 是 | c11 | 0.0000 (0/30);30/30 全部闭合、0% 截断、输出中位 730 token |
| 811 | 4 | 30 | 是 | c12 | 0.1333 (4/30);29/30 闭合、1/30(3%)截断、中位 9505 token |
| 876 | — | — | 是 | c12, c17 | 0.06666666666666667 (2/30)。注意这是**全量 30 题**、evaluate.py 默认参数(… |
| 912 | 3 | 4 | 是 | c8, c12, c17 | 只拿到第 1 次:0.13333333333333333。计划 4 次,agent 在 i=949/953 杀掉了整个比… |
| 912 | 3 | 3 | 是 | c8, c12, c17 | 只拿到第 1 次:0.13333333333333333。计划 4 次,agent 在 i=949/953 杀掉了整个比… |
| 912 | 3 | 4 | 是 | c8, c15 | 未拿到 —— `eval_avg.sh sft_runM/checkpoint-500 3 30 16 r1500_av… |
| 912 | 3 | 3 | 是 | c8, c15 | 未拿到 —— `eval_avg.sh sft_runM/checkpoint-500 3 30 16 r1500_av… |
| 1020 | 4 | 30 | 否 | c8, c12, c17 | 拿到了,但不在启动事件的输出里:cap_extra1 = 0.16666666666666666(i=1057 打印),… |
| 1020 | 4 | 30 | 否 | c8, c12, c17 | 拿到了,但不在启动事件的输出里:cap_extra1 = 0.16666666666666666(i=1057 打印),… |
| 1020 | 4 | 30 | 否 | c8, c15 | 未拿到 —— r1_extra1 跑到 10/30 时被 kill(i=1110 显示进程 19495 = evalua… |
| 1020 | 4 | 30 | 否 | c8, c15 | 未拿到 —— r1_extra1 跑到 10/30 时被 kill(i=1110 显示进程 19495 = evalua… |
| 1142 | 4 | — | 是 | c12, c17, c3 | 0.13333333333333333 (4/30);27/30 闭合、3/30(10%)截断、中位 10115 tok… |
| 1175 | 4 | 30 | 否 | c8, c15 | 未拿到 —— r1cmp1 跑到 8/30 被 kill(i=1196 「r1cmp1 prog: 8/30」,i=12… |
| 1217 | 4 | 30 | 是 | c12, c17 | 0.1 (3/30);25/30 闭合、5/30(17%)截断、中位 11218 token |

### 异常与存疑

- **5 次验证没有拿到信号**:i=[1020, 1020, 1020, 1020, 1175]
- **分类学缺口提案 1 条**
  - eval_protocol_noise_control(i=422, i=911, i=912, i=1172)
- **定义缺陷 6 条**
  - (i=655, i=647, i=665, i=764, i=859, i=871)
  - (i=876, i=896, i=912, i=923, i=440)
  - (i=946, i=953, i=1074, i=1075)
  - (i=199, i=211, i=214)
  - (i=313, i=836, i=909, i=986, i=100)
  - (i=469, i=665, i=764, i=871, i=1172)
- **边界情形 4 条**
  - (i=231, i=473, i=479)
  - (i=558, i=467, i=873)
  - (i=986, i=986, i=1172)
  - (i=1091, i=1198, i=1172)

## claude_non_api_max_claude-opus-4-8_10h_run2__aime2025_Qwen_Qwen3-1.7B-Base_17334259
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 8.52h | 0.0333333333… |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 103 | C3 | 数据来源与配方定型:写 data_prep.py,从 open-r1/OpenR1-Math-220k 每题取「最短的、math_verify 判为正确」的 R1 轨迹,只保留整数答案,token 窗口 200–12000,产出 31,723 条 SFT 样本。 | i=86, i=105, i=161 |
| 103 | C2 | 格式对齐:同一份 data_prep.py 里,user 侧逐字复制评测的 USER_PROMPT_TEMPLATE,assistant 侧在 R1 轨迹尾部补一行 `ANSWER: {整数}`,对齐 aime_scorer 的 numeric match 口径。 | i=28, i=86 |
| 109 | C2 | 自定义训练用 chat 模板(带 {% generation %} 标记),使训练前缀与评测端 qwen3.jinja + add_generation_prompt 产生的 `<\|im_start\|>assistant` 头逐字一致,并把 `<\|im_end\|>` 纳入 loss;当场做了… | i=109, i=112, i=114 |
| 150 | C4 | 训练方法定型:写 train_sft.py —— TRL SFTTrainer 全参微调,packing(bfd)+padding_free+flash_attention_2,assistant_only_loss,lr 1e-5 cosine,warmup 0.03,bs2/accum8,max… | i=137, i=162 |
| 195 | C4 | 加 --use_liger 开关并装 liger-kernel:用 fused linear cross-entropy 解掉 151k 词表 logits 的 OOM,顺带把 maxlen 16384→12288(数据最长 11999,不截断)。 | i=181, i=203, i=217 |
| 272 | C1 | 写 patch_model.py:把 config.json 与 generation_config.json 的 eos_token_id 设为 [151645, 151643](让 vLLM 在 <\|im_end\|> 停),并写入 temperature/top_p/top_k(--gree… | i=73, i=102, i=410, i=411 |
| 386 | C3 | 备选配方 v2:每题最多保留 2 条最短正确轨迹(data/sft_openr1_x2.jsonl,跑到 57,713 行被 kill),作为 Run2 的数据选项;最终没有被任何一次训练使用。 | i=385, i=386 |
| 414 | C1 | 从 generation_config 中删除 base 继承来的 `max_new_tokens: 2048`(以及 max_length),防止它成为 vLLM 的生成长度回退上限而截断长推理链。 | i=411, i=413, i=417 |
| 515 | C3 | 引入第二个数据源 nvidia/OpenMathReasoning(cot 切片):按 expected_answer 与最后一个 \boxed 的整数是否相等做正确性过滤,取 45,000 条全新题目。 | i=510, i=515 |
| 613 | C3 | 污染清洗:写 clean_contamination.py(子串 + 探针 + Jaccard>0.5)剔除与 AIME2025 重叠的训练题 —— OMR 丢 391 条(确有 AIME25 #7),OpenR1 丢 192 条(agent 自判为假阳性)。 | i=608, i=610, i=613, i=620 |
| 623 | C3 | Run2 配方成型:清洗后的 OpenR1(31,531)+ OMR(44,609)拼成 sft_mix_clean.jsonl 共 76,140 条,并复跑清洗器确认 0 残留。 | i=623, i=626 |
| 717 | C1 | 解码配置:在 temperature 0.6 之上加 repetition_penalty 1.1,目的是打断把输出撑到 16k token 的复读循环。 | i=710, i=717 |
| 759 | C5 | 交付产物选择:把 run1 的训练终点权重(不是任何中间 checkpoint)复制成 final_model 作为保底提交;run1 存了 checkpoint-512/768/1024/1280/1536/1538,一个都没单独评过。 | i=758, i=759 |
| 910 | C1 | 换成确定性解码:temperature 0.0(并 pop 掉 top_p/top_k)+ repetition_penalty 1.1,理由是评分方只跑一次,采样方差已造成 0/30 与 1/15 的分歧。 | i=909, i=910 |
| 958 | C1 | 在 greedy+rep1.1 之上加 frequency_penalty 0.4,针对 rep_penalty 打不断的「Wait, no. Wait...」变奏式循环。 | i=957, i=958 |
| 986 | C1 | 把同一份 greedy+rep1.1 解码配置施加到 run1,以便与 run2 在同口径下比较。 | i=985, i=986 |
| 1026 | C5 | 模型选择:在两次独立训练的成品之间选 run1(2 epoch × 31k OpenR1)而不是 run2(1 epoch × 48k 混合),依据是同口径 greedy+rep1.1 全量分 2/30 vs 1/30。注意这是跨训练的成品选择,不是 C5 定义的同一次训练内选步数(见 bounda… | i=1026 |
| 1027 | C1 | 把 final_model/generation_config.json 锁定为 temperature 0.0 + repetition_penalty 1.1(不带 frequency_penalty),并核对 final_model 与 run1 的权重字节数一致。 | i=1027, i=1030 |
| 1031 | C1 | 解码试探:repetition_penalty 1.1→1.15(只改这一项),想多打断几个循环。 | i=1030, i=1031 |
| 1098 | C1 | 解码试探:greedy+rep1.1 之上加一个很轻的 frequency_penalty 0.1(0.4 太重的折中版)。 | i=1097, i=1098 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 162 | smoke | 0.03h | killed | **C4** | baseline —— 本 run 第一次训练。1024 条样本 × 1 epoch,train_sft.py 默认值(bs2/accum8/maxlen16384/无 liger/开梯度检查点),目的是跑通管线并量步时。第 2 步 OOM。 |
| 203 | smoke | 0.03h | killed | **C4** | 相对 i=162:数据完全不变(同 1024 条),加 --use_liger 1、maxlen 16384→12288、显式 bs2/accum8。单一目的是验证 Liger 能否解掉 logits OOM 并重测吞吐。 |
| 226 | real | 0.04h | killed | **C4** | 相对 i=203:bs 2→8、accum 8→2(有效 batch 不变),数据从 1024 子样切到全量 31,723,epochs 1→3,save_strategy steps/250。agent 的陈述意图只有吞吐一项;数据源与配方(sft_openr1.jsonl)没变。 |
| 264 | smoke | 4.14h | killed | **C4** | 相对 i=226:--grad_ckpt 0(关梯度检查点),bs 8→4、accum 2→4,样本 1500。单变量意图明确:测关掉梯度检查点是否更快。结局是启动约 26 秒后 CUDA OOM(00:12:41 启动 → 00:13:07 已读到 traceback),不是骨架写的 4.14h。 |
| 274 | smoke | 0.09h | consumed | **unclear** | 相对 i=264:恢复 --grad_ckpt 1、bs8/accum2、2500 条样本、save_strategy epoch。但这次训练本身不在验数据配方也不在验超参 —— 它只是为了产出一个 checkpoint,好去验 C1(eos 能不能让 vLLM 停)与 C2(输出里有没有 ANSW… |
| 467 | real | 3.42h | consumed | **C4** | 相对被作废的 i=226:唯一实质变化是 epochs 3→2(因为 3 epoch 实测 2307 步 × 8.5s ≈ 5.4h,压不进预算),数据/lr/bs/accum/maxlen/liger/grad_ckpt 全同,save_steps 250→256。这是 run1 的正式训练,tr… |
| 793 | real | 0.01h | superseded | **unclear** | 这一行没有对应任何真实训练:复合命令的首条 pkill 没匹配到进程返回 1,整条命令中断,train_sft.py 从未执行(tool_result 只有 Exit code 1;紧接着的检查显示 train_run2.log 为空、GPU 0 MiB;agent 自己写下 Run2 didn't… |
| 802 | real | 0.06h | killed | **C3** | 相对 run1(i=467):数据从 sft_openr1.jsonl(31,723 × 2 epoch)换成 sft_mix_clean.jsonl 取 62,000(× 1 epoch);lr 1e-5 / bs8 / accum2 / maxlen12288 / liger / grad_ck… |
| 845 | real | 3.25h | consumed | **C3** | 相对 i=802:只把 max_samples 62000→48000、save_steps 400→350,纯粹为压进剩余墙钟;数据文件与全部超参不变。相对 run1 仍是同一个受测变量(OpenR1+OMR 混合 vs 纯 OpenR1)。这是 run2 的正式训练,1377 步 / 3:11:… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 117 | 3 | 8 | 是 |  | 0.000(limit 8;base 模型基线,同时用来验通评测管线) |
| 333 | 3 | 8 | 否 |  | 未拿到 —— 这条命令里的 evaluate.py 从未执行(见 definition_defect d4) |
| 416 | 3 | 8 | 是 | c2, c3, c6, c7 | 0.0(0/8);真正读回去的信息是格式与停止:6/8 stop=max_tokens、id=2 走完 ANSWER: … |
| 672 | 4 | 30 | 是 | c1, c2, c4, c6 | 0.03333333333333333(1/30);27/30 撞 max_tokens、29/30 缺 ANSWER |
| 717 | 3 | 15 | 是 | c12 | 0.06666666666666667(1/15);10/15 改为正常结束,循环被打断 |
| 751 | 4 | 30 | 是 | c12 | 0.0(0/30);16/30 缺 ANSWER —— 与同配置 limit 15 的 1/15 冲突,agent 据此… |
| 910 | 4 | 30 | 是 | c9, c10, c11, c14 | 0.03333333333333333(1/30);18/30 缺 ANSWER,贪婪比采样循环更狠 |
| 958 | 4 | 30 | 是 | c15 | 0.0(0/30);截断降到 11/30 但连原本对的 id=0 也做错,判定 freq 0.4 伤推理 |
| 986 | 4 | 30 | 是 | c16, c17 | 0.06666666666666667(2/30,id 0 与 id 16);同口径胜过 run2 的 1/30 |
| 1031 | 4 | 30 | 是 | c19 | 0.03333333333333333(1/30,id 13);missing_ANSWER 30/30,判定 rep … |
| 1063 | 4 | 30 | 是 | c13, c18 | 0.06666666666666667(2/30,id 0 与 id 16);对 final_model 产物本身的复现… |
| 1098 | 4 | 30 | 是 | c20 | 0.03333333333333333(1/30,只剩 id 0);判定任何 frequency_penalty 都伤推… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[274, 793]
- **1 次验证没有拿到信号**:i=[333]
- **分类学缺口提案 2 条**
  - contamination-guard(i=171, i=174, i=608, i=613, i=622)
  - budget-fit-resize(i=245, i=835)
- **定义缺陷 6 条**
  - 骨架把 i=793 记成 run2 的一次真实训练(background / 0.01h / superseded)。实际上这条复合命令的第一句 `pkill -9 -f "evaluate.py"` 没匹配到进程、返回 1,整条命令中断,train_sft.py 从未被执行:tool_result 只有 Exit code 1,随后的检查显示 train_run2.log 为空且 GPU 0 M…(i=793, i=795, i=799, i=801)
  - 骨架给 /tmp/smoke_nockpt(i=264)记了 4.14h、结局 killed。实际它在启动约 26 秒后就 CUDA OOM 了:启动 ts=00:12:41,00:13:07 的 tool_result 已经打出 torch.OutOfMemoryError 与 `0/38 [00:01<?, ?it/s]`,agent 当场写下 "No-ckpt OOMs"。4.14h 恰好落…(i=264, i=269, i=271, i=836)
  - 骨架给 /tmp/val_model(i=274)记的时长是 0.09h ≈ 327 秒(00:14:31 → 00:19:58),而该次训练器自报 `train_runtime: 512.7136` 秒 —— 比值 327/512.7 = 0.64 < 1.0,正是 §4.1 断言「必然不会出现」的物理上不可能的值。原因是配对把 i=333(ts 00:19:58)当成了产物被消费:那条命令确实…(i=333, i=330, i=407, i=407, i=409)
  - 骨架把 i=333 记成一次 limit 8 的评测(拿到分数=是、分数 0.0)。同 d3:那条命令是排队等待器,里面的 `evaluate.py --model-path /tmp/val_model --limit 8` 从未执行。agent 在 i=407 查到 GPU 0 MiB、无进程、val_metrics.json 不存在,在 i=409 判定「eval 没跑」,随后在 i=410…(i=333, i=407, i=409, i=416)
  - 骨架给 i=672 / 751 / 910 / 958 / 1031 / 1098 全部标了 0.066666…,但 agent 逐条读回的实际值是:i=672 → 0.03333333333333333(1/30)、i=751 → 0.0(0/30)、i=910 → 0.03333333333333333(1/30)、i=958 → 0.0(0/30)、i=1031 → 0.0333333333…(i=701, i=785, i=951, i=982, i=1060, i=1133, i=1150)
  - 第四档的开销被写成了档位的属性,实际上它是题量的函数。AIME2025 的全量测试集只有 30 题,本 run 里 8 次 `--limit 30`(i=672/751/910/958/986/1031/1063/1098,即全量、官方计分口径)全部在 5–8 分钟内返回了分数:i=751 启动 04:09:52 → 结果 04:15:34(5.7 分钟);i=986 启动 07:49:51 → …(i=751, i=785, i=1063, i=1095)
- **边界情形 3 条**
  - i=274 的 /tmp/val_model 训练:它的受测变量既不是数据来源也不是超参,而是「训好之后 eos 补丁与 ANSWER 格式能不能让评测正常收分」—— 也就是用一次训练去验 C1/C2。tested_variable 的四个取值(C3/C4/both/unclear)里没有能表达这件事的;标 unclear 会把它混进「证据不足」那一堆,而这里证据其实很充分。建议 schema 增…(i=271, i=446)
  - i=1026 的模型选择:在两次独立训练的成品(run1 vs run2)之间挑一个交。C5 的定义是「在已训好的若干步数里挑一个交」,即同一次训练内部选 checkpoint;跨训练的成品选择按现定义判不了,但它和 C5 一样是零训练成本的纯选择动作、效应量级相同(2/30 vs 1/30)。要么把 C5 的边界扩到「任何已存在权重之间的选择」,要么单列。顺带:本 run 存了 6 个 run1…(i=1026, i=665)
  - 建好但从未拿去训练的数据集算不算一次 change?c8(sft_openr1_x2,跑到 57,713 行)整个被放弃,c9/c10/c11 造出的 76,140 条混合集也只有 48,000 条进过训练。按 §1「为提升分数而做的一次有意图的修改」它们都算,但它们在 changes↔verifications 链接表里是永远没有入边的孤立点。如果算,C3 的计数会被大量未被任何验证器判定过的候…(i=386, i=640, i=845)

## claude_non_api_max_claude-fable-5_1m__10h_run1__bfcl_HuggingFaceTB_SmolLM3-3B-Base_17412199
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-fable-5 | claude-code | bfcl | HuggingFaceTB_SmolLM3-3B-Base | 6.91h | 0.96 |

### 改动序列(30 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 169 | C3 | 首次构建训练语料：从三个公开函数调用数据集（xlam-60k / ToolACE / hermes-fc）抽单调用样本，统一转成 BFCL tool schema，去重后 30,690 行 | i=169, i=199, i=199 |
| 196 | C3 | 修复 hermes 数据源的 tools 字段解析（从 <tools> 标签改为直接读 tools 列）后重建语料 | i=196, i=199 |
| 216 | C2 | 格式对齐：训练样本的 prompt 走与评测服务端完全相同的渲染链（create_tool_info_from_dict → openai_chat_tool_param → ChatCompletionToolsParam.model_dump → str() → smollm.jinja），re… | i=216, i=219, i=106 |
| 228 | C4 | 启动全参 SFT（2 epochs / lr 2e-5 / bf16 autocast / flash-attn / prompt-masked loss），并用 --save_each_epoch 同时落盘 epoch-1 与 epoch-2 两个 checkpoint（为后续 C5 选择预留候选… | i=228 |
| 234 | C4 | CUDA OOM 后重启：加 --grad_ckpt 与 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，数据与 lr/epochs 不变 | i=234, i=231 |
| 289 | C3 | 数值替换增广 augment_values.py：在 query 与 answer 中一致地重写数值参数，教精确值接地（首批 8,360 行） | i=289, i=290 |
| 295 | C3 | 修正增广的取值范围（避免生成 “February 60” 这类不合理值），重跑增广 | i=295, i=292 |
| 311 | C3 | 拼出 v2 语料（train_rows + aug_rows）并 tokenize 成 tokenized_v2。注：该语料**从未被训练**，被随后的 v3 取代 | i=311, i=320 |
| 355 | C3 | 接入第四个数据源 glaive-function-calling-v2，带严格 schema + groundedness（参数值必须在 query 里出现）过滤 | i=355, i=470 |
| 363 | C3 | 过滤掉 glaive query 里泄漏的 “ASSISTANT:”/“USER:” 多轮标记后重跑抽取，并对该批单独做污染检查 | i=363 |
| 423 | C1 | 解码/服务配置：finalize_model.py 把 checkpoint 转成可服务目录，**整份重写** generation_config.json 为固定 5 键 {_from_model_config, eos_token_id:[128012,128001], temperature:… | i=423, i=525, i=517, i=455 |
| 467 | C3 | v3 语料：把 data_utils 的 Callable 类型映射从丢弃改成 string，找回 102 行 lambda 字符串答案的训练行，并合入 glaive + 增广，固定 dev/train 划分；train 42,676 / dev 1,400 | i=467, i=470, i=470, i=442 |
| 510 | C5 | checkpoint 选择（第 1 次）：v1_e1 与 v1_e2 全量评测同为 0.92，选 epoch-2 的 v1_e2 入 final_model | i=510, i=478, i=480 |
| 522 | proposed:pipeline-repair | 修复提交流水线缺陷（第 1 次）：transformers 4.57 把 config.json 的 torch_dtype 改名为 dtype 导致 dtype 探测失效、走错分支；且 save_pretrained 的 GenerationConfig 校验拒绝 temperature=0.0 … | i=522, i=513, i=515 |
| 562 | C5 | checkpoint 选择（第 2 次）：v3_e1=0.93 高于 v3_e2=0.92 与 v1 的 0.92，把 v3_e1 换入 final_model | i=562, i=555, i=557 |
| 572 | C2 | 答案格式对齐：把训练行里的 lambda 字符串归一化为 BFCL gold 的紧凑算子间距（3 * x ** 2 → 3*x**2，保留 +/- 两侧空格） | i=572, i=560 |
| 572 | C3 | v4 语料：列表替换增广（1,805 行，练长列表的忠实拷贝与百分比→小数转换）+ 降低 lambda 行的上采样倍率，train 44,294 行 | i=572, i=575, i=575 |
| 616 | C6 | 权重平均：soup.py 把 models/v4_e2 与 models/v3_e1 等权平均。此次启动因 soup.py 的 save_pretrained 校验失败 + FileExistsError 而未完成 | i=616, i=619, i=619 |
| 626 | proposed:pipeline-repair | 修复提交流水线缺陷（第 2 次）：soup.py 与 finalize_model.py 中同一个 GenerationConfig 保存校验问题 + src==dst 时的 copytree 冲突，改完重跑 | i=626, i=621 |
| 626 | C6 | 权重平均重跑成功：soup(v4_e2, v3_e1) 全量评测 0.95，高于任何单一 checkpoint（最高 0.93） | i=626, i=629, i=629 |
| 632 | C5 | checkpoint 选择（第 3 次）：把 soup_34（0.95）换入 final_model | i=632, i=635 |
| 638 | C6 | 三路等权 soup(v4_e2, v3_e1, v1_e2) → 0.95，与两路 soup 错题集逐项相同，未採用 | i=638, i=641, i=645 |
| 648 | C7 | 自建代理验证器 gen_and_score.py：离线贪婪生成 + exact-match 打分，在 1,400 行 dev 上读到 0.8650、后续在 10k 训练切片上 0.9440，用于错误挖掘而非排名 | i=648, i=651, i=661 |
| 664 | C3 | 构造 DPO 偏好对：从 749 条挖到的错误里用 groundedness 过滤（gold 参数值必须在 query 里出现或是合法 enum/bool）筛出 250 对，并对其做污染检查 | i=664, i=667 |
| 668 | C4 | 训练方法切换：在 soup_34 权重上做 DPO（lr 5e-7, beta 0.1, 2 epochs，250 对）。训练完成但保存失败，rewards/accuracies 仅 0.539 | i=668, i=671, i=675 |
| 680 | C4 | DPO 超参重试：lr 5e-7→2e-6、epochs 2→3，其余（beta 0.1、250 对、起点权重 soup_34）不变；rewards/accuracies 0.539→0.753 | i=680, i=683 |
| 696 | C3 | v5 语料：新增等差数列（AP）长列表增广 1,600 行，专门针对 calculate_mean 那一道 30 项规律序列拷贝错误；v5 共 45,873 行 | i=696, i=699, i=699 |
| 769 | C6 | 等权 soup(v5_e2, v4_e2, v3_e1) → 0.94，低于 soup_34 的 0.95，被判定为 v5 稀释了混合 | i=769, i=772, i=774 |
| 775 | C6 | **加权** soup：把 v4_e2 列两次、再加 v3_e1 与 v1_e2（即 2:1:1 权重）→ 0.96，是整条 run 的最高分与最终提交 | i=775, i=778, i=778 |
| 785 | C5 | checkpoint 选择（第 4 次，终局）：把 soup_w 固定为 final_model，并逐字核对 5 键 generation_config、config.json eos=128012、tokenizer eos=<\|im_end\|>、dtype=bfloat16 | i=785, i=788, i=788 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 228 | real | 0.03h | superseded | **both** | baseline；首次训练，数据（tokenized_v1 = 30,690 行 xlam+ToolACE+hermes 单调用）与方法（全参 SFT、2 epochs、lr 2e-5、prompt-masked）同时确定，不隔离任何单一变量。实际结局不是单纯的 superseded：启动后约 2 … |
| 234 | real | 1.26h | last_seen | **both** | 与 i=228 相比：数据（tokenized_v1）、lr、epochs、--save_each_epoch 逐字相同，只多了 --grad_ckpt 和 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，目的是绕过 OOM 而非验证某个变量。它是实… |
| 491 | real | 1.01h | last_seen | **C3** | 与 i=234 相比：train_sft.py 命令行除 --data 外逐字相同（--epochs 2 --lr 2e-5 --save_each_epoch --grad_ckpt），只把 tokenized_v1 换成 tokenized_v3：语料从 30,690 行增到 42,676 行（… |
| 581 | real | 1.48h | last_seen | **C3** | 与 i=491 相比：命令行除 --data 外逐字相同，只把 tokenized_v3 换成 tokenized_v4：在 v3 配方基础上加 lambda 紧凑风格归一化、列表替换增广 1,805 行、降低 lambda 上采样（train 42,676 → 44,294 行）。真实结局：跑到自… |
| 668 | real | 0.03h | returned | **C4** | 与之前全部 SFT 相比：换算法（SFT → DPO）、换起点权重（base → models/soup_34）、换目标（250 对挖到的错误偏好对）；lr 5e-7、beta 0.1、2 epochs。真实结局：前台运行完成（train_runtime 100.4s）但 **保存失败** —— s… |
| 680 | real | 0.05h | returned | **C4** | 与 i=668 相比：同一份 250 对、同一个起点权重 models/soup_34、同一个 beta 0.1，**只改两个超参**：lr 5e-7→2e-6、epochs 2→3（另加了保存前重置采样字段的修复）。真实结局：完成并成功落盘 ckpt_dpo/final，rewards/accur… |
| 700 | real | 1.04h | last_seen | **C3** | 与 i=581 相比：train_sft.py 命令行除 --data / --out 外逐字相同，只把 tokenized_v4 换成 tokenized_v5：在 v4 基础上加 1,600 行等差数列长列表增广（44,294 → 45,873 行）。注：本事件同时包含污染检查与 tokeniz… |

### 验证序列(16 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 56 | 3 | 10 | 是 |  | 0.000 — base 模型基线，--limit 10（真正的第三档）。作用是验管线通不通 + 拿到零点锚点，不判定任… |
| 427 | — | — | 是 | c1, c2, c3, c4, c5, c11 | 0.92 — 全量 100 题（run_eval.sh 不传第三个参数 ⇒ --limit -1 ⇒ evaluate.… |
| 460 | — | — | 是 | c14 | 0.92 — 后台全量评测 epoch-1 checkpoint，分数在 i=478 随 tokenize 输出一起读回… |
| 552 | — | — | — | c12, c17 | v3_e2 = 0.92、v3_e1 = 0.93，两次全量评测。**这一行不在骨架的评测表里**：两次 run_eva… |
| 606 | — | — | 是 | c15, c16 | v4_e2 = 0.92（全量 100 题）。判定 v4 语料（列表增广 + lambda 风格归一化）：lambda … |
| 606 | — | — | 是 | c15, c16 | v4_e2 = 0.92（全量 100 题）。判定 v4 语料（列表增广 + lambda 风格归一化）：lambda … |
| 606 | — | — | 是 | c15, c16 | v4_e1 = 0.90（全量 100 题）。同一条命令里的第二次 run_eval.sh，判定 v4 的 epoch-… |
| 606 | — | — | 是 | c15, c16 | v4_e1 = 0.90（全量 100 题）。同一条命令里的第二次 run_eval.sh，判定 v4 的 epoch-… |
| 616 | — | — | 是 | c18 | **未拿到分数**。soup.py 在 save_pretrained 处被 GenerationConfig 校验拦下… |
| 626 | — | — | 是 | c19, c20 | 0.95（全量 100 题）—— 本条 run 里首次超过任何单一 checkpoint（最高 0.93）。判定的是权重… |
| 638 | — | — | 是 | c22 | 0.95（全量 100 题）。判定三路 soup 是否优于两路；错题 id 与 soup_34 逐项相同，判为无增益 |
| 648 | — | — | — | c20, c23 | 自建代理验证器（C7），非官方口径：1,400 行 dev 上 exact-match 0.8650，产出 189 条错… |
| 658 | — | — | — | c20, c23 | 代理验证器第二次：10,000 行训练切片上 exact-match 0.9440，产出 560 条错误；与 dev 的… |
| 684 | — | — | 是 | c25, c26 | 0.95（全量 100 题）—— 与 DPO 前的 soup_34 逐项同分、错题 id 完全相同，DPO 被判为 ev… |
| 759 | — | — | 是 | c27 | 0.93（全量 100 题）。判定 v5 的 AP 列表增广：目标题（id 15 calculate_mean）**没修… |
| 769 | — | — | 是 | c28 | 0.94（全量 100 题），低于 soup_34 的 0.95，v5 被判定为稀释了混合 |
| 775 | — | — | 是 | c29 | 0.96（全量 100 题）—— 整条 run 的最高分。判定加权 soup；calculate_mean 那道因 v1… |
| 789 | 4 | — | 是 | c29, c30 | 0.96 — 用官方默认调用（python evaluate.py，无任何 --limit / --model-path… |

### 异常与存疑

- **3 次验证没有拿到信号**:i=[552, 648, 658]
- **分类学缺口提案 1 条**
  - pipeline-repair(i=513, i=515, i=619, i=621, i=671, i=673)
- **定义缺陷 8 条**
  - (i=427, i=428, i=7, i=32, i=790)
  - (i=552, i=555, i=557, i=562)
  - (i=616, i=619, i=619, i=621, i=629)
  - (i=749, i=750, i=759, i=539, i=547, i=552)
  - (i=169, i=467, i=513, i=381)
  - (i=555, i=629, i=778, i=792, i=796, i=796, i=780)
  - (i=775, i=774, i=638)
  - (i=228, i=491, i=231, i=56)
- **边界情形 6 条**
  - C2 与 C4 在同一个产物里分不开：tokenize_data.py 同时实现了“与评测逐字节对齐的渲染”（C2）与“prompt 段 label 置 -100 的损失掩码”（C4 训练方法）。两者写在同一次写入、同一条命令里执行、产出同一份 tokenized_v1，没有任何事件可以把它们分开归类。现定义下只能二选一，本次选了 C2。(i=216, i=219, i=219)
  - C2 与 C3 的边界：把 data_utils 的 Callable 类型从丢弃改成映射为 string（i=462）。意图是纯粹的**答案格式对齐**（让模型学会吐 `lambda x: 3*x**2` 而不是原式数学记号），机制却是**改变语料里有哪些行**（找回 102 行）。更要紧的是它破了 C2 “可用最低档 = 第一档、分钟级、零 GPU、确定性” 这一条：改完之后必须重跑一次 1.…(i=442, i=467, i=470, i=560, i=491)
  - 机械层把 i=700 的产物记为 `tokenized_v5`，但该事件是一条 `污染检查 ; tokenize && nohup train_sft &` 的复合链，真正的训练产物是 `ckpt_v5`，且训练在命令发出后约 2 分 12 秒（tokenize 跑完）才开始。按“启动时刻”还是“首个训练 step”计起点，现有定义没说；两者差 0.037h。定义不变也能用，但归属字段会误导下游聚…(i=700, i=730, i=738)
  - i=234 这类“崩溃后重启”的训练，`tested_variable` 没有合适取值。它与上一次的唯一差异是两个 C4 旋钮（--grad_ckpt 与显存分配器），按字面应该写 C4；但这两个旋钮不是被“验证”的对象，它们只是让盒子不 OOM，而这次训练实际验的是整套 v1 配方（both）。本次选了 both。建议 schema 加一个 `restart_of` 字段或 `tested_va…(i=228, i=234, i=231, i=233)
  - i=234 的时长口径存疑：骨架记 1.26h，而 trainer 自报 train_runtime=3790.9s（1.053h），且产物在 23:43:40（启动后 1.03h）就被 finalize 消费了。比值 1.26/1.053 = 1.20，落在 §4.1 后台校准表的四分位 1.04–1.14 之外。我无法从事件流定位它配到了哪个“消费”事件，所以只登记不下结论。(i=234, i=420, i=420, i=423)
  - C7 与“不算验证器”的边界：analyze_log.py（i=256 写入，全转调用 7 次）不自己打分，只把官方 inspect 日志里的错题拆成 WRONG ARGS / NO CALL / MULTI 类别。它不产生新分数（非 C7），但它是本 run 里每一次改动决策的直接依据，成本是秒级、零 GPU、确定性——完全符合第一档的成本剖面却不在第一档的四项枚举里。建议第一档增一项“对已有评…(i=433, i=434, i=558)

## claude_non_api_max_claude-fable-5_1m__10h_run1__bfcl_Qwen_Qwen3-1.7B-Base_17415253
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-fable-5 | claude-code | bfcl | Qwen_Qwen3-1.7B-Base | 6.72h | 0.96 |

### 改动序列(34 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 109 | C3 | 选定 lockon/xlam-function-calling-60k(未 gate 的镜像)作为主数据源,写 prep_data.py 只保留单次调用样本并按 schema 校验,产出 25,836 条。 | i=106, i=111 |
| 142 | C2 | 训练侧工具 schema 补上 additionalProperties: false,使 prep_data 渲染出的 tools JSON 与 inspect-ai 在评测时序列化的逐字节一致。 | i=141 |
| 162 | C2 | 绕开 apply_chat_template 渲染 assistant 轮次(qwen3 模板会插入空 <think>),改为只渲染 user + add_generation_prompt,再手工拼 <tool_call>...</tool_call><\|im_end\|> 作为监督目标。 | i=156 |
| 167 | C1 | 写 finalize_model.py,给每个待评 checkpoint 整份重写 generation_config.json:temperature 0.0(贪婪)+ eos_token_id 双值 [151645, 151643],让 vLLM 在 <\|im_end\|> 停下。 | i=163, i=290 |
| 196 | proposed:infra-repair | v1 首次启动 OOM 后改 train_sft.py:bs 16->8、ga 4->8(有效 batch 64 不变)、开 gradient checkpointing、加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments。目的是把 boxes 塞进显存,不是… | i=194, i=197 |
| 240 | C7 | 自建 eval_local.py:在 575 条 held-out xlam 验证集上做 exact-match 打分,作为官方全量评测之外的廉价代理验证器,全程与官方分数并列使用。 | i=238 |
| 260 | C3 | 写 mine_hard.py:用当前 checkpoint 对 20,000 条训练样本采样,把答错的挑出来做 hard 集与 DPO 负例(自生成 + 验证过滤,即 §C3 的来源 (d))。 | i=258 |
| 276 | C3 | 写 clean_numeric.py,剔除 gold 参数里出现了查询中无法推出的数值(标注者臆造的经纬度等)的样本,四个文件共丢掉约 1,200 条。 | i=270, i=277 |
| 296 | C1 | 把 finalize_model.py 写出的 generation_config 从 8 字段裁到 6 字段:删掉 top_p:1.0 / top_k:0 / repetition_penalty:1.0(top_k:0 会触发 vLLM SamplingParams 校验)。最终落盘内容见 i=… | i=295 |
| 321 | C3 | 构建 v2_mix(38,247 条):xlam+toolace+hermes 清洗后合并,并做单工具增广——把多工具样本裁成只带 gold 工具的一份副本,使约 48% 的训练样本工具数为 1,对齐评测里永远只有 1 个工具的形态。 | i=320, i=322 |
| 326 | C4 | 写 train_dpo.py,把 DPO(带 sft-alpha 混合 NLL 项)加为 SFT 之外的第二种训练方法。 | i=324 |
| 358 | C2 | canonicalization:把 gold answer 里取值恰好等于 schema default 的可选参数删掉(占全部参数 7.5%),使训练目标与官方 scorer 的严格等值约定一致;3,058 条被改写。 | i=357, i=359 |
| 380 | C6 | 写 average_models.py(权重平均 / model soup),作为不需要训练的额外候选生成手段。 | i=378 |
| 443 | C3 | 写 build_canon_pairs.py:纯从训练数据构造 DPO 偏好对,chosen 为删掉 default-equal 可选参数的答案,rejected 为保留它们的原答案。 | i=442 |
| 460 | C3 | 把挖掘得到的 dpo_pairs_v2 与 canon_pairs 去重合并成 data/dpo_all.jsonl,共 6,433 对。 | i=457, i=460 |
| 473 | proposed:infra-repair | DPO 首次启动 OOM(全词表 fp32 log_softmax)后改 train_dpo.py:bs 4->2、ga 8->16,有效 batch 32 不变。 | i=471, i=474 |
| 495 | C3 | 构建 v3b polish 混料(13,132 条):hard_v2 复制两份 + 从 v2_mix 随机取 6,000 条,让难例占比大幅上升。 | i=495, i=496 |
| 504 | C4 | v3b 抛光 SFT(写在 chain_v3.sh 里):从 runs/v2/checkpoint-1196 续训 1 epoch,lr 降到 3e-6 并加 warmup-ratio 0.05,而不是从 base 重训。 | i=501, i=504 |
| 531 | C4 | 第二轮 DPO 只用 canon_pairs,并把 beta 0.1->0.15、lr 7e-7->1.2e-6、sft-alpha 0.2->0.1,专门去打两条 calculate_investment_value 题。 | i=530, i=531 |
| 559 | C3 | 构建 v4_mix(41,813 条)= v2_mix + 再加一份 hard_v2,使难例在混料中出现两次。 | i=559, i=560 |
| 573 | C3 | 新增第四个数据源 argilla/Synth-APIGen-v0.1(python 工具风格,更贴近 exec_simple 域),写 prep_synth.py 转换。 | i=571 |
| 587 | proposed:infra-repair | prep_synth.py 首次跑出 written: 0(用了 xlam 的扁平参数解析器),改用 hermes 的 OpenAI-schema normalizer 后得到 17,330 条 / 清洗后 16,856 条。 | i=585, i=591 |
| 596 | C3 | 构建 v5_mix(64,606 条)= v2_mix 的 canonicalized 底料 + 22,793 行 Synth-APIGen(含单工具增广),用 random.Random(11) 打乱。 | i=595, i=596, i=597 |
| 606 | C5 | 把当时按本地验证集最好的 v2_dpo 用 finalize_model.py 装成 final_model 占位提交,以保证后续任何失败都仍留有可用产物。 | i=605, i=606 |
| 643 | C5 | final_model 换成 v4_ep2(官方 0.96),并当场打印其 generation_config 确认贪婪配置仍在。 | i=643, i=644 |
| 655 | C6 | 在 CPU 上把 v4 的 checkpoint-653 与 checkpoint-1306 平均成 runs/v4_soup,作为零训练成本的额外候选。 | i=654, i=655 |
| 704 | C3 | 启动 chain_v6.sh:从当前最佳 checkpoint 重新挖掘 hard/wrong 生成,合成 dpo_all_v4 偏好对再做一轮 DPO(与 v2 那轮同超参,只换偏好数据来源)。 | i=703, i=704 |
| 736 | proposed:infra-repair | chain_v6 第一次因 runs/v4/checkpoint-1306 里没有 tokenizer 文件(save_only_model=True 的后果)在 mine_hard 处崩溃,改用已 finalize 的 eval_models/v4_ep2 目录重启同一条链。 | i=731, i=736 |
| 756 | proposed:infra-repair | DPO 跑完但在 save_pretrained 处被 transformers 严格校验拒绝(do_sample=False 同时 temperature=0.0),改 train_dpo.py 在保存前把 policy.generation_config 换成只含 bos/eos/pad 的干净… | i=754, i=753 |
| 780 | proposed:seed-redraw | 把 v4 的 SFT 命令逐字重跑一次,只把 --seed 43 换成 --seed 1234(runs/v6),目的是在评测噪声上再抽一次签而不是改配方。 | i=780, i=559 |
| 807 | C5 | final_model 换成 v4_dpo(bench 0.96 / 本地验证 0.8765,是 0.96 组里本地验证最高的),并立即用默认 evaluate.py 复核。 | i=807 |
| 817 | C3 | 第二轮挖掘:改用 eval_models/v4_dpo 自己去挖 20,000 条,合成 dpo_all_v4dpo 偏好对再做一轮 DPO(lr 7e-7->6e-7),产出 runs/v4_dpo2。 | i=817, i=817 |
| 839 | C5 | final_model 换成 v4_dpo2(bench 0.96,本地验证 0.8817,全程最高),这是最终提交。 | i=839, i=833 |
| 849 | proposed:seed-redraw | 最后一次种子彩票:v4 配方再跑一遍,--seed 777(runs/v7),事先声明只有严格优于 0.96 且本地验证不降才换。 | i=849, i=848 |

### 训练序列(12 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 169 | smoke | 0.01h | returned | **unclear** | baseline(冒烟)。128 条样本、1 epoch,只验 train_sft.py 能不能跑通与吞吐率,不验 C3 也不验 C4。见 boundary_case bc3。 |
| 175 | real | 0.04h | superseded | **both** | baseline(首次真实训练)。同时确立数据(prep_data.py 的 25,836 条 xlam 单调用)与方法(全参 SFT,2 epoch,bs 16 ga 4 lr 1e-5),不构成对照。约 1.5 分钟后 OOM 崩溃(大词表 logits @ bs16 × 2048 tok),被… |
| 197 | real | 0.47h | consumed | **C4** | 与 i=175 同数据、同 epochs、同 lr;只改 bs 16->8 与 ga 4->8(有效 batch 64 不变)、开 gradient checkpointing、加 expandable_segments。纯 OOM 修复,不为提分。结局:跑完,27 分钟,i=333 看门狗返回 c… |
| 364 | real | 1.42h | last_seen | **C3** | 与 i=197 的启动命令**逐字只差 --data data/v2_mix.jsonl**(epochs/bs/ga/lr/save-each-epoch 全同,train_sft.py 期间未改)。数据从 25,836 条原始 xlam 换成 38,247 条 v2_mix(三源合并 + 数值清… |
| 462 | real | 0.02h | superseded | **C4** | 训练方法从全参 SFT 换成 DPO(beta 0.1 / lr 7e-7 / sft-alpha 0.2),初始权重 runs/v2/checkpoint-1196。启动约 1 分钟后 OOM(全词表 fp32 log_softmax),被 i=474 取代。 |
| 474 | real | 0.61h | last_seen | **C4** | 与 i=462 同 pairs、同 init、同 beta/lr/epochs/sft-alpha;只改 bs 4->2、ga 8->16(有效 batch 32 不变)。纯 OOM 修复。结局:跑完,启动 21:00:01,i=499 看门狗于 21:10:57 返回 completed,i=50… |
| 531 | real | 0.10h | consumed | **both** | 与上一轮 DPO(i=474)相比同时动了两类:C3——偏好对从 dpo_all.jsonl(6,433 对,挖掘 + canon)换成只用 canon_pairs.jsonl;C4——beta 0.1->0.15、lr 7e-7->1.2e-6、sft-alpha 0.2->0.1,初始权重从 r… |
| 559 | real | 2.31h | consumed | **both** | 与 i=364(v2)相比:C3——data v2_mix -> v4_mix(41,813 条 = v2_mix + 再加一份 hard_v2,难例出现两次);同时首次显式加上 --seed 43(v2 未指定种子)。种子这一维按现有定义只能挂到 C4,但它不改变期望,见 proposed_cat… |
| 758 | real | 1.06h | consumed | **both** | 这是 chain_v6 那轮 DPO 的第三次尝试(前两次分别死于 checkpoint 缺 tokenizer 与 save_pretrained 校验)。与 i=474 的 DPO 相比:C3——偏好对换成 dpo_all_v4.jsonl(从 v4_ep2 重新挖掘 + canon 去重);C… |
| 780 | real | 1.94h | run_end | **C4** | 与 i=559(v4)的 SFT 调用**逐字只差 --seed 1234(原 43)与输出目录**:同 data v4_mix、同 epochs 2、bs 8、ga 8、lr 1e-5、save-each-epoch。纯种子重抽。结局**不是** run 结尾:i=804 于 00:53:30 返… |
| 817 | real | 1.12h | run_end | **both** | **该行的产物 data/hard_v4dpo.jsonl 是一次 mining(推理)输出,不是权重**;同一条 nohup 链里真正的训练是 --output runs/v4_dpo2,机械层没登记它(见 definition_defect dd3)。就那次训练而言,相对 i=758:C3——偏… |
| 849 | real | 0.77h | run_end | **C4** | 与 i=559(v4)逐字只差 --seed 777(原 43)与输出目录,data/epochs/bs/ga/lr 全同。第二次种子重抽。结局**不是** run 结尾:i=871 于 01:57:44 打印 saved to runs/v7/final,i=874 于 02:01:53 返回 c… |

### 验证序列(13 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 55 | 3 | 20 | 是 |  | 0.0 |
| 206 | — | — | 是 |  | 未拿到(该事件根本不是评测) |
| 336 | 4 | -1 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep2,全量 100 题) |
| 336 | 4 | -1 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep2,全量 100 题) |
| 336 | 4 | -1 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep1,全量 100 题) |
| 336 | 4 | -1 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep1,全量 100 题) |
| 547 | 4 | -1 | 否 | c17, c18, c19, c14 | 0.95(bench)/ 0.8713(本地 575 题) |
| 758 | 4 | -1 | 否 | c27, c28, c29, c7 | 0.96(bench)/ 0.8765(本地) |
| 780 | 4 | -1 | 否 | c30 | 0.94(v6_ep2) |
| 780 | 4 | -1 | 否 | c30 | 0.94(v6_ep2) |
| 780 | 4 | -1 | 否 | c30 | 0.94(v6_ep1) |
| 780 | 4 | -1 | 否 | c30 | 0.94(v6_ep1) |
| 807 | 4 | — | 否 | c31 | 0.96 |
| 817 | 4 | -1 | 否 | c32 | 0.96(bench)/ 0.8817(本地) |
| 839 | 4 | — | 否 | c33 | 0.96 |
| 849 | 4 | -1 | 否 | c34 | 0.95(v7_ep2) |
| 849 | 4 | -1 | 否 | c34 | 0.95(v7_ep2) |
| 849 | 4 | -1 | 否 | c34 | 0.94(v7_ep1) |
| 849 | 4 | -1 | 否 | c34 | 0.94(v7_ep1) |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[169]
- **17 次验证没有拿到信号**:i=[336, 336, 336, 336, 547, 758, 780, 780, 780, 780, 807, 817, 839, 849, 849, 849, 849]
- **分类学缺口提案 2 条**
  - seed-redraw(i=780, i=849, i=848, i=847)
  - infra-repair(i=194, i=471, i=754, i=731, i=585)
- **定义缺陷 7 条**
  - (i=206, i=209)
  - (i=504, i=629, i=704, i=697)
  - (i=817, i=817)
  - (i=366, i=406, i=407)
  - (i=804, i=871, i=832)
  - (i=111, i=229, i=588)
  - (i=194, i=197)
- **边界情形 6 条**
  - 单工具增广:把多工具训练样本裁成只带 gold 工具的副本,使约 48% 的样本工具数为 1。(i=320)
  - canonicalization:删掉 gold answer 里取值等于 schema default 的可选参数,以匹配官方 scorer 的严格等值约定。(i=357)
  - 冒烟训练在 tested_variable 的四值域 {C3, C4, both, unclear} 里没有合法取值。(i=169)
  - 首次真实训练(baseline)同样没有合法的 tested_variable 取值。(i=175)
  - 为绕过 transformers 保存校验而改写 policy 的 GenerationConfig。(i=754, i=644)
  - v4 相对 v2 同时换了混料(C3)并**首次加上** --seed 43(v2 未指定种子)。(i=559, i=364)

## claude_non_api_max_claude-opus-4-8_10h_run1__bfcl_Qwen_Qwen3-4B-Base_17409821
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | bfcl | Qwen_Qwen3-4B-Base | 9.06h | 0.95 |

### 改动序列(19 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 105 | C2 | 用评测自己的 templates/qwen3.jinja 渲染 xLAM 单调用样本的 prompt（tools + user + add_generation_prompt），completion 手工拼成 <tool_call>...</tool_call><\|im_end\|>，使训练样本与… | i=105, i=105 |
| 113 | C3 | 自写 gen_synthetic.py 生成合成计算类 function-calling 数据（物理/数学/金融/统计等），对齐 BFCL exec_simple 域，作为第二个数据来源。 | i=113, i=113 |
| 123 | C3 | 把 xlam_sft.jsonl + synthetic_sft.jsonl 合并成 combined_sft.jsonl，并用 contamination_check.py 对 decon_text 字段做污染校验（此后每个数据版本都重跑）。 | i=123, i=123 |
| 139 | C4 | 确定训练配方：train_sft.py 全参 SFT（非 LoRA）、bf16 + flash_attention_2 + paged_adamw_8bit、lr 1e-5 cosine、bs 16 / grad_accum 2 / max_len 1536、只在 completion 上算 los… | i=139, i=142 |
| 145 | proposed:verifier-yield-… | 自建 analyze_failures.py：直接读官方 inspect_ai 评测日志，把总分拆成“格式错 vs 语义错”并逐题列出 predicted vs target。它不替代官方评测（官方全量评测在 bfcl 上只要分钟级），而是从同一次官方评测里多榜出决策信号。 | i=145, i=331 |
| 252 | C3 | 把合成数据从初版扩到 20000 条（seed 1234，56 种 function），并另用 seed 9999 生成 500 条 held-out dev 集（该 dev 集在全 run 中从未被用来打分）。 | i=252, i=252, i=255 |
| 283 | C1 | 把 C1 修复固化成流水线不变量：写 pipeline.sh = train -> finalize_model.py -> evaluate.py，此后每一次训练产出都自动重新打 eos/greedy 补丁，避免 save_pretrained 每存一次就复发。 | i=283, i=283 |
| 322 | C1 | 对已训好的 models/smoke 就地打 eos/greedy 补丁（finalize_model.py）：config.json 的 eos_token_id 由 151643 改成 [151645, 151643]；generation_config.json 整份重写；tokenizer_… | i=322, i=168, i=570, i=672 |
| 380 | C3 | synthetic_v2（22000 条）：新增 integrate_polynomial / evaluate_expression / sort_numbers / analyze_integer / convert_currency 等边缘技能工厂，合成 combined_v3（50461 条… | i=380, i=386, i=387 |
| 445 | proposed:verifier-yield-… | 自建 compare_runs.py：对比两份官方评测日志，给出逐题的 FIXED / BROKEN 集合。它把“95 vs 94”这种小于噪声的总分差，换成了可归因的项目集差，后续每一次 C3 改动都是据此决定的。 | i=445, i=566, i=568 |
| 476 | C3 | synthetic_v4（24000 条）：加入科学记数法 / 负数 / 高精度 / 可选参数类技能，合成 combined_v4。 | i=476, i=476, i=485 |
| 498 | C4 | 对 CUDA OOM 的超参修复：开启 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，并把 pipeline.sh 的训练参数从 --bs 16 --grad_accum 2（max_len 默认 1536）改成 --bs 12 --grad_ac… | i=498, i=497, i=505, i=522 |
| 569 | C5 | 候选选择（第一次）：把 models/smoke 拷贝成 final_model 作为 0.95 的保底，并核验拷贝后的 config 仍带 eos 修复与 greedy。注：train_sft.py 用 save_strategy='no'，本 run 没有任何中间 checkpoint，所以选的… | i=569, i=569, i=570 |
| 571 | C4 | 确认 expandable_segments 已解决碎片化后，把 batch size 从 12 改回 16（grad_accum 3、max_len 1280 不变，有效 batch 48）。此后 vc / vd / ve 三条训练均用该配置。 | i=571, i=522 |
| 585 | C3 | synthetic_v5（24000 条）：随机化表达式的空格风格、新增逐字拷贝字符串的 register_item 工厂，目标是治理 4*x**3 被改写成 4 * x ** 3、rice bowl 被改成 rice bowls 这类不逐字拷贝的失败；合成 combined_v5（额外用 seed… | i=585, i=585, i=572 |
| 671 | C5 | 候选选择（第二次）：final_model 换成 models/vc（0.95，数据更丰富），并重跑一次 finalize_model.py 确认幂等。 | i=671, i=672 |
| 683 | C3 | synthetic_v6（25000 条）：新增 summarize_dataset / plan_route / build_playlist 等工厂，教“可选数据参数出现 ≠ 该设相关的可选布尔参数”；合成 combined_v6。 | i=683, i=683, i=674 |
| 729 | C5 | 候选选择（第三次，最终提交）：final_model 换成 models/vd（与 vc 同分、逐题失败集完全相同，选 vd 因为数据流水线最精细），随即对 final_model 目录做一次独立验证评测。 | i=729, i=729, i=728 |
| 751 | C4 | 最后一次实验：用与 vd 逐字相同的 combined_v6，只把 epochs 从 1 改成 2（bs 16 / grad_accum 3 / max_len 1280 不变）。这是本 run 唯一一次真正单变量的 C4 对照。 | i=751, i=689, i=790 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 139 | smoke | 0.26h | consumed | **unclear** | baseline（本 run 第一次训练）。这是冒烟跑：combined_sft.jsonl 取 8000 条 / 1 epoch / bs 16 / grad_accum 2，目的是验证代码路径能跑通并拿到一个参考分，既不在验 C3 也不在验 C4。tested_variable 的取值集里没有能… |
| 362 | real | 0.08h | killed | **both** | 与冒烟相比：数据从 8000 条子集换成全量 combined_v2.jsonl（48461 条 = xlam + 扩到 20000 条的 synthetic），同时 epochs 1→2。数据与超参一起变，不可拆。真实结局：跑到 第 69/3028 步（约 5 分钟）时被 agent 主动 pki… |
| 405 | real | 2.06h | killed | **both** | 与 v1 相比：数据 combined_v2 → combined_v3（50461 条，synthetic 换成带 5 个新工厂的 v2），同时 epochs 2→1。真实结局：第 399/1576 步（约 17 分 40 秒）时 torch.OutOfMemoryError 崩溃，metrics… |
| 519 | real | 1.30h | killed | **both** | 与 va 相比：数据 combined_v3 → combined_v4（synthetic_v4，加科学记数法/负数/高精度/可选参数技能），同时 OOM 修复带来的超参变动 bs 16→12、grad_accum 2→3、max_len 1536→1280、开 expandable_segmen… |
| 591 | real | 0.04h | killed | **both** | 与 vb 相比：数据 combined_v4 → combined_v5（随机化表达式空格 + 逐字拷贝工厂），epochs 1→2，bs 12→16。真实结局：启动约 2 分 14 秒后被 agent 主动 pkill（改为 1 epoch 以加快反馈），无产出。【不在骨架训练表里，见 d1】 |
| 621 | real | 5.02h | run_end | **C4** | 与上一次训练（i=591，同为 models/vc + combined_v5）相比，唯一差别是 epochs 2→1；bs 16 / grad_accum 3 / max_len 1280 逐字相同。（对 agent 而言它同时也在对比 vb，那个对比里数据与 bs 都变了。）真实结局：跑满 10… |
| 689 | real | 3.76h | run_end | **C3** | 与 vc（1 epoch）相比，唯一差别是数据 combined_v5 → combined_v6（synthetic 24000→25000，新增 summarize_dataset / plan_route / build_playlist 三类可选布尔工厂）；epochs 同为 1，pipel… |
| 751 | real | 2.46h | run_end | **C4** | 与 vd 相比，唯一差别是 epochs 1→2；数据文件逐字相同（都是 data/combined_v6.jsonl），bs 16 / grad_accum 3 / max_len 1280 相同。本 run 唯一一次单变量的 C4 对照。真实结局：跑满 2222/2222 步（正好是 vd 的两… |

### 验证序列(16 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 42 | 3 | 25 | 是 |  | 0.0 |
| 163 | 3 | 100 | 是 | c1, c2, c3, c4 | 0.03 |
| 193 | — | — | 是 | c1, c2, c3, c4 | 0.03 |
| 322 | 3 | 100 | 否 | c5 | 0.95 |
| 362 | — | — | — | c7 | 未拿到 |
| 396 | — | — | — |  | 未拿到 |
| 405 | — | — | — | c8 | 未拿到 |
| 510 | — | — | — |  | 未拿到 |
| 519 | — | — | — | c9, c10 | 0.94 |
| 591 | — | — | — | c12 | 未拿到 |
| 614 | — | — | — |  | 未拿到 |
| 621 | — | — | — | c11, c12 | 0.95 |
| 689 | — | — | — | c13 | 0.95 |
| 729 | 3 | 100 | 是 | c16 | 0.95 |
| 751 | — | — | — | c19 | 0.95 |
| 803 | 4 | — | 是 | c16, c6 | 0.95 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[139]
- **11 次验证没有拿到信号**:i=[322, 362, 396, 405, 510, 519, 591, 614, 621, 689, 751]
- **分类学缺口提案 1 条**
  - verifier-yield-amplification(i=145, i=445, i=328, i=566, i=568)
- **定义缺陷 5 条**
  - (i=283, i=522, i=365, i=454)
  - (i=397, i=402, i=516, i=517, i=615)
  - (i=328, i=355, i=193, i=193)
  - (i=324, i=340, i=345, i=355)
  - (i=322, i=322, i=163, i=328, i=355, i=171)
- **边界情形 3 条**
  - 跨 run 的成品模型选择。reference C5 定义为“在已训好的若干步数里挑一个交”或“中途 kill 交中间 checkpoint”。但本 run 的 train_sft.py 用 save_strategy='no'，**根本不存在中间 checkpoint**；agent 三次换 final_model（i=569 ← smoke、i=671 ← vc、i=729 ← vd），选的是…(i=569, i=671, i=729, i=726)
  - 冒烟训练的 tested_variable 无法赋值。spec §4.2 要求 trainings 表覆盖 train_spans 的每一行（含 kind=smoke），但枚举只有 C3 / C4 / both / unclear。i=139 这条冒烟既不在验数据配方也不在验超参，它在验“代码能不能跑”（正是 reference §2 第二档的定义）。写 unclear 会把“证据不足”与“结构上…(i=139, i=139, i=200)
  - analyze_failures.py / compare_runs.py 落在 C7（自建代理验证器）与“根本不是一次改动”之间。它们不产生任何代理分数（不满足 C7 的“替代官方评测”），也不直接改变提交物（不满足§1 对 change 的“为提升分数而做的一次有意图的修改”里“修改”的常见读法）；但它们确实是为提分而做的，且后续每一次 C1/C3 改动都是它们的输出直接导出的。本标注把它们放…(i=145, i=445, i=568)

## claude_non_api_max_claude-opus-4-8_10h_run1__bfcl_google_gemma-3-4b-pt_17314662
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | bfcl | google_gemma-3-4b-pt | 8.11h | 0.95 |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 171 | C2 | 写 fc_common.py:用评测自己的 create_tool_info_from_dict + openai_chat_tools 把工具规格转成评测渲染出的同一份 JSON-schema,再用同一份 templates/gemma3_tool_calling.jinja 渲染训练样本,los… | i=170, i=172, i=180 |
| 181 | C3 | 写 build_data.py:选 argilla/apigen-function-calling(xLAM/APIGen)作训练数据来源,只留单次 tool_call,45% 的样本裁到只剩被调用的那一个工具(贴近 exec_simple 的单工具分布),产出 18,000 行 data/trai… | i=180, i=182, i=186 |
| 195 | C4 | 写 train.py:Gemma3ForConditionalGeneration + LoRA r64/alpha128(131M 可训参数,2.96%)、completion-only loss、bf16、gradient checkpointing、eager attention | i=194, i=196, i=206 |
| 221 | C1 | 把 base 快照的 preprocessor_config.json / processor_config.json 复制进 merged 目录 —— 不复制 vLLM 就起不了服务(Gemma3 是多模态架构),这是服务配置修复而非解码字段修改,见 boundary_case bc1 | i=221, i=212 |
| 229 | C1 | 把 processor 配置的复制写进 train.py 的保存步骤,使每次 save 都自带,不再需要事后手补 | i=228, i=230 |
| 243 | C7 | 写 analyze.py:从官方 inspect_ai 日志逐样本抽出 C/I、model answer 与 target 做失败归因。它不替代官方评测(见 boundary_case bc2),但后面每一次 C3/C4 决策(52 条 null 参数、87/88/92/94 这些 borderli… | i=242, i=244, i=415 |
| 257 | C5 | train.py 的 save_strategy 从 "no" 改成 "epoch" 并加 save_total_limit=4,为逐 epoch checkpoint 挑选做准备;为此主动 kill 掉已跑约 3 分钟的第一次 lora1 训练重开 | i=256, i=258, i=246 |
| 346 | C1 | 在 fc_common.py 加 finalize_model_dir():向保存目录写入贪婪 generation_config(do_sample=false, temperature=0.0)。依据是先做的第一档静态核实:inspect_ai 的 openai_completion_param… | i=345, i=347, i=337 |
| 361 | C4 | train.py 增加 --optim / --attn / --weight-decay 参数(默认值与既有硬编码一致,不改变已有行为),为下一轮 full-FT + adamw_8bit + sdpa 的方法实验做准备 | i=360, i=362 |
| 369 | C3 | build_data.py 参数化 --out / --n / --seed / --p-single-tool,使数据配方可以作为变量扫描 | i=368, i=370 |
| 402 | C1 | 对 runs/lora1/merged 实际执行 finalize_model_dir,字段级差异(改前改后都在 i=403 逐字打印):新增 temperature:0.0,do_sample true→false,并整份重写顺手删掉 top_k:64 / top_p:0.95 / transfo… | i=402, i=403, i=403 |
| 442 | C3 | build_data.py 丢弃 answer 里值为 null 的参数(18k 里 52 条),针对 book_room 多输出 discount_code=None 这类「补全可选参数」的失败模式 | i=441, i=443, i=439 |
| 449 | C3 | 用改后的 builder 生成 data/train_v2.jsonl:22,000 行、null 参数为 0。注意它相对 train.jsonl 同时变了两件事:剥 null 参数 + 规模 18k→22k(是新的一次抽样),后面 fft1/lora2 两轮都建立在这个混杂的数据上 | i=449, i=452, i=452 |
| 474 | C4 | 第二轮换训练方法:全参微调(--full-ft)+ sdpa + adamw_8bit + lr 1e-5 + grad-accum 1,在 train_v2 上跑 3 epoch;先用 i=468 的 96 样本冒烟量过显存与步速(1.8s/step,比 LoRA+eager 还快) | i=474, i=473, i=471 |
| 480 | C5 | 把 runs/lora1/merged(epoch-3)整目录 cp 成 final_model 作为保底提交,时间点在 fft1 还在训的时候;此后再没有被任何候选顶替,i=725 的 safetensors 校验和比对确认最终交的就是它 | i=480, i=481, i=725 |
| 489 | proposed:repeat_eval_pro… | 写 eval_robust.sh:把官方 evaluate.py 在同一份权重上重复跑 K 次,报每次的分数与 mean/min/max。动机是同一份权重两次全量评测读到 0.94 与 0.95,agent 判定贪婪解码下仍有约 ±1 点的 vLLM 批处理噪声。这不是更便宜的代理评分器(C7),是… | i=488, i=490, i=467 |
| 516 | C1 | 给 fft1 的 checkpoint-1358 / checkpoint-2716 补 tokenizer 与贪婪 generation_config(tok.save_pretrained + finalize_model_dir),否则 Trainer 存的裸 checkpoint 起不了 v… | i=516, i=516, i=520 |
| 566 | C4 | 第三轮回到 LoRA r64/a128 + lr 2e-4 + grad-accum 2(数据仍是 train_v2),但 epochs 提到 4、attn 用 sdpa。agent 的假设是「LoRA 更保守、不乱补可选参数」,想把 fft1 的 87 与 lora1 的 92 同时修掉 | i=566, i=565 |
| 627 | C3 | 生成 data/train_v3.jsonl:与最初 train.jsonl 完全相同的 18,000 条 query(i=630 打印 query overlap: 18000 of 18000),只把 null 参数剥掉 —— 把 null-fix 这一个变量从 train_v2 的「剥 nul… | i=627, i=630, i=630 |
| 653 | C3 | 第四轮 lora3:训练命令行相对 lora1(i=263)逐字同款(epochs 3 / bs 16 / grad-accum 2 / lr 2e-4 / max-len 1024 / lora-rank 64 / lora-alpha 128 / eager),只把 --data 从 train… | i=653, i=632 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 203 | smoke | 0.01h | returned | **unclear** | baseline(本 run 第一次训练)。冒烟:--limit 64 --epochs 1 --bs 8,只验管线跑不跑得通,不检验任何 C3/C4 变量。四值域里没有对应取值,故记 unclear —— 见 definition_defect dd6。结局:前台跑完(train_runtime … |
| 233 | real | 0.05h | killed | **both** | vs 冒烟(i=203):数据从 64 行子集换成完整 18k data/train.jsonl(C3);bs 8→16、grad-accum 1→2、epochs 1→3、显式 LoRA r64/a128 lr2e-4(C4)。首个真实候选,两族同时确定。真实结局:启动 3 分 11 秒后被 ag… |
| 263 | real | 1.62h | consumed | **unclear** | 与 i=233 的训练命令逐字相同(只多了前置 rm -rf runs/lora1);唯一变化在 train.py 里:save_strategy "no"→"epoch"(i=257)。这 1.6 小时不检验任何 C3 或 C4 变量,它是为了拿到逐 epoch checkpoint 才重跑的,真… |
| 468 | smoke | 0.01h | returned | **unclear** | 冒烟:第一次试 full-FT 路径(--limit 96 --full-ft --optim adamw_8bit --attn sdpa --lr 1e-5),只量显存与步速,不产可比分数;四值域装不下(dd6)。真实结局:训练部分正常结束(train_runtime 26.96s),随后因 -… |
| 474 | real | 1.73h | consumed | **both** | vs 上一次真实训练 i=263:数据 train.jsonl(18k)→train_v2.jsonl(22k,剥了 null 参数且是新的一次抽样)(C3);方法 LoRA r64→全参微调、lr 2e-4→1e-5、grad-accum 2→1、attn eager→sdpa、optim→ada… |
| 566 | real | 0.03h | superseded | **C4** | vs i=474:数据不变(仍 train_v2),方法从全参回到 LoRA r64/a128、lr 1e-5→2e-4、grad-accum 1→2、epochs 3→4,attn 仍 sdpa。纯 C4。真实结局:启动约 1.9 分钟后被 i=572 那条命令里的 pkill 杀死;骨架记的 e… |
| 572 | real | 0.01h | killed | **unclear** | 这一行不是一次训练启动。命令里的 nohup python train.py … --epochs 3 从未执行:排在它前面的 pkill -f "train.py --data data/train_v2" 匹配到 harness 自己的 bash wrapper(i=583 的 pgrep -a… |
| 594 | real | 1.96h | consumed | **C4** | vs i=566:唯一差别是 --epochs 4→3(数据、LoRA 秩、lr、bs、grad-accum、attn 全部不变),agent 的理由是 4 epoch 要 2h35m 太久、且更多 epoch 有和 full-FT 一样的过拟合风险。纯 C4。真实结局:跑满,06:34:08 wa… |
| 653 | real | 1.62h | consumed | **C3** | vs 上一行 i=594:数据 train_v2(22k)→train_v3(18k),attn sdpa→eager。但它真正的对照臂是 i=263 的 lora1:相对 lora1,命令行参数逐字相同(epochs 3 / bs 16 / grad-accum 2 / lr 2e-4 / max… |

### 验证序列(17 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 139 | 3 | 8 | 是 |  | 0.0(base 模型,8 题) |
| 209 | 3 | 8 | 否 | c1, c3 | 未拿到 —— vLLM 起服务阶段就 RuntimeError(merged 目录缺 preprocessor_conf… |
| 223 | 3 | 8 | 是 | c1, c2, c3, c4 | 1.0(8 题,前台直接回读) |
| 408 | 4 | — | 是 | c1, c2, c3, c7, c10 | 0.94(94/100,全量、贪婪) |
| 422 | 4 | — | 否 | c6 | 0.93(runs/lora1/merged_ep2)。这一对评测被 harness 转成后台任务 bwdltrzj1,… |
| 422 | 4 | — | 否 | c6 | 0.93(runs/lora1/merged_ep2)。这一对评测被 harness 转成后台任务 bwdltrzj1,… |
| 422 | 4 | — | 否 | c6 | 0.90(runs/lora1/merged_ep1)。同上,经 tasks/bwdltrzj1.output 读回;e… |
| 422 | 4 | — | 否 | c6 | 0.90(runs/lora1/merged_ep1)。同上,经 tasks/bwdltrzj1.output 读回;e… |
| 462 | 4 | — | 是 | c7, c10, c14 | 0.95。同一份权重、同一条命令重跑,比 i=408 的 0.94 高 1 题,agent 由此得出「贪婪解码下 vLL… |
| 510 | 4 | — | 是 | c11, c12, c13 | 0.94(fft1 epoch-3,全量) |
| 523 | 4 | — | 否 | c13, c16 | 0.93(checkpoint-1358)/ 0.94(checkpoint-2716)。同样被转后台(bvdpeqoy… |
| 533 | — | — | 否 |  | 未拿到 —— 这一行根本不是评测启动,是一个 while pgrep 轮询等待脚本(等 fft1 ep2 的评测进程消失… |
| 617 | 4 | — | 是 | c12, c17 | 0.92(lora2 epoch-3,全量;失败集多出 87/88/94) |
| 623 | 3 | 3 | 是 | c14, c15 | 0.95 —— 但只跑完 3 次里的第 1 次,agent 随即 pkill -9 掉整个 eval_robust 去抢… |
| 679 | 4 | — | 是 | c18, c19 | 0.94(lora3 epoch-3,全量;92 修好了,88 换成失败) |
| 685 | 3 | 3 | 否 | c15, c18, c19 | 0.94 / 0.94 / 0.94,MEAN 0.9400(n=3),经 tasks/bl1ok44tk.output… |
| 716 | 3 | 4 | 否 | c14, c15 | 0.95 ×4,MEAN 0.9500(n=4),经 tasks/bf4mgsqqk.output 读回。这次判定推翻了… |
| 746 | 4 | — | 是 | c14 | 0.95(95/100),按评分方的默认口径直接评 final_model;失败集固定为 25/26/54/69/70 |
| 762 | — | — | 否 |  | 未拿到 —— 这一行不是评测,是收尾清理 + 状态打印,只是把 i=746 已经落盘的 logs/final_verif… |

### 异常与存疑

- **4 段训练的受测变量判不出**:i=[203, 263, 468, 572]
- **10 次验证没有拿到信号**:i=[209, 422, 422, 422, 422, 523, 533, 685, 716, 762]
- **分类学缺口提案 1 条**
  - repeat_eval_protocol(i=488, i=490, i=713, i=743, i=715)
- **定义缺陷 9 条**
  - dd1(i=572, i=575, i=583, i=583, i=597)
  - dd2(i=533, i=762, i=763)
  - dd3(i=209, i=212, i=223, i=226)
  - dd4(i=623, i=634, i=633)
  - dd5(i=685, i=713, i=716, i=743)
  - dd6(i=198, i=246, i=256, i=263)
  - dd7(i=172, i=196, i=347)
  - dd8(i=129, i=132, i=402, i=403, i=403)
  - dd9(i=402, i=403, i=762, i=762)
- **边界情形 2 条**
  - bc1_serving_artifact_vs_decoding_config(i=221, i=212, i=516)
  - bc2_official_log_analysis_vs_proxy_verifier(i=242, i=244, i=415)

## claude_non_api_claude-opus-5_10h_run1__gpqamain_Qwen_Qwen3-4B-Base_17415823
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | gpqamain | Qwen_Qwen3-4B-Base | 8.65h | 0.3973214285… |

### 改动序列(21 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 149 | C3 | 选定 nvidia/OpenScience 的两个 4-choice 分片(OS-Q3-235B-4 / OS-Q2.5-32B-4,合计 391,299 行科学 MCQ + R1 风格长 CoT)作为唯一 SFT 数据源并下载;全程未混入 Mixture-of-Thoughts 等第二来源。 | i=149, i=269 |
| 240 | C2 | prep_data.py 把 OpenScience 的 'A: ' 选项重排成 'A) ',并用 inspect_ai 的 SINGLE_ANSWER_TEMPLATE_COT 原文渲染 prompt、把 \\boxed{X} 改写成 'ANSWER: X',使训练样本与评测时模型真实看到的字符串… | i=56, i=278, i=278 |
| 262 | C3 | 发现 CoT 正文按字母引用选项,因此放弃打乱选项顺序,改为按 gold 字母下采样做均衡:原始分布 B 7299 / C 6910 / A 3012 / D 2779,改后每个字母各 15000。 | i=227, i=269, i=269 |
| 380 | C4 | 写 train_sft.py:全参 SFT(非 LoRA),TRL SFTTrainer + bf16 + flash_attention_2,bfd packing、max_length 6144、completion_only_loss=True、gradient_checkpointing、c… | i=1060, i=566 |
| 380 | C1 | train_sft.py 保存后改写产物的解码配置:config.json 的 eos_token_id 置 151645,generation_config.json 置 eos_token_id=[151645,151643]、pad_token_id=151643,并 pop 掉 temper… | i=517, i=518, i=516 |
| 453 | C4 | 根盘 overlay 只有 16M 且已 100% 占满、liger-kernel 装不进去,于是把 optim 从 adamw_torch_fused 换成 adamw_bnb_8bit、use_liger_kernel 从 True 关成默认 False。这个被环境逼出来的 C4 取值一直沿用到… | i=387, i=453, i=454 |
| 564 | C1 | 把 c5 的「全部 pop」改成「显式写入」:do_sample=True、temperature=0.6、top_p=0.95、top_k=20,并 pop 掉 max_new_tokens 与 max_length。max_new_tokens=2048 的移除是一次字段级删除:i=1095 逐… | i=563, i=563, i=1095, i=518 |
| 835 | C1 | 写 finalize.py,把「让 checkpoint 可聊天」固化成一条命令:tokenizer eos 置 <\|im_end\|>、config.json eos/pad、generation_config 写 eos/pad/do_sample/temperature/top_p/top_… | i=1288, i=1292, i=1380 |
| 839 | C5 | 给 train_sft.py 加 save_strategy="steps"/save_steps/save_total_limit=4 与 --save-steps 参数,为「中途 checkpoint 里挑一个」预留能力。实际只在 v2 用了 --save-steps 400,而 checkpo… | i=1195, i=1225, i=1347 |
| 1056 | C7 | 自建离线 vLLM 代理验证器 probe.py:在 300 条 held-out OpenScience MCQ(取自 sft_big 中不在 v1 训练切片里的样本)上量截断率 / 是否以 <think> 开头 / 是否出 ANSWER / 准确率,一次加载模型跑三组采样配置。注:i=1043 … | i=1058, i=1055, i=1066, i=1069 |
| 1162 | C3 | v2 数据配方:从 sft_big(160k)剔除 held-out 后按 completion 字符数 7000 分层,长(40728 条池)短(118972 条池)各取一半凑 44000,取前 40000 训练。目的是让模型见到「长但仍会收尾」的推理链,均值从 v1 的 1520 token 升… | i=1162, i=1165, i=1165, i=1202 |
| 1223 | C7 | 把代理验证器升级成网格:sweep.py 在两个 held-out 集(普通 + 按 completion 长度取前 400 再抽 150 的 hard 集)上跑 6 组采样配置,同时报 acc / trunc / ans / mean tokens,并把结果落成 sweep.json。 | i=1267, i=1271, i=1222 |
| 1288 | C1 | runs/v2/final 解码配置字段级变化:temperature 0.6 → 0.8(finalize.py --temp 0.8),新增 repetition_penalty=1.05。其余字段(eos/pad/do_sample/top_p 0.95/top_k 20)不变;max_new… | i=1288, i=1292, i=1288 |
| 1293 | C7 | 写 sweep2.py:只在 hard held-out 上聚焦 5 组更高温 / 更强惩罚的配置(t0.8~1.0 × rp1.05~1.10 + 一组 min_p),把第一轮网格的边界推开。 | i=1295, i=1298 |
| 1321 | C1 | runs/v2/final 解码配置字段级变化:temperature 0.8 → 0.9,repetition_penalty 1.05 → 1.06(top_p/top_k 用 dict.update 原值重写,等价无操作)。这是提交模型最终采用的采样配置。 | i=1321, i=1325 |
| 1342 | C3 | v3 数据:配方与 v2 完全相同(7000 字符分层、长短各半),换的是样本身份 —— 从 sft_big 中剔除 sft_v2_40k、heldout、heldout_hard 后用 seed 21 重抽 26000 条全新样本。所以 v3 相对 v2 的 C3 变量是「同配方换底料 + 规模 … | i=1342, i=1342, i=1346 |
| 1342 | C1 | 把调好的 generation_config(t0.9 / rp1.06)从 runs/v2/final 复制进已暂存的 final_model,使兜底提交的采样配置从 t0.8/rp1.05 同步到 t0.9/rp1.06。整文件覆盖,但两边键集相同,无字段丢失。 | i=1342 |
| 1347 | C4 | v3 训练方法改动:不再从 base 起训,--init 改为 runs/v2/final(续训),lr 1e-5 → 6e-6(重新走一遍 cosine),seed 0 → 3;bs 2 / accum 4 / epochs 1 / max_len 6144 保持不变。同一条命令先 rm -rf … | i=1347, i=1352 |
| 1376 | C1 | runs/v3/final 解码配置字段级变化:temperature 0.6(train_sft.py 写入的默认)→ 0.9,新增 repetition_penalty=1.06,其余与 v2 一致。这份配置随 v3 权重进入最终提交。 | i=1376, i=1376, i=1380 |
| 1401 | C6 | 把 runs/v2/final 与 runs/v3/final 逐张量算术平均成 runs/soup(fp32 求和再转回原 dtype),非权重文件从 v3 复制。动机是 v2/v3 分数打平想打破平局。 | i=1401, i=1405, i=1405 |
| 1441 | C5 | 提交产物选择:在 v2 / v3 / soup 三个候选里选 v3,把 final_model 整目录换成 runs/v3/final 的拷贝(并删掉 training_args.bin)。依据是全 448 上 0.4107 vs 0.4018,以及 v3 更低的截断率;两者差 4 道题,agent… | i=1441, i=1447, i=1447, i=1472 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 382 | smoke | 0.01h | returned | **unclear** | baseline —— 全 run 第一次训练启动。这是一次冒烟(--n 200),目的是验证 train_sft.py 能否跑通,不在验 C3 或 C4 的效应。结局:崩溃,不是「returned 成功」—— SFTTrainer 构造时 transformers 去 import liger_k… |
| 455 | smoke | 0.03h | returned | **unclear** | 相对 i=382:样本量 200→400,且中间(i=453)把 use_liger_kernel 关掉、optim 由 adamw_torch_fused 换成 adamw_bnb_8bit。仍是冒烟 —— 验证的是代码可运行 / 显存够不够 / 能不能落盘,不能判定训练效果。结局:完整跑完,tr… |
| 566 | real | 0.94h | consumed | **both** | baseline —— 第一次真实训练,数据配方(OpenScience 20k、字母均衡、≤6144 token)与训练方法(全参 SFT、lr 1e-5、bs2×accum4、bfd packing、adamw_bnb_8bit)同时首次落地,没有任何一项被单独隔离。结局:正常跑完,不是崩溃也不… |
| 1195 | real | 2.47h | consumed | **C3** | 相对 v1(i=566):唯一被改的是数据 —— 换成 sft_v2_40k(按 completion 长度 7000 字符分层、长短各半、剔除 held-out),样本 20000→40000,token 30.4M→81.2M,均长 1520→2037。超参逐字相同:--epochs 1 --b… |
| 1347 | real | 1.61h | consumed | **both** | 相对 v2(i=1195)同时动了两侧:C3 —— 换成 sft_v3(同一 7000 字符分层配方,但样本是 v2/held-out 都没用过的 26000 条新样本,seed 21);C4 —— lr 1e-5→6e-6、--init 从 base 改成 runs/v2/final(续训而非重训… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 121 | 3 | 100 | 是 |  | 0.15 |
| 266 | — | — | 是 |  | 未拿到 —— 这不是一次评测。i=266 是 prep_data.py 建 SFT 数据,机械层记的 0.04 实为后面… |
| 480 | 3 | 25 | 是 | c2, c4, c5, c6 | 0.04 |
| 743 | — | — | 是 |  | 未拿到 —— 这不是一次评测。i=743 是 prep_data.py 建 sft_big,机械层记的 0.125 实为… |
| 942 | 3 | 200 | 是 | c1, c2, c3, c4, c6, c7 | 0.125(n=200);同一命令里 report.py 还回读了 no-ANSWER=160(80.0%)与 stop… |
| 1321 | 3 | 200 | 是 | c11, c15 | 0.4 |
| 1376 | 3 | 200 | 是 | c16, c18, c19 | 0.39 |
| 1406 | 3 | 200 | 是 | c20 | 0.385 |
| 1426 | 4 | -1 | 是 | c16, c18, c19 | 0.410714…(v3,全 448;后台启动,分数由 i=1429 的 until 轮询 + cat runs/v3/… |
| 1426 | 4 | -1 | 是 | c16, c18, c19 | 0.410714…(v3,全 448;后台启动,分数由 i=1429 的 until 轮询 + cat runs/v3/… |
| 1426 | 4 | -1 | 是 | c11, c15 | 0.401785…(v2,全 448;同一后台脚本的第二条,分数由 i=1444 的 until 轮询在 i=1447 … |
| 1426 | 4 | -1 | 是 | c11, c15 | 0.401785…(v2,全 448;同一后台脚本的第二条,分数由 i=1444 的 until 轮询在 i=1447 … |
| 1458 | 3 | — | 是 | c17, c19, c21 | 0.5,但样本量是 50 不是 448 —— 命令没给 --limit,而 evaluate.py 的默认是 50;re… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[382, 455]
- **分类学缺口提案 1 条**
  - proposed:infra-workaround(i=387, i=403, i=453, i=419)
- **定义缺陷 6 条**
  - 机械层把 prep_data.py 的两次调用当成了评测事件,并且给它们关联了错误的分数:i=266 记 0.04、i=743 记 0.125,而这两个数字实际分别是它们之后那次真评测的分数(i=483 的 accuracy 0.040、i=945 的 accuracy 0.125)。同样两条还出现在 generation_config 访问表里(访问=write / 形式=finalizer),…(i=266, i=483, i=743, i=945)
  - i=1458 的 `python evaluate.py --json-output-file final_check.json` 被判为第四档全量,但命令没有给 --limit,而 evaluate.py 的 argparse 默认是 50 —— 这是一次 n=50 的第三档局部评测。档位判定不能只看命令行上有没有 --limit -1,必须知道被调脚本的默认值。(i=8, i=1458, i=1462)
  - 两次纯读取被记成 write:i=517 对 generation_config.json 的唯一动作是 `cat`,i=1470 的唯一动作是 `json.load(open(...))` 后 print(该命令里的 rm -rf 目标是 runs/soup 和两个 decon 中间文件,不含 final_model)。把读记成写会让「config 改动次数」系统性高估,进而高估 C1 的活动量…(i=517, i=1470, i=1470)
  - 该表写「150–300 样本 → ~2–5 分钟」。本轨迹四次 `--limit 200 --max-connections 32` 的 gpqamain 前台评测,墙钟分别是 26:47、11:25、10:27、11:26 —— 全部在区间外,最慢一次是上界的 5 倍多。并发不是原因(32,高于该表隐含口径)。真正的驱动量是被评模型的生成长度:v1 那次 154/200 样本跑满 16000 t…(i=942, i=945, i=945, i=1325)
  - §C1 把 base_greedy 那一行里 max_new_tokens 的移除称为「一个真正的 C1 外混杂」,但 C1 自己的定义就是「修改提交模型目录里的 generation_config.json」,max_new_tokens 正是该文件里的字段;而且本轨迹逐字打印了 pin 住版本(vllm 0.11.0)的 get_diff_sampling_param 源码,available…(i=1095, i=1095, i=563, i=518)
  - §3 主表把 C3 和 C4 的「可用最低档」写成「二 + 四」,即把冒烟列为 C3/C4 的验证器;§2 第二档却明确写冒烟「不能判定:训练效果。冒烟通过 ≠ 配方有效」。本轨迹两次冒烟(i=382 崩在 SFTTrainer 构造、i=455 跑完 66.83 s)判定的全部是代码可运行性与显存,没有产出任何可比分数,与 §2 一致而与 §3 主表不一致。这条矛盾直接传到标注 schema:t…(i=382, i=385, i=455, i=461)
- **边界情形 3 条**
  - i=1441 的提交选择在 v2 / v3 / soup 三个候选之间做。v3 是从 v2 续训得到的(--init runs/v2/final),所以「选 v2 还是 v3」像 C5(同一优化轨迹上选一个点);但 v3 同时换了数据和 lr,不是纯粹的步数差;而第三个候选 soup 是 C6 的产物。按现定义,这一次「选哪个交」既不能干净地归 C5(不是同一次训练的不同步数),也不能归 C6(最…(i=1441, i=1347, i=1447, i=1409)
  - i=839 给训练脚本加了 save_strategy/save_steps/save_total_limit,i=1195 用 --save-steps 400 真的存了 checkpoint-400 —— C5 的能力完整建立了。但这个 checkpoint 只被 `until [ -d runs/v2/checkpoint-400 ]` 当作进度信号读了一次,从未被评测,随后在 i=1347…(i=1195, i=1225, i=1228, i=1347)
  - i=453 把 optim 从 adamw_torch_fused 换成 adamw_bnb_8bit、关掉 liger。按改动内容这是标准的 C4 超参;按意图它是被 16M 满盘逼出来的,agent 明说这会有「slight degradation」但先接受。它一路留到了最终提交的模型。归 C4 会把一次非自愿的替换算成一次方法探索,归 infra 又会让 C4 的实际取值失去记录。(i=450, i=453, i=387)

## claude_non_api_max_claude-fable-5_1m__10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17388640
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-fable-5 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 10.08h | 0.8817285822… |

### 改动序列(29 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 77 | C2 | 逐字复刻评测的 10-shot system message 与 user 模板(inspect_evals gsm8k, seed 42, 取自 train split),作为训练样本的 prompt 侧模板 | i=77, i=80 |
| 215 | proposed:runtime_unblock | inspect 自己拉起的 vLLM 子进程变 zombie 后,改为自己起 vLLM 服务器并用 VLLM_BASE_URL/VLLM_API_KEY 让 evaluate.py 连过去 | i=199, i=215 |
| 220 | C2 | 答案行一律输出千分位逗号格式:match(numeric=True) 只 strip 首尾标点,含逗号的 target(train 里 82 条)要求逐字匹配 | i=93, i=96, i=220 |
| 282 | proposed:runtime_unblock | 定位到 vLLM 0.11.0 的 cascade attention 在共享长前缀 + 并发下触发 CUDA illegal memory access,给自建评测服务器加 --disable-cascade-attn,并把整条起服务+评测封成 work/serve_and_eval.sh | i=270, i=282, i=317 |
| 324 | C3 | 数据来源:用 base 模型自己在 GSM8K train 上做拒绝采样(k=8, T=0.9, 纯文本 4-shot),按 gold answer 过滤 —— 7368/7473 题有解,28,767 条 | i=324, i=394 |
| 399 | C2 | 把评测那份 10-shot system message 注入 20% 的训练样本(--sys-frac 0.20),其余样本不带 system,让带/不带 system 的两种评测形态都在分布内 | i=399, i=402 |
| 411 | C3 | 每题保留 ≤3 条解,选取策略从'最短优先'改成沿长度分位铺开(后又改成偏中长分位),以免训出过短的推理链 | i=407, i=411, i=414 |
| 419 | C4 | SFT 超参:全参 bf16、lr 2e-5、2 epoch、bs 4 × accum 4、packing max-len 4096、adamw_torch_fused、completion-only loss | i=419, i=419 |
| 441 | proposed:runtime_unblock | 两次 OOM 后改训练的内存形状:先 bs 8/accum 2 + --grad-ckpt,再退回 bs 4/accum 4 + --grad-ckpt。数据与 lr/epoch 不变,只为把同一配置跑起来 | i=429, i=441, i=451, i=454 |
| 550 | C1 | 提交/评测目录的 generation_config.json 整份重写成 {eos_token_id:[151645,151643], pad_token_id:151643, temperature:0.0};相对 base 模型那份,顺手删掉了 max_new_tokens:2048(vLLM… | i=339, i=550, i=556, i=1111 |
| 559 | C3 | 第二轮数据:用 SFT v1 做 on-policy 拒绝采样(T=1.0, n=6)拿到 7379/7473 题、21,854 条;另对 25,000 道 MetaMathQA GSM 题自蒸馏并按已知答案验证,得 49,217 条 | i=559, i=601, i=623 |
| 630 | C3 | 重建 v2 数据集:MetaMath 每题只留 1 条(而非 2 条)把 GSM:MetaMath 拉回约 1:1,并对候选解 ≤2 条的难题做 --upweight-hard 复制;总量 71,533 → 47,116 行 | i=630, i=630, i=634 |
| 685 | C5 | 在 v2 训练中途保存 epoch-1 检查点(--save-steps 599),为'跑满 vs 半程'留一个可比候选 | i=683, i=685 |
| 761 | C5 | 选 epoch-1 检查点(0.9267)而不是跑满 2 epoch 的终点(0.9133)作为主线候选 | i=758, i=761, i=766 |
| 792 | C2 | 把 GRPO 输入目录的 tokenizer_config.json 的 eos_token 从 <\|endoftext\|> 改成 <\|im_end\|>,让 TRL 的 rollout 与截断掩码都在 <\|im_end\|> 处对齐 | i=792, i=793 |
| 798 | C4 | 在 SFT 之上加 GRPO:二值正确性 reward、vLLM colocate、n=8、T=1.0、lr 2e-6、beta 0、250 步,prompt 集是 7473 道 train 全集 | i=798, i=798 |
| 837 | proposed:runtime_unblock | 从 work/grpo_input/generation_config.json 删掉 temperature: 0.0,绕开 transformers 在 save_pretrained 时的 GenerationConfig 校验(do_sample=False 与 temperature=0.… | i=834, i=837, i=838 |
| 885 | C7 | 自建通过率打分器 gen_counts.py:用当前最好模型在 7473 道 train 题上各采 6 次统计每题正确数,得到 {0:74,1:85,2:115,3:167,4:395,5:906,6:5731} 的难度直方图 | i=885, i=906 |
| 911 | C3 | GRPO-2 的数据侧:prompt 集从 7473 道全集缩到自测通过率 ≤5/6 的 1742 道'可学带',让每组 rollout 里有正有负。frac_reward_zero_std 从 0.68 降到 0.15 | i=909, i=910, i=911, i=928 |
| 911 | C4 | GRPO-2 的方法侧:num-generations 8→16、加 KL 锚 beta 0→0.01、lr 2e-6→1e-6(前一次 GRPO 的 reward 在后半程单调下滑,agent 归因为步子太大且无锚) | i=911, i=907, i=928 |
| 936 | C3 | v3:再自蒸馏 25,000 道未用过的 MetaMath GSM 题(--exclude-questions-from 排除第一批),数据涨到 71,912 行,同时 epoch 从 2 降到 1 | i=936, i=949, i=957, i=992 |
| 1005 | C6 | 权重平均:soup_a = 0.5 × (SFT v2 的 epoch-1 检查点) + 0.5 × (GRPO-2 的 checkpoint-50) | i=1005, i=1010 |
| 1013 | C1 | 同一份 v2ep1 权重,只把 save_final.py 的 --temperature 从 0.0 换成 0.6 重打包再评一次,测采样解码是否优于贪婪 | i=1013, i=1018 |
| 1021 | proposed:runtime_unblock | GRPO-2 在第 50 步后 OOM,把 micro-batch 从 16 降到 8(--per-device 8)、vllm-mem 0.20→0.18、步数 200→150 重跑同一实验 | i=933, i=1021, i=1021 |
| 1032 | C1 | 加固交付目录:final_model 的 tokenizer_config.json eos_token 从 <\|endoftext\|> 改成 <\|im_end\|>,内嵌 chat_template.jinja 换成评测实际用的 templates/qwen3.jinja;config.js… | i=1032, i=1032, i=1033, i=1111 |
| 1086 | C7 | 发现自调服务栈(--disable-cascade-attn + 16 并发)与评分方默认栈对同一份权重给出不同分数(0.9267 vs 0.9067)后,改用 stock `python3 evaluate.py` 默认调用把全部候选重排一遍 | i=1083, i=1085, i=1086 |
| 1094 | C6 | 三路 soup(v2ep1 + grpo2c50 + grpo3c50)= soup_b | i=1094, i=1099 |
| 1110 | C5 | 最终交付选 soup_a 而不是单次分数最高的 v2ep1:判据从'峰值最高'改成'跨服务栈方差最小'(soupA 每次 ≥0.92;v2ep1 是 0.9267/0.9067) | i=1108, i=1109, i=1110 |
| 1133 | C6 | greedy-soup 再加一路 v3(soup_c),被默认栈评测否掉(0.9067) | i=1133, i=1138 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 419 | real | 0.01h | superseded | **both** | baseline —— 本 run 第一次训练。数据 work/sft_data_v1.jsonl(21,990 行,round-1 拒绝采样,每题 ≤3 解,20% 带评测 system message),超参 2 epoch / lr 2e-5 / bs 4 / accum 4 / packin… |
| 436 | real | 0.00h | superseded | **unclear** | 意图是相对 i=419 只改内存形状(bs 4→8、accum 4→2、加 --grad-ckpt),数据逐字相同。但这条命令以 `pkill -f train_sft.py` 开头,pkill 匹配到自己的 shell,整条以 exit 144 结束——训练进程从未启动(i=439 的 ps 为空… |
| 441 | real | 0.01h | superseded | **both** | 把 i=436 想做的事重发一遍(bs 8 / accum 2 / --grad-ckpt),这次真的起来了(pid 4930),第 1/556 步 OOM(要 14.27 GiB,vocab 151936 的 logits 上抛 fp32)。数据仍是 sft_data_v1.jsonl。属 419… |
| 454 | real | 0.82h | consumed | **both** | 相对 i=441 把 bs 8→4、accum 2→4,保留 --grad-ckpt;数据仍是 sft_data_v1.jsonl。这一段真的跑完:train_runtime 2910.9s(0.808h,机械层记 0.82h,比值 1.014),产出 work/sft_model_v1。至此 41… |
| 637 | real | 0.07h | superseded | **C3** | 相对 i=454 **只换数据**:sft_data_v1.jsonl(21,990 行)→ sft_data_v2.jsonl(47,116 行 = round-1 + SFT-v1 on-policy round-2 + 24,800 条 MetaMath 自蒸馏)。`--epochs 2 --… |
| 685 | real | 4.58h | consumed | **C3** | 与 i=637 的数据与超参逐字相同,仅新增 `--save-steps 599`。跑完,train_runtime 6353.67s = 1.765h(墙钟 1.84h,比值 1.043)。它与 i=454 构成本 run 唯一一次**超参逐字相同的 C3 对照**:同样自调栈 150 题贪婪,0… |
| 798 | real | 0.14h | superseded | **C4** | 第一次从 SFT 换成 GRPO:初始权重 work/grpo_input(= v2 的 epoch-1 检查点副本),二值正确性 reward,vLLM colocate,n=8,T=1.0,lr 2e-6,beta 0,250 步,prompt 集是 7473 道 train 全集。第 50 步… |
| 849 | real | 0.42h | last_seen | **C4** | 与 i=798 的命令行逐字相同;唯一差别在输入目录——i=837 把 work/grpo_input/generation_config.json 里的 temperature: 0.0 删掉了。跑完 250 步(约 25 分钟),产出 work/grpo_model,自调栈 0.9133,低于起… |
| 911 | real | 1.90h | consumed | **both** | 相对 i=849 同时改数据和方法:prompt 集从 7473 道全集缩到通过率 ≤5/6 的 1742 道(--idx-file work/grpo_band_idx.json,C3),num-generations 8→16、beta 0→0.01、lr 2e-6→1e-6、步数 250→20… |
| 1021 | real | 1.02h | consumed | **both** | 与 i=911 的数据(同一个 idx-file)、lr、beta、num-generations 逐字相同;只改内存形状与预算:新增 --per-device 8、vllm-mem 0.20→0.18、max-steps 200→150。跑完 150 步(约 0.69h,机械层记 1.02h),c… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 70 | 3 | 150 | 否 |  | 未拿到 |
| 77 | — | — | 否 | c1 | 未拿到 |
| 215 | 3 | 150 | 是 | c3 | 未拿到 |
| 282 | 3 | 150 | 是 | c3, c4 | 0.4066666666666667 |
| 1069 | 3 | 150 | 是 | c10, c14, c24 | 0.9066666666666666 |
| 1086 | 3 | 150 | 是 | c19, c29, c20, c21, c23, c25 | 0.9133333333333333 / 0.92 / 0.92 / 0.9133333333333333(一个事件里跑… |
| 1094 | 3 | 150 | 是 | c26 | 0.92 |
| 1102 | 3 | 150 | 是 | c21 | 0.92 |
| 1102 | 3 | 150 | 是 | c21 | 0.92 |
| 1102 | 3 | 150 | 是 | c19, c29 | 0.92 |
| 1102 | 3 | 150 | 是 | c19, c29 | 0.92 |
| 1113 | 3 | 150 | 是 | c21, c24, c27 | 0.9266666666666666 |
| 1133 | 3 | 150 | 是 | c28 | 0.9066666666666666 |
| 1141 | 3 | 150 | 是 | c21, c27 | 0.92 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[436]
- **2 次验证没有拿到信号**:i=[70, 77]
- **分类学缺口提案 2 条**
  - runtime_unblock(i=441, i=834, i=837, i=1021, i=270)
  - verifier_stack_recalibration(i=1083, i=1085, i=1086, i=1108, i=1109)
- **定义缺陷 10 条**
  - (i=436, i=437, i=438, i=439)
  - (i=936, i=975, i=992, i=992, i=623, i=626)
  - (i=750, i=753, i=1005, i=933, i=925)
  - (i=550, i=556, i=761, i=766, i=1054, i=1086, i=1091)
  - (i=77, i=80, i=79)
  - (i=108, i=158, i=249, i=253, i=260, i=317)
  - (i=792, i=792, i=793)
  - (i=550, i=556, i=1110, i=1111, i=339, i=945)
  - (i=1032, i=1032, i=1033, i=1111)
  - (i=543, i=547, i=1149)
- **边界情形 5 条**
  - (i=419, i=441, i=451, i=454, i=849)
  - (i=436, i=437, i=439)
  - (i=885, i=906, i=909, i=910)
  - (i=792, i=793, i=1032, i=1033)
  - (i=1083, i=1108, i=1109, i=1110)

## claude_non_api_claude-opus-4-7_10h__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17123798
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 7.29h | 0.5822592873… |

### 改动序列(14 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 109 | C3 | v1 数据配方:从 OpenMathInstruct-2 里筛出 problem_source in {gsm8k, augmented_gsm8k},按长度过滤后抽 150,000 条 augmented + 50,000 条 gsm8k,再拼上 GSM8K 官方 train 全部 7,473 条… | i=104, i=109, i=112, i=112 |
| 109 | C2 | 训练样本的 user 侧提示逐字抄评测的 GSM8K prompt(含 'ANSWER: $ANSWER' 的两段说明与结尾的 'Reasoning:'),assistant 侧把 \boxed{X} 剥掉并统一改写成结尾一行 'ANSWER: X'——与 inspect_evals 的 match… | i=109, i=109, i=112, i=75 |
| 120 | C4 | 训练方法定型:全参 SFT(非 LoRA),bf16 + flash_attention_2 + gradient checkpointing,只对 assistant 段算 loss(按 '<\|im_start\|>assistant' 标记做 label mask),cosine 学习率 + … | i=120, i=120, i=120, i=418 |
| 120 | C1 | 解码相关配置:在训练脚本里把 tokenizer.eos_token 设成 <\|im_end\|>、pad_token 设成 <\|end_of_text\|>,并直接改 model.config.eos_token_id;Trainer 保存时把这些值写进产出目录的 generation_con… | i=120, i=120, i=151 |
| 126 | C2 | 格式对齐修正:apply_chat_template 的 enable_thinking 由 False 改成 True(与评测默认的 /think 模式一致),label mask 的锚点标记同步从 '<\|im_start\|>assistant\n<think>\n\n</think>\n' … | i=126, i=126, i=125, i=123 |
| 163 | C4 | v1 超参定型:lr 1e-5、bs 16、grad_accum 2(等效 32)、max_len 896、1 epoch、warmup_ratio 0.03、max_samples 150000、save_steps 2000。max_len/bs 来自 i=154 的吞吐实测(17.7 samp… | i=163, i=160 |
| 244 | C4 | v1 被静默杀死后的重启形态:max_samples 150000→100000、save_steps 2000→500、logging_steps 50→25,并改用 nohup 后台启动;数据文件与 lr/bs/grad_accum/max_len/warmup 逐字不变。动机是防再次丢失,不是… | i=241, i=244 |
| 319 | C3 | v2 数据:把 train.jsonl 的第 100000 行之后切出来成 train_v2.jsonl(107,473 条),即 v1 训练时没见过的那一半,数据源与格式与 v1 完全同族。 | i=319, i=320 |
| 321 | C4 | v2 方法:不从 base 起训,而是 --model 指向已训好的 sft_v1 做续训,并把 lr 从 1e-5 降到 3e-6;bs/grad_accum/max_len/warmup 不变。 | i=321, i=321 |
| 330 | C5 | 候选提交(保底):v2 还在训的时候先把 sft_v1 整份 cp 进 final_model,保证任何时刻 final_model 里都是一个已测过 56% 的模型。 | i=329, i=330 |
| 376 | C3 | v3 数据配方:换成混合来源——OMI2 子集 100,000 条(80k augmented + 20k gsm8k)+ MetaMathQA 里 type 以 GSM 开头的 4 个子集抽 100,000 条(经 fix_metamath 把 'The answer is: X' 归一成 'AN… | i=374, i=374, i=377, i=377, i=377, i=377 |
| 379 | C4 | v3 方法:退回从 base 起训(命令里没有 --model),lr 回到 1e-5,max_samples 100000→200000,save_steps 500→1500;bs/grad_accum/max_len/warmup 与 v1 逐字相同。 | i=379, i=379 |
| 449 | C5 | 候选提交(定稿):用 sft_v3 覆盖 final_model 里的 sft_v1,依据是 150 题局部评测 0.587 > 0.56。两个候选是两次不同训练的终点模型,不是同一次训练的不同 checkpoint。 | i=448, i=449 |
| 463 | proposed:cross_run_knowl… | run 收尾时把配方、超参、三次评测数字与踩坑写进 ~/.claude/projects/-home-ben-task/memory/MEMORY.md 与 gsm8k_training.md。不改任何产物、对本 run 分数零影响,面向后续 run;C1–C7 没有能装下它的类别。 | i=458, i=463 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 143 | smoke | 0.00h | returned | **unclear** | baseline;本 run 第一次训练启动,--epochs 0.005 --max_samples 500 的冒烟,验的是 train_sft.py 跑不跑得通(第二档),既不在验 C3 也不在验 C4。harness 层以 Exit code 144 失败,没有任何训练输出。 |
| 145 | smoke | 0.00h | returned | **unclear** | 与 i=143 的命令逐字相同,只把 `\| tail -60` 改成 `\| tail -80`;仍以 Exit code 144 失败。受测的是管线可运行性,不是 C3/C4。 |
| 147 | smoke | 0.07h | last_seen | **unclear** | 与 i=145 同参数,只把输出从管道改成重定向到 /tmp/train_test.log 以绕开 Exit code 144。这次跑通:train_runtime 15.2769 秒、train_loss 0.536、权重落盘成功。受测的仍是管线可运行性。 |
| 154 | smoke | 0.02h | returned | **C4** | 相对 i=147:bs 4→16、grad_accum 2→1、max_len 默认 1024→896、max_samples 500→1024、epochs 0.005→1.0。目的是量吞吐以定真实训练规模,读到 17.7 samp/s 后直接决定 v1 用 150k 样本。受测变量是 C4 的 … |
| 163 | real | 2.45h | consumed | **both** | baseline —— 第一次真实训练,同时确定 v1 的数据配方(c1/c2)与超参(c3/c6),两者绑在一次训练里,轨迹里没有把它们分开的对照。实际结局与机械层记录不符:该命令被 harness 自动转入后台(i=165 返回 'Command running in background wi… |
| 244 | real | 1.41h | consumed | **both** | i=163 被杀后的重启:max_samples 150000→100000、save_steps 2000→500、logging_steps 50→25、改用 nohup;数据与 lr/bs/grad_accum/max_len/warmup 逐字不变。因为 i=163 一个分数都没产出,这一次… |
| 321 | real | 1.35h | last_seen | **both** | 相对 i=244 同时动了两类:C3——数据换成 train_v2.jsonl(train.jsonl 第 100000 行之后的 107,473 条未见样本);C4——初始权重从 base 换成 --model .../sft_v1 的续训、lr 1e-5→3e-6。bs/grad_accum/m… |
| 379 | real | 2.69h | consumed | **both** | 相对 i=321:C3——数据换成 train_v3.jsonl(OMI2 100k + MetaMathQA GSM 100k 的混合来源);C4——回到从 base 起训、lr 3e-6→1e-5、max_samples→200000、save_steps 500→1500。结局:正常跑完 62… |

### 验证序列(6 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 100 | 3 | 50 | 是 |  | 0.22 |
| 111 | — | — | 否 |  | 未拿到(该事件不是评测) |
| 292 | 3 | 150 | 否 | c1, c2, c3, c4, c5, c6, c7 | 0.56 |
| 364 | 3 | 150 | 是 | c8, c9 | 0.5533333333333333 |
| 439 | 3 | 150 | 是 | c11, c12 | 0.5866666666666667 |
| 453 | 3 | 150 | 是 | c13 | 0.5733333333333334 |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[143, 145, 147]
- **2 次验证没有拿到信号**:i=[111, 292]
- **分类学缺口提案 1 条**
  - cross_run_knowledge_capture(i=458, i=463)
- **定义缺陷 5 条**
  - (i=165, i=224, i=221, i=241)
  - (i=321, i=321, i=330)
  - (i=111, i=112, i=376, i=377)
  - (i=120, i=120, i=151)
  - (i=126, i=125)
- **边界情形 3 条**
  - 冒烟训练的 tested_variable 无值可取。i=143/145/147 三次的受测对象是『train_sft.py 跑不跑得通』(reference §2 第二档),既不是 C3 也不是 C4;spec §4.2 的四值域只有 C3/C4/both/unclear,只能退成 unclear,于是『schema 缺一个取值』被混进了『证据不足』的比例里。建议给 trainings 加一个 …(i=143, i=147)
  - C5 的定义是『在已训好的若干步数里挑一个交』——同一次训练的不同 checkpoint。本 run 的两次候选选择(i=330 交 sft_v1、i=449 换成 sft_v3)是在**两次不同训练的终点模型**之间挑,配方与超参都不同。按现定义既不是 C5(不是步数选择),也不属于 C1–C4/C6/C7(没有改动任何东西,只是选)。它和 C5 共享『零训练成本的纯选择动作』这个性质,但混进 …(i=330, i=449)
  - 一次写入同时承载两类。i=109 的 prepare_data.py 一份文件里既定了数据来源与配比(C3),又定了逐字对齐评测的 prompt 与 'ANSWER: X' 答案格式(C2)。category 是单值字段:拆成两条 change 会让两条共用同一个锚点 i(本标注就是这么做的,c1/c2),不拆则必须丢掉一类。同样的问题出现在 i=376 的 prepare_data_v3.py(…(i=112, i=112)

## claude_non_api_claude-opus-4-7_10h_run2__gsm8k_google_gemma-3-4b-pt_17133447
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | google_gemma-3-4b-pt | 8.56h | 0.6535253980… |

### 改动序列(18 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 52 | C2 | 把 GSM8K 官方 train 7,473 题重排成评测 prompt 的逐字格式:抄 inspect_evals/gsm8k 的原话提示词、去掉 <<...>> 计算器标记、答案统一成 'ANSWER: <数字>' 结尾。 | i=52, i=53, i=53 |
| 83 | C4 | 训练脚本改用 Gemma3ForConditionalGeneration 加载,并冻结 vision_tower 与 multi_modal_projector,只全参训练文本侧(不是 LoRA)。 | i=83, i=79 |
| 96 | C4 | v1 超参:自 base 起全参 SFT,3 epoch / bs16 / ga2 / lr1e-5 / max_len512。 | i=96 |
| 118 | proposed:artifact-servin… | 从 base snapshot 复制 preprocessor_config.json 与 processor_config.json 进 checkpoint 目录,修复 vLLM 无法加载自训权重导致评测在启动阶段就崩的问题。此后每次评测前都重做一遍(i=155/185/226/253/270)… | i=118, i=113 |
| 141 | C3 | 训练数据换来源:MetaMathQA 的 GSM_* 子集 240,000 条(经 'The answer is:' 正则 + float() 过滤)+ GSM8K 官方 train 7,473 条 = 247,473 条。选它的理由是 WebSearch 确认 MetaMathQA 只由 GSM8… | i=141, i=142, i=142 |
| 147 | C4 | v2 超参:仍自 base 起(未接 v1),epochs 3.0→1.0,max_len 512→768(由 i=145 的 token 长度分布 p99=646 决定),bs16/ga2/lr1e-5 与 v1 相同。 | i=147, i=145 |
| 157 | C5 | 把 final_model 指向 sft_v1 作为兜底提交(此时 v2 还在训)。 | i=157, i=157 |
| 216 | C2 | 把 3 条 few-shot 上下文嵌进训练样本的 user turn,复刻评测 prompt 的结构(题干 + Reasoning + ANSWER 重复三遍再接目标题),目的是教模型答完一题就停,不再顺着 few-shot 模式继续生成。诊断来自 i=206/i=210 两次手工 generat… | i=216, i=213, i=217 |
| 221 | C4 | v3 超参:改为接续 sft_v2 而非从 base 起,2 epoch / bs8 / ga4 / lr5e-6 / max_len1280。 | i=221, i=221 |
| 231 | C5 | final_model 从 sft_v1 改指 sft_v3(49.3% > 36%)。 | i=231 |
| 238 | C3 | 把 few-shot 格式扩到大数据:MetaMathQA GSM_* 与 GSM8K train 混合、每条配 3 条 GSM8K-train few-shot,生成 60,000 条。 | i=238, i=239 |
| 243 | C3 | 为时间预算把 60,000 条截成前 40,000 条作为 v4 训练集。 | i=243, i=244 |
| 245 | C4 | v4 超参:接续 sft_v3,epochs 2.0→1.0,bs8/ga4/lr5e-6/max_len1280 与 v3 逐字相同。 | i=245, i=245 |
| 258 | C5 | final_model 从 sft_v3 改指 sft_v4(66% > 49.3%)。 | i=258, i=259 |
| 262 | C3 | v5 数据:取 fewshot_metamath 的后 20,000 条(与 v4 用的 head-40000 不相交,即全新样本)再拼上全部 7,473 条 gsm8k_fewshot,shuf 后共 27,473 条。 | i=262, i=263 |
| 264 | C4 | v5 超参:接续 sft_v4,lr 5e-6→3e-6,bs8/ga4/max_len1280/1 epoch 与 v4 相同。 | i=264 |
| 276 | C5 | 否掉 v5(58% < 66%),确认 final_model 仍指 sft_v4,即最终提交 v4。 | i=275, i=276, i=277 |
| 281 | proposed:artifact-servin… | 把 final_model 从符号链接实体化成真实目录拷贝,避免提交链路解不开链接。 | i=281, i=281 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 89 | smoke | 0.01h | returned | **unclear** | baseline(本 run 第一次训练启动)。timeout 300 前台冒烟,--epochs 0.02、bs4、ga1,数据用默认 gsm8k_train.jsonl。意图是验 i=81/i=83 两处 Edit 之后训练脚本跑不跑得通(i=85:'to catch errors'),不在检验… |
| 96 | real | 0.29h | consumed | **both** | baseline(第一次真实训练)。数据 = GSM8K 官方 train 7,473 条按评测模板重排(c1);超参 = 自 base 全参 SFT、3 epoch / bs16 / ga2 / lr1e-5 / max_len512(c2+c3)。数据来源与方法超参在此同时首次设定、无任何对照臂… |
| 147 | real | 4.14h | consumed | **both** | vs v1:数据从 GSM8K train 7,473 条换成 MetaMathQA GSM_* 240,000 + GSM8K train 7,473 = 247,473 条(c5,C3 来源 (b) 公开蒸馏数据集);同时 epochs 3.0→1.0、max_len 512→768(c6,C4… |
| 221 | real | 0.49h | consumed | **unclear** | vs v2:初始化从 base 改为接续 sft_v2;数据从 247,473 条 metamath_gsm 换成 7,473 条 gsm8k_fewshot —— 题目仍是同一批 GSM8K train,变的只是把 3 条 few-shot 上下文嵌进 user turn 以对齐评测 prompt… |
| 245 | real | 1.36h | consumed | **C3** | vs v3:接续 sft_v3;数据从 7,473 条 gsm8k_fewshot 换成 40,000 条 fewshot_metamath_40k(MetaMathQA GSM_* + GSM8K train 混合,同为 3-shot 格式);bs8 / ga4 / lr5e-6 / max_le… |
| 264 | real | 0.98h | consumed | **both** | vs v4:接续 sft_v4;数据换成 v5_data.jsonl 27,473 条 = fewshot_metamath 的后 20,000 条(与 v4 的 head-40000 不相交,全新样本)+ 全部 7,473 条 gsm8k_fewshot,shuf 过(C3);同时 lr 5e-6… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 44 | 3 | 50 | 是 |  | 0.0 |
| 110 | 3 | 150 | 否 | c3 | 未拿到 —— 这次评测在 inspect_ai 的 eval_init 阶段就抛了 RuntimeError: Fail… |
| 120 | 3 | 150 | 是 | c1, c2, c3, c4 | 0.36 |
| 185 | 3 | 150 | 是 | c5, c6 | 0.02666666666666667 |
| 226 | 3 | 150 | 是 | c8, c9 | 0.49333333333333335 |
| 253 | 3 | 150 | 是 | c11, c12, c13 | 0.66 |
| 270 | 3 | 150 | 是 | c15, c16 | 0.58 |
| 284 | 4 | -1 | 是 | c17, c18 | 0.6557998483699773 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[89, 221]
- **1 次验证没有拿到信号**:i=[110]
- **分类学缺口提案 1 条**
  - artifact-serving-repair(i=113, i=118, i=281)
- **定义缺陷 4 条**
  - 同一条 run 里两次评测共用一个 --json-output-file 时,后一次的分数会被回填给前一次,把一次崩溃的评测记成'拿到分数'。(i=110, i=113, i=120)
  - 训练预算也会服务 C2。这条 run 有一次完整训练(0.44h)检验的是一个格式对齐假设,而取值域装不下 C2,该行只能落进 unclear,时间在统计上要么消失要么被错摊给 C3/C4。(i=213, i=217, i=221)
  - _FINALIZER 的候选词里含 prepare,于是把训练**数据**准备脚本的调用全部误记成 generation_config.json 的写入。这条 run 上 5 行里 4 行是假阳性。(i=52, i=53, i=238, i=239)
  - 读侧被漏算了:cat/head 形式的访问,内容机械可得,它就在配对的 tool_result 里,只是 config_writes 只看命令不看结果。(i=196, i=197)
- **边界情形 3 条**
  - smoke-has-no-tested-variable(i=89, i=85)
  - cross-recipe-candidate-selection-vs-C5(i=60, i=157, i=258)
  - passive-hyperparam-adjustment(i=221, i=245, i=145)

## claude_non_api_max_claude-opus-4-8_10h_run1__gsm8k_Qwen_Qwen3-1.7B-Base_17315721
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 10.08h | 0.6118271417… |

### 改动序列(14 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 111 | C2 | 写 scripts/prep_data.py：把 SFT 样本渲染成与 inspect_evals gsm8k 逐字相同的 prompt（同一段 MATH_PROMPT_TEMPLATE + templates/qwen3.jinja + add_generation_prompt=True），co… | i=47, i=116 |
| 113 | C3 | 训练数据来源定为 GSM8K 官方 train split（7,473 条人写解答），不引入外部语料 | i=113, i=116 |
| 131 | C4 | 写 scripts/train_sft.py，确定初始训练方法：**全参微调**（非 LoRA）、completion-only loss（prompt 段 label 置 -100）、bf16 + flash_attention_2、max-len 1536（丢弃 >1536 的样本）、cosin… | i=123, i=130 |
| 135 | C5 | 改造 train_sft.py：每个 epoch 落一个 checkpoint 供事后扫描选最优，并把 EOS/解码修复从训练脚本里拆出来交给 fix_model.py（使任意 checkpoint 都可评） | i=134 |
| 137 | C1 | 写 scripts/fix_model.py，对每个待评 checkpoint 的 generation_config.json 做字段级合并（不是整份重写）：eos_token_id 151643→151645(<\|im_end\|>)、新增 pad_token_id 151643、do_sam… | i=53, i=106, i=484 |
| 141 | C4 | 删掉与 TrainingArguments(gradient_checkpointing=True) 重复的 model.gradient_checkpointing_enable() 调用，避免冲突 | i=140 |
| 193 | C3 | 配方换成 GSM8K train ×3 上采样（22,419）+ MetaMathQA 的 GSM 子集 80,000，合计 102,419 行（MetaMath 衍生自 GSM8K/MATH train，agent 自判合规） | i=193, i=208 |
| 268 | C4 | 给 train_sft.py 加 --group-by-length 开关（按序列长度分桶，提高 102K 大数据集的吞吐） | i=267 |
| 309 | C4 | iteration-2 超参：epochs 3→2、bs 16→32、accum 2→1、启用 group_by_length（lr 保持 2e-5） | i=309 |
| 358 | C4 | OOM 后回退超参：bs 32→16、accum 1→2（有效 batch 仍为 32），并设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True | i=350, i=358 |
| 398 | C4 | 引入 GRPO RL（TRL 0.27.2，use_vllm=True + vllm_mode=colocate，num_generations 8，lr 1e-6，beta 0，scale_rewards），奖励 = correctness（取最后一个数字与 gold 比）+ 0.2 格式奖励 | i=396, i=522 |
| 483 | C5 | 把 runs/sft2（epoch-2，limit-200 上 63%）整目录拷进 final_model 作为 safety net；轨迹结束前再没更新过 final_model，官方最终分 0.6118 就来自这一步（而同时 runs/grpo1/checkpoint-200 在 limit-2… | i=482, i=483 |
| 526 | C5 | 把 GRPO 存档频率从 save_steps 100 / save_total_limit 4 改成 save_steps 50 / save_total_limit 8，为 checkpoint 扫描准备更多候选 | i=525 |
| 531 | C4 | GRPO 全量运行参数：从 runs/sft2 权重起训、steps 250、bs 16、accum 8、num-gen 8（相对冒烟的 3 步 / bs 8 / accum 1 放大） | i=529, i=531 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 147 | real | 0.34h | consumed | **both** | baseline（首次训练，无前序可比）。它同时第一次确定了数据来源（GSM8K train 7,473，c3）与方法/超参（全参 SFT、3 epoch、lr 2e-5、bs 16/accum 2、max-len 1536、completion-only loss）。真实结局：正常跑完，train… |
| 309 | real | 0.06h | superseded | **both** | 相对 i=147：数据 7,473→102,419 行（GSM8K×3 + MetaMath 80K，c4）**且** 超参 epochs 3→2、bs 16→32、accum 2→1、加 group_by_length（c7/c8）——两类同时变。真实结局：**不是被 superseded，是第一… |
| 351 | real | 0.01h | superseded | **C4** | 相对 i=309 只改 bs 32→16、accum 1→2 并加 expandable_segments（c9，纯 C4）。真实结局：**这次启动根本没有发生** —— 命令整体 exit 1（i=352），i=355 查到进程列表为空、GPU 0 MiB，agent 自己在 i=357 写明重启… |
| 358 | real | 1.62h | consumed | **both** | 相对直接前身 i=351（未启动）/ i=309（OOM）：仅 bs 32→16、accum 1→2、加 expandable_segments（C4）。但相对上一次**跑完的**训练 i=147，数据与超参两类同时变了，而 agent 就是用这一对（46%→63%）作判断并归因给 MetaMath… |
| 487 | smoke | 0.06h | discarded | **C4** | 首次 GRPO：方法从全参 SFT 换成 GRPO RL（c11），3 步冒烟，bs 8/accum 1/num-gen 8。**它验的是“GRPO 管线能不能跑”而不是 C4 的效应**（见 definition_defect dd7）。真实结局：3 步全部跑完（train_runtime 26.… |
| 531 | real | 0.73h | consumed | **C4** | 相对 i=358（SFT）：方法换成 GRPO，起点是 runs/sft2 的权重；相对 i=487 冒烟：规模从 3 步/bs 8/accum 1 放大到 250 步/bs 16/accum 8，并把 save_steps 100→50（c12）。注：训练 prompt 同时从 combined_… |

### 验证序列(7 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 74 | 3 | 150 | 是 |  | 0.09333333333333334 |
| 259 | 3 | 200 | 是 | c1, c2, c3, c5, c6, c14 | 0.455 |
| 259 | 3 | 200 | 是 | c1, c2, c3, c5, c6, c14 | 0.455 |
| 259 | 3 | 200 | 是 | c5 | 0.46 |
| 259 | 3 | 200 | 是 | c5 | 0.46 |
| 450 | 3 | 200 | 是 | c4, c7, c8, c9 | 0.63 |
| 450 | 3 | 200 | 是 | c4, c7, c8, c9 | 0.63 |
| 450 | 3 | 200 | 是 | c5 | 0.61 |
| 450 | 3 | 200 | 是 | c5 | 0.61 |
| 592 | 3 | 200 | 否 | c11, c12, c13 | 五个 checkpoint 全部拿到：50=0.67 / 100=0.735 / 150=0.72 / 200=0.80… |
| 617 | 3 | 500 | 否 | c11, c13 | 未拿到。后台任务 bp0cg4qp2 到轨迹末尾（i=638，2026-06-08T00:54:15Z）仍未返回，lim… |

### 异常与存疑

- **2 次验证没有拿到信号**:i=[592, 617]
- **分类学缺口提案 2 条**
  - proposed:eval_harness_tooling(i=263, i=609)
  - proposed:run_continuity(i=416, i=482, i=616)
- **定义缺陷 7 条**
  - (i=113, i=193)
  - (i=296, i=480)
  - (i=613, i=614)
  - (i=355, i=357)
  - (i=259, i=260)
  - (i=91, i=94)
  - (i=486, i=531)
- **边界情形 4 条**
  - 方法切换强制带来数据形态切换。i=531 从 SFT 换成 GRPO 时，训练数据必然从 combined_sft.jsonl（102,419 条 prompt+completion、含 40% few-shot 混合）变成 GSM8K train 原始 7,473 条 zero-shot prompt+gold——因为 RL 根本不吃 completion。按“C3 = 决定训练数据从哪来”的字…(i=396, i=531)
  - GRPO 的 correctness_reward 是官方评分器的再实现（同样取回复里最后一个数字与 gold 做数值比较），在 250 步 × 每步 128 条 completion 上当判定信号用。C7 的定义是“因为第四档太贵，自己搭一个更便宜的评测代理”，而这个打分器的消费者是优化器而不是 agent；它既符合 C7 的功能形状（口径对齐的廉价代理评分器），又完全内嵌在 C4 的训练目标里…(i=47, i=522)
  - run_eval.sh 在每次评测前无条件执行 `python scripts/fix_model.py "$DIR"`，于是 C1 被绑进了**每一次**验证。后果有两层：（1）这条 run 里没有任何一次评测能把 C1 与其他改动分开，9.3%→45.5% 永远是 C1+C2+C3+C4 的合项；（2）run_eval.sh 本身算一次“改动”（它是 C1 的交付机制）还是算验证器工装，按现定…(i=609, i=74, i=259)
  - 已备好但从未执行的候选改动。agent 在等训练的空档里写完了整条 RFT/STaR 通路（i=217 scripts/gen_rft.py 拒绝采样生成 + i=318 scripts/prep_from_triples.py 三元组转训练格式），这正是 reference §C3 的来源 (d)「自生成 + 验证过滤」。两个脚本在 i=413 只被 `ast.parse` 语法检查过（i=41…(i=216, i=317, i=414)

## claude_non_api_claude-opus-5_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17415829
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 9.16h | 0.9120545868… |

### 改动序列(18 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 203 | C1 | 把 eos_token_id 设成 [151645, 151643](<\|im_end\|> 与 <\|endoftext\|> 都接受),config.json 的 eos 设成 151645,让生成停在 chat 结束符上。 | i=203, i=1164 |
| 226 | C3 | 用 prep_data.py 从 nvidia/OpenMathInstruct-2 建首版 SFT 语料(80,049 行:60k augmented_gsm8k + 7,385 gsm8k + math/augmented_math),把 \boxed{} 解答改写成以 'ANSWER: X' … | i=226, i=203 |
| 315 | C2 | 绕开 apply_chat_template,手工拼 '<\|im_start\|>user…<\|im_end\|>\n<\|im_start\|>assistant\n' + completion + <\|im_end\|>,使训练文本与评测时模型真实看到的串逐字一致(qwen3.jinja … | i=315, i=315 |
| 408 | C3 | 改配方:按题目去重(新增 --sols-per-problem),各源上限改成 8000/100000/8000/14000,产出 95,136 行(其中 73.6k 条不重复 augmented_gsm8k 题目)。 | i=408 |
| 574 | proposed:resource-fit | 把 SFT 的全量 logits 交叉熵换成 ChunkedLossTrainer(只在有标签位置上 gather,再分块 checkpoint 跑 lm_head),因为 bs=16 时 [16,2048,151936] logits 直接 OOM;先试 liger-kernel 但根 overl… | i=574, i=498 |
| 713 | C1 | 在提交模型目录的 generation_config.json 里写 temperature: 0.0(读 vllm/config/model.py 的 get_diff_sampling_param 确认 do_sample 被忽略、只认 temperature/top_p/top_k/min_p… | i=713, i=734, i=189, i=1195, i=8 |
| 810 | C5 | 在开始 RL 之前先把 SFT 贪婪版装成 final_model,作为随时可交的安全默认。 | i=810 |
| 814 | C3 | 用 SFT checkpoint 自采样 k=4 跑 20,473 道 gsm8k+augmented_gsm8k 题(81,892 次生成、13 分钟),按通过率筛出 6,531 条非饱和 prompt 作为 RL 训练集,并副产 19,453 条 on-policy 正确解。 | i=814, i=836 |
| 847 | C4 | 在 SFT 之后加一段 GRPO RL(dr_grpo、beta=0、scale_rewards=none、mask_truncated_completions、lr 1e-6、num_gen 8、colocate vLLM)。 | i=847 |
| 976 | C3 | 把 RL 提示里的 10-shot 前缀彻底去掉(--fewshot-frac 0.35 → 0.0),明确以「训练/评测提示不一致」换吞吐与显存。 | i=976, i=978 |
| 1012 | proposed:resource-fit | 为让 GRPO 在单张 H100 上跑起来:vllm_gpu_memory_utilization 0.30→0.25→0.20、加 torch_empty_cache_steps=4、micro-batch 8→4(并因与 sleep mode 冲突而去掉 PYTORCH_CUDA_ALLOC_C… | i=1012, i=1012 |
| 1054 | C5 | 把 GRPO1 的两个候选(终点 step-240 与中途 checkpoint-150)都 finalize 成可服务目录,准备正面对比选一个交。 | i=1054, i=1054 |
| 1161 | proposed:resource-fit | 改 finalize.py:不再直接复制 safetensors,而是以 bfloat16 重新 from_pretrained/save_pretrained,把 Trainer 存出的 fp32 checkpoint(15G)压回 7.6G,以便按 evaluate.py 默认 gpu-memo… | i=1161, i=1164 |
| 1168 | C4 | GRPO 第二轮:lr 1e-6 → 3e-6,起点从 ckpt/sft1 换成 grpo1 的 step-150 权重,seed 0 → 7;数据仍是同一份 data/hard.jsonl。 | i=1168 |
| 1232 | C1 | 整份重写 ckpt/grpo150_bf16/generation_config.json 为 {bos,eos,pad,do_sample:true,transformers_version},顺手删掉 temperature:0.0 / top_p:1.0 / top_k:0 三个字段——目的是… | i=1232, i=1232 |
| 1275 | proposed:resource-fit | 427G 卷写满(422G/100%)导致 checkpoint-160 只落盘 9.3G 而损坏;删掉 grpo1 全部 checkpoint 与 grpo2 的 checkpoint-40/160 回收磁盘——这一步直接从 C5 的候选池里移走了 GRPO2 的终点 checkpoint。 | i=1275, i=1266 |
| 1308 | C6 | 把 ckpt/sft1、ckpt/grpo150_bf16、ckpt/grpo2/checkpoint-120 三份权重均匀平均成 ckpt/soup(fp32 累加后转 bf16),再 finalize 成 eval_soup。 | i=1308 |
| 1373 | C5 | 全量评测失败、放弃大样本对比后,用 eval_soup 覆盖 final_model 作为最终提交(理由是降方差,而非它测得最高)。 | i=1373 |

### 训练序列(13 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 326 | smoke | 0.02h | returned | **unclear** | baseline(本 run 第一次训练启动)。600 样本 1 epoch 的冒烟,只验 train_sft.py 能跑通并落盘;不检验任何 C3/C4 变量。 |
| 447 | real | 0.11h | superseded | **both** | 相对 i=326 冒烟:换成完整 data/sft_big.jsonl(95,136 行,C3)+ 真实超参 --epochs 1 --lr 1e-5 --bs 16 --accum 4(C4,bs 8→16)。第一次真训练尝试,step 0 反向传播即 CUDA OOM(要 18.13 GiB),… |
| 504 | smoke | 0.01h | returned | **unclear** | 相对 i=447:train_sft.py 加了 use_liger_kernel=True 想解 OOM。2000 样本冒烟,验的是这个显存修法能不能跑,不是配方。实际连 import 都没过:OSError: [Errno 28] No space left on device(根 overla… |
| 581 | smoke | 0.05h | returned | **unclear** | 相对 i=504:撤掉 liger,改用自写 ChunkedLossTrainer。3000 样本冒烟,验它在 bs=16 下不 OOM 并量吞吐(138.4s / 2886 样本 ≈ 10k tok/s),据此决定全量只跑 1 epoch。仍不检验 C3/C4 效应。 |
| 591 | real | 1.26h | consumed | **both** | 命令行相对 i=447 逐字相同(同 data/sft_big.jsonl、同 --epochs 1 --lr 1e-5 --bs 16 --accum 4),只有脚本内部的 loss 实现从全量 logits 换成分块;即配方未变,只是这次真的跑完了(1441 步 / 73 分钟,train_ru… |
| 847 | real | 0.05h | superseded | **both** | 相对 i=591:换阶段。数据换成自筛的 data/hard.jsonl(6,531 条非饱和 prompt,C3),方法换成 GRPO(dr_grpo、lr 1e-6、num-gen 8、gen-batch 192、bs 8、max-completion 512、fewshot-frac 0.35… |
| 863 | real | 0.06h | superseded | **unclear** | 与 i=847 命令逐字相同(仅脚本里加了 as_text() 把会话式 completion 摊平)。纯崩溃重启,不检验任何变量。1 分 48 秒后在 optimizer.step 的 exp_avg_sq 分配上 OOM。 |
| 888 | real | 0.01h | superseded | **unclear** | 与 i=863 逐字相同,只有 --bs 8 → 4;脚本里 vllm_enable_sleep_mode False→True、vllm_gpu_memory_utilization 0.28→0.30。纯显存重启。22 秒后崩在 vLLM 的 AssertionError:expandable_… |
| 894 | real | 0.48h | superseded | **unclear** | 与 i=888 逐字相同,只去掉了前缀 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。纯环境重启。跑到第 14 步(115 s/step)后在 backward 里 OOM,无 checkpoint 落盘。 |
| 978 | real | 0.07h | superseded | **unclear** | 相对 i=894 改了 5 个量:steps 260→240、gen-batch 192→128、bs 4→8、max-completion 512→448、fewshot-frac 0.35→0.0,外加 vllm util 0.30→0.25。理由全是墙钟与显存(115 s/step 太慢),不… |
| 1012 | real | 1.64h | consumed | **both** | 与 i=978 逐字相同,只有 --bs 8 → 4;脚本里 vllm util 0.25→0.20、加 torch_empty_cache_steps=4。启动动机是 OOM,但这是 GRPO1 六次启动里唯一跑完的一次(240 步 / 1.64h,22.5 s/step),产物 ckpt/grp… |
| 1168 | real | 0.38h | discarded | **C4** | 相对 i=1012:数据逐字相同(同一份 data/hard.jsonl),只改方法侧——起点 ckpt/sft1 → eval_grpo150(grpo1 的 step-150 权重)、lr 1e-6 → 3e-6、steps 240 → 180、seed 0 → 7。第 50 步存盘时崩:Val… |
| 1236 | real | 1.21h | consumed | **C4** | 与 i=1168 同一个问题的重跑:同数据、同 lr 3e-6、同 seed 7,起点换成 ckpt/grpo150_bf16(与 eval_grpo150 逐字节同权重,只是 generation_config.json 被重写过以通过校验),steps 180→160、save_steps 50… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 85 | 3 | 200 | 是 |  | 0.465 |
| 734 | 3 | 200 | 是 | c1, c2, c3, c5, c6 | 0.93 |
| 747 | 3 | 200 | 是 | c5 | 0.855 |
| 1072 | 3 | 300 | 是 | c8, c12 | 0.9233333333333333 |
| 1087 | 3 | 300 | 是 | c1, c3, c8, c12 | 两个分数:eval_sft1_greedy=0.92,eval_grpo150=0.9366666666666666(骨… |
| 1087 | 3 | 300 | 是 | c1, c3, c8, c12 | 两个分数:eval_sft1_greedy=0.92,eval_grpo150=0.9366666666666666(骨… |
| 1277 | 3 | 300 | 是 | c12, c14 | 0.93 |
| 1281 | 3 | 300 | 是 | c12, c14 | 0.8766666666666667 |
| 1308 | 3 | 300 | 是 | c16 | 0.93 |
| 1327 | 4 | -1 | 否 | c8, c12 | 未拿到。13:34:29 启动,90 分钟后撞 harness 5400s bash 超时转后台;agent 连查三条通… |
| 1375 | 3 | 600 | 否 | c16 | 未拿到。15:05:13 启动(此时卡是空的,i=1373 读到 0 MiB),45 分钟后转后台;logs/soup_… |
| 1420 | 3 | 150 | 是 | c5, c16, c18 | 0.9133333333333333 |
| 1442 | 3 | 800 | 否 | c16 | 未拿到。timeout 1500 触发,exit=124,logs/soup_800.json 从未生成;agent 没… |
| 1457 | 3 | 500 | 否 | c8, c12 | 未拿到。timeout 900 触发,exit=124,logs/grpo150_500.json 从未生成;此后 ag… |
| 1496 | 3 | 150 | 是 | c5, c16, c18 | 0.9266666666666666(与 i=1420 同权重、同 150 题、同贪婪解码,只差 --max-conne… |

### 异常与存疑

- **7 段训练的受测变量判不出**:i=[326, 504, 581, 863, 888, 894, 978]
- **4 次验证没有拿到信号**:i=[1327, 1375, 1442, 1457]
- **分类学缺口提案 1 条**
  - resource-fit(i=574, i=1012, i=1275, i=498)
- **定义缺陷 8 条**
  - (i=1087, i=1090)
  - (i=727, i=1327)
  - (i=1512)
  - (i=734, i=1054)
  - (i=1232)
  - (i=888, i=863, i=870)
  - (i=85, i=86)
  - (i=1012, i=1275)
- **边界情形 4 条**
  - 同一次修改可以读成三类而没有一类占优:(a) C3——它改的是 RL 训练数据的构成;(b) 反向的 C2——评测永远是 10-shot,去掉前缀让训练提示离评测提示更远,是主动放弃格式对齐;(c) resource-fit——agent 自陈理由是 115 s/step 太慢且长序列 OOM。现定义要求单选一个 category,选哪个都会丢掉另外两层。我按 C3 记,但这条判定不稳。(i=976, i=978)
  - C1 的定义是「修改提交模型目录里的 generation_config.json(temperature / eos / 惩罚项)」,可用验证器是第一档+第三档。这次改的是 ckpt/grpo150_bf16——一份将要被 GRPO2 读进去当初始权重的目录——目的是通过 HF 的 GenerationConfig 校验让训练能启动,对任何一次评测的解码行为都没有影响,也从来没有被任何验证器判定…(i=1232, i=1175)
  - 脚本里的 last_number() 逐条镜像了官方 inspect match(numeric=True, location='end') 的抽取规则(先 re.sub 掉 $,£€*_,再删非数字前的 .,再从右往左取第一个数值 token),同一份实现又被 train_grpo.py 的 reward_correct 用作训练奖励。C7 的定义是「自己搭一个更便宜的评测代理」去替代第四档判定…(i=814, i=836)
  - C5 写的是「在已训好的若干步数里挑一个交」,指的是同一次训练的不同 step。这两次动作挑的是跨阶段、甚至跨合并方式的产物(SFT 贪婪版 → soup),而且 i=810 的动机是「先放一个随时可交的安全默认」,属于提交风险管理而非候选择优。reference §4.4 已经把「提交流水线缺回归守卫」列成一类扣分项,说明这条轴存在,但 C1–C7 里没有它的格子。我按 C5 记。(i=810, i=1373)

## claude_non_api_claude-opus-5_10h_run2__gsm8k_Qwen_Qwen3-1.7B-Base_17418513
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 9.23h | 0.8241091736… |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 283 | C3 | 选定训练数据来源:本地缓存的 nvidia/OpenMathInstruct-2,每题最多 2 个解,目标 40 万 gsm8k 系 + 6 万 MATH 系;实得 gsm 160537 + math 60000 = 220537 行。 | i=283, i=346 |
| 299 | C1 | 把 base 权重符号链接成 work/base_greedy,并整份重写 generation_config.json 为 {bos,eos,temperature:0.0,transformers_version} —— 相对 base 快照删掉了 do_sample:false 与 max_n… | i=299, i=203, i=280 |
| 359 | C2 | 格式对齐:逐字抄评测的 MATH_PROMPT_TEMPLATE,把 \boxed{} 改写成 ANSWER: X 结尾并手工拼 <\|im_end\|>;prompt 用评测同一份 templates/qwen3.jinja 渲染且只渲染到 assistant 头(绕开模板对真实 assistan… | i=359, i=362, i=362 |
| 398 | C4 | 吞吐调参:把 gradient_checkpointing 改成环境变量开关并关掉它,bs 16→32、accum 4→2 试跑 —— 结果 CUDA OOM。 | i=398, i=401 |
| 413 | C4 | 装 liger-kernel 0.8.1 并加 use_liger_kernel 开关,想用 fused linear cross-entropy 压 logits 显存;GC 仍关着,结果照样 OOM。 | i=413, i=412, i=416 |
| 429 | C4 | 回到 GC=1 并把 bs 提到 64、accum 降到 1、样本 6000,量到 35.9 samples/s(train_runtime 167s),定下正式 SFT 的批大小。 | i=429, i=450 |
| 488 | C2 | few-shot 注入概率 0.5→0.35、k 集合收窄,并新增 1800 token 长度过滤:超长样本整条丢弃而不是截断(避免学到被截断的答案)。220537 → 206785 行,平均 586 token。 | i=488, i=491, i=485 |
| 501 | C4 | 正式 SFT 的方法与超参:全参 TRL SFTTrainer,completion-only loss、padding_free、bf16 + FlashAttention-2 + liger,1 epoch、bs 64、accum 2、lr 1.5e-5、maxlen 1800。 | i=501, i=500 |
| 648 | C4 | 从 SFT 切到 GRPO:TRL GRPOTrainer、dr_grpo、beta=0、vLLM colocate、num_gen 8、gen_batch 256、lr 1e-6、220 步;reward 直接 import 评测自己的 match_str 作为正确性打分。 | i=648, i=1370 |
| 715 | C4 | 把 sft_v1 的 eos_token 从 <\|endoftext\|>(151643)改成 <\|im_end\|>(151645),同时改 tokenizer_config.json / special_tokens_map.json / config.json。原因是 TRL 按 toke… | i=715, i=702, i=718 |
| 822 | C4 | GRPO 第二版:lr 1e-6→3e-6、max_completion_length 600→480、save_steps 40→30;依据是 40 步 reward 平坦、grad_norm 只有 0.07。 | i=822, i=817, i=822 |
| 894 | proposed:trainer-compat-… | 为绕开 save_pretrained 的 GenerationConfig 校验(do_sample=False 与 temperature=0.0 不能共存),把 sft_v1/generation_config.json 的 temperature 弹掉、写 do_sample:False;同… | i=894, i=873, i=895 |
| 1027 | C1 | 对 work/rl_v2/checkpoint-30 与 -60 只改 generation_config.json:弹掉 do_sample、写入 temperature:0.0。改后内容逐字为 {'eos_token_id': [151645, 151643], 'pad_token_id': … | i=1027, i=1028, i=1025 |
| 1055 | C5 | checkpoint 选择:把 rl_v2/checkpoint-60 提升为 final_model(顶掉此前的 sft_v1 副本),依据是 i=1029 读到的 0.847 vs sft_v1 的 0.822。注意 cp -r 顺带让 final_model 的 generation_conf… | i=1055, i=1056 |
| 1057 | C4 | GRPO 起点从 sft_v1 换成 rl_start60(= rl_v2/checkpoint-60 的副本),再续训 170 步,其余超参与 c11 相同。 | i=1057, i=1054 |
| 1076 | C3 | 备好第二批与第一批不重叠的 OMI-2 切片(prep_data2.py + build_sft2.py → sft_raw2/sft_pc2),作为 SFT round 2 的候选数据;最终因时间不够从未训练。 | i=1076, i=980, i=1370 |
| 1132 | C4 | 崩溃修复兼配方微调:vllm_max_model_length 2560→4096(rl_v3 就是被 2766 token 的 10-shot prompt 打死的)、few-shot k 上限 10→8、save_steps 30→25。 | i=1132, i=1135, i=1132 |
| 1155 | C4 | rl_v4:把 random.seed 改成读 DSEED 环境变量并传 91(换掉 prompt 打乱顺序 → 在 max_steps=150 截断下等于换掉看到哪批 prompt),同时 lr 3e-6→2e-6、rollout temperature 1.0→0.9、150 步。数据侧与方法侧… | i=1155, i=1155, i=1154 |
| 1224 | C1 | 对 rl_v4/checkpoint-75 与 -125 同样弹 do_sample、写 temperature:0.0,让中途 checkpoint 与顶层 rl_v4 目录(脚本已写 temperature:0.0)同口径可比。 | i=1224, i=1210 |
| 1242 | C5 | 定稿:把 rl_v4(150 步跑满)整份拷成 final_model 并删掉 checkpoint-*/README.md/training_args.bin;在 checkpoint-75(400 题 0.853)略高于 rl_v4(400 题 0.850)的情况下选了训练更久的那个,再用 80… | i=1242, i=1241 |

### 训练序列(12 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 378 | smoke | 0.03h | returned | **unclear** | baseline(首次)。2000 行冒烟,只验 train_sft.py 能不能跑通 —— 受测变量既不是 C3 也不是 C4,四值枚举里没有对应取值(见 definition_defect d7)。真实结局:32 步跑完,前台返回。 |
| 398 | smoke | 0.01h | returned | **C4** | 相对 i=378:关掉 gradient_checkpointing,bs 16→32、accum 4→2,样本 2000→3000。读的是 train_runtime / OOM,不是分数。真实结局:CUDA OOM 崩溃。 |
| 413 | smoke | 0.01h | returned | **C4** | 相对 i=398:唯一新增 use_liger_kernel=1(GC 仍为 0,bs/accum/n 逐字相同)。真实结局:同样 CUDA OOM。 |
| 429 | smoke | 3.07h | run_end | **C4** | 相对 i=413:GC 0→1,bs 32→64,accum 2→1,n 3000→6000。真实结局:跑完,train_runtime 167.0167s / 35.9 samples/s,该配置被正式 SFT 采用。 |
| 501 | real | 1.31h | consumed | **both** | 首次真实训练。相对冒烟同时确定了数据(206785 行 OMI-2 派生、格式对齐、35% few-shot;c2/c3/c7)与方法超参(lr 1.5e-5、bs 64、accum 2、1 epoch、maxlen 1800;c8),两者绑在同一次训练里,拆不开。真实结局:train_runtim… |
| 648 | real | 0.01h | superseded | **C4** | 方法从 SFT 换成 GRPO(dr_grpo、beta=0、num_gen 8、gen_batch 256、lr 1e-6、maxcomp 512);数据换成 GSM8K train 的 7473 条 gold prompt。真实结局:34 秒内因 GRPOConfig 拒绝同时设置 genera… |
| 654 | real | 0.05h | superseded | **C4** | 相对 i=648:只从 grpo.py 里删掉 steps_per_generation 那一行,命令行逐字相同。真实结局:跑到第 2 步,agent 发现 clipped_ratio=1.0 / entropy=0 / grad_norm=0(EOS 记账错导致全样本被掩),在 i=704 用 p… |
| 732 | real | 0.46h | last_seen | **C4** | 相对 i=654:max_completion_length 512→600、steps 220→240,并先修好 sft_v1 的 eos_token(c10)。真实结局(骨架记 last_seen):跑到第 40 步、24:48 后,19:40:39 被 agent pkill -9,产物随即 … |
| 822 | real | 0.31h | discarded | **C4** | 相对 i=732:lr 1e-6→3e-6、maxcomp 600→480、save_steps 40→30(产物名换成 rl_v2)。真实结局:跑到第 30 步存 checkpoint 时被 transformers 的 GenerationConfig 校验挡下(do_sample=False … |
| 896 | real | 0.59h | consumed | **C4** | 相对 i=822:超参逐字相同,只有 --steps 240→220,外加先把 sft_v1/generation_config.json 的 temperature 弹掉(c12)让存盘能过。真实结局:一直跑到 21:27:43 才被 pkill -9(第 68/220 步),即约 1.47h,而… |
| 1057 | real | 2.15h | consumed | **C4** | 相对 i=896:起点从 sft_v1 换成 rl_start60(rl_v2/checkpoint-60),步数 220→170,其余超参不变。真实结局:22:11:08 因一条 2766 token 的 10-shot prompt 超过 vllm_max_model_length=2560 而… |
| 1155 | real | 1.69h | consumed | **both** | 相对 i=1057:数据侧 DSEED 7→91 换 prompt 打乱顺序(max_steps=150 截断 epoch,等于换掉看到的那批 prompt)+ few-shot k 上限 10→8;方法侧 lr 3e-6→2e-6、rollout temperature 1.0→0.9、步数 17… |

### 验证序列(15 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 137 | 3 | 150 | 是 |  | 0.087 |
| 299 | 3 | 150 | 是 | c1 | 0.287 |
| 583 | 3 | 150 | 是 | c2, c3, c7, c8 | 0.820 |
| 599 | 3 | 500 | 是 | c2, c3, c7, c8 | 0.822 |
| 943 | 3 | 150 | 是 | c9, c11, c12 | 未拿到 |
| 1009 | 3 | 300 | 是 | c9, c11, c15 | 0.733 |
| 1029 | 3 | 300 | 是 | c13, c9, c11 | 0.847 |
| 1132 | 3 | 300 | 是 | c15, c19 | 0.840 |
| 1207 | 3 | 400 | 是 | c18 | 0.850 |
| 1224 | 3 | 400 | 是 | c18, c19 | 0.853 |
| 1242 | 3 | 800 | 是 | c18, c20 | 0.833 |
| 1246 | 3 | 800 | 是 | c18, c19, c20 | 0.829 |
| 1266 | 3 | 800 | 是 | c15, c18, c20 | 0.8213(n=610,部分) |
| 1300 | 3 | 150 | 是 | c20 | 0.840 |
| 1302 | — | — | — | c13, c18, c19, c20 | 8 条历史 run 的重算分数,含 i=1266 唯一能拿到的 0.8213@610,以及同一权重同一解码在重叠 300… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[378]
- **1 次验证没有拿到信号**:i=[1302]
- **分类学缺口提案 2 条**
  - trainer-compat-fix(i=894, i=873, i=1055, i=893)
  - compliance-guard(i=373, i=435)
- **定义缺陷 10 条**
  - (i=943, i=947, i=949, i=1012)
  - (i=1266, i=1282, i=1299, i=1303)
  - (i=10, i=271, i=269, i=611)
  - (i=1009, i=1029, i=1027, i=1028, i=1012, i=1032, i=257, i=257, i=1303, i=22)
  - (i=896, i=943, i=947, i=1005, i=1003)
  - (i=1057, i=1105, i=1105, i=1130, i=1155)
  - (i=378, i=398, i=413, i=429)
  - (i=378, i=381)
  - (i=1370, i=1004, i=1054, i=1153, i=987)
  - (i=894, i=1055, i=1155, i=873, i=715)
- **边界情形 4 条**
  - i=704/715 把 sft_v1 的 eos_token 从 <\|endoftext\|> 改成 <\|im_end\|>。§3 C1 明文把 eos 列为 C1 的内容,但这次改的文件是 tokenizer_config.json / special_tokens_map.json / config.json(不是 generation_config.json),动机与效应都在训练侧:TR…(i=715, i=702, i=703)
  - i=1155 的 DSEED=91。换 shuffle 种子既没换数据来源也没换配比(仍是 GSM8K train 7473 条),按 §3 C3 的字面(「决定训练数据从哪来、按什么比例混」)不算 C3;但因为 max_steps=150 截断了 epoch,种子实际决定了这 150 步看到哪 4800 条 prompt,按 §3 C3 那次消融被推翻的理由(random.Random(7) v…(i=1155, i=1155, i=1154)
  - i=488 的一条命令同时改了两类东西:few-shot 注入概率 0.5→0.35 与 k 集合(§3 C2 明文覆盖「按比例注入 few-shot 上下文」),以及新增 1800 token 长度过滤把 13752 条样本整条丢弃(数据筛选,属 C3)。changes 表的 category 是单值,装不下;拆成两条又会让两条共用同一个锚点事件与同一次训练,判定归属时重复计数。(i=488, i=491, i=485)
  - i=1302 用自写脚本遍历 logs/*.json 重算了 8 条历史评测的 accuracy,其中包括 i=1266 那次崩溃评测唯一可得的 610 题 0.8213,以及同一权重同一解码在重叠 300 题前缀上的 0.8467 vs 0.8333(agent 据此量出约 1.3 点的批处理噪声)。这个动作零 GPU、几秒钟(第一档的开销),读出来的却是真实评测的 accuracy(第三档的信…(i=1302, i=1303, i=1317)

## claude_non_api_max_claude-opus-4-8_10h_run1__humaneval_HuggingFaceTB_SmolLM3-3B-Base_17323515
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | humaneval | HuggingFaceTB_SmolLM3-3B-Base | 9.01h | 0.5182926829… |

### 改动序列(16 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 140 | C3 | 构建 SFT 数据 data_sft.jsonl:30,000 条 bigcode/self-oss-instruct-sc2-exec-filter-50k(纯 Python、执行过滤)+ 970 条按 signature+docstring 重排并经执行验证的 MBPP,共 30,970 条。明… | i=140, i=143 |
| 158 | C2 | 把单函数解里的顶层 import 挪进函数体,以匹配 HumanEval 的写法(164 题里 0 条 canonical solution 用顶层 import),否则评测的 extract_function_body 会把 import 丢掉。30,970 条里改了 8,858 条。 | i=155, i=158, i=159 |
| 173 | C4 | 自写 train_sft.py:全参微调,bf16 + flash_attention_2 + gradient checkpointing,靠 {% generation %} 掩码做 assistant-only loss 并在 assistant 段末尾补 EOS,max_len 1024、l… | i=173, i=165 |
| 249 | C1 | 用 generation_config.json 强制贪婪解码:写入 temperature 0.0 / do_sample false / top_p 1.0 / top_k -1,并把 eos_token_id 改成 [128001, 128012](加上 <\|im_end\|>)。先在 vl… | i=249, i=247, i=555 |
| 262 | C3 | 造 reasoning 变体 data_sft_reason.jsonl:把 self-oss 原 response 里代码块之前的自然语言推理裁剪后包进 <think>…</think> 放在代码块前,30,970 条里 15,305 条带 think;代码块本身与 data_sft.jsonl … | i=262, i=265 |
| 282 | C5 | 写 eval_epochs.sh:阻塞等训练 PID 结束,然后对每个 epoch checkpoint 逐个补 tokenizer 文件、写贪婪 generation_config、跑 evaluate.py --limit 150。这是本 run 全部 checkpoint 选择的机制。 | i=282, i=285 |
| 311 | C2 | 把 13,787 条 self-oss 样本的 prompt 反向改写成评测同款 signature+docstring 形式(data_sft_hf.jsonl),又与 reason 变体交叉出 data_sft_hf_reason.jsonl。两份都建好了,但直到 run 结束都没被任何一次训练… | i=311, i=314, i=327 |
| 369 | C2 | 用 ast.unparse 把四份数据里所有代码块统一成 4 空格缩进。起因:模型输出 2 空格缩进时,评测的 extract_function_body(找 ':\n    ')会错抓到嵌套 if 的冒号,静默截断函数体。data_sft.jsonl 30,970 条里改了 15,215 条,un… | i=346, i=369, i=372 |
| 433 | C3 | 扩量增多样:self-oss 取满 50,000 + Magicoder-OSS-Instruct-75K 的 python 子集 34,537 + 验证过的 MBPP 970 = 85,507 条,同样 4 空格归一化。 | i=433, i=446 |
| 517 | C3 | 把 reasoning 前缀回填到 85,507 条大数据集上(data_sft_big_reason.jsonl),但只有 25,732 条能匹配回源 prose,think 比例从 49% 稀释到 30%。 | i=517, i=520 |
| 551 | proposed:submission-safe… | 在 E3 长训练开始后立刻把当时最好的 sft_reason 拷成 final_model 并写贪婪 config,先保证任何时刻都有一份合法提交;随后清掉误拷进去的 checkpoint 子目录(24GB→5.8GB)并改 finalize 脚本只拷模型/tokenizer 文件。 | i=551, i=552, i=558 |
| 609 | C7 | 自建拒绝采样验证器 rft_generate.py:用 sft_reason 对 971 道 MBPP 各采 8 个样(temp 0.8),按官方评分器的 find_code() / extract_function_body() 抽代码再跑该题的 assert 测试,只留通过的。解出 853/97… | i=609, i=645 |
| 648 | C3 | 把 1,513 条自采样-执行验证过的解归一化后并进 E2 配方,得到 32,483 条 data_sft_reason_rft.jsonl(self_oss 30,000 + rft 1,513 + mbpp 970)。 | i=648, i=649 |
| 679 | C4 | 把已验证最好的 E2 配方(data_sft_reason,30,970 条)从 3 epoch 加到 4 epoch,其余超参不动,理由是 E2 到 epoch 3 曲线仍在上升而更大的数据集反而更早过拟合。 | i=679, i=680 |
| 694 | proposed:verifier-protoc… | 发现自己一路用的 --max-connections 16 与 grader 的 evaluate.py 默认(--max-connections 1、--gpu-memory-utilization 0.3)不同,后者无批处理因而确定性;于是写 final_select.sh,把候选全部按 gra… | i=694, i=711, i=729 |
| 732 | C5 | 最终选中 sft_r4/checkpoint-2901(E5 第 3 个 epoch,不是训练终点)作为 final_model,顶掉安全网里的 sft_reason;按 grader 口径 52.7% vs 50.7%。 | i=732, i=731, i=733 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 190 | real | 0.05h | consumed | **unclear** | baseline(本 run 第一次训练)。但它不是在测数据也不是在测超参:agent 明说是拿 4,000 条样本、1 epoch 跑通 train→save→eval→confirm stopping 这条链路的冒烟。实测 140 秒结束,--save_strategy no 不留 checkp… |
| 213 | real | 0.30h | discarded | **both** | vs i=190:数据从 data_sft.jsonl 的前 4,000 行扩到全部 30,970 行(C3),同时 epochs 1→3、--save_strategy no→epoch(C4),lr 从默认改为显式 2e-5(数值相同)。两条都变了,结果无法归给任一边。 |
| 379 | real | 8.13h | last_seen | **C3** | vs i=213:命令行逐字相同,唯一差别是 data_sft.jsonl 的内容被 normalize_all.py 用 ast.unparse 重写成 4 空格缩进(30,970 行改了 15,215 行)。注意这在实质上是一次 C2(格式对齐)改动,只是被迫用一次完整训练来交付 —— 见 bo… |
| 476 | real | 1.18h | consumed | **C3** | vs i=379:命令行只差 --data 与 --out,超参逐字相同(epochs 3、bs 8、accum 4、lr 2e-5、max_len 默认 1024、seed 42)。两份数据 30,970 条一一对应、代码块相同,唯一差别是其中 15,305 条的 assistant target… |
| 529 | real | 3.03h | last_seen | **C3** | vs i=476:只换数据 —— 30,970(self-oss 30k + MBPP 970)→ 85,507(self-oss 50k + Magicoder-python 34,537 + MBPP 970),带 <think> 的比例从 15,305/30,970 稀释到 25,732/85… |
| 658 | real | 2.90h | run_end | **C3** | vs i=529:只换数据 —— 回到 data_sft_reason 的 30,970 条,再并入 1,513 条由 sft_reason 自采样、经 MBPP 测试执行验证过的解,共 32,483 条。超参逐字相同(epochs 3、bs 8、accum 4、lr 2e-5、max_len 10… |
| 680 | real | 1.55h | last_seen | **C4** | vs 紧邻的 i=658 是数据换回 data_sft_reason 且 epochs 3→4;但 agent 明写的对照对象是 i=476 的 E2,相对 i=476 命令行唯一差别是 --epochs 3→4(--max_len 1024 等于默认)。受测变量因此是 epoch 数,不是数据。 |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 97 | 3 | 40 | 是 |  | 0.025 |
| 202 | 3 | 30 | 是 | c1, c2, c3 | 0.4 |
| 285 | 3 | 150 | 否 | c1, c2, c3 | 未拿到 |
| 385 | 3 | 150 | 否 | c1, c2, c3, c4, c8 | 0.4533333333333333 / 0.4666666666666667 / 0.4933333333333333… |
| 476 | 3 | 150 | 是 | c5 | 0.42 / 0.48 / 0.5066666666666667 / 0.52 |
| 545 | 3 | 150 | 是 | c9, c10 | 0.41333333333333333 / 0.49333333333333335 / 0.48666666666666… |
| 648 | 4 | 164 | 是 | c5 | 0.47560975609756095 |
| 658 | 3 | 150 | 是 | c12, c13 | 45.3 / 49.3 / 46.0 / 46.7 |
| 680 | 3 | 150 | 是 | c14 | 42.0 / 44.7 / 52.0 / 50.7 / 51.3 |
| 711 | — | — | 否 |  | 未拿到 |
| 711 | — | — | 是 |  | 未拿到 |
| 711 | — | — | 否 | c14, c15 | 0.5066666666666667 / 0.5266666666666666 |
| 711 | — | — | 是 | c14, c15 | 0.5066666666666667 / 0.5266666666666666 |
| 736 | — | — | 是 | c15, c16 | 0.5266666666666666 / 0.4666666666666667 / 0.5066666666666667 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[190]
- **4 次验证没有拿到信号**:i=[285, 385, 711, 711]
- **分类学缺口提案 2 条**
  - verifier-protocol-alignment(i=694, i=711, i=729, i=731, i=774)
  - submission-safety-net(i=551, i=552, i=558, i=692)
- **定义缺陷 8 条**
  - `discarded` 似乎由「配对事件的命令里出现 rm」判定,既不检查被删的路径是不是该产物,也不检查同一条命令是否正在消费它。本 run 四个 discarded 里三个是错的。反例一:i=680 的 sft_r4 被记成 discarded,配对时刻 01:16:51 对应 i=736,而 i=736 删的是 runs/grader_sft_r4.json 这个陈旧的评测结果 JSON —…(i=736, i=732, i=552, i=552)
  - i=379 的 sft_full 被记成 8.13h / discarded。真实值 ≤1.02h:该训练在 17:09:18 启动,i=385 启动的编排脚本会阻塞等它的 PID 结束、再逐个评 4 个 checkpoint,而编排脚本在 20:10:34 CEST(=18:10:34 UTC)就报告全部评完。8.13h 的终点 01:17 对应 i=736 那条 rm -f runs/grad…(i=467, i=282, i=736, i=677, i=677)
  - i=97 的 --limit 被记成空、档位记成第四档(全量)。实际命令是 `nohup python evaluate.py --model-path ... \` 换行后 `--limit 40 ...`,机械层只留下了反斜杠之前的第一行(骨架「命令」列也正好断在反斜杠处),于是把一次 n=40 的第三档局部评测(按 §2 噪声 ±10–15pp)记成了第四档全量评测。这条恰好是本 run 唯…(i=97, i=122)
  - i=711 与 i=736 的 --limit 都被记成 2。这两条命令里没有任何数字样本量参数(`nohup ./final_select.sh <dirs> > runs/xxx.log 2>&1 &`),唯一的 2 来自 `2>&1`。脚本里真实跑的是 --limit 150:agent 逐字报出 76/150 与 79/150。样本量记成 2 会让这两次评测的噪声估计完全失真。(i=711, i=725, i=731)
  - i=711 在骨架里产生了两行评测记录,其中第一行的命令是 `chmod +x final_select.sh multi_eval.sh` —— 只是给脚本加执行位,不是一次评测。抽取器显然按脚本文件名匹配,于是同一个事件里凡提到 final_select.sh / multi_eval.sh 的片段都被计成一次评测。这正是 §1 抽取纪律里「评测次数被高估 30–50%」那类失效在本 run …(i=711)
  - 骨架只列出 4 次 generation_config.json 访问,因为抽取器按字面路径匹配。本 run 真实写入次数远不止:每次 eval_epochs.sh / final_select.sh / multi_eval.sh 评一个 checkpoint 前都会 `python set_genconfig.py <dir> greedy` 重写一遍。反例是一次连命令行都写得明明白白、却仍未…(i=648, i=555)
  - i=777 被记成对 final_model/generation_config.json 的 write。该命令对 generation_config.json 只有 `g=json.load(open('final_model/generation_config.json'))` 这一次读取;它唯一的删除动作是 `rm -f final_model/trainer_state.json`,与 …(i=777, i=777)
  - 本 run 的冒烟测试按现规则一条都识别不出来,骨架里 7 段训练全部标成 real。i=190 在意图上是标准冒烟 —— agent 明说是拿子集把 train→save→eval→confirm stopping 跑通再上大 run,140 秒结束、--save_strategy no 不落 checkpoint —— 但产物名叫 sft_quick(不在 smoke/sanity 名单里),…(i=183, i=190, i=190, i=200)
- **边界情形 3 条**
  - c2-change-delivered-by-retraining(i=346, i=369, i=372, i=379)
  - think-prefix-c2-or-c3(i=265, i=379, i=476)
  - smoke-training-has-no-tested-variable(i=183, i=190)

## claude_non_api_max_claude-opus-4-8_10h_run1__humaneval_Qwen_Qwen3-4B-Base_17315101
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | humaneval | Qwen_Qwen3-4B-Base | 10.08h | 0.6280487804… |

### 改动序列(21 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 32 | C1 | 把保存 checkpoint 的 eos 从 base 的 <\|endoftext\|>(151643) 改成 <\|im_end\|>(151645),同时写进 config.json 与 generation_config.json,否则 vLLM 在评测时不会停。实测 checkpoint-… | i=32, i=31, i=274 |
| 69 | C3 | 确定 SFT 语料来源与配方:self-oss-instruct-sc2-exec-filter-50k + Magicoder-Evol-Instruct-110K + Magicoder-OSS-Instruct-75K(仅 python),按 10-gram shingle 对 HumanEv… | i=69, i=89, i=91 |
| 69 | C2 | 把训练响应重写成单个 ```python 代码块、且把 import 内联进函数体,理由是官方评分器 find_code 取第一个代码块、extract_function_body 丢弃签名行之前的一切——即让训练样本与评测抽取口径对齐。 | i=69, i=69, i=18 |
| 83 | C4 | 训练方法:全参 SFT(非 LoRA)+ completion-only loss 掩码(prompt token 置 -100),bf16 + flash_attention_2 + gradient checkpointing,lr 1e-5 / cosine_with_min_lr / war… | i=83, i=106 |
| 83 | C2 | 训练时用评测同一份 templates/qwen3.jinja 渲染,并逐 token 验证 prompt 是 full 的前缀、completion 恰为 <think>\n\n</think>\n\n```python...```<\|im_end\|>,保证训练串与评测时模型真实看到的串一致。 | i=83, i=202, i=202 |
| 97 | C5 | 把 save_strategy 从 no 改成 epoch(save_total_limit=4),一次训练同时留下 epoch-1(checkpoint-780)与 epoch-2(checkpoint-1560)两个候选,供事后按分数挑选。 | i=97, i=299 |
| 115 | C4 | OOM 修复(151,936 词表的 logits 撑爆显存):per-device bs 32→16、accum 2→4、max-len 1536→1024,并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。有效 batch 仍为 64。 | i=115, i=116 |
| 150 | C3 | 新增自生成训练数据的题库:把 MBPP(train/validation/test/prompt 全部 split)重写成 HumanEval 风格「签名+docstring」题面,并配可执行 check_code,得 971 道。 | i=150, i=156 |
| 157 | C7 | 自建执行式代理验证器:把候选函数与 MBPP 的 assert 拼成一段源码丢进 subprocess 执行(超时 8s、32 线程),用返回码判对错,替代跑一次真实评测来判定候选解。agent 自述抽取/清洗走的是与评测同一套逻辑,但轨迹里没有逐字对照过官方 find_code。 | i=157, i=157, i=391 |
| 166 | C3 | 为加快迭代,把训练语料从全量 99,384 条降采样到 50,000 条(--subsample 50000,过滤后 49,910)。 | i=166, i=188, i=191 |
| 166 | C4 | 同一次调整里把 epoch 从 3 降到 2(动机是墙钟:3 epoch 估算 2.5 小时太长)。 | i=166, i=188 |
| 326 | C5 | 在同一次训练的两个 checkpoint 之间选 epoch-1(checkpoint-780, 59.3%)而非 epoch-2(sft_v1, 56.0%):后续的 RFT 生成用的是 checkpoint-780,并据此把下一次训练的 epoch 定成 1。 | i=326, i=334 |
| 334 | C3 | 用 sft_v1/checkpoint-780 对 971 道 MBPP 题各采样 k=16(temperature 0.8, top_p 0.95, max_tokens 768),执行验证后每题最多保留 4 个去重解:15,507 个候选 → 解出 854/971 题 → 3,102 条已验证样… | i=334, i=391, i=391 |
| 354 | C3 | RFT 生成崩溃(prompt 3720 token > max_model_len 2048)后的修复:gen_prompt 里只放前 3 条测试断言,并把 vLLM max_model_len 2048→4096。 | i=354, i=361 |
| 393 | C4 | epoch 2→1,依据是 v1 的 epoch-2(56.0%)反而低于 epoch-1(59.3%),判定为过拟合。 | i=393, i=398 |
| 393 | C3 | v2 的配方:全量 99,384 条 SFT + 3,102 条 MBPP-RFT 各重复 3 次,合成 108,690 条(data/combined_v2.jsonl)。 | i=393, i=394, i=395 |
| 409 | C3 | 构建「简短推理」训练集:把 self_oss 响应里代码块之前的说明文字塞进 <think>...</think>、后接干净代码块,推理长度过滤 60–1200 字符,去污染后 25,008 条(data/sft_reason.jsonl)。放弃 KodCode-R1 的深推理,理由是 p50 约 … | i=409, i=402, i=415 |
| 421 | C1 | 在 train.py 保存路径里把 generation_config 强制成贪婪(do_sample=False, temperature=0.0, top_p=1.0, top_k=0)。触发原因:save_pretrained 存出来的 checkpoint 里 do_sample 变成 No… | i=421, i=421, i=274 |
| 443 | C1 | 对已训好的 sft_v2 就地改 generation_config.json:eos_token_id=151645, do_sample=false, temperature=0.0, top_p=1.0, top_k=0。字段级看是 load→update→dump 的合并写,不是整份重写——… | i=443, i=447, i=447, i=447 |
| 467 | C3 | 为第 4 轮准备更大的自生成题库:从 KodCode-V1-SFT-R1 流式扫描出 8,000 道 HumanEval 风格题面 + pytest check_code,并用一个必然失败的 dummy 解校验过 check_code 会正确报错。轨迹结束前未被用于任何训练。 | i=467, i=473, i=477 |
| 483 | proposed:submission_guar… | 在 iteration-3 训练还在跑、结果未知时,先把当前最好的 sft_v2 复制成 final_model 作为「安全网」。不改权重、不改数据、不改超参,只决定「此刻把哪个已有候选写进提交目录」。这次动作直接决定了本 run 的最终分:它发生在轨迹最后一个 Bash 事件,23 秒后轨迹结束。 | i=483, i=484, i=487 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 106 | real | 0.02h | superseded | **both** | baseline(首次训练)。99,384 条 SFT 语料 × 3 epoch,全参 SFT + completion-only loss,lr 1e-5 / bs 32 / accum 2 / max-len 1536。真实结局:不是 superseded,是启动约 3 秒后 CUDA OOM(… |
| 116 | real | 0.12h | superseded | **C4** | 相对 i=106 只动 C4 旋钮:bs 32→16、accum 2→4、max-len 1536→1024、加 expandable_segments;数据(data/sft.jsonl 全量 99,384,过滤后 99,200)与 epoch 数(3)逐字不变。真实结局:不是被新启动 super… |
| 167 | real | 0.02h | killed | **both** | 意图是同时动两项:数据 99,384→50,000(--subsample 50000)、epoch 3→2,其余(lr/bs/accum/max-len)不变。但这次训练从未启动:命令返回 Exit code 144,5 秒后 `ps aux \| grep train.py` 输出为空、GPU … |
| 188 | real | 0.85h | consumed | **both** | 相对 i=116 那次真正跑起来的训练同时动了两项:数据 99,384→50,000(过滤后 49,910),epoch 3→2;lr 1e-5 / bs 16 / accum 4 / max-len 1024 逐字不变。真实结局:跑满 1560 步(20:36:39→约 21:27,约 51 分钟… |
| 398 | real | 0.91h | consumed | **both** | 相对 i=188 两项都动,且两项都是有意的:(C3)数据从 50k 子集换成 data/combined_v2.jsonl = 99,384 条 SFT + 3,102 条 MBPP-RFT ×3 = 108,690(过滤后 108,506);(C4)epoch 2→1,依据是 v1 ep2(56… |
| 462 | real | 0.07h | last_seen | **C3** | 受测变量是数据配方:响应形态从「纯 ```python 代码块」换成「<think>简短推理</think> + 代码块」,语料换成 25,008 条 self_oss 推理数据(vs v2 的 108,690 条直出混合数据)。epoch 1→2 与 max-len 1024→1280 是为把 t… |

### 验证序列(5 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 39 | 3 | 30 | 是 |  | 0.5333333333333333(limit 30,原始 Qwen/Qwen3-4B-Base 基线;此时尚无任何改… |
| 303 | 3 | 150 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.5933333333333334(v1_ep1 = sft_v1/checkpoint-780,limit 150,… |
| 303 | 3 | 150 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.5933333333333334(v1_ep1 = sft_v1/checkpoint-780,limit 150,… |
| 303 | 3 | 150 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.56(v1_ep2 = sft_v1 终点,limit 150,max-connections 16;骨架把这一行也… |
| 303 | 3 | 150 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.56(v1_ep2 = sft_v1 终点,limit 150,max-connections 16;骨架把这一行也… |
| 443 | 3 | 150 | 是 | c10, c11, c12, c13, c14, c15, c17, c20 | 0.6266666666666667(v2_runA = sft_v2,limit 150,max-connection… |
| 450 | 3 | 150 | 是 | c17 | 0.64(v2_runB,与 runA 同权重、同 --limit 150、同 --max-connections 16… |

### 异常与存疑

- **分类学缺口提案 1 条**
  - submission_guard(i=483, i=484, i=494)
- **定义缺陷 4 条**
  - (i=168, i=177, i=185, i=178)
  - (i=303, i=323, i=326)
  - (i=80, i=443, i=447)
  - (i=481, i=481, i=494, i=495)
- **边界情形 2 条**
  - (i=402, i=409, i=415)
  - (i=157, i=391, i=391)

## claude_non_api_claude-opus-5_10h_run2__humaneval_Qwen_Qwen3-4B-Base_17419570
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | humaneval | Qwen_Qwen3-4B-Base | 8.20h | 0.8353658536… |

### 改动序列(22 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 241 | C2 | build_data.py 的抽取管线按官方评分器的行为重排代码:只保留单个顶层函数,把非 typing 的 import 移进函数体(typing import 留在模块级,因为签名注解要用),因为 inspect_evals.humaneval 的 extract_function_body 从… | i=234, i=261 |
| 528 | C3 | 选定 notbadai/python_functions_reasoning(206k 行,OpenCoder SFT 派生)为主训练数据源,make_sft.py 产出 10 万行 sft_v1(styleA 51508 / styleB 48492);sft_v1 过污染检查 0 命中。 | i=528, i=531 |
| 597 | C2 | 训练文本手工拼成 <\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n,绕开 qwen3 chat 模板在渲染真实 assistant 轮次时插入的空 <think> 块(推理时不插入,训练/推理不对称);user 侧前缀 IN… | i=568, i=533, i=572 |
| 664 | C3 | 生成 data/sft_v1_nr.jsonl:把 assistant 目标里第一个 ```python 之前的推理段整段删掉,作为「含推理 vs 不含推理」对照的另一臂数据(10 万行全部保留)。 | i=664, i=663 |
| 766 | C7 | 自建代理验证器 verify_rs.py:把 inspect_evals.humaneval 的 pattern_1/pattern_2、find_code()、extract_function_body() 逐字复制进来(源码里标注 '# ---- verbatim from inspect_ev… | i=763, i=1258 |
| 825 | C1 | train_sft.py 保存的 generation_config.json 被整份改写:eos_token_id 由 [151645, 151643] 收成单值 151645、新增 temperature: 0.0、**原有的 max_new_tokens: 3900 被顺手删掉**(agent… | i=824, i=824, i=796 |
| 872 | C1 | finalize_model.py:对每个 checkpoint 整份重写 generation_config.json,并同步改 config.json 的 eos_token_id 与 tokenizer_config.json 的 eos_token。字段级差异(base → 交付件,逐字核对… | i=871, i=105, i=1575, i=1575 |
| 874 | proposed:eval-invocation… | 自建 run_eval.sh 封装官方 evaluate.py,把 --max-connections 从默认 1 提到 16、--gpu-memory-utilization 0.3 提到 0.85,limit 走位置参数。全 run 的取数都经它;收尾时又特意用官方默认参数(limit 150 … | i=874, i=874, i=1573 |
| 941 | C2 | 把 OpenCodeInstruct 的 assert 单测用 AST 解析成 '>>> call' / 'expected' 的 doctest 行注入题面 docstring(每题最多 3 条),并在有 doctest 时截掉原题面的 **Sample Input / Example 段,理由是… | i=938, i=959 |
| 967 | C3 | build_rs_problems.py 从缓存的 nvidia/OpenCodeInstruct parquet 里抽 6 万道 average_test_score==1.0 的题(带单测),重建成 HumanEval 风格的 签名+docstring prompt;过滤掉 stdin/prin… | i=967, i=1066 |
| 1090 | C3 | 用 ckpt/a_reason 自己对 22000 道 RS 题各采 4 个样本(temperature 0.8 / top_p 0.95 / max_tokens 1100),再用 c7 的执行验证过滤——即 reference §C3 的来源 (d) 自生成+验证过滤。88000 个候选里 76… | i=1090, i=1317 |
| 1133 | C5 | 给 train_sft.py 加 --save-strategy / save_total_limit=3,让 v2 在 epoch 边界落中间 checkpoint,以便训完后在 epoch-1 与 epoch-2 之间挑一个交(这是本 run 唯一一次真正的「同一次训练不同步数」选择)。 | i=1125, i=1367 |
| 1185 | C5 | 防御性提交:训练还在跑时先把 ckpt/a_reason 整个复制成 final_model,保证任何时刻都有一个可交的模型(reference §4.4 把它的缺失列为扣分项)。 | i=1185, i=1184 |
| 1359 | C3 | make_sft2.py 的候选挑选规则由「取最短的正确候选」改成「取中位长度的正确候选」,并按 26000 styleA + 13000 styleB + 20565 RS = 59565 行拼出 sft_v2(--n-rs 21000 未被打满,因为可用 RS 题只有 20565)。sft_v2… | i=1359, i=1359, i=1362 |
| 1367 | C4 | v2 训练把 epoch 从 1 提到 2、样本量从 25000 提到 60000(lr / bs / accum 不变)。事后判定为过拟合:epoch-2 80.7% < epoch-1 82.0% < a_reason 84.7%(同为 --limit 150)。 | i=1367, i=1125 |
| 1425 | C3 | v3 配方:RS 数据整批去掉,只用 pfr —— 用 random.Random(7) 打乱 sft_v1 的 10 万行池子后取 styleA 前 34000 + styleB 前 17000 = 51000 行。 | i=1425, i=1425 |
| 1425 | C4 | v3 同时把 epoch 从 2 退回 1(样本量 60000→51000);lr 1e-5 / bs 16 / accum 4 不变。与 c14 同一条命令发出,所以 v3 这次训练的数据与超参是一起变的。 | i=1425, i=1424 |
| 1461 | C6 | 现写 soup.py 做逐张量均匀平均(fp32 累加后转回 bf16),合成 ckpt/soup_av3 = mean(a_reason, v3)。 | i=1461, i=1460 |
| 1507 | C5 | 把 v3 装成 final_model(rm -rf 后整目录复制 + finalize),作为当时的交付候选。 | i=1507, i=1506 |
| 1510 | C3 | v4 配方:RS 占比压到 10000/61000,且候选挑选由「中位长度」改成「最长」(cands[-1]);pfr 那 51000 行是用 random.Random(11) 重新打乱后再取 34000+17000 —— 而这个 rng 在打乱 pfr 之前已经先 shuffle 过 20565… | i=1510, i=1510, i=1506 |
| 1541 | C6 | 第二次权重平均:ckpt/soup_v34 = mean(v3, v4)。 | i=1541, i=1540 |
| 1574 | C5 | 最终交付换成 v4(v3 与 v4 全量 164 打平 0.8414634,以「v4 还训了执行验证过的数据」作为 tie-break),并删掉 final_model/training_args.bin。 | i=1574, i=1573 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 635 | smoke | 0.01h | returned | **unclear** | baseline —— 全 run 第一次训练,200 条 / 1 epoch / bs 4 accum 2 的冒烟,产物 ckpt/smoke,38 秒返回(train_runtime 15.3s)。它只验管线跑通(reference §2 第二档),不测任何 C3/C4 变量;四选一里没有这个取… |
| 664 | real | 0.36h | consumed | **C3** | baseline(首次真实训练)。数据 = sft_v1 打乱后前 25000 行(pfr,assistant 目标含推理段);超参 1 epoch / bs 16 / accum 4 / 默认 lr 1e-5。它被明确设计成「含推理 vs 不含推理」这组数据对照的参照臂——同一条命令里就生成了对照… |
| 1002 | real | 0.00h | killed | **C3** | 与 a_reason 逐字同超参(--limit 25000 --epochs 1 --bs 16 --accum 4,同一份未改动的 train_sft.py),唯一差别是 --data 由 data/sft_v1.jsonl 换成 data/sft_v1_nr.jsonl(assistant 目… |
| 1367 | real | 2.30h | consumed | **both** | 相对 a_reason:数据换成 sft_v2(26000 styleA + 13000 styleB + 20565 条执行验证过的 RS,RS 取中位长度候选),样本量 25000→60000,epoch 1→2,并新增 --save-strategy epoch。数据配方与超参同时变,拆不开。 |
| 1425 | real | 0.84h | consumed | **both** | 相对 v2:RS 数据整批去掉(纯 pfr,Random(7) 抽 34000 styleA + 17000 styleB = 51000),epoch 2→1,样本量 60000→51000。数据与超参同时变。若改与 a_reason 比,则是同 epoch 下 pfr 数据量 25000→510… |
| 1514 | real | 0.97h | consumed | **C3** | 相对 v3:--epochs 1 / --bs 16 / --accum 4 / --lr 1e-5 四项逐字相同,且 v3 与 v4 之间没有再编辑过 train_sft.py(最后一次编辑在 i=1135,11:59 之前)。只有数据变:61000 = 51000 pfr + 10000 RS(… |

### 验证序列(13 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 131 | 3 | 60 | 是 |  | 0.45 |
| 1002 | 3 | 60 | 是 | c1, c2, c3, c5 | 0.9333333333333333 |
| 1065 | 3 | 150 | 是 | c1, c2, c3, c5 | 0.8466666666666667 |
| 1257 | — | — | 是 | c7 | 非评测事件:命令里既无 evaluate.py 也无 run_eval.sh,跑的是自建校验器 verify_rs.py… |
| 1308 | — | — | 是 | c7, c10 | 非评测事件:同样只跑 verify_rs.py。真实读回在 i=1317:problems with >=1 pass:… |
| 1381 | 3 | 150 | 是 | c12, c13, c4 | 0.8066666666666666 |
| 1392 | 3 | 150 | 是 | c11, c12, c13, c4 | 0.82 |
| 1431 | 4 | -1 | 是 | c14, c15, c4 | 0.8414634146341463 |
| 1450 | 4 | -1 | 是 | c1, c2, c3, c5 | 0.8353658536585366 |
| 1461 | 4 | -1 | 是 | c16 | 0.8292682926829268 |
| 1522 | 4 | -1 | 是 | c18, c4 | 0.8414634146341463 |
| 1541 | 4 | -1 | 是 | c19 | 0.8353658536585366 |
| 1576 | 3 | — | 是 | c20, c5, c21 | 0.8333333333333334 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[635]
- **分类学缺口提案 2 条**
  - eval-invocation-settings(i=874, i=1576, i=1573)
  - target-response-style(i=663, i=1359, i=1506)
- **定义缺陷 5 条**
  - (i=1002, i=1002, i=1019)
  - (i=1257, i=1308, i=1384)
  - (i=874, i=1450, i=1392)
  - (i=1431, i=1522, i=1002)
  - (i=1450, i=1453, i=1431)
- **边界情形 4 条**
  - (i=664, i=518)
  - (i=1359, i=1510)
  - (i=1573, i=1392)
  - (i=635, i=625)

## codex_non_api_xhigh_gpt-5.5_10h_run1__aime2025_Qwen_Qwen3-4B-Base_17134425
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.5 | codex | aime2025 | Qwen_Qwen3-4B-Base | 7.93h | 0.0333333333… |

### 改动序列(17 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 72 | C2 | 建 train_math_lora.py:训练样本用评测器逐字相同的 USER_PROMPT_TEMPLATE + <\|im_start\|>user/assistant chat 包装,mask prompt token 只训 assistant 侧,target 末尾统一为 'ANSWER: … | i=72, i=41, i=325 |
| 85 | C3 | 配方 v1:28,000 条 = Numina 16,000 + OpenR1 12,000 + 老 AIME(1983-2023)DeepSeek-R1 traces 上采样 + Hendrycks 难题,全部过 unsafe_2025_reference 过滤。 | i=85, i=86 |
| 86 | C4 | 方法定为 LoRA(r64/alpha128)+ merge 回全权重。trainable 132,120,576 参数与 36 层 q/k/v/o/gate/up/down 的 r=64 之和逐位吻合,说明未设 modules_to_save,lm_head/embed_tokens 未训 —— … | i=86, i=72, i=77 |
| 156 | C1 | 改 final_model 的 config.json 与 generation_config.json(写入事件 i=157/158,file_change 无内容):eos_token_id 151643 → [151645, 151643]。字段级判定:i=152 改前与 i=237 改后键集… | i=156, i=152, i=237 |
| 163 | C2 | 写 train_concise_lora.py 并在 final_model 之上再训一遍(i=169,0.55h):换成短的官方/竞赛式解答,删掉重复的 ANSWER 标记,只在末尾追加一个 ANSWER 行,目的是治 verbosity/不终止。意图是格式对齐(C2),但实现手段是一次完整训练。 | i=163, i=169 |
| 191 | C2 | 写 train_answer_only_lora.py 并在 final_model_concise 之上再训(i=194,0.50h):target 只保留 ANSWER 行、刻意不训推理,强制模型在可计分的 'ANSWER: n' 之后立刻停。同样是 C2 意图 + 完整训练成本。 | i=191, i=194 |
| 228 | C1 | final_model_answer 的 config.json + generation_config.json:eos_token_id [151645,151643] → [124,151645,151643,198](把模型实际吐出的垃圾 token 124 与换行 198 当停止符)。字段… | i=228, i=215, i=229 |
| 236 | C1 | final_model 与 final_model_concise 的 config.json + generation_config.json:eos_token_id → [124,88372,70564,151645,151643],并**新增 repetition_penalty: 1.08… | i=236, i=237, i=235 |
| 249 | C3 | 配方 v2 'clean':新脚本 train_clean_math_lora.py,回到原始 base 重训(不再在 final_model_concise 上叠加),数据只留老 AIME + Hendrycks MATH 人写解答 + Numina 4000,去掉被判为'教坏续写'的 R1 长 … | i=249, i=251 |
| 262 | C4 | 把 clean 训练的 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 改成 4/4(有效 batch 仍是 16),命令其余部分与 i=255 逐字相同,数据统计逐位相同。纯为吞吐(3.7s/step → 2s/st… | i=262, i=258, i=270 |
| 291 | C3 | 配方 v3 'r1clean':只用 4,500 条已验证的 pre-2025 AIME DeepSeek-R1 traces,numina 归零,aime-weight 2,max-solution-chars 5000;同时把 lr 从 8e-5 降到 6e-5。 | i=291, i=290 |
| 335 | C3 | 配方 v4 'r1long':发现脚本按题目去重把 R1 traces 压成每题一条后,改成 AIME-only 长 trace,1,444 条、max-solution-chars 50000、max-length 8192;同时 lr 4e-5、bs1/ga8、seed 77。 | i=335, i=329 |
| 364 | C3 | 配方 v5 'openr1clean':给 clean 脚本加一个筛过的 OpenR1 数值答案子集(非 amc_aime 来源、答案是 6 位内整数、solution ≤6000 字符),训 24,000 条、aime-weight 4;同时 lr 6e-5、bs4/ga4、max-length … | i=364, i=361 |
| 423 | C4 | 唯一一次换训练范式:在 final_model_openr1clean 之上跑 GRPO + 答案校验奖励(train_grpo_aime.py),420 条 pre-2025 AIME 题、80 步、num-generations 4、lr 5e-6、LoRA r32。 | i=423, i=389, i=422 |
| 459 | C5 | 候选选择:两个全量都是 0/30 的情况下,用 `rm -rf final_model && cp -a final_model_openr1clean final_model` 选 SFT-only 而非 GRPO,理由是输出更少被奖励扭曲。副作用(整份覆盖):final_model 的 gene… | i=459, i=458, i=460 |
| 466 | C1 | 最后一次解码实验:先把 generation_config.json 备份成 generation_config.greedy.json,再把 do_sample 设 True、temperature 0.7、top_p 0.9、top_k 50,同一条命令里接着跑 limit 5 评测。json.… | i=466, i=466, i=465 |
| 470 | C1 | 采样也 0/5,用 mv 把备份还原回去,提交态恢复贪婪解码(do_sample False / temperature 1.0 / top_p 1.0 / eos [151643,151645])。这次改动没有任何评测判定它 —— run 到此结束。 | i=470, i=471 |

### 训练序列(12 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 76 | smoke | 0.05h | returned | **unclear** | baseline(冒烟)。32 条样本、epochs 0.01,只验 train_math_lora.py 能否跑通并 merge。它检验的是'管线跑不跑得通'(第二档),既不是 C3 也不是 C4;枚举里没有合法取值,故被迫记 unclear —— 见 definition_defect d2。 |
| 86 | real | 2.35h | returned | **both** | baseline(首次真实训练)。相对'不训练'同时确定了数据配方(Numina 16k + OpenR1 12k + 老 AIME R1 traces + Hendrycks)与方法(LoRA r64/alpha128、lr 2e-4、1 epoch、ga16),两类变量一次性引入,拆不开。 |
| 166 | smoke | 0.02h | returned | **unclear** | 冒烟:验 train_concise_lora.py 能否以 final_model 为底座跑通并 merge。同 i=76,枚举无合法取值。 |
| 169 | real | 0.55h | returned | **both** | 底座从 base 换成 final_model(继续训);数据换成短的官方/竞赛式解答、只留一个末尾 ANSWER 行,14,000 条(vs 28,000);超参 lr 8e-5(vs 2e-4)、lora-r 32(vs 64)、bs2/ga8(vs 1/16)。数据与超参同时变。agent 自… |
| 194 | real | 0.50h | returned | **both** | 底座换成 final_model_concise;数据换成 answer-only(24,000 条,只有 ANSWER 行、无推理);超参 2 epochs、lr 1.5e-4、bs8/ga2。数据与超参同时变;真实意图仍是终止行为(C2)。无冒烟直接上(agent 在 i=193 明确考虑过要不… |
| 252 | smoke | 0.01h | returned | **unclear** | 冒烟:验新脚本 train_clean_math_lora.py 能否跑通(32 条、epochs 0.01、r8)。同 i=76。 |
| 255 | real | 0.07h | returned | **both** | 回到原始 base 重训(不再叠加);新脚本;数据只剩老 AIME + Hendrycks + Numina 4000,去掉 R1/OpenR1;lr 8e-5、r64、bs1/ga16、max-length 4096。**真实结局:不是正常结束 —— agent 在 i=259 自己 pkill … |
| 262 | real | 0.50h | returned | **both** | 与 i=255 的命令逐字相同,只把 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 换成 4/4(有效 batch 仍 16),数据构建统计逐位相同(14000 条 / avg 511.3 / max 2832 / … |
| 291 | real | 0.40h | returned | **both** | 数据:改用 4,500 条已验证 pre-2025 AIME R1 traces,numina 归零,aime-weight 2,max-solution-chars 5000;同时 lr 从 8e-5 降到 6e-5。agent 自述主变量是数据,但 lr 同时动了,不是单变量。 |
| 335 | real | 0.18h | returned | **both** | 数据:AIME-only 长 trace,1,444 条(vs 10,377),max-solution-chars 50000(vs 5000),hard-math-weight 0;超参:lr 4e-5(vs 6e-5)、bs1/ga8(vs 4/4)、max-length 8192(vs 40… |
| 364 | real | 1.21h | returned | **both** | 数据:新增 16,000 条筛过的 OpenR1 数值答案子集,numina 4000、r1 1000、aime-weight 4、24,000 条(vs 1,444);超参:lr 6e-5(vs 4e-5)、bs4/ga4(vs 1/8)、max-length 4096(vs 8192)、seed… |
| 423 | real | 0.77h | returned | **both** | 方法整体换掉:SFT → GRPO + 答案校验奖励(新脚本 train_grpo_aime.py),80 步、num-generations 4、lr 5e-6、r32、max-completion-length 768;数据也整体换掉:420 条 pre-2025 AIME 题当 RL prom… |

### 验证序列(20 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 42 | 3 | 3 | 是 |  | 0.0(0/3);base 模型的起点测量,不判定任何改动 |
| 138 | 3 | 5 | 是 | c1, c2, c3 | 0.0(0/5);真正读到的有用信号不是分数而是'5 题共 20,480 个输出 token 全部打满上限、从不终止' |
| 160 | 3 | 5 | 是 | c4 | 0.0(0/5);agent 没有在文本里复述这次的数字,直接转去写第二段训练脚本 |
| 176 | 3 | 5 | 是 | c5 | 0.0(0/5);读到的信号是'仍然每题打满',由此判定问题是行为而非格式 |
| 208 | 3 | 5 | 是 | c6 | 0.0(0/5);仍然打满上限,agent 判定 answer-only checkpoint 不能作为最终模型 |
| 230 | 3 | 5 | 是 | c7 | 0.0(0/5),但拿到的决策信号是生成长度从打满 4096 降到 44 个输出 token —— 这次评测判定的是终止… |
| 238 | 3 | 5 | 是 | c8 | 0.0(0/5);仍打满,agent 由此推断 final_model 的垃圾后缀 token 与本地探针里的不同 |
| 244 | 3 | 5 | 是 | c8 | 0.0(0/5);agent 未在任何文本里复述这次结果,直接转去写 clean 脚本 |
| 273 | 3 | 5 | 是 | c9, c10 | 0.0(0/5);关键信号是'终于能正常终止、长度合理',判定 C2 目标达成而准确率未动 |
| 276 | 4 | — | 是 | c9, c10 | 0.0(0/30) |
| 283 | 4 | — | 是 |  | 0.0(0/30);base 的全量起点,用来排除'是 SFT 把模型训坏了'这一解释 |
| 296 | 3 | 5 | 是 | c11 | 0.0(0/5);终止正常但答案错 |
| 312 | 4 | — | 是 | c11 | 0.0(0/30) |
| 345 | 3 | 5 | 是 | c12 | 0.0(0/5);输出更长更深但常常连 ANSWER 行都没写完 |
| 417 | 3 | 5 | 是 | c13 | 0.0(0/5) |
| 441 | 3 | 5 | 是 | c14 | 0.0(0/5) |
| 446 | 4 | — | 是 | c14 | 0.0(0/30) |
| 451 | 4 | — | 是 | c13 | 0.0(0/30);与 i=446 并列,两个候选打平在 0,最终选择只能改用非分数依据 |
| 462 | 3 | 3 | 是 | c15 | 0.0(0/3);这次的判定内容是'提交目录能不能被 evaluate.py 正常加载',不是准确率 |
| 466 | 3 | 5 | 是 | c16 | 0.0(0/5);据此把采样配置回滚成贪婪 |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[76, 166, 252]
- **分类学缺口提案 1 条**
  - contamination-guard(i=325, i=83, i=414, i=416)
- **定义缺陷 3 条**
  - 骨架把 i=255 的 runs/clean 记成 end_reason=returned,与另外 11 次正常跑完并 merge 的训练同标签。反例:这一段是 agent 在 i=259 亲手 pkill 掉的,只跑了约 4 分钟就中止,没有产出 final_model_clean(随后 i=262 用同一产物目录重启)。前台 span 的 '返回了 tool_result' 与 '训练正常结束…(i=259, i=261, i=262)
  - 枚举对冒烟训练没有合法取值。反例:i=76 的 runs/smoke 是 32 条样本、epochs 0.01 的管线连通性测试,它检验的是 reference §2 第二档的内容('依赖装齐了没有、参数名对不对、能不能落盘'),既不是数据配方也不是方法超参,证据也并不'不足'。本 run 12 行训练里有 3 行是冒烟,被迫填 unclear,直接把 unclear 率从 0%(9 次真实训练全…(i=75, i=76, i=252)
  - 表把 C2 的开销只按'判定'口径记,于是读起来像'格式对齐是分钟级、零 GPU 的一类改动'(§4 主表进一步写成'其余全部合计 <2% 预算')。反例:本 run 连着两次完整训练(i=169 concise 0.55h、i=194 answer_only 0.50h,合计约 1.05 GPU 小时,占 7.93h 墙钟的 13%)的**唯一声明目标就是 C2** —— 让模型只输出一个末尾 …(i=163, i=191, i=227, i=225)
- **边界情形 3 条**
  - 纯吞吐的超参改动落在 C4 与'不是改动'之间。i=262 相对 i=255 只把 microbatch 从 1/16 拆成 4/4,有效 batch 仍是 16,数据构建统计逐位相同 —— 按 §3 C4 的字面定义('lr/epoch/batch/序列长度')它是 C4,但它对模型是构造上的空操作,动机是墙钟(3.7s/step → 2s/step)。把它计入 C4 的改动数,会用一个先验效应…(i=258, i=262, i=97)
  - n=5 的 `evaluate.py --limit 5` 落在第二档与第三档之间。本 run 20 次验证里有 15 次是 limit 5(13 次)或 limit 3(2 次),只有 5 次是全量:它们跑的是真实评测器(第三档的机器),但 30 题基准上取 5 题,准确率不携带任何可用信息(全程读数恒为 0);agent 实际用它们判定的是二值的行为问题 —— 会不会终止、输出多长、有没有可计…(i=232, i=275, i=461)
  - 最终候选选择落在 C5 与'无验证器可用'之间。§3 C5 定义为'在已训好的若干步数里挑一个交',而 i=459 的选择是在两个**不同配方**的终态模型之间做(openr1clean vs grpo),且两者全量都是 0/30 —— 第四档给出的排序是平局,选择实际依据是自写脚本统计的输出长度/ANSWER 行覆盖(i=455/456)加上一句'RL 输出被奖励扭曲更多'的先验。这既不是 C5…(i=456, i=458, i=459)

## codex_non_api_xhigh_gpt-5.5_10h_run2__aime2025_google_gemma-3-4b-pt_17134874
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.5 | codex | aime2025 | google_gemma-3-4b-pt | 7.69h | 0.0 |

### 改动序列(17 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 40 | proposed:contamination_h… | 删除每次评测产生的 logs/ 目录，以免后续训练脚本读到 AIME 2025 题面；全 run 重复 13 次（i=40,151,262,271,352,361,371,380,387,394,402,408,413），并在 build 管线里内置 reject_2025() 过滤器。不提分，但决… | i=39, i=254, i=428 |
| 64 | C2 | 新建 train_lora_sft.py，把训练样本的 prompt 写成与 evaluate.py 实际下发的 USER_PROMPT_TEMPLATE 逐字一致（同样的两段指令、ANSWER: $ANSWER 末行），外层手工拼 gemma3 的 <start_of_turn>user / <s… | i=63, i=234, i=240 |
| 64 | C3 | 初始数据配方：Prompt48 旧 AIME 1983–2024 带解答题 + EleutherAI/hendrycks_math + AI-MO/NuminaMath-CoT（按 source 分桶限额），并拒收任何标记 2025 AIME 的记录。 | i=63, i=74, i=74 |
| 70 | proposed:tooling_repair | 把 NuminaMath 的加载从 streaming=True 改成普通缓存路径 + .shuffle(seed)，目的是绕开 datasets 的解释器退出 bug 让脚本能落盘；意图是修 bug 不是提分，但它会改变实际选中的 numina 行（见 boundary_case b3）。 | i=69, i=290, i=299 |
| 84 | C4 | exp1 的方法与超参选型：LoRA r=64 / alpha=128、lr 1.2e-4、cosine + warmup 0.04、1 epoch、bs 2 × grad-accum 8、max-length 3072，结束后 merge 回全量权重存成 final_model。 | i=84, i=137 |
| 156 | C7 | 自建代理验证器（第一版）：直接用 vllm.LLM 在 gneubig 旧 AIME 2024-II 题上跑推理，用与官方逐字相同的 prompt 和手拼的 gemma3 头，看模型能不能按格式输出且答对。 | i=156, i=155 |
| 163 | C3 | AIME 标签审计：把 Prompt48 的解答文本与 gneubig 官方答案键按 (Year, Part, Problem Number) 对拼，删掉 89 行答案不一致的解答，只留 578 行可信旧 AIME。 | i=159, i=161, i=165 |
| 167 | C3 | exp2 的配方重配：numina-limit 24000→28000、aime-repeat 10→18（配合 c7 的清洗集）；所有训练超参与 exp1 逐字相同。 | i=167, i=84, i=167 |
| 244 | C7 | 自建代理验证器（第二版，换独立测试集）：用与官方完全相同的 Inspect/vLLM 通路跑 inspect_evals/aime2024 --limit 10，把 AIME 2024 当成 held-out 代理集，避开读 2025 题面。 | i=244, i=245 |
| 260 | C1 | 改写 final_model_clean/generation_config.json（字段级差异，由 i=257 与 i=265 两次 cat 对出）：do_sample true→false、新增 temperature 0.0、**删掉 top_k 64 与 top_p 0.95**；bos_… | i=257, i=257, i=257, i=265, i=265, i=259 |
| 293 | C3 | 新增 OpenR1-Math-220k 加载器：只取 is_reasoning_complete / correctness 验过的 R1 推理轨迹，同样过 reject_2025，重用同一套 prompt / ANSWER 格式；不替换原有 numina/aime 通路而是追加。 | i=286, i=298 |
| 301 | C3 | exp3 的数据配方：加入 openr1-limit 18000（实际入选 11547 行）、numina-limit 28000→10000、aime-repeat 18→30、max-examples 36000→30000。 | i=301, i=342 |
| 301 | C4 | exp3 同时把学习率从 1.2e-4 降到 1.0e-4（其余超参与 exp1/exp2 相同），所以这次训练同时动了 C3 和 C4。 | i=301, i=167 |
| 351 | C1 | 改写 final_model_openr1/generation_config.json（由 i=348 与 i=355 对出）：do_sample true→false、删掉 top_k 64 与 top_p 0.95，**但这次没有加 temperature**——因为上一次评测的 vLLM 输… | i=348, i=355, i=343, i=268 |
| 385 | C5 | 候选选择：根据两次全量评测（0.033 vs 0.000）把 final_model_openr1 整目录拷贝覆盖 final_model（先 rm -rf），此举同时**永久删掉了 exp1 的权重**。注：选的是三个独立训出来的模型而不是同一次训练的不同 step，与 C5 现有定义不完全吻合（… | i=384, i=385, i=404 |
| 407 | C1 | 把 final_model/generation_config.json 换回 gemma 原生的采样配置，专门为了在全量 30 题上做一次贪婪 vs 采样的对照（权重已经 diff -qr 确认与 final_model_openr1 逐字节相同）。 | i=406, i=406, i=397 |
| 411 | C1 | 采样没有提分（两边都是 0/30），改回确定性解码作为交付态；最终落盘的 config 为 do_sample false、无 temperature、无 top_k/top_p。 | i=410, i=414 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 67 | — | — | — | **C3** | baseline。注：带 --skip-train，只跑数据组装与分词，不加载模型、没有任何优化器步（参见 definition_defect d1）。它验的是 c2/c3 的数据侧：schema、答案抽取、污染过滤器。 |
| 71 | — | — | — | **C3** | 命令与 i=67 逐字相同；两次之间只改了 train_lora_sft.py 的 NuminaMath loader（streaming→缓存，即 c4）。同样是 --skip-train，不训练。 |
| 81 | smoke | 0.01h | returned | **C4** | 本 run 唯一一次真正上 GPU 的冒烟：故意缩小成 lora-r 8 / alpha 16 / lr 1e-4 / epochs 0.05 / bs 1 / accum 1，只跑 3 个优化器步，验的是 PEFT 能不能挂上 Gemma3 、Trainer 跑不跑得动，与数据配方无关。 |
| 84 | real | 2.17h | returned | **both** | baseline（首次真实训练）。数据配方（c3）与方法/超参（c5）同时第一次被确定，没有可对照对象，所以这 2.17h 同时服务 C3 和 C4。 |
| 164 | — | — | — | **C3** | 仍是 --skip-train 的数据构建检查；与 i=71 之间的唯一变化是 c7 的 AIME 标签清洗，输出也只汇报数据侧数字。 |
| 167 | real | 2.18h | returned | **C3** | 与 i=84 相比，训练超参逐字相同（--max-examples 36000 --max-length 3072 --epochs 1.0 --batch-size 2 --grad-accum 8 --lora-r 64 --lora-alpha 128 --lr 1.2e-4 --warmup… |
| 297 | — | — | — | **C3** | --skip-train 数据构建检查；与 i=164 之间的唯一变化是 c11 新增的 OpenR1 loader，输出就是 OpenR1 入选行数。 |
| 301 | real | 2.43h | returned | **both** | 与 i=167 相比同时动了两类：数据（+OpenR1 18000、numina 28000→10000、aime-repeat 18→30、max-examples 36000→30000，c12）与超参（lr 1.2e-4→1.0e-4，c13）。最终交付的就是这一个，但它相对 exp2 的 +… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 23 | 3 | 2 | 是 |  | 0.0 |
| 147 | 3 | 5 | 是 | c2, c3, c5 | 0.0 |
| 267 | 3 | 5 | 是 | c7, c8, c10 | 0.0 |
| 357 | 3 | 5 | 是 | c11, c12, c13, c14 | 0.0 |
| 366 | 4 | — | 是 | c11, c12, c13, c14 | 0.03333333333333333 |
| 376 | 4 | — | 是 | c7, c8, c10 | 0.0 |
| 387 | 4 | — | 是 | c15 | 0.0 |
| 398 | 4 | — | 是 | c15 | 0.0 |
| 408 | 4 | — | 是 | c16 | 0.0 |

### 异常与存疑

- **分类学缺口提案 2 条**
  - contamination_hygiene(i=39, i=254, i=412, i=428)
  - tooling_repair(i=69, i=299)
- **定义缺陷 5 条**
  - (i=67, i=72, i=297, i=82)
  - (i=385, i=404, i=406, i=147)
  - (i=244, i=245, i=156, i=159)
  - (i=257, i=348, i=257, i=14)
  - (i=413, i=414, i=396)
- **边界情形 3 条**
  - i=385 的候选选择是在**三个独立训出来的模型**（exp1 / exp2_clean / exp3_openr1）之间挑一个交，而 C5 的定义写的是“在已训好的若干**步数**里挑一个交”、“同一次训练的相邻 checkpoint”。两者的结构属性完全一致（零训练成本、靠第三/四档判定、纯选择动作），但按现有文字判不了。如果把它归入 C5，那 §4.2 里 C5 的“±15 点，受控”就不…(i=384, i=385)
  - c7 的 AIME 标签审计改变的是训练数据（C3），而 §3 主表把 C3 定义为“可用最低档 = 二 + 四”、“基本不能判定”。但这一次改动的正确性是在 13 秒内、零 GPU、确定性地判定的：把两个数据集对拼就数出 89 行答案不一致。即存在一类 **C3 子动作，它有第一档确定性验证器**（数据标签的内部一致性），这与 §3 主表“C3 在完整训练之前没有任何中间档可用”直接矛盾。需要拆…(i=161, i=165, i=159)
  - c4（i=70）把 NuminaMath 从 streaming 换成 load_dataset(...).shuffle(seed=seed)，意图是修一个解释器退出 bug，但两种读法选中的 24000 行根本不是同一批样本——按意图它不是 C3，按效果它就是 C3。而且轨迹里没有任何信息能判断 exp1 实际吃到的混合与 i=67 dry run 量到的是否一致。与 reference §C…(i=69, i=290)

## codex_non_api_max_gpt-5.6-sol_10h_run1__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17390212
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 7.37h | 0.1333333333… |

### 改动序列(22 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 166 | C3 | 建 prepare_training_data.py:从 OpenR1-Math-220k 的已验证 R1 轨迹筛出 22,495 条(107.6M token)作 SFT 语料,并用内存内 AIME2025 deny-list(字面 2025 / 精确匹配 / 五词 shingle 重叠)去污染 | i=168, i=183, i=179 |
| 185 | C2 | 训练样本按评测自己的 templates/smollm.jinja 渲染:统一 ANSWER: 标记、以 <\|im_end\|> 收尾、completion-only 掩码;启动前在第一档做逐 token 核对(每条 think 2/2、answer 3、im_end 2、ends True) | i=185, i=681, i=7 |
| 337 | C3 | augment_specialization.py:把 AIME 专项集从 727 条扩到 753 条(补 17 条 7B 教师验证轨迹 + 9 条 AIME2024 已验证解题路径) | i=339, i=340, i=347 |
| 355 | C3 | generate_aime24_teacher.py:用沙箱内已缓存的 DeepSeek-R1-Distill-Qwen-14B 本地推理,对 30 道 AIME2024 各采 4 次,只保留答案可验证的 19 道(纯自生成+验证过滤,不调外部 API) | i=850, i=922, i=924 |
| 366 | C4 | 开 DPO/APO 分支:prepare_preferences.py 造 2,214 对 verifier 确认的 正确 vs 错误 偏好对,train_dpo.py 按官方 SmolLM3 APO 配方训练 | i=368, i=371, i=1014 |
| 437 | C3 | prepare_second_pass.py:再造一份与一期按 example id 完全不相交的二期语料 12,055 条 / 53.3M token,用于第二个 SFT pass | i=439, i=442, i=447 |
| 449 | proposed:eval_readback | 自写 summarize_eval.py:回读 inspect_ai 自己的日志,导出官方 console 不打印的诊断(correct 数、completion 长度分位、has_answer_label / has_think_close / stop_reasons)。它不替代官方评分器,而是… | i=451, i=452, i=452 |
| 687 | C1 | 改 checkpoint_general/generation_config.json(+config.json):由 {_from_model_config,bos_token_id,eos_token_id:128001,transformers_version} 变成加上 do_sample:… | i=684, i=685, i=705, i=459 |
| 715 | C2 | prepare_termination_calibration.py:造 5,838 条简短解答的 termination 校准集,目的是高频监督 </think>、ANSWER:、<\|im_end\|>、EOS 四个终止标记(意图是格式对齐,但交付方式是一次完整 SFT) | i=717, i=729, i=859 |
| 1052 | C1 | checkpoint_general_greedy:先 cp -al 硬链接整份 checkpoint、再 cp --remove-destination 断开该 JSON 的硬链接,然后只改两个字段 do_sample true→false、temperature 0.6→0.0;eos_toke… | i=1049, i=1054, i=1055 |
| 1081 | C4 | 崩溃修复:第一次 APO 在参考 logprob 跑完后死于 LengthGroupedSampler 的 ValueError,补丁关掉 length grouping(train_dpo.py),命令行其余部分不动 | i=1078, i=1079, i=1080 |
| 1089 | C6 | 自写 interpolate_checkpoints.py,做 base↔tuned 的 α 加权插值(不是均匀平均):checkpoint_aime_mix25(α=.25)、checkpoint_calibrated_mix10(α=.10),用来保留被整体拒收的改动的一部分 | i=1091, i=1095, i=1169 |
| 1179 | C1 | checkpoint_general_rep102:在 t=0.6 基线上加 1.02 的重复惩罚(字段名轨迹里未显示,值与意图明确) | i=1176, i=1183, i=1177 |
| 1186 | C1 | checkpoint_general_t08:temperature 0.6→0.8,其余字段不变(最终 final_model 的配置,i=1238 逐字确认 do_sample true / temperature 0.8 / top_p 0.95 / top_k 0) | i=1183, i=1238, i=1238 |
| 1193 | C1 | checkpoint_general_t10:temperature 1.0 | i=1189, i=1197 |
| 1198 | proposed:eval_readback | 临时写脚本从 inspect_ai 日志里取每次评测答对的题号集合,做跨评测的重叠/并集分析(t06/t08/greedy/t10;两次 t08 的 overlap 4 题、union 8 题),用来在噪声下判断 5/30 与 6/30 的差异是否可信 | i=1198, i=1199, i=1243 |
| 1212 | C1 | checkpoint_general_t07:temperature 0.7 | i=1208, i=1216 |
| 1219 | C1 | checkpoint_general_t09:temperature 0.9 | i=1216, i=1223 |
| 1224 | C6 | checkpoint_apo_mix25:把 APO 的权重 delta 按 α=.25 插回 checkpoint_general | i=1224, i=1223 |
| 1228 | C1 | 给 checkpoint_apo_mix25 打上当时的最优解码配置(temperature 0.8),使权重实验与解码实验不混在一起 | i=1222, i=1227 |
| 1234 | C5 | 选交付物:在 6 个训练产物 + 3 个插值产物 + 7 个解码变体里选 checkpoint_general_t08(第一段 SFT 的权重 + temperature 0.8),cp -a 成 final_model | i=1234, i=1233, i=1254 |
| 1249 | C1 | checkpoint_general_t08_k20:在 t08 之上把 top_k 0→20(采样尾部截断) | i=1245, i=1246, i=1253 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 171 | — | — | — | **unclear** | 不是训练:prepare_training_data.py 只做筛选/分词/落盘(输出 data/general_train 与 data/aime_specialize),没有任何权重更新。机械层因脚本名含 train 而误记为训练启动 |
| 185 | real | 2.23h | returned | **both** | baseline:全 run 第一次训练,同时定下数据配方(22,495 条 OpenR1 已验证轨迹 / 107.6M token)与方法(全参 BF16 SFT,1 epoch,lr 2e-5,bs 2 × accum 4)。train_runtime 7981s,与机械层量到的 2.226h … |
| 713 | real | 1.10h | returned | **both** | vs i=185:数据换成 data/general_train_pass2(12,055 条与一期按 id 不相交的新题),lr 2e-5→1e-5,起点 base→checkpoint_general;bs/accum/epochs 不变。数据与超参同时变,拆不开 C3/C4 |
| 925 | real | 0.17h | returned | **both** | vs i=713:数据换成 data/aime_specialize_teacher(772 条 AIME 专项,含 19 道 AIME2024 教师验证轨迹),epochs 1→2,lr 1e-5→5e-6,起点回到 checkpoint_general |
| 943 | — | — | — | **unclear** | 不是训练:这是一条监控命令,trainer 命令行出现在 pgrep -f 的引号模式里;返回只有 ps 的 ELAPSED/STAT 两行。机械层误记为一次 0.00h 的训练启动 |
| 964 | real | 0.13h | returned | **both** | vs i=925:数据换成 data/termination_calibration(5,838 条简短解答),lr 5e-6→3e-6,bs 2→4,accum 4→2,起点 checkpoint_aime。意图是格式/终止对齐,但落在数据+超参同时变的一次完整训练里 |
| 1018 | real | 0.18h | returned | **both** | vs i=964:方法由 SFT 换成 DPO/APO(train_dpo.py),数据换成 data/preferences(2,214 偏好对),lr 3e-6→1e-6,起点回 checkpoint_general。结局:参考 logprob 算完(1107/1107)后死于 LengthGr… |
| 1082 | real | 0.50h | returned | **C4** | vs i=1018:命令行逐字相同(同模型、同数据、同 lr、同 epochs),唯一变量是 i=1081 对 train_dpo.py 的补丁——关掉 length grouping。这是崩溃重启而非新实验,是本 run 唯一一次只动 C4 不动 C3 的训练 |

### 验证序列(19 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 12 | 3 | 5 | 是 |  | 未拿到(vLLM server 启动失败,exit_code 1;这次评测从未产出分数) |
| 59 | 3 | 3 | 是 |  | 0.0(0/3,base 模型 limit 3 探针) |
| 689 | 4 | — | 是 | c1, c2, c10 | 0.2(6/30);同一次评测同时裁决语料选择、格式对齐、解码配置三项,不可拆 |
| 841 | 4 | — | 是 | c4 | 0.133(4/30),二期语料被否决 |
| 997 | 4 | — | 是 | c7 | 0.067(2/30),终止校准被否决(输出总 token 从 376k 降到 80.7k 但正确率跌到 2/30) |
| 1006 | 4 | — | 是 | c5, c6 | 0.067(2/30),AIME 专项化被否决 |
| 1147 | 4 | — | 是 | c8, c22 | 0.133(4/30),APO 被否决 |
| 1155 | 4 | — | 是 | c11 | 0.167(5/30),贪婪解码 |
| 1164 | 4 | — | 是 | c18 | 0.100(3/30);机械层这一行记成 0.16666666666666666 是错的——那是上一次 greedy 评… |
| 1170 | 4 | — | 是 | c18 | 0.133(4/30) |
| 1180 | 4 | — | 是 | c12 | 0.167(5/30),重复惩罚 1.02 |
| 1187 | 4 | — | 是 | c13 | 0.2(6/30),temperature 0.8 第一次 |
| 1194 | 4 | — | 是 | c14 | 0.133(4/30),temperature 1.0 |
| 1201 | 4 | — | 是 | c10 | 0.167(5/30);与 i=689 同权重同 config 的重复评测,0.2→0.167,是本 run 直接测到的… |
| 1205 | 4 | — | 是 | c13 | 0.2(6/30),temperature 0.8 第二次,与第一次答对题号只重合 4 题 |
| 1213 | 4 | — | 是 | c15 | 0.1(3/30),temperature 0.7 |
| 1220 | 4 | — | 是 | c16 | 0.167(5/30),temperature 0.9 |
| 1229 | 4 | — | 是 | c19, c20 | 0.167(5/30),APO 25% 插值被否决 |
| 1250 | 4 | — | 是 | c17 | 0.167(5/30),top_k=20 被否决 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[171, 943]
- **分类学缺口提案 1 条**
  - eval_readback(i=451, i=452, i=712, i=1199, i=1243)
- **定义缺陷 8 条**
  - (i=12, i=57, i=59, i=452)
  - (i=1164, i=1167, i=1167, i=1169)
  - (i=171, i=183, i=183)
  - (i=943, i=943, i=944)
  - (i=185, i=171, i=1082)
  - (i=997, i=1003, i=1016)
  - (i=1251, i=705, i=1250)
  - (i=455, i=455, i=685)
- **边界情形 4 条**
  - termination calibration(c7,训练于 i=964):意图是格式对齐——高频监督 </think>、ANSWER:、<\|im_end\|>、EOS 四个终止标记,正是 C2 说的'让模型输出与评分器读的东西对齐';但交付方式是造一份新数据集再跑一次完整 SFT,最便宜的验证器是第二+第四档,不是第一档。按 §3 主表,C2 那一行写的是'可用最低档:一;需训练:否',对这次…(i=729, i=859, i=964, i=1005)
  - 解码扫描(c11–c17 + 两次重复,共 8 次全量评测跑在同一份权重上):按机制每一次都是 C1(改 generation_config.json),按功能整体是一次候选选择——在冻结权重上挑一个解码配置交付。§C5 的定义只覆盖'在已训好的若干步数里挑一个',没有'在同一 checkpoint 的若干解码配置里挑一个'这一格。照现定义要记成 7 次独立 C1 改动,既会重复计入 C1 的效应…(i=1183, i=1208, i=1216, i=1253)
  - i=1081 关掉 length grouping:按 §1 的定义,改动是'agent 为提升分数而做的一次有意图的修改'。这一条不是为提分,是为了让一次已经崩掉的训练能跑起来;但它改的确实是一个训练器超参(group_by_length),按 C4 的字面定义又能装进去。修复类改动在本 run 至少占 1 次训练启动(i=1018 整段作废)+1 次补丁,现分类学没有位置,记成 C4 会把'修…(i=1078, i=1080, i=1079)
  - interpolate_checkpoints.py(c18/c19)做的是 base↔tuned 的 α 加权插值(α=.25/.10),不是 §C6 定义的'把多个 checkpoint 权重均匀平均'。它的用途也不同:不是降方差的 soup,而是把一个整体被否决的训练结果按比例保留一部分(aime_mix25 从 2/30 的专项化里取 25%,结果 3/30;calibrated_mix1…(i=1091, i=1169, i=1173, i=1233)

## codex_non_api_max_gpt-5.6-sol_10h_run2__aime2025_Qwen_Qwen3-1.7B-Base_17404103
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | aime2025 | Qwen_Qwen3-1.7B-Base | 9.87h | 0.0666666666… |

### 改动序列(38 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 82 | C2 | train_sft.py 里手写与评测逐字一致的 PROMPT(<\|im_start\|>user + "The last line of your response should be of the form ..."),绕开 apply_chat_template,completion 侧固定… | i=17, i=404 |
| 84 | C3 | 建立污染安全的赛前数学 SFT 混料:Bespoke-Stratos 前 10k + DeepScaleR 全量 + NuminaMath-CoT 六个来源按配额抽样,去重后 84,772 条(seed 20250719),写 data/sft_train.jsonl 与 manifest。 | i=84, i=87 |
| 115 | C7 | 自建代理验证器 evaluate_aime2024.py:用 AIME 2024(允许使用、与计分集不重叠)当开发集,复用官方 aime_scorer,成为此后几乎所有 accept/reject 的实际判据。 | i=343, i=476 |
| 197 | C4 | 写 train_grpo.py:在 SFT 之上加一段 GRPO/RLVR,奖励用 math_verify 的 accuracy_reward + format_reward。 | i=190, i=385 |
| 217 | C3 | prepare_long_reasoning_data.py:从 Bespoke-Stratos 抽长推理轨迹另建一份数据(9,993 条),用于第二段长思维链 SFT。 | i=217, i=225 |
| 279 | C4 | 读 trl 的 grpo_trainer.py 后给 GRPO rollout 的 SamplingParams 显式加上停止 token,避免 rollout 不在 <\|im_end\|> 停。 | i=261, i=267 |
| 286 | C1 | 写 configure_checkpoint.py 并首次作用于 checkpoint:把 generation_config 的 eos_token_id 改成 [<\|im_end\|>, <\|endoftext\|>] 两个都接受、pad_token_id 设为 <\|endoftext\|… | i=286, i=298 |
| 294 | C1 | 改 configure_checkpoint.py 让它不再把 temperature/top_k/top_p 以 null 写进 generation_config.json。字段级:{temperature:null, top_k:null, top_p:null} 三个字段被整体删掉,结果文件… | i=291, i=294 |
| 306 | proposed:diagnostic-inst… | 写 analyze_eval_log.py:对 inspect_ai 日志做结构化统计(format_rate / think_rate / 平均长度 / stop_reason),本身不打分、不替代官方评测,但它给出的诊断直接决定了下一步改什么;i=795 又因为发现它把 reasoning 段读… | i=350, i=766 |
| 371 | C3 | 把长轨迹的 4k 截断从"只留开头"改成头尾各留(62% 头 + 桥接句 + 尾),因为原方案常把最干净的最终推导丢掉。命令行逐字不变,差异只在 train_sft.py 里。 | i=368, i=404 |
| 468 | C6 | 写 average_checkpoints.py:在两个全参 checkpoint 之间做线性权重插值(alpha 可调),后面全部 soup 都由它生成。 | i=468, i=809 |
| 522 | C3 | select_grpo_problems.py:用 sft_v1 自己在 held-out 老题上采样 8 次,只留通过率居中(非全对非全错)的题当 RL 训练集,192 选 16。 | i=522, i=527 |
| 585 | C3 | 自生成 + 验证过滤的数据:1,024 题 × 8 采样,只留 math_verify 判对的 rollout,得 710 条 / 460 题,经 prepare_rft_data.py 变成 data/rft_train.jsonl 去做拒绝采样微调。 | i=585, i=599 |
| 640 | C1 | 对权重完全不动的 runs/sft_v1 执行 configure_checkpoint.py --sample:generation_config 由"无 temperature 字段"改成 do_sample:true + temperature 0.6 + top_p 0.95 + top_k… | i=640, i=706 |
| 654 | C3 | prepare_curriculum_data.py:从已有池子里筛出答案是 0–999 整数的 17,114 道竞赛题,做一份贴近 AIME 答案空间的课程数据。 | i=654, i=660 |
| 688 | C3 | prepare_extension_data.py:再从 NuminaMath 里挑 100,000 条第一轮没用过的题(按 normalized problem 去重),做大规模覆盖扩展。 | i=688, i=685 |
| 826 | C3 | 再建一份与前 184,772 条完全不相交的 42,733 条 reserve 数据(--exclude 两份已用文件 + 换 seed 20250722),用来做第三段续训。 | i=826, i=846 |
| 882 | C6 | 在 sft_v1 与 extension_v1 之间取 alpha = 0.25 / 0.50 / 0.75 三个权重插值候选。 | i=882, i=878 |
| 922 | C6 | 在 AIME2024 上看到 25% 最好之后,细化插值网格到 0.125 / 0.1875 / 0.3125 / 0.375。 | i=922, i=940 |
| 958 | C1 | 对同一份 ext_soup_a025 权重扫解码温度:先 configure_checkpoint 不加 --sample(当时以为是贪婪)得 0/30,再扫 0.3 与 0.8,对照 0.6。 | i=958, i=957 |
| 978 | C1 | 对权重不动的 sft_v1 扫温度 0.2/0.4/0.8/1.0(i=978)与 0.50/0.55/0.65/0.70(i=987),每个温度写一次 generation_config 再跑一次全量 AIME2025。 | i=978, i=986 |
| 996 | C1 | 同权重扫 top_k(5/10/40/50)与 top_p(0.80/1.00),温度固定 0.6。 | i=996, i=1009 |
| 1012 | C1 | 给 configure_checkpoint.py 加 --repetition-penalty 并扫 1.03/1.05/1.10/1.15;全部 0/30 且输出长度翻倍,回退到 1.0。 | i=1012, i=1022 |
| 1133 | C6 | 在 extension_v1 与 reserve_v1 之间做 0.125/0.25/0.375/0.5/0.75 五个插值。 | i=1133, i=1128 |
| 1199 | C6 | sft_v1 × extension_v1 的 16 点细网格插值(alpha 从 0.03125 到 0.9375),统一配 temperature 0.6 采样后逐点跑全量 AIME2025。 | i=1199, i=1198 |
| 1384 | C1 | 读 vllm/config/model.py 的 get_diff_sampling_param 确认 min_p 会被读取后,给 configure_checkpoint.py 加 --min-p,在 sft_v1 的四份逐字节相同拷贝上扫 0.01/0.02/0.05/0.10。 | i=1384, i=1383 |
| 1416 | C5 | 把 runs/sft_v1 整份拷成 final_model 作为提交产物(此时选中的是"原始广域 SFT + temperature 0.6")。 | i=1416, i=1408 |
| 1494 | C1 | 发现"贪婪"其实是 temperature 1.0 采样(vLLM 不读 do_sample),改 configure_checkpoint.py 强制真贪婪;因为 transformers 拒绝把 do_sample=False 与显式 temperature 一起序列化,先用 do_sample… | i=1445, i=1494 |
| 1545 | C1 | OpenAI 兼容层把 1e-8 夹到 0.01,于是直接改写 18 份 generation_config.json:字段级差异只有 temperature 1e-08 -> 0.0(该文件此时为 {bos_token_id, do_sample:true, eos_token_id:[15164… | i=1544, i=1537 |
| 1630 | C1 | 把 configure_checkpoint.py 改成默认直接写 vLLM 的真贪婪设置,免掉手工改 JSON,保证后续候选评测契约一致。 | i=1629, i=1636 |
| 1637 | C6 | 在真贪婪下把 sft_v1 × extension_v1 的插值补成均匀 1/32 网格(补上 0.125/0.1875/0.25/0.3125/0.375/0.5/0.53125/... 等缺点)。 | i=1637, i=1636 |
| 1765 | C6 | 对两个 3/30 峰(alpha=0.21875 与 0.8125)以 1/256 间距做局部加密插值,各取两侧共 16 个点。 | i=1765, i=1764 |
| 1891 | C1 | 在选定的 21.875% 插值的 8 份逐字节相同拷贝上扫 repetition_penalty 0.95/0.975/0.99/1.01/1.025/1.05/1.075/1.10。 | i=1891, i=1887 |
| 1959 | C5 | 改选提交候选:用 AIME2024 给两个并列 3/30 的插值做 tie-break(0.21875 得 2/30、0.8125 得 0/30),选 runs/sft_ext_fine_a021875 作为新的 final_model。 | i=1959, i=1887 |
| 1968 | proposed:submission-expo… | 提交管道本身:先把旧的 final_model 归档成 runs/final_model_previous_export 再原子换上新导出,并用 sha256 + safetensors tensor 计数 + GenerationConfig 断言核对导出物与被选 checkpoint 逐字节一致… | i=1968, i=1972 |
| 2003 | C6 | 在评测器默认的 6 并发下,对 19.5%–22.1% 这段唯一稳定保住 2/30 的区间再做一次高分辨率插值搜索。 | i=2003, i=2032 |
| 2047 | C1 | 对 final_model 的 6 份拷贝扫极小幅度的 repetition_penalty(0.995/0.9975/0.999/1.001/1.0025/1.005),用官方默认 6 并发跑。 | i=2047, i=2046 |
| 2086 | C1 | 把 final_model 的 repetition_penalty 就地设成 0.999(该值在 6 并发下读到 3/30);随后同一份权重的复跑只有 2/30,agent 用 sha256 证明两者逐字节相同,把差异归给执行期非确定性。 | i=2086, i=2093 |

### 训练序列(13 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 88 | smoke | 0.01h | returned | **C4** | baseline(首次训练,冒烟):2,000 例 / 0.03 epoch / batch-size 4 / grad-accum 1,目的是跑通管线并量吞吐。 |
| 91 | smoke | 0.03h | returned | **C4** | 与 i=88 逐字相同,只把 --batch-size 4 改成 8:显存/吞吐阶梯的第二级。 |
| 96 | smoke | 0.01h | returned | **C4** | --batch-size 8 -> 12(--epochs 也从 0.03 微调到 0.025);结论是 12 在词表 loss 投影处爆显存,batch 8 成为此后所有训练的固定值。 |
| 100 | real | 1.07h | returned | **both** | 首次真实训练:同时把数据配方(84,772 条混料,c1)与方法/超参(全参 SFT、bs8×accum2、1 epoch、lr 2e-5)一次性定死;冒烟只跑过 2,000 例 / 0.03 epoch。此后所有分支都以它为 parent,因此它既不是纯 C3 也不是纯 C4 的测试。 |
| 360 | real | 0.11h | returned | **C3** | 换训练数据:从 84,772 条简洁解答换成 9,993 条 Bespoke 长推理轨迹(data/sft_long_reasoning.jsonl),parent 改为 runs/sft_v1;同时 lr 2e-5 -> 1e-5(agent 自述为配合长轨迹的保守设置)。真实结局是 agent … |
| 373 | real | 0.50h | returned | **C3** | 命令行与 i=360 逐字相同(只有 --output 从 sft_long_v1 变成 sft_long_v2);唯一差异在 train_sft.py:长轨迹截断由"只留头部"改成头 62%+桥接句+尾部。受测变量是同一批数据的截断方式。 |
| 528 | smoke | 0.01h | returned | **C4** | 方法切换的冒烟:SFT -> GRPO(train_grpo.py,--max-steps 1)。当场崩:PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 与 vLLM 的 CuMemAllocator 不兼容,断言失败,零优化步。 |
| 534 | smoke | 0.02h | returned | **C4** | 与 i=528 的唯一差异是 env -u PYTORCH_CUDA_ALLOC_CONF(去掉那个环境变量);跑通,得到一个 4/8 正确、全部格式合法的奖励组。 |
| 544 | real | 0.17h | returned | **both** | 从 sft_v1 出发的真实 GRPO:方法侧是 SFT->RLVR(--max-steps 100 --learning-rate 1e-6 --beta 0.001 --temperature 0.8),数据侧是 select_grpo_problems 从 192 题里筛出的 16 道通过率居… |
| 615 | real | 0.04h | returned | **C3** | 拒绝采样微调:数据换成模型自己生成并经 math_verify 验证过的 710 条轨迹(data/rft_train.jsonl),parent 仍是 sft_v1;为配合 710 条的小规模,epochs 1.0->4.0、lr 2e-5->5e-6 同时变了,所以不是单变量。 |
| 661 | real | 0.21h | returned | **C3** | 换数据配方:17,114 条答案为 0–999 整数的"AIME 形状"课程数据,parent 仍是 sft_v1,bs/accum/epochs 与 sft_v1 相同,lr 2e-5 -> 5e-6。 |
| 707 | real | 1.51h | returned | **C3** | 数据规模/覆盖面:100,000 条此前没用过的 NuminaMath 题(data/sft_extension.jsonl),parent 仍是 sft_v1,bs/accum/epochs 同 sft_v1,lr 2e-5 -> 1e-5。这是全 run 最长的一次训练(1,652 步 / 87… |
| 1049 | real | 0.61h | returned | **C3** | 再加一批与前 184,772 条完全不相交的 42,733 条数据;与前几次不同的是 parent 从 sft_v1 换成 runs/extension_v1(链式续训),lr 1e-5 -> 5e-6。 |

### 验证序列(115 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 38 | 3 | 5 | 是 |  | 0.0 |
| 346 | 4 | — | 是 | c1, c2, c7, c8 | 0.0 |
| 426 | 4 | — | 否 |  | 未拿到：该事件不是评测，是 `ps -eo pid,stat,cmd \| rg 'evaluate.py --help… |
| 703 | 4 | — | 是 | c14 | 0.067 |
| 964 | 4 | — | 是 | c18 | 0.0 |
| 969 | 4 | — | 是 | c18 | 0.0 |
| 969 | 4 | — | 是 | c18 | 0.0 |
| 969 | 4 | — | 是 | c18 | 0.0 |
| 969 | 4 | — | 是 | c18 | 0.067 |
| 969 | 4 | — | 是 | c18 | 0.067 |
| 969 | 4 | — | 是 | c18 | 0.067 |
| 969 | 4 | — | 是 | c16 | 0.067 |
| 969 | 4 | — | 是 | c16 | 0.067 |
| 969 | 4 | — | 是 | c16 | 0.067 |
| 978 | 4 | — | 是 | c21 | 0.033；该事件是一个 for 循环，实际跑了4 个温度：T=0.2 -> 1/30、T=0.4 -> 0/30、T=… |
| 987 | 4 | — | 是 | c21 | 0.033；for 循环，T=0.50 -> 1/30、0.55 -> 1/30、0.65 -> 0/30、0.70 -… |
| 996 | 4 | — | 是 | c22 | 0.033；for 循环，top_k 5/10/40/50 与 top_p 0.80/1.00 共 6 次，读到 1/3… |
| 1012 | 4 | — | 是 | c23 | 0.0；for 循环，repetition_penalty 1.03/1.05/1.10/1.15 全部 0/30（机械… |
| 1031 | 4 | — | 是 | c19 | 0.033；for 循环，4 个细插值点读到 1/30、1/30、0/30、2/30，均未超过 2/30（机械层取回 0… |
| 1124 | 4 | — | 是 | c17 | 0.0 |
| 1158 | 4 | — | 是 | c24 | 0.0 |
| 1164 | 4 | — | 是 | c24 | 0.033 |
| 1170 | 4 | — | 是 | c24 | 0.033 |
| 1175 | 4 | — | 是 | c24 | 0.033 |
| 1181 | 4 | — | 是 | c24 | 0.033 |
| 1271 | 4 | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1272 | 4 | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1273 | 4 | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1274 | 4 | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1289 | 4 | — | 是 | c25 | 0.0 |
| 1290 | 4 | — | 是 | c25 | 0.0 |
| 1297 | 4 | — | 是 | c25 | 0.033 |
| 1298 | 4 | — | 是 | c25 | 0.067 |
| 1307 | 4 | — | 是 | c25 | 0.067 |
| 1308 | 4 | — | 是 | c25 | 0.033 |
| 1316 | 4 | — | 是 | c25 | 0.0 |
| 1317 | 4 | — | 是 | c25 | 0.0 |
| 1323 | 4 | — | 是 | c25 | 0.0 |
| 1324 | 4 | — | 是 | c25 | 0.067 |
| 1333 | 4 | — | 是 | c25 | 0.033 |
| 1334 | 4 | — | 是 | c25 | 0.0 |
| 1342 | 4 | — | 是 | c25 | 0.0 |
| 1343 | 4 | — | 是 | c25 | 0.0 |
| 1350 | 4 | — | 是 | c25 | 0.0 |
| 1351 | 4 | — | 是 | c25 | 0.033 |
| 1394 | 4 | — | 是 | c26 | 0.0 |
| 1395 | 4 | — | 是 | c26 | 0.033 |
| 1401 | 4 | — | 是 | c26 | 0.033 |
| 1402 | 4 | — | 是 | c26 | 0.033 |
| 1437 | 4 | — | 是 | c27 | 0.0 |
| 1532 | 4 | — | 是 | c28 | 0.033 |
| 1533 | 4 | — | 是 | c28 | 0.033 |
| 1547 | 4 | — | 是 | c29 | 0.033 |
| 1548 | 4 | — | 是 | c29 | 0.0 |
| 1556 | 4 | — | 是 | c25, c29 | 0.0 |
| 1557 | 4 | — | 是 | c25, c29 | 0.0 |
| 1563 | 4 | — | 是 | c25, c29 | 0.033 |
| 1564 | 4 | — | 是 | c25, c29 | 0.033 |
| 1570 | 4 | — | 是 | c25, c29 | 0.067 |
| 1571 | 4 | — | 是 | c25, c29 | 0.1 |
| 1581 | 4 | — | 是 | c25, c29 | 0.033 |
| 1582 | 4 | — | 是 | c25, c29 | 0.067 |
| 1589 | 4 | — | 是 | c25, c29 | 0.067 |
| 1590 | 4 | — | 是 | c25, c29 | 0.033 |
| 1596 | 4 | — | 是 | c25, c29 | 0.067 |
| 1597 | 4 | — | 是 | c25, c29 | 0.033 |
| 1603 | 4 | — | 是 | c25, c29 | 0.067 |
| 1604 | 4 | — | 是 | c25, c29 | 0.1 |
| 1611 | 4 | — | 是 | c25, c29 | 0.067 |
| 1612 | 4 | — | 是 | c25, c29 | 0.033 |
| 1703 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1704 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1705 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1706 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1720 | 4 | — | 是 | c31 | 0.033 |
| 1721 | 4 | — | 是 | c31 | 0.033 |
| 1725 | 4 | — | 是 | c31 | 0.067 |
| 1726 | 4 | — | 是 | c31 | 0.067 |
| 1731 | 4 | — | 是 | c31 | 0.033 |
| 1732 | 4 | — | 是 | c31 | 0.033 |
| 1735 | 4 | — | 是 | c31 | 0.067 |
| 1736 | 4 | — | 是 | c31 | 0.033 |
| 1741 | 4 | — | 是 | c31 | 0.0 |
| 1742 | 4 | — | 是 | c31 | 0.033 |
| 1745 | 4 | — | 是 | c31 | 0.033 |
| 1746 | 4 | — | 是 | c31 | 0.0 |
| 1751 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1752 | 4 | — | 是 | c31 | 0.033 |
| 1758 | 4 | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，且 agent 没有重跑这一点 |
| 1759 | 4 | — | 是 | c31 | 0.033 |
| 1832 | 4 | — | 是 | c32 | 0.067 |
| 1833 | 4 | — | 是 | c32 | 0.067 |
| 1837 | 4 | — | 是 | c32 | 0.033 |
| 1838 | 4 | — | 是 | c32 | 0.033 |
| 1841 | 4 | — | 是 | c32 | 0.067 |
| 1842 | 4 | — | 是 | c32 | 0.033 |
| 1846 | 4 | — | 是 | c32 | 0.067 |
| 1847 | 4 | — | 是 | c32 | 0.067 |
| 1851 | 4 | — | 是 | c32 | 0.033 |
| 1852 | 4 | — | 是 | c32 | 0.033 |
| 1855 | 4 | — | 是 | c32 | 0.033 |
| 1856 | 4 | — | 是 | c32 | 0.067 |
| 1859 | 4 | — | 是 | c32 | 0.033 |
| 1860 | 4 | — | 是 | c32 | 0.033 |
| 1864 | 4 | — | 是 | c32 | 0.033 |
| 1865 | 4 | — | 是 | c32 | 0.033 |
| 1924 | 4 | — | 是 | c33 | 0.0 |
| 1925 | 4 | — | 是 | c33 | 0.0 |
| 1928 | 4 | — | 是 | c33 | 0.033 |
| 1929 | 4 | — | 是 | c33 | 0.067 |
| 1933 | 4 | — | 是 | c33 | 0.033 |
| 1934 | 4 | — | 是 | c33 | 0.0 |
| 1937 | 4 | — | 是 | c33 | 0.0 |
| 1938 | 4 | — | 否 | c33 | 未拿到：vLLM 服务器启动失败，且 agent 没有重跑这一点 |
| 1973 | 4 | — | 是 | c34 | 0.067 |
| 1978 | 4 | — | 是 | c34 | 0.033 |
| 1982 | 4 | — | 是 | c34 | 0.033；for 循环，8 个 2/30 候选在 6 并发下重测，结果 1/30~2/30，无人超过导出件（机械层取回… |
| 1995 | 4 | — | 是 | c34 | 0.067；for 循环，8 个邻域/边界候选，读到 1/30~2/30，无人超过 2/30（机械层取回 0.067） |
| 2032 | 4 | — | 是 | c36 | 0.067；for 循环，9 个 19.5%~22.1% 的高分辨率插值点，读到 0/30~2/30（机械层取回 0.0… |
| 2072 | 4 | — | 是 | c37 | 0.067；for 循环，6 个 repetition_penalty 微调候选；0.999 读到 3/30（=0.1）… |
| 2090 | 4 | — | 是 | c38 | 0.067 |

### 异常与存疑

- **12 次验证没有拿到信号**:i=[426, 1271, 1272, 1273, 1274, 1703, 1704, 1705, 1706, 1751, 1758, 1938]
- **分类学缺口提案 2 条**
  - diagnostic-instrumentation(i=350, i=517, i=794)
  - submission-export-plumbing(i=1968, i=2094, i=2099)
- **定义缺陷 6 条**
  - (i=426, i=427)
  - (i=1282, i=1277, i=1715, i=1757)
  - (i=84, i=85, i=647)
  - (i=1023, i=1027)
  - (i=572, i=575, i=698)
  - (i=369, i=368)
- **边界情形 3 条**
  - 序列截断/打包策略的改动（i=371）。它改变的是模型实际看到的 token（长轨迹从「只留头部」变成「头 62% + 桥接句 + 尾部」），按 C3「决定训练数据从哪来、按什么比例混」读不到它；但它实现在 train_sft.py 里，按 C4「全参/LoRA、SFT/DPO/GRPO、lr/epoch/bs」也读不到。两次训练（i=360 vs i=373）的命令行逐字相同，差异只在代码里 —…(i=373, i=368, i=404)
  - configure_checkpoint.py 把 eos_token_id 设成 [<\|im_end\|>, <\|endoftext\|>]（i=286）。C1 的定义里明写「改 generation_config.json（temperature / eos / 惩罚项）」，C2 的定义里也明写「把 <\|im_end\|> 训成结束符」。同一个动作被两条定义同时点名，而本 run 里这一…(i=286, i=298)
  - 本 run 把权重插值当成一个连续的选择变量扫了 40+ 个 alpha（c18/c19/c25/c31/c32/c36）。C6 的定义是「把同一轨迹上的多个 checkpoint 权重均匀平均」：这里两端是两条不同数据上的不同训练（sft_v1 vs extension_v1），且权重不均匀、alpha 本身就是被搜索的超参。行为上它更像 C5（在一堆候选里挑一个交，零训练成本，每个 2–5 分…(i=1198, i=1636, i=1764)

## codex_non_api_xhigh_gpt-5.5_10h_run2__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17138216
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.5 | codex | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 9.85h | 0.7573919636… |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 38 | C2 | 用评测器自己的 chat 模板 + inspect_evals 的 10-shot "ANSWER:" 提示渲染训练样本(prompt-mode exact),只对 assistant 段计损失,而不手写格式。 | i=38, i=33 |
| 50 | C4 | 改用全参微调而非 LoRA:两次 2 步冒烟(--lora vs 全参)比显存与吞吐后决定。 | i=49, i=50, i=53 |
| 72 | C3 | 给训练脚本加 --metamath-count:从 MetaMathQA 流式取 original_question 精确命中 GSM8K train 的 GSM 行(全库 240k 行命中 train、0 行命中 test),输出转成评测器的 ANSWER: 格式。 | i=70, i=75 |
| 101 | C1 | 字节复制权重后只改 eos:config.json 与 generation_config.json 里的 eos_token_id 由 128001 改成 [128001, 128012],让 vLLM 在 <\|im_end\|> 停止。字段级差异 = 1 个字段改值,无字段被删。 | i=100, i=101, i=98, i=107 |
| 108 | C1 | 把 eos 修复写进 train_math_sft.py(model.config.eos_token_id / model.generation_config.eos_token_id = [eos, <\|im_end\|>]),使此后每个保存的 checkpoint 自带两个 eos。 | i=107, i=444, i=151 |
| 110 | C3 | 第二阶段:在 exact-SFT checkpoint 之上加 100k train-derived MetaMath(direct 提示);同一条命令里还改了 epochs 2→1、bs 2→4、max-length 4096→2048,数据与超参是捆绑的。 | i=109, i=110 |
| 153 | C4 | 低 LR(5e-6)的 exact-format refresh 阶段,把被 direct 数据拉走的模型拉回计分提示格式。 | i=152, i=153 |
| 196 | C4 | 再加一轮 exact epoch,唯一变量是 --learning-rate 5e-6→2e-6(其余参数与 c7 逐字相同)。 | i=181, i=196 |
| 226 | C4 | 阶段顺序实验:与 c6 逐字相同的命令,只把 --model-name-or-path 换成 base 模型本身(MetaMath-first 而非 exact-SFT-first),随后再套用与 c7 相同的 refresh 配方。 | i=225, i=226, i=233 |
| 358 | C6 | 自写 average_checkpoints.py,把当时最强的两个 checkpoint 按 alpha=0.5 权重平均成 runs/soup_best_exactstop_alpha050。 | i=355, i=358 |
| 367 | C6 | 第二次权重平均,alpha=0.8 偏向当时最好的 checkpoint。 | i=366, i=367 |
| 393 | proposed:eval_protocol_c… | 校准官方评测器的调用口径:读 --help 与源码后发现 evaluate.py 的 --limit 默认是 150(不是全量)、真实默认 max_connections=2 / max_tokens=4000 / gpu_memory_utilization=0.3;此后固定口径比较,全量必须显式… | i=384, i=386, i=393 |
| 402 | C1 | 字节复制 refresh_e1 后在 generation_config.json 上新增三个字段 do_sample:false / temperature:0.0 / top_p:1.0(_from_model_config、bos_token_id、eos_token_id、transform… | i=397, i=402, i=401, i=565 |
| 409 | C1 | 同一 3 字段贪婪补丁应用到 full_exact_e2_stop_imend 的副本,用来在同一解码配置下重排候选。 | i=408, i=409 |
| 415 | C1 | 同一 3 字段贪婪补丁应用到 exact_e2_metamath100k_direct_e1 的副本。 | i=414, i=415, i=565 |
| 446 | C1 | 把 model.generation_config.do_sample = False 写进 train_math_sft.py,想让贪婪配置随权重一起落盘。该行确实生效于内存对象,但 save_pretrained 没有把它写进 generation_config.json,补丁失败。 | i=618, i=560 |
| 448 | C3 | 把 train-derived MetaMath 行数 100k→200k,其余全部逐字相同(同起点权重、同 prompt-mode、同 epochs/bs/accum/max-length/lr/warmup、同默认 --seed 1234)。因 metamath_train_rows 用同一 s… | i=445, i=448, i=618 |
| 568 | C1 | 同一 3 字段贪婪补丁应用到 200k checkpoint 的 generation_config.json(补 c16 未能持久化的洞)。 | i=560, i=569, i=598 |
| 597 | C5 | 候选选择:在全量分数上比较后,把 runs/exact_e2_metamath200k_direct_e1 整体拷成 final_model 提交(带贪婪 generation_config)。 | i=592, i=597 |
| 623 | C3 | 收尾的额外一轮:从 final_model 继续,换一批 75k 重新洗过的 MetaMath 行(--seed 2026)并降 LR 到 5e-6、warmup 0.02。数据与超参同时变。 | i=622, i=623 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 47 | smoke | 0.01h | returned | **unclear** | baseline(全 run 第一次训练)。2 步 LoRA 冒烟,目的明确写着是验管线(tokenizer / loss mask / 模型加载),既不在验 C3 也不在验 C4;四值域里没有"两者都不验"这一档,故记 unclear。 |
| 50 | smoke | 0.01h | returned | **C4** | 与 i=47 相比去掉 --lora 改全参、bs 1→2、显式 --learning-rate 1e-5;--train-limit 16 / --prompt-mode exact / --max-steps 2 / --max-length 3072 逐字相同。验的是全参能否放得下、跑多快。 |
| 54 | real | 0.85h | returned | **unclear** | baseline(第一次真实训练):GSM8K 官方 train split、exact 提示、2 epoch、bs 2×accum 4、max-length 4096、lr 1e-5。没有可比的上一次真实训练,受测变量无从定义。 |
| 110 | real | 1.30h | returned | **both** | 起点换成 runs/full_exact_e2_stop_imend;prompt-mode exact→direct 且加 100k train-derived MetaMath(C3),同时 epochs 2→1、bs 2→4、max-length 4096→2048(C4)。数据与超参捆绑在同… |
| 153 | real | 0.43h | returned | **both** | 起点 runs/exact_e2_metamath100k_direct_e1;prompt-mode direct→exact 且不再带 metamath(C3),同时 lr 1e-5→5e-6、bs 4→2、max-length 2048→4096(C4)。 |
| 196 | real | 0.43h | returned | **C4** | 与 i=153 相比唯一差异是 --learning-rate 5e-6→2e-6(起点顺延为 refresh_e1);prompt-mode / epochs / bs / accum / max-length / warmup / logging-steps 逐字相同,数据集完全一致。 |
| 226 | real | 1.31h | returned | **C4** | 与 i=110 逐字相同的一条命令,只把 --model-name-or-path 从 runs/full_exact_e2_stop_imend 换成 HuggingFaceTB/SmolLM3-3B-Base。数据配方一字未改,变的是多阶段课程的顺序/初始化——归 C4 属勉强(见 bounda… |
| 304 | real | 0.43h | returned | **C4** | 顺序实验的第二臂:与 i=153 的 refresh 配方逐字相同(exact / 1 epoch / bs 2 / accum 4 / max-length 4096 / lr 5e-6 / warmup 0.03),只是起点是 base_metamath100k_direct_e1。 |
| 448 | real | 2.52h | returned | **C3** | 与 i=110 相比唯一实质差异是 --metamath-count 100000→200000(--logging-steps 50→100 不影响训练);起点权重、prompt-mode、epochs、bs、accum、max-length、lr、warmup、默认 --seed 1234 全部… |
| 623 | real | 1.16h | returned | **both** | 起点 final_model;--metamath-count 200000→75000 且 --seed 1234→2026(换一批数据,C3),--learning-rate 1e-5→5e-6、--warmup-ratio 0.03→0.02(C4)。训练本身跑完,但保存阶段被 Generat… |

### 验证序列(22 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 34 | 3 | 20 | 是 |  | 0.0(未改动的 base 模型基线,n=20;不判定任何改动) |
| 90 | 3 | 50 | 是 | c1, c2 | 0.1(n=50;agent 读到 10%,判定为"不够") |
| 104 | 3 | 50 | 是 | c3 | 0.46(n=50;与 i=90 严格同权重、同 --limit 50 与同评测参数,唯一变量是 eos_token_i… |
| 168 | 3 | 50 | 是 | c6, c7 | 0.5(n=50;对比 i=104 的 0.46) |
| 174 | 3 | 150 | 是 | c6, c7 | 0.607(n=150) |
| 178 | 3 | 150 | 是 | c1, c2, c3 | 0.6(n=150;把 exact-only 基线拉到与 i=174 同样本量做对照) |
| 192 | 3 | 150 | 是 | c6 | 0.593(n=150;refresh 之前的纯 direct-MetaMath 阶段) |
| 210 | 3 | 150 | 是 | c8 | 0.567(n=150;低 LR 额外一轮回退,候选被弃) |
| 300 | 3 | 150 | 是 | c9 | 0.593(n=150;MetaMath-first 第一阶段) |
| 342 | 3 | 150 | 是 | c9 | 0.473(n=150;顺序实验第二臂,明显劣于 exact-first 的 0.607) |
| 362 | 3 | 150 | 是 | c10 | 0.593(n=150;alpha 0.5 soup 低于被平均的最好 checkpoint) |
| 369 | 3 | 150 | 是 | c11 | 0.567(n=150;alpha 0.8 soup 更差,权重平均整条线被放弃) |
| 382 | 3 | — | 是 | c6, c7 | 0.56;意图是全量,实际只跑了 evaluate.py 的默认 150 题(agent 当场发现),故这条不是第四档 |
| 394 | 3 | 150 | 是 | c12 | 0.52(n=150;同一份权重、同一批 150 题,只把评测参数换成脚本真实默认值,就从 i=174 的 0.607 … |
| 405 | 3 | 150 | 是 | c13 | 0.70(n=150;与 i=394 同权重同题同评测参数,唯一变量是新增的 do_sample/temperature… |
| 412 | 3 | 150 | 是 | c14 | 0.713(n=150) |
| 418 | 3 | 150 | 是 | c15 | 0.747(n=150;贪婪解码下候选排序相对采样解码完全反转) |
| 424 | 4 | 1319 | 是 | c6, c15 | 0.7467778620166793(全量 1319 题) |
| 570 | 3 | 150 | 是 | c17, c18 | 0.733(n=150;低于 100k 版的 0.747,但 agent 判断落在短样本噪声内,决定上全量) |
| 582 | 4 | 1319 | 是 | c17 | 0.755117513267627(全量 1319 题;与 i=424 构成 100k vs 200k 的单变量 C3 … |
| 600 | 3 | 150 | 是 | c19 | 0.733(n=150;确认 final_model 目录本身能加载,与 i=570 同数) |
| 787 | 4 | 1319 | 是 | c19 | 0.7573919636087946(全量 1319 题;与 i=582 是同一份权重同一份配置,却差 3 道题:0.7… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[47, 54]
- **分类学缺口提案 1 条**
  - eval_protocol_calibration(i=384, i=393, i=386)
- **定义缺陷 4 条**
  - (i=384, i=383)
  - (i=424, i=384)
  - (i=46, i=53)
  - (i=776, i=778, i=786)
- **边界情形 2 条**
  - 多阶段训练的"顺序 / 初始化 checkpoint"落在 C3 与 C4 之间。i=226 与 i=110 是逐字相同的命令,只换了 --model-name-or-path(base vs exact-SFT 后的 checkpoint):数据配方一字未改,超参一字未改,变的是同一批数据在课程里的先后。按 C3(数据来源与配方)判不了——数据没变;按 C4(训练方法与超参)判也勉强——没有任何…(i=226, i=110, i=233)
  - C1 与 C5 的归属交叉:同一批 checkpoint 的排序被解码配置整个翻转。采样解码下三个候选读到 refresh_e1 0.607 > exact_stop 0.600 > meta_direct 0.593;换上贪婪 generation_config 后同样 150 题读到 meta_direct 0.747 > exact_stop 0.713 > refresh_e1 0.700…(i=181, i=421, i=414)

## codex_non_api_max_gpt-5.6-sol_10h_run2__gsm8k_Qwen_Qwen3-1.7B-Base_17404247
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gsm8k | Qwen_Qwen3-1.7B-Base | 7.54h | 0.6239575435… |

### 改动序列(29 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 83 | C2 | Format alignment to the evaluator: every training response is rewritten to end with a bare 'ANSWER: N' line matching evaluate.py's prompt, and <\|im_e… | i=83, i=307, i=173 |
| 90 | C3 | Build the 'broad' training corpus: 302,847 dedup'd examples mixed from Orca-Math (151,021) + OpenMathInstruct-2 augmented-gsm8k (70,669) + MetaMathQA … | i=90, i=110, i=307 |
| 124 | C4 | After the bs=16 launch OOM'd in the first backward pass, halve the microbatch and double accumulation (16/4 -> 8/8, same effective batch 64) and add P… | i=124, i=123 |
| 146 | C4 | Patch train_sft.py to disable the Transformers length-grouped sampler (group_by_length) after it spent minutes on Arrow scalar lookups before the firs… | i=145, i=146 |
| 195 | C2 | Build data/system_train.jsonl: 7,371 GSM8K-train-only examples wrapped in the evaluator's exact fixed 10-shot system turn (5,872 characters, reproduce… | i=195, i=196, i=361 |
| 287 | C1 | Whole-file rewrite of runs/broad/checkpoint-2000/generation_config.json plus a tokenizer re-save with eos_token='<\|im_end\|>'. FIELD-LEVEL DIFF again… | i=287, i=285, i=419 |
| 322 | C1 | Second C1 edit on the same file: eos_token_id scalar 151645 -> list [151643, 151645] (accept either Qwen terminator), and train_sft.py's saver patched… | i=322, i=323, i=414, i=328 |
| 330 | C7 | Self-built proxy scoring: re-parse the inspect_ai eval log and compute 'first ANSWER: correct', 'last ANSWER: correct', answer-tag rate, multi-answer … | i=348, i=941, i=908 |
| 595 | C1 | Package the finished runs/broad root artifact: config.json use_cache False->True and generation_config.json rewritten from the Trainer-emitted 117 byt… | i=595, i=596, i=952 |
| 617 | C3 | Second-stage 'focus' curriculum from runs/broad on the 122,890-row GSM-specific corpus; learning rate also dropped 2e-5 -> 8e-6 and --no-checkpoints a… | i=617, i=496 |
| 709 | C3 | Self-generated preference corpus: sample 6 completions per contamination-filtered GSM8K-train question from runs/broad; chosen is always the human gol… | i=709, i=736, i=225 |
| 737 | C4 | Reference-free ORPO preference stage on runs/broad: 5 epochs, lr 5e-7, beta 0.1, ~500 updates. | i=737, i=736 |
| 766 | C4 | New training objective: --eos-weight 16 upweights the terminal <\|im_end\|> token in the response-only cross-entropy, applied to the exact-context cor… | i=766, i=765 |
| 800 | C4 | --eos-only: supervise ONLY the terminal token, giving no target label to any reasoning token, at lr 5e-7. Motivated by the c12 result (termination fix… | i=800, i=796 |
| 836 | C4 | --eos-binary --eos-weight 16: binary stop-vs-continue objective adding explicit 'do not stop yet' supervision at every non-final response position, lr… | i=836, i=835 |
| 858 | C3 | Rejection-sampled self-distillation corpus: 4 samples per GSM8K-train question from runs/eos_binary in the exact evaluator context, keep the shortest … | i=858, i=870 |
| 871 | C4 | ReST/rejection SFT on the c15 corpus from runs/eos_binary at lr 8e-7 with the EOS weight relaxed from 16 to 4. | i=871, i=870 |
| 933 | C3 | Exact-context preference corpus: 8 samples per question from runs/eos_binary inside the full evaluator few-shot prefix; keep pairs needing both a veri… | i=933, i=960 |
| 972 | C4 | Custom frozen-reference DPO (self-written train_dpo_exact.py with projected response logits to avoid materialising 151k vocab over the ~2,180-token sh… | i=972, i=924 |
| 1010 | C3 | Low-LR continuation of the leader on the ORIGINAL 301,991-row broad corpus (lr 2e-6, max-steps 1000, max-length 1024) to test whether more general mat… | i=1010, i=1009 |
| 1045 | C6 | Weight-space interpolation runs/eos_binary x runs/rest_sft at alpha=0.25 via a self-written interpolate_models.py, used as a cheap probe of 'a smaller… | i=1045, i=1039 |
| 1057 | C4 | Second binary stop-calibration pass, identical data/objective/bs/accum to c14, only lr 8e-7 -> 3e-7 and parent runs/broad -> runs/eos_binary. Produced… | i=1057, i=1053 |
| 1072 | C6 | Weight-space interpolation runs/eos_binary x runs/eos_binary2 at alpha=0.5 to trade premature stopping against spillover. | i=1072, i=1071 |
| 1089 | C5 | Resurrect the retained step-4,000 broad-SFT checkpoint as an evaluable candidate by repackaging its weights (alpha=1.0) with runs/broad's tokenizer/co… | i=1089, i=1088 |
| 1098 | C4 | Apply the c14 binary calibration recipe verbatim (same data, objective, lr 8e-7, bs 2 x accum 8) to the step-4,000 parent instead of the step-4,719 pa… | i=1098, i=1097 |
| 1117 | C6 | Weight-space interpolation runs/eos_binary x runs/eos_binary2 at alpha=0.25. | i=1117, i=1116 |
| 1132 | proposed:verifier-protoc… | Change the VERIFIER, not the artifact: --max-tokens 1024 -> 4000 (and --max-connections 32 -> 16 in the same command) on the official evaluate.py, to … | i=1132, i=1137, i=1139 |
| 1158 | proposed:weight-space-sc… | EXTRAPOLATION past the trained endpoint: interpolate_models.py patched to allow alpha>1 and run at alpha=1.5 along the eos_binary -> eos_binary2 delta… | i=1158, i=1155 |
| 1174 | C5 | Final candidate selection and packaging: runs/eos_binary2 chosen on the full 1,319-item paired 4,000-token sweep and copied to final_model (alpha=0.0 … | i=1174, i=1168, i=1179 |

### 训练序列(14 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 111 | real | 0.06h | returned | **both** | baseline -- first training of the run; simultaneously introduces the C3 broad corpus (302,847 rows) and the C4 recipe (full-parameter BF16 SFT, not Lo… |
| 124 | real | 0.04h | returned | **C4** | vs i=111: microbatch 16->8, grad-accum 4->8 (effective batch 64 unchanged), + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Same data, same lr. Ab… |
| 146 | real | 0.95h | returned | **C4** | vs i=124: CLI byte-identical; train_sft.py patched to disable the group_by_length sampler. This is the launch that actually trained (to step 2,015, th… |
| 298 | real | 1.53h | returned | **unclear** | vs i=146: nothing under test. Same command plus --resume-from-checkpoint runs/broad/checkpoint-2000; it simply finishes the run the agent had paused a… |
| 617 | real | 0.62h | returned | **both** | vs i=298: data broad_train.jsonl (302,847) -> focus_train.jsonl (122,890, GSM-specific verified traces); parent Qwen3-1.7B-Base -> runs/broad; lr 2e-5… |
| 737 | real | 0.28h | returned | **both** | vs i=617: data -> 3,188 self-generated preference pairs; objective SFT -> ORPO (reference-free preference); parent runs/broad (branching back, not fro… |
| 766 | real | 0.20h | returned | **both** | vs i=737: data -> system_train.jsonl (7,371 rows in the evaluator's exact 10-shot context); objective ORPO -> weighted-EOS SFT (--eos-weight 16); pare… |
| 800 | real | 0.20h | returned | **C4** | vs i=766: SAME data, SAME parent (runs/broad), same bs/accum/max-length. Only the objective and lr change: --eos-weight 16 -> --eos-only (no label on … |
| 836 | real | 0.20h | returned | **C4** | vs i=800: SAME data, SAME parent, same bs/accum/max-length. Objective --eos-only -> --eos-binary --eos-weight 16 (binary stop-vs-continue at every res… |
| 871 | real | 0.21h | returned | **both** | vs i=836: data system_train.jsonl -> rejection_sft.jsonl (7,371 self-generated verified traces); parent runs/broad -> runs/eos_binary; objective --eos… |
| 972 | real | 0.32h | returned | **both** | vs i=871: data -> exact_preferences.jsonl (3,616 in-context preference pairs); objective SFT -> custom frozen-reference DPO (beta 0.1); lr 8e-7 -> 5e-… |
| 1010 | real | 0.48h | returned | **both** | vs i=972: data -> back to broad_train.jsonl (301,991 retained); objective DPO -> plain SFT; lr 5e-7 -> 2e-6; max-length 3072 -> 1024; bs 2 -> 8, accum… |
| 1057 | real | 0.20h | returned | **C4** | vs i=1010: back to the i=836 recipe verbatim (system_train.jsonl, --eos-binary --eos-weight 16, max-length 3072, bs 2 x accum 8). Two fields differ fr… |
| 1098 | real | 0.20h | returned | **C4** | vs i=836 the command is byte-identical except --model runs/broad4000_pack (instead of runs/broad) and --output. Coded C4 because --model is a training… |

### 验证序列(26 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 17 | 3 | 50 | 是 |  | 0.06 (base Qwen3-1.7B-Base, n=50) -- reference point, judges… |
| 291 | 3 | 150 | 是 | c1, c2, c3, c4, c5 | 0.3933333333333333 (59/150) |
| 598 | 3 | 150 | 是 | c1, c2, c6, c8 | 0.44 (66/150); self-diagnostic: first-answer 0.60, non-stop … |
| 696 | 3 | 150 | 是 | c7 | 0.4066666666666667; first-answer fell 0.60 -> 0.487, so reje… |
| 759 | 3 | 150 | 是 | c9, c10 | 0.38; first-answer 0.593, spillover up -- ORPO rejected |
| 788 | 3 | 150 | 是 | c11, c12 | 0.447; non-stop 0.34 -> 0.04 but first-answer 0.60 -> 0.46 |
| 829 | 3 | 150 | 是 | c13 | 0.553; first-answer 0.593 preserved, non-stop 0.167 |
| 851 | 3 | 150 | 是 | c14 | 0.62; first-answer 0.627, non-stop 0.067, multi-answer 0.04 … |
| 887 | 3 | 150 | 是 | c15, c16 | 0.5733333333333334; first-answer 0.607 < 0.627 -- rejected |
| 1001 | 3 | 150 | 是 | c17, c18 | 0.58; first-answer 0.607 -- rejected |
| 1030 | 3 | 150 | 是 | c19 | 0.567; first-answer 0.6066666666666667 -- rejected |
| 1047 | 3 | 150 | 是 | c20 | 0.62 (ties the leader); first-answer 0.607 -- rejected |
| 1066 | 3 | 150 | 是 | c21 | 0.62 (ties); max-length failures 0.067 -> 0.027 but answer-t… |
| 1074 | 3 | 150 | 是 | c22 | 0.633 -- best 150-item result seen, +2 answers over 0.62 |
| 1081 | 3 | 300 | 是 | c22 | 0.60 at n=300 -- the n=150 gain does not survive |
| 1084 | 3 | 300 | 是 | c14, c22 | 0.61 at n=300 -- paired control arm for i=1081; original bea… |
| 1091 | 3 | 150 | 是 | c23 | 0.42; first-answer 0.6133333333333333 (higher than the final… |
| 1110 | 3 | 150 | 是 | c24 | 0.607; first-answer also 0.607 -- rejected |
| 1119 | 3 | 300 | 是 | c25 | 0.613 at n=300, one answer above the original's 0.61 -- judg… |
| 1123 | 4 | -1 | 是 | c25 | 0.5898407884761183 (full 1,319, 1024-token cap) |
| 1126 | 4 | -1 | 是 | c14, c25 | 0.6163760424564063 (812/1,319, 1024-token cap) -- paired con… |
| 1132 | 4 | -1 | 是 | c14, c28 | 0.5883244882486732 -- same weights, same 1,319 items as i=11… |
| 1140 | 4 | -1 | 是 | c21, c28 | 0.6186504927975739 (816/1,319) -- c21 loses at n=150/300 but… |
| 1148 | 4 | -1 | 是 | c22 | 0.5936315390447309 (full, 4000-token cap) |
| 1161 | 4 | -1 | 是 | c26 | 0.6057619408642911 -- alpha=1.5 falls below alpha=1.0 (0.618… |
| 1191 | 3 | 150 | 是 | c27 | 0.633 (n=150 at the 4000-token cap) -- packaging reproductio… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[298]
- **分类学缺口提案 2 条**
  - weight-space-scaling(i=1158, i=1155, i=1165, i=1072)
  - verifier-protocol-change(i=1132, i=1126, i=1137, i=1145, i=1139)
- **定义缺陷 5 条**
  - (i=287, i=419, i=285)
  - (i=1178, i=1179)
  - (i=90, i=107, i=196)
  - (i=1098, i=836, i=1097)
  - (i=285, i=419, i=952)
- **边界情形 5 条**
  - A resumption of a training the agent had deliberately paused mid-run to evaluate an intermediate checkpoint. Same command plus --resume-from-checkpoint; the corpus, objective and hyperparameters are u…(i=298, i=280, i=297)
  - Disabling the group_by_length sampler is formally a TrainingArguments change (C4), but the intent is not to raise the score -- the previous two launches never reached a first optimizer step. The chang…(i=145, i=123, i=119)
  - The action is a training with a fixed, already-validated recipe; the only free variable is which checkpoint it starts from. Classified by ACTION it is C4; classified by HYPOTHESIS it is C5. The curren…(i=1098, i=1097, i=1089)
  - The agent's 'first-answer accuracy / non-stop rate / multi-answer rate' diagnostics are computed by re-parsing the inspect_ai log that the OFFICIAL evaluation just produced. They cost zero extra GPU t…(i=348, i=910, i=1053)
  - This run attacks one failure mode -- the model not emitting a terminator -- from both sides at once, and the taxonomy splits them. Accepting either terminator is a C1 edit to generation_config.json (c…(i=328, i=835, i=833)
