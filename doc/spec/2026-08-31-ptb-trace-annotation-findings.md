# PostTrainBench 轨迹标注:逐 run 详情

规格 `doc/spec/2026-08-31-ptb-trace-annotation.md`。方法论 `doc/reference/verifier_tiers_and_change_types.md`。
本文件是**记录**,不是结论;reference 里的每个数字都应能走回这里的某个 run 段落。

**批次状态**:81 条 run · 2116 条改动 · 818 段训练 · 1593 次验证。
指针校验 5256 条判断,作废 1 条(**0.02%**)。
提案:101 类别 / 357 定义缺陷 / 270 边界情形。

`tested_variable` 已按 spec §10 第 1 条拆分:`smoke` 与 `baseline` 不再占用 `unclear`。

---

## claude_non_api_claude-opus-4-7_10h__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17129537
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 7.51h | 0.1 |

### 改动序列(22 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 107 | C3 | 选定 v1 训练数据来源与配方:OpenR1-Math-220k 十个 parquet 分片,每题取第一条 correctness_math_verify 判对的 generation,丢掉 >28000 字符的长 trace,目标 8000 条。 | i=107, i=107 |
| 107 | C2 | 同一个脚本里做格式对齐:user 侧逐字抄 inspect_evals/aime 的提示词、assistant 侧补一行 "ANSWER: X"、chat_template 直接读评测用的 templates/smollm.jinja、add_generation_prompt=False,使训练样… | i=107, i=107, i=107 |
| 120 | C11 | 写脚本解析官方 inspect_ai 日志的 samples 数组,打印每题 id / target / 模型输出首尾,把 evaluate.py 只给出的一个 accuracy 标量展开成逐题可读信号。 | i=120, i=120 |
| 134 | C3 | 收紧数据长度过滤并加大样本量:MAX_TOKENS 8192→6144、TARGET_SAMPLES 8000→10000,自述动机是「smaller max length for faster training」。 | i=134, i=134, i=133 |
| 138 | C4 | v1 训练方法:TRL SFTTrainer 全参 bf16 SFT(非 LoRA),lr 2e-5 cosine、warmup 0.03、bs1×accum16、gradient_checkpointing、packing bfd + padding_free、max_length 6144、se… | i=138, i=138, i=138 |
| 145 | proposed:smoke_harness | 自建第二档冒烟工装:head -200 切出 train_smoke.jsonl,复制 train.py 的全部管线设置但 max_steps=3、save_strategy="no"、输出到 work/smoke,末尾 print("SMOKE OK")。目的是在花掉一次真实训练之前抓出实现错误。 | i=145, i=145, i=143 |
| 151 | C4 | 冒烟通过后把 v1 的 num_train_epochs 从 1 改成 2(数据未变)。 | i=151, i=150 |
| 194 | C11 | 写 eval_and_save.py:把 evaluate.py 包成带默认参数的函数、跑完自动读回 json 指标并打印。注:该脚本此后从未被调用,全部评测仍是手敲 evaluate.py。 | i=194, i=194 |
| 252 | C8 | 删掉 sft_run1 的四个中间 checkpoint(200/400/600/792)腾磁盘,目录从 41G 降到 5.8G。副作用是 C5(checkpoint 选择)在这条 run 上被永久排除。 | i=252, i=254, i=255 |
| 257 | C9 | 首次写 final_model:把 sft_run1(0.1)整目录拷成 final_model 作为兜底提交,不改动任何权重。 | i=257, i=249 |
| 269 | C3 | v2 数据配方:OpenR1 改成「取最短的正确 generation」(800–35000 字符窗口)取 8000 条,另加 Bespoke-Stratos-17k 里含 \boxed 的 3000 条并从 \boxed 反解答案,长度上限放宽到 8192,rng 种子 1234。 | i=269, i=269, i=269 |
| 273 | C4 | v2 训练方法:改成从 final_model(= sft_run1 权重)续训而不是从 base 起,lr 2e-5→1e-5、epoch 2→1、max_length 6144→8192、seed 42→123。 | i=273, i=273, i=273 |
| 290 | C8 | v2 测出 0.067(低于 v1 的 0.1)后,rm -rf 掉整个 sft_run2 权重目录。此后 v2 无法再被复评或参与权重平均。 | i=290, i=288 |
| 292 | C3 | v3 数据配方:放弃 Stratos,只用 OpenR1,按 correctness_count>=2 且 source∈{amc_aime, olympiads, aops_forum} 做高质量筛,取最短正确 trace(800–25000 字符),长度回到 6144,种子 77,实得 9936… | i=292, i=292, i=295 |
| 296 | C4 | v3 训练方法:回到从 base 起、2 epoch,lr 设 1.5e-5(注释写明「Slightly lower than v1」),weight_decay 0→0.01,save_strategy 由 steps 改成 "no"(不再存中间 checkpoint),seed 77。 | i=296, i=296, i=296 |
| 314 | C9 | v3 测出 0.167 > v1 的 0.1 后把 final_model 换成 sft_run3。做法是先 rm -rf final_model 再 cp -r,非原子:此后约 7 分钟里 final_model 只剩 3 个文件。 | i=314, i=314, i=313 |
| 342 | C11 | 把官方 inspect_ai 日志转成逐题表:score/extracted answer/reasoning_chars/text_chars/stop_reason,由此读出前 15 题里有 4 题 stop='max_tokens'(16000 token 上限截断)。该截断信号没有转化成任何… | i=342, i=342, i=343 |
| 350 | C3 | v4 数据配方:把来源过滤从「amc_aime/olympiads/aops_forum」放宽成「任意 source,只要 correctness_count>=2」,规模 9936→20000,种子 2025,长度过滤仍是 6144。 | i=350, i=350, i=350 |
| 372 | C4 | v4 训练方法:相对 v3 只动了 num_train_epochs 2→1,agent 自述这是为了在数据翻倍时保持总步数与 v3 相当(lr/weight_decay/max_length/accum 与 v3 逐字相同)。 | i=372, i=372, i=371 |
| 460 | C9 | v4 测出 0.2 > v3 的 0.167 后把 final_model 换成 sft_run4(rm -rf final_model/* 再 cp -r)。 | i=460, i=459 |
| 476 | C11 | 改用重复评测取均值来把噪声与真实差异分开:先 sha256 确认 final_model 与 sft_run4 权重逐字节相同,再对同一份权重连跑第三次全量评测,并把 sft_run3 也补到三次。 | i=475, i=476, i=472 |
| 497 | C9 | 重复评测显示 v3 与 v4 三次均值都是 14.4% 后,决定不回滚,保留 v4 在 final_model 里(理由是数据更多更 diverse、成绩打平)。这是本条 run 的最终提交决定。 | i=497 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 147 | smoke | 0.01h | returned | **smoke** | 冒烟,不测任何变量:200 条子集 + max_steps=3 跑通 train.py 的管线,train_runtime 6.5 秒,print SMOKE OK 后产物 work/smoke 立即被 rm 掉。`tested_variable` 的取值域(C3/C4/both/unclear)装… |
| 155 | real | 1.55h | consumed | **baseline** | baseline —— 本条 run 的第一次真实训练,没有上一次可比。数据 = OpenR1 10000 条(c1+c3),格式 = 对齐评测 prompt(c2),方法 = 全参 SFT lr 2e-5 / 2 epoch(c4+c6)。同样是取值域装不下 baseline,不是证据不足。 |
| 275 | real | 0.92h | consumed | **both** | 相对 v1 同时改数据和方法。数据:换成 data_v2(OpenR1 改取最短正确 trace + 新增 Bespoke-Stratos 3000 条,上限 8192,种子 1234)。方法:起点从 base 改成 final_model 续训、lr 2e-5→1e-5、epoch 2→1、max… |
| 298 | real | 1.42h | consumed | **both** | v2 已被判负并删除,agent 明确以 v1 为参照(注释「Slightly lower than v1」)。数据:换成 data_v3 高质量过滤(correctness_count>=2 且 amc_aime/olympiads/aops_forum,9936 条,回到 6144)。方法:仍从… |
| 378 | real | 1.72h | consumed | **both** | 相对 v3:数据 9936→20000 且来源过滤从三个 source 放宽到任意 source(种子 77→2025);超参只有 num_train_epochs 2→1,其余(lr 1.5e-5、weight_decay 0.01、max_length 6144、bs1×accum16)与 v3… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 84 | 3.0 | 5.0 | 是 |  | 0.0 |
| 221 | — | — | 是 | c1, c2, c3, c4, c6 | 0.1 |
| 287 | — | — | 是 | c11, c12 | 0.067 |
| 310 | — | — | 是 | c14, c15 | 0.167 |
| 445 | — | — | 是 | c18, c19 | 0.2 |
| 462 | — | — | 是 | c20 | 0.1 |
| 476 | — | — | 是 | c18, c19, c21 | 0.13333333333333333 |
| 485 | — | — | 是 | c14, c15, c21 | 0.1 |
| 491 | — | — | 是 | c14, c15, c21 | 0.16666666666666666 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - smoke_harness(i=145, i=145, i=143, i=142)
- **定义缺陷 3 条**
  - (i=138, i=219, i=221, i=285, i=308, i=437)
  - (i=145, i=145, i=148, i=153)
  - (i=473, i=457, i=467, i=480, i=452, i=470, i=497)
- **边界情形 3 条**
  - v4 相对 v3 唯一的超参变动是 num_train_epochs 2→1,而 agent 在脚本 docstring 和 text 里两次写明这是为了在数据量翻倍时保持总步数与 v3 相当。按 reference §3 现行定义(C4 = 「lr/epoch/batch/序列长度」)epoch 变了就得记 both;按 spec §10.2 拟议的判据(「若超参变动是为保持某个量不变而做的补偿…(i=372, i=371, i=372)
  - 本条 run 5 行训练里有 2 行(i=147 冒烟、i=155 首次真实训练)在 C3/C4/both/unclear 里没有正确取值,只能填 unclear。这两行的证据都是充分的 —— 冒烟的 max_steps=3 和 baseline 的「没有上一次」都写在事件流里。于是 unclear 比例 40% 中的全部 40% 是取值域装不下,0% 是证据不足。spec §10.1 已把补 s…(i=147, i=145, i=155)
  - i=290 的 `rm -rf work/sft_run2` 判不了。C8 的定义是「让管线跑得起来」的可行性修复,这里管线没坏,同一条命令里也没有任何磁盘检查(相比之下 i=252 的 checkpoint 清理带着 du -sh / df -h,是干净的 C8);C9 的定义是「不改任何产物,只决定把哪个候选写进 final_model」,而这个动作恰恰不可逆地销毁了一份产物、也没碰 fina…(i=290, i=254, i=482)

## claude_non_api_claude-opus-4-7_10h__aime2025_Qwen_Qwen3-1.7B-Base_17116642
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 7.76h | 0.1 |

### 改动序列(22 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 39 | C3 | 选定训练数据来源与配方:流式读 open-r1/OpenR1-Math-220k,只保留 correctness_math_verify 与 is_reasoning_complete 同时为真、且含 <think>/</think> 的第一条生成,再按 200–8192 token 过滤,取 20… | i=39, i=39 |
| 58 | C2 | 训练侧格式对齐评测侧:训练脚本把评测用的同一份 templates/qwen3.jinja 装成 tokenizer.chat_template,并把 <\|im_end\|> 指定为训练用 eos,使模型学会在评测模板的 assistant 边界收尾。 | i=58, i=58 |
| 58 | C4 | 初版 SFT 配方:对 Qwen3-1.7B-Base 全参微调,1 epoch、lr 1e-5、cosine、bs1×accum16、max_length 8192、bf16 + flash_attention_2 + gradient checkpointing、completion_only_… | i=58, i=58 |
| 79 | C4 | 启动前调超参:per_device_train_batch_size 1→2、gradient_accumulation_steps 16→8(有效 batch 仍是 16)、warmup_ratio 0.05→0.03、save_strategy epoch→no、packing False→Tr… | i=79, i=79 |
| 123 | C8 | 运行时修复:i=82 启动的训练随 harness 后台任务一起被回收(i=114 的 nvidia-smi 已显示 0MiB / No running processes,sft_run1 目录为空)。改用 nohup + disown 脱离 harness 重启同一脚本,并先 rm -rf 掉上… | i=114, i=123, i=123 |
| 194 | C9 | 提交守卫:把 sft_run1 的产物整目录拷进 final_model,首次让一个候选进入提交位。 | i=194 |
| 204 | C11 | 验证器工装:自写脚本读 inspect_ai 的 JSON 日志,把官方评分器的输出拆成可决策信号——逐题打印 target、assistant 输出长度、HEAD/TAIL 与 aime_scorer 的 answer。据此判定 0.0 的成因是解码乱码(多语言碎片),而不是数学能力。 | i=204, i=205 |
| 225 | C1 | 解码配置:整份重写 final_model/generation_config.json,加 do_sample=true / temperature 0.6 / top_p 0.95 / top_k 20 与 bos_token_id。整份重写顺手删掉了原有的 max_new_tokens: 20… | i=222, i=225, i=225 |
| 251 | C7 | 自建代理验证器(行为探针 v1):不跑 evaluate.py,直接用 transformers 加载 checkpoint,对比「模板默认 prompt」与「prompt 后手工补 <think>\n」两种起点的贪婪续写。默认起点直接吐 <\|im_end\|>,补了前缀才正常推理——用这个零评测… | i=251, i=252 |
| 270 | C4 | completion_only_loss True→False,SFTConfig 其余字段逐字未动;意图是让 <think> 前缀参与 loss。 | i=270 |
| 321 | C7 | 行为探针 v2:打印 assistant 生成边界上的 top-10 next-token 概率 + 三次 temp=0.6 采样。sft_run2 的 top-1 是 <\|im_end\|>(0.0771),三次采样全是 <\|im_end\|>,据此直接判 sft_run2 报废——这次判定完… | i=321, i=322 |
| 334 | C4 | 第三版训练脚本 train_sft2.py:数据在进 trainer 之前先用 apply_chat_template 渲染成纯 text,改走 dataset_text_field='text' 的整序列 LM loss(不再传 messages),同时把 lr 从 1e-5 抬到 1.5e-5。 | i=334, i=334 |
| 385 | C9 | 提交守卫:rm -rf final_model 后把 sft_run3 整目录拷进提交位,替换掉 sft_run1 那份候选。 | i=385 |
| 385 | C1 | 解码配置(增量版):用 python 在 final_model/generation_config.json 上做 dict.update,加 do_sample / temperature 0.6 / top_p 0.95 / top_k 20。i=386 的 stdout 逐字打印了改前改后—… | i=385, i=386 |
| 395 | C1 | 解码配置(整份重写版):同一文件再被 Write 整份覆盖。字段级差异为 eos_token_id [151643] → [151645, 151643](把 <\|im_end\|> 加进停止符)、新增 bos_token_id 151643、并把 max_new_tokens: 2048 整个删… | i=393, i=395 |
| 400 | C11 | 验证器工装(第二次):读 limit=10 那次评测的 inspect_ai 日志,逐题打印 target / answer / assistant 输出字符长度。i=402 显示 10 题里 2 题 len=0(空输出),其余答案全错——把单一的「0.0」拆成了「空输出」与「答错」两种失效模式。 | i=400, i=402 |
| 407 | C3 | 第二批数据:同一 OpenR1-Math-220k 流,用 SKIP=20000 跳过已用过的前 20k,再按同样的筛选条件取 20,000 条写 train2.jsonl(i=415 确认 kept=20000 / skipped=17859)。 | i=407, i=415 |
| 417 | C3 | 训练语料从 20k 扩到 40k:train.jsonl 与 train2.jsonl 一起加载后打乱。 | i=417, i=417 |
| 417 | C4 | 改成从 sft_run3 的 checkpoint 续训而不是从 base 起训,并把 lr 降到 7e-6、warmup_ratio 0.03→0.02、seed 42→43。 | i=417, i=417 |
| 509 | C1 | 对 sft_run4/generation_config.json 做同一套整份重写:eos_token_id [151643] → [151645, 151643]、新增 bos_token_id、加采样四件套、删掉 max_new_tokens: 2048(i=506 读到的原文里有该字段)。 | i=506, i=509, i=509 |
| 511 | C9 | 提交守卫:rm -rf final_model 后把 sft_run4 拷进提交位,并当场 cat 回读确认落进去的 generation_config.json 就是改过的那份。注意这是无条件覆盖——sft_run4 此时还没有任何评测分数。 | i=511 |
| 534 | C9 | 提交守卫(收尾):拿到 0.0667 之后先把 final_model 整目录备份成 final_model_v1_backup,随即判断「再训练的退化风险大于收益」而停手;i=540 又把备份删掉腾磁盘,提交位保持 sft_run4 那一份。 | i=534, i=540 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 82 | real | 0.19h | discarded | **baseline** | baseline —— 这条 run 的第一次训练。schema 没有 baseline 取值,只能填 unclear;这不是证据不足。 |
| 123 | real | 1.35h | consumed | **unclear** | 纯重启:与 i=82 用的是逐字同一个 train_sft.py 和同一份 SFTConfig(两次启动之间没有任何 Edit 事件)。唯一变化是启动方式从 harness 后台任务改成 nohup+disown(C8,见 c5),不测任何受测变量。schema 没有 restart 取值。 |
| 272 | real | 5.83h | run_end | **C4** | 相对 i=123 只改了一项:completion_only_loss True→False(i=270 的 Edit,old_string/new_string 除这一行外逐字相同)。数据仍是同一份 train.jsonl 20k,lr 1e-5 / bs2×accum8 / packing bf… |
| 336 | real | 1.36h | consumed | **C4** | 换成 train_sft2.py:同一份 train.jsonl 20k(样本集合不变),改的是喂法——先 apply_chat_template 渲染成 text 再用 dataset_text_field='text' 走整序列 LM loss,不再传 messages;同时 lr 1e-5→1… |
| 431 | real | 2.67h | consumed | **both** | 相对 i=336 同时动了两类:C3——训练语料 20k(train.jsonl)扩成 40k(train.jsonl + train2.jsonl,后者是 OpenR1 流里 SKIP 掉前 20k 之后的新样本);C4——初始权重从 base 换成 sft_run3 的 checkpoint 续… |

### 验证序列(4 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 196 | 3.0 | 5.0 | 是 | c1, c3, c4, c6 | 0.0 |
| 227 | 3.0 | 5.0 | 是 | c8 | 0.0 |
| 397 | 3.0 | 10.0 | 是 | c12, c13, c14, c15 | 0.0 |
| 517 | — | — | 是 | c18, c19, c20, c21 | 0.0666666666666667 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[123]
- **分类学缺口提案 1 条**
  - proposed:behavioral-probe(i=251, i=252, i=321, i=322, i=382, i=384)
- **定义缺陷 1 条**
  - (i=123, i=114, i=191, i=316, i=378, i=501, i=334)
- **边界情形 2 条**
  - (i=254, i=334, i=322)
  - (i=82, i=123, i=114)

## claude_non_api_claude-opus-4-7_10h__aime2025_Qwen_Qwen3-4B-Base_17107174
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 8.96h | — |

### 改动序列(28 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 94 | C2 | 格式对齐:prepare_data.py 把 inspect_evals 的 AIME 用户提示词逐字抄进训练样本,assistant 侧统一收在 `ANSWER: {numeric}`,让训练文本与评测时模型真实看到的 prompt 一致。 | i=94, i=94 |
| 94 | C3 | v1 数据配方:OpenR1-Math-220k,按 correctness_math_verify & is_reasoning_complete & 末尾 \boxed{} 可解析成整数 & ≤8192 token 过滤,取 6000 条。 | i=94, i=97 |
| 124 | C8 | 冒烟方式修复:i=110 用「后台启动 + sleep 180 + kill -9」跑冒烟,结果 checkpoints/smoke 是空的、输出文件只有 `=== done ===`;改成前台 `--epochs 0.01` 跑满 30 步才拿到可读的管线验证。 | i=124, i=118 |
| 139 | proposed:cross_run_memor… | 把环境事实(H100/包版本/HF cache 位置)、评分器行为与训练配方写进 harness 提供的持久 memory 文件;i=451 又补上 v1–v6 的完整结果表与结论。收益对象是后续 run,不是本 run 的分数,C1–C11 都装不下。 | i=139, i=139, i=451 |
| 170 | C8 | 磁盘可行性:训练完删中间 checkpoint,run1 目录 53G → 7.6G。此后每次训练结束都重复(i=231/264/316/341)。 | i=170, i=171 |
| 172 | C9 | 首次提交守卫:把 final_model 软链到 checkpoints/run1。本 run 全部 6 次提交都用 `ln -sfn <ckpt> final_model` 这个惯用法,不产生新权重。 | i=172 |
| 192 | C1 | 解码配置:整份重写 final_model/generation_config.json —— eos_token_id 由 [151643] 改成 [151643,151645](加 <\|im_end\|>),**同时顺手删掉了原有的 max_new_tokens:2048**。两个字段一起变,… | i=182, i=192, i=185 |
| 205 | C11 | 验证器工装:自写脚本读 inspect_ai 的 json log,把官方评分器的输出转成可决策信号 —— 每题分成 CORRECT / NO_ANSWER / WRONG,并打印 completion 长度,用来判断失分是不是截断。 | i=205, i=206 |
| 214 | C3 | v2 数据配方:5000 条 OpenR1 + 518 条 jonathanyin/aime_1983_2023 历史 AIME 推理链、按 3 倍上采样,合计 6554 条;同时改成「同一题的多条正确生成里取最短的一条」。 | i=214, i=214, i=217 |
| 214 | C10 | 去污染:v2 里显式挂 `EXCLUDE_AIME_2024_2025 = True  # paranoia filter`,v5 里同样以 `if year >= 2024: continue  # paranoia` 剔除 2024/2025 年的历史题,避免与 AIME2025 测试集重叠。 | i=214, i=324 |
| 219 | C9 | 候选保全:启动 run2 之前先 `mv checkpoints/run1 checkpoints/run1_backup` 并删掉 final_model 软链,把 v1 权重从后续覆盖中摘出来。这一步是 i=237 能回滚的前提。 | i=219 |
| 231 | C1 | 对 run2 重放同一份解码配置改动,这次是显式的 python 两行:`c['eos_token_id'] = [151643, 151645]` 与 `c.pop('max_new_tokens', None)`;结果在同一事件里打印出来。 | i=231, i=232 |
| 237 | C9 | 提交回滚:v2 全量 0.033 低于 v1 的 0.133,把 final_model 软链改回 run1_backup;不改任何权重,只换提交对象。 | i=237, i=236 |
| 245 | C3 | v3 数据配方:只保留 ≤4000 token 的短推理链、答案域收紧到 0–999(AIME 形状),共 8000 条。明写的假设是 v2 变差是因为回答太长撞上 max_tokens。 | i=245, i=245, i=244 |
| 264 | C1 | 对 run3 重放同一份解码配置改动(eos_token_id 加 151645、pop 掉 max_new_tokens),与该次评测同一条命令。 | i=264, i=265 |
| 281 | C11 | 验证器工装:自写脚本扫 logs/*.json,把每次全量评测答对的题号集合并排打出来,用于判断分数变化是不是落在同一批题上。 | i=281, i=282 |
| 285 | C3 | v4 数据配方:12000 条 OpenR1,≤6000 token,落盘格式从渲染好的 text 改成 messages(user/assistant)以便配合 assistant_only_loss。 | i=285, i=285, i=291 |
| 292 | C4 | 方法改动:新写 train_v4.py,开 `assistant_only_loss=True` 并通过 `chat_template_path` 指定训练用模板,把损失从全序列改成只算 assistant 段;max_length 6144。 | i=292, i=292 |
| 298 | C8 | 可行性修复:c18 首次启动(i=294)90 秒即崩,报模板缺 `{% generation %}`;于是手写 qwen3_train.jinja,在 assistant 段包上 generation 标记,并在 i=300 用 return_assistant_tokens_mask 做了一次零… | i=298, i=295, i=301 |
| 316 | C1 | 对 run4 重放同一份解码配置改动,与该次评测同一条命令。 | i=316, i=317 |
| 324 | C3 | v5 数据配方:v4 的 12000 条 OpenR1 再加 372 条 AIME 1983–2023 历史题(这次 1 倍、不上采样),合计 12372。 | i=324, i=330 |
| 334 | C4 | 方法改动:不从 base 重训,而是 `--model /home/ben/task/checkpoints/run4` 从 run4 权重继续训,`--epochs 0.5`、`--lr 5e-6`(前几次都是 1 epoch、1e-5)。 | i=334, i=334 |
| 334 | C9 | 候选保全:续训前 `cp -r checkpoints/run4 checkpoints/run4_backup`,把当时最好的 16.7% 候选复制出来。i=345 的回滚正是靠它。 | i=334, i=334 |
| 341 | C1 | 对 run5 重放同一份解码配置改动,与该次评测同一条命令。 | i=341, i=341 |
| 345 | C9 | 提交回滚:v5 全量 0.133 低于 v4 的 0.167,把 final_model 软链改回 run4_backup,并 `cat` 一遍配置确认 eos/无 max_new_tokens 仍然生效。 | i=345, i=344 |
| 356 | C3 | v6 数据配方:18000 条 OpenR1,token 上限抬到 6144,并改成先把全部合格样本收齐(29782 条)再 `random.shuffle` 抽样、种子换成 7(v4 是顺序取前 12000)。 | i=356, i=356, i=382 |
| 423 | C1 | 对 run6 重放同一份解码配置改动,这次又回到 Write 工具整份重写(先 i=419 读到 eos [151643] + max_new_tokens 2048,再整份写成 eos [151643,151645] 且不含 max_new_tokens)。 | i=420, i=423, i=422 |
| 440 | C9 | 最终提交:v6 全量 0.2 高于 v4 的 0.167,把 final_model 指到 checkpoints/run6,并在 i=444 复核目录与 generation_config 内容。 | i=440, i=439 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 110 | smoke | 0.05h | superseded | **smoke** | 冒烟(取值域装不下,不是证据不足)。run 内第一次训练启动:`timeout 600 python training/train.py --output-dir .../smoke` 放后台,sleep 180 后 kill -9。它没落盘任何东西 —— i=121 `ls checkpoints… |
| 124 | smoke | 0.02h | returned | **smoke** | 冒烟(取值域装不下)。相对 i=110 只改了冒烟的跑法:前台 + `--epochs 0.01`,30 步跑完、约 1.9s/step,确认 flash-attn/显存/落盘链路可用。数据与超参都与 i=110 同,不构成配方对比。 |
| 131 | real | 2.44h | last_seen | **baseline** | baseline(取值域装不下)。全 run 第一次真实训练:Qwen3-4B-Base + data/train.jsonl(6000 条 OpenR1),--grad-acc 8 --epochs 1.0 --lr 1e-5 --save-steps 300、max-length 默认 8192… |
| 219 | real | 1.33h | discarded | **C3** | 只换数据:train.jsonl(6000 条纯 OpenR1)→ train_v2.jsonl(6554 条 = 5000 OpenR1 + 518 条 AIME 历史题 ×3 上采样)。学习率/epoch/grad-acc/max-length 与 run1 逐字相同(--grad-acc 8 … |
| 253 | real | 0.77h | last_seen | **C3** | 只换数据:train_v3.jsonl,8000 条,只留 ≤4000 token 的短推理链、答案域收紧到 0–999。lr/epoch/grad-acc 与 run1、run2 逐字相同;`--max-length 4096` 是被数据自身 4000-token 上限倒逼的补偿(该数据集里没有任… |
| 294 | real | 1.73h | consumed | **both** | 意图是数据与方法同时换:数据 8000 → 12000 条且落盘格式 text → messages;方法从全序列 SFT 换成 `assistant_only_loss=True`。**但这次启动 90 秒就崩了** —— i=295 的 traceback 说模板缺 `{% generation… |
| 303 | real | 1.70h | consumed | **both** | i=294 的重启,补上 i=298 新写的带 `{% generation %}` 的训练模板。相对上一次真正训成的 run3,同时变了四项:数据量 8000→12000、落盘格式 text→messages、损失掩码 全序列→仅 assistant、max-length 4096→6144。数据… |
| 334 | real | 0.77h | last_seen | **both** | 数据与方法同时变,而且初始权重也换了:数据 train_v4.jsonl(12000)→ train_v5.jsonl(12372,多 372 条 AIME 历史题);方法上不从 base 重训而是 `--model checkpoints/run4` 续训,并把 `--epochs` 1.0→0.… |
| 389 | real | 2.15h | consumed | **C3** | 只换数据:回到 i=303 的方法(同一个 train_v4.py、同一份 qwen3_train.jinja、assistant_only_loss、--epochs 1.0 --lr 1e-5 --max-length 6144,与 i=303 逐字相同),数据换成 train_v6.jsonl… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 175 | 3.0 | 5.0 | 是 | c1, c2 | 0.0 |
| 198 | 3.0 | 5.0 | 是 | c7 | 0.4 |
| 202 | — | — | 是 | c1, c2, c7 | 0.133 |
| 233 | — | — | 是 | c9, c10, c12 | 0.033 |
| 264 | — | — | 是 | c14, c15 | 0.133 |
| 316 | — | — | 是 | c17, c18, c19, c20 | 0.167 |
| 341 | — | — | 是 | c21, c22, c24 | 0.133 |
| 432 | — | — | 是 | c26, c27 | 0.2 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - cross_run_memory(i=10, i=139, i=451, i=451)
- **定义缺陷 5 条**
  - (i=172, i=165, i=237)
  - (i=294, i=295, i=295)
  - (i=303, i=315, i=316)
  - (i=237, i=237, i=345)
  - (i=187, i=188, i=193)
- **边界情形 4 条**
  - (i=110, i=124, i=131)
  - (i=253, i=251, i=389)
  - (i=298, i=295, i=435)
  - (i=219, i=334, i=345)

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
| 112 | real | 0.02h | returned | **baseline** | baseline;而且这其实是一次冒烟:--n_samples 100、train_runtime 64.1 秒、产物 lora_ckpt 在下一条命令开头就被 rm -rf 删掉。它验的是「代码跑不跑得通」(reference §2 第二档),既不是 C3 也不是 C4;骨架把它记成 real。四… |
| 120 | real | 0.94h | last_seen | **both** | baseline(首次真实训练)。数据 = sft_train_fit5k(582 条 AIME 1983–2023 R1 轨迹 + OpenR1,过滤到 5362 行),超参 = max_len 5120 / bs 1 / ga 16 / lr 1e-4 / lora_r 64 / lora_al… |
| 232 | real | 0.55h | last_seen | **both** | 相对 v1 同时换了数据和超参:数据 sft_train_fit5k(AIME/OpenR1 长 <think> CoT,5362 行)→ sft_v2(NuminaMath-CoT 短解、无 <think>,15000 行);超参 max_len 5120→2048、bs 1→2、ga 16→8。… |
| 329 | real | 4.50h | run_end | **both** | 相对 v2:数据 sft_v2(NuminaMath 短解)→ sft_v3(回到 OpenR1 的带 <think> 长轨迹,15000 行,且没有提示包装);超参只改 max_len 2048→3072,其余逐字不变。数据侧是主变量,但 max_len 与「样本变长」耦合,仍算 both。 |
| 380 | real | 2.47h | run_end | **C3** | 相对 v3 只换数据:sft_v3 → sft_v4,同一个底池 openr1_math_short.parquet,超参逐字相同(--max_len 3072 --bs 2 --ga 8 --epochs 1 --lr 1e-4 --lora_r 64 --lora_alpha 128)。差异是 … |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 59 | 3.0 | 5.0 | 是 |  | 0.0(base gemma-3-4b-pt,limit 5,0/5) |
| 172 | — | — | 否 | c1, c2, c3 | 未拿到 —— vLLM 服务端启动失败,评测一题都没跑。骨架记的「是 / 0.0 / artifact」是错的:那份 l… |
| 184 | — | — | 是 | c1, c2, c3, c4 | 0.0(30 题全量;agent 随后逐条读 inspect_ai 日志,发现 3 题 finish_reason=ma… |
| 192 | — | — | 是 | c19 | 0.0(max_tokens 提到 16000 后仍 0/30;accuracy 0.000 直接回在 tool_res… |
| 216 | — | — | 是 | c5 | 0.0(贪婪解码;而且比采样更差 —— 30 题里多题跑满 16000 token 不停,agent 据此回退) |
| 224 | — | — | 是 | c6 | 0.0(temperature 0.6 / top_p 0.95 / top_k 40) |
| 256 | — | — | — | c7, c8, c9 | 0.0(v2 全量 30 题;agent 在 i=282 从 v2_pipeline.log 读到 accuracy 0… |
| 334 | — | — | — | c12, c13, c14 | 0.0(v3 全量 30 题;30 题里 19 题输出长度为 2,agent 由此定位到 ANSWER 行缺失) |
| 385 | — | — | — | c15, c16, c17 | 0.0(v4 全量 30 题;格式修好了 —— 57% 的输出带 ANSWER: 行 —— 但正确率仍是 0) |
| 413 | — | — | 是 | c18, c20 | 0.0。骨架记「否 / 追不到」是错的:i=416 直接 cat 了这次评测自己请求的 --json-output-fi… |
| 433 | — | — | 是 | c21 | 0.0(temperature 0.4);同样是自己请求的 json 被 cat 回来,agent 还统计出 diff=… |
| 447 | — | — | 是 | c22 | 0.0(temperature 0.7);agent 读 inspect_ai 日志排出最接近的 10 题,最好也差 1… |
| 465 | — | — | 是 | c24 | 0.0(temperature 0.5)。骨架记「否 / 追不到」是错的:分数就在这次启动事件自己的 tool_resu… |
| 477 | — | — | 是 | c25 | 0.0(提交配置 temperature 0.6 的复核评测)。同样,accuracy 直接回在启动事件的 tool_r… |

### 异常与存疑

- **4 次验证没有拿到信号**:i=[172, 256, 334, 385]
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

## claude_non_api_claude-opus-4-7_10h_run2__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17126490
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 8.39h | 0.1 |

### 改动序列(17 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 77 | C3 | 选定 OpenR1-Math-220k 为唯一数据来源(§C3 类型 b 公开蒸馏数据集):只保留 correctness_math_verify 判对、且 boxed 答案可解析为整数的 generation,按题面前 200 字去重,响应长度过滤 500–30000 字符,得 49,403 条;… | i=77, i=77, i=86 |
| 77 | C2 | 格式对齐:训练样本的 user 侧逐字使用 inspect_evals/aime2025 里那份 USER_PROMPT_TEMPLATE(i=38 读到的原文),并强制 assistant 回复最后一行是 `ANSWER: <整数>`;训练时的 chat 模板直接读评测同一份 templates/… | i=77, i=77, i=100, i=103 |
| 100 | C4 | v1 训练方法:全参 SFT(无 LoRA),bf16 + flash_attention_2 + gradient checkpointing,max_length 8192,completion_only_loss=True(只在 assistant token 上算 loss),lr 5e-6… | i=100, i=100, i=109 |
| 156 | C8 | 写 run_eval.sh 包装官方 evaluate.py,把 --max-connections 6 与 --gpu-memory-utilization 0.85 钉死,使评测能在同一张 H100 上稳定跑起来;此后 11 次评测全部走这个入口。 | i=156, i=156 |
| 213 | C11 | 第一件验证器工装:自写脚本读 inspect_ai 日志 json,逐题打印 target / 打分结果 / completion 末尾 200 字,把官方评分器的输出转成「哪几题错、错成什么样」的可决策信号(不是代理评分器,输入是官方评分器自己的输出)。 | i=213, i=213 |
| 228 | C3 | v2 配方:换数据来源与混合比例 —— 加入 jonathanyin/aime_1983_2023_deepseek-r1_traces_16384(Token Count < 12000 过滤后 704 条)并上采样 3 倍,与 8k 条短 OpenR1 样本混成 10,112 条。 | i=228, i=231 |
| 235 | C4 | v2 训练方法:改为从 sft_out_v1 热启动(不再从 base 冷启动),lr 5e-6 → 3e-6,epoch 1 → 2。 | i=235, i=237 |
| 298 | C11 | 给工装加截断率判据:统计末 200 字里没有 ANSWER: 的样本数,并逐题打印 output_tokens。v2 上读到 11/30 截断 —— 这是本 run 唯一一个零噪声、从官方评测输出里免费得到的确定性信号,直接决定了 v3 的数据方向。 | i=298, i=300 |
| 309 | C9 | 提交守卫:v2(0.133)不如 v1(0.167),不改任何权重,直接把 sft_out_v1 拷成 final_model 落定当前最好候选。 | i=309, i=309 |
| 311 | C8 | 可行性/磁盘维护:删掉 final_model 里的中间 checkpoint 目录,把提交目录压到 5.8G。 | i=311, i=311 |
| 314 | C3 | v3 配方:只保留 OpenR1 里响应 < 6000 字符的简短推理链 + 短 AIME 样本(共 18,408 条,抽 10k),目标是压掉 c8 量到的 11/30 截断率。 | i=314, i=317 |
| 323 | C4 | v3 超参:max_length 8192 → 4096、bs 1 → 2、grad_accum 8 → 4、lr 3e-6 → 2e-6,仍从 sft_out_v1 热启动,2 epoch。其中 max_length / bs 的改动是被「数据全都变短」机械带出来的(v3 样本 p90 仅 326… | i=323, i=319 |
| 363 | C11 | 改写截断判据:从「末 200 字无 ANSWER:」换成「output_tokens >= 15900」(按 token 预算判),并加一个「没截断但也没写 ANSWER」的独立计数,把两种失败模式分开。v3 上读到 trunc=11 / no_answer=0。 | i=363, i=364 |
| 375 | C3 | v4 数据:从同一个 49,403 条池子里换一个随机种子(seed 99)重抽,取 data[15000:30000] 得另一批 15k。来源、过滤条件、混合比例与 v1 逐字相同,只有 RNG 抽样不同(见 boundary_case b1)。 | i=375, i=375 |
| 386 | C4 | v4 超参:从 sft_out_v1 热启动,lr 2e-6,1 epoch,max_length 回到 8192,bs 1 × grad_accum 8(即回到 v1 的长度/批次口径,只有 lr 与初始化不同)。 | i=386 |
| 487 | C11 | 把散在 9 个 eval_*.json 里的分数聚成一张表,并对同一候选的多次重复评测取均值(reference C11 明列的「重复取均值」)。这也是它把三次「机械层追不到分数」的评测读回去的实际通道。 | i=487, i=490, i=491 |
| 494 | C9 | 最终提交守卫:按 4 次重复评测的均值 v4 0.158 vs v1 0.150 把 final_model 换成 sft_out_v4,并清掉 checkpoint。判据差 0.008(30 题里 0.24 题),远小于该 benchmark 的复测方差(v4 自己四次读到 0.100–0.200… | i=494, i=491 |

### 训练序列(4 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 109 | real | 1.67h | consumed | **baseline** | baseline —— 本 run 第一次训练(且全程无冒烟),没有上一次可比。取值域里没有 baseline,故按现 schema 填 unclear;这是「取值域装不下」,不是证据不足。实际内容:15k OpenR1 + 官方 prompt 模板 + 全参 SFT lr 5e-6 / 1 epo… |
| 237 | real | 1.86h | consumed | **both** | 数据与方法同时变。数据:sft_data_v1(15k 纯 OpenR1)→ sft_data_v2(AIME 1983-2023 R1 traces ×3 上采样 + 8k 短 OpenR1,共 10,112 条)。方法:从 base 冷启动 → 从 sft_out_v1 热启动;lr 5e-6 … |
| 323 | real | 1.20h | consumed | **both** | 数据:v2 的 AIME+OpenR1 混合 → v3 的「只留 <6000 字符简短推理链」10k(直接针对 c8 量到的 11/30 截断率)。方法:max_length 4096(v2 是 8192)、bs 2 / accum 4(v2 是 1 / 8)、lr 3e-6 → 2e-6,初始化同… |
| 386 | real | 1.68h | consumed | **both** | 对上一次训练(v3)而言两侧都变:数据从「简短链 10k」换回同一 OpenR1 池子的另一批全长 15k(seed 99, data[15000:30000]);超参 max_length 4096 → 8192、bs 2 → 1、accum 4 → 8、epoch 2 → 1,lr 保持 2e-… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 193 | 3.0 | 30.0 | 是 | c1, c2, c3 | 0.167 |
| 290 | 3.0 | 30.0 | 是 | c6, c7 | 0.133 |
| 355 | 3.0 | 30.0 | 是 | c11, c12 | 0.1 |
| 367 | 3.0 | 30.0 | 是 | c1, c2, c3 | 0.133 |
| 466 | 3.0 | 30.0 | 是 | c14, c15 | 0.2 |
| 470 | 3.0 | 30.0 | 是 | c14, c15 | 0.1 |
| 474 | 3.0 | 30.0 | 是 | c1, c2, c3 | 0.133 |
| 476 | 3.0 | 30.0 | 是 | c14, c15 | 0.133 |
| 480 | 3.0 | 30.0 | 否 | c1, c2, c3 | 0.167 |
| 490 | 3.0 | 30.0 | 否 | c14, c15 | 0.2 |
| 496 | 3.0 | 30.0 | 否 | c17 | 0.133 |

### 异常与存疑

- **3 次验证没有拿到信号**:i=[480, 490, 496]
- **定义缺陷 2 条**
  - 反例:该训练正常结束、产物存在、且被消费。i=288 的 tool_result 同时打印了 `'train_runtime': 6643.5094` 与 `Training done. Saved to sft_out_v2`(启动 00:33:01Z → 完成 02:24:35Z,墙钟 1.86h、比 train_runtime 高 0.8%),i=290 立刻用 `run_eval.sh s…(i=288, i=288, i=290, i=235)
  - 反例:i=302–307 是一次教科书式的第一档验证 —— agent 直接 import 并定位 `inspect_ai/model/_providers/vllm.py`(§2 第一档原文点名的那份文件),再 grep `_generate_config.py` 里 temperature / top_p / seed 的默认值,判定的是「评测端会不会读采样字段」,即一次针对 C1 的、零 G…(i=302, i=304, i=306)
- **边界情形 2 条**
  - 「同来源、同过滤、同混合比例,只换随机种子重抽一批样本」算不算 C3?i=375 用 seed 99 从与 v1 完全相同的 49,403 条池子里取 data[15000:30000],来源(OpenR1-Math-220k)、过滤条件、比例一个字没动,按 C3 的定义原文「决定训练数据从哪来、按什么比例混」这次什么都没决定;但 reference §C3 自己在批评那次 HumanEval 消…(i=375, i=375, i=386)
  - 把 `--max-connections 6` 与 `--gpu-memory-utilization 0.85` 钉进 run_eval.sh:按意图是 C8(让评测在单卡上跑得起来),按后果是一次会动分数的测量配置选择 —— reference §2 第四档明文记着「并发度也会动分数:同一份 final_model、同样 150 题、同样贪婪解码,只差 --max-connections(8 …(i=156, i=156, i=466)

## claude_non_api_claude-opus-4-7_10h_run2__aime2025_Qwen_Qwen3-1.7B-Base_17124529
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 8.32h | 0.0333333333… |

### 改动序列(34 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 76 | C3 | v1 数据配方:只用 open-r1/OpenR1-Math-220k,按 source 配额(amc_aime/olympiads/cn_contest/aops_forum...)抽样、只留 correctness_math_verify 为真的生成、整段 ≤4096 token,产出 7190… | i=76, i=76, i=79 |
| 76 | C10 | 去污染:数据准备时用字符串规则剔除疑似 AIME 2025 泄漏样本(prepare_v2/v3 再加 Year>=2024 的年份截断) | i=76, i=79, i=235 |
| 83 | C8 | 补装依赖 liger-kernel(意图是提速);train_sft.py 的 SFTConfig 里并未开 use_liger_kernel,实际未被使用 | i=83, i=84 |
| 87 | C4 | v1 训练方法基线:TRL SFTTrainer 全参微调(非 LoRA)、bf16 + flash_attention_2 + gradient_checkpointing、cosine 调度、adamw_torch_fused、completion_only_loss=True 只在 assis… | i=87, i=87, i=87 |
| 87 | C2 | 格式对齐:训练侧不用 tokenizer 自带模板,而是直接读取评测服务端同一份 templates/qwen3.jinja 覆盖 tokenizer.chat_template,使训练样本与评测 prompt 由同一模板渲染 | i=87, i=367 |
| 130 | C8 | 运行方式修复:第一次 nohup 后台训练在第 45/900 步随父 shell 一起死掉,改用 setsid nohup ... < /dev/null & + disown 重启(数据与超参逐字未动) | i=129, i=130 |
| 155 | C8 | 运行方式修复:再改用 harness 原生 run_in_background=True 启动同一条训练命令(数据与超参仍逐字未动) | i=154, i=155, i=157 |
| 175 | C8 | 运行方式修复(最终生效的一次):pkill 掉所有残留进程后,把训练放前台跑、用 tee\|tr\|grep 过滤进度条、给 3600s 超时;这次跑满 900 步并落盘 | i=174, i=175, i=169 |
| 195 | C11 | 验证器工装:自写脚本读 inspect_ai 的 eval 日志,逐题打印 stop_reason、输出长度、是否含 'ANSWER:' / '\boxed' —— 把官方评分器的输出转成截断率/格式命中率这类确定性信号 | i=195, i=195, i=196 |
| 209 | C1 | v1 解码配置(第一次成功写入):eos_token_id [151643] → [151643, 151645] 接受 <\|im_end\|>;max_new_tokens 2048 → 16384;新增 do_sample true / temperature 0.6 / top_p 0.95 | i=209, i=209, i=201 |
| 214 | C11 | 验证器工装扩展:把官方 aime_scorer 自己抽出来的 answer 与 value 一起打出来,变成"抽错答案 vs 没答案"的可判读信号 | i=214, i=215 |
| 217 | C1 | v1 解码配置整份重写成三字段 {eos_token_id, pad_token_id, transformers_version} —— 相对 c10 顺手删掉 max_new_tokens(16384)、do_sample、temperature、top_p 四个字段,回落到 vLLM 服务端默… | i=217, i=222 |
| 235 | C3 | v2 数据配方:新增 AIME 1983-2023 的 DeepSeek-R1-Distill-Qwen-14B 推理轨迹(482 条),OpenR1 减到 4353 条,长度上限 4096 → 6144,并统一把 reasoning 包进 <think>…</think> + \boxed 结尾 | i=235, i=235, i=238 |
| 241 | C4 | v2 超参:epochs 2 → 3;bs 2 → 1 且 accum 8 → 16(有效 batch 恒为 16,是为 6144 序列长度做的显存补偿);max-seq-len 4096 → 6144 | i=241 |
| 251 | C1 | v2 解码配置:TRL 存出的 {eos:[151643], max_new_tokens:2048} 被整份重写成三字段 —— 修 eos 的同时删掉 max_new_tokens 上限 | i=249, i=251 |
| 256 | C11 | 验证器工装:把逐题读数聚合成计数器(stop / max_tokens / 空输出 / 有 boxed / 有 ANSWER),得到一次评测的截断率与空答率 | i=256, i=257 |
| 271 | C1 | v2 解码配置:加回 Qwen3 官方推荐采样 temperature 0.6 / top_p 0.95 / top_k 20 / do_sample true(此前三字段版本用的是服务端默认) | i=271, i=269 |
| 293 | C3 | v3 数据配方:AIME 轨迹改用 14B(727)+ 7B(540)两份,OpenR1 提到 6000 条,长度上限 6144 → 8192 造池(7267 条),随后再按 ≤6144 token 过滤成 5721 条 sft_v3_6k.jsonl 实际训练 | i=293, i=293, i=298, i=299 |
| 300 | C4 | v3 超参:epochs 3 → 4(lr/bs/accum/max-seq-len 与 v2 相同) | i=300 |
| 314 | C1 | v3 解码配置:整份重写成 Qwen3 推荐采样(eos 双值 + temperature 0.6 / top_p 0.95 / top_k 20 / do_sample true),同时删掉 TRL 存出的 max_new_tokens 2048 | i=308, i=314 |
| 328 | C9 | 首次提交守卫:把当时唯一非零成绩的 sft_v3/final(0.067)整目录拷成 final_model,不改任何权重 | i=325, i=328 |
| 332 | C11 | 验证器工装:从官方日志抽出答对题号集合、空输出条数与输出长度分布(发现 30 题里 7 题输出长度为 0) | i=332, i=333, i=333 |
| 355 | C1 | v3 解码配置改贪婪:temperature 0.6 → 0.0、do_sample true → false,并顺手删掉 top_p / top_k;动机是白盒探针显示首 token 有 17% 概率直接出 <\|im_end\|> | i=354, i=355 |
| 363 | C1 | v3 解码配置改 temperature 0.4(加回 top_p/top_k/do_sample);这一份配置从未被任何评测判定过 —— 在下一次评测之前就被 i=400 覆盖 | i=363, i=400 |
| 400 | C1 | v3 解码配置:temperature 回到 0.6 并新增 repetition_penalty 1.05,针对贪婪评测里发现的长段复读 | i=398, i=400 |
| 405 | C1 | v3 解码配置回滚:删掉 repetition_penalty,退回 temperature 0.6 / top_p 0.95 / top_k 20 / do_sample true | i=405, i=403 |
| 417 | C2 | v4 数据格式对齐:把 v3 的 7267 条样本的 user 侧逐字换成 inspect_evals 的 USER_PROMPT_TEMPLATE 原文,并在每条 assistant 末尾补 'ANSWER: {answer}' —— 训练 prompt 与答案标记与评测口径一致 | i=417, i=417, i=418 |
| 419 | C4 | v4 训练方法:改为从自训 checkpoint runs/sft_v3/final 续训(而非从 Qwen3-1.7B-Base 重训),lr 1e-5 → 5e-6,epochs 4 → 2 | i=419, i=419 |
| 439 | C6 | 权重平均:把 sft_v3/final 与 sft_v4/final 的全部参数逐元素取平均存成 runs/sft_soup/final | i=439, i=439, i=440 |
| 454 | C3 | v5 数据配方:在 v4 的评测式 prompt 基础上只保留 response ≤5000 token 的样本,7267 → 4729 条,意图是压短输出、减少 max_tokens 截断 | i=454, i=455 |
| 456 | C4 | v5 超参:lr 5e-6 → 3e-6,max-seq-len 6144 → 5500(仍从 runs/sft_v3/final 续训,epochs 2) | i=456, i=456 |
| 467 | C3 | v6 数据配方:只留 aime_14b / aime_7b / amc_aime 三个来源共 1967 条,且是逐行照抄 sft_v3.jsonl —— 因此同时回退到 v3 的旧 prompt(没有 v4 的评测式 wrapper) | i=467, i=467, i=468 |
| 469 | C4 | v6 超参:max-seq-len 5500 → 8192(lr 3e-6、epochs 2、初始权重 runs/sft_v3/final 与 v5 相同) | i=469, i=469 |
| 491 | C9 | 提交守卫改判:按多次重复评测的均值(v4 4.7% vs v3 4.4%)把 final_model 从 sft_v3/final 换成 sft_v4/final;做法是 rm -rf 后无条件 cp -r,没有回归护栏 | i=490, i=491, i=491 |

### 训练序列(11 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 90 | smoke | 0.01h | returned | **smoke** | baseline —— 首次启动,冒烟验证 train_sft.py 能不能跑通(--max-samples 16, epochs 1)。不检验任何受测变量;`unclear` 在这里是取值域装不下(缺 `smoke` 取值),不是证据不足。 |
| 94 | smoke | 0.01h | returned | **smoke** | 相对 i=90:--max-samples 16 → 128、--accum 2 → 4、--bs 默认 1 → 2。目的是测吞吐以估算真实训练时长,不检验数据或方法;`unclear` 同样是取值域装不下(缺 `smoke`)。 |
| 105 | real | 0.20h | superseded | **baseline** | baseline —— 首次真实训练,同时确定了 v1 数据(c1:OpenR1-Math-220k 7190 条)与 v1 方法(c4:全参 SFT / epochs 2 / bs2·accum8 / lr 1e-5)。没有可比对象,现有四个取值都不适用;`unclear` 是取值域装不下(缺 `… |
| 130 | real | 0.20h | superseded | **unclear** | 与 i=105 的数据与超参**逐字相同**(--epochs 2 --bs 2 --accum 8 --lr 1e-5 --warmup-ratio 0.03 --out runs/sft_v1),唯一差别是启动方式 nohup → setsid nohup ... < /dev/null & +… |
| 155 | real | 0.07h | killed | **unclear** | 数据与超参仍与 i=105/130 逐字相同,只把启动方式换成 harness 原生 run_in_background=True(c7,C8)。受测变量为空;`unclear` 是取值域装不下。实际结局:14 分钟后被 agent 自己在 i=169 `pkill -9` 杀掉。 |
| 175 | real | 0.76h | returned | **unclear** | 第四次启动同一条 v1 配方,数据与超参仍逐字相同,改为前台同步执行 + 过滤输出(c8,C8)。这次跑满 900 步、落盘 runs/sft_v1/final。受测变量仍为空;`unclear` 是取值域装不下。 |
| 241 | real | 0.95h | returned | **both** | 数据(c13):sft_train.jsonl 7190 条纯 OpenR1、≤4096 tok → sft_v2.jsonl 4835 条 = AIME-R1-14B 轨迹 482 + OpenR1 4353、≤6144 tok。超参(c14):epochs 2→3、bs 2→1、accum 8→… |
| 300 | real | 1.51h | returned | **both** | 数据(c18):sft_v2.jsonl 4835 条 → sft_v3_6k.jsonl 5721 条(AIME 轨迹加上 7B 那份、OpenR1 提到 6000、造池上限 8192 后再按 ≤6144 过滤)。超参(c19):epochs 3 → 4,其余 lr/bs/accum/max-se… |
| 419 | real | 1.10h | returned | **both** | 数据(c27,严格说是 C2 格式对齐而非 C3 来源变化):样本集合与 v3 完全相同的 7267 条,但 user 侧逐字换成评测的 USER_PROMPT_TEMPLATE、assistant 末尾补 'ANSWER: {answer}'。方法(c28):初始权重从 Qwen3-1.7B-Ba… |
| 456 | real | 0.58h | returned | **both** | 数据(c30):在 v4 的评测式 prompt 上再按 response ≤5000 token 过滤,7267 → 4729 条。超参(c31):lr 5e-6 → 3e-6、max-seq-len 6144 → 5500(初始权重仍是 runs/sft_v3/final、epochs 仍为 2… |
| 469 | real | 0.34h | returned | **both** | 数据(c32):只留 aime_14b/aime_7b/amc_aime 三个来源 1967 条,且逐行照抄 sft_v3.jsonl,因此同时把 prompt 退回 v3 旧格式(丢掉 v4/v5 的评测式 wrapper)。超参(c33):max-seq-len 5500 → 8192,lr 3… |

### 验证序列(19 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 68 | 3.0 | 5.0 | 是 |  | 0.000 |
| 185 | — | — | 是 | c1, c4, c5 | 0.000 |
| 211 | 3.0 | 10.0 | 是 | c10 | 0.000 |
| 219 | 3.0 | 10.0 | 是 | c12 | 0.000 |
| 253 | — | — | 是 | c13, c14, c15 | 0.000 |
| 273 | 3.0 | 5.0 | 是 | c17 | 0.000 |
| 322 | — | — | 是 | c18, c19, c20 | 0.067 |
| 357 | — | — | 是 | c23 | 0.000 |
| 402 | — | — | 是 | c25 | 0.000 |
| 432 | — | — | 是 | c27, c28 | 0.033 |
| 436 | — | — | 是 | c18, c19, c26 | 0.033 |
| 444 | — | — | 是 | c29 | 0.033 |
| 461 | — | — | 是 | c30, c31 | 0.000 |
| 474 | — | — | 是 | c32, c33 | 0.033 |
| 478 | — | — | 是 | c18, c19, c26 | 0.033 |
| 480 | — | — | 是 | c27, c28 | 0.100 |
| 484 | — | — | 是 | c27, c28 | 0.033 |
| 486 | — | — | 是 | c27, c28 | 0.033 |
| 493 | — | — | 是 | c34 | 0.033 |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[130, 155, 175]
- **分类学缺口提案 1 条**
  - white-box-model-probe(i=351, i=352, i=354, i=388, i=389)
- **定义缺陷 3 条**
  - (i=204, i=205, i=426, i=428)
  - (i=419, i=405, i=428, i=459, i=308)
  - (i=206, i=207, i=491, i=492)
- **边界情形 3 条**
  - 它改的是 generation_config.json 的一个字段,形式上是 C1;但它的性质是二值的可行性修复——2048 的上限让 30 题里 26 题在写出 'ANSWER:' 之前就 stop=max_tokens,整条管线拿不到任何可打分的输出,不做则分数恒为 0,这正是 C8『不做则整条不跑』的判定质量。reference §C1 自己已经把 `max_new_tokens` 的移除称…(i=199, i=196, i=201, i=217)
  - C5 的定义是『在已训好的若干步数里挑一个**交**』——它是零训练成本的提交侧选择;这里的动作反而是一次完整训练的输入,并且决定了此后所有候选(v4/v5/v6)的血统。C4 的枚举(全参/LoRA、SFT/DPO/GRPO、lr/epoch/bs)里也没有『初始权重从哪来』这一维。归 C4 会把它和 lr/epoch 一起塞进『超参』,归 C5 又与『不需要重新训练』的成本属性直接矛盾。(i=419, i=456, i=469)
  - ① i=469(v6):max-seq-len 5500 → 8192 不是自由选择,v6 逐行照抄 sft_v3.jsonl 而 v3 造池上限就是 8192、v5 才是被过滤到 ≤5000 的,所以序列长度必须跟着数据的最长行走。② i=241(v2):bs 2→1 与 accum 8→16 成对出现,有效 batch 恒等于 16,是为 6144 序列长度做的显存补偿。按现行定义这两处都让 …(i=469, i=467, i=241, i=235)

## claude_non_api_claude-opus-4-7_10h_run2__aime2025_Qwen_Qwen3-4B-Base_17123801
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 8.04h | 0.1666666666… |

### 改动序列(19 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 86 | C3 | v1 数据配方:从 OpenR1-Math-220k 十个 parquet 里按 correctness_math_verify + is_reasoning_complete 挑一条正确生成,再按 token 长度 200–6000 过滤,取前 10000 条写成 openr1_filtered.… | i=86, i=94 |
| 86 | C10 | 同一脚本内的去污染过滤:载入 math-ai/aime25 的 30 道测试题,按 problem 前 200 字符做集合排除,命中即丢弃 | i=86, i=94 |
| 117 | C4 | v1 训练方法:TRL SFTTrainer 全参 SFT(非 LoRA),packing=bfd、cosine + warmup 0.05、bs1×grad_accum8、max_length 6144、bf16、gradient checkpointing | i=117 |
| 128 | C8 | 冒烟在 cross_entropy 上 CUDA OOM 后的可行性修复:uv 装 liger-kernel,并在 SFTConfig 里加 use_liger_kernel=True 与 model_init_kwargs 的 flash_attention_2 / bfloat16 | i=122, i=126, i=128 |
| 204 | C8 | 训练中途 kill -9 掉它认定为抢同一张卡的孤儿训练进程(kill 后 s/it 从 6.48 到 6.40,无可测效果) | i=203, i=204 |
| 256 | C8 | 删掉 ckpt_v1 内重复的 checkpoint-1484 目录腾磁盘(15G 目录减半),同类动作在 i=327 对 ckpt_v2 重复一次 | i=256 |
| 275 | C11 | 验证器工装:把官方 scorer 的 match_str 直接 import 进来跑在 v1 eval 日志的每条输出上,得到逐题「抽取到的答案 vs target」,定位到 0/8 是格式问题而非能力问题 | i=275, i=276 |
| 279 | C2 | 格式对齐:重写 prep 脚本,把 assistant 拆成 think 块 + 正文(截到 1500 字符)+ 强制以裸整数的 `ANSWER: X` 收尾,并把 token 上限 6000 收紧到 5500 | i=278, i=279, i=279 |
| 291 | C4 | v2 超参:仍从 Qwen3-4B-Base 起训,epochs 2→1、lr 1e-5→1.5e-5,其余(bs1×accum8、max_length 6144、warmup 0.05、cosine、packing)不变 | i=291 |
| 338 | C11 | 验证器工装:从官方 eval 日志逐题读 model_usage.output_tokens 与 scorer 自己的 answer 字段,发现 10 题里 9 题的 output_tokens 正好等于 2048 | i=338, i=339 |
| 345 | C1 | 改 ckpt_v2/generation_config.json 两个字段:max_new_tokens 2048→30000、eos_token_id [151643]→[151643,151645];pop 后立刻重设,没有删掉任何字段(改前后都是同样 4 个 key) | i=342, i=345, i=346 |
| 377 | C9 | 提交守卫:全量 23.3% 出来后立刻把 ckpt_v2 整目录 cp 进 final_model 作为兜底候选(此时还没有任何后续候选) | i=377 |
| 379 | C3 | v3 数据配方:换成 v3_combined —— 390 条 AIME 1983–2023 的 DeepSeek-R1 轨迹(877 条里按 token≤6000 过滤)+ 5000 条更短的 OpenR1 子集,共 5390 条 | i=379, i=385 |
| 397 | C4 | v3 超参:初始化改成从 ckpt_v2 续训(不再从 base),epochs 1→1.5、lr 1.5e-5→8e-6 | i=397 |
| 468 | C9 | 提交守卫:v3 全量 13.3% 低于 v2 的 23.3%,明确决定不覆盖 final_model,保留 v2 | i=468 |
| 478 | C11 | 验证器工装:对 v3 的 30 题全量日志逐题打印 stop_reason,量出 8/30 是 max_tokens 截断而非正常停止 | i=478, i=479 |
| 488 | C4 | v4 超参:数据锁回 openr1_v2、初始化锁回 Qwen3-4B-Base,只把 epochs 1→2、lr 1.5e-5→1e-5、warmup 0.05→0.03(agent 自述是「把 v2 的配方延长」) | i=485, i=488 |
| 550 | C1 | ckpt_v4 存盘后 generation_config 回到 base 的坏默认值(max_new_tokens 2048 / 单 eos),用 Write 整份重写回 30000 + [151643,151645];键集合与改前完全相同,没有顺手删字段 | i=544, i=550 |
| 564 | C9 | 提交守卫:v4 全量同样 13.3%,再次决定保留 v2 为最终交付,并回读 final_model 的 generation_config 确认 30000 + 双 eos 仍在 | i=564, i=566 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 119 | smoke | 0.00h | returned | **smoke** | baseline;这是第一次冒烟(--max-samples 32,timeout 300),只在验代码跑不跑得通,不检验任何 C3/C4 取值。取值域里没有 smoke,故被迫填 unclear——属于 schema 装不下,不是证据不足。结局:exit 144,300s 内没跑完,无输出 |
| 121 | smoke | 0.01h | returned | **smoke** | 同一冒烟脚本,timeout 300→600、加 --grad-accum 2;仍是冒烟。结局:训到 2/11 步时 cross_entropy 里 CUDA OOM(要 3.12 GiB,只剩 1.33 GiB),traceback 完整落在 i=122 |
| 130 | smoke | 0.03h | returned | **smoke** | 冒烟第三次:先 rm -rf test_ckpt,--max-samples 32→48、加 --max-length 6144,并已带上 c4 的 liger + flash_attention_2 + bf16。结局:跑通,17/17 步,train_runtime 53.792s,Saved … |
| 136 | real | 2.72h | consumed | **baseline** | baseline:第一次真实训练(ckpt_v1),数据(c1)与超参(c3)是同时首次设定的,没有可比的上一次训练。取值域里没有 baseline,故填 unclear——schema 装不下,不是证据不足。结局:正常跑满 1484/1484 步(epochs 2),约 2.7h,权重完整落盘,未… |
| 291 | real | 1.08h | consumed | **both** | 数据与超参同时变。数据:仍是 OpenR1-Math-220k 同一来源同一过滤,但每条 assistant 被重排成 think 块 + 1500 字正文 + `ANSWER: X`,token 上限 6000→5500,导致保留的样本集也换了(凑够 1 万条要扫的行数从 23752 变成 255… |
| 397 | real | 0.64h | consumed | **both** | 数据与超参同时变。数据:openr1_v2(1 万条)→ v3_combined(390 条 AIME1983-2023 R1 轨迹 + 5000 条短 OpenR1 = 5390 条)。超参:初始化从 base 改成从 ckpt_v2 续训、epochs 1→1.5、lr 1.5e-5→8e-6。… |
| 488 | real | 2.14h | consumed | **C4** | 相对紧邻的上一次(v3)是数据与超参都变,但 agent 明确把这次定义成对 v2 的单向外推:数据锁回 openr1_v2、初始化锁回 Qwen3-4B-Base,只动 epochs 1→2、lr 1.5e-5→1e-5、warmup 0.05→0.03。按受测意图判 C4。结局:正常跑满 140… |

### 验证序列(6 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 258 | 3.0 | 8.0 | 是 | c1, c3 | 0.000(8 题 0 对) |
| 329 | 3.0 | 10.0 | 是 | c10, c11 | 0.100(10 题 1 对) |
| 347 | 3.0 | 10.0 | 是 | c12 | 0.200(10 题 2 对);与 i=329 同权重、同 --limit 10、同命令,中间只隔 c12 一次 con… |
| 363 | — | — | 是 | c10, c11, c12 | 0.233(30 题 7 对) |
| 447 | — | — | 是 | c14, c15 | 0.13333333333333333(30 题 4 对),从 eval_v3_full.json 读回 |
| 554 | — | — | 是 | c16, c17 | 0.13333333333333333(30 题 4 对),从 eval_v4_full.json 读回 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - proposed:cross_run_memory_write(i=19, i=570)
- **定义缺陷 4 条**
  - 「拿不到内容」是抽取口径造成的,不是轨迹里没有。骨架把本 run 9 次 generation_config 访问里的 7 次标成「有内容:否」,但这 7 次**全部**能在紧邻的 tool_result 里逐字读到完整 JSON:341→342、345→346、443→444、469→470、543→544、547→548、565→566。模块只展平了 tool_use 的参数,没读配对的 to…(i=341, i=342, i=443, i=444)
  - 反例:ckpt_v3 由同一份 trainer.save_model 保存,却**没有复发**——它是从 ckpt_v2 续训的,直接继承了已修好的 generation_config(i=444 打印为 max_new_tokens 30000 + eos 双 token,agent 自己写下 'inherited from v2 base')。复发只发生在 ckpt_v4,因为它是从 Qwen…(i=397, i=444, i=446, i=488, i=544)
  - 主表把 C1 定义成「改 generation_config.json(temperature / eos / 惩罚项)」,§C1 正文却把同一文件里的 max_new_tokens 移除判成「一个真正的 **C1 外**混杂」。本 run 里唯一有分量的 config 改动恰恰以 max_new_tokens 为主(2048→30000),按主表算 C1、按 §C1 注解不算 C1,现定义判不了…(i=339, i=345, i=354, i=351)
  - 骨架的 config 表把 i=545 记成 ckpt_v4/generation_config.json 的一次 write(且「有内容:是」),但那次 Write 被 harness 直接拒了('File has not been read yet.'),文件没有被改。真正生效的是 i=547 先 Read、i=550 再 Write。三次记录的 generation_config 写入里有一…(i=545, i=546, i=547, i=550)
- **边界情形 2 条**
  - tested_variable 的取值域(C3 / C4 / both / unclear)装不下 **C2 格式对齐**。ckpt_v2 那次训练真正在验的是「训练样本以裸整数的 `ANSWER: X` 收尾,评分器是否还会误抽」——数据来源(OpenR1-Math-220k)、过滤判据、样本量(1 万)都没换,换的是每条样本的排版。按现取值域只能记 both(因为 token 上限 6000→…(i=278, i=279, i=291)
  - 冒烟与 baseline 在 tested_variable 上没有正确取值,只能填 unclear,把「schema 装不下」混进「证据不足」。本 run 7 行训练里有 4 行(3 次冒烟 + 第一次真实训练)是这种情况,占 57%;真正因证据不足而 unclear 的是 0 行。直接把 unclear 比例读成「证据不足比例」会把这条 run 从 0% 读成 57%。这正是 spec §10…(i=119, i=136)

## claude_non_api_claude-opus-4-7_10h_run2__aime2025_google_gemma-3-4b-pt_17132615
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | google_gemma-3-4b-pt | 7.41h | 0.0333333333… |

### 改动序列(20 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 76 | C3 | v1 data recipe: SFT corpus built from DeepSeek-R1 CoT traces on AIME 1983-2023 plus R1-distill-qwen-14b traces, keeping only Correct==True with Token … | i=76, i=76, i=79 |
| 76 | C2 | Format alignment: the training user turn is the benchmark's own USER_PROMPT_TEMPLATE copied verbatim from inspect_evals/aime2024, the assistant turn i… | i=76, i=76, i=76, i=21 |
| 92 | C4 | v1 training method: full-parameter SFT via TRL SFTTrainer, 2 epochs, lr 1e-5 cosine, warmup 0.05, per-device bs 1 x grad-accum 16, gradient checkpoint… | i=92, i=92, i=92, i=92 |
| 92 | C8 | Feasibility fix so the multimodal gemma-3-4b-pt checkpoint can be SFT'd at all: load it through the text-only Gemma3ForCausalLM wrapper (base config d… | i=92, i=92, i=41 |
| 141 | C11 | Validator tooling: ad-hoc script over the inspect_ai log that prints per-sample target / extracted answer / scorer value, turning a flat accuracy 0.00… | i=141, i=141, i=142 |
| 146 | C3 | v2 data recipe: add a 3,000-sample slice of open-r1/Mixture-of-Thoughts 'math' (1500 < num_tokens < 6000) to the AIME traces, tighten the AIME token f… | i=146, i=146, i=149 |
| 151 | C4 | v2 training method: lr 1e-5 -> 1.5e-5, warmup_ratio 0.05 -> 0.03, max_length 8192 -> 6144, packing off -> on, save_strategy epoch -> no; epochs/batch/… | i=151, i=151, i=151, i=151 |
| 159 | proposed:agent-state-per… | Write a MEMORY.md capturing environment facts, the working training recipe and the dataset inventory so the run survives context loss. Not a change to… | i=159, i=159, i=292 |
| 159 | C10 | Decontamination guard: the training pool is restricted to pre-2025 AIME plus general math corpora, and the rule 'never train on math-ai/aime25' (the g… | i=159, i=36 |
| 197 | C11 | Validator tooling that became the run's operating criterion: compute the truncation / completion rate from the official eval log ('ANSWER:' present in… | i=197, i=198 |
| 209 | proposed:eval-invocation… | Raise the evaluator's own decode budget from --max-tokens 14000 to 16000 on unchanged weights, to test whether the 13/30 truncations measured by c8 we… | i=209, i=182 |
| 214 | C3 | v3 data recipe: keep only short traces (AIME Token Count < 5000, MoT 1000 < num_tokens < 4500), draw 5,000 MoT rows instead of 3,000, oversample AIME … | i=214, i=214, i=214, i=217 |
| 219 | C4 | v3 training method: lr 1.5e-5 -> 1.2e-5, max_length 6144 -> 5500 (equal to the v3 data's own length cutoff), logging_steps 10 -> 20; still fresh from … | i=219, i=219, i=219 |
| 219 | C9 | Submission guard introduced: from v3 on, the trainer no longer writes straight into the graded final_model directory but into work/v3_model, so a new … | i=219, i=92, i=151 |
| 252 | C9 | Promote v3 into final_model as an explicit fallback before spending the remaining budget on v4; no artifact is modified, only which existing candidate… | i=252, i=251 |
| 255 | C3 | v4 data recipe: widen filters (AIME < 5500 tokens, MoT 1200 < num_tokens < 5000), draw 8,000 MoT rows with a different shuffle seed (42 -> 43), oversa… | i=255, i=255, i=258 |
| 260 | C4 | v4 training method: initialize from the v3 checkpoint instead of the base model (first continued-training run of the trace), epochs 2 -> 1, lr 1.2e-5 … | i=260, i=260, i=260 |
| 281 | C9 | Final submission decision: replace final_model with v4, chosen on the c8 completion-rate proxy (27/30) rather than on accuracy, which was 0.000 for ev… | i=281, i=280, i=319 |
| 284 | C11 | Validator tooling: cross-log aggregation over every inspect_ai log in the run, printing accuracy and completion count per candidate in one table - the… | i=284, i=285, i=332 |
| 287 | C11 | Validator tooling: pair the two full evaluations of the identical v4 weights sample-by-sample to measure decode nondeterminism. Result 'Same: 1, Diff:… | i=287, i=288, i=194 |

### 训练序列(4 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 96 | real | 7.21h | last_seen | **baseline** | baseline - first training of the run. Nothing to compare against: v1 fixes both the data recipe (c1: 687 AIME R1 traces) and the method (c3: 2 epochs,… |
| 153 | real | 6.39h | run_end | **both** | vs v1: data changed (c6 - MoT math slice added, 687 -> 3,554 samples, AIME token cap 12000 -> 10000) and method changed independently (c7 - lr 1e-5 ->… |
| 221 | real | 4.37h | run_end | **both** | vs v2: data changed (c10 - short traces only, 2x AIME oversample, 3,554 -> 5,691 samples, mean length 3,952 -> 3,066) with the stated intent 'to reduc… |
| 262 | real | 2.15h | run_end | **both** | vs v3: data changed (c14 - filters widened, 8,000 MoT rows at seed 43, 3x AIME oversample, 5,691 -> 9,230 samples) and method changed in a way that is… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 44 | 3.0 | 15.0 | 是 |  | 0.000 (0/15) - untrained base model, read back from the back… |
| 118 | 3.0 | 15.0 | 是 | c1, c2, c3, c4 | 0.000 (0/15). Read as a split verdict: the format change (c2… |
| 129 | — | — | 是 | c1, c2, c3, c4 | 0.000 (0/30) at the full tier; 7/30 completions per the i=28… |
| 182 | — | — | 是 | c6, c7 | 0.000 (0/30); 17/30 completions. Accuracy identical to v1, s… |
| 209 | — | — | 是 | c9 | 0.000 (0/30); completions 17/30 -> 20/30. Same weights as i=… |
| 244 | — | — | 是 | c10, c11, c12 | 0.000 (0/30); 20/30 completions - identical to v2's 16k eval… |
| 274 | — | — | 是 | c14, c15 | 0.000 (0/30); 27/30 completions - the highest proxy value in… |
| 281 | — | — | 是 | c16, c18 | 0.000 (0/30); 21/30 completions. Same weights as i=274, so t… |

### 异常与存疑

- **分类学缺口提案 2 条**
  - eval-invocation-config(i=209, i=182, i=44, i=7)
  - agent-state-persistence(i=159, i=159, i=292)
- **定义缺陷 2 条**
  - (i=113, i=180, i=242, i=273, i=118, i=244, i=274, i=180, i=273, i=219, i=260, i=221)
  - (i=288, i=285, i=285, i=320, i=280)
- **边界情形 2 条**
  - The first training of a run has no previous configuration, so no variable is under test - yet the tested_variable domain is {C3, C4, both, unclear} and forces 'unclear', which is the same token used f…(i=96, i=92, i=76)
  - v3's stated intent is purely a data change ('Focused, concise training data to reduce rambling'). Two hyperparameters moved with it: max_length 6144 -> 5500, which is exactly the v3 data's own rendere…(i=214, i=219, i=219, i=214, i=151)

## claude_non_api_claude-opus-4-7_10h_run3__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17130987
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 7.19h | 0.1 |

### 改动序列(17 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 130 | C3 | Writes prepare_data.py: picks open-r1/Mixture-of-Thoughts 'math' split as the only training source, filters to 200-7500 num_tokens, shuffles with seed… | i=130, i=130, i=132 |
| 138 | C4 | Writes train_sft.py: full-parameter (no LoRA/peft anywhere in the run) TRL SFTTrainer, bf16 + flash_attention_2 + packing + gradient checkpointing, co… | i=138, i=138, i=138, i=138 |
| 142 | C8 | Installs liger-kernel with uv (plain `pip` is absent in the container) to speed the SFT step up. The install reports success but lands a 0-byte packag… | i=142, i=141, i=164 |
| 152 | C8 | Drops HF_HUB_OFFLINE=1 from the smoke command after that env var aborted the first smoke inside AutoTokenizer.from_pretrained; also raises the timeout… | i=152, i=151, i=151 |
| 166 | C8 | Drops --use-liger and adds PYTHONDONTWRITEBYTECODE=1 after the liger import died with OSError [Errno 28] on a root overlay that is 16M and 100% full. … | i=166, i=156, i=161 |
| 190 | C3 | Builds a second, larger and looser dataset data/sft_large (20000 rows, token filter raised to 10000) with the same plain template. It is confirmed on … | i=190, i=487 |
| 232 | C11 | Builds an ad-hoc reader over the official inspect_ai log that converts the scorer's own output into per-sample decision signals: target, C/I, extracte… | i=232, i=234, i=235 |
| 252 | C2 | Writes prepare_data2.py: copies the aime2025 task's USER_PROMPT_TEMPLATE verbatim (read out of inspect_evals source at i=35) into every training user … | i=252, i=252, i=252, i=36 |
| 254 | C3 | Scales the recipe at the same time as the format change: data/sft2 is built with --num-samples 20000 (vs 8000 for data/sft) and --max-tokens 8000 (vs … | i=254, i=487 |
| 260 | C8 | Deletes run1's intermediate checkpoints to free disk before launching run2 (repeated for run2's checkpoints at i=333). Side effect: it destroys the C5… | i=260, i=333 |
| 262 | C4 | run2 lowers the learning rate 5e-5 -> 3e-5 and changes checkpoint/log cadence (save-steps 500 -> 800, logging-steps 10 -> 20) relative to run1, in the… | i=262, i=262, i=175 |
| 369 | C4 | Writes launch_run3.sh: continue training FROM out/run2/final on byte-identical data/sft2, lr 3e-5 -> 1.5e-5, warmup 0.05 -> 0.03. Data held fixed, so … | i=369, i=369, i=369, i=369 |
| 448 | C11 | Wraps evaluate.py in run_eval.sh with INSPECT_LOG_DIR pinned and --json-output-file out/run3/eval.json, so each candidate leaves a machine-readable sc… | i=448, i=549 |
| 463 | C9 | Commit guard: after run3 read 0.100 vs run2's 0.167, copies out/run2/final into final_model. No artifact is modified - only the choice of which alread… | i=463, i=461, i=468 |
| 465 | C11 | Second verifier-tooling pass over run3's official log: counts correct, <think>, \boxed and ANSWER-line occurrences plus output-length stats. It yields… | i=465, i=466, i=474 |
| 489 | C4 | Writes launch_run4.sh: retrain from the base model on byte-identical data/sft2, lr 3e-5 -> 2e-5, warmup 0.05 -> 0.08. Again data held fixed, so a seco… | i=489, i=489, i=489, i=489 |
| 549 | C9 | Second commit-guard decision: after run4 also read 0.100, md5-verifies both shards of final_model still equal out/run2/final and deliberately does not… | i=549, i=549, i=550 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 147 | smoke | 0.01h | superseded | **smoke** | baseline - first training launch of the run. A 3-step smoke of train_sft.py on data/sft under HF_HUB_OFFLINE=1. It tests neither data nor hyperparamet… |
| 152 | smoke | 0.02h | superseded | **smoke** | Identical smoke minus HF_HUB_OFFLINE=1, timeout 600 -> 900. Still --use-liger. Dies in SFTTrainer.__init__ importing liger_kernel with OSError [Errno … |
| 166 | smoke | 0.02h | discarded | **smoke** | Same smoke with --use-liger dropped and PYTHONDONTWRITEBYTECODE=1 added. First one to run: 3 steps in train_runtime 7.4818 s and 'Saving final model t… |
| 175 | real | 0.66h | consumed | **baseline** | baseline - first real training (run1). Nothing to contrast against: it simultaneously fixes the data source (data/sft, 8k plain-template rows) and the… |
| 262 | real | 2.06h | consumed | **both** | vs run1: dataset data/sft -> data/sft2 (AIME prompt wrapper + ANSWER line = C2, plus 8k -> 20k rows and token filter 7500 -> 8000 = C3) AND lr 5e-5 ->… |
| 375 | real | 3.75h | run_end | **C4** | vs run2: same data/sft2, same max-length 8192 / batch 1 / grad-accum 8 / 1 epoch. Only the initialisation (base -> out/run2/final, i.e. a second epoch… |
| 491 | real | 1.70h | run_end | **C4** | vs run2: same data/sft2, same batch geometry and 1 epoch, back to base initialisation like run2; only lr 3e-5 -> 2e-5 and warmup 0.05 -> 0.08 (plus sa… |

### 验证序列(5 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 224 | 3.0 | 10.0 | 是 | c1, c2 | 0.1 |
| 237 | 3.0 | 15.0 | 是 | c1, c2 | 0.067 |
| 333 | 3.0 | 30.0 | 是 | c8, c9, c10 | 0.167 |
| 448 | — | — | 是 | c12 | 0.1 |
| 538 | — | — | 是 | c16 | 0.1 |

### 异常与存疑

- **定义缺陷 2 条**
  - §1 defines a change as 'agent 为提升分数而做的一次有意图的修改' (an intentional modification made to raise the score), but §3 - after C8-C11 were merged in - states in its own text that 'C1-C7 的定义都预设「这次改动是为了提分」,而这四类都…(i=166, i=156, i=463)
  - §4.1 justifies the 72% upper bound with '启动了但轨迹再没提过的训练——它们何时结束无从判断'. That is false for both background trainings here. Each is polled roughly 20 times, and each prints its exact train_runtime into the…(i=443, i=443, i=534, i=534, i=448, i=538)
- **边界情形 2 条**
  - prepare_data2.py (c8) is delivered as a data-preparation script - C3's stated territory, '决定训练数据从哪来、按什么比例混' - but its entire content is format alignment to the eval prompt, which is C2's definition, '…(i=252, i=254, i=262, i=344)
  - The verifier's own serving/sampling configuration is changed mid-run and no category covers it. run2 is measured with --max-connections 4 and default --gpu-memory-utilization (i=333) and reads 0.167; …(i=333, i=448, i=448, i=337, i=461, i=474)

## claude_non_api_claude-opus-4-7_10h_run3__aime2025_Qwen_Qwen3-1.7B-Base_17129517
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 4.07h | 0.0666666666… |

### 改动序列(11 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 56 | C3 | Data source and recipe fixed in prepare_data.py: open-r1/OpenR1-Math-220k 'default' subset (93,733 rows), keep only the first generation whose correct… | i=56, i=56, i=56, i=338 |
| 56 | C2 | Byte-level format alignment to the official scorer: the agent first read inspect_evals/aime2025.py + aime2024.py (i=37/38), then copied that USER_PROM… | i=56, i=56, i=38 |
| 64 | C4 | Training method written in train_sft.py: full-parameter SFT (no LoRA) via TRL SFTTrainer, bf16 + flash_attention_2 + gradient checkpointing, cosine sc… | i=64, i=64, i=64 |
| 102 | C4 | Replaced the tokenizer chat template during training with a hand-written single-turn template carrying {% generation %} / {% endgeneration %} around t… | i=102, i=102, i=99 |
| 116 | C1 | Post-training hook added to train_sft.py that rewrites the saved generation_config.json with eos_token_id=[151643, 151645] so generation stops on <\|i… | i=116, i=198 |
| 116 | C2 | Same edit also re-saves the ORIGINAL Qwen3 tokenizer (and therefore the stock chat template) into the checkpoint directory, so the custom generation-t… | i=116, i=116 |
| 217 | C1 | The run's decisive change. After the first full eval read 0.000, the agent diagnosed the saved generation_config (cat at i=197 -> max_new_tokens 2048,… | i=217, i=217, i=218, i=280, i=309 |
| 226 | C11 | Built an analysis script over the official inspect_ai log JSON that turns the scalar accuracy into per-item decision signals: per-sample correctness, … | i=226, i=228 |
| 245 | C3 | Second data recipe: build_short_data.py keeps only traces <=2500 tokens, dedupes on the first 200 chars of the user prompt, and merges the int and all… | i=245, i=245, i=248 |
| 293 | C11 | Repaired the C11 instrument after discovering vLLM returns message.content as a structured list for reasoning models (i=291), so the previous 'ANSWER:… | i=293, i=293, i=291, i=294 |
| 297 | C9 | Submission decision: with sft-v2 at 0.000 and sft-v1 at 0.067, wrote sft-v1 into final_model and then pruned the nested checkpoint-* dirs, README.md a… | i=297, i=299, i=296 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 119 | smoke | 0.01h | returned | **smoke** | First execution of train_sft.py. 32 examples, bs 1 x accum 2, 1 epoch, output to checkpoints/test, deleted three events later at i=122. Purpose is cod… |
| 125 | smoke | 0.01h | returned | **smoke** | vs i=119: max-examples 32->64, batch-size 1->4, grad-accum 2->4 (effective batch 2->16), wrapped in `timeout 120`. The stated purpose is throughput/VR… |
| 136 | real | 1.79h | consumed | **baseline** | baseline - first real training, no prior real run to contrast with. openr1_int.jsonl (19,776 records, 13,819 survive the <=4096-token filter), full-pa… |
| 252 | real | 0.94h | consumed | **both** | vs sft-v1 (i=136). Data: openr1_int.jsonl -> openr1_short.jsonl (19,776 -> 9,059 records, traces <=2500 tokens, deduped, non-integer answers re-admitt… |
| 302 | real | 0.36h | consumed | **C4** | vs sft-v2 (i=252). Data path byte-identical (openr1_short.jsonl) and max-seq-len 2560 / batch-size 8 / grad-accum 2 / seed 42 all unchanged. Only the … |

### 验证序列(4 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 202 | — | — | 是 | c1, c2, c3, c4, c5, c6 | 0.0 |
| 219 | — | — | 是 | c7, c1, c2, c3, c4, c5, c6 | 0.067 |
| 282 | — | — | 是 | c9 | 0.000 |
| 309 | — | — | 是 |  | 0.000 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - early_stop_budget_forfeit(i=322, i=322, i=328, i=300)
- **定义缺陷 3 条**
  - The sft-v1 span is credited 3.22h; its true length is ~1.79h, an overstatement of ~1.43h on a 4.07h run. The training process is already gone and the artifact already complete at i=195 (16:09:18): `ps…(i=202, i=195, i=195, i=200, i=302)
  - Counterexample to reading C11's 'deterministic' as 'trustworthy'. The agent's C11 instrument printed 'correct: 2, complete: 1, truncated: 29' for the 0.067 eval. The truncation column was invalid by c…(i=228, i=291, i=294, i=320)
  - Coverage gap, not a wrong value. The skeleton marks all 4 generation_config.json accesses in this run as 'has content: no', but for 3 of them the complete file is present verbatim in the immediately f…(i=198, i=279, i=326)
- **边界情形 3 条**
  - Data-forced hyperparameter changes - the exact boundary spec section 9 says has no rule, with numbers. sft-v2 changes the data (19,776 -> 9,059 records, <=2500 tokens) and four hyperparameters, but th…(i=252, i=191, i=275)
  - One edit with three equally defensible homes. Replacing the tokenizer's chat template with a generation-tagged one is (a) a training-method enabler - assistant_only_loss cannot build a mask without {%…(i=102, i=102, i=99)
  - 3 of this run's 5 training rows (i=119, i=125 smoke; i=136 the first real training) carry full, unambiguous evidence about what they do, and are still forced to `unclear` because the value domain C3/C…(i=119, i=136, i=125)

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
| 101 | smoke | 0.01h | returned | **smoke** | baseline。这是冒烟测试而不是真实训练:max_steps=2、只读 20 行数据、output_dir=/tmp/test_trl,目的是验 TRL + flash-attn + packing 配置跑不跑得通。它既不测 C3 也不测 C4 —— 见 boundary_case b2。真实占… |
| 107 | real | 6.35h | last_seen | **both** | baseline:首次真实训练,数据(OpenR1 过滤后 4,500 条 / 700–6500 tok)与方法(全参 SFT、1 epoch、lr 1e-5、bs1×accum8、packing=bfd、completion_only_loss)同时被首次确定,时间无法拆给任一方。**真实时长 1… |
| 203 | real | 5.12h | last_seen | **both** | vs v1:数据 4,500 → 10,000 条、长度带 700–6500 → 800–7000(C3);同时 lr 1e-5 → 2e-5、grad_accum 8 → 4(C4)。两类都动了,且 lr 翻倍是独立的一阶能力杠杆,不是数据的从属调整。**真实时长 4379.46 s = 1.21… |
| 300 | real | 1.37h | last_seen | **C3** | vs v2:只换数据 —— 616 条 AIME 1983–2023 R1 轨迹(×3)+ 5,000 条更长 OpenR1,共 6,848 条。lr / epoch / bs / accum / scheduler / warmup / packing 逐字未变;唯一伴随的 max_length … |
| 389 | real | 2.15h | last_seen | **both** | vs v3:数据换成 ≤8000 token 的短轨迹(511 AIME ×3 + 3,000 短 OpenR1 = 4,533)(C3),同时 num_train_epochs 1 → 2、max_length 回落 8192(C4)。agent 自述就是「materially different… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 126 | 3.0 | 10.0 | 是 | c1, c2, c3, c4 | 0.0(n=10)。agent 立刻去读 inspect_ai 日志,发现输出被 max_new_tokens=2048… |
| 150 | 3.0 | 10.0 | 是 | c5 | 0.3(n=10) |
| 154 | — | — | 是 | c5 | 0.0(n=30 全量,贪婪)。同一份权重 n=10 读到 0.3、n=30 读到 0.0 —— 本 run 内部就复现… |
| 165 | — | — | 是 | c6 | 0.033(n=30 全量,temperature 0.6) |
| 256 | — | — | 是 | c7, c8, c9 | 0.233(7/30);诊断出 16/30 撞 max_tokens |
| 273 | — | — | 是 | c10 | 0.100(--max-tokens 16000,对照 i=256 的 15000) |
| 326 | — | — | 是 | c12, c13, c14 | 0.200(6/30);仍有 16/30 撞 max_tokens |
| 355 | — | — | 是 | c7, c8, c15, c20 | 0.167 —— v2 第二次全量。分数由 i=373 直接 Read eval_v2_confirm.json 取回(… |
| 441 | — | — | 是 | c16, c17, c18 | 0.167(5/30)—— v4 第一次全量,分数经后台任务 b40d9b2fr 的 TaskOutput 在 i=45… |
| 454 | — | — | 否 |  | 未拿到。这次启动本身作废:同一条命令里 `rm -rf final_model` 把 shell 的 cwd 删掉了,后… |
| 463 | — | — | 是 | c7, c8, c20 | 0.133(4/30)—— v2 第三次全量,分数经后台任务 borb3iw22 在 i=473 连同 `cat eva… |
| 476 | — | — | 否 |  | 未拿到,且确实不存在。与 i=454 同一形态(cwd 被自己删掉 → ln -s 全失败 → 对空目录起 evalua… |
| 480 | — | — | 否 | c16, c17, c20 | 0.267(8/30)—— **机械层没关联到,但 agent 确实拿到了**。取回通道:i=482 起了一条 `sle… |
| 493 | — | — | 否 | c16, c17, c19, c20 | 0.233(7/30)—— 同样机械层没关联到而 agent 拿到了。通道:i=495 的后台任务 bvh6b4ngq … |

### 异常与存疑

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

## claude_non_api_claude-opus-4-7_10h_run3__aime2025_google_gemma-3-4b-pt_17126498
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | aime2025 | google_gemma-3-4b-pt | 7.88h | 0.0 |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 64 | C3 | Writes work/prepare_data.py: the v1 corpus comes from open-r1/OpenR1-Math-220k only, keeping the shortest math_verify-correct generation per problem, … | i=64, i=64, i=67 |
| 64 | C2 | Same script builds every training prompt by reproducing inspect_evals.aime2025 USER_PROMPT_TEMPLATE verbatim (read at i=23) and guarantees the respons… | i=64, i=64, i=23 |
| 64 | C10 | Same script adds an AIME-2025 decontamination pass: it loads math-ai/aime25 test.jsonl, whitespace/case-normalises, and drops any training problem who… | i=64, i=64, i=67 |
| 89 | C4 | Writes work/train_sft.py: the method for the whole run. LoRA via peft (default r=64/alpha=128) on Gemma3ForCausalLM text-only, TRL SFTTrainer with com… | i=89, i=89, i=92 |
| 104 | C8 | Writes work/merge_adapter.py, which loads the base weights, applies the adapter with PeftModel and merge_and_unload(), then save_pretrained()s a plain… | i=104, i=104, i=146 |
| 159 | C11 | Builds a reader over the inspect_ai eval log that extracts per-sample stop_reason and output_tokens alongside the score. This turns the official score… | i=159, i=160, i=236 |
| 185 | C3 | Writes prepare_data_v2.py: drops the generic OpenR1 mixture for AIME-specific material - jonathanyin/aime_1983_2023_deepseek-r1_traces_32768 (513 trac… | i=185, i=185, i=188 |
| 192 | C4 | Long-context training profile for v2: max-length 3072->8192, per-device-bs 2->1, grad-accum 4->8, lr 2e-4->1e-4, epochs 1->2 (LoRA rank unchanged at 6… | i=192, i=97 |
| 215 | C8 | Kills the OOM-crashed v2 attempt and its two orphaned children (PIDs 2429379/2429636/2429637), which were still pinning 71273 MiB of the H100 at 100% … | i=215, i=213, i=216 |
| 217 | C8 | Feasibility fix, not a hypothesis: after torch.OutOfMemoryError at i=210 the v2 launch is repeated with --max-length 8192 -> 6144 and everything else … | i=217, i=210, i=207 |
| 246 | C3 | Writes prepare_data_v3.py after seeing 17/30 v2 samples truncate at max_tokens: swaps the long R1 <think> traces for the short 'Solution Attempt' fiel… | i=246, i=246, i=251 |
| 263 | C4 | Short-context profile for v3: max-length 6144->2048, per-device-bs 1->2, grad-accum 8->4, lr 1e-4->2e-4, epochs 2->1, i.e. a full revert to the v1 hyp… | i=263, i=217 |
| 293 | C8 | Housekeeping delete of lora_v1 and of every intermediate checkpoint of lora_v2/lora_v3 (repeated at i=190 and i=323). Not disk pressure - i=358 shows … | i=293, i=190, i=358 |
| 310 | C3 | Writes prepare_data_v4.py: mixes the two previous extremes - 128 full R1 reasoning traces (<=3500 tokens) + all 877 solution-only rows + 8000 short Nu… | i=310, i=310, i=313 |
| 314 | C4 | Doubles LoRA capacity for v4: --lora-r 64->128 and --lora-alpha 128->256 (trainable params later reported as 238,419,968 / 5.79% at i=363), while max-… | i=314, i=263, i=363 |
| 336 | C2 | Writes prepare_data_v5.py whose stated and only new idea is answer-domain alignment: keep a training row only if its gold answer matches ^\d+$, so the… | i=336, i=336, i=334 |
| 360 | C4 | Budget-capped profile for v5: adds --max-steps 1200 (stops at epoch 0.71 of the 13474-row set), lr 1e-4->2e-4, epochs 2->1, save-steps 500->1200; max-… | i=360, i=385 |
| 428 | C9 | First submission decision: copies outputs/merged_v5 to final_model. Pure candidate selection with no artifact change - and at this point every candida… | i=428, i=419, i=415 |
| 446 | C3 | Writes prepare_data_v6.py, reversing c11: back to full R1 reasoning (337 traces <=5500 tokens, p50 response 5815 chars) plus the 876 solution-only row… | i=446, i=446, i=449 |
| 451 | C4 | Long-context profile for v6: max-length 3072->6144, per-device-bs 2->1, grad-accum 4->8, lr 2e-4->1.5e-4, --max-steps dropped. The script's own docstr… | i=451, i=446, i=360 |
| 489 | C9 | Second submission decision: deletes final_model and replaces it with merged_v6 on the strength of a single 1/30 reading. No artifact is modified. The … | i=489, i=488 |
| 503 | C11 | Fixes the C11 reader: inspect_ai returns message.content either as a str or as a list of {type:'reasoning'\|'text'} parts, and the str-only version ha… | i=503, i=498, i=504 |
| 516 | C11 | Repeats the full 30-sample eval on byte-identical final_model weights purely to size the scorer's noise - the '重复取均值' branch of C11. Two repeats (i=50… | i=515, i=516, i=524 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 91 | smoke | 0.01h | returned | **smoke** | baseline - first launch of the run and a pure smoke: --subset 20 --max-steps 2 --per-device-bs 1 on sft_5k_short, foreground, 0.01h. It tests that tra… |
| 97 | real | 4.27h | discarded | **baseline** | baseline - first real training. sft_10k_mid (10000 rows from OpenR1-Math-220k) at max-length 3072 / bs2 / accum4 / lr 2e-4 / 1 epoch / LoRA r=64. Ther… |
| 192 | real | 0.01h | superseded | **both** | vs lora_v1: data sft_10k_mid -> sft_v2_aime_focused (c7) AND max-length 3072->8192, bs 2->1, accum 4->8, lr 2e-4->1e-4, epochs 1->2 (c8). Two axes mov… |
| 207 | real | 0.03h | superseded | **both** | Byte-identical relaunch of i=192 with `disown` added so the process survives the harness task; it therefore tests the same v2 data + v2 hyperparameter… |
| 217 | real | 2.85h | last_seen | **both** | vs i=207: only --max-length 8192 -> 6144, a pure OOM compensation (c10) that per spec §10.2 should not count as a tested variable. Against the last tr… |
| 263 | real | 1.12h | last_seen | **both** | vs lora_v2: data sft_v2_aime_focused -> sft_v3_concise (c11, ~6x shorter responses) AND max-length 6144->2048, bs 1->2, accum 8->4, lr 1e-4->2e-4, epo… |
| 314 | real | 1.38h | last_seen | **both** | vs lora_v3: data sft_v3_concise -> sft_v4 (c14, mixes long R1 traces back in) AND LoRA r 64->128 / alpha 128->256, max-length 2048->3072, lr 2e-4->1e-… |
| 360 | real | 0.61h | last_seen | **both** | vs lora_v4: data sft_v4 -> sft_v5 (c16, integer-answer-only filter) AND --max-steps 1200 added, lr 1e-4->2e-4, epochs 2->1 (c17). The agent's stated h… |
| 451 | real | 0.76h | last_seen | **both** | vs lora_v5: data sft_v5 -> sft_v6 (c19, full R1 reasoning, 3916 rows) AND max-length 3072->6144, bs 2->1, accum 4->8, lr 2e-4->1.5e-4, --max-steps rem… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 149 | 3.0 | 5.0 | 是 | c1, c2, c3, c4, c5 | 0.0 (0/5) |
| 162 | — | — | 是 | c1, c2, c3, c4, c5 | 0.0 (0/30) |
| 232 | — | — | 是 | c7, c8, c10 | 0.0 (0/30); read alongside the C11 truncation signal - 17/30… |
| 255 | — | — | 是 |  | 0.0 - untrained google/gemma-3-4b-pt baseline anchor, judges… |
| 273 | — | — | 是 | c11, c12 | 0.0 (0/30) - confirmed sample-by-sample at i=282 'Total corr… |
| 323 | — | — | 是 | c14, c15 | 0.0 (0/30) |
| 402 | — | — | 是 | c16, c17 | 0.0 (0/30) - read out of outputs/eval_v5.json at i=416/417 a… |
| 434 | — | — | 是 | c18 | 0.0 - confirmation that final_model (= merged_v5) loads and … |
| 480 | — | — | 是 | c19, c20 | 0.03333333333333333 (1/30) - the only non-zero reading of th… |
| 507 | — | — | 否 | c21 | 0.0 (0/30) - the skeleton marks this eval as no-signal, but … |
| 516 | — | — | 否 | c21, c23 | 0.0 (0/30) - same channel as i=507: i=518 backgrounds the ca… |

### 异常与存疑

- **2 次验证没有拿到信号**:i=[507, 516]
- **分类学缺口提案 1 条**
  - cross_run_memory(i=529, i=529, i=28, i=531)
- **定义缺陷 3 条**
  - The consumption list is closed over upstream-supplied tooling and misses run-local tooling, which inflates background training spans. This run consumes every adapter with its own work/merge_adapter.py…(i=141, i=141, i=145, i=293, i=227, i=229)
  - A launch command is counted as a training start even when the process demonstrably never existed. At i=192 the nohup train_sft.py was issued inside a harness background bash (run_in_background), and t…(i=192, i=202, i=200, i=205, i=207)
  - The skeleton marks i=507 and i=516 as got_signal=False, but both scores did reach the agent. The pattern is a fifth channel the extractor does not follow: claude-code's TaskOutput. The agent backgroun…(i=507, i=509, i=513, i=518, i=522)
- **边界情形 4 条**
  - prepare_data_v5.py (c16) filters the corpus on the shape of the gold answer - keep the row only if the answer matches ^\d+$ - because AIME targets are integers and latex/letter answers can never match…(i=336, i=336, i=334, i=246)
  - This run is a clean instance of §9's unresolved boundary - data changes mechanically forcing hyperparameter changes - and it shows the rule has to cut three ways, not two. At i=451 the docstring state…(i=446, i=451, i=263, i=210)
  - The run's first three artefacts - prepare_data.py (i=64), train_sft.py (i=89), merge_adapter.py (i=104) - are the construction of the pipeline, not a modification of one. §1 defines a change as 'agent…(i=104, i=104, i=89, i=62)
  - C9 assumes a submission guard chooses among candidates on some signal, and §4.4 prices its absence ('手里握着 80% 的模型,交了 61%'). Here both C9 events fire with no usable signal at all. At i=428 every candid…(i=415, i=488, i=486, i=515, i=524)

## claude_non_api_claude-opus-4-8_10h_run1__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17308543
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 7.72h | 0.2 |

### 改动序列(18 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 99 | C3 | 选定训练数据来源与配方:OpenR1-Math-220k default split,只留 correctness_math_verify 为真且答案是纯整数的题,每题取最短的带 <think>/</think> 的正确 R1 轨迹,产出 32,482 条 sft_data.jsonl | i=99, i=118 |
| 99 | C2 | 格式对齐:把评测端的 USER_PROMPT_TEMPLATE 逐字抄进训练样本,并在每条 assistant 末尾补一行 ANSWER: <整数>,使训练样本与 aime_scorer 取分的最后一行格式一致 | i=99, i=99 |
| 185 | C2 | 格式对齐(模板层):训练时把 tokenizer.chat_template 换成评测同一份 templates/smollm.jinja,并用 assistant_only_loss 只在 assistant 段计损失,训练渲染与推理渲染逐字一致 | i=185, i=193 |
| 185 | C4 | 训练方法与超参:TRL SFTTrainer 全参 SFT,packing=True、assistant_only_loss=True、max_length 16384、bf16 + flash_attention_2 + gradient_checkpointing、lr 1e-5 cosine、… | i=185, i=185 |
| 262 | C5 | 把 save_total_limit 从 1 改成 3,为的是留下两个 epoch checkpoint 以便事后挑选(实际上直到 run 结束都没评过 epoch-1 checkpoint) | i=262, i=261 |
| 421 | C1 | 写 set_gen_config.py:把 eos_token_id 设为 [128001,128012](同时接受 <\|end_of_text\|> 与 chat 模板收尾的 <\|im_end\|>),并同步改 config.json 的 eos_token_id;修的是模板收尾符与 base… | i=421, i=409 |
| 493 | C1 | 把 sft_run1 的解码配置设成贪婪:temperature 0.0,并 pop 掉 top_p / top_k(set_gen_config.py 是就地读改写,不是整份重写,原有字段保留) | i=493, i=494 |
| 565 | C11 | 自建验证器工装:直接读 inspect_ai 的 logs/*.json,从官方评分器自己的输出里统计 hasANSWER / closed </think> / completion token 长度 / >=15800 截断数,得到零噪声可排序的终止率判据 | i=565, i=593, i=596 |
| 650 | C1 | 把 sft_run1 的解码从贪婪改成采样 temperature 0.6 / top_p 0.95,假设是采样能跳出贪婪的不终止循环 | i=650, i=651 |
| 655 | C3 | 第二版数据配方 prep_data2.py:从同一个 sft_data.jsonl 池里只保留 assistant <=5120 token 的样本,取前 20,000 条(中位 2662 token,原池中位 3832),目标是让模型在 16k 生成预算内收尾 | i=655, i=669 |
| 706 | C4 | run2 的序列长度超参:--max_len 8192 --max_tok_filter 8192(run1 用默认 16384);agent 自述这是随数据变短做的效率下调,不是被测假设 | i=706, i=705 |
| 757 | C1 | 把 sft_run2 的解码配置设成贪婪 temperature 0.0(与 run1 同一套 set_gen_config.py) | i=757, i=758 |
| 841 | C1 | 把 sft_run2 的解码改成 temperature 0.6 / top_p 0.95 | i=841, i=842 |
| 980 | C9 | 提交守卫:在还剩 2h39m 时把当时读到最高分的候选 sft_run2 整目录复制成 final_model 并配 temp0.6 config,先锁一个安全交付物再继续做实验(不改任何产物,只决定此刻交哪个) | i=980, i=983 |
| 989 | C8 | 清掉 final_model 里随整目录复制进来的 checkpoint-991 / checkpoint-1982 子目录,使交付目录自洽并把体积压到 5.8G | i=989, i=988 |
| 989 | C1 | 把 sft_run2 的采样温度提到 0.8 / top_p 0.95,测更高温度能否把终止率再推高 | i=989, i=990 |
| 1084 | C1 | 在 temp 0.6 / top_p 0.95 之上加 repetition_penalty 1.1;写法是 json.load 后改键再 dump,不是整份重写,eos/bos/pad 等原有字段全部保留(下一事件逐字打印了结果) | i=1084, i=1085 |
| 1198 | C1 | 把 sft_run1 的解码从贪婪改回 temperature 0.6 / top_p 0.95,为的是让 run1 与 run2 在同一解码配置下对比数据配方 | i=1198, i=1199 |

### 训练序列(3 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 206 | smoke | 0.08h | discarded | **smoke** | 本 run 第一次训练启动,是冒烟:400 条样本、1 epoch、accum 4、timeout 420,只为量吞吐与验证 packing/assistant 掩码跑得通,不测任何数据或超参假设。冒烟在 timeout 内跑完并落盘(i=234 成功从 smoke_out 加载模型生成),随后被 … |
| 264 | real | 4.00h | consumed | **baseline** | baseline —— 本 run 第一次真实训练,没有前一次可比。18,000 条 sft_data.jsonl、2 epoch、lr 1e-5、accum 8、max_len 默认 16384。结局:干净跑完并落两个 epoch checkpoint,trainer 自报 train_runti… |
| 706 | real | 2.62h | consumed | **C3** | 相对 sft_run1 只换数据配方:sft_data2.jsonl(同一池子里只留 assistant <=5120 token 的 20,000 条,中位 2662 vs 原 3832);epoch / lr / accum 逐字不变(2 / 1e-5 / 8)。唯一同时动的超参是 max_le… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 62 | — | — | 是 |  | 0.0(base 模型 0/30,基线测量,不判定任何改动) |
| 497 | — | — | 是 | c1, c2, c3, c4, c6, c7 | 0.1(3/30);随后由 c8 工装从同一次评测的 inspect_ai 日志里再读出 closed </think>… |
| 652 | 3.0 | 15.0 | 是 | c9 | 0.2(3/15,n=15 属第三档);工装同时读出 hasANSWER=10/15、medtok=706 |
| 757 | — | — | 是 | c10, c11, c12 | 0.1(3/30);工装读出 closed=0/30、hasANSWER=6/30、trunc=24/30 —— 浓缩数… |
| 841 | — | — | 是 | c10, c11, c13 | 0.2(6/30,本 run 最高一次);工装读出 hasANSWER=19/30、medtok=647、trunc=1… |
| 989 | — | — | 是 | c16 | 0.1(3/30);工装读出 hasANSWER=21/30、trunc=7 —— 终止率升而准确率降 |
| 1084 | — | — | 是 | c17 | 0.16666666666666666(5/30);工装读出 hasANSWER=27/30、trunc=2 —— 截断… |
| 1160 | — | — | 是 | c14 | 0.13333333333333333(4/30)—— 同一份权重同一份 config 的重复测量,与 i=841 的 … |
| 1201 | — | — | 是 | c1, c10, c18 | 0.16666666666666666(5/30)—— 与 run2 在同一 temp0.6 下比,两版数据配方判不出差… |

### 异常与存疑

- **分类学缺口提案 1 条**
  - cross_session_state(i=358, i=1180, i=467)
- **定义缺陷 2 条**
  - config 访问表按命令文本里是否出现字面量 generation_config.json 来定位,在这条 run 上同时产生假阳性和大面积漏抓。假阳性:i=362 是往 MEMORY.md 追加一行笔记,那行笔记的正文里提到了 generation_config.json,被记成一次 read。漏抓:真正的写入全部走 set_gen_config.py <dir> <temp> [top_p]…(i=362, i=493, i=1198, i=494)
  - C1 的效应方向在这条 run 上是反的,而且是同一份权重的近单变量对照。sft_run2 同一目录、同一条 evaluate.py 全量 30 题,只改 generation_config.json:temperature 0.0(贪婪)读到 0.1(3/30),temperature 0.6 + top_p 0.95 读到 0.2(6/30)。同一次评测的官方日志给出确定性判据:贪婪下 clo…(i=824, i=833, i=909, i=924, i=840)
- **边界情形 3 条**
  - i=706 这次训练的 tested_variable 落在 C3 与 both 之间。数据配方换了(浓缩到 <=5120 token),同时 max_len / max_tok_filter 从 16384 降到 8192。reference §3 明确把序列长度列为 C4,按字面口径应记 both;但 agent 自述这一项是随数据变短做的效率下调(with a smaller max_len…(i=706, i=705, i=669)
  - rm -rf final_model/checkpoint-* 落在 C8 与 C9 之间。按 C8 它是清磁盘 / 让交付目录自洽(5.8G),按 C9 它改的正是「此刻交付目录里有什么」——而且删掉的两个 checkpoint 恰恰是 C5 意义上的备选。两类定义都能套上,现有边界判不了。本条记为 C8。(i=989, i=988, i=983)
  - 改动的粒度边界:reference §1 把「改动」定义为一次有意图的修改,可机械抽取字段是「触发它的命令、时刻、所属类型」——所属类型是单值,且以事件为键。这条 run 上有两处一条事件承载两类改动:i=99 的一次 Write 同时决定了数据来源(C3)与 prompt/答案格式对齐(C2);i=989 的一条命令同时是 C8(删 checkpoint)与 C1(设 temp 0.8)。要么把…(i=99, i=99, i=989, i=989)

## claude_non_api_claude-opus-4-8_10h_run1__aime2025_Qwen_Qwen3-1.7B-Base_17308542
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 10.08h | 0.0333333333… |

### 改动序列(27 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 124 | C3 | 从 open-r1/OpenR1-Math-220k 蒸馏 SFT 数据:只留 correctness_math_verify 判对且答案为整数的题,每题取最短的正确 generation,全序列 token 数限制 200-14000;93733 行筛出 30707 条。 | i=124, i=124, i=200 |
| 124 | C2 | 训练文本逐字复刻评测 prompt:USER_PROMPT_TEMPLATE 与 inspect_evals/aime2025 同文,assistant 段以 'ANSWER: N' 结尾,外层用 qwen3 模板标记手工拼接(不走 apply_chat_template);先读了 match 评分… | i=124, i=124, i=102 |
| 183 | C10 | 写 filter_contam.py,对 math-ai/aime25 test 做 12-gram 重叠去污染,30707 条里剔掉 220 条命中样本。 | i=183, i=217 |
| 355 | C8 | bs=4 触发 CUDA OOM(要 37GB)后装 liger-kernel 并在 SFTConfig 里开 use_liger_kernel=True;显存从 49GB 降到 23GB,有效 batch 不变。 | i=340, i=355, i=358, i=365 |
| 389 | C8 | 把 gradient_checkpointing 改成 --grad_ckpt 开关并试着关掉以提速(bs2/accum8),重启后再次 OOM,随即恢复开启。 | i=389, i=398, i=421 |
| 422 | C5 | 把 save_strategy 从 no 改成 steps(save_steps=145, save_total_limit=8),让训练中途产出可评测、可挑选的 checkpoint。 | i=422, i=422, i=748 |
| 643 | C1 | 把 checkpoint 拷到 /tmp/ck145test 并整份重写 generation_config.json:max_new_tokens 2048->30000,新增 temperature 0.6 / top_p 0.95 / top_k 20;eos_token_id 仍为 [151… | i=619, i=643, i=642, i=673 |
| 703 | C1 | 写 prep_and_eval.sh:每次评测前把 checkpoint 复制到 runs/eval_<tag> 并整份写入 generation_config.json —— eos_token_id 加入 151645(<\|im_end\|>)、max_new_tokens 30000、tem… | i=703, i=703 |
| 825 | C11 | 把官方评测 JSON 的输出转成确定性判据:统计 30 题里多少条含 'ANSWER:'、多少条闭合 '</think>'、输出长度分布。分数被噪声吞掉(连续多次 0/30)时,这个收尾率是唯一有序信号。 | i=825, i=826, i=824 |
| 845 | C1 | 把 prep_and_eval.sh 参数化(TEMP/TOPP/TOPK/REP/MAXNEW),使 generation_config 的解码字段可以逐次扫;同时给写出的 config 加上 repetition_penalty 字段。 | i=845, i=845 |
| 854 | C1 | 把 repetition_penalty 从 1.0 提到 1.15,想压住输出末端的字面重复(第一次 30 题全量后发现 0/30 都没收尾)。 | i=854, i=908 |
| 880 | C3 | 按整条字符长度切出精简子集 sft_short8k(<=8000 字符,9560 条,原 30487 条),目的是让模型学会在 16k 预算内收尾。 | i=880, i=881, i=879 |
| 1148 | C4 | 换训练目标:新写 train_pc.py,packing True->False、completion_only_loss=True、eos_token='<\|im_end\|>'、group_by_length=True、lr 1e-5->2e-5、maxlen 16384->4096;同一批样… | i=1148, i=1148, i=1102, i=1147 |
| 1259 | C3 | 过滤掉 completion/prompt 里非拉丁字符超过 5 个的样本(pc_clean 30487->30077,pc_short8k 9560->9382),因为 R1 轨迹会漂到中文/阿拉伯语。 | i=1259, i=1262 |
| 1361 | C3 | kill 掉还在跑的 pc_short 训练,从 pc_clean_en 里按整条 <=16500 字符(约 6000 token)切出 pc_mid(20751 条),把配方从'很短'改成'中等长度'。 | i=1361, i=1361, i=1384 |
| 1406 | C4 | 给 train_pc.py 加 --pack 开关,把 packing 与 padding_free 同时接到该开关上并与 completion_only_loss 共存;动机是不打包太慢、装不进剩余时间预算。事前读了 TRL 源码确认 packing 路径会保留 completion_mask。 | i=1406, i=1408, i=1392 |
| 1476 | C9 | 把已经评测过的 runs/eval_pc1ep 整份拷进 final_model 作 safety net,保证任何时刻都有一个格式完整、能停下来的可提交产物。 | i=1476, i=1475 |
| 1623 | C9 | pc_mid checkpoint-525 拿到 3/30 后,用 runs/eval_mid1ep 覆盖 final_model。 | i=1623, i=1622 |
| 1673 | C5 | 在同一次 pc_mid 训练的两个 checkpoint 之间选:ck1050(2 epoch)全量 30 题 2/30,低于 ck525(1 epoch)的 3/30,保留 ck525 作为提交权重。 | i=1673, i=1727, i=1611 |
| 1799 | C1 | 解码从 temperature 0.6 改成贪婪(TEMP=0.0)并加 repetition_penalty 1.05,想换掉采样带来的方差。 | i=1799, i=1833 |
| 1843 | C9 | 把 runs/eval_m525_greedy(贪婪 rep1.05 的那份 config)整份拷成 final_model,并立刻用官方 evaluate.py 自检。 | i=1843, i=1890 |
| 1907 | C1 | 试 repetition_penalty 1.15 + 贪婪,想提高收尾率;该次评测崩了,没拿到任何数字。 | i=1907, i=1946 |
| 2191 | C1 | 在 grader 默认设置(mc=6)下贪婪 rep1.05 拿到 0/30 后,整份改写 final_model/generation_config.json:temperature 0.0->0.6、repetition_penalty 1.05->1.0(top_p 0.95 / top_k … | i=2191, i=2191, i=2169 |
| 2571 | C3 | 换成偏长轨迹的 pc_full15k(pc_clean_en 里最长 8000 条 + 其余隔一取一 7000 条,共 15000 条),想教会模型在长推理后也收尾;同时 maxlen 8192->16384、epochs 2->1。 | i=2571, i=2571, i=2570 |
| 2929 | C1 | 把 final_model 改成严格贪婪:temperature 0.6->0.0、top_p 0.95->1.0、top_k 20->-1、repetition_penalty 保持 1.0,再用 grader 默认命令测。理由是贪婪近乎确定性,grader 能复现。 | i=2929, i=2928 |
| 3058 | C9 | 贪婪评测还没回来时先把 final_model 回退到 temperature 0.6 / top_p 0.95 / top_k 20 的安全默认,保证时间用尽也有一个有效提交。 | i=3058, i=3057 |
| 3116 | C9 | 贪婪 rep1.0 在 grader 精确设置下拿到 2/30 后,把 final_model 锁定为该 config(temperature 0.0 / top_p 1.0 / top_k -1 / rep 1.0),权重仍是 pc_mid checkpoint-525。 | i=3116, i=3108 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 231 | real | 0.09h | superseded | **baseline** | baseline(全 run 第一次真实启动)。sft_clean 30487 条,epochs 2 / lr 1e-5 / maxlen 16384 / bs 1 / accum 16(有效 batch 16)。测到 15.4s/it 约 5 小时后被 kill。它不与任何前一次训练对照,test… |
| 333 | real | 0.02h | superseded | **unclear** | 相对 i=231 只把 bs 1->4、accum 16->4,有效 batch 仍是 16;数据/lr/epochs/maxlen 逐字未变。动机是 bs=1 下 H100 吞吐太低。启动约 30 秒后 CUDA OOM。这是 §3 C8(运行时/可行性)而非 C4 的受测变量,但取值域装不下 C… |
| 360 | real | 0.02h | superseded | **unclear** | 命令与 i=333 逐字相同,唯一差别是 train_sft.py 里加了 use_liger_kernel=True。纯显存修复(49GB->23GB),数据与全部优化超参未变。同 i=333,取值域装不下 C8。 |
| 391 | real | 0.02h | superseded | **unclear** | 相对 i=360 改 bs 4->2、accum 4->8(有效 batch 仍 16)并 --grad_ckpt 0,想用省下的显存换 30% 速度;再次 OOM,随后恢复 grad_ckpt=1。同为 C8 动机。 |
| 424 | real | 0.44h | consumed | **unclear** | 前四次同产物启动全部被作废后,这一次是真正跑完的 baseline:数据 sft_clean、epochs 2、lr 1e-5、maxlen 16384,有效 batch 仍 16(bs4/accum4/grad_ckpt1),唯一新增的是 save_steps=145。相对 i=231 的意图没有… |
| 909 | real | 5.46h | killed | **both** | 相对 i=424 同时动了两件事:数据 sft_clean(30487)-> sft_short8k(9560,<=8000 字符)属 C3;epochs 2->3 属 C4。lr/maxlen/bs/accum/grad_ckpt 逐字未变。agent 自述只提数据('retrain from b… |
| 1167 | real | 4.68h | killed | **C4** | 数据内容与 i=909 完全同一批 9560 条样本(只是从 text 单列改写成 prompt/completion 两列,i=1147 打印 9560 与 i=881 的 9560 一致),变的全是方法:换 train_pc.py、packing True->False、completion_o… |
| 1410 | real | 4.13h | killed | **both** | 相对 i=1167 数据换成 pc_mid(20751 条,<=16500 字符,且已过英文过滤)属 C3;方法上 packing 0->1 与 padding_free 打开、maxlen 4096->8192、epochs 3->2、bs 8->4/accum 2->4(有效 batch 仍 1… |
| 2571 | real | 1.56h | killed | **C3** | 相对 i=1410 只有数据配方是被测的:pc_mid(<=16500 字符,20751)-> pc_full15k(最长 8000 条 + 其余隔一取一 7000 条,共 15000,刻意偏长)。同时 maxlen 8192->16384 是对更长样本的机械补偿,epochs 2->1 是剩余 2… |

### 验证序列(18 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 42 | 3.0 | 15.0 | 是 |  | 0.0(15 题基线,accuracy 0.000)。同时读到 base 模型输出完全不连贯,直接决定走 SFT 蒸馏路… |
| 540 | 3.0 | 4.0 | 是 | c1, c2 | 0.0(4 题)。真正读回去的不是分数而是输出形态:8192 output token 被 max_new_tokens… |
| 645 | 3.0 | 2.0 | 是 | c7 | 0.0(2 题)。判定的是解码上限:输出从 8,192 token 变成 32,000 token,确认 vLLM 读的… |
| 757 | — | — | — | c1, c2, c8 | 0.0(全量 30 题;骨架记的 0.066666 与流里 agent 自己 cat 出来的 {"accuracy": … |
| 854 | — | — | — | c10, c11 | 0.0(全量 30 题)。C11 判据:haveANSWER 0 / closed 4,rep 1.15 没有提高收尾率… |
| 975 | — | — | — | c12 | 0.0(全量 30 题)。C11 判据从 0 升到 haveANSWER 12 / closed 1 —— 精简数据让模… |
| 1277 | — | — | — | c13 | 0.0(全量 30 题,不是骨架记的 limit 8;那个 8 是 --max-connections)。C11 判据:… |
| 1563 | — | — | — | c14, c15, c16 | 0.1 = 3/30(全量 30 题,不是骨架记的 limit 8)。本 run 第一个非零分。 |
| 1673 | — | — | — | c_ckpt | 0.066666… = 2/30(ck1050,2 epoch),低于 ck525 的 3/30。 |
| 1744 | — | — | — | c_ckpt | 0.033333… = 1/30。同一份 ck525、同样 temp 0.6,与 i=1563 的 3/30 差 2 题… |
| 1799 | — | — | — | c19 | 0.1 = 3/30,concluded 4。贪婪 rep1.05。 |
| 1843 | — | — | 是 | c19, c20 | 0.066666… = 2/30。权重与 config 是 runs/eval_m525_greedy 的逐份拷贝,评测… |
| 1907 | — | — | — | c21 | 未拿到。评测崩在 AttributeError: 'NoneType' object has no attribute … |
| 1988 | — | — | 是 | c19, c20 | 0.0 = 0/30,concluded 4。这是 grader 默认命令(max_connections=6, gpu… |
| 2191 | — | — | 否 | c22 | 0.0 = 0/30,concluded 5。骨架记'没拿到分数',但 agent 在 i=2381 通过 Read h… |
| 2430 | — | — | — | c13, c19 | 确实未拿到。prep_and_eval.sh 要拷的 runs/pc_short 因为该训练在 i=1361 被中途 k… |
| 2839 | — | — | — | c23 | 0.033333… = 1/30,concluded 3。偏长数据反而收尾更少,假设被证伪。分数经 Read 后台任务输… |
| 2929 | — | — | 否 | c24 | 0.066666… = 2/30(答对第 60、49 题),concluded 1。骨架记'没拿到分数',实际在 i=3… |

### 异常与存疑

- **4 段训练的受测变量判不出**:i=[333, 360, 391, 424]
- **13 次验证没有拿到信号**:i=[757, 854, 975, 1277, 1563, 1673, 1744, 1799, 1907, 2191, 2430, 2839, 2929]
- **分类学缺口提案 1 条**
  - proposed:process-memo(i=1995, i=1481, i=2906, i=3141)
- **定义缺陷 8 条**
  - (i=540, i=742, i=746)
  - (i=971, i=1361, i=1667, i=2828, i=3128)
  - (i=845, i=1563, i=1611)
  - (i=791, i=792, i=792)
  - (i=1945, i=1946, i=1973)
  - (i=2380, i=2381, i=3108)
  - (i=2552, i=3128, i=2543)
  - (i=1102, i=1143, i=1244)
- **边界情形 4 条**
  - (i=231, i=333, i=391, i=325)
  - (i=1384, i=1410, i=1143)
  - (i=2542, i=2570, i=2571)
  - (i=1799, i=1833, i=1843, i=1890)

## claude_non_api_claude-opus-4-8_10h_run1__aime2025_Qwen_Qwen3-4B-Base_17308541
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 10.08h | 0.2333333333… |

### 改动序列(13 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 117 | C3 | 选定训练数据来源与配方:从 open-r1/OpenR1-Math-220k 抽 R1 长 CoT 解答,只保留 correctness_math_verify 为真的生成,再按「\boxed 里是 0..999 的纯整数」(AIME 答案域)与 300–12000 token 长度过滤,上限 20… | i=117, i=117, i=117, i=187 |
| 117 | C2 | 格式对齐:把 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE 逐字抄进训练样本的 user 侧(i=15 是原文来源),并强制每条 assistant 结尾追加一行 'ANSWER: <整数>',与官方 match(numeric=True) 的取末尾数字… | i=117, i=117, i=15 |
| 221 | C4 | 训练方法与配置写死在 train_sft.py:全参 SFT(不用 LoRA,规避 reference §C4 的 lm_head/embed_tokens 陷阱)、bf16 + flash_attention_2、TRL packing(bfd)、max_length 16384、cosine +… | i=221, i=221, i=221 |
| 221 | C2 | 格式对齐的第二半:训练文本不用模型自带模板,而是把评测用的 templates/qwen3.jinja 读进来当 chat_template,再 apply_chat_template(add_generation_prompt=False) 渲染,使训练串与评测串同模板;该等价性在 i=180/1… | i=221, i=221, i=183 |
| 289 | C8 | 冒烟通过后删掉 ckpt_smoke 释放磁盘,再看 timer(C8 「清磁盘」一支;不改任何候选,也不指望提分)。 | i=289 |
| 291 | C3 | 在取子集之前插入 ds.shuffle(seed=42):把「取 sft_data.jsonl 前 12000 行」改成「打乱后取 12000 行」,避免子集被数据集原始顺序(来源/难度)系统性偏置。这是唯一一次对配方的修订,且发生在真实训练启动前 3 秒。 | i=291, i=291 |
| 293 | C4 | 真实训练的超参取值:lr 抬到 1.5e-5(脚本默认 1e-5)、epochs 2、accum 16(冒烟时是 2)、max_length 16384。这些值与冒烟那次不同,但没有任何对照,只有一次训练。 | i=293, i=228 |
| 293 | C3 | 把训练集从已造好的 20000 条砍到 12000 条。理由是纯预算算术(冒烟测得 ~7000 tok/s → 12k×2 epoch ≈ 108M token ≈ 4.3h),不是任何关于数据的假设。见 boundary_case bc1。 | i=293, i=288 |
| 348 | C1 | 解码/服务配置(停止符支):finalize.py 把 model.config 与 generation_config 的 eos_token_id 同时设成 [151645(<\|im_end\|>), 151643(<\|endoftext\|>)],让 vLLM 在 assistant 轮结… | i=348, i=348, i=155 |
| 348 | C1 | 解码/服务配置(采样支):把 do_sample=true / temperature=0.6 / top_p=0.95 / top_k=20(Qwen3 thinking 推荐值)写进 final_model 的 generation_config.json。触发它的是第一档结论——evaluat… | i=348, i=348, i=343, i=331 |
| 432 | proposed:harness-control… | 改自己的等待机制:前台 sleep 被 harness 拦下(i=424)后,agent 先想用 Monitor(i=428/429),最后改成后台跑一条 until grep -qaE "saved to ckpt_sft\|Traceback\|..." train_main.log 的守卫,用… | i=432, i=424, i=419 |
| 510 | C9 | 提交:python finalize.py ckpt_sft final_model,把训练终点(epoch 2 结束的 ckpt_sft)无条件写进 final_model。同目录下还有 checkpoint-218(epoch 1)可选,两者都没被评过,也没有任何回归守卫或比较。C9 在这里是「… | i=510, i=503 |
| 519 | C1 | 解码配置的最后一次单字段修改:pop 掉 generation_config.json 里的 max_new_tokens=2048。该字段是 base 模型的 generation_config 经 save_pretrained 一路继承下来的(i=513 的打印里还在),而 finalize.… | i=519, i=513, i=520, i=518 |

### 训练序列(2 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 228 | smoke | 0.08h | discarded | **smoke** | baseline 之前的冒烟:200 条 × 1 epoch × accum 2,目的写明是「catch errors and measure throughput」,不验任何数据或方法假设。tested_variable 只能填 unclear,但这是**取值域装不下**(spec §10 要补的… |
| 293 | real | 3.70h | consumed | **baseline** | baseline —— 全 run 唯一一次真实训练,没有第二次可比。相对冒烟同时动了数据量(200→12000)、shuffle、epochs(1→2)、accum(2→16)、lr(1e-5→1.5e-5),但冒烟不产出可比分数,所以这不构成任何受测变量。仍填 unclear,同样是取值域装不下… |

### 验证序列(2 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 85 | 3.0 | 30.0 | 是 |  | 0.0 |
| 522 | 3.0 | 4.0 | 否 | c2, c4, c9, c10, c13 | 未拿到 |

### 异常与存疑

- **1 次验证没有拿到信号**:i=[522]
- **分类学缺口提案 1 条**
  - harness-control-loop(i=424, i=428, i=432, i=503, i=532)
- **定义缺陷 3 条**
  - (i=328, i=331, i=343, i=180)
  - (i=85, i=8, i=47)
  - (i=343, i=348, i=513, i=520)
- **边界情形 2 条**
  - 这是 spec §10 第 2 条(「数据改动强制的超参改动」)的**镜像**:那边是数据变逼着超参变,这边是预算逼着数据量变。建议写判据时把两个方向一起处理:凡是为保持某个外生量(墙钟、显存)不变而做的补偿性改动,都不计入受测变量。(i=288, i=293)
  - 建议把 C9 收紧成「存在一个会导致**不提交**的条件分支」(比较、阈值、回滚),无条件覆盖记为 C9 缺席而不是 C9。本条 run 按收紧后的定义就是 C9 缺席。(i=510, i=503)

## claude_non_api_claude-opus-4-8_10h_run1__aime2025_google_gemma-3-4b-pt_17311188
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | google_gemma-3-4b-pt | 8.98h | 0.0 |

### 改动序列(28 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 43 | C3 | 数据配方 v1:从 open-r1/OpenR1-Math-220k 抽 R1 长链 CoT,只留整数答案 0-999,每题在 correctness_math_verify 判对的 generation 里挑最短的一条(200-12000 completion token),seed 0 打乱写成… | i=43, i=43, i=132 |
| 43 | C2 | 格式对齐:训练样本的 user 侧提示词逐字抄自 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE(先在 i=16/17 把该文件 cat 出来读过),包括 ANSWER: $ANSWER 行与「不需要用 boxed」那句;assistant 侧把 <thi… | i=17, i=43, i=43 |
| 70 | C4 | 训练方法:自写 train_sft.py,全参微调(非 LoRA)、bf16、adamw_bnb_8bit、gradient checkpointing、cosine + warmup 0.03,bs1 × grad_accum16、lr 1e-5,prompt 段 label 置 -100 只在 … | i=70, i=70, i=70 |
| 70 | C2 | 格式对齐(训练侧):绕开 apply_chat_template,手工拼 <start_of_turn>user … <end_of_turn> <start_of_turn>model 三段,并在每条 completion 末尾追加 EOT(<end_of_turn>, id 106)进 labe… | i=70, i=70 |
| 70 | C1 | 解码/服务配置:训练器保存后自动把 eos_token_id=106 与 bos_token_id 写进 config.json 和 generation_config.json,理由写在注释里——让 vllm 能干净停下。 | i=70, i=70, i=70 |
| 85 | C8 | 运行时/可行性:把加载类从 AutoModelForCausalLM 换成 Gemma3ForConditionalGeneration,attn 先试 flash_attention_2 失败回落 sdpa,并冻结 vision_tower / multi_modal_projector 只训语言… | i=85, i=85, i=126 |
| 99 | C8 | 运行时/可行性:第一版 prep_data.py 单线程分词跑了 8 分钟看不到头,agent 在 i=97 kill 掉,改写成 ds.filter + ds.map(num_proc=16) 的并行版;数据语义(seed 0、整数答案过滤、最短正确 trace、12000 token 上限)逐字… | i=95, i=97, i=99 |
| 141 | C5 | checkpoint 选择的前置改动:save_strategy 由 "no" 改成 "epoch",加 save_only_model=True 与 save_total_limit=4,目的是留下 epoch-1 中途 checkpoint 以便和终点对比 / 崩溃兜底。 | i=141, i=141, i=139 |
| 193 | C1 | 解码配置:读完 inspect_ai 的 openai_completion_params 确认 evaluate.py 根本不发 temperature、vLLM 会回落读模型目录的 generation_config.json 之后,把 patch_eos.py 扩成强制贪婪——do_sampl… | i=192, i=193, i=193 |
| 213 | C11 | 验证器工装:写 run_eval.sh 把「先 patch generation_config → 跑官方 evaluate.py → cat metrics_<tag>.json」固化成一条带 tag 的命令,此后每次评测都走它,保证被评模型的解码配置与读分数的口径一致。 | i=213, i=213, i=213 |
| 302 | C8 | 运行时/可行性:i=279 的评测在 vLLM 引擎初始化就崩(Can't load image processor for 'run1'),因为 Trainer.save_model 不会拷多模态 processor 配置;从 base snapshot 把 preprocessor_config… | i=288, i=290, i=302 |
| 411 | C11 | 验证器工装:自写内联脚本读 inspect_ai 自己落盘的 logs/*aime2025*.json,统计 stop_reason 直方图(截断率)、completion 长度、是否出现 ANSWER 行、以及每题 pred vs target——把官方评分器的输出转成确定性的可决策信号,分数恒为… | i=411, i=412, i=519 |
| 417 | C1 | 解码配置:贪婪下 30 题里 28 题打满 max_tokens 从未写出 ANSWER 行,于是把 patch_eos.py 从贪婪翻成采样——do_sample=True、temperature=0.6、top_p=0.95、top_k=64(R1-distill 风格)。 | i=415, i=417, i=417 |
| 525 | C1 | 解码配置:temp 0.6→0.7 并加 repetition_penalty=1.1,针对 15 题里 11 题仍打满 token 的重复循环。 | i=525, i=520 |
| 596 | C9 | 提交守卫:在启动 run2 之前先把 run1 整份拷成 final_model 并 patch 解码配置,作为「安全网」,保证此后任何崩溃都还有一个可交的候选。不产生新产物,只决定此刻交谁。 | i=596, i=597, i=599 |
| 602 | C3 | 数据配方 v2(run2):同一份 sft_data.jsonl,但把 completion token 上限从 9000 收到 6000(要更简洁的推理链)、训练条数 11000→14000(要更广覆盖);max_len 也随之 9500→6500。 | i=601, i=602 |
| 602 | C4 | 超参:run2 把学习率从 1e-5 提到 1.5e-5(epochs、bs、grad_accum 保持 2 / 1 / 16 不变)。 | i=601, i=602 |
| 619 | C8 | 运行时/吞吐:序列变短后 GPU 只有 26% 利用率,给 TrainingArguments 加 group_by_length=True 并配合 bs 1→4 / grad_accum 16→4(有效 batch 仍 16)想提吞吐。 | i=619, i=614 |
| 646 | C8 | 运行时/可行性:bs=4 触发 CUDA OOM(262k 词表的 cross-entropy logits 单次要 24.56 GiB),改成 bs=2 / grad_accum=8 并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,有效 ba… | i=645, i=646, i=646 |
| 722 | C8 | 运行时/预算:bs=2 实测反而更慢(9.7s/step,ETA 4.7h),回到 run1 验证过的 bs1/grad_accum16,并把训练条数 14000→10000 以塞进剩余墙钟。 | i=717, i=722, i=722 |
| 791 | C8 | 运行时/可行性:把 processor 配置的拷贝动作固化进 run_eval.sh,任何后续 checkpoint 目录在评测前自动补齐 preprocessor_config.json / processor_config.json,避免 i=279 那类崩溃复发。 | i=790, i=791 |
| 795 | proposed:verifier-throug… | 把官方 evaluate.py 的 --max-connections 从写死的 6 改成可传参、默认 12,纯为把一次全量评测的墙钟从约 30 分钟压到约 15 分钟。改的是验证器自身的测量条件,不动任何被评产物,也不产生新判据。 | i=793, i=794, i=795 |
| 932 | C1 | 解码配置:把 patch_eos.py 整份改写成从环境变量取值(GEN_TEMP / GEN_TOPP / GEN_TOPK / GEN_REP,do_sample = temp > 0),使解码参数变成可扫的旋钮;随后 5 次评测就是这套旋钮的扫描。注:它是 json.load 后逐字段更新再 … | i=932, i=932, i=1499 |
| 1028 | C7 | 自建代理验证器:写 my_eval2024.py,用同一套 inspect_ai 管线在 held-out 的 AIME 2024 上评 run2,作为独立于官方 30 题 AIME2025 的第二个测试集(避免只在 30 题上做选择)。 | i=1028, i=1028 |
| 1229 | C9 | 提交守卫:把 final_model 从 run1 换成 run2(所有 30 题全量评测都是 0.0,依据是 C11 的截断率/收束率而非分数),同时补 processor 配置并把选定的解码参数 GEN_TEMP=0.7 GEN_REP=1.05 烤进 generation_config.jso… | i=1229, i=1229, i=1227 |
| 1325 | C3 | 数据配方 v3(run3):把每题的 trace 选择规则从「最短的正确 generation」翻成「最长的正确 generation(≤9000 token)」,seed 0→1,落成 sft_thorough.jsonl;假设是更详尽的推理示范能教会模型更细致的 casework。 | i=1325, i=1325, i=1323 |
| 1497 | C9 | 提交守卫(否决):run3 全量 0/30 后决定不把 final_model 指向 run3,保留 run2。不动任何产物,只在两个已有候选之间做保留决定;依据同样是收束率(run2 的 25/30 收束 vs run3 的 21/30),不是分数。 | i=1497, i=1496 |
| 1510 | C8 | 提交产物清理:把 final_model 里从 run2 拷进来的 checkpoint-* 子目录删掉(run2 目录带 checkpoint-625 / checkpoint-1250),压缩提交体积并避免加载歧义;删完在 i=1514 又跑了一次 limit 1 的加载验证。 | i=1510, i=1510 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 123 | smoke | 0.02h | returned | **smoke** | baseline(本 run 第一次启动)。8 行 dummy 数据、4 步,唯一目的是验证 train_sft.py 能加载 Gemma3、跑通 forward/backward 并落盘。不验任何数据或超参假设——取值域装不下,不是证据不足。结局:前台跑完,train_runtime 4.06s,… |
| 135 | real | 0.02h | returned | **baseline** | 对比 i=123 的冒烟:换成真实 sft_data.jsonl 的 64 条、grad_accum 8、max_comp_tok 6000、max_len 8192。目的是量 sec/example 好给 run1 定规模(读到 2.46 examples/s ≈ 7400 tok/s),不检验任… |
| 146 | real | 2.88h | consumed | **unclear** | baseline —— 第一次真实训练,没有可比的上一次。数据 = 配方 v1(最短正确 trace,max_comp_tok 9000 / max_len 9500 / 11000 条),超参 = lr 1e-5、2 epoch、bs1 × ga16。取值域装不下,不是证据不足。结局:跑满 2 个… |
| 602 | real | 0.03h | superseded | **both** | 对比 run1:数据侧 max_comp_tok 9000→6000、max_len 9500→6500、条数 11000→14000(要更简洁的推理链 + 更广覆盖);方法侧 lr 1e-5→1.5e-5。epochs / bs / grad_accum 未变。两类同时动,拆不开。结局:启动后 1… |
| 624 | real | 0.04h | superseded | **unclear** | 与 i=602 同数据、同 lr、同 epochs;只把 bs 1→4 / grad_accum 16→4(有效 batch 仍锁在 16)并加 group_by_length=True,动机是 GPU 利用率。按 reference §3 C8 的说明这是可行性/吞吐改动、不该计入受测变量,但四个… |
| 646 | real | 0.06h | superseded | **unclear** | 与 i=624 同数据、同 lr;bs 4→2 / grad_accum 4→8(有效 batch 仍 16)+ expandable_segments,纯为绕开上一次的 OOM。同样是 C8 形态,取值域装不下。结局:跑到第 5 步(9.74s/it)后被 agent 主动 kill(pid 88… |
| 722 | real | 2.15h | consumed | **both** | 紧接 i=646:batch 几何回到 run1 验证过的 bs1/ga16,条数 14000→10000(为墙钟)。这是 run2 四次启动里唯一跑完的一臂,它产出的候选与上一次跑完的训练(run1)相比,受测的是数据配方(max_comp_tok 9000→6000、max_len 9500→6… |
| 1335 | real | 2.06h | consumed | **C3** | 对比 run2:唯一被检验的是数据选择规则 —— sft_data.jsonl(每题最短的正确 trace)换成 sft_thorough.jsonl(每题最长的正确 trace ≤9000 token,seed 0→1)。lr 1.5e-5、epochs 2、bs 1、grad_accum 16 … |

### 验证序列(15 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 29 | 3.0 | 6.0 | 是 |  | 0.0 |
| 279 | 3.0 | 30.0 | 是 |  | 未拿到 —— vLLM 引擎初始化就失败(Engine core initialization failed / OSE… |
| 305 | 3.0 | 30.0 | 是 | c1, c3, c9, c11 | 0.0(30 题全量)。真正被读走的信号不是分数而是 C11 的 stop_reason 直方图:{'max_token… |
| 421 | 3.0 | 15.0 | 是 | c13 | 0.06666666666666667(15 题里 1 题对);stops {'max_tokens': 11, 'st… |
| 527 | 3.0 | 15.0 | 是 | c14 | 0.0(15 题);stops {'stop': 13, 'max_tokens': 2}、has_ANSWER 13 … |
| 850 | 3.0 | 30.0 | 是 | c16, c17, c14 | 0.0(30 题全量,run2 首次评测,解码沿用 temp 0.7 / rep 1.1)。 |
| 934 | 3.0 | 30.0 | 是 | c23 | 0.0(30 题,GEN_TEMP=0.3 GEN_REP=1.1);stops {'stop': 13, 'max_t… |
| 979 | 3.0 | 30.0 | 是 | c23 | 0.0(30 题,GEN_TEMP=1.0 GEN_REP=1.1 GEN_TOPP=0.95);stops {'sto… |
| 1030 | — | — | 是 | c24, c16 | 0.0(held-out AIME 2024 全 30 题,temp 0.6);逐题 tgt/pred 全错且多题 (n… |
| 1075 | 3.0 | 30.0 | 是 | c23 | 0.0(30 题,GEN_TEMP=0.7 GEN_REP=1.05);stops {'stop': 25, 'max_… |
| 1147 | 3.0 | 30.0 | 是 | c23 | 0.0(30 题,GEN_TEMP=0.6 GEN_REP=1.0,即去掉 repetition_penalty);st… |
| 1192 | 3.0 | 30.0 | 是 | c16, c17 | 0.0(30 题,run1,GEN_TEMP=0.6 GEN_REP=1.03);stops {'max_tokens'… |
| 1235 | 3.0 | 2.0 | 是 | c25 | 0.0(limit 2)。用途不是打分而是确认 final_model 能被官方 evaluate.py 加载并跑通;a… |
| 1417 | 3.0 | 30.0 | 否 | c26 | 0.0(30 题全量);stops {'stop': 21, 'max_tokens': 9}。机械层记的「没拿到分数」… |
| 1514 | 3.0 | 1.0 | 是 | c27 | 0.0(limit 1)。用途是确认删掉 checkpoint 子目录之后 final_model 仍能被官方 eval… |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[146, 624, 646]
- **1 次验证没有拿到信号**:i=[1417]
- **分类学缺口提案 1 条**
  - verifier-throughput-config(i=793, i=794, i=795)
- **定义缺陷 4 条**
  - (i=193, i=213, i=1499)
  - (i=288, i=304, i=305)
  - (i=1419, i=1495, i=1497)
  - (i=135, i=138, i=146)
- **边界情形 2 条**
  - 8 行训练里有 3 行按现定义无法归入 C3/C4/both,而原因不是证据不足:i=123 是冒烟(8 行 dummy 数据),i=135 是吞吐标定(真实数据真实配方,但目的是量 sec/example,产物随即被删),i=146 是基线首训(没有可比的上一次)。spec §10 第 1 条提议补 smoke 与 baseline 两个取值,但 i=135 两个都装不下——它用真实数据跑真实配…(i=123, i=135, i=146)
  - i=624(bs1/ga16 → bs4/ga4)与 i=646(bs4/ga4 → bs2/ga8)动的都是 C4 主表点名的旋钮(batch),数据与 lr 逐字未变,而有效 batch 每次都被刻意锁定在 16;动机分别是 GPU 利用率(26%)与 OOM 规避。按 reference §3 C8 的说明该归 C8 且不进受测变量,按 C4 的字面定义又该记 C4,两者没有优先级规则。这是…(i=624, i=646, i=644)

## claude_non_api_claude-opus-4-8_10h_run2__aime2025_HuggingFaceTB_SmolLM3-3B-Base_17311665
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | HuggingFaceTB_SmolLM3-3B-Base | 10.08h | — |

### 改动序列(7 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 73 | proposed:agent_memory_no… | 把已经用第一档静态检查确认的任务事实(评测命令口径、scorer 取最后一个数、smollm.jinja 的默认 system 块行为、HF_HOME 里已有的数据集/模型清单、包版本)写进 agent 自己的持久记忆文件。不触碰任何训练或评测产物,C1–C11 都装不下。 | i=71, i=73 |
| 102 | C3 | 写 prep_data.py 定下 v1 训练数据配方:jonathanyin/aime_1983_2023_deepseek-r1_traces_32768 里 Correct 且 Token Count<=15000 的 763 条(AIME 同分布),加 open-r1/OpenR1-Math… | i=102, i=102, i=107 |
| 102 | C2 | 同一次写入里做格式对齐:user 侧逐字复制 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE,assistant 侧统一成 <think>…</think> + 解答 + 末行 ANSWER: {answer},渲染走同一份 templates/smoll… | i=102, i=102, i=97 |
| 120 | C4 | 写 train_sft.py 定下训练方法与超参:TRL SFTTrainer 全参微调(非 LoRA)、bf16 + flash_attention_2、packing、assistant_only_loss=True、max_length 16384、cosine + warmup_ratio … | i=120, i=120, i=130 |
| 130 | C5 | 把 --save_steps 从冒烟的 1000 调到 120,明确意图是让 120/240/360/480 四个 checkpoint 都落盘以便事后挑一个(epoch-1 vs epoch-2)。结果被 c3 里的 save_total_limit=2 抹掉:训练结束时只剩 checkpoint… | i=130, i=163, i=214 |
| 169 | C3 | 写 prep_data_v2.py:v2 配方把 OpenR1 收紧成只保留 \boxed 答案是干净整数的题(11000 条),AIME 历史 763 条改为 x3 直接拼接,共 13289 条 sft_data_v2.jsonl。数据在 i=173/176 已生成,但轨迹结束前从未被任何训练使用… | i=169, i=169, i=176 |
| 220 | C1 | 用 heredoc 整份重写 sft_v1/generation_config.json,加 temperature 0.6 与 top_p 0.95(理由是 R1 蒸馏模型的推荐采样值)。字段级差异是纯新增两项:原有 _from_model_config / eos_token_id [12800… | i=217, i=220, i=220 |

### 训练序列(2 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 124 | smoke | 0.02h | returned | **smoke** | baseline(本 run 第一次训练)。这是冒烟:--max_steps 3 + --accum 2,只验管线能不能跑(packing 与 assistant_only_loss 掩码是否生效、显存够不够、能不能落盘),不测任何 C3/C4 取值。填 unclear 是因为现取值域没有 smok… |
| 130 | real | 2.71h | consumed | **baseline** | 本 run 唯一一次真实训练,之前只有冒烟,没有可比的上一次真实训练,所以它就是基线。相对冒烟只放开了规模旋钮(--max_steps 3 → 2 epochs、--accum 2 → 16、--save_steps 1000 → 120),数据文件(sft_data.jsonl)与脚本(train… |

### 验证序列(2 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 76 | — | — | 是 |  | 0.0 |
| 220 | — | — | 否 | c1, c2, c3, c6 | 未拿到 |

### 异常与存疑

- **1 次验证没有拿到信号**:i=[220]
- **分类学缺口提案 1 条**
  - agent_memory_note(i=71, i=73)
- **定义缺陷 4 条**
  - (i=102, i=102)
  - (i=73, i=71)
  - (i=5, i=213, i=233)
  - (i=216, i=217)
- **边界情形 2 条**
  - 本 run 两行 trainings 都落在 tested_variable 现取值域之外,导致 unclear 率 100%,但没有一行是「证据不足」:i=124 是冒烟,i=130 是首次(也是唯一)真实训练即基线。两者都不在「这次在验 C3 还是 C4」的语义里。这与 spec §10 第 1 条(给 tested_variable 补 smoke / baseline 两个取值)完全吻合,…(i=124, i=130)
  - C4 与 C5 的交叉,按现定义判不了归属:--save_steps 120 是纯 C5 意图(让 120/240/360/480 都落盘以便挑 checkpoint),而同一份训练脚本里的 save_total_limit=2 是 C4 侧的存盘策略,它把中间 checkpoint 删到只剩 480/488,C5 的候选集因此为空,agent 想评的 epoch-1 checkpoint 不存在…(i=163, i=120, i=214)

## claude_non_api_claude-opus-4-8_10h_run2__aime2025_Qwen_Qwen3-1.7B-Base_17311180
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | Qwen_Qwen3-1.7B-Base | 10.08h | 0.0 |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 57 | C3 | 初始数据源选 open-r1/OpenR1-Math-220k:按 correctness_math_verify 过滤、只留整数答案、取最短正确 generation。6 分钟后因发现本地缓存里有更好的 MoT 而作废(i=130 kill)。 | i=57, i=59 |
| 76 | C4 | 训练方法基线:Qwen3-1.7B-Base 全参 SFT(非 LoRA),completion-only 损失掩码(prompt 段 label=-100),bf16 + flash_attention_2 + gradient checkpointing,cosine + warmup 0.03… | i=76, i=75 |
| 84 | C8 | 删掉 group_by_length 的 lengths 预计算分支(Trainer 并不消费 ds.length),避免多一遍无用的长度估算。 | i=84 |
| 138 | C3 | 换数据主干:Mixture-of-Thoughts(math) 22k 条为底料 + AIME'83–'23 DeepSeek-R1 轨迹按 AIME_REPEAT=4 上采样,共 25,176 行(i=143)。这是 sft_v1 实际训的配方。 | i=138, i=129, i=143 |
| 138 | C10 | 去污染守卫:载入 math-ai/aime25 test 仅用于排除,规范化后完全命中或词重叠 >0.75(且长度 >12 词)即丢弃;实际拦掉 5 条(i=143)。选数据集时也刻意挑了只覆盖 1983–2023 的 AIME 轨迹。 | i=138, i=114, i=143 |
| 138 | C2 | 格式对齐:训练样本用评测的原话提示词模板渲染(含 'ANSWER: $ANSWER' 那两句),assistant 以 'ANSWER: X' 收尾;动机是 match 打分器取 completion 里最后一个数字。 | i=138, i=46 |
| 152 | C8 | 给 train_sft.py 加 MAX_STEPS 环境开关并把 logging_steps 5→2,目的是先跑 10 步吞吐探针再决定要不要提交整块预算。 | i=152, i=151 |
| 192 | C8 | 把 gradient checkpointing 从硬编码改成 GC 环境变量可关(USE_GC),准备测无 GC 时的吞吐/显存权衡。 | i=192, i=191 |
| 194 | C5 | 打开周期性存档:save_strategy 'no'→'steps',加 SAVE_STEPS / SAVE_LIMIT 环境变量,为中途 checkpoint 留候选。实际只产出了 sft_v2/checkpoint-856 与 checkpoint-1712,全程没有对任何中途 checkpoi… | i=194, i=191 |
| 289 | C8 | OOM 后重做数据以适配显存:MOT_MAX_TOK 9000 / AIME_MAX_TOK 14000→9400(过滤而非截断),行数 25,176→25,012;同时 rm -rf 掉崩掉的 sft_v1 与 tmp_tput。 | i=289, i=288 |
| 295 | C8 | OOM 的超参补偿:BS 4→2、GA 8→16(有效 batch 32 不变)、MAXLEN 15000→10240、加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments、SAVE_STEPS 350→200。全部为让 fp32 logits 装得下,不是在验… | i=295, i=288 |
| 540 | C11 | 验证器工装:直接读 inspect_ai 落盘日志(logs/*_aime2025_*.json),抽 stop_reason、completion 长度与 target/score —— 官方评测输出被转成确定性判据,而不是另建代理评分器。 | i=540, i=543 |
| 558 | C1 | 整份重写 sft_v1/generation_config.json。字段级差异(对照 i=557 的原文):删掉 max_new_tokens:2048(每个样本恰好生成 2048 token 的根因)、eos_token_id 151643 → [151643,151645](收 <\|im_e… | i=553, i=553, i=558, i=557 |
| 684 | C3 | run-2 数据配方:MOT_N 22000→24000、AIME_REPEAT 4→5、MOT_MAX_TOK 9000→6500、AIME_MAX_TOK 9400→7500,行数 25,012→27,380。假设是更短、更能收尾的轨迹会降低截断率。 | i=684, i=684, i=707 |
| 760 | C11 | 把 (truncated=max_tokens / produced_ANSWER / correct / start_think + token 长度分布) 固化成每次评测都跑一遍的标准读数;此后每一次评测都同时报分数与这组确定性计数。 | i=760, i=763 |
| 766 | C1 | 单字段加 repetition_penalty=1.1 到 sft_v1/generation_config.json(其余字段逐字不变,见 i=767 打印的完整 dict),并先备份成 generation_config.base.json。 | i=766, i=767 |
| 823 | C1 | 回退 repetition_penalty:cp generation_config.base.json 覆盖回去。理由是 rep 惩罚把截断从 24/30 治到 2/15,但 accuracy 同时归零。 | i=823, i=826 |
| 899 | C1 | 把解码修复固化成 set_genconfig.py(可带 temperature/top_p/top_k 参数),因为 save_pretrained 每存一个新 checkpoint 都会把配置退回带 2048 上限的默认值。 | i=899, i=845 |
| 1017 | C1 | 对 sft_v2 应用同一份解码修复:temperature 0.6 / top_p 0.95 / top_k 20 / eos [151643,151645] / 无 max_new_tokens(i=1018 逐字打印了写入内容)。 | i=1017, i=1018 |
| 1161 | C9 | 提交守卫:在还没做温度实验之前,先 rm -rf final_model && cp -r sft_v2 final_model 并删掉 final_model/checkpoint-*,把已测到的 6.7% 锁成下限。此后对 sft_v2 的 config 改动不再波及 final_model。 | i=1161, i=1164 |
| 1167 | C1 | 温度扫描:把 sft_v2 的 temperature 0.6→1.0(top_p 0.95 / top_k 20 不变,i=1168 打印确认),试图用更高的熵打断退化循环。 | i=1167, i=1168 |
| 1294 | C9 | 定案:temp 1.0 更差,final_model 保持 v2@temp0.6 不动;顺手把 sft_v2 的 config 还原成 0.6,并对 final_model 这个真实产物跑一次全量复验。 | i=1294, i=1293 |
| 1298 | C11 | 用 (truncated, correct) 指纹从最近 6 份官方日志里反查出 v2@temp0.6 那一份,再读被截断样本的尾部,判定截断是逐字重复的退化循环而非推理没写完 —— 这个判定直接决定了停止继续投训练。 | i=1298, i=1303 |

### 训练序列(4 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 154 | real | 0.06h | killed | **baseline** | baseline —— 本 run 第一次训练启动。MAX_STEPS=10 / OUT=tmp_tput 的 10 步吞吐探针,只为量 s/it 与显存占用;i=183 被 pkill,产物 i=289 被 rm -rf,从未评测。既不在验数据也不在验方法,是可行性/吞吐探针。取 unclear … |
| 198 | real | 0.23h | last_seen | **unclear** | 首次真实训练(baseline):在 prep_data2 的 25,176 行上全参 SFT。相对 i=154 探针:去掉 MAX_STEPS、BS 2→4、EPOCHS 1→2、MAXLEN=15000、SAVE_STEPS=350。数据配方与训练方法在这一次同时首次确定,没有可比对象,故无受测… |
| 295 | real | 0.16h | last_seen | **unclear** | i=198 的 OOM 重启,同一个 baseline 配方:BS 4→2、GA 8→16(有效 batch 32 逐字不变)、MAXLEN 15000→10240、加 expandable_segments、SAVE_STEPS 350→200;数据重做为 25,012 行(长度上限 9000/9… |
| 827 | real | 3.01h | last_seen | **C3** | 相对 i=295 只换数据:sft_data.jsonl(25,012 行)→ sft_data_v2.jsonl(27,380 行),MOT_N 22000→24000、AIME_REPEAT 4→5、trace 长度上限 9000/9400→6500/7500。BS=2 / GA=16 / LR… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 30 | 3.0 | 30.0 | 是 |  | 0.0333333 (1/30),base 模型基线;i=66 读回 |
| 512 | 3.0 | 30.0 | 是 | c2, c3, c5, c11 | 0.0333333 (1/30),与 base 持平;i=533 读回。事后(i=553)确认这一读数被 max_new… |
| 561 | 3.0 | 6.0 | 是 | c13 | 0.3333333 (2/6),n=6 局部;i=645 读回;i=651 同时读出 4/6 干净停止、2/6 撞 16… |
| 654 | 3.0 | 30.0 | 是 | c13, c2, c5 | 0.0666666 (2/30);i=756 读回;i=763 同时读出 truncated=24 / produced… |
| 766 | 3.0 | 15.0 | 是 | c16 | 0.0(0/15);i=820 读回;i=824 同时读出 truncated=2 / answered=13 / co… |
| 1017 | 3.0 | 30.0 | 是 | c15, c19 | 0.0666666 (2/30),与 sft_v1 逐位打平;i=1153 读回;i=1158 同时读出 truncat… |
| 1167 | 3.0 | 30.0 | 是 | c21 | 0.0333333 (1/30),劣于 temp 0.6;i=1290 读回 |
| 1294 | 3.0 | 30.0 | 是 | c20, c22 | 0.0666666 (2/30),对 final_model 这个真实产物的复验;i=1396 读回;i=1401 同时… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[198, 295]
- **分类学缺口提案 1 条**
  - budget_guard(i=954, i=967, i=969)
- **定义缺陷 4 条**
  - train_span_crash_as_run_end(i=198, i=285, i=289, i=295)
  - train_span_completed_as_run_end(i=506, i=512, i=1015, i=1017)
  - smoke_rule_misses_throughput_probe(i=154, i=155, i=183)
  - config_write_via_script_read_as_read(i=1017, i=1018, i=1167, i=555, i=899)
- **边界情形 3 条**
  - data_forced_maxlen_compensation(i=827, i=294, i=684)
  - oom_restart_has_no_tested_variable(i=288, i=295)
  - c1a_c1b_inseparable_in_one_write(i=553, i=553)

## claude_non_api_claude-opus-4-8_10h_run2__aime2025_Qwen_Qwen3-4B-Base_17310164
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 7.57h | 0.2333333333… |

### 改动序列(16 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 69 | C3 | 建 SFT 数据源与配方:流式扫 open-r1/OpenR1-Math-220k,只留 correctness_math_verify 命中的生成、答案是 <=4 位整数、整条 <=12000 token,写出 24,000 行 train.jsonl(实际 seen 80,465 / writt… | i=69, i=69, i=69 |
| 69 | C2 | 训练样本与评测 prompt 逐字对齐:USER_PROMPT_TEMPLATE 抄 inspect_evals aime 原文,手工拼 <\|im_start\|>user/assistant 而不调 apply_chat_template,assistant 末尾强制补一行 ANSWER: {a… | i=69, i=69, i=68 |
| 79 | C4 | 训练方法与超参:TRL SFTTrainer 全参微调,bf16 + packing + flash_attention_2 + gradient_checkpointing,cosine/warmup 0.03,默认 lr 1e-5;同时在训练脚本里把 tokenizer 与 model.gene… | i=79, i=79, i=79 |
| 154 | proposed:harness-wake-sc… | 写 wait_train.sh:每 60s 轮询 train_sft1.log 里的 SAVED to 或训练进程消失,用 harness 的后台任务完成通知把自己叫醒;起因是 agent 发现前台 sleep 会被 harness 自动转后台,无法用来给自己计时 | i=154, i=190 |
| 213 | C1 | 第一次给 runs/sft1 打解码配置:跑 finalize.py 逐键写入 eos_token_id=151645、pad_token_id=151643、temperature=0.6、top_p=0.95、top_k=20、do_sample=true(净增 6 个字段);逐键写入所以原有 … | i=213, i=152, i=214 |
| 217 | C1 | 删掉 runs/sft1 generation_config 里 base 带来的 max_new_tokens=2048(评测传 --max-tokens 16000,2048 上限会腰斩长推理);这是本 run 唯一一次字段删除类改动,配置从 8 个字段变 7 个 | i=217, i=216, i=218 |
| 222 | proposed:harness-wake-sc… | 同一套路的 wait_eval.sh:轮询 --json-output-file 是否落盘、evaluate.py 是否还活着;此后 6 次评测的分数全部靠它把 agent 叫醒后再 cat 取回 | i=222, i=222 |
| 258 | C11 | 自写分析脚本读官方 inspect_ai 日志,把官方评分器自己的输出转成可决策信号:stop_reason 分布(截断数)、completion_tokens、correct-among-concluded;此后每次评测都用它,截断数成了比 30 题 accuracy 更稳的判据 | i=258, i=259 |
| 266 | C1 | 加 repetition_penalty=1.05,单字段改动,目标是打断退化重复循环把 16k 截断转成正常收尾(改前 21/30 撞 max_tokens) | i=266, i=265 |
| 306 | C3 | 第二次训练的数据侧改动:训练长度上限 8192→12288(train.py 按 ntok<=max_len 过滤,上限直接改变入选样本,等于放进更长的 R1 轨迹)、样本数 14000→12000;数据文件仍是同一份 train.jsonl | i=306, i=79, i=305 |
| 306 | C4 | 同一次启动里的方法侧改动:epochs 2→1.5(独立改动);bs 2→1 与 accum 8→16 是为在 12288 长度下保住有效 batch 16(2x8=1x16)的被迫补偿,lr 1e-5 不变 | i=306, i=121 |
| 455 | C1 | 给 runs/sft2 一次性写解码配置:pop 掉 max_new_tokens,update eos/pad/temperature 0.6/top_p 0.95/top_k 20/do_sample,并直接带上已被验证过的 repetition_penalty=1.05(没走 finalize… | i=455, i=456 |
| 490 | C1 | repetition_penalty 1.05→1.1,单字段试探,想把更多截断转成收尾 | i=490, i=489 |
| 494 | C9 | 剩 2:49 时先把 runs/sft1 拷成 final_model 并把 repetition_penalty 写回 1.05(runs/sft1 当时正带着实验用的 1.1),不产生任何新权重,只是把已有候选钉进提交位,保证随时有一个已验证的产物 | i=494, i=493 |
| 494 | C8 | 同一条命令顺手 rm -rf 掉 final_model / runs/sft1 / runs/sft2 的全部中途 checkpoint 腾磁盘;sft1 此前存有 checkpoint-800 与 checkpoint-866 且从未被评过,此后 C5 式的 checkpoint 挑选在这条 r… | i=494, i=210 |
| 566 | C1 | temperature 0.6→0.5(repetition_penalty 保持 1.05),最后一次单字段解码试探 | i=566, i=565 |

### 训练序列(3 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 107 | real | 0.08h | discarded | **baseline** | 首次启动,但不是候选训练:timeout 600 的吞吐/显存探针,--max_steps 6 --max_examples 4000,用的正是准备用于 sft1 的配置。跑完 6 步(实测 14.2s/step)后在 i=121 被 rm -rf 丢弃。unclear 的原因是取值域装不下冒烟行,… |
| 121 | real | 3.48h | consumed | **unclear** | baseline —— 第一次真实训练,没有可对照的上一次训练(只有 0% 的未训练 base)。数据配方(c1/c2)与方法超参(c3)在此同时首次确定。结局已核实:866/866 步跑满,日志打出 SAVED to runs/sft1,监视器返回 DONE_SAVED,权重文件时间戳 11:57… |
| 306 | real | 2.98h | consumed | **both** | 对比 sft1 同时改了两侧:数据侧 max_len 8192→12288(train.py 按 ntok<=max_len 过滤,长度上限直接换掉入选样本)、max_examples 14000→12000;方法侧 epochs 2→1.5。bs 2→1 / accum 8→16 是为保住有效 b… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 37 | 3.0 | 30.0 | 是 |  | 0.0 |
| 220 | 3.0 | 30.0 | 是 | c1, c2, c3, c4, c5 | 0.2 |
| 222 | — | — | 是 | c1, c2, c3, c4, c5 | 0.2 |
| 266 | 3.0 | 30.0 | 是 | c6 | 0.23333333333333334 |
| 266 | — | — | 是 | c6 | 0.23333333333333334 |
| 266 | 3.0 | 30.0 | 是 | c6 | 0.23333333333333334 |
| 266 | — | — | 是 | c6 | 0.23333333333333334 |
| 455 | 3.0 | 30.0 | 是 | c7, c8, c9 | 0.13333333333333333 |
| 490 | 3.0 | 30.0 | 是 | c10 | 0.16666666666666666 |
| 490 | — | — | 是 | c10 | 0.16666666666666666 |
| 490 | 3.0 | 30.0 | 是 | c10 | 0.16666666666666666 |
| 490 | — | — | 是 | c10 | 0.16666666666666666 |
| 531 | 3.0 | 30.0 | 是 | c11, c6 | 0.2 |
| 531 | — | — | 是 | c11, c6 | 0.2 |
| 531 | 3.0 | 30.0 | 是 | c11, c6 | 0.2 |
| 531 | — | — | 是 | c11, c6 | 0.2 |
| 566 | 3.0 | 30.0 | 是 | c13 | 0.16666666666666666 |
| 566 | — | — | 是 | c13 | 0.16666666666666666 |
| 566 | 3.0 | 30.0 | 是 | c13 | 0.16666666666666666 |
| 566 | — | — | 是 | c13 | 0.16666666666666666 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[121]
- **分类学缺口提案 1 条**
  - harness-wake-scheduling(i=154, i=222, i=190, i=345)
- **定义缺陷 2 条**
  - (i=213, i=152, i=214)
  - (i=222, i=220)
- **边界情形 3 条**
  - 这条 run 3 行训练里有 2 行只能填 unclear,但两行都不是证据不足:i=107 是 --max_steps 6 的吞吐探针(冒烟),i=121 是第一次真实训练、没有可对照的上一次训练(baseline)。按 spec §4.2 的定义 tested_variable 是「与上一次训练相比在验哪一项」,这两行根本没有 referent。若不区分,这条 run 会报出 67% 的 un…(i=107, i=121)
  - i=306 把 max_len 8192→12288 的同时把 bs 2→1、accum 8→16。后两项是为在更长序列下保住有效 batch 16(2x8=1x16)而做的纯补偿,不是被测的超参;真正独立的方法侧改动只有 epochs 2→1.5。本条按「存在独立的 epochs 改动」判为 both,但若采纳 spec §10 第 2 条建议的判据(补偿性超参不计入受测变量),这次训练就应判 …(i=306, i=121)
  - reference §1 把改动的机械字段定义为「触发它的命令、时刻」,spec §4.1 用 (run_id, i) 做主键,都隐含一次命令对应一类改动。这条 run 有三处一个事件承载两类:i=494 一条 shell 里既做 C9(cp -r runs/sft1 final_model 并写回 rep 1.05)又做 C8(rm -rf 全部中途 checkpoint 腾磁盘,顺带永久消灭了…(i=494, i=494, i=69, i=306)

## claude_non_api_claude-opus-4-8_10h_run2__aime2025_google_gemma-3-4b-pt_17310159
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | aime2025 | google_gemma-3-4b-pt | 7.64h | 0.0 |

### 改动序列(29 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 77 | C3 | v1 数据配方:jonathanyin/aime_1983_2023 的 DeepSeek-R1 轨迹(全部标 Correct,850 条)+ OpenR1-Math-220k(amc_aime 上限 3000、其余上限 4000,只取 correctness_math_verify 通过的那条 g… | i=77, i=77, i=77, i=96 |
| 77 | C2 | 格式对齐:训练样本的 user 侧用 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE 原文渲染,prompt 手工拼成 '<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n'(即… | i=77, i=77, i=249, i=251 |
| 103 | C4 | v1 训练方法与超参:全参 SFT(Gemma3ForCausalLM,非 LoRA),completion-only 损失(prompt 段 labels 置 -100),bf16 + gradient_checkpointing + optim='adamw_8bit',lr 1e-5 / co… | i=103, i=103, i=103, i=117 |
| 103 | C1 | 终止符修复(写在训练脚本的保存段):保存前把 generation_config.eos_token_id 设成 [1, 106],即在 <eos> 之外补上 <end_of_turn>(id 106,在 i=102 查得),否则 base 模型不会在 gemma 模板的回合结束处停。这是本 run… | i=103, i=102, i=174 |
| 103 | C8 | 可行性修复:base ckpt 的 architectures 是多模态 Gemma3ForConditionalGeneration(i=61),用 Gemma3ForCausalLM 加载后 config.architectures 变成 None(i=66),vLLM 会认不出。保存前显式写死… | i=103, i=61, i=66, i=174 |
| 109 | C8 | 吞吐优化:sed 把 train.py 的 attn_implementation 从 'eager' 改成 'flash_attention_2'(先在 i=106 确认 flash_attn 2.8.3 已装)。不改变训练目标或任何有效超参,只为在 10h 预算内跑完。归到 C8 有争议,见 b… | i=109, i=107, i=105 |
| 150 | C10 | 合规/防污染的判据选择:把解码调参的 dev 集定在 AIME 2024(Maxwell-Jia/AIME_2024),明确为了不在 AIME 2025 测试集上调参。dev_eval.py 的 --dataset 默认值即 aime2024。期望效应为零或负 —— 它决定分数有效,不决定分数高低。… | i=149, i=150, i=459 |
| 150 | C7 | 自建代理验证器 dev_eval.py:直接用 vLLM 的 python API 跑,自己实现答案抽取(去 \boxed、取最后一行 ANSWER: 后的数字),自己设 stop_token_ids=[1,106],可传任意 temperature/top_p/top_k。口径上复刻了官方 pro… | i=150, i=150, i=150, i=185 |
| 194 | C1 | out_sft_v1/generation_config.json 字段级差异(机械层拿不到内容,此处从命令读出):这是 json.load → 改键 → json.dump 的增量写,不是整份重写,没有任何字段被删。相对 i=174 读到的原状态 {bos_token_id:2, cache_im… | i=194, i=194, i=195, i=174 |
| 204 | C11 | 验证器工装:写脚本读官方 inspect_ai 日志(logs/*.json)统计每题的 stop_reason 分布,并 dump 一条 completion 的尾部。这是把官方评测器**自己的输出**转成可决策信号(截断率 / 停止原因),零 GPU、秒级、确定性。它直接给出了本 run 的核心… | i=204, i=205, i=221, i=207 |
| 214 | C1 | out_sft_v1/generation_config.json 第二次改动,同样是 json.load + c.update 的增量写,无字段删除。净差异两项:temperature 0.6→0.7,新增 repetition_penalty=1.3。目的是做一次判决性测试 —— 若输出形态大变… | i=214, i=214, i=213 |
| 225 | C7 | 自建 sweep_decode.py:一次 vLLM 加载内扫多组解码配置(temperature/top_p/top_k/repetition_penalty 五组),每组报 ACC、finish 率、长度分位。把 C1 的搜索从「一次改 config + 一次官方评测」压成一次模型加载。 | i=225, i=225, i=232 |
| 252 | C3 | v2 数据配方:长度上限从 11000 收到 MAXLEN=8000(直接针对 i=243 观察到的 40% 撞 16k 上限),OpenR1 配额提到 MAX_AMC=3500 / MAX_OTHER=7000,'other' 侧加整数答案过滤 is_clean_int。实得 6489 行(aim… | i=252, i=252, i=257 |
| 260 | C3 | 配方内的重加权:把 src∈{aime,amc} 的行整体复制一份追加(2× 上采样),6489 → 7776 行,竞赛数学占比从 20% 提到 33%,平均长度 4140 tok。产物 sft_data_v2_up.jsonl 才是实际喂给训练的文件。 | i=260, i=260, i=261 |
| 264 | C5 | 开启 checkpoint 候选生成:train.py 的 save_strategy 从 'no' 改成 'epoch',加 save_only_model=True、save_total_limit=4,目的是「so I can pick the best」。注意:本 run 最终**从未评测过… | i=264, i=264, i=307, i=427 |
| 277 | C4 | v2 训练的 epoch 数 3→2。动机是预算而非假设检验:i=276 明说 3 epoch 的 ETA ~4h 不够留缓冲,2 epoch 的 LR 调度也完整。数据文件与 lr/accum 逐字不变,是本 run 唯一一次严格单超参改动(相对被作废的 i=267)。 | i=277, i=276, i=267 |
| 298 | C1 | out_sft_v1/generation_config.json 第三次改动,仍是 json.load + c.update 增量写,无字段删除。净差异两项:temperature 0.7→0.6,repetition_penalty 1.3→1.05(1.3 在 i=221 被证明会退化成词表噪… | i=298, i=298, i=299, i=224 |
| 298 | C9 | 提交守卫(第一次):在 v2 还在训练、GPU 被占的时刻,把已有候选 out_sft_v1 整目录复制成 final_model,理由是「so a valid submission always exists」。不产生任何新产物,只决定此刻 final_model 里放谁。 | i=298, i=297, i=299 |
| 310 | C7 | 验证器配置收窄:根据 v1 的扫描结果把 sweep_decode.py 的候选网格从 5 组改成 4 组,去掉已被证伪的高温/高惩罚组合(1.0/1.1、0.9/1.15),集中在 temperature 0.6–0.8 × repetition_penalty 1.0–1.1。 | i=310, i=310, i=318 |
| 321 | C1 | out_sft_v2/generation_config.json:同样是 json.load + c.update 增量写,无字段删除。相对 trainer 保存的原状态,净差异两项:新增 temperature=0.7、新增 repetition_penalty=1.1;eos_token_id… | i=321, i=321, i=322 |
| 334 | C7 | 换一个有信号的独立测试集:给 sweep_decode.py 加 'mathdev' 数据源 —— EleutherAI/hendrycks_math 七个 subject 的 test split,只保留 Level 5 且 \boxed 答案是 1–4 位整数的题,固定 seed 7 打乱。动机… | i=334, i=334, i=328, i=344 |
| 337 | C7 | 验证器网格改向:把温度扫描区间整体下移(0.3/0.5/0.7 @ rep 1.1,外加 0.6 @ rep 1.05),放弃 0.8–1.0 的高温区。这是对 C1 搜索空间的重新定位,不改任何产物。 | i=337, i=337, i=344 |
| 347 | C7 | 自建 est_aime.py:在**官方测试集 math-ai/aime25 全 30 题**上做 n=8 重复采样,报 pass@1(avg)、pass@k(any)、majority@k、finish 率与逐题命中比例。口径上用官方 prompt 模板与 gemma 模板,但打分是重写的。它比官… | i=347, i=347, i=347, i=354 |
| 363 | C3 | v3 数据大转向(本 run 最大的一次改动):主体换成 nvidia/OpenMathInstruct-2 的**简短解**(problem_source='math' 6000 条 + 'augmented_math' 2000 条,要求 expected_answer 是整数且 solutio… | i=363, i=363, i=363, i=368, i=362 |
| 371 | C4 | 给 train.py 加 --bs 超参并把 per_device_train_batch_size 从写死的 1 改成 args.bs,v3 首次启动用 --bs 4 --accum 4。**有效 batch 仍是 16**(4×4 = 1×16),即这是一次为吞吐做的补偿性改动,不是在验 bat… | i=371, i=373, i=375, i=370 |
| 385 | C8 | OOM 修复:i=375 的 bs=4 在 group_by_length 把最长序列聚到一起时爆显存(要 27.34 GiB),降到 --bs 2 --accum 8,有效 batch 仍保持 16。诊断正确(i=383 归因到 262k 词表的 logits),这是 C8 定义里点名的「OOM … | i=385, i=382, i=384 |
| 419 | C7 | 验证器网格再下探:sed 把 mathdev 扫描的最低温从 0.3 改成 0.2,在 v3 上把 limit 从 80 提到 100 题以降低网格比较的噪声。 | i=419, i=419, i=424 |
| 427 | C1 | out_sft_v3/generation_config.json:json.load + c.update 增量写,无字段删除。净差异两项:新增 temperature=0.6、新增 repetition_penalty=1.1;eos_token_id 重写成 [1,106](空操作)。取值不是… | i=427, i=427, i=446, i=424 |
| 427 | C9 | 提交守卫(第二次,最终):无条件 rm -rf final_model 后把 out_sft_v3 整目录复制进去,并 rm -rf final_model/checkpoint-* 清掉中途存档。此时手上有 v1(官方 1/30)与 v3(官方尚未评),它选了 v3;i=437–444 事后补了 … | i=427, i=427, i=426, i=444 |

### 训练序列(8 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 111 | smoke | 0.02h | returned | **smoke** | baseline —— 本 run 第一次训练启动,是冒烟:sft_data.jsonl 的前 40 行,--epochs 2 --accum 4,前台带 timeout 420。它在验代码跑不跑得通(flash_attention_2 是否可用、completion-only 掩码是否成立、能否落… |
| 117 | real | 1.54h | consumed | **baseline** | baseline —— 本 run 第一次真实训练,没有可比的前一次(前一行是冒烟)。它同时确立了数据配方(c1/c2)和方法超参(c3/c4)的初值,两者都不是相对某个前值的单变量改动。`unclear` 同样是 schema 装不下(需要 `baseline` 取值),不是证据不足。 |
| 267 | real | 0.26h | superseded | **both** | 相对 i=117(上一次真实训练):数据换成 sft_data_v2_up.jsonl —— 长度上限 11000→8000、OpenR1 配额提高、AIME+AMC 2× 上采样、5061→7776 行、平均长度 4887→4140 tok(C3,c12+c13);同时 epochs 2→3(C4… |
| 277 | — | — | — | **unclear** | 这一行不对应任何一次训练。命令是 `kill 7632; sleep 3; pkill …; sleep 2` + `nvidia-smi` + `nohup python train.py … &` 的复合串,shell 在 pkill 处以 **exit code 144** 退出,后面的 no… |
| 286 | real | 2.04h | consumed | **both** | 相对 i=117(上一次真正跑完的训练):数据配方(c12+c13)与 epochs(c15)同时变,故 both。补充口径说明:若与被作废的 i=267 相比,则**只有 epochs 3→2 一项变**(数据文件、lr、accum 逐字相同),而且这一项是预算驱动的补偿而非假设检验 —— 按 s… |
| 375 | real | 0.02h | superseded | **both** | 相对 i=286:数据整体换成 sft_data_v3.jsonl(OpenMathInstruct-2 简短解为主,平均长度 4140→1357 tok,C3/c23),同时 bs 1→4、accum 16→4(C4/c24)。故 both。但 bs/accum 的乘积仍是 16,**有效 bat… |
| 385 | — | — | — | **unclear** | 这一行同样不对应任何一次训练:与 i=277 一模一样的失效形态 —— `pkill -f …; sleep 5; nvidia-smi` + `nohup python train.py … &` 复合串在 pkill 处以 exit code 144 退出,nohup 未执行。没有进程就没有受测… |
| 392 | real | 1.24h | consumed | **both** | 相对 i=286(上一次真正跑完的训练):数据配方 v3(C3/c23)与 bs/accum(C4/c24 + C8/c25)同时变,故 both。相对 i=375 则只差 bs 4→2、accum 4→8,而这一项纯粹由 OOM 逼出、且有效 batch 恒为 16,按 spec §10.2 提议… |

### 验证序列(5 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 69 | 3.0 | 8.0 | 是 |  | 0.0 |
| 196 | 3.0 | 30.0 | 是 | c1, c2, c3, c4, c9 | 0.033 |
| 214 | 3.0 | 6.0 | 是 | c10 | 0.0 |
| 321 | 3.0 | 30.0 | 是 | c12, c13, c15, c19 | 0.0 |
| 427 | 3.0 | 30.0 | 是 | c23, c24, c25, c27, c28 | 0.0 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[277, 385]
- **分类学缺口提案 1 条**
  - noise-resolving-replication(i=347, i=347, i=354, i=416, i=442, i=444)
- **定义缺陷 3 条**
  - (i=277, i=280, i=284, i=285, i=386, i=391)
  - (i=347, i=349, i=354, i=434, i=201)
  - (i=196, i=321, i=427, i=7)
- **边界情形 3 条**
  - (i=286, i=276, i=375, i=370, i=392, i=384)
  - (i=347, i=416, i=418, i=205)
  - (i=109, i=105, i=371, i=370)

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
| 199 | real | 0.07h | superseded | **C4** | baseline —— 第一次启动 train_sft.py。20 步冒烟(--subset 3000 --max_steps 20 --grad_accum 4),目的是量 FFD+FA2 padding-free 管线的吞吐(实测 ~14k token/s、4.68 s/step)并确认不 OO… |
| 211 | real | 0.01h | returned | **C4** | vs i=199:--subset 3000→2500、--max_steps 20→8、--grad_accum 4→1,并加 --no_ckpt。唯一被测的变量是**关掉 gradient checkpointing 能否换来吞吐**。结局:第 0 步就 OOM(峰值 80,855 MiB / … |
| 228 | real | 0.59h | killed | **both** | baseline(第一次真实训练)。同一次启动里**同时**固定了数据侧(OpenR1 shortest-correct、≤12288 token、25,159 例、119M token)和方法/超参侧(全参 SFT、2 epoch、lr 1e-5、grad_accum 8、cosine),没有任何… |
| 476 | real | 5.80h | consumed | **both** | vs runA:--max_len 12288→10240、新增 --subset 20000(25,159→20,000 例)、--save_steps 200→250;lr / epochs / grad_accum / 数据来源逐字相同。agent 明写的意图有两条,分属两类:(a) 用更短的… |
| 713 | real | 0.03h | superseded | **C3** | vs runM:--data data_sft_tokenized→data_concise(R1 长 CoT → OpenR1 简短参考解,中位长度 4222→677 token)、--max_len 10240→4096、--epochs 2→3、去掉 --subset。agent 声明的假设只… |
| 726 | real | 0.72h | last_seen | **C3** | 与 i=713 的训练命令逐字相同(--data data_concise --max_len 4096 --epochs 3 --lr 1e-5 --grad_accum 8 --save_steps 200),只是在 i=723 确认 GPU 已空(procs train=0 eval=0)后重… |
| 805 | real | 1.91h | consumed | **C3** | vs sft_concise:--data data_concise→data_capped(同一批 OpenR1 shortest-correct 轨迹,但 reasoning >5000 token 的被截断并追加强制收尾句 + 正确答案,占 32%)、--max_len 4096→8192、-… |
| 980 | smoke | 0.03h | discarded | **C4** | 第一次 RL 冒烟:训练方法从 SFT 换成 GRPO(num_gen 8 / prompt_bs 8 / grad_accum 2 / max_completion 8192 / lr 1e-6 / beta 0 / max_steps 3 / subset 200),起点是 sft_capped… |
| 988 | smoke | 0.05h | superseded | **C4** | 与 i=980 的 GRPO 参数逐字相同;唯一变化在模型目录 —— i=986 把 smollm.jinja 写进了 sft_capped/tokenizer_config.json。结局:过了模板这关,但在把 [batch×seq×128256] logits 转 fp32 时 OOM(单次要 … |
| 997 | smoke | 0.06h | discarded | **C4** | vs i=988:--prompt_bs 8→2、--grad_accum 2→4、--max_completion 8192→7168、--max_steps 3→2,纯粹为压 logits 显存。结局:跑完 1 步(51.5 s)后再次 OOM;而且那一步的诊断显示 completions/cl… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 153 | 3.0 | 30.0 | 是 |  | 0.03333333333333333 (1/30) —— base 模型原样的基线,不判定任何改动 |
| 440 | 3.0 | 30.0 | 是 | c1, c2, c3, c4, c5, c7 | 0.13333333333333333 (4/30);同一份日志经 i=466 的自写脚本读出 18/30 闭合、12/… |
| 655 | 3.0 | 30.0 | 是 | c10, c5 | 0.0333 (1/30);16/30 闭合、14/30(47%)截断、中位 15860 token —— 比只训 0.… |
| 669 | 3.0 | 30.0 | 是 | c16, c10 | checkpoint-250 = 0.1000 (3/30, 50% 截断);checkpoint-500 = 0.13… |
| 732 | 3.0 | 30.0 | 是 | c11 | 0.0000 (0/30);30/30 全部闭合、0% 截断、输出中位 730 token |
| 811 | 3.0 | 30.0 | 是 | c12 | 0.1333 (4/30);29/30 闭合、1/30(3%)截断、中位 9505 token |
| 876 | — | — | — | c12, c17 | 0.06666666666666667 (2/30)。注意这是**全量 30 题**、evaluate.py 默认参数(… |
| 912 | 3.0 | 4.0 | 是 | c8, c12, c17 | 只拿到第 1 次:0.13333333333333333。计划 4 次,agent 在 i=949/953 杀掉了整个比… |
| 912 | 3.0 | 3.0 | 是 | c8, c12, c17 | 只拿到第 1 次:0.13333333333333333。计划 4 次,agent 在 i=949/953 杀掉了整个比… |
| 912 | 3.0 | 4.0 | 是 | c8, c15 | 未拿到 —— `eval_avg.sh sft_runM/checkpoint-500 3 30 16 r1500_av… |
| 912 | 3.0 | 3.0 | 是 | c8, c15 | 未拿到 —— `eval_avg.sh sft_runM/checkpoint-500 3 30 16 r1500_av… |
| 1020 | 3.0 | 30.0 | 否 | c8, c12, c17 | 拿到了,但不在启动事件的输出里:cap_extra1 = 0.16666666666666666(i=1057 打印),… |
| 1020 | 3.0 | 30.0 | 否 | c8, c12, c17 | 拿到了,但不在启动事件的输出里:cap_extra1 = 0.16666666666666666(i=1057 打印),… |
| 1020 | 3.0 | 30.0 | 否 | c8, c15 | 未拿到 —— r1_extra1 跑到 10/30 时被 kill(i=1110 显示进程 19495 = evalua… |
| 1020 | 3.0 | 30.0 | 否 | c8, c15 | 未拿到 —— r1_extra1 跑到 10/30 时被 kill(i=1110 显示进程 19495 = evalua… |
| 1142 | — | — | 是 | c12, c17, c3 | 0.13333333333333333 (4/30);27/30 闭合、3/30(10%)截断、中位 10115 tok… |
| 1175 | 3.0 | 30.0 | 否 | c8, c15 | 未拿到 —— r1cmp1 跑到 8/30 被 kill(i=1196 「r1cmp1 prog: 8/30」,i=12… |
| 1217 | 3.0 | 30.0 | 是 | c12, c17 | 0.1 (3/30);25/30 闭合、5/30(17%)截断、中位 11218 token |

### 异常与存疑

- **6 次验证没有拿到信号**:i=[876, 1020, 1020, 1020, 1020, 1175]
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
| 274 | smoke | 0.09h | consumed | **smoke** | 相对 i=264:恢复 --grad_ckpt 1、bs8/accum2、2500 条样本、save_strategy epoch。但这次训练本身不在验数据配方也不在验超参 —— 它只是为了产出一个 checkpoint,好去验 C1(eos 能不能让 vLLM 停)与 C2(输出里有没有 ANSW… |
| 467 | real | 3.42h | consumed | **C4** | 相对被作废的 i=226:唯一实质变化是 epochs 3→2(因为 3 epoch 实测 2307 步 × 8.5s ≈ 5.4h,压不进预算),数据/lr/bs/accum/maxlen/liger/grad_ckpt 全同,save_steps 250→256。这是 run1 的正式训练,tr… |
| 793 | — | — | — | **unclear** | 这一行没有对应任何真实训练:复合命令的首条 pkill 没匹配到进程返回 1,整条命令中断,train_sft.py 从未执行(tool_result 只有 Exit code 1;紧接着的检查显示 train_run2.log 为空、GPU 0 MiB;agent 自己写下 Run2 didn't… |
| 802 | real | 0.06h | killed | **C3** | 相对 run1(i=467):数据从 sft_openr1.jsonl(31,723 × 2 epoch)换成 sft_mix_clean.jsonl 取 62,000(× 1 epoch);lr 1e-5 / bs8 / accum2 / maxlen12288 / liger / grad_ck… |
| 845 | real | 3.25h | consumed | **C3** | 相对 i=802:只把 max_samples 62000→48000、save_steps 400→350,纯粹为压进剩余墙钟;数据文件与全部超参不变。相对 run1 仍是同一个受测变量(OpenR1+OMR 混合 vs 纯 OpenR1)。这是 run2 的正式训练,1377 步 / 3:11:… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 117 | 3.0 | 8.0 | 是 |  | 0.000(limit 8;base 模型基线,同时用来验通评测管线) |
| 333 | 3.0 | 8.0 | 是 |  | 未拿到 —— 这条命令里的 evaluate.py 从未执行(见 definition_defect d4) |
| 416 | 3.0 | 8.0 | 是 | c2, c3, c6, c7 | 0.0(0/8);真正读回去的信息是格式与停止:6/8 stop=max_tokens、id=2 走完 ANSWER: … |
| 672 | 3.0 | 30.0 | 是 | c1, c2, c4, c6 | 0.03333333333333333(1/30);27/30 撞 max_tokens、29/30 缺 ANSWER |
| 717 | 3.0 | 15.0 | 是 | c12 | 0.06666666666666667(1/15);10/15 改为正常结束,循环被打断 |
| 751 | 3.0 | 30.0 | 是 | c12 | 0.0(0/30);16/30 缺 ANSWER —— 与同配置 limit 15 的 1/15 冲突,agent 据此… |
| 910 | 3.0 | 30.0 | 是 | c9, c10, c11, c14 | 0.03333333333333333(1/30);18/30 缺 ANSWER,贪婪比采样循环更狠 |
| 958 | 3.0 | 30.0 | 是 | c15 | 0.0(0/30);截断降到 11/30 但连原本对的 id=0 也做错,判定 freq 0.4 伤推理 |
| 986 | 3.0 | 30.0 | 是 | c16, c17 | 0.06666666666666667(2/30,id 0 与 id 16);同口径胜过 run2 的 1/30 |
| 1031 | 3.0 | 30.0 | 是 | c19 | 0.03333333333333333(1/30,id 13);missing_ANSWER 30/30,判定 rep … |
| 1063 | 3.0 | 30.0 | 是 | c13, c18 | 0.06666666666666667(2/30,id 0 与 id 16);对 final_model 产物本身的复现… |
| 1098 | 3.0 | 30.0 | 是 | c20 | 0.03333333333333333(1/30,只剩 id 0);判定任何 frequency_penalty 都伤推… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[793]
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

## claude_non_api_claude-opus-5_10h_run1__aime2025_Qwen_Qwen3-4B-Base_17415288
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | aime2025 | Qwen_Qwen3-4B-Base | 8.91h | 0.1666666666… |

### 改动序列(16 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 391 | C2 | prep_data.py 把 inspect_evals/aime2025 的 USER_PROMPT_TEMPLATE 逐字复制进训练样本,并把每条 completion 统一收尾成 `<think>…</think>\n\n{solution}\n\nANSWER: {answer}`,让训练目… | i=391, i=45 |
| 391 | C3 | 确定训练数据来源:open-r1/OpenR1-Math-220k(只留 correctness_math_verify 通过的)、nvidia/OpenMathReasoning cot 分片(边跑边下到 30 片)、jonathanyin AIME 1983–2023 的 R1 轨迹(只留 Co… | i=391, i=324 |
| 454 | C4 | train_sft.py 定下训练方法:全参 SFT、bf16、flash_attention_2、BFD packing 到 16384、padding_free、completion_only_loss、gradient checkpointing、adamw_bnb_8bit、cosine +… | i=454, i=454 |
| 533 | C3 | mix.py 定义配方权重:按子来源打分(amc_aime 3.0 / aops_forum 2.0 / …)、按 pass_rate_72b_tir 偏好难而可解、按 trace 长度加权,再取前 N 条。 | i=533, i=533 |
| 605 | C4 | 第一次冒烟在 backward 处 OOM(要 18.51 GiB)后,装 liger-kernel 并在 SFTConfig 打开 use_liger_kernel=True(fused linear CE,不物化 152k 词表的 logits),保住 bs=2 × 16384 的配置。 | i=605, i=599 |
| 649 | C2 | 把训练侧 prompt 改成手工拼接 `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n`,与 templates/qwen3.jinja 的 add_generation_prompt=True 渲染逐字节一致(先在 i=64… | i=649, i=648 |
| 729 | C7 | 写 eval_dev.py:自建代理验证器,在 AIME 2024(训练里已按 year<2024 排除,held-out)上用 vLLM 离线推理跑 n=4 采样,除 acc 外还报 mean_len / trunc_rate / fmt_ok_rate —— 后两个指标是这条 run 后续所有决… | i=729, i=729 |
| 961 | C1 | package_model.py 用 json.dump **整份重写** final_model/generation_config.json:do_sample false→true、eos_token_id 由标量 151643 改成 [151645, 151643](接受 <\|im_end… | i=961, i=475, i=961 |
| 1047 | C7 | 写 eval_dev_multi.py:一次 vLLM 加载里连跑多组采样配置,把 C1 的候选评估成本从『每组一次装载』降到『每组一次 generate』。 | i=1047 |
| 1084 | C3 | 改 prep_data.py 的去重规则:同一道题保留**最短**的正确 trace(原来是保留第一条),目的是压低模型的生成长度。池子 87,500 → 97,880 条。 | i=1084, i=1085 |
| 1128 | C3 | 改 mix.py 的长度权重,从偏好 4k–9k token 改成偏好 2k–4k(注释写明理由:模型推理时长度大约翻倍,超过 16k 得 0 分)。mix 的 mean ctoks 由 6183 降到 3764。 | i=1128, i=1134 |
| 1174 | C1 | 自建第三档消融(i=1096)显示 repetition_penalty=1.05 把 acc 0.158→0.225、trunc 0.75→0.50 之后,在 package_model.py 的 generation_config 里加上 repetition_penalty(默认 1.05)。… | i=1174, i=1096 |
| 1176 | C5 | 在 sft2 还在训练时,先把 ckpt/sft1/final 打包成 final_model 作为兜底提交(此时 generation_config 已带 repetition_penalty 1.05)。 | i=1176, i=1177 |
| 1356 | C3 | liger 崩溃损失 50 分钟后,把第二轮的配方规模从 24,000 条(90.3M token)砍到 19,000 条(71.6M token)以塞进剩余时钟,其余选择逻辑不变。 | i=1356, i=1359 |
| 1551 | proposed:budget-forcing | 写 make_wrapup.py:取已有的长 think 轨迹,在 4000–7500 token 处截断,接上一段固定的『我已经想够了,下面写答案』收尾语,再拼回原 solution 与 ANSWER 行,造 1300 条 s1 式 budget-forcing 样本,用来教模型在 16k 预算内… | i=1551, i=1559 |
| 1789 | C5 | 最终候选选择:`rm -rf final_model` 后用 ckpt/sft3/final 重新打包,顶掉 c15 的 sft1 版本。generation_config 逐字段与 c15 相同(temp 0.6 / top_p 0.95 / top_k 20 / rep 1.05),**只换了权… | i=1789, i=1791, i=1788 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 582 | real | 0.02h | returned | **C4** | baseline(本 run 第一次训练启动)。意图是量吞吐:train_mix8k(既有数据)+ --max-steps 6 --bs 2 --accum 4 --max-length 16384。**真实结局:第 0 步 backward 就 CUDA OOM(要 18.51 GiB),0 步完… |
| 607 | real | 0.04h | returned | **C4** | 与 i=582 **单变量**对照:同一份 data/train_mix8k.jsonl、同样 `--max-steps 6 --save-steps 1000 --bs 2 --accum 4`,只加了 use_liger_kernel=True 与 PYTORCH_CUDA_ALLOC_CONF… |
| 672 | real | 1.80h | consumed | **both** | baseline(第一次真实训练)。数据与方法同时首次确定:train_mix10k(10,000 条,62.8M token,mean ctoks 6183)+ 全参 SFT / epochs 1 / lr 1.4e-5 / bs 2 / accum 4 / save-steps 150,从 Qw… |
| 1154 | real | 0.95h | superseded | **C3** | vs sft1 **只换数据**:train_mix_r2(24,000 条、90.3M token、最短-trace 去重、长度权重偏向 2–4k,mean ctoks 3764 vs 6183);`--epochs 1 --lr 1.4e-5 --bs 2 --accum 4` 逐字相同,同样从… |
| 1360 | real | 0.60h | consumed | **C3** | 崩溃后的重启:配方不变,规模 24,000→19,000(71.6M token,mean ctoks 3771),PYTHONPATH 指向自装的 /home/ben/task/liglib,save-steps 200→180。相对 sft1 仍是**超参逐字相同、只换数据**的一臂(数据同时变… |
| 1641 | — | — | — | **both** | 意图与 i=1701 完全相同(见下),但**这次启动从未发生**:同一条命令开头的 `pkill -f eval_dev_multi` 匹配到了 harness 给这条命令自己套的 bash wrapper(wrapper 的命令文本里含 eval_dev_multi),整条命令在 nohup 之… |
| 1701 | real | 0.00h | consumed | **both** | vs sft2:同时动了两侧。数据侧 train_mix_r3 = 2,700 条新的短 trace(1000–4500 token,排除 r2b 用过的题)+ 1,300 条 make_wrapup.py 造的 budget-forcing 样本,合 15.2M token(约为前两轮的 1/5)… |

### 验证序列(2 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 152 | 3.0 | 30.0 | 是 |  | 0.0(0/30)。分数不是从 --json-output-file 指定的 runs/baseline.json 读回… |
| 1792 | 3.0 | 30.0 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c11, c12… | 0.23333333333333334(7/30,stderr 0.0785),i=1799 从 runs/final_… |

### 异常与存疑

- **分类学缺口提案 1 条**
  - budget-forcing(i=1551, i=1631, i=1772, i=1009)
- **定义缺陷 5 条**
  - (i=1409, i=1448, i=1441)
  - (i=1705, i=1710)
  - (i=1666, i=1700, i=1641)
  - (i=763, i=779, i=648, i=1096)
  - (i=337, i=338)
- **边界情形 2 条**
  - (i=1789, i=961)
  - (i=1646, i=1700)

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
| 56 | 3.0 | 10.0 | 是 |  | 0.000 — base 模型基线，--limit 10（真正的第三档）。作用是验管线通不通 + 拿到零点锚点，不判定任… |
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
| 789 | — | — | 是 | c29, c30 | 0.96 — 用官方默认调用（python evaluate.py，无任何 --limit / --model-path… |

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
| 169 | smoke | 0.01h | returned | **smoke** | baseline(冒烟)。128 条样本、1 epoch,只验 train_sft.py 能不能跑通与吞吐率,不验 C3 也不验 C4。见 boundary_case bc3。 |
| 175 | real | 0.04h | superseded | **both** | baseline(首次真实训练)。同时确立数据(prep_data.py 的 25,836 条 xlam 单调用)与方法(全参 SFT,2 epoch,bs 16 ga 4 lr 1e-5),不构成对照。约 1.5 分钟后 OOM 崩溃(大词表 logits @ bs16 × 2048 tok),被… |
| 197 | real | 0.47h | consumed | **C4** | 与 i=175 同数据、同 epochs、同 lr;只改 bs 16->8 与 ga 4->8(有效 batch 64 不变)、开 gradient checkpointing、加 expandable_segments。纯 OOM 修复,不为提分。结局:跑完,27 分钟,i=333 看门狗返回 c… |
| 364 | real | 4.75h | last_seen | **C3** | 与 i=197 的启动命令**逐字只差 --data data/v2_mix.jsonl**(epochs/bs/ga/lr/save-each-epoch 全同,train_sft.py 期间未改)。数据从 25,836 条原始 xlam 换成 38,247 条 v2_mix(三源合并 + 数值清… |
| 462 | real | 0.02h | superseded | **C4** | 训练方法从全参 SFT 换成 DPO(beta 0.1 / lr 7e-7 / sft-alpha 0.2),初始权重 runs/v2/checkpoint-1196。启动约 1 分钟后 OOM(全词表 fp32 log_softmax),被 i=474 取代。 |
| 474 | real | 0.61h | last_seen | **C4** | 与 i=462 同 pairs、同 init、同 beta/lr/epochs/sft-alpha;只改 bs 4->2、ga 8->16(有效 batch 32 不变)。纯 OOM 修复。结局:跑完,启动 21:00:01,i=499 看门狗于 21:10:57 返回 completed,i=50… |
| 531 | real | 0.10h | consumed | **both** | 与上一轮 DPO(i=474)相比同时动了两类:C3——偏好对从 dpo_all.jsonl(6,433 对,挖掘 + canon)换成只用 canon_pairs.jsonl;C4——beta 0.1->0.15、lr 7e-7->1.2e-6、sft-alpha 0.2->0.1,初始权重从 r… |
| 559 | real | 0.75h | consumed | **both** | 与 i=364(v2)相比:C3——data v2_mix -> v4_mix(41,813 条 = v2_mix + 再加一份 hard_v2,难例出现两次);同时首次显式加上 --seed 43(v2 未指定种子)。种子这一维按现有定义只能挂到 C4,但它不改变期望,见 proposed_cat… |
| 758 | real | 1.03h | consumed | **both** | 这是 chain_v6 那轮 DPO 的第三次尝试(前两次分别死于 checkpoint 缺 tokenizer 与 save_pretrained 校验)。与 i=474 的 DPO 相比:C3——偏好对换成 dpo_all_v4.jsonl(从 v4_ep2 重新挖掘 + canon 去重);C… |
| 780 | real | 0.68h | last_seen | **C4** | 与 i=559(v4)的 SFT 调用**逐字只差 --seed 1234(原 43)与输出目录**:同 data v4_mix、同 epochs 2、bs 8、ga 8、lr 1e-5、save-each-epoch。纯种子重抽。结局**不是** run 结尾:i=804 于 00:53:30 返… |
| 817 | real | 1.12h | run_end | **both** | **该行的产物 data/hard_v4dpo.jsonl 是一次 mining(推理)输出,不是权重**;同一条 nohup 链里真正的训练是 --output runs/v4_dpo2,机械层没登记它(见 definition_defect dd3)。就那次训练而言,相对 i=758:C3——偏… |
| 849 | real | 0.68h | last_seen | **C4** | 与 i=559(v4)逐字只差 --seed 777(原 43)与输出目录,data/epochs/bs/ga/lr 全同。第二次种子重抽。结局**不是** run 结尾:i=871 于 01:57:44 打印 saved to runs/v7/final,i=874 于 02:01:53 返回 c… |

### 验证序列(13 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 55 | 3.0 | 20.0 | 是 |  | 0.0 |
| 206 | — | — | 是 |  | 未拿到(该事件根本不是评测) |
| 336 | 4.0 | -1.0 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep2,全量 100 题) |
| 336 | 4.0 | -1.0 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep2,全量 100 题) |
| 336 | 4.0 | -1.0 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep1,全量 100 题) |
| 336 | 4.0 | -1.0 | 否 | c1, c2, c3, c4, c5, c8, c9 | 0.95(v1_ep1,全量 100 题) |
| 547 | 4.0 | -1.0 | 否 | c17, c18, c19, c14 | 0.95(bench)/ 0.8713(本地 575 题) |
| 758 | 4.0 | -1.0 | 否 | c27, c28, c29, c7 | 0.96(bench)/ 0.8765(本地) |
| 780 | 4.0 | -1.0 | 否 | c30 | 0.94(v6_ep2) |
| 780 | 4.0 | -1.0 | 否 | c30 | 0.94(v6_ep2) |
| 780 | 4.0 | -1.0 | 否 | c30 | 0.94(v6_ep1) |
| 780 | 4.0 | -1.0 | 否 | c30 | 0.94(v6_ep1) |
| 807 | — | — | 否 | c31 | 0.96 |
| 817 | 4.0 | -1.0 | 否 | c32 | 0.96(bench)/ 0.8817(本地) |
| 839 | — | — | 否 | c33 | 0.96 |
| 849 | 4.0 | -1.0 | 否 | c34 | 0.95(v7_ep2) |
| 849 | 4.0 | -1.0 | 否 | c34 | 0.95(v7_ep2) |
| 849 | 4.0 | -1.0 | 否 | c34 | 0.94(v7_ep1) |
| 849 | 4.0 | -1.0 | 否 | c34 | 0.94(v7_ep1) |

### 异常与存疑

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

## claude_non_api_max_claude-opus-4-8_10h_run1__bfcl_Qwen_Qwen3-1.7B-Base_17315719
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | bfcl | Qwen_Qwen3-1.7B-Base | 9.54h | 0.92 |

### 改动序列(19 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 150 | C2 | prep_data.py 用评测自己的 templates/qwen3.jinja + add_generation_prompt=True 渲染训练 prompt,completion 固定为 <tool_call>\n{...}\n</tool_call><\|im_end\|>,与评测 pro… | i=150, i=402 |
| 162 | C3 | 训练数据来源:本地缓存的 NewEden/xlam-function-calling-60k-shareGPT(28,461)+ interstellarninja(478)+ gorilla(1),过滤成单次调用、Python 类型转 JSON Schema,共 28,940 条。 | i=162, i=175 |
| 166 | C4 | train.py:全参 SFT(非 LoRA),completion-only loss(prompt 段 label 置 -100),max_len 2048,transformers Trainer。 | i=166, i=166 |
| 166 | C1 | 把保存出来的 config.json / generation_config.json 的 eos_token_id 由 base 的 151643(<\|endoftext\|>)改成 151645(<\|im_end\|>),让 vLLM 在一次 tool_call 后停住。 | i=166, i=126 |
| 208 | C4 | OOM 后把 per-device batch 16→8、grad_accum 2→4(有效 batch 不变 32),并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。 | i=200, i=208 |
| 252 | C4 | 发现 Qwen3-Base 的 chat/tool 特殊 token 是保留但未训练的(嵌入范数 0.37 vs 正常 1.58),把 6 个特殊 token 的嵌入重初始化为已训练 token 的均值向量+噪声。 | i=249, i=252 |
| 262 | C4 | 均值初始化失败(高维均值互相抵消,范数只有 0.381),改为按维度 Gaussian(mean,std) 采样,并把 base 的 <\|endoftext\|> 嵌入直接拷进 <\|im_end\|>。 | i=259, i=262, i=269 |
| 375 | C1 | 在保存后用原始 JSON 改写 generation_config.json 写入贪婪解码:temperature=0.0、top_p=1.0、do_sample=False、删掉 top_k,eos 仍为 151645(json.load→update→json.dump 是合并而非整份重写,ba… | i=374, i=375, i=667 |
| 393 | C3 | 配方改动:把 train_full 里函数名含计算类关键词的 4,760 条复制一份接在原集之后再打乱,得到 train_boost.jsonl(33,700 条,是 train_full 的严格超集)。 | i=393, i=394 |
| 438 | C1 | 写 finalize.py:幂等地把任一模型目录的 config.json/generation_config.json 合并更新成 eos=151645 + 贪婪,并可 copytree 到 final_model。 | i=438, i=438 |
| 506 | proposed:pipeline-plumbi… | 写 train_eval.sh:一条命令串起 train.py → finalize.py → evaluate.py 并打印分数,把后续每次迭代压成单个后台任务。 | i=506, i=506 |
| 537 | C5 | 候选选择:把已测得 0.91 的 model_v1 通过 finalize.py 复制成 final_model,作为带守卫的提交基线(此后只有分数更高才覆盖)。 | i=537, i=536 |
| 551 | C4 | epochs 3→4(model_v2),数据/lr/bs/grad_accum 与 v1 逐字相同。该次训练跑完 4 个 epoch 后在保存时崩溃,权重未落盘。 | i=551, i=655 |
| 660 | proposed:pipeline-plumbi… | 修复保存路径:去掉在内存 GenerationConfig 上设 temperature/do_sample 的两行(transformers 4.57 在 save 时校验拒绝 do_sample=False + temperature=0.0),改为只走保存后的原始 JSON 改写。 | i=660, i=653 |
| 705 | C3 | 写 gen_synth.py 程序化合成 2,900 条新函数(非 BFCL 名)的单次调用样本,针对 v1 失败的并列列表参数映射与最小参数集,拼成 train_synth.jsonl(31,840 条)。 | i=705, i=710 |
| 719 | proposed:pipeline-plumbi… | 写 chain_v4.sh:轮询 /tmp/eval_model_v3.json 出现且 v3 进程退出后自动启动 v4,让 GPU 在两次迭代之间不空转。 | i=719, i=719 |
| 793 | C3 | 写 gen_synth_optional.py 合成约 1,700 条「只有查询里明确提到才带上可选参数」的样本,拼成 train_synth2.jsonl(33,540 条)。 | i=793, i=840 |
| 895 | C4 | epochs 3→4 的干净重跑(model_v6),数据仍是 train_full,lr 2e-5、bs 8、grad_accum 4 与 v1 相同,是 v2 崩溃后补上的单变量对照。 | i=894, i=895 |
| 933 | C5 | 候选选择:v6(0.92)高于当时的 final_model(v1, 0.91),把 model_v6 提升为 final_model,并重跑一次全量评测确认交付物复现 0.92。 | i=933, i=932 |

### 训练序列(11 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 188 | smoke | 0.07h | superseded | **smoke** | baseline:全流程第一次训练。目的是「跑通管线、特别是 eos/停止行为」,既不是在验数据配方也不是在验超参,C3/C4 两个取值都装不下(见 definition_defect)。bs 16 / grad_accum 2 / subset 3000 / 1 epoch / lr 1e-5。第… |
| 208 | smoke | 0.07h | consumed | **C4** | 相对 i=188:唯一改动是 per-device batch 16→8、grad_accum 2→4(有效 batch 仍 32)加 expandable_segments;数据、subset 3000、1 epoch、lr 1e-5 全部不变。验的是「bs=8 能不能不 OOM 跑完」。trai… |
| 256 | smoke | 0.02h | superseded | **C4** | 相对 i=208:同时改了三项——train.py 加入特殊 token 嵌入均值重初始化(c6),subset 3000→5000,epochs 1→2,lr 1e-5→2e-5。启动 30 秒后 agent 从日志看到 new norm ~0.381 判定初始化方案无效。 |
| 264 | smoke | 0.14h | consumed | **C4** | 相对 i=256:命令行逐字相同(subset 5000 / 2 epochs / bs 8 / grad_accum 4 / lr 2e-5),唯一差别是 train.py 里初始化方案从均值改成按维度 Gaussian(mean,std) 且 <\|im_end\|> 拷 base eos。注意… |
| 341 | real | 0.14h | consumed | **C4** | 相对 i=264:去掉 --subset 5000(改用全部 28,930 条,同一个 train_full.jsonl、同一来源同一配方),epochs 2→3;lr 2e-5、bs 8、grad_accum 4、嵌入初始化全部不变。agent 的假设是「更多训练能稳住特殊 token」。注:「同… |
| 551 | smoke | 1.65h | last_seen | **C4** | 相对 i=341(model_v1):唯一变量是 epochs 3→4;数据 train_full.jsonl、lr 2e-5、bs 8、grad_accum 4 均相同(train_eval.sh 内写死 --bs 8 --grad_accum 4)。真实结局:4 个 epoch 全部跑完(362… |
| 664 | smoke | 0.01h | returned | **smoke** | 不是配方或超参实验:subset 64 / 1 epoch 的前台冒烟,唯一目的是确认 c13 的保存路径修复后 save_pretrained 不再抛异常、且 generation_config.json 写对。24 秒返回,exit 0。同 i=188,C3/C4 装不下。 |
| 670 | smoke | 5.84h | run_end | **C3** | 相对 i=341(model_v1):唯一变量是数据文件 train_full.jsonl(28,940)→ train_boost.jsonl(33,700,= train_full 的严格超集 + 4,760 条计算类样本各复制一份后整体打乱);epochs 3、lr 2e-5、bs 8、gra… |
| 721 | real | 5.72h | run_end | **C3** | 相对 i=670(v3):数据回到 train_full 并加上 2,900 条自造合成样本(train_synth.jsonl,31,840);epochs 3、lr 2e-5、bs 8、grad_accum 4 不变。注意 i=721 只是链式脚本启动,真正的 train.py 到 02:37 … |
| 837 | smoke | 3.09h | run_end | **C3** | 相对 i=721(v4):数据再叠加约 1,700 条「条件性可选参数」样本(train_synth2.jsonl,33,540);epochs 3、lr 2e-5、bs 8、grad_accum 4 不变。真实结局:05:19 前出分 0.91,两条 calculate_investment_va… |
| 895 | smoke | 1.65h | run_end | **C4** | 相对 i=341(model_v1):唯一变量 epochs 3→4,数据 train_full.jsonl、lr 2e-5、bs 8、grad_accum 4、评测口径全同——即 v2 崩掉的那次单变量对照的干净重跑。真实结局:07:00 前出分 0.92(v1/v3/v4/v5 全是 0.91)… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 67 | — | — | 是 |  | 未拿到分数:这是用评测自己的 create_tool_info_from_dict + openai_chat_tool… |
| 85 | 3.0 | 25.0 | 是 |  | 0.0(base 模型 Qwen3-1.7B-Base,--limit 25;分数在 i=124 才被读回,通道是 /t… |
| 224 | 3.0 | 25.0 | 否 | c1, c2, c3, c4, c5 | 0.0(--limit 25);同时读到 38,255 输出 token / 25 题,agent 据此判定模型不停,进… |
| 315 | — | — | 是 | c6, c7 | 0.72(全量 100 题;i=328 读回);相对 i=224 的 0.0 不是单变量对照——subset 3000→… |
| 397 | — | — | 是 |  | 未拿到:这个 nohup 版 orchestrator 在 model_v1 训练还剩约一小时(step 309/271… |
| 416 | — | — | 是 |  | 未拿到:第二个 orchestrator,i=450 把脚本改写成先跑 finalize.py 之后,i=454 用 p… |
| 461 | — | — | 是 | c8, c9 | 0.91(第三个 orchestrator,先跑 finalize.py model_v1 写入贪婪解码再跑全量评测;i… |
| 881 | — | — | 是 | c16 | 0.91(直接对交付物 final_model 复测;与 i=461 对同一份权重同一份配置的 0.91 逐位相同) |
| 937 | — | — | 是 | c17, c19 | 0.92(v6 提升为 final_model 后的复测;与 i=930 对同一份权重的 0.92 逐位相同) |

### 异常与存疑

- **1 次验证没有拿到信号**:i=[224]
- **分类学缺口提案 1 条**
  - pipeline-plumbing(i=375, i=653, i=664, i=506, i=719, i=454)
- **定义缺陷 6 条**
  - (i=341, i=397, i=398, i=519, i=425)
  - (i=506, i=670, i=769, i=830, i=930)
  - (i=653, i=649, i=774, i=820, i=874)
  - (i=405, i=454, i=534, i=67)
  - (i=664, i=663, i=183)
  - (i=577, i=653, i=655)
- **边界情形 4 条**
  - §C5 把 checkpoint 选择定义为「在已训好的若干步数里挑一个交」。本条 run 里 agent 做的是同一动作的另一种形态:在 6 次独立训练产出的最终模型之间挑一个交(i=537 选 v1、i=933 改选 v6),零训练成本、只花两次全量评测,效应 +1 分。但候选之间差的是 C4 变量(epochs 3 vs 4)而非训练步数,按现定义既不是 C5(不是同一次训练的不同步数)也不…(i=933, i=930, i=806)
  - 训练集规模(同一数据池里取多少条)按现定义无家可归:C3 是「数据从哪来、按什么比例混」,C4 列举的是「全参/LoRA、SFT/DPO/GRPO、lr/epoch/batch/序列长度」。i=341 一步之内把 --subset 5000 去掉(5,000 → 28,930 条,同一个 train_full.jsonl、同一来源同一配方)并把 epochs 2→3。这两项的合力都是「更多优化步数…(i=341, i=344, i=259)
  - /tmp/gen_test.py(i=240)是 agent 自建的廉价判定器:直接用 HF generate 贪婪解码一条构造样本,打印首 token id、top-5 logits 与解码文本,用来决定值不值得花一次全量评测。它符合 C7 的动机(「因为第四档太贵,自己搭一个更便宜的」)却不产出分数,只给二值/结构性答案,因此又像第一档。C7 现在的定义是「自己写一个便宜的打分器代替官方评测」…(i=240, i=312, i=314)
  - i=256 与 i=264 两次训练写同一个产物目录 sanity_model 和同一个日志 /tmp/sanity_train2.log。骨架判 i=256 superseded(0.02h)、i=264 consumed(0.14h),但轨迹两边都支持不了:i=267 说明 i=264 的进程 PID 3316 启动了并把 norm=1.539 写进日志,而 i=294 的 `ps -C py…(i=267, i=294, i=306, i=283)

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
| 139 | smoke | 0.26h | consumed | **smoke** | baseline（本 run 第一次训练）。这是冒烟跑：combined_sft.jsonl 取 8000 条 / 1 epoch / bs 16 / grad_accum 2，目的是验证代码路径能跑通并拿到一个参考分，既不在验 C3 也不在验 C4。tested_variable 的取值集里没有能… |
| 362 | real | 0.08h | killed | **both** | 与冒烟相比：数据从 8000 条子集换成全量 combined_v2.jsonl（48461 条 = xlam + 扩到 20000 条的 synthetic），同时 epochs 1→2。数据与超参一起变，不可拆。真实结局：跑到 第 69/3028 步（约 5 分钟）时被 agent 主动 pki… |
| 405 | real | 2.06h | killed | **both** | 与 v1 相比：数据 combined_v2 → combined_v3（50461 条，synthetic 换成带 5 个新工厂的 v2），同时 epochs 2→1。真实结局：第 399/1576 步（约 17 分 40 秒）时 torch.OutOfMemoryError 崩溃，metrics… |
| 519 | real | 1.30h | killed | **both** | 与 va 相比：数据 combined_v3 → combined_v4（synthetic_v4，加科学记数法/负数/高精度/可选参数技能），同时 OOM 修复带来的超参变动 bs 16→12、grad_accum 2→3、max_len 1536→1280、开 expandable_segmen… |
| 591 | real | 0.04h | killed | **both** | 与 vb 相比：数据 combined_v4 → combined_v5（随机化表达式空格 + 逐字拷贝工厂），epochs 1→2，bs 12→16。真实结局：启动约 2 分 14 秒后被 agent 主动 pkill（改为 1 epoch 以加快反馈），无产出。【不在骨架训练表里，见 d1】 |
| 621 | real | 1.21h | last_seen | **C4** | 与上一次训练（i=591，同为 models/vc + combined_v5）相比，唯一差别是 epochs 2→1；bs 16 / grad_accum 3 / max_len 1280 逐字相同。（对 agent 而言它同时也在对比 vb，那个对比里数据与 bs 都变了。）真实结局：跑满 10… |
| 689 | real | 1.23h | last_seen | **C3** | 与 vc（1 epoch）相比，唯一差别是数据 combined_v5 → combined_v6（synthetic 24000→25000，新增 summarize_dataset / plan_route / build_playlist 三类可选布尔工厂）；epochs 同为 1，pipel… |
| 751 | real | 2.40h | last_seen | **C4** | 与 vd 相比，唯一差别是 epochs 1→2；数据文件逐字相同（都是 data/combined_v6.jsonl），bs 16 / grad_accum 3 / max_len 1280 相同。本 run 唯一一次单变量的 C4 对照。真实结局：跑满 2222/2222 步（正好是 vd 的两… |

### 验证序列(16 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 42 | 3.0 | 25.0 | 是 |  | 0.0 |
| 163 | 3.0 | 100.0 | 是 | c1, c2, c3, c4 | 0.03 |
| 193 | — | — | 是 | c1, c2, c3, c4 | 0.03 |
| 322 | 3.0 | 100.0 | 否 | c5 | 0.95 |
| 362 | — | — | — | c7 | 未拿到 |
| 396 | — | — | — |  | 未拿到 |
| 405 | — | — | — | c8 | 未拿到 |
| 510 | — | — | — |  | 未拿到 |
| 519 | — | — | — | c9, c10 | 0.94 |
| 591 | — | — | — | c12 | 未拿到 |
| 614 | — | — | — |  | 未拿到 |
| 621 | — | — | — | c11, c12 | 0.95 |
| 689 | — | — | — | c13 | 0.95 |
| 729 | 3.0 | 100.0 | 是 | c16 | 0.95 |
| 751 | — | — | — | c19 | 0.95 |
| 803 | — | — | 是 | c16, c6 | 0.95 |

### 异常与存疑

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
| 203 | smoke | 0.01h | returned | **smoke** | baseline(本 run 第一次训练)。冒烟:--limit 64 --epochs 1 --bs 8,只验管线跑不跑得通,不检验任何 C3/C4 变量。四值域里没有对应取值,故记 unclear —— 见 definition_defect dd6。结局:前台跑完(train_runtime … |
| 233 | real | 0.05h | killed | **both** | vs 冒烟(i=203):数据从 64 行子集换成完整 18k data/train.jsonl(C3);bs 8→16、grad-accum 1→2、epochs 1→3、显式 LoRA r64/a128 lr2e-4(C4)。首个真实候选,两族同时确定。真实结局:启动 3 分 11 秒后被 ag… |
| 263 | real | 1.62h | consumed | **unclear** | 与 i=233 的训练命令逐字相同(只多了前置 rm -rf runs/lora1);唯一变化在 train.py 里:save_strategy "no"→"epoch"(i=257)。这 1.6 小时不检验任何 C3 或 C4 变量,它是为了拿到逐 epoch checkpoint 才重跑的,真… |
| 468 | smoke | 0.01h | returned | **smoke** | 冒烟:第一次试 full-FT 路径(--limit 96 --full-ft --optim adamw_8bit --attn sdpa --lr 1e-5),只量显存与步速,不产可比分数;四值域装不下(dd6)。真实结局:训练部分正常结束(train_runtime 26.96s),随后因 -… |
| 474 | real | 1.73h | consumed | **both** | vs 上一次真实训练 i=263:数据 train.jsonl(18k)→train_v2.jsonl(22k,剥了 null 参数且是新的一次抽样)(C3);方法 LoRA r64→全参微调、lr 2e-4→1e-5、grad-accum 2→1、attn eager→sdpa、optim→ada… |
| 566 | real | 0.03h | superseded | **C4** | vs i=474:数据不变(仍 train_v2),方法从全参回到 LoRA r64/a128、lr 1e-5→2e-4、grad-accum 1→2、epochs 3→4,attn 仍 sdpa。纯 C4。真实结局:启动约 1.9 分钟后被 i=572 那条命令里的 pkill 杀死;骨架记的 e… |
| 572 | — | — | — | **unclear** | 这一行不是一次训练启动。命令里的 nohup python train.py … --epochs 3 从未执行:排在它前面的 pkill -f "train.py --data data/train_v2" 匹配到 harness 自己的 bash wrapper(i=583 的 pgrep -a… |
| 594 | real | 1.96h | consumed | **C4** | vs i=566:唯一差别是 --epochs 4→3(数据、LoRA 秩、lr、bs、grad-accum、attn 全部不变),agent 的理由是 4 epoch 要 2h35m 太久、且更多 epoch 有和 full-FT 一样的过拟合风险。纯 C4。真实结局:跑满,06:34:08 wa… |
| 653 | real | 1.62h | consumed | **C3** | vs 上一行 i=594:数据 train_v2(22k)→train_v3(18k),attn sdpa→eager。但它真正的对照臂是 i=263 的 lora1:相对 lora1,命令行参数逐字相同(epochs 3 / bs 16 / grad-accum 2 / lr 2e-4 / max… |

### 验证序列(17 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 139 | 3.0 | 8.0 | 是 |  | 0.0(base 模型,8 题) |
| 209 | 3.0 | 8.0 | 否 | c1, c3 | 未拿到 —— vLLM 起服务阶段就 RuntimeError(merged 目录缺 preprocessor_conf… |
| 223 | 3.0 | 8.0 | 是 | c1, c2, c3, c4 | 1.0(8 题,前台直接回读) |
| 408 | — | — | 是 | c1, c2, c3, c7, c10 | 0.94(94/100,全量、贪婪) |
| 422 | — | — | 否 | c6 | 0.93(runs/lora1/merged_ep2)。这一对评测被 harness 转成后台任务 bwdltrzj1,… |
| 422 | — | — | 否 | c6 | 0.93(runs/lora1/merged_ep2)。这一对评测被 harness 转成后台任务 bwdltrzj1,… |
| 422 | — | — | 否 | c6 | 0.90(runs/lora1/merged_ep1)。同上,经 tasks/bwdltrzj1.output 读回;e… |
| 422 | — | — | 否 | c6 | 0.90(runs/lora1/merged_ep1)。同上,经 tasks/bwdltrzj1.output 读回;e… |
| 462 | — | — | 是 | c7, c10, c14 | 0.95。同一份权重、同一条命令重跑,比 i=408 的 0.94 高 1 题,agent 由此得出「贪婪解码下 vLL… |
| 510 | — | — | 是 | c11, c12, c13 | 0.94(fft1 epoch-3,全量) |
| 523 | — | — | 否 | c13, c16 | 0.93(checkpoint-1358)/ 0.94(checkpoint-2716)。同样被转后台(bvdpeqoy… |
| 533 | — | — | 否 |  | 未拿到 —— 这一行根本不是评测启动,是一个 while pgrep 轮询等待脚本(等 fft1 ep2 的评测进程消失… |
| 617 | — | — | 是 | c12, c17 | 0.92(lora2 epoch-3,全量;失败集多出 87/88/94) |
| 623 | 3.0 | 3.0 | 是 | c14, c15 | 0.95 —— 但只跑完 3 次里的第 1 次,agent 随即 pkill -9 掉整个 eval_robust 去抢… |
| 679 | — | — | 是 | c18, c19 | 0.94(lora3 epoch-3,全量;92 修好了,88 换成失败) |
| 685 | 3.0 | 3.0 | 否 | c15, c18, c19 | 0.94 / 0.94 / 0.94,MEAN 0.9400(n=3),经 tasks/bl1ok44tk.output… |
| 716 | 3.0 | 4.0 | 否 | c14, c15 | 0.95 ×4,MEAN 0.9500(n=4),经 tasks/bf4mgsqqk.output 读回。这次判定推翻了… |
| 746 | — | — | 是 | c14 | 0.95(95/100),按评分方的默认口径直接评 final_model;失败集固定为 25/26/54/69/70 |
| 762 | — | — | 否 |  | 未拿到 —— 这一行不是评测,是收尾清理 + 状态打印,只是把 i=746 已经落盘的 logs/final_verif… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[263, 572]
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

## claude_non_api_max_claude-opus-4-8_10h_run1__gpqamain_google_gemma-3-4b-pt_17314663
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gpqamain | google_gemma-3-4b-pt | 8.68h | 0.2611607142… |

### 改动序列(22 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 123 | C3 | 数据来源定为 nvidia/Llama-Nemotron-Post-Training-Dataset 的 science split(620k 条 R1 蒸馏的科学 MCQ 推理);GPQA 只被载入用来建 13-gram shingle 污染过滤器 | i=123, i=120 |
| 123 | C2 | build_data.py 用逐字复制的 inspect SINGLE_ANSWER_TEMPLATE_COT 渲染训练 prompt,并把 teacher 答案统一改写成末行 `ANSWER: X` | i=123, i=23 |
| 143 | C1 | train.py 在每次 merge 落盘时写入贪婪 generation_config.json(do_sample false / temperature 0.0 / eos [1,106]),把解码配置固化进每个候选 | i=143, i=73 |
| 143 | C4 | train.py: LoRA r32/alpha64 限定在 language_model 分支(正则避开 vision tower 同名 q/k/v proj),bf16 + prompt masking 的 SFT 训练脚本成型 | i=143 |
| 173 | C4 | 装 liger-kernel 并在 train.py 顶部打 fused-linear-cross-entropy 补丁,解决 gemma-3 262k 词表 logits.float() 的 OOM,使 bs=16 可用 | i=173, i=189 |
| 203 | C4 | 给 train.py 加 --grad_ckpt 开关以便消融梯度检查点;i=207 实测关掉即 OOM,遂固定为 1 | i=203, i=207 |
| 269 | proposed:eval-log-instru… | 自写 show_eval.py:寄生在官方 inspect 日志上,把单一 accuracy 拆成 no-ANSWER 率、平均补全长度、逐题 target/pred。不替代官方评测(必须先跑一次),但本 run 的每个决策都由它给出 | i=269 |
| 407 | C1 | 给 v1 的 merged generation_config 单字段加 repetition_penalty=1.3,作为对付重复循环的推理侧补丁 | i=407 |
| 413 | C3 | build_concise.py:训练目标改成只取 teacher 回答里 </think> 之后的摘要,丢弃会诱发循环的 think 块(mean 250 tok) | i=413, i=427 |
| 421 | C2 | 摘要清洗:删掉领头的答案声明行,保证训练目标是「先推理后 ANSWER」的顺序;fallback 改为按句边界取最后 5 句 | i=421 |
| 435 | C2 | 上一条不够:再加正则删掉同一行内领头的 "The answer is (X)." 整句;为此杀掉已跑 1 分钟的构建重来 | i=435, i=433 |
| 478 | C1 | 回退:把 repetition_penalty 从 v1 的 config 里 pop 掉,恢复干净基线以便后续比较 | i=478 |
| 552 | C3 | v3 数据:从 70k 摘要集里只保留 assist_tok>=180 的「有实质内容」子集(37,987 条) | i=552 |
| 566 | C3 | build_deriv.py:训练目标改成有界的 think 块本身(200-750 tok)并按 wait/hmm/but wait 等循环标记词过滤,提供真正的分步推导 | i=566, i=574 |
| 612 | C3 | v4 配方:45k 推导 + 30k 摘要 = 75k(推导占 0.60) | i=612 |
| 662 | C3 | v5 配方:推导更重,45k 推导 + 15k 摘要 = 60k(推导占 0.75) | i=662 |
| 662 | C4 | v5 同时把 LoRA rank 32->48 / alpha 64->96(可训练参数 59.6M->89.4M),与 c15 同一次启动、无法拆开 | i=662, i=669 |
| 695 | C3 | v6 配方:重新平衡到 45k 推导 + 40k 摘要(推导占 0.53),并把 rank 退回 32 以便与 v4 单变量对比 | i=695, i=698 |
| 738 | C1 | final_model 的 generation_config 被整份重写为贪婪版(字段集与 train.py 写出的一致,未丢字段) | i=738 |
| 738 | C5 | 候选选择:把 v6 的 merged 拷成 final_model 作为提交(在 v1..v6 六个已训好的候选里挑,零训练成本) | i=738 |
| 781 | C1 | 把 repetition_penalty=1.1 锁进 final_model 的 generation_config(相对 c19 恰好多一个字段) | i=781 |
| 836 | C1 | 把 work/out_v6/merged 的 config 改回纯贪婪(无 repetition_penalty),用于和 final_model 做全集 448 题的单字段对照 | i=836 |

### 训练序列(13 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 155 | smoke | 0.01h | returned | **smoke** | baseline(首次启动)。意图是验证 Gemma3 多模态前向/反向 + LoRA + merge 代码路径跑不跑得通,不在验 C3 或 C4 的任何一项;顺带量到 3.86 samples/s 的基线吞吐。四个取值里只有 unclear 装得下(见 definition_defect d6) |
| 161 | smoke | 0.04h | superseded | **C4** | 相对 i=155:bs 4->16、accum 2->1、attn eager->flash_attention_2、limit 48->256;数据文件不变(science_sft.jsonl)。结果 OOM(262k 词表 logits.float() 44GB) |
| 191 | smoke | 0.03h | returned | **C4** | 相对 i=161:唯一变化是 train.py 顶部加了 liger fused-linear-CE(i=189),bs/attn 不变,limit 256->512。数据不变。8 samples/s,OOM 消失 |
| 197 | smoke | 0.03h | superseded | **C4** | 相对 i=191:bs 16->32,limit 512->384;数据、liger、attn 不变。7.587 samples/s、峰值 47967 MiB,判定不如 bs=16 |
| 207 | smoke | 0.15h | discarded | **C4** | 相对 i=197:bs 32->16 且 grad_ckpt 1->0(其余逐字相同,limit 均 384)。结果 CUDA OOM,判定梯度检查点必需 |
| 213 | real | 0.06h | consumed | **baseline** | 相对 i=207:grad_ckpt 回到 1、limit 384->1024、save_merged 0->1。意图是验证 merge->save->vLLM 装载这条未测过的通路,不在验 C3/C4 变量(同 d6) |
| 238 | real | 1.19h | consumed | **unclear** | baseline(首次真实训练,v1)。全量 40k science_sft(完整 R1 trace)、accum 1->2、limit 全量;超参沿用冒烟阶段定下的 bs16/FA2/grad_ckpt。它建立基准而不是检验某一项,故记 unclear(同 d6) |
| 478 | real | 0.08h | superseded | **both** | 相对 v1:数据从 science_sft(40k 完整 R1 trace)换成 science_summary(50k post-</think> 摘要)=C3,同时 epochs 1->2、bs 16->32、max_len 2816->1408 = C4。两类同时变,拆不开 |
| 501 | real | 0.03h | last_seen | **C4** | 相对被它作废的 i=478:数据文件逐字相同(science_summary.jsonl),唯一变化 epochs 2->1。动机是「先拿信号」而非「2 epoch 更差」,但结果确实是一次单变量的 epoch 改动 |
| 560 | real | 0.03h | last_seen | **both** | 相对 v2:数据 science_summary(50k)->science_v3(38k,来自 summary_big 的 >=180 tok 子集)= C3,同时 epochs 1->2 = C4。事后 agent 把回归归因给 epochs 过拟合,但两项同时变 |
| 612 | real | 4.06h | last_seen | **both** | 相对 v3:数据 science_v3(纯摘要)->science_mix(45k 有界推导 + 30k 摘要)= C3,同时 epochs 2->1 = C4。agent 明说「1 epoch only」是从 v3 学到的教训,所以两项都是有意的 |
| 662 | real | 0.00h | discarded | **both** | 【骨架该行的产物标错,见 definition_defect d1】i=662 真正启动的训练是 v5:--data science_mix_v5(45k 推导+15k 摘要,推导占比 0.60->0.75)= C3,同时 rank 32->48 / alpha 64->96 = C4。agent … |
| 695 | real | 1.39h | consumed | **C3** | 相对 v5:配方从 45k+15k(0.75)改成 45k+40k(0.53),摘要池换成 summary_big;同时把 rank 从 48 退回 32、alpha 退回 64 —— 退回的是 v4 的取值,目的正是让本次与 v4 只差配方一项。所以受测变量是配方(C3),rank 的回退是为对齐… |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 82 | 3.0 | 8.0 | 是 |  | 0.0 |
| 225 | 3.0 | 16.0 | 是 | c1, c2, c3, c4 | 0.125 (n=16);同一份日志在 i=272 被 show_eval 复读:2/16 正确、7/16 没有 ANS… |
| 356 | 3.0 | 128.0 | 是 | c1, c2, c3, c4, c9 | 0.125 (n=128);show_eval:16/128 正确、50/128 没有 ANSWER、平均 22546 … |
| 407 | 3.0 | 96.0 | 是 | c7 | 0.13541666666666666 (n=96),57/96 没有 ANSWER、平均 23631 字符 —— 判定… |
| 523 | — | — | — |  | 未拿到:这条 nohup 链在 8 秒后于 i=527 被 kill,由 i=529 的 tracked 版本取代(同一… |
| 529 | — | — | — | c9, c10, c11 | 0.234375 (n=128),7/128 没有 ANSWER、平均 3303 字符 —— 判定「摘要形态的训练目标」… |
| 576 | — | — | — | c12 | 0.1640625 (n=128),5/128 没有 ANSWER、平均 1792 字符 —— 相对 v2 回退,age… |
| 616 | — | — | — |  | 未拿到:这条 inline `&` 链在 i=622 被 pkill,由 i=633 的 tracked 版本取代 |
| 633 | — | — | — | c13, c14 | 0.2734375 (n=128),12/128 没有 ANSWER、平均 4775 字符 —— 判定「有界推导 + 摘… |
| 672 | — | — | — | c15, c16 | 0.21875 (n=128),22/128 没有 ANSWER、平均 10378 字符 —— 同时裁决配方(0.75 … |
| 707 | — | — | — | c17 | 0.296875 (n=128),10/128 没有 ANSWER、平均 3774 字符 —— 本 run 读到的最高值… |
| 742 | — | — | — | c20 | 拿到了(骨架记 got_signal=False,是假阴性):两臂 n=256,none=0.2422 / 27 条无 … |
| 803 | 3.0 | 448.0 | 是 | c17, c18, c19, c20 | 0.27232142857142855 (n=448 全集),20 条无 ANSWER、平均 2885 字符 —— 这是… |
| 836 | 3.0 | 448.0 | 否 | c20, c21 | 拿到了(骨架记 got_signal=False,是假阴性):0.22321428571428573 (n=448)。取… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[238]
- **9 次验证没有拿到信号**:i=[523, 529, 576, 616, 633, 672, 707, 742, 836]
- **分类学缺口提案 2 条**
  - target-trace-surgery(i=413, i=566, i=552, i=404, i=549)
  - eval-log-instrumentation(i=269, i=404, i=549)
- **定义缺陷 6 条**
  - (i=662, i=662, i=692)
  - (i=785, i=785, i=800)
  - (i=501, i=523, i=529, i=549, i=604)
  - (i=612, i=656, i=785, i=735, i=738)
  - (i=155, i=213, i=238)
  - (i=803, i=836)
- **边界情形 3 条**
  - 跨 run 的候选选择。C5 的定义是『在已训好的若干步数里挑一个交』——同一次训练的不同 checkpoint。本 run 的选择动作结构上完全一样(零训练成本、用第三/四档裁决、纯选择),但候选来自**六次不同的训练**(v1..v6),不是一次训练的不同步数:i=738 把 v6 拷成 final_model,i=785 又把 v4 拉回来在同一档位(limit 256、rep 1.1)复测…(i=738, i=785)
  - 改写训练目标里答案句的位置。i=421/i=435 把 teacher 摘要开头的 "The answer is (X)." 删掉,让推理先行。按 C2(『统一答案标记』、让训练样本符合评测期望的形态)读,它是格式对齐;按 C3(数据配方)读,它是对训练数据内容的改写。两读都能自圆:它不改 prompt 侧的任何字节(C2 的核心判据是与评测 prompt 逐字节一致),但也不换数据源、不改混合比…(i=421, i=435)
  - `python3 -c` 整份重写 generation_config.json 时,如何数改动次数。train.py(i=143)写出的 config 相对 base 模型的 config(i=81 记述:`do_sample:true, top_k:64, top_p:0.95`、无 temperature)是一次整份重写:加了 temperature/do_sample false,同时**…(i=143, i=81)

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
| 382 | smoke | 0.01h | returned | **smoke** | baseline —— 全 run 第一次训练启动。这是一次冒烟(--n 200),目的是验证 train_sft.py 能否跑通,不在验 C3 或 C4 的效应。结局:崩溃,不是「returned 成功」—— SFTTrainer 构造时 transformers 去 import liger_k… |
| 455 | smoke | 0.03h | returned | **smoke** | 相对 i=382:样本量 200→400,且中间(i=453)把 use_liger_kernel 关掉、optim 由 adamw_torch_fused 换成 adamw_bnb_8bit。仍是冒烟 —— 验证的是代码可运行 / 显存够不够 / 能不能落盘,不能判定训练效果。结局:完整跑完,tr… |
| 566 | real | 0.94h | consumed | **both** | baseline —— 第一次真实训练,数据配方(OpenScience 20k、字母均衡、≤6144 token)与训练方法(全参 SFT、lr 1e-5、bs2×accum4、bfd packing、adamw_bnb_8bit)同时首次落地,没有任何一项被单独隔离。结局:正常跑完,不是崩溃也不… |
| 1195 | real | 2.47h | consumed | **C3** | 相对 v1(i=566):唯一被改的是数据 —— 换成 sft_v2_40k(按 completion 长度 7000 字符分层、长短各半、剔除 held-out),样本 20000→40000,token 30.4M→81.2M,均长 1520→2037。超参逐字相同:--epochs 1 --b… |
| 1347 | real | 1.61h | consumed | **both** | 相对 v2(i=1195)同时动了两侧:C3 —— 换成 sft_v3(同一 7000 字符分层配方,但样本是 v2/held-out 都没用过的 26000 条新样本,seed 21);C4 —— lr 1e-5→6e-6、--init 从 base 改成 runs/v2/final(续训而非重训… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 121 | 3.0 | 100.0 | 是 |  | 0.15 |
| 266 | — | — | 是 |  | 未拿到 —— 这不是一次评测。i=266 是 prep_data.py 建 SFT 数据,机械层记的 0.04 实为后面… |
| 480 | 3.0 | 25.0 | 是 | c2, c4, c5, c6 | 0.04 |
| 743 | — | — | 是 |  | 未拿到 —— 这不是一次评测。i=743 是 prep_data.py 建 sft_big,机械层记的 0.125 实为… |
| 942 | 3.0 | 200.0 | 是 | c1, c2, c3, c4, c6, c7 | 0.125(n=200);同一命令里 report.py 还回读了 no-ANSWER=160(80.0%)与 stop… |
| 1321 | 3.0 | 200.0 | 是 | c11, c15 | 0.4 |
| 1376 | 3.0 | 200.0 | 是 | c16, c18, c19 | 0.39 |
| 1406 | 3.0 | 200.0 | 是 | c20 | 0.385 |
| 1426 | 4.0 | -1.0 | 是 | c16, c18, c19 | 0.410714…(v3,全 448;后台启动,分数由 i=1429 的 until 轮询 + cat runs/v3/… |
| 1426 | 4.0 | -1.0 | 是 | c16, c18, c19 | 0.410714…(v3,全 448;后台启动,分数由 i=1429 的 until 轮询 + cat runs/v3/… |
| 1426 | 4.0 | -1.0 | 是 | c11, c15 | 0.401785…(v2,全 448;同一后台脚本的第二条,分数由 i=1444 的 until 轮询在 i=1447 … |
| 1426 | 4.0 | -1.0 | 是 | c11, c15 | 0.401785…(v2,全 448;同一后台脚本的第二条,分数由 i=1444 的 until 轮询在 i=1447 … |
| 1458 | — | — | 是 | c17, c19, c21 | 0.5,但样本量是 50 不是 448 —— 命令没给 --limit,而 evaluate.py 的默认是 50;re… |

### 异常与存疑

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
| 436 | — | — | — | **unclear** | 意图是相对 i=419 只改内存形状(bs 4→8、accum 4→2、加 --grad-ckpt),数据逐字相同。但这条命令以 `pkill -f train_sft.py` 开头,pkill 匹配到自己的 shell,整条以 exit 144 结束——训练进程从未启动(i=439 的 ps 为空… |
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
| 70 | 3.0 | 150.0 | 否 |  | 未拿到 |
| 77 | — | — | 否 | c1 | 未拿到 |
| 215 | 3.0 | 150.0 | 是 | c3 | 未拿到 |
| 282 | 3.0 | 150.0 | 是 | c3, c4 | 0.4066666666666667 |
| 1069 | 3.0 | 150.0 | 是 | c10, c14, c24 | 0.9066666666666666 |
| 1086 | 3.0 | 150.0 | 是 | c19, c29, c20, c21, c23, c25 | 0.9133333333333333 / 0.92 / 0.92 / 0.9133333333333333(一个事件里跑… |
| 1094 | 3.0 | 150.0 | 是 | c26 | 0.92 |
| 1102 | 3.0 | 150.0 | 是 | c21 | 0.92 |
| 1102 | 3.0 | 150.0 | 是 | c21 | 0.92 |
| 1102 | 3.0 | 150.0 | 是 | c19, c29 | 0.92 |
| 1102 | 3.0 | 150.0 | 是 | c19, c29 | 0.92 |
| 1113 | 3.0 | 150.0 | 是 | c21, c24, c27 | 0.9266666666666666 |
| 1133 | 3.0 | 150.0 | 是 | c28 | 0.9066666666666666 |
| 1141 | 3.0 | 150.0 | 是 | c21, c27 | 0.92 |

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
| 143 | smoke | 0.00h | returned | **smoke** | baseline;本 run 第一次训练启动,--epochs 0.005 --max_samples 500 的冒烟,验的是 train_sft.py 跑不跑得通(第二档),既不在验 C3 也不在验 C4。harness 层以 Exit code 144 失败,没有任何训练输出。 |
| 145 | smoke | 0.00h | returned | **smoke** | 与 i=143 的命令逐字相同,只把 `\| tail -60` 改成 `\| tail -80`;仍以 Exit code 144 失败。受测的是管线可运行性,不是 C3/C4。 |
| 147 | smoke | 0.07h | discarded | **smoke** | 与 i=145 同参数,只把输出从管道改成重定向到 /tmp/train_test.log 以绕开 Exit code 144。这次跑通:train_runtime 15.2769 秒、train_loss 0.536、权重落盘成功。受测的仍是管线可运行性。 |
| 154 | smoke | 0.02h | returned | **C4** | 相对 i=147:bs 4→16、grad_accum 2→1、max_len 默认 1024→896、max_samples 500→1024、epochs 0.005→1.0。目的是量吞吐以定真实训练规模,读到 17.7 samp/s 后直接决定 v1 用 150k 样本。受测变量是 C4 的 … |
| 163 | real | 1.21h | discarded | **both** | baseline —— 第一次真实训练,同时确定 v1 的数据配方(c1/c2)与超参(c3/c6),两者绑在一次训练里,轨迹里没有把它们分开的对照。实际结局与机械层记录不符:该命令被 harness 自动转入后台(i=165 返回 'Command running in background wi… |
| 244 | real | 1.24h | consumed | **both** | i=163 被杀后的重启:max_samples 150000→100000、save_steps 2000→500、logging_steps 50→25、改用 nohup;数据与 lr/bs/grad_accum/max_len/warmup 逐字不变。因为 i=163 一个分数都没产出,这一次… |
| 321 | real | 1.35h | consumed | **both** | 相对 i=244 同时动了两类:C3——数据换成 train_v2.jsonl(train.jsonl 第 100000 行之后的 107,473 条未见样本);C4——初始权重从 base 换成 --model .../sft_v1 的续训、lr 1e-5→3e-6。bs/grad_accum/m… |
| 379 | real | 2.52h | consumed | **both** | 相对 i=321:C3——数据换成 train_v3.jsonl(OMI2 100k + MetaMathQA GSM 100k 的混合来源);C4——回到从 base 起训、lr 3e-6→1e-5、max_samples→200000、save_steps 500→1500。结局:正常跑完 62… |

### 验证序列(6 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 100 | 3.0 | 50.0 | 是 |  | 0.22 |
| 111 | — | — | 否 |  | 未拿到(该事件不是评测) |
| 292 | 3.0 | 150.0 | 否 | c1, c2, c3, c4, c5, c6, c7 | 0.56 |
| 364 | 3.0 | 150.0 | 是 | c8, c9 | 0.5533333333333333 |
| 439 | 3.0 | 150.0 | 是 | c11, c12 | 0.5866666666666667 |
| 453 | 3.0 | 150.0 | 是 | c13 | 0.5733333333333334 |

### 异常与存疑

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

## claude_non_api_claude-opus-4-7_10h__gsm8k_Qwen_Qwen3-1.7B-Base_17122414
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 2.08h | — |

### 改动序列(15 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 73 | C3 | 确定训练数据来源与配方:GSM8K 官方 train(7,473)+ MetaMathQA 的四个 GSM_* 子集(239,998),共 247,471 条;显式排除 MATH_* 子集。 | i=73, i=85 |
| 73 | C2 | 把训练答案改写成评测要求的 `ANSWER: X` 结尾,并剥掉 GSM8K 的 `####` 与 `<<...>>` 计算器标注,使训练目标与评分器口径一致。 | i=73, i=73 |
| 95 | C4 | 确定训练方法:全参 SFT + completion-only loss(prompt 位置置 -100)、bf16、cosine、adamw_torch_fused、save_strategy=no(不存中间 checkpoint)。 | i=95, i=95 |
| 192 | C11 | 把官方 inspect_ai 日志转成可决策信号:逐题 match 判定计数(3/150)+ 输出长度,据此认定问题不是「算错」而是「输出乱码」,直接导向 c4 的诊断。 | i=192, i=193 |
| 196 | C4 | 默认学习率 1e-5 → 5e-6,其后所有训练(sft_test2 / sft_v2 / sft_fs_test / sft_v3)都用 5e-6。轨迹里没有给出理由。 | i=196, i=214 |
| 196 | C2 | 改用手工拼接 chat 轮次,避开 qwen3.jinja 在真实 assistant 轮插入的 `<think>\n\n</think>`(推理时不插入);同时把 user 轮包进评测的 MATH_PROMPT_TEMPLATE 原话。这是本 run 唯一一次把分数从 0.020 抬起来的改动。 | i=195, i=196, i=196 |
| 199 | proposed:candidate_teard… | 删掉唯一一个已训好、已评过分的候选 sft_v1(0.020),此时 final_model 里什么都没有。 | i=199 |
| 234 | C1 | sft_v2 的 generation_config.json:eos_token_id 由 [151643] 改成 [151645, 151643](加入 <\|im_end\|>);同一条命令还把 tokenizer_config.json 的 eos_token 改成 <\|im_end\|>… | i=232, i=234, i=235 |
| 250 | C1 | sft_v2 的 config.json:eos_token_id 151643 → 151645,并把 use_cache 由 false 翻成 true。同为逐字段改写,其余字段不变。机械层的 config 表没有这一行(它只跟 generation_config.json)。 | i=249, i=250, i=250, i=251 |
| 255 | C11 | 从同一份官方日志读停止原因分布(截断率):stop 99 / max_tokens 51,用它判断 c6+c7 的 eos 修补是否真的让模型停下来。确定性、零额外开销,与 reference §3 C11 的定义完全吻合。 | i=255, i=256 |
| 280 | C2 | 训练样本按 0.9 概率注入 8–10 条来自 GSM8K train 的 few-shot 作为 system 消息,对齐评测的 fewshot=10 system 提示(reference §3 C2 定义里的「按比例注入 few-shot 上下文」)。 | i=279, i=280, i=298 |
| 280 | C2 | 丢弃被 max_len 截断、末尾不是 <\|im_end\|>(151645)的训练样本,保证每条训练目标都以结束符收尾。 | i=280 |
| 280 | C1 | 把 c6/c7 的 eos 修补写进训练脚本的保存路径(save_model 之后逐个 patch generation_config.json 与 config.json),使之后每个 checkpoint 出生即带修复,不必再手工补。 | i=280, i=280 |
| 283 | proposed:candidate_teard… | 剩余 8h19m 时删掉唯一幸存的已训候选 sft_v2(最好读数 0.040)。此举并非必要:下一次训练写的是另一个路径 sft_v3,且磁盘只用了 1%。删除之后这条 run 再也没有产出过任何权重。 | i=283, i=284, i=398 |
| 298 | C4 | 为容纳 few-shot 拉长的序列而改超参:max-len 1024→3584、bs 16→4、grad-accum 2→4(有效 batch 32→16)、max-samples 80000→60000。四项都是补偿性的,不是被检验的对象 —— 见 boundary_case bc1。 | i=298, i=214 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 98 | smoke | 0.01h | returned | **smoke** | baseline —— 本 run 第一次训练。500 样本冒烟,只验 train_sft.py 跑不跑得通,不检验 C3 或 C4 中的任何一项;现取值域里没有对应值(spec §10 第 1 条要补的 smoke 值)。 |
| 105 | real | 0.59h | consumed | **baseline** | baseline(首次真实训练)。同时第一次确定 C3(80,000 条 GSM8K+MetaMathQA-GSM)与 C4(全参 SFT、bs16×accum2、len1024、lr1e-5、1 epoch),没有可比的上一次;spec §4.2 规定首次训练 vs_previous 记 base… |
| 201 | smoke | 0.01h | returned | **smoke** | 与 i=98 同为 500 样本冒烟,但同时变了两项:训练侧提示格式(手工拼接 + MATH 提示词包装,C2)与 lr 1e-5→5e-6(C4)。主变量是 C2,不在 {C3, C4, both} 取值域内。 |
| 214 | real | 0.66h | consumed | **unclear** | 与 sft_v1 相比:数据文件、抽样种子、条数(80,000)、bs 16、accum 2、max-len 1024、1 epoch 全部逐字相同;只有训练侧提示格式(C2)与 lr(1e-5→5e-6,C4)变了。发起这次重训的理由 agent 自己写明是 chat template 的 <th… |
| 285 | smoke | 0.04h | returned | **smoke** | 1,000 样本冒烟,验 few-shot 版脚本能否跑通;bs2×accum4、len3072 是为放下 few-shot 序列所设。非 C3/C4 检验。 |
| 298 | real | 0.27h | last_seen | **unclear** | 与 sft_v2 相比:训练样本按 0.9 概率注入 8–10 条 few-shot(C2,见 c8);同时 max-len 1024→3584、bs 16→4、accum 2→4、max-samples 80000→60000。后四项是为容纳变长序列的补偿性改动,按 spec §10 第 2 条建… |

### 验证序列(7 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 49 | 3.0 | 50.0 | 是 |  | 0.180 |
| 172 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.020 |
| 203 | 3.0 | 30.0 | 是 | c4, c5 | 0.100 |
| 225 | 3.0 | 150.0 | 是 | c4, c5 | 0.053 |
| 236 | 3.0 | 150.0 | 是 | c6 | 0.020 |
| 252 | 3.0 | 150.0 | 是 | c7 | 0.040 |
| 287 | 3.0 | 30.0 | 是 | c8, c9, c10 | 0.033 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[214, 298]
- **分类学缺口提案 1 条**
  - candidate_teardown(i=199, i=283, i=403)
- **定义缺陷 3 条**
  - (i=320, i=284, i=403, i=403)
  - (i=232, i=226, i=237, i=253, i=234, i=250)
  - (i=105, i=214, i=195, i=279)
- **边界情形 2 条**
  - sft_v3(i=298)相对 sft_v2(i=214)同时动了三组东西:few-shot 注入(§3 C2 定义里逐字列了『按比例注入 few-shot 上下文』)、样本数 80000→60000(读作 C3)、bs/accum/max-len(读作 C4)。后两组都是为容纳 few-shot 拉长的序列而做的补偿 —— 序列从 1024 token 上限涨到 3584,单卡放不下原来的 bs…(i=298, i=214, i=280, i=308)
  - i=201 是 500 样本、63 步的冒烟(机械层按 --max-samples 500 判为 smoke,判得没错),但它的产物立刻被 evaluate.py --limit 30 评了一次读到 0.100(i=203/204),而这个读数正是 agent 启动 sft_v2 的唯一依据 —— 也就是说,一次冒烟的分数被当成了训练效果的证据。§2 第二档写明冒烟『不产出可比的分数』『不能判定训…(i=201, i=203, i=204)

## claude_non_api_claude-opus-4-7_10h__gsm8k_Qwen_Qwen3-4B-Base_17107218
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 8.34h | 0.6231993934… |

### 改动序列(28 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 59 | C3 | 数据来源:只用 GSM8K 官方 train split 全量 7473 条,不混任何外部数据(prepare_data.py)。 | i=59, i=62 |
| 59 | C2 | 格式对齐:训练样本的 user 侧逐字复制 inspect_evals gsm8k 的 MATH_PROMPT_TEMPLATE,assistant 侧把 '#### N' 改写成评测计分用的 'ANSWER: N'。 | i=59, i=59 |
| 90 | C4 | train_sft.py 默认超参上调:epochs 2.0→3.0、lr 5e-6→1e-5、max_length 1024→768(实际启动时又用 --max_length 640 覆盖)。 | i=90, i=90 |
| 147 | C2 | 重写 train_sft.py:绕开 tokenizer.apply_chat_template,手工拼 '<\|im_start\|>user…<\|im_start\|>assistant\n',消除 qwen3 模板对末条 assistant 插入的 '<think>\n\n</think>'… | i=141, i=147 |
| 199 | C1 | ckpt_sft_v2/generation_config.json 单字段增量改写:eos_token_id 由 151643 变成 [151643, 151645],让 vLLM 认 <\|im_end\|> 为停止符。字段级核实:改前(i=196)与改后(i=200)都打印了整份 JSON,b… | i=196, i=199, i=200 |
| 209 | C11 | 验证器工装:直接读官方 inspect_ai 日志,统计错题数与模型输出长度分布(mean/median/max),把官方评分器的输出转成可决策信号。 | i=209, i=210 |
| 213 | C9 | 提交守卫:把当时唯一可用的候选 ckpt_sft_v2(0.480)整目录拷成 final_model,不改动任何产物内容。 | i=213 |
| 218 | C3 | 数据配方:GSM8K 7473 条 + MetaMathQA 的 GSM 衍生子集 40000 条 = 47473 条(公开蒸馏数据集,来源类型 b)。 | i=218, i=221 |
| 229 | C4 | 超参:batch_size 8→16(有效 batch 16→32)、epochs 3→2.0、新增 --warmup_ratio 0.03。 | i=229 |
| 264 | C1 | 对 ckpt_sft_v3 重复施加同一条 eos_token_id 单字段补丁(json.load→改一键→json.dump,不是整份重写);结果 JSON 在 i=265 打印,max_new_tokens=2048 仍在。 | i=264, i=265 |
| 268 | C11 | 验证器工装升级:从官方日志算「超长未停」率——150 题里 41 题输出 >2000 字符、其中 39 题判错,把 '模型没停下来' 变成确定性信号。 | i=268, i=269 |
| 278 | C3 | 数据配方回退:丢掉 MetaMathQA,--data 换回纯 GSM8K 的 train_data.jsonl。 | i=278 |
| 278 | C4 | 超参:epochs 2.0→6.0、lr 1e-5→2e-5(与 c12 同一条命令,数据与超参一起变,构成混杂)。 | i=278 |
| 294 | C1 | 对 ckpt_sft_v4 重复同一条 eos_token_id 单字段补丁。 | i=294 |
| 298 | C9 | 提交守卫:删掉旧 final_model(v2, 0.480)并换成 ckpt_sft_v4(0.713),是本 run 唯一一次候选替换。 | i=298 |
| 304 | C3 | 数据配方:GSM8K + 20000 条 MetaMathQA GSM_Rephrased(只挑风格最接近 GSM8K 的一个子类,规模也比 c8 小一半)。 | i=304, i=307 |
| 308 | C4 | 超参:epochs 6.0→4.0(数据量变大时压 epoch)。 | i=308 |
| 323 | C1 | 对 ckpt_sft_v5 重复同一条 eos_token_id 单字段补丁。 | i=323 |
| 332 | C4 | epoch 扫描第 1 点:相对 v4 只把 epochs 6.0→8.0,数据/lr/bs/max_length/warmup/seed 全部逐字相同。 | i=332, i=329 |
| 341 | C1 | 对 ckpt_sft_v6 重复同一条 eos_token_id 单字段补丁。 | i=341 |
| 348 | C4 | epoch 扫描第 2 点:相对 v4 只把 epochs 6.0→5.0。 | i=348 |
| 368 | C1 | 对 ckpt_sft_v7 重复同一条 eos_token_id 单字段补丁。 | i=368 |
| 375 | C4 | epoch 扫描第 3 点:相对 v4 只把 epochs 6.0→7.0。 | i=375 |
| 380 | C1 | 对 ckpt_sft_v8 重复同一条 eos_token_id 单字段补丁。 | i=380 |
| 387 | C4 | lr 扫描:相对 v4 只把 lr 2e-5→1.5e-5,epochs 保持 6.0,其余逐字相同。 | i=387, i=385 |
| 392 | C1 | 对 ckpt_sft_v9 重复同一条 eos_token_id 单字段补丁(本 run 第 8 次)。 | i=392 |
| 399 | C9 | 提交守卫的复核:md5sum 逐分片确认 final_model 与 ckpt_sft_v4 权重逐字节一致,并复读 final_model 的 generation_config(eos 列表在、max_new_tokens 在);不改动任何产物。 | i=399, i=400, i=398 |
| 410 | proposed:knowledge_persi… | 把本 run 的获胜配方与两个坑写进 harness 的跨 session 记忆文件 MEMORY.md。它不改动 final_model、对本 run 分数为零效应,目标是后续 session 的分数——C1–C11 都预设改动作用于当前 run 的产物,装不下这一类。 | i=56, i=410 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 92 | real | 0.40h | consumed | **baseline** | baseline —— 本 run 第一次训练,无可比对象。全量 GSM8K 7473 条、3 epoch、lr 1e-5、bs 8×2。跑到底:i=131 目录里已出现 training_args.bin 等全部产物,随后被 i=133 的评测消费。 |
| 151 | real | 0.43h | consumed | **unclear** | 与 v1 的启动参数逐字相同(--epochs 3 --lr 1e-5 --batch_size 8 --grad_accum 2 --max_length 640)、数据文件也相同;唯一差异是 i=147 重写 train_sft.py 手工拼训练文本、去掉 <think> 标签。受测变量其实是 … |
| 229 | real | 1.55h | consumed | **both** | 数据 train_data.jsonl → train_data_v3.jsonl(7473 → 47473,加了 4 万条 MetaMathQA);同时超参 epochs 3→2.0、batch_size 8→16、新增 --warmup_ratio 0.03。数据与超参一起变。跑到底:i=257… |
| 278 | real | 0.62h | consumed | **both** | 数据回退到纯 GSM8K(train_data_v3.jsonl → train_data.jsonl);同时 epochs 2.0→6.0、lr 1e-5→2e-5。跑到底:i=291 打印 train_runtime 2109.4577 秒并 'Saved to'。 |
| 308 | real | 1.40h | consumed | **both** | 相对 v4:数据 train_data.jsonl → train_data_v5.jsonl(+2 万条 GSM_Rephrased);同时 epochs 6.0→4.0。lr/bs/max_length/warmup 不变。跑到底:i=321 打印 train_runtime 4923.4678… |
| 332 | real | 0.81h | consumed | **C4** | 相对上一次训练 v5:数据回到纯 GSM8K 且 epochs 4.0→8.0;但 agent 明示对照对象是 v4(i=329 'Try v6: v4 recipe with more epochs'),相对 v4 只有 epochs 6.0→8.0 一处不同,数据/lr/bs/grad_accu… |
| 348 | real | 0.63h | consumed | **C4** | epoch 扫描:相对 v6 为 8.0→5.0,相对基准 v4 为 6.0→5.0,其余启动参数逐字相同。跑到底:i=352 进度条 1170/1170,i=367 产物齐全。 |
| 375 | real | 0.75h | consumed | **C4** | epoch 扫描:相对 v7 为 5.0→7.0,相对基准 v4 为 6.0→7.0,其余逐字相同。跑到底:i=379 打印 train_runtime 2478.8227 秒。 |
| 387 | real | 0.67h | consumed | **C4** | lr 扫描:epochs 回到 6.0(与 v4 相同),lr 2e-5→1.5e-5,数据与其余超参逐字相同 —— 相对 v4 是严格单变量。跑到底:i=391 打印 train_runtime 2115.8908 秒。 |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 50 | 3.0 | 50.0 | 是 |  | 0.340 |
| 133 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.040 |
| 185 | 3.0 | 150.0 | 是 | c4 | 0.040 |
| 205 | 3.0 | 150.0 | 是 | c5 | 0.480 |
| 264 | 3.0 | 150.0 | 是 | c8, c9, c10 | 0.480 |
| 294 | 3.0 | 150.0 | 是 | c12, c13, c14 | 0.713 |
| 323 | 3.0 | 150.0 | 是 | c16, c17, c18 | 0.627 |
| 341 | 3.0 | 150.0 | 是 | c19, c20 | 0.607 |
| 368 | 3.0 | 150.0 | 是 | c21, c22 | 0.613 |
| 380 | 3.0 | 150.0 | 是 | c23, c24 | 0.680 |
| 392 | 3.0 | 150.0 | 是 | c25, c26 | 0.547 |
| 401 | 3.0 | 150.0 | 是 | c15, c27 | 0.700 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[151]
- **分类学缺口提案 1 条**
  - knowledge_persistence(i=56, i=410, i=412)
- **定义缺陷 2 条**
  - (i=185, i=186, i=186, i=206, i=196, i=199, i=200, i=205, i=206)
  - (i=92, i=155, i=291, i=391)
- **边界情形 3 条**
  - i=199 的 eos_token_id 改写按 C1 的字面定义(改 generation_config.json 的 eos)完全命中,但它的效应形状是 C8 式的二值修复而非 C1 式的连续解码调优:改之前管线并没有崩、也确实返回了数字(0.040),只是模型答对之后不停机、继续吐垃圾,正确答案被后续文本淹没;改之后 0.480,+44 点。这个量级是 reference §4.2 给 C1…(i=194, i=198, i=199, i=206, i=186)
  - --max_length 640 不是被测的超参,而是被数据的 token 长度分布机械决定的:i=87/88 先量出 7473 条样本 max=649、p99=503,i=90 才把默认值从 1024 降到 768、启动时(i=92)用 640。这正是 spec §10 第 2 条要写判据的那一类『数据一改就机械地逼着超参跟着改』。按现定义(C4 = lr/epoch/batch/序列长度)它算…(i=88, i=90, i=92)
  - i=151 的训练(v2)与上一次训练(v1)启动参数逐字相同、数据文件相同,唯一差异是 i=147 重写了训练文本的拼接方式(去掉 <think> 标签)。它的受测变量明确无误是 C2 格式对齐,证据充分,但 tested_variable 的取值域只有 C3/C4/both/unclear,装不下,只能填 unclear —— 这正是 spec §9 第 4 条要求分开报的『取值域装不下』与『…(i=147, i=151, i=92)

## claude_non_api_claude-opus-4-7_10h__gsm8k_google_gemma-3-4b-pt_17116632
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | google_gemma-3-4b-pt | 5.02h | 0.3570887035… |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 61 | C2 | prepare_data.py: 训练样本的 user 侧逐字复制 inspect_evals/gsm8k.py 的 MATH_PROMPT_TEMPLATE(含 'Reasoning:' 结尾),assistant 侧统一成 '<推理>\n\nANSWER: <数字>',剥掉 GSM8K 的 <<… | i=61, i=61, i=23 |
| 61 | C3 | v1 配方:GSM8K 官方 train split 上采样 3 倍(22,419)+ MetaMathQA 的 GSM* 子集随机抽 30,000,合计 52,419 行。 | i=61, i=64, i=64 |
| 66 | C4 | v1 方法:LoRA r=64 / alpha=128,只挂 7 个投影(q/k/v/o/gate/up/down),没有 modules_to_save,即 lm_head 与 embed_tokens 不训(reference §C4 的 LoRA 陷阱);SFT 且 prompt 段 labe… | i=66, i=66, i=66 |
| 77 | C8 | 启动前把 attn_implementation 由 eager 改成 sdpa(注释自承 eager 只是保险),纯吞吐/显存动机,不改学习目标。 | i=77, i=77 |
| 79 | C4 | 启动前 num_train_epochs 2→1、learning_rate 1.5e-4→2e-4(缩 epoch 同时抬 lr 作补偿)。 | i=79, i=79, i=79 |
| 84 | C4 | MAX_LEN 1024→800,依据是 i=81 现量的 token 长度分布(p99=604、max=832);属于被数据长度反推出来的序列长度设定。 | i=84, i=84, i=82 |
| 104 | C8 | 写 merge_and_save.py:把 LoRA adapter merge_and_unload 进 base 再 save_pretrained 到 final_model —— 没有它 vLLM 侧根本吃不到 adapter 目录。副作用是每次 merge 都重写一份 generation… | i=104, i=104 |
| 191 | C11 | 从官方 inspect_ai 日志里自造两个确定性信号:『输出里是否出现 Solve the following』(续写率)与『是否缺 ANSWER:』(截断率),另加输出长度分布。结果 24/150 续写、0 截断 —— 定位到问题不是截断而是不停下来。 | i=191, i=191, i=192 |
| 208 | C1 | generation_config.json:do_sample true→false,并 pop 掉 top_k(64) 与 top_p(0.95),但**没有写 temperature**。自称 greedy(输出文件名 eval_v1_greedy),实际是把 top_k/top_p 约束拿掉… | i=208, i=208, i=208 |
| 217 | C3 | v2 配方:GSM8K train 上采样 4 倍 + MetaMathQA GSM 25,000 + OpenMathInstruct-2 里 problem_source==augmented_gsm8k 的 20,000,合计 74,892 行。 | i=217, i=217, i=223 |
| 227 | C8 | 清盘:删 sft_out/checkpoint-500 与 -3000、整个 logs/、以及 final_model 的 safetensors 与 index。此时磁盘只用了 3%,并非真实压力;代价是 v1 的 merge 产物被删掉,后面 i=300 必须重新 merge(从而又把 gene… | i=227, i=226 |
| 231 | C4 | v2 方法:LoRA r 64→128、alpha 128→256、lr 2e-4→1.5e-4、MAX_LEN 800→768、最短长度过滤 20→30、save_steps 500→1000。 | i=231, i=231, i=231 |
| 254 | C1 | generation_config.json 在 merge 重置后的基础上补写 temperature=0.6(do_sample/top_k=64/top_p=0.95 本来就在),即从『无 temperature 的采样』改成『T=0.6 的采样』。 | i=254, i=254 |
| 268 | C1 | 只改 **config.json** 的 eos_token_id:1 → [1, 106],让 <end_of_turn> 也算停止符。这次改动完全不碰 generation_config.json,因此机械层的 config 表里没有它。 | i=268, i=268 |
| 273 | C11 | 在 c22 基础上加『一条输出里出现几次 ANSWER:』的计数器 —— 84/150 有多个 ANSWER,而 match(numeric=True, location=end) 取最后一个数,直接解释了掉分机制。这个判据零噪声、从同一次官方评测里免费得到。 | i=273, i=274 |
| 276 | C1 | 三处 eos 一起改:tokenizer_config.json 的 eos_token <eos>→<end_of_turn>、config.json 的 eos_token_id [1,106]→106、generation_config.json 的 eos_token_id 写成 [1,10… | i=276, i=276 |
| 281 | C1 | 把 c13/c14 全部回滚到 i=251 的状态:tokenizer eos 回 <eos>、config.json eos_token_id 回 1、generation_config 恢复 do_sample=True/top_k=64/top_p=0.95 并 pop 掉 temperatu… | i=281, i=281, i=281 |
| 300 | C9 | 不做任何新训练,只决定交哪一个:删掉装着 v2 的 final_model,把 v1 的 adapter 重新 merge 回 final_model(依据是 v1 0.367 > v2 0.327)。 | i=300 |
| 308 | C3 | v3 配方:GSM8K train 上采样 5 倍(37,365)+ MetaMathQA GSM 40,000 = 77,365 行,**去掉 OpenMathInstruct-2**(v2 掉分后的回退)。 | i=308, i=308, i=311 |
| 312 | C4 | v3 方法:彻底换掉 LoRA,改**全参微调**;lr 1.5e-4→5e-6、bs 8×accum2 → 4×accum4、optim adamw_torch→adamw_bnb_8bit,MAX_LEN 保持 768。 | i=312, i=312, i=312 |
| 337 | C9 | 把 v3 无条件覆盖进 final_model(rm -rf final_model 后 cp sft_out_v3 的权重),**在评测之前就换掉了当时手里最好的候选**。 | i=337 |
| 355 | C9 | v3 读到 0.040 后把 final_model 回滚成 v1(重新 merge sft_out)。这一步是本 run 最贵的守卫动作:不做的话交上去的就是 4% 的模型。 | i=355 |
| 371 | C9 | cp -r final_model final_model_v1_backup,给已确认最好的候选留一份安全副本。 | i=371 |

### 训练序列(3 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 87 | real | 4.87h | last_seen | **baseline** | baseline |
| 233 | real | 1.81h | discarded | **both** | 同时改了数据与方法,无法拆开:数据 52,419→74,892(GSM8K 3x→4x、MetaMath 30k→25k、新增 OpenMathInstruct-2 augmented_gsm8k 20k);方法 LoRA r 64→128、alpha 128→256、lr 2e-4→1.5e-4、… |
| 316 | real | 1.83h | last_seen | **both** | 相对 v2 同时改数据与方法:数据 74,892→77,365(GSM8K 4x→5x、MetaMath 25k→40k、删掉 OpenMath);方法从 LoRA 换成全参微调,lr 1.5e-4→5e-6、bs 8×2→4×4、optim adamw_torch→adamw_bnb_8bit。l… |

### 验证序列(10 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 179 | 3.0 | 150.0 | 是 | c1, c2, c3, c5, c6 | 0.367 |
| 210 | 3.0 | 150.0 | 是 | c8 | 0.327 |
| 251 | 3.0 | 150.0 | 是 | c10, c11 | 0.327 |
| 256 | 3.0 | 150.0 | 是 | c12 | 0.293 |
| 270 | 3.0 | 150.0 | 是 | c13 | 0.320 |
| 278 | 3.0 | 150.0 | 是 | c14 | 0.267 |
| 283 | 3.0 | 150.0 | 是 | c15 | 0.240 |
| 302 | 3.0 | 150.0 | 是 | c16 | 0.373 |
| 340 | 3.0 | 150.0 | 是 | c17, c18, c19 | 0.040 |
| 361 | 3.0 | 150.0 | 是 | c20 | 0.333 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - proposed:self_imposed_halt(i=383, i=383, i=376, i=399)
- **定义缺陷 4 条**
  - 骨架把三次训练全部记成 `run_end`(4.89h / 3.65h / 1.85h),但三次的结束在轨迹里都逐字可见,而且都有明确的产物消费点。反例:train_v2 在 i=247 打印了 `'train_runtime': 5188.9518` 并列出完整的 checkpoint-4681 与顶层 adapter,紧接着 i=249 就被 `merge_and_save.py sft_ou…(i=247, i=249, i=335, i=158)
  - 与轨迹不符。三段训练都被反复轮询:i=238 就是 `sleep 1700; ls sft_out_v2/ 2>/dev/null; tail -c 500 train_v2.log`,i=243/245 同类。真正的根因不是『没被提及』,而是**产物目录写在训练脚本内部**(i=231 的 `OUT_DIR = "sft_out_v2"`),启动命令行上没有任何产物名可供配对。但**日志文件名在…(i=238, i=231, i=233, i=316)
  - C1 被定义成『修改提交模型目录里的 generation_config.json』,连举的例子(eos_token_id 设成两个都接受)本身就可以发生在别的文件里。反例:i=268 只改 `final_model/config.json` 的 `eos_token_id`(1 → [1,106]),完全不碰 generation_config.json,却由 i=270 一次独立评测单独判定(…(i=268, i=270, i=276)
  - 本 run 给出反例。i=168 的 merge_and_save.py 走的正是 `save_pretrained`,而 i=189 把生成出来的 final_model/generation_config.json 逐字打印出来,里面 `"do_sample": true` 明明白白在。同时 i=43 列出的 base 快照根本**没有 generation_config.json 这个文件*…(i=189, i=43, i=208)
- **边界情形 3 条**
  - MAX_LEN 1024→800 是在 i=81 实测完 token 长度分布(p99=604、max=832)之后立刻做的,目的是『别让上限白占显存』,即由数据长度反推的补偿量,而不是一个被检验的超参。按现有定义它是 C4,按 §10-2 的提案判据它不该计入受测变量 —— 规则没写死之前判不了。(i=82, i=84)
  - attn_implementation eager→sdpa:动机纯是吞吐(注释自承 eager 只是保险选项),按 C8『让管线跑得起来/跑得动』;但它确实换掉了一个 TrainingArguments 之外的训练侧实现选项,并会微扰数值。C8 与 C4 的现有定义都没说注意力核实现算哪边。(i=77, i=77)
  - C9 定义为『不改任何产物,只决定此刻把哪个已有候选写进 final_model』。i=337 确实是这样一个决定,但它是**无条件覆盖**:v3 一个字的分数都还没读到,就把当时手里最好的 v1 顶掉了(v1 的 merge 产物早在 i=227 被删,只剩 adapter)。同一个动作,按正向定义是一次 C9,按 §4.4 又正是『缺回归守卫』这一扣分项本身。现定义无法区分『做了守卫』与『做了…(i=337, i=342, i=227)

## claude_non_api_claude-opus-4-7_10h_run2__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17126494
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 8.94h | 0.6626231993… |

### 改动序列(26 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 75 | C2 | 把 inspect_evals/gsm8k 的 MATH_PROMPT_TEMPLATE 逐字抄进 prepare_data.py 作为训练样本的 user 侧，assistant 侧统一以 "ANSWER: {target}" 结尾，与官方 match 评分器读的那一行对齐。 | i=38, i=75, i=75 |
| 75 | C3 | 初始数据配方：GSM8K 官方 train split（7,473）+ MetaMathQA 的 GSM_* 四个子类（每类上限 15,000），显式排除 MATH_* 子类；答案统一归一化成整数串。 | i=75, i=75, i=78, i=78 |
| 75 | C10 | 去污染：先看一眼 test split 的首题「to know what to avoid」，然后在配方里只取 MetaMathQA 的 GSM_* 子类并在代码注释里写明其派生自 GSM8K train；全程未触碰 test split。 | i=73, i=75 |
| 88 | C2 | 结束符与模板对齐：把评测用的 templates/smollm.jinja 装进训练 tokenizer，eos 改成 <\|im_end\|>(128012)、pad 改成 <\|end_of_text\|>(128001)，模型 config 与 generation_config 的 eos … | i=88, i=88, i=88 |
| 88 | C4 | 训练方法基线：TRL SFTTrainer 全参 SFT（无 LoRA）、bf16、cosine 调度、flash_attention_2、adamw_torch_fused、completion_only_loss。整条 run 未用过 LoRA，因此没踩 reference §C4 的 lm_h… | i=88, i=88, i=88 |
| 106 | C8 | 在观察到显存占用为 0 MiB 后，给 train_sft.py 加 --no_grad_ckpt 开关，把 gradient_checkpointing 变成可关，用显存换吞吐。此改动数学上不改变被训函数，只买速度。（见 boundary_case b1：C4/C8 两条定义都不严格成立。） | i=104, i=106, i=103 |
| 111 | C8 | OOM 回退：bs16/accum1 在 loss.float() 处爆显存后，退回 bs8/accum2 并保留 --no_grad_ckpt，这一组合成为此后全部 6 次真实训练的固定配置。 | i=108, i=109, i=111 |
| 121 | C4 | 首次真实训练的超参：lr 1e-5（脚本默认 5e-6）、1 epoch、warmup_ratio 0.03；max_seq_len 1024 由一次 token 长度统计定出（2000 样本里超 1024 的只有 0.05%）。 | i=115, i=121, i=121 |
| 140 | C11 | 验证器工装 run_eval.sh：把 evaluate.py 的 --json-output-file 结果 cat 出来，保证 accuracy 一定回到轨迹里（本 run 11/11 次评测都拿到了分数）；同时把 --max-connections 提到 4、--gpu-memory-util… | i=140, i=140 |
| 159 | C11 | 自写脚本读 inspect_ai 的 json 日志，把官方评分器的逐样本 scores.match.value 拆成对/错两桶并打印错题原文，得到确定性的失败模式清单（81 对 / 69 错）。 | i=159, i=160 |
| 169 | C3 | 只放大数据量：MetaMathQA GSM_* 每类上限 15,000 → 80,000，train_data.jsonl 由 67,473 行涨到 247,473 行；训练超参一个没动。 | i=167, i=169, i=172 |
| 197 | C11 | 从官方日志里算格式合规率：统计错题里「输出不含 ANSWER:」的条数，得到 0，据此排除 C2 格式失配、把剩余失败全部归到算术推理上——直接决定了下一步转向拒绝采样而不是继续改模板。 | i=197, i=198 |
| 206 | C3 | 引入自生成 + 验证过滤的数据来源（reference §C3 的 (d) 类）：用 sft_out_v2 对全部 7,473 道 GSM8K train 题各采 8 条（temp 0.7 / top_p 0.95 / stop 128012,128001），只保留归一化后答案与 ground tr… | i=206, i=206, i=213 |
| 216 | C3 | v3 混合比例：52,318 条 RFT 数据 + 从 247k 旧池里随机抽的 50,000 行，合计 102,318。 | i=216, i=221 |
| 224 | C4 | v3 的方法侧改动：初始化从 base 换成 sft_out_v2 热启动，lr 由 1e-5 降到 5e-6。 | i=224, i=224 |
| 248 | C3 | 拒绝采样第二轮：换成用更强的 sft_out_v3 当生成器，温度由 0.7 提到 0.8，同样 8 条/题，保留 53,381 条。 | i=248, i=251 |
| 255 | C3 | v4 混合比例：53,381 条 RFT_v3 + 20,000 行旧池（多样性配额由 50,000 砍到 20,000），合计 73,381。 | i=255, i=258 |
| 259 | C4 | v4 的方法侧改动：热启动源由 sft_out_v2 换成 sft_out_v3，lr 再降到 2e-6。 | i=259, i=259 |
| 274 | C3 | v5 混合方案第一版：RFT_v3 全量 + RFT_v2 抽 25,000 + 旧池 15,000 = 93,381。此文件在被任何训练消费之前就被 i=282 整份覆盖，因此从未受过验证。 | i=274, i=277 |
| 282 | C3 | v5 混合方案第二版（覆盖第一版）：RFT_v3 全量 + 官方 GSM8K train 人写解答按 2 倍权重复制 + 旧池 30,000 = 98,327；意图是提高人写 canonical 解答的占比。 | i=282, i=285 |
| 286 | C4 | v5 的方法侧改动：去掉 --model，改为从 base 冷启动（前三次真实训练都是热启动），lr 回到 1e-5。agent 自陈意图是「train from base on pure RFT data」。 | i=281, i=286, i=286 |
| 305 | C9 | 提交守卫：在还剩 2:25 时先把当时的最优候选 sft_out_v3 整份拷成 final_model，然后才去跑最后一次实验；不改任何产物，只是把已有候选钉进提交位。 | i=304, i=305 |
| 316 | C3 | v6 混合比例：RFT_v2 + RFT_v3 两轮拒绝采样数据全量 + 官方 canonical GSM8K，合计 113,172，不再掺 MetaMathQA。 | i=316, i=316 |
| 318 | C4 | v6 的方法侧改动：回到从 sft_out_v3 热启动（放弃 v5 的冷启动），lr 降到 1e-6。 | i=318, i=318 |
| 338 | C8 | 删掉 sft_out_v1/v2/v4/v5/v6 五个候选目录，只留 sft_out_v3 与 final_model。字面上命中 C8 的「清磁盘」，但当时没有磁盘压力，且此举不可逆地销毁了除提交候选外的全部候选。（见 boundary_case b2。） | i=338, i=339, i=26 |
| 346 | proposed:cross-run-memor… | 跨 run 知识持久化：开局先去 ~/.claude/projects/-home-ben-task/memory/ 找先验（空），收尾把配方、显存边界、eos/模板要点、以及自测到的评测噪声写进 MEMORY.md。收益不落在本 run 的分数上。（见 proposed_category p2。） | i=18, i=346, i=346 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 96 | smoke | 0.02h | returned | **smoke** | baseline —— 本 run 第一次训练启动，是管线冒烟（max_samples 100、bs2/accum2、300 秒 timeout）。它既不验数据配方也不验超参，只验代码跑不跑得通；tested_variable 取值域没有 smoke/baseline，故填 unclear（属取值域… |
| 99 | smoke | 0.02h | returned | **smoke** | 冒烟第 2 次：max_samples 100→200、batch_size 2→8、新增 --max_seq_len 1024、logging_steps 5→2、save_steps 1000→10000、timeout 300→180。目的是探显存与吞吐上限，不是验配方或超参对分数的影响；取值… |
| 108 | smoke | 0.01h | returned | **smoke** | 冒烟第 3 次：batch_size 8→16、grad_accum 2→1、新增 --no_grad_ckpt、logging_steps 2→1。这是显存上限探测，结局是 OOM（在 ForCausalLMLoss 的 logits.float() 处要 6.90 GiB 时失败）；取值域不足，… |
| 111 | smoke | 0.02h | returned | **smoke** | 冒烟第 4 次：batch_size 16→8、grad_accum 1→2（对上一次 OOM 的回退），保留 --no_grad_ckpt，其余与上次相同。确认 bs8/accum2/no_grad_ckpt 可跑；取值域不足，填 unclear。 |
| 121 | real | 0.79h | consumed | **baseline** | baseline —— 首次真实训练。数据 train_data.jsonl（67,473 行）与超参（bs8/accum2/1 epoch/lr 1e-5/max_seq_len 1024/warmup 0.03）在这里同时第一次成立，没有对照臂可比，因此不构成对 C3 或 C4 任何一项的检验；… |
| 174 | real | 2.74h | consumed | **C3** | 只有数据变：train_data.jsonl 由 67,473 行重建为 247,473 行（MetaMathQA GSM_* 每类上限 15,000→80,000）。训练侧逐字相同：--batch_size 8 --grad_accum 2 --epochs 1 --lr 1e-5 --max_s… |
| 224 | real | 1.10h | consumed | **both** | 数据与方法同时变。数据：换成 train_data_v3.jsonl（52,318 条 v2 自生成并按答案验证过滤的 RFT 数据 + 50,000 行旧池 = 102,318），来源类别从「官方 train + 公开蒸馏集」变成「自生成 + 验证过滤」。方法：初始化由 base 改为 sft_o… |
| 259 | real | 0.80h | consumed | **both** | 数据：换成 train_data_v4.jsonl（53,381 条由 sft_out_v3 在 temp 0.8 下重新生成的 RFT 数据 + 20,000 行旧池 = 73,381，多样性配额 50k→20k）。方法：热启动源 sft_out_v2→sft_out_v3，lr 5e-6→2e-… |
| 286 | real | 1.07h | consumed | **both** | 数据：换成 train_data_v5.jsonl（RFT_v3 53,381 + 官方 GSM8K train 人写解答 ×2 倍权重 + 旧池 30,000 = 98,327）。方法：去掉 --model，从 base 冷启动（前三次真实训练均为热启动），lr 2e-6→1e-5。agent 明… |
| 318 | real | 1.14h | consumed | **both** | 数据：换成 train_data_v6.jsonl（RFT_v2 + RFT_v3 + 官方 canonical = 113,172，彻底不掺 MetaMathQA）。方法：放弃冷启动，回到从 sft_out_v3 热启动，lr 1e-5→1e-6。真实结局：正常跑完并落盘，train_runtim… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 153 | 3.0 | 150.0 | 是 | c1, c2, c4, c5, c8 | 0.54 |
| 194 | 3.0 | 150.0 | 是 | c11 | 0.58 |
| 238 | 3.0 | 150.0 | 是 | c13, c14, c15 | 0.7066666666666667 |
| 245 | 3.0 | 500.0 | 是 | c13, c14, c15 | 0.716 |
| 265 | 3.0 | 500.0 | 是 | c16, c17, c18 | 0.708 |
| 295 | 3.0 | 500.0 | 是 | c20, c21 | 0.596 |
| 307 | 3.0 | 150.0 | 是 | c22 | 0.6266666666666667 |
| 310 | 3.0 | 150.0 | 是 |  | 0.68 |
| 313 | 3.0 | 500.0 | 是 | c22 | 0.692 |
| 328 | 3.0 | 500.0 | 是 | c23, c24 | 0.68 |
| 335 | 3.0 | 150.0 | 是 | c22 | 0.6733333333333333 |

### 异常与存疑

- **分类学缺口提案 2 条**
  - verifier-cost-tuning(i=140, i=9)
  - cross-run-memory(i=18, i=346, i=345)
- **定义缺陷 2 条**
  - (i=245, i=246, i=154)
  - (i=96, i=97, i=97)
- **边界情形 4 条**
  - (i=104, i=106, i=97)
  - (i=26, i=338, i=339)
  - (i=96, i=111, i=121)
  - (i=87, i=341, i=88, i=88)

## claude_non_api_claude-opus-4-7_10h_run2__gsm8k_Qwen_Qwen3-1.7B-Base_17124733
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 9.64h | 0.6656557998… |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 61 | C2 | 训练样本改用评测端逐字相同的提示词模板与 ANSWER: X 结尾标记,并剥掉 GSM8K 答案里的 <<...>> 计算器标注,把 #### 换成 ANSWER: | i=61, i=61, i=67 |
| 61 | C3 | 数据来源定为 GSM8K 官方 train(7,473)+ MetaMathQA 的 GSM_AnsAug / GSM_Rephrased 两类共 160,000 行,合计 167,473 | i=61, i=64 |
| 69 | C4 | 训练脚本 train.py:全参 SFT、prompt 掩码只对 assistant 段计损失、bf16 + flash_attention_2、lr 1e-5 cosine;手工拼 Qwen3 chat 模板而不调 apply_chat_template | i=69, i=69 |
| 81 | C8 | 加 --no_gc 开关关掉梯度检查点换吞吐(冒烟测出速度瓶颈后);此后每次真实训练都带该开关 | i=79, i=81, i=83 |
| 176 | C1 | 整份重写 sft_v1 的 generation_config.json:eos_token_id 151643 -> [151645, 151643](接受 <\|im_end\|>),新增 pad_token_id;同时把 tokenizer_config.json 的 eos_token 改成… | i=176, i=176, i=172, i=179, i=178 |
| 203 | C11 | 自写脚本从官方 inspect_ai 日志里抽两个确定性判据:第一处 ANSWER 的正确数 vs 官方 match(location=end) 的得分数(90 vs 21),以及 stop_reason 分布(stop 107 / max_tokens 43)。输入是官方评分器自己的输出,不构成代… | i=203, i=204, i=217, i=220 |
| 230 | C2 | 训练样本前面加 K 条 few-shot 的 system 段,复刻评测端 system(10-shot)/ user / assistant 三条消息的布局;few-shot 用评测格式渲染并保留 <<>> 标注 | i=230, i=230, i=201 |
| 230 | C3 | 配方 v2:GSM8K + MetaMath(GSM 两类)合池后随机取 60,000 条;MetaMath 答案加纯数值正则过滤 | i=230, i=233 |
| 235 | C1 | 把 generation_config 修复烧进训练脚本 train_v2.py:每次 save_model 之后自动写 eos_token_id [151645,151643],使该 C1 缺陷不会随每个新 checkpoint 复发。但它只复刻了 fix_generation_config.py… | i=235, i=345, i=615, i=176 |
| 235 | C2 | train_v2.py 支持 system 段并按 <\|im_start\|>system ... <\|im_end\|> 渲染;超长时从左截断(先丢 system/few-shot)以保住 assistant 标签 | i=235, i=235 |
| 270 | C3 | few-shot 条数分布三次调参(10 重 -> 6 重 -> 10 重 -> 最终 [0,2,4,5,6,8,8,8]),目的是把 p99 token 长度压到 max_len 2048 以下(最终 p99=2190、超长 173/5000) | i=243, i=259, i=270, i=273 |
| 277 | C4 | v2 超参:max_len 640->2048、bsz 16->4、grad_accum 2->4、epochs 2->1、subset 80000->60000。其中 max_len 抬高是 few-shot system 段把中位长度从 274 推到 1568 token 之后的被迫补偿 | i=277, i=74, i=238 |
| 291 | C8 | harness 丢了后台任务句柄(No task found with ID)后改用 disown 重启 v2 训练;ps 显示原进程 1698073 仍在跑,遂 kill 掉刚起的重复进程 1699086,避免两份训练抢同一张卡 | i=290, i=291, i=294, i=298 |
| 371 | C4 | v3 相对 v2 只把 epochs 1 改成 2:同一份 train_data_v2.jsonl(mtime 停在两次启动之前),max_len/bsz/grad_accum/lr/--no_gc 逐字相同,--subset 80000 对 60,000 行文件是空操作,步数 3750->7500… | i=371, i=277, i=470, i=407 |
| 371 | C8 | sft_v3 首次启动(i=365)因工作目录不对当场夭折(/proc/1798935 不存在、train_v3.log 从未创建),改为先 cd 到 work 目录再 nohup 重启 | i=370, i=371 |
| 448 | C8 | 中止 OpenMathInstruct-2(1,397 万行)的下载并删掉已落盘的 12G 缓存,止损磁盘与时间预算 | i=448, i=449 |
| 451 | C3 | 自生成 + 验证过滤:用 sft_v3 对 GSM8K train 每题采样 4 条(T=0.7),只留数值与金标准一致的,并把末行 ANSWER 重写成金标准,产出 25,136 条 | i=451, i=453, i=471 |
| 476 | C3 | 配方 v4:GSM8K 7,473 + MetaMath 30,000 + 拒绝采样 25,136 x2 = 87,745;few-shot 分布上调到 [0,2,4,6,8,8,10,10] | i=476, i=479 |
| 481 | C4 | v4 超参:max_len 2048->3072、grad_accum 4->8(有效 batch 16->32)、epochs 2->1.0、--subset 70000 | i=481, i=481 |
| 548 | C9 | 把 sft_v3 拷进 final_model —— 不产出任何新权重,只在 v4(0.675@200)劣于 v3(0.693@150)之后决定此刻交哪一个 | i=547, i=548 |
| 579 | C3 | 配方 v5:去掉 MetaMath,只用 GSM8K x2 + 拒绝采样 x1 = 40,082;few-shot 分布改成 10 重 | i=579, i=579, i=582 |
| 628 | C4 | v6:初始权重从 base 改成 sft_v3(续训)、lr 1e-5->3e-6、max_len 3072->2048、v5 数据只取 20,000 条 | i=628, i=628 |
| 658 | C9 | v6 在 500 题上 0.602,低于 v3 的 0.672,守卫生效:不覆盖 final_model,保留 v3 | i=658, i=656 |

### 训练序列(16 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 76 | smoke | 0.01h | returned | **smoke** | baseline —— train.py 的第一次冒烟(200 样本 / max_len 640 / bsz 8 x accum 2),验代码跑不跑得通,不测配方也不测方法。四值域里没有 smoke 这个取值,故填 unclear |
| 85 | smoke | 0.01h | returned | **smoke** | 对 i=76:bsz 8->16、grad_accum 2->1、新增 --no_gc(关梯度检查点)。只为量吞吐与显存(i=88 紧接着查 nvidia-smi),没有读任何分数 |
| 96 | real | 0.75h | consumed | **baseline** | baseline —— 第一次真实训练:train_data.jsonl 取 80,000 条,max_len 640、bsz 16 x accum 2、epochs 2、lr 1e-5、--no_gc。没有可比的前一次,受测变量无从定义。结局:07:31 起、08:16 完成落盘(i=156 打印… |
| 240 | smoke | 0.01h | returned | **smoke** | 换到 train_v2.py + train_data_v2.jsonl 之后的第一次冒烟:max_len 3072、bsz 2、grad_accum 16。冒烟,不测受测变量 |
| 248 | smoke | 0.02h | returned | **smoke** | 对 i=240:max_len 3072->2048、bsz 2->4、grad_accum 16->4、加 --no_gc。冒烟,在扫显存/吞吐可行域 |
| 253 | smoke | 0.00h | returned | **smoke** | 对 i=248:bsz 4->8、grad_accum 4->2。真实结局是当场 CUDA OOM(i=254 的 loss.backward traceback,0/8 步就挂),这一格被排除 |
| 256 | smoke | 0.03h | returned | **smoke** | 对 i=253:退回 bsz 4 / accum 4,并去掉 --no_gc(重新开梯度检查点)—— 量开 GC 的代价,耗时从 15s 涨到 96s |
| 264 | smoke | 0.01h | returned | **smoke** | 对 i=256:max_len 2048->2560、重新加 --no_gc(此时 k_choices 刚被调成 10 重,序列变长) |
| 267 | smoke | 0.01h | returned | **smoke** | 对 i=264:max_len 2560->2048、bsz 4->8、去掉 --no_gc —— 与 i=253 同 bsz 8 但开着 GC,这次没 OOM,确认 bsz 8 只有在开 GC 时可行 |
| 277 | real | 0.03h | superseded | **both** | 对 sft_v1:数据 train_data.jsonl -> train_data_v2.jsonl(加 K 条 few-shot 的 system 段、60,000 条、MetaMath 数值过滤),同时 max_len 640->2048、bsz 16->4、grad_accum 2->4、e… |
| 291 | real | 1.24h | consumed | **unclear** | 与 i=277 逐字同参的重复启动(仅 nohup 换成 disown),动机是 harness 丢了后台任务句柄。起来的 PID 1699086 在 12 秒后被 i=294 的 kill 杀掉,**没有产出任何权重**;骨架把 1.24h 记在这一行是错的(那 1.24h 属于 i=277) |
| 365 | real | 0.01h | superseded | **C4** | 对 sft_v2:唯一实质变化是 epochs 1->2(--subset 80000 对 60,000 行文件是空操作;--model 写死的值正是脚本默认值;--data 未给,仍是 train_data_v2.jsonl)。真实结局:当场夭折,i=369/370 显示 /proc/179893… |
| 371 | real | 2.44h | consumed | **C4** | i=365 的重启(加了 cd 到 work 目录)。相对 sft_v2 只改 epochs 1->2:train_data_v2.jsonl 的 mtime 停在 10:38(i=470),早于 v2/v3 两次启动,两次训练读到的是同一份字节;max_len/bsz/grad_accum/lr/… |
| 481 | real | 1.75h | consumed | **both** | 对 sft_v3:数据换成 train_data_v4.jsonl(GSM8K 7,473 + MetaMath 30,000 + 拒绝采样 25,136 x2 = 87,745,取 70,000),同时 max_len 2048->3072、grad_accum 4->8(有效 batch 16-… |
| 583 | real | 0.09h | last_seen | **C3** | 对 sft_v4:**超参逐字相同**(max_len 3072 / bsz 4 / grad_accum 8 / lr 1e-5 / epochs 1.0 / --no_gc / 同一个 base),只换数据 —— train_data_v4.jsonl(87,745 取 70,000)-> tr… |
| 628 | real | 0.62h | consumed | **both** | 对 sft_v5(未产出)/ sft_v3(在位冠军):初始权重从 base 改成 /home/ben/task/work/sft_v3(续训)、lr 1e-5->3e-6、max_len 3072->2048(方法侧);同一份 train_data_v5.jsonl 只取 20,000 条(数据侧… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 160 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.060 |
| 180 | 3.0 | 150.0 | 是 | c5 | 0.140 |
| 348 | 3.0 | 150.0 | 是 | c7, c8, c9, c10, c11, c12 | 0.673 |
| 433 | 3.0 | 150.0 | 是 | c23 | 0.693 |
| 539 | 3.0 | 200.0 | 是 | c16, c17, c18 | 0.675 |
| 560 | 3.0 | 500.0 | 是 | c23 | 0.672 |
| 569 | 3.0 | 500.0 | 是 | c16, c17, c18 | 0.656 |
| 617 | 3.0 | 200.0 | 是 | c19 | 0.665 |
| 650 | 3.0 | 500.0 | 是 | c20, c21 | 0.602 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[291]
- **定义缺陷 2 条**
  - (i=292, i=294, i=298, i=306)
  - (i=617, i=623, i=658)
- **边界情形 3 条**
  - (i=81, i=83, i=256)
  - (i=74, i=238, i=277, i=273)
  - (i=254, i=290, i=294)

## claude_non_api_claude-opus-4-7_10h_run2__gsm8k_Qwen_Qwen3-4B-Base_17124526
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 6.55h | 0.8180439727… |

### 改动序列(24 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 51 | C2 | 首版 prepare_data.py:训练样本的 user 侧逐字抄评测的 10-shot 提示模板(含 "ANSWER: $ANSWER" 与 "Reasoning:" 收尾),assistant 侧统一成「推理 + 空行 + ANSWER: <数字>」,并用正则剥掉 GSM8K 答案里的 <<.… | i=51, i=51 |
| 51 | C3 | 同一次 Write 里定下的数据来源:GSM8K 官方 train split(7,473,来源 a)+ MetaMathQA 中所有 GSM_* 类型(240,000,来源 b),合计 247,473 条。 | i=51, i=54 |
| 59 | C3 | 重写 prepare_data.py 收紧配方:GSM8K train 上采样 ×3(7,473→22,419),MetaMathQA 只留 GSM_AnsAug 与 GSM_Rephrased 各随机抽 20,000(丢弃 GSM_FOBAR / GSM_SV 与全部 MATH_*),random… | i=59, i=59, i=62 |
| 67 | C4 | 首版 train.py:全参 SFT(不用 LoRA)、bf16 + flash_attention_2、cosine + warmup 0.03、gradient_checkpointing、max_length 1024、packing=False、completion_only_loss=Tr… | i=67, i=67 |
| 80 | C4 | 启动第一次真实训练前改 train.py 的默认超参:epochs 2.0→1.0、lr 5e-6→1e-5、bsz 4→8、grad-accum 4→2(有效 batch 16 不变)。 | i=80, i=80 |
| 88 | C11 | 写 run_eval.sh,把 evaluate.py 调用与 `cat "$OUTFILE"` 包在一起,让分数直接回到轨迹而不必再去翻 inspect_ai 日志。实测**全程从未被调用**——除了 i=90 的 chmod +x,再没出现过。 | i=88, i=90 |
| 146 | C1 | sft-v1 的 config.json 与 generation_config.json:eos_token_id 由 [151643](<\|endoftext\|>,base 模型继承下来的)改成 151645(<\|im_end\|>,chat 模板真正用的结束符)。写法是 load→改单字… | i=136, i=137, i=146 |
| 148 | proposed:verifier-config | 评测预算 --max-tokens 2048 → 1536,与 c7 的 eos 修复同一批发生。后果:i=123→i=148 的 0.040→0.080 这一步不是单变量对比 —— 轨迹里没有第二次 2048 的读数可以把两个因素分开。 | i=123, i=148 |
| 169 | C1 | sft-v1 的 tokenizer_config.json:eos_token 由 '<\|endoftext\|>' 改成 '<\|im_end\|>'。这是 i=146 漏掉的第三个文件——上一次评测(0.080)之所以仍然不停,是因为 tokenizer 侧还写着旧结束符。 | i=167, i=169 |
| 201 | C1 | 给 sft-v1 的 generation_config.json 追加 temperature=0.0 / top_p=1.0 / top_k=-1 / do_sample=False(贪婪解码)。动机是先读了 inspect_ai 的 GenerateConfig,确认 evaluate.py … | i=198, i=201, i=202 |
| 215 | C2 | 新建 simple_template.jinja:去掉 qwen3.jinja 在渲染真实 assistant 轮次时插入的空 <think>\n\n</think>\n\n。i=218 的第一档比对确认训练渲染与评测的 add_generation_prompt 输出此后完全对称。 | i=215, i=217 |
| 220 | C2 | 把 train.py 用的 chat_template 从 templates/qwen3.jinja 换成 simple_template.jinja。**这是 sft-v1 与 sft-v2 之间 train.py 的唯一一处改动**,两次启动命令逐字相同、数据集未重建、seed 固定 42。 | i=220, i=220 |
| 252 | C1 | 对 sft-v2 施加与 v1 完全相同的三文件 eos 修复(config.json / generation_config.json 的 eos_token_id=151645,tokenizer_config.json 的 eos_token='<\|im_end\|>'),这次一条命令做完,… | i=252, i=252 |
| 255 | proposed:verifier-config | 评测预算 --max-tokens 1536 → 1024,与 c11(换模板重训)、c12(v2 eos 修复)同一批发生。此后所有评测都用 1024,口径才稳定下来。i=422 事后验证训练样本 p99 只有 445 token、frac > 1024 为 0。 | i=255, i=422 |
| 266 | C1 | 给 sft-v2 的 generation_config.json 设贪婪解码(同一组四字段)。这一步与前一次评测之间没有任何其他改动,构成本 run 最干净的一次 C1 内部对照。 | i=266, i=267 |
| 275 | proposed:verifier-config | 全量评测把 --max-connections 从 2 提到 4(仅此一次);i=358 的 v3 全量又用回 2。**决定提交内容的那次对比(v2 0.823 vs v3 0.807)因此在并发度上不对齐**——reference §2 实测同一份权重仅改 --max-connections(8 … | i=275, i=358 |
| 284 | C11 | 自写失败分析脚本,读官方 inspect_ai 日志把 234 条失败拆成 wrong_math / truncated / format_err 三档(判据:有没有 ANSWER: 行、内容是否 >900 字符)。结论截断率 0——排除了「加大生成预算能提分」这条路,是确定性的、零 GPU 的判据… | i=284, i=285 |
| 294 | C4 | 训练 sft-v3:epochs 1 → 2,lr / bsz / grad-accum / 数据全部与 v2 逐字相同。 | i=294, i=293 |
| 335 | C1 | 对 sft-v3 一次性施加 eos 修复 + 贪婪解码(自此三处 config 合并成一条命令,v4/v5 沿用)。i=348 打印出写后全文,四个新字段就位且 max_new_tokens: 2048 仍在。 | i=335, i=348 |
| 361 | C9 | 比较两次全量分(v2 82.3% vs v3 80.7%)后,rm -rf final_model 再把 checkpoints/sft-v2 整目录拷成 final_model。有比较、有依据,但覆盖是无条件的(先 rm 再 cp,中间没有回归守卫)。此后 v4(39.3%)、v5(78.0%)都… | i=361, i=362 |
| 378 | C4 | 训练 sft-v4:lr 1e-5 → 5e-6,其余与 v2 逐字相同(epochs 1.0、bsz 8、grad-accum 2、同一份 62,419 行数据)。 | i=378, i=373 |
| 388 | C1 | 对 sft-v4 施加 eos + 贪婪解码(同 i=335 的合并写法)。 | i=388 |
| 408 | C4 | 训练 sft-v5:lr 5e-6 → 1.5e-5(相对 v2 的 1e-5 上调),其余不变。这是 lr 的第三个取值,v2/v4/v5 构成 1e-5 / 5e-6 / 1.5e-5 的单变量 lr 扫。 | i=408 |
| 412 | C1 | 对 sft-v5 施加 eos + 贪婪解码,并在同一条命令里 && 接上 150 题评测。 | i=412 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 74 | smoke | 0.01h | returned | **smoke** | baseline —— 冒烟。`--subsample 64` + `timeout 300`,只验证 train.py 能加载模型、分词、走完 8 个 optimizer step 并落盘,不测任何受测变量。i=75 打印 train_runtime 7.52 秒 + DONE,通过;产物在 i=… |
| 82 | real | 0.82h | consumed | **baseline** | baseline —— 第一次全量训练,建立起点。相对冒烟只是从 64 条放大到 62,419 条并换上 i=80 的默认超参(epochs 1、lr 1e-5、bsz 8、accum 2),没有可比对象。结局:正常跑完,i=118 打印 train_runtime 2791.58s + DONE(… |
| 224 | real | 0.80h | consumed | **unclear** | **受测变量是 C2(格式对齐),而 tested_variable 的取值域只有 C3/C4/both/unclear,装不下。** 启动命令与 v1 逐字相同(`--epochs 1 --lr 1e-5 --bsz 8 --grad-accum 2`),data/sft 未重建(i=61 生成后… |
| 294 | real | 1.60h | consumed | **C4** | 相对 v2 只改 `--epochs 1` → `--epochs 2`,lr / bsz / grad-accum / 数据全部逐字相同。步数 3902 → 7804,train_runtime 2771s → 5563s(恰好两倍),意图在 i=293 写明「train longer (2 ep… |
| 378 | real | 0.86h | consumed | **C4** | 相对 v2 只改 lr 1e-5 → 5e-6(v3 的 2 epoch 已被 i=358 的全量 0.807 否掉,基准回到 v2)。数据同一份 62,419 行(i=373/374 当场复核)。结局:正常跑完,i=387 打印 train_runtime 2781.89s + DONE。 |
| 408 | real | 0.84h | consumed | **C4** | 相对 v4 把 lr 从 5e-6 提到 1.5e-5(相对 v2 的 1e-5 上调),其余不变;v2/v4/v5 合起来是 1e-5 / 5e-6 / 1.5e-5 的单变量 lr 扫。启动前 i=405 用 diff 逐字确认 v2 与 v4 的 chat_template / config.… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 123 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5 | 0.040 |
| 148 | 3.0 | 150.0 | 是 | c7, c22 | 0.080 |
| 171 | 3.0 | 150.0 | 是 | c8 | 0.060 |
| 203 | 3.0 | 150.0 | 是 | c9 | 0.013 |
| 255 | 3.0 | 150.0 | 是 | c11, c12, c23 | 0.233 |
| 268 | 3.0 | 150.0 | 是 | c13 | 0.820 |
| 275 | 4.0 | -1.0 | 是 | c11, c13, c24 | 0.823 |
| 350 | 3.0 | 150.0 | 是 | c15, c16 | 0.813 |
| 358 | 4.0 | -1.0 | 是 | c15 | 0.807 |
| 390 | 3.0 | 150.0 | 是 | c18, c19 | 0.393 |
| 412 | 3.0 | 150.0 | 是 | c20, c21 | 0.780 |
| 418 | 3.0 | 150.0 | 是 | c17 | 0.813 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[224]
- **分类学缺口提案 1 条**
  - verifier-config(i=123, i=148, i=255, i=275, i=358)
- **定义缺陷 3 条**
  - (i=82, i=224, i=220, i=209, i=272, i=85, i=99, i=230, i=212)
  - (i=266, i=259, i=272, i=201, i=175, i=209)
  - (i=202, i=267, i=348, i=417)
- **边界情形 3 条**
  - sft-v2 相对 sft-v1 的唯一变量是 chat 模板(i=220),这按 §3 是 **C2 格式对齐**。但 `tested_variable` 只允许 C3 / C4 / both / unclear —— C2 不在域内。填 C3 会把「改了数据来源或混比」说成事实(数据一行没变),填 C4 更错,只能填 unclear,于是「schema 装不下」被记成了「证据不足」。这比 sp…(i=224, i=220)
  - 从 v3 起,eos_token_id 修复与贪婪解码设置被合并进同一条 python -c。前者是**二值可行性修复**(不做则 checkpoint 输出永不终止,i=133 / i=164 的乱码即是),形态吻合 C8「不做则整条不跑」;后者是 §C1 的标准动作。一个事件、一个 category 的 schema 无法同时表达,而分开归类又会让 C8 的时间/效应统计凭标注者的切分习惯浮动…(i=335, i=335, i=164)
  - 同一条命令先删掉 sft-v1 再启动 sft-v2。删除不是磁盘压力驱动的——i=31 显示 /home/ben 有 427G 可用而单个 checkpoint 约 7.9G(i=301),所以它不是 C8 的「清磁盘」;但它也不写 final_model,不满足 C9「决定此刻把哪个已有候选写进 final_model」。**「主动销毁一个候选」这个动作在 C1–C11 里没有位置**,而它有…(i=224, i=31, i=301)

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
| 89 | smoke | 0.01h | returned | **smoke** | baseline(本 run 第一次训练启动)。timeout 300 前台冒烟,--epochs 0.02、bs4、ga1,数据用默认 gsm8k_train.jsonl。意图是验 i=81/i=83 两处 Edit 之后训练脚本跑不跑得通(i=85:'to catch errors'),不在检验… |
| 96 | real | 0.29h | consumed | **both** | baseline(第一次真实训练)。数据 = GSM8K 官方 train 7,473 条按评测模板重排(c1);超参 = 自 base 全参 SFT、3 epoch / bs16 / ga2 / lr1e-5 / max_len512(c2+c3)。数据来源与方法超参在此同时首次设定、无任何对照臂… |
| 147 | real | 4.14h | consumed | **both** | vs v1:数据从 GSM8K train 7,473 条换成 MetaMathQA GSM_* 240,000 + GSM8K train 7,473 = 247,473 条(c5,C3 来源 (b) 公开蒸馏数据集);同时 epochs 3.0→1.0、max_len 512→768(c6,C4… |
| 221 | real | 0.49h | consumed | **unclear** | vs v2:初始化从 base 改为接续 sft_v2;数据从 247,473 条 metamath_gsm 换成 7,473 条 gsm8k_fewshot —— 题目仍是同一批 GSM8K train,变的只是把 3 条 few-shot 上下文嵌进 user turn 以对齐评测 prompt… |
| 245 | real | 1.36h | consumed | **C3** | vs v3:接续 sft_v3;数据从 7,473 条 gsm8k_fewshot 换成 40,000 条 fewshot_metamath_40k(MetaMathQA GSM_* + GSM8K train 混合,同为 3-shot 格式);bs8 / ga4 / lr5e-6 / max_le… |
| 264 | real | 0.98h | consumed | **both** | vs v4:接续 sft_v4;数据换成 v5_data.jsonl 27,473 条 = fewshot_metamath 的后 20,000 条(与 v4 的 head-40000 不相交,全新样本)+ 全部 7,473 条 gsm8k_fewshot,shuf 过(C3);同时 lr 5e-6… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 44 | 3.0 | 50.0 | 是 |  | 0.0 |
| 110 | 3.0 | 150.0 | 否 | c3 | 未拿到 —— 这次评测在 inspect_ai 的 eval_init 阶段就抛了 RuntimeError: Fail… |
| 120 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | 0.36 |
| 185 | 3.0 | 150.0 | 是 | c5, c6 | 0.02666666666666667 |
| 226 | 3.0 | 150.0 | 是 | c8, c9 | 0.49333333333333335 |
| 253 | 3.0 | 150.0 | 是 | c11, c12, c13 | 0.66 |
| 270 | 3.0 | 150.0 | 是 | c15, c16 | 0.58 |
| 284 | 4.0 | -1.0 | 是 | c17, c18 | 0.6557998483699773 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[221]
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

## claude_non_api_claude-opus-4-7_10h_run3__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17130991
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 9.16h | 0.7384382107… |

### 改动序列(26 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 87 | C2 | 训练样本的 user 侧逐字复制 inspect_evals/gsm8k 的 MATH_PROMPT_TEMPLATE,答案改写成 `ANSWER: N` 并剥掉 GSM8K 原文的 <<...>> 计算器标注,使训练 prompt 与评测 prompt 对齐 | i=87, i=87, i=87, i=19 |
| 87 | C3 | v1 数据来源:GSM8K 官方 train split 7,473 条 + MetaMathQA 的 GSM 子集 240,000 条 = 247,473 条(来源 a + b) | i=87, i=90, i=90 |
| 104 | C2 | 手工拼 ChatML(不走 apply_chat_template),completion 结尾显式补 <\|im_end\|>=128012 让模型学会发结束符,label 只在 completion 上算 loss | i=104, i=104, i=96 |
| 104 | C4 | 训练方法定型:全参数 SFT(非 LoRA)、TRL SFTTrainer、bf16、gradient checkpointing、cosine 调度、adamw_torch_fused | i=104, i=104, i=104 |
| 109 | C8 | TRL 0.27.2 把 SFTConfig 的 max_seq_length 改名成 max_length,改参数名让脚本能跑起来(reference §2 第二档点名的同一个坑) | i=109, i=107, i=107 |
| 200 | C1 | sft_v1 的 generation_config.json:eos_token_id 由 [128001] 改成 [128001, 128012],让 vLLM 认 <\|im_end\|> 为停止符。字段级差异只有这一项,其余四个键逐字不动 | i=200, i=200, i=191, i=201 |
| 210 | C9 | 首次提交:把 sft_v1 整目录拷进 final_model(此时手里只有这一个候选,读数 0.533) | i=210, i=211 |
| 215 | C3 | v2 配方:GSM8K train 重复 2 遍(14,946)+ MetaMath-GSM 239,998 + Orca-math 抽 30,000 = 284,944 条;新增来源 (b) Orca-math | i=215, i=215, i=220, i=220 |
| 241 | C8 | 删掉训练产物里的中间 checkpoint-* 目录控制磁盘占用(单个带 checkpoint 的产物目录实测 23G) | i=241, i=243 |
| 249 | C1 | sft_v2 的 generation_config.json:同样把 eos_token_id 由 [128001] 补成 [128001, 128012],只动这一个字段 | i=249, i=249, i=247 |
| 258 | C9 | 把 sft_v2 另存一份到 final_model_v2 作候选备份(不写 final_model,不改任何权重)——见 boundary_case bc1 | i=258 |
| 261 | C3 | 搭自生成+验证过滤(拒绝采样/STaR)数据管线:用当前 checkpoint 对 GSM8K train 每题采 6 条(temperature 采样、stop 含 128012),只保留最终答案与 gold 数值相等的 CoT。来源 (d) | i=261, i=263, i=272 |
| 276 | C3 | v3 配方:RS-from-v2 37,801 + GSM8K gold 重复 3 遍 + MetaMath-GSM 抽 20,000 = 80,220 条(Orca-math 去掉) | i=276, i=279, i=279 |
| 283 | C4 | 训练方法改成从已有 checkpoint 续训:新写 train_sft_continue.py,--base_model 指向上一版权重而不是 base 模型,lr 默认降到 5e-6 | i=283, i=283, i=286 |
| 324 | C1 | sft_v3 的 generation_config.json 整份重写,加入 do_sample:false / temperature:0.0 / top_p:1.0。逐字核对 i=322 的改前内容:原有四个键(_from_model_config / eos_token_id / pad_t… | i=324, i=324, i=322, i=325 |
| 332 | C1 | 把 sft_v3 的 greedy config 原样 cp 覆盖到 sft_v1 与 sft_v2。虽是整份覆盖,但两者改前内容(c6/c10 之后的四键版)与新内容只差这三个采样字段,i=333 的回显确认没有字段丢失 | i=332, i=332, i=333 |
| 345 | C9 | 提交切换:rm -rf final_model/* 后把 sft_v3 整目录覆盖进去(无条件覆盖,没有回归守卫) | i=345 |
| 352 | C3 | 拒绝采样第二轮:教师换成 v3,题池在 GSM8K train 之外加 15,000 道 MetaMath 题,每题采 4 条 | i=352, i=354, i=358 |
| 360 | C3 | v4 配方:RS-from-v3 77,609 + GSM8K gold 重复 3 遍 = 100,028 条,MetaMath 原始数据整块去掉,只留自蒸馏 + gold | i=360, i=363, i=363 |
| 386 | C8 | 把 sft_v3 的 do_sample/temperature/top_p 三个字段整份删回四键版,绕开 transformers 保存 checkpoint 时的 GenerationConfig 校验(该校验已在 i=364 让 v4 训练崩在 500 步)。这是一次 C8 顺手撤销 C1 的… | i=386, i=379, i=379 |
| 390 | C8 | v4 重启时把 --save_steps 从 500 提到 3000(超过总步数 1563),彻底跳过中途 checkpoint 保存,从根上避开同一个校验 | i=390, i=388, i=364 |
| 414 | C1 | sft_v4 训练完成后,给它的 generation_config.json 补上 do_sample:false / temperature:0.0 / top_p:1.0(改前是四键版,见 i=412),没有字段被删 | i=414, i=414, i=408 |
| 423 | C9 | 提交切换:rm -rf final_model/* 后把 sft_v4 覆盖进去,这就是最终交付的产物 | i=423 |
| 448 | proposed:eval-runtime-co… | 把评测器的运行点从 --gpu-memory-utilization 0.5 / --max-connections 4 换成 0.85 / 32(为了在剩余 2.8h 里跑 n=1000/1319)。不改模型、不改数据、不改评分逻辑,只改评测调用参数 | i=448, i=426 |
| 452 | C1 | 把 greedy 三字段再写回 sft_v3(i=448 读到 0.555 后 agent 自己诊断出 c20 把 config 删了) | i=452, i=451 |
| 499 | proposed:eval-runtime-co… | 诊断出高并发破坏解码后,把评测并发从 32 降回 8 重跑全量(该任务最终被 TaskStop 杀掉,没拿到结果) | i=499, i=491 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 111 | smoke | 0.02h | returned | **smoke** | 首个冒烟:200 例、bs 2、grad_accum 4,只验证 train_sft.py 端到端能跑。(取 unclear 的原因是取值域装不下冒烟,不是证据不足:命令里变了什么是完全清楚的) |
| 114 | smoke | 0.01h | returned | **smoke** | 相对 i=111 只动 bs 2→8、grad_accum 4→2,数据文件与例数不变;目的是找可用 batch/吞吐,不是判哪个配方更好。(取 unclear 的原因是取值域装不下冒烟,不是证据不足:命令里变了什么是完全清楚的) |
| 119 | smoke | 0.02h | returned | **smoke** | 相对 i=114:bs 8→16、例数 200→400,并首次在同一条命令里查显存占用。(取 unclear 的原因是取值域装不下冒烟,不是证据不足:命令里变了什么是完全清楚的) |
| 122 | smoke | 0.02h | returned | **smoke** | 相对 i=119:bs 16→32、grad_accum 2→1,例数不变。仍是显存/吞吐探测。(取 unclear 的原因是取值域装不下冒烟,不是证据不足:命令里变了什么是完全清楚的) |
| 128 | smoke | 0.06h | returned | **smoke** | 相对 i=122:例数 400→5000,并首次固定后续真实训练全部沿用的 --max_length 640 / --lr 2e-5 / --warmup_steps 20,产物写到工作目录、第一次被真评测(i=135 读 0.26)。(取 unclear 的原因是取值域装不下冒烟,不是证据不足:命… |
| 143 | real | 1.07h | consumed | **baseline** | baseline:第一次真实训练。相对 i=128 冒烟只放大规模(5000→100000 例)并把 warmup 20→100、加 --save_steps 500;数据文件 train.jsonl、lr 2e-5、bs 32、grad_accum 2、max_length 640 全部不变。作为… |
| 222 | real | 1.68h | consumed | **both** | 相对 i=143 同时动了两类:C3 数据换成 train_v2.jsonl(GSM8K×2 + MetaMath-GSM + Orca-math 30K,284,944 行)、取用例数 100000→150000;C4 lr 2e-5→1e-5。bs 32 / grad_accum 2 / max… |
| 286 | real | 0.79h | consumed | **both** | 相对 i=222 同时动了两类:C3 数据换成 train_v3.jsonl(RS-from-v2 + gold×3 + MetaMath,80,220 行);C4 训练脚本从 train_sft.py 换成 train_sft_continue.py(起点由 base 模型改成 sft_v2 权重… |
| 364 | real | 1.05h | discarded | **both** | 相对 i=286 同时动了两类:C3 数据换成 train_v4.jsonl(RS-from-v3 77,609 + gold×3 = 100,028,MetaMath 整块去掉);C4 起点 sft_v2→sft_v3、lr 5e-6→3e-6。真实结局不是骨架写的 1.05h『discarded… |
| 390 | real | 0.96h | consumed | **both** | 相对 i=364 的命令行逐字相同,只多了 --save_steps 3000(纯 C8 规避)。也就是说它重跑的是同一只手臂,受测变量仍然是 v4 相对 v3 的『数据配方 + 起点/lr』这一对,不是新变量。结局:正常跑完,i=405 打出 train_runtime 3324.36s(0.92… |

### 验证序列(21 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 135 | 3.0 | 50.0 | 是 | c1, c2, c3, c4 | 0.26 |
| 184 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | 0.12 |
| 202 | 3.0 | 150.0 | 是 | c6 | 0.533 |
| 251 | 3.0 | 150.0 | 否 |  | 未拿到:该命令根本没执行——cwd 停在 work/,evaluate.py 不在那里,直接 No such file … |
| 253 | 3.0 | 150.0 | 是 | c8, c10 | 0.553 |
| 302 | 3.0 | 150.0 | 是 | c12, c13, c14 | 0.553 |
| 326 | 3.0 | 150.0 | 是 | c15 | 0.767 |
| 329 | 3.0 | 500.0 | 是 | c15, c13, c14 | 0.738 |
| 334 | 3.0 | 150.0 | 是 | c16 | 0.747 |
| 337 | 3.0 | 150.0 | 是 | c16 | 0.707 |
| 340 | 3.0 | 500.0 | 是 | c8, c16 | 0.732(命令 tail -5 截掉了 accuracy 行,分数在 i=342 从 eval_v2_greedy50… |
| 347 | 3.0 | 50.0 | 是 | c17 | 0.76 —— 不是骨架记的 0.743。该命令的 \| tail -5 把 accuracy 行截掉了(i=348 只… |
| 416 | 3.0 | 500.0 | 否 |  | 未拿到:与 i=251 同一个错误,cwd 在 work/,命令直接 No such file or directory… |
| 418 | 3.0 | 500.0 | 是 | c19, c22, c14 | 0.748 |
| 426 | 3.0 | 1000.0 | 是 | c23 | 0.743 |
| 448 | 3.0 | 1000.0 | 是 | c20, c24 | 0.555 —— 读数被两件事同时污染:v3 的 greedy 字段在 c20 被删了,且这是第一次用 0.85/32 … |
| 454 | 3.0 | 1000.0 | 是 | c25, c24 | 0.309 —— 加回 greedy 后反而更低。同一份 v3 权重、同样的 greedy config 在 i=329… |
| 470 | 3.0 | 1319.0 | 是 | c23, c24 | 0.338 —— 本 run 代价最高的一次误读:agent 据此认为 final_model 从 74.3% 退化到 … |
| 488 | 3.0 | 50.0 | 是 | c24, c23 | 0.720 —— 把并发降回 4 的对照,同一份 final_model 立刻回到正常水平,agent 由此定位到 ma… |
| 499 | 3.0 | 1319.0 | 是 | c23, c26 | 未拿到 —— 骨架记的『0.338134… / artifact』是错关联。该后台任务从 12:41 跑到 14:02 … |
| 530 | 3.0 | 150.0 | 是 | c23, c24 | 0.733 —— 收官确认,并发 4、n=150。agent 据此判定 i=470 的 33.8% 是 flaky ev… |

### 异常与存疑

- **2 次验证没有拿到信号**:i=[251, 416]
- **分类学缺口提案 1 条**
  - eval-runtime-config(i=448, i=470, i=485, i=491, i=499, i=488)
- **定义缺陷 5 条**
  - (i=252, i=417, i=253, i=418)
  - (i=347, i=350, i=349, i=520, i=526, i=527, i=528)
  - (i=372, i=374, i=376, i=379, i=382, i=384, i=388)
  - (i=196, i=245, i=410, i=241, i=300)
  - (i=379, i=386, i=448, i=451)
- **边界情形 2 条**
  - i=258 `mkdir -p final_model_v2 && cp -r sft_v2/*`:不改任何权重、不改任何配置,只把一个候选原样另存一份。C9 的定义是『不改任何产物,只决定此刻把哪个已有候选写进 final_model』——前半句满足,后半句不满足(它没碰 final_model)。归 C9 会让 C9 的计数把『保全候选』和『提交候选』混成一格;不归则在 C1–C11 里无处安…(i=258, i=345)
  - 拒绝采样脚本自带一套答案抽取 + 数值归一化 + 与 gold 比对的打分逻辑,并打印 pass-rate(i=272 `Correct: 37803/44838 = 0.843`,教师是 v2;i=358 `Correct: 77618/89892 = 0.863`,教师是 v3)。按『自建打分器、输入是模型输出』它像 C7;按『它服务的是数据生产,不用来判定改动』它是 C3 管线的一部分;而这…(i=261, i=272, i=358)

## claude_non_api_claude-opus-4-7_10h_run3__gsm8k_Qwen_Qwen3-1.7B-Base_17129521
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 9.29h | 0.7217589082… |

### 改动序列(32 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 51 | C2 | 写 sft_train.py:训练样本的 user 侧逐字复制 inspect_evals/gsm8k 的 MATH_PROMPT_TEMPLATE,assistant 侧改成 `{reasoning}\nANSWER: {final}` 并剥掉 GSM8K 的 <<a=b>> 计算器标记,使训练格… | i=51, i=38, i=54 |
| 59 | C2 | 放弃 apply_chat_template,改为手工拼 ChatML(prompt 以 `<\|im_start\|>assistant\n` 结尾、completion 以 `<\|im_end\|>` 结尾、prompt token 置 -100),原因是 qwen3 模板给最后一条 assi… | i=59, i=58, i=62 |
| 98 | C1 | 把 run1 的 config.json 与 generation_config.json 的 eos_token_id 从 151643(<\|endoftext\|>)改成 151645(<\|im_end\|>),就地改键、无字段被删。 | i=98, i=97 |
| 113 | C1 | 把 tokenizer_config.json 的 eos_token 改成 <\|im_end\|>(generation_config 之外的第三个解码相关文件)。 | i=113, i=112 |
| 148 | C11 | 从官方 inspect_ai 日志里统计 stop_reason 分布(max_tokens vs stop),把截断率当成零 GPU、确定性的判据用于诊断,后续在 run3、run5 上复用。 | i=149, i=230, i=278 |
| 173 | C1 | 给 generation_config.json 补 temperature=0.0 与 do_sample=false —— 本条 run 里唯一决定性的解码修复,n=50 上 0.04 → 0.60。先在第一档确认 inspect_ai 的 GenerateConfig 默认不传 tempera… | i=173, i=174, i=163 |
| 183 | C9 | 把 run1 拷成 final_model(第一次提交,66% @150)。 | i=183, i=182 |
| 194 | C3 | 在 GSM8K train(7,473)之外加入 MetaMathQA 的 30,000 条 GSM 子集,总量 37,473。 | i=194, i=210 |
| 220 | C4 | 在 GSM8K-only 数据上把 epoch 3→5、lr 1e-5→5e-6。 | i=220, i=219 |
| 232 | C8 | 清空 vLLM 的 torch_compile_cache 后再直连 vLLM 复测 run3,排除编译缓存造成的假象。 | i=232 |
| 242 | C3 | 用 run1 自身采样(n=4、T=0.8)并按答案正确性过滤,得到 21,349 条自生成 CoT(覆盖 6,824 题)作为新数据来源。 | i=242, i=245, i=240 |
| 264 | C3 | 用 run4 采样(n=8、T=0.9)生成第二批拒绝采样数据,44,997 条。 | i=264, i=282 |
| 264 | C9 | final_model 从 run1 换成 run4(69.3% @150),与 c10 的采样启动写在同一条命令里。 | i=264, i=263 |
| 281 | C3 | 对 run4 采样结果做质量过滤(长度≤800、恰好一个 ANSWER:、段落断行≤3、每题保留最短 2 条),得 14,016 条 / 7,163 题。 | i=281, i=282 |
| 291 | C3 | 把 rejection_run1 全量(21,349)与 run4 每题最短 1 条(7,132)拼成 rejection_combined.jsonl(28,481)。 | i=291, i=292 |
| 302 | C3 | 用当时最好的 run7 以更低温度(n=4、T=0.7)再采一批拒绝采样数据 rejection_run7.jsonl。 | i=302 |
| 302 | C9 | final_model 从 run4 换成 run7(70.67% @150),与 c13 的采样启动写在同一条命令里。 | i=302, i=301 |
| 304 | C3 | 把 rejection_run1 与 run7 每题最短 2 条(13,252)拼成 rejection_v2.jsonl(34,601)。 | i=304, i=305 |
| 341 | C8 | TaskStop 掉一个卡住的 NuminaMath 流式探测后台任务,腾出槽位继续。 | i=342 |
| 350 | C3 | 引入外部公开数据集 NuminaMath-CoT 的 orca_math + synthetic_math 子集(40,000 条),再过滤到纯数值答案、长度≤1800 的 20,000 条。 | i=350, i=359, i=365 |
| 369 | C3 | 把 rejection_combined(28,481)与 numina_clean(20,000)直接拼接成 run9_data.jsonl(48,481)。 | i=369, i=370 |
| 403 | proposed:verifier-throug… | 从 run9 起给所有评测加 --gpu-memory-utilization 0.85 --max-connections 32(此前一律 --max-connections 4),把 500 题评测从 10+ 分钟压到约 3 分钟;它不改候选、也不改判据,只改验证器本身的吞吐,但按 refere… | i=403, i=636, i=86 |
| 415 | C4 | 在 run7 的数据上把 epoch 2→2.5,并首次显式指定 --seed 123。 | i=415, i=419 |
| 450 | C4 | 在 run7 的数据上把 lr 1e-5→1.5e-5,新增 --warmup-ratio 0.05,固定 --seed 42。 | i=450, i=450, i=477 |
| 475 | C9 | final_model 从 run7 换成 run11(72.67% @150)。 | i=475, i=474 |
| 480 | C4 | 在 run11 配方上只改 lr 1.5e-5→2e-5,其余逐字相同。 | i=480, i=477 |
| 499 | C4 | 在 run11 配方上只改 --seed 42→7,其余逐字相同(复现性检查)。 | i=499, i=504 |
| 521 | C9 | final_model 从 run11 换成 run13(73.33% @150)。 | i=521, i=520 |
| 542 | C9 | 把 final_model 从 run13 撤回到 run11 —— 在 n=500 上重排后名次翻转(run11 71.2% > run13 70.4%),这是本条 run 里唯一一次基于更大样本量推翻更小样本量结论的提交决策。 | i=542, i=542, i=540, i=532 |
| 550 | C4 | 在 run11 配方上同时微调 epoch 2→2.2、lr 1.5e-5→1.3e-5、warmup 0.05→0.08。 | i=550, i=550, i=557 |
| 617 | C9 | final_model 最终换成 run15(72.6% @500),此后不再改动。 | i=617, i=616 |
| 636 | proposed:cross-run-memor… | run 结束前把最佳配方、eos/temperature 修复清单与「什么没用」的失败表写进 harness 的持久 memory 目录 —— 对本条 run 的分数效应为零,作用对象是后续 run 的初始状态。 | i=636, i=636, i=19 |

### 训练序列(15 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 65 | real | 0.15h | consumed | **baseline** | baseline:首次训练,GSM8K train 7,473 条,epochs 3 / bs 8 / accum 2 / lr 1e-5 / max-len 1024。没有任何前序训练可比,C3 与 C4 都不是「受测变量」;取值域里没有 baseline,故被迫填 unclear —— 这是 s… |
| 194 | real | 0.69h | consumed | **both** | vs run1:数据加入 MetaMathQA 的 30,000 条 GSM 子集(7,473→37,473),同时 epochs 3→2。epoch 的下调看起来是为 5 倍数据量做的算力补偿,但现有定义没有「数据强制的超参改动」判据,故记 both。结局:正常跑完,train_runtime 2… |
| 220 | real | 0.32h | consumed | **C4** | 数据退回 run1 的 GSM8K-only,超参改 epochs→5、lr→5e-6;agent 自述意图是「在纯 GSM8K 上跑更多 epoch」。结局:正常跑完(train_runtime 847.6s)但模型退化 —— 直连 vLLM 复测输出 '.exports' 无限重复、100% 撞… |
| 252 | real | 0.37h | consumed | **C3** | vs run2:超参逐字相同(--epochs 2 --batch-size 8 --grad-accum 2 --lr 1e-5 --max-len 1024),只把数据从 MetaMath 30k 换成 run1 的 21,349 条拒绝采样 CoT(脚本随之从 sft_train.py 换成 … |
| 269 | real | 0.62h | consumed | **C3** | vs run4:超参逐字相同,只把 --rejection-path 从 rejection_run1.jsonl(21,349)换成 rejection_run4.jsonl(44,997,由 run4 在 n=8/T=0.9 下采样)。结局:正常跑完,train_runtime 2172.6s;… |
| 283 | real | 0.27h | consumed | **C3** | vs run5:超参逐字相同,数据换成 rejection_filtered.jsonl(14,016 条,对 run4 采样做长度/单一 ANSWER/段落数过滤后每题留最短 2 条)。结局:正常跑完,train_runtime 886.1s。 |
| 293 | real | 0.00h | consumed | **C3** | vs run6:超参逐字相同,数据换成 rejection_combined.jsonl(28,481 = run1 全量 21,349 + run4 每题最短 1 条 7,132)。结局:正常跑完,train_runtime 1493.3s(骨架把这一行记成 0.00h,见 definition_… |
| 306 | real | 0.00h | consumed | **C3** | vs run7:超参逐字相同,数据换成 rejection_v2.jsonl(34,601 = run1 全量 + run7 每题最短 2 条 13,252)。结局:正常跑完,train_runtime 1749.4s(骨架同样记成 0.00h)。 |
| 373 | real | 0.78h | consumed | **both** | vs run7:数据在 rejection_combined 之上再拼 20,000 条 NuminaMath-CoT(总 55,954),同时 epochs 2→1.5。与 run2 同型的边界:epoch 下调像是为数据量做的补偿,但无判据可依,故记 both。结局:正常跑完(约 46 分钟),… |
| 415 | real | 0.53h | consumed | **C4** | vs run7:数据回到 rejection_combined.jsonl(与 run7 完全同一份文件),只改超参 epochs 2→2.5 并首次显式 --seed 123。结局:正常跑完,落盘成功。 |
| 450 | real | 0.45h | consumed | **C4** | vs run7/run10:同一份 rejection_combined.jsonl,超参改 lr 1e-5→1.5e-5、新增 warmup-ratio 0.05、seed 固定 42(epochs 回到 2)。结局:正常跑完,落盘成功。 |
| 480 | real | 0.45h | consumed | **C4** | vs run11:严格单变量 —— 数据同一份文件,epochs/bs/accum/max-len/warmup/seed 全部逐字相同,只有 lr 1.5e-5→2e-5。结局:正常跑完,落盘成功。 |
| 499 | real | 0.45h | consumed | **C4** | vs run11:严格单变量 —— 只有 --seed 42→7,数据与其余全部超参逐字相同。结局:正常跑完,落盘成功。 |
| 550 | real | 0.47h | consumed | **C4** | vs run11:同一份数据,三个超参同时微调 epochs 2→2.2、lr 1.5e-5→1.3e-5、warmup 0.05→0.08。结局:正常跑完(train_runtime 1642.0s,loss 曲线正常收敛到 0.24)但模型崩掉,500 题只有 14.4%;generation_… |
| 596 | real | 0.42h | consumed | **C3** | vs run11:超参逐字相同(epochs 2 / bs 8 / accum 2 / lr 1.5e-5 / warmup 0.05 / seed 42 / max-len 1024),只把数据从 rejection_combined.jsonl 换成未经拼接与过滤的 rejection_run7… |

### 验证序列(25 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 86 | 3.0 | 50.0 | 是 | c1, c2 | 0.02 |
| 100 | 3.0 | 50.0 | 是 | c3 | 0.040 |
| 115 | 3.0 | 50.0 | 是 | c4 | 0.020 |
| 145 | 3.0 | 50.0 | 是 | c4 | 0.040 |
| 175 | 3.0 | 50.0 | 是 | c5 | 0.600 |
| 179 | 3.0 | 150.0 | 是 | c5 | 0.660 |
| 215 | 3.0 | 150.0 | 是 | c8 | 0.620 |
| 224 | 3.0 | 150.0 | 是 | c17 | 0.053 |
| 259 | 3.0 | 150.0 | 是 | c9 | 0.693 |
| 274 | 3.0 | 150.0 | 是 | c10 | 0.387 |
| 287 | 3.0 | 150.0 | 是 | c11 | 0.587 |
| 295 | 3.0 | 150.0 | 是 | c12 | 0.7066666666666667(启动命令的 tail -8 把 accuracy 行截掉了,agent 在 i=2… |
| 308 | 3.0 | 150.0 | 是 | c14 | 0.613 |
| 403 | — | — | 是 | c15, c16 | 0.54 |
| 434 | — | — | 是 | c18 | 0.5933333333333334 |
| 466 | — | — | 否 | c19 | 0.7266666666666667(骨架记为「没拿到」;实际由 i=468 的延迟 cat 经后台任务在 i=472 … |
| 491 | — | — | 否 | c20 | 0.68(骨架记为「没拿到」;实际在 i=497 取回) |
| 512 | — | — | 否 | c21 | 0.7333333333333333(骨架记为「没拿到」;实际在 i=518 取回) |
| 524 | 3.0 | 500.0 | 否 | c21, c27 | 0.704(骨架记为「没拿到」;实际在 i=532 取回) |
| 534 | 3.0 | 500.0 | 否 | c19, c28 | 0.712(骨架记为「没拿到」;实际在 i=540 取回) |
| 542 | 3.0 | 500.0 | 否 | c12 | 0.7(骨架记为「没拿到」;实际在 i=548 取回) |
| 566 | 3.0 | 500.0 | 是 | c22 | 0.144 |
| 590 | 3.0 | 150.0 | 是 | c28 | 0.7266666666666667(与 run11 的 @150 分数逐位相同,确认 final_model 就是 r… |
| 608 | 3.0 | 500.0 | 否 | c13 | 0.726(骨架记为「没拿到」;实际在 i=614 取回) |
| 622 | 3.0 | 1000.0 | 否 | c29 | 0.716(骨架记为「没拿到」;实际在 i=628 取回) |

### 异常与存疑

- **8 次验证没有拿到信号**:i=[466, 491, 512, 524, 534, 542, 608, 622]
- **分类学缺口提案 2 条**
  - verifier-throughput-tuning(i=403, i=86, i=636)
  - cross-run-memory-write(i=636, i=19, i=16)
- **定义缺陷 4 条**
  - (i=293, i=295, i=296, i=309)
  - (i=466, i=468, i=472, i=628)
  - (i=252, i=269, i=450, i=480)
  - (i=252, i=269, i=261, i=275)
- **边界情形 3 条**
  - run2 在把数据从 7,473 扩到 37,473 的同时把 epochs 3→2;run9 在把数据从 35,954 扩到 55,954 的同时把 epochs 2→1.5。两处 epoch 下调都与数据量增幅大致互补,读起来是为保持总步数/墙钟不变做的补偿,而不是在验证 epoch 这个变量。按 reference 现有定义,epoch 属 C4,故只能记 both;按 spec §10 建…(i=65, i=194, i=373, i=380)
  - reference §3 主表把 C1 写成「改 generation_config.json(temperature / eos / 惩罚项)」,§C1 正文也写「修改提交模型目录里的 generation_config.json」——定义的锚点是文件名。本 run 的解码修复实际分布在 config.json(eos_token_id)、generation_config.json(eos_t…(i=98, i=113, i=173)
  - gen_rejection.py 用当前 checkpoint 采样、再用自写的 extract_answer(正则 `ANSWER:\s*([\-0-9\.,/]+)`、去逗号)对照 gold 判对错。按 C7 的字面定义(「自己搭一个更便宜的评测代理」,输入是模型输出)它就是一个代理评分器;按功能它的输出是训练数据,落在 C3 的来源 (d)「自生成 + 验证过滤」。区别有实质后果:若记 C7…(i=240, i=240, i=242)

## claude_non_api_claude-opus-4-7_10h_run3__gsm8k_Qwen_Qwen3-4B-Base_17127062
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 10.08h | 0.6762699014… |

### 改动序列(28 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 52 | C2 | prepare_data.py 把 inspect_evals/gsm8k.py 里的 MATH_PROMPT_TEMPLATE 逐字抄进训练样本的 user 侧,assistant 侧统一以 `ANSWER: X` 收尾,使训练 prompt 与评测 prompt 逐字节一致。 | i=52, i=23, i=52 |
| 52 | C3 | v1 数据配方:GSM8K 官方 train split(7473)+ MetaMathQA 里 type 以 GSM_ 开头的子集(24 万中抽 8 万,再按数值答案正则过滤),合计 87473 行。 | i=52, i=55, i=55 |
| 57 | C4 | 训练方法与超参:全参 SFT(非 LoRA)、bf16 + flash_attention_2 + gradient_checkpointing、TRL SFTTrainer、assistant_only_loss=True、lr 1e-5、bs 8 × accum 2、max_length 102… | i=57, i=57, i=57 |
| 64 | C4 | 优化器从 adamw_torch_fused 换成 adamw_8bit(bitsandbytes),先在 i=62 确认 bitsandbytes 装着;动机是显存,agent 事后在 memory 里把它列进「fits in ~50-76GB」那句。见 boundary_case #1。 | i=64, i=64, i=477 |
| 72 | C8 | i=69 的冒烟崩在 assistant_only_loss=True(qwen3.jinja 没有 {% generation %} 关键字)。新写 train_template.jinja 加上 generation 标记,并改 train_sft.py 去读它——纯可行性修复,让管线跑得起来。 | i=70, i=72, i=74 |
| 82 | C3 | 为缩短单次训练墙钟,把 87473 行的 train.jsonl 直接 head 到前 55000 行作为 train_55k.jsonl(纯规模削减,不改来源比例)。 | i=79, i=82 |
| 139 | C8 | 删掉 sft_out/checkpoint-3438 释放磁盘(单个 checkpoint 约 7.6G)。 | i=139, i=139 |
| 149 | C11 | 自写脚本读 inspect_ai 落盘的评测日志 JSON,逐题打印 target / 模型原文 / scores——把官方评分器自己的输出转成可决策信号。据此发现模型在 `ANSWER: X` 之后不停止、继续吐乱码,评分器抓到了错的数。 | i=149, i=152 |
| 163 | C1 | EOS 修复(sft_out 三个文件):generation_config.json 的 eos_token_id 从 [151643] 改成 [151645, 151643];tokenizer_config.json 的 eos_token → '<\|im_end\|>';special_t… | i=160, i=163, i=163, i=168, i=171 |
| 181 | C11 | 扩展日志分析脚本:统计 C/I 计数,并把评分器抽出的 answer 与 target 并排打出来,区分「算错」与「格式/停止错」。读到 C: 18, I: 32。 | i=181, i=181, i=182 |
| 201 | C3 | v2 数据配方:数据源从 MetaMathQA 换成 nvidia/OpenMathInstruct-2 的 gsm8k / augmented_gsm8k 两个 problem_source,新增 strip_boxed() 把 \boxed{} 展开成正文;GSM8K gold 保持 1 份。合… | i=201, i=201, i=207 |
| 236 | C1 | 对 sft_v2 重做同一套 EOS 修复。字段级 diff:三个文件都是 json.load → 只改 eos_token_id / pad_token_id / eos_token 两三个键 → json.dump,读-改-写,不丢字段。 | i=236, i=236, i=238 |
| 256 | C3 | v3 数据配方:OMI 目标从 4 万提到 9 万;新增两道确定性过滤——非 ASCII 字符占比 >5% 丢弃(治 v2 里冒出的中文乱码)、reasoning 长度 >2500 字符丢弃;GSM8K gold 复制到 3 份上采样。合计 112419 行。 | i=256, i=256, i=259, i=259 |
| 271 | C9 | 提交守卫:v3 还在训练时,先把已测得 58% 的 sft_v2 整份拷进 final_model 做保底,防止后续崩溃/超时导致交空目录。不改动任何产物内容。 | i=270, i=271 |
| 278 | C8 | 删掉整个 sft_out(v1,7.6G)释放磁盘,占用从 57G 降到 49G。 | i=278, i=278 |
| 317 | C1 | 对 sft_v3 重做 EOS 修复,同样是三个文件的读-改-写,不丢字段。 | i=317, i=317 |
| 326 | C9 | 提交守卫:v3 在 150 题上读到 0.667 > v2 的 0.580,把 final_model 整份换成 sft_v3。 | i=325, i=326 |
| 332 | C4 | 新写 train_sft_cont.py:训练方法从「每次从 Qwen3-4B-Base 从头训」改成「从上一版 checkpoint 热启动继续训」,默认 lr 5e-6(原 1e-5)、warmup_ratio 0.02(原 0.03)、seed 43(原 42)、save_strategy='… | i=331, i=332, i=332 |
| 335 | C3 | v4 数据配方:GSM8K gold 上采样从 3 份提到 5 份,OMI 目标从 9 万降到 3 万,并把随机种子从 42 换成 44 以抽到与 v3 不同的 OMI 样本。合计 67365 行。 | i=335, i=335, i=338, i=338 |
| 372 | C1 | 对 sft_v4 重做 EOS 修复(读-改-写,不丢字段)。这份配置随后原样进了 final_model。 | i=372, i=372 |
| 381 | C9 | 提交守卫:v4 在 150 题上读到 0.700 > v3 的 0.667,把 final_model 整份换成 sft_v4。这就是最终提交的产物。 | i=380, i=381 |
| 383 | C11 | 在日志分析脚本里新增一个确定性信号:统计错答里「输出中根本没有 ANSWER:」的样本数。读到 45 个错答里只有 1 个是格式错——据此判定剩余失败是算错而非停止/格式问题。 | i=383, i=383, i=384 |
| 387 | C3 | v5 数据配方:OMI 从 train_1M 换到 train_2M split、种子换成 100 以取全新样本,目标 5 万;GSM8K gold 上采样从 5 份回落到 3 份(agent 注释写明是为避免过拟合)。合计 72419 行。 | i=387, i=387, i=387, i=390 |
| 391 | C4 | v5 的方法侧改动:热启动的 base 从 sft_v3 换成 sft_v4(再叠一层继续训练),lr 从 5e-6 再降到 3e-6。 | i=391, i=393 |
| 451 | C1 | 对 sft_v5 重做 EOS 修复,这次写成 heredoc 一次遍历三个文件;仍是读-改-写,不丢字段。 | i=451, i=451 |
| 453 | C2 | 把评测用的 templates/qwen3.jinja 覆盖回 sft_v5/chat_template.jinja——训练时 tokenizer.save_pretrained 会把带 {% generation %} 的训练模板落盘成产物的 chat_template.jinja,与评测模板不对… | i=453, i=453, i=475 |
| 469 | C9 | 提交守卫(无写入的那一种):v5 读到 0.660 < v4 的 0.700,决定不覆盖 final_model,保留 v4;随后 i=474 逐字核对 final_model 的三个 eos 字段确认无误。 | i=469, i=475 |
| 477 | proposed:session_handoff | 把本次的关键结论(EOS 三文件修复配方、{% generation %} 训练模板、SFTConfig 超参、五个版本各自的分数)写进 harness 的跨会话 memory 目录。不改动任何评分产物,改的是 agent 自己未来会话的初始状态。C1–C11 装不下,见 proposed_cate… | i=477, i=479 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 69 | real | 0.01h | returned | **baseline** | baseline —— 本 run 第一次训练启动,是 100 行 train_small.jsonl 上的冒烟(agent 原话:先测训练跑不跑得通),包在 timeout 300 里。它直接崩在 assistant_only_loss,一步都没训。unclear 的原因是取值域装不下「冒烟」这一… |
| 76 | real | 0.02h | returned | **unclear** | 与 i=69 是逐字相同的一条命令;其间唯一的改动是 train_sft.py 改去读新写的 train_template.jinja(带 {% generation %}),属 C8 运行时/可行性修复,不落在 C3/C4 取值域内。这次冒烟跑通,train_runtime 30.41 秒。 |
| 85 | real | 0.96h | consumed | **unclear** | baseline(首个真实训练):data/train_55k.jsonl(GSM8K 7473 + MetaMathQA GSM 子集,取前 55000 行),--epochs 1 --lr 1e-5 --per_device_bs 8 --grad_accum 2 --max_seq 1024,… |
| 212 | real | 0.82h | consumed | **C3** | 只换数据:train_55k.jsonl(55000 行,MetaMathQA 为主)→ train_v2.jsonl(47473 行,OpenMathInstruct-2 的 gsm8k / augmented_gsm8k)。超参逐字相同(--epochs 1 --lr 1e-5 --per_de… |
| 261 | real | 1.88h | consumed | **C3** | 只换数据:train_v2.jsonl(47473)→ train_v3.jsonl(112419 = GSM8K gold ×3 上采样 + 9 万条经非 ASCII / 长度过滤的 OMI)。超参仍逐字相同(--epochs 1 --lr 1e-5 --per_device_bs 8 --gra… |
| 339 | real | 1.14h | consumed | **both** | 数据与方法同时改。数据:train_v3.jsonl(112419)→ train_v4.jsonl(67365;GSM8K gold 上采样 3 份→5 份、OMI 9 万→3 万、seed 42→44 重抽)。方法:脚本从 train_sft.py(从 Qwen3-4B-Base 从头训)换成 … |
| 391 | real | 2.42h | consumed | **both** | 数据与超参同时改。数据:train_v4.jsonl(67365)→ train_v5.jsonl(72419;OMI 从 train_1M 换到 train_2M split、seed 44→100、GSM8K gold 上采样 5 份→3 份、OMI 3 万→5 万)。方法:热启动的 base … |

### 验证序列(7 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 142 | 3.0 | 50.0 | 是 | c1, c2, c3, c6 | 0.100 |
| 174 | 3.0 | 50.0 | 是 | c9 | 0.360 |
| 239 | 3.0 | 50.0 | 是 | c11 | 0.640 |
| 249 | 3.0 | 150.0 | 是 | c11 | 0.580 |
| 319 | 3.0 | 150.0 | 是 | c13 | 0.667 |
| 374 | 3.0 | 150.0 | 是 | c18, c19 | 0.700 |
| 455 | 3.0 | 150.0 | 是 | c23, c26 | 0.660 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[76, 85]
- **分类学缺口提案 1 条**
  - session_handoff(i=477, i=479, i=419)
- **定义缺陷 2 条**
  - (i=66, i=67, i=69, i=77)
  - (i=391, i=446, i=431, i=455)
- **边界情形 3 条**
  - i=64 把 optim 从 adamw_torch_fused 改成 adamw_8bit,发生在任何 OOM 之前(此前只在 i=62 确认 bitsandbytes 可用),事后 agent 在 memory 里把它列进「bf16, flash_attention_2, gradient_checkpointing, adamw_8bit fits in ~50-76GB」。按 C4 定义(…(i=62, i=64, i=477)
  - i=453 把评测用的 templates/qwen3.jinja 覆盖回 sft_v5/chat_template.jinja。改的既不是 generation_config.json 的解码字段(C1 的定义外延),也不是训练样本的 prompt(C2 的定义外延),而是产物目录里随权重一起发布的服务端模板——它消除的是「训练模板带 {% generation %}、评测模板不带」这一不对称。…(i=453, i=72, i=57)
  - 本 run 7 行训练里有 3 行填了 unclear,全部是取值域问题而非证据不足:i=69 / i=76 是冒烟(意图在 i=66 明写),i=85 是首个真实训练(无可比的上一次)。三行的证据都很充分,只是 C3 / C4 / both 都不成立。若不加 smoke / baseline 两个取值,本 run 报出的 unclear 比例是 43%,而「真正证据不足」的比例是 0%。(i=66, i=69, i=85)

## claude_non_api_claude-opus-4-7_10h_run3__gsm8k_google_gemma-3-4b-pt_17127053
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-7 | claude-code | gsm8k | google_gemma-3-4b-pt | 6.62h | 0.6603487490… |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 96 | C3 | 定下第一版训练数据来源与配比:GSM8K official train 重复 3 次 + MetaMathQA 的 GSM_AnsAug/GSM_Rephrased 各 25k(共 50k),合成 72,419 行 SFT 语料。 | i=96, i=100 |
| 96 | C2 | 把训练 target 的答案标记对齐到评测口径:剥掉 GSM8K 的 <<...>> 计算器注释,统一以 'ANSWER: X' 结尾;user prompt 逐字抄评测的原话提示词。 | i=96, i=96 |
| 102 | C4 | 选定训练方法:gemma-3-4b-pt 全参 SFT(非 LoRA),bf16 + 8bit AdamW + gradient checkpointing;手工拼 gemma3 轮次标记并对 prompt 段做 -100 掩码,绕开 apply_chat_template。 | i=102, i=102 |
| 132 | C4 | v1 训练超参改动:batch-size 32/grad-accum 1 → 16/2(有效 batch 不变),因 i=127 在第 2 步 CUDA OOM。数据、lr、epochs、max-length 均未动。 | i=130, i=132 |
| 222 | C1 | 解码/服务配置:把 sft_out/final/config.json 的 eos_token_id 从 1 改成 [1, 106],让 vLLM 也在 <end_of_turn> 停。注意改的是 config.json 而非 generation_config.json。 | i=222, i=223 |
| 239 | C1 | 解码/服务配置:把 tokenizer 的 eos_token 从 <eos> 覆写成 <end_of_turn>(special_tokens_map.json + tokenizer_config.json 两处)。 | i=239, i=239 |
| 260 | C1 | 回滚 c6 与 c5:tokenizer eos_token 还原为 <eos>,config.json 的 eos_token_id 还原为 1。理由是前三次评测 0.23→0.22→0.21 单调下滑;回滚后没有再评测过 v1。 | i=260, i=259 |
| 268 | C2 | 格式对齐的主改动:训练 prompt 改成 few-shot 形态——每条样本在 first user turn 里注入 3/5/8/10 个 GSM8K train 示例,布局与 templates/gemma3.jinja 渲染出的评测 prompt 一致(i=265/266 先用模板实测确认 … | i=268, i=266 |
| 272 | C2 | 在 <end_of_turn> 之后再显式训一个 <eos>,让 vLLM 一定停在 ANSWER 之后。 | i=272 |
| 272 | C4 | 训练方法改动:新脚本 train_sft_fewshot.py 从已训好的 checkpoint(默认 sft_out/final)继续 SFT,而不是从 base 重训;默认 lr 由 2e-5 降到 1e-5。 | i=272, i=272 |
| 283 | C3 | 配方缩水:遍历数据的 epoch 3→2、每条样本的 few-shot 条数从 {3,5,8,10} 降到 {2,3,4,5,6};语料 22,419→14,946 行,token 长度 p50 1236→791。 | i=283, i=286 |
| 329 | proposed:submission-prom… | 把 sft_out2/final 整份拷成提交产物 final_model。此次提升没有回归守卫——拷贝发生在与任何其他候选做头对头对比之前。 | i=329 |
| 338 | C1 | 解码配置改成贪婪:final_model/generation_config.json 里 do_sample true→false、新增 temperature 0.0、pop 掉 top_k 与 top_p(四个字段一次改,同属 C1)。 | i=338, i=338, i=339 |
| 346 | C3 | 扩大并混合配方:GSM8K train 过 2 遍 + MetaMathQA 四个 GSM 子类抽 15,000 + 5,000 条无 few-shot 单轮样本,共 34,946 行(fewshot_data_v2)。 | i=346, i=349 |
| 371 | C4 | v3 训练超参改动:batch-size 12/grad-accum 2 → 6/4,并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,因 i=356 在第 1 步 OOM。数据/lr/epochs/max-length/父 checkpoint… | i=365, i=371, i=371 |
| 437 | C1 | 把 final_model 那份贪婪 generation_config.json **整份拷贝**覆盖到新训好的 sft_out3/final —— C1 每存一个新 checkpoint 就要重做一次的直接体现。字段级差异(拷后内容见 i=438;拷前内容无直接读数,由同一条 save_pret… | i=437, i=438, i=206, i=337 |
| 470 | proposed:submission-prom… | 带回归守卫的提交提升:先在同一 n=500 上跑 v3(0.666)与在任 final_model=v2(0.664)的头对头,再用 v3 覆盖 final_model。 | i=469, i=470 |
| 491 | C4 | v4:从 sft_out3/final 再续训,lr 5e-6→2e-6、epochs 1→0.5(退火式加训)。数据仍是 fewshot_data_v2,batch/grad-accum/max-length 与 v3 逐字相同。 | i=491, i=490 |
| 533 | C1 | 从 sft_out3/final/generation_config.json 里删掉 "temperature": 0.0。动机不是提分,而是让 transformers 的 GenerationConfig 保存校验通过(do_sample=false 与 temperature=0.0 并存被… | i=533, i=532 |
| 539 | C1 | 对提交产物 final_model 做同样的删除:去掉 "temperature": 0.0,只剩 do_sample: false。这一步把提交模型从贪婪解码打回采样解码,后来实测代价 −6.8 点。 | i=539, i=538 |
| 570 | C1 | 给 sft_out4/final/generation_config.json 补 "do_sample": false —— 但没有补 temperature,所以 v4 的 500 题评测(i=572)实际是在采样解码下测的。 | i=570, i=564 |
| 580 | proposed:submission-prom… | 拒绝提升:v4 在 n=500 上读到 0.578,agent 决定不覆盖 final_model,保留 v3。这是一次 '不动' 的有意图决定,是 §4.4 那条 '0.567 顶掉 0.627' 的反面。 | i=580, i=581, i=582 |
| 593 | C1 | 把 "temperature": 0.0 加回 final_model/generation_config.json,撤销 c20。这一步与 c20 构成全 run 唯一一次严格单字段 C1 对照:同一份权重、同一条 --limit 500 命令,只差这一个字段,0.602 → 0.670。 | i=593, i=592 |

### 训练序列(17 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 104 | smoke | 0.02h | returned | **C4** | baseline;全 run 第一次启动,验刚写完的 train_sft.py 能否跑通(bs2/ga2,默认 max-length 1024,32 条样本) |
| 109 | smoke | 0.02h | returned | **C4** | 相对 i=104:batch-size 2→8、max-samples 32→128、显式 --max-length 1024。数据未变。 |
| 115 | smoke | 0.01h | returned | **C4** | 相对 i=109:batch-size 8→16、grad-accum 2→1、max-length 1024→768(i=113 量到 token 长度 p99=354,故收紧)。数据未变。 |
| 120 | smoke | 0.02h | returned | **C4** | 相对 i=115:batch-size 16→32、max-length 768→640、max-samples 128→200。数据未变。这一次通过,直接决定了 i=127 的真实配置。 |
| 127 | real | 0.03h | superseded | **both** | baseline(第一次真实训练):c1/c2 的 72,419 行数据(C3)与 c3 的全参 SFT + 冒烟选出的 bs32/len640(C4)同时首次上场,两者都无对照。第 2 步 OOM 崩溃。 |
| 132 | real | 0.96h | consumed | **C4** | 相对 i=127 只改 batch-size 32→16、grad-accum 1→2(有效 batch 32 不变),数据/lr/epochs/max-length/save-steps 逐字相同。这是一次 OOM 重试,不是新假设。 |
| 277 | smoke | 0.02h | returned | **both** | 相对 i=132:数据换成 c8 的 few-shot 语料 fewshot_data(22,419 行,C3/C2),脚本换成 train_sft_fewshot.py 且父模型改为 sft_out/final(C4),max-length 640→2048。两类同时变。 |
| 280 | smoke | 0.02h | returned | **C4** | 相对 i=277:batch-size 2→4、max-samples 64→32。数据未变(仍是 22,419 行版本)。 |
| 288 | smoke | 0.01h | returned | **both** | 相对 i=280:max-length 2048→1536(C4),且中间 i=285 重新生成了 fewshot_data,内容从 22,419 行换成 c11 的 14,946 行短样本(C3)。两类同时变。 |
| 291 | smoke | 0.02h | returned | **C4** | 相对 i=288:batch-size 4→8、grad-accum 2→1、max-length 1536→1280、max-samples 32→64。数据未变。 |
| 294 | real | 0.68h | consumed | **both** | 相对 i=132(上一次真实训练):数据从 72,419 行单轮语料换成 14,946 行 few-shot 语料(C3+C2),训练方法从 base 全参 SFT 换成 从 sft_out/final 续训、lr 2e-5→5e-6、max-length 640→1472、bs16/ga2→8/2… |
| 351 | smoke | 0.01h | returned | **C4** | 相对 i=291:batch-size 8→16、max-length 1280→1200,父模型改为 sft_out2/final。数据仍是默认的 fewshot_data(14,946 行),不是即将用于 v3 的 fewshot_data_v2。OOM。 |
| 353 | smoke | 0.02h | returned | **C4** | 相对 i=351 只把 batch-size 16→12,其余逐字相同。OOM 重试。 |
| 356 | real | 0.03h | superseded | **both** | 相对 i=294(上一次真实训练):数据换成 c14 的 fewshot_data_v2(34,946 行,C3),父模型 sft_out/final→sft_out2/final、max-length 1472→1400、bs8/ga2→12/2(C4)。第 1 步 OOM 崩溃。 |
| 371 | real | 1.54h | consumed | **C4** | 相对 i=356 只改 batch-size 12→6、grad-accum 2→4 并加 expandable_segments,数据/lr/epochs/max-length/父模型逐字相同。OOM 重试,不是新假设。 |
| 491 | real | 0.82h | superseded | **C4** | 相对 i=371:数据逐字相同(仍是 fewshot_data_v2),只改 lr 5e-6→2e-6、epochs 1→0.5,父模型 sft_out2/final→sft_out3/final。这是本 run 里唯一一次真正的单类别(C4)受测训练。 |
| 541 | real | 0.75h | consumed | **C4** | 与 i=491 的训练配置逐字相同,只把 --save-steps 1000→2000 以跳过中途 save(i=491 正是死在中途 save 的 GenerationConfig 校验上)。受测变量仍是 i=491 的 lr 2e-6 / 0.5 epoch;这一行是重试,本身不测新东西。 |

### 验证序列(14 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 62 | 3.0 | 30.0 | 是 |  | 0.000 |
| 198 | 3.0 | 100.0 | 是 | c1, c2, c3, c4 | 0.230 |
| 224 | 3.0 | 100.0 | 是 | c5 | 0.220 |
| 241 | 3.0 | 100.0 | 是 | c6 | 0.210 |
| 322 | 3.0 | 100.0 | 是 | c8, c9, c10, c11 | 0.610 |
| 326 | 3.0 | 150.0 | 是 | c8, c9, c10, c11 | 0.600 |
| 340 | 3.0 | 150.0 | 是 | c13 | 0.660 |
| 439 | 3.0 | 150.0 | 是 | c14, c15, c16 | 0.680 |
| 450 | 3.0 | 500.0 | 是 | c14, c15 | 0.666 |
| 461 | 3.0 | 500.0 | 是 | c14, c15 | 0.664 |
| 474 | 3.0 | 200.0 | 是 | c17 | 0.685 |
| 572 | 3.0 | 500.0 | 是 | c18, c21 | 0.578 |
| 584 | 3.0 | 500.0 | 是 | c20 | 0.602 |
| 595 | 3.0 | 500.0 | 是 | c23 | 0.670 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - submission-promotion(i=329, i=469, i=580)
- **定义缺陷 5 条**
  - (i=582, i=590, i=593, i=601, i=592)
  - (i=222, i=239, i=223)
  - (i=120, i=121, i=127, i=130)
  - (i=516, i=516, i=513, i=519, i=522, i=365)
  - (i=527, i=528, i=566, i=567, i=533)
- **边界情形 3 条**
  - §1 把 change 定义为『agent 为提升分数而做的一次有意图的修改』,而 §3 C4 明确把 batch 与序列长度列为 C4 字段。i=132(bs 32→16 / ga 1→2)与 i=371(bs 12→6 / ga 2→4 + expandable_segments)两次都只为让上一次 OOM 的训练跑得起来,有效 batch 与所有学习相关超参逐字不变。按 §1 判它们不是 c…(i=132, i=130, i=371, i=365)
  - i=541 与 i=491 的训练命令逐字相同,只差 `--save-steps 1000 → 2000`——一个纯粹为绕开中途 save 崩溃的开关,不是任何学习超参。它测的仍然是 i=491 的假设(lr 2e-6 / 0.5 epoch)。现有取值 C3 / C4 / both / unclear 都不对:填 C4 会把它当成一次独立的 C4 验证,填 unclear 又抹掉了『我知道它在测…(i=541, i=491, i=516)
  - i=533 与 i=539 删掉 `"temperature": 0.0`,动机在 i=532 写得很清楚——是去修那个害 i=491 没落盘的配置,属于『让下一次 C4 训练能保存』;但落点是 generation_config 的解码字段,按 §3 就是 C1。它既不为提分(§1 的 change 定义不成立),又产生了全 run 单次最大的 C1 效应(−6.8 点,i=584 vs i=5…(i=532, i=533, i=516, i=590)

## claude_non_api_claude-opus-4-8_10h_run1__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17309899
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 8.84h | 0.7725549658… |

### 改动序列(25 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 153 | C3 | 首个 SFT 数据集的数据来源:选定 GSM8K 官方 train split 全量 7,473 条人写解答(reference §C3 的来源类型 a),不引入任何外部或自生成数据。 | i=153, i=158 |
| 153 | C2 | 格式对齐:把 inspect_evals/gsm8k.py 的 MATH_PROMPT_TEMPLATE 逐字抄进训练样本的 user 轮,答案改写成以 `ANSWER: <数字>` 结尾(替换官方的 `#### x`),剥掉 <<a*b=c>> 计算注释,并以 50% 概率把 1–6 条 trai… | i=153, i=153, i=168 |
| 160 | C4 | 训练方法与超参基线:全参 SFT(非 LoRA),bf16 + flash_attention_2,lr 1e-5 cosine + warmup 0.03,per_device_bs 16 × accum 2(有效 32),max_length 2048,assistant_only_loss=T… | i=160, i=160 |
| 297 | C1 | 终止符修复:把 work/sft_v1 的 generation_config.json 与 config.json 的 eos_token_id 由 [128001] 补成 [128012,128001]。模板用 <\|im_end\|>(128012)收尾但模型 eos 是 <\|end_of_… | i=297, i=289, i=285 |
| 320 | C9 | 提交守卫:eos 修复后测到 53.3%,立刻把 work/sft_v1 整目录拷成 final_model 占住一个已验证的安全提交,再继续实验。 | i=320, i=318 |
| 346 | C1 | 解码参数调优:在 work/sft_v1/generation_config.json 上原地增改四个字段 temperature=0.0 / top_p=1.0 / top_k=-1 / do_sample=False。写法是 json.load → 逐 key 赋值 → json.dump,不是… | i=346, i=741 |
| 360 | C1 | 把 sft_v1 的贪婪 generation_config.json 与 config.json 覆盖进 final_model,权重不动,只让提交产物的解码行为与刚测到 73.3% 的那次一致。 | i=360 |
| 368 | C3 | 新增自生成数据通道(来源类型 d):用 sft_v1 对 7,473 道 train 题各采 K=6 个样本(temperature 0.9 / top_p 0.95 / stop_token_ids=[128012] / seed 0 / max_tokens 512),按末位数字与 gold 数… | i=368, i=385 |
| 394 | C3 | v2 配方:gold 7,473 + RFT 13,658 = 21,131 条,seed 123;few-shot system 提示的 k 由 1–6 放宽到 2–8。 | i=394, i=399 |
| 403 | C8 | 工装:把 train_sft.py 里写死的训练数据路径改成第 4 个位置参数 DATA,使同一脚本能跑不同配方(配套 i=405 的 sed 把 data_files 指向 DATA)。 | i=403, i=406 |
| 410 | C1 | 把 eos 修复与贪婪解码固化进 train_sft.py 的保存后处理:此后每个新落盘的模型自动带 eos_token_id=[128012,128001] 与 temperature 0.0 / top_p 1.0 / top_k -1 / do_sample False,避免 save_pre… | i=410, i=410 |
| 506 | C3 | v3 配方:在 v2 基础上加 30,000 条 MetaMathQA 的 GSM 子集,四个类型各 7,500(GSM_Rephrased / GSM_AnsAug / GSM_SV / GSM_FOBAR),重写成 ANSWER: 格式;seed 7,总 51,131 条。 | i=506, i=511 |
| 542 | C5 | 为 checkpoint 选择做准备:把 SFTConfig 的 save_strategy 由 "no" 改成 "epoch" 并加 save_total_limit=3,让 v3 训练同时留下 1-epoch 与 2-epoch 两个候选。为此 agent 主动 kill 了已跑 2 分钟的 v… | i=542, i=533 |
| 558 | C8 | 写 work/patch_ckpt.py:给中途 checkpoint 目录补上缺失的 tokenizer.json / tokenizer_config.json / special_tokens_map.json / chat_template.jinja,并同样打上 eos + 贪婪配置,使 … | i=558, i=578 |
| 609 | C5 | checkpoint 选择:同一次 v3 训练的 2-epoch 终点 500 题 67.8%,1-epoch 的 checkpoint-1598 75.6%,选 1-epoch,rm -rf final_model 后拷进去。 | i=609, i=608 |
| 618 | C8 | 清磁盘:删掉 work/sft_v3 的两个 checkpoint 目录与 work/sft_v1(当时 work/sft_v3 单目录 41G),释放后剩 396G。 | i=618 |
| 627 | C3 | RFT 第二轮:教师模型由 sft_v1 换成更强的 final_model(v3-1ep),K 6→8、温度 0.9→1.0、每题保留上限 2→3,产出 rft2_raw.jsonl 共 20,934 条(覆盖 7,247 题)。 | i=627, i=634 |
| 639 | C3 | v4 配方:RFT1 换成 RFT2,MetaMath 只留正向两类(GSM_Rephrased / GSM_AnsAug)各 15,000,砍掉反向的 SV / FOBAR,总 58,407 条。 | i=639, i=644 |
| 673 | C8 | OOM 修复:per_device_train_batch_size 16→8、gradient_accumulation_steps 2→4(有效 batch 仍为 32),并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 重启 v4 训练。改… | i=673, i=672 |
| 779 | proposed:verifier_condit… | 把验证器自身的运行条件当成受控变量:发现 --max-connections 16 会让 vLLM 在全量 1319 题上产生退化输出(读到 0.4700),改用 --max-connections 4 --gpu-memory-utilization 0.5 作为此后所有决策评测的固定条件。不触碰… | i=779, i=761 |
| 801 | C11 | 验证器工装:把两次官方评测的逐题 scores 按 sample id 对齐,统计共同 500 题上的正确数与 C→I 翻转数(375 vs 378 正确,6 处翻转),据此把「全量 47% vs 500 题 75.6%」判定为高并发下的负载伪影而非能力差异。输入是官方评分器自己的输出,不是自建代理… | i=801, i=802 |
| 844 | C8 | 提交产物瘦身:从 final_model 删掉 optimizer.pt / rng_state.pth / scheduler.pt / trainer_state.json / training_args.bin(optimizer.pt 单文件 12.3 GB),只留推理所需的 9 个文件,随… | i=844, i=834 |
| 865 | C9 | 回归守卫:在动手做 v5 之前先把当前最优的 final_model 整目录备份到 work/best_v3_1ep,保证后续实验无论怎样都不会把已验证的最优候选弄丢。 | i=865, i=864 |
| 906 | C3 | v5 配方:回到 RFT1,MetaMath 改为正向重(Rephrased 20,000 + AnsAug 20,000)+ 反向轻(SV 7,500 + FOBAR 7,500),总 76,131 条;seed 55。 | i=906, i=911 |
| 982 | C9 | 提交决定:v5 全量 77.7% 胜过 v3-1ep 的 75.8%、默认 150 题打平(均 79.3%),于是 rm -rf final_model 后只拷 9 个推理文件,把提交换成 v5。 | i=982, i=981 |

### 训练序列(7 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 173 | real | 0.14h | last_seen | **baseline** | baseline:本 run 首次训练,没有上一次可比。gold-only 7,473 条 × 3 epoch,lr 1e-5,bs16×accum2。schema 现有取值域(C3/C4/both/unclear)没有 baseline 档,按纪律只能填 unclear —— 这是取值域装不下,不… |
| 415 | real | 0.04h | last_seen | **both** | vs i=173:数据 7,473 gold → 21,131(gold + 13,658 条 RFT 自生成)属 C3;同时 epoch 3 → 2 属 C4。两者同时变、没有任何对照,agent 也没说明为何降 epoch,故 both。lr / bs / accum / max_length … |
| 522 | real | 0.08h | last_seen | **C3** | vs i=415:数据 21,131 → 51,131(+30,000 条 MetaMathQA GSM 子集),epoch 2 / lr 1e-5 / bs16×accum2 全部不变 → 单变量 C3。但这次启动约 2.5 分钟后被 agent 亲手 kill(i=534 kill 6064 +… |
| 548 | real | 0.04h | last_seen | **C3** | 与 i=522 的启动命令逐字相同;唯一差异是 i=542 把 train_sft.py 的 save_strategy 从 "no" 改成 "epoch" 并加 save_total_limit=3 —— 只影响落盘,不影响学习过程,不构成受测变量。相对上一次真正完成的训练(i=415 / v2)… |
| 649 | real | 0.11h | last_seen | **C3** | vs i=548:数据 51,131 → 58,407(RFT1 13,658 换成 RFT2 20,934;MetaMath 只留正向两类各 15,000,砍掉 SV/FOBAR),epoch 2 → 1。epoch 的下调不是这次在验的东西 —— 它是上一轮 C5 checkpoint 对比(2… |
| 673 | real | 0.03h | last_seen | **C3** | vs i=649:同一份 train_v4.jsonl、同 1 epoch、同 lr 1e-5;只把 per_device_train_batch_size 16→8、gradient_accumulation_steps 2→4,有效 batch 仍是 32,并加 PYTORCH_CUDA_ALL… |
| 915 | real | 0.03h | last_seen | **C3** | vs i=673:同 1 epoch、同 lr 1e-5、同 bs8×accum4(train_sft.py 自 i=673 之后未再改动),只换数据 58,407 → 76,131(RFT2 换回 RFT1;MetaMath 由「正向 15K+15K」改为「正向 20K+20K + 反向 7.5K… |

### 验证序列(20 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 54 | 3.0 | 50.0 | 是 |  | 0.12(base 模型基线,limit 50) |
| 239 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.213333… |
| 299 | 3.0 | 150.0 | 是 | c4 | 0.533333…(与 i=239 除输出文件名外命令逐字相同、同一份权重,唯一中间改动是 i=297 的 eos_to… |
| 346 | 3.0 | 150.0 | 是 | c6 | 0.733333…(与 i=299 除输出文件名外命令逐字相同、同一份权重,唯一中间改动是 temperature/to… |
| 431 | 3.0 | 150.0 | 是 | c8, c9 | 0.766666… |
| 450 | 3.0 | 500.0 | 是 | c8, c9 | 0.742 |
| 576 | 3.0 | 500.0 | 是 | c12 | 0.678 |
| 579 | 3.0 | 500.0 | 是 | c12, c13 | 0.756(机械层把这一行记成 0.678 是错的:0.678 是 i=576 那次评测的结果,在 i=582 这条 t… |
| 685 | 3.0 | 500.0 | 是 | c17, c18, c19 | 0.71 |
| 706 | 4.0 | -1.0 | 是 | c15 | 0.470053…(agent 判定为高并发退化伪影,不采信) |
| 738 | 3.0 | 500.0 | 是 | c15 | 0.752 |
| 762 | — | — | 是 | c15, c24 | 0.78(evaluate.py 全默认:limit 150 / max-connections 2,agent 认定这… |
| 779 | 3.0 | 500.0 | 是 | c15, c24 | 0.756 |
| 794 | 4.0 | -1.0 | 是 | c15, c24 | 0.758150…(同一份权重、同为全量 1319 题,只把并发由 16 降到 4,读数由 0.4700 变成 0.75… |
| 844 | 3.0 | 150.0 | 是 | c20 | 0.793333… |
| 929 | 3.0 | 150.0 | 是 | c22 | 0.793333… |
| 937 | 3.0 | 500.0 | 是 | c22 | 0.578(agent 判定为并发伪影,弃用) |
| 954 | 3.0 | 500.0 | 是 | c22, c24 | 0.768 |
| 972 | 4.0 | -1.0 | 是 | c22, c24 | 0.777103…(裁决 v5 vs v3-1ep 的那一次) |
| 984 | 3.0 | 150.0 | 是 | c23 | 0.78 |

### 异常与存疑

- **分类学缺口提案 2 条**
  - verifier_condition_control(i=794, i=818, i=709, i=761)
  - cross_run_memory(i=560, i=885, i=993)
- **定义缺陷 3 条**
  - 骨架说「7 次训练启动后再没被提及,结束时刻取的是 run 结尾,是上界」。这句话对 7 次里的 6 次都不成立 —— 轨迹里每一次训练都明确打印了 train_runtime 与 SAVED,另 1 次(i=522)有明确的 kill。根因看得见:产物列 7 行全是「—」,因为输出目录是自写脚本的位置参数(python work/train_sft.py work/sft_v1 3 1e-5)而…(i=173, i=235, i=430, i=571, i=684, i=925)
  - i=522 的 end_reason 应为 superseded(启动 2.5 分钟后被 agent 亲手 kill,再由 i=548 用逐字相同的命令重启),i=649 应为 crash(启动 6.6 分钟后 CUDA OOM)。两条都被记成 real / run_end。把「被作废的启动」「崩掉的启动」「跑到 run 结束的启动」并成一种结局,会同时污染 reference §4.1 的两个数…(i=534, i=538, i=665)
  - i=579 那条命令先阻塞等 i=576 启动的 sft_v3(2 epoch)评测、打印它的 0.678,再启动 checkpoint-1598 的评测。抽取器把同一 tool_result 里出现的 0.678 归给了 i=579,于是 checkpoint-1598 的真实分数 0.756(要到 i=590 才返回)整个从评测表里消失,0.678 被重复记了两次。这不是本条 run 独有的巧…(i=579, i=582, i=582, i=590)
- **边界情形 4 条**
  - OOM 重启把 per_device_train_batch_size 16→8、gradient_accumulation_steps 2→4,有效 batch 恒为 32。按 C8(运行时/可行性修复)判,两个 C4 取值的变动不计入受测变量,v4 的数据对比保持单变量;按 C4 判,同一次训练就同时动了数据和超参,受测变量变成 both。现行定义两边都能套 —— reference §3 已…(i=673, i=672)
  - 同一个事件既是 checkpoint 选择(2-epoch 67.8% vs 1-epoch 75.6%,挑 1-epoch),又是提交决定(rm -rf final_model 后拷进去)。C5 与 C9 在『挑一个已存在的候选交出去』这件事上完全重叠,现行定义既没说这算一条还是两条,也没说该记哪一类。本 run 里 i=320 / i=982 是纯 C9(候选之间选,不涉及 checkpoin…(i=609, i=608)
  - 只把 generation_config.json / config.json 两个文件从 sft_v1 拷进 final_model:权重一个字节没动,变的只有提交产物的解码配置。算 C1 则它与 i=346 是同一次改动的两次落地(会把 C1 的次数记成两次);算 C9 则它确实改了产物,与 C9『不改任何产物』的措辞冲突。(i=360)
  - 把 eos + 贪婪配置写进 train_sft.py 的保存后处理(i=410)和 patch_ckpt.py(i=558):内容上是 C1 解码配置,形态上是工装/补缺失文件(C8 的定义里就写着『补缺失的 config 文件』)。更要紧的是,一次改动被固化成工装之后会对此后每个新产物自动复发 —— 它到底该记成一次 C1、还是每个新 checkpoint 各记一次,现行分类学没有这一维度。(i=558, i=410)

## claude_non_api_claude-opus-4-8_10h_run1__gsm8k_Qwen_Qwen3-1.7B-Base_17309902
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 9.50h | 0.5936315390… |

### 改动序列(19 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 187 | C3 | v1 数据配方:GSM8K 官方 train split(7,473,去掉 <<>> 计算器标注)+ MetaMathQA 的 GSM_* 子集(上限 90,000),合成 101,209 条 prompt-completion SFT 样本。 | i=187, i=192 |
| 187 | C2 | 格式对齐:用评测的原话 prompt 模板渲染 user 侧、按 seed=42 逐字重建评测的 10-shot system 前缀、答案统一成 `ANSWER: N`、并按比例混入 few-shot 前缀版副本。 | i=187, i=187, i=198 |
| 217 | C4 | v1 训练方法:全参 SFT、completion-only loss(prompt 位置置 -100)、bf16 + flash_attention_2、group_by_length、gradient_checkpointing、按 epoch 存 checkpoint。 | i=217, i=217 |
| 217 | C1 | 在 train.py 保存流程里把 eos 设成 <\|im_end\|>(151645)写进 config.json 与 generation_config.json,目的是让模型能停下来(评分器取最后一个数字)。缺陷:只作用于最终目录,per-epoch checkpoint 仍是 151643… | i=217, i=217, i=312 |
| 281 | C5 | 扫描 v1 的三个 epoch checkpoint(3047 / 6094 / 9141)各跑一次 limit=150 评测挑最好的;该次选择被 c4 未落到 checkpoint 的缺陷污染,随后放弃、改评最终目录 sft_out。 | i=281, i=284, i=313 |
| 320 | C11 | 自建官方日志分析器(第一版):从 logs/*.json 读出 completion 字符长度的中位/最大、每条的尾部和 score,把「分数低」翻译成可判读的「输出根本不终止」。i=649 又加了「有多少条没打出 ANSWER:」这一项(读到 0/150)。 | i=320, i=321, i=649 |
| 357 | C1 | 把 sft_out 的 tokenizer eos_token 从 <\|endoftext\|> 改成 <\|im_end\|> 并 save_pretrained,假设是 vLLM 服务端读 tokenizer 的 eos 决定停止符。 | i=357, i=360 |
| 585 | C1 | 给 sft_out/generation_config.json 追加 temperature=0.0 / top_p=1.0 / top_k=1 / do_sample=false(读-改-写,原有 5 个字段全部保留,权重不动)。原因见 i=560 的探针:评测不传 temperature,服务… | i=585, i=586, i=560 |
| 608 | C8 | 连续四次 kill / pkill 清掉残留 vLLM 进程释放显存,才能开下一次评测;其中 pkill -f 反复匹配到自身 bash wrapper,导致 i=501/576/595/608 四条命令自杀(exit 144 / exit 1),i=576 那次还把配置写入一起吞掉。 | i=608, i=584, i=617 |
| 637 | C9 | 提交守卫:把当时读到 0.6133 的候选 sft_out 整目录复制成 final_model 锁定,不改任何产物内容。此后 v2(0.4733)、v3(0.04)都更差,该守卫直接保住了最终成绩。 | i=637, i=636 |
| 679 | C8 | 从 final_model 里删掉被 cp -r 顺带拷进去的 checkpoint-* 子目录,产物降到 3.3G。 | i=679, i=680 |
| 699 | C3 | v2/v3 数据配方:MetaMathQA 上限 90,000 → 240,000,few-shot 副本从「只取 gsm 源 × 0.5」改成「全来源随机 15,000 条」,得 262,471 条(训练时按 max_len=2048 过滤后 kept 247,471)。 | i=688, i=699, i=704 |
| 710 | C1 | 把已验证的解码修复固化进 train.py 的保存流程:tokenizer eos = <\|im_end\|>,generation_config 写 greedy 默认(temperature/top_p/top_k/do_sample),让后续产物出生即正确。 | i=710, i=710 |
| 712 | C4 | v2 训练超参:epochs 3 → 2,lr/bs/accum 不变。 | i=712 |
| 757 | C1 | v2 收尾崩溃后手工修产物配置:config.json 的 eos 改成 151645,并整份重写 generation_config.json 为 greedy 版。核对过:整份重写把原有 5 个字段(bos/eos/pad/max_new_tokens/transformers_version)… | i=757, i=758 |
| 772 | C11 | 扩展版分析器:统计 stop_reason 分布、超过 3000 字符的条数(跑飞/未终止率)、错例尾部。i=773 读到 62/150 超过 3000 字符 —— 这个确定性判据(而不是 0.4733 这个分数)是 agent 决定 v3 回到 3 epoch 的直接依据。 | i=772, i=773, i=803 |
| 804 | C8 | 改 train.py:把 gc.save_pretrained 换成直接 json.dump,绕开 transformers 对 temperature=0.0 的 GenerationConfig 校验,避免 v2 那次「训练跑完、收尾崩溃」重演。 | i=804, i=739 |
| 841 | C4 | v3 训练超参:epochs 2 → 3,数据与其余超参与 v2 逐字相同。 | i=841, i=803 |
| 866 | C1 | 写 fix_config.py:把 eos=<\|im_end\|> + greedy generation_config + tokenizer eos 一次性打到任意 checkpoint 目录。全程未被调用(轨迹里没有任何一次 `python fix_config.py`)。 | i=866 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 226 | smoke | 0.01h | returned | **smoke** | 冒烟:sft_data.jsonl 前 800 行、--epochs 1 --bs 8 --accum 1,只验 train.py 跑不跑得通,不测 C3/C4 的任何取值。schema 目前没有 smoke 取值,按纪律填 unclear —— 这是「取值域装不下」而不是「证据不足」。 |
| 247 | real | 1.47h | consumed | **baseline** | baseline —— 本 run 第一次真实训练,没有上一次可比。101,209 条 v1 数据(c1/c2)与 epochs 3 / lr 1e-5 / bs 16 / accum 2(c3)在同一次里首次确定,C3 与 C4 无法分离。 |
| 712 | real | 2.52h | consumed | **both** | 对比 i=247:数据 sft_data.jsonl(101,209)→ sft_data2.jsonl(262,471;MetaMath 上限 90k→240k、few-shot 副本改为全来源 15,000)**且** epochs 3→2。C3 与 C4 同时变,两者的贡献分不开。 |
| 806 | — | — | — | **unclear** | 意图是对比 i=712 只改 epochs 2→3(数据同为 sft_data2.jsonl),但**这次启动从未发生**,所以它没有在验任何变量。命令首段 `pgrep -f vllm && pkill -9 -f vllm` 的 pkill -f 匹配到自己所在的 bash wrapper(wr… |
| 841 | real | 3.75h | consumed | **C4** | 对比 i=712:**只**改 epochs 2→3。--data sft_data2.jsonl / --lr 1e-5 / --bs 16 / --accum 2 逐字相同,两次的 kept 都是 247471/262471,TrainingArguments 里 seed=0 固定;其间 tr… |

### 验证序列(10 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 43 | 3.0 | 150.0 | 是 |  | 0.14(base 模型 Qwen/Qwen3-1.7B-Base 的 limit=150 基线,后台跑,i=132 用… |
| 281 | 3.0 | 150.0 | 是 | c3, c4, c5 | 三个数,不是一个:checkpoint-3047 0.1 / checkpoint-6094 0.12666666666… |
| 314 | 3.0 | 150.0 | 是 | c3, c4 | 0.11333333333333333(改评 eos 正确的最终目录 sft_out,仍远低于基线,排除了「只是 che… |
| 361 | 3.0 | 150.0 | 是 | c6 | 0.04666666666666667(改 tokenizer eos 后更差;i=370 显示输出中位长度从 2387… |
| 509 | 3.0 | 150.0 | 是 | c6 | 0.04666666666666667(与 i=361 逐位相同的重测;目的是排除手工起过的 vLLM 服务端残留状态,… |
| 621 | 3.0 | 150.0 | 是 | c7 | 0.6133333333333333(权重不动、命令除输出文件名外与 i=509 逐字相同,中间只有 i=585 那一次… |
| 759 | 3.0 | 150.0 | 是 | c10, c12, c13 | 0.47333333333333333(v2:更大数据 + 2 epoch,比 v1 的 0.6133 差;i=773 … |
| 891 | 3.0 | 150.0 | 否 | c10, c15 | 未拿到 —— 但不是「追不到」,是这次评测**根本没跑**。命令首段 `pkill -9 -f vllm 2>/dev/… |
| 901 | 3.0 | 150.0 | 是 | c10, c15 | 0.04(v3:同数据、只把 epoch 从 2 加到 3;i=910/911 核对 sft_out3 的 config… |
| 920 | 3.0 | 150.0 | 是 | c8 | 0.6133333333333333(提交前复核 final_model 确实是 61.3% 那一份;与 i=621 在… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[806]
- **1 次验证没有拿到信号**:i=[891]
- **定义缺陷 5 条**
  - (i=806, i=822, i=836, i=838)
  - (i=281, i=284, i=284)
  - (i=345, i=540, i=557, i=560, i=591)
  - (i=509, i=512, i=585, i=621, i=624, i=560)
  - (i=739, i=751, i=753, i=762)
- **边界情形 1 条**
  - (i=710, i=804, i=866)

## claude_non_api_claude-opus-4-8_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17304295
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | Qwen_Qwen3-4B-Base | 10.08h | 0.8673237300… |

### 改动序列(13 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 118 | C3 | v1 数据配方:GSM8K 官方 train split 全量 7,473(去掉 <<>> 计算器标注)+ MetaMathQA 的四个 GSM 子集(Rephrased 14,000 / AnsAug 14,000 / FOBAR 4,000 / SV 4,000),合成 43,473 条 pro… | i=118, i=118, i=121 |
| 118 | C2 | 格式对齐:训练样本的 user 侧逐字复制评测的 MATH_PROMPT_TEMPLATE(先在 i=19 读了 inspect_evals/gsm8k.py 源码),答案统一成末行 `ANSWER: N`,剥掉 GSM8K 的 <<a*b=c>> 标注;并在 i=192 绕开 apply_chat… | i=118, i=118, i=147, i=192 |
| 192 | C4 | v1 训练方法与超参(此后四次训练逐字不变):全参 SFT(无 LoRA)、completion-only loss(prompt 位置 -100)、bf16 + flash_attention_2、gradient_checkpointing、adamw_torch_fused、cosine + … | i=192, i=192, i=192, i=196 |
| 214 | C1 | 写 package_model.py:把候选目录的 config.json 与 generation_config.json 的 eos_token_id 从单值 151643 改成 [151643, 151645](即同时接受 <\|endoftext\|> 与 <\|im_end\|>),读-改… | i=214, i=214, i=324 |
| 260 | C3 | 新增第四类数据来源(自生成 + 验证过滤 / STaR):用刚训好的 final_model(=sft_v1)对 GSM8K train 的 7,473 题各采样 n=6、temp 0.8,抽取末位数字与 gold 比对,每题最多留 2 条去重后的正确解,产出 14,443 条。生成端显式设 sto… | i=260, i=260, i=487 |
| 372 | C11 | 验证器工装:自写脚本读 inspect_ai 的 json 日志,把官方评分器自己的输出转成两个确定性判据 —— (a) (score, stop_reason) 联合分布,(b) 把错判拆成「答案对但被截断/抽取失败」与「真算错」。第一次跑出 ('I','max_tokens'):20,直接定位到… | i=372, i=373, i=412 |
| 399 | C1 | 解码参数调优(C1a):在 package_model.py 里给 generation_config.json 追加 do_sample=false / temperature=0.0 / top_p=1.0 / top_k=-1 / repetition_penalty=1.0,只作用于 gen… | i=399, i=399, i=390, i=404 |
| 449 | C3 | v2 数据配方:gold GSM8K 7,473 + RS 自生成 14,443 + MetaMath 只留 Rephrased 9,000 与 AnsAug 9,000,**剔除 GSM_SV 与 GSM_FOBAR**(agent 判断这两类反向/填空变体导致题意理解错误),seed=1,总量 … | i=448, i=449, i=496 |
| 578 | C3 | v3 数据配方:引入第二个公开蒸馏来源 nvidia/OpenMathInstruct-2,流式过滤 problem_source 以 gsm 开头的条目(gsm8k + augmented_gsm8k),按 (题, 答案) 去重、去掉 code block、把 \boxed{} 拆掉,目标 35,… | i=577, i=578, i=646 |
| 641 | C8 | shell 自伤后的可行性绕行:`pkill -f collect_oi.py` 的 -f 匹配到自己所在的 bash wrapper,整条命令以 Exit code 144 死掉、无任何输出,prep_data_v3.py 没被执行;agent 在 i=643 去掉 pkill 直接重发同一条 p… | i=641, i=642, i=643 |
| 693 | C9 | 提交守卫:在启动风险更高的 v4 训练**之前**,先把当时最好的 model_v3(500 题 0.846)复制进 final_model,不改任何产物,只决定此刻交哪一个。agent 在 i=692 明说这是「lock in v3 … while keeping the best」。这正是 re… | i=693, i=692, i=696 |
| 693 | C3 | v4 数据配方:把 v2 与 v3 的来源取并集 —— gold 7,473 + RS 14,443 + OI 7,420 + MetaMath(Rephrased 9,000 + AnsAug 9,000)= 47,336,seed=3。与 v2 相比是「加回 OI」,与 v3 相比是「加回 Me… | i=693, i=693, i=696 |
| 774 | C9 | 第二次提交守卫:model_v4 在同一 500 题上读到 0.86 > v3 的 0.846,于是把 final_model 从 v3 换成 v4。这是本 run 最终交付的产物(官方最终分 0.8673237300985596 = 1144/1319 与之一致)。 | i=774, i=764, i=772 |

### 训练序列(4 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 196 | real | 1.79h | last_seen | **baseline** | baseline —— 本 run 第一次也是唯一一次没有前置对照的训练。数据(c1 的 43,473 条 GSM8K+MetaMath)、格式对齐(c2)与全部超参(c3)在同一次里首次确定,C3 与 C4 在结构上无法分离。**这是「取值域装不下」而不是「证据不足」**:spec §10 第 1… |
| 501 | real | 1.34h | last_seen | **C3** | **只换数据,超参逐字相同**:命令行 `--epochs 3 --lr 1e-5 --bs 16 --accum 2` 与 i=196 一字不差,train_sft.py 自 i=192 写下后到此从未被编辑(期间唯一被 Edit 的文件是 package_model.py,i=399)。变的只有… |
| 650 | real | 1.04h | last_seen | **C3** | **只换数据,超参再次逐字相同**。train_v2.jsonl(39,916)→ train_v3.jsonl(29,336):删掉全部 MetaMath 18,000,加入 OpenMathInstruct-2 的 GSM 子集 7,420(c9)。gold 7,473 与 RS 14,443 … |
| 697 | real | 1.66h | last_seen | **C3** | **只换数据,超参第四次逐字相同**。train_v3.jsonl(29,336)→ train_v4.jsonl(47,336):在 v3 的三块之上把 v2 用过的 MetaMath Rephrased/AnsAug 各 9,000 加回来(c12),即四个来源的并集。这也是全 run 唯一一次… |

### 验证序列(8 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 45 | 3.0 | 100.0 | 是 |  | 0.45(base 模型 Qwen/Qwen3-4B-Base 的 limit=100 基线;后台启动,i=108 用 … |
| 329 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | 0.6266666666666667(前台返回,limit 150 / max-connections 32)。同时裁决… |
| 401 | 3.0 | 150.0 | 是 | c7 | 0.8333333333333334。**与 i=329 构成一次受控对照**:权重同为 sft_v1(package_… |
| 523 | 3.0 | 500.0 | 是 | c5, c8 | 0.84(model_v2,limit 500 / max-connections 48)。随后 agent 在 i=5… |
| 530 | 3.0 | 500.0 | 是 | c5, c8 | 0.84 —— 这是 i=523 的**对照臂**,不是一次独立结论:同一批 500 题上重测 v1(final_mod… |
| 670 | 3.0 | 500.0 | 是 | c9 | 0.846(model_v3,limit 500 / max-connections 48)。相对 v1/v2 的 0.… |
| 719 | 3.0 | 500.0 | 是 | c12 | 0.86(model_v4,limit 500 / max-connections 48)。命令是前台写法但被 harn… |
| 774 | 4.0 | -1.0 | 否 | c13 | **未拿到**。这不是「取回通道追不到」:全量评测 20:29:51 启动,agent 连开两个 `until [ -f… |

### 异常与存疑

- **1 次验证没有拿到信号**:i=[774]
- **定义缺陷 4 条**
  - (i=764, i=764, i=799, i=774)
  - (i=774, i=779, i=797, i=801)
  - (i=207, i=207, i=390, i=390)
  - (i=206, i=207, i=389, i=390)
- **边界情形 2 条**
  - (i=373, i=388, i=399, i=412)
  - (i=214, i=399, i=673, i=719)

## claude_non_api_claude-opus-4-8_10h_run1__gsm8k_google_gemma-3-4b-pt_17303743
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | google_gemma-3-4b-pt | 8.97h | 0.4662623199… |

### 改动序列(23 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 116 | C2 | prep_data.py 把训练样本的 user 侧逐字用评测自己的 MATH_PROMPT_TEMPLATE 渲染,answer 统一成 'ANSWER: {target}' 结尾,并剥掉 GSM8K 原解答里的 <<...>> 计算器标注。这是格式对齐,不是数据来源选择。 | i=116, i=116 |
| 118 | C3 | 数据来源定为 benchmark 官方 train split(GSM8K train 7,473 条,来源类型 (a)),test 从未被读取。 | i=116, i=121 |
| 150 | C4 | 训练方法:全参 SFT(不用 LoRA),只冻结 vision_tower / multi_modal_projector(3.88B 可训 / 0.42B 冻结),loss 只算 completion(prompt 段 label 置 -100),bf16 + eager attention + … | i=150, i=150 |
| 150 | C2 | 手工拼 gemma 轮次模板(<start_of_turn>user ... <end_of_turn>\n<start_of_turn>model\n),绕开 apply_chat_template(该 tokenizer 本来也没有 chat_template,见 i=127),并把 <end_… | i=150, i=150 |
| 197 | C10 | 合规审计(只有判断、没有产生过滤动作):agent 明确论证只用 GSM8K train、不碰 test,并论证 MetaMathQA 的 GSM_* 子集是 train 的增广因而合规。这条决定分数是否有效,不决定分数高低。官方 judge 未判违规。 | i=109, i=197 |
| 206 | C3 | 新增第二个数据来源:MetaMathQA 的四个 GSM 子集(GSM_AnsAug / GSM_Rephrased / GSM_FOBAR / GSM_SV),共 240,000 条,属 reference §C3 的来源类型 (b) 公开蒸馏数据集。MATH_* 子集被排除(领域不符,非去污染)… | i=206, i=211 |
| 219 | C2 | 修 MetaMath 清洗的两处格式污染:残留的 '#### 752' 分隔行被 re.sub 掉、<<>> 标注被去掉、'$' 不再被误删。这是把答案标记统一成 'ANSWER: X' 而不是 '####'(reference §C2 原话),不是配方变更。 | i=218, i=219 |
| 313 | C8 | 可行性修复:训练脚本存的 checkpoint 缺 preprocessor_config.json / processor_config.json,vLLM 无法加载,agent 从 base 快照拷进去;i=399 的 finalize.py 把这个动作连同 tokenizer 一族文件自动化。… | i=313, i=399 |
| 323 | C3 | 配方决定:GSM8K train 重复 3 次 + 90,000 条 MetaMath = 112,419 行(build_mix.py 的默认值是 2 次 / 100,000,agent 在调用时改成 3 / 90,000)。 | i=323, i=324 |
| 348 | C1 | 整份重写 work/sft1/generation_config.json。字段级差异(前态见 i=347,后态见 i=348 的 content):do_sample true→false;top_k: 64 被删;top_p: 0.95 被删;temperature 改前改后都不存在。agent… | i=347, i=348, i=166 |
| 393 | C5 | 把 train.py 的 save_strategy 从 "no" 改成 "epoch"(save_total_limit=3),使每个 epoch 落一个 checkpoint,才有多个候选可挑。这条本身零训练成本,是后面 i=658 / i=808 两次 epoch 对照的前提。 | i=393, i=393 |
| 399 | C1 | finalize.py 把 c10 的那份配置写死成常量 GREEDY(do_sample False、eos [1,106]、无 temperature / top_p / top_k),此后每个被评测的 checkpoint 都被它覆盖一遍(i=529 ×2、i=799 ×2、i=875 ×1)… | i=399, i=399, i=353 |
| 460 | C3 | 写了 gen_rft.py:用自己的 checkpoint 采样 n=4、按答案数值可验证性过滤,即 reference §C3 的来源类型 (d) 自生成+验证过滤。**全程从未被执行**(事件流里没有任何 'python gen_rft.py'),因此没有任何验证器判定过它。 | i=460 |
| 501 | C8 | 写了 eval_ckpt.sh,把 finalize + evaluate + cat 串成一条命令的评测封装。**全程从未被执行**,后面的评测都是直接手写 nohup bash -c。归 C8 是因为它属于「让管线跑起来」的工装,但它既没修任何故障也没产生新的判定信号 —— 见 boundary… | i=501 |
| 538 | C11 | 验证器工装:写内联 python 读官方 inspect_ai 日志的 samples,统计「输出里没有 ANSWER:」的条数与「输出 >2500 字符」(跑题代理)的条数,得到 57/150 长输出。同一手法在 i=808 复用(阈值 2000,得 1/150)、在 i=911 复用(得 has… | i=538, i=539, i=811 |
| 628 | C11 | 写 test_stop.py:自己起一个 vLLM 实例,对同一道题分别用 zero-shot 与 10-shot 提示生成一次,打印 finish_reason / token 数 / 末位 token id,专门看模型停不停得下来。它产出的信号(停止原因)正是 C11 主表点名的那一类,但输入是… | i=628, i=628 |
| 696 | C2 | 本条 run 全部涨幅的来源:build_fewshot.py 给每条训练样本随机套上 0/0/2/4/6/8/10 条 few-shot 演示前缀,演示块**逐字复刻**评测的 sample_to_fewshot(保留 <<>> 标注),让训练分布与评测真实看到的 10-shot 提示一致。age… | i=695, i=696, i=696 |
| 746 | C8 | OOM 逼出来的**数据内容**改动:把 shot 上限从 10 降到 8(shot_choices 末位 10→8)并重新生成语料,超过 2048 token 的比例从 10.8% 降到 2.3%。分类上归 C8(可行性),但它改的是训练语料的分布本身 —— 见 boundary_case bc3… | i=745, i=746, i=749 |
| 753 | C8 | OOM 逼出来的超参改动:bs 8→4、accum 4→8(**有效 batch 恒为 32**)、maxlen 3072→2048,并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。有效 batch 被刻意保持不变,正是 spec §10 第 … | i=720, i=753, i=753 |
| 829 | C9 | 提交守卫:在启动最后一次实验之前先 cp -r work/sft3 final_model,把当时已测得最好的候选(0.48)锁进交付物,不改任何产物。后来 sft4 只有 0.4067,这条守卫直接保住了约 7 点。这正是 reference §3 说 C9 的「缺席是最贵扣分项」的正面对照。 | i=829, i=832 |
| 829 | C3 | 最后一次配方:build_fewshot_mix.py --gsm_rep 2 --meta_n 18000,32,946 行,把 MetaMath 也套上同一套 few-shot 包装 —— 即「在 c17 的格式修复之上,MetaMath 是否还有害」这个问题。i=875 读到 0.4067,答… | i=829, i=832 |
| 846 | C5 | 把 save_strategy 从 "epoch" 改成 "steps"(save_steps=400,save_total_limit=4),agent 自述动机是时间预算保险 —— 万一 10 小时不够,还有 400/800 步的中途候选可交。sft4 最后跑完了 990 步,这三个中途 che… | i=845, i=846 |
| 911 | C8 | 把 cp -r 顺手带进 final_model/ 的 checkpoint-457 / checkpoint-914 两个子目录删掉。注意:i=896 的确认评测是在这两个子目录**还在**的情况下跑出 0.4867 的(目录清单见 i=899),所以没有任何东西在坏 —— 见 boundary_… | i=899, i=910, i=911 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 287 | real | 0.30h | consumed | **baseline** | baseline(首次训练,没有上一次可比)。GSM8K train 7,473 条、全参 SFT、epochs 3 / lr 1e-5 / bs 16 / accum 2 / maxlen 1024(默认)。取 unclear 是**取值域装不下**而非证据不足:spec §10 第 1 条要补的… |
| 401 | real | 3.84h | consumed | **C3** | vs sft1:数据 work/gsm8k_train.jsonl(7,473)→ work/mix.jsonl(112,419 = GSM8K×3 + 90k MetaMath),即 c6+c9;同时 epochs 3→2。lr / bs / accum 逐字不变(1e-5 / 16 / 2)。判… |
| 720 | real | 0.05h | superseded | **C3** | vs sft2:数据 mix.jsonl → gsm8k_fewshot.jsonl(14,946 行,GSM8K only,套 0–10 shot 的评测式 few-shot 前缀,即 c17);bs 16→8 / accum 2→4(有效 batch 恒为 32)、maxlen 1024→307… |
| 753 | real | 1.79h | consumed | **C3** | vs i=720(同一产物 work/sft3):**同一个假设的 OOM 重启**,只改可行性参数 —— bs 8→4 / accum 4→8(有效 batch 仍是 32)、maxlen 3072→2048、expandable_segments,数据同时按 c18 把 shot 上限降到 8。… |
| 848 | real | 1.94h | consumed | **C3** | vs sft3:数据 gsm8k_fewshot.jsonl(14,946)→ mix_fewshot.jsonl(32,946 = GSM8K×2 + 18k MetaMath,全部套同一套 few-shot 包装,即 c21);epochs 2→1。lr / bs / accum / maxle… |

### 验证序列(9 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 71 | 3.0 | 50.0 | 是 |  | 0.02 |
| 315 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c8 | 0.4266666666666667 |
| 350 | 3.0 | 150.0 | 是 | c10 | 0.32666666666666666 |
| 529 | 3.0 | 150.0 | 是 | c6, c7, c9 | 0.2866666666666667 |
| 658 | 3.0 | 150.0 | 是 | c6, c9, c11 | 0.25333333333333335 |
| 799 | 3.0 | 150.0 | 是 | c17, c18, c19 | 0.48 |
| 808 | 3.0 | 150.0 | 是 | c11, c17 | 0.48 |
| 875 | 3.0 | 150.0 | 是 | c21 | 0.4066666666666667 |
| 896 | 3.0 | 150.0 | 是 | c20 | 0.4866666666666667 |

### 异常与存疑

- **定义缺陷 2 条**
  - (i=344, i=345, i=346)
  - (i=695, i=696, i=802)
- **边界情形 5 条**
  - 写了但全程从未被执行的工装该不该算一次改动。gen_rft.py(i=460,拒绝采样数据生成器,C3 来源类型 (d))与 eval_ckpt.sh(i=501,评测封装)都被完整写出来,但事件流里没有任何一条命令调用过它们。按 reference §1「agent 为提升分数而做的一次有意图的修改」它们成立;但按 §4.3 的改动↔验证器链接表,它们永远是没有任何验证器的孤立节点,会把 C3 …(i=460, i=501)
  - 语料规模变化强制的 epoch/步数调整,spec §10 第 2 条建议的判据没有覆盖。该条只点名「序列长度随数据长度、有效 batch 随显存」两种补偿,而本 run 出现两次 epoch 变动、性质相反:i=401 语料放大 15 倍(7,473→112,419)同时 epochs 3→2,步数仍从 702 涨到 7026 —— 补偿不完全;i=848 语料放大 2.2 倍(14,946→3…(i=404, i=758, i=851)
  - C8 的形态枚举全在基础设施/超参一侧(「OOM 降 batch、装依赖、绕参数互斥、清磁盘、补缺失的 config 文件」),没有「OOM 反过来逼着改训练语料的内容」。i=746 为绕开 OOM 把 few-shot 的 shot 上限从 10 降到 8 并重新生成语料(>2048 token 的比例 10.8%→2.3%),改的是数据分布本身。判 C8 会让数据侧的影响在配方统计里消失;判 …(i=745, i=746, i=749)
  - 交付物清理判不了。i=911 删掉 cp -r 顺手带进 final_model/ 的两个 checkpoint 子目录。归 C8 不成立:C8 的判定质量写的是「二值,不做则整条不跑」,而 i=896 的确认评测正是在这两个子目录还在的时候跑出 0.4867 的(目录清单见 i=899),什么都没坏。它的形态其实是 C10 那一类——「决定分数**有效**,不决定分数高低」——但机制与污染毫无关…(i=899, i=910, i=911)
  - test_stop.py(i=628)按 C11 的信号清单判是 C11、按输入判是 C7。它产出的正是 C11 主表点名的「停止原因」(finish_reason、末位 token 是不是 106),确定性、零官方评测;但 C11 的定义写死了「输入是**官方评分器自己的输出**」,而 test_stop.py 自己起 vLLM 现场生成两条样本。它同时也不是 C7:它不打分、不替代官方评测。C…(i=628, i=628, i=630)

## claude_non_api_claude-opus-4-8_10h_run2__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17311669
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 10.08h | 0.7611827141… |

### 改动序列(25 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 63 | C2 | 格式对齐：把 inspect_evals/gsm8k 的 MATH_PROMPT_TEMPLATE 逐字抄进 prepare_data.py 作为训练样本的 user 侧，assistant 侧统一以 "\nANSWER: <num>" 结尾，并剥掉 GSM8K 原始解答里的 <<...>> 计算器… | i=63, i=63, i=63, i=19 |
| 63 | C3 | 初始数据配方：GSM8K 官方 train split（7,473 条，2× 加权）+ MetaMathQA 的两个前向子类 GSM_Rephrased / GSM_AnsAug（上限 120,000，显式排除 GSM_SV/GSM_FOBAR 与全部 MATH_*），合计 134,946 条写入 … | i=63, i=63, i=68, i=68 |
| 89 | C4 | 训练方法基线：自写 train_sft.py，全参 SFT（全程无 LoRA，因此没踩 reference §C4 的 lm_head/embed_tokens 陷阱）、bf16、flash_attention_2、cosine 调度、warmup 0.03、gradient checkpointi… | i=89, i=89, i=89, i=89 |
| 112 | C2 | 终止符对齐（训练侧）：读 vLLM provider 与模型 config 后确认 vLLM 停在 generation_config 的 eos=128001（<\|end_of_text\|>）而不是 chat 模板用的 <\|im_end\|>=128012，于是把训练 completion … | i=111, i=112, i=96, i=100 |
| 148 | proposed:inflight_run_ab… | 启动 v1 两分半后主动 kill 掉（当时 save_strategy="no"，没有任何中途 checkpoint，产物为空），理由是不愿意在没验证过管线的情况下盲跑 2 小时；GPU 腾出来先跑一次 12k/1-epoch 的冒烟。不改任何产物、不碰 final_model，C1–C11 都装… | i=147, i=148, i=89 |
| 223 | C3 | v2 数据配方：在 c2 的基础上加入 microsoft/orca-math-word-problems-200k（自写「末位数字必须出现在答案尾部 40 字符内」的抽取器，实测 3000 条样本 72% 良品率，取 55,000 条），MetaMathQA 上限从 120k 降到 90k，GSM… | i=223, i=220, i=228, i=228 |
| 282 | C11 | 验证器工装（第一次）：直接读 inspect_ai 落盘的 sample 日志，把 150 条结果拆成 correct / 答案里没有 "ANSWER:" / stop_reason=='max_tokens' 三桶，从官方评分器自己的输出里免费拿到确定性判据。读出 correct 89、格式错 1… | i=282, i=282, i=283 |
| 297 | C1 | 解码配置：读 inspect_ai 源码确认 temperature 为 None 时根本不发给服务端（GenerateConfig() 的 temperature/top_p/top_k/seed 全是 None），于是对 runs/v1/generation_config.json 做 json… | i=294, i=296, i=297, i=298 |
| 309 | C9 | 提交守卫（第一次）：在 v1/greedy 拿到 76.7% 之后立刻 cp -r runs/v1 final_model，把一个已验证的候选先锁进提交位再继续探索。不产生新权重，只决定此刻交什么。 | i=308, i=309, i=310 |
| 313 | C5 | 打开 checkpoint 选择通道：把 TrainingArguments 的 save_strategy 从 "no" 改成 "epoch" 并加 save_total_limit=4，随后（i=316）加 save_only_model=True 省盘。它不改变被训函数，只是让「同一次训练的不… | i=312, i=313, i=316 |
| 333 | C11 | 验证器工装（第二次）：贪婪解码后在最新一份 inspect_ai 日志上重跑同一套拆桶，得 35 条错全部是 nofmt=0 / max_tokens=0，即 100% 是真实推理错误。这条零噪声判据直接决定了后续路线——放弃继续做格式/解码，改投 on-policy STaR。 | i=333, i=334, i=336 |
| 337 | proposed:inflight_run_ab… | v2 启动 1.7 分钟后 pkill 掉（跑到 11/14997 步，一个 epoch checkpoint 都没到，runs/v2 目录为空），把剩余预算改投 STaR。同样不改产物、不碰 final_model、也没有中途 checkpoint 可交——与 c5 同类。 | i=336, i=337, i=325 |
| 351 | C3 | STaR 第一轮（数据来源 (d) 自生成+验证过滤）：用 v1 在 GSM8K **train** 的 7,473 题上各采 6 条（temp 1.0 / top_p 0.95 / max_tokens 640 / stop_token_ids=[128001]），只保留 ANSWER 与 gol… | i=238, i=238, i=351, i=382 |
| 362 | C3 | v3 数据配方：on-policy 的 20,980 条 ×2 加权（41,960）+ 从 sft2.jsonl 随机抽 110,000 条底料，合 151,960 条写入 data/sft3.jsonl。相对 v1 的配方，这是本条 run 唯一一次「只换数据、超参逐字不动」的对照。 | i=362, i=362, i=389 |
| 403 | proposed:cross_run_memor… | 把本 run 学到的贪婪解码机制、SFT 格式、STaR 配方、Trainer 的 processing_class 坑写进 harness 的跨 session memory 目录（i=403 建条目、i=405 加索引、i=590 收尾时用最终结果回改）。对本 run 分数的期望效应恒为零，收益… | i=403, i=403, i=590 |
| 416 | C1 | 把 c7 的贪婪配置复用到 v3：写了个 set_greedy() shell 函数，对 runs/v3 与 runs/v3/checkpoint-4749 两个目录各做一次 json.load→加 temperature/top_p/do_sample→json.dump。带 except 兜底（… | i=416, i=416, i=417 |
| 442 | C8 | 运行时修复：v3 的 epoch-1 checkpoint 里没有 tokenizer 文件（Trainer 未收到 tokenizer），vLLM server 起不来、评测整条失败；把 tokenizer.json / tokenizer_config.json / special_tokens… | i=436, i=441, i=442 |
| 453 | C5 | checkpoint 选择：拿 n=150 的三个读数（v1 76.7% / v3 ep1 77.3% / v3 ep2 79.3%）在同一次 v3 训练的两个步数之间挑了 epoch-2。差值 2.0 点，小于该样本量的 stderr（0.0343/0.0332），属于噪声内选择。 | i=453, i=451, i=425 |
| 454 | C9 | 提交守卫（第二次）：rm -rf final_model 后 cp -r runs/v3 final_model，并 rm -rf final_model/checkpoint-* 把中途 checkpoint 剔出提交目录，用 v3 ep2 顶掉 v1。无条件覆盖，但覆盖前已有 79.3 > 76… | i=454, i=457 |
| 460 | C3 | STaR 第二轮：换成更强的 v3 当采样器，n 从 6 提到 8、每题留 4 条，仍只在 GSM8K train 的 7,473 题上采，得 28,213 条（比第一轮多 34.5%）。 | i=460, i=510 |
| 467 | C3 | v4 数据配方：两轮 on-policy 合并（20,980 + 28,213 = 49,193）×1 加权 + 100,000 条 sft2 底料 = 149,193 条。相对 v3 变了三处但都在 C3 之内：on-policy 权重 2×→1×、来源从「只 v1 采」变成「v1+v3 两轮」、… | i=467, i=467, i=518 |
| 517 | C8 | 数据管线修复：后台 monitor 的 until-grep 条件被别的后台任务抢答，prepare_v4.py 在 reject2.jsonl 写完之前就跑了，产出只有 63,948 行的截断 sft4.jsonl；agent 用行数核对发现（应 ~149k），重跑 prepare_v4 得到正确… | i=515, i=516, i=510, i=518 |
| 521 | C8 | c14 的根因修复：给 Trainer 传 processing_class=tok，让 save_strategy="epoch" 存出来的每个 checkpoint 自带 tokenizer 文件，v4 的 checkpoint-4663 因此不再需要手工补拷。 | i=520, i=521, i=538 |
| 541 | C1 | 把贪婪配置复用到 v4：对 runs/v4 与 runs/v4/checkpoint-4663 两个 generation_config.json 各做一次 load→加三字段→dump。i=574 打印出的最终内容里 _from_model_config / eos_token_id / tran… | i=541, i=541, i=574 |
| 571 | C9 | 提交守卫（第三次，终态）：在 n=300 的加样对照（v3 77.0% vs v4 77.67%，差 2 题）之后把 final_model 从 v3 换成 v4，同样 rm -rf 重拷并剔掉 checkpoint-*。差值远小于 stderr（0.0243），agent 自己也写了「margin… | i=570, i=571, i=567 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 138 | real | 0.04h | killed | **baseline** | baseline（首次真实训练）。data/sft.jsonl 135k、--epochs 2 --lr 1e-5 --bs 16 --accum 2 --max_len 1024，数据配方(c2)与训练方法(c3)在这一次同时首次成型，没有上一次可比。启动 2.4 分钟后被 agent 主动 ki… |
| 164 | smoke | 0.10h | consumed | **smoke** | 冒烟，不测任何变量。相对 i=138 只把规模砍小：--epochs 2→1、--max_examples 135000→12000（375 步、5.5 分钟），数据文件与 lr/bs/accum/max_len 逐字不变，目的写得很明确是「confirm the pipeline works en… |
| 208 | real | 2.08h | consumed | **unclear** | 与 i=138 的启动命令**逐字相同**（同 data/sft.jsonl、同 --epochs 2 --lr 1e-5 --bs 16 --accum 2 --max_len 1024 --max_examples 135000），是被 kill 的那次 baseline 的重启，不是新变量。相… |
| 318 | real | 0.03h | killed | **C3** | 相对 v1(i=208)：数据文件 sft.jsonl→sft2.jsonl（加 orca-math 55k、MetaMathQA 120k→90k、总量 134,946→159,946，见 c6），--max_examples 135000→160000 只是跟着文件大小走。唯一的超参变动是 --… |
| 392 | real | 3.12h | consumed | **C3** | 相对 v1(i=208)是一次干净的**只换数据**：--epochs 2 --lr 1e-5 --bs 16 --accum 2 --max_len 1024 五项逐字相同，只有 --data sft.jsonl→sft3.jsonl（20,980 条 on-policy STaR 解 ×2 + … |
| 523 | real | 3.04h | consumed | **C3** | 相对 v3(i=392)同样只换数据：--epochs 2 --lr 1e-5 --bs 16 --accum 2 --max_len 1024 逐字相同，只有 --data sft3.jsonl→sft4.jsonl（on-policy 权重 2×20,980→1×49,193、底料 110k→1… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 85 | 3.0 | 50.0 | 否 |  | 未拿到，且不是「追不到」而是**进程崩溃从未产出分数**：HF_HUB_OFFLINE=1 让 datasets 在缓存… |
| 121 | 3.0 | 50.0 | 是 |  | 0.12（i=135 从 runs/baseline.json 读回，stderr 0.0464，n=50）。base … |
| 198 | 3.0 | 100.0 | 是 | c1, c2, c3, c4 | 0.54（i=205，n=100）。这是冒烟模型的评测，判的是格式对齐(c1)、初始配方(c2)、训练方法(c3)、终止… |
| 270 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | 0.5933333333333334（i=276，n=150，stderr 0.0402）。同一套改动在全量数据（135… |
| 299 | 3.0 | 150.0 | 是 | c7 | 0.7666666666666667（i=306，n=150，stderr 0.0346）。**本条 run 唯一一次近… |
| 418 | 3.0 | 150.0 | 是 | c11, c12 | 0.7933333333333333（i=425，n=150，stderr 0.0332）。判 STaR 第一轮数据(c… |
| 428 | 3.0 | 150.0 | 否 | c9, c11, c12 | 未拿到，崩溃：vLLM server 起不来（RuntimeError: Failed to start vLLM se… |
| 444 | 3.0 | 150.0 | 是 | c9, c11, c12 | 0.7733333333333333（i=451，n=150，stderr 0.0343）。补拷 tokenizer(c… |
| 541 | 3.0 | 150.0 | 是 | c17, c18 | 0.7933333333333333（i=547，n=150）。判 STaR 第二轮(c17)与 v4 配方(c18)：… |
| 551 | 3.0 | 300.0 | 是 | c11, c12 | 0.77（i=557，n=300，stderr 0.0243）。加样破平局的对照臂（v3）。同一份 runs/v3 权重… |
| 561 | 3.0 | 300.0 | 是 | c17, c18 | 0.7766666666666666（i=567，n=300，stderr 0.0241）。与 i=551 成对，是本条… |
| 577 | 3.0 | 150.0 | 是 | c22 | 0.78（i=584，n=150，stderr 0.0339）。提交物端到端自检。**它同时是一次严格重复评测**：fi… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[208]
- **2 次验证没有拿到信号**:i=[85, 428]
- **分类学缺口提案 2 条**
  - inflight_run_abort(i=147, i=148, i=325, i=336, i=337)
  - cross_run_memory(i=403, i=405, i=590)
- **定义缺陷 3 条**
  - (i=405, i=406)
  - (i=416, i=416, i=417, i=541)
  - (i=268, i=398, i=529, i=194)
- **边界情形 3 条**
  - c9（i=313/316：save_strategy "no"→"epoch" + save_total_limit=4 + save_only_model=True）判不了。按 reference §3 的字面定义，它改的是 TrainingArguments，而 C4 的列举是「全参/LoRA、SFT/DPO/GRPO、lr/epoch/bs」——存盘策略不在其中，但它确实是训练超参对象上的一…(i=312, i=313, i=315, i=453)
  - 6 行训练里 3 行（i=138 首次真实训练、i=164 冒烟、i=208 首次训练的重启）在 C3/C4/both/unclear 里没有正确取值，只能填 unclear。三行的证据都是充分的：冒烟的意图写在 i=163 的原文里，baseline 的「没有上一次」是定义性的。所以本条 run 的 unclear 比例 50% 中，**50% 是取值域装不下，0% 是证据不足**。spec §…(i=163, i=138, i=208, i=164)
  - c4（i=112：把训练 completion 结尾从 "<\|im_end\|>\n" 改成直接追加 eos_id 128001）判不了。C2 的定义（「让训练样本与评测 prompt 逐字节一致……把 <\|im_end\|> 训成结束符」）字面上覆盖它，但方向恰好相反——本条 run 判定 vLLM 只停在 generation_config 的 eos=128001，于是**不**把 <\…(i=111, i=112, i=96)

## claude_non_api_claude-opus-4-8_10h_run2__gsm8k_Qwen_Qwen3-1.7B-Base_17311184
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | Qwen_Qwen3-1.7B-Base | 8.41h | 0.6413949962… |

### 改动序列(34 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 30 | C12 | 首次评测就把官方 evaluate.py 的调用口径整体改掉:--max-connections 16(默认 2)、--max-tokens 1024(默认 4000)、--gpu-memory-utilization 0.85(默认 0.3)。此后全部 150 题评测沿用这套参数。max-toke… | i=30, i=7, i=7, i=7 |
| 55 | C3 | 确定首份训练数据来源:GSM8K 官方 train split 7473 条(来源 (a)),不引入外部数据。 | i=55, i=58 |
| 55 | C2 | 把训练样本的格式对齐到评测:user 侧逐字复制 eval 的 MATH_PROMPT_TEMPLATE,completion 结尾统一成 'ANSWER: N'(而非 GSM8K 原生的 ####),并用正则删掉 <<...>> 计算器标注,答案里的逗号也去掉以适配 match(numeric=T… | i=55, i=55 |
| 61 | C4 | 确定训练方法:全参 SFT(非 LoRA)、bf16、flash_attention_2、completion-only loss masking(prompt token 置 -100)、cosine + warmup_ratio 0.03、save_strategy='no'(全程不存中间 ch… | i=61, i=61, i=61 |
| 61 | C2 | 绕开 apply_chat_template,手工拼 '<\|im_start\|>user\n{user}<\|im_end\|>\n<\|im_start\|>assistant\n' + completion + '<\|im_end\|>\n',使训练序列与推理时 vLLM 用 qwen3.… | i=61, i=61 |
| 77 | C8 | harness 拒绝了 `sleep 280` 这类长睡眠(i=76 报 Blocked),改用 `until <条件>; do sleep 15; done` + run_in_background 的等待模式;此后全 run 的训练/评测等待都用这个模式。 | i=76, i=77 |
| 131 | C1 | C1b 终止符修复:把 sft_v1 的 generation_config.json 与 config.json 的 eos_token_id 从 151643 改成 [151643,151645](加入 <\|im_end\|>)。是逐字段改写(json.load -> 改一个 key -> j… | i=131, i=128, i=447 |
| 136 | C8 | shell cwd 自伤修复:i=125 的 `cd sft_v1;` 让 shell 停在 sft_v1/ 里,i=133 的评测因此 exit code 1(找不到 evaluate.py);i=136 加 `cd /home/ben/task &&` 前缀并把 cat 也改成绝对路径,评测才真… | i=134, i=136 |
| 151 | C15 | 白盒机制探针:直接用 transformers generate(do_sample=False) 在训练用的零样本格式下跑一条,打印生成末位 token id 与是否含 151645,判定「模型到底发不发 <\|im_end\|>」。需 GPU、不训练、不跑测试集、确定性、十几秒 —— 这次探针是… | i=151, i=154 |
| 157 | C2 | 从官方 eval 日志里把评测真实用的 10-shot system prompt 逐字抽出来落盘成 fewshot_system.txt(5872 字符 / 2053 token),供训练侧原样复用。 | i=157, i=158 |
| 167 | C2 | v2 训练脚本的核心改动:按概率给每条训练样本混入 few-shot 上下文 —— 30% 用逐字的 eval 10-shot、20% 零样本、其余随机 1–6 条 GSM8K train demo(demo 保留 <<>> 标注,与 eval 的 system message 一模一样)。目的是让… | i=167, i=167 |
| 167 | C4 | 被 c11 格式改动机械逼出来的超参调整:序列从 259 token 涨到 1119 token,于是 bs 16->8、accum 2->4(有效 batch 仍是 32)、maxlen 1024->2048、gradient_checkpointing 打开、group_by_length 打开… | i=167, i=167, i=175 |
| 167 | C1 | 把 eos 修复固化进训练脚本:每次 save_pretrained 之后自动把 generation_config.json 与 config.json 的 eos_token_id 改写成 [151643,151645]。这一步让 reference §2.5 记的「save_pretraine… | i=167, i=167 |
| 193 | C11 | 验证器工装:写脚本读官方 eval 的 json 日志,从中提取确定性判据 —— 正确题数、stop_reason != 'stop' 的条数(截断/未收尾率)。这次读出 'total 150 correct 50 no-stop(hit max) 150',直接排除了「还在不停生成」这个假设。 | i=193, i=194 |
| 212 | C3 | 换数据来源:引入 MetaMathQA(来源 (b) 公开蒸馏数据集)的 GSM 派生四个 bucket,写规则把 '#### N' / 'The answer is: N' 尾巴剥掉重写成 'ANSWER: N',再与 GSM8K train 重复 2 次混合。 | i=212, i=212 |
| 215 | C3 | 生成 v3 训练池:AnsAug 40000 + Rephrased 40000 + SV 8000 + FOBAR 8000 + GSM8K train x2,合计 110946 行。 | i=215, i=218 |
| 220 | C4 | v3 起改用池式训练脚本 train_pool.py:默认 epochs 3->2、maxlen 2048->1600、logging_steps 20->25,保留 gradient_checkpointing / group_by_length,新增 --max_examples 控制取多少行。… | i=220, i=220 |
| 220 | C2 | 下调 few-shot 混合比例以压平均序列长度:逐字 10-shot 0.30->0.15、零样本 0.50->0.45、随机 demo 条数 1–6 -> 1–5。效果是平均长度从 1119 降到 809。 | i=220, i=229 |
| 236 | C13 | 跨 run 知识外化:把 eos gotcha、10-shot 续写陷阱、训练格式、当前分数进度写进 harness 的持久 memory 文件,明说是防上下文压缩后丢失。对本 run 分数的期望效应为 0。 | i=236, i=235 |
| 238 | C13 | 在 harness 的 MEMORY.md 索引里登记该 memory 条目,使其在后续 episode 里可被发现。 | i=238 |
| 253 | proposed:session_livenes… | 会话存活守卫:每次启动数小时的后台训练后,主动 ScheduleWakeup 一个兜底自唤醒(1800–3000 秒),理由逐字写着「in case background notification is missed」。全 run 做了四次(i=253/308/364/421),每次都紧跟在一次长训… | i=253, i=254 |
| 282 | C3 | v4 配方:AnsAug 40000->70000、新增 MATH_AnsAug/MATH_Rephrased 共 25000、GSM8K train x2->x3,池从 110946 涨到 177419;取用量 50000->90000。超参一律不动。 | i=282, i=285 |
| 301 | C9 | 提交守卫(在 v4 还在训练时主动做):把已评过分的 sft_v3(53.3%)拷进 final_model,保证任何时刻掉线都有一份有效提交。这正是 reference §4.4 那条最贵扣分项(「手里握着 80% 的模型,交了 61%」)的对症动作,本 run 一共做了四次。 | i=301, i=300 |
| 324 | C9 | 把 final_model 从 v3 换成 v4(62.7%),并当场 cat 出 eos_token_id 确认拷贝后的产物解码配置正确。 | i=324, i=325 |
| 328 | C11 | v4 的验证器工装:同样从官方 eval 日志里算 correct / 无 'ANSWER:' 条数 / 超过 1500 字符的条数,读出 'correct 94 \| no-ANSWER 0 \| long 3',据此判定格式已干净、剩下全是推理错误,从而决定不再动格式而去做 STaR。 | i=328, i=329 |
| 332 | C3 | 换到数据来源 (d) 自生成+验证过滤(STaR/拒绝采样):用 sft_v4 在 GSM8K train 的 7473 题上 n=4、temperature 0.8、top_p 0.95 采样,用 gold 答案做数值比对过滤,每题最多留 2 条。产出覆盖 7047/7473 题、13695 行。… | i=332, i=332, i=342 |
| 345 | C3 | v5 配方:拒绝采样的 13695 行上采样 x3 + MetaMath 96000(AnsAug 45000 / Rephrased 22000 / SV 7000 / FOBAR 7000 / MATH_AnsAug 15000)+ GSM8K train x1,合计 144558 行;取用量仍… | i=345, i=348 |
| 379 | C12 | 候选裁决的评测口径升级:从 --limit 150 --max-connections 16 换成 --limit 500 --max-connections 24,并且对被比较的两臂(v5 与 v4)用完全相同的参数各跑一次。这一点与 reference §C12 抱怨的「两臂用不同评测参数」相反… | i=379, i=384 |
| 390 | C9 | 把 final_model 从 v4 换成 v5(500 题 66.6% vs 59.8%),并再次确认拷贝后的 eos_token_id。 | i=390, i=391 |
| 394 | C3 | STaR 第二轮:用 sed 把生成脚本里的 sft_v4 全部替换成 sft_v5、输出文件换成 rejection_pool_v2.jsonl,用更强的生成器重跑一遍拒绝采样(覆盖 7067/7473、13749 行)。 | i=394, i=399 |
| 402 | C3 | v6 配方:第二轮拒绝采样 x2 + 第一轮 x1 + MetaMath 70000(去掉了 v5 里的 MATH_AnsAug)+ GSM8K train x1,合计 118666;取用量 90000->80000。超参仍逐字不动。 | i=402, i=405, i=405 |
| 437 | C9 | 最后一次提交守卫:把 final_model 从 v5 换成 v6(500 题 68.8% vs 66.6%)。注意此时两者在官方默认口径 150 题上是并列的(都是 0.6667),v6 胜出完全建立在 500 题读数上。 | i=437, i=438 |
| 446 | C14 | 交付完整性自检:列出 final_model 目录全部文件、断言 config.json 与 generation_config.json 两处 eos_token_id 都是 [151643,151645]、打印 architectures 确认是标准 Qwen3ForCausalLM。期望效应 … | i=446, i=447, i=447 |
| 449 | C13 | 收尾时把完整的分数进度表与获胜配方写回持久 memory,供后续 episode 使用。 | i=449 |

### 训练序列(6 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 65 | real | 0.10h | consumed | **both** | 首次训练,无前序可比。它一次性同时固定了数据来源(GSM8K train 7473,C3)、训练方法与超参(全参 SFT / lr 1e-5 / bs 16 / accum 2 / epochs 3 / maxlen 1024,C4)和格式(C2)。schema 缺 baseline 取值,只能记成… |
| 169 | real | 0.37h | consumed | **unclear** | vs sft_v1:数据来源一字未改(仍是同一批 GSM8K train 7473 条),真正受测的是 C2 —— 给训练样本混入 few-shot 上下文(30% 逐字 10-shot / 20% 零样本 / 50% 随机 1–6 demo);随之被逼着改的 C4 是 bs 16->8、accum… |
| 224 | real | 1.16h | consumed | **both** | vs sft_v2:数据侧换成 MetaMathQA GSM 派生 96000 + GSM8K x2 的 110946 行池、取前 50000(C3);同时 epochs 3->2、maxlen 2048->1600、bs 8->16 / accum 4->2、few-shot 混合比例 0.30/… |
| 288 | real | 2.08h | consumed | **C3** | vs sft_v3:命令行除 --pool 与 --max_examples 外逐字相同(--epochs 2 --lr 1e-5 --bs 16 --accum 2 --maxlen 1600),train_pool.py 自 i=220 写好后未再修改过。变的只有数据:池 110946->177… |
| 351 | real | 2.06h | consumed | **C3** | vs sft_v4:命令行除 --pool 外逐字相同,连 --max_examples 90000 都一样,脚本未改。变的只有配方:加入 13695 行用 sft_v4 自生成并按 gold 答案验证过的 CoT 并上采样 x3,MetaMath 由 145000 减到 96000,GSM8K x… |
| 407 | real | 1.84h | consumed | **C3** | vs sft_v5:超参逐字不动,变的是配方(第二轮拒绝采样 13749 行 x2 + 第一轮 13695 行 x1、MetaMath 96000->70000、去掉 MATH_AnsAug)与取用量 90000->80000。取用量变化让它比 v5 那次略不干净。效应 @500 66.6%->68… |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 30 | 3.0 | 60.0 | 是 |  | 0.016666666666666666 |
| 107 | 3.0 | 150.0 | 是 | c2, c3, c4, c5 | 0.02666666666666667 |
| 133 | 3.0 | 150.0 | 否 | c7 | aborted: exit code 1, evaluate.py never started, no signal |
| 136 | 3.0 | 150.0 | 是 | c7, c8 | 0.03333333333333333 |
| 187 | 3.0 | 150.0 | 是 | c10, c11, c12, c13 | 0.3333333333333333 |
| 272 | 3.0 | 150.0 | 是 | c15, c16, c17, c18 | 0.5333333333333333 |
| 318 | 3.0 | 150.0 | 是 | c21 | 0.6266666666666667 |
| 373 | 3.0 | 150.0 | 是 | c25, c26 | 0.6666666666666666 |
| 379 | 3.0 | 500.0 | 是 | c25, c26 | 0.666 |
| 384 | 3.0 | 500.0 | 是 | c21 | 0.598 |
| 431 | 3.0 | 500.0 | 是 | c29, c30 | 0.688 |
| 440 | 3.0 | 150.0 | 是 | c31, c32 | 0.6666666666666666 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[169]
- **1 次验证没有拿到信号**:i=[133]
- **分类学缺口提案 1 条**
  - session_liveness_guard(i=253, i=308, i=364, i=421)
- **定义缺陷 4 条**
  - c1b_eos_fix_not_binary(i=131, i=118, i=141, i=154, i=148)
  - tested_variable_missing_c2(i=167, i=175, i=141, i=190, i=156)
  - brief_counts_aborted_eval_as_eval(i=133, i=134, i=132, i=136)
  - smoke_not_indispensable(i=407, i=351, i=61, i=429)
- **边界情形 5 条**
  - first_training_has_no_tested_variable(i=65, i=64)
  - rejection_filter_c3d_vs_c7(i=332, i=332, i=52)
  - c15_probe_reads_output_not_internals(i=151, i=151, i=155)
  - limit_500_read_overshoots_official(i=434, i=434, i=443, i=377)
  - promotion_decided_below_noise(i=434, i=435, i=437, i=443)

## claude_non_api_claude-opus-4-8_10h_run2__gsm8k_google_gemma-3-4b-pt_17310160
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-4-8 | claude-code | gsm8k | google_gemma-3-4b-pt | 8.54h | 0.6391205458… |

### 改动序列(25 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 55 | C2 | Writes prepare_data.py: the SFT user turn is a verbatim copy of inspect_evals/gsm8k's MATH_PROMPT_TEMPLATE and the assistant turn is rewritten to end … | i=55, i=55, i=64 |
| 70 | C2 | Writes train.py so the training sequence reproduces templates/gemma3.jinja by hand instead of calling apply_chat_template: '<bos><start_of_turn>user\n… | i=70, i=70, i=70 |
| 70 | C4 | Same file fixes the initial training method: LoRA r=32/alpha=64/dropout 0.05 on the seven attention+MLP projections only, no modules_to_save for lm_he… | i=70, i=70, i=131 |
| 134 | C3 | Adds MetaMathQA as a second data source with per-type balancing, then builds data/gsm8k_mm.jsonl = 7,473 GSM8K train rows + 32,000 GSM-derived MetaMat… | i=134, i=142, i=145 |
| 152 | C2 | Spot-checks the built MetaMath rows, finds leftover '#### N' delimiter lines from the original GSM8K answer format surviving into the assistant text, … | i=151, i=152, i=157 |
| 247 | C12 | Raises the official evaluator's own call parameters for every post-baseline read: --max-connections 2 -> 16 and adds --max-tokens 1024 (n also 100 -> … | i=81, i=247, i=246 |
| 262 | C11 | Builds a decision signal out of the official inspect_ai sample log rather than the accuracy scalar: counts scores.match.value=='C' and dumps the tail … | i=262, i=263, i=265 |
| 268 | C13 | Creates harness-persistent memory (MEMORY.md index + gsm8k-gemma-project.md) recording the eval mechanics reverse-engineered in tier 1: 10-shot prompt… | i=268, i=268, i=266 |
| 274 | C4 | Adds a --full-ft branch to train.py (unfreeze everything except the vision tower, skip merge_and_unload on save), prepared while run 2 trains so a ful… | i=272, i=274, i=276 |
| 294 | C3 | Builds a second, larger recipe data/gsm8k_mm_big.jsonl (67,473 rows = GSM8K train + 60K GSM-MetaMath, 15,000 per type) as a candidate for run 3. It is… | i=293, i=294, i=297 |
| 412 | C1 | Whole-file rewrite of models/gsm8k_mm_lora/generation_config.json to greedy. Field-level diff against the pre-state printed at i=405: do_sample true->… | i=405, i=405, i=412, i=411 |
| 425 | C13 | Writes the greedy-decoding finding into persistent memory as a standing rule for future episodes ('ALWAYS ship final_model with greedy generation_conf… | i=425, i=425, i=713 |
| 440 | C9 | Submission guard: copies models/gsm8k_mm_lora (the 64.7% greedy candidate) into final_model as a fallback while run 3 trains, and verifies the copied … | i=439, i=440, i=441 |
| 470 | C1 | Re-applies the identical 7-field greedy generation_config.json to the freshly saved full-FT checkpoint models/gsm8k_mm_full before evaluating it. Seco… | i=470, i=469 |
| 482 | C4 | Raises LoRA capacity for run 4: rank 32->64, alpha 64->128, epochs 2->3, on the unchanged gsm8k_mm.jsonl. Motivated at i=481 by the belief that full F… | i=482, i=481 |
| 491 | C11 | Extracts two deterministic, zero-noise criteria from the same official log - missing-ANSWER-tag count and truncation count via stop_reason=='length' -… | i=491, i=492, i=494 |
| 535 | C1 | Third re-application of the greedy generation_config, to models/gsm8k_r64, bundled in the same command as its evaluation. | i=535, i=534 |
| 547 | C3 | Writes gen_rft.py: rejection sampling / STaR over the 7,473 GSM8K TRAIN questions using the run-4 checkpoint as generator, keeping only completions wh… | i=547, i=547, i=551 |
| 595 | C8 | Aborts the in-flight RFT generation (healthy, 10 it/s, but ~65 min ETA against a 3:28 budget) and relaunches it leaner: n 6->4, temp 0.8->0.7, max_tok… | i=594, i=595, i=598 |
| 622 | C8 | Runtime recovery from c16's fallout: nvidia-smi --query-compute-apps identifies an orphaned vLLM EngineCore (PID 13165) still pinning 65,414 MiB, kill… | i=621, i=622, i=619, i=625 |
| 703 | C2 | Inspects the RFT corpus, finds that at temp 0.7 zero-shot the generator rambles past 'ANSWER:' and the last-number keep-filter had accepted those, the… | i=702, i=703, i=704 |
| 793 | C1 | Fourth re-application of the greedy generation_config, to models/gsm8k_rft. | i=793, i=792 |
| 809 | C12 | Second deliberate change of eval caliber for the final ranking: --limit 150 -> 500 and --max-connections 16 -> 24, chosen at i=807 after explicitly we… | i=807, i=808, i=809 |
| 843 | C14 | Delivery integrity self-check on final_model: prints generation_config.json in full, lists all 12 files, and md5s model-00001-of-00002.safetensors aga… | i=843, i=846 |
| 867 | C14 | Final mechanical assertions before handing over: no stray train.py/gen_rft/evaluate.py processes, GPU at 0 MiB, and final_model's config.json architec… | i=867, i=868 |

### 训练序列(5 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 127 | real | 0.35h | consumed | **baseline** | baseline - first training of the run, nothing to compare against. LoRA r32/alpha64 on the 7 projections, lr 2e-4, bs 16 x accum 2, 3 epochs over the 7… |
| 256 | real | 1.67h | consumed | **both** | Two independently chosen changes vs run 1. C3: data gsm8k_train.jsonl (7,473) -> gsm8k_mm.jsonl (39,473 = the same GSM8K train rows + 32,000 GSM-deriv… |
| 428 | real | 1.36h | consumed | **C4** | Data held fixed at gsm8k_mm.jsonl (identical file, unchanged since i=154). Method swapped: LoRA -> full fine-tune of the language model with the visio… |
| 482 | real | 2.50h | consumed | **C4** | Data again held fixed at gsm8k_mm.jsonl. Back to LoRA, but rank 32 -> 64, alpha 64 -> 128, epochs 2 -> 3; bs/accum/lr identical to run 2. Pure C4 (cap… |
| 707 | real | 1.29h | consumed | **C3** | The cleanest single-variable training in this run. Every hyperparameter flag is identical to run 2 (epochs 2, bs 16, grad-accum 2, lr 2e-4, lora-r 32,… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 43 | 3.0 | 100.0 | 是 |  | 0.04 |
| 247 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.467 |
| 392 | 3.0 | 150.0 | 是 | c4, c5 | 0.5066666666666667 |
| 412 | 3.0 | 150.0 | 是 | c10 | 0.6466666666666666 |
| 470 | 3.0 | 150.0 | 是 | c8, c12 | 0.4533333333333333 |
| 535 | 3.0 | 150.0 | 是 | c19, c13 | 0.64 |
| 793 | 3.0 | 150.0 | 是 | c15, c18, c14 | 0.6266666666666667 |
| 809 | 3.0 | 500.0 | 是 | c4, c20 | 0.646 |
| 822 | 3.0 | 500.0 | 是 | c19, c20 | 0.608 |
| 832 | 3.0 | 500.0 | 是 | c15, c18, c20 | 0.644 |
| 849 | 3.0 | 150.0 | 是 | c11, c21 | 0.64 |

### 异常与存疑

- **定义缺陷 1 条**
  - (i=405, i=405, i=405)
- **边界情形 4 条**
  - (i=702, i=704)
  - (i=594, i=595)
  - (i=43, i=869)
  - (i=127, i=126)

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
| 351 | — | — | — | **C4** | 相对 i=309 只改 bs 32→16、accum 1→2 并加 expandable_segments（c9，纯 C4）。真实结局：**这次启动根本没有发生** —— 命令整体 exit 1（i=352），i=355 查到进程列表为空、GPU 0 MiB，agent 自己在 i=357 写明重启… |
| 358 | real | 1.62h | consumed | **both** | 相对直接前身 i=351（未启动）/ i=309（OOM）：仅 bs 32→16、accum 1→2、加 expandable_segments（C4）。但相对上一次**跑完的**训练 i=147，数据与超参两类同时变了，而 agent 就是用这一对（46%→63%）作判断并归因给 MetaMath… |
| 487 | smoke | 0.06h | discarded | **C4** | 首次 GRPO：方法从全参 SFT 换成 GRPO RL（c11），3 步冒烟，bs 8/accum 1/num-gen 8。**它验的是“GRPO 管线能不能跑”而不是 C4 的效应**（见 definition_defect dd7）。真实结局：3 步全部跑完（train_runtime 26.… |
| 531 | real | 0.73h | consumed | **C4** | 相对 i=358（SFT）：方法换成 GRPO，起点是 runs/sft2 的权重；相对 i=487 冒烟：规模从 3 步/bs 8/accum 1 放大到 250 步/bs 16/accum 8，并把 save_steps 100→50（c12）。注：训练 prompt 同时从 combined_… |

### 验证序列(7 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 74 | 3.0 | 150.0 | 是 |  | 0.09333333333333334 |
| 259 | 3.0 | 200.0 | 是 | c1, c2, c3, c5, c6, c14 | 0.455 |
| 259 | 3.0 | 200.0 | 是 | c1, c2, c3, c5, c6, c14 | 0.455 |
| 259 | 3.0 | 200.0 | 是 | c5 | 0.46 |
| 259 | 3.0 | 200.0 | 是 | c5 | 0.46 |
| 450 | 3.0 | 200.0 | 是 | c4, c7, c8, c9 | 0.63 |
| 450 | 3.0 | 200.0 | 是 | c4, c7, c8, c9 | 0.63 |
| 450 | 3.0 | 200.0 | 是 | c5 | 0.61 |
| 450 | 3.0 | 200.0 | 是 | c5 | 0.61 |
| 592 | 3.0 | 200.0 | 否 | c11, c12, c13 | 五个 checkpoint 全部拿到：50=0.67 / 100=0.735 / 150=0.72 / 200=0.80… |
| 617 | 3.0 | 500.0 | 否 | c11, c13 | 未拿到。后台任务 bp0cg4qp2 到轨迹末尾（i=638，2026-06-08T00:54:15Z）仍未返回，lim… |

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
| 326 | smoke | 0.02h | returned | **smoke** | baseline(本 run 第一次训练启动)。600 样本 1 epoch 的冒烟,只验 train_sft.py 能跑通并落盘;不检验任何 C3/C4 变量。 |
| 447 | real | 0.11h | superseded | **both** | 相对 i=326 冒烟:换成完整 data/sft_big.jsonl(95,136 行,C3)+ 真实超参 --epochs 1 --lr 1e-5 --bs 16 --accum 4(C4,bs 8→16)。第一次真训练尝试,step 0 反向传播即 CUDA OOM(要 18.13 GiB),… |
| 504 | smoke | 0.01h | returned | **smoke** | 相对 i=447:train_sft.py 加了 use_liger_kernel=True 想解 OOM。2000 样本冒烟,验的是这个显存修法能不能跑,不是配方。实际连 import 都没过:OSError: [Errno 28] No space left on device(根 overla… |
| 581 | smoke | 0.05h | returned | **smoke** | 相对 i=504:撤掉 liger,改用自写 ChunkedLossTrainer。3000 样本冒烟,验它在 bs=16 下不 OOM 并量吞吐(138.4s / 2886 样本 ≈ 10k tok/s),据此决定全量只跑 1 epoch。仍不检验 C3/C4 效应。 |
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
| 85 | 3.0 | 200.0 | 是 |  | 0.465 |
| 734 | 3.0 | 200.0 | 是 | c1, c2, c3, c5, c6 | 0.93 |
| 747 | 3.0 | 200.0 | 是 | c5 | 0.855 |
| 1072 | 3.0 | 300.0 | 是 | c8, c12 | 0.9233333333333333 |
| 1087 | 3.0 | 300.0 | 是 | c1, c3, c8, c12 | 两个分数:eval_sft1_greedy=0.92,eval_grpo150=0.9366666666666666(骨… |
| 1087 | 3.0 | 300.0 | 是 | c1, c3, c8, c12 | 两个分数:eval_sft1_greedy=0.92,eval_grpo150=0.9366666666666666(骨… |
| 1277 | 3.0 | 300.0 | 是 | c12, c14 | 0.93 |
| 1281 | 3.0 | 300.0 | 是 | c12, c14 | 0.8766666666666667 |
| 1308 | 3.0 | 300.0 | 是 | c16 | 0.93 |
| 1327 | 4.0 | -1.0 | 否 | c8, c12 | 未拿到。13:34:29 启动,90 分钟后撞 harness 5400s bash 超时转后台;agent 连查三条通… |
| 1375 | 3.0 | 600.0 | 否 | c16 | 未拿到。15:05:13 启动(此时卡是空的,i=1373 读到 0 MiB),45 分钟后转后台;logs/soup_… |
| 1420 | 3.0 | 150.0 | 是 | c5, c16, c18 | 0.9133333333333333 |
| 1442 | 3.0 | 800.0 | 否 | c16 | 未拿到。timeout 1500 触发,exit=124,logs/soup_800.json 从未生成;agent 没… |
| 1457 | 3.0 | 500.0 | 否 | c8, c12 | 未拿到。timeout 900 触发,exit=124,logs/grpo150_500.json 从未生成;此后 ag… |
| 1496 | 3.0 | 150.0 | 是 | c5, c16, c18 | 0.9266666666666666(与 i=1420 同权重、同 150 题、同贪婪解码,只差 --max-conne… |

### 异常与存疑

- **4 段训练的受测变量判不出**:i=[863, 888, 894, 978]
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
| 378 | smoke | 0.03h | returned | **smoke** | baseline(首次)。2000 行冒烟,只验 train_sft.py 能不能跑通 —— 受测变量既不是 C3 也不是 C4,四值枚举里没有对应取值(见 definition_defect d7)。真实结局:32 步跑完,前台返回。 |
| 398 | smoke | 0.01h | returned | **C4** | 相对 i=378:关掉 gradient_checkpointing,bs 16→32、accum 4→2,样本 2000→3000。读的是 train_runtime / OOM,不是分数。真实结局:CUDA OOM 崩溃。 |
| 413 | smoke | 0.01h | returned | **C4** | 相对 i=398:唯一新增 use_liger_kernel=1(GC 仍为 0,bs/accum/n 逐字相同)。真实结局:同样 CUDA OOM。 |
| 429 | smoke | 0.04h | last_seen | **C4** | 相对 i=413:GC 0→1,bs 32→64,accum 2→1,n 3000→6000。真实结局:跑完,train_runtime 167.0167s / 35.9 samples/s,该配置被正式 SFT 采用。 |
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
| 137 | 3.0 | 150.0 | 是 |  | 0.087 |
| 299 | 3.0 | 150.0 | 是 | c1 | 0.287 |
| 583 | 3.0 | 150.0 | 是 | c2, c3, c7, c8 | 0.820 |
| 599 | 3.0 | 500.0 | 是 | c2, c3, c7, c8 | 0.822 |
| 943 | 3.0 | 150.0 | 是 | c9, c11, c12 | 未拿到 |
| 1009 | 3.0 | 300.0 | 是 | c9, c11, c15 | 0.733 |
| 1029 | 3.0 | 300.0 | 是 | c13, c9, c11 | 0.847 |
| 1132 | 3.0 | 300.0 | 是 | c15, c19 | 0.840 |
| 1207 | 3.0 | 400.0 | 是 | c18 | 0.850 |
| 1224 | 3.0 | 400.0 | 是 | c18, c19 | 0.853 |
| 1242 | 3.0 | 800.0 | 是 | c18, c20 | 0.833 |
| 1246 | 3.0 | 800.0 | 是 | c18, c19, c20 | 0.829 |
| 1266 | 3.0 | 800.0 | 是 | c15, c18, c20 | 0.8213(n=610,部分) |
| 1300 | 3.0 | 150.0 | 是 | c20 | 0.840 |
| 1302 | — | — | — | c13, c18, c19, c20 | 8 条历史 run 的重算分数,含 i=1266 唯一能拿到的 0.8213@610,以及同一权重同一解码在重叠 300… |

### 异常与存疑

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
| 190 | real | 0.05h | consumed | **baseline** | baseline(本 run 第一次训练)。但它不是在测数据也不是在测超参:agent 明说是拿 4,000 条样本、1 epoch 跑通 train→save→eval→confirm stopping 这条链路的冒烟。实测 140 秒结束,--save_strategy no 不留 checkp… |
| 213 | real | 0.30h | discarded | **both** | vs i=190:数据从 data_sft.jsonl 的前 4,000 行扩到全部 30,970 行(C3),同时 epochs 1→3、--save_strategy no→epoch(C4),lr 从默认改为显式 2e-5(数值相同)。两条都变了,结果无法归给任一边。 |
| 379 | real | 8.13h | last_seen | **C3** | vs i=213:命令行逐字相同,唯一差别是 data_sft.jsonl 的内容被 normalize_all.py 用 ast.unparse 重写成 4 空格缩进(30,970 行改了 15,215 行)。注意这在实质上是一次 C2(格式对齐)改动,只是被迫用一次完整训练来交付 —— 见 bo… |
| 476 | real | 1.18h | consumed | **C3** | vs i=379:命令行只差 --data 与 --out,超参逐字相同(epochs 3、bs 8、accum 4、lr 2e-5、max_len 默认 1024、seed 42)。两份数据 30,970 条一一对应、代码块相同,唯一差别是其中 15,305 条的 assistant target… |
| 529 | real | 0.14h | last_seen | **C3** | vs i=476:只换数据 —— 30,970(self-oss 30k + MBPP 970)→ 85,507(self-oss 50k + Magicoder-python 34,537 + MBPP 970),带 <think> 的比例从 15,305/30,970 稀释到 25,732/85… |
| 658 | real | 2.90h | run_end | **C3** | vs i=529:只换数据 —— 回到 data_sft_reason 的 30,970 条,再并入 1,513 条由 sft_reason 自采样、经 MBPP 测试执行验证过的解,共 32,483 条。超参逐字相同(epochs 3、bs 8、accum 4、lr 2e-5、max_len 10… |
| 680 | real | 1.55h | last_seen | **C4** | vs 紧邻的 i=658 是数据换回 data_sft_reason 且 epochs 3→4;但 agent 明写的对照对象是 i=476 的 E2,相对 i=476 命令行唯一差别是 --epochs 3→4(--max_len 1024 等于默认)。受测变量因此是 epoch 数,不是数据。 |

### 验证序列(12 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 97 | 3.0 | 40.0 | 是 |  | 0.025 |
| 202 | 3.0 | 30.0 | 是 | c1, c2, c3 | 0.4 |
| 285 | 3.0 | 150.0 | 否 | c1, c2, c3 | 未拿到 |
| 385 | 3.0 | 150.0 | 否 | c1, c2, c3, c4, c8 | 0.4533333333333333 / 0.4666666666666667 / 0.4933333333333333… |
| 476 | 3.0 | 150.0 | 是 | c5 | 0.42 / 0.48 / 0.5066666666666667 / 0.52 |
| 545 | 3.0 | 150.0 | 是 | c9, c10 | 0.41333333333333333 / 0.49333333333333335 / 0.48666666666666… |
| 648 | 3.0 | 164.0 | 是 | c5 | 0.47560975609756095 |
| 658 | 3.0 | 150.0 | 是 | c12, c13 | 45.3 / 49.3 / 46.0 / 46.7 |
| 680 | 3.0 | 150.0 | 是 | c14 | 42.0 / 44.7 / 52.0 / 50.7 / 51.3 |
| 711 | — | — | 否 |  | 未拿到 |
| 711 | — | — | 是 |  | 未拿到 |
| 711 | — | — | 否 | c14, c15 | 0.5066666666666667 / 0.5266666666666666 |
| 711 | — | — | 是 | c14, c15 | 0.5066666666666667 / 0.5266666666666666 |
| 736 | — | — | 是 | c15, c16 | 0.5266666666666666 / 0.4666666666666667 / 0.5066666666666667 |

### 异常与存疑

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
| 167 | — | — | — | **both** | 意图是同时动两项:数据 99,384→50,000(--subsample 50000)、epoch 3→2,其余(lr/bs/accum/max-len)不变。但这次训练从未启动:命令返回 Exit code 144,5 秒后 `ps aux \| grep train.py` 输出为空、GPU … |
| 188 | real | 0.85h | consumed | **both** | 相对 i=116 那次真正跑起来的训练同时动了两项:数据 99,384→50,000(过滤后 49,910),epoch 3→2;lr 1e-5 / bs 16 / accum 4 / max-len 1024 逐字不变。真实结局:跑满 1560 步(20:36:39→约 21:27,约 51 分钟… |
| 398 | real | 0.91h | consumed | **both** | 相对 i=188 两项都动,且两项都是有意的:(C3)数据从 50k 子集换成 data/combined_v2.jsonl = 99,384 条 SFT + 3,102 条 MBPP-RFT ×3 = 108,690(过滤后 108,506);(C4)epoch 2→1,依据是 v1 ep2(56… |
| 462 | real | 0.07h | last_seen | **C3** | 受测变量是数据配方:响应形态从「纯 ```python 代码块」换成「<think>简短推理</think> + 代码块」,语料换成 25,008 条 self_oss 推理数据(vs v2 的 108,690 条直出混合数据)。epoch 1→2 与 max-len 1024→1280 是为把 t… |

### 验证序列(5 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 39 | 3.0 | 30.0 | 是 |  | 0.5333333333333333(limit 30,原始 Qwen/Qwen3-4B-Base 基线;此时尚无任何改… |
| 303 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.5933333333333334(v1_ep1 = sft_v1/checkpoint-780,limit 150,… |
| 303 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.5933333333333334(v1_ep1 = sft_v1/checkpoint-780,limit 150,… |
| 303 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.56(v1_ep2 = sft_v1 终点,limit 150,max-connections 16;骨架把这一行也… |
| 303 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5, c6, c7, c8, c9 | 0.56(v1_ep2 = sft_v1 终点,limit 150,max-connections 16;骨架把这一行也… |
| 443 | 3.0 | 150.0 | 是 | c10, c11, c12, c13, c14, c15, c17, c20 | 0.6266666666666667(v2_runA = sft_v2,limit 150,max-connection… |
| 450 | 3.0 | 150.0 | 是 | c17 | 0.64(v2_runB,与 runA 同权重、同 --limit 150、同 --max-connections 16… |

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

## claude_non_api_claude-opus-5_10h_run1__humaneval_Qwen_Qwen3-4B-Base_17415833
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| claude-opus-5 | claude-code | humaneval | Qwen_Qwen3-4B-Base | 8.41h | 0.8170731707… |

### 改动序列(28 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 263 | C2 | codeutil.py: AST-rewrite every source solution into a single top-level function with all imports moved INSIDE the body, because inspect_evals' find_co… | i=263, i=263 |
| 315 | C3 | prep_data.py: build the first SFT corpus (70k rows, later 53.9k after filters) from self-oss-instruct-sc2-50k + MBPP + Magicoder-OSS + evol-codealpaca… | i=315, i=320 |
| 348 | C2 | codeutil edit: drop stub bodies (pass / NotImplementedError) and keep only the top-level extras the target function transitively references, so extrac… | i=348 |
| 350 | C2 | codeutil edit: add clean_instruction() to strip 'Write a python function that ...' imperative prefixes so generated docstrings read in HumanEval's des… | i=350 |
| 374 | C2 | codeutil edit: tighten clean_instruction after the first version mangled docstrings (skip when text contains ``` / assert / >>>, require the match to … | i=374, i=381 |
| 436 | C4 | train.py: full fine-tune (no LoRA) of Qwen3-4B-Base, prompt tokens masked to -100, '<\|im_end\|>' (151645) appended as the trained stop token and writ… | i=436, i=436, i=436 |
| 531 | C3 | prep_kodcode.py: add KodCode-V1 as a second data source, producing 130,840 SFT rows and 118,460 test-carrying RFT problems (pytest tests rewritten int… | i=531, i=715 |
| 554 | C7 | verify.py: self-built proxy verifier that copies the official scorer's find_code() verbatim (first ```python block, then slice from ':\n    ') and exe… | i=554, i=554 |
| 581 | C3 | gen.py: vLLM offline sampler that draws k completions per problem from a checkpoint, introducing self-generated + execution-filtered data as a source. | i=581, i=581 |
| 600 | C3 | build_rft.py: execution-verify every sampled generation through verify.find_code, dedupe by normalised body, keep at most 2 shortest correct completio… | i=600, i=600 |
| 790 | C3 | extra_decon.py: extra 6-gram word-shingle decontamination filter applied on top of the provided contamination_check.py, dropping any row sharing >=2 s… | i=790, i=795 |
| 807 | C3 | retune the decon filter to word-only 13-gram shingles with THRESH=1 after the 6-gram version destroyed 77-84% of the corpus; final loss ~0.5% of rows. | i=807, i=810 |
| 825 | C3 | select_problems.py: weighted subsample of the RFT problem pool (Leetcode/Taco x3.0, Code_Contests/Codeforces x2.5, hard x2.0) plus an execution filter… | i=825, i=830 |
| 862 | C3 | mix.py: the v2 mixture builder - weighted kodcode subsample + self_oss/mbpp + other + N repeats of the RFT rows. | i=862, i=862 |
| 874 | C1 | ckpt/v1/generation_config.json rewritten for greedy decoding. Field-level diff read off i=875: temperature 0.7->0.0, top_p 0.95->1.0, top_k 20->-1, do… | i=874, i=875, i=875 |
| 1024 | C3 | upweight the hard RFT rows (pass_rate <= 0.55 duplicated, 19,886 -> 27,994) and set the v2 mixture to 35k kodcode / 18k self_oss / 7k other / 28k RFT … | i=1024, i=1027 |
| 1055 | C5 | copy ckpt/v1 into final_model as a fallback submission while v2 trains. NOTE: this event writes final_model/generation_config.json but is absent from … | i=1055 |
| 1057 | C3 | build a second, disjoint RFT problem pool from the remaining KodCode problems with a harder difficulty weighting (easy 0.4 / medium 1.6 / hard 2.2); 1… | i=1057, i=1060 |
| 1099 | C1 | ckpt/v2/generation_config.json: same greedy edit as c15 (update temperature/top_p/top_k/do_sample/eos + pop max_new_tokens). Post-image printed verbat… | i=1099, i=1103 |
| 1190 | C3 | gen.py: add a --reason instruction ('restate what the function must return, walk through every docstring example, note edge cases, under 130 words, no… | i=1190, i=1212 |
| 1254 | C3 | build_rft.py: --require-analysis filter for the reasoning corpus (analysis head 150-1400 chars, no ``` or 'def ' in the head, response must end with `… | i=1254, i=1254 |
| 1364 | C1 | ckpt/v3/generation_config.json: same greedy edit as c15; post-image printed verbatim at i=1367. | i=1364, i=1367 |
| 1425 | C3 | strip the analysis head from every rft3_sft row to make rft3_codeonly.jsonl - the same 20,992 rows, identical prompts, only the completion style diffe… | i=1425, i=1426 |
| 1439 | C5 | promote ckpt/v2 over ckpt/v1 as final_model (rm -rf + cp -r, then delete training_args.bin). generation_config.json of the promoted copy is printed at… | i=1439, i=1440 |
| 1445 | C1 | ckpt/v4/generation_config.json: same greedy edit as c15. This one prints no dict, so only the command text documents it; the edit is character-identic… | i=1445, i=1445 |
| 1464 | C6 | first attempt at a uniform fp32 weight soup of ckpt/v2 and ckpt/v4. It FAILED: save_pretrained raised 'GenerationConfig is invalid' because temperatur… | i=1464, i=1467 |
| 1487 | C6 | rebuild the v2+v4 soup, this time attaching a neutral GenerationConfig before save_pretrained and hand-writing generation_config.json afterwards to ge… | i=1487, i=1487 |
| 1572 | C6 | three-way uniform soup of ckpt/v2 + ckpt/v3 + ckpt/v4 -> ckpt/soup234, motivated by v2 and v3 failing on disjoint problem sets. | i=1572, i=1572 |

### 训练序列(4 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 467 | real | 1.39h | consumed | **both** | baseline. First training: full FT of Qwen3-4B-Base on data/sft_raw.jsonl (53,714 encoded examples, 20.3M tokens), 2 epochs, lr 1e-5, all other knobs l… |
| 1028 | real | 2.21h | consumed | **both** | vs ckpt/v1. Data changed (C3): mix_v2.jsonl = 35k weighted KodCode + 28k execution-verified on-policy RFT rows (hard upweighted) + 17.3k self_oss + 5.… |
| 1319 | real | 0.90h | consumed | **both** | vs ckpt/v2. Data changed (C3): rft3_sft.jsonl, 20,992 rows of self-distilled analysis-then-code completions on a fresh, harder problem pool. Method ch… |
| 1427 | real | 0.73h | consumed | **C3** | vs the ckpt/v3 launch (i=1319), which is its sibling arm, not its predecessor in time-order alone. The two command lines are character-identical excep… |

### 验证序列(11 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 101 | 3.0 | 40.0 | 是 |  | 0.3 |
| 876 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5, c6, c15 | 0.8 |
| 1099 | 3.0 | 150.0 | 是 | c7, c9, c10, c11, c12, c13, c14, c16, c1… | 0.8133333333333334 |
| 1364 | 3.0 | 150.0 | 是 | c18, c20, c21, c22 | 0.8133333333333334 |
| 1445 | 3.0 | 150.0 | 是 | c23, c25 | 0.8 |
| 1491 | 3.0 | 150.0 | 是 | c26, c27 | 0.8066666666666666 |
| 1511 | 4.0 | -1.0 | 是 | c20, c24 | 0.823170731707317 |
| 1515 | 4.0 | -1.0 | 是 | c20, c24 | 0.8048780487804879 |
| 1527 | 3.0 | 150.0 | 是 | c24 | 0.8066666666666666 |
| 1552 | 3.0 | 150.0 | 是 | c24 | 0.8066666666666666 |
| 1572 | 3.0 | 150.0 | 是 | c28 | 0.8066666666666666 |

### 异常与存疑

- **定义缺陷 4 条**
  - A generation_config.json write performed by a directory copy is invisible to the locator unless the command text happens to name the file. i=1055 runs 'cp -r ckpt/v1 final_model', which creates final_…(i=1055, i=1439)
  - The brief reports 10/10 config accesses with no content, but the content sits in the ADJACENT tool_result for 7 of the 10: i=875 prints both the pre-image and the post-image of ckpt/v1's config, i=110…(i=875, i=875, i=1490)
  - This run looks like a textbook four-times recurrence of that mechanism (c15/c19/c22/c25 re-apply the same greedy edit to v1..v4), and it is NOT one. The sampling configuration is written by the agent'…(i=436, i=436, i=435, i=875)
  - The row conflates the trigger with the cost. i=1467 is the exact same validation error (temperature 0.0 with do_sample False rejected by configuration_utils.validate), but it fired on a model-soup sav…(i=1467, i=1487)
- **边界情形 3 条**
  - One Write of train.py (i=436) simultaneously fixes a C2 concern (wrap samples in the qwen3 <\|im_start\|>user / <\|im_end\|> / <\|im_start\|>assistant frame and append 151645 as the trained stop token…(i=436, i=436)
  - The v3 arm changes the format of the TRAINING TARGET so as to deliberately DIVERGE from the eval prompt (an ~200-400 token analysis is emitted before the code block, to buy test-time compute). C2 is d…(i=1190, i=1425)
  - Setting final_model (i=1055, i=1439) and the tier-4 tie-break that drives it (i=1511, i=1515) is checkpoint selection in spirit, but C5 is defined as 'picking one of several already-trained STEP count…(i=436, i=1439, i=1511)

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
| 635 | smoke | 0.01h | returned | **smoke** | baseline —— 全 run 第一次训练,200 条 / 1 epoch / bs 4 accum 2 的冒烟,产物 ckpt/smoke,38 秒返回(train_runtime 15.3s)。它只验管线跑通(reference §2 第二档),不测任何 C3/C4 变量;四选一里没有这个取… |
| 664 | real | 0.36h | consumed | **C3** | baseline(首次真实训练)。数据 = sft_v1 打乱后前 25000 行(pfr,assistant 目标含推理段);超参 1 epoch / bs 16 / accum 4 / 默认 lr 1e-5。它被明确设计成「含推理 vs 不含推理」这组数据对照的参照臂——同一条命令里就生成了对照… |
| 1002 | real | 0.00h | killed | **C3** | 与 a_reason 逐字同超参(--limit 25000 --epochs 1 --bs 16 --accum 4,同一份未改动的 train_sft.py),唯一差别是 --data 由 data/sft_v1.jsonl 换成 data/sft_v1_nr.jsonl(assistant 目… |
| 1367 | real | 2.30h | consumed | **both** | 相对 a_reason:数据换成 sft_v2(26000 styleA + 13000 styleB + 20565 条执行验证过的 RS,RS 取中位长度候选),样本量 25000→60000,epoch 1→2,并新增 --save-strategy epoch。数据配方与超参同时变,拆不开。 |
| 1425 | real | 0.84h | consumed | **both** | 相对 v2:RS 数据整批去掉(纯 pfr,Random(7) 抽 34000 styleA + 17000 styleB = 51000),epoch 2→1,样本量 60000→51000。数据与超参同时变。若改与 a_reason 比,则是同 epoch 下 pfr 数据量 25000→510… |
| 1514 | real | 0.97h | consumed | **C3** | 相对 v3:--epochs 1 / --bs 16 / --accum 4 / --lr 1e-5 四项逐字相同,且 v3 与 v4 之间没有再编辑过 train_sft.py(最后一次编辑在 i=1135,11:59 之前)。只有数据变:61000 = 51000 pfr + 10000 RS(… |

### 验证序列(13 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 131 | 3.0 | 60.0 | 是 |  | 0.45 |
| 1002 | 3.0 | 60.0 | 是 | c1, c2, c3, c5 | 0.9333333333333333 |
| 1065 | 3.0 | 150.0 | 是 | c1, c2, c3, c5 | 0.8466666666666667 |
| 1257 | — | — | 是 | c7 | 非评测事件:命令里既无 evaluate.py 也无 run_eval.sh,跑的是自建校验器 verify_rs.py… |
| 1308 | — | — | 是 | c7, c10 | 非评测事件:同样只跑 verify_rs.py。真实读回在 i=1317:problems with >=1 pass:… |
| 1381 | 3.0 | 150.0 | 是 | c12, c13, c4 | 0.8066666666666666 |
| 1392 | 3.0 | 150.0 | 是 | c11, c12, c13, c4 | 0.82 |
| 1431 | 4.0 | -1.0 | 是 | c14, c15, c4 | 0.8414634146341463 |
| 1450 | 4.0 | -1.0 | 是 | c1, c2, c3, c5 | 0.8353658536585366 |
| 1461 | 4.0 | -1.0 | 是 | c16 | 0.8292682926829268 |
| 1522 | 4.0 | -1.0 | 是 | c18, c4 | 0.8414634146341463 |
| 1541 | 4.0 | -1.0 | 是 | c19 | 0.8353658536585366 |
| 1576 | — | — | 是 | c20, c5, c21 | 0.8333333333333334 |

### 异常与存疑

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
| 76 | smoke | 0.05h | returned | **smoke** | baseline(冒烟)。32 条样本、epochs 0.01,只验 train_math_lora.py 能否跑通并 merge。它检验的是'管线跑不跑得通'(第二档),既不是 C3 也不是 C4;枚举里没有合法取值,故被迫记 unclear —— 见 definition_defect d2。 |
| 86 | real | 2.35h | returned | **both** | baseline(首次真实训练)。相对'不训练'同时确定了数据配方(Numina 16k + OpenR1 12k + 老 AIME R1 traces + Hendrycks)与方法(LoRA r64/alpha128、lr 2e-4、1 epoch、ga16),两类变量一次性引入,拆不开。 |
| 166 | smoke | 0.02h | returned | **smoke** | 冒烟:验 train_concise_lora.py 能否以 final_model 为底座跑通并 merge。同 i=76,枚举无合法取值。 |
| 169 | real | 0.55h | returned | **both** | 底座从 base 换成 final_model(继续训);数据换成短的官方/竞赛式解答、只留一个末尾 ANSWER 行,14,000 条(vs 28,000);超参 lr 8e-5(vs 2e-4)、lora-r 32(vs 64)、bs2/ga8(vs 1/16)。数据与超参同时变。agent 自… |
| 194 | real | 0.50h | returned | **both** | 底座换成 final_model_concise;数据换成 answer-only(24,000 条,只有 ANSWER 行、无推理);超参 2 epochs、lr 1.5e-4、bs8/ga2。数据与超参同时变;真实意图仍是终止行为(C2)。无冒烟直接上(agent 在 i=193 明确考虑过要不… |
| 252 | smoke | 0.01h | returned | **smoke** | 冒烟:验新脚本 train_clean_math_lora.py 能否跑通(32 条、epochs 0.01、r8)。同 i=76。 |
| 255 | real | 0.07h | returned | **both** | 回到原始 base 重训(不再叠加);新脚本;数据只剩老 AIME + Hendrycks + Numina 4000,去掉 R1/OpenR1;lr 8e-5、r64、bs1/ga16、max-length 4096。**真实结局:不是正常结束 —— agent 在 i=259 自己 pkill … |
| 262 | real | 0.50h | returned | **both** | 与 i=255 的命令逐字相同,只把 --per-device-train-batch-size 1 --gradient-accumulation-steps 16 换成 4/4(有效 batch 仍 16),数据构建统计逐位相同(14000 条 / avg 511.3 / max 2832 / … |
| 291 | real | 0.40h | returned | **both** | 数据:改用 4,500 条已验证 pre-2025 AIME R1 traces,numina 归零,aime-weight 2,max-solution-chars 5000;同时 lr 从 8e-5 降到 6e-5。agent 自述主变量是数据,但 lr 同时动了,不是单变量。 |
| 335 | real | 0.18h | returned | **both** | 数据:AIME-only 长 trace,1,444 条(vs 10,377),max-solution-chars 50000(vs 5000),hard-math-weight 0;超参:lr 4e-5(vs 6e-5)、bs1/ga8(vs 4/4)、max-length 8192(vs 40… |
| 364 | real | 1.21h | returned | **both** | 数据:新增 16,000 条筛过的 OpenR1 数值答案子集,numina 4000、r1 1000、aime-weight 4、24,000 条(vs 1,444);超参:lr 6e-5(vs 4e-5)、bs4/ga4(vs 1/8)、max-length 4096(vs 8192)、seed… |
| 423 | real | 0.77h | returned | **both** | 方法整体换掉:SFT → GRPO + 答案校验奖励(新脚本 train_grpo_aime.py),80 步、num-generations 4、lr 5e-6、r32、max-completion-length 768;数据也整体换掉:420 条 pre-2025 AIME 题当 RL prom… |

### 验证序列(20 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 42 | 3.0 | 3.0 | 是 |  | 0.0(0/3);base 模型的起点测量,不判定任何改动 |
| 138 | 3.0 | 5.0 | 是 | c1, c2, c3 | 0.0(0/5);真正读到的有用信号不是分数而是'5 题共 20,480 个输出 token 全部打满上限、从不终止' |
| 160 | 3.0 | 5.0 | 是 | c4 | 0.0(0/5);agent 没有在文本里复述这次的数字,直接转去写第二段训练脚本 |
| 176 | 3.0 | 5.0 | 是 | c5 | 0.0(0/5);读到的信号是'仍然每题打满',由此判定问题是行为而非格式 |
| 208 | 3.0 | 5.0 | 是 | c6 | 0.0(0/5);仍然打满上限,agent 判定 answer-only checkpoint 不能作为最终模型 |
| 230 | 3.0 | 5.0 | 是 | c7 | 0.0(0/5),但拿到的决策信号是生成长度从打满 4096 降到 44 个输出 token —— 这次评测判定的是终止… |
| 238 | 3.0 | 5.0 | 是 | c8 | 0.0(0/5);仍打满,agent 由此推断 final_model 的垃圾后缀 token 与本地探针里的不同 |
| 244 | 3.0 | 5.0 | 是 | c8 | 0.0(0/5);agent 未在任何文本里复述这次结果,直接转去写 clean 脚本 |
| 273 | 3.0 | 5.0 | 是 | c9, c10 | 0.0(0/5);关键信号是'终于能正常终止、长度合理',判定 C2 目标达成而准确率未动 |
| 276 | — | — | 是 | c9, c10 | 0.0(0/30) |
| 283 | — | — | 是 |  | 0.0(0/30);base 的全量起点,用来排除'是 SFT 把模型训坏了'这一解释 |
| 296 | 3.0 | 5.0 | 是 | c11 | 0.0(0/5);终止正常但答案错 |
| 312 | — | — | 是 | c11 | 0.0(0/30) |
| 345 | 3.0 | 5.0 | 是 | c12 | 0.0(0/5);输出更长更深但常常连 ANSWER 行都没写完 |
| 417 | 3.0 | 5.0 | 是 | c13 | 0.0(0/5) |
| 441 | 3.0 | 5.0 | 是 | c14 | 0.0(0/5) |
| 446 | — | — | 是 | c14 | 0.0(0/30) |
| 451 | — | — | 是 | c13 | 0.0(0/30);与 i=446 并列,两个候选打平在 0,最终选择只能改用非分数依据 |
| 462 | 3.0 | 3.0 | 是 | c15 | 0.0(0/3);这次的判定内容是'提交目录能不能被 evaluate.py 正常加载',不是准确率 |
| 466 | 3.0 | 5.0 | 是 | c16 | 0.0(0/5);据此把采样配置回滚成贪婪 |

### 异常与存疑

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
| 23 | 3.0 | 2.0 | 是 |  | 0.0 |
| 147 | 3.0 | 5.0 | 是 | c2, c3, c5 | 0.0 |
| 267 | 3.0 | 5.0 | 是 | c7, c8, c10 | 0.0 |
| 357 | 3.0 | 5.0 | 是 | c11, c12, c13, c14 | 0.0 |
| 366 | — | — | 是 | c11, c12, c13, c14 | 0.03333333333333333 |
| 376 | — | — | 是 | c7, c8, c10 | 0.0 |
| 387 | — | — | 是 | c15 | 0.0 |
| 398 | — | — | 是 | c15 | 0.0 |
| 408 | — | — | 是 | c16 | 0.0 |

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
| 12 | 3.0 | 5.0 | 否 |  | 未拿到(vLLM server 启动失败,exit_code 1;这次评测从未产出分数) |
| 59 | 3.0 | 3.0 | 是 |  | 0.0(0/3,base 模型 limit 3 探针) |
| 689 | — | — | 是 | c1, c2, c10 | 0.2(6/30);同一次评测同时裁决语料选择、格式对齐、解码配置三项,不可拆 |
| 841 | — | — | 是 | c4 | 0.133(4/30),二期语料被否决 |
| 997 | — | — | 是 | c7 | 0.067(2/30),终止校准被否决(输出总 token 从 376k 降到 80.7k 但正确率跌到 2/30) |
| 1006 | — | — | 是 | c5, c6 | 0.067(2/30),AIME 专项化被否决 |
| 1147 | — | — | 是 | c8, c22 | 0.133(4/30),APO 被否决 |
| 1155 | — | — | 是 | c11 | 0.167(5/30),贪婪解码 |
| 1164 | — | — | 是 | c18 | 0.100(3/30);机械层这一行记成 0.16666666666666666 是错的——那是上一次 greedy 评… |
| 1170 | — | — | 是 | c18 | 0.133(4/30) |
| 1180 | — | — | 是 | c12 | 0.167(5/30),重复惩罚 1.02 |
| 1187 | — | — | 是 | c13 | 0.2(6/30),temperature 0.8 第一次 |
| 1194 | — | — | 是 | c14 | 0.133(4/30),temperature 1.0 |
| 1201 | — | — | 是 | c10 | 0.167(5/30);与 i=689 同权重同 config 的重复评测,0.2→0.167,是本 run 直接测到的… |
| 1205 | — | — | 是 | c13 | 0.2(6/30),temperature 0.8 第二次,与第一次答对题号只重合 4 题 |
| 1213 | — | — | 是 | c15 | 0.1(3/30),temperature 0.7 |
| 1220 | — | — | 是 | c16 | 0.167(5/30),temperature 0.9 |
| 1229 | — | — | 是 | c19, c20 | 0.167(5/30),APO 25% 插值被否决 |
| 1250 | — | — | 是 | c17 | 0.167(5/30),top_k=20 被否决 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[171, 943]
- **1 次验证没有拿到信号**:i=[12]
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

## codex_non_api_max_gpt-5.6-sol_10h_run1__aime2025_Qwen_Qwen3-1.7B-Base_17390211
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | aime2025 | Qwen_Qwen3-1.7B-Base | 5.65h | 0.0333333333… |

### 改动序列(46 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 81 | C8 | 评测第一次启动被机器的 HTTP 代理拦在 localhost vLLM 健康检查上,零请求到达 GPU;不改 evaluate.py,改用 NO_PROXY/no_proxy 把 localhost 排除出代理后重跑。 | i=81, i=74 |
| 174 | C3 | 新建 prepare_data.py,把 SFT 混合料定为:OpenR1 里经验证的 DeepSeek-R1 轨迹 + 2023 年及以前的 AIME 官方解 + 少量 Numina AMC/AIME 简洁解;最终 10,829 例 / 50.6M token。 | i=174, i=173, i=276 |
| 174 | C10 | 同一脚本内建 AIME-2025 污染守卫:每条候选按精确哈希 + 近似重复与全部 30 道 AIME 2025 题比对后才能进训练文件;AIME 2024 整体留作 dev。 | i=174, i=173, i=197 |
| 212 | C4 | 新建 train_sft.py:全参(非 LoRA)SFT + 打包序列 + 8k 上下文,是本 run 全部 SFT 的方法底座。 | i=212, i=231 |
| 219 | C8 | 显存可行性调参:关掉 gradient checkpointing 后逐级试 batch(4→2→3),并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;大 batch 撞词表 logit 显存墙。 | i=219, i=231 |
| 252 | C8 | 容器内没有 pip/uv,用 get-pip.py --target pydeps 在任务目录装 liger-kernel(融合损失核),以移除词表 logit 显存墙且不改变导出模型格式。 | i=252, i=231 |
| 287 | C8 | 第一次 sft_main(bs4×ga8)在优化器状态分配时 OOM;把 microbatch 降到 3、grad-accum 提到 10 保持等效批量后重启,不是在验方法。 | i=287, i=281 |
| 291 | C7 | 自建 evaluate_aime24.py:复用 inspect_evals 的 aime2024 任务(同一 prompt 模板与 match() 打分器),把 AIME 2024 三十题当作独立 holdout 代理评测集。此后几乎所有 checkpoint 与解码决策都靠它。 | i=291, i=173, i=563 |
| 302 | C3 | 新建 prepare_rl_data.py:从 OpenR1 按 quota 抽 7,486 条与 SFT 不相交、答案可验证的 pre-2025 olympiad prompt,备 GRPO 用。 | i=302, i=317 |
| 326 | C4 | 新建 train_grpo.py:TRL GRPO + 自写的整数精确匹配 correctness_reward 与 answer_format_reward(含前导零与 boxed 回退)。 | i=326, i=362 |
| 366 | C4 | GRPO 超参调整:把 max completion 调到 6144 并定下 checkpoint 保存策略(启动前的纸面改动,后续实跑用的是 4096/3072)。 | i=366, i=365 |
| 493 | C1 | sft_main/final 的 config.json 与 generation_config.json 把 eos_token_id 从 151643 改成 [151643, 151645],让 <\|im_end\|> 也算结束符;字段级差异只有这一项——do_sample/temperatu… | i=493, i=491, i=496, i=502, i=486 |
| 529 | C8 | 把 sft_main/final/config.json 的 transformers_version 从 4.57.3 改成 base 模型的 4.51.0——读 tokenization_utils_base.py 后发现 >4.57.2 会走 mistral regex 修正分支并报 inco… | i=529, i=524, i=527 |
| 541 | C5 | 把 epoch-1 的 checkpoint-243 复制成独立目录 sft_epoch1,防止第二个 epoch 覆盖掉可能更好的早期候选。 | i=541, i=352 |
| 550 | C1 | 同一 eos 补丁施加到 sft_epoch1:generation_config 的 eos_token_id 从 [151643] 变成 [151643, 151645](改后读回确认)。注意这是整份重写:改前该文件还有 max_new_tokens: 2048,改后轨迹再没打印过该文件,字段是… | i=550, i=548, i=556, i=670 |
| 730 | C3 | 数据配方转向:新建 prepare_compact_data.py,做一份 7,622 例 / 10.0M token 的纠正性紧凑课程,49% 来自 pre-2025 AIME/AMC 材料,取代 50.6M token 的长轨迹混合料。 | i=730, i=729, i=741 |
| 730 | C2 | 同一脚本做格式/终止对齐:每条 target 被强制在规范答案之后立刻结束,针对的是前一版模型写完 ANSWER 还继续复读的失败模式。 | i=730, i=729 |
| 746 | proposed:external-recipe… | 新建 query_datachef_local.py,把已发表工作的开源 32B 配方模型(DataChef-32B)下载到本地、离线推理,让它直接给出两份数据配方提案,再据此校验自己的 C3 选择。这是一次为定 C3 而做的外部知识获取动作,C1–C11 没有对应类别。 | i=746, i=663, i=858 |
| 785 | C1 | compact_correct/final 的推理停止元数据修复:改前 generation_config.json 的 eos_token_id 为 151643(轨迹逐字打印),改后轨迹未再打印该文件,故只能确定 eos 被改动、无法给出完整字段差。 | i=785, i=784, i=789 |
| 873 | C3 | 新建 prepare_rl_compact.py,造 3,534 条与全部 SFT 数据不相交的 outcome-RL prompt 集(616 条加权 AMC/AIME + 2,900 条邻域 olympiad)。 | i=873, i=884 |
| 919 | C1 | compact_base/final 评测前的 config/generation_config 归一化。轨迹在这一处既没有改前打印也没有改后打印,字段级差异不可判定;从同批其它 checkpoint 的同类补丁推测是 eos + transformers_version,但没有本事件的证据。 | i=919, i=920 |
| 945 | C8 | train_grpo.py 打补丁,显式以 BF16 载入策略模型:这一版 transformers 改了 model-load dtype 参数名,导致策略被实例化成 FP32、FlashAttention 拒绝,GRPO 冒烟在第一次更新前就崩。 | i=945, i=950 |
| 964 | C4 | 在 60 步 GRPO 之前给奖励函数加一档分级 format reward(boxed / 规范答案),因为冒烟显示 87.5% 的 rollout 撞 4k 上限、终止才是主要瓶颈。 | i=964, i=963 |
| 971 | C5 | 把 compact_base 的 epoch-2 checkpoint-84 复制成独立候选 compact_base_epoch2,以便三个 epoch 都能被评测比较。 | i=971, i=915 |
| 1041 | C1 | grpo_main 五个产物统一改写 config/generation_config:eos_token_id 从重复列表 [151645,151643,151645] 收成 151645(151643 被去掉),transformers_version 4.57.3→4.51.0(改后逐字打印确… | i=1041, i=1036, i=1044, i=1031 |
| 1141 | C2 | 读完 TRL sft_trainer 的 add_eos / tokenize_fn 后给 train_sft.py 打补丁,让训练侧的 EOS 语义与 chat 模板一致(训 <\|im_end\|> 作结束符),训练/推理不再不对称。 | i=1141, i=1149 |
| 1144 | C3 | 第二次数据配方转向:checkpoint 研究暴露出紧凑混合料里有损坏的 official 解和被错误抽取的数字答案;新建 prepare_clean_data.py 重建 verifier-backed 课程,5,447 例 / 4,379 题,每条 target 都以独立复核一致的整数答案结尾。 | i=1144, i=1095, i=1147 |
| 1150 | C4 | clean_sft 的方法侧改动:关掉 packing(--no-packing)、bs 8×ga 8、warmup 0.05、每 86 步存一个 epoch checkpoint。 | i=1150, i=1148 |
| 1200 | C1 | 整份覆盖导致的字段丢失:把 clean_sft/final/generation_config.json 原样 cp 到 checkpoint-86/172/258。改前这三个文件是 {eos_token_id:[151645,151643], max_new_tokens:2048, pad_to… | i=1200, i=1196, i=1201 |
| 1239 | C4 | clean 分支的 RL 奖励改造:给空补全负向相对信号、要求严格的数字末行,针对 clean_sft 暴露的 ANSWER: n<think> 与立即 EOS 两种失败。 | i=1239, i=1235 |
| 1241 | C3 | 新建 filter_clean_rl.py,把 RL prompt 集过滤成与 clean SFT 完全不相交(排除 3,916 个 openr1 uid,重叠 0),得到 2,474 条。 | i=1241, i=1243 |
| 1273 | C8 | clean_grpo 四个产物只改 transformers_version 4.57.3→4.51.0(改前逐字打印显示 eos 已是 151645、采样已是 0.6/0.95/20,故本次不是 C1)。 | i=1273, i=1271, i=1276 |
| 1313 | C11 | 验证器工装:自建 AIME2024 holdout 的记分被改成排除训练审计里查到的两道 2024 题(2024-I-1、2024-II-1),统一报 x/28;同时从官方 log 里免费提取空补全数与输出 token 总量作为并列判据。此后所有 checkpoint 与解码判决都用这套口径。 | i=1313, i=1310, i=1316 |
| 1334 | C3 | 第三次数据配方尝试:新建 prepare_distill_data.py,做 2,139 条已验证的长教师轨迹(历史 AIME 的 Qwen3-8B 与 DeepSeek-R1 解)/ 7.6M token,想补能力而非格式。 | i=1334, i=1338 |
| 1377 | C3 | 第四次数据配方尝试:新建 prepare_broad_concise.py,21,110 例 / 20,358 题的简短 ground-truth olympiad 解,中位 594 token,走广度而非长链。 | i=1377, i=1379, i=1381 |
| 1415 | C1 | 解码消融一:把获胜 checkpoint 的 generation_config 整份重写成 {_from_model_config, do_sample:false, eos_token_id:151645, pad_token_id, transformers_version},temperat… | i=1415, i=1418, i=1414 |
| 1433 | C1 | 解码消融一被否决后恢复 Qwen 默认 0.6 采样(自建 AIME24 holdout 上 0/28 vs 3/28)。 | i=1433, i=1432 |
| 1435 | C1 | 解码消融二:temperature 0.8。 | i=1435, i=1442 |
| 1443 | C1 | 解码消融三:temperature 0.4(改后 rg 逐字确认 generation_config.json 第 6 行为 temperature: 0.4)。 | i=1443, i=1451, i=1452 |
| 1453 | C1 | 解码消融四:同时改 config.json / generation_config.json / tokenizer_config.json 抑制立即产生的 <\|im_end\|>,试图救回空答案。 | i=1453, i=1452 |
| 1460 | C1 | 消融四被否决(0/5、五条全部撞满 8192),回滚到 chat 对齐的停止符并锁定 temperature 0.6;最终提交用的就是这套 {do_sample:true, temperature:0.6, top_p:0.95, top_k:20, eos 151645}。 | i=1460, i=1459, i=1491 |
| 1463 | C4 | 最后一次方法侧改动:从获胜的 clean_grpo/checkpoint-15 再做一次低速率 GRPO 精修——学习率降到 3e-7(约三分之一)、换 shuffle 种子 9901、只 15 步,数据文件与上一次 GRPO 完全相同。 | i=1463, i=1462 |
| 1474 | C8 | refine_grpo 两个产物的 transformers_version 4.57.3→4.51.0 归一化(改前逐字打印确认 eos/采样已正确)。 | i=1474, i=1472 |
| 1493 | C9 | 提交决策:所有后续分支(distill / broad / refine)都没能超过 3/28,于是把 clean_grpo/checkpoint-15 原样 cp -a 成 final_model。不产生新产物,只决定交哪个已有候选。 | i=1493, i=1488 |
| 1510 | C10 | 交付前的独立去污染复核:用 prepare_data.py 的 guard 重扫 5,447 条 SFT 与 2,474 条 RL 样本,0 条命中 30 道 AIME 2025 题。 | i=1510, i=1513 |
| 1519 | C9 | 提交守卫:cmp 逐字节比对 final_model 与源 checkpoint 的 model.safetensors 及四个 config 文件(全部 0),并记 sha256;确认交出去的就是选中的那份权重与那套解码配置。 | i=1519, i=1520 |

### 训练序列(21 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 214 | smoke | 0.01h | returned | **smoke** | baseline:本 run 第一次训练启动,64 例 / 2 步,只验 train_sft.py 能不能跑通。取值域缺 smoke/baseline 两档,证据本身是充分的。 |
| 217 | smoke | 0.02h | returned | **smoke** | 对 i=214:batch-size 1→4、样本 64→128、步数 2→3,并加 --no-gradient-checkpointing。纯显存/吞吐探针(C8),不在验 C3 或 C4。 |
| 219 | smoke | 0.00h | returned | **smoke** | 对 i=217:batch-size 4→2 并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。显存探针;本次因残留 GPU 进程在 _move_model_to_device 处崩掉。 |
| 226 | smoke | 0.01h | returned | **smoke** | 与 i=219 逐字相同的重跑——清掉占卡的孤儿进程之后重试,不改任何变量。 |
| 229 | smoke | 0.02h | returned | **smoke** | 对 i=226:batch-size 2→3、样本 128→192。显存探针。 |
| 261 | smoke | 0.02h | returned | **smoke** | 对 i=229:首次加 --use-liger(融合损失核)并把 batch 拉到 8、样本 192→256。测的是装了 liger 之后显存墙有没有被移除,是 C8 而非 C4。 |
| 269 | smoke | 0.01h | returned | **smoke** | 对 i=261:batch-size 8→4、步数 3→4,仍是 liger 下的显存探针。 |
| 272 | smoke | 0.01h | returned | **smoke** | 对 i=269:batch-size 4→6、样本 256→320。显存探针的最后一档,定出 bs4/ga8 的正式配置。 |
| 277 | real | 0.03h | returned | **baseline** | baseline:第一次真实训练。data_curated 全量、2 epoch、bs4×ga8、lr 2e-5、8k、liger。没有可比的上一次真实训练,取值域缺 baseline 一档。本次在优化器状态分配时 OOM,未产出任何被接受的 checkpoint。 |
| 284 | real | 0.01h | returned | **unclear** | OOM 之后的 2 步显存探针(bs3×ga2、--max-steps 2),机械层按 kind=real 记录,但意图上是冒烟。不在验任何 C3/C4 变量。 |
| 287 | real | 1.03h | returned | **unclear** | 对 i=277:唯一差异是 batch-size 4→3、grad-accum 8→10(有效批量 32→30 基本不变),其余逐字相同。这是 OOM 逼出来的 C8 重启,不是在验 C4;按 reference §3 C8 的说明不应记成 C4,否则会污染缺口 2 的拆分。 |
| 736 | real | 0.18h | returned | **both** | 对 i=287 同时换了数据和方法:训练文件从 data_curated(10,829 例 / 50.6M token 长轨迹)换成 data_compact(7,622 例 / 10.0M token 紧凑纠正课程),同时 init 从 base 换成 sft_epoch1、max-length … |
| 869 | real | 0.27h | returned | **C4** | 本 run 唯一一次数据逐字不变的对照:训练文件仍是 data_compact/train.jsonl,只把 init 从 sft_epoch1 换成 Qwen/Qwen3-1.7B-Base、epochs 2→3、lr 1e-5→2e-5;agent 自己写明目的是不继承长轨迹模型的复读行为。 |
| 941 | smoke | 0.02h | returned | **smoke** | GRPO 冒烟(2 步、64 例)。因为这一版 transformers 改了 model-load dtype 参数名,策略被建成 FP32、FlashAttention 拒绝,第一次更新前就崩。 |
| 951 | smoke | 0.02h | returned | **smoke** | 与 i=941 逐字相同的重跑,差别只在脚本里加了显式 BF16 载入(i=945 的补丁)。C8 修复后的复跑。 |
| 966 | real | 0.37h | returned | **C4** | 对 i=869:同一权重起点(compact_base/final)、同一批数据来源(data_compact),换的是训练方法——从 SFT 换成 60 步 GRPO,并加了分级 format reward。数据文件名从 train.jsonl 变成 rl_train.jsonl 是 GRPO 只… |
| 1150 | real | 0.33h | returned | **both** | 对 i=869 同时换数据和方法:数据从 data_compact(含损坏的 official 解与错抽答案)换成重建的 data_clean(5,447 例、答案独立复核);方法上加 --no-packing、bs 6→8、ga 10→8、warmup 0.05、save-steps 86,并配合… |
| 1245 | real | 0.19h | returned | **both** | 对 i=966:数据换成 filter_clean_rl.py 产的 data_clean/rl_train.jsonl(2,474 条,与 clean SFT 重叠 0);方法上奖励函数改成惩罚空补全 + 要求严格数字末行,max-completion 4096→3072,步数 60→45,起点换… |
| 1339 | real | 0.18h | returned | **both** | 对 i=1150:数据换成 data_distill(2,139 条已验证长教师轨迹 / 7.6M token),同时 max-length 4096→6144、lr 2e-5→5e-6、epochs 3→2、起点换成 clean_grpo/checkpoint-15。max-length 是数据长… |
| 1382 | real | 0.18h | returned | **both** | 对 i=1339:数据从长教师轨迹换成 data_broad(21,110 例简短 ground-truth olympiad 解,中位 594 token),同时 max-length 6144→3072、lr 5e-6→3e-6、epochs 2→1。同一条数据/超参耦合边界。 |
| 1463 | real | 0.08h | returned | **C4** | 对 i=1245:训练文件逐字相同(data_clean/rl_train.jsonl)、max-completion 3072 相同、grad-accum 8 相同,只改学习率(降到 3e-7,约三分之一)、换 shuffle 种子 9901、步数 45→15,起点换成自己的 checkpoint… |

### 验证序列(25 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 23 | 3.0 | 5.0 | 否 |  | 未拿到。这次不是「追不到分数」,是评测根本没跑成:机器的 HTTP 代理拦住了 evaluate.py 对 localh… |
| 81 | 3.0 | 5.0 | 是 |  | 0.0(baseline,5 题 0 对)。分数从 baseline5.json 与 logs/*_aime2025_*… |
| 451 | 3.0 | 5.0 | 是 | c2, c4 | 0.0(0/5)。真正被读回的不是分数而是失败模式:模型给出 ANSWER 后继续复读到 8192 上限,评测抽不出答案… |
| 533 | 3.0 | 5.0 | 是 | c12 | 0.2(1/5)。逐样本读回 stop_reason 与 output_tokens:5 题里 3 题正常 stop、1… |
| 859 | 3.0 | 5.0 | 是 | c16, c17, c19 | 0.0(0/5)。读回的判据是「过度纠正」:3/5 在 350 token 内终止、1 条空回答、0 对(i=868)。 |
| 923 | 3.0 | 5.0 | 是 | c16, c17, c21 | 0.0(0/5)。读回 2 条空回答、2 条无最终答案、平均输出降到 1.9k token(i=940);据此判定该走 … |
| 1049 | 3.0 | 5.0 | 是 | c10, c20, c23, c25 | 0.0(0/5)。同时读回 checkpoint 载入正常与 <\|im_end\|> 停止行为(i=1052)。 |
| 1058 | 3.0 | 5.0 | 是 | c10, c20, c23, c25 | 0.0(0/5),但生成稳定性未退化(i=1064)。 |
| 1062 | 3.0 | 5.0 | 是 | c10, c20, c23, c25 | 0.0(0/5),且比前两个 checkpoint 更啰嗦(i=1070)。分数经 tail eval_ckpt45.o… |
| 1069 | 3.0 | 5.0 | 是 | c10, c20, c23, c25 | 0.0(0/5)。四个 RL checkpoint 全部 0/5,直接导致 i=1095 的数据配方转向。 |
| 1203 | 3.0 | 5.0 | 是 | c26, c27, c28, c29 | 0.0(0/5),但输出从 18–30k token 掉到 3.2k,同时出现 2 条空答案(i=1210)。 |
| 1211 | 3.0 | 5.0 | 是 | c26, c27, c28, c29 | 0.0(0/5),没修好 early-EOS 且更不稳,暂时否决(i=1219)。 |
| 1217 | 3.0 | 5.0 | 是 | c26, c27, c28, c29 | 0.0(0/5)。读回的关键判据是失败形态:有用推理常以 ANSWER: n<think> 结尾、两题立刻终止(i=12… |
| 1278 | 3.0 | 5.0 | 是 | c30, c31 | 0.2(1/5),clean 分支第一次拿到非零,且 3 题保持短答案(i=1284)。 |
| 1285 | 3.0 | 5.0 | 是 | c30, c31 | 0.2(1/5),输出只有 1.5k token、无失控生成(i=1292)。 |
| 1291 | 3.0 | 5.0 | 是 | c30, c31 | 0.0(0/5),回退,被否决(i=1300)。 |
| 1323 | — | — | 是 | c30, c31 | 0.033(1/30)。这是全 run 唯一一次官方口径全量评测,agent 用 tail eval_...full.o… |
| 1359 | 3.0 | 5.0 | 是 | c34 | 0.0(0/5),三条撞满上限的死循环(i=1365)。 |
| 1364 | 3.0 | 5.0 | 是 | c34 | 0.0(0/5),与 epoch1 同样的循环失败,整条长轨迹分支被否决(i=1370)。 |
| 1402 | 3.0 | 5.0 | 是 | c35 | 0.0(0/5),仍然简洁但没有迁移增益(i=1409)。 |
| 1407 | 3.0 | 5.0 | 是 | c35 | 0.0(0/5),广度分支两个 checkpoint 全部回退、被否决(i=1414)。 |
| 1420 | 3.0 | 5.0 | 是 | c36 | 0.2(1/5)。读回的除了分数还有:立即 EOS 从 2 条降到 1 条、答对的题换了一道(i=1422/1425)。… |
| 1454 | 3.0 | 5.0 | 是 | c40 | 0.0(0/5),五条全部撞满 8192 上限(i=1456/1459),消融被否决并回滚。 |
| 1477 | 3.0 | 5.0 | 是 | c42 | 0.0(0/5),不取代母 checkpoint(i=1482)。 |
| 1500 | 3.0 | 1.0 | 是 | c44, c41 | 0.0(1 题冒烟)。这一次读回的不是能力信号,而是产物可交付性:evaluate.py 能加载 final_model… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[284, 287]
- **1 次验证没有拿到信号**:i=[23]
- **分类学缺口提案 1 条**
  - external-knowledge-acquisition(i=746, i=663, i=850, i=858)
- **定义缺陷 4 条**
  - (i=938, i=936, i=936)
  - (i=284, i=281, i=287)
  - (i=291, i=1426, i=1322, i=1452)
  - (i=1043, i=1043)
- **边界情形 4 条**
  - data-forced-hyperparameter-change(i=1339, i=1338, i=1382, i=1385)
  - decontaminating-the-dev-set-scoring(i=1310, i=1313, i=1316)
  - greedy-arm-not-verifiably-greedy(i=1418, i=1415, i=1430, i=936)
  - n5-decisions(i=1049, i=1284, i=1300)

## codex_non_api_max_gpt-5.6-sol_10h_run1__aime2025_Qwen_Qwen3-4B-Base_17390210
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | aime2025 | Qwen_Qwen3-4B-Base | 6.44h | 0.1666666666… |

### 改动序列(41 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 66 | C8 | 在 evaluate.py 前加 env no_proxy/NO_PROXY=localhost,127.0.0.1,0.0.0.0,绕过容器 squid 代理;此前 inspect_ai 的客户端把本地 vLLM 端口也走了代理,整次评测以 RuntimeError 崩溃。 | i=66, i=63 |
| 69 | C11 | 自写 inspect_ai 日志分析脚本,把官方评分器的输出拆成每题 score / 输出字符数 / 是否出现 ANSWER 标记,用来把失败归因到格式而不是能力。 | i=69, i=70 |
| 137 | C3 | 新建 prepare_data.py:SFT 语料定为 Bespoke-Stratos-17k + s1K + NuminaMath-CoT,全部 pin 到 AIME2025 之前的快照,打包成 4k / 8k 两段课程。 | i=137, i=136 |
| 137 | C10 | 同一脚本内建污染守卫:对 AIME 2025 与 held-out AIME 2024 做归一化 + 模糊拒绝,命中即丢弃;实测拒掉 2 条近似 AIME2024 题、AIME2025 重叠 0。 | i=137, i=136, i=168 |
| 137 | C7 | 新建 evaluate_validation.py:把官方 AIME2025 harness 逐字复制成 AIME2024 held-out 评测器(同模板、同 scorer、同 vllm 参数),作为不消耗计分集的代理验证器。 | i=137, i=430 |
| 157 | C8 | 污染过滤从 SequenceMatcher 模糊比对改成 token 重叠过滤:第一次 prepare_data.py 因 O(n^2) 相似度计算卡住被 Ctrl-C,改写后 4 分钟跑完。 | i=157, i=155, i=156 |
| 223 | C1 | 改 merge_model.py:合并 LoRA 时把 model.config 与 tokenizer 的 EOS 对齐到 <\|im_end\|>(151645),并用 GenerationConfig(...) 整份重写 generation_config.json —— 该重写顺手丢掉了 b… | i=223, i=222, i=247, i=107 |
| 250 | C1 | 把 candidates/stage1/generation_config.json 改成贪婪解码,测不训练能不能修好不停机问题;改前逐字为 do_sample:true / temperature:0.6 / top_k:20 / top_p:0.95(i=247)。 | i=250, i=249, i=247 |
| 266 | C2 | 重写 prepare_data.py + merge_model.py:训练目标 token 从 <think> / <\|im_end\|> 换成 base 已训练的 <\|endoftext\|>(151643)。诊断是 LoRA 不更新绑定的词表头,导致这两个 token 的输出嵌入始终未训练… | i=266, i=265, i=265 |
| 314 | C3 | 新建 prepare_concise.py:课程从长 think 蒸馏轨迹换成「精简最终解」——Bespoke / Numina / s1K 的 polished 解答段,强制含显式答案且截到 2k token。 | i=314, i=313 |
| 390 | C3 | 新建 prepare_aime_special.py:直接任务专精集,取 AmirMohseni/AIME-1983-2024-Qwen3-8B 中 735 条答案已验证的历史 AIME 解答(排除 2024),repeat 5 加权。 | i=390, i=389, i=392 |
| 420 | C5 | 改交中途 checkpoint:merge runs/aime_special/checkpoint-100 而不是训练终点 adapter,理由是终点已把专精集背下来(loss<0.1)。 | i=420, i=419 |
| 433 | C5 | checkpoint 括弧扫描:把 checkpoint-50 / 150 也 merge 出来评,确认 100 是真最优而不是采样波动。结果 50→3/30、100→5/30、150→2/30。 | i=433, i=444 |
| 448 | C1 | 把 candidates/aime_special_100/generation_config.json 改成贪婪(禁 sampling),测能否降低 run 间方差;改前逐字为 do_sample:true / temperature:0.6 / top_k:20 / top_p:0.95(i=4… | i=448, i=447, i=446 |
| 453 | C1 | 回滚:把 candidates/aime_special_100 的解码配置改回采样(temperature 0.6 / top_k 20),因为贪婪掉到 2/30。 | i=453, i=452, i=475 |
| 468 | C3 | 新建 prepare_aime_long.py:400 条 8192-token 的长推理轨迹课程,想补回被 concise 课程砍掉的探索长度。 | i=468, i=470, i=475 |
| 493 | C4 | 新建 prepare_dpo.py + train_dpo.py:换训练方法为偏好优化(DPO,beta 0.1,lr 5e-7),配对取 think 正确 / no_think 错误的 256 对。 | i=493, i=493, i=495 |
| 526 | C3 | 新建 prepare_recent_aime.py:按年份切片的近期课程(--first-year 2010 --repeat 3),用当代竞赛风格取代 1983 起的全历史集。 | i=526, i=528, i=525 |
| 559 | C3 | 新建 prepare_aime24_official.py:把此前一直当 held-out 验证集用的 AIME 2024 官方题(HuggingFaceH4/aime_2024)也变成训练数据,repeat 10。此后 evaluate_validation.py 再没被调用过。 | i=559, i=560 |
| 577 | C1 | 把 leader(candidates/recent_aime_20)的 temperature 从 0.6 提到 0.8。 | i=577, i=576, i=578 |
| 580 | C1 | 回滚 temperature 到 0.6(0.8 掉到 2/30)。 | i=580, i=582 |
| 583 | C6 | 新建 interpolate_adapter.py 并做 LoRA 权重平均:aime_special/ckpt-100 与 recent_aime/ckpt-20 的 alpha=.5 与 .75 插值。 | i=583, i=585, i=582 |
| 624 | C5 | 给 train_sft.py 加 --stop-after 受控中断回调(TrainerCallback),把「训到 N 步就停」从手工 Ctrl-C 变成确定性行为——上一次 seed4040 是靠 KeyboardInterrupt 停的。 | i=624, i=623, i=629, i=617 |
| 640 | C3 | 近期课程再收窄:--first-year 2015 生成 data_recent_aime_2015(456 条),超参与 2010 版逐字相同。 | i=640 |
| 650 | C11 | 把候选间比较从标量分数升级为「解出题号集合」:自写脚本从 inspect_ai 日志里取每个候选答对的题号,用交集/差集判断两个专家是否互补,而不是比 0.2 vs 0.167。 | i=650, i=651, i=653 |
| 654 | C6 | 同血统 adapter 平均:recent_aime/ckpt-20 与 recent_aime_2015/ckpt-15 的 alpha=.5 插值,想同时保住两个专家各自解出的题。 | i=654, i=653 |
| 667 | C3 | 近期课程放宽:--first-year 2005 生成 data_recent_aime_2005(1000 条),超参与 2010 / 2015 版逐字相同。 | i=667 |
| 681 | C3 | 新建 generate_selftrain_aime24.py:用当前 leader 在 AIME2024 上 on-policy 拒绝采样(每题 8 条、留 4 条正确的、repeat 4),数据来源改为自生成+可验证过滤。 | i=681, i=686, i=680 |
| 699 | C1 | 给 leader 的 generation_config 加轻微 repetition_penalty(1.05),想压掉少数不收敛的循环输出。 | i=699, i=698, i=700 |
| 703 | C1 | 回滚 repetition_penalty(掉到 4/30)。 | i=703, i=702, i=715 |
| 710 | C4 | 新建 train_grpo.py:换 RL 方法,5 步 exact-answer GRPO,每 prompt 8 条 on-policy 采样,只给最终整数正确的 rollout 奖励,lr 1e-7。 | i=710, i=715 |
| 719 | C8 | 去掉 train_grpo.py 里 TRL 0.27.2 不接受的 steps_per_generation 参数,修复 GRPO 启动即崩。 | i=719, i=718, i=717 |
| 744 | C4 | 新建 train_full.py:改全参微调(不走 LoRA),10 步 @5e-7,专门测「LoRA 容量是不是瓶颈」。 | i=744, i=746 |
| 765 | C1 | leader 的 top_k 从 20 收窄到 10。 | i=765, i=764, i=766 |
| 782 | C1 | 把 top_k 改回 20 再复测一次,给两档做同轮次对照。 | i=782, i=781, i=783 |
| 790 | proposed:eval-protocol-f… | 把自测评测的调用口径对齐到官方计分默认:去掉一直在用的 --max-tokens 8192 / --max-connections 6 / --gpu-memory-utilization .85,改用 evaluate.py 的默认(--max-tokens 16000)。此前 6.5 小时里所有… | i=790, i=789, i=787 |
| 795 | C1 | 在 16k 默认口径下把 top_k 设回 10,与 c36 的 top_k 20 直接对照。 | i=795, i=794, i=796 |
| 799 | C1 | 再试 top_k 15。 | i=799, i=798, i=800 |
| 804 | C9 | 提交动作:把 runs/recent_aime/checkpoint-20 重新 merge 成 final_model —— 在 GRPO(6/30 后 3/30)、全参(4/30)、自训练(3/30)全部被拒之后,把手里最稳的候选写进提交目录。 | i=804, i=803 |
| 810 | C1 | final_model/generation_config.json 把 top_k 20 改成 10 —— merge_model.py 刚把它整份重写成 top_k 20(i=808 逐字),这次写入把锁定的解码配置补回去。 | i=810, i=808, i=815 |
| 812 | C9 | 提交完整性守卫:sha256 逐分片比对 final_model 与被选中的 candidates/recent_aime_20,再用本地 Transformers 加载校验 config / tokenizer / GenerationConfig。 | i=812, i=815 |

### 训练序列(20 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 169 | smoke | 0.01h | returned | **smoke** | baseline —— 本 run 第一次训练启动。纯吞吐/显存冒烟(5 步,bs2/ga1),验的是管线跑不跑得通,既不在验数据配方也不在验超参。现有四取值没有 smoke 位置,故记 unclear(见 boundary_case b1)。 |
| 180 | real | 1.10h | returned | **baseline** | baseline —— 首个真实训练。数据集与 i=169 冒烟相同(data/stage1_4096),只把 --max-steps 5 放大成 --epochs 1 并 bs 2→4 / grad-accum 1→4。没有前一次真实训练可对照,它是被后续一切对照参照的那个基线本身,而不是在验某一… |
| 275 | smoke | 0.01h | returned | **C3** | 数据集 data/stage1_4096 → data_v2/stage1_4096(prepare_data.py 已重写:训练目标 token 从 <think>/<\|im_end\|> 换成 base 的 <\|endoftext\|>)。虽然产物名叫 smoke_v2、只跑 10 步,但它… |
| 282 | real | 0.49h | returned | **C3** | 与 i=180 相比,超参逐字相同(--epochs 1 --batch-size 4 --grad-accum 4 --learning-rate 0.0002 --warmup-ratio 0.05 --save-steps 100 --logging-steps 10),唯一变量是数据集 da… |
| 320 | real | 0.11h | returned | **both** | 数据换成全新的 concise 课程(data_v2/stage1_4096 → data_concise/train_2048,长 think 轨迹 → 2k 内的精简最终解),同时超参也全换:bs 4→8、grad-accum 4→2、lr 2e-4→1e-4、--epochs 1 → --ma… |
| 334 | real | 0.31h | returned | **C4** | 数据集不变(data_concise/train_2048),bs/grad-accum 不变(8/2),只延长训练并退火:--max-steps 100→300、lr 1e-4→5e-5、warmup 0.05→0.03,并从 runs/concise_100 续训。 |
| 395 | real | 0.18h | returned | **both** | 数据换成历史 AIME 专精集(data_concise → data_aime_special/train_2048,735 条已验证解答 ×5),同时 lr 5e-5→2e-5、--max-steps 300 → --epochs 1;bs/ga 保持 8/2。数据与超参同时动,判不出哪一项在被… |
| 476 | real | 0.12h | returned | **both** | 数据换成 8192-token 的长轨迹集(data_aime_special/train_2048 → data_aime_long/train_8192),序列长度 4× 之后 batch-size 被迫 8→1、grad-accum 2→4,lr 2e-5→1e-5。超参改动里至少 bs/ga… |
| 513 | real | 0.06h | returned | **both** | 训练方法从 SFT 换成 DPO(train_dpo.py,beta .1,lr 5e-7,bs 1/ga 8),同时训练数据换成全新的 256 对偏好数据 data_dpo/train。方法与数据同时换。 |
| 531 | real | 0.05h | returned | **both** | 回到 SFT,数据换成 2010–2024 的近期 AIME 切片(data_recent_aime/train_2048)。相对同一父 adapter 的 i=395:bs/ga 相同(8/2),lr 2e-5→5e-6、save-steps 50→20 并首次显式给 --seed 2029。数据… |
| 563 | real | 0.02h | returned | **both** | 数据换成 AIME 2024 官方题 ×10(data_aime24_official/train_2048),父 adapter 换成新 leader recent_aime/ckpt-20,同时 bs 8→4、ga 2→4、lr 5e-6→2e-6、warmup .05→.1。 |
| 601 | real | 0.06h | returned | **C3** | 超参与 i=531 逐字相同(--batch-size 8 --grad-accum 2 --learning-rate 5e-6 --warmup-ratio .05),唯一在验的是数据:把 leader 再放回广义 concise 奥数语料跑 50 步,看能不能在不改输出风格的前提下捡回一般推理… |
| 615 | real | 0.03h | returned | **C3** | 与 i=531 逐字相同的命令,唯一差别是 --seed 2029 → 4040(agent 自己称之为 different deterministic data order)。数据顺序单变量对照。结局:22/47 步时被 KeyboardInterrupt(i=617),checkpoint-20… |
| 629 | real | 0.03h | returned | **C3** | 与 i=615 相比只换 --seed 4040 → 5050,并首次用新加的 --stop-after 20 代替手工 Ctrl-C。同一配方的第三个数据顺序。 |
| 643 | real | 0.02h | returned | **C3** | 超参与 i=531/615/629 相同(bs 8 / ga 2 / lr 5e-6 / warmup .05 / seed 2029),只把数据的年份切片从 2010 起收窄到 2015 起(739 → 456 条),save/stop 步数随之 20→15。 |
| 670 | real | 0.03h | returned | **C3** | 同上配方,年份切片反向放宽到 2005 起(1000 条),超参逐字不变(bs 8 / ga 2 / lr 5e-6 / warmup .05 / seed 2029 / stop-after 20)。 |
| 690 | real | 0.01h | returned | **both** | 数据来源换成自生成+可验证过滤(leader 在 AIME2024 上的 on-policy 拒绝采样,64 条),同时 bs 8→4、ga 2→4、lr 5e-6→1e-6、warmup .05→.1、save-steps 2。数据来源类别与超参同时换。 |
| 716 | real | 0.01h | returned | **unclear** | GRPO 首次启动,加载完权重后立刻 ValueError 崩溃(generation_batch_size 与 steps_per_generation 互斥),没产出任何权重,因此没有受测变量可言。 |
| 720 | real | 0.15h | returned | **C4** | 与 i=690 相比,训练数据来源基本同源(都是 leader 在 AIME2024 题上的 on-policy 采样),换的是方法:监督地拟合过滤后的 rollout → GRPO 在线策略优化 + 二值 exact-answer 奖励,lr 1e-6→1e-7。注:机械层把这行记成 smoke,… |
| 747 | real | 0.02h | returned | **C4** | 与 i=531 相比,数据集逐字相同(data_recent_aime/train_2048)、seed 相同(2029),换的只有方法:LoRA → 全参微调(train_full.py),lr 5e-6→5e-7、10 步。agent 明说是在测 LoRA 容量是不是瓶颈。本 run 最干净的 … |

### 验证序列(36 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 33 | 3.0 | 6.0 | 否 |  | 未拿到分数 —— vLLM 服务器启动失败(RuntimeError: Failed to start vLLM ser… |
| 66 | 3.0 | 6.0 | 是 | c1 | 0.0(6 题 0/6);同时确认 c1 的代理旁路修复生效 |
| 359 | 3.0 | 6.0 | 是 | c10 | 0.333(6 题 2/6) |
| 363 | — | — | 是 | c10 | 0.0667(30 题 2/30) |
| 415 | — | — | 是 | c11 | 0.1(3/30) |
| 423 | — | — | 是 | c11, c12 | 0.1667(5/30) |
| 439 | — | — | 是 | c12, c13 | 一次 tool call 跑了两个候选:checkpoint-50 = 0.1(3/30)、checkpoint-150… |
| 450 | — | — | 是 | c14 | 0.0667(2/30),贪婪解码 |
| 466 | — | — | 是 | c15 | 0.1667(5/30),恢复采样后与 i=423 逐位一致 |
| 503 | — | — | 否 | c16 | 0.0(0/30,aime_long checkpoint-50);循环里的第二个候选 checkpoint-100 从… |
| 520 | — | — | 是 | c17 | 0.0333(1/30) |
| 539 | — | — | 是 | c18 | 两个候选:checkpoint-20 = 0.2(6/30)、checkpoint-40 = 0.1667(5/30)(… |
| 568 | — | — | 是 | c19 | 0.1333(4/30) |
| 578 | — | — | 是 | c20 | 0.0667(2/30),temperature 0.8 |
| 589 | — | — | 是 | c22 | 0.1667(5/30),alpha=.5 插值 |
| 596 | — | — | 是 | c22 | 0.1333(4/30),alpha=.75 插值 |
| 607 | — | — | 否 | c10 | 0.0(0/30,rebalance checkpoint-25);第二个候选 checkpoint-50 从未评测,循… |
| 621 | — | — | 是 | c18 | 0.1(3/30),seed 4040 |
| 635 | — | — | 是 | c18 | 0.1(3/30),seed 5050;agent 在 i=638 逐字 cat 到 {"accuracy": 0.1} |
| 647 | — | — | 是 | c24 | 0.2(6/30),2015 起切片 |
| 658 | — | — | 是 | c26 | 0.1667(5/30),recent_mix 平均 |
| 661 | — | — | 是 | c18, c24 | 复测两个候选:recent_aime_20 = 0.2(6/30)、recent_aime_2015_15 = 0.13… |
| 675 | — | — | 是 | c27 | 0.1(3/30),2005 起切片 |
| 695 | — | — | 是 | c28 | 0.1(3/30),自训练后反而下降 |
| 700 | — | — | 是 | c29 | 0.1333(4/30),repetition_penalty 1.05 |
| 729 | — | — | 是 | c31 | 两个候选:grpo checkpoint-2 = 0.1667(5/30)、checkpoint-4 = 0.2(6/3… |
| 737 | — | — | 是 | c31 | 0.1(3/30);同一份 grpo_4 权重两次全量评测 6/30 → 3/30,agent 据此判定 GRPO 不稳… |
| 755 | — | — | 是 | c33 | 0.1333(4/30),全参微调 |
| 766 | — | — | 是 | c34 | 0.2(6/30),top_k=10;并在 i=770 取到答对题号集合 [1, 8, 9, 13, 26, 29] |
| 772 | — | — | 是 | c34 | 0.2333(7/30),top_k=10 复测 |
| 776 | — | — | 是 | c34 | 0.0667(2/30),top_k=10 第三次;agent 在 i=780 逐字 cat 到 accuracy 0.… |
| 783 | — | — | 是 | c35 | 0.0667(2/30),top_k=20 第三次 |
| 790 | — | — | 是 | c35, c36 | 0.1667(5/30),16k 默认口径 + top_k 20;i=793 逐字 cat |
| 796 | — | — | 是 | c36, c37 | 0.1667(5/30),16k 默认口径 + top_k 10 |
| 800 | — | — | 是 | c36, c38 | 0.1333(4/30),16k 默认口径 + top_k 15 |
| 816 | — | — | 是 | c39, c40, c41 | 0.2667(8/30),final_model 端到端;i=820 逐字 8/30 (26.7%),i=827 另给 … |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[716]
- **3 次验证没有拿到信号**:i=[33, 503, 607]
- **分类学缺口提案 1 条**
  - eval-protocol-fidelity(i=789, i=787, i=790, i=33, i=783, i=820)
- **定义缺陷 3 条**
  - (i=720, i=725, i=727, i=747, i=755)
  - (i=729, i=736, i=511, i=609)
  - (i=229, i=430, i=240, i=410)
- **边界情形 3 条**
  - 冒烟行(i=169)验的是管线跑不跑得通,是第二档验证器的定义域,既不是 C3 也不是 C4;首个真实训练(i=180)是后续一切对照的参照物本身,没有「上一次」可比。两者都只能落 unclear,于是「schema 装不下」和「证据不足」被算进同一个数字里 —— 本 run 3/20 的 unclear 里有 2 条是这一类,只有 i=716(启动即崩)是真正的证据不足。与 reference …(i=168, i=169, i=180)
  - i=282 是本 run 唯一一次严格单变量训练:超参与 i=180 逐字相同,只换数据集。但两个数据集的语料来源、pin 住的 revision、混合比例全都一样,差别只在训练目标 token(<think> / <\|im_end\|> → base 已训练的 <\|endoftext\|>),按 §3 这是 C2 而不是 C3。tested_variable 的四取值没有 C2,只能记成 C…(i=265, i=265, i=282, i=180)
  - i=476 把数据从 2048-token 的集换成 8192-token 的集,序列长度 4× 之后 batch-size 只能从 8 降到 1、grad-accum 从 2 升到 4,否则显存放不下。这个超参改动不是一次独立的方法选择,而是数据改动的算术后果。按现定义既可记 C3(唯一的自主选择是数据)也可记 both(命令行上 bs/ga/lr 确实都变了),没有规则可依。这正是 refer…(i=476, i=395, i=470)

## codex_non_api_max_gpt-5.6-sol_10h_run1__aime2025_google_gemma-3-4b-pt_17390208
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | aime2025 | google_gemma-3-4b-pt | 8.66h | 0.0 |

### 改动序列(51 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 91 | C8 | 绕过 Squid 代理:用 no_proxy / VLLM_BASE_URL / VLLM_API_KEY 环境变量重跑 evaluate.py。首次基线评测(i=17)之所以卡死,是 inspect_ai 的 wait_for_server 对 localhost:35182 的 readines… | i=83, i=90, i=91 |
| 164 | C3 | 新建 prepare_data.py,构造第一版 SFT 混料 35,352 条:NuminaMath-CoT(amc_aime/aops_forum/math/olympiads/synthetic_amc)32,118 + AIME 1983–2023 三教师 polished 解 2,779 … | i=164, i=167, i=175 |
| 164 | C10 | 同一次改动里写 DATA_PROVENANCE.md 并把去污染写进构造器:只取问题集截止于 2025 年之前的来源,脚本从不打开 math-ai/aime25 或 opencompass/AIME2025。 | i=644, i=830 |
| 173 | C4 | 改 train_sft.py 的 LoRA target-module 正则:gemma-3-4b 是 Gemma3ForConditionalGeneration,含 vision_tower;把适配器限制在 language decoder 上,视觉塔不训。 | i=171, i=175 |
| 186 | C8 | 把 microbatch 2→4、grad-accum 8→4(有效 batch 仍为 16),纯为吞吐;i=176 的启动被 KeyboardInterrupt 于 13/2210 步作废。除这两项外命令逐字相同。 | i=183, i=186, i=184 |
| 196 | C1 | 改 train_sft.py 的导出段,把提交模型的 generation_config 写死:do_sample=True、eos_token_id=[tokenizer.eos_token_id, <end_of_turn>]、temperature=0.6、top_p=0.95、top_k=6… | i=195, i=688, i=158 |
| 199 | C3 | 改 prepare_data.py,增加第二阶段语料输出 data/aime_stage2.jsonl(7,612 条,偏重历史 AIME 教师轨迹与 Numina 的 pre-2025 AMC/AIME 子集)。 | i=208, i=211 |
| 321 | C1 | 字段级:candidate_lora1/generation_config.json 的 eos_token_id 从 [1, 1, 106] 去重成 [1, 106]。其余字段(bos 2 / cache_implementation hybrid / do_sample true / pad 0… | i=319, i=320, i=412 |
| 332 | C11 | 自写脚本,把 inspect_ai 日志的官方评分器输出转成可决策信号:逐题解析 ^ANSWER: 行、输出 token 数、stop_reason。用来把「能力失败」和「格式/解码失败」分开——这一档判定是确定性的、零 GPU,而同期的分数被 0/10 完全吞掉。 | i=332, i=333 |
| 339 | C1 | 字段级:candidate_lora1/generation_config.json 的 do_sample true→false,做贪婪对照。temperature 0.6 / top_p 0.95 / top_k 64 三个采样字段**保留未删**——i=412 的读取逐字证实,且 vLLM 在… | i=336, i=341, i=412 |
| 345 | C3 | 改 prepare_data.py + train_sft.py:第二阶段语料纳入 DeepSeek-R1 / Grok-3-mini-high / QwQ-32B 在 AIME 1983–2023 上的全部正确长轨迹(各上采样一次)并改用保尾截断,7,612 → 11,958 条。 | i=343, i=348 |
| 361 | C8 | 把 microbatch 1→2、grad-accum 16→8(有效 batch 仍为 16),纯为吞吐;i=351 的启动被 KeyboardInterrupt 于 8/748 步作废。agent 自己写明是补偿式改动。 | i=358, i=359 |
| 414 | C8 | 字段级:把 candidate_lora1/generation_config.json 的 do_sample 改回 true(c9 的回滚),解掉 Trainer 在 400/748 步存档时抛的 GenerationConfig 校验失败。父目录的 config 被子训练继承,是这条 run … | i=407, i=410 |
| 498 | C8 | OpenMathInstruct-2 的全量 23 分片扫描挂死,agent kill 掉进程(i=494)并改成随机抽 300 个 row-group 做统计,把一次会吃掉数十分钟的分析压到 6 秒。 | i=494, i=498 |
| 502 | C3 | 新建 prepare_omi.py,数据来源整体换成 nvidia/OpenMathInstruct-2(405B 教师生成、答案过滤、pre-2025),按 row-group 随机抽 90,000 条 math / augmented_math。 | i=482, i=505 |
| 508 | C3 | 改 prepare_omi.py,在 OMI 90k 之上混入 8,049 条历史 AIME/AMC replay 以保域行为,合计 98,049 条(data/omi_math_mix.jsonl)。 | i=511 |
| 703 | C3 | 新建 prepare_sharpen.py:只留简洁的、已验证的 pre-2025 AIME/AMC 解,6,702 条(aime_polished 2,779 + numina:amc_aime 3,923),目标是缩短推理轨迹。 | i=706 |
| 742 | C11 | 升级同一套工装:逐题 target / pred(\boxed 抽取)/ token 数 / stop_reason,外加两个聚合的确定性判据——撞 8192 上限的题数(at8192)与无 \boxed 的题数(no_box)。截断率此后成为它真正在用的排序信号(47%→0% 那类),分数则一直是… | i=742, i=743 |
| 769 | C1 | 字段级:candidate_sharp4/generation_config.json 新增 repetition_penalty: 1.05,其余采样字段(temperature 0.6 / top_p 0.95 / top_k 64)不动。效应是确定性的:全量 30 题总输出 token 从 7… | i=768, i=774 |
| 778 | C1 | 字段级:candidate_sharp4 转贪婪,写法是 temperature 0.6→0.0 且 **保留 do_sample: true**(i=887 的读取逐字确认),而不是 c9 那种 do_sample:false —— agent 吸取了 i=407 的存档崩溃教训。这是本 run … | i=774, i=887, i=887 |
| 787 | C1 | 字段级:candidate_sharp4 temperature 0.0→0.2(rep 1.05 保留)。同样 1/30,但答对的是另一题(22 号 vs 14 号)。 | i=785, i=796 |
| 797 | C3 | 改 prepare_sharpen.py 加 --aime-only:语料再收窄到 2,779 条 polished 历史 AIME(去掉 numina:amc_aime 那 3,923 条)。 | i=796, i=805 |
| 808 | C1 | 把 candidate_aime5 的 generation_config 设成当前最佳解码档(temperature 0.0 + repetition_penalty 1.05),以便与 sharp4 同口径比。 | i=807 |
| 813 | C1 | 把 candidate_omi3 也改成同一贪婪 + rep 1.05 解码档,补一次同口径复评(它此前的全量评测用的是采样解码,不可比)。 | i=812 |
| 817 | C1 | candidate_full2 改成同一贪婪 + rep 1.05 解码档,同口径复评。 | i=812 |
| 822 | C1 | candidate_lora1 改成同一贪婪 + rep 1.05 解码档,同口径复评。 | i=821 |
| 838 | C1 | repetition_penalty 扫描第 1 点:1.05 → 1.00(等价于关掉惩罚),权重与其余字段不动。 | i=839, i=853 |
| 842 | C1 | repetition_penalty 扫描第 2 点:→ 1.10。 | i=843 |
| 846 | C1 | repetition_penalty 扫描第 3 点:→ 1.03。 | i=847 |
| 850 | C1 | repetition_penalty 扫描第 4 点:→ 1.04。 | i=851 |
| 854 | C1 | repetition_penalty 扫描第 5 点:→ 1.06。 | i=855 |
| 859 | C1 | 回到 rep 1.05,把 temperature 0.0→0.1,测贪婪邻域。 | i=860, i=863 |
| 870 | C1 | 把 candidate_high6 设成同一贪婪 + rep 1.05 解码档,再评。 | i=869 |
| 876 | C1 | 把 candidate_sharp4 的 generation_config 恢复成贪婪 / rep 1.05(扫描结束后的定稿值),准备冻结。 | i=875 |
| 877 | C9 | 提交守卫 + 提交:先断言 final_model 不存在(存在即 exit 2,防覆盖),再 cp -a candidate_sharp4 final_model。不改任何权重,只决定交哪一个候选。 | i=875, i=877 |
| 895 | C1 | 改到 final_model 上重扫 temperature:0.0→0.2。触发原因是 i=890 发现同一份权重、同一份 config,只把 --max-connections 4 + --max-tokens 8192 换成 evaluate.py 的默认值,分数就从 1/30 掉到 0/30… | i=894, i=896 |
| 899 | C1 | final_model temperature → 0.4。 | i=900 |
| 904 | C1 | final_model temperature → 0.6。 | i=905, i=906 |
| 909 | C1 | final_model temperature → 0.8。 | i=910 |
| 913 | C1 | final_model temperature → 1.0(默认口径粗扫的最后一点,0.0–1.0 全零)。 | i=914, i=919 |
| 925 | C1 | 把 candidate_alt7 设成同一贪婪解码档,在 evaluate.py 默认口径下评。 | i=924, i=926 |
| 931 | C1 | final_model temperature → 0.3(默认口径细扫)。这是默认口径下第一个非零点:1/30。 | i=930, i=932 |
| 938 | C1 | final_model temperature → 0.25(0.3 的相邻点)。 | i=935, i=939 |
| 941 | C1 | final_model temperature → 0.35(0.3 的另一相邻点)。 | i=942 |
| 946 | C1 | final_model temperature 回到 0.3,原样重跑一次做可复现性检验。 | i=945 |
| 952 | C1 | 字段级:temperature 保持 0.3,把 top_p 0.95→0.90、top_k 64→50(i=964 的读取逐字确认两项一起改了,不是只改 top_p)。 | i=951, i=964 |
| 967 | C1 | 回退 top_p→0.95、top_k→64(temperature 0.3、rep 1.05 保留),同时把安全数据谱系写进 final_model/training_summary.json。i=969 的读取给出定稿字段集。 | i=966, i=969 |
| 968 | proposed:submission_inte… | 提交前对 final_model 做结构与元数据断言:883 个张量与 index 逐一对齐、三个分片非空、断言 generation_config 的 temperature==0.3 且 repetition_penalty==1.05,通过才打印 FINAL_MODEL_VALID。不改任何产… | i=968, i=969 |
| 973 | C1 | final_model top_p 0.95→0.80(temperature 0.3 不动)。 | i=972, i=974 |
| 985 | C1 | 最终恢复 top_p 0.95 / top_k 64,提交配置定稿为 temperature 0.3 / top_p 0.95 / top_k 64 / repetition_penalty 1.05 / eos [1,106](i=987 逐字确认)。 | i=984, i=987 |
| 986 | proposed:submission_inte… | 同类守卫的第二次执行,断言集合更全:六个必需文件存在且非空、四元解码组合等于选定值、training_summary.json 的 contamination_policy 里含 'no AIME 2025'、每个分片可 safe_open 且键非空。把 C10 的去污染声明也纳入了机械断言。 | i=986, i=987 |

### 训练序列(11 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 176 | real | 0.04h | returned | **baseline** | baseline —— 本 run 第一次训练:LoRA r64、lr 1.2e-4、bs 2 × accum 8、max-len 4096、1 epoch,数据 data/math_sft.jsonl(35,352 条)。tested_variable 填 unclear 是**取值域装不下**,… |
| 186 | real | 1.40h | returned | **unclear** | 与 i=176 除 batch-size 2→4、grad-accum 8→4 外命令逐字相同,有效 batch 仍为 16;唯一动机是吞吐(c5)。本次不测任何变量,只是重启——schema 里没有对应取值,故 unclear,同样属于取值域装不下。跑满 1.40h,产出 candidate_lo… |
| 351 | real | 0.04h | returned | **both** | 同时换数据和方法:数据从 math_sft.jsonl(35,352)换成 i=345 新造的 aime_stage2_v2.jsonl(11,958,含三教师全部正确长轨迹);方法从 LoRA r64 换成全参,lr 1.2e-4→5e-6,基座换成上一阶段产物 candidate_lora1,b… |
| 361 | real | 0.50h | returned | **both** | 与 i=351 除 batch-size 1→2、grad-accum 16→8 外逐字相同(有效 batch 仍 16,agent 明说是补偿式改动)。相对上一次**完成**的训练(i=186)仍是数据与方法同时变,故 both。结局:跑到 400/748 步存档时因父目录 generation_… |
| 416 | real | 0.93h | returned | **both** | 与 i=361 数据、种子、超参逐字相同的重启(agent 自述并用 step 10 的 loss ~0.778 核对过轨迹一致),唯一变化在 run 之外——i=414 修好了父目录的 generation_config。相对上一次完成的训练仍是 both。跑满 0.93h(train_runti… |
| 514 | real | 2.50h | returned | **both** | 数据源整体更换(aime_stage2_v2 11,958 → omi_math_mix 98,049,即 OpenMathInstruct-2 的 90k 已验证 MATH + 8,049 AIME replay),同时 lr 5e-6→3e-6、max-len 4096→2048、有效 batc… |
| 684 | — | — | — | **unclear** | **这一行不是训练启动**。命令是 `python train_sft.py --help \| sed -n '1,220p'`,返回的是 argparse 用法文本,没有加载模型、没有产物、没有优化步。机械层把它记成 kind=real 的启动(见 definition_defect d1)。 |
| 747 | real | 0.40h | returned | **both** | 数据从 omi_math_mix(98,049)换成 aime_concise.jsonl(6,702,只留简洁的 pre-2025 AIME/AMC 解),同时 epochs 1→2;lr 3e-6、bs 4 × accum 8、max-len 2048 与 i=514 逐字相同。基座 candi… |
| 798 | real | 0.26h | returned | **both** | 数据再收窄(aime_concise 6,702 → aime_polished 2,779,只留 polished 历史 AIME,去掉 numina:amc_aime),同时 lr 3e-6→2e-6、epochs 2→3;基座换成 candidate_sharp4。真实产物是 candidat… |
| 864 | real | 0.21h | returned | **C4** | **数据不变**:与 i=747 用同一份 data/aime_concise.jsonl(6,702),bs 4 × accum 8、max-len 2048 也逐字相同;只改学习率 3e-6→8e-6、epochs 2→1,基座取上一阶段产物 candidate_sharp4。agent 自述目… |
| 920 | real | 0.21h | returned | **C4** | 本 run 最接近单变量的一次:与领先者 candidate_sharp4(i=747)同基座(candidate_omi3)、同数据(data/aime_concise.jsonl 6,702)、同 bs 4 × accum 8、同 max-len 2048,只把「2 轮 3e-6」换成「1 轮 … |

### 验证序列(35 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 17 | 3.0 | 4.0 | 否 |  | 未拿到 —— 但不是「追不到」:进程真的没产出分数。evaluate.py 起了 vLLM 并加载完模型(curl --… |
| 91 | 3.0 | 4.0 | 是 | c1 | 0.0(baseline4.json,limit 4)。同时判定 c1:分数拿回来这件事本身就是代理绕过成功的证据。ag… |
| 324 | 3.0 | 10.0 | 是 | c2, c3, c4, c6, c8 | 0.0(limit 10)。第一次评 candidate_lora1,一次同时裁决数据混料、去污染约束、LoRA 挂载范… |
| 340 | 3.0 | 10.0 | 是 | c9 | 0.0(limit 10,贪婪)。同一份权重、同一 --limit 10,只翻 do_sample —— 结论「采样不是… |
| 477 | 3.0 | 10.0 | 是 | c10 | 0.0(limit 10)。agent 另外读出:平均输出长度从 ~590 涨到 ~2,084 token,10 题里 … |
| 731 | — | — | 是 | c13, c14 | 0.0(全量 30 题)。第一次全量;agent 用 c48 的工装读出 8/30 撞 token 上限。 |
| 770 | — | — | 是 | c15, c16 | 0.0(全量 30)。分数没动,但确定性副产物很明确:总输出 token 74k→18.6k,循环归零。 |
| 779 | — | — | 是 | c17 | 0.033(1/30,第 14 题)。全 run 第一个非零分。相对 i=770 只翻了 temperature 0.6… |
| 788 | — | — | 是 | c18 | 0.033(1/30,第 22 题)。与 i=779 同分但答对的是另一题——agent 由此判定这是低概率通过,不是稳… |
| 809 | — | — | 是 | c19, c20 | 0.0(全量 30)。candidate_aime5 被明确拒收,保留 sharp4 为领先者。 |
| 814 | — | — | 是 | c21 | 0.0(全量 30)。candidate_omi3 在同一贪婪 + rep 1.05 口径下复评。 |
| 818 | — | — | 是 | c22 | 0.0(全量 30)。candidate_full2 同口径复评。 |
| 823 | — | — | 是 | c23 | 0.0(全量 30)。candidate_lora1 同口径复评。四个历史候选在同一解码档下全零,只有 sharp4 非… |
| 839 | — | — | 是 | c24 | 0.0(rep 1.00) |
| 843 | — | — | 是 | c25 | 0.0(rep 1.10) |
| 847 | — | — | 是 | c26 | 0.0(rep 1.03) |
| 851 | — | — | 是 | c27 | 0.0(rep 1.04) |
| 855 | — | — | 是 | c28 | 0.0(rep 1.06)。至此 1.00/1.03/1.04/1.06/1.10 全零、1.05 为 1/30,age… |
| 860 | — | — | 是 | c29 | 0.0(temperature 0.1 + rep 1.05) |
| 871 | — | — | 是 | c30 | 0.0(全量 30)。candidate_high6 被拒,i=920 的 C4 对照因此成立(两个 lr 变体都归零)… |
| 890 | — | — | 是 | c31, c32 | 0.0(final_model 首评)。**同一份权重、同一份 generation_config**,只把 --max… |
| 896 | — | — | 是 | c33 | 0.0(默认口径,temperature 0.2) |
| 900 | — | — | 是 | c34 | 0.0(默认口径,temperature 0.4) |
| 905 | — | — | 是 | c35 | 0.0(默认口径,temperature 0.6) |
| 910 | — | — | 是 | c36 | 0.0(默认口径,temperature 0.8) |
| 914 | — | — | 是 | c37 | 0.0(默认口径,temperature 1.0)。0.0–1.0 粗扫全零。 |
| 926 | — | — | 是 | c38 | 0.0(默认口径)。candidate_alt7 被拒;这同时是 i=920 那次 C4 对照的判定端,但口径与 sha… |
| 932 | — | — | 是 | c39 | 0.033(1/30,第 29 题)。默认口径下第一个非零点,agent 当即把 temperature 0.3 定为提… |
| 939 | — | — | 是 | c40 | 0.0(temperature 0.25) |
| 942 | — | — | 是 | c41 | 0.0(temperature 0.35)。agent 读成「0.30 是尖峰」。 |
| 947 | — | — | 是 | c42 | 0.0 —— 与 i=932 **逐字同配置、同权重、同口径**的重复运行,1/30 → 0/30。agent 自己写下… |
| 953 | — | — | 是 | c43 | 0.033(1/30,第 14 题;top_p 0.90 + top_k 50) |
| 960 | — | — | 是 |  | 0.0 —— i=953 的逐字重复,无任何新改动,1/30 → 0/30。第二对同配置翻转。 |
| 974 | — | — | 是 | c45 | 0.033(1/30,第 13 题;top_p 0.80) |
| 981 | — | — | 是 |  | 0.0 —— i=974 的逐字重复,无任何新改动,1/30 → 0/30。第三对同配置翻转。三对全部不可复现,agen… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[186, 684]
- **1 次验证没有拿到信号**:i=[17]
- **分类学缺口提案 1 条**
  - submission_integrity_guard(i=968, i=969, i=986, i=987)
- **定义缺陷 5 条**
  - 骨架「训练」表把 i=684 记成 kind=real 的训练启动(0.00h、产物『—』)。它其实是 `python train_sft.py --help \| sed -n '1,220p'`,返回 argparse 用法文本,没有加载模型、没有优化步、没有产物。抽取器只匹配了命令里的 `train_sft.py`。后果:这条 run 的『训练启动次数』被记成 11 次而非 10 次;在 §…(i=684, i=685)
  - 骨架把 i=798 的产物记成 `data/aime_polished.jsonl`。原命令是 `python prepare_sharpen.py --aime-only --output data/aime_polished.jsonl && python train_sft.py --base-model candidate_sharp4 --data data/aime_polished.…(i=798, i=809)
  - C9 的名字是『提交守卫』,定义却写成『不改任何产物,只决定此刻把哪个已有候选写进 final_model』——只覆盖了『选择』这一半。本 run 有两次纯粹的提交守卫既不改产物也不选候选:i=968 与 i=986 对 final_model 做结构与元数据断言(883 张量与 index 对齐、分片非空、解码四元组等于选定值、污染声明字符串存在),断言失败即阻断提交。按现定义它们在正向分类学里…(i=986, i=987)
  - §2 第二档把冒烟定义为『产物名为 smoke/sanity/bench 或 --max-samples ≤ 5000』。按这条判据,本 run 冒烟次数 = 0(骨架训练表 11 行全是 kind=real)。但 agent 确实做了两种冒烟:(a) i=711 用 `CUDA_VISIBLE_DEVICES=''` 在 CPU 上单跑训练脚本的 tokenize 阶段,验证新数据能被 toke…(i=711, i=712)
  - 第四档把噪声写成档位属性『±1–3pp,采样解码下更大』。在 aime2025 上这个区间不可表达:全量只有 30 题,单题即 3.33pp,不存在 1–3pp 这个刻度。本 run 提供了三对**逐字同权重、同 generation_config、同评测命令**的全量重复:i=932↔i=947、i=953↔i=960、i=974↔i=981,三对全部是 1/30 → 0/30,即 3/3 不可…(i=947, i=951, i=966, i=984)
- **边界情形 4 条**
  - 『为保持有效 batch 不变而做的 microbatch / grad-accum 补偿』落在 C4 与 C8 之间。i=186(bs 2→4、accum 8→4)与 i=361(bs 1→2、accum 16→8)都改了 §3 明确列为 C4 的超参(batch),但两次的有效 batch 都锁死在 16,唯一动机是吞吐,agent 在 i=358 逐字写明。按 C4 判会污染缺口 2 的拆分…(i=183, i=358)
  - 『数据规模一变就机械地逼着 epochs / lr 跟着变』——spec §9 说多名标注者独立指出的那个边界,在本 run 上是 i=747 与 i=798。i=747 把数据从 98,049 收窄到 6,702(15 倍)的同时 epochs 1→2;i=798 再收窄到 2,779 的同时 epochs 2→3 且 lr 3e-6→2e-6。两次的 epochs 增加都可以读成『补偿数据量缩…(i=746, i=796, i=751)
  - tested_variable 的取值域装不下三种情况,全被迫填 unclear:首次训练(i=176,应为 baseline)、以及两次纯吞吐重启中被作废的那一半(i=186 是 i=176 的重启;i=684 根本不是训练)。本 run 3/11 的 unclear **没有一条是证据不足** —— 三条的证据都很充分,只是 schema 没有对应取值。这与 spec §9 第 4 条的观察一…(i=175, i=183, i=685)
  - i=920 是本 run 唯一一次接近严格单变量的 C4 对照(同基座 candidate_omi3、同数据 aime_concise.jsonl、同 bs/accum/max-len,只把『2 轮 3e-6』换成『1 轮 6e-6』),但它在**验证器一端**不受控:领先臂 candidate_sharp4 拿到 1/30 那次用的是 `--max-connections 4 --max-tok…(i=921, i=894)

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
| 38 | 3.0 | 5.0 | 是 |  | 0.0 |
| 346 | — | — | 是 | c1, c2, c7, c8 | 0.0 |
| 426 | — | — | 否 |  | 未拿到：该事件不是评测，是 `ps -eo pid,stat,cmd \| rg 'evaluate.py --help… |
| 703 | — | — | 是 | c14 | 0.067 |
| 964 | — | — | 是 | c18 | 0.0 |
| 969 | — | — | 是 | c18 | 0.0 |
| 969 | — | — | 是 | c18 | 0.0 |
| 969 | — | — | 是 | c18 | 0.0 |
| 969 | — | — | 是 | c18 | 0.067 |
| 969 | — | — | 是 | c18 | 0.067 |
| 969 | — | — | 是 | c18 | 0.067 |
| 969 | — | — | 是 | c16 | 0.067 |
| 969 | — | — | 是 | c16 | 0.067 |
| 969 | — | — | 是 | c16 | 0.067 |
| 978 | — | — | 是 | c21 | 0.033；该事件是一个 for 循环，实际跑了4 个温度：T=0.2 -> 1/30、T=0.4 -> 0/30、T=… |
| 987 | — | — | 是 | c21 | 0.033；for 循环，T=0.50 -> 1/30、0.55 -> 1/30、0.65 -> 0/30、0.70 -… |
| 996 | — | — | 是 | c22 | 0.033；for 循环，top_k 5/10/40/50 与 top_p 0.80/1.00 共 6 次，读到 1/3… |
| 1012 | — | — | 是 | c23 | 0.0；for 循环，repetition_penalty 1.03/1.05/1.10/1.15 全部 0/30（机械… |
| 1031 | — | — | 是 | c19 | 0.033；for 循环，4 个细插值点读到 1/30、1/30、0/30、2/30，均未超过 2/30（机械层取回 0… |
| 1124 | — | — | 是 | c17 | 0.0 |
| 1158 | — | — | 是 | c24 | 0.0 |
| 1164 | — | — | 是 | c24 | 0.033 |
| 1170 | — | — | 是 | c24 | 0.033 |
| 1175 | — | — | 是 | c24 | 0.033 |
| 1181 | — | — | 是 | c24 | 0.033 |
| 1271 | — | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1272 | — | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1273 | — | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1274 | — | — | 否 | c25 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1289 | — | — | 是 | c25 | 0.0 |
| 1290 | — | — | 是 | c25 | 0.0 |
| 1297 | — | — | 是 | c25 | 0.033 |
| 1298 | — | — | 是 | c25 | 0.067 |
| 1307 | — | — | 是 | c25 | 0.067 |
| 1308 | — | — | 是 | c25 | 0.033 |
| 1316 | — | — | 是 | c25 | 0.0 |
| 1317 | — | — | 是 | c25 | 0.0 |
| 1323 | — | — | 是 | c25 | 0.0 |
| 1324 | — | — | 是 | c25 | 0.067 |
| 1333 | — | — | 是 | c25 | 0.033 |
| 1334 | — | — | 是 | c25 | 0.0 |
| 1342 | — | — | 是 | c25 | 0.0 |
| 1343 | — | — | 是 | c25 | 0.0 |
| 1350 | — | — | 是 | c25 | 0.0 |
| 1351 | — | — | 是 | c25 | 0.033 |
| 1394 | — | — | 是 | c26 | 0.0 |
| 1395 | — | — | 是 | c26 | 0.033 |
| 1401 | — | — | 是 | c26 | 0.033 |
| 1402 | — | — | 是 | c26 | 0.033 |
| 1437 | — | — | 是 | c27 | 0.0 |
| 1532 | — | — | 是 | c28 | 0.033 |
| 1533 | — | — | 是 | c28 | 0.033 |
| 1547 | — | — | 是 | c29 | 0.033 |
| 1548 | — | — | 是 | c29 | 0.0 |
| 1556 | — | — | 是 | c25, c29 | 0.0 |
| 1557 | — | — | 是 | c25, c29 | 0.0 |
| 1563 | — | — | 是 | c25, c29 | 0.033 |
| 1564 | — | — | 是 | c25, c29 | 0.033 |
| 1570 | — | — | 是 | c25, c29 | 0.067 |
| 1571 | — | — | 是 | c25, c29 | 0.1 |
| 1581 | — | — | 是 | c25, c29 | 0.033 |
| 1582 | — | — | 是 | c25, c29 | 0.067 |
| 1589 | — | — | 是 | c25, c29 | 0.067 |
| 1590 | — | — | 是 | c25, c29 | 0.033 |
| 1596 | — | — | 是 | c25, c29 | 0.067 |
| 1597 | — | — | 是 | c25, c29 | 0.033 |
| 1603 | — | — | 是 | c25, c29 | 0.067 |
| 1604 | — | — | 是 | c25, c29 | 0.1 |
| 1611 | — | — | 是 | c25, c29 | 0.067 |
| 1612 | — | — | 是 | c25, c29 | 0.033 |
| 1703 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1704 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1705 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1706 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1720 | — | — | 是 | c31 | 0.033 |
| 1721 | — | — | 是 | c31 | 0.033 |
| 1725 | — | — | 是 | c31 | 0.067 |
| 1726 | — | — | 是 | c31 | 0.067 |
| 1731 | — | — | 是 | c31 | 0.033 |
| 1732 | — | — | 是 | c31 | 0.033 |
| 1735 | — | — | 是 | c31 | 0.067 |
| 1736 | — | — | 是 | c31 | 0.033 |
| 1741 | — | — | 是 | c31 | 0.0 |
| 1742 | — | — | 是 | c31 | 0.033 |
| 1745 | — | — | 是 | c31 | 0.033 |
| 1746 | — | — | 是 | c31 | 0.0 |
| 1751 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，本次评测没有跑；骨架里的分数来自后面用同名 --json-output-file 重跑… |
| 1752 | — | — | 是 | c31 | 0.033 |
| 1758 | — | — | 否 | c31 | 未拿到：vLLM 服务器启动失败，且 agent 没有重跑这一点 |
| 1759 | — | — | 是 | c31 | 0.033 |
| 1832 | — | — | 是 | c32 | 0.067 |
| 1833 | — | — | 是 | c32 | 0.067 |
| 1837 | — | — | 是 | c32 | 0.033 |
| 1838 | — | — | 是 | c32 | 0.033 |
| 1841 | — | — | 是 | c32 | 0.067 |
| 1842 | — | — | 是 | c32 | 0.033 |
| 1846 | — | — | 是 | c32 | 0.067 |
| 1847 | — | — | 是 | c32 | 0.067 |
| 1851 | — | — | 是 | c32 | 0.033 |
| 1852 | — | — | 是 | c32 | 0.033 |
| 1855 | — | — | 是 | c32 | 0.033 |
| 1856 | — | — | 是 | c32 | 0.067 |
| 1859 | — | — | 是 | c32 | 0.033 |
| 1860 | — | — | 是 | c32 | 0.033 |
| 1864 | — | — | 是 | c32 | 0.033 |
| 1865 | — | — | 是 | c32 | 0.033 |
| 1924 | — | — | 是 | c33 | 0.0 |
| 1925 | — | — | 是 | c33 | 0.0 |
| 1928 | — | — | 是 | c33 | 0.033 |
| 1929 | — | — | 是 | c33 | 0.067 |
| 1933 | — | — | 是 | c33 | 0.033 |
| 1934 | — | — | 是 | c33 | 0.0 |
| 1937 | — | — | 是 | c33 | 0.0 |
| 1938 | — | — | 否 | c33 | 未拿到：vLLM 服务器启动失败，且 agent 没有重跑这一点 |
| 1973 | — | — | 是 | c34 | 0.067 |
| 1978 | — | — | 是 | c34 | 0.033 |
| 1982 | — | — | 是 | c34 | 0.033；for 循环，8 个 2/30 候选在 6 并发下重测，结果 1/30~2/30，无人超过导出件（机械层取回… |
| 1995 | — | — | 是 | c34 | 0.067；for 循环，8 个邻域/边界候选，读到 1/30~2/30，无人超过 2/30（机械层取回 0.067） |
| 2032 | — | — | 是 | c36 | 0.067；for 循环，9 个 19.5%~22.1% 的高分辨率插值点，读到 0/30~2/30（机械层取回 0.0… |
| 2072 | — | — | 是 | c37 | 0.067；for 循环，6 个 repetition_penalty 微调候选；0.999 读到 3/30（=0.1）… |
| 2090 | — | — | 是 | c38 | 0.067 |

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

## codex_non_api_max_gpt-5.6-sol_10h_run1__bfcl_HuggingFaceTB_SmolLM3-3B-Base_17396029
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | bfcl | HuggingFaceTB_SmolLM3-3B-Base | 6.71h | 0.94 |

### 改动序列(42 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 70 | C2 | Render every training row through the evaluator's own chat template and the evaluator's OpenAI tool wrapper (type/function, object properties, require… | i=70, i=108 |
| 89 | C3 | Build the initial SFT corpus from public function-calling sources (smoltalk2 xlam + hermes, ToolACE, locally generated synthetic) with explicit per-so… | i=89, i=33 |
| 104 | C4 | Create train_function_calling.py: full-parameter BF16 SFT with a completion-only collator, pad=<\|finetune_right_pad_id\|> (128004) and eos retargeted… | i=104, i=112 |
| 124 | C8 | Create package_checkpoint.py so weight-only Trainer checkpoints (no tokenizer/config) become directly loadable by the evaluator; without it vLLM rejec… | i=124, i=335 |
| 131 | C3 | make_stage2_data.py: derive a strict one-tool specialization split from the stage-1 corpus with extra weight on boolean/array/nested typing. | i=131, i=135 |
| 163 | C7 | Create validate_public.py: a self-built cheap proxy scorer (greedy vLLM exact-match on the held-out public split) used as a stand-in for the official … | i=163, i=211 |
| 214 | C7 | Re-align the proxy scorer to the official serving path by mirroring vLLM's non-streaming Hermes extraction semantics verbatim, after the proxy read 88… | i=214, i=581, i=202 |
| 249 | C3 | Reweight the stage-2 corpus toward long / nested / multi-argument traces after the failure analysis attributed BFCL losses to malformed nested JSON. | i=249, i=248 |
| 277 | C3 | prepare_glaive_data.py: add the Glaive function-calling corpus as a new data source (21,415 kept of 112,960 scanned). | i=277, i=297 |
| 311 | C3 | prepare_structured_data.py: locally generated deterministic recursive JSON-schema examples aimed at long/nested calls (10,489 kept). | i=311, i=326 |
| 323 | C3 | make_stage3_data.py: combine stage-2 + Glaive + structured synthetic into a 93,967-row corpus. | i=323, i=332 |
| 380 | C4 | Add --format-loss-weight / --json-close-loss-weight to the trainer, up-weighting only closing braces, </tool_call> and end-of-turn (not the opening ta… | i=380, i=379 |
| 400 | C3 | make_redundant_close_data.py: insert one extra </tool_call> token before <\|im_end\|> in every target. Built and verified but never used for a trainin… | i=400, i=403 |
| 422 | C4 | Make the boundary loss weighting symmetric (open AND close tokens) after the aggregate showed the boundary failure is symmetric (21 close-only vs 16 o… | i=422, i=426 |
| 426 | C10 | Adopt an explicit benchmark firewall for eval-log analysis: only tag-presence counts and correctness flags are read out of the official logs, never te… | i=426, i=169, i=96 |
| 447 | C11 | analyze_eval_format.py: turn the official inspect_ai eval log into decision signals (tool-tag presence pattern, parsed-but-wrong count, per-sample cor… | i=447, i=450, i=426 |
| 462 | C3 | make_boundary_data.py: supervise only the open/close/im_end and JSON closing-bracket token pieces. Built but never used for a training run. | i=462, i=466 |
| 514 | C2 | make_think_data.py: prefix every target with an empty thought block <think>\n\n</think>\n so the training target matches the evaluator's /think prompt… | i=514, i=512, i=518 |
| 632 | C1 | Write temperature: 0.0 into checkpoints/stage1/checkpoint-1400/generation_config.json and candidate_model/generation_config.json after finding evaluat… | i=631, i=632, i=639 |
| 651 | C1 | Change package_checkpoint.py so every packaged checkpoint gets temperature 0.0, making greedy decoding the default for all later candidates. | i=651, i=658 |
| 656 | C8 | Work around transformers' GenerationConfig.validate rejecting temperature with do_sample=False by patching generation_config.json as raw JSON after sa… | i=656, i=654, i=1366 |
| 735 | C3 | prepare_semantic_curriculum.py: 80,000 locally generated argument-binding / optional-omission examples across 20 task families, targeted at the six re… | i=735, i=734, i=739 |
| 762 | C4 | Add --argument-loss-weight: multiply the loss on tokens after the ' "arguments":' marker (marker fixed at i=792 to include the leading space). | i=762, i=907, i=794 |
| 817 | C3 | make_reasoning_data.py: derive a deterministic concise argument-mapping scratchpad target (<think> ... </think> before the call) from the semantic cor… | i=817, i=820 |
| 981 | C1 | Build candidate_rp{090,095,098,102} by hard-linking checkpoint-1400 and copying its generation_config.json, then adding repetition_penalty. Field-leve… | i=981, i=984, i=986 |
| 1000 | C1 | Temperature sweep: candidate_temp{005,010,020,030} (extended to 035/040/050 at i=1055, and applied to two trained variants at i=1060). Same copy-then-… | i=1000, i=1003, i=1040 |
| 1023 | C3 | make_semantic_focus_data.py: keep only the 80,000 locally generated semantic_curriculum rows, dropping all public sources. | i=1023, i=1025 |
| 1034 | C8 | Fix make_reasoning_focus_data.py's source filter, which selected on the pre-transform source name and produced a 0-row training set. | i=1034, i=1029, i=1036 |
| 1080 | C1 | Constrained-sampling sweep: candidate_sample_{p080,p090,p095,k2,k3,k5}, later p096-p099 (i=1144) and a temperature x top_p grid t020p095..t038p095 (i=… | i=1080, i=1083, i=1152 |
| 1233 | C9 | Commit final_model = an untouched copy of stage1/checkpoint-1400 packaged with temperature 0.0, deliberately choosing the reproducible 0.94 over the s… | i=1233, i=1232 |
| 1247 | C10 | Final compliance/integrity audit of the submitted artifact: shard/index key equality, tokenizer and generation ids, temperature, no .bin files, plus s… | i=1247, i=1248 |
| 1268 | C3 | prepare_apigen_full.py: add argilla/apigen-function-calling as a new source (45,554 single-call rows kept of 109,402). | i=1268, i=1271 |
| 1273 | C3 | make_apigen_novel_data.py: keep only the 19,614 APIGen rows that are not token-identical to xLAM rows already used in stage 1. | i=1273, i=1275 |
| 1283 | C3 | prepare_toolace_novel.py: the ToolACE single-call rows not already used in stage 1 (5,008 kept). Built but never used for a training run. | i=1283, i=1286 |
| 1288 | C3 | prepare_hermes_novel.py: the Hermes rows not already used in stage 1 (24 novel). Built but never used for a training run. | i=1288, i=1290 |
| 1382 | C1 | Repackage stage_apigen_novel/checkpoint-600 to restore temperature 0.0 after the raw trainer export (candidate_apigen_novel, byte-identical weights) s… | i=1382, i=1385, i=1377 |
| 1426 | C6 | interpolate_checkpoints.py: linear weight interpolation (model soup) between two checkpoints with an alpha coefficient; used for 10 soups over three c… | i=1426, i=1428 |
| 1615 | C3 | Repair the OpenFunctions-v1 loader: reverse a character-per-line corruption in the public mirror, lifting parseable single-call answers from 2 to 11,1… | i=1615, i=1614, i=1612 |
| 1659 | proposed:checkpoint_tens… | splice_checkpoints.py: build a candidate by taking whole trained tensors from one checkpoint and the rest from another (first/last N transformer block… | i=1659, i=1661, i=1701 |
| 1720 | C1 | candidates/tokenfix: copy final_model and re-save its tokenizer with fix_mistral_regex=True, acting on the transformers warning emitted on every load. | i=1720, i=1719 |
| 1747 | proposed:checkpoint_tens… | Extend splice_checkpoints.py with a seeded random per-tensor donor choice ('tensor soup', p=0.5) between checkpoint-1050 and checkpoint-1400. | i=1747, i=1749, i=1746 |
| 1757 | C8 | Delete the accumulated candidates/ directory (114 GB) after safetensors hit ENOSPC with the filesystem 100% full, freeing 115 GB so the last experimen… | i=1757, i=1750, i=1755 |

### 训练序列(13 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 109 | real | 0.56h | returned | **baseline** | baseline - first training. Full-parameter SFT of SmolLM3-3B-Base on training_data (44,943 rows), 1400 steps, lr 2e-5, micro-batch 8 x grad-accum 4, ch… |
| 254 | real | 0.20h | returned | **both** | Data changed (training_data -> training_data_stage2: strict single-tool split with typed/long/nested oversampling, 51,574 rows) AND method changed (co… |
| 382 | real | 0.07h | returned | **both** | Data changed (training_data_stage3 = stage2 + Glaive + structured synthetic, 93,967 rows) AND method changed (branch back to stage1/checkpoint-1400, l… |
| 424 | real | 0.21h | returned | **C4** | Relaunch of the aborted i=382 run with byte-identical data-dir, steps, batch, grad-accum, lr and loss-weight flags; the only difference is the trainer… |
| 524 | real | 0.25h | returned | **both** | Data changed (training_data_think = stage3 targets prefixed with <think>\n\n</think>\n) AND method changed (lr 3e-6 -> 5e-6, 400 -> 500 steps, the for… |
| 741 | real | 0.41h | returned | **C3** | Data changed (training_data_semantic = 80,000 new locally generated argument-binding examples mixed with the original 44,943 public rows). lr 5e-6, ba… |
| 907 | real | 0.22h | returned | **C4** | Same data-dir (training_data_semantic), same lr 5e-6, same batch 8 x grad-accum 4 as i=741; the added variable is --argument-loss-weight 3.0 (3x loss … |
| 1075 | real | 0.23h | returned | **both** | Data changed (training_data_reasoning_focus = the 80,000 semantic rows rewritten with an explicit argument-mapping scratchpad target) AND lr changed 5… |
| 1137 | real | 0.22h | returned | **both** | Data changed (training_data_semantic_focus = the same 80,000 semantic rows WITHOUT the scratchpad rewrite) AND lr changed 1e-5 -> 2e-5. The agent stat… |
| 1280 | real | 0.23h | returned | **both** | Data changed (training_data_apigen_novel = 19,614 APIGen rows not token-identical to stage-1 xLAM) AND lr changed 2e-5 -> 5e-6, steps 500 -> 600, chec… |
| 1546 | real | 0.10h | returned | **C4** | Identical data (training_data_stage2, same as i=254), identical lr 5e-6, identical batch 8 x grad-accum 4; the variable under test is the starting che… |
| 1567 | real | 0.05h | returned | **unclear** | Byte-identical to i=1546 except --save-steps 10 --eval-steps 10 and a new output dir. No data and no hyperparameter under test: the run exists solely … |
| 1634 | real | 0.06h | returned | **both** | Data changed (training_data_gorilla_recovered_split = 3,628 rows recovered from the repaired OpenFunctions-v1 mirror) AND lr changed 5e-6 -> 2e-6, ste… |

### 验证序列(54 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 18 | 3.0 | 20.0 | 是 |  | 0.0 (untouched base model, n=20; agent read it as 0/20 exact… |
| 198 | 4.0 | -1.0 | 是 | c1, c2, c3 | 0.56 |
| 357 | 4.0 | -1.0 | 是 | c7, c8 | 0.54 |
| 361 | 4.0 | -1.0 | 是 | c7, c8 | 0.53 |
| 498 | 4.0 | -1.0 | 是 | c9, c10, c11, c13 | 0.51 |
| 505 | 4.0 | -1.0 | 是 | c9, c10, c11, c13 | 0.54 |
| 664 | — | — | 是 | c16, c17 | 0.94 (same weights as the 0.56 read at i=198; only generatio… |
| 673 | — | — | 是 | c15 | 0.93 |
| 677 | — | — | 是 | c15 | 0.94 |
| 688 | — | — | 是 | c7, c8, c9, c10, c11, c13, c17 | stage2_250 0.94 / stage2_500 0.94 / stage3_200 0.94 / stage3… |
| 714 | — | — | 是 | c1, c2, c3 | stage1_350 0.90 / stage1_700 0.93 / stage1_1050 0.94 (three … |
| 890 | — | — | 是 | c19 | 250 0.94 / 500 0.94 / 750 0.94 / 1000 0.94 (read at i=902) |
| 998 | — | — | 是 | c20 | argweighted-250 0.94 / argweighted-500 0.94 (read at i=1013;… |
| 998 | — | — | 是 | c20 | argweighted-250 0.94 / argweighted-500 0.94 (read at i=1013;… |
| 998 | — | — | 是 | c22 | rp090 0.92 / rp095 0.93 / rp098 0.94 / rp102 0.93 - read fro… |
| 998 | — | — | 是 | c22 | rp090 0.92 / rp095 0.93 / rp098 0.94 / rp102 0.93 - read fro… |
| 1015 | — | — | 是 | c23 | temp005 0.94 / temp010 0.94 / temp020 0.95 / temp030 0.95 - … |
| 1058 | — | — | 是 | c23 | temp035 0.95 / temp040 0.94 / temp050 0.92 - read at i=1064 |
| 1066 | — | — | 是 | c23, c19, c20 | semantic-temp020 0.94 / argweighted-temp020 0.94 - read at i… |
| 1123 | — | — | 是 | c21 | reasoning-125/250/375/500 all 0.93 (read at i=1124; the skel… |
| 1132 | — | — | 是 | c24 | p080 0.94 / p090 0.94 / p095 0.95 / k2 0.94 / k3 0.94 / k5 0… |
| 1181 | — | — | 是 | c24 | p096 0.95 / p097 0.95 / p098 0.94 / p099 0.93 / t020p095..t0… |
| 1193 | — | — | 是 | c25 | semantic-highlr-125 0.92 / 250 0.93 / 375 0.83 / 500 0.82 (r… |
| 1197 | — | — | 是 | c23, c24 | temp020-run1 0.94 / temp020-run2 0.94 / sample_p095-run1 0.9… |
| 1241 | — | — | 是 | c27 | 0.94 |
| 1320 | — | — | 是 | c29, c30 | 0.92 |
| 1333 | — | — | 是 | c29, c30 | 0.93 |
| 1339 | — | — | 是 | c29, c30 | 0.92 |
| 1344 | — | — | 否 | c29, c30 | no score - vLLM refused the directory because checkpoint-600… |
| 1358 | — | — | 是 | c29, c30 | 0.55 - later shown to be a decoding artifact (raw trainer ex… |
| 1384 | — | — | 是 | c29, c30, c31 | 0.92 |
| 1431 | — | — | 是 | c32 | 0.93 |
| 1468 | — | — | 是 | c32 | 0.94 |
| 1478 | — | — | 是 | c32 | 0.94 |
| 1489 | — | — | 是 | c32 | 0.94 |
| 1498 | — | — | 是 | c32 | 0.94 |
| 1505 | — | — | 是 | c32 | 0.93 |
| 1514 | — | — | 是 | c32 | 0.94 |
| 1522 | — | — | 是 | c32 | 0.93 |
| 1530 | — | — | 是 | c32 | 0.94 |
| 1536 | — | — | 是 | c32 | 0.93 |
| 1555 | — | — | 是 |  | 0.94 (judges the i=1546 training, whose data/hyperparameters… |
| 1561 | — | — | 是 |  | 0.94 (same as above) |
| 1579 | — | — | 是 |  | 0.94 (judges the i=1567 checkpoint-harvesting run; no change… |
| 1587 | — | — | 是 | c32 | 0.93 |
| 1642 | — | — | 是 | c33 | 0.93 |
| 1648 | — | — | 是 | c33 | 0.93 |
| 1666 | — | — | 是 | c34 | 0.93 |
| 1672 | — | — | 是 | c34 | 0.93 |
| 1683 | — | — | 是 | c34 | 0.94 |
| 1689 | — | — | 是 | c34 | 0.94 |
| 1709 | — | — | 是 | c34 | 0.94 |
| 1713 | — | — | 是 | c34 | 0.94 |
| 1723 | — | — | 是 | c36 | 0.67 |
| 1730 | — | — | 是 | c27 | 0.94 (default evaluator settings, no --limit / --max-connect… |
| 1762 | — | — | 是 | c35 | 0.93 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[1567]
- **1 次验证没有拿到信号**:i=[1344]
- **分类学缺口提案 2 条**
  - checkpoint_tensor_splice(i=1661, i=1702, i=1749, i=1655)
  - checkpoint_harvest_rerun(i=1567, i=1568, i=1584)
- **定义缺陷 4 条**
  - eval_score_association_wrong_on_loop_launches(i=998, i=1013, i=1124, i=1195)
  - got_signal_false_for_shell_loop_launches(i=1015, i=1042, i=1040)
  - end_reason_returned_conflates_completed_and_aborted(i=426, i=419, i=1568)
  - config_access_type_mislabelled_write(i=1739, i=1739)
- **边界情形 5 条**
  - baseline_training_has_no_tested_variable(i=109, i=108)
  - max_steps_scaling_with_corpus_size(i=741, i=739)
  - starting_checkpoint_as_tested_variable(i=1546, i=1419)
  - tokenizer_file_is_serving_config_but_not_generation_config(i=1720, i=1726, i=1728)
  - packaging_step_is_both_C8_and_C1(i=644, i=658, i=1377, i=1383)

## codex_non_api_max_gpt-5.6-sol_10h_run1__bfcl_Qwen_Qwen3-1.7B-Base_17409825
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | bfcl | Qwen_Qwen3-1.7B-Base | 8.91h | 0.95 |

### 改动序列(40 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 55 | C3 | 建初始 SFT 语料 prepare_data.py:只留单次调用行,来源 minpeter/xlam-function-calling-60k-parsed(xlam 27,077)+ NousResearch/hermes-function-calling-v1(458),归一化成 OpenAI… | i=55, i=54 |
| 61 | C10 | 把 28,535 条候选整体过一遍官方 contamination_check.py(reference=../test_data.json),0 命中;过滤后再复检一次 train.jsonl,仍 0 命中。此后每一份新语料入训前都过同一道检查(全 run 约 30 次)。 | i=61, i=60, i=64 |
| 72 | C4 | 写 train_sft.py:全参 SFT、只对 response 段算 loss(prompt 全 mask)、用评测同一份 templates/qwen3.jinja 渲染、按长度分桶、每半个 epoch 存 checkpoint。 | i=72, i=74 |
| 80 | C8 | 首次训练 batch-size 32 在第一个 microbatch 就 OOM。改 16x2 梯度累积保持等效 batch 32 不变——纯可行性补偿,不动优化配方。 | i=80, i=79 |
| 85 | C8 | 16x2 仍 OOM(长度分桶把最长序列排在最前)。加 --gradient-checkpointing 与 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,训练才跑起来。这是本 run 唯一一次真正解锁管线的改动。 | i=85, i=85, i=84 |
| 110 | C1 | 改所有 sft_v1 checkpoint 的 generation_config:eos_token_id 由 [151643] 改成 [151645,151643](把训练出来的 <\|im_end\|> 也认成结束符),max_new_tokens 2048->256,补 bos/pad=15… | i=110, i=110, i=109, i=107 |
| 115 | C5 | 扫 sft_v1 的 0.5/1.0/1.5 epoch checkpoint(430/860/1290)与终点 1722,选中途最好的一个:0.89/0.90/0.94 vs 终点 0.88。本 run 第一次、也是收益最大的一次 checkpoint 选择(+6 点)。 | i=115, i=114, i=118 |
| 129 | C7 | 自建代理验证器 validate_model.py:直接用 vLLM 在自己划出的 1,000 条 xlam holdout 上贪婪生成、正则抽 <tool_call>、逐字段比对,并按 tool 数/值类型给出分类错误统计(892/1000 正确)。 | i=129, i=381, i=132 |
| 144 | C3 | stage2 难例课程:从 c8 代理验证器的错例里挑 87 条,复制 4 份,配 800 条校准 + 1,200 条稳定性样本,共 2,348 行;另留 200 条完全没动过的 holdout。 | i=144, i=143, i=147 |
| 163 | C6 | 写 interpolate_models.py 做 checkpoint 线性插值(safetensors 逐张量 lerp),先在 v1_1290 与 v2_hard_148 之间做 a=0.25/0.50 两个 soup。 | i=163, i=165 |
| 186 | C1 | 读 vllm/config/model.py 的 get_diff_sampling_param 后发现 evaluate.py 不传 temperature、vLLM 回退到 1.0,遂尝试写 temperature=0.0 + do_sample=False。GenerationConfig.v… | i=186, i=186, i=188, i=185 |
| 193 | C1 | 绕开校验:temperature=0.0 且 do_sample=None(i=190 先单测了 True/False/None 三种取值)。这是相对 c6 配置的**单字段**改动,i=194 逐字打印了改后 diff。 | i=193, i=194, i=191 |
| 198 | C4 | 第二次从 base 开始的完整训练,把学习率从 4e-5 降到 2e-5、换 shuffle 种子 9929、并用全部 28,535 条 clean_all(而非留出 validation 后的 27,535),目的是检验 v1 的取值类错误是不是种子/优化过程特有的。 | i=198, i=197, i=197 |
| 235 | C3 | 加 ToolACE(lockon/ToolACE)语料:AST 解析 system 段里的函数表 + 首轮 assistant 的单次调用,保守保留 2,040 行;后来重写解析器扩到 5,057 行(i=390)。 | i=235, i=238, i=393 |
| 244 | C3 | 写 augment_counterfactual.py:只在原始值逐字出现在 query 里时才替换它(ID/日期/坐标/枚举/数值),同时改 query 与 call,生成 12,000 条「值复制」反事实样本,目的是压掉对常见 ID 的记忆。 | i=244, i=691 |
| 276 | C1 | 写 configure_inference.py 作为统一解码配置工装,此后每个 checkpoint 都过一遍:do_sample=True/temperature=1.0/top_k=1/top_p=1.0(用 top_k=1 拿确定性 argmax,规避 temperature=0 被 tra… | i=276, i=988, i=284, i=283 |
| 316 | C11 | 写 compare_evals.py:把官方 inspect_ai 日志里每条样本的 bfcl_scorer 判定抽成正确性向量,输出 correct / wrong_ids / oracle_union。这是本 run 的决策主力——分数在 0.92~0.95 的噪声带里说明不了什么,错题号集合与… | i=316, i=1381, i=1381 |
| 368 | C10 | 去污染的自定政策:凡是自称基于 BFCL 条目构建的数据集,一律排除,不管官方 checker 判不判它污染。 | i=368 |
| 375 | C3 | 加 argilla/Synth-APIGen-v0.1:筛出 19,088 条单次调用行,归一化后过污染检查 0 命中。 | i=375, i=378 |
| 411 | C3 | 自查发现反事实生成器有子串碰撞 bug(两字母语言码被替换进无关单词里,如 'sentence'->'ssvtsvce'),35 条被改的值不再逐字出现在 query 中。改成 token 边界替换后重生成,重跑不变量审计得 missing_changed_literal=0。同时**主动 kill… | i=408, i=411, i=404, i=414 |
| 478 | C9 | 写 smoke_test_model.py:用 Transformers(不经 vLLM、不经 evaluate.py)加载产物、贪婪生成一个构造好的工具调用,断言输出逐字段等于期望的 call。交付前跑过一次,通过。 | i=478, i=1654 |
| 554 | C3 | 按 Hammer(Lin et al. 2024)的思路做标识符 masking:把函数名与参数名换成随机无意义标识符并同步改目标 call,逼模型读 description 并从 schema 里抄。读了 Hammer 仓库源码后发现其真实 masking 比例是 2/3 而非 1/3(i=699… | i=554, i=553, i=703 |
| 613 | C3 | 加 Seal-Tools 官方 train split(git clone,3,580 条单次调用),显式只用 train、绝不碰它的 test_in_domain / test_out_domain。 | i=613, i=630, i=615 |
| 888 | C8 | v6 在 batch-size 32 上 OOM(仅全词表 logits 就要再要 37 GB),退回已验证的 16x2 等效 batch 重启,优化配方不变。 | i=888, i=887 |
| 938 | C3 | 写 augment_optional_schema.py:往干净样本的 schema 里注入若干「用户没要就不该填」的可选参数(include_metadata/output_format/language_code/...),query 与目标 call 完全不动,构造一个规则可验的「别乱填可选字… | i=938, i=937, i=941 |
| 973 | C3 | 写 mask_optional_schema.py:对 c20 的可选参数集再叠一层参数名 masking(302,035 个参数被改名),与未 mask 版一起构成 stage_grounding 课程。 | i=973, i=975 |
| 1030 | C6 | 跨分支 soup:以 v4_1260 为基,混入被判「多解出两题」的 v2_hard/checkpoint-74,权重 0.15/0.30/0.50,想吸收互补的正确项。 | i=1030, i=1029 |
| 1047 | C11 | 发现并纠正自己工装里的 log->checkpoint 映射错误:此前把 soup_hard 的日志当成 v2_hard 的日志,算出 96% 的 oracle 上界。用 read_eval_log 逐个日志打印 eval.model 与 accuracy 重建映射后,真实上界是 95%。 | i=1047, i=1053, i=1048 |
| 1058 | C1 | 写 configure_sampling.py:用硬链接做零拷贝的「只改 generation_config」checkpoint 视图,然后在 v4/v7 上扫 (temperature, top_k, top_p) 网格——0.2/0.25/0.3/0.35/0.4/0.425/0.485/0.… | i=1058, i=1489, i=1279 |
| 1064 | C6 | SWA 式 soup:v4_aug_clean 的 checkpoint-1260 与 checkpoint-1890 五五平均。 | i=1064 |
| 1148 | C3 | 用沙箱缓存里的 Qwen3-4B-Instruct-2507 做本地教师,给已有 schema+call 重新生成自然语言 query,并机械校验每个标注值都逐字出现在生成的 query 里(7,081/12,000 通过)。不调用任何外部 API。 | i=1148, i=1147, i=1219 |
| 1400 | C4 | 同一份 optional_schema 数据、同样 0.5 epoch,把学习率从 1e-6 抬到 5e-6,并把存点间隔缩到 100 步,专门测「是不是训得不够」。 | i=1400, i=1399 |
| 1450 | C3 | 写 generate_binding_curriculum.py:纯合成的「参数绑定」课程 30,000 条——多参数、值顺序打乱、同类型干扰值、相似竞争工具、可选项省略,专门训 grounding 与选择。 | i=1450, i=1449 |
| 1497 | C1 | 写 configure_penalty.py 扫 repetition_penalty 1.005/1.01/1.02/1.05(先在 vllm 源码里确认该字段确实会被 get_diff_sampling_param 读到)。1.005 持平 0.95,1.01 掉到 0.94,后两档没跑。 | i=1497, i=1499, i=1516 |
| 1539 | C4 | 同一份 optional_schema 数据,学习率再抬一档到 2e-5、epochs 缩到 0.3,做一次刻意高幅度的冲击,验这个决策边界到底动不动得了。 | i=1539, i=1538 |
| 1562 | C6 | 改 interpolate_models.py 放开权重下界,做**负权重**任务向量外推:以 v7_620 为基、v6_2570 为 other,other_weight = -0.25/-0.5/-1(即 base_weight 1.25/1.5/2.0),想把 v7 相对 v6 的更新方向放大… | i=1562, i=1564, i=1565, i=1558 |
| 1583 | C1 | 针对反复出现的 transformers 分词器警告写 configure_tokenizer_fix.py,做一个带 fix_mistral_regex 的打包变体。i=1586 先用第一档静态检查证明两者对同一串文本产出**完全相同的 38 个 token id**,但仍然又花了一次全量评测确认… | i=1583, i=1576, i=1586 |
| 1603 | C3 | 写 add_optional_defaults.py:给 schema 的可选参数补上显式 default 值(共 117,370 处),教模型「schema 里的 default 不等于该填进 call」。 | i=1603, i=1599, i=1604 |
| 1639 | C9 | 提交守卫:先确认 final_model 尚不存在,再 cp -a v7_stage_grounding/checkpoint-620 过去,重跑 configure_inference,然后 (a) sha256 逐字节核对权重与源 checkpoint 一致、(b) 检查无外部软链、(c) 用 … | i=1633, i=1639, i=1745, i=1746 |
| 1717 | C4 | 最后一条训练路线:用 trl 的 DPOTrainer 在 6,500 条合成偏好对上做半个 epoch——chosen 是正确 call,rejected 是多填一个未被请求的可选键。 | i=1717, i=1708 |

### 训练序列(18 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 75 | real | 0.02h | returned | **baseline** | baseline:本 run 第一次真实训练。data/train.jsonl 27,535 行(xlam+hermes),全参 SFT,2 epoch,lr 4e-5,等效 batch 32。取值域装不下「baseline」故填 unclear,不是证据不足。结局:第一个 microbatch 就… |
| 80 | real | 0.02h | returned | **unclear** | 与 i=75 逐字同配方,只把 batch 32 拆成 16x2 累积以保持等效 batch 不变——纯显存补偿,没有任何受测变量。仍然 OOM,0 步。 |
| 85 | real | 0.45h | returned | **unclear** | 仍是同一个 baseline 配方,再加 gradient checkpointing 与 expandable_segments。这是三次尝试里唯一跑通的一次(train_runtime 1561s,2 epoch,eval_loss 0.0192),产出 checkpoint-430/860/1… |
| 149 | real | 0.07h | returned | **both** | 从 sft_v1/checkpoint-1290 热启;数据换成 2,348 行难例课程(c9),同时把 lr 从 4e-5 降到 5e-6、epoch 从 2 提到 3。lr 的量级变化不是数据强制的补偿,而是「在已有 checkpoint 上做低速精修」这一设计的一部分,因此数据与超参都在变。 |
| 198 | real | 0.46h | returned | **both** | 重新从 base 开始。数据:clean_all.jsonl 28,535(把原本留作 validation 的 1,000 行也放回训练);超参:lr 4e-5->2e-5、seed 9929。agent 自陈这两项都是要测的。 |
| 357 | real | 0.15h | returned | **C3** | 从 base 开始,数据换成 augmented(41,375 = 27,535 原始 + 12,000 反事实 + 1,840 ToolACE)。lr/batch/累积与 i=198 逐字相同;epochs 1.5 vs 2 是语料变大后维持总步数的补偿(1,940 步 vs 1,784 步),按… |
| 427 | real | 0.53h | returned | **C3** | 与 i=357 的命令**逐字相同**(只换 output-dir),数据是修好 bug 后重生成的同一份 augmented。这是本 run 最接近干净的一次 C3 对照:超参完全不变,只有数据的正确性变了。结果 0.5/1.0/1.46/1.5 epoch = 0.92/0.94/0.94/0.… |
| 667 | real | 0.83h | returned | **C3** | 数据换成 diverse(65,760 = broad 62,180 + Seal-Tools 3,580,新增 Synth-APIGen 18,088 与扩版 ToolACE 4,557)。lr 2e-5 / batch 16x2 / epochs 1.5 与 i=427 逐字相同,只有 seed… |
| 882 | real | 0.05h | returned | **both** | 数据换成 augmented_mask(82,243 = augmented + 40,868 masked 变体),同时把 batch 提到 32 / 累积 1(agent 在 i=837-838 明确是在测 H100 上 batch32 的吞吐)。两项都是有意的。结局:反向传播时 OOM(仅 l… |
| 888 | real | 0.92h | returned | **C3** | 与 i=882 同数据,batch 退回已验证的 16x2。相对上一次成功训练(i=667)受测的是 masking 增强;epochs 1.25 是语料再变大后的步数补偿。结果 0.93/0.94/0.94——追平 v4 的 94,但错题集一字不差,masking 没换来新解出的题。 |
| 1114 | real | 0.71h | returned | **both** | 从 v6/checkpoint-2570 热启;数据换成 stage_grounding(59,633 = 20,000 标准 masked + 29,633 可选参数 masked + 10,000 可选参数未 masked),同时把 lr 一次性降到 3e-6(比 2e-5 低近一个量级)、ep… |
| 1228 | real | 0.25h | returned | **both** | 从冠军 v7/checkpoint-620 热启;数据换成 teacher_stage(21,242,本地教师改写的 query x2 + 复习集),lr 再降到 2e-6。数据与 lr 都换了。结果两个 checkpoint 都是 0.94,把 v7 刚解出的那题又丢了回去,分支作废。 |
| 1347 | real | 0.24h | returned | **both** | 从 v7/620 热启;数据 = optional_schema 30,000(纯未 masked 版),epochs 0.5、lr 1e-6。相对上一次训练(i=1228)数据与超参都变了,且 agent 把「短、低速」当成干预设计本身在描述。三个 checkpoint 的正确性向量与 v7 **… |
| 1400 | real | 0.19h | returned | **C4** | **本 run 最接近单变量的 C4 对照**:训练文件、验证文件、epochs(0.5)、batch、累积、起始 checkpoint 全部与 i=1347 相同,只把 lr 1e-6 -> 5e-6(另换了 seed 与存点间隔)。四个 checkpoint 全是 0.95 且错题集与 v7 完… |
| 1459 | real | 0.27h | returned | **C3** | 从 v7/620 热启;数据换成 binding_stage(50,000 = 30,000 合成参数绑定 + 20,000 grounding 复习)。lr 3e-6 与 v7 那次精修相同(即回到已验证过的取值,不是新的自变量),epochs 0.4。四个 checkpoint 全 0.95 且… |
| 1539 | real | 0.13h | returned | **C4** | 同 optional_schema 数据的第三档学习率:1e-6(v9)-> 5e-6(v10)-> **2e-5**(本次),epochs 0.3。数据文件与起点 checkpoint 逐字相同,受测的就是幅度。结局:agent 在 checkpoint-140 后主动中断(KeyboardInt… |
| 1607 | real | 0.15h | returned | **both** | 从 v7/620 热启;数据换成 optional_defaults(30,000,在 optional_schema 基础上再给可选参数补显式 default 值),lr 1e-5(介于 v10 的 5e-6 与 v12 的 2e-5 之间的新取值),epochs 0.4。数据与超参都动了。结局:… |
| 1722 | real | 0.12h | returned | **C4** | 换**训练方法**:trl DPOTrainer 代替 SFT,从 v7/620 出发,0.5 epoch。数据(6,500 条偏好对)必须换是方法本身的结构要求,不是独立的自变量,故记 C4。骨架「产物」列为空,实际产物是 runs/sft_v14_optional_dpo(checkpoint-… |

### 验证序列(62 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 11 | 3.0 | 80.0 | 是 |  | 0.0(未训练 base,80 题全错;agent 在 i=17 记为 0/80) |
| 325 | — | — | 是 | c13 | 0.92 |
| 332 | — | — | 否 | c38, c13 | 0.93(v3_446)/ 0.92(v3_892);机械层追不到是因为 stdout 被重定向进文件,agent 在 … |
| 340 | — | — | 否 | c38, c13 | 0.93(v3_1338)/ 0.93(v3_1784);同上,i=342-343 用 cat + compare_ev… |
| 350 | — | — | 否 | c10 | 0.92 / 0.92 / 0.93(a025/a050/a075);i=353-354 用 cat + compare… |
| 643 | — | — | 是 | c15, c16, c14 | 0.92 |
| 648 | — | — | 是 | c15, c16, c14 | 0.94;i=651-652 用 python -c json.load 取回 |
| 654 | — | — | 是 | c15, c16, c14 | 0.94;i=656-657 取回 |
| 658 | — | — | 否 | c15, c16, c14 | 0.93;i=660-661 用 compare_evals 取回(v4_1940 correct=93) |
| 852 | — | — | 是 | c17, c18 | 0.92;i=854-855 取回 |
| 856 | — | — | 是 | c17, c18 | 0.92;i=858-859 取回 |
| 860 | — | — | 否 | c17, c18 | 0.93;i=862-863 用 compare_evals 取回(v5_final correct=93) |
| 1092 | — | — | 是 | c19 | 0.93 |
| 1096 | — | — | 是 | c19 | 0.94 |
| 1099 | — | — | 是 | c19 | 0.94 |
| 1107 | — | — | 是 | c29 | 0.93 |
| 1109 | — | — | 是 | c28 | 0.93 |
| 1111 | — | — | 是 | c24 | 0.94 |
| 1198 | — | — | 是 | c20, c36 | 0.95(本 run 最高,冠军 checkpoint) |
| 1205 | — | — | 是 | c20, c36 | 0.94 |
| 1207 | — | — | 是 | c20, c36 | 0.94 |
| 1258 | — | — | 是 | c22 | 0.94 |
| 1260 | — | — | 是 | c22 | 0.94 |
| 1269 | — | — | 是 | c24 | 0.94 |
| 1272 | — | — | 是 | c24 | 0.95 |
| 1281 | — | — | 是 | c24 | 0.95 |
| 1291 | — | — | 是 | c24 | 0.93 |
| 1296 | — | — | 是 | c24 | 0.95 |
| 1329 | — | — | 是 | c24 | 0.95 |
| 1335 | — | — | 是 | c24 | 0.95 |
| 1371 | — | — | 是 | c20 | 0.95 |
| 1385 | — | — | 是 | c20 | 0.95 |
| 1390 | — | — | 是 | c20 | 0.95 |
| 1415 | — | — | 是 | c39 | 0.95 |
| 1420 | — | — | 是 | c39 | 0.95 |
| 1425 | — | — | 是 | c39 | 0.95 |
| 1430 | — | — | 是 | c39 | 0.95 |
| 1467 | — | — | 是 | c23 | 0.95 |
| 1471 | — | — | 是 | c23 | 0.95 |
| 1476 | — | — | 是 | c23 | 0.95 |
| 1480 | — | — | 是 | c23 | 0.95 |
| 1501 | — | — | 是 | c25 | 0.95 |
| 1505 | — | — | 是 | c25 | 0.94 |
| 1520 | — | — | 是 | c24 | 0.95 |
| 1524 | — | — | 是 | c24 | 0.94 |
| 1528 | — | — | 是 | c24 | 0.95 |
| 1533 | — | — | 是 | c24 | 0.92 |
| 1548 | — | — | 是 | c40 | 0.94 |
| 1552 | — | — | 是 | c40 | 0.95 |
| 1567 | — | — | 是 | c27 | 0.95 |
| 1571 | — | — | 是 | c27 | 0.94 |
| 1588 | — | — | 是 | c26 | 0.95(与贪婪基线错题集完全一致;i=1586 的静态 token 比对已提前给出同样结论) |
| 1614 | — | — | 是 | c35 | 0.94 |
| 1618 | — | — | 是 | c35 | 0.95 |
| 1662 | — | — | 是 | c31 | 0.95,且错题集与被选中的 v7_620 逐题相同——用 evaluate.py 默认并发 6 / max-token… |
| 1671 | — | — | 是 | c24 | 0.93 |
| 1682 | — | — | 是 | c24 | 0.95 |
| 1688 | — | — | 是 | c24 | 0.92 |
| 1697 | — | — | 是 | c24 | 0.94 |
| 1702 | — | — | 是 | c24 | 0.95 |
| 1733 | — | — | 是 | c34 | 0.95 |
| 1737 | — | — | 是 | c34 | 0.95 |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[80, 85]
- **5 次验证没有拿到信号**:i=[332, 340, 350, 658, 860]
- **分类学缺口提案 1 条**
  - data_invariant_check(i=404, i=408, i=414, i=499, i=880)
- **定义缺陷 3 条**
  - (i=77, i=409, i=885, i=1543, i=1610, i=408)
  - (i=115, i=118, i=110, i=156, i=169)
  - (i=118, i=188, i=188, i=194, i=197, i=330)
- **边界情形 2 条**
  - (i=1562, i=1564, i=1565, i=1558, i=1576)
  - (i=1583, i=1576, i=1586, i=1588)

## codex_non_api_max_gpt-5.6-sol_10h_run1__bfcl_Qwen_Qwen3-4B-Base_17409826
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | bfcl | Qwen_Qwen3-4B-Base | 8.71h | 0.92 |

### 改动序列(36 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 58 | C2 | 用评测自己的 templates/qwen3.jinja 渲染训练样本的 prompt,并做 token 级前缀对齐验证(前 1000 条全部 prefix-aligned),保证训练文本与推理时模型真实看到的 prompt 逐字一致。第一档、零 GPU、确定性。 | i=58, i=98 |
| 78 | C10 | 建立强制去污染流程:自写 filter_contamination.py,并规定每一份候选语料在训练前先跑官方 ../contamination_check.py,按行号剔除命中项。全 run 共跑 15 次检查,全部 0 命中。 | i=78, i=89, i=93 |
| 78 | C3 | prepare_data.py 建首份 SFT 语料 train_data.jsonl:xLAM/APIGen train split 26,351 条 + 3,000 条程序化合成,共 29,351 条;另切 1,401 条 xlam_test 作独立 holdout。 | i=78, i=94 |
| 78 | C7 | 自建代理验证器 score_local.py:在 1,401 条独立 xLAM holdout 上用 vLLM 做 exact-match 打分,替代昂贵的官方 BFCL 评测做候选筛选。 | i=78, i=147 |
| 100 | C4 | train_lora.py 改成 response-only loss:mask 掉 system/tool/user 全部 prompt token,只有 assistant 的 tool-call 序列参与损失。 | i=100, i=102 |
| 167 | C4 | 新增 calibrate_tool_tokens.py 解 LoRA 陷阱:Qwen3-Base 为 <tool_call>/</tool_call>/<\|im_end\|> 预留了 token id,但 LoRA 不训 lm_head,这三行输出向量始终冻结,导致 500/500 生成全部解析失… | i=167, i=169, i=172 |
| 200 | C7 | score_local.py 加 --details-output,把 holdout 逐题结果落盘,据此按参数个数/类型做误差归因(1 参 91.1%、2 参 84.2%、3 参 81.6%、4 参 71.8%),用来决定下一轮语料配方。 | i=200, i=202, i=218 |
| 219 | C3 | build_stage2.py:按参数复杂度加权重采样,把 30,752 行源语料扩成 54,545 行课程,上采样 3-12 参数与混合类型样本。 | i=219, i=222 |
| 267 | C7 | prepare_hermes.py:从 NousResearch/hermes-function-calling-v1 切出 400 条严格 JSON 的第二个独立 holdout(与训练集不相交),作为跨语料的第二个代理验证器。 | i=267, i=269 |
| 330 | C3 | prepare_glaive.py:新增第二个数据来源 glaiveai/glaive-function-calling-v2,按函数名封顶 50 条做多样性均衡,产出 6,340 条。 | i=330, i=332 |
| 364 | C3 | prepare_procedural_v2.py:程序化合成 10,000 条 4-6 参数、混合 JSON 类型、乱序子句、带干扰工具的压力样本(后又补嵌套对象与对象数组)。 | i=364, i=367 |
| 391 | C3 | build_diversity_curriculum.py:把 procedural_v2 10,000 + glaive 6,340 + hermes 78 + xlam_natural 10,000 + 挖出的 621 条难例合成 27,039 行多样性语料。 | i=391, i=394 |
| 410 | C3 | 收紧 Glaive 语料质量:剔除工具名重复的记录(6,340 条里 12 条)、强制 required 字段与 schema 校验,最终 6,337 条全部通过 valid_call。 | i=410, i=408, i=426 |
| 421 | C8 | 可行性修复:复用 prepare_data.valid_call 时 params.get('required') 可能是 bool,prepare_glaive.py 抛 TypeError 崩掉;补 schema 归一化后跑通。 | i=421, i=419 |
| 470 | C8 | train_lora.py 加 --gradient-checkpointing 与 --resume-from-checkpoint,用于 iter2 在 step 2,950/3,408 撞 CUDA OOM 后的恢复。 | i=470, i=468, i=476 |
| 562 | C11 | 验证器工装:第一次 score_local 因 vLLM 冗长初始化把会话冲断、分数拿不回来;改用 VLLM_LOGGING_LEVEL=ERROR 重跑把分数取回。此后全部评测命令都带这个前缀 —— 这正是 got_signal 的取回通道问题。 | i=562, i=558 |
| 664 | C5 | checkpoint 选择:把 iter1 的 checkpoint-1500(约 82% SFT 进度)merge 出来做早停候选,与 1834 步终点对比。 | i=664, i=667 |
| 677 | C5 | checkpoint 选择:merge iter3(多样性分支)的 checkpoint-800 做中点候选。 | i=677, i=676 |
| 724 | C4 | 低剂量 replay:不新建语料,把原始 train_data.jsonl 以 0.25 epoch、lr 2e-5 从 iter1 adapter 上再走一遍,checkpoint 落在 200/400 步以扫剂量。 | i=724, i=723 |
| 750 | C5 | checkpoint 选择:merge iter4_replay 的 checkpoint-200 做低剂量候选(replay 终点已被否)。 | i=750, i=749 |
| 770 | proposed:weight_row_surg… | scale_tool_rows.py:把已标定的 3 行 lm_head 行向量整体乘 0.75 / 1.25,其余输出行与全部语义权重逐位继承,再重新评测。不训练、不平均、不动 generation_config —— C1/C4/C6 都装不下(见 proposed_category p1)。 | i=770, i=772, i=775 |
| 797 | C6 | 新增 interpolate_models.py:两个兼容因果语言模型的逐参数线性插值(fp32 计算后回落 bf16),后续所有 soup / blend 都靠它。 | i=797, i=949 |
| 799 | C6 | 权重插值:iter1_final 与 iter3_final(多样性分支)按 alpha=0.10 混合,即保留 90% 现任权重、借 10% 多样性分支。 | i=799, i=796 |
| 813 | C6 | 权重插值收窄到 alpha=0.03。 | i=813, i=812 |
| 843 | C3 | 语料消融:从 train_data.jsonl 里只保留 26,351 条 xlam_train,剔除全部 3,000 条 procedural,生成 train_xlam_only.jsonl,重跑污染检查后训练。 | i=843, i=833 |
| 843 | C8 | 可行性修复:容器里没有 jq,原计划的 jq -c 'select(.source=="xlam_train")' 过滤失败,改用 rg 按行做 source 过滤。 | i=843, i=840, i=842 |
| 908 | proposed:variance_replic… | 种子重复训练:语料与全部超参与 i=846 逐字相同,只换 --seed(20260730,后再来一次 20260731)。目的是量化 run 间方差,并为 C6 权重平均造互补分量;不测任何配方或方法(见 proposed_category p2 / boundary_case b2)。 | i=908, i=910, i=1112 |
| 1000 | C3 | 数据规模消融语料:train_xlam_all.jsonl = train_xlam_only(26,351)+ validation(1,401)= 27,752 行,先过污染检查(0 命中)再训练。 | i=1000, i=1003 |
| 1088 | C6 | model soup:seed 20260729 与 seed 20260730 两个独立训练的合并模型做 50/50 权重平均。 | i=1088, i=1082 |
| 1091 | C3 | 把结构 token 标定所用的语料从默认 90/10 混合语料换成纯 train_xlam_only.jsonl(purecal)。语义权重逐位不变,只有这 3 行的标定数据变了。 | i=1091, i=1099 |
| 1178 | C6 | model soup:把 seed3 以 alpha=1/3 插进两 seed soup,得到三 seed 等权平均。 | i=1178, i=1125 |
| 1201 | C6 | model soup 降权变体:90% 两 seed soup + 10% seed3(alpha=0.1),预先声明后再评。 | i=1201, i=1200 |
| 1278 | C9 | 提交守卫:用 test ! -e final_model 先确认目标不存在再 cp -a 两 seed soup(当时的 0.940 冠军)进 final_model,避免无条件覆盖。 | i=1278, i=1273 |
| 1299 | C1 | 解码配置修复:final_model/generation_config.json 由 4 字段(eos_token_id/max_new_tokens/pad_token_id/transformers_version)改成 7 字段,新增 do_sample:false、temperature:… | i=1299, i=1296, i=1298, i=1358 |
| 1324 | C1 | 把同一份显式贪婪 generation_config 应用到 4 个挑战者目录(iter567_3seed_soup / iter567_seed3_alpha10 / iter5_xlam_only / iter8_xlam_all 的 purecal),消除采样混杂后在默认并发 6 下重排候选。 | i=1324, i=1323 |
| 1354 | C9 | 提交守卫 + 换产物:mv 旧 final_model 为 final_model_two_seed_backup,再 cp -a 三 seed soup 进 final_model,并用 sha256sum + cmp 逐字节核验两个 shard 与源目录一致。 | i=1354, i=1358, i=1359 |

### 训练序列(10 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 103 | real | 0.60h | returned | **baseline** | baseline —— 全 run 第一次训练。train_data.jsonl(26,351 xLAM + 3,000 procedural),rank-64 all-linear LoRA,1 epoch,bs4×accum4(有效 16),lr 1.5e-4,save-steps 500。取值… |
| 228 | real | 1.01h | returned | **both** | 相对 iter1 同时改了三件事:语料换成 train_stage2.jsonl(54,545 行复杂度加权课程,vs 29,351 行原始混合)=C3;lr 5e-5 vs 1.5e-4 = C4;并从 iter1_adapter 续训而非从 base 重训。数据与超参不可分。 |
| 472 | real | 0.01h | returned | **unclear** | 不是新实验:iter2 在 step 2,950/3,408 撞 CUDA OOM 后的第一次恢复尝试,bs 4→2 / accum 4→8 保持有效 batch 16 不变。Trainer 检出 batch-size 元数据变化后直接退出,零次参数更新 —— 什么变量都没验到。四个取值没有一个描述… |
| 477 | real | 0.73h | returned | **both** | 第二次 OOM 恢复并跑完 iter2:回到 checkpoint 原始 bs4×accum4,只加 --gradient-checkpointing(纯显存换算力,不改优化数学,保住原 3,408 步 schedule)。这是 C8 修复,受测变量仍是 iter2 的语料 + lr,故与 i=22… |
| 578 | real | 0.85h | returned | **both** | 语料换成 train_diversity_candidates.jsonl(27,039 行:procedural_v2 10,000 + glaive 6,340 + hermes 78 + xlam_natural 10,000 + 挖出难例 621)= C3;lr 再降到 2e-5 = C4。… |
| 724 | real | 0.16h | returned | **both** | 语料从 diversity 换回原始 train_data.jsonl = C3;同时把剂量压到 0.25 epoch、lr 2e-5、save-steps 200 = C4。目的是测『在原配方上再走一小段』的剂量。 |
| 846 | real | 0.61h | returned | **C3** | 相对 iter1(i=103)唯一的实质变化是语料:train_xlam_only.jsonl(26,351)= train_data 剔除 3,000 条 procedural。epochs 1.0 / bs4 / accum4 / lr 1.5e-4 与 i=103 逐字相同,同样从 base … |
| 908 | real | 0.54h | returned | **unclear** | 与 i=846 语料逐字相同(同一个 train_xlam_only.jsonl 文件)、超参逐字相同,只换 --seed 20260730(以及 --save-steps 800 vs 500)。受测的是 run 间方差与为 soup 造互补分量,C3/C4/both 都不描述它 —— 见 bou… |
| 1113 | real | 0.54h | returned | **unclear** | 第三次种子重复:语料与超参与 i=846/i=908 相同,只换 --seed 20260731(--save-steps 2000 即关掉中途存盘)。同 b2,受测变量是方差而非配方或方法。 |
| 1189 | real | 0.62h | returned | **C3** | 相对 i=846 只改语料规模:train_xlam_all.jsonl = train_xlam_only(26,351)+ 独立 xLAM holdout(1,401)= 27,752 行。--seed 20260729 与 i=846 完全相同,lr/bs/accum/epochs 逐字相同,… |

### 验证序列(28 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 23 | 3.0 | 100.0 | 是 |  | 0.0 |
| 184 | — | — | 是 | c2, c3, c5, c6 | 0.86 |
| 571 | — | — | 是 | c8 | 0.78 |
| 658 | — | — | 是 | c10, c11, c12 | 0.81 |
| 671 | — | — | 是 | c17 | 0.82 |
| 712 | — | — | 是 | c18 | 0.72 |
| 742 | — | — | 是 | c35 | 0.80 |
| 757 | — | — | 是 | c19 | 0.78 |
| 788 | — | — | 是 | c20 | 0.84 |
| 793 | — | — | 是 | c20 | 0.79 |
| 809 | — | — | 是 | c21, c22 | 0.83 |
| 822 | — | — | 是 | c21, c23 | 0.83 |
| 901 | — | — | 是 | c25 | 0.87 |
| 1076 | — | — | 是 | c36 | 0.87 |
| 1096 | — | — | 是 | c26 | 0.89 |
| 1105 | — | — | 是 | c26, c27 | 0.94 |
| 1184 | — | — | 是 | c28 | 0.91 |
| 1262 | — | — | 是 | c30 | 0.89 |
| 1270 | — | — | 是 | c29 | 0.90 |
| 1285 | — | — | 是 | c31 | 0.89 —— 与同一份权重此前读到的 0.940 不符,触发字节级核验并暴露出 C1 缺陷 |
| 1301 | — | — | 是 | c32 | 0.92 |
| 1314 | — | — | 是 | c32 | 0.92(与 i=1301 逐字复现,token 数同为 31,517) |
| 1319 | — | — | 是 | c32 | 0.92(并发降到评测器默认的 6 仍是 0.920) |
| 1326 | — | — | 是 | c33, c26 | 0.89(models/iter5_xlam_only_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c26 | 0.89(models/iter5_xlam_only_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c26 | 0.89(models/iter5_xlam_only_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c26 | 0.89(models/iter5_xlam_only_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c30 | 0.90(models/iter8_xlam_all_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c30 | 0.90(models/iter8_xlam_all_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c30 | 0.90(models/iter8_xlam_all_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c30 | 0.90(models/iter8_xlam_all_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c28 | 0.92(models/iter567_3seed_soup_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c28 | 0.92(models/iter567_3seed_soup_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c28 | 0.92(models/iter567_3seed_soup_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c28 | 0.92(models/iter567_3seed_soup_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c29 | 0.92(models/iter567_seed3_alpha10_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c29 | 0.92(models/iter567_seed3_alpha10_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c29 | 0.92(models/iter567_seed3_alpha10_purecal,贪婪重评) |
| 1326 | — | — | 是 | c33, c29 | 0.92(models/iter567_seed3_alpha10_purecal,贪婪重评) |
| 1368 | — | — | 是 | c34, c32 | 0.92(最终提交产物的收尾全量评测) |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[472, 908, 1113]
- **分类学缺口提案 2 条**
  - weight_row_surgery(i=770, i=772, i=775, i=792)
  - variance_replicate(i=908, i=1113, i=910, i=1082)
- **定义缺陷 3 条**
  - (i=23, i=185, i=1368, i=1372)
  - (i=1330, i=1337, i=1318, i=1108)
  - (i=1337, i=1350, i=1354)
- **边界情形 4 条**
  - i=472 被机械层记为 kind=real / 结局 returned / 时长 0.01h,会作为『一次真实训练启动』进入分母并被要求填 tested_variable。但它是 OOM 恢复的第一次尝试,Trainer 检出 batch-size 元数据变化后直接退出,**一次参数更新都没做**。它不是冒烟(没有缩小规模的意图),也不是被后续启动作废(它自己正常退出),四个取值没有一个描述它 …(i=472, i=476)
  - --seed 在 train_lora.py 的命令行上与 lr/bs/epochs 并列,按字面是 C4 超参;但语料文件与全部其它超参逐字相同时,这次训练测的是 run 间方差,不是方法。判 C4 会把约 1.08h(本 run 训练墙钟的约 19%)错误地算进『用于验证超参的时间』;判 C3 更错(语料是同一个文件);判 both 则两边都错。现定义无法裁决。(i=908, i=1113, i=910)
  - reference §3 说 C8『顺手改掉 C4 的取值』。本 run 比那更麻烦:i=472 的 OOM 修复把 bs 4→2 / accum 4→8(有效 batch 不变),i=477 又改回 4/4 并改用 gradient checkpointing;但**下一个实验 i=578(iter3)继承了 bs2/accum8**。于是 iter3 相对 iter2 的 microbatch…(i=472, i=578, i=468)
  - i=1285 与 i=1368 的被评模型都是 `final_model`,但中间经 `mv` + `cp -a` 把目录内容从两 seed soup 换成了三 seed soup。任何按『被评模型路径』聚合候选的机械口径,都会把两条不同产物的评测记成同一个候选的重复测量(0.89 与 0.92 会被读成噪声)。现有实体定义里『改动』与『被评产物』之间没有指向关系,判不了。(i=1285, i=1354, i=1368)

## codex_non_api_max_gpt-5.6-sol_10h_run1__bfcl_google_gemma-3-4b-pt_17396026
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | bfcl | google_gemma-3-4b-pt | 7.51h | 0.95 |

### 改动序列(31 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 44 | C2 | 把训练样本用**未改动的评测模板** templates/gemma3_tool_calling.jinja 渲染,并逐 token 校验 prompt-only 渲染是完整渲染的前缀(prefix 校验),使训练侧与评测侧的 prompt 逐字节一致。 | i=44, i=43, i=46 |
| 59 | C1 | merge_lora.py 在每次导出 standalone 权重时写死贪婪解码:do_sample=False、temperature/top_k/top_p=None、num_beams=1、eos_token_id=[1,106]、pad_token_id=0。落盘后 do_sample 因等… | i=59, i=538, i=538, i=107 |
| 59 | C7 | 在 train_lora.py 里建立 held-out 验证集 + 每个 save-steps 算一次 completion loss,当作比官方评测便宜得多的门控判据(rank-128 的继续/停止就是按预先声明的 loss 判据做的)。全 run 反复证明它与 BFCL 反向漂移:更低的 ev… | i=239, i=422, i=455, i=865 |
| 61 | C3 | 建 data/xlam_bfcl_v1:从 minpeter/xlam-function-calling-60k-parsed 只取 exactly-one-call 的行,校验函数名在自带 schema 内、arguments 可解析,得 27,883 train / 568 validation… | i=61, i=62, i=62 |
| 140 | C8 | 把 base snapshot 的 processor + tokenizer 七个文件原样拷回三个已导出模型,并修 merge_lora.py 永久保留它们 —— 不做的话 Gemma3 多模态包装在 vLLM 里 EngineCore 直接起不来(i=126 那次评测因此零产出)。同时把重新序列… | i=140, i=129, i=131, i=138 |
| 173 | C11 | 自写脚本读 inspect_ai 官方日志,把评分器**自己的输出**转成可决策信号:每个 checkpoint 的正确题号集合,加一张错误分类表(wrong_function / missing_argument_keys / extra_argument_keys / value_type_mi… | i=173, i=174, i=147 |
| 182 | C3 | 加第二个数据源 ToolACE(minpeter/toolace-parsed 首轮单调用,4,995+96 条),先建 data/toolace_bfcl_v1 再与 xLAM 合并;第二阶段训练时把 ToolACE 重复列一遍做 ×2 过采样(约占 26%)。 | i=182, i=183, i=186, i=185 |
| 251 | C6 | 自写 interpolate_lora.py:把两个 LoRA adapter 按 rank 拼接后加权求和(exact concatenated-rank weighted sum),等价于两个 delta 的精确带权和,而不是因子矩阵的朴素平均。全 run 用它造了 12 个 soup 候选,权… | i=251, i=254, i=535, i=675 |
| 293 | C4 | 把 LoRA 容量从 rank 64 / alpha 128 抬到 rank 128 / alpha 256(可训参数 119.2M→238.4M),同时把 lr 从 2e-4 降到 1.5e-4、seed 换成 8173。 | i=293, i=293, i=445 |
| 321 | C10 | 去污染审计:对 BFCL 的 100 条 user query 做空白归一 + casefold + sha256,与 xLAM(60,000)、ToolACE(12,224)、Hermes 两个子集(1,893 / 12,540)逐条比对,重叠 0;随后又做了跨源去重(hermes vs xlam… | i=321, i=322, i=338 |
| 334 | C3 | 准备第三个数据源 Hermes(NousResearch/hermes-function-calling-v1 的 func_calling_singleturn + glaive_func_calling,3,699+64 条),先隔离不入训。 | i=334, i=335, i=347 |
| 342 | C4 | 给 train_lora.py 加 --stop-after-step:提前停但保留 --max-steps 作为 scheduler 视界,使「只跑 100 步的短课程」与「跑满 200 步」共享同一条学习率曲线。 | i=342, i=529, i=550 |
| 372 | C3 | 造 data/xlam_singletool_v2:只保留「可用工具恰好 1 个」的 xLAM 行(7,855 train / 168 val),把「参数抽取」从「干扰项选择」里分离出来;平均输入长度 328 token,与评测的约 298 token 更匹配。 | i=372, i=373, i=396 |
| 406 | C4 | relabel_arguments.py:同一批 7,855 条样本不动,只改 labels —— 只在 arguments 的 JSON 与 <end_of_turn> 上算 loss,函数名/键名/包裹标签全部 mask 掉。这是本 run 512-token 口径下唯一一次+2 点的改动(91… | i=406, i=407, i=574 |
| 514 | C11 | 把 logs/*.json 全量聚合成「模型 → accuracy」表,并成对算正确集合的 A-only/B-only/union/intersection,用互补性(union 到 95)而不是单点分数来挑下一步该往哪个方向插值。 | i=514, i=515, i=518 |
| 615 | C11 | 为剩下 7 道错题写了一个安全 AST 求值器(支持 BinOp/Lambda/List/Dict),把 BFCL 的 target 字符串解析成 (函数名, 参数字典) 再与模型输出逐项比对,结论是错题已经函数名对、键名对,只错在值(数值变换 / 列表 / 表达式)。 | i=615, i=616, i=625 |
| 628 | C3 | filter_hard_arguments.py:只保留「非拷贝」样本 —— 正确参数值不是 user query 里的字面片段(2,669 / 7,855 = 34%),想直接针对评测里剩下的 value-reasoning 失败。 | i=628, i=629, i=625 |
| 656 | C4 | relabel_values.py:把 loss 再收窄一档 —— 只在 JSON 的**参数值** token 上算 loss(平均 11.1 个 token/条,7,748 条),键名与结构也 mask 掉。 | i=656, i=662, i=654 |
| 779 | proposed:merge_export_nu… | 给 merge_lora.py 加 --dtype,用 float32 全精度合并并以 float32 落盘(17 GB),假设是 bf16 舍入让小权重 soup 表现不稳。同一 adapter、同一评测口径下 93%→91%。 | i=779, i=776, i=784 |
| 794 | proposed:merge_export_nu… | 第三条合并数值路径 accurate_bfloat16:高精度合并后再转回 bf16 落盘。同一 adapter 下 93%→89%,agent 据此判定 93% 是 bf16 直接合并这条路径特有的。 | i=794, i=802 |
| 803 | C3 | 单变量种子复现:与 i=70 的配方逐字相同(data/xlam_bfcl_v1、900 步、lr 2e-4、cosine、warmup .03、rank 64 / alpha 128),只把 seed 3407 换成 2027。因为 train_lora.py 里 data_seed=args.s… | i=803, i=803, i=239 |
| 866 | C4 | 反方向的容量实验:rank 32 / alpha 64(59.6M 可训参数,是 incumbent 的一半),其余与 i=70 逐字相同(同数据、lr 2e-4、cosine、warmup .03、seed 3407、900 步)。 | i=866, i=885 |
| 912 | C3 | 唯一一次干净的配方单变量对照:三源联合混合 data/xlam_toolace_hermes_v3(36,577 条)对 xLAM-only,超参与 i=70 逐字相同(900 步、lr 2e-4、cosine、warmup .03、rank 64 / alpha 128、seed 3407)。 | i=912, i=912, i=354 |
| 968 | C9 | 第一次冻结提交:把当时 512-token 口径最高的 xlam600_singletool_arguments_step100(93%)整目录拷成 final_model,并先 `test ! -e final_model` 防止覆盖。不改任何权重。 | i=968, i=967, i=979 |
| 1021 | C9 | 换提交:按 evaluate.py 的**原始默认口径**(max_connections=6, max_tokens=16000)重筛后,朴素 xLAM step-600 拿 94% 而 arg100 只有 92%,于是删掉 final_model 换成 models/xlam_v1_step60… | i=1021, i=1020, i=1009 |
| 1052 | C1 | 字段级差异(骨架够不到的那一项):补丁前 final_model/generation_config.json 的内容在 i=1024 那次 `cat` 的返回里逐字可见(i=1025):{bos_token_id:2, cache_implementation:"hybrid", eos_toke… | i=1052, i=1051, i=1025, i=1025, i=1055, i=1069, i=1069, i=1095 |
| 1090 | C1 | 把验证过的 temperature-0 服务端默认值固化进 merge_lora.py,让以后每次导出都自带确定性解码;同时更新 final_model 的 README 与 training_metadata。 | i=1090, i=1087, i=1166 |
| 1108 | C1 | 把同一个 temperature-0 补丁批量打到 7 个候选模型目录,目的是让重筛在同一解码下可比。补丁前的内容有直接证据:i=1102 的 for 循环把这 7 个目录的 generation_config.json 逐字打印出来(i=1103),全部是 temperature/top_k/to… | i=1108, i=1098, i=1103 |
| 1125 | C1 | 同样的 temperature-0 补丁再打到 5 个 soup 目录(w725/w74/w755/w80/w875),为在确定性解码下重扫插值权重区间做准备。 | i=1125, i=1124 |
| 1149 | C8 | 磁盘 100% 满(427G 用掉 423G,只剩 3.7G),删掉已作废的 17 GB fp32 导出腾空间,否则续训根本落不了盘。 | i=1149, i=1139, i=1148 |
| 1169 | C8 | 导出器再修一次:transformers 的 GenerationConfig.validate(strict=True) 拒绝 do_sample=False 同时带 temperature/top_k,导致 merge 直接抛错;改成先存合法的 transformers 配置、再把服务端需要的字… | i=1169, i=1163, i=1168 |

### 训练序列(19 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 65 | smoke | 0.01h | returned | **smoke** | baseline —— 5 步冒烟,只验多模态 Gemma 包装、adapter 目标模块、显存与吞吐能不能跑通,不产出可比分数。`tested_variable` 的四个取值都装不下「冒烟」这一种,填 unclear 属于取值域缺口而非证据不足。 |
| 70 | real | 0.66h | returned | **baseline** | baseline —— 第一次真实训练,同时确定了数据(xlam_bfcl_v1,27,883 条单调用)与方法(rank 64 / alpha 128、lr 2e-4、cosine、bs 8 × grad-accum 4、900 步 ≈ 1.03 epoch)。基线不检验任何单一变量,取值域装不下… |
| 186 | real | 0.31h | returned | **both** | 对 i=70:数据换成 xLAM + ToolACE×2(37,873 条,ToolACE 约占 26%)= C3;同时 lr 2e-4→5e-5、步数 900→400、从 checkpoint-600 的 adapter 续训而非从头 = C4。两类一起动,拆不开。 |
| 293 | real | 0.67h | returned | **C4** | 对 i=70:数据集文件完全相同(data/xlam_bfcl_v1),只动方法侧 —— rank 64→128、alpha 128→256、lr 2e-4→1.5e-4、save-steps 300→225。附带把 seed 3407→8173(即数据顺序也变了),这是一处混杂,但被测意图明确是容… |
| 456 | real | 0.10h | returned | **both** | 对 i=293:数据换成单工具子集 data/xlam_singletool_v2(7,855 条)= C3;同时 lr→2e-5、scheduler cosine→constant_with_warmup、warmup .05、步数 200、从 checkpoint-600 续训 = C4。 |
| 475 | real | 0.10h | returned | **C4** | 对 i=456:样本集合**没变** —— data/xlam_singletool_arguments_v3 与 data/xlam_singletool_v2 是同一批 7,855 条、同样的 input_ids(i=574 逐字确认 length 266 / source_id 4 相同),只… |
| 528 | — | — | — | **unclear** | **这不是一次训练启动**。命令是 `python scripts/train_lora.py --help \| sed -n '1,240p'`,返回的是 argparse 帮助文本,没有加载模型、没有优化步、没有产物。机械层把它记成 real 启动(时长 0.00h、产物 —),见 defin… |
| 550 | real | 0.06h | returned | **C3** | 对 i=475:唯一变量是 `--seed 3407→8173`(数据、lr 1e-5、scheduler、warmup、步数、adapter-init 全部逐字相同)。train_lora.py 里 `data_seed=args.seed`,所以换 seed 就是换数据顺序;按 spec §5.… |
| 570 | real | 0.06h | returned | **C4** | 对 i=475:唯一变量是 lr 1e-5→5e-6(seed 回到 3407、数据同、scheduler 同、步数同、adapter-init 同)。纯超参单变量。 |
| 631 | real | 0.06h | returned | **both** | 对 i=570:数据换成「非拷贝」难样本子集 data/xlam_arguments_hard_v4(2,669 条,i=628 造)= C3;起点 checkpoint 也从 xlam_lora_v1/ckpt-600(91%)换成 arguments_stage3/ckpt-100(93%),步… |
| 664 | real | 0.06h | returned | **both** | 对 i=631:样本集合从 2,669 条难子集换回全量单工具集的 value-relabel 版 data/xlam_singletool_values_v5(7,748 条)= C3;监督范围再收窄到只算参数**值** token = C4;lr 5e-6→2e-6 = C4。起点 checkp… |
| 803 | real | 0.66h | returned | **C3** | 对 i=70:配方逐字相同(data/xlam_bfcl_v1、900 步、lr 2e-4、cosine、warmup .03、rank 64 / alpha 128、bs 8 × 4 默认),唯一变量 seed 3407→2027 ⇒ 数据顺序。这是全 run 最干净的单变量对照之一。 |
| 853 | real | 0.06h | returned | **unclear** | 对 i=475:数据、lr 1e-5、scheduler、warmup、步数、seed 全部逐字相同,**唯一变量是起点 checkpoint**(xlam_lora_v1/ckpt-600 → xlam_lora_seed2027/ckpt-900)。受测的是「这套 argument 课程能不能迁… |
| 866 | real | 0.66h | returned | **C4** | 对 i=70:数据、lr 2e-4、cosine、warmup .03、seed 3407、900 步全部相同,唯一变量 rank 64→32 / alpha 128→64(可训参数 119.2M→59.6M)。纯容量单变量。 |
| 899 | real | 0.05h | returned | **unclear** | 对 i=475:唯一变量是起点 checkpoint(→ xlam_lora_r32/ckpt-600),数据与全部超参逐字相同。与 i=853 同一类边界:受测变量是 `--adapter-init`。 |
| 912 | real | 0.65h | returned | **C3** | 对 i=70:超参逐字相同(900 步、lr 2e-4、cosine、warmup .03、rank 64 / alpha 128、seed 3407),唯一变量是数据 —— xLAM-only(27,883)换成 xLAM+ToolACE+Hermes 联合混合(36,577)。本 run 唯一一… |
| 955 | real | 0.06h | returned | **unclear** | 对 i=475:唯一变量是起点 checkpoint(→ xlam_toolace_stage2/ckpt-200),数据与全部超参逐字相同。与 i=853 / i=899 同一类边界。 |
| 1149 | real | 0.12h | returned | **unclear** | 对 i=70:同一份数据、同一套超参、直接 `--resume-from-checkpoint checkpoints/xlam_lora_v1/checkpoint-600` 沿原 optimizer/scheduler 续跑到 750 步,只把 save-steps 300→25、save-to… |
| 1178 | real | 0.04h | returned | **unclear** | 对 i=1149:同数据同超参同起点,只把停止步从 750 改成 650(save-steps 也从 25 回到 300)。同样是 C5 意图的快照步数搜索,取值域装不下。 |

### 验证序列(60 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 13 | 3.0 | 50.0 | 是 |  | 0.02 |
| 126 | — | — | 否 | c3 | 未拿到分数 —— vLLM EngineCore 启动失败(Gemma3 多模态包装缺 preprocessor_con… |
| 143 | — | — | 是 | c1, c2, c3, c4 | 0.89 |
| 148 | — | — | 是 | c1, c2, c3, c4 | 0.91 |
| 153 | — | — | 是 | c1, c2, c3, c4 | 0.87 |
| 203 | — | — | 是 | c5 | 0.91 |
| 209 | — | — | 是 | c5 | 0.87 |
| 259 | — | — | 是 | c6 | 0.90 |
| 278 | — | — | 是 | c6 | 0.90 |
| 281 | — | — | 是 | c6 | 0.87 |
| 450 | — | — | 是 | c28 | 0.88 |
| 492 | — | — | 是 | c9 | 0.90 |
| 495 | — | — | 是 | c9 | 0.91 |
| 499 | — | — | 是 | c9, c10 | 0.93 |
| 502 | — | — | 是 | c9, c10 | 0.91 |
| 545 | — | — | 是 | c6 | 0.90 |
| 561 | — | — | 是 |  | 0.90 —— 判的是 i=550 那次换种子的训练,不对应 changes 表里的条目 |
| 581 | — | — | 是 |  | 0.89 —— 判的是 i=570 的 lr 5e-6 训练 |
| 640 | — | — | 是 | c11 | 0.90 |
| 645 | — | — | 是 | c11 | 0.87 |
| 669 | — | — | 是 | c12 | 0.91 |
| 680 | — | — | 是 | c6 | 0.93 |
| 690 | — | — | 是 | c6 | 0.92 |
| 699 | — | — | 是 | c6 | 0.93 |
| 712 | — | — | 是 | c6 | 0.90 |
| 724 | — | — | 是 | c6 | 0.89 |
| 734 | — | — | 是 | c6 | 0.91 |
| 744 | — | — | 是 | c6 | 0.91 |
| 754 | — | — | 是 | c6 | 0.92 |
| 767 | — | — | 是 | c6 | 0.87 |
| 786 | — | — | 是 | c13 | 0.91 |
| 797 | — | — | 是 | c14 | 0.89 |
| 839 | — | — | 是 | c30 | 0.88 |
| 845 | — | — | 是 | c30 | 0.91 |
| 860 | — | — | 是 | c10 | 0.89 |
| 889 | — | — | 是 | c29 | 0.91 |
| 893 | — | — | 是 | c29 | 0.91 |
| 905 | — | — | 是 | c10 | 0.91 |
| 945 | — | — | 是 | c31 | 0.90 |
| 949 | — | — | 是 | c31 | 0.88 |
| 962 | — | — | 是 | c10 | 0.86 |
| 987 | — | — | 是 | c15 | 0.89 |
| 995 | — | — | 是 | c15 | 0.92 |
| 1000 | — | — | 是 | c6 | 0.90 |
| 1004 | — | — | 是 | c2 | 0.94 |
| 1010 | — | — | 是 | c5 | 0.88 |
| 1015 | — | — | 是 | c29 | 0.90 |
| 1029 | — | — | 是 | c16 | 0.93 |
| 1037 | — | — | 是 | c16 | 0.91 |
| 1042 | — | — | 是 | c10 | 0.89 |
| 1057 | — | — | 是 | c17 | 0.95 |
| 1079 | — | — | 是 | c17 | 0.95 |
| 1084 | — | — | — |  | 0.95 —— **不是一次评测**。该事件只是 `pgrep` 加回读 runs/final_model_temp0_… |
| 1110 | — | — | 是 | c18 | 0.92 |
| 1116 | — | — | 是 | c18 | 0.92 |
| 1120 | — | — | 是 | c18, c6 | 0.95 |
| 1127 | — | — | 是 | c19, c6 | 0.95 |
| 1130 | — | — | 是 | c19, c6 | 0.95 |
| 1174 | — | — | 是 |  | 0.94 —— 判的是 i=1149 那次续训到 750 步 |
| 1183 | — | — | 是 |  | 0.95 —— 判的是 i=1178 那次续训到 650 步 |

### 异常与存疑

- **6 段训练的受测变量判不出**:i=[528, 853, 899, 955, 1149, 1178]
- **2 次验证没有拿到信号**:i=[126, 1084]
- **分类学缺口提案 1 条**
  - merge_export_numerics(i=776, i=779, i=794, i=802, i=784)
- **定义缺陷 5 条**
  - (i=528, i=529, i=529)
  - (i=1084, i=1085, i=1085, i=1079)
  - (i=186, i=186, i=192)
  - (i=675, i=739, i=749, i=535)
  - (i=1121, i=1128, i=1131, i=1081, i=1134)
- **边界情形 6 条**
  - (i=853, i=899, i=955, i=475)
  - (i=574, i=574, i=407, i=475)
  - (i=1149, i=1178, i=1161)
  - (i=239, i=550, i=803, i=569)
  - (i=1108, i=1125, i=1098)
  - (i=422, i=455, i=844, i=898)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gpqamain_HuggingFaceTB_SmolLM3-3B-Base_17398714
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gpqamain | HuggingFaceTB_SmolLM3-3B-Base | 5.47h | 0.296875 |

### 改动序列(44 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 77 | C11 | 写 analyze_eval.py:把 inspect_ai 日志转成确定性决策信号(parse_rate、think_closed_rate、response_chars、答案字母分布),此后几乎每次评测后都被调用 | i=77, i=138 |
| 79 | C10 | prepare_data.py 内置 GPQA 去污染门:对全部 448 条测试题做 8-word 问题 ngram 与 >=6-word 答案短语匹配,命中即丢弃;每份语料的 manifest 都记 gpqa_contamination 计数 | i=79, i=104, i=104 |
| 97 | C3 | 第一份真实语料 data/v1:smoltalk2 的 Mixture_of_Thoughts_science 抽 12000 条 MCQ,seq_len 1536,四个答案字母近乎等量(2996/2997/2999/2998) | i=97, i=99 |
| 106 | C8 | OOM 修复:加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 并把 batch 8→6(步数 180→240 保持 token 预算);数据与 lr 不变 | i=106, i=105, i=104 |
| 130 | C1 | EOS 元数据修正(字段级 diff 由前后两次读推出):generation_config.json 与 config.json 的 eos_token_id 128001→128012,special_tokens_map.json / tokenizer_config.json 的 eos_t… | i=130, i=126, i=129, i=400, i=677 |
| 132 | C1 | 把同一处 EOS 修正写进 train_sft.py 的保存路径,使之后每个 checkpoint 保存时就带 128012,而不是每次另存后再手工补 | i=132, i=400, i=677 |
| 149 | C3 | data/v2 换配方:在 science_mcq 16000 之外加 medmcqa 8000 + megascience 8000 + medical-reasoning 4000,seq_len 1536→2048 | i=149, i=165 |
| 181 | C3 | data/v3 换来源结构:megascience 24000 只留 biology,physics,chemistry,science_mcq 降到 8000 且跳过前 16000 条,去掉 medmcqa/medical-reasoning | i=181, i=188 |
| 200 | C6 | 自写 merge_models.py,把 runs/v1a 与 base 模型按 90/10 线性插值成 runs/v1a_base10(不训练,几分钟) | i=171, i=200, i=207 |
| 216 | C3 | data/v4:与 v1 同源同格式,但 --science-skip 24000 取不相交切片,把'是不是只要更多同类数据'与'换来源'分开问 | i=216, i=242 |
| 322 | C3 | prepare_data.py 加 --science-permutations:同一道题生成四份,正确答案分别落在 A/B/C/D,用来对齐评测器每次独立重洗四个选项的行为 | i=322, i=324, i=327 |
| 341 | C8 | 给 runs/v4a 的四个 checkpoint 目录符号链接进 tokenizer.json / tokenizer_config.json / special_tokens_map.json,否则 vLLM 起不来 | i=341, i=341 |
| 404 | C1 | 显式解码配置实验:用符号链接复制 runs/v1a 的权重(逐字节相同),只在各自目录写不同的 generation_config.json —— greedy(temperature 0.0)、temp03(0.3/top_p 0.95)、temp06(0.6/top_p 0.95);原始 v1a… | i=404, i=407, i=429, i=403 |
| 484 | proposed:data-integrity-… | 自审发现排列语料的 rationale 里还残留原始选项字母(**B: No**、(C, D)、'- **A (alpha)**'),与重排后的正确答案自相矛盾;扩展 strip_option_labels 把它们清掉,并作废已跑的训练重建语料 | i=479, i=481, i=486, i=473 |
| 532 | proposed:data-integrity-… | 第二次自审发现打包层捷径:同一道题的 A/B 与 C/D 排列会落进同一条因果序列,后面的排列能注意到前面的答案;改成四遍分开生成,同一序列内只有不同题目 | i=532, i=526, i=572 |
| 601 | C5 | 用 cp -al 把 checkpoint-400 硬链接另存成 runs/vperm_ck400 并删掉 optimizer/scheduler,防止 checkpoint 轮转把中途步数删掉;checkpoint-600 同样处理 | i=601, i=604, i=617 |
| 693 | C1 | runs/vperm_a_rep105:权重符号链接自 vperm_a,只换 generation_config.json 加重复惩罚(目录名 rep105,具体字段值未在流里回显) | i=691, i=693, i=681 |
| 701 | C4 | train_sft.py 加 --eos-loss-weight:把 chat-end token 的 loss 权重放大,目标函数改动 | i=701, i=716, i=977 |
| 742 | C1 | runs/vperm_a_dual:权重符号链接自 vperm_a,generation_config.json 改成同时接受 128012 与 base 的 128001 两个终止 token | i=740, i=742, i=739 |
| 761 | C2 | prepare_data.py 加 --science-answer-only:completion 只留一个空 think 块加 'ANSWER: X',让每条训练轨迹都在几个 token 内到达终止符,与评测器实际解析的形式对齐 | i=761, i=766, i=771 |
| 808 | C8 | 给 runs/direct_a 的三个 checkpoint 补 tokenizer 符号链接,修掉 i=802 那次 vLLM 起不来导致整次评测无分数 | i=808, i=810 |
| 828 | C1 | runs/direct_a_greedy:权重符号链接自 direct_a,只写 temperature 0.0 的 generation_config.json | i=826, i=828 |
| 837 | C1 | runs/direct_a_temp06 并随后把 temperature 0.6 / top_p 0.95(SmolLM3 官方推理档)锁定成默认解码配方,之后每个候选都建一个 _temp06 包装目录 | i=835, i=837, i=842, i=985 |
| 850 | C2 | prepare_data.py 加 --science-semantic-answer:先抄一遍被选选项的原文再给字母,把'选对科学答案'与'映射到被打乱的 A–D 位置'拆开 | i=848, i=853, i=855 |
| 884 | C5 | checkpoint 选择:direct_broad_a 的 200 步 gate(36%)优于 600 步终点(29%)与 400 步(28%),把 checkpoint-200 立为 incumbent | i=884, i=887, i=894 |
| 934 | C8 | 清磁盘:一次删掉七个已否决的 32–45 GB checkpoint 目录(此前 i=512、之后 i=1196 各做过一次),否则 427 GB 盘装不下后续训练 | i=934, i=931 |
| 1017 | C2 | prepare_data.py 加 --science-compact-answer:把 completion 压到 'ANSWER: X.' 并以句点收尾,使终止有一个可被解码配置抓住的字面 token | i=1017, i=1019, i=1015 |
| 1072 | C1 | *_period 包装目录:从 Trainer checkpoint 复制 config.json / generation_config.json 后改写 —— eos_token_id 由标量 128012 改成 [128012, 13](13 就是句点),并补上 do_sample/tempe… | i=1069, i=1072, i=990, i=1215 |
| 1153 | C1 | compact_bal_ck25_period 的 greedy 与 temp03 两个解码变体,权重符号链接不变 | i=1150, i=1153 |
| 1187 | C1 | short_ck*_semicolon:短 rationale 格式以分号收尾,于是把 eos_token_id 设成 [128012, 26](26 是分号) | i=1185, i=1187, i=1215 |
| 1219 | C1 | 把分号 EOS 换回句点 EOS(runs/short_ck10_period),因为多数回答仍以句点结尾,分号-only 停止会漏掉一部分而放任 512-token 重复级联 | i=1217, i=1219, i=1213 |
| 1250 | C4 | focus_answer_mask.py:从 data/vcompact_tail_perm4 派生出 input_ids 逐字节相同、只把 loss_mask 缩到答案字母那一个 token 的语料(loss_tokens 79916 = 每条一个 token) | i=1248, i=1250, i=1253, i=1255 |
| 1271 | C5 | checkpoint 扫描工装:把已验证的 compact_bal_ck25_period 配置目录整份复制,再把权重换成 answer_focus_a 各步的符号链接,保证扫描 10/20/30/40 步时解码配置与 tokenizer 逐字相同 | i=1271, i=1272 |
| 1285 | C1 | temperature 扫描补两点:compact_bal_ck25_period 的 temp08 与 temp10 变体(权重 cp -a 后只改 generation_config.json) | i=1283, i=1285, i=1289 |
| 1325 | C4 | train_sft.py 加 --answer-loss-weight:保留全部格式 token 的 loss 作锚,只给答案字母加权(对上一次'只留答案字母'的失败做的修正) | i=1325, i=1331, i=1332 |
| 1360 | C11 | 验证器工装:import evaluate 后把 DEFAULT_EPOCHS 设成 3,同一份权重在一次调用里跑三次独立洗牌(1344 次呈现)取均值,把 448 题的 stderr 从 ~0.022 压到 ~0.017 —— 这是这条 run 后半段唯一能分辨候选的判据 | i=1360, i=1361, i=1483 |
| 1391 | C3 | data/vpublic_stem_compact 换来源:不再用 smoltalk2,改成 MedMCQA 40000(10000 题 × 4 个位置)+ MMLU 学科分卷的 STEM 14000,同样过 GPQA 去污染门(排除 28 条疑似重叠) | i=1391, i=1396 |
| 1443 | C6 | 权重平均:merge_models.py 把 compact_bal_ck25_period 与 weight_ck10_period 按 50/50 合成 runs/soup_parent_weight50,理由写明是降方差而不是因为它测得最高 | i=1443, i=1452 |
| 1449 | C1 | 补回被合并工具吃掉的字段:merge_models.py 从模型对象重新生成 config,把 eos_token_id 由 [128012, 13] 塌回标量 128012(句点终止符没了);agent 先 rg 了一遍才发现,随即改回列表形式 | i=1446, i=1447, i=1449, i=1479 |
| 1463 | C1 | soup 的窄温度扫描:temp05 与 temp07 两个包装目录(config 从 compact_bal_ck25_period 复制,权重符号链接到 soup),只改 temperature | i=1461, i=1461, i=1463 |
| 1478 | C9 | 提交守卫:test ! -e final_model 才做 cp -a(不无条件覆盖),并把权重复制成实体文件而非符号链接 | i=1478, i=1481 |
| 1486 | C10 | 给 final_model 写 README.md 与 training_manifest.json,记血统与去污染口径(只从指定 base 训练、GPQA 仅用于评测与污染过滤) | i=1486, i=1566 |
| 1488 | C9 | 提交前校验:manifest JSON 可解析、find -type l 有命中就 exit 1、本地 AutoModel 加载核对参数量 3075098624 / eos [128012,13] / do_sample+0.6+0.95,最后 sha256 比对复制出的两个分片与 soup 逐字节… | i=1488, i=1489, i=1497 |
| 1538 | proposed:posthoc-head-su… | 对已训好的 final_model 直接做输出头手术:先读源码确认这版 vLLM 的 get_diff_sampling_param 不支持持久 logit_bias,于是解开 tie_word_embeddings、把 A/B/C/D 四个 token 的 lm_head 行各加一个常数 logi… | i=1509, i=1536, i=1538, i=1324, i=1542 |

### 训练序列(25 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 93 | smoke | 0.01h | returned | **smoke** | baseline / 冒烟。--max-steps 2 走通管线,train_runtime 13.2 秒。受测变量取值域里没有 smoke 这一项,只能填 unclear —— 这是取值域装不下,不是证据不足。 |
| 103 | real | 0.01h | returned | **baseline** | baseline(第一次真实训练)。相对冒烟换成 data/v1、180 步、bs 8、lr 2e-5。跑 0 步就 CUDA OOM,exit 1。首次训练无可比对象,取值域里也没有 baseline 这一项。 |
| 106 | real | 0.08h | returned | **unclear** | 仍是 baseline。相对 i=103 只做了 C8 运行时补偿:expandable_segments + bs 8→6 + 步数 180→240 保 token 预算,数据与 lr 逐字不变。按 reference §3 C8 的规则,这类被迫的超参变动不计入受测变量。完成,train_run… |
| 147 | real | 0.09h | returned | **C4** | 本条 run 唯一一次干净的单变量训练:数据仍是 data/v1(逐字相同),从 runs/v1a 继续,只改 lr 2e-5→1e-5、步数 240→300、seed 43。agent 自己说是'在同一份均衡语料上做低学习率续训,测单纯多看几遍有没有用'。结果 50 题 38%→36%,被否决。 |
| 166 | real | 0.12h | returned | **both** | 数据 v1→v2(加 medmcqa/megascience/medical-reasoning,seq 1536→2048)与超参(lr 1e-5→8e-6、bs 6→4、accum 2→3)同时改。 |
| 193 | real | 0.00h | returned | **both** | 未发生训练:命令写成不存在的 --data-dir,argparse 直接 exit 2。意图是数据 v2→v3(纯 core-science)且 lr 8e-6→3e-6,两者同时改。 |
| 197 | real | 0.11h | returned | **both** | 把 --data-dir 改成 --data 重启(纯 C8 语法修复)。相对 v2a 仍是数据 v2→v3 与 lr 8e-6→3e-6 同时改。完成 358.4 秒,50 题 26%,被否决。 |
| 259 | real | 0.01h | returned | **both** | 数据 v3→v4(同源不相交切片,science-skip 24000)与超参(lr 3e-6→5e-6、bs 4→6、accum 3→2、步数 250→300)同时改。跑到第 3 步 OOM,exit 1 —— 与残留的评测显存撞车。 |
| 294 | real | 0.10h | returned | **both** | i=259 的 OOM 重启:bs 6→5、步数 300→360(保 token 预算),数据与 lr 逐字不变。受测变量沿用 i=259 的(数据切片 + 学习率同时动)。完成 338.7 秒。 |
| 465 | real | 0.02h | returned | **both** | 数据 v4→vperm(首次 --science-permutations 4)与超参(lr 5e-6→8e-6、步数 360→850、bs 6→5)同时改,并从 runs/v1a 重新分支。跑到 66/850 步被 agent 用 KeyboardInterrupt 主动杀掉 —— 数据审计发现 … |
| 495 | real | 0.08h | returned | **both** | 与 i=465 同一实验,数据用修好 strip_option_labels 后重建的 vperm,步数 850→830,其余逐字相同。跑到 286/830 步再次被 KeyboardInterrupt —— 第二次审计发现同题排列在打包时互相泄漏。 |
| 542 | real | 0.23h | returned | **both** | 第三次启动同一实验:语料再次重建(四遍分开打包),超参与 i=465 逐字相同(bs 5 / accum 2 / lr 8e-6 / seed 424242 / 850 步)。跑满 1 epoch,train_runtime 795.7 秒。相对上一个完成的训练(v4a)仍是数据与 lr 同时变。 |
| 716 | real | 0.11h | returned | **both** | 从 runs/vperm_a 继续。方法侧:首次 --eos-loss-weight 20,lr 8e-6→4e-6、步数 850→400。数据侧:vperm→vperm1(同源同配方,但不跳过前 12000 条、题量 16000→48000)。两侧都动。 |
| 775 | real | 0.16h | returned | **both** | 从 runs/vperm_a 分支。数据 vperm1→vdirect:同源同题量,但 completion 换成空 think 块 + ANSWER: X(--science-answer-only),seq 1536→1024。同时 eos-loss-weight 回到默认 1、lr 4e-6→… |
| 843 | real | 0.16h | returned | **both** | 从 runs/direct_a 继续。数据 vdirect→vdirect_broad(同格式,--science-skip 36000 的不相交 30000 题,排列数从 4 降到 1)、lr 6e-6→3e-6;bs/accum/步数/save-steps 不变。 |
| 895 | real | 0.11h | returned | **both** | 从 runs/direct_broad_ck200_temp06 分支。数据换成 vsemantic(--science-semantic-answer:先抄选项原文再给字母)、lr 3e-6→4e-6、步数 600→400、save-steps 200→100。agent 自称这是 'one ad… |
| 973 | — | — | — | **unclear** | 不是训练启动:命令就是 `python train_sft.py --help`,输出只有 argparse 用法表,没有 dataset 行、没有 loss、没有 train_runtime。机械层把它记成一次 real 训练(0.00h / returned)。 |
| 1052 | real | 0.08h | returned | **both** | 从 runs/direct_broad_ck200_temp06 分支。数据 vsemantic→vcompact_tail(--science-compact-answer,以句点收尾;--science-skip 66000 的新切片,排列 1,30000 题)、lr 4e-6→3e-6、步数 … |
| 1111 | real | 0.04h | returned | **both** | 从 runs/compact_tail_ck100_period 继续。数据 vcompact_tail→vcompact_tail_perm4(同格式同切片,排列 1→4、题量 30000→79940,四个字母各 19979 条完全等量)、lr 3e-6→1e-6、步数 300→100、warmu… |
| 1140 | real | 0.02h | returned | **both** | 从 runs/compact_bal_ck25_period 分支。数据换成 vcompact_gap(--science-skip 12000 的另一段题,排列 1,24000 题)、lr 1e-6→5e-7、步数 100→30、warmup 0.05→0.1。agent 自述是'新数据'实验,超… |
| 1182 | real | 0.02h | returned | **C3** | 本条 run 里最接近受控的一次数据侧对照:与 i=1140 同一父 checkpoint(compact_bal_ck25_period)、bs 8 / accum 2 / lr 5e-7 / warmup 0.1 / wd 0.1 / 30 步 / save-steps 10 全部逐字相同,只有… |
| 1254 | real | 0.00h | returned | **C4** | 未发生训练:--model 指向裸 Trainer checkpoint runs/compact_bal_a/checkpoint-25,缺 tokenizer 文件,加载阶段 AttributeError 退出。意图是换 loss mask(见 i=1258)。 |
| 1258 | real | 0.04h | returned | **C4** | i=1254 的 C8 修复重启(--model 改成自带 tokenizer 的 compact_bal_ck25_period)。受测的是目标函数:语料 vcompact_tail_perm4_answer 的 input_ids 与 vcompact_tail_perm4 逐字节相同,只有 l… |
| 1331 | real | 0.04h | returned | **C4** | 干净的方法侧改动:语料回到 data/vcompact_tail_perm4(与父训练 i=1111 逐字相同),只改目标函数与超参 —— 新增 --answer-loss-weight 4(保留 22 个格式 token 的 loss 作锚)、lr 1e-6→1e-7、accum 2→4、wd 0… |
| 1397 | real | 0.04h | returned | **both** | 从 runs/compact_bal_ck25_period 分支。数据换来源(smoltalk2 → MedMCQA 40000 + MMLU-STEM 14000,共 53952 条),同时去掉 --answer-loss-weight、lr 1e-7→5e-8、accum 4→8、步数 40→… |

### 验证序列(62 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 13 | 3.0 | 20.0 | 是 |  | 0.1 —— base 模型 20 题基线,accuracy 在 baseline20.json 里回读 |
| 112 | 3.0 | 20.0 | 是 | c41 | 0.05 —— 比 base 还低,但 agent 读 analyze_eval 后判定是 EOS 元数据缺陷而非推理退… |
| 133 | 3.0 | 20.0 | 是 | c4, c5 | 0.35 —— 与 i=112 构成近受控 C1 对照:同一路径同一份权重、--limit 20 / --max-tok… |
| 141 | 3.0 | 50.0 | 是 | c4, c41 | 0.38 —— 换 50 题复核 35% 是否代表性 |
| 160 | 3.0 | 50.0 | 是 |  | 0.36 —— 判定 i=147 那次同数据低 lr 续训(C4),38%→36%,分支被否决 |
| 183 | 3.0 | 50.0 | 是 | c6 | 0.30 —— 判定 data/v2 混合配方,明显回退 |
| 247 | 3.0 | 50.0 | 是 | c7 | 0.32 —— 判定 90/10 与 base 的权重插值,低于 38%,否决 |
| 253 | 3.0 | 50.0 | 是 | c8 | 0.26 —— 判定 data/v3 纯 core-science 配方,否决 |
| 341 | 3.0 | 100.0 | 是 | c9, c11 | 0.04 —— v4a 的 100 步早期 checkpoint,几乎每条都撞 token 上限 |
| 365 | 3.0 | 20.0 | 是 | c9 | 0.45 —— 同一分支的终点 checkpoint 在 20 题上 45%,与上一行的 4% 完全相反 |
| 375 | 3.0 | 100.0 | 是 | c9 | 0.12 —— 同一份权重换成 100 题只有 12%,推翻上一行的 45%;agent 由此判定 20 题探针不可用 |
| 397 | 3.0 | 100.0 | 是 |  | 0.10 —— 把 incumbent v1a 也放到同样的 100 题上作对照,发现它同样有长尾失控,指向采样默认值而… |
| 415 | 3.0 | 100.0 | 是 | c12 | 0.01 —— 贪婪解码进入确定性循环 |
| 439 | 3.0 | 100.0 | 是 | c12 | 0.05 —— temperature 0.6 也救不了这一版权重 |
| 636 | 3.0 | 100.0 | 是 | c10, c13, c14 | 0.18 —— 判定排列语料 + 两次数据完整性修复后的整块训练;agent 读 analyze_eval 发现前半段解… |
| 694 | 3.0 | 100.0 | 是 | c16 | 0.10 —— 重复惩罚不但没改善终止(57/100 撞 cap),还压低了分数,弃用 |
| 733 | 3.0 | 100.0 | 是 | c17 | 0.15 —— eos-loss-weight 20 的 400 步 gate;agent 判定 EOS 加权没解决问题 |
| 743 | 3.0 | 100.0 | 是 | c18 | 0.06 —— 双 EOS 假设被推翻 |
| 802 | 3.0 | 100.0 | 否 | c19 | 未拿到 —— 不是'追不到分数',是这次评测根本没产出分数:vLLM server 在 uvloop.run(run_s… |
| 811 | 3.0 | 100.0 | 是 | c19, c20 | 0.29 —— 100% parse rate、零 cap、中位 15 token,第一个稳健候选 |
| 818 | 3.0 | 100.0 | 是 | c19 | 0.31 —— 600 步终点,略优于 200 步 |
| 821 | 3.0 | 100.0 | 是 | c19 | 0.24 —— 400 步反而最差,checkpoint 曲线非单调 |
| 829 | 3.0 | 100.0 | 是 | c42 | 0.26 —— 贪婪低于 temperature 1.0 的 31% |
| 838 | 3.0 | 100.0 | 是 | c43 | 0.33 —— temperature 0.6 / top_p 0.95 是当时 100 题最好,解码配方就此锁定 |
| 877 | 3.0 | 100.0 | 是 | c43 | 0.29 —— direct_broad 终点低于 33% 的 incumbent,不提升 |
| 884 | 3.0 | 100.0 | 是 | c22, c43 | 0.36 —— 200 步 gate 成为新 incumbent |
| 891 | 3.0 | 100.0 | 是 | c22, c43 | 0.28 —— 400 步跌回,确认峰值在 200 步 |
| 918 | 3.0 | 100.0 | 是 | c21 | 0.12 —— 语义 scratchpad 把失控行为带了回来(约 1500 token/条),整条分支被弃 |
| 937 | — | — | 是 | c22 | 0.36,但随即被 agent 自己作废:他发现 evaluate.py 省略 --limit 时缺省是 50 而不是全… |
| 960 | 3.0 | 448.0 | 是 | c22 | 0.071 —— 真正的 448 题全量。与同一份权重的 100 题 0.36 差 29 个点,原因是 72% 的输出撞… |
| 1074 | 3.0 | 150.0 | 是 | c23, c24 | 0.26 —— compact 格式 + 句点 EOS 的 100 步 gate,150 题四秒跑完、平均 9 个输出 … |
| 1084 | 3.0 | 150.0 | 是 | c23, c24 | 0.073 —— 200 步过拟合,失控回来 |
| 1089 | 3.0 | 448.0 | 是 | c23, c24 | 0.306 —— 第一次全量可信数字,100% 解析、总输出 4032 token |
| 1120 | 3.0 | 150.0 | 是 | c10, c24 | 0.20 —— 150 题上低于 incumbent 的 26%,但 agent 没据此否决,而是给了一次全量复核 |
| 1125 | 3.0 | 448.0 | 是 | c10, c24 | 0.33 —— 全量推翻 150 题结论(20%→33.0%),立为新 incumbent。这是短切片判据失效的直接证据 |
| 1135 | 3.0 | 448.0 | 是 | c24 | 0.25 —— 50 步越过过拟合边界,25 步保持 incumbent |
| 1148 | 3.0 | 448.0 | 是 | c24 | 0.286 —— 判定 i=1140 那次新切片续训,低于 33.0% |
| 1163 | 3.0 | 448.0 | 是 | c25 | 0.29 —— 贪婪解码 |
| 1188 | 3.0 | 150.0 | 是 | c26 | 0.347 —— 短 rationale + 分号 EOS 在 150 题上最高,但 agent 检查后发现多数回答以句… |
| 1221 | 3.0 | 448.0 | 是 | c27 | 0.295 —— 换回句点 EOS 后全量只有 29.5%,i=1188 的表面优势消失,该分支被否 |
| 1228 | 3.0 | 448.0 | 是 | c25 | 0.299 —— temperature 0.3 |
| 1274 | 3.0 | 448.0 | 是 | c28, c29 | 0.25 —— 只监督答案字母的 10 步就把固定输出前缀打散、解析失败,整条分支被否 |
| 1286 | 3.0 | 448.0 | 是 | c40 | 0.288 —— temperature 0.8,补完 greedy/0.3/0.6/0.8 四点曲线,固定 0.6 |
| 1335 | 3.0 | 448.0 | 是 | c30 | 0.353 —— answer-loss-weight 的 10 步 gate,单次全量最高分 |
| 1343 | 3.0 | 448.0 | 是 | c30 | 0.228 —— 20 步越界,约 27000 个多余输出 token |
| 1348 | 3.0 | 448.0 | 是 | c30 | 0.288 —— 同一份权重、同一条命令的第二次全量,从 0.353 掉到 0.288。agent 由此判定 35.3%… |
| 1352 | 3.0 | 448.0 | 是 |  | 0.317 —— 对 incumbent 做同样的重复全量(vs i=1125 的 0.33),两个候选两轮均值 32.… |
| 1360 | 3.0 | 448.0 | 是 | c31 | 0.297 —— incumbent 的三轮洗牌估计(1344 次呈现);机械层未登记 |
| 1364 | 3.0 | 448.0 | 是 | c30, c31 | 0.301 —— weight_ck10 的三轮估计,与 incumbent 的五轮合计 30.8% vs 30.9% … |
| 1401 | 3.0 | 448.0 | 是 | c32 | 0.297 —— public-STEM 的 5 步 gate |
| 1409 | 3.0 | 448.0 | 是 | c32 | 0.333 —— 10 步 gate |
| 1413 | 3.0 | 448.0 | 是 | c32 | 0.333 —— 15 步同分,看起来是平台而非尖峰 |
| 1417 | 3.0 | 448.0 | 是 | c32 | 0.228 —— 20 步越界 |
| 1421 | 3.0 | 448.0 | 是 | c32, c31 | 0.086 —— public-STEM 10 步:单轮 448 是 0.333,三轮洗牌只有 8.6%(后面的采样轮触… |
| 1436 | 3.0 | 448.0 | 是 | c32 | 0.268 —— public-STEM 的贪婪解码复核,整条分支否决 |
| 1451 | 3.0 | 448.0 | 是 | c33, c31 | 0.322 —— soup 的三轮估计,超过两个父候选;机械层未登记 |
| 1456 | 3.0 | 448.0 | 是 | c33 | 0.322 —— soup 的重复三轮,逐位相同,六轮合计 32.2%、2688/2688 全部紧凑输出,成为最终 in… |
| 1465 | 3.0 | 448.0 | 是 | c44 | 0.308 —— temperature 0.5,低于 0.6;机械层未登记 |
| 1469 | 3.0 | 448.0 | 否 | c44 | 未拿到 —— agent 看到长尾变慢就判定已进入 runaway 区,在跑完前主动 Ctrl-C 掉,没有分数回来;机… |
| 1492 | 3.0 | 448.0 | 是 | c33, c34, c35, c37 | 0.31026785714285715 —— 对提交目录本身、用评测器默认 16000 token 预算的端到端复核,1… |
| 1544 | 3.0 | 448.0 | 是 | c38 | 0.329 —— 校准头三轮 32.9%,预测分布从 17%A/35%D 改善到 21%A/32%D;机械层未登记 |
| 1555 | 3.0 | 448.0 | 是 | c38 | 0.295 —— 同一份校准头的重复三轮跌到 29.5%,六轮合计 31.2% < 未校准的 32.2%,被否决,fin… |

### 异常与存疑

- **2 段训练的受测变量判不出**:i=[106, 973]
- **2 次验证没有拿到信号**:i=[802, 1469]
- **分类学缺口提案 2 条**
  - data-integrity-audit-fix(i=473, i=486, i=526, i=572)
  - posthoc-head-surgery(i=1324, i=1538, i=1542, i=1542)
- **定义缺陷 5 条**
  - (i=973, i=977, i=942)
  - (i=1451, i=1453, i=1426, i=1460)
  - (i=1335, i=1348, i=1351, i=1483)
  - (i=1182, i=1140, i=1175, i=1127)
  - (i=803, i=806, i=810, i=811)
- **边界情形 3 条**
  - (i=1250, i=1255, i=1255, i=1331)
  - (i=327, i=313, i=333, i=1091)
  - (i=93, i=103, i=973)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gpqamain_Qwen_Qwen3-1.7B-Base_17398713
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gpqamain | Qwen_Qwen3-1.7B-Base | 6.05h | 0.2790178571… |

### 改动序列(55 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 103 | C2 | 把训练样本统一渲染成评测器原话的 MCQ 提示与 'ANSWER: $LETTER' 结尾合同(EVAL_MC_PROMPT 抄自 inspect_ai 的 SINGLE_ANSWER_TEMPLATE_COT) | i=103, i=275 |
| 106 | C3 | 首个 SFT 混合 data/sft_v1:Nemotron syn_mcq 60k + MegaScience 60k + MMLU-Pro 10k | i=106, i=108 |
| 121 | C3 | 重写 Nemotron MCQ 结构解析(容忍 A: / (A) / A. 等标签样式),bad_structure 40203->2、kept 55608->95614,重建为 data/sft_v3 | i=121, i=123, i=124 |
| 131 | C2 | 规范化 MCQ 选项标签与答案字母(canonical 重排),建 data/sft_v4:170,000 行里 100,177 条严格以 ANSWER: 结尾 | i=131, i=138, i=138 |
| 145 | C8 | 首训 OOM 后重启:micro-batch 8->4、grad-accum 8->16、加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,有效 batch/lr/步数刻意保持不变 | i=145, i=144, i=144 |
| 157 | C11 | 自写 scripts/eval_checkpoints.py:对一个 run 目录下全部 checkpoint 用同一 limit/max-tokens/并发扫分并落 summary_*.json | i=157, i=926 |
| 201 | C11 | 自写 inspect_ai 日志诊断:统计 ANSWER: 解析成功数、答案字母分布、</think> 闭合数、stop_reason 与输出长度(把官方评分器输出转成可决策信号) | i=201, i=202, i=202 |
| 211 | C3 | 建 data/concise_v1:只保留严格 MCQ 格式的 100,177 条,并把推理正文截到 900 字符,保证结尾一定是答案行 | i=211, i=212 |
| 215 | C5 | 从 sft_v1 的 150/300/450/600/680/final 六个候选里选 checkpoint-300(50 题探针 22%)作为下一阶段起点 | i=215, i=198 |
| 275 | C10 | 数据构建脚本刻意不加载 GPQA,并在每份 manifest 里写 gpqa_used=false;后续所有派生数据继承该字段 | i=275, i=212 |
| 302 | C1 | 手改 checkpoint 的 generation_config.json,把 <\|im_end\|>(151645)加进 eos_token_id,让评测器不再收到答案之后的垃圾续写 | i=302, i=291, i=270 |
| 313 | C1 | 写 scripts/configure_decoding.py 统一改写各 checkpoint 的解码配置(此后所有 C1 改动都走这个脚本) | i=313, i=315 |
| 327 | C3 | 写 build_balanced_data.py:把每道四选题循环置换到四个答案位,得 A/B/C/D 各 28,583 的 data/balanced4_v1(114,332 行) | i=327, i=335, i=326 |
| 332 | C8 | 修 build_balanced_data.py 的 KeyError: 'a'(小写标签未进映射表) | i=332, i=330 |
| 361 | C3 | 写 build_dpo_data.py:同一 rationale 换一个错字母做负例,得 32,000 条偏好对 data/dpo4_v1 | i=361, i=364, i=884 |
| 367 | C4 | 写 scripts/train_dpo.py:TRL DPO,sigmoid loss + rpo_alpha 0.1 + padding_free | i=367, i=369 |
| 373 | C11 | 写 scripts/summarize_inspect_log.py:从官方日志一次性给出 accuracy / parsed / answer_distribution / stop_reasons / 输出 token 分位 | i=373, i=375 |
| 387 | C1 | 发现 HF GenerationConfig 不置 do_sample=true 时 temperature/top_p/top_k 被判为无效标志,改脚本显式写 do_sample | i=387, i=385, i=389 |
| 495 | C1 | 给 configure_decoding.py 加 --eos-mode base\|chat、--temperature、--greedy 三个开关,把停止行为与采样温度拆成两个可单独测的维度 | i=495, i=494 |
| 497 | C1 | 回退 <\|im_end\|> 停止:concise_v1 改回 base EOS(只留 151643)+ temperature 0.6 | i=497, i=498 |
| 513 | C1 | temperature 0.6 -> 0.2(base EOS 不变) | i=513 |
| 520 | C1 | 贪婪解码对照(--greedy,base EOS) | i=520 |
| 526 | C5 | 选 concise_v1/checkpoint-411(100 题 24%,且 empty=0)作为 balanced 阶段起点,弃后段 375/final | i=526, i=483, i=486 |
| 561 | C1 | chat EOS + temperature 0.2 的复核变体 | i=561 |
| 567 | C5 | 选 balanced_v1/checkpoint-50(28%)作为 DPO 起点,弃 checkpoint-100(14%)与 final(22%) | i=567, i=559 |
| 613 | C11 | 把判定口径从 n=100 探针升到 448 全量,并对领先候选做重复全量取均值(理由:同一份 DPO-200 权重两次 100 题读到 26% 与 15%) | i=613, i=613, i=696 |
| 613 | C5 | 定三名决赛候选:balanced_v1/ckpt-50、dpo_v1/ckpt-100、concise_v1/ckpt-411,统一在 base-EOS + t=0.2 下比 | i=613 |
| 641 | C3 | 写 build_direct_data.py:把 balanced4_v1 的 completion 截成只剩最终答案行(同题同标签,只换监督目标),得 data/direct4_v1 | i=641, i=644 |
| 674 | C5 | 把 direct_v1/checkpoint-25(100 题 31%,全 run 探针最高)提为全量复核候选 | i=674, i=676 |
| 717 | C1 | max_new_tokens 2048 -> 4096 | i=717 |
| 723 | C5 | 把 sft_v1/checkpoint-600(更多科学 token 的更晚 checkpoint)重新提出来当候选,准备给它一条便宜的替代尾巴 | i=723 |
| 724 | C1 | 4096 探针未改善解析率,max_new_tokens 回退 2048 | i=724, i=723 |
| 761 | C5 | 定 dpo_v1/checkpoint-100(两次全量 27.68% / 29.02%)为提交候选,否掉 alt_bal600 分支 | i=761 |
| 773 | C9 | 把 runs/dpo_v1/checkpoint-100 复制成 final_model(首次落交付物,此前 final_model 不存在) | i=773, i=762 |
| 798 | C1 | dpo_min64:chat EOS + min_new_tokens 64,想在不动权重的前提下压掉立即 EOS 的空回答 | i=798, i=797 |
| 810 | C1 | dpo_min512:同一想法把 min_new_tokens 提到 512 | i=810, i=812 |
| 821 | C1 | dpo_prefill:改提交目录里的 chat_template.jinja,在 assistant 头之后预填 <think> 开头,再用 chat EOS 正常停 | i=821, i=818 |
| 836 | C1 | dpo_nothink:反方向的模板变体,推理时显式关掉隐藏推理块 | i=836, i=837 |
| 875 | C3 | 换新数据源 nvidia/OpenScience OS-Q3-235B-4(卡片声明对 GPQA 去污染),每个答案字母水塘抽样 30,000 条,建 data/openscience_v1 | i=875, i=856 |
| 886 | C3 | 用 OpenScience 造同构偏好对 data/open_dpo_v1(32,000 条,seed 20260721) | i=886, i=887 |
| 935 | C5 | 选 open_v1/checkpoint-100(27%)作为 OpenScience DPO 的起点,理由是 150 步掉到 23%、200 步没有增益 | i=935 |
| 966 | C1 | open100_chat:open_v1/checkpoint-100 的 chat-EOS 变体(测新阶段的格式训练是否让 chat 停止变可行) | i=966, i=965 |
| 976 | C8 | 发现 evaluate.py 省略 --limit 时默认只跑 50 题,补上 --limit 448 重跑,并声明不用那次误跑的结果做决策 | i=976, i=977 |
| 982 | C6 | 写 scripts/interpolate_checkpoints.py:(1-alpha)*base + alpha*other 的逐张量线性插值 | i=982, i=985 |
| 984 | C6 | 造 soup_open25 / soup_open50:把 open_v1/ckpt-100 以 0.25 / 0.50 权重并进 dpo_v1/ckpt-100 | i=984, i=991 |
| 1004 | C11 | 从两次全量的官方日志算 target x 预测 的混淆矩阵与字母分布,定位 D 位系统性欠预测(59-74 vs 100-122) | i=1004, i=1005 |
| 1011 | C11 | 统计评分器判为未解析的回答里有多少能从 reasoning/text 里回收出答案(57->11、46->8),据此判定剩下的失败是死循环而不是晚决策 | i=1011, i=1012, i=1015 |
| 1023 | C1 | dpo_concise_prefill:另一种 <think> 预填模板,这次配 base EOS(上一次预填用的是 chat EOS,组合没测过) | i=1023, i=1017 |
| 1036 | C1 | dpo_rep105 / dpo_rep110:在不动权重的副本上加 repetition_penalty 1.05 / 1.10 | i=1036, i=1036, i=1030 |
| 1071 | proposed:artifact_integr… | 写 final_model/README.md、final_model/training_manifest.json、metrics/final_summary.json:记录四段训练血统、解码策略与合规声明 | i=1071, i=1071 |
| 1140 | C1 | dpo_out4096:权重与 dpo ckpt-100 逐字节相同的副本,只把 max_new_tokens 改成 4096,做全量对照 | i=1140, i=1140, i=1139 |
| 1179 | C5 | 补测唯一一个和领先者打平却没做过全量的候选 dpo_v1/checkpoint-200 | i=1179 |
| 1212 | C8 | 容器里没有 pip / uv / pdftotext / pypdf,改用 curl 拉 arXiv e-print 再 tar -xzO 直接读 LaTeX 源码,以完成数据来源审计 | i=1212, i=1207, i=1209 |
| 1229 | C10 | 审计 Dr.SCI:原论文写了 13-gram benchmark 去污染,但可得的复现版流水线里没有这一步,据此拒绝把它用于训练 | i=1229, i=1220 |
| 1237 | proposed:artifact_integr… | 交付物自检:断言 final_model 的 config/eos/generation_config 逐字段符合预期、manifest 的 compliance 标志为假、权重与所选 checkpoint 逐字节相同、记 sha256 | i=1237, i=1237, i=1073, i=1044 |

### 训练序列(9 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 141 | real | 0.02h | returned | **baseline** | baseline —— 全 run 第一次训练,无上一次可比。取值域里没有 baseline 档,按纪律只能填 unclear;这不是证据不足。启动 75 秒后 OOM,没写出任何 checkpoint。 |
| 145 | real | 1.03h | returned | **unclear** | 与 i=141 同数据(sft_v4)、同 lr 1e-5、同 epochs、同 max-length 2048、同 save-steps 150;只把 micro-batch 8->4、grad-accum 8->16 并加 expandable_segments。agent 明说是为保持有效 b… |
| 224 | real | 0.48h | returned | **C3** | 换数据:sft_v4(170,000 行、完整推理)-> concise_v1(100,177 行、推理截 900 字符、一定以答案行收尾);起点从 base 换成 sft_v1/ckpt-300。同时 lr 1e-5->3e-6、max-length 2048->1536、save-steps 1… |
| 526 | real | 0.13h | returned | **C3** | 换数据:concise_v1 -> balanced4_v1(同一批题循环置换到四个答案位,A/B/C/D 各 28,583)。起点换成 concise_v1/ckpt-411;lr 3e-6->1e-6、新增 --max-steps 100、save-steps 75->50,都是「短对齐尾巴」的… |
| 568 | real | 0.25h | returned | **C4** | 换方法:SFT -> DPO/RPO。数据 dpo4_v1 是 balanced4_v1 的同题改造(同 rationale 换错字母做负例),不是新来源;bs 4->2、accum 16->32 保持有效 batch 64,是偏好对双份前向的显存补偿。受测的是偏好目标本身。 |
| 660 | real | 0.05h | returned | **both** | 与 i=568 同起点(balanced_v1/ckpt-50)的并行分支。data/direct4_v1 与 balanced4_v1 是同一批题、同一批标签(114,332 行 / A-D 各 28,583),唯一差别是 completion 被截成只剩最终答案行 —— 机械上是换数据集(C3)… |
| 731 | real | 0.07h | returned | **C4** | 全 run 最接近单变量的一次训练对照:与 i=526 用同一份 data/balanced4_v1、同 lr 1e-6、同 bs 4 / accum 16、同 max-length 1536,只换 --model(concise_v1/ckpt-411 -> sft_v1/ckpt-600),外加… |
| 880 | real | 0.23h | returned | **C3** | 换数据来源:自建的 Nemotron/MegaScience/MMLU-Pro 混合 -> nvidia/OpenScience OS-Q3-235B-4(120,000 行,每标签 30k)。lr 1e-6、bs 4 / accum 16、max-length 1536 与 i=526 逐字相同;… |
| 936 | real | 0.15h | returned | **C3** | 与 i=568 同方法(train_dpo.py)、同 lr 5e-7、同 bs 2 / accum 32、同 max-samples 16000,只换偏好数据来源(dpo4_v1 -> open_dpo_v1)与起点(balanced ckpt-50 -> open_v1 ckpt-100);步数… |

### 验证序列(34 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 10 | 3.0 | 50.0 | 是 |  | 0.08 |
| 207 | 3.0 | 50.0 | 是 | c9, c7 | 0.24 |
| 500 | 3.0 | 100.0 | 是 | c19, c20 | 0.22 |
| 513 | 3.0 | 100.0 | 是 | c21 | 0.25 |
| 520 | 3.0 | 100.0 | 是 | c22 | 0.16 |
| 561 | 3.0 | 100.0 | 是 | c24 | 0.15 |
| 614 | 4.0 | -1.0 | 是 | c13, c25, c27, c26 | 0.258928… |
| 638 | 4.0 | -1.0 | 是 | c15, c16, c25, c27, c26 | 0.276785… |
| 681 | 4.0 | -1.0 | 是 | c28, c29 | 0.254464… |
| 697 | 4.0 | -1.0 | 是 | c26 | 0.290178… |
| 717 | 3.0 | 100.0 | 是 | c30 | 0.27 |
| 724 | 3.0 | 100.0 | 是 | c32 | 0.24 |
| 751 | 3.0 | 100.0 | 是 | c32 | 0.23 |
| 757 | 3.0 | 100.0 | 是 | c32 | 0.17 |
| 802 | 3.0 | 100.0 | 是 | c35 | 0.19 |
| 811 | 3.0 | 100.0 | 是 | c36 | 0.25 |
| 825 | 3.0 | 100.0 | 是 | c37 | 0.21 |
| 838 | 3.0 | 100.0 | 是 | c38 | 0.17 |
| 928 | 3.0 | 100.0 | 是 | c39 | 0.22(该事件其实是 50/100/150/200 四次评测的 bash 循环,读到 0.22 / 0.27 / 0.… |
| 958 | 3.0 | 100.0 | 是 | c40 | 0.26(该事件其实是 25/50/75/100 四次评测的 bash 循环,读到 0.26 / 0.22 / 0.23… |
| 968 | 3.0 | 100.0 | 是 | c42 | 0.23 |
| 971 | — | — | 是 |  | 0.2(误跑:省略 --limit 默认只取 50 题,agent 明确声明不据此决策) |
| 976 | 3.0 | 448.0 | 是 | c43, c39 | 0.252 |
| 998 | 3.0 | 100.0 | 是 | c44, c45 | 0.16(该事件含 alpha 0.25 与 0.50 两次评测,读到 0.16 / 0.20) |
| 1025 | 3.0 | 100.0 | 是 | c37 | 0.22 |
| 1025 | 3.0 | 100.0 | 是 | c37 | 0.22 |
| 1025 | 3.0 | 100.0 | 是 | c48 | 0.24 |
| 1025 | 3.0 | 100.0 | 是 | c48 | 0.24 |
| 1037 | 3.0 | 100.0 | 是 | c49 | 0.26 |
| 1037 | 3.0 | 100.0 | 是 | c49 | 0.26 |
| 1037 | 3.0 | 100.0 | 是 | c49 | 0.2 |
| 1037 | 3.0 | 100.0 | 是 | c49 | 0.2 |
| 1077 | 3.0 | 50.0 | 是 | c34 | 0.12 |
| 1087 | 3.0 | 448.0 | 是 | c34 | 0.286 |
| 1097 | — | — | — |  | 未拿到 —— 这不是一次评测,是对 i=1087 那次评测的进程轮询(rg 计数 + ps/pgrep) |
| 1132 | 3.0 | 100.0 | 是 | c31 | 0.24 |
| 1142 | 3.0 | 448.0 | 是 | c51 | 0.27 |
| 1182 | 3.0 | 448.0 | 是 | c52 | 0.237 |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[145]
- **1 次验证没有拿到信号**:i=[1097]
- **分类学缺口提案 1 条**
  - artifact_integrity_check(i=1044, i=1073, i=1237, i=1237, i=1085)
- **定义缺陷 3 条**
  - (i=1097, i=1097, i=1087, i=1098)
  - (i=561, i=565, i=567, i=657)
  - (i=434, i=486, i=189, i=928, i=933)
- **边界情形 5 条**
  - (i=644, i=659, i=660)
  - (i=731, i=526, i=730)
  - (i=821, i=818, i=823)
  - (i=613, i=696, i=674)
  - (i=971, i=977, i=976)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gpqamain_Qwen_Qwen3-4B-Base_17397513
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gpqamain | Qwen_Qwen3-4B-Base | 8.59h | 0.3526785714… |

### 改动序列(34 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 19 | C2 | 格式对齐:先从 inspect_ai 里打印出评测器真实用的 SINGLE_ANSWER_TEMPLATE_COT,再在 prepare_sft.py 里以 EVAL_MC_PROMPT 逐字复用,把 MCQ 训练目标统一成 'ANSWER: $LETTER' 终行契约。 | i=19, i=613, i=1370 |
| 83 | C10 | 去污染准入规则:只用 pin 到 revision 的 TextbookReasoning / Nemotron syn_mcq,排除任何 provenance 里出现 GPQA 的混合集(即使声称已去污染);训练侧从不读 evaluate.py 的日志或 GPQA 缓存。 | i=83, i=1370 |
| 128 | C4 | SFT 方法:对 Qwen3-4B-Base 直接全参 BF16 训练,2,048 打包序列、response-only loss、Flash Attention 2、cosine 衰减、峰值 lr 3e-5、有效 batch 64。 | i=128, i=1370, i=306 |
| 130 | C3 | 建主 SFT 混合 data/sft_train.jsonl:MegaScience/TextbookReasoning 40,000 行(bio/chem/phys/med/math 平衡)+ nvidia Nemotron-SFT-Science-v2 syn_mcq 9,833 行四选一,共 … | i=130, i=132 |
| 138 | C3 | 修 prepare_sft.py 的 extract_letter 正则以吃下 Markdown 粗体形态(**Answer: B**),合格 MCQ 行从 parsed 3,739 提到 12,051(unparsed 8,512→200)。 | i=138, i=141, i=136 |
| 217 | C8 | 写 merge_adapter.py 把 PEFT adapter 合进稠密父模型:evaluate.py 无法直接加载 LoRA,不合并则任何 GRPO 候选都评不了。 | i=217, i=1157 |
| 217 | C4 | RL 方法:GRPO + LoRA rank 32 / alpha 64 覆盖全部 attention 与 MLP 投影,每 prompt 4 条 completion、有效 prompt batch 32、1,536 token completion、dr_grpo 损失、KL 系数 1e-3。 | i=217, i=1370 |
| 219 | C3 | prepare_rl.py:从 MiniByte-666/Dr.SCI 的 rule-verifiable 四选一行建 RL 池,并删掉与 SFT 的 1,016 条精确重叠,得 3,459 题。 | i=219, i=239 |
| 226 | C11 | 自写 analyze_eval.py:从 inspect_ai 日志抽 strict_final_format / contains_answer_marker / 停止原因 / 生成长度分布,把官方评测器自己的输出转成确定性可决策信号。 | i=226, i=683, i=1198 |
| 264 | C3 | prepare_long_sft.py:从 smohammadi/OpenThoughts3-science(pin b2e0778)挑近 4K token 的长科学推理轨迹,建 4096 上下文续训集 data/long_sft.jsonl。 | i=264, i=267 |
| 281 | C8 | 首轮 SFT 在 19% 处 OOM(差约 50MB):以 batch-size 3→2 重启,其余数据/lr/epoch 逐字不变;顺带把 save-strategy 从 epoch 改成 steps 122 以保住四分之一 epoch 候选。 | i=281, i=279 |
| 351 | C3 | 把长推理集从 1 个分片扩到全部 7 个分片、每题保留 2 条不同长度的教师轨迹,得 8,838 条轨迹 / 4,590 道题。 | i=351, i=477 |
| 355 | C3 | prepare_continuation.py:长轨迹 8,838 条 + 6,000 行标签平衡的 MCQ/bio/med/math replay 缓冲,防止续训把答案协议冲掉。 | i=355, i=468, i=477 |
| 412 | C1 | 自写 configure_model.py 重写 generation_config.json:eos_token_id 单值 151643 → 双值 [151645(<\|im_end\|>),151643]、max_new_tokens 2048→16000、补 pad_token_id,并提供… | i=412, i=507, i=230 |
| 478 | C4 | 把续训打包换成 best-fit-decreasing(data/packed_continuation_bfd_4096):打包效率 99.91%,块数从 11,280 降到 8,864,同样 36.3M 监督 token 少烧约 21% 计算。 | i=478, i=549, i=488 |
| 509 | C5 | 把 0.5-epoch 的 checkpoint-122 硬链接成 runs/sft_quarter,使其脱离 trainer 的三份 checkpoint 保留策略;它后来成为全程的 SFT 领跑者与 GRPO 父模型。 | i=509, i=692 |
| 519 | C3 | 把 RL 池精确平衡成 A/B/C/D 各 796 条(共 3,184 行),消除金标签先验。 | i=519, i=535 |
| 527 | C4 | 设计 GRPO 奖励:正确性权重 1.0 为主,终行格式 0.05,有界的「有用推理」0.15;并要求答案标记必须真正终结,以抑制短答塌缩。 | i=527, i=1370, i=535 |
| 610 | C3 | permute_rl.py:对每道 RL 题做确定性选项置换、同时保持金标签精确平衡,生成 data/rl_mcq_permuted(3,184 行)。 | i=610, i=614 |
| 648 | C8 | 给 configure_model.py 加 --tokenizer 参数:HuggingFace 元数据 ReadTimeout 让 tokenizer 拷贝失败,改成指向已在本地的 checkpoint,去掉网络依赖。 | i=648, i=654, i=646 |
| 730 | C5 | 给长推理续训加一道 epoch-1 闸门:写完 checkpoint-277 就停机评测,只有 GPQA 确认提升才继续剩下两个 epoch(实测省下约两小时)。 | i=730, i=831 |
| 839 | C8 | 去掉 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —— vLLM 的 sleep-mode 内存池拒绝 PyTorch expandable segments,GRPO 冒烟在任何更新之前就崩。 | i=839, i=840 |
| 850 | C8 | 关掉 mask_truncated_completions:TRL 的 colocated-vLLM 路径返回的 token id 不含停止符,导致全部 completion 被判为截断、梯度恒为零。 | i=850, i=846 |
| 857 | C4 | GRPO 生成路径从 colocated vLLM 换成同模型 on-policy Transformers 生成(--no-use-vllm),代价是慢,收益是消掉重要性比远低于 1 的 off-policy 失配。 | i=857, i=856 |
| 874 | C8 | 把训练侧 tokenizer 的 EOS 显式设成 <\|im_end\|>:Qwen chat 输出以 <\|im_end\|> 结束,而 tokenizer 只声明 <\|endoftext\|>,Transformers GRPO 等错单一 ID,每条序列都跑满 1,536 token。 | i=874, i=865 |
| 980 | C8 | stage2 前把训练用 pad token 也设成 <\|im_end\|>,使以 <\|endoftext\|> 结尾的序列及时停止并被正确 mask,而不是持续累积 padding loss。 | i=980, i=979 |
| 1065 | proposed:eval-protocol-t… | 调官方评测器的调用参数本身:--max-tokens 4096→8192→16000、--max-connections 8→6,目的是让长推理不再撞 token 上限,从而改变测量能看见什么(不改模型,也不改判据加工)。 | i=1065, i=817, i=1068 |
| 1093 | C1 | 把提交/评测用解码从 Qwen 推荐采样改成贪婪:temperature 0.6→0.0、top_k 20→-1、top_p 0.95→1.0(default 被 to_diff_dict 省略);do_sample、双 eos、max_new_tokens 16000、pad/bos 全部不变。 | i=1093, i=1334, i=507 |
| 1179 | C9 | 提交守卫第一次落子:把当时领跑的 runs/grpo_stage2_step80_merged 整目录复制进 final_model 并按 greedy 归一化 config,再直接从 final_model 跑一次全量确认。 | i=1179, i=1180 |
| 1189 | C11 | final_model 只读到 0.319 而源 checkpoint 曾读到 0.388 → 先做逐字节完整性审计(sha256 + cmp 两个权重分片、generation_config、tokenizer),排除打包出错再谈分数。 | i=1189, i=1193 |
| 1201 | C11 | 自写脚本横比四份 inspect_ai 日志的 per-target 命中与预测分布,定位分差来源是 evaluate.py 每次调用都 shuffle_choices 重洗选项,而非权重或解码——由此改用重复全量取均值+离散度选型。 | i=1201, i=1200, i=1202 |
| 1211 | C3 | rl_mcq_permuted4:把每道 Dr.SCI 题扩成全部四个正确标签位置(12,736 行,每标签 3,184),用于选项位置不变性的 RL 增强。 | i=1211, i=1215 |
| 1268 | C9 | 按三次独立全量评测的均值换交:stage-3 midpoint(36.2/34.4/36.4,均值 35.6%)顶掉 stage-2(38.8/31.9/31.5,均值 34.1%),旧产物保留成 runs/final_model_stage2_packaged 备份并逐字节 cmp。 | i=1268, i=1267, i=1269 |
| 1343 | C8 | 最后一次改 generation_config:do_sample true→false 并删掉 top_k(-1),使 Transformers .generate() 也能加载(原配置 temperature=0.0 + do_sample=True 被 Transformers 拒绝,而 vL… | i=1343, i=1318, i=1346 |

### 训练序列(22 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 147 | smoke | 0.02h | returned | **smoke** | baseline / 冒烟:全程第一次调用 train_sft.py,--max-steps 1,只为跑通 packed loss 路径与量显存吞吐,不测任何配方或超参变量。tested_variable 取值域里没有 smoke,故记 unclear(取值域装不下,非证据不足)。 |
| 156 | — | — | — | **unclear** | 相对 i=147:batch-size 2→4、加 --packed-cache 复用打包、关梯度检查点。但 --max-steps 3 --save-strategy no --no-final-save 说明它结构上不产出任何权重——这是 C8 可行性/吞吐探针,按 reference §3 C… |
| 161 | — | — | — | **unclear** | 相对 i=156:batch-size 4→2 并加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments,仍是 --save-strategy no --no-final-save 的吞吐探针,不产候选。 |
| 164 | — | — | — | **unclear** | 相对 i=161:batch-size 2→3,同为不落盘的吞吐探针;结论是 batch 3 稳定、约 9–10k token/s,直接决定了 i=168 的取值。 |
| 168 | real | 0.28h | returned | **baseline** | baseline:第一次真实训练。data/sft_train.jsonl 49,833 行、2 epoch、lr 3e-5、有效 batch 96(3×32)、save-strategy epoch。没有「上一次」可比,tested_variable 取值域里也没有 baseline。结局:19%… |
| 281 | real | 1.52h | returned | **unclear** | 相对 i=168:数据、lr、epoch、max-length 逐字相同;唯一差异是 batch-size 3→2(OOM 强制的补偿)与 save-strategy epoch→steps 122(为造 C5 候选)。这次训练不在验任何配方或方法主张,是被中断的 baseline 的重跑。结局:2… |
| 286 | — | — | — | **unclear** | 不是训练:--prepare-only,只把 data/long_sft.jsonl 打包成 data/packed_long_4096(3,428 块)后退出,没有任何优化步。机械层误记为训练启动,已写入 definition_defect。 |
| 308 | — | — | — | **unclear** | 不是训练:`train_grpo.py --help` 重定向到 /tmp/grpo_help.txt,随后是 transformers/trl/peft 版本自省。机械层误记为训练启动。 |
| 461 | — | — | — | **unclear** | 不是训练:先跑 prepare_continuation.py 合成 data/continuation_sft.jsonl,再用 train_sft.py --prepare-only 生成 packed_continuation_4096。机械层误记为训练启动。 |
| 480 | — | — | — | **unclear** | 不是训练:py_compile 之后再次 --prepare-only,产出 best-fit-decreasing 版打包 packed_continuation_bfd_4096(8,864 块,99.91% 效率)。机械层误记为训练启动。 |
| 711 | — | — | — | **unclear** | 冒烟:3 步,验 4096 上下文 + batch 1 的显存与步时(12.2 s/step),据此推算三个 epoch 约 2h49m。不测配方。 |
| 715 | real | 0.93h | returned | **both** | 相对 i=281:数据换成 data/continuation_sft.jsonl(OpenThoughts3 长推理 8,838 条 + 6,000 行 replay,C3),父模型换成 runs/sft_quarter;同时 max-length 2048→4096、batch 2→1、epoc… |
| 836 | smoke | 0.01h | returned | **smoke** | 冒烟:2 步 GRPO,首次跑 train_grpo.py。因 PYTORCH_CUDA_ALLOC_CONF=expandable_segments 与 vLLM sleep-mode 内存池冲突,在任何更新前失败。 |
| 840 | smoke | 0.02h | returned | **smoke** | 冒烟重跑:相对 i=836 只去掉 allocator 环境变量。加载成功,但暴露 mask_truncated_completions 把全部 completion 掩掉、梯度为零。 |
| 852 | smoke | 0.01h | returned | **smoke** | 冒烟:相对 i=840 关掉 mask_truncated_completions(输出目录改 grpo_smoke2)。梯度与熵恢复非零,但 vLLM 采样的 token logprob 与可训练 Transformers 策略的序列重要性比远低于 1。 |
| 857 | smoke | 0.05h | returned | **smoke** | 冒烟:相对 i=852 加 --no-use-vllm 走 on-policy Transformers 生成。诊断出 tokenizer 只认单一 EOS <\|endoftext\|>,每条序列跑满 1,536 token,被主动停掉。 |
| 876 | smoke | 0.02h | returned | **smoke** | 冒烟:相对 i=857 只加了训练侧 EOS=<\|im_end\|> 修正。通过:0% 截断、completion 68–299 token、格式 100%、梯度非零、约 24 s/step。 |
| 883 | real | 0.82h | returned | **both** | 相对 i=715(已被否决的长续训):方法从 SFT 换成 GRPO + LoRA r32/a64、on-policy Transformers 生成、dr_grpo、KL 1e-3、lr 5e-6、160 步(C4);数据从 continuation_sft 换成 data/rl_mcq_bala… |
| 982 | smoke | 0.04h | returned | **smoke** | 冒烟:2 步,验证 stage2 的双 EOS + pad=<\|im_end\|> 掩码改动是否正确(0% 截断、掩码正确、梯度非零)。父模型换成 grpo_step80_merged。 |
| 988 | real | 0.78h | returned | **both** | 相对 i=883:父模型换成合并后的 runs/grpo_step80_merged(续训),lr 5e-6→2.5e-6、步数 160→80(C4);数据集仍是默认的 rl_mcq_balanced,但 --seed 314159 重新洗牌——按 reference §5.3 对 C3 消融的口径… |
| 1128 | real | 0.43h | returned | **both** | 相对 i=988:数据集显式换成 data/rl_mcq_permuted(选项确定性置换版,C3);lr 2.5e-6→1.25e-6、步数 80→40、seed 314159→161803(C4)。父模型 runs/grpo_stage2_step80_merged。结局:跑满 40 步,25:… |
| 1214 | real | 0.38h | returned | **C3** | 相对 i=1128:父模型(runs/grpo_stage2_step80_merged)、lr 1.25e-6、--max-steps 40 逐字相同;唯一实质差异是数据集 rl_mcq_permuted(3,184 行 / 每题 1 种置换)→ rl_mcq_permuted4(12,736 行… |

### 验证序列(38 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 22 | 3.0 | 10.0 | 是 |  | 0.2 |
| 50 | 3.0 | 50.0 | 是 |  | 0.14 |
| 678 | 3.0 | 100.0 | 是 | c1, c3, c4, c33, c16 | 0.35 |
| 686 | 3.0 | 100.0 | 否 | c1, c33 | 未拿到:进程在加载前就 FileNotFoundError 退出——trainer 的三份保留策略已删掉 runs/sf… |
| 693 | 3.0 | 100.0 | 是 | c1, c3, c4, c33 | 0.34 |
| 698 | 3.0 | 100.0 | 是 | c1, c3, c4, c33 | 0.30 |
| 705 | 3.0 | 200.0 | 是 | c16, c33 | 0.33 |
| 707 | 3.0 | 200.0 | 是 | c16, c33 | 0.315 |
| 817 | 3.0 | 200.0 | 是 | c11, c12, c13, c14, c17, c30 | 0.315(并读到 183/200 strict finals、10 次 token 上限停止) |
| 965 | 3.0 | 200.0 | 是 | c7, c8, c9, c34, c19, c20, c21, c31 | 0.33 |
| 1050 | 3.0 | 200.0 | 是 | c22, c34 | 0.32 |
| 1056 | 3.0 | 200.0 | 是 | c22, c34 | 0.335 |
| 1065 | 3.0 | 448.0 | 是 | c22, c34, c30 | 0.328 |
| 1075 | 3.0 | 448.0 | 是 | c34, c20, c21 | 0.312 |
| 1082 | 3.0 | 448.0 | 是 | c1, c33, c16 | 0.339 |
| 1093 | 3.0 | 200.0 | 是 | c23 | 0.34(对照 i=705 同权重同 200 题的 0.33) |
| 1098 | 3.0 | 448.0 | 是 | c23 | 0.353(对照 i=1082 同权重同 448 题同 max-tokens 的 0.339,+1.4 点) |
| 1106 | 3.0 | 200.0 | 是 | c23 | 0.345 |
| 1110 | 3.0 | 448.0 | 是 | c23, c34 | 0.388(对照 i=1065 同权重同 448 题的 0.328,+6.0 点;后被 i=1258/1262 的重复评… |
| 1119 | 3.0 | 200.0 | 是 | c23 | 0.35 |
| 1123 | 3.0 | 448.0 | 是 | c23, c34 | 0.355 |
| 1162 | 3.0 | 200.0 | 是 | c10 | 0.345 |
| 1166 | 3.0 | 200.0 | 是 | c10 | 0.325 |
| 1171 | 3.0 | 448.0 | 是 | c10 | 0.362 |
| 1182 | 3.0 | 448.0 | 是 | c24 | 0.319(与源 checkpoint 的 0.388 对不上,触发 c25/c26 的审计) |
| 1232 | 3.0 | 200.0 | 是 | c27 | 0.35 |
| 1236 | 3.0 | 200.0 | 是 | c27 | 0.36 |
| 1241 | 3.0 | 448.0 | 是 | c27 | 0.306 |
| 1247 | 3.0 | 448.0 | 是 | c27 | 0.315 |
| 1253 | 3.0 | 448.0 | 是 | c10, c26 | 0.344 |
| 1258 | 3.0 | 448.0 | 是 | c24, c26 | 0.315 |
| 1263 | 3.0 | 448.0 | 是 | c10, c26 | 0.364 |
| 1272 | 3.0 | 448.0 | 是 | c1, c33, c16, c26 | 0.364 |
| 1277 | 3.0 | 448.0 | 是 | c1, c33, c16, c26 | 0.337 |
| 1282 | 3.0 | 448.0 | 是 | c34, c26 | 0.315 |
| 1288 | 3.0 | 448.0 | 是 | c34, c26 | 0.321 |
| 1304 | 3.0 | 10.0 | 是 | c28 | 0.3(10 题冒烟,只验产物能被 evaluate.py 加载与服务) |
| 1361 | 3.0 | 10.0 | 是 | c29, c28 | 0.3(改完 do_sample=false 后的最终产物冒烟;agent 明说小样本采样误差大,选型仍以三次 448 … |

### 异常与存疑

- **9 段训练的受测变量判不出**:i=[156, 161, 164, 281, 286, 308, 461, 480, 711]
- **1 次验证没有拿到信号**:i=[686]
- **分类学缺口提案 1 条**
  - eval-protocol-tuning(i=817, i=1065, i=1068, i=820)
- **定义缺陷 3 条**
  - (i=1082, i=1098, i=1341, i=1262)
  - (i=308, i=286, i=480, i=461)
  - (i=156, i=161, i=164, i=168)
- **边界情形 4 条**
  - (i=147, i=156, i=836, i=874)
  - (i=168, i=281, i=279)
  - (i=715, i=630, i=281)
  - (i=988, i=990, i=883)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gpqamain_google_gemma-3-4b-pt_17398712
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gpqamain | google_gemma-3-4b-pt | 6.39h | 0.2678571428… |

### 改动序列(30 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 89 | C10 | 去污染:审计 Mixture-of-Thoughts 里是否出现 gpqa/Idavidrein,并在 prepare_data 里过滤任何提到 GPQA 的记录 | i=89, i=78, i=162 |
| 96 | C3 | 建 SFT 语料 v1:Nemotron-SFT-Science-v2 合成 MCQ 60,000 条 + open-r1/Mixture-of-Thoughts science 15,000 条 | i=96, i=89 |
| 104 | C1 | merge_adapter.py 每次导出候选都改写 generation_config:base 是 {do_sample:true, top_k:64, top_p:0.95} 且**没有 temperature 字段**,改后为 do_sample=true / temperature=0.6… | i=104, i=488, i=102, i=324 |
| 162 | C2 | 训练样本对齐评测答案契约:统一成单个末行 `ANSWER: X`、completion-only 掩码、长度上限 3,072 token | i=162, i=89 |
| 197 | C8 | 显存/吞吐搜索:加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments、关梯度检查点、batch 从 8 退到 4 | i=197, i=247, i=213 |
| 201 | C3 | 第二份独立语料 data/sft_v1_gpqa_style:改用 Nemotron-Science-v1 的 MCQ.jsonl(与 v2 的 uuid 重叠为 0) | i=201, i=190 |
| 206 | C8 | 把训练样本截到 ≤1,024 token(留下 61K 条),为的是让未做梯度检查点的长 batch 不再撑爆显存 | i=206, i=213 |
| 239 | C8 | 自写 CompletionOnlyTrainer:只把被监督位置的 hidden 投影进 Gemma 的 262K 词表,数学等价但避开 OOM | i=239, i=779, i=276 |
| 303 | C5 | checkpoint 选择:每条训练都存中点与终点、分别合并成独立候选再评测,只保留胜者(最终选 grpo_aligned8 的 250 步而非 500 步) | i=303, i=1504 |
| 482 | C7 | 自建 GRPO 代理打分器 train_grpo.py:correctness_reward(合成标签精确字母)+ format_reward,复用评测器的 `ANSWER: X` 正则 | i=482, i=1098, i=607 |
| 531 | C8 | 把合并产物 config.json 的 transformers_version 压回 ≤4.57.2,并逐字节复制 base 的 tokenizer 文件,消除 transformers 误判 Mistral-regex 而发出的 "incorrect tokenization" 警告路径 | i=531, i=370, i=488, i=529 |
| 649 | C2 | filter_four_choice.py:只保留选项字母恰为 A/B/C/D 的记录,让训练样本的答案空间与 GPQA 一致(17,319 条) | i=649, i=669 |
| 731 | C3 | balance_answers.py / stratify_sft.py:按 学科 × 答案字母 12 格分层配平(每格 1,000,共 12,000) | i=731, i=736 |
| 783 | C4 | train_sft.py 加 --answer-token-weight:只对末尾 `ANSWER: X` 的答案 token 额外加权(phase4 用 8) | i=783, i=809, i=808 |
| 898 | C8 | 修 GRPOTrainer 的非法 kwarg:把 reward_weights 从 Trainer 参数挪进 GRPOConfig(此前构造即崩) | i=896, i=898 |
| 911 | C8 | 解除 truncated-completion 掩码:Gemma 以 <end_of_turn>(106)收尾而 TRL 只认 EOS=1,导致全部 rollout 被判 truncated、梯度恒为零 | i=908, i=911 |
| 925 | C8 | 把 vLLM 重要性采样从默认 sequence_mask 换成 token 级截断:序列级比值塌到 ~0.03 会把整个更新清零 | i=922, i=925, i=920 |
| 1014 | C1 | temperature 0.2 变体(与 phase3_6000 同权重、config 仅 temperature 一项不同) | i=1014, i=1109 |
| 1030 | C4 | 低学习率延续:同一份 data/sft_v1_1024、同 batch/步数,lr 5e-5→2e-5,换 shuffle 种子 | i=1030, i=1029 |
| 1118 | C1 | temperature 1.0 + top_k 64 变体,恢复 base gemma 的探索性解码 | i=1118, i=1117 |
| 1131 | C3 | permute_choices.py:生成四种循环选项置换、剔除引用选项位置的解释,得到 41,860 条 A/B/C/D 精确配平样本 | i=1131, i=1145 |
| 1186 | C3 | stratify_sft.py 按学科配比 40/40/20(physics/chemistry/biology)抽出 data/sft_permuted_ratio_1024(27,792 条) | i=1186, i=1198 |
| 1326 | C2 | GRPO 提示改走评测器未改动的 templates/gemma3.jinja 对话框架(此前 TRL 把裸字符串当续写),token 级 40/40 逐字对齐验证通过 | i=1325, i=1326, i=1332 |
| 1420 | C4 | GRPO 每 prompt 8 个 rollout、batch 8(此前 4/4),步数 200→500 | i=1420, i=1419 |
| 1701 | C4 | 正则化对照:rank 32→16、batch 8→16、lr 1e-6→5e-7,数据池刻意保持 grpo_permuted_ratio 不变 | i=1701, i=1685, i=1695 |
| 1735 | C3 | filter_grpo_difficulty.py:用当前策略自身 4 次 rollout 在线筛掉「全对/全错」的题,4,000 条留 1,280 条中等难度 | i=1735, i=1861, i=1892 |
| 1964 | C1 | 在 0.6 附近做窄幅解码扫描:temperature 0.5 与 0.7 两个变体 | i=1964, i=1967, i=1963 |
| 1975 | C9 | 提交守卫:final_model 导出的是被保留的 grpo_aligned8/checkpoint-250(而非最新权重),导出后用 sha256 核对两个 safetensors 分片与 index 与候选逐字节一致 | i=1975, i=2009, i=488 |
| 1999 | C1 | greedy 变体:把 generation_config 改成 do_sample=false / temperature=0.0 / top_p=1.0 / top_k=0 | i=1999, i=2002 |
| 2032 | C1 | final_model 的解码策略最终改成 greedy:默认协议下 greedy 两次 30%/28%(均值 29%),采样 0.6 三次 34/22/16(均值 24%);权重一字未动 | i=2032, i=2031 |

### 训练序列(21 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 174 | smoke | 0.02h | returned | **smoke** | baseline —— 全 run 第一次训练启动。1 步、batch 2、开梯度检查点,唯一目的是看 train_sft.py 跑不跑得通(结果:通过,train_runtime 4.66s)。受测的是可行性(C8),不是 C3 或 C4;现有取值域装不下,故填 unclear。 |
| 184 | smoke | 0.02h | returned | **smoke** | 相对 174:batch 2→8,加 --no-gradient-checkpointing;数据集不变(data/sft_v1)。纯显存/吞吐探测,结果 exit_code 1(在 compute_loss 处炸掉)。按 reference §3 C8 的口径,显存驱动的 batch 变动不得计入… |
| 197 | smoke | 0.02h | returned | **smoke** | 相对 184:batch 8→4,加 PYTORCH_CUDA_ALLOC_CONF=expandable_segments;数据集仍是 data/sft_v1。仍然 exit_code 1。同上,受测变量是显存可行性。 |
| 209 | smoke | 0.01h | returned | **smoke** | 相对 197:数据集 data/sft_v1 → data/sft_v1_1024(长度截到 ≤1024,61,281 条),batch 4→8。数据换了,但换的理由是显存(见 i=213),不是配方假设;仍然 exit_code 1。 |
| 241 | smoke | 0.01h | returned | **smoke** | 相对 209:命令行逐字相同(除 --output),差别在 train_sft.py 已换成 CompletionOnlyTrainer(i=239)。仍在 batch 8 上 OOM:'Tried to allocate 6.55 GiB'。受测的是新损失实现够不够省显存。 |
| 250 | smoke | 0.01h | returned | **smoke** | 相对 241:batch 8→4,其余逐字相同。首次跑通 10 步(loss 1.6181→…),由此定死 batch 4 + no-gc + completion-only loss 的可行配置。 |
| 254 | real | 0.16h | returned | **baseline** | baseline —— 第一次真实训练(phase1)。data/sft_v1_1024、2000 步、batch 4、lr 1.5e-4。没有上一次真实训练可比,"本次受测哪一项"在现有取值域里无对应值(spec §10 提议的 baseline 取值)。 |
| 373 | real | 0.30h | returned | **both** | 相对 phase1:数据集换成独立的 data/sft_v1_gpqa_style_1024(Nemotron-v1 来源,c9),同时 lr 1.5e-4→1e-4、步数 2000→4000、加 --seed 20260717,并从 phase1 的 adapter 续训。数据与超参同时动,age… |
| 593 | real | 0.44h | returned | **both** | 相对 phase2:数据集换回 data/sft_v1_1024(c1 的配方),同时 lr 1e-4→5e-5、步数 4000→6000。agent 自述是"换回 phase-1 的数据分布 + 24,000 例的放大",两条一起动。 |
| 809 | real | 0.21h | returned | **both** | 相对 phase3:数据集换成 data/sft_v1_abcd_stratified_1024(四选一过滤 + 学科×字母配平,c10/c11),同时 lr 5e-5→2e-5、步数 6000→3000、新增 --answer-token-weight 8(c12)。数据配方与训练目标同时改。 |
| 894 | real | 0.01h | returned | **unclear** | 首次 GRPO 尝试(runs/grpo1),从 candidate_gemma_phase3_6000 起。**在 GRPOTrainer 构造处即崩**(reward_weights 是非法 kwarg),一步没跑、无梯度、无产物。什么变量都没测到。 |
| 901 | real | 0.04h | returned | **unclear** | 与 894 的命令逐字相同(同 --output runs/grpo1、同 --seed 20260730),差别仅在 i=898 修好了 reward_weights。跑到第 2/200 步被 agent 主动中断(exit_code 130),用途是诊断 EOS 掩码 bug;agent 自己称… |
| 913 | real | 0.03h | returned | **unclear** | 相对 901:仅 --output grpo1→grpo2、--seed 20260730→20260731,外加 i=911 解除了 truncated 掩码。同样跑 2 步即被中断(exit_code 130),用途是诊断序列级重要性采样比值塌缩。 |
| 927 | real | 0.18h | returned | **both** | 相对 913:仅 --output grpo2→grpo3、--seed→20260801,外加 i=925 把重要性采样改成 token 级截断。**这是第一条跑完的 RL**。相对上一条跑完的训练(phase4)则同时换了方法(SFT→GRPO,c13/c20)和数据(data/grpo_sci… |
| 1030 | real | 0.44h | returned | **C4** | 相对 phase3:数据集逐字相同(data/sft_v1_1024)、batch 4、步数 6000、同样从 runs/phase3/final_adapter 起;只有 lr 5e-5→2e-5 和 shuffle 种子不同。数据来源与配比未动,故判 C4。 |
| 1286 | real | 0.38h | returned | **C3** | 相对 phase5:**只换数据集**(data/sft_v1_1024 → data/sft_permuted_ratio_1024,即 c18 的选项置换 + c19 的 40/40/20 学科配比);--adapter、--max-steps 6000、--save-steps 3000、--… |
| 1420 | real | 0.56h | returned | **both** | 相对 grpo3:数据池 grpo_science_abcd_stratified→grpo_permuted_ratio(c18/c19),同时 batch 4→8、num-generations 4→8、步数 200→500,并叠加 i=1326 的 chat-framing 修正(c17)。三… |
| 1505 | real | 0.29h | returned | **both** | 相对 grpo_aligned8:起点权重换成 grpo_aligned8_250,数据池换成独立的 data/grpo_gpqa_style_permuted(Nemotron-v1 来源),lr 1e-6→5e-7,步数 500→250。数据与超参同时动,agent 自述亦然。 |
| 1686 | real | 0.03h | returned | **both** | 本意是做 C4 正则化对照(rank 32→16、batch 8→16、lr 减半),但同时把数据池从 grpo_permuted_ratio 换成 grpo_gpqa_style_permuted。agent 在第 7/250 步意识到这会混杂,主动 SIGINT 杀掉(exit_code 130… |
| 1701 | real | 0.40h | returned | **C4** | 相对 grpo_aligned8:**数据池刻意保持 data/grpo_permuted_ratio 不变**,只改 rank 32→16、batch 8→16、lr 1e-6→5e-7、步数 500→250。起点权重同为 candidate_gemma_phase3_6000。本条 run 里最… |
| 1893 | real | 0.21h | returned | **both** | 相对 grpo_batch16_rank16_orig:数据池换成在线难度筛过的 data/grpo_difficulty_cp250(c21,1,280 条),lr 5e-7→2.5e-7,步数 250→125,起点权重从 phase3_6000 换成 grpo_aligned8_250;rank… |

### 验证序列(37 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 16 | 3.0 | 10.0 | 是 |  | 0.20 |
| 352 | 3.0 | 20.0 | 是 | c1, c2, c7, c28 | 0.30 |
| 367 | 3.0 | 20.0 | 是 | c1, c2, c7, c28 | 0.35 |
| 578 | 3.0 | 20.0 | 是 | c9, c28 | 0.15 |
| 587 | 3.0 | 20.0 | 是 | c9, c28 | 0.35 |
| 799 | 3.0 | 20.0 | 是 | c1, c28 | 0.20 |
| 804 | 3.0 | 20.0 | 是 | c1, c28 | 0.40 |
| 881 | 3.0 | 20.0 | 是 | c10, c11, c12, c28 | 0.20 |
| 888 | 3.0 | 20.0 | 是 | c10, c11, c12, c28 | 0.15 |
| 1005 | 3.0 | 20.0 | 是 | c13, c16, c28 | 0.30 |
| 1011 | 3.0 | 20.0 | 是 | c13, c16, c28 | 0.35 |
| 1019 | 3.0 | 20.0 | 是 | c22 | 0.10 |
| 1255 | 3.0 | 50.0 | 是 | c23 | 0.24 |
| 1256 | 3.0 | 50.0 | 否 |  | 未拿到 —— 与 i=1255 同时启动、各申请 gpu-memory-utilization 0.4,本条的 vLLM… |
| 1272 | 3.0 | 50.0 | 是 | c7, c23 | 0.36 |
| 1276 | 3.0 | 50.0 | 是 | c30, c28 | 0.28 |
| 1281 | 3.0 | 50.0 | 是 | c30, c28 | 0.30 |
| 1409 | 3.0 | 50.0 | 是 | c18, c19, c28 | 0.26 |
| 1415 | 3.0 | 50.0 | 是 | c18, c19, c28 | 0.26 |
| 1496 | 3.0 | 50.0 | 是 | c17, c18, c19, c20, c28 | 0.40 |
| 1501 | 3.0 | 50.0 | 是 | c17, c18, c19, c20, c28 | 0.20 |
| 1669 | 3.0 | 50.0 | 是 | c9, c28 | 0.22 |
| 1674 | 3.0 | 50.0 | 是 | c9, c28 | 0.32 |
| 1678 | 3.0 | 50.0 | 是 | c17, c20 | 0.34 |
| 1682 | 3.0 | 50.0 | 是 | c17, c20 | 0.22 |
| 1843 | 3.0 | 50.0 | 是 | c29, c28 | 0.30 |
| 1847 | 3.0 | 50.0 | 是 | c29, c28 | 0.28 |
| 1955 | 3.0 | 50.0 | 是 | c21, c28 | 0.38 |
| 1960 | 3.0 | 50.0 | 是 | c21, c28 | 0.28 |
| 1969 | 3.0 | 50.0 | 是 | c24 | 0.22 |
| 1971 | 3.0 | 50.0 | 是 | c24 | 0.36 |
| 1987 | 3.0 | 50.0 | 是 | c26 | 0.22 |
| 1993 | 3.0 | 50.0 | 是 | c24 | 0.20 |
| 2001 | 3.0 | 50.0 | 是 | c25 | 0.30 |
| 2024 | 3.0 | 50.0 | 是 | c26, c27 | 0.16 |
| 2028 | 3.0 | 50.0 | 是 | c25, c27 | 0.28 |
| 2036 | 3.0 | 50.0 | 是 | c26, c27 | 0.28 |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[894, 901, 913]
- **1 次验证没有拿到信号**:i=[1256]
- **分类学缺口提案 1 条**
  - experiment_design_guard(i=1695, i=1690, i=1701)
- **定义缺陷 2 条**
  - 本条 run 里有第二组**严格单字段** C1 对照,而且方向相反。runs/candidate_gemma_phase3_6000 与 runs/candidate_gemma_phase3_t02 由同一条 merge_adapter.py、同一 base、同一 runs/phase3/final_adapter 产生,i=1108 把两份 generation_config.json 逐字…(i=1109, i=1108, i=804, i=806, i=1019, i=1027)
  - 该判据实际是按**产物名**判定,不是按意图。本 run 里 i=894 / i=901 / i=913 三次启动都被机械层记成 kind=real,而 agent 自己在 i=908 把它们称作 "The GRPO smoke test";它们分别是构造即崩(0 步)、跑 2 步被 SIGINT、跑 2 步被 SIGINT,没有一次产出 checkpoint 或参与任何候选。它们只是恰好叫 ru…(i=908, i=896, i=909, i=923)
- **边界情形 4 条**
  - 本 run 21 行训练里有 9 行受测的既不是 C3 也不是 C4,而是「管线跑不跑得起来 / 更新是不是零」:6 次冒烟(174/184/197/209/241/250)在搜 batch 与损失实现,3 次 GRPO 启动(894/901/913)在诊断非法 kwarg、EOS 掩码、重要性采样塌缩。按 reference §3,C8 顺手改掉的 batch/序列长度**不得**计入 C4,所…(i=908, i=922, i=213)
  - i=254(phase1)是第一次真实训练,没有"上一次"可比,「本次受测哪一项」无定义。现在只能填 unclear,把「schema 装不下」混进「证据不足」——正是 spec §10 第 1 条要修的。(i=254, i=276)
  - i=649 的 filter_four_choice.py 只保留选项字母恰为 A/B/C/D 的记录。按 C2(让训练样本与评测 prompt 逐字节一致)成立——它对齐的是评测的答案空间;按 C3(决定训练数据从哪来、按什么比例混)也成立——它把语料从 61K 砍到 17,319 条,直接改变了配方。现定义没有优先级规则。同类还有 i=1131 的选项置换(格式增强 vs 数据扩增)。(i=649, i=669, i=1131)
  - i=925 把 vLLM importance sampling 从 sequence_mask 换成 token 级截断。它在形式上是 GRPO 的一个超参(C4 的取值域之内),但不做则整条 RL 的更新被乘上 ~0.03、等于没训——满足 C8 的「二值:不做则整条不跑」判据。i=911 的 EOS 掩码修复同理:训练进程一直在跑,只是梯度恒为零。现定义按「跑不跑得起来」区分,而这两例是「跑…(i=922, i=908, i=925)

## codex_non_api_xhigh_gpt-5.5_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17134243
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.5 | codex | gsm8k | Qwen_Qwen3-4B-Base | 1.74h | — |

### 改动序列(5 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 53 | C2 | 在自写的 train_gsm8k_lora.py 里把训练样本构造成评测器同款的对话格式:10 条 train-set few-shot 进 system、评测的 MATH_PROMPT_TEMPLATE 包 user、completion 以 'ANSWER: X' 收尾;并且只用 templat… | i=34, i=52, i=48, i=75, i=62 |
| 53 | C1 | 同一个脚本里把提交模型的生成停止符改成 <\|im_end\|>(151645)。动机是读到 base 的 GenerationConfig 里 eos_token_id 是 151643(<\|endoftext\|>),chat 格式下 vLLM 不会在 <\|im_end\|> 停。改动本身写… | i=50, i=51, i=52 |
| 53 | C3 | 训练数据只用 GSM8K 官方 train split(7,473 条,来源类别 a),并明确否掉了 MetaMathQA / GSM8K-Rephrased 这类公开蒸馏集,理由是可能含 GSM8K test、有污染风险;也没有采用自己想过的合成算术题增广。 | i=33, i=33, i=62 |
| 76 | C4 | 第一个真实候选的方法与超参:LoRA r=64 / alpha=128 / dropout=0.03 挂在 attention + MLP 投影上,bf16,3 epochs(702 optimizer steps),batch 8 × grad-accum 4,lr 2e-4,max-length… | i=76, i=75, i=78 |
| 76 | proposed:submission-pack… | 把训练产物的 merge 目标直接设成计分路径 final_model:训练脚本 --output-dir final_model,没有中转目录、没有先评测再晋级的守卫。这是一次关于「哪份权重占据提交位」的有意选择,发生在任何 checkpoint 存在之前,不是 C5 的候选间挑选。本 run 里… | i=75, i=76, i=89 |

### 训练序列(2 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 61 | smoke | 0.02h | returned | **smoke** | baseline(本 run 第一次训练启动)。一步冒烟:--max-steps 1 / batch 1 / accum 1 / r8 / alpha16,产物 runs/smoke_model。意图明确写在 i=60:验 tokenizer、掩码、merge、save 四条链路能否跑通,不是在验任… |
| 76 | real | — | no_result | **both** | 上一次训练是 i=61 的一步冒烟,不是候选臂,所以这次没有对照对象——它是本 run 唯一的真实候选,一次性同时锁定了数据配方(GSM8K 官方 train 7,473 条 + 评测同款 10-shot 对话格式,c1/c3)和方法/超参(LoRA r64 alpha128 dropout0.03… |

### 验证序列(1 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 35 | 3.0 | 20.0 | 是 |  | 0.5 |

### 异常与存疑

- **分类学缺口提案 1 条**
  - submission-packaging(i=75, i=76)
- **定义缺陷 3 条**
  - (i=61, i=60)
  - (i=53, i=52, i=51)
  - (i=29, i=50, i=35)
- **边界情形 1 条**
  - 首次(baseline)真实训练在 tested_variable 上结构性落入 both,而它与「一次有意同时动数据和超参的双变量试验」不是同一种认识论状态。i=76 是本 run 唯一的真实候选:它同时确定了数据配方(c3)与方法/超参(c4),但它不是在对照任何前一臂——它就是第一臂。按现定义只能记 both;若改记 unclear 又是错的(证据充分)。后果是缺口 2 的拆分被系统性稀释:…(i=75, i=76, i=61)

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
| 47 | smoke | 0.01h | returned | **smoke** | baseline(全 run 第一次训练)。2 步 LoRA 冒烟,目的明确写着是验管线(tokenizer / loss mask / 模型加载),既不在验 C3 也不在验 C4;四值域里没有"两者都不验"这一档,故记 unclear。 |
| 50 | smoke | 0.01h | returned | **C4** | 与 i=47 相比去掉 --lora 改全参、bs 1→2、显式 --learning-rate 1e-5;--train-limit 16 / --prompt-mode exact / --max-steps 2 / --max-length 3072 逐字相同。验的是全参能否放得下、跑多快。 |
| 54 | real | 0.85h | returned | **baseline** | baseline(第一次真实训练):GSM8K 官方 train split、exact 提示、2 epoch、bs 2×accum 4、max-length 4096、lr 1e-5。没有可比的上一次真实训练,受测变量无从定义。 |
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
| 34 | 3.0 | 20.0 | 是 |  | 0.0(未改动的 base 模型基线,n=20;不判定任何改动) |
| 90 | 3.0 | 50.0 | 是 | c1, c2 | 0.1(n=50;agent 读到 10%,判定为"不够") |
| 104 | 3.0 | 50.0 | 是 | c3 | 0.46(n=50;与 i=90 严格同权重、同 --limit 50 与同评测参数,唯一变量是 eos_token_i… |
| 168 | 3.0 | 50.0 | 是 | c6, c7 | 0.5(n=50;对比 i=104 的 0.46) |
| 174 | 3.0 | 150.0 | 是 | c6, c7 | 0.607(n=150) |
| 178 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.6(n=150;把 exact-only 基线拉到与 i=174 同样本量做对照) |
| 192 | 3.0 | 150.0 | 是 | c6 | 0.593(n=150;refresh 之前的纯 direct-MetaMath 阶段) |
| 210 | 3.0 | 150.0 | 是 | c8 | 0.567(n=150;低 LR 额外一轮回退,候选被弃) |
| 300 | 3.0 | 150.0 | 是 | c9 | 0.593(n=150;MetaMath-first 第一阶段) |
| 342 | 3.0 | 150.0 | 是 | c9 | 0.473(n=150;顺序实验第二臂,明显劣于 exact-first 的 0.607) |
| 362 | 3.0 | 150.0 | 是 | c10 | 0.593(n=150;alpha 0.5 soup 低于被平均的最好 checkpoint) |
| 369 | 3.0 | 150.0 | 是 | c11 | 0.567(n=150;alpha 0.8 soup 更差,权重平均整条线被放弃) |
| 382 | — | — | 是 | c6, c7 | 0.56;意图是全量,实际只跑了 evaluate.py 的默认 150 题(agent 当场发现),故这条不是第四档 |
| 394 | 3.0 | 150.0 | 是 | c12 | 0.52(n=150;同一份权重、同一批 150 题,只把评测参数换成脚本真实默认值,就从 i=174 的 0.607 … |
| 405 | 3.0 | 150.0 | 是 | c13 | 0.70(n=150;与 i=394 同权重同题同评测参数,唯一变量是新增的 do_sample/temperature… |
| 412 | 3.0 | 150.0 | 是 | c14 | 0.713(n=150) |
| 418 | 3.0 | 150.0 | 是 | c15 | 0.747(n=150;贪婪解码下候选排序相对采样解码完全反转) |
| 424 | 3.0 | 1319.0 | 是 | c6, c15 | 0.7467778620166793(全量 1319 题) |
| 570 | 3.0 | 150.0 | 是 | c17, c18 | 0.733(n=150;低于 100k 版的 0.747,但 agent 判断落在短样本噪声内,决定上全量) |
| 582 | 3.0 | 1319.0 | 是 | c17 | 0.755117513267627(全量 1319 题;与 i=424 构成 100k vs 200k 的单变量 C3 … |
| 600 | 3.0 | 150.0 | 是 | c19 | 0.733(n=150;确认 final_model 目录本身能加载,与 i=570 同数) |
| 787 | 3.0 | 1319.0 | 是 | c19 | 0.7573919636087946(全量 1319 题;与 i=582 是同一份权重同一份配置,却差 3 道题:0.7… |

### 异常与存疑

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

## codex_non_api_max_gpt-5.6-sol_10h_run1__gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17397511
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gsm8k | HuggingFaceTB_SmolLM3-3B-Base | 9.85h | 0.7414708112… |

### 改动序列(33 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 107 | C3 | Stage-1 corpus fixed to MetaMathQA rows explicitly derived from the GSM8K train split (GSM_AnsAug + GSM_Rephrased, 160k rows); Orca-Math excluded beca… | i=107, i=84 |
| 108 | C2 | train_math.py renders every training sample in the evaluator's exact ChatML dialogue and ends the assistant target with 'ANSWER: <number>' followed by… | i=59, i=688 |
| 120 | C4 | Stage-1 method: full-parameter BF16 SFT with TRL best-fit-decreasing packing (92.9% efficiency), max_length 1024, batch 8 x accum 2, lr 1e-5 cosine, o… | i=120, i=133 |
| 162 | C1 | Saved-checkpoint tokenizer registers the already-present <\|im_start\|>/<\|im_end\|> ids as special tokens (no vocab resize) so vLLM strips the bounda… | i=164, i=685 |
| 190 | C3 | Added an Orca-Math (200k word problems) continuation stage, reversing the i=107 exclusion after the paper's contamination section was read. | i=189 |
| 256 | proposed:label_quality_r… | Dropped the synthetically appended 'ANSWER:' suffix from Orca rows after a 25-row audit showed the trailing number is frequently a unit descriptor, no… | i=255, i=251 |
| 280 | C11 | Built analyze_eval.py, which reads the inspect_ai eval log and reports answer_suffix_rate, fake_followup_rate, empty_rate and output-token statistics … | i=284, i=282 |
| 312 | C10 | Self-built protected_files.sha256 manifest over evaluate.py and templates/*, re-verified at every later integrity pass, to prove the graded evaluator … | i=312, i=1503 |
| 424 | C4 | Exposed --lr-scheduler-type so the Orca control could follow the published constant 1e-6 schedule instead of an annealed cosine. | i=423, i=426 |
| 452 | C8 | Relaunched training with HF_DATASETS_OFFLINE/HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE after a HuggingFace metadata ReadTimeout stalled the first align prob… | i=452, i=449 |
| 496 | C3 | Added --broad-all-types: switch from the two GSM transformation families to the published mixed 100k MetaMathQA curriculum (~61% GSM / 39% MATH transf… | i=489, i=494 |
| 505 | C8 | Pass the cached base snapshot path instead of the hub id, because Transformers' tokenizer heuristic issues an online model-info call even under offlin… | i=501, i=505 |
| 555 | C6 | Built merge_models.py (two-checkpoint linear interpolation, FP32 accumulation stored BF16) and swept the stage1_model/mixed_model blend at alpha = 0.1… | i=559, i=574 |
| 601 | C3 | Added --sample-offset so subsequent fresh-from-base branches train on disjoint 100k slices of MetaMathQA (rows 100k-200k, 200k-300k) with otherwise id… | i=600, i=691 |
| 644 | C6 | Three-way soups: mix12_50 = 0.5*(mixed_model + mixed2_model), then soup_s15/s25/s30 = alpha*stage1_model + (1-alpha)*mix12_50. | i=644, i=643 |
| 737 | C6 | Re-applied the winning 30/70 interpolation recipe to the other branches: s2_merge30 = 0.30*stage1 + 0.70*mixed2 (i=737) and s3_merge30 = 0.30*stage1 +… | i=737, i=1197 |
| 763 | C2 | Added an opt-in fix_mistral_regex tokenizer switch after finding the corrected regex retokenizes 6,371 of 7,473 GSM8K training questions (digit splitt… | i=754, i=781 |
| 791 | C6 | Built merge_many.py for a direct N-checkpoint FP32-weighted merge to avoid the rounding of sequential BF16 merges; only --help was ever run, no evalua… | i=787, i=794 |
| 846 | C4 | Added --official-recipe: packing disabled, global shuffle then select 100k with a 1% holdout, warmup 0.05, batch 2 x accum 8 -> 6,188 optimizer update… | i=827, i=840, i=961 |
| 1161 | C3 | Added disjoint-offset sampling to the broad stage so the second format-adaptation pass consumed fresh GSM-derived rows (offset 20000, 40k rows). | i=1163, i=1164 |
| 1252 | C9 | First submission decision: package merge_30 (63.3% on 150, 60.2% on 500) into final_model by rebuilding the 0.30/0.70 interpolation directly into that… | i=1252, i=1256 |
| 1334 | C1 | Added "temperature": 0.0 to final_model/generation_config.json. Field-level diff (reconstructed from the read at i=1318/1319 and the later read at i=1… | i=1333, i=1319, i=1431 |
| 1353 | C1 | Wrote the same greedy generation_config into merge_25/merge_29/merge_31/merge_35/mixed_model/stage1_model so the whole interpolation weight sweep coul… | i=1352, i=1353 |
| 1382 | C8 | Rewrote stage1_model/generation_config.json from temperature 0.0 to do_sample=true + temperature=1e-06 because transformers GenerationConfig.validate(… | i=1381, i=1379, i=1376 |
| 1383 | C6 | Broadened the interpolation sweep toward the pure stage-1 endpoint under greedy decoding: merge_g50/g60/g70/g80/g90 = alpha*stage1_model + (1-alpha)*m… | i=1383, i=1371 |
| 1397 | C9 | Submission swap: cp -a stage1_model/. final_model/ after the full-set greedy tie-break put the plain stage-1 SFT branch (74.53%) 1.9 points above the … | i=1397, i=1396 |
| 1404 | C9 | Deleted final_model/merge_manifest.json, the stale lineage file inherited from the merge_30 packaging, so the submitted artifact does not carry a mani… | i=1404, i=1436 |
| 1404 | C1 | Restored exact greedy in final_model/generation_config.json after the cp -a at i=1397 had copied stage1_model's do_sample=true / temperature=1e-06 pai… | i=1402, i=1431 |
| 1416 | C1 | Added repetition_penalty 1.02 on top of greedy decoding in final_model/generation_config.json as a last decoding refinement. | i=1415, i=1417 |
| 1420 | C1 | Reverted the repetition penalty after it cost 1.3 points on the 150-item slice (72.7% vs 74.0%). | i=1422 |
| 1423 | C1 | Set merge_g50/generation_config.json to exact temperature 0.0 so its head-to-head against the leader was run at true greedy rather than vLLM's clamped… | i=1422, i=1423 |
| 1434 | proposed:submission_docu… | Wrote EXPERIMENTS.md and final_model/README.md recording the training lineage and the evaluation evidence for the submitted checkpoint. | i=1433, i=1434 |
| 1447 | C1 | Wrote greedy generation_config into the 10 remaining branch/soup directories (mixed2, mixed3, s2_merge30, s3_merge30, soup_s15/25/30, leader_polish20,… | i=1440, i=1447 |

### 训练序列(16 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 114 | smoke | 0.01h | returned | **smoke** | baseline / smoke. First launch: 10 optimizer steps on 2,000 rows purely to measure throughput and memory before committing to the full stage. 'unclear… |
| 122 | real | 0.88h | returned | **baseline** | baseline: first real training. 160k MetaMathQA GSM8K-train-derived rows, packed, max_len 1024, bs 8 x accum 2, lr 1e-5 cosine, one epoch. Nothing to c… |
| 394 | real | 0.17h | returned | **both** | vs stage1_model: adds a new data source (20k Orca-Math word problems, C3) AND a new continuation regime (lr 5e-6 cosine from an already-trained checkp… |
| 426 | real | 0.17h | returned | **C4** | vs orca_probe_hi: near-clean single-axis ablation. Same start checkpoint (stage1_model), same 20,000 Orca rows, same formatting and packed blocks; onl… |
| 444 | smoke | 0.04h | returned | **smoke** | Never trained. Launched the align stage (GSM8K train split rendered as the evaluator dialogue) but a HuggingFace metadata ReadTimeout stalled dataset … |
| 452 | smoke | 0.09h | returned | **both** | vs orca_probe_lo (last real training): both axes move. Data becomes the GSM8K training split rendered byte-for-byte as the evaluator dialogue with ass… |
| 468 | smoke | 0.09h | returned | **C4** | vs align_probe_hi: near-clean single-axis ablation. Identical 2,000-example subset and identical rendering; only lr 5e-6 cosine -> 5e-7 constant. Cave… |
| 509 | real | 0.70h | returned | **both** | vs align_probe_lo, and more meaningfully vs stage1_model: fresh from the base checkpoint on the published mixed 100k curriculum (61% GSM / 39% MATH tr… |
| 603 | real | 0.70h | returned | **C3** | vs mixed_model: clean single-axis C3 replicate. Identical recipe, identical hyperparameters, only --sample-offset 100000 selecting a disjoint 100k sli… |
| 691 | real | 0.70h | returned | **C3** | vs mixed2_model: same clean single-axis C3 replicate, third disjoint 100k slice (--sample-offset 200000), identical optimization settings. Together wi… |
| 962 | real | 2.43h | returned | **both** | vs mixed3_model: the --official-recipe run. C4 is the stated hypothesis (packing off, 6,188 optimizer updates vs 1,126, warmup 0.05); C3 also moves (g… |
| 1140 | real | 0.12h | returned | **both** | vs official_model: continuation on 20k GSM-derived MetaMath rows re-rendered in the evaluator's Reasoning:/ANSWER: contract (C3, and again a C2 rider … |
| 1164 | real | 0.22h | returned | **both** | vs official_adapt20: continues the same adaptation on the next disjoint 40k GSM-derived rows (C3, --sample-offset 20000) at lr 1e-6 -> 2e-6 (C4). |
| 1208 | real | 0.13h | returned | **both** | vs official_adapt60: a gentle polish of a different parent (merge_30, the interpolated leader) on a fresh 20k mixed-curriculum block never seen by its… |
| 1464 | real | 0.00h | returned | **unclear** | Never trained. Intended as a disjoint continuation of stage1_model (--sample-offset 160000), but the GSM-derived source has exactly 160,000 rows so da… |
| 1472 | real | 0.31h | returned | **both** | vs leader_polish20: the align stage on 7,000 GSM8K training rows rendered as the evaluator dialogue (C3/C2) at max_length 3072, batch 2 x accum 4, lr … |

### 验证序列(49 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 26 | 3.0 | 50.0 | 是 |  | base model, 0.14 (7/50); the untouched baseline, later re-re… |
| 380 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | stage1_model = 0.547 (82/150); becomes the reference every l… |
| 409 | 3.0 | 150.0 | 是 | c5, c6 | orca_probe_hi = 0.480, a regression from 0.547; ANSWER: comp… |
| 435 | 3.0 | 150.0 | 是 | c5, c9 | orca_probe_lo = 0.473, also below 0.547; fake follow-ups ros… |
| 459 | 3.0 | 150.0 | 是 | c2 | align_probe_hi = 0.387; diagnosed as arithmetic forgetting (… |
| 474 | 3.0 | 150.0 | 是 | c2, c9 | align_probe_lo = 0.467; direct GSM8K alignment rejected |
| 542 | 3.0 | 150.0 | 是 | c11, c12 | mixed_model = 0.553 (83/150), narrowly above stage1's 0.547 |
| 560 | 3.0 | 150.0 | 是 | c13 | merge_50 = 0.553 |
| 567 | 3.0 | 150.0 | 是 | c13 | merge_25 = 0.607 |
| 567 | 3.0 | 150.0 | 是 | c13 | merge_25 = 0.607 |
| 567 | 3.0 | 150.0 | 是 | c13 | merge_75 = 0.553 |
| 567 | 3.0 | 150.0 | 是 | c13 | merge_75 = 0.553 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_15 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_15 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_15 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_15 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_20 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_20 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_20 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_20 = 0.607 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_30 = 0.633 (95/150), the sampled-decoding leader |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_30 = 0.633 (95/150), the sampled-decoding leader |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_30 = 0.633 (95/150), the sampled-decoding leader |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_30 = 0.633 (95/150), the sampled-decoding leader |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_35 = 0.533 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_35 = 0.533 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_35 = 0.533 |
| 577 | 3.0 | 150.0 | 是 | c13 | merge_35 = 0.533 |
| 586 | 3.0 | 500.0 | 是 | c13 | merge_30 = 0.602 (301/500); read back through analyze_eval.p… |
| 590 | 3.0 | 500.0 | 是 | c13 | merge_20 = 0.160 (80/500) with 786 mean output tokens; the 1… |
| 635 | 3.0 | 150.0 | 是 | c14, c11 | mixed2_model = 0.587 |
| 647 | 3.0 | 150.0 | 是 | c15 | mix12_50 = 0.573 |
| 647 | 3.0 | 150.0 | 是 | c15 | mix12_50 = 0.573 |
| 647 | 3.0 | 150.0 | 是 | c15 | mix12_50 = 0.573 |
| 647 | 3.0 | 150.0 | 是 | c15 | mix12_50 = 0.573 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s15 = 0.633 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s15 = 0.633 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s15 = 0.633 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s15 = 0.633 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s25 = 0.640 (96/150), the sampled-decoding peak |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s25 = 0.640 (96/150), the sampled-decoding peak |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s25 = 0.640 (96/150), the sampled-decoding peak |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s25 = 0.640 (96/150), the sampled-decoding peak |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s30 = 0.553 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s30 = 0.553 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s30 = 0.553 |
| 647 | 3.0 | 150.0 | 是 | c15 | soup_s30 = 0.553 |
| 657 | 3.0 | 500.0 | 是 | c15 | soup_s25 = 0.580 on 500, below merge_30's 0.602; the 150-ite… |
| 662 | 3.0 | 500.0 | 是 | c15 | soup_s15 = 0.588 on 500, also below 0.602; three-way soups r… |
| 1130 | 3.0 | 150.0 | 是 | c19 | official_model = 0.060; ~917 mean output tokens, 90.7% fake … |
| 1154 | 3.0 | 150.0 | 是 | c19, c2 | official_adapt20 = 0.373; mean output 917 -> 505 tokens, fol… |
| 1185 | 3.0 | 150.0 | 是 | c19, c20 | official_adapt60 = 0.347, a regression; the official-recipe … |
| 1189 | 3.0 | 150.0 | 是 | c16 | s2_merge30 = 0.620 |
| 1193 | 3.0 | 150.0 | 是 | c14 | mixed3_model = 0.573 |
| 1199 | 3.0 | 150.0 | 是 | c16 | s3_merge30 = 0.593, rejected |
| 1203 | 3.0 | 500.0 | 是 | c16 | s2_merge30 = 0.594 on 500, below merge_30's 0.602 |
| 1221 | 3.0 | 150.0 | 是 | c14 | leader_polish20 = 0.567, a regression from 0.633; the polish… |
| 1225 | 3.0 | 500.0 | 是 | c13 | merge_25 = 0.592 on 500 |
| 1235 | 3.0 | 150.0 | 是 | c13 | merge_29 = 0.580 |
| 1238 | 3.0 | 150.0 | 是 | c13 | merge_31 = 0.607; the alpha search is closed at 0.30 |
| 1257 | 3.0 | 1319.0 | 是 | c21 | final_model = 0.061 (81/1319) with 933 mean output tokens; p… |
| 1312 | 3.0 | 150.0 | 是 | c21 | final_model = 0.513 on the same 150 items that scored 0.633 … |
| 1336 | 3.0 | 150.0 | 是 | c22 | final_model = 0.740. Controlled single-field C1 measurement:… |
| 1341 | 3.0 | 1319.0 | 是 | c22 | final_model = 0.726 (958/1319), against 0.061 for the same w… |
| 1346 | — | — | 是 | c22 | final_model = 0.727 under evaluate.py's untouched defaults (… |
| 1355 | 3.0 | 150.0 | 是 | c23, c13 | One shell loop = SIX evaluations at 150 greedy (merge_25, me… |
| 1364 | 3.0 | 1319.0 | 是 | c23, c13 | One shell loop = FOUR full-set evaluations. Read back verbat… |
| 1386 | 3.0 | 150.0 | 是 | c33, c24 | One shell loop = FIVE evaluations at 150 (merge_g50..g90). m… |
| 1392 | 3.0 | 1319.0 | 是 | c33 | merge_g50 = 0.728 on 1319; the 0.800/150 peak did not genera… |
| 1406 | — | — | 是 | c25, c26 | final_model (now stage1_model weights) = 0.740 under stock d… |
| 1417 | 3.0 | 150.0 | 是 | c28 | final_model with repetition_penalty 1.02 = 0.727, below 0.74… |
| 1424 | 3.0 | 150.0 | 是 | c30 | merge_g50 at exact temperature 0.0 = 0.793 on 150, against 0… |
| 1427 | 3.0 | 1319.0 | 是 | c30 | merge_g50 at exact greedy = 0.736 on 1319, against 0.728 cla… |
| 1449 | 3.0 | 150.0 | 是 | c32, c15, c16 | One shell loop = TEN evaluations at 150 greedy, all ten read… |
| 1458 | 3.0 | 1319.0 | 是 | c32, c15, c16 | One shell loop = TWO full-set evaluations: soup_s30 0.738 an… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[1464]
- **分类学缺口提案 2 条**
  - label_quality_repair(i=255, i=494, i=251)
  - submission_documentation(i=1433, i=1434, i=336)
- **定义缺陷 5 条**
  - (i=468, i=477, i=426)
  - (i=1465, i=1471, i=1500)
  - (i=1364, i=1373, i=1455)
  - (i=574, i=582, i=1373)
  - (i=961, i=1134, i=1157)
- **边界情形 5 条**
  - Registering <\|im_start\|> and <\|im_end\|> as tokenizer special tokens in the saved checkpoint. Its purpose is exactly C1's stop-token branch (make the served model terminate so the scorer reads a ba…(i=164, i=685)
  - The self-built protected_files.sha256 manifest over evaluate.py and templates/*. C10's role (compliance audit, effect zero or negative, decides whether the score is valid) fits precisely, but C10's en…(i=312, i=1503)
  - Excluding Orca-Math from the stage-1 corpus. Read forward it is the C3 data-source decision; read backward it is a decontamination filter with an expected non-positive effect ('despite its strong resu…(i=107, i=189)
  - The do_sample=true / temperature=1e-06 rewrite is a pure C8 feasibility fix (transformers refuses to serialize a temperature with do_sample=False, which blocked merge_models.py) that silently changes …(i=1381, i=1422)
  - One file_change event that both restores a C1 value in generation_config.json and deletes the stale merge_manifest.json inherited from the previous packaging. The first half is C1; the second half is …(i=1404, i=1436)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gsm8k_Qwen_Qwen3-1.7B-Base_17397510
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gsm8k | Qwen_Qwen3-1.7B-Base | 8.59h | 0.5724033358… |

### 改动序列(47 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 82 | C1 | train_sft.py 保存 checkpoint 时把 generation_config 的 eos_token_id 写成 [<\|im_end\|>151645, <\|endoftext\|>151643] 双停止符。字段级差异(由 i=53 与 i=148 两次 cat 直接读出):b… | i=82, i=53, i=148, i=952, i=1093 |
| 94 | C2 | 建 prepare_data.py / train_sft.py:训练 prompt 用与 evaluate.py 逐字相同的 MATH_PROMPT_TEMPLATE,completion-only loss,目标以 `ANSWER: X<\|im_end\|>` 收尾 | i=94, i=916 |
| 96 | C3 | 一次性产出三份 train-only 数据源:gsm8k_train 7473 / gsm8k_train_context 7473 / openmath2_gsm_train 153,271(OpenMathInstruct-2 只取 gsm8k + augmented_gsm8k 两个来源) | i=96, i=97 |
| 96 | C10 | 数据准备内置合规过滤:GSM8K test 精确重叠拒收(命中 0)+ 剔除 40 行畸形样本,并写 provenance 到 manifest | i=97, i=97 |
| 104 | C8 | 首次 SFT 用 batch-size 32 撑爆显存、无 checkpoint;降到 16 并把 accumulation 2→4 保持有效 batch 64 不变后重启 | i=103, i=104 |
| 106 | C10 | 自写 TF-IDF char-ngram + 最近邻脚本,对 OpenMathInstruct-2 的 81,059 条唯一题面与 GSM8K test 1319 题做近似重叠审计(>=0.95 命中 0) | i=106, i=108 |
| 116 | C2 | 把 evaluate.py 实际使用的那一份固定 10-shot system context 原样写进 gsm8k_train_context.jsonl 的 system 字段,并与评测日志里的 system 消息逐字比对确认一致 | i=116, i=117, i=114, i=172 |
| 157 | C8 | broad(153K OpenMath2)训练同样 OOM;micro-batch 16→8、accumulation 4→8,有效 batch 64 不变后重启 | i=156, i=157 |
| 169 | C11 | 自写脚本从官方 inspect_ai 评测日志里抽确定性判据:首个 ANSWER 是否正确、多答案条数、stop_reason 分布、输出长度 —— 官方评分器输出的再加工,零噪声、零额外开销 | i=169, i=170 |
| 185 | C1 | 发现重新序列化 tokenizer 会改变其 regex 表示,于是改 train_sft.py 在保存后还原 base 的 tokenizer 文件,使提交目录的分词与评测基线逐字节一致(sha256 核对通过) | i=137, i=183, i=185, i=245 |
| 190 | C3 | 建 self_distill.py:用当前 checkpoint 对 train 题多次采样,只保留数值答案正确且 ANSWER 行干净的轨迹作为新 SFT 数据(C3 第 (d) 类来源) | i=190, i=226 |
| 201 | proposed:pipeline_instru… | 把 train_sft.py 改成无缓冲输出,以便在训练进行中通过 tail/ps 观测进度 —— 不改任何影响分数的量,只改自己管线的可观测性 | i=200, i=201 |
| 345 | C3 | train_sft.py 加 --sources,可按 source 字段只训练数据文件的子集 | i=344, i=345, i=427 |
| 360 | C8 | self_distill.py 因 huggingface.co HEAD 超时崩溃;改为从本地 snapshot 载 tokenizer 并全程 HF_HUB_OFFLINE / HF_DATASETS_OFFLINE | i=353, i=359, i=360 |
| 378 | C8 | 读 transformers 的 _patch_mistral_regex 源码后,改 train_sft.py 归一化保存的 config version,绕开会改写分词器 regex 的检测路径 | i=377, i=378, i=374 |
| 444 | C3 | make_curriculum.py:用当前 checkpoint 在 train 集上的自采样覆盖率(543 题零覆盖、677 题单覆盖)加权造 10,999 行 hard 课程数据 | i=444, i=446 |
| 461 | C3 | prepare_orca.py:引入 Orca-Math 200K 作为新的独立合成数据来源(199,459 行保留) | i=461, i=464 |
| 471 | C4 | train_sft.py 加 --lr-scheduler / --warmup-ratio,以复现 Orca-Math 论文的 constant 1e-6、零 warmup、单 epoch 配方 | i=470, i=471, i=486 |
| 473 | C2 | train_sft.py 加 --solution-only / --plain-question-prompt:对 Orca 数据改用纯问题 prompt、不追加 ANSWER 行,只训 solution + EOS | i=472, i=473, i=916 |
| 496 | C3 | generate_preferences.py:对每道 train 题采样 4 条,数值验证后取正例,配本地错误轨迹或跨题合成负例,产出 7,473 行偏好对 | i=496, i=768 |
| 498 | C4 | 建 train_dpo.py:rank-16 LoRA DPO,冻结参考模型 | i=498, i=669 |
| 510 | C10 | audit_orca_contamination.py:对 Orca 199,459 题与 GSM8K test 1319 题做 TF-IDF cosine + bigram Jaccard 双指标审计,找出 3 对高重叠模板 | i=510, i=512 |
| 519 | C10 | 保守剔除 4 行高重叠(改名换数)样本产出 orca_math_train_clean.jsonl,并为此主动中止已跑约 6 分钟的 orca 训练、明确作废其 checkpoint 后重跑 | i=517, i=519, i=520 |
| 561 | C4 | 建 train_kto.py:LoRA KTO(TRL experimental API) | i=561, i=768 |
| 583 | C9 | 建 package_final.py 作为提交守卫:只挑一个已有候选写进 final_model,剥掉 optimizer/训练产物,还原 canonical tokenizer,并在写入前校验本地可加载 | i=583, i=593 |
| 608 | C6 | 建 interpolate_models.py:对同族 checkpoint 做线性权重插值(model soup) | i=608, i=714 |
| 609 | C6 | soup:gsm_ctx_star 与 gsm_ctx_star2 按 alpha=0.5 平均 -> gsm_ctx_star_star2_half | i=609, i=611 |
| 613 | C6 | soup:gsm_ctx_star 与 gsm_ctx_star_hard 按 alpha=0.25 平均 -> gsm_ctx_star_hard_quarter。该候选造出来后从未被任何一次评测使用(悬空产物) | i=613, i=614 |
| 699 | C4 | 建 train_grpo.py:只用 GSM8K train 题、以精确数值答案 + 格式为 reward 的在线 GRPO(LoRA,训练后 merge) | i=699, i=702 |
| 737 | C6 | soup:gsm_ctx_star 与 gsm_ctx_star_orca_clean 按 alpha=0.5 与 0.25 各出一份;其中 orca_quarter 造出后从未被评测 | i=737, i=738 |
| 753 | C8 | 全量评测撞 vLLM illegal memory access 后,把评测协议从 --max-connections 24 / --max-tokens 4000 / gpu-mem 0.5 降到 12 / 3000 / 0.45,此后所有候选都用新协议评测 | i=750, i=752, i=753 |
| 933 | C2 | train_grpo.py 改成在 RL prompt 里带上评测那一份固定 10-shot system context,让 reward 优化对准真实评测分布 | i=932, i=933 |
| 955 | C8 | GRPO 冒烟暴露:TRL 只用 <\|endoftext\|> 作停止符,而模型以 <\|im_end\|> 收尾,导致 clipped_ratio=1.0、所有 reward 恒 0;改用 Qwen 的对话终止符 | i=948, i=954, i=955 |
| 957 | C1 | train_grpo.py 再打一个补丁:保存时把 generation_config 的 eos_token_id 补回 [151645, 151643] 双停止符(TRL 会把它对齐成单个 151645) | i=956, i=957, i=960, i=1000 |
| 1015 | C2 | add_preference_context.py:给同一批 7,473 条偏好对加上评测的 10-shot system context(prompt 中位 2214 token),train_dpo.py 相应加 max-length / max-prompt-length / max-comp… | i=1015, i=1018, i=1010 |
| 1025 | C8 | context-DPO 在 batch 8 反向时约 90GB 爆显存、无产物;micro-batch 8→4、accumulation 4→8(有效 batch 32 不变)+ expandable_segments 后重启 | i=1024, i=1025 |
| 1105 | C6 | soup:gsm_ctx_star_dpo 与崩溃的 gsm_ctx_star_dpo_context 按 alpha=0.25 平均,用插值把不稳定候选救回可评测状态 | i=1105, i=1099 |
| 1148 | C6 | soup:gsm_ctx_star_dpo 与 gsm_ctx_star_dpo_b05 按 alpha=0.5 平均 | i=1148, i=1147 |
| 1175 | C3 | generate_preferences.py 加 --local-only:每题采 8 条,只保留同题「本模型答对 vs 本模型答错」的硬负例对,剔除原数据里 4,383 条跨题合成负例 | i=1174, i=1175, i=1178 |
| 1205 | C6 | soup:gsm_ctx_star_dpo 与 gsm_ctx_star_dpo_seed43 按 alpha=0.5 平均(两个独立 seed 的同配方端点) | i=1205, i=1201 |
| 1230 | C10 | 偏好数据不变量审计:逐条核对 chosen 的终止 ANSWER 数值等于 train 标答、rejected 不等于标答、无重复对、prompt 可映射回原题;3,814 行 0 违反,并用 assert 硬失败 | i=1230, i=1231 |
| 1265 | C3 | make_hybrid_preferences.py:在保持 7,473 行与 234 步 schedule 完全不变的前提下,把 3,814 行原负例换成同题硬负例(为 C3 造一个严格同规模对照) | i=1265, i=1268, i=1264 |
| 1285 | C3 | make_hybrid_preferences.py 加 --fraction,只替换 25%(954 行),其余 6,519 行保持原偏好集不变 | i=1285, i=1288 |
| 1303 | C9 | 执行提交守卫:把 gsm_ctx_star_dpo(0.5906)写进 final_model,产物只剩 8 个文件、无 optimizer 状态、无符号链接 | i=1303, i=1304 |
| 1350 | proposed:verifier_operat… | 为裁决 b05 与 leader 之间 1 题的差距,对同一批权重扫官方评测器的运行参数(conn2/tok4000 vs conn12/tok3000 vs conn12/tok4000);不改任何候选,却把 final_model 的读数从 771 移到 781 题 | i=1349, i=1350, i=1358 |
| 1361 | C9 | package_final.py 更新:把最终评测的样本量/并发/max_tokens/显存占比与分数一并写进 training_manifest,使报数口径可追溯 | i=1359, i=1361, i=1364 |
| 1363 | C9 | 按新分数重打包并 sha256 核对 final_model/model.safetensors 与所选 checkpoint 逐字节相同,再做文件清单/无符号链接/张量数审计 | i=1363, i=1364, i=1369 |

### 训练序列(44 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 99 | real | 0.01h | returned | **baseline** | baseline:全 run 第一次训练(gsm8k_train 7473 行,4 epoch,lr 3e-5,有效 batch 64)。无上一次可比,取值域缺 `baseline` 档。结局:batch 32 OOM,无 checkpoint |
| 104 | real | 0.11h | returned | **unclear** | 与 i=99 同数据、同 epoch/lr、同有效 batch 64,仅 micro-batch 32→16 / accum 2→4。这是 C8 强制的可行性重启,按「为保持某量不变而做的补偿不计入受测变量」判据,本次不在验任何变量 |
| 152 | real | 0.02h | returned | **both** | 相对 i=104:数据换成 OpenMathInstruct-2 派生的 153,271 行(C3),同时起点从 base 重训、epoch 4→1、lr 3e-5→2e-5(C4)。结局:OOM,无 checkpoint |
| 157 | real | 0.71h | returned | **unclear** | 与 i=152 同数据同 lr 同有效 batch 64,仅 micro-batch 16→8 / accum 4→8;C8 强制的可行性重启,不在验任何变量 |
| 255 | real | 0.08h | returned | **both** | 相对 i=157:起点换成 candidates/broad、数据换回 gsm8k_train 7473 行(C3),epoch 1→3、micro-batch 8→16 / accum 8→4(C4)。是「先广后专」两阶段配方的第二阶段 |
| 287 | real | 0.21h | returned | **unclear** | 受测变量是 **C2 格式对齐**,取值域装不下:起点 gsm_only,题目与解答与 i=104 完全同一批 7473 条,唯一的数据差异是 prompt 加上评测那份固定 10-shot system context;序列长度 284→2343 token 逼着 max-length 1024→… |
| 367 | real | 0.01h | returned | **C3** | 相对 i=287:起点 gsm_ctx,数据加入 self_distill 产出的 13,025 条验证过滤轨迹(gsm8k_train ×2 + 自蒸馏);lr 1e-5→8e-6、max-length 3072→1024 是随数据不带 context 变短的补偿。结局:tokenizer 阶段撞… |
| 371 | real | 0.10h | returned | **C3** | 与 i=367 逐字同数据同超参,只加 HF_HUB_OFFLINE / HF_DATASETS_OFFLINE 与 --tokenizer 本地路径(C8 重启)。它交付的仍是 i=367 那次 C3 实验:自蒸馏数据是否有用 |
| 404 | real | 0.15h | returned | **C3** | 相对 i=371:起点顺推到 gsm_ctx_star,数据再加第二轮自蒸馏 self_distilled_v2(27,971→41,154 行);lr 8e-6→5e-6 是随轮次递减的配套整合率 |
| 413 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 422 | real | 0.12h | returned | **C3** | 相对 i=404:同起点 gsm_ctx_star,把自蒸馏数据换成 OpenMathInstruct-2 的 405B 蒸馏轨迹(用新加的 --sources 过滤),测「谁做教师」;lr 8e-6→4e-6 配套 |
| 448 | real | 0.05h | returned | **C3** | 相对 i=422:同起点 gsm_ctx_star、同 lr 4e-6、同有效 batch 64,只把数据换成按覆盖率加权的 hard 课程 10,999 行。除数据外几乎逐字相同,是一次较干净的 C3 对照 |
| 475 | real | 0.11h | returned | **both** | 相对 i=448:数据换成 Orca-Math 199,459 行(C3),同时改 lr 4e-6→1e-6 + constant scheduler + warmup 0(C4),并改成 --solution-only --plain-question-prompt(C2)。agent 自陈是整体… |
| 479 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 504 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 522 | real | 0.90h | returned | **both** | 与 i=475 逐字相同(同起点/同 lr/同 scheduler/同 batch/同 solution-only),数据仅少 4 行被审计剔除的高重叠样本。它交付的是 i=475 那次 both 实验;真实跑完 0.90h |
| 530 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 543 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 575 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 590 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 600 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 616 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 621 | — | — | — | **unclear** | 机械层误判:该事件是 `ps ... \| rg 'python train_sft.py...'` 轮询监控命令,没有启动任何训练(见 definition_defect d1) |
| 769 | real | 0.27h | returned | **both** | 相对之前所有 SFT:第一次进入偏好学习。数据是新造的 7,473 行 verifier-labeled 偏好对(C3),训练目标从 completion-only SFT 换成 LoRA KTO(C4),二者由该方法的输入格式绑定,不可分。起点 gsm_ctx_star,lr 5e-6 beta … |
| 833 | real | 0.06h | returned | **unclear** | 受测变量是 **C2 格式修复**,取值域装不下:KTO 后模型不再干净终止、评测直接崩,于是用 gsm8k_train_context.jsonl 做 30 步 lr 5e-6 constant 的「精确上下文刷新」把终止行为拉回来。它不在验数据来源也不在验方法,而在修复格式漂移 |
| 846 | real | 0.16h | returned | **C4** | **全 run 最干净的一次 C4 单变量对照**:与 i=769 同起点 candidates/gsm_ctx_star、同数据文件 data/gsm8k_preferences.jsonl、同 lr 5e-6、同 beta 0.3、同 epochs 1、同有效 batch 32(16×2 vs … |
| 877 | real | 0.16h | returned | **both** | 相对 i=846:起点顺推到 gsm_ctx_star_dpo,数据换成用新策略重采样的第二轮偏好对 gsm8k_preferences_dpo.jsonl(C3),同时 lr 5e-6→3e-6(C4,agent 明说是为了「refine rather than overwrite」) |
| 891 | real | 0.06h | returned | **unclear** | 受测变量是 **C2 格式修复**,取值域装不下:与 i=833 同一套 30 步精确上下文刷新,对象换成 dpo2。目的是把 dpo2 的长生成不稳定修回可评测 |
| 901 | — | — | — | **unclear** | 机械层误判:该事件是 `--help` / 读源码,没有启动任何训练(见 definition_defect d1) |
| 906 | real | 0.08h | returned | **C4** | **逐字单变量 C4 对照(epoch 长度)**:与 i=877 同起点、同数据、同 lr 3e-6、同 beta 0.3、同 batch/accum,唯一差别 --epochs 1 → 0.5 |
| 946 | smoke | 0.02h | returned | **smoke** | 冒烟测试,取值域缺 `smoke` 档:GRPO 管线 2 步跑通性验证。结果暴露停止符 bug(clipped_ratio 1.0、reward 全 0),不产出任何可比分数 |
| 959 | smoke | 0.01h | returned | **smoke** | 冒烟测试,取值域缺 `smoke` 档:与 i=946 同参数,验证停止符补丁生效(clipped_ratio 1.0→0.25,出现非零 reward) |
| 968 | real | 0.32h | returned | **C4** | 相对 i=846:同起点 gsm_ctx_star_dpo、同一批 GSM8K train 题目(数据来源不变),把优化方式从离线偏好换成在线 verifier-reward GRPO 60 步(C4) |
| 1020 | real | 0.01h | returned | **unclear** | 受测变量是 **C2 prompt 上下文**,取值域装不下:与 i=846 同起点、同 lr 5e-6、同 beta 0.3、同 epochs、同一批 7,473 条偏好对,唯一差别是每条 prompt 前面加上评测那份 10-shot system context;batch 16→8 / ac… |
| 1025 | real | 0.80h | returned | **unclear** | 与 i=1020 逐字相同,micro-batch 8→4 / accum 4→8(有效 batch 32 不变)+ expandable_segments;C8 强制重启,交付的仍是 i=1020 那次 C2 实验 |
| 1117 | real | 0.08h | returned | **C4** | **逐字单变量 C4 对照(epoch 长度)**:与 i=846 同起点、同数据、同 lr 5e-6、同 beta 0.3、同 batch 16 / accum 2、同 seed 42,唯一差别 --epochs 1 → 0.5 |
| 1131 | real | 0.16h | returned | **C4** | 相对 i=846:同起点同数据同 batch 同 epochs,lr 5e-6→3e-6 且 beta 0.3→0.5;agent 明说是刻意保持 beta×lr 乘积不变,只加强参考模型约束 |
| 1156 | — | — | — | **unclear** | 机械层误判:该事件是 `--help` / 读源码,没有启动任何训练(见 definition_defect d1) |
| 1163 | real | 0.16h | returned | **unclear** | **取值域装不下的一类:同配方随机重复**。与 i=846 逐字相同(同起点/同数据/同 lr/beta/epochs/batch),唯一差别 --seed 42→43。它不在验 C3 也不在验 C4,而是在量同一配方的随机方差(见 boundary_case b4) |
| 1235 | real | 0.08h | returned | **C3** | 相对 i=846:同起点同超参逐字相同(lr 5e-6 / beta 0.3 / epochs 1 / batch 16×2 / seed 42),只把偏好数据换成 --local-only 产出的 3,814 条同题硬负例。非严格单变量:样本量 7473→3814,步数随之减半 |
| 1254 | — | — | — | **unclear** | 机械层误判:该事件是 `--help` / 读源码,没有启动任何训练(见 definition_defect d1) |
| 1258 | real | 0.02h | returned | **unclear** | 受测变量是 **C2 prompt 校准**,取值域装不下:在 leader 上只走 5 步、lr 1e-6 constant 的最小化上下文刷新,用官方 GSM8K train 解答按评测 system context 渲染。agent 自陈比 i=833/891 那次「gentler by tw… |
| 1270 | real | 0.16h | returned | **C3** | **本 run 最干净的 C3 对照**:与 i=846 除数据文件外逐字相同,且刻意保持 7,473 行 / 234 步 schedule 与原胜出 run 完全一致,只把其中 3,814 行的负例换成同题硬负例 |
| 1290 | real | 0.16h | returned | **C3** | 与 i=1270 逐字相同,只把硬负例替换比例从 100%(3,814 行)降到 25%(954 行);和 i=846 / i=1270 一起构成一条 0% / 25% / 100% 的 C3 剂量曲线 |

### 验证序列(37 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 44 | 3.0 | 50.0 | 是 |  | 0.06(50 题基线);agent 同时读回失败模式:base 不会停,续写出第二道题 |
| 124 | 3.0 | 150.0 | 是 | c1, c2, c3 | 0.4866666666666667(150 题) |
| 242 | 3.0 | 150.0 | 是 | c3 | 0.22(33/150);agent 另从日志读出 first_answer_correct 77、多答案 55、max… |
| 279 | 3.0 | 150.0 | 是 | c3 | 0.12(18/150);多答案 103,格式病理比 i=242 更差 |
| 332 | 3.0 | 150.0 | 是 | c7 | 0.5533(83/150);stop_reason 干净 149/150,多答案 2 |
| 391 | 3.0 | 150.0 | 是 | c11 | 0.6333(95/150);150/150 干净终止 |
| 417 | 3.0 | 150.0 | 是 | c11 | 0.58(150 题);判为回退 |
| 429 | 3.0 | 150.0 | 是 | c13, c3 | 0.50(150 题);判为回退 |
| 433 | — | — | 是 | c11 | 0.60,但 --limit 缺省=150,这次名义上的 full eval 实际只跑了 150 题;agent 当场从… |
| 439 | 3.0 | 1319.0 | 是 | c11 | 0.5807429871114481(766/1319,第一份真正的全量分) |
| 452 | 3.0 | 1319.0 | 是 | c16 | 0.5458680818802123(54.6%) |
| 729 | 3.0 | 1319.0 | 是 | c17, c19, c18, c21 | 0.5792(764/1319);并读出分层结果:5+ 步题 32.4%→36.9%、1-2 步题下降 |
| 740 | 3.0 | 1319.0 | 否 | c30 | 未拿到数值。vLLM illegal memory access 崩溃;agent 判为「已知评测器故障」,降并发/降 … |
| 753 | 3.0 | 1319.0 | 是 | c30, c31 | 0.5550(55.5%) |
| 757 | 3.0 | 1319.0 | 否 | c27 | 未拿到数值,但拿到了一个**被立即采用的二值结果**:降并发后仍崩 → agent 判定「不稳定本身即不合格」,直接否决… |
| 825 | 3.0 | 1319.0 | 否 | c24 | 未拿到数值(illegal memory access,scoring 前崩)。骨架记的 0.534 是 kto_ctx… |
| 841 | 3.0 | 1319.0 | 是 | c24, c7 | 0.5344(53.4%) |
| 861 | 3.0 | 1319.0 | 是 | c23, c22 | 0.5905989385898408(779/1319);另读出 stop 1311 / max_tokens 8、平均… |
| 887 | 3.0 | 1319.0 | 否 | c23 | 未拿到数值(长生成不稳定 + kernel fault)。骨架记的 0.547 是 dpo2_ctx30 的分,属机械层… |
| 896 | 3.0 | 1319.0 | 是 | c23, c7 | 0.5473843821076573(54.7%) |
| 938 | 3.0 | 1319.0 | 是 | c23 | 0.5837755875663382(58.4%) |
| 1003 | 3.0 | 1319.0 | 是 | c29, c32, c33, c34 | 0.574677786201668(57.5%);训练侧 reward 一路 62%→86% 却完全没有转移到测试集 |
| 1096 | 3.0 | 1319.0 | 否 | c35 | 未拿到数值(长生成失败模式触发 illegal memory error)。骨架记的 0.566 是 context25… |
| 1107 | 3.0 | 1319.0 | 是 | c35, c37 | 0.5655799848369977(56.6%) |
| 1125 | 3.0 | 1319.0 | 否 | c23 | 未拿到数值,但拿到二值结果并直接据此下结论:half-epoch 候选同样长生成不稳定 → 「后半个 epoch 对稳定… |
| 1144 | 3.0 | 1319.0 | 是 | c23 | 0.5875663381349507(775/1319,58.8%),且稳定 |
| 1150 | 3.0 | 1319.0 | 是 | c38 | 0.5852918877937832(772/1319,58.5%) |
| 1198 | 3.0 | 1319.0 | 是 | c23 | 0.583(58.3%) |
| 1207 | 3.0 | 1319.0 | 是 | c40 | 0.569(56.9%) |
| 1249 | 3.0 | 1319.0 | 否 | c39 | 未拿到数值,但拿到二值结果:同样触发长生成/CUDA 故障 → agent 判为「结构性不稳定、不合格」,直接终止该分支 |
| 1260 | 3.0 | 1319.0 | 是 | c7 | 0.575(57.5%) |
| 1281 | 3.0 | 1319.0 | 是 | c42, c39 | 0.579(57.9%) |
| 1297 | 3.0 | 1319.0 | 是 | c43, c39 | 0.575(57.5%) |
| 1313 | 3.0 | 1319.0 | 是 | c44, c25, c45 | 0.5845337376800607(771/1319)—— 同一份权重,改用 evaluate.py 缺省协议(con… |
| 1335 | 3.0 | 1319.0 | 是 | c45 | 0.5852918877937832(772/1319) |
| 1350 | 3.0 | 1319.0 | 是 | c45 | 0.5807429871114481(766/1319) |
| 1353 | 3.0 | 1319.0 | 是 | c45, c47 | 0.5921152388172858(781/1319)—— 同一份权重的第三个读数;agent 用它推翻了 conn2… |

### 异常与存疑

- **22 段训练的受测变量判不出**:i=[413, 479, 504, 530, 543, 575, 590, 600, 616, 621, 901, 1156, 1254, 104, 157, 287, 833, 891, 1020, 1025, 1163, 1258]
- **7 次验证没有拿到信号**:i=[740, 757, 825, 887, 1096, 1125, 1249]
- **分类学缺口提案 2 条**
  - verifier_operating_point(i=1349, i=1358, i=1358, i=1307)
  - pipeline_instrumentation(i=200, i=201, i=621, i=1322, i=1326)
- **定义缺陷 4 条**
  - (i=413, i=413, i=901, i=1254)
  - (i=828, i=830, i=831, i=845)
  - (i=759, i=1127, i=1251)
  - (i=287, i=338, i=1257, i=833)
- **边界情形 5 条**
  - (i=137, i=245)
  - (i=948, i=954, i=956)
  - (i=103, i=1024, i=274)
  - (i=1163, i=1201)
  - (i=1230, i=1231)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gsm8k_Qwen_Qwen3-4B-Base_17397509
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gsm8k | Qwen_Qwen3-4B-Base | 7.91h | 0.8392721758… |

### 改动序列(67 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 54 | C3 | 选定第一批训练数据来源:官方 GSM8K train(7,473)+ 公开合成语料 clarkkitchen22/SynthGSM8K-50K(先核实其生成器只用 GSM8K train 种子)。 | i=54, i=57 |
| 58 | C2 | 写 prepare_data.py / train_sft.py:训练样本按评测端 ChatML 渲染、末行用评测器的 `ANSWER:` 契约、只对 assistant 段算 loss。 | i=58, i=57 |
| 60 | C10 | 去污染:归一化精确匹配 + TF-IDF 近邻,对全部 GSM8K test 问题做单向排除;剔除 35 条近重复,保留 67,528 条。 | i=60, i=67 |
| 68 | C8 | 修 JSONL 读取:`splitlines()` 把一条含 Unicode 行分隔符的合成样本切成两条,改成逐行读取后重启训练(数据未动)。 | i=68, i=67 |
| 115 | C3 | 加入第二个数据源 MetaMathQA(GSM_Rephrased / GSM_FOBAR / GSM_SV),同样做 test 近重复排除:剔 303 条,得 157,761 条。 | i=115, i=221 |
| 145 | C1 | 新增 configure_inference.py:把导出 checkpoint 的 generation_config.json 的 eos_token_id 由 151643 改成 [151643(<\|endoftext\|>), 151645(<\|im_end\|>)],并写入 do_sa… | i=145, i=144, i=157 |
| 151 | C5 | 在 Trainer 的 save rotation 删掉 checkpoint-120 之前,用硬链接把它另存为 runs/sft_candidates/step120(该 checkpoint 后来成为整条 run 的锚点)。 | i=151, i=160 |
| 155 | C8 | 绕开 transformers 的 GenerationConfig.validate(strict=True):它拒绝 do_sample=False 同时 temperature=0.0。改成先 save_pretrained(temperature/top_p=None)再对 generati… | i=155, i=152, i=504 |
| 234 | C8 | MetaMath 全参训练在一个 8×1024 token 的长批次上 OOM,重启时加 `--optim adamw_bnb_8bit`(8-bit Adam 状态)腾出 ~20GB。 | i=234, i=233 |
| 260 | C3 | 写 prepare_official.py,把官方 GSM8K train 7,473 条按部署格式落成 data/gsm8k_train_all.jsonl。 | i=260, i=263 |
| 305 | C4 | 写 train_grpo.py:LoRA GRPO,奖励 = 精确数值答案 + 很小的格式项;并在第一档单测该奖励函数。 | i=305, i=307 |
| 355 | C3 | 按 type 拆 MetaMath,单独切出前向 GSM_AnsAug 子集(79,847 条,剔 153 条近重复)做一次目标分布阶段。 | i=355, i=360, i=407 |
| 389 | C2 | prepare_official.py --include-eval-system:只用 GSM8K train 复现评测器那份固定 10-shot 系统上下文,做 token 级对齐的训练语料(平均 2,343 token,无截断)。 | i=389, i=392, i=407 |
| 449 | C8 | 修 train_sft.py 存档路径上的 generation config 归一化:`temperature: 0` 让 Transformers 在保存 checkpoint 时序列化失败,checkpoint-300 整个不可用。 | i=449, i=448 |
| 465 | C3 | 再切出 MetaMath 的 GSM_Rephrased 前向子集单独成文件,把导致 loss 恶化的 backward-variable 格式排除在后续实验之外。 | i=465, i=464 |
| 636 | C8 | 修 GRPO 的截断掩码:vLLM 在 `<\|im_end\|>` 上停但不返回该 token,TRL 因此把每条完整回答都判为 truncated,梯度恒为 0;关掉这一处掩码后重启。 | i=636, i=635 |
| 656 | C8 | 修 merge_adapter.py,使 LoRA adapter 能合并成独立可评测的 checkpoint。 | i=656, i=658 |
| 693 | C4 | GRPO 第二版:采样温度提到 1.2、LoRA rank 32、max-steps 100,目的是让容易题也产生对错混合的组,从而有非零优势。数据文件与第一版逐字相同。 | i=693, i=689 |
| 696 | C6 | 写 interpolate_models.py:两个 checkpoint 的权重按 alpha 线性插值输出一个新模型。 | i=696, i=698 |
| 724 | C11 | 从 inspect_ai 日志里读官方 scorer 的逐题记录(match.value=='C'),算出各候选相对锚点的 gain/loss 题号集合与 oracle 上限——确定性、零噪声、从同一次评测里免费得到。 | i=724, i=725 |
| 727 | C6 | 第一组权重插值扫描:step120 × ansaug_step300,alpha 依次 0.5 / 0.25 / 0.75 / 0.625 / 0.875 / 0.70(跨两个不同数据分支,非均匀系数)。 | i=727, i=731, i=794, i=932 |
| 748 | C3 | 写 generate_self_rft.py:用当前最好模型对官方 train 每题采样 n 条解答,只保留精确答案校验通过的 rationale(自生成 + 验证过滤)。 | i=748, i=750 |
| 815 | C3 | 用 interp_ansaug_a075(83.5%)做教师,t=0.9、n=4 采样官方 train 全部 7,473 题,得 26,316 条验证通过的 rationale,覆盖 7,249 题。 | i=815, i=850 |
| 842 | C3 | 写 prepare_balanced_rft.py:每题最多 2 条自生成解 + 1 条人写解,把采样解不出的 224 题用人写 rationale 补回,得 21,693 行。 | i=842, i=845 |
| 878 | C6 | 把教师 interp_ansaug_a075 与 self_rft_v5/checkpoint-25 以 alpha 0.5 插值。 | i=878 |
| 913 | C11 | 把逐题 gain/loss 分析推到全量 1,319 题:balanced 相对教师改了 32 题(18 修好 / 14 变坏),oracle 上限 85.7%。 | i=913, i=929 |
| 919 | C6 | 把教师 interp_ansaug_a075 与 balanced_rft_v6/checkpoint-25 以 alpha 0.5 插值。 | i=919 |
| 924 | C3 | 难度加权变体:人写 rationale 按教师解出该题的频率反比重复,集中在 224 道失败题与低通过率题,得 18,298 行。 | i=924, i=926, i=927 |
| 961 | C8 | 磁盘管理:只删本 run 产生的中间模型目录,保留全部最好 checkpoint 与评测日志,腾出继续开分支的空间。 | i=961, i=956 |
| 973 | C6 | 把 step120 与 rephrased_v8/checkpoint-100 以 alpha 0.75 插值,检验改述分支能否并入成功盆地。 | i=973 |
| 979 | proposed:stochastic-repl… | 同一配方只换随机种子重复训练(v6 seed 27182 → v9 seed 4242;后来 polish 也做了 seed 20260716 / 424242 两次),目的不是换数据也不是换超参,而是测方差、并制造互补错误供权重平均。注意 max-steps 75 × 有效 batch 128 =… | i=979, i=978, i=1401 |
| 996 | C6 | 把两个只差随机种子的 balanced-RFT checkpoint 平均(v6 ckpt-25 × v9 ckpt-25, alpha 0.5)——这一条是严格意义上的 model soup。 | i=996 |
| 1008 | C11 | 从官方 log 里把判错的样本连题面/模型输出一起打印出来做失败归因(分清语义理解错 vs 格式错)。 | i=1008, i=1024 |
| 1025 | C3 | 引入外部语料 microsoft/orca-math-word-problems-200k,理由是它的教师解答正对着刚定位到的语义理解弱点。 | i=1025, i=1024 |
| 1027 | C10 | 对 Orca 语料做同一套去污染:200,035 行保留 199,218,剔 206 条近 test 匹配及不可用行。 | i=1027, i=1038 |
| 1047 | C8 | Orca 训练首个长上下文批次 OOM(未走到任何 optimizer step),把 --max-length 从 1024 降到 512 重启。 | i=1047, i=1046 |
| 1077 | C6 | 对 Orca 方向做「任务向量阻尼」并做系数搜索:champion × orca_ckpt-25,alpha 先 0.25,随后在全量上扫 0.05/0.10/0.15/0.20/0.125/0.175/0.14/0.145/0.155。峰值锐利地落在 0.15。 | i=1077, i=1236, i=1253, i=1276 |
| 1099 | C3 | 配方变体:每题人写解重复 2 次(--official-repeats 2),同时仍保留 2 条自生成解。 | i=1099, i=1098 |
| 1115 | C3 | 配方变体:每题只留 1 条(最短的)自生成解 + 1 条人写解。 | i=1115, i=1114 |
| 1124 | C3 | 第二轮拒绝采样:改用已提升的模型自采样,得 26,289 条验证 rationale,覆盖 7,251/7,473 题。 | i=1124, i=1129 |
| 1146 | proposed:artifact-side-p… | 把一段固定 system prompt(“You are a meticulous grade-school…”)注入**提交模型自己的** chat_template.jinja——不改权重、不改训练数据、不改 generation_config、也不改评测端模板。 | i=1146, i=1151, i=1137 |
| 1150 | C8 | `cp -al` 建的是硬链接,上一条 prompt 注入把改动写回了受保护的冠军 checkpoint;断链并从 interp_ansaug_a075 恢复其 chat_template。 | i=1150, i=1151 |
| 1162 | C4 | 扩展 generate_self_rft.py 产出 DPO 偏好对:同一题上模型采样错的解作 rejected、人写解作 chosen。 | i=1162, i=1164 |
| 1167 | C4 | 写 train_dpo.py:参数高效 DPO,用当前 checkpoint 自身作隐式 reference,避免同时驻留第二个 4B 模型。 | i=1167, i=1172 |
| 1190 | C6 | 把 DPO 方向阻尼到 25% 再评(直接用 DPO 合并模型时掉到 124/150)。 | i=1190, i=1189 |
| 1207 | C3 | 改用本地缓存的冻结 Qwen2.5-Math-7B-Instruct 作离线教师,只在官方 GSM8K train 问题上推理;明确不碰缓存里被禁的 Qwen3-4B instruct。 | i=1207, i=1206 |
| 1213 | C8 | 修拒绝采样校验器:Qwen2.5-Math 用 \\boxed{} 而不是 `ANSWER:`,校验器因此一条都没收;加 BOXED_RE 并把接受的 trace 归一到部署答案格式,重跑后收下 6,003 条(80.3% 题)。 | i=1213, i=1215, i=1212 |
| 1230 | C6 | 写 compose_task_vector.py:`base + scale × (tuned − origin)`,三个模型的任务向量合成(不是同轨迹 checkpoint 平均)。 | i=1230, i=1312 |
| 1232 | C6 | 用任务向量合成把 7B 教师分支以 0.25 的尺度加到最好模型上,保住已有增益而不是把它平均掉。 | i=1232, i=1229 |
| 1302 | C11 | 对**同一份权重**重跑一次全量评测测评测器可重复性:1,110 vs 1,105,据此把单次差异按噪声处理。 | i=1302, i=1310 |
| 1314 | C6 | 把第二轮 self-RFT 分支以 0.05 尺度合成到 interp_orca_a015 上。 | i=1314 |
| 1336 | C4 | 对已保留的 interp_orca_a015 做极低学习率(2e-7)、10 步的 “polish”:语料沿用 balanced self-RFT,只改学习率/步数/warmup,目的是加宽答案 margin 而不擦掉 Orca 增益。 | i=1336, i=1319 |
| 1363 | C6 | 把 interp_orca_a015 与 5 步 polish checkpoint 插值并做系数搜索:0.5 / 0.25 / 0.75 / 0.40 / 0.60,以及最后的 0.49 / 0.51 局部扫描;0.50 三次复现 1,111。 | i=1363, i=1392, i=1581, i=1591 |
| 1407 | C6 | 对另外两个独立种子的 polish checkpoint 做同样的 0.5 平均(seed2 / seed3),检验数据顺序是否可复制。 | i=1407, i=1416 |
| 1422 | C3 | 最后一次针对性训练:只用官方 GSM8K train 的人写解(data/gsm8k_train_all.jsonl),学习率再降到 1e-7、5 步,理由是这种 rationale 风格最贴评测器。 | i=1422, i=1421 |
| 1430 | C6 | 把 interp_polish_a05 与 official-only polish checkpoint 以 0.5 插值。 | i=1430 |
| 1445 | C11 | 对留下的 208 道错题做确定性审计:统计 match 取值分布、是否都产出了合法 `ANSWER:` 行、有没有撞 token 上限——结论是零格式错、零截断。 | i=1445, i=1464 |
| 1457 | proposed:artifact-side-p… | 第二次改提交模型自己的 chat_template.jinja,这次在 user 轮末尾追加一句校验指令(suffix 变体)。 | i=1457, i=1464 |
| 1484 | C3 | 引入 open-r1/OpenR1-Math-220k 的 Algebra 文字题子集作为最后一个正交外部语料(扫 93,733,留 11,113)。 | i=1484, i=1487 |
| 1486 | C10 | 对 OpenR1 子集做同一套去污染审计:exact 0 条、near 0 条,保留全部 11,113 行(阈值 0.68,最大保留相似度 0.6747)。 | i=1486, i=1487 |
| 1496 | C8 | OpenR1 训练在 batch 8 上 OOM,改 batch 4 + max-length 768 重启(明确声明不改数据与模型)。 | i=1496, i=1495 |
| 1501 | C6 | 把 OpenR1 分支以 0.25、随后 0.10 的尺度阻尼后并入领先模型。 | i=1501, i=1507 |
| 1516 | C9 | 提交守卫:把 runs/interp_polish_a05(重复三次都是 1,111)整目录拷成 final_model,再逐分片 sha256 比对 + 校验 index 与 tokenizer,然后对 final_model 本身再跑一次全量。 | i=1516, i=1516, i=1526 |
| 1531 | proposed:inference-numer… | 把提交副本 config.json 里的推理 dtype 从 bfloat16 改成 float16(权重逐位不变、数据不变、generation_config 不变),全量分数 1,111 → 1,115。 | i=1531, i=1526, i=1537 |
| 1547 | proposed:inference-numer… | 同一手法换成 float32 作稳定性检查;结果回落到 1,104,最终保留 BF16。 | i=1547, i=1568 |
| 1569 | C6 | 最后一次正交尝试:把 7B 教师分支的任务向量以 ±0.025 的极小尺度加到最好模型上(两个方向都试)。 | i=1569, i=1574, i=1580 |
| 1592 | C9 | 冻结 BF16 导出:核对 final_model 的 config/generation_config 与 chat_template 与源目录一致、evaluate.py 与 templates/ 的 sha256 未被改动、删掉全部落选候选,并写 RESULTS.md 审计。 | i=1592, i=1595, i=1591 |

### 训练序列(30 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 64 | real | 0.00h | returned | **baseline** | baseline —— 本 run 第一次训练。数据 = 官方 GSM8K train + SynthGSM8K-50K 去污染后的 67,528 条;超参 lr 2e-5 / bs 8 × accum 16 / 1 epoch / save-steps 120。没有前一次可比,`tested_va… |
| 69 | real | 0.51h | returned | **unclear** | 与 i=64 的命令逐字相同(仅前面多一句 py_compile)。数据与超参一个字都没改,改的是 train_sft.py 的 JSONL 读取器(C8)。因此它不在验 C3 也不在验 C4。这次跑满 29m53s,产出 checkpoint-120/360/480/528。 |
| 208 | real | 0.10h | returned | **both** | vs i=69:数据 67,528(synth+official)→ 157,761 条 MetaMath(Rephrased/FOBAR/SV);初始权重 base → runs/sft_candidates/step120;lr 2e-5 → 1e-5;save/eval-steps 120 →… |
| 234 | real | 0.39h | returned | **both** | i=208 的重启:命令逐字相同,只多了 `--optim adamw_bnb_8bit`(C8,为 OOM 腾显存)。被测的实验没变,仍是「MetaMath 数据 + 更低 lr」这一组合,故继承 both。 |
| 453 | real | 0.18h | returned | **both** | vs i=234:数据 157,761 全类型 MetaMath → 79,847 条纯前向 GSM_AnsAug;lr 1e-5 → 3e-6;save/eval-steps 300 → 150。agent 自述是「换更干净的前向答案课程 + 论文建议的更低学习率」,两件事一起改。 |
| 520 | real | 0.17h | returned | **C4** | vs i=453:数据文件、lr、batch、accum、optim 逐字相同,只加 `--resume-from-checkpoint runs/sft_ansaug_v3/checkpoint-150`,把同一次训练从 150 步续到 300 步。受测变量是训练步数,纯 C4,而且是本 run … |
| 571 | real | 0.21h | returned | **both** | vs i=520:数据 → data/gsm8k_train_eval_context.jsonl(评测器那份固定 10-shot 上下文);bs 8×accum 16 → 2×64(有效 batch 都是 128,属长序列的补偿)、max-length → 3072、开 gradient-chec… |
| 609 | real | 0.01h | returned | **both** | vs i=571:训练方法 SFT → LoRA GRPO(train_grpo.py,精确答案奖励),同时数据换成 data/metamath_rephrased_train.jsonl。方法与数据同时变。这次 40 秒即失败。agent 自己称这一步为 “GRPO smoke step”,但产物… |
| 624 | real | 0.03h | returned | **both** | i=609 的重启:CLI 与 i=609 逐字相同(仅去掉 PYTORCH_CUDA_ALLOC_CONF、前置 rm -rf),改的是 train_grpo.py 本身。此次被 agent 在 i=631 手动 kill(诊断出 TRL 把全部回答判为 truncated、梯度恒零)。继承 bo… |
| 637 | real | 0.09h | returned | **both** | i=624 的第二次重启:CLI 逐字相同,改的仍是 train_grpo.py(关掉截断掩码,C8)。这次跑满 50 步、梯度非零,产出 checkpoint-25/50。继承 both。 |
| 693 | real | 0.22h | returned | **C4** | vs i=637:`--train-file data/metamath_rephrased_train.jsonl` 与初始权重逐字相同,变的是 `--temperature 1.2`(新增)、`--lora-rank 32`(新增)、`--max-steps 50 → 100`、`--seed … |
| 838 | real | 0.11h | returned | **C3** | 回到 SFT 分支。agent 的对照对象是它的初始权重 interp_ansaug_a075 本身(i=869 明确用 “versus 86.7% for its teacher” 做比较),两者之间唯一的差别就是这批 self-RFT 数据。相对上一次 SFT(i=571)的超参差异(bs 2→… |
| 884 | real | 0.09h | returned | **both** | vs i=838:数据 self_rft_a075_t09_n4(26,316,纯自生成)→ self_rft_balanced_n2(21,693,补回 224 道未解题的人写解);同时 max-steps 100 → 75、seed 31415 → 27182。步数变化改变了 cosine 调度… |
| 930 | real | 0.06h | returned | **both** | vs i=884:数据 → self_rft_hard_weighted(18,298,按教师通过率反比重复人写解);同时 lr 1e-6 → 7.5e-7、max-steps 75 → 50、seed 27182 → 16180。 |
| 950 | real | 0.11h | returned | **both** | vs i=930:开了一条独立分支——初始权重回到 runs/sft_candidates/step120、数据换成 metamath_rephrased_train.jsonl、lr 7.5e-7 → 3e-6、max-steps 50 → 200、save-steps 25 → 100。数据、起… |
| 979 | real | 0.04h | returned | **unclear** | vs i=884:除 `--seed 27182 → 4242` 外,初始权重、数据文件、lr、batch、accum、max-length、max-steps、warmup 全部逐字相同。种子既不在 C3 的定义(数据从哪来、按什么比例混)里,也不在 C4 的枚举(全参/LoRA、SFT/DPO/… |
| 1034 | real | 0.06h | returned | **C3** | vs i=979:数据换成外部语料 data/orca_math_safe.jsonl(Orca-Math 去污染后 199,218 行),初始权重换成当时的冠军 sft_balanced_rft_v6/checkpoint-25。lr 1e-6、batch 8×16、max-steps 75、ma… |
| 1047 | real | 0.16h | returned | **C3** | i=1034 的重启:只把 `--max-length 1024 → 512` 并加 PYTORCH_CUDA_ALLOC_CONF(C8 显存补偿),数据、lr、batch、accum、steps、seed 全部逐字相同。被测变量仍是 Orca 语料。 |
| 1104 | real | 0.04h | returned | **C3** | vs i=884(v6):`--model runs/interp_ansaug_a075`、lr 1e-6、batch 8×16、max-length 1024、max-steps 75、warmup 0.05、**seed 27182** 逐字相同,唯一差别是训练文件 self_rft_bala… |
| 1115 | real | 0.04h | returned | **C3** | 同上:超参与 seed 27182 与 v6/v11 逐字相同,只换训练文件为 self_rft_balanced_n1.jsonl(每题只留 1 条最短自生成解)。C3 单变量。 |
| 1130 | real | 0.04h | returned | **C3** | 同上:超参与 seed 27182 逐字相同,只换成第二轮拒绝采样得到的 self_rft_round2_balanced_n2.jsonl。agent 自述是「先用同一个共同教师隔离数据质量」。C3 单变量。 |
| 1175 | real | 0.07h | returned | **both** | vs i=1130:训练方法 SFT → LoRA DPO(train_dpo.py, beta 0.1, lora-rank 32, batch 2×16, lr 5e-6),数据类型同时从 SFT 语料变成偏好对 dpo_preferences_train.jsonl。方法与数据在 DPO 这里… |
| 1222 | real | 0.04h | returned | **C3** | vs i=1175 之前的同族对照(v6/v11/v12/v13):`--model runs/interp_ansaug_a075`、lr 1e-6、batch 8×16、max-length 1024、max-steps 75、warmup 0.05、seed 27182 逐字相同,只换训练文件… |
| 1320 | — | — | — | **unclear** | **这不是一次训练启动。** 该命令里的 train_sft.py 调用是 `python train_sft.py --help`,后面接 wc -l 与 rg;下一事件 i=1321 返回的就是 argparse 的 usage 文本。骨架把它记成一次 real 训练(产物「—」、时长 0.00… |
| 1336 | real | 0.02h | returned | **C4** | vs 同语料的 v6(i=884):训练文件同为 data/self_rft_balanced_n2.jsonl,变的是初始权重(→ runs/interp_orca_a015)、lr 1e-6 → 2e-7、max-steps 75 → 10、save-steps 25 → 5、warmup 0.… |
| 1402 | real | 0.02h | returned | **unclear** | vs i=1336:初始权重、数据文件、lr 2e-7、batch 8×16、max-length 1024、warmup 0.0 全部逐字相同;差别是 `--max-steps 10 → 5`(而被比较的正是 step-5 checkpoint,i=1336 也已经存过 step-5)与 `--s… |
| 1413 | real | 0.02h | returned | **unclear** | vs i=1402:除 `--seed 20260716 → 424242` 外命令逐字相同。第三个种子副本。 |
| 1422 | real | 0.01h | returned | **both** | vs i=1413:训练文件 self_rft_balanced_n2 → data/gsm8k_train_all.jsonl(只有官方人写解);同时初始权重换成 runs/interp_polish_a05、lr 2e-7 → 1e-7。数据与学习率同时动。 |
| 1489 | real | 0.01h | returned | **both** | vs i=1422:数据换成外部 data/openr1_word_safe.jsonl(OpenR1-Math Algebra 子集 11,113 行);同时 lr 1e-7 → 1e-6、max-steps 5 → 15、warmup 0.0 → 0.05。首个批次 OOM,零 step。 |
| 1496 | real | 0.03h | returned | **both** | i=1489 的重启:`--batch-size 8 → 4`、`--max-length 1024 → 768`(C8 显存补偿,有效 batch 因此从 128 降到 64),数据、lr、steps、seed 不变。agent 明说「只改吞吐/上下文处理,不改数据与模型」。被测实验未变,继承 b… |

### 验证序列(89 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 15 | 3.0 | 32.0 | 是 |  | 0.5(base 模型基线,32 题,tool_result 直接返回) |
| 187 | 3.0 | 64.0 | 是 | c1, c2, c3, c5, c7 | 0.797 |
| 191 | 3.0 | 64.0 | 是 | c1, c2, c3, c5, c7 | 0.75 |
| 194 | 3.0 | 64.0 | 是 | c1, c2, c3, c5, c7 | 0.797 |
| 513 | 3.0 | 64.0 | 是 | c11 | 0.781 |
| 563 | 3.0 | 64.0 | 是 | c11 | 0.797 |
| 568 | 3.0 | 150.0 | 是 | c7 | 0.833(83.3% 锚点) |
| 604 | 3.0 | 64.0 | 是 | c12 | 0.797 |
| 661 | 3.0 | 64.0 | 是 | c10, c15, c16 | 0.781 |
| 661 | 3.0 | 64.0 | 是 | c10, c15, c16 | 0.781 |
| 661 | 3.0 | 64.0 | 是 | c10, c15, c16 | 0.812 |
| 661 | 3.0 | 64.0 | 是 | c10, c15, c16 | 0.812 |
| 667 | 3.0 | 150.0 | 是 | c10, c15 | 0.827 |
| 671 | 3.0 | 150.0 | 是 | c7 | 0.76 |
| 677 | 3.0 | 150.0 | 否 |  | 未拿到 —— 这次 evaluate.py 根本没执行:命令误用 `--output-file`,i=678 返回 ar… |
| 681 | 3.0 | 150.0 | 是 | c11 | 0.827 |
| 685 | 3.0 | 150.0 | 是 | c12 | 0.82 |
| 774 | 3.0 | 64.0 | 是 | c19 | 0.766 |
| 778 | 3.0 | 64.0 | 是 | c19 | 0.828 |
| 783 | 3.0 | 150.0 | 是 | c19 | 0.867(130/150,新的最好) |
| 792 | 3.0 | 64.0 | 是 | c17 | 0.797 |
| 801 | 3.0 | 150.0 | 是 | c19 | 0.847 |
| 803 | 3.0 | 150.0 | 是 | c19 | 0.833 |
| 811 | 4.0 | -1.0 | 是 | c19 | 0.835(1,102/1,319,第一次全量) |
| 863 | 3.0 | 150.0 | 是 | c21, c22 | 0.86 |
| 866 | 3.0 | 150.0 | 是 | c21, c22 | 0.847 |
| 870 | 4.0 | -1.0 | 是 | c21, c22 | 0.836(1,103/1,319) |
| 880 | 3.0 | 150.0 | 是 | c24 | 0.86 |
| 900 | 3.0 | 150.0 | 是 | c23 | 0.867 |
| 903 | 3.0 | 150.0 | 是 | c23 | 0.84 |
| 906 | 4.0 | -1.0 | 是 | c23 | 0.839(1,106/1,319,新冠军) |
| 921 | 3.0 | 150.0 | 是 | c25 | 0.847 |
| 942 | 3.0 | 150.0 | 是 | c27 | 0.853 |
| 945 | 3.0 | 150.0 | 是 | c19 | 0.853 |
| 970 | 3.0 | 150.0 | 是 | c67 | 0.84 |
| 975 | 3.0 | 150.0 | 是 | c28 | 0.82 |
| 982 | 3.0 | 150.0 | 是 | c64 | 0.86(129/150) |
| 999 | 3.0 | 150.0 | 是 | c29 | 0.847(127/150) |
| 1065 | 3.0 | 150.0 | 否 |  | 未拿到 —— evaluate.py 未执行:同一条命令里 `configure_inference.py` 先抛 Ty… |
| 1071 | 3.0 | 150.0 | 是 | c31, c32, c33 | 0.84(126/150) |
| 1077 | 3.0 | 150.0 | 是 | c34 | 0.873(131/150) |
| 1082 | — | — | 是 | c34 | 0.88(132/150;该命令漏了 --limit,evaluate.py 默认 150) |
| 1086 | 3.0 | 1319.0 | 是 | c34 | 0.836(1,103/1,319) |
| 1093 | 3.0 | 1319.0 | 是 | c64 | 0.837(1,104/1,319) |
| 1111 | 3.0 | 150.0 | 是 | c35 | 0.853(128/150) |
| 1117 | 3.0 | 150.0 | 是 | c36 | 0.853(128/150) |
| 1132 | 3.0 | 150.0 | 是 | c37 | 0.853(128/150) |
| 1148 | 3.0 | 150.0 | 是 | c38 | 0.86(129/150) |
| 1185 | 3.0 | 150.0 | 是 | c40, c41 | 0.827(124/150) |
| 1190 | 3.0 | 150.0 | 是 | c42 | 0.853 |
| 1226 | 3.0 | 150.0 | 是 | c43, c44 | 0.847(127/150) |
| 1232 | 3.0 | 150.0 | 是 | c46 | 0.86(129/150) |
| 1236 | 3.0 | 1319.0 | 是 | c34 | 0.839(1,107/1,319) |
| 1245 | 3.0 | 1319.0 | 是 | c34 | 0.839(1,106/1,319) |
| 1253 | 3.0 | 1319.0 | 否 |  | 未拿到 —— evaluate.py 未执行:同命令里 `configure_inference.py --model-… |
| 1257 | 3.0 | 1319.0 | 否 |  | 未拿到 —— 同上,`--output-file` 再次被 argparse 拒绝(i=1259)。骨架同样错关联了 i… |
| 1261 | 3.0 | 1319.0 | 是 | c34 | 0.842(1,110/1,319,Orca 系数的峰值) |
| 1270 | 3.0 | 1319.0 | 是 | c34 | 0.838(1,105/1,319) |
| 1280 | 3.0 | 1319.0 | 是 | c34 | 0.836 = 1,103/1,319 |
| 1284 | 3.0 | 1319.0 | 是 | c34 | 0.835 = 1,102/1,319 |
| 1289 | 3.0 | 1319.0 | 是 | c34 | 0.838 = 1,105/1,319 |
| 1291 | 3.0 | 1319.0 | 是 | c34 | 0.837 = 1,104/1,319 |
| 1295 | 3.0 | 1319.0 | 否 | c34 | 1,106/1,319(acc 0.8385140257771039)—— 输出重定向进 eval_interp_orc… |
| 1302 | 3.0 | 1319.0 | 否 | c34, c47 | 1,105/1,319(acc 0.8377558756633814),同一份权重与 i=1261 的 1,110 相差… |
| 1314 | 3.0 | 1319.0 | 否 | c48 | 1,098/1,319(acc 0.8324488248673237),在 i=1317 读回 |
| 1346 | 3.0 | 1319.0 | 否 | c65 | 1,107/1,319(acc 0.8392721758908264),在 i=1349 读回 |
| 1352 | 3.0 | 1319.0 | 否 | c65 | 1,103/1,319(acc 0.8362395754359363),在 i=1355 读回 |
| 1357 | 3.0 | 1319.0 | 否 | c65, c47 | 1,108/1,319,同权重重复,在 i=1360 读回 |
| 1363 | 3.0 | 1319.0 | 是 | c49 | 1,111/1,319(acc 0.8423047763457164),在 i=1366 读回 |
| 1369 | 3.0 | 1319.0 | 是 | c49, c47 | 1,111/1,319,重复一次仍是 1,111,在 i=1372 读回 |
| 1375 | 3.0 | 1319.0 | 否 | c49 | 1,103/1,319,在 i=1378 读回 |
| 1380 | 3.0 | 1319.0 | 否 | c49 | 1,111/1,319,在 i=1383 读回 |
| 1386 | 3.0 | 1319.0 | 否 | c49, c47 | 1,105/1,319,重复后掉 6 题,在 i=1389 读回 |
| 1392 | 3.0 | 1319.0 | 否 | c49 | 1,105/1,319,在 i=1395 读回 |
| 1396 | 3.0 | 1319.0 | 否 | c49 | 1,102/1,319,在 i=1399 读回 |
| 1407 | 3.0 | 1319.0 | 否 | c50, c64 | 1,103/1,319,在 i=1410 读回 |
| 1416 | 3.0 | 1319.0 | 否 | c50, c64 | 1,101/1,319,在 i=1419 读回 |
| 1425 | 3.0 | 1319.0 | 否 | c66 | 1,109/1,319,在 i=1428 读回 |
| 1430 | 3.0 | 1319.0 | 否 | c51 | 1,109/1,319,在 i=1433 读回 |
| 1459 | 3.0 | 1319.0 | 否 | c53 | 1,108/1,319,在 i=1462 读回 |
| 1501 | 3.0 | 1319.0 | 否 | c54, c55, c56, c57 | 1,108/1,319,在 i=1504 读回 |
| 1507 | 3.0 | 1319.0 | 否 | c54, c55, c56, c57 | 1,106/1,319,在 i=1510 读回 |
| 1519 | 3.0 | 1319.0 | 是 | c58 | 1,111/1,319(0.8423047763457164),final_model 第三次复现同一数值,在 i=15… |
| 1532 | 3.0 | 1319.0 | 否 | c59 | 1,115/1,319(acc 0.8453373768006065),在 i=1535 读回 |
| 1538 | 3.0 | 1319.0 | 否 | c59, c47 | 1,111/1,319,fp16 第二次,在 i=1541 读回 |
| 1549 | 3.0 | 1319.0 | 是 | c60 | 1,104/1,319(0.837),在 i=1565 与后续读回 |
| 1562 | 3.0 | 1319.0 | 否 | c59, c47 | 1,108/1,319,fp16 第三次(三次 1,115/1,111/1,108),在 i=1565 读回 |
| 1569 | 3.0 | 1319.0 | 否 | c61 | 1,107/1,319,在 i=1572 读回 |
| 1574 | 3.0 | 1319.0 | 否 | c61 | 1,104/1,319,在 i=1577 读回 |
| 1581 | 3.0 | 1319.0 | 否 | c49 | 1,110/1,319,在 i=1584 读回 |
| 1585 | 3.0 | 1319.0 | 否 | c49 | 1,100/1,319,在 i=1588 读回 |

### 异常与存疑

- **5 段训练的受测变量判不出**:i=[69, 979, 1320, 1402, 1413]
- **29 次验证没有拿到信号**:i=[677, 1065, 1253, 1257, 1295, 1302, 1314, 1346, 1352, 1357, 1375, 1380, 1386, 1392, 1396, 1407, 1416, 1425, 1430, 1459, 1501, 1507, 1532, 1538, 1562, 1569, 1574, 1581, 1585]
- **分类学缺口提案 4 条**
  - weight-space-model-editing(i=1077, i=1230, i=1312, i=791, i=1276)
  - inference-numeric-precision(i=1531, i=1537, i=1568)
  - artifact-side-prompt-injection(i=1146, i=1151, i=1457, i=1464)
  - stochastic-replication(i=979, i=978, i=845, i=1412)
- **定义缺陷 5 条**
  - (i=1320, i=1321)
  - (i=677, i=678, i=1255, i=1066)
  - (i=635, i=609, i=631)
  - (i=996, i=1004, i=1368)
  - (i=1295, i=1298, i=1299)
- **边界情形 5 条**
  - (i=979, i=845, i=884)
  - (i=1212, i=1215, i=1221)
  - (i=857, i=177)
  - (i=234, i=1047, i=1496, i=1495)
  - (i=1302, i=1310)

## codex_non_api_max_gpt-5.6-sol_10h_run1__gsm8k_google_gemma-3-4b-pt_17397508
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | gsm8k | google_gemma-3-4b-pt | 9.81h | 0.6345716451… |

### 改动序列(35 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 77 | C3 | Stage-1 data source chosen: the 240k GSM-derived rows of meta-math/MetaMathQA (GSM_AnsAug / GSM_Rephrased / GSM_FOBAR / GSM_SV), stratified, 100k samp… | i=68, i=76, i=77 |
| 77 | C2 | Second training stage `aligned` written into train_sft.py: it re-renders the evaluator's own fixed 10-shot prefix and `render_prompt` reproduces templ… | i=16, i=76, i=299 |
| 86 | C4 | Per-device batch size raised 4 -> 16 for the metamath stage after a 256-example memory/throughput probe; adopted for the 100k stage-1 run. | i=88, i=92 |
| 93 | C3 | Stage-1 recipe instantiated: 100,000 stratified GSM-derived MetaMathQA examples, 1 epoch, max-length 1024, with a deliberate midpoint checkpoint at st… | i=93, i=92 |
| 132 | C1 | train_sft.py patched to stamp the saved generation_config.json of every checkpoint with `temperature: 0.0` while keeping `do_sample: true` (transforme… | i=120, i=129, i=130, i=135, i=132 |
| 192 | proposed:env-repair | Processor loading repointed from the Hub resolver to the immutable local snapshot path (/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots… | i=179, i=180, i=192, i=194 |
| 194 | C1 | For stage1_model and stage1_model/checkpoint-3125 only, the agent deliberately UNSET temperature (`if getattr(g,'temperature',None)==0.0: g.temperatur… | i=174, i=194, i=891 |
| 220 | C2 | After the 4.0% eval showed the model answering correctly and then continuing into the next few-shot problem, train_sft.py's target format was patched … | i=205, i=218, i=220, i=221 |
| 226 | proposed:env-repair | HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 prefixed onto every subsequent training launch after the first trial_align run died on a huggingface.co ReadTim… | i=224, i=226 |
| 231 | C4 | First aligned-stage training: lr 3e-6, bs 4 x accum 4, max-length 3072, 1 epoch over all 7,473 aligned examples, initialised from stage1_model. | i=231, i=230 |
| 290 | C4 | Second aligned epoch on the identical 7,473-example corpus, only lr 3e-6 -> 1.5e-6 and seed 42 -> 43, continued from aligned_model. The agent fixed th… | i=290, i=322 |
| 309 | C3 | train_sft.py extended with `--metamath-offset` so a later run can consume the 140k GSM-derived MetaMathQA variants that stage 1 did not see, with no o… | i=308, i=309, i=311 |
| 348 | C4 | Third aligned epoch, same corpus, lr halved again 1.5e-6 -> 7.5e-7, seed 44, continued from aligned2_model. | i=348, i=347 |
| 385 | C3 | Diversity continuation from aligned2_model on the 140,000 previously unseen GSM-derived MetaMath variants (--metamath-offset 100000). Note the learnin… | i=385, i=384 |
| 560 | C3 | Full-dose recovery: one exact-format aligned epoch on top of diverse_model (lr 2e-6, seed 45) so the 140k-variant arm becomes measurable on the same 1… | i=560, i=558 |
| 612 | C5 | The half-dose branch is continued from diverse_model/checkpoint-4375 (the 4375/8750-step midpoint) instead of the endpoint diverse_model; the recovery… | i=612, i=614 |
| 667 | C4 | Second cleanup epoch on the new best, same 7,473 corpus, lr 2e-6 -> 1e-6, seed 45 -> 46, explicitly mirroring the earlier lr-halving epoch that took 8… | i=667, i=666 |
| 772 | C6 | interpolate_models.py added: linear weight-space interpolation between two checkpoints with an arbitrary alpha (accepting negative alpha), writing a f… | i=772, i=774 |
| 774 | C6 | First merge produced: aligned2 x aligned3 at alpha 0.25 (aligned23_a025_model). Never evaluated - the agent moved on after the full-set scores reorder… | i=774, i=776 |
| 785 | C7 | Official scorer read out of inspect_evals/gsm8k/gsm8k.py (prompt text and the `match` scorer) so that the self-training generator's outcome filter val… | i=785, i=786, i=818 |
| 788 | C3 | Fourth data source added: self-generated + verification-filtered. generate_selftrain.py samples 4 candidates per allowed GSM8K train question at tempe… | i=788, i=815, i=832, i=834 |
| 798 | C6 | 50/50 soup of aligned_model and aligned2_model (aligned12_a050_model). | i=798, i=814 |
| 835 | C4 | Self-training SFT hyperparameters: lr 1e-6, bs 4 x accum 4, 1 epoch over the 6,248 filtered rows, initialised from the 62.62% incumbent aligned_model. | i=835, i=834 |
| 835 | C5 | Dense checkpointing of the self-training run (--save-steps 98 over 391 steps) so quarter/half/three-quarter/endpoint candidates can be selected after … | i=835, i=824, i=858 |
| 868 | C6 | alpha 0.25 interpolation of aligned_model toward selftrained_model/checkpoint-98 - a deliberate 'tiny update' worth ~6.25% of a self-training epoch, b… | i=868, i=872 |
| 879 | C6 | alpha = -0.25 NEGATIVE extrapolation away from aligned2_model, i.e. a step backwards along the aligned1->aligned2 overfitting direction. Not an averag… | i=879, i=884 |
| 893 | C6 | 90/10 soup of aligned_model with its pre-alignment ancestor stage1_model - a partial rollback toward the diverse MetaMath checkpoint. This is the dire… | i=893, i=898 |
| 901 | C6 | aligned1 x aligned2 at alpha 0.25 (aligned12_a025_model) - built as a tighter search than the 0.5 soup, but never evaluated; the run's remaining eval … | i=901, i=904 |
| 908 | C6 | alpha sweep continues: aligned1 x stage1 at 0.05, the low bracket around the 0.10 peak. | i=908, i=907 |
| 913 | C6 | alpha sweep: aligned1 x stage1 at 0.15, the high bracket around the 0.10 peak. | i=913 |
| 920 | C6 | alpha sweep: aligned1 x stage1 at 0.125 - the winning merge, exported as final_model. | i=920, i=929 |
| 930 | C6 | alpha sweep: aligned1 x stage1 at 0.1125, a half-interval point below the 0.125 peak. | i=930 |
| 934 | C6 | alpha sweep: aligned1 x stage1 at 0.1375, a half-interval point above the 0.125 peak. | i=934 |
| 951 | C5 | Final submission: aligned_stage1_a0125_model copied to final_model with cp -a and verified byte-identical by sha256 on both safetensors shards, then r… | i=947, i=951, i=952 |
| 963 | proposed:artifact-integr… | A first-tier static guard on the submitted artifact itself, not on any candidate's score: assert the safetensors index and shard key sets match exactl… | i=963, i=963, i=966 |

### 训练序列(13 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 79 | real | 0.01h | returned | **baseline** | baseline - first training of the run. A pure code smoke: --num-samples 64 --max-steps 5 at bs 4, run straight after `python -m py_compile train_sft.py… |
| 88 | real | 0.01h | returned | **C4** | Same stage and data as i=79; batch-size 4 -> 16 and num-samples 64 -> 256, max-steps 5 -> 3. A memory/throughput probe for one hyperparameter, and its… |
| 93 | real | 1.61h | returned | **both** | First real training; nothing to compare against yet, so it is the baseline that every later run is measured from. It fixes the data recipe (100k strat… |
| 222 | real | 0.02h | returned | **both** | First smoke of the new `aligned` stage: the data source changes (100k MetaMathQA variants -> the 7,473 official GSM8K train items rendered with the ev… |
| 226 | real | 0.01h | returned | **both** | Byte-identical retry of i=222 with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` prefixed - an environment repair, not a change of variable. Same thing und… |
| 231 | real | 0.54h | returned | **both** | First real aligned-stage training, from stage1_model. Data changes (7,473 official GSM8K train items in the evaluator's exact 10-shot rendering, with … |
| 290 | real | 0.54h | returned | **C4** | Same corpus as i=231 (--num-samples 7473, same stage, same bs 4 x accum 4, same max-length 3072, same warmup 0.03); only lr 3e-6 -> 1.5e-6 and seed 42… |
| 348 | real | 0.54h | returned | **C4** | Same corpus and same batching as i=290; lr halved again 1.5e-6 -> 7.5e-7, seed 43 -> 44, continued from aligned2_model. Single hyperparameter arm. |
| 385 | real | 2.20h | returned | **both** | The run's longest training (2.20h). Declared intent is C3: consume the 140,000 GSM-derived MetaMath variants disjoint from stage one (--metamath-offse… |
| 560 | real | 0.54h | returned | **C3** | Full-dose arm of the diversity experiment. The recovery recipe is held fixed against its twin at i=612 (same 7,473 corpus, lr 2e-6, warmup .03, seed 4… |
| 612 | real | 0.54h | returned | **C3** | Half-dose arm: the recovery command differs from i=560 in exactly one token, --model diverse_model/checkpoint-4375 instead of diverse_model. Read as C… |
| 667 | real | 0.54h | returned | **C4** | Same 7,473 corpus and batching as i=612; lr 2e-6 -> 1e-6 and seed 45 -> 46, continued from midpoint_aligned_model. Explicitly a replay of the i=290 lr… |
| 835 | real | 0.46h | returned | **C3** | Fourth data source under test: 6,248 self-generated, execution-verified trajectories (--stage selftrain --data-file selftrain.jsonl) instead of any cu… |

### 验证序列(26 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 20 | 3.0 | 30.0 | 是 |  | 0.033 - base-model anchor on n=30; judges no change of the a… |
| 199 | 3.0 | 150.0 | 是 | c01, c03, c27, c05 | 0.04 (6/150) - read from eval_half_sampled.json and confirme… |
| 266 | 3.0 | 150.0 | 是 | c02, c08, c04, c28 | 0.5933333333333334 (89/150) - read from eval_aligned_greedy.… |
| 337 | 3.0 | 150.0 | 是 | c29 | 0.62 (93/150) - accepted, beating the pre-declared 89/150 ba… |
| 372 | 3.0 | 150.0 | 是 | c30 | 0.607 (91/150) - rejected against the 93/150 incumbent. |
| 606 | 3.0 | 150.0 | 是 | c09, c31, c32 | 0.607 (91/150) - the full-dose diversity arm, rejected again… |
| 654 | 3.0 | 150.0 | 是 | c09, c31, c10 | 0.6266666666666667 (94/150) - the half-dose arm; accepted as… |
| 708 | 3.0 | 150.0 | 是 | c34 | 0.633 (95/150) - accepted on the 150-slice; this ranking is … |
| 723 | 4.0 | -1.0 | 是 | c34 | 0.6027293404094011 (795/1319) - first tier-4 run of the run.… |
| 730 | 4.0 | -1.0 | 是 | c09, c31, c10 | 0.6072782410917361 - reverses the 150-slice ordering: the on… |
| 738 | 4.0 | -1.0 | 是 | c29, c31 | 0.623199393479909 - the decisive full-set verdict on the 140… |
| 751 | 4.0 | -1.0 | 是 | c30 | 0.6087945413191812 - confirms on the full set that the third… |
| 770 | 4.0 | -1.0 | 是 | c02, c08, c04, c28 | 0.6262319939347991 (826/1319) - the highest-scoring trained … |
| 802 | 4.0 | -1.0 | 是 | c12, c11 | 0.62 - the 50/50 aligned1 x aligned2 soup, below the 62.62% … |
| 860 | 4.0 | -1.0 | 是 | c22, c23, c35, c24 | 0.6171341925701289 (814/1319) - quarter-epoch self-training … |
| 866 | 4.0 | -1.0 | 是 | c22, c23, c35, c24 | 0.6118271417740713 (807/1319) - half-epoch checkpoint; estab… |
| 876 | 4.0 | -1.0 | 是 | c13 | 0.618 - the 6.25%-epoch interpolated self-training dose, sti… |
| 885 | 4.0 | -1.0 | 是 | c14 | 0.622 - alpha = -0.25 negative extrapolation, below the incu… |
| 899 | 4.0 | -1.0 | 是 | c15 | 0.6315390447308568 (833/1319) - first merge to beat the incu… |
| 912 | 4.0 | -1.0 | 是 | c17 | 0.6239575435936315 (823/1319) - alpha 0.05, below both align… |
| 918 | 4.0 | -1.0 | 是 | c18 | 0.6277482941622441 (828/1319) - alpha 0.15, brackets the 0.1… |
| 925 | 4.0 | -1.0 | 是 | c19 | 0.6376042456406369 (841/1319) - the best number the run ever… |
| 933 | 4.0 | -1.0 | 是 | c20 | 0.624 - alpha 0.1125 collapses back below aligned1, which th… |
| 940 | 4.0 | -1.0 | 是 | c21 | 0.6368460955269143 (840/1319) - alpha 0.1375, one problem be… |
| 967 | 4.0 | -1.0 | 是 | c25, c19 | 0.6292645943896892 (830/1319) - SAME weights as i=925 (sha25… |
| 996 | 4.0 | -1.0 | 是 | c25 | 0.6338134950720242 (836/1319) - third measurement of the sam… |

### 异常与存疑

- **分类学缺口提案 2 条**
  - proposed:env-repair(i=179, i=224, i=226, i=180)
  - proposed:artifact-integrity-check(i=951, i=963, i=963, i=1015)
- **定义缺陷 4 条**
  - Two independent contradictions from one run. (a) 'uniformly average' is wrong: every merge here is a weighted interpolation with alpha as a SEARCHED hyperparameter - 0.5, 0.25, 0.15, 0.1375, 0.125, 0.…(i=879, i=920, i=947, i=961, i=796)
  - The module returns early on every event whose type is not tool_use, so it never looks at the tool_result that follows. For shell reads the file content is right there, one event later, as literal JSON…(i=83, i=84, i=891)
  - writes_for_run only fires on a command that literally contains the string 'generation_config.json' or matches _FINALIZER. `GenerationConfig.from_pretrained(d)` followed by `g.save_pretrained(d)` write…(i=194, i=194, i=129, i=891)
  - For a foreground launch end_reason is set to 'returned' whenever the tool_result carries a timestamp; is_error and extra.exit_code are never read. Counterexample: the smoke at i=222 died inside build_…(i=224, i=224, i=222)
- **边界情形 3 条**
  - C3 and C5 are satisfied by the same act and the run's cleanest controlled contrast falls between them. i=560 and i=612 issue the same recovery command differing in exactly one token: --model diverse_m…(i=560, i=612, i=611)
  - The tested_variable domain {C3, C4, both, unclear} cannot express two things this run spends real time on. (a) Code smokes: i=79 is a compile-and-run probe with no comparison arm - it tests neither a …(i=79, i=231, i=271, i=203)
  - Whether the winning C6 merge counts as an effect at all is undecidable under §2's tier-4 noise band. The run measured its own tier-4 noise directly: three full-test evaluations of byte-identical weigh…(i=973, i=974, i=986, i=995, i=1009)

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
| 17 | 3.0 | 50.0 | 是 |  | 0.06 (base Qwen3-1.7B-Base, n=50) -- reference point, judges… |
| 291 | 3.0 | 150.0 | 是 | c1, c2, c3, c4, c5 | 0.3933333333333333 (59/150) |
| 598 | 3.0 | 150.0 | 是 | c1, c2, c6, c8 | 0.44 (66/150); self-diagnostic: first-answer 0.60, non-stop … |
| 696 | 3.0 | 150.0 | 是 | c7 | 0.4066666666666667; first-answer fell 0.60 -> 0.487, so reje… |
| 759 | 3.0 | 150.0 | 是 | c9, c10 | 0.38; first-answer 0.593, spillover up -- ORPO rejected |
| 788 | 3.0 | 150.0 | 是 | c11, c12 | 0.447; non-stop 0.34 -> 0.04 but first-answer 0.60 -> 0.46 |
| 829 | 3.0 | 150.0 | 是 | c13 | 0.553; first-answer 0.593 preserved, non-stop 0.167 |
| 851 | 3.0 | 150.0 | 是 | c14 | 0.62; first-answer 0.627, non-stop 0.067, multi-answer 0.04 … |
| 887 | 3.0 | 150.0 | 是 | c15, c16 | 0.5733333333333334; first-answer 0.607 < 0.627 -- rejected |
| 1001 | 3.0 | 150.0 | 是 | c17, c18 | 0.58; first-answer 0.607 -- rejected |
| 1030 | 3.0 | 150.0 | 是 | c19 | 0.567; first-answer 0.6066666666666667 -- rejected |
| 1047 | 3.0 | 150.0 | 是 | c20 | 0.62 (ties the leader); first-answer 0.607 -- rejected |
| 1066 | 3.0 | 150.0 | 是 | c21 | 0.62 (ties); max-length failures 0.067 -> 0.027 but answer-t… |
| 1074 | 3.0 | 150.0 | 是 | c22 | 0.633 -- best 150-item result seen, +2 answers over 0.62 |
| 1081 | 3.0 | 300.0 | 是 | c22 | 0.60 at n=300 -- the n=150 gain does not survive |
| 1084 | 3.0 | 300.0 | 是 | c14, c22 | 0.61 at n=300 -- paired control arm for i=1081; original bea… |
| 1091 | 3.0 | 150.0 | 是 | c23 | 0.42; first-answer 0.6133333333333333 (higher than the final… |
| 1110 | 3.0 | 150.0 | 是 | c24 | 0.607; first-answer also 0.607 -- rejected |
| 1119 | 3.0 | 300.0 | 是 | c25 | 0.613 at n=300, one answer above the original's 0.61 -- judg… |
| 1123 | 4.0 | -1.0 | 是 | c25 | 0.5898407884761183 (full 1,319, 1024-token cap) |
| 1126 | 4.0 | -1.0 | 是 | c14, c25 | 0.6163760424564063 (812/1,319, 1024-token cap) -- paired con… |
| 1132 | 4.0 | -1.0 | 是 | c14, c28 | 0.5883244882486732 -- same weights, same 1,319 items as i=11… |
| 1140 | 4.0 | -1.0 | 是 | c21, c28 | 0.6186504927975739 (816/1,319) -- c21 loses at n=150/300 but… |
| 1148 | 4.0 | -1.0 | 是 | c22 | 0.5936315390447309 (full, 4000-token cap) |
| 1161 | 4.0 | -1.0 | 是 | c26 | 0.6057619408642911 -- alpha=1.5 falls below alpha=1.0 (0.618… |
| 1191 | 3.0 | 150.0 | 是 | c27 | 0.633 (n=150 at the 4000-token cap) -- packaging reproductio… |

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

## codex_non_api_max_gpt-5.6-sol_10h_run1__humaneval_HuggingFaceTB_SmolLM3-3B-Base_17398718
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | humaneval | HuggingFaceTB_SmolLM3-3B-Base | 6.54h | 0.5426829268… |

### 改动序列(49 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 72 | C2 | prepare_data.py renders every training example into the evaluator's exact contract: user turn = bare signature+docstring stub, assistant turn = a sing… | i=65, i=72 |
| 72 | C10 | HumanEval decontamination filter inside prepare_data.py: reject exact benchmark function-name match, any shared normalized 20-token span, or >=3 share… | i=78, i=72 |
| 72 | C3 | Initial SFT corpus data/code_sft.jsonl: 80,797 unique examples = SelfCodeAlign 45,716 + Magicoder OSS-Instruct 20,342 + Evol-Instruct 14,581 + 158 non… | i=109, i=72 |
| 72 | C1 | train_sft.py writes a generation_config at save time: eos_token_id=[<\|im_end\|>,<\|end_of_text\|>], do_sample=False, max_new_tokens=2048, pad=<\|end_… | i=284, i=72 |
| 88 | C8 | Pad-token crash fix: SFTTrainer rejected pad_token '<\|reserved_special_token_1\|>' as absent from the vocabulary; PAD_TOKEN switched to <\|end_of_tex… | i=86, i=87 |
| 124 | C3 | Second data recipe data/code_sft_plus.jsonl: adds 22,241 interpreter-verified synthetic Python functions from wuyetao/spp; 7,725 further candidates re… | i=130, i=124 |
| 150 | C11 | Failure-mode classifier over the official inspect_ai sample log: bucket every non-correct sample into syntax / indent / name / type / timeout / assert… | i=150, i=162 |
| 180 | C3 | prepare_opencode.py: new source = nvidia/OpenCodeInstruct rows where every generated unit test passed; 84,820 unique functions kept from 300,000 rows,… | i=193, i=180 |
| 257 | C3 | make_concise.py: AST-strip implementation docstrings and generated unittest classes from the assistant targets of the verified corpus (~18% less targe… | i=261, i=257 |
| 300 | C11 | Per-task flip analysis between two official eval logs (both / a_only / b_only / neither) to measure whether two checkpoints solve complementary subset… | i=300, i=301 |
| 312 | C6 | interpolate_models.py: linear weight interpolation (alpha blend) of two safetensors checkpoints, motivated by the observation that two earlier checkpo… | i=311, i=312 |
| 317 | C6 | Soup #1: sft250 x sft850plus at alpha 0.5 -> runs/sft250_850_merge050. Built but never evaluated; abandoned. | i=317 |
| 355 | C2 | Inference-side format ablation: symlink runs/qwen_sft250 -> sft250 so evaluate.py picks templates/qwen3.jinja for the same weights, dropping the ~240-… | i=355, i=359 |
| 360 | C2 | Add scripts/minimal_chatml.jinja plus a --chat-template option to train_sft.py so a model could be trained against the minimal prompt. Never exercised… | i=360, i=359 |
| 400 | C5 | Add --save-steps to train_sft.py so intermediate checkpoints exist, making post-hoc checkpoint selection possible instead of only submitting the train… | i=397, i=400 |
| 431 | C6 | Soup #2: sft250 x sft250_open600 at alpha 0.75 -> runs/sft250_open_merge075. Built but never evaluated; skipped. | i=431 |
| 434 | C11 | Truncation / stop-reason and output-length audit derived from the official log: all 150 samples ended with stop_reason 'stop' and averaged 592 charact… | i=434, i=435 |
| 460 | C3 | prepare_dpo.py: mine repeated OpenCodeInstruct prompts that carry both a test-perfect and a test-failing solution into decontaminated preference pairs… | i=446, i=460 |
| 488 | C4 | train_dpo.py: DPO stage (TRL DPOTrainer, beta, precomputed reference logps) as an alternative to further SFT. | i=488, i=484 |
| 515 | C3 | Label-quality audit: re-execute each candidate 'rejected' program against the chosen program's own tests in a resource-limited sandbox instead of trus… | i=546, i=515 |
| 535 | C6 | Soup #3: sft250_open600 x open600_concise300/checkpoint-150 at alpha 0.5 -> runs/open_concise150_merge050. | i=535 |
| 547 | C3 | prepare_mutation_dpo.py: synthesise hard negatives by AST-mutating verified-correct programs (boundary operators, constants, arithmetic) and keep only… | i=546, i=547 |
| 579 | C8 | Sandbox fix: give each executed test its own temp directory after concurrent programs collided on file writes in data/execution_sandbox. | i=578, i=579 |
| 615 | C8 | Throughput fix: tighten per-program execution limits and add progress reporting, then abandon the oversized 3,000-pair build, because deliberately bro… | i=612, i=615 |
| 661 | C3 | Add --start-shard to prepare_opencode.py so a disjoint seven-shard OpenCodeInstruct corpus (shards 3-9) can be drawn instead of re-using the three sha… | i=660, i=661 |
| 685 | C9 | Artifact overwrite guard adopted for training too: from this launch on, every training command is prefixed with `test ! -e <output>` instead of the ea… | i=685, i=561 |
| 699 | C3 | combine_sft.py: exact-dedup union of both verified concise corpora -> 276,850 unique examples (data/opencode_concise_all.jsonl). | i=705, i=699 |
| 756 | proposed:eval_protocol_c… | Evaluation protocol changed mid-run: --max-tokens raised from 1024 to 4000 for every subsequent evaluate.py call. All later scores (including the 48.7… | i=756, i=605 |
| 810 | C3 | filter_unseen_sft.py: exact-dedup the ten-shard held-out pool against every example already trained on, leaving 226,899 with zero overlap. | i=815, i=810 |
| 858 | C6 | Soup #4: interpolate the two adjacent checkpoints of the winning run (c150_new400 checkpoint-100 x checkpoint-200) at alpha 0.25 / 0.50 / 0.75. | i=858, i=854 |
| 867 | C8 | Disk reclaim: rm -rf the rejected interpolation models and two rejected training branches before launching the next run. | i=867, i=866 |
| 881 | C8 | Add --dataset-num-proc to train_sft.py to work around a filesystem stall in the TRL packing pass that blocked the seed-replicate before any optimizer … | i=877, i=881 |
| 893 | C8 | Add --no-packing to bypass the stalling offline packing pass, compensating the per-step token budget with batch size 16 x accumulation 6 so the replic… | i=890, i=893 |
| 924 | C3 | prepare_mbpp_all.py: human-authored MBPP (all splits), each canonical solution execution-verified against its public tests, same HumanEval overlap rej… | i=933, i=924 |
| 960 | C3 | Distribution matching: extract an exact-unseen, execution-perfect single-function subset (144,077 rows) from the harder held-out pool, matched to Huma… | i=963, i=960 |
| 976 | C9 | Submit guard: copy the measured-best runs/c150_new400/checkpoint-200 (48.7%, not the most recent artifact) into final_model, behind a `test ! -e final… | i=975, i=976 |
| 982 | C10 | Write final_model/README.md and final_model/training_provenance.json recording base model, curriculum, the decontamination rule (function-name + 20-to… | i=982, i=975 |
| 982 | C1 | Set use_cache: true in final_model/config.json when packaging ('enabling inference cache'); the source checkpoint had use_cache false. | i=975, i=982 |
| 998 | C1 | Revert use_cache to false in final_model/config.json after the copied model scored 41.3% versus the source checkpoint's 48.7%. The revert did not rest… | i=991, i=998 |
| 1005 | C11 | Localise the 48.7 -> 41.3 gap: cmp the two safetensors shards (identical) and read the eval logs' model/model_args/config, ruling out weights and poin… | i=1004, i=1005 |
| 1013 | C1 | Add temperature: 0 and do_sample: false to final_model/generation_config.json, making the submitted model explicitly greedy, after diagnosing that eva… | i=1012, i=1013 |
| 1021 | C11 | Sampling-noise audit: compare the raw completions of three identical-weight evaluations; only 14/150 completions matched, establishing that the evalua… | i=1024, i=1021 |
| 1025 | C1 | Set temperature 0.2 (do_sample true) in final_model/generation_config.json. | i=1024, i=1025 |
| 1026 | C11 | Decision-rule change: evaluate each decoding policy 3-5 times and select on the replicate mean rather than a single draw. | i=1024, i=1026 |
| 1036 | C1 | Set temperature 0.1 in final_model/generation_config.json. | i=1035, i=1036 |
| 1043 | C1 | Set temperature 0.05 in final_model/generation_config.json. | i=1041, i=1043 |
| 1052 | C1 | Set temperature 0.02 in final_model/generation_config.json. | i=1051, i=1052 |
| 1061 | C1 | Lock the submitted decoding policy back to temperature 0.05 after 0.02 measured below the 0.05 three-run mean; README and provenance updated with the … | i=1056, i=1061 |
| 1068 | proposed:eval_protocol_c… | Final validation deliberately drops --max-connections, --gpu-memory-utilization and --max-tokens so the measurement uses evaluate.py's own defaults, i… | i=1066, i=1068 |

### 训练序列(16 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 84 | real | 0.01h | returned | **baseline** | First training launch of the run (baseline): base SmolLM3-3B-Base, data/code_sft.jsonl, 250 steps, lr 3e-5, bs 8 x accum 4. Nothing to compare against… |
| 89 | real | 0.27h | returned | **unclear** | Relaunch of i=84, byte-identical hyperparameters and data; the only delta is the C8 pad-token fix in train_sft.py (c5) plus `set -o pipefail`. This is… |
| 168 | real | 0.63h | returned | **both** | vs sft250: data code_sft.jsonl -> code_sft_plus.jsonl (+22,241 SPP verified functions), lr 3e-5 -> 2e-5, steps 250 -> 600, seed 3407 -> 3411, and it c… |
| 249 | real | 0.62h | returned | **C3** | Near-clean data ablation against i=168: same parent (runs/sft250), same 600 steps, same lr 2e-5, same bs 8 x accum 4; only --data changes (code_sft_pl… |
| 424 | real | 0.32h | returned | **both** | vs i=249: parent moves to the new best (sft250_open600), data becomes the concise transform of the SAME corpus (docstrings + unittest scaffolding stri… |
| 561 | real | 0.16h | returned | **C4** | Clean schedule ablation against i=424: identical parent (sft250_open600), identical data (opencode_concise.jsonl), identical lr 1e-5 / bs 8 / accum 4.… |
| 638 | real | 0.08h | returned | **both** | Method switch SFT -> DPO (train_dpo.py, beta 0.1, lr 5e-7, bs 4 x accum 8) from the 46.7% checkpoint, on a brand-new preference corpus (1,351 executio… |
| 685 | real | 0.43h | returned | **C3** | vs the SFT lineage at i=424 (the DPO branch was discarded): same parent open600_concise300/checkpoint-150, new data = seven disjoint OpenCode shards (… |
| 770 | real | 0.54h | returned | **both** | The agent calls this a 'controlled branch' but three things move at once versus i=685: the branch point (back to runs/sft250 instead of the concise-15… |
| 835 | real | 0.23h | returned | **C3** | Same move as i=685 applied to the new best: parent c150_new400/checkpoint-200, data = the 226,899 exact-unseen deduped rows from ten further shards; l… |
| 867 | real | 0.05h | returned | **C4** | Seed/schedule replicate of i=685: identical parent and identical data (opencode_concise_new.jsonl); only seed 1327 -> 1361 and max-steps 400 -> 250. K… |
| 886 | real | 0.03h | returned | **C4** | Same experiment as i=867, relaunched with --dataset-num-proc 1 as a pure C8 workaround for the packing stall. Tested variable unchanged (seed + schedu… |
| 895 | real | 0.02h | returned | **C4** | Same experiment, relaunched with --no-packing and bs 8 x accum 4 -> bs 16 x accum 6. The agent states the batch change is a compensation to hold the p… |
| 904 | real | 0.21h | returned | **C4** | Fourth and successful launch of the same seed/schedule replicate: --dataset-num-proc back to 16 with --no-packing kept. Ran ~13 min, stopped by the ag… |
| 934 | real | 0.05h | returned | **C3** | New data source from the same parent as i=835 (c150_new400/checkpoint-200): human-authored MBPP, all splits, execution-verified, 3,208 repeated rows. … |
| 964 | real | 0.11h | returned | **C3** | Distribution-matching data test from the same parent: 144,077 exact-unseen single-function rows filtered to HumanEval's simpler structural profile, ve… |

### 验证序列(33 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 22 | 3.0 | 20.0 | 是 |  | 0.100 (base model, n=20) - the run's zero baseline; judges n… |
| 140 | 3.0 | 30.0 | 是 | c1, c2, c3, c4, c5 | 0.700 (n=30). Read from the foreground stdout. |
| 163 | 3.0 | 150.0 | 是 | c1, c2, c3, c4 | 0.347 (n=150). The agent treats this, not the n=30 0.700, as… |
| 244 | 3.0 | 150.0 | 是 | c6 | 0.340 (n=150) - the SPP-enriched mixture is flat/slightly be… |
| 411 | 3.0 | 150.0 | 是 | c7 | 0.427 (n=150) - execution-filtered OpenCode data is the run'… |
| 418 | 3.0 | 150.0 | 是 | c11 | 0.333 (n=150) vs 0.347 for the same weights under the suppli… |
| 525 | 3.0 | 150.0 | 是 | c8, c13 | 0.467 (n=150) at the mid checkpoint - the concise transform'… |
| 528 | 3.0 | 150.0 | 是 | c8 | 0.420 (n=150) at the endpoint - later low-rate updates lost … |
| 538 | 3.0 | 150.0 | 是 | c18, c9 | 0.367 (n=150) - the soup is far below both parents (0.427 / … |
| 600 | 3.0 | 150.0 | 是 |  | 0.407 (n=150), checkpoint-50 of the schedule ablation (train… |
| 602 | 3.0 | 150.0 | 是 |  | 0.393 (n=150), checkpoint-100 of the same schedule ablation. |
| 605 | 3.0 | 150.0 | 是 |  | 0.427 (n=150), checkpoint-150. Best of the three, still belo… |
| 648 | 3.0 | 150.0 | 是 | c15, c17, c19, c16 | 0.453 (n=150) at DPO step 20 - below the 0.467 pre-DPO check… |
| 652 | 3.0 | 150.0 | 是 | c15, c17, c19, c16 | 0.380 (n=150) at DPO step 40 - a 7.3-point collapse despite … |
| 656 | 3.0 | 150.0 | 是 | c15, c17, c19, c16 | 0.420 (n=150) at DPO step 60. Whole DPO branch discarded. |
| 756 | 3.0 | 150.0 | 是 | c22, c24 | Four evaluations in one shell loop, all four scores read bac… |
| 826 | 3.0 | 150.0 | 是 | c23 | Five evaluations in one loop; scores read back inline: 0.340… |
| 849 | 3.0 | 150.0 | 是 | c25 | Four evaluations in one loop: 0.4067, 0.4267, 0.3867, 0.4067… |
| 860 | 3.0 | 150.0 | 是 | c29, c9 | Three evaluations in one loop: alpha 0.25 -> 0.4133, 0.50 ->… |
| 912 | 3.0 | 150.0 | 是 | c27, c28 | Four evaluations in one loop: 0.4400, 0.4533, 0.4067, 0.4200… |
| 939 | 3.0 | 150.0 | 是 | c30 | Four evaluations in one loop: 0.4333, 0.4200, 0.4067, 0.4200… |
| 970 | 3.0 | 150.0 | 是 | c31 | Four evaluations in one loop: 0.4467, 0.3933, 0.4400, 0.4200… |
| 988 | 3.0 | 150.0 | 是 | c32, c34 | 0.4133 (n=150) from the submitted path - 7.3 points below th… |
| 1001 | 3.0 | 150.0 | 是 | c36 | 0.4133 (n=150) after reverting use_cache to false - identica… |
| 1014 | 3.0 | 150.0 | 是 | c39 | 0.4267 (n=150) under explicit greedy decoding. Skeleton reco… |
| 1026 | 3.0 | 150.0 | 是 | c40, c41 | Three evaluations in one loop at temperature 0.2: 0.4400, 0.… |
| 1032 | 3.0 | 150.0 | 是 | c40, c41 | Two more replicates at temperature 0.2: 0.5333 and 0.5400; f… |
| 1037 | 3.0 | 150.0 | 是 | c42, c41 | Three evaluations at temperature 0.1: 0.5267, 0.5267, 0.5467… |
| 1044 | 3.0 | 150.0 | 是 | c43, c41 | 0.5467 (n=150) at temperature 0.05, first replicate. Skeleto… |
| 1048 | 3.0 | 150.0 | 是 | c43, c41 | Two further replicates at temperature 0.05: 0.5533 and 0.546… |
| 1053 | 3.0 | 150.0 | 是 | c44 | 0.5467 (n=150) at temperature 0.02, single run - slightly be… |
| 1063 | 4.0 | -1.0 | 是 | c45 | 0.53659 on the full 164-problem set (tier 4), the only full-… |
| 1068 | 3.0 | 150.0 | 是 | c45, c46 | 0.5400 (n=150) under evaluate.py's own defaults (no --max-co… |

### 异常与存疑

- **1 段训练的受测变量判不出**:i=[89]
- **分类学缺口提案 2 条**
  - branch_point(i=769, i=698, i=248)
  - eval_protocol_change(i=756, i=605, i=1066)
- **定义缺陷 3 条**
  - (i=1054, i=982, i=1059)
  - (i=756, i=767, i=826)
  - (i=284, i=735, i=1012)
- **边界情形 3 条**
  - §9/§10 record the known boundary in one direction: a data change forces hyperparameters (sequence length, effective batch) to follow, and no rule says whether that is C3 or both. i=638 is the mirror i…(i=638, i=637, i=424)
  - At i=515/517 the agent builds a resource-limited subprocess sandbox and re-runs each candidate 'rejected' program against the chosen program's own tests, instead of trusting the dataset's average_test…(i=546, i=515, i=510)
  - i=400 adds --save-steps to train_sft.py. save_steps is literally a training-config field (C4), but nothing about optimisation changes - its only function is to manufacture the candidate set that C5 th…(i=400, i=397, i=424)

## codex_non_api_max_gpt-5.6-sol_10h_run1__humaneval_Qwen_Qwen3-1.7B-Base_17398717
| agent | harness | benchmark | base model | 时长 | 最终分 |
|---|---|---|---|---|---|
| gpt-5.6-sol | codex | humaneval | Qwen_Qwen3-1.7B-Base | 6.59h | 0.6951219512… |

### 改动序列(37 条)

| i | 类别 | 做了什么 | 证据 |
|---|---|---|---|
| 61 | C2 | 格式对齐:prepare_data.py 把评测器的原话指令包装(`Read the following function signature and docstring...`)抄进训练 prompt,并把每条样本改成「签名+docstring → 缩进函数体」的补全形状,与 humaneval … | i=61, i=57 |
| 100 | C10 | 去污染:158 个 HumanEval 函数名黑名单 + prompt 8-gram + 解法 12-gram 三重拒收;benchmark 文本只当拒收器用,不写进训练集。v1 因此丢掉 25,482 条。 | i=100, i=100, i=57 |
| 100 | C3 | v1 语料 data/train.jsonl:nvidia/OpenCodeInstruct 取 50,000 条(全部单元测试通过、标准库内、无 stdin/print 的独立函数)+ MBPP-train 62 条 ×4 份复制 = 50,248 条。 | i=63, i=100, i=103 |
| 110 | C4 | 训练方法:全参 BF16 SFT(明确不用 LoRA 适配器),assistant 侧单独掩码算损失,并把结束符一起训进去并显式导出。整条 run 21 次训练启动全部沿用这一条,从未出现 LoRA、DPO 或 GRPO。 | i=110, i=110 |
| 121 | C8 | OOM 可行性修复:batch 32(i=116)→16(i=121)→8(i=124),梯度累积 2→4→8,有效 batch 恒定 64。agent 明说这是显存峰值问题、不改优化实验;前两次都是 0 步/1 步即崩,没有权重更新。 | i=120, i=120, i=121, i=124 |
| 146 | C8 | 撞上 `GenerationConfig is invalid`:do_sample=False 与 temperature=0.0 同时出现被 save_pretrained 拒收,导致训练末尾的元数据导出抛异常。**但权重和 tokenizer 已先落盘**,五个目录的 model.safete… | i=146, i=146, i=148, i=151 |
| 153 | C1 | 手工整份重写 sft1 的 5 份 generation_config.json(250/500/750/777/model)。字段级差异(前 i=151,后 i=160):eos_token_id [151643] → [151645, 151643](补上 <\|im_end\|>);新增 do… | i=151, i=153, i=160, i=160, i=160 |
| 162 | C8 | config.json 的 use_cache:false → true。Trainer 存盘时把 KV cache 关掉了(i=160 逐字可见),推理端读到 false 会拖慢/改变 vLLM 行为;此后每个新 checkpoint 都要重打这个补丁(共 20+ 次)。 | i=160, i=162, i=343 |
| 180 | C8 | config.json 的 transformers_version:4.57.3 → 4.51.0。动机不是模型行为,而是 transformers 4.57.3 会按 config 里的版本号触发 tokenizer「incorrect regex pattern / fix_mistral_r… | i=178, i=173, i=180, i=343 |
| 189 | C5 | checkpoint 选择策略:发现 checkpoint-500 在 20 题切片上 80%,而 eval_loss 更低的终点只有 70%,于是把选择判据从 eval_loss 换成功能性评测。此后每个训练都密集存 checkpoint 并逐个上第三档。 | i=189, i=189 |
| 223 | C8 | 数据管线 bug 修复:OpenCodeInstruct 的 unit_tests 是「一个 JSON 列表,元素才是 assert 语句」,strict 筛选器按外层 list 解析,导致命中数恒为 0(i=219 三个 shard 全 0)。修好后同样三个 shard 变成 173/1315/1… | i=221, i=219, i=226 |
| 244 | C3 | v2 strict 语料 data/strict_algorithmic.jsonl:只要 algorithmic 域、全部测试通过、三项 LLM judge 都是 5 分、单函数无依赖,并显式排除一阶段用过的 50,000 个 source_id,得 11,290 条。 | i=244, i=244, i=244 |
| 297 | C3 | v3 balanced 语料 data/balanced_corrected.jsonl:generic/algorithmic × self/evol 四格配额(12k/12k/20k/5k)、shard 顺序按 seed 73 打乱以避免只吃到前几个 generic shard、并在选目标函数之… | i=277, i=297, i=297 |
| 318 | C1 | train_sft.py 的 export_generation_config:把 eos_token_id 写成 [<\|im_end\|>, <\|endoftext\|>] 两个都接受、temperature 写成 0.01(注释说明是为了在 Transformers 和 vLLM 两边都合法… | i=318, i=318, i=318 |
| 339 | C11 | 验证器工装(第一档):自写脚本读 inspect_ai 的 json log,把官方评分器自己的输出拆成可决策信号 —— 逐题 C/I、stop_reason(stop vs max_tokens)、异常类型直方图、completion 词数分位。用它确认失分是语义错误(AssertionError… | i=339, i=340, i=340 |
| 386 | C6 | 权重平均工装 average_models.py(两模型按 alpha 线性插值,只碰 model.safetensors,其余文件从 model-b 拷)。全 run 共造了二十多个 soup 目录;跨 run 的 soup 全部不如单模型,只有同轨迹内的成功(见 c19)。 | i=384, i=386, i=896 |
| 425 | C2 | prompt 长度对齐:先量出 HumanEval prompt 中位 396 字符、自建语料 898 字符,再写 prepare_concise_data.py 把 docstring 截到 500 字符。先做 50/50 混合(sft4),再做 100% 压缩(sft5),压缩后四分位 487/… | i=421, i=421, i=425, i=467 |
| 490 | C3 | 推理蒸馏语料:nvidia/OpenCodeReasoning,只取 split=train、每个不重复题目一条 trace、解法只用标准库,过同一套去污染,得 3,156 条;再把 <think> 段截到 2,400 字符、保留末尾已验证程序,使 token 长度中位落到 1,094。 | i=490, i=490, i=518 |
| 585 | C4 | 两段式训练:先在推理语料上训出 runs/reasoning1/model,再把它当初始化跑与 sft5 逐字相同的 concise SFT(同数据、同 bs8×ga8、同 lr 1.5e-5、同 save-steps 150)。唯一变量是初始化点。结果 66.0%,低于同配方从 base 起的 6… | i=585, i=585, i=632 |
| 601 | C3 | 全新样本语料:用 shard-seed 191 重建一份 balanced 语料(与 seed 73 那份 source_id 只重叠 16,711),再用 subtract_corpus.py 把重叠部分减掉,得 29,065 条完全没被 leader 见过的同格式样本。测的是「更多不重复的高质量… | i=575, i=601, i=601 |
| 633 | C4 | 续训协议:不从 base 重训,而是把当前 leader(sft5/checkpoint-450,后来是 soup_sft5_450_600)当初始化,用远低的 lr(3e-6 / 4e-6)、有时只跑 0.5 epoch 继续训。全 run 用了三次(sft7 / sft14 / sft15),三… | i=633, i=633, i=1318 |
| 687 | C6 | 同一训练轨迹内的 checkpoint 平均:sft5 的 450 与 600 各自 69.3%,50/50 平均后 70.0%(105/150),是本 run 第一次突破 69.3% 的动作。随后 alpha 扫 0.25/0.45/0.55/0.75 全部回落,最优点尖锐地落在 0.50。这个配… | i=687, i=692, i=710 |
| 711 | C4 | 单变量超参对照:除 learning-rate 由 1.5e-5 降到 1e-5 外,与 sft5 的启动命令逐字相同(同数据 balanced_all_concise、同 bs8×ga8、同 1 epoch、同 save-steps 150)。agent 自己写明「changes one vari… | i=710, i=711, i=711 |
| 722 | C1 | 「贪婪解码」实验:cp -a 复制 soup_sft5_450_600 得到逐字节相同的 soup_sft5_greedy,只改 generation_config.json —— 删掉 temperature:0.01 与 top_p:1.0、do_sample 由 true 改 false。同权… | i=716, i=720, i=722, i=749, i=762 |
| 779 | C4 | 单变量种子对照:与 sft5 的启动命令逐字相同,只多了 `--seed 123`(sft5 用默认 42)。语料文件逐字节相同,变的是 Trainer 的洗牌/初始化 RNG。 | i=779, i=779, i=898 |
| 797 | C1 | 改回真贪婪:把 temperature 显式写成 0.0(而不是删掉),GenerationConfig.to_diff_dict 逐字确认 soup_sft5_greedy 变成 {temperature: 0.0},随后同一份权重读回 0.700 —— 与 temperature 0.01 的 … | i=797, i=938, i=931, i=1090 |
| 801 | C3 | 短解课程语料 short_mbpp_mix.jsonl:只保留 response ≤500 字符的样本(39,944 条),外加 60 条去污染后的 MBPP-train ×20 份复制,共 41,144 条。测的是「把回答长度分布压向 HumanEval」。 | i=801, i=801, i=801 |
| 832 | C3 | 换数据来源:从本地 HF cache 里翻出 notbadai/python_functions_reasoning(OpenCoder-LLM/opc-sft-stage1 子集),扫 206,299 条、独立过同一套去污染(拒 8,434 条)后取 50,000 条,再与自建 concise 语… | i=832, i=832, i=841 |
| 905 | C4 | 按 OpenCodeInstruct 论文的报告配方改调度:epochs 1→3、learning-rate 1.5e-5→5e-6、save-steps 708(每个 epoch 末存一次),数据不变。跑到 68%(1446/2124 步)时因为两个 epoch 的 eval_loss 都追不上 … | i=905, i=1003, i=1004 |
| 993 | C3 | edge5 语料:在 balanced 的基础上再加一条硬条件 —— LLM judge 的 edge_case_consideration 必须是 5 分,algorithmic/evol 配额归零,shard-seed 换成 317,得 36,543 条。动机是 i=949/950 的工装读出剩… | i=961, i=993, i=993 |
| 1035 | C3 | large 语料:把配额放大到 generic 25k/25k + algorithmic-self 40k、min-edge-score 降到 4、shard-seed 509,得 75,838 条,其中 43,494 条是当前 leader 语料里没有的新题。测的是「覆盖面 vs 过滤严格度」。 | i=1001, i=1035, i=1043 |
| 1153 | C11 | 跨日志逐题对照工装:扫全部 limit=150 的 inspect_ai log,按被评模型建逐题对错向量,对每个候选算相对 leader 的 gain / loss / oracle / agree。这不是代理评分器(输入是官方评分器自己的输出),用途是挑 soup 搭档 —— 挑「gain 高、… | i=1153, i=1153, i=1379 |
| 1226 | C2 | doctest 增强:发现 HumanEval 的 docstring 常用 `>>>` 举例,而自建语料把示例全删了;于是写 prepare_doctest_augmented.py,只用**每条样本自己的 unit_tests**反推出 `>>>` 示例塞回 docstring(每条 ≤3 个、… | i=1203, i=1226, i=1226, i=1226 |
| 1283 | C10 | 对增强后的语料重跑污染审计:8-gram 命中从基线语料的 1,019 涨到 3,112,agent 没有直接重滤,而是把命中的 n-gram 逐条打出来,发现全是 ('you','are','given','a','list','of','integers') 一类通用题面模板,判定为误报并保留语… | i=1283, i=1283, i=1287 |
| 1454 | C8 | shell 自伤及其自查:用 `cp -al`(硬链接)复制模型目录做温度对照,结果两个目录的 generation_config.json 是同一个 inode,patch 一次同时改到两边(i=1459 两行读数完全一样)。agent 立刻察觉,用「先 delete 再 add」断开硬链接,并用… | i=1454, i=1459, i=1461, i=1464, i=1464 |
| 1488 | C3 | doctest 比例的剂量反应扫描:concise:doctest = 100:0 / 75:25 / 50:50 / 25:75 四个点,除数据混合比外训练配方逐字相同(from base、max-length 1024、bs8×ga8、1 epoch、lr 1.5e-5、seed 42、480/… | i=1488, i=1297, i=1414, i=1482 |
| 1546 | C9 | 提交守卫 + 自包含性审计:`test ! -e final_model` 先确认不会覆盖,再 `cp -a`(实拷贝,不是软链)把 soup_sft16_480_600 导出成 final_model;随后核对 310 个权重张量、无 .bin、无符号链接、config/tokenizer/gen… | i=1546, i=1546, i=1493, i=1555, i=1556 |

### 训练序列(22 段)

| i | 类型 | 时长 | 结局 | 受测变量 | 相对上一次 |
|---|---|---|---|---|---|
| 113 | — | — | — | **unclear** | **这一行不是训练启动**。命令是 `python -m py_compile train_sft.py && python train_sft.py --help`,输出是 argparse 的 usage 文本,没有加载模型、没有产物、没有一步优化。机械层把它记成 real 训练是误判(见 de… |
| 116 | real | 0.01h | returned | **baseline** | baseline —— run 内第一次真实训练启动。data/train.jsonl(v1 50,248 条)、batch 32×accum 2、1 epoch、lr 2e-5、save-steps 250。没有可比对象。**且它没跑成**:0/777 步就 CUDA OOM(Qwen 152K … |
| 121 | real | 0.01h | returned | **unclear** | 相对 i=116 只改 batch 32→16、accum 2→4,**有效 batch 仍是 64**,数据与 lr/epoch/save-steps 逐字未动。这是 C8 的 OOM 重启,agent 明说「without changing the optimization experiment… |
| 124 | real | 0.30h | returned | **unclear** | 相对 i=121 只改 batch 16→8、accum 4→8,有效 batch 仍是 64。同一条 OOM 重启链的第三次,这次跑通(777 步,约 17 分钟)。仍不是配方或超参对照。 |
| 251 | real | 0.07h | returned | **both** | 相对 i=124 同时变了两类:数据换成 v2 strict 语料(strict_algorithmic.jsonl,11,290 条,与一阶段不相交)**且**方法从「从 base 训一轮」改成「从 sft1/checkpoint-500 续训、lr 2e-5→5e-6、validation-si… |
| 300 | real | 0.30h | returned | **both** | 相对 i=251 回到从 base 训(model-path Qwen/Qwen3-1.7B-Base),数据换成 balanced_corrected.jsonl(45,776 条,修好 unit_tests 解码 + 四格配额 + shard 打乱),同时 lr 从 5e-6 改成 1.5e-5… |
| 435 | real | 0.27h | returned | **C3** | 相对 i=300 **只换数据**:balanced_corrected.jsonl → balanced_mixed_concise.jsonl(同一批 45,776 条样本,一半的 docstring 被截到 500 字符)。model-path、validation-size 512、batc… |
| 505 | real | 0.20h | returned | **C3** | 相对 i=435 **只换数据**:balanced_mixed_concise → balanced_all_concise(full-fraction 0.5 → 0.0,全部样本都压缩)。其余启动参数逐字相同。剂量反应的第二个点,读数 68.0% → 69.3%。 |
| 570 | real | 0.06h | returned | **both** | 相对 i=505 换成完全不同的语料(code_reasoning_compact.jsonl,3,156 条竞赛推理 trace,回答里含 <think> 段)**并且**改了 max-length 1024→1536、batch 8→4、accum 8→16、lr 1.5e-5→1e-5、val… |
| 585 | real | 0.13h | returned | **C4** | 相对 i=505 **只换初始化**:model-path 由 Qwen/Qwen3-1.7B-Base 换成 runs/reasoning1/model,数据仍是 balanced_all_concise.jsonl,batch 8×accum 8、epochs 1、lr 1.5e-5、valid… |
| 633 | real | 0.14h | returned | **both** | 相对 i=505 同时换数据(new_unique_concise_seed191.jsonl,29,065 条 leader 没见过的同格式样本)和方法(从 sft5/checkpoint-450 续训、lr 1.5e-5→3e-6、save-steps 150→100)。主诉是「更多不重复数据」… |
| 711 | real | 0.20h | returned | **C4** | 相对 i=505 **只改 learning-rate 1.5e-5 → 1e-5**,数据、model-path、validation-size、batch、accum、epochs、save-steps 全部逐字相同。agent 自己写明「changes one variable only」。读… |
| 779 | real | 0.20h | returned | **C4** | 相对 i=505 **只加 `--seed 123`**(sft5 用脚本默认 42),语料文件、lr、batch、accum、epochs、save-steps 全部逐字相同。变的是 Trainer 的洗牌与初始化 RNG,语料本身逐字节未变。读数 65.3%(ckpt450)vs 69.3%。 |
| 905 | real | 0.39h | returned | **C4** | 相对 i=505 只改调度:epochs 1→3、lr 1.5e-5→5e-6、save-steps 150→708(每 epoch 末存一次),数据 balanced_all_concise 与 batch/accum 不变。跑到 1446/2124 步被 agent 主动 kill(两个 epo… |
| 1006 | real | 0.16h | returned | **C3** | 相对 i=505 **只换数据**:balanced_all_concise(45,776 条) → balanced_edge5_concise(36,543 条,加了 edge_case_consideration=5 的硬条件、algorithmic-evol 归零、shard-seed 31… |
| 1040 | real | 0.28h | returned | **C3** | 相对 i=505 **只换数据**:换成 balanced_large75838_concise(75,838 条,min-edge-score 4、配额放大),其余逐字相同,只有 save-steps 150→250 是为了在更长的 epoch 上落在同样的相对位置(64% / 85%)。测「覆盖… |
| 1091 | real | 0.15h | returned | **C3** | 相对 i=505 **只换数据**:short_mbpp_mix.jsonl(41,144 条:response ≤500 字符的子集 + MBPP-train ×20 复制),其余逐字相同,save-steps 150→125 同样只是相对位置对齐。读数 64.0%(本 run 所有正式候选里最低… |
| 1167 | real | 0.11h | returned | **both** | 相对 i=633 换了续训用的数据(verified_opencoder_mix.jsonl:自建 concise 25,000 + 外部 OpenCoder 语料 25,000),初始化改成 soup_sft5_450_600,并第一次显式写 max-length 1024、epochs 0.5、… |
| 1246 | real | 0.12h | returned | **both** | 相对 i=1167 数据换成 balanced_doctest_augmented.jsonl(43,342 条带 >>> 示例),lr 3e-6→4e-6、save-steps 200→170,初始化与 epochs 0.5、max-length 1024 相同。数据与 lr 同变。读数 66.7… |
| 1306 | real | 0.25h | returned | **C3** | 改回从 base 训,数据是 doctest_mix50k.jsonl(concise 25,000 + doctest 增强 25,000)。相对 i=505 的差别只有数据 + 显式 max-length 1024 + save-steps 120(为在 774 步上落到 480/600 这两个… |
| 1416 | real | 0.20h | returned | **C3** | 相对 i=1306 **只换数据混合比**:50:50 → 75:25(concise 37,500 + doctest 12,500),启动参数其余逐字相同。剂量反应的第二个点,480/600 平均后 68.7%。 |
| 1489 | real | 0.21h | returned | **C3** | 相对 i=1306 **只换数据混合比**:50:50 → 25:75(concise 12,500 + doctest 37,500),其余逐字相同。剂量反应的第四个点,480/600 平均后 68.0%,i=1523 在 616/774 步被 agent 主动 kill(需要的 480 与 60… |

### 验证序列(55 次)

| i | 档位 | 样本量 | 拿到信号 | 判定了哪些改动 | 读到什么 |
|---|---|---|---|---|---|
| 19 | 3.0 | 20.0 | 是 |  | 0.05 |
| 164 | 3.0 | 20.0 | 是 | c1, c3, c4, c8, c9 | 0.7 |
| 184 | 3.0 | 20.0 | 是 | c1, c3, c4, c8, c9, c18 | 0.8 |
| 190 | 3.0 | 150.0 | 是 | c1, c3, c4, c8, c9 | 0.547 |
| 264 | 3.0 | 20.0 | 是 | c12, c13, c37 | 0.75 |
| 268 | 3.0 | 20.0 | 是 | c12, c13, c37 | 0.75 |
| 372 | 3.0 | 20.0 | 是 | c14, c12 | 0.9 |
| 377 | 3.0 | 150.0 | 是 | c14, c12 | 0.667 |
| 381 | 3.0 | 20.0 | 是 | c14, c18 | 0.85 |
| 388 | 3.0 | 20.0 | 是 | c15 | 0.8 |
| 408 | 3.0 | 20.0 | 是 | c15 | 0.85 |
| 494 | 3.0 | 20.0 | 是 | c16 | 0.85 |
| 497 | 3.0 | 20.0 | 是 | c16, c18 | 0.8 |
| 501 | 3.0 | 150.0 | 是 | c16 | 0.68 |
| 550 | 3.0 | 20.0 | 是 | c16 | 0.85 |
| 553 | 3.0 | 150.0 | 是 | c16 | 0.693 |
| 563 | 3.0 | 20.0 | 是 | c15 | 0.9 |
| 566 | 3.0 | 150.0 | 是 | c15 | 0.667 |
| 626 | 3.0 | 20.0 | 是 | c17, c36 | 0.85 |
| 629 | 3.0 | 150.0 | 是 | c17, c36 | 0.66 |
| 662 | 3.0 | 20.0 | 否 |  | 未拿到 |
| 668 | 3.0 | 20.0 | 是 | c30, c37 | 0.85 |
| 671 | 3.0 | 150.0 | 是 | c30, c37 | 0.687 |
| 674 | 3.0 | 20.0 | 是 | c30, c37 | 0.85 |
| 677 | 3.0 | 150.0 | 是 | c30, c37 | 0.667 |
| 681 | 3.0 | 150.0 | 是 | c18 | 0.693 |
| 689 | 3.0 | 150.0 | 是 | c19 | 0.7 |
| 696 | 3.0 | 150.0 | 是 | c19 | 0.673 |
| 699 | 3.0 | 150.0 | 是 | c19 | 0.667 |
| 704 | 3.0 | 150.0 | 是 | c19 | 0.667 |
| 707 | 3.0 | 150.0 | 是 | c19 | 0.687 |
| 754 | 3.0 | 150.0 | 是 | c33 | 0.673 |
| 758 | 3.0 | 150.0 | 是 | c20 | 0.36 |
| 892 | 3.0 | 150.0 | 是 | c34 | 0.653 |
| 902 | 3.0 | 150.0 | 是 | c15, c34 | 0.68 |
| 1031 | 3.0 | 150.0 | 是 | c28 | 0.66 |
| 1083 | 3.0 | 150.0 | 是 | c29, c19 | 0.667 |
| 1087 | 3.0 | 150.0 | 是 | c21 | 0.7 |
| 1119 | 3.0 | 150.0 | 是 | c31, c19 | 0.64 |
| 1160 | 3.0 | 150.0 | 是 | c15 | 0.7 |
| 1238 | 3.0 | 150.0 | 是 | c32, c37 | 0.673 |
| 1302 | 3.0 | 150.0 | 是 | c22, c37 | 0.667 |
| 1363 | 3.0 | 150.0 | 是 | c22, c19 | 0.707 |
| 1372 | 3.0 | 150.0 | 是 | c19 | 0.687 |
| 1376 | 3.0 | 150.0 | 是 | c15 | 0.687 |
| 1386 | 3.0 | 150.0 | 是 | c15 | 0.673 |
| 1398 | 3.0 | 150.0 | 是 | c15 | 0.673 |
| 1410 | 3.0 | 150.0 | 是 | c22 | 0.693 |
| 1449 | 3.0 | 150.0 | 是 | c26, c19 | 0.687 |
| 1452 | 3.0 | 150.0 | 是 | c15 | 0.687 |
| 1468 | 3.0 | 150.0 | 是 | c21 | 0.68 |
| 1475 | — | — | 是 | c22, c19 | 0.7 |
| 1479 | — | — | 是 | c19, c21 | 0.687 |
| 1540 | 3.0 | 150.0 | 是 | c26 | 0.68 |
| 1549 | — | — | 是 | c27 | 0.7 |

### 异常与存疑

- **3 段训练的受测变量判不出**:i=[113, 121, 124]
- **1 次验证没有拿到信号**:i=[662]
- **分类学缺口提案 1 条**
  - (i=190, i=377, i=1475, i=1488, i=1549)
- **定义缺陷 7 条**
  - (i=113, i=114, i=114)
  - (i=614, i=1004, i=1447, i=1523, i=1528)
  - (i=257, i=581, i=257)
  - (i=146, i=148, i=151, i=151)
  - (i=565, i=567, i=187, i=191)
  - (i=720, i=749, i=762, i=931, i=1090)
  - (i=663, i=666, i=641)
- **边界情形 4 条**
  - (i=120, i=121, i=124)
  - (i=421, i=435, i=1203, i=1306)
  - (i=779, i=575, i=898)
  - (i=685, i=687, i=1049, i=1410)
