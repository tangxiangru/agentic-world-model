# 从已发生的轨迹回答：WMA 在什么时候有用

本次只核查三条指定轨迹及一条同轨迹反例。所有时间为 UTC；行号指原始 `solve_parsed.txt`。重点不是 WMA 给了多少建议，而是 scientist 在建议前做了什么、WMA 增加了什么事实、建议后有什么动作与实际结果。没有用反事实节约量替代观察。

三个结果根目录如下，后文简称 C3、C2、D1：

- C3 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r03_formal_r8/gsm8k_google_gemma-3-4b-pt_91022`
- C2 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-c-probe-before-fail-x4-v2_w08r02_formal_r8/gsm8k_google_gemma-3-4b-pt_91021`
- D1 = `/home/robtang_google_com/gangda_workspace/agentic-world-model/data/ptb/results/claude_vertex_high_awm_claude-opus-5_1m__10h_gangda_wma_evolve_wma-gsm8k-gemma4b-high-r02-d-checkpoint-precondition-x4-v2_w09r01_formal_r9/gsm8k_google_gemma-3-4b-pt_91024`

## 1. w08r03 / exp-03：scientist 的方向已经对了，WMA 发现实现其实没有执行该方向

**之前：greedy 是 scientist 已经选定的方向，不应归功给这次 WMA review。**9月3日18:55:09，scientist在上一张卡的 next_step 主动写明下张卡做 greedy variant（C3/solve_parsed.txt:6026）。18:55:17 构建该variant（:6046）；18:55:57锁前card说配置只有 `do_sample:false`、移除了top_k/top_p，声称这样会变成greedy（:6104、:6120）。更早16:38:48的构建脚本已经这样写（:4829）。前文的 WMA 已提醒base是sampled，scientist知道该问题；但在这张卡前的可见轨迹里，没有发现 scientist 已经确认 `do_sample` 对vLLM无效、必须写temperature0的证据。

**WMA 带来的新信息：它检查的不是“配置文件写没写”，而是这些字节实际被运行库解释成什么。**19:00:39首次summary返回L3=no（:6257）；19:00:42 scientist读到具体probe（:6284–6285）：原来parent的有效参数是t1/top_p.95/top_k64；所谓greedy文件经resolver后是t1/top_p1/top_k0，仍sampling且范围更宽。加入temperature0后才得到greedy。首要修复是同时改artifact和`make_variant.py`（:6272、:6274）。

**之后：有明确承认、立即修改、复核和真实执行。**19:00:48 scientist说：

> “The verdict caught a real defect: `do_sample=false` is a no-op for vLLM” （:6296）

19:00:52改脚本加入`gc["temperature"]=0.0`（:6312）；19:00:55读回的diff确实有temperature0（:6345）；19:01:13以“WMA probe p2 showed the locked artifact was not greedy”重新lock（:6366）。19:06:39新review变yes（:6403），19:06:44才启动评估（:6409）。19:08:38实际工具输出exit0、accuracy .653333、150/150 stop、format1.0（:6441–6447）；19:09:01 scientist又用vLLM自己的resolver确认temperature0（:6543）。卡里同权重的sampled comparator是.5533，差+10.0pp；27题修复、12题回退（`task/memory/cards/exp-03.yaml/result`；trace:6765）。

**这条轨迹能说什么：**WMA 在scientist已经选对实验、但一个API语义误解令实验名不副实的时候，提供了正确而此前未被落实的实现证据；建议在launch前被采用，最后真实执行的是预期的greedy实验，并出现+10pp的该次测量增益。这是三例中最强的“具体修复后有实际性能收益”的链条。但+10pp是corrected greedy对sampled的结果，不是有/无WMA随机对照效应；不能声称没有WMA就永远不会自行发现，也不能把整个greedy想法归功给WMA。

## 2. w08r02 / exp-06：scientist 把失败当成功，WMA 用实际文件状态纠正

**之前：并不是 scientist 已经知道 checkpoint 不存在。**9月3日22:25:57，包装checkpoint900时用`grep "packaged|shards"`过滤输出（C2/solve_parsed.txt:11953）；22:27:11只看到“packaged ... copied from base ...”一行（:11994）。22:28:06 card仍声明checkpoint900的weights已包装并asserted（:12064–12065），它是三arm评估的第一个arm（:12009）。前面的可见轨迹一直把900当作存在的中期点，没有发现它已被rotation删除的承认。

**WMA 带来的新信息：卡上的“检查已通过”与盘上产物不一致。**22:32:40初判L0=no/L1=no（:12214），直接指出该served目录没有config和safetensors，checkpoint900已经被save_total_limit3轮转删除，并强调card里“packaged and asserted”不符合盘面（:12216）。

**之后：9秒后查盘，证据与WMA一致。**22:32:49 scientist执行ls并改用1200（:12230）。22:33:23实际输出只列出源checkpoint1200/1500/1553/final；旧served目录只有配置/tokenizer文件，新1200包装显示**2个safetensors shard，8.6GB**（:12259–12276）。22:33:53 scientist写下：

> “package_model.py's own >5 GB shard assertion fired, but I had grepped its output ... and read the first line as success.” （:12322）

同一时刻relock明说“WMA verdict caught a dead arm before launch”，换1200并使单个arm失败不会终止其余arm（:12336）。22:40:26新review为yes（:12384）。22:55:52三个arm都返回结果，22:56:02表中soup234=.7058、soup37=.7013、checkpoint1200=.6801（n1319，:12511及邻近表）；22:57:01记录p=.520未满足替换标准，保留incumbent（:12575）。

**这条轨迹能说什么：**WMA在scientist过滤/误读执行输出后，独立核对产物，确实让一个没有weights的评估arm在launch前被替换为可加载的arm。效果是“坏计划被实际修复并完成”，没有观察到这次checkpoint替换提高最后模型分数：1200比incumbent更差，最终也没被选中。没有运行原始坏arm的对照，因此不报节省了多少分钟。

这里还有细节：第二次review指出1200在权重空间距final非常近，不能代表真正中期checkpoint（:12386）；scientist在读outcome后沿用这条解释（:12566）。这是评估含义的澄清，不是把差checkpoint变成了好checkpoint。

## 3. w09r01 / exp-06：scientist 检查了错误的一组条件，WMA 找到了真正的接口不匹配

**之前：scientist知道要做格式处理，但没有验证关键load接口。**9月4日00:04:44的`soup.py`会校验三个源checkpoint之间key集合相同，然后将原磁盘keys直接交给模型对象`load_state_dict`（D1/solve_parsed.txt:8760、:8765）。:8755甚至有一行被`if False`禁用的prefix处理，因此不能把“存在前缀问题的可能性”说成WMA从零提出。00:05:26卡只声称已附加tokenizer/processor/generation_config、assert源keys一致（:8814）；没有模型可load的smoke。卡的trigger说明soup本身来自前两次WMA cheaper_variant（:8803），但这次最直接的帮助是实现修复。

**WMA 带来的新信息：源与源一致，不代表源与模型接口一致。**00:12:36初判L0=no、L3=defer（:9011）。00:12:43给出具体差异（:9035、:9045、:9047）：磁盘保存的是`language_model.model.*`，live model是`model.language_model.*`，0/883 keys匹配；转换映射只在`from_pretrained`执行，不在当前`load_state_dict`执行。它给出两种明确修法，scientist选了“直接按磁盘keys写safetensors，再走正常from_pretrained”。

**之后：明确承认，实改，真实load通过。**00:12:58 scientist说：

> “Correct catch — the merge would have crashed on key-name mismatch.” （:9059）

00:13:20重写脚本，取消模型对象load，加入保存后CPU reload（:9063、:9135）；00:13:26relock完整记录WMA发现与所改接口（:9166）。00:20:30新verdict为yes（:9273）；00:20:38运行merge（:9282）。00:21:13真实工具输出：三份各883 tensors accumulated、soup saved、**`reload OK: 4.300B params`**（:9317–9322）。随后评估完成；n150与incumbent同为.7733，n500 soup=.774、incumbent=.790，拒绝soup（card result/conclusion；:9492）。

**这条轨迹能说什么：**WMA在已有static检查与真正消费者接口之间补上了具体证据，scientist按证据修复，结果成功保存并加载。不能把这个可靠性收益包装成性能收益；它同时提出的soup想法在本次最终没有优于incumbent。也不能声称节省了原来整次运行成本，因为原始脚本没真正跑到底。

## 同一 scientist 的反例：泛化的数据过滤建议并没有同样扎实

w08r02 / exp-04，9月3日20:14:02，WMA要求随机抽查标签，并建议删除4887条“答案数字也出现在问题中”的orca行（C2/solve_parsed.txt:10550–10551）。Scientist20:14:23照做了抽查，但没有执行删除（:10563–10571）；20:14:24的实际输出包含“25公里/5公里每小时=5小时”的正确样本（:10593–10594）。该答案5既在题面中也正确，说明这个启发式可以误伤。20:14:41按原数据启动（:10642），事后card明确说14/14抽查正确、拒绝了filter（:10950）。

这里应拆开两个观察：WMA触发额外检查是真的；“这些行大多错误，应该删除”并没有得到本地证据支持。由于没有执行过滤，不存在已经观察到的WMA致害，只存在一条被scientist证据挡下的建议。14条样本也不能推出所有4887条都正确。

## 由这些轨迹本身支持的回答

这三例里，WMA最明确有用的时刻都是：**scientist 已经有具体方案，并且确信配置或检查通过；WMA读取运行语义或真实产物，发现一个具体不一致；这个不一致在launch前能改，scientist实际改了。**三种被纠正的确信分别是“do_sample=false就greedy”“输出说packaged就有weights”“源keys一致就能load进模型”。

可观察的收益层次不同：第一例是修正实施并测到+10pp；第二例是补齐了一个原本不存在的artifact，完成评估；第三例是让merge保存/加载真实通过，但模型未变好。把它们都写成“WMA提高准确率”会失真，把它们都写成“只给建议没有作用”同样不符合轨迹。
