#!/bin/bash
# hv_recipe: execute a recipe from the corpus with no LLM anywhere in the loop.
#
# WHY THIS EXISTS. Every run in PostTrainBench was written AND executed by the
# same agent, so "the recipe caused the score" and "the agent caused both" are
# observationally identical -- see doc/split-redesign.md, "The confound no split
# can fix". This agent breaks that by holding the executor fixed and varying only
# the recipe: one deterministic script, K recipes, same everything else.
#
# The split between "recipe" and "execution" is not a judgement call here, it is
# read off the extraction schema:
#
#   VARIES (the recipe -- what tools/extract_recipes.py actually recorded)
#     which datasets, which subsets, how many repeats, how much of each,
#     number of epochs, nominal batch size
#
#   FIXED (execution -- what the extraction never captured, and what the agents
#          therefore differed on invisibly)
#     learning rate and schedule, precision, sequence length, target format,
#     EOS handling, which checkpoint ships, decode settings at eval
#
# `learning_rate` is None on all 24 SFT-only runs of the gsm8k x Qwen3-1.7B cell,
# which is exactly why it belongs in the FIXED column: nothing was recorded to
# vary. The prediction under "the recipe is what mattered" is that the corpus
# ordering survives; under "the agent is what mattered" the spread collapses.
#
# Selected by $AGENT_CONFIG, which run_task.sh passes through --cleanenv:
#   bash src/run_task.sh gsm8k hv_recipe Qwen/Qwen3-1.7B-Base <cid> 4 r042.s0 1
# The suffix after the dot is the seed. Two seeds per recipe is not optional:
# a single-seed contrast here would be uninterpretable against seed noise.
#
# The trainer is embedded rather than shipped as a second file because
# run_task.sh copies only solve.sh into the sandbox -- so this file is a
# complete, auditable record of what ran.
set -x
echo "hv_recipe starting: AGENT_CONFIG=${AGENT_CONFIG}"
nvidia-smi || true
python -c "import torch,transformers,trl,datasets;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'ds',datasets.__version__)"

# The eval's OWN chat template, embedded byte-for-byte.
#
# src/eval/tasks/gsm8k/evaluate.py hands vLLM `chat_template=templates/qwen3.jinja`,
# but templates/ lives at src/eval/templates/ and run_task.sh copies only solve.sh
# into the sandbox -- so the agent cannot read it. Falling back to the tokenizer's
# own template would train on one prompt and grade on another, and nothing in the
# logs would say so. sha256 below is the file at the pinned SHA; the trainer
# re-hashes what it writes, so drift is a loud failure rather than a quiet one.
#   sha256 ddd80c7700049450b953ae6acffa1731d696c5df5b1fbfb2de7e4ead17d708c6
cat > /home/ben/task/qwen3.jinja <<'JINJA'
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" }}{{ '{"name": <function-name>, "arguments": <args-json-object>}' }}{{- "\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
JINJA

cat > /home/ben/task/recipes.json <<'RECIPES'
{
  "r733": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1740314", "corpus_acc": 0.733,
    "sources": [
      {"name": "gsm8k", "repeat": 2},
      {"name": "metamathqa", "subsets": ["GSM_AnsAug", "GSM_Rephrased", "GSM_SV", "GSM_FOBAR"], "cap": 20000}
    ],
    "epochs": 1, "batch_size": 2
  },
  "r699": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1714029", "corpus_acc": 0.699,
    "sources": [{"name": "gsm8k", "repeat": 1}],
    "epochs": 1, "batch_size": 4
  },
  "r600": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1687229", "corpus_acc": 0.600,
    "sources": [{"name": "metamathqa", "cap": 30000}],
    "epochs": 1, "batch_size": 8
  },
  "r544": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1685440", "corpus_acc": 0.544,
    "sources": [{"name": "gsm8k", "repeat": 1}],
    "epochs": 3, "batch_size": 8
  },
  "r401": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1678720", "corpus_acc": 0.401,
    "sources": [{"name": "metamathqa", "subsets": ["GSM_AnsAug", "GSM_FOBAR", "GSM_Rephrased", "GSM_SV"], "cap": 30000}],
    "epochs": 2, "batch_size": 4
  },
  "r042": {
    "corpus_run": "gsm8k_Qwen_Qwen3-1.7B-Base_1696178", "corpus_acc": 0.042,
    "sources": [
      {"name": "metamathqa", "subsets": ["GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"], "cap": 20000},
      {"name": "gsm8k", "repeat": 3}
    ],
    "epochs": 2, "batch_size": 8
  }
}
RECIPES

cat > /home/ben/task/hv_train.py <<'TRAINER'
import hashlib, json, os, random, re, sys

# --- FIXED execution settings. Every arm gets exactly these. -----------------
LR = 1e-5
WARMUP_RATIO = 0.03
MAX_LEN = 2560               # measured, not guessed. The eval's own 10-shot system
                             # message is 1715 tokens, so a rendered row runs
                             # p50 1929 / max 2551 across both sources. At 1024 --
                             # where this started -- 100% of rows truncate, and
                             # under completion_only_loss it is the COMPLETION that
                             # goes, so every row carries zero loss tokens, every
                             # arm scores the base-model floor, and the run exits 0.
                             # 2560 truncates 0.00% of gsm8k and 0.00% of metamath.
MAX_EXAMPLES_SEEN = 20_000   # epochs x rows. Capped on examples SEEN, not on
                             # optimizer steps: examples-seen is what costs wall
                             # time, so every arm gets the same compute budget,
                             # while capping steps instead would have handed the
                             # small-batch arms a quarter of the data. Logged
                             # loudly when it binds. Halved from 40k once MAX_LEN
                             # went to 2560: each row now carries the 1715-token
                             # eval prefix, so 40k rows is ~80M tokens and does not
                             # fit the 3h cell budget alongside the eval.
FEWSHOT = 10                 # inspect_evals/gsm8k default -- match the grader
OUT = "/home/ben/task/final_model"

# The ARM is AGENT_CONFIG, so a default here is not a convenience -- it is the one
# failure that voids the whole experiment quietly. If run_task.sh ever stopped
# forwarding --env AGENT_CONFIG, every cell would fall through to the same recipe
# and the pack would finish green with 6 identical arms and a spread of zero,
# which reads as "recipe does not matter". Refuse instead.
cfg = os.environ.get("AGENT_CONFIG", "").strip()
rid, _, sd = cfg.partition(".")
_RECIPES = json.load(open("/home/ben/task/recipes.json"))
if rid not in _RECIPES:
    raise SystemExit(f"[hv] FATAL: AGENT_CONFIG={cfg!r} -> recipe {rid!r} not in "
                     f"{sorted(_RECIPES)}; refusing to fall back to a default arm")
if not (sd.startswith("s") and sd[1:].isdigit()):
    raise SystemExit(f"[hv] FATAL: AGENT_CONFIG={cfg!r} has no s<N> seed suffix")
seed = int(sd[1:])
recipe = _RECIPES[rid]
print(f"[hv] recipe={rid} seed={seed} spec={json.dumps(recipe)}", flush=True)

# $PROMPT is the only channel that reaches this agent: run_task.sh injects
# MODEL_TO_TRAIN only inside AGENT_CONTEXT_ENV, which is gated on the job having
# an agents/<name>/payload dir, and neither hv agent has one. So the model id has
# to be read out of the rendered prompt -- and get_prompt.py escapes every
# backtick as \` , so the obvious `([^`]+/[^`]+)` capture takes the backslash
# with it and yields 'Qwen/Qwen3-1.7B-Base\', which from_pretrained rejects with
# HFValidationError on every cell. Tolerate the escape, and refuse to guess: a
# silent fallback to a hardcoded default would train the wrong model without
# saying so, and the run would look successful.
_m = re.search(r"\\?`([^`\\]+/[^`\\]+)\\?`", os.environ.get("PROMPT", ""))
MODEL = _m.group(1).strip() if _m else ""
if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", MODEL):
    raise SystemExit(f"[hv] FATAL: no valid base model id in $PROMPT (got {MODEL!r})")
print(f"[hv] base model = {MODEL}", flush=True)

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(seed)
random.seed(seed)
np.random.seed(seed)

# --- the grader's prompt, read from the grader itself ------------------------
# inspect_evals/gsm8k wraps every test question in MATH_PROMPT_TEMPLATE and
# prepends a 10-shot system message. Train on a different wrapper and every arm
# is answering a question it was never shown, which depresses the whole board and
# compresses exactly the between-recipe spread this experiment is trying to read.
#
# The import used to scan `dir(inspect_evals.gsm8k)` -- the PACKAGE -- but the
# constant lives in the `inspect_evals.gsm8k.gsm8k` MODULE and the package does
# not re-export it, so the scan found nothing and the fallback fired every time,
# silently. Verified in opus_5.sif: the package imports fine and the scan returns
# []. Import the module by name instead, and make the miss loud.
#
# The two differ in their closing instruction and in the trailing "Reasoning:"
# line, so the old fallback was NOT the grader's prompt. Measured cost of
# adopting the real one, over all 7473 gsm8k train rows with the eval's chat
# template: p50 1905 -> 1944 tokens, max 2208 -> 2247, still 0.00% over MAX_LEN.
_MM = None
try:
    from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE as _MM
except Exception as e:                                    # noqa: BLE001
    print(f"[hv] WARNING: inspect_evals.gsm8k.gsm8k not importable "
          f"({type(e).__name__}: {e}); falling back to the pinned literal")

# The pinned literal is the grader's template verbatim as of inspect_evals in
# opus_5.sif, with the placeholder renamed. It exists so a cell still runs if the
# import breaks -- and the comparison below turns a silent divergence into a log
# line rather than a mystery in the scores.
_PINNED = ('Solve the following math problem step by step. The last line of your '
           'response should be of the form "ANSWER: $ANSWER" (without quotes) '
           'where $ANSWER is the answer to the problem.\n\n{question}\n\n'
           'Remember to put your answer on its own line at the end in the form '
           '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to '
           'the problem, and you do not need to use a \\boxed command.\n\n'
           'Reasoning:')
if _MM:
    USER_TMPL = _MM.replace("{prompt}", "{question}")
    print(f"[hv] user template from inspect_evals.gsm8k.gsm8k.MATH_PROMPT_TEMPLATE"
          f" (matches pinned literal: {USER_TMPL == _PINNED})", flush=True)
else:
    USER_TMPL = _PINNED

# render() calls USER_TMPL.format(question=...), so any other brace in the
# grader's template would raise mid-run, and a template with no {question} would
# hand every row the same prompt. Both are cheap to rule out here.
_braces = re.findall(r"\{[^}]*\}", USER_TMPL)
if _braces != ["{question}"]:
    raise SystemExit(f"[hv] FATAL: user template has unexpected placeholders "
                     f"{_braces}; refusing to train on a prompt render() cannot fill")
print(f"[hv] user template ({len(USER_TMPL)} chars):\n{USER_TMPL}", flush=True)

CALC = re.compile(r"<<[^>]*>>")


def gsm8k_rows():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        ans = r["answer"]
        body, _, final = ans.rpartition("####")
        body = CALC.sub("", body).strip()
        final = final.strip().replace(",", "")
        if not final:
            continue
        out.append({"q": r["question"].strip(),
                    "a": f"{body}\nANSWER: {final}"})
    return out


def metamath_rows(subsets=None):
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    keep = set(subsets or [])
    out = []
    for r in ds:
        if keep and r.get("type") not in keep:
            continue
        if not keep and not str(r.get("type", "")).startswith("GSM"):
            continue          # the cell's benchmark is gsm8k; MATH_* is off-task
        resp = r["response"]
        mm = re.search(r"The answer is:?\s*([^\s]+)", resp)
        if not mm:
            continue
        body = resp.split("The answer is")[0]
        #: MetaMathQA bodies carry GSM8K's own `#### 752` line. Left in, the
        #: target teaches two answer formats and the grader reads the wrong one.
        #: This is precisely the class of execution detail the extraction never
        #: records -- so it is fixed here, once, identically for every arm.
        body = re.sub(r"\n*####[^\n]*", "", body).strip()
        out.append({"q": r["query"].strip(),
                    "a": f"{body}\nANSWER: {mm.group(1).strip().rstrip('.')}"})
    return out


# --- build the training set from the recipe's data spec ----------------------
pool, base_gsm = [], None
for src in recipe["sources"]:
    if src["name"] == "gsm8k":
        rows = base_gsm = base_gsm or gsm8k_rows()
    elif src["name"] == "metamathqa":
        rows = metamath_rows(src.get("subsets"))
    else:
        raise ValueError(src["name"])
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if src.get("cap"):
        rows = rows[:src["cap"]]
    rows = rows * int(src.get("repeat", 1))
    print(f"[hv] source {src['name']} -> {len(rows)} rows", flush=True)
    pool += rows
random.Random(seed).shuffle(pool)
if base_gsm is None:
    base_gsm = gsm8k_rows()

epochs = float(recipe["epochs"])
if len(pool) * epochs > MAX_EXAMPLES_SEEN:
    keep = max(1, int(MAX_EXAMPLES_SEEN / epochs))
    print(f"[hv] CAP BINDS: {len(pool)}x{epochs} = {len(pool)*epochs:.0f} rows "
          f"seen > {MAX_EXAMPLES_SEEN}; truncating pool to {keep}", flush=True)
    pool = pool[:keep]
else:
    print(f"[hv] cap does not bind: {len(pool)*epochs:.0f} rows seen", flush=True)

# --- render with the eval chat template, fewshot system message --------------
tok = AutoTokenizer.from_pretrained(MODEL)
tpl = "/home/ben/task/qwen3.jinja"
_t = open(tpl).read()
_want = "ddd80c7700049450b953ae6acffa1731d696c5df5b1fbfb2de7e4ead17d708c6"
_got = hashlib.sha256(_t.encode()).hexdigest()
assert _got == _want, f"qwen3.jinja drifted: {_got} != {_want}"
tok.chat_template = _t
print(f"[hv] chat template {tpl} sha256={_got} (the eval's own)", flush=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

#: inspect_evals builds its fewshot block as bare question/answer pairs. Repeating
#: the full instruction once per exemplar instead pushes the prompt past 2k tokens
#: on its own, and at MAX_LEN the target is then truncated away entirely -- the run
#: trains on nothing and still exits 0.
#: The 10 shots come from gsm8k's train split under a fixed seed of my own, not
#: the eval's fewshot_seed. They only have to teach the SHAPE of the answer -- the
#: eval supplies its own exemplars at grading time -- and being fixed across arms
#: is what the contrast needs.
shots = random.Random(12345).sample(base_gsm, FEWSHOT)   # same shots every arm
_INSTR = USER_TMPL.split("{question}")[0].strip()        # never .format(): a real
SYSTEM = (_INSTR + "\n\n"                                # template may hold others
          + "\n\n".join(f"{s['q']}\n{s['a']}" for s in shots))

#: The turn terminator, taken from the chat template rather than from
#: `tok.eos_token`. On a *Base* checkpoint those differ -- eos_token is
#: <|endoftext|> while the template (and therefore vLLM at eval) stops on
#: <|im_end|>. Train on the wrong one and the model never emits a stop token, the
#: answer is buried in the tail, and the score collapses for a reason that looks
#: nothing like a data problem.
_SENT = "HVSTOPPROBE"
_probe = tok.apply_chat_template([{"role": "user", "content": "x"},
                                  {"role": "assistant", "content": _SENT}],
                                 tokenize=False)
STOP = _probe.split(_SENT, 1)[1].strip() if _SENT in _probe else ""
STOP = STOP or tok.eos_token
print(f"[hv] eos_token={tok.eos_token!r} template turn terminator={STOP!r}",
      flush=True)


def render(r):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(question=r["q"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    return {"prompt": prompt, "completion": r["a"] + STOP}


from datasets import Dataset
train = Dataset.from_list([render(r) for r in pool])
print("[hv] EXAMPLE PROMPT >>>\n" + train[0]["prompt"][-1200:]
      + "\n<<< EXAMPLE COMPLETION >>>\n" + train[0]["completion"][:600] + "\n<<<",
      flush=True)
print(f"[hv] {len(train)} training rows, {epochs} epochs", flush=True)

#: Truncation guard. TRL right-truncates the concatenated prompt+completion at
#: max_length, so under completion_only_loss an over-long row loses its COMPLETION
#: and contributes no loss tokens at all -- the trainer reports a healthy loss over
#: the surviving rows, saves a checkpoint, and exits 0. This is how a wrong MAX_LEN
#: reads as "the recipe made no difference" instead of as a bug.
_s = train.select(range(min(2000, len(train))))
_lens = np.array([len(tok(r["prompt"]).input_ids) + len(tok(r["completion"]).input_ids)
                  for r in _s])
_over = float((_lens > MAX_LEN).mean())
print("[hv] rendered tokens: p50 %d p95 %d max %d | over MAX_LEN(%d): %.2f%%"
      % (np.percentile(_lens, 50), np.percentile(_lens, 95), _lens.max(),
         MAX_LEN, 100 * _over), flush=True)
assert _over <= 0.02, (
    f"{100*_over:.1f}% of rows exceed MAX_LEN={MAX_LEN}; those rows would train "
    "on zero loss tokens and the run would still exit 0")

#: Target format. The grader reads the last `ANSWER:` line and nothing else, so a
#: stray `####` from a MetaMathQA body teaches a second answer format and the two
#: compete at decode time. Checked here rather than trusted, because the two
#: sources reach this point through different cleanup paths.
_tail = re.compile(r"\nANSWER: [^\n]+\Z")
_mal = [r for r in _s if not _tail.search(r["completion"][:-len(STOP)])]
assert not _mal, f"{len(_mal)}/{len(_s)} targets do not end in an ANSWER: line"
_hash = [r for r in _s if "####" in r["completion"]]
assert not _hash, f"{len(_hash)}/{len(_s)} targets still carry a #### marker"
print(f"[hv] target format OK on {len(_s)} sampled rows", flush=True)

# --- train -------------------------------------------------------------------
from trl import SFTConfig, SFTTrainer

nominal = int(recipe["batch_size"])
per_dev = min(nominal, 8)
accum = max(1, nominal // per_dev)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="sdpa")
args = SFTConfig(
    output_dir="/tmp/hv_out", overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=per_dev, gradient_accumulation_steps=accum,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=WARMUP_RATIO,
    bf16=True, max_length=MAX_LEN, packing=False,
    completion_only_loss=True,
    logging_steps=25, save_strategy="no", report_to=[], seed=seed,
    gradient_checkpointing=True,
)
SFTTrainer(model=model, args=args, train_dataset=train,
           processing_class=tok).train()

# The final checkpoint is what ships -- always, for every arm. Choosing a
# checkpoint is execution, and several corpus runs lost their score right here.
#: The other half of the terminator problem; the probe above only fixed the training
#: half. `save_pretrained` copies the Base checkpoint's generation_config verbatim, so
#: eos_token_id ships as <|endoftext|> -- the token this run deliberately did NOT train
#: on -- while vLLM takes its stop set from exactly this file. The model then emits
#: STOP, the server does not honour it, and generation runs to the 4000-token cap with
#: `match(numeric=True)` reading the last number out of the tail. Measured over the 27
#: shipped PTB gsm8k checkpoints on disk: all nine that kept the Base default scored
#: <= 0.126, and every score above 0.13 came from one that listed the terminator.
_eos = tok.encode(STOP, add_special_tokens=False)
assert len(_eos) == 1, f"terminator {STOP!r} tokenises to {_eos}, not one id"
_prev = model.generation_config.eos_token_id
_prev = _prev if isinstance(_prev, list) else ([] if _prev is None else [_prev])
model.generation_config.eos_token_id = _eos + [i for i in _prev if i != _eos[0]]

model.save_pretrained(OUT)
tok.save_pretrained(OUT)

#: Decode, written straight into the JSON because transformers REFUSES to save a
#: config with `temperature` set while `do_sample` is False -- it raises, and a raise
#: here costs the whole cell. vLLM ignores `do_sample` (a transformers field) and reads
#: temperature/top_p/top_k, so a checkpoint that names neither is evaluated at the
#: library default of 1.0. Same four fields, same reasoning, as ptb_ops/
#: make_greedy_shadow.py, which exists because two of twenty-three cells on the
#: one-hour board happened to write temperature 0.0 and were the only two above 0.5.
#: This is a FIXED execution setting, not a recipe field: every arm gets exactly these.
_gcp = os.path.join(OUT, "generation_config.json")
with open(_gcp) as _f:
    _gc = json.load(_f)
_gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
with open(_gcp, "w") as _f:
    json.dump(_gc, _f, indent=2)
print(f"[hv] decode pinned: {_gc}", flush=True)

print("[hv] saved", OUT, sorted(os.listdir(OUT)), flush=True)
TRAINER

python /home/ben/task/hv_train.py
TRAINER_RC=$?
echo "hv_recipe trainer exit ${TRAINER_RC}"
ls -la /home/ben/task/final_model || true
echo "hv_recipe done"
# Propagate it. run_task.sh reports this as `exit_code` / `status:` in SOLVE
# DIAGNOSTICS and branches on nothing, so an honest nonzero costs nothing --
# while the bare `exit 0` that used to be here recorded "exited normally" beside
# `final_model_files: 0` for a trainer that had raised. That is the same
# dishonest zero run_task.sh's own pipefail fix exists to remove, reintroduced
# one layer down.
exit "${TRAINER_RC}"
