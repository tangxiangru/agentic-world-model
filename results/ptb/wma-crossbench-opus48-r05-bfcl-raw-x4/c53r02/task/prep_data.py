import json, re, random
from datasets import load_dataset

random.seed(0)

d = load_dataset('minpeter/xlam-function-calling-60k-parsed', split='train')

out = []
kept = 0
for ex in d:
    msgs = ex['messages']
    if len(msgs) < 2:
        continue
    user = msgs[0]
    asst = msgs[-1]
    if user['role'] != 'user' or asst['role'] != 'assistant':
        continue
    tcs = asst.get('tool_calls')
    if not tcs or len(tcs) != 1:
        continue
    tools = ex['tools']
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            continue
    if not tools:
        continue
    call = tcs[0]['function']
    name = call['name']
    # ensure called function is among provided tools
    tool_names = set()
    for t in tools:
        fn = t.get('function', t)
        tool_names.add(fn.get('name'))
    if name not in tool_names:
        continue
    args = call['arguments']
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
        except Exception:
            continue
    else:
        args_obj = args
    if not isinstance(args_obj, dict):
        continue
    q = user['content']
    if not q or not isinstance(q, str) or len(q.strip()) < 5:
        continue
    out.append({
        'query': q.strip(),
        'tools': tools,          # OpenAI format: [{"type":"function","function":{...}}]
        'name': name,
        'arguments': args_obj,
    })
    kept += 1

print('kept single-call examples:', kept)
random.shuffle(out)
with open('xlam_single.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')

# Write plain-text queries for contamination check
with open('xlam_queries.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps({'text': r['query']}) + '\n')
print('wrote xlam_single.jsonl and xlam_queries.jsonl')
