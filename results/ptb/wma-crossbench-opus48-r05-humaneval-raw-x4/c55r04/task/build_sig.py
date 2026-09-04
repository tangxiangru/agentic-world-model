import json, re, random, ast
random.seed(0)
# Build HumanEval-style "complete this function" examples from independent corpus.
# For each response containing a python code block with a top-level function that
# has a docstring, create: instruction = eval-style prompt + signature+docstring,
# response = ```python full function ```.
INSTRUCTION = ("Read the following function signature and docstring, and fully implement\n"
    "the function described. Your response should only contain the code for\n"
    "this function.\n\n")

rows=[json.loads(l) for l in open('clean_pool.jsonl')]
out=[]
pat=re.compile(r"```python\n(.*?)```", re.DOTALL)
for r in rows:
    m=pat.search(r['response'])
    if not m: continue
    code=m.group(1).rstrip()
    if 'def ' not in code: continue
    try:
        tree=ast.parse(code)
    except Exception:
        continue
    # find first top-level function def with a docstring
    fn=None
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            ds=ast.get_docstring(node)
            if ds and len(ds)>25:
                fn=node; break
    if fn is None: continue
    lines=code.split('\n')
    # function spans fn.lineno..end; capture the def line(s) through end of docstring
    start=fn.lineno-1
    # find end of docstring: the docstring is first statement
    body0=fn.body[0]
    if not (isinstance(body0,ast.Expr) and isinstance(getattr(body0,'value',None),ast.Constant)):
        continue
    doc_end=body0.end_lineno  # 1-indexed inclusive
    func_end=fn.end_lineno
    # collect any decorators start
    if fn.decorator_list:
        start=min(d.lineno-1 for d in fn.decorator_list)
    sig_doc='\n'.join(lines[start:doc_end])
    full='\n'.join(lines[start:func_end])
    # require function body beyond docstring
    if func_end-doc_end < 1: continue
    if len(full)>3500 or len(sig_doc)>1500: continue
    instr=INSTRUCTION+sig_doc+"\n"
    resp="```python\n"+full+"\n```"
    out.append({"instruction":instr,"response":resp})
random.shuffle(out)
with open('sig_pool.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("sig examples:",len(out))
if out:
    print("=== sample instruction ===")
    print(out[0]['instruction'][:500])
    print("=== sample response ===")
    print(out[0]['response'][:400])
