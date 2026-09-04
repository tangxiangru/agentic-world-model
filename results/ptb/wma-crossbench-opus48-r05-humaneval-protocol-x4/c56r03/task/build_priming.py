#!/usr/bin/env python3
"""Derive HumanEval-shaped SFT examples from Magicoder-OSS-Instruct Python
solutions: (imports + signature + docstring)  ->  fenced full function.
This is NOT derived from HumanEval; it mirrors only the general shape/format.
Output: data/priming_raw.jsonl {instruction,response,text,src}
"""
import ast, json, os, re, random, argparse
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
from datasets import load_dataset

random.seed(1)
INSTRUCTION = ("\nRead the following function signature and docstring, and fully implement\n"
               "the function described. Your response should only contain the code for\n"
               "this function.\n\n")
PY_BLOCK = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)

def extract_block(resp):
    m = PY_BLOCK.search(resp)
    return m.group(1) if m else None

def build_one(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    lines = code.split("\n")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    if len(funcs) != 1 or others:
        return None
    fn = funcs[0]
    doc = ast.get_docstring(fn, clean=False)
    if not doc or len(doc.strip()) < 25:
        return None
    if len(fn.body) < 2:  # only a docstring, no real body
        return None
    # source segments
    imp_src = "\n".join(ast.get_source_segment(code, n) for n in imports).strip()
    func_src = ast.get_source_segment(code, fn)
    if func_src is None:
        return None
    # header through end of docstring
    doc_node = fn.body[0]
    end = getattr(doc_node, "end_lineno", None)
    if end is None:
        return None
    header = "\n".join(lines[fn.lineno - 1:end])
    if len(func_src) > 3000 or len(func_src) < 60:
        return None
    prompt_code = (imp_src + "\n\n" + header if imp_src else header) + "\n"
    target_code = (imp_src + "\n\n" + func_src if imp_src else func_src)
    instruction = INSTRUCTION + prompt_code
    response = "```python\n" + target_code.rstrip() + "\n```"
    return instruction, response

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/priming_raw.jsonl")
    a = ap.parse_args()
    oss = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    oss_py = [r for r in oss if r["lang"] == "python"]
    random.shuffle(oss_py)
    rows = []
    for r in oss_py:
        blk = extract_block(r["solution"])
        if not blk:
            continue
        built = build_one(blk)
        if built:
            instruction, response = built
            rows.append({"instruction": instruction, "response": response,
                         "text": instruction + "\n\n" + response, "src": "oss_priming"})
        if len(rows) >= a.n:
            break
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} priming rows -> {a.out}")
    if rows:
        print("=== example instruction ===\n", rows[0]["instruction"][:400])
        print("=== example response ===\n", rows[0]["response"][:400])

if __name__ == "__main__":
    main()
