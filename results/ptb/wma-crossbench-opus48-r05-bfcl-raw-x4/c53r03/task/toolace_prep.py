import json, re, ast
from datasets import load_dataset
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = open("templates/gemma3_tool_calling.jinja").read()
tok = AutoTokenizer.from_pretrained(SNAP)

TYPEMAP = {"dict": "object", "int": "integer", "float": "number", "str": "string",
           "string": "string", "bool": "boolean", "boolean": "boolean",
           "array": "array", "list": "array", "tuple": "array", "number": "number",
           "integer": "integer", "object": "object", "any": "string"}

def conv_type(t):
    if not isinstance(t, str):
        return "string"
    t0 = t.split(",")[0].strip().lower()
    if t0.startswith("list") or t0.startswith("tuple") or t0 == "array":
        return "array"
    if t0.startswith("dict"):
        return "object"
    return TYPEMAP.get(t0, "string")

def conv_param(sch):
    if not isinstance(sch, dict):
        return {"type": "string"}
    jt = conv_type(sch.get("type", "string"))
    out = {"type": jt, "description": sch.get("description", "") or ""}
    if jt == "array":
        it = sch.get("items")
        if isinstance(it, dict) and "type" in it:
            out["items"] = {"type": conv_type(it["type"])}
    if jt == "object" and isinstance(sch.get("properties"), dict):
        out["properties"] = {k: conv_param(v) for k, v in sch["properties"].items()}
    d = sch.get("default", None)
    if d not in (None, "") and not sch.get("required_flag"):
        out["default"] = d
    return out

def convert_tool(t):
    name = t.get("name")
    if not name:
        return None
    params = t.get("parameters") or {}
    props_in = params.get("properties") or {}
    req = params.get("required") or []
    if req is None:
        req = []
    props = {}
    for pn, sch in props_in.items():
        props[pn] = conv_param(sch)
    return {"type": "function",
            "function": {"name": name, "description": t.get("description", "") or "",
                         "parameters": {"type": "object", "properties": props,
                                        "required": list(req)}}}

def extract_tools(system):
    idx = system.find("[")
    if idx < 0:
        return None
    s = system
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(idx, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        tools = json.loads(s[idx:end + 1])
    except Exception:
        return None
    if not isinstance(tools, list):
        return None
    return tools

def parse_calls(text):
    """Parse '[Func(a=1), Func2(b=2)]' -> list of (name, args). Return None on failure."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        node = ast.parse(text, mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.List):
        return None
    calls = []
    for el in node.elts:
        if not isinstance(el, ast.Call) or not isinstance(el.func, ast.Name):
            return None
        if el.args:
            return None
        args = {}
        for kw in el.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                return None
        calls.append((el.func.id, args))
    return calls

def build(query, tools_norm, name, args):
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": name, "arguments": args}}]},
    ]
    full = tok.apply_chat_template(messages, tools=tools_norm, tokenize=False,
                                   add_generation_prompt=False, chat_template=TEMPLATE)
    prompt = tok.apply_chat_template(messages[:1], tools=tools_norm, tokenize=False,
                                     add_generation_prompt=True, chat_template=TEMPLATE)
    return full, prompt

def main():
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    out = []
    n_tools_fail = 0
    for ex in ds:
        conv = ex["conversations"]
        if len(conv) < 2:
            continue
        if conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
            continue
        tools = extract_tools(ex["system"])
        if not tools:
            n_tools_fail += 1
            continue
        calls = parse_calls(conv[1]["value"])
        if not calls or len(calls) != 1:
            continue
        name, args = calls[0]
        if not isinstance(args, dict):
            continue
        tools_norm = [convert_tool(t) for t in tools]
        tools_norm = [t for t in tools_norm if t]
        tmap = {t["function"]["name"]: t for t in tools_norm}
        if name not in tmap:
            continue
        pkeys = set(tmap[name]["function"]["parameters"]["properties"].keys())
        if not set(args.keys()).issubset(pkeys):
            continue
        query = conv[0]["value"].strip()
        if not query:
            continue
        try:
            full, prompt = build(query, tools_norm, name, args)
        except Exception:
            continue
        if len(full) > 12000:
            continue
        out.append({"text": full, "prompt": prompt, "query": query,
                    "answer": f"{name}({args})", "origin": "toolace"})
    print("toolace built", len(out), "tools_fail", n_tools_fail)
    with open("train_toolace.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open("decon_toolace.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["query"] + " " + r["answer"]}) + "\n")

if __name__ == "__main__":
    main()
