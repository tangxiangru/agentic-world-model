import json, random, re
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(0)
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = open("templates/gemma3_tool_calling.jinja").read()
tok = AutoTokenizer.from_pretrained(SNAP)

BASE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
        "number": "number", "integer": "integer", "string": "string",
        "boolean": "boolean", "array": "array", "object": "object",
        "dict": "object", "list": "array", "set": "array", "tuple": "array",
        "any": "string", "none": "string"}

def conv_type(t):
    t0 = t.split(",")[0].strip()
    tl = t0.lower()
    if tl.startswith("list") or tl.startswith("tuple") or tl == "set":
        return "array"
    if tl.startswith("dict"):
        return "object"
    if tl.startswith("callable"):
        return "string"
    if tl.startswith("union") or tl.startswith("optional"):
        return "string"
    return BASE.get(tl, "string")

def array_items(t):
    t0 = t.split(",")[0].strip()
    m = re.match(r"(?:List|list|Tuple|tuple)\[(.+)\]$", t0)
    if m:
        inner = m.group(1).strip()
        return {"type": conv_type(inner)}
    return None

def convert_tool(t):
    # already OpenAI/JSON-schema format (e.g. distilabel origin)
    if isinstance(t.get("function"), dict):
        f = t["function"]
        name = f.get("name")
        if not name:
            return None
        params = f.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(params, dict):
            return None
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params.setdefault("required", [])
        return {"type": "function",
                "function": {"name": name, "description": f.get("description", "") or "",
                             "parameters": params}}
    name = t.get("name")
    if not name:
        return None
    desc = t.get("description", "") or ""
    props = {}
    required = []
    for pname, sch in (t.get("parameters") or {}).items():
        if not isinstance(sch, dict):
            continue
        raw = str(sch.get("type", "string"))
        jt = conv_type(raw)
        entry = {"type": jt, "description": sch.get("description", "") or ""}
        if jt == "array":
            it = array_items(raw)
            if it:
                entry["items"] = it
        d = sch.get("default", None)
        optional = "optional" in raw.lower()
        if not optional:
            required.append(pname)
        else:
            if d not in (None, ""):
                entry["default"] = d
        props[pname] = entry
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}

def build_example(query, tools_norm, call_name, call_args):
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": call_name, "arguments": call_args}}]},
    ]
    full = tok.apply_chat_template(messages, tools=tools_norm, tokenize=False,
                                   add_generation_prompt=False, chat_template=TEMPLATE)
    prompt = tok.apply_chat_template(messages[:1], tools=tools_norm, tokenize=False,
                                     add_generation_prompt=True, chat_template=TEMPLATE)
    return full, prompt

def main():
    ds = load_dataset("argilla/apigen-function-calling", split="train")
    out = []
    for ex in ds:
        try:
            ans = json.loads(ex["answers"])
            tools = json.loads(ex["tools"])
        except Exception:
            continue
        if len(ans) != 1:
            continue
        query = (ex.get("query") or "").strip()
        if not query:
            continue
        call = ans[0]
        name = call.get("name")
        args = call.get("arguments")
        if not name or not isinstance(args, dict):
            continue
        tools_norm = [convert_tool(t) for t in tools]
        tools_norm = [t for t in tools_norm if t]
        tnames = {t["function"]["name"] for t in tools_norm}
        if name not in tnames:
            continue
        # ensure answer arg keys exist in chosen tool params
        chosen = next(t for t in tools_norm if t["function"]["name"] == name)
        pkeys = set(chosen["function"]["parameters"]["properties"].keys())
        if not set(args.keys()).issubset(pkeys):
            continue
        try:
            full, prompt = build_example(query, tools_norm, name, args)
        except Exception:
            continue
        if len(full) > 12000:
            continue
        out.append({"text": full, "prompt": prompt, "query": query,
                    "answer": f"{name}({args})", "origin": ex.get("origin", "")})
    random.shuffle(out)
    print("built", len(out), "examples")
    from collections import Counter
    print("origins", Counter(r["origin"] for r in out))
    with open("train_v2.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open("decon_v2.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["query"] + " " + r["answer"]}) + "\n")

if __name__ == "__main__":
    main()
