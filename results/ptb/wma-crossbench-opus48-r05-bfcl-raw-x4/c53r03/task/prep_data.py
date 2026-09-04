import json, random, os
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(0)
SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TEMPLATE = open("templates/gemma3_tool_calling.jinja").read()

tok = AutoTokenizer.from_pretrained(SNAP)

def norm_tool(t):
    """Normalize a tool dict to OpenAI-ish {name,description,parameters} used by eval prompt."""
    if "function" in t:
        f = t["function"]
    else:
        f = t
    name = f["name"]
    desc = f.get("description", "") or ""
    params = f.get("parameters") or {"type": "object", "properties": {}}
    # ensure structure
    if "type" not in params:
        params = {"type": "object", "properties": params}
    params.setdefault("properties", {})
    return {"type": "function",
            "function": {"name": name, "description": desc, "parameters": params}}

def build_example(query, tools, call_name, call_args):
    tools_norm = [norm_tool(t) for t in tools]
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": call_name, "arguments": call_args}}]},
    ]
    # full rendered text
    full = tok.apply_chat_template(messages, tools=tools_norm, tokenize=False,
                                   add_generation_prompt=False, chat_template=TEMPLATE)
    # prompt only (up to model turn)
    prompt = tok.apply_chat_template(messages[:1], tools=tools_norm, tokenize=False,
                                     add_generation_prompt=True, chat_template=TEMPLATE)
    return full, prompt

def main():
    ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    out = []
    for ex in ds:
        msgs = ex["messages"]
        asst = [m for m in msgs if m["role"] == "assistant"]
        usr = [m for m in msgs if m["role"] == "user"]
        if len(asst) != 1 or len(usr) != 1:
            continue
        tc = asst[0]["tool_calls"]
        if not tc or len(tc) != 1:
            continue
        try:
            tools = json.loads(ex["tools"])
        except Exception:
            continue
        fn = tc[0]["function"]
        name = fn["name"]
        try:
            args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        query = usr[0]["content"]
        if not query or not query.strip():
            continue
        # sanity: tool name must be among provided tools
        tnames = set()
        for t in tools:
            tnames.add((t.get("function", t)).get("name"))
        if name not in tnames:
            continue
        try:
            full, prompt = build_example(query, tools, name, args)
        except Exception as e:
            continue
        # basic length guard
        if len(full) > 12000:
            continue
        out.append({"text": full, "prompt": prompt, "query": query,
                    "answer": f"{name}({args})"})
    random.shuffle(out)
    print("built", len(out), "examples")
    with open("train_full.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    # decon file: query + answer text
    with open("decon_check.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"text": r["query"] + " " + r["answer"]}) + "\n")

if __name__ == "__main__":
    main()
