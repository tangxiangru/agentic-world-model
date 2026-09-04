#!/usr/bin/env python3
"""Build eval-format function-calling SFT data from xlam-60k single-call examples."""
import json, re, random
from pathlib import Path

random.seed(0)
XLAM = "/home/ben/hf_cache/hub/datasets--lockon--xlam-function-calling-60k/snapshots/26d14ebfe18b1f7b524bd39b404b50af5dc97866/xlam_function_calling_60k.json"

def map_type(t):
    if t is None:
        return "string"
    base = re.split(r"[\[\(,]", t.strip())[0].strip().lower()
    return {
        "str": "string", "string": "string",
        "int": "integer", "integer": "integer",
        "float": "number", "number": "number", "double": "number",
        "bool": "boolean", "boolean": "boolean",
        "list": "array", "array": "array", "tuple": "array", "set": "array",
        "dict": "object", "object": "object",
    }.get(base, "string")

def inner_type(t):
    m = re.search(r"[\[\(]\s*([A-Za-z_]+)", t or "")
    if m:
        return map_type(m.group(1))
    return None

def convert_param(ps):
    t = ps.get("type", "str")
    schema = {"type": map_type(t), "description": ps.get("description", "")}
    if schema["type"] == "array":
        it = inner_type(t)
        if it:
            schema["items"] = {"type": it}
    d = ps.get("default", None)
    has_default = ("default" in ps) and (d is not None) and (d != "")
    if has_default:
        schema["default"] = d
    return schema, has_default

def convert_tool(tool):
    props = {}
    required = []
    for pn, ps in tool["parameters"].items():
        schema, has_default = convert_param(ps)
        props[pn] = schema
        if not has_default:
            required.append(pn)
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", "") or "",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

def main():
    data = json.load(open(XLAM))
    out = []
    skipped = 0
    for d in data:
        try:
            ans = json.loads(d["answers"])
            tools = json.loads(d["tools"])
        except Exception:
            skipped += 1; continue
        if len(ans) != 1:
            continue
        call = ans[0]
        fname = call.get("name")
        cargs = call.get("arguments", {})
        if not isinstance(cargs, dict):
            skipped += 1; continue
        tool_names = {t["name"] for t in tools}
        if fname not in tool_names:
            skipped += 1; continue
        # ensure the called tool's params cover the given args
        called = next(t for t in tools if t["name"] == fname)
        if not all(k in called["parameters"] for k in cargs.keys()):
            skipped += 1; continue
        conv_tools = [convert_tool(t) for t in tools]
        rec = {
            "messages": [
                {"role": "user", "content": d["query"]},
                {"role": "assistant", "tool_calls": [
                    {"type": "function", "function": {"name": fname, "arguments": cargs}}
                ]},
            ],
            "tools": conv_tools,
            "query": d["query"],
            "answer_str": f"{fname}(" + ", ".join(f"{k}={v}" for k, v in cargs.items()) + ")",
        }
        out.append(rec)
    random.shuffle(out)
    Path("train_records.jsonl").write_text("\n".join(json.dumps(r) for r in out))
    # decontamination file: question + answer
    with open("decon_input.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps({"question": r["query"], "answer": r["answer_str"]}) + "\n")
    print(f"total single-call usable: {len(out)}  skipped: {skipped}")

if __name__ == "__main__":
    main()
