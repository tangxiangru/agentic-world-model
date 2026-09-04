#!/usr/bin/env python3
"""Build BFCL-style single-tool-call SFT data from argilla/apigen-function-calling.

Renders prompt+completion with the exact grader chat template
(templates/gemma3_tool_calling.jinja) so training format == eval format.
"""
import argparse
import json
import random

from datasets import load_dataset
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
TMPL = open("/home/ben/task/templates/gemma3_tool_calling.jinja").read()

TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "double": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array",
    "dict": "object", "object": "object",
}


def map_type(t):
    if not isinstance(t, str):
        return "string"
    base = t.split(",")[0].strip().lower()  # strip ", optional"
    return TYPE_MAP.get(base, "string")


def convert_param(name, spec):
    if not isinstance(spec, dict):
        return {"type": "string"}
    p = {"type": map_type(spec.get("type"))}
    if spec.get("description"):
        p["description"] = spec["description"]
    # keep a meaningful default like the eval schemas do
    if "default" in spec and spec["default"] not in ("", None):
        p["default"] = spec["default"]
    if spec.get("enum"):
        p["enum"] = spec["enum"]
    if p["type"] == "array" and isinstance(spec.get("items"), dict):
        it = spec["items"]
        p["items"] = {"type": map_type(it.get("type"))}
    return p


def convert_tool(tool, required_keys=None):
    params = tool.get("parameters") or {}
    props = {}
    for k, v in params.items():
        props[k] = convert_param(k, v)
    if required_keys is None:
        required = [k for k, v in params.items()
                    if not (isinstance(v, dict) and v.get("default") not in ("", None))]
    else:
        required = [k for k in required_keys if k in props]
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=30000)
    ap.add_argument("--max-tools", type=int, default=6)
    ap.add_argument("--out", default="data/train_raw.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tk = AutoTokenizer.from_pretrained(SNAP)
    ds = load_dataset("argilla/apigen-function-calling", split="train")

    rows = []
    seen = set()
    n_bad = 0
    for r in ds:
        try:
            answers = json.loads(r["answers"])
            tools = json.loads(r["tools"])
        except Exception:
            n_bad += 1
            continue
        if len(answers) != 1 or not tools:
            continue
        if len(tools) > args.max_tools:
            continue
        ans = answers[0]
        gold_name = ans.get("name")
        gold_args = ans.get("arguments", {})
        if not isinstance(gold_args, dict):
            continue
        tool_by_name = {t.get("name"): t for t in tools if isinstance(t, dict) and t.get("name")}
        if gold_name not in tool_by_name:
            continue
        gold_tool = tool_by_name[gold_name]
        gold_params = gold_tool.get("parameters") or {}
        # gold args keys must be valid params of the gold tool
        if not set(gold_args.keys()).issubset(set(gold_params.keys())):
            continue
        query = (r.get("query") or "").strip()
        if not query:
            continue
        key = (gold_name, query)
        if key in seen:
            continue
        seen.add(key)

        # convert tools; gold tool marks used args as required (BFCL pattern)
        conv_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            rk = list(gold_args.keys()) if t.get("name") == gold_name else None
            conv_tools.append(convert_tool(t, required_keys=rk))

        messages = [{"role": "user", "content": query}]
        assistant = {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": gold_name, "arguments": gold_args}}]}
        try:
            prompt = tk.apply_chat_template(messages, tools=conv_tools, chat_template=TMPL,
                                            tokenize=False, add_generation_prompt=True)
            full = tk.apply_chat_template(messages + [assistant], tools=conv_tools, chat_template=TMPL,
                                          tokenize=False, add_generation_prompt=False)
        except Exception:
            n_bad += 1
            continue
        if not full.startswith(prompt):
            n_bad += 1
            continue
        completion = full[len(prompt):].rstrip("\n")
        if not completion.endswith("<end_of_turn>"):
            n_bad += 1
            continue
        rows.append({"prompt": prompt, "completion": completion, "query": query,
                     "gold_name": gold_name, "n_tools": len(conv_tools)})

    random.seed(args.seed)
    random.shuffle(rows)
    if len(rows) > args.max_examples:
        rows = rows[:args.max_examples]

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.out} (bad={n_bad})")
    from collections import Counter
    print("n_tools dist:", Counter(r["n_tools"] for r in rows).most_common())


if __name__ == "__main__":
    main()
