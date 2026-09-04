#!/usr/bin/env python3
"""Build BFCL-style single-tool-call SFT data from xLAM (APIGen), rendered
through the EXACT eval template + inspect-ai tool serialization so training
strings match grading byte-for-byte."""
import json, argparse, random
from transformers import AutoTokenizer
from inspect_ai.model._openai import openai_chat_tool_param
from bfcl_evaluation_code import create_tool_info_from_dict
from datasets import load_dataset

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

def norm_tool(fn):
    """xLAM function dict -> exact eval tool dict via inspect ToolInfo."""
    d = {"name": fn["name"],
         "description": fn.get("description", "") or "",
         "parameters": fn["parameters"]}
    ti = create_tool_info_from_dict(d)
    return json.loads(json.dumps(openai_chat_tool_param(ti)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAP)
    tok.chat_template = open("templates/gemma3_tool_calling.jinja").read()

    ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)

    rows = []
    lens = []
    n_skip_build = n_skip_len = n_skip_multi = n_skip_fmt = 0
    seen = set()
    for i in idxs:
        if len(rows) >= args.limit:
            break
        ex = ds[i]
        msgs = ex["messages"]
        asst = next((m for m in msgs if m["role"] == "assistant"), None)
        if asst is None:
            continue
        tcs = asst.get("tool_calls")
        if not tcs or len(tcs) != 1:
            n_skip_multi += 1
            continue
        user_msgs = [{"role": m["role"], "content": m["content"]}
                     for m in msgs if m["role"] in ("user", "system")]
        if not user_msgs:
            continue
        tools_raw = ex["tools"]
        if isinstance(tools_raw, str):
            tools_raw = json.loads(tools_raw)
        try:
            tool_dicts = [norm_tool(t.get("function", t)) for t in tools_raw]
        except Exception:
            n_skip_build += 1
            continue
        # target tool call
        tc = tcs[0]["function"]
        name = tc["name"]
        arg_str = tc["arguments"]
        try:
            arguments = json.loads(arg_str) if isinstance(arg_str, str) else arg_str
            if not isinstance(arguments, dict):
                raise ValueError
        except Exception:
            n_skip_fmt += 1
            continue
        # the called function must be one of the provided tools
        names = {t["function"]["name"] for t in tool_dicts}
        if name not in names:
            n_skip_fmt += 1
            continue
        assistant = {"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function",
                                     "function": {"name": name, "arguments": arguments}}]}
        try:
            prompt = tok.apply_chat_template(user_msgs, tools=tool_dicts,
                                             add_generation_prompt=True, tokenize=False)
            full = tok.apply_chat_template(user_msgs + [assistant], tools=tool_dicts,
                                           add_generation_prompt=False, tokenize=False)
        except Exception:
            n_skip_build += 1
            continue
        if not full.startswith(prompt):
            n_skip_build += 1
            continue
        completion = full[len(prompt):]
        if not completion.rstrip("\n").endswith("<end_of_turn>") or "<tool_call>" not in completion:
            n_skip_fmt += 1
            continue
        # dedup
        key = (user_msgs[-1]["content"][:200], name)
        if key in seen:
            continue
        seen.add(key)
        # length
        n_tok = len(tok(full, add_special_tokens=False).input_ids)
        if n_tok > args.max_seq_len:
            n_skip_len += 1
            continue
        lens.append(n_tok)
        rows.append({"prompt": prompt, "completion": completion})

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens.sort()
    def pct(p):
        return lens[int(len(lens)*p)] if lens else 0
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"skips: multi/zero={n_skip_multi} build={n_skip_build} fmt={n_skip_fmt} len={n_skip_len}")
    print(f"token len p50={pct(.5)} p90={pct(.9)} p99={pct(.99)} max={lens[-1] if lens else 0}")

if __name__ == "__main__":
    main()
