#!/usr/bin/env python3
"""Build single-turn, single-call function-calling SFT data.

Sources: Team-ACE/ToolACE and NousResearch/hermes-function-calling-v1.
Output: JSONL with {tools: [openai-wrapped], user: str, name: str, arguments: dict}.
No BFCL data is used. Contamination is checked separately.
"""
import argparse, ast, json, re, sys
from datasets import load_dataset


def wrap_tool(fn: dict) -> dict:
    """Wrap a {name, description, parameters} function dict in OpenAI tool format."""
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    # normalise BFCL/ToolACE 'dict' type to json-schema 'object'
    params = json.loads(json.dumps(params).replace('"type": "dict"', '"type": "object"'))
    return {"type": "function", "function": {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": params,
    }}


def extract_toolace_tools(system: str):
    """Extract the JSON function list embedded in a ToolACE system prompt."""
    i = system.find("[")
    if i < 0:
        return None
    # find matching closing bracket by scanning
    depth = 0
    for j in range(i, len(system)):
        c = system[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(system[i:j + 1])
                except Exception:
                    return None
    return None


def parse_call_list(value: str):
    """Parse a ToolACE assistant value like [func(a="x", b=5)] into calls.

    Returns list of (name, args_dict) or None if not parseable / not literal.
    Function names must be valid identifiers (matches BFCL exec_simple style).
    """
    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        node = ast.parse(s, mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.List):
        return None
    calls = []
    for el in node.elts:
        if not isinstance(el, ast.Call) or not isinstance(el.func, ast.Name):
            return None
        if el.args:  # only keyword args
            return None
        args = {}
        for kw in el.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                return None
        calls.append((el.func.id, args))
    return calls


def build_toolace(max_examples=None):
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    out = []
    for ex in ds:
        tools_raw = extract_toolace_tools(ex["system"])
        if not tools_raw:
            continue
        conv = ex["conversations"]
        # first user + first assistant
        user = None
        asst = None
        for m in conv:
            if m["from"] == "user" and user is None:
                user = m["value"]
            elif m["from"] == "assistant" and user is not None:
                asst = m["value"]
                break
        if user is None or asst is None:
            continue
        calls = parse_call_list(asst)
        if not calls or len(calls) != 1:
            continue
        name, args = calls[0]
        # tool with this name must exist
        tool_names = {t.get("name") for t in tools_raw}
        if name not in tool_names:
            continue
        tools = [wrap_tool(t) for t in tools_raw]
        out.append({"tools": tools, "user": user, "name": name, "arguments": args})
        if max_examples and len(out) >= max_examples:
            break
    return out


def build_toolace_allturns(max_examples=None):
    """Extract every (user -> single-call assistant) adjacent pair as an
    independent single-turn example. Increases volume and diversity."""
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    out = []
    seen = set()
    for ex in ds:
        tools_raw = extract_toolace_tools(ex["system"])
        if not tools_raw:
            continue
        tool_names = {t.get("name") for t in tools_raw}
        tools = None
        conv = ex["conversations"]
        for i in range(1, len(conv)):
            prev, cur = conv[i - 1], conv[i]
            if cur["from"] != "assistant" or prev["from"] != "user":
                continue
            calls = parse_call_list(cur["value"])
            if not calls or len(calls) != 1:
                continue
            name, args = calls[0]
            if name not in tool_names:
                continue
            key = (prev["value"], name, json.dumps(args, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            if tools is None:
                tools = [wrap_tool(t) for t in tools_raw]
            out.append({"tools": tools, "user": prev["value"], "name": name, "arguments": args})
            if max_examples and len(out) >= max_examples:
                return out
    return out


def build_hermes(max_examples=None):
    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
    out = []
    tc_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    for ex in ds:
        try:
            tools_raw = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
        except Exception:
            continue
        if not tools_raw:
            continue
        conv = ex["conversations"]
        user = None
        gpt = None
        for m in conv:
            r = m.get("from")
            if r in ("human", "user") and user is None:
                user = m.get("value")
            elif r in ("gpt", "assistant") and user is not None:
                gpt = m.get("value")
                break
        if not user or not gpt:
            continue
        matches = tc_re.findall(gpt)
        if len(matches) != 1:
            continue
        try:
            call = json.loads(matches[0])
        except Exception:
            continue
        name = call.get("name")
        args = call.get("arguments")
        if not name or not isinstance(args, dict):
            continue
        # tools already in openai-wrapped form? detect
        wrapped = []
        names = set()
        for t in tools_raw:
            if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
                wrapped.append(t)
                names.add(t["function"].get("name"))
            elif isinstance(t, dict) and "name" in t:
                wrapped.append(wrap_tool(t))
                names.add(t["name"])
        if name not in names:
            continue
        out.append({"tools": wrapped, "user": user, "name": name, "arguments": args})
        if max_examples and len(out) >= max_examples:
            break
    return out


def _extract_json_objects(text):
    """Yield top-level JSON objects found in text via brace matching."""
    objs = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            j = i
            instr = False
            esc = False
            while j < n:
                c = text[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = not instr
                elif not instr:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            chunk = text[i:j + 1]
            try:
                objs.append(json.loads(chunk))
            except Exception:
                pass
            i = j + 1
        else:
            i += 1
    return objs


def build_glaive(max_examples=None):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    out = []
    seen = set()
    call_re = re.compile(r"ASSISTANT:\s*<functioncall>(.*?)<\|endoftext\|>", re.DOTALL)
    user_re = re.compile(r"USER:(.*?)(?:ASSISTANT:|FUNCTION RESPONSE:|$)", re.DOTALL)
    for ex in ds:
        sys_text = ex.get("system") or ""
        fns = _extract_json_objects(sys_text)
        fns = [f for f in fns if isinstance(f, dict) and "name" in f and "parameters" in f]
        if not fns:
            continue
        names = {f["name"] for f in fns}
        chat = ex.get("chat") or ""
        cm = call_re.search(chat)
        um = user_re.search(chat)
        if not cm or not um:
            continue
        user = um.group(1).strip()
        raw = cm.group(1).strip()
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
        if not nm:
            continue
        name = nm.group(1)
        if name not in names:
            continue
        # arguments value: after "arguments":  -> '{...}' or {...}
        am = re.search(r'"arguments"\s*:\s*', raw)
        args = {}
        if am:
            rest = raw[am.end():].strip()
            if rest and rest[0] in "'\"":
                q = rest[0]
                end = rest.find(q, 1)
                inner = rest[1:end] if end > 0 else rest[1:]
            else:
                # take the json object
                objs = _extract_json_objects(rest)
                inner = json.dumps(objs[0]) if objs else "{}"
            try:
                args = json.loads(inner)
            except Exception:
                continue
        if not isinstance(args, dict):
            continue
        key = (user, name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        tools = [wrap_tool(f) for f in fns]
        out.append({"tools": tools, "user": user, "name": name, "arguments": args})
        if max_examples and len(out) >= max_examples:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-toolace", type=int, default=None)
    ap.add_argument("--max-hermes", type=int, default=None)
    ap.add_argument("--mode", choices=["first", "all"], default="first")
    ap.add_argument("--max-glaive", type=int, default=0)
    args = ap.parse_args()

    ta = build_toolace_allturns(args.max_toolace) if args.mode == "all" else build_toolace(args.max_toolace)
    print(f"toolace: {len(ta)}", file=sys.stderr)
    he = build_hermes(args.max_hermes)
    print(f"hermes: {len(he)}", file=sys.stderr)
    gl = build_glaive(args.max_glaive) if args.max_glaive else []
    print(f"glaive: {len(gl)}", file=sys.stderr)
    rows = ta + he + gl
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"total: {len(rows)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
