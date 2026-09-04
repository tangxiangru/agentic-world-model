import json, re, sys, os
from collections import Counter
path = sys.argv[1]; out = sys.argv[2]
d = json.load(open(path))
samples = d["samples"]
rows=[]; tags=Counter()
for s in samples:
    comp = s["output"]["choices"][0]["message"]["content"]
    if isinstance(comp, list): comp = "".join(c.get("text","") for c in comp)
    stop = s["output"]["choices"][0].get("stop_reason")
    corr = s["scores"]["match"]["value"] == "C"
    tgt = s["target"] if isinstance(s["target"], str) else s["target"][0]
    has_marker = "ANSWER:" in comp
    # what the grader read
    ans = s["scores"]["match"].get("answer")
    if corr: tag="correct"
    elif stop == "max_tokens": tag="no_stop_maxtokens"
    elif not has_marker: tag="no_answer_marker"
    else:
        # marker present: is the last ANSWER line the right number?
        m = re.findall(r"ANSWER:\s*([^\n]*)", comp)
        last = m[-1].strip().replace(",","").replace("$","").rstrip(".") if m else ""
        tag = "reasoning_wrong" if last and last != tgt else "format_after_marker"
    tags[tag]+=1
    rows.append({"id": s["id"], "question": s["input"], "gold": tgt, "tag": tag,
                 "stop_reason": stop, "n_chars": len(comp), "read_answer": ans,
                 "completion_tail": comp[-400:]})
print(tags, "n=", len(rows))
print("mean chars:", sum(r["n_chars"] for r in rows)/len(rows))
json.dump(rows, open(out,"w"), indent=1)
# watch set = the failures
with open(out.replace(".json","_watch.jsonl"),"w") as f:
    for r in rows:
        if r["tag"]!="correct":
            f.write(json.dumps({"id":r["id"],"question":r["question"],"gold":r["gold"]})+"\n")
print("wrote", out)
for t in ["no_stop_maxtokens","no_answer_marker","reasoning_wrong","format_after_marker"]:
    ex=[r for r in rows if r["tag"]==t][:1]
    if ex: print("\n=====",t,"| gold",ex[0]["gold"],"| read",repr(ex[0]["read_answer"])[:80],"\n...",ex[0]["completion_tail"][-350:])
