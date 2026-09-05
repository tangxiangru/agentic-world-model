import json, sys
def call_str(ans):
    args=", ".join(f"{k}={v}" for k,v in ans["arguments"].items())
    return f'{ans["name"]}({args})'
for path in sys.argv[1:]:
    outp=path.replace(".jsonl","_deconcheck.jsonl")
    with open(path) as f, open(outp,"w") as o:
        for line in f:
            r=json.loads(line)
            o.write(json.dumps({"question":r["query"],"answer":call_str(r["answer"])})+"\n")
    print("wrote",outp)
