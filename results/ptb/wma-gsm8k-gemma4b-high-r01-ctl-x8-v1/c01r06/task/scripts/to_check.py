import json, sys, re
src, dst = sys.argv[1], sys.argv[2]
lim = int(sys.argv[3]) if len(sys.argv)>3 else 0
pat = re.compile(r"\n\nRemember to put your answer", re.S)
n=0
with open(src) as f, open(dst,"w") as o:
    for line in f:
        r=json.loads(line)
        p=r["prompt"]
        # question sits between the instruction header and the 'Remember to' footer
        body=p.split("<start_of_turn>user\n",1)[1].rsplit("<end_of_turn>",1)[0]
        head='(without quotes) where $ANSWER is the answer to the problem.\n\n'
        q=body.split(head,1)[-1]
        q=pat.split(q)[0].strip()
        a=r["completion"].replace("<end_of_turn>","").strip()
        o.write(json.dumps({"question":q,"answer":a})+"\n")
        n+=1
        if lim and n>=lim: break
print("wrote",n,dst)
