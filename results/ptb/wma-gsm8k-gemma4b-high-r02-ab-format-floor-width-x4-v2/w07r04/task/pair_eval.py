#!/usr/bin/env python3
"""Paired item-level comparison of two eval diagnostics (McNemar counts)."""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
A = {r["id"]: r["scored"] for r in a["samples"]}
B = {r["id"]: r["scored"] for r in b["samples"]}
ids = [i for i in A if i in B]
gained = [i for i in ids if A[i] == "I" and B[i] == "C"]
lost = [i for i in ids if A[i] == "C" and B[i] == "I"]
out = {"n_paired": len(ids), "acc_a": a["accuracy"], "acc_b": b["accuracy"],
       "gained_by_b": len(gained), "lost_by_b": len(lost),
       "discordant": len(gained) + len(lost),
       "gained_ids": gained, "lost_ids": lost}
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_ids")}, indent=1))
