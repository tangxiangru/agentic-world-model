"""Percentage->decimal drilling set (generic). Teaches: when a value is stated as
a percentage in prose, the argument must be the decimal fraction. Includes single
values AND lists, with diverse phrasings ('X%', 'X percent', 'a rate of X%',
'X%, Y%, and Z% respectively'). Also includes control examples where plain numbers
(no percent) stay unchanged, so the model converts only when '%'/'percent' appears.
Generic, domain-matching, independently designed; decontaminated separately.
"""
import json, random
rng = random.Random(555)

def pctstr(dv):
    p = round(dv * 100, 4)
    if p == int(p): p = int(p)
    return p

def phrase_pct(dv):
    p = pctstr(dv)
    return rng.choice([f"{p}%", f"{p} percent"])

def TOOL(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}}
def P(t, d, **e):
    o = {"type": t, "description": d}; o.update(e); return o

out = []

def add(tool, q, args):
    out.append({"query": q, "tools": [tool], "name": tool["function"]["name"], "arguments": args})

# ---- single percentage -> decimal ----
def single():
    dv = round(rng.uniform(0.005, 0.98), rng.choice([2, 3]))
    name, desc, an, rn = rng.choice([
        ("real_return", "Compute inflation-adjusted return given a nominal return rate (decimal) and an amount.", "amount", "return_rate"),
        ("loan_interest", "Compute interest on a loan principal given an interest rate (decimal).", "principal", "interest_rate"),
        ("conversion_yield", "Compute yield from a base given a yield rate (decimal).", "base", "yield_rate"),
    ])
    amt = rng.choice([500, 1000, 2500, 10000, 50000, 130, 7800])
    tool = TOOL(name, desc, {an: P("number", f"The {an}."), rn: P("number", "The rate as a decimal fraction (e.g. 0.05 for 5%).")}, [an, rn])
    q = rng.choice([
        f"Compute {name.replace('_',' ')} for {an} {amt} at a rate of {phrase_pct(dv)}.",
        f"With {an} of {amt} and a rate of {phrase_pct(dv)}, find the {name.replace('_',' ')}.",
        f"The {rn.replace('_',' ')} is {phrase_pct(dv)} and the {an} is {amt}; compute the {name.replace('_',' ')}.",
    ])
    add(tool, q, {an: amt, rn: dv})

# ---- list of percentages -> list of decimals ----
def listed():
    k = rng.randint(3, 6)
    rates = [round(rng.uniform(0.005, 0.09), rng.choice([2, 3])) for _ in range(k)]
    base = rng.choice([1000, 5000, 20000, 100000, 250000, 12000])
    name, desc, bn, rn = rng.choice([
        ("multi_year_projection", "Project a base amount across years using a list of yearly rates given as decimals.", "base_amount", "yearly_rates"),
        ("inflation_adjust_series", "Adjust a base amount over periods using per-period rates as decimals.", "amount", "period_rates"),
    ])
    tool = TOOL(name, desc, {bn: P("number", f"The {bn}."),
                            rn: P("array", "The rates, each a decimal fraction.", items={"type": "number"})}, [bn, rn])
    parts = [phrase_pct(r) for r in rates]
    if rng.random() < 0.5:
        ptxt = ", ".join(parts[:-1]) + f", and {parts[-1]} respectively"
    else:
        ptxt = ", ".join(parts)
    q = rng.choice([
        f"Using a {bn.replace('_',' ')} of {base}, apply the yearly rates {ptxt}. Compute the {name.replace('_',' ')}.",
        f"I predict the rates to be {ptxt}. With a {bn.replace('_',' ')} of {base}, run {name.replace('_',' ')}.",
        f"For a {bn.replace('_',' ')} of {base} and rates of {ptxt}, compute {name.replace('_',' ')}.",
    ])
    add(tool, q, {bn: base, rn: rates})

# ---- control: plain numbers stay as-is (no percent) ----
def control():
    k = rng.randint(3, 6); vals = [rng.randint(1, 500) for _ in range(k)]
    tool = TOOL("sum_list", "Sum a list of numbers.",
                {"numbers": P("array", "Numbers to sum.", items={"type": "number"})}, ["numbers"])
    lt = ", ".join(map(str, vals))
    add(tool, rng.choice([f"Add up the numbers {lt}.", f"What is the total of {lt}?"]), {"numbers": vals})

N = 6000
for i in range(N):
    r = rng.random()
    if r < 0.42: listed()
    elif r < 0.80: single()
    else: control()

# dedup by (name,query)
seen = set(); uniq = []
for r in out:
    key = (r["name"], r["query"])
    if key in seen: continue
    seen.add(key); uniq.append(r)

with open("synth3.jsonl", "w") as f:
    for r in uniq: f.write(json.dumps(r) + "\n")
with open("synth3_queries.jsonl", "w") as f:
    for r in uniq: f.write(json.dumps({"text": r["query"]}) + "\n")
print("wrote", len(uniq))
