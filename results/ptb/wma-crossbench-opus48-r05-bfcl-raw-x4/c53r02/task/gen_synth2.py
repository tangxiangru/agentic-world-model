"""Augmented generic synthetic generator (v2). Adds skills:
 - pure positional order mapping with generic param names (a,b / x,y): first value->first param
 - stronger percentage->decimal (single and list)
 - integer identifier params kept as ints; price/amount extraction
 - optional boolean flags
All generic, domain-matching, independently designed; decontaminated separately.
"""
import json, random
rng = random.Random(20260904)

def pct(dv):
    p = round(dv * 100, 2)
    if p == int(p): p = int(p)
    return f"{p}%"

def TOOL(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False}}}
def P(t, d, **e):
    o = {"type": t, "description": d}; o.update(e); return o

def g_order():
    a = rng.randint(2, 200); b = rng.randint(2, 200)
    name, desc = rng.choice([
        ("least_common_multiple", "Return the least common multiple of two integers a and b."),
        ("greatest_common_divisor", "Return the greatest common divisor of two integers a and b."),
        ("integer_ratio", "Return the ratio of integer a to integer b."),
        ("modulo", "Return a modulo b for integers a and b."),
    ])
    tool = TOOL(name, desc, {"a": P("integer", "The first integer, a."), "b": P("integer", "The second integer, b.")}, ["a", "b"])
    word = name.replace("_", " ")
    q = rng.choice([
        f"Find the {word} of {a} and {b}.",
        f"What is the {word} of {a} and {b}?",
        f"Compute the {word} for {a} and {b}.",
        f"I need the {word} of the numbers {a} and {b}.",
    ])
    return tool, q, {"a": a, "b": b}

def g_order_xy():
    x = round(rng.uniform(-20, 20), rng.choice([0, 1, 2])); y = round(rng.uniform(-20, 20), rng.choice([0, 1, 2]))
    x = int(x) if x == int(x) else x; y = int(y) if y == int(y) else y
    name, desc = rng.choice([
        ("euclidean_distance_1d", "Distance between point x and point y on a line."),
        ("power", "Raise x to the power of y."),
        ("difference", "Compute x minus y."),
    ])
    tool = TOOL(name, desc, {"x": P("number", "The value x."), "y": P("number", "The value y.")}, ["x", "y"])
    q = rng.choice([
        f"Compute {name.replace('_',' ')} with x = {x} and y = {y}.",
        f"Using x={x} and y={y}, evaluate {name.replace('_',' ')}.",
        f"Take x as {x} and y as {y}; run {name.replace('_',' ')}.",
    ])
    return tool, q, {"x": x, "y": y}

def g_pct_single():
    dv = round(rng.uniform(0.01, 0.95), 2)
    amt = rng.choice([100, 250, 1200, 5000, 40000, 99.99, 750, 3200])
    name, desc, pn, dn = rng.choice([
        ("tax_amount", "Compute tax owed on an amount given a tax rate.", "amount", "tax_rate"),
        ("tip_amount", "Compute a tip on a bill given a tip rate.", "bill", "tip_rate"),
        ("commission", "Compute commission earned on sales given a commission rate.", "sales", "rate"),
    ])
    tool = TOOL(name, desc, {pn: P("number", f"The {pn}."), dn: P("number", f"The rate as a decimal.")}, [pn, dn])
    q = rng.choice([
        f"Compute the {name.replace('_',' ')} on {'$' if pn!='sales' else ''}{amt} at a rate of {pct(dv)}.",
        f"With {pn} of {amt} and a rate of {pct(dv)}, what is the {name.replace('_',' ')}?",
    ])
    return tool, q, {pn: amt, dn: dv}

def g_pct_list():
    k = rng.randint(3, 6)
    rates = [round(rng.uniform(0.005, 0.09), 3) for _ in range(k)]
    base = rng.choice([1000, 5000, 20000, 100000, 250000])
    tool = TOOL("cumulative_growth",
                "Apply a sequence of per-period rates (given as decimals) to a base amount.",
                {"base": P("number", "Base amount."),
                 "rates": P("array", "Per-period rates as decimals.", items={"type": "number"})}, ["base", "rates"])
    ptxt = ", ".join(pct(r) for r in rates)
    q = rng.choice([
        f"Starting from {base}, apply annual rates of {ptxt} in order and compute the cumulative growth.",
        f"With a base of {base} and successive rates {ptxt}, compute the cumulative growth.",
    ])
    return tool, q, {"base": base, "rates": rates}

def g_prob():
    n = rng.randint(5, 200); k = rng.randint(0, n); p = round(rng.uniform(0.05, 0.95), 2)
    tool = TOOL("binomial_chance", "Probability of exactly k successes in n trials with per-trial probability p (decimal).",
                {"n": P("integer", "Number of trials."), "k": P("integer", "Number of successes."),
                 "p": P("number", "Per-trial success probability as a decimal.")}, ["n", "k", "p"])
    q = rng.choice([
        f"Across {n} trials with a {pct(p)} success rate each, what's the probability of exactly {k} successes?",
        f"Compute the probability of {k} successes out of {n} attempts when each attempt succeeds {pct(p)} of the time.",
    ])
    return tool, q, {"n": n, "k": k, "p": p}

def g_booking():
    rtype = rng.choice(["standard", "deluxe", "suite", "king", "double"])
    price = rng.choice([80, 120, 250, 999, 1500, 65, 340])
    cid = rng.randint(100, 999999)
    tool = TOOL("reserve_room", "Reserve a hotel room.",
                {"room_type": P("string", "Type of room."),
                 "nightly_price": P("number", "Nightly price."),
                 "nights": P("integer", "Number of nights."),
                 "customer_id": P("integer", "Numeric customer identifier.")},
                ["room_type", "nightly_price", "nights", "customer_id"])
    nights = rng.randint(1, 14)
    q = rng.choice([
        f"Reserve a {rtype} room at ${price} per night for {nights} nights for customer {cid}.",
        f"Book a {rtype} room for customer {cid}: {nights} nights at a nightly price of {price}.",
    ])
    return tool, q, {"room_type": rtype, "nightly_price": price, "nights": nights, "customer_id": cid}

def g_list_stat():
    k = rng.randint(3, 8); vals = [rng.randint(1, 100) for _ in range(k)]
    stat = rng.choice(["mean", "median", "max", "min", "sum", "stdev"])
    tool = TOOL("aggregate", "Aggregate a list of numbers with a chosen statistic.",
                {"data": P("array", "The numbers.", items={"type": "number"}),
                 "metric": P("string", "Statistic to compute.", enum=["mean", "median", "max", "min", "sum", "stdev"])},
                ["data", "metric"])
    lt = ", ".join(map(str, vals))
    q = rng.choice([f"For the numbers {lt}, compute the {stat}.", f"What is the {stat} of {lt}?"])
    return tool, q, {"data": vals, "metric": stat}

def g_sort():
    k = rng.randint(4, 9); vals = [rng.randint(1, 99) for _ in range(k)]
    desc = rng.random() < 0.5
    tool = TOOL("order_list", "Order a list of numbers ascending or descending.",
                {"items": P("array", "Numbers to order.", items={"type": "number"}),
                 "descending": P("boolean", "Sort descending if true.", default=False)}, ["items", "descending"])
    lt = ", ".join(map(str, vals))
    ph = rng.choice(["in descending order", "largest first", "high to low"]) if desc else rng.choice(["in ascending order", "smallest first", "low to high"])
    q = rng.choice([f"Order {lt} {ph}.", f"Sort these values {ph}: {lt}."])
    return tool, q, {"items": vals, "descending": desc}

GENS = [g_order, g_order, g_order_xy, g_pct_single, g_pct_single, g_pct_list, g_pct_list,
        g_prob, g_booking, g_list_stat, g_sort]

N = 9000
seen = set(); out = []; att = 0
while len(out) < N and att < N * 6:
    att += 1
    tool, q, args = rng.choice(GENS)()
    key = (tool["function"]["name"], q)
    if key in seen: continue
    seen.add(key)
    tools = [tool]
    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 2)):
            dt, _, _ = rng.choice(GENS)()
            if dt["function"]["name"] != tool["function"]["name"]:
                tools.append(dt)
        rng.shuffle(tools)
    out.append({"query": q, "tools": tools, "name": tool["function"]["name"], "arguments": args})

with open("synth2.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
with open("synth2_queries.jsonl", "w") as f:
    for r in out: f.write(json.dumps({"text": r["query"]}) + "\n")
print("wrote", len(out))
