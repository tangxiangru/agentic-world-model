"""
Generate GENERIC single-function-call training examples that exercise general
skills the model is weak at: (a) percentages in prose -> decimal args,
(b) numeric list/array arguments (preserve all elements & order),
(c) optional boolean flags stated in prose, (d) positional ordering of numeric args.

These are generic, domain-matching examples (allowed by the rules). They do NOT
copy or paraphrase any specific benchmark test item; functions, names, phrasings
and values are independently designed and randomized. Output is decontaminated
against the test set separately.
"""
import json, random

rng = random.Random(1234)

def num(lo, hi, dec=0):
    if dec == 0:
        return rng.randint(int(lo), int(hi))
    v = round(rng.uniform(lo, hi), dec)
    return v

def pct_phrase(dec_val):
    # dec_val is a decimal like 0.07 ; express as percent in prose
    p = round(dec_val * 100, 2)
    if p == int(p):
        p = int(p)
    return f"{p}%"

TOOL = lambda name, desc, props, req: {
    "type": "function",
    "function": {"name": name, "description": desc,
                 "parameters": {"type": "object", "properties": props, "required": req,
                                "additionalProperties": False}}}

def P(t, d, **extra):
    o = {"type": t, "description": d}
    o.update(extra)
    return o

examples = []

# ---------- (a) percentage -> decimal, single/multi numeric ----------
def gen_probability():
    n = rng.randint(5, 200)
    k = rng.randint(0, n)
    p_dec = round(rng.uniform(0.05, 0.95), 2)
    tool = TOOL("probability_success_runs",
                "Compute the probability of observing exactly k successes across n independent attempts given a per-attempt success probability.",
                {"attempts": P("integer", "Total number of independent attempts."),
                 "successes": P("integer", "Desired number of successful outcomes."),
                 "success_probability": P("number", "Probability of success on a single attempt, as a decimal between 0 and 1.")},
                ["attempts", "successes", "success_probability"])
    templates = [
        f"An experiment runs {n} independent attempts, each succeeding {pct_phrase(p_dec)} of the time. What is the probability of getting exactly {k} successes?",
        f"If each trial has a {pct_phrase(p_dec)} chance of working and I run {n} trials, how likely is it that exactly {k} of them succeed?",
        f"Estimate the chance of exactly {k} wins out of {n} tries when the per-try win rate is {pct_phrase(p_dec)}.",
    ]
    q = rng.choice(templates)
    return tool, q, {"attempts": n, "successes": k, "success_probability": p_dec}

def gen_interest():
    principal = rng.choice([1000, 2500, 5000, 10000, 25000, 100000, 1500, 750, 12000])
    rate = round(rng.uniform(0.01, 0.15), 3)
    years = rng.randint(1, 40)
    tool = TOOL("compound_growth",
                "Calculate the future value of a principal amount compounded annually at a given yearly rate.",
                {"principal": P("number", "The initial amount of money."),
                 "annual_rate": P("number", "The annual growth rate expressed as a decimal."),
                 "years": P("integer", "The number of years the amount is invested.")},
                ["principal", "annual_rate", "years"])
    templates = [
        f"I invest ${principal} at an annual return of {pct_phrase(rate)} for {years} years. What will it grow to?",
        f"How much will ${principal} become after {years} years if it earns {pct_phrase(rate)} per year, compounded annually?",
        f"Project the value of a ${principal} deposit growing at {pct_phrase(rate)} annually over {years} years.",
    ]
    return tool, rng.choice(templates), {"principal": principal, "annual_rate": rate, "years": years}

def gen_discount():
    price = rng.choice([19.99, 49.5, 120, 300, 89.9, 1500, 42, 7.5])
    disc = round(rng.uniform(0.05, 0.75), 2)
    tool = TOOL("apply_discount",
                "Compute the final price after applying a discount rate to an original price.",
                {"original_price": P("number", "The pre-discount price."),
                 "discount_rate": P("number", "Discount fraction as a decimal (e.g. 0.2 for 20%).")},
                ["original_price", "discount_rate"])
    templates = [
        f"An item costs ${price} and is on sale at {pct_phrase(disc)} off. What is the final price?",
        f"Apply a {pct_phrase(disc)} discount to a ${price} product.",
        f"If a {pct_phrase(disc)} markdown is applied to ${price}, what do I pay?",
    ]
    return tool, rng.choice(templates), {"original_price": price, "discount_rate": disc}

# ---------- (b) list / array arguments ----------
def gen_stats_list():
    k = rng.randint(3, 8)
    vals = [num(1, 100) for _ in range(k)]
    stat = rng.choice(["mean", "median", "variance", "standard deviation", "maximum", "minimum"])
    tool = TOOL("summarize_numbers",
                "Compute a summary statistic over a list of numbers.",
                {"numbers": P("array", "The list of numbers to summarize.", items={"type": "number"}),
                 "statistic": P("string", "Which statistic to compute.",
                                enum=["mean", "median", "variance", "stdev", "max", "min"])},
                ["numbers", "statistic"])
    stat_map = {"mean": "mean", "median": "median", "variance": "variance",
                "standard deviation": "stdev", "maximum": "max", "minimum": "min"}
    listtxt = ", ".join(str(v) for v in vals)
    templates = [
        f"Given the numbers {listtxt}, compute the {stat}.",
        f"What is the {stat} of the dataset {listtxt}?",
        f"I have these measurements: {listtxt}. Find the {stat}.",
    ]
    return tool, rng.choice(templates), {"numbers": vals, "statistic": stat_map[stat]}

def gen_dot():
    k = rng.randint(2, 5)
    a = [num(-9, 9) for _ in range(k)]
    b = [num(-9, 9) for _ in range(k)]
    tool = TOOL("vector_dot_product",
                "Compute the dot product of two equal-length numeric vectors.",
                {"vector_one": P("array", "The first vector.", items={"type": "number"}),
                 "vector_two": P("array", "The second vector.", items={"type": "number"})},
                ["vector_one", "vector_two"])
    templates = [
        f"Compute the dot product of the vectors {a} and {b}.",
        f"I have two vectors, {a} and {b}. What's their dot product?",
        f"Find the inner product between {a} and {b}.",
    ]
    return tool, rng.choice(templates), {"vector_one": a, "vector_two": b}

def gen_weighted_list():
    k = rng.randint(3, 5)
    scores = [num(0, 100) for _ in range(k)]
    weights = [round(rng.uniform(0.05, 0.9), 2) for _ in range(k)]
    tool = TOOL("weighted_average",
                "Compute a weighted average from parallel lists of values and their weights.",
                {"values": P("array", "The list of values.", items={"type": "number"}),
                 "weights": P("array", "The list of weights, aligned with values.", items={"type": "number"})},
                ["values", "weights"])
    vt = ", ".join(str(v) for v in scores)
    wt = ", ".join(str(w) for w in weights)
    templates = [
        f"Compute the weighted average of the values {vt} with weights {wt} respectively.",
        f"I have values {vt} and corresponding weights {wt}. What's the weighted mean?",
    ]
    return tool, rng.choice(templates), {"values": scores, "weights": weights}

def gen_pct_list():
    k = rng.randint(3, 6)
    rates = [round(rng.uniform(0.01, 0.09), 2) for _ in range(k)]
    base = rng.choice([1000, 5000, 20000, 100000])
    tool = TOOL("project_adjusted_value",
                "Project a value over several periods applying a per-period rate provided as a list of decimals.",
                {"base_amount": P("number", "The starting amount."),
                 "period_rates": P("array", "Per-period rates as decimals.", items={"type": "number"})},
                ["base_amount", "period_rates"])
    ptxt = ", ".join(pct_phrase(r) for r in rates)
    templates = [
        f"Starting from ${base}, apply these successive yearly rates: {ptxt}. Project the adjusted value.",
        f"With a base of ${base} and per-year rates of {ptxt}, compute the projected value.",
    ]
    return tool, rng.choice(templates), {"base_amount": base, "period_rates": rates}

# ---------- (c) optional boolean flags ----------
def gen_sort():
    k = rng.randint(4, 8)
    vals = [num(1, 99) for _ in range(k)]
    descending = rng.random() < 0.5
    tool = TOOL("sort_values",
                "Sort a list of numbers.",
                {"values": P("array", "Numbers to sort.", items={"type": "number"}),
                 "descending": P("boolean", "Whether to sort in descending order.", default=False)},
                ["values", "descending"])
    listtxt = ", ".join(str(v) for v in vals)
    if descending:
        order = rng.choice(["in descending order", "from largest to smallest", "high to low"])
    else:
        order = rng.choice(["in ascending order", "from smallest to largest", "low to high"])
    templates = [
        f"Sort the numbers {listtxt} {order}.",
        f"Please arrange {listtxt} {order}.",
    ]
    return tool, rng.choice(templates), {"values": vals, "descending": descending}

def gen_search():
    words = ["apple", "delta", "kernel", "orbit", "maple", "quartz", "harbor", "vertex"]
    q = rng.choice(words)
    case_sensitive = rng.random() < 0.5
    tool = TOOL("text_search",
                "Search for a term in a document.",
                {"term": P("string", "The search term."),
                 "case_sensitive": P("boolean", "Whether the search is case sensitive.", default=False)},
                ["term", "case_sensitive"])
    if case_sensitive:
        phr = rng.choice(["with case sensitivity on", "matching case exactly", "case-sensitively"])
    else:
        phr = rng.choice(["ignoring case", "case-insensitively", "regardless of capitalization"])
    templates = [
        f"Search for the term '{q}' {phr}.",
        f"Find '{q}' in the document, {phr}.",
    ]
    return tool, rng.choice(templates), {"term": q, "case_sensitive": case_sensitive}

# ---------- (d) positional ordering of numeric args ----------
def gen_two_num():
    a = num(2, 99); b = num(2, 99)
    op = rng.choice([
        ("greatest_common_divisor", "Compute the greatest common divisor of two integers.", "first_number", "second_number"),
        ("power_modulo", "Compute the remainder when the first number is raised to the second.", "base", "exponent"),
        ("subtract_values", "Subtract the second number from the first.", "minuend", "subtrahend"),
    ])
    name, desc, pa, pb = op
    tool = TOOL(name, desc,
                {pa: P("integer", f"The value for {pa}."),
                 pb: P("integer", f"The value for {pb}.")},
                [pa, pb])
    templates = [
        f"Using {a} as the {pa.replace('_',' ')} and {b} as the {pb.replace('_',' ')}, run {name.replace('_',' ')}.",
        f"Compute {name.replace('_',' ')} where {pa.replace('_',' ')} is {a} and {pb.replace('_',' ')} is {b}.",
    ]
    return tool, rng.choice(templates), {pa: a, pb: b}

def gen_physics():
    v0 = num(0, 50); acc = round(rng.uniform(0.5, 12), 1); t = num(1, 30)
    tool = TOOL("final_speed",
                "Compute the final speed given an initial speed, acceleration and elapsed time.",
                {"initial_speed": P("number", "Initial speed in m/s."),
                 "acceleration": P("number", "Acceleration in m/s^2."),
                 "elapsed_time": P("number", "Elapsed time in seconds.")},
                ["initial_speed", "acceleration", "elapsed_time"])
    templates = [
        f"An object starts at {v0} m/s and accelerates at {acc} m/s^2 for {t} seconds. What is its final speed?",
        f"Compute the final speed for an initial speed of {v0} m/s, acceleration {acc} m/s^2, over {t} s.",
    ]
    return tool, rng.choice(templates), {"initial_speed": v0, "acceleration": acc, "elapsed_time": t}

GENS = [gen_probability, gen_interest, gen_discount, gen_stats_list, gen_dot,
        gen_weighted_list, gen_pct_list, gen_sort, gen_search, gen_two_num, gen_physics]

N = 5500
# sometimes add 1-2 distractor tools so the model learns selection too
DISTRACTORS = None

seen = set()
out = []
attempts = 0
while len(out) < N and attempts < N * 5:
    attempts += 1
    g = rng.choice(GENS)
    tool, q, args = g()
    key = (tool["function"]["name"], q)
    if key in seen:
        continue
    seen.add(key)
    tools = [tool]
    # occasionally add a distractor tool from another generator
    if rng.random() < 0.4:
        for _ in range(rng.randint(1, 2)):
            dt, _, _ = rng.choice(GENS)()
            if dt["function"]["name"] != tool["function"]["name"]:
                tools.append(dt)
        rng.shuffle(tools)
    out.append({"query": q, "tools": tools, "name": tool["function"]["name"], "arguments": args})

with open("synth.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
with open("synth_queries.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps({"text": r["query"]}) + "\n")
print("wrote", len(out), "synthetic examples")
