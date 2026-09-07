"""Choice-set selection: the scientist produced N same-parent siblings; each ranker picks one.

Pairs with |gap| < 1pt were never judged/scored and count as ties for every method.
Knockout follows the paper: candidates in random order, winner stays; averaged over orders.
"""
import json, glob, random, statistics
from collections import defaultdict
from pathlib import Path

J = Path("data/analysis/rpm/judge"); JC = Path("data/analysis/rpm/judge_code")
labels = {l["id"]: l for l in json.load(open(J / "hidden_labels.json"))}
ex = {json.loads(l)["example_id"]: json.loads(l) for l in open("data/analysis/outcome_prediction/examples.jsonl")}

def judge(d):
    return {json.load(open(p))["id"]: json.load(open(p))["p_a"] for p in glob.glob(str(d / "*.json")) if json.load(open(p)).get("valid")}
def learned(name, method):
    out = {}
    for l in open(f"data/analysis/rpm/{name}/pair_predictions.jsonl"):
        d = json.loads(l); out[(d["a_id"], d["b_id"])] = d["probabilities"][method]
    return {pid: out[(l["a_id"], l["b_id"])] for pid, l in labels.items() if (l["a_id"], l["b_id"]) in out}
methods = {
    "RPM analog: frozen judge + setup + code": judge(JC / "outputs/code_within_run"),
    "RPM analog: frozen judge, recipes only": judge(J / "outputs/within_run"),
    "ours: fixed logistic C=1": learned("learned_siblings", "fixed_recipe_numeric_logistic_C1"),
    "ours: recipe forest": learned("learned_siblings", "exploratory_recipe_numeric_forest"),
    "ours: kNN k=5": json.load(open(JC / "knn5.json")),
}
# P(x beats y) lookup; unjudged (gap<1pt) -> 0.5
def prob(pa, x, y):
    for pid, l in labels.items():
        if l["a_id"] == x and l["b_id"] == y: return pa.get(pid, 0.5)
        if l["a_id"] == y and l["b_id"] == x: return 1 - pa.get(pid, 0.5)
    return 0.5
# choice sets: same cell, same parent set, labeled, non-merge, complete lineage (same filter as pairs)
members = set()
for l in labels.values(): members.update([l["a_id"], l["b_id"]])
sets = defaultdict(list)
for eid in members:
    r = ex[eid]; sets[(r["cell_id"], tuple(sorted(r["parent_ids"])))].append(eid)
sets = {k: sorted(v) for k, v in sets.items() if len(v) >= 2}
print(f"choice sets: {len(sets)} across {len({k[0] for k in sets})} runs; sizes:", sorted(statistics.multimode([len(v) for v in sets.values()])), dict(sorted(((n, sum(1 for v in sets.values() if len(v)==n)) for n in set(map(len, sets.values()))))))

def pick_roundrobin(pa, cands):
    score = {c: sum(prob(pa, c, o) for o in cands if o != c) for c in cands}
    best = max(score.values()); tied = [c for c in cands if score[c] == best]
    return statistics.mean(ex[c]["y"] for c in tied)  # expected accuracy under random tie-break
def pick_knockout(pa, cands, orders):
    acc = []
    for order in orders:
        champ = order[0]
        for c in order[1:]:
            p = prob(pa, c, champ)
            champ = c if p > 0.5 else champ if p < 0.5 else random.Random(hash((c, champ)) & 0xffff).choice([c, champ])
        acc.append(ex[champ]["y"])
    return statistics.mean(acc)

rng = random.Random(7)
rows = defaultdict(list)
for key, cands in sets.items():
    ys = {c: ex[c]["y"] for c in cands}
    orders = [rng.sample(cands, len(cands)) for _ in range(200)]
    rows["oracle"].append(max(ys.values())); rows["random candidate"].append(statistics.mean(ys.values()))
    rows["last-registered sibling"].append(ys[max(cands, key=lambda c: ex[c]["first_submitted_at"])])
    for name, pa in methods.items():
        rows[name + " | round-robin"].append(pick_roundrobin(pa, cands))
        rows[name + " | knockout"].append(pick_knockout(pa, cands, orders))
print(f"\n{'selector':58s} mean picked acc   advantage over mean candidate (pp)   regret vs oracle (pp)")
for name, v in rows.items():
    adv = statistics.mean(a - b for a, b in zip(v, rows["random candidate"])) * 100
    reg = statistics.mean(a - b for a, b in zip(rows["oracle"], v)) * 100
    print(f"{name:58s} {statistics.mean(v):.3f}            {adv:+6.2f}                           {reg:5.2f}")

# head-to-head: our pick vs RPM pick, per choice set
print("\nhead-to-head per choice set (round-robin picks; ties = same expected accuracy):")
ref = "RPM analog: frozen judge + setup + code | round-robin"
for name in [k for k in rows if k.startswith("ours") and k.endswith("round-robin")] + ["last-registered sibling"]:
    w = sum(a > b + 1e-9 for a, b in zip(rows[name], rows[ref])); t = sum(abs(a - b) <= 1e-9 for a, b in zip(rows[name], rows[ref])); l = len(sets) - w - t
    print(f"  {name:45s} vs code-aware judge: wins {w:2d}  ties {t:2d}  losses {l:2d}   mean Δ {statistics.mean(a-b for a,b in zip(rows[name],rows[ref]))*100:+.2f}pp")
json.dump({k: v for k, v in rows.items()}, open(JC / "choice_sets.json", "w"), indent=1)
