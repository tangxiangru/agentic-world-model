"""Per-choice-set comparison: agentic RPM vs frozen judge tournament vs prior-run rankers."""
import glob, json, random, statistics
from collections import defaultdict
from pathlib import Path

A = Path("data/analysis/rpm/agentic"); J = Path("data/analysis/rpm/judge"); JC = Path("data/analysis/rpm/judge_code")
ex = {json.loads(l)["example_id"]: json.loads(l) for l in open("data/analysis/outcome_prediction/examples.jsonl")}
labels = {l["id"]: l for l in json.load(open(J / "hidden_labels.json"))}
hidden = {h["set_id"]: h for h in json.load(open(A / "hidden_labels.json"))}

def judge(d):
    return {json.load(open(p))["id"]: json.load(open(p))["p_a"] for p in glob.glob(str(d / "*.json")) if json.load(open(p)).get("valid")}
def learned(name, method):
    out = {}
    for l in open(f"data/analysis/rpm/{name}/pair_predictions.jsonl"):
        d = json.loads(l); out[(d["a_id"], d["b_id"])] = d["probabilities"][method]
    return {pid: out[(l["a_id"], l["b_id"])] for pid, l in labels.items() if (l["a_id"], l["b_id"]) in out}
pairwise = {
    "frozen judge (+setup+code), tournament": judge(JC / "outputs/code_within_run"),
    "ours: fixed logistic C=1": learned("learned_siblings", "fixed_recipe_numeric_logistic_C1"),
    "ours: recipe forest": learned("learned_siblings", "exploratory_recipe_numeric_forest"),
    "ours: kNN k=5": json.load(open(JC / "knn5.json")),
}
def prob(pa, x, y):
    for pid, l in labels.items():
        if l["a_id"] == x and l["b_id"] == y: return pa.get(pid, 0.5)
        if l["a_id"] == y and l["b_id"] == x: return 1 - pa.get(pid, 0.5)
    return 0.5
def roundrobin(pa, cands):
    score = {c: sum(prob(pa, c, o) for o in cands if o != c) for c in cands}
    best = max(score.values()); return statistics.mean(ex[c]["y"] for c in cands if score[c] == best)

agent = {}
for p in A.glob("outputs/set-*.json"):
    o = json.load(open(p))
    if o["valid"]:
        agent[o["set_id"]] = hidden[o["set_id"]]["y"][o["decision"]["choice"]]
sets = [s for s in hidden if s in agent]
picked = defaultdict(dict)
for s in sets:
    h = hidden[s]; cands = list(h["mapping"].values()); ys = list(h["y"].values())
    picked["oracle"][s] = max(ys); picked["random candidate"][s] = statistics.mean(ys)
    picked["scientist's last sibling"][s] = ex[max(cands, key=lambda c: ex[c]["first_submitted_at"])]["y"]
    picked["agentic RPM (tools + same bank)"][s] = agent[s]
    for name, pa in pairwise.items():
        picked[name][s] = roundrobin(pa, cands)

rng = random.Random(11); boots = [[rng.choice(sets) for _ in sets] for _ in range(4000)]
def mean_over(d, S): return statistics.mean(d[s] for s in S)
print(f"{len(sets)} choice sets with a valid agent decision ({sum(len(hidden[s]['y']) for s in sets)} candidates)\n")
print(f"{'selector':42s} picked acc  adv. over mean cand (pp)  regret vs oracle (pp)")
for name, d in picked.items():
    adv = 100 * (mean_over(d, sets) - mean_over(picked["random candidate"], sets)); reg = 100 * (mean_over(picked["oracle"], sets) - mean_over(d, sets))
    print(f"{name:42s} {mean_over(d, sets):.3f}       {adv:+6.2f}                  {reg:5.2f}")
ref = "agentic RPM (tools + same bank)"
print(f"\nhead-to-head vs {ref} (per set; ties = same expected accuracy):")
for name in [k for k in picked if k.startswith("ours") or k.startswith("frozen") or k.startswith("scientist")]:
    w = sum(picked[name][s] > picked[ref][s] + 1e-9 for s in sets); t = sum(abs(picked[name][s] - picked[ref][s]) <= 1e-9 for s in sets)
    diffs = sorted(100 * (mean_over(picked[name], b) - mean_over(picked[ref], b)) for b in boots)
    print(f"  {name:42s} wins {w:2d} ties {t:2d} losses {len(sets)-w-t:2d}   mean Δ {100*(mean_over(picked[name],sets)-mean_over(picked[ref],sets)):+.2f}pp  95% CI [{diffs[100]:+.2f}, {diffs[-100]:+.2f}]")
json.dump({k: v for k, v in picked.items()}, open(A / "comparison_picks.json", "w"), indent=1)
