"""Filter sampled solutions -> deduplicated correct-reasoning SFT rows (RFT)."""
import argparse
import json
import re
from collections import defaultdict

from common import extract_answer, norm_num

EQ_RE = re.compile(r"[-+*/=()\d.]{3,}")


def signature(text: str):
    """Coarse reasoning fingerprint: the ordered list of arithmetic expressions."""
    body = text.split("ANSWER:")[0]
    eqs = [e.strip(" .") for e in EQ_RE.findall(body)]
    eqs = [e for e in eqs if any(c in e for c in "+-*/=")]
    return tuple(eqs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-question", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=3000)
    a = p.parse_args()

    by_q = defaultdict(list)
    total = 0
    for path in a.inp:
        for line in open(path):
            r = json.loads(line)
            total += 1
            if not r["correct"]:
                continue
            text = r["completion"].strip()
            # must be a clean, self-contained solution ending in the answer line
            idx = text.rfind("ANSWER:")
            if idx == -1:
                continue
            text = text[:idx].rstrip() + "\n\nANSWER: " + str(
                norm_num(extract_answer(text)))
            if len(text) > a.max_chars or text.count("ANSWER:") != 1:
                continue
            if "<start_of_turn>" in text or "Reasoning:" in text:
                continue
            by_q[r["qidx"]].append((r["question"], r["answer"], text))

    rows, kept_sigs = [], 0
    for qidx, cands in by_q.items():
        seen, chosen = set(), []
        cands.sort(key=lambda c: len(c[2]))       # prefer concise correct solutions
        for q, ans, text in cands:
            s = signature(text)
            if s in seen:
                continue
            seen.add(s)
            chosen.append((q, ans, text))
            if len(chosen) >= a.per_question:
                break
        kept_sigs += len(chosen)
        for q, ans, text in chosen:
            rows.append({"question": q, "solution": text, "answer": ans,
                         "source": "rft"})

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"read {total} samples -> {len(by_q)} solved questions -> {len(rows)} rows")


if __name__ == "__main__":
    main()
