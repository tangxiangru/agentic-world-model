"""Summarise an inspect-ai gsm8k log: accuracy, format compliance, stop reasons."""
import argparse
import json
import re
import sys
from pathlib import Path

ANSWER_LINE = re.compile(r"^ANSWER:\s*\$?-?[\d,]+(\.\d+)?\.?$", re.M)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to the inspect json log; default = newest in logs/")
    ap.add_argument("--dump-failures", default=None)
    ap.add_argument("--n-show", type=int, default=0)
    args = ap.parse_args()

    if args.log:
        path = Path(args.log)
    else:
        cands = sorted(Path("logs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        cands = [c for c in cands if c.name[0].isdigit()]
        if not cands:
            sys.exit("no inspect logs in logs/")
        path = cands[-1]
    log = json.loads(path.read_text())

    samples = log.get("samples") or []
    metrics = {}
    try:
        for s in log["results"]["scores"][0]["metrics"].values():
            metrics[s["name"]] = s["value"]
    except Exception:
        pass

    n = len(samples)
    correct = fmt_ok = truncated = 0
    fails = []
    for s in samples:
        out = ""
        try:
            out = s["output"]["choices"][0]["message"]["content"]
            if isinstance(out, list):
                out = "".join(c.get("text", "") for c in out)
        except Exception:
            pass
        stop = ""
        try:
            stop = s["output"]["choices"][0]["stop_reason"] or ""
        except Exception:
            pass
        ok = (s.get("scores", {}).get("match", {}) or {}).get("value") == "C"
        correct += ok
        last = [ln for ln in out.strip().splitlines() if ln.strip()]
        if last and ANSWER_LINE.match(last[-1].strip()):
            fmt_ok += 1
        if stop in ("max_tokens", "length"):
            truncated += 1
        if not ok:
            fails.append({"id": s.get("id"), "target": s.get("target"),
                          "stop": stop, "n_chars": len(out), "output_tail": out[-400:]})

    print(f"log            : {path}")
    print(f"n              : {n}")
    print(f"accuracy       : {correct/n:.4f} ({correct}/{n})   [log metrics: {metrics}]")
    print(f"format ok      : {fmt_ok/n:.4f}  (last line is 'ANSWER: <num>')")
    print(f"hit token cap  : {truncated/n:.4f}")
    if args.dump_failures:
        Path(args.dump_failures).write_text("\n".join(json.dumps(f) for f in fails))
        print(f"failures -> {args.dump_failures}")
    for f in fails[: args.n_show]:
        print("-" * 70)
        print(f["id"], "gold=", f["target"], "stop=", f["stop"], "chars=", f["n_chars"])
        print(f["output_tail"])


if __name__ == "__main__":
    main()
