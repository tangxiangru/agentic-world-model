import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
values = [json.loads(line)["value"] for line in args.data.read_text().splitlines()]
factor = int(os.environ["FACTOR"])
result = {"n": len(values), "sum": sum(values) * factor, "label": args.label,
          "factor": factor, "attempt": os.environ["AWM_EXP_ATTEMPT_ID"]}
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
print("aggregation finished", file=__import__("sys").stderr)
