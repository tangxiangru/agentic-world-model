import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "partial.json").write_text(json.dumps({"n": 2, "sum": 3, "complete": False}) + "\n")
print("first two records retained", flush=True)
print("fixture input rejected before third record", file=sys.stderr, flush=True)
raise SystemExit(23)
