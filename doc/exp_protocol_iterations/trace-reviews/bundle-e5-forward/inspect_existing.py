import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[1]).read_text())
assert reference["sum"] == 6 and reference["n"] == 3
print(json.dumps({"checked_existing": str(Path(sys.argv[1])), "valid": True}))
