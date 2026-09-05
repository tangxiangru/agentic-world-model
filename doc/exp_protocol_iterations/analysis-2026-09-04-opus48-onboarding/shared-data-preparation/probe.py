"""Read approved exact HumanEval cache path; metadata output only, no execution."""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/opt/ptb-humaneval")
import task as profile

rows = profile.load_rows()
native = profile.upstream_module()
samples = [native.record_to_sample()(row) for row in rows]
contract = profile.selection_contract(samples)
assert profile.canonical_hash(contract) == profile.FULL_SELECTION_SHA256
record = {"image_label": sys.argv[1], "rows": len(rows), "epochs": 1,
          "dataset_revision": profile.REVISION, "parquet_sha256": profile.SOURCE_SHA256,
          "full_selection_sha256": profile.canonical_hash(contract),
          "cache_path": str(profile.dataset_path()), "network_fetch": False,
          "model_called": False, "dataset_code_executed": False}
with (Path("/home/ben") / (sys.argv[1] + "-shared-data.json")).open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps(record, indent=2))
