"""Private grade-availability metadata for a pinned historical comparison.

This is a separate preparation phase, not a WM feature/label loader. It decodes
only archive identities and the final-label object, never plans, code or prior
observations. It emits a validity boolean, never any score, rank, gap or reason
that depends on a valid score's magnitude. Zero is a valid official score.
Freeze models/settings independently of this metadata; later scoring must join
the actual labels separately, after decisions have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.outcome_prediction.wm_parameter_pilot import _value_end


def project(line):
    """Skip all non-allowlisted top-level values without JSON-decoding them."""
    decoder = json.JSONDecoder()
    if not line.lstrip().startswith("{"):
        raise ValueError("Inventory record must be an object")
    cursor, seen, result = line.index("{") + 1, set(), {}
    while cursor < len(line):
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        if cursor >= len(line):
            break
        if line[cursor] == "}":
            if line[cursor + 1 :].strip():
                raise ValueError("Trailing inventory content")
            if set(result) != {"example_id", "cell_id", "label"}:
                raise ValueError("Missing archive identity or label field")
            return result
        key, cursor = decoder.raw_decode(line, cursor)
        if not isinstance(key, str) or key in seen:
            raise ValueError("Invalid or duplicate top-level field")
        seen.add(key)
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        if cursor >= len(line) or line[cursor] != ":":
            raise ValueError("Malformed inventory field")
        cursor += 1
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
        end = _value_end(line, cursor)
        if key in {"example_id", "cell_id", "label"}:
            result[key] = json.loads(line[cursor:end])
        cursor = end
        if cursor < len(line) and line[cursor] == ",":
            cursor += 1
            if line[cursor:].lstrip().startswith("}"):
                raise ValueError("Trailing inventory comma")
    raise ValueError("Unterminated inventory object")


def valid_official_label(label):
    """Match final TRAIN-label validity; never interpret absence as zero."""
    if not isinstance(label, dict):
        return False
    accuracy, official = label.get("accuracy"), label.get("official_metric")
    return (
        type(accuracy) in (int, float)
        and math.isfinite(accuracy)
        and 0 <= accuracy <= 1
        and isinstance(official, dict)
        and type(official.get("accuracy")) in (int, float)
        and official["accuracy"] == accuracy
    )


def summarize(lines, partition):
    """Only identities, partitions and magnitude-invariant availability escape."""
    records, seen = [], set()
    for line in lines:
        if not line.strip():
            continue
        row = project(line)
        cell, key = row["cell_id"], row["example_id"]
        if (
            not isinstance(cell, str)
            or not isinstance(key, str)
            or cell not in partition
            or partition[cell] not in {"train", "test"}
            or not key.startswith(cell + "/")
            or key in seen
        ):
            raise ValueError("Unknown, mismatched or duplicate archive identity")
        seen.add(key)
        records.append(
            {
                "example_id": key,
                "cell_id": cell,
                "partition": partition[cell],
                "has_valid_official_final_grade": valid_official_label(row["label"]),
            }
        )
    return sorted(records, key=lambda row: row["example_id"])


def build(inventory, split, *, inventory_sha256, split_sha256, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new private metadata file")
    inventory_raw, split_raw = Path(inventory).read_bytes(), Path(split).read_bytes()
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    if sha(inventory_raw) != inventory_sha256 or sha(split_raw) != split_sha256:
        raise ValueError("Pinned inventory or split changed")
    result = {
        "schema": "wm-final-grade-availability-v1",
        "inventory_sha256": inventory_sha256,
        "split_sha256": split_sha256,
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "scope": "grade_availability_only_not_model_features_or_evaluation_results",
        "score_values_emitted": False,
        "valid_score_magnitude_affects_output": False,
        "label_objects_decoded_privately": True,
        "records": summarize(
            inventory_raw.decode().splitlines(), json.loads(split_raw)["cell_partition"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    output.chmod(0o600)
    return {"records": len(result["records"]), "score_values_emitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "split", "inventory-sha256", "split-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args))))


if __name__ == "__main__":
    main()
