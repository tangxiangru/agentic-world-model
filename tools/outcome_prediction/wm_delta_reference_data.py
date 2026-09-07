"""Add published fixed-base references to the immutable observed-parent cohort.

No models are fitted and no missing accuracy is imputed. Published scores are
shared reference constants, not measurements of individual trajectory parents.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from tools.outcome_prediction import wm_one_step_data as original
from tools.outcome_prediction.wm_grade_inventory import valid_official_label

SOURCE = Path("data/analysis/wm_one_step/01406da734fb_v1")
OUTPUT = Path("data/analysis/wm_delta_reference/01406da734fb_v1")
PUBLIC_URL = "https://raw.githubusercontent.com/aisa-group/PostTrainBench/main/scripts/baselines.json"
BASES = {
    "gsm8k": ("google/gemma-3-4b-pt", "gemma-3-4b-pt", 0.06141015921152388),
    "aime2025": ("Qwen/Qwen3-4B-Base", "Qwen3-4B-Base", 0.03333333333333333),
}
FILES = {
    "inputs.json", "labels.json", "split.json", "summary.json", "policy.json",
    "public_reference_source.json",
}
COHORTS = {"combined", "measured_parent", "published_base"}


def score_valid(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_public_source(source):
    """Verify the captured response and explicitly approved reference series."""
    if source.get("url") != PUBLIC_URL or not isinstance(source.get("body"), str):
        raise ValueError("Invalid published reference source")
    digest = hashlib.sha256(source["body"].encode("utf-8")).hexdigest()
    if source.get("sha256") != digest:
        raise ValueError("Changed published reference response")
    payload = json.loads(source["body"])
    for benchmark, (_, key, expected) in BASES.items():
        value = payload.get("zeroshot", {}).get(key, {}).get(benchmark)
        if not score_valid(value) or value != expected:
            raise ValueError("Published reference value differs from approved source")
    return payload


def _unique(rows):
    by_id = {r["example_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate example identity")
    return by_id


def build_bundle(targets, target_labels, measured, measured_labels, decisions, registry, split, public_source):
    """Pure derivation; incomplete labels are excluded, identities fail closed."""
    public = validate_public_source(public_source)
    target_by_id, measured_by_id = _unique(targets), _unique(measured)
    if not set(measured_by_id).issubset(target_by_id):
        raise ValueError("Measured cohort contains a non-target identity")
    examples, labels = [], {}
    for key, target in target_by_id.items():
        decision, entry = decisions[key], registry[key]
        label = target_labels.get(key)
        if not valid_official_label(label) or not decision.get("eligible"):
            continue
        if (
            entry.get("status") != "official_archive_correlated"
            or entry.get("accuracy") != label["accuracy"]
            or entry.get("cell_id") != target["cell_id"]
            or entry.get("benchmark") != target["benchmark"]
            or split.get(target["cell_id"]) not in {"train", "test"}
        ):
            raise ValueError("Target identity, score, or split mismatch")
        if key in measured_by_id:
            example = copy.deepcopy(measured_by_id[key])
            parent = example.get("parent") or {}
            measured_label = measured_labels.get(key)
            if not valid_official_label(measured_label) or not score_valid(parent.get("accuracy")):
                continue
            producer = registry.get(parent.get("producer_id"), {})
            if (
                not decision.get("one_step_eligible") or parent.get("reference_known") is not True
                or parent.get("producer_id") == key
                or producer.get("status") != "official_archive_correlated"
                or producer.get("accuracy") != parent["accuracy"]
                or producer.get("cell_id") != target["cell_id"]
                or producer.get("benchmark") != target["benchmark"]
                or measured_label["accuracy"] != label["accuracy"]
                or measured_label.get("delta_accuracy") != label["accuracy"] - parent["accuracy"]
                or any(example.get(k) != target.get(k) for k in ("example_id", "cell_id", "benchmark", "model_input"))
            ):
                raise ValueError("Invalid measured-parent reference")
            parent["kind"] = "measured"
            kind = "measured_parent"
        else:
            parsed = decision.get("checkpoint_inputs") or {}
            if parsed.get("status") != "base":
                continue
            inputs = parsed.get("inputs") or []
            benchmark = target["benchmark"]
            if benchmark not in BASES or len(inputs) != 1:
                raise ValueError("Unrecognized base input identity")
            model, public_key, _ = BASES[benchmark]
            if (
                inputs[0].get("base_model") != model or not inputs[0].get("path")
                or (entry.get("archived_setup") or {}).get("base_model") != model
            ):
                raise ValueError("Benchmark/base-model identity mismatch")
            example = copy.deepcopy(target)
            kind = "published_base"
            parent = {
                "kind": "published_base", "reference_known": True,
                "producer_id": "published_base:" + benchmark + ":" + model,
                "consumed_path": inputs[0]["path"], "base_model": model,
                "accuracy": public["zeroshot"][public_key][benchmark],
                "evaluation_n": None, "path_kind": "published_base_model",
                "official_grade_availability": "published_shared_reference_not_per_run_measurement",
                "source_url": PUBLIC_URL, "source_sha256": public_source["sha256"],
                "reference_key": ["zeroshot", public_key, benchmark],
            }
            example["parent"] = parent
            example["history"] = []
            example["history_complete_to_base"] = True
        example["reference_kind"] = kind
        labels[key] = {
            **copy.deepcopy(label), "reference_kind": kind,
            "delta_accuracy": label["accuracy"] - parent["accuracy"],
        }
        examples.append(example)
    summary = {}
    for kind in ("combined", "measured_parent", "published_base"):
        chosen = [e for e in examples if kind == "combined" or e["reference_kind"] == kind]
        summary[kind] = {
            "total": len(chosen),
            "train": sum(split[e["cell_id"]] == "train" for e in chosen),
            "test": sum(split[e["cell_id"]] == "test" for e in chosen),
            "by_benchmark": dict(Counter(e["benchmark"] for e in chosen)),
            "target_zero": sum(labels[e["example_id"]]["accuracy"] == 0 for e in chosen),
            "parent_zero": sum(e["parent"]["accuracy"] == 0 for e in chosen),
        }
    return {"inputs": examples, "labels": labels, "split": {"cell_partition": split}, "summary": summary}


def _from_original(source, public_source):
    cohorts = {}
    for cohort in ("target", "one_step"):
        rows, labels = [], {}
        for part in ("train", "test"):
            part_rows, part_labels = original.load_partition(source, part, cohort=cohort)
            rows.extend(part_rows)
            labels.update(part_labels)
        cohorts[cohort] = (rows, labels)
    return build_bundle(
        *cohorts["target"], *cohorts["one_step"],
        original.read(source / "private/decisions.json"),
        original.read(source / "private/registry.json"),
        original.read(source / "split.json")["cell_partition"], public_source,
    )


def policy():
    return {
        "version": 1, "no_missing_score_imputation": True,
        "valid_zero_scores_retained": True, "no_models_fitted": True,
        "split": "Exact unchanged source whole-session train/test assignments",
        "target": "Valid official target accuracy minus valid measured or published reference accuracy",
        "measured_parent": "Unchanged membership of the strict observed-parent cohort",
        "published_base": "Explicit published shared constants, not per-run measured parent accuracy",
        "selection": "Source structural eligibility and score availability only; never score magnitude",
        "limitations": [
            "Published baselines do not certify identical historical sampling or evaluation configuration.",
            "The upstream zeroshot series name does not certify literal zero-shot prompting.",
            "Published base evaluation_n is unspecified; no fictitious per-run measurement count is assigned.",
            "Observed checkpoint-parent grades are retrospective; deployment assumes the score is supplied.",
            "This preserves previously explored development splits, not a new confirmatory test.",
        ],
    }


def freeze(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing frozen dataset: " + str(output))
    with urlopen(PUBLIC_URL, timeout=30) as response:
        body = response.read().decode("utf-8")
    published = {
        "url": PUBLIC_URL, "body": body,
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle = _from_original(source, published)
    bundle.update(policy=policy(), public_reference_source=published)
    output.mkdir(parents=True)
    for key, value in bundle.items():
        original.write(output / (key + ".json"), value)
    dependencies = {str(source / "manifest.json"): original.sha(source / "manifest.json")}
    for name in ("wm_delta_reference_data.py", "wm_one_step_data.py", "wm_grade_inventory.py", "wm_clean_refresh.py"):
        path = Path(__file__).resolve().parent / name
        dependencies[str(path)] = original.sha(path)
    original.write(output / "manifest.json", {
        "version": 1, "source_directory": str(source), "sources": dependencies,
        "files": {name: original.sha(output / name) for name in sorted(FILES)},
    })
    load_partition(output, "train")
    load_partition(output, "test")
    return bundle["summary"]


def load_partition(directory, partition, *, cohort="combined"):
    """Verify dependencies and exact source-derived content before returning rows."""
    if partition not in {"train", "test"} or cohort not in COHORTS:
        raise ValueError("Invalid cohort or partition")
    directory = Path(directory)
    manifest = original.read(directory / "manifest.json")
    if set(manifest.get("files", {})) != FILES or not manifest.get("sources"):
        raise ValueError("Incomplete reference dataset manifest")
    source = Path(manifest["source_directory"])
    required = {str(source / "manifest.json")}
    required.update(str(Path(__file__).resolve().parent / name) for name in (
        "wm_delta_reference_data.py", "wm_one_step_data.py", "wm_grade_inventory.py", "wm_clean_refresh.py",
    ))
    if set(manifest["sources"]) != required:
        raise ValueError("Incomplete source dependencies")
    for path, digest in manifest["sources"].items():
        if original.sha(path) != digest:
            raise ValueError("Changed reference dataset dependency: " + path)
    for name, digest in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or original.sha(path) != digest:
            raise ValueError("Changed frozen reference dataset file: " + name)
    public_source = original.read(directory / "public_reference_source.json")
    expected = _from_original(source, public_source)
    for key, value in {**expected, "policy": policy()}.items():
        if original.read(directory / (key + ".json")) != value:
            raise ValueError("Reference dataset differs from verified source derivation: " + key)
    split = expected["split"]["cell_partition"]
    rows = [e for e in expected["inputs"] if split[e["cell_id"]] == partition
            and (cohort == "combined" or e["reference_kind"] == cohort)]
    return rows, {e["example_id"]: expected["labels"][e["example_id"]] for e in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
