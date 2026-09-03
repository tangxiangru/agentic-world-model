"""Read-only replay of this audit's three GSM8K Inspect metadata extractions.

Usage: python extract_developer_metadata.py ORIGINAL_LOG.json [...]
Emits JSONL to stdout; reads full original logs, never executes a model.
The intentionally narrow one-model-call/match-scorer assumptions fail loudly.
"""

import hashlib
import json
import sys


def digest(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def extract(path):
    with open(path, "rb") as stream:
        raw = stream.read()
    doc = json.loads(raw)
    samples = doc["samples"]
    inputs, requests, scores, configs = [], [], [], []
    for sample in samples:
        identity = {"id": sample["id"], "epoch": sample.get("epoch")}
        inputs.append({**identity, "input": sample["input"], "target": sample["target"]})
        calls = [event for event in sample.get("events", []) if event.get("event") == "model"]
        if len(calls) != 1:
            raise ValueError(f"{path}: expected one model call per sample")
        event = calls[0]
        requests.append({**identity, "messages": [
            {"role": message["role"], "content": message["content"]} for message in event["input"]
        ]})
        scores.append(sample["scores"]["match"]["value"])
        if event["config"] not in configs:
            configs.append(event["config"])
    result = doc["results"]
    score = result["scores"][0]
    return {
        "source": path, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw), "status": doc["status"], "actual_n": len(samples),
        "total_samples": result["total_samples"], "completed_samples": result["completed_samples"],
        "scored_samples": score["scored_samples"], "unscored_samples": score["unscored_samples"],
        "score_values": {key: scores.count(key) for key in set(scores)},
        "accuracy": score["metrics"]["accuracy"]["value"], "model": doc["eval"]["model"],
        "model_args": doc["eval"]["model_args"],
        "generation_config": doc["eval"]["model_generate_config"],
        "packages": doc["eval"]["packages"], "model_event_configs": configs,
        "ordered_input_target_sha256": digest(inputs),
        "ordered_request_role_content_sha256": digest(requests),
    }


if __name__ == "__main__":
    for source in sys.argv[1:]:
        print(json.dumps(extract(source), ensure_ascii=False))
