"""Build a pinned scripts + serving -> ten-run accuracy benchmark, without fitting.

Raw sources, audit metadata and labels never enter predictor input payloads.
The controlled matrix is the primary track. Native rescoring is inventoried
separately until its effective serving state is independently established.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "wm-scripts-serving-benchmark-v1"
REPO = "JerrrrryL/awm-gsm8k-trajectories"
N_QUESTIONS = {"gsm8k": 1319, "aime2025": 30}
QUARANTINE = {"r0-25-exp-02", "aime-r0-11-exp-02", "aime2-r0-12-exp-05"}
SALT = "wm-scripts-serving-v1-20260912"
SAMPLING_FIELDS = (
    "n", "temperature", "top_k", "top_p", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "max_tokens", "min_tokens",
    "stop", "stop_token_ids", "ignore_eos", "bad_words",
    "include_stop_str_in_output", "skip_special_tokens",
    "spaces_between_special_tokens", "truncate_prompt_tokens",
    "structured_outputs", "extra_args",
)
REQUIRED_SAMPLING = {
    "n", "temperature", "top_k", "top_p", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "max_tokens", "min_tokens",
    "stop", "stop_token_ids", "ignore_eos",
}
MODEL_CONFIG_FIELDS = {
    "_sliding_window_pattern", "architectures", "attention_bias", "attention_dropout",
    "attn_logit_softcapping", "boi_token_index", "bos_token_id", "dtype", "eoi_token_index",
    "eos_token_id", "final_logit_softcapping", "head_dim", "hidden_act", "hidden_activation",
    "hidden_size", "image_token_index", "initializer_range", "intermediate_size", "layer_types",
    "max_position_embeddings", "max_window_layers", "mm_tokens_per_image", "model_type",
    "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "pad_token_id",
    "quantization_config", "query_pre_attn_scalar", "rms_norm_eps", "rope_local_base_freq",
    "rope_scaling", "rope_theta", "sliding_window", "text_config", "tie_word_embeddings",
    "torch_dtype", "use_bidirectional_attention", "use_cache", "use_sliding_window",
    "vision_config", "vocab_size",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def is_rate(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_label(data, benchmark, expected_questions=None):
    """Recompute the target and check every question/run, independently of reports."""
    errors = []
    n = N_QUESTIONS.get(benchmark)
    rates = data.get("per_epoch_accuracy")
    matrix = data.get("per_problem")
    if data.get("epochs") != 10 or not isinstance(rates, list) or len(rates) != 10:
        errors.append("requires_exactly_ten_runs")
    if not isinstance(rates, list) or not all(is_rate(v) for v in rates):
        errors.append("invalid_run_rates")
    if n is None or data.get("n_problems") != n:
        errors.append("wrong_benchmark_question_count")
    if not isinstance(matrix, dict) or len(matrix) != n:
        errors.append("wrong_per_question_count")
    elif any(not isinstance(v, list) or len(v) != 10 or
             any(type(x) not in (int, float) or x not in (0, 1) for x in v)
             for v in matrix.values()):
        errors.append("incomplete_or_nonbinary_question_runs")
    if (expected_questions is not None and isinstance(matrix, dict)
            and set(matrix) != set(expected_questions)):
        errors.append("benchmark_question_ids_mismatch")
    if errors:
        return {"status": "invalid", "errors": errors, "y": None}
    actual = [sum(v[r] for v in matrix.values()) / n for r in range(10)]
    if any(abs(a - b) > 1e-12 for a, b in zip(actual, rates)):
        errors.append("run_rates_disagree_with_question_scores")
    mean = statistics.mean(actual)
    if not is_rate(data.get("accuracy")) or abs(data["accuracy"] - mean) > 1e-12:
        errors.append("aggregate_disagrees_with_question_scores")
    sd = statistics.pstdev(actual)
    if not is_rate(data.get("std_across_epochs")) or abs(data["std_across_epochs"] - sd) > 1e-12:
        errors.append("stored_run_sd_mismatch")
    if data.get("n_sample_errors", 0) != 0:
        errors.append("sample_errors_present")
    protocol = data.get("protocol")
    if isinstance(protocol, dict):
        if protocol.get("all_runs_complete") is not True or protocol.get("n_runs") != 10:
            errors.append("protocol_runs_incomplete")
        records = protocol.get("per_run", [])
        if (not isinstance(records, list) or len(records) != 10
                or any(not isinstance(r, dict) or type(r.get("repeat_id")) is not int for r in records)
                or {r.get("repeat_id") for r in records} != set(range(10))):
            errors.append("protocol_repeat_ids_invalid")
        else:
            for r in records:
                i = r["repeat_id"]
                if (r.get("complete") is not True or r.get("n_questions") != n or
                    r.get("n_scored") != n or r.get("n_correct") != round(actual[i] * n) or
                    not is_rate(r.get("pass_rate")) or abs(r["pass_rate"] - actual[i]) > 1e-12):
                    errors.append("protocol_repeat_inconsistent")
                    break
        if protocol.get("seeds_verified") is not True or protocol.get("seed_mismatches"):
            errors.append("protocol_seeds_unverified")
        if protocol.get("n_seed_ok") != 10 * n:
            errors.append("protocol_seed_coverage_mismatch")
    meta = data.get("eval_matrix_1k")
    if isinstance(meta, dict):
        complete = meta.get("completeness") or {}
        if (complete.get("inspect_status") != "success" or complete.get("epochs") != 10
                or complete.get("problems") != n or complete.get("n_sample_errors") != 0
                or complete.get("problems_found")):
            errors.append("matrix_completeness_invalid")
        if not is_rate(meta.get("avg_pass_rate")) or abs(meta["avg_pass_rate"] - mean) > 1e-12:
            errors.append("matrix_target_mismatch")
    return {
        "status": "complete" if not errors else "invalid", "errors": sorted(set(errors)),
        "y": mean if not errors else None, "metric": "mean_accuracy_over_ten_runs",
        "scale": "fraction_0_to_1", "n_questions": n, "n_runs": 10,
        "run_accuracies": actual, "run_sd": sd,
        "question_ids_sha256": digest(sorted(matrix)),
        "question_ids": sorted(matrix),
        "correctness": [matrix[q] for q in sorted(matrix)],
    }


def literal(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise ValueError("nonliteral_resolved_serving_field") from None


def extract_serving(data, source_root, checkpoint_id, benchmark):
    """Only the controlled matrix records complete server-resolved parameters."""
    meta = data.get("eval_matrix_1k")
    if not isinstance(meta, dict):
        path = Path(source_root) / "checkpoints_meta" / checkpoint_id / "generation_config.json"
        archived = read_json(path) if path.exists() else None
        return {"status": "candidate", "errors": ["effective_native_serving_unverified"],
                "config": None, "audit": {"archived_generation_config": archived,
                "request_policy": {k: v for k, v in (data.get("request_policy") or {}).items()
                                   if k in SAMPLING_FIELDS},
                "archived_config_available": archived is not None}}
    errors = []
    server = meta.get("server") or {}
    resolved = server.get("resolved_sampling_params") or {}
    if not REQUIRED_SAMPLING.issubset(resolved):
        errors.append("incomplete_resolved_serving_fields")
    try:
        sampling = {k: literal(resolved[k]) for k in SAMPLING_FIELDS if k in resolved}
    except ValueError:
        sampling = {}
        errors.append("nonliteral_resolved_serving_field")
    if sampling.get("top_k") == -1:
        sampling["top_k"] = 0
    if server.get("generation_config") != "vllm":
        errors.append("uncontrolled_generation_config_inheritance")
    if server.get("inherited_extra_stop_ids"):
        errors.append("unresolved_additional_stop_ids")
    required_numeric = ("temperature", "top_k", "top_p", "min_p", "max_tokens",
                        "min_tokens", "repetition_penalty", "presence_penalty", "frequency_penalty")
    if any(type(sampling.get(k)) not in (int, float) or not math.isfinite(sampling[k])
           for k in required_numeric):
        errors.append("invalid_numeric_serving_fields")
    if type(sampling.get("stop")) is not list or type(sampling.get("stop_token_ids")) is not list:
        errors.append("invalid_stop_settings")
    else:
        if any(not isinstance(s, str) for s in sampling["stop"]):
            errors.append("invalid_stop_strings")
        if any(type(i) is not int or i < 0 for i in sampling["stop_token_ids"]):
            errors.append("invalid_stop_token_ids")
    if type(sampling.get("ignore_eos")) is not bool:
        errors.append("invalid_ignore_eos")
    if sampling.get("n") != 1 or type(sampling.get("n")) is not int:
        errors.append("requires_one_answer_per_run_question")
    if not errors and (sampling["temperature"] < 0 or not 0 < sampling["top_p"] <= 1
            or not 0 <= sampling["min_p"] <= 1 or sampling["repetition_penalty"] <= 0
            or type(sampling["top_k"]) is not int or sampling["top_k"] < 0
            or type(sampling["max_tokens"]) is not int or sampling["max_tokens"] <= 0
            or type(sampling["min_tokens"]) is not int
            or not 0 <= sampling["min_tokens"] <= sampling["max_tokens"]):
        errors.append("invalid_sampling_parameter_range")
    family = (meta.get("preflight") or {}).get("family")
    template_name = {"gemma": "gemma3.jinja", "qwen": "qwen3.jinja"}.get(family)
    template = None
    if template_name:
        template_path = Path(source_root) / "rescore10/eval/templates" / template_name
        expected = (meta.get("digests") or {}).get("chat_template_sha256")
        if template_path.exists() and file_digest(template_path) == expected:
            template = template_path.read_text()
        else:
            errors.append("chat_template_missing_or_hash_mismatch")
    else:
        errors.append("unsupported_model_family")
    max_model_len = None
    log = server.get("cli_args_logged", "")
    if isinstance(log, str) and "{" in log:
        try:
            args = ast.literal_eval(log[log.index("{"):])
            max_model_len = args.get("max_model_len")
        except (ValueError, SyntaxError, AttributeError):
            pass
    if type(max_model_len) is not int or max_model_len <= 0:
        errors.append("missing_model_context_limit")
    request = data.get("request_policy") or {}
    for key in REQUIRED_SAMPLING - {"stop", "n"}:
        if key in request:
            a, b = sampling.get(key), request[key]
            if key == "top_k" and b == -1:
                b = 0
            # Greedy serving normalizes otherwise inactive sampling parameters.
            if sampling.get("temperature") == 0 and key in {"top_k", "top_p", "min_p"}:
                continue
            if a != b:
                errors.append("request_resolved_mismatch:" + key)
    protocol = data.get("protocol") or {}
    model_path = Path(source_root) / "checkpoints_meta" / checkpoint_id / "config.json"
    expected_model_hash = ((meta.get("preflight") or {}).get("file_sha256") or {}).get("config.json")
    model_config = None
    if model_path.exists() and expected_model_hash and file_digest(model_path) == expected_model_hash:
        raw_config = read_json(model_path)
        model_config = {k: v for k, v in raw_config.items() if k in MODEL_CONFIG_FIELDS}
        if isinstance(model_config.get("text_config"), dict):
            model_config["text_config"] = {k: v for k, v in model_config["text_config"].items()
                                            if k in MODEL_CONFIG_FIELDS}
    else:
        errors.append("serving_model_config_missing_or_hash_mismatch")
    config = {
        "benchmark": benchmark,
        "base_model": {"gemma": "google/gemma-3-4b-pt", "qwen": "Qwen/Qwen3-4B-Base"}.get(family),
        "sampling": sampling, "max_model_len": max_model_len,
        "generation_config_mode": server.get("generation_config"),
        "dtype": server.get("dtype"), "chat_template": template,
        "model_config": model_config,
        "runtime": {k: v for k, v in (meta.get("runtime") or {}).items()
                    if k in {"vllm", "inspect_ai", "transformers", "torch"}},
        "seed_policy": protocol.get("seed_formula"),
    }
    if not config["seed_policy"]:
        errors.append("missing_seed_policy")
    return {"status": "eligible" if not errors else "candidate", "errors": sorted(set(errors)),
            "config": config, "audit": {"server_resolved": True}}


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, value):
        self.parents.setdefault(value, value)
        if self.parents[value] != value:
            self.parents[value] = self.find(self.parents[value])
        return self.parents[value]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def assign_groups(rows, script_records):
    """Keep sessions, learned parents and all same-weight serving variants together."""
    uf = UnionFind()
    by_checkpoint = {row["checkpoint_id"]: row["session_id"] for row in rows}
    by_checkpoint.update({k: v["session_id"] for k, v in script_records.items()})
    # Unscored ancestors can connect scored descendants to other sessions.
    for script in script_records.values():
        uf.find(script["session_id"])
        for parent in script.get("parent_checkpoint_ids", []):
            if parent in by_checkpoint:
                uf.union(script["session_id"], by_checkpoint[parent])
    by_weight = {}
    for row in rows:
        session = row["session_id"]
        uf.find(session)
        h = row.get("weights_sha256")
        if h:
            if h in by_weight:
                uf.union(session, by_weight[h])
            by_weight[h] = session
        for parent in script_records.get(row["checkpoint_id"], {}).get("parent_checkpoint_ids", []):
            if parent in by_checkpoint:
                uf.union(session, by_checkpoint[parent])
    members = defaultdict(set)
    for session in list(uf.parents):
        members[uf.find(session)].add(session)
    reserved = {uf.find(r["session_id"]) for r in rows
                if r.get("source_split") in {"locked_test", "locked_session_test"}}
    for row in rows:
        component = uf.find(row["session_id"])
        group = "group-" + digest(sorted(members[component]))[:16]
        row["group_id"] = group
        if component in reserved:
            split = "test"
        else:
            bucket = int(hashlib.sha256((SALT + group).encode()).hexdigest()[:8], 16) % 5
            split = "validation" if bucket == 0 else "train"
        row["split"] = split
    return [{"group_id": "group-" + digest(sorted(v))[:16], "sessions": sorted(v),
             "reserved_test": k in reserved} for k, v in sorted(members.items())]


def choose_representatives(rows):
    """Select one ten-run record per verified weights/effective-config cell."""
    groups = defaultdict(list)
    for row in rows:
        if row.get("weights_sha256") and row.get("serving_sha256"):
            key = (row["track"], row["benchmark"], row["weights_sha256"], row["serving_sha256"],
                   row.get("protocol_fingerprint"), row.get("model_artifacts_fingerprint"))
        else:
            key = (row["example_id"],)
        groups[key].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda r: (r["status"] != "eligible", r["example_id"]))
        for row in ordered[1:]:
            row["duplicate_of"] = ordered[0]["example_id"]
            row["status"] = "duplicate"
            row["exclusion_reasons"].append("duplicate_weights_and_effective_serving")


def result_identity(path, data):
    if "eval_matrix_1k" in data:
        m = data["eval_matrix_1k"]
        split = {"locked_session_test": "locked_test", "locked_test": "locked_test",
                 "development": "development"}.get(m.get("split"))
        if split is None:
            raise ValueError("Unknown archived matrix split: " + str(m.get("split")))
        if ("locked_test" in Path(path).parts) != (split == "locked_test"):
            raise ValueError("Archived matrix split disagrees with file location: " + str(path))
        return m.get("checkpoint_id"), m.get("benchmark"), "ptb_controlled", split
    m = data.get("rescore10") or {}
    checkpoint = m.get("id", Path(path).stem)
    track = "dojo_native" if checkpoint.startswith("abgsm8k-") else "ptb_native"
    return checkpoint, m.get("benchmark"), track, None


def unknown_script(checkpoint, benchmark):
    session = checkpoint.rsplit("-exp", 1)[0]
    return {"session_id": session, "benchmark": benchmark, "scripts": [], "launch": {},
            "status": "candidate", "exclusion_reasons": ["experiment_scripts_not_in_recorder_index"],
            "review_flags": [], "parent_checkpoint_ids": [], "provenance": {}}


def predictor_payload(script, serving):
    """Allowlist the two input components; audit fields and outcomes stay elsewhere."""
    def clean_launch(launch):
        return {k: v for k, v in launch.items() if k in {"argv", "cwd", "env", "script", "entrypoint"}}

    def package(record):
        scripts = [{k: item[k] for k in ("path", "role", "content") if k in item}
                   for item in record["scripts"]]
        launch = record.get("launch", {})
        result = {"files": scripts, "launch": clean_launch(launch)}
        if launch.get("data_builders"):
            result["data_builder_launches"] = [clean_launch(item) for item in launch["data_builders"]]
        if launch.get("parent_recipes"):
            result["parent_recipes"] = [package(item) for item in launch["parent_recipes"]]
        return result

    result = package(script)
    # Preserve path equality and script behavior relationships while removing
    # source-session IDs. These are input aliases, never executed code rewrites.
    serialized = json.dumps(result, ensure_ascii=False)
    identifiers = sorted(set(re.findall(
        r"(?:aime2?-r0|gsm2-r0|r0|opus\w+-r0|glm\w+-r0)-\d+(?:-exp-[A-Za-z0-9_-]+)?",
        serialized)), key=lambda value: (-len(value), value))
    aliases = {value: f"checkpoint_ref_{i + 1}" for i, value in enumerate(identifiers)}

    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, str):
            value = value.replace("/home/ben/task", "/workspace")
            value = re.sub(r"/dojo-runs/\d+", "/workspace", value)
            for original, alias in aliases.items():
                value = value.replace(original, alias)
        return value
    return {"experiment_scripts": normalize(result), "serving_config": serving}


def build(source_root, receipt_path, out, *, script_records=None, verify_source=True):
    source_root, out = Path(source_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be a new or empty directory")
    receipt = read_json(receipt_path)
    revision = receipt["revision"]
    dependency_names = {
        "prefix_dataset.py", "prefix_code_recovery.py", "prefix_code_fragments.py",
        "prefix_generation.py", "rpm_code_provenance.py", "wm_checkpoint_inputs.py",
        "wm_checkpoint_paths.py", "wm_code_features.py",
    }
    builder_files = sorted(p for p in (ROOT / "tools/outcome_prediction").glob("*.py")
                           if p.name.startswith("hf_benchmark") or p.name in dependency_names)
    builder_sources = {p.name: file_digest(p) for p in builder_files}
    source_files = receipt.get("files", [])
    if receipt.get("unavailable_file_count", len(receipt.get("unavailable", []))):
        raise ValueError("Selected source fetch is incomplete; inspect its receipt")
    verified = {r["path"]: r for r in source_files}
    if verify_source:
        inventory_path = Path(receipt_path).parent / "inventory.json"
        for key, path in (("inventory_sha256", inventory_path),
                          ("selection_plan_sha256", Path(receipt_path).parent / "selection_plan.json")):
            if receipt.get(key) and file_digest(path) != receipt[key]:
                raise ValueError("Source receipt anchor mismatch: " + key)
        for rel, record in verified.items():
            if file_digest(source_root / rel) != record["sha256"]:
                raise ValueError("Source digest mismatch: " + rel)
    if script_records is None:
        from tools.outcome_prediction.hf_benchmark_dojo import extract_dojo_scripts
        from tools.outcome_prediction.hf_benchmark_scripts import extract_ptb_scripts
        script_records = extract_ptb_scripts(source_root)
        script_records.update(extract_dojo_scripts(source_root))
    paths = sorted(rel for rel in verified if rel.endswith(".json") and
                   (rel.startswith("rescore10/results/") or
                    (rel.startswith("eval_matrix_1k/") and "/results/" in rel)))
    rows, labels, inputs = [], {}, {}
    expected_questions = {}
    # The archived historical benchmark matrices define fixed question IDs.
    # Require unanimous per-benchmark ID sets, rather than silently accepting N only.
    for rel in paths:
        data = read_json(source_root / rel)
        _, benchmark, _, _ = result_identity(rel, data)
        if benchmark in N_QUESTIONS and isinstance(data.get("per_problem"), dict):
            ids = frozenset(data["per_problem"])
            if len(ids) == N_QUESTIONS[benchmark]:
                expected_questions.setdefault(benchmark, Counter())[ids] += 1
    expected_questions = {b: count.most_common(1)[0][0] for b, count in expected_questions.items()}
    for rel in paths:
        data = read_json(source_root / rel)
        checkpoint, benchmark, track, source_split = result_identity(rel, data)
        if not checkpoint:
            raise ValueError("Missing checkpoint identity: " + rel)
        example = "wm-" + digest({"revision": revision, "source_path": rel})[:20]
        script = script_records.get(checkpoint, unknown_script(checkpoint, benchmark))
        label = validate_label(data, benchmark, expected_questions.get(benchmark))
        serving = extract_serving(data, source_root, checkpoint, benchmark)
        reasons = list(script.get("exclusion_reasons", [])) + list(label["errors"]) + serving["errors"]
        if script["status"] != "eligible" and not script.get("exclusion_reasons"):
            reasons.append("script_binding_incomplete")
        if checkpoint in QUARANTINE:
            reasons.append("checkpoint_recipe_binding_quarantine")
        if script.get("benchmark") and benchmark != script["benchmark"]:
            reasons.append("script_label_benchmark_mismatch")
        if (script.get("base_model") and serving.get("config")
                and script["base_model"] != serving["config"].get("base_model")):
            reasons.append("script_serving_base_model_mismatch")
        matrix = data.get("eval_matrix_1k") or {}
        if matrix and matrix.get("exp_id") != Path(rel).stem:
            reasons.append("matrix_example_id_mismatch")
        sidecar = rel.replace("/results/", "/trajectories/").removesuffix(".json") + ".json.gz"
        # Source inventory, rather than the deliberately small fetch selection,
        # determines whether a large log is archived remotely.
        inventory_path = Path(receipt_path).parent / "inventory.json"
        if not rows:
            inv = read_json(inventory_path)
            inventory_paths = {r["path"] for r in inv["files"]}
        row = {
            "example_id": example, "source_path": rel, "source_revision": revision,
            "source_sha256": verified[rel]["sha256"], "checkpoint_id": checkpoint,
            "benchmark": benchmark, "track": track, "session_id": script["session_id"],
            "source_split": source_split, "label_status": label["status"],
            "script_status": script["status"], "serving_status": serving["status"],
            "status": "eligible" if not reasons else "candidate",
            "exclusion_reasons": sorted(set(reasons)),
            "review_flags": sorted(set(script.get("review_flags", []))),
            "weights_sha256": (matrix.get("digests") or {}).get("weights_sha256"),
            "model_artifacts_fingerprint": digest({k: v for k, v in
                ((matrix.get("preflight") or {}).get("file_sha256") or {}).items()
                if k in {"config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"}})
                if matrix else None,
            "serving_sha256": digest(serving["config"]) if serving["config"] else None,
            "trajectory_archived": sidecar in inventory_paths,
            "trajectory_source_path": sidecar if sidecar in inventory_paths else None,
            "script_provenance": script.get("provenance", {}),
            "serving_audit": serving["audit"],
            "protocol_fingerprint": digest({"runtime": matrix.get("runtime"),
                "evaluator": (matrix.get("digests") or {}).get("evaluator_kit_sha256"),
                "dataset": (matrix.get("digests") or {}).get("dataset_fingerprint_sha256"),
                "scorer_patch": data.get("scorer_patch")}),
        }
        rows.append(row)
        labels[example] = {"example_id": example, **label}
        if script.get("scripts") and serving.get("config"):
            inputs[example] = {"example_id": example, "x": predictor_payload(script, serving["config"])}
    # Include unscored indexed checkpoints in the availability audit, not as zero labels.
    scored_checkpoints = {r["checkpoint_id"] for r in rows}
    unscored = [{"checkpoint_id": k, "session_id": v["session_id"],
                 "script_status": v["status"], "reason": "no_ten_run_result",
                 "exclusion_reasons": v.get("exclusion_reasons", [])}
                for k, v in sorted(script_records.items()) if k not in scored_checkpoints]
    groups = assign_groups(rows, script_records)
    choose_representatives(rows)
    for row in rows:
        if row["example_id"] in inputs:
            row["input_sha256"] = digest(inputs[row["example_id"]]["x"])
    eligible = [r for r in rows if r["status"] == "eligible"]
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [r for r in eligible if r["split"] == split]
        write_jsonl(out / "inputs" / (split + ".jsonl"), [inputs[r["example_id"]] for r in subset])
        write_jsonl(out / "labels" / (split + ".jsonl"), [labels[r["example_id"]] for r in subset])
    write_jsonl(out / "audit" / "registry.jsonl", rows)
    write_jsonl(out / "audit" / "unscored_checkpoints.jsonl", unscored)
    write_jsonl(out / "audit" / "all_labels.jsonl", [labels[r["example_id"]] for r in rows])
    write_jsonl(out / "audit" / "script_records.jsonl", [{"checkpoint_id": k, **v}
                                                         for k, v in sorted(script_records.items())])
    write_json(out / "groups.json", groups)
    write_json(out / "protocol.json", {
        "schema": SCHEMA, "x": ["experiment_scripts", "serving_config"],
        "y": "mean_accuracy_over_ten_runs", "scale": "fraction_0_to_1",
        "n_runs": 10, "question_counts": N_QUESTIONS,
        "split": "existing locked matrix groups remain test; hash 20% of other groups into validation",
        "split_salt": SALT,
        "primary_metric": "per-benchmark group-balanced MAE in percentage points",
        "comparison": "paired group bootstrap on identical rows; same TRAIN examples for all predictors",
        "missing_predictions": "report coverage; full benchmark score requires every eligible prediction",
        "deduplication": "one ten-run record per verified weights and effective serving config per track",
        "labels_access": "only train labels may be provided to predictor fitting; keep audit and other labels out of model tools",
        "exposure": "retrospective historically inspected sessions; locked aggregate reports previously reviewed",
        "scope": "ten evaluations of realized weights, not ten independent training repetitions",
    })
    counts = {
        "source_result_files": len(rows), "complete_labels": sum(r["label_status"] == "complete" for r in rows),
        "eligible_examples": len(eligible), "candidate_examples": sum(r["status"] == "candidate" for r in rows),
        "duplicate_examples": sum(r["status"] == "duplicate" for r in rows),
        "unscored_checkpoints": len(unscored),
        "eligible_checkpoint_ids": len({r["checkpoint_id"] for r in eligible}),
        "eligible_groups": len({r["group_id"] for r in eligible}),
        "eligible_by_split": dict(Counter(r["split"] for r in eligible)),
        "eligible_by_benchmark": dict(Counter(r["benchmark"] for r in eligible)),
        "source_by_track": dict(Counter(r["track"] for r in rows)),
        "eligible_by_track": dict(Counter(r["track"] for r in eligible)),
        "exclusion_reasons": dict(Counter(reason for r in rows for reason in r["exclusion_reasons"])),
        "results_without_archived_trajectory": sum(not r["trajectory_archived"] for r in rows),
    }
    manifest = {
        "schema": SCHEMA, "release_status": "retrospective_benchmark_draft_automated_gates",
        "repository": REPO, "source_revision": revision, "source_receipt_sha256": file_digest(receipt_path),
        "counts": counts, "construction_complete": True, "all_source_results_accounted_for": True,
        "limitations": [
            "Automatic source, label and split validation is not an exhaustive semantic code review.",
            "Core eligibility requires extracted launch scripts and recorded effective serving settings.",
            "Native and Dojo results remain candidates when effective serving state is unverified.",
            "Missing full logs do not invalidate complete per-question labels; their absence limits re-audit.",
            "Archive IDs and file hashes are metadata, not predictor features.",
            "Source archive excludes some external helper/data dependencies; extraction records its limits.",
            "Historical exposure precludes claiming an untouched prospective training-recipe test.",
        ],
        "builder_sources": builder_sources,
    }
    if builder_sources != {p.name: file_digest(p) for p in builder_files}:
        raise ValueError("Builder source changed during construction; rebuild in a fresh directory")
    write_json(out / "manifest.json", manifest)
    checks = verify(out)
    write_json(out / "verification.json", checks)
    if not checks["passed"]:
        raise ValueError("Benchmark verification failed: " + ", ".join(checks["errors"]))
    write_readme(out, manifest)
    hashes = {str(p.relative_to(out)): file_digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "files.sha256.json", hashes)
    return manifest


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(out):
    out = Path(out)
    registry = read_jsonl(out / "audit/registry.jsonl")
    eligible = {r["example_id"]: r for r in registry if r["status"] == "eligible"}
    errors, memberships, x_count, y_count = [], defaultdict(set), 0, 0
    if (out / "files.sha256.json").exists():
        for rel, expected in read_json(out / "files.sha256.json").items():
            path = out / rel
            if not path.is_file() or file_digest(path) != expected:
                errors.append("artifact_hash_mismatch:" + rel)
    all_ids = set()
    for split in ("train", "validation", "test"):
        inputs = read_jsonl(out / "inputs" / (split + ".jsonl"))
        labels = read_jsonl(out / "labels" / (split + ".jsonl"))
        if not inputs:
            errors.append("empty_core_split:" + split)
        ids = [r["example_id"] for r in inputs]
        label_ids = [r["example_id"] for r in labels]
        if len(ids) != len(set(ids)) or set(ids) & all_ids:
            errors.append("duplicated_example_membership")
        all_ids.update(ids)
        if len(label_ids) != len(set(label_ids)):
            errors.append("duplicate_exported_label_ids")
        if set(ids) != set(label_ids) or len(ids) != len(labels):
            errors.append("input_label_join_mismatch:" + split)
        for record in inputs:
            if set(record) != {"example_id", "x"} or set(record["x"]) != {"experiment_scripts", "serving_config"}:
                errors.append("unexpected_predictor_input_fields")
            row = eligible.get(record["example_id"])
            if row is None or row["split"] != split:
                errors.append("ineligible_or_wrong_split_input")
                continue
            if row.get("source_split") in {"locked_test", "locked_session_test"} and split != "test":
                errors.append("reserved_source_example_outside_test")
            if digest(record["x"]) != row["input_sha256"]:
                errors.append("input_digest_mismatch")
            if not record["x"]["experiment_scripts"]["files"]:
                errors.append("empty_scripts")
            memberships["group:" + row["group_id"]].add(split)
            memberships["checkpoint:" + row["checkpoint_id"]].add(split)
            if row.get("weights_sha256"):
                memberships["weights:" + row["weights_sha256"]].add(split)
        for label in labels:
            if label["status"] != "complete" or not is_rate(label["y"]) or len(label["run_accuracies"]) != 10:
                errors.append("invalid_exported_label")
            if abs(statistics.mean(label["run_accuracies"]) - label["y"]) > 1e-12:
                errors.append("exported_label_mean_mismatch")
        x_count += len(inputs)
        y_count += len(labels)
    if all_ids != set(eligible):
        errors.append("eligible_export_membership_mismatch")
    if any(len(splits) != 1 for splits in memberships.values()):
        errors.append("group_or_weights_cross_split")
    if not x_count:
        errors.append("no_eligible_examples")
    return {"passed": not errors, "errors": sorted(set(errors)), "input_rows_checked": x_count,
            "label_rows_checked": y_count, "registry_rows_checked": len(registry),
            "does_not_certify": ["exhaustive semantic leakage freedom", "unrecorded weight ancestry",
                                 "prospective untouched test exposure", "complete external dependencies"]}


def write_readme(out, manifest):
    c = manifest["counts"]
    text = f"""# Scripts + serving benchmark v1

Pinned HF revision: `{manifest['source_revision']}`.

X is the supplied experiment scripts (including bound launch arguments/helpers) plus
the effective serving configuration. Y is mean benchmark accuracy over ten complete
runs on the same questions, on a 0–1 scale. All ten repeats yield one example.

This is a retrospective benchmark draft with automated extraction gates. It contains
**{c['eligible_examples']} eligible examples**, **{c['eligible_checkpoint_ids']} checkpoint IDs**
and **{c['eligible_groups']} session/learned-weight groups**. All **{c['source_result_files']}**
source results are accounted for in the audit, including duplicates and candidates.

| Split | Eligible examples |
|---|---:|
| Train | {c['eligible_by_split'].get('train', 0)} |
| Validation | {c['eligible_by_split'].get('validation', 0)} |
| Test | {c['eligible_by_split'].get('test', 0)} |

## Files and use

- `inputs/<split>.jsonl`: opaque example ID and X only.
- `labels/<split>.jsonl`: Y, ten run accuracies and the per-question correctness matrix.
- `audit/registry.jsonl`: every result's provenance, eligibility, grouping and exclusion reasons.
- `audit/unscored_checkpoints.jsonl`: indexed script/checkpoint records without ten-run labels.
- `audit/script_records.jsonl`: extracted scripts and their evidence; not a model retrieval corpus.
- `manifest.json`, `protocol.json`, `groups.json`, `verification.json`, `files.sha256.json`:
  version, task definition, split groups, automated verification and file hashes.

Score a prediction file from the repository root:

```sh
.venv/bin/python -m tools.outcome_prediction.hf_benchmark_score BENCHMARK_DIR predictions.jsonl --split test
```

Add `--compare other_predictions.jsonl` for paired group-level differences and intervals.
Verify an exported release with `python -m tools.outcome_prediction.hf_benchmark --out BENCHMARK_DIR --verify-only`.

Provide predictors only the intended inputs and TRAIN labels. Validation/test labels,
the audit, raw archive and previous analysis reports must stay outside predictor tools.
Run fitting, feature selection and prompt selection using training/validation groups;
freeze them before final test scoring. Preserve the existing reserved matrix sessions.
Additional same-session or same-weight native scores cannot become training evidence.
Supplied script packages include bound data-builder invocations and, for eligible
continuations, explicitly reconstructed parent recipe scripts. Those are part of X;
parent scores and outcome text are excluded. Unresolved parent recipes remain candidates.

The primary score is per-benchmark group-balanced MAE in percentage points. Also report
cell-weighted MAE and R². Confidence intervals must resample entire groups and compare
predictors on the same examples. Prediction files contain `example_id` and `prediction`
(a finite number in [0, 1]); the scorer reports missing-prediction coverage explicitly.

## Coverage and limits

The primary track uses controlled PTB matrix serving. Historical native-config and
Dojo results are inventoried separately; a valid numeric label alone does not establish
effective serving or script binding. Their candidates can be promoted in a later
version when missing evidence is resolved. Dojo uses Qwen/GSM8K and selects final
checkpoints by development performance, unlike the Gemma/GSM8K PTB population.

Duplicate weights/configs retain one deterministic ten-run label; repeats and aliases
are not additional independent recipes. Missing full trajectory sidecars are recorded
separately from validity of the per-question labels. Automated code sanitation and
timestamp checks are not an exhaustive semantic or execution-provenance review.

The sessions and some locked aggregate outcomes have been inspected historically.
Report retrospective generalization, not an untouched prospective training-recipe test.
Ten repeated evaluations characterize one trained checkpoint, not retraining variation.
No model training, GPU evaluations, or API predictor calls were run to build this release.
"""
    (Path(out) / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--script-records", type=Path,
                        help="Reuse a separately extracted JSON map with verified provenance")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.out)
    else:
        if not args.source_root or not args.receipt:
            parser.error("--source-root and --receipt are required for building")
        result = build(args.source_root, args.receipt, args.out,
                       script_records=read_json(args.script_records) if args.script_records else None)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
