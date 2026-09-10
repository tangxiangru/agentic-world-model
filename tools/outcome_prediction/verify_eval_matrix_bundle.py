"""Read-only, stdlib validation of the portable 1,000-cell proposal bundle.

This checks static consistency only. It neither verifies remote weights/runtime
behavior nor approves GPU spending, and it never launches jobs or uses a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "experiments/eval_matrix_1k"
PHASE_COUNTS = {
    "1_operational_pilot": 100,
    "2_development_core": 597,
    "3_diagnostic_extensions": 16,
    "4_locked_test": 287,
}
CORE = {"gsm8k": {"G01", "G02"}, "aime2025": {"A01", "A02", "A03"}}
EXTRA = {"A04", "A05"}
POLICIES = set().union(*CORE.values(), EXTRA)
POLICY_SETTINGS = {
    "G01": (1, 64, .95, 1, 4000), "G02": (0, 0, 1, 1, 4000),
    "A01": (1, 0, 1, 1, 16000), "A02": (.6, 20, .95, 1, 16000),
    "A03": (1, 0, 1, 1, 2048), "A04": (.6, 20, .95, 1.05, 16000),
    "A05": (0, 0, 1, 1, 16000),
}
DISCLAIMER = (
    "STATIC VALIDATION ONLY: not launch readiness, weight verification, "
    "resolved-runtime verification, or GPU execution approval."
)


class InvalidBundle(ValueError):
    """A static consistency requirement failed."""


def require(condition, message):
    if not condition:
        raise InvalidBundle(message)


def local_path(root, relative):
    require(isinstance(relative, str) and relative != "", "Empty/non-string asset path")
    parts = PurePosixPath(relative)
    require(
        not parts.is_absolute() and ".." not in parts.parts and "\\" not in relative
        and str(parts) == relative and relative != ".",
        f"Unsafe/noncanonical asset path: {relative}",
    )
    path = root / relative
    require(path.resolve().is_relative_to(root), f"Asset escapes bundle: {relative}")
    return path


def read_json(root, relative, lines=False):
    path = local_path(root, relative)
    try:
        content = path.read_text(encoding="utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()] \
            if lines else json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidBundle(f"Cannot read {relative}: {type(exc).__name__}") from exc


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hash_map(root, required):
    manifest = read_json(root, "bundle_files.sha256.json")
    require(isinstance(manifest, dict), "Bundle hash map must be a JSON object")
    require(manifest.get("schema") == "eval-matrix-bundle-files-sha256-v1",
            "Unsupported bundle hash-map schema")
    require(manifest.get("algorithm") == "sha256", "Hash algorithm must be sha256")
    files, excluded = manifest.get("files"), manifest.get("excluded", [])
    require(isinstance(files, dict) and files, "Bundle hash map must contain files")
    require(isinstance(excluded, list) and all(isinstance(x, str) for x in excluded),
            "Hash-map exclusions must be explicit relative paths")
    require(not (set(files) & set(excluded)), "Asset cannot be both hashed and excluded")
    require(required <= set(files), f"Required assets absent from hash map: {sorted(required - set(files))}")
    for relative in excluded:
        local_path(root, relative)
    for relative, expected in files.items():
        path = local_path(root, relative)
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"Invalid SHA256 for {relative}")
        require(path.is_file(), f"Hashed asset missing: {relative}")
        require(sha256(path) == expected, f"SHA256 mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    missing = actual - set(files) - set(excluded) - {"bundle_files.sha256.json"}
    require(not missing, f"Unhashed assets not explicitly excluded: {sorted(missing)}")
    return len(files), excluded


def reject_outcomes(value):
    forbidden = {"accuracy", "avg_pass_rate", "per_run_pass_rate", "label", "output"}
    if isinstance(value, dict):
        require(not (forbidden & set(value)), "Outcome fields found in proposal records")
        for item in value.values():
            reject_outcomes(item)
    elif isinstance(value, list):
        for item in value:
            reject_outcomes(item)


def canonical_rows(rows):
    return sorted(json.dumps(row, sort_keys=True) for row in rows)


def _verify(root, warnings):
    require(root.is_dir(), f"Bundle directory missing: {root}")
    required = {
        "experiment_matrix.jsonl", "selected_checkpoints.jsonl", "selected_ids_400.json",
        "checkpoint_inventory.json", "protocol.json", "splits.json", "matrix_summary.json",
    }
    required |= {f"phases/{phase}.jsonl" for phase in PHASE_COUNTS}
    required |= {f"configs/{p}/{name}.json" for p in POLICIES
                 for name in ("generation_config", "request_template")}
    hashed_count, excluded = check_hash_map(root, required)
    jobs = read_json(root, "experiment_matrix.jsonl", lines=True)
    checkpoints = read_json(root, "selected_checkpoints.jsonl", lines=True)
    ids = read_json(root, "selected_ids_400.json")
    inventory = read_json(root, "checkpoint_inventory.json")["checkpoints"]
    protocol = read_json(root, "protocol.json")
    splits = read_json(root, "splits.json")
    require(len(jobs) == 1000, "Expected 1,000 matrix rows")
    require(len({r["exp_id"] for r in jobs}) == 1000, "Duplicate exp_id")
    require(len({(r["checkpoint_id"], r["generation_config_id"]) for r in jobs}) == 1000,
            "Duplicate checkpoint-policy combination")
    require(len(checkpoints) == len({r["checkpoint_id"] for r in checkpoints}) == 400,
            "Expected 400 unique checkpoint IDs")
    by_id = {r["checkpoint_id"]: r for r in checkpoints}
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 400 and set(ids) == set(by_id),
            "Selected ID list disagrees with checkpoint records")
    require({r["checkpoint_id"] for r in jobs} == set(by_id), "Matrix checkpoint IDs disagree")
    require(Counter(r["benchmark"] for r in checkpoints) == {"gsm8k": 240, "aime2025": 160},
            "Checkpoint benchmark allocation must be 240 GSM8K / 160 AIME")
    candidates = {r["exp_id"] for r in inventory if r["candidate_weight_bool"]}
    base = {r["exp_id"] for r in inventory if r["from_base"] and r.get("eligible", True)}
    require(set(by_id) <= candidates, "Selection includes a non-candidate weight/alias record")
    require(not any(r.get("alias_targets") or r.get("weight_alias_representative_if_explicitly_known")
                    for r in inventory if r["exp_id"] in by_id),
            "Selection includes a declared weight-alias record")
    require(len(base) == 313 and base <= set(by_id), "All 313 eligible from-base IDs must be retained")
    require(sum(r["parent_kind"] == "base" for r in checkpoints) == 313,
            "Selected checkpoint parent-kind count is inconsistent")
    reject_outcomes(jobs)
    reject_outcomes(checkpoints)
    session_splits, group_splits, checkpoint_policies = defaultdict(set), defaultdict(set), defaultdict(set)
    for job in jobs:
        cp = by_id[job["checkpoint_id"]]
        for field in ("benchmark", "trajectory_id", "known_lineage_group", "split", "checkpoint_uri"):
            require(job[field] == cp[field], f"Checkpoint/job {field} disagreement: {job['exp_id']}")
        bench, policy = job["benchmark"], job["generation_config_id"]
        require(bench in CORE and policy in CORE[bench] | (EXTRA if bench == "aime2025" else set()),
                f"Incompatible benchmark/policy: {job['exp_id']}")
        require(job["exp_id"] == f"poc1k-{job['checkpoint_id']}-{policy}", "Unexpected exp_id binding")
        require(type(job["n_passes"]) is int and job["n_passes"] == 10, "Every cell requires ten passes")
        require(type(job["n_questions_per_pass"]) is int and job["n_questions_per_pass"] ==
                (1319 if bench == "gsm8k" else 30), "Incorrect fixed benchmark question count")
        require(job["release_status"] == "proposal_only_preflight_required", "Cell is not preflight-required")
        require(job["historical_reuse_status"] == "unverified_full_protocol_and_artifact_equivalence",
                "Historical reuse must remain explicitly unverified")
        require(type(job["historical_sampling_cap_match"]) is bool, "Invalid historical-match flag")
        require(type(job["primary_evaluation"]) is bool and job["primary_evaluation"] == (policy in CORE[bench]),
                "Core/diagnostic flag disagrees with policy")
        require(job["split"] in {"development", "locked_session_test"}, "Unknown split")
        require((job["phase"] == "4_locked_test") == (job["split"] == "locked_session_test"),
                "Locked split/phase disagreement")
        require(job["primary_evaluation"] or job["split"] == "development", "Diagnostic cell in locked test")
        session_splits[job["trajectory_id"]].add(job["split"])
        group_splits[job["known_lineage_group"]].add(job["split"])
        checkpoint_policies[job["checkpoint_id"]].add(policy)
        for stem in ("generation_config", "request_template"):
            relative = f"configs/{policy}/{stem}.json"
            require(job[stem + "_path"] == relative, f"Noncanonical {stem} binding")
            require(sha256(local_path(root, relative)) == job[stem + "_sha256"],
                    f"Policy hash mismatch for {job['exp_id']}")
        require(job["protocol_path"] == "protocol.json", "Unexpected protocol path")
    require(all(len(v) == 1 for v in session_splits.values()), "Session mixed across splits")
    require(all(len(v) == 1 for v in group_splits.values()), "Learned-weight group mixed across splits")
    for checkpoint, policies in checkpoint_policies.items():
        require(CORE[by_id[checkpoint]["benchmark"]] <= policies, "Checkpoint missing a shared core policy")
        require(not (policies & EXTRA) or EXTRA <= policies, "Incomplete paired diagnostic policies")
    require(sum(r["primary_evaluation"] for r in jobs) == 960, "Expected 960 core and 40 diagnostic cells")
    require(Counter(r["phase"] for r in jobs) == PHASE_COUNTS, "Incorrect phase allocation")
    for phase in PHASE_COUNTS:
        phase_jobs = read_json(root, f"phases/{phase}.jsonl", lines=True)
        expected = [r for r in jobs if r["phase"] == phase]
        require(canonical_rows(phase_jobs) == canonical_rows(expected),
                f"Phase partition/content mismatch: {phase}")
    pilot = [r for r in jobs if r["phase"] == "1_operational_pilot"]
    require(Counter(r["generation_config_id"] for r in pilot) ==
            {"G01": 20, "G02": 20, "A01": 12, "A02": 12, "A03": 12, "A04": 12, "A05": 12},
            "Pilot must cover all seven policies with the 20-GSM/12-AIME allocation")
    locked = [r for r in jobs if r["split"] == "locked_session_test"]
    require(len({r["trajectory_id"] for r in locked}) == 40 and
            len({r["known_lineage_group"] for r in locked}) == 40,
            "Expected 40 locked whole-session groups")
    for bench in CORE:
        actual = {r["trajectory_id"] for r in locked if r["benchmark"] == bench}
        require(len(actual) == 20 and actual == set(splits[bench]["locked_sessions"])
                and splits[bench]["n_locked_sessions"] == 20, "Split manifest disagrees with locked sessions")
        require(not (actual & set(splits[bench]["known_anchor_sessions_development_only"])),
                "Mechanism-anchor session appears in locked test")
    for policy in sorted(POLICIES):
        gen = read_json(root, f"configs/{policy}/generation_config.json")
        request = read_json(root, f"configs/{policy}/request_template.json")
        extra = request["extra_body"]
        settings = tuple(gen[k] for k in ("temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"))
        require(settings == POLICY_SETTINGS[policy] and gen["min_p"] == 0,
                f"Policy settings differ from the frozen proposal: {policy}")
        require(gen["eos_token_id"] == ([1, 106] if policy.startswith("G") else [151643, 151645]),
                f"Policy stop-token set differs from the proposal: {policy}")
        for field in ("temperature", "top_p"):
            require(gen[field] == request[field], f"Config/request {field} mismatch: {policy}")
        for field in ("top_k", "min_p", "repetition_penalty"):
            require(gen[field] == extra[field], f"Config/request {field} mismatch: {policy}")
        require(gen["max_new_tokens"] == request["max_tokens"], f"Output cap mismatch: {policy}")
        require(gen["eos_token_id"] == extra["stop_token_ids"], f"Stop-token mismatch: {policy}")
        require(gen["do_sample"] is (gen["temperature"] > 0), f"Sampling intent mismatch: {policy}")
        require(extra["ignore_eos"] is False and extra["min_tokens"] == 0 and request["n"] == 1,
                f"Invalid request count/stop defaults: {policy}")
        require(request["presence_penalty"] == request["frequency_penalty"] == 0 and request["stop"] == [],
                f"Unexpected extra penalties/stops: {policy}")
    require(protocol["status"] == "proposal_not_execution_authorization", "Protocol must remain a proposal")
    require(protocol["n_passes"] == 10 and protocol["replicate_ids"] == list(range(10)),
            "Protocol replicate definition must match ten passes")
    require(protocol["server_generation_config_mode"] == "vllm", "Unexpected generation inheritance mode")
    require(protocol["target_metric"] == "mean_pass_at_1_over_ten_complete_equal_question_passes",
            "Unexpected target metric")
    require(protocol["missing_pass_policy"] == "invalid_until_retried_not_zero_not_dropped",
            "Unsafe missing-pass policy")
    require(isinstance(protocol.get("runtime_release_requirements"), list)
            and len(protocol["runtime_release_requirements"]) >= 5, "Runtime preflight requirements missing")
    for flag, description in (
        ("weight_shard_hashes_verified", "lack verified weight-shard hashes"),
        ("tokenizer_template_compatibility_verified", "lack verified tokenizer/template compatibility"),
    ):
        n = sum(r.get(flag) is not True for r in checkpoints)
        if n:
            warnings.append(f"{n} checkpoints {description}.")
    n = sum(r.get("training_only_launch_bundle_status") != "frozen" for r in checkpoints)
    if n:
        warnings.append(f"{n} training-launch-only input bundles are not frozen.")
    for flag in ("v6_missing_declared_code", "v6_content_review_required"):
        n = sum(bool(r.get(flag)) for r in checkpoints)
        if n:
            warnings.append(f"{n} checkpoints retain {flag} flags.")
    matches = sum(r["historical_sampling_cap_match"] for r in jobs)
    locked_matches = sum(r["historical_sampling_cap_match"] for r in locked)
    warnings.append(f"{matches} historical sampling/cap matches ({locked_matches} locked) are not verified replays or fresh counterfactuals.")
    warnings.append("Hash-map exclusions are not authenticated: " + ", ".join(excluded))
    return {"cells": len(jobs), "checkpoint_candidates": len(checkpoints), "core_cells": 960,
            "diagnostic_cells": 40, "locked_sessions": 40, "locked_cells": len(locked),
            "phase_counts": PHASE_COUNTS, "policy_count": 7, "hashed_assets": hashed_count}


def verify_bundle(bundle=DEFAULT_BUNDLE):
    """Return JSON-serializable checks; never mutate files or inspect remote assets."""
    warnings = [DISCLAIMER]
    try:
        counts = _verify(Path(bundle).resolve(), warnings)
    except (InvalidBundle, KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        return {"ok": False, "status": "static_validation_failed", "errors": [str(exc)],
                "warnings": warnings, "counts": {}}
    return {"ok": True, "status": "static_validation_passed_not_launch_ready", "errors": [],
            "warnings": warnings, "counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", action="store_true", help="Print the structured static-validation report")
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["status"])
        for error in result["errors"]:
            print("ERROR: " + error)
        if result["counts"]:
            print(json.dumps(result["counts"], sort_keys=True))
        for warning in result["warnings"]:
            print("WARNING: " + warning)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
