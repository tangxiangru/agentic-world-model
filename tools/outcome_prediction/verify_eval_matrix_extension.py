"""Statically verify the additive 2K/combined 3K portable proposal; never run jobs.

Only stdlib and the sibling portable 1K verifier are used. Passing does not prove
remote weight identity, input readiness, effective runtime behavior or approval
to spend GPU compute. The original frozen 1K core is checked byte-for-byte.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_matrix_base_verifier", Path(__file__).with_name("verify_eval_matrix_bundle.py")
)
BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BASE)
DEFAULT_BUNDLE = BASE.DEFAULT_BUNDLE
require = BASE.require

# Published baseline commit: 6656e82f9f43fb2ba03e6bc6e159bba8d2a8fee9.
# Captured from the published 1K specification, before the additive expansion.
# These are intentionally independent of the rewritable bundle checksum map.
ORIGINAL_SHA256 = {
    "checkpoint_inventory.json": "2bdc443dbafddb4da8310d992760f4f715dfb28259d085f9a8c25d1ca264b3e5",
    "configs/A01/generation_config.json": "e98580631de0665089853578744594cf92a431a448c258771c30f5958f9d3595",
    "configs/A01/request_template.json": "651f284af98d4780d6e8cd23d8670033bb1b36202d0f46b699958ae46a0d983a",
    "configs/A02/generation_config.json": "ab705d398d1f092772ea1527c6501181ac7561e4e59d9554b5e013768ab6713b",
    "configs/A02/request_template.json": "7d0b7d56d17be76d172f2eb74e7738cc8f9c104a693c7390c875bcfcc35a6eef",
    "configs/A03/generation_config.json": "40f166e4166a5b1c1edf374ddac06f8d58c91b400d28a5da1171b8ace7f14375",
    "configs/A03/request_template.json": "0e046024ef109d91ec6b9bb8a5529ede19c3047de02bade5046dd6121aac3b70",
    "configs/A04/generation_config.json": "5dcedd852a243f6743780b73b3e1dcff41d795f56010ed3db5f76d96c03e73c0",
    "configs/A04/request_template.json": "b971721aa3906febe70a38493132d73a6752c4c0f589b6f8c480f09d679e6500",
    "configs/A05/generation_config.json": "2794abf7c3b355abd4827625494dbe0cf6e04ea17d68126fe3c30c6986fced8b",
    "configs/A05/request_template.json": "4c23940787b454c535ff8fe1d7a5c61315d3907b3f513679810f938b1f12f3d8",
    "configs/G01/generation_config.json": "fc88150763cfbb1e9fb503ccc06c974f3f80579d75ca5ad1229a706fb4dd82ee",
    "configs/G01/request_template.json": "db743798fbdefcb04e88eb46855d14ccb69523039682651554c2f2d5433b8db2",
    "configs/G02/generation_config.json": "f718405061bb4d493e63be39d47b75a78b67f01c3efd818462e81ef2c2ab7449",
    "configs/G02/request_template.json": "bbc2a9617625b74d5def983475c91e549feb3d09c91596ec35a1e94083386957",
    "experiment_matrix.jsonl": "ad22d150782a69cd30fedd6ce7db56a910d86e60f41725d74faf660e55663a8d",
    "generation_policies.json": "d2d3980c392de02ab2e31f40f6083bc170171d566b734ee2901a12d876e58729",
    "matrix_summary.json": "cba28f27b202025445a94b63df98e71cf9d6d98a809b897bf774cdd5e04b89d6",
    "phases/1_operational_pilot.jsonl": "b4a334fbdde003b004e7cd9fa39c696f5e8f3463ef50e3809213d8ef97ac0927",
    "phases/2_development_core.jsonl": "38dcee70818c786ce7b8af18a9a5d392f5863a3fb451a3266c1b95a9d3b50475",
    "phases/3_diagnostic_extensions.jsonl": "f7b33dfae1bb763e271fe579f0c8b2438316d7abf09181589ac9cbb9203b5345",
    "phases/4_locked_test.jsonl": "6d56bb92abf0a8f7a7cfbec42374abb3d4107a80f67506782f3014a02e0beef6",
    "protocol.json": "ca1f03b981431f0a5e18ccb00acd458ca5463e034ecf41d001791f5f7b7fa5c6",
    "selected_checkpoints.jsonl": "854a216d5017b73a9ebb00da8aa8764046f05c879e62caaed7cd77286791a76f",
    "selected_ids_400.json": "fffcce6b7ab680537437a9817b3cb3f4a8c27428a914a01d0c9abd708f855e97",
    "splits.json": "9e631c482aafb51e395952714f63346884f1e1ae1de726997c83c533630ed5c3",
}
NEW_SETTINGS = {
    "G03": (1, 0, 1, 1, 4000), "GX01": (.4, 64, .95, 1, 4000),
    "GX02": (.4, 0, 1, 1, 4000), "AX01": (.6, 0, 1, 1, 16000),
    "AX02": (1, 20, .95, 1, 16000), "AX03": (1, 0, 1, 1, 4096),
}
GRID = {
    "gsm8k": {"G01", "G02", "G03", "GX01", "GX02"},
    "aime2025": {"A01", "A02", "A03", "A04", "A05", "AX01", "AX02"},
}
NEW_PHASES = {"5_extension_pilot": 96, "6_extension_development": 1171, "7_extension_locked_test": 733}
STAGES = (
    ("pilot", 196, ["1_operational_pilot", "5_extension_pilot"]),
    ("development", 1784, ["2_development_core", "3_diagnostic_extensions", "6_extension_development"]),
    ("locked_test", 1020, ["4_locked_test", "7_extension_locked_test"]),
)
ADDITIVE_FILES = {
    "experiment_matrix_extension_2k.jsonl", "experiment_matrix_all_3k.jsonl",
    "selected_checkpoints_all_516.jsonl", "selected_ids_516.json", "extension_summary.json",
    "execution_plan.json", "generation_policies_all_13.json",
} | {f"phases/{phase}.jsonl" for phase in NEW_PHASES} | {
    f"configs/{policy}/{stem}.json" for policy in NEW_SETTINGS
    for stem in ("generation_config", "request_template")
}


def _validate_new_policy(root, policy):
    temp, top_k, top_p, penalty, cap = NEW_SETTINGS[policy]
    stops = [1, 106] if policy.startswith("G") else [151643, 151645]
    expected_gen = {"do_sample": temp > 0, "temperature": temp, "top_k": top_k,
                    "top_p": top_p, "min_p": 0, "repetition_penalty": penalty,
                    "max_new_tokens": cap, "eos_token_id": stops}
    expected_request = {
        "temperature": temp, "top_p": top_p, "max_tokens": cap, "n": 1,
        "presence_penalty": 0, "frequency_penalty": 0, "stop": [],
        "extra_body": {"top_k": top_k, "min_p": 0, "repetition_penalty": penalty,
                       "min_tokens": 0, "ignore_eos": False, "stop_token_ids": stops},
    }
    generation = BASE.read_json(root, f"configs/{policy}/generation_config.json")
    request = BASE.read_json(root, f"configs/{policy}/request_template.json")
    require(all(type(generation[key]) in (int, float) for key in expected_gen
                if key not in {"do_sample", "eos_token_id"})
            and generation["do_sample"] is (temp > 0), f"Invalid generation parameter types: {policy}")
    require(all(type(request[key]) in (int, float) for key in
                ("temperature", "top_p", "max_tokens", "n", "presence_penalty", "frequency_penalty"))
            and all(type(request["extra_body"][key]) in (int, float) for key in
                    ("top_k", "min_p", "repetition_penalty", "min_tokens"))
            and request["extra_body"]["ignore_eos"] is False, f"Invalid request parameter types: {policy}")
    require(generation == expected_gen,
            f"Unexpected frozen extension generation policy: {policy}")
    require(request == expected_request,
            f"Unexpected extension request parameters: {policy}")


def _verify(root, warnings):
    original_report = BASE.verify_bundle(root)
    require(original_report["ok"], "Original 1K bundle check failed: " + "; ".join(original_report["errors"]))
    for relative, expected in ORIGINAL_SHA256.items():
        require(BASE.sha256(BASE.local_path(root, relative)) == expected,
                f"Original frozen bytes changed: {relative}")
    hashed_count, _ = BASE.check_hash_map(root, ADDITIVE_FILES)
    original = BASE.read_json(root, "experiment_matrix.jsonl", lines=True)
    extension = BASE.read_json(root, "experiment_matrix_extension_2k.jsonl", lines=True)
    combined = BASE.read_json(root, "experiment_matrix_all_3k.jsonl", lines=True)
    original_bytes = (root / "experiment_matrix.jsonl").read_bytes()
    extension_bytes = (root / "experiment_matrix_extension_2k.jsonl").read_bytes()
    require((root / "experiment_matrix_all_3k.jsonl").read_bytes() == original_bytes + extension_bytes,
            "Combined matrix must be the exact original bytes followed by extension bytes")
    require(len(extension) == 2000 and len(combined) == 3000, "Expected 2,000 added / 3,000 combined cells")
    require(len({r["exp_id"] for r in combined}) == 3000, "Duplicate combined exp_id")
    require(len({(r["checkpoint_id"], r["generation_config_id"]) for r in combined}) == 3000,
            "Duplicate combined checkpoint-policy combination")
    old_by_id = {r["exp_id"]: r for r in original}
    require(combined[:1000] == original and combined[1000:] == extension, "Original row membership/order changed")
    require(not (set(old_by_id) & {r["exp_id"] for r in extension}), "Extension reuses an original exp_id")
    inventory = BASE.read_json(root, "checkpoint_inventory.json")["checkpoints"]
    inv_by_id = {r["exp_id"]: r for r in inventory}
    candidates = {r["exp_id"] for r in inventory if r["candidate_weight_bool"]}
    require(len(candidates) == 516, "Original inventory no longer has 516 candidates")
    checkpoints = BASE.read_json(root, "selected_checkpoints_all_516.jsonl", lines=True)
    require(len(checkpoints) == len({r["checkpoint_id"] for r in checkpoints}) == 516,
            "Expected 516 unique candidate checkpoint records")
    cp_by_id = {r["checkpoint_id"]: r for r in checkpoints}
    ids = BASE.read_json(root, "selected_ids_516.json")
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == 516
            and set(ids) == set(cp_by_id) == candidates, "516-ID selection must equal the original candidate inventory")
    require(Counter(r["benchmark"] for r in checkpoints) == {"gsm8k": 316, "aime2025": 200},
            "Expanded checkpoint allocation must be 316 GSM8K / 200 AIME")
    original_cps = BASE.read_json(root, "selected_checkpoints.jsonl", lines=True)
    require(all(cp_by_id[r["checkpoint_id"]] == r for r in original_cps), "Original checkpoint metadata changed")
    old_locked = {s for b in BASE.read_json(root, "splits.json").values() for s in b["locked_sessions"]}
    require(len(old_locked) == 40, "Original locked session count changed")
    old_diagnostics = {r["checkpoint_id"] for r in original if r["generation_config_id"] == "A04"}
    require(len(old_diagnostics) == 20, "Original AIME diagnostic set changed")
    old_pilot = {r["checkpoint_id"] for r in original if r["phase"] == "1_operational_pilot"}
    expected_pairs = {(cp, policy) for cp, r in cp_by_id.items() for policy in GRID[r["benchmark"]]}
    expected_pairs |= {(cp, "AX03") for cp in old_diagnostics}
    require({(r["checkpoint_id"], r["generation_config_id"]) for r in combined} == expected_pairs,
            "Combined policy grid differs from the declared factorial/diagnostic allocation")
    sessions, groups = defaultdict(set), defaultdict(set)
    for cp, row in cp_by_id.items():
        source = inv_by_id[cp]
        require(row["trajectory_id"] == source["session"] and row["benchmark"] == source["benchmark"]
                and row["checkpoint_uri"] == source["checkpoint_uri"], f"Inventory binding mismatch: {cp}")
        require(source["eligible"] and not source.get("alias_targets")
                and not source.get("weight_alias_representative_if_explicitly_known"),
                f"Ineligible/alias checkpoint in expanded selection: {cp}")
        require(row["split"] == ("locked_session_test" if source["session"] in old_locked else "development"),
                f"Original session reservation changed: {cp}")
    for job in combined:
        cp = cp_by_id[job["checkpoint_id"]]
        policy, bench = job["generation_config_id"], job["benchmark"]
        require(set(job) == set(original[0]), "Extension row schema differs from original rows")
        for field in ("benchmark", "trajectory_id", "known_lineage_group", "split", "checkpoint_uri"):
            require(job[field] == cp[field], f"Checkpoint/job {field} disagreement: {job['exp_id']}")
        is_original = job["exp_id"] in old_by_id
        prefix = "poc1k" if is_original else "poc3k"
        require(job["exp_id"] == f"{prefix}-{job['checkpoint_id']}-{policy}", "Unexpected exp_id binding")
        require(type(job["n_passes"]) is int and job["n_passes"] == 10, "Every cell requires ten passes")
        require(type(job["n_questions_per_pass"]) is int and job["n_questions_per_pass"] ==
                (1319 if bench == "gsm8k" else 30), "Incorrect benchmark question count")
        require(job["protocol_path"] == "protocol.json", "Cell bypasses frozen seed/runtime protocol")
        require(job["release_status"] == "proposal_only_preflight_required", "Cell is not preflight-required")
        require(job["historical_reuse_status"] == "unverified_full_protocol_and_artifact_equivalence",
                "Historical reuse must remain unverified")
        require(type(job["historical_sampling_cap_match"]) is bool, "Invalid historical-match flag")
        require(type(job["primary_evaluation"]) is bool and
                job["primary_evaluation"] == (policy in BASE.CORE[bench]), "Expanded core-policy flag mismatch")
        sessions[job["trajectory_id"]].add(job["split"])
        groups[job["known_lineage_group"]].add(job["split"])
        for stem in ("generation_config", "request_template"):
            relative = f"configs/{policy}/{stem}.json"
            require(job[stem + "_path"] == relative, f"Unexpected {stem} path binding")
            require(job[stem + "_sha256"] == BASE.sha256(BASE.local_path(root, relative)),
                    f"Cell policy hash mismatch: {job['exp_id']}")
        if not is_original:
            expected_phase = ("7_extension_locked_test" if cp["split"] == "locked_session_test" else
                              "5_extension_pilot" if cp["checkpoint_id"] in old_pilot and policy in NEW_SETTINGS else
                              "6_extension_development")
            require(job["phase"] == expected_phase, f"Extension phase assignment mismatch: {job['exp_id']}")
    require(all(len(v) == 1 for v in sessions.values()), "Session mixed across splits")
    require(all(len(v) == 1 for v in groups.values()), "Learned-weight group mixed across splits")
    require(len(sessions) == 124, "Expanded matrix must retain the original 124 sessions")
    require({s for s, split in sessions.items() if split == {"locked_session_test"}} == old_locked,
            "Locked session membership changed")
    require(len({r["known_lineage_group"] for r in combined if r["split"] == "locked_session_test"}) == 40,
            "Expected 40 locked session/lineage groups")
    require(Counter(r["split"] for r in combined) == {"development": 1980, "locked_session_test": 1020},
            "Combined split allocation must be 1,980 development / 1,020 locked")
    require(sum(r["primary_evaluation"] for r in combined) == 1232, "Expected 1,232 shared-core cells")
    require(Counter(r["phase"] for r in extension) == NEW_PHASES, "Incorrect extension phase counts")
    for phase in NEW_PHASES:
        phase_rows = BASE.read_json(root, f"phases/{phase}.jsonl", lines=True)
        require(BASE.canonical_rows(phase_rows) == BASE.canonical_rows([r for r in extension if r["phase"] == phase]),
                f"Extension phase content/partition mismatch: {phase}")
    require(Counter(r["generation_config_id"] for r in extension if r["phase"] == "5_extension_pilot") ==
            {"G03": 20, "GX01": 20, "GX02": 20, "AX01": 12, "AX02": 12, "AX03": 12},
            "New pilot must cover six new policies on the original pilot checkpoints")
    for policy in NEW_SETTINGS:
        _validate_new_policy(root, policy)
    catalog = BASE.read_json(root, "generation_policies_all_13.json")
    original_catalog = BASE.read_json(root, "generation_policies.json")
    require(set(catalog) == set(original_catalog) | set(NEW_SETTINGS) and len(catalog) == 13,
            "Combined generation catalog must contain exactly thirteen policy IDs")
    require(all(catalog[key] == value for key, value in original_catalog.items()),
            "Original seven generation catalog entries changed")
    for policy, entry in catalog.items():
        gen = BASE.read_json(root, f"configs/{policy}/generation_config.json")
        params = {key: gen[key] for key in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")}
        params["max_tokens"] = gen["max_new_tokens"]
        require(entry["parameters"] == params and entry["stop_token_ids"] == gen["eos_token_id"],
                f"Generation catalog parameters/stops mismatch: {policy}")
        require(entry["benchmark"] == ("gsm8k" if policy.startswith("G") else "aime2025"),
                f"Generation catalog benchmark mismatch: {policy}")
        require(entry["config_sha256"] == BASE.sha256(root / f"configs/{policy}/generation_config.json")
                and entry["request_template_sha256"] == BASE.sha256(root / f"configs/{policy}/request_template.json"),
                f"Generation catalog file hash mismatch: {policy}")
    BASE.reject_outcomes(extension)
    BASE.reject_outcomes(checkpoints)
    BASE.reject_outcomes(catalog)
    summary = BASE.read_json(root, "extension_summary.json")
    require(summary["schema"] == "eval-matrix-extension-summary-v1"
            and summary["status"] == "proposal_not_execution_authorization", "Extension summary must remain a proposal")
    new_checkpoints = [r for r in checkpoints if r["checkpoint_id"] not in {x["checkpoint_id"] for x in original_cps}]
    summary_counts = {
        "n_original_cells": len(original), "n_extension_cells": len(extension),
        "n_total_cells": len(combined), "n_candidate_weight_sets": len(checkpoints),
        "counts_by_benchmark": dict(Counter(r["benchmark"] for r in combined)),
        "original_cells": len(original), "extension_cells": len(extension), "combined_cells": len(combined),
        "candidate_checkpoints": len(checkpoints),
        "candidate_checkpoints_by_benchmark": dict(Counter(r["benchmark"] for r in checkpoints)),
        "newly_added_checkpoints": len(new_checkpoints),
        "newly_added_checkpoints_by_benchmark": dict(Counter(r["benchmark"] for r in new_checkpoints)),
        "extension_cells_by_benchmark": dict(Counter(r["benchmark"] for r in extension)),
        "extension_cells_by_split": dict(Counter(r["split"] for r in extension)),
        "extension_cells_by_phase": dict(Counter(r["phase"] for r in extension)),
        "extension_cells_by_policy": dict(Counter(r["generation_config_id"] for r in extension)),
        "extension_primary_evaluation_cells": sum(r["primary_evaluation"] for r in extension),
        "combined_primary_evaluation_cells": sum(r["primary_evaluation"] for r in combined),
        "combined_cells_by_benchmark": dict(Counter(r["benchmark"] for r in combined)),
        "combined_cells_by_split": dict(Counter(r["split"] for r in combined)),
        "extension_full_benchmark_passes": sum(r["n_passes"] for r in extension),
        "combined_full_benchmark_passes": sum(r["n_passes"] for r in combined),
        "extension_question_completions": sum(r["n_passes"] * r["n_questions_per_pass"] for r in extension),
        "combined_question_completions": sum(r["n_passes"] * r["n_questions_per_pass"] for r in combined),
        "new_checkpoint_code_status": dict(Counter(inv_by_id[r["checkpoint_id"]]["code_status"] for r in new_checkpoints)),
        "new_checkpoint_entrypoint_available": {
            "true": sum(bool(inv_by_id[r["checkpoint_id"]]["entrypoint_available"]) for r in new_checkpoints),
            "false": sum(not bool(inv_by_id[r["checkpoint_id"]]["entrypoint_available"]) for r in new_checkpoints),
        },
    }
    for key, value in summary_counts.items():
        require(summary[key] == value, f"Extension summary count disagrees with matrix: {key}")
    require(summary["generation_policies_all_13_sha256"] == BASE.sha256(root / "generation_policies_all_13.json"),
            "Extension summary generation-catalog hash mismatch")
    require(summary["new_policy_payload_sha256"] == {
        policy: {stem: BASE.sha256(root / f"configs/{policy}/{stem}.json")
                 for stem in ("generation_config", "request_template")} for policy in NEW_SETTINGS
    }, "Extension summary policy-payload hashes mismatch")
    preserved = summary["preserved_original_sha256"]
    expected_preserved = set(ORIGINAL_SHA256) - {"checkpoint_inventory.json", "matrix_summary.json", "generation_policies.json"}
    require(expected_preserved <= set(preserved), "Extension summary omits preserved original byte digests")
    for relative, expected in preserved.items():
        require(BASE.sha256(BASE.local_path(root, relative)) == expected, f"Preserved original hash mismatch: {relative}")
        if relative in ORIGINAL_SHA256:
            require(expected == ORIGINAL_SHA256[relative], "Summary attempted to redefine original frozen bytes")
    plan = BASE.read_json(root, "execution_plan.json")
    require(plan["status"] == "proposal_not_execution_authorization", "Execution plan is not a proposal")
    require(plan["require_explicit_approval"] is True and plan["auto_advance"] is False
            and plan["automatic_launch"] is False
            and plan.get("auto_launch", False) is False, "Execution plan cannot auto-launch/advance or waive approval")
    require(plan["authoritative_combined_manifest"] == "experiment_matrix_all_3k.jsonl"
            and plan["extension_only_manifest"] == "experiment_matrix_extension_2k.jsonl", "Execution-plan manifest binding mismatch")
    require(plan["n_cells_total"] == 3000 and plan["n_extension_cells"] == 2000, "Execution-plan cell totals disagree")
    require(plan["n_original_cells"] == 1000 and plan["frozen_original_manifest"] == "experiment_matrix.jsonl"
            and plan["selected_checkpoints_manifest"] == "selected_checkpoints_all_516.jsonl"
            and plan["selected_ids_manifest"] == "selected_ids_516.json"
            and plan["generation_policy_catalog"] == "generation_policies_all_13.json",
            "Execution plan points to the wrong checkpoint/policy manifests")
    require(plan["required_stage_order"] == [s for s, _, _ in STAGES], "Execution stage-order declaration disagrees")
    require(len(plan["stages"]) == 3, "Expected three ordered execution stages")
    for actual, (stage, count, phases) in zip(plan["stages"], STAGES):
        require(actual["stage"] == stage and actual["n_cells"] == count
                and actual["phase_manifests"] == [f"phases/{p}.jsonl" for p in phases],
                "Execution stages must be ordered pilot196/development1784/locked1020")
        require(sum(r["phase"] in phases for r in combined) == count, "Execution stage disagrees with matrix rows")
    for flag, description in (("weight_shard_hashes_verified", "unverified weight-shard identity"),
                              ("tokenizer_template_compatibility_verified", "unverified tokenizer/template compatibility")):
        n = sum(r.get(flag) is not True for r in checkpoints)
        if n:
            warnings.append(f"{n} candidate checkpoints retain {description}.")
    n = sum(r.get("training_only_launch_bundle_status") != "frozen" for r in checkpoints)
    if n:
        warnings.append(f"{n} launch-only predictor input bundles are not frozen.")
    for flag in ("v6_missing_declared_code", "v6_content_review_required"):
        n = sum(bool(r.get(flag)) for r in checkpoints)
        if n:
            warnings.append(f"{n} candidate checkpoints retain {flag} flags.")
    warnings.append("Historical sampling/cap matches are not verified reuse or fresh counterfactual labels.")
    warnings.append("516 candidate IDs are not proof of 516 distinct weight tensors; locked sessions remain historically inspected.")
    return {"original_cells": 1000, "extension_cells": 2000, "combined_cells": 3000,
            "candidate_checkpoints": 516, "checkpoint_counts": {"gsm8k": 316, "aime2025": 200},
            "core_cells": 1232, "development_cells": 1980, "locked_cells": 1020,
            "locked_sessions": 40, "extension_phase_counts": NEW_PHASES,
            "execution_stage_counts": {s: n for s, n, _ in STAGES}, "hashed_assets": hashed_count,
            "policy_count": 13, "question_completions": summary_counts["combined_question_completions"]}


def verify_extension(bundle=DEFAULT_BUNDLE):
    warnings = [BASE.DISCLAIMER]
    try:
        counts = _verify(Path(bundle).resolve(), warnings)
    except (BASE.InvalidBundle, KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        return {"ok": False, "status": "static_extension_validation_failed", "errors": [str(exc)],
                "warnings": warnings, "counts": {}}
    return {"ok": True, "status": "static_extension_validation_passed_not_launch_ready", "errors": [],
            "warnings": warnings, "counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = verify_extension(args.bundle)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(report["status"])
        for error in report["errors"]:
            print("ERROR: " + error)
        if report["counts"]:
            print(json.dumps(report["counts"], sort_keys=True))
        for warning in report["warnings"]:
            print("WARNING: " + warning)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
