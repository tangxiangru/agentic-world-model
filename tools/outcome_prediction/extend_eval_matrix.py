"""Build the additive 2,000-cell extension to the frozen 1,000-cell matrix.

The builder is deterministic, performs no network/GPU work, and reads only the
tracked bundle supplied with ``--bundle``.  It writes only the additive files
listed in ``OWNED_OUTPUTS``; the original 1,000-cell artifacts are inputs and
are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "experiments/eval_matrix_1k"

ORIGINAL_POLICIES = {"G01", "G02", "A01", "A02", "A03", "A04", "A05"}
NEW_POLICIES = {
    "G03": {
        "benchmark": "gsm8k",
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "max_tokens": 4000,
        "stop_token_ids": [1, 106],
    },
    "GX01": {
        "benchmark": "gsm8k",
        "temperature": 0.4,
        "top_k": 64,
        "top_p": 0.95,
        "max_tokens": 4000,
        "stop_token_ids": [1, 106],
    },
    "GX02": {
        "benchmark": "gsm8k",
        "temperature": 0.4,
        "top_k": 0,
        "top_p": 1,
        "max_tokens": 4000,
        "stop_token_ids": [1, 106],
    },
    "AX01": {
        "benchmark": "aime2025",
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 1,
        "max_tokens": 16000,
        "stop_token_ids": [151643, 151645],
    },
    "AX02": {
        "benchmark": "aime2025",
        "temperature": 1,
        "top_k": 20,
        "top_p": 0.95,
        "max_tokens": 16000,
        "stop_token_ids": [151643, 151645],
    },
    "AX03": {
        "benchmark": "aime2025",
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "max_tokens": 4096,
        "stop_token_ids": [151643, 151645],
    },
}

GSM_GRID = ["G01", "G02", "G03", "GX01", "GX02"]
AIME_GRID = ["A01", "A02", "A03", "A04", "A05", "AX01", "AX02"]
PRIMARY_POLICIES = {"G01", "G02", "A01", "A02", "A03"}
NEW_PILOT_POLICIES = {
    "gsm8k": {"G03", "GX01", "GX02"},
    "aime2025": {"AX01", "AX02", "AX03"},
}

ORIGINAL_HASH_INPUTS = [
    "experiment_matrix.jsonl",
    "selected_checkpoints.jsonl",
    "selected_ids_400.json",
    "protocol.json",
    "splits.json",
    "checkpoint_inventory.json",
    "config_audit.json",
    "matrix_summary.json",
    "generation_policies.json",
    "source_pins.json",
    "phases/1_operational_pilot.jsonl",
    "phases/2_development_core.jsonl",
    "phases/3_diagnostic_extensions.jsonl",
    "phases/4_locked_test.jsonl",
    *[
        f"configs/{policy_id}/{filename}"
        for policy_id in sorted(ORIGINAL_POLICIES)
        for filename in ("generation_config.json", "request_template.json")
    ],
]

OWNED_OUTPUTS = {
    "experiment_matrix_extension_2k.jsonl",
    "experiment_matrix_all_3k.jsonl",
    "selected_checkpoints_all_516.jsonl",
    "selected_ids_516.json",
    "generation_policies_all_13.json",
    "extension_summary.json",
    "execution_plan.json",
    "phases/5_extension_pilot.jsonl",
    "phases/6_extension_development.jsonl",
    "phases/7_extension_locked_test.jsonl",
    *{
        f"configs/{policy_id}/{filename}"
        for policy_id in NEW_POLICIES
        for filename in ("generation_config.json", "request_template.json")
    },
}

FORBIDDEN_OUTPUT_KEYS = {
    "accuracy",
    "avg_pass_rate",
    "correctness",
    "label",
    "outcome",
    "per_run_pass_rate",
    "scientist",
    "score",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha256(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()


def encode_jsonl(values: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(value, sort_keys=True) + "\n" for value in values).encode()


class OutputWriter:
    """Constrain every write to a declared additive, non-symlinked target."""

    def __init__(
        self,
        output_dir: Path,
        bundle: Path,
        *,
        allow_overwrite: bool,
    ) -> None:
        self.output_dir = output_dir.absolute()
        self.bundle = bundle.resolve(strict=True)
        self.allow_overwrite = allow_overwrite
        self.original_paths = [(self.bundle / relative).resolve(strict=True) for relative in ORIGINAL_HASH_INPUTS]
        self._preflight_tree()

    def _preflight_tree(self) -> None:
        if self.output_dir.is_symlink():
            raise ValueError(f"Output directory may not be a symlink: {self.output_dir}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {self.output_dir}")
        if not self.output_dir.exists() or self.output_dir.resolve() == self.bundle:
            return
        allowed_dirs = {
            str(parent)
            for relative in OWNED_OUTPUTS
            for parent in Path(relative).parents
            if str(parent) != "."
        }
        for path in self.output_dir.rglob("*"):
            relative = path.relative_to(self.output_dir).as_posix()
            if path.is_symlink():
                raise ValueError(f"Separate output tree contains a symlink: {relative}")
            if path.is_file() and relative not in OWNED_OUTPUTS:
                raise ValueError(f"Separate output tree contains an undeclared file: {relative}")
            if path.is_dir() and relative not in allowed_dirs:
                raise ValueError(f"Separate output tree contains an undeclared directory: {relative}")

    def _target(self, relative: str) -> Path:
        if relative not in OWNED_OUTPUTS:
            raise ValueError(f"Refusing undeclared output: {relative}")
        target = self.output_dir / relative
        current = self.output_dir
        if current.exists() and current.is_symlink():
            raise ValueError(f"Output directory may not be a symlink: {current}")
        for part in Path(relative).parts[:-1]:
            current /= part
            if current.exists() and current.is_symlink():
                raise ValueError(f"Output parent may not be a symlink: {current}")
        if target.is_symlink():
            raise ValueError(f"Output target may not be a symlink: {target}")
        resolved_root = self.output_dir.resolve(strict=False)
        resolved_target = target.resolve(strict=False)
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(f"Output target escapes output directory: {target}")
        if target.exists():
            for original in self.original_paths:
                if os.path.samefile(target, original):
                    raise ValueError(f"Output target aliases frozen input: {target}")
        return target

    def write_bytes(self, relative: str, content: bytes) -> None:
        target = self._target(relative)
        if target.exists():
            if target.read_bytes() == content:
                return
            if not self.allow_overwrite:
                raise ValueError(
                    f"Refusing to replace differing generated output without --overwrite: {relative}"
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after mkdir so a pre-existing internal symlink cannot escape.
        target = self._target(relative)
        target.write_bytes(content)

    def write_json(self, relative: str, value: Any) -> None:
        self.write_bytes(relative, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())

    def write_jsonl(self, relative: str, values: list[dict[str, Any]]) -> None:
        self.write_bytes(relative, encode_jsonl(values))


def ensure_no_outcomes(value: Any) -> None:
    if isinstance(value, dict):
        bad = set(value) & FORBIDDEN_OUTPUT_KEYS
        if bad:
            raise ValueError(f"Outcome-bearing fields leaked into extension output: {sorted(bad)}")
        for item in value.values():
            ensure_no_outcomes(item)
    elif isinstance(value, list):
        for item in value:
            ensure_no_outcomes(item)


def new_generation_config(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": policy["temperature"] > 0,
        "eos_token_id": policy["stop_token_ids"],
        "max_new_tokens": policy["max_tokens"],
        "min_p": 0,
        "repetition_penalty": 1,
        "temperature": policy["temperature"],
        "top_k": policy["top_k"],
        "top_p": policy["top_p"],
    }


def new_request_template(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "extra_body": {
            "ignore_eos": False,
            "min_p": 0,
            "min_tokens": 0,
            "repetition_penalty": 1,
            "stop_token_ids": policy["stop_token_ids"],
            "top_k": policy["top_k"],
        },
        "frequency_penalty": 0,
        "max_tokens": policy["max_tokens"],
        "n": 1,
        "presence_penalty": 0,
        "stop": [],
        "temperature": policy["temperature"],
        "top_p": policy["top_p"],
    }


def settings_tuple(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (
        settings["max_tokens"],
        settings.get("min_p", 0),
        settings.get("repetition_penalty", 1),
        settings["temperature"],
        settings["top_k"],
        settings["top_p"],
    )


def policy_settings(bundle: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    original = read_json(bundle / "generation_policies.json")
    for policy_id, policy in original.items():
        result[policy_id] = dict(policy["parameters"])
    for policy_id, policy in NEW_POLICIES.items():
        result[policy_id] = {
            "max_tokens": policy["max_tokens"],
            "min_p": 0,
            "repetition_penalty": 1,
            "temperature": policy["temperature"],
            "top_k": policy["top_k"],
            "top_p": policy["top_p"],
        }
    return result


def historical_settings_by_checkpoint(bundle: Path) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    audit = read_json(bundle / "config_audit.json")
    for benchmark in audit["benchmarks"].values():
        for policy in benchmark["policies"]:
            normalized = settings_tuple(policy["settings"])
            for checkpoint_id in policy["source_exp_ids"]:
                if checkpoint_id in result and result[checkpoint_id] != normalized:
                    raise ValueError(f"Conflicting historical config mappings for {checkpoint_id}")
                result[checkpoint_id] = normalized
    return result


def checkpoint_from_inventory(
    row: dict[str, Any], locked_sessions: dict[str, set[str]]
) -> dict[str, Any]:
    benchmark = row["benchmark"]
    split = "locked_session_test" if row["session"] in locked_sessions[benchmark] else "development"
    return {
        "benchmark": benchmark,
        "card_family": row["raw_recipe_family"],
        "checkpoint_id": row["exp_id"],
        "checkpoint_uri": row["checkpoint_uri"],
        "declared_training_script": row["exact_declared_script"],
        "historical_plan_script_status": row["code_status"],
        "known_lineage_group": row["split_group_primary"],
        "lora": 1.0 if row["lora"] else 0.0,
        "parent_checkpoint_ids": row["declared_parent_ids"],
        "parent_kind": row["parent_kind"],
        "pilot": False,
        "selection_basis": "outcome_free_remaining_candidate_weight_coverage",
        "split": split,
        "tokenizer_template_compatibility_verified": False,
        "training_only_launch_bundle_status": "not_frozen",
        "trajectory_id": row["session"],
        "v6_content_review_required": row["v6_content_review_required"],
        "v6_missing_declared_code": row["v6_missing_declared_code"],
        "weight_shard_hashes_verified": False,
    }


def build(bundle: Path, output_dir: Path, *, allow_overwrite: bool = False) -> dict[str, Any]:
    bundle = bundle.resolve(strict=True)
    output_dir = output_dir.absolute()
    required = [
        "experiment_matrix.jsonl",
        "selected_checkpoints.jsonl",
        "selected_ids_400.json",
        "checkpoint_inventory.json",
        "config_audit.json",
        "generation_policies.json",
        "splits.json",
    ]
    missing = [relative for relative in required if not (bundle / relative).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing required tracked inputs: {missing}")

    writer = OutputWriter(output_dir, bundle, allow_overwrite=allow_overwrite)

    original_hashes = {relative: sha256(bundle / relative) for relative in ORIGINAL_HASH_INPUTS}
    original_matrix_bytes = (bundle / "experiment_matrix.jsonl").read_bytes()
    original_checkpoint_bytes = (bundle / "selected_checkpoints.jsonl").read_bytes()
    original_jobs = read_jsonl(bundle / "experiment_matrix.jsonl")
    original_checkpoints = read_jsonl(bundle / "selected_checkpoints.jsonl")
    original_ids = read_json(bundle / "selected_ids_400.json")
    inventory = read_json(bundle / "checkpoint_inventory.json")
    splits = read_json(bundle / "splits.json")

    if len(original_jobs) != 1000 or len({row["exp_id"] for row in original_jobs}) != 1000:
        raise ValueError("Frozen original matrix must contain 1,000 unique exp_ids")
    if len(original_checkpoints) != 400 or len(original_ids) != 400:
        raise ValueError("Frozen original selection must contain 400 checkpoints")
    if [row["checkpoint_id"] for row in original_checkpoints] != original_ids:
        raise ValueError("Frozen checkpoint and ID manifests disagree in order")
    original_checkpoint_ids = set(original_ids)
    inventory_rows = inventory["checkpoints"]
    candidates = [row for row in inventory_rows if row["candidate_weight_bool"]]
    if len(candidates) != 516 or Counter(row["benchmark"] for row in candidates) != {
        "gsm8k": 316,
        "aime2025": 200,
    }:
        raise ValueError("Expected the audited 516-candidate checkpoint cohort")
    candidate_ids = {row["exp_id"] for row in candidates}
    if not original_checkpoint_ids <= candidate_ids:
        raise ValueError("Frozen selection is not a subset of candidate checkpoints")
    remaining = sorted(
        (row for row in candidates if row["exp_id"] not in original_checkpoint_ids),
        key=lambda row: row["exp_id"],
    )
    if len(remaining) != 116 or Counter(row["benchmark"] for row in remaining) != {
        "gsm8k": 76,
        "aime2025": 40,
    }:
        raise ValueError("Expected 116 remaining candidates (76 GSM8K, 40 AIME)")

    locked_sessions = {
        benchmark: set(details["locked_sessions"])
        for benchmark, details in splits.items()
    }
    all_checkpoints = original_checkpoints + [
        checkpoint_from_inventory(row, locked_sessions) for row in remaining
    ]
    all_ids = original_ids + [row["exp_id"] for row in remaining]
    if len(all_ids) != 516 or len(all_ids) != len(set(all_ids)):
        raise ValueError("Combined checkpoint IDs are not 516 unique values")
    by_checkpoint = {row["checkpoint_id"]: row for row in all_checkpoints}

    original_pairs = {
        (row["checkpoint_id"], row["generation_config_id"]) for row in original_jobs
    }
    original_exp_ids = {row["exp_id"] for row in original_jobs}
    diagnostic_a04 = {
        row["checkpoint_id"] for row in original_jobs if row["generation_config_id"] == "A04"
    }
    diagnostic_a05 = {
        row["checkpoint_id"] for row in original_jobs if row["generation_config_id"] == "A05"
    }
    if diagnostic_a04 != diagnostic_a05 or len(diagnostic_a04) != 20:
        raise ValueError("AX03 requires the frozen shared 20-checkpoint A04/A05 diagnostic set")
    pilot_ids = {row["checkpoint_id"] for row in original_checkpoints if row["pilot"]}
    if Counter(by_checkpoint[checkpoint_id]["benchmark"] for checkpoint_id in pilot_ids) != {
        "gsm8k": 20,
        "aime2025": 12,
    }:
        raise ValueError("Expected the frozen 20-GSM/12-AIME pilot checkpoint set")
    if not {
        checkpoint_id
        for checkpoint_id in pilot_ids
        if by_checkpoint[checkpoint_id]["benchmark"] == "aime2025"
    } <= diagnostic_a04:
        raise ValueError("Frozen AIME pilot must be contained in the diagnostic set")

    output_dir.mkdir(parents=True, exist_ok=True)
    for policy_id, policy in sorted(NEW_POLICIES.items()):
        writer.write_json(f"configs/{policy_id}/generation_config.json", new_generation_config(policy))
        writer.write_json(f"configs/{policy_id}/request_template.json", new_request_template(policy))

    config_hashes: dict[str, str] = {}
    request_hashes: dict[str, str] = {}
    for policy_id in sorted(ORIGINAL_POLICIES | set(NEW_POLICIES)):
        source_root = output_dir if policy_id in NEW_POLICIES else bundle
        config_hashes[policy_id] = sha256(source_root / f"configs/{policy_id}/generation_config.json")
        request_hashes[policy_id] = sha256(source_root / f"configs/{policy_id}/request_template.json")

    desired_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        checkpoint_id = candidate["exp_id"]
        policy_ids = GSM_GRID if candidate["benchmark"] == "gsm8k" else AIME_GRID
        desired_pairs.update((checkpoint_id, policy_id) for policy_id in policy_ids)
    desired_pairs.update((checkpoint_id, "AX03") for checkpoint_id in diagnostic_a04)
    extension_pairs = desired_pairs - original_pairs
    if len(desired_pairs) != 3000 or len(extension_pairs) != 2000:
        raise ValueError("Extension grid arithmetic must be 3,000 total / 2,000 additive")
    if original_pairs - desired_pairs:
        raise ValueError("Frozen original matrix is not a subset of the combined grid")

    settings = policy_settings(bundle)
    original_policy_catalog = read_json(bundle / "generation_policies.json")
    if set(original_policy_catalog) != ORIGINAL_POLICIES:
        raise ValueError("Frozen generation_policies.json must contain the original seven policies")
    audit = read_json(bundle / "config_audit.json")
    support_by_settings: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    for benchmark_name, benchmark in audit["benchmarks"].items():
        for policy in benchmark["policies"]:
            support_by_settings[(benchmark_name, settings_tuple(policy["settings"]))] = policy
    combined_policy_catalog = json.loads(json.dumps(original_policy_catalog))
    for policy_id, policy in sorted(NEW_POLICIES.items()):
        support = support_by_settings.get((policy["benchmark"], settings_tuple(settings[policy_id])))
        combined_policy_catalog[policy_id] = {
            "benchmark": policy["benchmark"],
            "config_sha256": config_hashes[policy_id],
            "corpus_records_with_sampling_and_cap_policy": support["n"] if support else 0,
            "parameters": settings[policy_id],
            "request_template_sha256": request_hashes[policy_id],
            "source_representative_exp_id": support["representative"]["exp_id"] if support else None,
            "stop_token_ids": policy["stop_token_ids"],
        }
    if any(combined_policy_catalog[policy_id] != original_policy_catalog[policy_id] for policy_id in ORIGINAL_POLICIES):
        raise ValueError("Original policy entries changed in the combined catalog")
    writer.write_json("generation_policies_all_13.json", combined_policy_catalog)

    historical = historical_settings_by_checkpoint(bundle)
    extension_jobs: list[dict[str, Any]] = []
    for checkpoint_id, policy_id in extension_pairs:
        checkpoint = by_checkpoint[checkpoint_id]
        benchmark = checkpoint["benchmark"]
        if policy_id in NEW_PILOT_POLICIES[benchmark] and checkpoint_id in pilot_ids:
            phase = "5_extension_pilot"
        elif checkpoint["split"] == "locked_session_test":
            phase = "7_extension_locked_test"
        else:
            phase = "6_extension_development"
        row = {
            "benchmark": benchmark,
            "checkpoint_id": checkpoint_id,
            "checkpoint_uri": checkpoint["checkpoint_uri"],
            "exp_id": f"poc3k-{checkpoint_id}-{policy_id}",
            "generation_config_id": policy_id,
            "generation_config_path": f"configs/{policy_id}/generation_config.json",
            "generation_config_sha256": config_hashes[policy_id],
            "historical_reuse_status": "unverified_full_protocol_and_artifact_equivalence",
            "historical_sampling_cap_match": historical.get(checkpoint_id) == settings_tuple(settings[policy_id]),
            "known_lineage_group": checkpoint["known_lineage_group"],
            "n_passes": 10,
            "n_questions_per_pass": 1319 if benchmark == "gsm8k" else 30,
            "phase": phase,
            "primary_evaluation": policy_id in PRIMARY_POLICIES,
            "protocol_path": "protocol.json",
            "release_status": "proposal_only_preflight_required",
            "request_template_path": f"configs/{policy_id}/request_template.json",
            "request_template_sha256": request_hashes[policy_id],
            "split": checkpoint["split"],
            "trajectory_id": checkpoint["trajectory_id"],
        }
        if set(row) != set(original_jobs[0]):
            raise ValueError("Extension row schema differs from frozen matrix row schema")
        extension_jobs.append(row)
    extension_jobs.sort(
        key=lambda row: (
            row["phase"],
            row["benchmark"],
            row["checkpoint_id"],
            row["generation_config_id"],
        )
    )
    if len({row["exp_id"] for row in extension_jobs}) != 2000:
        raise ValueError("Extension exp_ids are not unique")
    if original_exp_ids & {row["exp_id"] for row in extension_jobs}:
        raise ValueError("Extension exp_ids collide with frozen original exp_ids")
    ensure_no_outcomes(extension_jobs)
    ensure_no_outcomes(all_checkpoints)

    phase_counts = Counter(row["phase"] for row in extension_jobs)
    if phase_counts != {
        "5_extension_pilot": 96,
        "6_extension_development": 1171,
        "7_extension_locked_test": 733,
    }:
        raise ValueError(f"Unexpected extension phase counts: {phase_counts}")
    split_counts = Counter(row["split"] for row in extension_jobs)
    if split_counts != {"development": 1267, "locked_session_test": 733}:
        raise ValueError(f"Unexpected extension split counts: {split_counts}")
    benchmark_counts = Counter(row["benchmark"] for row in extension_jobs)
    if benchmark_counts != {"gsm8k": 1100, "aime2025": 900}:
        raise ValueError(f"Unexpected extension benchmark counts: {benchmark_counts}")

    extension_bytes = encode_jsonl(extension_jobs)
    writer.write_bytes("experiment_matrix_extension_2k.jsonl", extension_bytes)
    writer.write_bytes("experiment_matrix_all_3k.jsonl", original_matrix_bytes + extension_bytes)
    writer.write_bytes(
        "selected_checkpoints_all_516.jsonl",
        original_checkpoint_bytes + encode_jsonl(all_checkpoints[400:])
    )
    writer.write_json("selected_ids_516.json", all_ids)
    for phase in ("5_extension_pilot", "6_extension_development", "7_extension_locked_test"):
        writer.write_jsonl(
            f"phases/{phase}.jsonl",
            [row for row in extension_jobs if row["phase"] == phase],
        )

    combined_jobs = original_jobs + extension_jobs
    combined_split_counts = Counter(row["split"] for row in combined_jobs)
    combined_benchmark_counts = Counter(row["benchmark"] for row in combined_jobs)
    remaining_status = Counter(row["code_status"] for row in remaining)
    remaining_entrypoint = Counter(bool(row["entrypoint_available"]) for row in remaining)
    summary = {
        "schema": "eval-matrix-extension-summary-v1",
        "status": "proposal_not_execution_authorization",
        "n_original_cells": 1000,
        "n_extension_cells": 2000,
        "n_total_cells": 3000,
        "n_candidate_weight_sets": 516,
        "counts_by_benchmark": {"aime2025": 1420, "gsm8k": 1580},
        "original_cells": 1000,
        "extension_cells": 2000,
        "combined_cells": 3000,
        "candidate_checkpoints": 516,
        "candidate_checkpoints_by_benchmark": {"aime2025": 200, "gsm8k": 316},
        "newly_added_checkpoints": 116,
        "newly_added_checkpoints_by_benchmark": {"aime2025": 40, "gsm8k": 76},
        "extension_cells_by_benchmark": dict(sorted(benchmark_counts.items())),
        "extension_cells_by_split": dict(sorted(split_counts.items())),
        "extension_cells_by_phase": dict(sorted(phase_counts.items())),
        "extension_cells_by_policy": dict(sorted(Counter(row["generation_config_id"] for row in extension_jobs).items())),
        "extension_primary_evaluation_cells": sum(row["primary_evaluation"] for row in extension_jobs),
        "combined_primary_evaluation_cells": sum(row["primary_evaluation"] for row in combined_jobs),
        "combined_cells_by_benchmark": dict(sorted(combined_benchmark_counts.items())),
        "combined_cells_by_split": dict(sorted(combined_split_counts.items())),
        "extension_full_benchmark_passes": 20000,
        "combined_full_benchmark_passes": 30000,
        "extension_question_completions": sum(
            row["n_passes"] * row["n_questions_per_pass"] for row in extension_jobs
        ),
        "combined_question_completions": sum(
            row["n_passes"] * row["n_questions_per_pass"] for row in combined_jobs
        ),
        "new_checkpoint_code_status": dict(sorted(remaining_status.items())),
        "new_checkpoint_entrypoint_available": {
            "false": remaining_entrypoint[False],
            "true": remaining_entrypoint[True],
        },
        "selected_ids_516_semantic_sha256": semantic_sha256(all_ids),
        "preserved_original_sha256": original_hashes,
        "new_policy_payload_sha256": {
            policy_id: {
                "generation_config": config_hashes[policy_id],
                "request_template": request_hashes[policy_id],
            }
            for policy_id in sorted(NEW_POLICIES)
        },
        "generation_policies_all_13_sha256": sha256(output_dir / "generation_policies_all_13.json"),
        "safety_checks": {
            "all_516_candidate_checkpoint_proxies_retained": True,
            "combined_manifest_has_exact_original_byte_prefix": True,
            "extension_contains_no_outcome_fields": True,
            "no_eval_jobs_launched": True,
            "original_1000_assets_not_rewritten": True,
            "whole_session_splits_inherited": True,
        },
        "limitations": [
            "The 516 candidate checkpoints are card-derived weight proxies, not tensor-hash-verified unique weights.",
            "Every cell remains gated on artifact loadability, weight/tokenizer hashes, predictor-input review, and resolved runtime-policy assertions.",
            "The 116 added checkpoints extend continuations and merges within existing sessions; they do not add new scientist sessions.",
        ],
    }
    if summary["extension_primary_evaluation_cells"] != 272 or summary["combined_primary_evaluation_cells"] != 1232:
        raise ValueError("Primary-evaluation cell arithmetic must be 272 extension / 1,232 combined")
    if summary["extension_question_completions"] != 14_779_000 or summary["combined_question_completions"] != 21_266_200:
        raise ValueError("Question-completion arithmetic mismatch")
    ensure_no_outcomes(summary)
    writer.write_json("extension_summary.json", summary)

    execution_plan = {
        "schema": "eval-matrix-execution-plan-v1",
        "status": "proposal_not_execution_authorization",
        "automatic_launch": False,
        "require_explicit_approval": True,
        "auto_advance": False,
        "n_cells_total": 3000,
        "n_original_cells": 1000,
        "n_extension_cells": 2000,
        "authoritative_combined_manifest": "experiment_matrix_all_3k.jsonl",
        "extension_only_manifest": "experiment_matrix_extension_2k.jsonl",
        "frozen_original_manifest": "experiment_matrix.jsonl",
        "selected_checkpoints_manifest": "selected_checkpoints_all_516.jsonl",
        "selected_ids_manifest": "selected_ids_516.json",
        "generation_policy_catalog": "generation_policies_all_13.json",
        "stages": [
            {
                "stage": "pilot",
                "n_cells": 196,
                "phase_manifests": [
                    "phases/1_operational_pilot.jsonl",
                    "phases/5_extension_pilot.jsonl",
                ],
            },
            {
                "stage": "development",
                "n_cells": 1784,
                "phase_manifests": [
                    "phases/2_development_core.jsonl",
                    "phases/3_diagnostic_extensions.jsonl",
                    "phases/6_extension_development.jsonl",
                ],
            },
            {
                "stage": "locked_test",
                "n_cells": 1020,
                "phase_manifests": [
                    "phases/4_locked_test.jsonl",
                    "phases/7_extension_locked_test.jsonl",
                ],
            },
        ],
        "required_stage_order": ["pilot", "development", "locked_test"],
        "release_gate": "Each stage requires explicit human authorization after the preceding stage passes the frozen runtime, artifact, completeness, and leakage checks.",
        "historical_reuse_policy": "Do not skip a proposed cell unless full artifact and protocol equivalence has been verified; sampling/cap similarity alone is insufficient.",
    }
    ensure_no_outcomes(execution_plan)
    writer.write_json("execution_plan.json", execution_plan)

    # Safe-overwrite and preservation assertions.  In a separate output tree,
    # every emitted file must still be one of the declared additive artifacts.
    emitted = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if output_dir != bundle and emitted != OWNED_OUTPUTS:
        raise ValueError(f"Unexpected additive outputs: {sorted(emitted ^ OWNED_OUTPUTS)}")
    if output_dir.resolve() == bundle:
        after_hashes = {relative: sha256(bundle / relative) for relative in ORIGINAL_HASH_INPUTS}
        if after_hashes != original_hashes:
            raise RuntimeError("A frozen original artifact changed during extension build")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Tracked frozen 1,000-cell bundle used as the only input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for additive outputs; defaults to --bundle",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace differing declared additive outputs; frozen original inputs remain protected",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.bundle
    summary = build(args.bundle, output_dir, allow_overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
