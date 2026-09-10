"""Select the frozen 400 checkpoint candidates for the 1,000-cell PoC.

This is a read-only, outcome-free selector. It prints a JSON array of checkpoint
IDs by default and never launches evaluation. Historical result values and the
inventory's ``selection_400`` field are neither loaded into the selection rows
nor consulted.

The selector preserves the implementation that produced
``eval_matrix_1k_design/selected_ids_400.json``:

* retain every eligible from-base checkpoint;
* allocate 50 GSM8K and 37 AIME continuation/merge checkpoints;
* force coverage of sessions without a from-base checkpoint;
* retain one best-available RL/distillation checkpoint per session;
* maximize continuation-session coverage using code-readiness evidence; and
* fill residual slots by method, operation, LoRA, and data-source novelty.

``candidate_weight`` remains a card/trajectory proxy, not tensor-hash proof.
The selected artifacts therefore still require weight and loadability preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/analysis/wm_exp_designs"
DEFAULT_TABLE = ANALYSIS / "table_v2/experiments.jsonl"
DEFAULT_INPUTS = ANALYSIS / "prefix_recipes_v6/inputs.jsonl"
DEFAULT_LABELS = ANALYSIS / "prefix_recipes_v6/labels.jsonl"
DEFAULT_FROZEN_IDS = ANALYSIS / "eval_matrix_1k_design/selected_ids_400.json"

CONTINUATION_QUOTAS = {"gsm8k": 50, "aime2025": 37}
CANDIDATE_KINDS = {
    "base_training_output",
    "continued_training_output",
    "parameter_merge_output",
}

# table_v2's broad semantic family is not always the weight operation. These
# finite corrections come directly from the recorded argv/card evidence audited
# in checkpoint_audit.md. They contain no result values.
MERGE_DISGUISED_AS_SFT = {
    "r0-02-exp-05",
    "r0-12-exp-05",
    "r0-12-exp-07",
    "r0-30-exp-07",
    "r0-30-exp-08",
    "aime-r0-30-exp-04",
}
EVAL_DISGUISED_AS_SFT = {
    "aime-r0-04-exp-05",
    "aime-r0-30-exp-06",
}
REEVAL_DISGUISED_AS_MERGE = {
    "r0-06-exp-09",
    "r0-08-exp-07",
    "r0-28-exp-09",
    "aime-r0-18-exp-06",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _eligible_ids(path: Path) -> set[str]:
    """Retain only the eligibility bit; do not expose result values downstream."""
    eligible: set[str] = set()
    for record in _read_jsonl(path):
        if record.get("output", {}).get("eligible_for_training") is True:
            eligible.add(record["exp_id"])
    return eligible


def _selection_rows(path: Path, eligible: set[str]) -> dict[str, dict[str, Any]]:
    """Project table_v2 onto outcome-free fields used by the selector."""
    projected: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        exp_id = record["example_id"]
        if exp_id not in eligible:
            continue
        recipe = record.get("recipe") or {}
        code = record.get("code") or {}
        projected[exp_id] = {
            "example_id": exp_id,
            "cell": record["cell"],
            "benchmark": record["benchmark"],
            "parent": {"parent_kind": (record.get("parent") or {}).get("parent_kind")},
            "recipe": {key: value for key, value in recipe.items()
                       if key in {"family", "lora"} or key.startswith("src_")},
            "code": {key: code.get(key) for key in ("training_status", "snapshot_matches_plan_code")},
            "codecfg": {"codecfg.entrypoint_available":
                        (record.get("codecfg") or {}).get("codecfg.entrypoint_available")},
        }
    return projected


def _quality_rows(path: Path, eligible: set[str]) -> dict[str, dict[str, Any]]:
    """Project prefix inputs onto their outcome-free extraction-quality flags."""
    projected: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record["exp_id"] in eligible:
            projected[record["exp_id"]] = record.get("quality") or {}
    return projected


def weight_change_kind(exp_id: str, rows: dict[str, dict[str, Any]]) -> str:
    row = rows[exp_id]
    family = row["recipe"].get("family")
    parent_kind = row["parent"].get("parent_kind")
    if parent_kind == "base":
        return "base_training_output"
    if exp_id in MERGE_DISGUISED_AS_SFT:
        return "parameter_merge_output"
    if exp_id in EVAL_DISGUISED_AS_SFT or exp_id in REEVAL_DISGUISED_AS_MERGE:
        return "same_or_selected_weight_bundle"
    if family in {"sft", "rft", "rl", "distill"}:
        return "continued_training_output"
    if family == "merge":
        return "parameter_merge_output"
    if family in {"decoding", "other"}:
        return "same_or_selected_weight_bundle"
    return "uncertain"


def _source_signature(exp_id: str, rows: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    recipe = rows[exp_id]["recipe"]
    return tuple(
        sorted(
            key.removeprefix("src_")
            for key, value in recipe.items()
            if key.startswith("src_")
            and isinstance(value, (int, float))
            and value > 0
        )
    )


def _quality_score(
    exp_id: str,
    rows: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, Any]],
) -> int:
    """Outcome-free readiness score used only to order otherwise valid rows."""
    row = rows[exp_id]
    recipe = row["recipe"]
    code = row["code"]
    flags = quality[exp_id]
    status_score = {
        "reconstructed": 30,
        "unavailable": 8,
        "blocked": 0,
    }.get(code.get("training_status"), 0)
    method_score = {
        "rl": 25,
        "distill": 25,
        "rft": 16,
        "merge": 12,
        "sft": 5,
        "other": 0,
    }.get(recipe.get("family"), 0)
    return (
        status_score
        + 10 * (row["codecfg"].get("codecfg.entrypoint_available") == 1)
        + 7 * (code.get("snapshot_matches_plan_code") is True)
        + 4 * (not flags.get("content_review_required"))
        + 3 * ((recipe.get("lora") or 0) == 1)
        + method_score
    )


def select(
    rows: dict[str, dict[str, Any]],
    quality: dict[str, dict[str, Any]],
) -> list[str]:
    if rows.keys() != quality.keys():
        missing_quality = sorted(rows.keys() - quality.keys())
        missing_rows = sorted(quality.keys() - rows.keys())
        raise ValueError(
            "Table/input eligibility mismatch: "
            f"missing_quality={missing_quality}, missing_rows={missing_rows}"
        )

    base_core = {
        exp_id
        for exp_id in rows
        if weight_change_kind(exp_id, rows) == "base_training_output"
    }
    selected = set(base_core)
    base_sessions = {rows[exp_id]["cell"] for exp_id in base_core}

    for benchmark, quota in CONTINUATION_QUOTAS.items():
        pool = [
            exp_id
            for exp_id in rows
            if rows[exp_id]["benchmark"] == benchmark
            and weight_change_kind(exp_id, rows)
            in {"continued_training_output", "parameter_merge_output"}
        ]
        pool_sessions = sorted({rows[exp_id]["cell"] for exp_id in pool})
        chosen: set[str] = set()

        # Guarantee overall source-session coverage when no base row exists.
        for session in pool_sessions:
            if session in base_sessions:
                continue
            candidates = [exp_id for exp_id in pool if rows[exp_id]["cell"] == session]
            chosen.add(
                max(
                    candidates,
                    key=lambda exp_id: (_quality_score(exp_id, rows, quality), exp_id),
                )
            )

        # Cover rare learned-weight operations without retaining every correlated
        # checkpoint from the same trajectory.
        for session in pool_sessions:
            for family in ("distill", "rl"):
                candidates = [
                    exp_id
                    for exp_id in pool
                    if rows[exp_id]["cell"] == session
                    and rows[exp_id]["recipe"].get("family") == family
                ]
                if candidates:
                    chosen.add(
                        max(
                            candidates,
                            key=lambda exp_id: (
                                _quality_score(exp_id, rows, quality),
                                exp_id,
                            ),
                        )
                    )

        # Add one best checkpoint from as many remaining continuation sessions as
        # the quota permits.
        represented = {rows[exp_id]["cell"] for exp_id in chosen}
        representatives: list[str] = []
        for session in pool_sessions:
            if session in represented:
                continue
            candidates = [exp_id for exp_id in pool if rows[exp_id]["cell"] == session]
            representatives.append(
                max(
                    candidates,
                    key=lambda exp_id: (_quality_score(exp_id, rows, quality), exp_id),
                )
            )
        representatives.sort(
            key=lambda exp_id: (
                _quality_score(exp_id, rows, quality),
                len(_source_signature(exp_id, rows)),
                exp_id,
            ),
            reverse=True,
        )
        for exp_id in representatives:
            if len(chosen) >= quota:
                break
            chosen.add(exp_id)

        # GSM has five residual slots after one-per-session coverage. Allocate
        # them by marginal categorical novelty plus the same readiness score.
        while len(chosen) < quota:
            family_counts = Counter(rows[x]["recipe"].get("family") for x in chosen)
            kind_counts = Counter(weight_change_kind(x, rows) for x in chosen)
            lora_counts = Counter(rows[x]["recipe"].get("lora") for x in chosen)
            data_counts = Counter(_source_signature(x, rows) for x in chosen)

            def marginal_score(exp_id: str) -> tuple[float, str]:
                recipe = rows[exp_id]["recipe"]
                score = (
                    50 / (1 + family_counts[recipe.get("family")])
                    + 30 / (1 + kind_counts[weight_change_kind(exp_id, rows)])
                    + 20 / (1 + lora_counts[recipe.get("lora")])
                    + 20 / (1 + data_counts[_source_signature(exp_id, rows)])
                    + _quality_score(exp_id, rows, quality)
                )
                return score, exp_id

            remaining = [exp_id for exp_id in pool if exp_id not in chosen]
            if not remaining:
                raise ValueError(f"Insufficient {benchmark} candidates for quota {quota}")
            chosen.add(max(remaining, key=marginal_score))

        if len(chosen) != quota:
            raise AssertionError(f"{benchmark}: selected {len(chosen)}, expected {quota}")
        selected.update(chosen)

    selected_ids = sorted(selected)
    _validate_selection(selected_ids, rows)
    return selected_ids


def _validate_selection(selected_ids: list[str], rows: dict[str, dict[str, Any]]) -> None:
    selected = set(selected_ids)
    candidates = {
        exp_id for exp_id in rows if weight_change_kind(exp_id, rows) in CANDIDATE_KINDS
    }
    base_ids = {
        exp_id for exp_id in rows if weight_change_kind(exp_id, rows) == "base_training_output"
    }
    benchmark_counts = Counter(rows[exp_id]["benchmark"] for exp_id in selected)
    if len(rows) != 579:
        raise AssertionError(f"Expected 579 eligible rows, found {len(rows)}")
    if len(candidates) != 516:
        raise AssertionError(f"Expected 516 candidate-weight proxies, found {len(candidates)}")
    if len(selected_ids) != 400 or len(selected) != 400:
        raise AssertionError("Selection must contain 400 unique IDs")
    if not selected <= candidates:
        raise AssertionError("Selection contains a non-candidate serving bundle")
    if len(base_ids) != 313 or not base_ids <= selected:
        raise AssertionError("Selection must retain all 313 eligible from-base rows")
    if benchmark_counts != {"gsm8k": 240, "aime2025": 160}:
        raise AssertionError(f"Unexpected benchmark allocation: {benchmark_counts}")
    if len({rows[exp_id]["cell"] for exp_id in selected}) != 124:
        raise AssertionError("Selection must cover all 124 source sessions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON array here instead of printing it to stdout.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        nargs="?",
        const=DEFAULT_FROZEN_IDS,
        help="Fail unless the generated IDs equal this JSON array (default: frozen IDs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eligible = _eligible_ids(args.labels)
    rows = _selection_rows(args.table, eligible)
    quality = _quality_rows(args.inputs, eligible)
    selected_ids = select(rows, quality)

    if args.check is not None:
        expected = json.loads(args.check.read_text())
        if selected_ids != expected:
            first_difference = next(
                (
                    index
                    for index, (actual, frozen) in enumerate(zip(selected_ids, expected))
                    if actual != frozen
                ),
                min(len(selected_ids), len(expected)),
            )
            raise SystemExit(
                f"Selection differs from {args.check} at index {first_difference}: "
                f"generated={len(selected_ids)}, expected={len(expected)}"
            )
        print(f"verified {len(selected_ids)} IDs against {args.check}", file=sys.stderr)

    payload = json.dumps(selected_ids, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
