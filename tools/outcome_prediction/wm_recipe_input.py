"""Complete-recipe / final-target-only input and supervision boundary.

This validates structure, DAG coverage, and externally reviewed content hashes.
It cannot prove that prose or code lacks paraphrased outcomes. Every retained
step therefore needs a content-bound review OUTSIDE the input. Do not pass legacy
recorder plans through unchanged or manufacture review approval.

The terminal-target supervision check is an OPTIONAL stricter policy, pending
the user's clarification about separately supervised shorter recipes. The hard
rule already established is no ancestor outcomes in a recipe's inputs. Do not
silently drop prefix-recipe targets by applying the optional check as a default.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

SCHEMA = "full_recipe_final_only_v1"
PLAN_FIELDS = {"problem", "hypothesis", "setup", "evaluation"}
OUTCOME_FIELDS = {
    "result",
    "results",
    "conclusion",
    "measurements",
    "prior_observations",
    "known_previous_checkpoints",
    "official_accuracy",
    "observed_accuracy",
    "observed_score",
    "official_metric",
    "official_correct_of_30",
    "stderr",
    "delta_vs_comparator",
    "label",
    "labels",
    "y",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _structural_scan(value):
    if isinstance(value, dict):
        if OUTCOME_FIELDS & set(value):
            raise ValueError("Outcome field in complete-recipe input")
        if "comparator" in value:
            comparator = value["comparator"]
            if not isinstance(comparator, dict) or set(comparator) - {"ref"}:
                raise ValueError("Comparator may identify a step, not its measured outcome")
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Recipe object keys must be strings")
            _structural_scan(item)
    elif isinstance(value, list):
        for item in value:
            _structural_scan(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("Non-finite recipe value")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise TypeError("Recipe must contain only JSON values")


def validate_full_recipe(payload):
    """Validate an ordered DAG whose every step contributes to its final target."""
    if not isinstance(payload, dict) or set(payload) != {"schema", "task", "recipe"}:
        raise ValueError("Complete recipe requires schema, task, and recipe only")
    if payload["schema"] != SCHEMA:
        raise ValueError("Unsupported full-recipe schema")
    task = payload["task"]
    if not isinstance(task, dict) or set(task) != {"benchmark", "base_model", "evaluation_n"}:
        raise ValueError("Task must specify benchmark, base model, and final evaluation size")
    if not _nonempty(task["benchmark"]) or not _nonempty(task["base_model"]):
        raise ValueError("Invalid task identity")
    if type(task["evaluation_n"]) is not int or task["evaluation_n"] < 1:
        raise ValueError("Invalid final evaluation size")
    recipe = payload["recipe"]
    if not isinstance(recipe, dict) or set(recipe) != {"steps", "final_step_id"}:
        raise ValueError("Recipe requires ordered steps and one final step")
    steps = recipe["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Empty recipe")
    seen = {}
    for step in steps:
        if not isinstance(step, dict) or set(step) != {
            "step_id",
            "role",
            "parents",
            "plan",
            "code",
        }:
            raise ValueError("Unexpected recipe-step fields")
        step_id = step["step_id"]
        if not _nonempty(step_id) or step_id in seen:
            raise ValueError("Invalid or duplicate recipe step")
        if step["role"] not in {"training", "merge", "decoding", "evaluation", "data_generation"}:
            raise ValueError("Unresolved intervention role requires review")
        if not isinstance(step["parents"], list):
            raise TypeError("Step parents must be explicit")
        edges = set()
        for edge in step["parents"]:
            if not isinstance(edge, dict) or set(edge) != {"step_id", "kind", "artifact"}:
                raise ValueError("Dependency must identify its producing step and exact artifact")
            if edge["step_id"] not in seen:
                raise ValueError("Missing, cyclic, or non-topological ancestor")
            if edge["kind"] not in {"weights", "generated_data", "configuration"}:
                raise ValueError("Unresolved dependency kind")
            if not _nonempty(edge["artifact"]):
                raise ValueError("Missing parent artifact; producer ID alone is insufficient")
            identity = tuple(edge[k] for k in ("step_id", "kind", "artifact"))
            if identity in edges:
                raise ValueError("Duplicate dependency")
            edges.add(identity)
        if not isinstance(step["plan"], dict) or set(step["plan"]) - PLAN_FIELDS:
            raise ValueError("Plan must contain only outcome-free scientific sections")
        if not isinstance(step["plan"].get("setup"), dict):
            raise TypeError("A complete recipe step needs its executable setup")
        if not isinstance(step["code"], list):
            raise TypeError("Code availability must be explicit")
        for code in step["code"]:
            if not isinstance(code, dict) or set(code) != {
                "role",
                "script_path",
                "status",
                "content",
            }:
                raise ValueError("Unexpected code fields; provenance belongs outside input")
            if not _nonempty(code["role"]):
                raise ValueError("Code role missing")
            if code["script_path"] is not None and not _nonempty(code["script_path"]):
                raise ValueError("Invalid script path")
            if code["status"] == "reconstructed":
                if not isinstance(code["content"], str):
                    raise ValueError("Reconstructed code needs reviewed content")
            elif (
                code["status"] not in {"unavailable", "not_declared"} or code["content"] is not None
            ):
                raise ValueError("Unknown code cannot be replaced with a later snapshot")
        _structural_scan(step)
        seen[step_id] = step
    final = recipe["final_step_id"]
    if final not in seen or final != steps[-1]["step_id"]:
        raise ValueError("Final target must be the last recipe step")
    ancestors = set()

    def visit(step_id):
        if step_id not in ancestors:
            ancestors.add(step_id)
            for edge in seen[step_id]["parents"]:
                visit(edge["step_id"])

    visit(final)
    if ancestors != set(seen):
        raise ValueError("Extraneous steps are not part of this target recipe")
    return payload


def build_reviewed_input(*, task, steps, final_step_id, reviews):
    """Package already cleaned/reviewed steps; refuse stale or absent reviews."""
    payload = {
        "schema": SCHEMA,
        "task": copy.deepcopy(task),
        "recipe": {
            "steps": copy.deepcopy(steps),
            "final_step_id": final_step_id,
        },
    }
    validate_full_recipe(payload)
    if not isinstance(reviews, dict) or set(reviews) != {s["step_id"] for s in steps}:
        raise ValueError("Every retained step requires an outcome-free content review")
    for step in steps:
        review = reviews[step["step_id"]]
        if not isinstance(review, dict) or set(review) != {
            "payload_sha256",
            "reviewer",
            "evidence_sha256",
            "outcome_free",
            "executable_recipe_preserved",
        }:
            raise ValueError("Incomplete recipe content review")
        if review["payload_sha256"] != digest(step):
            raise ValueError("Step changed after outcome review")
        if review["outcome_free"] is not True or review["executable_recipe_preserved"] is not True:
            raise ValueError("Unapproved recipe content")
        if (
            not _nonempty(review["reviewer"])
            or not isinstance(review["evidence_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", review["evidence_sha256"])
        ):
            raise ValueError("Review requires attributable, fingerprinted evidence")
    return payload


def assert_terminal_target_supervision(rows):
    """No supervised target can be an internal node of another included recipe.

    Each row has example_id, cell_id, model_input, and step_sources mapping local
    step IDs to full source example IDs. Call on the entire proposed supervised
    corpus, before partitioning or fitting. This does not inspect score values.
    """
    targets, ancestors = set(), set()
    for row in rows:
        validate_full_recipe(row["model_input"])
        recipe = row["model_input"]["recipe"]
        sources = row["step_sources"]
        if not isinstance(sources, dict) or set(sources) != {s["step_id"] for s in recipe["steps"]}:
            raise ValueError("Incomplete step provenance")
        if any(
            not isinstance(v, str) or not v.startswith(row["cell_id"] + "/")
            for v in sources.values()
        ):
            raise ValueError("Recipe steps must remain in their scientist run")
        target = sources[recipe["final_step_id"]]
        if target != row["example_id"] or target in targets:
            raise ValueError("Duplicate or mismatched final target")
        if len(set(sources.values())) != len(sources):
            raise ValueError("One source card cannot masquerade as two recipe steps")
        targets.add(target)
        ancestors.update(v for k, v in sources.items() if k != recipe["final_step_id"])
    if targets & ancestors:
        raise ValueError("A supervised final target is an internal checkpoint of another recipe")
    return {"target_ids": sorted(targets), "unlabeled_ancestor_ids": sorted(ancestors)}
