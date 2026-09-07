"""Read-only recorder inventory for recipe/outcome world-model datasets.

This module does not write files, call models, or infer checkpoint lineage.
Eligibility is an initial schema/fidelity screen, not a certification that free
plan prose or recovered code is free of implicit outcome clues. The caller must
purify features and freeze its information boundary before modeling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tools.outcome_prediction.build_examples import FIDELITY_EXCLUSIONS
from tools.outcome_prediction.rpm_code_provenance import TraceIndex, reconstruct_card

PLAN_FIELDS = ("problem", "hypothesis", "setup", "evaluation")
CONFIG_KEYS = {
    "config",
    "config_path",
    "generation_config",
    "generation_config_path",
    "generation_config_file",
    "tokenizer_config",
    "tokenizer_config_path",
    "model_config_path",
    "config_file",
}


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _time(value):
    if not isinstance(value, str):
        raise TypeError("Record timestamp must be an ISO string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Record timestamp must include a timezone")
    return result


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _populated(value):
    """Zero/False are supplied values; recursively empty placeholders are not."""
    if isinstance(value, dict):
        return any(_populated(item) for item in value.values())
    if isinstance(value, list):
        return any(_populated(item) for item in value)
    return value is not None and value != ""


def _plan(card):
    return {key: copy.deepcopy(card[key]) for key in PLAN_FIELDS if key in card}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def classify_role(card):
    """Classify the declared intervention, without reading results/conclusions."""
    setup = _mapping(card.get("setup"))
    method = _mapping(setup.get("method"))
    family = re.sub(r"[-\s]+", "_", str(method.get("family") or "").lower())
    if any(word in family for word in ("merge", "averag", "soup")):
        return "merge"
    if "decod" in family:
        return "decoding"
    if "eval" in family or family in {"checkpoint_selection", "model_selection"}:
        return "evaluation"
    if family in {
        "sft",
        "distill",
        "distillation",
        "grpo",
        "ppo",
        "dpo",
        "rft",
        "rl",
        "training",
        "pretrain",
        "finetune",
        "full_finetune",
        "continued_pretraining",
    }:
        return "training"
    command = _mapping(setup.get("command"))
    text = (str(command.get("script") or "") + " " + json.dumps(command.get("argv") or [])).lower()
    if re.search(r"(?:merge|soup|average_weights|weight_averag)", text):
        return "merge"
    if re.search(r"(?:train|finetune|fine_tune|distill)[a-z0-9_.-]*\.py|\b(?:axolotl|trl)\b", text):
        return "training"
    if re.search(
        r"generation_config|decode[_-]|decoding|--(?:temperature|top[_-]p|repetition[_-]penalty)",
        text,
    ):
        return "decoding"
    if re.search(r"(?:evaluate|eval|benchmark|select_checkpoint)[a-z0-9_.-]*\.py", text):
        return "evaluation"
    return "unknown"


def _stage(record, path, ledger):
    submits = [
        entry
        for entry in ledger
        if entry.get("event") == "submit" and entry.get("card_id") == path.parent.name
    ]
    number = int(re.search(r"record-(\d+)\.json$", path.name)[1])
    exact = [
        entry
        for entry in submits
        if entry.get("record_n") == number or Path(str(entry.get("path") or "")).name == path.name
    ]
    matching = exact or [entry for entry in submits if entry.get("at") == record.get("at")]
    if (
        not matching
        and submits
        and all(not any(entry.get(key) for key in ("at", "record_n", "path")) for entry in submits)
    ):
        matching = submits[:1]
    stages = {entry.get("stage") for entry in matching if entry.get("stage") is not None}
    if not stages and record.get("stage"):
        stages = {record["stage"]}
    if len(stages) != 1:
        return None, []
    missing = sorted({str(item) for entry in matching for item in (entry.get("missing") or [])})
    return next(iter(stages)), missing


def _declared_configs(card):
    paths = set()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key.replace("-", "_") in CONFIG_KEYS
                    and isinstance(item, str)
                    and item.lower().endswith((".json", ".yaml", ".yml"))
                ):
                    paths.add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(_plan(card))
    setup = _mapping(card.get("setup"))
    evaluation = _mapping(card.get("evaluation"))
    for command in (
        _mapping(setup.get("command")).get("argv"),
        _mapping(evaluation.get("protocol")).get("command"),
    ):
        if isinstance(command, str):
            try:
                command = shlex.split(command)
            except ValueError:
                continue
        if not isinstance(command, list):
            continue
        for index, token in enumerate(command):
            if not isinstance(token, str) or not token.startswith("--"):
                continue
            flag, separator, argument = token[2:].partition("=")
            if flag.replace("-", "_") in CONFIG_KEYS:
                value = (
                    argument
                    if separator
                    else command[index + 1]
                    if index + 1 < len(command)
                    else None
                )
                if isinstance(value, str) and value.lower().endswith((".json", ".yaml", ".yml")):
                    paths.add(value)
    return sorted(paths)


def _code(card, cutoff, index, recover_code):
    """Main-command role is inherited from the recorder, including eval recipes."""
    setup = _mapping(card.get("setup"))
    safe_card = {**card, "setup": {**setup, "command": _mapping(setup.get("command"))}}
    if index is not None and cutoff is not None:
        entries = reconstruct_card(index, safe_card, cutoff)
    else:
        requests = [("training", _mapping(setup.get("command")).get("script"))]
        requests += [
            (f"data_builder_{i}", data.get("built_by"))
            for i, data in enumerate(setup.get("data") or [])
            if isinstance(data, dict) and data.get("built_by")
        ]
        status = "not_requested" if not recover_code else "unavailable"
        entries = [
            {
                "role": role,
                "script_path": path,
                "status": status,
                "content": None,
                "blockers": [
                    {
                        "reason": "code recovery disabled"
                        if not recover_code
                        else "trace or first registration unavailable"
                    }
                ],
            }
            for role, path in requests
        ]
    configs = _declared_configs(card)
    for i, path in enumerate(configs):
        if index is not None and cutoff is not None:
            entries.append({"role": f"evaluation_config_{i}", **index.reconstruct(path, cutoff)})
        else:
            entries.append(
                {
                    "role": f"evaluation_config_{i}",
                    "script_path": path,
                    "status": "not_requested" if not recover_code else "unavailable",
                    "content": None,
                }
            )
    if not configs:
        entries.append(
            {
                "role": "evaluation_config",
                "script_path": None,
                "status": "not_declared",
                "content": None,
            }
        )
    # Future trace hashes and audit timestamps are not predictive features.
    features = [
        {key: entry.get(key) for key in ("role", "script_path", "status", "content")}
        for entry in entries
    ]
    evidence = [
        {key: copy.deepcopy(value) for key, value in entry.items() if key != "content"}
        for entry in entries
    ]
    return features, evidence


def _fidelity(card, final):
    """Report post-hoc consistency signals only, never certify recipe identity."""
    initial_setup = _mapping(card.get("setup"))
    final_setup = _mapping(final.get("setup"))
    changed = sorted(
        key
        for key in initial_setup.keys() | final_setup.keys()
        if (key in initial_setup) != (key in final_setup)
        or initial_setup.get(key) != final_setup.get(key)
    )
    declared = initial_setup.get("output_dir")
    produced = _mapping(final.get("result")).get("output_checkpoint")
    mismatch = (
        isinstance(declared, str)
        and isinstance(produced, str)
        and declared.rstrip("/") != produced.rstrip("/")
    )
    flags = []
    if not final:
        flags.append("final_card_missing")
    if changed:
        flags.append("first_final_setup_changed")
    if mismatch:
        flags.append("declared_output_checkpoint_path_differs")
    if not declared or not produced:
        flags.append("declared_or_produced_output_unknown")
    return {
        "label_fidelity_review_required": True,
        "label_fidelity_status": "provisional_unadjudicated",
        "label_fidelity_flags": flags,
        "first_final_setup_changed": bool(changed),
        "first_final_setup_changed_fields": changed,
        "output_artifact_path_mismatch": mismatch,
        "output_artifact_comparison": {
            "first_declared_output_dir": copy.deepcopy(declared),
            "final_output_checkpoint": copy.deepcopy(produced),
            "comparison": "Literal paths ignoring trailing slash only; aliases, checkpoint selection, and artifact content are not adjudicated.",
        },
    }


def extract_recorder_dataset(
    *,
    raw_root,
    manifest_name,
    source_revision,
    benchmark,
    base_model,
    evaluation_n,
    recover_code=True,
):
    """Return all recorder rows and an audit, without writing any artifacts.

    `model_input` contains only the first plan's four sections and recovered code.
    `label`, `prior_observations`, provenance, and eligibility are separate fields.
    Earlier observations retain every submission and each original measurement;
    their n may count repeated generations, not distinct benchmark questions.
    Missing labels/code never imply failure and do not independently reject rows.
    """
    raw_root = Path(raw_root)
    if not all(
        isinstance(value, str) and value
        for value in (manifest_name, source_revision, benchmark, base_model)
    ):
        raise ValueError(
            "Manifest, revision, benchmark, and base model must be explicit nonempty strings"
        )
    if isinstance(evaluation_n, bool) or not isinstance(evaluation_n, int) or evaluation_n <= 0:
        raise ValueError("evaluation_n must be a positive integer")
    manifest_path = raw_root / manifest_name
    if not manifest_path.resolve().is_relative_to(raw_root.resolve()):
        raise ValueError("Manifest must be inside raw_root")
    manifest = _read(manifest_path)
    if not isinstance(manifest, list):
        raise TypeError("Recorder manifest must be a list of cells")
    ids = [meta.get("cell_id") for meta in manifest]
    if any(
        not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}
        for value in ids
    ) or len(ids) != len(set(ids)):
        raise ValueError("Manifest cell IDs must be unique single path components")
    task = {"benchmark": benchmark, "base_model": base_model, "evaluation_n": evaluation_n}
    rows, counts, issues = [], Counter(), []
    for meta in sorted(manifest, key=lambda item: item["cell_id"]):
        for key, expected in (("benchmark", benchmark), ("base_model", base_model)):
            if meta.get(key) is not None and meta[key] != expected:
                raise ValueError(
                    f"Manifest {key} disagrees with explicit task for {meta['cell_id']}"
                )
        cell_id = meta["cell_id"]
        cell = raw_root / "cells" / cell_id
        ledger_path = cell / "wm/records.jsonl"
        ledger = (
            [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
            if ledger_path.exists()
            else []
        )
        directories = {
            path.name: path for path in (cell / "wm/cards").glob("exp-*") if path.is_dir()
        }
        for metric_path in (cell / "wm_metrics").glob("exp-*.json"):
            directories.setdefault(metric_path.stem, cell / "wm/cards" / metric_path.stem)
        node_records, all_records, node_errors = {}, [], defaultdict(list)
        for card_id, folder in sorted(directories.items()):
            records = []
            for path in sorted(folder.glob("record-*.json")):
                try:
                    record = _read(path)
                    if not isinstance(record, dict):
                        raise TypeError("record must be an object")
                    at = _time(record.get("at"))
                    if not isinstance(record.get("card"), dict):
                        raise TypeError("record lacks a card object")
                    stage, missing = _stage(record, path, ledger)
                    records.append(
                        {
                            "raw": record,
                            "time": at,
                            "path": path,
                            "sha256": _sha(path),
                            "stage": stage,
                            "missing": missing,
                            "card_id": card_id,
                        }
                    )
                except (ValueError, TypeError, KeyError) as error:
                    issue = {
                        "cell_id": cell_id,
                        "card_id": card_id,
                        "path": str(path),
                        "reason": str(error),
                    }
                    issues.append(issue)
                    node_errors[card_id].append(issue)
            records.sort(key=lambda item: (item["time"], item["path"].name))
            node_records[card_id] = records
            all_records.extend(records)
        all_records.sort(key=lambda item: (item["time"], item["card_id"], item["path"].name))
        trace_path = cell / "solve_out_sanitized.txt"
        index = TraceIndex(trace_path) if recover_code and trace_path.exists() else None
        for card_id, folder in sorted(directories.items()):
            records = node_records[card_id]
            first = records[0] if records else None
            card = first["raw"]["card"] if first else {}
            cutoff = first["raw"]["at"] if first else None
            stage = first["stage"] if first else None
            example_id = cell_id + "/" + card_id
            reasons, warnings = [], []
            if first is None:
                reasons.append("missing_versioned_registration")
            if stage != "plan":
                reasons.append("first_submission_not_plan")
            if _populated(card.get("result")):
                reasons.append("first_result_not_empty")
            if first is not None and not isinstance(card.get("setup"), dict):
                reasons.append("first_plan_missing_setup")
            if node_errors[card_id]:
                reasons.append("invalid_versioned_record")
            if card.get("card_id") not in (None, card_id):
                reasons.append("first_card_id_mismatch")
            declared_base = _mapping(card.get("setup")).get("base_model")
            if declared_base and declared_base != base_model:
                reasons.append("declared_base_model_mismatch")
            if benchmark.lower() == "gsm8k" and example_id in FIDELITY_EXCLUSIONS:
                reasons.append("known_fidelity_exclusion")
            if _populated(card.get("conclusion")):
                warnings.append("first_conclusion_populated; excluded from model_input")
            if first and first["missing"]:
                warnings.append("first_registration_reports_missing_fields")
            role = classify_role(card)
            code, code_evidence = _code(card, cutoff, index, recover_code)
            metric_path = cell / "wm_metrics" / (card_id + ".json")
            official = _read(metric_path) if metric_path.exists() else None
            accuracy = _number(official.get("accuracy")) if isinstance(official, dict) else None
            stderr = _number(official.get("stderr")) if isinstance(official, dict) else None
            valid_label = accuracy is not None and 0 <= accuracy <= 1
            correct = round(accuracy * evaluation_n) if valid_label else None
            if valid_label and not math.isclose(accuracy * evaluation_n, correct, abs_tol=1e-8):
                correct = None
                warnings.append("official_accuracy_not_integer_count_at_evaluation_n")
            prior = []
            for earlier in all_records:
                if (
                    first is None
                    or earlier["time"] >= first["time"]
                    or earlier["card_id"] == card_id
                ):
                    continue
                previous = earlier["raw"]["card"]
                result = _mapping(previous.get("result"))
                prior.append(
                    {
                        "example_id": cell_id + "/" + earlier["card_id"],
                        "card_id": earlier["card_id"],
                        "at": earlier["raw"]["at"],
                        "stage": earlier["stage"],
                        "measurements": copy.deepcopy(result.get("measurements") or []),
                        "evaluation": copy.deepcopy(previous.get("evaluation") or {}),
                        "record": copy.deepcopy(earlier["raw"]),
                        "scope": "Scientist submission strictly before this proposal; local protocols and repeated-generation counts remain uncollapsed. No official archive score is injected.",
                        "provenance": {"path": str(earlier["path"]), "sha256": earlier["sha256"]},
                    }
                )
            eligible = not reasons
            final_path = folder / "card.json"
            final = _mapping(_read(final_path)) if final_path.exists() else {}
            audit = {
                "eligible": eligible,
                "eligibility_status": "provisional_plan_stage_screen",
                **_fidelity(card, final),
                "reasons": reasons,
                "warnings": warnings,
                "has_official_label": valid_label,
                "first_missing_fields": first["missing"] if first else [],
                "record_errors": copy.deepcopy(node_errors[card_id]),
                "known_fidelity_reason": FIDELITY_EXCLUSIONS.get(example_id)
                if "known_fidelity_exclusion" in reasons
                else None,
                "code_provenance": code_evidence,
                "code_unknowns": [
                    {"role": item["role"], "status": item["status"]}
                    for item in code
                    if item["status"] != "reconstructed"
                ],
                "lineage_status": "not_inferred",
                "purity_scope": "Structural first-plan screen only; prose/code outcome clues and candidate-set dependencies require separate review.",
            }
            rows.append(
                {
                    "example_id": example_id,
                    "cell_id": cell_id,
                    "card_id": card_id,
                    "scientist_model": meta.get("scientist_model", "unknown"),
                    "benchmark": benchmark,
                    "task": copy.deepcopy(task),
                    "first_submitted_at": cutoff,
                    "first_stage": stage,
                    "role": role,
                    "eligible": eligible,
                    "model_input": {"task": copy.deepcopy(task), "plan": _plan(card), "code": code},
                    "label": {
                        "official_metric": copy.deepcopy(official),
                        "accuracy": accuracy,
                        "stderr": stderr,
                        "correct_count": correct,
                        "evaluation_n": evaluation_n,
                    },
                    "prior_observations": prior,
                    "audit": audit,
                    "provenance": {
                        "raw_root": str(raw_root),
                        "manifest_name": manifest_name,
                        "manifest_sha256": _sha(manifest_path),
                        "source_revision": source_revision,
                        "first_record_path": str(first["path"]) if first else None,
                        "first_record_sha256": first["sha256"] if first else None,
                        "card_path": str(folder / "card.json"),
                        "card_sha256": _sha(folder / "card.json"),
                        "official_metric_path": str(metric_path) if metric_path.exists() else None,
                        "official_metric_sha256": _sha(metric_path),
                        "record_count": len(records),
                        "ledger_path": str(ledger_path),
                        "ledger_sha256": _sha(ledger_path),
                    },
                }
            )
            counts["rows"] += 1
            counts["eligible"] += eligible
            counts["official_labels"] += valid_label
            counts["eligible_labeled"] += eligible and valid_label
            counts["missing_versioned_registration"] += first is None
            counts["role_" + role] += 1
            counts["first_final_setup_changed"] += audit["first_final_setup_changed"]
            counts["output_artifact_path_mismatch"] += audit["output_artifact_path_mismatch"]
        counts["cells"] += 1
    return rows, {
        "schema": "wm-recorder-dataset-v1",
        "task": task,
        "source_revision": source_revision,
        "manifest_name": manifest_name,
        "manifest_sha256": _sha(manifest_path),
        "counts": dict(counts),
        "record_errors": issues,
        "exclusion_counts": dict(
            Counter(reason for row in rows for reason in row["audit"]["reasons"])
        ),
        "limits": [
            "No lineage or candidate-set independence is inferred.",
            "Final cards and snapshots are never a source of model-input content.",
            "Eligibility is provisional; first/final setup equality does not prove that an official checkpoint realizes the proposed recipe.",
            "Prior observations are separate from model_input and require the caller's explicit information policy.",
            "Code slot 'training' names the primary command script, even for decoding/evaluation rows; absence is unknown, not a crash.",
            "Only explicitly declared config paths are considered; transitive imports and implicit checkpoint configs remain unknown.",
        ],
    }


def stratified_run_split(rows, *, test_count, seed):
    """Freeze whole-run partitions from benchmark/scientist/IDs, never outcomes."""
    if isinstance(test_count, bool) or not isinstance(test_count, int) or test_count < 0:
        raise ValueError("test_count must be a nonnegative integer per stratum")
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise TypeError("seed must be an explicit integer or string")
    cells, examples = {}, {}
    for row in rows:
        cell_id, example_id = row["cell_id"], row["example_id"]
        benchmark = row.get("benchmark") or (row.get("task") or {}).get("benchmark")
        scientist = row.get("scientist_model")
        if not all(
            isinstance(value, str) and value
            for value in (cell_id, example_id, benchmark, scientist)
        ):
            raise ValueError("Each row needs cell, example, benchmark, and scientist identities")
        stratum = (benchmark, scientist)
        if cell_id in cells and cells[cell_id] != stratum:
            raise ValueError("A cell cannot belong to multiple benchmark/scientist strata")
        if example_id in examples:
            raise ValueError("Example IDs must be unique")
        cells[cell_id] = stratum
        examples[example_id] = cell_id
    grouped = defaultdict(list)
    for cell_id, stratum in cells.items():
        grouped[stratum].append(cell_id)
    test_cells, strata = set(), []
    for (benchmark, scientist), members in sorted(grouped.items()):
        if test_count >= len(members) and test_count:
            raise ValueError(
                f"test_count leaves no training run in stratum {(benchmark, scientist)}"
            )
        ordered = sorted(
            members,
            key=lambda cell_id: hashlib.sha256(
                json.dumps(["wm-run-split-v1", seed, benchmark, scientist, cell_id]).encode()
            ).hexdigest(),
        )
        chosen = ordered[:test_count]
        test_cells.update(chosen)
        strata.append(
            {
                "benchmark": benchmark,
                "scientist_model": scientist,
                "train_cell_ids": sorted(set(members) - set(chosen)),
                "test_cell_ids": sorted(chosen),
            }
        )
    train_cells = set(cells) - test_cells
    return {
        "schema": "wm-whole-run-split-v1",
        "status": "frozen_by_identity",
        "seed": seed,
        "test_count_per_stratum": test_count,
        "train_cell_ids": sorted(train_cells),
        "test_cell_ids": sorted(test_cells),
        "train_example_ids": sorted(eid for eid, cid in examples.items() if cid in train_cells),
        "test_example_ids": sorted(eid for eid, cid in examples.items() if cid in test_cells),
        "cell_partition": {cid: "test" if cid in test_cells else "train" for cid in sorted(cells)},
        "strata": strata,
        "policy": "All supplied rows, including rejected/unlabeled ones, stay with their run; no labels or model features determine membership.",
    }
