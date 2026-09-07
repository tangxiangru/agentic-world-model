"""Build auditable, prospective recipe-to-checkpoint examples from the r0 release.

Generated data stays under the ignored data/ tree. No transcript or script snapshot
is a model feature: snapshots were overwritten on later card submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

HP_NUM = {
    "lr",
    "learning_rate",
    "epochs",
    "num_train_epochs",
    "max_steps",
    "steps",
    "batch_size",
    "per_device_train_batch_size",
    "grad_accum",
    "gradient_accumulation_steps",
    "max_seq_len",
    "max_length",
    "seq_len",
    "warmup",
    "warmup_ratio",
    "warmup_steps",
    "weight_decay",
    "seed",
    "lora_rank",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "max_grad_norm",
    "beta",
    "temperature",
    "num_generations",
    "max_completion_length",
    "max_prompt_length",
    "kl_coef",
    "cliprange",
    "r",
    "alpha",
    "save_steps",
}
HP_CAT = {"precision", "scheduler", "optimizer", "optim", "loss", "attn_implementation"}
FLAG_ALIASES = {
    "lr": "lr",
    "learning_rate": "lr",
    "epochs": "epochs",
    "num_train_epochs": "epochs",
    "bs": "batch_size",
    "batch_size": "batch_size",
    "per_device_train_batch_size": "batch_size",
    "ga": "grad_accum",
    "grad_accum": "grad_accum",
    "gradient_accumulation_steps": "grad_accum",
    "max_len": "max_seq_len",
    "max_length": "max_seq_len",
    "max_seq_length": "max_seq_len",
    "max_seq_len": "max_seq_len",
    "seq_len": "max_seq_len",
    "warmup": "warmup",
    "warmup_ratio": "warmup",
    "max_steps": "max_steps",
    "seed": "seed",
    "save_steps": "save_steps",
    "num_gen": "num_generations",
    "num_generations": "num_generations",
    "max_completion": "max_completion_length",
    "max_prompt": "max_prompt_length",
    "weight_decay": "weight_decay",
    "beta": "beta",
    "temperature": "temperature",
}
PARENT_FLAGS = {
    "model",
    "models",
    "inputs",
    "srcs",
    "sources",
    "base",
    "base_model",
    "parent",
    "checkpoints",
    "ckpt",
    "model_a",
    "model_b",
    "a",
    "b",
    "resume_from_checkpoint",
}
MERGE_FAMILIES = {"merge", "model_soup", "model_merge"}
DATASET_IDS = {
    "openai/gsm8k",
    "nvidia/OpenMathInstruct-2",
    "meta-math/MetaMathQA",
    "microsoft/orca-math-word-problems-200k",
    "google/gemma-3-4b-pt",
}
EVAL_IDS = {
    "r0-06/exp-09",
    "r0-08/exp-07",
    "r0-18/exp-08",
    "r0-20/exp-06",
    "r0-26/exp-07",
    "r0-28/exp-08",
    "r0-28/exp-09",
}
# Read-only manual audit, recorded explicitly rather than silently repairing recipes.
FIDELITY_EXCLUSIONS = {
    "r0-02/exp-03": "official checkpoint includes an unplanned soup",
    "r0-05/exp-04": "official label selects epoch 1 although recipe plans 2 epochs",
    "r0-12/exp-06": "planned soup appears in hypothesis but setup encodes only SFT",
    "r0-12/exp-08": "planned soup appears in hypothesis but setup encodes only SFT",
    "r0-22/exp-07": "GRPO plus six-checkpoint soup is not fully encoded by setup",
    "r0-14/exp-05": "setup notes say merge and evaluation launched before registration",
}
# Ingredients verified against the first plan's command/hypothesis. Multiple
# snapshots from one producing card are represented once; intermediate snapshot
# identity remains a fidelity limitation, not a new official label.
MERGE_PARENTS = {
    "r0-01": {7: [5, 2], 8: [5, 2, 4], 9: [5, 2]},
    "r0-02": {5: [2, 4]},
    "r0-04": {3: [1, 2], 8: [2, 6], 9: [1, 2, 6], 10: [1, 2]},
    "r0-06": {5: [2, 4], 6: [2, 1], 7: [2]},
    "r0-08": {4: [1, 3], 6: [5, 1]},
    "r0-10": {4: [2, 3]},
    "r0-12": {5: [1, 4], 7: [1, 4, 6]},
    "r0-14": {
        5: [3, 4],
        6: [1, 3, 4],
        7: [3, 4],
        8: [1, 3, 4],
        9: [1, 4],
        10: [1, 3, 4],
        11: [1, 3, 4],
    },
    "r0-16": {3: [1, 2], 5: [1, 2]},
    "r0-18": {5: [1, 4], 6: [1, 4], 7: [1, 4]},
    "r0-20": {7: [3, 5]},
    "r0-22": {6: [4, 5]},
    "r0-24": {4: [1, 3], 6: [3, 5]},
    "r0-28": {5: [2, 4], 7: [2, 4, 6]},
    "r0-29": {5: [2]},
    "r0-30": {7: [3], 8: [3, 6]},
    "r0-31": {6: [5, 4]},
    "r0-32": {5: [4, 3, 2]},
}


def read(path):
    return json.loads(path.read_text())


def number(value):
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def normalize_path(value):
    return str(value or "").replace("/home/ben/task/", "").removeprefix("./").rstrip("/")


def argv_flags(argv):
    if not isinstance(argv, list):
        return {}
    result = {}
    key = None
    for item in argv:
        item = str(item)
        if item.startswith("--"):
            key, _, value = item[2:].replace("-", "_").partition("=")
            result.setdefault(key, []).extend([value] if value else [])
        elif key is not None:
            result[key].append(item)
    return result


def canonical(card):
    """Whitelisted recipe: no outcome text, IDs, file names, or researcher identity."""
    setup = card.get("setup") or {}
    method = setup.get("method") or {}
    family = str(method.get("family") or "unknown").lower()
    flags = argv_flags((setup.get("command") or {}).get("argv"))
    if (
        family in MERGE_FAMILIES
        or "soup" in str((setup.get("command") or {}).get("script", "")).lower()
    ):
        family = "merge"
    elif family == "sft_continued":
        family = "sft"
    hp = method.get("hyperparams") or {}
    if not isinstance(hp, dict):
        hp = {}
    values = {
        key: number(value)
        for key, value in hp.items()
        if key in HP_NUM and number(value) is not None
    }
    for flag, items in flags.items():
        if flag in FLAG_ALIASES and len(items) == 1 and number(items[0]) is not None:
            values[FLAG_ALIASES[flag]] = number(items[0])
    cats = {}
    categorical_tokens = {
        "precision": ["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        "scheduler": ["cosine", "linear", "constant", "polynomial"],
        "optimizer": ["adamw", "adam", "sgd", "adafactor", "8bit", "fused", "paged"],
        "optim": ["adamw", "adam", "sgd", "adafactor", "8bit", "fused", "paged"],
        "loss": ["completion", "causal", "cross_entropy", "cross-entropy", "kl"],
        "attn_implementation": ["flash_attention_2", "sdpa", "eager"],
    }
    for key in HP_CAT:
        if hp.get(key) is not None:
            found = [t for t in categorical_tokens[key] if t in str(hp[key]).lower()]
            cats[key] = "+".join(found) or "unspecified"
    data = setup.get("data") or []
    if not isinstance(data, list):
        data = [data]
    sources = []
    counts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source") or "unknown")
        repo_ids = sorted(s for s in DATASET_IDS if s.lower() in src.lower())
        words = [
            w
            for w in (
                "gsm8k",
                "synthetic",
                "self-generated",
                "self_generated",
                "math",
                "metamath",
                "openmathinstruct",
                "distill",
            )
            if w in src.lower()
        ]
        entry = {"source_ids": repo_ids, "source_tags": words}
        for key in ("n_examples", "n", "mixture_weight", "repeats"):
            val = number(item.get(key))
            if val is not None:
                entry[key] = val
        sources.append(entry)
        count = number(item.get("n_examples", item.get("n")))
        if count is not None:
            counts.append(count)
    origin = str((setup.get("parent_checkpoint") or {}).get("origin") or "unknown")
    parent_kind = "base" if origin == "base_model" else "checkpoint"
    if family == "merge":
        parent_kind = "merge"
    framework_text = str(method.get("framework") or "").lower()
    packages = {}
    for package in (
        "transformers",
        "trl",
        "torch",
        "liger-kernel",
        "vllm",
        "peft",
        "safetensors",
        "bitsandbytes",
        "flash-attn",
    ):
        pattern = (
            re.escape(package).replace(r"\-", "[-_]")
            + r"(?:\.[a-z]+)?\s*([0-9]+(?:\.[0-9]+){1,2})?"
        )
        match = re.search(pattern, framework_text)
        if match:
            packages[package] = match.group(1) or "present"
    obj = {
        "base_model": str(setup.get("base_model") or "unknown"),
        "method": family,
        "parent_kind": parent_kind,
        "framework": packages,
        "hyperparameters": values,
        "configuration": cats,
        "data": sources,
    }
    peft = str(method.get("peft") or "unknown").lower()
    obj["peft"] = (
        "lora"
        if "lora" in peft
        else "none"
        if peft in {"none", "full", "full finetune", "full fine-tuning"}
        else "unknown"
    )
    budget = setup.get("budget") or {}
    if number(budget.get("planned_h")) is not None:
        obj["planned_hours"] = number(budget["planned_h"])
    num = {f"hp_{k}": v for k, v in values.items()}
    num["data_components"] = len(sources)
    if counts:
        num["data_examples"] = sum(counts)
    if "planned_hours" in obj:
        num["planned_hours"] = obj["planned_hours"]
    if family == "merge":
        member_paths = []
        for key in ("models", "inputs", "srcs", "sources", "checkpoints", "ckpt", "a", "b"):
            member_paths.extend(p for v in flags.get(key, []) for p in v.split(","))
        argv = (setup.get("command") or {}).get("argv") or []
        if (
            not member_paths
            and len(argv) > 2
            and isinstance(argv[2], str)
            and "," in argv[2]
            and "/" in argv[2]
        ):
            member_paths = argv[2].split(",")
        weights = [
            number(v)
            for s in flags.get("weights", [])
            for v in re.split(r"[, /]+", s)
            if number(v) is not None
        ]
        if not weights and len(flags.get("wa", [])) == 1 and number(flags["wa"][0]) is not None:
            wa = number(flags["wa"][0])
            weights = [wa, 1 - wa]
        if not weights:
            declared = str(hp.get("weights") or "")
            each = re.search(r"1/(\d+)\s+each", declared)
            if each:
                weights = [1 / int(each.group(1))] * int(each.group(1))
            elif declared and "exp" not in declared:
                weights = [float(x) for x in re.findall(r"(?<!\w)-?\d*\.\d+", declared)]
            if len(weights) == 1 and "each" in declared and member_paths:
                weights *= len(member_paths)
        if not weights and member_paths and "uniform" in str(hp.get("other", "")).lower():
            weights = [1 / len(member_paths)] * len(member_paths)
        if (
            not weights
            and len(argv) > 3
            and isinstance(argv[3], str)
            and re.fullmatch(r"[0-9., -]+", argv[3])
        ):
            weights = [float(v) for v in argv[3].split(",")]
        details = {
            "member_count": len(member_paths) or number(hp.get("members")),
            "weights": weights,
            "intermediate_steps": [
                int(s) for p in member_paths for s in re.findall(r"checkpoint[-_]?(\d+)", p)
            ],
        }
        obj["merge"] = details
        if details["member_count"]:
            num["merge_member_count"] = details["member_count"]
        for i, weight in enumerate(weights):
            num[f"merge_weight_{i}"] = weight
        num["merge_intermediate_count"] = len(details["intermediate_steps"])
    for key, value in list(num.items()):
        if value > 0:
            num["log10_" + key] = math.log10(value)
    categories = {"method": family, "parent_kind": parent_kind, "peft": obj["peft"], **cats}
    for src in sources:
        for tag in src["source_ids"] + src["source_tags"]:
            categories["data_" + tag] = "present"
    return obj, num, categories


def local_accuracy(card):
    measurements = (card.get("result") or {}).get("measurements") or []
    candidates = []
    for m in measurements:
        if not isinstance(m, dict):
            continue
        value = number(m.get("value"))
        name = str(m.get("name") or m.get("metric") or "").lower()
        if value is not None and ("accuracy" in name or "gsm8k" in name):
            if 1 < value <= 100:
                value /= 100
            if 0 <= value <= 1:
                candidates.append((number(m.get("n")) or 0, value))
    return max(candidates)[1] if candidates else None


def unique_largest_measurement(card):
    """A prior score only when the largest-n measurement is unambiguous."""
    measurements = (card.get("result") or {}).get("measurements") or []
    candidates = [
        (number(m.get("n")) or 0, number(m.get("value")))
        for m in measurements
        if isinstance(m, dict)
        and (
            "accuracy" in str(m.get("name", "")).lower()
            or "gsm8k" in str(m.get("name", "")).lower()
        )
        and number(m.get("value")) is not None
        and 0 <= number(m.get("value")) <= 1
    ]
    if not candidates:
        return None
    maximum = max(n for n, _ in candidates)
    values = {v for n, v in candidates if n == maximum}
    return next(iter(values)) if len(values) == 1 else None


def build(root):
    manifest = {x["cell_id"]: x for x in read(root / "manifest_r0.json")}
    rows = []
    audit = {
        "source_revision": "7294afde88f6e70bb7d0899de3e8e10f3172c622",
        "manual_fidelity_exclusions": FIDELITY_EXCLUSIONS,
    }
    for cell_id, meta in sorted(manifest.items()):
        cell = root / "cells" / cell_id
        ledger = [
            json.loads(s) for s in (cell / "wm/records.jsonl").read_text().splitlines() if s.strip()
        ]
        nodes = {}
        for path in sorted((cell / "wm/cards").glob("exp-*/card.json")):
            cid = path.parent.name
            records = [read(p) for p in sorted(path.parent.glob("record-*.json"))]
            if not records:
                continue
            first = records[0]
            submits = [x for x in ledger if x.get("card_id") == cid and x.get("event") == "submit"]
            metric_path = cell / "wm_metrics" / (cid + ".json")
            metric = read(metric_path) if metric_path.exists() else {}
            nodes[cid] = {
                "records": records,
                "first": first,
                "stage": submits[0].get("stage") if submits else None,
                "final": read(path),
                "y": number(metric.get("accuracy")),
                "stderr": number(metric.get("stderr")),
                "path": str(path.relative_to(root)),
                "first_missing": submits[0].get("missing", []) if submits else [],
            }

        def get_parents(cid, card, cell_id=cell_id, nodes=nodes):
            setup = card.get("setup") or {}
            pc = setup.get("parent_checkpoint") or {}
            current_num = int(cid.split("-")[1])
            if cell_id == "r0-08" and cid == "exp-03":
                return [
                    "exp-02"
                ], []  # copied checkpoint500; earliest recipe explicitly says resume
            if current_num in MERGE_PARENTS.get(cell_id, {}):
                return sorted(f"exp-{i:02d}" for i in MERGE_PARENTS[cell_id][current_num]), []
            flags = argv_flags((setup.get("command") or {}).get("argv"))
            parent_paths = [pc.get("path")]
            for flag in PARENT_FLAGS:
                parent_paths.extend(
                    piece for value in flags.get(flag, []) for piece in value.split(",")
                )
            refs = set()
            unknown = []
            for raw in parent_paths:
                path = normalize_path(raw)
                if (
                    not path
                    or path in {"google/gemma-3-4b-pt", "gemma-3-4b-pt"}
                    or "models--google--gemma-3-4b-pt" in path
                ):
                    continue
                matches = []
                for other, node in nodes.items():
                    if int(other.split("-")[1]) >= current_num:
                        continue
                    aliases = []
                    for rec in node["records"]:
                        c = rec["card"]
                        aliases.extend(
                            [
                                (c.get("setup") or {}).get("output_dir"),
                                (c.get("result") or {}).get("output_checkpoint"),
                            ]
                        )
                    for alias in aliases:
                        alias = normalize_path(alias)
                        if alias and (path == alias or path.startswith(alias + "/")):
                            matches.append((len(alias), other))
                if matches:
                    refs.add(max(matches)[1])
                else:
                    ids = re.findall(r"exp[-_]?0*(\d+)", path)
                    valid = [
                        f"exp-{int(i):02d}"
                        for i in ids
                        if f"exp-{int(i):02d}" in nodes and int(i) < current_num
                    ]
                    if valid:
                        refs.update(valid)
                    else:
                        unknown.append(path)
            if not refs:
                origin = str(pc.get("origin") or "")
                refs.update(
                    f"exp-{int(i):02d}"
                    for i in re.findall(r"exp[-_]?0*(\d+)", origin)
                    if f"exp-{int(i):02d}" in nodes and int(i) < current_num
                )
                if refs:
                    unknown = []
                elif origin not in {"base_model", ""}:
                    unknown.append("unresolved origin")
            family = canonical(card)[0]["method"]
            if family == "merge" and len(refs) < 2:
                unknown.append(
                    "merge ingredients incomplete (may merge checkpoints within one experiment)"
                )
            return sorted(refs), unknown

        def data_dependencies(cid, card, nodes=nodes):
            # Include explicitly named generators/filters even on weight restarts.
            text = json.dumps((card.get("setup") or {}).get("data", []))
            return sorted(
                {
                    f"exp-{int(i):02d}"
                    for i in re.findall(r"exp[-_]?0*(\d+)", text)
                    if int(i) < int(cid.split("-")[1]) and f"exp-{int(i):02d}" in nodes
                }
            )

        for cid, node in nodes.items():
            first = node["first"]
            card = first["card"]
            canonical_card, num, cats = canonical(card)
            final_canonical = canonical(node["final"])[0]
            eid = cell_id + "/" + cid
            reasons = []
            if node["stage"] != "plan":
                reasons.append("first_submission_not_plan")
            if local_accuracy(card) is not None:
                reasons.append("first_submission_has_own_measurement")
            if canonical_card != final_canonical:
                reasons.append("canonical_recipe_changed_after_first_submission")
            if eid in FIDELITY_EXCLUSIONS:
                reasons.append("manual_recipe_fidelity_exclusion")
            if eid in EVAL_IDS or canonical_card["method"].startswith("eval"):
                reasons.append("evaluation_or_selection_only")
            cutoff = first["at"]
            lineage, full_cards, visited, lineage_issues = [], [], set(), []

            def visit(
                current,
                supplied=None,
                visited=visited,
                nodes=nodes,
                cutoff=cutoff,
                lineage_issues=lineage_issues,
                cell_id=cell_id,
                lineage=lineage,
                full_cards=full_cards,
            ):
                if current in visited:
                    return
                visited.add(current)
                prior = nodes[current]
                available = [x for x in prior["records"] if x["at"] <= cutoff]
                if not available and supplied is None:
                    lineage_issues.append(current + ": no recipe available at target plan time")
                    return
                state = supplied or available[-1]["card"]
                parents, issues = get_parents(current, state)
                lineage_issues.extend(current + ": " + issue for issue in issues)
                for parent in sorted(set(parents + data_dependencies(current, state))):
                    visit(parent)
                if cell_id + "/" + current in FIDELITY_EXCLUSIONS:
                    lineage_issues.append(current + ": audited recipe mismatch or incomplete setup")
                lineage.append({"card_id": current, "recipe": canonical(state)[0]})
                full_cards.append({"setup": state.get("setup", {})})

            visit(cid, card)
            parents, _issues = get_parents(cid, card)
            num["lineage_length"] = len(lineage)
            num["n_direct_parents"] = len(parents)
            ancestor_data = []
            for ancestor in lineage[:-1]:
                for item in ancestor["recipe"]["data"]:
                    if number(item.get("n_examples")) is not None:
                        ancestor_data.append(number(item["n_examples"]))
            if ancestor_data:
                num["ancestor_data_examples_sum"] = sum(ancestor_data)
                num["log10_ancestor_data_examples_sum"] = math.log10(max(sum(ancestor_data), 1))
            comp = (card.get("evaluation") or {}).get("comparator") or {}
            comparator = number(comp.get("value")) if isinstance(comp, dict) else None
            if comparator is not None and not 0 <= comparator <= 1:
                comparator = None
            parent_y = nodes[parents[0]]["y"] if len(parents) == 1 else None
            parent_local = None
            comparator_refs = (
                re.findall(r"exp[-_]?0*(\d+)", str(comp.get("ref", "")))
                if isinstance(comp, dict)
                else []
            )
            comparator_cid = (
                f"exp-{int(comparator_refs[0]):02d}" if len(comparator_refs) == 1 else None
            )
            if comparator_cid not in nodes or int(comparator_cid.split("-")[1]) >= int(
                cid.split("-")[1]
            ):
                comparator_cid = None
            if len(parents) == 1:
                avail = [x for x in nodes[parents[0]]["records"] if x["at"] < cutoff]
                for record in avail:
                    parent_local = unique_largest_measurement(record["card"])
                if comparator_cid == parents[0] and comparator is not None:
                    parent_local = comparator
            current_parent_path = normalize_path(
                ((card.get("setup") or {}).get("parent_checkpoint") or {}).get("path")
            )
            official_parent_path = (
                normalize_path(
                    (nodes[parents[0]]["final"].get("result") or {}).get("output_checkpoint")
                )
                if len(parents) == 1
                else ""
            )
            row = {
                "example_id": eid,
                "cell_id": cell_id,
                "card_id": cid,
                "scientist_model": meta["scientist_model"],
                "y": node["y"],
                "stderr": node["stderr"],
                "eligible": not reasons,
                "exclusion_reasons": reasons,
                "prospective": node["stage"] == "plan" and local_accuracy(card) is None,
                "first_stage": node["stage"],
                "first_submitted_at": cutoff,
                "first_missing": node["first_missing"],
                "recipe": canonical_card,
                "recipe_text": json.dumps(canonical_card, sort_keys=True),
                "lineage_text": "\n".join(
                    f"step {i + 1}: " + json.dumps(x["recipe"], sort_keys=True)
                    for i, x in enumerate(lineage)
                ),
                "plan_text": json.dumps(card.get("setup", {}), sort_keys=True),
                "lineage_plan_text": json.dumps(full_cards, sort_keys=True),
                "numeric_features": num,
                "categorical_features": cats,
                "lineage": lineage,
                "lineage_complete": not lineage_issues,
                "lineage_issues": sorted(set(lineage_issues)),
                "parent_ids": parents,
                "data_dependency_ids": data_dependencies(cid, card),
                "parent_accuracy": parent_y,
                "parent_local_accuracy": parent_local,
                "parent_accuracy_exact_match": bool(
                    current_parent_path and current_parent_path == official_parent_path
                ),
                "comparator_card_id": comparator_cid,
                "comparator_official_accuracy": nodes[comparator_cid]["y"]
                if comparator_cid
                else None,
                "comparator_accuracy": comparator,
                "local_accuracy": local_accuracy(node["final"]),
                "final_execution": (node["final"].get("result") or {}).get("execution"),
                "source_card": node["path"],
                "source_first_record_sha256": hashlib.sha256(
                    json.dumps(first, sort_keys=True).encode()
                ).hexdigest(),
            }
            rows.append(row)
    audit["counts"] = {
        "all_cards": len(rows),
        "labeled": sum(x["y"] is not None for x in rows),
        "prospective_labeled": sum(x["prospective"] and x["y"] is not None for x in rows),
        "eligible_labeled": sum(x["eligible"] and x["y"] is not None for x in rows),
        "eligible_labeled_complete_lineage": sum(
            x["eligible"] and x["y"] is not None and x["lineage_complete"] for x in rows
        ),
    }
    audit["exclusion_counts_labeled"] = dict(
        Counter(r for x in rows if x["y"] is not None for r in x["exclusion_reasons"])
    )
    audit["exclusions"] = [
        {"id": x["example_id"], "reasons": x["exclusion_reasons"]}
        for x in rows
        if x["exclusion_reasons"]
    ]
    audit["lineage_issues"] = [
        {"id": x["example_id"], "issues": x["lineage_issues"]} for x in rows if x["lineage_issues"]
    ]
    audit["limitations"] = [
        "Plan stage is a registration-time proxy, not independently verified launch time for every record.",
        "Strict canonical setup omits data-selection prose and implementation details; same canonical recipe need not mean same executed code.",
        "Final checkpoint labels may reflect undocumented packaging/selection; manual fidelity exclusions catch observed cases, not a proof for every checkpoint.",
        "Lineage uses weight parents plus explicitly named data-generation dependencies; implicit teacher/data dependencies can still be missing.",
        "Lineage completeness is at card granularity: intermediate checkpoints within a card do not have separate recipes or labels.",
        "parent_accuracy is an oracle: retrospective official parent labels are not guaranteed available at prediction time.",
        "plan_text contains prior observations and file naming style; it is a separate exploratory context arm, not a pure recipe feature.",
        "local_accuracy is the best reported largest-n postexecution score, possibly for another decode setting; a diagnostic ceiling only.",
        "Parent-local input uses the declared comparator when it names the parent; otherwise an unambiguous largest-n earlier measurement.",
        "Missing official labels do not imply experiment failure.",
    ]
    return rows, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/traj/raw/awm-gsm8k-trajectories")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis/outcome_prediction"))
    args = parser.parse_args()
    rows, audit = build(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "examples.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows)
    )
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {"counts": audit["counts"], "exclusions": audit["exclusion_counts_labeled"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
