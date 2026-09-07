"""Positive pre-proposal features for the broad, exploratory predictor study.

The legacy comparison cohort/folds are reproduced, not claimed prospectively
certified. Base references are legacy constants, NOT verified measurements.
Own child labels stay outside feature construction. Generic data dependencies
never supply scores; some archived chains contain the target itself.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.outcome_prediction.wm_grade_inventory import valid_official_label

SEED = 20260905
BASE_REFERENCES = {"gsm8k": 0.045, "aime2025": 0.05}
HP_ALIASES = {
    "lr": ("lr", "learning_rate"),
    "epochs": ("epochs", "num_train_epochs"),
    "batch_size": ("batch_size", "per_device_train_batch_size"),
    "grad_accum": ("grad_accum", "gradient_accumulation_steps"),
    "max_seq_len": ("max_seq_len", "max_seq_length", "max_length"),
    "max_steps": ("max_steps",),
    "warmup_ratio": ("warmup", "warmup_ratio"),
    "warmup_steps": ("warmup_steps",),
    "weight_decay": ("weight_decay",),
    "lora_r": ("lora_r", "r"),
    "lora_alpha": ("lora_alpha",),
    "lora_dropout": ("lora_dropout",),
    "temperature": ("temperature",),
    "top_p": ("top_p",),
    "top_k": ("top_k",),
    "max_tokens": ("max_tokens", "max_new_tokens"),
    "repetition_penalty": ("repetition_penalty",),
    "beta": ("beta",),
    "seed": ("seed",),
}
DATASETS = (
    "gsm8k",
    "openmathinstruct",
    "openmathreasoning",
    "openr1",
    "metamath",
    "numina",
    "math",
    "aime",
    "synthetic",
    "derived",
    "deepseek",
    "orca",
)
FAMILIES = ("sft", "rft", "grpo", "ppo", "dpo", "rl", "distill", "merge", "decoding", "evaluation")
CODE_NAMES = (
    "SFTTrainer",
    "GRPOTrainer",
    "DPOTrainer",
    "LoraConfig",
    "get_peft_model",
    "train_on_responses_only",
    "DataCollatorForCompletionOnlyLM",
    "DataCollatorForLanguageModeling",
    "apply_chat_template",
    "load_dataset",
    "generate",
    "set_seed",
    "gradient_checkpointing_enable",
    "save_pretrained",
)
FLAGS = (
    "completion_only",
    "assistant_only",
    "packing",
    "gradient_checkpointing",
    "flash_attention",
    "chat_template",
    "do_sample",
    "greedy",
    "eos_token",
    "pad_token",
    "boxed",
    "answer:",
    "####",
    "<end_of_turn>",
    "think",
)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not NUMBER.fullmatch(value.strip()):
            return None
        value = float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def stamp(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("Naive proposal time")
    return t


def folds(rows, k=8):
    """Exactly reproduce the legacy session/scientist-stratified assignment."""
    groups = defaultdict(list)
    for cell, scientist in sorted({(r["cell_id"], r["scientist_model"]) for r in rows}):
        groups[scientist].append(cell)
    rng, result, counter = random.Random(SEED), {}, 0
    for scientist in sorted(groups):
        group = groups[scientist]
        rng.shuffle(group)
        for cell in group:
            result[cell] = counter % k
            counter += 1
    return result


def cohort(cards):
    by_id = {r["example_id"]: r for r in cards}
    if len(by_id) != len(cards):
        raise ValueError("Duplicate card ID")
    selected, excluded = [], []
    for r in cards:
        reason = None
        if r["official_accuracy"] is None:
            reason = "missing_child_label"
        elif not r["lineage_complete"]:
            reason = "incomplete_legacy_lineage"
        parent_ids = [r["cell_id"] + "/" + p for p in r["parents"]]
        if r["example_id"] in parent_ids:
            raise ValueError("Self parent would expose child target")
        values = [
            by_id[p]["official_accuracy"]
            for p in parent_ids
            if p in by_id and by_id[p]["official_accuracy"] is not None
        ]
        if not values and parent_ids:
            reason = reason or "missing_parent_label"
        if reason:
            excluded.append({"example_id": r["example_id"], "reason": reason})
            continue
        reference = float(np.mean(values)) if values else BASE_REFERENCES[r["benchmark"]]
        selected.append(
            {
                "example_id": r["example_id"],
                "cell_id": r["cell_id"],
                "scientist_model": r["scientist_model"],
                "benchmark": r["benchmark"],
                "parent_ids": parent_ids,
                "parent_reference": reference,
                "reference_source": "legacy_base_constant_not_verified"
                if not parent_ids
                else (
                    "declared_parent_card_official"
                    if len(parent_ids) == 1
                    else "mean_declared_parent_scores"
                ),
                "from_base": not parent_ids,
                "audit": {
                    "recorded_plan_stage": r["plan_stage_recorded"],
                    "setup_changed": r["setup_changed"],
                    "n_parents": len(parent_ids),
                },
            }
        )
    for benchmark in BASE_REFERENCES:
        rs = [r for r in selected if r["benchmark"] == benchmark]
        assignment = folds(rs)
        for r in rs:
            r["fold"] = assignment[r["cell_id"]]
    return selected, excluded


def proposal_available(inventory_row):
    return inventory_row.get("first_stage", "plan") == "plan" and "first_result_not_empty" not in (
        inventory_row.get("audit", {}).get("reasons", [])
    )


def step_features(inventory_row, *, completed_history=False):
    """Whitelist only first-plan settings and timestamp-reconstructed syntax.

    A current candidate first recorded only after closing supplies no proposal
    features. Counts/generation yields and free-form ``other`` text are excluded.
    Completed ancestor recipes may be known even if first logged at closing.
    Numeric code literals are admitted only for named training settings.
    """
    if not proposal_available(inventory_row) and not completed_history:
        return {"proposal.available": 0.0}
    source = inventory_row["model_input"]
    setup = source["plan"].get("setup") or {}
    method = setup.get("method") or {}
    hp = method.get("hyperparams") or {}
    family = str(method.get("family") or "").lower().replace("-", "_")
    family = next((f for f in FAMILIES if f in family), "other")
    result = {"family." + f: float(f == family) for f in (*FAMILIES, "other")}
    result["proposal.available"] = 1.0
    for key, aliases in HP_ALIASES.items():
        values = [numeric(hp.get(alias)) for alias in aliases]
        result["hp." + key] = next((v for v in values if v is not None), None)
    for field, allowed in {
        "precision": ("bf16", "bfloat16", "fp16", "float16", "fp32"),
        "scheduler": ("cosine", "linear", "constant", "constant_with_warmup"),
    }.items():
        value = str(hp.get(field) or "").lower().strip()
        for allowed_value in allowed:
            result[field + "." + allowed_value] = float(value == allowed_value)
    result["peft.lora"] = float("lora" in str(method.get("peft") or "").lower())
    data = [d for d in setup.get("data") or [] if isinstance(d, dict)]
    result["data.sources"] = float(len(data))
    result["data.max_mixture_weight"] = max(
        (v for v in (numeric(d.get("mixture_weight")) for d in data) if v is not None),
        default=None,
    )
    source_text = " ".join(str(d.get("source") or "") for d in data).lower()
    for category in DATASETS:
        result["dataset." + category] = float(category in source_text)
    mechanism_text = " ".join(str(method.get(k) or "") for k in ("framework", "target_format"))
    mechanism_text = mechanism_text.lower()
    for flag in FLAGS:
        result["plan.flag." + flag] = float(flag in mechanism_text)
    result["planned_hours"] = numeric((setup.get("budget") or {}).get("planned_h"))
    names, parsed, unavailable, code_values = set(), 0, 0, defaultdict(list)
    alias_key = {alias: key for key, aliases in HP_ALIASES.items() for alias in aliases}
    for script in source.get("code") or []:
        if script.get("status") != "reconstructed" or not isinstance(script.get("content"), str):
            unavailable += 1
            continue
        try:
            tree = ast.parse(script["content"])
        except (SyntaxError, ValueError):
            unavailable += 1
            continue
        parsed += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            assignments = []
            if isinstance(node, ast.keyword):
                assignments.append((node.arg, node.value))
            elif isinstance(node, ast.Assign):
                assignments.extend(
                    (t.id.lower(), node.value) for t in node.targets if isinstance(t, ast.Name)
                )
            for key, expression in assignments:
                if key in alias_key and isinstance(expression, ast.Constant):
                    value = numeric(expression.value)
                    if value is not None:
                        code_values[alias_key[key]].append(value)
    for name in CODE_NAMES:
        result["code.calls." + name] = float(name in names)
    result["code.parsed_scripts"] = float(parsed)
    result["code.unavailable_scripts"] = float(unavailable)
    for key in HP_ALIASES:
        values = code_values[key]
        result["code.setting." + key] = float(np.median(values)) if values else None
        result["code.setting_conflict." + key] = float(len(set(values)) > 1)
    bs, ga = result["hp.batch_size"], result["hp.grad_accum"]
    effective = bs * ga if bs is not None and ga is not None and bs > 0 and ga > 0 else None
    result["dose.effective_batch"] = effective
    lr, length = (result["hp." + k] for k in ("lr", "max_seq_len"))
    steps = result["hp.max_steps"]
    steps = steps if steps is not None and steps > 0 else None
    exposure = effective * steps if effective is not None and steps is not None else None
    result["dose.planned_max_steps"] = steps
    result["dose.planned_max_examples"] = exposure
    result["dose.lr_times_steps"] = lr * steps if lr is not None and steps is not None else None
    result["dose.estimated_tokens"] = exposure * length if exposure is not None and length else None
    # Compress positive scales without silently parsing arbitrary free prose.
    for key in list(result):
        value = result[key]
        if value is not None and key.startswith("dose."):
            result[key] = math.log1p(max(value, 0))
    return result


def combine(current, parents, history, reference, from_base, view):
    if view not in ("current", "parent", "history", "reference"):
        raise ValueError("Unknown context view")
    out = {"parent_reference": reference, "base_reference_is_assumed": float(from_base)}
    if view == "reference":
        return out
    out.update({"current." + k: v for k, v in current.items()})
    chosen = [] if view == "current" else (parents if view == "parent" else history)
    if view != "current":
        out["history.count"] = float(len(chosen))
        for relation in ("weights", "data"):
            out["history.role." + relation] = float(sum(s["relation"] == relation for s in chosen))
        keys = sorted({k for s in chosen for k in s["features"]})
        for key in keys:
            observed = [(i, s["features"].get(key)) for i, s in enumerate(chosen)]
            observed = [(i, v) for i, v in observed if v is not None]
            if not observed:
                continue
            vals = [v for _, v in observed]
            weights = [0.5 ** (observed[-1][0] - i) for i, _ in observed]
            out["history.mean." + key] = float(np.mean(vals))
            out["history.recent." + key] = vals[-1]
            out["history.decayed." + key] = float(np.average(vals, weights=weights))
            if current.get(key) is not None:
                out["action_minus_recent." + key] = current[key] - vals[-1]
    return out


def safe_text(current, history):
    """Canonical semantic summaries, not raw prose or arbitrary source code."""

    def describe(features):
        return "; ".join(
            k.replace(".", " ").replace("_", " ") + " = " + format(v, ".5g")
            for k, v in sorted(features.items())
            if v is not None and v != 0
        )

    parts = ["Proposed experiment: " + describe(current)]
    for number, entry in enumerate(reversed(history), 1):
        parts.append(f"Prior step {number} ({entry['relation']}): " + describe(entry["features"]))
    return "\n".join(parts)


def build_examples(cards, inventory, selected):
    by_id, memo = {r["example_id"]: r for r in cards}, {}

    def step(identifier, completed_history=False):
        key = (identifier, completed_history)
        if key not in memo:
            memo[key] = step_features(inventory[identifier], completed_history=completed_history)
        return memo[key]

    examples, labels = [], {}
    for row in selected:
        identifier, card = row["example_id"], by_id[row["example_id"]]
        inv = inventory[identifier]
        if not valid_official_label(inv.get("label")):
            raise ValueError("Invalid official label: " + identifier)
        target = inv["label"]["accuracy"]
        if abs(target - card["official_accuracy"]) > 1e-12:
            raise ValueError("Legacy / official label mismatch: " + identifier)
        labels[identifier] = target
        cutoff = stamp(inv["first_submitted_at"])
        history, omitted, seen = [], [], set()
        for entry in card["recipe"][:-1]:
            ancestor = card["cell_id"] + "/" + entry["card_id"]
            if ancestor == identifier or ancestor in seen:
                omitted.append({"id": ancestor, "reason": "self_or_duplicate_dependency"})
                continue
            seen.add(ancestor)
            if (
                ancestor not in inventory
                or stamp(inventory[ancestor]["first_submitted_at"]) >= cutoff
            ):
                omitted.append({"id": ancestor, "reason": "not_available_before_proposal"})
                continue
            history.append(
                {
                    "id": ancestor,
                    "relation": entry["relation"],
                    "features": step(ancestor, completed_history=True),
                }
            )
        history.sort(key=lambda s: (inventory[s["id"]]["first_submitted_at"], s["id"]))
        parents = [s for s in history if s["id"] in row["parent_ids"]]
        current = step(identifier)
        views = {
            view: combine(
                current, parents, history, row["parent_reference"], row["from_base"], view
            )
            for view in ("reference", "current", "parent", "history")
        }
        examples.append(
            {
                **row,
                "views": views,
                "embedding_text": safe_text(current, history),
                "history_ids": [s["id"] for s in history],
                "audit": {
                    **row["audit"],
                    "omitted_history": omitted,
                    "initial_fidelity_eligible": inv["initial_eligible"],
                    "fidelity_reviewed_eligible": inv["eligible"],
                    "proposal_features_available": proposal_available(inv),
                    "history_scores_used": False,
                },
            }
        )
    return examples, labels
