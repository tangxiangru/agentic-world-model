"""Compact, outcome-blind features for clean one-step delta prediction.

``features(example, mode)`` expects an input-only example from the verified
``wm_one_step_data`` loader (or an equivalently screened deployment request):

* ``parent.accuracy`` is a finite numeric accuracy in [0, 1]. ``parent.kind`` is
  ``measured`` or ``published_base``. Legacy exports may instead identify a
  measured reference by ``reference_known=True`` and an archive ``path_kind``.
* ``model_input`` contains the current first-plan recipe and only code whose
  reconstructed status was checked against pre-proposal trace provenance.
* ``history`` is oldest-to-newest, excludes the current recipe, and includes
  entries with ``recipe_status=screened`` plus ``model_input`` or quarantined
  placeholders. The last entry denotes the immediate parent's recipe.

The adapter re-applies the positive-input projection and fixed numeric source
extractors. It does not certify timestamps, ancestry, archives, or score
provenance: the loader/caller owns those checks. It never reads identifiers,
target labels, or any ancestor scores. No script is executed or embedded.

The three fixed-width ablations have 3/72/151 columns for parent/current/history.
Missing settings become zero plus an observed mask, not a claimed zero setting.
History summarizes all supplied ancestors using ten core settings and seven
fixed method indicators. Quarantined recipes remain missing; latest never
backfills an older recipe. Decay has a fixed one-recipe half-life, normalized
over observed values, with intervening missing recipes retaining their distance.
Means/maxima summarize declarations, not actual completed training exposure.
No preprocessing is fitted here; scaling/model fitting must use training only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Real

from tools.outcome_prediction.wm_code_benchmark import positive_input
from tools.outcome_prediction.wm_code_data_features import extract_data_features
from tools.outcome_prediction.wm_code_features import extract_config_features

SCHEMA_VERSION = 1
MODES = ("parent", "current", "history")
CONFIG_FIELDS = (
    "learning_rate",
    "epochs",
    "effective_batch_per_device",
    "sequence_length",
    "max_steps",
    "warmup_ratio",
    "weight_decay",
    "lora_rank",
    "lora_alpha_per_rank",
    "completion_only",
    "assistant_only",
    "packing",
    "temperature",
    "top_p",
    "max_new_tokens",
    "num_generations",
    "kl_beta",
    "lora_present",
    "gradient_checkpointing",
    "entrypoint_available",
)
DATA_FIELDS = (
    "quota.requested_n",
    "mixture.quota_sum",
    "mixture.quota_entropy",
    "mixture.math_fraction",
    "mixture.augmented_fraction",
    "mixture.fresh_fraction",
    "cap.solutions_per_problem",
    "cap.max_tokens",
    "fewshot.fraction",
    "policy.evidence.correctness_filter",
    "policy.evidence.exclude_prior",
)
CURRENT_FIELDS = tuple("config." + key for key in CONFIG_FIELDS) + tuple(
    "data." + key for key in DATA_FIELDS
)
HISTORY_FIELDS = (
    "config.learning_rate",
    "config.epochs",
    "config.effective_batch_per_device",
    "config.sequence_length",
    "config.max_steps",
    "config.lora_rank",
    "config.completion_only",
    "data.quota.requested_n",
    "data.mixture.quota_sum",
    "data.policy.evidence.correctness_filter",
)
FAMILY_TOKENS = {
    "sft": frozenset({"sft", "supervised"}),
    "rft": frozenset({"rft"}),
    "rl": frozenset({"rl", "grpo", "ppo", "dpo"}),
    "merge": frozenset({"merge", "merging", "soup", "averaging"}),
    "decoding": frozenset({"decode", "decoding", "evaluation", "eval", "packaging"}),
    "distill": frozenset({"distill", "distillation"}),
}
FAMILY_KEYS = tuple("family." + name for name in (*FAMILY_TOKENS, "other"))


def _parent(example):
    parent = example.get("parent")
    if not isinstance(parent, Mapping):
        raise TypeError("A supplied parent reference is required")
    accuracy = parent.get("accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, Real):
        raise TypeError("Parent accuracy must be a finite numeric value in [0, 1]")
    try:
        accuracy = float(accuracy)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Parent accuracy must be finite") from exc
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Parent accuracy must be a finite numeric value in [0, 1]")
    if "reference_known" in parent and parent["reference_known"] is not True:
        raise ValueError("Unknown parent references cannot be encoded as measured scores")
    kind = parent.get("kind")
    if (
        kind is None
        and parent.get("reference_known") is True
        and parent.get("path_kind")
        in {
            "immutable_archive",
            "archived_source",
        }
    ):
        kind = "measured"
    if kind not in {"measured", "published_base"}:
        raise ValueError("Parent kind must be measured or published_base")
    return {
        "parent.accuracy": accuracy,
        "parent.reference.measured": float(kind == "measured"),
        "parent.reference.published_base": float(kind == "published_base"),
    }, kind


def _number(value, key):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Whitelist extractor returned a nonnumeric feature: " + key)
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Whitelist extractor returned a nonfinite feature: " + key) from exc
    if not math.isfinite(number):
        raise ValueError("Whitelist extractor returned a nonfinite feature: " + key)
    return number


def _step(model_input):
    if not isinstance(model_input, dict):
        raise TypeError("A screened recipe must supply a model_input dictionary")
    safe = positive_input({"model_input": model_input})
    config, _ = extract_config_features(safe)
    data, _ = extract_data_features(safe)
    result = {"config." + key: _number(config.get("codecfg." + key), key) for key in CONFIG_FIELDS}
    result.update({"data." + key: _number(data.get(key), key) for key in DATA_FIELDS})
    family = safe["plan"]["setup"]["method"].get("family")
    tokens = set(re.findall(r"[a-z]+", family.lower())) if isinstance(family, str) else set()
    result.update(
        {"family." + name: float(bool(tokens & allowed)) for name, allowed in FAMILY_TOKENS.items()}
    )
    result["family.other"] = float(not any(result[key] for key in FAMILY_KEYS[:-1]))
    return result


def _current(output, step):
    for key in CURRENT_FIELDS:
        value = step[key]
        output["current." + key] = 0.0 if value is None else value
        output["current.observed." + key] = float(value is not None)
    output.update({"current." + key: step[key] for key in FAMILY_KEYS})


def _history(output, example, kind):
    entries = example.get("history", [])
    if not isinstance(entries, (list, tuple)):
        raise TypeError("History must be an oldest-to-newest sequence")
    if kind == "published_base" and entries:
        raise ValueError("A published base reference must not have checkpoint ancestors")
    complete = example.get("history_complete_to_base", kind == "published_base")
    if not isinstance(complete, bool):
        raise TypeError("History completeness must be a boolean")
    steps = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Every history entry must declare its recipe status")
        status = entry.get("recipe_status")
        if status == "quarantined":
            steps.append(None)
        elif status == "screened":
            steps.append(_step(entry.get("model_input")))
        else:
            raise ValueError("History recipe status must be screened or quarantined")
    n, known = len(steps), sum(step is not None for step in steps)
    output.update(
        {
            "history.steps": float(n),
            "history.screened_steps": float(known),
            "history.screened_fraction": known / n if n else 0.0,
            "history.complete_to_base": float(complete),
            "history.latest_screened": float(bool(steps) and steps[-1] is not None),
        }
    )
    for key in HISTORY_FIELDS:
        observed = [
            (index, step[key])
            for index, step in enumerate(steps)
            if step is not None and step[key] is not None
        ]
        values = [value for _, value in observed]
        latest = steps[-1][key] if steps and steps[-1] is not None else None
        if observed:
            latest_observed_index = observed[-1][0]
            weights = [0.5 ** (latest_observed_index - index) for index, _ in observed]
            decay = math.fsum(weight * value for weight, value in zip(weights, values)) / math.fsum(
                weights
            )
        else:
            decay = 0.0
        output.update(
            {
                "history.mean." + key: math.fsum(values) / len(values) if values else 0.0,
                "history.max." + key: max(values) if values else 0.0,
                "history.latest." + key: 0.0 if latest is None else latest,
                "history.decay." + key: decay,
                "history.observed_fraction." + key: len(values) / n if n else 0.0,
                "history.latest_observed." + key: float(latest is not None),
            }
        )
    for key in FAMILY_KEYS:
        values = [step[key] for step in steps if step is not None]
        output["history.mean." + key] = math.fsum(values) / known if known else 0.0
        output["history.latest." + key] = steps[-1][key] if steps and steps[-1] is not None else 0.0


def features(example, mode="current") -> dict[str, float]:
    """Return only finite, fixed-schema scalars; fitting is deliberately external."""
    if mode not in MODES:
        raise ValueError("Feature mode must be parent, current, or history")
    if not isinstance(example, Mapping):
        raise TypeError("Example must be a dictionary-like mapping")
    output, kind = _parent(example)
    if mode != "parent":
        _current(output, _step(example.get("model_input")))
    if mode == "history":
        _history(output, example, kind)
    if any(not isinstance(value, float) or not math.isfinite(value) for value in output.values()):
        raise ValueError("Feature adapter produced a nonfinite or non-float output")
    return output
