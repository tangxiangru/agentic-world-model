"""Train-only model selection for reviewed complete recipes and final outcomes.

Only ``full_recipe_final_only_v1`` payloads are accepted. Training rows require a
content-bound approval plus the per-step reviews accepted by wm_recipe_input.
No ancestor measurements, legacy model_input, or test-set calibration is used.
Content approval is an external scientific audit, not manufactured by this file.

Three fixed variants are compared by four-fold scientist-run CV separately per
benchmark. All vocabulary, numeric scaling and dimensionality reduction are fit
inside each fold. Select by equal-run final-recipe selection regret, then MAE,
then fixed variant order. Fit only the champion on all approved training rows.
This is a small outcome predictor, not a causal simulator or a guarantee of a
five-percentage-point downstream improvement. No real dataset is fit on import.

Serving retains the six-key evidence config. Each candidate_payloads value is an
approved record with model_input and content_review, not a bare legacy payload.
Only its validated model_input reaches prediction. Model serialization is trusted
local pickle; a matching hash is an integrity check, not protection from an
untrusted publisher's pickle. Keep training manifests and labels off the gateway.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tools.outcome_prediction import wm_recipe_input as recipe_input

SEED = 20260905
VARIANTS = ("text_ridge", "structured_trees", "hybrid_trees")
SPEC = {
    "schema": recipe_input.SCHEMA,
    "variants": list(VARIANTS),
    "cv_folds": 4,
    "selection": "equal_run_full_recipe_selection_regret_then_equal_run_mae_then_variant_order",
    "tfidf": {"ngram_range": [1, 2], "max_features": 20000, "sublinear_tf": True},
    "ridge_alpha": 10.0,
    "trees": {"n_estimators": 128, "min_samples_leaf": 2, "max_features": 1.0},
    "svd_components": 32,
    "seed": SEED,
    "sample_weight": "equal_total_per_scientist_run_normalized_to_mean_one",
    "target": "final_official_accuracy_of_complete_recipe",
}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TOKEN_PATTERN = r"(?u)(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][\w.:-]*)"
NUMERIC_SETTING = re.compile(
    rf"(?P<key>--[\w-]+|\b[A-Za-z_]\w*)\s*(?:=|\s)\s*(?P<number>{NUMBER})(?![\w.])"
)


def digest(value):
    return recipe_input.digest(value)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def approved_payload(row):
    """Require externally approved, exact whole-payload and per-step hashes."""
    if not isinstance(row, dict):
        raise TypeError("Approved candidate must be a record object")
    review = row.get("content_review")
    payload = row.get("model_input")
    if not isinstance(review, dict) or set(review) != {"status", "payload_sha256", "step_reviews"}:
        raise ValueError("A content-bound whole-recipe review is required")
    if review["status"] != "approved":
        raise ValueError("Recipe content review is not approved")
    recipe_input.validate_full_recipe(payload)
    if review["payload_sha256"] != digest(payload):
        raise ValueError("Recipe changed after content review")
    return recipe_input.build_reviewed_input(
        task=payload["task"],
        steps=payload["recipe"]["steps"],
        final_step_id=payload["recipe"]["final_step_id"],
        reviews=review["step_reviews"],
    )


def canonical_payload(payload):
    """Normalize explicit DAG IDs, never replace coincident numeric code/settings."""
    recipe_input.validate_full_recipe(payload)
    names = {
        step["step_id"]: f"step_{index:03d}"
        for index, step in enumerate(payload["recipe"]["steps"])
    }
    canonical = copy.deepcopy(payload)
    for step in canonical["recipe"]["steps"]:
        step["step_id"] = names[step["step_id"]]
        for edge in step["parents"]:
            edge["step_id"] = names[edge["step_id"]]
    canonical["recipe"]["final_step_id"] = names[canonical["recipe"]["final_step_id"]]

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return re.sub(r"(?<![\w])(?:aime-)?r0-\d+\b", "scientist_run", value)
        return value

    return clean(canonical)


def feature_document(payload):
    """All approved scientific text/code and numeric settings stay in the document."""
    return json.dumps(
        canonical_payload(payload), sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def structured_features(payload):
    """Ordered numeric/categorical setup, code defaults and explicit DAG features.

    Missing numeric settings are represented by absence plus a separate presence
    feature; an explicit zero is not missing. Free prose/code stays in TF-IDF.
    """
    canonical = canonical_payload(payload)
    steps = canonical["recipe"]["steps"]
    features = {"n_steps": float(len(steps))}
    depths = {}

    def numeric(path, value):
        value = float(value)
        if math.isfinite(value):
            features[path + ".present"] = 1.0
            features[path] = value

    def flatten(value, path):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                flatten(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flatten(item, f"{path}[{index}]")
        elif type(value) in (int, float):
            numeric(path, value)
        elif isinstance(value, bool):
            features[f"{path}={str(value).lower()}"] = 1.0
        elif isinstance(value, str):
            if re.fullmatch(NUMBER, value.strip()):
                numeric(path, value)
            elif len(value) <= 100 and not any(c.isspace() for c in value):
                features[f"{path}={value}"] = 1.0
            for index, match in enumerate(NUMERIC_SETTING.finditer(value)):
                numeric(f"{path}.setting[{index}].{match['key']}", match["number"])

    for index, step in enumerate(steps):
        prefix = f"step[{index:03d}]"
        features[f"{prefix}.role={step['role']}"] = 1.0
        features[f"{prefix}.n_parents"] = float(len(step["parents"]))
        depth = max((depths[edge["step_id"]] + 1 for edge in step["parents"]), default=0)
        depths[step["step_id"]] = depth
        features[f"{prefix}.depth"] = float(depth)
        for edge in step["parents"]:
            key = f"{prefix}.parent.{edge['kind']}={edge['step_id']}"
            features[key] = features.get(key, 0.0) + 1.0
        flatten(step["plan"], prefix + ".plan")
        flatten(step["code"], prefix + ".code")
    features["final_depth"] = float(depths[steps[-1]["step_id"]])
    return features


def _vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2), max_features=20000, sublinear_tf=True, token_pattern=TOKEN_PATTERN
    )


def _weights(rows):
    counts = Counter(row["cell_id"] for row in rows)
    return np.array([len(rows) / (len(counts) * counts[row["cell_id"]]) for row in rows])


class _Variant:
    """All learned preprocessing is owned by this one fold/model instance."""

    def __init__(self, name):
        if name not in VARIANTS:
            raise ValueError("Unknown fixed model variant")
        self.name = name

    def fit(self, payloads, targets, weights):
        if self.name != "structured_trees":
            self.text = _vectorizer()
            text_x = self.text.fit_transform([feature_document(p) for p in payloads])
        if self.name != "text_ridge":
            self.structured = DictVectorizer(sparse=True)
            numeric_x = self.structured.fit_transform([structured_features(p) for p in payloads])
            self.scaler = StandardScaler(with_mean=False)
            numeric_x = self.scaler.fit_transform(numeric_x, sample_weight=weights)
        if self.name == "text_ridge":
            features = text_x
            self.regressor = Ridge(alpha=SPEC["ridge_alpha"], solver="lsqr")
        else:
            features = numeric_x
            if self.name == "hybrid_trees":
                components = max(
                    1, min(SPEC["svd_components"], text_x.shape[0] - 1, text_x.shape[1] - 1)
                )
                self.svd = TruncatedSVD(n_components=components, random_state=SEED)
                text_low_rank = self.svd.fit_transform(text_x)
                features = sparse.hstack(
                    [numeric_x, sparse.csr_matrix(text_low_rank)], format="csr"
                )
            self.regressor = ExtraTreesRegressor(**SPEC["trees"], random_state=SEED, n_jobs=1)
        self.regressor.fit(features, targets, sample_weight=weights)
        return self

    def transform(self, payloads):
        if self.name == "text_ridge":
            return self.text.transform([feature_document(p) for p in payloads])
        numeric_x = self.scaler.transform(
            self.structured.transform([structured_features(p) for p in payloads])
        )
        if self.name == "structured_trees":
            return numeric_x
        text_x = self.text.transform([feature_document(p) for p in payloads])
        return sparse.hstack(
            [numeric_x, sparse.csr_matrix(self.svd.transform(text_x))], format="csr"
        )

    def predict(self, payloads):
        return np.clip(self.regressor.predict(self.transform(payloads)), 0, 1)

    def feature_audit(self):
        return {
            "variant": self.name,
            "text_vocabulary_sha256": digest(
                {key: int(value) for key, value in self.text.vocabulary_.items()}
            )
            if hasattr(self, "text")
            else None,
            "structured_features_sha256": digest(self.structured.get_feature_names_out().tolist())
            if hasattr(self, "structured")
            else None,
            "svd_components": self.svd.n_components if hasattr(self, "svd") else None,
        }


def _selection_metrics(rows, targets, predictions):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["cell_id"]].append(index)
    regrets, errors, squared_errors = [], [], []
    for indices in groups.values():
        selected = min(indices, key=lambda i: (-float(predictions[i]), rows[i]["example_id"]))
        regrets.append(max(float(targets[i]) for i in indices) - float(targets[selected]))
        errors.append(float(np.mean([abs(targets[i] - predictions[i]) for i in indices])))
        squared_errors.append(float(np.mean([(targets[i] - predictions[i]) ** 2 for i in indices])))
    return {
        "run_macro_selection_regret": float(np.mean(regrets)),
        "run_macro_mae": float(np.mean(errors)),
        "run_macro_rmse": float(np.sqrt(np.mean(squared_errors))),
        "scientist_runs": len(groups),
        "multi_recipe_runs": sum(len(v) > 1 for v in groups.values()),
    }


def _folds(rows):
    cells = sorted({r["cell_id"] for r in rows}, key=lambda cell: _sha(f"{SEED}:{cell}".encode()))
    if len(cells) < SPEC["cv_folds"]:
        raise ValueError("Each benchmark requires at least four approved training scientist runs")
    return {cell: index % SPEC["cv_folds"] for index, cell in enumerate(cells)}


def _select_training_rows(rows, split):
    if not isinstance(split, dict):
        raise TypeError("Split must be an explicit cell partition")
    train_cells, test_cells = split.get("train_cell_ids"), split.get("test_cell_ids")
    for cells in (train_cells, test_cells):
        if (
            not isinstance(cells, list)
            or not cells
            or any(not isinstance(c, str) or not c for c in cells)
            or len(cells) != len(set(cells))
        ):
            raise ValueError("Both split partitions need distinct nonempty cell IDs")
    train_cells, test_cells = set(train_cells), set(test_cells)
    if train_cells & test_cells:
        raise ValueError("Training and test cells overlap")
    empty_cells = split.get("empty_train_cell_ids", [])
    if (
        not isinstance(empty_cells, list)
        or any(not isinstance(cell, str) for cell in empty_cells)
        or len(empty_cells) != len(set(empty_cells))
        or not set(empty_cells) <= train_cells
    ):
        raise ValueError("Explicit empty training cells must be a unique subset of TRAIN")
    empty_cells = set(empty_cells)
    selected = []
    for row in rows:
        # Critical: do not inspect ID, label, payload, review, eligibility or any
        # feature on a held-out row. A poison-mapping test enforces this ordering.
        cell = row["cell_id"]
        if cell in train_cells:
            selected.append(row)
        elif cell not in test_cells:
            raise ValueError("Row belongs to neither frozen split partition")
    if {row["cell_id"] for row in selected} != train_cells - empty_cells:
        raise ValueError("Approved training rows do not cover the declared training cells")
    if not selected:
        raise ValueError("No approved training recipes remain")
    return selected, train_cells, test_cells, empty_cells


def feature_contract_hash():
    return digest(
        {
            "spec": SPEC,
            "number_pattern": NUMBER,
            "token_pattern": TOKEN_PATTERN,
            "numeric_setting_pattern": NUMERIC_SETTING.pattern,
            "outcome_fields": sorted(recipe_input.OUTCOME_FIELDS),
            "sources": {
                obj.__name__: inspect.getsource(obj)
                for obj in (
                    approved_payload,
                    canonical_payload,
                    feature_document,
                    structured_features,
                    _vectorizer,
                    _weights,
                    _Variant,
                    _selection_metrics,
                    _folds,
                    _select_training_rows,
                    recipe_input.validate_full_recipe,
                    recipe_input.build_reviewed_input,
                    recipe_input._structural_scan,
                    FinalRecipeWorldModel.fit,
                    FinalRecipeWorldModel.predict,
                )
            },
        }
    )


class FinalRecipeWorldModel:
    """A train-CV selected per-benchmark model; never learns from held-out inputs."""

    def __init__(self):
        self.models = {}
        self.training_manifest = None

    def fit(self, rows, split):
        selected, train_cells, test_cells, empty_cells = _select_training_rows(rows, split)
        grouped, provenance, seen = defaultdict(list), [], set()
        for row in selected:
            example_id = row["example_id"]
            if (
                not isinstance(example_id, str)
                or not example_id.startswith(row["cell_id"] + "/")
                or example_id in seen
            ):
                raise ValueError("Invalid or duplicate training target ID")
            seen.add(example_id)
            payload = approved_payload(row)
            label = row.get("label")
            target = label.get("accuracy") if isinstance(label, dict) else None
            if (
                type(target) not in (int, float)
                or not math.isfinite(target)
                or not 0 <= target <= 1
            ):
                raise ValueError(
                    "Approved training recipe needs a final official accuracy in [0,1]"
                )
            benchmark = payload["task"]["benchmark"]
            grouped[benchmark].append((row, payload, float(target)))
            provenance.append(
                {
                    "example_id": example_id,
                    "cell_id": row["cell_id"],
                    "benchmark": benchmark,
                    "payload_sha256": digest(payload),
                    "content_review_sha256": digest(row["content_review"]),
                    "final_official_accuracy": float(target),
                }
            )
        cell_benchmarks = defaultdict(set)
        for benchmark, items in grouped.items():
            for row, _, _ in items:
                cell_benchmarks[row["cell_id"]].add(benchmark)
        if any(len(tasks) != 1 for tasks in cell_benchmarks.values()):
            raise ValueError("One scientist run cannot span benchmark strata")
        fitted, reports = {}, {}
        for benchmark, items in sorted(grouped.items()):
            items.sort(key=lambda item: item[0]["example_id"])
            bench_rows, payloads, y = zip(*items)
            y = np.asarray(y)
            folds = _folds(bench_rows)
            predictions = {name: np.full(len(items), np.nan) for name in VARIANTS}
            fold_audits = []
            for fold in range(SPEC["cv_folds"]):
                training = [i for i, row in enumerate(bench_rows) if folds[row["cell_id"]] != fold]
                validation = [
                    i for i, row in enumerate(bench_rows) if folds[row["cell_id"]] == fold
                ]
                train_payloads = [payloads[i] for i in training]
                val_payloads = [payloads[i] for i in validation]
                weights = _weights([bench_rows[i] for i in training])
                for name in VARIANTS:
                    model = _Variant(name).fit(train_payloads, y[training], weights)
                    predictions[name][validation] = model.predict(val_payloads)
                    fold_audits.append(
                        {
                            "fold": fold,
                            "variant": name,
                            "fit_example_ids": [bench_rows[i]["example_id"] for i in training],
                            "validation_example_ids": [
                                bench_rows[i]["example_id"] for i in validation
                            ],
                            "features": model.feature_audit(),
                        }
                    )
            metrics = {
                name: _selection_metrics(bench_rows, y, predictions[name]) for name in VARIANTS
            }
            champion = min(
                VARIANTS,
                key=lambda name: (
                    metrics[name]["run_macro_selection_regret"],
                    metrics[name]["run_macro_mae"],
                    VARIANTS.index(name),
                ),
            )
            final_model = _Variant(champion).fit(payloads, y, _weights(bench_rows))
            retrieval = _vectorizer()
            retrieval_x = retrieval.fit_transform([feature_document(p) for p in payloads])
            examples = [
                {
                    "example_id": row["example_id"],
                    "cell_id": row["cell_id"],
                    "final_official_accuracy": float(target),
                }
                for row, _, target in items
            ]
            fitted[benchmark] = {
                "model": final_model,
                "retrieval": retrieval,
                "retrieval_x": retrieval_x,
                "examples": examples,
                "cv_rmse": metrics[champion]["run_macro_rmse"],
            }
            reports[benchmark] = {
                "champion": champion,
                "candidate_metrics": metrics,
                "fold_by_cell": folds,
                "fold_audits": fold_audits,
                "final_features": final_model.feature_audit(),
                "oof_predictions": {name: values.tolist() for name, values in predictions.items()},
                "ordered_training_ids": [row["example_id"] for row in bench_rows],
                "selection_scope": "training grouped CV only; not held-out performance",
            }
        provenance.sort(key=lambda row: row["example_id"])
        self.models = fitted
        self.training_manifest = {
            "schema_version": 1,
            "spec": copy.deepcopy(SPEC),
            "feature_contract_sha256": feature_contract_hash(),
            "train_cell_ids": sorted(train_cells),
            "fitted_train_cell_ids": sorted(train_cells - empty_cells),
            "empty_train_cell_ids": sorted(empty_cells),
            "forbidden_cell_ids": sorted(test_cells),
            "training_examples": provenance,
            "training_data_sha256": digest(provenance),
            "benchmark_train_cv": reports,
            "test_data_used": False,
            "content_review_scope": "Externally approved hashes; not proof of unobserved causal execution",
            "five_percentage_point_improvement_demonstrated": False,
        }
        return self

    def predict(self, payload, neighbors=3):
        recipe_input.validate_full_recipe(payload)
        if type(neighbors) is not int or not 0 <= neighbors <= 10:
            raise ValueError("Neighbor count must be an integer between zero and ten")
        benchmark = payload["task"]["benchmark"]
        if self.training_manifest is None or benchmark not in self.models:
            raise ValueError("No trained model for this benchmark")
        bundle = self.models[benchmark]
        model = bundle["model"]
        x = model.transform([payload])
        prediction = float(np.clip(model.regressor.predict(x)[0], 0, 1))
        if hasattr(model.regressor, "estimators_"):
            diagnostic = {
                "kind": "tree_disagreement_std",
                "value": float(
                    np.std([tree.predict(x)[0] for tree in model.regressor.estimators_])
                ),
            }
        else:
            diagnostic = {"kind": "training_grouped_cv_residual_rmse", "value": bundle["cv_rmse"]}
        diagnostic["calibrated"] = False
        similarities = (
            (bundle["retrieval_x"] @ bundle["retrieval"].transform([feature_document(payload)]).T)
            .toarray()
            .ravel()
        )
        order = sorted(range(len(similarities)), key=lambda i: (-float(similarities[i]), i))[
            :neighbors
        ]
        return {
            "predicted_official_accuracy": prediction,
            "benchmark": benchmark,
            "diagnostic_uncertainty": diagnostic,
            "nearest_training_examples": [
                {**bundle["examples"][i], "text_similarity": float(similarities[i])} for i in order
            ],
            "training_data_sha256": self.training_manifest["training_data_sha256"],
            "warnings": [
                "Final full-recipe forecast, not an observed outcome or causal guarantee.",
                "Diagnostic uncertainty is not calibrated.",
                "Neighbors contain training-recipe final labels only; the caller must keep query runs disjoint from training.",
            ],
        }

    def save(self, path):
        if self.training_manifest is None:
            raise ValueError("Cannot save an unfitted final-recipe model")
        path = Path(path)
        with path.open("xb") as handle:
            joblib.dump(self, handle)
        return _sha(path.read_bytes())

    @classmethod
    def load(cls, path, *, expected_sha256):
        raw = Path(path).read_bytes()
        if _sha(raw) != expected_sha256:
            raise ValueError("Model artifact hash mismatch")
        model = joblib.load(io.BytesIO(raw))
        if not isinstance(model, cls) or model.training_manifest is None:
            raise ValueError("Not a fitted final-recipe model")
        if model.training_manifest.get("feature_contract_sha256") != feature_contract_hash():
            raise ValueError("Model source/feature contract has changed")
        return model


def train(rows, split):
    return FinalRecipeWorldModel().fit(rows, split)


def serve(config):
    from tools.outcome_prediction.wm_evidence import EvidenceStore, serve_stdio

    if not isinstance(config, dict) or set(config) != {
        "model_path",
        "model_sha256",
        "candidate_payloads",
        "evidence_root",
        "files",
        "audit_log",
    }:
        raise ValueError("Unexpected final-recipe server configuration")
    model = FinalRecipeWorldModel.load(config["model_path"], expected_sha256=config["model_sha256"])
    candidates = config["candidate_payloads"]
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Serving requires explicitly approved candidate records")
    payloads = {name: approved_payload(row) for name, row in candidates.items()}
    with EvidenceStore(
        config["evidence_root"],
        config["files"],
        audit_log=config["audit_log"],
        candidates=tuple(payloads),
        predict=lambda name: model.predict(payloads[name]),
    ) as store:
        serve_stdio(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--train-records", type=Path, required=True)
    trainer.add_argument("--split", type=Path, required=True)
    trainer.add_argument("--output", type=Path, required=True)
    server = commands.add_parser("serve")
    server.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        serve(json.loads(args.config.read_text()))
        return
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing model output directory")
    split = json.loads(args.split.read_text())
    rows = [
        json.loads(line) for line in args.train_records.read_text().splitlines() if line.strip()
    ]
    if any(row["cell_id"] not in set(split["train_cell_ids"]) for row in rows):
        raise ValueError("--train-records must contain training cells only")
    model = train(rows, split)
    args.output.mkdir(parents=True, exist_ok=False)
    model_hash = model.save(args.output / "wm.joblib")
    manifest = {
        **model.training_manifest,
        "model_sha256": model_hash,
        "source_sha256": _sha(Path(__file__).read_bytes()),
        "train_records_sha256": _sha(args.train_records.read_bytes()),
        "split_sha256": _sha(args.split.read_bytes()),
    }
    (args.output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "model_sha256": model_hash,
                "train_examples": len(model.training_manifest["training_examples"]),
                "evaluation_scope": "training CV only; no held-out predictions",
            }
        )
    )


if __name__ == "__main__":
    # Use the canonical module for both serialization and type checks. Running
    # this file with -m otherwise defines a second class under __main__, which
    # rejects models saved through the imported training API.
    from tools.outcome_prediction.wm_final_model import main as canonical_main

    canonical_main()
