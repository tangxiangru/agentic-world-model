"""Train-only recipe-to-outcome WM used by a matched RPM agent ablation.

This is a learned outcome predictor, not a simulator or a newly trained task LLM.
Each benchmark gets one fixed text/context regressor. No test-based model search
or calibration occurs here. Serving accepts fixed, public candidate payloads only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_SPEC = {
    "family": "per-benchmark TF-IDF plus ExtraTrees outcome regressor",
    "ngram_range": [1, 2],
    "max_features": 20000,
    "n_estimators": 300,
    "min_samples_leaf": 4,
    "max_features_tree": 1.0,
    "random_state": 20260905,
    "sample_weight": "equal total weight per scientist run",
    "target": "immediate archived configuration official accuracy",
    "uncertainty": "tree disagreement is descriptive, not a calibrated interval",
    "training_target_fidelity": "provisional recorder labels; full recipe-to-artifact adjudication pending",
}


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def public_payload(row):
    """Only the explicit prospective payload and timestamp-checked history enter X.

    Labels, eligibility decisions, scientist identity, and source metadata remain
    outside the feature boundary. Callers must build/audit the input itself; this
    function cannot detect a score paraphrased inside an otherwise allowed plan.
    """
    if not isinstance(row.get("model_input"), dict):
        raise TypeError("row requires a separate model_input dictionary")
    payload = copy.deepcopy(row["model_input"])
    forbidden = {"label", "labels", "official_metric", "y", "audit", "result", "conclusion"}
    if forbidden & set(payload):
        raise ValueError("Outcome/audit field at public input boundary")
    plan = payload.get("plan", {})
    if not isinstance(plan, dict) or set(plan) - {"problem", "hypothesis", "setup", "evaluation"}:
        raise ValueError("plan must contain only the first four prospective sections")
    if "prior_observations" in row:
        payload["prior_observations"] = copy.deepcopy(row["prior_observations"])
    return payload


def feature_document(payload):
    """Remove provenance identifiers while retaining numeric scientific settings."""
    ignored_keys = {
        "at",
        "timestamp",
        "first_submitted_at",
        "source_file",
        "source_path",
        "trace_path",
        "trace_sha256",
        "source_revision",
        "cell_id",
        "example_id",
        "scientist_model",
        "sha256",
        "evidence",
        "cutoff",
        "result_line",
        "call_line",
        "tool_use_id",
        "last_recorded_content_sha256",
        "provenance",
    }

    def clean(value, path=()):
        if isinstance(value, dict):
            scientific = (
                bool(path)
                and path[0] == "plan"
                or any(part in {"card", "evaluation", "measurements"} for part in path)
            )
            return {
                k: clean(v, (*path, k))
                for k, v in value.items()
                if scientific or k not in ignored_keys
            }
        if isinstance(value, list):
            return [clean(v, (*path, str(i))) for i, v in enumerate(value)]
        if isinstance(value, str):
            value = re.sub(r"(?<![\w])(?:aime-)?r0-\d+\b", "scientist_run", value)
            return re.sub(r"(?i)(?<![A-Za-z0-9])exp[-_ ]?0*(\d+)(?!\d)", r"node_\1", value)
        return value

    return json.dumps(clean(payload), ensure_ascii=False, sort_keys=True)


def benchmark_of(payload):
    task = payload.get("task")
    name = task.get("benchmark") if isinstance(task, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError("Public task must specify its benchmark")
    return name


def feature_contract_hash():
    return digest(
        {
            function.__name__: inspect.getsource(function)
            for function in (public_payload, feature_document, benchmark_of)
        }
    )


def _accuracy(row):
    label = row.get("label") or {}
    value = label.get("accuracy")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Official accuracy must be numeric or missing")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Official accuracy must be finite and in [0,1]")
    return float(value)


class RecipeWorldModel:
    """Fixed per-task outcome models with an inspectable training provenance."""

    def __init__(self):
        self.models = {}
        self.training_manifest = None

    def fit(self, rows, *, train_cell_ids, forbidden_cell_ids=()):
        train_cells, forbidden = set(train_cell_ids), set(forbidden_cell_ids)
        if not train_cells or train_cells & forbidden:
            raise ValueError("Training cells must be nonempty and disjoint from held-out cells")
        ids = [r["example_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate example IDs")
        unknown = train_cells - {r["cell_id"] for r in rows}
        if unknown:
            raise ValueError("Unknown training cells")
        selected = [
            r
            for r in rows
            if r["cell_id"] in train_cells
            and r.get("audit", {}).get("eligible") is True
            and _accuracy(r) is not None
        ]
        if not selected:
            raise ValueError("No eligible labeled training examples")
        grouped = {}
        for row in selected:
            payload = public_payload(row)
            grouped.setdefault(benchmark_of(payload), []).append((row, payload))
        self.models = {}
        provenance = []
        for benchmark, items in sorted(grouped.items()):
            items.sort(key=lambda item: item[0]["example_id"])
            docs = [feature_document(payload) for _, payload in items]
            vectorizer = TfidfVectorizer(
                ngram_range=tuple(MODEL_SPEC["ngram_range"]),
                max_features=MODEL_SPEC["max_features"],
                sublinear_tf=True,
                token_pattern=r"(?u)\b[\w][\w.\-]*\b",
            )
            features = vectorizer.fit_transform(docs)
            counts = Counter(row["cell_id"] for row, _ in items)
            weights = np.array([1 / counts[row["cell_id"]] for row, _ in items])
            targets = np.array([_accuracy(row) for row, _ in items])
            regressor = ExtraTreesRegressor(
                n_estimators=MODEL_SPEC["n_estimators"],
                min_samples_leaf=MODEL_SPEC["min_samples_leaf"],
                max_features=MODEL_SPEC["max_features_tree"],
                random_state=MODEL_SPEC["random_state"],
                n_jobs=1,
            )
            regressor.fit(features, targets, sample_weight=weights)
            examples = []
            for row, payload in items:
                examples.append(
                    {
                        "example_id": row["example_id"],
                        "cell_id": row["cell_id"],
                        "benchmark": benchmark,
                        "payload_sha256": digest(payload),
                        "accuracy": _accuracy(row),
                    }
                )
            self.models[benchmark] = {
                "vectorizer": vectorizer,
                "regressor": regressor,
                "training_features": features,
                "training_examples": examples,
            }
            provenance.extend(examples)
        self.training_manifest = {
            "model_spec": copy.deepcopy(MODEL_SPEC),
            "feature_contract_sha256": feature_contract_hash(),
            "declared_train_cell_ids": sorted(train_cells),
            "fitted_train_cell_ids": sorted({r["cell_id"] for r in selected}),
            "forbidden_cell_ids": sorted(forbidden),
            "training_examples": provenance,
            "training_data_sha256": digest(provenance),
            "fitted_benchmarks": sorted(self.models),
            "fit_uses_test_features_or_labels": False,
        }
        return self

    def predict(self, payload, *, neighbors=3):
        benchmark = benchmark_of(payload)
        if benchmark not in self.models:
            raise ValueError("No trained WM for this benchmark; cross-domain fallback is disabled")
        if (
            isinstance(neighbors, bool)
            or not isinstance(neighbors, int)
            or not 0 <= neighbors <= 10
        ):
            raise ValueError("neighbors must be an integer from 0 to 10")
        model = self.models[benchmark]
        features = model["vectorizer"].transform([feature_document(payload)])
        predictions = np.array(
            [tree.predict(features)[0] for tree in model["regressor"].estimators_]
        )
        similarities = (model["training_features"] @ features.T).toarray().ravel()
        order = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))[:neighbors]
        return {
            "predicted_official_accuracy": float(np.clip(predictions.mean(), 0, 1)),
            "tree_disagreement_std": float(predictions.std()),
            "benchmark": benchmark,
            "known_feature_count": int(features.nnz),
            "nearest_training_examples": [
                {**model["training_examples"][i], "text_similarity": float(similarities[i])}
                for i in order
            ],
            "warnings": [
                "This predicts a noisy archived outcome, not a guaranteed future result.",
                "Tree disagreement is not a calibrated confidence interval.",
                "Neighbors are training observations, not independent causal evidence.",
                "Use alongside the same plans, code, and raw history available to the baseline.",
            ],
            "training_data_sha256": self.training_manifest["training_data_sha256"],
        }

    def save(self, path):
        """Create an immutable trusted model artifact; do not overwrite old fits."""
        path = Path(path)
        if self.training_manifest is None:
            raise ValueError("Cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            joblib.dump(self, handle)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path, *, expected_sha256):
        """Only load a trusted local artifact with the separately frozen digest."""
        raw = Path(path).read_bytes()
        raw_hash = hashlib.sha256(raw).hexdigest()
        if raw_hash != expected_sha256:
            raise ValueError("WM artifact hash mismatch")
        model = joblib.load(io.BytesIO(raw))
        if not isinstance(model, cls) or model.training_manifest is None:
            raise ValueError("Not a fitted RecipeWorldModel")
        if model.training_manifest.get("feature_contract_sha256") != feature_contract_hash():
            raise ValueError("WM feature contract differs from the fitted artifact")
        return model


def serve(config):
    """Wire a trusted model to fixed candidate inputs; labels are never accepted."""
    from tools.outcome_prediction.wm_evidence import EvidenceStore, serve_stdio

    expected_keys = {
        "model_path",
        "model_sha256",
        "candidate_payloads",
        "evidence_root",
        "files",
        "audit_log",
    }
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise ValueError("WM server config has missing or unexpected keys")
    model = RecipeWorldModel.load(config["model_path"], expected_sha256=config["model_sha256"])
    candidates = config["candidate_payloads"]
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Serving requires fixed candidate payloads")
    for payload in candidates.values():
        public_payload({"model_input": payload})
    with EvidenceStore(
        config["evidence_root"],
        config["files"],
        audit_log=config["audit_log"],
        candidates=tuple(candidates),
        predict=lambda candidate_id: model.predict(candidates[candidate_id]),
    ) as store:
        serve_stdio(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    serve(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
