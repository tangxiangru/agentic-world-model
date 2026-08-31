#!/usr/bin/env python3
"""Create deterministic artifacts for the ExperimentBundle smoke fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path


root = Path(os.environ["AWM_EXPERIMENT_DIR"])
checkpoint = root / "artifacts" / "candidate-checkpoint"
checkpoint.mkdir(parents=True, exist_ok=True)
(checkpoint / "model.json").write_text('{"fixture": "local-smoke"}\n')
measurements = root / "measurements"
measurements.mkdir(parents=True, exist_ok=True)
(measurements / "diagnostic-v1.json").write_text(
    json.dumps({"metric": "accuracy", "value": 1.0, "parent_value": 0.25, "n": 4}) + "\n"
)
