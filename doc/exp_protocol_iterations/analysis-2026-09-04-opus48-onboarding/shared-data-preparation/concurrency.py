"""Actual pinned native defaults; no Task, model or dataset constructed."""
import hashlib
import json
from pathlib import Path
import sys

import anyio
import inspect_ai._eval.task.run as run_module
from inspect_ai.log import EvalConfig
from inspect_ai.model import GenerateConfig

async def check():
    gate = run_module.create_sample_semaphore(EvalConfig(), GenerateConfig(max_connections=1))
    assert gate.value == 1
    record = {"image_label": sys.argv[1], "explicit_max_samples": None,
              "max_connections": 1, "effective_sample_concurrency": gate.value,
              "native_run_source_sha256": hashlib.sha256(Path(run_module.__file__).read_bytes()).hexdigest(),
              "model_called": False, "dataset_loaded": False}
    with (Path("/home/ben") / (sys.argv[1] + "-concurrency.json")).open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))

anyio.run(check)
