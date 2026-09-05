"""Freeze decode evidence without choosing settings, loading weights or evaluating.

File intent, API request, native defaults and a caller-observed native request
object are separate layers. None proves what an engine actually executed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from . import schema

FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "max_new_tokens",
        "min_tokens",
        "seed",
        "n",
        "best_of",
        "stop",
        "stop_token_ids",
        "ignore_eos",
        "eos_token_id",
        "do_sample",
    }
)


class DecodeEvidenceError(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def bad(value):
        raise DecodeEvidenceError(f"non-finite JSON value: {value}")

    value = json.loads(data, parse_constant=bad)
    # Also rejects overflowing numeric literals such as 1e999.
    try:
        json.dumps(value, allow_nan=False)
    except ValueError as exc:
        raise DecodeEvidenceError("non-finite JSON value") from exc
    return value


def _mode(fields):
    temperature = fields.get("temperature")
    if type(temperature) not in (int, float) or not math.isfinite(temperature):
        return "unknown"
    if temperature == 0:
        return "greedy_requested"
    if temperature > 0:
        return "sampling_requested"
    return "unknown"


def _pointer(value, pointer):
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise DecodeEvidenceError("request pointer must be an RFC6901 JSON pointer")
    try:
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            value = value[int(token)] if isinstance(value, list) else value[token]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DecodeEvidenceError("request pointer does not resolve") from exc
    return value


def _native_defaults(model_config, selected_path):
    import vllm
    from vllm.config import ModelConfig

    if vllm.__version__ != "0.11.0" or type(model_config) is not ModelConfig:
        raise DecodeEvidenceError("requires native vLLM0.11.0 ModelConfig")
    method = model_config.get_diff_sampling_param
    source = Path(inspect.getsourcefile(ModelConfig.get_diff_sampling_param)).resolve()
    expected = Path(vllm.__file__).resolve().parent / "config/model.py"
    if (
        getattr(method, "__func__", None) is not ModelConfig.get_diff_sampling_param
        or source != expected
    ):
        raise DecodeEvidenceError("native ModelConfig resolver was replaced")
    model_path = Path(model_config.hf_config_path or model_config.model).resolve()
    if (
        selected_path.name != "generation_config.json"
        or model_path != selected_path.parent
        or model_config.generation_config != "auto"
    ):
        raise DecodeEvidenceError(
            "native defaults must read this selected model directory in auto mode"
        )
    fields = method()
    if not isinstance(fields, dict):
        raise DecodeEvidenceError("native resolver returned unsupported evidence")
    return {
        "status": "observed_native_method_return",
        "vllm": vllm.__version__,
        "source_sha256": schema.sha256_file(source),
        "fields": fields,
        "mode_from_returned_fields": _mode(fields),
        "scope": "model defaults plus native overrides; not a resolved API request or engine execution",
    }


def _native_request(params):
    import vllm
    from vllm import SamplingParams

    if vllm.__version__ != "0.11.0" or type(params) is not SamplingParams:
        raise DecodeEvidenceError("requires native vLLM0.11.0 SamplingParams")
    from .sampling import _plain

    fields = {name: _plain(getattr(params, name)) for name in params.__struct_fields__}
    return {
        "status": "observed_caller_supplied_native_object",
        "vllm": vllm.__version__,
        "source_sha256": schema.sha256_file(Path(inspect.getsourcefile(SamplingParams))),
        "fields": fields,
        "mode_from_object": _mode(fields),
        "connection_to_captured_request": "not_independently_verified",
        "engine_execution": "unknown",
    }


def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def freeze_decode_evidence(
    selected_json,
    output_dir,
    *,
    intent=None,
    request_path=None,
    request_pointer="",
    native_model_config=None,
    native_request_params=None,
):
    """Freeze exact selected JSON and an actual request's supported fields on CPU.

    request_path is an existing JSON record; request_pointer locates the request
    object in it. Prompt text/headers are not copied. Native objects are optional:
    missing layers remain unknown. This never constructs native model objects.
    """
    if intent not in (None, "greedy", "sampling", "unspecified"):
        raise DecodeEvidenceError("intent must be greedy, sampling, unspecified or None")
    selected_path = Path(selected_json).resolve()
    selected_bytes = selected_path.read_bytes()
    selected = _json(selected_bytes)
    if not isinstance(selected, dict):
        raise DecodeEvidenceError("selected generation JSON must be an object")
    request = {"status": "unknown"}
    request_bytes = None
    if request_path is not None:
        request_file = Path(request_path).resolve()
        request_bytes = request_file.read_bytes()
        actual = _pointer(_json(request_bytes), request_pointer)
        if not isinstance(actual, dict):
            raise DecodeEvidenceError("captured request must be an object")
        extra = actual.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise DecodeEvidenceError("request extra_body must be an object")
        request = {
            "status": "observed_file_fields",
            "path": str(request_file),
            "sha256": _sha(request_bytes),
            "json_pointer": request_pointer,
            "model": actual.get("model"),
            "fields": {k: v for k, v in actual.items() if k in FIELDS},
            "extra_body_fields": {k: v for k, v in extra.items() if k in FIELDS},
            "engine_execution": "unknown",
        }
    elif request_pointer:
        raise DecodeEvidenceError("request pointer needs a request file")
    report = {
        "schema_version": "awm-decode-evidence-v1",
        "created_at": schema.now(),
        "intent": intent,
        "selected": {
            "path": str(selected_path),
            "sha256": _sha(selected_bytes),
            "fields": {k: v for k, v in selected.items() if k in FIELDS},
            "mode_from_explicit_temperature": _mode(selected),
            "do_sample_is_vllm_mode_evidence": False,
        },
        "request": request,
        "native_defaults": {"status": "unknown"}
        if native_model_config is None
        else _native_defaults(native_model_config, selected_path),
        "native_request": {"status": "unknown"}
        if native_request_params is None
        else _native_request(native_request_params),
        "effective_engine_decode": "unknown",
        "model_loading": "not_performed",
        "scientific_validation": "not_performed",
    }
    if selected_path.read_bytes() != selected_bytes:
        raise DecodeEvidenceError("selected JSON changed during observation")
    if request_bytes is not None and request_file.read_bytes() != request_bytes:
        raise DecodeEvidenceError("request source changed during observation")
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    destination = Path(output_dir)
    destination.mkdir(parents=True)  # Existing evidence is never overwritten.
    _write(destination / "selected-generation-config.json", selected_bytes)
    _write(destination / "decode-evidence.json", encoded)
    return report


def verify_decode_evidence(directory):
    """Check retained selected bytes and source identity, not engine/model behavior."""
    directory = Path(directory)
    report = _json((directory / "decode-evidence.json").read_bytes())
    if report.get("schema_version") != "awm-decode-evidence-v1":
        raise DecodeEvidenceError("unsupported decode evidence schema")
    saved = (directory / "selected-generation-config.json").read_bytes()
    if _sha(saved) != report["selected"]["sha256"]:
        raise DecodeEvidenceError("retained selected bytes changed")
    if Path(report["selected"]["path"]).read_bytes() != saved:
        raise DecodeEvidenceError("selected serving JSON changed since observation")
    request = report["request"]
    if (
        request["status"] != "unknown"
        and schema.sha256_file(Path(request["path"])) != request["sha256"]
    ):
        raise DecodeEvidenceError("request source changed since observation")
    return {"status": "unchanged_evidence", "effective_engine_decode": "unknown"}
