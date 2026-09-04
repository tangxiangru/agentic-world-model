"""Opt-in generation-config save transactions; never a parent-file preflight.

This is a compatibility adapter, not a security boundary or a checkpoint validator.
Only the native save call observed inside ``saving`` is covered. Importing this
module does not import torch/Transformers or inspect an evaluator's model files.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from uuid import uuid4

TRANSFORMERS_VERSION = "4.57.3"
SOURCE_HASHES = {
    "generation/configuration_utils.py": "d83f2281f939402be1633a29f3c760e29f5d2f284258d8ed99693b873744074b",
    "configuration_utils.py": "08a3889c1b8a73d340a37c93f72346ee0e385e339b7a72a03b5c026d4943baf7",
    "modeling_utils.py": "bf1c6b2a43cf7c36fb79f37c981424dd6ae78eb863fcaa5d2a37e76c9828611d",
    "trainer.py": "c3eac8fa28bda6330fd4dee131e7fe7d9bdc0fa35d9de944fce52cb97b379bab",
}
INACTIVE_SAMPLING_DEFAULTS = {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "min_p": None,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
}
_MISSING = object()
_ACTIVE_MODELS: set[int] = set()
_ACTIVE_OUTPUTS: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


class SaveContractError(RuntimeError):
    """The requested save was not certified by this narrow adapter."""


class UnsupportedSavePath(SaveContractError):
    """An untested library, model, or distributed/custom writer was requested."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _config_hash(config, generation) -> str:
    return _sha(
        _json_bytes(
            {
                "model_config": config.to_dict(),
                "generation_config": generation.to_dict() if generation else None,
            }
        )
    )


def validate_runtime() -> dict:
    """Verify installed source, version, and native API before using the adapter."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedSavePath("Transformers 4.57.3 from the pinned image is required") from exc
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise UnsupportedSavePath(f"unsupported Transformers version: {transformers.__version__}")
    root = Path(transformers.__file__).parent
    for relative, expected in SOURCE_HASHES.items():
        if _sha((root / relative).read_bytes()) != expected:
            raise UnsupportedSavePath(f"pinned Transformers source mismatch: {relative}")
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

    methods = (
        (GenerationConfig.validate, "generation/configuration_utils.py", "strict"),
        (GenerationConfig.save_pretrained, "generation/configuration_utils.py", "save_directory"),
        (PreTrainedModel.save_pretrained, "modeling_utils.py", "save_directory"),
        (PretrainedConfig._get_non_default_generation_parameters, "configuration_utils.py", "self"),
    )
    for method, relative, parameter in methods:
        if (
            Path(inspect.getsourcefile(method) or "").resolve() != (root / relative).resolve()
            or parameter not in inspect.signature(method).parameters
        ):
            raise UnsupportedSavePath("native Transformers API was replaced or is incompatible")
    return {"transformers": TRANSFORMERS_VERSION, "source_sha256": dict(SOURCE_HASHES)}


def _require_model(model, *, _instrumented_save=None) -> dict:
    identity = validate_runtime()
    from torch import distributed
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

    if (
        not isinstance(model, PreTrainedModel)
        or not type(model).__module__.startswith("transformers.models.")
        or getattr(model.save_pretrained, "__func__", None)
        is not (_instrumented_save or PreTrainedModel.save_pretrained)
    ):
        raise UnsupportedSavePath("requires an unwrapped native Transformers model/save_pretrained")
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        placement = str(device_map.get("", ""))
        single_device = set(device_map) == {""} and (
            placement == "cpu"
            or placement == "cuda"
            or placement.startswith("cuda:")
            or placement.isdecimal()
        )
        if not single_device:
            raise UnsupportedSavePath("sharded/offloaded device maps are unsupported")
    if (
        getattr(model, "_hf_peft_config_loaded", False)
        or hasattr(model, "peft_config")
        or getattr(model, "hf_quantizer", None) is not None
        or getattr(model, "_auto_class", None) is not None
        or getattr(model, "_tp_size", None) not in (None, 0, 1)
    ):
        raise UnsupportedSavePath("PEFT, quantized/offloaded, custom, and TP saves are unsupported")
    try:
        multiple_processes = int(os.environ.get("WORLD_SIZE", "1")) != 1
    except ValueError as exc:
        raise UnsupportedSavePath("invalid WORLD_SIZE") from exc
    if multiple_processes or (distributed.is_initialized() and distributed.get_world_size() != 1):
        raise UnsupportedSavePath("only single-process saves are supported")
    config = model.config
    if (
        not isinstance(config, PretrainedConfig)
        or not type(config).__module__.startswith("transformers.")
        or getattr(config._get_non_default_generation_parameters, "__func__", None)
        is not PretrainedConfig._get_non_default_generation_parameters
    ):
        raise UnsupportedSavePath("custom model configuration/migration is unsupported")
    if (
        model.can_generate()
        and type(getattr(model, "generation_config", None)) is not GenerationConfig
    ):
        raise UnsupportedSavePath("generation-capable model needs a native GenerationConfig")
    return identity


def _project(model, report=None) -> tuple[object, object, dict]:
    """Pure configuration-only projection; caller establishes native API identity."""
    original_generation = getattr(model, "generation_config", None)
    config = copy.deepcopy(model.config)
    generation = copy.deepcopy(original_generation)
    report = {} if report is None else report
    report.update(
        {
            "input_hash": _config_hash(model.config, original_generation),
            "migration": {},
            "normalization": {},
            "status": "not_applicable",
        }
    )
    if model.can_generate():
        # The supported native path is not PEFT: it always performs this migration.
        migration = config._get_non_default_generation_parameters()
        for key, value in migration.items():
            setattr(generation, key, value)
            setattr(config, key, None)
        report["migration"] = copy.deepcopy(migration)
        report["effective_hash"] = _config_hash(config, generation)
        report["serializer_hash"] = report["effective_hash"]
        report["status"] = "invalid"
        try:
            generation.validate(strict=True)
        except ValueError:
            if generation.do_sample is not False:
                raise
            for key, neutral in INACTIVE_SAMPLING_DEFAULTS.items():
                value = getattr(generation, key)
                if value is not None and value != neutral:
                    report["normalization"][key] = {"before": value, "after": neutral}
                    setattr(generation, key, neutral)
            report["serializer_hash"] = _config_hash(config, generation)
            # Non-sampling errors are not swallowed; no exception-text heuristics.
            generation.validate(strict=True)
        report["status"] = "normalizable" if report["normalization"] else "valid"
    else:
        report["effective_hash"] = _config_hash(config, generation)
    report["serializer_hash"] = _config_hash(config, generation)
    return config, generation, report


@contextmanager
def _exclusive(model, output: Path | None = None):
    key = str(output) if output is not None else None
    with _ACTIVE_LOCK:
        if id(model) in _ACTIVE_MODELS or (key is not None and key in _ACTIVE_OUTPUTS):
            raise SaveContractError("reentrant/concurrent save or check is unsupported")
        _ACTIVE_MODELS.add(id(model))
        if key is not None:
            _ACTIVE_OUTPUTS.add(key)
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_MODELS.remove(id(model))
            if key is not None:
                _ACTIVE_OUTPUTS.remove(key)


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Atomic individual file replacement, NOT an atomic model/checkpoint save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _selected_json(data: bytes | None) -> bytes | None:
    if data is None:
        return None
    if not isinstance(data, bytes):
        raise TypeError("selected_serving_json must be frozen bytes, not a path or mutable object")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate serving JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite serving JSON value: {value}")

    parsed = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(parsed, dict):
        raise TypeError("selected serving JSON must contain an object")
    # parse_constant catches NaN/Infinity literals, not a numeric overflow (1e999).
    _json_bytes(parsed)
    return data


class GenerationSaveContract:
    """Native save protection with inspectable and durable per-invocation evidence.

    ``audit_dir`` defaults to ``OUTPUT_PARENT/.exp-protocol-save-events``. Audit
    failures fail a successful save; they never replace an existing writer error.
    ``records`` retains the failure evidence even if its disk write fails.
    """

    def __init__(self, *, policy="inactive_sampling_v1", audit_dir=None):
        if policy != "inactive_sampling_v1":
            raise ValueError(f"unsupported save policy: {policy}")
        self.policy = policy
        self.audit_dir = Path(audit_dir).resolve() if audit_dir is not None else None
        self.records: list[dict] = []

    def check_before_compute(self, model) -> dict:
        """Check after in-code repairs; no model forwards, weight writes, or parent reads."""
        with _exclusive(model):
            identity = _require_model(model)
            _, _, report = _project(model)
            return {
                **report,
                "policy": self.policy,
                "library": identity,
                "coverage": "configuration_only_at_check_time",
            }

    def _audit(self, event, path, original_error=None):
        try:
            _atomic_bytes(path, _json_bytes(event))
        except BaseException as exc:
            event["audit_error"] = f"{type(exc).__name__}: {exc}"
            if original_error is None:
                raise
            if hasattr(original_error, "add_note"):
                original_error.add_note(f"save audit also failed: {exc}")

    @contextmanager
    def saving(self, model, output_dir, *, selected_serving_json=None):
        """Observe exactly one native save to this output; restore objects on all exits.

        The instance method is temporarily instrumented (never the global class).
        A scope with no observed call, a swallowed writer failure, or a different
        output cannot certify old files. Uncooperative direct/class-method calls
        remain bypasses, not protection. Exclusive model use is required.
        """
        output = Path(output_dir).resolve()
        with _exclusive(model, output):
            event = {
                "schema_version": 1,
                "id": uuid4().hex,
                "policy": self.policy,
                "output": str(output),
                "outcome": "started",
                "observed_native_calls": 0,
                "coverage": "native_model_save_only_not_whole_checkpoint",
            }
            self.records.append(event)
            audit_path = (self.audit_dir or output.parent / ".exp-protocol-save-events") / (
                event["id"] + ".json"
            )
            original_instance_save = model.__dict__.get("save_pretrained", _MISSING)
            instrumented = False
            try:
                event["library"] = _require_model(model)
                selected = _selected_json(selected_serving_json)
                if selected is not None and not model.can_generate():
                    raise UnsupportedSavePath(
                        "serving generation JSON needs a generation-capable model"
                    )
                event["selected_serving_hash"] = _sha(selected) if selected is not None else None
                if output.is_file():
                    raise SaveContractError("save output is a file, not a directory")
                # Early failure before yielding; reproject again at the actual invocation.
                event["precheck"] = {}
                _project(model, event["precheck"])
                self._audit(event, audit_path)
                native_save = model.save_pretrained

                def checked_save(instance, *args, **kwargs):
                    event["native_save_completed"] = False
                    event["observed_native_calls"] += 1
                    if event["observed_native_calls"] != 1:
                        raise SaveContractError(
                            "exactly one native save is allowed per transaction"
                        )
                    _require_model(instance, _instrumented_save=checked_save)
                    bound = inspect.signature(native_save).bind(*args, **kwargs)
                    bound.apply_defaults()
                    if Path(bound.arguments["save_directory"]).resolve() != output:
                        raise SaveContractError(
                            "native save output differs from transaction output"
                        )
                    extra = bound.arguments.get("kwargs", {})
                    if (
                        not bound.arguments["is_main_process"]
                        or bound.arguments["push_to_hub"]
                        or ("save_config" in extra and not extra["save_config"])
                    ):
                        raise UnsupportedSavePath(
                            "non-writer ranks, config suppression and Hub writes unsupported"
                        )
                    if selected is not None and not instance.can_generate():
                        raise UnsupportedSavePath("generation capability changed during save scope")
                    old_config = instance.config
                    old_generation = instance.__dict__.get("generation_config", _MISSING)
                    event["projection"] = {}
                    config, generation, _ = _project(instance, event["projection"])
                    try:
                        instance.config = config
                        if old_generation is not _MISSING:
                            instance.generation_config = generation
                        result = native_save(*args, **kwargs)
                    finally:
                        instance.config = old_config
                        if old_generation is not _MISSING:
                            instance.generation_config = old_generation
                        else:
                            instance.__dict__.pop("generation_config", None)
                    filename = (
                        "generation_config.json" if instance.can_generate() else "config.json"
                    )
                    serialized = (output / filename).read_bytes()
                    event["serialized_file_hash"] = _sha(serialized)
                    event["verified_filename"] = filename
                    if selected is not None:
                        _atomic_bytes(output / "generation_config.json", selected)
                        actual = (output / "generation_config.json").read_bytes()
                        if _sha(actual) != event["selected_serving_hash"]:
                            raise SaveContractError("selected serving-file hash mismatch")
                        event["serving_file_hash"] = _sha(actual)
                    event["native_save_completed"] = True
                    return result

                model.save_pretrained = MethodType(checked_save, model)
                instrumented = True
                yield event
                if not event.get("native_save_completed") or event["observed_native_calls"] != 1:
                    raise SaveContractError(
                        "no successfully observed native save; output is unverified"
                    )
                expected_hash = event.get("serving_file_hash", event["serialized_file_hash"])
                if _sha((output / event["verified_filename"]).read_bytes()) != expected_hash:
                    raise SaveContractError(
                        "saved configuration changed before transaction completion"
                    )
                event["outcome"] = "saved"
                self._audit(event, audit_path)
            except BaseException as exc:
                event["outcome"] = "failed"
                event["error"] = f"{type(exc).__name__}: {exc}"
                self._audit(event, audit_path, original_error=exc)
                raise
            finally:
                if instrumented:
                    if original_instance_save is _MISSING:
                        model.__dict__.pop("save_pretrained", None)
                    else:
                        model.save_pretrained = original_instance_save
