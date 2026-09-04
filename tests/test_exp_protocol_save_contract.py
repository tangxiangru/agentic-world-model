"""CPU-only transaction tests plus explicitly dependency-gated native save tests.

The small stand-ins exercise error plumbing, not Transformers compatibility.
Tests named ``native`` use the actual pinned library and never train/forward.
"""

import copy
import json
import threading
from pathlib import Path

import pytest

from awm.exp_protocol import save_contract as sc


class Config:
    def __init__(self, **values):
        self.__dict__.update(values)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def _get_non_default_generation_parameters(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in sc.INACTIVE_SAMPLING_DEFAULTS
            and v is not None
            and v != sc.INACTIVE_SAMPLING_DEFAULTS[k]
        }


class Generation(Config):
    def __init__(self, **values):
        super().__init__(
            **{
                **sc.INACTIVE_SAMPLING_DEFAULTS,
                "do_sample": False,
                "eos_token_id": [1, 106],
                "max_new_tokens": None,
                **values,
            }
        )

    def validate(self, strict=False):
        assert strict
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("unrepairable length")
        if self.do_sample is False:
            for key, value in sc.INACTIVE_SAMPLING_DEFAULTS.items():
                if getattr(self, key) is not None and getattr(self, key) != value:
                    raise ValueError("stand-in greedy validation failure")


class Model:
    def __init__(self, **generation):
        self.config = Config(nested={"untouched": [1, 2]})
        self.generation_config = Generation(**generation)

    def can_generate(self):
        return self.generation_config is not None

    def save_pretrained(self, save_directory, is_main_process=True, push_to_hub=False, **kwargs):
        if self.can_generate():
            self.generation_config.validate(strict=True)
        if kwargs.get("fault"):
            # Mutate only copies to verify alias/object restoration even after errors.
            self.config.nested["untouched"].append(3)
            raise kwargs["fault"]
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.config.to_dict()))
        if self.can_generate():
            (path / "generation_config.json").write_text(
                json.dumps(self.generation_config.to_dict())
            )
        return "native result"


@pytest.fixture
def standin(monkeypatch):
    monkeypatch.setattr(sc, "_require_model", lambda model, **kwargs: {"test_double": True})
    return Model


def test_import_and_unused_contract_are_inert(tmp_path):
    # Both historical pure-evaluator cases may hold serializer-invalid parent JSON.
    parent = tmp_path / "generation_config.json"
    raw = b'{"do_sample":false,"temperature":0.0,"top_k":0}'
    parent.write_bytes(raw)
    saves = sc.GenerationSaveContract()
    assert saves.records == []
    assert parent.read_bytes() == raw


@pytest.mark.parametrize(
    "values",
    [{}, {"do_sample": True, "top_k": 64, "top_p": 0.95}, {"temperature": None, "top_k": None}],
)
def test_valid_and_already_repaired_configs_are_unchanged(standin, tmp_path, values):
    model = standin(**values)
    config, generation = model.config, model.generation_config
    before = generation.to_dict()
    saves = sc.GenerationSaveContract()
    assert saves.check_before_compute(model)["normalization"] == {}
    with saves.saving(model, tmp_path / "out") as event:
        assert model.save_pretrained(tmp_path / "out") == "native result"
    assert event["outcome"] == "saved"
    assert model.config is config and model.generation_config is generation
    assert generation.to_dict() == before
    assert "save_pretrained" not in model.__dict__


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0),
        ("top_k", 0),
        ("top_p", 0.8),
        ("typical_p", 0.8),
        ("min_p", 0.1),
        ("epsilon_cutoff", 0.01),
        ("eta_cutoff", 0.01),
    ],
)
def test_each_inactive_sampling_field(standin, tmp_path, field, value):
    model = standin(**{field: value})
    saves = sc.GenerationSaveContract()
    report = saves.check_before_compute(model)
    assert report["status"] == "normalizable"
    assert report["normalization"] == {
        field: {"before": value, "after": sc.INACTIVE_SAMPLING_DEFAULTS[field]}
    }
    assert getattr(model.generation_config, field) == value
    with saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out")
    result = json.loads((tmp_path / "out/generation_config.json").read_bytes())
    assert result[field] == sc.INACTIVE_SAMPLING_DEFAULTS[field]
    assert result["do_sample"] is False and result["eos_token_id"] == [1, 106]


def test_effective_migration_and_mutation_since_early_check(standin, tmp_path):
    model = standin()
    saves = sc.GenerationSaveContract()
    assert saves.check_before_compute(model)["status"] == "valid"
    model.config.temperature = 0.0
    original = model.config
    with saves.saving(model, tmp_path / "out") as event:
        model.generation_config.top_k = 0
        model.save_pretrained(tmp_path / "out")
    assert event["projection"]["migration"] == {"temperature": 0.0}
    assert set(event["projection"]["normalization"]) == {"temperature", "top_k"}
    assert model.config is original and model.config.temperature == 0.0


def test_hard_validation_error_precedes_writer(standin, tmp_path):
    model = standin(temperature=0, max_new_tokens=0)
    saves = sc.GenerationSaveContract()
    with pytest.raises(ValueError, match="unrepairable"), saves.saving(model, tmp_path / "out"):
        pytest.fail("invalid projection yielded to writer")
    assert not (tmp_path / "out").exists()
    assert saves.records[-1]["outcome"] == "failed"
    assert saves.records[-1]["precheck"]["status"] == "invalid"
    assert saves.records[-1]["precheck"]["input_hash"]


@pytest.mark.parametrize("failure", [OSError("disk"), KeyboardInterrupt("interrupted")])
def test_writer_error_restores_exact_objects_and_nested_values(standin, tmp_path, failure):
    model = standin(temperature=0)
    originals = (model.config, model.generation_config)
    before = [c.to_dict() for c in originals]
    saves = sc.GenerationSaveContract()
    with pytest.raises(type(failure)) as caught, saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out", fault=failure)
    assert caught.value is failure
    assert model.config is originals[0] and model.generation_config is originals[1]
    assert [c.to_dict() for c in originals] == before
    assert "save_pretrained" not in model.__dict__
    assert saves.records[-1]["outcome"] == "failed"


def test_selected_exact_bytes_and_distinct_serializer_identity(standin, tmp_path):
    selected = b'{ "do_sample": false, "temperature": 0.0, "eos_token_id": [1,106] }\n'
    model = standin(temperature=0)
    saves = sc.GenerationSaveContract()
    with saves.saving(model, tmp_path / "out", selected_serving_json=selected) as event:
        model.save_pretrained(tmp_path / "out")
    assert (tmp_path / "out/generation_config.json").read_bytes() == selected
    assert event["serving_file_hash"] == sc._sha(selected)
    assert event["serialized_file_hash"] != event["serving_file_hash"]
    audit = json.loads(next((tmp_path / ".exp-protocol-save-events").glob("*.json")).read_text())
    assert audit == event


@pytest.mark.parametrize(
    "selected",
    ["{}", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":{"nested":[-1e999]}}'],
)
def test_selected_json_rejects_mutable_ambiguous_or_nonfinite_input(selected):
    with pytest.raises((TypeError, ValueError)):
        sc._selected_json(selected)


def test_serving_write_failure_is_incomplete_but_objects_restored(standin, monkeypatch, tmp_path):
    model = standin(temperature=0)
    generation = model.generation_config
    real_atomic = sc._atomic_bytes

    def broken(path, data):
        if path.name == "generation_config.json":
            raise OSError("serving write failed")
        real_atomic(path, data)

    monkeypatch.setattr(sc, "_atomic_bytes", broken)
    saves = sc.GenerationSaveContract()
    with (
        pytest.raises(OSError, match="serving write"),
        saves.saving(model, tmp_path / "out", selected_serving_json=b"{}"),
    ):
        model.save_pretrained(tmp_path / "out")
    assert model.generation_config is generation
    assert saves.records[-1]["outcome"] == "failed"


def test_failed_audit_does_not_mask_original_exception(standin, monkeypatch, tmp_path):
    model = standin()
    saves = sc.GenerationSaveContract()
    original = OSError("original writer error")
    with (
        pytest.raises(OSError, match="original writer error") as caught,
        saves.saving(model, tmp_path / "out"),
    ):
        monkeypatch.setattr(sc, "_atomic_bytes", lambda *a: (_ for _ in ()).throw(OSError("audit")))
        model.save_pretrained(tmp_path / "out", fault=original)
    assert caught.value is original
    assert "audit_error" in saves.records[-1]


def test_no_native_call_never_certifies_stale_files(standin, tmp_path):
    model = standin()
    model.save_pretrained(tmp_path / "out")
    with (
        pytest.raises(sc.SaveContractError, match="no successfully observed"),
        sc.GenerationSaveContract().saving(model, tmp_path / "out"),
    ):
        pass


def test_swallowed_writer_failure_and_config_rewrite_are_not_success(standin, tmp_path):
    model = standin()
    with (
        pytest.raises(sc.SaveContractError, match="no successfully observed"),
        sc.GenerationSaveContract().saving(model, tmp_path / "out"),
    ):
        try:
            model.save_pretrained(tmp_path / "out", fault=OSError("failed"))
        except OSError:
            pass
    with (
        pytest.raises(sc.SaveContractError, match="changed"),
        sc.GenerationSaveContract().saving(model, tmp_path / "out"),
    ):
        model.save_pretrained(tmp_path / "out")
        (tmp_path / "out/generation_config.json").write_text("{}")


def test_reentry_and_concurrent_calls_across_contract_instances(standin, tmp_path):
    model = standin()
    errors = []

    def other_thread():
        try:
            sc.GenerationSaveContract().check_before_compute(model)
        except sc.SaveContractError as exc:
            errors.append(exc)

    with sc.GenerationSaveContract().saving(model, tmp_path / "out"):
        with pytest.raises(sc.SaveContractError, match="concurrent"):
            sc.GenerationSaveContract().check_before_compute(model)
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(errors) == 1
        model.save_pretrained(tmp_path / "out")


def test_non_generation_save_skips_generation_validation(standin, tmp_path):
    model = standin()
    model.generation_config = None
    saves = sc.GenerationSaveContract()
    assert saves.check_before_compute(model)["status"] == "not_applicable"
    with saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out")
    assert not (tmp_path / "out/generation_config.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"is_main_process": False}, {"push_to_hub": True}, {"save_config": False}, {"save_config": 0}],
)
def test_writer_suppression_and_network_paths_rejected(standin, tmp_path, kwargs):
    model = standin()
    with (
        pytest.raises(sc.UnsupportedSavePath),
        sc.GenerationSaveContract().saving(model, tmp_path / "out"),
    ):
        model.save_pretrained(tmp_path / "out", **kwargs)


@pytest.fixture
def native_model():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sc.validate_runtime()  # A present but different library is a failure, not a skip.
    from transformers import GPT2Config, GPT2LMHeadModel

    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=8))


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": 0.0},
        {"temperature": 0.0, "top_k": 0},
        {"temperature": 1.0, "top_p": None, "top_k": None},
        {},
        {"do_sample": True, "temperature": None},
        {"do_sample": True, "top_k": 64, "top_p": 0.95},
        {"epsilon_cutoff": 0.01, "eta_cutoff": 0.01, "min_p": 0.1, "typical_p": 0.8},
    ],
)
def test_native_historical_and_extra_field_fixtures(native_model, tmp_path, values):
    from transformers import GenerationConfig

    model = native_model
    for field, value in values.items():
        setattr(model.generation_config, field, value)
    before = model.generation_config.to_dict()
    config, generation = model.config, model.generation_config
    saves = sc.GenerationSaveContract()
    with saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out")
    GenerationConfig.from_pretrained(tmp_path / "out", local_files_only=True).validate(strict=True)
    assert model.config is config and model.generation_config is generation
    assert generation.to_dict() == before
    assert (tmp_path / "out/model.safetensors").is_file()


def test_native_migration_and_selected_serving(native_model, tmp_path):
    model = native_model
    model.config.temperature = 0.0
    selected = b'{"do_sample":false,"temperature":0.0,"top_k":0}\n'
    saves = sc.GenerationSaveContract()
    report = saves.check_before_compute(model)
    assert report["migration"] == {"temperature": 0.0}
    assert report["normalization"]["temperature"]["after"] == 1.0
    with saves.saving(model, tmp_path / "out", selected_serving_json=selected):
        model.save_pretrained(tmp_path / "out")
    assert (tmp_path / "out/generation_config.json").read_bytes() == selected
    assert model.config.temperature == 0.0


def test_native_hard_error_custom_save_and_source_mismatch(native_model, tmp_path, monkeypatch):
    model = native_model
    model.generation_config.max_new_tokens = 0
    with pytest.raises(ValueError):
        sc.GenerationSaveContract().check_before_compute(model)
    model.generation_config.max_new_tokens = None
    model.save_pretrained = lambda *args, **kwargs: None
    with pytest.raises(sc.UnsupportedSavePath):
        sc.GenerationSaveContract().check_before_compute(model)
    del model.save_pretrained
    monkeypatch.setitem(sc.SOURCE_HASHES, "trainer.py", "0" * 64)
    with pytest.raises(sc.UnsupportedSavePath, match="source mismatch"):
        sc.GenerationSaveContract().check_before_compute(model)


def test_native_non_generation_and_unsupported_flags(native_model, tmp_path):
    from transformers import GPT2Model

    model = GPT2Model(native_model.config)
    saves = sc.GenerationSaveContract()
    assert saves.check_before_compute(model)["status"] == "not_applicable"
    with saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out")
    assert not (tmp_path / "out/generation_config.json").exists()
    native_model._hf_peft_config_loaded = True
    with pytest.raises(sc.UnsupportedSavePath):
        saves.check_before_compute(native_model)


def test_native_contrastive_top_k_predicate_is_pinned_not_assumed(native_model, tmp_path):
    model = native_model
    model.generation_config.penalty_alpha = 0.6
    model.generation_config.top_k = 4
    # In 4.57.3, unlike older libraries, top_k has no penalty_alpha exemption.
    with pytest.raises(ValueError, match="top_k"):
        model.generation_config.validate(strict=True)
    model.generation_config.temperature = 0.0
    saves = sc.GenerationSaveContract()
    with saves.saving(model, tmp_path / "out") as event:
        model.save_pretrained(tmp_path / "out")
    assert set(event["projection"]["normalization"]) == {"temperature", "top_k"}
    assert model.generation_config.penalty_alpha == 0.6
    assert model.generation_config.top_k == 4
    saved = json.loads((tmp_path / "out/generation_config.json").read_bytes())
    assert saved["penalty_alpha"] == 0.6


def test_native_migration_can_make_sampling_fields_active(native_model, tmp_path):
    model = native_model
    model.config.do_sample = True
    model.generation_config.top_k = 4
    model.generation_config.top_p = 0.8
    saves = sc.GenerationSaveContract()
    report = saves.check_before_compute(model)
    assert report["migration"] == {"do_sample": True}
    assert report["normalization"] == {}
    with saves.saving(model, tmp_path / "out"):
        model.save_pretrained(tmp_path / "out")
    saved = json.loads((tmp_path / "out/generation_config.json").read_bytes())
    assert saved["top_k"] == 4 and saved["top_p"] == 0.8 and saved["do_sample"] is True


def test_native_writer_fault_restores_objects(native_model, tmp_path):
    model = native_model
    config, generation = model.config, model.generation_config
    generation.temperature = 0.0
    failure = OSError("native weight writer fault")

    def failed_weight_writer(*args, **kwargs):
        raise failure

    saves = sc.GenerationSaveContract()
    with pytest.raises(OSError) as caught, saves.saving(model, tmp_path / "out"):
        model.save_pretrained(
            tmp_path / "out", safe_serialization=False, save_function=failed_weight_writer
        )
    assert caught.value is failure
    assert model.config is config and model.generation_config is generation
    assert generation.temperature == 0.0 and "save_pretrained" not in model.__dict__
    assert saves.records[-1]["outcome"] == "failed"


def test_native_tiny_gemma3_save_without_forward_or_download(native_model, tmp_path):
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        GenerationConfig,
        SiglipVisionConfig,
    )

    text = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=8,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
    )
    config = Gemma3Config(
        text_config=text.to_dict(), vision_config=vision.to_dict(), mm_tokens_per_image=16
    )
    model = Gemma3ForConditionalGeneration(config)
    model.config.temperature = 0.0
    model.generation_config.top_k = 0
    generation = model.generation_config
    saves = sc.GenerationSaveContract()
    report = saves.check_before_compute(model)
    assert report["migration"] == {"temperature": 0.0}
    assert set(report["normalization"]) == {"temperature", "top_k"}
    with saves.saving(model, tmp_path / "gemma"):
        model.save_pretrained(tmp_path / "gemma")
    GenerationConfig.from_pretrained(tmp_path / "gemma", local_files_only=True).validate(
        strict=True
    )
    assert (tmp_path / "gemma/model.safetensors").is_file()
    assert model.config is config and model.generation_config is generation
    assert model.config.temperature == 0.0 and generation.top_k == 0
    # The historical g01r06 merge loads its model with device_map="cpu".
    loaded = Gemma3ForConditionalGeneration.from_pretrained(
        tmp_path / "gemma", device_map="cpu", local_files_only=True
    )
    with saves.saving(loaded, tmp_path / "gemma-resaved"):
        loaded.save_pretrained(tmp_path / "gemma-resaved")
    assert (tmp_path / "gemma-resaved/model.safetensors").is_file()
    loaded.hf_device_map = {"model": "cpu", "lm_head": "disk"}
    with pytest.raises(sc.UnsupportedSavePath, match="device maps"):
        saves.check_before_compute(loaded)
