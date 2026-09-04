"""Real CPU native Trainer save/checkpoint integration; no train/evaluate calls."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel, TrainingArguments

from awm.exp_protocol.save_contract import GenerationSaveContract, UnsupportedSavePath
from awm.exp_protocol.save_trainer import SaveSafeTrainer


@pytest.fixture
def trainer(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=8))
    model.generation_config.temperature = 0.0
    args = TrainingArguments(
        output_dir=str(tmp_path / "training"),
        use_cpu=True,
        report_to=[],
        save_strategy="no",
        save_only_model=True,
    )
    saves = GenerationSaveContract()
    return SaveSafeTrainer(model=model, args=args, generation_save_contract=saves)


def test_native_trainer_checkpoint_and_final_without_training(trainer, tmp_path):
    model = trainer.model
    config, generation = model.config, model.generation_config
    assert trainer.generation_save_contract.check_before_compute(model)["status"] == "normalizable"
    trainer.state.global_step = 3
    trainer._save_checkpoint(model, trial=None)
    trainer.save_model(str(tmp_path / "final"))
    for output in (tmp_path / "training/checkpoint-3", tmp_path / "final"):
        assert (output / "model.safetensors").is_file()
        assert (output / "training_args.bin").is_file()
        GenerationConfig.from_pretrained(output, local_files_only=True).validate(strict=True)
    assert (tmp_path / "training/checkpoint-3/trainer_state.json").is_file()
    assert model.config is config and model.generation_config is generation
    assert generation.temperature == 0.0
    assert len(trainer.generation_save_contract.records) == 2
    assert all(r["outcome"] == "saved" for r in trainer.generation_save_contract.records)


def test_native_trainer_resolves_replacement_model_and_preserves_processing_save(trainer, tmp_path):
    replacement = GPT2LMHeadModel(trainer.model.config)
    replacement.generation_config.top_k = 0
    trainer.model = replacement

    class Processing:
        def save_pretrained(self, output):
            (tmp_path / "processing_called.json").write_text(json.dumps({"output": output}))

    trainer.processing_class = Processing()
    trainer.save_model(str(tmp_path / "replaced"))
    assert (tmp_path / "processing_called.json").is_file()
    assert trainer.generation_save_contract.records[-1]["projection"]["normalization"] == {
        "top_k": {"before": 0, "after": 50}
    }
    assert replacement.generation_config.top_k == 0


def test_native_trainer_rejects_unsupported_backend_and_restores_on_processing_error(
    trainer, tmp_path
):
    trainer.is_fsdp_enabled = True
    with pytest.raises(UnsupportedSavePath):
        trainer.save_model(str(tmp_path / "unsupported"))
    trainer.is_fsdp_enabled = False
    original = trainer.model.generation_config

    class Processing:
        def save_pretrained(self, output):
            raise OSError("processing failed")

    trainer.processing_class = Processing()
    with pytest.raises(OSError, match="processing failed"):
        trainer.save_model(str(tmp_path / "failed"))
    assert trainer.model.generation_config is original
    assert trainer.generation_save_contract.records[-1]["outcome"] == "failed"


def test_native_trainer_rejects_hub_before_native_constructor(trainer, monkeypatch):
    import awm.exp_protocol.save_trainer as adapter

    trainer.args.push_to_hub = True

    def forbidden_model_check(*args, **kwargs):
        pytest.fail("model check ran before rejecting Hub settings")

    monkeypatch.setattr(adapter, "_require_model", forbidden_model_check)
    with pytest.raises(UnsupportedSavePath):
        SaveSafeTrainer(model=trainer.model, args=trainer.args)


def test_native_trainer_preserves_explicit_state_dict(trainer, tmp_path):
    from safetensors.torch import load_file

    state = {key: value.clone().zero_() for key, value in trainer.model.state_dict().items()}
    trainer._save(str(tmp_path / "explicit-state"), state_dict=state)
    actual = load_file(tmp_path / "explicit-state/model.safetensors")
    assert actual and all(value.count_nonzero().item() == 0 for value in actual.values())
