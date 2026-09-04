"""Opt-in adapter for ordinary single-process native Trainer model saves only."""

import inspect
from pathlib import Path

from transformers import Trainer, TrainingArguments
from transformers.trainer import is_sagemaker_mp_enabled, is_torch_xla_available

from .save_contract import (
    GenerationSaveContract,
    UnsupportedSavePath,
    _require_model,
    validate_runtime,
)


def _require_trainer_api():
    validate_runtime()
    if is_sagemaker_mp_enabled() or is_torch_xla_available():
        raise UnsupportedSavePath("SageMaker and TPU/XLA Trainer paths are unsupported")
    source = Path(inspect.getsourcefile(Trainer)).resolve()
    for method in (Trainer._save, Trainer.save_model, Trainer._save_checkpoint):
        if Path(inspect.getsourcefile(method) or "").resolve() != source:
            raise UnsupportedSavePath("native Trainer save API has been replaced")


class SaveSafeTrainer(Trainer):
    """Preserve native _save behavior, instrumenting its actual model serializer.

    Explicitly call ``generation_save_contract.check_before_compute(trainer.model)``
    after repairs and before train. This adapter never trains or evaluates itself.
    Final ``save_model`` outputs are serializable checkpoints, not selected serving
    exports. Use a separate explicit ``saving(..., selected_serving_json=bytes)``
    transaction for a serving export.
    """

    def __init__(self, *args, generation_save_contract=None, **kwargs):
        _require_trainer_api()
        arguments = (
            inspect.signature(Trainer.__init__).bind_partial(None, *args, **kwargs).arguments
        )
        model = arguments.get("model")
        settings = arguments.get("args")
        # Trainer.__init__ can create a Hub repository. Reject before calling it.
        if (
            type(settings) is not TrainingArguments
            or settings.push_to_hub
            or settings.deepspeed
            or settings.fsdp
            or settings.world_size != 1
            or model is None
            or arguments.get("model_init") is not None
        ):
            raise UnsupportedSavePath(
                "explicit native model/TrainingArguments and local single-process saves required"
            )
        _require_model(model)
        self.generation_save_contract = generation_save_contract or GenerationSaveContract()
        super().__init__(*args, **kwargs)
        self._require_native_trainer()

    def _require_native_trainer(self):
        _require_trainer_api()
        if (
            type(self) is not SaveSafeTrainer
            or self.args.world_size != 1
            or self.is_fsdp_enabled
            or self.is_deepspeed_enabled
            or getattr(self.accelerator, "parallelism_config", None) is not None
            or self.args.push_to_hub
            or str(self.accelerator.distributed_type).split(".")[-1] != "NO"
            or str(self.args.device).split(":")[0] not in ("cpu", "cuda")
        ):
            raise UnsupportedSavePath(
                "only ordinary single-process native Trainer saves are supported"
            )
        # Do not certify a wrapper or stale trainer.model: resolve on each save.
        resolved = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if resolved is not self.model:
            raise UnsupportedSavePath("wrapped/compiled Trainer models are unsupported")
        _require_model(resolved)
        return resolved

    def save_model(self, output_dir=None, _internal_call=False):
        self._require_native_trainer()
        return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save(self, output_dir=None, state_dict=None):
        model = self._require_native_trainer()
        destination = Path(output_dir if output_dir is not None else self.args.output_dir)
        with self.generation_save_contract.saving(model, destination):
            return super()._save(output_dir=str(destination), state_dict=state_dict)
