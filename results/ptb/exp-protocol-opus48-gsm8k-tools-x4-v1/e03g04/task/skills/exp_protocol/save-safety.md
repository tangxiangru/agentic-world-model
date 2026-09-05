# Actual save safety: opt-in native Transformers adapter

Use this for a model save, not as a blanket evaluation/card gate. A parent
`generation_config.json` can be invalid for Transformers serialization but valid
for the selected serving setup, or already repaired in memory. No parent file is
read by this helper. Importing it does not import Torch/Transformers.

## Calls and ordering

Create/check/lock the experiment card before model-executing work. After model
construction and your in-code config repairs, check before expensive training or
merge accumulation. The check only copies/config-validates; it does no forwards.

```python
from awm.exp_protocol.save_contract import GenerationSaveContract
saves = GenerationSaveContract(policy="inactive_sampling_v1")
report = saves.check_before_compute(model)
with saves.saving(model, checkpoint_directory) as event:
    model.save_pretrained(checkpoint_directory)
```

The context observes exactly one native save to the declared output. It checks
again at that call, not only when entering the scope. An empty scope, swallowed
writer failure, different output, or config changed after saving is not success.
The instance save method is temporarily instrumented; no global class is patched.

For ordinary native Trainer, replace its constructor explicitly:

```python
from awm.exp_protocol.save_trainer import SaveSafeTrainer
trainer = SaveSafeTrainer(model=model, args=training_args,
                         generation_save_contract=saves)
saves.check_before_compute(trainer.model)  # after your repairs, before train
# Existing locked training code may now call trainer.train().
# Both intermediate _save_checkpoint and final save_model traverse the adapter.
trainer.save_model(checkpoint_directory)
```

Trainer outputs remain serializable checkpoints. For a selected serving export,
provide the frozen bytes you actually selected; the helper never chooses decoding:

```python
selected = selected_generation_config_path.read_bytes()
with saves.saving(model, export_directory, selected_serving_json=selected):
    model.save_pretrained(export_directory)
assert (export_directory / "generation_config.json").read_bytes() == selected
```

Tokenizer/processor completeness and selection's actual artifact/config hash
still need separate verification. Do not call a normalized checkpoint the selected
serving artifact unless its on-disk serving settings match the selection record.

## What is protected

The helper deep-copies both live configuration objects and reproduces native
model-config-to-generation-config migration before strict validation. Valid configs
are unchanged. Only an invalid `do_sample is False` config may neutralize offending
sampling fields: temperature 1, top_k 50, top_p/typical_p 1, min_p null,
epsilon_cutoff/eta_cutoff 0. Strict validation runs again; unrelated errors fail.
Pinned 4.57.3 does not exempt contrastive `penalty_alpha` from the top_k predicate.
Original config references/nested values are restored after each native save,
including writer exceptions and interruption. Token IDs/cache/other fields are
not reset. A non-generation model does not acquire a generation-config requirement.

Selected JSON must be frozen bytes containing one unambiguous finite JSON object;
duplicate keys, NaN/Infinity and numeric overflow are rejected. Successful native
serialization precedes atomic replacement of the individual serving JSON file.
Its exact byte hash is verified; export-file failure leaves an incomplete export.
This does not validate selected settings for a serving engine or make a whole
checkpoint atomic. Serving JSON can intentionally remain non-serializable by HF.

Each invocation has a unique audit in OUTPUT_PARENT/.exp-protocol-save-events
(or explicit `audit_dir`), plus `saves.records`. Records include source identity,
input/effective/serializer hashes, migration/normalization, output, outcome, and
separate serialized/selected-serving file hashes. Failed validation retains its
partial projection. Audit errors fail successful saves; if a writer already failed,
its original exception survives and in-memory evidence records the audit failure.
An audit's native-model-save success is not whole-Trainer-checkpoint completion.

## Supported boundary and known gaps

Supported: pinned Transformers 4.57.3 native PreTrainedModel saves and ordinary
single-process SaveSafeTrainer, CPU/CUDA, including an unwrapped whole-model
single-device map (`device_map="cpu"` in the historical merge). Source hashes and
native APIs are checked. Remote/custom model/save/config code, PEFT, quantization,
sharded/offloaded maps, FSDP, DeepSpeed, TP, TPU/XLA, SageMaker, model factories,
wrapped/compiled models, Trainer subclasses, config-suppressed saves, and Hub
writes are unsupported. Unsupported constructor settings are rejected before
calling the native Trainer constructor. Replaced native Trainer models are resolved
and checked again at save time. Generation-capable models need GenerationConfig.

Exclusive model use is required. Reentry/concurrent guarded access to a model or
output is rejected; uncooperative concurrent generation cannot be made safe.
Free-form scripts, class-method calls and other processes can bypass the utility.
Imports/declarations/check reports cannot prove launch-wide save coverage. These
audits are local operational evidence, not a tamper-proof security boundary.
Disk/OOM/optimizer/weight correctness, process death, and arbitrary serializer
side effects remain outside the contract. Neither CPU tests nor these helpers
establish score gains, scientific completion, or permission to launch experiments.

## Construction evidence and reproduction (2026-09-04)

Original repository Python: 32 unit/control tests passed; 15 skip reports identify
missing Torch (14 contract cases plus the Trainer module containing 5 tests).
The pinned runtime suite passed 51 tests: 32 controls and 19 real native cases.
Native CPU tests use image-installed Torch
2.8.0+cu129/Transformers 4.57.3 with no GPU devices/network. They serialize tiny
random GPT2 and Gemma3 models, reload only local configs/tiny CPU weights, exercise
actual Trainer checkpoint/final/state-dict paths, and inject failures. No training,
evaluation, forward passes, model downloads or package installs are involved.

The image's runtime libraries were extracted with `unsquashfs -processors 4
-no-progress -o 57344 -d RUNTIME/rootfs IMAGE usr/bin/python3.10
usr/lib/python3.10 usr/lib/x86_64-linux-gnu usr/local/lib/python3.10/dist-packages`.
IMAGE is `/rmeng_data/robtang/ptb-containers/opus_5.sif`; receipt image SHA256 is
`35f287e7b17d62ab44cd95db26dfeeac166943daed5f7b557b008bae51acc759` (not rehashed).
Pinned source SHA256 values are in `save_contract.SOURCE_HASHES`:
generation/configuration_utils.py (strict validation 528–725),
configuration_utils.py (migration 1149–1191), modeling_utils.py (save 3913–3930),
trainer.py (checkpoint 3325, native save 4305–4346; on_save 3229 is too late).

Exact native reproduction from the isolated worktree; pytest is borrowed as pure
Python from the original environment, after the pinned image's dependency paths:

```bash
save_repo=/rmeng_data/robtang/exp-protocol-bundle-work-5iV6EzGB/repo
save_runtime=/tmp/exp-protocol-save-runtime.JEZlHo
cd "$save_repo"
bwrap --ro-bind / / --unshare-net --dev /dev \
  --bind "$save_runtime" "$save_runtime" \
  --ro-bind "$save_runtime/rootfs/usr/bin/python3.10" /usr/bin/python3.10 \
  --ro-bind "$save_runtime/rootfs/usr/lib/python3.10" /usr/lib/python3.10 \
  --ro-bind "$save_runtime/rootfs/usr/lib/x86_64-linux-gnu" /usr/lib/x86_64-linux-gnu \
  --ro-bind "$save_runtime/rootfs/usr/local/lib/python3.10" /usr/local/lib/python3.10 \
  --setenv TMPDIR "$save_runtime" --setenv CUDA_VISIBLE_DEVICES '' \
  --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv PYTHONDONTWRITEBYTECODE 1 --setenv PYTHONNOUSERSITE 1 \
  --setenv PYTHONPATH "$save_repo" /usr/bin/python3.10 -c \
  'import sys; sys.path.append("/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/lib/python3.13/site-packages"); import pytest; raise SystemExit(pytest.main(["tests/test_exp_protocol_save_contract.py", "tests/test_exp_protocol_save_trainer.py", "-ra", "-p", "no:cacheprovider"]))'
```
