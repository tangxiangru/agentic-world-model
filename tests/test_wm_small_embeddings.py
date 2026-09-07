import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.outcome_prediction import wm_small_embeddings as wm


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[200 + sum(map(ord, word)) % 53 for word in text.split()])


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, outputs, inputs):
        assert outputs == ["last_hidden_state"]
        assert set(inputs) == {"input_ids", "attention_mask", "token_type_ids"}
        assert all(value.dtype == np.int64 for value in inputs.values())
        assert not inputs["token_type_ids"].any()
        self.calls.append(copy.deepcopy(inputs))
        ids = inputs["input_ids"]
        states = np.zeros((*ids.shape, wm.DIMENSION), dtype=np.float32)
        states[..., 0] = ids
        states[..., 1] = np.where(ids == 0, 10000, 1)  # padding must be masked out
        states[..., 2] = ids % 7
        return [states]


@pytest.fixture
def encoder(monkeypatch, tmp_path):
    paths = {}
    for name in wm.RESOURCE_SHA256:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic resource, not a model")
        paths[name] = path
    calls = []

    def resources(cache_dir, *, local_files_only):
        calls.append(local_files_only)
        return paths

    session = FakeSession()
    monkeypatch.setattr(wm, "prepare_resources", resources)
    monkeypatch.setattr(wm, "_load_backend", lambda paths: (FakeTokenizer(), session))
    monkeypatch.setattr(wm.metadata, "version", lambda name: "synthetic-version")
    obj = wm.FrozenTextEncoder()
    assert calls == [True]
    return obj, session


def test_chunks_have_fixed_limits_and_deterministic_prefix_truncation():
    assert wm._chunks([]) == [[101, 102]]
    assert wm._chunks([3, 4]) == [[101, 3, 4, 102]]
    ids = list(range(3000))
    chunks = wm._chunks(ids)
    assert len(chunks) == 8
    assert all(len(chunk) == 256 for chunk in chunks)
    assert [x for chunk in chunks for x in chunk[1:-1]] == ids[:2032]
    assert wm._chunks(ids) == chunks


def test_output_shape_dtype_norm_repeatability_and_padding_invariance(encoder):
    model, _ = encoder
    alone = model.encode_texts(["short"])
    together = model.encode_texts(["short", "another much longer piece of text", ""])
    assert together.shape == (3, 384) and together.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(alone[0], together[0], atol=1e-7)
    np.testing.assert_array_equal(
        together, model.encode_texts(["short", "another much longer piece of text", ""])
    )


def test_empty_batch_no_dispatch_and_metadata_copies(encoder):
    model, session = encoder
    result = model.encode_texts([])
    assert result.shape == (0, 384) and result.dtype == np.float32
    assert not session.calls
    metadata = model.metadata()
    assert metadata["last_encoding"]["texts"] == 0
    assert metadata["last_encoding"]["seconds_per_text"] is None
    metadata["resource_sha256"].clear()
    assert model.metadata()["resource_sha256"]


def test_chunk_cap_batch32_and_content_free_latency_metadata(encoder):
    model, session = encoder
    marker = "PRIVATE_EXPLICIT_RECIPE_TEXT"
    texts = [marker + " " + "word " * 2100] * 5
    encoded = model.encode_texts(texts)
    assert encoded.shape == (5, 384)
    assert [len(call["input_ids"]) for call in session.calls] == [32, 8]
    assert all(call["input_ids"].shape[1] <= 256 for call in session.calls)
    meta = model.metadata()
    assert meta["last_encoding"]["per_text_chunk_counts"] == [8] * 5
    assert meta["last_encoding"]["truncated_texts"] == 5
    assert meta["last_encoding"]["retained_content_tokens"] == 2032 * 5
    assert meta["last_encoding"]["elapsed_seconds"] >= 0
    assert marker not in repr(meta)
    assert meta["cpu_intra_op_threads"] == meta["cpu_inter_op_threads"] == 1


@pytest.mark.parametrize("bad", ["single string", b"bytes", [None], [1], [{"text": "no"}]])
def test_rejects_records_nontext_and_single_string_without_dispatch(encoder, bad):
    model, session = encoder
    with pytest.raises(TypeError):
        model.encode_texts(bad)
    assert not session.calls


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_invalid_backend_output_rejected(encoder, bad):
    model, session = encoder
    session.run = lambda names, inputs: [
        np.full((*inputs["input_ids"].shape, 384), bad, dtype=np.float32)
    ]
    with pytest.raises(ValueError):
        model.encode_texts(["synthetic"])


def test_prepare_resources_pins_revision_hashes_and_public_local_only(monkeypatch, tmp_path):
    import huggingface_hub

    paths, digests, calls = {}, {}, []
    for filename in wm.RESOURCE_SHA256:
        data = ("safe synthetic " + filename).encode()
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths[filename] = path
        digests[filename] = hashlib.sha256(data).hexdigest()

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(paths[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(wm, "RESOURCE_SHA256", digests)
    assert wm.prepare_resources() == paths
    assert all(repo == wm.MODEL_ID for repo, _, _ in calls)
    assert all(args["revision"] == wm.MODEL_REVISION for _, _, args in calls)
    assert all(args["local_files_only"] is True and args["token"] is False for _, _, args in calls)
    paths["onnx/model.onnx"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        wm.prepare_resources()


def test_default_function_uses_one_reusable_local_session(monkeypatch):
    class FakeEncoder:
        def __init__(self, *, local_files_only):
            assert local_files_only is True
            calls.append("loaded")

        def encode_texts(self, texts):
            return np.ones((len(texts), 384), dtype=np.float32)

        def metadata(self):
            return {"synthetic": True}

    calls = []
    wm.get_encoder.cache_clear()
    monkeypatch.setattr(wm, "FrozenTextEncoder", FakeEncoder)
    try:
        assert wm.encode_texts(["a"]).shape == (1, 384)
        assert wm.encode_texts(["b", "c"]).shape == (2, 384)
        assert wm.encoder_metadata() == {"synthetic": True}
        assert calls == ["loaded"]
    finally:
        wm.get_encoder.cache_clear()


def test_embedding_column_skips_unselected_values_and_preserves_order(monkeypatch):
    original = json.JSONDecoder.raw_decode

    def guarded(self, text, index=0):
        if text[index:].startswith('"PRIVATE_LABEL_POISON"'):
            raise AssertionError("Unselected label value decoded")
        return original(self, text, index)

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", guarded)
    raw = (
        b'[{"label":"PRIVATE_LABEL_POISON","embedding_text":"safe first"},'
        b'{"embedding_text":"safe second","ignored":{"nested":[1,2]}}]'
    )
    assert wm._embedding_text_column(raw) == ["safe first", "safe second"]
    assert wm._embedding_text_column(b"[]") == []


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[{}]",
        b'[{"embedding_text":null}]',
        b'[{"embedding_text":"a","embedding_text":"b"}]',
        b'[{"embedding_text":"a",}]',
        b'[{"embedding_text":"a"},]',
        b'[{"embedding_text":"a"}] trailing',
    ],
)
def test_bad_embedding_input_column_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        wm._embedding_text_column(raw)


def test_bundle_export_exact_hash_binding_and_no_label_reads(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    inputs = bundle / "inputs.json"
    inputs.write_text(
        json.dumps(
            [
                {"embedding_text": "safe first", "label": "DO_NOT_DECODE"},
                {"embedding_text": "safe second", "other": {"ignored": 1}},
            ]
        )
    )
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps({"inputs.json": wm._file_sha256(inputs)}))
    (bundle / "labels.json").write_text("MUST NEVER OPEN THIS")

    class Encoder:
        def encode_texts(self, texts):
            assert texts == ["safe first", "safe second"]
            return np.eye(2, 384, dtype=np.float32)

        def metadata(self):
            return {"synthetic": True}

    original = Path.read_bytes

    def guarded(path):
        if path.name == "labels.json":
            raise AssertionError("Label file was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    output = tmp_path / "output"
    meta = wm.embed_bundle(bundle, output, encoder=Encoder())
    assert meta["input_sha256"] == wm._file_sha256(inputs)
    assert meta["embeddings_sha256"] == wm._file_sha256(output / "embeddings.npy")
    assert meta["labels_read"] is False and meta["input_rows"] == 2
    assert json.loads((output / "metadata.json").read_text()) == meta
    assert np.load(output / "embeddings.npy", allow_pickle=False).shape == (2, 384)
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    with pytest.raises(FileExistsError):
        wm.embed_bundle(bundle, output, encoder=Encoder())
    inputs.write_text(inputs.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        wm.embed_bundle(bundle, tmp_path / "stale", encoder=Encoder())
    assert not (tmp_path / "stale").exists()
