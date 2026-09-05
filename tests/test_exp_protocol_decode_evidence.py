"""Selected-file/request evidence and native CPU metadata, never model inference."""

import json

import pytest

from awm.exp_protocol import decode_evidence as decode


def selected(tmp_path, fields):
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps(fields, indent=2) + "\n")
    return path


def test_do_sample_only_is_not_greedy_and_file_is_preserved_exactly(tmp_path):
    path = selected(tmp_path, {"do_sample": False, "eos_token_id": [1, 106]})
    before = path.read_bytes()
    report = decode.freeze_decode_evidence(path, tmp_path / "evidence", intent="greedy")
    assert report["selected"]["mode_from_explicit_temperature"] == "unknown"
    assert report["effective_engine_decode"] == "unknown"
    assert report["native_defaults"]["status"] == "unknown"
    assert (
        path.read_bytes()
        == (tmp_path / "evidence/selected-generation-config.json").read_bytes()
        == before
    )
    assert decode.verify_decode_evidence(tmp_path / "evidence")["status"] == "unchanged_evidence"


def test_actual_request_is_located_hashed_and_only_decode_fields_retained(tmp_path):
    path = selected(tmp_path, {"do_sample": False, "temperature": 0})
    request = tmp_path / "inspect.json"
    request.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "request": {
                            "temperature": 0.8,
                            "max_tokens": 100,
                            "messages": "do not copy prompts",
                            "extra_headers": {"token": "do not copy headers"},
                            "extra_body": {"top_k": 20},
                        }
                    }
                ]
            }
        )
    )
    report = decode.freeze_decode_evidence(
        path,
        tmp_path / "evidence",
        intent="greedy",
        request_path=request,
        request_pointer="/samples/0/request",
    )
    assert report["selected"]["mode_from_explicit_temperature"] == "greedy_requested"
    assert report["request"]["fields"] == {"temperature": 0.8, "max_tokens": 100}
    assert report["request"]["extra_body_fields"] == {"top_k": 20}
    saved = (tmp_path / "evidence/decode-evidence.json").read_text()
    assert "do not copy" not in saved
    assert report["effective_engine_decode"] == "unknown"  # No fictional precedence resolver.
    request.write_text(request.read_text() + " ")
    with pytest.raises(decode.DecodeEvidenceError, match="request source changed"):
        decode.verify_decode_evidence(tmp_path / "evidence")


def test_fresh_namespace_and_changed_selected_bytes_are_detected(tmp_path):
    path = selected(tmp_path, {"temperature": 0.8})
    decode.freeze_decode_evidence(path, tmp_path / "evidence")
    with pytest.raises(FileExistsError):
        decode.freeze_decode_evidence(path, tmp_path / "evidence")
    path.write_text('{"temperature":0}')
    with pytest.raises(decode.DecodeEvidenceError, match="selected serving JSON changed"):
        decode.verify_decode_evidence(tmp_path / "evidence")


@pytest.mark.parametrize("contents", ["[]", '{"temperature":NaN}', '{"temperature":1e999}'])
def test_invalid_selected_json_creates_no_success_evidence(tmp_path, contents):
    path = tmp_path / "generation_config.json"
    path.write_text(contents)
    with pytest.raises(decode.DecodeEvidenceError):
        decode.freeze_decode_evidence(path, tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()


def test_native_sampling_object_is_observed_but_not_claimed_engine_execution(tmp_path):
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    path = selected(tmp_path, {"do_sample": False})
    params = SamplingParams(temperature=0.7, max_tokens=32)
    report = decode.freeze_decode_evidence(
        path,
        tmp_path / "evidence",
        intent="greedy",
        native_request_params=params,
    )
    assert report["native_request"]["fields"]["temperature"] == 0.7
    assert report["native_request"]["mode_from_object"] == "sampling_requested"
    assert (
        report["native_request"]["connection_to_captured_request"] == "not_independently_verified"
    )
    assert params.temperature == 0.7 and report["effective_engine_decode"] == "unknown"


def test_real_native_defaults_ignore_do_sample_without_constructing_model(tmp_path):
    pytest.importorskip("vllm")
    from vllm.config import ModelConfig

    path = selected(tmp_path, {"do_sample": False, "eos_token_id": 1})
    # Real pinned resolver on CPU; no ModelConfig/LLM constructor or weights.
    config = ModelConfig.__new__(ModelConfig)
    config.model = str(tmp_path)
    config.hf_config_path = None
    config.generation_config = "auto"
    config.trust_remote_code = False
    config.revision = None
    config.override_generation_config = {}
    first = decode.freeze_decode_evidence(path, tmp_path / "first", native_model_config=config)
    assert "do_sample" not in first["native_defaults"]["fields"]
    assert "temperature" not in first["native_defaults"]["fields"]
    path.write_text('{"do_sample":false,"temperature":0.0}')
    second = decode.freeze_decode_evidence(path, tmp_path / "second", native_model_config=config)
    assert second["native_defaults"]["fields"]["temperature"] == 0
    config.generation_config = "vllm"
    with pytest.raises(decode.DecodeEvidenceError, match="selected model directory"):
        decode.freeze_decode_evidence(path, tmp_path / "bad", native_model_config=config)
