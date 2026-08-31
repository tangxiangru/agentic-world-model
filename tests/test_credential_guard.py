"""Fail-closed credential checks for raw historical study inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from awm import credential_guard as package_guard
from awm.wm.agents.llm import _validate_raw_corpus
from awm.wm.schema import WMError

REPO = Path(__file__).resolve().parent.parent
REVISION = "39d3fcd794df51c062c8bd3b7f8523ba707aaeb3"
DATASET = {
    "repo": "example/PostTrainBench-Trajectories",
    "repo_type": "dataset",
    "revision": REVISION,
}
RUN = "cfg_a/gsm8k_google_gemma-3-4b-pt_1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def standalone_guard():
    return _load(REPO / "rollout" / "validate_study_corpus.py", "standalone_corpus_guard")


@pytest.mark.parametrize(
    ("data", "rule_id"),
    [
        (b"CLAUDE_CODE_MESSAGING_TOKEN=a-credential-value-12345\n", "secret-env-assignment"),
        (b"anthropic_api_key=a-credential-value-12345\n", "secret-env-assignment"),
        (b"claude_code_oauth_token=a-credential-value-12345\n", "secret-env-assignment"),
        (b"export Anthropic_Api_Key=a-credential-value-12345\n", "secret-env-assignment"),
        (b"Hf_ToKeN=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"Vllm_Api_Key=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"SgLang_Api_Key=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"VLLM_API_KEY=a-credential-value-12345\n", "secret-env-assignment"),
        (b"HF_TOKEN=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"hf_abcdefghijklmnopqrstuvwxyz123456\n", "huggingface-token"),
        (b"sk-abcdefghijklmnopqrstuvwxyz123456\n", "secret-key-token"),
        (b"ya29.abcdefghijklmnopqrstuvwxyz123456\n", "google-oauth-token"),
        (b"ghp_abcdefghijklmnopqrstuvwxyz123456\n", "github-token"),
        (b'{"refreshToken":"a-credential-value-12345"}\n', "secret-json-field"),
        (b"{'client_secret': 'alphabeticonlysecret'}\n", "secret-json-field"),
        (b"client_secret: a-credential-value-12345\n", "secret-text-field"),
        (b"Authorization: Bearer abcdefghijklmnopqrstuvwxyz\n", "authorization-bearer"),
        (b"-----BEGIN PRIVATE KEY-----\n", "private-key-block"),
        (b"-----BEGIN ENCRYPTED PRIVATE KEY-----\n", "private-key-block"),
        (b"-----BEGIN PGP PRIVATE KEY BLOCK-----\n", "private-key-block"),
        (b"github_pat_abcdefghijklmnopqrstuvwxyz123456\n", "github-token"),
        (b"AUTHTOKEN=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"CLIENT_SECRET_VALUE=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"ACCESS_TOKEN_VALUE=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"AWS_SECRET_ACCESS_KEY=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP\n", "secret-env-assignment"),
        (b"ACCESS_KEY_ID=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"ACCESS_KEY=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"API_KEY_VALUE=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"PRIVATE_KEY_VALUE=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"ACCESS_KEY_VALUE=alphabeticonlysecret\n", "secret-env-assignment"),
        (b"AKIAABCDEFGHIJKLMNOP\n", "aws-access-key-id"),
        (b'{"client_secret_value":"alphabeticonlysecret"}\n', "secret-json-field"),
        (b'{"apiKeyValue":"alphabeticonlysecret"}\n', "secret-json-field"),
        (rb"{\'client_secret\':\'alphabeticonlysecret\'}", "secret-json-field"),
        (b"tool --api-key alphabeticonlysecret\n", "secret-cli-argument"),
        (
            b"('aUtHoRiZaTiOn', 'bEaReR abcdefghijklmnopqrstuvwxyz')\n",
            "authorization-bearer",
        ),
        (
            rb'{\"OPENAI_API_KEY\":\"sk\u002dproj\u002dabcdefghijklmnopqrstuvwx_1234\"}',
            "secret-json-field",
        ),
        (
            rb'{"OPENAI_API_KEY":"s\u006b\u002dproj\u002dabcdefghijklmnopqrstuvwx_1234"}',
            "secret-json-field",
        ),
    ],
)
def test_package_and_standalone_scanners_have_identical_findings(
    standalone_guard, data: bytes, rule_id: str
) -> None:
    package = package_guard.scan_credential_bytes(data, path="run/solve_out.txt")
    standalone = standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt")
    assert standalone == package
    assert rule_id in {str(row["rule_id"]) for row in package}


def test_package_and_standalone_rule_contracts_are_identical(standalone_guard) -> None:
    assert standalone_guard.credential_ruleset_contract() == (
        package_guard.credential_ruleset_contract()
    )


@pytest.mark.parametrize(
    "data",
    [
        b"API_KEY=<omitted-api-key>\nHF_TOKEN=UNDEFINED\n",
        b'{"api_key":"<omitted-api-key>","access_token":"UNDEFINED"}\n',
        b"client_secret: <omitted-api-key>\ntoken: UNDEFINED\n",
        b"tool --api-key <omitted-api-key> --token UNDEFINED\n",
        b'MAX_TOKENS=32768\n{"input_tokens":"1200","apiKeySource":"none"}\n',
        b'HF_TOKEN="$HF_TOKEN"\nCLAUDE_CODE_OAUTH_TOKEN=${CLAUDE_CODE_OAUTH_TOKEN}\n',
        b'anthropic_api_key="$ANTHROPIC_API_KEY"\nexport Anthropic_Api_Key=${ANTHROPIC_API_KEY}\n',
        b"pad_token=<|pad|>\neos_token=tokenizer.eos_token\napi_key=a-credential-value-12345\n",
        b"PAD_TOKEN=alphabeticonlyvalue\nSPACE_TOKEN=anotheralphabeticvalue\n",
        b'VLLM_API_KEY=<your-api-key>\nVLLM_API_KEY=os.environ["VLLM_API_KEY"]\n',
        b'{"api_key":"<YOUR_API_KEY>","client_secret":"os.getenv(\\"client_secret\\")"}\n',
        b"OPENAI_API_KEY=local-test-stub\n",
        b"OpenAi_Api_Key=local-test-stub\n",
        b"VLLM_API_KEY=inspectai\n",
        b"Vllm_Api_Key=inspectai\n",
        b'VLLM_API_KEY=inspectai"}\n',
        b'VLLM_API_KEY=inspectai","is_error":false\n',
        rb"VLLM_API_KEY=inspectai\"}\n",
        rb"VLLM_API_KEY=os.environ[\"VLLM_API_KEY\"]\n",
        b"MAX_TOKENS=32768\nCLIENT_SECRET_VALUE=<your-secret>\n",
        b"tool --api-key <omitted-api-key> --max-tokens 32768\n",
        b"tool --api-key inspectai --max-tokens 32768\n",
        (
            b"NUM_TOKEN_RE=compiled-pattern\n"
            b"tool --response-stop-token stop --stop-token-id 1 --token-budget 10000000\n"
        ),
    ],
)
def test_audited_placeholders_and_token_telemetry_pass(standalone_guard, data: bytes) -> None:
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == []
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == []


@pytest.mark.parametrize(
    "data",
    [
        b"OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456\n",
        b"OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwx_1234\n",
        b"OPENAI_API_KEY=sk-svcacct-abcdefghijklmnopqrstuvwx_1234\n",
        b"OPENAI_API_KEY=sk-admin-abcdefghijklmnopqrstuvwx_1234\n",
    ],
)
def test_openai_api_key_credible_provider_shapes_are_rejected(
    standalone_guard, data: bytes
) -> None:
    for guard in (package_guard, standalone_guard):
        findings = guard.scan_credential_bytes(data, path="run/solve_out.txt")
        assert "secret-env-assignment" in {row["rule_id"] for row in findings}


@pytest.mark.parametrize(
    "data",
    [
        b"OPENAI_API_KEY=${OPENAI_API_KEY:-local-stub}\n",
        b'OPENAI_API_KEY=\'os.getenv("OPENAI_API_KEY", "local-stub")\'\n',
        b'OPENAI_API_KEY=os.getenv("OPENAI_API_KEY","local-stub")\n',
        b'OPENAI_API_KEY="$OPENAI_API_KEY-suffix"\n',
        b"OPENAI_API_KEY=`printenv OPENAI_API_KEY`\n",
        b'OPENAI_API_KEY=\'os.environ["OPENAI_API_KEY"] + "suffix"\'\n',
        b"HF_TOKEN=${HF_TOKEN:-alphabeticfallback}\n",
        b'HF_TOKEN=os.getenv("HF_TOKEN"),"literalfallback"\n',
        b'HF_TOKEN="HF_TOKEN"suffix\n',
        b'HF_TOKEN="HF_TOKEN",malicioussuffix\n',
        b'HF_TOKEN="$HF_TOKEN"+suffix\n',
        b'HF_TOKEN=`HF_TOKEN`\n',
        rb'HF_TOKEN=\"HF_TOKEN\"suffix\n',
        rb'HF_TOKEN=short\escapedmaterial\n',
        rb'HF_TOKEN=\escapedmaterial\n',
        rb"HF_TOKEN=\"HF_TOKEN-suffix\"\n",
        rb"HF_TOKEN=\"${HF_TOKEN:-literal}\"\n",
    ],
)
def test_nonexact_self_reference_expressions_remain_material(standalone_guard, data: bytes) -> None:
    package = package_guard.scan_credential_bytes(data, path="run/solve_out.txt")
    standalone = standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt")
    assert standalone == package
    assert package == [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-env-assignment",
            "count": 1,
        }
    ]


def test_json_container_suffix_must_be_structurally_valid(standalone_guard) -> None:
    rejected = (
        b'{"api_key":"<your-api-key>"},material\n',
        b'{"api_key":"<your-api-key>"},"unterminated material\n',
        b'{"api_key":"<your-api-key>"],"unterminated material\n',
        b'{"api_key":"<your-api-key>"},"x":unquoted-material\n',
        b'{"api_key":"<your-api-key>","x":}\n',
        b'{"api_key":"<your-api-key>","x":unquoted-material}\n',
        rb'{"api_key":"<your-api-key>","x":"mismatched\"}\n',
        rb'{"api_key":"<your-api-key>",\"x":"mismatched"}\n',
        b'{"api_key":"<your-api-key>","x\':"mismatched"}\n',
    )
    expected = [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-json-field",
            "count": 1,
        }
    ]
    for data in rejected:
        assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected
        assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected


@pytest.mark.parametrize(
    "data",
    [
        b"HF_TOKEN=$HF_TOKEN\n",
        b"HF_TOKEN=${HF_TOKEN}\n",
        b"HF_TOKEN='os.environ[\"HF_TOKEN\"]'\n",
        b"HF_TOKEN='os.getenv(\"HF_TOKEN\")'\n",
        rb"HF_TOKEN=\"HF_TOKEN\"\n",
        rb"SGLANG_API_KEY=\"SGLANG_API_KEY\"\n",
    ],
)
def test_exact_self_references_are_exempt(standalone_guard, data: bytes) -> None:
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == []
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == []


def test_encoded_newline_ends_assignment_without_hiding_next_secret(standalone_guard) -> None:
    data = rb"VLLM_API_KEY=\"VLLM_API_KEY\"\nHF_TOKEN=alphabeticonlysecret\n"
    expected = [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-env-assignment",
            "count": 1,
        }
    ]
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected


def test_key_specific_local_sentinel_does_not_exempt_other_tokens(standalone_guard) -> None:
    data = b"HF_TOKEN=inspectai\n"
    expected = [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-env-assignment",
            "count": 1,
        }
    ]
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected


def test_angle_placeholder_and_unicode_escaped_provider_shape(standalone_guard) -> None:
    rejected = (
        b"HF_TOKEN=<alphabeticmaterial>\n"
        b'{"OPENAI_API_KEY":"sk\\u002dproj\\u002dabcdefghijklmnopqrstuvwx_1234"}\n'
    )
    allowed = b"HF_TOKEN=<HF_TOKEN>\nHF_TOKEN=<your-token>\n"
    for guard in (package_guard, standalone_guard):
        findings = guard.scan_credential_bytes(rejected, path="run/solve_out.txt")
        assert sum(int(row["count"]) for row in findings) >= 2
        assert guard.scan_credential_bytes(allowed, path="run/solve_out.txt") == []


def test_non_ascii_sensitive_values_fail_closed(standalone_guard) -> None:
    data = b"HF_TOKEN=" + (b"\xff" * 12) + b"\n"
    for guard in (package_guard, standalone_guard):
        assert guard.scan_credential_bytes(data, path="run/solve_out.txt") == [
            {
                "path": "run/solve_out.txt",
                "rule_id": "secret-env-assignment",
                "count": 1,
            }
        ]


def test_operational_token_names_remain_sensitive(standalone_guard) -> None:
    data = b"ACCESS_TOKEN=alphabeticonlysecret\nAUTH_TOKEN=alphabeticonlysecret\n"
    expected = [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-env-assignment",
            "count": 2,
        }
    ]
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected


def test_sensitive_fragment_position_does_not_bypass_candidate_filter(
    standalone_guard,
) -> None:
    suffix = b"_" + (b"X" * 60)
    data = b"CLIENT_SECRET" + suffix + b"=alphabeticonlysecret\n"
    expected = [
        {
            "path": "run/solve_out.txt",
            "rule_id": "secret-env-assignment",
            "count": 1,
        }
    ]
    assert package_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected
    assert standalone_guard.scan_credential_bytes(data, path="run/solve_out.txt") == expected


def test_rejection_message_never_contains_matched_value() -> None:
    secret = "a-credential-value-that-must-not-be-logged-12345"
    findings = package_guard.scan_credential_bytes(
        f"HF_TOKEN={secret}\n".encode(), path="cfg/run/solve_out.txt"
    )
    message = package_guard.format_credential_rejection(findings)
    assert secret not in json.dumps(findings)
    assert secret not in message
    assert "rule_id=secret-env-assignment" in message
    assert 'path="cfg/run/solve_out.txt"' in message
    assert "count=1" in message


def test_mixed_case_env_rejection_never_contains_matched_value(standalone_guard) -> None:
    secret = "mixed-case-credential-value-that-must-not-be-logged-12345"
    data = f"export Anthropic_Api_Key={secret}\n".encode()
    for guard in (package_guard, standalone_guard):
        findings = guard.scan_credential_bytes(data, path="cfg/run/solve_out.txt")
        message = guard.format_credential_rejection(findings)
        assert findings == [
            {
                "path": "cfg/run/solve_out.txt",
                "rule_id": "secret-env-assignment",
                "count": 1,
            }
        ]
        assert secret not in json.dumps(findings)
        assert secret not in message


def _raw_source(root: Path, trace: bytes) -> Path:
    run = root / RUN
    run.mkdir(parents=True)
    (run / "solve_out.txt").write_bytes(trace)
    (run / "metrics.json").write_text(json.dumps({"accuracy": 0.5}))
    (run / "time_taken.txt").write_text("01:00:00\n")
    return root


def _builder():
    module = _load(REPO / "tools" / "build_prior_runs.py", "credential_guard_builder")
    # This suite isolates credential handling; source-revision verification has
    # its own rollout tests and would require unrelated Hub sidecars here.
    module._verify_source_revision = lambda *_args, **_kwargs: None
    return module


def _build(module, raw: Path, out: Path, **kwargs):
    return module.build(
        [(RUN, "train")],
        raw,
        out,
        split_id="example/split-v1",
        dataset=DATASET,
        sides=("train",),
        **kwargs,
    )


def test_builder_rejects_before_copy_and_does_not_publish(tmp_path: Path) -> None:
    module = _builder()
    secret = "a-credential-value-that-must-not-be-logged-12345"
    raw = _raw_source(tmp_path / "raw", f"HF_TOKEN={secret}\n".encode())
    out = tmp_path / "prior-runs"
    with pytest.raises(module.PriorRunsError) as caught:
        _build(module, raw, out)
    assert secret not in str(caught.value)
    assert "rule_id=secret-env-assignment" in str(caught.value)
    assert not out.exists()


def test_builder_rescans_copied_bytes_before_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _builder()
    raw = _raw_source(tmp_path / "raw", b"ordinary complete trajectory\n")
    out = tmp_path / "prior-runs"
    real_copy = module.shutil.copy2

    def copy_then_inject(source: Path, destination: Path):
        result = real_copy(source, destination)
        if Path(destination).name == "solve_out.txt":
            Path(destination).write_text("HF_TOKEN=a-credential-value-in-staging-12345\n")
        return result

    monkeypatch.setattr(module.shutil, "copy2", copy_then_inject)
    with pytest.raises(module.PriorRunsError, match="raw corpus credential guard rejected"):
        _build(module, raw, out)
    assert not out.exists()


def _replace_trace_and_reattest(out: Path, replacement: bytes) -> str:
    trace = out / RUN / "solve_out.txt"
    original_size = trace.stat().st_size
    assert len(replacement) <= original_size
    trace.write_bytes(replacement + b" " * (original_size - len(replacement)))
    manifest_path = out / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["runs"][0]["files"]["solve_out.txt"]
    record["bytes"] = trace.stat().st_size
    record["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def test_standalone_and_wma_rescan_fully_attested_raw_files(
    tmp_path: Path, standalone_guard
) -> None:
    module = _builder()
    raw = _raw_source(tmp_path / "raw", b"ordinary trajectory\n" + b"." * 200)
    out = tmp_path / "prior-runs"
    _build(module, raw, out)
    secret = b"HF_TOKEN=a-credential-value-that-is-attested-12345\n"
    digest = _replace_trace_and_reattest(out, secret)

    with pytest.raises(standalone_guard.ValidationError) as standalone_error:
        standalone_guard.validate_raw(out, ("train",), digest)
    with pytest.raises(WMError) as wma_error:
        _validate_raw_corpus(out, ("train",))
    for error in (standalone_error.value, wma_error.value):
        assert secret.decode().strip() not in str(error)
        assert "rule_id=secret-env-assignment" in str(error)


def test_builder_index_only_rescans_existing_attested_copy(tmp_path: Path) -> None:
    module = _builder()
    raw = _raw_source(tmp_path / "raw", b"ordinary trajectory\n" + b"." * 200)
    out = tmp_path / "prior-runs"
    _build(module, raw, out)
    _replace_trace_and_reattest(out, b"HF_TOKEN=a-credential-value-in-existing-copy-12345\n")
    with pytest.raises(module.PriorRunsError, match="rule_id=secret-env-assignment"):
        _build(module, raw, out, copy=False)
