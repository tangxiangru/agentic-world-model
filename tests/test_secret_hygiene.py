from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SANITIZER = REPO_ROOT / "rollout" / "sanitize_result_tree.py"
ATTESTATION = "secret-sanitization.json"


def run_sanitizer(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SANITIZER), str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_sanitizer_redacts_text_and_writes_value_free_attestation(tmp_path: Path) -> None:
    root = tmp_path / "result"
    nested = root / "task"
    nested.mkdir(parents=True)
    hf_value = "hf_" + "A" * 32
    assignment_value = "b" * 32
    bearer_value = "c" * 32
    solve_output = nested / "solve_out.txt"
    solve_output.write_text(
        "safe telemetry stays\n"
        f"cache read: {hf_value}\n"
        f"--env SERVICE_AUTH_TOKEN={assignment_value}\n"
        f"Authorization: Bearer {bearer_value}\n",
        encoding="utf-8",
    )
    solve_output.chmod(0o640)

    first = run_sanitizer(root)

    assert first.returncode == 3
    sanitized = solve_output.read_text(encoding="utf-8")
    assert "safe telemetry stays" in sanitized
    assert sanitized.count("<redacted>") == 3
    assert hf_value not in sanitized
    assert assignment_value not in sanitized
    assert bearer_value not in sanitized
    assert (solve_output.stat().st_mode & 0o777) == 0o640

    attestation_path = root / ATTESTATION
    attestation_text = attestation_path.read_text(encoding="utf-8")
    attestation = json.loads(attestation_text)
    assert attestation["schema_version"] == "awm-secret-sanitization-v1"
    assert attestation["status"] == "redacted"
    assert attestation["files_redacted"] == 1
    assert attestation["redaction_count"] == 3
    assert attestation["redacted_paths"] == ["task/solve_out.txt"]
    assert "rules" not in attestation
    assert (attestation_path.stat().st_mode & 0o777) == 0o600

    retained_output = first.stdout + first.stderr + attestation_text
    for value in (hf_value, assignment_value, bearer_value):
        assert value not in retained_output

    second = run_sanitizer(root)
    assert second.returncode == 0
    second_attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert second_attestation["status"] == "clean"
    assert second_attestation["files_redacted"] == 0


def test_sanitizer_skips_model_binaries_and_does_not_follow_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    fake_value = "hf_" + "D" * 32

    model = root / "final_model" / "model.safetensors"
    model.parent.mkdir()
    model_bytes = ("binary model payload " + fake_value).encode()
    model.write_bytes(model_bytes)

    opaque = root / "opaque.dat"
    opaque_bytes = b"\x00binary payload " + fake_value.encode()
    opaque.write_bytes(opaque_bytes)

    outside_file = tmp_path / "outside.txt"
    outside_text = "external " + fake_value
    outside_file.write_text(outside_text, encoding="utf-8")
    (root / "outside-link.txt").symlink_to(outside_file)

    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "nested.txt").write_text(outside_text, encoding="utf-8")
    (root / "outside-directory-link").symlink_to(outside_directory, target_is_directory=True)
    (root / "notes.txt").write_text("ordinary retained output\n", encoding="utf-8")

    result = run_sanitizer(root)

    assert result.returncode == 0
    assert model.read_bytes() == model_bytes
    assert opaque.read_bytes() == opaque_bytes
    assert outside_file.read_text(encoding="utf-8") == outside_text
    assert (outside_directory / "nested.txt").read_text(encoding="utf-8") == outside_text

    attestation_text = (root / ATTESTATION).read_text(encoding="utf-8")
    attestation = json.loads(attestation_text)
    assert attestation["skipped"]["known_binary"] == {
        "count": 1,
        "paths": ["final_model/model.safetensors"],
    }
    assert attestation["skipped"]["binary"] == {
        "count": 1,
        "paths": ["opaque.dat"],
    }
    assert attestation["skipped"]["symlink"] == {
        "count": 2,
        "paths": ["outside-directory-link", "outside-link.txt"],
    }
    assert fake_value not in result.stdout + result.stderr + attestation_text


def test_sanitizer_rejects_symlink_attestation_without_mutating_targets(tmp_path: Path) -> None:
    root = tmp_path / "result"
    root.mkdir()
    fake_value = "sk-" + "E" * 32
    solve_output = root / "solve_out.txt"
    original_output = "captured " + fake_value
    solve_output.write_text(original_output, encoding="utf-8")

    outside_attestation = tmp_path / "outside-attestation.json"
    outside_attestation.write_text("do not replace\n", encoding="utf-8")
    (root / ATTESTATION).symlink_to(outside_attestation)

    result = run_sanitizer(root)

    assert result.returncode == 2
    assert json.loads(result.stderr)["status"] == "error"
    assert fake_value not in result.stdout + result.stderr
    assert solve_output.read_text(encoding="utf-8") == original_output
    assert outside_attestation.read_text(encoding="utf-8") == "do not replace\n"


def test_sanitizer_rejects_symlink_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-result"
    real_root.mkdir()
    root_link = tmp_path / "result-link"
    root_link.symlink_to(real_root, target_is_directory=True)

    result = run_sanitizer(root_link)

    assert result.returncode == 2
    assert json.loads(result.stderr)["status"] == "error"
    assert not (real_root / ATTESTATION).exists()
