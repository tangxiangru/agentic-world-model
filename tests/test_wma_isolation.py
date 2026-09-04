"""Real OS canaries for the library/cache gap in transcript path scanning."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from exp_protocol_cards import closed_card, plan_card

from awm.exp_protocol import schema as cards
from awm.wma import backends, isolation, schema


def make_brief(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    card = session / "exp-01.yaml"
    card.write_text("card_id: exp-01\n")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("Review the proposal.")
    return SimpleNamespace(card_path=card, skill_dir=skill, session_dir=session,
                           verdict_path=session / "exp-01.verdict.json", extra={})


def python_command(script):
    return f"{shlex.quote(str(Path(sys.executable).resolve()))} -I -c {shlex.quote(script)}"


@pytest.fixture
def sandbox(tmp_path):
    brief = make_brief(tmp_path)
    try:
        with isolation.isolated_tools(brief, ["claude", "--add-dir", "/sensitive"], "claude") as configured:
            yield brief, configured
    except isolation.IsolationError as exc:
        if "Landlock" in str(exc) or "seccomp" in str(exc):
            pytest.skip(f"kernel sandbox unavailable (production fails closed): {exc}")
        raise


def test_file_and_library_reads_are_confined_and_children_inherit(sandbox, tmp_path):
    brief, configured = sandbox
    secret = tmp_path / "hf-cache" / "test-answers.json"
    secret.parent.mkdir()
    secret.write_text("SYNTHETIC_HELD_OUT_CANARY")
    exported = configured.policy["files"][0]["path"]
    # Import a helper module, emulating a library which constructs the dataset
    # cache path internally: the command need not mention a literal path.
    scratch = Path(configured.policy["scratch"])
    (scratch / "loader.py").write_text(
        "from pathlib import Path\ndef load():\n"
        f"    return Path({str(secret)!r}).read_text()\n"
    )
    code = f"""
import os, pathlib, subprocess, sys
assert 'card_id' in pathlib.Path({exported!r}).read_text()
sys.path.insert(0, os.getcwd())
import loader
try:
    loader.load()
except PermissionError:
    print('library-read-denied')
else:
    raise AssertionError('cache leaked')
p = subprocess.run([sys.executable, '-I', '-c', {('import pathlib; pathlib.Path(' + repr(str(secret)) + ').read_text()')!r}], capture_output=True, text=True)
assert p.returncode != 0 and 'PermissionError' in p.stderr
try:
    pathlib.Path({str(brief.card_path)!r}).write_text('mutated')
except PermissionError:
    print('source-write-denied')
else:
    raise AssertionError('source modified')
print('child-inherited')
"""
    result = isolation.run_probe(configured.policy, python_command(code))
    assert result["returncode"] == 0, result
    assert result["stdout"] == "library-read-denied\nsource-write-denied\nchild-inherited\n"
    assert brief.card_path.read_text() == "card_id: exp-01\n"


def test_network_and_proc_credential_reads_are_denied(sandbox):
    _, configured = sandbox
    code = """
import socket, pathlib
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try:
        socket.socket(family)
    except PermissionError:
        continue
    raise AssertionError('network socket allowed')
try:
    pathlib.Path('/proc/1/environ').read_bytes()
except PermissionError:
    pass
else:
    raise AssertionError('host process readable')
print('all-denied')
"""
    result = isolation.run_probe(configured.policy, python_command(code))
    assert result["returncode"] == 0, result
    assert result["stdout"] == "all-denied\n"


def test_environment_cannot_restore_cache_or_credentials(sandbox, monkeypatch):
    _, configured = sandbox
    monkeypatch.setenv("ANTHROPIC_API_KEY", "SYNTHETIC_API_CANARY")
    monkeypatch.setenv("HF_HOME", "/sensitive/cache")
    code = "import os; assert 'ANTHROPIC_API_KEY' not in os.environ; assert os.environ['HF_HOME'].endswith('/scratch/hf-cache'); print('clean')"
    result = isolation.run_probe(configured.policy, python_command(code))
    assert result["returncode"] == 0, result
    assert result["stdout"] == "clean\n"


def test_children_cannot_escape_process_group_and_resource_caps(sandbox):
    _, configured = sandbox
    import resource
    inherited_nproc = resource.getrlimit(resource.RLIMIT_NPROC)
    child_code = """
import ctypes, errno, os, resource
for operation in (os.setsid, lambda: os.setpgid(0, 0)):
    try:
        operation()
    except PermissionError:
        continue
    raise AssertionError('escaped broker process group')
libc = ctypes.CDLL(None, use_errno=True)
assert libc.syscall(435, 0, 0) == -1 and ctypes.get_errno() == errno.ENOSYS
assert 0 < resource.getrlimit(resource.RLIMIT_AS)[1] <= 512 * 1024 * 1024
assert resource.getrlimit(resource.RLIMIT_NPROC) == INHERITED_NPROC
print('ordinary-child-constrained')
""".replace("INHERITED_NPROC", repr(inherited_nproc))
    parent_code = f"import subprocess, sys; subprocess.run([sys.executable, '-I', '-c', {child_code!r}], check=True)"
    result = isolation.run_probe(configured.policy, python_command(parent_code))
    assert result["returncode"] == 0, result
    assert result["stdout"] == "ordinary-child-constrained\n"


def test_probe_group_budget_stops_children_without_counting_other_uid_jobs(sandbox):
    _, configured = sandbox
    configured.policy["limits"]["processes_per_probe"] = 4
    code = "import subprocess,sys,time; children=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']) for _ in range(8)]; time.sleep(10)"
    with pytest.raises(isolation.IsolationError, match="supervised task limit"):
        isolation.run_probe(configured.policy, python_command(code), timeout=3)


def test_scratch_symlink_cannot_escape(sandbox, tmp_path):
    _, configured = sandbox
    secret = tmp_path / "answers"
    secret.write_text("SYNTHETIC_CANARY")
    (Path(configured.policy["scratch"]) / "alias").symlink_to(secret)
    result = isolation.run_probe(configured.policy, "cat alias")
    assert result["returncode"] != 0
    assert "SYNTHETIC_CANARY" not in result["stdout"]


def test_broker_exports_only_frozen_inputs_and_publishes_result(tmp_path):
    brief = make_brief(tmp_path)
    (brief.session_dir / "test.json").write_text("PRIVATE_DATA")
    with isolation.isolated_tools(brief, ["claude", "--add-dir", "/history"], "claude") as configured:
        assert "--add-dir" not in configured.argv
        assert configured.argv[configured.argv.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in configured.argv
        assert "--bare" in configured.argv
        brief.card_path.write_text("changed after snapshot")
        assert isolation.broker_call(configured.policy, "read_file", {"path": str(brief.card_path)}) == "card_id: exp-01\n"
        with pytest.raises(isolation.IsolationError, match="not an exported"):
            isolation.broker_call(configured.policy, "read_file", {"path": str(brief.session_dir / "test.json")})
        isolation.broker_call(configured.policy, "write_result", {"result": {"decision": "uncertain"}})
    assert json.loads(brief.verdict_path.read_text()) == {"decision": "uncertain"}


def test_mcp_transport_preserves_tool_errors_and_success(sandbox):
    _, configured = sandbox
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "read_file", "arguments": {"path": "/etc/shadow"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "run", "arguments": {"command": "printf probe-ok"}}},
    ]
    proc = subprocess.run([sys.executable, "-I", isolation.__file__, "--serve", configured.policy["policy_file"]],
                          input="\n".join(json.dumps(x) for x in messages) + "\n", text=True,
                          capture_output=True, check=True)
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4]
    assert len(replies[1]["result"]["tools"]) == 4
    assert replies[2]["result"]["isError"] is True
    assert json.loads(replies[3]["result"]["content"][0]["text"])["stdout"] == "probe-ok"


def test_unsupported_backend_and_symlink_exports_fail_closed(tmp_path):
    brief = make_brief(tmp_path)
    with pytest.raises(isolation.IsolationError, match="unavailable for backend"), isolation.isolated_tools(brief, ["codex"], "codex"):
        pytest.fail("unisolated fallback")
    secret = tmp_path / "secret"
    secret.write_text("CANARY")
    alias = brief.session_dir / "alias"
    alias.symlink_to(secret)
    brief.extra["probe_files"] = [alias]
    with pytest.raises(isolation.IsolationError, match="symlinks"), isolation.isolated_tools(brief, ["claude"], "claude"):
        pytest.fail("followed symlink")


def test_kernel_failure_never_retries_without_isolation(tmp_path, monkeypatch):
    brief = make_brief(tmp_path)
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(args)
        raise isolation.IsolationError("Landlock disabled")
    monkeypatch.setattr(isolation, "run_probe", unavailable)
    with pytest.raises(isolation.IsolationError, match="Landlock disabled"), isolation.isolated_tools(brief, ["claude"], "claude"):
        pytest.fail("unisolated fallback")
    assert len(calls) == 1


def test_review_exports_locked_code_closed_past_and_curated_history_only(tmp_path):
    brief = make_brief(tmp_path)
    brief.card_id = "exp-03"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    brief.card_path = frozen / "exp-03.yaml"
    card = plan_card()
    card["card_id"] = "exp-03"
    cards.dump_card(brief.card_path, card)
    script = brief.session_dir / "train.py"
    script.write_text("print('train plan')")
    (frozen / "exp-03.lock.json").write_text(json.dumps({
        "plan_sha256": cards.plan_hash(card),
        "script": {"path": "/home/ben/task/train.py", "sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
    }))
    (frozen / "exp-03.preflight.json").write_text('{"ok": true}')
    for number, closed in ((1, True), (2, False), (3, True), (4, True)):
        previous = closed_card() if closed else plan_card()
        previous["card_id"] = f"exp-{number:02d}"
        cards.dump_card(brief.session_dir / "memory" / "cards" / f"exp-{number:02d}.yaml", previous)
    (brief.session_dir / "memory" / "index.md").write_text("future result: SECRET")
    history = tmp_path / "history"
    history.mkdir()
    run = tmp_path / "curated-train-run"
    run.mkdir()
    cards.dump_card(run / "exp-01.yaml", closed_card())
    (run / "index.md").write_text("prior run summary")
    (run / "raw-test.json").write_text("TEST_SECRET")
    (run / "exp-02.yaml").symlink_to(brief.session_dir / "memory" / "cards" / "exp-04.yaml")
    (history / "r-train").symlink_to(run, target_is_directory=True)
    brief.history_dir = history
    selected = isolation.collect_review_inputs(brief)
    assert brief.card_path in selected
    assert script in selected
    assert frozen / "exp-03.preflight.json" in selected
    assert run / "exp-01.yaml" in selected and run / "index.md" in selected
    assert brief.session_dir / "memory" / "cards" / "exp-01.yaml" in selected
    assert brief.session_dir / "memory" / "cards" / "exp-02.yaml" not in selected
    assert brief.session_dir / "memory" / "cards" / "exp-03.yaml" not in selected
    assert brief.session_dir / "memory" / "cards" / "exp-04.yaml" not in selected
    assert brief.session_dir / "memory" / "index.md" not in selected
    assert run / "exp-02.yaml" not in selected and run / "raw-test.json" not in selected
    with isolation.isolated_tools(brief, ["claude"], "claude") as configured:
        assert isolation.broker_call(configured.policy, "read_file", {"path": str(brief.card_path)}) == brief.card_path.read_text()
        assert len(configured.policy["files"]) == len(selected) + 1  # skill
    script.write_text("changed after lock")
    with pytest.raises(isolation.IsolationError, match="frozen hash"), isolation.isolated_tools(brief, ["claude"], "claude"):
        pytest.fail("reviewed a different script")


def test_locked_script_cannot_export_file_outside_session(tmp_path):
    brief = make_brief(tmp_path)
    brief.card_id = "exp-01"
    card = plan_card()
    cards.dump_card(brief.card_path, card)
    outside = tmp_path / "private.py"
    outside.write_text("SYNTHETIC_TEST_ANSWERS")
    (brief.card_path.with_suffix(".lock.json")).write_text(json.dumps({
        "plan_sha256": cards.plan_hash(card),
        "script": {"path": str(outside), "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()},
    }))
    assert outside not in isolation.collect_review_inputs(brief)


def test_command_backend_uses_broker_and_stamps_measured_boundary(tmp_path):
    inputs = make_brief(tmp_path)
    brief = backends.Brief(
        card_id="exp-01", session_dir=inputs.session_dir, card_path=inputs.card_path,
        verdict_path=inputs.verdict_path, skill_dir=inputs.skill_dir,
        mode="online", budget=backends.Budget(wall_min=1), model=None, prompt="Review this card.",
    )
    backends.HeuristicBackend().run(brief)
    result = schema.load_verdict(brief.verdict_path)
    result["isolation"] = {"method": "model-forged"}
    result["cost"] = {"usd": 999}
    # This fake replaces only the model API process; the backend boundary,
    # kernel self-check, MCP subprocess and final result publication are real.
    fake_cli = tmp_path / "fake_claude.py"
    fake_cli.write_text(f"""
import json, subprocess, sys
assert '--bare' in sys.argv and '--strict-mcp-config' in sys.argv
assert sys.argv[sys.argv.index('--tools') + 1] == ''
assert 'Enforced WMA tool boundary' in sys.stdin.read()
config = json.load(open(sys.argv[sys.argv.index('--mcp-config') + 1]))['mcpServers']['wma_probe']
request = {{'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {{'name': 'write_result', 'arguments': {{'result': {result!r}}}}}}}
response = subprocess.run([config['command'], *config['args']], input=json.dumps(request)+'\\n', capture_output=True, text=True, check=True)
assert json.loads(response.stdout)['result']['content']
print(json.dumps({{'type': 'result', 'total_cost_usd': 0.25, 'num_turns': 1}}))
""")
    backend = backends.CommandBackend("claude", [sys.executable, str(fake_cli)],
                                      isolated=True, transcript="stream-json")
    backend.run(brief)
    final = schema.load_verdict(brief.verdict_path)
    assert final["isolation"]["method"] == "landlock-seccomp-mcp-v1"
    assert final["isolation"]["network"] == "denied-for-probes"
    assert final["isolation"]["inputs"][0]["source"] == str(brief.card_path)
    assert final["cost"]["usd"] == 0.25
