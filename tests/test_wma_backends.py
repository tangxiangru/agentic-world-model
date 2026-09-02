"""The heuristic backend is the baseline and the test double; a command backend is any agent that writes the file."""

from __future__ import annotations

import json
import os
import stat

import pytest

from awm.exp_protocol import schema as cards
from awm.wma import backends, schema
from exp_protocol_cards import plan_card


def brief(tmp_path, card=None) -> backends.Brief:
    """Shaped like a replay: the session is a subdirectory, so siblings such as _truth/ are outside it."""
    card = card or plan_card()
    session = tmp_path / "session"
    card_path = session / "memory" / "cards" / f"{card['card_id']}.yaml"
    cards.dump_card(card_path, card)
    return backends.Brief(card_id=card["card_id"], session_dir=session, card_path=card_path,
                          verdict_path=schema.verdict_path(card_path), skill_dir=tmp_path / "skills" / "wma",
                          mode="offline", budget=backends.Budget(cpu_min=1, gpu_min=0, wall_min=1),
                          model=None, prompt="write the verdict")


def test_heuristic_writes_a_valid_verdict_for_a_training_card(tmp_path) -> None:
    b = brief(tmp_path)
    backends.HeuristicBackend().run(b)
    v = schema.load_verdict(b.verdict_path)
    assert schema.validate_verdict(v).ok, schema.validate_verdict(v).render()
    assert v["backend"] == "heuristic" and v["mode"] == "offline" and v["wma_skill"] == "heuristic-priors"
    assert v["levels"]["L2_effect"]["interval"][0] <= v["levels"]["L2_effect"]["interval"][1]
    assert v["levels"]["L0_runs"]["basis"] and v["evidence"]


def test_heuristic_defers_late_in_the_run_and_widens_for_non_training_families(tmp_path) -> None:
    card = plan_card()
    card["situation"]["elapsed_h"] = 9.0
    card["setup"]["method"] = {"family": "merge"}
    b = brief(tmp_path, card)
    backends.HeuristicBackend().run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["levels"]["L3_worth_now"]["answer"] == "defer"
    lo, hi = v["levels"]["L2_effect"]["interval"]
    assert hi - lo > 0.05


def fake_executable(tmp_path, body: str):
    exe = tmp_path / "fake-agent"
    exe.write_text("#!/bin/bash\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_command_backend_accepts_an_agent_that_writes_the_file(tmp_path) -> None:
    b = brief(tmp_path)
    good = json.dumps({**schema.empty_verdict("exp-01"), "levels": {
        "L0_runs": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L1_valid": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L2_effect": {"metric": "accuracy", "direction": "higher", "interval": [0.0, 0.05], "confidence": 0.5, "basis": []},
        "L3_worth_now": {"answer": "yes", "confidence": 0.6, "expected_cost_h": 1.0, "basis": []}}})
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\ncat > '{b.verdict_path}' <<'J'\n{good}\nJ\n")
    be = backends.CommandBackend("fake", [str(exe), "--model", "{model}"], model="m1")
    be.run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["backend"] == "fake" and v["levels"]["L0_runs"]["answer"] == "yes"


def test_command_backend_rejects_an_agent_that_writes_nothing(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, "cat > /dev/null\necho did nothing\n")
    with pytest.raises(backends.BackendError, match="no verdict"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_command_backend_rejects_an_invalid_verdict(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\necho '{{\"schema_version\": \"x\"}}' > '{b.verdict_path}'\n")
    with pytest.raises(backends.BackendError, match="invalid"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_command_backend_times_out_on_the_wall_budget(tmp_path) -> None:
    b = brief(tmp_path)
    b.budget = backends.Budget(cpu_min=1, gpu_min=0, wall_min=0.01)
    exe = fake_executable(tmp_path, "sleep 5\n")
    with pytest.raises(backends.BackendError, match="timed out"):
        backends.CommandBackend("fake", [str(exe)]).run(b)


def test_registry_names_the_three_backends_and_passes_the_model(tmp_path) -> None:
    assert set(backends.BACKENDS) == {"heuristic", "claude", "codex"}
    be = backends.get_backend("claude", model="claude-opus-5")
    assert isinstance(be, backends.CommandBackend) and "claude-opus-5" in be.argv()
    with pytest.raises(backends.BackendError):
        backends.get_backend("nope")
    assert os.path.basename(be.argv()[0]) == "claude"


def test_an_empty_model_leaves_no_dangling_model_flag() -> None:
    for model in (None, ""):
        argv = backends.get_backend("claude", model=model).argv()
        assert "--model" not in argv and "{model}" not in " ".join(argv), argv


# ---- real-CLI plumbing: measured cost, leak detection, history access (2026-09-02) ----

def stream_events(*events) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


def tool_use(name, **inp):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}}


def result_event(cost=0.42, turns=7):
    return {"type": "result", "subtype": "success", "total_cost_usd": cost, "num_turns": turns, "duration_ms": 61000}


def good_verdict_json() -> str:
    return json.dumps({**schema.empty_verdict("exp-01"), "levels": {
        "L0_runs": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L1_valid": {"answer": "yes", "confidence": 0.9, "basis": []},
        "L2_effect": {"metric": "accuracy", "direction": "higher", "interval": [0.0, 0.05], "confidence": 0.5, "basis": []},
        "L3_worth_now": {"answer": "yes", "confidence": 0.6, "expected_cost_h": 1.0, "basis": []}}})


def streaming_fake(tmp_path, b, events: str):
    """An agent that writes the verdict, then prints a stream-json transcript."""
    return fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\n"
                                     f"cat > '{b.verdict_path}' <<'J'\n{good_verdict_json()}\nJ\n"
                                     f"cat <<'S'\n{events}S\n")


def test_measured_cost_and_turns_are_stamped_from_the_stream(tmp_path) -> None:
    b = brief(tmp_path)
    exe = streaming_fake(tmp_path, b, stream_events(tool_use("Read", file_path=str(b.card_path)), result_event(0.42, 7)))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["cost"]["usd"] == 0.42 and v["cost"]["turns"] == 7 and v["cost"]["wall_min"] > 0
    assert v["access"] == {"files": 1, "outside": []} and "leak_suspected" not in v


def test_a_read_outside_the_session_and_history_is_flagged_as_a_suspected_leak(tmp_path) -> None:
    b = brief(tmp_path)
    hist = tmp_path / "hist"
    hist.mkdir()
    b.history_dir = hist
    truth = tmp_path / "_truth" / "r-x" / "exp-01.yaml"
    truth.parent.mkdir(parents=True)
    truth.write_text("card_id: exp-01\n")
    exe = streaming_fake(tmp_path, b, stream_events(
        tool_use("Read", file_path=str(b.card_path)),
        tool_use("Read", file_path=str(hist / "r-y" / "exp-02.yaml")),
        tool_use("Read", file_path=str(truth)),
        tool_use("Bash", command=f"cat {truth}"),
        result_event()))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["leak_suspected"] is True
    assert v["access"]["files"] == 3 and str(truth) in v["access"]["outside"]
    assert any("cat " in o for o in v["access"]["outside"])   # the Bash read is caught too


def test_the_skill_dir_and_the_session_are_inside_the_fence(tmp_path) -> None:
    b = brief(tmp_path)
    b.skill_dir.mkdir(parents=True)
    exe = streaming_fake(tmp_path, b, stream_events(
        tool_use("Read", file_path=str(b.skill_dir / "SKILL.md")),
        tool_use("Read", file_path="skills/wma/verdict.example.json"),   # relative to cwd
        tool_use("Glob", pattern="memory/cards/*.yaml"),
        result_event()))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert "leak_suspected" not in v and v["access"]["outside"] == []


def test_claude_argv_carries_history_add_dir_and_max_turns(tmp_path) -> None:
    b = brief(tmp_path)
    b.history_dir = tmp_path / "hist"
    b.budget = backends.Budget(cpu_min=1, gpu_min=0, wall_min=5, max_turns=30)
    be = backends.get_backend("claude", model="claude-opus-5")
    argv = be.argv(b)
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert argv[argv.index("--add-dir") + 1] == str((tmp_path / "hist").resolve())
    assert argv[argv.index("--max-turns") + 1] == "30"
    assert "--model" in argv and "claude-opus-5" in argv


def test_a_plain_backend_without_a_transcript_still_works(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\n"
                                    f"cat > '{b.verdict_path}' <<'J'\n{good_verdict_json()}\nJ\necho done\n")
    backends.CommandBackend("fake", [str(exe)]).run(b)
    v = schema.load_verdict(b.verdict_path)
    assert "usd" not in v["cost"] and "access" not in v


def test_add_dir_also_covers_where_the_history_links_point(tmp_path) -> None:
    """history/<run> are symlinks into the corpus; Claude Code resolves them, so the corpus side must be allowed too."""
    corpus_train = tmp_path / "corpus" / "train"
    (corpus_train / "r-a").mkdir(parents=True)
    (corpus_train / "r-b").mkdir()
    hist = tmp_path / "hist"
    hist.mkdir()
    os.symlink(corpus_train / "r-a", hist / "r-a")
    os.symlink(corpus_train / "r-b", hist / "r-b")
    b = brief(tmp_path)
    b.history_dir = hist
    argv = backends.get_backend("claude", model="m").argv(b)
    dirs = [argv[i + 1] for i, a in enumerate(argv) if a == "--add-dir"]
    assert dirs == [str(hist.resolve()), str(corpus_train.resolve())]


# ---- effort and model are part of the measurement: passed explicitly, stamped on the verdict (2026-09-02) ----

def test_effort_is_passed_explicitly_to_each_cli_and_omitted_when_unset() -> None:
    claude = backends.get_backend("claude", model="claude-opus-5", effort="high").argv()
    assert claude[claude.index("--effort") + 1] == "high"
    codex = backends.get_backend("codex", model="gpt-5", effort="high").argv()
    assert "model_reasoning_effort=high" in codex
    for be in (backends.get_backend("claude", model="claude-opus-5"), backends.get_backend("codex", model="gpt-5")):
        assert "--effort" not in be.argv() and not any("model_reasoning_effort" in a for a in be.argv())


def test_the_backend_stamps_the_model_and_effort_it_ran_with(tmp_path) -> None:
    b = brief(tmp_path)
    exe = streaming_fake(tmp_path, b, stream_events(result_event(0.1, 2)))
    backends.CommandBackend("fake", [str(exe)], "m-1", effort="high", transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["model"] == "m-1" and v["effort"] == "high"


def test_a_backend_without_model_or_effort_stamps_neither(tmp_path) -> None:
    b = brief(tmp_path)
    exe = streaming_fake(tmp_path, b, stream_events(result_event(0.1, 2)))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert "model" not in v and "effort" not in v


# ---- an invalid verdict is moved aside with what it cost; the harness's fields are never the agent's (2026-09-02) ----

def test_an_invalid_verdict_is_moved_aside_with_its_measured_cost_so_the_sample_can_be_retried(tmp_path) -> None:
    b = brief(tmp_path)
    bad = json.loads(good_verdict_json())
    bad["levels"]["L2_effect"]["direction"] = "sideways"
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\n"
                                    f"cat > '{b.verdict_path}' <<'J'\n{json.dumps(bad)}\nJ\n"
                                    f"cat <<'S'\n{stream_events(result_event(0.31, 9))}S\n")
    with pytest.raises(backends.BackendError, match="direction"):
        backends.CommandBackend("fake", [str(exe)], "m-1", effort="high", transcript="stream-json").run(b)
    assert not b.verdict_path.exists(), "an invalid verdict must not block a retry"
    rejected = b.verdict_path.with_name(b.verdict_path.name + ".rejected")
    r = json.loads(rejected.read_text())
    assert r["rejected"]["cost"]["usd"] == 0.31 and "direction" in r["rejected"]["reason"]
    assert r["verdict"]["levels"]["L2_effect"]["direction"] == "sideways"
    assert r["rejected"]["model"] == "m-1" and r["rejected"]["effort"] == "high"


def test_unparseable_output_is_moved_aside_too(tmp_path) -> None:
    b = brief(tmp_path)
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\n"
                                    f"echo 'not json' > '{b.verdict_path}'\n")
    with pytest.raises(backends.BackendError, match="invalid verdict JSON"):
        backends.CommandBackend("fake", [str(exe)]).run(b)
    assert not b.verdict_path.exists()
    r = json.loads(b.verdict_path.with_name(b.verdict_path.name + ".rejected").read_text())
    assert r["raw"].startswith("not json")


def test_only_paths_that_exist_count_as_reads_outside_the_fence(tmp_path) -> None:
    """The first real verdict was flagged on sed/awk regex literals ('/^result:/,/^conclusion:/p', '/1.7B/')
    and on a mistyped path: none of those can leak anything. A glob that reaches the truth can."""
    b = brief(tmp_path)
    hist = b.session_dir / "history"
    (hist / "r-y").mkdir(parents=True)
    (hist / "r-y" / "exp-01.yaml").write_text("card_id: exp-01\n")
    truth = tmp_path / "_truth" / "r-x" / "exp-01.yaml"
    truth.parent.mkdir(parents=True)
    truth.write_text("card_id: exp-01\n")
    inside = [
        f"H={hist}; for f in $H/*/exp-01.yaml; do grep -oE 'Qwen[^ ]*' $f; done | awk -F'|' '$2 ~ /1.7B/'",
        f"sed -n '/^result:/,/^conclusion:/p' {hist}/r-y/exp-01.yaml | head -40",
        f"ls {tmp_path}/no/such/dir/history",                       # a typo, nothing there to read
        "grep -c 016546b4 /proc/self/status >/dev/null; echo ok",   # exists, but not a session file either
    ]
    exe = streaming_fake(tmp_path, b, stream_events(
        *[tool_use("Bash", command=c) for c in inside],
        tool_use("Grep", pattern="/^result:/", path=str(hist)),
        result_event()))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["access"]["outside"] == [f"bash: {inside[3]}"], v["access"]["outside"]   # /proc is real and outside; the rest is noise
    b2 = brief(tmp_path / "two")
    exe = streaming_fake(tmp_path / "two", b2, stream_events(
        tool_use("Bash", command=f"cat {tmp_path}/_truth/*/exp-01.yaml"), result_event()))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b2)
    v2 = schema.load_verdict(b2.verdict_path)
    assert v2["leak_suspected"] is True and "_truth/*" in v2["access"]["outside"][0]


def test_the_transcript_is_kept_beside_the_verdict_for_hand_reading_and_rescans(tmp_path) -> None:
    b = brief(tmp_path)
    events = stream_events(tool_use("Read", file_path=str(b.card_path)), result_event(0.2, 3))
    exe = streaming_fake(tmp_path, b, events)
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    t = backends.transcript_path(b.verdict_path)
    assert t.name == "exp-01.transcript.jsonl" and t.read_text() == events
    tagged = schema.verdict_path(b.card_path, tag="agent-b")
    assert backends.transcript_path(tagged).name == "exp-01.transcript.agent-b.jsonl"
    # a rejected verdict keeps its transcript too — that is where the money went
    bad = json.loads(good_verdict_json())
    bad["levels"] = {}
    b2 = brief(tmp_path / "two")
    exe = fake_executable(tmp_path / "two", f"cat > /dev/null\nmkdir -p $(dirname '{b2.verdict_path}')\n"
                                            f"cat > '{b2.verdict_path}' <<'J'\n{json.dumps(bad)}\nJ\n"
                                            f"cat <<'S'\n{events}S\n")
    with pytest.raises(backends.BackendError):
        backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b2)
    assert backends.transcript_path(b2.verdict_path).read_text() == events


def test_bare_slashes_and_the_cli_spill_dir_are_not_reads_outside_the_fence(tmp_path, monkeypatch) -> None:
    """From the 20-verdict pass: sed 's/.*path: //' yields the token '//' (which exists: it is /), and the
    agent re-reads its own oversized tool output from ~/.claude/projects/<cwd-mangled>/…/tool-results/."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    b = brief(tmp_path)
    spill = backends.cli_project_dir(b.session_dir) / "9d52-uuid" / "tool-results" / "b6eylsbwb.txt"
    spill.parent.mkdir(parents=True)
    spill.write_text("== r-a ==\n")
    other = tmp_path / "home" / ".claude" / "projects" / "-some-other-cwd" / "x" / "tool-results" / "y.txt"
    other.parent.mkdir(parents=True)
    other.write_text("not ours\n")
    exe = streaming_fake(tmp_path, b, stream_events(
        tool_use("Bash", command="grep path memory/index.md | sed 's/.*path: //' | sort -u"),
        tool_use("Bash", command="awk -F'|' '{print $2}' memory/index.md | sed 's////////'"),
        tool_use("Bash", command=f"grep -c '^==' {spill}"),
        tool_use("Bash", command=f"cat {other}"),
        result_event()))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    assert v["access"]["outside"] == [f"bash: cat {other}"], v["access"]["outside"]


def test_direction_synonyms_are_normalised_before_validation(tmp_path) -> None:
    b = brief(tmp_path)
    v = json.loads(good_verdict_json())
    v["levels"]["L2_effect"]["direction"] = "none"
    exe = fake_executable(tmp_path, f"cat > /dev/null\nmkdir -p $(dirname '{b.verdict_path}')\n"
                                    f"cat > '{b.verdict_path}' <<'J'\n{json.dumps(v)}\nJ\n")
    backends.CommandBackend("fake", [str(exe)]).run(b)
    assert schema.load_verdict(b.verdict_path)["levels"]["L2_effect"]["direction"] == "flat"


def test_rescan_rederives_access_from_the_kept_transcript_and_touches_nothing_else(tmp_path) -> None:
    b = brief(tmp_path)
    exe = streaming_fake(tmp_path, b, stream_events(tool_use("Read", file_path=str(b.card_path)), result_event(0.5, 4)))
    backends.CommandBackend("fake", [str(exe)], transcript="stream-json").run(b)
    v = schema.load_verdict(b.verdict_path)
    v["access"] = {"files": 0, "outside": ["bash: sed 's/x//'"]}     # what an older fence wrote
    v["leak_suspected"] = True
    schema.dump_verdict(b.verdict_path, v)
    before_levels = v["levels"]
    changed = backends.rescan([tmp_path])
    after = schema.load_verdict(b.verdict_path)
    assert changed == {"scanned": 1, "changed": 1}
    assert after["access"] == {"files": 1, "outside": []} and "leak_suspected" not in after
    assert after["levels"] == before_levels and after["cost"]["usd"] == 0.5
