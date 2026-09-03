"""The scientist sees only a queue client; the private sidecar owns WMA policy."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from exp_protocol_cards import plan_card

from awm import wma_client
from awm.wma import backends, sidecar


def _session(tmp_path: Path, attached: bool = True) -> Path:
    session = tmp_path / "session"
    cards = session / "memory/cards"
    cards.mkdir(parents=True)
    card = plan_card()
    (cards / "exp-01.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    (cards / "exp-01.lock.json").write_text("{}\n", encoding="utf-8")
    if attached:   # what the sidecar does first thing when run_task.sh starts it, before the scientist
        (session / ".wma" / "requests").mkdir(parents=True)
    return session


def _client_main(argv: list[str]) -> int:
    """The scientist's `awm wma …` — the thin client, which this machine's full CLI would shadow."""
    import argparse

    parser = argparse.ArgumentParser()
    wma_client.register(parser.add_subparsers(dest="top", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


def test_client_enqueues_a_batch_without_wma_policy(tmp_path: Path) -> None:
    session = _session(tmp_path)

    request_id, path = wma_client.enqueue(session, ["exp-01"])

    request = json.loads(path.read_text(encoding="utf-8"))
    assert request == {
        "schema_version": "awm-wma-review-request-v1",
        "request_id": request_id,
        "created_at": request["created_at"],
        "card_ids": ["exp-01"],
    }
    assert not (session / "skills/wma").exists()


def test_private_sidecar_writes_the_verdict_without_exposing_skill(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    skill = tmp_path / "private/skills/wma"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("private WMA policy\n", encoding="utf-8")
    wma_client.enqueue(session, ["exp-01"])
    config = sidecar.Config(
        session_dir=session,
        skill_dir=skill,
        history_dir=None,
        backend="heuristic",
        model="claude-opus-5",
        effort="high",
        budget=backends.Budget(),
        jobs=2,
    )

    assert sidecar.run(config, once=True) == 0

    assert (session / "memory/cards/exp-01.verdict.json").is_file()
    response_path = next((session / ".wma/responses").glob("*.json"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["state"] == "completed"
    assert response["ranking"] == ["exp-01"]
    assert not (session / "skills/wma").exists()
    assert not list((session / "memory/cards").glob("*.transcript*.jsonl"))


def test_private_sidecar_refuses_an_unlocked_card(tmp_path: Path) -> None:
    session = _session(tmp_path)
    (session / "memory/cards/exp-01.lock.json").unlink()
    skill = tmp_path / "private/skills/wma"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("private WMA policy\n", encoding="utf-8")
    wma_client.enqueue(session, ["exp-01"])
    config = sidecar.Config(
        session_dir=session,
        skill_dir=skill,
        history_dir=None,
        backend="heuristic",
        model="claude-opus-5",
        effort="high",
        budget=backends.Budget(),
        jobs=1,
    )

    sidecar.run(config, once=True)

    response = json.loads(next((session / ".wma/responses").glob("*.json")).read_text())
    assert response["state"] == "failed"
    assert "locked card required" in response["errors"]["exp-01"]
    assert not (session / "memory/cards/exp-01.verdict.json").exists()


# ---- the control arm: same checkout, same protocol, no sidecar — the client says so (2026-09-02) ----

def test_without_a_sidecar_the_client_says_none_is_attached_and_queues_nothing(tmp_path: Path, capsys) -> None:
    """A control cell ships the same client and the same protocol step 4b; what differs is that no sidecar
    was started, so `.wma/requests` never appears. The client must tell the scientist that, not queue."""
    session = _session(tmp_path, attached=False)
    try:
        wma_client.enqueue(session, ["exp-01"])
    except wma_client.NoSidecar:
        pass
    else:
        raise AssertionError("enqueue must refuse when no sidecar is attached")
    assert not (session / ".wma").exists()
    assert _client_main(["wma", "review", "--dir", str(session), "exp-01", "--background"]) == 0
    out = capsys.readouterr().out
    assert "no world-model agent is attached" in out and "no verdict" in out
    assert _client_main(["wma", "status", "--dir", str(session)]) == 0
    assert "no world-model agent is attached" in capsys.readouterr().out
    # once the sidecar has opened its queue, the same call enqueues
    (session / ".wma" / "requests").mkdir(parents=True)
    assert _client_main(["wma", "review", "--dir", str(session), "exp-01", "--background"]) == 0
    assert "queued" in capsys.readouterr().out and list((session / ".wma" / "requests").glob("*.json"))


# ---- the verdict is part of the lock (2026-09-03): the blocking review waits, then hands the launch back ----

def _heuristic_config(session: Path, skill: Path) -> sidecar.Config:
    return sidecar.Config(session_dir=session, skill_dir=skill, history_dir=None, backend="heuristic",
                          model="claude-opus-5", effort="high", budget=backends.Budget(), jobs=1)


def _private_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "private/skills/wma"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("private WMA policy\n", encoding="utf-8")
    return skill


def test_blocking_review_waits_for_the_sidecar_and_prints_the_verdict_line(tmp_path: Path) -> None:
    import threading
    session = _session(tmp_path)
    config = _heuristic_config(session, _private_skill(tmp_path))
    lines: list[str] = []
    # the sidecar answers a little later, as it does in a cell
    worker = threading.Timer(0.2, lambda: sidecar.run(config, once=True))
    worker.start()
    try:
        result = wma_client.review_and_wait(session, "exp-01", timeout_min=0.5, out=lines.append,
                                            poll_s=0.05, heartbeat_s=0.1)
    finally:
        worker.join()
    assert result["state"] == "delivered" and result["request_id"] and result["waited_s"] >= 0.0
    assert result["verdict_path"] == str(session / "memory/cards/exp-01.verdict.json")
    assert any(line.startswith("verdict: L0_runs=") for line in lines), lines
    assert any("do not start it" in line for line in lines)
    assert not (session / "skills/wma").exists()


def test_blocking_review_reports_a_failed_review_instead_of_waiting_out_the_clock(tmp_path: Path) -> None:
    import threading
    session = _session(tmp_path)
    (session / "memory/cards/exp-01.lock.json").unlink()      # the sidecar refuses an unlocked card
    config = _heuristic_config(session, _private_skill(tmp_path))
    lines: list[str] = []
    worker = threading.Timer(0.2, lambda: sidecar.run(config, once=True))
    worker.start()
    try:
        result = wma_client.review_and_wait(session, "exp-01", timeout_min=0.5, out=lines.append,
                                            poll_s=0.05, heartbeat_s=0.1)
    finally:
        worker.join()
    assert result["state"] == "failed" and "locked card required" in result["error"]
    assert result["waited_s"] < 20
    assert any("you may launch" in line for line in lines)


def test_blocking_review_times_out_with_a_heartbeat_and_records_it(tmp_path: Path) -> None:
    session = _session(tmp_path)
    ticks = iter(range(0, 10_000, 7))
    lines: list[str] = []
    result = wma_client.review_and_wait(session, "exp-01", timeout_min=1.0, out=lines.append,
                                        poll_s=0.0, heartbeat_s=30.0, clock=lambda: next(ticks),
                                        sleep=lambda _s: None)
    assert result["state"] == "timeout" and result["waited_s"] >= 60
    assert sum("elapsed" in line for line in lines) >= 1          # the heartbeat kept the tool alive
    assert any("recorded as a timeout; you may launch" in line for line in lines)
    assert (session / ".wma/requests").is_dir() and list((session / ".wma/requests").glob("*.json"))


def test_blocking_review_does_not_return_a_stale_verdict_on_relock(tmp_path: Path) -> None:
    import threading
    session = _session(tmp_path)
    target = session / "memory/cards/exp-01.verdict.json"
    target.write_text('{"generation": "old"}\n', encoding="utf-8")
    prior = wma_client._verdict_version(target)

    rewritten = threading.Timer(
        0.15, lambda: target.write_text('{"generation": "fresh", "larger": true}\n', encoding="utf-8"))
    rewritten.start()
    try:
        result = wma_client.wait_for_verdict(
            session, "exp-01", "new-request", timeout_s=2, poll_s=0.02,
            heartbeat_s=10, out=lambda _line: None, prior_verdict=prior)
    finally:
        rewritten.join()

    assert result["state"] == "delivered"
    assert result["waited_s"] >= 0.1
    assert json.loads(target.read_text())["generation"] == "fresh"


def test_blocking_review_cli_returns_at_once_on_the_control_arm(tmp_path: Path, capsys) -> None:
    session = _session(tmp_path, attached=False)
    assert _client_main(["wma", "review", "--dir", str(session), "exp-01"]) == 0
    assert "no world-model agent is attached" in capsys.readouterr().out
    assert _client_main(["wma", "review", "--dir", str(session), "exp-01", "exp-02"]) == 2
    assert "batch several with --background" in capsys.readouterr().out


def test_the_wait_heartbeat_reaches_a_redirected_log_while_the_wait_is_still_running(tmp_path: Path) -> None:
    """w10r04 (2026-09-03) ran `nohup awm exp_protocol lock … > logs/lock.log &` and tailed the log: with a
    block-buffered stdout the file stayed empty for the whole wait. Every progress line must land at once."""
    import subprocess
    import sys
    import time as _time
    session = _session(tmp_path)
    log = tmp_path / "lock.log"
    script = (
        "import sys; from awm import wma_client\n"
        f"wma_client.review_and_wait({str(session)!r}, 'exp-01', timeout_min=0.05, heartbeat_s=0.1, poll_s=0.02)\n"
    )
    with log.open("w") as handle:
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=handle, stderr=subprocess.STDOUT)
        try:
            deadline = _time.monotonic() + 2.0
            seen = ""
            while _time.monotonic() < deadline and "min elapsed" not in seen:
                _time.sleep(0.1)
                seen = log.read_text(encoding="utf-8")
            assert proc.poll() is None, "the wait must still be running when the heartbeat is read"
            assert "WMA review requested for exp-01" in seen and "min elapsed" in seen, seen
        finally:
            proc.wait(timeout=10)
