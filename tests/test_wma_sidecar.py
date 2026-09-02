"""The scientist sees only a queue client; the private sidecar owns WMA policy."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from exp_protocol_cards import plan_card

from awm import wma_client
from awm.wma import backends, sidecar


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    cards = session / "memory/cards"
    cards.mkdir(parents=True)
    card = plan_card()
    (cards / "exp-01.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    (cards / "exp-01.lock.json").write_text("{}\n", encoding="utf-8")
    return session


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
