"""review: point a backend at a card, refuse post-hoc verdicts, stamp what the backend left blank."""

from __future__ import annotations

import pytest

from awm.exp_protocol import schema as cards
from awm.wma import backends, review, schema
from exp_protocol_cards import closed_card, plan_card


@pytest.fixture
def skill(tmp_path):
    d = tmp_path / "repo-skills" / "wma"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: wma\n---\nestimate, do not decide\n")
    return d


def session_with(tmp_path, card):
    s = tmp_path / "session"
    cards.dump_card(s / "memory" / "cards" / f"{card['card_id']}.yaml", card)
    return s


def test_review_writes_a_stamped_verdict_beside_the_card(tmp_path, skill) -> None:
    s = session_with(tmp_path, plan_card())
    v = review.review(s, "exp-01", backends.HeuristicBackend(), mode="offline", skill_dir=skill)
    path = schema.verdict_path(s / "memory" / "cards" / "exp-01.yaml")
    assert path.is_file() and schema.load_verdict(path) == v
    assert v["wma_skill"] == "heuristic-priors" and v["backend"] == "heuristic" and v["mode"] == "offline"


def test_review_stamps_the_skill_hash_when_the_backend_leaves_it_blank(tmp_path, skill) -> None:
    from awm.wma import backends

    class Blank(backends.Backend):
        name = "blank"

        def run(self, brief):
            v = schema.empty_verdict(brief.card_id)
            v["levels"]["L0_runs"]["answer"] = "yes"
            schema.dump_verdict(brief.verdict_path, v)

    s = session_with(tmp_path, plan_card())
    v = review.review(s, "exp-01", Blank(), mode="offline", skill_dir=skill)
    assert v["wma_skill"] == schema.skill_sha(skill) and v["backend"] == "blank"


def test_review_refuses_a_card_that_already_has_a_result(tmp_path, skill) -> None:
    s = session_with(tmp_path, closed_card())
    with pytest.raises(review.ReviewError, match="post-hoc"):
        review.review(s, "exp-01", backends.HeuristicBackend(), mode="offline", skill_dir=skill)
    v = review.review(s, "exp-01", backends.HeuristicBackend(), mode="offline", skill_dir=skill, force=True)
    assert v["card_id"] == "exp-01"


def test_review_migrates_a_v1_card_in_memory_without_touching_the_file(tmp_path, skill) -> None:
    card = plan_card()
    card["schema_version"] = "awm-experiment-card-v1"
    card["elapsed_h"] = card["situation"].pop("elapsed_h")
    s = session_with(tmp_path, card)
    before = (s / "memory" / "cards" / "exp-01.yaml").read_text()
    review.review(s, "exp-01", backends.HeuristicBackend(), mode="offline", skill_dir=skill)
    assert (s / "memory" / "cards" / "exp-01.yaml").read_text() == before


def test_prepare_session_links_the_skill_idempotently(tmp_path, skill) -> None:
    s = tmp_path / "session"
    s.mkdir()
    review.prepare_session(s, skill)
    review.prepare_session(s, skill)
    link = s / "skills" / "wma"
    assert link.is_symlink() and (link / "SKILL.md").is_file()


def test_prompt_names_the_paths_the_mode_and_the_rules(tmp_path, skill) -> None:
    s = session_with(tmp_path, plan_card())
    b = review.make_brief(s, "exp-01", mode="offline", budget=backends.Budget(cpu_min=3, gpu_min=0, wall_min=5),
                          model=None, skill_dir=skill)
    text = b.prompt
    assert str(b.verdict_path) in text and "skills/wma/SKILL.md" in text
    assert "offline" in text and "static" in text
    assert "do not modify" in text.lower() and "wall_min=5" in text
    b2 = review.make_brief(s, "exp-01", mode="online", budget=backends.Budget(), model="m", skill_dir=skill,
                           history_dir=tmp_path / "hist")
    assert "online" in b2.prompt and str(tmp_path / "hist") in b2.prompt


def test_review_stamps_model_and_effort_when_the_backend_leaves_them_blank(tmp_path, skill) -> None:
    class Blank(backends.Backend):
        name = "blank"

        def run(self, brief):
            assert brief.effort == "high" and brief.model == "m-1"
            schema.dump_verdict(brief.verdict_path, schema.empty_verdict(brief.card_id))

    s = session_with(tmp_path, plan_card())
    v = review.review(s, "exp-01", Blank(), mode="offline", skill_dir=skill, model="m-1", effort="high")
    assert v["model"] == "m-1" and v["effort"] == "high"
    v2 = review.review(session_with(tmp_path / "b", plan_card()), "exp-01", backends.HeuristicBackend(),
                       mode="offline", skill_dir=skill)
    assert "model" not in v2 and "effort" not in v2


def test_harness_fields_are_measured_not_copied_from_the_agent(tmp_path, skill) -> None:
    """The first real verdict copied wma_skill, backend and issued_at from the example file. The harness knows
    these; what the agent wrote is overwritten."""
    class Copier(backends.Backend):
        name = "copier"

        def run(self, brief):
            v = schema.empty_verdict(brief.card_id)
            v.update({"wma_skill": "skills/wma/SKILL.md", "backend": "claude-opus-5", "mode": "online",
                      "issued_at": "2026-09-02T00:00:00Z", "model": "made-up", "effort": "max", "card_id": "exp-99"})
            schema.dump_verdict(brief.verdict_path, v)

    s = session_with(tmp_path, plan_card())
    v = review.review(s, "exp-01", Copier(), mode="offline", skill_dir=skill, model="m-1", effort="high")
    assert v["wma_skill"] == schema.skill_sha(skill) and v["backend"] == "copier" and v["mode"] == "offline"
    assert v["model"] == "m-1" and v["effort"] == "high" and v["card_id"] == "exp-01"
    assert v["issued_at"] != "2026-09-02T00:00:00Z"
    # the heuristic backend does not read the skill; its own stamp stays
    v2 = review.review(session_with(tmp_path / "h", plan_card()), "exp-01", backends.HeuristicBackend(),
                       mode="offline", skill_dir=skill)
    assert v2["wma_skill"] == "heuristic-priors"
