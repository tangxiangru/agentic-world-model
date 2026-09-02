"""The skill's own files obey the skill."""

from __future__ import annotations

import re

from awm import paths
from awm.wma import review, schema


def test_example_verdict_validates() -> None:
    v = schema.load_verdict(review.default_skill_dir() / "verdict.example.json")
    r = schema.validate_verdict(v)
    assert r.ok, r.render()


def test_skill_md_names_the_levels_the_file_and_the_offline_rule() -> None:
    text = (review.default_skill_dir() / "SKILL.md").read_text()
    assert text.startswith("---\nname: wma\n")
    assert re.search(r"^description: .+", text, re.MULTILINE)
    for needle in schema.LEVELS + ("verdict.json", "static_check", "Offline mode", "Never a new direction"):
        assert needle in text, needle


def test_repo_symlink_and_agents_md() -> None:
    link = paths.REPO_ROOT / ".claude" / "skills" / "wma"
    assert link.is_symlink() and (link / "SKILL.md").is_file()
    assert "skills/wma/SKILL.md" in (paths.REPO_ROOT / "AGENTS.md").read_text()
