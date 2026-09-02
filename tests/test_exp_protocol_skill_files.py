"""The skill's own files obey the skill."""

from __future__ import annotations

import re

import yaml

from awm.exp_protocol import preflight, schema


def test_example_card_validates_before_and_after() -> None:
    card = schema.load_card(preflight.skill_dir() / "example-card.yaml")
    plan = schema.validate_plan(card)
    assert plan.ok, plan.render()
    result = schema.validate_result(card)
    assert result.ok, result.render()


def test_template_parses_and_has_every_section() -> None:
    data = yaml.safe_load((preflight.skill_dir() / "card.template.yaml").read_text())
    assert data["schema_version"] == schema.CARD_SCHEMA
    for section in schema.PLAN_SECTIONS + schema.RESULT_SECTIONS:
        assert isinstance(data.get(section), dict), section


def test_skill_md_has_frontmatter_and_names_every_command() -> None:
    text = (preflight.skill_dir() / "SKILL.md").read_text()
    assert text.startswith("---\nname: exp_protocol\n")
    assert re.search(r"^description: .+", text, re.MULTILINE)
    for cmd in ("new", "check", "preflight", "lock", "close", "index"):
        assert f"awm exp_protocol {cmd}" in text, cmd


def test_stop_hook_is_stdlib_only_and_blocks_once() -> None:
    src = (preflight.skill_dir() / "hooks" / "stop_open_cards.py").read_text()
    assert "import yaml" not in src and "from awm" not in src
    assert "stop_hook_active" in src


def test_repo_symlinks_resolve_to_the_skills() -> None:
    from awm import paths
    for name in ("exp_protocol", "exp_protocol_meta"):
        link = paths.REPO_ROOT / ".claude" / "skills" / name
        assert link.is_symlink(), link
        assert (link / "SKILL.md").is_file(), link


def test_agents_md_points_codex_at_skills() -> None:
    from awm import paths
    text = (paths.REPO_ROOT / "AGENTS.md").read_text()
    assert "skills/exp_protocol/SKILL.md" in text and "skills/exp_protocol_meta/SKILL.md" in text
