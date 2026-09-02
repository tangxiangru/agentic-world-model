"""The scientist gets the protocol; never the meta skill."""

from __future__ import annotations

import os

import pytest

from awm.exp_protocol import install


def test_both_tools_get_the_skill_the_symlink_and_the_agents_block(tmp_path) -> None:
    written = install.install(tmp_path, "both")
    assert (tmp_path / "skills" / "exp_protocol" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "exp_protocol" / "pitfalls.yaml").is_file()
    link = tmp_path / ".claude" / "skills" / "exp_protocol"
    assert link.is_symlink() and os.readlink(link) == "../../skills/exp_protocol"
    assert (link / "SKILL.md").is_file()
    agents = (tmp_path / "AGENTS.md").read_text()
    assert install.BEGIN in agents and "skills/exp_protocol/SKILL.md" in agents
    assert all(p.exists() or p.is_symlink() for p in written)


def test_codex_only_writes_no_claude_symlink(tmp_path) -> None:
    install.install(tmp_path, "codex")
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "AGENTS.md").is_file()


def test_claude_only_writes_no_agents_md(tmp_path) -> None:
    install.install(tmp_path, "claude")
    assert not (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / ".claude" / "skills" / "exp_protocol").is_symlink()
    # the pointer block, not just the skill: --setting-sources "" would hide the skill,
    # and even with it loaded the tool may not open it on its own
    text = (tmp_path / "CLAUDE.md").read_text()
    assert install.BEGIN in text and "skills/exp_protocol/SKILL.md" in text


def test_a_claude_md_the_task_already_has_is_kept(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Task notes\n\nkeep me\n")
    install.install(tmp_path, "claude")
    install.install(tmp_path, "claude")
    text = (tmp_path / "CLAUDE.md").read_text()
    assert text.count(install.BEGIN) == 1 and "keep me" in text


def test_install_is_idempotent_and_keeps_existing_agents_text(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Task\n\nkeep me\n")
    install.install(tmp_path, "both")
    install.install(tmp_path, "both")
    text = (tmp_path / "AGENTS.md").read_text()
    assert text.count(install.BEGIN) == 1 and "keep me" in text


def test_refuses_to_install_the_meta_skill(tmp_path, monkeypatch) -> None:
    meta = tmp_path / "src" / "exp_protocol_meta"
    meta.mkdir(parents=True)
    (meta / "SKILL.md").write_text("---\nname: exp_protocol_meta\n---\n")
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(meta))
    with pytest.raises(install.InstallError):
        install.install(tmp_path / "task", "both")


# ---- review findings (2026-09-01) --------------------------------------------

def test_a_file_the_scientist_wrote_under_the_skill_dir_survives_reinstall(tmp_path) -> None:
    install.install(tmp_path, "both")
    notes = tmp_path / "skills" / "exp_protocol" / "my_notes.md"
    notes.write_text("keep\n")
    install.install(tmp_path, "both")
    assert notes.read_text() == "keep\n"


def test_a_symlinked_skill_dir_is_replaced_not_crashed(tmp_path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "skills" / "exp_protocol").symlink_to(tmp_path / "elsewhere")
    install.install(tmp_path, "both")
    assert not (tmp_path / "skills" / "exp_protocol").is_symlink()
    assert (tmp_path / "skills" / "exp_protocol" / "SKILL.md").is_file()


def test_a_dangling_begin_marker_is_repaired(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Task\n\nkeep me\n\n" + install.BEGIN + "\ntruncated\n")
    install.install(tmp_path, "both")
    text = (tmp_path / "AGENTS.md").read_text()
    assert text.count(install.BEGIN) == 1 and text.count(install.END) == 1 and "keep me" in text


def test_the_copy_is_writable_even_from_a_read_only_source(tmp_path, monkeypatch) -> None:
    # In the sandbox the skill comes from a read-only bind; the scientist's copy is theirs to edit.
    import shutil
    import stat

    from awm import paths

    source = tmp_path / "ro" / "exp_protocol"
    shutil.copytree(paths.REPO_ROOT / "skills" / "exp_protocol", source)
    for path in (source, *source.rglob("*")):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    monkeypatch.setenv("AWM_EXP_PROTOCOL_DIR", str(source))
    task = tmp_path / "task"
    install.install(task, "claude")
    skill = task / "skills" / "exp_protocol"
    assert all(p.stat().st_mode & stat.S_IWUSR for p in (skill, *skill.rglob("*")) if not p.is_symlink())
    (skill / "notes.md").write_text("mine\n")
    install.install(task, "claude")  # the refresh must not choke on its own earlier copy
    assert (skill / "notes.md").read_text() == "mine\n"
    for path in (source, *source.rglob("*")):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
