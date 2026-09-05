"""The meta skill exists, is named, and points at the record template and the metrics."""

from __future__ import annotations

import re

from awm import paths


def meta_dir():
    return paths.REPO_ROOT / "skills" / "exp_protocol_meta"


def test_meta_skill_has_frontmatter_and_the_loop() -> None:
    text = (meta_dir() / "SKILL.md").read_text()
    assert text.startswith("---\nname: exp_protocol_meta\n")
    assert re.search(r"^description: .+", text, re.MULTILINE)
    for needle in ("awm exp_protocol collect", "iteration_record.template.md", "metrics.md",
                   "held-out", "seeds", "never", "pitfalls.yaml", "trace_review.md",
                   "subagent", "directions-ledger.md", "4-cell screening"):
        assert needle in text, needle


def test_metrics_doc_defines_the_four_metrics() -> None:
    text = (meta_dir() / "metrics.md").read_text()
    for m in ("accuracy", "pitfalls_cost_h", "n_locked_open", "fields_filled",
              "greedy_shipped", "largest_eval_n", "hours_to_first_train_launch"):
        assert m in text, m


def test_record_template_has_the_sections() -> None:
    text = (meta_dir() / "iteration_record.template.md").read_text()
    for h in ("## Variants", "## Cells", "## Results", "## Trace review", "## Directions",
              "## Decision", "## Change", "## Evidence"):
        assert h in text, h


def test_trace_review_brief_has_the_questions_and_the_schema() -> None:
    text = (meta_dir() / "trace_review.md").read_text()
    for needle in ("## The brief", "## The synthesis brief", "tools/exp_protocol_cell_read.py",
                   "tools/exp_protocol_trace_timeline.py", "greedy_shipped:", "largest_eval_n:",
                   "one_protocol_change:", "modify the repository"):
        assert needle in text, needle
