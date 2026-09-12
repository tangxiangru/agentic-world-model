import json

from tools.outcome_prediction.prefix_code_fragments import recover_fragment_versions
from tools.outcome_prediction.rpm_code_provenance import TraceIndex


def event(second, content, result=None):
    at = f"2026-09-04T00:00:{second:02d}Z"
    value = {"timestamp": at, "message": {"content": content}}
    if result is not None:
        value["tool_use_result"] = result
    return f"[{at}] {json.dumps(value)}\n"


def bash(second, command, stdout, error=False):
    ident = str(second)
    return event(
        second, [{"type": "tool_use", "id": ident, "name": "Bash", "input": {"command": command}}]
    ) + event(
        second + 1,
        [{"type": "tool_result", "tool_use_id": ident, "is_error": error}],
        {"stdout": stdout, "stderr": "", "interrupted": False},
    )


def fragments(
    content="a\nb\nc\nd\ne\n",
    size=None,
    head_path="evaluate.py",
    tail_path="evaluate.py",
    middle="",
    head_error=False,
    marker="---END---",
):
    lines = content.splitlines()
    size = len(content.encode()) if size is None else size
    head = bash(
        1,
        f'echo "---LS---" && ls -la && echo "---CODE---" && cat {head_path} | head -3',
        f"---LS---\n-rw-r--r-- 1 u g {size} Sep 4 00:00 evaluate.py\n---CODE---\n"
        + "\n".join(lines[:3]),
        error=head_error,
    )
    tail = bash(
        5, f'cat {tail_path} | tail -3 && echo "---END---"', "\n".join(lines[-3:]) + "\n" + marker
    )
    return head + middle + tail


def recover(tmp_path, trace):
    path = tmp_path / "trace.txt"
    path.write_text(trace)
    return recover_fragment_versions(TraceIndex(path))


def test_unique_overlap_and_observed_size_recovers_exact_bytes(tmp_path):
    versions = recover(tmp_path, fragments())
    assert len(versions) == 1
    assert versions[0]["content"] == "a\nb\nc\nd\ne\n"
    assert versions[0]["observed_bytes"] == 10
    assert versions[0]["overlap_lines"] == 1
    assert versions[0]["call_line"] == 3
    assert versions[0]["time"].isoformat() == "2026-09-04T00:00:06+00:00"


def test_missing_final_newline_resolved_by_size(tmp_path):
    versions = recover(tmp_path, fragments("a\nb\nc\nd\ne"))
    assert versions[0]["content"] == "a\nb\nc\nd\ne"


def test_wrong_byte_size_is_not_reconstructed(tmp_path):
    assert recover(tmp_path, fragments(size=12345)) == []


def test_path_mismatch_is_not_reconstructed(tmp_path):
    assert recover(tmp_path, fragments(tail_path="other.py")) == []


def test_failed_read_is_not_reconstructed(tmp_path):
    assert recover(tmp_path, fragments(head_error=True)) == []


def test_wrong_output_delimiter_is_not_reconstructed(tmp_path):
    assert recover(tmp_path, fragments(marker="---NOT-END---")) == []


def test_ambiguous_overlap_is_not_reconstructed(tmp_path):
    assert recover(tmp_path, fragments("a\na\na\na\na\n")) == []


def test_intervening_edit_blocks_reconstruction(tmp_path):
    edit = event(
        3,
        [
            {
                "type": "tool_use",
                "id": "edit",
                "name": "Edit",
                "input": {
                    "file_path": "/home/ben/task/evaluate.py",
                    "old_string": "c",
                    "new_string": "X",
                },
            }
        ],
    )
    assert recover(tmp_path, fragments(middle=edit)) == []


def test_intervening_unknown_script_blocks_reconstruction(tmp_path):
    command = bash(3, "python mutate.py", "")
    assert recover(tmp_path, fragments(middle=command)) == []


def test_mutating_tail_suffix_is_rejected(tmp_path):
    trace = fragments().replace('echo \\"---END---\\"', 'echo \\"---END---\\" && python mutate.py')
    assert recover(tmp_path, trace) == []


def test_nonliteral_path_is_rejected(tmp_path):
    assert recover(tmp_path, fragments(head_path="$FILE")) == []


def test_missing_size_observation_is_rejected(tmp_path):
    trace = fragments().replace("-rw-r--r-- 1 u g 10 Sep 4 00:00 evaluate.py", "no listing")
    assert recover(tmp_path, trace) == []
