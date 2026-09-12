"""Generation config evidence must be path-bound, pre-cutoff and complete."""

import json

from tools.outcome_prediction.prefix_generation import index_session, resolve_for_checkpoint


def event(second, content, result=None):
    at = f"2026-09-04T00:00:{second:02d}Z"
    value = {"timestamp": at, "message": {"content": content}}
    if result is not None:
        value["tool_use_result"] = result
    return f"[{at}] {json.dumps(value)}\n"


def bash(second, command, stdout="", ident=None, error=False):
    ident = ident or str(second)
    return event(
        second, [{"type": "tool_use", "id": ident, "name": "Bash", "input": {"command": command}}]
    ) + event(
        second + 1,
        [{"type": "tool_result", "tool_use_id": ident, "is_error": error, "content": stdout}],
        {"stdout": stdout, "stderr": "", "interrupted": False},
    )


def resolve(tmp_path, events, paths=("out",), cutoff="2026-09-04T00:00:59Z", **kwargs):
    (tmp_path / "solve_out_sanitized.txt").write_text(events)
    return resolve_for_checkpoint(index_session(tmp_path), paths, cutoff, **kwargs)


def test_unrelated_base_config_does_not_leak(tmp_path):
    record = resolve(
        tmp_path, bash(1, "cat /cache/base/generation_config.json", '{"temperature": 1}')
    )
    assert record["generation_config"] is None
    assert record["status"] == "unknown"


def test_exact_path_read_preserves_absent_fields(tmp_path):
    record = resolve(tmp_path, bash(1, "cat out/generation_config.json", '{"do_sample": false}'))
    assert record["generation_config"] == {"do_sample": False}
    assert record["status"] == "trajectory_reconstructed"
    assert "temperature" not in record["generation_config"]


def test_result_at_cutoff_is_excluded(tmp_path):
    record = resolve(
        tmp_path,
        bash(1, "cat out/generation_config.json", '{"temperature": 1}'),
        cutoff="2026-09-04T00:00:02Z",
    )
    assert record["generation_config"] is None


def test_oneoff_patch_print_recovers_complete_dict_without_base(tmp_path):
    command = "python - <<'EOF'\nimport json\np='out/generation_config.json'\nc=json.load(open(p))\nc['temperature']=0.0\nc.pop('top_p',None)\njson.dump(c,open(p,'w'))\nprint(c)\nEOF"
    record = resolve(
        tmp_path,
        bash(1, command, "{'temperature': 0.0, 'do_sample': False, 'eos_token_id': [1, 106]}"),
    )
    assert record["generation_config"] == {
        "temperature": 0.0,
        "do_sample": False,
        "eos_token_id": [1, 106],
    }
    assert record["evidence"][-1]["kind"] == "path_bound_stdout_read"


def test_patch_replays_on_exact_prior_read(tmp_path):
    trace = bash(
        1, "cat out/generation_config.json", '{"temperature": 1, "top_p": 0.95, "eos_token_id": 7}'
    )
    trace += bash(
        3,
        "python -c \"import json; p='out/generation_config.json'; c=json.load(open(p)); c['temperature']=0; c.pop('top_p',None); json.dump(c,open(p,'w'))\"",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0, "eos_token_id": 7}


def test_patch_without_base_is_partial_not_full(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; p='out/generation_config.json'; c=json.load(open(p)); c['temperature']=0; json.dump(c,open(p,'w'))\"",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["partial_fields"] == {"temperature": 0}


def test_later_sed_blocks_prior_known_config(tmp_path):
    trace = bash(1, "cat out/generation_config.json", '{"temperature": 1}')
    trace += bash(3, "sed -i 's/1/0/' out/generation_config.json")
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["evidence"][-1]["kind"] == "unresolved_mutation"


def test_later_patch_after_cutoff_cannot_change_input(tmp_path):
    trace = bash(1, "cat out/generation_config.json", '{"temperature": 1}')
    trace += bash(
        5,
        "python -c \"import json; json.dump({'temperature':0},open('out/generation_config.json','w'))\"",
    )
    record = resolve(tmp_path, trace, cutoff="2026-09-04T00:00:04Z")
    assert record["generation_config"] == {"temperature": 1}


def test_no_confusion_between_two_printed_configs(tmp_path):
    trace = bash(
        1,
        "cat out/generation_config.json; cat other/generation_config.json",
        '{"temperature": 1}\n{"temperature": 0}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None


def test_literal_python_write(tmp_path):
    trace = bash(
        1,
        "python - <<'PY'\nimport json\nout='out'\nc={'temperature':0, 'do_sample':False}\nwith open(out+'/generation_config.json','w') as f:\n    json.dump(c,f)\nPY",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0, "do_sample": False}


def test_snapshot_not_claimed_archived_verified(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text('{"temperature": 0}')
    record = resolve(tmp_path, "", archived_config_path=path)
    assert record["status"] == "snapshot_file_observed"
    assert record["generation_config"] == {"temperature": 0}


def test_print_before_mutation_is_not_post_mutation_state(tmp_path):
    trace = bash(
        1,
        "python - <<'PY'\nimport json\np='out/generation_config.json'\nc=json.load(open(p))\nprint(c)\nc['temperature']=0\njson.dump(c,open(p,'w'))\nPY",
        '{"temperature": 1}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["partial_fields"] == {"temperature": 0}


def test_cat_before_write_is_not_post_write_state(tmp_path):
    trace = bash(
        1,
        "cat out/generation_config.json; python -c \"import json; json.dump({'temperature':0},open('out/generation_config.json','w'))\"",
        '{"temperature": 1}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0}


def test_pending_write_blocks_earlier_complete_read(tmp_path):
    trace = bash(1, "cat out/generation_config.json", '{"temperature": 1}')
    trace += event(
        3,
        [
            {
                "type": "tool_use",
                "id": "pending",
                "name": "Write",
                "input": {
                    "file_path": "/home/ben/task/out/generation_config.json",
                    "content": '{"temperature": 0}',
                },
            }
        ],
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["blockers"]


def test_unsupported_conditional_mutation_is_unknown(tmp_path):
    trace = bash(1, "cat out/generation_config.json", '{"temperature": 1}')
    trace += bash(
        3,
        "python - <<'PY'\nimport json\np='out/generation_config.json'\nc=json.load(open(p))\nif dynamic_condition:\n    c['temperature']=0\njson.dump(c,open(p,'w'))\nPY",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None


def test_json_dumps_print_preserves_path_binding(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; print(json.dumps(json.load(open('out/generation_config.json'))))\"",
        '{"temperature": 0}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0}


def test_declared_script_config_is_audit_only(tmp_path):
    source = "import json\nOUT = parse_args().output\ndef train():\n    json.dump({'temperature':0, 'do_sample':False}, open(OUT+'/generation_config.json','w'))\ntrain()\n"
    script = "/home/ben/task/train.py"
    trace = event(
        1,
        [
            {
                "type": "tool_use",
                "id": "write",
                "name": "Write",
                "input": {"file_path": script, "content": source},
            }
        ],
    )
    trace += event(
        2,
        [{"type": "tool_result", "tool_use_id": "write", "is_error": False}],
        {"filePath": script, "content": source, "userModified": False},
    )
    trace += bash(3, "python train.py --output out")
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["declared_config_candidates"][0]["declared_generation_config"] == {
        "temperature": 0,
        "do_sample": False,
    }
    assert (
        record["declared_config_candidates"][0]["status"] == "declared_only_not_checkpoint_verified"
    )


def test_declared_but_unlaunched_script_is_not_candidate(tmp_path):
    source = "import json\njson.dump({'temperature':0}, open('out/generation_config.json','w'))\n"
    script = "/home/ben/task/train.py"
    trace = event(
        1,
        [
            {
                "type": "tool_use",
                "id": "write",
                "name": "Write",
                "input": {"file_path": script, "content": source},
            }
        ],
    )
    trace += event(
        2,
        [{"type": "tool_result", "tool_use_id": "write", "is_error": False}],
        {"filePath": script, "content": source, "userModified": False},
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert not record["declared_config_candidates"]


def test_mutated_memory_without_save_is_not_file_evidence(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; c=json.load(open('out/generation_config.json')); c['temperature']=0; print(c)\"",
        '{"temperature": 0}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None


def test_loaded_config_written_to_new_path_binds_new_path(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; c=json.load(open('/cache/base/generation_config.json')); c['temperature']=0; json.dump(c,open('out/generation_config.json','w')); print(c)\"",
        '{"temperature": 0, "eos_token_id": 7}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0, "eos_token_id": 7}


def test_path_read_text_json_loads_is_bound(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; from pathlib import Path; print(json.loads(Path('out/generation_config.json').read_text()))\"",
        '{"temperature": 0}',
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0}


def test_read_tool_complete_structured_file(tmp_path):
    path = "/home/ben/task/out/generation_config.json"
    trace = event(
        1, [{"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": path}}]
    )
    trace += event(
        2,
        [{"type": "tool_result", "tool_use_id": "read", "is_error": False}],
        {
            "type": "text",
            "file": {
                "filePath": path,
                "content": '{"temperature":0}',
                "startLine": 1,
                "numLines": 1,
                "totalLines": 1,
            },
        },
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {"temperature": 0}
    assert record["evidence"][0]["kind"] == "complete_read_tool_result"


def test_empty_config_is_not_missing_config(tmp_path):
    record = resolve(tmp_path, bash(1, "cat out/generation_config.json", "{}"))
    assert record["generation_config"] == {}
    assert record["status"] == "trajectory_reconstructed"


def test_masked_shell_failure_does_not_establish_literal_write(tmp_path):
    trace = bash(
        1,
        "python -c \"import json; json.dump({'temperature':0},open('out/generation_config.json','w'))\"; echo done",
        "Traceback (most recent call last):\nPermissionError: Permission denied\ndone",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None


def test_verified_literal_file_copy_in_same_call(tmp_path):
    command = 'cat > final_model/generation_config.json << \'JSON\'\n{"eos_token_id":[151645,151643],"pad_token_id":151643,"transformers_version":"4.57.3"}\nJSON\ncat final_model/generation_config.json\n# Also fix in ckpts\ncp final_model/generation_config.json ckpts/exp-01/generation_config.json\necho OK'
    trace = bash(
        1,
        command,
        '{"eos_token_id":[151645,151643],"pad_token_id":151643,"transformers_version":"4.57.3"}\nOK',
    )
    record = resolve(tmp_path, trace, paths=("ckpts/exp-01",))
    assert record["generation_config"] == {
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "transformers_version": "4.57.3",
    }
    assert record["evidence"][-1]["kind"] == "exact_foreground_config_copy"


def test_unknown_copy_source_invalidates_old_destination(tmp_path):
    trace = bash(1, "cat out/generation_config.json", '{"temperature":1}')
    trace += bash(3, "cp unknown/generation_config.json out/generation_config.json")
    record = resolve(tmp_path, trace)
    assert record["generation_config"] is None
    assert record["evidence"][-1]["kind"] == "exact_foreground_config_copy"


def test_failed_copy_does_not_recover_source_at_destination(tmp_path):
    trace = bash(1, "cat src/generation_config.json", '{"temperature":1}')
    trace += bash(
        3,
        "cp src/generation_config.json out/generation_config.json",
        "Permission denied",
        error=True,
    )
    assert resolve(tmp_path, trace)["generation_config"] is None


def test_background_copy_is_not_foreground_evidence(tmp_path):
    trace = bash(1, "cat src/generation_config.json", '{"temperature":1}')
    trace += bash(3, "cp src/generation_config.json out/generation_config.json &")
    assert resolve(tmp_path, trace)["generation_config"] is None


def test_extra_mutation_after_copy_invalidates_destination(tmp_path):
    trace = bash(1, "cat src/generation_config.json", '{"temperature":1}')
    trace += bash(
        3,
        "cp src/generation_config.json out/generation_config.json; sed -i 's/1/0/' out/generation_config.json",
    )
    assert resolve(tmp_path, trace)["generation_config"] is None


def test_two_python_snippets_replay_print_then_saved_patch(tmp_path):
    command = "python -c \"import json; gc=json.load(open('out/generation_config.json')); print('current:',gc)\"\npython -c \"import json; gc=json.load(open('out/generation_config.json')); gc['eos_token_id']=[1,106]; gc['do_sample']=False; gc.pop('temperature',None); gc.pop('top_k',None); gc.pop('top_p',None); json.dump(gc,open('out/generation_config.json','w')); print('done')\""
    trace = bash(
        1,
        command,
        "current: {'bos_token_id':2,'cache_implementation':'hybrid','eos_token_id':[1,106],'pad_token_id':0,'transformers_version':'4.57.3'}\ndone",
    )
    record = resolve(tmp_path, trace)
    assert record["generation_config"] == {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "transformers_version": "4.57.3",
        "do_sample": False,
    }
    assert record["evidence"][0]["kind"] == "path_bound_stdout_read"
    assert record["evidence"][-1]["kind"] == "python_json_write"
    assert record["evidence"][0]["command_offset"] < record["evidence"][-1]["command_offset"]


def test_ambiguous_two_prints_do_not_seed_a_later_patch(tmp_path):
    command = "python -c \"import json; print(json.load(open('out/generation_config.json')))\"\npython -c \"import json; print(json.load(open('other/generation_config.json')))\"\npython -c \"import json; c=json.load(open('out/generation_config.json')); c['temperature']=0; json.dump(c,open('out/generation_config.json','w'))\""
    trace = bash(1, command, '{"temperature":1}\n{"temperature":0.5}')
    assert resolve(tmp_path, trace)["generation_config"] is None


def test_read_then_patch_failure_remains_unknown(tmp_path):
    command = "python -c \"import json; print(json.load(open('out/generation_config.json')))\"\npython -c \"import json; c=json.load(open('out/generation_config.json')); c['temperature']=0; json.dump(c,open('out/generation_config.json','w'))\""
    trace = bash(1, command, '{"temperature":1}\nPermission denied', error=True)
    assert resolve(tmp_path, trace)["generation_config"] is None
