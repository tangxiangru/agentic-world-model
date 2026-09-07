import copy
import hashlib
import json

import pytest

from tools.outcome_prediction.wm_lineage import build_artifact, build_graph, card_refs


def row(number, *, cell="r0-90", out=None, parent="base", argv=None, time=None):
    card = f"exp-{number:02d}"
    base = "org/base-model"
    parent_value = base if parent == "base" else parent
    setup = {
        "base_model": base,
        "parent_checkpoint": {
            "origin": "base_model" if parent == "base" else "unknown",
            "path": parent_value,
            "hash": None,
        },
        "output_dir": out or f"/task/ckpts/{card}",
        "command": {
            "cwd": "/task",
            "script": "/task/train.py",
            "argv": argv or ["python", "train.py", "--model", parent_value],
        },
        "method": {"family": "sft"},
        "data": [],
    }
    return {
        "example_id": f"{cell}/{card}",
        "cell_id": cell,
        "card_id": card,
        "first_submitted_at": time or f"2026-01-01T{number:02d}:00:00Z",
        "model_input": {"task": {"base_model": base}, "plan": {"setup": setup}},
        "label": {"accuracy": 0.4},
        "audit": {"eligible": True},
    }


def setup(record):
    return record["model_input"]["plan"]["setup"]


def edges(graph, number, kind="weights", cell="r0-90"):
    return [e for e in graph["nodes"][f"{cell}/exp-{number:02d}"]["parents"] if e["kind"] == kind]


def test_declared_artifacts_preserve_intermediate_variant_and_topology():
    graph = build_graph(
        [row(1), row(2, parent="ckpts/exp-01/checkpoint-123"), row(3, parent="ckpts/exp-02/final")]
    )
    edge = edges(graph, 2)[0]
    assert edge["producer_id"] == "r0-90/exp-01"
    assert edge["artifact"] == "/task/ckpts/exp-01/checkpoint-123"
    assert edge["artifact_variant"] == "checkpoint-123" and not edge["content_verified"]
    assert graph["nodes"]["r0-90/exp-03"]["topological_closure"] == [
        "r0-90/exp-01",
        "r0-90/exp-02",
        "r0-90/exp-03",
    ]
    assert not graph["nodes"]["r0-90/exp-01"]["terminal_in_resolved_graph"]
    assert graph["nodes"]["r0-90/exp-03"]["terminal_in_resolved_graph"]
    assert all(not n["full_recipe_execution_certified"] for n in graph["nodes"].values())


def test_command_path_wins_over_wrong_origin():
    previous = row(1, out="/task/ckpts/origin")
    actual = row(2, out="/task/ckpts/actual")
    child = row(3, parent="ckpts/actual/final")
    setup(child)["parent_checkpoint"]["origin"] = "exp-01"
    graph = build_graph([previous, actual, child])
    assert edges(graph, 3)[0]["producer_id"] == actual["example_id"]
    assert (
        "origin_reference_disagrees_with_resolved_weight_inputs"
        in graph["nodes"][child["example_id"]]["warnings"]
    )


def test_unknown_command_path_does_not_fall_back_to_origin_or_base():
    child = row(2, parent="ckpts/not-declared")
    setup(child)["parent_checkpoint"]["origin"] = "exp-01"
    graph = build_graph([row(1), child])
    edge = edges(graph, 2)[0]
    assert edge["producer_id"] is None and edge["resolution_status"] == "unresolved"


def test_declared_base_hf_cache_snapshot_resolves_without_losing_revision():
    snapshot = "/cache/hub/models--org--base-model/snapshots/" + "a" * 40
    example = row(1, parent=snapshot)
    graph = build_graph([example])
    edge = edges(graph, 1)[0]
    assert edge["resolution_status"] == "declared_base_model"
    assert edge["base_revision"] == "a" * 40
    assert edge["artifact"] == snapshot
    assert not edge["content_verified"]
    assert not graph["nodes"][example["example_id"]]["closure_has_unresolved_dependencies"]


@pytest.mark.parametrize(
    "snapshot",
    [
        "/cache/hub/models--other--base-model/snapshots/" + "a" * 40,
        "/cache/hub/models--org--base-model/snapshots/not-a-revision",
        "/cache/hub/models--org--base-model/snapshots/" + "a" * 40 + "/other",
        "/models/org/base-model",
    ],
)
def test_arbitrary_or_wrong_model_cache_path_is_not_the_base(snapshot):
    graph = build_graph([row(1, parent=snapshot)])
    assert edges(graph, 1)[0]["resolution_status"] == "unresolved"


def test_merge_retains_three_artifacts_from_two_producers():
    merge = row(
        3,
        argv=[
            "python",
            "soup.py",
            "--ckpt",
            "ckpts/exp-01:1",
            "--ckpt",
            "ckpts/exp-02/checkpoint-257:1",
            "--ckpt",
            "ckpts/exp-02:1",
            "--out",
            "ckpts/soup",
        ],
    )
    setup(merge)["method"]["family"] = "merge"
    graph = build_graph([row(1), row(2), merge])
    selected = edges(graph, 3)
    assert len(selected) == 3
    assert [e["producer_id"] for e in selected] == ["r0-90/exp-01", "r0-90/exp-02", "r0-90/exp-02"]
    assert len({e["artifact"] for e in selected}) == 3
    assert all(e["merge_coefficient"] == "1" for e in selected)


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "soup.py", "--models", "ckpts/exp-01,ckpts/exp-02", "--out", "ckpts/new"],
        ["python", "soup.py", "--srcs", "ckpts/exp-01", "ckpts/exp-02", "--dst", "ckpts/new"],
        ["python", "soup.py", "--a", "ckpts/exp-01", "--b", "ckpts/exp-02"],
    ],
)
def test_merge_cli_ingredient_forms(argv):
    graph = build_graph([row(1), row(2), row(3, argv=argv)])
    assert {e["producer_id"] for e in edges(graph, 3)} == {"r0-90/exp-01", "r0-90/exp-02"}


def test_launch_alias_and_inferred_cwd():
    initial = row(1)
    s = setup(initial)
    del s["output_dir"]
    del s["command"]
    s["launch"] = {
        "script": "/task/scripts/train.py",
        "command": ["python", "scripts/train.py", "--out", "ckpts/alias"],
    }
    graph = build_graph([initial, row(2, parent="ckpts/alias/checkpoint-10")])
    assert edges(graph, 2)[0]["producer_id"] == initial["example_id"]
    assert "cwd_inferred_from_declared_script" in graph["nodes"][initial["example_id"]]["warnings"]


def test_data_reuse_does_not_require_producer_training():
    producer = row(1)
    setup(producer)["data"] = [
        {
            "path": "data/built.jsonl",
            "built_by": "build.py",
            "build_command": ["python", "build.py", "--out", "data/built.jsonl"],
        }
    ]
    consumer = row(
        2, argv=["python", "train.py", "--model", "org/base-model", "--data", "data/built.jsonl"]
    )
    graph = build_graph([producer, consumer])
    selected = edges(graph, 2, "generated_data")
    assert selected[0]["producer_id"] == producer["example_id"]
    assert selected[0]["dependency_scope"] == "data_builder_only"
    assert all(e["producer_id"] is None for e in edges(graph, 2, "weights"))
    assert graph["nodes"][consumer["example_id"]]["operation_expansion_required"]


def test_generated_rollouts_require_generator_weights_not_data_only():
    consumer = row(2)
    setup(consumer)["data"] = [
        {
            "path": "data/rft.jsonl",
            "built_by": "gen.py",
            "build_command": [
                "python",
                "gen.py",
                "--model",
                "ckpts/exp-01/checkpoint-12",
                "--out",
                "data/rft.jsonl",
            ],
        }
    ]
    graph = build_graph([row(1), consumer])
    edge = next(e for e in edges(graph, 2, "generated_data") if e["producer_id"])
    assert edge["producer_id"] == "r0-90/exp-01"
    assert edge["dependency_scope"] == "requires_producer_weights"
    assert edge["artifact_variant"] == "checkpoint-12"


def test_shorthand_data_refs_are_ambiguous_data_not_weights():
    consumer = row(3)
    setup(consumer)["data"] = [
        {
            "source": "derived:exp01/02; historical values irrelevant",
            "selection": "dedup against exp09",
        }
    ]
    graph = build_graph([row(1), row(2), consumer])
    selected = edges(graph, 3, "generated_data")
    assert {e["producer_id"] for e in selected} == {"r0-90/exp-01", "r0-90/exp-02"}
    assert all(e["dependency_scope"] == "unresolved" and e["ambiguity_reason"] for e in selected)
    assert all(e["producer_id"] is None for e in edges(graph, 3, "weights"))
    assert "exp-09" not in json.dumps(graph)


def test_configuration_reference_is_typed_separately():
    child = row(
        2,
        argv=[
            "python",
            "train.py",
            "--model",
            "org/base-model",
            "--tokenizer-from",
            "ckpts/exp-01",
        ],
    )
    graph = build_graph([row(1), child])
    assert edges(graph, 2, "configuration")[0]["producer_id"] == "r0-90/exp-01"
    assert edges(graph, 2, "configuration")[0]["dependency_scope"] == "configuration_only"


def test_longest_prefix_then_latest_prior_declaration():
    earlier = row(1, out="/task/ckpts/shared")
    later = row(2, out="/task/ckpts/shared")
    graph = build_graph([earlier, later, row(3, parent="ckpts/shared/final")])
    assert edges(graph, 3)[0]["producer_id"] == later["example_id"]


def test_same_time_producers_ambiguous_and_future_never_selected():
    a = row(1, out="/task/ckpts/shared")
    b = row(2, out="/task/ckpts/shared", time=a["first_submitted_at"])
    graph = build_graph([a, b, row(3, parent="ckpts/shared")])
    assert edges(graph, 3)[0]["unresolved_reason"] == "ambiguous_same_time_producers"
    future = row(4, out="/task/ckpts/future")
    graph = build_graph([row(2, parent="ckpts/future"), future])
    assert edges(graph, 2)[0]["unresolved_reason"] == "artifact_has_no_strictly_earlier_producer"


def test_cross_run_alias_never_resolves():
    graph = build_graph([row(1, cell="r0-91"), row(2, parent="ckpts/exp-01")])
    assert edges(graph, 2)[0]["producer_id"] is None


def test_missing_parent_not_silently_base():
    item = row(1)
    setup(item).pop("parent_checkpoint")
    setup(item)["command"]["argv"] = ["python", "train.py"]
    assert edges(build_graph([item]), 1)[0]["resolution_status"] == "unresolved"


def test_dynamic_paths_and_shell_commands_never_execute(tmp_path):
    marker = tmp_path / "must-not-exist"
    child = row(
        2, argv=["bash", "-c", f"touch {marker}; python train.py --model '$MODEL' --out ckpts/new"]
    )
    graph = build_graph([row(1), child])
    assert not marker.exists()
    assert edges(graph, 2)[0]["unresolved_reason"] == "missing_or_dynamic_artifact"


def test_internal_merge_then_eval_does_not_create_self_cycle():
    child = row(
        3,
        argv=[
            "bash",
            "-c",
            "python soup.py --srcs ckpts/exp-01 ckpts/exp-02 --dst ckpts/merged && python eval.py --model ckpts/merged --out results/eval.json",
        ],
    )
    graph = build_graph([row(1), row(2), child])
    selected = edges(graph, 3)
    assert any(e["resolution_status"] == "internal_declared_operation" for e in selected)
    assert not any(e["producer_id"] == child["example_id"] for e in selected)
    assert not any(
        o["artifact"].endswith("eval.json") for o in graph["nodes"][child["example_id"]]["outputs"]
    )


def test_inline_python_config_literal_without_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    code = f"open('{marker}','w').write('bad'); p='ckpts/exp-01/generation_config.json'; prior_score=0.99"
    child = row(2, argv=["python", "-c", code])
    graph = build_graph([row(1), child])
    assert not marker.exists()
    assert edges(graph, 2, "configuration")[0]["producer_id"] == "r0-90/exp-01"
    assert "0.99" not in json.dumps(graph)


def test_merge_crosscheck_does_not_invent_ingredients():
    a = row(2, cell="r0-01")
    b = row(5, cell="r0-01")
    merge = row(
        7,
        cell="r0-01",
        argv=["python", "soup.py", "--models", "ckpts/unknown", "--out", "ckpts/new"],
    )
    graph = build_graph([a, b, merge])
    node = graph["nodes"][merge["example_id"]]
    assert node["merge_crosscheck"]["expected_card_ids"] == ["exp-02", "exp-05"]
    assert not node["merge_crosscheck"]["matches"]
    assert all(e["producer_id"] is None for e in node["parents"])


def test_all_label_history_prose_and_code_mutations_leave_graph_identical():
    originals = [row(1), row(2, parent="ckpts/exp-01/final")]
    changed = copy.deepcopy(originals)
    for item in changed:
        item["label"] = {"accuracy": 999, "anything": "private target"}
        item["result"] = {"accuracy": 0.98}
        item["audit"] = {"eligible": False}
        item["prior_observations"] = [{"card": "exp-99", "accuracy": 0.87}]
        item["model_input"]["known_previous_checkpoints"] = [{"score": 0.12345}]
        item["model_input"]["code"] = [{"content": "exp99 = 0.76"}]
        item["model_input"]["plan"].update(
            problem={"evidence": "exp88 failed"},
            hypothesis="exp77 won",
            evaluation={"accuracy": 0.66},
        )
        setup(item)["method"]["other"] = "exp-60 got score 0.987"
        setup(item)["notes"] = "result winner exp-55"
    assert build_graph(originals) == build_graph(changed)
    assert len(build_graph(changed)["nodes"]) == 2


def test_data_source_prose_values_not_copied_into_graph():
    one, two = row(1), row(2)
    setup(two)["data"] = [{"source": "derived:exp01 accuracy 0.12345"}]
    altered = copy.deepcopy(two)
    setup(altered)["data"][0]["source"] = "derived:exp01 accuracy 0.99876"
    assert build_graph([one, two]) == build_graph([one, altered])
    assert "0.12345" not in json.dumps(build_graph([one, two]))


def test_card_reference_parser_shorthand():
    assert card_refs("exp01/02/3 + exp-04; exp_05") == [
        "exp-01",
        "exp-02",
        "exp-03",
        "exp-04",
        "exp-05",
    ]


def test_short_options_do_not_become_generated_artifacts():
    target = row(1)
    setup(target)["data"] = [
        {"build_command": ["python", "gen.py", "--out", "data/generated.jsonl", "-n", "4"]}
    ]
    outputs = build_graph([target])["nodes"][target["example_id"]]["outputs"]
    assert all(not o["artifact"].endswith(("/-n", "/4")) for o in outputs)


def test_artifact_immutable_and_raw_hash_outside_blind_graph(tmp_path):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(row(1)) + "\n")
    output = tmp_path / "graph"
    graph = build_artifact(inventory, output)
    assert len(graph["nodes"]) == 1
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
    assert provenance["label_values_used"] is False
    with pytest.raises(FileExistsError):
        build_artifact(inventory, output)
