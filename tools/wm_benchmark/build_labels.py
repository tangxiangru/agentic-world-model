"""Step 2 of the construction log: Y and S for every ten-run evaluation on the HF mirror.

    python -m tools.wm_benchmark.build_labels <mirror_root> <out_dir>

Reads  eval_matrix_1k/{results,locked_test/results}/*.json   (3,000 controlled cells)
       rescore10/results/*.json                              (1,182 native runs)
       eval_matrix_1k/policies.json, checkpoints_meta/*/generation_config.json,
       rescore10/eval/templates/*.jinja
Writes <out_dir>/labels.jsonl   one row per evaluation: ids, validation status, Y recomputed
                                from the per-question matrix, per-run rates, S, weight identity,
                                and the path of the Inspect log that is Z.
       <out_dir>/labels_summary.json

Y is never copied from the file: it is recomputed from per_problem and the stored aggregate is
only compared against it. Rows that fail any check keep status "invalid" with the reasons;
nothing is repaired or dropped here.
"""
from __future__ import annotations

import ast
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

N_QUESTIONS = {"gsm8k": 1319, "aime2025": 30}
KIT_MAX_TOKENS = {"gsm8k": 4000, "aime2025": 16000}
TEMPLATE_BY_FAMILY = {"gemma": "gemma3.jinja", "qwen": "qwen3.jinja"}
BASE_BY_FAMILY = {"gemma": "google/gemma-3-4b-pt", "qwen": "Qwen/Qwen3-4B-Base"}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def literal(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def validate_matrix(data: dict, benchmark: str):
    """Ten complete, matched, binary runs; aggregates agree with the per-question matrix."""
    errors = []
    n = N_QUESTIONS[benchmark]
    pp = data.get("per_problem")
    if data.get("epochs") != 10:
        errors.append("epochs_not_10")
    if data.get("n_problems") != n:
        errors.append("n_problems_mismatch")
    if not isinstance(pp, dict) or len(pp) != n:
        errors.append("per_problem_count_mismatch")
        return errors, None
    bad = [q for q, v in pp.items() if not isinstance(v, list) or len(v) != 10 or any(x not in (0, 1, 0.0, 1.0) for x in v)]
    if bad:
        errors.append(f"nonbinary_or_short_rows:{len(bad)}")
        return errors, None
    runs = [sum(pp[q][r] for q in pp) / n for r in range(10)]
    stored = data.get("per_epoch_accuracy") or []
    if len(stored) != 10 or any(abs(a - b) > 1e-9 for a, b in zip(runs, stored)):
        errors.append("per_epoch_accuracy_disagrees")
    if abs(statistics.mean(runs) - float(data.get("accuracy", -1))) > 1e-9:
        errors.append("accuracy_disagrees")
    if data.get("n_sample_errors", 0) not in (0, None):
        errors.append(f"sample_errors:{data['n_sample_errors']}")
    return errors, runs


def family_of(config: dict | None) -> str | None:
    arch = " ".join(config.get("architectures", [])).lower() if config else ""
    if "gemma" in arch:
        return "gemma"
    if "qwen" in arch:
        return "qwen"
    return None


def derive_native_sampling(gen_cfg: dict | None, benchmark: str) -> dict:
    """What vLLM 0.11.0 resolves when the request carries only max_tokens.

    Rule (vllm/config/model.py get_diff_sampling_param + entrypoints/openai/protocol.py
    ChatCompletionRequest.to_sampling_params, tag v0.11.0): with --generation-config auto (the
    default), repetition_penalty / temperature / top_k / top_p / min_p / max_new_tokens are read
    from the checkpoint's generation_config.json when present; anything absent takes vLLM's
    neutral default (1.0 / 1.0 / 0 / 1.0 / 0.0). `do_sample` is never consulted, so a config with
    do_sample=false but no temperature is served at temperature 1.0. The request's max_tokens
    overrides max_new_tokens. stop_token_ids default to the generation config's eos_token_id list
    (serving_chat.py). Recorded as *derived*: no server log exists to confirm it."""
    out = {"rule": "vllm-0.11.0 generation_config=auto; do_sample ignored; request max_tokens wins",
           "max_tokens": KIT_MAX_TOKENS[benchmark], "n": 1}
    if not gen_cfg:
        out["generation_config"] = "missing"
        return out
    neutral = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    for key, default in neutral.items():
        out[key] = gen_cfg[key] if gen_cfg.get(key) is not None else default
    out["from_generation_config"] = sorted(k for k in neutral if gen_cfg.get(k) is not None)
    if "do_sample" in gen_cfg:
        out["do_sample_in_config_ignored_by_vllm"] = gen_cfg["do_sample"]
    if gen_cfg.get("max_new_tokens") is not None:
        out["saved_max_new_tokens_overridden_by_request"] = gen_cfg["max_new_tokens"]
    eos = gen_cfg.get("eos_token_id")
    out["stop_token_ids"] = eos if isinstance(eos, list) else [eos] if eos is not None else []
    out["greedy"] = out["temperature"] == 0
    return out


def main(argv):
    root, out_dir = Path(argv[1]), Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = json.loads((root / "eval_matrix_1k/policies.json").read_text())
    templates = {p.name: sha_file(p) for p in (root / "rescore10/eval/templates").glob("*.jinja")}
    kit = {p.name: sha_file(p) for t in ("gsm8k", "aime2025") for p in (root / "rescore10/eval/tasks" / t).glob("*.py")}
    rows, summary = [], Counter()

    for sub in ("eval_matrix_1k/results", "eval_matrix_1k/locked_test/results"):
        for path in sorted((root / sub).glob("*.json")):
            data = json.loads(path.read_text())
            m = data["eval_matrix_1k"]
            benchmark = m["benchmark"]
            errors, runs = validate_matrix(data, benchmark)
            server = m.get("server") or {}
            resolved = {k: literal(v) for k, v in (server.get("resolved_sampling_params") or {}).items()}
            if not resolved:
                errors.append("no_resolved_sampling_params")
            if (m.get("completeness") or {}).get("inspect_status") != "success":
                errors.append("inspect_status_not_success")
            if not (m.get("per_run") and all(r.get("complete") for r in m["per_run"])):
                errors.append("runs_not_all_complete")
            max_model_len = None
            log = server.get("cli_args_logged", "")
            if isinstance(log, str) and "{" in log:
                try:
                    max_model_len = ast.literal_eval(log[log.index("{"):]).get("max_model_len")
                except (ValueError, SyntaxError):
                    pass
            pre = m.get("preflight") or {}
            fam = pre.get("family")
            z_rel = f"{sub.replace('/results', '/trajectories')}/{m['exp_id']}.json.gz"
            row = {
                "example_id": f"{m['checkpoint_id']}@{m['policy_id']}",
                "checkpoint_id": m["checkpoint_id"], "serving_id": m["policy_id"],
                "session_id": m["checkpoint_id"].rsplit("-exp", 1)[0],
                "benchmark": benchmark, "base_model": BASE_BY_FAMILY.get(fam),
                "track": "matrix", "matrix_split": m.get("split"), "matrix_phase": m.get("phase"),
                "status": "valid" if not errors else "invalid", "errors": errors,
                "y_mean": statistics.mean(runs) if runs else None,
                "runs": runs, "run_sd": statistics.pstdev(runs) if runs else None,
                "S": {
                    "s_resolution": "server_verified",
                    "sampling": resolved, "policy": policies.get(m["policy_id"]),
                    "request_template": (data.get("request_template") or {}).get("template"),
                    "dtype": server.get("dtype"), "generation_config_mode": server.get("generation_config"),
                    "inherited_extra_stop_ids": server.get("inherited_extra_stop_ids"),
                    "max_model_len": max_model_len,
                    "chat_template": {"name": pre.get("chat_template"), "sha256": pre.get("chat_template_sha256")},
                    "runtime": m.get("runtime"), "digests": m.get("digests"),
                    "seed_formula": (m.get("seeds") or {}).get("formula"),
                    "evaluator_deviations": m.get("evaluator_deviations"), "kit_patch": m.get("kit_patch"),
                    "scorer_patch": data.get("scorer_patch"), "site": m.get("site"), "evaluated_at": m.get("evaluated_at"),
                },
                "weights": {"weights_sha256": pre.get("weights_sha256"), "file_sha256": pre.get("file_sha256"),
                            "shard_sha256": pre.get("shard_sha256"), "archive_uri": pre.get("archive_uri"),
                            "architectures": pre.get("architectures"), "saved_dtype": pre.get("saved_dtype"),
                            "inherited_generation_config": pre.get("inherited_generation_config")},
                "aggregates": {"finish_reasons": m.get("finish_reasons"), "tokens": m.get("tokens"),
                               "n_sample_errors": data.get("n_sample_errors")},
                "z_log": {"path": z_rel, "present": (root / z_rel).exists()},
                "source_file": str(path.relative_to(root)),
            }
            rows.append(row)
            summary[("matrix", row["status"])] += 1

    for path in sorted((root / "rescore10/results").glob("*.json")):
        data = json.loads(path.read_text())
        meta = data.get("rescore10") or {}
        cid = meta.get("id", path.stem)
        benchmark = meta.get("benchmark")
        errors, runs = validate_matrix(data, benchmark) if benchmark in N_QUESTIONS else (["unknown_benchmark"], None)
        meta_dir = root / "checkpoints_meta" / cid
        cfg = json.loads((meta_dir / "config.json").read_text()) if (meta_dir / "config.json").exists() else None
        gen = json.loads((meta_dir / "generation_config.json").read_text()) if (meta_dir / "generation_config.json").exists() else None
        fam = family_of(cfg)
        template = TEMPLATE_BY_FAMILY.get(fam)
        if cfg is None:
            errors.append("checkpoint_config_missing")
        if gen is None:
            errors.append("generation_config_missing")
        if "n_sample_errors" not in data:
            errors.append("sample_errors_unknown_until_log_checked")
        explicit = data.get("request_policy")
        z_rel = f"rescore10/trajectories/{cid}.json.gz"
        row = {
            "example_id": f"{cid}@native",
            "checkpoint_id": cid, "serving_id": "native",
            "session_id": cid.rsplit("-exp", 1)[0] if "-exp" in cid else cid.rsplit("-s", 1)[0],
            "benchmark": benchmark, "base_model": BASE_BY_FAMILY.get(fam),
            "track": "native", "matrix_split": None, "matrix_phase": None,
            "status": "valid" if not [e for e in errors if not e.startswith("sample_errors_unknown")] else "invalid",
            "errors": errors,
            "y_mean": statistics.mean(runs) if runs else None,
            "runs": runs, "run_sd": statistics.pstdev(runs) if runs else None,
            "S": {
                "s_resolution": "request_explicit_server_unverified" if explicit else "derived_unverified",
                "request": explicit or {"max_tokens": KIT_MAX_TOKENS.get(benchmark), "epochs": 10,
                                        "note": "evaluate_epochs.py passes only max_tokens/epochs/max_connections"},
                "request_template": (data.get("request_template") or {}).get("template"),
                "protocol": data.get("protocol"),
                "generation_config_json": gen,
                "derived_sampling": derive_native_sampling(gen, benchmark) if benchmark in N_QUESTIONS else None,
                "chat_template": {"name": template, "sha256": templates.get(template)},
                "kit_sha256": {k: v for k, v in kit.items() if k.startswith(("evaluate", "score", "task"))},
                "runtime": meta.get("eval_env"), "kit_env_reference": meta.get("kit_env_reference"),
                "kit_patch": meta.get("kit_patch"), "scorer_patch": data.get("scorer_patch"),
                "site": meta.get("site"), "evaluated_at": meta.get("evaluated_at"),
            },
            "weights": {"weights_sha256": None, "architectures": (cfg or {}).get("architectures"),
                        "saved_dtype": (cfg or {}).get("torch_dtype") or (cfg or {}).get("dtype"),
                        "hf_repo": meta.get("repo")},
            "aggregates": {"n_sample_errors": data.get("n_sample_errors"), "inspect_metrics": data.get("inspect_metrics")},
            "z_log": {"path": z_rel, "present": (root / z_rel).exists()},
            "source_file": str(path.relative_to(root)),
        }
        rows.append(row)
        summary[("native", row["status"])] += 1

    with (out_dir / "labels.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    errs = Counter(e.split(":")[0] for r in rows for e in r["errors"])
    out = {"rows": len(rows), "by_track_status": {f"{k[0]}/{k[1]}": v for k, v in sorted(summary.items())},
           "error_kinds": dict(errs.most_common()),
           "checkpoints": len({r["checkpoint_id"] for r in rows}),
           "sessions": len({r["session_id"] for r in rows}),
           "z_logs_present": sum(1 for r in rows if r["z_log"]["present"])}
    (out_dir / "labels_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv)
